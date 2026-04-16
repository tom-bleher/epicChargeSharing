// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/**
 * @file EventAction.cc
 * @brief Per-event bookkeeping: first contact, pixel classification, charge sharing, and scorer readout.
 */
#include "EventAction.hh"

#include "Config.hh"
#include "DetectorConstruction.hh"
#include "RunAction.hh"
#include "RuntimeConfig.hh"
#include "SteppingAction.hh"

#include "G4Event.hh"
#include "G4GenericMessenger.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"

#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace {
const std::vector<G4double> kEmptyDoubleVector;
}

void EventAction::SetEmitDistanceAlpha(G4bool enabled) {
    fOutputDistanceAlpha = enabled;
    fChargeSharing.SetEmitDistanceAlpha(enabled);
    if (fRunAction) {
        fRunAction->SetChargeSharingDistanceAlphaMeta(enabled);
    }
}

void EventAction::SetComputeFullFractions(G4bool enabled) {
    fStoreFullGridFractions = enabled;
    fChargeSharing.SetComputeFullGridFractions(enabled);
    if (fRunAction) {
        fRunAction->ConfigureFullFractionBranch(enabled);
    }
}

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
    : fRunAction(runAction), fDetector(detector),

      fFirstContactPos(0., 0., 0.),

      fChargeSharing(detector),

      fNeighborhoodLayout(detector ? detector->GetNeighborhoodRadius() : Constants::NEIGHBORHOOD_RADIUS),
      fOutputDistanceAlpha(true), fStoreFullGridFractions(ECS::RuntimeConfig::Instance().storeFullGrid) {
    // Override physics parameters from runtime config
    const auto& rtConfig = ECS::RuntimeConfig::Instance();
    fIonizationEnergy = rtConfig.ionizationEnergy;
    fGain = rtConfig.gain;
    fD0 = rtConfig.d0;

    if (detector) {
        fChargeSharing.SetNeighborhoodRadius(detector->GetNeighborhoodRadius());
    }
    fChargeSharing.SetEmitDistanceAlpha(fOutputDistanceAlpha);
    fChargeSharing.SetComputeFullGridFractions(fStoreFullGridFractions);
    if (fRunAction) {
        fRunAction->SetChargeSharingDistanceAlphaMeta(fOutputDistanceAlpha);
        fRunAction->ConfigureFullFractionBranch(fStoreFullGridFractions);
    }

    fMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/chargeSharing/", "Charge sharing configuration");
    auto& cmd = fMessenger->DeclareProperty("computeDistanceAlpha", fOutputDistanceAlpha,
                                            "Enable per-neighbor distance and alpha outputs");
    cmd.SetStates(G4State_PreInit, G4State_Idle);
    cmd.SetToBeBroadcasted(true);
    auto& fullCmd = fMessenger->DeclareProperty("computeFullFractions", fStoreFullGridFractions,
                                                "Enable per-event full-detector charge fractions (F_i)");
    fullCmd.SetStates(G4State_PreInit, G4State_Idle);
    fullCmd.SetToBeBroadcasted(true);

}

void EventAction::BeginOfEventAction(const G4Event* /*event*/) {
    if (fSteppingAction) {
        fSteppingAction->Reset();
    }

    fPixelRowIndex = -1;
    fPixelColIndex = -1;
    fPixelTrueDeltaX = 0.;
    fPixelTrueDeltaY = 0.;
    fScorerEnergyDeposit = 0.0;
    fFirstContactPos = G4ThreeVector(0., 0., 0.);
    fFirstContactTime = 0.0;
    fFirstContactPosValid = false;
    fNearestPixelGlobalId = -1;
    fHitWithinDetector = false;

    fChargeSharing.ResetForEvent();
    EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
}

void EventAction::EndOfEventAction(const G4Event* event) {
    // Read energy deposit from SteppingAction (replaces G4PSEnergyDeposit scorer)
    fScorerEnergyDeposit = fSteppingAction ? fSteppingAction->GetTotalEdep() : 0.0;

    const G4double finalEdep = fScorerEnergyDeposit;
    // Use first-contact position when available; fall back to primary vertex
    // so that TrueX/TrueY always report where the beam aimed, even for misses.
    G4ThreeVector hitPos = DetermineHitPosition();
    if (!fFirstContactPosValid && event && event->GetPrimaryVertex()) {
        hitPos = event->GetPrimaryVertex()->GetPosition();
    }

    G4ThreeVector nearestPixel(0., 0., 0.);
    G4bool firstContactIsPixel = false;
    G4bool geometricIsPixel = false;
    G4bool isPixelHitCombined = false;
    UpdatePixelAndHitClassification(hitPos, nearestPixel, firstContactIsPixel, geometricIsPixel, isPixelHitCombined);

    // Skip charge sharing for edge pixels whose neighborhood extends beyond the detector.
    // These produce incomplete neighborhoods with biased fractions.
    G4bool isEdgePixel = false;
    if (fDetector) {
        const G4int radius = fDetector->GetNeighborhoodRadius();
        const G4int numBlocks = fDetector->GetNumBlocksPerSide();
        const G4int minI = fDetector->GetMinIndexX();
        const G4int minJ = fDetector->GetMinIndexY();
        const G4int maxI = minI + numBlocks - 1;
        const G4int maxJ = minJ + numBlocks - 1;
        isEdgePixel = (fPixelRowIndex < minI + radius || fPixelRowIndex > maxI - radius ||
                       fPixelColIndex < minJ + radius || fPixelColIndex > maxJ - radius);
    }

    // Compute charge sharing if applicable
    const G4bool computeChargeSharing = (!isPixelHitCombined) && (finalEdep > 0.0) && !isEdgePixel;
    const G4double eventGain = computeChargeSharing ? SampleEventGain(finalEdep) : fGain;
    const ChargeSharingCalculator::Result* chargeResult = nullptr;
    if (computeChargeSharing) {
        chargeResult = &ComputeChargeSharingForEvent(hitPos, finalEdep, eventGain);
    } else {
        fChargeSharing.ResetForEvent();
        EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
    }

    // Extract primary particle momentum from the G4Event
    G4ThreeVector primaryMomentum(0., 0., 0.);
    if (event && event->GetPrimaryVertex()) {
        const auto* primary = event->GetPrimaryVertex()->GetPrimary();
        if (primary) {
            primaryMomentum = primary->GetMomentum();
        }
    }

    // Get path length accumulated by SteppingAction
    const G4double pathLength = fSteppingAction ? fSteppingAction->GetPathLength() : 0.0;

    // Populate per-step energy deposit vectors for Landau fluctuation output
    fStepEdeps.clear();
    fStepX.clear();
    fStepY.clear();
    fStepZ.clear();
    fStepTimes.clear();
    if (fSteppingAction) {
        for (const auto& s : fSteppingAction->GetStepDeposits()) {
            fStepEdeps.push_back(s.edep);
            fStepX.push_back(s.position.x());
            fStepY.push_back(s.position.y());
            fStepZ.push_back(s.position.z());
            fStepTimes.push_back(s.time);
        }
    }

    // Build event record
    ECS::IO::EventRecord record{};
    record.summary =
        BuildEventSummary(finalEdep, hitPos, nearestPixel, firstContactIsPixel, geometricIsPixel, isPixelHitCombined);
    record.summary.primaryMomentumX = primaryMomentum.x();
    record.summary.primaryMomentumY = primaryMomentum.y();
    record.summary.primaryMomentumZ = primaryMomentum.z();
    record.summary.hitTime = fFirstContactTime;
    record.summary.pathLength = pathLength;
    record.summary.eventGain = eventGain;
    record.summary.nSteps = static_cast<G4int>(fStepEdeps.size());
    record.stepEdep = std::span<const G4double>(fStepEdeps.data(), fStepEdeps.size());
    record.stepX = std::span<const G4double>(fStepX.data(), fStepX.size());
    record.stepY = std::span<const G4double>(fStepY.data(), fStepY.size());
    record.stepZ = std::span<const G4double>(fStepZ.data(), fStepZ.size());
    record.stepTime = std::span<const G4double>(fStepTimes.data(), fStepTimes.size());
    record.nearestPixelI = fPixelRowIndex;
    record.nearestPixelJ = fPixelColIndex;
    record.nearestPixelGlobalId = fNearestPixelGlobalId;

    if (chargeResult) {
        PopulateRecordFromChargeResult(record, *chargeResult);
    } else {
        // No charge sharing computed - use default values
        record.totalGridCells = static_cast<G4int>(fNeighborhoodLayout.TotalCells());
        record.neighborChargesAmp =
            std::span<const G4double>(fNeighborhoodChargeAmp.data(), fNeighborhoodChargeAmp.size());
        record.neighborChargesMeas =
            std::span<const G4double>(fNeighborhoodChargeMeas.data(), fNeighborhoodChargeMeas.size());
        // Row/Col/Block mode noisy charges (default case)
        record.neighborChargesAmpRow =
            std::span<const G4double>(fNeighborhoodChargeAmpRow.data(), fNeighborhoodChargeAmpRow.size());
        record.neighborChargesMeasRow =
            std::span<const G4double>(fNeighborhoodChargeMeasRow.data(), fNeighborhoodChargeMeasRow.size());
        record.neighborChargesAmpCol =
            std::span<const G4double>(fNeighborhoodChargeAmpCol.data(), fNeighborhoodChargeAmpCol.size());
        record.neighborChargesMeasCol =
            std::span<const G4double>(fNeighborhoodChargeMeasCol.data(), fNeighborhoodChargeMeasCol.size());
        record.neighborChargesAmpBlock =
            std::span<const G4double>(fNeighborhoodChargeAmpBlock.data(), fNeighborhoodChargeAmpBlock.size());
        record.neighborChargesMeasBlock =
            std::span<const G4double>(fNeighborhoodChargeMeasBlock.data(), fNeighborhoodChargeMeasBlock.size());
        record.includeDistanceAlpha = false;
        record.mode = ChargeSharingCalculator::ChargeMode::Neighborhood;
        record.geometry = BuildDefaultGridGeometry();
        record.fullGridRows = record.geometry.nRows;
        record.fullGridCols = record.geometry.nCols;
    }

    // Get event and run IDs for EDM4hep output
    const std::uint64_t eventId = event ? static_cast<std::uint64_t>(event->GetEventID()) : 0;
    const G4int runId =
        G4RunManager::GetRunManager()->GetCurrentRun() ? G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID() : 0;

    fRunAction->FillTree(record, eventId, runId);
}

const G4ThreeVector& EventAction::DetermineHitPosition() const {
    if (fFirstContactPosValid) {
        return fFirstContactPos;
    }
    static const G4ThreeVector zero(0., 0., 0.);
    return zero;
}

G4ThreeVector EventAction::CalcNearestPixel(const G4ThreeVector& pos) {
    const auto location = fDetector->FindNearestPixel(pos);

    fPixelRowIndex = location.indexI;
    fPixelColIndex = location.indexJ;
    fHitWithinDetector = location.withinDetector;

    return location.center;
}

void EventAction::UpdatePixelAndHitClassification(const G4ThreeVector& hitPos, G4ThreeVector& nearestPixel,
                                                  G4bool& firstContactIsPixel, G4bool& geometricIsPixel,
                                                  G4bool& isPixelHitCombined) {
    firstContactIsPixel = fSteppingAction ? fSteppingAction->FirstContactIsPixel() : false;

    nearestPixel = CalcNearestPixel(hitPos);

    const G4double halfPixel = fDetector->GetPixelSize() / 2.0;
    const G4double dxAbs = std::abs(hitPos.x() - nearestPixel.x());
    const G4double dyAbs = std::abs(hitPos.y() - nearestPixel.y());

    fPixelTrueDeltaX = dxAbs;
    fPixelTrueDeltaY = dyAbs;

    geometricIsPixel = (std::max(dxAbs, dyAbs) <= halfPixel);
    isPixelHitCombined = firstContactIsPixel || geometricIsPixel;
}

const ChargeSharingCalculator::Result& EventAction::ComputeChargeSharingForEvent(const G4ThreeVector& hitPos,
                                                                                 G4double energyDeposit,
                                                                                 G4double eventGain) {
    const auto& rtConfig = ECS::RuntimeConfig::Instance();
    const ChargeSharingCalculator::Result* resultPtr = nullptr;

    // Per-step charge sharing: compute fractions from each step's position independently
    if (rtConfig.perStepChargeSharing && fSteppingAction) {
        const auto& deposits = fSteppingAction->GetStepDeposits();
        if (deposits.size() > 1) {
            std::vector<ChargeSharingCalculator::StepInput> inputs;
            inputs.reserve(deposits.size());
            for (const auto& d : deposits) {
                inputs.push_back({d.position, d.edep});
            }
            resultPtr = &fChargeSharing.ComputeFromSteps(inputs, energyDeposit, fIonizationEnergy, eventGain, fD0,
                                                         fElementaryCharge);
        }
    }

    // Fallback: legacy single-point computation
    if (!resultPtr) {
        resultPtr = &fChargeSharing.Compute(hitPos, energyDeposit, fIonizationEnergy, eventGain, fD0,
                                            fElementaryCharge);
    }

    const auto& result = *resultPtr;

    // Use centroid (from result.hit) for pixel indexing when per-step was used
    const G4ThreeVector effectiveHitPos(result.hit.trueX, result.hit.trueY, result.hit.trueZ);
    UpdatePixelIndices(result, effectiveHitPos);
    fNeighborhoodLayout.SetRadius(result.gridRadius);
    EnsureNeighborhoodBuffers(result.totalCells);

    const NeighborContext context = MakeNeighborContext();
    PopulateNeighborCharges(result, context);
    return result;
}

G4double EventAction::SampleEventGain(G4double energyDeposit) const {
    const auto& rtConfig = ECS::RuntimeConfig::Instance();
    if (!rtConfig.gainFluctuationEnabled) {
        return fGain;
    }

    const G4double nPrimary = (energyDeposit / CLHEP::eV) / fIonizationEnergy;

    // Gain saturation: large deposits create space charge that screens the gain field
    const G4double satFactor = 1.0 / (1.0 + nPrimary / rtConfig.gainSaturationCharge);
    const G4double effectiveGain = fGain * satFactor;

    // Stochastic amplification: McIntyre excess noise factor
    const G4double sigmaRel = std::sqrt(rtConfig.gainExcessNoiseFactor / std::max(1.0, nPrimary));
    const G4double sampledGain = G4RandGauss::shoot(effectiveGain, effectiveGain * sigmaRel);

    return std::max(1.0, sampledGain);
}

void EventAction::EnsureNeighborhoodBuffers(std::size_t totalCells) {
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    neighbor::ResizeAndFill(fNeighborhoodChargeAmp, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeMeas, totalCells, nan);
    // Row/Col/Block mode noisy charges
    neighbor::ResizeAndFill(fNeighborhoodChargeAmpRow, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeMeasRow, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeAmpCol, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeMeasCol, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeAmpBlock, totalCells, nan);
    neighbor::ResizeAndFill(fNeighborhoodChargeMeasBlock, totalCells, nan);
}

void EventAction::UpdatePixelIndices(const ChargeSharingCalculator::Result& result, const G4ThreeVector& hitPos) {
    fPixelRowIndex = result.pixelRowIndex;
    fPixelColIndex = result.pixelColIndex;
    fPixelTrueDeltaX = std::abs(result.nearestPixelCenter.x() - hitPos.x());
    fPixelTrueDeltaY = std::abs(result.nearestPixelCenter.y() - hitPos.y());
    const G4int numBlocks = fDetector ? fDetector->GetNumBlocksPerSide() : 0;
    if (numBlocks > 0 && fPixelRowIndex >= 0 && fPixelColIndex >= 0) {
        fNearestPixelGlobalId = (fPixelRowIndex * numBlocks) + fPixelColIndex;
    } else {
        fNearestPixelGlobalId = -1;
    }
}

EventAction::NeighborContext EventAction::MakeNeighborContext() const {
    NeighborContext context{};
    context.sigmaNoise = ECS::RuntimeConfig::Instance().noiseElectronCount * fElementaryCharge;
    return context;
}

void EventAction::PopulateNeighborCharges(const ChargeSharingCalculator::Result& result,
                                          const NeighborContext& context) {
    const auto targetCells = std::max<std::size_t>(1, result.totalCells);
    if (fNeighborhoodChargeAmp.size() != targetCells) {
        EnsureNeighborhoodBuffers(targetCells);
    } else {
        neighbor::Fill(fNeighborhoodChargeAmp, 0.0);
        neighbor::Fill(fNeighborhoodChargeMeas, 0.0);
        neighbor::Fill(fNeighborhoodChargeAmpRow, 0.0);
        neighbor::Fill(fNeighborhoodChargeMeasRow, 0.0);
        neighbor::Fill(fNeighborhoodChargeAmpCol, 0.0);
        neighbor::Fill(fNeighborhoodChargeMeasCol, 0.0);
        neighbor::Fill(fNeighborhoodChargeAmpBlock, 0.0);
        neighbor::Fill(fNeighborhoodChargeMeasBlock, 0.0);
    }

    const auto& gainSigmas = fDetector ? fDetector->GetPixelGainSigmas() : kEmptyDoubleVector;
    const auto gainCount = gainSigmas.size();
    const bool hasGainNoise = gainCount > 0;
    const bool hasAdditiveNoise = context.sigmaNoise > 0.0;

    for (const auto& cell : result.cells) {
        if (cell.gridIndex < 0) {
            continue;
        }
        const auto gridIndex = static_cast<std::size_t>(cell.gridIndex);
        if (gridIndex >= targetCells) {
            continue;
        }

        const G4int globalId = cell.globalPixelId;
        if (globalId < 0) {
            continue;
        }

        // Draw noise ONCE per pixel — each physical pad has one gain variation
        // and one electronic noise sample, shared across all normalization modes.
        G4double gainFactor = 1.0;
        if (hasGainNoise) {
            const auto gid = static_cast<std::size_t>(globalId);
            if (gid < gainCount) {
                const G4double sigmaGain = gainSigmas[gid];
                if (sigmaGain > 0.0) {
                    gainFactor = G4RandGauss::shoot(1.0, sigmaGain);
                }
            }
        }
        const G4double additiveNoise = hasAdditiveNoise ? G4RandGauss::shoot(0.0, context.sigmaNoise) : 0.0;

        // Apply the same noise realization to all normalization modes.
        // chargeAmplified = Qi * Gauss(1, pixelGainSigma), clamped ≥ 0 (LGAD gain step)
        // chargeReadout   = chargeAmplified + Gauss(0, encNoise), clamped ≥ 0 (ASIC digitization)
        auto applyNoise = [gainFactor, additiveNoise](G4double baseCharge) -> std::pair<G4double, G4double> {
            const G4double chargeAmplified = std::max(0.0, baseCharge * gainFactor);
            const G4double chargeReadout   = std::max(0.0, chargeAmplified + additiveNoise);
            return {chargeAmplified, chargeReadout};
        };

        // Neighborhood mode
        auto [ampCharge, rdoCharge] = applyNoise(cell.chargeInd);
        fNeighborhoodChargeAmp[gridIndex] = ampCharge;
        fNeighborhoodChargeMeas[gridIndex] = rdoCharge;

        // Row mode
        auto [ampRow, rdoRow] = applyNoise(cell.chargeIndRow);
        fNeighborhoodChargeAmpRow[gridIndex] = ampRow;
        fNeighborhoodChargeMeasRow[gridIndex] = rdoRow;

        // Col mode
        auto [ampCol, rdoCol] = applyNoise(cell.chargeIndCol);
        fNeighborhoodChargeAmpCol[gridIndex] = ampCol;
        fNeighborhoodChargeMeasCol[gridIndex] = rdoCol;

        // Block mode
        auto [ampBlock, rdoBlock] = applyNoise(cell.chargeIndBlock);
        fNeighborhoodChargeAmpBlock[gridIndex] = ampBlock;
        fNeighborhoodChargeMeasBlock[gridIndex] = rdoBlock;
    }

    // Apply readout threshold in ThresholdAboveNoise mode:
    // Zero out pads where the final charge (post-noise) falls below N×σ_noise.
    // This simulates the per-channel discriminator in real ASICs (EICROC, FCFD, ALTIROC)
    // which gates readout — only pads exceeding the threshold produce hit data.
    const G4int pixelModeInt = ECS::RuntimeConfig::Instance().activePixelMode;
    if (pixelModeInt == static_cast<G4int>(Constants::ActivePixelMode::ThresholdAboveNoise)) {
        const G4double thresholdSigma = ECS::RuntimeConfig::Instance().readoutThresholdSigma;
        const G4double threshold = thresholdSigma * context.sigmaNoise;
        for (std::size_t i = 0; i < targetCells; ++i) {
            if (fNeighborhoodChargeMeas[i] < threshold) {
                fNeighborhoodChargeMeas[i] = 0.0;
                fNeighborhoodChargeMeasRow[i] = 0.0;
                fNeighborhoodChargeMeasCol[i] = 0.0;
                fNeighborhoodChargeMeasBlock[i] = 0.0;
                // Qn (ChargeAmp) preserved: intermediate truth should not be
                // destroyed by readout threshold, matching EIC convention that
                // truth/diagnostic quantities are immutable (cf. SiliconTrackerDigi).
            }
        }
    }
}


ECS::IO::EventSummaryData EventAction::BuildEventSummary(G4double edep, const G4ThreeVector& hitPos,
                                                         const G4ThreeVector& nearestPixel, G4bool firstContactIsPixel,
                                                         G4bool geometricIsPixel, G4bool isPixelHitCombined) const {
    ECS::IO::EventSummaryData summary{};
    summary.edep = edep;
    summary.hitX = hitPos.x();
    summary.hitY = hitPos.y();
    summary.hitZ = hitPos.z();
    summary.nearestPixelX = nearestPixel.x();
    summary.nearestPixelY = nearestPixel.y();
    summary.pixelTrueDeltaX = fPixelTrueDeltaX;
    summary.pixelTrueDeltaY = fPixelTrueDeltaY;
    summary.firstContactIsPixel = firstContactIsPixel;
    summary.geometricIsPixel = geometricIsPixel;
    summary.isPixelHitCombined = isPixelHitCombined;
    summary.hitWithinDetector = fHitWithinDetector;
    return summary;
}

void EventAction::PopulateRecordFromChargeResult(ECS::IO::EventRecord& record,
                                                 const ChargeSharingCalculator::Result& result) const {
    record.totalGridCells = static_cast<G4int>(result.totalCells);
    record.neighborCells =
        std::span<const ChargeSharingCalculator::Result::NeighborCell>(result.cells.data(), result.cells.size());
    record.chargeBlock = std::span<const ChargeSharingCalculator::Result::NeighborCell>(result.chargeBlock.data(),
                                                                                        result.chargeBlock.size());
    record.neighborChargesAmp = std::span<const G4double>(fNeighborhoodChargeAmp.data(), fNeighborhoodChargeAmp.size());
    record.neighborChargesMeas =
        std::span<const G4double>(fNeighborhoodChargeMeas.data(), fNeighborhoodChargeMeas.size());
    // Row/Col/Block mode noisy charges
    record.neighborChargesAmpRow =
        std::span<const G4double>(fNeighborhoodChargeAmpRow.data(), fNeighborhoodChargeAmpRow.size());
    record.neighborChargesMeasRow =
        std::span<const G4double>(fNeighborhoodChargeMeasRow.data(), fNeighborhoodChargeMeasRow.size());
    record.neighborChargesAmpCol =
        std::span<const G4double>(fNeighborhoodChargeAmpCol.data(), fNeighborhoodChargeAmpCol.size());
    record.neighborChargesMeasCol =
        std::span<const G4double>(fNeighborhoodChargeMeasCol.data(), fNeighborhoodChargeMeasCol.size());
    record.neighborChargesAmpBlock =
        std::span<const G4double>(fNeighborhoodChargeAmpBlock.data(), fNeighborhoodChargeAmpBlock.size());
    record.neighborChargesMeasBlock =
        std::span<const G4double>(fNeighborhoodChargeMeasBlock.data(), fNeighborhoodChargeMeasBlock.size());
    record.includeDistanceAlpha = fOutputDistanceAlpha;
    record.mode = result.mode;
    record.geometry = result.geometry;
    record.hit = result.hit;
    record.neighborhoodGridBounds = result.patch.patch;

    // Populate full grid charge arrays
    const auto& full = result.full;
    record.fullGridRows = full.Rows();
    record.fullGridCols = full.Cols();

    auto makeSpan = [](const auto& grid) {
        return grid.Empty() ? std::span<const G4double>{} : std::span<const G4double>(grid.Data(), grid.Size());
    };

    // Fractions
    record.fullFi = makeSpan(full.signalFraction);
    record.fullFiRow = makeSpan(full.signalFractionRow);
    record.fullFiCol = makeSpan(full.signalFractionCol);
    record.fullFiBlock = makeSpan(full.signalFractionBlock);
    // Neighborhood-mode charges
    record.fullQ_ind = makeSpan(full.chargeInduced);
    record.fullQ_amp = makeSpan(full.chargeAmp);
    record.fullQ_meas = makeSpan(full.chargeMeas);
    // Row-mode charges
    record.fullQ_indRow = makeSpan(full.chargeInducedRow);
    record.fullQ_ampRow = makeSpan(full.chargeAmpRow);
    record.fullQ_measRow = makeSpan(full.chargeMeasRow);
    // Col-mode charges
    record.fullQ_indCol = makeSpan(full.chargeInducedCol);
    record.fullQ_ampCol = makeSpan(full.chargeAmpCol);
    record.fullQ_measCol = makeSpan(full.chargeMeasCol);
    // Block-mode charges
    record.fullQ_indBlock = makeSpan(full.chargeInducedBlock);
    record.fullQ_ampBlock = makeSpan(full.chargeAmpBlock);
    record.fullQ_measBlock = makeSpan(full.chargeMeasBlock);
    // Geometry
    record.fullDistance = makeSpan(full.distance);
    record.fullAlpha = makeSpan(full.alpha);
    record.fullPixelX = makeSpan(full.pixelX);
    record.fullPixelY = makeSpan(full.pixelY);
}

ChargeSharingCalculator::PixelGridGeometry EventAction::BuildDefaultGridGeometry() const {
    ChargeSharingCalculator::PixelGridGeometry geom{};
    if (!fDetector) {
        return geom;
    }

    const G4int numBlocks = std::max(0, fDetector->GetNumBlocksPerSide());
    const G4double spacing = fDetector->GetPixelSpacing();
    const G4ThreeVector& detPos = fDetector->GetDetectorPos();
    const G4double gridOffset = fDetector->GetGridOffset();

    // DD4hep-style grid geometry: position = index * pitch + offset
    geom.nRows = numBlocks;
    geom.nCols = numBlocks;
    geom.pitchX = spacing;
    geom.pitchY = spacing;
    geom.x0 = detPos.x() + gridOffset;
    geom.y0 = detPos.y() + gridOffset;

    return geom;
}
