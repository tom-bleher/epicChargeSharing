/**
 * @file EventAction.cc
 * @brief Per-event bookkeeping: first contact, pixel classification, charge sharing, and scorer readout.
 */
#include "EventAction.hh"

#include "Config.hh"
#include "RuntimeConfig.hh"
#include "DetectorConstruction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "G4Event.hh"
#include "G4GenericMessenger.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4StateManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4THitsMap.hh"

#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace
{
const std::vector<G4double> kEmptyDoubleVector;
}

void EventAction::SetEmitDistanceAlpha(G4bool enabled)
{
    fEmitDistanceAlphaOutputs = enabled;
    fChargeSharing.SetEmitDistanceAlpha(enabled);
    if (fRunAction) {
        fRunAction->SetChargeSharingDistanceAlphaMeta(enabled);
    }
}

void EventAction::SetComputeFullFractions(G4bool enabled)
{
    fComputeFullFractions = enabled;
    fChargeSharing.SetComputeFullGridFractions(enabled);
    if (fRunAction) {
        fRunAction->ConfigureFullFractionBranch(enabled);
    }
}

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
    :
      fRunAction(runAction),
      fDetector(detector),

      fFirstContactPos(0., 0., 0.),

      fChargeSharing(detector),

      fNeighborhoodLayout(detector ? detector->GetNeighborhoodRadius()
                                   : Constants::NEIGHBORHOOD_RADIUS),
      fEmitDistanceAlphaOutputs(true),
      fComputeFullFractions(Constants::STORE_FULL_GRID)
{
    // Override physics parameters from runtime config
    const auto& rtConfig = ECS::RuntimeConfig::Instance();
    fIonizationEnergy = rtConfig.ionizationEnergy;
    fGain = rtConfig.gain;
    fD0 = rtConfig.d0;

    if (detector) {
        fChargeSharing.SetNeighborhoodRadius(detector->GetNeighborhoodRadius());
    }
    fChargeSharing.SetEmitDistanceAlpha(fEmitDistanceAlphaOutputs);
    fChargeSharing.SetComputeFullGridFractions(fComputeFullFractions);
    if (fRunAction) {
        fRunAction->SetChargeSharingDistanceAlphaMeta(fEmitDistanceAlphaOutputs);
        fRunAction->ConfigureFullFractionBranch(fComputeFullFractions);
    }

    fMessenger = std::make_unique<G4GenericMessenger>(this,
                                                      "/epic/chargeSharing/",
                                                      "Charge sharing configuration");
    auto& cmd = fMessenger->DeclareProperty("computeDistanceAlpha",
                                            fEmitDistanceAlphaOutputs,
                                            "Enable per-neighbor distance and alpha outputs");
    cmd.SetStates(G4State_PreInit, G4State_Idle);
    cmd.SetToBeBroadcasted(true);
    auto& fullCmd = fMessenger->DeclareProperty("computeFullFractions",
                                                fComputeFullFractions,
                                                "Enable per-event full-detector charge fractions (F_i)");
    fullCmd.SetStates(G4State_PreInit, G4State_Idle);
    fullCmd.SetToBeBroadcasted(true);
}

void EventAction::BeginOfEventAction(const G4Event* /*event*/)
{
    fChargeSharing.SetEmitDistanceAlpha(fEmitDistanceAlphaOutputs);
    fChargeSharing.SetComputeFullGridFractions(fComputeFullFractions);
    if (fRunAction) {
        fRunAction->ConfigureFullFractionBranch(fComputeFullFractions);
    }

    if (fSteppingAction) {
        fSteppingAction->Reset();
    }

    fPixelIndexI = -1;
    fPixelIndexJ = -1;
    fPixelTrueDeltaX = 0.;
    fPixelTrueDeltaY = 0.;
    fScorerEnergyDeposit = 0.0;
    fFirstContactPos = G4ThreeVector(0., 0., 0.);
    fFirstContactTime = 0.0;
    fHasFirstContactPos = false;
    fNearestPixelGlobalId = -1;

    fChargeSharing.ResetForEvent();
    EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    CollectScorerData(event);

    const G4double finalEdep = fScorerEnergyDeposit;
    const G4ThreeVector& hitPos = DetermineHitPosition();

    G4ThreeVector nearestPixel(0., 0., 0.);
    G4bool firstContactIsPixel = false;
    G4bool geometricIsPixel = false;
    G4bool isPixelHitCombined = false;
    UpdatePixelAndHitClassification(hitPos,
                                    nearestPixel,
                                    firstContactIsPixel,
                                    geometricIsPixel,
                                    isPixelHitCombined);

    // Compute charge sharing if applicable
    const G4bool computeChargeSharing = (!isPixelHitCombined) && (finalEdep > 0.0);
    const ChargeSharingCalculator::Result* chargeResult = nullptr;
    if (computeChargeSharing) {
        chargeResult = &ComputeChargeSharingForEvent(hitPos, finalEdep);
        ReconstructPosition(*chargeResult, hitPos);
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

    // Build event record
    ECS::IO::EventRecord record{};
    record.summary = BuildEventSummary(finalEdep, hitPos, nearestPixel,
                                       firstContactIsPixel, geometricIsPixel, isPixelHitCombined);
    record.summary.primaryMomentumX = primaryMomentum.x();
    record.summary.primaryMomentumY = primaryMomentum.y();
    record.summary.primaryMomentumZ = primaryMomentum.z();
    record.summary.hitTime = fFirstContactTime;
    record.summary.pathLength = pathLength;
    record.nearestPixelI = fPixelIndexI;
    record.nearestPixelJ = fPixelIndexJ;
    record.nearestPixelGlobalId = fNearestPixelGlobalId;

    if (chargeResult) {
        PopulateRecordFromChargeResult(record, *chargeResult);
    } else {
        // No charge sharing computed - use default values
        record.totalGridCells = static_cast<G4int>(fNeighborhoodLayout.TotalCells());
        record.neighborChargesNew = std::span<const G4double>(fNeighborhoodChargeNew.data(),
                                                              fNeighborhoodChargeNew.size());
        record.neighborChargesFinal = std::span<const G4double>(fNeighborhoodChargeFinal.data(),
                                                                fNeighborhoodChargeFinal.size());
        // Row/Col/Block mode noisy charges (default case)
        record.neighborChargesNewRow = std::span<const G4double>(fNeighborhoodChargeNewRow.data(),
                                                                  fNeighborhoodChargeNewRow.size());
        record.neighborChargesFinalRow = std::span<const G4double>(fNeighborhoodChargeFinalRow.data(),
                                                                    fNeighborhoodChargeFinalRow.size());
        record.neighborChargesNewCol = std::span<const G4double>(fNeighborhoodChargeNewCol.data(),
                                                                  fNeighborhoodChargeNewCol.size());
        record.neighborChargesFinalCol = std::span<const G4double>(fNeighborhoodChargeFinalCol.data(),
                                                                    fNeighborhoodChargeFinalCol.size());
        record.neighborChargesNewBlock = std::span<const G4double>(fNeighborhoodChargeNewBlock.data(),
                                                                    fNeighborhoodChargeNewBlock.size());
        record.neighborChargesFinalBlock = std::span<const G4double>(fNeighborhoodChargeFinalBlock.data(),
                                                                      fNeighborhoodChargeFinalBlock.size());
        record.includeDistanceAlpha = false;
        record.mode = ChargeSharingCalculator::ChargeMode::Patch;
        record.geometry = BuildDefaultGridGeometry();
        record.fullGridRows = record.geometry.nRows;
        record.fullGridCols = record.geometry.nCols;
    }

    // Get event and run IDs for EDM4hep output
    const std::uint64_t eventId = event ? static_cast<std::uint64_t>(event->GetEventID()) : 0;
    const G4int runId = G4RunManager::GetRunManager()->GetCurrentRun()
                            ? G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID()
                            : 0;

    fRunAction->FillTree(record, eventId, runId);
}

const G4ThreeVector& EventAction::DetermineHitPosition() const
{
    if (fHasFirstContactPos) {
        return fFirstContactPos;
    }
    static const G4ThreeVector zero(0., 0., 0.);
    return zero;
}

G4ThreeVector EventAction::CalcNearestPixel(const G4ThreeVector& pos)
{
    const auto location = fDetector->FindNearestPixel(pos);

    fPixelIndexI = location.indexI;
    fPixelIndexJ = location.indexJ;

    if (location.withinDetector) {
        fPixelTrueDeltaX = std::abs(location.center.x() - pos.x());
        fPixelTrueDeltaY = std::abs(location.center.y() - pos.y());
    } else {
        fPixelTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fPixelTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }

    return location.center;
}

void EventAction::UpdatePixelAndHitClassification(const G4ThreeVector& hitPos,
                                                  G4ThreeVector& nearestPixel,
                                                  G4bool& firstContactIsPixel,
                                                  G4bool& geometricIsPixel,
                                                  G4bool& isPixelHitCombined)
{
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
                                                                                G4double energyDeposit)
{
    const ChargeSharingCalculator::Result& result = fChargeSharing.Compute(hitPos,
                                                                           energyDeposit,
                                                                           fIonizationEnergy,
                                                                           fGain,
                                                                           fD0,
                                                                           fElementaryCharge);

    UpdatePixelIndices(result, hitPos);
    fNeighborhoodLayout.SetRadius(result.gridRadius);
    EnsureNeighborhoodBuffers(result.totalCells);

    const NeighborContext context = MakeNeighborContext();
    PopulateNeighborCharges(result, context);
    return result;
}

void EventAction::EnsureNeighborhoodBuffers(std::size_t totalCells)
{
    neighbor::ResizeAndFill(fNeighborhoodChargeNew, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinal, totalCells, 0.0);
    // Row/Col/Block mode noisy charges
    neighbor::ResizeAndFill(fNeighborhoodChargeNewRow, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinalRow, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeNewCol, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinalCol, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeNewBlock, totalCells, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinalBlock, totalCells, 0.0);
}

void EventAction::UpdatePixelIndices(const ChargeSharingCalculator::Result& result,
                                     const G4ThreeVector& hitPos)
{
    fPixelIndexI = result.pixelIndexI;
    fPixelIndexJ = result.pixelIndexJ;
    fPixelTrueDeltaX = std::abs(result.nearestPixelCenter.x() - hitPos.x());
    fPixelTrueDeltaY = std::abs(result.nearestPixelCenter.y() - hitPos.y());
    const G4int numBlocks = fDetector ? fDetector->GetNumBlocksPerSide() : 0;
    if (numBlocks > 0 && fPixelIndexI >= 0 && fPixelIndexJ >= 0) {
        fNearestPixelGlobalId = (fPixelIndexI * numBlocks) + fPixelIndexJ;
    } else {
        fNearestPixelGlobalId = -1;
    }
}

EventAction::NeighborContext EventAction::MakeNeighborContext() const
{
    NeighborContext context{};
    context.sigmaNoise = ECS::RuntimeConfig::Instance().noiseElectronCount * fElementaryCharge;
    return context;
}

void EventAction::PopulateNeighborCharges(const ChargeSharingCalculator::Result& result,
                                          const NeighborContext& context)
{
    const auto targetCells = std::max<std::size_t>(1, result.totalCells);
    if (fNeighborhoodChargeNew.size() != targetCells) {
        EnsureNeighborhoodBuffers(targetCells);
    } else {
        neighbor::Fill(fNeighborhoodChargeNew, 0.0);
        neighbor::Fill(fNeighborhoodChargeFinal, 0.0);
        neighbor::Fill(fNeighborhoodChargeNewRow, 0.0);
        neighbor::Fill(fNeighborhoodChargeFinalRow, 0.0);
        neighbor::Fill(fNeighborhoodChargeNewCol, 0.0);
        neighbor::Fill(fNeighborhoodChargeFinalCol, 0.0);
        neighbor::Fill(fNeighborhoodChargeNewBlock, 0.0);
        neighbor::Fill(fNeighborhoodChargeFinalBlock, 0.0);
    }

    const auto& gainSigmas = fDetector ? fDetector->GetPixelGainSigmas() : kEmptyDoubleVector;
    const auto gainCount = gainSigmas.size();
    const bool hasGainNoise = gainCount > 0;
    const bool hasAdditiveNoise = context.sigmaNoise > 0.0;

    // Helper lambda to apply noise to a base charge
    auto applyNoise = [&](G4double baseCharge, G4int globalId) -> std::pair<G4double, G4double> {
        G4double noisyCharge = baseCharge;
        if (hasGainNoise && globalId >= 0) {
            const auto gid = static_cast<std::size_t>(globalId);
            if (gid < gainCount) {
                const G4double sigmaGain = gainSigmas[gid];
                if (sigmaGain > 0.0) {
                    noisyCharge *= G4RandGauss::shoot(1.0, sigmaGain);
                }
            }
        }
        G4double finalCharge = noisyCharge;
        if (hasAdditiveNoise) {
            finalCharge += G4RandGauss::shoot(0.0, context.sigmaNoise);
        }
        finalCharge = std::max(0.0, finalCharge);
        return {noisyCharge, finalCharge};
    };

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

        // Neighborhood mode (using cell.charge)
        auto [noisyCharge, finalCharge] = applyNoise(cell.charge, globalId);
        fNeighborhoodChargeNew[gridIndex] = noisyCharge;
        fNeighborhoodChargeFinal[gridIndex] = finalCharge;

        // Row mode (using cell.chargeRow)
        auto [noisyRow, finalRow] = applyNoise(cell.chargeRow, globalId);
        fNeighborhoodChargeNewRow[gridIndex] = noisyRow;
        fNeighborhoodChargeFinalRow[gridIndex] = finalRow;

        // Col mode (using cell.chargeCol)
        auto [noisyCol, finalCol] = applyNoise(cell.chargeCol, globalId);
        fNeighborhoodChargeNewCol[gridIndex] = noisyCol;
        fNeighborhoodChargeFinalCol[gridIndex] = finalCol;

        // Block mode (using cell.chargeBlock)
        auto [noisyBlock, finalBlock] = applyNoise(cell.chargeBlock, globalId);
        fNeighborhoodChargeNewBlock[gridIndex] = noisyBlock;
        fNeighborhoodChargeFinalBlock[gridIndex] = finalBlock;
    }
}

void EventAction::CollectScorerData(const G4Event* event)
{
    fScorerEnergyDeposit = 0.0;

    G4HCofThisEvent* const hce = event->GetHCofThisEvent();
    if (!hce) {
        return;
    }

    static G4int edepID = -1;
    if (edepID < 0) {
        if (auto* sdm = G4SDManager::GetSDMpointer()) {
            edepID = sdm->GetCollectionID("SiliconDetector/EnergyDeposit");
        }
    }

    if (edepID >= 0) {
        if (auto* edepMap = dynamic_cast<G4THitsMap<G4double>*>(hce->GetHC(edepID))) {
            for (const auto& kv : *edepMap->GetMap()) {
                fScorerEnergyDeposit += *(kv.second);
            }
        }
    }
}

void EventAction::ReconstructPosition(const ChargeSharingCalculator::Result& /*result*/,
                                      const G4ThreeVector& /*hitPos*/)
{
    // Position reconstruction is done via post-processing Gaussian fits
    // (ReconRowX/ColY or ReconX_2D/Y_2D).
}

ECS::IO::EventSummaryData EventAction::BuildEventSummary(G4double edep,
                                                            const G4ThreeVector& hitPos,
                                                            const G4ThreeVector& nearestPixel,
                                                            G4bool firstContactIsPixel,
                                                            G4bool geometricIsPixel,
                                                            G4bool isPixelHitCombined) const
{
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
    return summary;
}

void EventAction::PopulateRecordFromChargeResult(ECS::IO::EventRecord& record,
                                                  const ChargeSharingCalculator::Result& result) const
{
    record.totalGridCells = static_cast<G4int>(result.totalCells);
    record.neighborCells = std::span<const ChargeSharingCalculator::Result::NeighborCell>(
        result.cells.data(), result.cells.size());
    record.chargeBlock = std::span<const ChargeSharingCalculator::Result::NeighborCell>(
        result.chargeBlock.data(), result.chargeBlock.size());
    record.neighborChargesNew = std::span<const G4double>(fNeighborhoodChargeNew.data(),
                                                          fNeighborhoodChargeNew.size());
    record.neighborChargesFinal = std::span<const G4double>(fNeighborhoodChargeFinal.data(),
                                                            fNeighborhoodChargeFinal.size());
    // Row/Col/Block mode noisy charges
    record.neighborChargesNewRow = std::span<const G4double>(fNeighborhoodChargeNewRow.data(),
                                                              fNeighborhoodChargeNewRow.size());
    record.neighborChargesFinalRow = std::span<const G4double>(fNeighborhoodChargeFinalRow.data(),
                                                                fNeighborhoodChargeFinalRow.size());
    record.neighborChargesNewCol = std::span<const G4double>(fNeighborhoodChargeNewCol.data(),
                                                              fNeighborhoodChargeNewCol.size());
    record.neighborChargesFinalCol = std::span<const G4double>(fNeighborhoodChargeFinalCol.data(),
                                                                fNeighborhoodChargeFinalCol.size());
    record.neighborChargesNewBlock = std::span<const G4double>(fNeighborhoodChargeNewBlock.data(),
                                                                fNeighborhoodChargeNewBlock.size());
    record.neighborChargesFinalBlock = std::span<const G4double>(fNeighborhoodChargeFinalBlock.data(),
                                                                  fNeighborhoodChargeFinalBlock.size());
    record.includeDistanceAlpha = fEmitDistanceAlphaOutputs;
    record.mode = result.mode;
    record.geometry = result.geometry;
    record.hit = result.hit;
    record.patchInfo = result.patch.patch;

    // Populate full grid charge arrays
    const auto& full = result.full;
    record.fullGridRows = full.Rows();
    record.fullGridCols = full.Cols();

    auto makeSpan = [](const auto& grid) {
        return grid.Empty() ? std::span<const G4double>{}
                            : std::span<const G4double>(grid.Data(), grid.Size());
    };

    // Fractions
    record.fullFi = makeSpan(full.signalFraction);
    record.fullFiRow = makeSpan(full.signalFractionRow);
    record.fullFiCol = makeSpan(full.signalFractionCol);
    record.fullFiBlock = makeSpan(full.signalFractionBlock);
    // Neighborhood-mode charges
    record.fullQi = makeSpan(full.chargeInduced);
    record.fullQn = makeSpan(full.chargeWithNoise);
    record.fullQf = makeSpan(full.chargeFinal);
    // Row-mode charges
    record.fullQiRow = makeSpan(full.chargeInducedRow);
    record.fullQnRow = makeSpan(full.chargeWithNoiseRow);
    record.fullQfRow = makeSpan(full.chargeFinalRow);
    // Col-mode charges
    record.fullQiCol = makeSpan(full.chargeInducedCol);
    record.fullQnCol = makeSpan(full.chargeWithNoiseCol);
    record.fullQfCol = makeSpan(full.chargeFinalCol);
    // Block-mode charges
    record.fullQiBlock = makeSpan(full.chargeInducedBlock);
    record.fullQnBlock = makeSpan(full.chargeWithNoiseBlock);
    record.fullQfBlock = makeSpan(full.chargeFinalBlock);
    // Geometry
    record.fullDistance = makeSpan(full.distance);
    record.fullAlpha = makeSpan(full.alpha);
    record.fullPixelX = makeSpan(full.pixelX);
    record.fullPixelY = makeSpan(full.pixelY);
}

ChargeSharingCalculator::PixelGridGeometry EventAction::BuildDefaultGridGeometry() const
{
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
