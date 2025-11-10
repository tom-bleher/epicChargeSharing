/**
 * @file EventAction.cc
 * @brief Per-event bookkeeping: first contact, pixel classification, charge sharing, and scorer readout.
 */
#include "EventAction.hh"

#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "SteppingAction.hh"
#include "Constants.hh"

#include "G4Event.hh"
#include "G4GenericMessenger.hh"
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
    : G4UserEventAction(),
      fRunAction(runAction),
      fDetector(detector),
      fSteppingAction(nullptr),
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS),
      fFirstContactPos(0., 0., 0.),
      fPixelIndexI(-1),
      fPixelIndexJ(-1),
      fPixelTrueDeltaX(0.),
      fPixelTrueDeltaY(0.),
      fIonizationEnergy(Constants::IONIZATION_ENERGY),
      fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
      fD0(Constants::D0_CHARGE_SHARING),
      fElementaryCharge(Constants::ELEMENTARY_CHARGE),
      fScorerEnergyDeposit(0.0),
      fChargeSharing(detector),
      fNearestPixelGlobalId(-1),
      fNeighborhoodLayout(detector ? detector->GetNeighborhoodRadius()
                                   : Constants::NEIGHBORHOOD_RADIUS)
{
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
    fHasFirstContactPos = false;
    fNearestPixelGlobalId = -1;

    fChargeSharing.ResetForEvent();
    EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    CollectScorerData(event);

    const G4double finalEdep = fScorerEnergyDeposit;
    const G4ThreeVector hitPos = DetermineHitPosition();

    G4ThreeVector nearestPixel(0., 0., 0.);
    G4bool firstContactIsPixel = false;
    G4bool geometricIsPixel = false;
    G4bool isPixelHitCombined = false;
    UpdatePixelAndHitClassification(hitPos,
                                    nearestPixel,
                                    firstContactIsPixel,
                                    geometricIsPixel,
                                    isPixelHitCombined);

    const G4bool computeChargeSharing = (!isPixelHitCombined) && (finalEdep > 0.0);
    const ChargeSharingCalculator::Result* chargeResult = nullptr;
    if (computeChargeSharing) {
        chargeResult = &ComputeChargeSharingForEvent(hitPos, finalEdep);
    } else {
        fChargeSharing.ResetForEvent();
        EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
    }

    RunAction::EventSummaryData summary{};
    summary.edep = finalEdep;
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

    RunAction::EventRecord record{};
    record.summary = summary;
    record.nearestPixelI = fPixelIndexI;
    record.nearestPixelJ = fPixelIndexJ;
    record.nearestPixelGlobalId = fNearestPixelGlobalId;

    if (chargeResult) {
        record.totalGridCells = static_cast<G4int>(chargeResult->totalCells);
        record.neighborCells = std::span<const ChargeSharingCalculator::Result::NeighborCell>(
            chargeResult->cells.data(), chargeResult->cells.size());
        record.neighborChargesNew = std::span<const G4double>(fNeighborhoodChargeNew.data(),
                                                              fNeighborhoodChargeNew.size());
        record.neighborChargesFinal = std::span<const G4double>(fNeighborhoodChargeFinal.data(),
                                                                fNeighborhoodChargeFinal.size());
        record.includeDistanceAlpha = fEmitDistanceAlphaOutputs;
        record.fullGridSide = chargeResult->fullGridSide;
        record.fullFractions = std::span<const G4double>(chargeResult->fullFractions.data(),
                                                         chargeResult->fullFractions.size());
        record.fullPixelIds = std::span<const G4int>(chargeResult->fullPixelIds.data(),
                                                     chargeResult->fullPixelIds.size());
        record.fullPixelX = std::span<const G4double>(chargeResult->fullPixelX.data(),
                                                      chargeResult->fullPixelX.size());
        record.fullPixelY = std::span<const G4double>(chargeResult->fullPixelY.data(),
                                                      chargeResult->fullPixelY.size());
    } else {
        record.totalGridCells = static_cast<G4int>(fNeighborhoodLayout.TotalCells());
        record.neighborChargesNew = std::span<const G4double>(fNeighborhoodChargeNew.data(),
                                                              fNeighborhoodChargeNew.size());
        record.neighborChargesFinal = std::span<const G4double>(fNeighborhoodChargeFinal.data(),
                                                                fNeighborhoodChargeFinal.size());
        record.includeDistanceAlpha = false;
        record.fullGridSide = fDetector ? fDetector->GetNumBlocksPerSide() : 0;
    }

    fRunAction->FillTree(record);
}

G4ThreeVector EventAction::DetermineHitPosition() const
{
    if (fHasFirstContactPos) {
        return fFirstContactPos;
    }
    return G4ThreeVector(0., 0., 0.);
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
                                                                           fAmplificationFactor,
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
        fNearestPixelGlobalId = fPixelIndexI * numBlocks + fPixelIndexJ;
    } else {
        fNearestPixelGlobalId = -1;
    }
}

EventAction::NeighborContext EventAction::MakeNeighborContext() const
{
    NeighborContext context{};
    context.sigmaNoise = Constants::NOISE_ELECTRON_COUNT * fElementaryCharge;
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

        G4double noisyCharge = cell.charge;
        if (hasGainNoise) {
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

        fNeighborhoodChargeNew[gridIndex] = noisyCharge;
        fNeighborhoodChargeFinal[gridIndex] = finalCharge;
    }
}

void EventAction::CollectScorerData(const G4Event* event)
{
    fScorerEnergyDeposit = 0.0;

    G4HCofThisEvent* hce = event->GetHCofThisEvent();
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
