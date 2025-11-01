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
      fChargeSharing(detector)
{
    if (detector) {
        fChargeSharing.SetNeighborhoodRadius(detector->GetNeighborhoodRadius());
    }
    fChargeSharing.SetEmitDistanceAlpha(fEmitDistanceAlphaOutputs);
    if (fRunAction) {
        fRunAction->SetChargeSharingDistanceAlphaMeta(fEmitDistanceAlphaOutputs);
    }

    fMessenger = std::make_unique<G4GenericMessenger>(this,
                                                      "/epic/chargeSharing/",
                                                      "Charge sharing configuration");
    auto& cmd = fMessenger->DeclareProperty("computeDistanceAlpha",
                                            fEmitDistanceAlphaOutputs,
                                            "Enable per-neighbor distance and alpha outputs");
    cmd.SetStates(G4State_PreInit, G4State_Idle);
    cmd.SetToBeBroadcasted(true);
}

void EventAction::BeginOfEventAction(const G4Event* /*event*/)
{
    fChargeSharing.SetEmitDistanceAlpha(fEmitDistanceAlphaOutputs);

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

    fChargeSharing.ResetForEvent();
    EnsureNeighborhoodBuffers(0);
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
        EnsureNeighborhoodBuffers(0);
        fNeighborhoodChargeNew.clear();
        fNeighborhoodChargeFinal.clear();
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

    if (chargeResult) {
        record.neighborFractions = std::span<const G4double>(chargeResult->fractions.data(),
                                                            chargeResult->fractions.size());
        record.neighborCharges = std::span<const G4double>(chargeResult->charges.data(),
                                                           chargeResult->charges.size());
        record.neighborChargesNew = std::span<const G4double>(fNeighborhoodChargeNew.data(),
                                                              fNeighborhoodChargeNew.size());
        record.neighborChargesFinal = std::span<const G4double>(fNeighborhoodChargeFinal.data(),
                                                                fNeighborhoodChargeFinal.size());
        if (fEmitDistanceAlphaOutputs) {
            record.includeDistanceAlpha = true;
            record.neighborDistances = std::span<const G4double>(chargeResult->distances.data(),
                                                                 chargeResult->distances.size());
            record.neighborAlphas = std::span<const G4double>(chargeResult->alphas.data(),
                                                              chargeResult->alphas.size());
        }
        record.neighborPixelX = std::span<const G4double>(chargeResult->pixelX.data(),
                                                          chargeResult->pixelX.size());
        record.neighborPixelY = std::span<const G4double>(chargeResult->pixelY.data(),
                                                          chargeResult->pixelY.size());
        record.neighborPixelIds = std::span<const G4int>(chargeResult->pixelIds.data(),
                                                         chargeResult->pixelIds.size());
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
    EnsureNeighborhoodBuffers(result.charges.size());

    const NeighborContext context = MakeNeighborContext();
    PopulateNeighborCharges(result, context);
    return result;
}

void EventAction::EnsureNeighborhoodBuffers(std::size_t targetSize)
{
    if (fNeighborhoodChargeNew.capacity() < targetSize) {
        fNeighborhoodChargeNew.reserve(targetSize);
    }
    if (fNeighborhoodChargeFinal.capacity() < targetSize) {
        fNeighborhoodChargeFinal.reserve(targetSize);
    }

    fNeighborhoodChargeNew.resize(targetSize);
    fNeighborhoodChargeFinal.resize(targetSize);

    std::fill(fNeighborhoodChargeNew.begin(), fNeighborhoodChargeNew.end(), 0.0);
    std::fill(fNeighborhoodChargeFinal.begin(), fNeighborhoodChargeFinal.end(), 0.0);
}

void EventAction::UpdatePixelIndices(const ChargeSharingCalculator::Result& result,
                                     const G4ThreeVector& hitPos)
{
    fPixelIndexI = result.pixelIndexI;
    fPixelIndexJ = result.pixelIndexJ;
    fPixelTrueDeltaX = std::abs(result.nearestPixelCenter.x() - hitPos.x());
    fPixelTrueDeltaY = std::abs(result.nearestPixelCenter.y() - hitPos.y());
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
    const auto totalCells = static_cast<G4int>(result.charges.size());
    const auto& pixelIds = result.pixelIds;
    const auto& gainSigmas = fDetector ? fDetector->GetPixelGainSigmas() : kEmptyDoubleVector;
    const G4int gainCount = static_cast<G4int>(gainSigmas.size());
    const bool hasGainNoise = gainCount > 0;
    const bool hasAdditiveNoise = context.sigmaNoise > 0.0;

    for (G4int idx = 0; idx < totalCells; ++idx) {
        const G4int globalId = (idx < static_cast<G4int>(pixelIds.size())) ? pixelIds[idx] : -1;

        if (globalId < 0) {
            fNeighborhoodChargeNew[idx] = 0.0;
            fNeighborhoodChargeFinal[idx] = 0.0;
            continue;
        }

        G4double noisyCharge = result.charges[idx];
        if (hasGainNoise && globalId < gainCount) {
            const G4double sigmaGain = gainSigmas[static_cast<std::size_t>(globalId)];
            if (sigmaGain > 0.0) {
                noisyCharge *= G4RandGauss::shoot(1.0, sigmaGain);
            }
        }

        G4double finalCharge = noisyCharge;
        if (hasAdditiveNoise) {
            finalCharge += G4RandGauss::shoot(0.0, context.sigmaNoise);
        }
        finalCharge = std::max(0.0, finalCharge);

        fNeighborhoodChargeNew[idx] = noisyCharge;
        fNeighborhoodChargeFinal[idx] = finalCharge;
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
