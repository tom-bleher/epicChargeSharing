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
#include "G4SystemOfUnits.hh"
#include "G4THitsMap.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"

#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
const std::vector<G4double> kEmptyDoubleVector;
const std::vector<G4int> kEmptyIntVector;
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
}

void EventAction::BeginOfEventAction(const G4Event* /*event*/)
{
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
    if (computeChargeSharing) {
        ComputeChargeSharingForEvent(hitPos, finalEdep);
    } else {
        fChargeSharing.ResetForEvent();
        fNeighborhoodChargeNew.clear();
        fNeighborhoodChargeFinal.clear();
        fRunAction->SetNeighborhoodChargeData(kEmptyDoubleVector, kEmptyDoubleVector);
        fRunAction->SetNeighborhoodChargeNewData(kEmptyDoubleVector);
        fRunAction->SetNeighborhoodChargeFinalData(kEmptyDoubleVector);
        fRunAction->SetNeighborhoodDistanceAlphaData(kEmptyDoubleVector, kEmptyDoubleVector);
        fRunAction->SetNeighborhoodPixelData(kEmptyDoubleVector, kEmptyDoubleVector, kEmptyIntVector);
    }

    fRunAction->SetEventData(finalEdep, hitPos.x(), hitPos.y(), hitPos.z());
    fRunAction->SetNearestPixelPos(nearestPixel.x(), nearestPixel.y());
    fRunAction->SetFirstContactIsPixel(firstContactIsPixel);
    fRunAction->SetGeometricIsPixel(geometricIsPixel);
    fRunAction->SetIsPixelHitCombined(isPixelHitCombined);
    fRunAction->SetPixelClassification(isPixelHitCombined, fPixelTrueDeltaX, fPixelTrueDeltaY);

    fRunAction->FillTree();
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

void EventAction::ComputeChargeSharingForEvent(const G4ThreeVector& hitPos, G4double energyDeposit)
{
    const ChargeSharingCalculator::Result& result = fChargeSharing.Compute(hitPos,
                                                                           energyDeposit,
                                                                           fIonizationEnergy,
                                                                           fAmplificationFactor,
                                                                           fD0,
                                                                           fElementaryCharge);

    fPixelIndexI = result.pixelIndexI;
    fPixelIndexJ = result.pixelIndexJ;
    fPixelTrueDeltaX = std::abs(result.nearestPixelCenter.x() - hitPos.x());
    fPixelTrueDeltaY = std::abs(result.nearestPixelCenter.y() - hitPos.y());

    EnsureNeighborhoodBuffers(result.charges.size());
    const G4int gridRadius = fChargeSharing.GetNeighborhoodRadius();
    const G4int gridDim = 2 * gridRadius + 1;
    const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
    const G4double sigmaNoise = Constants::NOISE_ELECTRON_COUNT * fElementaryCharge;

    for (std::size_t idx = 0; idx < result.charges.size(); ++idx) {
        const G4int di = static_cast<G4int>(idx) / gridDim - gridRadius;
        const G4int dj = static_cast<G4int>(idx) % gridDim - gridRadius;
        const G4int gi = fPixelIndexI + di;
        const G4int gj = fPixelIndexJ + dj;

        G4double noisyCharge = 0.0;
        G4double finalCharge = 0.0;

        if (gi >= 0 && gi < numBlocksPerSide && gj >= 0 && gj < numBlocksPerSide) {
            const G4int globalId = gi * numBlocksPerSide + gj;
            const G4double sigmaGain = fDetector->GetPixelGainSigma(globalId);
            const G4double gainFactor =
                (sigmaGain > 0.0) ? G4RandGauss::shoot(1.0, sigmaGain) : 1.0;
            noisyCharge = result.charges[idx] * gainFactor;
            const G4double additiveNoise =
                (sigmaNoise > 0.0) ? G4RandGauss::shoot(0.0, sigmaNoise) : 0.0;
            finalCharge = std::max(0.0, noisyCharge + additiveNoise);
        }

        fNeighborhoodChargeNew[idx] = noisyCharge;
        fNeighborhoodChargeFinal[idx] = finalCharge;
    }

    fRunAction->SetNeighborhoodChargeData(result.fractions, result.charges);
    fRunAction->SetNeighborhoodChargeNewData(fNeighborhoodChargeNew);
    fRunAction->SetNeighborhoodChargeFinalData(fNeighborhoodChargeFinal);
    fRunAction->SetNeighborhoodDistanceAlphaData(result.distances, result.alphas);
    fRunAction->SetNeighborhoodPixelData(result.pixelX, result.pixelY, result.pixelIds);
}

void EventAction::EnsureNeighborhoodBuffers(std::size_t targetSize)
{
    if (fNeighborhoodChargeNew.capacity() < targetSize) {
        fNeighborhoodChargeNew.reserve(targetSize);
    }
    if (fNeighborhoodChargeFinal.capacity() < targetSize) {
        fNeighborhoodChargeFinal.reserve(targetSize);
    }
    fNeighborhoodChargeNew.resize(targetSize, 0.0);
    fNeighborhoodChargeFinal.resize(targetSize, 0.0);
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
