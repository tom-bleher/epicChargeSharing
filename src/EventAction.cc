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
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include "Randomize.hh"

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetector(detector),
  fSteppingAction(nullptr),
  fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS),
  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE),
  fScorerEnergyDeposit(0.0)
{ 
}

void EventAction::BeginOfEventAction(const G4Event* event)
{
  if (fSteppingAction) {
      fSteppingAction->Reset();
  }
  
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDeltaX = 0.;
  fPixelTrueDeltaY = 0.;

  fNeighborhoodChargeFractions.clear();
  fNeighborhoodCharge.clear();
  fNeighborhoodDistance.clear();
  fNeighborhoodAlpha.clear();
  // Keep any final-charge vector in sync
  {
    std::vector<G4double> empty;
    fRunAction->SetNeighborhoodChargeNewData(empty);
    fRunAction->SetNeighborhoodChargeFinalData(empty);
  }
  fNeighborhoodPixelX.clear();
  fNeighborhoodPixelY.clear();
  fNeighborhoodPixelID.clear();

  fScorerEnergyDeposit = 0.0;
  fFirstContactPos = G4ThreeVector(0.,0.,0.);
  fHasFirstContactPos = false;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    CollectScorerData(event);
    
    // initial energy not persisted per igor.txt
    G4double finalEdep = fScorerEnergyDeposit;
    const G4ThreeVector hitPos = DetermineHitPosition();

    G4ThreeVector nearestPixel(0,0,0);
    G4bool firstContactIsPixel = false;
    G4bool geometricIsPixel = false;
    G4bool isPixelHitCombined = false;
    UpdatePixelAndHitClassification(hitPos, nearestPixel,
                                    firstContactIsPixel,
                                    geometricIsPixel,
                                    isPixelHitCombined);

    // Compute charge sharing only for non-pixel-pad hits and only if there was energy deposit
    const G4bool computeChargeSharing = (!isPixelHitCombined) && (finalEdep > 0.0);
    if (computeChargeSharing) {
        ComputeChargeSharingForEvent(hitPos);
    } else {
        // Ensure vectors are empty when not computing sharing to avoid stale data
        fNeighborhoodChargeFractions.clear();
        fNeighborhoodCharge.clear();
        fNeighborhoodDistance.clear();
        fNeighborhoodAlpha.clear();
    }

    // Always compute neighborhood pixel geometry and IDs (independent of charge sharing)
    CalcNeighborhoodPixelGeometryAndIDs(hitPos);
    fRunAction->SetEventData(finalEdep, hitPos.x(), hitPos.y(), hitPos.z());
    fRunAction->SetNearestPixelPos(nearestPixel.x(), nearestPixel.y());
    fRunAction->SetFirstContactIsPixel(firstContactIsPixel);
    fRunAction->SetGeometricIsPixel(geometricIsPixel);
    fRunAction->SetIsPixelHitCombined(isPixelHitCombined);
    fRunAction->SetPixelClassification(isPixelHitCombined, fPixelTrueDeltaX, fPixelTrueDeltaY);
    
    // Compute noisy charge per pixel using per-pixel gain sigmas and additive noise
    std::vector<G4double> neighborhoodChargeNew;
    std::vector<G4double> neighborhoodChargeFinal;
    neighborhoodChargeFinal.reserve(fNeighborhoodCharge.size());
    neighborhoodChargeNew.reserve(fNeighborhoodCharge.size());
    if (computeChargeSharing && !fNeighborhoodCharge.empty()) {
        const G4int gridRadius = fNeighborhoodRadius;
        const G4int gridDim = 2 * gridRadius + 1;
        const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
        const G4double q_e = fElementaryCharge;
        const G4double sigma1 = Constants::NOISE_ELECTRON_COUNT * q_e; // [C]

        for (G4int idx = 0; idx < (G4int)fNeighborhoodCharge.size(); ++idx) {
            // Map local grid index to global pixel indices (i,j)
            const G4int di = idx / gridDim - gridRadius;
            const G4int dj = idx % gridDim - gridRadius;
            const G4int gi = fPixelIndexI + di;
            const G4int gj = fPixelIndexJ + dj;
            G4double Qi = fNeighborhoodCharge[idx];
            if (gi < 0 || gi >= numBlocksPerSide || gj < 0 || gj >= numBlocksPerSide) {
                neighborhoodChargeNew.push_back(0.0);
                neighborhoodChargeFinal.push_back(0.0);
                continue;
            }
            const G4int globalId = gi * numBlocksPerSide + gj;
            const G4double sigma_gain = fDetector->GetPixelGainSigma(globalId);

            // Multiplicative noise: N(1, sigma_gain)
            G4double mult = 1.0;
            if (sigma_gain > 0.0) {
                mult = G4RandGauss::shoot(1.0, sigma_gain);
            }
            G4double Qnew = Qi * mult;
            if (Qnew < 0.0) Qnew = 0.0;
            neighborhoodChargeNew.push_back(Qnew);

            // Additive noise: N(0, sigma1)
            G4double add = 0.0;
            if (sigma1 > 0.0) {
                add = G4RandGauss::shoot(0.0, sigma1);
            }
            G4double Qfinal = Qnew + add;
            if (Qfinal < 0.0) Qfinal = 0.0;
            neighborhoodChargeFinal.push_back(Qfinal);
        }
    }

    // Pass both raw and final
    fRunAction->SetNeighborhoodChargeData(fNeighborhoodChargeFractions, fNeighborhoodCharge);
    fRunAction->SetNeighborhoodChargeNewData(neighborhoodChargeNew);
    fRunAction->SetNeighborhoodChargeFinalData(neighborhoodChargeFinal);
    fRunAction->SetNeighborhoodPixelData(fNeighborhoodPixelX, fNeighborhoodPixelY, fNeighborhoodPixelID);
    fRunAction->SetNeighborhoodDistanceAlphaData(fNeighborhoodDistance, fNeighborhoodAlpha);
    
    fRunAction->FillTree();
}

// Implementation of the nearest pixel calculation method
G4ThreeVector EventAction::CalcNearestPixel(const G4ThreeVector& pos)
{
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  G4ThreeVector relativePos = pos - detectorPos;
  
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  G4bool isWithinDetector = (i >= 0 && i < numBlocksPerSide && j >= 0 && j < numBlocksPerSide);
  
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  G4double pixelZ = detectorPos.z() + Constants::DETECTOR_WIDTH/2 + Constants::PIXEL_WIDTH/2;
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  if (isWithinDetector) {
    fPixelTrueDeltaX = std::abs(pixelX - pos.x());
    fPixelTrueDeltaY = std::abs(pixelY - pos.y());
  } else {
    fPixelTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    fPixelTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
  }
  
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Removed CalcNeighborhoodGridAngles: angle computation is internal to charge sharing only

G4double EventAction::CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  const G4double dx = hitX - pixelCenterX;
  const G4double dy = hitY - pixelCenterY;
  const G4double d = std::sqrt(dx*dx + dy*dy);
  
  const G4double l = (pixelWidth + pixelHeight) / 2.0;
  
  const G4double numerator = (l/2.0) * std::sqrt(2.0);
  const G4double denominator = numerator + d;
  
  G4double alpha;
  if (denominator < Constants::MIN_DENOMINATOR_VALUE) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  return alpha; // Return in radians
}

// Calc charge sharing for pixels in a neighborhood grid around the hit pixel
void EventAction::CalcNeighborhoodChargeSharing(const G4ThreeVector& hitPos)
{
  fNeighborhoodChargeFractions.clear();
  fNeighborhoodCharge.clear();
  fNeighborhoodDistance.clear();
  fNeighborhoodAlpha.clear();
  
  const G4double edepInEV = fScorerEnergyDeposit / eV; // Convert to eV using CLHEP units
  const G4double numElectrons = edepInEV / fIonizationEnergy;
  
  const G4double totalCharge = numElectrons * fAmplificationFactor;
  
  const G4double pixelSize = fDetector->GetPixelSize();
  const G4double pixelSpacing = fDetector->GetPixelSpacing();
  const G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  const G4double detSize = fDetector->GetDetSize();
  const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  
  const G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  const G4double d0 = fD0 * micrometer; // d0 in Geant4 length units (mm)
  
  const G4double hitX = hitPos.x();
  const G4double hitY = hitPos.y();
  
  const G4int gridRadius = fNeighborhoodRadius;
  const G4int gridDim = 2 * gridRadius + 1;
  const G4int totalCells = gridDim * gridDim;
  std::vector<G4double> weightGrid(totalCells, 0.0);
  std::vector<G4bool>   inBoundsGrid(totalCells, false);
  fNeighborhoodDistance.reserve(totalCells);
  fNeighborhoodAlpha.reserve(totalCells);
  
  for (G4int di = -gridRadius; di <= gridRadius; ++di) {
    for (G4int dj = -gridRadius; dj <= gridRadius; ++dj) {
      const G4int idx = (di + gridRadius) * gridDim + (dj + gridRadius);
      const G4int gridPixelI = fPixelIndexI + di;
      const G4int gridPixelJ = fPixelIndexJ + dj;
      
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide ||
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        inBoundsGrid[idx] = false;
        weightGrid[idx] = 0.0;
        fNeighborhoodDistance.push_back(std::numeric_limits<G4double>::quiet_NaN());
        fNeighborhoodAlpha.push_back(std::numeric_limits<G4double>::quiet_NaN());
        continue;
      }
      
      const G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      const G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      const G4double dx = hitX - pixelCenterX;
      const G4double dy = hitY - pixelCenterY;
      const G4double distance = std::sqrt(dx*dx + dy*dy);
      const G4double alpha = CalcPixelAlphaSubtended(hitX, hitY, pixelCenterX, pixelCenterY, pixelSize, pixelSize);
      
      G4double weight = 0.0;
      const G4double minLogArg = std::exp(Constants::MIN_LOG_VALUE);
      G4double logArg = (distance > 0.0 && d0 > 0.0) ? (distance / d0) : minLogArg;
      logArg = std::max(logArg, minLogArg);
      G4double logValue = std::log(logArg);
      logValue = std::max(logValue, Constants::MIN_LOG_VALUE);
      weight = alpha * (1.0 / logValue);
      
      inBoundsGrid[idx] = true;
      weightGrid[idx] = weight;
      fNeighborhoodDistance.push_back(distance);
      fNeighborhoodAlpha.push_back(alpha);
    }
  }
  
  G4double totalWeight = 0.0;
  for (G4int idx = 0; idx < totalCells; ++idx) {
    if (inBoundsGrid[idx]) totalWeight += weightGrid[idx];
  }
  
  fNeighborhoodChargeFractions.reserve(totalCells);
  fNeighborhoodCharge.reserve(totalCells);
  for (G4int idx = 0; idx < totalCells; ++idx) {
    if (!inBoundsGrid[idx]) {
      fNeighborhoodChargeFractions.push_back(Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL); // Invalid marker for OOB
      fNeighborhoodCharge.push_back(0.0);
      continue;
    }
    G4double fraction = (totalWeight > 0.0) ? (weightGrid[idx] / totalWeight) : 0.0;
    G4double chargeCoulombs = fraction * totalCharge * fElementaryCharge;
    fNeighborhoodChargeFractions.push_back(fraction);
    fNeighborhoodCharge.push_back(chargeCoulombs);
  }
}

// Decide on x_hit/y_hit position for the event
G4ThreeVector EventAction::DetermineHitPosition() const
{
  if (fHasFirstContactPos) {
    return fFirstContactPos;
  }
  return G4ThreeVector(0.,0.,0.);
}

// Update nearest pixel, flags, and deltas based on hit position
void EventAction::UpdatePixelAndHitClassification(const G4ThreeVector& hitPos,
                                                  G4ThreeVector& nearestPixel,
                                                  G4bool& firstContactIsPixel,
                                                  G4bool& geometricIsPixel,
                                                  G4bool& isPixelHitCombined)
{
  // First-contact classification
  firstContactIsPixel = fSteppingAction ? fSteppingAction->FirstContactIsPixel() : false;

  // Nearest pixel center
  nearestPixel = CalcNearestPixel(hitPos);

  // Geometry-based pixel check
  const G4double halfPixel = fDetector->GetPixelSize()/2.0;
  const G4double dxAbs = std::abs(hitPos.x() - nearestPixel.x());
  const G4double dyAbs = std::abs(hitPos.y() - nearestPixel.y());

  // Persist deltas for downstream consumers and ROOT output
  fPixelTrueDeltaX = dxAbs;
  fPixelTrueDeltaY = dyAbs;

  geometricIsPixel = (std::max(dxAbs, dyAbs) <= halfPixel);
  isPixelHitCombined = firstContactIsPixel || geometricIsPixel;
}

// Wrap charge sharing computation with provided hit position
void EventAction::ComputeChargeSharingForEvent(const G4ThreeVector& hitPos)
{
  // Internals use member state already set prior to this call
  CalcNeighborhoodChargeSharing(hitPos);
}

void EventAction::CollectScorerData(const G4Event* event)
{
  // Reset per–event scorer data
  fScorerEnergyDeposit = 0.0;

  // Defensive check – HC may be null for empty events
  G4HCofThisEvent* hce = event->GetHCofThisEvent();
  if (!hce) return;

  // Retrieve collection IDs once (static cache so cost is negligible)
  static G4int edepID  = -1;
  if (edepID < 0)
  {
      G4SDManager* sdm = G4SDManager::GetSDMpointer();
      if(sdm) {
          edepID = sdm->GetCollectionID("SiliconDetector/EnergyDeposit");
      }
  }

  // Energy deposit map
  if (edepID >= 0)
  {
    auto* edepMap = dynamic_cast<G4THitsMap<G4double>*>(hce->GetHC(edepID));
    if (edepMap && edepMap->GetMap()->size() > 0)
    {
      for (const auto& kv : *edepMap->GetMap())
      {
        fScorerEnergyDeposit += *(kv.second);
      }
    }
  }
}

// Compute neighborhood pixel centers (X,Y) and global grid IDs around current pixel index
void EventAction::CalcNeighborhoodPixelGeometryAndIDs(const G4ThreeVector& /*hitPos*/)
{
  fNeighborhoodPixelX.clear();
  fNeighborhoodPixelY.clear();
  fNeighborhoodPixelID.clear();

  const G4double pixelSize = fDetector->GetPixelSize();
  const G4double pixelSpacing = fDetector->GetPixelSpacing();
  const G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  const G4double detSize = fDetector->GetDetSize();
  const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();

  const G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;

  const G4int gridRadius = fNeighborhoodRadius;
  const G4int gridDim = 2 * gridRadius + 1;
  const G4int totalCells = gridDim * gridDim;
  fNeighborhoodPixelX.reserve(totalCells);
  fNeighborhoodPixelY.reserve(totalCells);
  fNeighborhoodPixelID.reserve(totalCells);

  for (G4int di = -gridRadius; di <= gridRadius; ++di) {
    for (G4int dj = -gridRadius; dj <= gridRadius; ++dj) {
      const G4int gridPixelI = fPixelIndexI + di;
      const G4int gridPixelJ = fPixelIndexJ + dj;

      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide ||
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        fNeighborhoodPixelX.push_back(std::numeric_limits<G4double>::quiet_NaN());
        fNeighborhoodPixelY.push_back(std::numeric_limits<G4double>::quiet_NaN());
        fNeighborhoodPixelID.push_back(-1);
        continue;
      }

      const G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      const G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      const G4int globalId = gridPixelI * numBlocksPerSide + gridPixelJ; // row-major

      fNeighborhoodPixelX.push_back(pixelCenterX);
      fNeighborhoodPixelY.push_back(pixelCenterY);
      fNeighborhoodPixelID.push_back(globalId);
    }
  }
}

