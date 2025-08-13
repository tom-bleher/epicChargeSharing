#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "SteppingAction.hh"
#include "Constants.hh"
#include "Control.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4THitsMap.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4PrimaryVertex.hh"
#include "G4RunManager.hh"
#include <exception>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdlib>

// Viewing angle α (analytical):
// α = arctan[(l/2·√2) / (l/2·√2 + d)]
// with l the pixel-pad side and d the hit–to–pad-center distance.

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetector(detector),
  fSteppingAction(nullptr),
  fNeighborhoodRadius(4), // Default to 9x9 grid (radius 4)
  

  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE),
  fScorerEnergyDeposit(0.0),
  fScorerHitCount(0),
  fScorerDataValid(false),
  fPureSiliconHit(false),
  fAluminumContaminated(false),
  fChargeCalculationEnabled(false)
{ 
}

EventAction::~EventAction()
{ 
}

void EventAction::BeginOfEventAction(const G4Event* event)
{
  // Reset all per-event variables.
  if (fSteppingAction) {
      fSteppingAction->Reset();
  }
  
  fPos = G4ThreeVector(0.,0.,0.);
  fNumPosSamples = 0;
  fHasHit = false;
  
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDeltaX = 0.;
  fPixelTrueDeltaY = 0.;
  fActualPixelDistance = -1.;
  fPixelHit = false;

  fNeighborhoodChargeFractions.clear();
  fNeighborhoodCharge.clear();

  // Reset scorer data.
  fScorerEnergyDeposit = 0.0;
  fScorerHitCount = 0;
  fScorerDataValid = false;

  // Reset hit purity data.
  fPureSiliconHit = false;
  fAluminumContaminated = false;
  fChargeCalculationEnabled = false;
  // Reset first-contact position tracking
  fFirstContactPos = G4ThreeVector(0.,0.,0.);
  fHasFirstContactPos = false;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    // ========================================
    // DATA COLLECTION AND PROCESSING
    // ========================================
    
    // Collect scorer data from MFD (prioritized)
    CollectScorerData(event);
    
    // Get primary particle info
    G4PrimaryVertex* primaryVertex = event->GetPrimaryVertex(0);
    G4PrimaryParticle* primaryParticle = primaryVertex ? primaryVertex->GetPrimary(0) : nullptr;
    G4double initialEnergy = primaryParticle ? primaryParticle->GetKineticEnergy() : 0.0;
    
    
    // Prefer MFD energy when available; otherwise fall back to internal accumulation (which may be ~0 if not used)
    G4double finalEdep = fScorerEnergyDeposit;
  // Use the first-contact position for x_hit/y_hit; fallback to averaged silicon position
    G4ThreeVector hitPos = fHasFirstContactPos
                             ? fFirstContactPos
                             : ((fNumPosSamples > 0) ? (fPos / (G4double)fNumPosSamples) : G4ThreeVector(0,0,0));
    
    // Determine hit purity via first-contact classification and geometric test
    const bool firstContactIsPixel = fSteppingAction ? fSteppingAction->FirstContactIsPixel() : false;
    fPureSiliconHit = !firstContactIsPixel; // legacy naming retained
    fAluminumContaminated = firstContactIsPixel;
    
    // Calculate nearest pixel and other pixel-related data based on first-contact position
    G4ThreeVector nearestPixel = CalcNearestPixel(hitPos);
    
    // Geometric pixel classification: inside pixel-pad if max(|dx|,|dy|) <= l/2
    // IMPORTANT: compute deltas directly from the authoritative hit position and the
    // returned nearest pixel center to avoid any dependency on earlier sentinel values.
    const G4double halfPixel = fDetector->GetPixelSize()/2.0;
    const G4double dxAbs = std::abs(hitPos.x() - nearestPixel.x());
    const G4double dyAbs = std::abs(hitPos.y() - nearestPixel.y());
    // Persist deltas for downstream consumers and ROOT output
    fPixelTrueDeltaX = dxAbs;
    fPixelTrueDeltaY = dyAbs;
    // Geometric pixel test per request: max(|dx|,|dy|) <= pixel_size/2
    const bool geometricIsPixel = (std::max(dxAbs, dyAbs) <= halfPixel);

    // Pixel-hit flag: first-contact is pixel-pad OR geometric test is inside pad
    const bool isPixelHitCombined = firstContactIsPixel || geometricIsPixel;

    // Enable charge sharing only for non–pixel-pad hits.
    fChargeCalculationEnabled = (!isPixelHitCombined);

    // Calculate neighborhood angles and charge sharing if applicable
    if (fChargeCalculationEnabled) {
        CalcNeighborhoodChargeSharing();
    }
    
    // Pass all data to RunAction for storage

    fRunAction->SetInitialEnergy(initialEnergy);
    fRunAction->SetEventData(finalEdep, hitPos.x(), hitPos.y(), hitPos.z());
    fRunAction->SetNearestPixelPos(nearestPixel.x(), nearestPixel.y());
    // Persist flags and deltas
    fRunAction->SetFirstContactIsPixel(firstContactIsPixel);
    fRunAction->SetGeometricIsPixel(geometricIsPixel);
    fRunAction->SetIsPixelHitCombined(isPixelHitCombined);
    fRunAction->SetPixelClassification(isPixelHitCombined, fPixelTrueDeltaX, fPixelTrueDeltaY);

    // Optional: radius check for non-pixel hits can be persisted if needed (kept for compatibility)
    bool radiusCheck = (!geometricIsPixel) && (std::isfinite(fPixelTrueDeltaX) && std::isfinite(fPixelTrueDeltaY))
                       && (std::max(fPixelTrueDeltaX, fPixelTrueDeltaY) > halfPixel);
    fRunAction->SetNonPixelPadRadiusCheck(radiusCheck);
    fRunAction->SetNeighborhoodChargeData(fNeighborhoodChargeFractions, fNeighborhoodCharge);
    fRunAction->SetScorerData(fScorerEnergyDeposit);
    fRunAction->SetScorerHitCount(fScorerHitCount);

    // Persist hit purity/tracking
    fRunAction->SetHitPurityData(fPureSiliconHit, fAluminumContaminated, fChargeCalculationEnabled);
    
    // Always record the event, including pixel-pad only interactions
    // Pixel-pad hits purposefully have no silicon edep but should be logged
    fRunAction->FillTree();
}

void EventAction::AddSiliconPos(const G4ThreeVector& pos)
{
  fPos += pos;
  fNumPosSamples += 1;
  fHasHit = true;
}

// Implementation of the nearest pixel calculation method
G4ThreeVector EventAction::CalcNearestPixel(const G4ThreeVector& pos)
{
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the position relative to the detector face
  G4ThreeVector relativePos = pos - detectorPos;
  
  // For the AC-LGAD, pixels are on the front surface (z > detector z)
  // Calc the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calc which pixel grid pos is closest (i and j indices)
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  // Check if the hit is within the detector bounds BEFORE clamping
  G4bool isWithinDetector = (i >= 0 && i < numBlocksPerSide && j >= 0 && j < numBlocksPerSide);
  
  // Clamp i and j to valid pixel indices (for geometry purposes)
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calc the actual pixel center position
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  // Pixels are on the detector front surface
  G4double pixelZ = detectorPos.z() + Constants::DETECTOR_WIDTH/2 + Constants::PIXEL_WIDTH/2; // detector half-width + pixel half-width
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calc and store distance from hit to pixel center (2D distance in detector plane)
  G4double dx = pos.x() - pixelX;
  G4double dy = pos.y() - pixelY;
  fActualPixelDistance = std::sqrt(dx*dx + dy*dy);
  
  // Determine if the hit was on a pixel using volume-based detection
  G4bool isOnPixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  
  // Calc and store delta values (pixel center - true pos) for all hits
  // within detector bounds (regardless of pixel contact)
  if (isWithinDetector) {
    fPixelTrueDeltaX = std::abs(pixelX - pos.x());
    fPixelTrueDeltaY = std::abs(pixelY - pos.y());
  } else {
    // For hits outside detector bounds or pixel hits, mark deltas as invalid
    fPixelTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    fPixelTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
  }
  
  // Store the pixel hit status
  fPixelHit = isOnPixel;
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Removed CalcNeighborhoodGridAngles: angle computation is internal to charge sharing only

// Calc the angular size subtended by a pixel as seen from a hit point (2D calculation)
// This now uses the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
G4double EventAction::CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calc distance from hit position to pixel center (2D distance in XY plane)
  G4double dx = hitX - pixelCenterX;
  G4double dy = hitY - pixelCenterY;
  G4double d = std::sqrt(dx*dx + dy*dy);
  
  // Use the pixel size as l (side of the pixel pad)
  // For simplicity, use the average of width and height if they differ
  G4double l = (pixelWidth + pixelHeight) / 2.0;
  
  // Apply the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
  G4double numerator = (l/2.0) * std::sqrt(2.0);
  G4double denominator = numerator + d;
  
  // Handle edge case where denominator could be very small
  G4double alpha;
  if (denominator < Constants::MIN_DENOMINATOR_VALUE) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  return alpha; // Return in radians
}

// Calc charge sharing for pixels in a neighborhood grid around the hit pixel
void EventAction::CalcNeighborhoodChargeSharing()
{
  // Clear previous data
  fNeighborhoodChargeFractions.clear();
  fNeighborhoodCharge.clear();
  
  // Note: Even if no energy was deposited, we still compute F_i fractions
  // so that they sum to 1 for in-bounds cells. Q_i will be zero in that case.
  
  // Charge sharing is executed only for events whose FIRST contact is silicon.
  // This function is called conditionally from EndOfEventAction based on
  // fChargeCalculationEnabled, so no additional pixel-hit checks are needed here.
  
  // Convert energy deposition to number of electrons
  // fScorerEnergyDeposit is in MeV, fIonizationEnergy is in eV
  // Use explicit CLHEP units instead of manual conversion
  G4double edepInEV = fScorerEnergyDeposit * MeV / eV; // Convert MeV to eV using CLHEP units
  G4double numElectrons = edepInEV / fIonizationEnergy;
  
  // Apply AC-LGAD amplification
  G4double totalCharge = numElectrons * fAmplificationFactor;
  
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // D0 constant for charge sharing formula (convert to consistent units)
  // fD0 is in microns, distances are in mm, so convert using CLHEP units
  G4double d0_mm = fD0 * micrometer / mm; // Convert microns to mm using CLHEP units
  
  // Use first-contact position for charge sharing geometry consistently with x_hit/y_hit
  const G4double hitX = fHasFirstContactPos
                          ? fFirstContactPos.x()
                          : ((fNumPosSamples > 0) ? (fPos.x() / static_cast<G4double>(fNumPosSamples)) : fPos.x());
  const G4double hitY = fHasFirstContactPos
                          ? fFirstContactPos.y()
                          : ((fNumPosSamples > 0) ? (fPos.y() / static_cast<G4double>(fNumPosSamples)) : fPos.y());
  
  // Prepare row-major grids for weights and in-bounds flags
  const G4int gridRadius = fNeighborhoodRadius;
  const G4int gridDim = 2 * gridRadius + 1;
  const G4int totalCells = gridDim * gridDim;
  std::vector<G4double> weightGrid(totalCells, 0.0);
  std::vector<G4bool>   inBoundsGrid(totalCells, false);
  
  // First pass: compute weights per grid cell, store into pre-sized arrays by index
  for (G4int di = -gridRadius; di <= gridRadius; ++di) {
    for (G4int dj = -gridRadius; dj <= gridRadius; ++dj) {
      const G4int idx = (di + gridRadius) * gridDim + (dj + gridRadius);
      const G4int gridPixelI = fPixelIndexI + di;
      const G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Bounds check
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide ||
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        inBoundsGrid[idx] = false;
        weightGrid[idx] = 0.0;
        continue;
      }
      
      // Pixel center
      const G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      const G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Geometry
      const G4double dx = hitX - pixelCenterX;
      const G4double dy = hitY - pixelCenterY;
      const G4double distance = std::sqrt(dx*dx + dy*dy);
      const G4double alpha = CalcPixelAlphaSubtended(hitX, hitY, pixelCenterX, pixelCenterY, pixelSize, pixelSize);
      
      // Weight: α_i * 1/ln(d_i/d0) with numerical guard
      G4double weight = 0.0;
      G4double logArg = distance / d0_mm;
      G4double logValue = std::log(logArg);
      logValue = std::max(logValue, Constants::MIN_LOG_VALUE);
      weight = alpha * (1.0 / logValue);
      
      inBoundsGrid[idx] = true;
      weightGrid[idx] = weight;
    }
  }
  
  // Total weight over in-bounds cells
  G4double totalWeight = 0.0;
  for (G4int idx = 0; idx < totalCells; ++idx) {
    if (inBoundsGrid[idx]) totalWeight += weightGrid[idx];
  }
  
  // Reserve and build output vectors in a single ordered pass (row-major),
  // always pushing exactly one value per grid cell
  fNeighborhoodChargeFractions.reserve(totalCells);
  fNeighborhoodCharge.reserve(totalCells);
  for (G4int idx = 0; idx < totalCells; ++idx) {
    if (!inBoundsGrid[idx]) {
      fNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker for OOB
      fNeighborhoodCharge.push_back(0.0);
      continue;
    }
    G4double fraction = (totalWeight > 0.0) ? (weightGrid[idx] / totalWeight) : 0.0;
    G4double chargeCoulombs = fraction * totalCharge * fElementaryCharge;
    fNeighborhoodChargeFractions.push_back(fraction);
    fNeighborhoodCharge.push_back(chargeCoulombs);
  }
}

void EventAction::CollectScorerData(const G4Event* event)
{
  // Reset per–event scorer data
  fScorerEnergyDeposit = 0.0;
  fScorerHitCount      = 0;
  fScorerDataValid     = false;

  // Defensive check – HC may be null for empty events
  G4HCofThisEvent* hce = event->GetHCofThisEvent();
  if (!hce) return;

  // Retrieve collection IDs once (static cache so cost is negligible)
  static G4int edepID  = -1;
  static G4int hitsID  = -1;
  if (edepID < 0 || hitsID < 0)
  {
      G4SDManager* sdm = G4SDManager::GetSDMpointer();
      if(sdm) {
          edepID = sdm->GetCollectionID("SiliconDetector/EnergyDeposit");
          hitsID = sdm->GetCollectionID("SiliconDetector/HitCount");
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

  // Hit-count map
  if (hitsID >= 0)
  {
    auto* hitsMap = dynamic_cast<G4THitsMap<G4int>*>(hce->GetHC(hitsID));
    if (hitsMap && hitsMap->GetMap()->size() > 0)
    {
      for (const auto& kv : *hitsMap->GetMap())
      {
        fScorerHitCount += *(kv.second);
      }
    }
  }

  // Data is valid if there is any energy or any hit count registered
  fScorerDataValid = (fScorerEnergyDeposit > 0.0) || (fScorerHitCount > 0);
}

