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
#include "G4RunManager.hh"
#include <exception>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>

// Alpha calculation method: ANALYTICAL
// This implementation uses the analytical formula for calculating the alpha angle:
// α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
// where:
//   l = side length of the pixel pad (pixel_size)
//   d = distance from event hit to center of pixel pad
// See Page 9: https://indico.cern.ch/event/813597/contributions/3727782/attachments/1989546/3540780/TREDI_Cartiglia.pdf

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetector(detector),
  fSteppingAction(nullptr),
  fNeighborhoodRadius(4), // Default to 9x9 grid (radius 4)
  fInitialPos(G4ThreeVector(0.,0.,0.)),

  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE),
  fScorerEnergyDeposit(0.0),
  fScorerHitCount(0),
  fScorerDataValid(false),
  fPureSiliconHit(false),
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
  
  fEdep = 0.;
  fPos = G4ThreeVector(0.,0.,0.);
  fHasHit = false;
  fInitialPos = G4ThreeVector(0.,0.,0.);
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDeltaX = 0.;
  fPixelTrueDeltaY = 0.;
  fActualPixelDistance = -1.;
  fPixelHit = false;

  fNeighborhoodChargeFractions.clear();
  fNeighborhoodDistances.clear();
  fNeighborhoodCharge.clear();

  // Reset scorer data.
  fScorerEnergyDeposit = 0.0;
  fScorerHitCount = 0;
  fScorerDataValid = false;

  // Reset hit purity data.
  fPureSiliconHit = false;
  fChargeCalculationEnabled = false;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    // Collect data from the multi-functional detector first.
    CollectScorerData(event);

    // Use the MFD data for energy deposition.
    G4double finalEdep = fScorerDataValid ? fScorerEnergyDeposit : 0.0;

    // The hit position is still determined by the energy-weighted average
    // from SteppingAction for charge sharing calculations.
    G4ThreeVector hitPos = fHasHit ? fPos : G4ThreeVector(0, 0, 0);

    // Determine hit purity and if charge sharing should be enabled.
    fPureSiliconHit = fSteppingAction ? fSteppingAction->IsValidSiliconHit() : false;
    fChargeCalculationEnabled = fPureSiliconHit && fScorerDataValid;

    // Calculate and store nearest pixel position (this calculates fActualPixelDistance)
    G4ThreeVector nearestPixel = CalcNearestPixel(hitPos);

    // If charge sharing is enabled, perform the calculation.
    if (fChargeCalculationEnabled) {
        CalcNeighborhoodChargeSharing();
        CalcNeighborhoodGridAngles(hitPos, fPixelIndexI, fPixelIndexJ);
    }

    // Pass all relevant data to RunAction for storage.
    fRunAction->SetEventData(finalEdep, hitPos.x(), hitPos.y(), hitPos.z());
    fRunAction->SetInitialPos(fInitialPos.x(), fInitialPos.y(), fInitialPos.z());
    fRunAction->SetNearestPixelPos(nearestPixel.x(), nearestPixel.y());
    fRunAction->SetPixelHitStatus(fSteppingAction ? fSteppingAction->IsPixelHit() : false);
    fRunAction->SetPixelClassification(fPixelHit, fPixelTrueDeltaX, fPixelTrueDeltaY);

    if (fChargeCalculationEnabled) {
        fRunAction->SetNeighborhoodGridData(fNeighborhoodAngles);
        fRunAction->SetNeighborhoodChargeData(fNeighborhoodChargeFractions, fNeighborhoodDistances, fNeighborhoodCharge, fNeighborhoodCharge);
    }
    
    fRunAction->SetHitPurityData(fPureSiliconHit, fChargeCalculationEnabled);

    // Fill the ROOT tree for this event.
    fRunAction->FillTree();
}

void EventAction::AddEdep(G4double edep, G4ThreeVector pos)
{
  // Accumulate energy deposited while particle travels through detector volume
  // Energy weighted pos calculation
  if (edep > 0) {
    if (!fHasHit) {
      fPos = pos * edep;
      fEdep = edep;  // First energy depositionit in detector
      fHasHit = true;
    } else {
      // Weight pos by energy deposition and sum total energy
      fPos = (fPos * fEdep + pos * edep) / (fEdep + edep);
      fEdep += edep;  // Accumulate total energy deposited in detector
    }
  }
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
  
  // Calc the pos relative to the detector face
  G4ThreeVector relativePos = pos - detectorPos;
  
  // For the AC-LGAD, pixels are on the front surface (z > detector z)
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calc which pixel grid pos is closest (i and j indices)
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  // Check if the hit is within the detector bounds BEFORE clamping
  G4bool isWithinDetector = (i >= 0 && i < numBlocksPerSide && j >= 0 && j < numBlocksPerSide);
  
  // Clamp i and j to valid pixel indices (for geometry purposes)
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calc the actual pixel center pos
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
  
  // Calc and store delta values (pixel center - true pos)
  // Only calculate meaningful deltas for hits within detector bounds AND non-pixel hits
  if (isWithinDetector && !isOnPixel) {
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

// Calc the angular size of pixel from hit pos
G4double EventAction::CalcPixelAlpha(const G4ThreeVector& hitPos, G4int pixelI, G4int pixelJ)
{
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  if (isInsidePixel) {
    return std::numeric_limits<G4double>::quiet_NaN(); // Return NaN for hits inside pixels (no alpha calculation needed)
  }

  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calc the center pos of the specified pixel
  G4double pixelCenterX = firstPixelPos + pixelI * pixelSpacing;
  G4double pixelCenterY = firstPixelPos + pixelJ * pixelSpacing;
  G4double pixelCenterZ = detectorPos.z();
  
  // Calc distance from hit pos to pixel center (2D distance in XY plane)
  G4double dx = hitPos.x() - pixelCenterX;
  G4double dy = hitPos.y() - pixelCenterY;
  G4double d = std::sqrt(dx*dx + dy*dy);
  
  // Use the pixel size as l (side of the pixel pad)
  G4double l = pixelSize;
  
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
  
  // Convert to degrees for storage
  G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
  
  return alphaInDegrees;
}

// Calc angles from hit pos to all pixels in a neighborhood grid around the hit pixel
void EventAction::CalcNeighborhoodGridAngles(const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ)
{
  // Clear previous data
  fNeighborhoodAngles.clear();
  
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  if (isInsidePixel) {
    // Fill all poss with NaN for inside-pixel hits
    G4int gridSize = 2 * fNeighborhoodRadius + 1;
    G4int totalPixels = gridSize * gridSize;
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        // Calc the pixel indices for this grid pos
        G4int gridPixelI = hitPixelI + di;
        G4int gridPixelJ = hitPixelJ + dj;
        
        // Store NaN angle
        fNeighborhoodAngles.push_back(std::numeric_limits<G4double>::quiet_NaN()); // Use same NaN as elsewhere
      }
    }
    return; // Exit early for inside-pixel hits
  }
  
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = hitPixelI + di;
      G4int gridPixelJ = hitPixelJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNeighborhoodAngles.push_back(-999.0); // Invalid angle marker
        continue;
      }
      
      // Calc the center pos of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calc the alpha angle for this pixel using the same algorithm as the Python demo
      G4double alpha = CalcPixelAlphaSubtended(hitPos.x(), hitPos.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Convert to degrees
      G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
      
      // Store the results
      fNeighborhoodAngles.push_back(alphaInDegrees);
    }
  }
}

// Calc the angular size subtended by a pixel as seen from a hit point (2D calculation)
// This now uses the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
G4double EventAction::CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calc distance from hit pos to pixel center (2D distance in XY plane)
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
  fNeighborhoodDistances.clear();
  fNeighborhoodCharge.clear();
  
  // Check if no energy was deposited
  if (fEdep <= 0) {
    // Fill all poss with zero for no-energy events
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        fNeighborhoodChargeFractions.push_back(0.0);
        fNeighborhoodDistances.push_back(-999.0);
        fNeighborhoodCharge.push_back(0.0);
      }
    }
    return;
  }
  
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  if (isInsidePixel) {
    // For pixel hits, energy deposition should be zero (per user requirement)
    // Therefore, charge is also zero
    G4double totalCharge = 0.0;
    
    // Fill all poss, giving all charge to the hit pixel and zero to others
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this is the pixel that was hit
        if (di == 0 && dj == 0) {
          // This is the center pixel (the one that was hit) - but charge is zero for pixel hits
          fNeighborhoodChargeFractions.push_back(0.0);
          fNeighborhoodCharge.push_back(0.0);
          fNeighborhoodDistances.push_back(0.0); // Distance to center of hit pixel is effectively zero
        } else if (gridPixelI >= 0 && gridPixelI < fDetector->GetNumBlocksPerSide() && 
                   gridPixelJ >= 0 && gridPixelJ < fDetector->GetNumBlocksPerSide()) {
          // This is a valid pixel in the detector but not the hit pixel
          fNeighborhoodChargeFractions.push_back(0.0);
          fNeighborhoodCharge.push_back(0.0);
          
        // Calc distance to this pixel center for completeness
        G4double pixelSize = fDetector->GetPixelSize();
        G4double pixelSpacing = fDetector->GetPixelSpacing();
        G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
        G4double detSize = fDetector->GetDetSize();
        G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
        
        G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
        G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
        G4double dx = fPos.x() - pixelCenterX;
        G4double dy = fPos.y() - pixelCenterY;
        G4double distance = std::sqrt(dx*dx + dy*dy);
          fNeighborhoodDistances.push_back(distance);
        } else {
          // This pixel is outside the detector bounds
          fNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fNeighborhoodDistances.push_back(-999.0);
          fNeighborhoodCharge.push_back(0.0);
        }
      }
    }
    return;
  }
  
  // Convert energy depositionit to number of electrons
  // fEdep is in MeV, fIonizationEnergy is in eV
  // Use explicit CLHEP units instead of manual conversion
  G4double edepInEV = fEdep * MeV / eV; // Convert MeV to eV using CLHEP units
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
  
  // Proceed with charge sharing calculation for non-pixel hits
  // First pass: collect valid pixels and calculate weights
  std::vector<G4double> weights;
  std::vector<G4double> distances;
  std::vector<G4double> angles;
  std::vector<G4int> validPixelI;
  std::vector<G4int> validPixelJ;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
        fNeighborhoodDistances.push_back(-999.0);
        fNeighborhoodCharge.push_back(0.0);
        continue;
      }
      
      // Calc the center pos of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calc the distance from the hit to the pixel center (in mm)
      G4double dx = fPos.x() - pixelCenterX;
      G4double dy = fPos.y() - pixelCenterY;
      G4double distance = std::sqrt(dx*dx + dy*dy);
      
      // Calc the alpha angle for this pixel using the same algorithm as elsewhere
      G4double alpha = CalcPixelAlphaSubtended(fPos.x(), fPos.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Store data for valid pixels
      distances.push_back(distance);
      angles.push_back(alpha);
      validPixelI.push_back(gridPixelI);
      validPixelJ.push_back(gridPixelJ);
      
      // Calc weight according to formula: α_i * ln(d_i/d_0)^(-1)
      // Handle the case where distance might be very small or zero
      G4double weight = 0.0;
      if (distance > d0_mm) {
        // Clamp the logarithm to avoid infinite weights when d ≈ d0
        G4double logArg = distance / d0_mm;
        G4double logValue = std::log(logArg);
        // Clamp log value to avoid division by very small numbers
        logValue = std::max(logValue, Constants::MIN_LOG_VALUE);
        weight = alpha * (1.0 / logValue);
      } else if (distance > 0) {
        // For very small distances, use a large weight
        weight = alpha * Constants::ALPHA_WEIGHT_MULTIPLIER; // Large weight for very close pixels
      } else {
        // Distance is zero (hit exactly on pixel center), give maximum weight
        weight = alpha * Constants::ALPHA_WEIGHT_MULTIPLIER;
      }
      
      weights.push_back(weight);
    }
  }
  
  // Calc total weight
  G4double totalWeight = 0.0;
  for (G4double weight : weights) {
    totalWeight += weight;
  }
  
  // Sec pass: calculate charge fractions and values
  size_t validIndex = 0;
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Already stored invalid data in first pass
        continue;
      }
      
      // Calc charge fraction and value
      G4double chargeFraction = 0.0;
      G4double chargeValue = 0.0;
      
      if (totalWeight > 0) {
        chargeFraction = weights[validIndex] / totalWeight;
        chargeValue = chargeFraction * totalCharge;
      }
      
      fNeighborhoodChargeFractions.push_back(chargeFraction);
      fNeighborhoodDistances.push_back(distances[validIndex]);
      fNeighborhoodCharge.push_back(chargeValue * fElementaryCharge);
      
      validIndex++;
    }
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

  // Data is considered valid if there were any hits recorded by the scorer.
  // This is the primary condition for enabling downstream processing like charge sharing.
  fScorerDataValid = (fScorerHitCount > 0);
}

