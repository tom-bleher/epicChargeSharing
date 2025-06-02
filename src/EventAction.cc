#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "MinuitGaussianFitter.hh"
#include "Constants.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

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
  fNeighborhoodRadius(4), // Default to 9x9 grid (radius 4)
  fEdep(0.),
  fPosition(G4ThreeVector(0.,0.,0.)),
  fInitialPosition(G4ThreeVector(0.,0.,0.)),
  fHasHit(false),
  fPixelIndexI(-1),
  fPixelIndexJ(-1),
  fPixelTrueDist(-1.),
  fPixelHit(false),
  fInitialParticleEnergy(0.),
  fFinalParticleEnergy(0.),
  fParticleMomentum(0.),
  fParticleName(""),
  fGaussianFitter(nullptr)
{ 
  // Create the 3D Gaussian fitter instance with detector geometry constraints
  // Gaussian3DFitter::DetectorGeometry geometry;

  // Create the Minuit-based 3D Gaussian fitter instance with detector geometry constraints
  MinuitGaussianFitter::DetectorGeometry geometry;

  geometry.detector_size = fDetector->GetDetSize();
  geometry.pixel_size = fDetector->GetPixelSize();
  geometry.pixel_spacing = fDetector->GetPixelSpacing();
  geometry.pixel_corner_offset = fDetector->GetPixelCornerOffset();
  geometry.num_blocks_per_side = fDetector->GetNumBlocksPerSide();
  geometry.pixel_exclusion_buffer = Constants::PIXEL_EXCLUSION_BUFFER; // 10 microns as requested
  
  //fGaussianFitter = new Gaussian3DFitter(geometry);
  fGaussianFitter = new MinuitGaussianFitter(geometry);
  
  G4cout << "EventAction: Using ROOT Minuit-based Gaussian fitter" << G4endl;
}

EventAction::~EventAction()
{ 
  // Clean up the Minuit-based 3D Gaussian fitter
  if (fGaussianFitter) {
    delete fGaussianFitter;
    fGaussianFitter = nullptr;
  }
}

void EventAction::BeginOfEventAction(const G4Event* event)
{
  // Reset per-event variables
  fEdep = 0.;
  fPosition = G4ThreeVector(0.,0.,0.);
  fHasHit = false;
  
  // Initialize particle position - this will be updated when the primary vertex is created
  fInitialPosition = G4ThreeVector(0.,0.,0.);
  
  // Reset pixel mapping variables
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDist = -1.;
  fPixelHit = false;
  
  // Reset neighborhood (9x9) grid angle data
  fGridNeighborhoodAngles.clear();
  fGridNeighborhoodPixelI.clear();
  fGridNeighborhoodPixelJ.clear();
  
  // Reset neighborhood (9x9) grid charge sharing data
  fGridNeighborhoodChargeFractions.clear();
  fGridNeighborhoodDistances.clear();
  fGridNeighborhoodCharge.clear();
  
  // Reset additional tracking variables
  fInitialParticleEnergy = 0.;
  fFinalParticleEnergy = 0.;
  fParticleMomentum = 0.;
  fParticleName = "";
  
  // Reset step energy deposition data
  fStepEnergyDeposition.clear();
  fStepZPositions.clear();
  fStepTimes.clear();
  fStepNumbers.clear();
  
  // Reset ALL step data
  fAllStepEnergyDeposition.clear();
  fAllStepZPositions.clear();
  fAllStepTimes.clear();
  fAllStepNumbers.clear();
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // Get the primary vertex position from the event
  if (event->GetPrimaryVertex()) {
    G4ThreeVector primaryPos = event->GetPrimaryVertex()->GetPosition();
    // Update the initial position
    fInitialPosition = primaryPos;
    
    // Get particle information from the primary vertex
    if (event->GetPrimaryVertex()->GetPrimary()) {
      G4PrimaryParticle* primary = event->GetPrimaryVertex()->GetPrimary();
      fInitialParticleEnergy = primary->GetKineticEnergy();
      fParticleMomentum = primary->GetMomentum().mag();
      
      // Get particle definition from PDG code
      G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
      if (G4ParticleDefinition* particleDef = particleTable->FindParticle(primary->GetPDGcode())) {
        fParticleName = particleDef->GetParticleName();
      }
    }
  }
  
  // Set event ID
  G4int eventID = event->GetEventID();
  
  // Always record event data, even if no energy was deposited
  // Values are passed in Geant4's internal units (MeV for energy, mm for length)
  fRunAction->SetEventData(fEdep, fPosition.x(), fPosition.y(), fPosition.z());
  fRunAction->SetInitialPosition(fInitialPosition.x(), fInitialPosition.y(), fInitialPosition.z());
  
  // Calculate and store nearest pixel position (in mm)
  G4ThreeVector nearestPixel = CalculateNearestPixel(fPosition);
  fRunAction->SetNearestPixelPosition(nearestPixel.x(), nearestPixel.y(), nearestPixel.z());
  
  // Pass pixel indices and distance information to RunAction
  fRunAction->SetPixelIndices(fPixelIndexI, fPixelIndexJ, fPixelTrueDist);
  
  // Calculate and pass pixel angular size to RunAction
  G4double pixelAlpha = CalculatePixelAlpha(fPosition, fPixelIndexI, fPixelIndexJ);
  fRunAction->SetPixelAlpha(pixelAlpha);
  
  // Calculate neighborhood (9x9) grid angles and pass to RunAction
  CalculateNeighborhoodGridAngles(fPosition, fPixelIndexI, fPixelIndexJ);
  fRunAction->SetNeighborhoodGridData(fGridNeighborhoodAngles, fGridNeighborhoodPixelI, fGridNeighborhoodPixelJ);
  
  // Calculate neighborhood (9x9) grid charge sharing and pass to RunAction
  CalculateNeighborhoodChargeSharing();
  fRunAction->SetNeighborhoodChargeData(fGridNeighborhoodChargeFractions, fGridNeighborhoodDistances, fGridNeighborhoodCharge, fGridNeighborhoodCharge);
  
  // Pass pixel hit status to RunAction
  fRunAction->SetPixelHit(fPixelHit);
  
  // Pass particle information to RunAction
  fRunAction->SetParticleInfo(eventID, fInitialParticleEnergy, fFinalParticleEnergy, 
                             fParticleMomentum, fParticleName);
  
  // Pass step energy deposition information to RunAction
  fRunAction->SetStepEnergyDeposition(fStepEnergyDeposition, fStepZPositions, fStepTimes);
  
  // Pass ALL step information to RunAction
  fRunAction->SetAllStepInfo(fAllStepEnergyDeposition, fAllStepZPositions, fAllStepTimes);
  
  // Perform 3D Gaussian fitting on charge distribution data
  if (fGaussianFitter && !fGridNeighborhoodChargeFractions.empty()) {
    // Extract coordinates and charge values for fitting
    std::vector<G4double> x_coords, y_coords, z_values;
    
    // Get detector parameters for coordinate calculation
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    
    // Convert grid indices to actual coordinates
    for (size_t i = 0; i < fGridNeighborhoodChargeFractions.size(); ++i) {
      if (fGridNeighborhoodChargeFractions[i] > 0) { // Only include pixels with charge
        // Calculate position relative to the nearest pixel center
        G4int pixelI = fGridNeighborhoodPixelI[i];
        G4int pixelJ = fGridNeighborhoodPixelJ[i];
        
        // Calculate offset from center pixel (which is at index 4,4 in 9x9 grid)
        G4int centerI = fPixelIndexI;
        G4int centerJ = fPixelIndexJ;
        
        G4double x_pos = nearestPixel.x() + (pixelI - centerI) * pixelSpacing;
        G4double y_pos = nearestPixel.y() + (pixelJ - centerJ) * pixelSpacing;
        
        x_coords.push_back(x_pos);
        y_coords.push_back(y_pos);
        z_values.push_back(fGridNeighborhoodChargeFractions[i]);
      }
    }
    
    // Perform fitting if we have enough data points
    if (x_coords.size() >= Constants::MIN_FIT_POINTS) { // Need at least 4 points for meaningful fit
      // Perform fitting with all data
      
      MinuitGaussianFitter::FitResults fitResults = fGaussianFitter->FitGaussian3D(
        x_coords, y_coords, z_values, std::vector<G4double>(), false); // verbose=false
      
      // Pass fit results to RunAction
      fRunAction->SetGaussianFitResults(
        fitResults.amplitude, fitResults.x0, fitResults.y0,
        fitResults.sigma_x, fitResults.sigma_y, fitResults.theta, fitResults.offset,
        fitResults.amplitude_err, fitResults.x0_err, fitResults.y0_err,
        fitResults.sigma_x_err, fitResults.sigma_y_err, fitResults.theta_err, fitResults.offset_err,
        fitResults.chi2red, fitResults.ndf, fitResults.Pp,
        fitResults.n_points,
        fitResults.residual_mean, fitResults.residual_std,
        fitResults.constraints_satisfied);
        
    } else {
      // Not enough data points for fitting - set default values
      fRunAction->SetGaussianFitResults(
        0, 0, 0, 0, 0, 0, 0,  // parameters
        0, 0, 0, 0, 0, 0, 0,  // errors
        0, 0, 0,              // chi2red, ndf, Pp
        0, 0, 0, false); // n_points, residual stats, constraints
    }
  } else {
    // No fitter or no charge data - set default values
    fRunAction->SetGaussianFitResults(
      0, 0, 0, 0, 0, 0, 0,  // parameters
      0, 0, 0, 0, 0, 0, 0,  // errors
      0, 0, 0,              // chi2red, ndf, Pp
      0, 0, 0, false); // n_points, residual stats, constraints
  }
  
  fRunAction->FillTree();
}

void EventAction::AddEdep(G4double edep, G4ThreeVector position)
{
  // Energy weighted position calculation
  if (edep > 0) {
    if (!fHasHit) {
      fPosition = position * edep;
      fEdep = edep;
      fHasHit = true;
    } else {
      // Weight position by energy deposition
      fPosition = (fPosition * fEdep + position * edep) / (fEdep + edep);
      fEdep += edep;
    }
  }
}

// Implementation of the new method to set the initial position
void EventAction::SetInitialPosition(const G4ThreeVector& position)
{
  fInitialPosition = position;
}

// Implementation of the nearest pixel calculation method
G4ThreeVector EventAction::CalculateNearestPixel(const G4ThreeVector& position)
{
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the position relative to the detector face
  G4ThreeVector relativePos = position - detectorPosition;
  
  // For the AC-LGAD, pixels are on the front surface (z > detector z)
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calculate which pixel grid position is closest (i and j indices)
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  // Clamp i and j to valid pixel indices
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calculate the actual pixel center position
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  // Pixels are on the detector front surface
  G4double pixelZ = detectorPosition.z() + 50*um/2 + 1*um/2; // detector half-width + pixel half-width
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calculate and store distance from hit to pixel center (2D distance in detector plane)
  fPixelTrueDist = std::sqrt(std::pow(position.x() - pixelX, 2) + 
                            std::pow(position.y() - pixelY, 2));
  
  // Determine if the hit was on a pixel using the detector's method
  fPixelHit = fDetector->IsPositionOnPixel(position);
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Calculate the angular size of pixel from hit position
G4double EventAction::CalculatePixelAlpha(const G4ThreeVector& hitPosition, G4int pixelI, G4int pixelJ)
{
  // Check if the hit is inside the pixel. If so, return NaN to indicate no alpha calculation
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    return std::numeric_limits<G4double>::quiet_NaN(); // Return NaN for hits inside pixels (no alpha calculation needed)
  }

  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calculate the center position of the specified pixel
  G4double pixelCenterX = firstPixelPos + pixelI * pixelSpacing;
  G4double pixelCenterY = firstPixelPos + pixelJ * pixelSpacing;
  G4double pixelCenterZ = detectorPosition.z();
  
  // Calculate distance from hit position to pixel center (2D distance in XY plane)
  G4double d = std::sqrt(std::pow(hitPosition.x() - pixelCenterX, 2) + 
                        std::pow(hitPosition.y() - pixelCenterY, 2));
  
  // Use the pixel size as l (side of the pixel pad)
  G4double l = pixelSize;
  
  // Apply the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
  G4double numerator = (l/2.0) * std::sqrt(2.0);
  G4double denominator = numerator + d;
  
  // Handle edge case where denominator could be very small
  G4double alpha;
  if (denominator < 1e-10) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  // Convert to degrees for storage
  G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
  
  return alphaInDegrees;
}

// Calculate angles from hit position to all pixels in a neighborhood grid around the hit pixel
void EventAction::CalculateNeighborhoodGridAngles(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ)
{
  // Clear previous data
  fGridNeighborhoodAngles.clear();
  fGridNeighborhoodPixelI.clear();
  fGridNeighborhoodPixelJ.clear();
  
  // Check if hit is inside a pixel - if so, all angles should be invalid
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    // Fill all positions with NaN for inside-pixel hits
    G4int gridSize = 2 * fNeighborhoodRadius + 1;
    G4int totalPixels = gridSize * gridSize;
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        // Calculate the pixel indices for this grid position
        G4int gridPixelI = hitPixelI + di;
        G4int gridPixelJ = hitPixelJ + dj;
        
        // Store pixel indices and NaN angle
        fGridNeighborhoodPixelI.push_back(gridPixelI);
        fGridNeighborhoodPixelJ.push_back(gridPixelJ);
        fGridNeighborhoodAngles.push_back(std::numeric_limits<G4double>::quiet_NaN()); // Use same NaN as elsewhere
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
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = hitPixelI + di;
      G4int gridPixelJ = hitPixelJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fGridNeighborhoodPixelI.push_back(gridPixelI);
        fGridNeighborhoodPixelJ.push_back(gridPixelJ);
        fGridNeighborhoodAngles.push_back(-999.0); // Invalid angle marker
        continue;
      }
      
      // Calculate the center position of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calculate the alpha angle for this pixel using the same algorithm as the Python demo
      G4double alpha = CalculatePixelAlphaSubtended(hitPosition.x(), hitPosition.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Convert to degrees
      G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
      
      // Store the results
      fGridNeighborhoodPixelI.push_back(gridPixelI);
      fGridNeighborhoodPixelJ.push_back(gridPixelJ);
      fGridNeighborhoodAngles.push_back(alphaInDegrees);
    }
  }
}

// Calculate the angular size subtended by a pixel as seen from a hit point (2D calculation)
// This now uses the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
G4double EventAction::CalculatePixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calculate distance from hit position to pixel center (2D distance in XY plane)
  G4double d = std::sqrt(std::pow(hitX - pixelCenterX, 2) + 
                        std::pow(hitY - pixelCenterY, 2));
  
  // Use the pixel size as l (side of the pixel pad)
  // For simplicity, use the average of width and height if they differ
  G4double l = (pixelWidth + pixelHeight) / 2.0;
  
  // Apply the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
  G4double numerator = (l/2.0) * std::sqrt(2.0);
  G4double denominator = numerator + d;
  
  // Handle edge case where denominator could be very small
  G4double alpha;
  if (denominator < 1e-10) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  return alpha; // Return in radians
}

// Calculate charge sharing for pixels in a neighborhood grid around the hit pixel
void EventAction::CalculateNeighborhoodChargeSharing()
{
  // Clear previous data
  fGridNeighborhoodChargeFractions.clear();
  fGridNeighborhoodDistances.clear();
  fGridNeighborhoodCharge.clear();
  
  // Check if no energy was deposited
  if (fEdep <= 0) {
    // Fill all positions with zero for no-energy events
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        fGridNeighborhoodChargeFractions.push_back(0.0);
        fGridNeighborhoodDistances.push_back(-999.0);
        fGridNeighborhoodCharge.push_back(0.0);
      }
    }
    return;
  }
  
  // Check if hit is inside a pixel - if so, assign all charge to that pixel
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(fPosition);
  if (isInsidePixel) {
    // Convert energy deposit to number of electrons and apply amplification
    G4double edepInEV = fEdep * 1e6; // Convert MeV to eV
    G4double numElectrons = edepInEV / fIonizationEnergy;
    G4double totalCharge = numElectrons * fAmplificationFactor;
    
    // Fill all positions, giving all charge to the hit pixel and zero to others
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this is the pixel that was hit
        if (di == 0 && dj == 0) {
          // This is the center pixel (the one that was hit)
          fGridNeighborhoodChargeFractions.push_back(1.0);
          fGridNeighborhoodCharge.push_back(totalCharge * fElementaryCharge);
          fGridNeighborhoodDistances.push_back(0.0); // Distance to center of hit pixel is effectively zero
        } else if (gridPixelI >= 0 && gridPixelI < fDetector->GetNumBlocksPerSide() && 
                   gridPixelJ >= 0 && gridPixelJ < fDetector->GetNumBlocksPerSide()) {
          // This is a valid pixel in the detector but not the hit pixel
          fGridNeighborhoodChargeFractions.push_back(0.0);
          fGridNeighborhoodCharge.push_back(0.0);
          
          // Calculate distance to this pixel center for completeness
          G4double pixelSize = fDetector->GetPixelSize();
          G4double pixelSpacing = fDetector->GetPixelSpacing();
          G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
          G4double detSize = fDetector->GetDetSize();
          G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
          
          G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
          G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
          G4double distance = std::sqrt(std::pow(fPosition.x() - pixelCenterX, 2) + 
                                       std::pow(fPosition.y() - pixelCenterY, 2));
          fGridNeighborhoodDistances.push_back(distance);
        } else {
          // This pixel is outside the detector bounds
          fGridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fGridNeighborhoodDistances.push_back(-999.0);
          fGridNeighborhoodCharge.push_back(0.0);
        }
      }
    }
    return;
  }
  
  // Convert energy deposit to number of electrons
  // fEdep is in MeV, fIonizationEnergy is in eV
  // Convert MeV to eV: 1 MeV = 1e6 eV
  G4double edepInEV = fEdep * 1e6; // Convert MeV to eV
  G4double numElectrons = edepInEV / fIonizationEnergy;
  
  // Apply AC-LGAD amplification
  G4double totalCharge = numElectrons * fAmplificationFactor;
  
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  G4double d0_mm = fD0 * 1e-3; // Convert microns to mm
  
  // Check if the distance from hit to the identified pixel center (fPixelIndexI, fPixelIndexJ) is <= D0
  G4double pixelCenterX = firstPixelPos + fPixelIndexI * pixelSpacing;
  G4double pixelCenterY = firstPixelPos + fPixelIndexJ * pixelSpacing;
  G4double distanceToIdentifiedPixel = std::sqrt(std::pow(fPosition.x() - pixelCenterX, 2) + 
                                                 std::pow(fPosition.y() - pixelCenterY, 2));
  
  // If the distance to the identified pixel center is <= D0, assign all charge to that pixel
  // This ensures consistency with the inside-pixel case
  if (distanceToIdentifiedPixel <= d0_mm) {
    // Calculate the charge without charge sharing
    G4double totalChargeValue = totalCharge;
    
    // Fill all positions, giving all charge to the identified pixel (fPixelIndexI, fPixelIndexJ) and zero to others
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this pixel is within bounds
        if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
            gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
          // Store invalid data for out-of-bounds pixels
          fGridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fGridNeighborhoodDistances.push_back(-999.0);
          fGridNeighborhoodCharge.push_back(0.0);
          continue;
        }
        
        // Calculate the center position and distance for this pixel
        G4double currentPixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
        G4double currentPixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
        G4double distance = std::sqrt(std::pow(fPosition.x() - currentPixelCenterX, 2) + 
                                     std::pow(fPosition.y() - currentPixelCenterY, 2));
        
        // Check if this is the identified pixel (center of the neighborhood grid)
        if (di == 0 && dj == 0) {
          // This is the identified pixel (fPixelIndexI, fPixelIndexJ) - assign all charge here
          fGridNeighborhoodChargeFractions.push_back(1.0);
          fGridNeighborhoodCharge.push_back(totalChargeValue * fElementaryCharge);
          fGridNeighborhoodDistances.push_back(distance);
        } else {
          // All other pixels get zero charge
          fGridNeighborhoodChargeFractions.push_back(0.0);
          fGridNeighborhoodCharge.push_back(0.0);
          fGridNeighborhoodDistances.push_back(distance);
        }
      }
    }
    return; // Exit early, no need for normal charge sharing calculation
  }
  
  // If distance > D0, proceed with normal charge sharing calculation
  // First pass: collect valid pixels and calculate weights
  std::vector<G4double> weights;
  std::vector<G4double> distances;
  std::vector<G4double> angles;
  std::vector<G4int> validPixelI;
  std::vector<G4int> validPixelJ;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fGridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
        fGridNeighborhoodDistances.push_back(-999.0);
        fGridNeighborhoodCharge.push_back(0.0);
        continue;
      }
      
      // Calculate the center position of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calculate the distance from the hit to the pixel center (in mm)
      G4double distance = std::sqrt(std::pow(fPosition.x() - pixelCenterX, 2) + 
                                   std::pow(fPosition.y() - pixelCenterY, 2));
      
      // Calculate the alpha angle for this pixel using the same algorithm as elsewhere
      G4double alpha = CalculatePixelAlphaSubtended(fPosition.x(), fPosition.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Store data for valid pixels
      distances.push_back(distance);
      angles.push_back(alpha);
      validPixelI.push_back(gridPixelI);
      validPixelJ.push_back(gridPixelJ);
      
      // Calculate weight according to formula: α_i * ln(d_i/d_0)^(-1)
      // Handle the case where distance might be very small or zero
      G4double weight = 0.0;
      if (distance > d0_mm) {
        weight = alpha * (1.0 / std::log(distance / d0_mm));
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
  
  // Calculate total weight
  G4double totalWeight = 0.0;
  for (G4double weight : weights) {
    totalWeight += weight;
  }
  
  // Second pass: calculate charge fractions and values
  size_t validIndex = 0;
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Already stored invalid data in first pass
        continue;
      }
      
      // Calculate charge fraction and value
      G4double chargeFraction = 0.0;
      G4double chargeValue = 0.0;
      
      if (totalWeight > 0) {
        chargeFraction = weights[validIndex] / totalWeight;
        chargeValue = chargeFraction * totalCharge;
      }
      
      fGridNeighborhoodChargeFractions.push_back(chargeFraction);
      fGridNeighborhoodDistances.push_back(distances[validIndex]);
      fGridNeighborhoodCharge.push_back(chargeValue * fElementaryCharge);
      
      validIndex++;
    }
  }
}

void EventAction::SetFinalParticleEnergy(G4double finalEnergy)
{
  fFinalParticleEnergy = finalEnergy;
}

void EventAction::AddStepEnergyDeposition(G4double edep, G4double z, G4double time)
{
  // Only store steps that actually deposit energy
  if (edep > 0.) {
    fStepEnergyDeposition.push_back(edep);
    fStepZPositions.push_back(z);
    fStepTimes.push_back(time / CLHEP::ns); // Convert to ns
  }
}

void EventAction::AddAllStepInfo(G4double edep, G4double z, G4double time)
{
  // Store ALL steps, including those with zero energy deposition
  fAllStepEnergyDeposition.push_back(edep);
  fAllStepZPositions.push_back(z);
  fAllStepTimes.push_back(time / CLHEP::ns); // Convert to ns
}