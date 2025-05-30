#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "Gaussian3DFitter.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetector(detector),
  fEdep(0.),
  fPosition(G4ThreeVector(0.,0.,0.)),
  fInitialPosition(G4ThreeVector(0.,0.,0.)),
  fHasHit(false),
  fPixelIndexI(-1),
  fPixelIndexJ(-1),
  fPixelDistance(-1.),
  fPixelHit(false),
  fInitialParticleEnergy(0.),
  fFinalParticleEnergy(0.),
  fParticleMomentum(0.),
  fParticleName(""),
  fCreatorProcess(""),
  fGlobalTime(0.),
  fLocalTime(0.),
  fProperTime(0.),
  fPhysicsProcess(""),
  fTrackID(-1),
  fParentID(-1),
  fStepCount(0),
  fTotalStepLength(0.),
  fGaussianFitter(nullptr)
{ 
  // Create the 3D Gaussian fitter instance
  fGaussianFitter = new Gaussian3DFitter();
}

EventAction::~EventAction()
{ 
  // Clean up the 3D Gaussian fitter
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
  fPixelDistance = -1.;
  fPixelHit = false;
  
  // Reset neighborhood (9x9) grid angle data
  fGridNeighborhoodAngles.clear();
  fGridNeighborhoodPixelI.clear();
  fGridNeighborhoodPixelJ.clear();
  
  // Reset neighborhood (9x9) grid charge sharing data
  fGridNeighborhoodChargeFractions.clear();
  fGridNeighborhoodDistances.clear();
  fGridNeighborhoodChargeValues.clear();
  fGridNeighborhoodChargeCoulombs.clear();
  
  // Reset additional tracking variables
  fInitialParticleEnergy = 0.;
  fFinalParticleEnergy = 0.;
  fParticleMomentum = 0.;
  fParticleName = "";
  fCreatorProcess = "";
  fGlobalTime = 0.;
  fLocalTime = 0.;
  fProperTime = 0.;
  fPhysicsProcess = "";
  fTrackID = -1;
  fParentID = -1;
  fStepCount = 0;
  fTotalStepLength = 0.;
  
  // Reset trajectory data
  fTrajectoryX.clear();
  fTrajectoryY.clear();
  fTrajectoryZ.clear();
  fTrajectoryTime.clear();
  
  // Reset step energy deposition data
  fStepEdepVec.clear();
  fStepZVec.clear();
  fStepTimeVec.clear();
  fStepLenVec.clear();
  fStepNumVec.clear();
  fStepLenVec.clear();
  fStepNumVec.clear();
  
  // Reset ALL step data
  fAllStepEdepVec.clear();
  fAllStepZVec.clear();
  fAllStepTimeVec.clear();
  fAllStepLenVec.clear();
  fAllStepNumVec.clear();
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
  fRunAction->SetPixelIndices(fPixelIndexI, fPixelIndexJ, fPixelDistance);
  
  // Calculate and pass pixel angular size to RunAction
  G4double pixelAlpha = CalculatePixelAlpha(fPosition, fPixelIndexI, fPixelIndexJ);
  fRunAction->SetPixelAlpha(pixelAlpha);
  
  // Calculate neighborhood (9x9) grid angles and pass to RunAction
  CalculateNeighborhoodGridAngles(fPosition, fPixelIndexI, fPixelIndexJ);
  fRunAction->SetNeighborhoodGridData(fGridNeighborhoodAngles, fGridNeighborhoodPixelI, fGridNeighborhoodPixelJ);
  
  // Calculate neighborhood (9x9) grid charge sharing and pass to RunAction
  CalculateNeighborhoodChargeSharing();
  fRunAction->SetNeighborhoodChargeData(fGridNeighborhoodChargeFractions, fGridNeighborhoodDistances, fGridNeighborhoodChargeValues, fGridNeighborhoodChargeCoulombs);
  
  // Pass pixel hit status to RunAction
  fRunAction->SetPixelHit(fPixelHit);
  
  // Pass particle information to RunAction
  fRunAction->SetParticleInfo(eventID, fInitialParticleEnergy, fFinalParticleEnergy, 
                             fParticleMomentum, fParticleName, fCreatorProcess);
  
  // Pass timing information to RunAction
  fRunAction->SetTimingInfo(fGlobalTime, fLocalTime, fProperTime);
  
  // Pass physics information to RunAction
  fRunAction->SetPhysicsInfo(fPhysicsProcess, fTrackID, fParentID, fStepCount, fTotalStepLength);
  
  // Pass trajectory information to RunAction
  fRunAction->SetTrajectoryInfo(fTrajectoryX, fTrajectoryY, fTrajectoryZ, fTrajectoryTime);
  
  // Pass step energy deposition information to RunAction
  fRunAction->SetStepEnergyDeposition(fStepEdepVec, fStepZVec, fStepTimeVec, 
                                     fStepLenVec, fStepNumVec);
  
  // Pass ALL step information to RunAction
  fRunAction->SetAllStepInfo(fAllStepEdepVec, fAllStepZVec, fAllStepTimeVec, 
                            fAllStepLenVec, fAllStepNumVec);
  
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
    if (x_coords.size() >= 4) { // Need at least 4 points for meaningful fit
      Gaussian3DFitter::FitResults fitResults = fGaussianFitter->FitGaussian3D(
        x_coords, y_coords, z_values, std::vector<G4double>(), false); // verbose=false
      
      // Pass fit results to RunAction
      fRunAction->SetGaussianFitResults(
        fitResults.amplitude, fitResults.x0, fitResults.y0,
        fitResults.sigma_x, fitResults.sigma_y, fitResults.theta, fitResults.offset,
        fitResults.amplitude_err, fitResults.x0_err, fitResults.y0_err,
        fitResults.sigma_x_err, fitResults.sigma_y_err, fitResults.theta_err, fitResults.offset_err,
        fitResults.chi2, fitResults.ndf, fitResults.prob, fitResults.r_squared,
        fitResults.n_points, fitResults.fit_successful,
        fitResults.residual_mean, fitResults.residual_std);
        
      // Optional: Print fit results for debugging (only for first few events)
      if (eventID < 5 && fitResults.fit_successful) {
        G4cout << "Event " << eventID << " - 3D Gaussian Fit Results:" << G4endl;
        G4cout << "  Center: (" << fitResults.x0 << ", " << fitResults.y0 << ") mm" << G4endl;
        G4cout << "  Sigma: (" << fitResults.sigma_x << ", " << fitResults.sigma_y << ") mm" << G4endl;
        G4cout << "  R²: " << fitResults.r_squared << G4endl;
      }
    } else {
      // Not enough data points for fitting - set default values
      fRunAction->SetGaussianFitResults(
        0, 0, 0, 0, 0, 0, 0,  // parameters
        0, 0, 0, 0, 0, 0, 0,  // errors
        0, 0, 0, 0,           // chi2, ndf, prob, r_squared
        x_coords.size(), false, 0, 0); // n_points, fit_successful, residual stats
    }
  } else {
    // No fitter or no charge data - set default values
    fRunAction->SetGaussianFitResults(
      0, 0, 0, 0, 0, 0, 0,  // parameters
      0, 0, 0, 0, 0, 0, 0,  // errors
      0, 0, 0, 0,           // chi2, ndf, prob, r_squared
      0, false, 0, 0);      // n_points, fit_successful, residual stats
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
  fPixelDistance = std::sqrt(std::pow(position.x() - pixelX, 2) + 
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
  
  // Calculate the four corners of the pixel
  G4double halfPixel = pixelSize / 2.0;
  G4ThreeVector corners[4];
  // Bottom-left (0)
  corners[0] = G4ThreeVector(pixelCenterX - halfPixel, pixelCenterY - halfPixel, pixelCenterZ);
  // Bottom-right (1)
  corners[1] = G4ThreeVector(pixelCenterX + halfPixel, pixelCenterY - halfPixel, pixelCenterZ);
  // Top-right (2)
  corners[2] = G4ThreeVector(pixelCenterX + halfPixel, pixelCenterY + halfPixel, pixelCenterZ);
  // Top-left (3)
  corners[3] = G4ThreeVector(pixelCenterX - halfPixel, pixelCenterY + halfPixel, pixelCenterZ);
  
  // Create a vector to store angles and corner indices
  struct AngleInfo {
    G4double angle;
    G4int cornerIndex;
    
    // Custom comparison operator for sorting
    bool operator<(const AngleInfo& other) const {
      return angle < other.angle;
    }
  };
  
  std::vector<AngleInfo> angles;
  
  // Calculate angles to each corner (in the XY plane)
  for (G4int i = 0; i < 4; i++) {
    // Calculate relative position vector from hit to corner
    G4double dx = corners[i].x() - hitPosition.x();
    G4double dy = corners[i].y() - hitPosition.y();
    
    // Calculate angle using atan2
    G4double angle = std::atan2(dy, dx);
    angles.push_back({angle, i});
  }
  
  // Sort by angle
  std::sort(angles.begin(), angles.end());
  
  // Calculate differences between consecutive angles
  std::vector<G4double> angleDiffs;
  for (size_t i = 0; i < angles.size(); i++) {
    size_t nextI = (i + 1) % angles.size();
    G4double diff = angles[nextI].angle - angles[i].angle;
    
    // Handle wrap-around (angles close to 2π)
    if (diff < 0) {
      diff += 2.0 * CLHEP::pi;
    }
    
    angleDiffs.push_back(diff);
  }
  
  // Find the largest angle difference
  G4double maxDiff = angleDiffs[0];
  size_t maxDiffIdx = 0;
  for (size_t i = 1; i < angleDiffs.size(); i++) {
    if (angleDiffs[i] > maxDiff) {
      maxDiff = angleDiffs[i];
      maxDiffIdx = i;
    }
  }
  
  // Calculate alpha as 2π minus the largest difference (matching Python implementation)
  G4double alpha = 2.0 * CLHEP::pi - maxDiff;
  
  // Get corner indices that define the alpha angle
  G4int corner1Idx = angles[maxDiffIdx].cornerIndex;
  G4int corner2Idx = angles[(maxDiffIdx + 1) % angles.size()].cornerIndex;
  
  // Determine if this is a same side case or adjacent sides case
  // This information could be used for further analysis (not returning it now)
  G4int pointType = 0;
  
  // Same side pairs are (0,1), (1,2), (2,3), or (3,0)
  std::vector<std::pair<G4int, G4int>> sameSidePairs = {{0,1}, {1,2}, {2,3}, {0,3}};
  std::pair<G4int, G4int> cornerPair;
  if (corner1Idx < corner2Idx) {
    cornerPair = {corner1Idx, corner2Idx};
  } else {
    cornerPair = {corner2Idx, corner1Idx};
  }
  
  bool isSameSide = false;
  for (const auto& pair : sameSidePairs) {
    if (pair.first == cornerPair.first && pair.second == cornerPair.second) {
      isSameSide = true;
      break;
    }
  }
  
  pointType = isSameSide ? 1 : 2;
  
  // Convert to degrees for storage
  G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
  
  return alphaInDegrees;
}

// Calculate angles from hit position to all pixels in a 9x9 grid around the hit pixel
void EventAction::CalculateNeighborhoodGridAngles(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ)
{
  // Clear previous data
  fGridNeighborhoodAngles.clear();
  fGridNeighborhoodPixelI.clear();
  fGridNeighborhoodPixelJ.clear();
  
  // Check if hit is inside a pixel - if so, all angles should be invalid
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    // Fill all 81 positions with NaN for inside-pixel hits
    for (G4int di = -4; di <= 4; di++) {
      for (G4int dj = -4; dj <= 4; dj++) {
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
  
  // Define the 9x9 grid: 4 pixels in each direction from the center
  for (G4int di = -4; di <= 4; di++) {
    for (G4int dj = -4; dj <= 4; dj++) {
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
// This matches the algorithm used in the Python alpha_demo.py
G4double EventAction::CalculatePixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calculate the four corners of the pixel
  G4double halfWidth = pixelWidth / 2.0;
  G4double halfHeight = pixelHeight / 2.0;
  
  G4ThreeVector corners[4];
  // Bottom-left (0)
  corners[0] = G4ThreeVector(pixelCenterX - halfWidth, pixelCenterY - halfHeight, 0);
  // Bottom-right (1)
  corners[1] = G4ThreeVector(pixelCenterX + halfWidth, pixelCenterY - halfHeight, 0);
  // Top-right (2)
  corners[2] = G4ThreeVector(pixelCenterX + halfWidth, pixelCenterY + halfHeight, 0);
  // Top-left (3)
  corners[3] = G4ThreeVector(pixelCenterX - halfWidth, pixelCenterY + halfHeight, 0);
  
  // Calculate angles to each corner from the hit point (2D only - XY plane)
  struct AngleInfo {
    G4double angle;
    G4int cornerIndex;
    
    // Custom comparison operator for sorting
    bool operator<(const AngleInfo& other) const {
      return angle < other.angle;
    }
  };
  
  std::vector<AngleInfo> angles;
  
  for (G4int i = 0; i < 4; i++) {
    // Calculate relative position vector from hit to corner (2D only)
    G4double dx = corners[i].x() - hitX;
    G4double dy = corners[i].y() - hitY;
    
    // Calculate angle using atan2 (this gives angle in XY plane)
    G4double angle = std::atan2(dy, dx);
    angles.push_back({angle, i});
  }
  
  // Sort by angle
  std::sort(angles.begin(), angles.end());
  
  // Calculate differences between consecutive angles
  std::vector<G4double> angleDiffs;
  for (size_t i = 0; i < angles.size(); i++) {
    size_t nextI = (i + 1) % angles.size();
    G4double diff = angles[nextI].angle - angles[i].angle;
    
    // Handle wrap-around (angles close to 2π)
    if (diff < 0) {
      diff += 2.0 * CLHEP::pi;
    }
    
    angleDiffs.push_back(diff);
  }
  
  // Find the largest angle difference
  G4double maxDiff = angleDiffs[0];
  for (size_t i = 1; i < angleDiffs.size(); i++) {
    if (angleDiffs[i] > maxDiff) {
      maxDiff = angleDiffs[i];
    }
  }
  
  // Calculate alpha as 2π minus the largest difference
  // This gives the angular size subtended by the pixel as seen from the hit point
  G4double alpha = 2.0 * CLHEP::pi - maxDiff;
  
  return alpha; // Return in radians
}

// Calculate charge sharing for pixels in a 9x9 grid around the hit pixel
void EventAction::CalculateNeighborhoodChargeSharing()
{
  // Clear previous data
  fGridNeighborhoodChargeFractions.clear();
  fGridNeighborhoodDistances.clear();
  fGridNeighborhoodChargeValues.clear();
  fGridNeighborhoodChargeCoulombs.clear();
  
  // Check if no energy was deposited
  if (fEdep <= 0) {
    // Fill all 81 positions with zero for no-energy events
    for (G4int di = -4; di <= 4; di++) {
      for (G4int dj = -4; dj <= 4; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        fGridNeighborhoodChargeFractions.push_back(0.0);
        fGridNeighborhoodDistances.push_back(-999.0);
        fGridNeighborhoodChargeValues.push_back(0.0);
        fGridNeighborhoodChargeCoulombs.push_back(0.0);
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
    
    // Fill all 81 positions, giving all charge to the hit pixel and zero to others
    for (G4int di = -4; di <= 4; di++) {
      for (G4int dj = -4; dj <= 4; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this is the pixel that was hit
        if (di == 0 && dj == 0) {
          // This is the center pixel (the one that was hit)
          fGridNeighborhoodChargeFractions.push_back(1.0);
          fGridNeighborhoodChargeValues.push_back(totalCharge);
          fGridNeighborhoodChargeCoulombs.push_back(totalCharge * fElementaryCharge);
          fGridNeighborhoodDistances.push_back(0.0); // Distance to center of hit pixel is effectively zero
        } else if (gridPixelI >= 0 && gridPixelI < fDetector->GetNumBlocksPerSide() && 
                   gridPixelJ >= 0 && gridPixelJ < fDetector->GetNumBlocksPerSide()) {
          // This is a valid pixel in the detector but not the hit pixel
          fGridNeighborhoodChargeFractions.push_back(0.0);
          fGridNeighborhoodChargeValues.push_back(0.0);
          fGridNeighborhoodChargeCoulombs.push_back(0.0);
          
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
          fGridNeighborhoodChargeValues.push_back(0.0);
          fGridNeighborhoodChargeCoulombs.push_back(0.0);
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
    
    // Fill all 81 positions, giving all charge to the identified pixel (fPixelIndexI, fPixelIndexJ) and zero to others
    for (G4int di = -4; di <= 4; di++) {
      for (G4int dj = -4; dj <= 4; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this pixel is within bounds
        if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
            gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
          // Store invalid data for out-of-bounds pixels
          fGridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fGridNeighborhoodDistances.push_back(-999.0);
          fGridNeighborhoodChargeValues.push_back(0.0);
          fGridNeighborhoodChargeCoulombs.push_back(0.0);
          continue;
        }
        
        // Calculate the center position and distance for this pixel
        G4double currentPixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
        G4double currentPixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
        G4double distance = std::sqrt(std::pow(fPosition.x() - currentPixelCenterX, 2) + 
                                     std::pow(fPosition.y() - currentPixelCenterY, 2));
        
        // Check if this is the identified pixel (center of the 9x9 grid)
        if (di == 0 && dj == 0) {
          // This is the identified pixel (fPixelIndexI, fPixelIndexJ) - assign all charge here
          fGridNeighborhoodChargeFractions.push_back(1.0);
          fGridNeighborhoodChargeValues.push_back(totalChargeValue);
          fGridNeighborhoodChargeCoulombs.push_back(totalChargeValue * fElementaryCharge);
          fGridNeighborhoodDistances.push_back(distance);
        } else {
          // All other pixels get zero charge
          fGridNeighborhoodChargeFractions.push_back(0.0);
          fGridNeighborhoodChargeValues.push_back(0.0);
          fGridNeighborhoodChargeCoulombs.push_back(0.0);
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
  
  // Define the 9x9 grid: 4 pixels in each direction from the center
  for (G4int di = -4; di <= 4; di++) {
    for (G4int dj = -4; dj <= 4; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fGridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
        fGridNeighborhoodDistances.push_back(-999.0);
        fGridNeighborhoodChargeValues.push_back(0.0);
        fGridNeighborhoodChargeCoulombs.push_back(0.0);
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
        weight = alpha * 1000.0; // Large weight for very close pixels
      } else {
        // Distance is zero (hit exactly on pixel center), give maximum weight
        weight = alpha * 10000.0;
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
  for (G4int di = -4; di <= 4; di++) {
    for (G4int dj = -4; dj <= 4; dj++) {
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
      fGridNeighborhoodChargeValues.push_back(chargeValue);
      fGridNeighborhoodChargeCoulombs.push_back(chargeValue * fElementaryCharge);
      
      validIndex++;
    }
  }
}

// Implementation of new setter methods for additional information
void EventAction::SetTimingInfo(G4double globalTime, G4double localTime, G4double properTime)
{
  // Only set timing on first call (first energy depositing step)
  if (fGlobalTime == 0.) {
    fGlobalTime = globalTime;
    fLocalTime = localTime;
    fProperTime = properTime;
  }
}

void EventAction::SetPhysicsProcessInfo(const G4String& processName, G4int trackID, G4int parentID, 
                                       G4int stepCount, G4double stepLength)
{
  // Store the dominant physics process (first one or most significant)
  if (fPhysicsProcess.empty()) {
    fPhysicsProcess = processName;
  }
  
  fTrackID = trackID;
  fParentID = parentID;
  fStepCount = stepCount;
  fTotalStepLength += stepLength;
}

void EventAction::AddTrajectoryPoint(G4double x, G4double y, G4double z, G4double time)
{
  fTrajectoryX.push_back(x);
  fTrajectoryY.push_back(y);
  fTrajectoryZ.push_back(z);
  fTrajectoryTime.push_back(time);
}

void EventAction::SetFinalParticleEnergy(G4double finalEnergy)
{
  fFinalParticleEnergy = finalEnergy;
}

void EventAction::AddStepEnergyDeposition(G4double edep, G4double z, G4double time, 
                                         G4double stepLength, G4int stepCount)
{
  // Only store steps that actually deposit energy
  if (edep > 0.) {
    fStepEdepVec.push_back(edep);
    fStepZVec.push_back(z);
    fStepTimeVec.push_back(time / CLHEP::ns); // Convert to ns
    fStepLenVec.push_back(stepLength);
    fStepNumVec.push_back(stepCount);
  }
  
}

void EventAction::AddAllStepInfo(G4double edep, G4double z, G4double time, 
                                G4double stepLength, G4int stepCount)
{
  // Store ALL steps, including those with zero energy deposition
  fAllStepEdepVec.push_back(edep);
  fAllStepZVec.push_back(z);
  fAllStepTimeVec.push_back(time / CLHEP::ns); // Convert to ns
  fAllStepLenVec.push_back(stepLength);
  fAllStepNumVec.push_back(stepCount);
}