#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <vector>
#include <algorithm>

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
  fPixelHit(false)
{ 
}

EventAction::~EventAction()
{ 
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
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // Get the primary vertex position from the event
  if (event->GetPrimaryVertex()) {
    G4ThreeVector primaryPos = event->GetPrimaryVertex()->GetPosition();
    // Update the initial position
    fInitialPosition = primaryPos;
  }
  
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
  
  // Pass pixel hit status to RunAction
  fRunAction->SetPixelHit(fPixelHit);
  
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
  
  // Calculate the position relative to the detector face (which is at z=-1.0*cm)
  G4ThreeVector relativePos = position - detectorPosition;
  
  // For the z-normal face (top/bottom), only x and y matter for pixel position
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
  G4double pixelZ = detectorPosition.z(); // Z position is the detector face
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calculate and store distance from hit to pixel center
  fPixelDistance = std::sqrt(std::pow(position.x() - pixelX, 2) + 
                            std::pow(position.y() - pixelY, 2));
  
  // Determine if the hit was on a pixel using the detector's method
  fPixelHit = fDetector->IsPositionOnPixel(position);
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Calculate the angular size of pixel from hit position
G4double EventAction::CalculatePixelAlpha(const G4ThreeVector& hitPosition, G4int pixelI, G4int pixelJ)
{
  // Check if the hit is inside the pixel. If so, set alpha to 0
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    return 0.0; // Temporarily set to zero for hits inside pixels
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