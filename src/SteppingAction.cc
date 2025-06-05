#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"

SteppingAction::SteppingAction(EventAction* eventAction)
: G4UserSteppingAction(),
  fEventAction(eventAction)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Collect energy deposited in this step
  G4double edep = step->GetTotalEnergyDeposit();
  
  // Get step information for tracking
  G4Track* track = step->GetTrack();
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Calculate position of this step (middle of step) - CONSISTENT with AddEdep calculation
  G4ThreeVector stepPosition = 0.5 * (prePoint->GetPosition() + postPoint->GetPosition());
  
  // Get step information for ALL steps
  G4double stepTime = postPoint->GetGlobalTime();
  G4double zPosition = stepPosition.z();  // Use same position calculation
  
  // Debug output for first few events
  static G4int debugEventCount = 0;
  G4int currentEventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  if (currentEventID < 5) { // Debug first 5 events
    if (currentEventID > debugEventCount) {
      debugEventCount = currentEventID;
      G4cout << "=== DEBUGGING EVENT " << currentEventID << " ===" << G4endl;
    }
    
    G4cout << "Step: edep=" << edep/MeV << " MeV, pos=(" 
           << stepPosition.x()/mm << "," << stepPosition.y()/mm << "," << stepPosition.z()/mm << ") mm";
  }
  
  // ENERGY DEPOSITION LOGIC FIX:
  // Only count energy deposition if the particle is traveling INSIDE the detector
  
  // Check if step is inside detector volume by checking logical volume names
  G4bool stepInsideDetector = false;
  G4String preVolumeName = "UNKNOWN";
  G4String postVolumeName = "UNKNOWN";
  
  try {
    G4VPhysicalVolume* preVolPhys = prePoint->GetTouchableHandle()->GetVolume();
    G4VPhysicalVolume* postVolPhys = postPoint->GetTouchableHandle()->GetVolume();
    
    if (preVolPhys && postVolPhys) {
      G4LogicalVolume* preVolume = preVolPhys->GetLogicalVolume();
      G4LogicalVolume* postVolume = postVolPhys->GetLogicalVolume();
      
      if (preVolume && postVolume) {
        preVolumeName = preVolume->GetName();
        postVolumeName = postVolume->GetName();
        
        // Only consider energy deposition if the step is inside the detector or pixel volumes
        stepInsideDetector = (preVolumeName == "logicCube") ||    // Main detector volume
                             (postVolumeName == "logicCube") ||   // Main detector volume  
                             (preVolumeName == "logicBlock") ||   // Pixel volumes
                             (postVolumeName == "logicBlock");    // Pixel volumes
      }
    }
  } catch (...) {
    // If any error occurs in volume checking, assume outside detector
    stepInsideDetector = false;
  }
  
  // Debug output continuation
  if (currentEventID < 5) {
    G4cout << ", volumes: " << preVolumeName << "->" << postVolumeName 
           << ", insideDetector=" << stepInsideDetector << G4endl;
  }
  
  // Get timing information from the first step that deposits energy INSIDE the detector
  if (edep > 0. && stepInsideDetector) {
    
    // IMPORTANT: Use consistent position calculation for both step tracking and total Edep
    // Get position of the energy deposit (middle of step) - CONSISTENT with AddEdep calculation
    G4ThreeVector edepPosition = 0.5 * (prePoint->GetPosition() + postPoint->GetPosition());
    
    // Add this energy deposit to the event using the SAME position - ONLY if inside detector
    fEventAction->AddEdep(edep, edepPosition);
    
    if (currentEventID < 5) {
      G4cout << "*** ENERGY DEPOSITED: " << edep/MeV << " MeV at position " 
             << edepPosition.x()/mm << ", " << edepPosition.y()/mm << ", " << edepPosition.z()/mm << " mm ***" << G4endl;
    }
  } else if (edep > 0. && !stepInsideDetector) {
    // Energy deposited outside detector - do not count towards total
    if (currentEventID < 5) {
      G4cout << "Energy deposited OUTSIDE detector (ignored): " << edep/MeV << " MeV" << G4endl;
    }
  }
}