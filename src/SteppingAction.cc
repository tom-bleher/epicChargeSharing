#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"

SteppingAction::SteppingAction(EventAction* eventAction)
: G4UserSteppingAction(),
  fEventAction(eventAction)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Collect energy deposited in this step while particle travels through detector
  G4double edep = step->GetTotalEnergyDeposit();
  
  // Get step information for tracking
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Calculate position of this step (middle of step) - CONSISTENT with AddEdep calculation
  G4ThreeVector stepPosition = 0.5 * (prePoint->GetPosition() + postPoint->GetPosition());
  
  // Check if energy was deposited
  if (edep > 0) {
    // Get volume names for debugging
    G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
    G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
    
    G4String preVolName = "UNKNOWN";
    G4String postVolName = "UNKNOWN";
    
    if (preVol) preVolName = preVol->GetLogicalVolume()->GetName();
    if (postVol) postVolName = postVol->GetLogicalVolume()->GetName();
    
    // Determine if we're inside the detector volume
    G4bool insideDetector = (preVolName == "logicCube" || postVolName == "logicCube");
    
    if (insideDetector) {
      // Energy deposited inside detector volume - accumulate in EventAction
      // Only energy deposited while particle travels through detector is counted
      fEventAction->AddEdep(edep, stepPosition);
    }
    // Note: Removed excessive debug output for "outside detector" cases
  }
}