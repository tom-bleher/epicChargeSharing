#include "SteppingAction.hh"
#include "EventAction.hh"
#include "SimulationLogger.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"

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
      
      // Log pixel hit information to SimulationLogger
      SimulationLogger* logger = SimulationLogger::GetInstance();
      if (logger) {
        // Get current event ID
        const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
        G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
        
        // Calculate step length
        G4double stepLength = step->GetStepLength();
        
        // For now, use placeholder pixel indices (they will be calculated properly in EventAction)
        // The real pixel hit determination happens in EventAction::EndOfEventAction
        G4int pixelI = -1;  // Will be determined later
        G4int pixelJ = -1;  // Will be determined later
        
        // Log this step as a hit (individual energy deposition)
        logger->LogPixelHit(eventID, pixelI, pixelJ, edep, stepPosition, stepLength);
      }
    }
    // Note: Removed excessive debug output for "outside detector" cases
  }
}