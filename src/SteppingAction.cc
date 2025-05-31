#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4StepPoint.hh"

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
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Calculate position of this step (middle of step) - CONSISTENT with AddEdep calculation
  G4ThreeVector stepPosition = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                     step->GetPostStepPoint()->GetPosition());
  
  // Get step information for ALL steps
  G4double stepTime = postPoint->GetGlobalTime();
  G4double zPosition = stepPosition.z();  // Use same position calculation
  
  // Record ALL step information (including zero energy deposition)
  fEventAction->AddAllStepInfo(edep, zPosition, stepTime);
  
  // Get timing information from the first step that deposits energy
  if (edep > 0.) {
    
    // Update final particle energy
    fEventAction->SetFinalParticleEnergy(track->GetKineticEnergy());
    
    // IMPORTANT: Use consistent position calculation for both step tracking and total Edep
    // Get position of the energy deposit (middle of step) - CONSISTENT with AddEdep calculation
    G4ThreeVector edepPosition = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                       step->GetPostStepPoint()->GetPosition());
    
    // Add step-by-step energy deposition information using SAME position as AddEdep
    fEventAction->AddStepEnergyDeposition(edep, edepPosition.z(), stepTime);
    
    // Add this energy deposit to the event using the SAME position
    fEventAction->AddEdep(edep, edepPosition);
  }
}