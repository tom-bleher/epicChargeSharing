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
  
  // Get step information regardless of energy deposit for trajectory tracking
  G4Track* track = step->GetTrack();
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  // Store trajectory point for all steps (including non-energy depositing ones)
  G4ThreeVector position = postPoint->GetPosition();
  G4double time = postPoint->GetGlobalTime();
  fEventAction->AddTrajectoryPoint(position.x(), position.y(), position.z(), time);
  
  // Calculate position of this step (middle of step) - CONSISTENT with AddEdep calculation
  G4ThreeVector stepPosition = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                     step->GetPostStepPoint()->GetPosition());
  
  // Get step information for ALL steps
  G4double stepLength = step->GetStepLength();
  G4int stepNumber = track->GetCurrentStepNumber();
  G4double stepTime = postPoint->GetGlobalTime();
  G4double zPosition = stepPosition.z();  // Use same position calculation
  
  // Record ALL step information (including zero energy deposition)
  fEventAction->AddAllStepInfo(edep, zPosition, stepTime, stepLength, stepNumber);
  
  // Get timing information from the first step that deposits energy
  if (edep > 0.) {
    // Set timing information (EventAction will only accept first call)
    fEventAction->SetTimingInfo(postPoint->GetGlobalTime(), 
                               postPoint->GetLocalTime(), 
                               postPoint->GetProperTime());
    
    // Get physics process information
    const G4VProcess* process = postPoint->GetProcessDefinedStep();
    if (process) {
      G4String processName = process->GetProcessName();
      G4int trackID = track->GetTrackID();
      G4int parentID = track->GetParentID();
      
      fEventAction->SetPhysicsProcessInfo(processName, trackID, parentID, stepNumber, stepLength);
    }
    
    // Update final particle energy
    fEventAction->SetFinalParticleEnergy(track->GetKineticEnergy());
    
    // IMPORTANT: Use consistent position calculation for both step tracking and total Edep
    // Get position of the energy deposit (middle of step) - CONSISTENT with AddEdep calculation
    G4ThreeVector edepPosition = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                       step->GetPostStepPoint()->GetPosition());
    
    // Add step-by-step energy deposition information using SAME position as AddEdep
    fEventAction->AddStepEnergyDeposition(edep, edepPosition.z(), stepTime, stepLength, stepNumber);
    
    // Add this energy deposit to the event using the SAME position
    fEventAction->AddEdep(edep, edepPosition);
  }
}