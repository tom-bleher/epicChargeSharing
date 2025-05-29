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
      G4int stepNumber = track->GetCurrentStepNumber();
      G4double stepLength = step->GetStepLength();
      
      fEventAction->SetPhysicsProcessInfo(processName, trackID, parentID, stepNumber, stepLength);
    }
    
    // Update final particle energy
    fEventAction->SetFinalParticleEnergy(track->GetKineticEnergy());
  }
  
  // Skip steps with no energy deposit for energy tracking
  if (edep <= 0.) return;
  
  // Get position of the energy deposit (middle of step)
  G4ThreeVector edepPosition = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                     step->GetPostStepPoint()->GetPosition());
  
  // Add this energy deposit to the event
  fEventAction->AddEdep(edep, edepPosition);
}