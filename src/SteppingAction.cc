#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"

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
  
  // Skip steps with no energy deposit
  if (edep <= 0.) return;
  
  // Get position of the energy deposit (middle of step)
  G4ThreeVector position = 0.5 * (step->GetPreStepPoint()->GetPosition() +
                                 step->GetPostStepPoint()->GetPosition());
  
  // Add this energy deposit to the event
  fEventAction->AddEdep(edep, position);
}