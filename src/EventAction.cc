#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

EventAction::EventAction(RunAction* runAction)
: G4UserEventAction(),
  fRunAction(runAction),
  fEdep(0.),
  fPosition(G4ThreeVector(0.,0.,0.)),
  fInitialPosition(G4ThreeVector(0.,0.,0.)),
  fHasHit(false)
{ 
}

EventAction::~EventAction()
{ 
}

void EventAction::BeginOfEventAction(const G4Event*)
{
  // Reset per-event variables
  fEdep = 0.;
  fPosition = G4ThreeVector(0.,0.,0.);
  fHasHit = false;
}

void EventAction::EndOfEventAction(const G4Event*)
{
  // If there was an energy deposit in this event, store it in the ROOT file
  if (fHasHit) {
    // Store data in the RunAction (which will be saved to ROOT)
    fRunAction->SetEventData(fEdep, fPosition.x(), fPosition.y(), fPosition.z());
    // Also store the initial position
    fRunAction->SetInitialPosition(fInitialPosition.x(), fInitialPosition.y(), fInitialPosition.z());
    fRunAction->FillTree();
  }
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