/**
 * @file SteppingAction.cc
 * @brief Records first-contact volume transitions and forwards positions to `EventAction`.
 */
#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"

SteppingAction::SteppingAction(EventAction* eventAction)
: G4UserSteppingAction(),
  fEventAction(eventAction),
  fFirstContactVolume("NONE")
{}

void SteppingAction::Reset()
{
    fFirstContactVolume = "NONE";
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step)
{
  if (fFirstContactVolume != "NONE") {
    return; // already recorded first-contact this event
  }

  G4StepPoint* postPoint = step->GetPostStepPoint();
  if (!postPoint || postPoint->GetStepStatus() != fGeomBoundary) {
    return; // only care about boundary crossings
  }

  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  if (!postVol) {
    return;
  }

  const G4String enteredName = postVol->GetLogicalVolume()->GetName();
  if (enteredName == "logicBlock" || enteredName == "logicCube") {
    fFirstContactVolume = enteredName;
    if (fEventAction) {
      fEventAction->RegisterFirstContact(postPoint->GetPosition());
    }
  }
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  TrackVolumeInteractions(step);
}

