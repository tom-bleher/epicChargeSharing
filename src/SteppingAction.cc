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

SteppingAction::~SteppingAction()
{}

void SteppingAction::Reset()
{
    fFirstContactVolume = "NONE";
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step)
{
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();

  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  G4String preVolName  = preVol  ? preVol->GetLogicalVolume()->GetName()  : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";

  if (postPoint->GetStepStatus() == fGeomBoundary) {
    if (fFirstContactVolume == "NONE" && postVol) {
      const G4String enteredName = postVol->GetLogicalVolume()->GetName();

      if (enteredName == "logicBlock" || enteredName == "logicCube") {
        fFirstContactVolume = enteredName;

        if (fEventAction) {
          fEventAction->RegisterFirstContact(postPoint->GetPosition());
        }
      }
    }
  }
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  TrackVolumeInteractions(step);
}

