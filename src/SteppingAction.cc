#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"

SteppingAction::SteppingAction(EventAction* eventAction, DetectorConstruction* detector)
: G4UserSteppingAction(),
  fEventAction(eventAction),
  fDetector(detector),
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
  // Volume tracking
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();

  // Logical volume names (defensive defaults)
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  G4String preVolName  = preVol  ? preVol->GetLogicalVolume()->GetName()  : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";

  // First volume entered (fGeomBoundary)
  if (postPoint->GetStepStatus() == fGeomBoundary) {
    // We have crossed a boundary and entered postVol
    if (fFirstContactVolume == "NONE" && postVol) {
      const G4String enteredName = postVol->GetLogicalVolume()->GetName();

      // Only for detector-relevant volumes
      if (enteredName == "logicBlock" || enteredName == "logicCube") {
        fFirstContactVolume = enteredName;

        // Register first-contact position
        if (fEventAction) {
          fEventAction->RegisterFirstContact(postPoint->GetPosition());
        }
      }
    }
  }
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Track first-contact volume
  TrackVolumeInteractions(step);
}

