#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "G4Step.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"
#include "G4LogicalVolume.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"

SteppingAction::SteppingAction(EventAction* eventAction, DetectorConstruction* detector)
: G4UserSteppingAction(),
  fEventAction(eventAction),
  fDetector(detector),
  fFirstContactVolume("NONE"),
  fEverTouchedPixel(false),
  fEverTouchedSilicon(false),
  fValidSiliconHit(false)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::Reset()
{
    fFirstContactVolume = "NONE";
    fEverTouchedPixel = false;
    fEverTouchedSilicon = false;
    fValidSiliconHit = false;
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step)
{
  // Get step information for volume tracking
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();

  // Determine logical volume names (defensive defaults)
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  G4String preVolName  = preVol  ? preVol->GetLogicalVolume()->GetName()  : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";

  // Track first volume entered irrespective of energy deposit (fGeomBoundary).
  if (postPoint->GetStepStatus() == fGeomBoundary) {
    // We have crossed a boundary and entered postVol
    if (fFirstContactVolume == "NONE" && postVol) {
      const G4String enteredName = postVol->GetLogicalVolume()->GetName();

      // Only register first contact for detector-relevant volumes
      if (enteredName == "logicBlock" || enteredName == "logicCube") {
        fFirstContactVolume = enteredName;

        // Register first-contact position (use post-point at boundary entrance)
        if (fEventAction) {
          fEventAction->RegisterFirstContact(postPoint->GetPosition());
        }
      }
    }
  }

  // Sticky flags: track touched pixel-pad/silicon at any point
  if (preVolName == "logicBlock" || postVolName == "logicBlock") {
    fEverTouchedPixel = true;
  }
  if (preVolName == "logicCube" || postVolName == "logicCube") {
    fEverTouchedSilicon = true;
  }

  // Valid silicon hit iff first contact was silicon (not pixel-pad)
  if (fFirstContactVolume == "logicBlock") {
    fValidSiliconHit = false;
  } else if (fFirstContactVolume == "logicCube") {
    fValidSiliconHit = true;
  }
}

  // Volume-based detection methods (replacing geometric-only pixel tests)
G4bool SteppingAction::IsInSiliconVolume(const G4Step* step) const
{
  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();
  
  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  
  G4String preVolName = preVol ? preVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  
  return (preVolName == "logicCube" || postVolName == "logicCube");
}

// Removed ShouldAccumulateEnergy: energy accounting delegated to MFD

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Track volume interactions for aluminum contamination detection
  TrackVolumeInteractions(step);

  G4StepPoint* prePoint = step->GetPreStepPoint();
  G4StepPoint* postPoint = step->GetPostStepPoint();

  G4VPhysicalVolume* preVol = prePoint->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
  
  G4String preVolName = preVol ? preVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  G4String postVolName = postVol ? postVol->GetLogicalVolume()->GetName() : "UNKNOWN";
  
  // Track if current step touches pixel volume (aluminum)
  // Sticky flag recorded in TrackVolumeInteractions()
  
  // Accumulate position data for pure silicon hits.
  // The MFD scorer will handle the energy summation.
  G4bool isSiliconHit = (preVolName == "logicCube" || postVolName == "logicCube");

  // Always record silicon positions, even if aluminum was encountered first.
  // This ensures x_hit/y_hit and px_hit_delta_* are well-defined for ALL events.
  // Charge sharing will still be skipped later for aluminum-precontact events.
  if (isSiliconHit) {
      G4ThreeVector stepPos = 0.5 * (prePoint->GetPosition() + postPoint->GetPosition());
      fEventAction->AddSiliconPos(stepPos);
  }
}

G4bool SteppingAction::IsValidSiliconHit() const
{
  return fValidSiliconHit;
}

G4bool SteppingAction::IsPixelHit() const
{
  return fEverTouchedPixel;
}
