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
  fFirstInteractionVolume(""),
  fValidSiliconHit(false),
  fIsPixelHit(false)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::Reset()
{
    fFirstInteractionVolume = "";
    fValidSiliconHit = false;
    fIsPixelHit = false;
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    // We only care about the first step in a volume
    G4StepPoint* prePoint = step->GetPreStepPoint();
    if (prePoint->GetStepStatus() != fGeomBoundary) {
        return;
    }

    G4LogicalVolume* currentVolume = prePoint->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    G4String currentVolumeName = currentVolume->GetName();

    // Track first interaction for contamination check.
    // The first time we enter ANY volume with an energy deposit, we check it.
    if (fFirstInteractionVolume.empty() && step->GetTotalEnergyDeposit() > 0) {
        fFirstInteractionVolume = currentVolumeName;
    }
    
    // Determine if the current hit is within a pixel pad.
    fIsPixelHit = (currentVolumeName == "logicBlock");

    // Check for a valid silicon hit.
    // A hit is valid if it's in the silicon ("logicCube") and the
    // first recorded interaction was NOT in the aluminum pads.
    if (currentVolumeName == "logicCube") {
        fValidSiliconHit = (fFirstInteractionVolume != "logicBlock");
    }

    // Accumulate energy deposit only for valid silicon hits.
    // The MFD handles the primary energy recording, but we still need
    // the energy-weighted position for charge sharing calculations.
    if (fValidSiliconHit) {
        G4double edep = step->GetTotalEnergyDeposit();
        if (edep > 0) {
            G4ThreeVector stepPos = 0.5 * (prePoint->GetPosition() + step->GetPostStepPoint()->GetPosition());
            fEventAction->AddEdep(edep, stepPos);
        }
    }
}

G4bool SteppingAction::IsValidSiliconHit() const
{
  return fValidSiliconHit;
}

G4bool SteppingAction::IsPixelHit() const
{
  return fIsPixelHit;
}
