/**
 * @file ActionInitialization.cc
 * @brief Wires together user actions: primary generator, run/event/stepping actions.
 */
#include "ActionInitialization.hh"
#include "PrimaryGenerator.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "DetectorConstruction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* detector)
: fDetector(detector)
{
}

void ActionInitialization::BuildForMaster() const
{
    SetUserAction(CreateRunAction());
}

void ActionInitialization::Build() const
{
    SetUserAction(new PrimaryGenerator(fDetector));

    RunAction* runAction = CreateRunAction();
    SetUserAction(runAction);

    EventAction* eventAction = new EventAction(runAction, fDetector);
    SetUserAction(eventAction);

    if (fDetector) {
        fDetector->SetEventAction(eventAction);
        eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    }

    SteppingAction* steppingAction = new SteppingAction(eventAction);
    SetUserAction(steppingAction);

    eventAction->SetSteppingAction(steppingAction);
}

RunAction* ActionInitialization::CreateRunAction() const
{
    RunAction* runAction = new RunAction();
    if (!fDetector) {
        return runAction;
    }

    fDetector->SetRunAction(runAction);
    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(),
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide());
    runAction->SetNeighborhoodRadiusMeta(fDetector->GetNeighborhoodRadius());
    return runAction;
}