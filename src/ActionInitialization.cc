/**
 * @file ActionInitialization.cc
 * @brief Wires together user actions: primary generator, run/event/stepping actions.
 */
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "PrimaryGenerator.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* detector) : fDetector(detector) {}

void ActionInitialization::BuildForMaster() const {
    SetUserAction(CreateRunAction());
}

void ActionInitialization::Build() const {
    SetUserAction(new PrimaryGenerator(fDetector));

    RunAction* runAction = CreateRunAction();
    SetUserAction(runAction);

    auto* eventAction = new EventAction(runAction, fDetector);
    SetUserAction(eventAction);

    if (fDetector) {
        fDetector->SetEventAction(eventAction);
        eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    }

    auto* steppingAction = new SteppingAction(eventAction);
    SetUserAction(steppingAction);

    eventAction->SetSteppingAction(steppingAction);
}

RunAction* ActionInitialization::CreateRunAction() const {
    auto* runAction = new RunAction();
    if (!fDetector) {
        return runAction;
    }

    fDetector->SetRunAction(runAction);
    runAction->SetDetectorGridParameters(fDetector->GetPixelSize(), fDetector->GetPixelSpacing(),
                                         fDetector->GetGridOffset(), fDetector->GetDetSize(),
                                         fDetector->GetNumBlocksPerSide());
    runAction->SetNeighborhoodRadiusMeta(fDetector->GetNeighborhoodRadius());
    return runAction;
}