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

ActionInitialization::~ActionInitialization()
{
}

void ActionInitialization::BuildForMaster() const 
{
    // RunAction for master
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);

    // Set detector grid parameters
    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(), 
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide()
    );
}

void ActionInitialization::Build() const
{
    // Primary generator
    PrimaryGenerator *generator = new PrimaryGenerator(fDetector);
    SetUserAction(generator);
    
    // RunAction
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);
    
    // Set detector grid parameters
    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(), 
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide()
    );
    
    // EventAction
    EventAction* eventAction = new EventAction(runAction, fDetector);
    // Initial position set per-event
    SetUserAction(eventAction);
    
    // Wire EventAction <-> DetectorConstruction
    fDetector->SetEventAction(eventAction);
    
    // Neighborhood radius
    eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    
    // SteppingAction
    SteppingAction* steppingAction = new SteppingAction(eventAction, fDetector);
    SetUserAction(steppingAction);
    
    // Wire SteppingAction -> EventAction
    eventAction->SetSteppingAction(steppingAction);
}