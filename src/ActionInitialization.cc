#include "ActionInitialization.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* detector)
: fDetector(detector)
{
}

ActionInitialization::~ActionInitialization()
{
}

void ActionInitialization::BuildForMaster() const 
{
    // Create RunAction for the master thread
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);

    // Set detector grid parameters in RunAction for saving to ROOT metadata
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
    // Create and register the primary generator with detector information
    PrimaryGenerator *generator = new PrimaryGenerator(fDetector);
    SetUserAction(generator);
    
    // Create and register RunAction
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);
    
    // Set detector grid parameters in RunAction for saving to ROOT metadata
    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(), 
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide()
    );
    
    // Create and register EventAction with detector information
    EventAction* eventAction = new EventAction(runAction, fDetector);
    // Don't set initial position here - it will be updated for each event
    SetUserAction(eventAction);
    
    // Connect EventAction and DetectorConstruction bidirectionally
    fDetector->SetEventAction(eventAction);
    
    // Set the current neighborhood radius in EventAction
    eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    
    // Create and register SteppingAction
    SteppingAction* steppingAction = new SteppingAction(eventAction, fDetector);
    SetUserAction(steppingAction);
    
    // Connect SteppingAction to EventAction for aluminum interaction tracking
    eventAction->SetSteppingAction(steppingAction);
}