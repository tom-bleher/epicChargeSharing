#include "ActionInitialization.hh"
#include "DetectorMessenger.hh"
#include "CrashHandler.hh"

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
    
    // Register RunAction with crash recovery system (master thread)
    CrashHandler::GetInstance().RegisterRunAction(runAction);
    
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
    
    // Register RunAction with crash recovery system
    CrashHandler::GetInstance().RegisterRunAction(runAction);
    
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
    
    // Set up DetectorMessenger with EventAction for neighborhood configuration
    // Note: The DetectorMessenger is created in DetectorConstruction constructor,
    // but we need to connect it to EventAction here
    if (fDetector->GetDetectorMessenger()) {
        fDetector->GetDetectorMessenger()->SetEventAction(eventAction);
    }
    
    // Create and register SteppingAction
    SteppingAction* steppingAction = new SteppingAction(eventAction, fDetector);
    SetUserAction(steppingAction);
    
    // Connect SteppingAction to EventAction for aluminum interaction tracking
    eventAction->SetSteppingAction(steppingAction);
}