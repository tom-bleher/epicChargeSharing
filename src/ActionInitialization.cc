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
    SetUserAction(new RunAction());
}


void ActionInitialization::Build() const
{
    // Create and register the primary generator with detector information
    PrimaryGenerator *generator = new PrimaryGenerator(fDetector);
    SetUserAction(generator);
    
    // Create and register RunAction
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);
    
    // Create and register EventAction with detector information
    EventAction* eventAction = new EventAction(runAction, fDetector);
    // Don't set initial position here - it will be set for each event
    SetUserAction(eventAction);
    
    // Create and register SteppingAction
    SetUserAction(new SteppingAction(eventAction));
}