#include "ActionInitialization.hh"


ActionInitialization::ActionInitialization()
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
    // Create and register the primary generator
    PrimaryGenerator *generator = new PrimaryGenerator();
    SetUserAction(generator);
    
    // Create and register RunAction
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);
    
    // Create and register EventAction
    EventAction* eventAction = new EventAction(runAction);
    SetUserAction(eventAction);
    
    // Create and register SteppingAction
    SetUserAction(new SteppingAction(eventAction));
}