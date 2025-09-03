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
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);

    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(), 
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide()
    );
    runAction->SetNeighborhoodRadiusMeta(fDetector->GetNeighborhoodRadius());
}

void ActionInitialization::Build() const
{
    PrimaryGenerator *generator = new PrimaryGenerator(fDetector);
    SetUserAction(generator);
    
    RunAction* runAction = new RunAction();
    SetUserAction(runAction);
    
    runAction->SetDetectorGridParameters(
        fDetector->GetPixelSize(),
        fDetector->GetPixelSpacing(), 
        fDetector->GetPixelCornerOffset(),
        fDetector->GetDetSize(),
        fDetector->GetNumBlocksPerSide()
    );
    runAction->SetNeighborhoodRadiusMeta(fDetector->GetNeighborhoodRadius());
    
    EventAction* eventAction = new EventAction(runAction, fDetector);
    SetUserAction(eventAction);
    
    fDetector->SetEventAction(eventAction);
    
    eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    
    SteppingAction* steppingAction = new SteppingAction(eventAction);
    SetUserAction(steppingAction);
    
    eventAction->SetSteppingAction(steppingAction);
}