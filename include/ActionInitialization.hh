#ifndef ACTIONINITIALIZATION_HH
#define ACTIONINITIALIZATION_HH

#include "G4VUserActionInitialization.hh"
#include "PrimaryGenerator.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "DetectorConstruction.hh"

// Initializes all user actions and wires dependencies between
// generator, run, event, and stepping actions. Uses the project
// terminology (pixel-pad, charge neighborhood, first-contact).
class ActionInitialization : public G4VUserActionInitialization
{
public:
    ActionInitialization(DetectorConstruction* detector);
    ~ActionInitialization();

    virtual void BuildForMaster() const;
    virtual void Build() const;

private:
    DetectorConstruction* fDetector;
};

#endif