#ifndef ACTIONINITIALIZATION_HH
#define ACTIONINITIALIZATION_HH

#include "G4VUserActionInitialization.hh"

// Forward declarations to minimize header coupling
class PrimaryGenerator;
class RunAction;
class EventAction;
class SteppingAction;
class DetectorConstruction;

// Initializes user actions and wires dependencies.
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