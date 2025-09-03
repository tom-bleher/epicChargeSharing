#ifndef ACTIONINITIALIZATION_HH
#define ACTIONINITIALIZATION_HH

#include "G4VUserActionInitialization.hh"

class PrimaryGenerator;
class RunAction;
class EventAction;
class SteppingAction;
class DetectorConstruction;

class ActionInitialization : public G4VUserActionInitialization
{
public:
    explicit ActionInitialization(DetectorConstruction* detector);
    ~ActionInitialization() override;

    void BuildForMaster() const override;
    void Build() const override;

private:
    DetectorConstruction* fDetector;
};

#endif