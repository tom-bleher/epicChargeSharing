#ifndef DetectorMessenger_h
#define DetectorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

class DetectorConstruction;
class G4UIdirectory;
class EventAction;

class DetectorMessenger : public G4UImessenger
{
public:
    DetectorMessenger(DetectorConstruction* detector);
    virtual ~DetectorMessenger();
    
    // Set EventAction pointer for neighborhood configuration
    void SetEventAction(EventAction* eventAction) { fEventAction = eventAction; }
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
private:
    DetectorConstruction* fDetector;
    EventAction* fEventAction;
    
    G4UIdirectory* fEpicDirectory;
    G4UIdirectory* fDetDirectory;
    
    G4UIcmdWithADoubleAndUnit* fBlockSizeCmd;
    G4UIcmdWithADoubleAndUnit* fBlockSpacingCmd;
    G4UIcmdWithADoubleAndUnit* fCornerOffsetCmd;
    G4UIcmdWithAnInteger*      fNumBlocksCmd;
    G4UIcmdWithAnInteger* fNeighborhoodRadiusCmd;
    
    // Automatic radius selection commands
    G4UIcmdWithABool* fAutoRadiusEnabledCmd;
    G4UIcmdWithAnInteger* fMinAutoRadiusCmd;
    G4UIcmdWithAnInteger* fMaxAutoRadiusCmd;
};

#endif