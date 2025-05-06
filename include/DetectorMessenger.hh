#ifndef DetectorMessenger_h
#define DetectorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class DetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAnInteger;

class DetectorMessenger : public G4UImessenger
{
public:
    DetectorMessenger(DetectorConstruction* detector);
    virtual ~DetectorMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
private:
    DetectorConstruction* fDetector;
    
    G4UIdirectory* fEpicDirectory;
    G4UIdirectory* fDetDirectory;
    
    G4UIcmdWithADoubleAndUnit* fBlockSizeCmd;
    G4UIcmdWithADoubleAndUnit* fBlockSpacingCmd;
    G4UIcmdWithADoubleAndUnit* fCornerOffsetCmd;
    G4UIcmdWithAnInteger*      fNumBlocksCmd;
};

#endif