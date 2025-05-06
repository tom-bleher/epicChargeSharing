#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4SystemOfUnits.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* detector)
: fDetector(detector)
{
    // Create directories for commands
    fEpicDirectory = new G4UIdirectory("/epicToy/");
    fEpicDirectory->SetGuidance("UI commands for the epicToy application");
    
    fDetDirectory = new G4UIdirectory("/epicToy/detector/");
    fDetDirectory->SetGuidance("Detector configuration commands");
    
    // Create commands for detector configuration
    fBlockSizeCmd = new G4UIcmdWithADoubleAndUnit("/epicToy/detector/setBlockSize", this);
    fBlockSizeCmd->SetGuidance("Set the size of each detector block");
    fBlockSizeCmd->SetParameterName("Size", false);
    fBlockSizeCmd->SetUnitCategory("Length");
    fBlockSizeCmd->SetRange("Size>0.");
    fBlockSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fBlockSpacingCmd = new G4UIcmdWithADoubleAndUnit("/epicToy/detector/setBlockSpacing", this);
    fBlockSpacingCmd->SetGuidance("Set the spacing between blocks");
    fBlockSpacingCmd->SetParameterName("Spacing", false);
    fBlockSpacingCmd->SetUnitCategory("Length");
    fBlockSpacingCmd->SetRange("Spacing>=0.");
    fBlockSpacingCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fCornerOffsetCmd = new G4UIcmdWithADoubleAndUnit("/epicToy/detector/setCornerOffset", this);
    fCornerOffsetCmd->SetGuidance("Set the offset from the corner");
    fCornerOffsetCmd->SetParameterName("Offset", false);
    fCornerOffsetCmd->SetUnitCategory("Length");
    fCornerOffsetCmd->SetRange("Offset>=0.");
    fCornerOffsetCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
}

DetectorMessenger::~DetectorMessenger()
{
    delete fBlockSizeCmd;
    delete fBlockSpacingCmd;
    delete fCornerOffsetCmd;
    delete fNumBlocksCmd;
    delete fDetDirectory;
    delete fEpicDirectory;
}

void DetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    // Get the current parameters from detector
    G4double currentSize = 100*um; // Default values, could be retrieved from detector if needed
    G4double currentSpacing = 500*um;
    G4double currentOffset = 1*um;
    G4int currentNumBlocks = 4;
    
    if (command == fBlockSizeCmd) {
        // Only update the size parameter
        G4double newSize = fBlockSizeCmd->GetNewDoubleValue(newValue);
        fDetector->SetGridParameters(newSize, currentSpacing, currentOffset, currentNumBlocks);
    }
    else if (command == fBlockSpacingCmd) {
        // Only update the spacing parameter
        G4double newSpacing = fBlockSpacingCmd->GetNewDoubleValue(newValue);
        fDetector->SetGridParameters(currentSize, newSpacing, currentOffset, currentNumBlocks);
    }
    else if (command == fCornerOffsetCmd) {
        // Only update the offset parameter
        G4double newOffset = fCornerOffsetCmd->GetNewDoubleValue(newValue);
        fDetector->SetGridParameters(currentSize, currentSpacing, newOffset, currentNumBlocks);
    }
    else if (command == fNumBlocksCmd) {
        // Only update the number of blocks parameter
        G4int newNumBlocks = fNumBlocksCmd->GetNewIntValue(newValue);
        fDetector->SetGridParameters(currentSize, currentSpacing, currentOffset, newNumBlocks);
    }
}