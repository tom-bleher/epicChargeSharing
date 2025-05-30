#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "EventAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4SystemOfUnits.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* detector)
: fDetector(detector), fEventAction(nullptr)
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
    fCornerOffsetCmd->SetGuidance("Set the fixed offset from detector edge to first pixel edge");
    fCornerOffsetCmd->SetGuidance("NOTE: Changing this will adjust detector size to maintain pixel grid");
    fCornerOffsetCmd->SetParameterName("Offset", false);
    fCornerOffsetCmd->SetUnitCategory("Length");
    fCornerOffsetCmd->SetRange("Offset>=0.");
    fCornerOffsetCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fNeighborhoodRadiusCmd = new G4UIcmdWithAnInteger("/epicToy/detector/setNeighborhoodRadius", this);
    fNeighborhoodRadiusCmd->SetGuidance("Set the neighborhood radius for charge sharing analysis");
    fNeighborhoodRadiusCmd->SetGuidance("Radius 4 = 9x9 grid, Radius 3 = 7x7 grid, Radius 2 = 5x5 grid, etc.");
    fNeighborhoodRadiusCmd->SetParameterName("Radius", false);
    fNeighborhoodRadiusCmd->SetRange("Radius>=1");
    fNeighborhoodRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
}

DetectorMessenger::~DetectorMessenger()
{
    delete fBlockSizeCmd;
    delete fBlockSpacingCmd;
    delete fCornerOffsetCmd;
    delete fNumBlocksCmd;
    delete fNeighborhoodRadiusCmd;
    delete fDetDirectory;
    delete fEpicDirectory;
}

void DetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    // Get the current parameters from detector
    G4double currentSize = fDetector->GetPixelSize();
    G4double currentSpacing = fDetector->GetPixelSpacing();
    G4double currentOffset = fDetector->GetPixelCornerOffset();
    G4int currentNumBlocks = fDetector->GetNumBlocksPerSide();
    
    if (command == fBlockSizeCmd) {
        // Update the size parameter
        G4double newSize = fBlockSizeCmd->GetNewDoubleValue(newValue);
        G4cout << "Setting pixel size to: " << newSize/um << " μm" << G4endl;
        fDetector->SetGridParameters(newSize, currentSpacing, currentOffset, currentNumBlocks);
    }
    else if (command == fBlockSpacingCmd) {
        // Update the spacing parameter
        G4double newSpacing = fBlockSpacingCmd->GetNewDoubleValue(newValue);
        G4cout << "Setting pixel spacing to: " << newSpacing/um << " μm" << G4endl;
        fDetector->SetGridParameters(currentSize, newSpacing, currentOffset, currentNumBlocks);
    }
    else if (command == fCornerOffsetCmd) {
        // Update the offset parameter (this may cause detector size adjustment)
        G4double newOffset = fCornerOffsetCmd->GetNewDoubleValue(newValue);
        fDetector->SetPixelCornerOffset(newOffset);
    }
    else if (command == fNumBlocksCmd) {
        // Number of blocks is now calculated automatically, warn user
        G4cerr << "WARNING: Number of blocks is now calculated automatically based on pixel size, spacing, and detector size." << G4endl;
        G4cerr << "This parameter is read-only." << G4endl;
    }
    else if (command == fNeighborhoodRadiusCmd) {
        // Update the neighborhood radius parameter
        G4int newRadius = fNeighborhoodRadiusCmd->GetNewIntValue(newValue);
        G4cout << "Setting neighborhood radius to: " << newRadius << G4endl;
        fDetector->SetNeighborhoodRadius(newRadius);
    }
}