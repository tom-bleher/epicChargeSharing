#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "CrashHandler.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcommand.hh"
#include "G4SystemOfUnits.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* detector)
: fDetector(detector), fEventAction(nullptr)
{
    // Create directories for commands
    fEpicDirectory = new G4UIdirectory("/epicChargeSharing/");
    fEpicDirectory->SetGuidance("UI commands for the EpicChargeSharingAnalysis application");
    
    fDetDirectory = new G4UIdirectory("/epicChargeSharing/detector/");
    fDetDirectory->SetGuidance("Detector configuration commands");
    
    // Create commands for detector configuration
    fBlockSizeCmd = new G4UIcmdWithADoubleAndUnit("/epicChargeSharing/detector/setBlockSize", this);
    fBlockSizeCmd->SetGuidance("Set the size of each detector block");
    fBlockSizeCmd->SetParameterName("Size", false);
    fBlockSizeCmd->SetUnitCategory("Length");
    fBlockSizeCmd->SetRange("Size>0.");
    fBlockSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fBlockSpacingCmd = new G4UIcmdWithADoubleAndUnit("/epicChargeSharing/detector/setBlockSpacing", this);
    fBlockSpacingCmd->SetGuidance("Set the spacing between blocks");
    fBlockSpacingCmd->SetParameterName("Spacing", false);
    fBlockSpacingCmd->SetUnitCategory("Length");
    fBlockSpacingCmd->SetRange("Spacing>=0.");
    fBlockSpacingCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fCornerOffsetCmd = new G4UIcmdWithADoubleAndUnit("/epicChargeSharing/detector/setCornerOffset", this);
    fCornerOffsetCmd->SetGuidance("Set the fixed offset from detector edge to first pixel edge");
    fCornerOffsetCmd->SetGuidance("NOTE: Changing this will adjust detector size to maintain pixel grid");
    fCornerOffsetCmd->SetParameterName("Offset", false);
    fCornerOffsetCmd->SetUnitCategory("Length");
    fCornerOffsetCmd->SetRange("Offset>=0.");
    fCornerOffsetCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fNeighborhoodRadiusCmd = new G4UIcmdWithAnInteger("/epicChargeSharing/detector/setNeighborhoodRadius", this);
    fNeighborhoodRadiusCmd->SetGuidance("Set the neighborhood radius for charge sharing analysis");
    fNeighborhoodRadiusCmd->SetGuidance("Radius 4 = 9x9 grid, Radius 3 = 7x7 grid, Radius 2 = 5x5 grid, etc.");
    fNeighborhoodRadiusCmd->SetParameterName("Radius", false);
    fNeighborhoodRadiusCmd->SetRange("Radius>=1");
    fNeighborhoodRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    // Automatic radius selection commands
    fAutoRadiusEnabledCmd = new G4UIcmdWithABool("/epicChargeSharing/detector/setAutoRadiusEnabled", this);
    fAutoRadiusEnabledCmd->SetGuidance("Enable automatic radius selection based on fit quality");
    fAutoRadiusEnabledCmd->SetParameterName("Enabled", false);
    fAutoRadiusEnabledCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fMinAutoRadiusCmd = new G4UIcmdWithAnInteger("/epicChargeSharing/detector/setMinAutoRadius", this);
    fMinAutoRadiusCmd->SetGuidance("Set minimum radius for automatic selection");
    fMinAutoRadiusCmd->SetParameterName("MinRadius", false);
    fMinAutoRadiusCmd->SetRange("MinRadius>=1");
    fMinAutoRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fMaxAutoRadiusCmd = new G4UIcmdWithAnInteger("/epicChargeSharing/detector/setMaxAutoRadius", this);
    fMaxAutoRadiusCmd->SetGuidance("Set maximum radius for automatic selection");
    fMaxAutoRadiusCmd->SetParameterName("MaxRadius", false);
    fMaxAutoRadiusCmd->SetRange("MaxRadius>=1");
    fMaxAutoRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    // Create crash recovery commands directory
    fCrashDirectory = new G4UIdirectory("/epicChargeSharing/crash/");
    fCrashDirectory->SetGuidance("Crash recovery and auto-save commands");
    
    // Create crash recovery commands
    fCrashAutoSaveEnabledCmd = new G4UIcmdWithABool("/epicChargeSharing/crash/setAutoSaveEnabled", this);
    fCrashAutoSaveEnabledCmd->SetGuidance("Enable/disable automatic saving during simulation");
    fCrashAutoSaveEnabledCmd->SetParameterName("Enabled", false);
    fCrashAutoSaveEnabledCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fCrashAutoSaveIntervalCmd = new G4UIcmdWithAnInteger("/epicChargeSharing/crash/setAutoSaveInterval", this);
    fCrashAutoSaveIntervalCmd->SetGuidance("Set auto-save interval in number of events");
    fCrashAutoSaveIntervalCmd->SetParameterName("Interval", false);
    fCrashAutoSaveIntervalCmd->SetRange("Interval>=100");
    fCrashAutoSaveIntervalCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fCrashBackupDirectoryCmd = new G4UIcmdWithAString("/epicChargeSharing/crash/setBackupDirectory", this);
    fCrashBackupDirectoryCmd->SetGuidance("Set directory for crash recovery backups");
    fCrashBackupDirectoryCmd->SetParameterName("Directory", false);
    fCrashBackupDirectoryCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fCrashForceSaveCmd = new G4UIcommand("/epicChargeSharing/crash/forceSave", this);
    fCrashForceSaveCmd->SetGuidance("Force immediate save of current simulation data");
    fCrashForceSaveCmd->AvailableForStates(G4State_Idle);
}

DetectorMessenger::~DetectorMessenger()
{
    delete fBlockSizeCmd;
    delete fBlockSpacingCmd;
    delete fCornerOffsetCmd;
    delete fNeighborhoodRadiusCmd;
    delete fAutoRadiusEnabledCmd;
    delete fMinAutoRadiusCmd;
    delete fMaxAutoRadiusCmd;
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
    else if (command == fNeighborhoodRadiusCmd) {
        // Update the neighborhood radius parameter
        G4int newRadius = fNeighborhoodRadiusCmd->GetNewIntValue(newValue);
        G4cout << "Setting neighborhood radius to: " << newRadius << G4endl;
        fDetector->SetNeighborhoodRadius(newRadius);
    }
    else if (command == fAutoRadiusEnabledCmd) {
        // Update the automatic radius selection enabled parameter
        G4bool newEnabled = fAutoRadiusEnabledCmd->GetNewBoolValue(newValue);
        G4cout << "Setting automatic radius selection enabled: " << (newEnabled ? "Enabled" : "Disabled") << G4endl;
        fDetector->SetAutoRadiusEnabled(newEnabled);
    }
    else if (command == fMinAutoRadiusCmd) {
        // Update the minimum automatic radius parameter
        G4int newMinRadius = fMinAutoRadiusCmd->GetNewIntValue(newValue);
        G4cout << "Setting minimum automatic radius to: " << newMinRadius << G4endl;
        fDetector->SetMinAutoRadius(newMinRadius);
    }
    else if (command == fMaxAutoRadiusCmd) {
        // Update the maximum automatic radius parameter
        G4int newMaxRadius = fMaxAutoRadiusCmd->GetNewIntValue(newValue);
        G4cout << "Setting maximum automatic radius to: " << newMaxRadius << G4endl;
        fDetector->SetMaxAutoRadius(newMaxRadius);
    }
}