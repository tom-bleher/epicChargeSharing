#include <iostream>

#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"

#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

int main(int argc, char** argv)
{
    // Check if we're running in batch mode
    G4bool isBatch = false;
    G4String macroFile = "";
    
    // Parse command line arguments
    for (G4int i = 1; i < argc; i++) {
        G4String arg = argv[i];
        if (arg == "-m" || arg == "--macro" || arg == "batch") {
            isBatch = true;
            if (i + 1 < argc && arg != "batch") {
                macroFile = argv[++i];
            } else if (arg == "batch" && i + 1 < argc) {
                // Handle the case when "batch" is directly used as an argument
                macroFile = argv[++i];
            }
        }
    }
    
    // Only create UI executive if we're not in batch mode
    G4UIExecutive *ui = nullptr;
    if (!isBatch) {
        ui = new G4UIExecutive(argc, argv, "Qt");
    }

    // Create the appropriate run manager
    G4RunManager* runManager = nullptr;
    
    #ifdef G4MULTITHREADED
    if (!isBatch) {
        // Use multithreaded mode only for interactive sessions
        G4MTRunManager* mtRunManager = new G4MTRunManager;
        // Use all available cores on the machine by default
        G4int nThreads = G4Threading::G4GetNumberOfCores();
        mtRunManager->SetNumberOfThreads(nThreads);
        G4cout << "Running in multithreaded mode with " << nThreads << " threads" << G4endl;
        runManager = mtRunManager;
    } else {
        // Use single-threaded mode for batch processing
        runManager = new G4RunManager;
        G4cout << "Running in batch mode (single-threaded)" << G4endl;
    }
    #else
        runManager = new G4RunManager;
        G4cout << "Running in single-threaded mode" << G4endl;
    #endif
    
    // Physics List
    runManager->SetUserInitialization(new PhysicsList());

    // Detector Construction
    DetectorConstruction* detConstruction = new DetectorConstruction();
    runManager->SetUserInitialization(detConstruction);

    // Action Initialization with detector construction
    runManager->SetUserInitialization(new ActionInitialization(detConstruction));

    // Get pointer to UI manager
    G4UImanager *uiManager = G4UImanager::GetUIpointer();
    
    // Initialize visualization only in interactive mode
    G4VisManager *visManager = nullptr;
    
    if (!isBatch) {
        // Only in interactive mode, set up visualization
        visManager = new G4VisExecutive();
        visManager->Initialize();
        
        // Execute the visualization macro
        // Use the macro from macros directory with fallback
        G4String command = "/control/execute macros/vis.mac";
        G4int status = uiManager->ApplyCommand(command);
        if (status != 0) {
            uiManager->ApplyCommand("/control/execute vis.mac");
        }
        
        ui->SessionStart();
        delete ui;
    } else {
        // Batch mode - execute the specified macro without visualization
        G4cout << "Running in batch mode with macro: " << macroFile << G4endl;
        G4String command = "/control/execute ";
        command += macroFile;
        uiManager->ApplyCommand(command);
    }
    
    // Clean up
    if (visManager) delete visManager;
    delete runManager;

    return 0;
}

