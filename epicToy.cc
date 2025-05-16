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
        if (arg == "-m" || arg == "--macro") {
            isBatch = true;
            if (i + 1 < argc) {
                macroFile = argv[++i];
            }
        }
    }
    
    // Only create UI executive if we're not in batch mode
    G4UIExecutive *ui = nullptr;
    if (!isBatch) {
        ui = new G4UIExecutive(argc, argv);
    }

    #ifdef G4MULTITHREADED
        G4MTRunManager *runManager = new G4MTRunManager;
        // Use all available cores on the machine by default
        G4int nThreads = G4Threading::G4GetNumberOfCores();
        runManager->SetNumberOfThreads(nThreads);
        G4cout << "Running in multithreaded mode with " << nThreads << " threads" << G4endl;
    #else
        G4RunManager *runManager = new G4RunManager;
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
    
    // Handle visualization only in interactive mode
    G4VisManager *visManager = nullptr;
    if (!isBatch) {
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
        // Batch mode - execute the specified macro
        G4String command = "/control/execute ";
        command += macroFile;
        uiManager->ApplyCommand(command);
    }
    
    // Clean up
    if (visManager) delete visManager;
    delete runManager;

    return 0;
}

