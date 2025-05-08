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
    G4UIExecutive *ui = new G4UIExecutive(argc, argv);

    #ifdef G4MULTITHREADED
        G4MTRunManager *runManager = new G4MTRunManager;
        // Reduce number of threads to 2 to reduce memory pressure
        G4int nThreads = 5; // Lower from 4 to reduce memory pressure
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

    G4VisManager *visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager *uiManager = G4UImanager::GetUIpointer();

    // Execute the visualization macro
    // Use the macro from macros directory with fallback
    G4String command = "/control/execute macros/vis.mac";
    G4int status = uiManager->ApplyCommand(command);
    if (status != 0) {
        uiManager->ApplyCommand("/control/execute vis.mac");
    }

    ui->SessionStart();

    delete ui;
    delete visManager;
    delete runManager;

    return 0;
}

