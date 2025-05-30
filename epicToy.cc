#include <iostream>

#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4Threading.hh"

#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

void PrintUsage() {
    G4cout << "\nUsage: ./epicToy [options] [macro_file]\n" << G4endl;
    G4cout << "Options:" << G4endl;
    G4cout << "  -m, --macro [file]     : Run in batch mode with specified macro file" << G4endl;
    G4cout << "  -t, --threads [N]      : Set number of threads (default: all available cores)" << G4endl;
    G4cout << "  --single-threaded      : Force single-threaded mode" << G4endl;
    G4cout << "  -h, --help             : Print this help message" << G4endl;
    G4cout << "\nExamples:" << G4endl;
    G4cout << "  ./epicToy                          : Interactive mode with multithreading" << G4endl;
    G4cout << "  ./epicToy -m macro.mac             : Batch mode with multithreading" << G4endl;
    G4cout << "  ./epicToy -m macro.mac -t 4        : Batch mode with 4 threads" << G4endl;
    G4cout << "  ./epicToy --single-threaded        : Interactive mode, single-threaded" << G4endl;
    G4cout << G4endl;
}

int main(int argc, char** argv)
{
    // Default settings
    G4bool isBatch = false;
    G4bool forceSingleThreaded = false;
    G4String macroFile = "";
    G4int requestedThreads = -1; // -1 means use all available cores
    
    // Set QT_QPA_PLATFORM environment variable to avoid Qt issues in batch mode
    char* oldQtPlatform = getenv("QT_QPA_PLATFORM");
    std::string oldQtPlatformValue = oldQtPlatform ? oldQtPlatform : "";
    
    // Parse command line arguments
    for (G4int i = 1; i < argc; i++) {
        G4String arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            PrintUsage();
            return 0;
        }
        else if (arg == "-m" || arg == "--macro") {
            isBatch = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                macroFile = argv[++i];
            } else {
                G4cerr << "Error: -m/--macro requires a filename argument" << G4endl;
                PrintUsage();
                return 1;
            }
        }
        else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                requestedThreads = std::atoi(argv[++i]);
                if (requestedThreads <= 0) {
                    G4cerr << "Error: Invalid number of threads: " << requestedThreads << G4endl;
                    PrintUsage();
                    return 1;
                }
            } else {
                G4cerr << "Error: -t/--threads requires a number argument" << G4endl;
                PrintUsage();
                return 1;
            }
        }
        else if (arg == "--single-threaded") {
            forceSingleThreaded = true;
        }
        else if (arg == "batch") {
            // Legacy support for old command format
            isBatch = true;
            if (i + 1 < argc) {
                macroFile = argv[++i];
            }
        }
        else if (arg[0] != '-') {
            // Assume it's a macro file if no flag specified
            isBatch = true;
            macroFile = arg;
        }
        else {
            G4cerr << "Error: Unknown option: " << arg << G4endl;
            PrintUsage();
            return 1;
        }
    }
    
    // Set QT_QPA_PLATFORM=offscreen in batch mode to avoid Qt issues
    if (isBatch) {
        G4cout << "Setting batch mode environment variables..." << G4endl;
        setenv("QT_QPA_PLATFORM", "offscreen", 1);
    }
    
    // Only create UI executive if we're not in batch mode
    G4UIExecutive *ui = nullptr;
    if (!isBatch) {
        // Use Qt session for interactive mode
        ui = new G4UIExecutive(argc, argv, "Qt");
    }

    // Create the appropriate run manager with enhanced multithreading support
    G4RunManager* runManager = nullptr;
    
    #ifdef G4MULTITHREADED
    if (!forceSingleThreaded) {
        // Use multithreaded mode by default for both interactive and batch
        G4MTRunManager* mtRunManager = new G4MTRunManager;
        
        // Determine number of threads to use
        G4int nThreads;
        if (requestedThreads > 0) {
            nThreads = requestedThreads;
        } else {
            // Use all available cores by default
            nThreads = G4Threading::G4GetNumberOfCores();
        }
        
        // Ensure we don't exceed system capabilities
        G4int maxThreads = G4Threading::G4GetNumberOfCores();
        if (nThreads > maxThreads) {
            G4cout << "Warning: Requested " << nThreads << " threads, but only " 
                   << maxThreads << " cores available. Using " << maxThreads << " threads." << G4endl;
            nThreads = maxThreads;
        }
        
        mtRunManager->SetNumberOfThreads(nThreads);
        
        G4cout << "=== MULTITHREADING ENABLED ===" << G4endl;
        G4cout << "Mode: " << (isBatch ? "Batch" : "Interactive") << G4endl;
        G4cout << "Threads: " << nThreads << " (of " << maxThreads << " available cores)" << G4endl;
        G4cout << "===============================" << G4endl;
        
        runManager = mtRunManager;
    } else {
        // Use single-threaded mode when explicitly requested
        runManager = new G4RunManager;
        G4cout << "=== SINGLE-THREADED MODE ===" << G4endl;
        G4cout << "Mode: " << (isBatch ? "Batch" : "Interactive") << G4endl;
        G4cout << "=============================" << G4endl;
    }
    #else
        runManager = new G4RunManager;
        G4cout << "=== SINGLE-THREADED MODE ===" << G4endl;
        G4cout << "Reason: GEANT4 compiled without multithreading support" << G4endl;
        G4cout << "Mode: " << (isBatch ? "Batch" : "Interactive") << G4endl;
        G4cout << "=============================" << G4endl;
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
        if (macroFile.empty()) {
            G4cerr << "Error: No macro file specified for batch mode" << G4endl;
            PrintUsage();
            delete runManager;
            return 1;
        }
        
        G4cout << "Executing macro file: " << macroFile << G4endl;
        G4String command = "/control/execute ";
        command += macroFile;
        G4int status = uiManager->ApplyCommand(command);
        
        if (status != 0) {
            G4cerr << "Error executing macro file: " << macroFile << G4endl;
            delete runManager;
            return 1;
        }
    }
    
    // Clean up
    if (visManager) delete visManager;
    delete runManager;
    
    // Restore original environment variable if it was changed
    if (isBatch) {
        if (!oldQtPlatformValue.empty()) {
            setenv("QT_QPA_PLATFORM", oldQtPlatformValue.c_str(), 1);
        } else {
            unsetenv("QT_QPA_PLATFORM");
        }
    }

    return 0;
}

