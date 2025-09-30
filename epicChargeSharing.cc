#include <iostream>

#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4Threading.hh"
#include "G4ScoringManager.hh"

#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

int main(int argc, char** argv)
{
    G4bool isBatch = false;
    G4String macroFile = "";
    G4int requestedThreads = -1; // -1: use all available cores
    
    // Set QT_QPA_PLATFORM environment variable to avoid Qt issues in batch mode
    char* oldQtPlatform = getenv("QT_QPA_PLATFORM");
    std::string oldQtPlatformValue = oldQtPlatform ? oldQtPlatform : "";
    
    // Parse command line arguments
    for (G4int i = 1; i < argc; i++) {
        G4String arg = argv[i];
        
        if (arg == "-m") {
            isBatch = true;
            if (i + 1 < argc) {
                macroFile = argv[++i];
            } else {
                G4cerr << "Error: -m requires a filename argument" << G4endl;
                return 1;
            }
        }
        // Disable header/interactive-only modes: only macro-driven runs allowed
        else if (arg == "-t") {
            if (i + 1 < argc) {
                requestedThreads = std::atoi(argv[++i]);
                if (requestedThreads <= 0) {
                    G4cerr << "Error: Invalid number of threads: " << requestedThreads << G4endl;
                    return 1;
                }
            } else {
                G4cerr << "Error: -t requires a number argument" << G4endl;
                return 1;
            }
        }
        else {
            G4cerr << "Error: Unknown option: " << arg << G4endl;
            return 1;
        }
    }
    
    // Set QT_QPA_PLATFORM=offscreen in batch mode to avoid Qt issues
    if (isBatch) {
        G4cout << "Setting batch mode environment variables..." << G4endl;
        setenv("QT_QPA_PLATFORM", "offscreen", 1);
    }
    
    // No interactive UI; macro-driven only
    G4UIExecutive *ui = nullptr;

    // Create the appropriate run manager with enhanced multithreading support
    G4RunManager* runManager = nullptr;
    
    #ifdef G4MULTITHREADED
    if (requestedThreads != 1) {
        // Use multithreaded mode unless explicitly set to 1 thread
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
        
        G4cout << "=== MULTITHREADING ENABLED ===\n"
               << "Mode: " << (isBatch ? "Batch" : "Interactive") << "\n"
               << "Threads: " << nThreads << " (of " << maxThreads << " available cores)\n"
               << "===============================" << G4endl;
        
        runManager = mtRunManager;
    } else {
        // Use single-threaded mode when explicitly requested with -t 1
        runManager = new G4RunManager;
        G4cout << "=== SINGLE-THREADED MODE ===\n"
               << "Mode: " << (isBatch ? "Batch" : "Interactive") << "\n"
               << "=============================" << G4endl;
    }
    #else
        runManager = new G4RunManager;
        G4cout << "=== SINGLE-THREADED MODE ===\n"
               << "Reason: GEANT4 compiled without multithreading support\n"
               << "Mode: " << (isBatch ? "Batch" : "Interactive") << "\n"
               << "=============================" << G4endl;
    #endif
    
    // ---------------------------------------------------------------
    // Enable command-based scoring so that the Multi-Functional
    // Detector’s primitive scorers actually create their hits
    // collections. Without this call the scorer collections are not
    // instantiated and CollectScorerData() finds no data.
    // ---------------------------------------------------------------
    G4ScoringManager::GetScoringManager();

    // Physics List
    runManager->SetUserInitialization(new PhysicsList());

    // Detector Construction
    DetectorConstruction* detConstruction = new DetectorConstruction();
    runManager->SetUserInitialization(detConstruction);

    // Action Initialization with detector construction
    runManager->SetUserInitialization(new ActionInitialization(detConstruction));


    // Get pointer to UI manager
    G4UImanager *uiManager = G4UImanager::GetUIpointer();
    
    // Batch mode execution only
    if (macroFile.empty()) {
        G4cerr << "Error: No macro file specified. Use -m <path-to-macro>." << G4endl;
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
    
    // Clean up
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

