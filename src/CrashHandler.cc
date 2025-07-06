#include "CrashHandler.hh"
#include "RunAction.hh"
#include "SimulationLogger.hh"
#include "G4RunManager.hh"
#include "G4Threading.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <sstream>

// Static member definitions
CrashHandler* CrashHandler::fInstance = nullptr;
std::mutex CrashHandler::fMutex;
std::atomic<bool> CrashHandler::fCrashInProgress{false};
const G4String CrashHandler::DEFAULT_BACKUP_DIR = "crash_recovery";

CrashHandler& CrashHandler::GetInstance() {
    std::lock_guard<std::mutex> lock(fMutex);
    if (!fInstance) {
        fInstance = new CrashHandler();
    }
    return *fInstance;
}

CrashHandler::CrashHandler() 
    : fRunManager(nullptr),
      fRunAction(nullptr),
      fAutoSaveEnabled(true),
      fAutoSaveInterval(DEFAULT_AUTO_SAVE_INTERVAL),
      fBackupDirectory(DEFAULT_BACKUP_DIR),
      fCurrentEvent(0),
      fTotalEvents(0),
      fLastSavedEvent(0),
      fRecoveryMode(false),
      fLastCrashInfo("") {
    
    fStartTime = std::chrono::steady_clock::now();
    fLastSaveTime = fStartTime;
    
    G4cout << "\n=== CRASH RECOVERY SYSTEM INITIALIZED ===" << G4endl;
    G4cout << "Auto-save enabled: " << (fAutoSaveEnabled ? "YES" : "NO") << G4endl;
    G4cout << "Auto-save interval: " << fAutoSaveInterval << " events" << G4endl;
    G4cout << "Backup directory: " << fBackupDirectory << G4endl;
    G4cout << "==========================================" << G4endl;
}

CrashHandler::~CrashHandler() {
    Finalize();
}

void CrashHandler::Initialize(G4RunManager* runManager) {
    std::lock_guard<std::mutex> lock(fMutex);
    
    fRunManager = runManager;
    
    // Install signal handlers
    std::signal(SIGINT, HandleInterrupt);     // Ctrl+C
    std::signal(SIGTERM, HandleTermination);  // Termination request
    std::signal(SIGSEGV, HandleCrash);        // Segmentation fault
    std::signal(SIGABRT, HandleCrash);        // Abort signal
    std::signal(SIGFPE, HandleCrash);         // Floating point exception
    std::signal(SIGILL, HandleCrash);         // Illegal instruction
    
    #ifndef _WIN32
    std::signal(SIGQUIT, HandleCrash);        // Quit signal (Unix only)
    std::signal(SIGBUS, HandleCrash);         // Bus error (Unix only)
    #endif
    
    // Create backup directory
    CreateBackupDirectory();
    
    // Check for previous crash recovery
    LoadPreviousCrashInfo();
    
    G4cout << "CrashHandler: Signal handlers installed successfully" << G4endl;
}

void CrashHandler::RegisterRunAction(RunAction* runAction) {
    std::lock_guard<std::mutex> lock(fMutex);
    fRunAction = runAction;
    G4cout << "CrashHandler: RunAction registered for auto-save operations" << G4endl;
}

void CrashHandler::SetAutoSaveEnabled(G4bool enabled, G4int eventInterval) {
    std::lock_guard<std::mutex> lock(fMutex);
    fAutoSaveEnabled = enabled;
    fAutoSaveInterval = eventInterval;
    
    G4cout << "CrashHandler: Auto-save " << (enabled ? "enabled" : "disabled");
    if (enabled) {
        G4cout << " (interval: " << eventInterval << " events)";
    }
    G4cout << G4endl;
}

void CrashHandler::SetBackupDirectory(const G4String& directory) {
    std::lock_guard<std::mutex> lock(fMutex);
    fBackupDirectory = directory;
    CreateBackupDirectory();
    G4cout << "CrashHandler: Backup directory set to: " << directory << G4endl;
}

void CrashHandler::UpdateProgress(G4int eventNumber, G4int totalEvents) {
    fCurrentEvent = eventNumber;
    if (totalEvents > 0) {
        fTotalEvents = totalEvents;
    }
    
    // Check for auto-save
    if (fAutoSaveEnabled && fRunAction && 
        eventNumber > 0 && 
        (eventNumber - fLastSavedEvent) >= fAutoSaveInterval) {
        
        PerformAutoSave();
    }
    
    // Progress reporting every 10% for long runs
    if (fTotalEvents > 0 && eventNumber > 0) {
        G4int progressInterval = std::max(1, fTotalEvents / 10);
        if (eventNumber % progressInterval == 0) {
            G4double percentage = (100.0 * eventNumber) / fTotalEvents;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fStartTime).count();
            
            G4cout << "Progress: " << std::fixed << std::setprecision(1) << percentage 
                   << "% (" << eventNumber << "/" << fTotalEvents << " events) "
                   << "Elapsed: " << elapsed << "s" << G4endl;
        }
    }
}

void CrashHandler::ForceSave(const G4String& reason) {
    std::lock_guard<std::mutex> lock(fMutex);
    
    if (!fRunAction) {
        G4cout << "CrashHandler: Cannot force save - no RunAction registered" << G4endl;
        return;
    }
    
    G4cout << "\n=== FORCING IMMEDIATE SAVE ===" << G4endl;
    G4cout << "Reason: " << reason << G4endl;
    G4cout << "Current event: " << fCurrentEvent.load() << G4endl;
    
    try {
        // Get ROOT file from RunAction and force write
        TFile* rootFile = fRunAction->GetRootFile();
        if (rootFile && rootFile->IsOpen() && !rootFile->IsZombie()) {
            // Use Write() only - avoid flush during regular operation to prevent corruption
            rootFile->Write();
            G4cout << "ROOT file written successfully" << G4endl;
        } else {
            G4cout << "ROOT file not available or already closed - skipping save" << G4endl;
        }
        
        // Save progress information
        SaveProgressInfo(reason);
        
        fLastSavedEvent = fCurrentEvent.load();
        fLastSaveTime = std::chrono::steady_clock::now();
        
        G4cout << "Force save completed successfully" << G4endl;
        
    } catch (const std::exception& e) {
        G4cerr << "Error during force save: " << e.what() << G4endl;
    } catch (...) {
        G4cerr << "Unknown error during force save" << G4endl;
    }
    
    G4cout << "===============================" << G4endl;
}

void CrashHandler::Finalize() {
    std::lock_guard<std::mutex> lock(fMutex);
    
    if (fRunAction) {
        G4cout << "\n=== CRASH HANDLER FINALIZATION ===" << G4endl;
        
        // Don't call ForceSave during normal termination as ROOT files may already be closed
        // ForceSave("Normal program termination");
        G4cout << "Skipping force save during normal termination (ROOT files already saved)" << G4endl;
        
        // Clear recovery files on normal termination
        try {
            std::filesystem::path recoveryFile = 
                std::filesystem::path(std::string(fBackupDirectory)) / "last_crash_info.txt";
            if (std::filesystem::exists(recoveryFile)) {
                std::filesystem::remove(recoveryFile);
                G4cout << "Cleaned up recovery files" << G4endl;
            }
        } catch (...) {
            // Ignore cleanup errors
        }
        
        G4cout << "Crash handler finalized successfully" << G4endl;
        G4cout << "====================================" << G4endl;
    }
}

// Static signal handlers
void CrashHandler::HandleCrash(int signal) {
    // Prevent recursive crashes
    if (fCrashInProgress.exchange(true)) {
        std::_Exit(EXIT_FAILURE); // Force immediate exit
    }
    
    const char* signalName = "UNKNOWN";
    switch (signal) {
        case SIGSEGV: signalName = "SIGSEGV (Segmentation fault)"; break;
        case SIGABRT: signalName = "SIGABRT (Abort)"; break;
        case SIGFPE:  signalName = "SIGFPE (Floating point exception)"; break;
        case SIGILL:  signalName = "SIGILL (Illegal instruction)"; break;
        #ifndef _WIN32
        case SIGQUIT: signalName = "SIGQUIT (Quit)"; break;
        case SIGBUS:  signalName = "SIGBUS (Bus error)"; break;
        #endif
    }
    
    std::cerr << "\n=== CRASH DETECTED ===" << std::endl;
    std::cerr << "Signal: " << signalName << " (" << signal << ")" << std::endl;
    std::cerr << "Attempting emergency save..." << std::endl;
    
    // Log crash information to SimulationLogger
    try {
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
            std::string additionalInfo = "Event: " + std::to_string(fInstance ? fInstance->fCurrentEvent.load() : -1);
            logger->LogCrashInfo(signalName, fInstance ? fInstance->fCurrentEvent.load() : -1, additionalInfo);
            logger->FlushAllLogs(); // Ensure crash info is written to disk
        }
    } catch (...) {
        // Ignore logging errors during crash
    }
    
    // Try to get instance and perform emergency save
    try {
        if (fInstance) {
            std::string reason = "CRASH: ";
            reason += signalName;
            fInstance->PerformEmergencySave(reason);
        }
    } catch (...) {
        std::cerr << "Emergency save failed" << std::endl;
    }
    
    std::cerr << "======================" << std::endl;
    
    // Exit immediately
    std::_Exit(EXIT_FAILURE);
}

void CrashHandler::HandleInterrupt(int signal) {
    // Handle Ctrl+C gracefully
    std::cout << "\n=== INTERRUPT SIGNAL RECEIVED ===" << std::endl;
    std::cout << "Performing graceful shutdown..." << std::endl;
    
    try {
        if (fInstance) {
            fInstance->PerformEmergencySave("User interrupt (Ctrl+C)");
        }
    } catch (...) {
        std::cerr << "Error during interrupt save" << std::endl;
    }
    
    std::cout << "==================================" << std::endl;
    std::exit(EXIT_SUCCESS);
}

void CrashHandler::HandleTermination(int signal) {
    // Handle termination requests
    std::cout << "\n=== TERMINATION SIGNAL RECEIVED ===" << std::endl;
    std::cout << "Performing graceful shutdown..." << std::endl;
    
    try {
        if (fInstance) {
            fInstance->PerformEmergencySave("Termination signal");
        }
    } catch (...) {
        std::cerr << "Error during termination save" << std::endl;
    }
    
    std::cout << "====================================" << std::endl;
    std::exit(EXIT_SUCCESS);
}

// Private helper methods
void CrashHandler::PerformEmergencySave(const G4String& reason) {
    std::cerr << "Emergency save: " << reason << std::endl;
    std::cerr << "Current event: " << fCurrentEvent.load() << std::endl;
    
    try {
        if (fRunAction) {
            // Try to save ROOT file
            TFile* rootFile = fRunAction->GetRootFile();
            if (rootFile && rootFile->IsOpen()) {
                // Only perform basic write and flush - avoid dangerous operations during crash
                rootFile->Write();
                rootFile->Flush();
                std::cerr << "ROOT file emergency save completed" << std::endl;
                
                // REMOVED: Emergency backup creation with tree cloning 
                // This was causing memory corruption during crash recovery
                // The tree cloning operation is not safe during a crash scenario
                // where memory may already be corrupted
            }
        }
        
        // Save crash information
        SaveProgressInfo(reason);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during emergency save: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error during emergency save" << std::endl;
    }
}

void CrashHandler::PerformAutoSave() {
    try {
        if (!fRunAction) return;
        
        // Use proper locking for thread safety during auto-save
        std::lock_guard<std::mutex> lock(fMutex);
        
        TFile* rootFile = fRunAction->GetRootFile();
        if (rootFile && rootFile->IsOpen()) {
            // Only perform write, avoid flush which can be expensive and cause issues
            rootFile->Write();
            
            auto now = std::chrono::steady_clock::now();
            auto totalMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - fLastSaveTime).count();
            auto totalSeconds = totalMilliseconds / 1000;
            auto milliseconds = totalMilliseconds % 1000;
            auto minutes = totalSeconds / 60;
            auto seconds = totalSeconds % 60;
            
            G4cout << "Auto-save completed (event " << fCurrentEvent.load();
            if (minutes > 0) {
                G4cout << ", " << minutes << " min " << seconds << " sec since last save)";
            } else if (seconds > 0) {
                G4cout << ", " << seconds << " sec since last save)";
            } else {
                G4cout << ", " << milliseconds << " ms since last save)";
            }
            G4cout << G4endl;
            
            fLastSavedEvent = fCurrentEvent.load();
            fLastSaveTime = now;
        }
    } catch (...) {
        G4cerr << "Error during auto-save operation" << G4endl;
    }
}

void CrashHandler::CreateBackupDirectory() {
    try {
        std::filesystem::create_directories(std::string(fBackupDirectory));
    } catch (const std::exception& e) {
        G4cerr << "Failed to create backup directory: " << e.what() << G4endl;
    }
}

G4String CrashHandler::GenerateBackupFileName(const G4String& suffix) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << fBackupDirectory << "/backup_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    if (!suffix.empty()) {
        ss << suffix;
    }
    
    ss << ".root";
    
    return ss.str();
}

void CrashHandler::SaveProgressInfo(const G4String& reason) {
    try {
        std::filesystem::path infoFile = 
            std::filesystem::path(std::string(fBackupDirectory)) / "last_crash_info.txt";
        
        std::ofstream file(infoFile);
        if (file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            file << "Crash Recovery Information\n";
            file << "=========================\n";
            file << "Timestamp: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
            file << "Reason: " << reason << "\n";
            file << "Current Event: " << fCurrentEvent.load() << "\n";
            file << "Total Events: " << fTotalEvents.load() << "\n";
            file << "Last Saved Event: " << fLastSavedEvent.load() << "\n";
            file << "Events Lost: " << (fCurrentEvent.load() - fLastSavedEvent.load()) << "\n";
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - fStartTime).count();
            file << "Elapsed Time: " << elapsed << " seconds\n";
            
            if (fTotalEvents > 0) {
                double progress = (100.0 * fCurrentEvent.load()) / fTotalEvents.load();
                file << "Progress: " << std::fixed << std::setprecision(2) << progress << "%\n";
            }
            
            file.close();
        }
    } catch (...) {
        // Ignore errors in progress info saving
    }
}

void CrashHandler::LoadPreviousCrashInfo() {
    try {
        std::filesystem::path infoFile = 
            std::filesystem::path(std::string(fBackupDirectory)) / "last_crash_info.txt";
        
        if (std::filesystem::exists(infoFile)) {
            std::ifstream file(infoFile);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                fLastCrashInfo = buffer.str();
                fRecoveryMode = true;
                
                G4cout << "\n=== CRASH RECOVERY MODE ===" << G4endl;
                G4cout << "Previous crash detected!" << G4endl;
                G4cout << "Recovery information available." << G4endl;
                G4cout << "============================" << G4endl;
                
                file.close();
            }
        }
    } catch (...) {
        // Ignore errors in loading crash info
    }
} 