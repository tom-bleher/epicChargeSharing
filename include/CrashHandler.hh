#ifndef CRASHHANDLER_HH
#define CRASHHANDLER_HH

#include "globals.hh"
#include <csignal>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <chrono>

// Forward declarations
class G4RunManager;
class RunAction;

/**
 * @brief Crash recovery and signal handling system for simulation safety
 * 
 * This class provides:
 * - Signal handling for graceful crash recovery
 * - Automatic ROOT file finalization on crashes
 * - Periodic auto-save functionality
 * - Thread-safe operation for multithreaded simulations
 * - Progress monitoring and reporting
 */
class CrashHandler {
public:
    // Singleton pattern for global access
    static CrashHandler& GetInstance();
    
    // Initialize the crash handler with run manager
    void Initialize(G4RunManager* runManager);
    
    // Register a RunAction for auto-save operations
    void RegisterRunAction(RunAction* runAction);
    
    // Enable/disable periodic auto-save (default: every 1000 events)
    void SetAutoSaveEnabled(G4bool enabled, G4int eventInterval = 1000);
    
    // Set the backup directory for crash recovery files
    void SetBackupDirectory(const G4String& directory);
    
    // Update event progress (called from EventAction)
    void UpdateProgress(G4int eventNumber, G4int totalEvents = -1);
    
    // Force an immediate save operation
    void ForceSave(const G4String& reason = "Manual save");
    
    // Get crash recovery status
    G4bool IsRecoveryMode() const { return fRecoveryMode; }
    G4String GetLastCrashInfo() const { return fLastCrashInfo; }
    
    // Cleanup and finalization
    void Finalize();
    
private:
    // Private constructor for singleton
    CrashHandler();
    ~CrashHandler();
    
    // Delete copy constructor and assignment operator
    CrashHandler(const CrashHandler&) = delete;
    CrashHandler& operator=(const CrashHandler&) = delete;
    
    // Signal handlers (static for C compatibility)
    static void HandleCrash(int signal);
    static void HandleInterrupt(int signal);
    static void HandleTermination(int signal);
    
    // Internal save operations
    void PerformEmergencySave(const G4String& reason);
    void PerformAutoSave();
    
    // File operations
    void CreateBackupDirectory();
    G4String GenerateBackupFileName(const G4String& suffix = "");
    void SaveProgressInfo(const G4String& reason);
    void LoadPreviousCrashInfo();
    
    // Thread safety
    static std::mutex fMutex;
    static std::atomic<bool> fCrashInProgress;
    
    // Singleton instance
    static CrashHandler* fInstance;
    
    // Core components
    G4RunManager* fRunManager;
    RunAction* fRunAction;
    
    // Configuration
    G4bool fAutoSaveEnabled;
    G4int fAutoSaveInterval;
    G4String fBackupDirectory;
    
    // Progress tracking
    std::atomic<G4int> fCurrentEvent;
    std::atomic<G4int> fTotalEvents;
    std::atomic<G4int> fLastSavedEvent;
    
    // Recovery information
    G4bool fRecoveryMode;
    G4String fLastCrashInfo;
    
    // Timing
    std::chrono::time_point<std::chrono::steady_clock> fStartTime;
    std::chrono::time_point<std::chrono::steady_clock> fLastSaveTime;
    
    // Constants
    static const G4int DEFAULT_AUTO_SAVE_INTERVAL = 1000;
    static const G4String DEFAULT_BACKUP_DIR;
};

#endif // CRASHHANDLER_HH 