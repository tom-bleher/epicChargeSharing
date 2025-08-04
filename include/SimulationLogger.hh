#ifndef SIMULATION_LOGGER_HH
#define SIMULATION_LOGGER_HH

#include "G4Types.hh"
#include "G4String.hh"
#include "G4ThreeVector.hh"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <memory>
#include <chrono>
#include <mutex>

// Forward declarations for fitting results
struct Gauss2DResultsCeres;
struct Lorentz2DResultsCeres;
struct Gauss3DResultsCeres;
struct Lorentz3DResultsCeres;

class SimulationLogger {
public:
    // Singleton pattern
    static SimulationLogger* GetInstance();
    
    // Main logging interface
    void Initialize(const std::string& outputDirectory = "logs");
    void Finalize();
    
    // Simulation lifecycle logging
    void LogSimulationStart();
    void LogSimulationEnd();
    void LogRunStart(G4int runID, G4int totalEvents);
    void LogRunEnd(G4int runID);
    void LogEventStart(G4int eventID);
    void LogEventEnd(G4int eventID);
    
    // Parameter logging
    void LogDetectorParameters(G4double detSize, G4double detWidth, G4double pixelSize, 
                              G4double pixelSpacing, G4double pixelCornerOffset, 
                              G4int numPixelsPerSide, G4int totalPixels);
    void LogPhysicsParameters(G4double cutValue, const std::string& physicsList);
    void LogPrimaryGeneratorParameters(const std::string& particleType, G4double energy,
                                     G4ThreeVector pos, G4ThreeVector direction);
    
    // Hit and energy depositionition logging
    void LogPixelHit(G4int eventID, G4int pixelI, G4int pixelJ, G4double energyDeposit,
                    G4ThreeVector pos, G4double stepLength);
    void LogTotalEnergyDepos(G4int eventID, G4double totalEnergy, G4int numHits);
    
    // Fitting results logging
    void LogGaussResults(G4int eventID, const Gauss2DResultsCeres& results);
    void LogLorentzResults(G4int eventID, const Lorentz2DResultsCeres& results);
    void Log3DLorentzResults(G4int eventID, const Lorentz3DResultsCeres& results);
    void Log3DGaussResults(G4int eventID, const Gauss3DResultsCeres& results);
    
    // Performance and statistics logging
    void LogPerformanceMetrics(G4int eventID, G4double eventProcessingTime,
                              G4double totalSimulationTime, G4double memoryUsage);
    void LogtingPerformance(G4int eventID, const std::string& fitType,
                              G4double fittingTime, G4bool converged, G4int iterations);
    void LogPixelHitPattern(G4int eventID, const std::vector<std::pair<G4int, G4int>>& hitPixels,
                           const std::vector<G4double>& energies);
    
    // System and environment logging
    void LogSystemInfo();
    void LogCompilationInfo();
    void LogEnvironmentVariables();
    
    // Configuration and settings logging
    void LogConfiguration(const std::map<std::string, std::string>& config);
    void LogCeresSettings(const std::string& fitType, const std::map<std::string, std::string>& settings);
    
    // Error and warning logging
    void LogError(const std::string& message, const std::string& location = "");
    void LogWarning(const std::string& message, const std::string& location = "");
    void LogInfo(const std::string& message, const std::string& location = "");
    void LogDebug(const std::string& message, const std::string& location = "");
    
    // Statistics and analysis logging
    void LogEventStatistics(G4int totalEvents, G4int successs, G4int faileds,
                           G4double averageChi2, G4double averageTime);
    void LogConvergenceStatistics(const std::string& fitType, G4int totalAttempts,
                                 G4int convergences, G4double averageIterations);
    
    // Crash and recovery logging
    void LogCrashInfo(const std::string& signal, G4int eventID, const std::string& additionalInfo);
    void LogRecoveryInfo(G4int lastSavedEvent, G4int currentEvent, const std::string& backupFile);
    
    // Progress and timing
    void LogProgress(G4int currentEvent, G4int totalEvents, G4double elapsedTime,
                    G4double estimatedTimeRemaining);
    
    // File management
    void FlushAllLogs();
    std::string GetLogDirectory() const { return fLogDirectory; }
    std::string GetMainLogFile() const { return fMainLogFile; }

private:
    SimulationLogger();
    ~SimulationLogger();
    
    // Delete copy constructor and assignment operator
    SimulationLogger(const SimulationLogger&) = delete;
    SimulationLogger& operator=(const SimulationLogger&) = delete;
    
    // Internal logging methods
    void WriteToMainLog(const std::string& level, const std::string& message, 
                       const std::string& location = "");
    void WriteToSpecializedLog(const std::string& filename, const std::string& content);
    std::string GetTimestamp() const;
    std::string FormatMessage(const std::string& level, const std::string& message,
                             const std::string& location) const;
    
    // File creation and management
    void CreateLogFiles();
    void CreateLogHeader(std::ofstream& file, const std::string& title);
    
    // Utility methods
    G4double GetMemoryUsage() const;
    std::string GetSystemInfo() const;
    
    // Member variables
    static SimulationLogger* fInstance;
    static std::mutex fInstanceMutex;
    
    std::string fLogDirectory;
    std::string fMainLogFile;
    std::string fPerformanceLogFile;
    std::string ftingLogFile;
    std::string fHitsLogFile;
    std::string fErrorLogFile;
    std::string fStatsLogFile;
    
    std::unique_ptr<std::ofstream> fMainLog;
    std::unique_ptr<std::ofstream> fPerformanceLog;
    std::unique_ptr<std::ofstream> ftingLog;
    std::unique_ptr<std::ofstream> fHitsLog;
    std::unique_ptr<std::ofstream> fErrorLog;
    std::unique_ptr<std::ofstream> fStatsLog;
    
    std::mutex fLogMutex;
    std::chrono::steady_clock::time_point fSimulationStartTime;
    std::chrono::steady_clock::time_point fRunStartTime;
    std::chrono::steady_clock::time_point fEventStartTime;
    
    G4bool fInitialized;
    G4int fCurrentRunID;
    G4int fCurrentEventID;
    G4int fTotalEvents;
    
    // Statistics tracking
    G4int fTotalFits;
    G4int fSuccessFits;
    G4int fFailFits;
    G4double fTotFitTime;
    std::map<std::string, G4int> fFitTypeCounters;
    std::map<std::string, G4double> fFitTypeTimings;
    std::map<std::string, G4int> fConvergenceCounters;
};

#endif // SIMULATION_LOGGER_HH 