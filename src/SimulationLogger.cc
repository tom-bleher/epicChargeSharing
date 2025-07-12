#include "SimulationLogger.hh"
#include "GaussFit2D.hh"
#include "LorentzFit2D.hh"
#include "PowerLorentzFit2D.hh"
#include "GaussFit3D.hh"
#include "LorentzFit3D.hh"
#include "PowerLorentzFit3D.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <fstream>
#include <thread>
#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Static member definitions
SimulationLogger* SimulationLogger::fInstance = nullptr;
std::mutex SimulationLogger::fInstanceMutex;

SimulationLogger* SimulationLogger::GetInstance() {
    std::lock_guard<std::mutex> lock(fInstanceMutex);
    if (fInstance == nullptr) {
        fInstance = new SimulationLogger();
    }
    return fInstance;
}

SimulationLogger::SimulationLogger()
    : fLogDirectory("logs"),
      fInitialized(false),
      fCurrentRunID(-1),
      fCurrentEventID(-1),
      fTotalEvents(0),
      fTotalFits(0),
      fSuccessFits(0),
      fFailFits(0),
      fTotFitTime(0.0)
{
}

SimulationLogger::~SimulationLogger() {
    if (fInitialized) {
        Finalize();
    }
}

void SimulationLogger::Initialize(const std::string& outputDirectory) {
    std::lock_guard<std::mutex> lock(fLogMutex);
    
    if (fInitialized) {
        return;
    }
    
    fLogDirectory = outputDirectory;
    
    // Create log directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(fLogDirectory, ec);
    if (ec) {
        std::cerr << "Warning: Could not create log directory: " << ec.message() << std::endl;
    }
    
    // Create log files
    CreateLogFiles();
    
    fSimulationStartTime = std::chrono::steady_clock::now();
    fInitialized = true;
    
    // Log system information
    LogSystemInfo();
    LogCompilationInfo();
    LogEnvironmentVariables();
    
    LogInfo("SimulationLogger initialized", "SimulationLogger::Initialize");
}

void SimulationLogger::CreateLogFiles() {
    std::string timestamp = GetTimestamp();
    
    // Define log file names
    fMainLogFile = fLogDirectory + "/simulation_" + timestamp + ".log";
    fPerformanceLogFile = fLogDirectory + "/performance_" + timestamp + ".log";
    ftingLogFile = fLogDirectory + "/fitting_" + timestamp + ".log";
    fHitsLogFile = fLogDirectory + "/hits_" + timestamp + ".log";
    fErrorLogFile = fLogDirectory + "/errors_" + timestamp + ".log";
    fStatsLogFile = fLogDirectory + "/statistics_" + timestamp + ".log";
    
    // Create log file objects
    fMainLog = std::make_unique<std::ofstream>(fMainLogFile, std::ios::out | std::ios::app);
    fPerformanceLog = std::make_unique<std::ofstream>(fPerformanceLogFile, std::ios::out | std::ios::app);
    ftingLog = std::make_unique<std::ofstream>(ftingLogFile, std::ios::out | std::ios::app);
    fHitsLog = std::make_unique<std::ofstream>(fHitsLogFile, std::ios::out | std::ios::app);
    fErrorLog = std::make_unique<std::ofstream>(fErrorLogFile, std::ios::out | std::ios::app);
    fStatsLog = std::make_unique<std::ofstream>(fStatsLogFile, std::ios::out | std::ios::app);
    
    // Create headers for each log file
    CreateLogHeader(*fMainLog, "EPIC CHARGE SHARING SIMULATION - MAIN LOG");
    CreateLogHeader(*fPerformanceLog, "EPIC CHARGE SHARING SIMULATION - PERFORMANCE LOG");
    CreateLogHeader(*ftingLog, "EPIC CHARGE SHARING SIMULATION - FIT RESULTS LOG");
    CreateLogHeader(*fHitsLog, "EPIC CHARGE SHARING SIMULATION - PIXEL HITS LOG");
    CreateLogHeader(*fErrorLog, "EPIC CHARGE SHARING SIMULATION - ERRORS AND WARNINGS LOG");
    CreateLogHeader(*fStatsLog, "EPIC CHARGE SHARING SIMULATION - STATISTICS LOG");
}

void SimulationLogger::CreateLogHeader(std::ofstream& file, const std::string& title) {
    file << "========================================================\n";
    file << title << "\n";
    file << "Generated on: " << GetTimestamp() << "\n";
    file << "========================================================\n\n";
    file.flush();
}

void SimulationLogger::Finalize() {
    std::lock_guard<std::mutex> lock(fLogMutex);
    
    if (!fInitialized) {
        return;
    }
    
    LogSimulationEnd();
    
    // Close all log files
    if (fMainLog) fMainLog->close();
    if (fPerformanceLog) fPerformanceLog->close();
    if (ftingLog) ftingLog->close();
    if (fHitsLog) fHitsLog->close();
    if (fErrorLog) fErrorLog->close();
    if (fStatsLog) fStatsLog->close();
    
    fInitialized = false;
}

void SimulationLogger::LogSimulationStart() {
    fSimulationStartTime = std::chrono::steady_clock::now();
    WriteToMainLog("INFO", "=== SIMULATION STARTED ===");
    WriteToMainLog("INFO", "Timestamp: " + GetTimestamp());
}

void SimulationLogger::LogSimulationEnd() {
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - fSimulationStartTime);
    
    WriteToMainLog("INFO", "=== SIMULATION ENDED ===");
    WriteToMainLog("INFO", "Total simulation time: " + std::to_string(duration.count()) + " seconds");
    WriteToMainLog("INFO", "Timestamp: " + GetTimestamp());
    
    // Log final statistics
    LogEventStatistics(fTotalEvents, fSuccessFits, fFailFits, 
                      fTotFitTime / fTotalFits, fTotFitTime);
}

void SimulationLogger::LogRunStart(G4int runID, G4int totalEvents) {
    fCurrentRunID = runID;
    fTotalEvents = totalEvents;
    fRunStartTime = std::chrono::steady_clock::now();
    
    WriteToMainLog("INFO", "Run " + std::to_string(runID) + " started with " + 
                   std::to_string(totalEvents) + " events");
}

void SimulationLogger::LogRunEnd(G4int runID) {
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - fRunStartTime);
    
    WriteToMainLog("INFO", "Run " + std::to_string(runID) + " completed in " + 
                   std::to_string(duration.count()) + " ms");
    
    fCurrentRunID = -1;
}

void SimulationLogger::LogEventStart(G4int eventID) {
    fCurrentEventID = eventID;
    fEventStartTime = std::chrono::steady_clock::now();
}

void SimulationLogger::LogEventEnd(G4int eventID) {
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - fEventStartTime);
    
    // Log to performance file
    if (fPerformanceLog) {
        *fPerformanceLog << "EVENT " << eventID << " COMPLETED: " << duration.count() 
                        << " μs, Memory: " << GetMemoryUsage() << " MB\n";
        fPerformanceLog->flush();
    }
    
    fCurrentEventID = -1;
}

void SimulationLogger::LogDetectorParameters(G4double detSize, G4double detWidth, G4double pixelSize, 
                                           G4double pixelSpacing, G4double pixelCornerOffset, 
                                           G4int numPixelsPerSide, G4int totalPixels) {
    std::ostringstream oss;
    oss << "\n=== DETECTOR PARAMETERS ===\n";
    oss << "Detector Size: " << detSize/mm << " mm\n";
    oss << "Detector Width: " << detWidth/mm << " mm\n";
    oss << "Pixel Size: " << pixelSize/mm << " mm\n";
    oss << "Pixel Spacing: " << pixelSpacing/mm << " mm\n";
    oss << "Pixel Corner Offset: " << pixelCornerOffset/mm << " mm\n";
    oss << "Pixels per Side: " << numPixelsPerSide << "\n";
    oss << "Total Pixels: " << totalPixels << "\n";
    oss << "Pixel Area: " << (pixelSize * pixelSize)/(mm*mm) << " mm²\n";
    oss << "Total Pixel Area: " << (totalPixels * pixelSize * pixelSize)/(mm*mm) << " mm²\n";
    oss << "Detector Area: " << (detSize * detSize)/(mm*mm) << " mm²\n";
    oss << "Pixel Coverage: " << (totalPixels * pixelSize * pixelSize)/(detSize * detSize) * 100.0 << "%\n";
    oss << "===========================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogPhysicsParameters(G4double cutValue, const std::string& physicsList) {
    std::ostringstream oss;
    oss << "\n=== PHYSICS PARAMETERS ===\n";
    oss << "Physics List: " << physicsList << "\n";
    oss << "Production Cut: " << cutValue/micrometer << " μm\n";
    oss << "==========================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogPrimaryGeneratorParameters(const std::string& particleType, G4double energy,
                                                   G4ThreeVector pos, G4ThreeVector direction) {
    std::ostringstream oss;
    oss << "\n=== PRIMARY GENERATOR PARAMETERS ===\n";
    oss << "Particle Type: " << particleType << "\n";
    oss << "Energy: " << energy/MeV << " MeV\n";
    oss << "Initial Pos: (" << pos.x()/mm << ", " << pos.y()/mm 
        << ", " << pos.z()/mm << ") mm\n";
    oss << "Direction: (" << direction.x() << ", " << direction.y() 
        << ", " << direction.z() << ")\n";
    oss << "====================================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogPixelHit(G4int eventID, G4int pixelI, G4int pixelJ, G4double energyDeposit,
                                 G4ThreeVector pos, G4double stepLength) {
    if (fHitsLog) {
        *fHitsLog << "EVENT:" << eventID 
                 << " PIXEL:(" << pixelI << "," << pixelJ << ")"
                 << " ENERGY:" << energyDeposit/keV << "keV"
                 << " POS:(" << pos.x()/mm << "," << pos.y()/mm 
                 << "," << pos.z()/mm << ")mm"
                 << " STEP:" << stepLength/micrometer << "μm\n";
        fHitsLog->flush();
    }
}

void SimulationLogger::LogTotalEnergyDepos(G4int eventID, G4double totalEnergy, G4int numHits) {
    WriteToMainLog("DEBUG", "Event " + std::to_string(eventID) + 
                   ": Total energy = " + std::to_string(totalEnergy/keV) + " keV, " +
                   "Number of hits = " + std::to_string(numHits));
}

void SimulationLogger::LogGaussResults(G4int eventID, const Gauss2DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - GAUSS FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "X Direction:\n";
            *ftingLog << "  Center: " << results.x_center/mm << " ± " << results.x_center_err/mm << " mm\n";
            *ftingLog << "  Sigma: " << results.x_sigma/mm << " ± " << results.x_sigma_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.x_amp << " ± " << results.x_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.x_chi2red << " (DOF: " << results.x_dof << ")\n";
            
            *ftingLog << "Y Direction:\n";
            *ftingLog << "  Center: " << results.y_center/mm << " ± " << results.y_center_err/mm << " mm\n";
            *ftingLog << "  Sigma: " << results.y_sigma/mm << " ± " << results.y_sigma_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.y_amp << " ± " << results.y_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.y_chi2red << " (DOF: " << results.y_dof << ")\n";
        }
        *ftingLog << "=================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::LogLorentzResults(G4int eventID, const Lorentz2DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - LORENTZ FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "X Direction:\n";
            *ftingLog << "  Center: " << results.x_center/mm << " ± " << results.x_center_err/mm << " mm\n";
            *ftingLog << "  Gamma: " << results.x_gamma/mm << " ± " << results.x_gamma_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.x_amp << " ± " << results.x_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.x_chi2red << " (DOF: " << results.x_dof << ")\n";
            
            *ftingLog << "Y Direction:\n";
            *ftingLog << "  Center: " << results.y_center/mm << " ± " << results.y_center_err/mm << " mm\n";
            *ftingLog << "  Gamma: " << results.y_gamma/mm << " ± " << results.y_gamma_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.y_amp << " ± " << results.y_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.y_chi2red << " (DOF: " << results.y_dof << ")\n";
        }
        *ftingLog << "====================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::LogPowerLorentzResults(G4int eventID, const PowerLorentz2DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - POWER LORENTZ FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "X Direction:\n";
            *ftingLog << "  Center: " << results.x_center/mm << " ± " << results.x_center_err/mm << " mm\n";
            *ftingLog << "  Gamma: " << results.x_gamma/mm << " ± " << results.x_gamma_err/mm << " mm\n";
            *ftingLog << "  Beta: " << results.x_beta << " ± " << results.x_beta_err << "\n";
            *ftingLog << "  Amp: " << results.x_amp << " ± " << results.x_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.x_chi2red << " (DOF: " << results.x_dof << ")\n";
            
            *ftingLog << "Y Direction:\n";
            *ftingLog << "  Center: " << results.y_center/mm << " ± " << results.y_center_err/mm << " mm\n";
            *ftingLog << "  Gamma: " << results.y_gamma/mm << " ± " << results.y_gamma_err/mm << " mm\n";
            *ftingLog << "  Beta: " << results.y_beta << " ± " << results.y_beta_err << "\n";
            *ftingLog << "  Amp: " << results.y_amp << " ± " << results.y_amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.y_chi2red << " (DOF: " << results.y_dof << ")\n";
        }
        *ftingLog << "==========================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::Log3DLorentzResults(G4int eventID, const Lorentz3DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - 3D LORENTZ FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "3D  Parameters:\n";
            *ftingLog << "  Center X: " << results.center_x/mm << " ± " << results.center_x_err/mm << " mm\n";
            *ftingLog << "  Center Y: " << results.center_y/mm << " ± " << results.center_y_err/mm << " mm\n";
            *ftingLog << "  Gamma X: " << results.gamma_x/mm << " ± " << results.gamma_x_err/mm << " mm\n";
            *ftingLog << "  Gamma Y: " << results.gamma_y/mm << " ± " << results.gamma_y_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.amp << " ± " << results.amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.chi2red << " (DOF: " << results.dof << ")\n";
        }
        *ftingLog << "=======================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::Log3DGaussResults(G4int eventID, const Gauss3DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - 3D GAUSS FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "3D Gauss Parameters:\n";
            *ftingLog << "  Center X: " << results.center_x/mm << " ± " << results.center_x_err/mm << " mm\n";
            *ftingLog << "  Center Y: " << results.center_y/mm << " ± " << results.center_y_err/mm << " mm\n";
            *ftingLog << "  Sigma X: " << results.sigma_x/mm << " ± " << results.sigma_x_err/mm << " mm\n";
            *ftingLog << "  Sigma Y: " << results.sigma_y/mm << " ± " << results.sigma_y_err/mm << " mm\n";
            *ftingLog << "  Amp: " << results.amp << " ± " << results.amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.chi2red << " (DOF: " << results.dof << ")\n";
        }
        *ftingLog << "======================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::Log3DPowerLorentzResults(G4int eventID, const PowerLorentz3DResultsCeres& results) {
    if (ftingLog) {
        *ftingLog << "\n=== EVENT " << eventID << " - 3D POWER LORENTZ FIT RESULTS ===\n";
        *ftingLog << "Success: " << (results.fit_success ? "YES" : "NO") << "\n";
        if (results.fit_success) {
            *ftingLog << "3D Power-Law Lorentz Parameters:\n";
            *ftingLog << "  Center X: " << results.center_x/mm << " ± " << results.center_x_err/mm << " mm\n";
            *ftingLog << "  Center Y: " << results.center_y/mm << " ± " << results.center_y_err/mm << " mm\n";
            *ftingLog << "  Gamma X: " << results.gamma_x/mm << " ± " << results.gamma_x_err/mm << " mm\n";
            *ftingLog << "  Gamma Y: " << results.gamma_y/mm << " ± " << results.gamma_y_err/mm << " mm\n";
            *ftingLog << "  Beta: " << results.beta << " ± " << results.beta_err << "\n";
            *ftingLog << "  Amp: " << results.amp << " ± " << results.amp_err << "\n";
            *ftingLog << "  Chi2/DOF: " << results.chi2red << " (DOF: " << results.dof << ")\n";
        }
        *ftingLog << "=============================================================\n\n";
        ftingLog->flush();
    }
    
    fTotalFits++;
    if (results.fit_success) {
        fSuccessFits++;
    } else {
        fFailFits++;
    }
}

void SimulationLogger::LogPerformanceMetrics(G4int eventID, G4double eventProcessingTime,
                                           G4double totalSimulationTime, G4double memoryUsage) {
    if (fPerformanceLog) {
        *fPerformanceLog << "EVENT:" << eventID 
                        << " PROC_TIME:" << eventProcessingTime << "ms"
                        << " TOTAL_TIME:" << totalSimulationTime << "s"
                        << " MEMORY:" << memoryUsage << "MB\n";
        fPerformanceLog->flush();
    }
}

void SimulationLogger::LogtingPerformance(G4int eventID, const std::string& fitType,
                                           G4double fittingTime, G4bool converged, G4int iterations) {
    if (fPerformanceLog) {
        *fPerformanceLog << "FIT EVENT:" << eventID 
                        << " TYPE:" << fitType
                        << " TIME:" << fittingTime << "ms"
                        << " CONVERGED:" << (converged ? "YES" : "NO")
                        << " ITERATIONS:" << iterations << "\n";
        fPerformanceLog->flush();
    }
    
    fTotFitTime += fittingTime;
    fFitTypeTimings[fitType] += fittingTime;
    fFitTypeCounters[fitType]++;
    if (converged) {
        fConvergenceCounters[fitType]++;
    }
}

void SimulationLogger::LogPixelHitPattern(G4int eventID, const std::vector<std::pair<G4int, G4int>>& hitPixels,
                                        const std::vector<G4double>& energies) {
    if (fHitsLog) {
        *fHitsLog << "EVENT:" << eventID << " HIT_PATTERN:";
        for (size_t i = 0; i < hitPixels.size() && i < energies.size(); ++i) {
            *fHitsLog << " (" << hitPixels[i].first << "," << hitPixels[i].second 
                     << ":" << energies[i]/keV << "keV)";
        }
        *fHitsLog << "\n";
        fHitsLog->flush();
    }
}

void SimulationLogger::LogSystemInfo() {
    std::ostringstream oss;
    oss << "\n=== SYSTEM INFORMATION ===\n";
    oss << GetSystemInfo();
    oss << "Memory Usage: " << GetMemoryUsage() << " MB\n";
    oss << "==========================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogCompilationInfo() {
    std::ostringstream oss;
    oss << "\n=== COMPILATION INFORMATION ===\n";
    oss << "Compiler: ";
#ifdef __GNUC__
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
#elif defined(_MSC_VER)
    oss << "MSVC " << _MSC_VER << "\n";
#else
    oss << "Unknown\n";
#endif
    oss << "Compile Date: " << __DATE__ << " " << __TIME__ << "\n";
    oss << "===============================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogEnvironmentVariables() {
    std::ostringstream oss;
    oss << "\n=== ENVIRONMENT VARIABLES ===\n";
    
    const char* envVars[] = {"G4DATADIR", "G4NEUTRONHPDATA", "G4LEDATA", "G4LEVELGAMMADATA", 
                            "G4RADIOACTIVEDATA", "G4PARTICLEXSDATA", "G4PIIDATA", 
                            "G4REALSURFACEDATA", "G4SAIDXSDATA", "G4ABLADATA", "G4ENSDFSTATEDATA"};
    
    for (const char* var : envVars) {
        const char* value = std::getenv(var);
        oss << var << ": " << (value ? value : "NOT SET") << "\n";
    }
    oss << "=============================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogConfiguration(const std::map<std::string, std::string>& config) {
    std::ostringstream oss;
    oss << "\n=== CONFIGURATION SETTINGS ===\n";
    for (const auto& pair : config) {
        oss << pair.first << ": " << pair.second << "\n";
    }
    oss << "===============================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogCeresSettings(const std::string& fitType, const std::map<std::string, std::string>& settings) {
    if (ftingLog) {
        *ftingLog << "\n=== CERES SETTINGS FOR " << fitType << " ===\n";
        for (const auto& pair : settings) {
            *ftingLog << pair.first << ": " << pair.second << "\n";
        }
        *ftingLog << "============================================\n\n";
        ftingLog->flush();
    }
}

void SimulationLogger::LogError(const std::string& message, const std::string& location) {
    WriteToMainLog("ERROR", message, location);
    if (fErrorLog) {
        *fErrorLog << GetTimestamp() << " ERROR";
        if (!location.empty()) {
            *fErrorLog << " [" << location << "]";
        }
        *fErrorLog << ": " << message << "\n";
        fErrorLog->flush();
    }
}

void SimulationLogger::LogWarning(const std::string& message, const std::string& location) {
    WriteToMainLog("WARNING", message, location);
    if (fErrorLog) {
        *fErrorLog << GetTimestamp() << " WARNING";
        if (!location.empty()) {
            *fErrorLog << " [" << location << "]";
        }
        *fErrorLog << ": " << message << "\n";
        fErrorLog->flush();
    }
}

void SimulationLogger::LogInfo(const std::string& message, const std::string& location) {
    WriteToMainLog("INFO", message, location);
}

void SimulationLogger::LogDebug(const std::string& message, const std::string& location) {
    WriteToMainLog("DEBUG", message, location);
}

void SimulationLogger::LogEventStatistics(G4int totalEvents, G4int successs, G4int faileds,
                                        G4double averageChi2, G4double averageTime) {
    if (fStatsLog) {
        *fStatsLog << "\n=== EVENT STATISTICS ===\n";
        *fStatsLog << "Total Events Processed: " << totalEvents << "\n";
        *fStatsLog << "Total ting Algorithm Runs: " << (successs + faileds) << "\n";
        *fStatsLog << "  - Success Algorithm s: " << successs << "\n";
        *fStatsLog << "  - Failed Algorithm s: " << faileds << "\n";
        G4int totalAttempts = successs + faileds;
        *fStatsLog << "Algorithm Success Rate: " << (totalAttempts > 0 ? 100.0 * successs / totalAttempts : 0.0) << "%\n";
        if (totalEvents > 0) {
            *fStatsLog << "Average s per Event: " << std::fixed << std::setprecision(1) << (G4double)totalAttempts / totalEvents << "\n";
        }
        *fStatsLog << "Average Chi2: " << averageChi2 << "\n";
        *fStatsLog << "Average  Time: " << averageTime << " ms\n";
        *fStatsLog << "========================\n\n";
        fStatsLog->flush();
    }
}

void SimulationLogger::LogConvergenceStatistics(const std::string& fitType, G4int totalAttempts,
                                              G4int convergences, G4double averageIterations) {
    if (fStatsLog) {
        *fStatsLog << "\n=== CONVERGENCE STATISTICS FOR " << fitType << " ===\n";
        *fStatsLog << "Total Attempts: " << totalAttempts << "\n";
        *fStatsLog << "Convergences: " << convergences << "\n";
        *fStatsLog << "Convergence Rate: " << (totalAttempts > 0 ? 100.0 * convergences / totalAttempts : 0.0) << "%\n";
        *fStatsLog << "Average Iterations: " << averageIterations << "\n";
        *fStatsLog << "=================================================\n\n";
        fStatsLog->flush();
    }
}

void SimulationLogger::LogCrashInfo(const std::string& signal, G4int eventID, const std::string& additionalInfo) {
    std::ostringstream oss;
    oss << "\n=== CRASH DETECTED ===\n";
    oss << "Signal: " << signal << "\n";
    oss << "Event ID: " << eventID << "\n";
    oss << "Timestamp: " << GetTimestamp() << "\n";
    oss << "Additional Info: " << additionalInfo << "\n";
    oss << "======================\n";
    
    WriteToMainLog("CRITICAL", oss.str());
    LogError("CRASH DETECTED: " + signal + " at event " + std::to_string(eventID), "CrashHandler");
}

void SimulationLogger::LogRecoveryInfo(G4int lastSavedEvent, G4int currentEvent, const std::string& backupFile) {
    std::ostringstream oss;
    oss << "\n=== RECOVERY INFORMATION ===\n";
    oss << "Last Saved Event: " << lastSavedEvent << "\n";
    oss << "Current Event: " << currentEvent << "\n";
    oss << "Events Lost: " << (currentEvent - lastSavedEvent) << "\n";
    oss << "Backup File: " << backupFile << "\n";
    oss << "============================\n";
    
    WriteToMainLog("INFO", oss.str());
}

void SimulationLogger::LogProgress(G4int currentEvent, G4int totalEvents, G4double elapsedTime,
                                 G4double estimatedTimeRemaining) {
    if (currentEvent % 100 == 0) {  // Log progress every 100 events
        G4double progress = totalEvents > 0 ? 100.0 * currentEvent / totalEvents : 0.0;
        
        std::ostringstream oss;
        oss << "Progress: " << currentEvent << "/" << totalEvents 
            << " (" << std::fixed << std::setprecision(1) << progress << "%), "
            << "Elapsed: " << std::fixed << std::setprecision(1) << elapsedTime << "s, "
            << "ETA: " << std::fixed << std::setprecision(1) << estimatedTimeRemaining << "s";
        
        WriteToMainLog("INFO", oss.str());
    }
}

void SimulationLogger::FlushAllLogs() {
    std::lock_guard<std::mutex> lock(fLogMutex);
    
    if (fMainLog) fMainLog->flush();
    if (fPerformanceLog) fPerformanceLog->flush();
    if (ftingLog) ftingLog->flush();
    if (fHitsLog) fHitsLog->flush();
    if (fErrorLog) fErrorLog->flush();
    if (fStatsLog) fStatsLog->flush();
}

void SimulationLogger::WriteToMainLog(const std::string& level, const std::string& message, 
                                    const std::string& location) {
    if (fMainLog) {
        *fMainLog << FormatMessage(level, message, location) << "\n";
        fMainLog->flush();
    }
}

void SimulationLogger::WriteToSpecializedLog(const std::string& filename, const std::string& content) {
    std::ofstream file(fLogDirectory + "/" + filename, std::ios::out | std::ios::app);
    if (file.is_open()) {
        file << content << std::endl;
        file.close();
    }
}

std::string SimulationLogger::GetTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string SimulationLogger::FormatMessage(const std::string& level, const std::string& message,
                                          const std::string& location) const {
    std::ostringstream oss;
    oss << GetTimestamp() << " [" << level << "]";
    if (!location.empty()) {
        oss << " {" << location << "}";
    }
    oss << ": " << message;
    return oss.str();
}

G4double SimulationLogger::GetMemoryUsage() const {
#ifdef __linux__
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    
    // Get process-specific memory usage
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string dummy;
            G4double memKB;
            iss >> dummy >> memKB;
            return memKB / 1024.0;  // Convert to MB
        }
    }
    return 0.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize / (1024.0 * 1024.0);  // Convert to MB
#else
    return 0.0;  // Unsupported platform
#endif
}

std::string SimulationLogger::GetSystemInfo() const {
    std::ostringstream oss;
    
#ifdef __linux__
    std::ifstream versionFile("/proc/version");
    std::string versionLine;
    if (std::getline(versionFile, versionLine)) {
        oss << "OS: " << versionLine.substr(0, 50) << "...\n";
    }
    
    oss << "CPU Cores: " << std::thread::hardware_concurrency() << "\n";
    
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    oss << "Total RAM: " << memInfo.totalram / (1024 * 1024 * 1024) << " GB\n";
    oss << "Free RAM: " << memInfo.freeram / (1024 * 1024 * 1024) << " GB\n";
#elif defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    oss << "CPU Cores: " << sysInfo.dwNumberOfProcessors << "\n";
    
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    oss << "Total RAM: " << memInfo.ullTotalPhys / (1024 * 1024 * 1024) << " GB\n";
    oss << "Free RAM: " << memInfo.ullAvailPhys / (1024 * 1024 * 1024) << " GB\n";
#else
    oss << "System info not available on this platform\n";
#endif
    
    return oss.str();
} 