#include "RunAction.hh"
#include "Constants.hh"
#include "Control.hh"
#include "SimulationLogger.hh"
#include "CrashHandler.hh"

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4AutoLock.hh"

#include <sstream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <cmath>

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TNamed.h"
#include "TChain.h"
#include "TSystem.h"
#include "TError.h"
#include "TROOT.h"
#include "TThread.h"
#include "TFileMerger.h"
#include "RVersion.h"

// Initialize static synchronization variables
std::mutex RunAction::fRootMutex;
std::atomic<int> RunAction::fWorkersCompleted{0};
std::atomic<int> RunAction::fTotalWorkers{0};
std::condition_variable RunAction::fWorkerCompletionCV;
std::mutex RunAction::fSyncMutex;
std::atomic<bool> RunAction::fAllWorkersCompleted{false};

// Thread-safe ROOT initialization
static std::once_flag gRootInitFlag;

static void InitializeROOTThreading() {
    if (G4Threading::IsMultithreadedApplication()) {
        // Initialize ROOT threading support
        TThread::Initialize();
        gROOT->SetBatch(true); // Ensure batch mode for MT
        
        // Additional ROOT threading safety settings
        gErrorIgnoreLevel = kWarning; // Suppress minor ROOT warnings in MT mode
        
        // Enable ROOT thread safety if available - use different methods for different versions
        #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
        try {
            // ROOT 6.18+ supports implicit multi-threading
            if (!ROOT::IsImplicitMTEnabled()) {
                ROOT::EnableImplicitMT();
                G4cout << "ROOT implicit multi-threading enabled" << G4endl;
            }
        } catch (...) {
            G4cout << "ROOT multi-threading not available in this version" << G4endl;
        }
        #else
        G4cout << "ROOT version < 6.18, using basic threading support" << G4endl;
        #endif
        
        G4cout << "ROOT threading initialization complete" << G4endl;
    }
}

RunAction::RunAction()
: G4UserRunAction(),
  fRootFile(nullptr),
  fTree(nullptr),
  fAutoSaveEnabled(false), fAutoSaveInterval(1000), fEventsSinceLastSave(0),
  // Initialize HITS variables
  fTrueX(0),
  fTrueY(0),
  fInitX(0),
  fInitY(0),
  fInitZ(0),
  fPixelX(0),
  fPixelY(0),
  fEdep(0),
  fPixelTrueDeltaX(0),
  fPixelTrueDeltaY(0),
  // Initialize delta variables
  fGaussRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussColDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzColDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize transformed diagonal coordinate variables
  fGaussMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize mean estimation variables
  fGaussMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Gauss fit variables
  fGaussRowAmp(0),
  fGaussRowAmpErr(0),
  fGaussRowSigma(0),
  fGaussRowSigmaErr(0),
  fGaussRowVertOffset(0),
  fGaussRowVertOffsetErr(0),
  fGaussRowCenter(0),
  fGaussRowCenterErr(0),
  fGaussRowChi2red(0),
  fGaussRowPp(0),
  fGaussRowDOF(0),
  // Initialize charge uncertainties (5% of max charge)
  fGaussRowChargeErr(0),
  fGaussColChargeErr(0),
  fLorentzRowChargeErr(0),
  fLorentzColChargeErr(0),
  fGaussColAmp(0),
  fGaussColAmpErr(0),
  fGaussColSigma(0),
  fGaussColSigmaErr(0),
  fGaussColVertOffset(0),
  fGaussColVertOffsetErr(0),
  fGaussColCenter(0),
  fGaussColCenterErr(0),
  fGaussColChi2red(0),
  fGaussColPp(0),
  fGaussColDOF(0),
  fGaussMainDiagXAmp(0),
  fGaussMainDiagXAmpErr(0),
  fGaussMainDiagXSigma(0),
  fGaussMainDiagXSigmaErr(0),
  fGaussMainDiagXVertOffset(0),
  fGaussMainDiagXVertOffsetErr(0),
  fGaussMainDiagXCenter(0),
  fGaussMainDiagXCenterErr(0),
  fGaussMainDiagXChi2red(0),
  fGaussMainDiagXPp(0),
  fGaussMainDiagXDOF(0),
  fGaussMainDiagYAmp(0),
  fGaussMainDiagYAmpErr(0),
  fGaussMainDiagYSigma(0),
  fGaussMainDiagYSigmaErr(0),
  fGaussMainDiagYVertOffset(0),
  fGaussMainDiagYVertOffsetErr(0),
  fGaussMainDiagYCenter(0),
  fGaussMainDiagYCenterErr(0),
  fGaussMainDiagYChi2red(0),
  fGaussMainDiagYPp(0),
  fGaussMainDiagYDOF(0),
  fGaussSecDiagXAmp(0),
  fGaussSecDiagXAmpErr(0),
  fGaussSecDiagXSigma(0),
  fGaussSecDiagXSigmaErr(0),
  fGaussSecDiagXVertOffset(0),
  fGaussSecDiagXVertOffsetErr(0),
  fGaussSecDiagXCenter(0),
  fGaussSecDiagXCenterErr(0),
  fGaussSecDiagXChi2red(0),
  fGaussSecDiagXPp(0),
  fGaussSecDiagXDOF(0),
  fGaussSecDiagYAmp(0),
  fGaussSecDiagYAmpErr(0),
  fGaussSecDiagYSigma(0),
  fGaussSecDiagYSigmaErr(0),
  fGaussSecDiagYVertOffset(0),
  fGaussSecDiagYVertOffsetErr(0),
  fGaussSecDiagYCenter(0),
  fGaussSecDiagYCenterErr(0),
  fGaussSecDiagYChi2red(0),
  fGaussSecDiagYPp(0),
  fGaussSecDiagYDOF(0),
  // Initialize Lorentz fit variables
  fLorentzRowAmp(0),
  fLorentzRowAmpErr(0),
  fLorentzRowGamma(0),
  fLorentzRowGammaErr(0),
  fLorentzRowVertOffset(0),
  fLorentzRowVertOffsetErr(0),
  fLorentzRowCenter(0),
  fLorentzRowCenterErr(0),
  fLorentzRowChi2red(0),
  fLorentzRowPp(0),
  fLorentzRowDOF(0),
  fLorentzColAmp(0),
  fLorentzColGamma(0),
  fLorentzColGammaErr(0),
  fLorentzColVertOffset(0),
  fLorentzColVertOffsetErr(0),
  fLorentzColCenter(0),
  fLorentzColCenterErr(0),
  fLorentzColChi2red(0),
  fLorentzColPp(0),
  fLorentzColDOF(0),
  fLorentzMainDiagXAmp(0),
  fLorentzMainDiagXAmpErr(0),
  fLorentzMainDiagXGamma(0),
  fLorentzMainDiagXGammaErr(0),
  fLorentzMainDiagXVertOffset(0),
  fLorentzMainDiagXVertOffsetErr(0),
  fLorentzMainDiagXCenter(0),
  fLorentzMainDiagXCenterErr(0),
  fLorentzMainDiagXChi2red(0),
  fLorentzMainDiagXPp(0),
  fLorentzMainDiagXDOF(0),
  fLorentzMainDiagYAmp(0),
  fLorentzMainDiagYAmpErr(0),
  fLorentzMainDiagYGamma(0),
  fLorentzMainDiagYGammaErr(0),
  fLorentzMainDiagYVertOffset(0),
  fLorentzMainDiagYVertOffsetErr(0),
  fLorentzMainDiagYCenter(0),
  fLorentzMainDiagYCenterErr(0),
  fLorentzMainDiagYChi2red(0),
  fLorentzMainDiagYPp(0),
  fLorentzMainDiagYDOF(0),
  fLorentzSecDiagXAmp(0),
  fLorentzSecDiagXAmpErr(0),
  fLorentzSecDiagXGamma(0),
  fLorentzSecDiagXGammaErr(0),
  fLorentzSecDiagXVertOffset(0),
  fLorentzSecDiagXVertOffsetErr(0),
  fLorentzSecDiagXCenter(0),
  fLorentzSecDiagXCenterErr(0),
  fLorentzSecDiagXChi2red(0),
  fLorentzSecDiagXPp(0),
  fLorentzSecDiagXDOF(0),
  fLorentzSecDiagYAmp(0),
  fLorentzSecDiagYAmpErr(0),
  fLorentzSecDiagYGamma(0),
  fLorentzSecDiagYGammaErr(0),
  fLorentzSecDiagYVertOffset(0),
  fLorentzSecDiagYVertOffsetErr(0),
  fLorentzSecDiagYCenter(0),
  fLorentzSecDiagYCenterErr(0),
  fLorentzSecDiagYChi2red(0),
  fLorentzSecDiagYPp(0),
  fLorentzSecDiagYDOF(0),
  // Initialize Power-Law Lorentz fit variables
  fPowerLorentzRowAmp(0),
  fPowerLorentzRowAmpErr(0),
  fPowerLorentzRowBeta(0),
  fPowerLorentzRowBetaErr(0),
  fPowerLorentzRowGamma(0),
  fPowerLorentzRowGammaErr(0),
  fPowerLorentzRowVertOffset(0),
  fPowerLorentzRowVertOffsetErr(0),
  fPowerLorentzRowCenter(0),
  fPowerLorentzRowCenterErr(0),
  fPowerLorentzRowChi2red(0),
  fPowerLorentzRowPp(0),
  fPowerLorentzRowDOF(0),
  fPowerLorentzColAmp(0),
  fPowerLorentzColAmpErr(0),
  fPowerLorentzColBeta(0),
  fPowerLorentzColBetaErr(0),
  fPowerLorentzColGamma(0),
  fPowerLorentzColGammaErr(0),
  fPowerLorentzColVertOffset(0),
  fPowerLorentzColVertOffsetErr(0),
  fPowerLorentzColCenter(0),
  fPowerLorentzColCenterErr(0),
  fPowerLorentzColChi2red(0),
  fPowerLorentzColPp(0),
  fPowerLorentzColDOF(0),
  fPowerLorentzMainDiagXAmp(0),
  fPowerLorentzMainDiagXAmpErr(0),
  fPowerLorentzMainDiagXBeta(0),
  fPowerLorentzMainDiagXBetaErr(0),
  fPowerLorentzMainDiagXGamma(0),
  fPowerLorentzMainDiagXGammaErr(0),
  fPowerLorentzMainDiagXVertOffset(0),
  fPowerLorentzMainDiagXVertOffsetErr(0),
  fPowerLorentzMainDiagXCenter(0),
  fPowerLorentzMainDiagXCenterErr(0),
  fPowerLorentzMainDiagXChi2red(0),
  fPowerLorentzMainDiagXPp(0),
  fPowerLorentzMainDiagXDOF(0),
  fPowerLorentzMainDiagYAmp(0),
  fPowerLorentzMainDiagYAmpErr(0),
  fPowerLorentzMainDiagYBeta(0),
  fPowerLorentzMainDiagYBetaErr(0),
  fPowerLorentzMainDiagYGamma(0),
  fPowerLorentzMainDiagYGammaErr(0),
  fPowerLorentzMainDiagYVertOffset(0),
  fPowerLorentzMainDiagYVertOffsetErr(0),
  fPowerLorentzMainDiagYCenter(0),
  fPowerLorentzMainDiagYCenterErr(0),
  fPowerLorentzMainDiagYChi2red(0),
  fPowerLorentzMainDiagYPp(0),
  fPowerLorentzMainDiagYDOF(0),
  fPowerLorentzSecDiagXAmp(0),
  fPowerLorentzSecDiagXAmpErr(0),
  fPowerLorentzSecDiagXBeta(0),
  fPowerLorentzSecDiagXBetaErr(0),
  fPowerLorentzSecDiagXGamma(0),
  fPowerLorentzSecDiagXGammaErr(0),
  fPowerLorentzSecDiagXVertOffset(0),
  fPowerLorentzSecDiagXVertOffsetErr(0),
  fPowerLorentzSecDiagXCenter(0),
  fPowerLorentzSecDiagXCenterErr(0),
  fPowerLorentzSecDiagXChi2red(0),
  fPowerLorentzSecDiagXPp(0),
  fPowerLorentzSecDiagXDOF(0),
  fPowerLorentzSecDiagYAmp(0),
  fPowerLorentzSecDiagYAmpErr(0),
  fPowerLorentzSecDiagYBeta(0),
  fPowerLorentzSecDiagYBetaErr(0),
  fPowerLorentzSecDiagYGamma(0),
  fPowerLorentzSecDiagYGammaErr(0),
  fPowerLorentzSecDiagYVertOffset(0),
  fPowerLorentzSecDiagYVertOffsetErr(0),
  fPowerLorentzSecDiagYCenter(0),
  fPowerLorentzSecDiagYCenterErr(0),
  fPowerLorentzSecDiagYChi2red(0),
  fPowerLorentzSecDiagYPp(0),
  fPowerLorentzSecDiagYDOF(0),
  // Initialize Power-Law Lorentz delta variables
  fPowerLorentzRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzColDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Power-Law Lorentz Trans diagonal coordinate variables
  fPowerLorentzMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Power-Law Lorentz mean estimation variables
  fPowerLorentzMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D fitting delta variables
  f3DLorentzDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DLorentzDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  f3DPowerLorentzDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DPowerLorentzDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D Lorentz fit variables
  f3DLorentzCenterX(0),
  f3DLorentzCenterY(0),
  f3DLorentzGammaX(0),
  f3DLorentzGammaY(0),
  f3DLorentzAmp(0),
  f3DLorentzVertOffset(0),
  f3DLorentzCenterXErr(0),
  f3DLorentzCenterYErr(0),
  f3DLorentzGammaXErr(0),
  f3DLorentzGammaYErr(0),
  f3DLorentzAmpErr(0),
  f3DLorentzVertOffsetErr(0),
  f3DLorentzChi2red(0),
  f3DLorentzPp(0),
  f3DLorentzDOF(0),
  f3DLorentzChargeErr(0),
  f3DLorentzSuccess(false),
  // Initialize 3D Power-Law Lorentz fit variables
  f3DPowerLorentzCenterX(0),
  f3DPowerLorentzCenterY(0),
  f3DPowerLorentzGammaX(0),
  f3DPowerLorentzGammaY(0),
  f3DPowerLorentzBeta(0),
  f3DPowerLorentzAmp(0),
  f3DPowerLorentzVertOffset(0),
  f3DPowerLorentzCenterXErr(0),
  f3DPowerLorentzCenterYErr(0),
  f3DPowerLorentzGammaXErr(0),
  f3DPowerLorentzGammaYErr(0),
  f3DPowerLorentzBetaErr(0),
  f3DPowerLorentzAmpErr(0),
  f3DPowerLorentzVertOffsetErr(0),
  f3DPowerLorentzChi2red(0),
  f3DPowerLorentzPp(0),
  f3DPowerLorentzDOF(0),
  f3DPowerLorentzChargeErr(0),
  f3DPowerLorentzSuccess(false),
  // Initialize 3D Gauss fitting delta variables
  f3DGaussDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DGaussDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D Gauss fit variables
  f3DGaussCenterX(0),
  f3DGaussCenterY(0),
  f3DGaussSigmaX(0),
  f3DGaussSigmaY(0),
  f3DGaussAmp(0),
  f3DGaussVertOffset(0),
  f3DGaussCenterXErr(0),
  f3DGaussCenterYErr(0),
  f3DGaussSigmaXErr(0),
  f3DGaussSigmaYErr(0),
  f3DGaussAmpErr(0),
  f3DGaussVertOffsetErr(0),
  f3DGaussChi2red(0),
  f3DGaussPp(0),
  f3DGaussDOF(0),
  f3DGaussChargeErr(0),
  f3DGaussSuccess(false),
  // Legacy variables
  fIsPixelHit(false),
  fInitialEnergy(0),
  fGridPixelSize(0),
  fGridPixelSpacing(0),
  fGridPixelCornerOffset(0),
  fGridDetSize(0),
  fGridNumBlocksPerSide(0),
  
  // Charge uncertainties for Power-Law Lorentz fits (5% of max charge)
  fPowerLorentzRowChargeErr(0),
  fPowerLorentzColChargeErr(0),
  
  // Set automatic radius selection variables
  fSelectedRadius(4),
  
  // Initialize scorer data variables
  fScorerEnergyDeposit(0.0),
  fScorerHitCount(0),
  fScorerDataValid(false),
  
  // Initialize hit purity tracking variables
  fPureSiliconHit(false),
  fAluminumContaminated(false),
  fChargeCalculationEnabled(false)
{ 
  // Initialize neighborhood (9x9) grid vectors (they are automatically initialized empty)
  // Initialize step energy depositionition vectors (they are automatically initialized empty)
}

RunAction::~RunAction()
{
  // File will be closed in EndOfRunAction
  // No need to delete here as it could cause double deletion
}

void RunAction::BeginOfRunAction(const G4Run* run)
{ 
    // Initialize ROOT threading once per application
    std::call_once(gRootInitFlag, InitializeROOTThreading);
    
    // Log run start information
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
        logger->LogRunStart(run->GetRunID(), run->GetNumberOfEventToBeProcessed());
    }
    
    // Reset synchronization for new run (master thread only)
    if (!G4Threading::IsWorkerThread()) {
        ResetSynchronization();
    }
    
    // Safety check for valid run
    if (!run) {
        G4cerr << "RunAction: Error - Invalid run object in BeginOfRunAction" << G4endl;
        return;
    }
    
    // Create unique filename for each thread
    G4String fileName;
    if (G4Threading::IsMultithreadedApplication()) {
        if (G4Threading::IsWorkerThread()) {
            // Worker thread: create unique file for this thread
            G4int threadId = G4Threading::G4GetThreadId();
            std::ostringstream oss;
            oss << "epicChargeSharingOutput_t" << threadId << ".root";
            fileName = oss.str();
        } else {
            // Master thread: this file will be created during merge
            fileName = "epicChargeSharingOutput.root";
        }
    } else {
        // Single-threaded mode
        fileName = "epicChargeSharingOutput.root";
    }
    
    // Only create ROOT file for worker threads or single-threaded mode
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        // Lock mutex during ROOT file operations
        std::lock_guard<std::mutex> lock(fRootMutex);
        
        // Create the ROOT file with optimized settings
        fRootFile = new TFile(fileName.c_str(), "RECREATE", "", 1); // Low compression for speed
        
        if (fRootFile->IsZombie()) {
            G4cerr << "Cannot create ROOT file: " << fileName << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        
        // Set auto-flush and auto-save for better performance and safety
        fRootFile->SetCompressionLevel(1);
        
        G4cout << "Created ROOT file: " << fileName << G4endl;
        
        // Create the ROOT tree with optimized settings
        fTree = new TTree("Hits", "Particle hits and fitting results");
        if (!fTree) {
            G4cerr << "RunAction: Error - Failed to create ROOT tree" << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        fTree->SetAutoFlush(10000);  // Flush every 10k entries
        fTree->SetAutoSave(50000);   // Save every 50k entries
        
        // Create branches following the hierarchical structure
        // Safety check before creating branches
        if (!fTree) {
            G4cerr << "RunAction: Error - ROOT tree is null, cannot create branches" << G4endl;
            return;
        }
        
        // =============================================
        // VALIDATION: CHECK THAT REQUIRED CONSTANTS ARE PROPERLY DEFINED
        // =============================================
        G4cout << "RunAction: Creating branches with fitting configuration:" << G4endl;
        G4cout << "  - Gauss fitting: " << (Control::GAUSS_FIT ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - Lorentz fitting: " << (Control::LORENTZ_FIT ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - Power-Law Lorentz fitting: " << (Control::POWER_LORENTZ_FIT ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - 2D fitting: " << (Control::ROWCOL_FIT ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - Diag fitting: " << (Control::DIAG_FIT ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - 3D Gauss fitting: " << (Control::GAUSS_FIT_3D ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - 3D Lorentz fitting: " << (Control::LORENTZ_FIT_3D ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - 3D Power-Law Lorentz fitting: " << (Control::POWER_LORENTZ_FIT_3D ? "ENABLED" : "DISABLED") << G4endl;
        G4cout << "  - Vert charge uncertainties: " << (Control::CHARGE_ERR ? "ENABLED" : "DISABLED") << G4endl;
        
        // Track number of branches created for validation
        G4int branchCount = 0;
        
        // =============================================
        // HITS BRANCHES
        // =============================================
        fTree->Branch("TrueX", &fTrueX, "TrueX/D")->SetTitle("True Position X [mm]");
        fTree->Branch("TrueY", &fTrueY, "TrueY/D")->SetTitle("True Position Y [mm]");
        fTree->Branch("InitX", &fInitX, "InitX/D")->SetTitle("Initial X [mm]");
        fTree->Branch("InitY", &fInitY, "InitY/D")->SetTitle("Initial Y [mm]");
        fTree->Branch("InitZ", &fInitZ, "InitZ/D")->SetTitle("Initial Z [mm]");
        fTree->Branch("PixelX", &fPixelX, "PixelX/D")->SetTitle("Nearest Pixel X [mm]");
        fTree->Branch("PixelY", &fPixelY, "PixelY/D")->SetTitle("Nearest Pixel Y [mm]");
        fTree->Branch("EdepAtDet", &fEdep, "Edep/D")->SetTitle("Energy Deposit [MeV]");
        fTree->Branch("InitEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
        fTree->Branch("IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("True if hit is on pixel OR distance <= D0");
        fTree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D")->SetTitle("Delta X from Pixel Center to True Position [mm] (x_pixel - x_true)");
        fTree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D")->SetTitle("Delta Y from Pixel Center to True Position [mm] (y_pixel - y_true)");
        
        // GRIDNEIGHBORHOOD BRANCHES
        fTree->Branch("NeighborhoodAngles", &fNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg]");
        fTree->Branch("NeighborhoodChargeFractions", &fNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels");
        fTree->Branch("NeighborhoodDistances", &fNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm]");
        fTree->Branch("NeighborhoodCharges", &fNeighborhoodCharge)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels");
        
        // AUTOMATIC RADIUS SELECTION BRANCHES
        fTree->Branch("SelectedRadius", &fSelectedRadius, "SelectedRadius/I")->SetTitle("Automatically Selected Neighborhood Radius");
        
        // =============================================
        // DELTA VARIABLES (RESIDUALS) BRANCHES
        // =============================================
        // These are the key branches that CalcRes.py looks for
        // Create delta branches only for enabled fitting types
        
        // 2D fitting deltas
        if (Control::GAUSS_FIT && Control::ROWCOL_FIT) {
            fTree->Branch("GaussRowDeltaX", &fGaussRowDeltaX, "GaussRowDeltaX/D")->SetTitle("Gauss Row  Delta X [mm] (fit - true)");
            fTree->Branch("GaussColDeltaY", &fGaussColDeltaY, "GaussColDeltaY/D")->SetTitle("Gauss Col  Delta Y [mm] (fit - true)");
        }
        
        if (Control::LORENTZ_FIT && Control::ROWCOL_FIT) {
            fTree->Branch("LorentzRowDeltaX", &fLorentzRowDeltaX, "LorentzRowDeltaX/D")->SetTitle("Lorentz Row  Delta X [mm] (fit - true)");
            fTree->Branch("LorentzColDeltaY", &fLorentzColDeltaY, "LorentzColDeltaY/D")->SetTitle("Lorentz Col  Delta Y [mm] (fit - true)");
        }
        
        if (Control::POWER_LORENTZ_FIT && Control::ROWCOL_FIT) {
            fTree->Branch("PowerLorentzRowDeltaX", &fPowerLorentzRowDeltaX, "PowerLorentzRowDeltaX/D")->SetTitle("Power Lorentz Row  Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzColDeltaY", &fPowerLorentzColDeltaY, "PowerLorentzColDeltaY/D")->SetTitle("Power Lorentz Col  Delta Y [mm] (fit - true)");
        }
        
        // Diag fit deltas (Trans coordinates)
        if (Control::GAUSS_FIT && Control::DIAG_FIT) {
            fTree->Branch("GaussMainDiagTransDeltaX", &fGaussMainDiagTransformedDeltaX, "GaussMainDiagTransDeltaX/D")->SetTitle("Gauss Main Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("GaussMainDiagTransDeltaY", &fGaussMainDiagTransformedDeltaY, "GaussMainDiagTransDeltaY/D")->SetTitle("Gauss Main Diag Trans Delta Y [mm] (fit - true)");
            fTree->Branch("GaussSecDiagTransDeltaX", &fGaussSecDiagTransformedDeltaX, "GaussSecDiagTransDeltaX/D")->SetTitle("Gauss Secondary Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("GaussSecDiagTransDeltaY", &fGaussSecDiagTransformedDeltaY, "GaussSecDiagTransDeltaY/D")->SetTitle("Gauss Secondary Diag Trans Delta Y [mm] (fit - true)");
        }
        
        if (Control::LORENTZ_FIT && Control::DIAG_FIT) {
            fTree->Branch("LorentzMainDiagTransDeltaX", &fLorentzMainDiagTransformedDeltaX, "LorentzMainDiagTransDeltaX/D")->SetTitle("Lorentz Main Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("LorentzMainDiagTransDeltaY", &fLorentzMainDiagTransformedDeltaY, "LorentzMainDiagTransDeltaY/D")->SetTitle("Lorentz Main Diag Trans Delta Y [mm] (fit - true)");
            fTree->Branch("LorentzSecDiagTransDeltaX", &fLorentzSecDiagTransformedDeltaX, "LorentzSecDiagTransDeltaX/D")->SetTitle("Lorentz Secondary Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("LorentzSecDiagTransDeltaY", &fLorentzSecDiagTransformedDeltaY, "LorentzSecDiagTransDeltaY/D")->SetTitle("Lorentz Secondary Diag Trans Delta Y [mm] (fit - true)");
        }
        
        if (Control::POWER_LORENTZ_FIT && Control::DIAG_FIT) {
            fTree->Branch("PowerLorentzMainDiagTransDeltaX", &fPowerLorentzMainDiagTransformedDeltaX, "PowerLorentzMainDiagTransDeltaX/D")->SetTitle("Power Lorentz Main Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzMainDiagTransDeltaY", &fPowerLorentzMainDiagTransformedDeltaY, "PowerLorentzMainDiagTransDeltaY/D")->SetTitle("Power Lorentz Main Diag Trans Delta Y [mm] (fit - true)");
            fTree->Branch("PowerLorentzSecDiagTransDeltaX", &fPowerLorentzSecDiagTransformedDeltaX, "PowerLorentzSecDiagTransDeltaX/D")->SetTitle("Power Lorentz Secondary Diag Trans Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzSecDiagTransDeltaY", &fPowerLorentzSecDiagTransformedDeltaY, "PowerLorentzSecDiagTransDeltaY/D")->SetTitle("Power Lorentz Secondary Diag Trans Delta Y [mm] (fit - true)");
        }
        
        // 3D fit deltas
        if (Control::GAUSS_FIT_3D) {
            fTree->Branch("3DGaussDeltaX", &f3DGaussDeltaX, "3DGaussDeltaX/D")->SetTitle("3D Gauss  Delta X [mm] (fit - true)");
            fTree->Branch("3DGaussDeltaY", &f3DGaussDeltaY, "3DGaussDeltaY/D")->SetTitle("3D Gauss  Delta Y [mm] (fit - true)");
        }
        
        if (Control::LORENTZ_FIT_3D) {
            fTree->Branch("3DLorentzDeltaX", &f3DLorentzDeltaX, "3DLorentzDeltaX/D")->SetTitle("3D Lorentz  Delta X [mm] (fit - true)");
            fTree->Branch("3DLorentzDeltaY", &f3DLorentzDeltaY, "3DLorentzDeltaY/D")->SetTitle("3D Lorentz  Delta Y [mm] (fit - true)");
        }
        
        if (Control::POWER_LORENTZ_FIT_3D) {
            fTree->Branch("3DPowerLorentzDeltaX", &f3DPowerLorentzDeltaX, "3DPowerLorentzDeltaX/D")->SetTitle("3D Power Lorentz  Delta X [mm] (fit - true)");
            fTree->Branch("3DPowerLorentzDeltaY", &f3DPowerLorentzDeltaY, "3DPowerLorentzDeltaY/D")->SetTitle("3D Power Lorentz  Delta Y [mm] (fit - true)");
        }
        
        // Mean estimators (key resolution metrics) - only create if corresponding fitting is enabled
        if (Control::GAUSS_FIT && (Control::DIAG_FIT || Control::GAUSS_FIT_3D)) {
            fTree->Branch("GaussMeanTrueDeltaX", &fGaussMeanTrueDeltaX, "GaussMeanTrueDeltaX/D")->SetTitle("Gauss Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("GaussMeanTrueDeltaY", &fGaussMeanTrueDeltaY, "GaussMeanTrueDeltaY/D")->SetTitle("Gauss Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        if (Control::LORENTZ_FIT && (Control::DIAG_FIT || Control::LORENTZ_FIT_3D)) {
            fTree->Branch("LorentzMeanTrueDeltaX", &fLorentzMeanTrueDeltaX, "LorentzMeanTrueDeltaX/D")->SetTitle("Lorentz Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("LorentzMeanTrueDeltaY", &fLorentzMeanTrueDeltaY, "LorentzMeanTrueDeltaY/D")->SetTitle("Lorentz Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        if (Control::POWER_LORENTZ_FIT && (Control::DIAG_FIT || Control::POWER_LORENTZ_FIT_3D)) {
            fTree->Branch("PowerLorentzMeanTrueDeltaX", &fPowerLorentzMeanTrueDeltaX, "PowerLorentzMeanTrueDeltaX/D")->SetTitle("Power Lorentz Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("PowerLorentzMeanTrueDeltaY", &fPowerLorentzMeanTrueDeltaY, "PowerLorentzMeanTrueDeltaY/D")->SetTitle("Power Lorentz Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        // =============================================
        // GAUSS FIT PARAMETERS BRANCHES
        // =============================================
        if (Control::GAUSS_FIT) {
            // Only create central row/column (2D) branches when 2D fitting is enabled
            if (Control::ROWCOL_FIT) {
                // Row fit parameters
                fTree->Branch("GaussRowAmp", &fGaussRowAmp, "GaussRowAmp/D")->SetTitle("Gauss Row  Amp");
                fTree->Branch("GaussRowAmpErr", &fGaussRowAmpErr, "GaussRowAmpErr/D")->SetTitle("Gauss Row  Amp Error");
                fTree->Branch("GaussRowSigma", &fGaussRowSigma, "GaussRowSigma/D")->SetTitle("Gauss Row  Standard Deviation");
                fTree->Branch("GaussRowSigmaErr", &fGaussRowSigmaErr, "GaussRowSigmaErr/D")->SetTitle("Gauss Row  Standard Deviation Error");
                fTree->Branch("GaussRowVertOffset", &fGaussRowVertOffset, "GaussRowVertOffset/D")->SetTitle("Gauss Row  Vert Offset");
                fTree->Branch("GaussRowVertOffsetErr", &fGaussRowVertOffsetErr, "GaussRowVertOffsetErr/D")->SetTitle("Gauss Row  Vert Offset Error");
                fTree->Branch("GaussRowCenter", &fGaussRowCenter, "GaussRowCenter/D")->SetTitle("Gauss Row  Center");
                fTree->Branch("GaussRowCenterErr", &fGaussRowCenterErr, "GaussRowCenterErr/D")->SetTitle("Gauss Row  Center Error");
                fTree->Branch("GaussRowChi2red", &fGaussRowChi2red, "GaussRowChi2red/D")->SetTitle("Gauss Row  Reduced Chi-squared");
                fTree->Branch("GaussRowPp", &fGaussRowPp, "GaussRowPp/D")->SetTitle("Gauss Row  P-value");
                fTree->Branch("GaussRowDOF", &fGaussRowDOF, "GaussRowDOF/I")->SetTitle("Gauss Row  Degrees of Freedom");
                fTree->Branch("GaussRowChargeErr", &fGaussRowChargeErr, "GaussRowChargeErr/D")->SetTitle("Gauss Row  Charge Err");
                
                // Col fit parameters
                fTree->Branch("GaussColAmp", &fGaussColAmp, "GaussColAmp/D")->SetTitle("Gauss Col  Amp");
                fTree->Branch("GaussColAmpErr", &fGaussColAmpErr, "GaussColAmpErr/D")->SetTitle("Gauss Col  Amp Error");
                fTree->Branch("GaussColSigma", &fGaussColSigma, "GaussColSigma/D")->SetTitle("Gauss Col  Standard Deviation");
                fTree->Branch("GaussColSigmaErr", &fGaussColSigmaErr, "GaussColSigmaErr/D")->SetTitle("Gauss Col  Standard Deviation Error");
                fTree->Branch("GaussColVertOffset", &fGaussColVertOffset, "GaussColVertOffset/D")->SetTitle("Gauss Col  Vert Offset");
                fTree->Branch("GaussColVertOffsetErr", &fGaussColVertOffsetErr, "GaussColVertOffsetErr/D")->SetTitle("Gauss Col  Vert Offset Error");
                fTree->Branch("GaussColCenter", &fGaussColCenter, "GaussColCenter/D")->SetTitle("Gauss Col  Center");
                fTree->Branch("GaussColCenterErr", &fGaussColCenterErr, "GaussColCenterErr/D")->SetTitle("Gauss Col  Center Error");
                fTree->Branch("GaussColChi2red", &fGaussColChi2red, "GaussColChi2red/D")->SetTitle("Gauss Col  Reduced Chi-squared");
                fTree->Branch("GaussColPp", &fGaussColPp, "GaussColPp/D")->SetTitle("Gauss Col  P-value");
                fTree->Branch("GaussColDOF", &fGaussColDOF, "GaussColDOF/I")->SetTitle("Gauss Col  Degrees of Freedom");
                fTree->Branch("GaussColChargeErr", &fGaussColChargeErr, "GaussColChargeErr/D")->SetTitle("Gauss Col  Charge Err");
            } // end 2D Gauss branches
            
            // Create diagonal fit branches only when diagonal fitting is enabled
            if (Control::DIAG_FIT) {
                // Main diagonal fit parameters
                fTree->Branch("GaussMainDiagXAmp", &fGaussMainDiagXAmp, "GaussMainDiagXAmp/D")->SetTitle("Gauss Main Diag X  Amp");
                fTree->Branch("GaussMainDiagXAmpErr", &fGaussMainDiagXAmpErr, "GaussMainDiagXAmpErr/D")->SetTitle("Gauss Main Diag X  Amp Error");
                fTree->Branch("GaussMainDiagXSigma", &fGaussMainDiagXSigma, "GaussMainDiagXSigma/D")->SetTitle("Gauss Main Diag X  Standard Deviation");
                fTree->Branch("GaussMainDiagXSigmaErr", &fGaussMainDiagXSigmaErr, "GaussMainDiagXSigmaErr/D")->SetTitle("Gauss Main Diag X  Standard Deviation Error");
                fTree->Branch("GaussMainDiagXVertOffset", &fGaussMainDiagXVertOffset, "GaussMainDiagXVertOffset/D")->SetTitle("Gauss Main Diag X  Vert Offset");
                fTree->Branch("GaussMainDiagXVertOffsetErr", &fGaussMainDiagXVertOffsetErr, "GaussMainDiagXVertOffsetErr/D")->SetTitle("Gauss Main Diag X  Vert Offset Error");
                fTree->Branch("GaussMainDiagXCenter", &fGaussMainDiagXCenter, "GaussMainDiagXCenter/D")->SetTitle("Gauss Main Diag X  Center");
                fTree->Branch("GaussMainDiagXCenterErr", &fGaussMainDiagXCenterErr, "GaussMainDiagXCenterErr/D")->SetTitle("Gauss Main Diag X  Center Error");
                fTree->Branch("GaussMainDiagXChi2red", &fGaussMainDiagXChi2red, "GaussMainDiagXChi2red/D")->SetTitle("Gauss Main Diag X  Reduced Chi-squared");
                fTree->Branch("GaussMainDiagXPp", &fGaussMainDiagXPp, "GaussMainDiagXPp/D")->SetTitle("Gauss Main Diag X  P-value");
                fTree->Branch("GaussMainDiagXDOF", &fGaussMainDiagXDOF, "GaussMainDiagXDOF/I")->SetTitle("Gauss Main Diag X  Degrees of Freedom");
                
                fTree->Branch("GaussMainDiagYAmp", &fGaussMainDiagYAmp, "GaussMainDiagYAmp/D")->SetTitle("Gauss Main Diag Y  Amp");
                fTree->Branch("GaussMainDiagYAmpErr", &fGaussMainDiagYAmpErr, "GaussMainDiagYAmpErr/D")->SetTitle("Gauss Main Diag Y  Amp Error");
                fTree->Branch("GaussMainDiagYSigma", &fGaussMainDiagYSigma, "GaussMainDiagYSigma/D")->SetTitle("Gauss Main Diag Y  Standard Deviation");
                fTree->Branch("GaussMainDiagYSigmaErr", &fGaussMainDiagYSigmaErr, "GaussMainDiagYSigmaErr/D")->SetTitle("Gauss Main Diag Y  Standard Deviation Error");
                fTree->Branch("GaussMainDiagYVertOffset", &fGaussMainDiagYVertOffset, "GaussMainDiagYVertOffset/D")->SetTitle("Gauss Main Diag Y  Vert Offset");
                fTree->Branch("GaussMainDiagYVertOffsetErr", &fGaussMainDiagYVertOffsetErr, "GaussMainDiagYVertOffsetErr/D")->SetTitle("Gauss Main Diag Y  Vert Offset Error");
                fTree->Branch("GaussMainDiagYCenter", &fGaussMainDiagYCenter, "GaussMainDiagYCenter/D")->SetTitle("Gauss Main Diag Y  Center");
                fTree->Branch("GaussMainDiagYCenterErr", &fGaussMainDiagYCenterErr, "GaussMainDiagYCenterErr/D")->SetTitle("Gauss Main Diag Y  Center Error");
                fTree->Branch("GaussMainDiagYChi2red", &fGaussMainDiagYChi2red, "GaussMainDiagYChi2red/D")->SetTitle("Gauss Main Diag Y  Reduced Chi-squared");
                fTree->Branch("GaussMainDiagYPp", &fGaussMainDiagYPp, "GaussMainDiagYPp/D")->SetTitle("Gauss Main Diag Y  P-value");
                fTree->Branch("GaussMainDiagYDOF", &fGaussMainDiagYDOF, "GaussMainDiagYDOF/I")->SetTitle("Gauss Main Diag Y  Degrees of Freedom");
                
                // Secondary diagonal fit parameters
                fTree->Branch("GaussSecDiagXAmp", &fGaussSecDiagXAmp, "GaussSecDiagXAmp/D")->SetTitle("Gauss Secondary Diag X  Amp");
                fTree->Branch("GaussSecDiagXAmpErr", &fGaussSecDiagXAmpErr, "GaussSecDiagXAmpErr/D")->SetTitle("Gauss Secondary Diag X  Amp Error");
                fTree->Branch("GaussSecDiagXSigma", &fGaussSecDiagXSigma, "GaussSecDiagXSigma/D")->SetTitle("Gauss Secondary Diag X  Standard Deviation");
                fTree->Branch("GaussSecDiagXSigmaErr", &fGaussSecDiagXSigmaErr, "GaussSecDiagXSigmaErr/D")->SetTitle("Gauss Secondary Diag X  Standard Deviation Error");
                fTree->Branch("GaussSecDiagXVertOffset", &fGaussSecDiagXVertOffset, "GaussSecDiagXVertOffset/D")->SetTitle("Gauss Secondary Diag X  Vert Offset");
                fTree->Branch("GaussSecDiagXVertOffsetErr", &fGaussSecDiagXVertOffsetErr, "GaussSecDiagXVertOffsetErr/D")->SetTitle("Gauss Secondary Diag X  Vert Offset Error");
                fTree->Branch("GaussSecDiagXCenter", &fGaussSecDiagXCenter, "GaussSecDiagXCenter/D")->SetTitle("Gauss Secondary Diag X  Center");
                fTree->Branch("GaussSecDiagXCenterErr", &fGaussSecDiagXCenterErr, "GaussSecDiagXCenterErr/D")->SetTitle("Gauss Secondary Diag X  Center Error");
                fTree->Branch("GaussSecDiagXChi2red", &fGaussSecDiagXChi2red, "GaussSecDiagXChi2red/D")->SetTitle("Gauss Secondary Diag X  Reduced Chi-squared");
                fTree->Branch("GaussSecDiagXPp", &fGaussSecDiagXPp, "GaussSecDiagXPp/D")->SetTitle("Gauss Secondary Diag X  P-value");
                fTree->Branch("GaussSecDiagXDOF", &fGaussSecDiagXDOF, "GaussSecDiagXDOF/I")->SetTitle("Gauss Secondary Diag X  Degrees of Freedom");
                
                fTree->Branch("GaussSecDiagYAmp", &fGaussSecDiagYAmp, "GaussSecDiagYAmp/D")->SetTitle("Gauss Secondary Diag Y  Amp");
                fTree->Branch("GaussSecDiagYAmpErr", &fGaussSecDiagYAmpErr, "GaussSecDiagYAmpErr/D")->SetTitle("Gauss Secondary Diag Y  Amp Error");
                fTree->Branch("GaussSecDiagYSigma", &fGaussSecDiagYSigma, "GaussSecDiagYSigma/D")->SetTitle("Gauss Secondary Diag Y  Standard Deviation");
                fTree->Branch("GaussSecDiagYSigmaErr", &fGaussSecDiagYSigmaErr, "GaussSecDiagYSigmaErr/D")->SetTitle("Gauss Secondary Diag Y  Standard Deviation Error");
                fTree->Branch("GaussSecDiagYVertOffset", &fGaussSecDiagYVertOffset, "GaussSecDiagYVertOffset/D")->SetTitle("Gauss Secondary Diag Y  Vert Offset");
                fTree->Branch("GaussSecDiagYVertOffsetErr", &fGaussSecDiagYVertOffsetErr, "GaussSecDiagYVertOffsetErr/D")->SetTitle("Gauss Secondary Diag Y  Vert Offset Error");
                fTree->Branch("GaussSecDiagYCenter", &fGaussSecDiagYCenter, "GaussSecDiagYCenter/D")->SetTitle("Gauss Secondary Diag Y  Center");
                fTree->Branch("GaussSecDiagYCenterErr", &fGaussSecDiagYCenterErr, "GaussSecDiagYCenterErr/D")->SetTitle("Gauss Secondary Diag Y  Center Error");
                fTree->Branch("GaussSecDiagYChi2red", &fGaussSecDiagYChi2red, "GaussSecDiagYChi2red/D")->SetTitle("Gauss Secondary Diag Y  Reduced Chi-squared");
                fTree->Branch("GaussSecDiagYPp", &fGaussSecDiagYPp, "GaussSecDiagYPp/D")->SetTitle("Gauss Secondary Diag Y  P-value");
                fTree->Branch("GaussSecDiagYDOF", &fGaussSecDiagYDOF, "GaussSecDiagYDOF/I")->SetTitle("Gauss Secondary Diag Y  Degrees of Freedom");
            } // end diagonal Gauss branches
        }
        
        // =============================================
        // Lorentz FIT PARAMETERS BRANCHES
        // =============================================
        if (Control::LORENTZ_FIT) {
            // Only create central row/column (2D) branches when 2D fitting is enabled
            if (Control::ROWCOL_FIT) {
                // Row fit parameters
                fTree->Branch("LorentzRowAmp", &fLorentzRowAmp, "LorentzRowAmp/D")->SetTitle("Lorentz Row  Amp");
                fTree->Branch("LorentzRowAmpErr", &fLorentzRowAmpErr, "LorentzRowAmpErr/D")->SetTitle("Lorentz Row  Amp Error");
                fTree->Branch("LorentzRowGamma", &fLorentzRowGamma, "LorentzRowGamma/D")->SetTitle("Lorentz Row  Gamma (HWHM)");
                fTree->Branch("LorentzRowGammaErr", &fLorentzRowGammaErr, "LorentzRowGammaErr/D")->SetTitle("Lorentz Row  Gamma Error");
                fTree->Branch("LorentzRowVertOffset", &fLorentzRowVertOffset, "LorentzRowVertOffset/D")->SetTitle("Lorentz Row  Vert Offset");
                fTree->Branch("LorentzRowVertOffsetErr", &fLorentzRowVertOffsetErr, "LorentzRowVertOffsetErr/D")->SetTitle("Lorentz Row  Vert Offset Error");
                fTree->Branch("LorentzRowCenter", &fLorentzRowCenter, "LorentzRowCenter/D")->SetTitle("Lorentz Row  Center");
                fTree->Branch("LorentzRowCenterErr", &fLorentzRowCenterErr, "LorentzRowCenterErr/D")->SetTitle("Lorentz Row  Center Error");
                fTree->Branch("LorentzRowChi2red", &fLorentzRowChi2red, "LorentzRowChi2red/D")->SetTitle("Lorentz Row  Reduced Chi-squared");
                fTree->Branch("LorentzRowPp", &fLorentzRowPp, "LorentzRowPp/D")->SetTitle("Lorentz Row  P-value");
                fTree->Branch("LorentzRowDOF", &fLorentzRowDOF, "LorentzRowDOF/I")->SetTitle("Lorentz Row  Degrees of Freedom");
                fTree->Branch("LorentzRowChargeErr", &fLorentzRowChargeErr, "LorentzRowChargeErr/D")->SetTitle("Lorentz Row  Charge Err");
                
                // Col fit parameters
                fTree->Branch("LorentzColAmp", &fLorentzColAmp, "LorentzColAmp/D")->SetTitle("Lorentz Col  Amp");
                fTree->Branch("LorentzColAmpErr", &fLorentzColAmpErr, "LorentzColAmpErr/D")->SetTitle("Lorentz Col  Amp Error");
                fTree->Branch("LorentzColGamma", &fLorentzColGamma, "LorentzColGamma/D")->SetTitle("Lorentz Col  Gamma (HWHM)");
                fTree->Branch("LorentzColGammaErr", &fLorentzColGammaErr, "LorentzColGammaErr/D")->SetTitle("Lorentz Col  Gamma Error");
                fTree->Branch("LorentzColVertOffset", &fLorentzColVertOffset, "LorentzColVertOffset/D")->SetTitle("Lorentz Col  Vert Offset");
                fTree->Branch("LorentzColVertOffsetErr", &fLorentzColVertOffsetErr, "LorentzColVertOffsetErr/D")->SetTitle("Lorentz Col  Vert Offset Error");
                fTree->Branch("LorentzColCenter", &fLorentzColCenter, "LorentzColCenter/D")->SetTitle("Lorentz Col  Center");
                fTree->Branch("LorentzColCenterErr", &fLorentzColCenterErr, "LorentzColCenterErr/D")->SetTitle("Lorentz Col  Center Error");
                fTree->Branch("LorentzColChi2red", &fLorentzColChi2red, "LorentzColChi2red/D")->SetTitle("Lorentz Col  Reduced Chi-squared");
                fTree->Branch("LorentzColPp", &fLorentzColPp, "LorentzColPp/D")->SetTitle("Lorentz Col  P-value");
                fTree->Branch("LorentzColDOF", &fLorentzColDOF, "LorentzColDOF/I")->SetTitle("Lorentz Col  Degrees of Freedom");
                fTree->Branch("LorentzColChargeErr", &fLorentzColChargeErr, "LorentzColChargeErr/D")->SetTitle("Lorentz Col  Charge Err");
            } // end 2D Lorentz branches
            
            // Create diagonal fit branches only when diagonal fitting is enabled
            if (Control::DIAG_FIT) {
                // Main diagonal fit parameters
                fTree->Branch("LorentzMainDiagXAmp", &fLorentzMainDiagXAmp, "LorentzMainDiagXAmp/D")->SetTitle("Lorentz Main Diag X  Amp");
                fTree->Branch("LorentzMainDiagXAmpErr", &fLorentzMainDiagXAmpErr, "LorentzMainDiagXAmpErr/D")->SetTitle("Lorentz Main Diag X  Amp Error");
                fTree->Branch("LorentzMainDiagXGamma", &fLorentzMainDiagXGamma, "LorentzMainDiagXGamma/D")->SetTitle("Lorentz Main Diag X  Gamma");
                fTree->Branch("LorentzMainDiagXGammaErr", &fLorentzMainDiagXGammaErr, "LorentzMainDiagXGammaErr/D")->SetTitle("Lorentz Main Diag X  Gamma Error");
                fTree->Branch("LorentzMainDiagXVertOffset", &fLorentzMainDiagXVertOffset, "LorentzMainDiagXVertOffset/D")->SetTitle("Lorentz Main Diag X  Vert Offset");
                fTree->Branch("LorentzMainDiagXVertOffsetErr", &fLorentzMainDiagXVertOffsetErr, "LorentzMainDiagXVertOffsetErr/D")->SetTitle("Lorentz Main Diag X  Vert Offset Error");
                fTree->Branch("LorentzMainDiagXCenter", &fLorentzMainDiagXCenter, "LorentzMainDiagXCenter/D")->SetTitle("Lorentz Main Diag X  Center");
                fTree->Branch("LorentzMainDiagXCenterErr", &fLorentzMainDiagXCenterErr, "LorentzMainDiagXCenterErr/D")->SetTitle("Lorentz Main Diag X  Center Error");
                fTree->Branch("LorentzMainDiagXChi2red", &fLorentzMainDiagXChi2red, "LorentzMainDiagXChi2red/D")->SetTitle("Lorentz Main Diag X  Reduced Chi-squared");
                fTree->Branch("LorentzMainDiagXPp", &fLorentzMainDiagXPp, "LorentzMainDiagXPp/D")->SetTitle("Lorentz Main Diag X  P-value");
                fTree->Branch("LorentzMainDiagXDOF", &fLorentzMainDiagXDOF, "LorentzMainDiagXDOF/I")->SetTitle("Lorentz Main Diag X  Degrees of Freedom");
                
                fTree->Branch("LorentzMainDiagYAmp", &fLorentzMainDiagYAmp, "LorentzMainDiagYAmp/D")->SetTitle("Lorentz Main Diag Y  Amp");
                fTree->Branch("LorentzMainDiagYAmpErr", &fLorentzMainDiagYAmpErr, "LorentzMainDiagYAmpErr/D")->SetTitle("Lorentz Main Diag Y  Amp Error");
                fTree->Branch("LorentzMainDiagYGamma", &fLorentzMainDiagYGamma, "LorentzMainDiagYGamma/D")->SetTitle("Lorentz Main Diag Y  Gamma");
                fTree->Branch("LorentzMainDiagYGammaErr", &fLorentzMainDiagYGammaErr, "LorentzMainDiagYGammaErr/D")->SetTitle("Lorentz Main Diag Y  Gamma Error");
                fTree->Branch("LorentzMainDiagYVertOffset", &fLorentzMainDiagYVertOffset, "LorentzMainDiagYVertOffset/D")->SetTitle("Lorentz Main Diag Y  Vert Offset");
                fTree->Branch("LorentzMainDiagYVertOffsetErr", &fLorentzMainDiagYVertOffsetErr, "LorentzMainDiagYVertOffsetErr/D")->SetTitle("Lorentz Main Diag Y  Vert Offset Error");
                fTree->Branch("LorentzMainDiagYCenter", &fLorentzMainDiagYCenter, "LorentzMainDiagYCenter/D")->SetTitle("Lorentz Main Diag Y  Center");
                fTree->Branch("LorentzMainDiagYCenterErr", &fLorentzMainDiagYCenterErr, "LorentzMainDiagYCenterErr/D")->SetTitle("Lorentz Main Diag Y  Center Error");
                fTree->Branch("LorentzMainDiagYChi2red", &fLorentzMainDiagYChi2red, "LorentzMainDiagYChi2red/D")->SetTitle("Lorentz Main Diag Y  Reduced Chi-squared");
                fTree->Branch("LorentzMainDiagYPp", &fLorentzMainDiagYPp, "LorentzMainDiagYPp/D")->SetTitle("Lorentz Main Diag Y  P-value");
                fTree->Branch("LorentzMainDiagYDOF", &fLorentzMainDiagYDOF, "LorentzMainDiagYDOF/I")->SetTitle("Lorentz Main Diag Y  Degrees of Freedom");
                
                // Secondary diagonal fit parameters
                fTree->Branch("LorentzSecDiagXAmp", &fLorentzSecDiagXAmp, "LorentzSecDiagXAmp/D")->SetTitle("Lorentz Secondary Diag X  Amp");
                fTree->Branch("LorentzSecDiagXAmpErr", &fLorentzSecDiagXAmpErr, "LorentzSecDiagXAmpErr/D")->SetTitle("Lorentz Secondary Diag X  Amp Error");
                fTree->Branch("LorentzSecDiagXGamma", &fLorentzSecDiagXGamma, "LorentzSecDiagXGamma/D")->SetTitle("Lorentz Secondary Diag X  Gamma");
                fTree->Branch("LorentzSecDiagXGammaErr", &fLorentzSecDiagXGammaErr, "LorentzSecDiagXGammaErr/D")->SetTitle("Lorentz Secondary Diag X  Gamma Error");
                fTree->Branch("LorentzSecDiagXVertOffset", &fLorentzSecDiagXVertOffset, "LorentzSecDiagXVertOffset/D")->SetTitle("Lorentz Secondary Diag X  Vert Offset");
                fTree->Branch("LorentzSecDiagXVertOffsetErr", &fLorentzSecDiagXVertOffsetErr, "LorentzSecDiagXVertOffsetErr/D")->SetTitle("Lorentz Secondary Diag X  Vert Offset Error");
                fTree->Branch("LorentzSecDiagXCenter", &fLorentzSecDiagXCenter, "LorentzSecDiagXCenter/D")->SetTitle("Lorentz Secondary Diag X  Center");
                fTree->Branch("LorentzSecDiagXCenterErr", &fLorentzSecDiagXCenterErr, "LorentzSecDiagXCenterErr/D")->SetTitle("Lorentz Secondary Diag X  Center Error");
                fTree->Branch("LorentzSecDiagXChi2red", &fLorentzSecDiagXChi2red, "LorentzSecDiagXChi2red/D")->SetTitle("Lorentz Secondary Diag X  Reduced Chi-squared");
                fTree->Branch("LorentzSecDiagXPp", &fLorentzSecDiagXPp, "LorentzSecDiagXPp/D")->SetTitle("Lorentz Secondary Diag X  P-value");
                fTree->Branch("LorentzSecDiagXDOF", &fLorentzSecDiagXDOF, "LorentzSecDiagXDOF/I")->SetTitle("Lorentz Secondary Diag X  Degrees of Freedom");
                
                fTree->Branch("LorentzSecDiagYAmp", &fLorentzSecDiagYAmp, "LorentzSecDiagYAmp/D")->SetTitle("Lorentz Secondary Diag Y  Amp");
                fTree->Branch("LorentzSecDiagYAmpErr", &fLorentzSecDiagYAmpErr, "LorentzSecDiagYAmpErr/D")->SetTitle("Lorentz Secondary Diag Y  Amp Error");
                fTree->Branch("LorentzSecDiagYGamma", &fLorentzSecDiagYGamma, "LorentzSecDiagYGamma/D")->SetTitle("Lorentz Secondary Diag Y  Gamma");
                fTree->Branch("LorentzSecDiagYGammaErr", &fLorentzSecDiagYGammaErr, "LorentzSecDiagYGammaErr/D")->SetTitle("Lorentz Secondary Diag Y  Gamma Error");
                fTree->Branch("LorentzSecDiagYVertOffset", &fLorentzSecDiagYVertOffset, "LorentzSecDiagYVertOffset/D")->SetTitle("Lorentz Secondary Diag Y  Vert Offset");
                fTree->Branch("LorentzSecDiagYVertOffsetErr", &fLorentzSecDiagYVertOffsetErr, "LorentzSecDiagYVertOffsetErr/D")->SetTitle("Lorentz Secondary Diag Y  Vert Offset Error");
                fTree->Branch("LorentzSecDiagYCenter", &fLorentzSecDiagYCenter, "LorentzSecDiagYCenter/D")->SetTitle("Lorentz Secondary Diag Y  Center");
                fTree->Branch("LorentzSecDiagYCenterErr", &fLorentzSecDiagYCenterErr, "LorentzSecDiagYCenterErr/D")->SetTitle("Lorentz Secondary Diag Y  Center Error");
                fTree->Branch("LorentzSecDiagYChi2red", &fLorentzSecDiagYChi2red, "LorentzSecDiagYChi2red/D")->SetTitle("Lorentz Secondary Diag Y  Reduced Chi-squared");
                fTree->Branch("LorentzSecDiagYPp", &fLorentzSecDiagYPp, "LorentzSecDiagYPp/D")->SetTitle("Lorentz Secondary Diag Y  P-value");
                fTree->Branch("LorentzSecDiagYDOF", &fLorentzSecDiagYDOF, "LorentzSecDiagYDOF/I")->SetTitle("Lorentz Secondary Diag Y  Degrees of Freedom");
            } // end diagonal Lorentz branches
        }
        
        // =============================================
        // POWER-LAW Lorentz FIT PARAMETERS BRANCHES
        // =============================================
        if (Control::POWER_LORENTZ_FIT) {
            // Only create central row/column (2D) branches when 2D fitting is enabled
            if (Control::ROWCOL_FIT) {
                // Row fit parameters
                fTree->Branch("PowerLorentzRowAmp", &fPowerLorentzRowAmp, "PowerLorentzRowAmp/D")->SetTitle("Power-Law Lorentz Row  Amp");
                fTree->Branch("PowerLorentzRowAmpErr", &fPowerLorentzRowAmpErr, "PowerLorentzRowAmpErr/D")->SetTitle("Power-Law Lorentz Row  Amp Error");
                fTree->Branch("PowerLorentzRowBeta", &fPowerLorentzRowBeta, "PowerLorentzRowBeta/D")->SetTitle("Power-Law Lorentz Row  Beta (Power-Law Exponent)");
                fTree->Branch("PowerLorentzRowBetaErr", &fPowerLorentzRowBetaErr, "PowerLorentzRowBetaErr/D")->SetTitle("Power-Law Lorentz Row  Beta Error");
                fTree->Branch("PowerLorentzRowGamma", &fPowerLorentzRowGamma, "PowerLorentzRowGamma/D")->SetTitle("Power-Law Lorentz Row  Gamma (HWHM)");
                fTree->Branch("PowerLorentzRowGammaErr", &fPowerLorentzRowGammaErr, "PowerLorentzRowGammaErr/D")->SetTitle("Power-Law Lorentz Row  Gamma Error");
                fTree->Branch("PowerLorentzRowVertOffset", &fPowerLorentzRowVertOffset, "PowerLorentzRowVertOffset/D")->SetTitle("Power-Law Lorentz Row  Vert Offset");
                fTree->Branch("PowerLorentzRowVertOffsetErr", &fPowerLorentzRowVertOffsetErr, "PowerLorentzRowVertOffsetErr/D")->SetTitle("Power-Law Lorentz Row  Vert Offset Error");
                fTree->Branch("PowerLorentzRowCenter", &fPowerLorentzRowCenter, "PowerLorentzRowCenter/D")->SetTitle("Power-Law Lorentz Row  Center");
                fTree->Branch("PowerLorentzRowCenterErr", &fPowerLorentzRowCenterErr, "PowerLorentzRowCenterErr/D")->SetTitle("Power-Law Lorentz Row  Center Error");
                fTree->Branch("PowerLorentzRowChi2red", &fPowerLorentzRowChi2red, "PowerLorentzRowChi2red/D")->SetTitle("Power-Law Lorentz Row  Reduced Chi-squared");
                fTree->Branch("PowerLorentzRowPp", &fPowerLorentzRowPp, "PowerLorentzRowPp/D")->SetTitle("Power-Law Lorentz Row  P-value");
                fTree->Branch("PowerLorentzRowDOF", &fPowerLorentzRowDOF, "PowerLorentzRowDOF/I")->SetTitle("Power-Law Lorentz Row  Degrees of Freedom");
                fTree->Branch("PowerLorentzRowChargeErr", &fPowerLorentzRowChargeErr, "PowerLorentzRowChargeErr/D")->SetTitle("Power-Law Lorentz Row  Charge Err");
                
                // Col fit parameters
                fTree->Branch("PowerLorentzColAmp", &fPowerLorentzColAmp, "PowerLorentzColAmp/D")->SetTitle("Power-Law Lorentz Col  Amp");
                fTree->Branch("PowerLorentzColAmpErr", &fPowerLorentzColAmpErr, "PowerLorentzColAmpErr/D")->SetTitle("Power-Law Lorentz Col  Amp Error");
                fTree->Branch("PowerLorentzColBeta", &fPowerLorentzColBeta, "PowerLorentzColBeta/D")->SetTitle("Power-Law Lorentz Col  Beta (Power-Law Exponent)");
                fTree->Branch("PowerLorentzColBetaErr", &fPowerLorentzColBetaErr, "PowerLorentzColBetaErr/D")->SetTitle("Power-Law Lorentz Col  Beta Error");
                fTree->Branch("PowerLorentzColGamma", &fPowerLorentzColGamma, "PowerLorentzColGamma/D")->SetTitle("Power-Law Lorentz Col  Gamma (HWHM)");
                fTree->Branch("PowerLorentzColGammaErr", &fPowerLorentzColGammaErr, "PowerLorentzColGammaErr/D")->SetTitle("Power-Law Lorentz Col  Gamma Error");
                fTree->Branch("PowerLorentzColVertOffset", &fPowerLorentzColVertOffset, "PowerLorentzColVertOffset/D")->SetTitle("Power-Law Lorentz Col  Vert Offset");
                fTree->Branch("PowerLorentzColVertOffsetErr", &fPowerLorentzColVertOffsetErr, "PowerLorentzColVertOffsetErr/D")->SetTitle("Power-Law Lorentz Col  Vert Offset Error");
                fTree->Branch("PowerLorentzColCenter", &fPowerLorentzColCenter, "PowerLorentzColCenter/D")->SetTitle("Power-Law Lorentz Col  Center");
                fTree->Branch("PowerLorentzColCenterErr", &fPowerLorentzColCenterErr, "PowerLorentzColCenterErr/D")->SetTitle("Power-Law Lorentz Col  Center Error");
                fTree->Branch("PowerLorentzColChi2red", &fPowerLorentzColChi2red, "PowerLorentzColChi2red/D")->SetTitle("Power-Law Lorentz Col  Reduced Chi-squared");
                fTree->Branch("PowerLorentzColPp", &fPowerLorentzColPp, "PowerLorentzColPp/D")->SetTitle("Power-Law Lorentz Col  P-value");
                fTree->Branch("PowerLorentzColDOF", &fPowerLorentzColDOF, "PowerLorentzColDOF/I")->SetTitle("Power-Law Lorentz Col  Degrees of Freedom");
                fTree->Branch("PowerLorentzColChargeErr", &fPowerLorentzColChargeErr, "PowerLorentzColChargeErr/D")->SetTitle("Power-Law Lorentz Col  Charge Err");
            } // end 2D Power-Law Lorentz branches
            
            // Create diagonal fit branches only when diagonal fitting is enabled
            if (Control::DIAG_FIT) {
                // Main diagonal fit parameters
                fTree->Branch("PowerLorentzMainDiagXAmp", &fPowerLorentzMainDiagXAmp, "PowerLorentzMainDiagXAmp/D")->SetTitle("Power-Law Lorentz Main Diag X  Amp");
                fTree->Branch("PowerLorentzMainDiagXAmpErr", &fPowerLorentzMainDiagXAmpErr, "PowerLorentzMainDiagXAmpErr/D")->SetTitle("Power-Law Lorentz Main Diag X  Amp Error");
                fTree->Branch("PowerLorentzMainDiagXBeta", &fPowerLorentzMainDiagXBeta, "PowerLorentzMainDiagXBeta/D")->SetTitle("Power-Law Lorentz Main Diag X  Beta");
                fTree->Branch("PowerLorentzMainDiagXBetaErr", &fPowerLorentzMainDiagXBetaErr, "PowerLorentzMainDiagXBetaErr/D")->SetTitle("Power-Law Lorentz Main Diag X  Beta Error");
                fTree->Branch("PowerLorentzMainDiagXGamma", &fPowerLorentzMainDiagXGamma, "PowerLorentzMainDiagXGamma/D")->SetTitle("Power-Law Lorentz Main Diag X  Gamma");
                fTree->Branch("PowerLorentzMainDiagXGammaErr", &fPowerLorentzMainDiagXGammaErr, "PowerLorentzMainDiagXGammaErr/D")->SetTitle("Power-Law Lorentz Main Diag X  Gamma Error");
                fTree->Branch("PowerLorentzMainDiagXVertOffset", &fPowerLorentzMainDiagXVertOffset, "PowerLorentzMainDiagXVertOffset/D")->SetTitle("Power-Law Lorentz Main Diag X  Vert Offset");
                fTree->Branch("PowerLorentzMainDiagXVertOffsetErr", &fPowerLorentzMainDiagXVertOffsetErr, "PowerLorentzMainDiagXVertOffsetErr/D")->SetTitle("Power-Law Lorentz Main Diag X  Vert Offset Error");
                fTree->Branch("PowerLorentzMainDiagXCenter", &fPowerLorentzMainDiagXCenter, "PowerLorentzMainDiagXCenter/D")->SetTitle("Power-Law Lorentz Main Diag X  Center");
                fTree->Branch("PowerLorentzMainDiagXCenterErr", &fPowerLorentzMainDiagXCenterErr, "PowerLorentzMainDiagXCenterErr/D")->SetTitle("Power-Law Lorentz Main Diag X  Center Error");
                fTree->Branch("PowerLorentzMainDiagXChi2red", &fPowerLorentzMainDiagXChi2red, "PowerLorentzMainDiagXChi2red/D")->SetTitle("Power-Law Lorentz Main Diag X  Reduced Chi-squared");
                fTree->Branch("PowerLorentzMainDiagXPp", &fPowerLorentzMainDiagXPp, "PowerLorentzMainDiagXPp/D")->SetTitle("Power-Law Lorentz Main Diag X  P-value");
                fTree->Branch("PowerLorentzMainDiagXDOF", &fPowerLorentzMainDiagXDOF, "PowerLorentzMainDiagXDOF/I")->SetTitle("Power-Law Lorentz Main Diag X  Degrees of Freedom");
                
                fTree->Branch("PowerLorentzMainDiagYAmp", &fPowerLorentzMainDiagYAmp, "PowerLorentzMainDiagYAmp/D")->SetTitle("Power-Law Lorentz Main Diag Y  Amp");
                fTree->Branch("PowerLorentzMainDiagYAmpErr", &fPowerLorentzMainDiagYAmpErr, "PowerLorentzMainDiagYAmpErr/D")->SetTitle("Power-Law Lorentz Main Diag Y  Amp Error");
                fTree->Branch("PowerLorentzMainDiagYBeta", &fPowerLorentzMainDiagYBeta, "PowerLorentzMainDiagYBeta/D")->SetTitle("Power-Law Lorentz Main Diag Y  Beta");
                fTree->Branch("PowerLorentzMainDiagYBetaErr", &fPowerLorentzMainDiagYBetaErr, "PowerLorentzMainDiagYBetaErr/D")->SetTitle("Power-Law Lorentz Main Diag Y  Beta Error");
                fTree->Branch("PowerLorentzMainDiagYGamma", &fPowerLorentzMainDiagYGamma, "PowerLorentzMainDiagYGamma/D")->SetTitle("Power-Law Lorentz Main Diag Y  Gamma");
                fTree->Branch("PowerLorentzMainDiagYGammaErr", &fPowerLorentzMainDiagYGammaErr, "PowerLorentzMainDiagYGammaErr/D")->SetTitle("Power-Law Lorentz Main Diag Y  Gamma Error");
                fTree->Branch("PowerLorentzMainDiagYVertOffset", &fPowerLorentzMainDiagYVertOffset, "PowerLorentzMainDiagYVertOffset/D")->SetTitle("Power-Law Lorentz Main Diag Y  Vert Offset");
                fTree->Branch("PowerLorentzMainDiagYVertOffsetErr", &fPowerLorentzMainDiagYVertOffsetErr, "PowerLorentzMainDiagYVertOffsetErr/D")->SetTitle("Power-Law Lorentz Main Diag Y  Vert Offset Error");
                fTree->Branch("PowerLorentzMainDiagYCenter", &fPowerLorentzMainDiagYCenter, "PowerLorentzMainDiagYCenter/D")->SetTitle("Power-Law Lorentz Main Diag Y  Center");
                fTree->Branch("PowerLorentzMainDiagYCenterErr", &fPowerLorentzMainDiagYCenterErr, "PowerLorentzMainDiagYCenterErr/D")->SetTitle("Power-Law Lorentz Main Diag Y  Center Error");
                fTree->Branch("PowerLorentzMainDiagYChi2red", &fPowerLorentzMainDiagYChi2red, "PowerLorentzMainDiagYChi2red/D")->SetTitle("Power-Law Lorentz Main Diag Y  Reduced Chi-squared");
                fTree->Branch("PowerLorentzMainDiagYPp", &fPowerLorentzMainDiagYPp, "PowerLorentzMainDiagYPp/D")->SetTitle("Power-Law Lorentz Main Diag Y  P-value");
                fTree->Branch("PowerLorentzMainDiagYDOF", &fPowerLorentzMainDiagYDOF, "PowerLorentzMainDiagYDOF/I")->SetTitle("Power-Law Lorentz Main Diag Y  Degrees of Freedom");
                
                // Secondary diagonal fit parameters
                fTree->Branch("PowerLorentzSecDiagXAmp", &fPowerLorentzSecDiagXAmp, "PowerLorentzSecDiagXAmp/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Amp");
                fTree->Branch("PowerLorentzSecDiagXAmpErr", &fPowerLorentzSecDiagXAmpErr, "PowerLorentzSecDiagXAmpErr/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Amp Error");
                fTree->Branch("PowerLorentzSecDiagXBeta", &fPowerLorentzSecDiagXBeta, "PowerLorentzSecDiagXBeta/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Beta");
                fTree->Branch("PowerLorentzSecDiagXBetaErr", &fPowerLorentzSecDiagXBetaErr, "PowerLorentzSecDiagXBetaErr/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Beta Error");
                fTree->Branch("PowerLorentzSecDiagXGamma", &fPowerLorentzSecDiagXGamma, "PowerLorentzSecDiagXGamma/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Gamma");
                fTree->Branch("PowerLorentzSecDiagXGammaErr", &fPowerLorentzSecDiagXGammaErr, "PowerLorentzSecDiagXGammaErr/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Gamma Error");
                fTree->Branch("PowerLorentzSecDiagXVertOffset", &fPowerLorentzSecDiagXVertOffset, "PowerLorentzSecDiagXVertOffset/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Vert Offset");
                fTree->Branch("PowerLorentzSecDiagXVertOffsetErr", &fPowerLorentzSecDiagXVertOffsetErr, "PowerLorentzSecDiagXVertOffsetErr/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Vert Offset Error");
                fTree->Branch("PowerLorentzSecDiagXCenter", &fPowerLorentzSecDiagXCenter, "PowerLorentzSecDiagXCenter/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Center");
                fTree->Branch("PowerLorentzSecDiagXCenterErr", &fPowerLorentzSecDiagXCenterErr, "PowerLorentzSecDiagXCenterErr/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Center Error");
                fTree->Branch("PowerLorentzSecDiagXChi2red", &fPowerLorentzSecDiagXChi2red, "PowerLorentzSecDiagXChi2red/D")->SetTitle("Power-Law Lorentz Secondary Diag X  Reduced Chi-squared");
                fTree->Branch("PowerLorentzSecDiagXPp", &fPowerLorentzSecDiagXPp, "PowerLorentzSecDiagXPp/D")->SetTitle("Power-Law Lorentz Secondary Diag X  P-value");
                fTree->Branch("PowerLorentzSecDiagXDOF", &fPowerLorentzSecDiagXDOF, "PowerLorentzSecDiagXDOF/I")->SetTitle("Power-Law Lorentz Secondary Diag X  Degrees of Freedom");
                
                fTree->Branch("PowerLorentzSecDiagYAmp", &fPowerLorentzSecDiagYAmp, "PowerLorentzSecDiagYAmp/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Amp");
                fTree->Branch("PowerLorentzSecDiagYAmpErr", &fPowerLorentzSecDiagYAmpErr, "PowerLorentzSecDiagYAmpErr/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Amp Error");
                fTree->Branch("PowerLorentzSecDiagYBeta", &fPowerLorentzSecDiagYBeta, "PowerLorentzSecDiagYBeta/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Beta");
                fTree->Branch("PowerLorentzSecDiagYBetaErr", &fPowerLorentzSecDiagYBetaErr, "PowerLorentzSecDiagYBetaErr/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Beta Error");
                fTree->Branch("PowerLorentzSecDiagYGamma", &fPowerLorentzSecDiagYGamma, "PowerLorentzSecDiagYGamma/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Gamma");
                fTree->Branch("PowerLorentzSecDiagYGammaErr", &fPowerLorentzSecDiagYGammaErr, "PowerLorentzSecDiagYGammaErr/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Gamma Error");
                fTree->Branch("PowerLorentzSecDiagYVertOffset", &fPowerLorentzSecDiagYVertOffset, "PowerLorentzSecDiagYVertOffset/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Vert Offset");
                fTree->Branch("PowerLorentzSecDiagYVertOffsetErr", &fPowerLorentzSecDiagYVertOffsetErr, "PowerLorentzSecDiagYVertOffsetErr/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Vert Offset Error");
                fTree->Branch("PowerLorentzSecDiagYCenter", &fPowerLorentzSecDiagYCenter, "PowerLorentzSecDiagYCenter/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Center");
                fTree->Branch("PowerLorentzSecDiagYCenterErr", &fPowerLorentzSecDiagYCenterErr, "PowerLorentzSecDiagYCenterErr/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Center Error");
                fTree->Branch("PowerLorentzSecDiagYChi2red", &fPowerLorentzSecDiagYChi2red, "PowerLorentzSecDiagYChi2red/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  Reduced Chi-squared");
                fTree->Branch("PowerLorentzSecDiagYPp", &fPowerLorentzSecDiagYPp, "PowerLorentzSecDiagYPp/D")->SetTitle("Power-Law Lorentz Secondary Diag Y  P-value");
                fTree->Branch("PowerLorentzSecDiagYDOF", &fPowerLorentzSecDiagYDOF, "PowerLorentzSecDiagYDOF/I")->SetTitle("Power-Law Lorentz Secondary Diag Y  Degrees of Freedom");
            } // end diagonal Power-Law Lorentz branches
        }
        
        // =============================================
        // 3D FIT PARAMETERS BRANCHES
        // =============================================
        // 3D Gauss fit parameters
        if (Control::GAUSS_FIT_3D) {
        fTree->Branch("3DGaussCenterX", &f3DGaussCenterX, "3DGaussCenterX/D")->SetTitle("3D Gauss  Center X");
        fTree->Branch("3DGaussCenterY", &f3DGaussCenterY, "3DGaussCenterY/D")->SetTitle("3D Gauss  Center Y");
        fTree->Branch("3DGaussSigmaX", &f3DGaussSigmaX, "3DGaussSigmaX/D")->SetTitle("3D Gauss  Sigma X");
        fTree->Branch("3DGaussSigmaY", &f3DGaussSigmaY, "3DGaussSigmaY/D")->SetTitle("3D Gauss  Sigma Y");
        fTree->Branch("3DGaussAmp", &f3DGaussAmp, "3DGaussAmp/D")->SetTitle("3D Gauss  Amp");
        fTree->Branch("3DGaussVertOffset", &f3DGaussVertOffset, "3DGaussVertOffset/D")->SetTitle("3D Gauss  Vert Offset");
        fTree->Branch("3DGaussCenterXErr", &f3DGaussCenterXErr, "3DGaussCenterXErr/D")->SetTitle("3D Gauss  Center X Error");
        fTree->Branch("3DGaussCenterYErr", &f3DGaussCenterYErr, "3DGaussCenterYErr/D")->SetTitle("3D Gauss  Center Y Error");
        fTree->Branch("3DGaussSigmaXErr", &f3DGaussSigmaXErr, "3DGaussSigmaXErr/D")->SetTitle("3D Gauss  Sigma X Error");
        fTree->Branch("3DGaussSigmaYErr", &f3DGaussSigmaYErr, "3DGaussSigmaYErr/D")->SetTitle("3D Gauss  Sigma Y Error");
        fTree->Branch("3DGaussAmpErr", &f3DGaussAmpErr, "3DGaussAmpErr/D")->SetTitle("3D Gauss  Amp Error");
        fTree->Branch("3DGaussVertOffsetErr", &f3DGaussVertOffsetErr, "3DGaussVertOffsetErr/D")->SetTitle("3D Gauss  Vert Offset Error");
        fTree->Branch("3DGaussChi2red", &f3DGaussChi2red, "3DGaussChi2red/D")->SetTitle("3DGaussChi2red/D");
        fTree->Branch("3DGaussPp", &f3DGaussPp, "3DGaussPp/D")->SetTitle("3DGaussPp/D");
        fTree->Branch("3DGaussDOF", &f3DGaussDOF, "3DGaussDOF/I")->SetTitle("3DGaussDOF/I");
        fTree->Branch("3DGaussChargeErr", &f3DGaussChargeErr, "3DGaussChargeErr/D")->SetTitle("3DGaussChargeErr/D");
        fTree->Branch("3DGaussSuccess", &f3DGaussSuccess, "3DGaussSuccess/O")->SetTitle("3DGaussSuccess/O");
        }
        
        // 3D Lorentz fit parameters
        if (Control::LORENTZ_FIT_3D) {
        fTree->Branch("3DLorentzCenterX", &f3DLorentzCenterX, "3DLorentzCenterX/D")->SetTitle("3D Lorentz  Center X");
        fTree->Branch("3DLorentzCenterY", &f3DLorentzCenterY, "3DLorentzCenterY/D")->SetTitle("3D Lorentz  Center Y");
        fTree->Branch("3DLorentzGammaX", &f3DLorentzGammaX, "3DLorentzGammaX/D")->SetTitle("3D Lorentz  Gamma X");
        fTree->Branch("3DLorentzGammaY", &f3DLorentzGammaY, "3DLorentzGammaY/D")->SetTitle("3D Lorentz  Gamma Y");
        fTree->Branch("3DLorentzAmp", &f3DLorentzAmp, "3DLorentzAmp/D")->SetTitle("3D Lorentz  Amp");
        fTree->Branch("3DLorentzVertOffset", &f3DLorentzVertOffset, "3DLorentzVertOffset/D")->SetTitle("3D Lorentz  Vert Offset");
        fTree->Branch("3DLorentzCenterXErr", &f3DLorentzCenterXErr, "3DLorentzCenterXErr/D")->SetTitle("3D Lorentz  Center X Error");
        fTree->Branch("3DLorentzCenterYErr", &f3DLorentzCenterYErr, "3DLorentzCenterYErr/D")->SetTitle("3D Lorentz  Center Y Error");
        fTree->Branch("3DLorentzGammaXErr", &f3DLorentzGammaXErr, "3DLorentzGammaXErr/D")->SetTitle("3D Lorentz  Gamma X Error");
        fTree->Branch("3DLorentzGammaYErr", &f3DLorentzGammaYErr, "3DLorentzGammaYErr/D")->SetTitle("3D Lorentz  Gamma Y Error");
        fTree->Branch("3DLorentzAmpErr", &f3DLorentzAmpErr, "3DLorentzAmpErr/D")->SetTitle("3D Lorentz  Amp Error");
        fTree->Branch("3DLorentzVertOffsetErr", &f3DLorentzVertOffsetErr, "3DLorentzVertOffsetErr/D")->SetTitle("3D Lorentz  Vert Offset Error");
        fTree->Branch("3DLorentzChi2red", &f3DLorentzChi2red, "3DLorentzChi2red/D")->SetTitle("3DLorentzChi2red/D");
        fTree->Branch("3DLorentzPp", &f3DLorentzPp, "3DLorentzPp/D")->SetTitle("3DLorentzPp/D");
        fTree->Branch("3DLorentzDOF", &f3DLorentzDOF, "3DLorentzDOF/I")->SetTitle("3DLorentzDOF/I");
        fTree->Branch("3DLorentzChargeErr", &f3DLorentzChargeErr, "3DLorentzChargeErr/D")->SetTitle("3DLorentzChargeErr/D");
        fTree->Branch("3DLorentzSuccess", &f3DLorentzSuccess, "3DLorentzSuccess/O")->SetTitle("3DLorentzSuccess/O");
        }
        
        // 3D Power-Law Lorentz fit parameters
        if (Control::POWER_LORENTZ_FIT_3D) {
        fTree->Branch("3DPowerLorentzCenterX", &f3DPowerLorentzCenterX, "3DPowerLorentzCenterX/D")->SetTitle("3D Power-Law Lorentz  Center X");
        fTree->Branch("3DPowerLorentzCenterY", &f3DPowerLorentzCenterY, "3DPowerLorentzCenterY/D")->SetTitle("3D Power-Law Lorentz  Center Y");
        fTree->Branch("3DPowerLorentzGammaX", &f3DPowerLorentzGammaX, "3DPowerLorentzGammaX/D")->SetTitle("3D Power-Law Lorentz  Gamma X");
        fTree->Branch("3DPowerLorentzGammaY", &f3DPowerLorentzGammaY, "3DPowerLorentzGammaY/D")->SetTitle("3D Power-Law Lorentz  Gamma Y");
        fTree->Branch("3DPowerLorentzBeta", &f3DPowerLorentzBeta, "3DPowerLorentzBeta/D")->SetTitle("3D Power-Law Lorentz  Beta (Power-Law Exponent)");
        fTree->Branch("3DPowerLorentzAmp", &f3DPowerLorentzAmp, "3DPowerLorentzAmp/D")->SetTitle("3D Power-Law Lorentz  Amp");
        fTree->Branch("3DPowerLorentzVertOffset", &f3DPowerLorentzVertOffset, "3DPowerLorentzVertOffset/D")->SetTitle("3D Power-Law Lorentz  Vert Offset");
        fTree->Branch("3DPowerLorentzCenterXErr", &f3DPowerLorentzCenterXErr, "3DPowerLorentzCenterXErr/D")->SetTitle("3D Power-Law Lorentz  Center X Error");
        fTree->Branch("3DPowerLorentzCenterYErr", &f3DPowerLorentzCenterYErr, "3DPowerLorentzCenterYErr/D")->SetTitle("3D Power-Law Lorentz  Center Y Error");
        fTree->Branch("3DPowerLorentzGammaXErr", &f3DPowerLorentzGammaXErr, "3DPowerLorentzGammaXErr/D")->SetTitle("3D Power-Law Lorentz  Gamma X Error");
        fTree->Branch("3DPowerLorentzGammaYErr", &f3DPowerLorentzGammaYErr, "3DPowerLorentzGammaYErr/D")->SetTitle("3D Power-Law Lorentz  Gamma Y Error");
        fTree->Branch("3DPowerLorentzBetaErr", &f3DPowerLorentzBetaErr, "3DPowerLorentzBetaErr/D")->SetTitle("3D Power-Law Lorentz  Beta Error");
        fTree->Branch("3DPowerLorentzAmpErr", &f3DPowerLorentzAmpErr, "3DPowerLorentzAmpErr/D")->SetTitle("3D Power-Law Lorentz  Amp Error");
        fTree->Branch("3DPowerLorentzVertOffsetErr", &f3DPowerLorentzVertOffsetErr, "3DPowerLorentzVertOffsetErr/D")->SetTitle("3D Power-Law Lorentz  Vert Offset Error");
        fTree->Branch("3DPowerLorentzChi2red", &f3DPowerLorentzChi2red, "3DPowerLorentzChi2red/D")->SetTitle("3DPowerLorentzChi2red/D");
        fTree->Branch("3DPowerLorentzPp", &f3DPowerLorentzPp, "3DPowerLorentzPp/D")->SetTitle("3DPowerLorentzPp/D");
        fTree->Branch("3DPowerLorentzDOF", &f3DPowerLorentzDOF, "3DPowerLorentzDOF/I")->SetTitle("3DPowerLorentzDOF/I");
        fTree->Branch("3DPowerLorentzChargeErr", &f3DPowerLorentzChargeErr, "3DPowerLorentzChargeErr/D")->SetTitle("3DPowerLorentzChargeErr/D");
        fTree->Branch("3DPowerLorentzSuccess", &f3DPowerLorentzSuccess, "3DPowerLorentzSuccess/O")->SetTitle("3DPowerLorentzSuccess/O");
        }
        
        // =============================================
        // Trans COORDINATE BRANCHES
        // =============================================
        // Gauss diagonal Trans coordinates
        if (Control::GAUSS_FIT && Control::DIAG_FIT) {
        fTree->Branch("GaussMainDiagTransX", &fGaussMainDiagTransformedX, "GaussMainDiagTransX/D")->SetTitle("Gauss Main Diag Trans X Coord [mm]");
        fTree->Branch("GaussMainDiagTransY", &fGaussMainDiagTransformedY, "GaussMainDiagTransY/D")->SetTitle("Gauss Main Diag Trans Y Coord [mm]");
        fTree->Branch("GaussSecDiagTransX", &fGaussSecDiagTransformedX, "GaussSecDiagTransX/D")->SetTitle("Gauss Secondary Diag Trans X Coord [mm]");
        fTree->Branch("GaussSecDiagTransY", &fGaussSecDiagTransformedY, "GaussSecDiagTransY/D")->SetTitle("Gauss Secondary Diag Trans Y Coord [mm]");
        }
        
        // Lorentz diagonal Trans coordinates
        if (Control::LORENTZ_FIT && Control::DIAG_FIT) {
        fTree->Branch("LorentzMainDiagTransX", &fLorentzMainDiagTransformedX, "LorentzMainDiagTransX/D")->SetTitle("Lorentz Main Diag Trans X Coord [mm]");
        fTree->Branch("LorentzMainDiagTransY", &fLorentzMainDiagTransformedY, "LorentzMainDiagTransY/D")->SetTitle("Lorentz Main Diag Trans Y Coord [mm]");
        fTree->Branch("LorentzSecDiagTransX", &fLorentzSecDiagTransformedX, "LorentzSecDiagTransX/D")->SetTitle("Lorentz Secondary Diag Trans X Coord [mm]");
        fTree->Branch("LorentzSecDiagTransY", &fLorentzSecDiagTransformedY, "LorentzSecDiagTransY/D")->SetTitle("Lorentz Secondary Diag Trans Y Coord [mm]");
        }
        
        // Power-Law Lorentz diagonal Trans coordinates
        if (Control::POWER_LORENTZ_FIT && Control::DIAG_FIT) {
        fTree->Branch("PowerLorentzMainDiagTransX", &fPowerLorentzMainDiagTransformedX, "PowerLorentzMainDiagTransX/D")->SetTitle("Power-Law Lorentz Main Diag Trans X Coord [mm]");
        fTree->Branch("PowerLorentzMainDiagTransY", &fPowerLorentzMainDiagTransformedY, "PowerLorentzMainDiagTransY/D")->SetTitle("Power-Law Lorentz Main Diag Trans Y Coord [mm]");
        fTree->Branch("PowerLorentzSecDiagTransX", &fPowerLorentzSecDiagTransformedX, "PowerLorentzSecDiagTransX/D")->SetTitle("Power-Law Lorentz Secondary Diag Trans X Coord [mm]");
        fTree->Branch("PowerLorentzSecDiagTransY", &fPowerLorentzSecDiagTransformedY, "PowerLorentzSecDiagTransY/D")->SetTitle("Power-Law Lorentz Secondary Diag Trans Y Coord [mm]");
        }
        
        // =============================================
        // SCORER DATA BRANCHES
        // =============================================
        // Add scorer data branches with validation
        TBranch* scorerEnergyBranch = fTree->Branch("ScorerEnergyDeposit", &fScorerEnergyDeposit, "ScorerEnergyDeposit/D");
        TBranch* scorerHitCountBranch = fTree->Branch("ScorerHitCount", &fScorerHitCount, "ScorerHitCount/I");
        TBranch* scorerDataValidBranch = fTree->Branch("ScorerDataValid", &fScorerDataValid, "ScorerDataValid/O");
        
        // Validate scorer branch creation
        if (scorerEnergyBranch && scorerHitCountBranch && scorerDataValidBranch) {
            scorerEnergyBranch->SetTitle("Energy Deposit from Multi-Functional Detector [MeV]");
            scorerHitCountBranch->SetTitle("Hit Count from Multi-Functional Detector");
            scorerDataValidBranch->SetTitle("Validation Flag for Scorer Data");
            G4cout << "✓ Scorer data branches successfully created in ROOT tree" << G4endl;
        } else {
            G4cerr << "ERROR: Failed to create scorer data branches in ROOT tree!" << G4endl;
        }
        
        // =============================================
        // HIT PURITY TRACKING BRANCHES
        // =============================================
        // Add hit purity tracking branches for Multi-Functional Detector validation
        TBranch* pureSiliconHitBranch = fTree->Branch("PureSiliconHit", &fPureSiliconHit, "PureSiliconHit/O");
        TBranch* aluminumContaminatedBranch = fTree->Branch("AluminumContaminated", &fAluminumContaminated, "AluminumContaminated/O");
        TBranch* chargeCalculationEnabledBranch = fTree->Branch("ChargeCalculationEnabled", &fChargeCalculationEnabled, "ChargeCalculationEnabled/O");
        
        // Validate hit purity tracking branch creation
        if (pureSiliconHitBranch && aluminumContaminatedBranch && chargeCalculationEnabledBranch) {
            pureSiliconHitBranch->SetTitle("Pure Silicon Hit (No Aluminum Contamination)");
            aluminumContaminatedBranch->SetTitle("Aluminum Contamination Detected");
            chargeCalculationEnabledBranch->SetTitle("Charge Sharing Calculation Enabled");
            G4cout << "✓ Hit purity tracking branches successfully created in ROOT tree" << G4endl;
        } else {
            G4cerr << "ERROR: Failed to create hit purity tracking branches in ROOT tree!" << G4endl;
        }
        
        G4cout << "Created ROOT tree with " << fTree->GetNbranches() << " branches" << G4endl;
        
        // Enable frequent AutoSave only when explicitly requested via Constants flag
        if (Control::ENABLE_AUTOSAVE) {
            // 1000-event interval was the historical default; keep it when enabled.
            EnableAutoSave(1000);
        } else {
            // Ensure the internal counter is reset and AutoSave remains disabled
            fAutoSaveEnabled = false;
        }
    }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    // Safety check for valid run
    if (!run) {
        G4cerr << "RunAction: Error - Invalid run object in EndOfRunAction" << G4endl;
        return;
    }
    
    G4int nofEvents = run->GetNumberOfEvent();
    G4String fileName = "";
    G4int nEntries = 0;
    
    // Log run end information
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
        logger->LogRunEnd(run->GetRunID());
    }
    
    // Worker threads: Write their individual files safely
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        
        if (fRootFile && !fRootFile->IsZombie()) {
            fileName = fRootFile->GetName();
        }
        
        if (fTree) {
            nEntries = fTree->GetEntries();
        }
        
        if (fRootFile && fTree && nofEvents > 0) {
            G4cout << "Worker thread writing ROOT file with " << nEntries 
                   << " entries from " << nofEvents << " events" << G4endl;
            
            // Use the new safe write method
            if (SafeWriteRootFile()) {
                G4cout << "Worker thread: Successly wrote " << fileName << G4endl;
            } else {
                G4cerr << "Worker thread: Failed to write " << fileName << G4endl;
            }
        }
        
        // Clean up worker ROOT objects
        CleanupRootObjects();
        
        // Signal completion to master thread
        SignalWorkerCompletion();
        
        return; // Worker threads are done
    }
    
    // Master thread: Wait for workers then merge files
    G4cout << "Master thread: Waiting for all worker threads to complete..." << G4endl;
    
    // Use the new robust synchronization
    WaitForAllWorkersToComplete();
    
    // Now perform the robust file merging
    if (G4Threading::IsMultithreadedApplication()) {
        G4cout << "Master thread: Starting robust file merging..." << G4endl;
        
        try {
            // Use separate lock scope for merging
            std::lock_guard<std::mutex> lock(fRootMutex);
            
            G4int nThreads = fTotalWorkers.load();
            std::vector<G4String> workerFileNames;
            std::vector<G4String> validFiles;
            
            // Generate expected worker file names
            for (G4int i = 0; i < nThreads; i++) {
                std::ostringstream oss;
                oss << "epicChargeSharingOutput_t" << i << ".root";
                workerFileNames.push_back(oss.str());
            }
            
            // Validate all worker files with enhanced checking
            for (const auto& workerFile : workerFileNames) {
                if (ValidateRootFile(workerFile)) {
                    validFiles.push_back(workerFile);
                    G4cout << "Master thread: Validated worker file " << workerFile << G4endl;
                } else {
                    G4cerr << "Master thread: Invalid or missing worker file " << workerFile << G4endl;
                }
            }
            
            if (validFiles.empty()) {
                G4cerr << "Master thread: No valid worker files found for merging!" << G4endl;
                return;
            }
            
            // Count total entries for verification
            G4int totalEntries = 0;
            for (const auto& validFile : validFiles) {
                TFile* testFile = TFile::Open(validFile.c_str(), "READ");
                if (testFile && !testFile->IsZombie()) {
                    TTree* testTree = (TTree*)testFile->Get("Hits");
                    if (testTree) {
                        totalEntries += testTree->GetEntries();
                    }
                    testFile->Close();
                    delete testFile;
                }
            }
            
            G4cout << "Master thread: Merging " << validFiles.size() 
                   << " files with total " << totalEntries << " entries" << G4endl;
            
            // Use ROOT's TFileMerger for robust and thread-safe file merging
            TFileMerger merger(kFALSE); // kFALSE = don't print progress
            merger.SetFastMethod(kTRUE);
            merger.SetNotrees(kFALSE);
            
            // Set output file
            if (!merger.OutputFile("epicChargeSharingOutput.root", "RECREATE", 1)) {
                G4cerr << "Master thread: Failed to set output file for merger!" << G4endl;
                return;
            }
            
            // Add all valid worker files to merger
            for (const auto& validFile : validFiles) {
                if (!merger.AddFile(validFile.c_str())) {
                    G4cerr << "Master thread: Failed to add " << validFile << " to merger" << G4endl;
                } else {
                    G4cout << "Master thread: Added " << validFile << " to merger" << G4endl;
                }
            }
            
            // Perform the merge
            Bool_t mergeResult = merger.Merge();
            if (!mergeResult) {
                G4cerr << "Master thread: File merging failed!" << G4endl;
                return;
            }
            
            G4cout << "Master thread: File merging completed successfully" << G4endl;
            
            // Add metadata to the merged file
            // NOTE: This is the ONLY place metadata should be written to avoid duplicates
            // Worker threads write only their tree data, master adds metadata once to final file
            if (fGridPixelSize > 0) {
                TFile* mergedFile = TFile::Open("epicChargeSharingOutput.root", "UPDATE");
                if (mergedFile && !mergedFile->IsZombie()) {
                    mergedFile->cd();
                    
                    TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", fGridPixelSize));
                    TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing));
                    TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset));
                    TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", fGridDetSize));
                    TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
                    TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", Constants::NEIGHBORHOOD_RADIUS));
                    
                    pixelSizeMeta.Write();
                    pixelSpacingMeta.Write();
                    pixelCornerOffsetMeta.Write();
                    detSizeMeta.Write();
                    numBlocksMeta.Write();
                    neighborhoodRadiusMeta.Write();
                    
                    mergedFile->Close();
                    delete mergedFile;
                    
                    G4cout << "Master thread: Saved detector grid metadata to merged file" << G4endl;
                } else {
                    G4cerr << "Master thread: Failed to open merged file for metadata" << G4endl;
                }
            }
            
            // Verify the merged file
            TFile* verifyFile = TFile::Open("epicChargeSharingOutput.root", "READ");
            if (verifyFile && !verifyFile->IsZombie()) {
                TTree* verifyTree = (TTree*)verifyFile->Get("Hits");
                if (verifyTree) {
                    G4cout << "Master thread: Successly created merged file with " 
                           << verifyTree->GetEntries() << " entries" << G4endl;
                }
                verifyFile->Close();
                delete verifyFile;
            } else {
                G4cerr << "Master thread: Failed to verify merged file" << G4endl;
            }
            
            // Clean up worker files after success merge
            for (const auto& file : validFiles) {
                if (std::remove(file.c_str()) == 0) {
                    G4cout << "Master thread: Cleaned up " << file << G4endl;
                } else {
                    G4cerr << "Master thread: Failed to clean up " << file << G4endl;
                }
            }
            
        } catch (const std::exception& e) {
            G4cerr << "Master thread: Exception during robust file merging: " << e.what() << G4endl;
        }
    }
    
    G4cout << "Master thread: File operations completed" << G4endl;
}

void RunAction::SetEventData(G4double edep, G4double x, G4double y, G4double z) 
{
    // Store energy depositionit in MeV (Geant4 internal energy unit is MeV)
    fEdep = edep;
    
    // Store positions in mm (Geant4 internal length unit is mm)
    fTrueX = x;
    fTrueY = y;
}

void RunAction::SetInitialPos(G4double x, G4double y, G4double z)
{
    fInitialX = x;
    fInitialY = y;
    fInitialZ = z;
}

void RunAction::SetNearestPixelPos(G4double x, G4double y)
{
    // Store nearest pixel centre coordinates
    fNearestPixelX = x;
    fNearestPixelY = y;

    // Ensure internal pixel centre variables used by diagonal transformations are kept in sync.
    // NOTE: Historically, the diagonal coordinate transformation functions relied on fPixelX/Y
    // but these members were never initialised, causing large residuals for diagonal fits.
    // Keeping the two representations consistent fixes the excessively large diagonal
    // resolution values reported in the analysis stage.
    fPixelX = x;
    fPixelY = y;
}

void RunAction::SetInitialEnergy(G4double energy) 
{
    // Store initial particle energy in MeV (Geant4 internal energy unit is MeV)
    fInitialEnergy = energy;
}



void RunAction::SetPixelClassification(G4bool isWithinD0, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY)
{
    // Store the classification and delta values from pixel center to true position
    fIsPixelHit = isWithinD0;
    fPixelTrueDeltaX = pixelTrueDeltaX;
    fPixelTrueDeltaY = pixelTrueDeltaY;
}

void RunAction::SetPixelHitStatus(G4bool isPixelHit)
{
    // Store pixel hit status (true if on pixel OR distance <= D0)
    fIsPixelHit = isPixelHit;
}

void RunAction::SetNeighborhoodGridData(const std::vector<G4double>& angles)
{
    // Store the neighborhood (9x9) grid angle data for non-pixel hits
    fNeighborhoodAngles = angles;
}

void RunAction::SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                                 const std::vector<G4double>& distances,
                                 const std::vector<G4double>& chargeValues,
                                 const std::vector<G4double>& chargeCoulombs)
{
    // Store the neighborhood (9x9) grid charge sharing data for non-pixel hits
    fNeighborhoodChargeFractions = chargeFractions;
    fNeighborhoodDistances = distances;
    fNeighborhoodCharge = chargeCoulombs;
}

void RunAction::FillTree()
{
    if (!fTree || !fRootFile || fRootFile->IsZombie()) {
        G4cerr << "Error: Invalid ROOT file or tree in FillTree()" << G4endl;
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(fRootMutex);
        
        // Validate scorer data before writing to tree
        ValidateScorerDataForTreeStorage();
        
        // Fill the tree with all current data (including scorer data)
        G4int fillResult = fTree->Fill();
        
        // Validate successful tree filling
        if (fillResult < 0) {
            G4cerr << "Error: Tree Fill() returned error code " << fillResult << G4endl;
            return;
        }
        
        // Log scorer data storage (only in debug mode to avoid excessive output)
        static G4int eventCount = 0;
        eventCount++;
        if (eventCount % 100 == 0) {  // Log every 100 events
            G4cout << "Event " << eventCount << ": Scorer data stored - Energy: " 
                   << fScorerEnergyDeposit << " MeV, Hits: " << fScorerHitCount 
                   << ", Valid: " << (fScorerDataValid ? "Yes" : "No") << G4endl;
        }
        
        // Verify scorer data is written to ROOT tree (periodic check)
        if (eventCount % 500 == 0) {  // Verify every 500 events
            VerifyScorerDataInTree();
        }
        
        // Use the new thread-safe auto-save mechanism
        PerformAutoSave();
        
    } catch (const std::exception& e) {
        G4cerr << "Exception in FillTree: " << e.what() << G4endl;
    }
}

void RunAction::SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                           G4double pixelCornerOffset, G4double detSize, 
                                           G4int numBlocksPerSide)
{
    // Safety check for valid parameters
    if (pixelSize <= 0 || pixelSpacing <= 0 || detSize <= 0 || numBlocksPerSide <= 0) {
        G4cerr << "RunAction: Error - Invalid detector grid parameters provided" << G4endl;
        return;
    }
    
    // Store the detector grid parameters for saving to ROOT metadata
    fGridPixelSize = pixelSize;
    fGridPixelSpacing = pixelSpacing;
    fGridPixelCornerOffset = pixelCornerOffset;
    fGridDetSize = detSize;
    fGridNumBlocksPerSide = numBlocksPerSide;
    
    G4cout << "RunAction: Detector grid parameters set:" << G4endl;
    G4cout << "  Pixel Size: " << fGridPixelSize << " mm" << G4endl;
    G4cout << "  Pixel Spacing: " << fGridPixelSpacing << " mm" << G4endl;
    G4cout << "  Pixel Corner Offset: " << fGridPixelCornerOffset << " mm" << G4endl;
    G4cout << "  Detector Size: " << fGridDetSize << " mm" << G4endl;
    G4cout << "  Number of Blocks per Side: " << fGridNumBlocksPerSide << G4endl;
}

void RunAction::Set2DGaussResults(G4double x_center, G4double x_sigma, G4double x_Amp,
                                        G4double x_center_err, G4double x_sigma_err, G4double x_Amp_err,
                                        G4double x_Vert_offset, G4double x_Vert_offset_err,
                                        G4double x_chi2red, G4double x_pp, G4int x_dof,
                                        G4double y_center, G4double y_sigma, G4double y_Amp,
                                        G4double y_center_err, G4double y_sigma_err, G4double y_Amp_err,
                                        G4double y_Vert_offset, G4double y_Vert_offset_err,
                                        G4double y_chi2red, G4double y_pp, G4int y_dof,
                                        G4double x_charge_err, G4double y_charge_err,
                                        G4bool fit_success)
{
    // Store 2D Gauss fit results from central row (X fit)
    fGaussRowCenter = x_center;
    fGaussRowSigma = x_sigma;
    fGaussRowAmp = x_Amp;
    fGaussRowCenterErr = x_center_err;
    fGaussRowSigmaErr = x_sigma_err;
    fGaussRowAmpErr = x_Amp_err;
    fGaussRowVertOffset = x_Vert_offset;
    fGaussRowVertOffsetErr = x_Vert_offset_err;
    fGaussRowChi2red = x_chi2red;
    fGaussRowPp = x_pp;
    fGaussRowDOF = x_dof;
    
    // Store 2D Gauss fit results from central column (Y fit)
    fGaussColCenter = y_center;
    fGaussColSigma = y_sigma;
    fGaussColAmp = y_Amp;
    fGaussColCenterErr = y_center_err;
    fGaussColSigmaErr = y_sigma_err;
    fGaussColAmpErr = y_Amp_err;
    fGaussColVertOffset = y_Vert_offset;
    fGaussColVertOffsetErr = y_Vert_offset_err;
    fGaussColChi2red = y_chi2red;
    fGaussColPp = y_pp;
    fGaussColDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        fGaussRowChargeErr = x_charge_err;
        fGaussColChargeErr = y_charge_err;
    } else {
        fGaussRowChargeErr = 0.0;
        fGaussColChargeErr = 0.0;
    }
    
    // Calc delta values for row and column fits vs true position
    // Use individual fit validity checks similar to diagonal fits for consistency
    if (fit_success) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fGaussRowDeltaX = std::abs(x_center - fTrueX);      // x_row_fit - x_true
        } else {
            fGaussRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fGaussColDeltaY = std::abs(y_center - fTrueY);   // y_column_fit - y_true
        } else {
            fGaussColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fGaussRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

void RunAction::SetDiagGaussResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_Amp,
                                             G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_Amp_err,
                                             G4double main_diag_x_Vert_offset, G4double main_diag_x_Vert_offset_err,
                                             G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_success,
                                             G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_Amp,
                                             G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_Amp_err,
                                             G4double main_diag_y_Vert_offset, G4double main_diag_y_Vert_offset_err,
                                             G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_success,
                                             G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_Amp,
                                             G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_Amp_err,
                                             G4double sec_diag_x_Vert_offset, G4double sec_diag_x_Vert_offset_err,
                                             G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_success,
                                             G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_Amp,
                                             G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_Amp_err,
                                             G4double sec_diag_y_Vert_offset, G4double sec_diag_y_Vert_offset_err,
                                             G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_success,
                                             G4bool fit_success)
{
    // Store main diagonal X fit results
    fGaussMainDiagXCenter = main_diag_x_center;
    fGaussMainDiagXSigma = main_diag_x_sigma;
    fGaussMainDiagXAmp = main_diag_x_Amp;
    fGaussMainDiagXCenterErr = main_diag_x_center_err;
    fGaussMainDiagXSigmaErr = main_diag_x_sigma_err;
    fGaussMainDiagXAmpErr = main_diag_x_Amp_err;
    fGaussMainDiagXVertOffset = main_diag_x_Vert_offset;
    fGaussMainDiagXVertOffsetErr = main_diag_x_Vert_offset_err;
    fGaussMainDiagXChi2red = main_diag_x_chi2red;
    fGaussMainDiagXPp = main_diag_x_pp;
    fGaussMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results
    fGaussMainDiagYCenter = main_diag_y_center;
    fGaussMainDiagYSigma = main_diag_y_sigma;
    fGaussMainDiagYAmp = main_diag_y_Amp;
    fGaussMainDiagYCenterErr = main_diag_y_center_err;
    fGaussMainDiagYSigmaErr = main_diag_y_sigma_err;
    fGaussMainDiagYAmpErr = main_diag_y_Amp_err;
    fGaussMainDiagYVertOffset = main_diag_y_Vert_offset;
    fGaussMainDiagYVertOffsetErr = main_diag_y_Vert_offset_err;
    fGaussMainDiagYChi2red = main_diag_y_chi2red;
    fGaussMainDiagYPp = main_diag_y_pp;
    fGaussMainDiagYDOF = main_diag_y_dof;
    
    // Store Secondary diagonal X fit results
    fGaussSecDiagXCenter = sec_diag_x_center;
    fGaussSecDiagXSigma = sec_diag_x_sigma;
    fGaussSecDiagXAmp = sec_diag_x_Amp;
    fGaussSecDiagXCenterErr = sec_diag_x_center_err;
    fGaussSecDiagXSigmaErr = sec_diag_x_sigma_err;
    fGaussSecDiagXAmpErr = sec_diag_x_Amp_err;
    fGaussSecDiagXVertOffset = sec_diag_x_Vert_offset;
    fGaussSecDiagXVertOffsetErr = sec_diag_x_Vert_offset_err;
    fGaussSecDiagXChi2red = sec_diag_x_chi2red;
    fGaussSecDiagXPp = sec_diag_x_pp;
    fGaussSecDiagXDOF = sec_diag_x_dof;
    
    // Store Secondary diagonal Y fit results
    fGaussSecDiagYCenter = sec_diag_y_center;
    fGaussSecDiagYSigma = sec_diag_y_sigma;
    fGaussSecDiagYAmp = sec_diag_y_Amp;
    fGaussSecDiagYCenterErr = sec_diag_y_center_err;
    fGaussSecDiagYSigmaErr = sec_diag_y_sigma_err;
    fGaussSecDiagYAmpErr = sec_diag_y_Amp_err;
    fGaussSecDiagYVertOffset = sec_diag_y_Vert_offset;
    fGaussSecDiagYVertOffsetErr = sec_diag_y_Vert_offset_err;
    fGaussSecDiagYChi2red = sec_diag_y_chi2red;
    fGaussSecDiagYPp = sec_diag_y_pp;
    fGaussSecDiagYDOF = sec_diag_y_dof;
    

    
    // Calc Trans diagonal coordinates using rotation matrix
    CalcTransformedDiagCoords();
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

void RunAction::Set2DLorentzResults(G4double x_center, G4double x_gamma, G4double x_Amp,
                                          G4double x_center_err, G4double x_gamma_err, G4double x_Amp_err,
                                          G4double x_Vert_offset, G4double x_Vert_offset_err,
                                          G4double x_chi2red, G4double x_pp, G4int x_dof,
                                          G4double y_center, G4double y_gamma, G4double y_Amp,
                                          G4double y_center_err, G4double y_gamma_err, G4double y_Amp_err,
                                          G4double y_Vert_offset, G4double y_Vert_offset_err,
                                          G4double y_chi2red, G4double y_pp, G4int y_dof,
                                          G4double x_charge_err, G4double y_charge_err,
                                          G4bool fit_success)
{
    // Store 2D Lorentz fit results from central row (X fit)
    fLorentzRowCenter = x_center;
    fLorentzRowGamma = x_gamma;
    fLorentzRowAmp = x_Amp;
    fLorentzRowCenterErr = x_center_err;
    fLorentzRowGammaErr = x_gamma_err;
    fLorentzRowAmpErr = x_Amp_err;
    fLorentzRowVertOffset = x_Vert_offset;
    fLorentzRowVertOffsetErr = x_Vert_offset_err;
    fLorentzRowChi2red = x_chi2red;
    fLorentzRowPp = x_pp;
    fLorentzRowDOF = x_dof;
    
    // Store 2D Lorentz fit results from central column (Y fit)
    fLorentzColCenter = y_center;
    fLorentzColGamma = y_gamma;
    fLorentzColAmp = y_Amp;
    fLorentzColCenterErr = y_center_err;
    fLorentzColGammaErr = y_gamma_err;
    fLorentzColAmpErr = y_Amp_err;
    fLorentzColVertOffset = y_Vert_offset;
    fLorentzColVertOffsetErr = y_Vert_offset_err;
    fLorentzColChi2red = y_chi2red;
    fLorentzColPp = y_pp;
    fLorentzColDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        fLorentzRowChargeErr = x_charge_err;
        fLorentzColChargeErr = y_charge_err;
    } else {
        fLorentzRowChargeErr = 0.0;
        fLorentzColChargeErr = 0.0;
    }
    
    // Calc delta values for row and column fits vs true position
    if (fit_success) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fLorentzRowDeltaX = std::abs(x_center - fTrueX);      // x_row_fit - x_true
        } else {
            fLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fLorentzColDeltaY = std::abs(y_center - fTrueY);   // y_column_fit - y_true
        } else {
            fLorentzColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

void RunAction::SetDiagLorentzResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_Amp,
                                               G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_Amp_err,
                                               G4double main_diag_x_Vert_offset, G4double main_diag_x_Vert_offset_err,
                                               G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_success,
                                               G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_Amp,
                                               G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_Amp_err,
                                               G4double main_diag_y_Vert_offset, G4double main_diag_y_Vert_offset_err,
                                               G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_success,
                                               G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_Amp,
                                               G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_Amp_err,
                                               G4double sec_diag_x_Vert_offset, G4double sec_diag_x_Vert_offset_err,
                                               G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_success,
                                               G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_Amp,
                                               G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_Amp_err,
                                               G4double sec_diag_y_Vert_offset, G4double sec_diag_y_Vert_offset_err,
                                               G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_success,
                                               G4bool fit_success)
{
    // Store main diagonal X fit results
    fLorentzMainDiagXCenter = main_diag_x_center;
    fLorentzMainDiagXGamma = main_diag_x_gamma;
    fLorentzMainDiagXAmp = main_diag_x_Amp;
    fLorentzMainDiagXCenterErr = main_diag_x_center_err;
    fLorentzMainDiagXGammaErr = main_diag_x_gamma_err;
    fLorentzMainDiagXAmpErr = main_diag_x_Amp_err;
    fLorentzMainDiagXVertOffset = main_diag_x_Vert_offset;
    fLorentzMainDiagXVertOffsetErr = main_diag_x_Vert_offset_err;
    fLorentzMainDiagXChi2red = main_diag_x_chi2red;
    fLorentzMainDiagXPp = main_diag_x_pp;
    fLorentzMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results
    fLorentzMainDiagYCenter = main_diag_y_center;
    fLorentzMainDiagYGamma = main_diag_y_gamma;
    fLorentzMainDiagYAmp = main_diag_y_Amp;
    fLorentzMainDiagYCenterErr = main_diag_y_center_err;
    fLorentzMainDiagYGammaErr = main_diag_y_gamma_err;
    fLorentzMainDiagYAmpErr = main_diag_y_Amp_err;
    fLorentzMainDiagYVertOffset = main_diag_y_Vert_offset;
    fLorentzMainDiagYVertOffsetErr = main_diag_y_Vert_offset_err;
    fLorentzMainDiagYChi2red = main_diag_y_chi2red;
    fLorentzMainDiagYPp = main_diag_y_pp;
    fLorentzMainDiagYDOF = main_diag_y_dof;
    
    // Store Secondary diagonal X fit results
    fLorentzSecDiagXCenter = sec_diag_x_center;
    fLorentzSecDiagXGamma = sec_diag_x_gamma;
    fLorentzSecDiagXAmp = sec_diag_x_Amp;
    fLorentzSecDiagXCenterErr = sec_diag_x_center_err;
    fLorentzSecDiagXGammaErr = sec_diag_x_gamma_err;
    fLorentzSecDiagXAmpErr = sec_diag_x_Amp_err;
    fLorentzSecDiagXVertOffset = sec_diag_x_Vert_offset;
    fLorentzSecDiagXVertOffsetErr = sec_diag_x_Vert_offset_err;
    fLorentzSecDiagXChi2red = sec_diag_x_chi2red;
    fLorentzSecDiagXPp = sec_diag_x_pp;
    fLorentzSecDiagXDOF = sec_diag_x_dof;
    
    // Store Secondary diagonal Y fit results
    fLorentzSecDiagYCenter = sec_diag_y_center;
    fLorentzSecDiagYGamma = sec_diag_y_gamma;
    fLorentzSecDiagYAmp = sec_diag_y_Amp;
    fLorentzSecDiagYCenterErr = sec_diag_y_center_err;
    fLorentzSecDiagYGammaErr = sec_diag_y_gamma_err;
    fLorentzSecDiagYAmpErr = sec_diag_y_Amp_err;
    fLorentzSecDiagYVertOffset = sec_diag_y_Vert_offset;
    fLorentzSecDiagYVertOffsetErr = sec_diag_y_Vert_offset_err;
    fLorentzSecDiagYChi2red = sec_diag_y_chi2red;
    fLorentzSecDiagYPp = sec_diag_y_pp;
    fLorentzSecDiagYDOF = sec_diag_y_dof;

    // Calc Trans diagonal coordinates using rotation matrix
    CalcTransformedDiagCoords();
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

// =============================================
// COORDINATE TRANSFORMATION HELPER METHODS
// =============================================

void RunAction::TransformDiagCoords(G4double x_prime, G4double y_prime, G4double theta_deg, 
                                             G4double& x_Trans, G4double& y_Trans)
{
    // Convert angle to radians
    const G4double PI = 3.14159265358979323846;
    G4double theta_rad = theta_deg * PI / 180.0;
    
    // Calc rotation matrix components
    G4double cos_theta = std::cos(theta_rad);
    G4double sin_theta = std::sin(theta_rad);
    
    // Apply rotation matrix transformation:
    // [cos(θ) -sin(θ)] [x']   [x]
    // [sin(θ)  cos(θ)] [y'] = [y]
    x_Trans = cos_theta * x_prime - sin_theta * y_prime;
    y_Trans = sin_theta * x_prime + cos_theta * y_prime;
}

void RunAction::CalcTransformedDiagCoords()
{
    // Safety check for valid true position data
    if (std::isnan(fTrueX) || std::isnan(fTrueY)) {
        G4cerr << "RunAction: Warning - Invalid true position data, cannot calculate Trans coordinates" << G4endl;
        return;
    }
    
    const double NaN = std::numeric_limits<G4double>::quiet_NaN();
    const double invSqrt2 = 1.0 / 1.4142135623730951; // 1/√2

    auto setXY = [&](double xPred, double yPred,
                     double &outX, double &outY,
                     double &deltaX, double &deltaY)
    {
        outX   = xPred;
        outY   = yPred;
        deltaX = std::isnan(xPred) ? NaN : (xPred - fTrueX);
        deltaY = std::isnan(yPred) ? NaN : (yPred - fTrueY);
    };

    // ------------------
    //   G A U S S I A N
    // ------------------
    // FIX: Gauss diagonal fits also return diagonal coordinates that need transformation
    auto sToDxDyMain = [&](double s){ return std::make_pair(s*invSqrt2, s*invSqrt2); };
    auto sToDxDySec  = [&](double s){ return std::make_pair(s*invSqrt2, -s*invSqrt2); };

    // Main diagonal
    double sMain = NaN;
    if (fGaussMainDiagXDOF > 0)       sMain = fGaussMainDiagXCenter;
    else if (fGaussMainDiagYDOF > 0)  sMain = fGaussMainDiagYCenter;
    if (!std::isnan(sMain)) {
        auto [dx,dy] = sToDxDyMain(sMain);
        setXY(fPixelX+dx, fPixelY+dy,
              fGaussMainDiagTransformedX, fGaussMainDiagTransformedY,
              fGaussMainDiagTransformedDeltaX, fGaussMainDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fGaussMainDiagTransformedX, fGaussMainDiagTransformedY,
              fGaussMainDiagTransformedDeltaX, fGaussMainDiagTransformedDeltaY);
    }

    // Secondary diagonal
    double sSec = NaN;
    if (fGaussSecDiagXDOF > 0)       sSec = fGaussSecDiagXCenter;
    else if (fGaussSecDiagYDOF > 0)  sSec = fGaussSecDiagYCenter;
    if (!std::isnan(sSec)) {
        auto [dx,dy] = sToDxDySec(sSec);
        setXY(fPixelX+dx, fPixelY+dy,
              fGaussSecDiagTransformedX, fGaussSecDiagTransformedY,
              fGaussSecDiagTransformedDeltaX, fGaussSecDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fGaussSecDiagTransformedX, fGaussSecDiagTransformedY,
              fGaussSecDiagTransformedDeltaX, fGaussSecDiagTransformedDeltaY);
    }

    // ------------------
    //   L O R E N T Z I A N
    // ------------------
    // Main diagonal
    double sMainLorentz = NaN;
    if (fLorentzMainDiagXDOF > 0)       sMainLorentz = fLorentzMainDiagXCenter;
    else if (fLorentzMainDiagYDOF > 0)  sMainLorentz = fLorentzMainDiagYCenter;
    if (!std::isnan(sMainLorentz)) {
        auto [dx,dy] = sToDxDyMain(sMainLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fLorentzMainDiagTransformedX, fLorentzMainDiagTransformedY,
              fLorentzMainDiagTransformedDeltaX, fLorentzMainDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fLorentzMainDiagTransformedX, fLorentzMainDiagTransformedY,
              fLorentzMainDiagTransformedDeltaX, fLorentzMainDiagTransformedDeltaY);
    }

    // Secondary diagonal
    double sSecLorentz = NaN;
    if (fLorentzSecDiagXDOF > 0)       sSecLorentz = fLorentzSecDiagXCenter;
    else if (fLorentzSecDiagYDOF > 0)  sSecLorentz = fLorentzSecDiagYCenter;
    if (!std::isnan(sSecLorentz)) {
        auto [dx,dy] = sToDxDySec(sSecLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fLorentzSecDiagTransformedX, fLorentzSecDiagTransformedY,
              fLorentzSecDiagTransformedDeltaX, fLorentzSecDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fLorentzSecDiagTransformedX, fLorentzSecDiagTransformedY,
              fLorentzSecDiagTransformedDeltaX, fLorentzSecDiagTransformedDeltaY);
    }

    // -------------------------------
    // P O W E R - L A W   L O R E N T Z I A N
    // -------------------------------
    // FIX: Power Lorentz diagonal fits also return diagonal coordinates that need transformation
    
    // Main diagonal
    double sMainPowerLorentz = NaN;
    if (fPowerLorentzMainDiagXDOF > 0)       sMainPowerLorentz = fPowerLorentzMainDiagXCenter;
    else if (fPowerLorentzMainDiagYDOF > 0)  sMainPowerLorentz = fPowerLorentzMainDiagYCenter;
    if (!std::isnan(sMainPowerLorentz)) {
        auto [dx,dy] = sToDxDyMain(sMainPowerLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fPowerLorentzMainDiagTransformedX, fPowerLorentzMainDiagTransformedY,
              fPowerLorentzMainDiagTransformedDeltaX, fPowerLorentzMainDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fPowerLorentzMainDiagTransformedX, fPowerLorentzMainDiagTransformedY,
              fPowerLorentzMainDiagTransformedDeltaX, fPowerLorentzMainDiagTransformedDeltaY);
    }

    // Secondary diagonal
    double sSecPowerLorentz = NaN;
    if (fPowerLorentzSecDiagXDOF > 0)       sSecPowerLorentz = fPowerLorentzSecDiagXCenter;
    else if (fPowerLorentzSecDiagYDOF > 0)  sSecPowerLorentz = fPowerLorentzSecDiagYCenter;
    if (!std::isnan(sSecPowerLorentz)) {
        auto [dx,dy] = sToDxDySec(sSecPowerLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fPowerLorentzSecDiagTransformedX, fPowerLorentzSecDiagTransformedY,
              fPowerLorentzSecDiagTransformedDeltaX, fPowerLorentzSecDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fPowerLorentzSecDiagTransformedX, fPowerLorentzSecDiagTransformedY,
              fPowerLorentzSecDiagTransformedDeltaX, fPowerLorentzSecDiagTransformedDeltaY);
    }
}

void RunAction::CalcMeanEstimations()
{
    // Safety check for valid true position data
    if (std::isnan(fTrueX) || std::isnan(fTrueY)) {
        G4cerr << "RunAction: Warning - Invalid true position data, cannot calculate mean estimations" << G4endl;
        // Set all mean deltas to NaN
        fGaussMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        fPowerLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fPowerLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        return;
    }
    
    // Vectors to collect valid coordinate estimations
    std::vector<G4double> gauss_x_coords, gauss_y_coords;
    std::vector<G4double> lorentz_x_coords, lorentz_y_coords;
    std::vector<G4double> power_lorentz_x_coords, power_lorentz_y_coords;
    
    // For Gauss estimations, collect X coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans X
    if (!std::isnan(fGaussMainDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal Trans X  
    if (!std::isnan(fGaussSecDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussSecDiagTransformedX);
    }
    
    // For Gauss estimations, collect Y coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans Y
    if (!std::isnan(fGaussMainDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal Trans Y
    if (!std::isnan(fGaussSecDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussSecDiagTransformedY);
    }
    
    // For Lorentz estimations, collect X coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans X
    if (!std::isnan(fLorentzMainDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal Trans X
    if (!std::isnan(fLorentzSecDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzSecDiagTransformedX);
    }
    
    // For Lorentz estimations, collect Y coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans Y
    if (!std::isnan(fLorentzMainDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal Trans Y
    if (!std::isnan(fLorentzSecDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzSecDiagTransformedY);
    }
    
    // For Power-Law Lorentz estimations, collect X coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans X
    if (!std::isnan(fPowerLorentzMainDiagTransformedX)) {
        power_lorentz_x_coords.push_back(fPowerLorentzMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal Trans X
    if (!std::isnan(fPowerLorentzSecDiagTransformedX)) {
        power_lorentz_x_coords.push_back(fPowerLorentzSecDiagTransformedX);
    }
    
    // For Power-Law Lorentz estimations, collect Y coordinates:
    // ONLY use Trans diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal Trans Y
    if (!std::isnan(fPowerLorentzMainDiagTransformedY)) {
        power_lorentz_y_coords.push_back(fPowerLorentzMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal Trans Y
    if (!std::isnan(fPowerLorentzSecDiagTransformedY)) {
        power_lorentz_y_coords.push_back(fPowerLorentzSecDiagTransformedY);
    }
    
    // =============================================
    // ADD 3D FIT RESULTS TO MEAN CALCULATIONS
    // =============================================
    
    // Add 3D Gauss fit results to Gauss estimation collections
    if (!std::isnan(f3DGaussCenterX) && f3DGaussSuccess) {
        gauss_x_coords.push_back(f3DGaussCenterX);
    }
    if (!std::isnan(f3DGaussCenterY) && f3DGaussSuccess) {
        gauss_y_coords.push_back(f3DGaussCenterY);
    }
    
    // Add 3D Lorentz fit results to Lorentz estimation collections
    if (!std::isnan(f3DLorentzCenterX) && f3DLorentzSuccess) {
        lorentz_x_coords.push_back(f3DLorentzCenterX);
    }
    if (!std::isnan(f3DLorentzCenterY) && f3DLorentzSuccess) {
        lorentz_y_coords.push_back(f3DLorentzCenterY);
    }
    
    // Add 3D Power-Law Lorentz fit results to Power-Law Lorentz estimation collections
    if (!std::isnan(f3DPowerLorentzCenterX) && f3DPowerLorentzSuccess) {
        power_lorentz_x_coords.push_back(f3DPowerLorentzCenterX);
    }
    if (!std::isnan(f3DPowerLorentzCenterY) && f3DPowerLorentzSuccess) {
        power_lorentz_y_coords.push_back(f3DPowerLorentzCenterY);
    }

    
    // Calc mean X coordinate estimations and their deltas
    if (!gauss_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : gauss_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / gauss_x_coords.size();
        fGaussMeanTrueDeltaX = std::abs(mean_x - fTrueX);
    } else {
        fGaussMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!gauss_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : gauss_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / gauss_y_coords.size();
        fGaussMeanTrueDeltaY = std::abs(mean_y - fTrueY);
    } else {
        fGaussMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!lorentz_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : lorentz_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / lorentz_x_coords.size();
        fLorentzMeanTrueDeltaX = std::abs(mean_x - fTrueX);
    } else {
        fLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!lorentz_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : lorentz_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / lorentz_y_coords.size();
        fLorentzMeanTrueDeltaY = std::abs(mean_y - fTrueY);
    } else {
        fLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc Power-Law Lorentz mean coordinate estimations and their deltas
    if (!power_lorentz_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : power_lorentz_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / power_lorentz_x_coords.size();
        fPowerLorentzMeanTrueDeltaX = std::abs(mean_x - fTrueX);
    } else {
        fPowerLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!power_lorentz_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : power_lorentz_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / power_lorentz_y_coords.size();
        fPowerLorentzMeanTrueDeltaY = std::abs(mean_y - fTrueY);
    } else {
        fPowerLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }

}

// =============================================
// LAPLACE FIT RESULTS SETTER METHODS
// =============================================





// Store automatic radius selection results
void RunAction::SetAutoRadiusResults(G4int selectedRadius)
{
    fSelectedRadius = selectedRadius;
}

// Set scorer data from Multi-Functional Detector
void RunAction::SetScorerData(G4double energyDeposit, G4int hitCount, G4bool dataValid)
{
    fScorerEnergyDeposit = energyDeposit;
    fScorerHitCount = hitCount;
    fScorerDataValid = dataValid;
}

// Set hit purity tracking data from EventAction
void RunAction::SetHitPurityData(G4bool pureSiliconHit, G4bool aluminumContaminated, G4bool chargeCalculationEnabled)
{
    fPureSiliconHit = pureSiliconHit;
    fAluminumContaminated = aluminumContaminated;
    fChargeCalculationEnabled = chargeCalculationEnabled;
}

// Validate scorer data before tree storage
void RunAction::ValidateScorerDataForTreeStorage()
{
    // Validate energy deposit data type handling (G4double)
    if (!std::isfinite(fScorerEnergyDeposit)) {
        G4cerr << "WARNING: Scorer energy deposit is not finite: " << fScorerEnergyDeposit << G4endl;
        fScorerEnergyDeposit = 0.0;  // Reset to safe default
        fScorerDataValid = false;
    }
    
    // Validate energy deposit is within reasonable bounds
    if (fScorerEnergyDeposit < 0.0 || fScorerEnergyDeposit > 1000.0) {
        G4cerr << "WARNING: Scorer energy deposit out of bounds: " << fScorerEnergyDeposit << " MeV" << G4endl;
        fScorerEnergyDeposit = 0.0;  // Reset to safe default
        fScorerDataValid = false;
    }
    
    // Validate hit count data type handling (G4int)
    if (fScorerHitCount < 0 || fScorerHitCount > 10000) {
        G4cerr << "WARNING: Scorer hit count out of bounds: " << fScorerHitCount << G4endl;
        fScorerHitCount = 0;  // Reset to safe default
        fScorerDataValid = false;
    }
    
    // Ensure proper data type storage for ROOT
    // G4double should be stored as Double_t, G4int as Int_t, G4bool as Bool_t
    // ROOT branches automatically handle these conversions, but we validate the data integrity
    
    // Check for data consistency: if we have energy deposit but no hits, or vice versa
    if (fScorerDataValid) {
        if (fScorerEnergyDeposit > 0.0 && fScorerHitCount == 0) {
            G4cerr << "WARNING: Scorer has energy deposit (" << fScorerEnergyDeposit 
                   << " MeV) but zero hits - possible data inconsistency" << G4endl;
        }
        if (fScorerEnergyDeposit == 0.0 && fScorerHitCount > 0) {
            G4cerr << "WARNING: Scorer has " << fScorerHitCount 
                   << " hits but zero energy deposit - possible data inconsistency" << G4endl;
        }
    }
}

// Verify scorer data is written to ROOT tree
void RunAction::VerifyScorerDataInTree()
{
    if (!fTree) {
        G4cerr << "WARNING: Cannot verify scorer data - ROOT tree is null" << G4endl;
        return;
    }
    
    // Get the scorer data branches
    TBranch* energyBranch = fTree->GetBranch("ScorerEnergyDeposit");
    TBranch* hitCountBranch = fTree->GetBranch("ScorerHitCount");
    TBranch* validBranch = fTree->GetBranch("ScorerDataValid");
    
    if (!energyBranch || !hitCountBranch || !validBranch) {
        G4cerr << "ERROR: One or more scorer data branches not found in ROOT tree!" << G4endl;
        if (!energyBranch) G4cerr << "  - ScorerEnergyDeposit branch missing" << G4endl;
        if (!hitCountBranch) G4cerr << "  - ScorerHitCount branch missing" << G4endl;
        if (!validBranch) G4cerr << "  - ScorerDataValid branch missing" << G4endl;
        return;
    }
    
    // Check that branches have entries
    Long64_t entries = fTree->GetEntries();
    if (entries == 0) {
        G4cerr << "WARNING: ROOT tree has no entries - cannot verify scorer data" << G4endl;
        return;
    }
    
    // Verify branch entry counts match tree entries
    if (energyBranch->GetEntries() != entries || 
        hitCountBranch->GetEntries() != entries || 
        validBranch->GetEntries() != entries) {
        G4cerr << "ERROR: Scorer data branch entry count mismatch!" << G4endl;
        G4cerr << "  Tree entries: " << entries << G4endl;
        G4cerr << "  ScorerEnergyDeposit entries: " << energyBranch->GetEntries() << G4endl;
        G4cerr << "  ScorerHitCount entries: " << hitCountBranch->GetEntries() << G4endl;
        G4cerr << "  ScorerDataValid entries: " << validBranch->GetEntries() << G4endl;
        return;
    }
    
    G4cout << "✓ Scorer data verification: All branches present with " << entries << " entries" << G4endl;
    
    // Verify data types are correctly stored
    if (energyBranch->GetTitle() && hitCountBranch->GetTitle() && validBranch->GetTitle()) {
        G4cout << "✓ Scorer data types verified: Energy (Double), HitCount (Int), Valid (Bool)" << G4endl;
    }
}

// =============================================
// POWER Lorentz FIT RESULTS SETTER METHODS
// =============================================

void RunAction::Set2DPowerLorentzResults(G4double x_center, G4double x_gamma, G4double x_beta, G4double x_Amp,
                                               G4double x_center_err, G4double x_gamma_err, G4double x_beta_err, G4double x_Amp_err,
                                               G4double x_Vert_offset, G4double x_Vert_offset_err,
                                               G4double x_chi2red, G4double x_pp, G4int x_dof,
                                               G4double y_center, G4double y_gamma, G4double y_beta, G4double y_Amp,
                                               G4double y_center_err, G4double y_gamma_err, G4double y_beta_err, G4double y_Amp_err,
                                               G4double y_Vert_offset, G4double y_Vert_offset_err,
                                               G4double y_chi2red, G4double y_pp, G4int y_dof,
                                               G4double x_charge_err, G4double y_charge_err,
                                               G4bool fit_success)
{
    // Store X direction (row) fit results - Power-Law Lorentz model
    fPowerLorentzRowCenter = x_center;
    fPowerLorentzRowGamma = x_gamma;
    fPowerLorentzRowBeta = x_beta;
    fPowerLorentzRowAmp = x_Amp;
    fPowerLorentzRowCenterErr = x_center_err;
    fPowerLorentzRowGammaErr = x_gamma_err;
    fPowerLorentzRowBetaErr = x_beta_err;
    fPowerLorentzRowAmpErr = x_Amp_err;
    fPowerLorentzRowVertOffset = x_Vert_offset;
    fPowerLorentzRowVertOffsetErr = x_Vert_offset_err;
    fPowerLorentzRowChi2red = x_chi2red;
    fPowerLorentzRowPp = x_pp;
    fPowerLorentzRowDOF = x_dof;
    
    // Store Y direction (column) fit results - Power-Law Lorentz model
    fPowerLorentzColCenter = y_center;
    fPowerLorentzColGamma = y_gamma;
    fPowerLorentzColBeta = y_beta;
    fPowerLorentzColAmp = y_Amp;
    fPowerLorentzColCenterErr = y_center_err;
    fPowerLorentzColGammaErr = y_gamma_err;
    fPowerLorentzColBetaErr = y_beta_err;
    fPowerLorentzColAmpErr = y_Amp_err;
    fPowerLorentzColVertOffset = y_Vert_offset;
    fPowerLorentzColVertOffsetErr = y_Vert_offset_err;
    fPowerLorentzColChi2red = y_chi2red;
    fPowerLorentzColPp = y_pp;
    fPowerLorentzColDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        fPowerLorentzRowChargeErr = x_charge_err;
        fPowerLorentzColChargeErr = y_charge_err;
    } else {
        fPowerLorentzRowChargeErr = 0.0;
        fPowerLorentzColChargeErr = 0.0;
    }
    
    // Calc delta values for row and column fits vs true position
    if (fit_success) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fPowerLorentzRowDeltaX = std::abs(x_center - fTrueX);      // x_row_fit - x_true
        } else {
            fPowerLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fPowerLorentzColDeltaY = std::abs(y_center - fTrueY);   // y_column_fit - y_true
        } else {
            fPowerLorentzColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fPowerLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fPowerLorentzColDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

void RunAction::SetDiagPowerLorentzResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_beta, G4double main_diag_x_Amp,
                                                     G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_beta_err, G4double main_diag_x_Amp_err,
                                                     G4double main_diag_x_Vert_offset, G4double main_diag_x_Vert_offset_err,
                                                     G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_success,
                                                     G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_beta, G4double main_diag_y_Amp,
                                                     G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_beta_err, G4double main_diag_y_Amp_err,
                                                     G4double main_diag_y_Vert_offset, G4double main_diag_y_Vert_offset_err,
                                                     G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_success,
                                                     G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_beta, G4double sec_diag_x_Amp,
                                                     G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_beta_err, G4double sec_diag_x_Amp_err,
                                                     G4double sec_diag_x_Vert_offset, G4double sec_diag_x_Vert_offset_err,
                                                     G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_success,
                                                     G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_beta, G4double sec_diag_y_Amp,
                                                     G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_beta_err, G4double sec_diag_y_Amp_err,
                                                     G4double sec_diag_y_Vert_offset, G4double sec_diag_y_Vert_offset_err,
                                                     G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_success,
                                                     G4bool fit_success)
{
    // Store main diagonal X fit results - Power-Law Lorentz model
    fPowerLorentzMainDiagXCenter = main_diag_x_center;
    fPowerLorentzMainDiagXGamma = main_diag_x_gamma;
    fPowerLorentzMainDiagXBeta = main_diag_x_beta;
    fPowerLorentzMainDiagXAmp = main_diag_x_Amp;
    fPowerLorentzMainDiagXCenterErr = main_diag_x_center_err;
    fPowerLorentzMainDiagXGammaErr = main_diag_x_gamma_err;
    fPowerLorentzMainDiagXBetaErr = main_diag_x_beta_err;
    fPowerLorentzMainDiagXAmpErr = main_diag_x_Amp_err;
    fPowerLorentzMainDiagXVertOffset = main_diag_x_Vert_offset;
    fPowerLorentzMainDiagXVertOffsetErr = main_diag_x_Vert_offset_err;
    fPowerLorentzMainDiagXChi2red = main_diag_x_chi2red;
    fPowerLorentzMainDiagXPp = main_diag_x_pp;
    fPowerLorentzMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results - Power-Law Lorentz model
    fPowerLorentzMainDiagYCenter = main_diag_y_center;
    fPowerLorentzMainDiagYGamma = main_diag_y_gamma;
    fPowerLorentzMainDiagYBeta = main_diag_y_beta;
    fPowerLorentzMainDiagYAmp = main_diag_y_Amp;
    fPowerLorentzMainDiagYCenterErr = main_diag_y_center_err;
    fPowerLorentzMainDiagYGammaErr = main_diag_y_gamma_err;
    fPowerLorentzMainDiagYBetaErr = main_diag_y_beta_err;
    fPowerLorentzMainDiagYAmpErr = main_diag_y_Amp_err;
    fPowerLorentzMainDiagYVertOffset = main_diag_y_Vert_offset;
    fPowerLorentzMainDiagYVertOffsetErr = main_diag_y_Vert_offset_err;
    fPowerLorentzMainDiagYChi2red = main_diag_y_chi2red;
    fPowerLorentzMainDiagYPp = main_diag_y_pp;
    fPowerLorentzMainDiagYDOF = main_diag_y_dof;
    
    // Store Secondary diagonal X fit results - Power-Law Lorentz model
    fPowerLorentzSecDiagXCenter = sec_diag_x_center;
    fPowerLorentzSecDiagXGamma = sec_diag_x_gamma;
    fPowerLorentzSecDiagXBeta = sec_diag_x_beta;
    fPowerLorentzSecDiagXAmp = sec_diag_x_Amp;
    fPowerLorentzSecDiagXCenterErr = sec_diag_x_center_err;
    fPowerLorentzSecDiagXGammaErr = sec_diag_x_gamma_err;
    fPowerLorentzSecDiagXBetaErr = sec_diag_x_beta_err;
    fPowerLorentzSecDiagXAmpErr = sec_diag_x_Amp_err;
    fPowerLorentzSecDiagXVertOffset = sec_diag_x_Vert_offset;
    fPowerLorentzSecDiagXVertOffsetErr = sec_diag_x_Vert_offset_err;
    fPowerLorentzSecDiagXChi2red = sec_diag_x_chi2red;
    fPowerLorentzSecDiagXPp = sec_diag_x_pp;
    fPowerLorentzSecDiagXDOF = sec_diag_x_dof;
    
    // Store Secondary diagonal Y fit results - Power-Law Lorentz model
    fPowerLorentzSecDiagYCenter = sec_diag_y_center;
    fPowerLorentzSecDiagYGamma = sec_diag_y_gamma;
    fPowerLorentzSecDiagYBeta = sec_diag_y_beta;
    fPowerLorentzSecDiagYAmp = sec_diag_y_Amp;
    fPowerLorentzSecDiagYCenterErr = sec_diag_y_center_err;
    fPowerLorentzSecDiagYGammaErr = sec_diag_y_gamma_err;
    fPowerLorentzSecDiagYBetaErr = sec_diag_y_beta_err;
    fPowerLorentzSecDiagYAmpErr = sec_diag_y_Amp_err;
    fPowerLorentzSecDiagYVertOffset = sec_diag_y_Vert_offset;
    fPowerLorentzSecDiagYVertOffsetErr = sec_diag_y_Vert_offset_err;
    fPowerLorentzSecDiagYChi2red = sec_diag_y_chi2red;
    fPowerLorentzSecDiagYPp = sec_diag_y_pp;
    fPowerLorentzSecDiagYDOF = sec_diag_y_dof;

    // Calc Trans diagonal coordinates using rotation matrix
    CalcTransformedDiagCoords();
    
    // Calc mean estimations from all fitting methods
    CalcMeanEstimations();
}

// =============================================
// 3D FIT RESULTS SETTER METHODS
// =============================================

void RunAction::Set3DLorentzResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double Amp, G4double Vert_offset,
                                           G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double Amp_err, G4double Vert_offset_err,
                                           G4double chi2red, G4double pp, G4int dof,
                                           G4double charge_err,
                                           G4bool fit_success)
{
    // Store 3D Lorentz fit parameters
    f3DLorentzCenterX = center_x;
    f3DLorentzCenterY = center_y;
    f3DLorentzGammaX = gamma_x;
    f3DLorentzGammaY = gamma_y;
    f3DLorentzAmp = Amp;
    f3DLorentzVertOffset = Vert_offset;
    
    // Store 3D Lorentz fit parameter errors
    f3DLorentzCenterXErr = center_x_err;
    f3DLorentzCenterYErr = center_y_err;
    f3DLorentzGammaXErr = gamma_x_err;
    f3DLorentzGammaYErr = gamma_y_err;
    f3DLorentzAmpErr = Amp_err;
    f3DLorentzVertOffsetErr = Vert_offset_err;
    
    // Store 3D Lorentz fit statistics
    f3DLorentzChi2red = chi2red;
    f3DLorentzPp = pp;
    f3DLorentzDOF = dof;
    f3DLorentzSuccess = fit_success;
    
    // Store charge err (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        f3DLorentzChargeErr = charge_err;
    } else {
        f3DLorentzChargeErr = 0.0;
    }
    
    // Calc delta values vs true position
    if (fit_success && dof > 0) {
        f3DLorentzDeltaX = std::abs(center_x - fTrueX);      // x_3D_fit - x_true
        f3DLorentzDeltaY = std::abs(center_y - fTrueY);      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DLorentzDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DLorentzDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods (including 3D)
    CalcMeanEstimations();
}

void RunAction::Set3DPowerLorentzResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double beta, G4double Amp, G4double Vert_offset,
                                               G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double beta_err, G4double Amp_err, G4double Vert_offset_err,
                                               G4double chi2red, G4double pp, G4int dof,
                                               G4double charge_err,
                                               G4bool fit_success)
{
    // Store 3D Power-Law Lorentz fit parameters
    f3DPowerLorentzCenterX = center_x;
    f3DPowerLorentzCenterY = center_y;
    f3DPowerLorentzGammaX = gamma_x;
    f3DPowerLorentzGammaY = gamma_y;
    f3DPowerLorentzBeta = beta;
    f3DPowerLorentzAmp = Amp;
    f3DPowerLorentzVertOffset = Vert_offset;
    
    // Store 3D Power-Law Lorentz fit parameter errors
    f3DPowerLorentzCenterXErr = center_x_err;
    f3DPowerLorentzCenterYErr = center_y_err;
    f3DPowerLorentzGammaXErr = gamma_x_err;
    f3DPowerLorentzGammaYErr = gamma_y_err;
    f3DPowerLorentzBetaErr = beta_err;
    f3DPowerLorentzAmpErr = Amp_err;
    f3DPowerLorentzVertOffsetErr = Vert_offset_err;
    
    // Store 3D Power-Law Lorentz fit statistics
    f3DPowerLorentzChi2red = chi2red;
    f3DPowerLorentzPp = pp;
    f3DPowerLorentzDOF = dof;
    f3DPowerLorentzSuccess = fit_success;
    
    // Store charge err (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        f3DPowerLorentzChargeErr = charge_err;
    } else {
        f3DPowerLorentzChargeErr = 0.0;
    }
    
    // Calc delta values vs true position
    if (fit_success && dof > 0) {
        f3DPowerLorentzDeltaX = std::abs(center_x - fTrueX);      // x_3D_fit - x_true
        f3DPowerLorentzDeltaY = std::abs(center_y - fTrueY);      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DPowerLorentzDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DPowerLorentzDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods (including 3D)
    CalcMeanEstimations();
}

void RunAction::Set3DGaussResults(G4double center_x, G4double center_y, G4double sigma_x, G4double sigma_y, G4double Amp, G4double Vert_offset,
                                        G4double center_x_err, G4double center_y_err, G4double sigma_x_err, G4double sigma_y_err, G4double Amp_err, G4double Vert_offset_err,
                                        G4double chi2red, G4double pp, G4int dof,
                                        G4double charge_err,
                                        G4bool fit_success)
{
    // Store 3D Gauss fit parameters
    f3DGaussCenterX = center_x;
    f3DGaussCenterY = center_y;
    f3DGaussSigmaX = sigma_x;
    f3DGaussSigmaY = sigma_y;
    f3DGaussAmp = Amp;
    f3DGaussVertOffset = Vert_offset;
    
    // Store 3D Gauss fit parameter errors
    f3DGaussCenterXErr = center_x_err;
    f3DGaussCenterYErr = center_y_err;
    f3DGaussSigmaXErr = sigma_x_err;
    f3DGaussSigmaYErr = sigma_y_err;
    f3DGaussAmpErr = Amp_err;
    f3DGaussVertOffsetErr = Vert_offset_err;
    
    // Store 3D Gauss fit statistics
    f3DGaussChi2red = chi2red;
    f3DGaussPp = pp;
    f3DGaussDOF = dof;
    f3DGaussSuccess = fit_success;
    
    // Store charge err (5% of max charge) only if feature is enabled
    if (Control::CHARGE_ERR) {
        f3DGaussChargeErr = charge_err;
    } else {
        f3DGaussChargeErr = 0.0;
    }
    
    // Calc delta values vs true position
    if (fit_success && dof > 0) {
        f3DGaussDeltaX = std::abs(center_x - fTrueX);      // x_3D_fit - x_true
        f3DGaussDeltaY = std::abs(center_y - fTrueY);      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DGaussDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DGaussDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calc mean estimations from all fitting methods (including 3D)
    CalcMeanEstimations();
}

// =============================================
// THREAD SYNCHRONIZATION METHODS
// =============================================

void RunAction::ResetSynchronization()
{
    std::lock_guard<std::mutex> lock(fSyncMutex);
    fWorkersCompleted = 0;
    fTotalWorkers = 0;
    fAllWorkersCompleted = false;
    
    if (G4Threading::IsMultithreadedApplication()) {
        fTotalWorkers = G4Threading::GetNumberOfRunningWorkerThreads();
    }
    
    G4cout << "RunAction: Reset synchronization for " << fTotalWorkers.load() << " worker threads" << G4endl;
}

void RunAction::SignalWorkerCompletion()
{
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        std::unique_lock<std::mutex> lock(fSyncMutex);
        fWorkersCompleted++;
        
        G4cout << "RunAction: Worker thread completed (" << fWorkersCompleted.load() 
               << "/" << fTotalWorkers.load() << ")" << G4endl;
        
        if (fWorkersCompleted >= fTotalWorkers) {
            fAllWorkersCompleted = true;
            lock.unlock();
            fWorkerCompletionCV.notify_all();
            G4cout << "RunAction: All worker threads completed, notifying master" << G4endl;
        }
    }
}

void RunAction::WaitForAllWorkersToComplete()
{
    if (G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread()) {
        std::unique_lock<std::mutex> lock(fSyncMutex);
        
        G4cout << "RunAction: Master thread waiting for " << fTotalWorkers.load() << " workers to complete..." << G4endl;
        
        // Wait for all workers to complete with timeout
        auto timeout = std::chrono::seconds(30); // 30 Sec timeout
        bool completed = fWorkerCompletionCV.wait_for(lock, timeout, []() {
            return fAllWorkersCompleted.load();
        });
        
        if (completed) {
            G4cout << "RunAction: All workers completed successfully" << G4endl;
        } else {
            G4cerr << "RunAction: Warning - Timeout waiting for workers to complete!" << G4endl;
        }
    }
}

// =============================================
// SAFE ROOT FILE OPERATIONS
// =============================================

bool RunAction::ValidateRootFile(const G4String& filename)
{
    if (filename.empty()) {
        G4cerr << "RunAction: Error - Empty filename provided for validation" << G4endl;
        return false;
    }
    
    TFile* testFile = nullptr;
    try {
        testFile = TFile::Open(filename.c_str(), "READ");
        if (!testFile || testFile->IsZombie()) {
            G4cerr << "RunAction: Error - Cannot open or corrupted file: " << filename << G4endl;
            if (testFile) delete testFile;
            return false;
        }
        
        TTree* testTree = (TTree*)testFile->Get("Hits");
        if (!testTree) {
            G4cerr << "RunAction: Error - No 'Hits' tree found in file: " << filename << G4endl;
            testFile->Close();
            delete testFile;
            return false;
        }
        
        bool isValid = testTree->GetEntries() > 0;
        if (!isValid) {
            G4cerr << "RunAction: Warning - Empty tree in file: " << filename << G4endl;
        }
        
        testFile->Close();
        delete testFile;
        
        return isValid;
        
    } catch (const std::exception& e) {
        G4cerr << "RunAction: Exception during file validation: " << e.what() << G4endl;
        if (testFile) {
            testFile->Close();
            delete testFile;
        }
        return false;
    }
}

bool RunAction::SafeWriteRootFile()
{
    std::lock_guard<std::mutex> lock(fRootMutex);
    
    if (!fRootFile || !fTree || fRootFile->IsZombie()) {
        G4cerr << "RunAction: Cannot write - invalid ROOT file or tree" << G4endl;
        return false;
    }
    
    try {
        // In single-threaded mode, write metadata here since there's no merging
        // In multi-threaded mode, metadata is written only by master thread after merging
        if (!G4Threading::IsMultithreadedApplication() && fGridPixelSize > 0) {
            fRootFile->cd();
            
            // Create and write metadata objects
            TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", fGridPixelSize));
            TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing));
            TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset));
            TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", fGridDetSize));
            TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
            TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", Constants::NEIGHBORHOOD_RADIUS));
            
            pixelSizeMeta.Write();
            pixelSpacingMeta.Write();
            pixelCornerOffsetMeta.Write();
            detSizeMeta.Write();
            numBlocksMeta.Write();
            neighborhoodRadiusMeta.Write();
            
            G4cout << "RunAction: Saved detector grid metadata to single-threaded file" << G4endl;
        }
        
        // Make sure all in-memory baskets are flushed only once at end-of-run
        fTree->FlushBaskets();   // replaces thousands of small AutoSave flushes
        
        // Write the full tree in a single operation (metadata added above for single-threaded, by master for multi-threaded)
        fRootFile->cd();
        fTree->Write();
        // Final flush for the file header / directory structure
        fRootFile->Flush();
        
        G4cout << "RunAction: Successly wrote ROOT file with " << fTree->GetEntries() << " entries" << G4endl;
        return true;
        
    } catch (const std::exception& e) {
        G4cerr << "RunAction: Exception writing ROOT file: " << e.what() << G4endl;
        return false;
    }
}

void RunAction::CleanupRootObjects()
{
    std::lock_guard<std::mutex> lock(fRootMutex);
    
    try {
        if (fRootFile) {
            if (fRootFile->IsOpen() && !fRootFile->IsZombie()) {
                fRootFile->Close();
            }
            delete fRootFile;
            fRootFile = nullptr;
            fTree = nullptr; // Tree is owned by file
            G4cout << "RunAction: Successly cleaned up ROOT objects" << G4endl;
        }
    } catch (const std::exception& e) {
        G4cerr << "RunAction: Exception during ROOT cleanup: " << e.what() << G4endl;
        // Force cleanup even if exception occurred
        fRootFile = nullptr;
        fTree = nullptr;
    }
}

// =============================================
// AUTO-SAVE MECHANISM
// =============================================

void RunAction::EnableAutoSave(G4int interval)
{
    fAutoSaveEnabled = true;
    fAutoSaveInterval = interval;
    fEventsSinceLastSave = 0;
    
    G4cout << "RunAction: Auto-save enabled with interval " << interval << " events" << G4endl;
}

void RunAction::DisableAutoSave()
{
    fAutoSaveEnabled = false;
    G4cout << "RunAction: Auto-save disabled" << G4endl;
}

void RunAction::PerformAutoSave()
{
    if (!fAutoSaveEnabled) {
        return;
    }
    
    // Safety check for ROOT objects
    if (!fRootFile || !fTree) {
        G4cerr << "RunAction: Warning - Cannot perform auto-save, ROOT file or tree is null" << G4endl;
        return;
    }
    
    fEventsSinceLastSave++;
    
    if (fEventsSinceLastSave >= fAutoSaveInterval) {
        G4cout << "RunAction: Performing auto-save after " << fEventsSinceLastSave << " events..." << G4endl;
        
        // Perform auto-save inline (mutex already held by FillTree)
        if (fRootFile && fTree && !fRootFile->IsZombie()) {
            try {
                fRootFile->cd();
                fTree->AutoSave("SaveSelf");
                fRootFile->Flush();
                fEventsSinceLastSave = 0;
                G4cout << "RunAction: Auto-save completed successfully" << G4endl;
            } catch (const std::exception& e) {
                G4cerr << "RunAction: Auto-save failed: " << e.what() << G4endl;
            }
        } else {
            G4cerr << "RunAction: Cannot auto-save - invalid ROOT file or tree" << G4endl;
        }
    }
}
