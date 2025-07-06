#include "RunAction.hh"
#include "Constants.hh"
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

// ROOT includes
#include "TFile.h"
#include "TTree.h"
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
static std::mutex gRootInitMutex;

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
  fGaussColumnDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzColumnDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize transformed diagonal coordinate variables
  fGaussMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize mean estimation variables
  fGaussMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Gaussian fit variables
  fGaussFitRowAmplitude(0),
  fGaussFitRowAmplitudeErr(0),
  fGaussFitRowStdev(0),
  fGaussFitRowStdevErr(0),
  fGaussFitRowVerticalOffset(0),
  fGaussFitRowVerticalOffsetErr(0),
  fGaussFitRowCenter(0),
  fGaussFitRowCenterErr(0),
  fGaussFitRowChi2red(0),
  fGaussFitRowPp(0),
  fGaussFitRowDOF(0),
  // Initialize charge uncertainties (5% of max charge)
  fGaussFitRowChargeUncertainty(0),
  fGaussFitColumnChargeUncertainty(0),
  fLorentzFitRowChargeUncertainty(0),
  fLorentzFitColumnChargeUncertainty(0),
  fGaussFitColumnAmplitude(0),
  fGaussFitColumnAmplitudeErr(0),
  fGaussFitColumnStdev(0),
  fGaussFitColumnStdevErr(0),
  fGaussFitColumnVerticalOffset(0),
  fGaussFitColumnVerticalOffsetErr(0),
  fGaussFitColumnCenter(0),
  fGaussFitColumnCenterErr(0),
  fGaussFitColumnChi2red(0),
  fGaussFitColumnPp(0),
  fGaussFitColumnDOF(0),
  fGaussFitMainDiagXAmplitude(0),
  fGaussFitMainDiagXAmplitudeErr(0),
  fGaussFitMainDiagXStdev(0),
  fGaussFitMainDiagXStdevErr(0),
  fGaussFitMainDiagXVerticalOffset(0),
  fGaussFitMainDiagXVerticalOffsetErr(0),
  fGaussFitMainDiagXCenter(0),
  fGaussFitMainDiagXCenterErr(0),
  fGaussFitMainDiagXChi2red(0),
  fGaussFitMainDiagXPp(0),
  fGaussFitMainDiagXDOF(0),
  fGaussFitMainDiagYAmplitude(0),
  fGaussFitMainDiagYAmplitudeErr(0),
  fGaussFitMainDiagYStdev(0),
  fGaussFitMainDiagYStdevErr(0),
  fGaussFitMainDiagYVerticalOffset(0),
  fGaussFitMainDiagYVerticalOffsetErr(0),
  fGaussFitMainDiagYCenter(0),
  fGaussFitMainDiagYCenterErr(0),
  fGaussFitMainDiagYChi2red(0),
  fGaussFitMainDiagYPp(0),
  fGaussFitMainDiagYDOF(0),
  fGaussFitSecondDiagXAmplitude(0),
  fGaussFitSecondDiagXAmplitudeErr(0),
  fGaussFitSecondDiagXStdev(0),
  fGaussFitSecondDiagXStdevErr(0),
  fGaussFitSecondDiagXVerticalOffset(0),
  fGaussFitSecondDiagXVerticalOffsetErr(0),
  fGaussFitSecondDiagXCenter(0),
  fGaussFitSecondDiagXCenterErr(0),
  fGaussFitSecondDiagXChi2red(0),
  fGaussFitSecondDiagXPp(0),
  fGaussFitSecondDiagXDOF(0),
  fGaussFitSecondDiagYAmplitude(0),
  fGaussFitSecondDiagYAmplitudeErr(0),
  fGaussFitSecondDiagYStdev(0),
  fGaussFitSecondDiagYStdevErr(0),
  fGaussFitSecondDiagYVerticalOffset(0),
  fGaussFitSecondDiagYVerticalOffsetErr(0),
  fGaussFitSecondDiagYCenter(0),
  fGaussFitSecondDiagYCenterErr(0),
  fGaussFitSecondDiagYChi2red(0),
  fGaussFitSecondDiagYPp(0),
  fGaussFitSecondDiagYDOF(0),
  // Initialize Lorentzian fit variables
  fLorentzFitRowAmplitude(0),
  fLorentzFitRowAmplitudeErr(0),
  fLorentzFitRowGamma(0),
  fLorentzFitRowGammaErr(0),
  fLorentzFitRowVerticalOffset(0),
  fLorentzFitRowVerticalOffsetErr(0),
  fLorentzFitRowCenter(0),
  fLorentzFitRowCenterErr(0),
  fLorentzFitRowChi2red(0),
  fLorentzFitRowPp(0),
  fLorentzFitRowDOF(0),
  fLorentzFitColumnAmplitude(0),
  fLorentzFitColumnGamma(0),
  fLorentzFitColumnGammaErr(0),
  fLorentzFitColumnVerticalOffset(0),
  fLorentzFitColumnVerticalOffsetErr(0),
  fLorentzFitColumnCenter(0),
  fLorentzFitColumnCenterErr(0),
  fLorentzFitColumnChi2red(0),
  fLorentzFitColumnPp(0),
  fLorentzFitColumnDOF(0),
  fLorentzFitMainDiagXAmplitude(0),
  fLorentzFitMainDiagXAmplitudeErr(0),
  fLorentzFitMainDiagXGamma(0),
  fLorentzFitMainDiagXGammaErr(0),
  fLorentzFitMainDiagXVerticalOffset(0),
  fLorentzFitMainDiagXVerticalOffsetErr(0),
  fLorentzFitMainDiagXCenter(0),
  fLorentzFitMainDiagXCenterErr(0),
  fLorentzFitMainDiagXChi2red(0),
  fLorentzFitMainDiagXPp(0),
  fLorentzFitMainDiagXDOF(0),
  fLorentzFitMainDiagYAmplitude(0),
  fLorentzFitMainDiagYAmplitudeErr(0),
  fLorentzFitMainDiagYGamma(0),
  fLorentzFitMainDiagYGammaErr(0),
  fLorentzFitMainDiagYVerticalOffset(0),
  fLorentzFitMainDiagYVerticalOffsetErr(0),
  fLorentzFitMainDiagYCenter(0),
  fLorentzFitMainDiagYCenterErr(0),
  fLorentzFitMainDiagYChi2red(0),
  fLorentzFitMainDiagYPp(0),
  fLorentzFitMainDiagYDOF(0),
  fLorentzFitSecondDiagXAmplitude(0),
  fLorentzFitSecondDiagXAmplitudeErr(0),
  fLorentzFitSecondDiagXGamma(0),
  fLorentzFitSecondDiagXGammaErr(0),
  fLorentzFitSecondDiagXVerticalOffset(0),
  fLorentzFitSecondDiagXVerticalOffsetErr(0),
  fLorentzFitSecondDiagXCenter(0),
  fLorentzFitSecondDiagXCenterErr(0),
  fLorentzFitSecondDiagXChi2red(0),
  fLorentzFitSecondDiagXPp(0),
  fLorentzFitSecondDiagXDOF(0),
  fLorentzFitSecondDiagYAmplitude(0),
  fLorentzFitSecondDiagYAmplitudeErr(0),
  fLorentzFitSecondDiagYGamma(0),
  fLorentzFitSecondDiagYGammaErr(0),
  fLorentzFitSecondDiagYVerticalOffset(0),
  fLorentzFitSecondDiagYVerticalOffsetErr(0),
  fLorentzFitSecondDiagYCenter(0),
  fLorentzFitSecondDiagYCenterErr(0),
  fLorentzFitSecondDiagYChi2red(0),
  fLorentzFitSecondDiagYPp(0),
  fLorentzFitSecondDiagYDOF(0),
  // Initialize Power-Law Lorentzian fit variables
  fPowerLorentzFitRowAmplitude(0),
  fPowerLorentzFitRowAmplitudeErr(0),
  fPowerLorentzFitRowBeta(0),
  fPowerLorentzFitRowBetaErr(0),
  fPowerLorentzFitRowGamma(0),
  fPowerLorentzFitRowGammaErr(0),
  fPowerLorentzFitRowVerticalOffset(0),
  fPowerLorentzFitRowVerticalOffsetErr(0),
  fPowerLorentzFitRowCenter(0),
  fPowerLorentzFitRowCenterErr(0),
  fPowerLorentzFitRowChi2red(0),
  fPowerLorentzFitRowPp(0),
  fPowerLorentzFitRowDOF(0),
  fPowerLorentzFitColumnAmplitude(0),
  fPowerLorentzFitColumnAmplitudeErr(0),
  fPowerLorentzFitColumnBeta(0),
  fPowerLorentzFitColumnBetaErr(0),
  fPowerLorentzFitColumnGamma(0),
  fPowerLorentzFitColumnGammaErr(0),
  fPowerLorentzFitColumnVerticalOffset(0),
  fPowerLorentzFitColumnVerticalOffsetErr(0),
  fPowerLorentzFitColumnCenter(0),
  fPowerLorentzFitColumnCenterErr(0),
  fPowerLorentzFitColumnChi2red(0),
  fPowerLorentzFitColumnPp(0),
  fPowerLorentzFitColumnDOF(0),
  fPowerLorentzFitMainDiagXAmplitude(0),
  fPowerLorentzFitMainDiagXAmplitudeErr(0),
  fPowerLorentzFitMainDiagXBeta(0),
  fPowerLorentzFitMainDiagXBetaErr(0),
  fPowerLorentzFitMainDiagXGamma(0),
  fPowerLorentzFitMainDiagXGammaErr(0),
  fPowerLorentzFitMainDiagXVerticalOffset(0),
  fPowerLorentzFitMainDiagXVerticalOffsetErr(0),
  fPowerLorentzFitMainDiagXCenter(0),
  fPowerLorentzFitMainDiagXCenterErr(0),
  fPowerLorentzFitMainDiagXChi2red(0),
  fPowerLorentzFitMainDiagXPp(0),
  fPowerLorentzFitMainDiagXDOF(0),
  fPowerLorentzFitMainDiagYAmplitude(0),
  fPowerLorentzFitMainDiagYAmplitudeErr(0),
  fPowerLorentzFitMainDiagYBeta(0),
  fPowerLorentzFitMainDiagYBetaErr(0),
  fPowerLorentzFitMainDiagYGamma(0),
  fPowerLorentzFitMainDiagYGammaErr(0),
  fPowerLorentzFitMainDiagYVerticalOffset(0),
  fPowerLorentzFitMainDiagYVerticalOffsetErr(0),
  fPowerLorentzFitMainDiagYCenter(0),
  fPowerLorentzFitMainDiagYCenterErr(0),
  fPowerLorentzFitMainDiagYChi2red(0),
  fPowerLorentzFitMainDiagYPp(0),
  fPowerLorentzFitMainDiagYDOF(0),
  fPowerLorentzFitSecondDiagXAmplitude(0),
  fPowerLorentzFitSecondDiagXAmplitudeErr(0),
  fPowerLorentzFitSecondDiagXBeta(0),
  fPowerLorentzFitSecondDiagXBetaErr(0),
  fPowerLorentzFitSecondDiagXGamma(0),
  fPowerLorentzFitSecondDiagXGammaErr(0),
  fPowerLorentzFitSecondDiagXVerticalOffset(0),
  fPowerLorentzFitSecondDiagXVerticalOffsetErr(0),
  fPowerLorentzFitSecondDiagXCenter(0),
  fPowerLorentzFitSecondDiagXCenterErr(0),
  fPowerLorentzFitSecondDiagXChi2red(0),
  fPowerLorentzFitSecondDiagXPp(0),
  fPowerLorentzFitSecondDiagXDOF(0),
  fPowerLorentzFitSecondDiagYAmplitude(0),
  fPowerLorentzFitSecondDiagYAmplitudeErr(0),
  fPowerLorentzFitSecondDiagYBeta(0),
  fPowerLorentzFitSecondDiagYBetaErr(0),
  fPowerLorentzFitSecondDiagYGamma(0),
  fPowerLorentzFitSecondDiagYGammaErr(0),
  fPowerLorentzFitSecondDiagYVerticalOffset(0),
  fPowerLorentzFitSecondDiagYVerticalOffsetErr(0),
  fPowerLorentzFitSecondDiagYCenter(0),
  fPowerLorentzFitSecondDiagYCenterErr(0),
  fPowerLorentzFitSecondDiagYChi2red(0),
  fPowerLorentzFitSecondDiagYPp(0),
  fPowerLorentzFitSecondDiagYDOF(0),
  // Initialize Power-Law Lorentzian delta variables
  fPowerLorentzRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzColumnDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Power-Law Lorentzian transformed diagonal coordinate variables
  fPowerLorentzMainDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecondDiagTransformedX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecondDiagTransformedY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMainDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecondDiagTransformedDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzSecondDiagTransformedDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize Power-Law Lorentzian mean estimation variables
  fPowerLorentzMeanTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fPowerLorentzMeanTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D fitting delta variables
  f3DLorentzianDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DLorentzianDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  f3DPowerLorentzianDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DPowerLorentzianDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D Lorentzian fit variables
  f3DLorentzianFitCenterX(0),
  f3DLorentzianFitCenterY(0),
  f3DLorentzianFitGammaX(0),
  f3DLorentzianFitGammaY(0),
  f3DLorentzianFitAmplitude(0),
  f3DLorentzianFitVerticalOffset(0),
  f3DLorentzianFitCenterXErr(0),
  f3DLorentzianFitCenterYErr(0),
  f3DLorentzianFitGammaXErr(0),
  f3DLorentzianFitGammaYErr(0),
  f3DLorentzianFitAmplitudeErr(0),
  f3DLorentzianFitVerticalOffsetErr(0),
  f3DLorentzianFitChi2red(0),
  f3DLorentzianFitPp(0),
  f3DLorentzianFitDOF(0),
  f3DLorentzianFitChargeUncertainty(0),
  f3DLorentzianFitSuccessful(false),
  // Initialize 3D Power-Law Lorentzian fit variables
  f3DPowerLorentzianFitCenterX(0),
  f3DPowerLorentzianFitCenterY(0),
  f3DPowerLorentzianFitGammaX(0),
  f3DPowerLorentzianFitGammaY(0),
  f3DPowerLorentzianFitBeta(0),
  f3DPowerLorentzianFitAmplitude(0),
  f3DPowerLorentzianFitVerticalOffset(0),
  f3DPowerLorentzianFitCenterXErr(0),
  f3DPowerLorentzianFitCenterYErr(0),
  f3DPowerLorentzianFitGammaXErr(0),
  f3DPowerLorentzianFitGammaYErr(0),
  f3DPowerLorentzianFitBetaErr(0),
  f3DPowerLorentzianFitAmplitudeErr(0),
  f3DPowerLorentzianFitVerticalOffsetErr(0),
  f3DPowerLorentzianFitChi2red(0),
  f3DPowerLorentzianFitPp(0),
  f3DPowerLorentzianFitDOF(0),
  f3DPowerLorentzianFitChargeUncertainty(0),
  f3DPowerLorentzianFitSuccessful(false),
  // Initialize 3D Gaussian fitting delta variables
  f3DGaussianDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  f3DGaussianDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 3D Gaussian fit variables
  f3DGaussianFitCenterX(0),
  f3DGaussianFitCenterY(0),
  f3DGaussianFitSigmaX(0),
  f3DGaussianFitSigmaY(0),
  f3DGaussianFitAmplitude(0),
  f3DGaussianFitVerticalOffset(0),
  f3DGaussianFitCenterXErr(0),
  f3DGaussianFitCenterYErr(0),
  f3DGaussianFitSigmaXErr(0),
  f3DGaussianFitSigmaYErr(0),
  f3DGaussianFitAmplitudeErr(0),
  f3DGaussianFitVerticalOffsetErr(0),
  f3DGaussianFitChi2red(0),
  f3DGaussianFitPp(0),
  f3DGaussianFitDOF(0),
  f3DGaussianFitChargeUncertainty(0),
  f3DGaussianFitSuccessful(false),
  // Legacy variables
  fIsPixelHit(false),
  fInitialEnergy(0),
  fGridPixelSize(0),
  fGridPixelSpacing(0),
  fGridPixelCornerOffset(0),
  fGridDetSize(0),
  fGridNumBlocksPerSide(0),
  
  // Charge uncertainties for Power-Law Lorentzian fits (5% of max charge)
  fPowerLorentzFitRowChargeUncertainty(0),
  fPowerLorentzFitColumnChargeUncertainty(0),
  
  // Set automatic radius selection variables
  fSelectedRadius(4)
{ 
  // Initialize neighborhood (9x9) grid vectors (they are automatically initialized empty)
  // Initialize step energy deposition vectors (they are automatically initialized empty)
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
        fTree->Branch("InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
        fTree->Branch("IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("True if hit is on pixel OR distance <= D0");
        fTree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D")->SetTitle("Delta X from Pixel Center to True Position [mm] (x_pixel - x_true)");
        fTree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D")->SetTitle("Delta Y from Pixel Center to True Position [mm] (y_pixel - y_true)");
        
        // GRIDNEIGHBORHOOD BRANCHES
        fTree->Branch("GridNeighborhoodAngles", &fNonPixel_GridNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg]");
        fTree->Branch("GridNeighborhoodChargeFractions", &fNonPixel_GridNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels");
        fTree->Branch("GridNeighborhoodDistances", &fNonPixel_GridNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm]");
        fTree->Branch("GridNeighborhoodCharges", &fNonPixel_GridNeighborhoodCharge)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels");
        
        // AUTOMATIC RADIUS SELECTION BRANCHES
        fTree->Branch("SelectedRadius", &fSelectedRadius, "SelectedRadius/I")->SetTitle("Automatically Selected Neighborhood Radius");
        
        // =============================================
        // DELTA VARIABLES (RESIDUALS) BRANCHES
        // =============================================
        // These are the key branches that CalcRes.py looks for
        
        // Gaussian 2D fit deltas
        if (Constants::ENABLE_GAUSSIAN_FITTING && Constants::ENABLE_2D_FITTING) {
            fTree->Branch("GaussRowDeltaX", &fGaussRowDeltaX, "GaussRowDeltaX/D")->SetTitle("Gaussian Row Fit Delta X [mm] (fit - true)");
            fTree->Branch("GaussColumnDeltaY", &fGaussColumnDeltaY, "GaussColumnDeltaY/D")->SetTitle("Gaussian Column Fit Delta Y [mm] (fit - true)");
        }
        
        // Lorentzian 2D fit deltas
        if (Constants::ENABLE_LORENTZIAN_FITTING && Constants::ENABLE_2D_FITTING) {
            fTree->Branch("LorentzRowDeltaX", &fLorentzRowDeltaX, "LorentzRowDeltaX/D")->SetTitle("Lorentzian Row Fit Delta X [mm] (fit - true)");
            fTree->Branch("LorentzColumnDeltaY", &fLorentzColumnDeltaY, "LorentzColumnDeltaY/D")->SetTitle("Lorentzian Column Fit Delta Y [mm] (fit - true)");
        }
        
        // Power Lorentzian 2D fit deltas
        if (Constants::ENABLE_POWER_LORENTZIAN_FITTING && Constants::ENABLE_2D_FITTING) {
            fTree->Branch("PowerLorentzRowDeltaX", &fPowerLorentzRowDeltaX, "PowerLorentzRowDeltaX/D")->SetTitle("Power Lorentzian Row Fit Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzColumnDeltaY", &fPowerLorentzColumnDeltaY, "PowerLorentzColumnDeltaY/D")->SetTitle("Power Lorentzian Column Fit Delta Y [mm] (fit - true)");
        }
        
        // Diagonal fit deltas (transformed coordinates)
        if (Constants::ENABLE_GAUSSIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("GaussMainDiagTransformedDeltaX", &fGaussMainDiagTransformedDeltaX, "GaussMainDiagTransformedDeltaX/D")->SetTitle("Gaussian Main Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("GaussMainDiagTransformedDeltaY", &fGaussMainDiagTransformedDeltaY, "GaussMainDiagTransformedDeltaY/D")->SetTitle("Gaussian Main Diagonal Transformed Delta Y [mm] (fit - true)");
            fTree->Branch("GaussSecondDiagTransformedDeltaX", &fGaussSecondDiagTransformedDeltaX, "GaussSecondDiagTransformedDeltaX/D")->SetTitle("Gaussian Secondary Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("GaussSecondDiagTransformedDeltaY", &fGaussSecondDiagTransformedDeltaY, "GaussSecondDiagTransformedDeltaY/D")->SetTitle("Gaussian Secondary Diagonal Transformed Delta Y [mm] (fit - true)");
        }
        
        if (Constants::ENABLE_LORENTZIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("LorentzMainDiagTransformedDeltaX", &fLorentzMainDiagTransformedDeltaX, "LorentzMainDiagTransformedDeltaX/D")->SetTitle("Lorentzian Main Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("LorentzMainDiagTransformedDeltaY", &fLorentzMainDiagTransformedDeltaY, "LorentzMainDiagTransformedDeltaY/D")->SetTitle("Lorentzian Main Diagonal Transformed Delta Y [mm] (fit - true)");
            fTree->Branch("LorentzSecondDiagTransformedDeltaX", &fLorentzSecondDiagTransformedDeltaX, "LorentzSecondDiagTransformedDeltaX/D")->SetTitle("Lorentzian Secondary Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("LorentzSecondDiagTransformedDeltaY", &fLorentzSecondDiagTransformedDeltaY, "LorentzSecondDiagTransformedDeltaY/D")->SetTitle("Lorentzian Secondary Diagonal Transformed Delta Y [mm] (fit - true)");
        }
        
        if (Constants::ENABLE_POWER_LORENTZIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("PowerLorentzMainDiagTransformedDeltaX", &fPowerLorentzMainDiagTransformedDeltaX, "PowerLorentzMainDiagTransformedDeltaX/D")->SetTitle("Power Lorentzian Main Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzMainDiagTransformedDeltaY", &fPowerLorentzMainDiagTransformedDeltaY, "PowerLorentzMainDiagTransformedDeltaY/D")->SetTitle("Power Lorentzian Main Diagonal Transformed Delta Y [mm] (fit - true)");
            fTree->Branch("PowerLorentzSecondDiagTransformedDeltaX", &fPowerLorentzSecondDiagTransformedDeltaX, "PowerLorentzSecondDiagTransformedDeltaX/D")->SetTitle("Power Lorentzian Secondary Diagonal Transformed Delta X [mm] (fit - true)");
            fTree->Branch("PowerLorentzSecondDiagTransformedDeltaY", &fPowerLorentzSecondDiagTransformedDeltaY, "PowerLorentzSecondDiagTransformedDeltaY/D")->SetTitle("Power Lorentzian Secondary Diagonal Transformed Delta Y [mm] (fit - true)");
        }
        
        // 3D fit deltas
        if (Constants::ENABLE_3D_GAUSSIAN_FITTING) {
            fTree->Branch("3DGaussianDeltaX", &f3DGaussianDeltaX, "3DGaussianDeltaX/D")->SetTitle("3D Gaussian Fit Delta X [mm] (fit - true)");
            fTree->Branch("3DGaussianDeltaY", &f3DGaussianDeltaY, "3DGaussianDeltaY/D")->SetTitle("3D Gaussian Fit Delta Y [mm] (fit - true)");
        }
        
        if (Constants::ENABLE_3D_LORENTZIAN_FITTING) {
            fTree->Branch("3DLorentzianDeltaX", &f3DLorentzianDeltaX, "3DLorentzianDeltaX/D")->SetTitle("3D Lorentzian Fit Delta X [mm] (fit - true)");
            fTree->Branch("3DLorentzianDeltaY", &f3DLorentzianDeltaY, "3DLorentzianDeltaY/D")->SetTitle("3D Lorentzian Fit Delta Y [mm] (fit - true)");
        }
        
        if (Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) {
            fTree->Branch("3DPowerLorentzianDeltaX", &f3DPowerLorentzianDeltaX, "3DPowerLorentzianDeltaX/D")->SetTitle("3D Power Lorentzian Fit Delta X [mm] (fit - true)");
            fTree->Branch("3DPowerLorentzianDeltaY", &f3DPowerLorentzianDeltaY, "3DPowerLorentzianDeltaY/D")->SetTitle("3D Power Lorentzian Fit Delta Y [mm] (fit - true)");
        }
        
        // Mean estimators (key resolution metrics)
        if (Constants::ENABLE_GAUSSIAN_FITTING) {
            fTree->Branch("GaussMeanTrueDeltaX", &fGaussMeanTrueDeltaX, "GaussMeanTrueDeltaX/D")->SetTitle("Gaussian Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("GaussMeanTrueDeltaY", &fGaussMeanTrueDeltaY, "GaussMeanTrueDeltaY/D")->SetTitle("Gaussian Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        if (Constants::ENABLE_LORENTZIAN_FITTING) {
            fTree->Branch("LorentzMeanTrueDeltaX", &fLorentzMeanTrueDeltaX, "LorentzMeanTrueDeltaX/D")->SetTitle("Lorentzian Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("LorentzMeanTrueDeltaY", &fLorentzMeanTrueDeltaY, "LorentzMeanTrueDeltaY/D")->SetTitle("Lorentzian Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        if (Constants::ENABLE_POWER_LORENTZIAN_FITTING) {
            fTree->Branch("PowerLorentzMeanTrueDeltaX", &fPowerLorentzMeanTrueDeltaX, "PowerLorentzMeanTrueDeltaX/D")->SetTitle("Power Lorentzian Mean Estimator Delta X [mm] (mean_fit - true)");
            fTree->Branch("PowerLorentzMeanTrueDeltaY", &fPowerLorentzMeanTrueDeltaY, "PowerLorentzMeanTrueDeltaY/D")->SetTitle("Power Lorentzian Mean Estimator Delta Y [mm] (mean_fit - true)");
        }
        
        // =============================================
        // GAUSSIAN FIT PARAMETERS BRANCHES
        // =============================================
        if (Constants::ENABLE_GAUSSIAN_FITTING) {
        // Row fit parameters
        fTree->Branch("GaussFitRowAmplitude", &fGaussFitRowAmplitude, "GaussFitRowAmplitude/D")->SetTitle("Gaussian Row Fit Amplitude");
        fTree->Branch("GaussFitRowAmplitudeErr", &fGaussFitRowAmplitudeErr, "GaussFitRowAmplitudeErr/D")->SetTitle("Gaussian Row Fit Amplitude Error");
        fTree->Branch("GaussFitRowStdev", &fGaussFitRowStdev, "GaussFitRowStdev/D")->SetTitle("Gaussian Row Fit Standard Deviation");
        fTree->Branch("GaussFitRowStdevErr", &fGaussFitRowStdevErr, "GaussFitRowStdevErr/D")->SetTitle("Gaussian Row Fit Standard Deviation Error");
        fTree->Branch("GaussFitRowVerticalOffset", &fGaussFitRowVerticalOffset, "GaussFitRowVerticalOffset/D")->SetTitle("Gaussian Row Fit Vertical Offset");
        fTree->Branch("GaussFitRowVerticalOffsetErr", &fGaussFitRowVerticalOffsetErr, "GaussFitRowVerticalOffsetErr/D")->SetTitle("Gaussian Row Fit Vertical Offset Error");
        fTree->Branch("GaussFitRowCenter", &fGaussFitRowCenter, "GaussFitRowCenter/D")->SetTitle("Gaussian Row Fit Center");
        fTree->Branch("GaussFitRowCenterErr", &fGaussFitRowCenterErr, "GaussFitRowCenterErr/D")->SetTitle("Gaussian Row Fit Center Error");
        fTree->Branch("GaussFitRowChi2red", &fGaussFitRowChi2red, "GaussFitRowChi2red/D")->SetTitle("Gaussian Row Fit Reduced Chi-squared");
        fTree->Branch("GaussFitRowPp", &fGaussFitRowPp, "GaussFitRowPp/D")->SetTitle("Gaussian Row Fit P-value");
        fTree->Branch("GaussFitRowDOF", &fGaussFitRowDOF, "GaussFitRowDOF/I")->SetTitle("Gaussian Row Fit Degrees of Freedom");
        fTree->Branch("GaussFitRowChargeUncertainty", &fGaussFitRowChargeUncertainty, "GaussFitRowChargeUncertainty/D")->SetTitle("Gaussian Row Fit Charge Uncertainty");
        
        // Column fit parameters
        fTree->Branch("GaussFitColumnAmplitude", &fGaussFitColumnAmplitude, "GaussFitColumnAmplitude/D")->SetTitle("Gaussian Column Fit Amplitude");
        fTree->Branch("GaussFitColumnAmplitudeErr", &fGaussFitColumnAmplitudeErr, "GaussFitColumnAmplitudeErr/D")->SetTitle("Gaussian Column Fit Amplitude Error");
        fTree->Branch("GaussFitColumnStdev", &fGaussFitColumnStdev, "GaussFitColumnStdev/D")->SetTitle("Gaussian Column Fit Standard Deviation");
        fTree->Branch("GaussFitColumnStdevErr", &fGaussFitColumnStdevErr, "GaussFitColumnStdevErr/D")->SetTitle("Gaussian Column Fit Standard Deviation Error");
        fTree->Branch("GaussFitColumnVerticalOffset", &fGaussFitColumnVerticalOffset, "GaussFitColumnVerticalOffset/D")->SetTitle("Gaussian Column Fit Vertical Offset");
        fTree->Branch("GaussFitColumnVerticalOffsetErr", &fGaussFitColumnVerticalOffsetErr, "GaussFitColumnVerticalOffsetErr/D")->SetTitle("Gaussian Column Fit Vertical Offset Error");
        fTree->Branch("GaussFitColumnCenter", &fGaussFitColumnCenter, "GaussFitColumnCenter/D")->SetTitle("Gaussian Column Fit Center");
        fTree->Branch("GaussFitColumnCenterErr", &fGaussFitColumnCenterErr, "GaussFitColumnCenterErr/D")->SetTitle("Gaussian Column Fit Center Error");
        fTree->Branch("GaussFitColumnChi2red", &fGaussFitColumnChi2red, "GaussFitColumnChi2red/D")->SetTitle("Gaussian Column Fit Reduced Chi-squared");
        fTree->Branch("GaussFitColumnPp", &fGaussFitColumnPp, "GaussFitColumnPp/D")->SetTitle("Gaussian Column Fit P-value");
        fTree->Branch("GaussFitColumnDOF", &fGaussFitColumnDOF, "GaussFitColumnDOF/I")->SetTitle("Gaussian Column Fit Degrees of Freedom");
        fTree->Branch("GaussFitColumnChargeUncertainty", &fGaussFitColumnChargeUncertainty, "GaussFitColumnChargeUncertainty/D")->SetTitle("Gaussian Column Fit Charge Uncertainty");
        
        // Main diagonal fit parameters
        fTree->Branch("GaussFitMainDiagXAmplitude", &fGaussFitMainDiagXAmplitude, "GaussFitMainDiagXAmplitude/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude");
        fTree->Branch("GaussFitMainDiagXAmplitudeErr", &fGaussFitMainDiagXAmplitudeErr, "GaussFitMainDiagXAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude Error");
        fTree->Branch("GaussFitMainDiagXStdev", &fGaussFitMainDiagXStdev, "GaussFitMainDiagXStdev/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation");
        fTree->Branch("GaussFitMainDiagXStdevErr", &fGaussFitMainDiagXStdevErr, "GaussFitMainDiagXStdevErr/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation Error");
        fTree->Branch("GaussFitMainDiagXVerticalOffset", &fGaussFitMainDiagXVerticalOffset, "GaussFitMainDiagXVerticalOffset/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset");
        fTree->Branch("GaussFitMainDiagXVerticalOffsetErr", &fGaussFitMainDiagXVerticalOffsetErr, "GaussFitMainDiagXVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset Error");
        fTree->Branch("GaussFitMainDiagXCenter", &fGaussFitMainDiagXCenter, "GaussFitMainDiagXCenter/D")->SetTitle("Gaussian Main Diagonal X Fit Center");
        fTree->Branch("GaussFitMainDiagXCenterErr", &fGaussFitMainDiagXCenterErr, "GaussFitMainDiagXCenterErr/D")->SetTitle("Gaussian Main Diagonal X Fit Center Error");
        fTree->Branch("GaussFitMainDiagXChi2red", &fGaussFitMainDiagXChi2red, "GaussFitMainDiagXChi2red/D")->SetTitle("Gaussian Main Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("GaussFitMainDiagXPp", &fGaussFitMainDiagXPp, "GaussFitMainDiagXPp/D")->SetTitle("Gaussian Main Diagonal X Fit P-value");
        fTree->Branch("GaussFitMainDiagXDOF", &fGaussFitMainDiagXDOF, "GaussFitMainDiagXDOF/I")->SetTitle("Gaussian Main Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("GaussFitMainDiagYAmplitude", &fGaussFitMainDiagYAmplitude, "GaussFitMainDiagYAmplitude/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude");
        fTree->Branch("GaussFitMainDiagYAmplitudeErr", &fGaussFitMainDiagYAmplitudeErr, "GaussFitMainDiagYAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude Error");
        fTree->Branch("GaussFitMainDiagYStdev", &fGaussFitMainDiagYStdev, "GaussFitMainDiagYStdev/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation");
        fTree->Branch("GaussFitMainDiagYStdevErr", &fGaussFitMainDiagYStdevErr, "GaussFitMainDiagYStdevErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation Error");
        fTree->Branch("GaussFitMainDiagYVerticalOffset", &fGaussFitMainDiagYVerticalOffset, "GaussFitMainDiagYVerticalOffset/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset");
        fTree->Branch("GaussFitMainDiagYVerticalOffsetErr", &fGaussFitMainDiagYVerticalOffsetErr, "GaussFitMainDiagYVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("GaussFitMainDiagYCenter", &fGaussFitMainDiagYCenter, "GaussFitMainDiagYCenter/D")->SetTitle("Gaussian Main Diagonal Y Fit Center");
        fTree->Branch("GaussFitMainDiagYCenterErr", &fGaussFitMainDiagYCenterErr, "GaussFitMainDiagYCenterErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Center Error");
        fTree->Branch("GaussFitMainDiagYChi2red", &fGaussFitMainDiagYChi2red, "GaussFitMainDiagYChi2red/D")->SetTitle("Gaussian Main Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("GaussFitMainDiagYPp", &fGaussFitMainDiagYPp, "GaussFitMainDiagYPp/D")->SetTitle("Gaussian Main Diagonal Y Fit P-value");
        fTree->Branch("GaussFitMainDiagYDOF", &fGaussFitMainDiagYDOF, "GaussFitMainDiagYDOF/I")->SetTitle("Gaussian Main Diagonal Y Fit Degrees of Freedom");
        
        // Secondary diagonal fit parameters
        fTree->Branch("GaussFitSecondDiagXAmplitude", &fGaussFitSecondDiagXAmplitude, "GaussFitSecondDiagXAmplitude/D")->SetTitle("Gaussian Secondary Diagonal X Fit Amplitude");
        fTree->Branch("GaussFitSecondDiagXAmplitudeErr", &fGaussFitSecondDiagXAmplitudeErr, "GaussFitSecondDiagXAmplitudeErr/D")->SetTitle("Gaussian Secondary Diagonal X Fit Amplitude Error");
        fTree->Branch("GaussFitSecondDiagXStdev", &fGaussFitSecondDiagXStdev, "GaussFitSecondDiagXStdev/D")->SetTitle("Gaussian Secondary Diagonal X Fit Standard Deviation");
        fTree->Branch("GaussFitSecondDiagXStdevErr", &fGaussFitSecondDiagXStdevErr, "GaussFitSecondDiagXStdevErr/D")->SetTitle("Gaussian Secondary Diagonal X Fit Standard Deviation Error");
        fTree->Branch("GaussFitSecondDiagXVerticalOffset", &fGaussFitSecondDiagXVerticalOffset, "GaussFitSecondDiagXVerticalOffset/D")->SetTitle("Gaussian Secondary Diagonal X Fit Vertical Offset");
        fTree->Branch("GaussFitSecondDiagXVerticalOffsetErr", &fGaussFitSecondDiagXVerticalOffsetErr, "GaussFitSecondDiagXVerticalOffsetErr/D")->SetTitle("Gaussian Secondary Diagonal X Fit Vertical Offset Error");
        fTree->Branch("GaussFitSecondDiagXCenter", &fGaussFitSecondDiagXCenter, "GaussFitSecondDiagXCenter/D")->SetTitle("Gaussian Secondary Diagonal X Fit Center");
        fTree->Branch("GaussFitSecondDiagXCenterErr", &fGaussFitSecondDiagXCenterErr, "GaussFitSecondDiagXCenterErr/D")->SetTitle("Gaussian Secondary Diagonal X Fit Center Error");
        fTree->Branch("GaussFitSecondDiagXChi2red", &fGaussFitSecondDiagXChi2red, "GaussFitSecondDiagXChi2red/D")->SetTitle("Gaussian Secondary Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("GaussFitSecondDiagXPp", &fGaussFitSecondDiagXPp, "GaussFitSecondDiagXPp/D")->SetTitle("Gaussian Secondary Diagonal X Fit P-value");
        fTree->Branch("GaussFitSecondDiagXDOF", &fGaussFitSecondDiagXDOF, "GaussFitSecondDiagXDOF/I")->SetTitle("Gaussian Secondary Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("GaussFitSecondDiagYAmplitude", &fGaussFitSecondDiagYAmplitude, "GaussFitSecondDiagYAmplitude/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Amplitude");
        fTree->Branch("GaussFitSecondDiagYAmplitudeErr", &fGaussFitSecondDiagYAmplitudeErr, "GaussFitSecondDiagYAmplitudeErr/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Amplitude Error");
        fTree->Branch("GaussFitSecondDiagYStdev", &fGaussFitSecondDiagYStdev, "GaussFitSecondDiagYStdev/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Standard Deviation");
        fTree->Branch("GaussFitSecondDiagYStdevErr", &fGaussFitSecondDiagYStdevErr, "GaussFitSecondDiagYStdevErr/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Standard Deviation Error");
        fTree->Branch("GaussFitSecondDiagYVerticalOffset", &fGaussFitSecondDiagYVerticalOffset, "GaussFitSecondDiagYVerticalOffset/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Vertical Offset");
        fTree->Branch("GaussFitSecondDiagYVerticalOffsetErr", &fGaussFitSecondDiagYVerticalOffsetErr, "GaussFitSecondDiagYVerticalOffsetErr/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("GaussFitSecondDiagYCenter", &fGaussFitSecondDiagYCenter, "GaussFitSecondDiagYCenter/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Center");
        fTree->Branch("GaussFitSecondDiagYCenterErr", &fGaussFitSecondDiagYCenterErr, "GaussFitSecondDiagYCenterErr/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Center Error");
        fTree->Branch("GaussFitSecondDiagYChi2red", &fGaussFitSecondDiagYChi2red, "GaussFitSecondDiagYChi2red/D")->SetTitle("Gaussian Secondary Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("GaussFitSecondDiagYPp", &fGaussFitSecondDiagYPp, "GaussFitSecondDiagYPp/D")->SetTitle("Gaussian Secondary Diagonal Y Fit P-value");
        fTree->Branch("GaussFitSecondDiagYDOF", &fGaussFitSecondDiagYDOF, "GaussFitSecondDiagYDOF/I")->SetTitle("Gaussian Secondary Diagonal Y Fit Degrees of Freedom");
        }
        
        // =============================================
        // LORENTZIAN FIT PARAMETERS BRANCHES
        // =============================================
        if (Constants::ENABLE_LORENTZIAN_FITTING) {
        // Row fit parameters
        fTree->Branch("LorentzFitRowAmplitude", &fLorentzFitRowAmplitude, "LorentzFitRowAmplitude/D")->SetTitle("Lorentzian Row Fit Amplitude");
        fTree->Branch("LorentzFitRowAmplitudeErr", &fLorentzFitRowAmplitudeErr, "LorentzFitRowAmplitudeErr/D")->SetTitle("Lorentzian Row Fit Amplitude Error");
        fTree->Branch("LorentzFitRowGamma", &fLorentzFitRowGamma, "LorentzFitRowGamma/D")->SetTitle("Lorentzian Row Fit Gamma (HWHM)");
        fTree->Branch("LorentzFitRowGammaErr", &fLorentzFitRowGammaErr, "LorentzFitRowGammaErr/D")->SetTitle("Lorentzian Row Fit Gamma Error");
        fTree->Branch("LorentzFitRowVerticalOffset", &fLorentzFitRowVerticalOffset, "LorentzFitRowVerticalOffset/D")->SetTitle("Lorentzian Row Fit Vertical Offset");
        fTree->Branch("LorentzFitRowVerticalOffsetErr", &fLorentzFitRowVerticalOffsetErr, "LorentzFitRowVerticalOffsetErr/D")->SetTitle("Lorentzian Row Fit Vertical Offset Error");
        fTree->Branch("LorentzFitRowCenter", &fLorentzFitRowCenter, "LorentzFitRowCenter/D")->SetTitle("Lorentzian Row Fit Center");
        fTree->Branch("LorentzFitRowCenterErr", &fLorentzFitRowCenterErr, "LorentzFitRowCenterErr/D")->SetTitle("Lorentzian Row Fit Center Error");
        fTree->Branch("LorentzFitRowChi2red", &fLorentzFitRowChi2red, "LorentzFitRowChi2red/D")->SetTitle("Lorentzian Row Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitRowPp", &fLorentzFitRowPp, "LorentzFitRowPp/D")->SetTitle("Lorentzian Row Fit P-value");
        fTree->Branch("LorentzFitRowDOF", &fLorentzFitRowDOF, "LorentzFitRowDOF/I")->SetTitle("Lorentzian Row Fit Degrees of Freedom");
        fTree->Branch("LorentzFitRowChargeUncertainty", &fLorentzFitRowChargeUncertainty, "LorentzFitRowChargeUncertainty/D")->SetTitle("Lorentzian Row Fit Charge Uncertainty");
        
        // Column fit parameters
        fTree->Branch("LorentzFitColumnAmplitude", &fLorentzFitColumnAmplitude, "LorentzFitColumnAmplitude/D")->SetTitle("Lorentzian Column Fit Amplitude");
        fTree->Branch("LorentzFitColumnAmplitudeErr", &fLorentzFitColumnAmplitudeErr, "LorentzFitColumnAmplitudeErr/D")->SetTitle("Lorentzian Column Fit Amplitude Error");
        fTree->Branch("LorentzFitColumnGamma", &fLorentzFitColumnGamma, "LorentzFitColumnGamma/D")->SetTitle("Lorentzian Column Fit Gamma (HWHM)");
        fTree->Branch("LorentzFitColumnGammaErr", &fLorentzFitColumnGammaErr, "LorentzFitColumnGammaErr/D")->SetTitle("Lorentzian Column Fit Gamma Error");
        fTree->Branch("LorentzFitColumnVerticalOffset", &fLorentzFitColumnVerticalOffset, "LorentzFitColumnVerticalOffset/D")->SetTitle("Lorentzian Column Fit Vertical Offset");
        fTree->Branch("LorentzFitColumnVerticalOffsetErr", &fLorentzFitColumnVerticalOffsetErr, "LorentzFitColumnVerticalOffsetErr/D")->SetTitle("Lorentzian Column Fit Vertical Offset Error");
        fTree->Branch("LorentzFitColumnCenter", &fLorentzFitColumnCenter, "LorentzFitColumnCenter/D")->SetTitle("Lorentzian Column Fit Center");
        fTree->Branch("LorentzFitColumnCenterErr", &fLorentzFitColumnCenterErr, "LorentzFitColumnCenterErr/D")->SetTitle("Lorentzian Column Fit Center Error");
        fTree->Branch("LorentzFitColumnChi2red", &fLorentzFitColumnChi2red, "LorentzFitColumnChi2red/D")->SetTitle("Lorentzian Column Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitColumnPp", &fLorentzFitColumnPp, "LorentzFitColumnPp/D")->SetTitle("Lorentzian Column Fit P-value");
        fTree->Branch("LorentzFitColumnDOF", &fLorentzFitColumnDOF, "LorentzFitColumnDOF/I")->SetTitle("Lorentzian Column Fit Degrees of Freedom");
        fTree->Branch("LorentzFitColumnChargeUncertainty", &fLorentzFitColumnChargeUncertainty, "LorentzFitColumnChargeUncertainty/D")->SetTitle("Lorentzian Column Fit Charge Uncertainty");
        
        // Main diagonal fit parameters
        fTree->Branch("LorentzFitMainDiagXAmplitude", &fLorentzFitMainDiagXAmplitude, "LorentzFitMainDiagXAmplitude/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude");
        fTree->Branch("LorentzFitMainDiagXAmplitudeErr", &fLorentzFitMainDiagXAmplitudeErr, "LorentzFitMainDiagXAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude Error");
        fTree->Branch("LorentzFitMainDiagXGamma", &fLorentzFitMainDiagXGamma, "LorentzFitMainDiagXGamma/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma");
        fTree->Branch("LorentzFitMainDiagXGammaErr", &fLorentzFitMainDiagXGammaErr, "LorentzFitMainDiagXGammaErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma Error");
        fTree->Branch("LorentzFitMainDiagXVerticalOffset", &fLorentzFitMainDiagXVerticalOffset, "LorentzFitMainDiagXVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset");
        fTree->Branch("LorentzFitMainDiagXVerticalOffsetErr", &fLorentzFitMainDiagXVerticalOffsetErr, "LorentzFitMainDiagXVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset Error");
        fTree->Branch("LorentzFitMainDiagXCenter", &fLorentzFitMainDiagXCenter, "LorentzFitMainDiagXCenter/D")->SetTitle("Lorentzian Main Diagonal X Fit Center");
        fTree->Branch("LorentzFitMainDiagXCenterErr", &fLorentzFitMainDiagXCenterErr, "LorentzFitMainDiagXCenterErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Center Error");
        fTree->Branch("LorentzFitMainDiagXChi2red", &fLorentzFitMainDiagXChi2red, "LorentzFitMainDiagXChi2red/D")->SetTitle("Lorentzian Main Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitMainDiagXPp", &fLorentzFitMainDiagXPp, "LorentzFitMainDiagXPp/D")->SetTitle("Lorentzian Main Diagonal X Fit P-value");
        fTree->Branch("LorentzFitMainDiagXDOF", &fLorentzFitMainDiagXDOF, "LorentzFitMainDiagXDOF/I")->SetTitle("Lorentzian Main Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("LorentzFitMainDiagYAmplitude", &fLorentzFitMainDiagYAmplitude, "LorentzFitMainDiagYAmplitude/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude");
        fTree->Branch("LorentzFitMainDiagYAmplitudeErr", &fLorentzFitMainDiagYAmplitudeErr, "LorentzFitMainDiagYAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude Error");
        fTree->Branch("LorentzFitMainDiagYGamma", &fLorentzFitMainDiagYGamma, "LorentzFitMainDiagYGamma/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma");
        fTree->Branch("LorentzFitMainDiagYGammaErr", &fLorentzFitMainDiagYGammaErr, "LorentzFitMainDiagYGammaErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma Error");
        fTree->Branch("LorentzFitMainDiagYVerticalOffset", &fLorentzFitMainDiagYVerticalOffset, "LorentzFitMainDiagYVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset");
        fTree->Branch("LorentzFitMainDiagYVerticalOffsetErr", &fLorentzFitMainDiagYVerticalOffsetErr, "LorentzFitMainDiagYVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("LorentzFitMainDiagYCenter", &fLorentzFitMainDiagYCenter, "LorentzFitMainDiagYCenter/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center");
        fTree->Branch("LorentzFitMainDiagYCenterErr", &fLorentzFitMainDiagYCenterErr, "LorentzFitMainDiagYCenterErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center Error");
        fTree->Branch("LorentzFitMainDiagYChi2red", &fLorentzFitMainDiagYChi2red, "LorentzFitMainDiagYChi2red/D")->SetTitle("Lorentzian Main Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitMainDiagYPp", &fLorentzFitMainDiagYPp, "LorentzFitMainDiagYPp/D")->SetTitle("Lorentzian Main Diagonal Y Fit P-value");
        fTree->Branch("LorentzFitMainDiagYDOF", &fLorentzFitMainDiagYDOF, "LorentzFitMainDiagYDOF/I")->SetTitle("Lorentzian Main Diagonal Y Fit Degrees of Freedom");
        
        // Secondary diagonal fit parameters
        fTree->Branch("LorentzFitSecondDiagXAmplitude", &fLorentzFitSecondDiagXAmplitude, "LorentzFitSecondDiagXAmplitude/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Amplitude");
        fTree->Branch("LorentzFitSecondDiagXAmplitudeErr", &fLorentzFitSecondDiagXAmplitudeErr, "LorentzFitSecondDiagXAmplitudeErr/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Amplitude Error");
        fTree->Branch("LorentzFitSecondDiagXGamma", &fLorentzFitSecondDiagXGamma, "LorentzFitSecondDiagXGamma/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Gamma");
        fTree->Branch("LorentzFitSecondDiagXGammaErr", &fLorentzFitSecondDiagXGammaErr, "LorentzFitSecondDiagXGammaErr/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Gamma Error");
        fTree->Branch("LorentzFitSecondDiagXVerticalOffset", &fLorentzFitSecondDiagXVerticalOffset, "LorentzFitSecondDiagXVerticalOffset/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Vertical Offset");
        fTree->Branch("LorentzFitSecondDiagXVerticalOffsetErr", &fLorentzFitSecondDiagXVerticalOffsetErr, "LorentzFitSecondDiagXVerticalOffsetErr/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Vertical Offset Error");
        fTree->Branch("LorentzFitSecondDiagXCenter", &fLorentzFitSecondDiagXCenter, "LorentzFitSecondDiagXCenter/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Center");
        fTree->Branch("LorentzFitSecondDiagXCenterErr", &fLorentzFitSecondDiagXCenterErr, "LorentzFitSecondDiagXCenterErr/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Center Error");
        fTree->Branch("LorentzFitSecondDiagXChi2red", &fLorentzFitSecondDiagXChi2red, "LorentzFitSecondDiagXChi2red/D")->SetTitle("Lorentzian Secondary Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitSecondDiagXPp", &fLorentzFitSecondDiagXPp, "LorentzFitSecondDiagXPp/D")->SetTitle("Lorentzian Secondary Diagonal X Fit P-value");
        fTree->Branch("LorentzFitSecondDiagXDOF", &fLorentzFitSecondDiagXDOF, "LorentzFitSecondDiagXDOF/I")->SetTitle("Lorentzian Secondary Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("LorentzFitSecondDiagYAmplitude", &fLorentzFitSecondDiagYAmplitude, "LorentzFitSecondDiagYAmplitude/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Amplitude");
        fTree->Branch("LorentzFitSecondDiagYAmplitudeErr", &fLorentzFitSecondDiagYAmplitudeErr, "LorentzFitSecondDiagYAmplitudeErr/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Amplitude Error");
        fTree->Branch("LorentzFitSecondDiagYGamma", &fLorentzFitSecondDiagYGamma, "LorentzFitSecondDiagYGamma/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Gamma");
        fTree->Branch("LorentzFitSecondDiagYGammaErr", &fLorentzFitSecondDiagYGammaErr, "LorentzFitSecondDiagYGammaErr/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Gamma Error");
        fTree->Branch("LorentzFitSecondDiagYVerticalOffset", &fLorentzFitSecondDiagYVerticalOffset, "LorentzFitSecondDiagYVerticalOffset/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Vertical Offset");
        fTree->Branch("LorentzFitSecondDiagYVerticalOffsetErr", &fLorentzFitSecondDiagYVerticalOffsetErr, "LorentzFitSecondDiagYVerticalOffsetErr/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("LorentzFitSecondDiagYCenter", &fLorentzFitSecondDiagYCenter, "LorentzFitSecondDiagYCenter/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Center");
        fTree->Branch("LorentzFitSecondDiagYCenterErr", &fLorentzFitSecondDiagYCenterErr, "LorentzFitSecondDiagYCenterErr/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Center Error");
        fTree->Branch("LorentzFitSecondDiagYChi2red", &fLorentzFitSecondDiagYChi2red, "LorentzFitSecondDiagYChi2red/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("LorentzFitSecondDiagYPp", &fLorentzFitSecondDiagYPp, "LorentzFitSecondDiagYPp/D")->SetTitle("Lorentzian Secondary Diagonal Y Fit P-value");
        fTree->Branch("LorentzFitSecondDiagYDOF", &fLorentzFitSecondDiagYDOF, "LorentzFitSecondDiagYDOF/I")->SetTitle("Lorentzian Secondary Diagonal Y Fit Degrees of Freedom");
        }
        
        // =============================================
        // POWER-LAW LORENTZIAN FIT PARAMETERS BRANCHES
        // =============================================
        if (Constants::ENABLE_POWER_LORENTZIAN_FITTING) {
        // Row fit parameters
        fTree->Branch("PowerLorentzFitRowAmplitude", &fPowerLorentzFitRowAmplitude, "PowerLorentzFitRowAmplitude/D")->SetTitle("Power-Law Lorentzian Row Fit Amplitude");
        fTree->Branch("PowerLorentzFitRowAmplitudeErr", &fPowerLorentzFitRowAmplitudeErr, "PowerLorentzFitRowAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Row Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitRowBeta", &fPowerLorentzFitRowBeta, "PowerLorentzFitRowBeta/D")->SetTitle("Power-Law Lorentzian Row Fit Beta (Power-Law Exponent)");
        fTree->Branch("PowerLorentzFitRowBetaErr", &fPowerLorentzFitRowBetaErr, "PowerLorentzFitRowBetaErr/D")->SetTitle("Power-Law Lorentzian Row Fit Beta Error");
        fTree->Branch("PowerLorentzFitRowGamma", &fPowerLorentzFitRowGamma, "PowerLorentzFitRowGamma/D")->SetTitle("Power-Law Lorentzian Row Fit Gamma (HWHM)");
        fTree->Branch("PowerLorentzFitRowGammaErr", &fPowerLorentzFitRowGammaErr, "PowerLorentzFitRowGammaErr/D")->SetTitle("Power-Law Lorentzian Row Fit Gamma Error");
        fTree->Branch("PowerLorentzFitRowVerticalOffset", &fPowerLorentzFitRowVerticalOffset, "PowerLorentzFitRowVerticalOffset/D")->SetTitle("Power-Law Lorentzian Row Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitRowVerticalOffsetErr", &fPowerLorentzFitRowVerticalOffsetErr, "PowerLorentzFitRowVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Row Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitRowCenter", &fPowerLorentzFitRowCenter, "PowerLorentzFitRowCenter/D")->SetTitle("Power-Law Lorentzian Row Fit Center");
        fTree->Branch("PowerLorentzFitRowCenterErr", &fPowerLorentzFitRowCenterErr, "PowerLorentzFitRowCenterErr/D")->SetTitle("Power-Law Lorentzian Row Fit Center Error");
        fTree->Branch("PowerLorentzFitRowChi2red", &fPowerLorentzFitRowChi2red, "PowerLorentzFitRowChi2red/D")->SetTitle("Power-Law Lorentzian Row Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitRowPp", &fPowerLorentzFitRowPp, "PowerLorentzFitRowPp/D")->SetTitle("Power-Law Lorentzian Row Fit P-value");
        fTree->Branch("PowerLorentzFitRowDOF", &fPowerLorentzFitRowDOF, "PowerLorentzFitRowDOF/I")->SetTitle("Power-Law Lorentzian Row Fit Degrees of Freedom");
        fTree->Branch("PowerLorentzFitRowChargeUncertainty", &fPowerLorentzFitRowChargeUncertainty, "PowerLorentzFitRowChargeUncertainty/D")->SetTitle("Power-Law Lorentzian Row Fit Charge Uncertainty");
        
        // Column fit parameters
        fTree->Branch("PowerLorentzFitColumnAmplitude", &fPowerLorentzFitColumnAmplitude, "PowerLorentzFitColumnAmplitude/D")->SetTitle("Power-Law Lorentzian Column Fit Amplitude");
        fTree->Branch("PowerLorentzFitColumnAmplitudeErr", &fPowerLorentzFitColumnAmplitudeErr, "PowerLorentzFitColumnAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Column Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitColumnBeta", &fPowerLorentzFitColumnBeta, "PowerLorentzFitColumnBeta/D")->SetTitle("Power-Law Lorentzian Column Fit Beta (Power-Law Exponent)");
        fTree->Branch("PowerLorentzFitColumnBetaErr", &fPowerLorentzFitColumnBetaErr, "PowerLorentzFitColumnBetaErr/D")->SetTitle("Power-Law Lorentzian Column Fit Beta Error");
        fTree->Branch("PowerLorentzFitColumnGamma", &fPowerLorentzFitColumnGamma, "PowerLorentzFitColumnGamma/D")->SetTitle("Power-Law Lorentzian Column Fit Gamma (HWHM)");
        fTree->Branch("PowerLorentzFitColumnGammaErr", &fPowerLorentzFitColumnGammaErr, "PowerLorentzFitColumnGammaErr/D")->SetTitle("Power-Law Lorentzian Column Fit Gamma Error");
        fTree->Branch("PowerLorentzFitColumnVerticalOffset", &fPowerLorentzFitColumnVerticalOffset, "PowerLorentzFitColumnVerticalOffset/D")->SetTitle("Power-Law Lorentzian Column Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitColumnVerticalOffsetErr", &fPowerLorentzFitColumnVerticalOffsetErr, "PowerLorentzFitColumnVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Column Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitColumnCenter", &fPowerLorentzFitColumnCenter, "PowerLorentzFitColumnCenter/D")->SetTitle("Power-Law Lorentzian Column Fit Center");
        fTree->Branch("PowerLorentzFitColumnCenterErr", &fPowerLorentzFitColumnCenterErr, "PowerLorentzFitColumnCenterErr/D")->SetTitle("Power-Law Lorentzian Column Fit Center Error");
        fTree->Branch("PowerLorentzFitColumnChi2red", &fPowerLorentzFitColumnChi2red, "PowerLorentzFitColumnChi2red/D")->SetTitle("Power-Law Lorentzian Column Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitColumnPp", &fPowerLorentzFitColumnPp, "PowerLorentzFitColumnPp/D")->SetTitle("Power-Law Lorentzian Column Fit P-value");
        fTree->Branch("PowerLorentzFitColumnDOF", &fPowerLorentzFitColumnDOF, "PowerLorentzFitColumnDOF/I")->SetTitle("Power-Law Lorentzian Column Fit Degrees of Freedom");
        fTree->Branch("PowerLorentzFitColumnChargeUncertainty", &fPowerLorentzFitColumnChargeUncertainty, "PowerLorentzFitColumnChargeUncertainty/D")->SetTitle("Power-Law Lorentzian Column Fit Charge Uncertainty");
        
        // Main diagonal fit parameters
        fTree->Branch("PowerLorentzFitMainDiagXAmplitude", &fPowerLorentzFitMainDiagXAmplitude, "PowerLorentzFitMainDiagXAmplitude/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Amplitude");
        fTree->Branch("PowerLorentzFitMainDiagXAmplitudeErr", &fPowerLorentzFitMainDiagXAmplitudeErr, "PowerLorentzFitMainDiagXAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitMainDiagXBeta", &fPowerLorentzFitMainDiagXBeta, "PowerLorentzFitMainDiagXBeta/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Beta");
        fTree->Branch("PowerLorentzFitMainDiagXBetaErr", &fPowerLorentzFitMainDiagXBetaErr, "PowerLorentzFitMainDiagXBetaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Beta Error");
        fTree->Branch("PowerLorentzFitMainDiagXGamma", &fPowerLorentzFitMainDiagXGamma, "PowerLorentzFitMainDiagXGamma/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Gamma");
        fTree->Branch("PowerLorentzFitMainDiagXGammaErr", &fPowerLorentzFitMainDiagXGammaErr, "PowerLorentzFitMainDiagXGammaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Gamma Error");
        fTree->Branch("PowerLorentzFitMainDiagXVerticalOffset", &fPowerLorentzFitMainDiagXVerticalOffset, "PowerLorentzFitMainDiagXVerticalOffset/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitMainDiagXVerticalOffsetErr", &fPowerLorentzFitMainDiagXVerticalOffsetErr, "PowerLorentzFitMainDiagXVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitMainDiagXCenter", &fPowerLorentzFitMainDiagXCenter, "PowerLorentzFitMainDiagXCenter/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Center");
        fTree->Branch("PowerLorentzFitMainDiagXCenterErr", &fPowerLorentzFitMainDiagXCenterErr, "PowerLorentzFitMainDiagXCenterErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Center Error");
        fTree->Branch("PowerLorentzFitMainDiagXChi2red", &fPowerLorentzFitMainDiagXChi2red, "PowerLorentzFitMainDiagXChi2red/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitMainDiagXPp", &fPowerLorentzFitMainDiagXPp, "PowerLorentzFitMainDiagXPp/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit P-value");
        fTree->Branch("PowerLorentzFitMainDiagXDOF", &fPowerLorentzFitMainDiagXDOF, "PowerLorentzFitMainDiagXDOF/I")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("PowerLorentzFitMainDiagYAmplitude", &fPowerLorentzFitMainDiagYAmplitude, "PowerLorentzFitMainDiagYAmplitude/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Amplitude");
        fTree->Branch("PowerLorentzFitMainDiagYAmplitudeErr", &fPowerLorentzFitMainDiagYAmplitudeErr, "PowerLorentzFitMainDiagYAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitMainDiagYBeta", &fPowerLorentzFitMainDiagYBeta, "PowerLorentzFitMainDiagYBeta/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Beta");
        fTree->Branch("PowerLorentzFitMainDiagYBetaErr", &fPowerLorentzFitMainDiagYBetaErr, "PowerLorentzFitMainDiagYBetaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Beta Error");
        fTree->Branch("PowerLorentzFitMainDiagYGamma", &fPowerLorentzFitMainDiagYGamma, "PowerLorentzFitMainDiagYGamma/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Gamma");
        fTree->Branch("PowerLorentzFitMainDiagYGammaErr", &fPowerLorentzFitMainDiagYGammaErr, "PowerLorentzFitMainDiagYGammaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Gamma Error");
        fTree->Branch("PowerLorentzFitMainDiagYVerticalOffset", &fPowerLorentzFitMainDiagYVerticalOffset, "PowerLorentzFitMainDiagYVerticalOffset/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitMainDiagYVerticalOffsetErr", &fPowerLorentzFitMainDiagYVerticalOffsetErr, "PowerLorentzFitMainDiagYVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitMainDiagYCenter", &fPowerLorentzFitMainDiagYCenter, "PowerLorentzFitMainDiagYCenter/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Center");
        fTree->Branch("PowerLorentzFitMainDiagYCenterErr", &fPowerLorentzFitMainDiagYCenterErr, "PowerLorentzFitMainDiagYCenterErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Center Error");
        fTree->Branch("PowerLorentzFitMainDiagYChi2red", &fPowerLorentzFitMainDiagYChi2red, "PowerLorentzFitMainDiagYChi2red/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitMainDiagYPp", &fPowerLorentzFitMainDiagYPp, "PowerLorentzFitMainDiagYPp/D")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit P-value");
        fTree->Branch("PowerLorentzFitMainDiagYDOF", &fPowerLorentzFitMainDiagYDOF, "PowerLorentzFitMainDiagYDOF/I")->SetTitle("Power-Law Lorentzian Main Diagonal Y Fit Degrees of Freedom");
        
        // Secondary diagonal fit parameters
        fTree->Branch("PowerLorentzFitSecondDiagXAmplitude", &fPowerLorentzFitSecondDiagXAmplitude, "PowerLorentzFitSecondDiagXAmplitude/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Amplitude");
        fTree->Branch("PowerLorentzFitSecondDiagXAmplitudeErr", &fPowerLorentzFitSecondDiagXAmplitudeErr, "PowerLorentzFitSecondDiagXAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitSecondDiagXBeta", &fPowerLorentzFitSecondDiagXBeta, "PowerLorentzFitSecondDiagXBeta/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Beta");
        fTree->Branch("PowerLorentzFitSecondDiagXBetaErr", &fPowerLorentzFitSecondDiagXBetaErr, "PowerLorentzFitSecondDiagXBetaErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Beta Error");
        fTree->Branch("PowerLorentzFitSecondDiagXGamma", &fPowerLorentzFitSecondDiagXGamma, "PowerLorentzFitSecondDiagXGamma/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Gamma");
        fTree->Branch("PowerLorentzFitSecondDiagXGammaErr", &fPowerLorentzFitSecondDiagXGammaErr, "PowerLorentzFitSecondDiagXGammaErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Gamma Error");
        fTree->Branch("PowerLorentzFitSecondDiagXVerticalOffset", &fPowerLorentzFitSecondDiagXVerticalOffset, "PowerLorentzFitSecondDiagXVerticalOffset/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitSecondDiagXVerticalOffsetErr", &fPowerLorentzFitSecondDiagXVerticalOffsetErr, "PowerLorentzFitSecondDiagXVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitSecondDiagXCenter", &fPowerLorentzFitSecondDiagXCenter, "PowerLorentzFitSecondDiagXCenter/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Center");
        fTree->Branch("PowerLorentzFitSecondDiagXCenterErr", &fPowerLorentzFitSecondDiagXCenterErr, "PowerLorentzFitSecondDiagXCenterErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Center Error");
        fTree->Branch("PowerLorentzFitSecondDiagXChi2red", &fPowerLorentzFitSecondDiagXChi2red, "PowerLorentzFitSecondDiagXChi2red/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitSecondDiagXPp", &fPowerLorentzFitSecondDiagXPp, "PowerLorentzFitSecondDiagXPp/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit P-value");
        fTree->Branch("PowerLorentzFitSecondDiagXDOF", &fPowerLorentzFitSecondDiagXDOF, "PowerLorentzFitSecondDiagXDOF/I")->SetTitle("Power-Law Lorentzian Secondary Diagonal X Fit Degrees of Freedom");
        
        fTree->Branch("PowerLorentzFitSecondDiagYAmplitude", &fPowerLorentzFitSecondDiagYAmplitude, "PowerLorentzFitSecondDiagYAmplitude/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Amplitude");
        fTree->Branch("PowerLorentzFitSecondDiagYAmplitudeErr", &fPowerLorentzFitSecondDiagYAmplitudeErr, "PowerLorentzFitSecondDiagYAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Amplitude Error");
        fTree->Branch("PowerLorentzFitSecondDiagYBeta", &fPowerLorentzFitSecondDiagYBeta, "PowerLorentzFitSecondDiagYBeta/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Beta");
        fTree->Branch("PowerLorentzFitSecondDiagYBetaErr", &fPowerLorentzFitSecondDiagYBetaErr, "PowerLorentzFitSecondDiagYBetaErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Beta Error");
        fTree->Branch("PowerLorentzFitSecondDiagYGamma", &fPowerLorentzFitSecondDiagYGamma, "PowerLorentzFitSecondDiagYGamma/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Gamma");
        fTree->Branch("PowerLorentzFitSecondDiagYGammaErr", &fPowerLorentzFitSecondDiagYGammaErr, "PowerLorentzFitSecondDiagYGammaErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Gamma Error");
        fTree->Branch("PowerLorentzFitSecondDiagYVerticalOffset", &fPowerLorentzFitSecondDiagYVerticalOffset, "PowerLorentzFitSecondDiagYVerticalOffset/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Vertical Offset");
        fTree->Branch("PowerLorentzFitSecondDiagYVerticalOffsetErr", &fPowerLorentzFitSecondDiagYVerticalOffsetErr, "PowerLorentzFitSecondDiagYVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Vertical Offset Error");
        fTree->Branch("PowerLorentzFitSecondDiagYCenter", &fPowerLorentzFitSecondDiagYCenter, "PowerLorentzFitSecondDiagYCenter/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Center");
        fTree->Branch("PowerLorentzFitSecondDiagYCenterErr", &fPowerLorentzFitSecondDiagYCenterErr, "PowerLorentzFitSecondDiagYCenterErr/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Center Error");
        fTree->Branch("PowerLorentzFitSecondDiagYChi2red", &fPowerLorentzFitSecondDiagYChi2red, "PowerLorentzFitSecondDiagYChi2red/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Reduced Chi-squared");
        fTree->Branch("PowerLorentzFitSecondDiagYPp", &fPowerLorentzFitSecondDiagYPp, "PowerLorentzFitSecondDiagYPp/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit P-value");
        fTree->Branch("PowerLorentzFitSecondDiagYDOF", &fPowerLorentzFitSecondDiagYDOF, "PowerLorentzFitSecondDiagYDOF/I")->SetTitle("Power-Law Lorentzian Secondary Diagonal Y Fit Degrees of Freedom");
        }
        
        // =============================================
        // 3D FIT PARAMETERS BRANCHES
        // =============================================
        // 3D Gaussian fit parameters
        if (Constants::ENABLE_3D_GAUSSIAN_FITTING) {
        fTree->Branch("3DGaussianFitCenterX", &f3DGaussianFitCenterX, "3DGaussianFitCenterX/D")->SetTitle("3D Gaussian Fit Center X");
        fTree->Branch("3DGaussianFitCenterY", &f3DGaussianFitCenterY, "3DGaussianFitCenterY/D")->SetTitle("3D Gaussian Fit Center Y");
        fTree->Branch("3DGaussianFitSigmaX", &f3DGaussianFitSigmaX, "3DGaussianFitSigmaX/D")->SetTitle("3D Gaussian Fit Sigma X");
        fTree->Branch("3DGaussianFitSigmaY", &f3DGaussianFitSigmaY, "3DGaussianFitSigmaY/D")->SetTitle("3D Gaussian Fit Sigma Y");
        fTree->Branch("3DGaussianFitAmplitude", &f3DGaussianFitAmplitude, "3DGaussianFitAmplitude/D")->SetTitle("3D Gaussian Fit Amplitude");
        fTree->Branch("3DGaussianFitVerticalOffset", &f3DGaussianFitVerticalOffset, "3DGaussianFitVerticalOffset/D")->SetTitle("3D Gaussian Fit Vertical Offset");
        fTree->Branch("3DGaussianFitCenterXErr", &f3DGaussianFitCenterXErr, "3DGaussianFitCenterXErr/D")->SetTitle("3D Gaussian Fit Center X Error");
        fTree->Branch("3DGaussianFitCenterYErr", &f3DGaussianFitCenterYErr, "3DGaussianFitCenterYErr/D")->SetTitle("3D Gaussian Fit Center Y Error");
        fTree->Branch("3DGaussianFitSigmaXErr", &f3DGaussianFitSigmaXErr, "3DGaussianFitSigmaXErr/D")->SetTitle("3D Gaussian Fit Sigma X Error");
        fTree->Branch("3DGaussianFitSigmaYErr", &f3DGaussianFitSigmaYErr, "3DGaussianFitSigmaYErr/D")->SetTitle("3D Gaussian Fit Sigma Y Error");
        fTree->Branch("3DGaussianFitAmplitudeErr", &f3DGaussianFitAmplitudeErr, "3DGaussianFitAmplitudeErr/D")->SetTitle("3D Gaussian Fit Amplitude Error");
        fTree->Branch("3DGaussianFitVerticalOffsetErr", &f3DGaussianFitVerticalOffsetErr, "3DGaussianFitVerticalOffsetErr/D")->SetTitle("3D Gaussian Fit Vertical Offset Error");
        fTree->Branch("3DGaussianFitChi2red", &f3DGaussianFitChi2red, "3DGaussianFitChi2red/D")->SetTitle("3D Gaussian Fit Reduced Chi-squared");
        fTree->Branch("3DGaussianFitPp", &f3DGaussianFitPp, "3DGaussianFitPp/D")->SetTitle("3D Gaussian Fit P-value");
        fTree->Branch("3DGaussianFitDOF", &f3DGaussianFitDOF, "3DGaussianFitDOF/I")->SetTitle("3D Gaussian Fit Degrees of Freedom");
        fTree->Branch("3DGaussianFitChargeUncertainty", &f3DGaussianFitChargeUncertainty, "3DGaussianFitChargeUncertainty/D")->SetTitle("3D Gaussian Fit Charge Uncertainty");
        fTree->Branch("3DGaussianFitSuccessful", &f3DGaussianFitSuccessful, "3DGaussianFitSuccessful/O")->SetTitle("3D Gaussian Fit Success Flag");
        }
        
        // 3D Lorentzian fit parameters
        if (Constants::ENABLE_3D_LORENTZIAN_FITTING) {
        fTree->Branch("3DLorentzianFitCenterX", &f3DLorentzianFitCenterX, "3DLorentzianFitCenterX/D")->SetTitle("3D Lorentzian Fit Center X");
        fTree->Branch("3DLorentzianFitCenterY", &f3DLorentzianFitCenterY, "3DLorentzianFitCenterY/D")->SetTitle("3D Lorentzian Fit Center Y");
        fTree->Branch("3DLorentzianFitGammaX", &f3DLorentzianFitGammaX, "3DLorentzianFitGammaX/D")->SetTitle("3D Lorentzian Fit Gamma X");
        fTree->Branch("3DLorentzianFitGammaY", &f3DLorentzianFitGammaY, "3DLorentzianFitGammaY/D")->SetTitle("3D Lorentzian Fit Gamma Y");
        fTree->Branch("3DLorentzianFitAmplitude", &f3DLorentzianFitAmplitude, "3DLorentzianFitAmplitude/D")->SetTitle("3D Lorentzian Fit Amplitude");
        fTree->Branch("3DLorentzianFitVerticalOffset", &f3DLorentzianFitVerticalOffset, "3DLorentzianFitVerticalOffset/D")->SetTitle("3D Lorentzian Fit Vertical Offset");
        fTree->Branch("3DLorentzianFitCenterXErr", &f3DLorentzianFitCenterXErr, "3DLorentzianFitCenterXErr/D")->SetTitle("3D Lorentzian Fit Center X Error");
        fTree->Branch("3DLorentzianFitCenterYErr", &f3DLorentzianFitCenterYErr, "3DLorentzianFitCenterYErr/D")->SetTitle("3D Lorentzian Fit Center Y Error");
        fTree->Branch("3DLorentzianFitGammaXErr", &f3DLorentzianFitGammaXErr, "3DLorentzianFitGammaXErr/D")->SetTitle("3D Lorentzian Fit Gamma X Error");
        fTree->Branch("3DLorentzianFitGammaYErr", &f3DLorentzianFitGammaYErr, "3DLorentzianFitGammaYErr/D")->SetTitle("3D Lorentzian Fit Gamma Y Error");
        fTree->Branch("3DLorentzianFitAmplitudeErr", &f3DLorentzianFitAmplitudeErr, "3DLorentzianFitAmplitudeErr/D")->SetTitle("3D Lorentzian Fit Amplitude Error");
        fTree->Branch("3DLorentzianFitVerticalOffsetErr", &f3DLorentzianFitVerticalOffsetErr, "3DLorentzianFitVerticalOffsetErr/D")->SetTitle("3D Lorentzian Fit Vertical Offset Error");
        fTree->Branch("3DLorentzianFitChi2red", &f3DLorentzianFitChi2red, "3DLorentzianFitChi2red/D")->SetTitle("3D Lorentzian Fit Reduced Chi-squared");
        fTree->Branch("3DLorentzianFitPp", &f3DLorentzianFitPp, "3DLorentzianFitPp/D")->SetTitle("3D Lorentzian Fit P-value");
        fTree->Branch("3DLorentzianFitDOF", &f3DLorentzianFitDOF, "3DLorentzianFitDOF/I")->SetTitle("3D Lorentzian Fit Degrees of Freedom");
        fTree->Branch("3DLorentzianFitChargeUncertainty", &f3DLorentzianFitChargeUncertainty, "3DLorentzianFitChargeUncertainty/D")->SetTitle("3D Lorentzian Fit Charge Uncertainty");
        fTree->Branch("3DLorentzianFitSuccessful", &f3DLorentzianFitSuccessful, "3DLorentzianFitSuccessful/O")->SetTitle("3D Lorentzian Fit Success Flag");
        }
        
        // 3D Power-Law Lorentzian fit parameters
        if (Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) {
        fTree->Branch("3DPowerLorentzianFitCenterX", &f3DPowerLorentzianFitCenterX, "3DPowerLorentzianFitCenterX/D")->SetTitle("3D Power-Law Lorentzian Fit Center X");
        fTree->Branch("3DPowerLorentzianFitCenterY", &f3DPowerLorentzianFitCenterY, "3DPowerLorentzianFitCenterY/D")->SetTitle("3D Power-Law Lorentzian Fit Center Y");
        fTree->Branch("3DPowerLorentzianFitGammaX", &f3DPowerLorentzianFitGammaX, "3DPowerLorentzianFitGammaX/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma X");
        fTree->Branch("3DPowerLorentzianFitGammaY", &f3DPowerLorentzianFitGammaY, "3DPowerLorentzianFitGammaY/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma Y");
        fTree->Branch("3DPowerLorentzianFitBeta", &f3DPowerLorentzianFitBeta, "3DPowerLorentzianFitBeta/D")->SetTitle("3D Power-Law Lorentzian Fit Beta (Power-Law Exponent)");
        fTree->Branch("3DPowerLorentzianFitAmplitude", &f3DPowerLorentzianFitAmplitude, "3DPowerLorentzianFitAmplitude/D")->SetTitle("3D Power-Law Lorentzian Fit Amplitude");
        fTree->Branch("3DPowerLorentzianFitVerticalOffset", &f3DPowerLorentzianFitVerticalOffset, "3DPowerLorentzianFitVerticalOffset/D")->SetTitle("3D Power-Law Lorentzian Fit Vertical Offset");
        fTree->Branch("3DPowerLorentzianFitCenterXErr", &f3DPowerLorentzianFitCenterXErr, "3DPowerLorentzianFitCenterXErr/D")->SetTitle("3D Power-Law Lorentzian Fit Center X Error");
        fTree->Branch("3DPowerLorentzianFitCenterYErr", &f3DPowerLorentzianFitCenterYErr, "3DPowerLorentzianFitCenterYErr/D")->SetTitle("3D Power-Law Lorentzian Fit Center Y Error");
        fTree->Branch("3DPowerLorentzianFitGammaXErr", &f3DPowerLorentzianFitGammaXErr, "3DPowerLorentzianFitGammaXErr/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma X Error");
        fTree->Branch("3DPowerLorentzianFitGammaYErr", &f3DPowerLorentzianFitGammaYErr, "3DPowerLorentzianFitGammaYErr/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma Y Error");
        fTree->Branch("3DPowerLorentzianFitBetaErr", &f3DPowerLorentzianFitBetaErr, "3DPowerLorentzianFitBetaErr/D")->SetTitle("3D Power-Law Lorentzian Fit Beta Error");
        fTree->Branch("3DPowerLorentzianFitAmplitudeErr", &f3DPowerLorentzianFitAmplitudeErr, "3DPowerLorentzianFitAmplitudeErr/D")->SetTitle("3D Power-Law Lorentzian Fit Amplitude Error");
        fTree->Branch("3DPowerLorentzianFitVerticalOffsetErr", &f3DPowerLorentzianFitVerticalOffsetErr, "3DPowerLorentzianFitVerticalOffsetErr/D")->SetTitle("3D Power-Law Lorentzian Fit Vertical Offset Error");
        fTree->Branch("3DPowerLorentzianFitChi2red", &f3DPowerLorentzianFitChi2red, "3DPowerLorentzianFitChi2red/D")->SetTitle("3D Power-Law Lorentzian Fit Reduced Chi-squared");
        fTree->Branch("3DPowerLorentzianFitPp", &f3DPowerLorentzianFitPp, "3DPowerLorentzianFitPp/D")->SetTitle("3D Power-Law Lorentzian Fit P-value");
        fTree->Branch("3DPowerLorentzianFitDOF", &f3DPowerLorentzianFitDOF, "3DPowerLorentzianFitDOF/I")->SetTitle("3D Power-Law Lorentzian Fit Degrees of Freedom");
        fTree->Branch("3DPowerLorentzianFitChargeUncertainty", &f3DPowerLorentzianFitChargeUncertainty, "3DPowerLorentzianFitChargeUncertainty/D")->SetTitle("3D Power-Law Lorentzian Fit Charge Uncertainty");
        fTree->Branch("3DPowerLorentzianFitSuccessful", &f3DPowerLorentzianFitSuccessful, "3DPowerLorentzianFitSuccessful/O")->SetTitle("3D Power-Law Lorentzian Fit Success Flag");
        }
        
        // =============================================
        // TRANSFORMED COORDINATE BRANCHES
        // =============================================
        // Gaussian diagonal transformed coordinates
        if (Constants::ENABLE_GAUSSIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("GaussMainDiagTransformedX", &fGaussMainDiagTransformedX, "GaussMainDiagTransformedX/D")->SetTitle("Gaussian Main Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("GaussMainDiagTransformedY", &fGaussMainDiagTransformedY, "GaussMainDiagTransformedY/D")->SetTitle("Gaussian Main Diagonal Transformed Y Coordinate [mm]");
            fTree->Branch("GaussSecondDiagTransformedX", &fGaussSecondDiagTransformedX, "GaussSecondDiagTransformedX/D")->SetTitle("Gaussian Secondary Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("GaussSecondDiagTransformedY", &fGaussSecondDiagTransformedY, "GaussSecondDiagTransformedY/D")->SetTitle("Gaussian Secondary Diagonal Transformed Y Coordinate [mm]");
        }
        
        // Lorentzian diagonal transformed coordinates
        if (Constants::ENABLE_LORENTZIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("LorentzMainDiagTransformedX", &fLorentzMainDiagTransformedX, "LorentzMainDiagTransformedX/D")->SetTitle("Lorentzian Main Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("LorentzMainDiagTransformedY", &fLorentzMainDiagTransformedY, "LorentzMainDiagTransformedY/D")->SetTitle("Lorentzian Main Diagonal Transformed Y Coordinate [mm]");
            fTree->Branch("LorentzSecondDiagTransformedX", &fLorentzSecondDiagTransformedX, "LorentzSecondDiagTransformedX/D")->SetTitle("Lorentzian Secondary Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("LorentzSecondDiagTransformedY", &fLorentzSecondDiagTransformedY, "LorentzSecondDiagTransformedY/D")->SetTitle("Lorentzian Secondary Diagonal Transformed Y Coordinate [mm]");
        }
        
        // Power-Law Lorentzian diagonal transformed coordinates
        if (Constants::ENABLE_POWER_LORENTZIAN_FITTING && Constants::ENABLE_DIAGONAL_FITTING) {
            fTree->Branch("PowerLorentzMainDiagTransformedX", &fPowerLorentzMainDiagTransformedX, "PowerLorentzMainDiagTransformedX/D")->SetTitle("Power-Law Lorentzian Main Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("PowerLorentzMainDiagTransformedY", &fPowerLorentzMainDiagTransformedY, "PowerLorentzMainDiagTransformedY/D")->SetTitle("Power-Law Lorentzian Main Diagonal Transformed Y Coordinate [mm]");
            fTree->Branch("PowerLorentzSecondDiagTransformedX", &fPowerLorentzSecondDiagTransformedX, "PowerLorentzSecondDiagTransformedX/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Transformed X Coordinate [mm]");
            fTree->Branch("PowerLorentzSecondDiagTransformedY", &fPowerLorentzSecondDiagTransformedY, "PowerLorentzSecondDiagTransformedY/D")->SetTitle("Power-Law Lorentzian Secondary Diagonal Transformed Y Coordinate [mm]");
        }
        
        G4cout << "Created ROOT tree with " << fTree->GetNbranches() << " branches" << G4endl;
        
        // Enable auto-save by default
        EnableAutoSave(1000);
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
                G4cout << "Worker thread: Successfully wrote " << fileName << G4endl;
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
                    G4cout << "Master thread: Successfully created merged file with " 
                           << verifyTree->GetEntries() << " entries" << G4endl;
                }
                verifyFile->Close();
                delete verifyFile;
            } else {
                G4cerr << "Master thread: Failed to verify merged file" << G4endl;
            }
            
            // Clean up worker files after successful merge
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

void RunAction::SetEventData(G4double edep, G4double x, G4double y)
{
    // Store energy deposition in MeV (Geant4 internal energy unit is MeV)
    fEdep = edep;
    
    // Store positions in mm (Geant4 internal length unit is mm)
    fTrueX = x;
    fTrueY = y;
}

void RunAction::SetInitialPosition(G4double x, G4double y, G4double z) 
{
    // Store positions in mm (Geant4 internal length unit is mm)
    fInitX = x;
    fInitY = y;
    fInitZ = z;
}

void RunAction::SetNearestPixelPosition(G4double x, G4double y)
{
    // Store positions in mm (Geant4 internal length unit is mm)
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
    fNonPixel_GridNeighborhoodAngles = angles;
}

void RunAction::SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                                 const std::vector<G4double>& distances,
                                 const std::vector<G4double>& chargeValues,
                                 const std::vector<G4double>& chargeCoulombs)
{
    // Store the neighborhood (9x9) grid charge sharing data for non-pixel hits
    fNonPixel_GridNeighborhoodChargeFractions = chargeFractions;
    fNonPixel_GridNeighborhoodDistances = distances;
    fNonPixel_GridNeighborhoodCharge = chargeCoulombs;
}

void RunAction::FillTree()
{
    if (!fTree || !fRootFile || fRootFile->IsZombie()) {
        G4cerr << "Error: Invalid ROOT file or tree in FillTree()" << G4endl;
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(fRootMutex);
        fTree->Fill();
        
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

void RunAction::Set2DGaussianFitResults(G4double x_center, G4double x_sigma, G4double x_amplitude,
                                        G4double x_center_err, G4double x_sigma_err, G4double x_amplitude_err,
                                        G4double x_vertical_offset, G4double x_vertical_offset_err,
                                        G4double x_chi2red, G4double x_pp, G4int x_dof,
                                        G4double y_center, G4double y_sigma, G4double y_amplitude,
                                        G4double y_center_err, G4double y_sigma_err, G4double y_amplitude_err,
                                        G4double y_vertical_offset, G4double y_vertical_offset_err,
                                        G4double y_chi2red, G4double y_pp, G4int y_dof,
                                        G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                        G4bool fit_successful)
{
    // Store 2D Gaussian fit results from central row (X fit)
    fGaussFitRowCenter = x_center;
    fGaussFitRowStdev = x_sigma;
    fGaussFitRowAmplitude = x_amplitude;
    fGaussFitRowCenterErr = x_center_err;
    fGaussFitRowStdevErr = x_sigma_err;
    fGaussFitRowAmplitudeErr = x_amplitude_err;
    fGaussFitRowVerticalOffset = x_vertical_offset;
    fGaussFitRowVerticalOffsetErr = x_vertical_offset_err;
    fGaussFitRowChi2red = x_chi2red;
    fGaussFitRowPp = x_pp;
    fGaussFitRowDOF = x_dof;
    
    // Store 2D Gaussian fit results from central column (Y fit)
    fGaussFitColumnCenter = y_center;
    fGaussFitColumnStdev = y_sigma;
    fGaussFitColumnAmplitude = y_amplitude;
    fGaussFitColumnCenterErr = y_center_err;
    fGaussFitColumnStdevErr = y_sigma_err;
    fGaussFitColumnAmplitudeErr = y_amplitude_err;
    fGaussFitColumnVerticalOffset = y_vertical_offset;
    fGaussFitColumnVerticalOffsetErr = y_vertical_offset_err;
    fGaussFitColumnChi2red = y_chi2red;
    fGaussFitColumnPp = y_pp;
    fGaussFitColumnDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fGaussFitRowChargeUncertainty = x_charge_uncertainty;
        fGaussFitColumnChargeUncertainty = y_charge_uncertainty;
    } else {
        fGaussFitRowChargeUncertainty = 0.0;
        fGaussFitColumnChargeUncertainty = 0.0;
    }
    
    // Calculate delta values for row and column fits vs true position
    // Use individual fit validity checks similar to diagonal fits for consistency
    if (fit_successful) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fGaussRowDeltaX = x_center - fTrueX;      // x_row_fit - x_true
        } else {
            fGaussRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fGaussColumnDeltaY = y_center - fTrueY;   // y_column_fit - y_true
        } else {
            fGaussColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fGaussRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

void RunAction::SetDiagonalGaussianFitResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_amplitude,
                                             G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_amplitude_err,
                                             G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                             G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                             G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_amplitude,
                                             G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_amplitude_err,
                                             G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                             G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                             G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_amplitude,
                                             G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_amplitude_err,
                                             G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                             G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                             G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_amplitude,
                                             G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_amplitude_err,
                                             G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                             G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                             G4bool fit_successful)
{
    // Store main diagonal X fit results
    fGaussFitMainDiagXCenter = main_diag_x_center;
    fGaussFitMainDiagXStdev = main_diag_x_sigma;
    fGaussFitMainDiagXAmplitude = main_diag_x_amplitude;
    fGaussFitMainDiagXCenterErr = main_diag_x_center_err;
    fGaussFitMainDiagXStdevErr = main_diag_x_sigma_err;
    fGaussFitMainDiagXAmplitudeErr = main_diag_x_amplitude_err;
    fGaussFitMainDiagXVerticalOffset = main_diag_x_vertical_offset;
    fGaussFitMainDiagXVerticalOffsetErr = main_diag_x_vertical_offset_err;
    fGaussFitMainDiagXChi2red = main_diag_x_chi2red;
    fGaussFitMainDiagXPp = main_diag_x_pp;
    fGaussFitMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results
    fGaussFitMainDiagYCenter = main_diag_y_center;
    fGaussFitMainDiagYStdev = main_diag_y_sigma;
    fGaussFitMainDiagYAmplitude = main_diag_y_amplitude;
    fGaussFitMainDiagYCenterErr = main_diag_y_center_err;
    fGaussFitMainDiagYStdevErr = main_diag_y_sigma_err;
    fGaussFitMainDiagYAmplitudeErr = main_diag_y_amplitude_err;
    fGaussFitMainDiagYVerticalOffset = main_diag_y_vertical_offset;
    fGaussFitMainDiagYVerticalOffsetErr = main_diag_y_vertical_offset_err;
    fGaussFitMainDiagYChi2red = main_diag_y_chi2red;
    fGaussFitMainDiagYPp = main_diag_y_pp;
    fGaussFitMainDiagYDOF = main_diag_y_dof;
    
    // Store secondary diagonal X fit results
    fGaussFitSecondDiagXCenter = sec_diag_x_center;
    fGaussFitSecondDiagXStdev = sec_diag_x_sigma;
    fGaussFitSecondDiagXAmplitude = sec_diag_x_amplitude;
    fGaussFitSecondDiagXCenterErr = sec_diag_x_center_err;
    fGaussFitSecondDiagXStdevErr = sec_diag_x_sigma_err;
    fGaussFitSecondDiagXAmplitudeErr = sec_diag_x_amplitude_err;
    fGaussFitSecondDiagXVerticalOffset = sec_diag_x_vertical_offset;
    fGaussFitSecondDiagXVerticalOffsetErr = sec_diag_x_vertical_offset_err;
    fGaussFitSecondDiagXChi2red = sec_diag_x_chi2red;
    fGaussFitSecondDiagXPp = sec_diag_x_pp;
    fGaussFitSecondDiagXDOF = sec_diag_x_dof;
    
    // Store secondary diagonal Y fit results
    fGaussFitSecondDiagYCenter = sec_diag_y_center;
    fGaussFitSecondDiagYStdev = sec_diag_y_sigma;
    fGaussFitSecondDiagYAmplitude = sec_diag_y_amplitude;
    fGaussFitSecondDiagYCenterErr = sec_diag_y_center_err;
    fGaussFitSecondDiagYStdevErr = sec_diag_y_sigma_err;
    fGaussFitSecondDiagYAmplitudeErr = sec_diag_y_amplitude_err;
    fGaussFitSecondDiagYVerticalOffset = sec_diag_y_vertical_offset;
    fGaussFitSecondDiagYVerticalOffsetErr = sec_diag_y_vertical_offset_err;
    fGaussFitSecondDiagYChi2red = sec_diag_y_chi2red;
    fGaussFitSecondDiagYPp = sec_diag_y_pp;
    fGaussFitSecondDiagYDOF = sec_diag_y_dof;
    

    
    // Calculate transformed diagonal coordinates using rotation matrix
    CalculateTransformedDiagonalCoordinates();
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

void RunAction::Set2DLorentzianFitResults(G4double x_center, G4double x_gamma, G4double x_amplitude,
                                          G4double x_center_err, G4double x_gamma_err, G4double x_amplitude_err,
                                          G4double x_vertical_offset, G4double x_vertical_offset_err,
                                          G4double x_chi2red, G4double x_pp, G4int x_dof,
                                          G4double y_center, G4double y_gamma, G4double y_amplitude,
                                          G4double y_center_err, G4double y_gamma_err, G4double y_amplitude_err,
                                          G4double y_vertical_offset, G4double y_vertical_offset_err,
                                          G4double y_chi2red, G4double y_pp, G4int y_dof,
                                          G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                          G4bool fit_successful)
{
    // Store 2D Lorentzian fit results from central row (X fit)
    fLorentzFitRowCenter = x_center;
    fLorentzFitRowGamma = x_gamma;
    fLorentzFitRowAmplitude = x_amplitude;
    fLorentzFitRowCenterErr = x_center_err;
    fLorentzFitRowGammaErr = x_gamma_err;
    fLorentzFitRowAmplitudeErr = x_amplitude_err;
    fLorentzFitRowVerticalOffset = x_vertical_offset;
    fLorentzFitRowVerticalOffsetErr = x_vertical_offset_err;
    fLorentzFitRowChi2red = x_chi2red;
    fLorentzFitRowPp = x_pp;
    fLorentzFitRowDOF = x_dof;
    
    // Store 2D Lorentzian fit results from central column (Y fit)
    fLorentzFitColumnCenter = y_center;
    fLorentzFitColumnGamma = y_gamma;
    fLorentzFitColumnAmplitude = y_amplitude;
    fLorentzFitColumnCenterErr = y_center_err;
    fLorentzFitColumnGammaErr = y_gamma_err;
    fLorentzFitColumnAmplitudeErr = y_amplitude_err;
    fLorentzFitColumnVerticalOffset = y_vertical_offset;
    fLorentzFitColumnVerticalOffsetErr = y_vertical_offset_err;
    fLorentzFitColumnChi2red = y_chi2red;
    fLorentzFitColumnPp = y_pp;
    fLorentzFitColumnDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fLorentzFitRowChargeUncertainty = x_charge_uncertainty;
        fLorentzFitColumnChargeUncertainty = y_charge_uncertainty;
    } else {
        fLorentzFitRowChargeUncertainty = 0.0;
        fLorentzFitColumnChargeUncertainty = 0.0;
    }
    
    // Calculate delta values for row and column fits vs true position
    if (fit_successful) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fLorentzRowDeltaX = x_center - fTrueX;      // x_row_fit - x_true
        } else {
            fLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fLorentzColumnDeltaY = y_center - fTrueY;   // y_column_fit - y_true
        } else {
            fLorentzColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

void RunAction::SetDiagonalLorentzianFitResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_amplitude,
                                               G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_amplitude_err,
                                               G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                               G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                               G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_amplitude,
                                               G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_amplitude_err,
                                               G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                               G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                               G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_amplitude,
                                               G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_amplitude_err,
                                               G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                               G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                               G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_amplitude,
                                               G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_amplitude_err,
                                               G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                               G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                               G4bool fit_successful)
{
    // Store main diagonal X fit results
    fLorentzFitMainDiagXCenter = main_diag_x_center;
    fLorentzFitMainDiagXGamma = main_diag_x_gamma;
    fLorentzFitMainDiagXAmplitude = main_diag_x_amplitude;
    fLorentzFitMainDiagXCenterErr = main_diag_x_center_err;
    fLorentzFitMainDiagXGammaErr = main_diag_x_gamma_err;
    fLorentzFitMainDiagXAmplitudeErr = main_diag_x_amplitude_err;
    fLorentzFitMainDiagXVerticalOffset = main_diag_x_vertical_offset;
    fLorentzFitMainDiagXVerticalOffsetErr = main_diag_x_vertical_offset_err;
    fLorentzFitMainDiagXChi2red = main_diag_x_chi2red;
    fLorentzFitMainDiagXPp = main_diag_x_pp;
    fLorentzFitMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results
    fLorentzFitMainDiagYCenter = main_diag_y_center;
    fLorentzFitMainDiagYGamma = main_diag_y_gamma;
    fLorentzFitMainDiagYAmplitude = main_diag_y_amplitude;
    fLorentzFitMainDiagYCenterErr = main_diag_y_center_err;
    fLorentzFitMainDiagYGammaErr = main_diag_y_gamma_err;
    fLorentzFitMainDiagYAmplitudeErr = main_diag_y_amplitude_err;
    fLorentzFitMainDiagYVerticalOffset = main_diag_y_vertical_offset;
    fLorentzFitMainDiagYVerticalOffsetErr = main_diag_y_vertical_offset_err;
    fLorentzFitMainDiagYChi2red = main_diag_y_chi2red;
    fLorentzFitMainDiagYPp = main_diag_y_pp;
    fLorentzFitMainDiagYDOF = main_diag_y_dof;
    
    // Store secondary diagonal X fit results
    fLorentzFitSecondDiagXCenter = sec_diag_x_center;
    fLorentzFitSecondDiagXGamma = sec_diag_x_gamma;
    fLorentzFitSecondDiagXAmplitude = sec_diag_x_amplitude;
    fLorentzFitSecondDiagXCenterErr = sec_diag_x_center_err;
    fLorentzFitSecondDiagXGammaErr = sec_diag_x_gamma_err;
    fLorentzFitSecondDiagXAmplitudeErr = sec_diag_x_amplitude_err;
    fLorentzFitSecondDiagXVerticalOffset = sec_diag_x_vertical_offset;
    fLorentzFitSecondDiagXVerticalOffsetErr = sec_diag_x_vertical_offset_err;
    fLorentzFitSecondDiagXChi2red = sec_diag_x_chi2red;
    fLorentzFitSecondDiagXPp = sec_diag_x_pp;
    fLorentzFitSecondDiagXDOF = sec_diag_x_dof;
    
    // Store secondary diagonal Y fit results
    fLorentzFitSecondDiagYCenter = sec_diag_y_center;
    fLorentzFitSecondDiagYGamma = sec_diag_y_gamma;
    fLorentzFitSecondDiagYAmplitude = sec_diag_y_amplitude;
    fLorentzFitSecondDiagYCenterErr = sec_diag_y_center_err;
    fLorentzFitSecondDiagYGammaErr = sec_diag_y_gamma_err;
    fLorentzFitSecondDiagYAmplitudeErr = sec_diag_y_amplitude_err;
    fLorentzFitSecondDiagYVerticalOffset = sec_diag_y_vertical_offset;
    fLorentzFitSecondDiagYVerticalOffsetErr = sec_diag_y_vertical_offset_err;
    fLorentzFitSecondDiagYChi2red = sec_diag_y_chi2red;
    fLorentzFitSecondDiagYPp = sec_diag_y_pp;
    fLorentzFitSecondDiagYDOF = sec_diag_y_dof;

    // Calculate transformed diagonal coordinates using rotation matrix
    CalculateTransformedDiagonalCoordinates();
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

// =============================================
// COORDINATE TRANSFORMATION HELPER METHODS
// =============================================

void RunAction::TransformDiagonalCoordinates(G4double x_prime, G4double y_prime, G4double theta_deg, 
                                             G4double& x_transformed, G4double& y_transformed)
{
    // Convert angle to radians
    const G4double PI = 3.14159265358979323846;
    G4double theta_rad = theta_deg * PI / 180.0;
    
    // Calculate rotation matrix components
    G4double cos_theta = std::cos(theta_rad);
    G4double sin_theta = std::sin(theta_rad);
    
    // Apply rotation matrix transformation:
    // [cos(θ) -sin(θ)] [x']   [x]
    // [sin(θ)  cos(θ)] [y'] = [y]
    x_transformed = cos_theta * x_prime - sin_theta * y_prime;
    y_transformed = sin_theta * x_prime + cos_theta * y_prime;
}

void RunAction::CalculateTransformedDiagonalCoordinates()
{
    // Safety check for valid true position data
    if (std::isnan(fTrueX) || std::isnan(fTrueY)) {
        G4cerr << "RunAction: Warning - Invalid true position data, cannot calculate transformed coordinates" << G4endl;
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
    // FIX: Gaussian diagonal fits also return diagonal coordinates that need transformation
    auto sToDxDyMain = [&](double s){ return std::make_pair(s*invSqrt2, s*invSqrt2); };
    auto sToDxDySec  = [&](double s){ return std::make_pair(s*invSqrt2, -s*invSqrt2); };

    // Main diagonal
    double sMain = NaN;
    if (fGaussFitMainDiagXDOF > 0)       sMain = fGaussFitMainDiagXCenter;
    else if (fGaussFitMainDiagYDOF > 0)  sMain = fGaussFitMainDiagYCenter;
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
    if (fGaussFitSecondDiagXDOF > 0)       sSec = fGaussFitSecondDiagXCenter;
    else if (fGaussFitSecondDiagYDOF > 0)  sSec = fGaussFitSecondDiagYCenter;
    if (!std::isnan(sSec)) {
        auto [dx,dy] = sToDxDySec(sSec);
        setXY(fPixelX+dx, fPixelY+dy,
              fGaussSecondDiagTransformedX, fGaussSecondDiagTransformedY,
              fGaussSecondDiagTransformedDeltaX, fGaussSecondDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fGaussSecondDiagTransformedX, fGaussSecondDiagTransformedY,
              fGaussSecondDiagTransformedDeltaX, fGaussSecondDiagTransformedDeltaY);
    }

    // ------------------
    //   L O R E N T Z I A N
    // ------------------
    // Main diagonal
    double sMainLorentz = NaN;
    if (fLorentzFitMainDiagXDOF > 0)       sMainLorentz = fLorentzFitMainDiagXCenter;
    else if (fLorentzFitMainDiagYDOF > 0)  sMainLorentz = fLorentzFitMainDiagYCenter;
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
    if (fLorentzFitSecondDiagXDOF > 0)       sSecLorentz = fLorentzFitSecondDiagXCenter;
    else if (fLorentzFitSecondDiagYDOF > 0)  sSecLorentz = fLorentzFitSecondDiagYCenter;
    if (!std::isnan(sSecLorentz)) {
        auto [dx,dy] = sToDxDySec(sSecLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fLorentzSecondDiagTransformedX, fLorentzSecondDiagTransformedY,
              fLorentzSecondDiagTransformedDeltaX, fLorentzSecondDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fLorentzSecondDiagTransformedX, fLorentzSecondDiagTransformedY,
              fLorentzSecondDiagTransformedDeltaX, fLorentzSecondDiagTransformedDeltaY);
    }

    // -------------------------------
    // P O W E R - L A W   L O R E N T Z I A N
    // -------------------------------
    // FIX: Power Lorentzian diagonal fits also return diagonal coordinates that need transformation
    
    // Main diagonal
    double sMainPowerLorentz = NaN;
    if (fPowerLorentzFitMainDiagXDOF > 0)       sMainPowerLorentz = fPowerLorentzFitMainDiagXCenter;
    else if (fPowerLorentzFitMainDiagYDOF > 0)  sMainPowerLorentz = fPowerLorentzFitMainDiagYCenter;
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
    if (fPowerLorentzFitSecondDiagXDOF > 0)       sSecPowerLorentz = fPowerLorentzFitSecondDiagXCenter;
    else if (fPowerLorentzFitSecondDiagYDOF > 0)  sSecPowerLorentz = fPowerLorentzFitSecondDiagYCenter;
    if (!std::isnan(sSecPowerLorentz)) {
        auto [dx,dy] = sToDxDySec(sSecPowerLorentz);
        setXY(fPixelX+dx, fPixelY+dy,
              fPowerLorentzSecondDiagTransformedX, fPowerLorentzSecondDiagTransformedY,
              fPowerLorentzSecondDiagTransformedDeltaX, fPowerLorentzSecondDiagTransformedDeltaY);
    } else {
        setXY(NaN,NaN,
              fPowerLorentzSecondDiagTransformedX, fPowerLorentzSecondDiagTransformedY,
              fPowerLorentzSecondDiagTransformedDeltaX, fPowerLorentzSecondDiagTransformedDeltaY);
    }
}

void RunAction::CalculateMeanEstimations()
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
    
    // For Gaussian estimations, collect X coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed X
    if (!std::isnan(fGaussMainDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal transformed X  
    if (!std::isnan(fGaussSecondDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussSecondDiagTransformedX);
    }
    
    // For Gaussian estimations, collect Y coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed Y
    if (!std::isnan(fGaussMainDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal transformed Y
    if (!std::isnan(fGaussSecondDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussSecondDiagTransformedY);
    }
    
    // For Lorentzian estimations, collect X coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed X
    if (!std::isnan(fLorentzMainDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal transformed X
    if (!std::isnan(fLorentzSecondDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzSecondDiagTransformedX);
    }
    
    // For Lorentzian estimations, collect Y coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed Y
    if (!std::isnan(fLorentzMainDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal transformed Y
    if (!std::isnan(fLorentzSecondDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzSecondDiagTransformedY);
    }
    
    // For Power-Law Lorentzian estimations, collect X coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed X
    if (!std::isnan(fPowerLorentzMainDiagTransformedX)) {
        power_lorentz_x_coords.push_back(fPowerLorentzMainDiagTransformedX);
    }
    
    // 2. Secondary diagonal transformed X
    if (!std::isnan(fPowerLorentzSecondDiagTransformedX)) {
        power_lorentz_x_coords.push_back(fPowerLorentzSecondDiagTransformedX);
    }
    
    // For Power-Law Lorentzian estimations, collect Y coordinates:
    // ONLY use transformed diagonal coordinates (exclude row/column fits)
    
    // 1. Main diagonal transformed Y
    if (!std::isnan(fPowerLorentzMainDiagTransformedY)) {
        power_lorentz_y_coords.push_back(fPowerLorentzMainDiagTransformedY);
    }
    
    // 2. Secondary diagonal transformed Y
    if (!std::isnan(fPowerLorentzSecondDiagTransformedY)) {
        power_lorentz_y_coords.push_back(fPowerLorentzSecondDiagTransformedY);
    }
    
    // =============================================
    // ADD 3D FITTING RESULTS TO MEAN CALCULATIONS
    // =============================================
    
    // Add 3D Gaussian fit results to Gaussian estimation collections
    if (!std::isnan(f3DGaussianFitCenterX) && f3DGaussianFitSuccessful) {
        gauss_x_coords.push_back(f3DGaussianFitCenterX);
    }
    if (!std::isnan(f3DGaussianFitCenterY) && f3DGaussianFitSuccessful) {
        gauss_y_coords.push_back(f3DGaussianFitCenterY);
    }
    
    // Add 3D Lorentzian fit results to Lorentzian estimation collections
    if (!std::isnan(f3DLorentzianFitCenterX) && f3DLorentzianFitSuccessful) {
        lorentz_x_coords.push_back(f3DLorentzianFitCenterX);
    }
    if (!std::isnan(f3DLorentzianFitCenterY) && f3DLorentzianFitSuccessful) {
        lorentz_y_coords.push_back(f3DLorentzianFitCenterY);
    }
    
    // Add 3D Power-Law Lorentzian fit results to Power-Law Lorentzian estimation collections
    if (!std::isnan(f3DPowerLorentzianFitCenterX) && f3DPowerLorentzianFitSuccessful) {
        power_lorentz_x_coords.push_back(f3DPowerLorentzianFitCenterX);
    }
    if (!std::isnan(f3DPowerLorentzianFitCenterY) && f3DPowerLorentzianFitSuccessful) {
        power_lorentz_y_coords.push_back(f3DPowerLorentzianFitCenterY);
    }

    
    // Calculate mean X coordinate estimations and their deltas
    if (!gauss_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : gauss_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / gauss_x_coords.size();
        fGaussMeanTrueDeltaX = mean_x - fTrueX;
    } else {
        fGaussMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!gauss_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : gauss_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / gauss_y_coords.size();
        fGaussMeanTrueDeltaY = mean_y - fTrueY;
    } else {
        fGaussMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!lorentz_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : lorentz_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / lorentz_x_coords.size();
        fLorentzMeanTrueDeltaX = mean_x - fTrueX;
    } else {
        fLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!lorentz_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : lorentz_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / lorentz_y_coords.size();
        fLorentzMeanTrueDeltaY = mean_y - fTrueY;
    } else {
        fLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate Power-Law Lorentzian mean coordinate estimations and their deltas
    if (!power_lorentz_x_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : power_lorentz_x_coords) {
            sum += coord;
        }
        G4double mean_x = sum / power_lorentz_x_coords.size();
        fPowerLorentzMeanTrueDeltaX = mean_x - fTrueX;
    } else {
        fPowerLorentzMeanTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    if (!power_lorentz_y_coords.empty()) {
        G4double sum = 0.0;
        for (const auto& coord : power_lorentz_y_coords) {
            sum += coord;
        }
        G4double mean_y = sum / power_lorentz_y_coords.size();
        fPowerLorentzMeanTrueDeltaY = mean_y - fTrueY;
    } else {
        fPowerLorentzMeanTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }

}

// =============================================
// LAPLACE FITTING RESULTS SETTER METHODS
// =============================================





// Store automatic radius selection results
void RunAction::SetAutoRadiusResults(G4int selectedRadius)
{
    fSelectedRadius = selectedRadius;
}

// =============================================
// POWER LORENTZIAN FITTING RESULTS SETTER METHODS
// =============================================

void RunAction::Set2DPowerLorentzianFitResults(G4double x_center, G4double x_gamma, G4double x_beta, G4double x_amplitude,
                                               G4double x_center_err, G4double x_gamma_err, G4double x_beta_err, G4double x_amplitude_err,
                                               G4double x_vertical_offset, G4double x_vertical_offset_err,
                                               G4double x_chi2red, G4double x_pp, G4int x_dof,
                                               G4double y_center, G4double y_gamma, G4double y_beta, G4double y_amplitude,
                                               G4double y_center_err, G4double y_gamma_err, G4double y_beta_err, G4double y_amplitude_err,
                                               G4double y_vertical_offset, G4double y_vertical_offset_err,
                                               G4double y_chi2red, G4double y_pp, G4int y_dof,
                                               G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                               G4bool fit_successful)
{
    // Store X direction (row) fit results - Power-Law Lorentzian model
    fPowerLorentzFitRowCenter = x_center;
    fPowerLorentzFitRowGamma = x_gamma;
    fPowerLorentzFitRowBeta = x_beta;
    fPowerLorentzFitRowAmplitude = x_amplitude;
    fPowerLorentzFitRowCenterErr = x_center_err;
    fPowerLorentzFitRowGammaErr = x_gamma_err;
    fPowerLorentzFitRowBetaErr = x_beta_err;
    fPowerLorentzFitRowAmplitudeErr = x_amplitude_err;
    fPowerLorentzFitRowVerticalOffset = x_vertical_offset;
    fPowerLorentzFitRowVerticalOffsetErr = x_vertical_offset_err;
    fPowerLorentzFitRowChi2red = x_chi2red;
    fPowerLorentzFitRowPp = x_pp;
    fPowerLorentzFitRowDOF = x_dof;
    
    // Store Y direction (column) fit results - Power-Law Lorentzian model
    fPowerLorentzFitColumnCenter = y_center;
    fPowerLorentzFitColumnGamma = y_gamma;
    fPowerLorentzFitColumnBeta = y_beta;
    fPowerLorentzFitColumnAmplitude = y_amplitude;
    fPowerLorentzFitColumnCenterErr = y_center_err;
    fPowerLorentzFitColumnGammaErr = y_gamma_err;
    fPowerLorentzFitColumnBetaErr = y_beta_err;
    fPowerLorentzFitColumnAmplitudeErr = y_amplitude_err;
    fPowerLorentzFitColumnVerticalOffset = y_vertical_offset;
    fPowerLorentzFitColumnVerticalOffsetErr = y_vertical_offset_err;
    fPowerLorentzFitColumnChi2red = y_chi2red;
    fPowerLorentzFitColumnPp = y_pp;
    fPowerLorentzFitColumnDOF = y_dof;
    
    // Store charge uncertainties (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fPowerLorentzFitRowChargeUncertainty = x_charge_uncertainty;
        fPowerLorentzFitColumnChargeUncertainty = y_charge_uncertainty;
    } else {
        fPowerLorentzFitRowChargeUncertainty = 0.0;
        fPowerLorentzFitColumnChargeUncertainty = 0.0;
    }
    
    // Calculate delta values for row and column fits vs true position
    if (fit_successful) {
        // Check X fit validity (row fit) - use dof as success indicator
        if (x_dof > 0) {
            fPowerLorentzRowDeltaX = x_center - fTrueX;      // x_row_fit - x_true
        } else {
            fPowerLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Check Y fit validity (column fit) - use dof as success indicator  
        if (y_dof > 0) {
            fPowerLorentzColumnDeltaY = y_center - fTrueY;   // y_column_fit - y_true
        } else {
            fPowerLorentzColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // Set row and column delta values to NaN for failed overall fits
        fPowerLorentzRowDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fPowerLorentzColumnDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

void RunAction::SetDiagonalPowerLorentzianFitResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_beta, G4double main_diag_x_amplitude,
                                                     G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_beta_err, G4double main_diag_x_amplitude_err,
                                                     G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                                     G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                                     G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_beta, G4double main_diag_y_amplitude,
                                                     G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_beta_err, G4double main_diag_y_amplitude_err,
                                                     G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                                     G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                                     G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_beta, G4double sec_diag_x_amplitude,
                                                     G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_beta_err, G4double sec_diag_x_amplitude_err,
                                                     G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                                     G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                                     G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_beta, G4double sec_diag_y_amplitude,
                                                     G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_beta_err, G4double sec_diag_y_amplitude_err,
                                                     G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                                     G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                                     G4bool fit_successful)
{
    // Store main diagonal X fit results - Power-Law Lorentzian model
    fPowerLorentzFitMainDiagXCenter = main_diag_x_center;
    fPowerLorentzFitMainDiagXGamma = main_diag_x_gamma;
    fPowerLorentzFitMainDiagXBeta = main_diag_x_beta;
    fPowerLorentzFitMainDiagXAmplitude = main_diag_x_amplitude;
    fPowerLorentzFitMainDiagXCenterErr = main_diag_x_center_err;
    fPowerLorentzFitMainDiagXGammaErr = main_diag_x_gamma_err;
    fPowerLorentzFitMainDiagXBetaErr = main_diag_x_beta_err;
    fPowerLorentzFitMainDiagXAmplitudeErr = main_diag_x_amplitude_err;
    fPowerLorentzFitMainDiagXVerticalOffset = main_diag_x_vertical_offset;
    fPowerLorentzFitMainDiagXVerticalOffsetErr = main_diag_x_vertical_offset_err;
    fPowerLorentzFitMainDiagXChi2red = main_diag_x_chi2red;
    fPowerLorentzFitMainDiagXPp = main_diag_x_pp;
    fPowerLorentzFitMainDiagXDOF = main_diag_x_dof;
    
    // Store main diagonal Y fit results - Power-Law Lorentzian model
    fPowerLorentzFitMainDiagYCenter = main_diag_y_center;
    fPowerLorentzFitMainDiagYGamma = main_diag_y_gamma;
    fPowerLorentzFitMainDiagYBeta = main_diag_y_beta;
    fPowerLorentzFitMainDiagYAmplitude = main_diag_y_amplitude;
    fPowerLorentzFitMainDiagYCenterErr = main_diag_y_center_err;
    fPowerLorentzFitMainDiagYGammaErr = main_diag_y_gamma_err;
    fPowerLorentzFitMainDiagYBetaErr = main_diag_y_beta_err;
    fPowerLorentzFitMainDiagYAmplitudeErr = main_diag_y_amplitude_err;
    fPowerLorentzFitMainDiagYVerticalOffset = main_diag_y_vertical_offset;
    fPowerLorentzFitMainDiagYVerticalOffsetErr = main_diag_y_vertical_offset_err;
    fPowerLorentzFitMainDiagYChi2red = main_diag_y_chi2red;
    fPowerLorentzFitMainDiagYPp = main_diag_y_pp;
    fPowerLorentzFitMainDiagYDOF = main_diag_y_dof;
    
    // Store secondary diagonal X fit results - Power-Law Lorentzian model
    fPowerLorentzFitSecondDiagXCenter = sec_diag_x_center;
    fPowerLorentzFitSecondDiagXGamma = sec_diag_x_gamma;
    fPowerLorentzFitSecondDiagXBeta = sec_diag_x_beta;
    fPowerLorentzFitSecondDiagXAmplitude = sec_diag_x_amplitude;
    fPowerLorentzFitSecondDiagXCenterErr = sec_diag_x_center_err;
    fPowerLorentzFitSecondDiagXGammaErr = sec_diag_x_gamma_err;
    fPowerLorentzFitSecondDiagXBetaErr = sec_diag_x_beta_err;
    fPowerLorentzFitSecondDiagXAmplitudeErr = sec_diag_x_amplitude_err;
    fPowerLorentzFitSecondDiagXVerticalOffset = sec_diag_x_vertical_offset;
    fPowerLorentzFitSecondDiagXVerticalOffsetErr = sec_diag_x_vertical_offset_err;
    fPowerLorentzFitSecondDiagXChi2red = sec_diag_x_chi2red;
    fPowerLorentzFitSecondDiagXPp = sec_diag_x_pp;
    fPowerLorentzFitSecondDiagXDOF = sec_diag_x_dof;
    
    // Store secondary diagonal Y fit results - Power-Law Lorentzian model
    fPowerLorentzFitSecondDiagYCenter = sec_diag_y_center;
    fPowerLorentzFitSecondDiagYGamma = sec_diag_y_gamma;
    fPowerLorentzFitSecondDiagYBeta = sec_diag_y_beta;
    fPowerLorentzFitSecondDiagYAmplitude = sec_diag_y_amplitude;
    fPowerLorentzFitSecondDiagYCenterErr = sec_diag_y_center_err;
    fPowerLorentzFitSecondDiagYGammaErr = sec_diag_y_gamma_err;
    fPowerLorentzFitSecondDiagYBetaErr = sec_diag_y_beta_err;
    fPowerLorentzFitSecondDiagYAmplitudeErr = sec_diag_y_amplitude_err;
    fPowerLorentzFitSecondDiagYVerticalOffset = sec_diag_y_vertical_offset;
    fPowerLorentzFitSecondDiagYVerticalOffsetErr = sec_diag_y_vertical_offset_err;
    fPowerLorentzFitSecondDiagYChi2red = sec_diag_y_chi2red;
    fPowerLorentzFitSecondDiagYPp = sec_diag_y_pp;
    fPowerLorentzFitSecondDiagYDOF = sec_diag_y_dof;

    // Calculate transformed diagonal coordinates using rotation matrix
    CalculateTransformedDiagonalCoordinates();
    
    // Calculate mean estimations from all fitting methods
    CalculateMeanEstimations();
}

// =============================================
// 3D FITTING RESULTS SETTER METHODS
// =============================================

void RunAction::Set3DLorentzianFitResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double amplitude, G4double vertical_offset,
                                           G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double amplitude_err, G4double vertical_offset_err,
                                           G4double chi2red, G4double pp, G4int dof,
                                           G4double charge_uncertainty,
                                           G4bool fit_successful)
{
    // Store 3D Lorentzian fit parameters
    f3DLorentzianFitCenterX = center_x;
    f3DLorentzianFitCenterY = center_y;
    f3DLorentzianFitGammaX = gamma_x;
    f3DLorentzianFitGammaY = gamma_y;
    f3DLorentzianFitAmplitude = amplitude;
    f3DLorentzianFitVerticalOffset = vertical_offset;
    
    // Store 3D Lorentzian fit parameter errors
    f3DLorentzianFitCenterXErr = center_x_err;
    f3DLorentzianFitCenterYErr = center_y_err;
    f3DLorentzianFitGammaXErr = gamma_x_err;
    f3DLorentzianFitGammaYErr = gamma_y_err;
    f3DLorentzianFitAmplitudeErr = amplitude_err;
    f3DLorentzianFitVerticalOffsetErr = vertical_offset_err;
    
    // Store 3D Lorentzian fit statistics
    f3DLorentzianFitChi2red = chi2red;
    f3DLorentzianFitPp = pp;
    f3DLorentzianFitDOF = dof;
    f3DLorentzianFitSuccessful = fit_successful;
    
    // Store charge uncertainty (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        f3DLorentzianFitChargeUncertainty = charge_uncertainty;
    } else {
        f3DLorentzianFitChargeUncertainty = 0.0;
    }
    
    // Calculate delta values vs true position
    if (fit_successful && dof > 0) {
        f3DLorentzianDeltaX = center_x - fTrueX;      // x_3D_fit - x_true
        f3DLorentzianDeltaY = center_y - fTrueY;      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DLorentzianDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DLorentzianDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods (including 3D)
    CalculateMeanEstimations();
}

void RunAction::Set3DPowerLorentzianFitResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double beta, G4double amplitude, G4double vertical_offset,
                                               G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double beta_err, G4double amplitude_err, G4double vertical_offset_err,
                                               G4double chi2red, G4double pp, G4int dof,
                                               G4double charge_uncertainty,
                                               G4bool fit_successful)
{
    // Store 3D Power-Law Lorentzian fit parameters
    f3DPowerLorentzianFitCenterX = center_x;
    f3DPowerLorentzianFitCenterY = center_y;
    f3DPowerLorentzianFitGammaX = gamma_x;
    f3DPowerLorentzianFitGammaY = gamma_y;
    f3DPowerLorentzianFitBeta = beta;
    f3DPowerLorentzianFitAmplitude = amplitude;
    f3DPowerLorentzianFitVerticalOffset = vertical_offset;
    
    // Store 3D Power-Law Lorentzian fit parameter errors
    f3DPowerLorentzianFitCenterXErr = center_x_err;
    f3DPowerLorentzianFitCenterYErr = center_y_err;
    f3DPowerLorentzianFitGammaXErr = gamma_x_err;
    f3DPowerLorentzianFitGammaYErr = gamma_y_err;
    f3DPowerLorentzianFitBetaErr = beta_err;
    f3DPowerLorentzianFitAmplitudeErr = amplitude_err;
    f3DPowerLorentzianFitVerticalOffsetErr = vertical_offset_err;
    
    // Store 3D Power-Law Lorentzian fit statistics
    f3DPowerLorentzianFitChi2red = chi2red;
    f3DPowerLorentzianFitPp = pp;
    f3DPowerLorentzianFitDOF = dof;
    f3DPowerLorentzianFitSuccessful = fit_successful;
    
    // Store charge uncertainty (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        f3DPowerLorentzianFitChargeUncertainty = charge_uncertainty;
    } else {
        f3DPowerLorentzianFitChargeUncertainty = 0.0;
    }
    
    // Calculate delta values vs true position
    if (fit_successful && dof > 0) {
        f3DPowerLorentzianDeltaX = center_x - fTrueX;      // x_3D_fit - x_true
        f3DPowerLorentzianDeltaY = center_y - fTrueY;      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DPowerLorentzianDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DPowerLorentzianDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods (including 3D)
    CalculateMeanEstimations();
}

void RunAction::Set3DGaussianFitResults(G4double center_x, G4double center_y, G4double sigma_x, G4double sigma_y, G4double amplitude, G4double vertical_offset,
                                        G4double center_x_err, G4double center_y_err, G4double sigma_x_err, G4double sigma_y_err, G4double amplitude_err, G4double vertical_offset_err,
                                        G4double chi2red, G4double pp, G4int dof,
                                        G4double charge_uncertainty,
                                        G4bool fit_successful)
{
    // Store 3D Gaussian fit parameters
    f3DGaussianFitCenterX = center_x;
    f3DGaussianFitCenterY = center_y;
    f3DGaussianFitSigmaX = sigma_x;
    f3DGaussianFitSigmaY = sigma_y;
    f3DGaussianFitAmplitude = amplitude;
    f3DGaussianFitVerticalOffset = vertical_offset;
    
    // Store 3D Gaussian fit parameter errors
    f3DGaussianFitCenterXErr = center_x_err;
    f3DGaussianFitCenterYErr = center_y_err;
    f3DGaussianFitSigmaXErr = sigma_x_err;
    f3DGaussianFitSigmaYErr = sigma_y_err;
    f3DGaussianFitAmplitudeErr = amplitude_err;
    f3DGaussianFitVerticalOffsetErr = vertical_offset_err;
    
    // Store 3D Gaussian fit statistics
    f3DGaussianFitChi2red = chi2red;
    f3DGaussianFitPp = pp;
    f3DGaussianFitDOF = dof;
    f3DGaussianFitSuccessful = fit_successful;
    
    // Store charge uncertainty (5% of max charge) only if feature is enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        f3DGaussianFitChargeUncertainty = charge_uncertainty;
    } else {
        f3DGaussianFitChargeUncertainty = 0.0;
    }
    
    // Calculate delta values vs true position
    if (fit_successful && dof > 0) {
        f3DGaussianDeltaX = center_x - fTrueX;      // x_3D_fit - x_true
        f3DGaussianDeltaY = center_y - fTrueY;      // y_3D_fit - y_true
    } else {
        // Set delta values to NaN for failed fits
        f3DGaussianDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        f3DGaussianDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Calculate mean estimations from all fitting methods (including 3D)
    CalculateMeanEstimations();
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
        auto timeout = std::chrono::seconds(30); // 30 second timeout
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
        // Save detector grid parameters as metadata before writing the tree
        if (fGridPixelSize > 0) {
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
        }
        
        // Write tree and flush data
        fRootFile->cd();
        fTree->Write();
        fRootFile->Flush();
        
        G4cout << "RunAction: Successfully wrote ROOT file with " << fTree->GetEntries() << " entries" << G4endl;
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
            G4cout << "RunAction: Successfully cleaned up ROOT objects" << G4endl;
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
