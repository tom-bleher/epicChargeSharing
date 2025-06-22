#include "RunAction.hh"
#include "Constants.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TError.h" // Added for gErrorIgnoreLevel and kWarning
#include "TNamed.h"
#include "TString.h"
#include "TROOT.h"  // Added for ROOT dictionary loading
#include "TThread.h" // Added for thread initialization
#include <chrono>
#include <thread>
#include <cstdio> // For std::remove
#include <fstream>
#include <limits>
#include <mutex>
#include <memory>

// Initialize the static mutex
std::mutex RunAction::fRootMutex;

// Thread-safe ROOT initialization
static std::once_flag gRootInitFlag;

static void InitializeROOTThreading() {
    if (G4Threading::IsMultithreadedApplication()) {
        TThread::Initialize();
        gROOT->SetBatch(true); // Ensure batch mode for MT
    }
}

RunAction::RunAction()
: G4UserRunAction(),
  fRootFile(nullptr),
  fTree(nullptr),
  // Initialize HITS variables
  fTrueX(0),
  fTrueY(0),
  fTrueZ(0),
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
  // Legacy variables
  fPixelZ(0),
  fIsPixelHit(false),
  fInitialEnergy(0),
  fGridPixelSize(0),
  fGridPixelSpacing(0),
  fGridPixelCornerOffset(0),
  fGridDetSize(0),
  fGridNumBlocksPerSide(0)
{ 
  // Initialize neighborhood (9x9) grid vectors (they are automatically initialized empty)
  // Initialize step energy deposition vectors (they are automatically initialized empty)
}

RunAction::~RunAction()
{
  // File will be closed in EndOfRunAction
  // No need to delete here as it could cause double deletion
}

void RunAction::BeginOfRunAction(const G4Run*)
{ 
  // Initialize ROOT threading once per application
  std::call_once(gRootInitFlag, InitializeROOTThreading);
  
  // Lock mutex during ROOT file creation
  std::lock_guard<std::mutex> lock(fRootMutex);
  
  // Create ROOT file and tree with error handling
  try {
    // Create a unique filename based on the thread ID for worker threads
    G4String fileName = "epicChargeSharingOutput";
    
    // Add thread ID to filename for worker threads
    if (G4Threading::IsWorkerThread()) {
      std::ostringstream oss;
      oss << "_t" << G4Threading::G4GetThreadId();
      fileName += oss.str();
    }
    
    fileName += ".root";
    
    // Make sure any previous file is properly closed and deleted
    if (fRootFile) {
      if (fRootFile->IsOpen()) fRootFile->Close();
      delete fRootFile;
      fRootFile = nullptr;
      fTree = nullptr;
    }
    
    fRootFile = new TFile(fileName.c_str(), "RECREATE");
    
    if (!fRootFile || fRootFile->IsZombie()) {
      G4cerr << "Error creating ROOT file " << fileName << "!" << G4endl;
      delete fRootFile;
      fRootFile = nullptr;
      return;
    }
    
    fTree = new TTree("Hits", "Energy Deposits with Units");
    if (!fTree) {
      G4cerr << "Error creating ROOT tree!" << G4endl;
      delete fRootFile;
      fRootFile = nullptr;
      return;
    }
    
    // Create branches following the new hierarchical structure
    // =============================================
    // HITS BRANCHES
    // =============================================
    fTree->Branch("TrueX", &fTrueX, "TrueX/D")->SetTitle("True Position X [mm]");
    fTree->Branch("TrueY", &fTrueY, "TrueY/D")->SetTitle("True Position Y [mm]");
    fTree->Branch("TrueZ", &fTrueZ, "TrueZ/D")->SetTitle("True Position Z [mm]");
    fTree->Branch("InitX", &fInitX, "InitX/D")->SetTitle("Initial X [mm]");
    fTree->Branch("InitY", &fInitY, "InitY/D")->SetTitle("Initial Y [mm]");
    fTree->Branch("InitZ", &fInitZ, "InitZ/D")->SetTitle("Initial Z [mm]");
    fTree->Branch("PixelX", &fPixelX, "PixelX/D")->SetTitle("Nearest Pixel X [mm]");
    fTree->Branch("PixelY", &fPixelY, "PixelY/D")->SetTitle("Nearest Pixel Y [mm]");
    fTree->Branch("PixelZ", &fPixelZ, "PixelZ/D")->SetTitle("Nearest to hit pixel center Z [mm]");
    fTree->Branch("EdepAtDet", &fEdep, "Edep/D")->SetTitle("Energy Deposit [MeV]");
    fTree->Branch("InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
    fTree->Branch("IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("True if hit is on pixel OR distance <= D0");
    fTree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D")->SetTitle("Delta X from Pixel Center to True Position [mm] (x_pixel - x_true)");
    fTree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D")->SetTitle("Delta Y from Pixel Center to True Position [mm] (y_pixel - y_true)");
    
    // Create delta branches conditionally based on enabled fitting models
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
      fTree->Branch("GaussRowDeltaX", &fGaussRowDeltaX, "GaussRowDeltaX/D")->SetTitle("Delta X from Gaussian Row Fit to True Position [mm]");
      fTree->Branch("GaussColumnDeltaY", &fGaussColumnDeltaY, "GaussColumnDeltaY/D")->SetTitle("Delta Y from Gaussian Column Fit to True Position [mm]");
    }
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
      fTree->Branch("LorentzRowDeltaX", &fLorentzRowDeltaX, "LorentzRowDeltaX/D")->SetTitle("Delta X from Lorentzian Row Fit to True Position [mm]");
      fTree->Branch("LorentzColumnDeltaY", &fLorentzColumnDeltaY, "LorentzColumnDeltaY/D")->SetTitle("Delta Y from Lorentzian Column Fit to True Position [mm]");
    }

    // TRANSFORMED DIAGONAL COORDINATES BRANCHES (conditionally created)
    // Transformed coordinates from rotation matrix (θ=45° and θ=-45°)
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
      fTree->Branch("GaussMainDiagTransformedX", &fGaussMainDiagTransformedX, "GaussMainDiagTransformedX/D")->SetTitle("Transformed X from Gaussian Main Diagonal (rotation matrix) [mm]");
      fTree->Branch("GaussMainDiagTransformedY", &fGaussMainDiagTransformedY, "GaussMainDiagTransformedY/D")->SetTitle("Transformed Y from Gaussian Main Diagonal (rotation matrix) [mm]");
      fTree->Branch("GaussSecondDiagTransformedX", &fGaussSecondDiagTransformedX, "GaussSecondDiagTransformedX/D")->SetTitle("Transformed X from Gaussian Secondary Diagonal (rotation matrix) [mm]");
      fTree->Branch("GaussSecondDiagTransformedY", &fGaussSecondDiagTransformedY, "GaussSecondDiagTransformedY/D")->SetTitle("Transformed Y from Gaussian Secondary Diagonal (rotation matrix) [mm]");
    }
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
      fTree->Branch("LorentzMainDiagTransformedX", &fLorentzMainDiagTransformedX, "LorentzMainDiagTransformedX/D")->SetTitle("Transformed X from Lorentzian Main Diagonal (rotation matrix) [mm]");
      fTree->Branch("LorentzMainDiagTransformedY", &fLorentzMainDiagTransformedY, "LorentzMainDiagTransformedY/D")->SetTitle("Transformed Y from Lorentzian Main Diagonal (rotation matrix) [mm]");
      fTree->Branch("LorentzSecondDiagTransformedX", &fLorentzSecondDiagTransformedX, "LorentzSecondDiagTransformedX/D")->SetTitle("Transformed X from Lorentzian Secondary Diagonal (rotation matrix) [mm]");
      fTree->Branch("LorentzSecondDiagTransformedY", &fLorentzSecondDiagTransformedY, "LorentzSecondDiagTransformedY/D")->SetTitle("Transformed Y from Lorentzian Secondary Diagonal (rotation matrix) [mm]");
    }
    
    // Delta values for transformed coordinates vs true position (conditionally created)
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
      fTree->Branch("GaussMainDiagTransformedDeltaX", &fGaussMainDiagTransformedDeltaX, "GaussMainDiagTransformedDeltaX/D")->SetTitle("Delta X from Gaussian Main Diagonal Transformed to True Position [mm]");
      fTree->Branch("GaussMainDiagTransformedDeltaY", &fGaussMainDiagTransformedDeltaY, "GaussMainDiagTransformedDeltaY/D")->SetTitle("Delta Y from Gaussian Main Diagonal Transformed to True Position [mm]");
      fTree->Branch("GaussSecondDiagTransformedDeltaX", &fGaussSecondDiagTransformedDeltaX, "GaussSecondDiagTransformedDeltaX/D")->SetTitle("Delta X from Gaussian Secondary Diagonal Transformed to True Position [mm]");
      fTree->Branch("GaussSecondDiagTransformedDeltaY", &fGaussSecondDiagTransformedDeltaY, "GaussSecondDiagTransformedDeltaY/D")->SetTitle("Delta Y from Gaussian Secondary Diagonal Transformed to True Position [mm]");
    }
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
      fTree->Branch("LorentzMainDiagTransformedDeltaX", &fLorentzMainDiagTransformedDeltaX, "LorentzMainDiagTransformedDeltaX/D")->SetTitle("Delta X from Lorentzian Main Diagonal Transformed to True Position [mm]");
      fTree->Branch("LorentzMainDiagTransformedDeltaY", &fLorentzMainDiagTransformedDeltaY, "LorentzMainDiagTransformedDeltaY/D")->SetTitle("Delta Y from Lorentzian Main Diagonal Transformed to True Position [mm]");
      fTree->Branch("LorentzSecondDiagTransformedDeltaX", &fLorentzSecondDiagTransformedDeltaX, "LorentzSecondDiagTransformedDeltaX/D")->SetTitle("Delta X from Lorentzian Secondary Diagonal Transformed to True Position [mm]");
      fTree->Branch("LorentzSecondDiagTransformedDeltaY", &fLorentzSecondDiagTransformedDeltaY, "LorentzSecondDiagTransformedDeltaY/D")->SetTitle("Delta Y from Lorentzian Secondary Diagonal Transformed to True Position [mm]");
    }

    // MEAN ESTIMATION BRANCHES (conditionally created)
    // Mean delta values from all estimation methods
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
      fTree->Branch("GaussMeanTrueDeltaX", &fGaussMeanTrueDeltaX, "GaussMeanTrueDeltaX/D")->SetTitle("Mean Delta X from all Gaussian estimation methods to True Position [mm]");
      fTree->Branch("GaussMeanTrueDeltaY", &fGaussMeanTrueDeltaY, "GaussMeanTrueDeltaY/D")->SetTitle("Mean Delta Y from all Gaussian estimation methods to True Position [mm]");
    }
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
      fTree->Branch("LorentzMeanTrueDeltaX", &fLorentzMeanTrueDeltaX, "LorentzMeanTrueDeltaX/D")->SetTitle("Mean Delta X from all Lorentzian estimation methods to True Position [mm]");
      fTree->Branch("LorentzMeanTrueDeltaY", &fLorentzMeanTrueDeltaY, "LorentzMeanTrueDeltaY/D")->SetTitle("Mean Delta Y from all Lorentzian estimation methods to True Position [mm]");
    }

    // GRIDNEIGHBORHOOD BRANCHES
    // Grid neighborhood data for 9x9 neighborhood around hits
    fTree->Branch("GridNeighborhoodAngles", &fNonPixel_GridNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg]");
    fTree->Branch("GridNeighborhoodChargeFractions", &fNonPixel_GridNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels");
    fTree->Branch("GridNeighborhoodDistances", &fNonPixel_GridNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm]");
    fTree->Branch("GridNeighborhoodCharges", &fNonPixel_GridNeighborhoodCharge)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels");
    
    // =============================================
    // AUTOMATIC RADIUS SELECTION BRANCHES
    // =============================================
    fTree->Branch("SelectedRadius", &fSelectedRadius, "SelectedRadius/I")->SetTitle("Automatically Selected Neighborhood Radius");
    
    // =============================================
    // GAUSSIAN FITS BRANCHES (conditionally created)
    // =============================================
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
    // GaussFitRow/GaussFitRowX
    fTree->Branch("GaussFitRowAmplitude", &fGaussFitRowAmplitude, "GaussFitRowAmplitude/D")->SetTitle("Gaussian Row Fit Amplitude");
    fTree->Branch("GaussFitRowAmplitudeErr", &fGaussFitRowAmplitudeErr, "GaussFitRowAmplitudeErr/D")->SetTitle("Gaussian Row Fit Amplitude Error");
    fTree->Branch("GaussFitRowStdev", &fGaussFitRowStdev, "GaussFitRowStdev/D")->SetTitle("Gaussian Row Fit Standard Deviation");
    fTree->Branch("GaussFitRowStdevErr", &fGaussFitRowStdevErr, "GaussFitRowStdevErr/D")->SetTitle("Gaussian Row Fit Standard Deviation Error");
    fTree->Branch("GaussFitRowVerticalOffset", &fGaussFitRowVerticalOffset, "GaussFitRowVerticalOffset/D")->SetTitle("Gaussian Row Fit Vertical Offset");
    fTree->Branch("GaussFitRowVerticalOffsetErr", &fGaussFitRowVerticalOffsetErr, "GaussFitRowVerticalOffsetErr/D")->SetTitle("Gaussian Row Fit Vertical Offset Error");
    fTree->Branch("GaussFitRowCenter", &fGaussFitRowCenter, "GaussFitRowCenter/D")->SetTitle("Gaussian Row Fit Center [mm]");
    fTree->Branch("GaussFitRowCenterErr", &fGaussFitRowCenterErr, "GaussFitRowCenterErr/D")->SetTitle("Gaussian Row Fit Center Error [mm]");
    fTree->Branch("GaussFitRowChi2red", &fGaussFitRowChi2red, "GaussFitRowChi2red/D")->SetTitle("Gaussian Row Fit Reduced Chi-squared");
    fTree->Branch("GaussFitRowPp", &fGaussFitRowPp, "GaussFitRowPp/D")->SetTitle("Gaussian Row Fit P-value");
    fTree->Branch("GaussFitRowDOF", &fGaussFitRowDOF, "GaussFitRowDOF/I")->SetTitle("Gaussian Row Fit Degrees of Freedom");
    // Conditionally create charge uncertainty branch for Gaussian row fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("GaussFitRowChargeUncertainty", &fGaussFitRowChargeUncertainty, "GaussFitRowChargeUncertainty/D")->SetTitle("Gaussian Row Fit Charge Uncertainty");
    }
    
    // GaussFitColumn/GaussFitColumnY
    fTree->Branch("GaussFitColumnAmplitude", &fGaussFitColumnAmplitude, "GaussFitColumnAmplitude/D")->SetTitle("Gaussian Column Fit Amplitude");
    fTree->Branch("GaussFitColumnAmplitudeErr", &fGaussFitColumnAmplitudeErr, "GaussFitColumnAmplitudeErr/D")->SetTitle("Gaussian Column Fit Amplitude Error");
    fTree->Branch("GaussFitColumnStdev", &fGaussFitColumnStdev, "GaussFitColumnStdev/D")->SetTitle("Gaussian Column Fit Standard Deviation");
    fTree->Branch("GaussFitColumnStdevErr", &fGaussFitColumnStdevErr, "GaussFitColumnStdevErr/D")->SetTitle("Gaussian Column Fit Standard Deviation Error");
    fTree->Branch("GaussFitColumnVerticalOffset", &fGaussFitColumnVerticalOffset, "GaussFitColumnVerticalOffset/D")->SetTitle("Gaussian Column Fit Vertical Offset");
    fTree->Branch("GaussFitColumnVerticalOffsetErr", &fGaussFitColumnVerticalOffsetErr, "GaussFitColumnVerticalOffsetErr/D")->SetTitle("Gaussian Column Fit Vertical Offset Error");
    fTree->Branch("GaussFitColumnCenter", &fGaussFitColumnCenter, "GaussFitColumnCenter/D")->SetTitle("Gaussian Column Fit Center [mm]");
    fTree->Branch("GaussFitColumnCenterErr", &fGaussFitColumnCenterErr, "GaussFitColumnCenterErr/D")->SetTitle("Gaussian Column Fit Center Error [mm]");
    fTree->Branch("GaussFitColumnChi2red", &fGaussFitColumnChi2red, "GaussFitColumnChi2red/D")->SetTitle("Gaussian Column Fit Reduced Chi-squared");
    fTree->Branch("GaussFitColumnPp", &fGaussFitColumnPp, "GaussFitColumnPp/D")->SetTitle("Gaussian Column Fit P-value");
    fTree->Branch("GaussFitColumnDOF", &fGaussFitColumnDOF, "GaussFitColumnDOF/I")->SetTitle("Gaussian Column Fit Degrees of Freedom");
    // Conditionally create charge uncertainty branch for Gaussian column fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("GaussFitColumnChargeUncertainty", &fGaussFitColumnChargeUncertainty, "GaussFitColumnChargeUncertainty/D")->SetTitle("Gaussian Column Fit Charge Uncertainty");
    }
    
    // GaussFitMainDiag/GaussFitMainDiagX
    fTree->Branch("GaussFitMainDiagXAmplitude", &fGaussFitMainDiagXAmplitude, "GaussFitMainDiagXAmplitude/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude");
    fTree->Branch("GaussFitMainDiagXAmplitudeErr", &fGaussFitMainDiagXAmplitudeErr, "GaussFitMainDiagXAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude Error");
    fTree->Branch("GaussFitMainDiagXStdev", &fGaussFitMainDiagXStdev, "GaussFitMainDiagXStdev/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation");
    fTree->Branch("GaussFitMainDiagXStdevErr", &fGaussFitMainDiagXStdevErr, "GaussFitMainDiagXStdevErr/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation Error");
    fTree->Branch("GaussFitMainDiagXVerticalOffset", &fGaussFitMainDiagXVerticalOffset, "GaussFitMainDiagXVerticalOffset/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset");
    fTree->Branch("GaussFitMainDiagXVerticalOffsetErr", &fGaussFitMainDiagXVerticalOffsetErr, "GaussFitMainDiagXVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset Error");
    fTree->Branch("GaussFitMainDiagXCenter", &fGaussFitMainDiagXCenter, "GaussFitMainDiagXCenter/D")->SetTitle("Gaussian Main Diagonal X Fit Center [mm]");
    fTree->Branch("GaussFitMainDiagXCenterErr", &fGaussFitMainDiagXCenterErr, "GaussFitMainDiagXCenterErr/D")->SetTitle("Gaussian Main Diagonal X Fit Center Error [mm]");
    fTree->Branch("GaussFitMainDiagXChi2red", &fGaussFitMainDiagXChi2red, "GaussFitMainDiagXChi2red/D")->SetTitle("Gaussian Main Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("GaussFitMainDiagXPp", &fGaussFitMainDiagXPp, "GaussFitMainDiagXPp/D")->SetTitle("Gaussian Main Diagonal X Fit P-value");
    fTree->Branch("GaussFitMainDiagXDOF", &fGaussFitMainDiagXDOF, "GaussFitMainDiagXDOF/I")->SetTitle("Gaussian Main Diagonal X Fit Degrees of Freedom");
    
    // GaussFitMainDiag/GaussFitMainDiagY
    fTree->Branch("GaussFitMainDiagYAmplitude", &fGaussFitMainDiagYAmplitude, "GaussFitMainDiagYAmplitude/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude");
    fTree->Branch("GaussFitMainDiagYAmplitudeErr", &fGaussFitMainDiagYAmplitudeErr, "GaussFitMainDiagYAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude Error");
    fTree->Branch("GaussFitMainDiagYStdev", &fGaussFitMainDiagYStdev, "GaussFitMainDiagYStdev/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation");
    fTree->Branch("GaussFitMainDiagYStdevErr", &fGaussFitMainDiagYStdevErr, "GaussFitMainDiagYStdevErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation Error");
    fTree->Branch("GaussFitMainDiagYVerticalOffset", &fGaussFitMainDiagYVerticalOffset, "GaussFitMainDiagYVerticalOffset/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset");
    fTree->Branch("GaussFitMainDiagYVerticalOffsetErr", &fGaussFitMainDiagYVerticalOffsetErr, "GaussFitMainDiagYVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("GaussFitMainDiagYCenter", &fGaussFitMainDiagYCenter, "GaussFitMainDiagYCenter/D")->SetTitle("Gaussian Main Diagonal Y Fit Center [mm]");
    fTree->Branch("GaussFitMainDiagYCenterErr", &fGaussFitMainDiagYCenterErr, "GaussFitMainDiagYCenterErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Center Error [mm]");
    fTree->Branch("GaussFitMainDiagYChi2red", &fGaussFitMainDiagYChi2red, "GaussFitMainDiagYChi2red/D")->SetTitle("Gaussian Main Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("GaussFitMainDiagYPp", &fGaussFitMainDiagYPp, "GaussFitMainDiagYPp/D")->SetTitle("Gaussian Main Diagonal Y Fit P-value");
    fTree->Branch("GaussFitMainDiagYDOF", &fGaussFitMainDiagYDOF, "GaussFitMainDiagYDOF/I")->SetTitle("Gaussian Main Diagonal Y Fit Degrees of Freedom");
    
    // GaussFitSecondDiag/GaussFitSecondDiagX
    fTree->Branch("GaussFitSecondDiagXAmplitude", &fGaussFitSecondDiagXAmplitude, "GaussFitSecondDiagXAmplitude/D")->SetTitle("Gaussian Second Diagonal X Fit Amplitude");
    fTree->Branch("GaussFitSecondDiagXAmplitudeErr", &fGaussFitSecondDiagXAmplitudeErr, "GaussFitSecondDiagXAmplitudeErr/D")->SetTitle("Gaussian Second Diagonal X Fit Amplitude Error");
    fTree->Branch("GaussFitSecondDiagXStdev", &fGaussFitSecondDiagXStdev, "GaussFitSecondDiagXStdev/D")->SetTitle("Gaussian Second Diagonal X Fit Standard Deviation");
    fTree->Branch("GaussFitSecondDiagXStdevErr", &fGaussFitSecondDiagXStdevErr, "GaussFitSecondDiagXStdevErr/D")->SetTitle("Gaussian Second Diagonal X Fit Standard Deviation Error");
    fTree->Branch("GaussFitSecondDiagXVerticalOffset", &fGaussFitSecondDiagXVerticalOffset, "GaussFitSecondDiagXVerticalOffset/D")->SetTitle("Gaussian Second Diagonal X Fit Vertical Offset");
    fTree->Branch("GaussFitSecondDiagXVerticalOffsetErr", &fGaussFitSecondDiagXVerticalOffsetErr, "GaussFitSecondDiagXVerticalOffsetErr/D")->SetTitle("Gaussian Second Diagonal X Fit Vertical Offset Error");
    fTree->Branch("GaussFitSecondDiagXCenter", &fGaussFitSecondDiagXCenter, "GaussFitSecondDiagXCenter/D")->SetTitle("Gaussian Second Diagonal X Fit Center [mm]");
    fTree->Branch("GaussFitSecondDiagXCenterErr", &fGaussFitSecondDiagXCenterErr, "GaussFitSecondDiagXCenterErr/D")->SetTitle("Gaussian Second Diagonal X Fit Center Error [mm]");
    fTree->Branch("GaussFitSecondDiagXChi2red", &fGaussFitSecondDiagXChi2red, "GaussFitSecondDiagXChi2red/D")->SetTitle("Gaussian Second Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("GaussFitSecondDiagXPp", &fGaussFitSecondDiagXPp, "GaussFitSecondDiagXPp/D")->SetTitle("Gaussian Second Diagonal X Fit P-value");
    fTree->Branch("GaussFitSecondDiagXDOF", &fGaussFitSecondDiagXDOF, "GaussFitSecondDiagXDOF/I")->SetTitle("Gaussian Second Diagonal X Fit Degrees of Freedom");
    
    // GaussFitSecondDiag/GaussFitSecondDiagY
    fTree->Branch("GaussFitSecondDiagYAmplitude", &fGaussFitSecondDiagYAmplitude, "GaussFitSecondDiagYAmplitude/D")->SetTitle("Gaussian Second Diagonal Y Fit Amplitude");
    fTree->Branch("GaussFitSecondDiagYAmplitudeErr", &fGaussFitSecondDiagYAmplitudeErr, "GaussFitSecondDiagYAmplitudeErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Amplitude Error");
    fTree->Branch("GaussFitSecondDiagYStdev", &fGaussFitSecondDiagYStdev, "GaussFitSecondDiagYStdev/D")->SetTitle("Gaussian Second Diagonal Y Fit Standard Deviation");
    fTree->Branch("GaussFitSecondDiagYStdevErr", &fGaussFitSecondDiagYStdevErr, "GaussFitSecondDiagYStdevErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Standard Deviation Error");
    fTree->Branch("GaussFitSecondDiagYVerticalOffset", &fGaussFitSecondDiagYVerticalOffset, "GaussFitSecondDiagYVerticalOffset/D")->SetTitle("Gaussian Second Diagonal Y Fit Vertical Offset");
    fTree->Branch("GaussFitSecondDiagYVerticalOffsetErr", &fGaussFitSecondDiagYVerticalOffsetErr, "GaussFitSecondDiagYVerticalOffsetErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("GaussFitSecondDiagYCenter", &fGaussFitSecondDiagYCenter, "GaussFitSecondDiagYCenter/D")->SetTitle("Gaussian Second Diagonal Y Fit Center [mm]");
    fTree->Branch("GaussFitSecondDiagYCenterErr", &fGaussFitSecondDiagYCenterErr, "GaussFitSecondDiagYCenterErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Center Error [mm]");
    fTree->Branch("GaussFitSecondDiagYChi2red", &fGaussFitSecondDiagYChi2red, "GaussFitSecondDiagYChi2red/D")->SetTitle("Gaussian Second Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("GaussFitSecondDiagYPp", &fGaussFitSecondDiagYPp, "GaussFitSecondDiagYPp/D")->SetTitle("Gaussian Second Diagonal Y Fit P-value");
    fTree->Branch("GaussFitSecondDiagYDOF", &fGaussFitSecondDiagYDOF, "GaussFitSecondDiagYDOF/I")->SetTitle("Gaussian Second Diagonal Y Fit Degrees of Freedom");
    
    } // End of Gaussian fitting branches
    
    // =============================================
    // LORENTZIAN FITS BRANCHES (conditionally created)
    // =============================================
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
    // LorentzFitRow/LorentzFitRowX
    fTree->Branch("LorentzFitRowAmplitude", &fLorentzFitRowAmplitude, "LorentzFitRowAmplitude/D")->SetTitle("Lorentzian Row Fit Amplitude");
    fTree->Branch("LorentzFitRowAmplitudeErr", &fLorentzFitRowAmplitudeErr, "LorentzFitRowAmplitudeErr/D")->SetTitle("Lorentzian Row Fit Amplitude Error");
    fTree->Branch("LorentzFitRowGamma", &fLorentzFitRowGamma, "LorentzFitRowGamma/D")->SetTitle("Lorentzian Row Fit Gamma Parameter");
    fTree->Branch("LorentzFitRowGammaErr", &fLorentzFitRowGammaErr, "LorentzFitRowGammaErr/D")->SetTitle("Lorentzian Row Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitRowVerticalOffset", &fLorentzFitRowVerticalOffset, "LorentzFitRowVerticalOffset/D")->SetTitle("Lorentzian Row Fit Vertical Offset");
    fTree->Branch("LorentzFitRowVerticalOffsetErr", &fLorentzFitRowVerticalOffsetErr, "LorentzFitRowVerticalOffsetErr/D")->SetTitle("Lorentzian Row Fit Vertical Offset Error");
    fTree->Branch("LorentzFitRowCenter", &fLorentzFitRowCenter, "LorentzFitRowCenter/D")->SetTitle("Lorentzian Row Fit Center [mm]");
    fTree->Branch("LorentzFitRowCenterErr", &fLorentzFitRowCenterErr, "LorentzFitRowCenterErr/D")->SetTitle("Lorentzian Row Fit Center Error [mm]");
    fTree->Branch("LorentzFitRowChi2red", &fLorentzFitRowChi2red, "LorentzFitRowChi2red/D")->SetTitle("Lorentzian Row Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitRowPp", &fLorentzFitRowPp, "LorentzFitRowPp/D")->SetTitle("Lorentzian Row Fit P-value");
    fTree->Branch("LorentzFitRowDOF", &fLorentzFitRowDOF, "LorentzFitRowDOF/I")->SetTitle("Lorentzian Row Fit Degrees of Freedom");
    // Conditionally create charge uncertainty branch for Lorentzian row fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("LorentzFitRowChargeUncertainty", &fLorentzFitRowChargeUncertainty, "LorentzFitRowChargeUncertainty/D")->SetTitle("Lorentzian Row Fit Charge Uncertainty");
    }

    // LorentzFitColumn/LorentzFitColumnY
    fTree->Branch("LorentzFitColumnAmplitude", &fLorentzFitColumnAmplitude, "LorentzFitColumnAmplitude/D")->SetTitle("Lorentzian Column Fit Amplitude");
    fTree->Branch("LorentzFitColumnAmplitudeErr", &fLorentzFitColumnAmplitudeErr, "LorentzFitColumnAmplitudeErr/D")->SetTitle("Lorentzian Column Fit Amplitude Error");
    fTree->Branch("LorentzFitColumnGamma", &fLorentzFitColumnGamma, "LorentzFitColumnGamma/D")->SetTitle("Lorentzian Column Fit Gamma Parameter");
    fTree->Branch("LorentzFitColumnGammaErr", &fLorentzFitColumnGammaErr, "LorentzFitColumnGammaErr/D")->SetTitle("Lorentzian Column Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitColumnVerticalOffset", &fLorentzFitColumnVerticalOffset, "LorentzFitColumnVerticalOffset/D")->SetTitle("Lorentzian Column Fit Vertical Offset");
    fTree->Branch("LorentzFitColumnVerticalOffsetErr", &fLorentzFitColumnVerticalOffsetErr, "LorentzFitColumnVerticalOffsetErr/D")->SetTitle("Lorentzian Column Fit Vertical Offset Error");
    fTree->Branch("LorentzFitColumnCenter", &fLorentzFitColumnCenter, "LorentzFitColumnCenter/D")->SetTitle("Lorentzian Column Fit Center [mm]");
    fTree->Branch("LorentzFitColumnCenterErr", &fLorentzFitColumnCenterErr, "LorentzFitColumnCenterErr/D")->SetTitle("Lorentzian Column Fit Center Error [mm]");
    fTree->Branch("LorentzFitColumnChi2red", &fLorentzFitColumnChi2red, "LorentzFitColumnChi2red/D")->SetTitle("Lorentzian Column Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitColumnPp", &fLorentzFitColumnPp, "LorentzFitColumnPp/D")->SetTitle("Lorentzian Column Fit P-value");
    fTree->Branch("LorentzFitColumnDOF", &fLorentzFitColumnDOF, "LorentzFitColumnDOF/I")->SetTitle("Lorentzian Column Fit Degrees of Freedom");
    // Conditionally create charge uncertainty branch for Lorentzian column fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("LorentzFitColumnChargeUncertainty", &fLorentzFitColumnChargeUncertainty, "LorentzFitColumnChargeUncertainty/D")->SetTitle("Lorentzian Column Fit Charge Uncertainty");
    }

    // LorentzFitMainDiag/LorentzFitMainDiagX
    fTree->Branch("LorentzFitMainDiagXAmplitude", &fLorentzFitMainDiagXAmplitude, "LorentzFitMainDiagXAmplitude/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude");
    fTree->Branch("LorentzFitMainDiagXAmplitudeErr", &fLorentzFitMainDiagXAmplitudeErr, "LorentzFitMainDiagXAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude Error");
    fTree->Branch("LorentzFitMainDiagXGamma", &fLorentzFitMainDiagXGamma, "LorentzFitMainDiagXGamma/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma Parameter");
    fTree->Branch("LorentzFitMainDiagXGammaErr", &fLorentzFitMainDiagXGammaErr, "LorentzFitMainDiagXGammaErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitMainDiagXVerticalOffset", &fLorentzFitMainDiagXVerticalOffset, "LorentzFitMainDiagXVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset");
    fTree->Branch("LorentzFitMainDiagXVerticalOffsetErr", &fLorentzFitMainDiagXVerticalOffsetErr, "LorentzFitMainDiagXVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset Error");
    fTree->Branch("LorentzFitMainDiagXCenter", &fLorentzFitMainDiagXCenter, "LorentzFitMainDiagXCenter/D")->SetTitle("Lorentzian Main Diagonal X Fit Center [mm]");
    fTree->Branch("LorentzFitMainDiagXCenterErr", &fLorentzFitMainDiagXCenterErr, "LorentzFitMainDiagXCenterErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Center Error [mm]");
    fTree->Branch("LorentzFitMainDiagXChi2red", &fLorentzFitMainDiagXChi2red, "LorentzFitMainDiagXChi2red/D")->SetTitle("Lorentzian Main Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitMainDiagXPp", &fLorentzFitMainDiagXPp, "LorentzFitMainDiagXPp/D")->SetTitle("Lorentzian Main Diagonal X Fit P-value");
    fTree->Branch("LorentzFitMainDiagXDOF", &fLorentzFitMainDiagXDOF, "LorentzFitMainDiagXDOF/I")->SetTitle("Lorentzian Main Diagonal X Fit Degrees of Freedom");

    // LorentzFitMainDiag/LorentzFitMainDiagY
    fTree->Branch("LorentzFitMainDiagYAmplitude", &fLorentzFitMainDiagYAmplitude, "LorentzFitMainDiagYAmplitude/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude");
    fTree->Branch("LorentzFitMainDiagYAmplitudeErr", &fLorentzFitMainDiagYAmplitudeErr, "LorentzFitMainDiagYAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude Error");
    fTree->Branch("LorentzFitMainDiagYGamma", &fLorentzFitMainDiagYGamma, "LorentzFitMainDiagYGamma/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma Parameter");
    fTree->Branch("LorentzFitMainDiagYGammaErr", &fLorentzFitMainDiagYGammaErr, "LorentzFitMainDiagYGammaErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitMainDiagYVerticalOffset", &fLorentzFitMainDiagYVerticalOffset, "LorentzFitMainDiagYVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset");
    fTree->Branch("LorentzFitMainDiagYVerticalOffsetErr", &fLorentzFitMainDiagYVerticalOffsetErr, "LorentzFitMainDiagYVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("LorentzFitMainDiagYCenter", &fLorentzFitMainDiagYCenter, "LorentzFitMainDiagYCenter/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center [mm]");
    fTree->Branch("LorentzFitMainDiagYCenterErr", &fLorentzFitMainDiagYCenterErr, "LorentzFitMainDiagYCenterErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center Error [mm]");
    fTree->Branch("LorentzFitMainDiagYChi2red", &fLorentzFitMainDiagYChi2red, "LorentzFitMainDiagYChi2red/D")->SetTitle("Lorentzian Main Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitMainDiagYPp", &fLorentzFitMainDiagYPp, "LorentzFitMainDiagYPp/D")->SetTitle("Lorentzian Main Diagonal Y Fit P-value");
    fTree->Branch("LorentzFitMainDiagYDOF", &fLorentzFitMainDiagYDOF, "LorentzFitMainDiagYDOF/I")->SetTitle("Lorentzian Main Diagonal Y Fit Degrees of Freedom");

    // LorentzFitSecondDiag/LorentzFitSecondDiagX
    fTree->Branch("LorentzFitSecondDiagXAmplitude", &fLorentzFitSecondDiagXAmplitude, "LorentzFitSecondDiagXAmplitude/D")->SetTitle("Lorentzian Second Diagonal X Fit Amplitude");
    fTree->Branch("LorentzFitSecondDiagXAmplitudeErr", &fLorentzFitSecondDiagXAmplitudeErr, "LorentzFitSecondDiagXAmplitudeErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Amplitude Error");
    fTree->Branch("LorentzFitSecondDiagXGamma", &fLorentzFitSecondDiagXGamma, "LorentzFitSecondDiagXGamma/D")->SetTitle("Lorentzian Second Diagonal X Fit Gamma Parameter");
    fTree->Branch("LorentzFitSecondDiagXGammaErr", &fLorentzFitSecondDiagXGammaErr, "LorentzFitSecondDiagXGammaErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitSecondDiagXVerticalOffset", &fLorentzFitSecondDiagXVerticalOffset, "LorentzFitSecondDiagXVerticalOffset/D")->SetTitle("Lorentzian Second Diagonal X Fit Vertical Offset");
    fTree->Branch("LorentzFitSecondDiagXVerticalOffsetErr", &fLorentzFitSecondDiagXVerticalOffsetErr, "LorentzFitSecondDiagXVerticalOffsetErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Vertical Offset Error");
    fTree->Branch("LorentzFitSecondDiagXCenter", &fLorentzFitSecondDiagXCenter, "LorentzFitSecondDiagXCenter/D")->SetTitle("Lorentzian Second Diagonal X Fit Center [mm]");
    fTree->Branch("LorentzFitSecondDiagXCenterErr", &fLorentzFitSecondDiagXCenterErr, "LorentzFitSecondDiagXCenterErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Center Error [mm]");
    fTree->Branch("LorentzFitSecondDiagXChi2red", &fLorentzFitSecondDiagXChi2red, "LorentzFitSecondDiagXChi2red/D")->SetTitle("Lorentzian Second Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitSecondDiagXPp", &fLorentzFitSecondDiagXPp, "LorentzFitSecondDiagXPp/D")->SetTitle("Lorentzian Second Diagonal X Fit P-value");
    fTree->Branch("LorentzFitSecondDiagXDOF", &fLorentzFitSecondDiagXDOF, "LorentzFitSecondDiagXDOF/I")->SetTitle("Lorentzian Second Diagonal X Fit Degrees of Freedom");

    // LorentzFitSecondDiag/LorentzFitSecondDiagY
    fTree->Branch("LorentzFitSecondDiagYAmplitude", &fLorentzFitSecondDiagYAmplitude, "LorentzFitSecondDiagYAmplitude/D")->SetTitle("Lorentzian Second Diagonal Y Fit Amplitude");
    fTree->Branch("LorentzFitSecondDiagYAmplitudeErr", &fLorentzFitSecondDiagYAmplitudeErr, "LorentzFitSecondDiagYAmplitudeErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Amplitude Error");
    fTree->Branch("LorentzFitSecondDiagYGamma", &fLorentzFitSecondDiagYGamma, "LorentzFitSecondDiagYGamma/D")->SetTitle("Lorentzian Second Diagonal Y Fit Gamma Parameter");
    fTree->Branch("LorentzFitSecondDiagYGammaErr", &fLorentzFitSecondDiagYGammaErr, "LorentzFitSecondDiagYGammaErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Gamma Parameter Error");
    fTree->Branch("LorentzFitSecondDiagYVerticalOffset", &fLorentzFitSecondDiagYVerticalOffset, "LorentzFitSecondDiagYVerticalOffset/D")->SetTitle("Lorentzian Second Diagonal Y Fit Vertical Offset");
    fTree->Branch("LorentzFitSecondDiagYVerticalOffsetErr", &fLorentzFitSecondDiagYVerticalOffsetErr, "LorentzFitSecondDiagYVerticalOffsetErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("LorentzFitSecondDiagYCenter", &fLorentzFitSecondDiagYCenter, "LorentzFitSecondDiagYCenter/D")->SetTitle("Lorentzian Second Diagonal Y Fit Center [mm]");
    fTree->Branch("LorentzFitSecondDiagYCenterErr", &fLorentzFitSecondDiagYCenterErr, "LorentzFitSecondDiagYCenterErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Center Error [mm]");
    fTree->Branch("LorentzFitSecondDiagYChi2red", &fLorentzFitSecondDiagYChi2red, "LorentzFitSecondDiagYChi2red/D")->SetTitle("Lorentzian Second Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("LorentzFitSecondDiagYPp", &fLorentzFitSecondDiagYPp, "LorentzFitSecondDiagYPp/D")->SetTitle("Lorentzian Second Diagonal Y Fit P-value");
    fTree->Branch("LorentzFitSecondDiagYDOF", &fLorentzFitSecondDiagYDOF, "LorentzFitSecondDiagYDOF/I")->SetTitle("Lorentzian Second Diagonal Y Fit Degrees of Freedom");

    } // End of Lorentzian fitting branches

    // =============================================
    // POWER-LAW LORENTZIAN FITS BRANCHES (conditionally created)
    // =============================================
    if (Constants::ENABLE_POWER_LORENTZIAN_FITTING) {
    // PowerLorentzFitRow/PowerLorentzFitRowX
    fTree->Branch("PowerLorentzFitRowAmplitude", &fPowerLorentzFitRowAmplitude, "PowerLorentzFitRowAmplitude/D")->SetTitle("Power-Law Lorentzian Row Fit Amplitude");
    fTree->Branch("PowerLorentzFitRowAmplitudeErr", &fPowerLorentzFitRowAmplitudeErr, "PowerLorentzFitRowAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Row Fit Amplitude Error");
    fTree->Branch("PowerLorentzFitRowBeta", &fPowerLorentzFitRowBeta, "PowerLorentzFitRowBeta/D")->SetTitle("Power-Law Lorentzian Row Fit Beta Parameter");
    fTree->Branch("PowerLorentzFitRowBetaErr", &fPowerLorentzFitRowBetaErr, "PowerLorentzFitRowBetaErr/D")->SetTitle("Power-Law Lorentzian Row Fit Beta Parameter Error");
    fTree->Branch("PowerLorentzFitRowGamma", &fPowerLorentzFitRowGamma, "PowerLorentzFitRowGamma/D")->SetTitle("Power-Law Lorentzian Row Fit Gamma Parameter");
    fTree->Branch("PowerLorentzFitRowGammaErr", &fPowerLorentzFitRowGammaErr, "PowerLorentzFitRowGammaErr/D")->SetTitle("Power-Law Lorentzian Row Fit Gamma Parameter Error");
    fTree->Branch("PowerLorentzFitRowVerticalOffset", &fPowerLorentzFitRowVerticalOffset, "PowerLorentzFitRowVerticalOffset/D")->SetTitle("Power-Law Lorentzian Row Fit Vertical Offset");
    fTree->Branch("PowerLorentzFitRowVerticalOffsetErr", &fPowerLorentzFitRowVerticalOffsetErr, "PowerLorentzFitRowVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Row Fit Vertical Offset Error");
    fTree->Branch("PowerLorentzFitRowCenter", &fPowerLorentzFitRowCenter, "PowerLorentzFitRowCenter/D")->SetTitle("Power-Law Lorentzian Row Fit Center [mm]");
    fTree->Branch("PowerLorentzFitRowCenterErr", &fPowerLorentzFitRowCenterErr, "PowerLorentzFitRowCenterErr/D")->SetTitle("Power-Law Lorentzian Row Fit Center Error [mm]");
    fTree->Branch("PowerLorentzFitRowChi2red", &fPowerLorentzFitRowChi2red, "PowerLorentzFitRowChi2red/D")->SetTitle("Power-Law Lorentzian Row Fit Reduced Chi-squared");
    fTree->Branch("PowerLorentzFitRowPp", &fPowerLorentzFitRowPp, "PowerLorentzFitRowPp/D")->SetTitle("Power-Law Lorentzian Row Fit P-value");
    fTree->Branch("PowerLorentzFitRowDOF", &fPowerLorentzFitRowDOF, "PowerLorentzFitRowDOF/I")->SetTitle("Power-Law Lorentzian Row Fit Degrees of Freedom");

    // PowerLorentzFitColumn/PowerLorentzFitColumnY
    fTree->Branch("PowerLorentzFitColumnAmplitude", &fPowerLorentzFitColumnAmplitude, "PowerLorentzFitColumnAmplitude/D")->SetTitle("Power-Law Lorentzian Column Fit Amplitude");
    fTree->Branch("PowerLorentzFitColumnAmplitudeErr", &fPowerLorentzFitColumnAmplitudeErr, "PowerLorentzFitColumnAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Column Fit Amplitude Error");
    fTree->Branch("PowerLorentzFitColumnBeta", &fPowerLorentzFitColumnBeta, "PowerLorentzFitColumnBeta/D")->SetTitle("Power-Law Lorentzian Column Fit Beta Parameter");
    fTree->Branch("PowerLorentzFitColumnBetaErr", &fPowerLorentzFitColumnBetaErr, "PowerLorentzFitColumnBetaErr/D")->SetTitle("Power-Law Lorentzian Column Fit Beta Parameter Error");
    fTree->Branch("PowerLorentzFitColumnGamma", &fPowerLorentzFitColumnGamma, "PowerLorentzFitColumnGamma/D")->SetTitle("Power-Law Lorentzian Column Fit Gamma Parameter");
    fTree->Branch("PowerLorentzFitColumnGammaErr", &fPowerLorentzFitColumnGammaErr, "PowerLorentzFitColumnGammaErr/D")->SetTitle("Power-Law Lorentzian Column Fit Gamma Parameter Error");
    fTree->Branch("PowerLorentzFitColumnVerticalOffset", &fPowerLorentzFitColumnVerticalOffset, "PowerLorentzFitColumnVerticalOffset/D")->SetTitle("Power-Law Lorentzian Column Fit Vertical Offset");
    fTree->Branch("PowerLorentzFitColumnVerticalOffsetErr", &fPowerLorentzFitColumnVerticalOffsetErr, "PowerLorentzFitColumnVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Column Fit Vertical Offset Error");
    fTree->Branch("PowerLorentzFitColumnCenter", &fPowerLorentzFitColumnCenter, "PowerLorentzFitColumnCenter/D")->SetTitle("Power-Law Lorentzian Column Fit Center [mm]");
    fTree->Branch("PowerLorentzFitColumnCenterErr", &fPowerLorentzFitColumnCenterErr, "PowerLorentzFitColumnCenterErr/D")->SetTitle("Power-Law Lorentzian Column Fit Center Error [mm]");
    fTree->Branch("PowerLorentzFitColumnChi2red", &fPowerLorentzFitColumnChi2red, "PowerLorentzFitColumnChi2red/D")->SetTitle("Power-Law Lorentzian Column Fit Reduced Chi-squared");
    fTree->Branch("PowerLorentzFitColumnPp", &fPowerLorentzFitColumnPp, "PowerLorentzFitColumnPp/D")->SetTitle("Power-Law Lorentzian Column Fit P-value");
    fTree->Branch("PowerLorentzFitColumnDOF", &fPowerLorentzFitColumnDOF, "PowerLorentzFitColumnDOF/I")->SetTitle("Power-Law Lorentzian Column Fit Degrees of Freedom");

    // PowerLorentzFitMainDiag/PowerLorentzFitMainDiagX
    fTree->Branch("PowerLorentzFitMainDiagXAmplitude", &fPowerLorentzFitMainDiagXAmplitude, "PowerLorentzFitMainDiagXAmplitude/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Amplitude");
    fTree->Branch("PowerLorentzFitMainDiagXAmplitudeErr", &fPowerLorentzFitMainDiagXAmplitudeErr, "PowerLorentzFitMainDiagXAmplitudeErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Amplitude Error");
    fTree->Branch("PowerLorentzFitMainDiagXBeta", &fPowerLorentzFitMainDiagXBeta, "PowerLorentzFitMainDiagXBeta/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Beta Parameter");
    fTree->Branch("PowerLorentzFitMainDiagXBetaErr", &fPowerLorentzFitMainDiagXBetaErr, "PowerLorentzFitMainDiagXBetaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Beta Parameter Error");
    fTree->Branch("PowerLorentzFitMainDiagXGamma", &fPowerLorentzFitMainDiagXGamma, "PowerLorentzFitMainDiagXGamma/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Gamma Parameter");
    fTree->Branch("PowerLorentzFitMainDiagXGammaErr", &fPowerLorentzFitMainDiagXGammaErr, "PowerLorentzFitMainDiagXGammaErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Gamma Parameter Error");
    fTree->Branch("PowerLorentzFitMainDiagXVerticalOffset", &fPowerLorentzFitMainDiagXVerticalOffset, "PowerLorentzFitMainDiagXVerticalOffset/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Vertical Offset");
    fTree->Branch("PowerLorentzFitMainDiagXVerticalOffsetErr", &fPowerLorentzFitMainDiagXVerticalOffsetErr, "PowerLorentzFitMainDiagXVerticalOffsetErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Vertical Offset Error");
    fTree->Branch("PowerLorentzFitMainDiagXCenter", &fPowerLorentzFitMainDiagXCenter, "PowerLorentzFitMainDiagXCenter/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Center [mm]");
    fTree->Branch("PowerLorentzFitMainDiagXCenterErr", &fPowerLorentzFitMainDiagXCenterErr, "PowerLorentzFitMainDiagXCenterErr/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Center Error [mm]");
    fTree->Branch("PowerLorentzFitMainDiagXChi2red", &fPowerLorentzFitMainDiagXChi2red, "PowerLorentzFitMainDiagXChi2red/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("PowerLorentzFitMainDiagXPp", &fPowerLorentzFitMainDiagXPp, "PowerLorentzFitMainDiagXPp/D")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit P-value");
    fTree->Branch("PowerLorentzFitMainDiagXDOF", &fPowerLorentzFitMainDiagXDOF, "PowerLorentzFitMainDiagXDOF/I")->SetTitle("Power-Law Lorentzian Main Diagonal X Fit Degrees of Freedom");

    // Delta branches for power Lorentzian fitting
    fTree->Branch("PowerLorentzRowDeltaX", &fPowerLorentzRowDeltaX, "PowerLorentzRowDeltaX/D")->SetTitle("Delta X from Power-Law Lorentzian Row Fit to True Position [mm]");
    fTree->Branch("PowerLorentzColumnDeltaY", &fPowerLorentzColumnDeltaY, "PowerLorentzColumnDeltaY/D")->SetTitle("Delta Y from Power-Law Lorentzian Column Fit to True Position [mm]");

        // Transformed diagonal coordinates branches for Power-Law Lorentzian
    fTree->Branch("PowerLorentzMainDiagTransformedX", &fPowerLorentzMainDiagTransformedX, "PowerLorentzMainDiagTransformedX/D")->SetTitle("Transformed X from Power-Law Lorentzian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("PowerLorentzMainDiagTransformedY", &fPowerLorentzMainDiagTransformedY, "PowerLorentzMainDiagTransformedY/D")->SetTitle("Transformed Y from Power-Law Lorentzian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("PowerLorentzSecondDiagTransformedX", &fPowerLorentzSecondDiagTransformedX, "PowerLorentzSecondDiagTransformedX/D")->SetTitle("Transformed X from Power-Law Lorentzian Secondary Diagonal (rotation matrix) [mm]");
    fTree->Branch("PowerLorentzSecondDiagTransformedY", &fPowerLorentzSecondDiagTransformedY, "PowerLorentzSecondDiagTransformedY/D")->SetTitle("Transformed Y from Power-Law Lorentzian Secondary Diagonal (rotation matrix) [mm]");

    // Delta values for transformed coordinates vs true position for Power-Law Lorentzian
    fTree->Branch("PowerLorentzMainDiagTransformedDeltaX", &fPowerLorentzMainDiagTransformedDeltaX, "PowerLorentzMainDiagTransformedDeltaX/D")->SetTitle("Delta X from Power-Law Lorentzian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("PowerLorentzMainDiagTransformedDeltaY", &fPowerLorentzMainDiagTransformedDeltaY, "PowerLorentzMainDiagTransformedDeltaY/D")->SetTitle("Delta Y from Power-Law Lorentzian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("PowerLorentzSecondDiagTransformedDeltaX", &fPowerLorentzSecondDiagTransformedDeltaX, "PowerLorentzSecondDiagTransformedDeltaX/D")->SetTitle("Delta X from Power-Law Lorentzian Secondary Diagonal Transformed to True Position [mm]");
    fTree->Branch("PowerLorentzSecondDiagTransformedDeltaY", &fPowerLorentzSecondDiagTransformedDeltaY, "PowerLorentzSecondDiagTransformedDeltaY/D")->SetTitle("Delta Y from Power-Law Lorentzian Secondary Diagonal Transformed to True Position [mm]");

    // Mean estimation branches for Power-Law Lorentzian
    fTree->Branch("PowerLorentzMeanTrueDeltaX", &fPowerLorentzMeanTrueDeltaX, "PowerLorentzMeanTrueDeltaX/D")->SetTitle("Mean Delta X from all Power-Law Lorentzian estimation methods to True Position [mm]");
    fTree->Branch("PowerLorentzMeanTrueDeltaY", &fPowerLorentzMeanTrueDeltaY, "PowerLorentzMeanTrueDeltaY/D")->SetTitle("Mean Delta Y from all Power-Law Lorentzian estimation methods to True Position [mm]");

    } // End of Power-Law Lorentzian fitting branches

    // =============================================
    // 3D LORENTZIAN FITS BRANCHES (conditionally created)
    // =============================================
    if (Constants::ENABLE_3D_LORENTZIAN_FITTING) {
    // 3D Lorentzian fit parameters
    fTree->Branch("3DLorentzianFitCenterX", &f3DLorentzianFitCenterX, "3DLorentzianFitCenterX/D")->SetTitle("3D Lorentzian Fit Center X [mm]");
    fTree->Branch("3DLorentzianFitCenterY", &f3DLorentzianFitCenterY, "3DLorentzianFitCenterY/D")->SetTitle("3D Lorentzian Fit Center Y [mm]");
    fTree->Branch("3DLorentzianFitGammaX", &f3DLorentzianFitGammaX, "3DLorentzianFitGammaX/D")->SetTitle("3D Lorentzian Fit Gamma X Parameter");
    fTree->Branch("3DLorentzianFitGammaY", &f3DLorentzianFitGammaY, "3DLorentzianFitGammaY/D")->SetTitle("3D Lorentzian Fit Gamma Y Parameter");
    fTree->Branch("3DLorentzianFitAmplitude", &f3DLorentzianFitAmplitude, "3DLorentzianFitAmplitude/D")->SetTitle("3D Lorentzian Fit Amplitude");
    fTree->Branch("3DLorentzianFitVerticalOffset", &f3DLorentzianFitVerticalOffset, "3DLorentzianFitVerticalOffset/D")->SetTitle("3D Lorentzian Fit Vertical Offset");
    
    // 3D Lorentzian fit parameter errors
    fTree->Branch("3DLorentzianFitCenterXErr", &f3DLorentzianFitCenterXErr, "3DLorentzianFitCenterXErr/D")->SetTitle("3D Lorentzian Fit Center X Error [mm]");
    fTree->Branch("3DLorentzianFitCenterYErr", &f3DLorentzianFitCenterYErr, "3DLorentzianFitCenterYErr/D")->SetTitle("3D Lorentzian Fit Center Y Error [mm]");
    fTree->Branch("3DLorentzianFitGammaXErr", &f3DLorentzianFitGammaXErr, "3DLorentzianFitGammaXErr/D")->SetTitle("3D Lorentzian Fit Gamma X Parameter Error");
    fTree->Branch("3DLorentzianFitGammaYErr", &f3DLorentzianFitGammaYErr, "3DLorentzianFitGammaYErr/D")->SetTitle("3D Lorentzian Fit Gamma Y Parameter Error");
    fTree->Branch("3DLorentzianFitAmplitudeErr", &f3DLorentzianFitAmplitudeErr, "3DLorentzianFitAmplitudeErr/D")->SetTitle("3D Lorentzian Fit Amplitude Error");
    fTree->Branch("3DLorentzianFitVerticalOffsetErr", &f3DLorentzianFitVerticalOffsetErr, "3DLorentzianFitVerticalOffsetErr/D")->SetTitle("3D Lorentzian Fit Vertical Offset Error");
    
    // 3D Lorentzian fit statistics
    fTree->Branch("3DLorentzianFitChi2red", &f3DLorentzianFitChi2red, "3DLorentzianFitChi2red/D")->SetTitle("3D Lorentzian Fit Reduced Chi-squared");
    fTree->Branch("3DLorentzianFitPp", &f3DLorentzianFitPp, "3DLorentzianFitPp/D")->SetTitle("3D Lorentzian Fit P-value");
    fTree->Branch("3DLorentzianFitDOF", &f3DLorentzianFitDOF, "3DLorentzianFitDOF/I")->SetTitle("3D Lorentzian Fit Degrees of Freedom");
    fTree->Branch("3DLorentzianFitSuccessful", &f3DLorentzianFitSuccessful, "3DLorentzianFitSuccessful/O")->SetTitle("3D Lorentzian Fit Success Flag");
    
    // Conditionally create charge uncertainty branch for 3D Lorentzian fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("3DLorentzianFitChargeUncertainty", &f3DLorentzianFitChargeUncertainty, "3DLorentzianFitChargeUncertainty/D")->SetTitle("3D Lorentzian Fit Charge Uncertainty");
    }
    
    // 3D Lorentzian delta branches
    fTree->Branch("3DLorentzianDeltaX", &f3DLorentzianDeltaX, "3DLorentzianDeltaX/D")->SetTitle("Delta X from 3D Lorentzian Fit to True Position [mm]");
    fTree->Branch("3DLorentzianDeltaY", &f3DLorentzianDeltaY, "3DLorentzianDeltaY/D")->SetTitle("Delta Y from 3D Lorentzian Fit to True Position [mm]");
    
    } // End of 3D Lorentzian fitting branches

    // =============================================
    // 3D POWER-LAW LORENTZIAN FITS BRANCHES (conditionally created)
    // =============================================
    if (Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) {
    // 3D Power-Law Lorentzian fit parameters
    fTree->Branch("3DPowerLorentzianFitCenterX", &f3DPowerLorentzianFitCenterX, "3DPowerLorentzianFitCenterX/D")->SetTitle("3D Power-Law Lorentzian Fit Center X [mm]");
    fTree->Branch("3DPowerLorentzianFitCenterY", &f3DPowerLorentzianFitCenterY, "3DPowerLorentzianFitCenterY/D")->SetTitle("3D Power-Law Lorentzian Fit Center Y [mm]");
    fTree->Branch("3DPowerLorentzianFitGammaX", &f3DPowerLorentzianFitGammaX, "3DPowerLorentzianFitGammaX/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma X Parameter");
    fTree->Branch("3DPowerLorentzianFitGammaY", &f3DPowerLorentzianFitGammaY, "3DPowerLorentzianFitGammaY/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma Y Parameter");
    fTree->Branch("3DPowerLorentzianFitBeta", &f3DPowerLorentzianFitBeta, "3DPowerLorentzianFitBeta/D")->SetTitle("3D Power-Law Lorentzian Fit Beta Parameter");
    fTree->Branch("3DPowerLorentzianFitAmplitude", &f3DPowerLorentzianFitAmplitude, "3DPowerLorentzianFitAmplitude/D")->SetTitle("3D Power-Law Lorentzian Fit Amplitude");
    fTree->Branch("3DPowerLorentzianFitVerticalOffset", &f3DPowerLorentzianFitVerticalOffset, "3DPowerLorentzianFitVerticalOffset/D")->SetTitle("3D Power-Law Lorentzian Fit Vertical Offset");
    
    // 3D Power-Law Lorentzian fit parameter errors
    fTree->Branch("3DPowerLorentzianFitCenterXErr", &f3DPowerLorentzianFitCenterXErr, "3DPowerLorentzianFitCenterXErr/D")->SetTitle("3D Power-Law Lorentzian Fit Center X Error [mm]");
    fTree->Branch("3DPowerLorentzianFitCenterYErr", &f3DPowerLorentzianFitCenterYErr, "3DPowerLorentzianFitCenterYErr/D")->SetTitle("3D Power-Law Lorentzian Fit Center Y Error [mm]");
    fTree->Branch("3DPowerLorentzianFitGammaXErr", &f3DPowerLorentzianFitGammaXErr, "3DPowerLorentzianFitGammaXErr/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma X Parameter Error");
    fTree->Branch("3DPowerLorentzianFitGammaYErr", &f3DPowerLorentzianFitGammaYErr, "3DPowerLorentzianFitGammaYErr/D")->SetTitle("3D Power-Law Lorentzian Fit Gamma Y Parameter Error");
    fTree->Branch("3DPowerLorentzianFitBetaErr", &f3DPowerLorentzianFitBetaErr, "3DPowerLorentzianFitBetaErr/D")->SetTitle("3D Power-Law Lorentzian Fit Beta Parameter Error");
    fTree->Branch("3DPowerLorentzianFitAmplitudeErr", &f3DPowerLorentzianFitAmplitudeErr, "3DPowerLorentzianFitAmplitudeErr/D")->SetTitle("3D Power-Law Lorentzian Fit Amplitude Error");
    fTree->Branch("3DPowerLorentzianFitVerticalOffsetErr", &f3DPowerLorentzianFitVerticalOffsetErr, "3DPowerLorentzianFitVerticalOffsetErr/D")->SetTitle("3D Power-Law Lorentzian Fit Vertical Offset Error");
    
    // 3D Power-Law Lorentzian fit statistics
    fTree->Branch("3DPowerLorentzianFitChi2red", &f3DPowerLorentzianFitChi2red, "3DPowerLorentzianFitChi2red/D")->SetTitle("3D Power-Law Lorentzian Fit Reduced Chi-squared");
    fTree->Branch("3DPowerLorentzianFitPp", &f3DPowerLorentzianFitPp, "3DPowerLorentzianFitPp/D")->SetTitle("3D Power-Law Lorentzian Fit P-value");
    fTree->Branch("3DPowerLorentzianFitDOF", &f3DPowerLorentzianFitDOF, "3DPowerLorentzianFitDOF/I")->SetTitle("3D Power-Law Lorentzian Fit Degrees of Freedom");
    fTree->Branch("3DPowerLorentzianFitSuccessful", &f3DPowerLorentzianFitSuccessful, "3DPowerLorentzianFitSuccessful/O")->SetTitle("3D Power-Law Lorentzian Fit Success Flag");
    
    // Conditionally create charge uncertainty branch for 3D Power-Law Lorentzian fit
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        fTree->Branch("3DPowerLorentzianFitChargeUncertainty", &f3DPowerLorentzianFitChargeUncertainty, "3DPowerLorentzianFitChargeUncertainty/D")->SetTitle("3D Power-Law Lorentzian Fit Charge Uncertainty");
    }
    
    // 3D Power-Law Lorentzian delta branches
    fTree->Branch("3DPowerLorentzianDeltaX", &f3DPowerLorentzianDeltaX, "3DPowerLorentzianDeltaX/D")->SetTitle("Delta X from 3D Power-Law Lorentzian Fit to True Position [mm]");
    fTree->Branch("3DPowerLorentzianDeltaY", &f3DPowerLorentzianDeltaY, "3DPowerLorentzianDeltaY/D")->SetTitle("Delta Y from 3D Power-Law Lorentzian Fit to True Position [mm]");
    
    } // End of 3D Power-Law Lorentzian fitting branches

    // Load vector dictionaries for ROOT to properly handle std::vector branches
    gROOT->ProcessLine("#include <vector>");

    G4cout << "Created ROOT file and tree successfully: " << fileName << G4endl;
  }
  catch (std::exception& e) {
    G4cerr << "Exception in BeginOfRunAction: " << e.what() << G4endl;
    if (fRootFile) {
      if (fRootFile->IsOpen()) fRootFile->Close();
      delete fRootFile;
      fRootFile = nullptr;
      fTree = nullptr;
    }
  }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    G4int nofEvents = run->GetNumberOfEvent();
    G4String fileName = "";
    G4int nEntries = 0;
    
    // Lock mutex during ROOT file operations
    {
        std::lock_guard<std::mutex> lock(fRootMutex);
        
        if (fRootFile && !fRootFile->IsZombie()) {
            fileName = fRootFile->GetName();
        }
        
        if (fTree) {
            nEntries = fTree->GetEntries();
        }
        
        if (fRootFile && fTree && nofEvents > 0) {
            try {
                G4cout << "Writing ROOT file with " << nEntries 
                       << " entries from " << nofEvents << " events" << G4endl;
                
                // Save detector grid parameters as metadata before writing the tree
                // Use the stored grid parameters that were set by DetectorConstruction
                if (fGridPixelSize > 0) {  // Check if grid parameters have been set
                    fRootFile->cd();
                    
                    // Create TNamed objects to store grid parameters as metadata (RAII)
                    std::unique_ptr<TNamed> pixelSizeMeta(new TNamed("GridPixelSize_mm", Form("%.6f", fGridPixelSize)));
                    std::unique_ptr<TNamed> pixelSpacingMeta(new TNamed("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing)));
                    std::unique_ptr<TNamed> pixelCornerOffsetMeta(new TNamed("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset)));
                    std::unique_ptr<TNamed> detSizeMeta(new TNamed("GridDetectorSize_mm", Form("%.6f", fGridDetSize)));
                    std::unique_ptr<TNamed> numBlocksMeta(new TNamed("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide)));
                    std::unique_ptr<TNamed> neighborhoodRadiusMeta(new TNamed("NeighborhoodRadius", Form("%d", Constants::NEIGHBORHOOD_RADIUS)));
                    
                    // Write metadata to the ROOT file
                    pixelSizeMeta->Write();
                    pixelSpacingMeta->Write();
                    pixelCornerOffsetMeta->Write();
                    detSizeMeta->Write();
                    numBlocksMeta->Write();
                    neighborhoodRadiusMeta->Write();
                    
                    // Objects automatically cleaned up by unique_ptr destructors
                    
                    G4cout << "Saved detector grid metadata to ROOT file" << G4endl;
                    G4cout << "  Final parameters used: " << fGridPixelSize << ", " << fGridPixelSpacing 
                           << ", " << fGridPixelCornerOffset << ", " << fGridDetSize << ", " << fGridNumBlocksPerSide << G4endl;
                } else {
                    G4cerr << "Warning: Grid parameters not set, cannot save metadata to ROOT file" << G4endl;
                }
                
                // Write tree to file and close file
                if (fRootFile->IsOpen()) {
                    fRootFile->cd();
                    fTree->Write();
                    // Explicitly flush to disk before closing
                    fRootFile->Flush();
                    fRootFile->Close();
                }
                
                G4cout << "Run ended. Data saved to " << fileName << G4endl;
            }
            catch (std::exception& e) {
                G4cerr << "Exception in EndOfRunAction when writing file: " << e.what() << G4endl;
                // Try to recover by forcing a close
                try {
                    if (fRootFile && fRootFile->IsOpen()) {
                        fRootFile->Close();
                    }
                } catch (...) {
                    G4cerr << "Failed to close ROOT file after error" << G4endl;
                }
            }
        }
        
        // Clean up ROOT objects
        if (fRootFile) {
            delete fRootFile;
            fRootFile = nullptr;
            fTree = nullptr; // Tree is owned by file, no need to delete separately
        }
    }
    
    // Master thread merges the ROOT files from worker threads
    if (G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread()) {
        G4cout << "Master thread: Merging ROOT files from worker threads..." << G4endl;
        
        // Wait for all worker threads to finish writing files
        // Use proper synchronization instead of fixed sleep
        if (G4Threading::IsMultithreadedApplication()) {
            G4int nThreads = G4Threading::GetNumberOfRunningWorkerThreads();
            G4int maxWaitTime = 10000; // Maximum wait time in milliseconds
            G4int waitInterval = 100;   // Check interval in milliseconds
            G4int totalWaitTime = 0;
            
            // Wait for all worker files to exist and be properly closed
            bool allFilesReady = false;
            while (!allFilesReady && totalWaitTime < maxWaitTime) {
                allFilesReady = true;
                for (G4int i = 0; i < nThreads; i++) {
                    std::ostringstream oss;
                    oss << "epicChargeSharingOutput_t" << i << ".root";
                    G4String workerFile = oss.str();
                    
                    // Check if file exists and can be opened
                    TFile *testFile = TFile::Open(workerFile.c_str(), "READ");
                    if (!testFile || testFile->IsZombie()) {
                        allFilesReady = false;
                        if (testFile) delete testFile;
                        break;
                    }
                    testFile->Close();
                    delete testFile;
                }
                
                if (!allFilesReady) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(waitInterval));
                    totalWaitTime += waitInterval;
                }
            }
            
            if (totalWaitTime >= maxWaitTime) {
                G4cout << "Warning: Timeout waiting for worker files, proceeding with merge..." << G4endl;
            } else {
                G4cout << "All worker files ready after " << totalWaitTime << " ms" << G4endl;
            }
        }
        
        try {
            std::lock_guard<std::mutex> lock(fRootMutex);
            
            // Use TChain to merge the trees from different files
            TChain chain("Hits");
            G4int validFiles = 0;
            G4int nThreads = G4Threading::GetNumberOfRunningWorkerThreads();
            std::vector<G4String> workerFileNames;
            
            // First verify all files exist and are valid
            for (G4int i = 0; i < nThreads; i++) {
                std::ostringstream oss;
                oss << "epicChargeSharingOutput_t" << i << ".root";
                G4String workerFile = oss.str();
                workerFileNames.push_back(workerFile);
                
                TFile *testFile = TFile::Open(workerFile.c_str(), "READ");
                if (testFile && !testFile->IsZombie()) {
                    TTree *testTree = (TTree*)testFile->Get("Hits");
                    if (testTree && testTree->GetEntries() > 0) {
                        testFile->Close();
                        chain.Add(workerFile.c_str());
                        validFiles++;
                        G4cout << "Added " << workerFile << " to chain" << G4endl;
                    } else {
                        G4cout << "File " << workerFile << " has no valid tree or entries" << G4endl;
                        testFile->Close();
                    }
                    delete testFile;
                } else {
                    if (testFile) {
                        delete testFile;
                    }
                    G4cout << "Could not open file " << workerFile << G4endl;
                }
            }
            
            if (validFiles > 0) {
                // Create the merged output file with compression
                TFile *mergedFile = TFile::Open("epicChargeSharingOutput.root", "RECREATE", "", 1); // compression level 1
                if (mergedFile && !mergedFile->IsZombie()) {
                    G4int totalEntries = chain.GetEntries();
                    
                    if (totalEntries > 0) {
                        // Suppress possible warnings during merging
                        G4int oldLevel = gErrorIgnoreLevel;
                        gErrorIgnoreLevel = kWarning;
                        
                        // Clone the chain to create a merged tree
                        TTree *mergedTree = chain.CloneTree(-1, "fast");
                        
                        // Reset error level
                        gErrorIgnoreLevel = oldLevel;
                        
                        if (mergedTree) {
                            mergedTree->SetDirectory(mergedFile);
                            mergedFile->cd();
                            mergedTree->Write();
                            
                            // Save detector grid parameters as metadata to merged file
                            // Use the stored grid parameters that were set by DetectorConstruction
                            if (fGridPixelSize > 0) {  // Check if grid parameters have been set
                                // Create TNamed objects to store grid parameters as metadata (RAII)
                                std::unique_ptr<TNamed> pixelSizeMeta(new TNamed("GridPixelSize_mm", Form("%.6f", fGridPixelSize)));
                                std::unique_ptr<TNamed> pixelSpacingMeta(new TNamed("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing)));
                                std::unique_ptr<TNamed> pixelCornerOffsetMeta(new TNamed("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset)));
                                std::unique_ptr<TNamed> detSizeMeta(new TNamed("GridDetectorSize_mm", Form("%.6f", fGridDetSize)));
                                std::unique_ptr<TNamed> numBlocksMeta(new TNamed("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide)));
                                std::unique_ptr<TNamed> neighborhoodRadiusMeta(new TNamed("NeighborhoodRadius_pixels", Form("%d", Constants::NEIGHBORHOOD_RADIUS)));
                                
                                // Write metadata to the merged ROOT file
                                pixelSizeMeta->Write();
                                pixelSpacingMeta->Write();
                                pixelCornerOffsetMeta->Write();
                                detSizeMeta->Write();
                                numBlocksMeta->Write();
                                neighborhoodRadiusMeta->Write();
                                
                                // Objects automatically cleaned up by unique_ptr destructors
                                
                                G4cout << "Saved detector grid metadata to merged ROOT file" << G4endl;
                                G4cout << "  Final parameters used: " << fGridPixelSize << ", " << fGridPixelSpacing 
                                       << ", " << fGridPixelCornerOffset << ", " << fGridDetSize << ", " << fGridNumBlocksPerSide << G4endl;
                            } else {
                                G4cerr << "Warning: Grid parameters not set, cannot save metadata to merged ROOT file" << G4endl;
                            }
                            
                            mergedFile->Flush();
                            
                            G4cout << "Successfully merged " << validFiles << " files with " 
                                    << mergedTree->GetEntries() << " total entries" << G4endl;
                        }
                    } else {
                        G4cerr << "No entries found in chain to merge" << G4endl;
                    }
                    
                    mergedFile->Close();
                    delete mergedFile;
                } else {
                    G4cerr << "Could not create merged output file" << G4endl;
                }
                
                // Clean up worker files after successful merge
                for (const auto& file : workerFileNames) {
                    if (std::remove(file.c_str()) == 0) {
                        G4cout << "Cleaned up worker file: " << file << G4endl;
                    } else {
                        G4cerr << "Failed to clean up worker file: " << file << G4endl;
                    }
                }
            } else {
                G4cerr << "No valid files to merge" << G4endl;
            }
        }
        catch (const std::exception& e) {
            G4cerr << "Exception during file merging: " << e.what() << G4endl;
        }
    }
}

void RunAction::SetEventData(G4double edep, G4double x, G4double y, G4double z) 
{
    // Store energy deposit in MeV (Geant4 internal energy unit is MeV)
    fEdep = edep;
    
    // Store positions in mm (Geant4 internal length unit is mm)
    fTrueX = x;
    fTrueY = y;
    fTrueZ = z;
}

void RunAction::SetInitialPosition(G4double x, G4double y, G4double z) 
{
    // Store positions in mm (Geant4 internal length unit is mm)
    fInitX = x;
    fInitY = y;
    fInitZ = z;
}

void RunAction::SetNearestPixelPosition(G4double x, G4double y, G4double z) 
{
    // Store positions in mm (Geant4 internal length unit is mm)
    fPixelX = x;
    fPixelY = y;
    fPixelZ = z;
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
        
        // Auto-save every 1000 entries to prevent data loss in case of crash
        if (fTree->GetEntries() % 1000 == 0) {
            fRootFile->SaveSelf();
        }
    }
    catch (const std::exception& e) {
        G4cerr << "Exception in FillTree: " << e.what() << G4endl;
    }
}

void RunAction::SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                           G4double pixelCornerOffset, G4double detSize, 
                                           G4int numBlocksPerSide)
{
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
