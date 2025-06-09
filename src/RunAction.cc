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
#include <chrono>
#include <thread>
#include <cstdio> // For std::remove
#include <fstream>
#include <limits>

// Initialize the static mutex
std::mutex RunAction::fRootMutex;

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
  fGaussMainDiagDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussMainDiagDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fGaussSecondDiagDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzRowDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzColumnDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzMainDiagDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fLorentzSecondDiagDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
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
  fLorentzFitColumnAmplitudeErr(0),
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
  // Legacy variables
  fPixelZ(0),
  fIsPixelHit(false),
  fInitialEnergy(0),
  fMomentum(0),
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
  // Lock mutex during ROOT file creation
  std::lock_guard<std::mutex> lock(fRootMutex);
  
  // Create ROOT file and tree with error handling
  try {
    // Create a unique filename based on the thread ID for worker threads
    G4String fileName = "epicToyOutput";
    
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
    fTree->Branch("Hits/TrueX", &fTrueX, "TrueX/D")->SetTitle("True Position X [mm]");
    fTree->Branch("Hits/TrueY", &fTrueY, "TrueY/D")->SetTitle("True Position Y [mm]");
    fTree->Branch("Hits/TrueZ", &fTrueZ, "TrueZ/D")->SetTitle("True Position Z [mm]");
    fTree->Branch("Hits/InitX", &fInitX, "InitX/D")->SetTitle("Initial X [mm]");
    fTree->Branch("Hits/InitY", &fInitY, "InitY/D")->SetTitle("Initial Y [mm]");
    fTree->Branch("Hits/InitZ", &fInitZ, "InitZ/D")->SetTitle("Initial Z [mm]");
    fTree->Branch("Hits/PixelX", &fPixelX, "PixelX/D")->SetTitle("Nearest Pixel X [mm]");
    fTree->Branch("Hits/PixelY", &fPixelY, "PixelY/D")->SetTitle("Nearest Pixel Y [mm]");
    fTree->Branch("Hits/PixelZ", &fPixelZ, "PixelZ/D")->SetTitle("Nearest to hit pixel center Z [mm]");
    fTree->Branch("Hits/Edep", &fEdep, "Edep/D")->SetTitle("Energy Deposit [MeV]");
    fTree->Branch("Hits/InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
    fTree->Branch("Hits/Momentum", &fMomentum, "Momentum/D")->SetTitle("Particle Momentum [MeV/c]");
    fTree->Branch("Hits/IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("True if hit is on pixel OR distance <= D0");
    fTree->Branch("Hits/PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D")->SetTitle("Delta X from Pixel Center to True Position [mm] (x_pixel - x_true)");
    fTree->Branch("Hits/PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D")->SetTitle("Delta Y from Pixel Center to True Position [mm] (y_pixel - y_true)");
    fTree->Branch("Hits/GaussRowDeltaX", &fGaussRowDeltaX, "GaussRowDeltaX/D")->SetTitle("Delta X from Gaussian Row Fit to True Position [mm]");
    fTree->Branch("Hits/GaussColumnDeltaY", &fGaussColumnDeltaY, "GaussColumnDeltaY/D")->SetTitle("Delta Y from Gaussian Column Fit to True Position [mm]");
    fTree->Branch("Hits/GaussMainDiagDeltaX", &fGaussMainDiagDeltaX, "GaussMainDiagDeltaX/D")->SetTitle("Delta X from Gaussian Main Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/GaussMainDiagDeltaY", &fGaussMainDiagDeltaY, "GaussMainDiagDeltaY/D")->SetTitle("Delta Y from Gaussian Main Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/GaussSecondDiagDeltaX", &fGaussSecondDiagDeltaX, "GaussSecondDiagDeltaX/D")->SetTitle("Delta X from Gaussian Second Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/GaussSecondDiagDeltaY", &fGaussSecondDiagDeltaY, "GaussSecondDiagDeltaY/D")->SetTitle("Delta Y from Gaussian Second Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzRowDeltaX", &fLorentzRowDeltaX, "LorentzRowDeltaX/D")->SetTitle("Delta X from Lorentzian Row Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzColumnDeltaY", &fLorentzColumnDeltaY, "LorentzColumnDeltaY/D")->SetTitle("Delta Y from Lorentzian Column Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzMainDiagDeltaX", &fLorentzMainDiagDeltaX, "LorentzMainDiagDeltaX/D")->SetTitle("Delta X from Lorentzian Main Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzMainDiagDeltaY", &fLorentzMainDiagDeltaY, "LorentzMainDiagDeltaY/D")->SetTitle("Delta Y from Lorentzian Main Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzSecondDiagDeltaX", &fLorentzSecondDiagDeltaX, "LorentzSecondDiagDeltaX/D")->SetTitle("Delta X from Lorentzian Second Diagonal Fit to True Position [mm]");
    fTree->Branch("Hits/LorentzSecondDiagDeltaY", &fLorentzSecondDiagDeltaY, "LorentzSecondDiagDeltaY/D")->SetTitle("Delta Y from Lorentzian Second Diagonal Fit to True Position [mm]");

    // TRANSFORMED DIAGONAL COORDINATES BRANCHES
    // Transformed coordinates from rotation matrix (θ=45° and θ=-45°)
    fTree->Branch("Hits/GaussMainDiagTransformedX", &fGaussMainDiagTransformedX, "GaussMainDiagTransformedX/D")->SetTitle("Transformed X from Gaussian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/GaussMainDiagTransformedY", &fGaussMainDiagTransformedY, "GaussMainDiagTransformedY/D")->SetTitle("Transformed Y from Gaussian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/GaussSecondDiagTransformedX", &fGaussSecondDiagTransformedX, "GaussSecondDiagTransformedX/D")->SetTitle("Transformed X from Gaussian Secondary Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/GaussSecondDiagTransformedY", &fGaussSecondDiagTransformedY, "GaussSecondDiagTransformedY/D")->SetTitle("Transformed Y from Gaussian Secondary Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/LorentzMainDiagTransformedX", &fLorentzMainDiagTransformedX, "LorentzMainDiagTransformedX/D")->SetTitle("Transformed X from Lorentzian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/LorentzMainDiagTransformedY", &fLorentzMainDiagTransformedY, "LorentzMainDiagTransformedY/D")->SetTitle("Transformed Y from Lorentzian Main Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/LorentzSecondDiagTransformedX", &fLorentzSecondDiagTransformedX, "LorentzSecondDiagTransformedX/D")->SetTitle("Transformed X from Lorentzian Secondary Diagonal (rotation matrix) [mm]");
    fTree->Branch("Hits/LorentzSecondDiagTransformedY", &fLorentzSecondDiagTransformedY, "LorentzSecondDiagTransformedY/D")->SetTitle("Transformed Y from Lorentzian Secondary Diagonal (rotation matrix) [mm]");
    
    // Delta values for transformed coordinates vs true position
    fTree->Branch("Hits/GaussMainDiagTransformedDeltaX", &fGaussMainDiagTransformedDeltaX, "GaussMainDiagTransformedDeltaX/D")->SetTitle("Delta X from Gaussian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/GaussMainDiagTransformedDeltaY", &fGaussMainDiagTransformedDeltaY, "GaussMainDiagTransformedDeltaY/D")->SetTitle("Delta Y from Gaussian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/GaussSecondDiagTransformedDeltaX", &fGaussSecondDiagTransformedDeltaX, "GaussSecondDiagTransformedDeltaX/D")->SetTitle("Delta X from Gaussian Secondary Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/GaussSecondDiagTransformedDeltaY", &fGaussSecondDiagTransformedDeltaY, "GaussSecondDiagTransformedDeltaY/D")->SetTitle("Delta Y from Gaussian Secondary Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/LorentzMainDiagTransformedDeltaX", &fLorentzMainDiagTransformedDeltaX, "LorentzMainDiagTransformedDeltaX/D")->SetTitle("Delta X from Lorentzian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/LorentzMainDiagTransformedDeltaY", &fLorentzMainDiagTransformedDeltaY, "LorentzMainDiagTransformedDeltaY/D")->SetTitle("Delta Y from Lorentzian Main Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/LorentzSecondDiagTransformedDeltaX", &fLorentzSecondDiagTransformedDeltaX, "LorentzSecondDiagTransformedDeltaX/D")->SetTitle("Delta X from Lorentzian Secondary Diagonal Transformed to True Position [mm]");
    fTree->Branch("Hits/LorentzSecondDiagTransformedDeltaY", &fLorentzSecondDiagTransformedDeltaY, "LorentzSecondDiagTransformedDeltaY/D")->SetTitle("Delta Y from Lorentzian Secondary Diagonal Transformed to True Position [mm]");

    // MEAN ESTIMATION BRANCHES
    // Mean delta values from all estimation methods
    fTree->Branch("Hits/GaussMeanTrueDeltaX", &fGaussMeanTrueDeltaX, "GaussMeanTrueDeltaX/D")->SetTitle("Mean Delta X from all Gaussian estimation methods to True Position [mm]");
    fTree->Branch("Hits/GaussMeanTrueDeltaY", &fGaussMeanTrueDeltaY, "GaussMeanTrueDeltaY/D")->SetTitle("Mean Delta Y from all Gaussian estimation methods to True Position [mm]");
    fTree->Branch("Hits/LorentzMeanTrueDeltaX", &fLorentzMeanTrueDeltaX, "LorentzMeanTrueDeltaX/D")->SetTitle("Mean Delta X from all Lorentzian estimation methods to True Position [mm]");
    fTree->Branch("Hits/LorentzMeanTrueDeltaY", &fLorentzMeanTrueDeltaY, "LorentzMeanTrueDeltaY/D")->SetTitle("Mean Delta Y from all Lorentzian estimation methods to True Position [mm]");

    // GRIDNEIGHBORHOOD BRANCHES
    // Grid neighborhood data for 9x9 neighborhood around hits
    fTree->Branch("GridNeighborhood/GridNeighborhoodAngles", &fNonPixel_GridNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg]");
    fTree->Branch("GridNeighborhood/GridNeighborhoodChargeFractions", &fNonPixel_GridNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels");
    fTree->Branch("GridNeighborhood/GridNeighborhoodDistances", &fNonPixel_GridNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm]");
    fTree->Branch("GridNeighborhood/GridNeighborhoodCharges", &fNonPixel_GridNeighborhoodCharge)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels");
    
    // =============================================
    // GAUSSIAN FITS BRANCHES
    // =============================================
    // GaussFitRow/GaussFitRowX
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowAmplitude", &fGaussFitRowAmplitude, "GaussFitRowAmplitude/D")->SetTitle("Gaussian Row Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowAmplitudeErr", &fGaussFitRowAmplitudeErr, "GaussFitRowAmplitudeErr/D")->SetTitle("Gaussian Row Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowStdev", &fGaussFitRowStdev, "GaussFitRowStdev/D")->SetTitle("Gaussian Row Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowStdevErr", &fGaussFitRowStdevErr, "GaussFitRowStdevErr/D")->SetTitle("Gaussian Row Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowVerticalOffset", &fGaussFitRowVerticalOffset, "GaussFitRowVerticalOffset/D")->SetTitle("Gaussian Row Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowVerticalOffsetErr", &fGaussFitRowVerticalOffsetErr, "GaussFitRowVerticalOffsetErr/D")->SetTitle("Gaussian Row Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowCenter", &fGaussFitRowCenter, "GaussFitRowCenter/D")->SetTitle("Gaussian Row Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowCenterErr", &fGaussFitRowCenterErr, "GaussFitRowCenterErr/D")->SetTitle("Gaussian Row Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowChi2red", &fGaussFitRowChi2red, "GaussFitRowChi2red/D")->SetTitle("Gaussian Row Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowPp", &fGaussFitRowPp, "GaussFitRowPp/D")->SetTitle("Gaussian Row Fit P-value");
    fTree->Branch("GaussFits/GaussFitRow/GaussFitRowX/GaussFitRowDOF", &fGaussFitRowDOF, "GaussFitRowDOF/I")->SetTitle("Gaussian Row Fit Degrees of Freedom");
    
    // GaussFitColumn/GaussFitColumnY
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnAmplitude", &fGaussFitColumnAmplitude, "GaussFitColumnAmplitude/D")->SetTitle("Gaussian Column Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnAmplitudeErr", &fGaussFitColumnAmplitudeErr, "GaussFitColumnAmplitudeErr/D")->SetTitle("Gaussian Column Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnStdev", &fGaussFitColumnStdev, "GaussFitColumnStdev/D")->SetTitle("Gaussian Column Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnStdevErr", &fGaussFitColumnStdevErr, "GaussFitColumnStdevErr/D")->SetTitle("Gaussian Column Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnVerticalOffset", &fGaussFitColumnVerticalOffset, "GaussFitColumnVerticalOffset/D")->SetTitle("Gaussian Column Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnVerticalOffsetErr", &fGaussFitColumnVerticalOffsetErr, "GaussFitColumnVerticalOffsetErr/D")->SetTitle("Gaussian Column Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnCenter", &fGaussFitColumnCenter, "GaussFitColumnCenter/D")->SetTitle("Gaussian Column Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnCenterErr", &fGaussFitColumnCenterErr, "GaussFitColumnCenterErr/D")->SetTitle("Gaussian Column Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnChi2red", &fGaussFitColumnChi2red, "GaussFitColumnChi2red/D")->SetTitle("Gaussian Column Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnPp", &fGaussFitColumnPp, "GaussFitColumnPp/D")->SetTitle("Gaussian Column Fit P-value");
    fTree->Branch("GaussFits/GaussFitColumn/GaussFitColumnY/GaussFitColumnDOF", &fGaussFitColumnDOF, "GaussFitColumnDOF/I")->SetTitle("Gaussian Column Fit Degrees of Freedom");
    
    // GaussFitMainDiag/GaussFitMainDiagX
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagAmplitude", &fGaussFitMainDiagXAmplitude, "GaussFitMainDiagAmplitude/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagAmplitudeErr", &fGaussFitMainDiagXAmplitudeErr, "GaussFitMainDiagAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal X Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagStdev", &fGaussFitMainDiagXStdev, "GaussFitMainDiagStdev/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagStdevErr", &fGaussFitMainDiagXStdevErr, "GaussFitMainDiagStdevErr/D")->SetTitle("Gaussian Main Diagonal X Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagVerticalOffset", &fGaussFitMainDiagXVerticalOffset, "GaussFitMainDiagVerticalOffset/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagVerticalOffsetErr", &fGaussFitMainDiagXVerticalOffsetErr, "GaussFitMainDiagVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal X Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagCenter", &fGaussFitMainDiagXCenter, "GaussFitMainDiagCenter/D")->SetTitle("Gaussian Main Diagonal X Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagCenterErr", &fGaussFitMainDiagXCenterErr, "GaussFitMainDiagCenterErr/D")->SetTitle("Gaussian Main Diagonal X Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagChi2red", &fGaussFitMainDiagXChi2red, "GaussFitMainDiagChi2red/D")->SetTitle("Gaussian Main Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagPp", &fGaussFitMainDiagXPp, "GaussFitMainDiagPp/D")->SetTitle("Gaussian Main Diagonal X Fit P-value");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagX/GaussFitMainDiagDOF", &fGaussFitMainDiagXDOF, "GaussFitMainDiagDOF/I")->SetTitle("Gaussian Main Diagonal X Fit Degrees of Freedom");
    
    // GaussFitMainDiag/GaussFitMainDiagY
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagAmplitude", &fGaussFitMainDiagYAmplitude, "GaussFitMainDiagAmplitude/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagAmplitudeErr", &fGaussFitMainDiagYAmplitudeErr, "GaussFitMainDiagAmplitudeErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagStdev", &fGaussFitMainDiagYStdev, "GaussFitMainDiagStdev/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagStdevErr", &fGaussFitMainDiagYStdevErr, "GaussFitMainDiagStdevErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagVerticalOffset", &fGaussFitMainDiagYVerticalOffset, "GaussFitMainDiagVerticalOffset/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagVerticalOffsetErr", &fGaussFitMainDiagYVerticalOffsetErr, "GaussFitMainDiagVerticalOffsetErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagCenter", &fGaussFitMainDiagYCenter, "GaussFitMainDiagCenter/D")->SetTitle("Gaussian Main Diagonal Y Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagCenterErr", &fGaussFitMainDiagYCenterErr, "GaussFitMainDiagCenterErr/D")->SetTitle("Gaussian Main Diagonal Y Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagChi2red", &fGaussFitMainDiagYChi2red, "GaussFitMainDiagChi2red/D")->SetTitle("Gaussian Main Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagPp", &fGaussFitMainDiagYPp, "GaussFitMainDiagPp/D")->SetTitle("Gaussian Main Diagonal Y Fit P-value");
    fTree->Branch("GaussFits/GaussFitMainDiag/GaussFitMainDiagY/GaussFitMainDiagDOF", &fGaussFitMainDiagYDOF, "GaussFitMainDiagDOF/I")->SetTitle("Gaussian Main Diagonal Y Fit Degrees of Freedom");
    
    // GaussFitSecondDiag/GaussFitSecondDiagX
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagAmplitude", &fGaussFitSecondDiagXAmplitude, "GaussFitSecondDiagAmplitude/D")->SetTitle("Gaussian Second Diagonal X Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagAmplitudeErr", &fGaussFitSecondDiagXAmplitudeErr, "GaussFitSecondDiagAmplitudeErr/D")->SetTitle("Gaussian Second Diagonal X Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagStdev", &fGaussFitSecondDiagXStdev, "GaussFitSecondDiagStdev/D")->SetTitle("Gaussian Second Diagonal X Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagStdevErr", &fGaussFitSecondDiagXStdevErr, "GaussFitSecondDiagStdevErr/D")->SetTitle("Gaussian Second Diagonal X Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagVerticalOffset", &fGaussFitSecondDiagXVerticalOffset, "GaussFitSecondDiagVerticalOffset/D")->SetTitle("Gaussian Second Diagonal X Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagVerticalOffsetErr", &fGaussFitSecondDiagXVerticalOffsetErr, "GaussFitSecondDiagVerticalOffsetErr/D")->SetTitle("Gaussian Second Diagonal X Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagCenter", &fGaussFitSecondDiagXCenter, "GaussFitSecondDiagCenter/D")->SetTitle("Gaussian Second Diagonal X Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagCenterErr", &fGaussFitSecondDiagXCenterErr, "GaussFitSecondDiagCenterErr/D")->SetTitle("Gaussian Second Diagonal X Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagChi2red", &fGaussFitSecondDiagXChi2red, "GaussFitSecondDiagChi2red/D")->SetTitle("Gaussian Second Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagPp", &fGaussFitSecondDiagXPp, "GaussFitSecondDiagPp/D")->SetTitle("Gaussian Second Diagonal X Fit P-value");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagX/GaussFitSecondDiagDOF", &fGaussFitSecondDiagXDOF, "GaussFitSecondDiagDOF/I")->SetTitle("Gaussian Second Diagonal X Fit Degrees of Freedom");
    
    // GaussFitSecondDiag/GaussFitSecondDiagY
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagAmplitude", &fGaussFitSecondDiagYAmplitude, "GaussFitSecondDiagAmplitude/D")->SetTitle("Gaussian Second Diagonal Y Fit Amplitude");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagAmplitudeErr", &fGaussFitSecondDiagYAmplitudeErr, "GaussFitSecondDiagAmplitudeErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Amplitude Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagStdev", &fGaussFitSecondDiagYStdev, "GaussFitSecondDiagStdev/D")->SetTitle("Gaussian Second Diagonal Y Fit Standard Deviation");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagStdevErr", &fGaussFitSecondDiagYStdevErr, "GaussFitSecondDiagStdevErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Standard Deviation Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagVerticalOffset", &fGaussFitSecondDiagYVerticalOffset, "GaussFitSecondDiagVerticalOffset/D")->SetTitle("Gaussian Second Diagonal Y Fit Vertical Offset");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagVerticalOffsetErr", &fGaussFitSecondDiagYVerticalOffsetErr, "GaussFitSecondDiagVerticalOffsetErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagCenter", &fGaussFitSecondDiagYCenter, "GaussFitSecondDiagCenter/D")->SetTitle("Gaussian Second Diagonal Y Fit Center [mm]");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagCenterErr", &fGaussFitSecondDiagYCenterErr, "GaussFitSecondDiagCenterErr/D")->SetTitle("Gaussian Second Diagonal Y Fit Center Error [mm]");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagChi2red", &fGaussFitSecondDiagYChi2red, "GaussFitSecondDiagChi2red/D")->SetTitle("Gaussian Second Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagPp", &fGaussFitSecondDiagYPp, "GaussFitSecondDiagPp/D")->SetTitle("Gaussian Second Diagonal Y Fit P-value");
    fTree->Branch("GaussFits/GaussFitSecondDiag/GaussFitSecondDiagY/GaussFitSecondDiagDOF", &fGaussFitSecondDiagYDOF, "GaussFitSecondDiagDOF/I")->SetTitle("Gaussian Second Diagonal Y Fit Degrees of Freedom");
    
    // =============================================
    // LORENTZIAN FITS BRANCHES
    // =============================================
    // LorentzFitRow/LorentzFitRowX
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowAmplitude", &fLorentzFitRowAmplitude, "LorentzFitRowAmplitude/D")->SetTitle("Lorentzian Row Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowAmplitudeErr", &fLorentzFitRowAmplitudeErr, "LorentzFitRowAmplitudeErr/D")->SetTitle("Lorentzian Row Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowGamma", &fLorentzFitRowGamma, "LorentzFitRowGamma/D")->SetTitle("Lorentzian Row Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowGammaErr", &fLorentzFitRowGammaErr, "LorentzFitRowGammaErr/D")->SetTitle("Lorentzian Row Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowVerticalOffset", &fLorentzFitRowVerticalOffset, "LorentzFitRowVerticalOffset/D")->SetTitle("Lorentzian Row Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowVerticalOffsetErr", &fLorentzFitRowVerticalOffsetErr, "LorentzFitRowVerticalOffsetErr/D")->SetTitle("Lorentzian Row Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowCenter", &fLorentzFitRowCenter, "LorentzFitRowCenter/D")->SetTitle("Lorentzian Row Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowCenterErr", &fLorentzFitRowCenterErr, "LorentzFitRowCenterErr/D")->SetTitle("Lorentzian Row Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowChi2red", &fLorentzFitRowChi2red, "LorentzFitRowChi2red/D")->SetTitle("Lorentzian Row Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowPp", &fLorentzFitRowPp, "LorentzFitRowPp/D")->SetTitle("Lorentzian Row Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitRow/LorentzFitRowX/LorentzFitRowDOF", &fLorentzFitRowDOF, "LorentzFitRowDOF/I")->SetTitle("Lorentzian Row Fit Degrees of Freedom");

    // LorentzFitColumn/LorentzFitColumnY
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnAmplitude", &fLorentzFitColumnAmplitude, "LorentzFitColumnAmplitude/D")->SetTitle("Lorentzian Column Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnAmplitudeErr", &fLorentzFitColumnAmplitudeErr, "LorentzFitColumnAmplitudeErr/D")->SetTitle("Lorentzian Column Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnGamma", &fLorentzFitColumnGamma, "LorentzFitColumnGamma/D")->SetTitle("Lorentzian Column Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnGammaErr", &fLorentzFitColumnGammaErr, "LorentzFitColumnGammaErr/D")->SetTitle("Lorentzian Column Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnVerticalOffset", &fLorentzFitColumnVerticalOffset, "LorentzFitColumnVerticalOffset/D")->SetTitle("Lorentzian Column Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnVerticalOffsetErr", &fLorentzFitColumnVerticalOffsetErr, "LorentzFitColumnVerticalOffsetErr/D")->SetTitle("Lorentzian Column Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnCenter", &fLorentzFitColumnCenter, "LorentzFitColumnCenter/D")->SetTitle("Lorentzian Column Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnCenterErr", &fLorentzFitColumnCenterErr, "LorentzFitColumnCenterErr/D")->SetTitle("Lorentzian Column Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnChi2red", &fLorentzFitColumnChi2red, "LorentzFitColumnChi2red/D")->SetTitle("Lorentzian Column Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnPp", &fLorentzFitColumnPp, "LorentzFitColumnPp/D")->SetTitle("Lorentzian Column Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitColumn/LorentzFitColumnY/LorentzFitColumnDOF", &fLorentzFitColumnDOF, "LorentzFitColumnDOF/I")->SetTitle("Lorentzian Column Fit Degrees of Freedom");

    // LorentzFitMainDiag/LorentzFitMainDiagX
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagAmplitude", &fLorentzFitMainDiagXAmplitude, "LorentzFitMainDiagAmplitude/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagAmplitudeErr", &fLorentzFitMainDiagXAmplitudeErr, "LorentzFitMainDiagAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagGamma", &fLorentzFitMainDiagXGamma, "LorentzFitMainDiagGamma/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagGammaErr", &fLorentzFitMainDiagXGammaErr, "LorentzFitMainDiagGammaErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagVerticalOffset", &fLorentzFitMainDiagXVerticalOffset, "LorentzFitMainDiagVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagVerticalOffsetErr", &fLorentzFitMainDiagXVerticalOffsetErr, "LorentzFitMainDiagVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagCenter", &fLorentzFitMainDiagXCenter, "LorentzFitMainDiagCenter/D")->SetTitle("Lorentzian Main Diagonal X Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagCenterErr", &fLorentzFitMainDiagXCenterErr, "LorentzFitMainDiagCenterErr/D")->SetTitle("Lorentzian Main Diagonal X Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagChi2red", &fLorentzFitMainDiagXChi2red, "LorentzFitMainDiagChi2red/D")->SetTitle("Lorentzian Main Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagPp", &fLorentzFitMainDiagXPp, "LorentzFitMainDiagPp/D")->SetTitle("Lorentzian Main Diagonal X Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagX/LorentzFitMainDiagDOF", &fLorentzFitMainDiagXDOF, "LorentzFitMainDiagDOF/I")->SetTitle("Lorentzian Main Diagonal X Fit Degrees of Freedom");

    // LorentzFitMainDiag/LorentzFitMainDiagY
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagAmplitude", &fLorentzFitMainDiagYAmplitude, "LorentzFitMainDiagAmplitude/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagAmplitudeErr", &fLorentzFitMainDiagYAmplitudeErr, "LorentzFitMainDiagAmplitudeErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagGamma", &fLorentzFitMainDiagYGamma, "LorentzFitMainDiagGamma/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagGammaErr", &fLorentzFitMainDiagYGammaErr, "LorentzFitMainDiagGammaErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagVerticalOffset", &fLorentzFitMainDiagYVerticalOffset, "LorentzFitMainDiagVerticalOffset/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagVerticalOffsetErr", &fLorentzFitMainDiagYVerticalOffsetErr, "LorentzFitMainDiagVerticalOffsetErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagCenter", &fLorentzFitMainDiagYCenter, "LorentzFitMainDiagCenter/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagCenterErr", &fLorentzFitMainDiagYCenterErr, "LorentzFitMainDiagCenterErr/D")->SetTitle("Lorentzian Main Diagonal Y Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagChi2red", &fLorentzFitMainDiagYChi2red, "LorentzFitMainDiagChi2red/D")->SetTitle("Lorentzian Main Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagPp", &fLorentzFitMainDiagYPp, "LorentzFitMainDiagPp/D")->SetTitle("Lorentzian Main Diagonal Y Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitMainDiag/LorentzFitMainDiagY/LorentzFitMainDiagDOF", &fLorentzFitMainDiagYDOF, "LorentzFitMainDiagDOF/I")->SetTitle("Lorentzian Main Diagonal Y Fit Degrees of Freedom");

    // LorentzFitSecondDiag/LorentzFitSecondDiagX
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagAmplitude", &fLorentzFitSecondDiagXAmplitude, "LorentzFitSecondDiagAmplitude/D")->SetTitle("Lorentzian Second Diagonal X Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagAmplitudeErr", &fLorentzFitSecondDiagXAmplitudeErr, "LorentzFitSecondDiagAmplitudeErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagGamma", &fLorentzFitSecondDiagXGamma, "LorentzFitSecondDiagGamma/D")->SetTitle("Lorentzian Second Diagonal X Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagGammaErr", &fLorentzFitSecondDiagXGammaErr, "LorentzFitSecondDiagGammaErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagVerticalOffset", &fLorentzFitSecondDiagXVerticalOffset, "LorentzFitSecondDiagVerticalOffset/D")->SetTitle("Lorentzian Second Diagonal X Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagVerticalOffsetErr", &fLorentzFitSecondDiagXVerticalOffsetErr, "LorentzFitSecondDiagVerticalOffsetErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagCenter", &fLorentzFitSecondDiagXCenter, "LorentzFitSecondDiagCenter/D")->SetTitle("Lorentzian Second Diagonal X Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagCenterErr", &fLorentzFitSecondDiagXCenterErr, "LorentzFitSecondDiagCenterErr/D")->SetTitle("Lorentzian Second Diagonal X Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagChi2red", &fLorentzFitSecondDiagXChi2red, "LorentzFitSecondDiagChi2red/D")->SetTitle("Lorentzian Second Diagonal X Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagPp", &fLorentzFitSecondDiagXPp, "LorentzFitSecondDiagPp/D")->SetTitle("Lorentzian Second Diagonal X Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagX/LorentzFitSecondDiagDOF", &fLorentzFitSecondDiagXDOF, "LorentzFitSecondDiagDOF/I")->SetTitle("Lorentzian Second Diagonal X Fit Degrees of Freedom");

    // LorentzFitSecondDiag/LorentzFitSecondDiagY
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagAmplitude", &fLorentzFitSecondDiagYAmplitude, "LorentzFitSecondDiagAmplitude/D")->SetTitle("Lorentzian Second Diagonal Y Fit Amplitude");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagAmplitudeErr", &fLorentzFitSecondDiagYAmplitudeErr, "LorentzFitSecondDiagAmplitudeErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Amplitude Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagGamma", &fLorentzFitSecondDiagYGamma, "LorentzFitSecondDiagGamma/D")->SetTitle("Lorentzian Second Diagonal Y Fit Gamma Parameter");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagGammaErr", &fLorentzFitSecondDiagYGammaErr, "LorentzFitSecondDiagGammaErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Gamma Parameter Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagVerticalOffset", &fLorentzFitSecondDiagYVerticalOffset, "LorentzFitSecondDiagVerticalOffset/D")->SetTitle("Lorentzian Second Diagonal Y Fit Vertical Offset");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagVerticalOffsetErr", &fLorentzFitSecondDiagYVerticalOffsetErr, "LorentzFitSecondDiagVerticalOffsetErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Vertical Offset Error");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagCenter", &fLorentzFitSecondDiagYCenter, "LorentzFitSecondDiagCenter/D")->SetTitle("Lorentzian Second Diagonal Y Fit Center [mm]");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagCenterErr", &fLorentzFitSecondDiagYCenterErr, "LorentzFitSecondDiagCenterErr/D")->SetTitle("Lorentzian Second Diagonal Y Fit Center Error [mm]");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagChi2red", &fLorentzFitSecondDiagYChi2red, "LorentzFitSecondDiagChi2red/D")->SetTitle("Lorentzian Second Diagonal Y Fit Reduced Chi-squared");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagPp", &fLorentzFitSecondDiagYPp, "LorentzFitSecondDiagPp/D")->SetTitle("Lorentzian Second Diagonal Y Fit P-value");
    fTree->Branch("LorentzFits/LorentzFitSecondDiag/LorentzFitSecondDiagY/LorentzFitSecondDiagDOF", &fLorentzFitSecondDiagYDOF, "LorentzFitSecondDiagDOF/I")->SetTitle("Lorentzian Second Diagonal Y Fit Degrees of Freedom");

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
                    
                    // Create TNamed objects to store grid parameters as metadata
                    TNamed *pixelSizeMeta = new TNamed("GridPixelSize", Form("%.6f", fGridPixelSize));
                    TNamed *pixelSpacingMeta = new TNamed("GridPixelSpacing", Form("%.6f", fGridPixelSpacing));  
                    TNamed *pixelCornerOffsetMeta = new TNamed("GridPixelCornerOffset", Form("%.6f", fGridPixelCornerOffset));
                    TNamed *detSizeMeta = new TNamed("GridDetectorSize", Form("%.6f", fGridDetSize));
                    TNamed *numBlocksMeta = new TNamed("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
                    TNamed *neighborhoodRadiusMeta = new TNamed("NeighborhoodRadius", Form("%d", Constants::NEIGHBORHOOD_RADIUS));
                    
                    // Write metadata to the ROOT file
                    pixelSizeMeta->Write();
                    pixelSpacingMeta->Write();
                    pixelCornerOffsetMeta->Write();
                    detSizeMeta->Write();
                    numBlocksMeta->Write();
                    neighborhoodRadiusMeta->Write();
                    
                    // Clean up metadata objects to prevent memory leaks
                    delete pixelSizeMeta;
                    delete pixelSpacingMeta;
                    delete pixelCornerOffsetMeta;
                    delete detSizeMeta;
                    delete numBlocksMeta;
                    delete neighborhoodRadiusMeta;
                    
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
        
        // Wait a moment to ensure worker threads have finished writing files
        G4long wait_time = 1000; // milliseconds
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        
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
                oss << "epicToyOutput_t" << i << ".root";
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
                TFile *mergedFile = TFile::Open("epicToyOutput.root", "RECREATE", "", 1); // compression level 1
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
                                TNamed *pixelSizeMeta = new TNamed("GridPixelSize", Form("%.6f", fGridPixelSize));
                                TNamed *pixelSpacingMeta = new TNamed("GridPixelSpacing", Form("%.6f", fGridPixelSpacing));  
                                TNamed *pixelCornerOffsetMeta = new TNamed("GridPixelCornerOffset", Form("%.6f", fGridPixelCornerOffset));
                                TNamed *detSizeMeta = new TNamed("GridDetectorSize", Form("%.6f", fGridDetSize));
                                TNamed *numBlocksMeta = new TNamed("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
                                TNamed *neighborhoodRadiusMeta = new TNamed("NeighborhoodRadius", Form("%d", Constants::NEIGHBORHOOD_RADIUS));
                                
                                // Write metadata to the merged ROOT file
                                pixelSizeMeta->Write();
                                pixelSpacingMeta->Write();
                                pixelCornerOffsetMeta->Write();
                                detSizeMeta->Write();
                                numBlocksMeta->Write();
                                neighborhoodRadiusMeta->Write();
                                
                                // Clean up metadata objects to prevent memory leaks
                                delete pixelSizeMeta;
                                delete pixelSpacingMeta;
                                delete pixelCornerOffsetMeta;
                                delete detSizeMeta;
                                delete numBlocksMeta;
                                delete neighborhoodRadiusMeta;
                                
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
    
    // Calculate delta values for diagonal fits vs true position
    if (fit_successful) {
        // Main diagonal delta values (using X and Y centers from main diagonal fits)
        if (main_diag_x_fit_successful) {
            fGaussMainDiagDeltaX = main_diag_x_center - fTrueX;  // x_diag_fit - x_true
        } else {
            fGaussMainDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (main_diag_y_fit_successful) {
            fGaussMainDiagDeltaY = main_diag_y_center - fTrueY;  // y_diag_fit - y_true
        } else {
            fGaussMainDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Secondary diagonal delta values (using X and Y centers from secondary diagonal fits)
        if (sec_diag_x_fit_successful) {
            fGaussSecondDiagDeltaX = sec_diag_x_center - fTrueX;  // x_secdiag_fit - x_true
        } else {
            fGaussSecondDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (sec_diag_y_fit_successful) {
            fGaussSecondDiagDeltaY = sec_diag_y_center - fTrueY;  // y_secdiag_fit - y_true
        } else {
            fGaussSecondDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // For failed overall diagonal fitting, set all delta values to NaN
        fGaussMainDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussMainDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        fGaussSecondDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussSecondDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
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
    
    // Calculate delta values for diagonal fits vs true position
    if (fit_successful) {
        // Main diagonal delta values (using X and Y centers from main diagonal fits)
        if (main_diag_x_fit_successful) {
            fLorentzMainDiagDeltaX = main_diag_x_center - fTrueX;  // x_diag_fit - x_true
        } else {
            fLorentzMainDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (main_diag_y_fit_successful) {
            fLorentzMainDiagDeltaY = main_diag_y_center - fTrueY;  // y_diag_fit - y_true
        } else {
            fLorentzMainDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Secondary diagonal delta values (using X and Y centers from secondary diagonal fits)
        if (sec_diag_x_fit_successful) {
            fLorentzSecondDiagDeltaX = sec_diag_x_center - fTrueX;  // x_secdiag_fit - x_true
        } else {
            fLorentzSecondDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (sec_diag_y_fit_successful) {
            fLorentzSecondDiagDeltaY = sec_diag_y_center - fTrueY;  // y_secdiag_fit - y_true
        } else {
            fLorentzSecondDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // For failed overall diagonal fitting, set all delta values to NaN
        fLorentzMainDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMainDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzSecondDiagDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzSecondDiagDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
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
    // Transform main diagonal coordinates (θ = 45°)
    // For Gaussian fits
    if (!std::isnan(fGaussFitMainDiagXCenter) && !std::isnan(fGaussFitMainDiagYCenter)) {
        TransformDiagonalCoordinates(fGaussFitMainDiagXCenter, fGaussFitMainDiagYCenter, 45.0,
                                    fGaussMainDiagTransformedX, fGaussMainDiagTransformedY);
        // Calculate delta values
        fGaussMainDiagTransformedDeltaX = fGaussMainDiagTransformedX - fTrueX;
        fGaussMainDiagTransformedDeltaY = fGaussMainDiagTransformedY - fTrueY;
    } else {
        fGaussMainDiagTransformedX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussMainDiagTransformedY = std::numeric_limits<G4double>::quiet_NaN();
        fGaussMainDiagTransformedDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussMainDiagTransformedDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // For Lorentzian fits
    if (!std::isnan(fLorentzFitMainDiagXCenter) && !std::isnan(fLorentzFitMainDiagYCenter)) {
        TransformDiagonalCoordinates(fLorentzFitMainDiagXCenter, fLorentzFitMainDiagYCenter, 45.0,
                                    fLorentzMainDiagTransformedX, fLorentzMainDiagTransformedY);
        // Calculate delta values
        fLorentzMainDiagTransformedDeltaX = fLorentzMainDiagTransformedX - fTrueX;
        fLorentzMainDiagTransformedDeltaY = fLorentzMainDiagTransformedY - fTrueY;
    } else {
        fLorentzMainDiagTransformedX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMainDiagTransformedY = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMainDiagTransformedDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzMainDiagTransformedDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // Transform secondary diagonal coordinates (θ = -45°)
    // For Gaussian fits
    if (!std::isnan(fGaussFitSecondDiagXCenter) && !std::isnan(fGaussFitSecondDiagYCenter)) {
        TransformDiagonalCoordinates(fGaussFitSecondDiagXCenter, fGaussFitSecondDiagYCenter, -45.0,
                                    fGaussSecondDiagTransformedX, fGaussSecondDiagTransformedY);
        // Calculate delta values
        fGaussSecondDiagTransformedDeltaX = fGaussSecondDiagTransformedX - fTrueX;
        fGaussSecondDiagTransformedDeltaY = fGaussSecondDiagTransformedY - fTrueY;
    } else {
        fGaussSecondDiagTransformedX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussSecondDiagTransformedY = std::numeric_limits<G4double>::quiet_NaN();
        fGaussSecondDiagTransformedDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fGaussSecondDiagTransformedDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
    
    // For Lorentzian fits
    if (!std::isnan(fLorentzFitSecondDiagXCenter) && !std::isnan(fLorentzFitSecondDiagYCenter)) {
        TransformDiagonalCoordinates(fLorentzFitSecondDiagXCenter, fLorentzFitSecondDiagYCenter, -45.0,
                                    fLorentzSecondDiagTransformedX, fLorentzSecondDiagTransformedY);
        // Calculate delta values
        fLorentzSecondDiagTransformedDeltaX = fLorentzSecondDiagTransformedX - fTrueX;
        fLorentzSecondDiagTransformedDeltaY = fLorentzSecondDiagTransformedY - fTrueY;
    } else {
        fLorentzSecondDiagTransformedX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzSecondDiagTransformedY = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzSecondDiagTransformedDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fLorentzSecondDiagTransformedDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
}

void RunAction::CalculateMeanEstimations()
{
    // Vectors to collect valid coordinate estimations
    std::vector<G4double> gauss_x_coords, gauss_y_coords;
    std::vector<G4double> lorentz_x_coords, lorentz_y_coords;
    
    // For Gaussian estimations, collect X coordinates:
    // 1. Row fit center (gives X coordinate)
    if (!std::isnan(fGaussFitRowCenter) && fGaussFitRowDOF > 0) {
        gauss_x_coords.push_back(fGaussFitRowCenter);
    }
    
    // 2. Main diagonal transformed X
    if (!std::isnan(fGaussMainDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussMainDiagTransformedX);
    }
    
    // 3. Secondary diagonal transformed X  
    if (!std::isnan(fGaussSecondDiagTransformedX)) {
        gauss_x_coords.push_back(fGaussSecondDiagTransformedX);
    }
    
    // For Gaussian estimations, collect Y coordinates:
    // 1. Column fit center (gives Y coordinate)
    if (!std::isnan(fGaussFitColumnCenter) && fGaussFitColumnDOF > 0) {
        gauss_y_coords.push_back(fGaussFitColumnCenter);
    }
    
    // 2. Main diagonal transformed Y
    if (!std::isnan(fGaussMainDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussMainDiagTransformedY);
    }
    
    // 3. Secondary diagonal transformed Y
    if (!std::isnan(fGaussSecondDiagTransformedY)) {
        gauss_y_coords.push_back(fGaussSecondDiagTransformedY);
    }
    
    // For Lorentzian estimations, collect X coordinates:
    // 1. Row fit center (gives X coordinate)
    if (!std::isnan(fLorentzFitRowCenter) && fLorentzFitRowDOF > 0) {
        lorentz_x_coords.push_back(fLorentzFitRowCenter);
    }
    
    // 2. Main diagonal transformed X
    if (!std::isnan(fLorentzMainDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzMainDiagTransformedX);
    }
    
    // 3. Secondary diagonal transformed X
    if (!std::isnan(fLorentzSecondDiagTransformedX)) {
        lorentz_x_coords.push_back(fLorentzSecondDiagTransformedX);
    }
    
    // For Lorentzian estimations, collect Y coordinates:
    // 1. Column fit center (gives Y coordinate)
    if (!std::isnan(fLorentzFitColumnCenter) && fLorentzFitColumnDOF > 0) {
        lorentz_y_coords.push_back(fLorentzFitColumnCenter);
    }
    
    // 2. Main diagonal transformed Y
    if (!std::isnan(fLorentzMainDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzMainDiagTransformedY);
    }
    
    // 3. Secondary diagonal transformed Y
    if (!std::isnan(fLorentzSecondDiagTransformedY)) {
        lorentz_y_coords.push_back(fLorentzSecondDiagTransformedY);
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
}
