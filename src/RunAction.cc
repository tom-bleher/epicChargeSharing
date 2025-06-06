#include "RunAction.hh"
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
  fEdep(0),
  fTrueX(0),
  fTrueY(0),
  fTrueZ(0),
  fInitX(0),
  fInitY(0),
  fInitZ(0),
  fPixelX(0),
  fPixelY(0),
  fPixelZ(0),
  fPixelI(-1),
  fPixelJ(-1),
  fPixelTrueDeltaX(0),
  fPixelTrueDeltaY(0),
  fIsPixelHit(false),
  fPixelHit_PixelAlpha(0),
  // Initialize delta variables for Gaussian fit estimations vs true position
  fNonPixel_GaussRowTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fNonPixel_GaussColumnTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fNonPixel_GaussDiagTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fNonPixel_GaussDiagTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  fNonPixel_GaussSecDiagTrueDeltaX(std::numeric_limits<G4double>::quiet_NaN()),
  fNonPixel_GaussSecDiagTrueDeltaY(std::numeric_limits<G4double>::quiet_NaN()),
  // Initialize 2D Gaussian fit variables
  fNonPixel_Fit2D_XCenter(0),
  fNonPixel_Fit2D_XSigma(0),
  fNonPixel_Fit2D_XAmplitude(0),
  fNonPixel_Fit2D_XCenterErr(0),
  fNonPixel_Fit2D_XSigmaErr(0),
  fNonPixel_Fit2D_XAmplitudeErr(0),
  fNonPixel_Fit2D_XChi2red(0),
  fNonPixel_Fit2D_XNPoints(0),
  fNonPixel_Fit2D_YCenter(0),
  fNonPixel_Fit2D_YSigma(0),
  fNonPixel_Fit2D_YAmplitude(0),
  fNonPixel_Fit2D_YCenterErr(0),
  fNonPixel_Fit2D_YSigmaErr(0),
  fNonPixel_Fit2D_YAmplitudeErr(0),
  fNonPixel_Fit2D_YChi2red(0),
  fNonPixel_Fit2D_YNPoints(0),
  fNonPixel_Fit2D_Successful(false),
  // Initialize diagonal Gaussian fit variables (4 separate fits: Main X, Main Y, Sec X, Sec Y)
  fNonPixel_FitDiag_MainXCenter(0),
  fNonPixel_FitDiag_MainXSigma(0),
  fNonPixel_FitDiag_MainXAmplitude(0),
  fNonPixel_FitDiag_MainXCenterErr(0),
  fNonPixel_FitDiag_MainXSigmaErr(0),
  fNonPixel_FitDiag_MainXAmplitudeErr(0),
  fNonPixel_FitDiag_MainXChi2red(0),
  fNonPixel_FitDiag_MainXNPoints(0),
  fNonPixel_FitDiag_MainXSuccessful(false),
  fNonPixel_FitDiag_MainYCenter(0),
  fNonPixel_FitDiag_MainYSigma(0),
  fNonPixel_FitDiag_MainYAmplitude(0),
  fNonPixel_FitDiag_MainYCenterErr(0),
  fNonPixel_FitDiag_MainYSigmaErr(0),
  fNonPixel_FitDiag_MainYAmplitudeErr(0),
  fNonPixel_FitDiag_MainYChi2red(0),
  fNonPixel_FitDiag_MainYNPoints(0),
  fNonPixel_FitDiag_MainYSuccessful(false),
  fNonPixel_FitDiag_SecXCenter(0),
  fNonPixel_FitDiag_SecXSigma(0),
  fNonPixel_FitDiag_SecXAmplitude(0),
  fNonPixel_FitDiag_SecXCenterErr(0),
  fNonPixel_FitDiag_SecXSigmaErr(0),
  fNonPixel_FitDiag_SecXAmplitudeErr(0),
  fNonPixel_FitDiag_SecXChi2red(0),
  fNonPixel_FitDiag_SecXNPoints(0),
  fNonPixel_FitDiag_SecXSuccessful(false),
  fNonPixel_FitDiag_SecYCenter(0),
  fNonPixel_FitDiag_SecYSigma(0),
  fNonPixel_FitDiag_SecYAmplitude(0),
  fNonPixel_FitDiag_SecYCenterErr(0),
  fNonPixel_FitDiag_SecYSigmaErr(0),
  fNonPixel_FitDiag_SecYAmplitudeErr(0),
  fNonPixel_FitDiag_SecYChi2red(0),
  fNonPixel_FitDiag_SecYNPoints(0),
  fNonPixel_FitDiag_SecYSuccessful(false),
  fNonPixel_FitDiag_Successful(false),
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
    
    // Create branches for the tree with explicit units in the titles
    // Note: Values are stored in Geant4's internal units (MeV for energy, mm for length)
    fTree->Branch("Edep", &fEdep, "Edep/D")->SetTitle("Energy Deposit [MeV]");
    fTree->Branch("TrueX", &fTrueX, "TrueX/D")->SetTitle("True Position X [mm]");
    fTree->Branch("TrueY", &fTrueY, "TrueY/D")->SetTitle("True Position Y [mm]");
    fTree->Branch("TrueZ", &fTrueZ, "TrueZ/D")->SetTitle("True Position Z [mm]");
    
    // Add branches for initial particle position with explicit units
    fTree->Branch("InitX", &fInitX, "InitX/D")->SetTitle("Initial X [mm]");
    fTree->Branch("InitY", &fInitY, "InitY/D")->SetTitle("Initial Y [mm]");
    fTree->Branch("InitZ", &fInitZ, "InitZ/D")->SetTitle("Initial Z [mm]");
    
    // Add branches for nearest pixel center position with explicit units
    fTree->Branch("PixelX", &fPixelX, "PixelX/D")->SetTitle("Nearest Pixel X [mm]");
    fTree->Branch("PixelY", &fPixelY, "PixelY/D")->SetTitle("Nearest Pixel Y [mm]");
    fTree->Branch("PixelZ", &fPixelZ, "PixelZ/D")->SetTitle("Nearest Pixel Z [mm]");
    
    // Add branches for pixel mapping information
    fTree->Branch("PixelI", &fPixelI, "PixelI/I")->SetTitle("Pixel Index X");
    fTree->Branch("PixelJ", &fPixelJ, "PixelJ/I")->SetTitle("Pixel Index Y");
    fTree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D")->SetTitle("Delta X from Hit to Pixel Center [mm]");
    fTree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D")->SetTitle("Delta Y from Hit to Pixel Center [mm]");
    
    // ==============================================
    // HIT CLASSIFICATION BRANCHES
    // ==============================================
    fTree->Branch("IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("Hit on Pixel OR distance <= D0");
    
    // ==============================================
    // PIXEL HIT DATA (distance <= D0 or on pixel)
    // ==============================================
    fTree->Branch("PixelHit_PixelAlpha", &fPixelHit_PixelAlpha, "PixelHit_PixelAlpha/D")->SetTitle("Angular Size of Pixel [deg] (for pixel hits)");
    
    // ==============================================
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    // ==============================================
    
    // Load vector dictionaries for ROOT to properly handle std::vector branches
    gROOT->ProcessLine("#include <vector>");
    
    // Add branches for neighborhood (9x9) grid angle data (non-pixel hits only)
    fTree->Branch("NonPixel_GridNeighborhoodAngles", &fNonPixel_GridNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg] (non-pixel hits)");
    fTree->Branch("NonPixel_GridNeighborhoodPixelI", &fNonPixel_GridNeighborhoodPixelI)->SetTitle("I Indices of Neighborhood Grid Pixels (non-pixel hits)");
    fTree->Branch("NonPixel_GridNeighborhoodPixelJ", &fNonPixel_GridNeighborhoodPixelJ)->SetTitle("J Indices of Neighborhood Grid Pixels (non-pixel hits)");
    
    // Add branches for neighborhood (9x9) grid charge sharing data (non-pixel hits only)
    fTree->Branch("NonPixel_GridNeighborhoodChargeFractions", &fNonPixel_GridNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels (non-pixel hits)");
    fTree->Branch("NonPixel_GridNeighborhoodDistances", &fNonPixel_GridNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm] (non-pixel hits)");
    fTree->Branch("NonPixel_GridNeighborhoodCharge", &fNonPixel_GridNeighborhoodCharge)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels (non-pixel hits)");
    
    // Add branches for particle information (reduced set)
    fTree->Branch("InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
    fTree->Branch("Momentum", &fMomentum, "Momentum/D")->SetTitle("Particle Momentum [MeV/c]");
    
    // Add branches for 2D Gaussian fit results (central row and column fitting)
    fTree->Branch("Fit2D_XCenter", &fNonPixel_Fit2D_XCenter, "Fit2D_XCenter/D")->SetTitle("Fitted X Center from Central Row [mm]");
    fTree->Branch("Fit2D_XSigma", &fNonPixel_Fit2D_XSigma, "Fit2D_XSigma/D")->SetTitle("Fitted X Sigma from Central Row [mm]");
    fTree->Branch("Fit2D_XAmplitude", &fNonPixel_Fit2D_XAmplitude, "Fit2D_XAmplitude/D")->SetTitle("Fitted X Amplitude from Central Row");
    fTree->Branch("Fit2D_XCenterErr", &fNonPixel_Fit2D_XCenterErr, "Fit2D_XCenterErr/D")->SetTitle("Error in Fitted X Center [mm]");
    fTree->Branch("Fit2D_XSigmaErr", &fNonPixel_Fit2D_XSigmaErr, "Fit2D_XSigmaErr/D")->SetTitle("Error in Fitted X Sigma [mm]");
    fTree->Branch("Fit2D_XAmplitudeErr", &fNonPixel_Fit2D_XAmplitudeErr, "Fit2D_XAmplitudeErr/D")->SetTitle("Error in Fitted X Amplitude");
    fTree->Branch("Fit2D_XChi2red", &fNonPixel_Fit2D_XChi2red, "Fit2D_XChi2red/D")->SetTitle("Reduced Chi-squared for X Fit");
    fTree->Branch("Fit2D_XNPoints", &fNonPixel_Fit2D_XNPoints, "Fit2D_XNPoints/I")->SetTitle("Number of Points Used in X Fit");
    
    fTree->Branch("Fit2D_YCenter", &fNonPixel_Fit2D_YCenter, "Fit2D_YCenter/D")->SetTitle("Fitted Y Center from Central Column [mm]");
    fTree->Branch("Fit2D_YSigma", &fNonPixel_Fit2D_YSigma, "Fit2D_YSigma/D")->SetTitle("Fitted Y Sigma from Central Column [mm]");
    fTree->Branch("Fit2D_YAmplitude", &fNonPixel_Fit2D_YAmplitude, "Fit2D_YAmplitude/D")->SetTitle("Fitted Y Amplitude from Central Column");
    fTree->Branch("Fit2D_YCenterErr", &fNonPixel_Fit2D_YCenterErr, "Fit2D_YCenterErr/D")->SetTitle("Error in Fitted Y Center [mm]");
    fTree->Branch("Fit2D_YSigmaErr", &fNonPixel_Fit2D_YSigmaErr, "Fit2D_YSigmaErr/D")->SetTitle("Error in Fitted Y Sigma [mm]");
    fTree->Branch("Fit2D_YAmplitudeErr", &fNonPixel_Fit2D_YAmplitudeErr, "Fit2D_YAmplitudeErr/D")->SetTitle("Error in Fitted Y Amplitude");
    fTree->Branch("Fit2D_YChi2red", &fNonPixel_Fit2D_YChi2red, "Fit2D_YChi2red/D")->SetTitle("Reduced Chi-squared for Y Fit");
    fTree->Branch("Fit2D_YNPoints", &fNonPixel_Fit2D_YNPoints, "Fit2D_YNPoints/I")->SetTitle("Number of Points Used in Y Fit");
    
    fTree->Branch("Fit2D_Successful", &fNonPixel_Fit2D_Successful, "Fit2D_Successful/O")->SetTitle("Whether 2D Fitting was Successful");
    
    // Add branches for diagonal Gaussian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    // Main diagonal X fit (X vs Charge for pixels on main diagonal)
    fTree->Branch("FitDiag_MainXCenter", &fNonPixel_FitDiag_MainXCenter, "FitDiag_MainXCenter/D")->SetTitle("Fitted X Center from Main Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_MainXSigma", &fNonPixel_FitDiag_MainXSigma, "FitDiag_MainXSigma/D")->SetTitle("Fitted X Sigma from Main Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_MainXAmplitude", &fNonPixel_FitDiag_MainXAmplitude, "FitDiag_MainXAmplitude/D")->SetTitle("Fitted X Amplitude from Main Diagonal X Fit");
    fTree->Branch("FitDiag_MainXCenterErr", &fNonPixel_FitDiag_MainXCenterErr, "FitDiag_MainXCenterErr/D")->SetTitle("Error in Fitted X Center from Main Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_MainXSigmaErr", &fNonPixel_FitDiag_MainXSigmaErr, "FitDiag_MainXSigmaErr/D")->SetTitle("Error in Fitted X Sigma from Main Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_MainXAmplitudeErr", &fNonPixel_FitDiag_MainXAmplitudeErr, "FitDiag_MainXAmplitudeErr/D")->SetTitle("Error in Fitted X Amplitude from Main Diagonal X Fit");
    fTree->Branch("FitDiag_MainXChi2red", &fNonPixel_FitDiag_MainXChi2red, "FitDiag_MainXChi2red/D")->SetTitle("Reduced Chi-squared for Main Diagonal X Fit");
    fTree->Branch("FitDiag_MainXNPoints", &fNonPixel_FitDiag_MainXNPoints, "FitDiag_MainXNPoints/I")->SetTitle("Number of Points Used in Main Diagonal X Fit");
    fTree->Branch("FitDiag_MainXSuccessful", &fNonPixel_FitDiag_MainXSuccessful, "FitDiag_MainXSuccessful/O")->SetTitle("Whether Main Diagonal X Fitting was Successful");
    
    // Main diagonal Y fit (Y vs Charge for pixels on main diagonal)
    fTree->Branch("FitDiag_MainYCenter", &fNonPixel_FitDiag_MainYCenter, "FitDiag_MainYCenter/D")->SetTitle("Fitted Y Center from Main Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_MainYSigma", &fNonPixel_FitDiag_MainYSigma, "FitDiag_MainYSigma/D")->SetTitle("Fitted Y Sigma from Main Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_MainYAmplitude", &fNonPixel_FitDiag_MainYAmplitude, "FitDiag_MainYAmplitude/D")->SetTitle("Fitted Y Amplitude from Main Diagonal Y Fit");
    fTree->Branch("FitDiag_MainYCenterErr", &fNonPixel_FitDiag_MainYCenterErr, "FitDiag_MainYCenterErr/D")->SetTitle("Error in Fitted Y Center from Main Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_MainYSigmaErr", &fNonPixel_FitDiag_MainYSigmaErr, "FitDiag_MainYSigmaErr/D")->SetTitle("Error in Fitted Y Sigma from Main Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_MainYAmplitudeErr", &fNonPixel_FitDiag_MainYAmplitudeErr, "FitDiag_MainYAmplitudeErr/D")->SetTitle("Error in Fitted Y Amplitude from Main Diagonal Y Fit");
    fTree->Branch("FitDiag_MainYChi2red", &fNonPixel_FitDiag_MainYChi2red, "FitDiag_MainYChi2red/D")->SetTitle("Reduced Chi-squared for Main Diagonal Y Fit");
    fTree->Branch("FitDiag_MainYNPoints", &fNonPixel_FitDiag_MainYNPoints, "FitDiag_MainYNPoints/I")->SetTitle("Number of Points Used in Main Diagonal Y Fit");
    fTree->Branch("FitDiag_MainYSuccessful", &fNonPixel_FitDiag_MainYSuccessful, "FitDiag_MainYSuccessful/O")->SetTitle("Whether Main Diagonal Y Fitting was Successful");
    
    // Secondary diagonal X fit (X vs Charge for pixels on secondary diagonal)
    fTree->Branch("FitDiag_SecXCenter", &fNonPixel_FitDiag_SecXCenter, "FitDiag_SecXCenter/D")->SetTitle("Fitted X Center from Secondary Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_SecXSigma", &fNonPixel_FitDiag_SecXSigma, "FitDiag_SecXSigma/D")->SetTitle("Fitted X Sigma from Secondary Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_SecXAmplitude", &fNonPixel_FitDiag_SecXAmplitude, "FitDiag_SecXAmplitude/D")->SetTitle("Fitted X Amplitude from Secondary Diagonal X Fit");
    fTree->Branch("FitDiag_SecXCenterErr", &fNonPixel_FitDiag_SecXCenterErr, "FitDiag_SecXCenterErr/D")->SetTitle("Error in Fitted X Center from Secondary Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_SecXSigmaErr", &fNonPixel_FitDiag_SecXSigmaErr, "FitDiag_SecXSigmaErr/D")->SetTitle("Error in Fitted X Sigma from Secondary Diagonal X Fit [mm]");
    fTree->Branch("FitDiag_SecXAmplitudeErr", &fNonPixel_FitDiag_SecXAmplitudeErr, "FitDiag_SecXAmplitudeErr/D")->SetTitle("Error in Fitted X Amplitude from Secondary Diagonal X Fit");
    fTree->Branch("FitDiag_SecXChi2red", &fNonPixel_FitDiag_SecXChi2red, "FitDiag_SecXChi2red/D")->SetTitle("Reduced Chi-squared for Secondary Diagonal X Fit");
    fTree->Branch("FitDiag_SecXNPoints", &fNonPixel_FitDiag_SecXNPoints, "FitDiag_SecXNPoints/I")->SetTitle("Number of Points Used in Secondary Diagonal X Fit");
    fTree->Branch("FitDiag_SecXSuccessful", &fNonPixel_FitDiag_SecXSuccessful, "FitDiag_SecXSuccessful/O")->SetTitle("Whether Secondary Diagonal X Fitting was Successful");
    
    // Secondary diagonal Y fit (Y vs Charge for pixels on secondary diagonal)
    fTree->Branch("FitDiag_SecYCenter", &fNonPixel_FitDiag_SecYCenter, "FitDiag_SecYCenter/D")->SetTitle("Fitted Y Center from Secondary Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_SecYSigma", &fNonPixel_FitDiag_SecYSigma, "FitDiag_SecYSigma/D")->SetTitle("Fitted Y Sigma from Secondary Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_SecYAmplitude", &fNonPixel_FitDiag_SecYAmplitude, "FitDiag_SecYAmplitude/D")->SetTitle("Fitted Y Amplitude from Secondary Diagonal Y Fit");
    fTree->Branch("FitDiag_SecYCenterErr", &fNonPixel_FitDiag_SecYCenterErr, "FitDiag_SecYCenterErr/D")->SetTitle("Error in Fitted Y Center from Secondary Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_SecYSigmaErr", &fNonPixel_FitDiag_SecYSigmaErr, "FitDiag_SecYSigmaErr/D")->SetTitle("Error in Fitted Y Sigma from Secondary Diagonal Y Fit [mm]");
    fTree->Branch("FitDiag_SecYAmplitudeErr", &fNonPixel_FitDiag_SecYAmplitudeErr, "FitDiag_SecYAmplitudeErr/D")->SetTitle("Error in Fitted Y Amplitude from Secondary Diagonal Y Fit");
    fTree->Branch("FitDiag_SecYChi2red", &fNonPixel_FitDiag_SecYChi2red, "FitDiag_SecYChi2red/D")->SetTitle("Reduced Chi-squared for Secondary Diagonal Y Fit");
    fTree->Branch("FitDiag_SecYNPoints", &fNonPixel_FitDiag_SecYNPoints, "FitDiag_SecYNPoints/I")->SetTitle("Number of Points Used in Secondary Diagonal Y Fit");
    fTree->Branch("FitDiag_SecYSuccessful", &fNonPixel_FitDiag_SecYSuccessful, "FitDiag_SecYSuccessful/O")->SetTitle("Whether Secondary Diagonal Y Fitting was Successful");
    
    fTree->Branch("FitDiag_Successful", &fNonPixel_FitDiag_Successful, "FitDiag_Successful/O")->SetTitle("Whether Diagonal Fitting was Successful");
    
    // Add branches for delta variables from Gaussian fit estimations vs true position
    fTree->Branch("GaussRowTrueDeltaX", &fNonPixel_GaussRowTrueDeltaX, "GaussRowTrueDeltaX/D")->SetTitle("Delta X from Row Fit Center to True Position [mm] (x_row_fit - x_true)");
    fTree->Branch("GaussColumnTrueDeltaY", &fNonPixel_GaussColumnTrueDeltaY, "GaussColumnTrueDeltaY/D")->SetTitle("Delta Y from Column Fit Center to True Position [mm] (y_column_fit - y_true)");
    fTree->Branch("GaussDiagTrueDeltaX", &fNonPixel_GaussDiagTrueDeltaX, "GaussDiagTrueDeltaX/D")->SetTitle("Delta X from Main Diagonal Fit Center to True Position [mm] (x_diag_fit - x_true)");
    fTree->Branch("GaussDiagTrueDeltaY", &fNonPixel_GaussDiagTrueDeltaY, "GaussDiagTrueDeltaY/D")->SetTitle("Delta Y from Main Diagonal Fit Center to True Position [mm] (y_diag_fit - y_true)");
    fTree->Branch("GaussSecDiagTrueDeltaX", &fNonPixel_GaussSecDiagTrueDeltaX, "GaussSecDiagTrueDeltaX/D")->SetTitle("Delta X from Secondary Diagonal Fit Center to True Position [mm] (x_secdiag_fit - x_true)");
    fTree->Branch("GaussSecDiagTrueDeltaY", &fNonPixel_GaussSecDiagTrueDeltaY, "GaussSecDiagTrueDeltaY/D")->SetTitle("Delta Y from Secondary Diagonal Fit Center to True Position [mm] (y_secdiag_fit - y_true)");
    
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
                    
                    // Write metadata to the ROOT file
                    pixelSizeMeta->Write();
                    pixelSpacingMeta->Write();
                    pixelCornerOffsetMeta->Write();
                    detSizeMeta->Write();
                    numBlocksMeta->Write();
                    
                    // Clean up metadata objects to prevent memory leaks
                    delete pixelSizeMeta;
                    delete pixelSpacingMeta;
                    delete pixelCornerOffsetMeta;
                    delete detSizeMeta;
                    delete numBlocksMeta;
                    
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
                                
                                // Write metadata to the merged ROOT file
                                pixelSizeMeta->Write();
                                pixelSpacingMeta->Write();
                                pixelCornerOffsetMeta->Write();
                                detSizeMeta->Write();
                                numBlocksMeta->Write();
                                
                                // Clean up metadata objects to prevent memory leaks
                                delete pixelSizeMeta;
                                delete pixelSpacingMeta;
                                delete pixelCornerOffsetMeta;
                                delete detSizeMeta;
                                delete numBlocksMeta;
                                
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

void RunAction::SetPixelIndices(G4int i, G4int j, G4double distance)
{
    // Store pixel indices and distance to center
    fPixelI = i;
    fPixelJ = j;
    // Note: The delta values will be set separately in SetPixelClassification
    // This method signature is kept for compatibility but distance parameter is no longer used
}

void RunAction::SetPixelAlpha(G4double alpha)
{
    // Store the angular size of the pixel from hit position (in degrees)
    fPixelHit_PixelAlpha = alpha;
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

void RunAction::SetNeighborhoodGridData(const std::vector<G4double>& angles, 
                               const std::vector<G4int>& pixelI, 
                               const std::vector<G4int>& pixelJ)
{
    // Store the neighborhood (9x9) grid angle data for non-pixel hits
    fNonPixel_GridNeighborhoodAngles = angles;
    fNonPixel_GridNeighborhoodPixelI = pixelI;
    fNonPixel_GridNeighborhoodPixelJ = pixelJ;
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
                                        G4double x_chi2red, G4int x_npoints,
                                        G4double y_center, G4double y_sigma, G4double y_amplitude,
                                        G4double y_center_err, G4double y_sigma_err, G4double y_amplitude_err,
                                        G4double y_chi2red, G4int y_npoints,
                                        G4bool fit_successful)
{
    // Store 2D Gaussian fit results from central row (X fit)
    fNonPixel_Fit2D_XCenter = x_center;
    fNonPixel_Fit2D_XSigma = x_sigma;
    fNonPixel_Fit2D_XAmplitude = x_amplitude;
    fNonPixel_Fit2D_XCenterErr = x_center_err;
    fNonPixel_Fit2D_XSigmaErr = x_sigma_err;
    fNonPixel_Fit2D_XAmplitudeErr = x_amplitude_err;
    fNonPixel_Fit2D_XChi2red = x_chi2red;
    fNonPixel_Fit2D_XNPoints = x_npoints;
    
    // Store 2D Gaussian fit results from central column (Y fit)
    fNonPixel_Fit2D_YCenter = y_center;
    fNonPixel_Fit2D_YSigma = y_sigma;
    fNonPixel_Fit2D_YAmplitude = y_amplitude;
    fNonPixel_Fit2D_YCenterErr = y_center_err;
    fNonPixel_Fit2D_YSigmaErr = y_sigma_err;
    fNonPixel_Fit2D_YAmplitudeErr = y_amplitude_err;
    fNonPixel_Fit2D_YChi2red = y_chi2red;
    fNonPixel_Fit2D_YNPoints = y_npoints;
    
    // Store overall fit success status
    fNonPixel_Fit2D_Successful = fit_successful;
    
    // Calculate delta values for row and column fits vs true position
    if (fit_successful) {
        fNonPixel_GaussRowTrueDeltaX = x_center - fTrueX;      // x_row_fit - x_true
        fNonPixel_GaussColumnTrueDeltaY = y_center - fTrueY;   // y_column_fit - y_true
    } else {
        // Set row and column delta values to NaN for failed fits
        fNonPixel_GaussRowTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fNonPixel_GaussColumnTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
}

void RunAction::SetDiagonalGaussianFitResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_amplitude,
                                             G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_amplitude_err,
                                             G4double main_diag_x_chi2red, G4int main_diag_x_npoints, G4bool main_diag_x_fit_successful,
                                             G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_amplitude,
                                             G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_amplitude_err,
                                             G4double main_diag_y_chi2red, G4int main_diag_y_npoints, G4bool main_diag_y_fit_successful,
                                             G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_amplitude,
                                             G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_amplitude_err,
                                             G4double sec_diag_x_chi2red, G4int sec_diag_x_npoints, G4bool sec_diag_x_fit_successful,
                                             G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_amplitude,
                                             G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_amplitude_err,
                                             G4double sec_diag_y_chi2red, G4int sec_diag_y_npoints, G4bool sec_diag_y_fit_successful,
                                             G4bool fit_successful)
{
    // Store main diagonal X fit results
    fNonPixel_FitDiag_MainXCenter = main_diag_x_center;
    fNonPixel_FitDiag_MainXSigma = main_diag_x_sigma;
    fNonPixel_FitDiag_MainXAmplitude = main_diag_x_amplitude;
    fNonPixel_FitDiag_MainXCenterErr = main_diag_x_center_err;
    fNonPixel_FitDiag_MainXSigmaErr = main_diag_x_sigma_err;
    fNonPixel_FitDiag_MainXAmplitudeErr = main_diag_x_amplitude_err;
    fNonPixel_FitDiag_MainXChi2red = main_diag_x_chi2red;
    fNonPixel_FitDiag_MainXNPoints = main_diag_x_npoints;
    fNonPixel_FitDiag_MainXSuccessful = main_diag_x_fit_successful;
    
    // Store main diagonal Y fit results
    fNonPixel_FitDiag_MainYCenter = main_diag_y_center;
    fNonPixel_FitDiag_MainYSigma = main_diag_y_sigma;
    fNonPixel_FitDiag_MainYAmplitude = main_diag_y_amplitude;
    fNonPixel_FitDiag_MainYCenterErr = main_diag_y_center_err;
    fNonPixel_FitDiag_MainYSigmaErr = main_diag_y_sigma_err;
    fNonPixel_FitDiag_MainYAmplitudeErr = main_diag_y_amplitude_err;
    fNonPixel_FitDiag_MainYChi2red = main_diag_y_chi2red;
    fNonPixel_FitDiag_MainYNPoints = main_diag_y_npoints;
    fNonPixel_FitDiag_MainYSuccessful = main_diag_y_fit_successful;
    
    // Store secondary diagonal X fit results
    fNonPixel_FitDiag_SecXCenter = sec_diag_x_center;
    fNonPixel_FitDiag_SecXSigma = sec_diag_x_sigma;
    fNonPixel_FitDiag_SecXAmplitude = sec_diag_x_amplitude;
    fNonPixel_FitDiag_SecXCenterErr = sec_diag_x_center_err;
    fNonPixel_FitDiag_SecXSigmaErr = sec_diag_x_sigma_err;
    fNonPixel_FitDiag_SecXAmplitudeErr = sec_diag_x_amplitude_err;
    fNonPixel_FitDiag_SecXChi2red = sec_diag_x_chi2red;
    fNonPixel_FitDiag_SecXNPoints = sec_diag_x_npoints;
    fNonPixel_FitDiag_SecXSuccessful = sec_diag_x_fit_successful;
    
    // Store secondary diagonal Y fit results
    fNonPixel_FitDiag_SecYCenter = sec_diag_y_center;
    fNonPixel_FitDiag_SecYSigma = sec_diag_y_sigma;
    fNonPixel_FitDiag_SecYAmplitude = sec_diag_y_amplitude;
    fNonPixel_FitDiag_SecYCenterErr = sec_diag_y_center_err;
    fNonPixel_FitDiag_SecYSigmaErr = sec_diag_y_sigma_err;
    fNonPixel_FitDiag_SecYAmplitudeErr = sec_diag_y_amplitude_err;
    fNonPixel_FitDiag_SecYChi2red = sec_diag_y_chi2red;
    fNonPixel_FitDiag_SecYNPoints = sec_diag_y_npoints;
    fNonPixel_FitDiag_SecYSuccessful = sec_diag_y_fit_successful;
    
    // Store overall fit success status
    fNonPixel_FitDiag_Successful = fit_successful;
    
    // Calculate delta values for diagonal fits vs true position
    if (fit_successful) {
        // Main diagonal delta values (using X and Y centers from main diagonal fits)
        if (main_diag_x_fit_successful) {
            fNonPixel_GaussDiagTrueDeltaX = main_diag_x_center - fTrueX;  // x_diag_fit - x_true
        } else {
            fNonPixel_GaussDiagTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (main_diag_y_fit_successful) {
            fNonPixel_GaussDiagTrueDeltaY = main_diag_y_center - fTrueY;  // y_diag_fit - y_true
        } else {
            fNonPixel_GaussDiagTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        // Secondary diagonal delta values (using X and Y centers from secondary diagonal fits)
        if (sec_diag_x_fit_successful) {
            fNonPixel_GaussSecDiagTrueDeltaX = sec_diag_x_center - fTrueX;  // x_secdiag_fit - x_true
        } else {
            fNonPixel_GaussSecDiagTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        }
        
        if (sec_diag_y_fit_successful) {
            fNonPixel_GaussSecDiagTrueDeltaY = sec_diag_y_center - fTrueY;  // y_secdiag_fit - y_true
        } else {
            fNonPixel_GaussSecDiagTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        }
    } else {
        // For failed overall diagonal fitting, set all delta values to NaN
        fNonPixel_GaussDiagTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fNonPixel_GaussDiagTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
        fNonPixel_GaussSecDiagTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
        fNonPixel_GaussSecDiagTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
}
