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
  fPixelDist(0),
  fIsPixelHit(false),
  fIsWithinD0(false),
  fDistanceToPixelCenter(0),
  fPixelHit_PixelAlpha(0),
  fNonPixel_FitAmplitude(0),
  fNonPixel_FitX0(0),
  fNonPixel_FitY0(0),
  fNonPixel_FitSigmaX(0),
  fNonPixel_FitSigmaY(0),
  fNonPixel_FitTheta(0),
  fNonPixel_FitOffset(0),
  fNonPixel_FitAmplitudeErr(0),
  fNonPixel_FitX0Err(0),
  fNonPixel_FitY0Err(0),
  fNonPixel_FitSigmaXErr(0),
  fNonPixel_FitSigmaYErr(0),
  fNonPixel_FitThetaErr(0),
  fNonPixel_FitOffsetErr(0),
  fNonPixel_FitChi2(0),
  fNonPixel_FitNDF(0),
  fNonPixel_FitChi2red(0),
  fNonPixel_FitPp(0),
  fNonPixel_FitNPoints(0),
  fNonPixel_FitResidualMean(0),
  fNonPixel_FitResidualStd(0),
  fNonPixel_FitConstraintsSatisfied(false),
  fNonPixel_GaussTrueDistance(std::numeric_limits<G4double>::quiet_NaN()),
  fEventID(-1),
  fInitialEnergy(0),
  fFinalEnergy(0),
  fMomentum(0),
  fParticleName(""),
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
    fTree->Branch("PixelTrueDistance", &fPixelDist, "PixelTrueDistance/D")->SetTitle("Distance to Pixel Center [mm]");
    
    // ==============================================
    // HIT CLASSIFICATION BRANCHES
    // ==============================================
    fTree->Branch("IsPixelHit", &fIsPixelHit, "IsPixelHit/O")->SetTitle("Hit on Pixel OR distance <= D0");
    fTree->Branch("IsWithinD0", &fIsWithinD0, "IsWithinD0/O")->SetTitle("Distance <= D0 (10 microns)");
    fTree->Branch("DistanceToPixelCenter", &fDistanceToPixelCenter, "DistanceToPixelCenter/D")->SetTitle("Distance to Nearest Pixel Center [mm]");
    
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
    
    // Add branches for particle information
    fTree->Branch("EventID", &fEventID, "EventID/I")->SetTitle("Event ID");
    fTree->Branch("InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
    fTree->Branch("FinalEnergy", &fFinalEnergy, "FinalEnergy/D")->SetTitle("Final Particle Energy [MeV]");
    fTree->Branch("Momentum", &fMomentum, "Momentum/D")->SetTitle("Particle Momentum [MeV/c]");
    fTree->Branch("ParticleName", &fParticleName)->SetTitle("Particle Type Name");
    
    // Add branches for step-by-step energy deposition information
    fTree->Branch("StepEnergyDeposition", &fStepEnergyDeposition)->SetTitle("Energy Deposited Per Step [MeV]");
    fTree->Branch("StepZPosition", &fStepZPositions)->SetTitle("Z Position of Each Energy Deposit [mm]");
    fTree->Branch("StepTime", &fStepTimes)->SetTitle("Time of Each Energy Deposit [ns]");
    
    // Add branches for ALL step information (including non-energy depositing steps)
    fTree->Branch("AllStepEnergyDeposition", &fAllStepEnergyDeposition)->SetTitle("Energy Deposited Per Step (All Steps) [MeV]");
    fTree->Branch("AllStepZPosition", &fAllStepZPositions)->SetTitle("Z Position of Each Step [mm]");
    fTree->Branch("AllStepTime", &fAllStepTimes)->SetTitle("Time of Each Step [ns]");
    
    // Add branches for 3D Gaussian fit results
    fTree->Branch("FitAmplitude", &fNonPixel_FitAmplitude, "FitAmplitude/D")->SetTitle("Fitted Gaussian Amplitude");
    fTree->Branch("FitX0", &fNonPixel_FitX0, "FitX0/D")->SetTitle("Fitted Gaussian Center X [mm]");
    fTree->Branch("FitY0", &fNonPixel_FitY0, "FitY0/D")->SetTitle("Fitted Gaussian Center Y [mm]");
    fTree->Branch("FitSigmaX", &fNonPixel_FitSigmaX, "FitSigmaX/D")->SetTitle("Fitted Gaussian Sigma X [mm]");
    fTree->Branch("FitSigmaY", &fNonPixel_FitSigmaY, "FitSigmaY/D")->SetTitle("Fitted Gaussian Sigma Y [mm]");
    fTree->Branch("FitTheta", &fNonPixel_FitTheta, "FitTheta/D")->SetTitle("Fitted Gaussian Rotation Angle [rad]");
    fTree->Branch("FitOffset", &fNonPixel_FitOffset, "FitOffset/D")->SetTitle("Fitted Gaussian Offset");
    
    fTree->Branch("FitAmplitudeErr", &fNonPixel_FitAmplitudeErr, "FitAmplitudeErr/D")->SetTitle("Error in Fitted Amplitude");
    fTree->Branch("FitX0Err", &fNonPixel_FitX0Err, "FitX0Err/D")->SetTitle("Error in Fitted Center X [mm]");
    fTree->Branch("FitY0Err", &fNonPixel_FitY0Err, "FitY0Err/D")->SetTitle("Error in Fitted Center Y [mm]");
    fTree->Branch("FitSigmaXErr", &fNonPixel_FitSigmaXErr, "FitSigmaXErr/D")->SetTitle("Error in Fitted Sigma X [mm]");
    fTree->Branch("FitSigmaYErr", &fNonPixel_FitSigmaYErr, "FitSigmaYErr/D")->SetTitle("Error in Fitted Sigma Y [mm]");
    fTree->Branch("FitThetaErr", &fNonPixel_FitThetaErr, "FitThetaErr/D")->SetTitle("Error in Fitted Rotation Angle [rad]");
    fTree->Branch("FitOffsetErr", &fNonPixel_FitOffsetErr, "FitOffsetErr/D")->SetTitle("Error in Fitted Offset");
    
    fTree->Branch("FitChi2", &fNonPixel_FitChi2, "FitChi2/D")->SetTitle("Fit Chi-squared Value");
    fTree->Branch("FitNDF", &fNonPixel_FitNDF, "FitNDF/D")->SetTitle("Fit Number of Degrees of Freedom");
    fTree->Branch("FitChi2red", &fNonPixel_FitChi2red, "FitChi2red/D")->SetTitle("Reduced Chi-squared (chi2red/NDF)");
    fTree->Branch("FitPp", &fNonPixel_FitPp, "FitPp/D")->SetTitle("Fit Probability (P-value)");
    fTree->Branch("FitNPoints", &fNonPixel_FitNPoints, "FitNPoints/I")->SetTitle("Number of Points Used in Fit");
    fTree->Branch("FitResidualMean", &fNonPixel_FitResidualMean, "FitResidualMean/D")->SetTitle("Mean of Fit Residuals");
    fTree->Branch("FitResidualStd", &fNonPixel_FitResidualStd, "FitResidualStd/D")->SetTitle("Standard Deviation of Fit Residuals");
    
    // Add branches for enhanced robustness metrics
    fTree->Branch("FitConstraintsSatisfied", &fNonPixel_FitConstraintsSatisfied, "FitConstraintsSatisfied/O")->SetTitle("Whether Geometric Constraints were Satisfied (non-pixel hits only, false for pixel hits)");
    
    // Add convenient alias branches for Gaussian center coordinates and distance calculation
    fTree->Branch("GaussTrueDistance", &fNonPixel_GaussTrueDistance, "GaussTrueDistance/D")->SetTitle("Distance from Gaussian Center to True Position [mm]");
    
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
                // Read from grid_parameters.txt file to get the final correct values
                std::ifstream gridFile("grid_parameters.txt");
                if (gridFile.is_open()) {
                    fRootFile->cd();
                    
                    G4double pixelSize, pixelSpacing, pixelCornerOffset, detSize;
                    G4int numBlocksPerSide;
                    
                    if (gridFile >> pixelSize >> pixelSpacing >> pixelCornerOffset >> detSize >> numBlocksPerSide) {
                        // Create TNamed objects to store grid parameters as metadata
                        TNamed *pixelSizeMeta = new TNamed("GridPixelSize", Form("%.6f", pixelSize));
                        TNamed *pixelSpacingMeta = new TNamed("GridPixelSpacing", Form("%.6f", pixelSpacing));  
                        TNamed *pixelCornerOffsetMeta = new TNamed("GridPixelCornerOffset", Form("%.6f", pixelCornerOffset));
                        TNamed *detSizeMeta = new TNamed("GridDetectorSize", Form("%.6f", detSize));
                        TNamed *numBlocksMeta = new TNamed("GridNumBlocksPerSide", Form("%d", numBlocksPerSide));
                        
                        // Write metadata to the ROOT file
                        pixelSizeMeta->Write();
                        pixelSpacingMeta->Write();
                        pixelCornerOffsetMeta->Write();
                        detSizeMeta->Write();
                        numBlocksMeta->Write();
                        
                        G4cout << "Saved detector grid metadata to ROOT file from grid_parameters.txt" << G4endl;
                        G4cout << "  Final parameters used: " << pixelSize << ", " << pixelSpacing 
                               << ", " << pixelCornerOffset << ", " << detSize << ", " << numBlocksPerSide << G4endl;
                    } else {
                        G4cerr << "Could not read grid parameters from file" << G4endl;
                    }
                    gridFile.close();
                } else if (fGridPixelSize > 0) {  // Fallback to member variables
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
                    
                    G4cout << "Saved detector grid metadata to ROOT file (fallback)" << G4endl;
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
                            // Read from grid_parameters.txt file to get the final correct values
                            std::ifstream gridFile("grid_parameters.txt");
                            if (gridFile.is_open()) {
                                G4double pixelSize, pixelSpacing, pixelCornerOffset, detSize;
                                G4int numBlocksPerSide;
                                
                                if (gridFile >> pixelSize >> pixelSpacing >> pixelCornerOffset >> detSize >> numBlocksPerSide) {
                                    TNamed *pixelSizeMeta = new TNamed("GridPixelSize", Form("%.6f", pixelSize));
                                    TNamed *pixelSpacingMeta = new TNamed("GridPixelSpacing", Form("%.6f", pixelSpacing));  
                                    TNamed *pixelCornerOffsetMeta = new TNamed("GridPixelCornerOffset", Form("%.6f", pixelCornerOffset));
                                    TNamed *detSizeMeta = new TNamed("GridDetectorSize", Form("%.6f", detSize));
                                    TNamed *numBlocksMeta = new TNamed("GridNumBlocksPerSide", Form("%d", numBlocksPerSide));
                                    
                                    // Write metadata to the merged ROOT file
                                    pixelSizeMeta->Write();
                                    pixelSpacingMeta->Write();
                                    pixelCornerOffsetMeta->Write();
                                    detSizeMeta->Write();
                                    numBlocksMeta->Write();
                                    
                                    G4cout << "Saved detector grid metadata to merged ROOT file from grid_parameters.txt" << G4endl;
                                    G4cout << "  Final parameters used: " << pixelSize << ", " << pixelSpacing 
                                           << ", " << pixelCornerOffset << ", " << detSize << ", " << numBlocksPerSide << G4endl;
                                } else {
                                    G4cerr << "Could not read grid parameters from file" << G4endl;
                                }
                                gridFile.close();
                            } else if (fGridPixelSize > 0) {  // Fallback to member variables
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
                                
                                G4cout << "Saved detector grid metadata to merged ROOT file (fallback)" << G4endl;
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
    fPixelDist = distance;
}

void RunAction::SetPixelAlpha(G4double alpha)
{
    // Store the angular size of the pixel from hit position (in degrees)
    fPixelHit_PixelAlpha = alpha;
}

void RunAction::SetPixelHitInfo(G4bool hit, G4double distanceToPixelCenter)
{
    // Store pixel hit information
    fIsPixelHit = hit;
    fDistanceToPixelCenter = distanceToPixelCenter;
}

void RunAction::SetPixelClassification(G4bool isWithinD0, G4double distanceToPixelCenter)
{
    // Store pixel classification based on D0 threshold
    fIsWithinD0 = isWithinD0;
    fDistanceToPixelCenter = distanceToPixelCenter;
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

void RunAction::SetParticleInfo(G4int eventID, G4double initialEnergy, G4double finalEnergy, 
                               G4double momentum, const G4String& particleName)
{
    // Store particle information
    fEventID = eventID;
    fInitialEnergy = initialEnergy;
    fFinalEnergy = finalEnergy;
    fMomentum = momentum;
    fParticleName = std::string(particleName);  // Convert G4String to std::string for ROOT
}

void RunAction::SetStepEnergyDeposition(const std::vector<G4double>& stepEdep,
                                       const std::vector<G4double>& stepZ,
                                       const std::vector<G4double>& stepTime)
{
    fStepEnergyDeposition = stepEdep;
    fStepZPositions = stepZ;
    fStepTimes = stepTime;
}

void RunAction::SetAllStepInfo(const std::vector<G4double>& stepEdep,
                              const std::vector<G4double>& stepZ,
                              const std::vector<G4double>& stepTime)
{
    fAllStepEnergyDeposition = stepEdep;
    fAllStepZPositions = stepZ;
    fAllStepTimes = stepTime;
}

void RunAction::SetGaussianFitResults(G4double amplitude, G4double x0, G4double y0,
                                     G4double sigma_x, G4double sigma_y, G4double theta, G4double offset,
                                     G4double amplitude_err, G4double x0_err, G4double y0_err,
                                     G4double sigma_x_err, G4double sigma_y_err, G4double theta_err, G4double offset_err,
                                     G4double chi2red, G4double ndf, G4double Pp,
                                     G4int n_points,
                                     G4double residual_mean, G4double residual_std,
                                     G4bool constraints_satisfied)
{
    // Store 3D Gaussian fit results
    fNonPixel_FitAmplitude = amplitude;
    fNonPixel_FitX0 = x0;
    fNonPixel_FitY0 = y0;
    fNonPixel_FitSigmaX = sigma_x;
    fNonPixel_FitSigmaY = sigma_y;
    fNonPixel_FitTheta = theta;
    fNonPixel_FitOffset = offset;
    
    fNonPixel_FitAmplitudeErr = amplitude_err;
    fNonPixel_FitX0Err = x0_err;
    fNonPixel_FitY0Err = y0_err;
    fNonPixel_FitSigmaXErr = sigma_x_err;
    fNonPixel_FitSigmaYErr = sigma_y_err;
    fNonPixel_FitThetaErr = theta_err;
    fNonPixel_FitOffsetErr = offset_err;
    
    fNonPixel_FitChi2 = chi2red;
    fNonPixel_FitNDF = ndf;
    fNonPixel_FitChi2red = (ndf > 0) ? chi2red / ndf : 0.0;
    fNonPixel_FitPp = Pp;
    fNonPixel_FitNPoints = n_points;
    fNonPixel_FitResidualMean = residual_mean;
    fNonPixel_FitResidualStd = residual_std;
    
    // Store enhanced robustness metrics
    fNonPixel_FitConstraintsSatisfied = constraints_satisfied;
    
    // Calculate distance from Gaussian center to true position ONLY if fit was successful
    if (constraints_satisfied) {
        fNonPixel_GaussTrueDistance = std::sqrt(std::pow(fNonPixel_FitX0 - fTrueX, 2) + std::pow(fNonPixel_FitY0 - fTrueY, 2));
    } else {
        // For failed fits or no fitting performed, set to NaN
        fNonPixel_GaussTrueDistance = std::numeric_limits<G4double>::quiet_NaN();
    }
}
