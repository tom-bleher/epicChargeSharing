#include "RunAction.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <sstream>
#include <vector>
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
  fPixelAlpha(0),
  fPixelHit(false),
  fGridPixelSize(0),
  fGridPixelSpacing(0),
  fGridPixelCornerOffset(0),
  fGridDetSize(0),
  fGridNumBlocksPerSide(0),
  fEventID(-1),
  fInitialEnergy(0),
  fFinalEnergy(0),
  fMomentum(0),
  fParticleName(""),
  fCreatorProcess(""),
  fGlobalTime(0),
  fLocalTime(0),
  fProperTime(0),
  fPhysicsProcess(""),
  fTrackID(-1),
  fParentID(-1),
  fStepNum(-1),
  fStepLen(0),
  fFitAmplitude(0),
  fFitX0(0),
  fFitY0(0),
  fFitSigmaX(0),
  fFitSigmaY(0),
  fFitTheta(0),
  fFitOffset(0),
  fFitAmplitudeErr(0),
  fFitX0Err(0),
  fFitY0Err(0),
  fFitSigmaXErr(0),
  fFitSigmaYErr(0),
  fFitThetaErr(0),
  fFitOffsetErr(0),
  fFitChi2(0),
  fFitNDF(0),
  fFitProb(0),
  fFitRSquared(0),
  fFitNPoints(0),
  fFitSuccessful(false),
  fFitResidualMean(0),
  fFitResidualStd(0)
{ 
  // Initialize neighborhood (9x9) grid vectors (they are automatically initialized empty)
  // Initialize trajectory vectors (they are automatically initialized empty)
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
    fTree->Branch("PixelDist", &fPixelDist, "PixelDist/D")->SetTitle("Distance to Pixel Center [mm]");
    fTree->Branch("PixelAlpha", &fPixelAlpha, "PixelAlpha/D")->SetTitle("Angular Size of Pixel [deg]");
    fTree->Branch("PixelHit", &fPixelHit, "PixelHit/O")->SetTitle("Hit on Pixel (Boolean)");
    
    // Load vector dictionaries for ROOT to properly handle std::vector branches
    gROOT->ProcessLine("#include <vector>");
    
    // Add branches for neighborhood (9x9) grid angle data
    fTree->Branch("GridNeighborhoodAngles", &fGridNeighborhoodAngles)->SetTitle("Angles from Hit to Neighborhood Grid Pixels [deg]");
    fTree->Branch("GridNeighborhoodPixelI", &fGridNeighborhoodPixelI)->SetTitle("I Indices of Neighborhood Grid Pixels");
    fTree->Branch("GridNeighborhoodPixelJ", &fGridNeighborhoodPixelJ)->SetTitle("J Indices of Neighborhood Grid Pixels");
    
    // Add branches for neighborhood (9x9) grid charge sharing data
    fTree->Branch("GridNeighborhoodChargeFractions", &fGridNeighborhoodChargeFractions)->SetTitle("Charge Fractions for Neighborhood Grid Pixels");
    fTree->Branch("GridNeighborhoodDistances", &fGridNeighborhoodDistances)->SetTitle("Distances from Hit to Neighborhood Grid Pixels [mm]");
    fTree->Branch("GridNeighborhoodChargeValues", &fGridNeighborhoodChargeValues)->SetTitle("Charge Values for Neighborhood Grid Pixels");
    fTree->Branch("GridNeighborhoodChargeCoulombs", &fGridNeighborhoodChargeCoulombs)->SetTitle("Charge Coulombs for Neighborhood Grid Pixels");
    
    // Add branches for particle information
    fTree->Branch("EventID", &fEventID, "EventID/I")->SetTitle("Event ID");
    fTree->Branch("InitialEnergy", &fInitialEnergy, "InitialEnergy/D")->SetTitle("Initial Particle Energy [MeV]");
    fTree->Branch("FinalEnergy", &fFinalEnergy, "FinalEnergy/D")->SetTitle("Final Particle Energy [MeV]");
    fTree->Branch("Momentum", &fMomentum, "Momentum/D")->SetTitle("Particle Momentum [MeV/c]");
    fTree->Branch("ParticleName", &fParticleName)->SetTitle("Particle Type Name");
    fTree->Branch("CreatorProcess", &fCreatorProcess)->SetTitle("Creator Process Name");
    
    // Add branches for timing information
    fTree->Branch("GlobalTime", &fGlobalTime, "GlobalTime/D")->SetTitle("Global Time [ns]");
    fTree->Branch("LocalTime", &fLocalTime, "LocalTime/D")->SetTitle("Local Time [ns]");
    fTree->Branch("ProperTime", &fProperTime, "ProperTime/D")->SetTitle("Proper Time [ns]");
    
    // Add branches for physics process information
    fTree->Branch("PhysicsProcess", &fPhysicsProcess)->SetTitle("Physics Process Name");
    fTree->Branch("TrackID", &fTrackID, "TrackID/I")->SetTitle("Track ID");
    fTree->Branch("ParentID", &fParentID, "ParentID/I")->SetTitle("Parent Track ID");
    fTree->Branch("StepNumber", &fStepNum, "StepNumber/I")->SetTitle("Step Number in Track");
    fTree->Branch("StepLength", &fStepLen, "StepLength/D")->SetTitle("Total Step Length [mm]");
    
    // Add branches for track trajectory information
    fTree->Branch("TrajectoryX", &fTrajectoryX)->SetTitle("X Positions Along Track [mm]");
    fTree->Branch("TrajectoryY", &fTrajectoryY)->SetTitle("Y Positions Along Track [mm]");
    fTree->Branch("TrajectoryZ", &fTrajectoryZ)->SetTitle("Z Positions Along Track [mm]");
    fTree->Branch("TrajectoryTime", &fTrajectoryTime)->SetTitle("Time at Each Trajectory Point [ns]");
    
    // Add branches for step-by-step energy deposition information
    fTree->Branch("StepEnergyDeposition", &fStepEdepVec)->SetTitle("Energy Deposited Per Step [MeV]");
    fTree->Branch("StepZPosition", &fStepZVec)->SetTitle("Z Position of Each Energy Deposit [mm]");
    fTree->Branch("StepTime", &fStepTimeVec)->SetTitle("Time of Each Energy Deposit [ns]");
    fTree->Branch("StepLengthVector", &fStepLenVec)->SetTitle("Length of Each Energy Depositing Step [mm]");
    fTree->Branch("StepNumberVector", &fStepNumVec)->SetTitle("Step Number for Each Energy Deposit");
    
    // Add branches for ALL step information (including non-energy depositing steps)
    fTree->Branch("AllStepEnergyDeposition", &fAllStepEdepVec)->SetTitle("Energy Deposited Per Step (All Steps) [MeV]");
    fTree->Branch("AllStepZPosition", &fAllStepZVec)->SetTitle("Z Position of Each Step [mm]");
    fTree->Branch("AllStepTime", &fAllStepTimeVec)->SetTitle("Time of Each Step [ns]");
    fTree->Branch("AllStepLengthVector", &fAllStepLenVec)->SetTitle("Length of Each Step [mm]");
    fTree->Branch("AllStepNumberVector", &fAllStepNumVec)->SetTitle("Step Number for Each Step");
    
    // Add branches for 3D Gaussian fit results
    fTree->Branch("FitAmplitude", &fFitAmplitude, "FitAmplitude/D")->SetTitle("Fitted Gaussian Amplitude");
    fTree->Branch("FitX0", &fFitX0, "FitX0/D")->SetTitle("Fitted Gaussian Center X [mm]");
    fTree->Branch("FitY0", &fFitY0, "FitY0/D")->SetTitle("Fitted Gaussian Center Y [mm]");
    fTree->Branch("FitSigmaX", &fFitSigmaX, "FitSigmaX/D")->SetTitle("Fitted Gaussian Sigma X [mm]");
    fTree->Branch("FitSigmaY", &fFitSigmaY, "FitSigmaY/D")->SetTitle("Fitted Gaussian Sigma Y [mm]");
    fTree->Branch("FitTheta", &fFitTheta, "FitTheta/D")->SetTitle("Fitted Gaussian Rotation Angle [rad]");
    fTree->Branch("FitOffset", &fFitOffset, "FitOffset/D")->SetTitle("Fitted Gaussian Offset");
    
    fTree->Branch("FitAmplitudeErr", &fFitAmplitudeErr, "FitAmplitudeErr/D")->SetTitle("Error in Fitted Amplitude");
    fTree->Branch("FitX0Err", &fFitX0Err, "FitX0Err/D")->SetTitle("Error in Fitted Center X [mm]");
    fTree->Branch("FitY0Err", &fFitY0Err, "FitY0Err/D")->SetTitle("Error in Fitted Center Y [mm]");
    fTree->Branch("FitSigmaXErr", &fFitSigmaXErr, "FitSigmaXErr/D")->SetTitle("Error in Fitted Sigma X [mm]");
    fTree->Branch("FitSigmaYErr", &fFitSigmaYErr, "FitSigmaYErr/D")->SetTitle("Error in Fitted Sigma Y [mm]");
    fTree->Branch("FitThetaErr", &fFitThetaErr, "FitThetaErr/D")->SetTitle("Error in Fitted Rotation Angle [rad]");
    fTree->Branch("FitOffsetErr", &fFitOffsetErr, "FitOffsetErr/D")->SetTitle("Error in Fitted Offset");
    
    fTree->Branch("FitChi2", &fFitChi2, "FitChi2/D")->SetTitle("Fit Chi-squared Value");
    fTree->Branch("FitNDF", &fFitNDF, "FitNDF/D")->SetTitle("Fit Number of Degrees of Freedom");
    fTree->Branch("FitProb", &fFitProb, "FitProb/D")->SetTitle("Fit Probability");
    fTree->Branch("FitRSquared", &fFitRSquared, "FitRSquared/D")->SetTitle("Fit R-squared Value");
    fTree->Branch("FitNPoints", &fFitNPoints, "FitNPoints/I")->SetTitle("Number of Points Used in Fit");
    fTree->Branch("FitSuccessful", &fFitSuccessful, "FitSuccessful/O")->SetTitle("Whether Fit was Successful");
    fTree->Branch("FitResidualMean", &fFitResidualMean, "FitResidualMean/D")->SetTitle("Mean of Fit Residuals");
    fTree->Branch("FitResidualStd", &fFitResidualStd, "FitResidualStd/D")->SetTitle("Standard Deviation of Fit Residuals");
    
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
    fPixelAlpha = alpha;
}

void RunAction::SetPixelHit(G4bool hit)
{
    // Store whether the hit was on a pixel
    fPixelHit = hit;
}

void RunAction::SetNeighborhoodGridData(const std::vector<G4double>& angles, 
                               const std::vector<G4int>& pixelI, 
                               const std::vector<G4int>& pixelJ)
{
    // Store the neighborhood (9x9) grid angle data
    fGridNeighborhoodAngles = angles;
    fGridNeighborhoodPixelI = pixelI;
    fGridNeighborhoodPixelJ = pixelJ;
}

void RunAction::SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                                 const std::vector<G4double>& distances,
                                 const std::vector<G4double>& chargeValues,
                                 const std::vector<G4double>& chargeCoulombs)
{
    // Store the neighborhood (9x9) grid charge sharing data
    fGridNeighborhoodChargeFractions = chargeFractions;
    fGridNeighborhoodDistances = distances;
    fGridNeighborhoodChargeValues = chargeValues;
    fGridNeighborhoodChargeCoulombs = chargeCoulombs;
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
                               G4double momentum, const G4String& particleName, 
                               const G4String& creatorProcess)
{
    // Store particle information
    fEventID = eventID;
    fInitialEnergy = initialEnergy;
    fFinalEnergy = finalEnergy;
    fMomentum = momentum;
    fParticleName = particleName;
    fCreatorProcess = creatorProcess;
}

void RunAction::SetTimingInfo(G4double globalTime, G4double localTime, G4double properTime)
{
    // Store timing information (convert to ns if needed)
    fGlobalTime = globalTime / CLHEP::ns;  // Convert to ns
    fLocalTime = localTime / CLHEP::ns;    // Convert to ns
    fProperTime = properTime / CLHEP::ns;  // Convert to ns
}

void RunAction::SetPhysicsInfo(const G4String& physicsProcess, G4int trackID, G4int parentID, 
                              G4int stepNumber, G4double stepLength)
{
    // Store physics process information
    fPhysicsProcess = physicsProcess;
    fTrackID = trackID;
    fParentID = parentID;
    fStepNum = stepNumber;
    fStepLen = stepLength;
}

void RunAction::SetTrajectoryInfo(const std::vector<G4double>& trackX, 
                                 const std::vector<G4double>& trackY, 
                                 const std::vector<G4double>& trackZ,
                                 const std::vector<G4double>& trackTime)
{
    // Store trajectory information
    fTrajectoryX = trackX;
    fTrajectoryY = trackY;
    fTrajectoryZ = trackZ;
    fTrajectoryTime = trackTime;
}

void RunAction::SetStepEnergyDeposition(const std::vector<G4double>& stepEdep,
                                       const std::vector<G4double>& stepZ,
                                       const std::vector<G4double>& stepTime,
                                       const std::vector<G4double>& stepLength,
                                       const std::vector<G4int>& stepNumber)
{
    // Store step-by-step energy deposition information
    fStepEdepVec = stepEdep;
    fStepZVec = stepZ;
    fStepTimeVec = stepTime;
    fStepLenVec = stepLength;
    fStepNumVec = stepNumber;
    
    // Store step-by-step energy deposition information
}

void RunAction::SetAllStepInfo(const std::vector<G4double>& stepEdep,
                              const std::vector<G4double>& stepZ,
                              const std::vector<G4double>& stepTime,
                              const std::vector<G4double>& stepLength,
                              const std::vector<G4int>& stepNumber)
{
    // Store ALL step information (including non-energy depositing steps)
    fAllStepEdepVec = stepEdep;
    fAllStepZVec = stepZ;
    fAllStepTimeVec = stepTime;
    fAllStepLenVec = stepLength;
    fAllStepNumVec = stepNumber;
}

void RunAction::SetGaussianFitResults(G4double amplitude, G4double x0, G4double y0,
                                     G4double sigma_x, G4double sigma_y, G4double theta, G4double offset,
                                     G4double amplitude_err, G4double x0_err, G4double y0_err,
                                     G4double sigma_x_err, G4double sigma_y_err, G4double theta_err, G4double offset_err,
                                     G4double chi2, G4double ndf, G4double prob, G4double r_squared,
                                     G4int n_points, G4bool fit_successful,
                                     G4double residual_mean, G4double residual_std)
{
    // Store 3D Gaussian fit results
    fFitAmplitude = amplitude;
    fFitX0 = x0;
    fFitY0 = y0;
    fFitSigmaX = sigma_x;
    fFitSigmaY = sigma_y;
    fFitTheta = theta;
    fFitOffset = offset;
    
    fFitAmplitudeErr = amplitude_err;
    fFitX0Err = x0_err;
    fFitY0Err = y0_err;
    fFitSigmaXErr = sigma_x_err;
    fFitSigmaYErr = sigma_y_err;
    fFitThetaErr = theta_err;
    fFitOffsetErr = offset_err;
    
    fFitChi2 = chi2;
    fFitNDF = ndf;
    fFitProb = prob;
    fFitRSquared = r_squared;
    fFitNPoints = n_points;
    fFitSuccessful = fit_successful;
    fFitResidualMean = residual_mean;
    fFitResidualStd = residual_std;
}
