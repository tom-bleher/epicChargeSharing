#include "RunAction.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <sstream>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TError.h" // Added for gErrorIgnoreLevel and kWarning

// Initialize the static mutex
std::mutex RunAction::fRootMutex;

RunAction::RunAction()
: G4UserRunAction(),
  fRootFile(nullptr),
  fTree(nullptr),
  fEdep(0),
  fPosX(0),
  fPosY(0),
  fPosZ(0),
  fInitX(0),
  fInitY(0),
  fInitZ(0),
  fPixelX(0),
  fPixelY(0),
  fPixelZ(0)
{ 
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
    
    // Create branches for the tree with units in the titles
    fTree->Branch("Edep", &fEdep, "Edep/D")->SetTitle("Energy Deposit [MeV]");
    fTree->Branch("PosX", &fPosX, "PosX/D")->SetTitle("Position X [mm]");
    fTree->Branch("PosY", &fPosY, "PosY/D")->SetTitle("Position Y [mm]");
    fTree->Branch("PosZ", &fPosZ, "PosZ/D")->SetTitle("Position Z [mm]");
    
    // Add branches for initial particle position with units
    fTree->Branch("InitX", &fInitX, "InitX/D")->SetTitle("Initial X [mm]");
    fTree->Branch("InitY", &fInitY, "InitY/D")->SetTitle("Initial Y [mm]");
    fTree->Branch("InitZ", &fInitZ, "InitZ/D")->SetTitle("Initial Z [mm]");
    
    // Add branches for nearest pixel center position with units
    fTree->Branch("PixelX", &fPixelX, "PixelX/D")->SetTitle("Nearest Pixel X [mm]");
    fTree->Branch("PixelY", &fPixelY, "PixelY/D")->SetTitle("Nearest Pixel Y [mm]");
    fTree->Branch("PixelZ", &fPixelZ, "PixelZ/D")->SetTitle("Nearest Pixel Z [mm]");
    
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
  
  // Save the filename before closing the file
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
        
        // Write tree to file and close file
        if (fRootFile->IsOpen()) {
          fRootFile->cd();
          fTree->Write();
          fRootFile->Close();
        }
        
        G4cout << "Run ended. Data saved to " << fileName << G4endl;
      }
      catch (std::exception& e) {
        G4cerr << "Exception in EndOfRunAction when writing file: " << e.what() << G4endl;
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
      
      // Add files from worker threads
      for (G4int i = 0; i < nThreads; i++) {
        std::ostringstream oss;
        oss << "epicToyOutput_t" << i << ".root";
        G4String workerFile = oss.str();
        
        // Check if the file exists and is valid before adding
        TFile *testFile = TFile::Open(workerFile.c_str(), "READ");
        if (testFile && !testFile->IsZombie()) {
          // Make sure tree exists in the file
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
        // Create the merged output file
        TFile *mergedFile = TFile::Open("epicToyOutput.root", "RECREATE");
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
              
              G4cout << "Successfully merged " << validFiles << " files with " 
                      << mergedTree->GetEntries() << " total entries" << G4endl;
            }
          } else {
            G4cout << "No entries found in valid files to merge." << G4endl;
          }
          
          mergedFile->Close();
          delete mergedFile;
        } else {
          G4cout << "Error creating merged output file!" << G4endl;
          if (mergedFile) delete mergedFile;
        }
      } else {
        G4cout << "No valid worker files found to merge." << G4endl;
      }
    }
    catch (std::exception& e) {
      G4cerr << "Exception during file merging: " << e.what() << G4endl;
    }
  }
}