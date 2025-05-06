#include "RunAction.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>

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
  fInitZ(0)
{ 
}

RunAction::~RunAction()
{
  // File will be closed in EndOfRunAction
  // No need to delete here as it could cause double deletion
}

void RunAction::BeginOfRunAction(const G4Run*)
{ 
  // Create ROOT file and tree with error handling
  try {
    fRootFile = new TFile("epicToyOutput.root", "RECREATE");
    if (!fRootFile || fRootFile->IsZombie()) {
      G4cerr << "Error creating ROOT file epicToyOutput.root!" << G4endl;
      delete fRootFile;
      fRootFile = nullptr;
      return;
    }
    
    fTree = new TTree("Hits", "Energy Deposits");
    if (!fTree) {
      G4cerr << "Error creating ROOT tree!" << G4endl;
      delete fRootFile;
      fRootFile = nullptr;
      return;
    }
    
    // Create branches for the tree
    fTree->Branch("Edep", &fEdep, "Edep/D");
    fTree->Branch("PosX", &fPosX, "PosX/D");
    fTree->Branch("PosY", &fPosY, "PosY/D");
    fTree->Branch("PosZ", &fPosZ, "PosZ/D");
    
    // Add branches for initial particle position
    fTree->Branch("InitX", &fInitX, "InitX/D");
    fTree->Branch("InitY", &fInitY, "InitY/D");
    fTree->Branch("InitZ", &fInitZ, "InitZ/D");
    
    G4cout << "Created ROOT file and tree successfully" << G4endl;
  }
  catch (std::exception& e) {
    G4cerr << "Exception in BeginOfRunAction: " << e.what() << G4endl;
    if (fRootFile) {
      delete fRootFile;
      fRootFile = nullptr;
    }
  }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  G4int nofEvents = run->GetNumberOfEvent();
  
  if (fRootFile && fTree && nofEvents > 0) {
    try {
      G4cout << "Writing ROOT file with " << fTree->GetEntries() 
             << " entries from " << nofEvents << " events" << G4endl;
      
      // Write tree to file and close file
      fRootFile->cd();
      fTree->Write();
      fRootFile->Close();
      
      G4cout << "Run ended. Data saved to epicToyOutput.root" << G4endl;
    }
    catch (std::exception& e) {
      G4cerr << "Exception in EndOfRunAction: " << e.what() << G4endl;
    }
  }
  
  // Clean up ROOT objects
  if (fRootFile) {
    delete fRootFile;
    fRootFile = nullptr;
    fTree = nullptr; // Tree is owned by file, no need to delete separately
  }
}