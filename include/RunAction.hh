#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"

// ROOT includes
#include "TFile.h"
#include "TTree.h"

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    virtual ~RunAction();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    // Access methods for ROOT objects
    TFile* GetRootFile() const { return fRootFile; }
    TTree* GetTree() const { return fTree; }
    
    // Variables for the branch
    void SetEventData(G4double edep, G4double x, G4double y, G4double z) {
        fEdep = edep;
        fPosX = x;
        fPosY = y;
        fPosZ = z;
    }
    
    void FillTree() { if (fTree) fTree->Fill(); }

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Variables for data storage
    G4double fEdep;
    G4double fPosX;
    G4double fPosY;
    G4double fPosZ;
};

#endif