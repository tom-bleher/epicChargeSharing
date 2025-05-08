#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "G4Threading.hh"
#include <mutex>

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
    
    // New method to set initial particle gun position
    void SetInitialPosition(G4double x, G4double y, G4double z) {
        fInitX = x;
        fInitY = y;
        fInitZ = z;
    }
    
    // New method to set nearest pixel position
    void SetNearestPixelPosition(G4double x, G4double y, G4double z) {
        fPixelX = x;
        fPixelY = y;
        fPixelZ = z;
    }
    
    void FillTree() { if (fTree) fTree->Fill(); }

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Variables for data storage
    G4double fEdep;
    G4double fPosX;
    G4double fPosY;
    G4double fPosZ;
    
    // New variables for initial particle gun position
    G4double fInitX;
    G4double fInitY;
    G4double fInitZ;
    
    // New variables for nearest pixel center position
    G4double fPixelX;
    G4double fPixelY;
    G4double fPixelZ;
};

#endif