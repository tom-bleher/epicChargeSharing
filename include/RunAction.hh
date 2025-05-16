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
    
    // Variables for the branch (edep [MeV], positions [mm])
    void SetEventData(G4double edep, G4double x, G4double y, G4double z);
    
    // Method to set initial particle gun position [mm]
    void SetInitialPosition(G4double x, G4double y, G4double z);
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPosition(G4double x, G4double y, G4double z);
    
    // Method to set pixel indices and distance
    void SetPixelIndices(G4int i, G4int j, G4double distance);
    
    // Fill the ROOT tree with current event data
    void FillTree();

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Variables for data storage with fixed units
    G4double fEdep;   // Energy deposit [MeV]
    G4double fPosX;   // Position X [mm]
    G4double fPosY;   // Position Y [mm]
    G4double fPosZ;   // Position Z [mm]
    
    // Variables for initial particle gun position
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    
    // Variables for nearest pixel center position
    G4double fPixelX; // Nearest pixel X [mm]
    G4double fPixelY; // Nearest pixel Y [mm]
    G4double fPixelZ; // Nearest pixel Z [mm]
    
    // Variables for pixel mapping
    G4int fPixelI;    // Pixel index in X direction
    G4int fPixelJ;    // Pixel index in Y direction
    G4double fPixelDist; // Distance from hit to pixel center [mm]
};

#endif