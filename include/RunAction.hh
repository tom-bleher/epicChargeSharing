#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <vector>

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
    
    // Method to set pixel alpha angle
    void SetPixelAlpha(G4double alpha);
    
    // Method to set pixel hit flag
    void SetPixelHit(G4bool hit);
    
    // Method to set neighborhood (9x9) grid angle data
    void SetNeighborhoodGridData(const std::vector<G4double>& angles, 
                        const std::vector<G4int>& pixelI, 
                        const std::vector<G4int>& pixelJ);
    
    // Method to set neighborhood (9x9) grid charge sharing data
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& distances,
                          const std::vector<G4double>& chargeValues,
                          const std::vector<G4double>& chargeCoulombs);
    
    // Method to set detector grid parameters for saving to ROOT
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    
    // Fill the ROOT tree with current event data
    void FillTree();

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Variables for data storage with fixed units
    G4double fEdep;   // Energy deposit [MeV]
    G4double fTrueX;   // True Hit position X [mm]
    G4double fTrueY;   // True Hit position Y [mm]
    G4double fTrueZ;   // True Hit position Z [mm]
    
    // Variables for initial particle gun position
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    
    // Variables for nearest pixel center position
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fPixelZ; // Nearest to hit pixel center Z [mm]
    
    // Variables for pixel mapping
    G4int fPixelI;    // Pixel index in X direction
    G4int fPixelJ;    // Pixel index in Y direction
    G4double fPixelDist; // Distance from hit to pixel center [mm]
    G4double fPixelAlpha; // Angular size of pixel from hit position [deg]
    G4bool fPixelHit;  // Flag to indicate if hit was on a pixel
    
    // Variables for neighborhood (9x9) grid angle data
    std::vector<G4double> fGridNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4int> fGridNeighborhoodPixelI;     // I indices of neighborhood grid pixels
    std::vector<G4int> fGridNeighborhoodPixelJ;     // J indices of neighborhood grid pixels
    
    // Variables for neighborhood (9x9) grid charge sharing data
    std::vector<G4double> fGridNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fGridNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fGridNeighborhoodChargeValues;        // Charge values for neighborhood grid pixels (electrons)
    std::vector<G4double> fGridNeighborhoodChargeCoulombs;       // Charge values in Coulombs for neighborhood grid pixels
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
};

#endif