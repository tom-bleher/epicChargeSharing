#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <vector>
#include <string>

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "G4Threading.hh"
#include <mutex>
#include <atomic>
#include <condition_variable>

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
    
    // Thread synchronization for ROOT file operations
    static void WaitForAllWorkersToComplete();
    static void SignalWorkerCompletion();
    static void ResetSynchronization();
    
    // Safe ROOT file operations
    bool SafeWriteRootFile();
    bool ValidateRootFile(const G4String& filename);
    void CleanupRootObjects();
    
    // Variables for the branch (edep [MeV], positions [mm])
    void SetEventData(G4double edep, G4double x, G4double y, G4double z);
    
    // Method to set initial particle gun position [mm]
    void SetInitialPos(G4double x, G4double y, G4double z);
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPos(G4double x, G4double y);
    
    // Method to set initial particle energy [MeV]
    void SetInitialEnergy(G4double energy);

    // Method to set initial particle name
    void SetInitialParticleName(const G4String& name);
    
    // Method to set pixel hit status
    void SetPixelHitStatus(G4bool isPixelHit);
    
    // Method to set pixel classification data (hit status and delta values)
    void SetPixelClassification(G4bool isWithinD0, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY);
    
    // Method to set neighborhood (9x9) grid angle data for non-pixel hits
    void SetNeighborhoodGridData(const std::vector<G4double>& angles);
    
    // Method to set neighborhood (9x9) grid charge sharing data for non-pixel hits
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& distances,
                          const std::vector<G4double>& chargeValues,
                          const std::vector<G4double>& chargeCoulombs);
    
    // Method to set detector grid parameters for saving to ROOT
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    
    void FillTree();

    // Method to set hit purity tracking data
    void SetHitPurityData(G4bool pureSiliconHit, G4bool chargeCalculationEnabled);
 
 private:

    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Thread synchronization for robust file operations
    static std::atomic<int> fWorkersCompleted;
    static std::atomic<int> fTotalWorkers;
    static std::condition_variable fWorkerCompletionCV;
    static std::mutex fSyncMutex;
    static std::atomic<bool> fAllWorkersCompleted;
    

    // Initial particle gun position
    G4double fInitialX, fInitialY, fInitialZ;

    // =============================================
    // HITS DATA VARIABLES
    // =============================================
    G4double fTrueX;   // True Hit pos X [mm]
    G4double fTrueY;   // True Hit pos Y [mm]
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fEdep;   // Energy depositionit [MeV]
    G4double fPixelTrueDeltaX; // Delta X from pixel center to true pos [mm] (x_pixel - x_true)
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to true pos [mm] (y_pixel - y_true)

    // Legacy variables that may still be used
    G4bool fIsPixelHit;  // True if hit is on pixel OR distance <= D0
    
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    std::vector<G4double> fNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4double> fNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels

    // Variables for particle information (reduced set)
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    G4String fInitialParticleName;  // Name of the initial particle
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;

    // Hit purity tracking variables
    G4bool fPureSiliconHit;
    G4bool fChargeCalculationEnabled;
};

#endif