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
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPos(G4double x, G4double y);
    
    // Method to set initial particle energy [MeV]
    void SetInitialEnergy(G4double energy);
    
    // Method to set pixel hit status
    void SetPixelHitStatus(G4bool isPixelHit);
    // New: set first-contact flag, geometric flag, and combined flag
    void SetFirstContactIsPixel(G4bool v) { fFirstContactIsPixel = v; }
    void SetGeometricIsPixel(G4bool v) { fGeometricIsPixel = v; }
    void SetIsPixelHitCombined(G4bool v) { fIsPixelHit = v; }
    
    // Method to set pixel classification data (hit status and delta values)
    void SetPixelClassification(G4bool isPixelHit, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY);
    void SetNonPixelPadRadiusCheck(G4bool passed);
    
    // Method to set neighborhood (9x9) grid charge sharing data for non-pixel hits
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& chargeCoulombs);
    
    // Method to set detector grid parameters for saving to ROOT
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    
    // Fill the ROOT tree with current event data
    void FillTree();
    
    // Method to set scorer data from Multi-Functional Detector
    void SetScorerData(G4double energyDeposit);
    void SetScorerHitCount(G4int hitCount);
    
    // Method to set hit purity tracking data
    void SetHitPurityData(G4bool pureSiliconHit, G4bool aluminumContaminated, G4bool chargeCalculationEnabled);
 
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
    

    // =============================================
    // HITS DATA VARIABLES
    // =============================================
    G4double fTrueX;   // True Hit pos X [mm]
    G4double fTrueY;   // True Hit pos Y [mm]
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fEdep;   // Energy depositionit [MeV]
    G4double fPixelTrueDeltaX; // Delta X from pixel center to true pos [mm] (x_pixel - x_true)
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to true pos [mm] (y_pixel - y_true)

    // Pixel flags
    G4bool fFirstContactIsPixel{false};  // First boundary entry was pixel
    G4bool fGeometricIsPixel{false};     // Orthogonal radius test indicates pixel region
    G4bool fIsPixelHit{false};           // OR of first-contact and geometric tests
    
    // NON-PIXEL HIT DATA (not on pixel)
    std::vector<G4double> fNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels

    // Variables for particle information (reduced set)
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
    
    // Scorer data variables
    G4double fScorerEnergyDeposit;  // Energy deposit from Multi-Functional Detector [MeV]
    G4int    fScorerNhit;           // Hit count from Multi-Functional Detector

    
    // Hit purity tracking variables for Multi-Functional Detector validation
    G4bool fPureSiliconHit;         // True if hit is purely in silicon (no aluminum contamination)
    G4bool fAluminumContaminated;   // True if hit has aluminum contamination
    G4bool fChargeCalculationEnabled; // True if charge sharing calculation was enabled

    // QA: for non-pixel-pad hits, check |dx|>PIXEL_SIZE/2 and |dy|>PIXEL_SIZE/2
    G4bool fNonPixelPadRadiusCheck;
};

#endif