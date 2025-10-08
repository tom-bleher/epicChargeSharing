/**
 * @file RunAction.hh
 * @brief Declares `RunAction`, which manages ROOT I/O, thread-safe file writes/merge,
 *        and per-run metadata publication for detector grid configuration.
 */
#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <condition_variable>

class TFile;
class TTree;

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;

    TFile* GetRootFile() const { return fRootFile; }
    TTree* GetTree() const { return fTree; }
    
    static void WaitForAllWorkersToComplete();
    static void SignalWorkerCompletion();
    static void ResetSynchronization();
    
    bool SafeWriteRootFile();
    bool ValidateRootFile(const G4String& filename);
    void CleanupRootObjects();
    
    void SetEventData(G4double edep, G4double x, G4double y, G4double z);
    
    void SetNearestPixelPos(G4double x, G4double y) { fPixelX = x; fPixelY = y; }
    
    void SetFirstContactIsPixel(G4bool v) { fFirstContactIsPixel = v; }
    void SetGeometricIsPixel(G4bool v) { fGeometricIsPixel = v; }
    void SetIsPixelHitCombined(G4bool v) { fIsPixelHit = v; }
    
    void SetPixelClassification(G4bool isPixelHit, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY);
    
    
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& chargeCoulombs);
    void SetNeighborhoodChargeNewData(const std::vector<G4double>& chargeCoulombsNew);
    void SetNeighborhoodChargeFinalData(const std::vector<G4double>& chargeCoulombsFinal);
    void SetNeighborhoodDistanceAlphaData(const std::vector<G4double>& distances,
                                          const std::vector<G4double>& alphas);
    
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    void SetNeighborhoodRadiusMeta(G4int radius) { fGridNeighborhoodRadius = radius; }
    
    // Neighborhood pixel geometry (per event)
    void SetNeighborhoodPixelData(const std::vector<G4double>& xs,
                                  const std::vector<G4double>& ys,
                                  const std::vector<G4int>& ids);
    
    void FillTree();
    
private:
    TFile* fRootFile;
    TTree* fTree;
    
    static std::mutex fRootMutex;
    
    static std::atomic<int> fWorkersCompleted;
    static std::atomic<int> fTotalWorkers;
    static std::condition_variable fWorkerCompletionCV;
    static std::mutex fSyncMutex;
    static std::atomic<bool> fAllWorkersCompleted;
    std::mutex fTreeMutex;
    
    G4double fTrueX;
    G4double fTrueY;
    G4double fPixelX;
    G4double fPixelY;
    G4double fEdep;
    G4double fPixelTrueDeltaX;
    G4double fPixelTrueDeltaY;
    
    G4bool fFirstContactIsPixel{false};
    G4bool fGeometricIsPixel{false};
    G4bool fIsPixelHit{false};
    
    std::vector<G4double> fNeighborhoodChargeFractions;
    std::vector<G4double> fNeighborhoodCharge;
    std::vector<G4double> fNeighborhoodChargeNew;
    std::vector<G4double> fNeighborhoodChargeFinal;
    // Derived per-neighborhood metrics used in charge sharing
    std::vector<G4double> fNeighborhoodDistance; // mm, distance from hit to each neighborhood pixel center
    std::vector<G4double> fNeighborhoodAlpha;    // rad, subtended angle used in weighting
    
    // Per–event neighborhood pixel geometry
    std::vector<G4double> fNeighborhoodPixelX;
    std::vector<G4double> fNeighborhoodPixelY;
    std::vector<G4int>    fNeighborhoodPixelID;   // flattened row-major IDs; -1 for OOB
    
    G4double fGridPixelSize;
    G4double fGridPixelSpacing;  
    G4double fGridPixelCornerOffset;
    G4double fGridDetSize;
    G4int fGridNumBlocksPerSide;
    G4int fGridNeighborhoodRadius{0};
    
    // Full-grid pixel IDs (size = N^2, row-major 0..N^2-1)
    std::vector<G4int> fGridPixelID;
    // Full-grid pixel centers
    std::vector<G4double> fGridPixelX; // mm, length N^2
    std::vector<G4double> fGridPixelY; // mm, length N^2

    // Classification flags written to ROOT branches per igor.txt data spec:
    // - first_contact_is_pixel: first entered volume is pixel-pad (logicalBlock)
    // - geometric_is_pixel: |x_hit-x_px|<=l/2 && |y_hit-y_px|<=l/2
    // - is_pixel_hit: OR of the above two

    // Neighborhood charge vectors (flattened row-major, size (2r+1)^2):
    // idx = (d_i + r) * (2r+1) + (d_j + r), for d_i,d_j in [-r, r]
    // F_i uses Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL for OOB cells; Q_i=0 there.
};

#endif // RUNACTION_HH
