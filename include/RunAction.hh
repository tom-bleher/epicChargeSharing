#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class TFile;
class TTree;
class RootFileWriterHelper;

/**
 * @file RunAction.hh
 * @brief Manages run lifecycle, ROOT I/O, and metadata publication using helper components.
 */
class RunAction : public G4UserRunAction
{
public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;

    TFile* GetRootFile() const;
    TTree* GetTree() const;

    static void WaitForAllWorkersToComplete();
    static void SignalWorkerCompletion();
    static void ResetSynchronization();

    bool SafeWriteRootFile();
    bool ValidateRootFile(const G4String& filename, bool* hasEntries = nullptr);
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

    void SetDetectorGridParameters(G4double pixelSize,
                                   G4double pixelSpacing,
                                   G4double pixelCornerOffset,
                                   G4double detSize,
                                   G4int numBlocksPerSide);
    void SetNeighborhoodRadiusMeta(G4int radius) { fGridNeighborhoodRadius = radius; }

    void SetNeighborhoodPixelData(const std::vector<G4double>& xs,
                                  const std::vector<G4double>& ys,
                                  const std::vector<G4int>& ids);

    void FillTree();

private:
    void RunPostProcessingFits();

    std::unique_ptr<RootFileWriterHelper> fRootWriter;
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
    std::vector<G4double> fNeighborhoodDistance;
    std::vector<G4double> fNeighborhoodAlpha;

    std::vector<G4double> fNeighborhoodPixelX;
    std::vector<G4double> fNeighborhoodPixelY;
    std::vector<G4int> fNeighborhoodPixelID;

    G4double fGridPixelSize;
    G4double fGridPixelSpacing;
    G4double fGridPixelCornerOffset;
    G4double fGridDetSize;
    G4int fGridNumBlocksPerSide;
    G4int fGridNeighborhoodRadius{0};

    std::vector<G4int> fGridPixelID;
    std::vector<G4double> fGridPixelX;
    std::vector<G4double> fGridPixelY;
};

#endif // RUNACTION_HH
