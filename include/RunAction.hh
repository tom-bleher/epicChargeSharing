#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include "Constants.hh"

#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

class TFile;
class TTree;
namespace runaction
{
class RootFileWriterHelper;
}

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

    struct EventSummaryData
    {
        G4double edep{0.0};
        G4double hitX{0.0};
        G4double hitY{0.0};
        G4double hitZ{0.0};
        G4double nearestPixelX{0.0};
        G4double nearestPixelY{0.0};
        G4double pixelTrueDeltaX{0.0};
        G4double pixelTrueDeltaY{0.0};
        G4bool firstContactIsPixel{false};
        G4bool geometricIsPixel{false};
        G4bool isPixelHitCombined{false};
    };

    struct EventRecord
    {
        EventSummaryData summary;
        std::span<const G4double> neighborFractions;
        std::span<const G4double> neighborCharges;
        std::span<const G4double> neighborChargesNew;
        std::span<const G4double> neighborChargesFinal;
        std::span<const G4double> neighborDistances;
        std::span<const G4double> neighborAlphas;
        std::span<const G4double> neighborPixelX;
        std::span<const G4double> neighborPixelY;
        std::span<const G4int> neighborPixelIds;
        G4bool includeDistanceAlpha{false};
    };

    void SetDetectorGridParameters(G4double pixelSize,
                                   G4double pixelSpacing,
                                   G4double pixelCornerOffset,
                                   G4double detSize,
                                   G4int numBlocksPerSide);
    void SetNeighborhoodRadiusMeta(G4int radius);
    void SetChargeSharingMetadata(Constants::ChargeSharingModel model,
                                  G4double betaPerMicron,
                                  G4double pitch);
    void SetChargeSharingDistanceAlphaMeta(G4bool enabled);

    void FillTree(const EventRecord& record);

private:
    void RunPostProcessingFits();
    void EnsureBranchBuffersInitialized();
    void EnsureVectorSized(std::vector<G4double>& vec, G4double initValue) const;
    void EnsureVectorSized(std::vector<G4int>& vec, G4int initValue) const;
    std::unique_lock<std::mutex> MakeTreeLock();
    void WriteMetadataToFile(TFile* file) const;
    std::vector<std::pair<std::string, std::string>> CollectMetadataEntries() const;

    std::unique_ptr<runaction::RootFileWriterHelper> fRootWriter;
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

    G4int fNeighborhoodCapacity{0};
    G4int fNeighborhoodActiveCells{0};

    G4double fGridPixelSize;
    G4double fGridPixelSpacing;
    G4double fGridPixelCornerOffset;
    G4double fGridDetSize;
    G4int fGridNumBlocksPerSide;
    G4int fGridNeighborhoodRadius{0};

    Constants::ChargeSharingModel fChargeSharingModel;
    G4double fChargeSharingBeta;
    G4double fChargeSharingPitch;
    G4bool fEmitDistanceAlphaMeta{false};

    std::vector<G4int> fGridPixelID;
    std::vector<G4double> fGridPixelX;
    std::vector<G4double> fGridPixelY;
};

#endif // RUNACTION_HH
