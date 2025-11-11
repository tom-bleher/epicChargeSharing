#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "ChargeSharingCalculator.hh"
#include "Constants.hh"
#include "G4Run.hh"
#include "G4UserRunAction.hh"
#include "globals.hh"
#include "internal/NeighborhoodBuffer.hh"

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
        std::span<const ChargeSharingCalculator::Result::NeighborCell> neighborCells;
        std::span<const G4double> neighborChargesNew;
        std::span<const G4double> neighborChargesFinal;
        std::span<const G4double> fullFi;
        std::span<const G4double> fullQi;
        std::span<const G4double> fullQn;
        std::span<const G4double> fullQf;
        std::span<const G4double> fullDistance;
        std::span<const G4double> fullAlpha;
        std::span<const G4double> fullPixelX;
        std::span<const G4double> fullPixelY;
        ChargeSharingCalculator::GridGeom geometry;
        ChargeSharingCalculator::HitInfo hit;
        ChargeSharingCalculator::ChargeMode mode{ChargeSharingCalculator::ChargeMode::Patch};
        ChargeSharingCalculator::PatchInfo patchInfo;
        G4int fullGridRows{0};
        G4int fullGridCols{0};
        G4int nearestPixelI{-1};
        G4int nearestPixelJ{-1};
        G4int nearestPixelGlobalId{-1};
        G4int totalGridCells{0};
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
    void SetGridPixelCenters(const std::vector<G4ThreeVector>& centers);
    void ConfigureFullFractionBranch(G4bool enable);

    void FillTree(const EventRecord& record);

private:
    struct ThreadContext
    {
        G4bool multithreaded{false};
        G4bool worker{false};
        G4bool master{false};
        G4int runId{-1};
        G4int threadId{-1};
        G4int totalWorkers{0};
    };

    ThreadContext BuildThreadContext(const G4Run* run) const;
    void LogBeginRun(const ThreadContext& context) const;
    G4String DetermineOutputFileName(const ThreadContext& context) const;
    void InitializeRootOutputs(const ThreadContext& context, const G4String& fileName);
    void ConfigureCoreBranches(TTree* tree);
    void ConfigureScalarBranches(TTree* tree);
    void ConfigureVectorBranches(TTree* tree);
    void ConfigureClassificationBranches(TTree* tree);
    void ConfigureNeighborhoodBranches(TTree* tree);

    void HandleWorkerEndOfRun(const ThreadContext& context, const G4Run* run);
    void HandleMasterEndOfRun(const ThreadContext& context, const G4Run* run);
    std::vector<G4String> CollectWorkerFileNames(G4int totalWorkers) const;
    std::vector<G4String> FilterExistingWorkerFiles(const std::vector<G4String>& workerFiles) const;
    bool MergeWorkerFilesAndPublishMetadata(const std::vector<G4String>& existingFiles);

    void UpdateSummaryScalars(const EventRecord& record);
    void PrepareNeighborhoodStorage(std::size_t requestedCells);
    void PopulateNeighborhoodFromRecord(const EventRecord& record);
    void PopulateFullFractionsFromRecord(const EventRecord& record);

    void RunPostProcessingFits();
    void EnsureBranchBuffersInitialized();
    bool EnsureFullFractionBuffer(G4int gridSide = -1);
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

    neighbor::Layout fNeighborhoodLayout;
    std::size_t fNeighborhoodCapacity{0};
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
    G4bool fStoreFullFractions{false};
    G4bool fFullFractionsBranchInitialized{false};

    std::vector<G4int> fGridPixelID;
    std::vector<G4double> fGridPixelX;
    std::vector<G4double> fGridPixelY;
    std::vector<G4double> fFullFi;
    std::vector<G4double> fFullQi;
    std::vector<G4double> fFullQn;
    std::vector<G4double> fFullQf;
    std::vector<G4double> fFullDistance;
    std::vector<G4double> fFullAlpha;
    std::vector<G4double> fFullPixelXGrid;
    std::vector<G4double> fFullPixelYGrid;
    G4int fNearestPixelI{-1};
    G4int fNearestPixelJ{-1};
    G4int fNearestPixelGlobalId{-1};
    G4int fFullGridSide{0};
};

#endif // RUNACTION_HH
