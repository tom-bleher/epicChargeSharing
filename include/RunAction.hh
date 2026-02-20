/// \file RunAction.hh
/// \brief Definition of the ECS::RunAction class.
///
/// This file declares the RunAction class which manages run-level
/// operations including ROOT file I/O and metadata publication.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_RUN_ACTION_HH
#define ECS_RUN_ACTION_HH

#include "ChargeSharingCalculator.hh"
#include "Config.hh"
#include "EDM4hepIO.hh"
#include "G4Run.hh"
#include "G4UserRunAction.hh"
#include "globals.hh"
#include "NeighborhoodUtils.hh"
#include "RootIO.hh"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

class TFile;
class TTree;

// Forward declaration
namespace ECS {
class RootFileWriter;
}

namespace ECS {

/// \brief Run-level action for managing simulation runs and ROOT I/O.
///
/// This class handles:
/// - Run lifecycle management (begin/end actions)
/// - ROOT file creation and configuration
/// - Branch setup for all output data categories
/// - Thread synchronization for multithreaded operation
/// - Worker file merging after parallel execution
/// - Metadata publication to ROOT files
/// - Post-processing fit macro invocation
///
/// The class supports five categories of ROOT branches:
/// 1. Core: True position, hit position, energy deposit
/// 2. Scalar: Nearest pixel, reconstruction results
/// 3. Vector: Neighborhood charge arrays
/// 4. Classification: Pixel hit flags
/// 5. Full grid: Complete detector charge map (optional)
class RunAction : public G4UserRunAction {
public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;

    [[nodiscard]] TFile* GetRootFile() const;
    [[nodiscard]] TTree* GetTree() const;

    static void WaitForAllWorkersToComplete();
    static void SignalWorkerCompletion();
    static void ResetSynchronization();

    bool SafeWriteRootFile();
    bool ValidateRootFile(const G4String& filename, bool* hasEntries = nullptr);
    void CleanupRootObjects();

    /// Type aliases for event data - definitions in ECS/IO/EventDataTypes.hh
    using EventSummaryData = IO::EventSummaryData;
    using EventRecord = IO::EventRecord;

    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, G4double gridOffset, G4double detSize,
                                   G4int numBlocksPerSide);
    void SetNeighborhoodRadiusMeta(G4int radius);
    void SetPosReconMetadata(Constants::PosReconModel model, G4double betaPerMicron, G4double pitch);
    void SetChargeSharingDistanceAlphaMeta(G4bool enabled);
    void SetGridPixelCenters(const std::vector<G4ThreeVector>& centers);
    void ConfigureFullFractionBranch(G4bool enable);

    void FillTree(const EventRecord& record, std::uint64_t eventNumber = 0, G4int runNumber = 0);

    // EDM4hep output control
    void SetEDM4hepEnabled(G4bool enabled);
    [[nodiscard]] G4bool IsEDM4hepEnabled() const;

private:
    struct ThreadContext {
        G4bool multithreaded{false};
        G4bool worker{false};
        G4bool master{false};
        G4int runId{-1};
        G4int threadId{-1};
        G4int totalWorkers{0};
    };

    static ThreadContext BuildThreadContext(const G4Run* run);
    static void LogBeginRun(const ThreadContext& context);
    [[nodiscard]] static G4String DetermineOutputFileName(const ThreadContext& context);
    void InitializeRootOutputs(const ThreadContext& context, const G4String& fileName);
    void ConfigureCoreBranches(TTree* tree);

    void HandleWorkerEndOfRun(const ThreadContext& context, const G4Run* run);
    void HandleMasterEndOfRun(const ThreadContext& context, const G4Run* run);
    [[nodiscard]] static std::vector<G4String> CollectWorkerFileNames(G4int totalWorkers);
    [[nodiscard]] static std::vector<G4String> FilterExistingWorkerFiles(const std::vector<G4String>& workerFiles);
    bool MergeWorkerFilesAndPublishMetadata(const std::vector<G4String>& existingFiles);

    void UpdateSummaryScalars(const EventRecord& record);
    void PrepareNeighborhoodStorage(std::size_t requestedCells);
    void PopulateNeighborhoodFromRecord(const EventRecord& record);
    void PopulateFullFractionsFromRecord(const EventRecord& record);

    void RunPostProcessingFits();
    void EnsureBranchBuffersInitialized();
    bool EnsureFullFractionBuffer(G4int gridSide = -1);
    std::unique_lock<std::mutex> MakeTreeLock();
    void WriteMetadataToTree(TTree* tree) const;
    [[nodiscard]] ECS::IO::MetadataPublisher::EntryList CollectMetadataEntries() const;
    [[nodiscard]] IO::MetadataPublisher BuildMetadataPublisher() const;

    std::unique_ptr<RootFileWriter> fRootWriter;
    std::mutex fTreeMutex;

    // Helper classes for decomposed functionality
    IO::BranchConfigurator fBranchConfigurator;
    IO::MetadataPublisher fMetadataPublisher;
    IO::PostProcessingRunner fPostProcessingRunner;

    G4double fTrueX{0.0};
    G4double fTrueY{0.0};
    G4double fPixelX{0.0};
    G4double fPixelY{0.0};
    G4double fEdep{0.0};
    G4double fPixelTrueDeltaX{0.0};
    G4double fPixelTrueDeltaY{0.0};

    G4bool fFirstContactIsPixel{false};
    G4bool fGeometricIsPixel{false};
    G4bool fIsPixelHit{false};

    // Neighborhood fractions
    std::vector<G4double> fNeighborhoodChargeFractions;
    std::vector<G4double> fNeighborhoodChargeFractionsRow;
    std::vector<G4double> fNeighborhoodChargeFractionsCol;
    std::vector<G4double> fNeighborhoodChargeFractionsBlock;
    // Neighborhood charges
    std::vector<G4double> fNeighborhoodCharge;
    std::vector<G4double> fNeighborhoodChargeNew;
    std::vector<G4double> fNeighborhoodChargeFinal;
    // Row-mode charges
    std::vector<G4double> fNeighborhoodChargeRow;
    std::vector<G4double> fNeighborhoodChargeNewRow;
    std::vector<G4double> fNeighborhoodChargeFinalRow;
    // Col-mode charges
    std::vector<G4double> fNeighborhoodChargeCol;
    std::vector<G4double> fNeighborhoodChargeNewCol;
    std::vector<G4double> fNeighborhoodChargeFinalCol;
    // Block-mode charges
    std::vector<G4double> fNeighborhoodChargeBlock;
    std::vector<G4double> fNeighborhoodChargeNewBlock;
    std::vector<G4double> fNeighborhoodChargeFinalBlock;
    // Common
    std::vector<G4double> fNeighborhoodDistance;
    std::vector<G4double> fNeighborhoodAlpha;

    std::vector<G4double> fNeighborhoodPixelX;
    std::vector<G4double> fNeighborhoodPixelY;
    std::vector<G4int> fNeighborhoodPixelID;

    NeighborhoodLayout fNeighborhoodLayout;
    std::size_t fNeighborhoodCapacity{0};
    G4int fNeighborhoodActiveCells{0};

    G4double fGridPixelSize{0.0};
    G4double fGridPixelSpacing{0.0};
    G4double fGridOffset{0.0}; ///< DD4hep-style grid offset
    G4double fGridDetSize{0.0};
    G4int fGridNumBlocksPerSide{0};
    G4int fGridNeighborhoodRadius{0};

    Constants::PosReconModel fPosReconModel{Constants::POS_RECON_MODEL};
    Constants::ActivePixelMode fActivePixelMode{Constants::ACTIVE_PIXEL_MODE};
    G4double fChargeSharingBeta{0.0};
    G4double fChargeSharingPitch{0.0};
    G4bool fEmitDistanceAlphaMeta{false};
    G4bool fStoreFullFractions{false};
    G4bool fFullFractionsBranchInitialized{false};

    std::vector<G4int> fGridPixelID;
    std::vector<G4double> fGridPixelX;
    std::vector<G4double> fGridPixelY;
    // Full grid fractions
    std::vector<G4double> fFullFi;
    std::vector<G4double> fFullFiRow;
    std::vector<G4double> fFullFiCol;
    std::vector<G4double> fFullFiBlock;
    // Full grid neighborhood charges
    std::vector<G4double> fFullQi;
    std::vector<G4double> fFullQn;
    std::vector<G4double> fFullQf;
    // Full grid row-mode charges
    std::vector<G4double> fFullQiRow;
    std::vector<G4double> fFullQnRow;
    std::vector<G4double> fFullQfRow;
    // Full grid col-mode charges
    std::vector<G4double> fFullQiCol;
    std::vector<G4double> fFullQnCol;
    std::vector<G4double> fFullQfCol;
    // Full grid block-mode charges
    std::vector<G4double> fFullQiBlock;
    std::vector<G4double> fFullQnBlock;
    std::vector<G4double> fFullQfBlock;
    // Full grid geometry
    std::vector<G4double> fFullDistance;
    std::vector<G4double> fFullAlpha;
    std::vector<G4double> fFullPixelXGrid;
    std::vector<G4double> fFullPixelYGrid;
    G4int fNearestPixelI{-1};
    G4int fNearestPixelJ{-1};
    G4int fNearestPixelGlobalId{-1};
    G4int fFullGridSide{0};

    // EDM4hep output support
    std::unique_ptr<IO::EDM4hepWriter> fEDM4hepWriter;
    IO::EDM4hepConfig fEDM4hepConfig;
    G4bool fWriteEDM4hep{false};
};

} // namespace ECS

// Backward compatibility alias
using RunAction = ECS::RunAction;

#endif // ECS_RUN_ACTION_HH
