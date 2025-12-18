/// \file RootIO.hh
/// \brief ROOT I/O classes for simulation output.
///
/// This file consolidates all ROOT I/O functionality:
/// - BranchConfigurator: Sets up TTree branches
/// - TreeFiller: Populates branches from event data
/// - MetadataPublisher: Writes simulation metadata
/// - PostProcessingRunner: Executes analysis macros
/// - EventDataTypes: Data transfer structures
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_ROOT_IO_HH
#define ECS_ROOT_IO_HH

#include "ChargeSharingCalculator.hh"
#include "Config.hh"
#include "NeighborhoodUtils.hh"
#include "globals.hh"

#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

class TFile;
class TTree;

namespace ECS::IO {

// ============================================================================
// Event Data Types
// ============================================================================

/// \brief Summary of scalar event data.
struct EventSummaryData {
    G4double edep{0.0};
    G4double hitX{0.0};
    G4double hitY{0.0};
    G4double hitZ{0.0};
    G4double nearestPixelX{0.0};
    G4double nearestPixelY{0.0};
    G4double pixelTrueDeltaX{0.0};
    G4double pixelTrueDeltaY{0.0};
    G4double reconDPCX{0.0};
    G4double reconDPCY{0.0};
    G4double reconDPCTrueDeltaX{0.0};
    G4double reconDPCTrueDeltaY{0.0};
    G4bool firstContactIsPixel{false};
    G4bool geometricIsPixel{false};
    G4bool isPixelHitCombined{false};
};

/// \brief Complete event record for ROOT tree filling.
struct EventRecord {
    EventSummaryData summary;
    std::span<const ChargeSharingCalculator::Result::NeighborCell> neighborCells;
    std::span<const ChargeSharingCalculator::Result::NeighborCell> chargeBlock;
    std::span<const G4double> neighborChargesNew;
    std::span<const G4double> neighborChargesFinal;
    // Row-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesNewRow;
    std::span<const G4double> neighborChargesFinalRow;
    // Col-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesNewCol;
    std::span<const G4double> neighborChargesFinalCol;
    // Block-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesNewBlock;
    std::span<const G4double> neighborChargesFinalBlock;
    // Full grid fractions
    std::span<const G4double> fullFi;
    std::span<const G4double> fullFiRow;    ///< Signal fractions with row denominator
    std::span<const G4double> fullFiCol;    ///< Signal fractions with column denominator
    std::span<const G4double> fullFiBlock;  ///< Signal fractions with 4-pad block denominator
    // Full grid neighborhood-mode charges
    std::span<const G4double> fullQi;
    std::span<const G4double> fullQn;
    std::span<const G4double> fullQf;
    // Full grid row-mode charges
    std::span<const G4double> fullQiRow;
    std::span<const G4double> fullQnRow;
    std::span<const G4double> fullQfRow;
    // Full grid col-mode charges
    std::span<const G4double> fullQiCol;
    std::span<const G4double> fullQnCol;
    std::span<const G4double> fullQfCol;
    // Full grid block-mode charges
    std::span<const G4double> fullQiBlock;
    std::span<const G4double> fullQnBlock;
    std::span<const G4double> fullQfBlock;
    // Full grid geometry
    std::span<const G4double> fullDistance;
    std::span<const G4double> fullAlpha;
    std::span<const G4double> fullPixelX;
    std::span<const G4double> fullPixelY;
    ChargeSharingCalculator::PixelGridGeometry geometry;
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

// ============================================================================
// Branch Configurator
// ============================================================================

/// \brief Configures ROOT TTree branches for simulation output.
class BranchConfigurator {
public:
    struct ScalarBuffers {
        G4double* trueX{nullptr};
        G4double* trueY{nullptr};
        G4double* pixelX{nullptr};
        G4double* pixelY{nullptr};
        G4double* edep{nullptr};
        G4double* pixelTrueDeltaX{nullptr};
        G4double* pixelTrueDeltaY{nullptr};
        G4double* reconDPCX{nullptr};
        G4double* reconDPCY{nullptr};
        G4double* reconDPCTrueDeltaX{nullptr};
        G4double* reconDPCTrueDeltaY{nullptr};
    };

    struct ClassificationBuffers {
        G4bool* isPixelHit{nullptr};
        G4int* neighborhoodActiveCells{nullptr};
        G4int* nearestPixelI{nullptr};
        G4int* nearestPixelJ{nullptr};
        G4int* nearestPixelGlobalId{nullptr};
    };

    struct VectorBuffers {
        std::vector<G4double>* chargeFractions{nullptr};
        std::vector<G4double>* chargeFractionsRow{nullptr};  ///< Row-denominator fractions
        std::vector<G4double>* chargeFractionsCol{nullptr};  ///< Column-denominator fractions
        std::vector<G4double>* chargeFractionsBlock{nullptr}; ///< Block-denominator fractions
        std::vector<G4double>* charge{nullptr};
        std::vector<G4double>* chargeNew{nullptr};
        std::vector<G4double>* chargeFinal{nullptr};
        std::vector<G4double>* chargeRow{nullptr};      ///< Charge based on row fraction
        std::vector<G4double>* chargeNewRow{nullptr};   ///< Noisy charge based on row fraction
        std::vector<G4double>* chargeFinalRow{nullptr}; ///< Final charge based on row fraction
        std::vector<G4double>* chargeCol{nullptr};      ///< Charge based on col fraction
        std::vector<G4double>* chargeNewCol{nullptr};   ///< Noisy charge based on col fraction
        std::vector<G4double>* chargeFinalCol{nullptr}; ///< Final charge based on col fraction
        std::vector<G4double>* chargeBlock{nullptr};      ///< Charge based on block fraction
        std::vector<G4double>* chargeNewBlock{nullptr};   ///< Noisy charge based on block fraction
        std::vector<G4double>* chargeFinalBlock{nullptr}; ///< Final charge based on block fraction
        std::vector<G4double>* pixelX{nullptr};
        std::vector<G4double>* pixelY{nullptr};
        std::vector<G4double>* distance{nullptr};
        std::vector<G4double>* alpha{nullptr};
        std::vector<G4int>* pixelID{nullptr};
    };

    struct FullGridBuffers {
        std::vector<G4double>* fi{nullptr};
        std::vector<G4double>* fiRow{nullptr};    ///< Row-denominator fractions
        std::vector<G4double>* fiCol{nullptr};    ///< Column-denominator fractions
        std::vector<G4double>* fiBlock{nullptr};  ///< Block-denominator fractions
        std::vector<G4double>* qi{nullptr};
        std::vector<G4double>* qn{nullptr};
        std::vector<G4double>* qf{nullptr};
        std::vector<G4double>* qiRow{nullptr};    ///< Charge based on row fraction
        std::vector<G4double>* qnRow{nullptr};    ///< Noisy charge based on row fraction
        std::vector<G4double>* qfRow{nullptr};    ///< Final charge based on row fraction
        std::vector<G4double>* qiCol{nullptr};    ///< Charge based on col fraction
        std::vector<G4double>* qnCol{nullptr};    ///< Noisy charge based on col fraction
        std::vector<G4double>* qfCol{nullptr};    ///< Final charge based on col fraction
        std::vector<G4double>* qiBlock{nullptr};  ///< Charge based on block fraction
        std::vector<G4double>* qnBlock{nullptr};  ///< Noisy charge based on block fraction
        std::vector<G4double>* qfBlock{nullptr};  ///< Final charge based on block fraction
        std::vector<G4double>* distance{nullptr};
        std::vector<G4double>* alpha{nullptr};
        std::vector<G4double>* pixelX{nullptr};
        std::vector<G4double>* pixelY{nullptr};
        G4int* gridSide{nullptr};
    };

    void ConfigureCoreBranches(TTree* tree, const ScalarBuffers& scalars,
                               const ClassificationBuffers& classification,
                               const VectorBuffers& vectors,
                               Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood,
                               Config::PosReconModel reconModel = Config::PosReconModel::LogA);
    void ConfigureScalarBranches(TTree* tree, const ScalarBuffers& buffers,
                                 Config::PosReconModel reconModel = Config::PosReconModel::LogA);
    void ConfigureClassificationBranches(TTree* tree, const ClassificationBuffers& buffers);
    void ConfigureVectorBranches(TTree* tree, const VectorBuffers& buffers,
                                 Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood);
    void ConfigureNeighborhoodBranches(TTree* tree, std::vector<G4int>* pixelID);
    bool ConfigureFullGridBranches(TTree* tree, const FullGridBuffers& buffers,
                                   Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood);

private:
    static constexpr int kDefaultBufferSize = 256000;       ///< 256KB for neighborhood branches
    static constexpr int kLargeVectorBufferSize = 512000;   ///< 512KB for full grid branches (>10KB/entry)
    static constexpr int kSplitLevel = 0;
};

// ============================================================================
// Tree Filler
// ============================================================================

/// \brief Populates ROOT tree branches from event data.
class TreeFiller {
public:
    TreeFiller();

    void SetNeighborhoodRadius(G4int radius);
    void SetStoreFullFractions(G4bool enable) { fStoreFullFractions = enable; }
    void SetGridNumBlocksPerSide(G4int numBlocks) { fGridNumBlocksPerSide = numBlocks; }

    bool Fill(TTree* tree, const EventRecord& record, std::mutex* treeMutex = nullptr);

    // Scalar accessors
    G4double& TrueX() { return fTrueX; }
    G4double& TrueY() { return fTrueY; }
    G4double& PixelX() { return fPixelX; }
    G4double& PixelY() { return fPixelY; }
    G4double& Edep() { return fEdep; }
    G4double& PixelTrueDeltaX() { return fPixelTrueDeltaX; }
    G4double& PixelTrueDeltaY() { return fPixelTrueDeltaY; }
    G4double& ReconDPCX() { return fReconDPCX; }
    G4double& ReconDPCY() { return fReconDPCY; }
    G4double& ReconDPCTrueDeltaX() { return fReconDPCTrueDeltaX; }
    G4double& ReconDPCTrueDeltaY() { return fReconDPCTrueDeltaY; }
    G4bool& IsPixelHit() { return fIsPixelHit; }
    G4int& NeighborhoodActiveCells() { return fNeighborhoodActiveCells; }
    G4int& NearestPixelI() { return fNearestPixelI; }
    G4int& NearestPixelJ() { return fNearestPixelJ; }
    G4int& NearestPixelGlobalId() { return fNearestPixelGlobalId; }

    // Vector accessors - Neighborhood mode
    std::vector<G4double>& ChargeFractions() { return fNeighborhoodChargeFractions; }
    std::vector<G4double>& ChargeFractionsRow() { return fNeighborhoodChargeFractionsRow; }
    std::vector<G4double>& ChargeFractionsCol() { return fNeighborhoodChargeFractionsCol; }
    std::vector<G4double>& ChargeFractionsBlock() { return fNeighborhoodChargeFractionsBlock; }
    std::vector<G4double>& Charge() { return fNeighborhoodCharge; }
    std::vector<G4double>& ChargeNew() { return fNeighborhoodChargeNew; }
    std::vector<G4double>& ChargeFinal() { return fNeighborhoodChargeFinal; }
    // Vector accessors - RowCol mode
    std::vector<G4double>& ChargeRow() { return fNeighborhoodChargeRow; }
    std::vector<G4double>& ChargeNewRow() { return fNeighborhoodChargeNewRow; }
    std::vector<G4double>& ChargeFinalRow() { return fNeighborhoodChargeFinalRow; }
    std::vector<G4double>& ChargeCol() { return fNeighborhoodChargeCol; }
    std::vector<G4double>& ChargeNewCol() { return fNeighborhoodChargeNewCol; }
    std::vector<G4double>& ChargeFinalCol() { return fNeighborhoodChargeFinalCol; }
    // Vector accessors - Block mode
    std::vector<G4double>& ChargeBlock() { return fNeighborhoodChargeBlock; }
    std::vector<G4double>& ChargeNewBlock() { return fNeighborhoodChargeNewBlock; }
    std::vector<G4double>& ChargeFinalBlock() { return fNeighborhoodChargeFinalBlock; }
    // Vector accessors - common
    std::vector<G4double>& Distance() { return fNeighborhoodDistance; }
    std::vector<G4double>& Alpha() { return fNeighborhoodAlpha; }
    std::vector<G4double>& PixelXVec() { return fNeighborhoodPixelX; }
    std::vector<G4double>& PixelYVec() { return fNeighborhoodPixelY; }
    std::vector<G4int>& PixelIDVec() { return fNeighborhoodPixelID; }

    // Full grid accessors - Neighborhood mode
    std::vector<G4double>& FullFi() { return fFullFi; }
    std::vector<G4double>& FullFiRow() { return fFullFiRow; }
    std::vector<G4double>& FullFiCol() { return fFullFiCol; }
    std::vector<G4double>& FullFiBlock() { return fFullFiBlock; }
    std::vector<G4double>& FullQi() { return fFullQi; }
    std::vector<G4double>& FullQn() { return fFullQn; }
    std::vector<G4double>& FullQf() { return fFullQf; }
    // Full grid accessors - RowCol mode
    std::vector<G4double>& FullQiRow() { return fFullQiRow; }
    std::vector<G4double>& FullQnRow() { return fFullQnRow; }
    std::vector<G4double>& FullQfRow() { return fFullQfRow; }
    std::vector<G4double>& FullQiCol() { return fFullQiCol; }
    std::vector<G4double>& FullQnCol() { return fFullQnCol; }
    std::vector<G4double>& FullQfCol() { return fFullQfCol; }
    // Full grid accessors - Block mode
    std::vector<G4double>& FullQiBlock() { return fFullQiBlock; }
    std::vector<G4double>& FullQnBlock() { return fFullQnBlock; }
    std::vector<G4double>& FullQfBlock() { return fFullQfBlock; }
    // Full grid accessors - common
    std::vector<G4double>& FullDistance() { return fFullDistance; }
    std::vector<G4double>& FullAlpha() { return fFullAlpha; }
    std::vector<G4double>& FullPixelX() { return fFullPixelXGrid; }
    std::vector<G4double>& FullPixelY() { return fFullPixelYGrid; }
    G4int& FullGridSide() { return fFullGridSide; }

private:
    void UpdateSummaryScalars(const EventRecord& record);
    void PrepareNeighborhoodStorage(std::size_t requestedCells);
    void PopulateNeighborhoodFromRecord(const EventRecord& record);
    void PopulateFullFractionsFromRecord(const EventRecord& record);
    bool EnsureFullFractionBuffer(G4int gridSide = -1);

    G4double fTrueX{0.0}, fTrueY{0.0};
    G4double fPixelX{0.0}, fPixelY{0.0};
    G4double fEdep{0.0};
    G4double fPixelTrueDeltaX{0.0}, fPixelTrueDeltaY{0.0};
    G4double fReconDPCX{0.0}, fReconDPCY{0.0};
    G4double fReconDPCTrueDeltaX{0.0}, fReconDPCTrueDeltaY{0.0};
    G4bool fIsPixelHit{false};
    G4int fNeighborhoodActiveCells{0};
    G4int fNearestPixelI{-1}, fNearestPixelJ{-1}, fNearestPixelGlobalId{-1};

    std::vector<G4double> fNeighborhoodChargeFractions;
    std::vector<G4double> fNeighborhoodChargeFractionsRow;
    std::vector<G4double> fNeighborhoodChargeFractionsCol;
    std::vector<G4double> fNeighborhoodChargeFractionsBlock;
    std::vector<G4double> fNeighborhoodCharge;
    std::vector<G4double> fNeighborhoodChargeNew;
    std::vector<G4double> fNeighborhoodChargeFinal;
    std::vector<G4double> fNeighborhoodChargeRow;
    std::vector<G4double> fNeighborhoodChargeNewRow;
    std::vector<G4double> fNeighborhoodChargeFinalRow;
    std::vector<G4double> fNeighborhoodChargeCol;
    std::vector<G4double> fNeighborhoodChargeNewCol;
    std::vector<G4double> fNeighborhoodChargeFinalCol;
    std::vector<G4double> fNeighborhoodChargeBlock;
    std::vector<G4double> fNeighborhoodChargeNewBlock;
    std::vector<G4double> fNeighborhoodChargeFinalBlock;
    std::vector<G4double> fNeighborhoodDistance;
    std::vector<G4double> fNeighborhoodAlpha;
    std::vector<G4double> fNeighborhoodPixelX;
    std::vector<G4double> fNeighborhoodPixelY;
    std::vector<G4int> fNeighborhoodPixelID;

    std::vector<G4double> fFullFi, fFullFiRow, fFullFiCol, fFullFiBlock;
    std::vector<G4double> fFullQi, fFullQn, fFullQf;
    std::vector<G4double> fFullQiRow, fFullQnRow, fFullQfRow;
    std::vector<G4double> fFullQiCol, fFullQnCol, fFullQfCol;
    std::vector<G4double> fFullQiBlock, fFullQnBlock, fFullQfBlock;
    std::vector<G4double> fFullDistance, fFullAlpha;
    std::vector<G4double> fFullPixelXGrid, fFullPixelYGrid;
    G4int fFullGridSide{0};

    NeighborhoodLayout fNeighborhoodLayout;
    std::size_t fNeighborhoodCapacity{0};
    G4int fGridNumBlocksPerSide{0};
    G4bool fStoreFullFractions{false};
};

// ============================================================================
// Metadata Publisher
// ============================================================================

/// \brief Publishes simulation metadata to ROOT files.
///
/// Uses TParameter<T> for typed metadata storage (double, int, bool)
/// and TNamed for string values. This preserves types and eliminates
/// the need for string parsing when reading metadata.
class MetadataPublisher {
public:
    /// Typed metadata value: double, int, bool, or string
    using MetaValue = std::variant<G4double, G4int, G4bool, std::string>;
    using Entry = std::pair<std::string, MetaValue>;
    using EntryList = std::vector<Entry>;

    struct GridMetadata {
        G4double pixelSize{0.0};
        G4double pixelSpacing{0.0};
        G4double gridOffset{0.0};  ///< DD4hep-style grid offset
        G4double detectorSize{0.0};
        G4double detectorThickness{0.0};
        G4double interpadGap{0.0};
        G4int numBlocksPerSide{0};
        G4int neighborhoodRadius{0};
        G4int fullGridSide{0};
        G4bool storeFullFractions{false};
    };

    struct ModelMetadata {
        Config::SignalModel signalModel{Config::SignalModel::LogA};  ///< Signal sharing model
        Config::PosReconModel model{Config::PosReconModel::DPC};     ///< Reconstruction method
        Config::ActivePixelMode activePixelMode{Config::ActivePixelMode::Neighborhood};
        G4double beta{0.0};  ///< Linear signal model beta parameter (per micron)
    };

    struct PhysicsMetadata {
        G4double d0{0.0};
        G4double ionizationEnergy{0.0};
        G4double gain{0.0};
    };

    struct NoiseMetadata {
        G4double gainSigmaMin{0.0};
        G4double gainSigmaMax{0.0};
        G4double electronCount{0.0};
    };

    struct SimulationMetadata {
        std::string timestamp;
        std::string geant4Version;
        std::string rootVersion;
    };

    void SetGridMetadata(const GridMetadata& m) { fGrid = m; }
    void SetModelMetadata(const ModelMetadata& m) { fModel = m; }
    void SetPhysicsMetadata(const PhysicsMetadata& m) { fPhysics = m; }
    void SetNoiseMetadata(const NoiseMetadata& m) { fNoise = m; }
    void SetSimulationMetadata(const SimulationMetadata& m) { fSimulation = m; }

    EntryList CollectEntries() const;
    void WriteToTree(TTree* tree, std::mutex* ioMutex = nullptr) const;
    static void WriteEntriesToUserInfo(TTree* tree, const EntryList& entries);

private:
    static std::string ModelToString(Config::PosReconModel model);
    static std::string SignalModelToString(Config::SignalModel model);

    GridMetadata fGrid;
    ModelMetadata fModel;
    PhysicsMetadata fPhysics;
    NoiseMetadata fNoise;
    SimulationMetadata fSimulation;
};

// ============================================================================
// Post-Processing Runner
// ============================================================================

/// \brief Executes ROOT macro post-processing.
class PostProcessingRunner {
public:
    struct Config {
        bool runFitGaus1D{Constants::FIT_GAUS_1D};
        bool runFitGaus2D{Constants::FIT_GAUS_2D};
        std::string sourceDir;
        std::string rootFileName{"epicChargeSharing.root"};
    };

    void SetConfig(const Config& config) { fConfig = config; }
    const Config& GetConfig() const { return fConfig; }

    bool Run();
    bool RunMacro(const std::string& macroPath, const std::string& entryPoint,
                  const std::string& rootFile);

private:
    void EnsureBatchMode();
    void ConfigureIncludePaths();
    std::string NormalizePath(const std::string& path);

    Config fConfig;
    bool fBatchModeSet{false};
    bool fIncludePathsConfigured{false};
};

} // namespace ECS::IO

#endif // ECS_ROOT_IO_HH
