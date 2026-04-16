// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

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
#include "globals.hh"
#include "NeighborhoodUtils.hh"

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
    G4double primaryMomentumX{0.0}; ///< Primary particle px (Geant4 units: MeV)
    G4double primaryMomentumY{0.0}; ///< Primary particle py (Geant4 units: MeV)
    G4double primaryMomentumZ{0.0}; ///< Primary particle pz (Geant4 units: MeV)
    G4double hitTime{0.0};          ///< Global time at first contact (Geant4 units: ns)
    G4double pathLength{0.0};       ///< Path length through sensitive volume (Geant4 units: mm)
    G4double eventGain{0.0};        ///< Sampled event-level gain (includes saturation + fluctuation)
    G4int nSteps{0};                ///< Number of Geant4 steps in sensitive volume
    G4bool firstContactIsPixel{false};
    G4bool geometricIsPixel{false};
    G4bool isPixelHitCombined{false};
    G4bool hitWithinDetector{false}; ///< True if hit position maps to a valid pixel (not clamped from outside grid)
};

/// \brief Complete event record for ROOT tree filling.
struct EventRecord {
    EventSummaryData summary;
    std::span<const ChargeSharingCalculator::Result::NeighborCell> neighborCells;
    std::span<const ChargeSharingCalculator::Result::NeighborCell> chargeBlock;
    std::span<const G4double> neighborChargesAmp;
    std::span<const G4double> neighborChargesMeas;
    // Row-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesAmpRow;
    std::span<const G4double> neighborChargesMeasRow;
    // Col-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesAmpCol;
    std::span<const G4double> neighborChargesMeasCol;
    // Block-mode neighborhood noisy charges
    std::span<const G4double> neighborChargesAmpBlock;
    std::span<const G4double> neighborChargesMeasBlock;
    // Full grid fractions
    std::span<const G4double> fullFi;
    std::span<const G4double> fullFiRow;   ///< Signal fractions with row denominator
    std::span<const G4double> fullFiCol;   ///< Signal fractions with column denominator
    std::span<const G4double> fullFiBlock; ///< Signal fractions with 4-pad block denominator
    // Full grid neighborhood-mode charges
    std::span<const G4double> fullQ_ind;
    std::span<const G4double> fullQ_amp;
    std::span<const G4double> fullQ_meas;
    // Full grid row-mode charges
    std::span<const G4double> fullQ_indRow;
    std::span<const G4double> fullQ_ampRow;
    std::span<const G4double> fullQ_measRow;
    // Full grid col-mode charges
    std::span<const G4double> fullQ_indCol;
    std::span<const G4double> fullQ_ampCol;
    std::span<const G4double> fullQ_measCol;
    // Full grid block-mode charges
    std::span<const G4double> fullQ_indBlock;
    std::span<const G4double> fullQ_ampBlock;
    std::span<const G4double> fullQ_measBlock;
    // Full grid geometry
    std::span<const G4double> fullDistance;
    std::span<const G4double> fullAlpha;
    std::span<const G4double> fullPixelX;
    std::span<const G4double> fullPixelY;
    ChargeSharingCalculator::PixelGridGeometry geometry;
    ChargeSharingCalculator::HitInfo hit;
    ChargeSharingCalculator::ChargeMode mode{ChargeSharingCalculator::ChargeMode::Neighborhood};
    ChargeSharingCalculator::NeighborhoodGridBounds neighborhoodGridBounds;
    G4int fullGridRows{0};
    G4int fullGridCols{0};
    G4int nearestPixelI{-1};
    G4int nearestPixelJ{-1};
    G4int nearestPixelGlobalId{-1};
    // Per-step energy deposits (Landau fluctuation data)
    std::span<const G4double> stepEdep;
    std::span<const G4double> stepX;
    std::span<const G4double> stepY;
    std::span<const G4double> stepZ;
    std::span<const G4double> stepTime;
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
        G4double* primaryMomentumX{nullptr};
        G4double* primaryMomentumY{nullptr};
        G4double* primaryMomentumZ{nullptr};
        G4double* hitTime{nullptr};
        G4double* pathLength{nullptr};
        G4double* eventGain{nullptr};
    };

    struct ClassificationBuffers {
        G4bool* isPixelHit{nullptr};
        G4bool* hitWithinDetector{nullptr};
        G4int* neighborhoodActiveCells{nullptr};
        G4int* nearestPixelI{nullptr};
        G4int* nearestPixelJ{nullptr};
        G4int* nearestPixelGlobalId{nullptr};
    };

    struct VectorBuffers {
        std::vector<G4double>* chargeFractions{nullptr};
        std::vector<G4double>* chargeFractionsRow{nullptr};   ///< Row-denominator fractions
        std::vector<G4double>* chargeFractionsCol{nullptr};   ///< Column-denominator fractions
        std::vector<G4double>* chargeFractionsBlock{nullptr}; ///< Block-denominator fractions
        std::vector<G4double>* chargeInd{nullptr};
        std::vector<G4double>* chargeAmp{nullptr};
        std::vector<G4double>* chargeMeas{nullptr};
        std::vector<G4double>* chargeRow{nullptr};        ///< Charge based on row fraction
        std::vector<G4double>* chargeAmpRow{nullptr};     ///< Noisy charge based on row fraction
        std::vector<G4double>* chargeMeasRow{nullptr};   ///< Final charge based on row fraction
        std::vector<G4double>* chargeCol{nullptr};        ///< Charge based on col fraction
        std::vector<G4double>* chargeAmpCol{nullptr};     ///< Noisy charge based on col fraction
        std::vector<G4double>* chargeMeasCol{nullptr};   ///< Final charge based on col fraction
        std::vector<G4double>* chargeIndBlock{nullptr};      ///< Charge based on block fraction
        std::vector<G4double>* chargeAmpBlock{nullptr};   ///< Noisy charge based on block fraction
        std::vector<G4double>* chargeMeasBlock{nullptr}; ///< Final charge based on block fraction
        std::vector<G4double>* pixelX{nullptr};
        std::vector<G4double>* pixelY{nullptr};
        std::vector<G4double>* distance{nullptr};
        std::vector<G4double>* alpha{nullptr};
        std::vector<G4int>* pixelID{nullptr};
    };

    struct FullGridBuffers {
        std::vector<G4double>* fi{nullptr};
        std::vector<G4double>* fiRow{nullptr};   ///< Row-denominator fractions
        std::vector<G4double>* fiCol{nullptr};   ///< Column-denominator fractions
        std::vector<G4double>* fiBlock{nullptr}; ///< Block-denominator fractions
        std::vector<G4double>* qInd{nullptr};
        std::vector<G4double>* qAmp{nullptr};
        std::vector<G4double>* qMeas{nullptr};
        std::vector<G4double>* qIndRow{nullptr};   ///< Charge based on row fraction
        std::vector<G4double>* qAmpRow{nullptr};    ///< Noisy charge based on row fraction
        std::vector<G4double>* qMeasRow{nullptr};     ///< Final charge based on row fraction
        std::vector<G4double>* qIndCol{nullptr};   ///< Charge based on col fraction
        std::vector<G4double>* qAmpCol{nullptr};    ///< Noisy charge based on col fraction
        std::vector<G4double>* qMeasCol{nullptr};     ///< Final charge based on col fraction
        std::vector<G4double>* qIndBlock{nullptr}; ///< Charge based on block fraction
        std::vector<G4double>* qAmpBlock{nullptr};  ///< Noisy charge based on block fraction
        std::vector<G4double>* qMeasBlock{nullptr};   ///< Final charge based on block fraction
        std::vector<G4double>* distance{nullptr};
        std::vector<G4double>* alpha{nullptr};
        std::vector<G4double>* pixelX{nullptr};
        std::vector<G4double>* pixelY{nullptr};
        G4int* gridSide{nullptr};
    };

    struct StepBuffers {
        G4int* nSteps{nullptr};
        std::vector<G4double>* edep{nullptr};
        std::vector<G4double>* x{nullptr};
        std::vector<G4double>* y{nullptr};
        std::vector<G4double>* z{nullptr};
        std::vector<G4double>* time{nullptr};
    };

    void ConfigureCoreBranches(TTree* tree, const ScalarBuffers& scalars, const ClassificationBuffers& classification,
                               const VectorBuffers& vectors,
                               Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood,
                               Config::PosReconModel reconModel = Config::PosReconModel::LogA);
    static void ConfigureScalarBranches(TTree* tree, const ScalarBuffers& buffers,
                                        Config::PosReconModel reconModel = Config::PosReconModel::LogA);
    static void ConfigureClassificationBranches(TTree* tree, const ClassificationBuffers& buffers);
    static void ConfigureVectorBranches(TTree* tree, const VectorBuffers& buffers,
                                        Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood);
    static void ConfigureNeighborhoodBranches(TTree* tree, std::vector<G4int>* pixelID);
    static void ConfigureStepBranches(TTree* tree, const StepBuffers& buffers);
    static bool ConfigureFullGridBranches(TTree* tree, const FullGridBuffers& buffers,
                                          Config::ActivePixelMode mode = Config::ActivePixelMode::Neighborhood);

private:
    static constexpr int kDefaultBufferSize = 256000;     ///< 256KB for neighborhood branches
    static constexpr int kLargeVectorBufferSize = 512000; ///< 512KB for full grid branches (>10KB/entry)
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

    /// \brief Fill the TTree with the current event data.
    ///
    /// Populates branch buffers from the EventRecord, then calls TTree::Fill().
    /// In multithreaded Geant4 (G4MTRunManager), multiple worker threads share
    /// the same TTree, so a mutex is needed to serialize access.
    ///
    /// @param tree        The TTree to fill (must have branches configured).
    /// @param record      Event data to write.
    /// @param treeMutex   Mutex for thread-safe TTree::Fill().
    ///        REQUIRED when running with G4MTRunManager (multi-threaded mode).
    ///        Passing nullptr in MT mode will cause data corruption in the
    ///        ROOT file due to unsynchronized concurrent TTree::Fill() calls.
    /// @return true on success, false if tree is null or Fill() fails.
    bool Fill(TTree* tree, const EventRecord& record, std::mutex* treeMutex = nullptr);

    // Scalar accessors
    G4double& TrueX() { return fTrueX; }
    G4double& TrueY() { return fTrueY; }
    G4double& PixelX() { return fPixelX; }
    G4double& PixelY() { return fPixelY; }
    G4double& Edep() { return fEdep; }
    G4double& PixelTrueDeltaX() { return fPixelTrueDeltaX; }
    G4double& PixelTrueDeltaY() { return fPixelTrueDeltaY; }
    G4double& PrimaryMomentumX() { return fPrimaryMomentumX; }
    G4double& PrimaryMomentumY() { return fPrimaryMomentumY; }
    G4double& PrimaryMomentumZ() { return fPrimaryMomentumZ; }
    G4double& HitTime() { return fHitTime; }
    G4double& PathLength() { return fPathLength; }
    G4double& EventGain() { return fEventGain; }
    G4bool& IsPixelHit() { return fIsPixelHit; }
    G4bool& HitWithinDetector() { return fHitWithinDetector; }
    G4int& NeighborhoodActiveCells() { return fNeighborhoodActiveCells; }
    G4int& NearestPixelI() { return fNearestPixelI; }
    G4int& NearestPixelJ() { return fNearestPixelJ; }
    G4int& NearestPixelGlobalId() { return fNearestPixelGlobalId; }

    // Vector accessors - Neighborhood mode
    std::vector<G4double>& ChargeFractions() { return fNeighborhoodChargeFractions; }
    std::vector<G4double>& ChargeFractionsRow() { return fNeighborhoodChargeFractionsRow; }
    std::vector<G4double>& ChargeFractionsCol() { return fNeighborhoodChargeFractionsCol; }
    std::vector<G4double>& ChargeFractionsBlock() { return fNeighborhoodChargeFractionsBlock; }
    std::vector<G4double>& ChargeInd() { return fNeighborhoodChargeInd; }
    std::vector<G4double>& ChargeAmp() { return fNeighborhoodChargeAmp; }
    std::vector<G4double>& ChargeMeas() { return fNeighborhoodChargeMeas; }
    // Vector accessors - RowCol mode
    std::vector<G4double>& ChargeIndRow() { return fNeighborhoodChargeIndRow; }
    std::vector<G4double>& ChargeAmpRow() { return fNeighborhoodChargeAmpRow; }
    std::vector<G4double>& ChargeMeasRow() { return fNeighborhoodChargeMeasRow; }
    std::vector<G4double>& ChargeIndCol() { return fNeighborhoodChargeIndCol; }
    std::vector<G4double>& ChargeAmpCol() { return fNeighborhoodChargeAmpCol; }
    std::vector<G4double>& ChargeMeasCol() { return fNeighborhoodChargeMeasCol; }
    // Vector accessors - Block mode
    std::vector<G4double>& ChargeIndBlock() { return fNeighborhoodChargeIndBlock; }
    std::vector<G4double>& ChargeAmpBlock() { return fNeighborhoodChargeAmpBlock; }
    std::vector<G4double>& ChargeMeasBlock() { return fNeighborhoodChargeMeasBlock; }
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
    std::vector<G4double>& FullQ_ind() { return fFullQ_ind; }
    std::vector<G4double>& FullQ_amp() { return fFullQ_amp; }
    std::vector<G4double>& FullQ_meas() { return fFullQ_meas; }
    // Full grid accessors - RowCol mode
    std::vector<G4double>& FullQ_indRow() { return fFullQ_indRow; }
    std::vector<G4double>& FullQ_ampRow() { return fFullQ_ampRow; }
    std::vector<G4double>& FullQ_measRow() { return fFullQ_measRow; }
    std::vector<G4double>& FullQ_indCol() { return fFullQ_indCol; }
    std::vector<G4double>& FullQ_ampCol() { return fFullQ_ampCol; }
    std::vector<G4double>& FullQ_measCol() { return fFullQ_measCol; }
    // Full grid accessors - Block mode
    std::vector<G4double>& FullQ_indBlock() { return fFullQ_indBlock; }
    std::vector<G4double>& FullQ_ampBlock() { return fFullQ_ampBlock; }
    std::vector<G4double>& FullQ_measBlock() { return fFullQ_measBlock; }
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
    G4double fPrimaryMomentumX{0.0}, fPrimaryMomentumY{0.0}, fPrimaryMomentumZ{0.0};
    G4double fHitTime{0.0};
    G4double fPathLength{0.0};
    G4double fEventGain{0.0};
    G4bool fIsPixelHit{false};
    G4bool fHitWithinDetector{false};
    G4int fNeighborhoodActiveCells{0};
    G4int fNearestPixelI{-1}, fNearestPixelJ{-1}, fNearestPixelGlobalId{-1};

    std::vector<G4double> fNeighborhoodChargeFractions;
    std::vector<G4double> fNeighborhoodChargeFractionsRow;
    std::vector<G4double> fNeighborhoodChargeFractionsCol;
    std::vector<G4double> fNeighborhoodChargeFractionsBlock;
    std::vector<G4double> fNeighborhoodChargeInd;
    std::vector<G4double> fNeighborhoodChargeAmp;
    std::vector<G4double> fNeighborhoodChargeMeas;
    std::vector<G4double> fNeighborhoodChargeIndRow;
    std::vector<G4double> fNeighborhoodChargeAmpRow;
    std::vector<G4double> fNeighborhoodChargeMeasRow;
    std::vector<G4double> fNeighborhoodChargeIndCol;
    std::vector<G4double> fNeighborhoodChargeAmpCol;
    std::vector<G4double> fNeighborhoodChargeMeasCol;
    std::vector<G4double> fNeighborhoodChargeIndBlock;
    std::vector<G4double> fNeighborhoodChargeAmpBlock;
    std::vector<G4double> fNeighborhoodChargeMeasBlock;
    std::vector<G4double> fNeighborhoodDistance;
    std::vector<G4double> fNeighborhoodAlpha;
    std::vector<G4double> fNeighborhoodPixelX;
    std::vector<G4double> fNeighborhoodPixelY;
    std::vector<G4int> fNeighborhoodPixelID;

    std::vector<G4double> fFullFi, fFullFiRow, fFullFiCol, fFullFiBlock;
    std::vector<G4double> fFullQ_ind, fFullQ_amp, fFullQ_meas;
    std::vector<G4double> fFullQ_indRow, fFullQ_ampRow, fFullQ_measRow;
    std::vector<G4double> fFullQ_indCol, fFullQ_ampCol, fFullQ_measCol;
    std::vector<G4double> fFullQ_indBlock, fFullQ_ampBlock, fFullQ_measBlock;
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
        G4double gridOffset{0.0}; ///< DD4hep-style grid offset
        G4double detectorSize{0.0};
        G4double detectorThickness{0.0};
        G4double interpadGap{0.0};
        G4int numBlocksPerSide{0};
        G4int neighborhoodRadius{0};
        G4int fullGridSide{0};
        G4bool storeFullFractions{false};
    };

    struct ModelMetadata {
        Config::SignalModel signalModel{Config::SignalModel::LogA}; ///< Signal sharing model
        Config::PosReconModel model{Config::PosReconModel::LogA};   ///< Reconstruction method
        Config::ActivePixelMode activePixelMode{Config::ActivePixelMode::Neighborhood};
        G4double beta{0.0}; ///< Linear signal model beta parameter (per micron)
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
        G4double readoutThresholdSigma{0.0};
    };

    struct SimulationMetadata {
        std::string timestamp;
        std::string geant4Version;
        std::string rootVersion;
    };

    struct BeamMetadata {
        std::string particleName;
        G4double energyGeV{0.0};
        G4bool useFixedPosition{false};
        G4double fixedXMM{0.0};
        G4double fixedYMM{0.0};
        G4double beamOvershoot{0.0};
        G4double energyMinGeV{0.0};
        G4double energyMaxGeV{0.0};
        G4double thetaMinMrad{0.0};
        G4double thetaMaxMrad{0.0};
        std::string presetName;
    };

    void SetGridMetadata(const GridMetadata& m) { fGrid = m; }
    void SetModelMetadata(const ModelMetadata& m) { fModel = m; }
    void SetPhysicsMetadata(const PhysicsMetadata& m) { fPhysics = m; }
    void SetNoiseMetadata(const NoiseMetadata& m) { fNoise = m; }
    void SetSimulationMetadata(const SimulationMetadata& m) { fSimulation = m; }
    void SetBeamMetadata(const BeamMetadata& m) { fBeam = m; }

    [[nodiscard]] EntryList CollectEntries() const;

    /// \brief Write metadata entries to TTree UserInfo.
    ///
    /// @param tree     The TTree to attach metadata to.
    /// @param ioMutex  Mutex for thread-safe ROOT object creation.
    ///        REQUIRED when running with G4MTRunManager (multi-threaded mode).
    ///        Passing nullptr in MT mode risks concurrent modification of
    ///        the TTree's UserInfo list.
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
    BeamMetadata fBeam;
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
    [[nodiscard]] const Config& GetConfig() const { return fConfig; }

    bool Run();
    bool RunMacro(const std::string& macroPath, const std::string& entryPoint, const std::string& rootFile);

private:
    void EnsureBatchMode();
    void ConfigureIncludePaths();
    static std::string NormalizePath(const std::string& path);

    Config fConfig;
    bool fBatchModeSet{false};
    bool fIncludePathsConfigured{false};
};

} // namespace ECS::IO

#endif // ECS_ROOT_IO_HH
