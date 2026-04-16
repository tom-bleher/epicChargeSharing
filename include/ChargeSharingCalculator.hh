// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// \file ChargeSharingCalculator.hh
/// \brief Definition of the ECS::ChargeSharingCalculator class.
///
/// This file declares the ChargeSharingCalculator class which implements
/// charge sharing models based on Tornago et al. (arXiv:2007.09528).
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_CHARGE_SHARING_CALCULATOR_HH
#define ECS_CHARGE_SHARING_CALCULATOR_HH

#include "G4ThreeVector.hh"
#include "globals.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace ECS {

// Forward declarations
class DetectorConstruction;

/// \brief Computes charge sharing fractions across pad neighborhoods.
///
/// This class implements the charge sharing model described in
/// arXiv:2007.09528 (Tornago et al.) for AC-LGAD / RSD detectors.
///
/// Paper terminology note: the paper refers to readout **pads**. This codebase
/// historically uses the term "pixel" for the same metal pad objects.
///
/// The calculator supports the terminology of Tornago et al.:
/// - **LogA**: Logarithmic attenuation model (paper Eq. (\ref{eq:masterformula}))
/// - **LinA**: Linear attenuation model (paper Eq. (\ref{eq:LA}))
///
/// For each event, the calculator:
/// 1. Finds the nearest pad to the hit position
/// 2. Defines a (2r+1)x(2r+1) neighborhood grid centered on that pad
/// 3. Computes charge fractions F_i for each pad in the grid
/// 4. Applies noise models (gain variations, electronic noise)
/// 5. Optionally computes full detector grid fractions
///
/// The Result struct contains all computed data for downstream processing.
///
/// @note Thread Safety: Each thread MUST own its own ChargeSharingCalculator instance.
/// The Compute() method mutates internal state (fResult, scratch buffers) and calls
/// G4RandGauss::shoot() for noise application, making it NOT safe for concurrent use
/// from multiple threads on the same instance. In the Geant4 simulation, this is
/// guaranteed by the worker-thread model: ActionInitialization::Build() creates a
/// fresh EventAction (which owns a ChargeSharingCalculator) for each worker thread.
class ChargeSharingCalculator {
public:
    struct PixelGridGeometry {
        G4int nRows{0};
        G4int nCols{0};
        G4double pitchX{0.0};
        G4double pitchY{0.0};
        G4double x0{0.0};
        G4double y0{0.0};
    };

    struct HitInfo {
        G4double trueX{std::numeric_limits<G4double>::quiet_NaN()};
        G4double trueY{std::numeric_limits<G4double>::quiet_NaN()};
        G4double trueZ{std::numeric_limits<G4double>::quiet_NaN()};
        G4int pixRow{-1};
        G4int pixCol{-1};
        G4double pixCenterX{std::numeric_limits<G4double>::quiet_NaN()};
        G4double pixCenterY{std::numeric_limits<G4double>::quiet_NaN()};
    };

    enum class ChargeMode { Neighborhood, FullGrid };

    template <typename T>
    struct Grid2D {
        Grid2D() = default;
        Grid2D(G4int rows, G4int cols) { Resize(rows, cols); }

        void Resize(G4int rows, G4int cols) { Resize(rows, cols, T{}); }

        void Resize(G4int rows, G4int cols, const T& value) {
            const auto safeRows = std::max<G4int>(0, rows);
            const auto safeCols = std::max<G4int>(0, cols);
            nRows = safeRows;
            nCols = safeCols;
            data.assign(static_cast<std::size_t>(safeRows) * static_cast<std::size_t>(safeCols), value);
        }

        void Clear() {
            nRows = 0;
            nCols = 0;
            data.clear();
        }

        void Fill(const T& value) { std::fill(data.begin(), data.end(), value); }

        [[nodiscard]] std::size_t Size() const { return data.size(); }
        [[nodiscard]] bool Empty() const { return data.empty(); }
        [[nodiscard]] G4int Rows() const { return nRows; }
        [[nodiscard]] G4int Cols() const { return nCols; }

        T& operator()(G4int row, G4int col) {
            const auto idx =
                (static_cast<std::size_t>(row) * static_cast<std::size_t>(nCols)) + static_cast<std::size_t>(col);
            return data[idx];
        }

        const T& operator()(G4int row, G4int col) const {
            const auto idx =
                (static_cast<std::size_t>(row) * static_cast<std::size_t>(nCols)) + static_cast<std::size_t>(col);
            return data[idx];
        }

        T* Data() { return data.data(); }
        [[nodiscard]] const T* Data() const { return data.data(); }

        /// \brief Type alias for Eigen matrix with row-major storage (matches Grid2D layout)
        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using EigenMap = Eigen::Map<EigenMatrix>;
        using ConstEigenMap = Eigen::Map<const EigenMatrix>;

        /// \brief Get an Eigen::Map view of the grid data for vectorized operations.
        /// \return Mutable Eigen::Map view of the underlying data.
        EigenMap AsEigen() { return EigenMap(data.data(), nRows, nCols); }

        /// \brief Get a const Eigen::Map view of the grid data for vectorized operations.
        /// \return Const Eigen::Map view of the underlying data.
        [[nodiscard]] ConstEigenMap AsEigen() const { return ConstEigenMap(data.data(), nRows, nCols); }

        /// \brief Compute the sum of all elements using Eigen's vectorized sum.
        /// \return Sum of all elements in the grid.
        [[nodiscard]] T Sum() const {
            if (data.empty())
                return T{};
            return AsEigen().sum();
        }

        /// \brief Compute row-wise sums using Eigen's vectorized operations.
        /// \return Vector of row sums (one per row).
        [[nodiscard]] Eigen::Matrix<T, Eigen::Dynamic, 1> RowSums() const {
            if (data.empty())
                return Eigen::Matrix<T, Eigen::Dynamic, 1>();
            return AsEigen().rowwise().sum();
        }

        /// \brief Compute column-wise sums using Eigen's vectorized operations.
        /// \return Vector of column sums (one per column).
        [[nodiscard]] Eigen::Matrix<T, 1, Eigen::Dynamic> ColSums() const {
            if (data.empty())
                return Eigen::Matrix<T, 1, Eigen::Dynamic>();
            return AsEigen().colwise().sum();
        }

        G4int nRows{0};
        G4int nCols{0};
        std::vector<T> data;
    };

    struct ChargeMatrixSet {
        ChargeMatrixSet() = default;
        ChargeMatrixSet(G4int rows, G4int cols) { Resize(rows, cols); }

        virtual void Resize(G4int rows, G4int cols) {
            signalFraction.Resize(rows, cols, 0.0);
            signalFractionRow.Resize(rows, cols, 0.0);
            signalFractionCol.Resize(rows, cols, 0.0);
            signalFractionBlock.Resize(rows, cols, 0.0);
            // Neighborhood-based charges
            chargeInduced.Resize(rows, cols, 0.0);
            chargeAmp.Resize(rows, cols, 0.0);
            chargeMeas.Resize(rows, cols, 0.0);
            // Row-based charges
            chargeInducedRow.Resize(rows, cols, 0.0);
            chargeAmpRow.Resize(rows, cols, 0.0);
            chargeMeasRow.Resize(rows, cols, 0.0);
            // Col-based charges
            chargeInducedCol.Resize(rows, cols, 0.0);
            chargeAmpCol.Resize(rows, cols, 0.0);
            chargeMeasCol.Resize(rows, cols, 0.0);
            // Block-based charges
            chargeInducedBlock.Resize(rows, cols, 0.0);
            chargeAmpBlock.Resize(rows, cols, 0.0);
            chargeMeasBlock.Resize(rows, cols, 0.0);
        }

        virtual void Clear() {
            signalFraction.Clear();
            signalFractionRow.Clear();
            signalFractionCol.Clear();
            signalFractionBlock.Clear();
            chargeInduced.Clear();
            chargeAmp.Clear();
            chargeMeas.Clear();
            chargeInducedRow.Clear();
            chargeAmpRow.Clear();
            chargeMeasRow.Clear();
            chargeInducedCol.Clear();
            chargeAmpCol.Clear();
            chargeMeasCol.Clear();
            chargeInducedBlock.Clear();
            chargeAmpBlock.Clear();
            chargeMeasBlock.Clear();
        }

        virtual void Zero() {
            if (!signalFraction.Empty())
                signalFraction.Fill(0.0);
            if (!signalFractionRow.Empty())
                signalFractionRow.Fill(0.0);
            if (!signalFractionCol.Empty())
                signalFractionCol.Fill(0.0);
            if (!signalFractionBlock.Empty())
                signalFractionBlock.Fill(0.0);
            if (!chargeInduced.Empty())
                chargeInduced.Fill(0.0);
            if (!chargeAmp.Empty())
                chargeAmp.Fill(0.0);
            if (!chargeMeas.Empty())
                chargeMeas.Fill(0.0);
            if (!chargeInducedRow.Empty())
                chargeInducedRow.Fill(0.0);
            if (!chargeAmpRow.Empty())
                chargeAmpRow.Fill(0.0);
            if (!chargeMeasRow.Empty())
                chargeMeasRow.Fill(0.0);
            if (!chargeInducedCol.Empty())
                chargeInducedCol.Fill(0.0);
            if (!chargeAmpCol.Empty())
                chargeAmpCol.Fill(0.0);
            if (!chargeMeasCol.Empty())
                chargeMeasCol.Fill(0.0);
            if (!chargeInducedBlock.Empty())
                chargeInducedBlock.Fill(0.0);
            if (!chargeAmpBlock.Empty())
                chargeAmpBlock.Fill(0.0);
            if (!chargeMeasBlock.Empty())
                chargeMeasBlock.Fill(0.0);
        }

        [[nodiscard]] G4int Rows() const { return signalFraction.Rows(); }
        [[nodiscard]] G4int Cols() const { return signalFraction.Cols(); }
        [[nodiscard]] bool Empty() const { return signalFraction.Empty(); }

        Grid2D<G4double> signalFraction;      ///< F_i: fraction of total signal
        Grid2D<G4double> signalFractionRow;   ///< F_i normalized by row sum (for 1D row fits)
        Grid2D<G4double> signalFractionCol;   ///< F_i normalized by column sum (for 1D col fits)
        Grid2D<G4double> signalFractionBlock; ///< F_i normalized by 4-pad block sum
        // Neighborhood-based charges
        Grid2D<G4double> chargeInduced;   ///< Induced charge before noise
        Grid2D<G4double> chargeAmp; ///< Charge after adding noise
        Grid2D<G4double> chargeMeas;     ///< Final processed charge
        // Row-based charges
        Grid2D<G4double> chargeInducedRow;   ///< Row-based induced charge
        Grid2D<G4double> chargeAmpRow; ///< Row-based charge after noise
        Grid2D<G4double> chargeMeasRow;     ///< Row-based final charge
        // Col-based charges
        Grid2D<G4double> chargeInducedCol;   ///< Col-based induced charge
        Grid2D<G4double> chargeAmpCol; ///< Col-based charge after noise
        Grid2D<G4double> chargeMeasCol;     ///< Col-based final charge
        // Block-based charges
        Grid2D<G4double> chargeInducedBlock;   ///< Block-based induced charge
        Grid2D<G4double> chargeAmpBlock; ///< Block-based charge after noise
        Grid2D<G4double> chargeMeasBlock;     ///< Block-based final charge
    };

    struct FullGridCharges : ChargeMatrixSet {
        using ChargeMatrixSet::ChargeMatrixSet;

        void Resize(G4int rows, G4int cols) override {
            ChargeMatrixSet::Resize(rows, cols);
            distance.Resize(rows, cols, 0.0);
            alpha.Resize(rows, cols, 0.0);
            pixelX.Resize(rows, cols, 0.0);
            pixelY.Resize(rows, cols, 0.0);
        }

        void Clear() override {
            ChargeMatrixSet::Clear();
            distance.Clear();
            alpha.Clear();
            pixelX.Clear();
            pixelY.Clear();
        }

        void Zero() override {
            ChargeMatrixSet::Zero();
            if (!distance.Empty())
                distance.Fill(0.0);
            if (!alpha.Empty())
                alpha.Fill(0.0);
            if (!pixelX.Empty())
                pixelX.Fill(0.0);
            if (!pixelY.Empty())
                pixelY.Fill(0.0);
        }

        /// \brief d_i grid: distance to pixel center.
        Grid2D<G4double> distance;
        /// \brief α_i grid (paper notation): pad angle of view.
        Grid2D<G4double> alpha;
        Grid2D<G4double> pixelX;
        Grid2D<G4double> pixelY;
    };

    struct NeighborhoodGridBounds {
        G4int rowMin{-1};
        G4int colMin{-1};
        G4int nRows{0};
        G4int nCols{0};

        [[nodiscard]] G4int rowMax() const { return rowMin + nRows; }
        [[nodiscard]] G4int colMax() const { return colMin + nCols; }
        [[nodiscard]] bool Valid() const { return rowMin >= 0 && colMin >= 0 && nRows > 0 && nCols > 0; }
    };

    struct PatchGridCharges {
        void Reset() {
            patch = NeighborhoodGridBounds{};
            charges.Clear();
        }

        void Resize(const NeighborhoodGridBounds& info) {
            patch = info;
            charges.Resize(info.nRows, info.nCols);
        }

        [[nodiscard]] bool Empty() const { return charges.Empty(); }

        NeighborhoodGridBounds patch;
        ChargeMatrixSet charges;
    };

    struct Result {
        struct NeighborCell {
            G4int gridIndex{-1};
            G4int globalPixelId{-1};
            G4ThreeVector center{0., 0., 0.};
            G4double fraction{0.0};      ///< Signal fraction (neighborhood denominator)
            G4double fractionRow{0.0};   ///< Signal fraction (row denominator, for 1D row fits)
            G4double fractionCol{0.0};   ///< Signal fraction (column denominator, for 1D col fits)
            G4double fractionBlock{0.0}; ///< Signal fraction (4-pad block denominator)
            G4double chargeInd{0.0};        ///< Induced charge (neighborhood denominator)
            G4double chargeIndRow{0.0};     ///< Induced charge (row denominator)
            G4double chargeIndCol{0.0};     ///< Induced charge (col denominator)
            G4double chargeIndBlock{0.0};   ///< Induced charge (block denominator)
            /// \brief d_i: distance from hit point to pixel center.
            ///
            /// Note: GEANT4 uses millimeters as base length unit; this value is stored in
            /// those internal units.
            G4double distance{0.0};

            /// \brief α_i (paper notation): pad angle of view.
            G4double alpha{0.0};
        };

        void Reset() {
            geometry = PixelGridGeometry{};
            hit = HitInfo{};
            mode = ChargeMode::Neighborhood;
            nearestPixelCenter = G4ThreeVector(0., 0., 0.);
            pixelRowIndex = 0;
            pixelColIndex = 0;
            gridRadius = 0;
            gridSide = 1;
            totalCells = 0;
            cells.clear();
            chargeBlock.clear();
            full.Zero();
            patch.Reset();
        }

        PixelGridGeometry geometry{};
        HitInfo hit{};
        ChargeMode mode{ChargeMode::Neighborhood};
        G4ThreeVector nearestPixelCenter{0., 0., 0.};
        G4int pixelRowIndex{0};
        G4int pixelColIndex{0};
        G4int gridRadius{0};
        G4int gridSide{1};
        std::size_t totalCells{0};
        std::vector<NeighborCell> cells;
        /// \brief Dominant-pad set.
        ///
        /// Historical note: stored as `chargeBlock` for backward compatibility with
        /// earlier analysis code. This corresponds to the highest-weight pads used by
        /// ChargeBlock active pixel modes.
        std::vector<NeighborCell> chargeBlock;
        FullGridCharges full;
        PatchGridCharges patch;
    };

    explicit ChargeSharingCalculator(const DetectorConstruction* detector = nullptr);

    void SetDetector(const DetectorConstruction* detector);
    void SetNeighborhoodRadius(G4int radius);
    [[nodiscard]] G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    void SetEmitDistanceAlpha(G4bool enabled) { fEmitDistanceAlpha = enabled; }
    [[nodiscard]] G4bool GetEmitDistanceAlpha() const { return fEmitDistanceAlpha; }
    void SetComputeFullGridFractions(G4bool enabled);
    [[nodiscard]] G4bool GetComputeFullGridFractions() const { return fComputeFullGridFractions; }

    void ResetForEvent();

    const Result& Compute(const G4ThreeVector& hitPos, G4double energyDeposit, G4double ionizationEnergy,
                          G4double amplificationFactor, G4double d0, G4double elementaryCharge);

    /// \brief Minimal step deposit for per-step charge sharing (decoupled from SteppingAction).
    struct StepInput {
        G4ThreeVector position;
        G4double edep{0.0};
    };

    /// \brief Compute charge sharing from multiple step positions (superposition).
    ///
    /// Each step contributes charge based on its own (x,y) position on the sensor.
    /// The neighborhood is centered on the edep-weighted centroid of all steps.
    /// Effective fractions are a convex combination of per-step fractions,
    /// guaranteeing sum=1 by construction.
    ///
    /// For single-step events, delegates to Compute() for bit-identical results.
    const Result& ComputeFromSteps(const std::vector<StepInput>& steps, G4double totalEnergyDeposit,
                                   G4double ionizationEnergy, G4double amplificationFactor, G4double d0,
                                   G4double elementaryCharge);

private:
    /// \brief Validated D0 parameters for charge sharing calculations.
    ///
    /// Contains validated D0 values in both mm and micron units.
    /// The core library handles guard logic internally.
    struct D0Params {
        G4double lengthMM{0.0}; ///< Validated D0 in mm
        G4double micron{0.0};   ///< Validated D0 in microns (for core config)
        G4bool isValid{false};  ///< True if original D0 was valid

        static constexpr G4double kMinD0 = 1e-6; ///< Minimum D0 in micrometers
    };

    /// \brief Validated charge model parameters.
    struct ChargeModelParams {
        G4bool useLinear{false}; ///< True for Linear model
        G4double beta{0.0};      ///< Attenuation coefficient in 1/um (Linear only)
    };

    static D0Params ValidateD0(G4double d0Raw, const char* callerName);
    [[nodiscard]] ChargeModelParams GetChargeModelParams(G4double pixelSpacing) const;

    void ReserveBuffers();
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
    void ComputeChargeFractions(const G4ThreeVector& hitPos, G4double totalChargeElectrons, G4double d0,
                                G4double elementaryCharge);
    void EnsureFullGridBuffer();
    void ComputeFullGridFractions(const G4ThreeVector& hitPos, G4double d0, G4double pixelSize, G4double pixelSpacing,
                                  G4int numBlocksPerSide, G4double totalChargeElectrons, G4double elementaryCharge);
    [[nodiscard]] PixelGridGeometry BuildGridGeometry() const;
    void PopulatePatchFromNeighbors(G4int numBlocksPerSide);

    const DetectorConstruction* fDetector{nullptr};
    G4int fNeighborhoodRadius{0};
    Result fResult;

    Grid2D<G4double> fNeighborhoodWeights; ///< Neighborhood weights for Eigen-based row/col sums
    Grid2D<G4double> fFullGridWeights;
    int fGridDim{0};
    G4bool fEmitDistanceAlpha{false};
    G4bool fComputeFullGridFractions{false};
    G4bool fNeedsReset{true};
};

} // namespace ECS

// Backward compatibility alias
using ChargeSharingCalculator = ECS::ChargeSharingCalculator;

#endif // ECS_CHARGE_SHARING_CALCULATOR_HH
