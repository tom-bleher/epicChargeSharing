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
/// - **DPC**: Discretized Positioning Circuit reconstruction (paper Section 3.4)
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
class ChargeSharingCalculator
{
public:
    struct PixelGridGeometry
    {
        G4int nRows{0};
        G4int nCols{0};
        G4double pitchX{0.0};
        G4double pitchY{0.0};
        G4double x0{0.0};
        G4double y0{0.0};
    };

    struct HitInfo
    {
        G4double trueX{std::numeric_limits<G4double>::quiet_NaN()};
        G4double trueY{std::numeric_limits<G4double>::quiet_NaN()};
        G4double trueZ{std::numeric_limits<G4double>::quiet_NaN()};
        G4int pixRow{-1};
        G4int pixCol{-1};
        G4double pixCenterX{std::numeric_limits<G4double>::quiet_NaN()};
        G4double pixCenterY{std::numeric_limits<G4double>::quiet_NaN()};
    };

    enum class ChargeMode
    {
        Patch,
        FullGrid
    };

    template<typename T>
    struct Grid2D
    {
        Grid2D() = default;
        Grid2D(G4int rows, G4int cols) { Resize(rows, cols); }

        void Resize(G4int rows, G4int cols)
        {
            Resize(rows, cols, T{});
        }

        void Resize(G4int rows, G4int cols, const T& value)
        {
            const auto safeRows = std::max<G4int>(0, rows);
            const auto safeCols = std::max<G4int>(0, cols);
            nRows = safeRows;
            nCols = safeCols;
            data.assign(static_cast<std::size_t>(safeRows) * static_cast<std::size_t>(safeCols), value);
        }

        void Clear()
        {
            nRows = 0;
            nCols = 0;
            data.clear();
        }

        void Fill(const T& value)
        {
            std::fill(data.begin(), data.end(), value);
        }

        std::size_t Size() const { return data.size(); }
        bool Empty() const { return data.empty(); }
        G4int Rows() const { return nRows; }
        G4int Cols() const { return nCols; }

        T& operator()(G4int row, G4int col)
        {
            const auto idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(nCols) +
                             static_cast<std::size_t>(col);
            return data[idx];
        }

        const T& operator()(G4int row, G4int col) const
        {
            const auto idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(nCols) +
                             static_cast<std::size_t>(col);
            return data[idx];
        }

        T* Data() { return data.data(); }
        const T* Data() const { return data.data(); }

        /// \brief Type alias for Eigen matrix with row-major storage (matches Grid2D layout)
        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using EigenMap = Eigen::Map<EigenMatrix>;
        using ConstEigenMap = Eigen::Map<const EigenMatrix>;

        /// \brief Get an Eigen::Map view of the grid data for vectorized operations.
        /// \return Mutable Eigen::Map view of the underlying data.
        EigenMap AsEigen()
        {
            return EigenMap(data.data(), nRows, nCols);
        }

        /// \brief Get a const Eigen::Map view of the grid data for vectorized operations.
        /// \return Const Eigen::Map view of the underlying data.
        ConstEigenMap AsEigen() const
        {
            return ConstEigenMap(data.data(), nRows, nCols);
        }

        /// \brief Compute the sum of all elements using Eigen's vectorized sum.
        /// \return Sum of all elements in the grid.
        T Sum() const
        {
            if (data.empty()) return T{};
            return AsEigen().sum();
        }

        /// \brief Compute row-wise sums using Eigen's vectorized operations.
        /// \return Vector of row sums (one per row).
        Eigen::Matrix<T, Eigen::Dynamic, 1> RowSums() const
        {
            if (data.empty()) return Eigen::Matrix<T, Eigen::Dynamic, 1>();
            return AsEigen().rowwise().sum();
        }

        /// \brief Compute column-wise sums using Eigen's vectorized operations.
        /// \return Vector of column sums (one per column).
        Eigen::Matrix<T, 1, Eigen::Dynamic> ColSums() const
        {
            if (data.empty()) return Eigen::Matrix<T, 1, Eigen::Dynamic>();
            return AsEigen().colwise().sum();
        }

        G4int nRows{0};
        G4int nCols{0};
        std::vector<T> data;
    };

    struct ChargeMatrixSet
    {
        ChargeMatrixSet() = default;
        ChargeMatrixSet(G4int rows, G4int cols) { Resize(rows, cols); }

        virtual void Resize(G4int rows, G4int cols)
        {
            signalFraction.Resize(rows, cols, 0.0);
            signalFractionRow.Resize(rows, cols, 0.0);
            signalFractionCol.Resize(rows, cols, 0.0);
            signalFractionBlock.Resize(rows, cols, 0.0);
            // Neighborhood-based charges
            chargeInduced.Resize(rows, cols, 0.0);
            chargeWithNoise.Resize(rows, cols, 0.0);
            chargeFinal.Resize(rows, cols, 0.0);
            // Row-based charges
            chargeInducedRow.Resize(rows, cols, 0.0);
            chargeWithNoiseRow.Resize(rows, cols, 0.0);
            chargeFinalRow.Resize(rows, cols, 0.0);
            // Col-based charges
            chargeInducedCol.Resize(rows, cols, 0.0);
            chargeWithNoiseCol.Resize(rows, cols, 0.0);
            chargeFinalCol.Resize(rows, cols, 0.0);
            // Block-based charges
            chargeInducedBlock.Resize(rows, cols, 0.0);
            chargeWithNoiseBlock.Resize(rows, cols, 0.0);
            chargeFinalBlock.Resize(rows, cols, 0.0);
        }

        virtual void Clear()
        {
            signalFraction.Clear();
            signalFractionRow.Clear();
            signalFractionCol.Clear();
            signalFractionBlock.Clear();
            chargeInduced.Clear();
            chargeWithNoise.Clear();
            chargeFinal.Clear();
            chargeInducedRow.Clear();
            chargeWithNoiseRow.Clear();
            chargeFinalRow.Clear();
            chargeInducedCol.Clear();
            chargeWithNoiseCol.Clear();
            chargeFinalCol.Clear();
            chargeInducedBlock.Clear();
            chargeWithNoiseBlock.Clear();
            chargeFinalBlock.Clear();
        }

        virtual void Zero()
        {
            if (!signalFraction.Empty()) signalFraction.Fill(0.0);
            if (!signalFractionRow.Empty()) signalFractionRow.Fill(0.0);
            if (!signalFractionCol.Empty()) signalFractionCol.Fill(0.0);
            if (!signalFractionBlock.Empty()) signalFractionBlock.Fill(0.0);
            if (!chargeInduced.Empty()) chargeInduced.Fill(0.0);
            if (!chargeWithNoise.Empty()) chargeWithNoise.Fill(0.0);
            if (!chargeFinal.Empty()) chargeFinal.Fill(0.0);
            if (!chargeInducedRow.Empty()) chargeInducedRow.Fill(0.0);
            if (!chargeWithNoiseRow.Empty()) chargeWithNoiseRow.Fill(0.0);
            if (!chargeFinalRow.Empty()) chargeFinalRow.Fill(0.0);
            if (!chargeInducedCol.Empty()) chargeInducedCol.Fill(0.0);
            if (!chargeWithNoiseCol.Empty()) chargeWithNoiseCol.Fill(0.0);
            if (!chargeFinalCol.Empty()) chargeFinalCol.Fill(0.0);
            if (!chargeInducedBlock.Empty()) chargeInducedBlock.Fill(0.0);
            if (!chargeWithNoiseBlock.Empty()) chargeWithNoiseBlock.Fill(0.0);
            if (!chargeFinalBlock.Empty()) chargeFinalBlock.Fill(0.0);
        }

        G4int Rows() const { return signalFraction.Rows(); }
        G4int Cols() const { return signalFraction.Cols(); }
        bool Empty() const { return signalFraction.Empty(); }

        Grid2D<G4double> signalFraction;      ///< F_i: fraction of total signal
        Grid2D<G4double> signalFractionRow;   ///< F_i normalized by row sum (for 1D row fits)
        Grid2D<G4double> signalFractionCol;   ///< F_i normalized by column sum (for 1D col fits)
        Grid2D<G4double> signalFractionBlock; ///< F_i normalized by 4-pad block sum
        // Neighborhood-based charges
        Grid2D<G4double> chargeInduced;       ///< Induced charge before noise
        Grid2D<G4double> chargeWithNoise;     ///< Charge after adding noise
        Grid2D<G4double> chargeFinal;         ///< Final processed charge
        // Row-based charges
        Grid2D<G4double> chargeInducedRow;    ///< Row-based induced charge
        Grid2D<G4double> chargeWithNoiseRow;  ///< Row-based charge after noise
        Grid2D<G4double> chargeFinalRow;      ///< Row-based final charge
        // Col-based charges
        Grid2D<G4double> chargeInducedCol;    ///< Col-based induced charge
        Grid2D<G4double> chargeWithNoiseCol;  ///< Col-based charge after noise
        Grid2D<G4double> chargeFinalCol;      ///< Col-based final charge
        // Block-based charges
        Grid2D<G4double> chargeInducedBlock;  ///< Block-based induced charge
        Grid2D<G4double> chargeWithNoiseBlock;///< Block-based charge after noise
        Grid2D<G4double> chargeFinalBlock;    ///< Block-based final charge
    };

    struct FullGridCharges : ChargeMatrixSet
    {
        using ChargeMatrixSet::ChargeMatrixSet;

        void Resize(G4int rows, G4int cols) override
        {
            ChargeMatrixSet::Resize(rows, cols);
            distance.Resize(rows, cols, 0.0);
            alpha.Resize(rows, cols, 0.0);
            pixelX.Resize(rows, cols, 0.0);
            pixelY.Resize(rows, cols, 0.0);
        }

        void Clear() override
        {
            ChargeMatrixSet::Clear();
            distance.Clear();
            alpha.Clear();
            pixelX.Clear();
            pixelY.Clear();
        }

        void Zero() override
        {
            ChargeMatrixSet::Zero();
            if (!distance.Empty()) distance.Fill(0.0);
            if (!alpha.Empty()) alpha.Fill(0.0);
            if (!pixelX.Empty()) pixelX.Fill(0.0);
            if (!pixelY.Empty()) pixelY.Fill(0.0);
        }

        /// \brief d_i grid: distance to pixel center.
        Grid2D<G4double> distance;
        /// \brief α_i grid (paper notation): pad angle of view.
        Grid2D<G4double> alpha;
        Grid2D<G4double> pixelX;
        Grid2D<G4double> pixelY;
    };

    struct PatchInfo
    {
        G4int row0{-1};
        G4int col0{-1};
        G4int nRows{0};
        G4int nCols{0};

        G4int Row1() const { return row0 + nRows; }
        G4int Col1() const { return col0 + nCols; }
        bool Valid() const { return row0 >= 0 && col0 >= 0 && nRows > 0 && nCols > 0; }
    };

    struct PatchGridCharges
    {
        void Reset()
        {
            patch = PatchInfo{};
            charges.Clear();
        }

        void Resize(const PatchInfo& info)
        {
            patch = info;
            charges.Resize(info.nRows, info.nCols);
        }

        bool Empty() const { return charges.Empty(); }

        PatchInfo patch;
        ChargeMatrixSet charges;
    };

    struct Result
    {
        struct NeighborCell
        {
            G4int gridIndex{-1};
            G4int globalPixelId{-1};
            G4ThreeVector center{0., 0., 0.};
            G4double fraction{0.0};       ///< Signal fraction (neighborhood denominator)
            G4double fractionRow{0.0};    ///< Signal fraction (row denominator, for 1D row fits)
            G4double fractionCol{0.0};    ///< Signal fraction (column denominator, for 1D col fits)
            G4double fractionBlock{0.0};  ///< Signal fraction (4-pad block denominator)
            G4double charge{0.0};         ///< Charge (neighborhood denominator)
            G4double chargeRow{0.0};      ///< Charge (row denominator)
            G4double chargeCol{0.0};      ///< Charge (col denominator)
            G4double chargeBlock{0.0};    ///< Charge (block denominator)
            /// \brief d_i: distance from hit point to pixel center.
            ///
            /// Note: GEANT4 uses millimeters as base length unit; this value is stored in
            /// those internal units.
            G4double distance{0.0};

            /// \brief α_i (paper notation): pad angle of view.
            G4double alpha{0.0};
        };

        void Reset()
        {
            geometry = PixelGridGeometry{};
            hit = HitInfo{};
            mode = ChargeMode::Patch;
            nearestPixelCenter = G4ThreeVector(0., 0., 0.);
            pixelIndexI = 0;
            pixelIndexJ = 0;
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
        ChargeMode mode{ChargeMode::Patch};
        G4ThreeVector nearestPixelCenter{0., 0., 0.};
        G4int pixelIndexI{0};
        G4int pixelIndexJ{0};
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
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    void SetEmitDistanceAlpha(G4bool enabled) { fEmitDistanceAlpha = enabled; }
    G4bool GetEmitDistanceAlpha() const { return fEmitDistanceAlpha; }
    void SetComputeFullGridFractions(G4bool enabled);
    G4bool GetComputeFullGridFractions() const { return fComputeFullGridFractions; }

    void ResetForEvent();

    const Result& Compute(const G4ThreeVector& hitPos,
                          G4double energyDeposit,
                          G4double ionizationEnergy,
                          G4double amplificationFactor,
                          G4double d0,
                          G4double elementaryCharge);

private:
    /// \brief Validated D0 parameters for charge sharing calculations.
    ///
    /// Contains pre-computed values derived from the D0 reference distance,
    /// ensuring numerical stability in the charge sharing model.
    struct D0Params {
        G4double length;           ///< Validated D0 length (clamped to minimum)
        G4double invLength;        ///< 1.0 / length for efficient division
        G4double minSafeDistance;  ///< Minimum distance to avoid singularities
        G4bool isValid;            ///< True if original D0 was valid

        static constexpr G4double kMinD0 = 1e-6;  ///< Minimum D0 in micrometers
        static constexpr G4double kGuardFactor = 1.0 + 1e-6;
    };

    /// \brief Validated charge model parameters.
    struct ChargeModelParams {
        G4bool useLinear;   ///< True for Linear/DPC models
        G4double beta;      ///< Attenuation coefficient (Linear/DPC only)
        G4double invMicron; ///< 1.0 / micrometer for unit conversion
    };

    D0Params ValidateD0(G4double d0Raw, const char* callerName) const;
    ChargeModelParams GetChargeModelParams(G4double pixelSpacing) const;

    void ReserveBuffers();
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
    G4double CalcDistanceToCenter(G4double dxToCenter,
                                   G4double dyToCenter) const;

    G4double CalcPadViewAngleApprox(G4double distanceToCenter,
                                    G4double padWidth,
                                    G4double padHeight) const;
    void ComputeChargeFractions(const G4ThreeVector& hitPos,
                                G4double totalChargeElectrons,
                                G4double d0,
                                G4double elementaryCharge);
    void BuildOffsets();
    void EnsureFullGridBuffer();
    void ComputeFullGridFractions(const G4ThreeVector& hitPos,
                                  G4double d0,
                                  G4double pixelSize,
                                  G4double pixelSpacing,
                                  G4int numBlocksPerSide,
                                  G4double totalChargeElectrons,
                                  G4double elementaryCharge);
    PixelGridGeometry BuildGridGeometry() const;
    void PopulatePatchFromNeighbors(G4int numBlocksPerSide);

    const DetectorConstruction* fDetector;
    G4int fNeighborhoodRadius;
    Result fResult;

    /// \brief Stores both original and potentially modified weights to avoid recomputation.
    ///
    /// The original weight is preserved for row/col/block fraction calculations,
    /// while the modified weight may be zeroed by DenominatorMode logic.
    struct WeightPair {
        G4double original;   ///< Original computed weight (never modified)
        G4double modified;   ///< Weight after DenominatorMode adjustments
    };
    std::vector<WeightPair> fWeightScratch;
    Grid2D<G4double> fNeighborhoodWeights;  ///< Neighborhood weights for Eigen-based row/col sums
    Grid2D<G4double> fFullGridWeights;
    struct Offset { int di; int dj; int idx; };
    std::vector<Offset> fOffsets;
    int fGridDim{0};
    int fOffsetsDim{0};
    G4bool fEmitDistanceAlpha{false};
    G4bool fComputeFullGridFractions{false};
    G4bool fNeedsReset{true};
};

} // namespace ECS

// Backward compatibility alias
using ChargeSharingCalculator = ECS::ChargeSharingCalculator;

#endif // ECS_CHARGE_SHARING_CALCULATOR_HH
