#ifndef CHARGESHARINGCALCULATOR_HH
#define CHARGESHARINGCALCULATOR_HH

#include "G4ThreeVector.hh"
#include "globals.hh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

class DetectorConstruction;

/**
 * Encapsulates charge-sharing computations and reusable buffers for the
 * (2r+1)x(2r+1) pixel neighborhood around the hit pixel.
 */
class ChargeSharingCalculator
{
public:
    struct GridGeom
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
            Fi.Resize(rows, cols, 0.0);
            Qi.Resize(rows, cols, 0.0);
            Qn.Resize(rows, cols, 0.0);
            Qf.Resize(rows, cols, 0.0);
        }

        virtual void Clear()
        {
            Fi.Clear();
            Qi.Clear();
            Qn.Clear();
            Qf.Clear();
        }

        virtual void Zero()
        {
            if (!Fi.Empty()) Fi.Fill(0.0);
            if (!Qi.Empty()) Qi.Fill(0.0);
            if (!Qn.Empty()) Qn.Fill(0.0);
            if (!Qf.Empty()) Qf.Fill(0.0);
        }

        G4int Rows() const { return Fi.Rows(); }
        G4int Cols() const { return Fi.Cols(); }
        bool Empty() const { return Fi.Empty(); }

        Grid2D<G4double> Fi;
        Grid2D<G4double> Qi;
        Grid2D<G4double> Qn;
        Grid2D<G4double> Qf;
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

        Grid2D<G4double> distance;
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
            G4double fraction{0.0};
            G4double charge{0.0};
            G4double distance{0.0};
            G4double alpha{0.0};
        };

        void Reset()
        {
            geometry = GridGeom{};
            hit = HitInfo{};
            mode = ChargeMode::Patch;
            nearestPixelCenter = G4ThreeVector(0., 0., 0.);
            pixelIndexI = 0;
            pixelIndexJ = 0;
            gridRadius = 0;
            gridSide = 1;
            totalCells = 0;
            cells.clear();
            full.Zero();
            patch.Reset();
        }

        GridGeom geometry{};
        HitInfo hit{};
        ChargeMode mode{ChargeMode::Patch};
        G4ThreeVector nearestPixelCenter{0., 0., 0.};
        G4int pixelIndexI{0};
        G4int pixelIndexJ{0};
        G4int gridRadius{0};
        G4int gridSide{1};
        std::size_t totalCells{0};
        std::vector<NeighborCell> cells;
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
    void ReserveBuffers();
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
    G4double CalcPixelAlphaSubtended(G4double distance,
                                     G4double pixelWidth,
                                     G4double pixelHeight) const;
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
    GridGeom BuildGridGeometry() const;
    void PopulatePatchFromNeighbors(G4int numBlocksPerSide);

    const DetectorConstruction* fDetector;
    G4int fNeighborhoodRadius;
    Result fResult;
    std::vector<G4double> fWeightScratch;
    Grid2D<G4double> fFullGridWeights;
    struct Offset { int di; int dj; int idx; };
    std::vector<Offset> fOffsets;
    int fGridDim{0};
    int fOffsetsDim{0};
    G4bool fEmitDistanceAlpha{false};
    G4bool fComputeFullGridFractions{false};
    G4bool fNeedsReset{true};
};

#endif // CHARGESHARINGCALCULATOR_HH
