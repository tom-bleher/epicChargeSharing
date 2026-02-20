/**
 * @file ChargeSharingCalculator.cc
 * @brief Implementation of charge sharing calculations for AC-LGAD detectors.
 *
 * This file implements the charge sharing model described in Tornago et al.
 * (arXiv:2007.09528) for AC-coupled Low Gain Avalanche Detectors.
 *
 * ## Charge Sharing Model
 *
 * The model computes the fraction of signal seen by each pad based on
 * the distance from the hit point to the pixel center (d_i).
 *
 * Paper terminology note: the paper refers to readout **pads**. This codebase
 * historically uses the term "pixel" for the same metal pad objects.
 *
 * ### Logarithmic Attenuation Model (LogA)
 *
 * Tornago et al. derive the signal-sharing fraction as:
 *   F_i = (alpha_i / ln(d_i/d0)) / Sum_j(alpha_j / ln(d_j/d0))
 *
 * In this implementation we compute an un-normalized weight
 *   w_i = alpha_i / ln(d_i/d0)
 * and then normalize to obtain F_i.
 *
 * Where:
 * - d0 is the transverse hit size parameter (paper: d_0, typically 1 µm)
 * - d_i is the distance from the hit point to the pixel center
 * - alpha_i is the pad angle of view (paper: α_i)
 *
 * ### Linear Attenuation Model (LinA)
 *
 * The paper also defines a linear attenuation model:
 *   w_i = (1 - beta * d_i) * alpha_i
 *
 * Where beta is the attenuation factor (1/um) and d_i is expressed in µm.
 *
 * ## Fraction Calculation
 * For all models, the charge fraction F_i is computed as:
 *   F_i = w_i / Sum_j(w_j)
 *
 * ## Noise Application
 * After computing ideal fractions:
 * 1. Qi = F_i * Q_total (ideal charge per pad)
 * 2. Qn = Qi * gain_noise (multiplicative gain variation)
 * 3. Qf = Qn + electronic_noise (additive noise)
 *
 * @author Tom Bleher, Igor Korover
 * @date 2025
 */
#include "ChargeSharingCalculator.hh"
#include "ChargeSharingCore.hh"

#include "Config.hh"
#include "RuntimeConfig.hh"
#include "DetectorConstruction.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace csc = epic::chargesharing::core;

namespace
{
std::once_flag gInvalidD0WarningFlag;

/// Map standalone signal model flag to core enum
csc::SignalModel mapSignalModel(bool useLinear)
{
    return useLinear ? csc::SignalModel::LinA : csc::SignalModel::LogA;
}

/// Map standalone ActivePixelMode to core enum
csc::ActivePixelMode mapActivePixelMode(Constants::ActivePixelMode mode)
{
    switch (mode) {
        case Constants::ActivePixelMode::Neighborhood:   return csc::ActivePixelMode::Neighborhood;
        case Constants::ActivePixelMode::RowCol:         return csc::ActivePixelMode::RowCol;
        case Constants::ActivePixelMode::RowCol3x3:      return csc::ActivePixelMode::RowCol3x3;
        case Constants::ActivePixelMode::ChargeBlock2x2: return csc::ActivePixelMode::ChargeBlock2x2;
        case Constants::ActivePixelMode::ChargeBlock3x3: return csc::ActivePixelMode::ChargeBlock3x3;
    }
    return csc::ActivePixelMode::Neighborhood;
}
} // anonymous namespace

// ============================================================================
// D0 Validation and Charge Model Parameter Helpers
// ============================================================================
//
// The D0 parameter is the reference distance for the logarithmic charge
// sharing model. It defines the scale over which charge is shared between
// neighboring pads. The ValidateD0 method ensures D0 is positive and
// finite to avoid numerical instabilities (division by zero, log of negative).
//
// The ChargeModelParams helper centralizes the logic for selecting between
// LogA and LinA models, computing the beta attenuation coefficient based
// on pixel pitch.
// ============================================================================

/// @brief Validate and prepare D0 parameters for charge sharing calculation.
///
/// Ensures D0 is positive and finite. If invalid, clamps to minimum value
/// and issues a one-time warning. Returns both mm and micron values.
/// The core library handles guard logic (min safe distance) internally.
ChargeSharingCalculator::D0Params
ChargeSharingCalculator::ValidateD0(G4double d0Raw, const char* callerName) const
{
    D0Params params{};
    params.isValid = std::isfinite(d0Raw) && d0Raw > 0.0;

    if (!params.isValid) {
        std::call_once(gInvalidD0WarningFlag, [callerName]() {
            G4Exception(callerName, "InvalidD0", JustWarning,
                        "d0 parameter is non-positive; clamping to minimum value.");
        });
        params.micron = D0Params::kMinD0;
    } else {
        params.micron = std::max(d0Raw, D0Params::kMinD0);
    }

    params.lengthMM = params.micron * micrometer;
    return params;
}

ChargeSharingCalculator::ChargeModelParams
ChargeSharingCalculator::GetChargeModelParams(G4double /*pixelSpacing*/) const
{
    ChargeModelParams params{};
    params.useLinear = Constants::USES_LINEAR_SIGNAL;
    params.beta = params.useLinear && fDetector
                      ? fDetector->GetLinearChargeModelBeta()
                      : 0.0;
    return params;
}

// ============================================================================
// Constructor and Configuration
// ============================================================================

ChargeSharingCalculator::ChargeSharingCalculator(const DetectorConstruction* detector)
    : fDetector(detector),
      fNeighborhoodRadius(detector ? detector->GetNeighborhoodRadius()
                                   : Constants::NEIGHBORHOOD_RADIUS)
{
    ReserveBuffers();
}

void ChargeSharingCalculator::SetDetector(const DetectorConstruction* detector)
{
    fDetector = detector;
    if (fDetector && fNeighborhoodRadius <= 0) {
        fNeighborhoodRadius = fDetector->GetNeighborhoodRadius();
    }
    ReserveBuffers();
}

void ChargeSharingCalculator::SetNeighborhoodRadius(G4int radius)
{
    fNeighborhoodRadius = std::max(0, radius);
    ReserveBuffers();
}

void ChargeSharingCalculator::SetComputeFullGridFractions(G4bool enabled)
{
    fComputeFullGridFractions = enabled;
    if (enabled) {
        EnsureFullGridBuffer();
    }
}

ChargeSharingCalculator::PixelGridGeometry ChargeSharingCalculator::BuildGridGeometry() const
{
    PixelGridGeometry geom{};
    if (!fDetector) {
        return geom;
    }

    const G4int numBlocks = std::max(0, fDetector->GetNumBlocksPerSide());
    const G4double spacing = fDetector->GetPixelSpacing();
    const G4ThreeVector& detectorPos = fDetector->GetDetectorPos();
    const G4double gridOffset = fDetector->GetGridOffset();

    // DD4hep-style grid geometry
    // x0, y0 represent the grid origin offset for position calculation
    geom.nRows = numBlocks;
    geom.nCols = numBlocks;
    geom.pitchX = spacing;
    geom.pitchY = spacing;
    geom.x0 = detectorPos.x() + gridOffset;
    geom.y0 = detectorPos.y() + gridOffset;
    return geom;
}

void ChargeSharingCalculator::PopulatePatchFromNeighbors(G4int numBlocksPerSide)
{
    fResult.patch.Reset();
    if (numBlocksPerSide <= 0) {
        return;
    }

    const G4int radius = std::max(0, fNeighborhoodRadius);
    const G4int row0 = std::max(0, fResult.pixelIndexI - radius);
    const G4int col0 = std::max(0, fResult.pixelIndexJ - radius);
    const G4int row1 = std::min(numBlocksPerSide, fResult.pixelIndexI + radius + 1);
    const G4int col1 = std::min(numBlocksPerSide, fResult.pixelIndexJ + radius + 1);
    const G4int nRows = std::max(0, row1 - row0);
    const G4int nCols = std::max(0, col1 - col0);
    if (nRows <= 0 || nCols <= 0) {
        return;
    }

    const PatchInfo info{row0, col0, nRows, nCols};
    fResult.patch.Resize(info);
    fResult.patch.charges.Zero();

    for (const auto& cell : fResult.cells) {
        if (cell.globalPixelId < 0) {
            continue;
        }
        const G4int globalRow = cell.globalPixelId / numBlocksPerSide;
        const G4int globalCol = cell.globalPixelId % numBlocksPerSide;
        if (globalRow < row0 || globalRow >= row1 || globalCol < col0 || globalCol >= col1) {
            continue;
        }
        const G4int localRow = globalRow - row0;
        const G4int localCol = globalCol - col0;
        fResult.patch.charges.signalFraction(localRow, localCol) = cell.fraction;
        fResult.patch.charges.signalFractionRow(localRow, localCol) = cell.fractionRow;
        fResult.patch.charges.signalFractionCol(localRow, localCol) = cell.fractionCol;
        fResult.patch.charges.signalFractionBlock(localRow, localCol) = cell.fractionBlock;
        fResult.patch.charges.chargeInduced(localRow, localCol) = cell.charge;
        fResult.patch.charges.chargeWithNoise(localRow, localCol) = cell.charge;
        fResult.patch.charges.chargeFinal(localRow, localCol) = cell.charge;
    }
}

void ChargeSharingCalculator::ResetForEvent()
{
    fResult.Reset();
    if (!fNeighborhoodWeights.Empty()) {
        fNeighborhoodWeights.Fill(0.0);
    }
    if (!fFullGridWeights.Empty()) {
        fFullGridWeights.Fill(0.0);
    }
    fNeedsReset = false;
}

const ChargeSharingCalculator::Result& ChargeSharingCalculator::Compute(
    const G4ThreeVector& hitPos,
    G4double energyDeposit,
    G4double ionizationEnergy,
    G4double amplificationFactor,
    G4double d0,
    G4double elementaryCharge)
{
    if (!fDetector) {
        G4Exception("ChargeSharingCalculator::Compute",
                    "MissingDetector",
                    FatalException,
                    "DetectorConstruction pointer is null.");
    }

    ReserveBuffers();
    if (fNeedsReset) {
        ResetForEvent();
    }
    fNeedsReset = true;

    fResult.mode = fComputeFullGridFractions ? ChargeMode::FullGrid : ChargeMode::Patch;
    fResult.gridRadius = fNeighborhoodRadius;
    fResult.gridSide = fGridDim;
    fResult.totalCells = static_cast<std::size_t>(fGridDim) * static_cast<std::size_t>(fGridDim);
    fResult.cells.clear();
    fResult.nearestPixelCenter = CalcNearestPixel(hitPos);
    fResult.hit.trueX = hitPos.x();
    fResult.hit.trueY = hitPos.y();
    fResult.hit.trueZ = hitPos.z();
    fResult.hit.pixRow = fResult.pixelIndexI;
    fResult.hit.pixCol = fResult.pixelIndexJ;
    fResult.hit.pixCenterX = fResult.nearestPixelCenter.x();
    fResult.hit.pixCenterY = fResult.nearestPixelCenter.y();
    fResult.geometry = BuildGridGeometry();

    const G4double edepInEV = energyDeposit / eV;
    const G4double primaryElectrons = ionizationEnergy > 0.0 ? (edepInEV / ionizationEnergy) : 0.0;
    const G4double totalChargeElectrons =
        std::max(0.0, primaryElectrons * amplificationFactor);

    ComputeChargeFractions(hitPos, totalChargeElectrons, d0, elementaryCharge);

    const G4int numBlocksPerSide = fDetector ? fDetector->GetNumBlocksPerSide() : 0;
    PopulatePatchFromNeighbors(numBlocksPerSide);

    if (fComputeFullGridFractions) {
        ComputeFullGridFractions(hitPos,
                                 d0,
                                 fDetector->GetPixelSize(),
                                 fDetector->GetPixelSpacing(),
                                 numBlocksPerSide,
                                 totalChargeElectrons,
                                 elementaryCharge);
    } else {
        fResult.full.Clear();
        fFullGridWeights.Clear();
    }

    return fResult;
}

using Cell = ChargeSharingCalculator::Result::NeighborCell;

void ChargeSharingCalculator::ReserveBuffers()
{
    const G4int gridRadius = std::max(0, fNeighborhoodRadius);
    const G4int newGridDim = (2 * gridRadius) + 1;
    const std::size_t newSize = static_cast<std::size_t>(newGridDim) * static_cast<std::size_t>(newGridDim);

    if (newGridDim != fGridDim) {
        fGridDim = newGridDim;
        fNeighborhoodWeights.Resize(newGridDim, newGridDim, 0.0);
    }

    if (fResult.cells.capacity() < newSize) {
        fResult.cells.reserve(newSize);
    }
}

G4ThreeVector ChargeSharingCalculator::CalcNearestPixel(const G4ThreeVector& pos)
{
    const auto location = fDetector->FindNearestPixel(pos);
    fResult.pixelIndexI = location.indexI;
    fResult.pixelIndexJ = location.indexJ;
    return location.center;
}


void ChargeSharingCalculator::ComputeChargeFractions(const G4ThreeVector& hitPos,
                                                     G4double totalChargeElectrons,
                                                     G4double d0,
                                                     G4double elementaryCharge)
{
    const G4double pixelSize = fDetector->GetPixelSize();
    const G4double pixelSpacing = fDetector->GetPixelSpacing();
    const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
    const G4double totalChargeCoulomb = totalChargeElectrons * elementaryCharge;

    const D0Params d0p = ValidateD0(d0, "ChargeSharingCalculator::ComputeChargeFractions");
    const ChargeModelParams model = GetChargeModelParams(pixelSpacing);

    const int radius = std::max(0, fNeighborhoodRadius);
    const int gridDim = (2 * radius) + 1;
    const G4bool recordDistanceAlpha = fEmitDistanceAlpha;
    const auto activePixelMode = Constants::ACTIVE_PIXEL_MODE;
    const bool useRowColMode = (activePixelMode == Constants::ActivePixelMode::RowCol ||
                                activePixelMode == Constants::ActivePixelMode::RowCol3x3);

    // Build core neighborhood config and delegate physics computation
    csc::NeighborhoodConfig coreConfig;
    coreConfig.signalModel = mapSignalModel(model.useLinear);
    coreConfig.activeMode = mapActivePixelMode(activePixelMode);
    coreConfig.radius = radius;
    coreConfig.pixelSizeMM = pixelSize;
    coreConfig.pixelSizeYMM = pixelSize;
    coreConfig.pixelSpacingMM = pixelSpacing;
    coreConfig.pixelSpacingYMM = pixelSpacing;
    coreConfig.d0Micron = d0p.micron;
    coreConfig.betaPerMicron = model.beta;
    coreConfig.numPixelsX = numBlocksPerSide;
    coreConfig.numPixelsY = numBlocksPerSide;

    const auto coreResult = csc::calculateNeighborhood(
        hitPos.x(), hitPos.y(),
        fResult.pixelIndexI, fResult.pixelIndexJ,
        fResult.nearestPixelCenter.x(), fResult.nearestPixelCenter.y(),
        coreConfig);

    // Build Eigen weight grid from core result for per-row/per-col normalization
    if (fNeighborhoodWeights.Rows() != gridDim || fNeighborhoodWeights.Cols() != gridDim) {
        fNeighborhoodWeights.Resize(gridDim, gridDim, 0.0);
    } else {
        fNeighborhoodWeights.Fill(0.0);
    }

    for (const auto& pixel : coreResult.pixels) {
        if (pixel.inBounds) {
            fNeighborhoodWeights(pixel.di + radius, pixel.dj + radius) = pixel.weight;
        }
    }

    const Eigen::VectorXd rowWeightSums = fNeighborhoodWeights.RowSums();
    const Eigen::RowVectorXd colWeightSums = fNeighborhoodWeights.ColSums();

    // Compute block fractions from core weights.
    // The core always computes a 2x2 block for fractionBlock diagnostics, but the
    // standalone uses 3x3 when in ChargeBlock3x3 mode. Rebuild the block here to
    // match the standalone's original behavior.
    const bool use3x3Block = (activePixelMode == Constants::ActivePixelMode::ChargeBlock3x3);
    const int blockSize = use3x3Block ? 3 : 2;

    // Find pixel with highest weight
    int maxDi = 0, maxDj = 0;
    G4double maxWeight = -1.0;
    for (const auto& pixel : coreResult.pixels) {
        if (pixel.inBounds && pixel.weight > maxWeight) {
            maxWeight = pixel.weight;
            maxDi = pixel.di;
            maxDj = pixel.dj;
        }
    }

    // Find best block containing max-weight pixel
    int blockCornerDi = maxDi, blockCornerDj = maxDj;
    G4double bestBlockSum = -1.0;
    const int maxOffset = use3x3Block ? -2 : -1;

    for (int oi = maxOffset; oi <= 0; ++oi) {
        for (int oj = maxOffset; oj <= 0; ++oj) {
            const int testDi = maxDi + oi;
            const int testDj = maxDj + oj;

            G4double sum = 0.0;
            for (const auto& p : coreResult.pixels) {
                if (p.inBounds &&
                    p.di >= testDi && p.di < testDi + blockSize &&
                    p.dj >= testDj && p.dj < testDj + blockSize) {
                    sum += p.weight;
                }
            }

            if (sum > bestBlockSum) {
                bestBlockSum = sum;
                blockCornerDi = testDi;
                blockCornerDj = testDj;
            }
        }
    }

    const G4double invBlockSum = (bestBlockSum > 0.0) ? (1.0 / bestBlockSum) : 0.0;

    // Map core result to standalone cells
    fResult.cells.clear();
    fResult.chargeBlock.clear();

    for (const auto& pixel : coreResult.pixels) {
        if (!pixel.inBounds) {
            continue;
        }

        Cell cell;
        cell.gridIndex = ((pixel.di + radius) * gridDim) + (pixel.dj + radius);
        cell.globalPixelId = pixel.globalIndex;
        cell.center = {pixel.centerX, pixel.centerY, fResult.nearestPixelCenter.z()};
        cell.distance = pixel.distance;
        if (recordDistanceAlpha) {
            cell.alpha = pixel.alpha;
        }

        // Main fraction from core (handles active pixel mode denominator)
        cell.fraction = pixel.fraction;
        cell.charge = pixel.fraction * totalChargeCoulomb;

        // Per-row/per-col fractions via Eigen normalization (backward-compatible)
        const int gridRow = pixel.di + radius;
        const int gridCol = pixel.dj + radius;

        if (useRowColMode) {
            cell.fractionRow = cell.fraction;
            cell.fractionCol = cell.fraction;
        } else {
            const G4double rowSum = rowWeightSums(gridRow);
            const G4double colSum = colWeightSums(gridCol);
            cell.fractionRow = (rowSum > 0.0) ? (pixel.weight / rowSum) : 0.0;
            cell.fractionCol = (colSum > 0.0) ? (pixel.weight / colSum) : 0.0;
        }

        // Block fraction: only non-zero for block members
        const bool inBlock = (pixel.di >= blockCornerDi && pixel.di < blockCornerDi + blockSize &&
                              pixel.dj >= blockCornerDj && pixel.dj < blockCornerDj + blockSize);
        cell.fractionBlock = inBlock ? (pixel.weight * invBlockSum) : 0.0;

        cell.chargeRow = cell.fractionRow * totalChargeCoulomb;
        cell.chargeCol = cell.fractionCol * totalChargeCoulomb;
        cell.chargeBlock = cell.fractionBlock * totalChargeCoulomb;

        fResult.cells.push_back(cell);

        if (inBlock) {
            fResult.chargeBlock.push_back(cell);
        }
    }
}


void ChargeSharingCalculator::EnsureFullGridBuffer()
{
    if (!fDetector) {
        fResult.full.Clear();
        fFullGridWeights.Clear();
        return;
    }
    const G4int numBlocksPerSide = std::max(0, fDetector->GetNumBlocksPerSide());
    if (numBlocksPerSide <= 0) {
        fResult.full.Clear();
        fFullGridWeights.Clear();
        return;
    }
    if (fResult.full.Rows() != numBlocksPerSide || fResult.full.Cols() != numBlocksPerSide) {
        fResult.full.Resize(numBlocksPerSide, numBlocksPerSide);
    } else {
        fResult.full.Zero();
    }

    if (fFullGridWeights.Rows() != numBlocksPerSide || fFullGridWeights.Cols() != numBlocksPerSide) {
        fFullGridWeights.Resize(numBlocksPerSide, numBlocksPerSide, 0.0);
    } else if (!fFullGridWeights.Empty()) {
        fFullGridWeights.Fill(0.0);
    }
}

void ChargeSharingCalculator::ComputeFullGridFractions(const G4ThreeVector& hitPos,
                                                       G4double d0,
                                                       G4double pixelSize,
                                                       G4double pixelSpacing,
                                                       G4int numBlocksPerSide,
                                                       G4double totalChargeElectrons,
                                                       G4double elementaryCharge)
{
    if (!fDetector) {
        fResult.full.Clear();
        fFullGridWeights.Clear();
        return;
    }

    const G4int gridSide = std::max(0, numBlocksPerSide);
    if (gridSide <= 0) {
        fResult.full.Clear();
        fFullGridWeights.Clear();
        return;
    }

    EnsureFullGridBuffer();
    const G4int rows = fResult.full.Rows();
    const G4int cols = fResult.full.Cols();
    if (rows <= 0 || cols <= 0) {
        return;
    }
    fResult.full.Zero();
    if (!fFullGridWeights.Empty()) {
        fFullGridWeights.Fill(0.0);
    }
    fResult.geometry = BuildGridGeometry();
    fResult.geometry.pitchX = pixelSpacing;
    fResult.geometry.pitchY = pixelSpacing;

    // Validated D0 and charge model parameters
    const D0Params d0p = ValidateD0(d0, "ChargeSharingCalculator::ComputeFullGridFractions");
    const ChargeModelParams chargeModel = GetChargeModelParams(pixelSpacing);

    const G4double hitX = hitPos.x();
    const G4double hitY = hitPos.y();
    const G4double nearestX = fResult.nearestPixelCenter.x();
    const G4double nearestY = fResult.nearestPixelCenter.y();
    const G4double baseDx = hitX - nearestX;
    const G4double baseDy = hitY - nearestY;

    const auto& gainSigmas = fDetector->GetPixelGainSigmas();
    const std::size_t gainCount = gainSigmas.size();
    const bool hasGainNoise = gainCount > 0;
    const G4double sigmaNoise = ECS::RuntimeConfig::Instance().noiseElectronCount * elementaryCharge;
    const bool hasAdditiveNoise = sigmaNoise > 0.0;
    const G4double totalChargeCoulomb = totalChargeElectrons * elementaryCharge;

    const G4ThreeVector& detectorPos = fDetector->GetDetectorPos();
    const G4double gridOffset = fDetector->GetGridOffset();
    const G4int minIndexX = fDetector->GetMinIndexX();
    const G4int minIndexY = fDetector->GetMinIndexY();

    for (G4int localI = 0; localI < rows; ++localI) {
        const G4int gridI = localI + minIndexX;
        const G4int di = gridI - fResult.pixelIndexI;

        for (G4int localJ = 0; localJ < cols; ++localJ) {
            const G4int gridJ = localJ + minIndexY;
            const G4int dj = gridJ - fResult.pixelIndexJ;

            const G4double dxToCenter = baseDx - (di * pixelSpacing);
            const G4double dyToCenter = baseDy - (dj * pixelSpacing);
            const G4double distanceToCenter = csc::calcDistanceToCenter(dxToCenter, dyToCenter);
            const G4double alpha = csc::calcPadViewAngle(distanceToCenter, pixelSize, pixelSize);

            // Delegate weight calculation to core (handles guard logic internally)
            G4double weight;
            if (chargeModel.useLinear) {
                weight = csc::calcWeightLinA(distanceToCenter, alpha, chargeModel.beta);
            } else {
                weight = csc::calcWeightLogA(distanceToCenter, alpha, d0p.lengthMM);
            }

            // DD4hep formula: position = index * pitch + offset
            const G4double pixelCenterX = detectorPos.x() + Constants::IndexToPosition(gridI, pixelSpacing, gridOffset);
            const G4double pixelCenterY = detectorPos.y() + Constants::IndexToPosition(gridJ, pixelSpacing, gridOffset);

            fFullGridWeights(localI, localJ) = weight;
            fResult.full.distance(localI, localJ) = distanceToCenter;
            fResult.full.alpha(localI, localJ) = alpha;
            fResult.full.pixelX(localI, localJ) = pixelCenterX;
            fResult.full.pixelY(localI, localJ) = pixelCenterY;
        }
    }

    // Use Eigen for vectorized computation
    auto weightsEigen = fFullGridWeights.AsEigen();
    const auto activePixelMode = Constants::ACTIVE_PIXEL_MODE;
    const bool useRowColMode = (activePixelMode == Constants::ActivePixelMode::RowCol ||
                                activePixelMode == Constants::ActivePixelMode::RowCol3x3);
    const bool include3x3 = (activePixelMode == Constants::ActivePixelMode::RowCol3x3);

    // Get center pixel indices (the hit pixel)
    const G4int centerI = fResult.pixelIndexI;
    const G4int centerJ = fResult.pixelIndexJ;

    // Helper lambda to check if pixel (i,j) is in the included region for RowCol modes
    auto isInRowColRegion = [&](G4int i, G4int j) -> bool {
        const G4int di = i - centerI;
        const G4int dj = j - centerJ;
        const bool inCross = (di == 0) || (dj == 0);
        const bool in3x3 = (std::abs(di) <= 1) && (std::abs(dj) <= 1);
        return inCross || (include3x3 && in3x3);
    };

    // Compute total weight based on denominator mode
    G4double totalWeight = 0.0;
    if (useRowColMode) {
        // RowCol/RowCol3x3 mode: sum only the cross (+ 3x3 block for RowCol3x3)
        for (G4int i = 0; i < rows; ++i) {
            for (G4int j = 0; j < cols; ++j) {
                if (isInRowColRegion(i, j)) {
                    totalWeight += fFullGridWeights(i, j);
                }
            }
        }
    } else {
        totalWeight = fFullGridWeights.Sum();
    }

    // Find contiguous block for ChargeBlock modes (2x2 or 3x3 square)
    const bool useChargeBlock3x3 = (activePixelMode == Constants::ActivePixelMode::ChargeBlock3x3);
    const int blockSizeDim = useChargeBlock3x3 ? 3 : 2;

    // Find the pixel with highest F_i
    G4int maxI = centerI, maxJ = centerJ;
    G4double maxWeight = 0.0;
    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            if (fFullGridWeights(i, j) > maxWeight) {
                maxWeight = fFullGridWeights(i, j);
                maxI = i;
                maxJ = j;
            }
        }
    }

    // Determine block corner (upper-left corner of the contiguous block)
    // For both 2x2 and 3x3: find the block containing the highest F_i pixel with maximum total F_i
    G4int blockCornerI = 0, blockCornerJ = 0;
    {
        G4double bestSum = -1.0;
        G4int bestCornerI = maxI, bestCornerJ = maxJ;
        const int maxOffset = useChargeBlock3x3 ? -2 : -1;  // -2 to 0 for 3x3, -1 to 0 for 2x2

        for (int offsetI = maxOffset; offsetI <= 0; ++offsetI) {
            for (int offsetJ = maxOffset; offsetJ <= 0; ++offsetJ) {
                const G4int testCornerI = maxI + offsetI;
                const G4int testCornerJ = maxJ + offsetJ;

                // Compute sum of this block
                G4double sum = 0.0;
                for (int bi = 0; bi < blockSizeDim; ++bi) {
                    for (int bj = 0; bj < blockSizeDim; ++bj) {
                        const G4int pi = testCornerI + bi;
                        const G4int pj = testCornerJ + bj;
                        if (pi >= 0 && pi < rows && pj >= 0 && pj < cols) {
                            sum += fFullGridWeights(pi, pj);
                        }
                    }
                }

                if (sum > bestSum) {
                    bestSum = sum;
                    bestCornerI = testCornerI;
                    bestCornerJ = testCornerJ;
                }
            }
        }
        blockCornerI = bestCornerI;
        blockCornerJ = bestCornerJ;
    }

    // Helper to check if pixel (i,j) is in the contiguous block
    auto isInBlock = [&](G4int i, G4int j) -> bool {
        return i >= blockCornerI && i < blockCornerI + blockSizeDim &&
               j >= blockCornerJ && j < blockCornerJ + blockSizeDim;
    };

    // Compute block sum (sum of weights for pixels in the contiguous block)
    G4double blockSum = 0.0;
    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            if (isInBlock(i, j)) {
                blockSum += fFullGridWeights(i, j);
            }
        }
    }
    const G4double invBlockSum = (blockSum > 0.0) ? (1.0 / blockSum) : 0.0;

    auto fractionsEigen = fResult.full.signalFraction.AsEigen();

    // Compute main fractions based on denominator mode
    if (useRowColMode) {
        // RowCol/RowCol3x3 mode: only pixels in the region get non-zero fractions
        fractionsEigen.setZero();
        if (totalWeight > 0.0) {
            for (G4int i = 0; i < rows; ++i) {
                for (G4int j = 0; j < cols; ++j) {
                    if (isInRowColRegion(i, j)) {
                        fractionsEigen(i, j) = fFullGridWeights(i, j) / totalWeight;
                    }
                }
            }
        }
    } else {
        // Neighborhood or ChargeBlock mode: use full grid sum
        if (totalWeight > 0.0) {
            fractionsEigen = weightsEigen / totalWeight;
        } else {
            fractionsEigen.setZero();
        }
    }

    // Compute row-normalized fractions
    {
        auto fractionRowEigen = fResult.full.signalFractionRow.AsEigen();
        if (useRowColMode) {
            // RowCol/RowCol3x3 mode: fractionRow is same as fraction (region-based, sums to 1 over region)
            fractionRowEigen = fractionsEigen;
        } else {
            // Other modes: each row normalized by its own sum
            Eigen::VectorXd rowSumsEigen = weightsEigen.rowwise().sum();
            for (G4int i = 0; i < rows; ++i) {
                const G4double invRowSum = (rowSumsEigen(i) > 0.0) ? (1.0 / rowSumsEigen(i)) : 0.0;
                fractionRowEigen.row(i) = weightsEigen.row(i) * invRowSum;
            }
        }
    }

    // Compute column-normalized fractions
    {
        auto fractionColEigen = fResult.full.signalFractionCol.AsEigen();
        if (useRowColMode) {
            // RowCol/RowCol3x3 mode: fractionCol is same as fraction (region-based, sums to 1 over region)
            fractionColEigen = fractionsEigen;
        } else {
            // Other modes: each column normalized by its own sum
            Eigen::RowVectorXd colSumsEigen = weightsEigen.colwise().sum();
            for (G4int j = 0; j < cols; ++j) {
                const G4double invColSum = (colSumsEigen(j) > 0.0) ? (1.0 / colSumsEigen(j)) : 0.0;
                fractionColEigen.col(j) = weightsEigen.col(j) * invColSum;
            }
        }
    }

    // Compute block fractions using contiguous block membership
    {
        auto fractionBlockEigen = fResult.full.signalFractionBlock.AsEigen();
        fractionBlockEigen.setZero();
        for (G4int i = 0; i < rows; ++i) {
            for (G4int j = 0; j < cols; ++j) {
                if (isInBlock(i, j)) {
                    fractionBlockEigen(i, j) = fFullGridWeights(i, j) * invBlockSum;
                }
            }
        }
    }

    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            const auto idx = (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) +
                             static_cast<std::size_t>(j);
            // Fractions already computed via Eigen above
            const G4double fraction = fResult.full.signalFraction(i, j);
            const G4double fractionRow = fResult.full.signalFractionRow(i, j);
            const G4double fractionCol = fResult.full.signalFractionCol(i, j);
            const G4double fractionBlock = fResult.full.signalFractionBlock(i, j);

            // Compute base charges for each mode
            const G4double baseCharge = fraction * totalChargeCoulomb;
            const G4double baseChargeRow = fractionRow * totalChargeCoulomb;
            const G4double baseChargeCol = fractionCol * totalChargeCoulomb;
            const G4double baseChargeBlock = fractionBlock * totalChargeCoulomb;

            // Apply noise model (same noise factors for all modes)
            G4double gainFactor = 1.0;
            if (hasGainNoise && idx < gainCount) {
                const G4double sigmaGain = gainSigmas[idx];
                if (sigmaGain > 0.0) {
                    gainFactor = G4RandGauss::shoot(1.0, sigmaGain);
                }
            }
            G4double additiveNoise = 0.0;
            if (hasAdditiveNoise) {
                additiveNoise = G4RandGauss::shoot(0.0, sigmaNoise);
            }

            // Apply noise to all modes
            auto applyNoise = [gainFactor, additiveNoise](G4double base) {
                const G4double noisy = base * gainFactor;
                const G4double final = noisy + additiveNoise;
                return std::make_tuple(base, noisy, std::max(0.0, final));
            };

            auto [qi, qn, qf] = applyNoise(baseCharge);
            auto [qiRow, qnRow, qfRow] = applyNoise(baseChargeRow);
            auto [qiCol, qnCol, qfCol] = applyNoise(baseChargeCol);
            auto [qiBlock, qnBlock, qfBlock] = applyNoise(baseChargeBlock);

            // Store fractions
            fResult.full.signalFraction(i, j) = fraction;
            fResult.full.signalFractionRow(i, j) = fractionRow;
            fResult.full.signalFractionCol(i, j) = fractionCol;
            fResult.full.signalFractionBlock(i, j) = fractionBlock;

            // Store neighborhood-mode charges
            fResult.full.chargeInduced(i, j) = qi;
            fResult.full.chargeWithNoise(i, j) = qn;
            fResult.full.chargeFinal(i, j) = qf;

            // Store row-mode charges
            fResult.full.chargeInducedRow(i, j) = qiRow;
            fResult.full.chargeWithNoiseRow(i, j) = qnRow;
            fResult.full.chargeFinalRow(i, j) = qfRow;

            // Store col-mode charges
            fResult.full.chargeInducedCol(i, j) = qiCol;
            fResult.full.chargeWithNoiseCol(i, j) = qnCol;
            fResult.full.chargeFinalCol(i, j) = qfCol;

            // Store block-mode charges
            fResult.full.chargeInducedBlock(i, j) = qiBlock;
            fResult.full.chargeWithNoiseBlock(i, j) = qnBlock;
            fResult.full.chargeFinalBlock(i, j) = qfBlock;
        }
    }
}
