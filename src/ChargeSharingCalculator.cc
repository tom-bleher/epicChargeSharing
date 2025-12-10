/**
 * @file ChargeSharingCalculator.cc
 * @brief Implementation of charge sharing calculations for AC-LGAD detectors.
 *
 * This file implements the charge sharing model described in Tornago et al.
 * (arXiv:2007.09528) for AC-coupled Low Gain Avalanche Detectors.
 *
 * ## Charge Sharing Model
 *
 * The model computes the fraction of charge collected by each pixel based on
 * the distance from the hit position to the pixel center. Two models are supported:
 *
 * ### Logarithmic Attenuation Model (LogA)
 * Weight_i = log(d0 / d_i), where:
 * - d0 = transverse hit size parameter (Tornago Eq. 4, typically 1 µm)
 * - d_i = distance from hit to pixel i center
 *
 * ### Linear Attenuation Model (LinA, DPC)
 * Weight_i = exp(-beta * d_i / um), where:
 * - beta = attenuation coefficient (pitch-dependent)
 * - d_i = distance from hit to pixel center (converted to micrometers)
 *
 * ## Fraction Calculation
 * For all models, the charge fraction F_i is computed as:
 *   F_i = Weight_i / Sum(Weight_j)
 *
 * ## Noise Application
 * After computing ideal fractions:
 * 1. Qi = F_i * Q_total (ideal charge per pixel)
 * 2. Qn = Qi * gain_noise (multiplicative gain variation)
 * 3. Qf = Qn + electronic_noise (additive noise)
 *
 * @author Tom Bleher, Igor Korover
 * @date 2025
 */
#include "ChargeSharingCalculator.hh"

#include "Config.hh"
#include "DetectorConstruction.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace
{
std::once_flag gInvalidD0WarningFlag;
}

// ============================================================================
// D0 Validation and Charge Model Parameter Helpers
// ============================================================================
//
// The D0 parameter is the reference distance for the logarithmic charge
// sharing model. It defines the scale over which charge is shared between
// neighboring pixels. The ValidateD0 method ensures D0 is positive and
// finite to avoid numerical instabilities (division by zero, log of negative).
//
// The ChargeModelParams helper centralizes the logic for selecting between
// Log and Linear models, computing the beta attenuation coefficient based
// on pixel pitch.
// ============================================================================

/// @brief Validate and prepare D0 parameters for charge sharing calculation.
///
/// Ensures D0 is positive and finite. If invalid, clamps to minimum value
/// and issues a one-time warning. Pre-computes derived values (inverse,
/// minimum safe distance) for efficient use in inner loops.
ChargeSharingCalculator::D0Params
ChargeSharingCalculator::ValidateD0(G4double d0Raw, const char* callerName) const
{
    D0Params params{};
    const G4double rawLength = d0Raw * micrometer;
    const G4double minLength = D0Params::kMinD0 * micrometer;

    params.isValid = std::isfinite(rawLength) && rawLength > 0.0;

    if (!params.isValid) {
        std::call_once(gInvalidD0WarningFlag, [callerName]() {
            G4Exception(callerName, "InvalidD0", JustWarning,
                        "d0 parameter is non-positive; clamping to minimum value.");
        });
        params.length = minLength;
    } else {
        params.length = std::max(rawLength, minLength);
    }

    params.invLength = 1.0 / params.length;
    params.minSafeDistance = params.length * D0Params::kGuardFactor;

    return params;
}

ChargeSharingCalculator::ChargeModelParams
ChargeSharingCalculator::GetChargeModelParams(G4double /*pixelSpacing*/) const
{
    ChargeModelParams params{};

    // Signal model (LogA or LinA) is now independent of reconstruction method.
    // This allows using DPC reconstruction with either signal model.
    params.useLinear = Constants::USES_LINEAR_SIGNAL;
    params.beta = params.useLinear && fDetector
                      ? fDetector->GetLinearChargeModelBeta()
                      : 0.0;
    params.invMicron = 1.0 / micrometer;

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
    const G4double detSize = fDetector->GetDetSize();
    const G4double cornerOffset = fDetector->GetPixelCornerOffset();
    const G4double pixelSize = fDetector->GetPixelSize();
    const G4double firstPixelCenter = -detSize / 2.0 + cornerOffset + pixelSize / 2.0;

    geom.nRows = numBlocks;
    geom.nCols = numBlocks;
    geom.pitchX = spacing;
    geom.pitchY = spacing;
    geom.x0 = detectorPos.x() + firstPixelCenter - 0.5 * spacing;
    geom.y0 = detectorPos.y() + firstPixelCenter - 0.5 * spacing;
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

    PatchInfo info{row0, col0, nRows, nCols};
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
    fWeightScratch.clear();
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
    fWeightScratch.clear();
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
    const G4int newGridDim = 2 * gridRadius + 1;
    const G4int totalCells = newGridDim * newGridDim;
    const std::size_t newSize = static_cast<std::size_t>(totalCells);

    if (newGridDim != fGridDim) {
        fGridDim = newGridDim;
        // Resize neighborhood weights grid for Eigen-based row/col sums
        fNeighborhoodWeights.Resize(newGridDim, newGridDim, 0.0);
    }

    if (fResult.cells.capacity() < newSize) {
        fResult.cells.reserve(newSize);
    }
    if (fWeightScratch.capacity() < newSize) {
        fWeightScratch.reserve(newSize);
    }

    if (fOffsets.empty() || fOffsetsDim != fGridDim) {
        BuildOffsets();
    }
}

G4ThreeVector ChargeSharingCalculator::CalcNearestPixel(const G4ThreeVector& pos)
{
    const auto location = fDetector->FindNearestPixel(pos);
    fResult.pixelIndexI = location.indexI;
    fResult.pixelIndexJ = location.indexJ;
    return location.center;
}

G4double ChargeSharingCalculator::CalcPixelAlphaSubtended(G4double distance,
                                                          G4double pixelWidth,
                                                          G4double pixelHeight) const
{
    const G4double l = (pixelWidth + pixelHeight) / 2.0;

    const G4double numerator = (l / 2.0) * std::sqrt(2.0);
    const G4double denominator = numerator + distance;
    if (distance == 0.0) {
        // atan(1) = pi/4 when distance == 0
        return std::atan(1.0);
    }
    return std::atan(numerator / denominator);
}

void ChargeSharingCalculator::ComputeChargeFractions(const G4ThreeVector& hitPos,
                                                     G4double totalChargeElectrons,
                                                     G4double d0,
                                                     G4double elementaryCharge)
{
    const G4double pixelSize = fDetector->GetPixelSize();
    const G4double pixelSpacing = fDetector->GetPixelSpacing();
    const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();

    // Use validated D0 and charge model parameters
    const D0Params d0p = ValidateD0(d0, "ChargeSharingCalculator::ComputeChargeFractions");
    const ChargeModelParams model = GetChargeModelParams(pixelSpacing);

    const G4double hitX = hitPos.x();
    const G4double hitY = hitPos.y();
    const G4double nearestX = fResult.nearestPixelCenter.x();
    const G4double nearestY = fResult.nearestPixelCenter.y();
    const G4double baseDx = hitX - nearestX;
    const G4double baseDy = hitY - nearestY;
    const G4bool recordDistanceAlpha = fEmitDistanceAlpha;

    G4double totalWeight = 0.0;

    fResult.cells.clear();
    fWeightScratch.clear();

    // Ensure neighborhood weights grid is properly sized and zeroed for Eigen operations
    const int gridDimLocal = 2 * std::max(0, fNeighborhoodRadius) + 1;
    if (fNeighborhoodWeights.Rows() != gridDimLocal || fNeighborhoodWeights.Cols() != gridDimLocal) {
        fNeighborhoodWeights.Resize(gridDimLocal, gridDimLocal, 0.0);
    } else {
        fNeighborhoodWeights.Fill(0.0);
    }

    for (const auto& off : fOffsets) {
        const G4int di = off.di;
        const G4int dj = off.dj;
        const G4int idx = off.idx;

        const G4int gridPixelI = fResult.pixelIndexI + di;
        const G4int gridPixelJ = fResult.pixelIndexJ + dj;

        if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || gridPixelJ < 0 ||
            gridPixelJ >= numBlocksPerSide) {
            // Keep defaults for out-of-bounds (NaNs/-1 set in ResetForEvent)
            continue;
        }

        const G4double pixelCenterX = nearestX + di * pixelSpacing;
        const G4double pixelCenterY = nearestY + dj * pixelSpacing;
        const G4int globalId = gridPixelI * numBlocksPerSide + gridPixelJ;

        const G4double dx = baseDx - di * pixelSpacing;
        const G4double dy = baseDy - dj * pixelSpacing;
        const G4double distanceSquared = dx * dx + dy * dy;
        const G4double distance = std::sqrt(distanceSquared);
        const G4double alpha = CalcPixelAlphaSubtended(distance, pixelSize, pixelSize);

        const G4double safeDistance = std::max(distance, d0p.minSafeDistance);
        const G4double logValue = std::log(safeDistance * d0p.invLength);
        const G4double logWeight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;

        G4double weight = logWeight;
        if (model.useLinear) {
            const G4double attenuation = std::max(0.0, 1.0 - model.beta * distance * model.invMicron);
            weight = attenuation * alpha;
        }

        Cell cell;
        cell.gridIndex = idx;
        cell.globalPixelId = globalId;
        cell.center = {pixelCenterX, pixelCenterY, fResult.nearestPixelCenter.z()};
        // Always store distance for chargeBlock calculation (sorting by d_i)
        cell.distance = distance; 
        if (recordDistanceAlpha) {
            cell.alpha = alpha;
        }
        fResult.cells.push_back(cell);
        fWeightScratch.push_back({weight, weight});  // Store original and modified (initially same)

        // Store weight in neighborhood grid for Eigen-based row/col sums
        const int gridRow = di + std::max(0, fNeighborhoodRadius);
        const int gridCol = dj + std::max(0, fNeighborhoodRadius);
        fNeighborhoodWeights(gridRow, gridCol) = weight;

        // Total weight is accumulated here, but if ChargeBlock mode is active, it will be recomputed below.
        totalWeight += weight;
    }

    const auto activePixelMode = Constants::ACTIVE_PIXEL_MODE;

    // Create index-weight pairs for sorting by F_i (weight)
    struct IndexWeight {
        std::size_t index;
        G4double weight;
    };
    std::vector<IndexWeight> sortedByWeight;
    sortedByWeight.reserve(fResult.cells.size());
    for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
        sortedByWeight.push_back({i, fWeightScratch[i].original});
    }

    // Sort by weight (F_i) in descending order to identify chargeBlock
    std::sort(sortedByWeight.begin(), sortedByWeight.end(),
              [](const IndexWeight& a, const IndexWeight& b) {
                  return a.weight > b.weight;  // Descending order: highest F_i first
              });

    // Identify chargeBlock (top 4 for 2x2, top 9 for 3x3) by highest F_i
    const bool use3x3Block = (activePixelMode == Constants::ActivePixelMode::ChargeBlock3x3);
    const std::size_t nBlock = std::min<std::size_t>(use3x3Block ? 9 : 4, sortedByWeight.size());
    fResult.chargeBlock.clear();
    fResult.chargeBlock.reserve(nBlock);
    for (std::size_t k = 0; k < nBlock; ++k) {
        fResult.chargeBlock.push_back(fResult.cells[sortedByWeight[k].index]);
    }

    // If using ChargeBlock2x2 or ChargeBlock3x3 for denominator, recompute totalWeight and zero out others
    if (activePixelMode == Constants::ActivePixelMode::ChargeBlock2x2 ||
        activePixelMode == Constants::ActivePixelMode::ChargeBlock3x3) {
        // ChargeBlock2x2: top 4 pixels by highest F_i (2x2 block)
        // ChargeBlock3x3: top 9 pixels by highest F_i (3x3 block)
        const std::size_t blockSize = (activePixelMode == Constants::ActivePixelMode::ChargeBlock2x2) ? 4 : 9;
        const std::size_t effectiveBlockSize = std::min(blockSize, sortedByWeight.size());

        // Build a set of block member indices for O(1) lookup
        std::vector<bool> isBlockMember(fResult.cells.size(), false);
        for (std::size_t k = 0; k < effectiveBlockSize; ++k) {
            isBlockMember[sortedByWeight[k].index] = true;
        }

        totalWeight = 0.0;

        // Recompute totalWeight using only block members (this becomes the new denominator)
        for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
             if (isBlockMember[i]) {
                 totalWeight += fWeightScratch[i].modified;
             } else {
                 // Zero out modified weight for non-block members to exclude them from sharing
                 fWeightScratch[i].modified = 0.0;
             }
        }
    }
    else if (activePixelMode == Constants::ActivePixelMode::RowCol ||
             activePixelMode == Constants::ActivePixelMode::RowCol3x3) {
        // RowCol mode: the denominator is the sum over the "main cross" of the neighborhood.
        // RowCol3x3 mode: the cross PLUS the center 3x3 block (cross + 4 corner pixels of 3x3).
        //
        // The main cross consists of:
        //   - Center row (di = 0, all dj values): horizontal arm
        //   - Center column (dj = 0, all di values): vertical arm
        // A pixel is in the cross if di == 0 OR dj == 0.
        //
        // The 3x3 block adds pixels where |di| <= 1 AND |dj| <= 1.

        const int gridRadius = std::max(0, fNeighborhoodRadius);
        const int gridDimLocal = 2 * gridRadius + 1;
        const bool include3x3 = (activePixelMode == Constants::ActivePixelMode::RowCol3x3);

        totalWeight = 0.0;
        for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
            const int idx = fResult.cells[i].gridIndex;
            // idx = (di + gridRadius) * gridDimLocal + (dj + gridRadius)
            // center is at (gridRadius, gridRadius)

            const int row = idx / gridDimLocal; // this is (di + gridRadius)
            const int col = idx % gridDimLocal; // this is (dj + gridRadius)

            // di = row - gridRadius, dj = col - gridRadius
            const int di = row - gridRadius;
            const int dj = col - gridRadius;

            // Check if in the cross: di == 0 OR dj == 0
            const bool inCross = (di == 0) || (dj == 0);
            // Check if in the 3x3 block: |di| <= 1 AND |dj| <= 1
            const bool in3x3 = (std::abs(di) <= 1) && (std::abs(dj) <= 1);

            const bool included = inCross || (include3x3 && in3x3);
            if (included) {
                totalWeight += fWeightScratch[i].modified;
            } else {
                 fWeightScratch[i].modified = 0.0;
            }
        }
    }

    // Use Eigen for vectorized row and column sum computation
    const int gridRadius = std::max(0, fNeighborhoodRadius);
    const Eigen::VectorXd rowWeightSums = fNeighborhoodWeights.RowSums();
    const Eigen::RowVectorXd colWeightSums = fNeighborhoodWeights.ColSums();

    // Compute block sum (sum of the 4 or 9 highest F_i pixels)
    G4double blockWeightSum = 0.0;
    std::vector<bool> isBlockMember(fResult.cells.size(), false);
    for (std::size_t k = 0; k < fResult.chargeBlock.size(); ++k) {
        // Find which cell index corresponds to this chargeBlock member
        for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
            if (fResult.cells[i].gridIndex == fResult.chargeBlock[k].gridIndex) {
                isBlockMember[i] = true;
                blockWeightSum += fWeightScratch[i].original;
                break;
            }
        }
    }

    const int gridDimForBounds = 2 * gridRadius + 1;
    const bool useRowColMode = (activePixelMode == Constants::ActivePixelMode::RowCol ||
                                activePixelMode == Constants::ActivePixelMode::RowCol3x3);

    for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
        auto& cell = fResult.cells[i];
        const G4double modifiedWeight = fWeightScratch[i].modified;
        const G4double fraction =
            (totalWeight > 0.0) ? (modifiedWeight / totalWeight) : 0.0;
        cell.fraction = fraction;
        cell.charge = fraction * totalChargeElectrons * elementaryCharge;

        // Compute row/column/block fractions
        const int idx = cell.gridIndex;
        if (idx >= 0 && idx < gridDimForBounds * gridDimForBounds) {
            const int row = idx / gridDimForBounds;
            const int col = idx % gridDimForBounds;

            const G4double origWeight = fWeightScratch[i].original;

            if (useRowColMode) {
                // RowCol/RowCol3x3 mode: fractionRow and fractionCol are the same as fraction
                // The region (cross or cross+3x3) sums to 1, and we use the same values for row/col reconstruction
                cell.fractionRow = cell.fraction;
                cell.fractionCol = cell.fraction;
            } else {
                // Neighborhood or ChargeBlock mode: use full row/column sums
                const G4double rowSum = rowWeightSums(row);
                const G4double colSum = colWeightSums(col);
                cell.fractionRow = (rowSum > 0.0) ? (origWeight / rowSum) : 0.0;
                cell.fractionCol = (colSum > 0.0) ? (origWeight / colSum) : 0.0;
            }

            // Block fraction: only non-zero for block members
            cell.fractionBlock = (isBlockMember[i] && blockWeightSum > 0.0) ? (origWeight / blockWeightSum) : 0.0;

            // Compute mode-specific charges based on their respective fractions
            const G4double totalChargeCoulomb = totalChargeElectrons * elementaryCharge;
            cell.chargeRow = cell.fractionRow * totalChargeCoulomb;
            cell.chargeCol = cell.fractionCol * totalChargeCoulomb;
            cell.chargeBlock = cell.fractionBlock * totalChargeCoulomb;
        }
    }
}

void ChargeSharingCalculator::BuildOffsets()
{
    const int gridRadius = std::max(0, fNeighborhoodRadius);
    const int gridDimLocal = 2 * gridRadius + 1;
    const int totalOffsets = gridDimLocal * gridDimLocal;

    fOffsets.clear();
    fOffsets.reserve(totalOffsets);
    for (int di = -gridRadius; di <= gridRadius; ++di) {
        for (int dj = -gridRadius; dj <= gridRadius; ++dj) {
            const int idx = (di + gridRadius) * gridDimLocal + (dj + gridRadius);
            fOffsets.push_back(Offset{di, dj, idx});
        }
    }
    fOffsetsDim = gridDimLocal;
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

    // Use validated D0 and charge model parameters
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
    const G4double sigmaNoise = Constants::NOISE_ELECTRON_COUNT * elementaryCharge;
    const bool hasAdditiveNoise = sigmaNoise > 0.0;
    const G4double totalChargeCoulomb = totalChargeElectrons * elementaryCharge;

    const G4ThreeVector& detectorPos = fDetector->GetDetectorPos();
    const G4double detSize = fDetector->GetDetSize();
    const G4double cornerOffset = fDetector->GetPixelCornerOffset();
    const G4double firstPixelPos = -detSize / 2.0 + cornerOffset + pixelSize / 2.0;

    for (G4int i = 0; i < rows; ++i) {
        const G4int di = i - fResult.pixelIndexI;
        for (G4int j = 0; j < cols; ++j) {
            const G4int dj = j - fResult.pixelIndexJ;
            const G4double dx = baseDx - di * pixelSpacing;
            const G4double dy = baseDy - dj * pixelSpacing;
            const G4double distanceSquared = dx * dx + dy * dy;
            const G4double distance = std::sqrt(distanceSquared);
            const G4double alpha = CalcPixelAlphaSubtended(distance, pixelSize, pixelSize);

            const G4double safeDistance = std::max(distance, d0p.minSafeDistance);
            const G4double logValue = std::log(safeDistance * d0p.invLength);
            G4double weight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;
            if (chargeModel.useLinear) {
                const G4double attenuation = std::max(0.0, 1.0 - chargeModel.beta * distance * chargeModel.invMicron);
                weight = attenuation * alpha;
            }

            const G4double pixelCenterX = detectorPos.x() + firstPixelPos + i * pixelSpacing;
            const G4double pixelCenterY = detectorPos.y() + firstPixelPos + j * pixelSpacing;

            fFullGridWeights(i, j) = weight;
            fResult.full.distance(i, j) = distance;
            fResult.full.alpha(i, j) = alpha;
            fResult.full.pixelX(i, j) = pixelCenterX;
            fResult.full.pixelY(i, j) = pixelCenterY;
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

    // Find the highest F_i pixels for block denominator (4 for 2x2, 9 for 3x3)
    const bool useChargeBlock3x3 = (activePixelMode == Constants::ActivePixelMode::ChargeBlock3x3);
    const std::size_t blockTargetSize = useChargeBlock3x3 ? 9 : 4;

    struct PixelWeight {
        G4int i, j;
        G4double weight;  // F_i value (before normalization)
    };
    std::vector<PixelWeight> pixelWeights;
    pixelWeights.reserve(static_cast<std::size_t>(rows * cols));
    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            pixelWeights.push_back({i, j, fFullGridWeights(i, j)});
        }
    }
    // Sort by weight (F_i) in descending order: highest F_i first
    std::partial_sort(pixelWeights.begin(),
                      pixelWeights.begin() + std::min<std::size_t>(blockTargetSize, pixelWeights.size()),
                      pixelWeights.end(),
                      [](const PixelWeight& a, const PixelWeight& b) {
                          return a.weight > b.weight;  // Descending order
                      });

    // Compute block sum (sum of weights for the highest F_i pixels)
    G4double blockSum = 0.0;
    const std::size_t nBlock = std::min<std::size_t>(blockTargetSize, pixelWeights.size());
    for (std::size_t k = 0; k < nBlock; ++k) {
        const G4int bi = pixelWeights[k].i;
        const G4int bj = pixelWeights[k].j;
        blockSum += fFullGridWeights(bi, bj);
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

    // Compute block fractions using precomputed block membership (highest F_i pixels)
    {
        auto fractionBlockEigen = fResult.full.signalFractionBlock.AsEigen();
        fractionBlockEigen.setZero();
        for (std::size_t k = 0; k < nBlock; ++k) {
            const G4int bi = pixelWeights[k].i;
            const G4int bj = pixelWeights[k].j;
            fractionBlockEigen(bi, bj) = fFullGridWeights(bi, bj) * invBlockSum;
        }
    }

    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            const auto idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) +
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
                G4double noisy = base * gainFactor;
                G4double final = noisy + additiveNoise;
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

