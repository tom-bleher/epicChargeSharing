/**
 * @file ChargeSharingCalculator.cc
 */
#include "ChargeSharingCalculator.hh"

#include "Constants.hh"
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

ChargeSharingCalculator::GridGeom ChargeSharingCalculator::BuildGridGeometry() const
{
    GridGeom geom{};
    if (!fDetector) {
        return geom;
    }

    const G4int numBlocks = std::max(0, fDetector->GetNumBlocksPerSide());
    const G4double spacing = fDetector->GetPixelSpacing();
    const G4ThreeVector detectorPos = fDetector->GetDetectorPos();
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
        fResult.patch.charges.Fi(localRow, localCol) = cell.fraction;
        fResult.patch.charges.Qi(localRow, localCol) = cell.charge;
        fResult.patch.charges.Qn(localRow, localCol) = cell.charge;
        fResult.patch.charges.Qf(localRow, localCol) = cell.charge;
    }
}

void ChargeSharingCalculator::ResetForEvent()
{
    fResult.Reset();
    fWeightScratch.clear();
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

    const G4double rawD0Length = d0 * micrometer;
    constexpr G4double minD0Length = 1e-6 * micrometer;
    G4double d0Length = rawD0Length;
    if (!std::isfinite(d0Length) || d0Length <= 0.0) {
        std::call_once(gInvalidD0WarningFlag,
                       []() {
                           G4Exception("ChargeSharingCalculator::ComputeChargeFractions",
                                       "InvalidD0",
                                       JustWarning,
                                       "d0 parameter is non-positive; clamping to minimum value to avoid instability.");
                       });
        d0Length = minD0Length;
    }
    d0Length = std::max(d0Length, minD0Length);
    const G4double hitX = hitPos.x();
    const G4double hitY = hitPos.y();

    constexpr G4double guardFactor = 1.0 + 1e-6;
    const G4double minSafeDistance = d0Length * guardFactor;
    const G4double invD0Length = 1.0 / d0Length;
    const G4double invMicrometer = 1.0 / micrometer;

    const G4double nearestX = fResult.nearestPixelCenter.x();
    const G4double nearestY = fResult.nearestPixelCenter.y();
    const G4double baseDx = hitX - nearestX;
    const G4double baseDy = hitY - nearestY;
    const G4bool recordDistanceAlpha = fEmitDistanceAlpha;

    const auto chargeModel = Constants::CHARGE_SHARING_MODEL;
    const bool useLinearModel = (chargeModel == Constants::ChargeSharingModel::Linear);
    const G4double beta = useLinearModel ? fDetector->GetLinearChargeModelBeta(pixelSpacing) : 0.0;

    G4double totalWeight = 0.0;

    fResult.cells.clear();
    fWeightScratch.clear();

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

        const G4double safeDistance = std::max(distance, minSafeDistance);
        const G4double logValue = std::log(safeDistance * invD0Length);
        const G4double logWeight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;

        G4double weight = logWeight;
        if (useLinearModel) {
            const G4double attenuation = std::max(0.0, 1.0 - beta * distance * invMicrometer);
            weight = attenuation * alpha;
        }

        Cell cell;
        cell.gridIndex = idx;
        cell.globalPixelId = globalId;
        cell.center = {pixelCenterX, pixelCenterY, fResult.nearestPixelCenter.z()};
        if (recordDistanceAlpha) {
            cell.distance = distance;
            cell.alpha = alpha;
        }
        fResult.cells.push_back(cell);
        fWeightScratch.push_back(weight);
        totalWeight += weight;
    }

    for (std::size_t i = 0; i < fResult.cells.size(); ++i) {
        auto& cell = fResult.cells[i];
        const G4double weight = fWeightScratch[i];
        const G4double fraction =
            (totalWeight > 0.0) ? (weight / totalWeight) : 0.0;
        cell.fraction = fraction;
        cell.charge = fraction * totalChargeElectrons * elementaryCharge;
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

    const G4double rawD0Length = d0 * micrometer;
    constexpr G4double minD0Length = 1e-6 * micrometer;
    G4double d0Length = rawD0Length;
    if (!std::isfinite(d0Length) || d0Length <= 0.0) {
        std::call_once(gInvalidD0WarningFlag,
                       []() {
                           G4Exception("ChargeSharingCalculator::ComputeFullGridFractions",
                                       "InvalidD0",
                                       JustWarning,
                                       "d0 parameter is non-positive; clamping to minimum value to avoid instability.");
                       });
        d0Length = minD0Length;
    }
    d0Length = std::max(d0Length, minD0Length);

    constexpr G4double guardFactor = 1.0 + 1e-6;
    const G4double minSafeDistance = d0Length * guardFactor;
    const G4double invD0Length = 1.0 / d0Length;
    const G4double invMicrometer = 1.0 / micrometer;

    const auto chargeModel = Constants::CHARGE_SHARING_MODEL;
    const bool useLinearModel = (chargeModel == Constants::ChargeSharingModel::Linear);
    const G4double beta = useLinearModel ? fDetector->GetLinearChargeModelBeta(pixelSpacing) : 0.0;

    const G4double hitX = hitPos.x();
    const G4double hitY = hitPos.y();
    const G4double nearestX = fResult.nearestPixelCenter.x();
    const G4double nearestY = fResult.nearestPixelCenter.y();
    const G4double baseDx = hitX - nearestX;
    const G4double baseDy = hitY - nearestY;

    G4double totalWeight = 0.0;

    const auto& gainSigmas = fDetector->GetPixelGainSigmas();
    const std::size_t gainCount = gainSigmas.size();
    const bool hasGainNoise = gainCount > 0;
    const G4double sigmaNoise = Constants::NOISE_ELECTRON_COUNT * elementaryCharge;
    const bool hasAdditiveNoise = sigmaNoise > 0.0;
    const G4double totalChargeCoulomb = totalChargeElectrons * elementaryCharge;

    const G4ThreeVector detectorPos = fDetector->GetDetectorPos();
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

            const G4double safeDistance = std::max(distance, minSafeDistance);
            const G4double logValue = std::log(safeDistance * invD0Length);
            G4double weight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;
            if (useLinearModel) {
                const G4double attenuation = std::max(0.0, 1.0 - beta * distance * invMicrometer);
                weight = attenuation * alpha;
            }

            const G4double pixelCenterX = detectorPos.x() + firstPixelPos + i * pixelSpacing;
            const G4double pixelCenterY = detectorPos.y() + firstPixelPos + j * pixelSpacing;

            fFullGridWeights(i, j) = weight;
            fResult.full.distance(i, j) = distance;
            fResult.full.alpha(i, j) = alpha;
            fResult.full.pixelX(i, j) = pixelCenterX;
            fResult.full.pixelY(i, j) = pixelCenterY;
            totalWeight += weight;
        }
    }

    const G4double invTotalWeight = (totalWeight > 0.0) ? (1.0 / totalWeight) : 0.0;
    for (G4int i = 0; i < rows; ++i) {
        for (G4int j = 0; j < cols; ++j) {
            const auto idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) +
                             static_cast<std::size_t>(j);
            const G4double weight = fFullGridWeights(i, j);
            const G4double fraction = weight * invTotalWeight;
            const G4double baseCharge = fraction * totalChargeCoulomb;
            G4double noisyCharge = baseCharge;
            if (hasGainNoise) {
                if (idx < gainCount) {
                    const G4double sigmaGain = gainSigmas[idx];
                    if (sigmaGain > 0.0) {
                        noisyCharge *= G4RandGauss::shoot(1.0, sigmaGain);
                    }
                }
            }
            G4double finalCharge = noisyCharge;
            if (hasAdditiveNoise) {
                finalCharge += G4RandGauss::shoot(0.0, sigmaNoise);
            }
            finalCharge = std::max(0.0, finalCharge);

            fResult.full.Fi(i, j) = fraction;
            fResult.full.Qi(i, j) = baseCharge;
            fResult.full.Qn(i, j) = noisyCharge;
            fResult.full.Qf(i, j) = finalCharge;
        }
    }
}

