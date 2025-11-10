/**
 * @file ChargeSharingCalculator.cc
 */
#include "ChargeSharingCalculator.hh"

#include "Constants.hh"
#include "DetectorConstruction.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"

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

void ChargeSharingCalculator::ResetForEvent()
{
    // Keep buffers sized; just reset contents to defaults
    fResult.cells.clear();
    fWeightScratch.clear();
    if (!fResult.fullFractions.empty()) {
        std::fill(fResult.fullFractions.begin(), fResult.fullFractions.end(), 0.0);
    }
    if (!fResult.fullPixelIds.empty()) {
        std::fill(fResult.fullPixelIds.begin(), fResult.fullPixelIds.end(), -1);
    }
    if (!fResult.fullPixelX.empty()) {
        std::fill(fResult.fullPixelX.begin(),
                  fResult.fullPixelX.end(),
                  std::numeric_limits<G4double>::quiet_NaN());
    }
    if (!fResult.fullPixelY.empty()) {
        std::fill(fResult.fullPixelY.begin(),
                  fResult.fullPixelY.end(),
                  std::numeric_limits<G4double>::quiet_NaN());
    }
    fResult.fullGridSide = 0;
    fResult.fullTotalCells = 0;
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

    fResult.gridRadius = fNeighborhoodRadius;
    fResult.gridSide = fGridDim;
    fResult.totalCells = static_cast<std::size_t>(fGridDim) * static_cast<std::size_t>(fGridDim);
    fResult.cells.clear();
    fWeightScratch.clear();
    fResult.nearestPixelCenter = CalcNearestPixel(hitPos);
    ComputeChargeFractions(hitPos,
                           energyDeposit,
                           ionizationEnergy,
                           amplificationFactor,
                           d0,
                           elementaryCharge);

    if (fComputeFullGridFractions) {
        ComputeFullGridFractions(hitPos,
                                 d0,
                                 fDetector->GetPixelSize(),
                                 fDetector->GetPixelSpacing(),
                                 fDetector->GetNumBlocksPerSide());
    } else {
        fResult.fullFractions.clear();
        fResult.fullPixelIds.clear();
        fResult.fullPixelX.clear();
        fResult.fullPixelY.clear();
        fResult.fullGridSide = 0;
        fResult.fullTotalCells = 0;
        fFullGridWeights.clear();
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
                                                     G4double energyDeposit,
                                                     G4double ionizationEnergy,
                                                     G4double amplificationFactor,
                                                     G4double d0,
                                                     G4double elementaryCharge)
{
    const G4double edepInEV = energyDeposit / eV;
    const G4double numElectrons = ionizationEnergy > 0.0 ? edepInEV / ionizationEnergy : 0.0;
    const G4double totalCharge = numElectrons * amplificationFactor;

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
        cell.charge = fraction * totalCharge * elementaryCharge;
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
        fResult.fullFractions.clear();
        fResult.fullPixelIds.clear();
        fResult.fullPixelX.clear();
        fResult.fullPixelY.clear();
        fFullGridWeights.clear();
        fResult.fullGridSide = 0;
        fResult.fullTotalCells = 0;
        return;
    }
    const G4int numBlocksPerSide = std::max(0, fDetector->GetNumBlocksPerSide());
    const std::size_t totalCells =
        static_cast<std::size_t>(numBlocksPerSide) * static_cast<std::size_t>(numBlocksPerSide);
    if (totalCells == 0U) {
        fResult.fullFractions.clear();
        fResult.fullPixelIds.clear();
        fResult.fullPixelX.clear();
        fResult.fullPixelY.clear();
        fFullGridWeights.clear();
        fResult.fullGridSide = 0;
        fResult.fullTotalCells = 0;
        return;
    }
    fResult.fullGridSide = numBlocksPerSide;
    fResult.fullTotalCells = totalCells;

    if (fResult.fullFractions.size() != totalCells) {
        fResult.fullFractions.assign(totalCells, 0.0);
    } else {
        std::fill(fResult.fullFractions.begin(), fResult.fullFractions.end(), 0.0);
    }
    if (fFullGridWeights.size() != totalCells) {
        fFullGridWeights.assign(totalCells, 0.0);
    } else {
        std::fill(fFullGridWeights.begin(), fFullGridWeights.end(), 0.0);
    }
    if (fResult.fullPixelIds.size() != totalCells) {
        fResult.fullPixelIds.assign(totalCells, -1);
    } else {
        std::fill(fResult.fullPixelIds.begin(), fResult.fullPixelIds.end(), -1);
    }
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    if (fResult.fullPixelX.size() != totalCells) {
        fResult.fullPixelX.assign(totalCells, nan);
    } else {
        std::fill(fResult.fullPixelX.begin(), fResult.fullPixelX.end(), nan);
    }
    if (fResult.fullPixelY.size() != totalCells) {
        fResult.fullPixelY.assign(totalCells, nan);
    } else {
        std::fill(fResult.fullPixelY.begin(), fResult.fullPixelY.end(), nan);
    }
}

void ChargeSharingCalculator::ComputeFullGridFractions(const G4ThreeVector& hitPos,
                                                       G4double d0,
                                                       G4double pixelSize,
                                                       G4double pixelSpacing,
                                                       G4int numBlocksPerSide)
{
    if (!fDetector) {
        fResult.fullFractions.clear();
        fFullGridWeights.clear();
        return;
    }

    const G4int gridSide = std::max(0, numBlocksPerSide);
    if (gridSide <= 0) {
        fResult.fullFractions.clear();
        fFullGridWeights.clear();
        return;
    }

    EnsureFullGridBuffer();
    if (fResult.fullFractions.empty()) {
        return;
    }

    const std::size_t totalCells =
        static_cast<std::size_t>(gridSide) * static_cast<std::size_t>(gridSide);
    fResult.fullGridSide = gridSide;
    fResult.fullTotalCells = totalCells;

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

    const G4ThreeVector detectorPos = fDetector->GetDetectorPos();
    const G4double detSize = fDetector->GetDetSize();
    const G4double cornerOffset = fDetector->GetPixelCornerOffset();
    const G4double firstPixelPos = -detSize / 2.0 + cornerOffset + pixelSize / 2.0;

    G4double totalWeight = 0.0;

    for (G4int i = 0; i < gridSide; ++i) {
        const G4int di = i - fResult.pixelIndexI;
        for (G4int j = 0; j < gridSide; ++j) {
            const G4int dj = j - fResult.pixelIndexJ;
            const std::size_t idx =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(gridSide) +
                static_cast<std::size_t>(j);

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
            const G4int globalId = i * gridSide + j;

            fFullGridWeights[idx] = weight;
            if (idx < fResult.fullPixelIds.size()) {
                fResult.fullPixelIds[idx] = globalId;
            }
            if (idx < fResult.fullPixelX.size()) {
                fResult.fullPixelX[idx] = pixelCenterX;
            }
            if (idx < fResult.fullPixelY.size()) {
                fResult.fullPixelY[idx] = pixelCenterY;
            }
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0) {
        const G4double invTotalWeight = 1.0 / totalWeight;
        for (std::size_t idx = 0; idx < totalCells; ++idx) {
            fResult.fullFractions[idx] = fFullGridWeights[idx] * invTotalWeight;
        }
    } else {
        std::fill(fResult.fullFractions.begin(), fResult.fullFractions.end(), 0.0);
    }
}
