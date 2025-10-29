/**
 * @file ChargeSharingCalculator.cc
 */
#include "ChargeSharingCalculator.hh"

#include "DetectorConstruction.hh"
#include "Constants.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

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

void ChargeSharingCalculator::ResetForEvent()
{
    // Keep buffers sized; just reset contents to defaults
    std::fill(fWeightGrid.begin(), fWeightGrid.end(), 0.0);
    std::fill(fInBoundsGrid.begin(), fInBoundsGrid.end(), false);

    const std::size_t n = fResult.fractions.size();
    if (n == 0) return;

    std::fill(fResult.fractions.begin(), fResult.fractions.end(), Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    std::fill(fResult.charges.begin(),   fResult.charges.end(),   0.0);

    const auto nan = std::numeric_limits<G4double>::quiet_NaN();
    std::fill(fResult.distances.begin(), fResult.distances.end(), nan);
    std::fill(fResult.alphas.begin(),    fResult.alphas.end(),    nan);
    std::fill(fResult.pixelX.begin(),    fResult.pixelX.end(),    nan);
    std::fill(fResult.pixelY.begin(),    fResult.pixelY.end(),    nan);
    std::fill(fResult.pixelIds.begin(),  fResult.pixelIds.end(),  -1);
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

    fResult.nearestPixelCenter = CalcNearestPixel(hitPos);
    ComputeChargeFractions(hitPos,
                           energyDeposit,
                           ionizationEnergy,
                           amplificationFactor,
                           d0,
                           elementaryCharge);

    return fResult;
}

void ChargeSharingCalculator::ReserveBuffers()
{
    const G4int gridRadius = std::max(0, fNeighborhoodRadius);
    const G4int newGridDim = 2 * gridRadius + 1;
    const G4int totalCells = newGridDim * newGridDim;
    const std::size_t newSize = static_cast<std::size_t>(totalCells);

    if (newGridDim != fGridDim) {
        fGridDim = newGridDim;
    }

    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();

    if (fResult.fractions.size() != newSize) {
        fResult.fractions.assign(newSize, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    }
    if (fResult.charges.size() != newSize) {
        fResult.charges.assign(newSize, 0.0);
    }
    if (fResult.distances.size() != newSize) {
        fResult.distances.assign(newSize, nan);
    }
    if (fResult.alphas.size() != newSize) {
        fResult.alphas.assign(newSize, nan);
    }
    if (fResult.pixelX.size() != newSize) {
        fResult.pixelX.assign(newSize, nan);
    }
    if (fResult.pixelY.size() != newSize) {
        fResult.pixelY.assign(newSize, nan);
    }
    if (fResult.pixelIds.size() != newSize) {
        fResult.pixelIds.assign(newSize, -1);
    }
    if (fWeightGrid.size() != newSize) {
        fWeightGrid.assign(newSize, 0.0);
    }
    if (fInBoundsGrid.size() != newSize) {
        fInBoundsGrid.assign(newSize, false);
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

G4double ChargeSharingCalculator::CalcPixelAlphaSubtended(G4double hitX,
                                                          G4double hitY,
                                                          G4double pixelCenterX,
                                                          G4double pixelCenterY,
                                                          G4double pixelWidth,
                                                          G4double pixelHeight) const
{
    const G4double dx = hitX - pixelCenterX;
    const G4double dy = hitY - pixelCenterY;
    // hypot is robust and may vectorize better
    const G4double distance = std::hypot(dx, dy);
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
    static std::once_flag invalidD0WarningFlag;
    G4double d0Length = rawD0Length;
    if (!std::isfinite(d0Length) || d0Length <= 0.0) {
        std::call_once(invalidD0WarningFlag,
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

    const G4int gridDim = fGridDim;
    const G4int totalCells = gridDim * gridDim;

    constexpr G4double guardFactor = 1.0 + 1e-6;

    const auto chargeModel = Constants::CHARGE_SHARING_MODEL;
    const bool useLinearModel = (chargeModel == Constants::ChargeSharingModel::Linear);
    const G4double beta = useLinearModel ? fDetector->GetLinearChargeModelBeta(pixelSpacing) : 0.0;

    G4double totalWeight = 0.0;
    const G4double nearestX = fResult.nearestPixelCenter.x();
    const G4double nearestY = fResult.nearestPixelCenter.y();

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

        const G4double dx = hitX - pixelCenterX;
        const G4double dy = hitY - pixelCenterY;
        const G4double distance = std::hypot(dx, dy);
        const G4double alpha = CalcPixelAlphaSubtended(hitX, hitY, pixelCenterX, pixelCenterY, pixelSize, pixelSize);

        const G4double safeDistance = std::max(distance, d0Length * guardFactor);
        const G4double logValue = std::log(safeDistance / d0Length);
        const G4double logWeight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;

        G4double weight = logWeight;
        if (useLinearModel) {
            const G4double attenuation = std::max(0.0, 1.0 - beta * distance / micrometer);
            weight = attenuation * alpha;
        }

        fInBoundsGrid[idx] = true;
        fWeightGrid[idx] = weight;
        if (fEmitDistanceAlpha) {
            fResult.distances[idx] = distance;
            fResult.alphas[idx] = alpha;
        }
        fResult.pixelX[idx] = pixelCenterX;
        fResult.pixelY[idx] = pixelCenterY;
        fResult.pixelIds[idx] = globalId;
        totalWeight += weight;
    }

    for (G4int idx = 0; idx < totalCells; ++idx) {
        if (!fInBoundsGrid[idx]) {
            // Keep defaults from ResetForEvent
            continue;
        }

        const G4double fraction = (totalWeight > 0.0) ? (fWeightGrid[idx] / totalWeight) : 0.0;
        const G4double chargeCoulombs = fraction * totalCharge * elementaryCharge;
        fResult.fractions[idx] = fraction;
        fResult.charges[idx] = chargeCoulombs;
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
