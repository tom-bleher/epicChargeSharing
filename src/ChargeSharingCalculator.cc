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
    fResult.fractions.clear();
    fResult.charges.clear();
    fResult.distances.clear();
    fResult.alphas.clear();
    fResult.pixelX.clear();
    fResult.pixelY.clear();
    fResult.pixelIds.clear();
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
    ResetForEvent();

    fResult.nearestPixelCenter = CalcNearestPixel(hitPos);
    ComputeChargeFractions(hitPos,
                           energyDeposit,
                           ionizationEnergy,
                           amplificationFactor,
                           d0,
                           elementaryCharge);
    ComputeNeighborhoodGeometry();

    return fResult;
}

void ChargeSharingCalculator::ReserveBuffers()
{
    const G4int gridRadius = std::max(0, fNeighborhoodRadius);
    const G4int gridDim = 2 * gridRadius + 1;
    const G4int totalCells = gridDim * gridDim;

    fResult.fractions.reserve(totalCells);
    fResult.charges.reserve(totalCells);
    fResult.distances.reserve(totalCells);
    fResult.alphas.reserve(totalCells);
    fResult.pixelX.reserve(totalCells);
    fResult.pixelY.reserve(totalCells);
   fResult.pixelIds.reserve(totalCells);
    fWeightGrid.resize(totalCells);
    fInBoundsGrid.resize(totalCells);
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
    const G4double distance = std::sqrt(dx * dx + dy * dy);
    const G4double l = (pixelWidth + pixelHeight) / 2.0;

    const G4double numerator = (l / 2.0) * std::sqrt(2.0);
    const G4double denominator = numerator + distance;
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
    const G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
    const G4double detSize = fDetector->GetDetSize();
    const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();

    const G4double firstPixelPos = -detSize / 2 + pixelCornerOffset + pixelSize / 2;
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

    const G4int gridRadius = std::max(0, fNeighborhoodRadius);
    const G4int gridDim = 2 * gridRadius + 1;
    const G4int totalCells = gridDim * gridDim;

    constexpr G4double guardFactor = 1.0 + 1e-6;

    std::fill(fWeightGrid.begin(), fWeightGrid.end(), 0.0);
    std::fill(fInBoundsGrid.begin(), fInBoundsGrid.end(), false);

    const auto chargeModel = Constants::CHARGE_SHARING_MODEL;

    G4double totalWeight = 0.0;
    for (G4int di = -gridRadius; di <= gridRadius; ++di) {
        for (G4int dj = -gridRadius; dj <= gridRadius; ++dj) {
            const G4int idx = (di + gridRadius) * gridDim + (dj + gridRadius);
            const G4int gridPixelI = fResult.pixelIndexI + di;
            const G4int gridPixelJ = fResult.pixelIndexJ + dj;

            if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || gridPixelJ < 0 ||
                gridPixelJ >= numBlocksPerSide) {
                fResult.distances.push_back(std::numeric_limits<G4double>::quiet_NaN());
                fResult.alphas.push_back(std::numeric_limits<G4double>::quiet_NaN());
                continue;
            }

            const G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
            const G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;

            const G4double dx = hitX - pixelCenterX;
            const G4double dy = hitY - pixelCenterY;
            const G4double distance = std::sqrt(dx * dx + dy * dy);
            const G4double alpha =
                CalcPixelAlphaSubtended(hitX, hitY, pixelCenterX, pixelCenterY, pixelSize, pixelSize);

            const G4double safeDistance = std::max(distance, d0Length * guardFactor);
            const G4double logValue = std::log(safeDistance / d0Length);
            const G4double logWeight = (logValue > 0.0 && std::isfinite(logValue))
                                           ? (alpha / logValue)
                                           : 0.0;

            G4double linearWeight = 0.0;
            if (chargeModel == Constants::ChargeSharingModel::Linear) {
                const G4double pitch = pixelSpacing;
                const G4double beta = fDetector->GetLinearChargeModelBeta(pitch);
                const G4double attenuation = std::max(0.0, 1.0 - beta * distance / micrometer);
                linearWeight = attenuation * alpha;
            }

            const G4double weight = (chargeModel == Constants::ChargeSharingModel::Log)
                                        ? logWeight
                                        : linearWeight;

            fInBoundsGrid[idx] = true;
            fWeightGrid[idx] = weight;
            fResult.distances.push_back(distance);
            fResult.alphas.push_back(alpha);
            totalWeight += weight;
        }
    }

    for (G4int idx = 0; idx < totalCells; ++idx) {
        if (!fInBoundsGrid[idx]) {
            fResult.fractions.push_back(Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
            fResult.charges.push_back(0.0);
            continue;
        }

        const G4double fraction = (totalWeight > 0.0) ? (fWeightGrid[idx] / totalWeight) : 0.0;
        const G4double chargeCoulombs = fraction * totalCharge * elementaryCharge;
        fResult.fractions.push_back(fraction);
        fResult.charges.push_back(chargeCoulombs);
    }
}

void ChargeSharingCalculator::ComputeNeighborhoodGeometry()
{
    const G4double pixelSize = fDetector->GetPixelSize();
    const G4double pixelSpacing = fDetector->GetPixelSpacing();
    const G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
    const G4double detSize = fDetector->GetDetSize();
    const G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();

    const G4double firstPixelPos = -detSize / 2 + pixelCornerOffset + pixelSize / 2;
    const G4int gridRadius = std::max(0, fNeighborhoodRadius);
    // gridDim not needed here; kept internal for indexing where used

    for (G4int di = -gridRadius; di <= gridRadius; ++di) {
        for (G4int dj = -gridRadius; dj <= gridRadius; ++dj) {
            const G4int gridPixelI = fResult.pixelIndexI + di;
            const G4int gridPixelJ = fResult.pixelIndexJ + dj;

            if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || gridPixelJ < 0 ||
                gridPixelJ >= numBlocksPerSide) {
                fResult.pixelX.push_back(std::numeric_limits<G4double>::quiet_NaN());
                fResult.pixelY.push_back(std::numeric_limits<G4double>::quiet_NaN());
                fResult.pixelIds.push_back(-1);
                continue;
            }

            const G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
            const G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
            const G4int globalId = gridPixelI * numBlocksPerSide + gridPixelJ;

            fResult.pixelX.push_back(pixelCenterX);
            fResult.pixelY.push_back(pixelCenterY);
            fResult.pixelIds.push_back(globalId);
        }
    }
}
