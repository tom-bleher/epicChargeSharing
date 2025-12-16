#include "ChargeSharingReconstructor.h"

#include <DDSegmentation/BitFieldCoder.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace epic::chargesharing {

namespace {

constexpr double kOutOfBoundsFraction = -999.0;
constexpr double kGuardFactor = 1.0 + 1e-6;
constexpr double kMillimeterPerMicron = 1.0e-3;

// Paper terminology: LogA and LinA are the signal sharing models.
enum class ChargeSharingModel { LogA, LinA };
constexpr ChargeSharingModel kChargeSharingModel = ChargeSharingModel::LinA;

constexpr double kLinearBetaNarrow = 0.003;  // 1/um
constexpr double kLinearBetaWide = 0.001;    // 1/um
constexpr double kLinearMinPitchUM = 100.0;
constexpr double kLinearBoundaryPitchUM = 200.0;
constexpr double kLinearMaxPitchUM = 500.0;

double nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

void ChargeSharingReconstructor::configure(const ChargeSharingConfig& cfg,
                                           const dd4hep::DDSegmentation::CartesianGridXY* segmentation,
                                           const dd4hep::DDSegmentation::BitFieldCoder* decoder) {
  m_cfg = cfg;
  m_segmentation = segmentation;
  m_decoder = decoder;
  m_haveSegmentation = cfg.segmentation.valid && (segmentation != nullptr) && (decoder != nullptr);

  if (m_cfg.neighborhoodRadius < 0) {
    m_cfg.neighborhoodRadius = 0;
  }

  if (m_cfg.pixelSpacingYMM <= 0.0) {
    m_cfg.pixelSpacingYMM = m_cfg.pixelSpacingMM;
  }
  if (m_cfg.pixelSizeYMM <= 0.0) {
    m_cfg.pixelSizeYMM = m_cfg.pixelSizeMM;
  }

  if (m_haveSegmentation) {
    m_boundsX.minIndex = m_cfg.segmentation.minIndexX;
    m_boundsX.maxIndex = m_cfg.segmentation.maxIndexX;
    m_boundsY.minIndex = m_cfg.segmentation.minIndexY;
    m_boundsY.maxIndex = m_cfg.segmentation.maxIndexY;

    if (!m_boundsX.hasBounds() && m_cfg.segmentation.numCellsX > 0) {
      m_boundsX.minIndex = m_cfg.segmentation.minIndexX;
      m_boundsX.maxIndex = m_cfg.segmentation.minIndexX + m_cfg.segmentation.numCellsX - 1;
    }
    if (!m_boundsY.hasBounds() && m_cfg.segmentation.numCellsY > 0) {
      m_boundsY.minIndex = m_cfg.segmentation.minIndexY;
      m_boundsY.maxIndex = m_cfg.segmentation.minIndexY + m_cfg.segmentation.numCellsY - 1;
    }

    if (m_cfg.pixelsPerSide <= 0 && m_cfg.segmentation.numCellsX > 0 &&
        m_cfg.segmentation.numCellsX == m_cfg.segmentation.numCellsY) {
      m_cfg.pixelsPerSide = m_cfg.segmentation.numCellsX;
    }
  } else {
    const int minIndex = 0;
    const int maxIndex = std::max(0, m_cfg.pixelsPerSide - 1);
    m_boundsX = IndexBounds{minIndex, maxIndex};
    m_boundsY = IndexBounds{minIndex, maxIndex};

    if (m_cfg.pixelsPerSide <= 0 && m_cfg.pixelSpacingMM > 0.0) {
      const double activeWidth = m_cfg.detectorSizeMM - 2.0 * m_cfg.pixelCornerOffsetMM;
      const double approxPixels = activeWidth / m_cfg.pixelSpacingMM;
      m_cfg.pixelsPerSide = static_cast<int>(std::round(approxPixels));
      if (m_cfg.pixelsPerSide < 1) {
        m_cfg.pixelsPerSide = 1;
      }
      m_boundsX.maxIndex = m_cfg.pixelsPerSide - 1;
      m_boundsY.maxIndex = m_cfg.pixelsPerSide - 1;
    }
  }

  ensureGrid();
}

ChargeSharingReconstructor::Result ChargeSharingReconstructor::process(const Input& input) {
  ensureGrid();
  resetBuffers();

  Result result{};

  PixelLocation nearest{};
  if (m_haveSegmentation) {
    nearest = findPixelFromSegmentation(input.cellID);
  } else if (input.pixelHintMM.has_value()) {
    nearest = findNearestPixelFallback(*input.pixelHintMM);
  } else {
    nearest = findNearestPixelFallback(input.hitPositionMM);
  }

  result.nearestPixelCenterMM = nearest.center;
  result.pixelIndexI = nearest.indexI;
  result.pixelIndexJ = nearest.indexJ;

  const double hitX = input.hitPositionMM[0];
  const double hitY = input.hitPositionMM[1];

  const double edepEV = input.energyDepositGeV * 1.0e9;  // GeV -> eV
  const double numElectrons = (m_cfg.ionizationEnergyEV > 0.0)
                                  ? (edepEV / m_cfg.ionizationEnergyEV)
                                  : 0.0;
  const double totalChargeElectrons = numElectrons * m_cfg.amplificationFactor;

  const double d0LengthMM = std::max(std::max(m_cfg.d0Micron * kMillimeterPerMicron, 1e-9), 0.0);
  const double minSafeDistance = d0LengthMM * kGuardFactor;
  const double invD0 = (d0LengthMM > 0.0) ? 1.0 / d0LengthMM : 0.0;

  const double pixelSpacingX = (m_haveSegmentation && m_cfg.segmentation.gridSizeXMM > 0.0)
                                   ? m_cfg.segmentation.gridSizeXMM
                                   : m_cfg.pixelSpacingMM;
  const double pixelSpacingY = (m_haveSegmentation && m_cfg.segmentation.gridSizeYMM > 0.0)
                                   ? m_cfg.segmentation.gridSizeYMM
                                   : m_cfg.pixelSpacingYMM;
  const double pixelWidth = (m_haveSegmentation && m_cfg.segmentation.cellSizeXMM > 0.0)
                                ? m_cfg.segmentation.cellSizeXMM
                                : m_cfg.pixelSizeMM;
  const double pixelHeight = (m_haveSegmentation && m_cfg.segmentation.cellSizeYMM > 0.0)
                                 ? m_cfg.segmentation.cellSizeYMM
                                 : m_cfg.pixelSizeYMM;
  const double pitchForBeta = std::max(0.5 * (pixelSpacingX + pixelSpacingY), 1.0e-6);
  const double beta = linearModelBeta(pitchForBeta);
  const bool useLinearModel = (kChargeSharingModel == ChargeSharingModel::LinA);

  double totalWeight = 0.0;

  for (const auto& offset : m_offsets) {
    const int di = offset.di;
    const int dj = offset.dj;
    const std::size_t idx = offset.idx;

    const int gridPixelI = nearest.indexI + di;
    const int gridPixelJ = nearest.indexJ + dj;

    if (!isPixelIndexInBounds(gridPixelI, gridPixelJ)) {
      continue;
    }

    const PixelLocation neighborLoc = pixelLocationFromIndices(gridPixelI, gridPixelJ);
    const double pixelCenterX = neighborLoc.center[0];
    const double pixelCenterY = neighborLoc.center[1];

    const double dxToCenter = hitX - pixelCenterX;
    const double dyToCenter = hitY - pixelCenterY;
    const double distanceToCenter = calcDistanceToCenter(dxToCenter, dyToCenter);
    const double alpha = calcPadViewAngle(distanceToCenter,
                                          (pixelWidth > 0.0) ? pixelWidth : pixelSpacingX,
                                          (pixelHeight > 0.0) ? pixelHeight : pixelSpacingY);

    const double safeDistance = std::max(distanceToCenter, minSafeDistance);
    const double logValue = (invD0 > 0.0) ? std::log(safeDistance * invD0) : 0.0;
    double weight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;

    if (useLinearModel) {
      const double distanceUM = distanceToCenter / kMillimeterPerMicron;
      const double attenuation = std::max(0.0, 1.0 - beta * distanceUM);
      weight = attenuation * alpha;
    }

    m_inBounds[idx] = true;
    m_weightGrid[idx] = weight;
    m_pixelX[idx] = pixelCenterX;
    m_pixelY[idx] = pixelCenterY;
    m_pixelIds[idx] = linearizedPixelId(gridPixelI, gridPixelJ);

    if (!m_distances.empty()) {
      m_distances[idx] = distanceToCenter;
      m_alphas[idx] = alpha;
    }

    totalWeight += weight;
  }

  const std::size_t totalCells = static_cast<std::size_t>(m_gridDim * m_gridDim);
  const double totalChargeCoulombs = totalChargeElectrons * m_cfg.elementaryChargeC;

  for (std::size_t idx = 0; idx < totalCells; ++idx) {
    if (!m_inBounds[idx]) {
      continue;
    }
    const double fraction = (totalWeight > 0.0) ? (m_weightGrid[idx] / totalWeight) : 0.0;
    m_fractions[idx] = fraction;
    m_charges[idx] = fraction * totalChargeCoulombs;
  }

  double weightedX = 0.0;
  double weightedY = 0.0;
  double weightSum = 0.0;
  double totalCollectedCharge = 0.0;

  result.neighbors.reserve(m_offsets.size());
  for (const auto& offset : m_offsets) {
    const std::size_t idx = offset.idx;
    if (!m_inBounds[idx]) {
      continue;
    }

    NeighborData neighbor{};
    neighbor.fraction = m_fractions[idx];
    neighbor.chargeC = m_charges[idx];
    neighbor.pixelXMM = m_pixelX[idx];
    neighbor.pixelYMM = m_pixelY[idx];
    neighbor.pixelId = m_pixelIds[idx];
    neighbor.di = offset.di;
    neighbor.dj = offset.dj;
    neighbor.distanceMM = m_distances.empty() ? 0.0 : m_distances[idx];
    neighbor.alphaRad = m_alphas.empty() ? 0.0 : m_alphas[idx];

    result.neighbors.push_back(neighbor);

    if (neighbor.fraction > 0.0) {
      weightSum += neighbor.fraction;
      weightedX += neighbor.fraction * neighbor.pixelXMM;
      weightedY += neighbor.fraction * neighbor.pixelYMM;
    }

    totalCollectedCharge += neighbor.chargeC;
  }

  result.totalCollectedChargeC = totalCollectedCharge;

  if (weightSum > 0.0) {
    result.reconstructedPositionMM = {weightedX / weightSum, weightedY / weightSum,
                                      nearest.center[2]};
  } else {
    result.reconstructedPositionMM = nearest.center;
  }

  return result;
}

void ChargeSharingReconstructor::ensureGrid() {
  const int desiredDim = 2 * std::max(0, m_cfg.neighborhoodRadius) + 1;
  const bool diagnosticsEnabled = m_cfg.emitNeighborDiagnostics;
  const bool diagStateMismatch = diagnosticsEnabled ? m_distances.empty() : !m_distances.empty();

  if (desiredDim == m_gridDim && !diagStateMismatch) {
    return;
  }

  m_gridDim = desiredDim;
  rebuildOffsets();

  const std::size_t totalCells = static_cast<std::size_t>(m_gridDim * m_gridDim);
  m_weightGrid.assign(totalCells, 0.0);
  m_inBounds.assign(totalCells, false);
  m_fractions.assign(totalCells, kOutOfBoundsFraction);
  m_charges.assign(totalCells, 0.0);
  m_pixelX.assign(totalCells, nan());
  m_pixelY.assign(totalCells, nan());
  m_pixelIds.assign(totalCells, -1);

  if (diagnosticsEnabled) {
    m_distances.assign(totalCells, nan());
    m_alphas.assign(totalCells, nan());
  } else {
    m_distances.clear();
    m_alphas.clear();
  }
}

void ChargeSharingReconstructor::resetBuffers() {
  std::fill(m_weightGrid.begin(), m_weightGrid.end(), 0.0);
  std::fill(m_inBounds.begin(), m_inBounds.end(), false);
  std::fill(m_fractions.begin(), m_fractions.end(), kOutOfBoundsFraction);
  std::fill(m_charges.begin(), m_charges.end(), 0.0);
  std::fill(m_pixelX.begin(), m_pixelX.end(), nan());
  std::fill(m_pixelY.begin(), m_pixelY.end(), nan());
  std::fill(m_pixelIds.begin(), m_pixelIds.end(), -1);

  if (!m_distances.empty()) {
    std::fill(m_distances.begin(), m_distances.end(), nan());
  }
  if (!m_alphas.empty()) {
    std::fill(m_alphas.begin(), m_alphas.end(), nan());
  }
}

void ChargeSharingReconstructor::rebuildOffsets() {
  const int radius = std::max(0, m_cfg.neighborhoodRadius);
  const int dim = 2 * radius + 1;
  m_offsets.clear();
  m_offsets.reserve(static_cast<std::size_t>(dim * dim));

  for (int di = -radius; di <= radius; ++di) {
    for (int dj = -radius; dj <= radius; ++dj) {
      const std::size_t idx = static_cast<std::size_t>((di + radius) * dim + (dj + radius));
      m_offsets.push_back(Offset{di, dj, idx});
    }
  }
}

double ChargeSharingReconstructor::calcDistanceToCenter(double dxToCenterMM,
                                                         double dyToCenterMM) const {
  // Compute d_i: Euclidean distance from hit point to pixel center.
  return std::hypot(dxToCenterMM, dyToCenterMM);
}

double ChargeSharingReconstructor::calcPadViewAngle(double distanceToCenterMM,
                                                    double padWidthMM,
                                                    double padHeightMM) const {
  const double l = (padWidthMM + padHeightMM) / 2.0;
  const double numerator = (l / 2.0) * std::sqrt(2.0);
  const double denominator = numerator + distanceToCenterMM;
  if (distanceToCenterMM == 0.0) {
    return std::atan(1.0);
  }
  return std::atan(numerator / denominator);
}

double ChargeSharingReconstructor::linearModelBeta(double pitchMM) const {
  if (kChargeSharingModel != ChargeSharingModel::LinA) {
    return 0.0;
  }

  const double pitchUM = pitchMM / kMillimeterPerMicron;
  if (pitchUM >= kLinearMinPitchUM && pitchUM <= kLinearBoundaryPitchUM) {
    return kLinearBetaNarrow;
  }
  if (pitchUM > kLinearBoundaryPitchUM && pitchUM <= kLinearMaxPitchUM) {
    return kLinearBetaWide;
  }
  return kLinearBetaNarrow;
}

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::findNearestPixelFallback(
    const std::array<double, 3>& positionMM) const {
  const double firstPixelX = firstPixelCenterCoordinate(m_cfg.detectorSizeMM, m_cfg.pixelCornerOffsetMM,
                                                        m_cfg.pixelSizeMM);
  const double firstPixelY = firstPixelCenterCoordinate(m_cfg.detectorSizeMM, m_cfg.pixelCornerOffsetMM,
                                                        m_cfg.pixelSizeYMM);

  int i = static_cast<int>(std::llround((positionMM[0] - firstPixelX) / m_cfg.pixelSpacingMM));
  int j = static_cast<int>(std::llround((positionMM[1] - firstPixelY) / m_cfg.pixelSpacingYMM));

  const int defaultMin = 0;
  const int defaultMax = std::max(0, m_cfg.pixelsPerSide - 1);
  const int minI = m_boundsX.hasBounds() ? m_boundsX.minIndex : defaultMin;
  const int maxI = m_boundsX.hasBounds() ? m_boundsX.maxIndex : defaultMax;
  const int minJ = m_boundsY.hasBounds() ? m_boundsY.minIndex : defaultMin;
  const int maxJ = m_boundsY.hasBounds() ? m_boundsY.maxIndex : defaultMax;

  i = std::clamp(i, minI, maxI);
  j = std::clamp(j, minJ, maxJ);

  return pixelLocationFromIndices(i, j);
}

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::findPixelFromSegmentation(
    std::uint64_t cellID) const {
  if (!m_haveSegmentation || m_decoder == nullptr) {
    return pixelLocationFromIndices(0, 0);
  }

  const int indexI = static_cast<int>(m_decoder->get(cellID, m_cfg.segmentation.fieldNameX));
  const int indexJ = static_cast<int>(m_decoder->get(cellID, m_cfg.segmentation.fieldNameY));
  return pixelLocationFromIndices(indexI, indexJ);
}

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::pixelLocationFromIndices(
    int indexI, int indexJ) const {
  PixelLocation loc{};
  loc.indexI = indexI;
  loc.indexJ = indexJ;

  const double pixelCenterZ = m_cfg.detectorZCenterMM + m_cfg.detectorThicknessMM / 2.0 +
                              m_cfg.pixelThicknessMM / 2.0;

  if (m_haveSegmentation) {
    const double pitchX = (m_cfg.segmentation.gridSizeXMM > 0.0) ? m_cfg.segmentation.gridSizeXMM
                                                                : m_cfg.pixelSpacingMM;
    const double pitchY = (m_cfg.segmentation.gridSizeYMM > 0.0) ? m_cfg.segmentation.gridSizeYMM
                                                                : m_cfg.pixelSpacingYMM;
    const double offsetX = m_cfg.segmentation.offsetXMM;
    const double offsetY = m_cfg.segmentation.offsetYMM;
    loc.center = {offsetX + indexI * pitchX, offsetY + indexJ * pitchY, pixelCenterZ};
  } else {
    const double firstPixelX = firstPixelCenterCoordinate(m_cfg.detectorSizeMM, m_cfg.pixelCornerOffsetMM,
                                                          m_cfg.pixelSizeMM);
    const double firstPixelY = firstPixelCenterCoordinate(m_cfg.detectorSizeMM, m_cfg.pixelCornerOffsetMM,
                                                          m_cfg.pixelSizeYMM);
    loc.center = {firstPixelX + indexI * m_cfg.pixelSpacingMM,
                  firstPixelY + indexJ * m_cfg.pixelSpacingYMM,
                  pixelCenterZ};
  }

  return loc;
}

bool ChargeSharingReconstructor::isPixelIndexInBounds(int indexI, int indexJ) const {
  return m_boundsX.contains(indexI) && m_boundsY.contains(indexJ);
}

double ChargeSharingReconstructor::firstPixelCenterCoordinate(double detectorSize, double cornerOffset,
                                                             double pixelSize) const {
  return -detectorSize / 2.0 + cornerOffset + pixelSize / 2.0;
}

int ChargeSharingReconstructor::linearizedPixelId(int indexI, int indexJ) const {
  if (m_haveSegmentation && m_cfg.segmentation.numCellsX > 0 && m_cfg.segmentation.numCellsY > 0 &&
      m_cfg.segmentation.maxIndexX >= m_cfg.segmentation.minIndexX &&
      m_cfg.segmentation.maxIndexY >= m_cfg.segmentation.minIndexY) {
    const int offsetI = indexI - m_cfg.segmentation.minIndexX;
    const int offsetJ = indexJ - m_cfg.segmentation.minIndexY;
    return offsetI * m_cfg.segmentation.numCellsY + offsetJ;
  }

  const int numPerSide = std::max(1, m_cfg.pixelsPerSide);
  return indexI * numPerSide + indexJ;
}

}  // namespace epic::chargesharing
