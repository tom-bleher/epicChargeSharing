#include "ChargeSharingReconstructor.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace epic::chargesharing {

void ChargeSharingReconstructor::configure(const ChargeSharingConfig& cfg) {
  m_cfg = cfg;

  if (m_cfg.neighborhoodRadius < 0) {
    m_cfg.neighborhoodRadius = 0;
  }

  // Set effective Y values if not specified
  if (m_cfg.pixelSizeYMM <= 0.0) {
    m_cfg.pixelSizeYMM = m_cfg.pixelSizeMM;
  }
  if (m_cfg.pixelSpacingYMM <= 0.0) {
    m_cfg.pixelSpacingYMM = m_cfg.pixelSpacingMM;
  }

  // Configure noise model
  core::NoiseConfig noiseConfig;
  noiseConfig.enabled = m_cfg.noiseEnabled;
  noiseConfig.gainSigmaMin = m_cfg.noiseGainSigmaMin;
  noiseConfig.gainSigmaMax = m_cfg.noiseGainSigmaMax;
  noiseConfig.electronNoiseCount = m_cfg.noiseElectronCount;
  noiseConfig.elementaryCharge = m_cfg.elementaryChargeC;
  m_noiseModel.setConfig(noiseConfig);
  if (m_cfg.noiseSeed != 0) {
    m_noiseModel.setSeed(m_cfg.noiseSeed);
  }

  // Set up bounds from segmentation config (populated by factory from DD4hep)
  if (m_cfg.segmentation.valid) {
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
    // No DD4hep segmentation: compute bounds from detector size and pitch
    if (m_cfg.pixelsPerSide <= 0 && m_cfg.pixelSpacingMM > 0.0) {
      const double approxPixels = m_cfg.detectorSizeMM / m_cfg.pixelSpacingMM;
      m_cfg.pixelsPerSide = static_cast<int>(std::round(approxPixels));
      if (m_cfg.pixelsPerSide < 1) {
        m_cfg.pixelsPerSide = 1;
      }
    }

    // DD4hep-style centered grid: compute symmetric bounds around 0
    const int halfGrid = m_cfg.pixelsPerSide / 2;
    const int minIndex = -halfGrid;
    const int maxIndex = m_cfg.pixelsPerSide - halfGrid - 1;
    m_boundsX = IndexBounds{minIndex, maxIndex};
    m_boundsY = IndexBounds{minIndex, maxIndex};
  }
}

ChargeSharingReconstructor::Result ChargeSharingReconstructor::process(const Input& input) {
  Result result{};

  // Find the center pixel
  PixelLocation nearest{};
  if (input.pixelIndexHint.has_value()) {
    // Pre-decoded grid indices from factory (fastest path)
    const auto& [idxI, idxJ] = *input.pixelIndexHint;
    nearest = pixelLocationFromIndices(idxI, idxJ);
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
  const double hitZ = input.hitPositionMM[2];

  // Configure neighborhood calculation
  core::NeighborhoodConfig neighborCfg;
  neighborCfg.signalModel = m_cfg.signalModel;
  neighborCfg.activeMode = m_cfg.activePixelMode;
  neighborCfg.radius = m_cfg.neighborhoodRadius;
  neighborCfg.pixelSizeMM = m_cfg.pixelSizeMM;
  neighborCfg.pixelSizeYMM = m_cfg.pixelSizeYMM;
  neighborCfg.pixelSpacingMM = m_cfg.pixelSpacingMM;
  neighborCfg.pixelSpacingYMM = m_cfg.pixelSpacingYMM;
  neighborCfg.d0Micron = m_cfg.d0Micron;
  neighborCfg.betaPerMicron = m_cfg.linearBetaPerMicron;
  neighborCfg.numPixelsX = m_cfg.pixelsPerSide;
  neighborCfg.numPixelsY = m_cfg.pixelsPerSide;

  // Calculate charge fractions using core algorithm
  core::NeighborhoodResult neighborhood = core::calculateNeighborhood(
      hitX, hitY,
      nearest.indexI, nearest.indexJ,
      nearest.center[0], nearest.center[1],
      neighborCfg);

  // Convert energy deposit to charge
  const double edepEV = input.energyDepositGeV * 1.0e9;  // GeV -> eV
  const double numElectrons = (m_cfg.ionizationEnergyEV > 0.0)
                                  ? (edepEV / m_cfg.ionizationEnergyEV)
                                  : 0.0;
  const double totalChargeElectrons = numElectrons * m_cfg.amplificationFactor;
  const double totalChargeCoulombs = totalChargeElectrons * m_cfg.elementaryChargeC;

  // Compute charges in the neighborhood and apply noise if enabled
  // This is used for fitting - the charge field will be used instead of fraction
  for (auto& pixel : neighborhood.pixels) {
    if (pixel.inBounds) {
      // Main charge using configured active mode denominator
      double chargeC = pixel.fraction * totalChargeCoulombs;
      if (m_cfg.noiseEnabled) {
        chargeC = m_noiseModel.applyNoise(chargeC);
      }
      pixel.charge = chargeC;

      // Mode-specific charges for diagnostics (using different denominators)
      // Note: noise could optionally be applied to these too, but we keep them
      // as "clean" values for comparison/diagnostics
      pixel.chargeRow = pixel.fractionRow * totalChargeCoulombs;
      pixel.chargeCol = pixel.fractionCol * totalChargeCoulombs;
      pixel.chargeBlock = pixel.fractionBlock * totalChargeCoulombs;
    }
  }

  // Populate neighbor data for output (charges already computed above with noise)
  result.neighbors.reserve(neighborhood.pixels.size());
  for (const auto& pixel : neighborhood.pixels) {
    if (!pixel.inBounds) continue;

    NeighborData neighbor{};
    neighbor.fraction = pixel.fraction;
    neighbor.chargeC = pixel.charge;  // Use already-noisy charge from neighborhood
    neighbor.distanceMM = pixel.distance;
    neighbor.alphaRad = pixel.alpha;
    neighbor.pixelXMM = pixel.centerX;
    neighbor.pixelYMM = pixel.centerY;
    neighbor.pixelId = pixel.globalIndex;
    neighbor.di = pixel.di;
    neighbor.dj = pixel.dj;

    // Mode-specific fractions and charges for diagnostics
    neighbor.fractionRow = pixel.fractionRow;
    neighbor.fractionCol = pixel.fractionCol;
    neighbor.fractionBlock = pixel.fractionBlock;
    neighbor.chargeRowC = pixel.chargeRow;
    neighbor.chargeColC = pixel.chargeCol;
    neighbor.chargeBlockC = pixel.chargeBlock;

    result.neighbors.push_back(neighbor);
  }

  result.totalCollectedChargeC = totalChargeCoulombs;

  // Store diagnostic metadata
  result.truthPositionMM = input.hitPositionMM;
  result.inputEnergyDepositGeV = input.energyDepositGeV;
  result.inputCellID = input.cellID;
  result.neighborhoodRadius = m_cfg.neighborhoodRadius;
  result.neighborhoodGridSize = (2 * m_cfg.neighborhoodRadius + 1) * (2 * m_cfg.neighborhoodRadius + 1);

  // Compute summary statistics
  result.maxNeighborChargeC = 0.0;
  result.numActiveNeighbors = 0;
  for (const auto& n : result.neighbors) {
    if (n.chargeC > 0.0) {
      result.numActiveNeighbors++;
      if (n.chargeC > result.maxNeighborChargeC) {
        result.maxNeighborChargeC = n.chargeC;
      }
    }
  }

  // Reconstruct position using configured method
  const double centerZ = nearest.center[2];
  result.reconstructedPositionMM = reconstructPosition(
      neighborhood, centerZ,
      result.fitRowX, result.fitColY, result.fit2D);

  return result;
}

std::array<double, 3> ChargeSharingReconstructor::reconstructPosition(
    const core::NeighborhoodResult& neighborhood,
    double centerZ,
    fit::GaussFit1DResult& fitRowX,
    fit::GaussFit1DResult& fitColY,
    fit::GaussFit2DResult& fit2D) const {

  switch (m_cfg.reconMethod) {
    case core::ReconMethod::Gaussian1D:
      return reconstructGaussian1D(neighborhood, centerZ, fitRowX, fitColY);

    case core::ReconMethod::Gaussian2D:
      return reconstructGaussian2D(neighborhood, centerZ, fit2D);

    case core::ReconMethod::Centroid:
    default:
      return reconstructCentroid(neighborhood, centerZ);
  }
}

std::array<double, 3> ChargeSharingReconstructor::reconstructCentroid(
    const core::NeighborhoodResult& neighborhood,
    double centerZ) const {

  double weightedX = 0.0;
  double weightedY = 0.0;
  double weightSum = 0.0;

  // Use charge values (which include noise if enabled) for weighting
  for (const auto& pixel : neighborhood.pixels) {
    if (!pixel.inBounds || pixel.charge <= 0.0) continue;
    weightSum += pixel.charge;
    weightedX += pixel.charge * pixel.centerX;
    weightedY += pixel.charge * pixel.centerY;
  }

  if (weightSum > 0.0) {
    return {weightedX / weightSum, weightedY / weightSum, centerZ};
  }

  // Fallback to center pixel
  return {neighborhood.centerPixelX, neighborhood.centerPixelY, centerZ};
}

std::array<double, 3> ChargeSharingReconstructor::reconstructGaussian1D(
    const core::NeighborhoodResult& neighborhood,
    double centerZ,
    fit::GaussFit1DResult& fitRowX,
    fit::GaussFit1DResult& fitColY) const {

  // Get center row and column slices
  auto rowSlice = neighborhood.getCenterRow();
  auto colSlice = neighborhood.getCenterCol();

  // Build position and charge vectors for row fit (X reconstruction)
  // Use charge values (which include noise if enabled) for fitting
  std::vector<double> rowPositions, rowCharges;
  rowPositions.reserve(rowSlice.size());
  rowCharges.reserve(rowSlice.size());

  double maxRowCharge = 0.0;
  for (const auto* pixel : rowSlice) {
    if (pixel->charge > 0.0) {
      rowPositions.push_back(pixel->centerX);
      rowCharges.push_back(pixel->charge);
      maxRowCharge = std::max(maxRowCharge, pixel->charge);
    }
  }

  // Build position and charge vectors for column fit (Y reconstruction)
  std::vector<double> colPositions, colCharges;
  colPositions.reserve(colSlice.size());
  colCharges.reserve(colSlice.size());

  double maxColCharge = 0.0;
  for (const auto* pixel : colSlice) {
    if (pixel->charge > 0.0) {
      colPositions.push_back(pixel->centerY);
      colCharges.push_back(pixel->charge);
      maxColCharge = std::max(maxColCharge, pixel->charge);
    }
  }

  // Configure fits
  const double muRangeX = m_cfg.muRangeMM();
  const double muRangeY = m_cfg.muRangeMM();

  // Set up distance-weighted error config if enabled
  fit::DistanceWeightedErrorConfig distErrCfg;
  distErrCfg.enabled = m_cfg.fitUseDistanceWeightedErrors;
  distErrCfg.scalePixels = m_cfg.fitDistanceScalePixels;
  distErrCfg.exponent = m_cfg.fitDistanceExponent;
  distErrCfg.floorPercent = m_cfg.fitDistanceFloorPercent;
  distErrCfg.capPercent = m_cfg.fitDistanceCapPercent;
  distErrCfg.powerInverse = m_cfg.fitDistancePowerInverse;
  distErrCfg.pixelSpacing = m_cfg.pixelSpacingMM;
  distErrCfg.truthCenterX = neighborhood.centerPixelX;  // Use center pixel as estimate
  distErrCfg.truthCenterY = neighborhood.centerPixelY;
  distErrCfg.preferTruthCenter = m_cfg.fitDistancePreferTruthCenter;

  fit::GaussFit1DConfig rowConfig;
  rowConfig.muLo = neighborhood.centerPixelX - muRangeX;
  rowConfig.muHi = neighborhood.centerPixelX + muRangeX;
  rowConfig.sigmaLo = m_cfg.sigmaLoBound();
  rowConfig.sigmaHi = m_cfg.sigmaHiBound();
  rowConfig.qMax = maxRowCharge;
  rowConfig.pixelSpacing = m_cfg.pixelSpacingMM;
  rowConfig.errorPercent = m_cfg.fitErrorPercentOfMax;
  rowConfig.distanceErrorConfig = distErrCfg;
  rowConfig.centerPosition = neighborhood.centerPixelX;

  fit::GaussFit1DConfig colConfig;
  colConfig.muLo = neighborhood.centerPixelY - muRangeY;
  colConfig.muHi = neighborhood.centerPixelY + muRangeY;
  colConfig.sigmaLo = m_cfg.sigmaLoBound();
  colConfig.sigmaHi = m_cfg.sigmaHiBound();
  colConfig.qMax = maxColCharge;
  colConfig.pixelSpacing = m_cfg.pixelSpacingYMM;
  colConfig.errorPercent = m_cfg.fitErrorPercentOfMax;
  colConfig.distanceErrorConfig = distErrCfg;
  colConfig.distanceErrorConfig.pixelSpacing = m_cfg.pixelSpacingYMM;
  colConfig.centerPosition = neighborhood.centerPixelY;

  // Perform fits
  fitRowX = fit::fitGaussian1D(rowPositions, rowCharges, rowConfig);
  fitColY = fit::fitGaussian1D(colPositions, colCharges, colConfig);

  // Extract reconstructed positions from fit means
  double reconX = neighborhood.centerPixelX;
  double reconY = neighborhood.centerPixelY;

  if (fitRowX.converged && std::isfinite(fitRowX.mu)) {
    reconX = fitRowX.mu;
  } else {
    // Fallback to weighted centroid for X
    auto [centroidX, ok] = fit::weightedCentroid(rowPositions, rowCharges, 0.0);
    if (ok) reconX = centroidX;
  }

  if (fitColY.converged && std::isfinite(fitColY.mu)) {
    reconY = fitColY.mu;
  } else {
    // Fallback to weighted centroid for Y
    auto [centroidY, ok] = fit::weightedCentroid(colPositions, colCharges, 0.0);
    if (ok) reconY = centroidY;
  }

  return {reconX, reconY, centerZ};
}

std::array<double, 3> ChargeSharingReconstructor::reconstructGaussian2D(
    const core::NeighborhoodResult& neighborhood,
    double centerZ,
    fit::GaussFit2DResult& fit2D) const {

  // Build position and charge vectors for 2D fit
  // Use charge values (which include noise if enabled) for fitting
  std::vector<double> xPositions, yPositions, charges;
  xPositions.reserve(neighborhood.pixels.size());
  yPositions.reserve(neighborhood.pixels.size());
  charges.reserve(neighborhood.pixels.size());

  double maxCharge = 0.0;
  for (const auto& pixel : neighborhood.pixels) {
    if (pixel.inBounds && pixel.charge > 0.0) {
      xPositions.push_back(pixel.centerX);
      yPositions.push_back(pixel.centerY);
      charges.push_back(pixel.charge);
      maxCharge = std::max(maxCharge, pixel.charge);
    }
  }

  // Configure 2D fit
  const double muRangeX = m_cfg.muRangeMM();
  const double muRangeY = m_cfg.muRangeMM();

  // Set up distance-weighted error config if enabled
  fit::DistanceWeightedErrorConfig distErrCfg;
  distErrCfg.enabled = m_cfg.fitUseDistanceWeightedErrors;
  distErrCfg.scalePixels = m_cfg.fitDistanceScalePixels;
  distErrCfg.exponent = m_cfg.fitDistanceExponent;
  distErrCfg.floorPercent = m_cfg.fitDistanceFloorPercent;
  distErrCfg.capPercent = m_cfg.fitDistanceCapPercent;
  distErrCfg.powerInverse = m_cfg.fitDistancePowerInverse;
  distErrCfg.pixelSpacing = m_cfg.pixelSpacingMM;
  distErrCfg.truthCenterX = neighborhood.centerPixelX;
  distErrCfg.truthCenterY = neighborhood.centerPixelY;
  distErrCfg.preferTruthCenter = m_cfg.fitDistancePreferTruthCenter;

  fit::GaussFit2DConfig config;
  config.muXLo = neighborhood.centerPixelX - muRangeX;
  config.muXHi = neighborhood.centerPixelX + muRangeX;
  config.muYLo = neighborhood.centerPixelY - muRangeY;
  config.muYHi = neighborhood.centerPixelY + muRangeY;
  config.sigmaLo = m_cfg.sigmaLoBound();
  config.sigmaHi = m_cfg.sigmaHiBound();
  config.qMax = maxCharge;
  config.pixelSpacing = m_cfg.pixelSpacingMM;
  config.errorPercent = m_cfg.fitErrorPercentOfMax;
  config.distanceErrorConfig = distErrCfg;

  // Perform fit
  fit2D = fit::fitGaussian2D(xPositions, yPositions, charges, config);

  // Extract reconstructed positions from fit means
  double reconX = neighborhood.centerPixelX;
  double reconY = neighborhood.centerPixelY;

  if (fit2D.converged && std::isfinite(fit2D.muX) && std::isfinite(fit2D.muY)) {
    reconX = fit2D.muX;
    reconY = fit2D.muY;
  } else {
    // Fallback to centroid
    return reconstructCentroid(neighborhood, centerZ);
  }

  return {reconX, reconY, centerZ};
}

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::findNearestPixelFallback(
    const std::array<double, 3>& positionMM) const {
  // DD4hep-style: index = floor((position + 0.5*pitch - offset) / pitch)
  const double offsetX = m_cfg.effectiveGridOffsetXMM();
  const double offsetY = m_cfg.effectiveGridOffsetYMM();

  int i = ChargeSharingConfig::positionToIndex(positionMM[0], m_cfg.pixelSpacingMM, offsetX);
  int j = ChargeSharingConfig::positionToIndex(positionMM[1], m_cfg.effectivePixelSpacingYMM(), offsetY);

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

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::pixelLocationFromIndices(
    int indexI, int indexJ) const {
  PixelLocation loc{};
  loc.indexI = indexI;
  loc.indexJ = indexJ;

  const double pixelCenterZ = m_cfg.detectorZCenterMM + m_cfg.detectorThicknessMM / 2.0 +
                              m_cfg.pixelThicknessMM / 2.0;

  // DD4hep-style: position = index * pitch + offset
  const double pitchX = m_cfg.pixelSpacingMM;
  const double pitchY = m_cfg.effectivePixelSpacingYMM();
  const double offsetX = m_cfg.effectiveGridOffsetXMM();
  const double offsetY = m_cfg.effectiveGridOffsetYMM();

  loc.center = {
      ChargeSharingConfig::indexToPosition(indexI, pitchX, offsetX),
      ChargeSharingConfig::indexToPosition(indexJ, pitchY, offsetY),
      pixelCenterZ
  };

  return loc;
}

bool ChargeSharingReconstructor::isPixelIndexInBounds(int indexI, int indexJ) const {
  return m_boundsX.contains(indexI) && m_boundsY.contains(indexJ);
}

}  // namespace epic::chargesharing
