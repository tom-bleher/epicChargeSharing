/// @file GaussianFit.hh
/// @brief Gaussian fitting routines for charge sharing position reconstruction.
///
/// Provides 1D and 2D Gaussian fitting for extracting hit positions from
/// charge distributions. Adapted from the simulation's GaussianFitter.cc
/// for use in EICrecon reconstruction.

#ifndef EPIC_GAUSSIAN_FIT_HH
#define EPIC_GAUSSIAN_FIT_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// ROOT fitting headers
#include <TF1.h>
#include <TF2.h>
#include <Math/MinimizerOptions.h>
#include <Math/WrappedMultiTF1.h>
#include <Fit/Fitter.h>
#include <Fit/BinData.h>

namespace epic::chargesharing::fit {

// ============================================================================
// Constants
// ============================================================================

namespace constants {
constexpr double kInvalidValue = std::numeric_limits<double>::quiet_NaN();
constexpr double kMinAmplitude = 1e-18;
constexpr double kDefaultErrorPercent = 5.0;
}  // namespace constants

// ============================================================================
// Distance-Weighted Error Configuration
// ============================================================================

/// Configuration for distance-weighted fit errors
struct DistanceWeightedErrorConfig {
  bool enabled{false};             ///< Master switch for distance-weighted errors
  double scalePixels{1.0};         ///< Distance scale in pixel units
  double exponent{1.0};            ///< Power law exponent
  double floorPercent{1.0};        ///< Minimum error as % of Q_max
  double capPercent{50.0};         ///< Maximum error as % of Q_max
  bool powerInverse{true};         ///< Use inverse power model (high weight at center)
  double pixelSpacing{0.5};        ///< Pixel pitch for distance calculation (mm)
  double truthCenterX{0.0};        ///< True hit position X (for distance calc)
  double truthCenterY{0.0};        ///< True hit position Y (for distance calc)
  bool preferTruthCenter{true};    ///< Use truth position instead of fit estimate
};

// ============================================================================
// Distance-Weighted Error Functions
// ============================================================================

/// Apply sigma bounds (floor and cap) as percentage of max charge
inline double applySigmaBounds(double sigma, double maxCharge,
                               double floorPercent, double capPercent) {
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    return constants::kInvalidValue;
  }
  double floorAbs = 0.0;
  if (std::isfinite(maxCharge) && maxCharge > 0.0 &&
      std::isfinite(floorPercent) && floorPercent > 0.0) {
    floorAbs = floorPercent * 0.01 * maxCharge;
  }
  double capAbs = std::numeric_limits<double>::infinity();
  if (std::isfinite(maxCharge) && maxCharge > 0.0 &&
      std::isfinite(capPercent) && capPercent > 0.0) {
    capAbs = capPercent * 0.01 * maxCharge;
  }
  if (floorAbs > 0.0 && sigma < floorAbs) {
    sigma = floorAbs;
  }
  if (sigma > capAbs) {
    sigma = capAbs;
  }
  return std::isfinite(sigma) && sigma > 0.0 ? sigma : constants::kInvalidValue;
}

/// Compute distance-weighted sigma using power law model
/// sigma = floor * (1 + r)^exponent, clamped to [floor, cap]
inline double distancePowerSigma(double distance, double maxCharge,
                                 double pixelSpacing, double scalePixels,
                                 double exponent, double floorPercent,
                                 double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0 ||
      !std::isfinite(maxCharge) || maxCharge <= 0.0 ||
      !std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return constants::kInvalidValue;
  }
  const double distanceScale = std::max(scalePixels * pixelSpacing, 1e-12);
  const double ratio = distance / distanceScale;
  const double baseSigma = std::max(0.0, floorPercent) * 0.01 * maxCharge *
                           std::pow(1.0 + ratio, std::max(0.0, exponent));
  return applySigmaBounds(baseSigma, maxCharge, floorPercent, capPercent);
}

/// Compute distance-weighted sigma using inverse power model
/// sigma = cap / (1 + r)^exponent, clamped to [floor, cap]
/// This gives smaller errors (higher weight) at the center
inline double distancePowerSigmaInverse(double distance, double maxCharge,
                                        double pixelSpacing, double scalePixels,
                                        double exponent, double floorPercent,
                                        double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0 ||
      !std::isfinite(maxCharge) || maxCharge <= 0.0 ||
      !std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return constants::kInvalidValue;
  }
  const double distanceScale = std::max(scalePixels * pixelSpacing, 1e-12);
  const double ratio = distance / distanceScale;
  double sigmaMax = (capPercent / 100.0) * maxCharge;
  const double denom = std::pow(1.0 + std::max(0.0, ratio), std::max(0.0, exponent));
  const double sigmaRaw = (denom > 0.0) ? (sigmaMax / denom) : sigmaMax;
  return applySigmaBounds(sigmaRaw, maxCharge, floorPercent, capPercent);
}

/// Compute distance-weighted error for a pixel
/// @param pixelX X position of pixel center
/// @param pixelY Y position of pixel center
/// @param config Distance-weighted error configuration
/// @param maxCharge Maximum charge in the distribution
/// @return Sigma (error) value for this pixel
inline double computeDistanceWeightedError(double pixelX, double pixelY,
                                           const DistanceWeightedErrorConfig& config,
                                           double maxCharge) {
  if (!config.enabled) {
    return config.floorPercent * 0.01 * maxCharge;  // Uniform error
  }

  const double dx = pixelX - config.truthCenterX;
  const double dy = pixelY - config.truthCenterY;
  const double distance = std::sqrt(dx * dx + dy * dy);

  if (config.powerInverse) {
    return distancePowerSigmaInverse(distance, maxCharge, config.pixelSpacing,
                                     config.scalePixels, config.exponent,
                                     config.floorPercent, config.capPercent);
  } else {
    return distancePowerSigma(distance, maxCharge, config.pixelSpacing,
                              config.scalePixels, config.exponent,
                              config.floorPercent, config.capPercent);
  }
}

/// Compute distance-weighted error for 1D fit (distance from center position)
inline double computeDistanceWeightedError1D(double pixelPos, double centerPos,
                                             const DistanceWeightedErrorConfig& config,
                                             double maxCharge) {
  if (!config.enabled) {
    return config.floorPercent * 0.01 * maxCharge;
  }

  const double distance = std::abs(pixelPos - centerPos);

  if (config.powerInverse) {
    return distancePowerSigmaInverse(distance, maxCharge, config.pixelSpacing,
                                     config.scalePixels, config.exponent,
                                     config.floorPercent, config.capPercent);
  } else {
    return distancePowerSigma(distance, maxCharge, config.pixelSpacing,
                              config.scalePixels, config.exponent,
                              config.floorPercent, config.capPercent);
  }
}

// ============================================================================
// Fit Result Structures
// ============================================================================

/// Result of 1D Gaussian fit
struct GaussFit1DResult {
  bool converged{false};
  double A{constants::kInvalidValue};       ///< Amplitude
  double mu{constants::kInvalidValue};      ///< Mean (position)
  double sigma{constants::kInvalidValue};   ///< Width
  double B{constants::kInvalidValue};       ///< Baseline offset
  double chi2{constants::kInvalidValue};
  double ndf{constants::kInvalidValue};
};

/// Result of 2D Gaussian fit
struct GaussFit2DResult {
  bool converged{false};
  double A{constants::kInvalidValue};       ///< Amplitude
  double muX{constants::kInvalidValue};     ///< Mean X (position)
  double muY{constants::kInvalidValue};     ///< Mean Y (position)
  double sigmaX{constants::kInvalidValue};  ///< Width X
  double sigmaY{constants::kInvalidValue};  ///< Width Y
  double B{constants::kInvalidValue};       ///< Baseline offset
  double chi2{constants::kInvalidValue};
  double ndf{constants::kInvalidValue};
};

/// Combined reconstruction result
struct ReconstructionResult {
  double reconX{constants::kInvalidValue};  ///< Reconstructed X position
  double reconY{constants::kInvalidValue};  ///< Reconstructed Y position
  double reconZ{constants::kInvalidValue};  ///< Z position (unchanged from hit)

  GaussFit1DResult fitRowX;   ///< Row fit result (for X position)
  GaussFit1DResult fitColY;   ///< Column fit result (for Y position)
  GaussFit2DResult fit2D;     ///< 2D fit result (alternative)

  bool valid() const {
    return std::isfinite(reconX) && std::isfinite(reconY);
  }
};

// ============================================================================
// Gaussian Functions
// ============================================================================

/// 1D Gaussian with constant offset: A * exp(-0.5*((x-mu)/sigma)^2) + B
inline double gauss1DPlusB(double* x, double* p) {
  const double A = p[0];
  const double mu = p[1];
  const double sigma = p[2];
  const double B = p[3];
  const double dx = (x[0] - mu) / sigma;
  return A * std::exp(-0.5 * dx * dx) + B;
}

/// 2D Gaussian with constant offset:
/// A * exp(-0.5 * ((x-mux)^2/sigx^2 + (y-muy)^2/sigy^2)) + B
inline double gauss2DPlusB(double* xy, double* p) {
  const double A = p[0];
  const double muX = p[1];
  const double muY = p[2];
  const double sigX = p[3];
  const double sigY = p[4];
  const double B = p[5];
  const double dx = (xy[0] - muX) / sigX;
  const double dy = (xy[1] - muY) / sigY;
  return A * std::exp(-0.5 * (dx * dx + dy * dy)) + B;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate weighted centroid as fallback
inline std::pair<double, bool> weightedCentroid(
    const std::vector<double>& positions,
    const std::vector<double>& charges,
    double baseline = 0.0) {

  if (positions.size() != charges.size() || positions.empty()) {
    return {constants::kInvalidValue, false};
  }

  double weightedSum = 0.0;
  double weightTotal = 0.0;

  for (size_t i = 0; i < positions.size(); ++i) {
    const double weight = std::max(0.0, charges[i] - baseline);
    if (weight > 0.0) {
      weightTotal += weight;
      weightedSum += weight * positions[i];
    }
  }

  if (weightTotal <= 0.0) {
    return {constants::kInvalidValue, false};
  }

  return {weightedSum / weightTotal, true};
}

/// Estimate initial sigma from weighted variance
inline double estimateSigma(
    const std::vector<double>& positions,
    const std::vector<double>& charges,
    double baseline,
    double pixelSpacing,
    double sigmaLo,
    double sigmaHi) {

  double weightedSum = 0.0;
  double weightTotal = 0.0;

  for (size_t i = 0; i < positions.size(); ++i) {
    const double weight = std::max(0.0, charges[i] - baseline);
    if (weight > 0.0) {
      weightTotal += weight;
      weightedSum += weight * positions[i];
    }
  }

  double sigma = constants::kInvalidValue;
  if (weightTotal > 0.0) {
    const double mean = weightedSum / weightTotal;
    double variance = 0.0;
    for (size_t i = 0; i < positions.size(); ++i) {
      const double weight = std::max(0.0, charges[i] - baseline);
      if (weight > 0.0) {
        const double dx = positions[i] - mean;
        variance += weight * dx * dx;
      }
    }
    variance /= weightTotal;
    sigma = std::sqrt(std::max(variance, 1e-12));
  }

  if (!std::isfinite(sigma) || sigma <= 0.0) {
    sigma = std::max(0.25 * pixelSpacing, 1e-6);
  }

  return std::clamp(sigma, sigmaLo, sigmaHi);
}

// ============================================================================
// 1D Gaussian Fitting
// ============================================================================

/// Configuration for 1D Gaussian fit
struct GaussFit1DConfig {
  double muLo;            ///< Lower bound for mu
  double muHi;            ///< Upper bound for mu
  double sigmaLo;         ///< Lower bound for sigma
  double sigmaHi;         ///< Upper bound for sigma
  double qMax;            ///< Maximum charge value (for uncertainty estimate)
  double pixelSpacing;    ///< Pixel pitch (mm)
  double errorPercent{constants::kDefaultErrorPercent};  ///< Error as % of max
  DistanceWeightedErrorConfig distanceErrorConfig{};     ///< Distance-weighted error config
  double centerPosition{0.0};  ///< Center position for distance calculation (1D)
};

/// Perform 1D Gaussian fit on charge distribution
/// @param positions Pixel center positions
/// @param charges Charge values at each position
/// @param config Fit configuration
/// @return Fit result with parameters and convergence status
inline GaussFit1DResult fitGaussian1D(
    const std::vector<double>& positions,
    const std::vector<double>& charges,
    const GaussFit1DConfig& config) {

  GaussFit1DResult result;

  if (positions.size() != charges.size() || positions.size() < 3) {
    return result;
  }

  // Find min/max for initial estimates
  auto [minIt, maxIt] = std::minmax_element(charges.begin(), charges.end());
  const double qMin = *minIt;
  const double qMax = *maxIt;
  const double A0 = std::max(constants::kMinAmplitude, qMax - qMin);
  const double B0 = qMin;

  // Find position of maximum charge
  const size_t maxIdx = std::distance(charges.begin(), maxIt);
  const double mu0 = positions[maxIdx];

  // Estimate initial sigma
  const double sigma0 = estimateSigma(positions, charges, B0,
                                       config.pixelSpacing,
                                       config.sigmaLo, config.sigmaHi);

  // Calculate base uncertainty
  const double uniformSigma = std::max(1e-12, config.qMax * config.errorPercent / 100.0);

  // Compute per-pixel errors (distance-weighted if enabled)
  std::vector<double> perPixelErrors(positions.size());
  const bool useDistanceErrors = config.distanceErrorConfig.enabled;
  for (size_t i = 0; i < positions.size(); ++i) {
    if (useDistanceErrors) {
      perPixelErrors[i] = computeDistanceWeightedError1D(
          positions[i], config.centerPosition,
          config.distanceErrorConfig, config.qMax);
      if (!std::isfinite(perPixelErrors[i]) || perPixelErrors[i] <= 0.0) {
        perPixelErrors[i] = uniformSigma;
      }
    } else {
      perPixelErrors[i] = uniformSigma;
    }
  }

  // Set up TF1 for fitting
  thread_local TF1 fitFunc("fitGauss1D", gauss1DPlusB, -1e9, 1e9, 4);

  auto [minPos, maxPos] = std::minmax_element(positions.begin(), positions.end());
  const double margin = 0.5 * config.pixelSpacing;
  fitFunc.SetRange(*minPos - margin, *maxPos + margin);

  const double amplitudeMax = std::max(constants::kMinAmplitude, 2.0 * qMax);
  const double baselineMax = std::max(constants::kMinAmplitude, qMax);

  fitFunc.SetParameters(A0, mu0, sigma0, B0);
  fitFunc.SetParLimits(0, constants::kMinAmplitude, amplitudeMax);
  fitFunc.SetParLimits(1, config.muLo, config.muHi);
  fitFunc.SetParLimits(2, config.sigmaLo, config.sigmaHi);
  fitFunc.SetParLimits(3, -baselineMax, baselineMax);

  // Set up ROOT Fitter
  ROOT::Math::WrappedMultiTF1 wrapped(fitFunc, 1);
  ROOT::Fit::BinData data(static_cast<unsigned int>(positions.size()), 1);

  for (size_t i = 0; i < positions.size(); ++i) {
    data.Add(positions[i], charges[i], perPixelErrors[i]);
  }

  ROOT::Fit::Fitter fitter;
  fitter.Config().SetMinimizer("Minuit2", "Fumili2");
  fitter.Config().MinimizerOptions().SetStrategy(0);
  fitter.Config().MinimizerOptions().SetTolerance(1e-4);
  fitter.Config().MinimizerOptions().SetPrintLevel(0);
  fitter.SetFunction(wrapped);

  fitter.Config().ParSettings(0).SetLimits(constants::kMinAmplitude, amplitudeMax);
  fitter.Config().ParSettings(1).SetLimits(config.muLo, config.muHi);
  fitter.Config().ParSettings(2).SetLimits(config.sigmaLo, config.sigmaHi);
  fitter.Config().ParSettings(3).SetLimits(-baselineMax, baselineMax);
  fitter.Config().ParSettings(0).SetValue(A0);
  fitter.Config().ParSettings(1).SetValue(mu0);
  fitter.Config().ParSettings(2).SetValue(sigma0);
  fitter.Config().ParSettings(3).SetValue(B0);

  bool ok = fitter.Fit(data);
  if (!ok) {
    // Retry with different minimizer
    fitter.Config().SetMinimizer("Minuit2", "Migrad");
    fitter.Config().MinimizerOptions().SetStrategy(1);
    fitter.Config().MinimizerOptions().SetTolerance(1e-3);
    ok = fitter.Fit(data);
  }

  if (ok) {
    result.converged = true;
    result.A = fitter.Result().Parameter(0);
    result.mu = fitter.Result().Parameter(1);
    result.sigma = fitter.Result().Parameter(2);
    result.B = fitter.Result().Parameter(3);
    result.chi2 = fitter.Result().Chi2();
    result.ndf = fitter.Result().Ndf();
  }

  return result;
}

// ============================================================================
// 2D Gaussian Fitting
// ============================================================================

/// Configuration for 2D Gaussian fit
struct GaussFit2DConfig {
  double muXLo, muXHi;      ///< Bounds for mu_x
  double muYLo, muYHi;      ///< Bounds for mu_y
  double sigmaLo, sigmaHi;  ///< Bounds for sigma
  double qMax;              ///< Maximum charge value
  double pixelSpacing;      ///< Pixel pitch (mm)
  double errorPercent{constants::kDefaultErrorPercent};
  DistanceWeightedErrorConfig distanceErrorConfig{};  ///< Distance-weighted error config
};

/// Perform 2D Gaussian fit on charge distribution
/// @param xPositions X positions of pixels
/// @param yPositions Y positions of pixels
/// @param charges Charge values at each pixel
/// @param config Fit configuration
/// @return Fit result with parameters and convergence status
inline GaussFit2DResult fitGaussian2D(
    const std::vector<double>& xPositions,
    const std::vector<double>& yPositions,
    const std::vector<double>& charges,
    const GaussFit2DConfig& config) {

  GaussFit2DResult result;

  const size_t nPts = xPositions.size();
  if (nPts != yPositions.size() || nPts != charges.size() || nPts < 5) {
    return result;
  }

  // Find min/max for initial estimates
  auto [minIt, maxIt] = std::minmax_element(charges.begin(), charges.end());
  const double qMin = *minIt;
  const double qMax = *maxIt;
  const double A0 = std::max(constants::kMinAmplitude, qMax - qMin);
  const double B0 = qMin;

  // Find position of maximum charge
  const size_t maxIdx = std::distance(charges.begin(), maxIt);
  const double muX0 = xPositions[maxIdx];
  const double muY0 = yPositions[maxIdx];

  // Estimate initial sigmas from weighted variance
  auto estimateSigma2D = [&](bool forX) -> double {
    double wsum = 0.0, m = 0.0;
    for (size_t i = 0; i < nPts; ++i) {
      const double w = std::max(0.0, charges[i] - B0);
      const double c = forX ? xPositions[i] : yPositions[i];
      wsum += w;
      m += w * c;
    }
    if (wsum <= 0.0) {
      return std::clamp(0.25 * config.pixelSpacing, config.sigmaLo, config.sigmaHi);
    }
    m /= wsum;
    double var = 0.0;
    for (size_t i = 0; i < nPts; ++i) {
      const double w = std::max(0.0, charges[i] - B0);
      const double c = forX ? xPositions[i] : yPositions[i];
      const double d = c - m;
      var += w * d * d;
    }
    var /= wsum;
    return std::clamp(std::sqrt(std::max(var, 1e-12)), config.sigmaLo, config.sigmaHi);
  };

  const double sigX0 = estimateSigma2D(true);
  const double sigY0 = estimateSigma2D(false);

  // Calculate base uncertainty
  const double uniformSigma = std::max(1e-12, config.qMax * config.errorPercent / 100.0);

  // Compute per-pixel errors (distance-weighted if enabled)
  std::vector<double> perPixelErrors(nPts);
  const bool useDistanceErrors = config.distanceErrorConfig.enabled;
  for (size_t i = 0; i < nPts; ++i) {
    if (useDistanceErrors) {
      perPixelErrors[i] = computeDistanceWeightedError(
          xPositions[i], yPositions[i],
          config.distanceErrorConfig, config.qMax);
      if (!std::isfinite(perPixelErrors[i]) || perPixelErrors[i] <= 0.0) {
        perPixelErrors[i] = uniformSigma;
      }
    } else {
      perPixelErrors[i] = uniformSigma;
    }
  }

  // Find range
  auto [xMinIt, xMaxIt] = std::minmax_element(xPositions.begin(), xPositions.end());
  auto [yMinIt, yMaxIt] = std::minmax_element(yPositions.begin(), yPositions.end());
  const double margin = 0.5 * config.pixelSpacing;
  const double xMin = *xMinIt - margin;
  const double xMax = *xMaxIt + margin;
  const double yMin = *yMinIt - margin;
  const double yMax = *yMaxIt + margin;

  // Set up TF2 for fitting
  thread_local TF2 fitFunc("fitGauss2D", gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);
  fitFunc.SetRange(xMin, yMin, xMax, yMax);

  const double amplitudeMax = std::max(constants::kMinAmplitude, 2.0 * qMax);
  const double baselineMax = std::max(constants::kMinAmplitude, qMax);

  // Set up ROOT Fitter
  ROOT::Math::WrappedMultiTF1 wrapped(fitFunc, 2);
  ROOT::Fit::BinData data(static_cast<unsigned int>(nPts), 2);

  for (size_t i = 0; i < nPts; ++i) {
    double xy[2] = {xPositions[i], yPositions[i]};
    data.Add(xy, charges[i], perPixelErrors[i]);
  }

  ROOT::Fit::Fitter fitter;
  fitter.Config().SetMinimizer("Minuit2", "Fumili2");
  fitter.Config().MinimizerOptions().SetStrategy(0);
  fitter.Config().MinimizerOptions().SetTolerance(1e-4);
  fitter.Config().MinimizerOptions().SetPrintLevel(0);
  fitter.SetFunction(wrapped);

  fitter.Config().ParSettings(0).SetLimits(constants::kMinAmplitude, amplitudeMax);
  fitter.Config().ParSettings(1).SetLimits(config.muXLo, config.muXHi);
  fitter.Config().ParSettings(2).SetLimits(config.muYLo, config.muYHi);
  fitter.Config().ParSettings(3).SetLimits(config.sigmaLo, config.sigmaHi);
  fitter.Config().ParSettings(4).SetLimits(config.sigmaLo, config.sigmaHi);
  fitter.Config().ParSettings(5).SetLimits(-baselineMax, baselineMax);
  fitter.Config().ParSettings(0).SetValue(A0);
  fitter.Config().ParSettings(1).SetValue(muX0);
  fitter.Config().ParSettings(2).SetValue(muY0);
  fitter.Config().ParSettings(3).SetValue(sigX0);
  fitter.Config().ParSettings(4).SetValue(sigY0);
  fitter.Config().ParSettings(5).SetValue(B0);

  bool ok = fitter.Fit(data);
  if (!ok) {
    // Retry with different minimizer
    fitter.Config().SetMinimizer("Minuit2", "Migrad");
    fitter.Config().MinimizerOptions().SetStrategy(1);
    fitter.Config().MinimizerOptions().SetTolerance(1e-3);
    ok = fitter.Fit(data);
  }

  if (ok) {
    result.converged = true;
    result.A = fitter.Result().Parameter(0);
    result.muX = fitter.Result().Parameter(1);
    result.muY = fitter.Result().Parameter(2);
    result.sigmaX = fitter.Result().Parameter(3);
    result.sigmaY = fitter.Result().Parameter(4);
    result.B = fitter.Result().Parameter(5);
    result.chi2 = fitter.Result().Chi2();
    result.ndf = fitter.Result().Ndf();
  }

  return result;
}

}  // namespace epic::chargesharing::fit

#endif  // EPIC_GAUSSIAN_FIT_HH
