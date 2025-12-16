#ifndef EPICCHARGESHARING_FITUNCERTAINTY_H_
#define EPICCHARGESHARING_FITUNCERTAINTY_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

inline constexpr double kElectronCharge = 1.602176634e-19;  // Coulombs

namespace charge_uncert {

enum class VerticalUncertaintyModel {
  UniformPercentOfMax,
  QnQiScaled,
};

inline constexpr std::string_view VerticalUncertaintyModelName(
    VerticalUncertaintyModel model) {
  switch (model) {
    case VerticalUncertaintyModel::UniformPercentOfMax:
      return "UniformPercentOfMax";
    case VerticalUncertaintyModel::QnQiScaled:
      return "QnQiScaled";
  }
  return "Unknown";
}

inline double UniformPercentOfMax(double percentOfMax, double maxCharge) {
  if (!std::isfinite(percentOfMax) || !std::isfinite(maxCharge)) {
    return 0.0;
  }
  if (maxCharge <= 0.0) {
    return 0.0;
  }
  const double percentClamped = std::max(0.0, percentOfMax);
  if (percentClamped <= 0.0) {
    return 0.0;
  }
  const double relErr = percentClamped * 0.01;
  if (relErr <= 0.0) {
    return 0.0;
  }
  return relErr * maxCharge;
}

inline constexpr double kQnQiDefaultScale = 0.065;

inline double QnQiScaled(double qi,
                         double qn,
                         double maxQi,
                         double scale = kQnQiDefaultScale) {
  if (!std::isfinite(qi) || !std::isfinite(qn) || !std::isfinite(maxQi)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (maxQi <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (qi == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double ratio = qn / qi;
  if (!std::isfinite(ratio) || ratio <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double scaledError = scale * maxQi * ratio;
  if (!std::isfinite(scaledError) || scaledError <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return scaledError;
}

inline double SelectVerticalSigma(double candidate,
                                  double uniformSigma,
                                  double fallback = 1.0) {
  if (std::isfinite(candidate) && candidate > 0.0) {
    return candidate;
  }
  if (std::isfinite(uniformSigma) && uniformSigma > 0.0) {
    return uniformSigma;
  }
  return fallback;
}

inline double ApplySigmaBounds(double sigma,
                               double maxCharge,
                               double floorPercent,
                               double capPercent) {
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double floorAbs = 0.0;
  if (std::isfinite(maxCharge) && maxCharge > 0.0 && std::isfinite(floorPercent) &&
      floorPercent > 0.0) {
    floorAbs = floorPercent * 0.01 * maxCharge;
  }
  double capAbs = std::numeric_limits<double>::infinity();
  if (std::isfinite(maxCharge) && maxCharge > 0.0 && std::isfinite(capPercent) &&
      capPercent > 0.0) {
    capAbs = capPercent * 0.01 * maxCharge;
  }
  if (floorAbs > 0.0 && sigma < floorAbs) {
    sigma = floorAbs;
  }
  if (sigma > capAbs) {
    sigma = capAbs;
  }
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sigma;
}

inline double DistancePowerSigma(double distance,
                                 double maxCharge,
                                 double pixelSpacing,
                                 double distanceScalePixels,
                                 double exponent,
                                 double floorPercent,
                                 double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(exponent)) {
    exponent = 1.0;
  }
  const double distanceScale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double ratio = distance / distanceScale;
  const double baseSigma = std::max(0.0, floorPercent) * 0.01 * maxCharge *
                           std::pow(1.0 + ratio, std::max(0.0, exponent));
  return ApplySigmaBounds(baseSigma, maxCharge, floorPercent, capPercent);
}

inline double DistancePowerSigmaInverse(double distance,
                                        double maxCharge,
                                        double pixelSpacing,
                                        double distanceScalePixels,
                                        double exponent,
                                        double floorPercent,
                                        double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(exponent)) {
    exponent = 0.0;
  }
  const double distanceScale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double ratio = distance / distanceScale;
  const double exponentClamped = std::max(0.0, exponent);
  double sigmaMax = std::numeric_limits<double>::quiet_NaN();
  if (std::isfinite(capPercent) && capPercent > 0.0) {
    sigmaMax = (capPercent / 100.0) * maxCharge;
  }
  double sigmaMin = 0.0;
  if (std::isfinite(floorPercent) && floorPercent > 0.0) {
    sigmaMin = (floorPercent / 100.0) * maxCharge;
  }
  double base = sigmaMax;
  if (!std::isfinite(base) || base <= 0.0) {
    base = sigmaMin;
  }
  if (!std::isfinite(base) || base <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double denom = std::pow(1.0 + std::max(0.0, ratio), exponentClamped);
  const double sigmaRaw = (denom > 0.0) ? (base / denom) : base;
  return ApplySigmaBounds(sigmaRaw, maxCharge, floorPercent, capPercent);
}

inline double SigmaLinear(double distance,
                          double maxCharge,
                          double pixelSpacing,
                          double distanceScalePixels,
                          double alpha,
                          double floorPercent,
                          double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(alpha)) {
    alpha = 0.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_raw =
      (floorPercent / 100.0) * maxCharge * (1.0 + std::max(0.0, alpha) * r);
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

inline double SigmaPower(double distance,
                         double maxCharge,
                         double pixelSpacing,
                         double distanceScalePixels,
                         double exponent,
                         double floorPercent,
                         double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(exponent)) {
    exponent = 1.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_min = (floorPercent / 100.0) * maxCharge;
  const double sigma_raw =
      sigma_min * std::pow(1.0 + std::max(0.0, r), std::max(0.0, exponent));
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

inline double SigmaPowerInverse(double distance,
                                double maxCharge,
                                double pixelSpacing,
                                double distanceScalePixels,
                                double exponent,
                                double floorPercent,
                                double capPercent) {
  return DistancePowerSigmaInverse(distance,
                                   maxCharge,
                                   pixelSpacing,
                                   distanceScalePixels,
                                   exponent,
                                   floorPercent,
                                   capPercent);
}

inline double SigmaQuadratic(double distance,
                             double maxCharge,
                             double pixelSpacing,
                             double distanceScalePixels,
                             double beta,
                             double floorPercent,
                             double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(beta)) {
    beta = 0.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_min = (floorPercent / 100.0) * maxCharge;
  const double sigma_raw =
      sigma_min * (1.0 + std::max(0.0, beta) * r * r);
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

inline double SigmaExpSaturating(double distance,
                                 double maxCharge,
                                 double pixelSpacing,
                                 double distanceScalePixels,
                                 double p,
                                 double b,
                                 double floorPercent,
                                 double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(p) || p <= 0.0) {
    p = 1.0;
  }
  if (!std::isfinite(b) || b <= 0.0) {
    b = 1.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_min = (floorPercent / 100.0) * maxCharge;
  const double sigma_max = (capPercent / 100.0) * maxCharge;
  const double base_ratio = r / std::max(b, 1e-12);
  const double arg = std::pow(std::max(0.0, base_ratio), p);
  const double sigma_raw =
      sigma_min + (sigma_max - sigma_min) * (1.0 - std::exp(-arg));
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

inline double SigmaPiecewiseCoreTail(double distance,
                                     double maxCharge,
                                     double pixelSpacing,
                                     double distanceScalePixels,
                                     double r0,
                                     double beta,
                                     double exponent,
                                     double floorPercent,
                                     double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(r0) || r0 < 0.0) {
    r0 = 0.0;
  }
  if (!std::isfinite(beta)) {
    beta = 0.0;
  }
  if (!std::isfinite(exponent) || exponent <= 0.0) {
    exponent = 1.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_min = (floorPercent / 100.0) * maxCharge;
  double sigma_raw = sigma_min;
  if (r > r0) {
    const double dr = std::max(0.0, r - r0);
    sigma_raw =
        sigma_min * (1.0 + std::max(0.0, beta) * std::pow(dr, exponent));
  }
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

inline double SigmaLogistic(double distance,
                            double maxCharge,
                            double pixelSpacing,
                            double distanceScalePixels,
                            double r0,
                            double k,
                            double floorPercent,
                            double capPercent) {
  if (!std::isfinite(distance) || distance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(maxCharge) || maxCharge <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(pixelSpacing) || pixelSpacing <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(distanceScalePixels) || distanceScalePixels <= 0.0) {
    distanceScalePixels = 1.0;
  }
  if (!std::isfinite(r0)) {
    r0 = 0.0;
  }
  if (!std::isfinite(k)) {
    k = 0.0;
  }
  const double scale = std::max(distanceScalePixels * pixelSpacing, 1e-12);
  const double r = distance / scale;
  const double sigma_min = (floorPercent / 100.0) * maxCharge;
  const double sigma_max = (capPercent / 100.0) * maxCharge;
  const double steepness = std::max(0.0, k);
  const double t = 1.0 / (1.0 + std::exp(-steepness * (r - r0)));
  const double sigma_raw = sigma_min + (sigma_max - sigma_min) * t;
  return ApplySigmaBounds(sigma_raw, maxCharge, floorPercent, capPercent);
}

}  // namespace charge_uncert

inline double ComputeQnQiPercent(double qi, double qn, double maxQi) {
  return charge_uncert::QnQiScaled(qi, qn, maxQi);
}

#endif  // EPICCHARGESHARING_FITUNCERTAINTY_H_
