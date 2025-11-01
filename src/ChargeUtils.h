#ifndef EPICCHARGESHARING_CHARGEUTILS_H_
#define EPICCHARGESHARING_CHARGEUTILS_H_

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

}  // namespace charge_uncert

inline double ComputeQnQiPercent(double qi, double qn, double maxQi) {
  return charge_uncert::QnQiScaled(qi, qn, maxQi);
}

#endif  // EPICCHARGESHARING_CHARGEUTILS_H_
