#ifndef EPICCHARGESHARING_CHARGEUTILS_H_
#define EPICCHARGESHARING_CHARGEUTILS_H_

#include <cmath>
#include <limits>

inline constexpr double kElectronCharge = 1.602176634e-19;  // Coulombs

inline double ComputeQnQiPercent(double qi, double qn, double maxQ) {
  if (!std::isfinite(qi) || !std::isfinite(qn) || !std::isfinite(maxQ)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (maxQ <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (qi == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double ratio = qn / qi;
  if (!std::isfinite(ratio) || ratio <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double scaledError = 0.05 * maxQ * ratio;
  if (!std::isfinite(scaledError) || scaledError <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return scaledError;
}

#endif  // EPICCHARGESHARING_CHARGEUTILS_H_