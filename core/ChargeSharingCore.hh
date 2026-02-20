/// @file ChargeSharingCore.hh
/// @brief Core charge sharing algorithms shared between simulation and reconstruction.
///
/// This header provides the charge sharing model calculations (LogA/LinA)
/// that are used both by the Geant4 simulation and the EICrecon plugin.
/// It is header-only to avoid build dependencies between the two systems.

#ifndef EPIC_CHARGE_SHARING_CORE_HH
#define EPIC_CHARGE_SHARING_CORE_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace epic::chargesharing::core {

// ============================================================================
// Signal Model Selection
// ============================================================================

/// Signal sharing model type
enum class SignalModel { LogA, LinA };

/// Active pixel selection mode for denominator calculation
enum class ActivePixelMode {
    Neighborhood,   ///< All pixels in neighborhood
    RowCol,         ///< Cross pattern (center row + center column)
    RowCol3x3,      ///< Cross pattern + 3x3 center block
    ChargeBlock2x2, ///< 4 pixels with highest weight (contiguous)
    ChargeBlock3x3  ///< 9 pixels with highest weight (contiguous)
};

/// Reconstruction method (position extraction)
enum class ReconMethod {
    Centroid,   ///< Charge-weighted centroid (simple)
    Gaussian1D, ///< Fit 1D Gaussians to row and column slices
    Gaussian2D  ///< Fit 2D Gaussian to full neighborhood
};

// ============================================================================
// Constants
// ============================================================================

namespace constants {
constexpr double kMillimeterPerMicron = 1.0e-3;
constexpr double kMicronPerMillimeter = 1.0e3;
constexpr double kGuardFactor = 1.0 + 1e-6;     // Prevents log(0) singularity when distance == d0, with minimal bias
constexpr double kMinD0Micron = 1e-6;           // Floor for d0 to avoid division by zero in log(d/d0)
constexpr double kOutOfBoundsFraction = -999.0; // Sentinel value indicating pixel is outside detector bounds

// Linear model beta coefficients (1/um) from paper -- empirical fits to AC-LGAD data
constexpr double kLinearBetaNarrow = 0.003;      // For pitch 100-200 um (steeper attenuation)
constexpr double kLinearBetaWide = 0.001;        // For pitch 200-500 um (gentler attenuation)
constexpr double kLinearMinPitchUM = 100.0;      // Minimum supported AC-LGAD pitch
constexpr double kLinearBoundaryPitchUM = 200.0; // Pitch threshold between narrow/wide beta regimes
constexpr double kLinearMaxPitchUM = 500.0;      // Maximum supported AC-LGAD pitch
} // namespace constants

// ============================================================================
// Utility Functions
// ============================================================================

/// Get quiet NaN for invalid values
inline double nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

/// Check if value is finite
inline bool isFinite(double v) {
    return std::isfinite(v);
}

/// Clamp value to range
template <typename T>
inline T clamp(T value, T lo, T hi) {
    return std::max(lo, std::min(value, hi));
}

// ============================================================================
// Charge Model Calculations
// ============================================================================

/// Calculate Euclidean distance from hit point to pixel center
inline double calcDistanceToCenter(double dxMM, double dyMM) {
    return std::hypot(dxMM, dyMM);
}

/// Calculate pad view angle (alpha) - approximate solid angle subtended by pad
/// @param distanceToCenterMM Distance from hit to pixel center (mm)
/// @param padWidthMM Pad width (mm)
/// @param padHeightMM Pad height (mm)
inline double calcPadViewAngle(double distanceToCenterMM, double padWidthMM, double padHeightMM) {
    const double l = (padWidthMM + padHeightMM) / 2.0;
    const double numerator = (l / 2.0) * std::sqrt(2.0);
    const double denominator = numerator + distanceToCenterMM;
    if (distanceToCenterMM == 0.0) {
        return std::atan(1.0); // pi/4
    }
    return std::atan(numerator / denominator);
}

/// Get linear model beta coefficient based on pixel pitch
/// @param pitchMM Pixel pitch in mm
inline double getLinearBeta(double pitchMM) {
    const double pitchUM = pitchMM * constants::kMicronPerMillimeter;
    if (pitchUM >= constants::kLinearMinPitchUM && pitchUM <= constants::kLinearBoundaryPitchUM) {
        return constants::kLinearBetaNarrow;
    }
    if (pitchUM > constants::kLinearBoundaryPitchUM && pitchUM <= constants::kLinearMaxPitchUM) {
        return constants::kLinearBetaWide;
    }
    return constants::kLinearBetaNarrow;
}

/// Calculate weight for LogA model: w_i = alpha_i / ln(d_i/d0)
/// @param distanceMM Distance from hit to pixel center (mm)
/// @param alphaPadViewAngle Pad view angle
/// @param d0MM Reference distance d0 (mm)
inline double calcWeightLogA(double distanceMM, double alphaPadViewAngle, double d0MM) {
    const double minSafeDistance = d0MM * constants::kGuardFactor;
    const double safeDistance = std::max(distanceMM, minSafeDistance);
    const double logValue = std::log(safeDistance / d0MM);
    if (!(logValue > 0.0 && std::isfinite(logValue))) {
        static thread_local bool warned = false;
        if (!warned) {
            std::cerr << "[ChargeSharingCore] Warning: calcWeightLogA returned 0 due to invalid logValue=" << logValue
                      << " (distance=" << distanceMM << ", d0=" << d0MM << ")\n";
            warned = true;
        }
        return 0.0;
    }
    return alphaPadViewAngle / logValue;
}

/// Calculate weight for LinA model: w_i = (1 - beta * d_i) * alpha_i
/// @param distanceMM Distance from hit to pixel center (mm)
/// @param alphaPadViewAngle Pad view angle
/// @param betaPerMicron Attenuation coefficient (1/um)
inline double calcWeightLinA(double distanceMM, double alphaPadViewAngle, double betaPerMicron) {
    const double distanceUM = distanceMM * constants::kMicronPerMillimeter;
    const double attenuation = std::max(0.0, 1.0 - betaPerMicron * distanceUM);
    const double weight = attenuation * alphaPadViewAngle;
    if (!(std::isfinite(weight) && weight >= 0.0)) {
        static thread_local bool warned = false;
        if (!warned) {
            std::cerr << "[ChargeSharingCore] Warning: calcWeightLinA returned non-finite/negative weight=" << weight
                      << " (distance=" << distanceMM << ", beta=" << betaPerMicron << ")\n";
            warned = true;
        }
        return 0.0;
    }
    return weight;
}

/// Calculate charge sharing weight using specified model
/// @param model Signal model (LogA or LinA)
/// @param distanceMM Distance from hit to pixel center (mm)
/// @param padWidthMM Pad width (mm)
/// @param padHeightMM Pad height (mm)
/// @param d0MM Reference distance for LogA (mm)
/// @param betaPerMicron Attenuation for LinA (1/um), or 0 to auto-compute
/// @param pitchMM Pixel pitch for auto-computing beta (mm)
inline double calcWeight(SignalModel model, double distanceMM, double padWidthMM, double padHeightMM, double d0MM,
                         double betaPerMicron = 0.0, double pitchMM = 0.0) {
    const double alpha = calcPadViewAngle(distanceMM, padWidthMM, padHeightMM);

    if (model == SignalModel::LinA) {
        const double beta = (betaPerMicron > 0.0) ? betaPerMicron : getLinearBeta(pitchMM);
        return calcWeightLinA(distanceMM, alpha, beta);
    }
    // Default to LogA
    return calcWeightLogA(distanceMM, alpha, d0MM);
}

// ============================================================================
// Neighborhood Data Structures
// ============================================================================

/// Data for a single pixel in the neighborhood
struct NeighborPixel {
    int di{0};            ///< Offset from center pixel (row)
    int dj{0};            ///< Offset from center pixel (col)
    int globalIndex{-1};  ///< Global pixel index
    double centerX{0.0};  ///< Pixel center X (mm)
    double centerY{0.0};  ///< Pixel center Y (mm)
    double distance{0.0}; ///< Distance from hit to pixel center (mm)
    double alpha{0.0};    ///< Pad view angle
    double weight{0.0};   ///< Raw weight (before normalization)
    double fraction{0.0}; ///< Normalized charge fraction (full neighborhood denominator)
    double charge{0.0};   ///< Charge in Coulombs (optional)
    bool inBounds{false}; ///< Within detector bounds

    // Mode-specific fractions (different denominators for diagnostics)
    double fractionRow{0.0};   ///< Fraction using row-only denominator
    double fractionCol{0.0};   ///< Fraction using col-only denominator
    double fractionBlock{0.0}; ///< Fraction using block denominator

    // Mode-specific charges (fraction * totalCharge)
    double chargeRow{0.0};   ///< Charge using row fraction
    double chargeCol{0.0};   ///< Charge using col fraction
    double chargeBlock{0.0}; ///< Charge using block fraction
};

/// Complete neighborhood calculation result
struct NeighborhoodResult {
    std::vector<NeighborPixel> pixels;
    int centerPixelI{0};
    int centerPixelJ{0};
    double centerPixelX{0.0};
    double centerPixelY{0.0};
    double totalWeight{0.0};

    /// Get pixel at grid offset (di, dj), returns nullptr if not found
    const NeighborPixel* getPixel(int di, int dj) const {
        for (const auto& p : pixels) {
            if (p.di == di && p.dj == dj && p.inBounds) {
                return &p;
            }
        }
        return nullptr;
    }

    /// Get center row slice (di varies, dj=0)
    std::vector<const NeighborPixel*> getCenterRow() const {
        std::vector<const NeighborPixel*> row;
        for (const auto& p : pixels) {
            if (p.dj == 0 && p.inBounds) {
                row.push_back(&p);
            }
        }
        // Sort by di
        std::sort(row.begin(), row.end(), [](const NeighborPixel* a, const NeighborPixel* b) { return a->di < b->di; });
        return row;
    }

    /// Get center column slice (di=0, dj varies)
    std::vector<const NeighborPixel*> getCenterCol() const {
        std::vector<const NeighborPixel*> col;
        for (const auto& p : pixels) {
            if (p.di == 0 && p.inBounds) {
                col.push_back(&p);
            }
        }
        // Sort by dj
        std::sort(col.begin(), col.end(), [](const NeighborPixel* a, const NeighborPixel* b) { return a->dj < b->dj; });
        return col;
    }
};

// ============================================================================
// Neighborhood Calculation
// ============================================================================

/// Configuration for neighborhood calculation
struct NeighborhoodConfig {
    SignalModel signalModel{SignalModel::LogA};
    ActivePixelMode activeMode{ActivePixelMode::Neighborhood};
    int radius{2};               ///< Neighborhood half-width (2 = 5x5)
    double pixelSizeMM{0.15};    ///< Pad size (mm)
    double pixelSizeYMM{0.15};   ///< Pad size Y (mm), 0 = same as X
    double pixelSpacingMM{0.5};  ///< Pixel pitch (mm)
    double pixelSpacingYMM{0.5}; ///< Pixel pitch Y (mm), 0 = same as X
    double d0Micron{1.0};        ///< LogA d0 parameter (um)
    double betaPerMicron{0.0};   ///< LinA beta (1/um), 0 = auto
    int numPixelsX{0};           ///< Total pixels in X (for bounds), 0 = unbounded
    int numPixelsY{0};           ///< Total pixels in Y (for bounds), 0 = unbounded
    int minIndexX{0};            ///< Minimum valid DD4hep index in X (can be negative)
    int minIndexY{0};            ///< Minimum valid DD4hep index in Y (can be negative)
};

/// Calculate charge fractions for neighborhood around hit position
/// @param hitX Hit X position (mm)
/// @param hitY Hit Y position (mm)
/// @param centerI Center pixel row index
/// @param centerJ Center pixel column index
/// @param centerX Center pixel X position (mm)
/// @param centerY Center pixel Y position (mm)
/// @param config Neighborhood configuration
/// @return Neighborhood result with pixel fractions
inline NeighborhoodResult calculateNeighborhood(double hitX, double hitY, int centerI, int centerJ, double centerX,
                                                double centerY, const NeighborhoodConfig& config) {

    NeighborhoodResult result;
    result.centerPixelI = centerI;
    result.centerPixelJ = centerJ;
    result.centerPixelX = centerX;
    result.centerPixelY = centerY;

    const int radius = std::max(0, config.radius);
    const int gridDim = 2 * radius + 1;
    const double d0MM = config.d0Micron * constants::kMillimeterPerMicron;
    const double padW = config.pixelSizeMM;
    const double padH = (config.pixelSizeYMM > 0) ? config.pixelSizeYMM : padW;
    const double pitchX = config.pixelSpacingMM;
    const double pitchY = (config.pixelSpacingYMM > 0) ? config.pixelSpacingYMM : pitchX;

    result.pixels.reserve(gridDim * gridDim);

    // First pass: calculate weights for all pixels
    for (int di = -radius; di <= radius; ++di) {
        for (int dj = -radius; dj <= radius; ++dj) {
            NeighborPixel pixel;
            pixel.di = di;
            pixel.dj = dj;

            const int globalI = centerI + di;
            const int globalJ = centerJ + dj;

            // Check bounds if specified (indices can be negative for centered DD4hep grids)
            if (config.numPixelsX > 0 && (globalI < config.minIndexX || globalI >= config.minIndexX + config.numPixelsX)) {
                pixel.inBounds = false;
                result.pixels.push_back(pixel);
                continue;
            }
            if (config.numPixelsY > 0 && (globalJ < config.minIndexY || globalJ >= config.minIndexY + config.numPixelsY)) {
                pixel.inBounds = false;
                result.pixels.push_back(pixel);
                continue;
            }

            pixel.inBounds = true;
            pixel.globalIndex =
                (config.numPixelsY > 0)
                    ? (globalI - config.minIndexX) * config.numPixelsY + (globalJ - config.minIndexY)
                    : globalI * gridDim + globalJ;
            pixel.centerX = centerX + di * pitchX;
            pixel.centerY = centerY + dj * pitchY;

            const double dx = hitX - pixel.centerX;
            const double dy = hitY - pixel.centerY;
            pixel.distance = calcDistanceToCenter(dx, dy);
            pixel.alpha = calcPadViewAngle(pixel.distance, padW, padH);

            pixel.weight =
                calcWeight(config.signalModel, pixel.distance, padW, padH, d0MM, config.betaPerMicron, pitchX);

            result.pixels.push_back(pixel);
        }
    }

    // Determine which pixels contribute to denominator based on active mode
    std::vector<bool> isActive(result.pixels.size(), false);

    switch (config.activeMode) {
        case ActivePixelMode::Neighborhood:
            // All in-bounds pixels are active
            for (size_t i = 0; i < result.pixels.size(); ++i) {
                isActive[i] = result.pixels[i].inBounds;
            }
            break;

        case ActivePixelMode::RowCol:
            // Center row (dj=0) OR center column (di=0)
            for (size_t i = 0; i < result.pixels.size(); ++i) {
                const auto& p = result.pixels[i];
                isActive[i] = p.inBounds && (p.di == 0 || p.dj == 0);
            }
            break;

        case ActivePixelMode::RowCol3x3:
            // Cross pattern + 3x3 center block
            for (size_t i = 0; i < result.pixels.size(); ++i) {
                const auto& p = result.pixels[i];
                const bool inCross = (p.di == 0 || p.dj == 0);
                const bool in3x3 = (std::abs(p.di) <= 1 && std::abs(p.dj) <= 1);
                isActive[i] = p.inBounds && (inCross || in3x3);
            }
            break;

        case ActivePixelMode::ChargeBlock2x2:
        case ActivePixelMode::ChargeBlock3x3: {
            // Find pixel with highest weight
            size_t maxIdx = 0;
            double maxWeight = -1.0;
            for (size_t i = 0; i < result.pixels.size(); ++i) {
                if (result.pixels[i].inBounds && result.pixels[i].weight > maxWeight) {
                    maxWeight = result.pixels[i].weight;
                    maxIdx = i;
                }
            }

            const int blockSize = (config.activeMode == ActivePixelMode::ChargeBlock3x3) ? 3 : 2;
            const int maxDi = result.pixels[maxIdx].di;
            const int maxDj = result.pixels[maxIdx].dj;

            // Find best block containing max pixel
            int bestCornerDi = maxDi, bestCornerDj = maxDj;
            double bestSum = -1.0;
            const int maxOffset = (blockSize == 3) ? -2 : -1;

            for (int oi = maxOffset; oi <= 0; ++oi) {
                for (int oj = maxOffset; oj <= 0; ++oj) {
                    const int testCornerDi = maxDi + oi;
                    const int testCornerDj = maxDj + oj;

                    double sum = 0.0;
                    for (const auto& p : result.pixels) {
                        if (p.inBounds && p.di >= testCornerDi && p.di < testCornerDi + blockSize &&
                            p.dj >= testCornerDj && p.dj < testCornerDj + blockSize) {
                            sum += p.weight;
                        }
                    }

                    if (sum > bestSum) {
                        bestSum = sum;
                        bestCornerDi = testCornerDi;
                        bestCornerDj = testCornerDj;
                    }
                }
            }

            // Mark pixels in best block as active
            for (size_t i = 0; i < result.pixels.size(); ++i) {
                const auto& p = result.pixels[i];
                isActive[i] = p.inBounds && p.di >= bestCornerDi && p.di < bestCornerDi + blockSize &&
                              p.dj >= bestCornerDj && p.dj < bestCornerDj + blockSize;
            }
            break;
        }
    }

    // Calculate total weight from active pixels
    result.totalWeight = 0.0;
    for (size_t i = 0; i < result.pixels.size(); ++i) {
        if (isActive[i]) {
            result.totalWeight += result.pixels[i].weight;
        }
    }

    // Normalize to get fractions (only active pixels get non-zero fractions)
    for (size_t i = 0; i < result.pixels.size(); ++i) {
        if (isActive[i] && result.totalWeight > 0.0) {
            result.pixels[i].fraction = result.pixels[i].weight / result.totalWeight;
        } else {
            result.pixels[i].fraction = 0.0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Compute mode-specific fractions for diagnostics
    // These use different denominators: row-only, col-only, and block
    // ─────────────────────────────────────────────────────────────────────────

    // Row-mode: denominator from center row only (dj=0)
    double rowWeightSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds && p.dj == 0) {
            rowWeightSum += p.weight;
        }
    }
    for (auto& p : result.pixels) {
        if (p.inBounds && rowWeightSum > 0.0) {
            p.fractionRow = p.weight / rowWeightSum;
        }
    }

    // Col-mode: denominator from center column only (di=0)
    double colWeightSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds && p.di == 0) {
            colWeightSum += p.weight;
        }
    }
    for (auto& p : result.pixels) {
        if (p.inBounds && colWeightSum > 0.0) {
            p.fractionCol = p.weight / colWeightSum;
        }
    }

    // Block-mode: find contiguous 2x2 block with highest weight sum
    // (similar to ChargeBlock2x2 mode, but always computed for diagnostics)
    size_t maxIdx = 0;
    double maxWeight = -1.0;
    for (size_t i = 0; i < result.pixels.size(); ++i) {
        if (result.pixels[i].inBounds && result.pixels[i].weight > maxWeight) {
            maxWeight = result.pixels[i].weight;
            maxIdx = i;
        }
    }

    const int blockSize = 2; // 2x2 block for diagnostics
    int blockCornerDi = result.pixels[maxIdx].di;
    int blockCornerDj = result.pixels[maxIdx].dj;
    double bestBlockSum = -1.0;

    // Find best 2x2 block containing max pixel
    for (int oi = -1; oi <= 0; ++oi) {
        for (int oj = -1; oj <= 0; ++oj) {
            const int testCornerDi = result.pixels[maxIdx].di + oi;
            const int testCornerDj = result.pixels[maxIdx].dj + oj;

            double sum = 0.0;
            for (const auto& p : result.pixels) {
                if (p.inBounds && p.di >= testCornerDi && p.di < testCornerDi + blockSize && p.dj >= testCornerDj &&
                    p.dj < testCornerDj + blockSize) {
                    sum += p.weight;
                }
            }

            if (sum > bestBlockSum) {
                bestBlockSum = sum;
                blockCornerDi = testCornerDi;
                blockCornerDj = testCornerDj;
            }
        }
    }

    // Compute block fractions
    for (auto& p : result.pixels) {
        if (p.inBounds && p.di >= blockCornerDi && p.di < blockCornerDi + blockSize && p.dj >= blockCornerDj &&
            p.dj < blockCornerDj + blockSize && bestBlockSum > 0.0) {
            p.fractionBlock = p.weight / bestBlockSum;
        }
    }

    return result;
}

// ============================================================================
// Noise Model
// ============================================================================

/// Configuration for noise injection
struct NoiseConfig {
    bool enabled{false};
    double gainSigmaMin{0.01};                ///< Minimum per-pixel gain variation (1%)
    double gainSigmaMax{0.05};                ///< Maximum per-pixel gain variation (5%)
    double electronNoiseCount{500.0};         ///< Electronic noise RMS (electrons)
    double elementaryCharge{1.602176634e-19}; ///< For converting electrons to Coulombs
};

/// Noise model for realistic charge simulation.
/// Applies gain variations and electronic noise to charge values.
///
/// @note Thread Safety: This class contains an std::mt19937 generator and
/// distribution objects whose operator() mutates internal state. Instances
/// are NOT thread-safe; each thread must own its own NoiseModel.
class NoiseModel {
public:
    NoiseModel() : m_generator(std::random_device{}()) {}

    explicit NoiseModel(unsigned int seed) : m_generator(seed) {}

    void setConfig(const NoiseConfig& config) { m_config = config; }
    const NoiseConfig& config() const { return m_config; }

    void setSeed(unsigned int seed) { m_generator.seed(seed); }

    /// Apply noise to a single charge value (in Coulombs)
    /// @param chargeC Input charge in Coulombs
    /// @return Noisy charge in Coulombs
    double applyNoise(double chargeC) {
        if (!m_config.enabled || chargeC <= 0.0) {
            return chargeC;
        }

        // 1. Apply gain variation (multiplicative)
        //    gain_effective = gain * (1 + gaussian(0, sigma))
        //    where sigma is uniformly sampled from [gainSigmaMin, gainSigmaMax]
        double gainSigma =
            m_uniformDist(m_generator) * (m_config.gainSigmaMax - m_config.gainSigmaMin) + m_config.gainSigmaMin;
        double gainFactor = 1.0 + m_gaussDist(m_generator) * gainSigma;
        gainFactor = std::max(0.0, gainFactor); // Prevent negative gain

        double noisyCharge = chargeC * gainFactor;

        // 2. Apply electronic noise (additive)
        //    noise = gaussian(0, noiseElectrons) * elementaryCharge
        double noiseC = m_gaussDist(m_generator) * m_config.electronNoiseCount * m_config.elementaryCharge;
        noisyCharge += noiseC;

        // Charge can't be negative after noise
        return std::max(0.0, noisyCharge);
    }

    /// Apply noise to all pixels in a neighborhood result
    /// Updates the charge field of each pixel
    void applyNoise(NeighborhoodResult& result) {
        if (!m_config.enabled) {
            return;
        }

        for (auto& pixel : result.pixels) {
            if (pixel.inBounds && pixel.charge > 0.0) {
                pixel.charge = applyNoise(pixel.charge);
            }
        }
    }

    /// Apply noise to a vector of charges (in Coulombs)
    void applyNoise(std::vector<double>& charges) {
        if (!m_config.enabled) {
            return;
        }

        for (auto& q : charges) {
            if (q > 0.0) {
                q = applyNoise(q);
            }
        }
    }

private:
    NoiseConfig m_config{};
    std::mt19937 m_generator;
    std::normal_distribution<double> m_gaussDist{0.0, 1.0};
    std::uniform_real_distribution<double> m_uniformDist{0.0, 1.0};
};

} // namespace epic::chargesharing::core

#endif // EPIC_CHARGE_SHARING_CORE_HH
