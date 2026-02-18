/// @file test_charge_sharing_core.cpp
/// @brief Unit tests for the header-only ChargeSharingCore library.
///
/// Uses plain assert() and a main() that returns 0 on success, 1 on failure.
/// No external test framework required -- only STL and ChargeSharingCore.hh.

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "ChargeSharingCore.hh"

namespace core = epic::chargesharing::core;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_testCount = 0;
static int g_passCount = 0;

/// Floating-point approximate equality.
static bool approxEqual(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

/// Run a single named test.  Prints PASS / FAIL and updates counters.
#define RUN_TEST(func)                                                    \
    do {                                                                  \
        ++g_testCount;                                                    \
        try {                                                             \
            func();                                                       \
            ++g_passCount;                                                \
            std::cout << "  PASS  " << #func << "\n";                     \
        } catch (const std::exception& e) {                               \
            std::cerr << "  FAIL  " << #func << " : " << e.what() << "\n";\
        } catch (...) {                                                   \
            std::cerr << "  FAIL  " << #func << " : unknown exception\n"; \
        }                                                                 \
    } while (false)

/// Assertion that throws on failure (so RUN_TEST can catch it).
#define TEST_ASSERT(cond)                                                 \
    do {                                                                  \
        if (!(cond)) {                                                    \
            throw std::runtime_error(                                     \
                std::string("Assertion failed: ") + #cond +               \
                " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");  \
        }                                                                 \
    } while (false)

// ===========================================================================
// 1. calcDistanceToCenter tests
// ===========================================================================

void test_distanceToCenter_zero() {
    double d = core::calcDistanceToCenter(0.0, 0.0);
    TEST_ASSERT(approxEqual(d, 0.0));
}

void test_distanceToCenter_known() {
    // 3-4-5 triangle
    double d = core::calcDistanceToCenter(3.0, 4.0);
    TEST_ASSERT(approxEqual(d, 5.0));
}

void test_distanceToCenter_symmetric() {
    double d1 = core::calcDistanceToCenter(1.0, 2.0);
    double d2 = core::calcDistanceToCenter(2.0, 1.0);
    TEST_ASSERT(approxEqual(d1, d2));
}

// ===========================================================================
// 2. calcPadViewAngle tests
// ===========================================================================

void test_padViewAngle_atZeroDistance() {
    // At distance=0 the function returns atan(1) = pi/4
    double angle = core::calcPadViewAngle(0.0, 0.15, 0.15);
    TEST_ASSERT(approxEqual(angle, std::atan(1.0)));
    TEST_ASSERT(approxEqual(angle, M_PI / 4.0));
}

void test_padViewAngle_decreasesWithDistance() {
    const double padW = 0.15;
    const double padH = 0.15;
    double prev = core::calcPadViewAngle(0.0, padW, padH);
    for (double d = 0.1; d <= 5.0; d += 0.1) {
        double curr = core::calcPadViewAngle(d, padW, padH);
        TEST_ASSERT(curr < prev);
        prev = curr;
    }
}

void test_padViewAngle_approachesZeroAtLargeDistance() {
    double angle = core::calcPadViewAngle(1000.0, 0.15, 0.15);
    TEST_ASSERT(angle > 0.0);
    TEST_ASSERT(angle < 0.001);  // Very small angle at large distance
}

void test_padViewAngle_symmetricInWidthHeight() {
    double a1 = core::calcPadViewAngle(1.0, 0.2, 0.3);
    double a2 = core::calcPadViewAngle(1.0, 0.3, 0.2);
    // The formula uses (w+h)/2, so swapping w,h gives same result
    TEST_ASSERT(approxEqual(a1, a2));
}

void test_padViewAngle_positiveForPositiveInputs() {
    double angle = core::calcPadViewAngle(0.5, 0.15, 0.15);
    TEST_ASSERT(angle > 0.0);
    TEST_ASSERT(std::isfinite(angle));
}

// ===========================================================================
// 3. calcWeightLogA tests
// ===========================================================================

void test_weightLogA_decreasesWithDistance() {
    const double padW = 0.15, padH = 0.15;
    const double d0 = 0.001;  // 1 um in mm
    double prevW = std::numeric_limits<double>::max();
    for (double d = 0.01; d <= 3.0; d += 0.05) {
        double alpha = core::calcPadViewAngle(d, padW, padH);
        double w = core::calcWeightLogA(d, alpha, d0);
        TEST_ASSERT(w >= 0.0);
        TEST_ASSERT(w < prevW);
        prevW = w;
    }
}

void test_weightLogA_nonNegative() {
    double alpha = core::calcPadViewAngle(2.0, 0.15, 0.15);
    double w = core::calcWeightLogA(2.0, alpha, 0.001);
    TEST_ASSERT(w >= 0.0);
    TEST_ASSERT(std::isfinite(w));
}

void test_weightLogA_guardFactor() {
    // At distance exactly d0, the guard factor kicks in.
    // Should produce a finite non-negative result.
    const double d0 = 0.001;
    double alpha = core::calcPadViewAngle(d0, 0.15, 0.15);
    double w = core::calcWeightLogA(d0, alpha, d0);
    TEST_ASSERT(std::isfinite(w));
    TEST_ASSERT(w >= 0.0);
}

void test_weightLogA_largeDistanceSmallWeight() {
    double alpha = core::calcPadViewAngle(100.0, 0.15, 0.15);
    double w = core::calcWeightLogA(100.0, alpha, 0.001);
    TEST_ASSERT(w >= 0.0);
    TEST_ASSERT(w < 0.01);  // Should be very small
}

// ===========================================================================
// 4. calcWeightLinA tests
// ===========================================================================

void test_weightLinA_decreasesWithDistance() {
    const double beta = core::constants::kLinearBetaNarrow;
    const double padW = 0.15, padH = 0.15;
    double prevW = std::numeric_limits<double>::max();
    for (double d = 0.0; d <= 0.3; d += 0.01) {
        double alpha = core::calcPadViewAngle(d, padW, padH);
        double w = core::calcWeightLinA(d, alpha, beta);
        TEST_ASSERT(w >= 0.0);
        TEST_ASSERT(w <= prevW + 1e-12);  // Allow tiny float tolerance
        prevW = w;
    }
}

void test_weightLinA_zeroAtLargeDistance() {
    // 1/beta (in um) = 1/0.003 = 333.3 um = 0.3333 mm
    // Beyond that, weight should be 0
    const double beta = core::constants::kLinearBetaNarrow;
    double alpha = core::calcPadViewAngle(1.0, 0.15, 0.15);
    double w = core::calcWeightLinA(1.0, alpha, beta);
    TEST_ASSERT(approxEqual(w, 0.0));
}

void test_weightLinA_nonNegative() {
    const double beta = core::constants::kLinearBetaWide;
    for (double d = 0.0; d <= 5.0; d += 0.1) {
        double alpha = core::calcPadViewAngle(d, 0.15, 0.15);
        double w = core::calcWeightLinA(d, alpha, beta);
        TEST_ASSERT(w >= 0.0);
        TEST_ASSERT(std::isfinite(w));
    }
}

void test_weightLinA_atZeroDistance() {
    double alpha = core::calcPadViewAngle(0.0, 0.15, 0.15);
    double w = core::calcWeightLinA(0.0, alpha, core::constants::kLinearBetaNarrow);
    // At d=0: attenuation = max(0, 1 - 0) = 1, so weight = alpha
    TEST_ASSERT(approxEqual(w, alpha));
}

// ===========================================================================
// 5. calcWeight dispatcher tests
// ===========================================================================

void test_calcWeight_logA_matchesDirect() {
    const double d = 0.5, padW = 0.15, padH = 0.15, d0 = 0.001;
    double alpha = core::calcPadViewAngle(d, padW, padH);
    double wDirect = core::calcWeightLogA(d, alpha, d0);
    double wDispatch = core::calcWeight(core::SignalModel::LogA, d, padW, padH, d0);
    TEST_ASSERT(approxEqual(wDirect, wDispatch));
}

void test_calcWeight_linA_matchesDirect() {
    const double d = 0.1, padW = 0.15, padH = 0.15;
    const double beta = core::constants::kLinearBetaNarrow;
    double alpha = core::calcPadViewAngle(d, padW, padH);
    double wDirect = core::calcWeightLinA(d, alpha, beta);
    double wDispatch = core::calcWeight(core::SignalModel::LinA, d, padW, padH,
                                        0.001, beta);
    TEST_ASSERT(approxEqual(wDirect, wDispatch));
}

// ===========================================================================
// 6. getLinearBeta tests
// ===========================================================================

void test_getLinearBeta_narrowPitch() {
    // 150 um => 0.15 mm, should get narrow beta
    double beta = core::getLinearBeta(0.15);
    TEST_ASSERT(approxEqual(beta, core::constants::kLinearBetaNarrow));
}

void test_getLinearBeta_widePitch() {
    // 300 um => 0.30 mm, should get wide beta
    double beta = core::getLinearBeta(0.30);
    TEST_ASSERT(approxEqual(beta, core::constants::kLinearBetaWide));
}

void test_getLinearBeta_boundaryPitch() {
    // 200 um => 0.20 mm, at boundary, should get narrow beta
    double beta = core::getLinearBeta(0.20);
    TEST_ASSERT(approxEqual(beta, core::constants::kLinearBetaNarrow));
}

// ===========================================================================
// 7. Neighborhood calculation + charge conservation tests
// ===========================================================================

void test_neighborhood_fractionsSumToOne_logA() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.activeMode = core::ActivePixelMode::Neighborhood;
    cfg.radius = 2;  // 5x5
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;
    cfg.numPixelsX = 0;  // unbounded
    cfg.numPixelsY = 0;

    // Hit exactly at center pixel
    double hitX = 0.0, hitY = 0.0;
    auto result = core::calculateNeighborhood(hitX, hitY, 5, 5, 0.0, 0.0, cfg);

    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) {
            fracSum += p.fraction;
        }
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

void test_neighborhood_fractionsSumToOne_linA() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LinA;
    cfg.activeMode = core::ActivePixelMode::Neighborhood;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;
    cfg.betaPerMicron = core::constants::kLinearBetaNarrow;
    cfg.numPixelsX = 0;
    cfg.numPixelsY = 0;

    double hitX = 0.1, hitY = -0.05;  // Offset from center
    auto result = core::calculateNeighborhood(hitX, hitY, 5, 5, 0.0, 0.0, cfg);

    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) {
            fracSum += p.fraction;
        }
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

void test_neighborhood_centerPixelHighestFraction() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.activeMode = core::ActivePixelMode::Neighborhood;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    // Hit exactly at center pixel
    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    const core::NeighborPixel* center = result.getPixel(0, 0);
    TEST_ASSERT(center != nullptr);

    // Center pixel should have the highest fraction when hit is at its center
    for (const auto& p : result.pixels) {
        if (p.inBounds) {
            TEST_ASSERT(center->fraction >= p.fraction - 1e-12);
        }
    }
}

void test_neighborhood_gridSize() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 3;  // 7x7
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 10, 10, 0.0, 0.0, cfg);
    int gridDim = 2 * cfg.radius + 1;
    TEST_ASSERT(static_cast<int>(result.pixels.size()) == gridDim * gridDim);
}

void test_neighborhood_allInBoundsUnbounded() {
    // With unbounded detector (numPixels=0), all pixels should be in bounds
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;
    cfg.numPixelsX = 0;
    cfg.numPixelsY = 0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);
    for (const auto& p : result.pixels) {
        TEST_ASSERT(p.inBounds);
    }
}

void test_neighborhood_boundsCheckClipsPixels() {
    // Small detector: only 3x3 pixels, center at (1,1), radius=2
    // So neighborhood goes from (-1,-1) to (3,3) relative, but global (0,0)-(2,2) is valid
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;
    cfg.numPixelsX = 3;
    cfg.numPixelsY = 3;

    auto result = core::calculateNeighborhood(0.5, 0.5, 1, 1, 0.5, 0.5, cfg);

    int outOfBounds = 0;
    int inBounds = 0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) ++inBounds;
        else ++outOfBounds;
    }
    // 5x5 grid = 25 total, 3x3 in-bounds = 9
    TEST_ASSERT(inBounds == 9);
    TEST_ASSERT(outOfBounds == 16);

    // Fractions should still sum to 1 for in-bounds pixels
    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) fracSum += p.fraction;
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

void test_neighborhood_symmetry() {
    // With hit at center pixel, fractions should be symmetric under 90-degree rotation
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.activeMode = core::ActivePixelMode::Neighborhood;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSizeYMM = 0.15;    // same as X for square pixels
    cfg.pixelSpacingMM = 0.5;
    cfg.pixelSpacingYMM = 0.5;  // same as X for square spacing
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    // pixel(di, dj) should equal pixel(dj, di) under 90-degree rotation
    for (int di = -2; di <= 2; ++di) {
        for (int dj = -2; dj <= 2; ++dj) {
            const auto* pA = result.getPixel(di, dj);
            const auto* pB = result.getPixel(dj, di);
            TEST_ASSERT(pA != nullptr && pB != nullptr);
            TEST_ASSERT(approxEqual(pA->fraction, pB->fraction, 1e-12));
        }
    }
}

// ===========================================================================
// 8. ActivePixelMode tests
// ===========================================================================

void test_activeMode_rowCol() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.activeMode = core::ActivePixelMode::RowCol;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    // In RowCol mode only center row (dj=0) and center col (di=0) get fractions
    // Pixels NOT on the cross should have fraction = 0
    for (const auto& p : result.pixels) {
        if (p.inBounds && p.di != 0 && p.dj != 0) {
            TEST_ASSERT(approxEqual(p.fraction, 0.0));
        }
    }

    // Active fractions should still sum to 1
    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        fracSum += p.fraction;
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

void test_activeMode_chargeBlock2x2() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.activeMode = core::ActivePixelMode::ChargeBlock2x2;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    // Exactly 4 pixels should have non-zero fraction
    int active = 0;
    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.fraction > 0.0) {
            ++active;
            fracSum += p.fraction;
        }
    }
    TEST_ASSERT(active == 4);
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

// ===========================================================================
// 9. NeighborhoodResult accessor tests
// ===========================================================================

void test_neighborhoodResult_getPixel() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 1;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    TEST_ASSERT(result.getPixel(0, 0) != nullptr);
    TEST_ASSERT(result.getPixel(1, 1) != nullptr);
    TEST_ASSERT(result.getPixel(-1, -1) != nullptr);
    // Beyond radius should not be found
    TEST_ASSERT(result.getPixel(2, 0) == nullptr);
}

void test_neighborhoodResult_getCenterRow() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);
    auto row = result.getCenterRow();
    // Center row (dj=0): di from -2 to +2 = 5 pixels
    TEST_ASSERT(row.size() == 5);
    // Should be sorted by di
    for (size_t i = 1; i < row.size(); ++i) {
        TEST_ASSERT(row[i]->di > row[i - 1]->di);
    }
}

void test_neighborhoodResult_getCenterCol() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);
    auto col = result.getCenterCol();
    // Center column (di=0): dj from -2 to +2 = 5 pixels
    TEST_ASSERT(col.size() == 5);
    for (size_t i = 1; i < col.size(); ++i) {
        TEST_ASSERT(col[i]->dj > col[i - 1]->dj);
    }
}

// ===========================================================================
// 10. NoiseModel tests
// ===========================================================================

void test_noiseModel_disabledPassesThrough() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = false;
    noise.setConfig(ncfg);

    double original = 1.5e-15;
    double result = noise.applyNoise(original);
    TEST_ASSERT(approxEqual(result, original));
}

void test_noiseModel_enabledChangesValue() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    ncfg.gainSigmaMin = 0.01;
    ncfg.gainSigmaMax = 0.05;
    ncfg.electronNoiseCount = 500.0;
    noise.setConfig(ncfg);

    double original = 1.5e-15;

    // Run many trials -- at least some should differ
    int changed = 0;
    for (int i = 0; i < 100; ++i) {
        noise.setSeed(static_cast<unsigned>(i));
        double result = noise.applyNoise(original);
        if (!approxEqual(result, original, 1e-20)) {
            ++changed;
        }
    }
    TEST_ASSERT(changed > 50);  // Overwhelmingly likely
}

void test_noiseModel_outputNonNegative() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    ncfg.gainSigmaMin = 0.1;
    ncfg.gainSigmaMax = 0.5;
    ncfg.electronNoiseCount = 5000.0;
    noise.setConfig(ncfg);

    // Even with aggressive noise, output should never be negative
    for (int i = 0; i < 1000; ++i) {
        noise.setSeed(static_cast<unsigned>(i));
        double result = noise.applyNoise(1.0e-16);
        TEST_ASSERT(result >= 0.0);
    }
}

void test_noiseModel_zeroChargePassesThrough() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    ncfg.electronNoiseCount = 1000.0;
    noise.setConfig(ncfg);

    // Zero charge should pass through without noise
    double result = noise.applyNoise(0.0);
    TEST_ASSERT(approxEqual(result, 0.0));
}

void test_noiseModel_negativeChargePassesThrough() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    noise.setConfig(ncfg);

    // Negative charge should pass through without noise
    double result = noise.applyNoise(-1.0e-15);
    TEST_ASSERT(approxEqual(result, -1.0e-15));
}

void test_noiseModel_vectorOverload() {
    core::NoiseModel noise(42);
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    ncfg.gainSigmaMin = 0.02;
    ncfg.gainSigmaMax = 0.05;
    ncfg.electronNoiseCount = 500.0;
    noise.setConfig(ncfg);

    std::vector<double> charges = {1e-15, 2e-15, 3e-15, 0.0, -1e-15};
    std::vector<double> original = charges;
    noise.applyNoise(charges);

    // Zero and negative entries should be unchanged
    TEST_ASSERT(approxEqual(charges[3], 0.0));
    TEST_ASSERT(approxEqual(charges[4], -1e-15));
    // Positive entries should be non-negative
    TEST_ASSERT(charges[0] >= 0.0);
    TEST_ASSERT(charges[1] >= 0.0);
    TEST_ASSERT(charges[2] >= 0.0);
}

void test_noiseModel_seedReproducibility() {
    core::NoiseConfig ncfg;
    ncfg.enabled = true;
    ncfg.gainSigmaMin = 0.02;
    ncfg.gainSigmaMax = 0.05;
    ncfg.electronNoiseCount = 500.0;

    core::NoiseModel noise1(12345);
    noise1.setConfig(ncfg);
    double r1 = noise1.applyNoise(1e-15);

    core::NoiseModel noise2(12345);
    noise2.setConfig(ncfg);
    double r2 = noise2.applyNoise(1e-15);

    TEST_ASSERT(approxEqual(r1, r2, 1e-30));  // Should be bit-identical
}

// ===========================================================================
// 11. Utility function tests
// ===========================================================================

void test_utility_nan() {
    double v = core::nan();
    TEST_ASSERT(std::isnan(v));
}

void test_utility_isFinite() {
    TEST_ASSERT(core::isFinite(1.0));
    TEST_ASSERT(core::isFinite(0.0));
    TEST_ASSERT(!core::isFinite(core::nan()));
    TEST_ASSERT(!core::isFinite(std::numeric_limits<double>::infinity()));
}

void test_utility_clamp() {
    TEST_ASSERT(core::clamp(5.0, 0.0, 10.0) == 5.0);
    TEST_ASSERT(core::clamp(-1.0, 0.0, 10.0) == 0.0);
    TEST_ASSERT(core::clamp(15.0, 0.0, 10.0) == 10.0);
    TEST_ASSERT(core::clamp(0, 0, 10) == 0);
    TEST_ASSERT(core::clamp(10, 0, 10) == 10);
}

// ===========================================================================
// 12. Edge cases
// ===========================================================================

void test_edge_zeroRadius() {
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 0;  // 1x1 neighborhood = center pixel only
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);
    TEST_ASSERT(result.pixels.size() == 1);
    TEST_ASSERT(result.pixels[0].inBounds);
    TEST_ASSERT(approxEqual(result.pixels[0].fraction, 1.0, 1e-9));
}

void test_edge_asymmetricPixels() {
    // Non-square pixels and spacing
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 1;
    cfg.pixelSizeMM = 0.10;
    cfg.pixelSizeYMM = 0.20;
    cfg.pixelSpacingMM = 0.4;
    cfg.pixelSpacingYMM = 0.8;
    cfg.d0Micron = 1.0;

    auto result = core::calculateNeighborhood(0.0, 0.0, 5, 5, 0.0, 0.0, cfg);

    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) {
            TEST_ASSERT(p.fraction >= 0.0);
            fracSum += p.fraction;
        }
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

void test_edge_hitFarFromCenter() {
    // Hit position far from center pixel -- fractions should still sum to 1
    core::NeighborhoodConfig cfg;
    cfg.signalModel = core::SignalModel::LogA;
    cfg.radius = 2;
    cfg.pixelSizeMM = 0.15;
    cfg.pixelSpacingMM = 0.5;
    cfg.d0Micron = 1.0;

    // Hit offset by half a pixel pitch in both dimensions
    auto result = core::calculateNeighborhood(0.25, 0.25, 5, 5, 0.0, 0.0, cfg);

    double fracSum = 0.0;
    for (const auto& p : result.pixels) {
        if (p.inBounds) {
            TEST_ASSERT(p.fraction >= 0.0);
            fracSum += p.fraction;
        }
    }
    TEST_ASSERT(approxEqual(fracSum, 1.0, 1e-9));
}

// ===========================================================================
// 13. Constants sanity checks
// ===========================================================================

void test_constants_values() {
    TEST_ASSERT(approxEqual(core::constants::kMillimeterPerMicron, 1.0e-3));
    TEST_ASSERT(approxEqual(core::constants::kMicronPerMillimeter, 1.0e3));
    TEST_ASSERT(core::constants::kGuardFactor > 1.0);
    TEST_ASSERT(core::constants::kMinD0Micron > 0.0);
    TEST_ASSERT(core::constants::kOutOfBoundsFraction < 0.0);
    TEST_ASSERT(core::constants::kLinearBetaNarrow > core::constants::kLinearBetaWide);
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
    std::cout << "=== ChargeSharingCore Unit Tests ===\n\n";

    std::cout << "[Distance to center]\n";
    RUN_TEST(test_distanceToCenter_zero);
    RUN_TEST(test_distanceToCenter_known);
    RUN_TEST(test_distanceToCenter_symmetric);

    std::cout << "\n[Pad view angle]\n";
    RUN_TEST(test_padViewAngle_atZeroDistance);
    RUN_TEST(test_padViewAngle_decreasesWithDistance);
    RUN_TEST(test_padViewAngle_approachesZeroAtLargeDistance);
    RUN_TEST(test_padViewAngle_symmetricInWidthHeight);
    RUN_TEST(test_padViewAngle_positiveForPositiveInputs);

    std::cout << "\n[Weight LogA]\n";
    RUN_TEST(test_weightLogA_decreasesWithDistance);
    RUN_TEST(test_weightLogA_nonNegative);
    RUN_TEST(test_weightLogA_guardFactor);
    RUN_TEST(test_weightLogA_largeDistanceSmallWeight);

    std::cout << "\n[Weight LinA]\n";
    RUN_TEST(test_weightLinA_decreasesWithDistance);
    RUN_TEST(test_weightLinA_zeroAtLargeDistance);
    RUN_TEST(test_weightLinA_nonNegative);
    RUN_TEST(test_weightLinA_atZeroDistance);

    std::cout << "\n[Weight dispatcher]\n";
    RUN_TEST(test_calcWeight_logA_matchesDirect);
    RUN_TEST(test_calcWeight_linA_matchesDirect);

    std::cout << "\n[Linear beta]\n";
    RUN_TEST(test_getLinearBeta_narrowPitch);
    RUN_TEST(test_getLinearBeta_widePitch);
    RUN_TEST(test_getLinearBeta_boundaryPitch);

    std::cout << "\n[Neighborhood -- charge conservation]\n";
    RUN_TEST(test_neighborhood_fractionsSumToOne_logA);
    RUN_TEST(test_neighborhood_fractionsSumToOne_linA);
    RUN_TEST(test_neighborhood_centerPixelHighestFraction);
    RUN_TEST(test_neighborhood_gridSize);
    RUN_TEST(test_neighborhood_allInBoundsUnbounded);
    RUN_TEST(test_neighborhood_boundsCheckClipsPixels);
    RUN_TEST(test_neighborhood_symmetry);

    std::cout << "\n[Active pixel modes]\n";
    RUN_TEST(test_activeMode_rowCol);
    RUN_TEST(test_activeMode_chargeBlock2x2);

    std::cout << "\n[Neighborhood result accessors]\n";
    RUN_TEST(test_neighborhoodResult_getPixel);
    RUN_TEST(test_neighborhoodResult_getCenterRow);
    RUN_TEST(test_neighborhoodResult_getCenterCol);

    std::cout << "\n[Noise model]\n";
    RUN_TEST(test_noiseModel_disabledPassesThrough);
    RUN_TEST(test_noiseModel_enabledChangesValue);
    RUN_TEST(test_noiseModel_outputNonNegative);
    RUN_TEST(test_noiseModel_zeroChargePassesThrough);
    RUN_TEST(test_noiseModel_negativeChargePassesThrough);
    RUN_TEST(test_noiseModel_vectorOverload);
    RUN_TEST(test_noiseModel_seedReproducibility);

    std::cout << "\n[Utilities]\n";
    RUN_TEST(test_utility_nan);
    RUN_TEST(test_utility_isFinite);
    RUN_TEST(test_utility_clamp);

    std::cout << "\n[Edge cases]\n";
    RUN_TEST(test_edge_zeroRadius);
    RUN_TEST(test_edge_asymmetricPixels);
    RUN_TEST(test_edge_hitFarFromCenter);

    std::cout << "\n[Constants]\n";
    RUN_TEST(test_constants_values);

    std::cout << "\n=== Results: " << g_passCount << " / " << g_testCount
              << " passed ===\n";

    if (g_passCount == g_testCount) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << (g_testCount - g_passCount) << " test(s) FAILED.\n";
        return 1;
    }
}
