/// @file test_gaussian_fit.cpp
/// @brief Unit tests for the header-only GaussianFit fitting library.
///
/// Tests fitGaussian1D and fitGaussian2D from core/GaussianFit.hh using
/// synthetic Gaussian data. Requires ROOT (for the fitting backend).

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "GaussianFit.hh"

namespace fit = epic::chargesharing::fit;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_testCount = 0;
static int g_passCount = 0;

static bool approxEqual(double a, double b, double tol = 1e-3) {
    return std::abs(a - b) < tol;
}

#define RUN_TEST(func)                                                                                                 \
    do {                                                                                                               \
        ++g_testCount;                                                                                                 \
        try {                                                                                                          \
            func();                                                                                                    \
            ++g_passCount;                                                                                             \
            std::cout << "  PASS  " << #func << "\n";                                                                  \
        } catch (const std::exception& e) {                                                                            \
            std::cerr << "  FAIL  " << #func << " : " << e.what() << "\n";                                             \
        } catch (...) {                                                                                                \
            std::cerr << "  FAIL  " << #func << " : unknown exception\n";                                              \
        }                                                                                                              \
    } while (false)

#define TEST_ASSERT(cond)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond + " (" + __FILE__ + ":" +               \
                                     std::to_string(__LINE__) + ")");                                                  \
        }                                                                                                              \
    } while (false)

// Generate 1D Gaussian data: A * exp(-0.5 * ((x-mu)/sigma)^2) + B
static void generateGaussian1D(double A, double mu, double sigma, double B, double pixelSpacing, int radius,
                                double center, std::vector<double>& positions, std::vector<double>& charges) {
    positions.clear();
    charges.clear();
    for (int di = -radius; di <= radius; ++di) {
        double x = center + di * pixelSpacing;
        double dx = (x - mu) / sigma;
        double q = A * std::exp(-0.5 * dx * dx) + B;
        positions.push_back(x);
        charges.push_back(q);
    }
}

// Generate 2D Gaussian data: A * exp(-0.5 * ((x-mux)/sx)^2 + ((y-muy)/sy)^2)) + B
static void generateGaussian2D(double A, double muX, double muY, double sigX, double sigY, double B,
                                double pixelSpacing, int radius, double centerX, double centerY,
                                std::vector<double>& xPos, std::vector<double>& yPos, std::vector<double>& charges) {
    xPos.clear();
    yPos.clear();
    charges.clear();
    for (int di = -radius; di <= radius; ++di) {
        for (int dj = -radius; dj <= radius; ++dj) {
            double x = centerX + di * pixelSpacing;
            double y = centerY + dj * pixelSpacing;
            double dx = (x - muX) / sigX;
            double dy = (y - muY) / sigY;
            double q = A * std::exp(-0.5 * (dx * dx + dy * dy)) + B;
            xPos.push_back(x);
            yPos.push_back(y);
            charges.push_back(q);
        }
    }
}

// ===========================================================================
// 1. fitGaussian1D basic convergence tests
// ===========================================================================

void test_fit1D_converges_centered() {
    // Perfect Gaussian centered at origin
    std::vector<double> pos, charges;
    generateGaussian1D(100.0, 0.0, 0.3, 5.0, 0.5, 2, 0.0, pos, charges);

    fit::GaussFit1DConfig cfg{
        .muLo = -1.0, .muHi = 1.0, .sigmaLo = 0.1, .sigmaHi = 2.0, .qMax = 105.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(approxEqual(result.mu, 0.0, 0.05));
    TEST_ASSERT(approxEqual(result.A, 100.0, 5.0));
    TEST_ASSERT(approxEqual(result.sigma, 0.3, 0.1));
    TEST_ASSERT(approxEqual(result.B, 5.0, 2.0));
    TEST_ASSERT(std::isfinite(result.chi2));
    TEST_ASSERT(std::isfinite(result.ndf));
    TEST_ASSERT(std::isfinite(result.muError));
}

void test_fit1D_converges_offset() {
    // Gaussian offset from pixel center
    std::vector<double> pos, charges;
    generateGaussian1D(80.0, 0.15, 0.25, 2.0, 0.5, 3, 0.0, pos, charges);

    fit::GaussFit1DConfig cfg{
        .muLo = -1.5, .muHi = 1.5, .sigmaLo = 0.05, .sigmaHi = 3.0, .qMax = 82.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(approxEqual(result.mu, 0.15, 0.1));
    TEST_ASSERT(result.A > 0.0);
    TEST_ASSERT(result.sigma > 0.0);
}

void test_fit1D_returns_muError() {
    std::vector<double> pos, charges;
    generateGaussian1D(100.0, 0.0, 0.3, 5.0, 0.5, 2, 0.0, pos, charges);

    fit::GaussFit1DConfig cfg{
        .muLo = -1.0, .muHi = 1.0, .sigmaLo = 0.1, .sigmaHi = 2.0, .qMax = 105.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(std::isfinite(result.muError));
    TEST_ASSERT(result.muError >= 0.0);
}

// ===========================================================================
// 2. fitGaussian1D edge cases
// ===========================================================================

void test_fit1D_too_few_points() {
    std::vector<double> pos = {0.0, 0.5};
    std::vector<double> charges = {10.0, 5.0};

    fit::GaussFit1DConfig cfg{
        .muLo = -1.0, .muHi = 1.0, .sigmaLo = 0.1, .sigmaHi = 2.0, .qMax = 10.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(!result.converged);
    TEST_ASSERT(std::isnan(result.mu));
}

void test_fit1D_empty_input() {
    std::vector<double> pos, charges;

    fit::GaussFit1DConfig cfg{
        .muLo = -1.0, .muHi = 1.0, .sigmaLo = 0.1, .sigmaHi = 2.0, .qMax = 10.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(!result.converged);
}

void test_fit1D_mismatched_sizes() {
    std::vector<double> pos = {0.0, 0.5, 1.0};
    std::vector<double> charges = {10.0, 5.0};

    fit::GaussFit1DConfig cfg{
        .muLo = -1.0, .muHi = 1.0, .sigmaLo = 0.1, .sigmaHi = 2.0, .qMax = 10.0, .pixelSpacing = 0.5};

    auto result = fit::fitGaussian1D(pos, charges, cfg);

    TEST_ASSERT(!result.converged);
}

void test_fit1D_default_result_is_invalid() {
    fit::GaussFit1DResult result;

    TEST_ASSERT(!result.converged);
    TEST_ASSERT(std::isnan(result.A));
    TEST_ASSERT(std::isnan(result.mu));
    TEST_ASSERT(std::isnan(result.sigma));
    TEST_ASSERT(std::isnan(result.B));
    TEST_ASSERT(std::isnan(result.chi2));
    TEST_ASSERT(std::isnan(result.muError));
}

// ===========================================================================
// 3. fitGaussian2D basic convergence tests
// ===========================================================================

void test_fit2D_converges_centered() {
    std::vector<double> xPos, yPos, charges;
    generateGaussian2D(100.0, 0.0, 0.0, 0.3, 0.3, 5.0, 0.5, 2, 0.0, 0.0, xPos, yPos, charges);

    fit::GaussFit2DConfig cfg{.muXLo = -1.0,
                              .muXHi = 1.0,
                              .muYLo = -1.0,
                              .muYHi = 1.0,
                              .sigmaLo = 0.1,
                              .sigmaHi = 2.0,
                              .qMax = 105.0,
                              .pixelSpacing = 0.5};

    auto result = fit::fitGaussian2D(xPos, yPos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(approxEqual(result.muX, 0.0, 0.1));
    TEST_ASSERT(approxEqual(result.muY, 0.0, 0.1));
    TEST_ASSERT(approxEqual(result.A, 100.0, 10.0));
    TEST_ASSERT(approxEqual(result.sigmaX, 0.3, 0.15));
    TEST_ASSERT(approxEqual(result.sigmaY, 0.3, 0.15));
    TEST_ASSERT(std::isfinite(result.chi2));
    TEST_ASSERT(std::isfinite(result.ndf));
}

void test_fit2D_converges_offset() {
    std::vector<double> xPos, yPos, charges;
    generateGaussian2D(80.0, 0.1, -0.1, 0.25, 0.35, 3.0, 0.5, 3, 0.0, 0.0, xPos, yPos, charges);

    fit::GaussFit2DConfig cfg{.muXLo = -1.5,
                              .muXHi = 1.5,
                              .muYLo = -1.5,
                              .muYHi = 1.5,
                              .sigmaLo = 0.05,
                              .sigmaHi = 3.0,
                              .qMax = 83.0,
                              .pixelSpacing = 0.5};

    auto result = fit::fitGaussian2D(xPos, yPos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(approxEqual(result.muX, 0.1, 0.1));
    TEST_ASSERT(approxEqual(result.muY, -0.1, 0.1));
}

void test_fit2D_returns_muErrors() {
    std::vector<double> xPos, yPos, charges;
    generateGaussian2D(100.0, 0.0, 0.0, 0.3, 0.3, 5.0, 0.5, 2, 0.0, 0.0, xPos, yPos, charges);

    fit::GaussFit2DConfig cfg{.muXLo = -1.0,
                              .muXHi = 1.0,
                              .muYLo = -1.0,
                              .muYHi = 1.0,
                              .sigmaLo = 0.1,
                              .sigmaHi = 2.0,
                              .qMax = 105.0,
                              .pixelSpacing = 0.5};

    auto result = fit::fitGaussian2D(xPos, yPos, charges, cfg);

    TEST_ASSERT(result.converged);
    TEST_ASSERT(std::isfinite(result.muXError));
    TEST_ASSERT(std::isfinite(result.muYError));
    TEST_ASSERT(result.muXError >= 0.0);
    TEST_ASSERT(result.muYError >= 0.0);
}

// ===========================================================================
// 4. fitGaussian2D edge cases
// ===========================================================================

void test_fit2D_too_few_points() {
    std::vector<double> xPos = {0.0, 0.5, 1.0, 0.0};
    std::vector<double> yPos = {0.0, 0.0, 0.0, 0.5};
    std::vector<double> charges = {10.0, 5.0, 2.0, 7.0};

    fit::GaussFit2DConfig cfg{.muXLo = -1.0,
                              .muXHi = 1.0,
                              .muYLo = -1.0,
                              .muYHi = 1.0,
                              .sigmaLo = 0.1,
                              .sigmaHi = 2.0,
                              .qMax = 10.0,
                              .pixelSpacing = 0.5};

    auto result = fit::fitGaussian2D(xPos, yPos, charges, cfg);

    TEST_ASSERT(!result.converged);
}

void test_fit2D_default_result_is_invalid() {
    fit::GaussFit2DResult result;

    TEST_ASSERT(!result.converged);
    TEST_ASSERT(std::isnan(result.A));
    TEST_ASSERT(std::isnan(result.muX));
    TEST_ASSERT(std::isnan(result.muY));
    TEST_ASSERT(std::isnan(result.sigmaX));
    TEST_ASSERT(std::isnan(result.sigmaY));
    TEST_ASSERT(std::isnan(result.B));
    TEST_ASSERT(std::isnan(result.muXError));
    TEST_ASSERT(std::isnan(result.muYError));
}

// ===========================================================================
// 5. Utility function tests
// ===========================================================================

void test_weightedCentroid_basic() {
    std::vector<double> positions = {-1.0, 0.0, 1.0};
    std::vector<double> charges = {1.0, 10.0, 1.0};
    auto [centroid, ok] = fit::weightedCentroid(positions, charges, 0.0);

    TEST_ASSERT(ok);
    TEST_ASSERT(approxEqual(centroid, 0.0, 1e-6));
}

void test_weightedCentroid_offset() {
    std::vector<double> positions = {-1.0, 0.0, 1.0};
    std::vector<double> charges = {1.0, 5.0, 10.0};
    auto [centroid, ok] = fit::weightedCentroid(positions, charges, 0.0);

    TEST_ASSERT(ok);
    TEST_ASSERT(centroid > 0.0); // Should be pulled toward position 1.0
}

void test_weightedCentroid_with_baseline() {
    std::vector<double> positions = {-1.0, 0.0, 1.0};
    std::vector<double> charges = {5.0, 15.0, 5.0};
    auto [centroid, ok] = fit::weightedCentroid(positions, charges, 5.0);

    TEST_ASSERT(ok);
    // With baseline=5, effective weights are {0, 10, 0}, so centroid = 0.0
    TEST_ASSERT(approxEqual(centroid, 0.0, 1e-6));
}

void test_weightedCentroid_empty() {
    std::vector<double> positions, charges;
    auto [centroid, ok] = fit::weightedCentroid(positions, charges, 0.0);

    TEST_ASSERT(!ok);
    TEST_ASSERT(std::isnan(centroid));
}

void test_estimateSigma_basic() {
    std::vector<double> positions = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<double> charges = {1.0, 5.0, 20.0, 5.0, 1.0};

    double sigma = fit::estimateSigma(positions, charges, 0.0, 0.5, 0.1, 5.0);

    TEST_ASSERT(std::isfinite(sigma));
    TEST_ASSERT(sigma > 0.0);
    TEST_ASSERT(sigma >= 0.1); // At least sigmaLo
    TEST_ASSERT(sigma <= 5.0); // At most sigmaHi
}

void test_estimateSigma_clamped_low() {
    std::vector<double> positions = {0.0};
    std::vector<double> charges = {10.0};

    // Very narrow data, sigma should be clamped to sigmaLo
    double sigma = fit::estimateSigma(positions, charges, 0.0, 0.5, 0.3, 5.0);

    TEST_ASSERT(std::isfinite(sigma));
    TEST_ASSERT(sigma >= 0.3 - 1e-12);
}

// ===========================================================================
// 6. Gaussian function tests
// ===========================================================================

void test_gauss1DPlusB_at_center() {
    double x[1] = {1.0};
    double p[4] = {10.0, 1.0, 0.5, 2.0}; // A=10, mu=1, sigma=0.5, B=2

    double val = fit::gauss1DPlusB(x, p);
    // At x=mu, exp term = 1, so val = A + B = 12
    TEST_ASSERT(approxEqual(val, 12.0, 1e-6));
}

void test_gauss1DPlusB_at_offset() {
    double x[1] = {2.0};
    double p[4] = {10.0, 1.0, 0.5, 2.0}; // A=10, mu=1, sigma=0.5, B=2

    double val = fit::gauss1DPlusB(x, p);
    // At x=2, dx/sigma = (2-1)/0.5 = 2, exp(-0.5*4) = exp(-2) ~ 0.1353
    double expected = 10.0 * std::exp(-2.0) + 2.0;
    TEST_ASSERT(approxEqual(val, expected, 1e-6));
}

void test_gauss2DPlusB_at_center() {
    double xy[2] = {1.0, 2.0};
    double p[6] = {10.0, 1.0, 2.0, 0.5, 0.5, 3.0}; // A=10, muX=1, muY=2, sigX=sigY=0.5, B=3

    double val = fit::gauss2DPlusB(xy, p);
    // At center: exp term = 1, val = A + B = 13
    TEST_ASSERT(approxEqual(val, 13.0, 1e-6));
}

// ===========================================================================
// 7. Distance-weighted error tests
// ===========================================================================

void test_distanceWeightedError_disabled() {
    fit::DistanceWeightedErrorConfig cfg;
    cfg.enabled = false;
    cfg.floorPercent = 5.0;

    double sigma = fit::computeDistanceWeightedError(0.0, 0.0, cfg, 100.0);
    // When disabled, returns floorPercent * 0.01 * maxCharge
    TEST_ASSERT(approxEqual(sigma, 5.0, 1e-6));
}

void test_distanceWeightedError_inverse_model() {
    fit::DistanceWeightedErrorConfig cfg;
    cfg.enabled = true;
    cfg.powerInverse = true;
    cfg.scalePixels = 1.0;
    cfg.exponent = 1.0;
    cfg.floorPercent = 1.0;
    cfg.capPercent = 50.0;
    cfg.pixelSpacing = 0.5;
    cfg.truthCenterX = 0.0;
    cfg.truthCenterY = 0.0;

    double sigmaAtCenter = fit::computeDistanceWeightedError(0.0, 0.0, cfg, 100.0);
    double sigmaFar = fit::computeDistanceWeightedError(1.0, 1.0, cfg, 100.0);

    // Inverse model: smaller sigma (higher weight) at center
    TEST_ASSERT(std::isfinite(sigmaAtCenter));
    TEST_ASSERT(std::isfinite(sigmaFar));
    TEST_ASSERT(sigmaAtCenter >= sigmaFar - 1e-6);
}

void test_applySigmaBounds_clamping() {
    // Within bounds
    TEST_ASSERT(approxEqual(fit::applySigmaBounds(5.0, 100.0, 1.0, 50.0), 5.0, 1e-6));

    // Below floor
    double result = fit::applySigmaBounds(0.5, 100.0, 1.0, 50.0);
    TEST_ASSERT(approxEqual(result, 1.0, 1e-6)); // floor = 1% of 100 = 1.0

    // Above cap
    result = fit::applySigmaBounds(60.0, 100.0, 1.0, 50.0);
    TEST_ASSERT(approxEqual(result, 50.0, 1e-6)); // cap = 50% of 100 = 50.0

    // Invalid input
    result = fit::applySigmaBounds(-1.0, 100.0, 1.0, 50.0);
    TEST_ASSERT(std::isnan(result));
}

// ===========================================================================
// 8. ReconstructionResult tests
// ===========================================================================

void test_reconstructionResult_valid() {
    fit::ReconstructionResult result;
    TEST_ASSERT(!result.valid()); // Default is invalid (NaN positions)

    result.reconX = 1.0;
    result.reconY = 2.0;
    TEST_ASSERT(result.valid());

    result.reconX = std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT(!result.valid());
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
    std::cout << "=== GaussianFit Unit Tests ===\n\n";

    std::cout << "[1D Gaussian fit -- convergence]\n";
    RUN_TEST(test_fit1D_converges_centered);
    RUN_TEST(test_fit1D_converges_offset);
    RUN_TEST(test_fit1D_returns_muError);

    std::cout << "\n[1D Gaussian fit -- edge cases]\n";
    RUN_TEST(test_fit1D_too_few_points);
    RUN_TEST(test_fit1D_empty_input);
    RUN_TEST(test_fit1D_mismatched_sizes);
    RUN_TEST(test_fit1D_default_result_is_invalid);

    std::cout << "\n[2D Gaussian fit -- convergence]\n";
    RUN_TEST(test_fit2D_converges_centered);
    RUN_TEST(test_fit2D_converges_offset);
    RUN_TEST(test_fit2D_returns_muErrors);

    std::cout << "\n[2D Gaussian fit -- edge cases]\n";
    RUN_TEST(test_fit2D_too_few_points);
    RUN_TEST(test_fit2D_default_result_is_invalid);

    std::cout << "\n[Utility functions]\n";
    RUN_TEST(test_weightedCentroid_basic);
    RUN_TEST(test_weightedCentroid_offset);
    RUN_TEST(test_weightedCentroid_with_baseline);
    RUN_TEST(test_weightedCentroid_empty);
    RUN_TEST(test_estimateSigma_basic);
    RUN_TEST(test_estimateSigma_clamped_low);

    std::cout << "\n[Gaussian functions]\n";
    RUN_TEST(test_gauss1DPlusB_at_center);
    RUN_TEST(test_gauss1DPlusB_at_offset);
    RUN_TEST(test_gauss2DPlusB_at_center);

    std::cout << "\n[Distance-weighted errors]\n";
    RUN_TEST(test_distanceWeightedError_disabled);
    RUN_TEST(test_distanceWeightedError_inverse_model);
    RUN_TEST(test_applySigmaBounds_clamping);

    std::cout << "\n[Reconstruction result]\n";
    RUN_TEST(test_reconstructionResult_valid);

    std::cout << "\n=== Results: " << g_passCount << " / " << g_testCount << " passed ===\n";

    if (g_passCount == g_testCount) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << (g_testCount - g_passCount) << " test(s) FAILED.\n";
        return 1;
    }
}
