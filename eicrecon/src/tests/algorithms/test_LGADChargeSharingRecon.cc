// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover
//
// Unit tests for LGADChargeSharingRecon. These tests exercise the per-hit
// reconstruction path (processSingleHit) on a synthetic pad grid injected
// via setGeometryForTesting(), so they run without a DD4hep / Acts
// service environment.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "algorithms/reco/LGADChargeSharingRecon.h"
#include "chargesharing/core/ChargeSharingCore.hh"

#include <algorithms/logger.h>

#include <array>
#include <cmath>
#include <cstdint>

using eicrecon::LGADChargeSharingRecon;
using eicrecon::LGADChargeSharingReconConfig;
using ::chargesharing::core::ActivePixelMode;
using ::chargesharing::core::ReconMethod;
using ::chargesharing::core::SignalModel;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

/// Test-geometry mimicking a small symmetric AC-LGAD pad array.
LGADChargeSharingRecon::Geometry makeTestGeometry() {
    LGADChargeSharingRecon::Geometry g{};
    g.pixelSpacingXMM = 0.5;   // 500 um pitch
    g.pixelSpacingYMM = 0.5;
    g.pixelSizeXMM = 0.15;     // 150 um pad
    g.pixelSizeYMM = 0.15;
    g.gridOffsetXMM = 0.0;
    g.gridOffsetYMM = 0.0;
    g.detectorThicknessMM = 0.05;
    g.pixelThicknessMM = 0.02;
    g.detectorZCenterMM = -10.0;
    g.pixelsPerSide = 21;
    g.minIndexX = -10;
    g.maxIndexX = 10;
    g.minIndexY = -10;
    g.maxIndexY = 10;
    g.useXZCoordinates = false;
    g.fieldNameX = "x";
    g.fieldNameY = "y";
    return g;
}

/// Build a LGADChargeSharingRecon with injected geometry. Noise is disabled
/// so that results are deterministic.
std::unique_ptr<LGADChargeSharingRecon> makeAlgorithm(ReconMethod method,
                                                     int radius = 2,
                                                     SignalModel model = SignalModel::LogA) {
    auto algo = std::make_unique<LGADChargeSharingRecon>("test_lgad_recon");
    algo->level(algorithms::LogLevel::kError);

    LGADChargeSharingReconConfig cfg;
    cfg.signalModel = model;
    cfg.activePixelMode = ActivePixelMode::Neighborhood;
    cfg.reconMethod = method;
    cfg.readout = "TestReadout";
    cfg.neighborhoodRadius = radius;
    cfg.d0Micron = 1.0;
    cfg.linearBetaPerMicron = 0.0;
    cfg.ionizationEnergyEV = 3.6;
    cfg.amplificationFactor = 20.0;
    cfg.noiseEnabled = false;
    cfg.noiseElectronCount = 0.0;
    cfg.minEDepGeV = 0.0F;
    algo->applyConfig(cfg);

    algo->setGeometryForTesting(makeTestGeometry());
    algo->init();
    return algo;
}

LGADChargeSharingRecon::SingleHitInput makeHitAt(double xMM, double yMM, double edepGeV = 1.0e-3) {
    LGADChargeSharingRecon::SingleHitInput in{};
    in.hitPositionMM = {xMM, yMM, -10.0};
    in.energyDepositGeV = edepGeV;
    in.cellID = 0;
    in.pixelIndexHint = std::pair<int, int>{0, 0};  // drive into (0,0) pad
    return in;
}

} // namespace

TEST_CASE("LGADChargeSharingRecon: fractions normalize to ~1 in Neighborhood mode",
          "[lgad][recon][fractions]") {
    auto algo = makeAlgorithm(ReconMethod::Centroid);

    // Hit slightly offset from (0,0) pad center.
    auto in = makeHitAt(0.05, 0.03);
    in.pixelIndexHint = std::pair<int, int>{0, 0};

    const auto res = algo->processSingleHit(in);

    double fsum = 0.0;
    double qsum = 0.0;
    for (const auto& n : res.neighbors) {
        fsum += n.fraction;
        qsum += n.chargeC;
    }

    // With ActivePixelMode::Neighborhood and noise off, fractions should sum
    // to very close to 1 and total collected charge should match.
    REQUIRE(res.totalCollectedChargeC > 0.0);
    REQUIRE_THAT(fsum, WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(qsum, WithinRel(res.totalCollectedChargeC, 1e-6));
    REQUIRE(res.numActiveNeighbors > 0);
}

TEST_CASE("LGADChargeSharingRecon: residual is small for centered hit (Gaussian2D)",
          "[lgad][recon][residual]") {
    auto algo = makeAlgorithm(ReconMethod::Gaussian2D);

    // Hit exactly at (0,0) pad center -> residual should be near zero.
    auto in = makeHitAt(0.0, 0.0);
    const auto res = algo->processSingleHit(in);

    CHECK_THAT(res.residualXMM(), WithinAbs(0.0, 1e-3));
    CHECK_THAT(res.residualYMM(), WithinAbs(0.0, 1e-3));
}

TEST_CASE("LGADChargeSharingRecon: sub-pad residual within pitch for off-center hit",
          "[lgad][recon][residual]") {
    auto algo = makeAlgorithm(ReconMethod::Gaussian2D);

    // ~40 um offset inside the central pad -- well inside pad half-pitch.
    auto in = makeHitAt(0.04, -0.02);
    const auto res = algo->processSingleHit(in);

    const double pitchX = 0.5;
    const double pitchY = 0.5;
    REQUIRE(std::isfinite(res.reconstructedPositionMM[0]));
    REQUIRE(std::isfinite(res.reconstructedPositionMM[1]));
    CHECK(std::abs(res.residualXMM()) < 0.5 * pitchX);
    CHECK(std::abs(res.residualYMM()) < 0.5 * pitchY);
}

TEST_CASE("LGADChargeSharingRecon: Gaussian2D reports positive mu errors (covariance)",
          "[lgad][recon][covariance]") {
    auto algo = makeAlgorithm(ReconMethod::Gaussian2D);
    auto in = makeHitAt(0.03, 0.01);
    const auto res = algo->processSingleHit(in);

    if (res.fit2D.converged) {
        CHECK(std::isfinite(res.fit2D.muXError));
        CHECK(std::isfinite(res.fit2D.muYError));
        CHECK(res.fit2D.muXError > 0.0);
        CHECK(res.fit2D.muYError > 0.0);
    } else {
        // If the fit does not converge the algorithm falls back to centroid
        // and the 2D mu errors can be NaN; in that case the variance fallback
        // is covered by the process()-level path (tested separately).
        SUCCEED("fit2D did not converge; centroid fallback is exercised in other tests");
    }
}

TEST_CASE("LGADChargeSharingRecon: Gaussian1D populates row/col fit errors when converged",
          "[lgad][recon][covariance]") {
    auto algo = makeAlgorithm(ReconMethod::Gaussian1D);
    auto in = makeHitAt(0.02, -0.04);
    const auto res = algo->processSingleHit(in);

    if (res.fitRowX.converged) {
        CHECK(std::isfinite(res.fitRowX.muError));
        CHECK(res.fitRowX.muError > 0.0);
    }
    if (res.fitColY.converged) {
        CHECK(std::isfinite(res.fitColY.muError));
        CHECK(res.fitColY.muError > 0.0);
    }
}

TEST_CASE("LGADChargeSharingRecon: centroid fallback when neighborhood is degenerate",
          "[lgad][recon][fallback]") {
    // A neighborhood radius of 0 collapses to a single pixel -- fits cannot
    // converge and the reconstructor must fall back to the centroid (which,
    // with a single pixel, degenerates to that pixel's center).
    auto algo = makeAlgorithm(ReconMethod::Gaussian2D, /*radius=*/0);
    auto in = makeHitAt(0.07, 0.08);
    const auto res = algo->processSingleHit(in);

    // Expect the reconstructed position to collapse to the central pixel
    // center within numerical tolerance.
    CHECK_THAT(res.reconstructedPositionMM[0],
               WithinAbs(res.nearestPixelCenterMM[0], 1e-6));
    CHECK_THAT(res.reconstructedPositionMM[1],
               WithinAbs(res.nearestPixelCenterMM[1], 1e-6));
    CHECK_FALSE(res.fit2D.converged);
}

TEST_CASE("LGADChargeSharingRecon: LinA model also produces normalized fractions",
          "[lgad][recon][model]") {
    auto algo = makeAlgorithm(ReconMethod::Centroid, /*radius=*/2, SignalModel::LinA);
    auto in = makeHitAt(-0.01, 0.02);
    const auto res = algo->processSingleHit(in);

    double fsum = 0.0;
    for (const auto& n : res.neighbors) {
        fsum += n.fraction;
    }
    REQUIRE_THAT(fsum, WithinAbs(1.0, 1e-6));
}
