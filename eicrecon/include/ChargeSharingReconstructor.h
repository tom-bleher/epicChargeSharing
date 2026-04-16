// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

// NOTE: <algorithms/algorithm.h> and "algorithms/interfaces/WithPodConfig.h" are provided
// by the eic/algorithms and eic/EICrecon packages.  If these headers are not found at build
// time, ensure that EICrecon (and its transitive algorithm dependency) is installed and
// that the include directories are set in CMakeLists.txt.
#include <algorithms/algorithm.h>
#include <algorithms/interfaces/WithPodConfig.h>

#include "ChargeSharingConfig.h"
#include "ChargeSharingCore.hh"
#include "GaussianFit.hh"

#include <edm4eic/MCRecoTrackerHitAssociationCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Forward declarations for DD4hep types used by the algorithm
namespace dd4hep {
class Detector;
namespace DDSegmentation {
class BitFieldCoder;
} // namespace DDSegmentation
namespace rec {
class CellIDPositionConverter;
} // namespace rec
} // namespace dd4hep

namespace eicrecon {

// Import types from the charge-sharing core library into this namespace
namespace core = epic::chargesharing::core;
namespace fit = epic::chargesharing::fit;

// ---------------------------------------------------------------------------
// Algorithm type alias -- defines the collection-level I/O contract.
// ---------------------------------------------------------------------------
using ChargeSharingReconAlgorithm = algorithms::Algorithm<
    algorithms::Input<edm4hep::SimTrackerHitCollection>,
    algorithms::Output<edm4eic::TrackerHitCollection, edm4eic::MCRecoTrackerHitAssociationCollection>>;

/// @brief Charge sharing position reconstruction for AC-LGAD sensors.
///
/// Implements the EICrecon algorithms::Algorithm interface so that it can be
/// driven by a thin JOmniFactory wrapper.  The algorithm receives a full
/// SimTrackerHitCollection, simulates charge sharing across a pixel
/// neighborhood, and produces a TrackerHitCollection with reconstructed
/// positions (via centroid or Gaussian fitting).
///
/// @note Thread Safety: The process() method mutates internal RNG state
/// (m_noise_model contains std::mt19937) and is NOT safe for concurrent use
/// from multiple threads on the same instance. In JANA2/EICrecon, this is
/// safe because JOmniFactory creates a separate factory instance per worker
/// thread, each owning its own ChargeSharingReconstructor. If this class is
/// used outside JANA2, callers MUST ensure each thread owns its own instance.
class ChargeSharingReconstructor
    : public ChargeSharingReconAlgorithm,
      public eicrecon::WithPodConfig<ChargeSharingConfig> {
public:
    ChargeSharingReconstructor(std::string_view name)
        : ChargeSharingReconAlgorithm{
              name,
              {"inputSimTrackerHits"},
              {"outputTrackerHits", "outputHitAssociations"},
              "Charge sharing position reconstruction for AC-LGAD sensors"} {}

    /// One-time initialization: apply config, compute grid bounds, set up noise model.
    void init() final;

    /// Process a full collection of SimTrackerHits and populate TrackerHits.
    /// This is the collection-level interface required by algorithms::Algorithm.
    void process(const Input&, const Output&) const final;

    // -----------------------------------------------------------------------
    // Internal helper types -- kept public for diagnostic/test access.
    // -----------------------------------------------------------------------

    struct SingleHitInput {
        std::array<double, 3> hitPositionMM{};               ///< Truth position in mm
        std::optional<std::array<double, 3>> pixelHintMM{};  ///< Optional center of seed pixel (mm)
        std::optional<std::pair<int, int>> pixelIndexHint{}; ///< Pre-decoded grid indices (i,j)
        double energyDepositGeV{0.0};                        ///< Energy in GeV
        std::uint64_t cellID{0};
    };

    struct NeighborData {
        double fraction{std::numeric_limits<double>::quiet_NaN()};
        double chargeC{0.0};
        double distanceMM{0.0}; ///< d_i: distance to pixel center (mm)
        double alphaRad{0.0};
        double pixelXMM{0.0};
        double pixelYMM{0.0};
        int pixelId{-1};
        int di{0};
        int dj{0};

        // Mode-specific fractions and charges for diagnostics
        double fractionRow{0.0};   ///< Fraction using row-only denominator
        double fractionCol{0.0};   ///< Fraction using col-only denominator
        double fractionBlock{0.0}; ///< Fraction using block denominator
        double chargeRowC{0.0};    ///< Charge using row fraction (Coulombs)
        double chargeColC{0.0};    ///< Charge using col fraction (Coulombs)
        double chargeBlockC{0.0};  ///< Charge using block fraction (Coulombs)
    };

    struct SingleHitResult {
        std::array<double, 3> nearestPixelCenterMM{};
        std::array<double, 3> reconstructedPositionMM{};
        int pixelRowIndex{0};
        int pixelColIndex{0};
        double totalCollectedChargeC{0.0};
        std::vector<NeighborData> neighbors;

        // Fit results (if Gaussian fitting enabled)
        fit::GaussFit1DResult fitRowX;
        fit::GaussFit1DResult fitColY;
        fit::GaussFit2DResult fit2D;

        // Diagnostic Metadata
        std::array<double, 3> truthPositionMM{}; ///< Original hit position (mm)
        double inputEnergyDepositGeV{0.0};       ///< Input energy deposit (GeV)
        std::uint64_t inputCellID{0};            ///< Input cell ID

        // Reconstruction residuals
        double residualXMM() const { return reconstructedPositionMM[0] - truthPositionMM[0]; }
        double residualYMM() const { return reconstructedPositionMM[1] - truthPositionMM[1]; }

        // Summary charge statistics
        double maxNeighborChargeC{0.0}; ///< Max charge in neighborhood
        int numActiveNeighbors{0};      ///< Number of pixels with charge > 0

        // Grid info
        int neighborhoodRadius{0};   ///< Neighborhood radius used
        int neighborhoodGridSize{0}; ///< Total grid size (2*r+1)^2
    };

    // NOTE: The base class algorithms::Algorithm defines Input and Output as
    // collection-level tuple types. The old single-hit types are now named
    // SingleHitInput and SingleHitResult. If legacy callers used
    // ChargeSharingReconstructor::Input or ::Result, they should be updated
    // to use SingleHitInput / SingleHitResult instead.

private:
    struct PixelLocation {
        std::array<double, 3> center;
        int indexI{0};
        int indexJ{0};
    };

    struct IndexBounds {
        int minIndex{0};
        int maxIndex{-1};

        bool hasBounds() const { return maxIndex >= minIndex; }
        bool contains(int value) const { return !hasBounds() || (value >= minIndex && value <= maxIndex); }
    };

    /// Process a single hit through the charge-sharing reconstruction.
    SingleHitResult processSingleHit(const SingleHitInput& input) const;

    PixelLocation findNearestPixelFallback(const std::array<double, 3>& positionMM) const;
    PixelLocation pixelLocationFromIndices(int indexI, int indexJ) const;
    bool isPixelIndexInBounds(int indexI, int indexJ) const;

    /// Perform position reconstruction using configured method
    std::array<double, 3> reconstructPosition(const core::NeighborhoodResult& neighborhood, double centerZ,
                                              fit::GaussFit1DResult& fitRowX, fit::GaussFit1DResult& fitColY,
                                              fit::GaussFit2DResult& fit2D) const;

    /// Reconstruct using charge-weighted centroid (fallback)
    std::array<double, 3> reconstructCentroid(const core::NeighborhoodResult& neighborhood, double centerZ) const;

    /// Reconstruct using 1D Gaussian fits on row and column
    std::array<double, 3> reconstructGaussian1D(const core::NeighborhoodResult& neighborhood, double centerZ,
                                                fit::GaussFit1DResult& fitRowX, fit::GaussFit1DResult& fitColY) const;

    /// Reconstruct using 2D Gaussian fit
    std::array<double, 3> reconstructGaussian2D(const core::NeighborhoodResult& neighborhood, double centerZ,
                                                fit::GaussFit2DResult& fit2D) const;

    IndexBounds m_bounds_x{};
    IndexBounds m_bounds_y{};
    /// Noise model for realistic charge simulation.
    /// Declared mutable so noise can be applied in logically-const contexts.
    /// Contains std::mt19937 RNG state -- not thread-safe for concurrent access.
    /// See class-level @note for threading contract.
    mutable core::NoiseModel m_noise_model{};

    // DD4hep geometry state populated during init()
    const dd4hep::DDSegmentation::BitFieldCoder* m_decoder{nullptr};
    const dd4hep::rec::CellIDPositionConverter* m_converter{nullptr};
    std::string m_field_name_x{"x"};
    std::string m_field_name_y{"y"};
};

} // namespace eicrecon
