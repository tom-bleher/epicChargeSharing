// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include "chargesharing/core/ChargeSharingCore.hh"

#include <string>

namespace eicrecon {

/// Configuration for LGADGaussianClustering.
///
/// Union-find clustering of TrackerHits followed by Gaussian-fit sub-pad
/// position extraction. Drop-in shape-compatible with upstream
/// LGADHitClustering but with Gaussian reconstruction bolted on.
struct LGADGaussianClusteringConfig {
    /// DD4hep readout name (required; set per detector).
    std::string readout;

    /// Time gate (ns) for union-find neighbour merge.
    double deltaT{1.0};

    /// Position reconstruction method within a cluster.
    /// 0 = charge-weighted centroid
    /// 1 = 1D Gaussian on row + column slices
    /// 2 = full 2D Gaussian
    int reconMethod{static_cast<int>(::chargesharing::core::ReconMethod::Gaussian2D)};

    /// Fit uncertainty as a percentage of the cluster's max charge.
    double fitErrorPercent{5.0};
};

} // namespace eicrecon
