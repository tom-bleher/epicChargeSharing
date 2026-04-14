// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include <string>

namespace eicrecon {

/// Configuration for physics-model charge sharing clustering.
///
/// Extends LGADHitClusteringConfig with Gaussian fitting parameters
/// for improved position reconstruction beyond max-ADC or centroid.
struct ChargeSharingClusteringConfig {
    // ---- Clustering ----
    std::string readout; ///< DD4hep readout name (set per detector in plugin registration)
    double deltaT = 1.0;                  ///< Time gate for merging hits (ns)

    // ---- Reconstruction ----
    /// 0 = centroid (charge-weighted average, same as LGADHitClustering)
    /// 1 = Gaussian 1D (fit row + column slices independently)
    /// 2 = Gaussian 2D (fit full 2D Gaussian to cluster)
    int reconMethod = 2;

    // ---- Fitting ----
    double fitErrorPercent = 5.0; ///< Base uncertainty as % of max charge
};

} // namespace eicrecon
