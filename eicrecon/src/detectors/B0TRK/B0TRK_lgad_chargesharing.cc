// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include <JANA/JApplication.h>

#include "extensions/jana/JOmniFactoryGeneratorT.h"
#include "factories/reco/LGADChargeSharingRecon_factory.h"
#include "factories/tracking/LGADGaussianClustering_factory.h"

extern "C" {

void InitPlugin(JApplication* app) {
    InitJANAPlugin(app);

    // -----------------------------------------------------------------------
    // B0 tracker: SimTrackerHit -> TrackerHit (+ MC association)
    // -----------------------------------------------------------------------
    eicrecon::LGADChargeSharingReconConfig b0_cfg;
    b0_cfg.readout = "B0TrackerHits";

    app->Add(new JOmniFactoryGeneratorT<eicrecon::LGADChargeSharingRecon_factory>(
        "B0TrackerChargeSharingHitReco",
        {"B0TrackerHits"},
        {"B0TrackerChargeSharingHits", "B0TrackerChargeSharingHitAssociations"},
        b0_cfg, app));

    // -----------------------------------------------------------------------
    // B0 tracker: TrackerHit -> Measurement2D (cluster positions)
    // -----------------------------------------------------------------------
    eicrecon::LGADGaussianClusteringConfig b0_cluster_cfg;
    b0_cluster_cfg.readout = "B0TrackerHits";
    b0_cluster_cfg.reconMethod = 2; // Gaussian 2D

    app->Add(new JOmniFactoryGeneratorT<eicrecon::LGADGaussianClustering_factory>(
        "B0TrackerChargeSharingClustering",
        {"B0TrackerChargeSharingHits"},
        {"B0TrackerClusterHits"},
        b0_cluster_cfg, app));
}

} // extern "C"
