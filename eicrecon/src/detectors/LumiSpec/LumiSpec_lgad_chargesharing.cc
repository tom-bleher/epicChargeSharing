// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include <JANA/JApplication.h>

#include "extensions/jana/JOmniFactoryGeneratorT.h"
#include "factories/reco/LGADChargeSharingRecon_factory.h"

extern "C" {

void InitPlugin(JApplication* app) {
    InitJANAPlugin(app);

    // -----------------------------------------------------------------------
    // Luminosity Spectrometer tracker: SimTrackerHit -> TrackerHit (+ MC assoc)
    // Clustering is intentionally *not* registered for LumiSpec -- add it here
    // when the segmentation is ready and the downstream analysis is wired up.
    // -----------------------------------------------------------------------
    eicrecon::LGADChargeSharingReconConfig lumi_cfg;
    lumi_cfg.readout = "LumiSpecTrackerHits";

    app->Add(new JOmniFactoryGeneratorT<eicrecon::LGADChargeSharingRecon_factory>(
        "LumiSpecTrackerChargeSharingHitReco",
        {"LumiSpecTrackerHits"},
        {"LumiSpecTrackerChargeSharingHits", "LumiSpecTrackerChargeSharingHitAssociations"},
        lumi_cfg, app));
}

} // extern "C"
