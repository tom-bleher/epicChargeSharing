// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include <JANA/JApplication.h>

#include "ChargeSharingConfig.h"
#include "ChargeSharingMonitor.h"
#include "ChargeSharingReconFactory.h"

#include "extensions/jana/JOmniFactoryGeneratorT.h"

extern "C" {
void InitPlugin(JApplication* app) {
    InitJANAPlugin(app);

    // ═══════════════════════════════════════════════════════════════════════════
    // B0 Tracker (forward region)
    // ═══════════════════════════════════════════════════════════════════════════
    eicrecon::ChargeSharingConfig b0_config;
    b0_config.readout = "B0TrackerHits";

    app->Add(new JOmniFactoryGeneratorT<eicrecon::ChargeSharingReconFactory>(
        "B0ChargeSharingRecon",         // Tag
        {"B0TrackerHits"},              // Input: B0 tracker SimTrackerHit
        {"B0ChargeSharingTrackerHits", "B0ChargeSharingTrackerHitAssociations"},
        b0_config, app));

    // ═══════════════════════════════════════════════════════════════════════════
    // Luminosity Spectrometer Tracker
    // ═══════════════════════════════════════════════════════════════════════════
    eicrecon::ChargeSharingConfig lumi_config;
    lumi_config.readout = "LumiSpecTrackerHits";
    // Lumi tracker may have different geometry - adjust as needed
    // lumi_config.pixelSizeMM = 0.5;
    // lumi_config.pixelSpacingMM = 0.5;

    app->Add(new JOmniFactoryGeneratorT<eicrecon::ChargeSharingReconFactory>(
        "LumiSpecTrackerChargeSharingRecon",         // Tag
        {"LumiSpecTrackerHits"},                     // Input: Lumi tracker SimTrackerHit
        {"LumiSpecTrackerChargeSharingHits", "LumiSpecTrackerChargeSharingHitAssociations"},
        lumi_config, app));

    // ═══════════════════════════════════════════════════════════════════════════
    // Monitoring processor (optional, for validation)
    // ═══════════════════════════════════════════════════════════════════════════
    app->Add(new eicrecon::ChargeSharingMonitor());
}
}
