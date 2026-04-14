// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include <JANA/JApplication.h>

#include "ChargeSharingConfig.h"
#include "ChargeSharingClusteringConfig.h"
#include "ChargeSharingMonitor.h"
#include "ChargeSharingReconstructor_factory.h"
#include "ChargeSharingClustering_factory.h"

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
    // Clustering with Gaussian fitting (drop-in for LGADHitClustering)
    // Takes calibrated TrackerHits, produces Measurement2D with fitted positions.
    // Generic — works with any detector using CartesianGridXY segmentation.
    // Wire additional detectors by adding more factory registrations here.
    // ═══════════════════════════════════════════════════════════════════════════

    // --- BTOF ---
    eicrecon::ChargeSharingClusteringConfig btof_cluster_config;
    btof_cluster_config.readout = "TOFBarrelHits";
    btof_cluster_config.reconMethod = 2; // Gaussian 2D

    app->Add(new JOmniFactoryGeneratorT<eicrecon::ChargeSharingClustering_factory>(
        "TOFBarrelCSClustering",
        {"TOFBarrelCalibratedHits"},
        {"TOFBarrelCSClusterHits"},
        btof_cluster_config, app));

    // --- B0 Tracker ---
    eicrecon::ChargeSharingClusteringConfig b0_cluster_config;
    b0_cluster_config.readout = "B0TrackerHits";
    b0_cluster_config.reconMethod = 2;

    app->Add(new JOmniFactoryGeneratorT<eicrecon::ChargeSharingClustering_factory>(
        "B0CSClustering",
        {"B0ChargeSharingTrackerHits"},
        {"B0CSClusterHits"},
        b0_cluster_config, app));

    // ═══════════════════════════════════════════════════════════════════════════
    // Monitoring processor (optional, for validation)
    // ═══════════════════════════════════════════════════════════════════════════
    app->Add(new eicrecon::ChargeSharingMonitor());
}
}
