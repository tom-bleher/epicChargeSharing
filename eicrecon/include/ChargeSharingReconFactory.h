// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include "ChargeSharingConfig.h"
#include "ChargeSharingReconstructor.h"

#include "extensions/jana/JOmniFactory.h"
#include "services/algorithms_init/AlgorithmsInit_service.h"
#include "services/geometry/dd4hep/DD4hep_service.h"

#include <edm4eic/MCRecoTrackerHitAssociationCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>

#include <memory>
#include <string>

namespace eicrecon {

/// Thin JOmniFactory wrapper around ChargeSharingReconstructor.
///
/// Following the EICrecon convention (e.g., TrackerHitReconstruction_factory),
/// this factory owns a std::unique_ptr<AlgoT>, configures it in Configure(),
/// and delegates to process() in Process().
class ChargeSharingReconFactory : public JOmniFactory<ChargeSharingReconFactory, ChargeSharingConfig> {
public:
    using AlgoT = ChargeSharingReconstructor;

private:
    std::unique_ptr<AlgoT> m_algo;

    PodioInput<edm4hep::SimTrackerHit> m_in_simhits{this};
    PodioOutput<edm4eic::TrackerHit> m_out_hits{this};
    PodioOutput<edm4eic::MCRecoTrackerHitAssociation> m_out_assocs{this};

    Service<AlgorithmsInit_service> m_algorithms_init{this};
    Service<DD4hep_service> m_dd4hep{this};

    // ─────────────────────────────── Mode Selection ─────────────────────────────
    // Storage for enum-backed integer parameters
    int m_signal_model_value{0};      // 0=LogA, 1=LinA
    int m_active_pixel_mode_value{0}; // 0=Neighborhood, 1=RowCol, etc.
    int m_recon_method_value{1};      // 0=Centroid, 1=Gaussian1D, 2=Gaussian2D

    ParameterRef<int> m_signal_model{this, "signalModel", m_signal_model_value,
                                     "Signal model: 0=LogA (logarithmic), 1=LinA (linear attenuation)"};
    ParameterRef<int> m_active_pixel_mode{
        this, "activePixelMode", m_active_pixel_mode_value,
        "Active pixel mode: 0=Neighborhood, 1=RowCol, 2=RowCol3x3, 3=ChargeBlock2x2, 4=ChargeBlock3x3"};
    ParameterRef<int> m_recon_method{this, "reconMethod", m_recon_method_value,
                                     "Reconstruction method: 0=Centroid, 1=Gaussian1D, 2=Gaussian2D"};

    // ─────────────────────────────── DD4hep Integration ─────────────────────────
    ParameterRef<std::string> m_readout{this, "readout", config().readout,
                                        "DD4hep readout name for segmentation lookup"};
    ParameterRef<float> m_min_edep{this, "minEDep", config().minEDepGeV, "Minimum energy deposit threshold (GeV)"};

    // ─────────────────────────────── Detector Geometry ────────────────────────────
    ParameterRef<int> m_neighborhood_radius{this, "neighborhoodRadius", config().neighborhoodRadius,
                                              "Neighborhood half-width (2 = 5x5 grid)"};
    ParameterRef<double> m_pixel_size{this, "pixelSizeMM", config().pixelSizeMM, "Pixel pad size (mm)"};
    ParameterRef<double> m_pixel_spacing{this, "pixelSpacingMM", config().pixelSpacingMM,
                                          "Center-to-center pixel pitch (mm)"};
    ParameterRef<double> m_grid_offset{this, "gridOffsetMM", config().gridOffsetMM,
                                        "DD4hep-style grid offset (mm), 0 = centered grid"};
    ParameterRef<double> m_detector_size{this, "detectorSizeMM", config().detectorSizeMM, "Sensor side length (mm)"};
    ParameterRef<double> m_detector_thickness{this, "detectorThicknessMM", config().detectorThicknessMM,
                                               "Silicon thickness (mm)"};
    ParameterRef<double> m_pixel_thickness{this, "pixelThicknessMM", config().pixelThicknessMM, "Pixel depth (mm)"};
    ParameterRef<double> m_detector_z_center{this, "detectorZCenterMM", config().detectorZCenterMM,
                                              "Detector reference plane (mm)"};
    ParameterRef<int> m_pixels_per_side{this, "pixelsPerSide", config().pixelsPerSide,
                                         "Total pixels per side (0 = auto from geometry)"};

    // ─────────────────────────────── Physics ────────────────────────────────────
    ParameterRef<double> m_ionization_energy{this, "ionizationEnergyEV", config().ionizationEnergyEV,
                                               "Energy per e-h pair in silicon (eV)"};
    ParameterRef<double> m_amplification_factor{this, "amplificationFactor", config().amplificationFactor,
                                                 "AC-LGAD gain factor (8-25 typical)"};
    ParameterRef<double> m_elementary_charge{this, "elementaryChargeC", config().elementaryChargeC,
                                              "Elementary charge (C)"};
    ParameterRef<double> m_d0_micron{this, "d0Micron", config().d0Micron,
                                      "Transverse hit size d0 for LogA model (um)"};
    ParameterRef<double> m_linear_beta{this, "linearBetaPerMicron", config().linearBetaPerMicron,
                                        "LinA attenuation coefficient (1/um), 0 = auto from pitch"};

    // ─────────────────────────────── Noise Model ─────────────────────────────────
    ParameterRef<bool> m_noise_enabled{this, "noiseEnabled", config().noiseEnabled,
                                       "Enable noise injection for realistic charge simulation"};
    ParameterRef<double> m_noise_gain_sigma_min{this, "noiseGainSigmaMin", config().noiseGainSigmaMin,
                                                 "Minimum per-pixel gain variation (1% = 0.01)"};
    ParameterRef<double> m_noise_gain_sigma_max{this, "noiseGainSigmaMax", config().noiseGainSigmaMax,
                                                 "Maximum per-pixel gain variation (5% = 0.05)"};
    ParameterRef<double> m_noise_electron_count{this, "noiseElectronCount", config().noiseElectronCount,
                                                 "Electronic noise RMS (electrons)"};

    // ─────────────────────────────── Fitting ───────────────────────────────────
    ParameterRef<double> m_fit_error_percent{this, "fitErrorPercentOfMax", config().fitErrorPercentOfMax,
                                               "Fit uncertainty as percentage of max charge"};
    ParameterRef<bool> m_fit_use_vertical_uncertainties{this, "fitUseVerticalUncertainties",
                                                         config().fitUseVerticalUncertainties,
                                                         "Enable vertical uncertainties in fits"};
    ParameterRef<bool> m_fit_use_distance_weighted_errors{this, "fitUseDistanceWeightedErrors",
                                                           config().fitUseDistanceWeightedErrors,
                                                           "Enable distance-weighted fit errors"};
    ParameterRef<double> m_fit_distance_scale_pixels{this, "fitDistanceScalePixels", config().fitDistanceScalePixels,
                                                      "Distance scale in pixel units"};
    ParameterRef<double> m_fit_distance_exponent{this, "fitDistanceExponent", config().fitDistanceExponent,
                                                  "Distance error power law exponent"};
    ParameterRef<double> m_fit_distance_floor_percent{this, "fitDistanceFloorPercent", config().fitDistanceFloorPercent,
                                                       "Minimum fit error (% of Q_max)"};
    ParameterRef<double> m_fit_distance_cap_percent{this, "fitDistanceCapPercent", config().fitDistanceCapPercent,
                                                     "Maximum fit error (% of Q_max)"};

    // ────────────────────────────── Diagnostics ─────────────────────────────────
    ParameterRef<bool> m_emit_neighbor_diagnostics{this, "emitNeighborDiagnostics", config().emitNeighborDiagnostics,
                                                   "Record per-neighbor distances/angles"};

public:
    void Configure() {
        // Convert integer parameters to enum values before passing to algorithm
        config().signalModel = static_cast<SignalModel>(m_signal_model_value);
        config().activePixelMode = static_cast<ActivePixelMode>(m_active_pixel_mode_value);
        config().reconMethod = static_cast<ReconMethod>(m_recon_method_value);

        // Create algorithm, apply config, and initialize
        m_algo = std::make_unique<AlgoT>(GetPrefix());
        m_algo->level(static_cast<algorithms::LogLevel>(logger()->level()));
        m_algo->applyConfig(config());
        m_algo->init();
    }

    void Process(int32_t /* run_number */, uint64_t /* event_number */) {
        m_algo->process({m_in_simhits()}, {m_out_hits().get(), m_out_assocs().get()});
    }
};

} // namespace eicrecon
