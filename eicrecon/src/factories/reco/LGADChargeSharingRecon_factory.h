// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include "algorithms/reco/LGADChargeSharingRecon.h"
#include "algorithms/reco/LGADChargeSharingReconConfig.h"

#include "extensions/jana/JOmniFactory.h"
#include "services/algorithms_init/AlgorithmsInit_service.h"

#include <edm4eic/MCRecoTrackerHitAssociationCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>

#include <memory>
#include <string>

namespace eicrecon {

/// Thin JOmniFactory wrapper around LGADChargeSharingRecon.
///
/// Exposes only the twelve user-facing knobs; everything else is derived from
/// DD4hep inside the algorithm's init().
class LGADChargeSharingRecon_factory
    : public JOmniFactory<LGADChargeSharingRecon_factory, LGADChargeSharingReconConfig> {
public:
    using AlgoT = LGADChargeSharingRecon;

private:
    std::unique_ptr<AlgoT> m_algo;

    PodioInput<edm4hep::SimTrackerHit> m_in_simhits{this};
    PodioOutput<edm4eic::TrackerHit> m_out_hits{this};
    PodioOutput<edm4eic::MCRecoTrackerHitAssociation> m_out_assocs{this};

    Service<AlgorithmsInit_service> m_algorithms_init{this};

    // Enum-backed integer parameters. We store the raw int alongside the config
    // enum so that the JANA CLI can set them, then cast in Configure().
    int m_signal_model_value{static_cast<int>(SignalModel::LogA)};
    int m_active_pixel_mode_value{static_cast<int>(ActivePixelMode::Neighborhood)};
    int m_recon_method_value{static_cast<int>(ReconMethod::Gaussian2D)};

    ParameterRef<int> m_signal_model{this, "signalModel", m_signal_model_value,
                                     "Signal model: 0=LogA (logarithmic), 1=LinA (linear)"};
    ParameterRef<int> m_active_pixel_mode{
        this, "activePixelMode", m_active_pixel_mode_value,
        "Active pixel mode: 0=Neighborhood, 1=RowCol, 2=RowCol3x3, 3=ChargeBlock2x2, 4=ChargeBlock3x3"};
    ParameterRef<int> m_recon_method{this, "reconMethod", m_recon_method_value,
                                     "Position recon method: 0=Centroid, 1=Gaussian1D, 2=Gaussian2D"};

    ParameterRef<std::string> m_readout{this, "readout", config().readout,
                                        "DD4hep readout name for segmentation lookup"};
    ParameterRef<float> m_min_edep{this, "minEDepGeV", config().minEDepGeV,
                                   "Minimum energy deposit threshold (GeV)"};
    ParameterRef<int> m_neighborhood_radius{this, "neighborhoodRadius", config().neighborhoodRadius,
                                            "Neighborhood half-width (2 = 5x5 grid)"};

    ParameterRef<double> m_d0_micron{this, "d0Micron", config().d0Micron,
                                     "LogA model: transverse hit size d0 (um)"};
    ParameterRef<double> m_linear_beta{this, "linearBetaPerMicron", config().linearBetaPerMicron,
                                       "LinA model: attenuation coefficient (1/um), 0 = auto from pitch"};
    ParameterRef<double> m_ionization_energy{this, "ionizationEnergyEV", config().ionizationEnergyEV,
                                             "Silicon e/h pair ionization energy (eV)"};
    ParameterRef<double> m_amplification{this, "amplificationFactor", config().amplificationFactor,
                                         "AC-LGAD amplification factor (gain)"};
    ParameterRef<bool> m_noise_enabled{this, "noiseEnabled", config().noiseEnabled,
                                       "Enable electronic noise injection"};
    ParameterRef<double> m_noise_electrons{this, "noiseElectronCount", config().noiseElectronCount,
                                           "Electronic noise RMS (electrons)"};

public:
    void Configure() {
        config().signalModel = static_cast<SignalModel>(m_signal_model_value);
        config().activePixelMode = static_cast<ActivePixelMode>(m_active_pixel_mode_value);
        config().reconMethod = static_cast<ReconMethod>(m_recon_method_value);

        m_algo = std::make_unique<AlgoT>(GetPrefix());
        m_algo->level(static_cast<algorithms::LogLevel>(logger()->level()));
        m_algo->applyConfig(config());
        m_algo->init();
    }

    void ChangeRun(int32_t /* run_number */) {}

    void Process(int32_t /* run_number */, uint64_t /* event_number */) {
        m_algo->process({m_in_simhits()}, {m_out_hits().get(), m_out_assocs().get()});
    }
};

} // namespace eicrecon
