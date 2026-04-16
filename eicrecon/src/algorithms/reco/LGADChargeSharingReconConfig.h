// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include "chargesharing/core/ChargeSharingCore.hh"

#include <string>

namespace eicrecon {

using ::chargesharing::core::ActivePixelMode;
using ::chargesharing::core::ReconMethod;
using ::chargesharing::core::SignalModel;

/// Configuration for LGADChargeSharingRecon.
///
/// Only user-settable fields live here; everything else (pixel pitch,
/// pad size, grid offset, detector size/thickness, cell counts, segmentation
/// field names) is derived from DD4hep inside the algorithm's init() and
/// cached on the private Geometry state.
struct LGADChargeSharingReconConfig {
    /// Charge-sharing physics model: LogA (logarithmic) or LinA (linear).
    SignalModel signalModel{SignalModel::LogA};

    /// Denominator selection for charge-fraction normalization.
    ActivePixelMode activePixelMode{ActivePixelMode::Neighborhood};

    /// Sub-pad position reconstruction method.
    ReconMethod reconMethod{ReconMethod::Gaussian2D};

    /// DD4hep readout name (required; set per detector).
    std::string readout;

    /// Energy deposit threshold for accepting a hit (GeV).
    float minEDepGeV{0.0F};

    /// Neighborhood half-width. 2 -> 5x5 grid.
    int neighborhoodRadius{2};

    /// LogA model: transverse hit size d0 (microns).
    double d0Micron{1.0};

    /// LinA model: attenuation coefficient (1/um). 0 = auto-select from pitch.
    double linearBetaPerMicron{0.0};

    /// Silicon e/h pair ionization energy (eV).
    double ionizationEnergyEV{3.6};

    /// AC-LGAD amplification factor (gain).
    double amplificationFactor{20.0};

    /// Enable per-pad gain variation + electronic noise injection.
    bool noiseEnabled{true};

    /// Electronic noise RMS (electrons).
    double noiseElectronCount{500.0};
};

} // namespace eicrecon
