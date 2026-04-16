// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// @file RuntimeConfig.cc
/// @brief Implementation of RuntimeConfig singleton and Geant4 messenger commands.

#include "RuntimeConfig.hh"
#include "G4GenericMessenger.hh"

namespace ECS {

RuntimeConfig& RuntimeConfig::Instance() {
    static RuntimeConfig instance;
    return instance;
}

RuntimeConfig::RuntimeConfig() {
    DefineCommands();
}

void RuntimeConfig::DefineCommands() {
    // Physics commands
    fPhysicsMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/physics/", "Physics parameter configuration");

    fPhysicsMessenger->DeclareProperty("ionizationEnergy", ionizationEnergy, "Energy per e-h pair in silicon (eV)")
        .SetRange("ionizationEnergy > 0");
    fPhysicsMessenger->DeclareProperty("gain", gain, "AC-LGAD gain factor").SetRange("gain > 0");
    fPhysicsMessenger->DeclareProperty("d0", d0, "LogA reference distance d0 (um)").SetRange("d0 > 0");
    fPhysicsMessenger->DeclareProperty("linearBeta", linearBeta, "LinA attenuation coefficient beta (1/um)")
        .SetRange("linearBeta >= 0");
    fPhysicsMessenger->DeclareProperty("gainExcessNoiseFactor", gainExcessNoiseFactor,
        "McIntyre excess noise factor F for avalanche gain fluctuation").SetRange("gainExcessNoiseFactor > 0");
    fPhysicsMessenger->DeclareProperty("gainSaturationCharge", gainSaturationCharge,
        "Gain saturation onset in primary electrons").SetRange("gainSaturationCharge > 0");
    fPhysicsMessenger->DeclareProperty("gainFluctuationEnabled", gainFluctuationEnabled,
        "Enable event-level stochastic gain fluctuation");
    fPhysicsMessenger->DeclareProperty("perStepChargeSharing", perStepChargeSharing,
        "Compute charge sharing per Geant4 step (for angled tracks)");

    // Noise commands
    fNoiseMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/noise/", "Noise model configuration");

    fNoiseMessenger->DeclareProperty("gainSigmaMin", pixelGainSigmaMin, "Min per-pixel gain variation")
        .SetRange("gainSigmaMin >= 0");
    fNoiseMessenger->DeclareProperty("gainSigmaMax", pixelGainSigmaMax, "Max per-pixel gain variation")
        .SetRange("gainSigmaMax >= 0");
    fNoiseMessenger->DeclareProperty("electronCount", noiseElectronCount, "Electronic noise RMS (electrons)")
        .SetRange("electronCount >= 0");
    fNoiseMessenger->DeclareProperty("thresholdSigma", readoutThresholdSigma,
                                     "Readout threshold in units of noise sigma (ThresholdAboveNoise mode)");

    // Fitting commands
    fFitMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/fit/", "Fit parameter configuration");

    fFitMessenger->DeclareProperty("errorPercentOfMax", fitErrorPercentOfMax, "Fit uncertainty (% of max charge)")
        .SetRange("errorPercentOfMax > 0");
    fFitMessenger->DeclareProperty("gaus1D", fitGaus1D, "Enable 1D Gaussian fitting");
    fFitMessenger->DeclareProperty("gaus2D", fitGaus2D, "Enable 2D Gaussian fitting");
    fFitMessenger->DeclareProperty("verticalUncertainties", fitUseVerticalUncertainties,
                                   "Enable distance-weighted vertical uncertainties in fits");

    // Mode commands
    fModeMessenger =
        std::make_unique<G4GenericMessenger>(this, "/ecs/mode/", "Signal model and active pixel mode selection");

    fModeMessenger->DeclareProperty("signalModel", activeMode,
                                    "Signal model: 0=LogA (logarithmic), 1=LinA (linear)");
    fModeMessenger->DeclareProperty(
        "activePixelMode", activePixelMode,
        "Active pixel mode: 0=Neighborhood, 1=RowCol, 2=RowCol3x3, 3=ChargeBlock2x2, 4=ChargeBlock3x3, "
        "5=ThresholdAboveNoise");

    // Output commands
    fOutputMessenger =
        std::make_unique<G4GenericMessenger>(this, "/ecs/output/", "Output and storage configuration");

    fOutputMessenger->DeclareProperty("storeFullGrid", storeFullGrid,
                                      "Store full-detector charge fractions per event");

    // Gun commands: useFixedPosition, fixedX, fixedY are owned by PrimaryGenerator's
    // own messenger (also at /ecs/gun/) which binds directly to its member variables.
    // RuntimeConfig only stores defaults that PrimaryGenerator reads at construction.
    // Only energy is registered here (PrimaryGenerator reads it from RuntimeConfig).
    fGunMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/gun/", "Particle gun configuration");

    fGunMessenger->DeclarePropertyWithUnit("energy", "GeV", particleEnergy, "Primary particle kinetic energy")
        .SetRange("energy > 0");

    // Charge sharing sigma commands (upstream EICrecon-compatible)
    fChargeSharingMessenger =
        std::make_unique<G4GenericMessenger>(this, "/ecs/chargeSharing/sigma/", "Charge sharing sigma configuration");

    fChargeSharingMessenger->DeclareProperty("mode", sigmaMode,
                                             "Sigma mode: \"abs\" (mm) or \"rel\" (fraction of cell width)");
    fChargeSharingMessenger->DeclareProperty("sharingX", sigmaSharingX, "X sharing sigma (0 = use physics model)")
        .SetRange("sharingX >= 0");
    fChargeSharingMessenger->DeclareProperty("sharingY", sigmaSharingY, "Y sharing sigma (0 = use physics model)")
        .SetRange("sharingY >= 0");
}

} // namespace ECS
