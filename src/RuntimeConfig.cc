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
    fPhysicsMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/physics/",
        "Physics parameter configuration");

    fPhysicsMessenger->DeclareProperty("ionizationEnergy", ionizationEnergy,
        "Energy per e-h pair in silicon (eV)").SetRange("ionizationEnergy > 0");
    fPhysicsMessenger->DeclareProperty("gain", gain,
        "AC-LGAD gain factor").SetRange("gain > 0");
    fPhysicsMessenger->DeclareProperty("d0", d0,
        "LogA reference distance d0 (um)").SetRange("d0 > 0");
    fPhysicsMessenger->DeclareProperty("linearBeta", linearBeta,
        "LinA attenuation coefficient beta (1/um)").SetRange("linearBeta >= 0");

    // Noise commands
    fNoiseMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/noise/",
        "Noise model configuration");

    fNoiseMessenger->DeclareProperty("gainSigmaMin", pixelGainSigmaMin,
        "Min per-pixel gain variation").SetRange("gainSigmaMin >= 0");
    fNoiseMessenger->DeclareProperty("gainSigmaMax", pixelGainSigmaMax,
        "Max per-pixel gain variation").SetRange("gainSigmaMax >= 0");
    fNoiseMessenger->DeclareProperty("electronCount", noiseElectronCount,
        "Electronic noise RMS (electrons)").SetRange("electronCount >= 0");

    // Fitting commands
    fFitMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/fit/",
        "Fit parameter configuration");

    fFitMessenger->DeclareProperty("errorPercentOfMax", fitErrorPercentOfMax,
        "Fit uncertainty (% of max charge)").SetRange("errorPercentOfMax > 0");
}

} // namespace ECS
