/// @file RuntimeConfig.hh
/// @brief Runtime configuration for the standalone simulation.
///
/// This class wraps compile-time defaults from Config.hh but allows
/// runtime overrides via Geant4 macro commands (/ecs/ command tree).
/// If no macro commands override values, the simulation runs identically
/// to the compile-time configuration.

#ifndef ECS_RUNTIME_CONFIG_HH
#define ECS_RUNTIME_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "globals.hh"
#include "Config.hh"  // For compile-time defaults

#include <memory>

class G4GenericMessenger;

namespace ECS {

/// Runtime configuration for the standalone simulation.
/// Initialized from compile-time defaults in Config.hh, but values
/// can be overridden via Geant4 macro commands (e.g., /ecs/physics/d0 5.0).
///
/// Usage in macro files:
///   /ecs/physics/d0 1.0
///   /ecs/physics/gain 20.0
///   /ecs/physics/ionizationEnergy 3.6
///   /ecs/noise/electronCount 500
///   /ecs/noise/gainSigmaMin 0.01
///   /ecs/noise/gainSigmaMax 0.05
///   /ecs/fit/errorPercentOfMax 5.0
///   /ecs/gun/useFixedPosition true
///
/// Note: Detector geometry parameters (pixelSize, pixelPitch, etc.) are
/// already handled by DetectorConstruction's own messenger at /ecs/detector/.
class RuntimeConfig {
public:
    static RuntimeConfig& Instance();

    // Delete copy/move
    RuntimeConfig(const RuntimeConfig&) = delete;
    RuntimeConfig& operator=(const RuntimeConfig&) = delete;
    RuntimeConfig(RuntimeConfig&&) = delete;
    RuntimeConfig& operator=(RuntimeConfig&&) = delete;

    // ---- Mode Selection ----
    // Mode is kept compile-time (structural choice affecting branches/fitting)

    // ---- Detector Geometry ----
    // These are exposed here for read access, but note that
    // DetectorConstruction's messenger handles runtime changes to geometry.
    // These values serve as initial defaults that DetectorConstruction reads.
    G4double detectorSize{Constants::DETECTOR_SIZE};
    G4double detectorWidth{Constants::DETECTOR_WIDTH};
    G4double pixelSize{Constants::PIXEL_SIZE};
    G4double pixelPitch{Constants::PIXEL_PITCH};
    G4int neighborhoodRadius{Constants::NEIGHBORHOOD_RADIUS};
    G4double pixelThickness{Constants::PIXEL_THICKNESS};
    G4double gridOffset{Constants::GRID_OFFSET};

    // ---- Physics ----
    G4double ionizationEnergy{Constants::IONIZATION_ENERGY};
    G4double gain{Constants::GAIN};
    G4double d0{Constants::D0};
    G4double linearBeta{Constants::LINEAR_CHARGE_MODEL_BETA};

    // ---- Noise ----
    G4double pixelGainSigmaMin{Constants::PIXEL_GAIN_SIGMA_MIN};
    G4double pixelGainSigmaMax{Constants::PIXEL_GAIN_SIGMA_MAX};
    G4double noiseElectronCount{Constants::NOISE_ELECTRON_COUNT};

    // ---- Particle Gun ----
    G4bool useFixedPosition{Constants::USE_FIXED_POSITION};

    // ---- Fitting ----
    G4double fitErrorPercentOfMax{Constants::FIT_ERROR_PERCENT_OF_MAX};

private:
    RuntimeConfig();
    void DefineCommands();

    std::unique_ptr<G4GenericMessenger> fPhysicsMessenger;
    std::unique_ptr<G4GenericMessenger> fNoiseMessenger;
    std::unique_ptr<G4GenericMessenger> fFitMessenger;
};

} // namespace ECS

#endif // ECS_RUNTIME_CONFIG_HH
