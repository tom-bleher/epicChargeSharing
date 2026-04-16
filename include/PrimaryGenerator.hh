// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// \file PrimaryGenerator.hh
/// \brief Definition of the ECS::PrimaryGenerator class.
///
/// This file declares the PrimaryGenerator class which configures the
/// particle gun and samples primary vertex positions.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_PRIMARY_GENERATOR_HH
#define ECS_PRIMARY_GENERATOR_HH

#include "G4ThreeVector.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

#include "Config.hh"

#include <memory>
#include <mutex>

class G4Event;
class G4ParticleGun;
class G4ParticleDefinition;
class G4GenericMessenger;

namespace ECS {

// Forward declarations
class DetectorConstruction;

/// \brief Primary particle generator for the simulation.
///
/// Configures and manages the particle gun, supporting:
/// - Fixed position mode: particles generated at a specific (x,y) coordinate
/// - Random position mode: uniform sampling within detector bounds
///
/// The generator automatically computes safe margins to avoid edge effects
/// based on the neighborhood radius configuration.
///
/// UI Commands:
/// - /ecs/gun/fixedPosition (bool): Enable/disable fixed position mode
/// - /ecs/gun/fixedX (double): X coordinate for fixed position
/// - /ecs/gun/fixedY (double): Y coordinate for fixed position
class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
public:
    /// \brief Construct with detector reference.
    /// \param detector Pointer to detector construction for geometry info
    explicit PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator() override = default;

    /// \brief Generate primary particles for an event.
    /// \param event The event to populate with primary particles
    void GeneratePrimaries(G4Event* event) override;

private:
    void ConfigureParticleGun();
    void ConfigureMessenger();
    G4double CalculateSafeMargin() const;
    void RecalculateSamplingWindow() const;

    /// \brief Sampling window parameters.
    struct SamplingWindow {
        G4double margin{0.0};            ///< Safe margin from detector edge (for fixed mode clamping)
        G4double halfExtent{0.0};        ///< Margin-reduced half-extent (fixed mode)
        G4double randomHalfExtent{0.0};  ///< Full detector + overshoot (random mode)
    };

    const SamplingWindow& EnsureSamplingWindow() const;
    static G4double ClampToWindow(G4double value, const SamplingWindow& window);
    G4ThreeVector MakeFixedPosition(const SamplingWindow& window) const;
    static G4ThreeVector MakeRandomPosition(const SamplingWindow& window);
    G4ThreeVector SamplePrimaryVertex() const;
    void ApplyParticlePosition(const G4ThreeVector& position);
    void GenerateRandomPos();

    void ApplyPreset(const G4String& name);
    void SyncToRuntimeConfig() const;

    std::unique_ptr<G4ParticleGun> fParticleGun;
    DetectorConstruction* fDetector{nullptr};

    std::unique_ptr<G4GenericMessenger> fMessenger;
    G4bool fUseFixedPosition{Constants::USE_FIXED_POSITION};
    G4double fFixedX{0.0};
    G4double fFixedY{0.0};
    G4double fBeamOvershoot{0.0};  ///< Fraction of detector half-size to extend random sampling beyond edges

    // Energy range: when fEnergyMax > fEnergyMin, sample uniformly per event
    G4double fEnergyMin{0.0};
    G4double fEnergyMax{0.0};

    // Angular range: theta from beam axis (-z), in radians
    G4double fThetaMin{0.0};
    G4double fThetaMax{0.0};

    G4String fPresetName{"default"};

    mutable std::once_flag fFixedPositionWarningFlag;
    mutable SamplingWindow fSamplingWindow;
    mutable G4bool fSamplingWindowValid{false};
};

} // namespace ECS

// Backward compatibility alias
using PrimaryGenerator = ECS::PrimaryGenerator;

#endif // ECS_PRIMARY_GENERATOR_HH
