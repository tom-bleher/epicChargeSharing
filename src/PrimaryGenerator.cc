// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/**
 * @file PrimaryGenerator.cc
 * @brief Configures particle type/energy and samples primary vertex positions safely within the detector.
 */
#include "PrimaryGenerator.hh"

#include "Config.hh"
#include "DetectorConstruction.hh"
#include "RuntimeConfig.hh"

#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4GenericMessenger.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>

namespace {
const G4ThreeVector kDefaultMomentumDirection(0.0, 0.0, -1.0);

G4ParticleDefinition* ResolveDefaultParticle() {
    G4ParticleTable* const particleTable = G4ParticleTable::GetParticleTable();
    if (!particleTable) {
        G4Exception("PrimaryGenerator::ResolveDefaultParticle", "MissingParticleTable", FatalException,
                    "Particle table is not available.");
        return nullptr;
    }

    G4ParticleDefinition* particle = particleTable->FindParticle("e-");
    if (!particle) {
        G4Exception("PrimaryGenerator::ResolveDefaultParticle", "MissingParticle", FatalException,
                    "Failed to find default particle 'e-'.");
    }
    return particle;
}
} // namespace

PrimaryGenerator::PrimaryGenerator(DetectorConstruction* detector)
    : fParticleGun(std::make_unique<G4ParticleGun>(1)), fDetector(detector),
      fUseFixedPosition(ECS::RuntimeConfig::Instance().useFixedPosition), fFixedX(-150.0 * um) {
    // Override from runtime config

    ConfigureParticleGun();
    ConfigureMessenger();

    if (fDetector) {
        fFixedY = 0.0; // Center of pad row (red line), not gap row
    }

    const auto& window = EnsureSamplingWindow();
    if (fDetector) {
        G4cout << "[PrimaryGenerator] Default fixed XY position (" << fFixedX / mm << ", " << fFixedY / mm
               << ") mm; random sampling +/-" << window.randomHalfExtent / mm << " mm"
               << " (overshoot " << fBeamOvershoot * 100 << "%)" << G4endl;
    } else {
        G4cout << "[PrimaryGenerator] No detector available; primaries generated at origin" << G4endl;
    }

    GenerateRandomPos();
}

void PrimaryGenerator::GeneratePrimaries(G4Event* event) {
    GenerateRandomPos();

    // Per-event energy sampling (uniform in [min, max])
    if (fEnergyMax > fEnergyMin) {
        const G4double energy = fEnergyMin + G4UniformRand() * (fEnergyMax - fEnergyMin);
        fParticleGun->SetParticleEnergy(energy);
    }

    // Per-event angular sampling (theta from -z axis, uniform phi)
    if (fThetaMax > 0.0) {
        const G4double theta = fThetaMin + G4UniformRand() * (fThetaMax - fThetaMin);
        const G4double phi = CLHEP::twopi * G4UniformRand();
        const G4ThreeVector dir(std::sin(theta) * std::cos(phi),
                                std::sin(theta) * std::sin(phi),
                                -std::cos(theta));
        fParticleGun->SetParticleMomentumDirection(dir);
    }

    fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGenerator::ConfigureParticleGun() {
    G4ParticleDefinition* const particle = ResolveDefaultParticle();

    if (particle) {
        fParticleGun->SetParticleDefinition(particle);
    }

    const G4double energy = ECS::RuntimeConfig::Instance().particleEnergy;
    fParticleGun->SetParticleEnergy(energy);
    fParticleGun->SetParticleMomentumDirection(kDefaultMomentumDirection);

    if (particle) {
        G4cout << "[PrimaryGenerator] Default particle " << particle->GetParticleName() << " at "
               << energy / GeV << " GeV, direction (" << kDefaultMomentumDirection.x() << ", "
               << kDefaultMomentumDirection.y() << ", " << kDefaultMomentumDirection.z() << ")" << G4endl;
    }

    SyncToRuntimeConfig();
}

void PrimaryGenerator::ConfigureMessenger() {
    fMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/gun/", "Primary generator configuration");

    auto& useFixedCmd =
        fMessenger->DeclareProperty("useFixedPosition", fUseFixedPosition,
                                    "Use fixed XY position instead of random sampling within the pixel-safety margin.");
    useFixedCmd.SetStates(G4State_PreInit, G4State_Idle);
    useFixedCmd.SetToBeBroadcasted(true);

    auto& fixedXCmd = fMessenger->DeclarePropertyWithUnit(
        "fixedX", "mm", fFixedX, "Fixed X coordinate for primaries when useFixedPosition is true.");
    fixedXCmd.SetStates(G4State_PreInit, G4State_Idle);
    fixedXCmd.SetToBeBroadcasted(true);

    auto& fixedYCmd = fMessenger->DeclarePropertyWithUnit(
        "fixedY", "mm", fFixedY, "Fixed Y coordinate for primaries when useFixedPosition is true.");
    fixedYCmd.SetStates(G4State_PreInit, G4State_Idle);
    fixedYCmd.SetToBeBroadcasted(true);

    auto& overshootCmd = fMessenger->DeclareProperty(
        "beamOvershoot", fBeamOvershoot,
        "Fraction of detector half-size to extend random sampling beyond edges (0 = full detector, 0.1 = 10% overshoot).");
    overshootCmd.SetStates(G4State_PreInit, G4State_Idle);
    overshootCmd.SetToBeBroadcasted(true);

    auto& energyMinCmd = fMessenger->DeclarePropertyWithUnit(
        "energyMin", "GeV", fEnergyMin,
        "Minimum energy for per-event uniform sampling (0 = use fixed /gun/energy).");
    energyMinCmd.SetStates(G4State_PreInit, G4State_Idle);
    energyMinCmd.SetToBeBroadcasted(true);

    auto& energyMaxCmd = fMessenger->DeclarePropertyWithUnit(
        "energyMax", "GeV", fEnergyMax,
        "Maximum energy for per-event uniform sampling (must be > energyMin to activate).");
    energyMaxCmd.SetStates(G4State_PreInit, G4State_Idle);
    energyMaxCmd.SetToBeBroadcasted(true);

    auto& thetaMinCmd = fMessenger->DeclarePropertyWithUnit(
        "thetaMin", "mrad", fThetaMin,
        "Minimum polar angle from beam axis for per-event angular sampling.");
    thetaMinCmd.SetStates(G4State_PreInit, G4State_Idle);
    thetaMinCmd.SetToBeBroadcasted(true);

    auto& thetaMaxCmd = fMessenger->DeclarePropertyWithUnit(
        "thetaMax", "mrad", fThetaMax,
        "Maximum polar angle from beam axis (must be > 0 to activate angular sampling).");
    thetaMaxCmd.SetStates(G4State_PreInit, G4State_Idle);
    thetaMaxCmd.SetToBeBroadcasted(true);

    fMessenger->DeclareMethod("preset", &PrimaryGenerator::ApplyPreset)
        .SetGuidance("Apply a detector preset: b0, lumi, or default")
        .SetParameterName("name", false)
        .SetStates(G4State_PreInit, G4State_Idle)
        .SetToBeBroadcasted(true);
}

G4double PrimaryGenerator::CalculateSafeMargin() const {
    if (!fDetector) {
        return 0.0;
    }

    const G4int radius = fDetector->GetNeighborhoodRadius();
    // For DD4hep-style centered grid, margin is based on neighborhood radius
    // to ensure all neighbor pixels stay within detector bounds
    const G4double margin = (0.5 * fDetector->GetPixelSize()) + (radius * fDetector->GetPixelSpacing());

    const G4double detHalfSize = fDetector->GetDetSize() / 2.0;
    if (margin >= detHalfSize) {
        G4Exception("PrimaryGenerator::CalculateSafeMargin", "MarginTooLarge", FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    return margin;
}

void PrimaryGenerator::RecalculateSamplingWindow() const {
    SamplingWindow window{};
    if (fDetector) {
        const G4double margin = CalculateSafeMargin();
        const G4double detHalfSize = fDetector->GetDetSize() / 2.0;
        window.margin = margin;
        window.halfExtent = std::max(0.0, detHalfSize - margin);
        window.randomHalfExtent = detHalfSize * (1.0 + fBeamOvershoot);
    }
    fSamplingWindow = window;
    fSamplingWindowValid = true;
}

const PrimaryGenerator::SamplingWindow& PrimaryGenerator::EnsureSamplingWindow() const {
    if (!fSamplingWindowValid) {
        RecalculateSamplingWindow();
    }
    return fSamplingWindow;
}

G4double PrimaryGenerator::ClampToWindow(G4double value, const SamplingWindow& window) {
    if (window.halfExtent <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -window.halfExtent, window.halfExtent);
}

G4ThreeVector PrimaryGenerator::MakeFixedPosition(const SamplingWindow& window) const {
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
    if (window.halfExtent <= 0.0) {
        return {0.0, 0.0, z};
    }

    const G4double targetX = ClampToWindow(fFixedX, window);
    const G4double targetY = ClampToWindow(fFixedY, window);

    if (targetX != fFixedX || targetY != fFixedY) {
        std::call_once(fFixedPositionWarningFlag, [&]() {
            std::ostringstream oss;
            oss << "Fixed primary position outside safe margin. Clamping to +/-" << window.halfExtent / mm << " mm.";
            G4Exception("PrimaryGenerator::MakeFixedPosition", "FixedPositionOutOfBounds", JustWarning,
                        oss.str().c_str());
        });
    }

    return {targetX, targetY, z};
}

G4ThreeVector PrimaryGenerator::MakeRandomPosition(const SamplingWindow& window) {
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
    if (window.randomHalfExtent <= 0.0) {
        return {0.0, 0.0, z};
    }

    const G4double halfExtent = window.randomHalfExtent;
    const G4double randomX = (2.0 * G4UniformRand() - 1.0) * halfExtent;
    const G4double randomY = (2.0 * G4UniformRand() - 1.0) * halfExtent;

    return {randomX, randomY, z};
}

G4ThreeVector PrimaryGenerator::SamplePrimaryVertex() const {
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;

    if (!fDetector) {
        return {0.0, 0.0, z};
    }

    const auto& window = EnsureSamplingWindow();
    return fUseFixedPosition ? MakeFixedPosition(window) : MakeRandomPosition(window);
}

void PrimaryGenerator::ApplyParticlePosition(const G4ThreeVector& position) {
    fParticleGun->SetParticlePosition(position);
}

void PrimaryGenerator::GenerateRandomPos() {
    fSamplingWindowValid = false;
    ApplyParticlePosition(SamplePrimaryVertex());
}

void PrimaryGenerator::ApplyPreset(const G4String& name) {
    G4ParticleTable* const table = G4ParticleTable::GetParticleTable();

    if (name == "b0") {
        // B0 tracker: forward protons, 50-250 GeV, 6-22 mrad incidence
        fParticleGun->SetParticleDefinition(table->FindParticle("proton"));
        fEnergyMin = 50.0 * GeV;
        fEnergyMax = 250.0 * GeV;
        fThetaMin = 6.0 * mrad;
        fThetaMax = 22.0 * mrad;
        fUseFixedPosition = false;
        fBeamOvershoot = 0.1;
        fPresetName = "b0";

        G4cout << "[PrimaryGenerator] Preset b0: proton, 50-250 GeV, 6-22 mrad" << G4endl;

    } else if (name == "lumi") {
        // Lumi spectrometer: BH pair e+/e-, 1-17.5 GeV, 10-50 mrad incidence
        fParticleGun->SetParticleDefinition(table->FindParticle("e-"));
        fEnergyMin = 1.0 * GeV;
        fEnergyMax = 17.5 * GeV;
        fThetaMin = 10.0 * mrad;
        fThetaMax = 50.0 * mrad;
        fUseFixedPosition = false;
        fBeamOvershoot = 0.1;
        fPresetName = "lumi";

        G4cout << "[PrimaryGenerator] Preset lumi: e-, 1-17.5 GeV, 10-50 mrad" << G4endl;

    } else if (name == "default") {
        // Reset to defaults
        fParticleGun->SetParticleDefinition(table->FindParticle("e-"));
        fParticleGun->SetParticleEnergy(ECS::RuntimeConfig::Instance().particleEnergy);
        fParticleGun->SetParticleMomentumDirection(kDefaultMomentumDirection);
        fEnergyMin = 0.0;
        fEnergyMax = 0.0;
        fThetaMin = 0.0;
        fThetaMax = 0.0;
        fUseFixedPosition = false;
        fBeamOvershoot = 0.0;
        fPresetName = "default";

        G4cout << "[PrimaryGenerator] Preset default: e-, "
               << ECS::RuntimeConfig::Instance().particleEnergy / GeV << " GeV, perpendicular" << G4endl;

    } else {
        G4Exception("PrimaryGenerator::ApplyPreset", "UnknownPreset", JustWarning,
                     ("Unknown preset '" + name + "'. Available: b0, lumi, default").c_str());
        return;
    }

    SyncToRuntimeConfig();
}

void PrimaryGenerator::SyncToRuntimeConfig() const {
    auto& rtConfig = ECS::RuntimeConfig::Instance();
    const G4ParticleDefinition* particle = fParticleGun->GetParticleDefinition();
    if (particle) rtConfig.particleName = particle->GetParticleName();
    rtConfig.fixedX = fFixedX;
    rtConfig.fixedY = fFixedY;
    rtConfig.beamOvershoot = fBeamOvershoot;
    rtConfig.energyMin = fEnergyMin;
    rtConfig.energyMax = fEnergyMax;
    rtConfig.thetaMin = fThetaMin;
    rtConfig.thetaMax = fThetaMax;
    rtConfig.presetName = fPresetName;
}
