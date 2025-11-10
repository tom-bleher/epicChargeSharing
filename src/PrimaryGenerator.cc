/**
 * @file PrimaryGenerator.cc
 * @brief Configures particle type/energy and samples primary vertex positions safely within the detector.
 */
#include "PrimaryGenerator.hh"

#include "Constants.hh"
#include "DetectorConstruction.hh"

#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4GenericMessenger.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <mutex>
#include <sstream>

namespace
{
constexpr G4double kDefaultParticleEnergy = 10.0 * GeV;
const G4ThreeVector kDefaultMomentumDirection(0.0, 0.0, -1.0);

G4ParticleDefinition* ResolveDefaultParticle()
{
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    if (!particleTable) {
        G4Exception("PrimaryGenerator::ResolveDefaultParticle",
                    "MissingParticleTable",
                    FatalException,
                    "Particle table is not available.");
        return nullptr;
    }

    G4ParticleDefinition* particle = particleTable->FindParticle("e-");
    if (!particle) {
        G4Exception("PrimaryGenerator::ResolveDefaultParticle",
                    "MissingParticle",
                    FatalException,
                    "Failed to find default particle 'e-'.");
    }
    return particle;
}
} // namespace

PrimaryGenerator::PrimaryGenerator(DetectorConstruction* detector)
    : fParticleGun(std::make_unique<G4ParticleGun>(1)),
      fDetector(detector)
{
    ConfigureParticleGun();
    ConfigureMessenger();

    fFixedX = 125.0*um;
    if (fDetector) {
        fFixedY = 0.5 * fDetector->GetPixelSpacing();
    }

    const auto& window = EnsureSamplingWindow();
    if (fDetector) {
        G4cout << "[PrimaryGenerator] Default fixed XY position (" << fFixedX / mm << ", "
               << fFixedY / mm << ") mm; random sampling confined to +/-" << window.halfExtent / mm
               << " mm" << G4endl;
    } else {
        G4cout << "[PrimaryGenerator] No detector available; primaries generated at origin"
               << G4endl;
    }

    GenerateRandomPos();
}

void PrimaryGenerator::GeneratePrimaries(G4Event* event)
{
    GenerateRandomPos();
    fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGenerator::ConfigureParticleGun()
{
    G4ParticleDefinition* particle = ResolveDefaultParticle();

    if (particle) {
        fParticleGun->SetParticleDefinition(particle);
    }

    fParticleGun->SetParticleEnergy(kDefaultParticleEnergy);
    fParticleGun->SetParticleMomentumDirection(kDefaultMomentumDirection);

    if (particle) {
        G4cout << "[PrimaryGenerator] Default particle " << particle->GetParticleName()
               << " at " << kDefaultParticleEnergy / GeV << " GeV, direction ("
               << kDefaultMomentumDirection.x() << ", " << kDefaultMomentumDirection.y()
               << ", " << kDefaultMomentumDirection.z() << ")" << G4endl;
    }
}

void PrimaryGenerator::ConfigureMessenger()
{
    fMessenger = std::make_unique<G4GenericMessenger>(
        this,
        "/epic/gun/",
        "Primary generator configuration");

    auto& useFixedCmd = fMessenger->DeclareProperty(
        "useFixedPosition",
        fUseFixedPosition,
        "Use fixed XY position instead of random sampling within the pixel-safety margin.");
    useFixedCmd.SetStates(G4State_PreInit, G4State_Idle);
    useFixedCmd.SetToBeBroadcasted(true);

    auto& fixedXCmd = fMessenger->DeclarePropertyWithUnit(
        "fixedX",
        "mm",
        fFixedX,
        "Fixed X coordinate for primaries when useFixedPosition is true.");
    fixedXCmd.SetStates(G4State_PreInit, G4State_Idle);
    fixedXCmd.SetToBeBroadcasted(true);

    auto& fixedYCmd = fMessenger->DeclarePropertyWithUnit(
        "fixedY",
        "mm",
        fFixedY,
        "Fixed Y coordinate for primaries when useFixedPosition is true.");
    fixedYCmd.SetStates(G4State_PreInit, G4State_Idle);
    fixedYCmd.SetToBeBroadcasted(true);
}

G4double PrimaryGenerator::CalculateSafeMargin() const
{
    if (!fDetector) {
        return 0.0;
    }

    const G4int radius = fDetector->GetNeighborhoodRadius();
    const G4double margin = fDetector->GetPixelCornerOffset() +
                            0.5 * fDetector->GetPixelSize() +
                            radius * fDetector->GetPixelSpacing();

    const G4double detHalfSize = fDetector->GetDetSize() / 2.0;
    if (margin >= detHalfSize) {
        G4Exception("PrimaryGenerator::CalculateSafeMargin",
                    "MarginTooLarge",
                    FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    return margin;
}

void PrimaryGenerator::RecalculateSamplingWindow() const
{
    SamplingWindow window{};
    if (fDetector) {
        const G4double margin = CalculateSafeMargin();
        const G4double halfExtent = fDetector->GetDetSize() / 2.0 - margin;
        window.margin = margin;
        window.halfExtent = std::max(0.0, halfExtent);
    }
    fSamplingWindow = window;
    fSamplingWindowValid = true;
}

const PrimaryGenerator::SamplingWindow& PrimaryGenerator::EnsureSamplingWindow() const
{
    if (!fSamplingWindowValid) {
        RecalculateSamplingWindow();
    }
    return fSamplingWindow;
}

G4double PrimaryGenerator::ClampToWindow(G4double value, const SamplingWindow& window) const
{
    if (window.halfExtent <= 0.0) {
        return 0.0;
    }
    return std::clamp(value, -window.halfExtent, window.halfExtent);
}

G4ThreeVector PrimaryGenerator::MakeFixedPosition(const SamplingWindow& window) const
{
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
    if (window.halfExtent <= 0.0) {
        return {0.0, 0.0, z};
    }

    const G4double targetX = ClampToWindow(fFixedX, window);
    const G4double targetY = ClampToWindow(fFixedY, window);

    if (targetX != fFixedX || targetY != fFixedY) {
        std::call_once(fFixedPositionWarningFlag, [&]() {
            std::ostringstream oss;
            oss << "Fixed primary position outside safe margin. Clamping to +/-"
                << window.halfExtent / mm << " mm.";
            G4Exception("PrimaryGenerator::MakeFixedPosition",
                        "FixedPositionOutOfBounds",
                        JustWarning,
                        oss.str().c_str());
        });
    }

    return {targetX, targetY, z};
}

G4ThreeVector PrimaryGenerator::MakeRandomPosition(const SamplingWindow& window) const
{
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
    if (window.halfExtent <= 0.0) {
        return {0.0, 0.0, z};
    }

    const G4double halfExtent = window.halfExtent;
    const G4double randomX = (2.0 * G4UniformRand() - 1.0) * halfExtent;
    const G4double randomY = (2.0 * G4UniformRand() - 1.0) * halfExtent;

    return {randomX, randomY, z};
}

G4ThreeVector PrimaryGenerator::SamplePrimaryVertex() const
{
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;

    if (!fDetector) {
        return {0.0, 0.0, z};
    }

    const auto& window = EnsureSamplingWindow();
    return fUseFixedPosition ? MakeFixedPosition(window) : MakeRandomPosition(window);
}

void PrimaryGenerator::ApplyParticlePosition(const G4ThreeVector& position)
{
    fParticleGun->SetParticlePosition(position);
}

void PrimaryGenerator::GenerateRandomPos()
{
    fSamplingWindowValid = false;
    ApplyParticlePosition(SamplePrimaryVertex());
}

