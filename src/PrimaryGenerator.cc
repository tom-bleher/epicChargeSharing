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

G4double ScriptedDefaultX()
{
    const G4double x = 25.0*um;
    return x;
}
} // namespace

PrimaryGenerator::PrimaryGenerator(DetectorConstruction* detector)
    : fParticleGun(std::make_unique<G4ParticleGun>(1)),
      fDetector(detector)
{
    ConfigureParticleGun();
    ConfigureMessenger();

    if (fDetector) {
        fFixedX = ScriptedDefaultX();
        fFixedY = 0.5 * fDetector->GetPixelSpacing();
        const G4double margin = ComputeSafeMargin();
        const G4double halfExtent = fDetector->GetDetSize() / 2.0 - margin;
        G4cout << "[PrimaryGenerator] Default fixed XY position (" << fFixedX / mm << ", "
               << fFixedY / mm << ") mm; random sampling confined to +/-" << halfExtent / mm
               << " mm" << G4endl;
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

G4double PrimaryGenerator::ComputeSafeMargin() const
{
    if (!fDetector) {
        G4Exception("PrimaryGenerator::ComputeSafeMargin",
                    "MissingDetector",
                    FatalException,
                    "DetectorConstruction pointer is null.");
        return 0.0;
    }

    const G4int radius = fDetector->GetNeighborhoodRadius();
    const G4double margin = fDetector->GetPixelCornerOffset() +
                            0.5 * fDetector->GetPixelSize() +
                            radius * fDetector->GetPixelSpacing();

    const G4double detSize = fDetector->GetDetSize();
    if (margin >= detSize / 2.0) {
        G4Exception("PrimaryGenerator::ComputeSafeMargin",
                    "MarginTooLarge",
                    FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    return margin;
}

G4ThreeVector PrimaryGenerator::SamplePrimaryVertex() const
{
    const G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;

    if (!fDetector) {
        return {ScriptedDefaultX(), 0.0, z};
    }

    const G4double margin = ComputeSafeMargin();
    const G4double detSize = fDetector->GetDetSize();
    const G4double halfExtent = detSize / 2.0 - margin;

    if (fUseFixedPosition) {
        G4double targetX = std::clamp(fFixedX, -halfExtent, halfExtent);
        G4double targetY = std::clamp(fFixedY, -halfExtent, halfExtent);

        if (targetX != fFixedX || targetY != fFixedY) {
            std::call_once(fFixedPositionWarningFlag, [&]() {
                std::ostringstream oss;
                oss << "Fixed primary position outside safe margin. Clamping to +/-"
                    << halfExtent / mm << " mm.";
                G4Exception("PrimaryGenerator::SamplePrimaryVertex",
                            "FixedPositionOutOfBounds",
                            JustWarning,
                            oss.str().c_str());
            });
        }

        return {targetX, targetY, z};
    }

    const G4double span = detSize - 2.0 * margin;
    const G4double randomX = G4UniformRand() * span - (detSize / 2.0 - margin);
    const G4double randomY = G4UniformRand() * span - (detSize / 2.0 - margin);

    return {randomX, randomY, z};
}

void PrimaryGenerator::ApplyParticlePosition(const G4ThreeVector& position)
{
    fParticleGun->SetParticlePosition(position);
}

void PrimaryGenerator::GenerateRandomPos()
{
    ApplyParticlePosition(SamplePrimaryVertex());
}

