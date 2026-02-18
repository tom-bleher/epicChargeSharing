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

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"

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
class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
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
    struct SamplingWindow
    {
        G4double margin{0.0};      ///< Safe margin from detector edge
        G4double halfExtent{0.0};  ///< Half-extent of sampling region
    };

    const SamplingWindow& EnsureSamplingWindow() const;
    G4double ClampToWindow(G4double value, const SamplingWindow& window) const;
    G4ThreeVector MakeFixedPosition(const SamplingWindow& window) const;
    G4ThreeVector MakeRandomPosition(const SamplingWindow& window) const;
    G4ThreeVector SamplePrimaryVertex() const;
    void ApplyParticlePosition(const G4ThreeVector& position);
    void GenerateRandomPos();

    std::unique_ptr<G4ParticleGun> fParticleGun;
    DetectorConstruction* fDetector{nullptr};

    std::unique_ptr<G4GenericMessenger> fMessenger;
    G4bool fUseFixedPosition{Constants::USE_FIXED_POSITION};
    G4double fFixedX{0.0};
    G4double fFixedY{0.0};
    mutable std::once_flag fFixedPositionWarningFlag;
    mutable SamplingWindow fSamplingWindow;
    mutable G4bool fSamplingWindowValid{false};
};

} // namespace ECS

// Backward compatibility alias
using PrimaryGenerator = ECS::PrimaryGenerator;

#endif // ECS_PRIMARY_GENERATOR_HH
