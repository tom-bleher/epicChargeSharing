/**
 * @file PrimaryGenerator.hh
 * @brief Declares `PrimaryGenerator`, responsible for configuring the particle gun
 *        and sampling primary vertex positions within a safe margin.
 */
#ifndef PRIMARYGENERATOR_HH
#define PRIMARYGENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"

#include <memory>
#include <mutex>

class DetectorConstruction;
class G4Event;
class G4ParticleGun;
class G4ParticleDefinition;
class G4GenericMessenger;

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    explicit PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator() override = default;

    void GeneratePrimaries(G4Event*) override;

private:
    void ConfigureParticleGun();
    void ConfigureMessenger();
    G4double ComputeSafeMargin() const;
    G4ThreeVector SamplePrimaryVertex() const;
    void ApplyParticlePosition(const G4ThreeVector& position);
    void GenerateRandomPos();

    std::unique_ptr<G4ParticleGun> fParticleGun;
    DetectorConstruction* fDetector;

    std::unique_ptr<G4GenericMessenger> fMessenger;
    G4bool fUseFixedPosition{true};
    G4double fFixedX{0.0};
    G4double fFixedY{0.0};
    mutable std::once_flag fFixedPositionWarningFlag;
};

#endif // PRIMARYGENERATOR_HH