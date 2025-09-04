/**
 * @file PrimaryGenerator.hh
 * @brief Declares `PrimaryGenerator`, responsible for configuring the particle gun
 *        and sampling primary vertex positions within a safe margin.
 */
#ifndef PRIMARYGENERATOR_HH
#define PRIMARYGENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

class DetectorConstruction;
class G4Event;
class G4ParticleGun;

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    explicit PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator() override;

    void GeneratePrimaries(G4Event*) override;

private:
    G4ParticleGun* fParticleGun;
    DetectorConstruction* fDetector;
    
    void GenerateRandomPos();
};

#endif // PRIMARYGENERATOR_HH