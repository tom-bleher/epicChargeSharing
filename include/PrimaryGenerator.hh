#ifndef PRIMARYGENERATOR_HH
#define PRIMARYGENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"

class DetectorConstruction;
class G4Event;

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

#endif