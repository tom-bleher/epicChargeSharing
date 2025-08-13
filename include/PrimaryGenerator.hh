#ifndef PRIMARYGENERATOR_HH
#define PRIMARYGENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"

class DetectorConstruction;
class G4Event;

// Shoots the primary particle toward the detector; constrains
// emission position to ensure a full charge neighborhood around
// the nearest pixel-pad.
class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator();

    virtual void GeneratePrimaries(G4Event*);

private:
    G4ParticleGun* fParticleGun;
    DetectorConstruction* fDetector;
    
    void GenerateRandomPos();
};

#endif