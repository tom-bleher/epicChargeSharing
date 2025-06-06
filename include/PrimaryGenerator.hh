#ifndef PRIMARYGENERATOR_HH
#define PRIMARYGENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RandomTools.hh"

class DetectorConstruction;

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator();

    virtual void GeneratePrimaries(G4Event* anEvent);

private:
    G4ParticleGun* fParticleGun;
    DetectorConstruction* fDetector;
    
    // Method to generate random particle position
    void GenerateRandomPosition();
};

#endif