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
    PrimaryGenerator(DetectorConstruction* detector);
    ~PrimaryGenerator();

    virtual void GeneratePrimaries(G4Event*);

private:
    G4ParticleGun* fParticleGun;
    DetectorConstruction* fDetector;
    
    // Central pixel assignment region boundaries (yellow square)
    G4double fCentralRegionXmin;
    G4double fCentralRegionXmax;
    G4double fCentralRegionYmin;
    G4double fCentralRegionYmax;
    
    void CalcCentralPixelRegion();
    void GenerateRandomPos();
};

#endif