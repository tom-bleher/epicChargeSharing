#include "PrimaryGenerator.hh"
#include "DetectorConstruction.hh"
#include "Constants.hh"
#include "Control.hh"
#include "Randomize.hh"
#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"

PrimaryGenerator::PrimaryGenerator(DetectorConstruction* detector)
: fDetector(detector)
{
    fParticleGun = new G4ParticleGun(1);

    G4ThreeVector mom(0.0, 0.0, -1.0);

    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle(Control::PARTICLE_TYPE);
    if (!particle) {
        G4cerr << "ERROR: Unknown particle type: " << Control::PARTICLE_TYPE << G4endl;
        G4cerr << "Falling back to electron" << G4endl;
        particle = particleTable->FindParticle("e-");
    }
    
    fParticleGun->SetParticleDefinition(particle);
    
    fParticleGun->SetParticleEnergy(Control::PARTICLE_ENERGY * GeV);
    
    fParticleGun->SetParticleMomentumDirection(mom);
    
    G4cout << "\nParticle gun (header-configured)" << G4endl;
    G4cout << "  type: " << Control::PARTICLE_TYPE << ", E = " << Control::PARTICLE_ENERGY << " GeV" << G4endl;

    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius(); // typically 4 for a 9x9 grid
    G4double margin   = fDetector->GetPixelCornerOffset() +                     // fixed edge offset
                        fDetector->GetPixelSize()/2 +                          // half-pixel to stay inside pad
                        radius * fDetector->GetPixelSpacing();                 // radius pads on each side

    if (margin >= detSize/2) {
        G4Exception("PrimaryGenerator", "MarginTooLarge", FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    G4cout << "Neighborhood: full " << (2*radius+1) << "×" << (2*radius+1) << " guaranteed" << G4endl;
    G4cout << "  allowed XY: [" << (-detSize/2 + margin)/mm << ", "
           << (detSize/2 - margin)/mm << "] mm (margin " << margin/mm << " mm)" << G4endl;

    GenerateRandomPos();
}

PrimaryGenerator::~PrimaryGenerator()
{
    delete fParticleGun;
}

void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    GenerateRandomPos();
    
    fParticleGun->GeneratePrimaryVertex(anEvent);
}


void PrimaryGenerator::GenerateRandomPos()
{
    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius();
    G4double margin   = fDetector->GetPixelCornerOffset() +
                        fDetector->GetPixelSize()/2 +
                        radius * fDetector->GetPixelSpacing();

    G4double x = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
    G4double y = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
 
    G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
 
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
} 