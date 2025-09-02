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

    // Momentum toward detector
    G4ThreeVector mom(0.0, 0.0, -1.0);

    // Particle type from Control.hh
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle(Control::PARTICLE_TYPE);
    if (!particle) {
        G4cerr << "ERROR: Unknown particle type: " << Control::PARTICLE_TYPE << G4endl;
        G4cerr << "Falling back to electron" << G4endl;
        particle = particleTable->FindParticle("e-");
    }
    
    fParticleGun->SetParticleDefinition(particle);
    
    // Particle energy from Control.hh
    fParticleGun->SetParticleEnergy(Control::PARTICLE_ENERGY * GeV);
    
    // Momentum direction
    fParticleGun->SetParticleMomentumDirection(mom);
    
    G4cout << "\nParticle gun (header-configured)" << G4endl;
    G4cout << "  type: " << Control::PARTICLE_TYPE << ", E = " << Control::PARTICLE_ENERGY << " GeV" << G4endl;

    // Uniform within region guaranteeing full neighborhood
    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius(); // typically 4 for a 9x9 grid
    G4double margin   = fDetector->GetPixelCornerOffset() +                     // fixed edge offset
                        fDetector->GetPixelSize()/2 +                          // half-pixel to stay inside pad
                        radius * fDetector->GetPixelSpacing();                 // radius pads on each side

    // Guard: margin < detSize/2
    if (margin >= detSize/2) {
        G4Exception("PrimaryGenerator", "MarginTooLarge", FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    G4cout << "Neighborhood: full " << (2*radius+1) << "×" << (2*radius+1) << " guaranteed" << G4endl;
    G4cout << "  allowed XY: [" << (-detSize/2 + margin)/mm << ", "
           << (detSize/2 - margin)/mm << "] mm (margin " << margin/mm << " mm)" << G4endl;

    // Set initial position randomly
    GenerateRandomPos();
}

PrimaryGenerator::~PrimaryGenerator()
{
    delete fParticleGun;
}

void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    // New random position each event
    GenerateRandomPos();
    
    // Create vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);
}


void PrimaryGenerator::GenerateRandomPos()
{
    // Random position ensuring full neighborhood
    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius();
    G4double margin   = fDetector->GetPixelCornerOffset() +
                        fDetector->GetPixelSize()/2 +
                        radius * fDetector->GetPixelSpacing();

    // Uniform in reduced square
    G4double x = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
    G4double y = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
 
    // Fixed z in front of detector
    G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
 
    // Set particle position
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
} 