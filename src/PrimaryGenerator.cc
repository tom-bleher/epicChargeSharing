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

    // Particle momentum direction - pointing toward the detector
    G4ThreeVector mom(0.0, 0.0, -1.0);

    // Set particle type from Control.hh (this overrides any macro file commands)
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle(Control::PARTICLE_TYPE);
    if (!particle) {
        G4cerr << "ERROR: Unknown particle type: " << Control::PARTICLE_TYPE << G4endl;
        G4cerr << "Falling back to electron" << G4endl;
        particle = particleTable->FindParticle("e-");
    }
    
    fParticleGun->SetParticleDefinition(particle);
    
    // Set particle energy from Control.hh (this overrides any macro file commands)
    fParticleGun->SetParticleEnergy(Control::PARTICLE_ENERGY * GeV);
    
    // Set particle momentum direction
    fParticleGun->SetParticleMomentumDirection(mom);
    
    G4cout << "\n=== PARTICLE GUN CONFIGURED FROM HEADER FILES ===" << G4endl;
    G4cout << "Particle type: " << Control::PARTICLE_TYPE << G4endl;
    G4cout << "Particle energy: " << Control::PARTICLE_ENERGY << " GeV" << G4endl;

    // Particle gun now shoots uniformly across the entire detector surface
    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius(); // typically 4 for a 9x9 grid
    G4double margin   = fDetector->GetPixelCornerOffset() +                     // fixed edge offset
                        fDetector->GetPixelSize()/2 +                          // half-pixel to stay inside pad
                        radius * fDetector->GetPixelSpacing();                 // radius pads on each side

    // Safety guard: ensure margin is smaller than half detector
    if (margin >= detSize/2) {
        G4Exception("PrimaryGenerator", "MarginTooLarge", FatalException,
                    "Neighborhood radius larger than detector allows.");
    }

    G4cout << "Full " << (2*radius+1) << "x" << (2*radius+1)
           << " neighbourhood guarantee enabled" << G4endl;
    G4cout << "Allowed XY range inside detector: [" << (-detSize/2 + margin)/mm << ", "
           << (detSize/2 - margin)/mm << "] mm" << G4endl;
    G4cout << "(Margin from edges: " << margin/mm << " mm)" << G4endl;
    G4cout << "===============================================" << G4endl;

    // Initial position will be set randomly on the detector surface
    GenerateRandomPos();
}

PrimaryGenerator::~PrimaryGenerator()
{
    delete fParticleGun;
}

void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    // Generate a new random position for each event within the central pixel region
    GenerateRandomPos();
    
    // Create Vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);
}


void PrimaryGenerator::GenerateRandomPos()
{
    // Generate random position ensuring the chosen pixel has a full neighbourhood
    G4double detSize = fDetector->GetDetSize();
    G4int radius      = fDetector->GetNeighborhoodRadius();
    G4double margin   = fDetector->GetPixelCornerOffset() +
                        fDetector->GetPixelSize()/2 +
                        radius * fDetector->GetPixelSpacing();

    // Uniform in the reduced square
    G4double x = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
    G4double y = (G4UniformRand() * (detSize - 2.0*margin)) - (detSize/2 - margin);
 
    // Fixed z position in front of the detector
    G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
 
    // Set the particle position
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
} 