#include "PrimaryGenerator.hh"
#include "DetectorConstruction.hh"
#include "Randomize.hh"
#include "G4Event.hh"

PrimaryGenerator::PrimaryGenerator(DetectorConstruction* detector)
: fDetector(detector)
{
    fParticleGun = new G4ParticleGun(1);

    // Particle momentum direction - pointing toward the detector
    G4double px = 0.0;
    G4double py = 0.0;
    G4double pz = -1.0;
    G4ThreeVector mom(px, py, pz);

    // Particle Type
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle("e-");

    // Initial position will be set randomly in GenerateRandomPosition()
    GenerateRandomPosition();
    
    fParticleGun->SetParticleMomentumDirection(mom);
    fParticleGun->SetParticleEnergy(2.0*GeV);
    fParticleGun->SetParticleDefinition(particle);
}

PrimaryGenerator::~PrimaryGenerator()
{
    delete fParticleGun;
}

void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    // Generate a new random position for each event
    GenerateRandomPosition();
    
    // Store the particle position in the event user info for later retrieval
    G4ThreeVector position = fParticleGun->GetParticlePosition();
    
    // Create Vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);

    // Set the primary vertex position in the event for EventAction to access
    if (anEvent->GetPrimaryVertex()) {
        // Use the actual vertex position which has been created
        // This is more reliable than using the gun position directly
    }
}

// Implementation of the new access method
G4ThreeVector PrimaryGenerator::GetParticlePosition() const
{
    return fParticleGun->GetParticlePosition();
}

// Generate random position within the detector area
void PrimaryGenerator::GenerateRandomPosition()
{
    // Get detector size from the detector construction
    G4double detSize = fDetector->GetDetSize();
    
    // Calculate limits for x and y (detector is centered at origin)
    G4double halfSize = detSize / 2.0;
    G4double xmin = -halfSize;
    G4double xmax = halfSize;
    G4double ymin = -halfSize;
    G4double ymax = halfSize;
    
    // Generate random position within the square
    G4double x = G4UniformRand() * (xmax - xmin) + xmin;
    G4double y = G4UniformRand() * (ymax - ymin) + ymin;
    
    // Fixed z position (in front of the detector)
    G4double z = 5.0*cm; // Place it away from the detector for visualization
    
    // Set the new position
    G4ThreeVector pos(x, y, z);
    fParticleGun->SetParticlePosition(pos);
}