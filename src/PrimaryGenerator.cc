#include "PrimaryGenerator.hh"

PrimaryGenerator::PrimaryGenerator()
{
    fParticleGun = new G4ParticleGun(1);

    // Particle Position - place it at a distance from the detector
    G4double x = 0.0*um;
    G4double y = 0.0*um;
    G4double z = 5.0*cm; // Place it further away from the detector for better visualization

    G4ThreeVector pos(x, y, z);
    G4double px = 0.0;
    G4double py = 0.0;
    G4double pz = -1.0;

    G4ThreeVector mom(px, py, pz);

    // Particle Type
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle("e-");

    fParticleGun->SetParticlePosition(pos);
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
    // Create Vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);
}

// Implementation of the new access method
G4ThreeVector PrimaryGenerator::GetParticlePosition() const
{
    return fParticleGun->GetParticlePosition();
}