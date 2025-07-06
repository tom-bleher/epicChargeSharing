#include "PrimaryGenerator.hh"
#include "DetectorConstruction.hh"
#include "Constants.hh"
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

    // Particle Type - using electrons for AC-LGAD simulation
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle("e-");

    // Calculate the full detector bounds
    CalculateDetectorBounds();
    
    // Print information about the shooting region
    G4cout << "\n=== PARTICLE GUN COVERING FULL DETECTOR ===" << G4endl;
    G4cout << "Full detector bounds:" << G4endl;
    G4cout << "  X range: [" << fDetectorXmin/mm << ", " << fDetectorXmax/mm << "] mm" << G4endl;
    G4cout << "  Y range: [" << fDetectorYmin/mm << ", " << fDetectorYmax/mm << "] mm" << G4endl;
    G4cout << "  Detector size: " << (fDetectorXmax-fDetectorXmin)/mm << " × " << (fDetectorYmax-fDetectorYmin)/mm << " mm²" << G4endl;
    G4cout << "All particles will be shot within the full detector area." << G4endl;
    G4cout << "=============================================" << G4endl;

    // Initial position will be set randomly in GenerateRandomPosition()
    GenerateRandomPosition();
    
    fParticleGun->SetParticleMomentumDirection(mom);
    fParticleGun->SetParticleEnergy(0.1*MeV); // Realistic MIP energy
    fParticleGun->SetParticleDefinition(particle);
}

PrimaryGenerator::~PrimaryGenerator()
{
    delete fParticleGun;
}

void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    // Generate a new random position for each event within the central pixel region
    GenerateRandomPosition();
    
    // Create Vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);
}

void PrimaryGenerator::CalculateDetectorBounds()
{
    // Get detector parameters
    G4double detSize = fDetector->GetDetSize();
    
    // Calculate the full detector bounds
    // The detector extends from -detSize/2 to +detSize/2 in both X and Y
    fDetectorXmin = -detSize/2;
    fDetectorXmax = +detSize/2;
    fDetectorYmin = -detSize/2;
    fDetectorYmax = +detSize/2;
}

void PrimaryGenerator::GenerateRandomPosition()
{
    // Generate random position anywhere within the full detector bounds
    G4double x = G4UniformRand() * (fDetectorXmax - fDetectorXmin) + fDetectorXmin;
    G4double y = G4UniformRand() * (fDetectorYmax - fDetectorYmin) + fDetectorYmin;
    
    // Fixed z position in front of the detector
    G4double z = Constants::PRIMARY_PARTICLE_Z_POSITION;
    
    // Set the particle position
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
} 