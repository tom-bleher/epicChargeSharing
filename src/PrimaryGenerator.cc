#include "PrimaryGenerator.hh"
#include "DetectorConstruction.hh"
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

    // Calculate the central pixel assignment region (yellow square)
    CalculateCentralPixelRegion();
    
    // Print information about the shooting region
    G4cout << "\n=== PARTICLE GUN CONSTRAINED TO CENTRAL PIXEL ===" << G4endl;
    G4cout << "Central pixel assignment region (yellow square):" << G4endl;
    G4cout << "  X range: [" << fCentralRegionXmin/mm << ", " << fCentralRegionXmax/mm << "] mm" << G4endl;
    G4cout << "  Y range: [" << fCentralRegionYmin/mm << ", " << fCentralRegionYmax/mm << "] mm" << G4endl;
    G4cout << "  Region size: " << (fCentralRegionXmax-fCentralRegionXmin)/mm << " × " << (fCentralRegionYmax-fCentralRegionYmin)/mm << " mm²" << G4endl;
    G4cout << "All particles will be shot within this region only." << G4endl;
    G4cout << "=================================================" << G4endl;

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

void PrimaryGenerator::CalculateCentralPixelRegion()
{
    // Get detector parameters
    G4double detSize = fDetector->GetDetSize();
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    G4double pixelSize = fDetector->GetPixelSize();
    G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
    G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
    
    // Calculate central pixel indices (middle of the detector grid)
    G4int centralPixelI = numBlocksPerSide / 2;
    G4int centralPixelJ = numBlocksPerSide / 2;
    
    // Calculate first pixel center position
    G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
    
    // Calculate central pixel center position
    G4double centralPixelX = firstPixelPos + centralPixelI * pixelSpacing;
    G4double centralPixelY = firstPixelPos + centralPixelJ * pixelSpacing;
    
    // Define the assignment region for the central pixel (yellow square)
    // Any hit within this region will be assigned to the central pixel
    G4double halfSpacing = pixelSpacing / 2.0;
    fCentralRegionXmin = centralPixelX - halfSpacing;
    fCentralRegionXmax = centralPixelX + halfSpacing;
    fCentralRegionYmin = centralPixelY - halfSpacing;
    fCentralRegionYmax = centralPixelY + halfSpacing;
}

void PrimaryGenerator::GenerateRandomPosition()
{
    // Generate random position ONLY within the central pixel assignment region (yellow square)
    G4double x = G4UniformRand() * (fCentralRegionXmax - fCentralRegionXmin) + fCentralRegionXmin;
    G4double y = G4UniformRand() * (fCentralRegionYmax - fCentralRegionYmin) + fCentralRegionYmin;
    
    // Fixed z position in front of the detector
    G4double z = 2.0*cm;
    
    // Set the particle position
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
} 