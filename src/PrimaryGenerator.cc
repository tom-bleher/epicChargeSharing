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

    G4cout << "\n=== PARTICLE GUN WITH FULL " << (2*radius+1) << "x" << (2*radius+1)
           << " NEIGHBOURHOOD GUARANTEE ===" << G4endl;
    G4cout << "Allowed XY range inside detector: [" << (-detSize/2 + margin)/mm << ", "
           << (detSize/2 - margin)/mm << "] mm" << G4endl;
    G4cout << "(Margin from edges: " << margin/mm << " mm)" << G4endl;
    G4cout << "===============================================================" << G4endl;

    // Initial position will be set randomly on the detector surface
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