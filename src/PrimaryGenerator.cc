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

    // Particle Type - using electrons for AC-LGAD simulation
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition *particle = particleTable->FindParticle("e-");

    // Print information about position constraints for dynamic neighborhood analysis
    G4double detSize = fDetector->GetDetSize();
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    G4int neighborhoodRadius = fDetector->GetNeighborhoodRadius();
    G4double margin = neighborhoodRadius * pixelSpacing;
    G4double constrainedArea = detSize - 2.0 * margin;
    G4int gridSize = 2 * neighborhoodRadius + 1;
    
    G4cout << "\n=== PRIMARY GENERATOR CONSTRAINTS ===" << G4endl;
    G4cout << "Detector size: " << detSize/mm << " mm × " << detSize/mm << " mm" << G4endl;
    G4cout << "Pixel spacing: " << pixelSpacing/mm << " mm" << G4endl;
    G4cout << "Neighborhood radius: " << neighborhoodRadius << " (for " << gridSize << "×" << gridSize << " analysis)" << G4endl;
    G4cout << "Required margin: " << margin/mm << " mm on each side" << G4endl;
    G4cout << "Constrained generation area: " << constrainedArea/mm << " mm × " << constrainedArea/mm << " mm" << G4endl;
    if (constrainedArea > 0) {
        G4double areaRatio = (constrainedArea * constrainedArea) / (detSize * detSize);
        G4cout << "Area utilization: " << areaRatio * 100.0 << "%" << G4endl;
    }
    G4cout << "====================================" << G4endl;

    // Initial position will be set randomly in GenerateRandomPosition()
    GenerateRandomPosition();
    
    fParticleGun->SetParticleMomentumDirection(mom);
    // Use more realistic energy for AC-LGAD testing (typical MIP energy)
    fParticleGun->SetParticleEnergy(1.0*MeV); // Realistic MIP energy instead of 12 GeV
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
        // The vertex has been created and can be accessed by EventAction
        // No additional code needed here as EventAction already reads the position
    }
}

// Implementation of the new access method
G4ThreeVector PrimaryGenerator::GetParticlePosition() const
{
    return fParticleGun->GetParticlePosition();
}

// Generate random position within the detector area
// Constrained to ensure complete pixel neighborhoods stay within detector bounds
void PrimaryGenerator::GenerateRandomPosition()
{
    // Get detector size and neighborhood radius from the detector construction
    G4double detSize = fDetector->GetDetSize();
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    G4int neighborhoodRadius = fDetector->GetNeighborhoodRadius();
    
    // Calculate margin needed to ensure complete neighborhood stays within bounds
    // The neighborhood extends neighborhoodRadius pixels in each direction from the center
    G4double margin = neighborhoodRadius * pixelSpacing;
    
    // Calculate limits for x and y (detector is centered at origin)
    G4double halfSize = detSize / 2.0;
    G4double xmin = -halfSize + margin;
    G4double xmax = halfSize - margin;
    G4double ymin = -halfSize + margin;
    G4double ymax = halfSize - margin;
    
    G4int gridSize = 2 * neighborhoodRadius + 1;
    
    // Check if we have valid range after applying margin
    if (xmax <= xmin || ymax <= ymin) {
        G4cerr << "ERROR: Detector too small for " << gridSize << "×" << gridSize << " neighborhood analysis!" << G4endl;
        G4cerr << "Detector size: " << detSize/mm << " mm" << G4endl;
        G4cerr << "Required margin: " << margin/mm << " mm on each side" << G4endl;
        G4cerr << "Minimum detector size needed: " << (2*margin)/mm << " mm" << G4endl;
        
        // Fall back to original behavior with warning
        G4cout << "WARNING: Using full detector area - some " << gridSize << "×" << gridSize << " neighborhoods may extend outside bounds" << G4endl;
        xmin = -halfSize;
        xmax = halfSize;
        ymin = -halfSize;
        ymax = halfSize;
    }
    
    // Generate random position within the constrained area
    G4double x = G4UniformRand() * (xmax - xmin) + xmin;
    G4double y = G4UniformRand() * (ymax - ymin) + ymin;
    
    // Fixed z position in front of the detector
    G4double z = 2.0*cm; // Place source close enough for good statistics
    
    // Set the new position
    G4ThreeVector pos(x, y, z);
    fParticleGun->SetParticlePosition(pos);
}