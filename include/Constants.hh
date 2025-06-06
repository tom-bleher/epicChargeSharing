#ifndef CONSTANTS_HH
#define CONSTANTS_HH

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "globals.hh"

namespace Constants {
    
    // ========================
    // DETECTOR GEOMETRY CONSTANTS
    // ========================
    
    // Default detector dimensions
    const G4double DEFAULT_DETECTOR_SIZE = 30*mm;        // 30 mm default detector size
    const G4double DEFAULT_DETECTOR_WIDTH = 0.05*mm;     // 50 microns thickness
    const G4double WORLD_SIZE = 5*cm;                    // World volume size
    const G4double DETECTOR_Z_POSITION = -1.0*cm;       // Fixed detector position
    
    // ========================
    // PIXEL GEOMETRY CONSTANTS
    // ========================
    
    // Parameters (all lengths are center–to–center except fPixelCornerOffset)
    
    // Default pixel dimensions and spacing
    const G4double DEFAULT_PIXEL_SIZE = 0.1*mm;          // "pixel" side‐length
    const G4double DEFAULT_PIXEL_WIDTH = 0.001*mm;       // Width/thickness of each pixel
    const G4double DEFAULT_PIXEL_SPACING = 0.5*mm;       // center–to–center pitch
    const G4double DEFAULT_PIXEL_CORNER_OFFSET = 0.1*mm; // from inner detector edge to first pixel
    
    // Pixel positioning precision
    const G4double GEOMETRY_TOLERANCE = 1*um;            // Geometry calculation tolerance
    const G4double PRECISION_TOLERANCE = 1*nm;           // High precision calculations
    
    // Grid and neighborhood configuration
    const G4int NEIGHBORHOOD_RADIUS = 4;                 // Default neighborhood radius for 9x9 grid
    
    // ========================
    // SIMULATION CONSTANTS
    // ========================
    
    // Step limiting
    const G4double MAX_STEP_SIZE = 10.0*micrometer;      // Maximum step size for tracking
    
    // Charge sharing calculations
    const G4double ALPHA_WEIGHT_MULTIPLIER = 1000.0;     // Weight for very close pixels
    
} // namespace Constants

#endif // CONSTANTS_HH 