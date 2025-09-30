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
    const G4double DETECTOR_SIZE = 30.0*mm;        // 30 mm default detector size
    const G4double DETECTOR_WIDTH = 0.05*mm;     // 50 microns thickness
    const G4double WORLD_SIZE = 5*cm;                    // World volume size
    const G4double DETECTOR_Z_POSITION = -1.0*cm;       // Fixed detector position
    
    // ========================
    // PIXEL GEOMETRY CONSTANTS
    // ========================
    
    // Parameters (all lengths are center–to–center except fPixelCornerOffset)
    
    // Default pixel dimensions and spacing
    const G4double PIXEL_SIZE = 0.1*mm;          // "pixel" side‐length
    const G4double PIXEL_WIDTH = 0.001*mm;       // Width/thickness of each pixel
    const G4double PIXEL_SPACING = 0.5*mm;       // center–to–center pitch
    const G4double PIXEL_CORNER_OFFSET = 0.1*mm; // from inner detector edge to first pixel
    
    // Pixel positioning precision
    const G4double GEOMETRY_TOLERANCE = 1*um;            // Geometry calculation tolerance
    const G4double PRECISION_TOLERANCE = 1*nm;           // High precision calculations
    
    // Grid and neighborhood configuration
    const G4int NEIGHBORHOOD_RADIUS = 2;                 // Neighborhood radius r (grid size (2r+1)x(2r+1); r=2 -> 5x5)
    
    // ========================
    // SIMULATION CONSTANTS
    // ========================
    
    // Step limiting
    const G4double MAX_STEP_SIZE = 20.0*micrometer;      // Maximum step size for tracking
        
    // Primary generator constants
    const G4double PRIMARY_PARTICLE_Z_POSITION = 0.0*cm; // Z position for primary particle generation
        
    // Neighborhood grid special markers
    const G4double OUT_OF_BOUNDS_FRACTION_SENTINEL = -999.0; // Marks F_i as invalid (out-of-bounds)
    
    // ========================
    // PHYSICS CONSTANTS
    // ========================
    
    // AC-LGAD physics parameters
    const G4double IONIZATION_ENERGY = 3.6;              // eV per electron-hole pair in silicon
    const G4double AMPLIFICATION_FACTOR = 20.0;          // AC-LGAD amplification factor (N_e' = N_e * 20)
    const G4double D0_CHARGE_SHARING = 1.0;             // x displacement in silicon (microns)
    const G4double ELEMENTARY_CHARGE = 1.602176634e-19;  // Elementary charge in Coulombs

    // ========================
    // NOISE MODEL CONSTANTS
    // ========================
    // Per-pixel multiplicative gain noise sigma range (dimensionless)
    const G4double PIXEL_GAIN_SIGMA_MIN = 0.010;          // lower bound for sigma in Gauss(1, sigma) - 1%
    const G4double PIXEL_GAIN_SIGMA_MAX = 0.050;          // upper bound for sigma in Gauss(1, sigma) - 5%
    // Additive electronic noise modeled as Gaussian with sigma1 = N_e * q_e [C]
    const G4double NOISE_ELECTRON_COUNT = 500.0;         // baseline number of noise electrons

    // ========================
    // POST-RUN PROCESSING (Gaussian fits over the charge neighborhood)
    // ========================
    // Control whether to run ROOT post-processing macros automatically after the run finishes
    const G4bool RUN_PROCESSING_2D = true;   // Calls proc/processing2D.C
    const G4bool RUN_PROCESSING_3D = true;  // Calls proc/processing3D.C

} // namespace Constants

#endif // CONSTANTS_HH
