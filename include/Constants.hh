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
    const G4int NEIGHBORHOOD_RADIUS = 2;                 // Default neighborhood radius for 9x9 grid
    
    // ========================
    // SIMULATION CONSTANTS
    // ========================
    
    // Step limiting
    const G4double MAX_STEP_SIZE = 5.0*micrometer;      // Maximum step size for tracking
        
    // Primary generator constants
    const G4double PRIMARY_PARTICLE_Z_POSITION = 0.0*cm; // Z position for primary particle generation
        
    // ========================
    // NUMERICAL TOLERANCE CONSTANTS
    // ========================
    
    // Numerical stability and solver tolerances
    const G4double MIN_UNCERTAINTY_VALUE = 1e-20;        // Minimum allowed uncertainty to prevent division by zero
    const G4double MIN_SAFE_PARAMETER = 1e-12;           // Minimum safe value for denominators and parameters
    const G4double MIN_LOG_VALUE = 1e-6;                 // Minimum allowed log value for numerical stability
    const G4double MIN_DENOMINATOR_VALUE = 1e-10;        // Minimum allowed denominator value

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
    // POST-RUN PROCESSING (Gaussian fits over the charge neighborhood)
    // ========================
    // Control whether to run ROOT post-processing macros automatically after the run finishes
    const G4bool RUN_PROCESSING_2D = true;   // Calls proc/processing2D.C
    const G4bool RUN_PROCESSING_3D = true;  // Calls proc/processing3D.C
    // Vertical uncertainty for the fits, as percent of the neighborhood max
    const G4double PROCESSING_ERROR_PERCENT = 5.0;

} // namespace Constants

#endif // CONSTANTS_HH
