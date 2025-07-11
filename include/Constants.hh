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
    // AUTOMATIC RADIUS SELECTION CONSTANTS
    // ========================
    
    // Automatic radius selection parameters
    const G4int MIN_AUTO_RADIUS = 4;                     // Minimum radius for auto selection (5x5 grid)
    const G4int MAX_AUTO_RADIUS = 10;                     // Maximum radius for auto selection (13x13 grid)
    
    // Fit quality thresholds for radius selection
    const G4int MIN_POINTS_FOR_FIT = 3;                  // Minimum number of points required for fitting
    
    // ========================
    // SIMULATION CONSTANTS
    // ========================
    
    // Step limiting
    const G4double MAX_STEP_SIZE = 10.0*micrometer;      // Maximum step size for tracking
    
    // Charge sharing calculations
    const G4double ALPHA_WEIGHT_MULTIPLIER = 1000.0;     // Weight for very close pixels
    
    // Primary generator constants
    const G4double PRIMARY_PARTICLE_Z_POSITION = 2.0*cm; // Z position for primary particle generation
        
    // ========================
    // NUMERICAL TOLERANCE CONSTANTS
    // ========================
    
    // Numerical stability and solver tolerances
    const G4double MIN_UNCERTAINTY_VALUE = 1e-20;        // Minimum allowed uncertainty to prevent division by zero
    const G4double MIN_SAFE_PARAMETER = 1e-12;           // Minimum safe value for denominators and parameters
    const G4double MIN_LOG_VALUE = 1e-6;                 // Minimum allowed log value for numerical stability
    const G4double MIN_DENOMINATOR_VALUE = 1e-10;        // Minimum allowed denominator value
    
    // Ceres solver default tolerances
    const G4double DEFAULT_FUNCTION_TOLERANCE = 1e-12;   // Default function tolerance for Ceres
    const G4double DEFAULT_GRADIENT_TOLERANCE = 1e-12;   // Default gradient tolerance for Ceres  
    const G4double DEFAULT_PARAMETER_TOLERANCE = 1e-15;  // Default parameter tolerance for Ceres
    const G4double HIGH_PRECISION_TOLERANCE = 1e-15;     // High precision tolerance for critical fits
    const G4double MEDIUM_PRECISION_TOLERANCE = 1e-12;   // Medium precision tolerance
    const G4double LOW_PRECISION_TOLERANCE = 1e-10;      // Lower precision tolerance for fallback
    
    // ========================
    // PHYSICS CONSTANTS
    // ========================
    
    // AC-LGAD physics parameters
    const G4double IONIZATION_ENERGY = 3.6;              // eV per electron-hole pair in silicon
    const G4double AMPLIFICATION_FACTOR = 10.0;          // AC-LGAD amplification factor
    const G4double D0_CHARGE_SHARING = 10.0;             // d0 constant for charge sharing (microns)
    const G4double ELEMENTARY_CHARGE = 1.602176634e-19;  // Elementary charge in Coulombs

} // namespace Constants

#endif // CONSTANTS_HH
