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
    
    // Default pixel dimensions and spacing
    const G4double DEFAULT_PIXEL_SIZE = 0.1*mm;          // 100 microns pixel size
    const G4double DEFAULT_PIXEL_WIDTH = 0.001*mm;       // 1 micron pixel thickness
    const G4double DEFAULT_PIXEL_SPACING = 0.5*mm;       // 500 microns center-to-center
    const G4double DEFAULT_PIXEL_CORNER_OFFSET = 0.1*mm; // 100 microns from detector edge
    const G4double PIXEL_EXCLUSION_BUFFER = 0.01*mm;     // Legacy pixel buffer (not used for d0 constraint)
    
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
    
    // Error estimation
    const G4double BASE_POSITION_ERROR = 0.005*mm;       // 5 microns base uncertainty
    const G4double ERROR_SCALE_FRACTION = 0.1;           // 10% error scaling
    
    // ========================
    // UNIT CONVERSIONS
    // ========================
    
    // Default values for initialization
    const G4double DEFAULT_ERROR = 1.0;                  // Default error when not specified
    const G4int INVALID_INDEX = -1;                      // Invalid pixel index
    const G4double INVALID_DISTANCE = -1.0;              // Invalid distance marker
    
    // ========================
    // OUTPUT AND DEBUGGING
    // ========================
    
    const G4int MAX_DEBUG_EVENTS = 5;                    // Maximum events to print debug info
    
} // namespace Constants

#endif // CONSTANTS_HH 