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
    const G4double PIXEL_EXCLUSION_BUFFER = 0.01*mm;     // 10 microns exclusion buffer
    
    // Pixel positioning precision
    const G4double GEOMETRY_TOLERANCE = 1*um;            // Geometry calculation tolerance
    const G4double PRECISION_TOLERANCE = 1*nm;           // High precision calculations
    
    // ========================
    // FITTING CONSTANTS
    // ========================
    
    // Gaussian fitting parameters
    const G4int GAUSSIAN_N_PARAMS = 7;                   // Number of fit parameters
    const G4int MAX_FIT_ATTEMPTS = 3;                    // Maximum fitting attempts
    const G4double CONSTRAINT_PENALTY = 1000.0;          // Large penalty for violations
    const G4double PENALTY_MULTIPLIER = 1000.0;          // Additional penalty scaling
    
    // Sigma constraints
    const G4double MIN_SIGMA = 0.005*mm;                 // 5 microns minimum sigma
    const G4double MIN_SIGMA_ALT = 0.01*mm;              // 10 microns alternative minimum
    const G4double SIGMA_FRACTION_OF_DETECTOR = 0.25;    // Max sigma = detector_size/4
    const G4double SIGMA_FRACTION_ALT = 1.0/6.0;         // Alternative: detector_size/6
    
    // Initial parameter estimates
    const G4double DEFAULT_SIGMA_ESTIMATE = 0.1*mm;      // 100 microns default sigma
    const G4double DEFAULT_AMPLITUDE_FRACTION = 0.1;     // 10% of amplitude for step size
    
    // Fitting algorithm parameters
    const G4int MAX_ITERATIONS = 2000;                   // Maximum fitting iterations
    const G4double FIT_TOLERANCE = 1e-8;                 // Fitting convergence tolerance
    const G4int MIN_FIT_POINTS = 4;                      // Minimum points for fitting
    const G4int NEIGHBORHOOD_RADIUS = 4;                 // 9x9 grid radius
    
    // Parameter step sizes and bounds
    const G4double POSITION_STEP_SIZE = 0.01*mm;         // Position parameter step
    const G4double SIGMA_STEP_SIZE = 0.01*mm;            // Sigma parameter step
    const G4double ANGLE_STEP_SIZE = 0.1;                // Angle parameter step (radians)
    const G4double DEFAULT_STEP_SIZE = 0.05*mm;          // Default step size (50 microns)
    const G4double MIN_STEP_SIZE = 0.01*mm;              // Minimum step size
    const G4double MIN_STEP_SIZE_ALT = 1e-6*mm;          // Alternative minimum step
    const G4double AMPLITUDE_MIN = 0.001;                // Minimum amplitude
    const G4double AMPLITUDE_MAX = 1e6;                  // Maximum amplitude
    const G4double OFFSET_MAX = 1e6;                     // Maximum offset bound
    
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
    
    // Common unit aliases for readability
    const G4double MICRON = micrometer;
    const G4double UM = micrometer;
    
    // ========================
    // NUMERICAL CONSTANTS
    // ========================
    
    // Mathematical constants (in addition to those in G4PhysicalConstants)
    const G4double HALF = 0.5;
    const G4double QUARTER = 0.25;
    const G4double TWO = 2.0;
    
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