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
    const G4bool ENABLE_AUTO_RADIUS = false;             // Enable automatic radius selection per hit
    
    // Fit quality thresholds for radius selection
    const G4double MIN_FIT_QUALITY_THRESHOLD = 0.5;      // Minimum acceptable fit quality (0-1 scale)
    const G4double RESIDUAL_OUTLIER_THRESHOLD = 3.0;     // Outlier threshold in standard deviations
    const G4int MIN_POINTS_FOR_FIT = 3;                  // Minimum number of points required for fitting
    
    // ========================
    // SIMULATION CONSTANTS
    // ========================
    
    // Step limiting
    const G4double MAX_STEP_SIZE = 10.0*micrometer;      // Maximum step size for tracking
    
    // Charge sharing calculations
    const G4double ALPHA_WEIGHT_MULTIPLIER = 1000.0;     // Weight for very close pixels
    
    // ========================
    // FITTING MODEL CONTROL FLAGS
    // ========================
    
    // Enable/disable different fitting models for performance and flexibility
    // Set to 'false' to disable a fitting model and skip its computation entirely
    // This reduces simulation time and ROOT output file size when specific models aren't needed
    const G4bool ENABLE_GAUSSIAN_FITTING = true;         // Enable Gaussian fitting (2D and diagonal)
    const G4bool ENABLE_LORENTZIAN_FITTING = true;       // Enable Lorentzian fitting (2D and diagonal) 
    const G4bool ENABLE_SKEWED_LORENTZIAN_FITTING = false; // Enable Skewed Lorentzian fitting (2D and diagonal)
    
    // Individual fitting type control (only used if main model is enabled)
    const G4bool ENABLE_2D_FITTING = true;               // Enable central row/column fitting
    const G4bool ENABLE_DIAGONAL_FITTING = true;         // Enable diagonal fitting
    
    // ========================
    // VERTICAL CHARGE UNCERTAINTIES CONTROL
    // ========================
    
    // Enable/disable vertical charge uncertainties in fitting and ROOT output
    // When enabled: Uses 5% of max charge as uncertainty for weighted least squares fitting
    // When disabled: Uses uniform weighting (uncertainty = 1.0) for unweighted fitting
    // Also controls whether uncertainty values are saved to ROOT file branches
    const G4bool ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = false;  // Enable charge uncertainties
    
    // USAGE EXAMPLES:
    // - To disable all Skewed Lorentzian: set ENABLE_SKEWED_LORENTZIAN_FITTING = false
    // - To enable only 2D fits (not diagonals): set ENABLE_DIAGONAL_FITTING = false  
    // - To run only Gaussian fits: set ENABLE_LORENTZIAN_FITTING and ENABLE_SKEWED_LORENTZIAN_FITTING to false
    // - To disable charge uncertainties: set ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = false
    
} // namespace Constants

// ================================================================================================
// HORIZONTAL ERROR AND WEIGHTING CONFIGURATION CONSTANTS
// ================================================================================================
// Configuration for advanced horizontal error techniques and spatial uncertainty modeling
// Implements the five core techniques described in: 
// "Weighting Residuals and Horizontal Errors for Pixel Detector Reconstruction"

namespace HorizontalErrorConfig {
    // ===========================
    // TECHNIQUE #1: CENTRAL PIXEL DOWNWEIGHTING
    // ===========================
    // Assign larger uncertainty to highest-charge pixel to reduce its dominance
    constexpr double GAUSSIAN_CENTRAL_WEIGHT_FACTOR = 0.08;      // Most aggressive (8%) for sharpest profiles
    constexpr double LORENTZIAN_CENTRAL_WEIGHT_FACTOR = 0.10;    // Moderate (10%) for wider tails
    
    constexpr double CENTRAL_DOWNWEIGHT_THRESHOLD = 1.8;         // Charge concentration threshold
    constexpr double MAX_CENTRAL_PIXEL_UNCERTAINTY = 10.0;      // Maximum uncertainty multiplier
    
    // ===========================
    // TECHNIQUE #2: DISTANCE-BASED WEIGHTS
    // ===========================
    // w_i ∝ 1/(1 + d_i/d₀) - give farther pixels more influence on position
    constexpr double GAUSSIAN_DISTANCE_SCALE_D0 = 30.0;         // Tightest scaling for Gaussian [μm]
    constexpr double LORENTZIAN_DISTANCE_SCALE_D0 = 40.0;       // Medium scaling for Lorentzian [μm]
    
    constexpr double DISTANCE_WEIGHT_CAP = 8.0;                 // Maximum distance-based weight multiplier
    
    // ===========================
    // TECHNIQUE #3: ROBUST LOSS FUNCTIONS
    // ===========================
    // Employ Cauchy/Huber losses to moderate single pixel influence
    constexpr double GAUSSIAN_ROBUST_THRESHOLD = 0.06;          // Most aggressive for Gaussian
    constexpr double LORENTZIAN_ROBUST_THRESHOLD = 0.10;        // Moderate for Lorentzian
    
    constexpr double DYNAMIC_LOSS_THRESHOLD = 2.0;              // Threshold for dynamic switching [sigma]
    
    // ===========================
    // TECHNIQUE #4: PIXEL INTEGRATION
    // ===========================
    // Model pixel response by integrating over pixel area instead of point sampling
    constexpr double HORIZONTAL_ERROR_SCALE = 0.6;              // Scale factor for horizontal errors
    constexpr double SPATIAL_UNCERTAINTY_FACTOR = 0.5;         // Pixel size uncertainty factor
    constexpr double PIXEL_EDGE_UNCERTAINTY_FACTOR = 1.0;      // Additional edge uncertainty
    
    // ===========================
    // TECHNIQUE #5: SPATIAL ERROR MAPS
    // ===========================
    // Position-dependent weighting based on reconstruction error maps
    constexpr double SPATIAL_ERROR_MAP_STRENGTH = 0.3;         // Strength of spatial error corrections
    constexpr double POSITION_DEPENDENT_BIAS_SCALE = 0.25;     // Scale for position-dependent bias
    
    // ===========================
    // ADVANCED FEATURES
    // ===========================
    constexpr double EDGE_PIXEL_BOOST_FACTOR = 2.0;            // Boost factor for edge pixels
    constexpr double CHARGE_UNCERTAINTY_FLOOR = 0.02;          // Minimum relative charge uncertainty
    constexpr double SYSTEMATIC_BIAS_STRENGTH = 0.4;           // Strength of systematic bias correction
    constexpr double CORRELATION_RADIUS = 1.5;                 // Inter-pixel correlation radius [pixels]
    constexpr double ADAPTIVE_WEIGHT_UPDATE_RATE = 0.7;        // Rate for iterative weight updates
    
    // ===========================
    // DISTRIBUTION-SPECIFIC TUNING
    // ===========================
    // Fine-tuning factors for different profile shapes
    constexpr double GAUSSIAN_SHARPNESS_FACTOR = 1.0;          // Reference for sharp profiles
    constexpr double LORENTZIAN_WIDTH_FACTOR = 0.8;            // Gentler for wider tails
}

// Convenience functions for applying horizontal error configurations
namespace HorizontalErrorUtils {
    
    // Calculate horizontal uncertainty for a specific pixel based on position and charge
    inline double CalculatePixelHorizontalUncertainty(
        double pixel_x, double pixel_charge, double center_estimate, 
        double pixel_spacing, double mean_charge, 
        const std::string& distribution_type = "GAUSSIAN") {
        
        // Base horizontal error from pixel size
        double horizontal_error = HorizontalErrorConfig::HORIZONTAL_ERROR_SCALE * pixel_spacing;
        
        // Distance-dependent uncertainty
        double dx = std::abs(pixel_x - center_estimate);
        double pixel_edge_distance = pixel_spacing / 2.0;
        
        if (dx > pixel_edge_distance) {
            double edge_factor = 1.0 + (dx - pixel_edge_distance) / pixel_spacing;
            
            // Apply distribution-specific scaling
            if (distribution_type == "LORENTZIAN") {
                edge_factor *= HorizontalErrorConfig::LORENTZIAN_WIDTH_FACTOR;
            }
            
            horizontal_error *= edge_factor;
        }
        
        // Charge-dependent uncertainty (higher charge = more precise position)
        double charge_factor = 1.0 / std::sqrt(std::max(0.1, pixel_charge / mean_charge));
        horizontal_error *= charge_factor;
        
        return horizontal_error;
    }
    
    // Get central pixel weight factor for a specific distribution
    inline double GetCentralPixelWeightFactor(const std::string& distribution_type) {
        if (distribution_type == "GAUSSIAN") {
            return HorizontalErrorConfig::GAUSSIAN_CENTRAL_WEIGHT_FACTOR;
        } else if (distribution_type == "LORENTZIAN") {
            return HorizontalErrorConfig::LORENTZIAN_CENTRAL_WEIGHT_FACTOR;
        }
        return HorizontalErrorConfig::GAUSSIAN_CENTRAL_WEIGHT_FACTOR; // Default
    }
    
    // Get distance scale d₀ for a specific distribution
    inline double GetDistanceScaleD0(const std::string& distribution_type) {
        if (distribution_type == "GAUSSIAN") {
            return HorizontalErrorConfig::GAUSSIAN_DISTANCE_SCALE_D0;
        } else if (distribution_type == "LORENTZIAN") {
            return HorizontalErrorConfig::LORENTZIAN_DISTANCE_SCALE_D0;
        }
        return HorizontalErrorConfig::GAUSSIAN_DISTANCE_SCALE_D0; // Default
    }
    
    // Get robust loss threshold for a specific distribution
    inline double GetRobustThreshold(const std::string& distribution_type) {
        if (distribution_type == "GAUSSIAN") {
            return HorizontalErrorConfig::GAUSSIAN_ROBUST_THRESHOLD;
        } else if (distribution_type == "LORENTZIAN") {
            return HorizontalErrorConfig::LORENTZIAN_ROBUST_THRESHOLD;
        }
        return HorizontalErrorConfig::GAUSSIAN_ROBUST_THRESHOLD; // Default
    }
}

// Enhanced horizontal error correction parameters for all fitting methods
// These implement the five key techniques requested for spatial uncertainty and bias reduction
namespace HorizontalErrorConstants {
    // Core horizontal error techniques - enable flags
    constexpr bool ENABLE_CENTRAL_PIXEL_DOWNWEIGHT = true;      // Technique #1: Downweight central pixel
    constexpr bool ENABLE_DISTANCE_BASED_WEIGHTS = true;        // Technique #2: Distance-based weighting  
    constexpr bool ENABLE_ROBUST_LOSSES = true;                 // Technique #3: Robust loss functions
    constexpr bool ENABLE_PIXEL_INTEGRATION = true;             // Technique #4: Pixel integration model
    constexpr bool ENABLE_SPATIAL_ERROR_MAPS = true;            // Technique #5: Spatial error maps
    
    // Technique #1: Central pixel downweighting parameters
    // These give the highest-charge pixel larger assumed uncertainty to reduce its pull on the fit
    constexpr double GAUSSIAN_CENTRAL_PIXEL_WEIGHT_FACTOR = 0.08;    // Reduce to 8% (most aggressive for sharp peak)
    constexpr double LORENTZIAN_CENTRAL_PIXEL_WEIGHT_FACTOR = 0.10;  // Reduce to 10% (moderate for wider tails)
    
    constexpr double CENTRAL_DOWNWEIGHT_THRESHOLD = 2.0;             // Charge concentration threshold for activation
    constexpr double MAX_CENTRAL_PIXEL_UNCERTAINTY = 8.0;           // Maximum uncertainty multiplier
    
    // Technique #2: Distance-based weighting parameters  
    // w_i ∝ 1/(1 + d_i/d₀) where d_i is distance from current estimate
    constexpr double GAUSSIAN_DISTANCE_SCALE_D0 = 30.0;         // Tightest for sharp Gaussian [μm]
    constexpr double LORENTZIAN_DISTANCE_SCALE_D0 = 40.0;       // Moderate for wider Lorentzian [μm]
    
    constexpr double DISTANCE_WEIGHT_CAP = 8.0;                 // Maximum weight multiplier
    
    // Technique #3: Robust loss parameters
    // Cauchy/Huber losses to moderate single pixel influence
    constexpr double GAUSSIAN_ROBUST_THRESHOLD_FACTOR = 0.06;   // Most aggressive for sharp peaks
    constexpr double LORENTZIAN_ROBUST_THRESHOLD_FACTOR = 0.10; // Moderate for wider tails
    
    constexpr double DYNAMIC_LOSS_THRESHOLD = 2.0;              // Sigma threshold for switching
    
    // Technique #4: Pixel integration parameters
    // Model pixel response by integrating over pixel area instead of point evaluation
    constexpr double HORIZONTAL_ERROR_SCALE = 0.6;              // Scale factor for horizontal errors
    constexpr double SPATIAL_UNCERTAINTY_FACTOR = 0.5;         // Pixel size uncertainty factor
    constexpr double PIXEL_EDGE_UNCERTAINTY_FACTOR = 1.0;      // Additional edge uncertainty
    
    // Technique #5: Spatial error map parameters
    // Position-dependent weighting based on reconstruction error patterns
    constexpr double SPATIAL_ERROR_MAP_STRENGTH = 0.3;         // Strength of spatial corrections
    constexpr double POSITION_DEPENDENT_BIAS_SCALE = 0.25;     // Bias correction scale
    
    // Advanced weighting features for enhanced performance
    constexpr double EDGE_PIXEL_BOOST_FACTOR = 2.0;            // Boost factor for edge pixels
    constexpr double CHARGE_UNCERTAINTY_FLOOR = 0.02;          // Minimum relative uncertainty (2%)
    constexpr double SYSTEMATIC_BIAS_STRENGTH = 0.4;           // Systematic bias correction strength
    constexpr double CORRELATION_RADIUS = 1.5;                 // Inter-pixel correlation radius [pixels]
    constexpr double ADAPTIVE_WEIGHT_UPDATE_RATE = 0.7;        // Rate for iterative updates
}

#endif // CONSTANTS_HH 