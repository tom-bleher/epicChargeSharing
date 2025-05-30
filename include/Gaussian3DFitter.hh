#ifndef GAUSSIAN3DFITTER_HH
#define GAUSSIAN3DFITTER_HH

#include "G4Types.hh"
#include "G4ThreeVector.hh"
#include <vector>

// ROOT includes for fitting (minimal to avoid Minuit issues)
#include "TMath.h"

/**
 * @brief 3D Gaussian fitting class for charge distribution data
 * 
 * This class implements 3D Gaussian fitting functionality similar to the Python 
 * ODR (Orthogonal Distance Regression) version using custom optimization.
 * It fits a 3D Gaussian function to charge distribution data from the simulation.
 */
class Gaussian3DFitter {
public:
    /**
     * @brief Enumeration for fit types
     */
    enum FitType {
        ALL_DATA = 0        // Fit with all data (no outlier removal)
        // OUTLIER_CLEANED = 1  // Fit with outlier detection and removal - COMMENTED OUT
    };

    /**
     * @brief Structure to hold detector geometry constraints
     */
    struct DetectorGeometry {
        G4double detector_size;        // Total detector size [mm]
        G4double pixel_size;          // Individual pixel size [mm] 
        G4double pixel_spacing;       // Center-to-center pixel spacing [mm]
        G4double pixel_corner_offset; // Offset from detector edge to first pixel [mm]
        G4int num_blocks_per_side;    // Number of pixels per side
        G4double pixel_exclusion_buffer; // Buffer around pixels to exclude [mm] (default: 0.01 mm = 10 microns)
        
        DetectorGeometry() : 
            detector_size(30.0), pixel_size(0.1), pixel_spacing(0.5), 
            pixel_corner_offset(0.1), num_blocks_per_side(59), pixel_exclusion_buffer(0.01) {}
    };

    /**
     * @brief Structure to hold fit results
     */
    struct FitResults {
        // Fit type identifier
        FitType fit_type;           // Type of fit performed
        
        // Fit parameters
        G4double amplitude;         // Peak height
        G4double x0;               // X center [mm]
        G4double y0;               // Y center [mm]
        G4double sigma_x;          // Standard deviation in X [mm]
        G4double sigma_y;          // Standard deviation in Y [mm]
        G4double theta;            // Rotation angle [radians]
        G4double offset;           // Background offset
        
        // Parameter errors
        G4double amplitude_err;
        G4double x0_err;
        G4double y0_err;
        G4double sigma_x_err;
        G4double sigma_y_err;
        G4double theta_err;
        G4double offset_err;
        
        // Fit statistics
        G4double chi2;             // Reduced chi-squared value (chi2/ndf)
        G4double ndf;              // Number of degrees of freedom
        G4double prob;             // Fit probability
        G4double r_squared;        // R-squared value
        G4int n_points;            // Number of data points used in fit
        // G4int n_outliers_removed;  // Number of outliers removed - COMMENTED OUT
        G4bool fit_successful;     // Whether fit converged successfully
        G4bool constraints_satisfied; // Whether all constraints are satisfied
        
        // Additional statistics
        G4double residual_mean;    // Mean of residuals
        G4double residual_std;     // Standard deviation of residuals
        
        // Robustness metrics
        G4double center_distance_from_detector_edge; // Distance from fit center to detector edge [mm]
        G4double min_distance_to_pixel;              // Minimum distance from fit center to any pixel [mm]
        G4int fit_attempt_number;                     // Which attempt succeeded (1-based)
        
        // Constructor to initialize with default values
        FitResults() : 
            fit_type(ALL_DATA),
            amplitude(0), x0(0), y0(0), sigma_x(0), sigma_y(0), theta(0), offset(0),
            amplitude_err(0), x0_err(0), y0_err(0), sigma_x_err(0), sigma_y_err(0), theta_err(0), offset_err(0),
            chi2(0), ndf(0), prob(0), r_squared(0), n_points(0), /* n_outliers_removed(0), */ 
            fit_successful(false), constraints_satisfied(false),
            residual_mean(0), residual_std(0),
            center_distance_from_detector_edge(0), min_distance_to_pixel(0), fit_attempt_number(0) {}
    };

    /**
     * @brief Constructor
     * @param detector_geometry Optional detector geometry constraints
     */
    Gaussian3DFitter(const DetectorGeometry& detector_geometry = DetectorGeometry());
    
    /**
     * @brief Destructor
     */
    ~Gaussian3DFitter();
    
    /**
     * @brief Set detector geometry constraints
     */
    void SetDetectorGeometry(const DetectorGeometry& geometry);
    
    /**
     * @brief Get current detector geometry
     */
    const DetectorGeometry& GetDetectorGeometry() const { return fDetectorGeometry; }
    
    /**
     * @brief Fit 3D Gaussian to charge distribution data with robustness enhancements
     * 
     * @param x_coords X coordinates of data points [mm]
     * @param y_coords Y coordinates of data points [mm]
     * @param z_values Charge values at each point
     * @param z_errors Uncertainties in charge values (optional)
     * @param verbose Whether to print fit progress and results
     * @return FitResults structure containing all fit information
     */
    /*
    FitResults FitGaussian3D(const std::vector<G4double>& x_coords,
                            const std::vector<G4double>& y_coords,
                            const std::vector<G4double>& z_values,
                            const std::vector<G4double>& z_errors = std::vector<G4double>(),
                            G4bool verbose = false);
    */
    
    /**
     * @brief Fit 3D Gaussian to charge distribution data WITHOUT outlier removal
     * 
     * @param x_coords X coordinates of data points [mm]
     * @param y_coords Y coordinates of data points [mm]
     * @param z_values Charge values at each point
     * @param z_errors Uncertainties in charge values (optional)
     * @param verbose Whether to print fit progress and results
     * @return FitResults structure containing all fit information (with fit_type = ALL_DATA)
     */
    FitResults FitGaussian3DAllData(const std::vector<G4double>& x_coords,
                                   const std::vector<G4double>& y_coords,
                                   const std::vector<G4double>& z_values,
                                   const std::vector<G4double>& z_errors = std::vector<G4double>(),
                                   G4bool verbose = false);
    
    /**
     * @brief Evaluate 3D Gaussian function at given coordinates
     * 
     * @param x X coordinate [mm]
     * @param y Y coordinate [mm]
     * @param params Array of 7 parameters [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
     * @return Gaussian value at (x,y)
     */
    static G4double Gaussian3DFunction(G4double x, G4double y, const G4double* params);
    
    /**
     * @brief ROOT function wrapper (for compatibility)
     */
    static G4double Gaussian3DFunctionWrapper(G4double* coords, G4double* params);

private:
    /**
     * @brief Calculate initial parameter estimates with multiple strategies
     */
    void CalculateInitialGuess(const std::vector<G4double>& x_coords,
                              const std::vector<G4double>& y_coords,
                              const std::vector<G4double>& z_values,
                              G4double* initialParams,
                              G4int strategy = 0);
    
    /**
     * @brief Remove outliers from data using robust statistical methods
     */
    /*
    void RemoveOutliers(std::vector<G4double>& x_coords,
                       std::vector<G4double>& y_coords,
                       std::vector<G4double>& z_values,
                       std::vector<G4double>& z_errors,
                       G4int& n_outliers_removed,
                       G4bool verbose = false);
    */
    
    /**
     * @brief Check if point is inside any pixel with buffer zone
     */
    G4bool IsPointInsidePixelZone(G4double x, G4double y) const;
    
    /**
     * @brief Calculate minimum distance from point to any pixel edge
     */
    G4double CalculateMinDistanceToPixel(G4double x, G4double y) const;
    
    /**
     * @brief Check if fit parameters satisfy all geometric constraints
     */
    G4bool CheckConstraints(const G4double* params, G4bool verbose = false) const;
    
    /**
     * @brief Apply parameter bounds during optimization
     */
    void ApplyParameterBounds(G4double* params) const;
    
    /**
     * @brief Calculate R-squared value
     */
    G4double CalculateRSquared(const std::vector<G4double>& x_coords,
                              const std::vector<G4double>& y_coords,
                              const std::vector<G4double>& z_values,
                              const G4double* fitParams);
    
    /**
     * @brief Calculate residual statistics
     */
    void CalculateResidualStats(const std::vector<G4double>& x_coords,
                               const std::vector<G4double>& y_coords,
                               const std::vector<G4double>& z_values,
                               const G4double* fitParams,
                               G4double& mean, G4double& std_dev);
    
    /**
     * @brief Calculate chi-squared value for given parameters
     */
    G4double CalculateChiSquared(const std::vector<G4double>& x_coords,
                                const std::vector<G4double>& y_coords,
                                const std::vector<G4double>& z_values,
                                const std::vector<G4double>& z_errors,
                                const G4double* params);
    
    /**
     * @brief Enhanced constrained chi-squared with penalty terms
     */
    G4double CalculateConstrainedChiSquared(const std::vector<G4double>& x_coords,
                                           const std::vector<G4double>& y_coords,
                                           const std::vector<G4double>& z_values,
                                           const std::vector<G4double>& z_errors,
                                           const G4double* params);
    
    /**
     * @brief Perform robust Nelder-Mead simplex optimization with constraints
     */
    void RobustSimplexFit(const std::vector<G4double>& x_coords,
                         const std::vector<G4double>& y_coords,
                         const std::vector<G4double>& z_values,
                         const std::vector<G4double>& z_errors,
                         G4double* params,
                         G4bool verbose);
    
    // Detector geometry constraints
    DetectorGeometry fDetectorGeometry;
    
    // Legacy ROOT fitting objects (kept for compatibility but not used)
    void* fGaussianFunction;  // Changed to void* to avoid ROOT dependencies
    void* fDataGraph;         // Changed to void* to avoid ROOT dependencies
    
    // Internal fitting parameters
    static const G4int fNParams = 7;  // Number of fit parameters
    
    // Robustness parameters
    static const G4int fMaxFitAttempts = 3;     // Maximum number of fitting attempts
    // static const G4double fOutlierThreshold;   // Z-score threshold for outlier removal - COMMENTED OUT
    static const G4double fConstraintPenalty;  // Penalty factor for constraint violations
};

#endif // GAUSSIAN3DFITTER_HH 