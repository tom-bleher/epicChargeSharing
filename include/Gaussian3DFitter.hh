#ifndef GAUSSIAN3DFITTER_HH
#define GAUSSIAN3DFITTER_HH

#include "G4Types.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "Constants.hh"
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
        ALL_DATA = 0        // Fit with all data
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
        G4double chi2red;             // Reduced chi-squared value (chi2red/ndf)
        G4double ndf;              // Number of degrees of freedom
        G4double Pp;             // Fit probability
        G4int n_points;            // Number of data points used in fit
        G4bool constraints_satisfied; // Whether all constraints are satisfied
        
        // Additional statistics
        G4double residual_mean;    // Mean of residuals
        G4double residual_std;     // Standard deviation of residuals
        
        // Constructor to initialize with default values
        FitResults() : 
            fit_type(ALL_DATA),
            amplitude(0), x0(0), y0(0), sigma_x(0), sigma_y(0), theta(0), offset(0),
            amplitude_err(0), x0_err(0), y0_err(0), sigma_x_err(0), sigma_y_err(0), theta_err(0), offset_err(0),
            chi2red(0), ndf(0), Pp(0), n_points(0), 
            constraints_satisfied(false),
            residual_mean(0), residual_std(0) {}
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
     * @brief Set the center pixel position for constraint enforcement
     * @param center_x X coordinate of center pixel [mm]
     * @param center_y Y coordinate of center pixel [mm]
     */
    void SetCenterPixelPosition(G4double center_x, G4double center_y);
    
    /**
     * @brief Set neighborhood radius for constraint calculation
     * @param radius Neighborhood radius (e.g., radius=4 for 9x9 grid)
     */
    void SetNeighborhoodRadius(G4int radius);
    
    /**
     * @brief Enable red area constraints mode
     * @param enable Whether to enable red area constraints
     */
    void SetUseRedAreaConstraints(G4bool enable);
    
    /**
     * @brief Fit 3D Gaussian to charge distribution data with quadrant-based red area optimization
     * 
     * @param x_coords X coordinates of data points [mm]
     * @param y_coords Y coordinates of data points [mm]
     * @param z_values Charge values at each point
     * @param z_errors Uncertainties in charge values (optional)
     * @param verbose Whether to print fit progress and results
     * @return FitResults structure containing all fit information
     */
    FitResults FitGaussian3DWithRedAreaConstraints(const std::vector<G4double>& x_coords,
                                                   const std::vector<G4double>& y_coords,
                                                   const std::vector<G4double>& z_values,
                                                   const std::vector<G4double>& z_errors = std::vector<G4double>(),
                                                   G4bool verbose = false);
    
    /**
     * @brief Fit 3D Gaussian to charge distribution data
     * 
     * @param x_coords X coordinates of data points [mm]
     * @param y_coords Y coordinates of data points [mm]
     * @param z_values Charge values at each point
     * @param z_errors Uncertainties in charge values (optional)
     * @param verbose Whether to print fit progress and results
     * @return FitResults structure containing all fit information
     */
    FitResults FitGaussian3D(const std::vector<G4double>& x_coords,
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

    /**
     * @brief Perform robust Nelder-Mead simplex optimization with constraints
     */
    void RobustSimplexFit(const std::vector<G4double>& x_coords,
                         const std::vector<G4double>& y_coords,
                         const std::vector<G4double>& z_values,
                         const std::vector<G4double>& z_errors,
                         G4double* params,
                         G4bool verbose);
    
    /**
     * @brief Perform robust simplex optimization with red area constraints
     */
    void RobustSimplexFitWithRedAreaConstraints(const std::vector<G4double>& x_coords,
                                               const std::vector<G4double>& y_coords,
                                               const std::vector<G4double>& z_values,
                                               const std::vector<G4double>& z_errors,
                                               G4double* params,
                                               G4bool verbose);
    
    /**
     * @brief Enhanced constrained chi-squared with red area penalty terms
     */
    G4double CalculateConstrainedChiSquaredWithRedArea(const std::vector<G4double>& x_coords,
                                                      const std::vector<G4double>& y_coords,
                                                      const std::vector<G4double>& z_values,
                                                      const std::vector<G4double>& z_errors,
                                                      const G4double* params);

    /**
     * @brief Generate initial guess within specified quadrant of red area
     * @param quadrant Quadrant number (0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right)
     * @param x_guess Output X coordinate guess [mm]
     * @param y_guess Output Y coordinate guess [mm]
     * @return True if valid guess found in quadrant
     */
    G4bool GenerateQuadrantGuess(G4int quadrant, G4double& x_guess, G4double& y_guess) const;
    
    /**
     * @brief Generate comprehensive set of initial guesses covering the red area efficiently
     * @param initial_guesses Vector to store initial guess coordinates (x, y pairs)
     * @param max_points Maximum number of initial guesses to generate
     * @return Number of valid initial guesses generated
     */
    G4int GenerateRedAreaSamples(std::vector<std::pair<G4double, G4double>>& initial_guesses, G4int max_points = 16) const;
    
    /**
     * @brief Generate systematic grid sampling of red area
     * @param samples Vector to store valid sample points
     * @param grid_resolution Resolution of the grid sampling
     * @return Number of valid samples found
     */
    G4int SampleRedAreaSystematically(std::vector<std::pair<G4double, G4double>>& samples, G4int grid_resolution = 20) const;
    
    /**
     * @brief Generate adaptive sampling focusing on promising regions of red area
     * @param samples Vector to store sample points
     * @param x_coords Data X coordinates for adaptive guidance
     * @param y_coords Data Y coordinates for adaptive guidance
     * @param z_values Data charge values for adaptive guidance
     * @param num_samples Number of adaptive samples to generate
     * @return Number of valid adaptive samples found
     */
    G4int SampleRedAreaAdaptively(std::vector<std::pair<G4double, G4double>>& samples,
                                  const std::vector<G4double>& x_coords,
                                  const std::vector<G4double>& y_coords,
                                  const std::vector<G4double>& z_values,
                                  G4int num_samples = 8) const;
    
    /**
     * @brief Analyze red area coverage and provide statistics
     * @param verbose Whether to print detailed analysis
     * @return Total valid area in red region [mm²]
     */
    G4double AnalyzeRedAreaCoverage(G4bool verbose = false) const;
    
    /**
     * @brief Calculate Gaussian true distance for fit quality assessment
     * @param x_coords X coordinates of data points [mm]
     * @param y_coords Y coordinates of data points [mm]
     * @param z_values Charge values at each point
     * @param params Gaussian parameters
     * @return True distance metric
     */
    G4double CalculateGaussianTrueDistance(const std::vector<G4double>& x_coords,
                                          const std::vector<G4double>& y_coords,
                                          const std::vector<G4double>& z_values,
                                          const G4double* params) const;

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
     * @brief Calculate physically reasonable initial parameter estimates with enhanced sigma bounds
     */
    void CalculatePhysicalInitialGuess(const std::vector<G4double>& x_coords,
                                      const std::vector<G4double>& y_coords,
                                      const std::vector<G4double>& z_values,
                                      G4double* initialParams,
                                      G4int strategy = 0);
    
    /**
     * @brief Check if point is inside any pixel with buffer zone
     */
    G4bool IsPointInsidePixelZone(G4double x, G4double y, G4double min_distance = 0.01*mm) const;
    
    /**
     * @brief Calculate minimum distance from point to any pixel edge
     */
    G4double CalculateMinDistanceToPixel(G4double x, G4double y) const;
    
    /**
     * @brief Calculate minimum distance from point to any pixel center
     */
    G4double CalculateMinDistanceToPixelCenter(G4double x, G4double y) const;
    
    /**
     * @brief Check if fit parameters satisfy all geometric constraints
     */
    G4bool CheckConstraints(const G4double* params, G4bool verbose = false) const;
    
    /**
     * @brief Get the pixel boundaries for a given pixel center position
     * @param center_x X coordinate of pixel center [mm]
     * @param center_y Y coordinate of pixel center [mm] 
     * @param bounds Array to store bounds [x_min, x_max, y_min, y_max]
     */
    void GetPixelBounds(G4double center_x, G4double center_y, G4double* bounds) const;
    
    /**
     * @brief Calculate neighborhood boundary around center pixel
     * @param bounds Array to store neighborhood bounds [x_min, x_max, y_min, y_max]
     */
    void CalculateNeighborhoodBounds(G4double* bounds) const;
    
    /**
     * @brief Check if point is in the "red area" - valid region for Gaussian center
     * @param x X coordinate [mm]
     * @param y Y coordinate [mm]
     * @return True if point is in valid red area
     */
    G4bool IsPointInRedArea(G4double x, G4double y) const;
    
    /**
     * @brief Check if point is within neighborhood boundary
     * @param x X coordinate [mm]
     * @param y Y coordinate [mm]
     * @return True if point is within neighborhood bounds
     */
    G4bool IsPointInNeighborhood(G4double x, G4double y) const;
    
    /**
     * @brief Apply parameter bounds during optimization
     */
    void ApplyParameterBounds(G4double* params) const;
    
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
    
    // Detector geometry constraints
    DetectorGeometry fDetectorGeometry;
    
    // Center pixel constraint
    G4double fCenterPixelX;
    G4double fCenterPixelY;
    G4bool fConstrainToCenterPixel;
    
    // Neighborhood constraint parameters
    G4int fNeighborhoodRadius;  // Radius for neighborhood calculation (e.g., 4 for 9x9 grid)
    
    // Red area constraint mode
    G4bool fUseRedAreaConstraints;  // Flag to enable red area constraint mode
    
    // Legacy ROOT fitting objects (kept for compatibility but not used)
    void* fGaussianFunction;  // Changed to void* to avoid ROOT dependencies
    void* fDataGraph;         // Changed to void* to avoid ROOT dependencies
    
    // Fitting parameters
    static const G4int fNParams = Constants::GAUSSIAN_N_PARAMS;  // Number of fit parameters
    
    // Algorithm parameters
    static const G4int fMaxFitAttempts = Constants::MAX_FIT_ATTEMPTS;     // Maximum number of fitting attempts
    static const G4double fConstraintPenalty;  // Penalty factor for constraint violations
};

#endif // GAUSSIAN3DFITTER_HH 