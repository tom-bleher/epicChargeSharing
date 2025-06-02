#ifndef MINUITGAUSSIANFITTER_HH
#define MINUITGAUSSIANFITTER_HH

#include "G4Types.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "Constants.hh"
#include <vector>

// ROOT includes for Minuit fitting
#include "TMinuit.h"
#include "TMath.h"
#include "TF2.h"

/**
 * @brief ROOT Minuit-based 3D Gaussian fitting class
 * 
 * This class implements 3D Gaussian fitting using ROOT's Minuit optimizer.
 * It provides the same interface as Gaussian3DFitter for easy switching.
 */
class MinuitGaussianFitter {
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
        G4double pixel_exclusion_buffer; // Buffer around pixels to exclude [mm]
        
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
        G4double chi2red;             // Chi-squared value
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
    MinuitGaussianFitter(const DetectorGeometry& detector_geometry = DetectorGeometry());
    
    /**
     * @brief Destructor
     */
    ~MinuitGaussianFitter();
    
    /**
     * @brief Set detector geometry constraints
     */
    void SetDetectorGeometry(const DetectorGeometry& geometry);
    
    /**
     * @brief Get current detector geometry
     */
    const DetectorGeometry& GetDetectorGeometry() const { return fDetectorGeometry; }
    
    /**
     * @brief Fit 3D Gaussian to charge distribution data using Minuit
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

private:
    /**
     * @brief Static function for Minuit chi-squared calculation
     */
    static void MinuitFcn(G4int& npar, G4double* gin, G4double& f, G4double* par, G4int iflag);
    
    /**
     * @brief Calculate initial parameter estimates
     */
    void CalculateInitialGuess(const std::vector<G4double>& x_coords,
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
     * @brief Calculate residual statistics
     */
    void CalculateResidualStats(const std::vector<G4double>& x_coords,
                               const std::vector<G4double>& y_coords,
                               const std::vector<G4double>& z_values,
                               const G4double* fitParams,
                               G4double& mean, G4double& std_dev);

    // Static data for Minuit callback
    static std::vector<G4double> fStaticXCoords;
    static std::vector<G4double> fStaticYCoords;
    static std::vector<G4double> fStaticZValues;
    static std::vector<G4double> fStaticZErrors;
    static const MinuitGaussianFitter* fStaticInstance;

    // Member variables
    DetectorGeometry fDetectorGeometry;
    TMinuit* fMinuit;
    
    // Fitting parameters  
    static const G4int fNParams = Constants::GAUSSIAN_N_PARAMS;  // Number of fit parameters
    static const G4int fMaxFitAttempts = Constants::MAX_FIT_ATTEMPTS;     // Maximum number of fitting attempts
};

#endif 