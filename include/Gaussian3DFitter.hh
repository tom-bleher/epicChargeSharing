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
     * @brief Structure to hold fit results
     */
    struct FitResults {
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
        G4double chi2;             // Chi-squared value
        G4double ndf;              // Number of degrees of freedom
        G4double prob;             // Fit probability
        G4double r_squared;        // R-squared value
        G4int n_points;            // Number of data points used in fit
        G4bool fit_successful;     // Whether fit converged successfully
        
        // Additional statistics
        G4double residual_mean;    // Mean of residuals
        G4double residual_std;     // Standard deviation of residuals
        
        // Constructor to initialize with default values
        FitResults() : 
            amplitude(0), x0(0), y0(0), sigma_x(0), sigma_y(0), theta(0), offset(0),
            amplitude_err(0), x0_err(0), y0_err(0), sigma_x_err(0), sigma_y_err(0), theta_err(0), offset_err(0),
            chi2(0), ndf(0), prob(0), r_squared(0), n_points(0), fit_successful(false),
            residual_mean(0), residual_std(0) {}
    };

    /**
     * @brief Constructor
     */
    Gaussian3DFitter();
    
    /**
     * @brief Destructor
     */
    ~Gaussian3DFitter();
    
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

private:
    /**
     * @brief Calculate initial parameter estimates
     */
    void CalculateInitialGuess(const std::vector<G4double>& x_coords,
                              const std::vector<G4double>& y_coords,
                              const std::vector<G4double>& z_values,
                              G4double* initialParams);
    
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
     * @brief Perform Nelder-Mead simplex optimization (robust, no derivatives needed)
     */
    void SimplexFit(const std::vector<G4double>& x_coords,
                    const std::vector<G4double>& y_coords,
                    const std::vector<G4double>& z_values,
                    const std::vector<G4double>& z_errors,
                    G4double* params,
                    G4bool verbose);
    
    // Legacy ROOT fitting objects (kept for compatibility but not used)
    void* fGaussianFunction;  // Changed to void* to avoid ROOT dependencies
    void* fDataGraph;         // Changed to void* to avoid ROOT dependencies
    
    // Internal fitting parameters
    static const G4int fNParams = 7;  // Number of fit parameters
};

#endif // GAUSSIAN3DFITTER_HH 