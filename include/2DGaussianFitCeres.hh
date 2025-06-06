#ifndef GAUSSIANFITCERES2D_HH
#define GAUSSIANFITCERES2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold 2D Gaussian fit results (same as ROOT version for compatibility)
struct GaussianFit2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;
    G4double x_sigma;
    G4double x_amplitude;
    G4double x_center_err;
    G4double x_sigma_err;
    G4double x_amplitude_err;
    G4double x_chi2red;
    G4int x_npoints;
    
    // Y direction fit results (central column)
    G4double y_center;
    G4double y_sigma;
    G4double y_amplitude;
    G4double y_center_err;
    G4double y_sigma_err;
    G4double y_amplitude_err;
    G4double y_chi2red;
    G4int y_npoints;
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    GaussianFit2DResultsCeres() : 
        x_center(0), x_sigma(0), x_amplitude(0),
        x_center_err(0), x_sigma_err(0), x_amplitude_err(0),
        x_chi2red(0), x_npoints(0),
        y_center(0), y_sigma(0), y_amplitude(0),
        y_center_err(0), y_sigma_err(0), y_amplitude_err(0),
        y_chi2red(0), y_npoints(0),
        fit_successful(false) {}
};

// Structure to hold diagonal fit results (same as ROOT version for compatibility)
struct DiagonalFitResultsCeres {
    // Main diagonal X fit results (X vs Charge for pixels on main diagonal)
    G4double main_diag_x_center;         // X center from main diagonal X fit
    G4double main_diag_x_sigma;          // X sigma from main diagonal X fit
    G4double main_diag_x_amplitude;      // X amplitude from main diagonal X fit
    G4double main_diag_x_center_err;     // X center error from main diagonal X fit
    G4double main_diag_x_sigma_err;      // X sigma error from main diagonal X fit
    G4double main_diag_x_amplitude_err;  // X amplitude error from main diagonal X fit
    G4double main_diag_x_chi2red;        // X reduced chi-squared from main diagonal X fit
    G4int main_diag_x_npoints;           // X number of points in main diagonal X fit
    G4bool main_diag_x_fit_successful;   // Whether main diagonal X fit was successful
    
    // Main diagonal Y fit results (Y vs Charge for pixels on main diagonal)
    G4double main_diag_y_center;         // Y center from main diagonal Y fit
    G4double main_diag_y_sigma;          // Y sigma from main diagonal Y fit
    G4double main_diag_y_amplitude;      // Y amplitude from main diagonal Y fit
    G4double main_diag_y_center_err;     // Y center error from main diagonal Y fit
    G4double main_diag_y_sigma_err;      // Y sigma error from main diagonal Y fit
    G4double main_diag_y_amplitude_err;  // Y amplitude error from main diagonal Y fit
    G4double main_diag_y_chi2red;        // Y reduced chi-squared from main diagonal Y fit
    G4int main_diag_y_npoints;           // Y number of points in main diagonal Y fit
    G4bool main_diag_y_fit_successful;   // Whether main diagonal Y fit was successful
    
    // Secondary diagonal X fit results (X vs Charge for pixels on secondary diagonal)
    G4double sec_diag_x_center;          // X center from secondary diagonal X fit
    G4double sec_diag_x_sigma;           // X sigma from secondary diagonal X fit
    G4double sec_diag_x_amplitude;       // X amplitude from secondary diagonal X fit
    G4double sec_diag_x_center_err;      // X center error from secondary diagonal X fit
    G4double sec_diag_x_sigma_err;       // X sigma error from secondary diagonal X fit
    G4double sec_diag_x_amplitude_err;   // X amplitude error from secondary diagonal X fit
    G4double sec_diag_x_chi2red;         // X reduced chi-squared from secondary diagonal X fit
    G4int sec_diag_x_npoints;            // X number of points in secondary diagonal X fit
    G4bool sec_diag_x_fit_successful;    // Whether secondary diagonal X fit was successful
    
    // Secondary diagonal Y fit results (Y vs Charge for pixels on secondary diagonal)
    G4double sec_diag_y_center;          // Y center from secondary diagonal Y fit
    G4double sec_diag_y_sigma;           // Y sigma from secondary diagonal Y fit
    G4double sec_diag_y_amplitude;       // Y amplitude from secondary diagonal Y fit
    G4double sec_diag_y_center_err;      // Y center error from secondary diagonal Y fit
    G4double sec_diag_y_sigma_err;       // Y sigma error from secondary diagonal Y fit
    G4double sec_diag_y_amplitude_err;   // Y amplitude error from secondary diagonal Y fit
    G4double sec_diag_y_chi2red;         // Y reduced chi-squared from secondary diagonal Y fit
    G4int sec_diag_y_npoints;            // Y number of points in secondary diagonal Y fit
    G4bool sec_diag_y_fit_successful;    // Whether secondary diagonal Y fit was successful
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    DiagonalFitResultsCeres() : 
        main_diag_x_center(0), main_diag_x_sigma(0), main_diag_x_amplitude(0),
        main_diag_x_center_err(0), main_diag_x_sigma_err(0), main_diag_x_amplitude_err(0),
        main_diag_x_chi2red(0), main_diag_x_npoints(0), main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_sigma(0), main_diag_y_amplitude(0),
        main_diag_y_center_err(0), main_diag_y_sigma_err(0), main_diag_y_amplitude_err(0),
        main_diag_y_chi2red(0), main_diag_y_npoints(0), main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_sigma(0), sec_diag_x_amplitude(0),
        sec_diag_x_center_err(0), sec_diag_x_sigma_err(0), sec_diag_x_amplitude_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_npoints(0), sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_sigma(0), sec_diag_y_amplitude(0),
        sec_diag_y_center_err(0), sec_diag_y_sigma_err(0), sec_diag_y_amplitude_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_npoints(0), sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Function to perform 2D Gaussian fitting using Ceres Solver with robust optimization
// Fits central row and column separately with Gaussian functions
// Function form: y(x) = A * exp(-(x - m)^2 / (2 * sigma^2)) + B
// Uses most robust Ceres settings regardless of computation time
GaussianFit2DResultsCeres Fit2DGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

// Function to perform diagonal Gaussian fitting using Ceres Solver with robust optimization
// Fits along main diagonal and secondary diagonal of the 9x9 grid
// Main diagonal: from bottom-left to top-right (slope = +1)
// Secondary diagonal: from top-left to bottom-right (slope = -1)
// Uses most robust Ceres settings regardless of computation time
DiagonalFitResultsCeres FitDiagonalGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // GAUSSIANFITCERES2D_HH 