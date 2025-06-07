#ifndef LORENTZIANFITCERES2D_HH
#define LORENTZIANFITCERES2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold 2D Lorentzian fit results (similar to Gaussian version)
struct LorentzianFit2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;           // m: peak center
    G4double x_gamma;            // γ: half-width at half-maximum (HWHM)
    G4double x_amplitude;        // A: peak amplitude above baseline
    G4double x_center_err;       // Error in peak center
    G4double x_gamma_err;        // Error in HWHM
    G4double x_amplitude_err;    // Error in amplitude
    G4double x_chi2red;          // Reduced chi-squared
    G4int x_npoints;             // Number of data points used
    
    // Y direction fit results (central column)
    G4double y_center;           // m: peak center
    G4double y_gamma;            // γ: half-width at half-maximum (HWHM)
    G4double y_amplitude;        // A: peak amplitude above baseline
    G4double y_center_err;       // Error in peak center
    G4double y_gamma_err;        // Error in HWHM
    G4double y_amplitude_err;    // Error in amplitude
    G4double y_chi2red;          // Reduced chi-squared
    G4int y_npoints;             // Number of data points used
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    LorentzianFit2DResultsCeres() : 
        x_center(0), x_gamma(0), x_amplitude(0),
        x_center_err(0), x_gamma_err(0), x_amplitude_err(0),
        x_chi2red(0), x_npoints(0),
        y_center(0), y_gamma(0), y_amplitude(0),
        y_center_err(0), y_gamma_err(0), y_amplitude_err(0),
        y_chi2red(0), y_npoints(0),
        fit_successful(false) {}
};

// Structure to hold diagonal Lorentzian fit results
struct DiagonalLorentzianFitResultsCeres {
    // Main diagonal X fit results (X vs Charge for pixels on main diagonal)
    G4double main_diag_x_center;         // X center from main diagonal X fit
    G4double main_diag_x_gamma;          // X gamma (HWHM) from main diagonal X fit
    G4double main_diag_x_amplitude;      // X amplitude from main diagonal X fit
    G4double main_diag_x_center_err;     // X center error from main diagonal X fit
    G4double main_diag_x_gamma_err;      // X gamma error from main diagonal X fit
    G4double main_diag_x_amplitude_err;  // X amplitude error from main diagonal X fit
    G4double main_diag_x_chi2red;        // X reduced chi-squared from main diagonal X fit
    G4int main_diag_x_npoints;           // X number of points in main diagonal X fit
    G4bool main_diag_x_fit_successful;   // Whether main diagonal X fit was successful
    
    // Main diagonal Y fit results (Y vs Charge for pixels on main diagonal)
    G4double main_diag_y_center;         // Y center from main diagonal Y fit
    G4double main_diag_y_gamma;          // Y gamma (HWHM) from main diagonal Y fit
    G4double main_diag_y_amplitude;      // Y amplitude from main diagonal Y fit
    G4double main_diag_y_center_err;     // Y center error from main diagonal Y fit
    G4double main_diag_y_gamma_err;      // Y gamma error from main diagonal Y fit
    G4double main_diag_y_amplitude_err;  // Y amplitude error from main diagonal Y fit
    G4double main_diag_y_chi2red;        // Y reduced chi-squared from main diagonal Y fit
    G4int main_diag_y_npoints;           // Y number of points in main diagonal Y fit
    G4bool main_diag_y_fit_successful;   // Whether main diagonal Y fit was successful
    
    // Secondary diagonal X fit results (X vs Charge for pixels on secondary diagonal)
    G4double sec_diag_x_center;          // X center from secondary diagonal X fit
    G4double sec_diag_x_gamma;           // X gamma (HWHM) from secondary diagonal X fit
    G4double sec_diag_x_amplitude;       // X amplitude from secondary diagonal X fit
    G4double sec_diag_x_center_err;      // X center error from secondary diagonal X fit
    G4double sec_diag_x_gamma_err;       // X gamma error from secondary diagonal X fit
    G4double sec_diag_x_amplitude_err;   // X amplitude error from secondary diagonal X fit
    G4double sec_diag_x_chi2red;         // X reduced chi-squared from secondary diagonal X fit
    G4int sec_diag_x_npoints;            // X number of points in secondary diagonal X fit
    G4bool sec_diag_x_fit_successful;    // Whether secondary diagonal X fit was successful
    
    // Secondary diagonal Y fit results (Y vs Charge for pixels on secondary diagonal)
    G4double sec_diag_y_center;          // Y center from secondary diagonal Y fit
    G4double sec_diag_y_gamma;           // Y gamma (HWHM) from secondary diagonal Y fit
    G4double sec_diag_y_amplitude;       // Y amplitude from secondary diagonal Y fit
    G4double sec_diag_y_center_err;      // Y center error from secondary diagonal Y fit
    G4double sec_diag_y_gamma_err;       // Y gamma error from secondary diagonal Y fit
    G4double sec_diag_y_amplitude_err;   // Y amplitude error from secondary diagonal Y fit
    G4double sec_diag_y_chi2red;         // Y reduced chi-squared from secondary diagonal Y fit
    G4int sec_diag_y_npoints;            // Y number of points in secondary diagonal Y fit
    G4bool sec_diag_y_fit_successful;    // Whether secondary diagonal Y fit was successful
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    DiagonalLorentzianFitResultsCeres() : 
        main_diag_x_center(0), main_diag_x_gamma(0), main_diag_x_amplitude(0),
        main_diag_x_center_err(0), main_diag_x_gamma_err(0), main_diag_x_amplitude_err(0),
        main_diag_x_chi2red(0), main_diag_x_npoints(0), main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_gamma(0), main_diag_y_amplitude(0),
        main_diag_y_center_err(0), main_diag_y_gamma_err(0), main_diag_y_amplitude_err(0),
        main_diag_y_chi2red(0), main_diag_y_npoints(0), main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_gamma(0), sec_diag_x_amplitude(0),
        sec_diag_x_center_err(0), sec_diag_x_gamma_err(0), sec_diag_x_amplitude_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_npoints(0), sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_gamma(0), sec_diag_y_amplitude(0),
        sec_diag_y_center_err(0), sec_diag_y_gamma_err(0), sec_diag_y_amplitude_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_npoints(0), sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Function to perform 2D Lorentzian fitting using Ceres Solver with robust optimization
// Fits central row and column separately with Lorentzian functions
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B
// where m = center, γ = HWHM, A = amplitude, B = baseline
// Uses most robust Ceres settings regardless of computation time
LorentzianFit2DResultsCeres Fit2DLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

// Function to perform diagonal Lorentzian fitting using Ceres Solver with robust optimization
// Fits along main diagonal and secondary diagonal of the 9x9 grid
// Main diagonal: from bottom-left to top-right (slope = +1)
// Secondary diagonal: from top-left to bottom-right (slope = -1)
// Uses most robust Ceres settings regardless of computation time
DiagonalLorentzianFitResultsCeres FitDiagonalLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // LORENTZIANFITCERES2D_HH 