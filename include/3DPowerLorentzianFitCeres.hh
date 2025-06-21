#ifndef POWERLORENTZIANFITCERES3D_HH
#define POWERLORENTZIANFITCERES3D_HH

#include <vector>
#include "globals.hh"

// Structure to hold 3D Power-Law Lorentzian fit results for entire neighborhood surface
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2)^β + B
// where mx, my = center coordinates, γx, γy = widths, β = power exponent, A = amplitude, B = baseline
struct PowerLorentzianFit3DResultsCeres {
    // 3D surface fit parameters
    G4double center_x;           // mx parameter (X center)
    G4double center_y;           // my parameter (Y center)
    G4double gamma_x;            // γx parameter (X width)
    G4double gamma_y;            // γy parameter (Y width)
    G4double beta;               // β parameter (power exponent)
    G4double amplitude;          // A parameter (peak amplitude)
    G4double vertical_offset;    // B parameter (baseline offset)
    
    // Parameter uncertainties
    G4double center_x_err;
    G4double center_y_err;
    G4double gamma_x_err;
    G4double gamma_y_err;
    G4double beta_err;
    G4double amplitude_err;
    G4double vertical_offset_err;
    
    // Fit quality metrics
    G4double chi2red;            // Reduced chi-squared
    G4double pp;                 // P-value (goodness of fit)
    G4int dof;                   // Degrees of freedom
    
    // Data arrays for ROOT analysis
    std::vector<double> x_coords;        // X coordinates of all fitted pixels
    std::vector<double> y_coords;        // Y coordinates of all fitted pixels  
    std::vector<double> charge_values;   // Charge values for all fitted pixels
    std::vector<double> charge_errors;   // Charge uncertainties for all fitted pixels
    
    // Charge uncertainty (5% of max charge in neighborhood if enabled)
    G4double charge_uncertainty;
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    PowerLorentzianFit3DResultsCeres() : 
        center_x(0), center_y(0), gamma_x(1), gamma_y(1), beta(1), amplitude(0), vertical_offset(0),
        center_x_err(0), center_y_err(0), gamma_x_err(0), gamma_y_err(0), beta_err(0),
        amplitude_err(0), vertical_offset_err(0),
        chi2red(0), pp(0), dof(0),
        charge_uncertainty(0),
        fit_successful(false) {}
};

// Function to perform 3D Power-Law Lorentzian fitting using Ceres Solver with robust optimization
// Fits entire neighborhood surface with 3D Power-Law Lorentzian function
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2)^β + B
// where mx, my = center coordinates, γx, γy = widths, β = power exponent, A = amplitude, B = baseline
// Uses most robust Ceres settings regardless of computation time
PowerLorentzianFit3DResultsCeres Fit3DPowerLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // POWERLORENTZIANFITCERES3D_HH 