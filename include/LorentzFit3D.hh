#ifndef LORENTZCERES3D_HH
#define LORENTZCERES3D_HH

#include <vector>
#include "globals.hh"

// Structure to hold 3D Lorentz fit results for entire neighborhood surface
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2) + B
// where mx, my = center coordinates, γx, γy = widths, A = amp, B = baseline
struct Lorentz3DResultsCeres {
    // 3D surface fit parameters
    G4double center_x;           // mx parameter (X center)
    G4double center_y;           // my parameter (Y center)
    G4double gamma_x;            // γx parameter (X width/HWHM)
    G4double gamma_y;            // γy parameter (Y width/HWHM)
    G4double amp;          // A parameter (peak amp)
    G4double vert_offset;    // B parameter (baseline offset)
    
    // Parameter uncertainties
    G4double center_x_err;
    G4double center_y_err;
    G4double gamma_x_err;
    G4double gamma_y_err;
    G4double amp_err;
    G4double vert_offset_err;
    
    //  quality metrics
    G4double chi2red;            // Reduced chi-squared
    G4double pp;                 // P-value (goodness of fit)
    G4int dof;                   // Degrees of freedom
    
    // Data arrays for ROOT analysis
    std::vector<double> x_coords;        // X coordinates of all fitted pixels
    std::vector<double> y_coords;        // Y coordinates of all fitted pixels  
    std::vector<double> charge_values;   // Charge values for all fitted pixels
    std::vector<double> charge_errors;   // Charge uncertainties for all fitted pixels
    
    // Charge err (5% of max charge in neighborhood if enabled)
    G4double charge_err;
    
    // Overall success status
    G4bool fit_success;
    
    // Constructor with default values
    Lorentz3DResultsCeres() : 
        center_x(0), center_y(0), gamma_x(0), gamma_y(0), amp(0), vert_offset(0),
        center_x_err(0), center_y_err(0), gamma_x_err(0), gamma_y_err(0), 
        amp_err(0), vert_offset_err(0),
        chi2red(0), pp(0), dof(0),
        charge_err(0),
        fit_success(false) {}
};

// Function to perform 3D Lorentz fitting using Ceres Solver with robust optimization
// s entire neighborhood surface with 3D Lorentz function
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2) + B
// where mx, my = center coordinates, γx, γy = widths, A = amp, B = baseline
// Uses most robust Ceres settings regardless of computation time
Lorentz3DResultsCeres LorentzCeres3D(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // LORENTZCERES3D_HH 