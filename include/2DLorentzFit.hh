#ifndef LORENTZCERES2D_HH
#define LORENTZCERES2D_HH

#include <vector>
#include "globals.hh"


// Structure to hold 2D Lorentz fit results (updated with new parameter names)
struct Lorentz2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;
    G4double x_gamma;
    G4double x_amp;
    G4double x_center_err;
    G4double x_gamma_err;
    G4double x_amp_err;
    G4double x_vert_offset;
    G4double x_vert_offset_err;
    G4double x_chi2red;
    G4double x_pp;
    G4int x_dof;
    
    // Y direction fit results (central column)
    G4double y_center;
    G4double y_gamma;
    G4double y_amp;
    G4double y_center_err;
    G4double y_gamma_err;
    G4double y_amp_err;
    G4double y_vert_offset;
    G4double y_vert_offset_err;
    G4double y_chi2red;
    G4double y_pp;
    G4int y_dof;
    
    // Charge error vectors for ROOT analysis
    std::vector<double> x_row_pixel_coords;     // X coordinates of pixels in central row
    std::vector<double> x_row_charge_values;    // Charge values for pixels in central row
    std::vector<double> x_row_charge_errors;    // 3x3 neighborhood charge errors for central row
    
    std::vector<double> y_col_pixel_coords;     // Y coordinates of pixels in central column
    std::vector<double> y_col_charge_values;    // Charge values for pixels in central column
    std::vector<double> y_col_charge_errors;    // 3x3 neighborhood charge errors for central column
    
    // Charge uncertainties (5% of max charge)
    G4double x_charge_err;  // Vert charge err for X direction fit
    G4double y_charge_err;  // Vert charge err for Y direction fit
    
    // Overall success status
    G4bool fit_success;
    
    // Constructor with default values
    Lorentz2DResultsCeres() : 
        x_center(0), x_gamma(0), x_amp(0),
        x_center_err(0), x_gamma_err(0), x_amp_err(0),
        x_vert_offset(0), x_vert_offset_err(0),
        x_chi2red(0), x_pp(0), x_dof(0),
        y_center(0), y_gamma(0), y_amp(0),
        y_center_err(0), y_gamma_err(0), y_amp_err(0),
        y_vert_offset(0), y_vert_offset_err(0),
        y_chi2red(0), y_pp(0), y_dof(0),
        x_charge_err(0), y_charge_err(0),
        fit_success(false) {}
};

// Structure to hold diagonal Lorentz fit results (updated with new parameter names)
struct DiagLorentzResultsCeres {
    // Main diagonal X fit results (X vs Charge for pixels on main diagonal)
    G4double main_diag_x_center;         // X center from main diagonal X fit
    G4double main_diag_x_gamma;          // X gamma from main diagonal X fit
    G4double main_diag_x_amp;      // X amp from main diagonal X fit
    G4double main_diag_x_center_err;     // X center error from main diagonal X fit
    G4double main_diag_x_gamma_err;      // X gamma error from main diagonal X fit
    G4double main_diag_x_amp_err;  // X amp error from main diagonal X fit
    G4double main_diag_x_vert_offset;        // X vert offset from main diagonal X fit
    G4double main_diag_x_vert_offset_err;    // X vert offset error from main diagonal X fit
    G4double main_diag_x_chi2red;        // X reduced chi-squared from main diagonal X fit
    G4double main_diag_x_pp;             // X p-value from main diagonal X fit
    G4int main_diag_x_dof;               // X degrees of freedom in main diagonal X fit
    G4bool main_diag_x_fit_success;   // Whether main diagonal X fit was success
    
    // Main diagonal Y fit results (Y vs Charge for pixels on main diagonal)
    G4double main_diag_y_center;         // Y center from main diagonal Y fit
    G4double main_diag_y_gamma;          // Y gamma from main diagonal Y fit
    G4double main_diag_y_amp;      // Y amp from main diagonal Y fit
    G4double main_diag_y_center_err;     // Y center error from main diagonal Y fit
    G4double main_diag_y_gamma_err;      // Y gamma error from main diagonal Y fit
    G4double main_diag_y_amp_err;  // Y amp error from main diagonal Y fit
    G4double main_diag_y_vert_offset;        // Y vert offset from main diagonal Y fit
    G4double main_diag_y_vert_offset_err;    // Y vert offset error from main diagonal Y fit
    G4double main_diag_y_chi2red;        // Y reduced chi-squared from main diagonal Y fit
    G4double main_diag_y_pp;             // Y p-value from main diagonal Y fit
    G4int main_diag_y_dof;               // Y degrees of freedom in main diagonal Y fit
    G4bool main_diag_y_fit_success;   // Whether main diagonal Y fit was success
    
    // Secondary diagonal X fit results (X vs Charge for pixels on secondary diagonal)
    G4double sec_diag_x_center;          // X center from secondary diagonal X fit
    G4double sec_diag_x_gamma;           // X gamma from secondary diagonal X fit
    G4double sec_diag_x_amp;       // X amp from secondary diagonal X fit
    G4double sec_diag_x_center_err;      // X center error from secondary diagonal X fit
    G4double sec_diag_x_gamma_err;       // X gamma error from secondary diagonal X fit
    G4double sec_diag_x_amp_err;   // X amp error from secondary diagonal X fit
    G4double sec_diag_x_vert_offset;         // X vert offset from secondary diagonal X fit
    G4double sec_diag_x_vert_offset_err;     // X vert offset error from secondary diagonal X fit
    G4double sec_diag_x_chi2red;         // X reduced chi-squared from secondary diagonal X fit
    G4double sec_diag_x_pp;              // X p-value from secondary diagonal X fit
    G4int sec_diag_x_dof;                // X degrees of freedom in secondary diagonal X fit
    G4bool sec_diag_x_fit_success;    // Whether secondary diagonal X fit was success
    
    // Secondary diagonal Y fit results (Y vs Charge for pixels on secondary diagonal)
    G4double sec_diag_y_center;          // Y center from secondary diagonal Y fit
    G4double sec_diag_y_gamma;           // Y gamma from secondary diagonal Y fit
    G4double sec_diag_y_amp;       // Y amp from secondary diagonal Y fit
    G4double sec_diag_y_center_err;      // Y center error from secondary diagonal Y fit
    G4double sec_diag_y_gamma_err;       // Y gamma error from secondary diagonal Y fit
    G4double sec_diag_y_amp_err;   // Y amp error from secondary diagonal Y fit
    G4double sec_diag_y_vert_offset;         // Y vert offset from secondary diagonal Y fit
    G4double sec_diag_y_vert_offset_err;     // Y vert offset error from secondary diagonal Y fit
    G4double sec_diag_y_chi2red;         // Y reduced chi-squared from secondary diagonal Y fit
    G4double sec_diag_y_pp;              // Y p-value from secondary diagonal Y fit
    G4int sec_diag_y_dof;                // Y degrees of freedom in secondary diagonal Y fit
    G4bool sec_diag_y_fit_success;    // Whether secondary diagonal Y fit was success
    
    // Charge error vectors for ROOT analysis - Main diagonal
    std::vector<double> main_diag_x_pixel_coords;     // X coordinates of pixels on main diagonal
    std::vector<double> main_diag_x_charge_values;    // Charge values for pixels on main diagonal (X fit)
    std::vector<double> main_diag_x_charge_errors;    // 3x3 neighborhood charge errors for main diagonal X
    
    std::vector<double> main_diag_y_pixel_coords;     // Y coordinates of pixels on main diagonal
    std::vector<double> main_diag_y_charge_values;    // Charge values for pixels on main diagonal (Y fit)
    std::vector<double> main_diag_y_charge_errors;    // 3x3 neighborhood charge errors for main diagonal Y
    
    // Charge error vectors for ROOT analysis - Secondary diagonal
    std::vector<double> sec_diag_x_pixel_coords;      // X coordinates of pixels on secondary diagonal
    std::vector<double> sec_diag_x_charge_values;     // Charge values for pixels on secondary diagonal (X fit)
    std::vector<double> sec_diag_x_charge_errors;     // 3x3 neighborhood charge errors for secondary diagonal X
    
    std::vector<double> sec_diag_y_pixel_coords;      // Y coordinates of pixels on secondary diagonal
    std::vector<double> sec_diag_y_charge_values;     // Charge values for pixels on secondary diagonal (Y fit)
    std::vector<double> sec_diag_y_charge_errors;     // 3x3 neighborhood charge errors for secondary diagonal Y
    
    // Overall success status
    G4bool fit_success;
    
    // Constructor with default values
    DiagLorentzResultsCeres() : 
        main_diag_x_center(0), main_diag_x_gamma(0), main_diag_x_amp(0),
        main_diag_x_center_err(0), main_diag_x_gamma_err(0), main_diag_x_amp_err(0),
        main_diag_x_vert_offset(0), main_diag_x_vert_offset_err(0),
        main_diag_x_chi2red(0), main_diag_x_pp(0), main_diag_x_dof(0), main_diag_x_fit_success(false),
        main_diag_y_center(0), main_diag_y_gamma(0), main_diag_y_amp(0),
        main_diag_y_center_err(0), main_diag_y_gamma_err(0), main_diag_y_amp_err(0),
        main_diag_y_vert_offset(0), main_diag_y_vert_offset_err(0),
        main_diag_y_chi2red(0), main_diag_y_pp(0), main_diag_y_dof(0), main_diag_y_fit_success(false),
        sec_diag_x_center(0), sec_diag_x_gamma(0), sec_diag_x_amp(0),
        sec_diag_x_center_err(0), sec_diag_x_gamma_err(0), sec_diag_x_amp_err(0),
        sec_diag_x_vert_offset(0), sec_diag_x_vert_offset_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_pp(0), sec_diag_x_dof(0), sec_diag_x_fit_success(false),
        sec_diag_y_center(0), sec_diag_y_gamma(0), sec_diag_y_amp(0),
        sec_diag_y_center_err(0), sec_diag_y_gamma_err(0), sec_diag_y_amp_err(0),
        sec_diag_y_vert_offset(0), sec_diag_y_vert_offset_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_pp(0), sec_diag_y_dof(0), sec_diag_y_fit_success(false),
        fit_success(false) {}
};

// Function to perform 2D Lorentz fitting using Ceres Solver with robust optimization
// s central row and column separately with Lorentz functions
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B
// where m = center, γ = HWHM, A = amp, B = baseline
// Uses most robust Ceres settings regardless of computation time
Lorentz2DResultsCeres LorentzCeres2D(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

// Function to perform diagonal Lorentz fitting using Ceres Solver with robust optimization
// s along main diagonal and secondary diagonal of the 9x9 grid
// Main diagonal: from bottom-left to top-right (slope = +1)
// Secondary diagonal: from top-left to bottom-right (slope = -1)
DiagLorentzResultsCeres DiagLorentzCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);



#endif // LORENTZCERES2D_HH 