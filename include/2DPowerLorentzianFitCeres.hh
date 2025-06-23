#ifndef POWERLORENTZIANFITCERES2D_HH
#define POWERLORENTZIANFITCERES2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold outlier removal results for Power-Law Lorentzian fitting
struct PowerLorentzianOutlierRemovalResult {
    std::vector<double> filtered_x_coords;
    std::vector<double> filtered_y_coords;
    std::vector<double> filtered_charge_values;
    int outliers_removed;
    bool filtering_applied;
    bool success;
    
    // Constructor with default values
    PowerLorentzianOutlierRemovalResult() : 
        outliers_removed(0), filtering_applied(false), success(false) {}
};

// Structure to hold 2D Power-Law Lorentzian fit results
// Model: y(x) = A / (1 + ((x-m)/gamma)^2)^beta + B
struct PowerLorentzianFit2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;      // m parameter
    G4double x_gamma;       // gamma parameter (width)
    G4double x_beta;        // beta parameter (power exponent)
    G4double x_amplitude;   // A parameter
    G4double x_center_err;
    G4double x_gamma_err;
    G4double x_beta_err;
    G4double x_amplitude_err;
    G4double x_vertical_offset;     // B parameter
    G4double x_vertical_offset_err;
    G4double x_chi2red;
    G4double x_pp;
    G4int x_dof;
    
    // Y direction fit results (central column)
    G4double y_center;      // m parameter
    G4double y_gamma;       // gamma parameter (width)
    G4double y_beta;        // beta parameter (power exponent)
    G4double y_amplitude;   // A parameter
    G4double y_center_err;
    G4double y_gamma_err;
    G4double y_beta_err;
    G4double y_amplitude_err;
    G4double y_vertical_offset;     // B parameter
    G4double y_vertical_offset_err;
    G4double y_chi2red;
    G4double y_pp;
    G4int y_dof;
    
    // Charge error vectors for ROOT analysis (same as Lorentzian)
    std::vector<double> x_row_pixel_coords;     // X coordinates of pixels in central row
    std::vector<double> x_row_charge_values;    // Charge values for pixels in central row
    std::vector<double> x_row_charge_errors;    // 3x3 neighborhood charge errors for central row
    
    std::vector<double> y_col_pixel_coords;     // Y coordinates of pixels in central column
    std::vector<double> y_col_charge_values;    // Charge values for pixels in central column
    std::vector<double> y_col_charge_errors;    // 3x3 neighborhood charge errors for central column
    
    // Charge uncertainties (same as Lorentzian)
    G4double x_charge_uncertainty;  // 5% of max charge in X direction (if enabled)
    G4double y_charge_uncertainty;  // 5% of max charge in Y direction (if enabled)
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    PowerLorentzianFit2DResultsCeres() : 
        x_center(0), x_gamma(1), x_beta(1), x_amplitude(0),
        x_center_err(0), x_gamma_err(0), x_beta_err(0), x_amplitude_err(0),
        x_vertical_offset(0), x_vertical_offset_err(0),
        x_chi2red(0), x_pp(0), x_dof(0),
        y_center(0), y_gamma(1), y_beta(1), y_amplitude(0),
        y_center_err(0), y_gamma_err(0), y_beta_err(0), y_amplitude_err(0),
        y_vertical_offset(0), y_vertical_offset_err(0),
        y_chi2red(0), y_pp(0), y_dof(0),
        x_charge_uncertainty(0), y_charge_uncertainty(0),
        fit_successful(false) {}
};

// Structure to hold diagonal Power-Law Lorentzian fit results
struct DiagonalPowerLorentzianFitResultsCeres {
    // Main diagonal X fit results
    G4double main_diag_x_center;
    G4double main_diag_x_gamma;
    G4double main_diag_x_beta;
    G4double main_diag_x_amplitude;
    G4double main_diag_x_center_err;
    G4double main_diag_x_gamma_err;
    G4double main_diag_x_beta_err;
    G4double main_diag_x_amplitude_err;
    G4double main_diag_x_vertical_offset;
    G4double main_diag_x_vertical_offset_err;
    G4double main_diag_x_chi2red;
    G4double main_diag_x_pp;
    G4int main_diag_x_dof;
    G4bool main_diag_x_fit_successful;
    
    // Main diagonal Y fit results
    G4double main_diag_y_center;
    G4double main_diag_y_gamma;
    G4double main_diag_y_beta;
    G4double main_diag_y_amplitude;
    G4double main_diag_y_center_err;
    G4double main_diag_y_gamma_err;
    G4double main_diag_y_beta_err;
    G4double main_diag_y_amplitude_err;
    G4double main_diag_y_vertical_offset;
    G4double main_diag_y_vertical_offset_err;
    G4double main_diag_y_chi2red;
    G4double main_diag_y_pp;
    G4int main_diag_y_dof;
    G4bool main_diag_y_fit_successful;
    
    // Secondary diagonal X fit results
    G4double sec_diag_x_center;
    G4double sec_diag_x_gamma;
    G4double sec_diag_x_beta;
    G4double sec_diag_x_amplitude;
    G4double sec_diag_x_center_err;
    G4double sec_diag_x_gamma_err;
    G4double sec_diag_x_beta_err;
    G4double sec_diag_x_amplitude_err;
    G4double sec_diag_x_vertical_offset;
    G4double sec_diag_x_vertical_offset_err;
    G4double sec_diag_x_chi2red;
    G4double sec_diag_x_pp;
    G4int sec_diag_x_dof;
    G4bool sec_diag_x_fit_successful;
    
    // Secondary diagonal Y fit results
    G4double sec_diag_y_center;
    G4double sec_diag_y_gamma;
    G4double sec_diag_y_beta;
    G4double sec_diag_y_amplitude;
    G4double sec_diag_y_center_err;
    G4double sec_diag_y_gamma_err;
    G4double sec_diag_y_beta_err;
    G4double sec_diag_y_amplitude_err;
    G4double sec_diag_y_vertical_offset;
    G4double sec_diag_y_vertical_offset_err;
    G4double sec_diag_y_chi2red;
    G4double sec_diag_y_pp;
    G4int sec_diag_y_dof;
    G4bool sec_diag_y_fit_successful;
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    DiagonalPowerLorentzianFitResultsCeres() : 
        main_diag_x_center(0), main_diag_x_gamma(1), main_diag_x_beta(1), main_diag_x_amplitude(0),
        main_diag_x_center_err(0), main_diag_x_gamma_err(0), main_diag_x_beta_err(0), main_diag_x_amplitude_err(0),
        main_diag_x_vertical_offset(0), main_diag_x_vertical_offset_err(0),
        main_diag_x_chi2red(0), main_diag_x_pp(0), main_diag_x_dof(0), main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_gamma(1), main_diag_y_beta(1), main_diag_y_amplitude(0),
        main_diag_y_center_err(0), main_diag_y_gamma_err(0), main_diag_y_beta_err(0), main_diag_y_amplitude_err(0),
        main_diag_y_vertical_offset(0), main_diag_y_vertical_offset_err(0),
        main_diag_y_chi2red(0), main_diag_y_pp(0), main_diag_y_dof(0), main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_gamma(1), sec_diag_x_beta(1), sec_diag_x_amplitude(0),
        sec_diag_x_center_err(0), sec_diag_x_gamma_err(0), sec_diag_x_beta_err(0), sec_diag_x_amplitude_err(0),
        sec_diag_x_vertical_offset(0), sec_diag_x_vertical_offset_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_pp(0), sec_diag_x_dof(0), sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_gamma(1), sec_diag_y_beta(1), sec_diag_y_amplitude(0),
        sec_diag_y_center_err(0), sec_diag_y_gamma_err(0), sec_diag_y_beta_err(0), sec_diag_y_amplitude_err(0),
        sec_diag_y_vertical_offset(0), sec_diag_y_vertical_offset_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_pp(0), sec_diag_y_dof(0), sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Legacy outlier removal function removed - now using shared FilterOutliersMad1D from StatsUtils

// Function to perform 2D Power-Law Lorentzian fitting
// Function form: y(x) = A / (1 + ((x-m)/gamma)^2)^beta + B
PowerLorentzianFit2DResultsCeres Fit2DPowerLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

// Function to perform diagonal Power-Law Lorentzian fitting
DiagonalPowerLorentzianFitResultsCeres FitDiagonalPowerLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);



#endif // POWERLORENTZIANFITCERES2D_HH 