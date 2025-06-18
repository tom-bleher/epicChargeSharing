#ifndef SKEWEDLORENTZIANFITCERES2D_HH
#define SKEWEDLORENTZIANFITCERES2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold outlier removal results for Skewed Lorentzian fitting
struct SkewedLorentzianOutlierRemovalResult {
    std::vector<double> filtered_x_coords;
    std::vector<double> filtered_y_coords;
    std::vector<double> filtered_charge_values;
    int outliers_removed;
    bool filtering_applied;
    bool success;
    
    // Constructor with default values
    SkewedLorentzianOutlierRemovalResult() : 
        outliers_removed(0), filtering_applied(false), success(false) {}
};

// Structure to hold 2D Skewed Lorentzian fit results
struct SkewedLorentzianFit2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;
    G4double x_beta;        // Shape parameter
    G4double x_lambda;      // Skewness parameter
    G4double x_gamma;       // Additional shape parameter
    G4double x_amplitude;
    G4double x_center_err;
    G4double x_beta_err;
    G4double x_lambda_err;
    G4double x_gamma_err;
    G4double x_amplitude_err;
    G4double x_vertical_offset;
    G4double x_vertical_offset_err;
    G4double x_chi2red;
    G4double x_pp;
    G4int x_dof;
    
    // Y direction fit results (central column)
    G4double y_center;
    G4double y_beta;        // Shape parameter
    G4double y_lambda;      // Skewness parameter
    G4double y_gamma;       // Additional shape parameter
    G4double y_amplitude;
    G4double y_center_err;
    G4double y_beta_err;
    G4double y_lambda_err;
    G4double y_gamma_err;
    G4double y_amplitude_err;
    G4double y_vertical_offset;
    G4double y_vertical_offset_err;
    G4double y_chi2red;
    G4double y_pp;
    G4int y_dof;
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    SkewedLorentzianFit2DResultsCeres() : 
        x_center(0), x_beta(1), x_lambda(0), x_gamma(1), x_amplitude(0),
        x_center_err(0), x_beta_err(0), x_lambda_err(0), x_gamma_err(0), x_amplitude_err(0),
        x_vertical_offset(0), x_vertical_offset_err(0),
        x_chi2red(0), x_pp(0), x_dof(0),
        y_center(0), y_beta(1), y_lambda(0), y_gamma(1), y_amplitude(0),
        y_center_err(0), y_beta_err(0), y_lambda_err(0), y_gamma_err(0), y_amplitude_err(0),
        y_vertical_offset(0), y_vertical_offset_err(0),
        y_chi2red(0), y_pp(0), y_dof(0),
        fit_successful(false) {}
};

// Structure to hold diagonal Skewed Lorentzian fit results
struct DiagonalSkewedLorentzianFitResultsCeres {
    // Main diagonal X fit results
    G4double main_diag_x_center;
    G4double main_diag_x_beta;
    G4double main_diag_x_lambda;
    G4double main_diag_x_gamma;
    G4double main_diag_x_amplitude;
    G4double main_diag_x_center_err;
    G4double main_diag_x_beta_err;
    G4double main_diag_x_lambda_err;
    G4double main_diag_x_gamma_err;
    G4double main_diag_x_amplitude_err;
    G4double main_diag_x_vertical_offset;
    G4double main_diag_x_vertical_offset_err;
    G4double main_diag_x_chi2red;
    G4double main_diag_x_pp;
    G4int main_diag_x_dof;
    G4bool main_diag_x_fit_successful;
    
    // Main diagonal Y fit results
    G4double main_diag_y_center;
    G4double main_diag_y_beta;
    G4double main_diag_y_lambda;
    G4double main_diag_y_gamma;
    G4double main_diag_y_amplitude;
    G4double main_diag_y_center_err;
    G4double main_diag_y_beta_err;
    G4double main_diag_y_lambda_err;
    G4double main_diag_y_gamma_err;
    G4double main_diag_y_amplitude_err;
    G4double main_diag_y_vertical_offset;
    G4double main_diag_y_vertical_offset_err;
    G4double main_diag_y_chi2red;
    G4double main_diag_y_pp;
    G4int main_diag_y_dof;
    G4bool main_diag_y_fit_successful;
    
    // Secondary diagonal X fit results
    G4double sec_diag_x_center;
    G4double sec_diag_x_beta;
    G4double sec_diag_x_lambda;
    G4double sec_diag_x_gamma;
    G4double sec_diag_x_amplitude;
    G4double sec_diag_x_center_err;
    G4double sec_diag_x_beta_err;
    G4double sec_diag_x_lambda_err;
    G4double sec_diag_x_gamma_err;
    G4double sec_diag_x_amplitude_err;
    G4double sec_diag_x_vertical_offset;
    G4double sec_diag_x_vertical_offset_err;
    G4double sec_diag_x_chi2red;
    G4double sec_diag_x_pp;
    G4int sec_diag_x_dof;
    G4bool sec_diag_x_fit_successful;
    
    // Secondary diagonal Y fit results
    G4double sec_diag_y_center;
    G4double sec_diag_y_beta;
    G4double sec_diag_y_lambda;
    G4double sec_diag_y_gamma;
    G4double sec_diag_y_amplitude;
    G4double sec_diag_y_center_err;
    G4double sec_diag_y_beta_err;
    G4double sec_diag_y_lambda_err;
    G4double sec_diag_y_gamma_err;
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
    DiagonalSkewedLorentzianFitResultsCeres() : 
        main_diag_x_center(0), main_diag_x_beta(1), main_diag_x_lambda(0), main_diag_x_gamma(1), main_diag_x_amplitude(0),
        main_diag_x_center_err(0), main_diag_x_beta_err(0), main_diag_x_lambda_err(0), main_diag_x_gamma_err(0), main_diag_x_amplitude_err(0),
        main_diag_x_vertical_offset(0), main_diag_x_vertical_offset_err(0),
        main_diag_x_chi2red(0), main_diag_x_pp(0), main_diag_x_dof(0), main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_beta(1), main_diag_y_lambda(0), main_diag_y_gamma(1), main_diag_y_amplitude(0),
        main_diag_y_center_err(0), main_diag_y_beta_err(0), main_diag_y_lambda_err(0), main_diag_y_gamma_err(0), main_diag_y_amplitude_err(0),
        main_diag_y_vertical_offset(0), main_diag_y_vertical_offset_err(0),
        main_diag_y_chi2red(0), main_diag_y_pp(0), main_diag_y_dof(0), main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_beta(1), sec_diag_x_lambda(0), sec_diag_x_gamma(1), sec_diag_x_amplitude(0),
        sec_diag_x_center_err(0), sec_diag_x_beta_err(0), sec_diag_x_lambda_err(0), sec_diag_x_gamma_err(0), sec_diag_x_amplitude_err(0),
        sec_diag_x_vertical_offset(0), sec_diag_x_vertical_offset_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_pp(0), sec_diag_x_dof(0), sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_beta(1), sec_diag_y_lambda(0), sec_diag_y_gamma(1), sec_diag_y_amplitude(0),
        sec_diag_y_center_err(0), sec_diag_y_beta_err(0), sec_diag_y_lambda_err(0), sec_diag_y_gamma_err(0), sec_diag_y_amplitude_err(0),
        sec_diag_y_vertical_offset(0), sec_diag_y_vertical_offset_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_pp(0), sec_diag_y_dof(0), sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Function to remove outliers
SkewedLorentzianOutlierRemovalResult RemoveSkewedLorentzianOutliers(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& charge_values,
    bool enable_outlier_removal,
    double sigma_threshold = 2.5,
    bool verbose = false
);

// Function to perform 2D Skewed Lorentzian fitting
// Function form: y(x) = A * (1/(1+(x-m)²)^β) * (1 + λ(x-m)/(1+γ(x-m)²)) + B
SkewedLorentzianFit2DResultsCeres Fit2DSkewedLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

// Function to perform diagonal Skewed Lorentzian fitting
DiagonalSkewedLorentzianFitResultsCeres FitDiagonalSkewedLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // SKEWEDLORENTZIANFITCERES2D_HH
 