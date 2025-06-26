#ifndef THREEDGAUSSIANFITCERES_HH
#define THREEDGAUSSIANFITCERES_HH

#include <vector>

// Results structure for 3D Gaussian fitting using Ceres
struct GaussianFit3DResultsCeres {
    // Input data storage
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<double> charge_values;
    std::vector<double> charge_errors;
    double charge_uncertainty = 0.0;
    
    // Fit parameters
    double amplitude = 0.0;
    double center_x = 0.0;
    double center_y = 0.0;
    double sigma_x = 0.0;
    double sigma_y = 0.0;
    double vertical_offset = 0.0;
    
    // Parameter uncertainties
    double amplitude_err = 0.0;
    double center_x_err = 0.0;
    double center_y_err = 0.0;
    double sigma_x_err = 0.0;
    double sigma_y_err = 0.0;
    double vertical_offset_err = 0.0;
    
    // Fit quality
    double chi2red = 0.0;
    int dof = 0;
    double pp = 0.0; // p-value
    bool fit_successful = false;
    
    // Constructor
    GaussianFit3DResultsCeres() = default;
};

// Main fitting function
GaussianFit3DResultsCeres Fit3DGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = true
);

#endif // THREEDGAUSSIANFITCERES_HH 