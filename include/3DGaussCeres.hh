#ifndef GAUSSCERES3D_HH
#define GAUSSCERES3D_HH

#include <vector>

// Results structure for 3D Gauss fitting using Ceres
struct Gauss3DResultsCeres {
    // Input data storage
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<double> charge_values;
    std::vector<double> charge_errors;
    double charge_err = 0.0;
    
    //  parameters
    double amp = 0.0;
    double center_x = 0.0;
    double center_y = 0.0;
    double sigma_x = 0.0;
    double sigma_y = 0.0;
    double vert_offset = 0.0;
    
    // Parameter uncertainties
    double amp_err = 0.0;
    double center_x_err = 0.0;
    double center_y_err = 0.0;
    double sigma_x_err = 0.0;
    double sigma_y_err = 0.0;
    double vert_offset_err = 0.0;
    
    //  quality
    double chi2red = 0.0;
    int dof = 0;
    double pp = 0.0; // p-value
    bool fit_success = false;
    
    // Constructor
    Gauss3DResultsCeres() = default;
};

// Main fitting function
Gauss3DResultsCeres GaussCeres3D(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = true
);

#endif // GAUSSCERES3D_HH 