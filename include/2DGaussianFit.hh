#ifndef GAUSSIANFIT2D_HH
#define GAUSSIANFIT2D_HH

#include <vector>
#include "globals.hh"

// ROOT includes for fitting
#include "TF1.h"
#include "TH1F.h"
#include "TGraphErrors.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"

// Structure to hold 2D Gaussian fit results
struct GaussianFit2DResults {
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
    GaussianFit2DResults() : 
        x_center(0), x_sigma(0), x_amplitude(0),
        x_center_err(0), x_sigma_err(0), x_amplitude_err(0),
        x_chi2red(0), x_npoints(0),
        y_center(0), y_sigma(0), y_amplitude(0),
        y_center_err(0), y_sigma_err(0), y_amplitude_err(0),
        y_chi2red(0), y_npoints(0),
        fit_successful(false) {}
};

// Function to perform 2D Gaussian fitting using ROOT
// Fits central row and column separately with Gaussian functions
// Function form: y(x) = A * exp(-(x - m)^2 / (2 * sigma^2)) + B
GaussianFit2DResults Fit2DGaussian(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false
);

#endif // GAUSSIANFIT2D_HH 