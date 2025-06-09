#ifndef LORENTZIANFITROOT2D_HH
#define LORENTZIANFITROOT2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold 2D Lorentzian fit results using ROOT fitting
struct LorentzianFit2DResultsROOT {
    // X direction fit results (central row)
    G4double x_center;
    G4double x_gamma;                 // Using gamma instead of sigma for Lorentzian
    G4double x_amplitude;
    G4double x_center_err;
    G4double x_gamma_err;
    G4double x_amplitude_err;
    G4double x_vertical_offset;
    G4double x_vertical_offset_err;
    G4double x_chi2red;
    G4double x_pp;
    G4int x_dof;
    G4double x_fwhm;                  // Full Width at Half Maximum (2*gamma)
    G4int x_fit_status;               // ROOT fit status code
    G4double x_edm;                   // Estimated Distance to Minimum
    G4int x_ndf;                      // Number of degrees of freedom
    
    // Y direction fit results (central column)
    G4double y_center;
    G4double y_gamma;                 // Using gamma instead of sigma for Lorentzian
    G4double y_amplitude;
    G4double y_center_err;
    G4double y_gamma_err;
    G4double y_amplitude_err;
    G4double y_vertical_offset;
    G4double y_vertical_offset_err;
    G4double y_chi2red;
    G4double y_pp;
    G4int y_dof;
    G4double y_fwhm;                  // Full Width at Half Maximum (2*gamma)
    G4int y_fit_status;               // ROOT fit status code
    G4double y_edm;                   // Estimated Distance to Minimum
    G4int y_ndf;                      // Number of degrees of freedom
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    LorentzianFit2DResultsROOT() : 
        x_center(0), x_gamma(0), x_amplitude(0),
        x_center_err(0), x_gamma_err(0), x_amplitude_err(0),
        x_vertical_offset(0), x_vertical_offset_err(0),
        x_chi2red(0), x_pp(0), x_dof(0), x_fwhm(0),
        x_fit_status(-1), x_edm(0), x_ndf(0),
        y_center(0), y_gamma(0), y_amplitude(0),
        y_center_err(0), y_gamma_err(0), y_amplitude_err(0),
        y_vertical_offset(0), y_vertical_offset_err(0),
        y_chi2red(0), y_pp(0), y_dof(0), y_fwhm(0),
        y_fit_status(-1), y_edm(0), y_ndf(0),
        fit_successful(false) {}
};

// Structure to hold diagonal Lorentzian fit results using ROOT fitting
struct DiagonalLorentzianFitResultsROOT {
    // Main diagonal X fit results (X vs Charge for pixels on main diagonal)
    G4double main_diag_x_center;         
    G4double main_diag_x_gamma;          // Using gamma for Lorentzian
    G4double main_diag_x_amplitude;      
    G4double main_diag_x_center_err;     
    G4double main_diag_x_gamma_err;      
    G4double main_diag_x_amplitude_err;  
    G4double main_diag_x_vertical_offset;        
    G4double main_diag_x_vertical_offset_err;    
    G4double main_diag_x_chi2red;        
    G4double main_diag_x_pp;             
    G4int main_diag_x_dof;               
    G4double main_diag_x_fwhm;           // Full Width at Half Maximum (2*gamma)
    G4int main_diag_x_fit_status;        
    G4double main_diag_x_edm;            
    G4int main_diag_x_ndf;               
    G4bool main_diag_x_fit_successful;   
    
    // Main diagonal Y fit results (Y vs Charge for pixels on main diagonal)
    G4double main_diag_y_center;         
    G4double main_diag_y_gamma;          // Using gamma for Lorentzian
    G4double main_diag_y_amplitude;      
    G4double main_diag_y_center_err;     
    G4double main_diag_y_gamma_err;      
    G4double main_diag_y_amplitude_err;  
    G4double main_diag_y_vertical_offset;        
    G4double main_diag_y_vertical_offset_err;    
    G4double main_diag_y_chi2red;        
    G4double main_diag_y_pp;             
    G4int main_diag_y_dof;               
    G4double main_diag_y_fwhm;           // Full Width at Half Maximum (2*gamma)
    G4int main_diag_y_fit_status;        
    G4double main_diag_y_edm;            
    G4int main_diag_y_ndf;               
    G4bool main_diag_y_fit_successful;   
    
    // Secondary diagonal X fit results (X vs Charge for pixels on secondary diagonal)
    G4double sec_diag_x_center;          
    G4double sec_diag_x_gamma;           // Using gamma for Lorentzian
    G4double sec_diag_x_amplitude;       
    G4double sec_diag_x_center_err;      
    G4double sec_diag_x_gamma_err;       
    G4double sec_diag_x_amplitude_err;   
    G4double sec_diag_x_vertical_offset;         
    G4double sec_diag_x_vertical_offset_err;     
    G4double sec_diag_x_chi2red;         
    G4double sec_diag_x_pp;              
    G4int sec_diag_x_dof;                
    G4double sec_diag_x_fwhm;            // Full Width at Half Maximum (2*gamma)
    G4int sec_diag_x_fit_status;         
    G4double sec_diag_x_edm;             
    G4int sec_diag_x_ndf;                
    G4bool sec_diag_x_fit_successful;    
    
    // Secondary diagonal Y fit results (Y vs Charge for pixels on secondary diagonal)
    G4double sec_diag_y_center;          
    G4double sec_diag_y_gamma;           // Using gamma for Lorentzian
    G4double sec_diag_y_amplitude;       
    G4double sec_diag_y_center_err;      
    G4double sec_diag_y_gamma_err;       
    G4double sec_diag_y_amplitude_err;   
    G4double sec_diag_y_vertical_offset;         
    G4double sec_diag_y_vertical_offset_err;     
    G4double sec_diag_y_chi2red;         
    G4double sec_diag_y_pp;              
    G4int sec_diag_y_dof;                
    G4double sec_diag_y_fwhm;            // Full Width at Half Maximum (2*gamma)
    G4int sec_diag_y_fit_status;         
    G4double sec_diag_y_edm;             
    G4int sec_diag_y_ndf;                
    G4bool sec_diag_y_fit_successful;    
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    DiagonalLorentzianFitResultsROOT() : 
        main_diag_x_center(0), main_diag_x_gamma(0), main_diag_x_amplitude(0),
        main_diag_x_center_err(0), main_diag_x_gamma_err(0), main_diag_x_amplitude_err(0),
        main_diag_x_vertical_offset(0), main_diag_x_vertical_offset_err(0),
        main_diag_x_chi2red(0), main_diag_x_pp(0), main_diag_x_dof(0),
        main_diag_x_fwhm(0), main_diag_x_fit_status(-1), main_diag_x_edm(0), main_diag_x_ndf(0),
        main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_gamma(0), main_diag_y_amplitude(0),
        main_diag_y_center_err(0), main_diag_y_gamma_err(0), main_diag_y_amplitude_err(0),
        main_diag_y_vertical_offset(0), main_diag_y_vertical_offset_err(0),
        main_diag_y_chi2red(0), main_diag_y_pp(0), main_diag_y_dof(0),
        main_diag_y_fwhm(0), main_diag_y_fit_status(-1), main_diag_y_edm(0), main_diag_y_ndf(0),
        main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_gamma(0), sec_diag_x_amplitude(0),
        sec_diag_x_center_err(0), sec_diag_x_gamma_err(0), sec_diag_x_amplitude_err(0),
        sec_diag_x_vertical_offset(0), sec_diag_x_vertical_offset_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_pp(0), sec_diag_x_dof(0),
        sec_diag_x_fwhm(0), sec_diag_x_fit_status(-1), sec_diag_x_edm(0), sec_diag_x_ndf(0),
        sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_gamma(0), sec_diag_y_amplitude(0),
        sec_diag_y_center_err(0), sec_diag_y_gamma_err(0), sec_diag_y_amplitude_err(0),
        sec_diag_y_vertical_offset(0), sec_diag_y_vertical_offset_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_pp(0), sec_diag_y_dof(0),
        sec_diag_y_fwhm(0), sec_diag_y_fit_status(-1), sec_diag_y_edm(0), sec_diag_y_ndf(0),
        sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Function to perform comprehensive 2D Lorentzian fitting using ROOT TF1 with ultra-robust features
// Fits central row and column separately with Lorentzian functions
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B
// where m = center, γ = HWHM (Half Width at Half Maximum), A = amplitude, B = baseline
// Uses ROOT's Minuit/Minuit2/Fumili minimizers with multiple backup strategies and enhanced robustness
// Features:
// - Multiple parameter initialization strategies
// - Multiple minimizer algorithms (Minuit2, Minuit, Fumili, GSLMultiMin)
// - Multiple fitting options and strategies
// - Enhanced outlier detection and filtering
// - Comprehensive error analysis and validation
// - Physics-based parameter constraints and bounds
// - Robust statistical parameter estimation
LorentzianFit2DResultsROOT Fit2DLorentzianROOT(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false,
    const std::string& minimizer = "Minuit2",
    int max_iterations = 5000,
    double tolerance = 1e-6,
    bool use_weighted_fit = true
);

// Function to perform comprehensive diagonal Lorentzian fitting using ROOT TF1 with ultra-robust features
// Fits along main diagonal and secondary diagonal of the 9x9 grid
// Main diagonal: from bottom-left to top-right (slope = +1)
// Secondary diagonal: from top-left to bottom-right (slope = -1)
// Uses ROOT's Minuit/Minuit2/Fumili minimizers with multiple backup strategies and enhanced robustness
// Features:
// - Separate X and Y coordinate fits for each diagonal direction
// - Multiple parameter initialization strategies
// - Multiple minimizer algorithms with automatic fallbacks
// - Enhanced outlier detection specific to Lorentzian distributions
// - Comprehensive error analysis and goodness-of-fit metrics
// - Physics-based parameter constraints optimized for charge sharing
// - Robust statistical parameter estimation with charge-weighted centroids
DiagonalLorentzianFitResultsROOT FitDiagonalLorentzianROOT(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false,
    const std::string& minimizer = "Minuit2",
    int max_iterations = 5000,
    double tolerance = 1e-6,
    bool use_weighted_fit = true
);

#endif // LORENTZIANFITROOT2D_HH 