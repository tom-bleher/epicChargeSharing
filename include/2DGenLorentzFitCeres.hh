#ifndef GENLORENTZFITCERES2D_HH
#define GENLORENTZFITCERES2D_HH

#include <vector>
#include "globals.hh"

// Structure to hold outlier removal results for GenLorentz fitting
struct GenLorentzOutlierRemovalResult {
    std::vector<double> filtered_x_coords;
    std::vector<double> filtered_y_coords;
    std::vector<double> filtered_charge_values;
    int outliers_removed;
    bool filtering_applied;
    bool success;
    
    // Constructor with default values
    GenLorentzOutlierRemovalResult() : 
        outliers_removed(0), filtering_applied(false), success(false) {}
};

// Structure to hold 2D GenLorentz fit results
struct GenLorentzFit2DResultsCeres {
    // X direction fit results (central row)
    G4double x_center;               // m (center parameter)
    G4double x_gamma;                // γ (scale parameter)
    G4double x_amplitude;            // A (amplitude scaling factor)
    G4double x_power;                // n (power parameter)
    G4double x_center_err;           // m error
    G4double x_gamma_err;            // γ error
    G4double x_amplitude_err;        // A error
    G4double x_power_err;            // n error
    G4double x_vertical_offset;      // B (baseline/vertical offset)
    G4double x_vertical_offset_err;  // B error
    G4double x_chi2red;              // reduced chi-squared
    G4double x_pp;                   // p-value
    G4int x_dof;                     // degrees of freedom
    
    // Y direction fit results (central column)
    G4double y_center;               // m (center parameter)
    G4double y_gamma;                // γ (scale parameter)
    G4double y_amplitude;            // A (amplitude scaling factor)
    G4double y_power;                // n (power parameter)
    G4double y_center_err;           // m error
    G4double y_gamma_err;            // γ error
    G4double y_amplitude_err;        // A error
    G4double y_power_err;            // n error
    G4double y_vertical_offset;      // B (baseline/vertical offset)
    G4double y_vertical_offset_err;  // B error
    G4double y_chi2red;              // reduced chi-squared
    G4double y_pp;                   // p-value
    G4int y_dof;                     // degrees of freedom
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    GenLorentzFit2DResultsCeres() : 
        x_center(0), x_gamma(0), x_amplitude(0), x_power(2),
        x_center_err(0), x_gamma_err(0), x_amplitude_err(0), x_power_err(0),
        x_vertical_offset(0), x_vertical_offset_err(0),
        x_chi2red(0), x_pp(0), x_dof(0),
        y_center(0), y_gamma(0), y_amplitude(0), y_power(2),
        y_center_err(0), y_gamma_err(0), y_amplitude_err(0), y_power_err(0),
        y_vertical_offset(0), y_vertical_offset_err(0),
        y_chi2red(0), y_pp(0), y_dof(0),
        fit_successful(false) {}
};

// Structure to hold diagonal GenLorentz fit results
struct DiagonalGenLorentzFitResultsCeres {
    // Main diagonal X fit results
    G4double main_diag_x_center;         
    G4double main_diag_x_gamma;          
    G4double main_diag_x_amplitude;      
    G4double main_diag_x_power;          
    G4double main_diag_x_center_err;     
    G4double main_diag_x_gamma_err;      
    G4double main_diag_x_amplitude_err;  
    G4double main_diag_x_power_err;      
    G4double main_diag_x_vertical_offset;        
    G4double main_diag_x_vertical_offset_err;    
    G4double main_diag_x_chi2red;        
    G4double main_diag_x_pp;             
    G4int main_diag_x_dof;               
    G4bool main_diag_x_fit_successful;   
    
    // Main diagonal Y fit results
    G4double main_diag_y_center;         
    G4double main_diag_y_gamma;          
    G4double main_diag_y_amplitude;      
    G4double main_diag_y_power;          
    G4double main_diag_y_center_err;     
    G4double main_diag_y_gamma_err;      
    G4double main_diag_y_amplitude_err;  
    G4double main_diag_y_power_err;      
    G4double main_diag_y_vertical_offset;        
    G4double main_diag_y_vertical_offset_err;    
    G4double main_diag_y_chi2red;        
    G4double main_diag_y_pp;             
    G4int main_diag_y_dof;               
    G4bool main_diag_y_fit_successful;   
    
    // Secondary diagonal X fit results
    G4double sec_diag_x_center;          
    G4double sec_diag_x_gamma;           
    G4double sec_diag_x_amplitude;       
    G4double sec_diag_x_power;           
    G4double sec_diag_x_center_err;      
    G4double sec_diag_x_gamma_err;       
    G4double sec_diag_x_amplitude_err;   
    G4double sec_diag_x_power_err;       
    G4double sec_diag_x_vertical_offset;         
    G4double sec_diag_x_vertical_offset_err;     
    G4double sec_diag_x_chi2red;         
    G4double sec_diag_x_pp;              
    G4int sec_diag_x_dof;                
    G4bool sec_diag_x_fit_successful;    
    
    // Secondary diagonal Y fit results
    G4double sec_diag_y_center;          
    G4double sec_diag_y_gamma;           
    G4double sec_diag_y_amplitude;       
    G4double sec_diag_y_power;           
    G4double sec_diag_y_center_err;      
    G4double sec_diag_y_gamma_err;       
    G4double sec_diag_y_amplitude_err;   
    G4double sec_diag_y_power_err;       
    G4double sec_diag_y_vertical_offset;         
    G4double sec_diag_y_vertical_offset_err;     
    G4double sec_diag_y_chi2red;         
    G4double sec_diag_y_pp;              
    G4int sec_diag_y_dof;                
    G4bool sec_diag_y_fit_successful;    
    
    // Overall success status
    G4bool fit_successful;
    
    // Constructor with default values
    DiagonalGenLorentzFitResultsCeres() : 
        main_diag_x_center(0), main_diag_x_gamma(0), main_diag_x_amplitude(0), main_diag_x_power(2),
        main_diag_x_center_err(0), main_diag_x_gamma_err(0), main_diag_x_amplitude_err(0), main_diag_x_power_err(0),
        main_diag_x_vertical_offset(0), main_diag_x_vertical_offset_err(0),
        main_diag_x_chi2red(0), main_diag_x_pp(0), main_diag_x_dof(0), main_diag_x_fit_successful(false),
        main_diag_y_center(0), main_diag_y_gamma(0), main_diag_y_amplitude(0), main_diag_y_power(2),
        main_diag_y_center_err(0), main_diag_y_gamma_err(0), main_diag_y_amplitude_err(0), main_diag_y_power_err(0),
        main_diag_y_vertical_offset(0), main_diag_y_vertical_offset_err(0),
        main_diag_y_chi2red(0), main_diag_y_pp(0), main_diag_y_dof(0), main_diag_y_fit_successful(false),
        sec_diag_x_center(0), sec_diag_x_gamma(0), sec_diag_x_amplitude(0), sec_diag_x_power(2),
        sec_diag_x_center_err(0), sec_diag_x_gamma_err(0), sec_diag_x_amplitude_err(0), sec_diag_x_power_err(0),
        sec_diag_x_vertical_offset(0), sec_diag_x_vertical_offset_err(0),
        sec_diag_x_chi2red(0), sec_diag_x_pp(0), sec_diag_x_dof(0), sec_diag_x_fit_successful(false),
        sec_diag_y_center(0), sec_diag_y_gamma(0), sec_diag_y_amplitude(0), sec_diag_y_power(2),
        sec_diag_y_center_err(0), sec_diag_y_gamma_err(0), sec_diag_y_amplitude_err(0), sec_diag_y_power_err(0),
        sec_diag_y_vertical_offset(0), sec_diag_y_vertical_offset_err(0),
        sec_diag_y_chi2red(0), sec_diag_y_pp(0), sec_diag_y_dof(0), sec_diag_y_fit_successful(false),
        fit_successful(false) {}
};

// Function declarations
GenLorentzOutlierRemovalResult RemoveGenLorentzOutliers(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& charge_values,
    bool enable_outlier_removal,
    double sigma_threshold = 2.5,
    bool verbose = false
);

GenLorentzFit2DResultsCeres Fit2DGenLorentzCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

DiagonalGenLorentzFitResultsCeres FitDiagonalGenLorentzCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false,
    bool enable_outlier_filtering = false
);

#endif // GENLORENTZFITCERES2D_HH 