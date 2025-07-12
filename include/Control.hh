#ifndef CONTROL_HH
#define CONTROL_HH

#include "globals.hh"

namespace Control {
    
    // ========================
    // AUTOMATIC RADIUS SELECTION CONTROL
    // ========================
    
    const G4bool AUTO_RADIUS = false;             // Enable automatic radius selection per hit
    
    // ========================
    // FITTING MODEL CONTROL FLAGS
    // ========================
    
    // Enable/disable different fitting models
    const G4bool GAUSS_FIT = true;         // Enable Gauss fitting (2D and diagonal)
    const G4bool LORENTZ_FIT = true;       // Enable Lorentz fitting (2D and diagonal) 
    const G4bool POWER_LORENTZ_FIT = false; // Enable Power Lorentz fitting (2D and diagonal)
    
    // Individual fitting type control (only used if main model is enabled)
    const G4bool ROWCOL_FIT = true;               // Enable central row/column fitting
    const G4bool DIAG_FIT = false;         // Enable diagonal fitting
    
    // 3D fitting control flags (fit entire neighborhood surface directly)
    const G4bool GAUSS_FIT_3D = true;      // Enable 3D Gauss surface fitting
    const G4bool LORENTZ_FIT_3D = true;    // Enable 3D Lorentz surface fitting
    const G4bool POWER_LORENTZ_FIT_3D = false; // Enable 3D Power-Law Lorentz surface fitting
    
    // ========================
    // VERTICAL CHARGE UNCERTAINTIES CONTROL
    // ========================
    
    // Enable/disable vert charge uncertainties in fitting and ROOT output
    // When enabled: Uses 5% of max charge as err for weighted least squares fitting
    // When disabled: Uses uniform weighting (err = 1.0) for unweighted fitting
    // Also controls whether err values are saved to ROOT file branches
    const G4bool CHARGE_ERR = true;  // Enable charge uncertainties
    
    // ========================
    // ROOT AUTO-SAVE CONTROL
    // ========================

    // Frequent AutoSave (TTree::AutoSave/Flush) inside RunAction significantly increases the
    // probability of producing partially-written baskets when the simulation is killed or the
    // file system momentarily blocks.  Set this flag to 'false' to disable the per-interval
    // AutoSave and rely on a single FlushBaskets()+Write() at the end of the run instead.
    const G4bool ENABLE_AUTOSAVE = false;  // disable by default

} // namespace Control

#endif // CONTROL_HH 
