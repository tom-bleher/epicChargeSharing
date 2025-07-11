#ifndef CONTROL_HH
#define CONTROL_HH

#include "globals.hh"

namespace Control {
    
    // ========================
    // AUTOMATIC RADIUS SELECTION CONTROL
    // ========================
    
    const G4bool ENABLE_AUTO_RADIUS = false;             // Enable automatic radius selection per hit
    
    // ========================
    // FITTING MODEL CONTROL FLAGS
    // ========================
    
    // Enable/disable different fitting models
    const G4bool ENABLE_GAUSS_FIT = true;         // Enable Gauss fitting (2D and diagonal)
    const G4bool ENABLE_LORENTZ_FIT = true;       // Enable Lorentz fitting (2D and diagonal) 
    const G4bool ENABLE_POWER_LORENTZ_FIT = true; // Enable Power Lorentz fitting (2D and diagonal)
    
    // Individual fitting type control (only used if main model is enabled)
    const G4bool ENABLE_ROWCOL_FIT = true;               // Enable central row/column fitting
    const G4bool ENABLE_DIAG_FIT = true;         // Enable diagonal fitting
    
    // 3D fitting control flags (fit entire neighborhood surface directly)
    const G4bool ENABLE_3D_GAUSS_FIT = true;      // Enable 3D Gauss surface fitting
    const G4bool ENABLE_3D_LORENTZ_FIT = true;    // Enable 3D Lorentz surface fitting
    const G4bool ENABLE_3D_POWER_LORENTZ_FIT = true; // Enable 3D Power-Law Lorentz surface fitting
    
    // ========================
    // VERTICAL CHARGE UNCERTAINTIES CONTROL
    // ========================
    
    // Enable/disable vert charge uncertainties in fitting and ROOT output
    // When enabled: Uses 5% of max charge as err for weighted least squares fitting
    // When disabled: Uses uniform weighting (err = 1.0) for unweighted fitting
    // Also controls whether err values are saved to ROOT file branches
    const G4bool ENABLE_VERT_CHARGE_ERR = true;  // Enable charge uncertainties
    
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