/// @file Config.hh
/// @brief Simulation configuration - edit values in USER SETTINGS section
/// @see docs/configuration.md for detailed documentation

#ifndef ECS_CONFIG_HH
#define ECS_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "globals.hh"
#include <cmath>

namespace Constants {

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                           USER SETTINGS                                   ║
// ║               Edit values below to configure the simulation               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

// ─────────────────────────────── Mode Selection ─────────────────────────────
// Choose reconstruction mode: LogA, LinA, or DPC
//   LogA - Logarithmic attenuation model (paper Eq. (\ref{eq:masterformula}))
//   LinA - Linear attenuation model (paper Eq. (\ref{eq:LA}))
//   DPC  - Discretized Positioning Circuit (paper Section 3.4; fast, no fitting)
enum class Mode { LogA, LinA, DPC };

inline constexpr Mode ACTIVE_MODE = Mode::LogA;

// For DPC mode only: which signal sharing model to use for charge calculation
inline constexpr Mode DPC_CHARGE_MODEL = Mode::LogA;

// ───────────────────────────── Detector Geometry ────────────────────────────
inline const G4double DETECTOR_SIZE        = 30.0 * mm;    // Sensor side length
inline const G4double DETECTOR_WIDTH       = 0.05 * mm;    // Silicon thickness
inline const G4double PIXEL_SIZE           = 0.15 * mm;    // Pixel pad size
inline const G4double PIXEL_PITCH          = 0.5  * mm;    // Pixel spacing (pitch)
inline constexpr G4int NEIGHBORHOOD_RADIUS = 2;            // Charge sharing radius
inline const G4double PIXEL_THICKNESS      = 0.02 * mm;    // Pixel depth/thickness

// DD4hep-style grid offset: position = index * pitch + offset
// Set to 0.0 for centered grid (indices can be negative), or non-zero to shift grid origin
inline const G4double GRID_OFFSET          = 0.0  * mm;    // Grid origin offset

// ─────────────────────────────── Physics ────────────────────────────────────
inline constexpr G4double IONIZATION_ENERGY = 3.6;         // eV per e-h pair (silicon at room temperature)
inline constexpr G4double GAIN              = 20.0;        // AC-LGAD gain factor; typical range 8-25
inline constexpr G4double D0                = 1.0;         // LogA reference distance d0 (um); controls charge spread width

// ────────────────────────────── Noise Model ─────────────────────────────────
// Gain sigma bounds: typical AC-LGAD per-pixel gain variation range from lab measurements
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.010;    // Min gain noise (1%)
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.050;    // Max gain noise (5%)
inline constexpr G4double NOISE_ELECTRON_COUNT = 500.0;    // Electronic noise (e-)

// ────────────────────────────── DPC Tuning ──────────────────────────────────
inline constexpr G4double DPC_K_CALIBRATION = 1.2;         // Empirical calibration constant from DPC reconstruction tuning (k = interpad * this)

// ───────────────────────────── Linear Model ─────────────────────────────────
// LinA attenuation coefficient beta in 1/um (paper Eq. (\ref{eq:LA})).
inline constexpr G4double LINEAR_CHARGE_MODEL_BETA = 0.002;

// ──────────────────────────── Particle Gun ──────────────────────────────────
// Use fixed position (true) or random sampling (false) for primary particles
inline constexpr G4bool USE_FIXED_POSITION = true;

// ─────────────────────────── Active Pixel Mode ─────────────────────────────
// Selects which pixels are "active" for signal fraction (F_i) calculation.
// The denominator sums F_i only over active pixels.
//
// 1D-compatible modes (work with FIT_GAUS_1D or FIT_GAUS_2D):
enum class ActivePixelMode1D { Neighborhood, RowCol, RowCol3x3 };
//   Neighborhood - All pixels in the neighborhood
//   RowCol       - Cross pattern (center row + center column)
//   RowCol3x3    - Cross pattern + 3x3 center block
//
// 2D-only modes (require FIT_GAUS_2D = true):
enum class ActivePixelMode2D { Neighborhood, ChargeBlock2x2, ChargeBlock3x3 };
//   Neighborhood   - All pixels in the neighborhood
//   ChargeBlock2x2 - 4 pads with highest weight
//   ChargeBlock3x3 - 9 pads with highest weight

inline constexpr ActivePixelMode1D ACTIVE_PIXEL_MODE_1D = ActivePixelMode1D::Neighborhood;

// 2D-only active pixel selection (used when FIT_GAUS_2D = true)
inline constexpr ActivePixelMode2D ACTIVE_PIXEL_MODE_2D = ActivePixelMode2D::Neighborhood;

// ─────────────────────────────── Fitting ───────────────────────────────────
// Enable Gaussian fitting for position reconstruction
// Note: To use ActivePixelMode2D, you must set FIT_GAUS_2D = true
inline constexpr G4bool FIT_GAUS_1D = true;
inline constexpr G4bool FIT_GAUS_2D = false;

// To switch to the 3x3 variant, update `ACTIVE_PIXEL_MODE_2D` above.

// ─────────────────────────── Fit Uncertainty Model ───────────────────────────
// Controls how per-pixel uncertainties (sigma) are assigned in chi-squared fits.
inline constexpr G4double FIT_ERROR_PERCENT_OF_MAX       = 5.0;   // Base uncertainty (%)
inline constexpr G4bool   FIT_USE_VERTICAL_UNCERTAINTIES = true;  // Enable weighted fits
inline constexpr G4bool   FIT_USE_QN_QI_ERRORS           = false; // Use Q_n/Q_i scaling

// Distance-weighted errors (pixels farther from hit center have different weights)
inline constexpr G4bool   FIT_USE_DISTANCE_WEIGHTED_ERRORS   = false; // Master switch
inline constexpr G4double FIT_DISTANCE_SCALE_PIXELS          = 1.0;   // Scale in pixel units
inline constexpr G4double FIT_DISTANCE_EXPONENT              = 1.0;   // Power law exponent
inline constexpr G4double FIT_DISTANCE_FLOOR_PERCENT         = 1.0;   // Min sigma (% of Q_max)
inline constexpr G4double FIT_DISTANCE_CAP_PERCENT           = 50.0;  // Max sigma (% of Q_max)
inline constexpr G4bool   FIT_DISTANCE_PREFER_TRUTH_CENTER   = true;  // Use true hit position
inline constexpr G4bool   FIT_DISTANCE_POWER_INVERSE         = true;  // Inverse power model

// Input charge branch for fitting
inline constexpr const char* FIT_CHARGE_BRANCH_1D = "Qf";
inline constexpr const char* FIT_CHARGE_BRANCH_2D = "Qf";

// ─────────────────────────────── 1D Fit Options ──────────────────────────────
inline constexpr G4bool FIT_1D_DIAGONALS       = false;  // Also fit diagonal slices
inline constexpr G4bool FIT_1D_SAVE_A          = true;   // Save amplitude
inline constexpr G4bool FIT_1D_SAVE_MU         = true;   // Save mean position
inline constexpr G4bool FIT_1D_SAVE_SIGMA      = true;   // Save width
inline constexpr G4bool FIT_1D_SAVE_B          = true;   // Save baseline offset
inline constexpr G4bool FIT_1D_SAVE_LINE_MEANS = false;  // Save weighted mean positions

// 1D diagonal fit parameter saving (when FIT_1D_DIAGONALS = true)
inline constexpr G4bool FIT_1D_SAVE_DIAG_A     = true;
inline constexpr G4bool FIT_1D_SAVE_DIAG_MU    = true;
inline constexpr G4bool FIT_1D_SAVE_DIAG_SIGMA = true;
inline constexpr G4bool FIT_1D_SAVE_DIAG_B     = true;

// 1D distance model tuning (can differ from 2D)
inline constexpr G4bool   FIT_1D_DIST_ENABLED       = true;
inline constexpr G4double FIT_1D_DIST_SCALE_PIXELS  = 1.5;
inline constexpr G4double FIT_1D_DIST_EXPONENT      = 1.5;
inline constexpr G4double FIT_1D_DIST_FLOOR_PERCENT = 4.0;
inline constexpr G4double FIT_1D_DIST_CAP_PERCENT   = 10.0;

// ─────────────────────────────── 2D Fit Options ──────────────────────────────
inline constexpr G4bool FIT_2D_SAVE_A    = true;   // Save amplitude
inline constexpr G4bool FIT_2D_SAVE_MUX  = true;   // Save mean X position
inline constexpr G4bool FIT_2D_SAVE_MUY  = true;   // Save mean Y position
inline constexpr G4bool FIT_2D_SAVE_SIGX = true;   // Save width in X
inline constexpr G4bool FIT_2D_SAVE_SIGY = true;   // Save width in Y
inline constexpr G4bool FIT_2D_SAVE_B    = true;   // Save baseline offset

// 2D distance model tuning (can differ from 1D)
inline constexpr G4bool   FIT_2D_DIST_ENABLED       = false;
inline constexpr G4double FIT_2D_DIST_SCALE_PIXELS  = 1.0;
inline constexpr G4double FIT_2D_DIST_EXPONENT      = 1.0;
inline constexpr G4double FIT_2D_DIST_FLOOR_PERCENT = 1.0;
inline constexpr G4double FIT_2D_DIST_CAP_PERCENT   = 50.0;

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    END USER SETTINGS - Internal Below                     ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

// ═══════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════

enum class SignalModel { LogA, LinA };
enum class ReconMethod { LogA, LinA, DPC };

// Unified enum for runtime use (combines both 1D and 2D modes)
enum class ActivePixelMode { Neighborhood, RowCol, RowCol3x3, ChargeBlock2x2, ChargeBlock3x3 };

using PosReconModel = ReconMethod;  // Legacy alias

// ═══════════════════════════════════════════════════════════════════════════
// Derived Settings (computed from ACTIVE_MODE)
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr G4bool IS_DPC_MODE = (ACTIVE_MODE == Mode::DPC);

inline constexpr ReconMethod RECON_METHOD =
    (ACTIVE_MODE == Mode::LogA) ? ReconMethod::LogA :
    (ACTIVE_MODE == Mode::LinA) ? ReconMethod::LinA :
                                    ReconMethod::DPC;

inline constexpr SignalModel SIGNAL_MODEL =
    IS_DPC_MODE ? (DPC_CHARGE_MODEL == Mode::LinA ? SignalModel::LinA : SignalModel::LogA)
                : (ACTIVE_MODE == Mode::LinA ? SignalModel::LinA : SignalModel::LogA);

inline constexpr G4bool USES_LINEAR_SIGNAL = (SIGNAL_MODEL == SignalModel::LinA);

inline constexpr PosReconModel POS_RECON_MODEL = RECON_METHOD;

// Map 1D mode enum to unified enum
inline constexpr ActivePixelMode ActivePixelModeFrom1D(ActivePixelMode1D m) {
    return (m == ActivePixelMode1D::Neighborhood) ? ActivePixelMode::Neighborhood :
           (m == ActivePixelMode1D::RowCol)       ? ActivePixelMode::RowCol :
                                                    ActivePixelMode::RowCol3x3;
}

// Map 2D mode enum to unified enum
inline constexpr ActivePixelMode ActivePixelModeFrom2D(ActivePixelMode2D m) {
    return (m == ActivePixelMode2D::Neighborhood)   ? ActivePixelMode::Neighborhood :
           (m == ActivePixelMode2D::ChargeBlock2x2) ? ActivePixelMode::ChargeBlock2x2 :
                                                      ActivePixelMode::ChargeBlock3x3;
}

// The active pixel mode used at runtime
// When FIT_GAUS_2D is enabled, use the 2D mode; otherwise use the 1D mode
inline constexpr ActivePixelMode ACTIVE_PIXEL_MODE =
    IS_DPC_MODE ? ActivePixelMode::Neighborhood
                : (FIT_GAUS_2D ? ActivePixelModeFrom2D(ACTIVE_PIXEL_MODE_2D)
                              : ActivePixelModeFrom1D(ACTIVE_PIXEL_MODE_1D));

// Full grid storage: disabled by default (saves only neighborhood/block/strip data)
// Set to true if you need per-event full-detector charge fractions
inline constexpr G4bool STORE_FULL_GRID = false;

// DPC uses the 4 closest pads (paper Section 3.4)
inline constexpr G4int DPC_TOP_N_PADS = 4;
inline constexpr G4int DPC_TOP_N_PIXELS = DPC_TOP_N_PADS;  // Backward-compatible alias

// ChargeBlock modes (2x2, 3x3) require 2D fitting to be enabled
inline constexpr G4bool IS_CHARGE_BLOCK_MODE =
    (ACTIVE_PIXEL_MODE == ActivePixelMode::ChargeBlock2x2 ||
     ACTIVE_PIXEL_MODE == ActivePixelMode::ChargeBlock3x3);
static_assert(!IS_CHARGE_BLOCK_MODE || FIT_GAUS_2D,
    "ChargeBlock2x2 and ChargeBlock3x3 active pixel modes require FIT_GAUS_2D = true");

// ═══════════════════════════════════════════════════════════════════════════
// Additional Geometry & Physics Constants
// ═══════════════════════════════════════════════════════════════════════════

inline const G4double WORLD_SIZE = 5.0 * cm;
inline const G4double DETECTOR_Z_POSITION = -1.0 * cm;

inline const G4double MAX_STEP_SIZE = 20.0 * micrometer;
inline const G4double PRIMARY_PARTICLE_Z_POSITION = 0.0 * cm;

inline const G4double GEOMETRY_TOLERANCE = 1.0 * um;
inline const G4double PRECISION_TOLERANCE = 1.0 * nm;

inline constexpr G4double ELEMENTARY_CHARGE = 1.602176634e-19;
inline constexpr G4double OUT_OF_BOUNDS_FRACTION_SENTINEL = -999.0;

// ═══════════════════════════════════════════════════════════════════════════
// DD4hep-style Grid Helper Functions
// ═══════════════════════════════════════════════════════════════════════════
//
// These functions implement the DD4hep CartesianGrid formulas:
//   position = index * gridSize + offset
//   index = floor((position + 0.5 * gridSize - offset) / gridSize)
//
// This allows for centered grids where indices can be negative.

/// @brief Convert grid index to position (DD4hep-style: position = index * pitch + offset)
/// @param index The grid index (can be negative for centered grids)
/// @param gridSize The grid pitch/spacing
/// @param offset The grid origin offset (default 0.0 for centered grid)
/// @return Position in world coordinates
inline G4double IndexToPosition(G4int index, G4double gridSize, G4double offset = 0.0) {
    return static_cast<G4double>(index) * gridSize + offset;
}

/// @brief Convert position to grid index (DD4hep-style: floor formula)
/// @param position Position in world coordinates
/// @param gridSize The grid pitch/spacing
/// @param offset The grid origin offset (default 0.0 for centered grid)
/// @return Grid index (can be negative for centered grids)
inline G4int PositionToIndex(G4double position, G4double gridSize, G4double offset = 0.0) {
    return static_cast<G4int>(std::floor((position + 0.5 * gridSize - offset) / gridSize));
}

} // namespace Constants

// Namespace aliases for backward compatibility
namespace Config = Constants;
namespace ECS { namespace Config = ::Constants; }

#endif // ECS_CONFIG_HH
