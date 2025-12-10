/// @file Config.hh
/// @brief Simulation configuration - edit values in USER SETTINGS section
/// @see docs/configuration.md for detailed documentation

#ifndef ECS_CONFIG_HH
#define ECS_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "globals.hh"

namespace Constants {

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                           USER SETTINGS                                   ║
// ║               Edit values below to configure the simulation               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

// ─────────────────────────────── Mode Selection ─────────────────────────────
// Choose reconstruction mode: Log, Linear, or DPC
//   Log    - Logarithmic attenuation + Gaussian fitting
//   Linear - Linear attenuation + Gaussian fitting
//   DPC    - Discretized Positioning Circuit (fast, no fitting)
enum class Mode { Log, Linear, DPC };

inline constexpr Mode ACTIVE_MODE = Mode::Linear;

// For DPC mode only: which signal model to use for charge calculation
inline constexpr Mode DPC_CHARGE_MODEL = Mode::Log;

// ───────────────────────────── Detector Geometry ────────────────────────────
inline const G4double DETECTOR_SIZE        = 30.0 * mm;    // Sensor side length
inline const G4double DETECTOR_WIDTH       = 0.05 * mm;    // Silicon thickness
inline const G4double PIXEL_SIZE           = 0.1  * mm;    // Pixel pad size
inline const G4double PIXEL_PITCH          = 0.5  * mm;    // Pixel spacing
inline const G4double PIXEL_CORNER_OFFSET  = 0.1  * mm;    // Edge to first pixel
inline constexpr G4int NEIGHBORHOOD_RADIUS = 2;            // Charge sharing radius

// ─────────────────────────────── Physics ────────────────────────────────────
inline constexpr G4double IONIZATION_ENERGY = 3.6;         // eV per e-h pair
inline constexpr G4double GAIN              = 20.0;        // AC-LGAD gain (8-25)
inline constexpr G4double D0                = 1.0;         // Hit size d0 (um)

// ────────────────────────────── Noise Model ─────────────────────────────────
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.010;    // Min gain noise (1%)
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.050;    // Max gain noise (5%)
inline constexpr G4double NOISE_ELECTRON_COUNT = 500.0;    // Electronic noise (e-)

// ────────────────────────────── DPC Tuning ──────────────────────────────────
inline constexpr G4double DPC_K_CALIBRATION = 1.2;         // k = interpad * this

// ───────────────────────────── Linear Model ─────────────────────────────────
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
enum class ActivePixelMode2D { ChargeBlock2x2, ChargeBlock3x3 };
//   ChargeBlock2x2 - 4 pixels with highest F_i
//   ChargeBlock3x3 - 9 pixels with highest F_i

inline constexpr ActivePixelMode1D ACTIVE_PIXEL_MODE_1D = ActivePixelMode1D::RowCol;

// ─────────────────────────────── Fitting ───────────────────────────────────
// Enable Gaussian fitting for position reconstruction
// Note: To use ActivePixelMode2D, you must set FIT_GAUS_2D = true
inline constexpr G4bool FIT_GAUS_1D = false;
inline constexpr G4bool FIT_GAUS_2D = false;

// Uncomment the following to use 2D-only modes (requires FIT_GAUS_2D = true above):
// inline constexpr ActivePixelMode2D ACTIVE_PIXEL_MODE_2D = ActivePixelMode2D::ChargeBlock2x2;

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
    (ACTIVE_MODE == Mode::Log)    ? ReconMethod::LogA :
    (ACTIVE_MODE == Mode::Linear) ? ReconMethod::LinA :
                                    ReconMethod::DPC;

inline constexpr SignalModel SIGNAL_MODEL =
    IS_DPC_MODE ? (DPC_CHARGE_MODEL == Mode::Linear ? SignalModel::LinA : SignalModel::LogA)
                : (ACTIVE_MODE == Mode::Linear ? SignalModel::LinA : SignalModel::LogA);

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
    return (m == ActivePixelMode2D::ChargeBlock2x2) ? ActivePixelMode::ChargeBlock2x2 :
                                                      ActivePixelMode::ChargeBlock3x3;
}

// The active pixel mode used at runtime (derived from ACTIVE_PIXEL_MODE_1D)
// To use 2D modes, user must define ACTIVE_PIXEL_MODE_2D and change this line
inline constexpr ActivePixelMode ACTIVE_PIXEL_MODE = ActivePixelModeFrom1D(ACTIVE_PIXEL_MODE_1D);

// Full grid storage: disabled by default (saves only neighborhood/block/strip data)
// Set to true if you need per-event full-detector charge fractions
inline constexpr G4bool STORE_FULL_GRID = false;

// DPC always uses 4 closest pixels
inline constexpr G4int DPC_TOP_N_PIXELS = 4;

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
inline const G4double PIXEL_WIDTH = 0.001 * mm;
inline const G4double MAX_STEP_SIZE = 20.0 * micrometer;
inline const G4double PRIMARY_PARTICLE_Z_POSITION = 0.0 * cm;

inline const G4double GEOMETRY_TOLERANCE = 1.0 * um;
inline const G4double PRECISION_TOLERANCE = 1.0 * nm;

inline constexpr G4double ELEMENTARY_CHARGE = 1.602176634e-19;
inline constexpr G4double OUT_OF_BOUNDS_FRACTION_SENTINEL = -999.0;

} // namespace Constants

// Namespace aliases for backward compatibility
namespace Config = Constants;
namespace ECS { namespace Config = ::Constants; }

#endif // ECS_CONFIG_HH
