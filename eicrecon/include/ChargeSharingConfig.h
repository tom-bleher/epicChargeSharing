#pragma once

#include "ChargeSharingCore.hh"
#include <cmath>
#include <string>

namespace epic::chargesharing {

// Import enums from core
using core::SignalModel;
using core::ActivePixelMode;
using core::ReconMethod;

struct SegmentationConfig {
  bool valid{false};
  bool useXZCoordinates{false};  ///< True if using CartesianGridXZ (e.g., B0 tracker)
  double gridSizeXMM{0.0};
  double gridSizeYMM{0.0};       ///< For XZ segmentation, this is the Z grid size
  double cellSizeXMM{0.0};
  double cellSizeYMM{0.0};       ///< For XZ segmentation, this is the Z cell size
  double offsetXMM{0.0};
  double offsetYMM{0.0};         ///< For XZ segmentation, this is the Z offset
  int minIndexX{0};
  int maxIndexX{-1};
  int minIndexY{0};              ///< For XZ segmentation, this is the Z min index
  int maxIndexY{-1};             ///< For XZ segmentation, this is the Z max index
  int numCellsX{0};
  int numCellsY{0};              ///< For XZ segmentation, this is the Z cell count
  std::string fieldNameX{"x"};
  std::string fieldNameY{"y"};   ///< For XZ segmentation, this is "z"
};

struct ChargeSharingConfig {
  // ───────────────────────────── Mode Selection ─────────────────────────────
  /// Signal sharing model: LogA (logarithmic) or LinA (linear attenuation)
  SignalModel signalModel{SignalModel::LogA};

  /// Active pixel mode for denominator calculation
  ActivePixelMode activePixelMode{ActivePixelMode::Neighborhood};

  /// Position reconstruction method
  ReconMethod reconMethod{ReconMethod::Gaussian1D};

  // ───────────────────────────── DD4hep Integration ─────────────────────────
  std::string readout = "ChargeSharingRecon";
  float minEDepGeV = 0.0F;

  // ───────────────────────────── Detector Geometry ────────────────────────────
  int neighborhoodRadius = 2;          ///< Charge sharing radius (2 = 5x5 grid)
  double pixelSizeMM = 0.15;           ///< Pixel pad size (mm)
  double pixelSizeYMM = 0.0;           ///< Y pixel size override (0 = same as X)
  double pixelSpacingMM = 0.5;         ///< Center-to-center pitch (mm)
  double pixelSpacingYMM = 0.0;        ///< Y pitch override (0 = same as X)
  double gridOffsetMM = 0.0;           ///< DD4hep-style grid offset (mm)
  double gridOffsetYMM = 0.0;          ///< Y grid offset override (0 = same as X)
  double detectorSizeMM = 30.0;        ///< Sensor side length (mm)
  double detectorThicknessMM = 0.05;   ///< Silicon thickness (mm)
  double pixelThicknessMM = 0.02;      ///< Pixel depth (mm)
  double detectorZCenterMM = -10.0;    ///< Detector reference plane (mm)
  int pixelsPerSide = 0;               ///< Overrides inferred value when > 0

  SegmentationConfig segmentation{};   ///< Geometry from DD4hep (if available)

  // ─────────────────────────────── Physics ────────────────────────────────────
  double ionizationEnergyEV = 3.6;           ///< eV per e-h pair in silicon
  double amplificationFactor = 20.0;         ///< AC-LGAD gain (8-25 typical)
  double elementaryChargeC = 1.602176634e-19;
  double d0Micron = 1.0;                     ///< Transverse hit size d0 (um)
  double linearBetaPerMicron = 0.0;          ///< LinA beta (1/um), 0 = auto from pitch

  // ─────────────────────────────── Noise Model ─────────────────────────────────
  /// Enable noise injection for realistic charge simulation
  bool noiseEnabled = true;

  /// Per-pixel gain variation (multiplicative noise)
  /// Each pixel's gain varies as: gain * (1 + gaussian(0, gainSigma))
  double noiseGainSigmaMin = 0.01;           ///< Min gain sigma (1% = 0.01)
  double noiseGainSigmaMax = 0.05;           ///< Max gain sigma (5% = 0.05)

  /// Electronic noise (additive, in electrons)
  /// Added as gaussian noise with this RMS value
  double noiseElectronCount = 500.0;         ///< Electronic noise RMS (electrons)

  /// Random seed for reproducibility (0 = use random seed)
  unsigned int noiseSeed = 0;

  // ─────────────────────────────── Fitting ───────────────────────────────────
  /// Error percentage of max charge for fit uncertainty
  double fitErrorPercentOfMax = 5.0;

  /// Enable vertical uncertainties in fits
  bool fitUseVerticalUncertainties = true;

  /// Distance-weighted errors: pixels farther from hit center have different weights
  bool fitUseDistanceWeightedErrors = false;     ///< Master switch (disabled by default)
  double fitDistanceScalePixels = 1.0;           ///< Scale in pixel units
  double fitDistanceExponent = 1.0;              ///< Power law exponent
  double fitDistanceFloorPercent = 1.0;          ///< Min sigma (% of Q_max)
  double fitDistanceCapPercent = 50.0;           ///< Max sigma (% of Q_max)
  bool fitDistancePreferTruthCenter = true;      ///< Use true hit position for distance calc
  bool fitDistancePowerInverse = true;           ///< Use inverse power model

  // ────────────────────────────── Diagnostics ─────────────────────────────────
  bool emitNeighborDiagnostics = false;      ///< Record per-neighbor distances/angles

  // ═══════════════════════════════════════════════════════════════════════════
  // Helper Methods
  // ═══════════════════════════════════════════════════════════════════════════

  /// Get effective Y pixel size (uses X if Y not specified)
  double effectivePixelSizeYMM() const {
    return (pixelSizeYMM > 0.0) ? pixelSizeYMM : pixelSizeMM;
  }

  /// Get effective Y pixel spacing (uses X if Y not specified)
  double effectivePixelSpacingYMM() const {
    return (pixelSpacingYMM > 0.0) ? pixelSpacingYMM : pixelSpacingMM;
  }

  /// Get effective X grid offset (from DD4hep segmentation or manual config)
  double effectiveGridOffsetXMM() const {
    if (segmentation.valid) {
      return segmentation.offsetXMM;
    }
    return gridOffsetMM;
  }

  /// Get effective Y grid offset (from DD4hep segmentation or manual config)
  double effectiveGridOffsetYMM() const {
    if (segmentation.valid) {
      return segmentation.offsetYMM;
    }
    return (gridOffsetYMM != 0.0) ? gridOffsetYMM : gridOffsetMM;
  }

  /// DD4hep-style: Convert grid index to position
  /// position = index * pitch + offset
  static double indexToPosition(int index, double pitch, double offset) {
    return static_cast<double>(index) * pitch + offset;
  }

  /// DD4hep-style: Convert position to grid index
  /// index = floor((position + 0.5*pitch - offset) / pitch)
  static int positionToIndex(double position, double pitch, double offset) {
    return static_cast<int>(std::floor((position + 0.5 * pitch - offset) / pitch));
  }

  /// Get sigma bounds for Gaussian fitting
  double sigmaLoBound() const {
    return pixelSizeMM;
  }

  double sigmaHiBound() const {
    return static_cast<double>(neighborhoodRadius) * pixelSpacingMM;
  }

  /// Get mu bounds for Gaussian fitting (centered on pixel)
  double muRangeMM() const {
    return 1.0 * pixelSpacingMM;  // Allow +/- 1 pixel from center
  }
};

}  // namespace epic::chargesharing
