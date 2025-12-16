#pragma once

#include <string>

namespace epic::chargesharing {

struct SegmentationConfig {
  bool valid{false};
  double gridSizeXMM{0.0};
  double gridSizeYMM{0.0};
  double cellSizeXMM{0.0};
  double cellSizeYMM{0.0};
  double offsetXMM{0.0};
  double offsetYMM{0.0};
  int minIndexX{0};
  int maxIndexX{-1};
  int minIndexY{0};
  int maxIndexY{-1};
  int numCellsX{0};
  int numCellsY{0};
  std::string fieldNameX{"x"};
  std::string fieldNameY{"y"};
};

struct ChargeSharingConfig {
  std::string readout = "ChargeSharingRecon";
  float minEDepGeV = 0.0F;
  int neighborhoodRadius = 2;

  double pixelSizeMM = 0.1;          ///< Square pixel/pad side length
  double pixelSizeYMM = 0.1;         ///< Optional Y pixel size override
  double pixelSpacingMM = 0.5;       ///< Center-to-center pitch
  double pixelSpacingYMM = 0.5;      ///< Optional Y pitch override
  double pixelCornerOffsetMM = 0.1;  ///< Distance from detector edge to first pixel/pad center
  double detectorSizeMM = 30.0;
  double detectorThicknessMM = 0.05;
  double pixelThicknessMM = 0.001;
  double detectorZCenterMM = -10.0;  ///< Detector reference plane (mm)
  int pixelsPerSide = 0;             ///< Overrides inferred value when > 0

  SegmentationConfig segmentation{}; ///< Geometry extracted from DD4hep (if available)

  double ionizationEnergyEV = 3.6;         ///< eV per e-h pair in silicon
  double amplificationFactor = 20.0;       ///< Gain
  double elementaryChargeC = 1.602176634e-19;
  double d0Micron = 1.0;                   ///< Characteristic diffusion length

  bool emitNeighborDiagnostics = false;    ///< Record per-neighbor distances/angles
};

}  // namespace epic::chargesharing
