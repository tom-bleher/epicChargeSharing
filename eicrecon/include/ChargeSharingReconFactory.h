#pragma once

#include "ChargeSharingConfig.h"
#include "ChargeSharingReconstructor.h"

#include "extensions/jana/JOmniFactory.h"
#include "services/geometry/dd4hep/DD4hep_service.h"

#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>

namespace epic::chargesharing {

class ChargeSharingReconFactory : public JOmniFactory<ChargeSharingReconFactory, ChargeSharingConfig> {
 public:
  using AlgoT = ChargeSharingReconstructor;

  void Configure();
  void Process(int32_t runNumber, uint64_t eventNumber);

 private:
  PodioInput<edm4hep::SimTrackerHit> m_in_simhits{this};
  PodioOutput<edm4eic::TrackerHit> m_out_hits{this};

  Service<DD4hep_service> m_dd4hep{this};

  // ─────────────────────────────── Mode Selection ─────────────────────────────
  // Storage for enum-backed integer parameters
  int m_signalModelValue{0};       // 0=LogA, 1=LinA
  int m_activePixelModeValue{0};   // 0=Neighborhood, 1=RowCol, 2=RowCol3x3, 3=ChargeBlock2x2, 4=ChargeBlock3x3
  int m_reconMethodValue{1};       // 0=Centroid, 1=Gaussian1D, 2=Gaussian2D

  ParameterRef<int> m_signalModel{this, "signalModel", m_signalModelValue,
      "Signal model: 0=LogA (logarithmic), 1=LinA (linear attenuation)"};
  ParameterRef<int> m_activePixelMode{this, "activePixelMode", m_activePixelModeValue,
      "Active pixel mode: 0=Neighborhood, 1=RowCol, 2=RowCol3x3, 3=ChargeBlock2x2, 4=ChargeBlock3x3"};
  ParameterRef<int> m_reconMethod{this, "reconMethod", m_reconMethodValue,
      "Reconstruction method: 0=Centroid, 1=Gaussian1D, 2=Gaussian2D"};

  // ─────────────────────────────── DD4hep Integration ─────────────────────────
  ParameterRef<std::string> m_readout{this, "readout", config().readout,
      "DD4hep readout name for segmentation lookup"};
  ParameterRef<float> m_minEDep{this, "minEDep", config().minEDepGeV,
      "Minimum energy deposit threshold (GeV)"};

  // ─────────────────────────────── Detector Geometry ────────────────────────────
  ParameterRef<int> m_neighborhoodRadius{this, "neighborhoodRadius", config().neighborhoodRadius,
      "Neighborhood half-width (2 = 5x5 grid)"};
  ParameterRef<double> m_pixelSize{this, "pixelSizeMM", config().pixelSizeMM,
      "Pixel pad size (mm)"};
  ParameterRef<double> m_pixelSpacing{this, "pixelSpacingMM", config().pixelSpacingMM,
      "Center-to-center pixel pitch (mm)"};
  ParameterRef<double> m_gridOffset{this, "gridOffsetMM", config().gridOffsetMM,
      "DD4hep-style grid offset (mm), 0 = centered grid"};
  ParameterRef<double> m_detectorSize{this, "detectorSizeMM", config().detectorSizeMM,
      "Sensor side length (mm)"};
  ParameterRef<double> m_detectorThickness{this, "detectorThicknessMM", config().detectorThicknessMM,
      "Silicon thickness (mm)"};
  ParameterRef<double> m_pixelThickness{this, "pixelThicknessMM", config().pixelThicknessMM,
      "Pixel depth (mm)"};
  ParameterRef<double> m_detectorZCenter{this, "detectorZCenterMM", config().detectorZCenterMM,
      "Detector reference plane (mm)"};
  ParameterRef<int> m_pixelsPerSide{this, "pixelsPerSide", config().pixelsPerSide,
      "Total pixels per side (0 = auto from geometry)"};

  // ─────────────────────────────── Physics ────────────────────────────────────
  ParameterRef<double> m_ionizationEnergy{this, "ionizationEnergyEV", config().ionizationEnergyEV,
      "Energy per e-h pair in silicon (eV)"};
  ParameterRef<double> m_amplificationFactor{this, "amplificationFactor", config().amplificationFactor,
      "AC-LGAD gain factor (8-25 typical)"};
  ParameterRef<double> m_elementaryCharge{this, "elementaryChargeC", config().elementaryChargeC,
      "Elementary charge (C)"};
  ParameterRef<double> m_d0Micron{this, "d0Micron", config().d0Micron,
      "Transverse hit size d0 for LogA model (um)"};
  ParameterRef<double> m_linearBeta{this, "linearBetaPerMicron", config().linearBetaPerMicron,
      "LinA attenuation coefficient (1/um), 0 = auto from pitch"};

  // ─────────────────────────────── Noise Model ─────────────────────────────────
  ParameterRef<bool> m_noiseEnabled{this, "noiseEnabled", config().noiseEnabled,
      "Enable noise injection for realistic charge simulation"};
  ParameterRef<double> m_noiseGainSigmaMin{this, "noiseGainSigmaMin", config().noiseGainSigmaMin,
      "Minimum per-pixel gain variation (1% = 0.01)"};
  ParameterRef<double> m_noiseGainSigmaMax{this, "noiseGainSigmaMax", config().noiseGainSigmaMax,
      "Maximum per-pixel gain variation (5% = 0.05)"};
  ParameterRef<double> m_noiseElectronCount{this, "noiseElectronCount", config().noiseElectronCount,
      "Electronic noise RMS (electrons)"};

  // ─────────────────────────────── Fitting ───────────────────────────────────
  ParameterRef<double> m_fitErrorPercent{this, "fitErrorPercentOfMax", config().fitErrorPercentOfMax,
      "Fit uncertainty as percentage of max charge"};
  ParameterRef<bool> m_fitUseVerticalUncertainties{this, "fitUseVerticalUncertainties",
      config().fitUseVerticalUncertainties, "Enable vertical uncertainties in fits"};
  ParameterRef<bool> m_fitUseDistanceWeightedErrors{this, "fitUseDistanceWeightedErrors",
      config().fitUseDistanceWeightedErrors, "Enable distance-weighted fit errors"};
  ParameterRef<double> m_fitDistanceScalePixels{this, "fitDistanceScalePixels",
      config().fitDistanceScalePixels, "Distance scale in pixel units"};
  ParameterRef<double> m_fitDistanceExponent{this, "fitDistanceExponent",
      config().fitDistanceExponent, "Distance error power law exponent"};
  ParameterRef<double> m_fitDistanceFloorPercent{this, "fitDistanceFloorPercent",
      config().fitDistanceFloorPercent, "Minimum fit error (% of Q_max)"};
  ParameterRef<double> m_fitDistanceCapPercent{this, "fitDistanceCapPercent",
      config().fitDistanceCapPercent, "Maximum fit error (% of Q_max)"};

  // ────────────────────────────── Diagnostics ─────────────────────────────────
  ParameterRef<bool> m_emitNeighborDiagnostics{this, "emitNeighborDiagnostics",
      config().emitNeighborDiagnostics, "Record per-neighbor distances/angles"};

  ChargeSharingReconstructor m_reconstructor;
};

}  // namespace epic::chargesharing

