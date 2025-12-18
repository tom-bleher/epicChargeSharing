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

  ParameterRef<std::string> m_readout{this, "readout", config().readout};
  ParameterRef<float> m_minEDep{this, "minEDep", config().minEDepGeV};
  ParameterRef<int> m_neighborhoodRadius{this, "neighborhoodRadius", config().neighborhoodRadius};
  ParameterRef<double> m_pixelSize{this, "pixelSizeMM", config().pixelSizeMM};
  ParameterRef<double> m_pixelSpacing{this, "pixelSpacingMM", config().pixelSpacingMM};
  ParameterRef<double> m_pixelCornerOffset{this, "pixelCornerOffsetMM", config().pixelCornerOffsetMM};
  ParameterRef<double> m_detectorSize{this, "detectorSizeMM", config().detectorSizeMM};
  ParameterRef<double> m_detectorThickness{this, "detectorThicknessMM", config().detectorThicknessMM};
  ParameterRef<double> m_pixelThickness{this, "pixelThicknessMM", config().pixelThicknessMM};
  ParameterRef<double> m_detectorZCenter{this, "detectorZCenterMM", config().detectorZCenterMM};
  ParameterRef<int> m_pixelsPerSide{this, "pixelsPerSide", config().pixelsPerSide};
  ParameterRef<double> m_ionizationEnergy{this, "ionizationEnergyEV", config().ionizationEnergyEV};
  ParameterRef<double> m_amplificationFactor{this, "amplificationFactor", config().amplificationFactor};
  ParameterRef<double> m_elementaryCharge{this, "elementaryChargeC", config().elementaryChargeC};
  ParameterRef<double> m_d0Micron{this, "d0Micron", config().d0Micron};
  ParameterRef<bool> m_emitNeighborDiagnostics{this, "emitNeighborDiagnostics", config().emitNeighborDiagnostics};

  ChargeSharingReconstructor m_reconstructor;
};

}  // namespace epic::chargesharing

