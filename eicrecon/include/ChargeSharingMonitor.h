/// @file ChargeSharingMonitor.h
/// @brief JEventProcessor for monitoring charge sharing reconstruction performance.
///
/// Creates histograms and a TTree with reconstructed vs truth position data,
/// similar to the main simulation output format.

#pragma once

#include <JANA/JEventProcessorSequentialRoot.h>

#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>

#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>

#include <map>
#include <string>
#include <vector>

namespace epic::chargesharing {

/// Monitor processor for charge sharing reconstruction validation.
///
/// This processor compares reconstructed TrackerHit positions to the
/// original SimTrackerHit truth positions and produces:
/// - Residual histograms (recon - truth)
/// - 2D position scatter plots
/// - Per-hit TTree for detailed analysis
///
/// Usage:
///   eicrecon -Pplugins=chargeSharingRecon \
///            -PChargeSharingMonitor:inputRecoHits=B0ChargeSharingTrackerHits \
///            -PChargeSharingMonitor:inputSimHits=B0TrackerHits \
///            input.edm4hep.root
///
class ChargeSharingMonitor : public JEventProcessorSequentialRoot {
 public:
  ChargeSharingMonitor();
  ~ChargeSharingMonitor() override = default;

  void InitWithGlobalRootLock() override;
  void ProcessSequential(const std::shared_ptr<const JEvent>& event) override;
  void FinishWithGlobalRootLock() override;

 private:
  /// Create histograms for a given detector
  void CreateHistograms(const std::string& detectorName);

  /// Fill histograms and tree for matched hits
  void FillData(const std::string& detectorName,
                const edm4eic::TrackerHit& recoHit,
                const edm4hep::SimTrackerHit& simHit);

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration
  // ─────────────────────────────────────────────────────────────────────────

  /// List of detector configurations: {recoCollection, simCollection, name}
  struct DetectorConfig {
    std::string recoCollection;
    std::string simCollection;
    std::string displayName;
  };
  std::vector<DetectorConfig> m_detectors;

  // Parameters (set via JANA configuration)
  std::string m_inputRecoHits{"B0ChargeSharingTrackerHits"};
  std::string m_inputSimHits{"B0TrackerHits"};

  // ─────────────────────────────────────────────────────────────────────────
  // Histograms (per detector)
  // ─────────────────────────────────────────────────────────────────────────

  struct HistogramSet {
    // 1D residuals
    TH1D* hResidualX{nullptr};
    TH1D* hResidualY{nullptr};
    TH1D* hResidualR{nullptr};

    // 2D correlations
    TH2D* hRecoVsTrueX{nullptr};
    TH2D* hRecoVsTrueY{nullptr};
    TH2D* hTrueXY{nullptr};
    TH2D* hRecoXY{nullptr};
    TH2D* hResidualVsTrueX{nullptr};
    TH2D* hResidualVsTrueY{nullptr};

    // Energy and charge
    TH1D* hEnergyDeposit{nullptr};
  };

  std::map<std::string, HistogramSet> m_histograms;

  // ─────────────────────────────────────────────────────────────────────────
  // TTree for detailed analysis
  // ─────────────────────────────────────────────────────────────────────────

  TTree* m_tree{nullptr};

  // Tree branches - matching main simulation output naming
  double m_trueX{0.0};
  double m_trueY{0.0};
  double m_trueZ{0.0};
  double m_reconX{0.0};
  double m_reconY{0.0};
  double m_reconZ{0.0};
  double m_residualX{0.0};   // reconX - trueX
  double m_residualY{0.0};   // reconY - trueY
  double m_residualR{0.0};   // sqrt(residualX^2 + residualY^2)
  double m_edep{0.0};        // Energy deposit (GeV)
  double m_time{0.0};        // Hit time
  uint64_t m_cellID{0};      // Cell ID from hit
  int m_eventNumber{0};
  int m_detectorIndex{0};    // Which detector (0=B0, 1=BackwardMPGD, etc.)
};

}  // namespace epic::chargesharing
