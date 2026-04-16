// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include <JANA/JEventProcessor.h>

#include <edm4eic/TrackerHitCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>

#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>

#include <map>
#include <string>
#include <vector>

namespace eicrecon {

/// JEventProcessor that produces truth-residual benchmark histograms for the
/// LGAD charge-sharing reconstruction. Histograms and the companion TTree are
/// written into the shared RootFile_service hist file (-Phistsfile=...) under
/// the LGADChargeSharing/ TDirectory.
class LGADChargeSharingMonitor : public JEventProcessor {
public:
    LGADChargeSharingMonitor();
    ~LGADChargeSharingMonitor() override = default;

    void Init() override;
    void ProcessSequential(const JEvent& event) override;
    void Finish() override;

private:
    void createHistograms(const std::string& detectorName);
    void fillData(const std::string& detectorName, const edm4eic::TrackerHit& recoHit,
                  const edm4hep::SimTrackerHit& simHit);

    struct DetectorConfig {
        std::string recoCollection;
        std::string simCollection;
        std::string displayName;
    };
    std::vector<DetectorConfig> m_detectors;

    struct HistogramSet {
        TH1D* hResidualX{nullptr};
        TH1D* hResidualY{nullptr};
        TH1D* hResidualR{nullptr};
        TH2D* hRecoVsTrueX{nullptr};
        TH2D* hRecoVsTrueY{nullptr};
        TH2D* hTrueXY{nullptr};
        TH2D* hRecoXY{nullptr};
        TH2D* hResidualVsTrueX{nullptr};
        TH2D* hResidualVsTrueY{nullptr};
        TH1D* hEnergyDeposit{nullptr};
    };
    std::map<std::string, HistogramSet> m_histograms;

    TTree* m_tree{nullptr};
    double m_true_x{0.0};
    double m_true_y{0.0};
    double m_true_z{0.0};
    double m_recon_x{0.0};
    double m_recon_y{0.0};
    double m_recon_z{0.0};
    double m_residual_x{0.0};
    double m_residual_y{0.0};
    double m_residual_r{0.0};
    double m_edep{0.0};
    double m_time{0.0};
    uint64_t m_cell_id{0};
    int m_event_number{0};
    int m_detector_index{0};
};

} // namespace eicrecon
