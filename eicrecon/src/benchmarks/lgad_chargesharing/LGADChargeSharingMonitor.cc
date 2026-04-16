// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include "LGADChargeSharingMonitor.h"

#include <JANA/JApplication.h>
#include <services/log/Log_service.h>
#include <services/rootfile/RootFile_service.h>

#include <TDirectory.h>
#include <TFile.h>

#include <cmath>
#include <fmt/core.h>

namespace eicrecon {

namespace {
constexpr const char* kMonitorDirName = "LGADChargeSharing";
}

LGADChargeSharingMonitor::LGADChargeSharingMonitor() {
    SetTypeName(NAME_OF_THIS);
}

void LGADChargeSharingMonitor::Init() {
    auto app = GetApplication();

    // Detector collection pairs to monitor. Collection names must agree with
    // the factory outputs registered under src/detectors/*.
    m_detectors = {
        {"B0TrackerChargeSharingHits", "B0TrackerHits", "B0Tracker"},
        {"LumiSpecTrackerChargeSharingHits", "LumiSpecTrackerHits", "LumiSpecTracker"},
    };

    // Shared ROOT output: use RootFile_service so histograms land in the same
    // file as every other EICrecon benchmark (activated with -Phistsfile=...).
    auto rootfile_svc = app->GetService<RootFile_service>();
    auto* rootfile = rootfile_svc->GetHistFile();

    TDirectory* mainDir = rootfile->mkdir(kMonitorDirName);
    if (mainDir == nullptr) {
        mainDir = rootfile->GetDirectory(kMonitorDirName);
    }
    mainDir->cd();

    for (const auto& det : m_detectors) {
        createHistograms(det.displayName);
    }

    mainDir->cd();
    m_tree = new TTree("hits", "LGAD charge sharing reconstruction hits");

    m_tree->Branch("trueX", &m_true_x, "trueX/D");
    m_tree->Branch("trueY", &m_true_y, "trueY/D");
    m_tree->Branch("trueZ", &m_true_z, "trueZ/D");
    m_tree->Branch("reconX", &m_recon_x, "reconX/D");
    m_tree->Branch("reconY", &m_recon_y, "reconY/D");
    m_tree->Branch("reconZ", &m_recon_z, "reconZ/D");
    m_tree->Branch("residualX", &m_residual_x, "residualX/D");
    m_tree->Branch("residualY", &m_residual_y, "residualY/D");
    m_tree->Branch("residualR", &m_residual_r, "residualR/D");
    m_tree->Branch("edep", &m_edep, "edep/D");
    m_tree->Branch("time", &m_time, "time/D");
    m_tree->Branch("cellID", &m_cell_id, "cellID/l");
    m_tree->Branch("eventNumber", &m_event_number, "eventNumber/I");
    m_tree->Branch("detectorIndex", &m_detector_index, "detectorIndex/I");
}

void LGADChargeSharingMonitor::createHistograms(const std::string& detectorName) {
    auto app = GetApplication();
    auto rootfile_svc = app->GetService<RootFile_service>();
    auto* rootfile = rootfile_svc->GetHistFile();

    TDirectory* detDir = rootfile->GetDirectory(kMonitorDirName);
    TDirectory* subDir = detDir->mkdir(detectorName.c_str());
    subDir->cd();

    HistogramSet hists;

    hists.hResidualX =
        new TH1D("hResidualX",
                 fmt::format("{} X Residual;#Deltax = x_{{reco}} - x_{{true}} [mm];Counts", detectorName).c_str(),
                 200, -0.5, 0.5);
    hists.hResidualY =
        new TH1D("hResidualY",
                 fmt::format("{} Y Residual;#Deltay = y_{{reco}} - y_{{true}} [mm];Counts", detectorName).c_str(),
                 200, -0.5, 0.5);
    hists.hResidualR =
        new TH1D("hResidualR", fmt::format("{} Radial Residual;#Deltar [mm];Counts", detectorName).c_str(),
                 100, 0.0, 0.5);

    hists.hRecoVsTrueX =
        new TH2D("hRecoVsTrueX",
                 fmt::format("{} Reco vs True X;x_{{true}} [mm];x_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);
    hists.hRecoVsTrueY =
        new TH2D("hRecoVsTrueY",
                 fmt::format("{} Reco vs True Y;y_{{true}} [mm];y_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);
    hists.hTrueXY =
        new TH2D("hTrueXY",
                 fmt::format("{} True Hit Positions;x_{{true}} [mm];y_{{true}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);
    hists.hRecoXY =
        new TH2D("hRecoXY",
                 fmt::format("{} Reco Hit Positions;x_{{reco}} [mm];y_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);
    hists.hResidualVsTrueX =
        new TH2D("hResidualVsTrueX",
                 fmt::format("{} X Residual vs True X;x_{{true}} [mm];#Deltax [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -0.5, 0.5);
    hists.hResidualVsTrueY =
        new TH2D("hResidualVsTrueY",
                 fmt::format("{} Y Residual vs True Y;y_{{true}} [mm];#Deltay [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -0.5, 0.5);

    hists.hEnergyDeposit =
        new TH1D("hEnergyDeposit",
                 fmt::format("{} Energy Deposit;E_{{dep}} [GeV];Counts", detectorName).c_str(), 100, 0.0, 0.001);

    m_histograms[detectorName] = hists;
}

void LGADChargeSharingMonitor::ProcessSequential(const JEvent& event) {
    m_event_number = static_cast<int>(event.GetEventNumber());

    for (std::size_t detIdx = 0; detIdx < m_detectors.size(); ++detIdx) {
        const auto& det = m_detectors[detIdx];
        m_detector_index = static_cast<int>(detIdx);

        const edm4eic::TrackerHitCollection* recoHits = nullptr;
        const edm4hep::SimTrackerHitCollection* simHits = nullptr;

        try {
            recoHits = event.GetCollection<edm4eic::TrackerHit>(det.recoCollection);
        } catch (...) {
            continue;
        }
        try {
            simHits = event.GetCollection<edm4hep::SimTrackerHit>(det.simCollection);
        } catch (...) {
            continue;
        }
        if (recoHits == nullptr || simHits == nullptr)
            continue;

        std::map<uint64_t, const edm4hep::SimTrackerHit*> simHitMap;
        for (const auto& simHit : *simHits) {
            simHitMap[simHit.getCellID()] = &simHit;
        }

        for (const auto& recoHit : *recoHits) {
            auto it = simHitMap.find(recoHit.getCellID());
            if (it != simHitMap.end()) {
                fillData(det.displayName, recoHit, *(it->second));
            }
        }
    }
}

void LGADChargeSharingMonitor::fillData(const std::string& detectorName,
                                        const edm4eic::TrackerHit& recoHit,
                                        const edm4hep::SimTrackerHit& simHit) {
    const auto& recoPos = recoHit.getPosition();
    const auto& simPos = simHit.getPosition();

    m_true_x = simPos.x;
    m_true_y = simPos.y;
    m_true_z = simPos.z;
    m_recon_x = recoPos.x;
    m_recon_y = recoPos.y;
    m_recon_z = recoPos.z;

    m_residual_x = m_recon_x - m_true_x;
    m_residual_y = m_recon_y - m_true_y;
    m_residual_r = std::hypot(m_residual_x, m_residual_y);

    m_edep = recoHit.getEdep();
    m_time = recoHit.getTime();
    m_cell_id = recoHit.getCellID();

    auto& hists = m_histograms[detectorName];
    hists.hResidualX->Fill(m_residual_x);
    hists.hResidualY->Fill(m_residual_y);
    hists.hResidualR->Fill(m_residual_r);
    hists.hRecoVsTrueX->Fill(m_true_x, m_recon_x);
    hists.hRecoVsTrueY->Fill(m_true_y, m_recon_y);
    hists.hTrueXY->Fill(m_true_x, m_true_y);
    hists.hRecoXY->Fill(m_recon_x, m_recon_y);
    hists.hResidualVsTrueX->Fill(m_true_x, m_residual_x);
    hists.hResidualVsTrueY->Fill(m_true_y, m_residual_y);
    hists.hEnergyDeposit->Fill(m_edep);

    m_tree->Fill();
}

void LGADChargeSharingMonitor::Finish() {
    auto app = GetApplication();
    auto log = app->GetService<Log_service>()->logger("LGADChargeSharingMonitor");

    for (const auto& [detName, hists] : m_histograms) {
        if (hists.hResidualX->GetEntries() > 0) {
            log->info("=== {} Summary ===", detName);
            log->info("  Entries: {}", static_cast<int>(hists.hResidualX->GetEntries()));
            log->info("  X Residual: mean = {:.4f} mm, RMS = {:.4f} mm", hists.hResidualX->GetMean(),
                      hists.hResidualX->GetRMS());
            log->info("  Y Residual: mean = {:.4f} mm, RMS = {:.4f} mm", hists.hResidualY->GetMean(),
                      hists.hResidualY->GetRMS());
            log->info("  R Residual: mean = {:.4f} mm", hists.hResidualR->GetMean());
        }
    }
    log->info("TTree 'hits' contains {} entries", m_tree->GetEntries());
}

} // namespace eicrecon
