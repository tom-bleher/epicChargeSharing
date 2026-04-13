// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// @file ChargeSharingMonitor.cc
/// @brief Implementation of charge sharing reconstruction monitoring.

#include "ChargeSharingMonitor.h"

#include <JANA/JApplication.h>
#include <services/log/Log_service.h>
#include <services/rootfile/RootFile_service.h>

#include <TDirectory.h>
#include <TFile.h>

#include <cmath>
#include <fmt/core.h>

namespace eicrecon {

ChargeSharingMonitor::ChargeSharingMonitor() {
    SetTypeName(NAME_OF_THIS);
}

void ChargeSharingMonitor::InitWithGlobalRootLock() {
    auto app = GetApplication();

    // Get input collection names from configuration
    app->SetDefaultParameter("ChargeSharingMonitor:inputRecoHits", m_input_reco_hits,
                             "Reconstructed TrackerHit collection name");
    app->SetDefaultParameter("ChargeSharingMonitor:inputSimHits", m_input_sim_hits,
                             "Input SimTrackerHit collection name");

    // Configure detector pairs to monitor
    // Each entry: {recoCollection, simCollection, displayName}
    // These must match the factory output collection names in ChargeSharingReconPlugin.cc
    m_detectors = {
        {"B0ChargeSharingTrackerHits", "B0TrackerHits", "B0Tracker"},
        {"LumiSpecTrackerChargeSharingHits", "LumiSpecTrackerHits", "LumiSpecTracker"},
    };

    // Note: Input collection names come from ePIC/ddsim simulation:
    // - B0TrackerHits: B0 forward tracker
    // - LumiSpecTrackerHits: Luminosity spectrometer tracker (AC-LGAD)

    // Get the shared ROOT file service
    auto rootfile_svc = app->GetService<RootFile_service>();
    auto rootfile = rootfile_svc->GetHistFile();

    // Create main directory for this plugin
    TDirectory* mainDir = rootfile->mkdir("ChargeSharingMonitor");
    mainDir->cd();

    // Create histograms for each detector
    for (const auto& det : m_detectors) {
        CreateHistograms(det.displayName);
    }

    // Create TTree for detailed per-hit data
    mainDir->cd();
    m_tree = new TTree("hits", "Charge Sharing Reconstruction Hits");

    // Branch setup - similar to main simulation output
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

void ChargeSharingMonitor::CreateHistograms(const std::string& detectorName) {
    auto app = GetApplication();
    auto rootfile_svc = app->GetService<RootFile_service>();
    auto rootfile = rootfile_svc->GetHistFile();

    // Create subdirectory for this detector
    TDirectory* detDir = rootfile->GetDirectory("ChargeSharingMonitor");
    TDirectory* subDir = detDir->mkdir(detectorName.c_str());
    subDir->cd();

    HistogramSet hists;

    // Residual histograms (in mm, typical AC-LGAD resolution ~10-50 um)
    hists.hResidualX = new TH1D(
        "hResidualX", fmt::format("{} X Residual;#Deltax = x_{{reco}} - x_{{true}} [mm];Counts", detectorName).c_str(),
        200, -0.5, 0.5);

    hists.hResidualY = new TH1D(
        "hResidualY", fmt::format("{} Y Residual;#Deltay = y_{{reco}} - y_{{true}} [mm];Counts", detectorName).c_str(),
        200, -0.5, 0.5);

    hists.hResidualR = new TH1D(
        "hResidualR", fmt::format("{} Radial Residual;#Deltar [mm];Counts", detectorName).c_str(), 100, 0.0, 0.5);

    // 2D correlation plots
    hists.hRecoVsTrueX =
        new TH2D("hRecoVsTrueX", fmt::format("{} Reco vs True X;x_{{true}} [mm];x_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);

    hists.hRecoVsTrueY =
        new TH2D("hRecoVsTrueY", fmt::format("{} Reco vs True Y;y_{{true}} [mm];y_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);

    hists.hTrueXY =
        new TH2D("hTrueXY", fmt::format("{} True Hit Positions;x_{{true}} [mm];y_{{true}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);

    hists.hRecoXY =
        new TH2D("hRecoXY", fmt::format("{} Reco Hit Positions;x_{{reco}} [mm];y_{{reco}} [mm]", detectorName).c_str(),
                 100, -20, 20, 100, -20, 20);

    hists.hResidualVsTrueX = new TH2D(
        "hResidualVsTrueX", fmt::format("{} X Residual vs True X;x_{{true}} [mm];#Deltax [mm]", detectorName).c_str(),
        100, -20, 20, 100, -0.5, 0.5);

    hists.hResidualVsTrueY = new TH2D(
        "hResidualVsTrueY", fmt::format("{} Y Residual vs True Y;y_{{true}} [mm];#Deltay [mm]", detectorName).c_str(),
        100, -20, 20, 100, -0.5, 0.5);

    // Energy deposit
    hists.hEnergyDeposit =
        new TH1D("hEnergyDeposit", fmt::format("{} Energy Deposit;E_{{dep}} [GeV];Counts", detectorName).c_str(), 100,
                 0.0, 0.001);

    m_histograms[detectorName] = hists;
}

void ChargeSharingMonitor::ProcessSequential(const std::shared_ptr<const JEvent>& event) {
    m_event_number = static_cast<int>(event->GetEventNumber());

    // Process each configured detector
    for (size_t detIdx = 0; detIdx < m_detectors.size(); ++detIdx) {
        const auto& det = m_detectors[detIdx];
        m_detector_index = static_cast<int>(detIdx);

        // Try to get the collections - may not exist for all detectors in all events
        const edm4eic::TrackerHitCollection* recoHits = nullptr;
        const edm4hep::SimTrackerHitCollection* simHits = nullptr;

        try {
            recoHits = event->GetCollection<edm4eic::TrackerHit>(det.recoCollection);
        } catch (...) {
            // Collection doesn't exist for this detector
            continue;
        }

        try {
            simHits = event->GetCollection<edm4hep::SimTrackerHit>(det.simCollection);
        } catch (...) {
            // No sim hits to compare against
            continue;
        }

        if (!recoHits || !simHits) {
            continue;
        }

        // Match reco hits to sim hits by cellID
        // Build map of cellID -> simHit for efficient lookup
        std::map<uint64_t, const edm4hep::SimTrackerHit*> simHitMap;
        for (const auto& simHit : *simHits) {
            simHitMap[simHit.getCellID()] = &simHit;
        }

        // Process each reco hit
        for (const auto& recoHit : *recoHits) {
            auto it = simHitMap.find(recoHit.getCellID());
            if (it != simHitMap.end()) {
                FillData(det.displayName, recoHit, *(it->second));
            }
        }
    }
}

void ChargeSharingMonitor::FillData(const std::string& detectorName, const edm4eic::TrackerHit& recoHit,
                                    const edm4hep::SimTrackerHit& simHit) {
    // Extract positions
    const auto& recoPos = recoHit.getPosition();
    const auto& simPos = simHit.getPosition();

    m_true_x = simPos.x;
    m_true_y = simPos.y;
    m_true_z = simPos.z;
    m_recon_x = recoPos.x;
    m_recon_y = recoPos.y;
    m_recon_z = recoPos.z;

    // Calculate residuals
    m_residual_x = m_recon_x - m_true_x;
    m_residual_y = m_recon_y - m_true_y;
    m_residual_r = std::hypot(m_residual_x, m_residual_y);

    // Other quantities
    m_edep = recoHit.getEdep();
    m_time = recoHit.getTime();
    m_cell_id = recoHit.getCellID();

    // Fill histograms
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

    // Fill tree
    m_tree->Fill();
}

void ChargeSharingMonitor::FinishWithGlobalRootLock() {
    // Calculate and print summary statistics
    auto app = GetApplication();
    auto log = app->GetService<Log_service>()->logger("ChargeSharingMonitor");

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
