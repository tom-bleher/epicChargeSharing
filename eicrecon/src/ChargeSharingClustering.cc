// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// @file ChargeSharingClustering.cc
/// @brief Improved LGAD hit clustering with Gaussian position reconstruction.
///
/// Union-find clustering logic follows LGADHitClustering (Chun Yuen Tsang, 2025).
/// Position reconstruction uses Gaussian fitting from core/GaussianFit.hh instead
/// of the max-ADC or centroid methods in the upstream implementation.

#include "ChargeSharingClustering.h"

#include "GaussianFit.hh"

#include <DD4hep/Detector.h>
#include <DDRec/CellIDPositionConverter.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <DDSegmentation/MultiSegmentation.h>

#include <spdlog/spdlog.h>

#include <algorithms/geo.h>
#include <algorithms/interfaces/ActsSvc.h>

#include <Acts/Surfaces/Surface.hpp>

#include <edm4hep/utils/vector_utils.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_map>

namespace eicrecon {

namespace cfit = epic::chargesharing::fit;

// ============================================================================
// Union-Find
// ============================================================================

ChargeSharingClustering::UnionFind::UnionFind(int n) : m_parent(n), m_rank(n, 0) {
    std::iota(m_parent.begin(), m_parent.end(), 0);
}

int ChargeSharingClustering::UnionFind::find(int id) {
    if (m_parent[id] == id)
        return id;
    return m_parent[id] = find(m_parent[id]); // path compression
}

void ChargeSharingClustering::UnionFind::merge(int id1, int id2) {
    int root1 = find(id1);
    int root2 = find(id2);
    if (root1 != root2) {
        if (m_rank[root1] > m_rank[root2])
            m_parent[root2] = root1;
        else if (m_rank[root1] < m_rank[root2])
            m_parent[root1] = root2;
        else {
            m_parent[root1] = root2;
            m_rank[root2]++;
        }
    }
}

// ============================================================================
// MultiSegmentation resolution (same as LGADHitClustering / SiliconChargeSharing)
// ============================================================================

const dd4hep::DDSegmentation::CartesianGridXY*
ChargeSharingClustering::getLocalSegmentation(const dd4hep::rec::CellID& cellID) const {
    auto segmentation_type = m_seg.type();
    const dd4hep::DDSegmentation::Segmentation* segmentation = m_seg.segmentation();
    while (segmentation_type == "MultiSegmentation") {
        const auto* multi =
            dynamic_cast<const dd4hep::DDSegmentation::MultiSegmentation*>(segmentation);
        segmentation = &multi->subsegmentation(cellID);
        segmentation_type = segmentation->type();
    }
    const auto* grid =
        dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation);
    if (!grid) {
        throw std::runtime_error(
            "ChargeSharingClustering: segmentation is not CartesianGridXY");
    }
    return grid;
}

// ============================================================================
// Initialization
// ============================================================================

void ChargeSharingClustering::init() {
    m_log = spdlog::default_logger()->clone(std::string(name()));

    m_converter = algorithms::GeoSvc::instance().cellIDPositionConverter();
    m_detector = algorithms::GeoSvc::instance().detector();
    m_seg = m_detector->readout(m_cfg.readout).segmentation();
    m_decoder = m_seg.decoder();
    m_acts_context = algorithms::ActsSvc::instance().acts_geometry_provider();

    if (m_cfg.reconMethod < 0 || m_cfg.reconMethod > 2) {
        m_log->warn("Invalid reconMethod={}, clamping to 2 (Gaussian 2D)", m_cfg.reconMethod);
        m_cfg.reconMethod = 2;
    }

    m_log->info("ChargeSharingClustering initialized: readout={}, deltaT={} ns, reconMethod={}",
                m_cfg.readout, m_cfg.deltaT, m_cfg.reconMethod);
}

// ============================================================================
// Main processing: cluster hits then reconstruct positions
// ============================================================================

void ChargeSharingClustering::process(const Input& input, const Output& output) const {
    const auto [hits] = input;
    auto [clusters] = output;

    if (!hits || !clusters)
        return;

    const int nHits = static_cast<int>(hits->size());
    if (nHits == 0)
        return;

    // --- Step 1: Index hits by cellID ---
    std::unordered_map<dd4hep::rec::CellID, std::vector<int>> hitsByCellID;
    for (int i = 0; i < nHits; ++i) {
        hitsByCellID[(*hits)[i].getCellID()].push_back(i);
    }

    // --- Step 2: Union-Find merge neighbors within time gate ---
    UnionFind uf(nHits);

    for (const auto& [cellID, hitIndices] : hitsByCellID) {
        // Find neighbors via DD4hep segmentation
        const auto* seg = getLocalSegmentation(cellID);
        std::set<dd4hep::rec::CellID> neighborCells;
        seg->neighbours(cellID, neighborCells);

        for (const auto& neighborID : neighborCells) {
            auto it = hitsByCellID.find(neighborID);
            if (it == hitsByCellID.end())
                continue;

            // Merge hit pairs if within time window
            for (int i : hitIndices) {
                for (int j : it->second) {
                    if (std::abs((*hits)[i].getTime() - (*hits)[j].getTime()) < m_cfg.deltaT) {
                        uf.merge(i, j);
                    }
                }
            }
        }
    }

    // --- Step 3: Group hits by cluster root ---
    std::unordered_map<int, std::vector<edm4eic::TrackerHit>> clusterMap;
    for (int i = 0; i < nHits; ++i) {
        clusterMap[uf.find(i)].push_back((*hits)[i]);
    }

    // --- Step 4: Reconstruct each cluster ---
    for (const auto& [root, clusterHits] : clusterMap) {
        reconstructCluster(output, clusterHits);
    }
}

// ============================================================================
// Cluster position reconstruction
// ============================================================================

void ChargeSharingClustering::reconstructCluster(
    const Output& output,
    const std::vector<edm4eic::TrackerHit>& hits) const {

    auto [clusters] = output;

    if (hits.empty())
        return;

    // Collect pixel positions and charges
    std::vector<double> xPos, yPos, charges;
    xPos.reserve(hits.size());
    yPos.reserve(hits.size());
    charges.reserve(hits.size());

    double maxEdep = 0.0;
    double totalEdep = 0.0;
    dd4hep::rec::CellID maxCellID = 0;
    double earliestTime = std::numeric_limits<double>::max();

    for (const auto& hit : hits) {
        const auto cellID = hit.getCellID();
        const auto pos = m_seg.position(cellID);
        const double edep = hit.getEdep();

        // DD4hep positions are in internal units — convert to mm
        xPos.push_back(pos.x() / dd4hep::mm);
        yPos.push_back(pos.y() / dd4hep::mm);
        charges.push_back(edep);
        totalEdep += edep;

        if (edep > maxEdep) {
            maxEdep = edep;
            maxCellID = cellID;
        }
        if (hit.getTime() < earliestTime) {
            earliestTime = hit.getTime();
        }
    }

    if (totalEdep <= 0.0 || charges.empty())
        return;

    // Get pixel pitch from segmentation
    const auto* seg = getLocalSegmentation(maxCellID);
    const double pitchX = seg->gridSizeX() / dd4hep::mm;
    const double pitchY = seg->gridSizeY() / dd4hep::mm;
    const double padSizeX = pitchX; // AC-LGAD: pad = pitch (no inter-pad gap in DD4hep)
    const double padSizeY = pitchY;

    // --- Reconstruct position ---
    double reconX = 0.0, reconY = 0.0;
    double sigma2X = 0.0, sigma2Y = 0.0;

    const int method = m_cfg.reconMethod;

    if (method == 0 || charges.size() < 3) {
        // Centroid: charge-weighted average (same as LGADHitClustering useAve=true)
        for (size_t i = 0; i < charges.size(); ++i) {
            reconX += charges[i] * xPos[i];
            reconY += charges[i] * yPos[i];
        }
        reconX /= totalEdep;
        reconY /= totalEdep;
        // Uncertainty: pitch/sqrt(12) (uniform distribution)
        sigma2X = pitchX * pitchX / 12.0;
        sigma2Y = pitchY * pitchY / 12.0;
    } else if (method == 1) {
        // Gaussian 1D: fit row and column slices independently
        // Find center pixel position
        const auto centerPos = m_seg.position(maxCellID);
        const double centerX = centerPos.x() / dd4hep::mm;
        const double centerY = centerPos.y() / dd4hep::mm;

        // Extract row (same Y as center) and column (same X as center)
        std::vector<double> rowX, rowQ, colY, colQ;
        const double tol = pitchY * 0.4; // tolerance for same-row/col
        const double tolX = pitchX * 0.4;

        for (size_t i = 0; i < charges.size(); ++i) {
            if (std::abs(yPos[i] - centerY) < tol) {
                rowX.push_back(xPos[i]);
                rowQ.push_back(charges[i]);
            }
            if (std::abs(xPos[i] - centerX) < tolX) {
                colY.push_back(yPos[i]);
                colQ.push_back(charges[i]);
            }
        }

        // Fit row (X position)
        bool fitXOk = false;
        if (rowX.size() >= 3) {
            cfit::GaussFit1DConfig cfg;
            cfg.muLo = centerX - pitchX;
            cfg.muHi = centerX + pitchX;
            cfg.sigmaLo = padSizeX;
            cfg.sigmaHi = pitchX * 3;
            cfg.qMax = maxEdep;
            cfg.pixelSpacing = pitchX;
            cfg.errorPercent = m_cfg.fitErrorPercent;
            auto result = cfit::fitGaussian1D(rowX, rowQ, cfg);
            if (result.converged) {
                reconX = result.mu;
                sigma2X = result.muError * result.muError;
                fitXOk = true;
            }
        }
        if (!fitXOk) {
            // Centroid fallback for X
            double sumWX = 0, sumW = 0;
            for (size_t i = 0; i < rowX.size(); ++i) {
                sumWX += rowQ[i] * rowX[i];
                sumW += rowQ[i];
            }
            reconX = sumW > 0 ? sumWX / sumW : centerX;
            sigma2X = pitchX * pitchX / 12.0;
        }

        // Fit column (Y position)
        bool fitYOk = false;
        if (colY.size() >= 3) {
            cfit::GaussFit1DConfig cfg;
            cfg.muLo = centerY - pitchY;
            cfg.muHi = centerY + pitchY;
            cfg.sigmaLo = padSizeY;
            cfg.sigmaHi = pitchY * 3;
            cfg.qMax = maxEdep;
            cfg.pixelSpacing = pitchY;
            cfg.errorPercent = m_cfg.fitErrorPercent;
            auto result = cfit::fitGaussian1D(colY, colQ, cfg);
            if (result.converged) {
                reconY = result.mu;
                sigma2Y = result.muError * result.muError;
                fitYOk = true;
            }
        }
        if (!fitYOk) {
            double sumWY = 0, sumW = 0;
            for (size_t i = 0; i < colY.size(); ++i) {
                sumWY += colQ[i] * colY[i];
                sumW += colQ[i];
            }
            reconY = sumW > 0 ? sumWY / sumW : centerY;
            sigma2Y = pitchY * pitchY / 12.0;
        }

    } else {
        // Gaussian 2D: fit full cluster
        const auto centerPos = m_seg.position(maxCellID);
        const double centerX = centerPos.x() / dd4hep::mm;
        const double centerY = centerPos.y() / dd4hep::mm;
        const double spacing = std::min(pitchX, pitchY);

        cfit::GaussFit2DConfig cfg;
        cfg.muXLo = centerX - pitchX;
        cfg.muXHi = centerX + pitchX;
        cfg.muYLo = centerY - pitchY;
        cfg.muYHi = centerY + pitchY;
        cfg.sigmaLo = std::min(padSizeX, padSizeY);
        cfg.sigmaHi = std::max(pitchX, pitchY) * 3;
        cfg.qMax = maxEdep;
        cfg.pixelSpacing = spacing;
        cfg.errorPercent = m_cfg.fitErrorPercent;

        auto result = cfit::fitGaussian2D(xPos, yPos, charges, cfg);
        if (result.converged) {
            reconX = result.muX;
            reconY = result.muY;
            sigma2X = result.muXError * result.muXError;
            sigma2Y = result.muYError * result.muYError;
        } else {
            // Centroid fallback
            for (size_t i = 0; i < charges.size(); ++i) {
                reconX += charges[i] * xPos[i];
                reconY += charges[i] * yPos[i];
            }
            reconX /= totalEdep;
            reconY /= totalEdep;
            sigma2X = pitchX * pitchX / 12.0;
            sigma2Y = pitchY * pitchY / 12.0;
        }
    }

    // --- Look up Acts surface ---
    const auto* context = m_converter->findContext(maxCellID);
    if (!context) {
        m_log->error("No DetElement context for cellID {:#018x}", maxCellID);
        return;
    }
    auto volID = context->identifier;
    const auto& surfaceMap = m_acts_context->surfaceMap();
    const auto is = surfaceMap.find(volID);
    if (is == surfaceMap.end()) {
        m_log->error("No Acts surface found for volume ID {:#018x}", volID);
        return;
    }
    const Acts::Surface* surface = is->second;

    // --- Build Measurement2D ---
    auto cluster = clusters->create();
    cluster.setSurface(surface->geometryId().value());

    // Transform reconstructed global position to Acts surface-local coordinates.
    // reconX/reconY are in mm (divided by dd4hep::mm); convert back for the 3D point.
    const auto refPos = m_seg.position(maxCellID);
    const Acts::Vector3 globalPos(reconX * dd4hep::mm, reconY * dd4hep::mm, refPos.z());
    auto locResult = surface->globalToLocal(Acts::GeometryContext{}, globalPos, Acts::Vector3::Zero());
    if (locResult.ok()) {
        cluster.setLoc({static_cast<float>(locResult.value()[0] / dd4hep::mm),
                        static_cast<float>(locResult.value()[1] / dd4hep::mm)});
    } else {
        m_log->warn("globalToLocal failed for cellID {:#018x}; using global coordinates", maxCellID);
        cluster.setLoc({static_cast<float>(reconX), static_cast<float>(reconY)});
    }
    cluster.setTime(static_cast<float>(earliestTime));

    // Covariance: {xx, yy, tt, xy}
    const float timeErr = static_cast<float>(m_cfg.deltaT / std::sqrt(12.0));
    cluster.setCovariance({
        static_cast<float>(sigma2X),
        static_cast<float>(sigma2Y),
        timeErr * timeErr,
        0.0f // no xy cross-term
    });

    // Back-references to input hits
    for (const auto& hit : hits) {
        cluster.addToHits(hit);
        cluster.addToWeights(static_cast<float>(hit.getEdep() / totalEdep));
    }
}

} // namespace eicrecon
