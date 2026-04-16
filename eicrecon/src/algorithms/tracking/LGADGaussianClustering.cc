// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include "LGADGaussianClustering.h"

#include "chargesharing/fit/GaussianFit.hh"

#include <DD4hep/Detector.h>
#include <DDRec/CellIDPositionConverter.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <DDSegmentation/MultiSegmentation.h>

#include <algorithms/geo.h>
#include <algorithms/interfaces/ActsSvc.h>

#include <Acts/Surfaces/Surface.hpp>

#include <edm4hep/utils/vector_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace eicrecon {

namespace cfit = ::chargesharing::fit;

// ============================================================================
// Union-Find
// ============================================================================

LGADGaussianClustering::UnionFind::UnionFind(int n) : m_parent(n), m_rank(n, 0) {
    std::iota(m_parent.begin(), m_parent.end(), 0);
}

int LGADGaussianClustering::UnionFind::find(int id) {
    if (m_parent[id] == id)
        return id;
    return m_parent[id] = find(m_parent[id]);
}

void LGADGaussianClustering::UnionFind::merge(int id1, int id2) {
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
// MultiSegmentation resolution (matches LGADHitClustering / SiliconChargeSharing)
// ============================================================================

const dd4hep::DDSegmentation::CartesianGridXY*
LGADGaussianClustering::getLocalSegmentation(const dd4hep::rec::CellID& cellID) const {
    auto segmentation_type = m_seg.type();
    const dd4hep::DDSegmentation::Segmentation* segmentation = m_seg.segmentation();
    while (segmentation_type == "MultiSegmentation") {
        const auto* multi =
            dynamic_cast<const dd4hep::DDSegmentation::MultiSegmentation*>(segmentation);
        segmentation = &multi->subsegmentation(cellID);
        segmentation_type = segmentation->type();
    }
    const auto* grid = dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation);
    if (!grid) {
        throw std::runtime_error(
            "LGADGaussianClustering: segmentation is not CartesianGridXY");
    }
    return grid;
}

// ============================================================================
// init()
// ============================================================================

void LGADGaussianClustering::init() {
    m_converter = algorithms::GeoSvc::instance().cellIDPositionConverter();
    m_detector = algorithms::GeoSvc::instance().detector();
    m_seg = m_detector->readout(m_cfg.readout).segmentation();
    m_decoder = m_seg.decoder();
    m_acts_context = algorithms::ActsSvc::instance().acts_geometry_provider();

    if (m_cfg.reconMethod < 0 || m_cfg.reconMethod > 2) {
        warning("Invalid reconMethod={}, clamping to 2 (Gaussian 2D)", m_cfg.reconMethod);
        m_cfg.reconMethod = 2;
    }

    info("LGADGaussianClustering initialized: readout={}, deltaT={} ns, reconMethod={}",
         m_cfg.readout, m_cfg.deltaT, m_cfg.reconMethod);
}

// ============================================================================
// process()
// ============================================================================

void LGADGaussianClustering::process(const Input& input, const Output& output) const {
    const auto [hits] = input;
    auto [clusters] = output;

    if (!hits || !clusters)
        return;

    const int nHits = static_cast<int>(hits->size());
    if (nHits == 0)
        return;

    std::unordered_map<dd4hep::rec::CellID, std::vector<int>> hitsByCellID;
    for (int i = 0; i < nHits; ++i) {
        hitsByCellID[(*hits)[i].getCellID()].push_back(i);
    }

    UnionFind uf(nHits);

    for (const auto& [cellID, hitIndices] : hitsByCellID) {
        const auto* seg = getLocalSegmentation(cellID);
        std::set<dd4hep::rec::CellID> neighborCells;
        seg->neighbours(cellID, neighborCells);

        for (const auto& neighborID : neighborCells) {
            auto it = hitsByCellID.find(neighborID);
            if (it == hitsByCellID.end())
                continue;
            for (int i : hitIndices) {
                for (int j : it->second) {
                    if (std::abs((*hits)[i].getTime() - (*hits)[j].getTime()) < m_cfg.deltaT) {
                        uf.merge(i, j);
                    }
                }
            }
        }
    }

    std::unordered_map<int, std::vector<edm4eic::TrackerHit>> clusterMap;
    for (int i = 0; i < nHits; ++i) {
        clusterMap[uf.find(i)].push_back((*hits)[i]);
    }

    for (const auto& [root, clusterHits] : clusterMap) {
        reconstructCluster(output, clusterHits);
    }
}

// ============================================================================
// Cluster position reconstruction
// ============================================================================

void LGADGaussianClustering::reconstructCluster(const Output& output,
                                                const std::vector<edm4eic::TrackerHit>& hits) const {
    auto [clusters] = output;

    if (hits.empty())
        return;

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

    const auto* seg = getLocalSegmentation(maxCellID);
    const double pitchX = seg->gridSizeX() / dd4hep::mm;
    const double pitchY = seg->gridSizeY() / dd4hep::mm;

    const auto centerPos = m_seg.position(maxCellID);
    const double centerX = centerPos.x() / dd4hep::mm;
    const double centerY = centerPos.y() / dd4hep::mm;

    const auto clusterPos = reconstructClusterPosition(m_cfg.reconMethod, xPos, yPos, charges,
                                                       centerX, centerY, maxEdep, pitchX, pitchY,
                                                       m_cfg.fitErrorPercent);
    const double reconX = clusterPos.reconX;
    const double reconY = clusterPos.reconY;
    const double sigma2X = clusterPos.sigma2X;
    const double sigma2Y = clusterPos.sigma2Y;

    const auto* context = m_converter->findContext(maxCellID);
    if (!context) {
        error("No DetElement context for cellID {:#018x}", maxCellID);
        return;
    }
    auto volID = context->identifier;
    const auto& surfaceMap = m_acts_context->surfaceMap();
    const auto is = surfaceMap.find(volID);
    if (is == surfaceMap.end()) {
        error("No Acts surface found for volume ID {:#018x}", volID);
        return;
    }
    const Acts::Surface* surface = is->second;

    auto cluster = clusters->create();
    cluster.setSurface(surface->geometryId().value());

    const auto refPos = m_seg.position(maxCellID);
    const Acts::Vector3 globalPos(reconX * dd4hep::mm, reconY * dd4hep::mm, refPos.z());
    auto locResult = surface->globalToLocal(Acts::GeometryContext{}, globalPos, Acts::Vector3::Zero());
    if (locResult.ok()) {
        cluster.setLoc({static_cast<float>(locResult.value()[0] / dd4hep::mm),
                        static_cast<float>(locResult.value()[1] / dd4hep::mm)});
    } else {
        warning("globalToLocal failed for cellID {:#018x}; using global coordinates", maxCellID);
        cluster.setLoc({static_cast<float>(reconX), static_cast<float>(reconY)});
    }
    cluster.setTime(static_cast<float>(earliestTime));

    const float timeErr = static_cast<float>(m_cfg.deltaT / std::sqrt(12.0));
    cluster.setCovariance({static_cast<float>(sigma2X), static_cast<float>(sigma2Y), timeErr * timeErr, 0.0f});

    for (const auto& hit : hits) {
        cluster.addToHits(hit);
        cluster.addToWeights(static_cast<float>(hit.getEdep() / totalEdep));
    }
}

// ============================================================================
// Pure-math cluster position reconstruction (unit-testable)
// ============================================================================

LGADGaussianClustering::ClusterPosition LGADGaussianClustering::reconstructClusterPosition(
    int method, const std::vector<double>& xPos, const std::vector<double>& yPos,
    const std::vector<double>& charges, double centerX, double centerY, double maxEdep, double pitchX,
    double pitchY, double fitErrorPercent) {
    ClusterPosition out{};

    double totalEdep = 0.0;
    for (double q : charges)
        totalEdep += q;

    if (totalEdep <= 0.0 || charges.empty()) {
        out.reconX = centerX;
        out.reconY = centerY;
        out.sigma2X = pitchX * pitchX / 12.0;
        out.sigma2Y = pitchY * pitchY / 12.0;
        return out;
    }

    const double padSizeX = pitchX;
    const double padSizeY = pitchY;

    auto centroid = [&](double& rx, double& ry) {
        rx = 0.0;
        ry = 0.0;
        for (std::size_t i = 0; i < charges.size(); ++i) {
            rx += charges[i] * xPos[i];
            ry += charges[i] * yPos[i];
        }
        rx /= totalEdep;
        ry /= totalEdep;
    };

    if (method == 0 || charges.size() < 3) {
        centroid(out.reconX, out.reconY);
        out.sigma2X = pitchX * pitchX / 12.0;
        out.sigma2Y = pitchY * pitchY / 12.0;
        return out;
    }

    if (method == 1) {
        std::vector<double> rowX, rowQ, colY, colQ;
        const double tolY = pitchY * 0.4;
        const double tolX = pitchX * 0.4;

        for (std::size_t i = 0; i < charges.size(); ++i) {
            if (std::abs(yPos[i] - centerY) < tolY) {
                rowX.push_back(xPos[i]);
                rowQ.push_back(charges[i]);
            }
            if (std::abs(xPos[i] - centerX) < tolX) {
                colY.push_back(yPos[i]);
                colQ.push_back(charges[i]);
            }
        }

        bool fitXOk = false;
        if (rowX.size() >= 3) {
            cfit::GaussFit1DConfig cfg;
            cfg.muLo = centerX - pitchX;
            cfg.muHi = centerX + pitchX;
            cfg.sigmaLo = padSizeX;
            cfg.sigmaHi = pitchX * 3;
            cfg.qMax = maxEdep;
            cfg.pixelSpacing = pitchX;
            cfg.errorPercent = fitErrorPercent;
            auto result = cfit::fitGaussian1D(rowX, rowQ, cfg);
            if (result.converged) {
                out.reconX = result.mu;
                out.sigma2X = result.muError * result.muError;
                fitXOk = true;
            }
        }
        if (!fitXOk) {
            double sumWX = 0, sumW = 0;
            for (std::size_t i = 0; i < rowX.size(); ++i) {
                sumWX += rowQ[i] * rowX[i];
                sumW += rowQ[i];
            }
            out.reconX = sumW > 0 ? sumWX / sumW : centerX;
            out.sigma2X = pitchX * pitchX / 12.0;
        }

        bool fitYOk = false;
        if (colY.size() >= 3) {
            cfit::GaussFit1DConfig cfg;
            cfg.muLo = centerY - pitchY;
            cfg.muHi = centerY + pitchY;
            cfg.sigmaLo = padSizeY;
            cfg.sigmaHi = pitchY * 3;
            cfg.qMax = maxEdep;
            cfg.pixelSpacing = pitchY;
            cfg.errorPercent = fitErrorPercent;
            auto result = cfit::fitGaussian1D(colY, colQ, cfg);
            if (result.converged) {
                out.reconY = result.mu;
                out.sigma2Y = result.muError * result.muError;
                fitYOk = true;
            }
        }
        if (!fitYOk) {
            double sumWY = 0, sumW = 0;
            for (std::size_t i = 0; i < colY.size(); ++i) {
                sumWY += colQ[i] * colY[i];
                sumW += colQ[i];
            }
            out.reconY = sumW > 0 ? sumWY / sumW : centerY;
            out.sigma2Y = pitchY * pitchY / 12.0;
        }

        out.fitConverged = fitXOk && fitYOk;
        return out;
    }

    // method >= 2 : Gaussian 2D
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
    cfg.errorPercent = fitErrorPercent;

    auto result = cfit::fitGaussian2D(xPos, yPos, charges, cfg);
    if (result.converged) {
        out.reconX = result.muX;
        out.reconY = result.muY;
        out.sigma2X = result.muXError * result.muXError;
        out.sigma2Y = result.muYError * result.muYError;
        out.fitConverged = true;
    } else {
        centroid(out.reconX, out.reconY);
        out.sigma2X = pitchX * pitchX / 12.0;
        out.sigma2Y = pitchY * pitchY / 12.0;
    }
    return out;
}

} // namespace eicrecon
