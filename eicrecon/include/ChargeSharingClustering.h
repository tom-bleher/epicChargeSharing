// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// @file ChargeSharingClustering.h
/// @brief Improved LGAD hit clustering with Gaussian position reconstruction.
///
/// Drop-in replacement for LGADHitClustering that uses Gaussian fitting
/// (from GaussianFit.hh) instead of max-ADC or simple centroid for
/// position extraction. Same union-find clustering, better reconstruction.

#pragma once

#include "ChargeSharingClusteringConfig.h"

#include <DD4hep/Detector.h>
#include <DD4hep/Segmentations.h>
#include <DDRec/CellIDPositionConverter.h>
#include <DDSegmentation/BitFieldCoder.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <algorithms/algorithm.h>
#include <edm4eic/Measurement2DCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <spdlog/logger.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "algorithms/interfaces/WithPodConfig.h"
#include "algorithms/tracking/ActsGeometryProvider.h"

namespace eicrecon {

using ChargeSharingClusteringAlgorithm =
    algorithms::Algorithm<algorithms::Input<edm4eic::TrackerHitCollection>,
                          algorithms::Output<edm4eic::Measurement2DCollection>>;

class ChargeSharingClustering : public ChargeSharingClusteringAlgorithm,
                                public WithPodConfig<ChargeSharingClusteringConfig> {
public:
    ChargeSharingClustering(std::string_view name)
        : ChargeSharingClusteringAlgorithm{name, {"inputHits"}, {"outputClusters"}, ""} {}

    void init() final;
    void process(const Input&, const Output&) const final;

private:
    /// Reconstruct cluster position using configured method (centroid, Gaussian 1D/2D).
    void reconstructCluster(const Output& output,
                            const std::vector<edm4eic::TrackerHit>& hits) const;

    /// Resolve MultiSegmentation to CartesianGridXY leaf.
    const dd4hep::DDSegmentation::CartesianGridXY*
    getLocalSegmentation(const dd4hep::rec::CellID& cellID) const;

    std::shared_ptr<spdlog::logger> m_log;
    const dd4hep::rec::CellIDPositionConverter* m_converter = nullptr;
    const dd4hep::DDSegmentation::BitFieldCoder* m_decoder = nullptr;
    const dd4hep::Detector* m_detector = nullptr;
    dd4hep::Segmentation m_seg;
    std::shared_ptr<const ActsGeometryProvider> m_acts_context;

    /// Cache: DetElement* → CartesianGridXY* (avoid repeated geometry lookups)
    mutable std::unordered_map<const dd4hep::DetElement*,
                               const dd4hep::DDSegmentation::CartesianGridXY*>
        m_segmentation_map;

    // ---- Union-Find for clustering ----
    class UnionFind {
        std::vector<int> mParent, mRank;

    public:
        explicit UnionFind(int n);
        int find(int id);
        void merge(int id1, int id2);
    };
};

} // namespace eicrecon
