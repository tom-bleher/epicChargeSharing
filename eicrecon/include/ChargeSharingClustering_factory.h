// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#pragma once

#include "ChargeSharingClustering.h"
#include "extensions/jana/JOmniFactory.h"
#include "services/geometry/dd4hep/DD4hep_service.h"
#include "services/geometry/acts/ACTSGeo_service.h"

namespace eicrecon {

class ChargeSharingClustering_factory
    : public JOmniFactory<ChargeSharingClustering_factory, ChargeSharingClusteringConfig> {
private:
    std::unique_ptr<ChargeSharingClustering> m_algo;

    PodioInput<edm4eic::TrackerHit> m_hits_input{this};
    PodioOutput<edm4eic::Measurement2D> m_clusters_output{this};

    ParameterRef<std::string> m_readout{this, "readout", config().readout};
    ParameterRef<double> m_deltaT{this, "deltaT", config().deltaT};
    ParameterRef<int> m_reconMethod{this, "reconMethod", config().reconMethod};
    ParameterRef<double> m_fitErrorPercent{this, "fitErrorPercent", config().fitErrorPercent};

public:
    void Configure() {
        m_algo = std::make_unique<ChargeSharingClustering>(GetPrefix());
        m_algo->level(static_cast<algorithms::LogLevel>(logger()->level()));
        m_algo->applyConfig(config());
        m_algo->init();
    }

    void ChangeRun(int32_t /* run_number */) {}

    void Process(int32_t /* run_number */, uint64_t /* event_number */) {
        m_algo->process({m_hits_input()}, {m_clusters_output().get()});
    }
};

} // namespace eicrecon
