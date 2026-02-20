/// \file EDM4hepIO.hh
/// \brief EDM4hep output writer for standalone GEANT4 simulation data.
///
/// This module writes simulation data in EDM4hep format with CellID encoding
/// compatible with DD4hep's BitFieldCoder. The encoding descriptor string
/// is configurable via EDM4hepConfig::cellIDEncoding and defaults to the
/// ePIC silicon tracker layout.
///
/// Enable with CMake option: -DWITH_EDM4HEP=ON
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_EDM4HEP_IO_HH
#define ECS_EDM4HEP_IO_HH

#include "RootIO.hh"

#include "BitFieldCoder.hh"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef WITH_EDM4HEP

#include <edm4hep/EventHeaderCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4hep/SimTrackerHitCollection.h>
#include <podio/Frame.h>
#include <podio/ROOTWriter.h>

namespace ECS::IO {

/// \brief Configuration for EDM4hep output.
struct EDM4hepConfig {
    std::string filename{"epicChargeSharing.edm4hep.root"};
    std::string simHitCollection{"ChargeSharingSimHits"};
    std::string mcParticleCollection{"MCParticles"};

    /// CellID encoding descriptor (DD4hep BitFieldCoder format).
    std::string cellIDEncoding{"system:8,layer:4,module:12,sensor:2,x:32:-16,y:-16"};

    /// Field names in cellIDEncoding that correspond to pixel column/row indices.
    std::string pixelFieldX{"x"};
    std::string pixelFieldY{"y"};

    /// Non-pixel CellID fields and their values.
    /// Keys must match field names in cellIDEncoding.
    /// Different detectors use different field layouts, e.g.:
    ///   B0:   {system, layer, module, sensor}
    ///   Lumi: {system, sector, module}
    std::vector<std::pair<std::string, int>> geoFields{{"system", 0}, {"layer", 0}, {"module", 0}, {"sensor", 0}};

    bool writeMetadata{true};
    bool enabled{true};

    // ---- ePIC detector presets ----

    /// B0 Tracker (far-forward, CartesianGridXZ, 70 um pixels)
    static EDM4hepConfig B0Tracker(int layer = 1) {
        EDM4hepConfig cfg;
        cfg.simHitCollection = "B0TrackerHits";
        cfg.cellIDEncoding = "system:8,layer:4,module:12,sensor:2,x:32:-16,z:-16";
        cfg.pixelFieldX = "x";
        cfg.pixelFieldY = "z";
        cfg.geoFields = {{"system", 150}, {"layer", layer}, {"module", 0}, {"sensor", 0}};
        return cfg;
    }

    /// Luminosity Spectrometer Tracker (far-backward, CartesianGridXY, 100 um pixels)
    static EDM4hepConfig LumiSpecTracker(int sector = 0, int module = 0) {
        EDM4hepConfig cfg;
        cfg.simHitCollection = "LumiSpecTrackerHits";
        cfg.cellIDEncoding = "system:8,sector:8,module:8,x:32:-16,y:-16";
        cfg.pixelFieldX = "x";
        cfg.pixelFieldY = "y";
        cfg.geoFields = {{"system", 193}, {"sector", sector}, {"module", module}};
        return cfg;
    }
};

/// \brief Writes simulation output in EDM4hep format.
///
/// This class converts EventRecord data to EDM4hep collections
/// and writes them using PODIO's ROOTWriter. CellIDs are encoded
/// using a standalone reimplementation of DD4hep's BitFieldCoder,
/// producing bit-identical CellIDs for the same descriptor string.
///
/// Thread Safety:
/// - One instance per worker thread in MT mode
/// - Uses mutex for file operations
///
/// Usage:
/// \code
///   EDM4hepWriter writer;
///   writer.Configure(config);
///   writer.Open("output.edm4hep.root");
///   // For each event:
///   writer.WriteEvent(record, eventNumber, runNumber);
///   // At end of run:
///   writer.Close();
/// \endcode
class EDM4hepWriter {
public:
    EDM4hepWriter();
    ~EDM4hepWriter();

    // Non-copyable
    EDM4hepWriter(const EDM4hepWriter&) = delete;
    EDM4hepWriter& operator=(const EDM4hepWriter&) = delete;

    // Movable
    EDM4hepWriter(EDM4hepWriter&&) noexcept;
    EDM4hepWriter& operator=(EDM4hepWriter&&) noexcept;

    /// \brief Configure the writer with specified settings.
    void Configure(const EDM4hepConfig& config);

    /// \brief Get current configuration.
    const EDM4hepConfig& GetConfig() const { return m_config; }

    /// \brief Open output file for writing.
    /// \param filename Output file path (should end in .edm4hep.root)
    /// \return true if successful
    bool Open(const std::string& filename);

    /// \brief Write a single event.
    /// \param record The event data from GEANT4
    /// \param eventNumber Sequential event number
    /// \param runNumber Current run number
    /// \return true if successful
    bool WriteEvent(const EventRecord& record, std::uint64_t eventNumber, int runNumber);

    /// \brief Write run-level metadata.
    /// \param metadata Simulation configuration parameters
    void WriteRunMetadata(const MetadataPublisher& metadata);

    /// \brief Finalize and close the output file.
    void Close();

    /// \brief Check if writer is open and ready.
    bool IsOpen() const { return m_isOpen; }

    /// \brief Check if writer is enabled.
    bool IsEnabled() const { return m_config.enabled; }

    /// \brief Get number of events written.
    std::uint64_t EventsWritten() const { return m_eventsWritten; }

private:
    /// \brief Encode pixel indices into a CellID (DD4hep BitFieldCoder format).
    std::uint64_t EncodeCellID(int pixelI, int pixelJ) const;

    /// \brief Create SimTrackerHit from EventRecord.
    void FillSimTrackerHit(edm4hep::SimTrackerHitCollection& hits, const edm4hep::MCParticle& particle,
                           const EventRecord& record);

    /// \brief Create MCParticle for the primary.
    edm4hep::MutableMCParticle FillMCParticle(edm4hep::MCParticleCollection& particles, const EventRecord& record);

    /// \brief Create EventHeader.
    void FillEventHeader(edm4hep::EventHeaderCollection& headers, std::uint64_t eventNumber, int runNumber);

    std::unique_ptr<podio::ROOTWriter> m_writer;
    EDM4hepConfig m_config;
    BitFieldCoder m_coder;
    bool m_isOpen{false};
    std::uint64_t m_eventsWritten{0};
    mutable std::mutex m_mutex;
};

/// \brief Factory function for conditional creation.
/// \return Pointer to new EDM4hepWriter
std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter();

} // namespace ECS::IO

#else // !WITH_EDM4HEP

// Stub declarations when EDM4hep is not available
namespace ECS::IO {

struct EDM4hepConfig {
    std::string filename{"epicChargeSharing.edm4hep.root"};
    std::string simHitCollection{"ChargeSharingSimHits"};
    std::string mcParticleCollection{"MCParticles"};
    std::string cellIDEncoding{"system:8,layer:4,module:12,sensor:2,x:32:-16,y:-16"};
    std::string pixelFieldX{"x"};
    std::string pixelFieldY{"y"};
    std::vector<std::pair<std::string, int>> geoFields{{"system", 0}, {"layer", 0}, {"module", 0}, {"sensor", 0}};
    bool writeMetadata{true};
    bool enabled{false}; // Disabled by default when not compiled

    static EDM4hepConfig B0Tracker(int layer = 1) {
        EDM4hepConfig cfg;
        cfg.simHitCollection = "B0TrackerHits";
        cfg.cellIDEncoding = "system:8,layer:4,module:12,sensor:2,x:32:-16,z:-16";
        cfg.pixelFieldX = "x";
        cfg.pixelFieldY = "z";
        cfg.geoFields = {{"system", 150}, {"layer", layer}, {"module", 0}, {"sensor", 0}};
        return cfg;
    }

    static EDM4hepConfig LumiSpecTracker(int sector = 0, int module = 0) {
        EDM4hepConfig cfg;
        cfg.simHitCollection = "LumiSpecTrackerHits";
        cfg.cellIDEncoding = "system:8,sector:8,module:8,x:32:-16,y:-16";
        cfg.pixelFieldX = "x";
        cfg.pixelFieldY = "y";
        cfg.geoFields = {{"system", 193}, {"sector", sector}, {"module", module}};
        return cfg;
    }
};

class EDM4hepWriter {
public:
    EDM4hepWriter() = default;
    ~EDM4hepWriter() = default;

    void Configure(const EDM4hepConfig&) {}
    [[nodiscard]] const EDM4hepConfig& GetConfig() const { return m_config; }
    static bool Open(const std::string&) { return false; }
    static bool WriteEvent(const EventRecord&, std::uint64_t, int) { return false; }
    void WriteRunMetadata(const MetadataPublisher&) {}
    void Close() {}
    [[nodiscard]] static bool IsOpen() { return false; }
    [[nodiscard]] static bool IsEnabled() { return false; }
    [[nodiscard]] static std::uint64_t EventsWritten() { return 0; }

private:
    EDM4hepConfig m_config;
};

inline std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter() {
    return std::make_unique<EDM4hepWriter>();
}

} // namespace ECS::IO

#endif // WITH_EDM4HEP

#endif // ECS_EDM4HEP_IO_HH
