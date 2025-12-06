/// \file EDM4hepIO.hh
/// \brief EDM4hep output writer for GEANT4 simulation data.
///
/// This module writes simulation data in EDM4hep format for compatibility
/// with the EIC reconstruction software stack (EICrecon, JANA2).
///
/// Enable with CMake option: -DWITH_EDM4HEP=ON
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_EDM4HEP_IO_HH
#define ECS_EDM4HEP_IO_HH

#include "RootIO.hh"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#ifdef WITH_EDM4HEP

#include <podio/ROOTWriter.h>
#include <podio/Frame.h>
#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4hep/EventHeaderCollection.h>

namespace ECS::IO {

/// \brief Configuration for EDM4hep output.
struct EDM4hepConfig {
    std::string filename{"epicChargeSharing.edm4hep.root"};
    std::string simHitCollection{"ChargeSharingSimHits"};
    std::string mcParticleCollection{"MCParticles"};
    int systemID{0};        ///< Detector system identifier
    int layerID{0};         ///< Detector layer identifier
    bool writeMetadata{true};
    bool enabled{true};
};

/// \brief Writes simulation output in EDM4hep format.
///
/// This class converts EventRecord data to EDM4hep collections
/// and writes them using PODIO's ROOTWriter.
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
    /// \brief Encode pixel indices into a cellID.
    /// Format: system:8,layer:4,x:16,y:16
    std::uint64_t EncodeCellID(int pixelI, int pixelJ) const;

    /// \brief Create SimTrackerHit from EventRecord.
    void FillSimTrackerHit(edm4hep::SimTrackerHitCollection& hits,
                           const edm4hep::MCParticle& particle,
                           const EventRecord& record);

    /// \brief Create MCParticle for the primary.
    edm4hep::MutableMCParticle FillMCParticle(edm4hep::MCParticleCollection& particles,
                                               const EventRecord& record);

    /// \brief Create EventHeader.
    void FillEventHeader(edm4hep::EventHeaderCollection& headers,
                         std::uint64_t eventNumber,
                         int runNumber);

    std::unique_ptr<podio::ROOTWriter> m_writer;
    EDM4hepConfig m_config;
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
    int systemID{0};
    int layerID{0};
    bool writeMetadata{true};
    bool enabled{false};  // Disabled by default when not compiled
};

class EDM4hepWriter {
public:
    EDM4hepWriter() = default;
    ~EDM4hepWriter() = default;

    void Configure(const EDM4hepConfig&) {}
    const EDM4hepConfig& GetConfig() const { return m_config; }
    bool Open(const std::string&) { return false; }
    bool WriteEvent(const EventRecord&, std::uint64_t, int) { return false; }
    void WriteRunMetadata(const MetadataPublisher&) {}
    void Close() {}
    bool IsOpen() const { return false; }
    bool IsEnabled() const { return false; }
    std::uint64_t EventsWritten() const { return 0; }

private:
    EDM4hepConfig m_config;
};

inline std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter() {
    return std::make_unique<EDM4hepWriter>();
}

} // namespace ECS::IO

#endif // WITH_EDM4HEP

#endif // ECS_EDM4HEP_IO_HH
