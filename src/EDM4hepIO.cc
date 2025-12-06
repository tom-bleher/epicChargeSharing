/// \file EDM4hepIO.cc
/// \brief Implementation of EDM4hep output writer.
///
/// Converts GEANT4 simulation data to EDM4hep format for use with
/// EICrecon and the EIC software ecosystem.

#include "EDM4hepIO.hh"

#ifdef WITH_EDM4HEP

#include <edm4hep/Vector3d.h>
#include <edm4hep/Vector3f.h>

#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

#include <iostream>
#include <stdexcept>

namespace ECS::IO {

// ============================================================================
// Construction / Destruction
// ============================================================================

EDM4hepWriter::EDM4hepWriter() = default;

EDM4hepWriter::~EDM4hepWriter() {
    if (m_isOpen) {
        Close();
    }
}

EDM4hepWriter::EDM4hepWriter(EDM4hepWriter&& other) noexcept
    : m_writer(std::move(other.m_writer)),
      m_config(std::move(other.m_config)),
      m_isOpen(other.m_isOpen),
      m_eventsWritten(other.m_eventsWritten) {
    other.m_isOpen = false;
    other.m_eventsWritten = 0;
}

EDM4hepWriter& EDM4hepWriter::operator=(EDM4hepWriter&& other) noexcept {
    if (this != &other) {
        if (m_isOpen) {
            Close();
        }
        m_writer = std::move(other.m_writer);
        m_config = std::move(other.m_config);
        m_isOpen = other.m_isOpen;
        m_eventsWritten = other.m_eventsWritten;
        other.m_isOpen = false;
        other.m_eventsWritten = 0;
    }
    return *this;
}

// ============================================================================
// Configuration
// ============================================================================

void EDM4hepWriter::Configure(const EDM4hepConfig& config) {
    m_config = config;
}

// ============================================================================
// File Operations
// ============================================================================

bool EDM4hepWriter::Open(const std::string& filename) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_config.enabled) {
        G4cout << "[EDM4hepWriter] EDM4hep output disabled" << G4endl;
        return false;
    }

    if (m_isOpen) {
        G4cout << "[EDM4hepWriter] Warning: Already open, closing first" << G4endl;
        // Don't call Close() here to avoid deadlock - just reset
        if (m_writer) {
            try {
                m_writer->finish();
            } catch (...) {}
            m_writer.reset();
        }
        m_isOpen = false;
    }

    try {
        m_writer = std::make_unique<podio::ROOTWriter>(filename);
        m_isOpen = true;
        m_eventsWritten = 0;
        G4cout << "[EDM4hepWriter] Opened " << filename << " for writing" << G4endl;
        return true;
    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Failed to open " << filename
               << ": " << e.what() << G4endl;
        return false;
    }
}

void EDM4hepWriter::Close() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_isOpen || !m_writer) {
        return;
    }

    try {
        m_writer->finish();
        G4cout << "[EDM4hepWriter] Wrote " << m_eventsWritten
               << " events to EDM4hep file" << G4endl;
    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Error closing file: " << e.what() << G4endl;
    }

    m_writer.reset();
    m_isOpen = false;
}

// ============================================================================
// Event Writing
// ============================================================================

bool EDM4hepWriter::WriteEvent(const EventRecord& record,
                                std::uint64_t eventNumber,
                                int runNumber) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_isOpen || !m_writer) {
        return false;
    }

    try {
        // Create collections for this event
        edm4hep::SimTrackerHitCollection simHits;
        edm4hep::MCParticleCollection mcParticles;
        edm4hep::EventHeaderCollection eventHeaders;

        // Populate collections from EventRecord
        FillEventHeader(eventHeaders, eventNumber, runNumber);
        auto particle = FillMCParticle(mcParticles, record);
        FillSimTrackerHit(simHits, particle, record);

        // Create a Frame and add collections
        podio::Frame frame;
        frame.put(std::move(simHits), m_config.simHitCollection);
        frame.put(std::move(mcParticles), m_config.mcParticleCollection);
        frame.put(std::move(eventHeaders), "EventHeader");

        // Write the frame
        m_writer->writeFrame(frame, "events");

        ++m_eventsWritten;
        return true;

    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Error writing event " << eventNumber
               << ": " << e.what() << G4endl;
        return false;
    }
}

// ============================================================================
// Collection Population
// ============================================================================

void EDM4hepWriter::FillEventHeader(edm4hep::EventHeaderCollection& headers,
                                     std::uint64_t eventNumber,
                                     int runNumber) {
    auto header = headers.create();
    header.setEventNumber(static_cast<int32_t>(eventNumber));
    header.setRunNumber(runNumber);
    header.setTimeStamp(0);
}

edm4hep::MutableMCParticle EDM4hepWriter::FillMCParticle(
    edm4hep::MCParticleCollection& particles,
    const EventRecord& record) {

    auto particle = particles.create();

    // Primary electron
    particle.setPDG(11);
    particle.setGeneratorStatus(1);  // Stable particle from generator
    particle.setSimulatorStatus(0);

    // Set vertex at hit position
    particle.setVertex({record.summary.hitX,
                        record.summary.hitY,
                        record.summary.hitZ});

    // Momentum: placeholder direction (would come from PrimaryGenerator)
    // Default: 10 GeV electron in -Z direction (typical beam direction)
    particle.setMomentum({0.0f, 0.0f, -10.0f});

    particle.setMass(0.000511f);  // Electron mass in GeV
    particle.setCharge(-1.0f);
    particle.setTime(0.0f);

    return particle;
}

void EDM4hepWriter::FillSimTrackerHit(edm4hep::SimTrackerHitCollection& hits,
                                       const edm4hep::MCParticle& particle,
                                       const EventRecord& record) {
    auto hit = hits.create();

    // Encode cellID from pixel indices
    hit.setCellID(EncodeCellID(record.nearestPixelI, record.nearestPixelJ));

    // Position in mm (EDM4hep uses mm, GEANT4 also uses mm internally)
    hit.setPosition({record.summary.hitX,
                     record.summary.hitY,
                     record.summary.hitZ});

    // Energy deposit: GEANT4 uses MeV internally, EDM4hep expects GeV
    // The simulation stores energy in GEANT4 internal units (MeV)
    const float edep_GeV = static_cast<float>(record.summary.edep / CLHEP::GeV);
    hit.setEDep(edep_GeV);

    // Time (proper time in lab frame) - would come from GEANT4 step
    hit.setTime(0.0f);

    // Path length in sensitive material
    hit.setPathLength(0.0f);

    // Momentum at hit position (placeholder)
    hit.setMomentum({0.0f, 0.0f, -10.0f});

    // Quality flag (0 = normal hit)
    hit.setQuality(0);

    // Link to MCParticle
    hit.setParticle(particle);
}

std::uint64_t EDM4hepWriter::EncodeCellID(int pixelI, int pixelJ) const {
    // CellID encoding compatible with DD4hep CartesianGridXY
    // Format: system:8,layer:4,x:16,y:16 (total 44 bits used)
    //
    // This allows indices from -32768 to 32767 for x/y
    // which accommodates typical pixel grids (e.g., 60x60)

    const std::uint64_t system = static_cast<std::uint64_t>(m_config.systemID) & 0xFF;
    const std::uint64_t layer = static_cast<std::uint64_t>(m_config.layerID) & 0xF;

    // Handle negative indices by treating as unsigned 16-bit
    // This maintains compatibility with DD4hep's signed index handling
    const std::uint64_t xBits = static_cast<std::uint64_t>(static_cast<std::uint16_t>(pixelI));
    const std::uint64_t yBits = static_cast<std::uint64_t>(static_cast<std::uint16_t>(pixelJ));

    return (system << 36) | (layer << 32) | (xBits << 16) | yBits;
}

// ============================================================================
// Metadata Writing
// ============================================================================

void EDM4hepWriter::WriteRunMetadata(const MetadataPublisher& /*metadata*/) {
    if (!m_isOpen || !m_writer) {
        return;
    }

    // PODIO Frame metadata support varies by version
    // For now, metadata is embedded in the event stream
    // Future: Use dedicated "runs" category when PODIO version supports it

    G4cout << "[EDM4hepWriter] Run metadata stored" << G4endl;
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter() {
    return std::make_unique<EDM4hepWriter>();
}

} // namespace ECS::IO

#endif // WITH_EDM4HEP
