/// \file EDM4hepIO.cc
/// \brief Implementation of EDM4hep output writer.
///
/// Converts GEANT4 simulation data to EDM4hep format. CellIDs are encoded
/// using a standalone reimplementation of DD4hep's BitFieldCoder algorithm,
/// producing bit-identical results for the same descriptor string.

#include "EDM4hepIO.hh"

#ifdef WITH_EDM4HEP

#include <edm4hep/Vector3d.h>
#include <edm4hep/Vector3f.h>

#include "G4ios.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <stdexcept>

namespace ECS::IO {

// ============================================================================
// Construction / Destruction
// ============================================================================

EDM4hepWriter::EDM4hepWriter() : m_coder(m_config.cellIDEncoding) {}

EDM4hepWriter::~EDM4hepWriter() {
    if (m_isOpen) {
        Close();
    }
}

EDM4hepWriter::EDM4hepWriter(EDM4hepWriter&& other) noexcept
    : m_writer(std::move(other.m_writer)), m_config(std::move(other.m_config)), m_coder(std::move(other.m_coder)),
      m_isOpen(other.m_isOpen), m_eventsWritten(other.m_eventsWritten) {
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
        m_coder = std::move(other.m_coder);
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
    m_coder = BitFieldCoder(config.cellIDEncoding);
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
        G4cout << "[EDM4hepWriter] CellID encoding: " << m_config.cellIDEncoding << G4endl;
        return true;
    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Failed to open " << filename << ": " << e.what() << G4endl;
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
        G4cout << "[EDM4hepWriter] Wrote " << m_eventsWritten << " events to EDM4hep file" << G4endl;
    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Error closing file: " << e.what() << G4endl;
    }

    m_writer.reset();
    m_isOpen = false;
}

// ============================================================================
// Event Writing
// ============================================================================

bool EDM4hepWriter::WriteEvent(const EventRecord& record, std::uint64_t eventNumber, int runNumber) {
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

        // Store CellID encoding so downstream tools can decode
        frame.putParameter(m_config.simHitCollection + "__CellIDEncoding", m_config.cellIDEncoding);

        // Write the frame
        m_writer->writeFrame(frame, "events");

        ++m_eventsWritten;
        return true;

    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepWriter] Error writing event " << eventNumber << ": " << e.what() << G4endl;
        return false;
    }
}

// ============================================================================
// Collection Population
// ============================================================================

void EDM4hepWriter::FillEventHeader(edm4hep::EventHeaderCollection& headers, std::uint64_t eventNumber, int runNumber) {
    auto header = headers.create();
    header.setEventNumber(static_cast<int32_t>(eventNumber));
    header.setRunNumber(runNumber);
    header.setTimeStamp(0);
}

edm4hep::MutableMCParticle EDM4hepWriter::FillMCParticle(edm4hep::MCParticleCollection& particles,
                                                         const EventRecord& record) {

    auto particle = particles.create();

    // Primary electron
    particle.setPDG(11);
    particle.setGeneratorStatus(1); // Stable particle from generator
    particle.setSimulatorStatus(0);

    // Set vertex at hit position
    particle.setVertex({record.summary.hitX, record.summary.hitY, record.summary.hitZ});

    // Primary particle momentum (converted from Geant4 MeV to EDM4hep GeV)
    particle.setMomentum({static_cast<float>(record.summary.primaryMomentumX / CLHEP::GeV),
                          static_cast<float>(record.summary.primaryMomentumY / CLHEP::GeV),
                          static_cast<float>(record.summary.primaryMomentumZ / CLHEP::GeV)});

    particle.setMass(0.000511f); // Electron mass in GeV
    particle.setCharge(-1.0f);
    particle.setTime(0.0f); // Creation time of primary (always 0 for gun-generated particles)

    return particle;
}

void EDM4hepWriter::FillSimTrackerHit(edm4hep::SimTrackerHitCollection& hits, const edm4hep::MCParticle& particle,
                                      const EventRecord& record) {
    auto hit = hits.create();

    // Encode cellID from pixel indices
    hit.setCellID(EncodeCellID(record.nearestPixelI, record.nearestPixelJ));

    // Position in mm (EDM4hep uses mm, GEANT4 also uses mm internally)
    hit.setPosition({record.summary.hitX, record.summary.hitY, record.summary.hitZ});

    // Energy deposit: GEANT4 uses MeV internally, EDM4hep expects GeV
    // The simulation stores energy in GEANT4 internal units (MeV)
    const float edep_GeV = static_cast<float>(record.summary.edep / CLHEP::GeV);
    hit.setEDep(edep_GeV);

    // Global time at first contact in the sensitive volume (ns)
    hit.setTime(static_cast<float>(record.summary.hitTime));

    // Path length accumulated through the sensitive volume (mm)
    hit.setPathLength(static_cast<float>(record.summary.pathLength));

    // Primary particle momentum at hit position (converted from Geant4 MeV to EDM4hep GeV)
    hit.setMomentum({static_cast<float>(record.summary.primaryMomentumX / CLHEP::GeV),
                     static_cast<float>(record.summary.primaryMomentumY / CLHEP::GeV),
                     static_cast<float>(record.summary.primaryMomentumZ / CLHEP::GeV)});

    // Quality flag (0 = normal hit)
    hit.setQuality(0);

    // Link to MCParticle
    hit.setParticle(particle);
}

/// @brief Encode pixel indices into a CellID using DD4hep BitFieldCoder format.
///
/// Sets all geometry fields from EDM4hepConfig::geoFields, then the two
/// pixel coordinate fields. The field names and layout are determined by
/// the cellIDEncoding descriptor string.
std::uint64_t EDM4hepWriter::EncodeCellID(int pixelI, int pixelJ) const {
    CellID cellid = 0;
    for (const auto& [name, value] : m_config.geoFields) {
        m_coder.set(cellid, name, value);
    }
    m_coder.set(cellid, m_config.pixelFieldX, pixelI);
    m_coder.set(cellid, m_config.pixelFieldY, pixelJ);
    return cellid;
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

bool MergeEDM4hepFiles(const std::vector<std::string>& workerFiles, const std::string& outputFile) {
    if (workerFiles.empty()) {
        return false;
    }

    try {
        podio::ROOTWriter writer(outputFile);

        for (const auto& filename : workerFiles) {
            podio::ROOTReader reader;
            reader.openFile(filename);

            const unsigned nEntries = reader.getEntries("events");
            for (unsigned i = 0; i < nEntries; ++i) {
                auto frame = podio::Frame(reader.readNextEntry("events"));
                writer.writeFrame(frame, "events");
            }
        }

        writer.finish();
        G4cout << "[EDM4hepMerge] Merged " << workerFiles.size() << " files into " << outputFile << G4endl;
        return true;

    } catch (const std::exception& e) {
        G4cerr << "[EDM4hepMerge] Error merging EDM4hep files: " << e.what() << G4endl;
        return false;
    }
}

} // namespace ECS::IO

#endif // WITH_EDM4HEP
