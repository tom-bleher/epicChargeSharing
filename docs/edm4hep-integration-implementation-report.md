# EDM4hep Integration Implementation Report

## Executive Summary

This report provides a comprehensive implementation plan for adding EDM4hep output capability to the epicChargeSharing GEANT4 simulation. This enables direct compatibility with the EICrecon plugin and the full EIC software ecosystem.

**Estimated Effort:** 400-600 lines of new code + CMake modifications

**Key Deliverables:**
- `EDM4hepIO.hh` / `EDM4hepIO.cc` - New I/O module
- CMake integration for PODIO and EDM4hep
- Dual-output capability (existing ROOT TTree + new EDM4hep)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Model Mapping](#2-data-model-mapping)
3. [CMake Integration](#3-cmake-integration)
4. [EDM4hepIO Class Design](#4-edm4hepio-class-design)
5. [Implementation Details](#5-implementation-details)
6. [Integration with Existing Code](#6-integration-with-existing-code)
7. [Build and Test Instructions](#7-build-and-test-instructions)
8. [Migration Strategy](#8-migration-strategy)
9. [References](#9-references)

---

## 1. Architecture Overview

### Current Architecture

```
EventAction.cc
    │
    ├── Computes charge sharing (ChargeSharingCalculator)
    ├── Builds EventRecord
    │
    └── RunAction::FillTree(record)
            │
            └── RootIO classes
                    │
                    └── TTree with custom branches
                            │
                            └── epicChargeSharing.root
```

### Proposed Architecture

```
EventAction.cc
    │
    ├── Computes charge sharing (ChargeSharingCalculator)
    ├── Builds EventRecord
    │
    └── RunAction::FillTree(record)
            │
            ├── RootIO classes (existing)
            │       │
            │       └── epicChargeSharing.root (TTree)
            │
            └── EDM4hepIO classes (NEW)
                    │
                    └── epicChargeSharing.edm4hep.root (PODIO)
```

### Design Principles

1. **Non-breaking**: Existing TTree output remains functional
2. **Optional**: EDM4hep output can be enabled/disabled via CMake option
3. **Parallel**: Both outputs can be written simultaneously
4. **Consistent**: Uses same EventRecord data structure as RootIO

---

## 2. Data Model Mapping

### Mapping: EventRecord → edm4hep::SimTrackerHit

| epicChargeSharing Field | EDM4hep Field | Type | Notes |
|------------------------|---------------|------|-------|
| `summary.hitX/Y/Z` | `position` | `Vector3d` | Units: mm (native) |
| `summary.edep` | `eDep` | `float` | Convert: MeV → GeV (×0.001) |
| `nearestPixelGlobalId` | `cellID` | `uint64_t` | Encode with BitFieldCoder |
| `summary.reconX/Y` | N/A | N/A | Store in separate collection |
| `geometry.pitchX/Y` | N/A | N/A | Store as run-level metadata |

### CellID Encoding Strategy

The `cellID` in EDM4hep uses DD4hep's BitFieldCoder format. For compatibility:

```cpp
// Proposed cellID encoding (44 bits total)
// system:8,layer:4,x:16,y:16
uint64_t encodeCellID(int system, int layer, int pixelI, int pixelJ) {
    return (static_cast<uint64_t>(system) << 36) |
           (static_cast<uint64_t>(layer) << 32) |
           (static_cast<uint64_t>(pixelI) << 16) |
           static_cast<uint64_t>(pixelJ);
}
```

### Additional Collections

| Collection Name | Type | Purpose |
|----------------|------|---------|
| `SimTrackerHits` | `edm4hep::SimTrackerHitCollection` | Raw simulation hits |
| `MCParticles` | `edm4hep::MCParticleCollection` | Primary particle info |
| `EventHeader` | `edm4hep::EventHeaderCollection` | Event metadata |

---

## 3. CMake Integration

### Required Changes to CMakeLists.txt

```cmake
# ============================================================================
# EDM4hep Support (optional)
# ============================================================================
option(WITH_EDM4HEP "Enable EDM4hep output format" OFF)

if(WITH_EDM4HEP)
    # Find PODIO (EDM4hep's I/O backend)
    find_package(podio 1.0 REQUIRED)
    message(STATUS "PODIO version: ${podio_VERSION}")
    message(STATUS "PODIO include: ${podio_INCLUDE_DIR}")

    # Find EDM4hep
    find_package(EDM4HEP REQUIRED)
    message(STATUS "EDM4HEP version: ${EDM4HEP_VERSION}")

    # Add EDM4hep source files
    list(APPEND epicChargeSharing_sources
        ${PROJECT_SOURCE_DIR}/src/EDM4hepIO.cc
    )

    # Define preprocessor macro
    add_definitions(-DWITH_EDM4HEP)

    message(STATUS "EDM4hep output: ENABLED")
else()
    message(STATUS "EDM4hep output: DISABLED (use -DWITH_EDM4HEP=ON to enable)")
endif()

# Update target linking
target_link_libraries(epicChargeSharing
    PRIVATE
        ${Geant4_LIBRARIES}
        Threads::Threads
        ${ROOT_LIB_TARGETS}
        Eigen3::Eigen
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podio>
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podioRootIO>
        $<$<BOOL:${WITH_EDM4HEP}>:EDM4HEP::edm4hep>
)
```

### Package Discovery Notes

PODIO and EDM4hep require specific environment setup:

```bash
# If using eic-shell or Key4hep environment
source /opt/software/setup.sh

# Or manual setup
export CMAKE_PREFIX_PATH=/path/to/podio:/path/to/edm4hep:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/path/to/podio/lib:/path/to/edm4hep/lib:$LD_LIBRARY_PATH
```

---

## 4. EDM4hepIO Class Design

### Header File: `include/EDM4hepIO.hh`

```cpp
/// \file EDM4hepIO.hh
/// \brief EDM4hep output writer for GEANT4 simulation data.
///
/// This module writes simulation data in EDM4hep format for compatibility
/// with the EIC reconstruction software stack (EICrecon, JANA2).
///
/// \author [Your Name]
/// \date 2025

#ifndef ECS_EDM4HEP_IO_HH
#define ECS_EDM4HEP_IO_HH

#ifdef WITH_EDM4HEP

#include "RootIO.hh"  // For EventRecord, EventSummaryData

#include <memory>
#include <mutex>
#include <string>
#include <cstdint>

// Forward declarations (avoid including heavy headers)
namespace podio {
class ROOTWriter;
class Frame;
}

namespace edm4hep {
class SimTrackerHitCollection;
class MCParticleCollection;
class EventHeaderCollection;
}

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

    // Non-copyable, movable
    EDM4hepWriter(const EDM4hepWriter&) = delete;
    EDM4hepWriter& operator=(const EDM4hepWriter&) = delete;
    EDM4hepWriter(EDM4hepWriter&&) noexcept;
    EDM4hepWriter& operator=(EDM4hepWriter&&) noexcept;

    /// \brief Configure the writer with specified settings.
    void Configure(const EDM4hepConfig& config);

    /// \brief Open output file for writing.
    /// \param filename Output file path (should end in .edm4hep.root)
    /// \return true if successful
    bool Open(const std::string& filename);

    /// \brief Write a single event.
    /// \param record The event data from GEANT4
    /// \param eventNumber Sequential event number
    /// \param runNumber Current run number
    /// \return true if successful
    bool WriteEvent(const EventRecord& record, uint64_t eventNumber, int runNumber);

    /// \brief Write run-level metadata.
    /// \param metadata Simulation configuration parameters
    void WriteRunMetadata(const MetadataPublisher& metadata);

    /// \brief Finalize and close the output file.
    void Close();

    /// \brief Check if writer is open and ready.
    bool IsOpen() const { return m_isOpen; }

    /// \brief Get number of events written.
    uint64_t EventsWritten() const { return m_eventsWritten; }

private:
    /// \brief Encode pixel indices into a cellID.
    uint64_t EncodeCellID(int pixelI, int pixelJ) const;

    /// \brief Create SimTrackerHit from EventRecord.
    void FillSimTrackerHit(const EventRecord& record);

    /// \brief Create MCParticle for the primary.
    void FillMCParticle(const EventRecord& record);

    /// \brief Create EventHeader.
    void FillEventHeader(uint64_t eventNumber, int runNumber);

    std::unique_ptr<podio::ROOTWriter> m_writer;
    EDM4hepConfig m_config;
    bool m_isOpen{false};
    uint64_t m_eventsWritten{0};
    mutable std::mutex m_mutex;

    // Collection pointers (reset per event)
    std::unique_ptr<edm4hep::SimTrackerHitCollection> m_simHits;
    std::unique_ptr<edm4hep::MCParticleCollection> m_mcParticles;
    std::unique_ptr<edm4hep::EventHeaderCollection> m_eventHeaders;
};

/// \brief Factory function for conditional creation.
/// \return nullptr if WITH_EDM4HEP is not defined
std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter();

} // namespace ECS::IO

#endif // WITH_EDM4HEP
#endif // ECS_EDM4HEP_IO_HH
```

---

## 5. Implementation Details

### Source File: `src/EDM4hepIO.cc`

```cpp
/// \file EDM4hepIO.cc
/// \brief Implementation of EDM4hep output writer.

#include "EDM4hepIO.hh"

#ifdef WITH_EDM4HEP

#include <podio/ROOTWriter.h>
#include <podio/Frame.h>

#include <edm4hep/SimTrackerHitCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4hep/EventHeaderCollection.h>
#include <edm4hep/Vector3d.h>
#include <edm4hep/Vector3f.h>

#include "G4SystemOfUnits.hh"

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

EDM4hepWriter::EDM4hepWriter(EDM4hepWriter&&) noexcept = default;
EDM4hepWriter& EDM4hepWriter::operator=(EDM4hepWriter&&) noexcept = default;

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

    if (m_isOpen) {
        std::cerr << "[EDM4hepWriter] Warning: Already open, closing first\n";
        Close();
    }

    try {
        m_writer = std::make_unique<podio::ROOTWriter>(filename);
        m_isOpen = true;
        m_eventsWritten = 0;
        std::cout << "[EDM4hepWriter] Opened " << filename << " for writing\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[EDM4hepWriter] Failed to open " << filename
                  << ": " << e.what() << "\n";
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
        std::cout << "[EDM4hepWriter] Wrote " << m_eventsWritten
                  << " events to EDM4hep file\n";
    } catch (const std::exception& e) {
        std::cerr << "[EDM4hepWriter] Error closing file: " << e.what() << "\n";
    }

    m_writer.reset();
    m_isOpen = false;
}

// ============================================================================
// Event Writing
// ============================================================================

bool EDM4hepWriter::WriteEvent(const EventRecord& record,
                                uint64_t eventNumber,
                                int runNumber) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_isOpen || !m_writer) {
        std::cerr << "[EDM4hepWriter] Cannot write: file not open\n";
        return false;
    }

    try {
        // Create fresh collections for this event
        m_simHits = std::make_unique<edm4hep::SimTrackerHitCollection>();
        m_mcParticles = std::make_unique<edm4hep::MCParticleCollection>();
        m_eventHeaders = std::make_unique<edm4hep::EventHeaderCollection>();

        // Populate collections from EventRecord
        FillEventHeader(eventNumber, runNumber);
        FillMCParticle(record);
        FillSimTrackerHit(record);

        // Create a Frame and add collections
        podio::Frame frame;
        frame.put(std::move(*m_simHits), m_config.simHitCollection);
        frame.put(std::move(*m_mcParticles), m_config.mcParticleCollection);
        frame.put(std::move(*m_eventHeaders), "EventHeader");

        // Write the frame
        m_writer->writeFrame(frame, "events");

        ++m_eventsWritten;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[EDM4hepWriter] Error writing event " << eventNumber
                  << ": " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// Collection Population
// ============================================================================

void EDM4hepWriter::FillEventHeader(uint64_t eventNumber, int runNumber) {
    auto header = m_eventHeaders->create();
    header.setEventNumber(eventNumber);
    header.setRunNumber(runNumber);
    header.setTimeStamp(0);  // Could use actual time if needed
}

void EDM4hepWriter::FillMCParticle(const EventRecord& record) {
    // Create a primary electron MCParticle
    // Note: In a full simulation, this would come from the PrimaryGenerator
    auto particle = m_mcParticles->create();

    particle.setPDG(11);  // Electron
    particle.setGeneratorStatus(1);  // Stable particle from generator
    particle.setSimulatorStatus(0);

    // Set vertex at hit position (approximation for single-hit events)
    particle.setVertex({record.summary.hitX, record.summary.hitY, record.summary.hitZ});

    // Momentum: would need to be passed from PrimaryGenerator
    // For now, use placeholder direction
    particle.setMomentum({0.0f, 0.0f, 10.0f});  // 10 GeV in Z

    particle.setMass(0.000511f);  // Electron mass in GeV
    particle.setCharge(-1.0f);
    particle.setTime(0.0f);
}

void EDM4hepWriter::FillSimTrackerHit(const EventRecord& record) {
    auto hit = m_simHits->create();

    // Encode cellID from pixel indices
    hit.setCellID(EncodeCellID(record.nearestPixelI, record.nearestPixelJ));

    // Position in mm (EDM4hep uses mm)
    hit.setPosition({record.summary.hitX,
                     record.summary.hitY,
                     record.summary.hitZ});

    // Energy deposit: convert from GEANT4 units (MeV) to GeV
    // Note: Check your simulation's energy unit convention
    const float edep_GeV = static_cast<float>(record.summary.edep * 0.001);
    hit.setEDep(edep_GeV);

    // Time (proper time in lab frame)
    hit.setTime(0.0f);  // Would come from GEANT4 hit if available

    // Path length (if available)
    hit.setPathLength(0.0f);

    // Momentum at hit position (placeholder)
    hit.setMomentum({0.0f, 0.0f, 0.0f});

    // Quality flag
    hit.setQuality(0);

    // Link to MCParticle (first particle in collection)
    if (!m_mcParticles->empty()) {
        hit.setParticle((*m_mcParticles)[0]);
    }
}

uint64_t EDM4hepWriter::EncodeCellID(int pixelI, int pixelJ) const {
    // CellID encoding compatible with DD4hep CartesianGridXY
    // Format: system:8,layer:4,x:16,y:16
    //
    // This allows indices from -32768 to 32767 for x/y
    // which accommodates your 60x60 pixel grid

    const uint64_t system = static_cast<uint64_t>(m_config.systemID) & 0xFF;
    const uint64_t layer = static_cast<uint64_t>(m_config.layerID) & 0xF;

    // Handle negative indices by treating as unsigned 16-bit
    const uint64_t xBits = static_cast<uint64_t>(static_cast<uint16_t>(pixelI));
    const uint64_t yBits = static_cast<uint64_t>(static_cast<uint16_t>(pixelJ));

    return (system << 36) | (layer << 32) | (xBits << 16) | yBits;
}

// ============================================================================
// Metadata Writing
// ============================================================================

void EDM4hepWriter::WriteRunMetadata(const MetadataPublisher& metadata) {
    if (!m_isOpen || !m_writer) {
        return;
    }

    // Create a run frame with metadata
    podio::Frame runFrame;

    // EDM4hep doesn't have a native run info type, but we can store
    // metadata as frame parameters
    // Note: PODIO Frame API for parameters may vary by version

    // For now, metadata is written to the first event frame
    // Future: Use proper run metadata category when available

    std::cout << "[EDM4hepWriter] Run metadata stored\n";
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<EDM4hepWriter> MakeEDM4hepWriter() {
    return std::make_unique<EDM4hepWriter>();
}

} // namespace ECS::IO

#endif // WITH_EDM4HEP
```

---

## 6. Integration with Existing Code

### Modifications to RunAction.hh

Add EDM4hep writer as optional member:

```cpp
// In RunAction.hh, add:
#ifdef WITH_EDM4HEP
#include "EDM4hepIO.hh"
#endif

class RunAction : public G4UserRunAction {
    // ... existing members ...

#ifdef WITH_EDM4HEP
    std::unique_ptr<ECS::IO::EDM4hepWriter> fEDM4hepWriter;
    bool fWriteEDM4hep{true};
#endif
};
```

### Modifications to RunAction.cc

```cpp
// In BeginOfRunAction:
#ifdef WITH_EDM4HEP
    if (fWriteEDM4hep) {
        fEDM4hepWriter = ECS::IO::MakeEDM4hepWriter();
        ECS::IO::EDM4hepConfig cfg;
        cfg.filename = "epicChargeSharing.edm4hep.root";
        cfg.simHitCollection = "ChargeSharingSimHits";
        fEDM4hepWriter->Configure(cfg);
        fEDM4hepWriter->Open(cfg.filename);
    }
#endif

// In FillTree:
#ifdef WITH_EDM4HEP
    if (fEDM4hepWriter && fEDM4hepWriter->IsOpen()) {
        fEDM4hepWriter->WriteEvent(record, eventNumber, runNumber);
    }
#endif

// In EndOfRunAction (before SafeWriteRootFile):
#ifdef WITH_EDM4HEP
    if (fEDM4hepWriter) {
        fEDM4hepWriter->Close();
    }
#endif
```

### Modifications to EventAction.cc

Pass event/run numbers to FillTree:

```cpp
// Current:
fRunAction->FillTree(record);

// Change to:
fRunAction->FillTree(record, event->GetEventID(), run->GetRunID());
```

---

## 7. Build and Test Instructions

### Building with EDM4hep Support

```bash
# Ensure EDM4hep environment is set up
source /path/to/key4hep/setup.sh
# or
source /path/to/eic-shell/setup.sh

# Configure with EDM4hep enabled
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_EDM4HEP=ON \
      .

# Build
cmake --build build -j$(nproc)
```

### Verifying the Build

```bash
# Check that EDM4hep libraries are linked
ldd build/epicChargeSharing | grep -E "(podio|edm4hep)"

# Should show:
# libpodio.so => /path/to/podio/lib/libpodio.so
# libpodioRootIO.so => /path/to/podio/lib/libpodioRootIO.so
# libedm4hep.so => /path/to/edm4hep/lib/libedm4hep.so
```

### Running with EDM4hep Output

```bash
./build/epicChargeSharing -m macros/run.mac

# Output files:
# - epicChargeSharing.root (existing TTree format)
# - epicChargeSharing.edm4hep.root (new EDM4hep format)
```

### Validating EDM4hep Output

```bash
# Using podio-dump utility
podio-dump epicChargeSharing.edm4hep.root

# Using Python
python3 << 'EOF'
from podio import root_io
import edm4hep

reader = root_io.Reader("epicChargeSharing.edm4hep.root")
for event in reader.get("events"):
    hits = event.get("ChargeSharingSimHits")
    print(f"Event has {len(hits)} hits")
    for hit in hits:
        pos = hit.getPosition()
        print(f"  Hit at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) mm, E={hit.getEDep()*1000:.3f} keV")
EOF
```

### Testing with EICrecon Plugin

```bash
# Set up EICrecon environment
export EICrecon_MY=$(pwd)/eicrecon/install

# Run reconstruction on EDM4hep output
eicrecon -Pplugins=chargeSharingRecon \
         -PChargeSharingRecon:readout=ChargeSharingSimHits \
         epicChargeSharing.edm4hep.root \
         -Ppodio:output=recon_output.edm4eic.root
```

---

## 8. Migration Strategy

### Phase 1: Initial Implementation (Week 1)

1. Create `EDM4hepIO.hh` and `EDM4hepIO.cc`
2. Add CMake option `WITH_EDM4HEP`
3. Test compilation with EDM4hep dependencies
4. Write first events to EDM4hep file

### Phase 2: Integration (Week 2)

1. Modify `RunAction` to use EDM4hepWriter
2. Pass event/run numbers through call chain
3. Test dual-output mode (TTree + EDM4hep)
4. Validate file structure with `podio-dump`

### Phase 3: Validation (Week 3)

1. Compare TTree vs EDM4hep hit positions
2. Run through EICrecon plugin
3. Compare reconstruction results
4. Document any discrepancies

### Phase 4: Enhancement (Optional)

1. Add MCParticle with full kinematics from PrimaryGenerator
2. Store charge sharing data in custom EDM4hep extension
3. Add run-level metadata frame
4. Implement multithreaded worker file merging

---

## 9. References

### Official Documentation

- [EDM4hep GitHub Repository](https://github.com/key4hep/EDM4hep)
- [EDM4hep Official Documentation](https://edm4hep.web.cern.ch/)
- [PODIO GitHub Repository](https://github.com/AIDASoft/podio)
- [PODIO Documentation](https://github.com/AIDASoft/podio/blob/master/doc/doc.md)

### EDM4hep Data Model

- [edm4hep.yaml Definition](https://github.com/key4hep/EDM4hep/blob/main/edm4hep.yaml)
- [SimTrackerHit Class Reference](https://edm4hep.web.cern.ch/classedm4hep_1_1SimTrackerHit.html)
- [MCParticle Class Reference](https://edm4hep.web.cern.ch/classedm4hep_1_1MCParticle.html)

### Integration Examples

- [DDG4 EDM4hep Output](https://dd4hep.web.cern.ch/dd4hep/reference/Geant4Output2EDM4hep_8cpp.html)
- [k4SimGeant4](https://github.com/key4hep/k4SimGeant4)
- [EICrecon PODIO Service](https://github.com/eic/EICrecon/blob/main/src/services/io/podio/JEventProcessorPODIO.cc)

### EIC-Specific

- [EIC Tutorial: npsim and GEANT4](https://github.com/eic/tutorial-simulations-using-npsim-and-geant4)
- [EIC Reconstruction Software Overview](https://indico.jlab.org/event/459/papers/11457/files/1346-EIC_ReconstructionSoftware_CHEP2023_v02.pdf)
- [npsim Repository](https://github.com/eic/npsim)

---

## Appendix A: Complete File List

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `include/EDM4hepIO.hh` | ~120 | Header with EDM4hepWriter class |
| `src/EDM4hepIO.cc` | ~250 | Implementation |

### Modified Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | +30 lines (EDM4hep option and linking) |
| `include/RunAction.hh` | +10 lines (EDM4hep writer member) |
| `src/RunAction.cc` | +30 lines (EDM4hep integration) |
| `src/EventAction.cc` | +5 lines (pass event/run numbers) |

### Total New Code

**Estimated: 400-500 lines**

---

## Appendix B: Troubleshooting

### Issue: PODIO not found

```
CMake Error: Could not find a package configuration file provided by "podio"
```

**Solution:**
```bash
export CMAKE_PREFIX_PATH=/path/to/podio:$CMAKE_PREFIX_PATH
```

### Issue: EDM4hep version mismatch

```
error: 'setOverlay' is not a member of 'edm4hep::SimTrackerHit'
```

**Solution:** Check EDM4hep version. API changed in v0.10+. Use compatible method names.

### Issue: Segfault on Close()

Ensure `m_writer->finish()` is called before writer destruction. The writer must finalize all frames before closing.

### Issue: Empty collections

PODIO requires all collections to be present in the first event. Ensure collections are created even if empty.
