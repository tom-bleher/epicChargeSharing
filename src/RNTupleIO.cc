/// \file RNTupleIO.cc
/// \brief Implementation of RNTuple I/O classes.

#include "RNTupleIO.hh"
#include "Config.hh"

#include "G4Exception.hh"
#include "G4ios.hh"

#include "RVersion.h"

// RNTuple headers (ROOT 6.34+)
#if ECS_RNTUPLE_AVAILABLE
#include <ROOT/RNTupleImporter.hxx>
#endif

#include <TFile.h>

#include <stdexcept>

namespace ECS::IO {

// ============================================================================
// Utility Functions
// ============================================================================

bool IsRNTupleSupported() {
#if ECS_RNTUPLE_AVAILABLE
    return true;
#else
    return false;
#endif
}

std::string GetRNTupleVersion() {
    const int major = ROOT_VERSION_CODE / 10000;
    const int minor = (ROOT_VERSION_CODE / 100) % 100;
    const int patch = ROOT_VERSION_CODE % 100;
#if ECS_RNTUPLE_AVAILABLE
    return "ROOT " + std::to_string(major) + "." +
           std::to_string(minor) + "." +
           std::to_string(patch) + " (RNTuple production-ready)";
#else
    return "ROOT " + std::to_string(major) + "." +
           std::to_string(minor) + "." +
           std::to_string(patch) + " (RNTuple not available, requires 6.34+)";
#endif
}

bool ConvertTreeToRNTuple(const std::string& inputFile,
                          const std::string& outputFile,
                          const std::string& treeName) {
#if ECS_RNTUPLE_AVAILABLE
    try {
        // RNTupleImporter remains in ROOT::Experimental even in ROOT 6.36+
        auto importer = ROOT::Experimental::RNTupleImporter::Create(
            inputFile, treeName, outputFile);
        importer->SetNTupleName(treeName);
        importer->Import();
        G4cout << "[RNTupleIO] Successfully converted " << inputFile
               << " to RNTuple format: " << outputFile << G4endl;
        return true;
    } catch (const std::exception& e) {
        G4Exception("ConvertTreeToRNTuple", "RNTupleConversionError",
                    JustWarning, e.what());
        return false;
    }
#else
    G4Exception("ConvertTreeToRNTuple", "RNTupleNotAvailable",
                JustWarning, "RNTuple requires ROOT 6.34+");
    (void)inputFile; (void)outputFile; (void)treeName;
    return false;
#endif
}

// ============================================================================
// RNTupleWriter Implementation
// ============================================================================

RNTupleWriter::RNTupleWriter() = default;

RNTupleWriter::~RNTupleWriter() {
    Close();
}

#if ECS_RNTUPLE_AVAILABLE
std::unique_ptr<ECS_RNTUPLE_NAMESPACE::RNTupleModel> RNTupleWriter::CreateModel() {
    auto model = ECS_RNTUPLE_NAMESPACE::RNTupleModel::Create();

    // Scalar fields - use shared pointers for ROOT 6.36+
    auto fTrueX = model->MakeField<double>("TrueX");
    auto fTrueY = model->MakeField<double>("TrueY");
    auto fPixelX = model->MakeField<double>("PixelX");
    auto fPixelY = model->MakeField<double>("PixelY");
    auto fEdep = model->MakeField<double>("Edep");
    auto fPixelTrueDeltaX = model->MakeField<double>("PixelTrueDeltaX");
    auto fPixelTrueDeltaY = model->MakeField<double>("PixelTrueDeltaY");
    auto fReconX = model->MakeField<double>("ReconX");
    auto fReconY = model->MakeField<double>("ReconY");
    auto fReconTrueDeltaX = model->MakeField<double>("ReconTrueDeltaX");
    auto fReconTrueDeltaY = model->MakeField<double>("ReconTrueDeltaY");

    // Classification fields
    auto fIsPixelHit = model->MakeField<bool>("isPixelHit");
    auto fNeighborhoodSize = model->MakeField<int>("NeighborhoodSize");
    auto fNearestPixelI = model->MakeField<int>("NearestPixelI");
    auto fNearestPixelJ = model->MakeField<int>("NearestPixelJ");
    auto fNearestPixelID = model->MakeField<int>("NearestPixelID");

    // Vector fields - Neighborhood mode
    auto fFi = model->MakeField<std::vector<double>>("Fi");
    auto fQi = model->MakeField<std::vector<double>>("Qi");
    auto fQn = model->MakeField<std::vector<double>>("Qn");
    auto fQf = model->MakeField<std::vector<double>>("Qf");

    // Vector fields - Geometry
    auto fNeighborhoodPixelX = model->MakeField<std::vector<double>>("NeighborhoodPixelX");
    auto fNeighborhoodPixelY = model->MakeField<std::vector<double>>("NeighborhoodPixelY");
    auto fDi = model->MakeField<std::vector<double>>("d_i");
    auto fAlphaI = model->MakeField<std::vector<double>>("alpha_i");
    auto fNeighborhoodPixelID = model->MakeField<std::vector<int>>("NeighborhoodPixelID");

    // Full grid fields (if enabled)
    if (fStoreFullGrid) {
        auto fFiGrid = model->MakeField<std::vector<double>>("FiGrid");
        auto fQiGrid = model->MakeField<std::vector<double>>("QiGrid");
        auto fQnGrid = model->MakeField<std::vector<double>>("QnGrid");
        auto fQfGrid = model->MakeField<std::vector<double>>("QfGrid");
        auto fDistanceGrid = model->MakeField<std::vector<double>>("DistanceGrid");
        auto fAlphaGrid = model->MakeField<std::vector<double>>("AlphaGrid");
        auto fPixelXGrid = model->MakeField<std::vector<double>>("PixelXGrid");
        auto fPixelYGrid = model->MakeField<std::vector<double>>("PixelYGrid");
        auto fFullGridSide = model->MakeField<int>("FullGridSide");
    }

    return model;
}

bool RNTupleWriter::Open(const std::string& filename, const std::string& ntupleName,
                          const RNTupleConfig& config) {
    std::lock_guard<std::mutex> lock(fMutex);

    if (fWriter) {
        Close();
    }

    try {
        auto model = CreateModel();
        if (!model) {
            G4Exception("RNTupleWriter::Open", "ModelCreationFailed",
                        FatalException, "Failed to create RNTuple model");
            return false;
        }

        // Configure write options
        ECS_RNTUPLE_NAMESPACE::RNTupleWriteOptions options;
        options.SetCompression(config.compressionLevel);

        fWriter = ECS_RNTUPLE_NAMESPACE::RNTupleWriter::Recreate(
            std::move(model), ntupleName, filename, options);

        G4cout << "[RNTupleIO] Opened RNTuple for writing: " << filename
               << " (compression=" << config.compressionLevel << ")" << G4endl;
        return true;
    } catch (const std::exception& e) {
        G4Exception("RNTupleWriter::Open", "RNTupleOpenError",
                    FatalException, e.what());
        return false;
    }
}

bool RNTupleWriter::IsOpen() const {
    std::lock_guard<std::mutex> lock(fMutex);
    return fWriter != nullptr;
}

bool RNTupleWriter::Fill() {
    std::lock_guard<std::mutex> lock(fMutex);
    if (!fWriter) return false;

    try {
        fWriter->Fill();
        return true;
    } catch (const std::exception& e) {
        G4Exception("RNTupleWriter::Fill", "RNTupleFillError",
                    JustWarning, e.what());
        return false;
    }
}

void RNTupleWriter::Flush() {
    std::lock_guard<std::mutex> lock(fMutex);
    if (fWriter) {
        fWriter->CommitCluster();
    }
}

void RNTupleWriter::Close() {
    std::lock_guard<std::mutex> lock(fMutex);
    if (fWriter) {
        fWriter.reset();
        G4cout << "[RNTupleIO] RNTuple closed successfully" << G4endl;
    }
}

bool RNTupleReader::Open(const std::string& filename, const std::string& ntupleName) {
    try {
        fReader = ECS_RNTUPLE_NAMESPACE::RNTupleReader::Open(ntupleName, filename);
        G4cout << "[RNTupleIO] Opened RNTuple for reading: " << filename
               << " (" << GetEntries() << " entries)" << G4endl;
        return true;
    } catch (const std::exception& e) {
        G4Exception("RNTupleReader::Open", "RNTupleOpenError",
                    JustWarning, e.what());
        return false;
    }
}

bool RNTupleReader::IsOpen() const {
    return fReader != nullptr;
}

long long RNTupleReader::GetEntries() const {
    return fReader ? static_cast<long long>(fReader->GetNEntries()) : 0;
}

bool RNTupleReader::GetEntry(long long entry) {
    if (!fReader || entry < 0 || entry >= GetEntries()) return false;

    try {
        fReader->LoadEntry(static_cast<std::uint64_t>(entry));
        return true;
    } catch (...) {
        return false;
    }
}

void RNTupleReader::Close() {
    if (fReader) {
        fReader.reset();
    }
}

#else // !ECS_RNTUPLE_AVAILABLE - Stub implementations

bool RNTupleWriter::Open(const std::string& filename, const std::string& ntupleName,
                          const RNTupleConfig& config) {
    G4Exception("RNTupleWriter::Open", "RNTupleNotAvailable",
                JustWarning, "RNTuple requires ROOT 6.34+. Use TTree output instead.");
    (void)filename; (void)ntupleName; (void)config;
    return false;
}

bool RNTupleWriter::IsOpen() const {
    return false;
}

bool RNTupleWriter::Fill() {
    return false;
}

void RNTupleWriter::Flush() {
}

void RNTupleWriter::Close() {
}

bool RNTupleReader::Open(const std::string& filename, const std::string& ntupleName) {
    G4Exception("RNTupleReader::Open", "RNTupleNotAvailable",
                JustWarning, "RNTuple requires ROOT 6.34+");
    (void)filename; (void)ntupleName;
    return false;
}

bool RNTupleReader::IsOpen() const {
    return false;
}

long long RNTupleReader::GetEntries() const {
    return 0;
}

bool RNTupleReader::GetEntry(long long entry) {
    (void)entry;
    return false;
}

void RNTupleReader::Close() {
}

#endif // ECS_RNTUPLE_AVAILABLE

// ============================================================================
// Common implementations (always available)
// ============================================================================

void RNTupleWriter::FillFromRecord(const EventRecord& record) {
    // Scalar fields
    fBuffers.trueX = record.summary.hitX;
    fBuffers.trueY = record.summary.hitY;
    fBuffers.pixelX = record.summary.nearestPixelX;
    fBuffers.pixelY = record.summary.nearestPixelY;
    fBuffers.edep = record.summary.edep;
    fBuffers.pixelTrueDeltaX = record.summary.pixelTrueDeltaX;
    fBuffers.pixelTrueDeltaY = record.summary.pixelTrueDeltaY;
    fBuffers.reconX = record.summary.reconX;
    fBuffers.reconY = record.summary.reconY;
    fBuffers.reconTrueDeltaX = record.summary.reconTrueDeltaX;
    fBuffers.reconTrueDeltaY = record.summary.reconTrueDeltaY;
    fBuffers.isPixelHit = record.summary.isPixelHitCombined;
    fBuffers.nearestPixelI = record.nearestPixelI;
    fBuffers.nearestPixelJ = record.nearestPixelJ;
    fBuffers.nearestPixelGlobalId = record.nearestPixelGlobalId;

    // Calculate neighborhood size
    const int side = 2 * fNeighborhoodRadius + 1;
    const std::size_t capacity = static_cast<std::size_t>(side * side);

    // Resize vectors
    fBuffers.chargeFractions.resize(capacity);
    fBuffers.charge.resize(capacity);
    fBuffers.chargeNew.resize(capacity);
    fBuffers.chargeFinal.resize(capacity);
    fBuffers.pixelXVec.resize(capacity);
    fBuffers.pixelYVec.resize(capacity);
    fBuffers.distance.resize(capacity);
    fBuffers.alpha.resize(capacity);
    fBuffers.pixelID.resize(capacity);

    // Fill from neighbor cells
    std::size_t activeCells = 0;
    for (const auto& cell : record.neighborCells) {
        if (cell.gridIndex < 0) continue;
        const auto idx = static_cast<std::size_t>(cell.gridIndex);
        if (idx >= capacity) continue;

        fBuffers.chargeFractions[idx] = cell.fraction;
        fBuffers.charge[idx] = cell.charge;
        fBuffers.pixelXVec[idx] = cell.center.x();
        fBuffers.pixelYVec[idx] = cell.center.y();
        fBuffers.pixelID[idx] = cell.globalPixelId;
        if (record.includeDistanceAlpha) {
            fBuffers.distance[idx] = cell.distance;
            fBuffers.alpha[idx] = cell.alpha;
        }
        ++activeCells;
    }
    fBuffers.neighborhoodActiveCells = static_cast<int>(activeCells);

    // Copy charge arrays
    const std::size_t copyNew = std::min(capacity, record.neighborChargesNew.size());
    for (std::size_t i = 0; i < copyNew; ++i) {
        fBuffers.chargeNew[i] = record.neighborChargesNew[i];
    }
    const std::size_t copyFinal = std::min(capacity, record.neighborChargesFinal.size());
    for (std::size_t i = 0; i < copyFinal; ++i) {
        fBuffers.chargeFinal[i] = record.neighborChargesFinal[i];
    }

    // Full grid data (if enabled)
    if (fStoreFullGrid && record.fullGridCols > 0) {
        fBuffers.fullGridSide = record.fullGridCols;

        fBuffers.fullFi.assign(record.fullFi.begin(), record.fullFi.end());
        fBuffers.fullQi.assign(record.fullQi.begin(), record.fullQi.end());
        fBuffers.fullQn.assign(record.fullQn.begin(), record.fullQn.end());
        fBuffers.fullQf.assign(record.fullQf.begin(), record.fullQf.end());
        fBuffers.fullDistance.assign(record.fullDistance.begin(), record.fullDistance.end());
        fBuffers.fullAlpha.assign(record.fullAlpha.begin(), record.fullAlpha.end());
        fBuffers.fullPixelX.assign(record.fullPixelX.begin(), record.fullPixelX.end());
        fBuffers.fullPixelY.assign(record.fullPixelY.begin(), record.fullPixelY.end());
    }

    Fill();
}

RNTupleReader::RNTupleReader() = default;

RNTupleReader::~RNTupleReader() {
    Close();
}

} // namespace ECS::IO
