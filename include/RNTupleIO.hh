/// \file RNTupleIO.hh
/// \brief RNTuple I/O classes for high-performance simulation output.
///
/// RNTuple is ROOT's next-generation columnar storage format (ROOT 6.34+).
/// Provides 20-35% smaller files and 2x faster I/O compared to TTree.
/// Reference: https://root.cern/doc/master/group__tutorial__ntuple.html
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_RNTUPLE_IO_HH
#define ECS_RNTUPLE_IO_HH

#include "RootIO.hh"
#include "Config.hh"
#include "globals.hh"

#include "RVersion.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Check ROOT version for RNTuple support
// ROOT 6.36+ uses ROOT:: namespace, ROOT 6.34-6.35 uses ROOT::Experimental::
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 36, 0)
#define ECS_RNTUPLE_AVAILABLE 1
#define ECS_RNTUPLE_NAMESPACE ROOT
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#elif ROOT_VERSION_CODE >= ROOT_VERSION(6, 34, 0)
#define ECS_RNTUPLE_AVAILABLE 1
#define ECS_RNTUPLE_NAMESPACE ROOT::Experimental
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#else
#define ECS_RNTUPLE_AVAILABLE 0
#endif

namespace ECS::IO {

// ============================================================================
// RNTuple Configuration
// ============================================================================

/// \brief Configuration for RNTuple output format.
struct RNTupleConfig {
    bool enabled{false};          ///< Use RNTuple instead of TTree
    int compressionLevel{4};       ///< ZSTD compression level (1-19)
    int clusterSize{50000};        ///< Entries per cluster (like TTree AutoFlush)
    std::string algorithm{"zstd"}; ///< Compression algorithm

    static RNTupleConfig Default() { return {}; }
};

// ============================================================================
// RNTuple Field Buffers
// ============================================================================

/// \brief Data buffers for RNTuple fields.
///
/// Matches the TTree branch structure for seamless migration.
struct RNTupleFieldBuffers {
    // Scalar fields
    double trueX{0.0}, trueY{0.0};
    double pixelX{0.0}, pixelY{0.0};
    double edep{0.0};
    double pixelTrueDeltaX{0.0}, pixelTrueDeltaY{0.0};
    double reconX{0.0}, reconY{0.0};
    double reconTrueDeltaX{0.0}, reconTrueDeltaY{0.0};
    bool isPixelHit{false};
    int neighborhoodActiveCells{0};
    int nearestPixelI{-1}, nearestPixelJ{-1}, nearestPixelGlobalId{-1};

    // Vector fields - Neighborhood mode
    std::vector<double> chargeFractions;
    std::vector<double> charge;
    std::vector<double> chargeNew;
    std::vector<double> chargeFinal;

    // Vector fields - Geometry
    std::vector<double> pixelXVec;
    std::vector<double> pixelYVec;
    std::vector<double> distance;
    std::vector<double> alpha;
    std::vector<int> pixelID;

    // Full grid fields
    std::vector<double> fullFi;
    std::vector<double> fullQi, fullQn, fullQf;
    std::vector<double> fullDistance, fullAlpha;
    std::vector<double> fullPixelX, fullPixelY;
    int fullGridSide{0};
};

// ============================================================================
// RNTuple Writer
// ============================================================================

/// \brief High-performance RNTuple writer for simulation output.
///
/// Provides 20-35% smaller files and faster I/O than TTree.
/// Requires ROOT 6.34+ with RNTuple production-ready format.
///
/// Example usage:
/// \code
/// RNTupleWriter writer;
/// writer.Open("output.root", "Hits");
/// // Fill buffers
/// writer.Fill();
/// writer.Close();
/// \endcode
class RNTupleWriter {
public:
    RNTupleWriter();
    ~RNTupleWriter();

    /// \brief Open RNTuple for writing.
    /// \param filename Output file path
    /// \param ntupleName Name of the RNTuple (like TTree name)
    /// \param config RNTuple configuration
    /// \return true on success
    bool Open(const std::string& filename, const std::string& ntupleName = "Hits",
              const RNTupleConfig& config = RNTupleConfig::Default());

    /// \brief Check if RNTuple is open for writing.
    bool IsOpen() const;

    /// \brief Fill one entry from current buffer values.
    /// \return true on success
    bool Fill();

    /// \brief Flush pending writes to disk.
    void Flush();

    /// \brief Close the RNTuple and finalize file.
    void Close();

    /// \brief Get mutable access to field buffers.
    RNTupleFieldBuffers& Buffers() { return fBuffers; }

    /// \brief Get const access to field buffers.
    const RNTupleFieldBuffers& Buffers() const { return fBuffers; }

    /// \brief Fill buffers from an EventRecord.
    void FillFromRecord(const EventRecord& record);

    /// \brief Set the denominator mode for fraction calculation.
    void SetDenominatorMode(Config::DenominatorMode mode) { fDenominatorMode = mode; }

    /// \brief Enable/disable full grid storage.
    void SetStoreFullGrid(bool enable) { fStoreFullGrid = enable; }

    /// \brief Set neighborhood radius for buffer sizing.
    void SetNeighborhoodRadius(int radius) { fNeighborhoodRadius = radius; }

private:
#if ECS_RNTUPLE_AVAILABLE
    /// \brief Create the RNTuple model with all fields.
    std::unique_ptr<ECS_RNTUPLE_NAMESPACE::RNTupleModel> CreateModel();

    std::unique_ptr<ECS_RNTUPLE_NAMESPACE::RNTupleWriter> fWriter;
#else
    void* fWriter{nullptr};
#endif

    RNTupleFieldBuffers fBuffers;
    Config::DenominatorMode fDenominatorMode{Config::DenominatorMode::Neighborhood};
    bool fStoreFullGrid{false};
    int fNeighborhoodRadius{2};
    mutable std::mutex fMutex;
};

// ============================================================================
// RNTuple Reader
// ============================================================================

/// \brief RNTuple reader for analysis with RDataFrame compatibility.
///
/// RDataFrame automatically detects RNTuple format and provides
/// the same interface as TTree reading.
///
/// Example with RDataFrame:
/// \code
/// ROOT::EnableImplicitMT();
/// ROOT::RDataFrame df("Hits", "output.root");
/// auto h = df.Filter("NeighborhoodSize > 4")
///            .Histo1D("TrueX");
/// \endcode
class RNTupleReader {
public:
    RNTupleReader();
    ~RNTupleReader();

    /// \brief Open RNTuple for reading.
    bool Open(const std::string& filename, const std::string& ntupleName = "Hits");

    /// \brief Check if RNTuple is open.
    bool IsOpen() const;

    /// \brief Get total number of entries.
    long long GetEntries() const;

    /// \brief Load entry into buffers.
    bool GetEntry(long long entry);

    /// \brief Get const access to field buffers.
    const RNTupleFieldBuffers& Buffers() const { return fBuffers; }

    /// \brief Close the RNTuple.
    void Close();

private:
    RNTupleFieldBuffers fBuffers;
#if ECS_RNTUPLE_AVAILABLE
    std::unique_ptr<ECS_RNTUPLE_NAMESPACE::RNTupleReader> fReader;
#else
    void* fReader{nullptr};
#endif
};

// ============================================================================
// Utility Functions
// ============================================================================

/// \brief Check if ROOT supports RNTuple (version 6.34+).
bool IsRNTupleSupported();

/// \brief Get RNTuple format version string.
std::string GetRNTupleVersion();

/// \brief Convert TTree file to RNTuple format.
/// \param inputFile Path to TTree file
/// \param outputFile Path for RNTuple output
/// \param treeName Name of TTree to convert
/// \return true on success
bool ConvertTreeToRNTuple(const std::string& inputFile,
                          const std::string& outputFile,
                          const std::string& treeName = "Hits");

} // namespace ECS::IO

#endif // ECS_RNTUPLE_IO_HH
