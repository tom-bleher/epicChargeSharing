/// \file RDataFrameHelpers.hh
/// \brief RDataFrame analysis helpers for charge sharing data.
///
/// RDataFrame provides a modern, high-level interface for ROOT data analysis
/// with automatic parallelization and lazy evaluation. Works seamlessly with
/// both TTree and RNTuple formats.
///
/// Reference: https://root.cern/doc/master/classROOT_1_1RDataFrame.html
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_RDATAFRAME_HELPERS_HH
#define ECS_RDATAFRAME_HELPERS_HH

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TH1D.h>
#include <TH2D.h>

#include <memory>
#include <string>
#include <vector>

namespace ECS::Analysis {

// ============================================================================
// RDataFrame Factory
// ============================================================================

/// \brief Create an RDataFrame from a ROOT file (auto-detects TTree or RNTuple).
///
/// Example:
/// \code
/// ROOT::EnableImplicitMT();  // Enable parallelization
/// auto df = CreateDataFrame("epicChargeSharing.root");
/// auto h = df.Histo1D({"h_dx", "Delta X", 100, -0.5, 0.5}, "ReconTrueDeltaX");
/// h->Draw();
/// \endcode
inline ROOT::RDataFrame CreateDataFrame(const std::string& filename, const std::string& treeName = "Hits") {
    return ROOT::RDataFrame(treeName, filename);
}

// ============================================================================
// Common Analysis Functions
// ============================================================================

/// \brief Filter to exclude pixel hits (only gap hits).
inline std::string GapHitsFilter() {
    return "isPixelHit == false";
}

/// \brief Filter to include only events with minimum neighborhood size.
inline std::string MinNeighborhoodFilter(int minSize = 5) {
    return "NeighborhoodSize >= " + std::to_string(minSize);
}

/// \brief Define the reconstructed position error magnitude.
inline std::string DefineReconError() {
    return "sqrt(ReconTrueDeltaX*ReconTrueDeltaX + ReconTrueDeltaY*ReconTrueDeltaY)";
}

// ============================================================================
// Standard Histograms
// ============================================================================

/// \brief Create a histogram model for delta X distribution.
inline ROOT::RDF::TH1DModel DeltaXModel(int nbins = 100, double xmin = -0.5, double xmax = 0.5) {
    return {"h_deltaX", "Reconstruction #DeltaX;#DeltaX (mm);Events", nbins, xmin, xmax};
}

/// \brief Create a histogram model for delta Y distribution.
inline ROOT::RDF::TH1DModel DeltaYModel(int nbins = 100, double ymin = -0.5, double ymax = 0.5) {
    return {"h_deltaY", "Reconstruction #DeltaY;#DeltaY (mm);Events", nbins, ymin, ymax};
}

/// \brief Create a histogram model for position error magnitude.
inline ROOT::RDF::TH1DModel ErrorMagnitudeModel(int nbins = 100, double max = 1.0) {
    return {"h_error", "Position Error;|#Delta r| (mm);Events", nbins, 0, max};
}

/// \brief Create a 2D histogram model for true vs reconstructed position.
inline ROOT::RDF::TH2DModel TrueVsReconModel(int nbins = 100, double range = 0.5) {
    return {"h2_true_vs_recon",
            "True vs Recon Position;True X (mm);Recon X (mm)",
            nbins,
            -range,
            range,
            nbins,
            -range,
            range};
}

// ============================================================================
// Analysis Workflow Helpers
// ============================================================================

/// \brief Standard preprocessing: filter gap hits and define error column.
///
/// Usage:
/// \code
/// auto df = CreateDataFrame("epicChargeSharing.root");
/// auto processed = StandardPreprocess(df);
/// auto h = processed.Histo1D(ErrorMagnitudeModel(), "ReconError");
/// \endcode
template <typename T>
auto StandardPreprocess(T& df) {
    return df.Filter(GapHitsFilter()).Define("ReconError", DefineReconError());
}

/// \brief Compute resolution statistics from a filtered dataframe.
///
/// Returns mean and RMS of ReconError column.
template <typename T>
std::pair<double, double> ComputeResolution(T& df) {
    auto mean = df.Mean("ReconError");
    auto stddev = df.StdDev("ReconError");
    return {*mean, *stddev};
}

// ============================================================================
// Batch Analysis
// ============================================================================

/// \brief Analyze multiple files in parallel.
///
/// Example:
/// \code
/// std::vector<std::string> files = {"run1.root", "run2.root", "run3.root"};
/// auto histos = AnalyzeFiles(files, [](ROOT::RDataFrame& df) {
///     return df.Filter("isPixelHit == false")
///              .Histo1D({"h", "DX", 100, -0.5, 0.5}, "ReconTrueDeltaX");
/// });
/// \endcode
template <typename Func>
auto AnalyzeFiles(const std::vector<std::string>& files, Func&& analyzer) {
    std::vector<decltype(analyzer(std::declval<ROOT::RDataFrame&>()))> results;
    results.reserve(files.size());

    for (const auto& file : files) {
        auto df = CreateDataFrame(file);
        results.push_back(analyzer(df));
    }

    return results;
}

// ============================================================================
// Vector Column Helpers
// ============================================================================

/// \brief Define sum of charge fractions (should be ~1.0 for valid events).
inline std::string DefineFractionSum() {
    return "Sum(Fi)";
}

/// \brief Define maximum charge fraction in neighborhood.
inline std::string DefineMaxFraction() {
    return "Max(Fi)";
}

/// \brief Define center pixel charge fraction (index depends on radius).
inline std::string DefineCenterFraction(int radius = 2) {
    int centerIdx = radius * (2 * radius + 1) + radius;
    return "Fi[" + std::to_string(centerIdx) + "]";
}

// ============================================================================
// Example Analysis Pipeline
// ============================================================================

/// \brief Run a complete resolution analysis on a file.
///
/// Returns:
/// - Delta X histogram
/// - Delta Y histogram
/// - Error magnitude histogram
/// - Resolution statistics (mean, RMS)
///
/// Example:
/// \code
/// ROOT::EnableImplicitMT();
/// auto [h_dx, h_dy, h_err, stats] = RunResolutionAnalysis("epicChargeSharing.root");
/// std::cout << "Resolution: " << stats.second << " mm (RMS)" << std::endl;
/// \endcode
struct ResolutionAnalysisResult {
    ROOT::RDF::RResultPtr<TH1D> histoDeltaX;
    ROOT::RDF::RResultPtr<TH1D> histoDeltaY;
    ROOT::RDF::RResultPtr<TH1D> histoError;
    double meanError{0.0};
    double rmsError{0.0};
};

inline ResolutionAnalysisResult RunResolutionAnalysis(const std::string& filename) {
    ResolutionAnalysisResult result;

    auto df = CreateDataFrame(filename);
    auto filtered = df.Filter(GapHitsFilter()).Define("ReconError", DefineReconError());

    result.histoDeltaX = filtered.Histo1D(DeltaXModel(), "ReconTrueDeltaX");
    result.histoDeltaY = filtered.Histo1D(DeltaYModel(), "ReconTrueDeltaY");
    result.histoError = filtered.Histo1D(ErrorMagnitudeModel(), "ReconError");

    auto mean = filtered.Mean("ReconError");
    auto rms = filtered.StdDev("ReconError");

    // Trigger computation
    result.histoDeltaX->GetEntries();

    result.meanError = *mean;
    result.rmsError = *rms;

    return result;
}

} // namespace ECS::Analysis

#endif // ECS_RDATAFRAME_HELPERS_HH
