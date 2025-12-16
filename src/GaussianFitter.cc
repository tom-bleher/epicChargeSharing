/// @file GaussianFitter.cc
/// @brief Compiled Gaussian fitting routines for position reconstruction.
///
/// This is a compiled version of FitGaussian1D.C and FitGaussian2D.C.
/// Configuration is read from Config.hh at compile time for better performance.

#include "GaussianFitter.hh"
#include "Config.hh"
#include "FitUncertainty.hh"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TList.h>
#include <TGraph.h>
#include <TGraph2D.h>
#include <TGraphErrors.h>
#include <TGraph2DErrors.h>
#include <TF1.h>
#include <TF2.h>
#include <TROOT.h>
#include <TError.h>
#include <TMath.h>
#include <Math/MinimizerOptions.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>
#include <Fit/Fitter.h>
#include <Fit/BinData.h>
#include <Fit/Chi2FCN.h>
#include <Math/WrappedMultiTF1.h>
#include <ROOT/TThreadExecutor.hxx>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <atomic>
#include <memory>
#include <span>
#include <utility>
#include <optional>

namespace ECS::Fit {

// ============================================================================
// Configuration from Config.hh (compile-time constants)
// ============================================================================
namespace Config {
    // Uncertainty model
    constexpr double ERROR_PERCENT_OF_MAX       = Constants::FIT_ERROR_PERCENT_OF_MAX;
    constexpr bool   USE_VERTICAL_UNCERTAINTIES = Constants::FIT_USE_VERTICAL_UNCERTAINTIES;
    constexpr bool   USE_QN_QI_ERRORS           = Constants::FIT_USE_QN_QI_ERRORS;

    // Distance-weighted errors
    constexpr bool   USE_DISTANCE_WEIGHTED_ERRORS   = Constants::FIT_USE_DISTANCE_WEIGHTED_ERRORS;
    constexpr double DISTANCE_SCALE_PIXELS          = Constants::FIT_DISTANCE_SCALE_PIXELS;
    constexpr double DISTANCE_EXPONENT              = Constants::FIT_DISTANCE_EXPONENT;
    constexpr double DISTANCE_FLOOR_PERCENT         = Constants::FIT_DISTANCE_FLOOR_PERCENT;
    constexpr double DISTANCE_CAP_PERCENT           = Constants::FIT_DISTANCE_CAP_PERCENT;
    constexpr bool   DISTANCE_PREFER_TRUTH_CENTER   = Constants::FIT_DISTANCE_PREFER_TRUTH_CENTER;
    constexpr bool   DISTANCE_POWER_INVERSE         = Constants::FIT_DISTANCE_POWER_INVERSE;

    // Charge branches
    constexpr const char* CHARGE_BRANCH_1D = Constants::FIT_CHARGE_BRANCH_1D;
    constexpr const char* CHARGE_BRANCH_2D = Constants::FIT_CHARGE_BRANCH_2D;

    // 1D fit options
    constexpr bool FIT_DIAGONALS       = Constants::FIT_1D_DIAGONALS;
    constexpr bool SAVE_1D_A           = Constants::FIT_1D_SAVE_A;
    constexpr bool SAVE_1D_MU          = Constants::FIT_1D_SAVE_MU;
    constexpr bool SAVE_1D_SIGMA       = Constants::FIT_1D_SAVE_SIGMA;
    constexpr bool SAVE_1D_B           = Constants::FIT_1D_SAVE_B;
    constexpr bool SAVE_LINE_MEANS     = Constants::FIT_1D_SAVE_LINE_MEANS;
    constexpr bool SAVE_DIAG_A         = Constants::FIT_1D_SAVE_DIAG_A;
    constexpr bool SAVE_DIAG_MU        = Constants::FIT_1D_SAVE_DIAG_MU;
    constexpr bool SAVE_DIAG_SIGMA     = Constants::FIT_1D_SAVE_DIAG_SIGMA;
    constexpr bool SAVE_DIAG_B         = Constants::FIT_1D_SAVE_DIAG_B;

    // 1D distance model
    constexpr bool   DIST_1D_ENABLED       = Constants::FIT_1D_DIST_ENABLED;
    constexpr double DIST_1D_SCALE_PIXELS  = Constants::FIT_1D_DIST_SCALE_PIXELS;
    constexpr double DIST_1D_EXPONENT      = Constants::FIT_1D_DIST_EXPONENT;
    constexpr double DIST_1D_FLOOR_PERCENT = Constants::FIT_1D_DIST_FLOOR_PERCENT;
    constexpr double DIST_1D_CAP_PERCENT   = Constants::FIT_1D_DIST_CAP_PERCENT;

    // 2D fit options
    constexpr bool SAVE_2D_A    = Constants::FIT_2D_SAVE_A;
    constexpr bool SAVE_2D_MUX  = Constants::FIT_2D_SAVE_MUX;
    constexpr bool SAVE_2D_MUY  = Constants::FIT_2D_SAVE_MUY;
    constexpr bool SAVE_2D_SIGX = Constants::FIT_2D_SAVE_SIGX;
    constexpr bool SAVE_2D_SIGY = Constants::FIT_2D_SAVE_SIGY;
    constexpr bool SAVE_2D_B    = Constants::FIT_2D_SAVE_B;

    // 2D distance model
    constexpr bool   DIST_2D_ENABLED       = Constants::FIT_2D_DIST_ENABLED;
    constexpr double DIST_2D_SCALE_PIXELS  = Constants::FIT_2D_DIST_SCALE_PIXELS;
    constexpr double DIST_2D_EXPONENT      = Constants::FIT_2D_DIST_EXPONENT;
    constexpr double DIST_2D_FLOOR_PERCENT = Constants::FIT_2D_DIST_FLOOR_PERCENT;
    constexpr double DIST_2D_CAP_PERCENT   = Constants::FIT_2D_DIST_CAP_PERCENT;
}

// ============================================================================
// Internal utilities
// ============================================================================
namespace detail {

inline bool IsFinite(double v) { return std::isfinite(v); }

// Helper to get TParameter<double> from tree's UserInfo
inline double GetDoubleMetadata(TTree* tree, const char* key) {
    if (tree) {
        TList* info = tree->GetUserInfo();
        if (info) {
            if (auto* param = dynamic_cast<TParameter<double>*>(info->FindObject(key))) {
                return param->GetVal();
            }
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

inline int GetIntMetadata(TTree* tree, const char* key) {
    if (tree) {
        TList* info = tree->GetUserInfo();
        if (info) {
            if (auto* param = dynamic_cast<TParameter<int>*>(info->FindObject(key))) {
                return param->GetVal();
            }
        }
    }
    return -1;
}

// 1D Gaussian with constant offset: A * exp(-0.5*((x-mu)/sigma)^2) + B
double GaussPlusB(double* x, double* p) {
    const double A     = p[0];
    const double mu    = p[1];
    const double sigma = p[2];
    const double B     = p[3];
    const double dx    = (x[0] - mu) / sigma;
    return A * std::exp(-0.5 * dx * dx) + B;
}

// 2D Gaussian with constant offset:
// A * exp(-0.5 * ((x-mux)^2/sigx^2 + (y-muy)^2/sigy^2)) + B
double Gauss2DPlusB(double* xy, double* p) {
    const double A   = p[0];
    const double mux = p[1];
    const double muy = p[2];
    const double sx  = p[3];
    const double sy  = p[4];
    const double B   = p[5];
    const double dx  = (xy[0] - mux) / sx;
    const double dy  = (xy[1] - muy) / sy;
    return A * std::exp(-0.5 * (dx*dx + dy*dy)) + B;
}

int InferRadiusFromTree(TTree* tree, const std::string& preferredBranch) {
    if (!tree) return -1;
    std::vector<double>* charges = nullptr;
    auto bind = [&](const char* branch) -> bool {
        if (!branch || tree->GetBranch(branch) == nullptr) return false;
        tree->SetBranchStatus(branch, 1);
        tree->SetBranchAddress(branch, &charges);
        return true;
    };
    bool bound = false;
    if (!preferredBranch.empty() && bind(preferredBranch.c_str())) {
        bound = true;
    } else if (bind("Qf")) {
        bound = true;
    } else if (bind("Fi")) {
        bound = true;
    } else if (bind("Qi")) {
        bound = true;
    }
    if (!bound) return -1;

    const Long64_t nEntries = std::min<Long64_t>(tree->GetEntries(), 50000);
    int inferredRadius = -1;
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (!charges || charges->empty()) continue;
        const int total = static_cast<int>(charges->size());
        const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
        if (N >= 3 && N * N == total) {
            inferredRadius = (N - 1) / 2;
            break;
        }
    }
    tree->ResetBranchAddresses();
    return inferredRadius;
}

std::string ResolveChargeBranch(TTree* tree, const std::string& requestedBranch) {
    auto hasBranch = [&](const std::string& name) {
        return !name.empty() && tree->GetBranch(name.c_str()) != nullptr;
    };
    if (hasBranch(requestedBranch)) return requestedBranch;

    for (const char* name : {"Qf", "QfBlock", "QfRow", "QfCol",
                             "Fi", "FiBlock", "FiRow", "FiCol",
                             "Qi", "QiBlock", "QiRow", "QiCol"}) {
        if (hasBranch(name)) return name;
    }
    return {};
}

// Branch helper to ensure branch exists and reset if necessary
TBranch* EnsureAndResetBranch(TTree* tree, const char* name, double* addr) {
    TBranch* br = tree->GetBranch(name);
    if (!br) {
        br = tree->Branch(name, addr);
    } else {
        tree->SetBranchAddress(name, addr);
        br = tree->GetBranch(name);
        if (br) {
            br->Reset();
            br->DropBaskets();
        }
    }
    return br;
}

struct GaussFitResult {
    bool converged = false;
    double A = std::numeric_limits<double>::quiet_NaN();
    double mu = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    double B = std::numeric_limits<double>::quiet_NaN();
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    double ndf = std::numeric_limits<double>::quiet_NaN();
    double prob = std::numeric_limits<double>::quiet_NaN();
};

struct GaussFitConfig {
    double muLo;
    double muHi;
    double sigmaLo;
    double sigmaHi;
    double qmax;
    double pixelSpacing;
    double seedA;
    double seedMu;
    double seedSigma;
    double seedB;
};

inline double ComputeUniformSigma(double errorPercentOfMax, double qmax) {
    return charge_uncert::UniformPercentOfMax(errorPercentOfMax, qmax);
}

inline std::pair<double, bool> WeightedCentroid(const std::vector<double>& positions,
                                                const std::vector<double>& charges,
                                                double baseline) {
    if (positions.size() != charges.size() || positions.empty()) {
        return {std::numeric_limits<double>::quiet_NaN(), false};
    }
    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const double weight = std::max(0.0, charges[i] - baseline);
        if (weight <= 0.0) continue;
        weightTotal += weight;
        weightedSum += weight * positions[i];
    }
    if (weightTotal <= 0.0) {
        return {std::numeric_limits<double>::quiet_NaN(), false};
    }
    return {weightedSum / weightTotal, true};
}

inline double SeedSigma(const std::vector<double>& positions,
                        const std::vector<double>& charges,
                        double baseline,
                        double pixelSpacing,
                        double sigmaLo,
                        double sigmaHi) {
    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const double weight = std::max(0.0, charges[i] - baseline);
        if (weight <= 0.0) continue;
        weightTotal += weight;
        weightedSum += weight * positions[i];
    }
    double sigma = std::numeric_limits<double>::quiet_NaN();
    if (weightTotal > 0.0) {
        const double mean = weightedSum / weightTotal;
        double variance = 0.0;
        for (size_t i = 0; i < positions.size(); ++i) {
            const double weight = std::max(0.0, charges[i] - baseline);
            if (weight <= 0.0) continue;
            const double dx = positions[i] - mean;
            variance += weight * dx * dx;
        }
        variance = (weightTotal > 0.0) ? (variance / weightTotal) : 0.0;
        sigma = std::sqrt(std::max(variance, 1e-12));
    }
    if (!IsFinite(sigma) || sigma <= 0.0) {
        sigma = std::max(0.25 * pixelSpacing, 1e-6);
    }
    if (sigma < sigmaLo) sigma = sigmaLo;
    if (sigma > sigmaHi) sigma = sigmaHi;
    return sigma;
}

GaussFitResult RunGaussianFit1D(const std::vector<double>& positions,
                                const std::vector<double>& charges,
                                const std::vector<double>* sigmaCandidates,
                                double uniformSigma,
                                const GaussFitConfig& cfg) {
    GaussFitResult result;
    if (positions.size() != charges.size() || positions.size() < 3) {
        return result;
    }

    thread_local TF1 fLoc("fGauss1D_helper", GaussPlusB, -1e9, 1e9, 4);
    auto minmaxPos = std::minmax_element(positions.begin(), positions.end());
    if (minmaxPos.first != positions.end()) {
        const double margin = 0.5 * cfg.pixelSpacing;
        const double rangeLo = (*minmaxPos.first) - margin;
        const double rangeHi = (*minmaxPos.second) + margin;
        fLoc.SetRange(rangeLo, rangeHi);
    }
    const double qmax = IsFinite(cfg.qmax) ? std::max(cfg.qmax, 0.0) : 0.0;
    const double amplitudeMax = std::max(1e-18, 2.0 * qmax);
    const double baselineMax = std::max(1e-18, qmax);
    fLoc.SetParameters(cfg.seedA, cfg.seedMu, cfg.seedSigma, cfg.seedB);
    fLoc.SetParLimits(0, 1e-18, amplitudeMax);
    fLoc.SetParLimits(1, cfg.muLo, cfg.muHi);
    fLoc.SetParLimits(2, cfg.sigmaLo, cfg.sigmaHi);
    fLoc.SetParLimits(3, -baselineMax, baselineMax);

    ROOT::Math::WrappedMultiTF1 wrapped(fLoc, 1);
    ROOT::Fit::BinData data(static_cast<int>(positions.size()), 1);
    const bool hasSigmaCandidates = sigmaCandidates && !sigmaCandidates->empty();
    for (size_t i = 0; i < positions.size(); ++i) {
        double candidate = std::numeric_limits<double>::quiet_NaN();
        if (hasSigmaCandidates && i < sigmaCandidates->size()) {
            candidate = (*sigmaCandidates)[i];
        }
        const double sigmaY = charge_uncert::SelectVerticalSigma(candidate, uniformSigma);
        data.Add(positions[i], charges[i], sigmaY);
    }

    thread_local ROOT::Fit::Fitter fitter;
    fitter.Config().SetMinimizer("Minuit2", "Fumili2");
    fitter.Config().MinimizerOptions().SetStrategy(0);
    fitter.Config().MinimizerOptions().SetTolerance(1e-4);
    fitter.Config().MinimizerOptions().SetPrintLevel(0);
    fitter.SetFunction(wrapped);
    fitter.Config().ParSettings(0).SetLimits(1e-18, amplitudeMax);
    fitter.Config().ParSettings(1).SetLimits(cfg.muLo, cfg.muHi);
    fitter.Config().ParSettings(2).SetLimits(cfg.sigmaLo, cfg.sigmaHi);
    fitter.Config().ParSettings(3).SetLimits(-baselineMax, baselineMax);
    const double stepA = std::max(1e-18, 0.01 * cfg.seedA);
    const double stepB = std::max(1e-18, 0.01 * std::max(std::abs(cfg.seedB), cfg.seedA));
    fitter.Config().ParSettings(0).SetStepSize(stepA);
    fitter.Config().ParSettings(1).SetStepSize(1e-4 * cfg.pixelSpacing);
    fitter.Config().ParSettings(2).SetStepSize(1e-4 * cfg.pixelSpacing);
    fitter.Config().ParSettings(3).SetStepSize(stepB);
    fitter.Config().ParSettings(0).SetValue(cfg.seedA);
    fitter.Config().ParSettings(1).SetValue(cfg.seedMu);
    fitter.Config().ParSettings(2).SetValue(cfg.seedSigma);
    fitter.Config().ParSettings(3).SetValue(cfg.seedB);

    bool ok = fitter.Fit(data);
    if (!ok) {
        fitter.Config().SetMinimizer("Minuit2", "Migrad");
        fitter.Config().MinimizerOptions().SetStrategy(1);
        fitter.Config().MinimizerOptions().SetTolerance(1e-3);
        fitter.Config().MinimizerOptions().SetPrintLevel(0);
        ok = fitter.Fit(data);
    }
    if (!ok) return result;

    result.converged = true;
    result.A = fitter.Result().Parameter(0);
    result.mu = fitter.Result().Parameter(1);
    result.sigma = fitter.Result().Parameter(2);
    result.B = fitter.Result().Parameter(3);
    result.chi2 = fitter.Result().Chi2();
    result.ndf = fitter.Result().Ndf();
    result.prob = (fitter.Result().Ndf() > 0)
                      ? TMath::Prob(fitter.Result().Chi2(), fitter.Result().Ndf())
                      : std::numeric_limits<double>::quiet_NaN();
    return result;
}

struct FlatVectorStore {
    void Initialize(size_t nEntries, size_t reservePerEntry = 0) {
        offsets.assign(nEntries, -1);
        sizes.assign(nEntries, 0);
        values.clear();
        if (reservePerEntry > 0) {
            values.reserve(nEntries * reservePerEntry);
        }
    }

    void Store(size_t index, const std::vector<double>& src) {
        if (src.empty()) return;
        offsets[index] = static_cast<int>(values.size());
        sizes[index] = static_cast<int>(src.size());
        values.insert(values.end(), src.begin(), src.end());
    }

    [[nodiscard]] std::span<const double> Get(size_t index) const {
        const int offset = offsets[index];
        const int size = sizes[index];
        if (offset < 0 || size <= 0) return {};
        return {values.data() + offset, static_cast<size_t>(size)};
    }

    std::vector<double> values;
    std::vector<int> offsets;
    std::vector<int> sizes;
};

} // namespace detail

// ============================================================================
// FitGaussian1D Implementation
// ============================================================================
int FitGaussian1D(const char* filename) {
    using namespace detail;

    if (!ROOT::IsImplicitMTEnabled()) {
        ROOT::EnableImplicitMT();
    }

    ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
    ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-3);
    ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(100);
    ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
    ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

    const double errorPercentOfMax = Config::ERROR_PERCENT_OF_MAX;
    const bool saveParamA = Config::SAVE_1D_A;
    const bool saveParamMu = Config::SAVE_1D_MU;
    const bool saveParamSigma = Config::SAVE_1D_SIGMA;
    const bool saveParamB = Config::SAVE_1D_B;

    auto fileHandle = std::unique_ptr<TFile>(TFile::Open(filename, "UPDATE"));
    if (!fileHandle || fileHandle->IsZombie()) {
        ::Error("FitGaussian1D", "Cannot open file: %s", filename);
        return 1;
    }

    TTree* tree = dynamic_cast<TTree*>(fileHandle->Get("Hits"));
    if (!tree) {
        ::Error("FitGaussian1D", "Hits tree not found in file: %s", filename);
        fileHandle->Close();
        return 3;
    }

    const std::string requestedBranch = Config::CHARGE_BRANCH_1D;
    const std::string chosenCharge = ResolveChargeBranch(tree, requestedBranch);
    if (chosenCharge.empty()) {
        ::Error("FitGaussian1D", "No charge branch found");
        fileHandle->Close();
        return 4;
    }

    double pixelSpacing = GetDoubleMetadata(tree, "GridPixelSpacing_mm");
    double pixelSize = GetDoubleMetadata(tree, "GridPixelSize_mm");
    int neighborhoodRadiusMeta = GetIntMetadata(tree, "NeighborhoodRadius");

    if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
        ::Error("FitGaussian1D", "Pixel spacing not available. Aborting.");
        fileHandle->Close();
        return 2;
    }
    if (!IsFinite(pixelSize) || pixelSize <= 0) {
        pixelSize = 0.5 * pixelSpacing;
    }
    if (neighborhoodRadiusMeta <= 0) {
        neighborhoodRadiusMeta = InferRadiusFromTree(tree, chosenCharge);
    }

    // Setup input branches
    double x_hit = 0.0, y_hit = 0.0;
    double x_px = 0.0, y_px = 0.0;
    Bool_t is_pixel_hit = kFALSE;
    std::vector<double>* Q = nullptr;

    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("TrueX", 1);
    tree->SetBranchStatus("TrueY", 1);
    tree->SetBranchStatus("PixelX", 1);
    tree->SetBranchStatus("PixelY", 1);
    tree->SetBranchStatus("isPixelHit", 1);
    tree->SetBranchStatus(chosenCharge.c_str(), 1);

    tree->SetBranchAddress("TrueX", &x_hit);
    tree->SetBranchAddress("TrueY", &y_hit);
    tree->SetBranchAddress("PixelX", &x_px);
    tree->SetBranchAddress("PixelY", &y_px);
    tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
    tree->SetBranchAddress(chosenCharge.c_str(), &Q);

    // Output values
    const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
    double ReconRowX = INVALID_VALUE, ReconColY = INVALID_VALUE;
    double ReconTrueDeltaRowX = INVALID_VALUE, ReconTrueDeltaColY = INVALID_VALUE;
    double GaussRowA = INVALID_VALUE, GaussRowMu = INVALID_VALUE;
    double GaussRowSigma = INVALID_VALUE, GaussRowB = INVALID_VALUE;
    double GaussRowChi2 = INVALID_VALUE, GaussRowNdf = INVALID_VALUE, GaussRowProb = INVALID_VALUE;
    double GaussColA = INVALID_VALUE, GaussColMu = INVALID_VALUE;
    double GaussColSigma = INVALID_VALUE, GaussColB = INVALID_VALUE;
    double GaussColChi2 = INVALID_VALUE, GaussColNdf = INVALID_VALUE, GaussColProb = INVALID_VALUE;

    // Setup output branches
    TBranch* br_x_rec = EnsureAndResetBranch(tree, "ReconRowX", &ReconRowX);
    TBranch* br_y_rec = EnsureAndResetBranch(tree, "ReconColY", &ReconColY);
    TBranch* br_dx_signed = EnsureAndResetBranch(tree, "ReconTrueDeltaRowX", &ReconTrueDeltaRowX);
    TBranch* br_dy_signed = EnsureAndResetBranch(tree, "ReconTrueDeltaColY", &ReconTrueDeltaColY);

    TBranch* br_row_A = saveParamA ? EnsureAndResetBranch(tree, "GaussRowA", &GaussRowA) : nullptr;
    TBranch* br_row_mu = saveParamMu ? EnsureAndResetBranch(tree, "GaussRowMu", &GaussRowMu) : nullptr;
    TBranch* br_row_sigma = saveParamSigma ? EnsureAndResetBranch(tree, "GaussRowSigma", &GaussRowSigma) : nullptr;
    TBranch* br_row_B = saveParamB ? EnsureAndResetBranch(tree, "GaussRowB", &GaussRowB) : nullptr;
    TBranch* br_row_chi2 = EnsureAndResetBranch(tree, "GaussRowChi2", &GaussRowChi2);
    TBranch* br_row_ndf = EnsureAndResetBranch(tree, "GaussRowNdf", &GaussRowNdf);
    TBranch* br_row_prob = EnsureAndResetBranch(tree, "GaussRowProb", &GaussRowProb);

    TBranch* br_col_A = saveParamA ? EnsureAndResetBranch(tree, "GaussColA", &GaussColA) : nullptr;
    TBranch* br_col_mu = saveParamMu ? EnsureAndResetBranch(tree, "GaussColMu", &GaussColMu) : nullptr;
    TBranch* br_col_sigma = saveParamSigma ? EnsureAndResetBranch(tree, "GaussColSigma", &GaussColSigma) : nullptr;
    TBranch* br_col_B = saveParamB ? EnsureAndResetBranch(tree, "GaussColB", &GaussColB) : nullptr;
    TBranch* br_col_chi2 = EnsureAndResetBranch(tree, "GaussColChi2", &GaussColChi2);
    TBranch* br_col_ndf = EnsureAndResetBranch(tree, "GaussColNdf", &GaussColNdf);
    TBranch* br_col_prob = EnsureAndResetBranch(tree, "GaussColProb", &GaussColProb);

    const Long64_t nEntries = tree->GetEntries();
    Long64_t nProcessed = 0;
    std::atomic<long long> nFitted{0};

    // Preload input data
    std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
    std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
    std::vector<char> v_is_pixel(nEntries);
    std::vector<int> v_gridDim(nEntries, 0);

    FlatVectorStore chargeStore;
    const int approxNeighborSide = (neighborhoodRadiusMeta > 0) ? (2 * neighborhoodRadiusMeta + 1) : 5;
    chargeStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSide * approxNeighborSide);

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        v_x_hit[i] = x_hit;
        v_y_hit[i] = y_hit;
        v_x_px[i] = x_px;
        v_y_px[i] = y_px;
        v_is_pixel[i] = is_pixel_hit ? 1 : 0;
        if (is_pixel_hit || !Q || Q->empty()) continue;

        chargeStore.Store(static_cast<size_t>(i), *Q);
        const int total = static_cast<int>(Q->size());
        const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
        if (N >= 3 && N * N == total) {
            v_gridDim[i] = N;
        }
    }

    // Prepare output buffers
    std::vector<double> out_x_rec(nEntries, INVALID_VALUE);
    std::vector<double> out_y_rec(nEntries, INVALID_VALUE);
    std::vector<double> out_dx_s(nEntries, INVALID_VALUE);
    std::vector<double> out_dy_s(nEntries, INVALID_VALUE);
    std::vector<double> out_row_A(nEntries, INVALID_VALUE);
    std::vector<double> out_row_mu(nEntries, INVALID_VALUE);
    std::vector<double> out_row_sigma(nEntries, INVALID_VALUE);
    std::vector<double> out_row_B(nEntries, INVALID_VALUE);
    std::vector<double> out_row_chi2(nEntries, INVALID_VALUE);
    std::vector<double> out_row_ndf(nEntries, INVALID_VALUE);
    std::vector<double> out_row_prob(nEntries, INVALID_VALUE);
    std::vector<double> out_col_A(nEntries, INVALID_VALUE);
    std::vector<double> out_col_mu(nEntries, INVALID_VALUE);
    std::vector<double> out_col_sigma(nEntries, INVALID_VALUE);
    std::vector<double> out_col_B(nEntries, INVALID_VALUE);
    std::vector<double> out_col_chi2(nEntries, INVALID_VALUE);
    std::vector<double> out_col_ndf(nEntries, INVALID_VALUE);
    std::vector<double> out_col_prob(nEntries, INVALID_VALUE);

    // Parallel fitting
    std::vector<int> indices(nEntries);
    std::iota(indices.begin(), indices.end(), 0);
    ROOT::TThreadExecutor exec;

    const int prevErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;

    exec.Foreach([&](int i) {
        if (v_is_pixel[i] != 0) return;
        const auto QLoc = chargeStore.Get(static_cast<size_t>(i));
        if (QLoc.empty()) return;
        const int N = v_gridDim[i];
        if (N < 3 || static_cast<size_t>(N * N) != QLoc.size()) return;

        const int R = (N - 1) / 2;
        const double x_px_loc = v_x_px[i];
        const double y_px_loc = v_y_px[i];

        thread_local std::vector<double> x_row, q_row, y_col, q_col;
        x_row.clear(); q_row.clear();
        y_col.clear(); q_col.clear();

        double qmaxNeighborhood = -1e300;
        for (int di = -R; di <= R; ++di) {
            for (int dj = -R; dj <= R; ++dj) {
                const int idx = (di + R) * N + (dj + R);
                const double q = QLoc[idx];
                if (!IsFinite(q) || q < 0) continue;
                if (q > qmaxNeighborhood) qmaxNeighborhood = q;
                if (dj == 0) {
                    x_row.push_back(x_px_loc + di * pixelSpacing);
                    q_row.push_back(q);
                }
                if (di == 0) {
                    y_col.push_back(y_px_loc + dj * pixelSpacing);
                    q_col.push_back(q);
                }
            }
        }

        if (x_row.size() < 3 || y_col.size() < 3) return;

        const double uniformSigma = ComputeUniformSigma(errorPercentOfMax, qmaxNeighborhood);

        auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
        auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
        double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
        double B0_row = *minmaxRow.first;
        double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
        double B0_col = *minmaxCol.first;

        const auto [rowCentroid, rowCentroidOk] = WeightedCentroid(x_row, q_row, B0_row);
        const auto [colCentroid, colCentroidOk] = WeightedCentroid(y_col, q_col, B0_col);

        int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
        int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
        double mu0_row = x_row[idxMaxRow];
        double mu0_col = y_col[idxMaxCol];

        const double sigLoBound = pixelSize;
        const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
        const double sigInitRow = SeedSigma(x_row, q_row, B0_row, pixelSpacing, sigLoBound, sigHiBound);
        const double sigInitCol = SeedSigma(y_col, q_col, B0_col, pixelSpacing, sigLoBound, sigHiBound);

        const double muXLo = x_px_loc - 1.0 * pixelSpacing;
        const double muXHi = x_px_loc + 1.0 * pixelSpacing;
        const double muYLo = y_px_loc - 1.0 * pixelSpacing;
        const double muYHi = y_px_loc + 1.0 * pixelSpacing;

        GaussFitConfig rowConfig{muXLo, muXHi, sigLoBound, sigHiBound,
                                 qmaxNeighborhood, pixelSpacing, A0_row,
                                 mu0_row, sigInitRow, B0_row};
        GaussFitConfig colConfig{muYLo, muYHi, sigLoBound, sigHiBound,
                                 qmaxNeighborhood, pixelSpacing, A0_col,
                                 mu0_col, sigInitCol, B0_col};

        GaussFitResult rowFit = RunGaussianFit1D(x_row, q_row, nullptr, uniformSigma, rowConfig);
        GaussFitResult colFit = RunGaussianFit1D(y_col, q_col, nullptr, uniformSigma, colConfig);

        if (rowFit.converged) {
            out_row_A[i] = rowFit.A;
            out_row_mu[i] = rowFit.mu;
            out_row_sigma[i] = rowFit.sigma;
            out_row_B[i] = rowFit.B;
            out_row_chi2[i] = rowFit.chi2;
            out_row_ndf[i] = rowFit.ndf;
            out_row_prob[i] = rowFit.prob;
        }
        if (colFit.converged) {
            out_col_A[i] = colFit.A;
            out_col_mu[i] = colFit.mu;
            out_col_sigma[i] = colFit.sigma;
            out_col_B[i] = colFit.B;
            out_col_chi2[i] = colFit.chi2;
            out_col_ndf[i] = colFit.ndf;
            out_col_prob[i] = colFit.prob;
        }

        double muX = rowFit.converged ? rowFit.mu : (rowCentroidOk ? rowCentroid : INVALID_VALUE);
        double muY = colFit.converged ? colFit.mu : (colCentroidOk ? colCentroid : INVALID_VALUE);

        if (IsFinite(muX) && IsFinite(muY)) {
            out_x_rec[i] = muX;
            out_y_rec[i] = muY;
            out_dx_s[i] = v_x_hit[i] - muX;
            out_dy_s[i] = v_y_hit[i] - muY;
            nFitted.fetch_add(1, std::memory_order_relaxed);
        }
    }, indices);

    gErrorIgnoreLevel = prevErrorLevel;

    // Write outputs
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->LoadTree(i);
        ReconRowX = out_x_rec[i];
        ReconColY = out_y_rec[i];
        ReconTrueDeltaRowX = out_dx_s[i];
        ReconTrueDeltaColY = out_dy_s[i];
        GaussRowA = out_row_A[i];
        GaussRowMu = out_row_mu[i];
        GaussRowSigma = out_row_sigma[i];
        GaussRowB = out_row_B[i];
        GaussRowChi2 = out_row_chi2[i];
        GaussRowNdf = out_row_ndf[i];
        GaussRowProb = out_row_prob[i];
        GaussColA = out_col_A[i];
        GaussColMu = out_col_mu[i];
        GaussColSigma = out_col_sigma[i];
        GaussColB = out_col_B[i];
        GaussColChi2 = out_col_chi2[i];
        GaussColNdf = out_col_ndf[i];
        GaussColProb = out_col_prob[i];

        br_x_rec->Fill();
        br_y_rec->Fill();
        br_dx_signed->Fill();
        br_dy_signed->Fill();
        if (br_row_A) br_row_A->Fill();
        if (br_row_mu) br_row_mu->Fill();
        if (br_row_sigma) br_row_sigma->Fill();
        if (br_row_B) br_row_B->Fill();
        br_row_chi2->Fill();
        br_row_ndf->Fill();
        br_row_prob->Fill();
        if (br_col_A) br_col_A->Fill();
        if (br_col_mu) br_col_mu->Fill();
        if (br_col_sigma) br_col_sigma->Fill();
        if (br_col_B) br_col_B->Fill();
        br_col_chi2->Fill();
        br_col_ndf->Fill();
        br_col_prob->Fill();
        nProcessed++;
    }

    tree->SetBranchStatus("*", 1);
    fileHandle->cd();
    tree->Write("", TObject::kOverwrite);
    fileHandle->Flush();
    fileHandle->Close();

    ::Info("FitGaussian1D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
    return 0;
}

// ============================================================================
// FitGaussian2D Implementation
// ============================================================================
int FitGaussian2D(const char* filename) {
    using namespace detail;

    ROOT::EnableImplicitMT();
    ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
    ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);
    ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(400);
    ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
    ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

    const bool verticalErrorsEnabled = Config::USE_VERTICAL_UNCERTAINTIES;
    const double errorPercentOfMax = Config::ERROR_PERCENT_OF_MAX;
    const bool saveParamA = Config::SAVE_2D_A;
    const bool saveParamMux = Config::SAVE_2D_MUX;
    const bool saveParamMuy = Config::SAVE_2D_MUY;
    const bool saveParamSigx = Config::SAVE_2D_SIGX;
    const bool saveParamSigy = Config::SAVE_2D_SIGY;
    const bool saveParamB = Config::SAVE_2D_B;

    auto file = std::unique_ptr<TFile>(TFile::Open(filename, "UPDATE"));
    if (!file || file->IsZombie()) {
        ::Error("FitGaussian2D", "Cannot open file: %s", filename);
        return 1;
    }

    TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
    if (!tree) {
        ::Error("FitGaussian2D", "Hits tree not found in file: %s", filename);
        file->Close();
        return 3;
    }

    double pixelSpacing = GetDoubleMetadata(tree, "GridPixelSpacing_mm");
    double pixelSize = GetDoubleMetadata(tree, "GridPixelSize_mm");
    int neighborhoodRadiusMeta = GetIntMetadata(tree, "NeighborhoodRadius");

    if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
        ::Error("FitGaussian2D", "Pixel spacing not available. Aborting.");
        file->Close();
        return 2;
    }
    if (!IsFinite(pixelSize) || pixelSize <= 0) {
        pixelSize = 0.5 * pixelSpacing;
    }

    const std::string chosenCharge = ResolveChargeBranch(tree, Config::CHARGE_BRANCH_2D);
    if (chosenCharge.empty()) {
        ::Error("FitGaussian2D", "No charge branch found.");
        file->Close();
        return 4;
    }
    if (neighborhoodRadiusMeta <= 0) {
        neighborhoodRadiusMeta = InferRadiusFromTree(tree, chosenCharge);
    }

    double x_hit = 0.0, y_hit = 0.0;
    double x_px = 0.0, y_px = 0.0;
    Bool_t is_pixel_hit = kFALSE;
    std::vector<double>* Q = nullptr;

    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("TrueX", 1);
    tree->SetBranchStatus("TrueY", 1);
    tree->SetBranchStatus("PixelX", 1);
    tree->SetBranchStatus("PixelY", 1);
    tree->SetBranchStatus("isPixelHit", 1);
    tree->SetBranchStatus(chosenCharge.c_str(), 1);

    tree->SetBranchAddress("TrueX", &x_hit);
    tree->SetBranchAddress("TrueY", &y_hit);
    tree->SetBranchAddress("PixelX", &x_px);
    tree->SetBranchAddress("PixelY", &y_px);
    tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
    tree->SetBranchAddress(chosenCharge.c_str(), &Q);

    const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
    double x_rec_3d = INVALID_VALUE, y_rec_3d = INVALID_VALUE;
    double rec_hit_delta_x_3d_signed = INVALID_VALUE, rec_hit_delta_y_3d_signed = INVALID_VALUE;
    double gauss3d_A = INVALID_VALUE, gauss3d_mux = INVALID_VALUE, gauss3d_muy = INVALID_VALUE;
    double gauss3d_sigx = INVALID_VALUE, gauss3d_sigy = INVALID_VALUE, gauss3d_B = INVALID_VALUE;
    double gauss3d_chi2 = INVALID_VALUE, gauss3d_ndf = INVALID_VALUE, gauss3d_prob = INVALID_VALUE;

    TBranch* br_x_rec = EnsureAndResetBranch(tree, "ReconX_2D", &x_rec_3d);
    TBranch* br_y_rec = EnsureAndResetBranch(tree, "ReconY_2D", &y_rec_3d);
    TBranch* br_dx_signed = EnsureAndResetBranch(tree, "ReconTrueDeltaX_2D", &rec_hit_delta_x_3d_signed);
    TBranch* br_dy_signed = EnsureAndResetBranch(tree, "ReconTrueDeltaY_2D", &rec_hit_delta_y_3d_signed);
    TBranch* br_A = saveParamA ? EnsureAndResetBranch(tree, "Gauss2D_A", &gauss3d_A) : nullptr;
    TBranch* br_mux = saveParamMux ? EnsureAndResetBranch(tree, "Gauss2D_mux", &gauss3d_mux) : nullptr;
    TBranch* br_muy = saveParamMuy ? EnsureAndResetBranch(tree, "Gauss2D_muy", &gauss3d_muy) : nullptr;
    TBranch* br_sigx = saveParamSigx ? EnsureAndResetBranch(tree, "Gauss2D_sigx", &gauss3d_sigx) : nullptr;
    TBranch* br_sigy = saveParamSigy ? EnsureAndResetBranch(tree, "Gauss2D_sigy", &gauss3d_sigy) : nullptr;
    TBranch* br_B = saveParamB ? EnsureAndResetBranch(tree, "Gauss2D_B", &gauss3d_B) : nullptr;
    TBranch* br_chi2 = EnsureAndResetBranch(tree, "Gauss2D_Chi2", &gauss3d_chi2);
    TBranch* br_ndf = EnsureAndResetBranch(tree, "Gauss2D_Ndf", &gauss3d_ndf);
    TBranch* br_prob = EnsureAndResetBranch(tree, "Gauss2D_Prob", &gauss3d_prob);

    const Long64_t nEntries = tree->GetEntries();
    Long64_t nProcessed = 0;
    std::atomic<long long> nFitted{0};

    // Preload inputs
    std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
    std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
    std::vector<char> v_is_pixel(nEntries);

    FlatVectorStore chargeStore;
    const int approxNeighborSide = (neighborhoodRadiusMeta > 0) ? (2 * neighborhoodRadiusMeta + 1) : 5;
    chargeStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSide * approxNeighborSide);

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        v_x_hit[i] = x_hit;
        v_y_hit[i] = y_hit;
        v_x_px[i] = x_px;
        v_y_px[i] = y_px;
        v_is_pixel[i] = is_pixel_hit ? 1 : 0;
        if (Q && !Q->empty()) chargeStore.Store(static_cast<size_t>(i), *Q);
    }

    // Output buffers
    std::vector<double> out_x_rec(nEntries, INVALID_VALUE);
    std::vector<double> out_y_rec(nEntries, INVALID_VALUE);
    std::vector<double> out_dx_s(nEntries, INVALID_VALUE);
    std::vector<double> out_dy_s(nEntries, INVALID_VALUE);
    std::vector<double> out_A(nEntries, INVALID_VALUE);
    std::vector<double> out_mux(nEntries, INVALID_VALUE);
    std::vector<double> out_muy(nEntries, INVALID_VALUE);
    std::vector<double> out_sigx(nEntries, INVALID_VALUE);
    std::vector<double> out_sigy(nEntries, INVALID_VALUE);
    std::vector<double> out_B(nEntries, INVALID_VALUE);
    std::vector<double> out_chi2(nEntries, INVALID_VALUE);
    std::vector<double> out_ndf(nEntries, INVALID_VALUE);
    std::vector<double> out_prob(nEntries, INVALID_VALUE);

    std::vector<int> indices(nEntries);
    std::iota(indices.begin(), indices.end(), 0);
    ROOT::TThreadExecutor exec;
    const int prevErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;

    exec.Foreach([&](int i) {
        if (v_is_pixel[i] != 0) return;
        const auto QLoc = chargeStore.Get(static_cast<size_t>(i));
        if (QLoc.empty()) return;

        const size_t total = QLoc.size();
        const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
        if (N * N != static_cast<int>(total) || N < 3) return;
        const int R = (N - 1) / 2;

        TGraph2D g2d;
        int p = 0;
        double qmaxNeighborhood = -1e300;
        const double x_px_loc = v_x_px[i];
        const double y_px_loc = v_y_px[i];

        for (int di = -R; di <= R; ++di) {
            for (int dj = -R; dj <= R; ++dj) {
                const int idx = (di + R) * N + (dj + R);
                const double q = QLoc[idx];
                if (!IsFinite(q) || q < 0) continue;
                const double x = x_px_loc + di * pixelSpacing;
                const double y = y_px_loc + dj * pixelSpacing;
                g2d.SetPoint(p++, x, y, q);
                if (q > qmaxNeighborhood) qmaxNeighborhood = q;
            }
        }
        if (g2d.GetN() < 5) return;

        double zmin = 1e300, zmax = -1e300;
        int idxMax = 0;
        for (int k = 0; k < g2d.GetN(); ++k) {
            const double z = g2d.GetZ()[k];
            if (z < zmin) zmin = z;
            if (z > zmax) { zmax = z; idxMax = k; }
        }
        double A0 = std::max(1e-18, zmax - zmin);
        double B0 = zmin;
        double mux0 = g2d.GetX()[idxMax];
        double muy0 = g2d.GetY()[idxMax];

        const double uniformSigma = verticalErrorsEnabled
            ? charge_uncert::UniformPercentOfMax(errorPercentOfMax, qmaxNeighborhood)
            : 1.0;
        const double sigLoBound = pixelSize;
        const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);

        // Sigma seeds from weighted variance
        auto sigmaSeed2D = [&](bool forX) -> double {
            double wsum = 0.0, m = 0.0;
            const int n = g2d.GetN();
            if (n <= 0) return std::clamp(std::max(0.25 * pixelSpacing, 1e-6), sigLoBound, sigHiBound);
            for (int k = 0; k < n; ++k) {
                const double w = std::max(0.0, g2d.GetZ()[k] - B0);
                const double c = forX ? g2d.GetX()[k] : g2d.GetY()[k];
                wsum += w;
                m += w * c;
            }
            if (wsum <= 0.0) return std::clamp(std::max(0.25 * pixelSpacing, 1e-6), sigLoBound, sigHiBound);
            m /= wsum;
            double var = 0.0;
            for (int k = 0; k < n; ++k) {
                const double w = std::max(0.0, g2d.GetZ()[k] - B0);
                const double c = forX ? g2d.GetX()[k] : g2d.GetY()[k];
                const double d = c - m;
                var += w * d * d;
            }
            var = (wsum > 0.0) ? (var / wsum) : 0.0;
            return std::clamp(std::sqrt(std::max(var, 1e-12)), sigLoBound, sigHiBound);
        };
        double sxInit = sigmaSeed2D(true);
        double syInit = sigmaSeed2D(false);

        std::vector<double> Xf, Yf, Zf;
        Xf.reserve(g2d.GetN());
        Yf.reserve(g2d.GetN());
        Zf.reserve(g2d.GetN());
        for (int k = 0; k < g2d.GetN(); ++k) {
            Xf.push_back(g2d.GetX()[k]);
            Yf.push_back(g2d.GetY()[k]);
            Zf.push_back(g2d.GetZ()[k]);
        }

        double xMinR = 1e300, xMaxR = -1e300, yMinR = 1e300, yMaxR = -1e300;
        for (int k = 0; k < g2d.GetN(); ++k) {
            xMinR = std::min(xMinR, g2d.GetX()[k]);
            xMaxR = std::max(xMaxR, g2d.GetX()[k]);
            yMinR = std::min(yMinR, g2d.GetY()[k]);
            yMaxR = std::max(yMaxR, g2d.GetY()[k]);
        }
        xMinR -= 0.5 * pixelSpacing;
        xMaxR += 0.5 * pixelSpacing;
        yMinR -= 0.5 * pixelSpacing;
        yMaxR += 0.5 * pixelSpacing;
        const double muXLo = v_x_px[i] - 1.0 * pixelSpacing;
        const double muXHi = v_x_px[i] + 1.0 * pixelSpacing;
        const double muYLo = v_y_px[i] - 1.0 * pixelSpacing;
        const double muYHi = v_y_px[i] + 1.0 * pixelSpacing;

        thread_local TF2 fModel("fModel2D", Gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);
        fModel.SetRange(xMinR, yMinR, xMaxR, yMaxR);
        const double AHi = std::max(1e-18, 2.0 * qmaxNeighborhood);
        const double BHi = std::max(1e-18, qmaxNeighborhood);

        ROOT::Math::WrappedMultiTF1 wModel(fModel, 2);
        const int nPts = static_cast<int>(Xf.size());
        ROOT::Fit::BinData data2D(nPts, 2);
        for (int k = 0; k < nPts; ++k) {
            double xy[2] = {Xf[k], Yf[k]};
            data2D.Add(xy, Zf[k], uniformSigma);
        }

        ROOT::Fit::Fitter fitter;
        fitter.Config().SetMinimizer("Minuit2", "Fumili2");
        fitter.Config().MinimizerOptions().SetStrategy(0);
        fitter.Config().MinimizerOptions().SetTolerance(1e-4);
        fitter.Config().MinimizerOptions().SetPrintLevel(0);
        fitter.SetFunction(wModel);
        fitter.Config().ParSettings(0).SetLimits(1e-18, AHi);
        fitter.Config().ParSettings(1).SetLimits(muXLo, muXHi);
        fitter.Config().ParSettings(2).SetLimits(muYLo, muYHi);
        fitter.Config().ParSettings(3).SetLimits(sigLoBound, sigHiBound);
        fitter.Config().ParSettings(4).SetLimits(sigLoBound, sigHiBound);
        fitter.Config().ParSettings(5).SetLimits(-BHi, BHi);
        fitter.Config().ParSettings(0).SetValue(A0);
        fitter.Config().ParSettings(1).SetValue(mux0);
        fitter.Config().ParSettings(2).SetValue(muy0);
        fitter.Config().ParSettings(3).SetValue(sxInit);
        fitter.Config().ParSettings(4).SetValue(syInit);
        fitter.Config().ParSettings(5).SetValue(B0);

        bool okFit = fitter.Fit(data2D);
        if (!okFit) {
            fitter.Config().SetMinimizer("Minuit2", "Migrad");
            fitter.Config().MinimizerOptions().SetStrategy(1);
            fitter.Config().MinimizerOptions().SetTolerance(1e-3);
            okFit = fitter.Fit(data2D);
        }

        if (okFit) {
            const auto& fitRes = fitter.Result();
            out_A[i] = fitRes.Parameter(0);
            out_mux[i] = fitRes.Parameter(1);
            out_muy[i] = fitRes.Parameter(2);
            out_sigx[i] = fitRes.Parameter(3);
            out_sigy[i] = fitRes.Parameter(4);
            out_B[i] = fitRes.Parameter(5);

            double chi2Calc = 0.0;
            double params[6];
            for (int ip = 0; ip < 6; ++ip) params[ip] = fitRes.Parameter(ip);
            for (int k = 0; k < nPts; ++k) {
                double xyVals[2] = {Xf[k], Yf[k]};
                const double model = Gauss2DPlusB(xyVals, params);
                const double pull = (Zf[k] - model) / uniformSigma;
                chi2Calc += pull * pull;
            }
            int nFree = fitRes.NFreeParameters();
            if (nFree <= 0) nFree = fitRes.NPar();
            int ndfCalc = nPts - nFree;
            if (ndfCalc < 0) ndfCalc = 0;

            out_chi2[i] = chi2Calc;
            out_ndf[i] = ndfCalc > 0 ? static_cast<double>(ndfCalc) : INVALID_VALUE;
            out_prob[i] = ndfCalc > 0 ? TMath::Prob(chi2Calc, ndfCalc) : INVALID_VALUE;

            out_x_rec[i] = out_mux[i];
            out_y_rec[i] = out_muy[i];
            out_dx_s[i] = v_x_hit[i] - out_mux[i];
            out_dy_s[i] = v_y_hit[i] - out_muy[i];
            nFitted.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Fallback to centroid
            double wsum = 0.0, xw = 0.0, yw = 0.0;
            for (int k = 0; k < nPts; ++k) {
                double w = std::max(0.0, Zf[k] - B0);
                wsum += w;
                xw += w * Xf[k];
                yw += w * Yf[k];
            }
            if (wsum > 0) {
                out_x_rec[i] = xw / wsum;
                out_y_rec[i] = yw / wsum;
                out_dx_s[i] = v_x_hit[i] - out_x_rec[i];
                out_dy_s[i] = v_y_hit[i] - out_y_rec[i];
                out_A[i] = A0;
                out_mux[i] = out_x_rec[i];
                out_muy[i] = out_y_rec[i];
                out_sigx[i] = sxInit;
                out_sigy[i] = syInit;
                out_B[i] = B0;
                nFitted.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }, indices);

    gErrorIgnoreLevel = prevErrorLevel;

    // Write outputs
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        x_rec_3d = out_x_rec[i];
        y_rec_3d = out_y_rec[i];
        rec_hit_delta_x_3d_signed = out_dx_s[i];
        rec_hit_delta_y_3d_signed = out_dy_s[i];
        gauss3d_A = out_A[i];
        gauss3d_mux = out_mux[i];
        gauss3d_muy = out_muy[i];
        gauss3d_sigx = out_sigx[i];
        gauss3d_sigy = out_sigy[i];
        gauss3d_B = out_B[i];
        gauss3d_chi2 = out_chi2[i];
        gauss3d_ndf = out_ndf[i];
        gauss3d_prob = out_prob[i];

        br_x_rec->Fill();
        br_y_rec->Fill();
        br_dx_signed->Fill();
        br_dy_signed->Fill();
        if (br_A) br_A->Fill();
        if (br_mux) br_mux->Fill();
        if (br_muy) br_muy->Fill();
        if (br_sigx) br_sigx->Fill();
        if (br_sigy) br_sigy->Fill();
        if (br_B) br_B->Fill();
        br_chi2->Fill();
        br_ndf->Fill();
        br_prob->Fill();
        nProcessed++;
    }

    tree->SetBranchStatus("*", 1);
    file->cd();
    tree->Write("", TObject::kOverwrite);
    file->Flush();
    file->Close();

    ::Info("FitGaussian2D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
    return 0;
}

// ============================================================================
// RunAllFits
// ============================================================================
bool RunAllFits(const std::string& filename) {
    bool executed = false;
    if (Constants::FIT_GAUS_1D) {
        FitGaussian1D(filename.c_str());
        executed = true;
    }
    if (Constants::FIT_GAUS_2D) {
        FitGaussian2D(filename.c_str());
        executed = true;
    }
    return executed;
}

} // namespace ECS::Fit
