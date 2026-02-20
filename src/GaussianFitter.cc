/// @file GaussianFitter.cc
/// @brief Compiled Gaussian fitting routines for position reconstruction.
///
/// This file orchestrates parallel Gaussian fitting over TTree entries,
/// delegating the actual fitting algorithms to core/GaussianFit.hh.
/// This follows the same delegation pattern as ChargeSharingCalculator.cc
/// delegating to core/ChargeSharingCore.hh.
///
/// Configuration is read from Config.hh at compile time for better performance.

#include "GaussianFitter.hh"
#include "Config.hh"
#include "GaussianFit.hh"

#include <Math/MinimizerOptions.h>
#include <ROOT/TThreadExecutor.hxx>
#include <TBranch.h>
#include <TError.h>
#include <TFile.h>
#include <TList.h>
#include <TMath.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TROOT.h>
#include <TTree.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace ECS::Fit {

namespace cfit = epic::chargesharing::fit;

// ============================================================================
// Configuration from Config.hh (compile-time constants)
// ============================================================================
namespace Config {
// Uncertainty model
constexpr double ERROR_PERCENT_OF_MAX = Constants::FIT_ERROR_PERCENT_OF_MAX;
constexpr bool USE_VERTICAL_UNCERTAINTIES = Constants::FIT_USE_VERTICAL_UNCERTAINTIES;

// Charge branches
constexpr const char* CHARGE_BRANCH_1D = Constants::FIT_CHARGE_BRANCH_1D;
constexpr const char* CHARGE_BRANCH_2D = Constants::FIT_CHARGE_BRANCH_2D;

// 1D fit options
constexpr bool SAVE_1D_A = Constants::FIT_1D_SAVE_A;
constexpr bool SAVE_1D_MU = Constants::FIT_1D_SAVE_MU;
constexpr bool SAVE_1D_SIGMA = Constants::FIT_1D_SAVE_SIGMA;
constexpr bool SAVE_1D_B = Constants::FIT_1D_SAVE_B;

// 2D fit options
constexpr bool SAVE_2D_A = Constants::FIT_2D_SAVE_A;
constexpr bool SAVE_2D_MUX = Constants::FIT_2D_SAVE_MUX;
constexpr bool SAVE_2D_MUY = Constants::FIT_2D_SAVE_MUY;
constexpr bool SAVE_2D_SIGX = Constants::FIT_2D_SAVE_SIGX;
constexpr bool SAVE_2D_SIGY = Constants::FIT_2D_SAVE_SIGY;
constexpr bool SAVE_2D_B = Constants::FIT_2D_SAVE_B;
} // namespace Config

// ============================================================================
// Internal utilities (I/O and parallelism helpers only)
// ============================================================================
namespace detail {

static inline bool IsFinite(double v) {
    return std::isfinite(v);
}

// Helper to get TParameter<double> from tree's UserInfo
static inline double GetDoubleMetadata(TTree* tree, const char* key) {
    if (tree) {
        const TList* info = tree->GetUserInfo();
        if (info) {
            if (auto* param = dynamic_cast<TParameter<double>*>(info->FindObject(key))) {
                return param->GetVal();
            }
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static inline int GetIntMetadata(TTree* tree, const char* key) {
    if (tree) {
        const TList* info = tree->GetUserInfo();
        if (info) {
            if (auto* param = dynamic_cast<TParameter<int>*>(info->FindObject(key))) {
                return param->GetVal();
            }
        }
    }
    return -1;
}

static int InferRadiusFromTree(TTree* tree, const std::string& preferredBranch) {
    if (!tree)
        return -1;
    std::vector<double>* charges = nullptr;
    auto bind = [&](const char* branch) -> bool {
        if (!branch || tree->GetBranch(branch) == nullptr)
            return false;
        tree->SetBranchStatus(branch, true);
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
    if (!bound)
        return -1;

    const Long64_t nEntries = std::min<Long64_t>(tree->GetEntries(), 50000);
    int inferredRadius = -1;
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (!charges || charges->empty())
            continue;
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

static std::string ResolveChargeBranch(TTree* tree, const std::string& requestedBranch) {
    auto hasBranch = [&](const std::string& name) {
        return !name.empty() && tree->GetBranch(name.c_str()) != nullptr;
    };
    if (hasBranch(requestedBranch))
        return requestedBranch;

    for (const char* name :
         {"Qf", "QfBlock", "QfRow", "QfCol", "Fi", "FiBlock", "FiRow", "FiCol", "Qi", "QiBlock", "QiRow", "QiCol"}) {
        if (hasBranch(name))
            return name;
    }
    return {};
}

// Branch helper to ensure branch exists and reset if necessary
static TBranch* EnsureAndResetBranch(TTree* tree, const char* name, double* addr) {
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
        if (src.empty())
            return;
        offsets[index] = static_cast<int>(values.size());
        sizes[index] = static_cast<int>(src.size());
        values.insert(values.end(), src.begin(), src.end());
    }

    [[nodiscard]] std::span<const double> Get(size_t index) const {
        const int offset = offsets[index];
        const int size = sizes[index];
        if (offset < 0 || size <= 0)
            return {};
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

    auto* tree = dynamic_cast<TTree*>(fileHandle->Get("Hits"));
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
    double x_hit = 0.0;
    double y_hit = 0.0;
    double x_px = 0.0;
    double y_px = 0.0;
    Bool_t is_pixel_hit = kFALSE;
    std::vector<double> const* Q = nullptr;

    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("TrueX", true);
    tree->SetBranchStatus("TrueY", true);
    tree->SetBranchStatus("PixelX", true);
    tree->SetBranchStatus("PixelY", true);
    tree->SetBranchStatus("isPixelHit", true);
    tree->SetBranchStatus(chosenCharge.c_str(), true);

    tree->SetBranchAddress("TrueX", &x_hit);
    tree->SetBranchAddress("TrueY", &y_hit);
    tree->SetBranchAddress("PixelX", &x_px);
    tree->SetBranchAddress("PixelY", &y_px);
    tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
    tree->SetBranchAddress(chosenCharge.c_str(), &Q);

    // Output values
    const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
    double ReconRowX = INVALID_VALUE;
    double ReconColY = INVALID_VALUE;
    double ReconTrueDeltaRowX = INVALID_VALUE;
    double ReconTrueDeltaColY = INVALID_VALUE;
    double GaussRowA = INVALID_VALUE;
    double GaussRowMu = INVALID_VALUE;
    double GaussRowSigma = INVALID_VALUE;
    double GaussRowB = INVALID_VALUE;
    double GaussRowChi2 = INVALID_VALUE;
    double GaussRowNdf = INVALID_VALUE;
    double GaussRowProb = INVALID_VALUE;
    double GaussColA = INVALID_VALUE;
    double GaussColMu = INVALID_VALUE;
    double GaussColSigma = INVALID_VALUE;
    double GaussColB = INVALID_VALUE;
    double GaussColChi2 = INVALID_VALUE;
    double GaussColNdf = INVALID_VALUE;
    double GaussColProb = INVALID_VALUE;

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
    std::vector<double> v_x_hit(nEntries);
    std::vector<double> v_y_hit(nEntries);
    std::vector<double> v_x_px(nEntries);
    std::vector<double> v_y_px(nEntries);
    std::vector<char> v_is_pixel(nEntries);
    std::vector<int> v_gridDim(nEntries, 0);

    FlatVectorStore chargeStore;
    const int approxNeighborSide = (neighborhoodRadiusMeta > 0) ? ((2 * neighborhoodRadiusMeta) + 1) : 5;
    chargeStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSide * approxNeighborSide);

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        v_x_hit[i] = x_hit;
        v_y_hit[i] = y_hit;
        v_x_px[i] = x_px;
        v_y_px[i] = y_px;
        v_is_pixel[i] = is_pixel_hit ? 1 : 0;
        if (is_pixel_hit || !Q || Q->empty())
            continue;

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

    // Parallel fitting (delegates to core/GaussianFit.hh)
    std::vector<int> indices(nEntries);
    std::iota(indices.begin(), indices.end(), 0);
    ROOT::TThreadExecutor exec;

    const int prevErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;

    exec.Foreach(
        [&](int i) {
            if (v_is_pixel[i] != 0)
                return;
            const auto QLoc = chargeStore.Get(static_cast<size_t>(i));
            if (QLoc.empty())
                return;
            const int N = v_gridDim[i];
            if (N < 3 || static_cast<size_t>(N * N) != QLoc.size())
                return;

            const int R = (N - 1) / 2;
            const double x_px_loc = v_x_px[i];
            const double y_px_loc = v_y_px[i];

            // Build row/col slices from charge grid
            thread_local std::vector<double> x_row;
            thread_local std::vector<double> q_row;
            thread_local std::vector<double> y_col;
            thread_local std::vector<double> q_col;
            x_row.clear();
            q_row.clear();
            y_col.clear();
            q_col.clear();

            double qmaxNeighborhood = -1e300;
            for (int di = -R; di <= R; ++di) {
                for (int dj = -R; dj <= R; ++dj) {
                    const int idx = ((di + R) * N) + (dj + R);
                    const double q = QLoc[idx];
                    if (!IsFinite(q) || q < 0)
                        continue;
                    qmaxNeighborhood = std::max(q, qmaxNeighborhood);
                    if (dj == 0) {
                        x_row.push_back(x_px_loc + (di * pixelSpacing));
                        q_row.push_back(q);
                    }
                    if (di == 0) {
                        y_col.push_back(y_px_loc + (dj * pixelSpacing));
                        q_col.push_back(q);
                    }
                }
            }

            if (x_row.size() < 3 || y_col.size() < 3)
                return;

            // Centroid fallback (using core utility)
            const double qMinRow = *std::ranges::min_element(q_row);
            const double qMinCol = *std::ranges::min_element(q_col);
            const auto [rowCentroid, rowCentroidOk] = cfit::weightedCentroid(x_row, q_row, qMinRow);
            const auto [colCentroid, colCentroidOk] = cfit::weightedCentroid(y_col, q_col, qMinCol);

            // Fit bounds
            const double sigLoBound = pixelSize;
            const double sigHiBound =
                std::max(sigLoBound,
                         static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
            const double muXLo = x_px_loc - (1.0 * pixelSpacing);
            const double muXHi = x_px_loc + (1.0 * pixelSpacing);
            const double muYLo = y_px_loc - (1.0 * pixelSpacing);
            const double muYHi = y_px_loc + (1.0 * pixelSpacing);

            // Delegate fitting to core library
            const cfit::GaussFit1DConfig rowCfg{.muLo = muXLo,
                                                .muHi = muXHi,
                                                .sigmaLo = sigLoBound,
                                                .sigmaHi = sigHiBound,
                                                .qMax = qmaxNeighborhood,
                                                .pixelSpacing = pixelSpacing,
                                                .errorPercent = errorPercentOfMax};
            const cfit::GaussFit1DConfig colCfg{.muLo = muYLo,
                                                .muHi = muYHi,
                                                .sigmaLo = sigLoBound,
                                                .sigmaHi = sigHiBound,
                                                .qMax = qmaxNeighborhood,
                                                .pixelSpacing = pixelSpacing,
                                                .errorPercent = errorPercentOfMax};

            const auto rowFit = cfit::fitGaussian1D(x_row, q_row, rowCfg);
            const auto colFit = cfit::fitGaussian1D(y_col, q_col, colCfg);

            if (rowFit.converged) {
                out_row_A[i] = rowFit.A;
                out_row_mu[i] = rowFit.mu;
                out_row_sigma[i] = rowFit.sigma;
                out_row_B[i] = rowFit.B;
                out_row_chi2[i] = rowFit.chi2;
                out_row_ndf[i] = rowFit.ndf;
                out_row_prob[i] = (rowFit.ndf > 0) ? TMath::Prob(rowFit.chi2, static_cast<int>(rowFit.ndf))
                                                    : INVALID_VALUE;
            }
            if (colFit.converged) {
                out_col_A[i] = colFit.A;
                out_col_mu[i] = colFit.mu;
                out_col_sigma[i] = colFit.sigma;
                out_col_B[i] = colFit.B;
                out_col_chi2[i] = colFit.chi2;
                out_col_ndf[i] = colFit.ndf;
                out_col_prob[i] = (colFit.ndf > 0) ? TMath::Prob(colFit.chi2, static_cast<int>(colFit.ndf))
                                                    : INVALID_VALUE;
            }

            double const muX = rowFit.converged ? rowFit.mu : (rowCentroidOk ? rowCentroid : INVALID_VALUE);
            double const muY = colFit.converged ? colFit.mu : (colCentroidOk ? colCentroid : INVALID_VALUE);

            if (IsFinite(muX) && IsFinite(muY)) {
                out_x_rec[i] = muX;
                out_y_rec[i] = muY;
                out_dx_s[i] = v_x_hit[i] - muX;
                out_dy_s[i] = v_y_hit[i] - muY;
                nFitted.fetch_add(1, std::memory_order_relaxed);
            }
        },
        indices);

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
        if (br_row_A)
            br_row_A->Fill();
        if (br_row_mu)
            br_row_mu->Fill();
        if (br_row_sigma)
            br_row_sigma->Fill();
        if (br_row_B)
            br_row_B->Fill();
        br_row_chi2->Fill();
        br_row_ndf->Fill();
        br_row_prob->Fill();
        if (br_col_A)
            br_col_A->Fill();
        if (br_col_mu)
            br_col_mu->Fill();
        if (br_col_sigma)
            br_col_sigma->Fill();
        if (br_col_B)
            br_col_B->Fill();
        br_col_chi2->Fill();
        br_col_ndf->Fill();
        br_col_prob->Fill();
        nProcessed++;
    }

    tree->SetBranchStatus("*", true);
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

    auto* tree = dynamic_cast<TTree*>(file->Get("Hits"));
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

    double x_hit = 0.0;
    double y_hit = 0.0;
    double x_px = 0.0;
    double y_px = 0.0;
    Bool_t is_pixel_hit = kFALSE;
    std::vector<double> const* Q = nullptr;

    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("TrueX", true);
    tree->SetBranchStatus("TrueY", true);
    tree->SetBranchStatus("PixelX", true);
    tree->SetBranchStatus("PixelY", true);
    tree->SetBranchStatus("isPixelHit", true);
    tree->SetBranchStatus(chosenCharge.c_str(), true);

    tree->SetBranchAddress("TrueX", &x_hit);
    tree->SetBranchAddress("TrueY", &y_hit);
    tree->SetBranchAddress("PixelX", &x_px);
    tree->SetBranchAddress("PixelY", &y_px);
    tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
    tree->SetBranchAddress(chosenCharge.c_str(), &Q);

    const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
    double x_rec_3d = INVALID_VALUE;
    double y_rec_3d = INVALID_VALUE;
    double rec_hit_delta_x_3d_signed = INVALID_VALUE;
    double rec_hit_delta_y_3d_signed = INVALID_VALUE;
    double gauss3d_A = INVALID_VALUE;
    double gauss3d_mux = INVALID_VALUE;
    double gauss3d_muy = INVALID_VALUE;
    double gauss3d_sigx = INVALID_VALUE;
    double gauss3d_sigy = INVALID_VALUE;
    double gauss3d_B = INVALID_VALUE;
    double gauss3d_chi2 = INVALID_VALUE;
    double gauss3d_ndf = INVALID_VALUE;
    double gauss3d_prob = INVALID_VALUE;

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
    std::vector<double> v_x_hit(nEntries);
    std::vector<double> v_y_hit(nEntries);
    std::vector<double> v_x_px(nEntries);
    std::vector<double> v_y_px(nEntries);
    std::vector<char> v_is_pixel(nEntries);

    FlatVectorStore chargeStore;
    const int approxNeighborSide = (neighborhoodRadiusMeta > 0) ? ((2 * neighborhoodRadiusMeta) + 1) : 5;
    chargeStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSide * approxNeighborSide);

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        v_x_hit[i] = x_hit;
        v_y_hit[i] = y_hit;
        v_x_px[i] = x_px;
        v_y_px[i] = y_px;
        v_is_pixel[i] = is_pixel_hit ? 1 : 0;
        if (Q && !Q->empty())
            chargeStore.Store(static_cast<size_t>(i), *Q);
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

    exec.Foreach(
        [&](int i) {
            if (v_is_pixel[i] != 0)
                return;
            const auto QLoc = chargeStore.Get(static_cast<size_t>(i));
            if (QLoc.empty())
                return;

            const size_t total = QLoc.size();
            const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
            if (N * N != static_cast<int>(total) || N < 3)
                return;
            const int R = (N - 1) / 2;

            // Collect points directly into vectors (no TGraph2D intermediate)
            thread_local std::vector<double> Xf;
            thread_local std::vector<double> Yf;
            thread_local std::vector<double> Zf;
            Xf.clear();
            Yf.clear();
            Zf.clear();

            double qmaxNeighborhood = -1e300;
            const double x_px_loc = v_x_px[i];
            const double y_px_loc = v_y_px[i];

            for (int di = -R; di <= R; ++di) {
                for (int dj = -R; dj <= R; ++dj) {
                    const int idx = ((di + R) * N) + (dj + R);
                    const double q = QLoc[idx];
                    if (!IsFinite(q) || q < 0)
                        continue;
                    Xf.push_back(x_px_loc + (di * pixelSpacing));
                    Yf.push_back(y_px_loc + (dj * pixelSpacing));
                    Zf.push_back(q);
                    qmaxNeighborhood = std::max(q, qmaxNeighborhood);
                }
            }
            if (Zf.size() < 5)
                return;

            const double sigLoBound = pixelSize;
            const double sigHiBound =
                std::max(sigLoBound,
                         static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
            const double muXLo = v_x_px[i] - (1.0 * pixelSpacing);
            const double muXHi = v_x_px[i] + (1.0 * pixelSpacing);
            const double muYLo = v_y_px[i] - (1.0 * pixelSpacing);
            const double muYHi = v_y_px[i] + (1.0 * pixelSpacing);

            // Map error model to core config.
            // When verticalErrorsEnabled=false, standalone used uniformSigma=1.0;
            // replicate by setting errorPercent such that qMax * errorPercent / 100 = 1.0.
            const double effectiveErrorPercent =
                verticalErrorsEnabled ? errorPercentOfMax
                                      : (qmaxNeighborhood > 0 ? 100.0 / qmaxNeighborhood : errorPercentOfMax);

            // Delegate fitting to core library
            const cfit::GaussFit2DConfig cfg2D{.muXLo = muXLo,
                                               .muXHi = muXHi,
                                               .muYLo = muYLo,
                                               .muYHi = muYHi,
                                               .sigmaLo = sigLoBound,
                                               .sigmaHi = sigHiBound,
                                               .qMax = qmaxNeighborhood,
                                               .pixelSpacing = pixelSpacing,
                                               .errorPercent = effectiveErrorPercent};

            const auto fitResult = cfit::fitGaussian2D(Xf, Yf, Zf, cfg2D);

            if (fitResult.converged) {
                out_A[i] = fitResult.A;
                out_mux[i] = fitResult.muX;
                out_muy[i] = fitResult.muY;
                out_sigx[i] = fitResult.sigmaX;
                out_sigy[i] = fitResult.sigmaY;
                out_B[i] = fitResult.B;
                out_chi2[i] = fitResult.chi2;
                out_ndf[i] = (fitResult.ndf > 0) ? fitResult.ndf : INVALID_VALUE;
                out_prob[i] = (fitResult.ndf > 0)
                                  ? TMath::Prob(fitResult.chi2, static_cast<int>(fitResult.ndf))
                                  : INVALID_VALUE;

                out_x_rec[i] = fitResult.muX;
                out_y_rec[i] = fitResult.muY;
                out_dx_s[i] = v_x_hit[i] - fitResult.muX;
                out_dy_s[i] = v_y_hit[i] - fitResult.muY;
                nFitted.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Centroid fallback
                const double B0 = *std::ranges::min_element(Zf);
                double wsum = 0.0;
                double xw = 0.0;
                double yw = 0.0;
                for (size_t k = 0; k < Zf.size(); ++k) {
                    double const w = std::max(0.0, Zf[k] - B0);
                    wsum += w;
                    xw += w * Xf[k];
                    yw += w * Yf[k];
                }
                if (wsum > 0) {
                    out_x_rec[i] = xw / wsum;
                    out_y_rec[i] = yw / wsum;
                    out_dx_s[i] = v_x_hit[i] - out_x_rec[i];
                    out_dy_s[i] = v_y_hit[i] - out_y_rec[i];
                    out_A[i] = std::max(1e-18, *std::ranges::max_element(Zf) - B0);
                    out_mux[i] = out_x_rec[i];
                    out_muy[i] = out_y_rec[i];
                    out_sigx[i] = cfit::estimateSigma(Xf, Zf, B0, pixelSpacing, sigLoBound, sigHiBound);
                    out_sigy[i] = cfit::estimateSigma(Yf, Zf, B0, pixelSpacing, sigLoBound, sigHiBound);
                    out_B[i] = B0;
                    nFitted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        },
        indices);

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
        if (br_A)
            br_A->Fill();
        if (br_mux)
            br_mux->Fill();
        if (br_muy)
            br_muy->Fill();
        if (br_sigx)
            br_sigx->Fill();
        if (br_sigy)
            br_sigy->Fill();
        if (br_B)
            br_B->Fill();
        br_chi2->Fill();
        br_ndf->Fill();
        br_prob->Fill();
        nProcessed++;
    }

    tree->SetBranchStatus("*", true);
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
