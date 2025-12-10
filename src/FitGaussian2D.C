// ROOT macro: FitGaussian2D.C
// Performs 2D Gaussian fit on the full charge neighborhood to reconstruct
// deltas, and appends them as new branches.
// Uses Q_f (noisy charge per pixel, Coulombs) when available; falls back to Q_i.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TGraph2D.h>
#include <TGraph2DErrors.h>
#include <TF2.h>
#include <TROOT.h>
#include <TError.h>
#include <TMath.h>
#include <Math/MinimizerOptions.h>
// Minuit2 least-squares API
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

// Fast least-squares fitter API (required for Fumili2)
#include <Fit/Fitter.h>
#include <Fit/BinData.h>
#include <Fit/Chi2FCN.h>
#include <Math/WrappedMultiTF1.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <atomic>
#include <memory>
#include <ROOT/TThreadExecutor.hxx>
#include <ROOT/RNTupleImporter.hxx>

#include "ChargeUtils.h"

// ============================================================================
// Branch utilities (inline to avoid Geant4 dependency from RootHelpers.hh)
// ============================================================================
namespace rootutils {

struct BranchInfo {
    const char* name;
    double* value;
    bool enabled;
    TBranch** handle;
    const char* leaflist = nullptr;
};

inline TBranch* EnsureAndResetBranch(TTree* tree, const BranchInfo& info) {
    if (!tree || !info.name || !info.value || !info.handle) {
        return nullptr;
    }

    TBranch* branch = tree->GetBranch(info.name);
    if (!branch) {
        // Branch doesn't exist - create it
        branch = info.leaflist ? tree->Branch(info.name, info.value, info.leaflist)
                               : tree->Branch(info.name, info.value);
    } else {
        // Branch exists - set address and clear for overwrite
        tree->SetBranchAddress(info.name, info.value);
        branch = tree->GetBranch(info.name);
        if (branch) {
            branch->Reset();        // Clear previous entries
            branch->DropBaskets();  // Drop old baskets to avoid mixing
        }
    }
    tree->SetBranchStatus(info.name, 1);
    return branch;
}

inline void RegisterBranches(TTree* tree, std::vector<BranchInfo>& branches) {
    if (!tree) return;
    for (auto& info : branches) {
        if (info.enabled && info.handle) {
            *info.handle = EnsureAndResetBranch(tree, info);
        }
    }
}

inline void FillBranches(const std::vector<BranchInfo>& branches) {
    for (const auto& info : branches) {
        if (info.enabled && info.handle && *info.handle) {
            (*info.handle)->Fill();
        }
    }
}

} // namespace rootutils

// ============================================================================
// Helper functions for metadata/radius inference
// ============================================================================
namespace {

int inferRadiusFromTree(TTree* tree, const std::string& preferredBranch) {
    if (!tree) {
        return -1;
    }
    std::vector<double>* charges = nullptr;
    auto bind = [&](const char* branch) -> bool {
        if (!branch || tree->GetBranch(branch) == nullptr) {
            return false;
        }
        tree->SetBranchStatus(branch, 1);
        tree->SetBranchAddress(branch, &charges);
        return true;
    };
    bool bound = false;
    if (!preferredBranch.empty() && bind(preferredBranch.c_str())) {
        bound = true;
    } else if (bind("Q_f")) {
        bound = true;
    } else if (bind("F_i")) {
        bound = true;
    } else if (bind("Q_i")) {
        bound = true;
    }
    if (!bound) {
        return -1;
    }
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

  inline bool IsFinite3D(double v) { return std::isfinite(v); }
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%)
// of the event's maximum charge within the neighborhood. The same error is
// applied to all data points used in the fit for that event.
int FitGaussian2D(const char* filename = "../build/epicChargeSharing.root",
                 double errorPercentOfMax = 5.0,
                 bool saveParamA = true,
                 bool saveParamMux = true,
                 bool saveParamMuy = true,
                 bool saveParamSigx = true,
                 bool saveParamSigy = true,
                 bool saveParamB = true,
                 const char* chargeBranch = "Q_f",
                 bool useQnQiPercentErrors = false,
                 bool useDistanceWeightedErrors = false,
                 double distanceErrorScalePixels = 1.0,
                 double distanceErrorExponent = 1.0,
                 double distanceErrorFloorPercent = 1.0,
                 double distanceErrorCapPercent = 50.0,
                 bool distanceErrorPreferTruthCenter = true,
                 bool useVerticalUncertainties = true) {
  // Enable ROOT Implicit Multithreading for automatic parallelization
  ROOT::EnableImplicitMT();

  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(400);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  const bool verticalErrorsEnabled = useVerticalUncertainties;
  const bool distanceErrorsEnabled = verticalErrorsEnabled && useDistanceWeightedErrors;
  const double distScalePx = (std::isfinite(distanceErrorScalePixels) &&
                              distanceErrorScalePixels > 0.0)
                                 ? distanceErrorScalePixels
                                 : 1.0;
  const double distExponent = std::isfinite(distanceErrorExponent)
                                  ? std::max(0.0, distanceErrorExponent)
                                  : 1.0;
  const double distFloorPct = std::isfinite(distanceErrorFloorPercent)
                                  ? std::max(0.0, distanceErrorFloorPercent)
                                  : 0.0;
  const double distCapPct = std::isfinite(distanceErrorCapPercent)
                                ? std::max(0.0, distanceErrorCapPercent)
                                : 0.0;
  const bool distPreferTruthCenter = distanceErrorPreferTruthCenter;

  if (!verticalErrorsEnabled) {
    if (useDistanceWeightedErrors) {
      ::Warning("FitGaussian2D",
                "Vertical uncertainties disabled; ignoring distance-weighted "
                "uncertainty model.");
    }
    if (useQnQiPercentErrors) {
      ::Warning("FitGaussian2D",
                "Vertical uncertainties disabled; ignoring Q_n/Q_i uncertainty "
                "model.");
    }
  }

  if (distanceErrorsEnabled && useQnQiPercentErrors) {
    ::Warning("FitGaussian2D",
              "Distance-weighted uncertainties requested; ignoring Q_n/Q_i"
              " vertical uncertainty model.");
  }

  // Open file for update
  auto file = std::unique_ptr<TFile>(TFile::Open(filename, "UPDATE"));
  if (!file || file->IsZombie()) {
    ::Error("FitGaussian2D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first for clearer diagnostics)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("FitGaussian2D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)", filename);
    file->Close();
    return 3;
  }

  // Helper to get TParameter<double> from tree's UserInfo
  auto getDoubleParam = [&tree](const char* name) -> double {
    TList* info = tree->GetUserInfo();
    if (!info) return NAN;
    if (auto* param = dynamic_cast<TParameter<double>*>(info->FindObject(name))) {
      return param->GetVal();
    }
    return NAN;
  };

  // Helper to get TParameter<int> from tree's UserInfo
  auto getIntParam = [&tree](const char* name) -> int {
    TList* info = tree->GetUserInfo();
    if (!info) return -1;
    if (auto* param = dynamic_cast<TParameter<int>*>(info->FindObject(name))) {
      return param->GetVal();
    }
    return -1;
  };

  // Fetch metadata from tree's UserInfo
  double pixelSpacing = getDoubleParam("GridPixelSpacing_mm");
  double pixelSize = getDoubleParam("GridPixelSize_mm");
  int neighborhoodRadiusMeta = getIntParam("NeighborhoodRadius");

  if (!IsFinite3D(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("FitGaussian2D", "Pixel spacing not available in tree metadata. Aborting.");
    file->Close();
    return 2;
  }
  if (!IsFinite3D(pixelSize) || pixelSize <= 0) {
    ::Error("FitGaussian2D", "Pixel size not available in tree metadata. Aborting.");
    file->Close();
    return 2;
  }
  if (neighborhoodRadiusMeta <= 0) {
    ::Error("FitGaussian2D", "Neighborhood radius not available in tree metadata. Aborting.");
    file->Close();
    return 2;
  }
  
  // Decide which charge branch to use (check multiple naming conventions)
  std::string chosenCharge = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("");
  auto hasBranch = [&](const char* b){ return tree->GetBranch(b) != nullptr; };
  if (chosenCharge.empty() || !hasBranch(chosenCharge.c_str())) {
    // Try standard names first, then Block/Row/Col suffix variants
    for (const char* name : {"Qf", "QfBlock", "QfRow", "QfCol",
                             "Q_f", "Fi", "FiBlock", "FiRow", "FiCol",
                             "F_i", "Qi", "QiBlock", "QiRow", "QiCol", "Q_i"}) {
      if (hasBranch(name)) {
        chosenCharge = name;
        break;
      }
    }
    if (chosenCharge.empty()) {
      ::Error("FitGaussian2D", "No charge branch found. Tried Qf/Fi/Qi variants.");
      file->Close();
      return 4;
    }
  }
  if (neighborhoodRadiusMeta <= 0) {
    neighborhoodRadiusMeta = inferRadiusFromTree(tree, chosenCharge);
  }

  // Existing branches (inputs)
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  // Use Q_f (noisy) for fits; fall back to Q_i (Coulombs)
  std::vector<double>* Q = nullptr;
  std::vector<double>* Qi = nullptr;
  std::vector<double>* Qn = nullptr;
  bool enableQiQnErrors = verticalErrorsEnabled && useQnQiPercentErrors && !distanceErrorsEnabled;
  bool haveQiBranchForErrors = false;
  bool haveQnBranchForErrors = false;

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("TrueX", 1);
  tree->SetBranchStatus("TrueY", 1);
  tree->SetBranchStatus("PixelX", 1);
  tree->SetBranchStatus("PixelY", 1);
  tree->SetBranchStatus("isPixelHit", 1);
  // Enable only the chosen charge branch
  tree->SetBranchStatus(chosenCharge.c_str(), 1);
  if (enableQiQnErrors) {
    haveQiBranchForErrors = tree->GetBranch("Q_i") != nullptr;
    haveQnBranchForErrors = tree->GetBranch("Q_n") != nullptr;
    if (haveQiBranchForErrors && haveQnBranchForErrors) {
      tree->SetBranchStatus("Q_i", 1);
      tree->SetBranchStatus("Q_n", 1);
    } else {
      ::Warning("FitGaussian2D", "Requested Q_i/Q_n vertical errors but required branches are missing. Falling back to percent-of-max uncertainty.");
      enableQiQnErrors = false;
    }
  }

  tree->SetBranchAddress("TrueX", &x_hit);
  tree->SetBranchAddress("TrueY", &y_hit);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);
  if (enableQiQnErrors && haveQiBranchForErrors) {
    tree->SetBranchAddress("Q_i", &Qi);
  }
  if (enableQiQnErrors && haveQnBranchForErrors) {
    tree->SetBranchAddress("Q_n", &Qn);
  }

  // New branches (outputs).
  // Use NaN so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double x_rec_3d = INVALID_VALUE;
  double y_rec_3d = INVALID_VALUE;
  double rec_hit_delta_x_3d_signed = INVALID_VALUE;
  double rec_hit_delta_y_3d_signed = INVALID_VALUE;
  // 2D Gaussian fit parameters
  double gauss3d_A = INVALID_VALUE;
  double gauss3d_mux = INVALID_VALUE;
  double gauss3d_muy = INVALID_VALUE;
  double gauss3d_sigx = INVALID_VALUE;
  double gauss3d_sigy = INVALID_VALUE;
  double gauss3d_B = INVALID_VALUE;
  double gauss3d_chi2 = INVALID_VALUE;
  double gauss3d_ndf = INVALID_VALUE;
  double gauss3d_prob = INVALID_VALUE;

  using rootutils::BranchInfo;
  TBranch* br_x_rec = nullptr;
  TBranch* br_y_rec = nullptr;
  TBranch* br_dx_signed = nullptr;
  TBranch* br_dy_signed = nullptr;
  TBranch* br_A = nullptr;
  TBranch* br_mux = nullptr;
  TBranch* br_muy = nullptr;
  TBranch* br_sigx = nullptr;
  TBranch* br_sigy = nullptr;
  TBranch* br_B = nullptr;
  TBranch* br_chi2 = nullptr;
  TBranch* br_ndf = nullptr;
  TBranch* br_prob = nullptr;

  std::vector<BranchInfo> coreBranches{
      {"ReconX_2D", &x_rec_3d, true, &br_x_rec, "ReconX_2D/D"},
      {"ReconY_2D", &y_rec_3d, true, &br_y_rec, "ReconY_2D/D"},
      {"ReconTrueDeltaX_2D", &rec_hit_delta_x_3d_signed, true, &br_dx_signed, "ReconTrueDeltaX_2D/D"},
      {"ReconTrueDeltaY_2D", &rec_hit_delta_y_3d_signed, true, &br_dy_signed, "ReconTrueDeltaY_2D/D"},
  };
  std::vector<BranchInfo> paramBranches{
      {"Gauss2D_A", &gauss3d_A, saveParamA, &br_A, "Gauss2D_A/D"},
      {"Gauss2D_mux", &gauss3d_mux, saveParamMux, &br_mux, "Gauss2D_mux/D"},
      {"Gauss2D_muy", &gauss3d_muy, saveParamMuy, &br_muy, "Gauss2D_muy/D"},
      {"Gauss2D_sigx", &gauss3d_sigx, saveParamSigx, &br_sigx, "Gauss2D_sigx/D"},
      {"Gauss2D_sigy", &gauss3d_sigy, saveParamSigy, &br_sigy, "Gauss2D_sigy/D"},
      {"Gauss2D_B", &gauss3d_B, saveParamB, &br_B, "Gauss2D_B/D"},
      {"Gauss2D_Chi2", &gauss3d_chi2, true, &br_chi2, "Gauss2D_Chi2/D"},
      {"Gauss2D_Ndf", &gauss3d_ndf, true, &br_ndf, "Gauss2D_Ndf/D"},
      {"Gauss2D_Prob", &gauss3d_prob, true, &br_prob, "Gauss2D_Prob/D"},
  };

  rootutils::RegisterBranches(tree, coreBranches);
  rootutils::RegisterBranches(tree, paramBranches);

  // 2D fit function kept for reference. We use Minuit2 on a compact window.
  TF2 f2D("f2D", Gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0;
  std::atomic<long long> nFitted{0};

  // Preload inputs sequentially to avoid ROOT I/O races
  std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
  std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
  std::vector<char> v_is_pixel(nEntries);
  std::vector<std::vector<double>> v_Q(nEntries);
  std::vector<std::vector<double>> v_Qi;
  std::vector<std::vector<double>> v_Qn;
  if (enableQiQnErrors) {
    v_Qi.resize(nEntries);
    v_Qn.resize(nEntries);
  }
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    v_x_hit[i] = x_hit;
    v_y_hit[i] = y_hit;
    v_x_px[i]  = x_px;
    v_y_px[i]  = y_px;
    v_is_pixel[i] = is_pixel_hit ? 1 : 0;
    if (Q && !Q->empty()) v_Q[i] = *Q; else v_Q[i].clear();
    if (enableQiQnErrors) {
      if (Qi && !Qi->empty()) v_Qi[i] = *Qi; else v_Qi[i].clear();
      if (Qn && !Qn->empty()) v_Qn[i] = *Qn; else v_Qn[i].clear();
    }
  }

  // Prepare output buffers
  std::vector<double> out_x_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_y_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_dx_s(nEntries, INVALID_VALUE);
  std::vector<double> out_dy_s(nEntries, INVALID_VALUE);
  // Output buffers for fit parameters
  std::vector<double> out_A(nEntries, INVALID_VALUE);
  std::vector<double> out_mux(nEntries, INVALID_VALUE);
  std::vector<double> out_muy(nEntries, INVALID_VALUE);
  std::vector<double> out_sigx(nEntries, INVALID_VALUE);
  std::vector<double> out_sigy(nEntries, INVALID_VALUE);
  std::vector<double> out_B(nEntries, INVALID_VALUE);
  std::vector<double> out_chi2(nEntries, INVALID_VALUE);
  std::vector<double> out_ndf(nEntries, INVALID_VALUE);
  std::vector<double> out_prob(nEntries, INVALID_VALUE);

  // Parallel computation across entries
  std::vector<int> indices(nEntries);
  std::iota(indices.begin(), indices.end(), 0);
  ROOT::TThreadExecutor exec;
  // Suppress expected Minuit2 error spam during Fumili2 attempts; we'll fallback to MIGRAD if needed
  const int prevErrorLevel_FitGaussian2D = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  exec.Foreach([&, verticalErrorsEnabled](int i){
    const bool isPix = v_is_pixel[i] != 0;
    const auto &QLoc = v_Q[i];
    if (isPix || QLoc.empty()) {
      return;
    }

    static thread_local std::vector<double> dist_err_vals;
    dist_err_vals.clear();

    const size_t total = QLoc.size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      return;
    }
    const int R = (N - 1) / 2;
    const bool haveQiQnForEvent = enableQiQnErrors &&
                                  static_cast<size_t>(i) < v_Qi.size() &&
                                  static_cast<size_t>(i) < v_Qn.size() &&
                                  v_Qi[i].size() == v_Qn[i].size() &&
                                  v_Qi[i].size() == QLoc.size();
    const std::vector<double>* QiLocPtr = haveQiQnForEvent ? &v_Qi[i] : nullptr;
    const std::vector<double>* QnLocPtr = haveQiQnForEvent ? &v_Qn[i] : nullptr;

    TGraph2D g2d;
    int p = 0;
    double qmaxNeighborhood = -1e300;
    double qmaxQiNeighborhood = -1e300;
    const double x_px_loc = v_x_px[i];
    const double y_px_loc = v_y_px[i];
    std::vector<double> err_vals;
    if (haveQiQnForEvent) {
      err_vals.reserve(N * N);
    }
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double q = QLoc[idx];
        if (!IsFinite3D(q) || q < 0) continue;
        const double x = x_px_loc + di * pixelSpacing;
        const double y = y_px_loc + dj * pixelSpacing;
        g2d.SetPoint(p++, x, y, q);
        if (haveQiQnForEvent) {
          const double qiVal = (*QiLocPtr)[idx];
          if (std::isfinite(qiVal) && qiVal > qmaxQiNeighborhood) {
            qmaxQiNeighborhood = qiVal;
          }
          const double errVal = charge_uncert::QnQiScaled(qiVal, (*QnLocPtr)[idx], qmaxQiNeighborhood);
          err_vals.push_back(errVal);
        }
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
      }
    }
    if (g2d.GetN() < 5) {
      return;
    }

    double zmin = 1e300, zmax = -1e300; int idxMax = 0;
    for (int k = 0; k < g2d.GetN(); ++k) {
      const double z = g2d.GetZ()[k];
      if (z < zmin) zmin = z;
      if (z > zmax) { zmax = z; idxMax = k; }
    }
    double A0 = std::max(1e-18, zmax - zmin);
    // Allow negative baseline seed
    double B0 = zmin;
    double mux0 = g2d.GetX()[idxMax];
    double muy0 = g2d.GetY()[idxMax];

    double centerX = mux0;
    double centerY = muy0;
    if (distanceErrorsEnabled) {
      if (distPreferTruthCenter) {
        if (IsFinite3D(v_x_hit[i])) {
          centerX = v_x_hit[i];
        }
        if (IsFinite3D(v_y_hit[i])) {
          centerY = v_y_hit[i];
        }
      }
      if (!IsFinite3D(centerX)) {
        centerX = mux0;
      }
      if (!IsFinite3D(centerX)) {
        centerX = v_x_px[i];
      }
      if (!IsFinite3D(centerY)) {
        centerY = muy0;
      }
      if (!IsFinite3D(centerY)) {
        centerY = v_y_px[i];
      }
    }

    // Error model and sigma bounds
    const double uniformSigma =
        verticalErrorsEnabled
            ? charge_uncert::UniformPercentOfMax(errorPercentOfMax,
                                                 qmaxNeighborhood)
            : 1.0;
    // Constrain sigma to be within [pixel size, radius * pitch]
    const double sigLoBound = pixelSize;
    const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
    auto sigmaSeed2D = [&](bool forX)->double {
      double wsum = 0.0, m = 0.0; const int n = g2d.GetN();
      if (n <= 0) {
        double s = std::max(0.25*pixelSpacing, 1e-6);
        if (s < sigLoBound) s = sigLoBound;
        if (s > sigHiBound) s = sigHiBound;
        return s;
      }
      for (int k=0;k<n;++k) { const double w = std::max(0.0, g2d.GetZ()[k] - B0); const double c = forX ? g2d.GetX()[k] : g2d.GetY()[k]; wsum += w; m += w*c; }
      if (wsum <= 0.0) {
        double s = std::max(0.25*pixelSpacing, 1e-6);
        if (s < sigLoBound) s = sigLoBound;
        if (s > sigHiBound) s = sigHiBound;
        return s;
      }
      m /= wsum; double var = 0.0;
      for (int k=0;k<n;++k) { const double w = std::max(0.0, g2d.GetZ()[k] - B0); const double c = forX ? g2d.GetX()[k] : g2d.GetY()[k]; const double d = c - m; var += w*d*d; }
      var = (wsum > 0.0) ? (var / wsum) : 0.0;
      double s = std::sqrt(std::max(var, 1e-12));
      if (s < sigLoBound) s = sigLoBound; if (s > sigHiBound) s = sigHiBound; return s;
    };
    double sxInitMoment = sigmaSeed2D(true);
    double syInitMoment = sigmaSeed2D(false);

    // Low-contrast: output centroid and seed parameters instead of NaNs
    // Use relative threshold: skip fitting only if amplitude is tiny relative to max charge
    const double lowContrastThreshold = std::max(1e-30, qmaxNeighborhood * 1e-6);
    if (A0 < lowContrastThreshold) {
      double wsum = 0.0, xw = 0.0, yw = 0.0;
      for (int k = 0; k < g2d.GetN(); ++k) {
        const double w = std::max(0.0, g2d.GetZ()[k] - B0);
        wsum += w; xw += w * g2d.GetX()[k]; yw += w * g2d.GetY()[k];
      }
      if (wsum > 0.0) {
        const double xr = xw / wsum;
        const double yr = yw / wsum;
        out_x_rec[i] = xr; out_y_rec[i] = yr;
        out_dx_s[i] = (v_x_hit[i] - xr);
        out_dy_s[i] = (v_y_hit[i] - yr);
        // Save best-effort parameters
        out_A[i]    = A0;
        out_mux[i]  = xr;
        out_muy[i]  = yr;
        out_sigx[i] = sxInitMoment;
        out_sigy[i] = syInitMoment;
        out_B[i]    = B0;
        nFitted.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Fall back to peak location if centroid fails
        out_A[i]    = A0;
        out_mux[i]  = mux0;
        out_muy[i]  = muy0;
        out_sigx[i] = sxInitMoment;
        out_sigy[i] = syInitMoment;
        out_B[i]    = B0;
      }
      return;
    }
    // Note: uniformSigma and sigma seeds were computed above

    // Build arrays
    std::vector<double> Xf, Yf, Zf;
    Xf.reserve(g2d.GetN()); Yf.reserve(g2d.GetN()); Zf.reserve(g2d.GetN());
    const double* Xarr = g2d.GetX();
    const double* Yarr = g2d.GetY();
    const double* Zarr = g2d.GetZ();
    // Use all points directly
    Xf.assign(Xarr, Xarr + g2d.GetN());
    Yf.assign(Yarr, Yarr + g2d.GetN());
    Zf.assign(Zarr, Zarr + g2d.GetN());
    const std::vector<double>* errValsPtr = nullptr;
    if (haveQiQnForEvent && err_vals.size() == static_cast<size_t>(g2d.GetN())) {
      errValsPtr = &err_vals;
    }

    bool distanceErrorApplied = false;
    if (distanceErrorsEnabled && !Xf.empty() && IsFinite3D(pixelSpacing) &&
        pixelSpacing > 0.0 && IsFinite3D(qmaxNeighborhood) &&
        qmaxNeighborhood > 0.0 && IsFinite3D(centerX) && IsFinite3D(centerY)) {
      dist_err_vals.reserve(Xf.size());
      bool anyFiniteSigma = false;
      for (size_t k = 0; k < Xf.size(); ++k) {
        double sigma = charge_uncert::DistancePowerSigma(
            std::hypot(Xf[k] - centerX, Yf[k] - centerY), qmaxNeighborhood,
            pixelSpacing, distScalePx, distExponent, distFloorPct, distCapPct);
        if (std::isfinite(sigma) && sigma > 0.0) {
          anyFiniteSigma = true;
          dist_err_vals.push_back(sigma);
        } else {
          dist_err_vals.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }
      if (anyFiniteSigma) {
        distanceErrorApplied = true;
      } else {
        dist_err_vals.clear();
      }
    }

    // Build range
    double xMinR =  1e300, xMaxR = -1e300; double yMinR =  1e300, yMaxR = -1e300;
    for (int k = 0; k < g2d.GetN(); ++k) { xMinR = std::min(xMinR, g2d.GetX()[k]); xMaxR = std::max(xMaxR, g2d.GetX()[k]); yMinR = std::min(yMinR, g2d.GetY()[k]); yMaxR = std::max(yMaxR, g2d.GetY()[k]); }
    xMinR -= 0.5 * pixelSpacing; xMaxR += 0.5 * pixelSpacing; yMinR -= 0.5 * pixelSpacing; yMaxR += 0.5 * pixelSpacing;
    const double muXLo = v_x_px[i] - 0.5 * pixelSpacing;
    const double muXHi = v_x_px[i] + 0.5 * pixelSpacing;
    const double muYLo = v_y_px[i] - 0.5 * pixelSpacing;
    const double muYHi = v_y_px[i] + 0.5 * pixelSpacing;

    // Thread-local TF2 reuse to avoid repeated allocations per fit
    thread_local TF2 fModel("fModel", Gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);
    fModel.SetRange(xMinR, yMinR, xMaxR, yMaxR);
    // Bounds for Q fits: A in (0, ~2*qmax], B in [-~qmax, ~qmax]
    const double AHi = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
    const double BHi = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
    // Enforce requested bounds on TF2 as well
    fModel.SetParLimits(0, 1e-18, AHi);
    fModel.SetParLimits(1, muXLo, muXHi);          // mux within pixel
    fModel.SetParLimits(2, muYLo, muYHi);          // muy within pixel
    fModel.SetParLimits(3, sigLoBound, sigHiBound);// sigx bounds
    fModel.SetParLimits(4, sigLoBound, sigHiBound);// sigy bounds
    fModel.SetParLimits(5, -BHi, BHi);             // B in [-~qmax, ~qmax]
    ROOT::Math::WrappedMultiTF1 wModel(fModel, 2);
    const int nPts = (int)Xf.size();
    ROOT::Fit::BinData data2D(nPts, 2);
    std::vector<double> sigmaVals;
    sigmaVals.reserve(nPts);
    for (int k = 0; k < nPts; ++k) {
      double candidate = std::numeric_limits<double>::quiet_NaN();
      if (distanceErrorApplied && static_cast<size_t>(k) < dist_err_vals.size()) {
        candidate = dist_err_vals[k];
      } else if (errValsPtr && static_cast<size_t>(k) < errValsPtr->size()) {
        candidate = (*errValsPtr)[k];
      }
      double xy[2] = {Xf[k], Yf[k]};
      const double sigmaUsed = charge_uncert::SelectVerticalSigma(candidate, uniformSigma);
      data2D.Add(xy, Zf[k], sigmaUsed);
      sigmaVals.push_back(sigmaUsed);
    }
    ROOT::Fit::Fitter fitter;
    fitter.Config().SetMinimizer("Minuit2", "Fumili2");
    fitter.Config().MinimizerOptions().SetStrategy(0);
    fitter.Config().MinimizerOptions().SetTolerance(1e-4);
    fitter.Config().MinimizerOptions().SetPrintLevel(0);
    fitter.SetFunction(wModel);
    fitter.Config().ParSettings(0).SetName("A");
    fitter.Config().ParSettings(1).SetName("mux");
    fitter.Config().ParSettings(2).SetName("muy");
    fitter.Config().ParSettings(3).SetName("sigx");
    fitter.Config().ParSettings(4).SetName("sigy");
    fitter.Config().ParSettings(5).SetName("B");
    // A in (0, ~2*qmax]
    fitter.Config().ParSettings(0).SetLimits(1e-18, AHi);
    fitter.Config().ParSettings(1).SetLimits(muXLo, muXHi);
    fitter.Config().ParSettings(2).SetLimits(muYLo, muYHi);
    fitter.Config().ParSettings(3).SetLimits(sigLoBound, sigHiBound);
    fitter.Config().ParSettings(4).SetLimits(sigLoBound, sigHiBound);
    // B in [-~qmax, ~qmax]
    fitter.Config().ParSettings(5).SetLimits(-BHi, BHi);
    const double stepA = std::max(1e-18, 0.01 * A0);
    const double stepB = std::max(1e-18, 0.01 * std::max(std::abs(B0), A0));
    fitter.Config().ParSettings(0).SetStepSize(stepA);
    fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(3).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(4).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(5).SetStepSize(stepB);
    fitter.Config().ParSettings(0).SetValue(A0);
    fitter.Config().ParSettings(1).SetValue(mux0);
    fitter.Config().ParSettings(2).SetValue(muy0);
    fitter.Config().ParSettings(3).SetValue(sxInitMoment);
    fitter.Config().ParSettings(4).SetValue(syInitMoment);
    fitter.Config().ParSettings(5).SetValue(B0);
    bool okFit = fitter.Fit(data2D);
    // If Fumili2 fails, retry once with MIGRAD which is more robust
    if (!okFit) {
      fitter.Config().SetMinimizer("Minuit2", "Migrad");
      fitter.Config().MinimizerOptions().SetStrategy(1);
      fitter.Config().MinimizerOptions().SetTolerance(1e-3);
      fitter.Config().MinimizerOptions().SetPrintLevel(0);
      okFit = fitter.Fit(data2D);
    }
    if (okFit) {
      const ROOT::Fit::FitResult& fitRes = fitter.Result();
      double params[6];
      for (int ip = 0; ip < 6; ++ip) {
        params[ip] = fitRes.Parameter(ip);
      }
      double chi2Calc = 0.0;
      for (int k = 0; k < nPts; ++k) {
        double xyVals[2] = {Xf[k], Yf[k]};
        const double model = Gauss2DPlusB(xyVals, params);
      const double sigma = (k < static_cast<int>(sigmaVals.size()) && sigmaVals[k] > 0.0)
                               ? sigmaVals[k]
                               : charge_uncert::SelectVerticalSigma(std::numeric_limits<double>::quiet_NaN(), uniformSigma);
        const double pull = (Zf[k] - model) / sigma;
        chi2Calc += pull * pull;
      }
      int nFree = fitRes.NFreeParameters();
      if (nFree <= 0) {
        nFree = fitRes.NPar();
      }
      int ndfCalc = nPts - nFree;
      if (ndfCalc < 0) {
        ndfCalc = 0;
      }
      // Save parameters on successful fit
      out_A[i]    = params[0];
      out_mux[i]  = params[1];
      out_muy[i]  = params[2];
      out_sigx[i] = params[3];
      out_sigy[i] = params[4];
      out_B[i]    = params[5];
      // Always store chi2/ndf/prob
      out_chi2[i] = chi2Calc;
      if (ndfCalc > 0) {
        out_ndf[i]  = static_cast<double>(ndfCalc);
        out_prob[i] = TMath::Prob(chi2Calc, ndfCalc);
      } else {
        out_ndf[i]  = INVALID_VALUE;
        out_prob[i] = INVALID_VALUE;
      }
      const double xr = out_mux[i];
      const double yr = out_muy[i];
      out_x_rec[i] = xr; out_y_rec[i] = yr;
      out_dx_s[i] = (v_x_hit[i] - xr);
      out_dy_s[i] = (v_y_hit[i] - yr);
      nFitted.fetch_add(1, std::memory_order_relaxed);
    } else {
      double wsum = 0.0, xw = 0.0, yw = 0.0;
      for (int k = 0; k < nPts; ++k) { double w = std::max(0.0, Zf[k] - B0); wsum += w; xw += w * Xf[k]; yw += w * Yf[k]; }
      if (wsum > 0) {
        const double xr = xw / wsum; const double yr = yw / wsum;
        out_x_rec[i] = xr; out_y_rec[i] = yr;
        out_dx_s[i] = (v_x_hit[i] - xr);
        out_dy_s[i] = (v_y_hit[i] - yr);
        // Save seed parameters so branches are always populated
        out_A[i]    = A0;
        out_mux[i]  = xr;
        out_muy[i]  = yr;
        out_sigx[i] = sxInitMoment;
        out_sigy[i] = syInitMoment;
        out_B[i]    = B0;
        nFitted.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Absolute fallback: use initial guesses
        out_A[i]    = A0;
        out_mux[i]  = mux0;
        out_muy[i]  = muy0;
        out_sigx[i] = sxInitMoment;
        out_sigy[i] = syInitMoment;
        out_B[i]    = B0;
      }
    }
  }, indices);
  // Restore previous error level
  gErrorIgnoreLevel = prevErrorLevel_FitGaussian2D;


  // Sequentially write outputs
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    x_rec_3d = out_x_rec[i];
    y_rec_3d = out_y_rec[i];
    rec_hit_delta_x_3d_signed = out_dx_s[i];
    rec_hit_delta_y_3d_signed = out_dy_s[i];
    gauss3d_A   = out_A[i];
    gauss3d_mux = out_mux[i];
    gauss3d_muy = out_muy[i];
    gauss3d_sigx = out_sigx[i];
    gauss3d_sigy = out_sigy[i];
    gauss3d_B   = out_B[i];
    gauss3d_chi2 = out_chi2[i];
    gauss3d_ndf  = out_ndf[i];
    gauss3d_prob = out_prob[i];
    rootutils::FillBranches(coreBranches);
    rootutils::FillBranches(paramBranches);
    nProcessed++;
  }

  // Re-enable all branches to avoid persisting disabled-status to file
  tree->SetBranchStatus("*", 1);

  // Overwrite tree in file
  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();

  ::Info("FitGaussian2D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
  return 0;
}


// Backward-compatible wrapper matching the previous signature.
// Maps saveFitParameters=true to saving all individual parameters.
int FitGaussian2D(const char* filename,
              double errorPercentOfMax,
              bool saveFitParameters,
              const char* chargeBranch) {
  return FitGaussian2D(filename,
                      errorPercentOfMax,
                      /*saveParamA*/   saveFitParameters,
                      /*saveParamMux*/ saveFitParameters,
                      /*saveParamMuy*/ saveFitParameters,
                      /*saveParamSigx*/ saveFitParameters,
                      /*saveParamSigy*/ saveFitParameters,
                      /*saveParamB*/   saveFitParameters,
                      chargeBranch);
}
