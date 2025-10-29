// ROOT macro: FitGaus2D.C
// Performs 2D Gaussian fit on the full charge neighborhood to reconstruct
// deltas, and appends them as new branches.
// Uses Q_f (noisy charge per pixel, Coulombs) when available; falls back to Q_i.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
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
#include <ROOT/TThreadExecutor.hxx>

#include "ChargeUtils.h"

namespace {
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
int FitGaus2D(const char* filename = "../build/epicChargeSharing.root",
                 double errorPercentOfMax = 5.0,
                 bool saveParamA = true,
                 bool saveParamMux = true,
                 bool saveParamMuy = true,
                 bool saveParamSigx = true,
                 bool saveParamSigy = true,
                 bool saveParamB = true,
                 const char* chargeBranch = "Q_f",
                 bool useQnQiPercentErrors = true) {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(400);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("FitGaus2D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first for clearer diagnostics)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("FitGaus2D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)", filename);
    file->Close();
    delete file;
    return 3;
  }

  // Fetch metadata (pixel spacing, pixel size, neighborhood radius) with fallbacks
  double pixelSpacing = NAN;
  if (auto* spacingObj = dynamic_cast<TNamed*>(file->Get("GridPixelSpacing_mm"))) {
    try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) {}
  }
  double pixelSize = NAN;
  if (auto* sizeObj = dynamic_cast<TNamed*>(file->Get("GridPixelSize_mm"))) {
    try { pixelSize = std::stod(sizeObj->GetTitle()); } catch (...) {}
  }
  int neighborhoodRadiusMeta = -1;
  if (auto* rObj = dynamic_cast<TNamed*>(file->Get("NeighborhoodRadius"))) {
    try { neighborhoodRadiusMeta = std::stoi(rObj->GetTitle()); }
    catch (...) {
      try { neighborhoodRadiusMeta = static_cast<int>(std::lround(std::stod(rObj->GetTitle()))); } catch (...) {}
    }
  }

  auto inferSpacingFromTree = [&](TTree* t) -> double {
    std::vector<double> xs; xs.reserve(5000);
    std::vector<double> ys; ys.reserve(5000);
    double x_px_tmp = 0.0, y_px_tmp = 0.0;
    t->SetBranchAddress("PixelX", &x_px_tmp);
    t->SetBranchAddress("PixelY", &y_px_tmp);
    Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
    for (Long64_t i=0;i<nToScan;++i) {
      t->GetEntry(i);
      if (IsFinite3D(x_px_tmp)) xs.push_back(x_px_tmp);
      if (IsFinite3D(y_px_tmp)) ys.push_back(y_px_tmp);
    }
    auto computeGap = [](std::vector<double>& v)->double{
      if (v.size() < 2) return NAN;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      if (v.size() < 2) return NAN;
      std::vector<double> gaps; gaps.reserve(v.size());
      for (size_t i=1;i<v.size();++i) {
        double d = v[i]-v[i-1];
        if (d > 1e-9 && IsFinite3D(d)) gaps.push_back(d);
      }
      if (gaps.empty()) return NAN;
      std::nth_element(gaps.begin(), gaps.begin()+gaps.size()/2, gaps.end());
      return gaps[gaps.size()/2];
    };
    double gx = computeGap(xs);
    double gy = computeGap(ys);
    if (IsFinite3D(gx) && gx>0 && IsFinite3D(gy) && gy>0) return 0.5*(gx+gy);
    if (IsFinite3D(gx) && gx>0) return gx;
    if (IsFinite3D(gy) && gy>0) return gy;
    return NAN;
  };

  auto inferRadiusFromTree = [&](TTree* t, const std::string& preferred) -> int {
    // Prefer requested branch; fall back to Q_f, F_i, Q_i
    std::vector<double>* Q_tmp = nullptr;
    auto bind = [&](const char* b)->bool { if (t->GetBranch(b)) { t->SetBranchStatus(b, 1); t->SetBranchAddress(b, &Q_tmp); return true; } return false; };
    if (!preferred.empty() && bind(preferred.c_str())) {}
    else if (bind("Q_f")) {}
    else if (bind("F_i")) {}
    else if (bind("Q_i")) {}
    else return -1;
    Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
    for (Long64_t i=0;i<nToScan;++i) {
      t->GetEntry(i);
      if (Q_tmp && !Q_tmp->empty()) {
        const size_t total = Q_tmp->size();
        const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
        if (N >= 3 && N*N == static_cast<int>(total)) {
          return (N - 1) / 2;
        }
      }
    }
    return -1;
  };

  if (!IsFinite3D(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
  }
  if (!IsFinite3D(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("FitGaus2D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 2;
  }
  if (!IsFinite3D(pixelSize) || pixelSize <= 0) {
    // Fallback: if pixel size metadata is missing, use half of pitch as a conservative lower bound
    pixelSize = 0.5 * pixelSpacing;
  }
  
  // Decide which charge branch to use
  std::string chosenCharge = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Q_f");
  auto hasBranch = [&](const char* b){ return tree->GetBranch(b) != nullptr; };
  if (!hasBranch(chosenCharge.c_str())) {
    if (hasBranch("Q_f")) chosenCharge = "Q_f";
    else if (hasBranch("F_i")) chosenCharge = "F_i";
    else if (hasBranch("Q_i")) chosenCharge = "Q_i";
    else {
      ::Error("FitGaus2D", "No charge branch found (requested '%s'). Tried Q_f, F_i, Q_i.", chargeBranch ? chargeBranch : "<null>");
      file->Close();
      delete file;
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
  bool enableQiQnErrors = useQnQiPercentErrors;
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
      ::Warning("FitGaus2D", "Requested Q_i/Q_n vertical errors but required branches are missing. Falling back to percent-of-max uncertainty.");
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
  
  auto ensureAndResetBranch = [&](const char* name, double* addr) -> TBranch* {
    TBranch* br = tree->GetBranch(name);
    if (!br) {
      // Explicit leaflist to force double type and avoid ROOT guessing issues
      std::string leaf = std::string(name) + "/D";
      br = tree->Branch(name, addr, leaf.c_str());
    } else {
      // Rebind address and clear any previous content
      tree->SetBranchAddress(name, addr);
      br = tree->GetBranch(name);
      if (br) {
        br->Reset();
        br->DropBaskets();
      }
    }
    // Ensure branch is enabled for I/O
    tree->SetBranchStatus(name, 1);
    return br;
  };

  TBranch* br_x_rec = ensureAndResetBranch("ReconX_2D", &x_rec_3d);
  TBranch* br_y_rec = ensureAndResetBranch("ReconY_2D", &y_rec_3d);
  TBranch* br_dx_signed = ensureAndResetBranch("ReconTrueDeltaX_2D", &rec_hit_delta_x_3d_signed);
  TBranch* br_dy_signed = ensureAndResetBranch("ReconTrueDeltaY_2D", &rec_hit_delta_y_3d_signed);
  // Parameter branches
  TBranch* br_A = nullptr;
  TBranch* br_mux = nullptr;
  TBranch* br_muy = nullptr;
  TBranch* br_sigx = nullptr;
  TBranch* br_sigy = nullptr;
  TBranch* br_B = nullptr;
  if (saveParamA)   br_A   = ensureAndResetBranch("Gauss2D_A", &gauss3d_A);
  if (saveParamMux) br_mux = ensureAndResetBranch("Gauss2D_mux", &gauss3d_mux);
  if (saveParamMuy) br_muy = ensureAndResetBranch("Gauss2D_muy", &gauss3d_muy);
  if (saveParamSigx) br_sigx = ensureAndResetBranch("Gauss2D_sigx", &gauss3d_sigx);
  if (saveParamSigy) br_sigy = ensureAndResetBranch("Gauss2D_sigy", &gauss3d_sigy);
  if (saveParamB)   br_B   = ensureAndResetBranch("Gauss2D_B", &gauss3d_B);
  TBranch* br_chi2 = ensureAndResetBranch("Gauss2D_Chi2", &gauss3d_chi2);
  TBranch* br_ndf  = ensureAndResetBranch("Gauss2D_Ndf", &gauss3d_ndf);
  TBranch* br_prob = ensureAndResetBranch("Gauss2D_Prob", &gauss3d_prob);

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
  const int prevErrorLevel_FitGaus2D = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  exec.Foreach([&](int i){
    const bool isPix = v_is_pixel[i] != 0;
    const auto &QLoc = v_Q[i];
    if (isPix || QLoc.empty()) {
      return;
    }

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
          const double errVal = ComputeQnQiPercent(qiVal, (*QnLocPtr)[idx], qmaxQiNeighborhood);
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

    // Error model and sigma bounds
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0) ? relErr * qmaxNeighborhood : 0.0;
    auto selectError = [&](double candidate) -> double {
      if (std::isfinite(candidate) && candidate > 0.0) return candidate;
      if (uniformSigma > 0.0) return uniformSigma;
      return 1.0;
    };
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
    if (A0 < 1e-6) {
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
    // Note: relErr/uniformSigma/sigma seeds computed above

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

    // Build range
    double xMinR =  1e300, xMaxR = -1e300; double yMinR =  1e300, yMaxR = -1e300;
    for (int k = 0; k < g2d.GetN(); ++k) { xMinR = std::min(xMinR, g2d.GetX()[k]); xMaxR = std::max(xMaxR, g2d.GetX()[k]); yMinR = std::min(yMinR, g2d.GetY()[k]); yMaxR = std::max(yMaxR, g2d.GetY()[k]); }
    xMinR -= 0.5 * pixelSpacing; xMaxR += 0.5 * pixelSpacing; yMinR -= 0.5 * pixelSpacing; yMaxR += 0.5 * pixelSpacing;
    const double muXLo = v_x_px[i] - 0.5 * pixelSpacing;
    const double muXHi = v_x_px[i] + 0.5 * pixelSpacing;
    const double muYLo = v_y_px[i] - 0.5 * pixelSpacing;
    const double muYHi = v_y_px[i] + 0.5 * pixelSpacing;

    TF2 fModel("fModel", Gauss2DPlusB, xMinR, xMaxR, yMinR, yMaxR, 6);
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
      const double candidate = (errValsPtr && k < static_cast<int>(errValsPtr->size())) ? (*errValsPtr)[k] : std::numeric_limits<double>::quiet_NaN();
      double xy[2] = {Xf[k], Yf[k]};
      const double sigmaUsed = selectError(candidate);
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
        const double sigma = (k < static_cast<int>(sigmaVals.size()) && sigmaVals[k] > 0.0) ? sigmaVals[k] : 1.0;
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
  gErrorIgnoreLevel = prevErrorLevel_FitGaus2D;

  // Sequentially write outputs
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    x_rec_3d = out_x_rec[i];
    y_rec_3d = out_y_rec[i];
    rec_hit_delta_x_3d_signed = out_dx_s[i];
    rec_hit_delta_y_3d_signed = out_dy_s[i];
    if (br_A)   gauss3d_A   = out_A[i];
    if (br_mux) gauss3d_mux = out_mux[i];
    if (br_muy) gauss3d_muy = out_muy[i];
    if (br_sigx) gauss3d_sigx = out_sigx[i];
    if (br_sigy) gauss3d_sigy = out_sigy[i];
    if (br_B)   gauss3d_B   = out_B[i];
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
    nProcessed++;
  }

  // Re-enable all branches to avoid persisting disabled-status to file
  tree->SetBranchStatus("*", 1);

  // Overwrite tree in file
  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();
  delete file;

  ::Info("FitGaus2D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
  return 0;
}


// Backward-compatible wrapper matching the previous signature.
// Maps saveFitParameters=true to saving all individual parameters.
int FitGaus2D(const char* filename,
              double errorPercentOfMax,
              bool saveFitParameters,
              const char* chargeBranch) {
  return FitGaus2D(filename,
                      errorPercentOfMax,
                      /*saveParamA*/   saveFitParameters,
                      /*saveParamMux*/ saveFitParameters,
                      /*saveParamMuy*/ saveFitParameters,
                      /*saveParamSigx*/ saveFitParameters,
                      /*saveParamSigy*/ saveFitParameters,
                      /*saveParamB*/   saveFitParameters,
                      chargeBranch);
}
