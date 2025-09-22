// ROOT macro: processing2D.C
// Performs 1D Gaussian fits on central row and column of the charge neighborhood
// using Q_f (noisy charge per pixel) to reconstruct (x_rec_2d, y_rec_2d) and
// deltas, and appends them as new branches. Falls back to Q_i if Q_f is absent.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TROOT.h>
#include <TError.h>
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

namespace {
  // 1D Gaussian with constant offset: A * exp(-0.5*((x-mu)/sigma)^2) + B
  double GaussPlusB(double* x, double* p) {
    const double A     = p[0];
    const double mu    = p[1];
    const double sigma = p[2];
    const double B     = p[3];
    const double dx    = (x[0] - mu) / sigma;
    return A * std::exp(-0.5 * dx * dx) + B;
  }

  inline bool IsFinite(double v) {
    return std::isfinite(v);
  }
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%)
// of the event's maximum charge within the neighborhood. The same error is
// applied to all data points used in the fits for that event.
int processing2D(const char* filename = "../build/epicChargeSharing.root",
                 double errorPercentOfMax = 5.0,
                 bool saveParamA = true,
                 bool saveParamMu = true,
                 bool saveParamSigma = true,
                 bool saveParamB = true,
                 const char* chargeBranch = "Q_f",
                 bool removeOutliers = false,
                 double outlierSigma = 4.0,
                 int minPointsAfterClip = 3,
                 bool saveOutlierMask = false,
                 bool fitDiagonals = true,
                 bool saveDiagParamA = true,
                 bool saveDiagParamMu = true,
                 bool saveDiagParamSigma = true,
                 bool saveDiagParamB = true) {
  // Favor faster least-squares: Minuit2 + Fumili2
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("processing2D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first to provide clearer error when wrong file is passed)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)", filename);
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
      if (IsFinite(x_px_tmp)) xs.push_back(x_px_tmp);
      if (IsFinite(y_px_tmp)) ys.push_back(y_px_tmp);
    }
    auto computeGap = [](std::vector<double>& v)->double{
      if (v.size() < 2) return NAN;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      if (v.size() < 2) return NAN;
      std::vector<double> gaps; gaps.reserve(v.size());
      for (size_t i=1;i<v.size();++i) {
        double d = v[i]-v[i-1];
        if (d > 1e-9 && IsFinite(d)) gaps.push_back(d);
      }
      if (gaps.empty()) return NAN;
      std::nth_element(gaps.begin(), gaps.begin()+gaps.size()/2, gaps.end());
      return gaps[gaps.size()/2];
    };
    double gx = computeGap(xs);
    double gy = computeGap(ys);
    if (IsFinite(gx) && gx>0 && IsFinite(gy) && gy>0) return 0.5*(gx+gy);
    if (IsFinite(gx) && gx>0) return gx;
    if (IsFinite(gy) && gy>0) return gy;
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

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
  }
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing2D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 2;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
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
      ::Error("processing2D", "No charge branch found (requested '%s'). Tried Q_f, F_i, Q_i.", chargeBranch ? chargeBranch : "<null>");
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
  // Use Q_f (noisy) for fits; fall back to Q_i if Q_f absent
  std::vector<double>* Q = nullptr; // used for fits (charges in Coulombs)

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("TrueX", 1);
  tree->SetBranchStatus("TrueY", 1);
  tree->SetBranchStatus("PixelX", 1);
  tree->SetBranchStatus("PixelY", 1);
  tree->SetBranchStatus("isPixelHit", 1);
  // Enable only the chosen charge branch
  tree->SetBranchStatus(chosenCharge.c_str(), 1);

  tree->SetBranchAddress("TrueX", &x_hit);
  tree->SetBranchAddress("TrueY", &y_hit);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);

  // New branches (outputs).
  // Use NaN sentinel so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double x_rec_2d = INVALID_VALUE;
  double y_rec_2d = INVALID_VALUE;
  double rec_hit_delta_x_2d = INVALID_VALUE;
  double rec_hit_delta_y_2d = INVALID_VALUE;
  double rec_hit_delta_x_2d_signed = INVALID_VALUE;
  double rec_hit_delta_y_2d_signed = INVALID_VALUE;
  // 1D Gaussian fit parameters (row=x, col=y)
  double gauss2d_row_a = INVALID_VALUE;
  double gauss2d_row_mu = INVALID_VALUE;
  double gauss2d_row_sigma = INVALID_VALUE;
  double gauss2d_row_b = INVALID_VALUE;
  double gauss2d_col_a = INVALID_VALUE;
  double gauss2d_col_mu = INVALID_VALUE;
  double gauss2d_col_sigma = INVALID_VALUE;
  double gauss2d_col_b = INVALID_VALUE;

  // Optional diagonal reconstructions and parameters
  double x_rec_diag_main = INVALID_VALUE;   // main diagonal (dj = di)
  double y_rec_diag_main = INVALID_VALUE;
  double x_rec_diag_sec  = INVALID_VALUE;   // secondary diagonal (dj = -di)
  double y_rec_diag_sec  = INVALID_VALUE;
  double gaussD1_a = INVALID_VALUE, gaussD1_mu = INVALID_VALUE, gaussD1_sigma = INVALID_VALUE, gaussD1_b = INVALID_VALUE;
  double gaussD2_a = INVALID_VALUE, gaussD2_mu = INVALID_VALUE, gaussD2_sigma = INVALID_VALUE, gaussD2_b = INVALID_VALUE;
  // Signed deltas for diagonals
  double mdiag_dx_signed = INVALID_VALUE;
  double mdiag_dy_signed = INVALID_VALUE;
  double sdiag_dx_signed = INVALID_VALUE;
  double sdiag_dy_signed = INVALID_VALUE;

  // If branches already exist, we will overwrite their contents
  auto ensureAndResetBranch = [&](const char* name, double* addr) -> TBranch* {
    TBranch* br = tree->GetBranch(name);
    if (!br) {
      br = tree->Branch(name, addr);
    } else {
      tree->SetBranchAddress(name, addr);
      br = tree->GetBranch(name);
      if (br) {
        br->Reset();        // clear previous entries
        br->DropBaskets();  // drop old baskets to avoid mixing old data
      }
    }
    return br;
  };

  TBranch* br_x_rec = ensureAndResetBranch("ReconX", &x_rec_2d);
  TBranch* br_y_rec = ensureAndResetBranch("ReconY", &y_rec_2d);
  // Commented out per request: do not save absolute-value delta branches
  // TBranch* br_dx    = ensureAndResetBranch("ReconTrueDeltaX", &rec_hit_delta_x_2d);
  // TBranch* br_dy    = ensureAndResetBranch("ReconTrueDeltaY", &rec_hit_delta_y_2d);
  TBranch* br_dx_signed = ensureAndResetBranch("ReconTrueDeltaX", &rec_hit_delta_x_2d_signed);
  TBranch* br_dy_signed = ensureAndResetBranch("ReconTrueDeltaY", &rec_hit_delta_y_2d_signed);
  // Parameter branches
  TBranch* br_row_A = nullptr;
  TBranch* br_row_mu = nullptr;
  TBranch* br_row_sigma = nullptr;
  TBranch* br_row_B = nullptr;
  TBranch* br_col_A = nullptr;
  TBranch* br_col_mu = nullptr;
  TBranch* br_col_sigma = nullptr;
  TBranch* br_col_B = nullptr;
  // Diagonal parameter/output branches
  TBranch* br_x_rec_diag_main = nullptr;
  TBranch* br_y_rec_diag_main = nullptr;
  TBranch* br_x_rec_diag_sec  = nullptr;
  TBranch* br_y_rec_diag_sec  = nullptr;
  TBranch* br_d1_A = nullptr;
  TBranch* br_d1_mu = nullptr;
  TBranch* br_d1_sigma = nullptr;
  TBranch* br_d1_B = nullptr;
  TBranch* br_d2_A = nullptr;
  TBranch* br_d2_mu = nullptr;
  TBranch* br_d2_sigma = nullptr;
  TBranch* br_d2_B = nullptr;
  // Diagonal signed delta branches
  TBranch* br_mdiag_dx_signed = nullptr;
  TBranch* br_mdiag_dy_signed = nullptr;
  TBranch* br_sdiag_dx_signed = nullptr;
  TBranch* br_sdiag_dy_signed = nullptr;
  if (saveParamA) {
    br_row_A     = ensureAndResetBranch("GaussRowA", &gauss2d_row_a);
    br_col_A     = ensureAndResetBranch("GaussColA", &gauss2d_col_a);
  }
  if (saveParamMu) {
    br_row_mu    = ensureAndResetBranch("GaussRowMu", &gauss2d_row_mu);
    br_col_mu    = ensureAndResetBranch("GaussColMu", &gauss2d_col_mu);
  }
  if (saveParamSigma) {
    br_row_sigma = ensureAndResetBranch("GaussRowSigma", &gauss2d_row_sigma);
    br_col_sigma = ensureAndResetBranch("GaussColSigma", &gauss2d_col_sigma);
  }
  if (saveParamB) {
    br_row_B     = ensureAndResetBranch("GaussRowB", &gauss2d_row_b);
    br_col_B     = ensureAndResetBranch("GaussColB", &gauss2d_col_b);
  }

  // Optional branches for sigma-clipping removal masks (0 = kept, 1 = removed)
  TBranch* br_row_mask = nullptr;
  TBranch* br_col_mask = nullptr;
  std::vector<int> row_mask;
  std::vector<int> col_mask;
  auto ensureAndResetVectorIntBranch = [&](const char* name, std::vector<int>* addr) -> TBranch* {
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
  };
  if (saveOutlierMask) {
    br_row_mask = ensureAndResetVectorIntBranch("GaussRowMaskRemoved", &row_mask);
    br_col_mask = ensureAndResetVectorIntBranch("GaussColMaskRemoved", &col_mask);
  }

  // Create diagonal branches if requested
  if (fitDiagonals) {
    br_x_rec_diag_main = ensureAndResetBranch("ReconX_DiagMain", &x_rec_diag_main);
    br_y_rec_diag_main = ensureAndResetBranch("ReconY_DiagMain", &y_rec_diag_main);
    br_x_rec_diag_sec  = ensureAndResetBranch("ReconX_DiagSecond", &x_rec_diag_sec);
    br_y_rec_diag_sec  = ensureAndResetBranch("ReconY_DiagSecond", &y_rec_diag_sec);
    br_mdiag_dx_signed = ensureAndResetBranch("MDiagReconTrueDeltaX", &mdiag_dx_signed);
    br_mdiag_dy_signed = ensureAndResetBranch("MDiagReconTrueDeltaY", &mdiag_dy_signed);
    br_sdiag_dx_signed = ensureAndResetBranch("SDiagReconTrueDeltaX", &sdiag_dx_signed);
    br_sdiag_dy_signed = ensureAndResetBranch("SDiagReconTrueDeltaY", &sdiag_dy_signed);
    if (saveDiagParamA) {
      br_d1_A = ensureAndResetBranch("GaussDiagMainA", &gaussD1_a);
      br_d2_A = ensureAndResetBranch("GaussDiagSecondA", &gaussD2_a);
    }
    if (saveDiagParamMu) {
      br_d1_mu = ensureAndResetBranch("GaussDiagMainMu", &gaussD1_mu);
      br_d2_mu = ensureAndResetBranch("GaussDiagSecondMu", &gaussD2_mu);
    }
    if (saveDiagParamSigma) {
      br_d1_sigma = ensureAndResetBranch("GaussDiagMainSigma", &gaussD1_sigma);
      br_d2_sigma = ensureAndResetBranch("GaussDiagSecondSigma", &gaussD2_sigma);
    }
    if (saveDiagParamB) {
      br_d1_B = ensureAndResetBranch("GaussDiagMainB", &gaussD1_b);
      br_d2_B = ensureAndResetBranch("GaussDiagSecondB", &gaussD2_b);
    }
  }

  // Fitting function for 1D gaussian + const (locals created per-fit below)

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0;
  std::atomic<long long> nFitted{0};

  // Preload inputs sequentially to avoid ROOT I/O races
  std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
  std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
  std::vector<char> v_is_pixel(nEntries);
  std::vector<std::vector<double>> v_Q(nEntries);
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    v_x_hit[i] = x_hit;
    v_y_hit[i] = y_hit;
    v_x_px[i]  = x_px;
    v_y_px[i]  = y_px;
    v_is_pixel[i] = is_pixel_hit ? 1 : 0;
    if (Q && !Q->empty()) v_Q[i] = *Q; else v_Q[i].clear();
  }

  // Prepare output buffers
  std::vector<double> out_x_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_y_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_dx(nEntries, INVALID_VALUE);
  std::vector<double> out_dy(nEntries, INVALID_VALUE);
  std::vector<double> out_dx_s(nEntries, INVALID_VALUE);
  std::vector<double> out_dy_s(nEntries, INVALID_VALUE);
  // Output buffers for fit parameters
  std::vector<double> out_row_A(nEntries, INVALID_VALUE);
  std::vector<double> out_row_mu(nEntries, INVALID_VALUE);
  std::vector<double> out_row_sigma(nEntries, INVALID_VALUE);
  std::vector<double> out_row_B(nEntries, INVALID_VALUE);
  std::vector<double> out_col_A(nEntries, INVALID_VALUE);
  std::vector<double> out_col_mu(nEntries, INVALID_VALUE);
  std::vector<double> out_col_sigma(nEntries, INVALID_VALUE);
  std::vector<double> out_col_B(nEntries, INVALID_VALUE);
  // Output buffers for removal masks
  std::vector<std::vector<int>> out_row_mask(nEntries);
  std::vector<std::vector<int>> out_col_mask(nEntries);

  // Diagonal buffers
  std::vector<double> out_x_rec_diag_main, out_y_rec_diag_main, out_x_rec_diag_sec, out_y_rec_diag_sec;
  std::vector<double> out_d1_A, out_d1_mu, out_d1_sigma, out_d1_B;
  std::vector<double> out_d2_A, out_d2_mu, out_d2_sigma, out_d2_B;
  std::vector<std::vector<int>> out_d1_mask, out_d2_mask;
  std::vector<double> out_mdiag_dx_s, out_mdiag_dy_s, out_sdiag_dx_s, out_sdiag_dy_s;
  if (fitDiagonals) {
    out_x_rec_diag_main.assign(nEntries, INVALID_VALUE);
    out_y_rec_diag_main.assign(nEntries, INVALID_VALUE);
    out_x_rec_diag_sec.assign(nEntries, INVALID_VALUE);
    out_y_rec_diag_sec.assign(nEntries, INVALID_VALUE);
    out_d1_A.assign(nEntries, INVALID_VALUE);
    out_d1_mu.assign(nEntries, INVALID_VALUE);
    out_d1_sigma.assign(nEntries, INVALID_VALUE);
    out_d1_B.assign(nEntries, INVALID_VALUE);
    out_d2_A.assign(nEntries, INVALID_VALUE);
    out_d2_mu.assign(nEntries, INVALID_VALUE);
    out_d2_sigma.assign(nEntries, INVALID_VALUE);
    out_d2_B.assign(nEntries, INVALID_VALUE);
    out_mdiag_dx_s.assign(nEntries, INVALID_VALUE);
    out_mdiag_dy_s.assign(nEntries, INVALID_VALUE);
    out_sdiag_dx_s.assign(nEntries, INVALID_VALUE);
    out_sdiag_dy_s.assign(nEntries, INVALID_VALUE);
    if (saveOutlierMask) {
      out_d1_mask.resize(nEntries);
      out_d2_mask.resize(nEntries);
    }
  }

  // Parallel computation over entries
  std::vector<int> indices(nEntries);
  std::iota(indices.begin(), indices.end(), 0);
  ROOT::TThreadExecutor exec; // uses ROOT IMT pool size by default
  // Suppress expected Minuit2 error spam during Fumili2 attempts; we'll fallback to MIGRAD if needed
  const int prevErrorLevel_processing2D = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  exec.Foreach([&](int i){
    const bool isPix = v_is_pixel[i] != 0;
    const auto &QLoc = v_Q[i];
    if (isPix || QLoc.empty()) {
      if (saveOutlierMask) {
        out_row_mask[i].clear();
        out_col_mask[i].clear();
      }
      return;
    }

    const size_t total = QLoc.size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      if (saveOutlierMask) {
        out_row_mask[i].clear();
        out_col_mask[i].clear();
      }
      return;
    }
    const int R = (N - 1) / 2;

    std::vector<double> x_row; x_row.reserve(N);
    std::vector<double> q_row; q_row.reserve(N);
    std::vector<double> y_col; y_col.reserve(N);
    std::vector<double> q_col; q_col.reserve(N);
    double qmaxNeighborhood = -1e300;
    const double x_px_loc = v_x_px[i];
    const double y_px_loc = v_y_px[i];
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx  = (di + R) * N + (dj + R);
        const double q = QLoc[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        if (dj == 0) {
          const double x = x_px_loc + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
        }
        if (di == 0) {
          const double y = y_px_loc + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
        }
      }
    }
    if (x_row.size() < 3 || y_col.size() < 3) {
      if (saveOutlierMask) {
        out_row_mask[i] = std::vector<int>(x_row.size(), 0);
        out_col_mask[i] = std::vector<int>(y_col.size(), 0);
      }
      return;
    }

    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    double B0_row = std::max(0.0, *minmaxRow.first);
    double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    double B0_col = std::max(0.0, *minmaxCol.first);

    // Low-contrast: fast centroid (relative to neighborhood max charge)
    const double contrastEps = (qmaxNeighborhood > 0.0) ? (1e-3 * qmaxNeighborhood) : 0.0;
    if (qmaxNeighborhood > 0.0 && A0_row < contrastEps && A0_col < contrastEps) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k] - B0_row); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k] - B0_col); wsumy += w; yw += w * y_col[k]; }
      if (wsumx > 0 && wsumy > 0) {
        const double xr = xw / wsumx;
        const double yr = yw / wsumy;
        out_x_rec[i] = xr;
        out_y_rec[i] = yr;
        out_dx[i] = std::abs(v_x_hit[i] - xr);
        out_dy[i] = std::abs(v_y_hit[i] - yr);
        out_dx_s[i] = (v_x_hit[i] - xr);
        out_dy_s[i] = (v_y_hit[i] - yr);
        nFitted.fetch_add(1, std::memory_order_relaxed);
        if (saveOutlierMask) {
          out_row_mask[i] = std::vector<int>(x_row.size(), 0);
          out_col_mask[i] = std::vector<int>(y_col.size(), 0);
        }
      }
      return;
    }

    // Seeds
    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    // Constrain sigma to be within [pixel size, radius * pitch]
    const double sigLoBound = pixelSize;
    const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
    auto sigmaSeed1D = [&](const std::vector<double>& xs, const std::vector<double>& qs, double B0)->double {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); wsum += w; xw += w * xs[k]; }
      if (wsum <= 0.0) {
        double s = std::max(0.25*pixelSpacing, 1e-6);
        if (s < sigLoBound) s = sigLoBound;
        if (s > sigHiBound) s = sigHiBound;
        return s;
      }
      const double mean = xw / wsum;
      double var = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); const double dx = xs[k] - mean; var += w * dx * dx; }
      var = (wsum > 0.0) ? (var / wsum) : 0.0;
      double s = std::sqrt(std::max(var, 1e-12));
      if (s < sigLoBound) s = sigLoBound;
      if (s > sigHiBound) s = sigHiBound;
      return s;
    };
    double sigInitRow = sigmaSeed1D(x_row, q_row, B0_row);
    double sigInitCol = sigmaSeed1D(y_col, q_col, B0_col);

    std::vector<double> x_row_fit = x_row;
    std::vector<double> q_row_fit = q_row;
    std::vector<double> y_col_fit = y_col;
    std::vector<double> q_col_fit = q_col;
    std::vector<int> maskRow, maskCol;
    std::vector<int> maskD1, maskD2;

    // Optional sigma-clipping outlier removal using seeded model
    auto medianVal = [&](std::vector<double> v)->double {
      if (v.empty()) return 0.0;
      size_t m = v.size()/2;
      std::nth_element(v.begin(), v.begin()+m, v.end());
      double med = v[m];
      if ((v.size() & 1U) == 0) {
        auto max_it = std::max_element(v.begin(), v.begin()+m);
        med = 0.5 * (med + *max_it);
      }
      return med;
    };
    auto madSigma = [&](const std::vector<double>& r)->double {
      if (r.empty()) return 0.0;
      std::vector<double> rcopy = r;
      double med = medianVal(rcopy);
      std::vector<double> dev; dev.reserve(r.size());
      for (double v : r) dev.push_back(std::abs(v - med));
      double mad = medianVal(dev);
      return 1.4826 * mad;
    };
    auto clip1D = [&](std::vector<double>& xs, std::vector<double>& qs,
                      double A0, double mu0, double sig0, double B0,
                      std::vector<int>* removedMask) {
      if (removedMask) removedMask->assign(xs.size(), 0);
      if (!removeOutliers || xs.size() < 3 || qs.size() != xs.size()) return;
      if (!(outlierSigma > 0.0) || minPointsAfterClip <= 0) return;
      std::vector<double> resid; resid.reserve(qs.size());
      for (size_t k=0;k<xs.size();++k) {
        const double dx = (xs[k] - mu0) / sig0;
        const double model = A0 * std::exp(-0.5 * dx * dx) + B0;
        resid.push_back(qs[k] - model);
      }
      const double sigR = madSigma(resid);
      if (!(sigR > 0.0)) return;
      const double thr = outlierSigma * sigR;
      std::vector<double> xs_new; xs_new.reserve(xs.size());
      std::vector<double> qs_new; qs_new.reserve(qs.size());
      const double rmed = medianVal(resid);
      for (size_t k=0;k<xs.size();++k) {
        const double r = resid[k] - rmed;
        if (std::abs(r) <= thr) { xs_new.push_back(xs[k]); qs_new.push_back(qs[k]); }
        else if (removedMask) { (*removedMask)[k] = 1; }
      }
      if (static_cast<int>(xs_new.size()) >= std::max(minPointsAfterClip, 3)) {
        xs.swap(xs_new);
        qs.swap(qs_new);
      } else {
        // Not enough points after clipping; revert mask to all-kept
        if (removedMask) removedMask->assign(removedMask->size(), 0);
      }
    };

    // Clip row and column independently using seeded parameters
    clip1D(x_row_fit, q_row_fit, A0_row, mu0_row, sigInitRow, B0_row, saveOutlierMask ? &maskRow : nullptr);
    clip1D(y_col_fit, q_col_fit, A0_col, mu0_col, sigInitCol, B0_col, saveOutlierMask ? &maskCol : nullptr);
    // Recompute seeds after clipping
    if (x_row_fit.size() >= 3) {
      auto minmaxRow2 = std::minmax_element(q_row_fit.begin(), q_row_fit.end());
      A0_row = std::max(1e-18, *minmaxRow2.second - *minmaxRow2.first);
      B0_row = std::max(0.0, *minmaxRow2.first);
      int idxMaxRow2 = std::distance(q_row_fit.begin(), std::max_element(q_row_fit.begin(), q_row_fit.end()));
      mu0_row = x_row_fit[idxMaxRow2];
      sigInitRow = sigmaSeed1D(x_row_fit, q_row_fit, B0_row);
    }
    if (y_col_fit.size() >= 3) {
      auto minmaxCol2 = std::minmax_element(q_col_fit.begin(), q_col_fit.end());
      A0_col = std::max(1e-18, *minmaxCol2.second - *minmaxCol2.first);
      B0_col = std::max(0.0, *minmaxCol2.first);
      int idxMaxCol2 = std::distance(q_col_fit.begin(), std::max_element(q_col_fit.begin(), q_col_fit.end()));
      mu0_col = y_col_fit[idxMaxCol2];
      sigInitCol = sigmaSeed1D(y_col_fit, q_col_fit, B0_col);
    }

    // sigInitRow/sigInitCol were computed above on unfiltered data. They are possibly
    // updated after clipping in the block above.

    TF1 fRowLoc("fRowLoc", GaussPlusB, -1e9, 1e9, 4);
    TF1 fColLoc("fColLoc", GaussPlusB, -1e9, 1e9, 4);
    fRowLoc.SetParameters(A0_row, mu0_row, sigInitRow, B0_row);
    fColLoc.SetParameters(A0_col, mu0_col, sigInitCol, B0_col);

    auto minmaxX = std::minmax_element(x_row_fit.begin(), x_row_fit.end());
    auto minmaxY = std::minmax_element(y_col_fit.begin(), y_col_fit.end());
    const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
    const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
    const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
    const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
    fRowLoc.SetRange(xMin, xMax);
    fColLoc.SetRange(yMin, yMax);
    const double muXLo = x_px_loc - 0.5 * pixelSpacing;
    const double muXHi = x_px_loc + 0.5 * pixelSpacing;
    const double muYLo = y_px_loc - 0.5 * pixelSpacing;
    const double muYHi = y_px_loc + 0.5 * pixelSpacing;
    fRowLoc.SetParLimits(1, muXLo, muXHi);
    fColLoc.SetParLimits(1, muYLo, muYHi);
    // Bounds for Q_i fits: A in (0, ~2*qmax], B in [0, ~qmax]
    const double AHi = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
    const double BHi = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
    fRowLoc.SetParLimits(0, 1e-18, AHi);
    fRowLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fRowLoc.SetParLimits(3, 0.0, BHi);
    fColLoc.SetParLimits(0, 1e-18, AHi);
    fColLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fColLoc.SetParLimits(3, 0.0, BHi);

    ROOT::Math::WrappedMultiTF1 wRow(fRowLoc, 1);
    ROOT::Math::WrappedMultiTF1 wCol(fColLoc, 1);
    ROOT::Fit::BinData dataRow(static_cast<int>(x_row_fit.size()), 1);
    ROOT::Fit::BinData dataCol(static_cast<int>(y_col_fit.size()), 1);
    for (size_t k = 0; k < x_row_fit.size(); ++k) {
      const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
      dataRow.Add(x_row_fit[k], q_row_fit[k], ey);
    }
    for (size_t k = 0; k < y_col_fit.size(); ++k) {
      const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
      dataCol.Add(y_col_fit[k], q_col_fit[k], ey);
    }
    ROOT::Fit::Fitter fitRow;
    ROOT::Fit::Fitter fitCol;
    fitRow.Config().SetMinimizer("Minuit2", "Fumili2");
    fitCol.Config().SetMinimizer("Minuit2", "Fumili2");
    fitRow.Config().MinimizerOptions().SetStrategy(0);
    fitCol.Config().MinimizerOptions().SetStrategy(0);
    fitRow.Config().MinimizerOptions().SetTolerance(1e-4);
    fitCol.Config().MinimizerOptions().SetTolerance(1e-4);
    fitRow.Config().MinimizerOptions().SetPrintLevel(0);
    fitCol.Config().MinimizerOptions().SetPrintLevel(0);
    fitRow.SetFunction(wRow);
    fitCol.SetFunction(wCol);
    // A in (0, ~2*qmax]
    fitRow.Config().ParSettings(0).SetLimits(1e-18, AHi);
    fitRow.Config().ParSettings(1).SetLimits(muXLo, muXHi);
    fitRow.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    // B in [0, ~qmax]
    fitRow.Config().ParSettings(3).SetLimits(0.0, BHi);
    // A in (0, ~2*qmax]
    fitCol.Config().ParSettings(0).SetLimits(1e-18, AHi);
    fitCol.Config().ParSettings(1).SetLimits(muYLo, muYHi);
    fitCol.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    // B in [0, ~qmax]
    fitCol.Config().ParSettings(3).SetLimits(0.0, BHi);
    const double stepA_row = std::max(1e-18, 0.01 * A0_row);
    const double stepA_col = std::max(1e-18, 0.01 * A0_col);
    const double stepB_row = std::max(1e-18, 0.01 * std::max(B0_row, A0_row));
    const double stepB_col = std::max(1e-18, 0.01 * std::max(B0_col, A0_col));
    fitRow.Config().ParSettings(0).SetStepSize(stepA_row);
    fitRow.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(3).SetStepSize(stepB_row);
    fitCol.Config().ParSettings(0).SetStepSize(stepA_col);
    fitCol.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(3).SetStepSize(stepB_col);
    fitRow.Config().ParSettings(0).SetValue(A0_row);
    fitRow.Config().ParSettings(1).SetValue(mu0_row);
    fitRow.Config().ParSettings(2).SetValue(sigInitRow);
    fitRow.Config().ParSettings(3).SetValue(B0_row);
    fitCol.Config().ParSettings(0).SetValue(A0_col);
    fitCol.Config().ParSettings(1).SetValue(mu0_col);
    fitCol.Config().ParSettings(2).SetValue(sigInitCol);
    fitCol.Config().ParSettings(3).SetValue(B0_col);

    bool okRowFit = fitRow.Fit(dataRow);
    bool okColFit = fitCol.Fit(dataCol);
    // If Fumili2 fails, retry once with MIGRAD which is more robust
    if (!okRowFit) {
      fitRow.Config().SetMinimizer("Minuit2", "Migrad");
      fitRow.Config().MinimizerOptions().SetStrategy(1);
      fitRow.Config().MinimizerOptions().SetTolerance(1e-3);
      fitRow.Config().MinimizerOptions().SetPrintLevel(0);
      okRowFit = fitRow.Fit(dataRow);
    }
    if (!okColFit) {
      fitCol.Config().SetMinimizer("Minuit2", "Migrad");
      fitCol.Config().MinimizerOptions().SetStrategy(1);
      fitCol.Config().MinimizerOptions().SetTolerance(1e-3);
      fitCol.Config().MinimizerOptions().SetPrintLevel(0);
      okColFit = fitCol.Fit(dataCol);
    }
    // Save parameters only if the corresponding fit converged
    if (okRowFit) {
      out_row_A[i]     = fitRow.Result().Parameter(0);
      out_row_mu[i]    = fitRow.Result().Parameter(1);
      out_row_sigma[i] = fitRow.Result().Parameter(2);
      out_row_B[i]     = fitRow.Result().Parameter(3);
    }
    if (okColFit) {
      out_col_A[i]     = fitCol.Result().Parameter(0);
      out_col_mu[i]    = fitCol.Result().Parameter(1);
      out_col_sigma[i] = fitCol.Result().Parameter(2);
      out_col_B[i]     = fitCol.Result().Parameter(3);
    }
    double muX = NAN, muY = NAN;
    if (okRowFit) muX = fitRow.Result().Parameter(1);
    if (okColFit) muY = fitCol.Result().Parameter(1);
    if (!okRowFit) {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row_fit.size();++k) { double w = std::max(0.0, q_row_fit[k] - B0_row); wsum += w; xw += w * x_row_fit[k]; }
      if (wsum > 0) { muX = xw / wsum; }
    }
    if (!okColFit) {
      double wsum = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col_fit.size();++k) { double w = std::max(0.0, q_col_fit[k] - B0_col); wsum += w; yw += w * y_col_fit[k]; }
      if (wsum > 0) { muY = yw / wsum; }
    }
    const bool okRow = IsFinite(muX);
    const bool okCol = IsFinite(muY);
    if (okRow && okCol) {
      out_x_rec[i] = muX;
      out_y_rec[i] = muY;
      out_dx[i] = std::abs(v_x_hit[i] - muX);
      out_dy[i] = std::abs(v_y_hit[i] - muY);
      out_dx_s[i] = (v_x_hit[i] - muX);
      out_dy_s[i] = (v_y_hit[i] - muY);
      nFitted.fetch_add(1, std::memory_order_relaxed);
    }
    if (saveOutlierMask) {
      // If masks were never initialized (e.g. no clipping), default to all-kept
      if (maskRow.empty()) maskRow = std::vector<int>(x_row.size(), 0);
      if (maskCol.empty()) maskCol = std::vector<int>(y_col.size(), 0);
      out_row_mask[i] = std::move(maskRow);
      out_col_mask[i] = std::move(maskCol);
    }

    // =========================
    // Diagonal fits (optional)
    // =========================
    if (fitDiagonals) {
      // Build main (dj=di) and secondary (dj=-di) diagonal samples
      std::vector<double> s_d1; s_d1.reserve(N);
      std::vector<double> q_d1; q_d1.reserve(N);
      std::vector<double> s_d2; s_d2.reserve(N);
      std::vector<double> q_d2; q_d2.reserve(N);
      for (int k = -R; k <= R; ++k) {
        // main diagonal
        {
          const int idx = (k + R) * N + (k + R);
          const double q = QLoc[idx];
          if (IsFinite(q) && q >= 0.0) {
            const double s = x_px_loc + k * pixelSpacing; // use X as the diagonal coordinate
            s_d1.push_back(s);
            q_d1.push_back(q);
          }
        }
        // secondary diagonal
        {
          const int idx = (k + R) * N + (-k + R);
          const double q = QLoc[idx];
          if (IsFinite(q) && q >= 0.0) {
            const double s = x_px_loc + k * pixelSpacing; // use X as the diagonal coordinate
            s_d2.push_back(s);
            q_d2.push_back(q);
          }
        }
      }

      if (saveOutlierMask) {
        maskD1.assign(s_d1.size(), 0);
        maskD2.assign(s_d2.size(), 0);
      }

      auto fitDiag = [&](const std::vector<double>& s_vals_in,
                         const std::vector<double>& q_vals_in,
                         std::vector<int>* maskOut,
                         double muLo, double muHi,
                         double& outA, double& outMu, double& outSig, double& outB) -> bool {
        if (s_vals_in.size() < 3) return false;
        // Seeds from min/max
        auto mm = std::minmax_element(q_vals_in.begin(), q_vals_in.end());
        double A0 = std::max(1e-18, *mm.second - *mm.first);
        double B0 = std::max(0.0, *mm.first);
        int idxMax = std::distance(q_vals_in.begin(), std::max_element(q_vals_in.begin(), q_vals_in.end()));
        double mu0 = s_vals_in[idxMax];
        // Copy for optional clipping
        std::vector<double> s_vals = s_vals_in;
        std::vector<double> q_vals = q_vals_in;
        // Sigma seed and bounds like rows/cols
        const double sigLoBound = pixelSize;
        const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
        auto sigmaSeed1D_local = [&](const std::vector<double>& xs, const std::vector<double>& qs, double B0L)->double {
          double wsum = 0.0, xw = 0.0;
          for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0L); wsum += w; xw += w * xs[k]; }
          if (wsum <= 0.0) {
            double s = std::max(0.25*pixelSpacing, 1e-6);
            if (s < sigLoBound) s = sigLoBound;
            if (s > sigHiBound) s = sigHiBound;
            return s;
          }
          const double mean = xw / wsum;
          double var = 0.0;
          for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0L); const double dx = xs[k] - mean; var += w * dx * dx; }
          var = (wsum > 0.0) ? (var / wsum) : 0.0;
          double s = std::sqrt(std::max(var, 1e-12));
          if (s < sigLoBound) s = sigLoBound;
          if (s > sigHiBound) s = sigHiBound;
          return s;
        };
        double sigInit = sigmaSeed1D_local(s_vals, q_vals, B0);
        // Optional clipping using seeded params
        if (removeOutliers && maskOut) {
          // reuse clip1D from above scope
          clip1D(s_vals, q_vals, A0, mu0, sigInit, B0, maskOut);
          // recompute seeds after clipping
          if (s_vals.size() >= 3) {
            auto mm2 = std::minmax_element(q_vals.begin(), q_vals.end());
            A0 = std::max(1e-18, *mm2.second - *mm2.first);
            B0 = std::max(0.0, *mm2.first);
            int idxMax2 = std::distance(q_vals.begin(), std::max_element(q_vals.begin(), q_vals.end()));
            mu0 = s_vals[idxMax2];
            sigInit = sigmaSeed1D_local(s_vals, q_vals, B0);
          }
        } else if (maskOut) {
          maskOut->assign(s_vals.size(), 0);
        }
        // Prepare wrapped TF1
        TF1 fLoc("fDiagLoc", GaussPlusB, -1e9, 1e9, 4);
        fLoc.SetParameters(A0, mu0, sigInit, B0);
        ROOT::Math::WrappedMultiTF1 wLoc(fLoc, 1);
        ROOT::Fit::BinData data(static_cast<int>(s_vals.size()), 1);
        for (size_t k=0;k<s_vals.size();++k) {
          const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
          data.Add(s_vals[k], q_vals[k], ey);
        }
        ROOT::Fit::Fitter fitter;
        fitter.Config().SetMinimizer("Minuit2", "Fumili2");
        fitter.Config().MinimizerOptions().SetStrategy(0);
        fitter.Config().MinimizerOptions().SetTolerance(1e-4);
        fitter.Config().MinimizerOptions().SetPrintLevel(0);
        fitter.SetFunction(wLoc);
        const double AHiL = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
        const double BHiL = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
        fitter.Config().ParSettings(0).SetLimits(1e-18, AHiL);
        fitter.Config().ParSettings(1).SetLimits(muLo, muHi);
        fitter.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
        fitter.Config().ParSettings(3).SetLimits(0.0, BHiL);
        fitter.Config().ParSettings(0).SetStepSize(std::max(1e-18, 0.01 * A0));
        fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(3).SetStepSize(std::max(1e-18, 0.01 * std::max(B0, A0)));
        fitter.Config().ParSettings(0).SetValue(A0);
        fitter.Config().ParSettings(1).SetValue(mu0);
        fitter.Config().ParSettings(2).SetValue(sigInit);
        fitter.Config().ParSettings(3).SetValue(B0);
        bool ok = fitter.Fit(data);
        if (!ok) {
          fitter.Config().SetMinimizer("Minuit2", "Migrad");
          fitter.Config().MinimizerOptions().SetStrategy(1);
          fitter.Config().MinimizerOptions().SetTolerance(1e-3);
          fitter.Config().MinimizerOptions().SetPrintLevel(0);
          ok = fitter.Fit(data);
        }
        if (ok) {
          outA = fitter.Result().Parameter(0);
          outMu = fitter.Result().Parameter(1);
          outSig = fitter.Result().Parameter(2);
          outB = fitter.Result().Parameter(3);
          return true;
        }
        // Fallback: baseline-subtracted weighted centroid
        double wsum = 0.0, sw = 0.0;
        for (size_t k=0;k<s_vals.size();++k) { double w = std::max(0.0, q_vals[k] - B0); wsum += w; sw += w * s_vals[k]; }
        if (wsum > 0) {
          outA = A0; outB = B0; outSig = sigInit; outMu = sw / wsum;
          return true;
        }
        return false;
      };

      const double muLo = x_px_loc - 0.5 * pixelSpacing;
      const double muHi = x_px_loc + 0.5 * pixelSpacing;
      double A1=INVALID_VALUE, mu1=INVALID_VALUE, S1=INVALID_VALUE, B1=INVALID_VALUE;
      double A2=INVALID_VALUE, mu2=INVALID_VALUE, S2=INVALID_VALUE, B2=INVALID_VALUE;
      bool ok1 = fitDiag(s_d1, q_d1, saveOutlierMask ? &maskD1 : nullptr, muLo, muHi, A1, mu1, S1, B1);
      bool ok2 = fitDiag(s_d2, q_d2, saveOutlierMask ? &maskD2 : nullptr, muLo, muHi, A2, mu2, S2, B2);
      if (ok1) {
        // Transform diagonal coordinate mu -> (x,y)
        const double dx = mu1 - x_px_loc;
        out_x_rec_diag_main[i] = mu1;
        out_y_rec_diag_main[i] = y_px_loc + dx; // main diag: dj = di
        out_mdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_main[i]);
        out_mdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_main[i]);
        if (saveDiagParamA) out_d1_A[i] = A1; if (saveDiagParamMu) out_d1_mu[i] = mu1; if (saveDiagParamSigma) out_d1_sigma[i] = S1; if (saveDiagParamB) out_d1_B[i] = B1;
        if (saveOutlierMask) out_d1_mask[i] = std::move(maskD1);
      }
      if (ok2) {
        const double dx = mu2 - x_px_loc;
        out_x_rec_diag_sec[i] = mu2;
        out_y_rec_diag_sec[i] = y_px_loc - dx; // secondary diag: dj = -di
        out_sdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_sec[i]);
        out_sdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_sec[i]);
        if (saveDiagParamA) out_d2_A[i] = A2; if (saveDiagParamMu) out_d2_mu[i] = mu2; if (saveDiagParamSigma) out_d2_sigma[i] = S2; if (saveDiagParamB) out_d2_B[i] = B2;
        if (saveOutlierMask) out_d2_mask[i] = std::move(maskD2);
      }
    }
  }, indices);
  // Restore previous error level
  gErrorIgnoreLevel = prevErrorLevel_processing2D;

  // Sequentially write outputs to the tree (thread-safe)
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i); // ensure correct entry numbering for branch fill
    x_rec_2d = out_x_rec[i];
    y_rec_2d = out_y_rec[i];
    rec_hit_delta_x_2d = out_dx[i];
    rec_hit_delta_y_2d = out_dy[i];
    rec_hit_delta_x_2d_signed = out_dx_s[i];
    rec_hit_delta_y_2d_signed = out_dy_s[i];
    gauss2d_row_a = out_row_A[i];
    gauss2d_row_mu = out_row_mu[i];
    gauss2d_row_sigma = out_row_sigma[i];
    gauss2d_row_b = out_row_B[i];
    gauss2d_col_a = out_col_A[i];
    gauss2d_col_mu = out_col_mu[i];
    gauss2d_col_sigma = out_col_sigma[i];
    gauss2d_col_b = out_col_B[i];
    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx_signed->Fill();
    br_dy_signed->Fill();
    if (br_row_A) br_row_A->Fill();
    if (br_row_mu) br_row_mu->Fill();
    if (br_row_sigma) br_row_sigma->Fill();
    if (br_row_B) br_row_B->Fill();
    if (br_col_A) br_col_A->Fill();
    if (br_col_mu) br_col_mu->Fill();
    if (br_col_sigma) br_col_sigma->Fill();
    if (br_col_B) br_col_B->Fill();
    if (saveOutlierMask) {
      row_mask = out_row_mask[i];
      col_mask = out_col_mask[i];
      if (br_row_mask) br_row_mask->Fill();
      if (br_col_mask) br_col_mask->Fill();
    }
    if (fitDiagonals) {
      x_rec_diag_main = out_x_rec_diag_main.empty() ? INVALID_VALUE : out_x_rec_diag_main[i];
      y_rec_diag_main = out_y_rec_diag_main.empty() ? INVALID_VALUE : out_y_rec_diag_main[i];
      x_rec_diag_sec  = out_x_rec_diag_sec.empty()  ? INVALID_VALUE : out_x_rec_diag_sec[i];
      y_rec_diag_sec  = out_y_rec_diag_sec.empty()  ? INVALID_VALUE : out_y_rec_diag_sec[i];
      if (br_x_rec_diag_main) br_x_rec_diag_main->Fill();
      if (br_y_rec_diag_main) br_y_rec_diag_main->Fill();
      if (br_x_rec_diag_sec)  br_x_rec_diag_sec->Fill();
      if (br_y_rec_diag_sec)  br_y_rec_diag_sec->Fill();
      mdiag_dx_signed = out_mdiag_dx_s.empty()? INVALID_VALUE : out_mdiag_dx_s[i];
      mdiag_dy_signed = out_mdiag_dy_s.empty()? INVALID_VALUE : out_mdiag_dy_s[i];
      sdiag_dx_signed = out_sdiag_dx_s.empty()? INVALID_VALUE : out_sdiag_dx_s[i];
      sdiag_dy_signed = out_sdiag_dy_s.empty()? INVALID_VALUE : out_sdiag_dy_s[i];
      if (br_mdiag_dx_signed) br_mdiag_dx_signed->Fill();
      if (br_mdiag_dy_signed) br_mdiag_dy_signed->Fill();
      if (br_sdiag_dx_signed) br_sdiag_dx_signed->Fill();
      if (br_sdiag_dy_signed) br_sdiag_dy_signed->Fill();
      gaussD1_a = out_d1_A.empty()? INVALID_VALUE : out_d1_A[i];
      gaussD1_mu = out_d1_mu.empty()? INVALID_VALUE : out_d1_mu[i];
      gaussD1_sigma = out_d1_sigma.empty()? INVALID_VALUE : out_d1_sigma[i];
      gaussD1_b = out_d1_B.empty()? INVALID_VALUE : out_d1_B[i];
      gaussD2_a = out_d2_A.empty()? INVALID_VALUE : out_d2_A[i];
      gaussD2_mu = out_d2_mu.empty()? INVALID_VALUE : out_d2_mu[i];
      gaussD2_sigma = out_d2_sigma.empty()? INVALID_VALUE : out_d2_sigma[i];
      gaussD2_b = out_d2_B.empty()? INVALID_VALUE : out_d2_B[i];
      if (br_d1_A && saveDiagParamA) br_d1_A->Fill();
      if (br_d1_mu && saveDiagParamMu) br_d1_mu->Fill();
      if (br_d1_sigma && saveDiagParamSigma) br_d1_sigma->Fill();
      if (br_d1_B && saveDiagParamB) br_d1_B->Fill();
      if (br_d2_A && saveDiagParamA) br_d2_A->Fill();
      if (br_d2_mu && saveDiagParamMu) br_d2_mu->Fill();
      if (br_d2_sigma && saveDiagParamSigma) br_d2_sigma->Fill();
      if (br_d2_B && saveDiagParamB) br_d2_B->Fill();
      if (saveOutlierMask) {
        // Persist diagonal masks if we created branches; reuse ensure/reset above for row/col (not created here)
        // To keep file size in check, we do not create separate branches unless needed later.
      }
    }
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

  ::Info("processing2D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
  return 0;
}

