// ROOT macro: processing2D.C
// Performs 1D Gaussian fits on central row and column of the charge neighborhood
// to reconstruct (x_rec_2d, y_rec_2d) and deltas, and appends them as new branches.

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
                 bool saveFitParameters = false) {
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

  // Fetch metadata (pixel spacing) with fallback to inference from x_px/y_px
  double pixelSpacing = NAN;
  if (auto* spacingObj = dynamic_cast<TNamed*>(file->Get("GridPixelSpacing_mm"))) {
    try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) {}
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

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
  }
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing2D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 2;
  }

  // Existing branches (inputs)
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  std::vector<double>* Fi = nullptr; // used for fits (fractions 0..1)

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("TrueX", 1);
  tree->SetBranchStatus("TrueY", 1);
  tree->SetBranchStatus("PixelX", 1);
  tree->SetBranchStatus("PixelY", 1);
  tree->SetBranchStatus("isPixelHit", 1);
  tree->SetBranchStatus("F_i", 1);

  tree->SetBranchAddress("TrueX", &x_hit);
  tree->SetBranchAddress("TrueY", &y_hit);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
  tree->SetBranchAddress("F_i", &Fi);

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
  TBranch* br_dx    = ensureAndResetBranch("ReconTrueDeltaX", &rec_hit_delta_x_2d);
  TBranch* br_dy    = ensureAndResetBranch("ReconTrueDeltaY", &rec_hit_delta_y_2d);
  TBranch* br_dx_signed = ensureAndResetBranch("ReconTrueDeltaX_Signed", &rec_hit_delta_x_2d_signed);
  TBranch* br_dy_signed = ensureAndResetBranch("ReconTrueDeltaY_Signed", &rec_hit_delta_y_2d_signed);
  // Parameter branches
  TBranch* br_row_A = nullptr;
  TBranch* br_row_mu = nullptr;
  TBranch* br_row_sigma = nullptr;
  TBranch* br_row_B = nullptr;
  TBranch* br_col_A = nullptr;
  TBranch* br_col_mu = nullptr;
  TBranch* br_col_sigma = nullptr;
  TBranch* br_col_B = nullptr;
  if (saveFitParameters) {
    br_row_A     = ensureAndResetBranch("Gauss2D_Row_A", &gauss2d_row_a);
    br_row_mu    = ensureAndResetBranch("Gauss2D_Row_mu", &gauss2d_row_mu);
    br_row_sigma = ensureAndResetBranch("Gauss2D_Row_sigma", &gauss2d_row_sigma);
    br_row_B     = ensureAndResetBranch("Gauss2D_Row_B", &gauss2d_row_b);
    br_col_A     = ensureAndResetBranch("Gauss2D_Col_A", &gauss2d_col_a);
    br_col_mu    = ensureAndResetBranch("Gauss2D_Col_mu", &gauss2d_col_mu);
    br_col_sigma = ensureAndResetBranch("Gauss2D_Col_sigma", &gauss2d_col_sigma);
    br_col_B     = ensureAndResetBranch("Gauss2D_Col_B", &gauss2d_col_b);
  }

  // Fitting function for 1D gaussian + const
  TF1 fRow("fRow", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol", GaussPlusB, -1e9, 1e9, 4);

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0;
  std::atomic<long long> nFitted{0};

  // Preload inputs sequentially to avoid ROOT I/O races
  std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
  std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
  std::vector<char> v_is_pixel(nEntries);
  std::vector<std::vector<double>> v_Fi(nEntries);
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    v_x_hit[i] = x_hit;
    v_y_hit[i] = y_hit;
    v_x_px[i]  = x_px;
    v_y_px[i]  = y_px;
    v_is_pixel[i] = is_pixel_hit ? 1 : 0;
    if (Fi && !Fi->empty()) v_Fi[i] = *Fi; else v_Fi[i].clear();
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

  // Parallel computation over entries
  std::vector<int> indices(nEntries);
  std::iota(indices.begin(), indices.end(), 0);
  ROOT::TThreadExecutor exec; // uses ROOT IMT pool size by default
  exec.Foreach([&](int i){
    const bool isPix = v_is_pixel[i] != 0;
    const auto &FiLoc = v_Fi[i];
    if (isPix || FiLoc.empty()) {
      return;
    }

    const size_t total = FiLoc.size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
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
        const double q = FiLoc[idx];
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
      return;
    }

    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    const double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    const double B0_row = std::max(0.0, *minmaxRow.first);
    const double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    const double B0_col = std::max(0.0, *minmaxCol.first);

    // Low-contrast: fast centroid
    if (A0_row < 1e-6 && A0_col < 1e-6) {
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
      }
      return;
    }

    // Seeds
    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    const std::vector<double>& x_row_fit = x_row;
    const std::vector<double>& q_row_fit = q_row;
    const std::vector<double>& y_col_fit = y_col;
    const std::vector<double>& q_col_fit = q_col;

    const double sigLoBound = std::max(1e-6, 0.02*pixelSpacing);
    const double sigHiBound = 3.0*pixelSpacing;
    auto sigmaSeed1D = [&](const std::vector<double>& xs, const std::vector<double>& qs, double B0)->double {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); wsum += w; xw += w * xs[k]; }
      if (wsum <= 0.0) return std::max(0.25*pixelSpacing, 1e-6);
      const double mean = xw / wsum;
      double var = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); const double dx = xs[k] - mean; var += w * dx * dx; }
      var = (wsum > 0.0) ? (var / wsum) : 0.0;
      double s = std::sqrt(std::max(var, 1e-12));
      if (s < sigLoBound) s = sigLoBound;
      if (s > sigHiBound) s = sigHiBound;
      return s;
    };
    const double sigInitRow = sigmaSeed1D(x_row, q_row, B0_row);
    const double sigInitCol = sigmaSeed1D(y_col, q_col, B0_col);

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
    fRowLoc.SetParLimits(0, 0.0, 1.0);
    fRowLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fRowLoc.SetParLimits(3, 0.0, std::max(0.0, *minmaxRow.first));
    fColLoc.SetParLimits(0, 0.0, 1.0);
    fColLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fColLoc.SetParLimits(3, 0.0, std::max(0.0, *minmaxCol.first));

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
    fitRow.Config().ParSettings(0).SetLowerLimit(0.0);
    fitRow.Config().ParSettings(1).SetLimits(muXLo, muXHi);
    fitRow.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    fitRow.Config().ParSettings(3).SetLowerLimit(0.0);
    fitCol.Config().ParSettings(0).SetLowerLimit(0.0);
    fitCol.Config().ParSettings(1).SetLimits(muYLo, muYHi);
    fitCol.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    fitCol.Config().ParSettings(3).SetLowerLimit(0.0);
    fitRow.Config().ParSettings(0).SetStepSize(1e-3);
    fitRow.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(3).SetStepSize(1e-3);
    fitCol.Config().ParSettings(0).SetStepSize(1e-3);
    fitCol.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(3).SetStepSize(1e-3);
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
  }, indices);

  // Sequentially write outputs to the tree (thread-safe)
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i); // ensure correct entry numbering for branch fill
    x_rec_2d = out_x_rec[i];
    y_rec_2d = out_y_rec[i];
    rec_hit_delta_x_2d = out_dx[i];
    rec_hit_delta_y_2d = out_dy[i];
    rec_hit_delta_x_2d_signed = out_dx_s[i];
    rec_hit_delta_y_2d_signed = out_dy_s[i];
    if (saveFitParameters) {
      gauss2d_row_a = out_row_A[i];
      gauss2d_row_mu = out_row_mu[i];
      gauss2d_row_sigma = out_row_sigma[i];
      gauss2d_row_b = out_row_B[i];
      gauss2d_col_a = out_col_A[i];
      gauss2d_col_mu = out_col_mu[i];
      gauss2d_col_sigma = out_col_sigma[i];
      gauss2d_col_b = out_col_B[i];
    }
    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx->Fill();
    br_dy->Fill();
    br_dx_signed->Fill();
    br_dy_signed->Fill();
    if (saveFitParameters) {
      if (br_row_A) br_row_A->Fill();
      if (br_row_mu) br_row_mu->Fill();
      if (br_row_sigma) br_row_sigma->Fill();
      if (br_row_B) br_row_B->Fill();
      if (br_col_A) br_col_A->Fill();
      if (br_col_mu) br_col_mu->Fill();
      if (br_col_sigma) br_col_sigma->Fill();
      if (br_col_B) br_col_B->Fill();
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

