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

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

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
int processing2D(const char* filename = "../build/epicChargeSharingOutput.root",
                 double errorPercentOfMax = 5.0) {
  // Use Minuit2 by default
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
  // Slightly relax tolerance for speed, preserving accuracy
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-5);
  // Reduce per-fit effort to speed up processing
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(200);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("processing2D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first to provide clearer error when wrong file is passed)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharingOutput.root?)", filename);
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
    t->SetBranchAddress("x_px", &x_px_tmp);
    t->SetBranchAddress("y_px", &y_px_tmp);
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
  std::vector<double>* Qi = nullptr; // retained for compatibility but unused

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("x_hit", 1);
  tree->SetBranchStatus("y_hit", 1);
  tree->SetBranchStatus("x_px", 1);
  tree->SetBranchStatus("y_px", 1);
  tree->SetBranchStatus("is_pixel_hit", 1);
  tree->SetBranchStatus("F_i", 1);
  tree->SetBranchStatus("Q_i", 1); // kept for compatibility

  tree->SetBranchAddress("x_hit", &x_hit);
  tree->SetBranchAddress("y_hit", &y_hit);
  tree->SetBranchAddress("x_px", &x_px);
  tree->SetBranchAddress("y_px", &y_px);
  tree->SetBranchAddress("is_pixel_hit", &is_pixel_hit);
  tree->SetBranchAddress("F_i", &Fi);
  tree->SetBranchAddress("Q_i", &Qi);

  // New branches (outputs).
  // Use NaN sentinel so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double x_rec_2d = INVALID_VALUE;
  double y_rec_2d = INVALID_VALUE;
  double rec_hit_delta_x_2d = INVALID_VALUE;
  double rec_hit_delta_y_2d = INVALID_VALUE;
  // Also store signed deltas (not absolute): (hit - rec)
  double rec_hit_signed_delta_x_2d = INVALID_VALUE;
  double rec_hit_signed_delta_y_2d = INVALID_VALUE;

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

  TBranch* br_x_rec = ensureAndResetBranch("x_rec_2d", &x_rec_2d);
  TBranch* br_y_rec = ensureAndResetBranch("y_rec_2d", &y_rec_2d);
  TBranch* br_dx    = ensureAndResetBranch("rec_hit_delta_x_2d", &rec_hit_delta_x_2d);
  TBranch* br_dy    = ensureAndResetBranch("rec_hit_delta_y_2d", &rec_hit_delta_y_2d);
  TBranch* br_sdx   = ensureAndResetBranch("rec_hit_signed_delta_x_2d", &rec_hit_signed_delta_x_2d);
  TBranch* br_sdy   = ensureAndResetBranch("rec_hit_signed_delta_y_2d", &rec_hit_signed_delta_y_2d);

  // Fitting function for 1D gaussian + const
  TF1 fRow("fRow", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol", GaussPlusB, -1e9, 1e9, 4);

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0, nFitted = 0;

  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    nProcessed++;

    // Default outputs: set to finite invalid sentinel (not NaN)
    x_rec_2d = y_rec_2d = INVALID_VALUE;
    rec_hit_delta_x_2d = rec_hit_delta_y_2d = INVALID_VALUE;

    // Only attempt fit for non-pixel-pad hits and valid vectors
    if (is_pixel_hit || !Fi || Fi->empty()) {
      // Fill outputs for this entry
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_sdx->Fill();
      br_sdy->Fill();
      
      continue;
    }

    const size_t total = Fi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      // malformed neighborhood
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_sdx->Fill();
      br_sdy->Fill();
      
      continue;
    }

    const int R = (N - 1) / 2;

    // Build central row (di=0) and central column (dj=0) datasets
    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);

    // Track maximum valid charge in the full neighborhood for error scaling
    double qmaxNeighborhood = -1e300;

    // Mapping note: di indexes X (i), dj indexes Y (j)
    // - Central ROW: fixed Y (dj==0), vary di across X → x = x_px + di*pixelSpacing
    // - Central COLUMN: fixed X (di==0), vary dj across Y → y = y_px + dj*pixelSpacing
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int irow = di + R;
        const int jcol = dj + R;
        const int idx  = irow * N + jcol;
        const double q = (*Fi)[idx];

        // skip clearly invalid entries (negative sentinel or non-finite)
        if (!IsFinite(q) || q < 0) continue;

        if (q > qmaxNeighborhood) qmaxNeighborhood = q;

        // Central row: vary X with di at fixed Y
        if (dj == 0) {
          const double x = x_px + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
        }
        // Central column: vary Y with dj at fixed X
        if (di == 0) {
          const double y = y_px + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
        }
      }
    }

    // Sanity: need at least 3 points to fit 1D gaussian
    if (x_row.size() < 3 || y_col.size() < 3) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_sdx->Fill();
      br_sdy->Fill();
      continue;
    }

    // Build graphs with optional vertical uncertainties
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    // Initial parameters (robust seeding from maxima)
    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    const double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    const double B0_row = std::max(0.0, *minmaxRow.first);
    const double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    const double B0_col = std::max(0.0, *minmaxCol.first);

    // Very low contrast: fall back to fast weighted centroids (skip fit)
    if (A0_row < 1e-6 && A0_col < 1e-6) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k]); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k]); wsumy += w; yw += w * y_col[k]; }
      if (wsumx > 0 && wsumy > 0) {
        x_rec_2d = xw / wsumx;
        y_rec_2d = yw / wsumy;
        rec_hit_delta_x_2d = std::abs(x_hit - x_rec_2d);
        rec_hit_delta_y_2d = std::abs(y_hit - y_rec_2d);
        rec_hit_signed_delta_x_2d = (x_hit - x_rec_2d);
        rec_hit_signed_delta_y_2d = (y_hit - y_rec_2d);
        
      }
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_sdx->Fill();
      br_sdy->Fill();
      
      continue;
    }

    // mu guess from maximum
    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    // Fit on the FULL data ranges (no trimming)
    const std::vector<double>& x_row_fit = x_row;
    const std::vector<double>& q_row_fit = q_row;
    const std::vector<double>& y_col_fit = y_col;
    const std::vector<double>& q_col_fit = q_col;

    fRow.SetParameters(A0_row, mu0_row, std::max(0.25*pixelSpacing, 1e-6), B0_row);
    fCol.SetParameters(A0_col, mu0_col, std::max(0.25*pixelSpacing, 1e-6), B0_col);

    // Restrict function ranges to the full data span
    auto minmaxX = std::minmax_element(x_row_fit.begin(), x_row_fit.end());
    auto minmaxY = std::minmax_element(y_col_fit.begin(), y_col_fit.end());
    const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
    const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
    const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
    const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
    fRow.SetRange(xMin, xMax);
    fCol.SetRange(yMin, yMax);
    // Constrain means to ±1/2 pitch about nearest pixel center
    const double muXLo = x_px - 0.5 * pixelSpacing;
    const double muXHi = x_px + 0.5 * pixelSpacing;
    const double muYLo = y_px - 0.5 * pixelSpacing;
    const double muYHi = y_px + 0.5 * pixelSpacing;
    fRow.SetParLimits(1, muXLo, muXHi);
    fCol.SetParLimits(1, muYLo, muYHi);

    // Fractions are unitless and bounded [0,1]
    fRow.SetParLimits(0, 0.0, 1.0);                   // A >= 0
    fRow.SetParLimits(2, std::max(1e-6, 0.05*pixelSpacing), 2.0*pixelSpacing); // tighter sigma range
    // Ensure non-negative baseline
    fRow.SetParLimits(3, 0.0, std::max(0.0, *minmaxRow.first));

    fCol.SetParLimits(0, 0.0, 1.0);
    fCol.SetParLimits(2, std::max(1e-6, 0.05*pixelSpacing), 2.0*pixelSpacing);
    // Ensure non-negative baseline
    fCol.SetParLimits(3, 0.0, std::max(0.0, *minmaxCol.first));

    // Minuit2 least-squares on compact windows (faster/more stable)
    auto chi2Row = [&](const double* p) -> double {
      const double A = p[0], mu = p[1], sig = p[2], B = p[3];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (size_t k=0;k<q_row_fit.size();++k) {
        const double dx = (x_row_fit[k] - mu)/sig;
        const double model = A * std::exp(-0.5*dx*dx) + B;
        const double r = (q_row_fit[k] - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };
    auto chi2Col = [&](const double* p) -> double {
      const double A = p[0], mu = p[1], sig = p[2], B = p[3];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (size_t k=0;k<q_col_fit.size();++k) {
        const double dy = (y_col_fit[k] - mu)/sig;
        const double model = A * std::exp(-0.5*dy*dy) + B;
        const double r = (q_col_fit[k] - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };

    std::unique_ptr<ROOT::Math::Minimizer> mRow(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
    std::unique_ptr<ROOT::Math::Minimizer> mCol(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
    mRow->SetMaxFunctionCalls(200);
    mCol->SetMaxFunctionCalls(200);
    mRow->SetTolerance(1e-5);
    mCol->SetTolerance(1e-5);

    ROOT::Math::Functor fRowChi2(chi2Row, 4);
    ROOT::Math::Functor fColChi2(chi2Col, 4);
    mRow->SetFunction(fRowChi2);
    mCol->SetFunction(fColChi2);

    const double sigInit = std::max(0.25*pixelSpacing, 1e-6);
    const double sigLo   = std::max(1e-6, 0.05*pixelSpacing);
    const double sigHi   = 2.0*pixelSpacing;

    mRow->SetLimitedVariable(0, "A", A0_row, 1e-3, 0.0, 1.0);
    mRow->SetLimitedVariable(1, "mu", mu0_row, 1e-4*pixelSpacing, xMin, xMax);
    mRow->SetLimitedVariable(2, "sigma", sigInit, 1e-4*pixelSpacing, sigLo, sigHi);
    mRow->SetLimitedVariable(3, "B", B0_row, 1e-3, 0.0, std::max(0.0, *minmaxRow.first));

    mCol->SetLimitedVariable(0, "A", A0_col, 1e-3, 0.0, 1.0);
    mCol->SetLimitedVariable(1, "mu", mu0_col, 1e-4*pixelSpacing, yMin, yMax);
    mCol->SetLimitedVariable(2, "sigma", sigInit, 1e-4*pixelSpacing, sigLo, sigHi);
    mCol->SetLimitedVariable(3, "B", B0_col, 1e-3, 0.0, std::max(0.0, *minmaxCol.first));

    bool okRow = mRow->Minimize();
    bool okCol = mCol->Minimize();

    double muX = okRow ? mRow->X()[1] : NAN;
    double muY = okCol ? mCol->X()[1] : NAN;

    // Fallback: weighted centroid if fit did not converge
    if (!okRow) {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row_fit.size();++k) { double w = std::max(0.0, q_row_fit[k]); wsum += w; xw += w * x_row_fit[k]; }
      if (wsum > 0) { muX = xw / wsum; okRow = true; }
    }
    if (!okCol) {
      double wsum = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col_fit.size();++k) { double w = std::max(0.0, q_col_fit[k]); wsum += w; yw += w * y_col_fit[k]; }
      if (wsum > 0) { muY = yw / wsum; okCol = true; }
    }

    if (okRow && okCol && IsFinite(muX) && IsFinite(muY)) {
      x_rec_2d = muX;
      y_rec_2d = muY;
      rec_hit_delta_x_2d = std::abs(x_hit - x_rec_2d);
      rec_hit_delta_y_2d = std::abs(y_hit - y_rec_2d);
      rec_hit_signed_delta_x_2d = (x_hit - x_rec_2d);
      rec_hit_signed_delta_y_2d = (y_hit - y_rec_2d);
      
      nFitted++;
    } else {
      x_rec_2d = y_rec_2d = INVALID_VALUE;
      rec_hit_delta_x_2d = rec_hit_delta_y_2d = INVALID_VALUE;
      rec_hit_signed_delta_x_2d = rec_hit_signed_delta_y_2d = INVALID_VALUE;
      
    }

    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx->Fill();
    br_dy->Fill();
    br_sdx->Fill();
    br_sdy->Fill();
    
  }

  // Overwrite tree in file
  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();
  delete file;

  ::Info("processing2D", "Processed %lld entries, fitted %lld.", nProcessed, nFitted);
  return 0;
}

