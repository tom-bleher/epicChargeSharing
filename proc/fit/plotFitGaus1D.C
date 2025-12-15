#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TList.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TROOT.h>
#include <TError.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>
#include <TLine.h>
#include <TBox.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TSystem.h>
#include <Math/MinimizerOptions.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

#include <Fit/Fitter.h>
#include <Fit/BinData.h>
#include <Math/WrappedMultiTF1.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

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

  // Helper function to read metadata from tree UserInfo
  double GetDoubleMetadata(TTree* tree, const char* key) {
    if (tree) {
      TList* info = tree->GetUserInfo();
      if (info) {
        if (auto* param = dynamic_cast<TParameter<double>*>(info->FindObject(key))) {
          return param->GetVal();
        }
      }
    }
    return NAN;
  }

  int GetIntMetadata(TTree* tree, const char* key) {
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
}

// processing2D_plots: Same logic as processing2D_plots, but draws
// column (left) and row (right) on a single 1800x700 canvas per page.
// Output: row_column.pdf (multi-page)
int processing2D_plots(const char* filename = "../build/epicChargeSharing.root",
                           double errorPercentOfMax = 5.0,
                           Long64_t nRandomEvents = 100) {
  // Match processing2D configuration exactly
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for read (try provided path, then fallback to ../.. /build relative to this file)
  TFile* file = TFile::Open(filename, "READ");
  if (!file || file->IsZombie()) {
    if (file) { file->Close(); delete file; file = nullptr; }
    TString macroDir = gSystem->DirName(__FILE__);
    TString fallback = macroDir + "/../../build/epicChargeSharing.root";
    file = TFile::Open(fallback, "READ");
    if (!file || file->IsZombie()) {
      ::Error("processing2D_plots", "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return 1;
    }
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D_plots", "Hits tree not found in file: %s", filename);
    file->Close();
    delete file;
    return 2;
  }

  // Pixel spacing/size/radius: prefer metadata (file-level or tree UserInfo); fallback to inference/defaults
  double pixelSpacing = GetDoubleMetadata(tree, "GridPixelSpacing_mm");
  double pixelSize    = GetDoubleMetadata(tree, "GridPixelSize_mm");
  int neighborhoodRadiusMeta = GetIntMetadata(tree, "NeighborhoodRadius");

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

  auto inferRadiusFromTree = [&](TTree* t) -> int {
    std::vector<double>* Qi_tmp = nullptr;
    if (t->GetBranch("Qi")) {
      t->SetBranchAddress("Qi", &Qi_tmp);
    }
    Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
    for (Long64_t i=0;i<nToScan;++i) {
      t->GetEntry(i);
      if (Qi_tmp && !Qi_tmp->empty()) {
        const size_t total = Qi_tmp->size();
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
    ::Error("processing2D_plots", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  // Pixel size: if not available in metadata, fall back to 0.5 * pitch (match processing2D)
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.5 * pixelSpacing; // mm
  }
  if (neighborhoodRadiusMeta <= 0) {
    neighborhoodRadiusMeta = inferRadiusFromTree(tree);
  }

  // Set up branches
  double x_true = 0.0, y_true = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Qi = nullptr; // use induced charge for fitting

  tree->SetBranchAddress("TrueX", &x_true);
  tree->SetBranchAddress("TrueY", &y_true);
  tree->SetBranchAddress("PixelX",  &x_px);
  tree->SetBranchAddress("PixelY",  &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_true);
  if (tree->GetBranch("Qi")) {
    tree->SetBranchAddress("Qi", &Qi);
  }

  // Prepare plotting objects
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  TCanvas c("c2up", "processing2D fits", 1800, 700);
  // Create two pads so each panel matches the original 900x700 size
  TPad pL("pL", "column-left", 0.0, 0.0, 0.5, 1.0);
  TPad pR("pR", "row-right",  0.5, 0.0, 1.0, 1.0);
  pL.SetTicks(1,1);
  pR.SetTicks(1,1);
  pL.Draw();
  pR.Draw();

  // Open multipage PDF
  c.Print("2Dfits.pdf[");

  // Functions for visualization (parameters will be set from the new fitter)
  TF1 fRow("fRow", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol", GaussPlusB, -1e9, 1e9, 4);

  const Long64_t nEntries = tree->GetEntries();
  std::vector<Long64_t> indices; indices.reserve(nEntries);
  for (Long64_t ii = 0; ii < nEntries; ++ii) indices.push_back(ii);
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::shuffle(indices.begin(), indices.end(), rng);
  Long64_t nPages = 0, nConsidered = 0;
  const Long64_t targetPages = (nRandomEvents < 0) ? 0 : nRandomEvents;

  for (Long64_t sampleIdx = 0; sampleIdx < nEntries && nPages < targetPages; ++sampleIdx) {
    const Long64_t eventIndex = indices[sampleIdx];
    tree->GetEntry(eventIndex);

    // Only non-pixel-pad trues with valid neighborhood (Qi present)
    if (is_pixel_true || !Qi || Qi->empty()) continue;

    const size_t total = Qi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Diagnostics: verify geometric non-pixel condition max(|dx|,|dy|) > size/2
    const double dxAbs = std::abs(x_true - x_px);
    const double dyAbs = std::abs(y_true - y_px);
    const double halfPixel = 0.5 * pixelSize;
    const double maxAbs = std::max(dxAbs, dyAbs);
    if (!(maxAbs > halfPixel)) {
      ::Warning("processing2D_plots",
                "Event %lld marked non-pixel but max(|dx|,|dy|)=%.3f \u00b5m <= halfPixel=%.3f \u00b5m (dx=%.3f \u00b5m, dy=%.3f \u00b5m).",
                eventIndex, 1000.0*maxAbs, 1000.0*halfPixel, 1000.0*dxAbs, 1000.0*dyAbs);
    }

    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);

    // Track maximum valid charge in the full neighborhood for error scaling
    double qmaxNeighborhood = -1e300;

    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int irow = di + R;
        const int jcol = dj + R;
        const int idx  = irow * N + jcol;
        const double q = (*Qi)[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        // Correct mapping: di moves along X, dj moves along Y
        if (dj == 0) { // central row (vary X at fixed Y)
          const double x = x_px + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
        }
        if (di == 0) { // central column (vary Y at fixed X)
          const double y = y_px + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
        }
      }
    }

    // Match processing2D.C: require BOTH row and column to have >=3 points
    if (x_row.size() < 3 || y_col.size() < 3) continue;
    nConsidered++;

    // Optional uniform vertical uncertainty from percent-of-max
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    // Build graphs (weighted or unweighted) for BOTH row and column
    bool useWeights = (uniformSigma > 0.0);
    // Row graph
    TGraph* baseRowPtr = nullptr;
    TGraph* baseColPtr = nullptr;
    TGraph gRowPlain;
    TGraph gColPlain;
    TGraphErrors gRowErr;
    TGraphErrors gColErr;
    if (useWeights) {
      gRowErr = TGraphErrors(static_cast<int>(x_row.size()));
      for (int k = 0; k < gRowErr.GetN(); ++k) {
        gRowErr.SetPoint(k, x_row[k], q_row[k]);
        gRowErr.SetPointError(k, 0.0, uniformSigma);
      }
      baseRowPtr = &gRowErr;
      gColErr = TGraphErrors(static_cast<int>(y_col.size()));
      for (int k = 0; k < gColErr.GetN(); ++k) {
        gColErr.SetPoint(k, y_col[k], q_col[k]);
        gColErr.SetPointError(k, 0.0, uniformSigma);
      }
      baseColPtr = &gColErr;
    } else {
      gRowPlain = TGraph(static_cast<int>(x_row.size()));
      for (int k = 0; k < gRowPlain.GetN(); ++k) gRowPlain.SetPoint(k, x_row[k], q_row[k]);
      baseRowPtr = &gRowPlain;
      gColPlain = TGraph(static_cast<int>(y_col.size()));
      for (int k = 0; k < gColPlain.GetN(); ++k) gColPlain.SetPoint(k, y_col[k], q_col[k]);
      baseColPtr = &gColPlain;
    }

    // Set titles and styles
    baseRowPtr->SetTitle(Form("Event %lld: Central row fit; x [mm]; Induced charge Q_{i} [C]", eventIndex));
    baseRowPtr->SetMarkerStyle(20);
    baseRowPtr->SetMarkerSize(0.9);
    baseRowPtr->SetLineColor(kBlue+1);

    baseColPtr->SetTitle(Form("Event %lld: Central column fit; y [mm]; Induced charge Q_{i} [C]", eventIndex));
    baseColPtr->SetMarkerStyle(21);
    baseColPtr->SetMarkerSize(0.9);
    baseColPtr->SetLineColor(kBlue+2);

    // New fitter logic (mirror of processing2D.C)
    // Seed parameters from min/max
    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    const double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    const double B0_row = std::max(0.0, *minmaxRow.first);
    const double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    const double B0_col = std::max(0.0, *minmaxCol.first);

    bool haveFitRow = false, haveFitCol = false;
    double muRow = NAN, muCol = NAN;
    double A_row = A0_row, B_row = B0_row, S_row = std::max(0.25*pixelSpacing, 1e-6);
    double A_col = A0_col, B_col = B0_col, S_col = std::max(0.25*pixelSpacing, 1e-6);

    // Very low contrast: fallback to fast weighted centroids (skip fit), baseline-subtracted
    const double contrastEps = (qmaxNeighborhood > 0.0) ? (1e-3 * qmaxNeighborhood) : 0.0;
    if (qmaxNeighborhood > 0.0 && A0_row < contrastEps && A0_col < contrastEps) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k] - B0_row); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k] - B0_col); wsumy += w; yw += w * y_col[k]; }
      if (wsumx > 0 && wsumy > 0) {
        muRow = xw / wsumx;
        muCol = yw / wsumy;
        haveFitRow = haveFitCol = true; // we have reconstructions, but will not draw model curves
      }
    } else {
      // mu guess from maximum index
      int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
      int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
      double mu0_row = x_row[idxMaxRow];
      double mu0_col = y_col[idxMaxCol];

      // Use FULL data ranges (no trimming) for visualization fits
      const std::vector<double>& x_row_fit = x_row;
      const std::vector<double>& q_row_fit = q_row;
      const std::vector<double>& y_col_fit = y_col;
      const std::vector<double>& q_col_fit = q_col;

      auto minmaxX = std::minmax_element(x_row_fit.begin(), x_row_fit.end());
      auto minmaxY = std::minmax_element(y_col_fit.begin(), y_col_fit.end());
      const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
      const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
      const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
      const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
      // Tighten mu bounds to  b11/2 pitch about nearest pixel center (match processing2D)
      const double muXLo = x_px - 0.5 * pixelSpacing;
      const double muXHi = x_px + 0.5 * pixelSpacing;
      const double muYLo = y_px - 0.5 * pixelSpacing;
      const double muYHi = y_px + 0.5 * pixelSpacing;

      // Sigma seeding and bounds identical to processing2D.C
      const double sigLoBound = pixelSize;
      const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
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

      // Wrapped model functions for the fitter
      ROOT::Math::WrappedMultiTF1 wRow(fRow, 1);
      ROOT::Math::WrappedMultiTF1 wCol(fCol, 1);

      // Build BinData with uniform errors (or 1.0 if not provided), matching processing2D
      ROOT::Fit::BinData dataRow(static_cast<int>(x_row_fit.size()), 1);
      ROOT::Fit::BinData dataCol(static_cast<int>(y_col_fit.size()), 1);
      for (size_t k=0;k<x_row_fit.size();++k) {
        const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
        dataRow.Add(x_row_fit[k], q_row_fit[k], ey);
      }
      for (size_t k=0;k<y_col_fit.size();++k) {
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

      // Parameter settings and limits (match processing2D)
      fitRow.Config().ParSettings(0).SetName("A");
      fitRow.Config().ParSettings(1).SetName("mu");
      fitRow.Config().ParSettings(2).SetName("sigma");
      fitRow.Config().ParSettings(3).SetName("B");
      fitCol.Config().ParSettings(0).SetName("A");
      fitCol.Config().ParSettings(1).SetName("mu");
      fitCol.Config().ParSettings(2).SetName("sigma");
      fitCol.Config().ParSettings(3).SetName("B");

      // Bounds for Qi fits: A in (0, ~2*qmax], B in [0, ~qmax]
      const double AHi = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
      const double BHi = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
      fitRow.Config().ParSettings(0).SetLimits(1e-18, AHi);
      fitRow.Config().ParSettings(1).SetLimits(muXLo, muXHi);
      fitRow.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
      fitRow.Config().ParSettings(3).SetLimits(0.0, BHi);
      fitCol.Config().ParSettings(0).SetLimits(1e-18, AHi);
      fitCol.Config().ParSettings(1).SetLimits(muYLo, muYHi);
      fitCol.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
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

      // Seed values
      fitRow.Config().ParSettings(0).SetValue(A0_row);
      fitRow.Config().ParSettings(1).SetValue(mu0_row);
      fitRow.Config().ParSettings(2).SetValue(sigInitRow);
      fitRow.Config().ParSettings(3).SetValue(B0_row);
      fitCol.Config().ParSettings(0).SetValue(A0_col);
      fitCol.Config().ParSettings(1).SetValue(mu0_col);
      fitCol.Config().ParSettings(2).SetValue(sigInitCol);
      fitCol.Config().ParSettings(3).SetValue(B0_col);

      haveFitRow = fitRow.Fit(dataRow);
      haveFitCol = fitCol.Fit(dataCol);

      if (haveFitRow) {
        A_row = fitRow.Result().Parameter(0);
        muRow = fitRow.Result().Parameter(1);
        S_row = fitRow.Result().Parameter(2);
        B_row = fitRow.Result().Parameter(3);
        fRow.SetRange(xMin, xMax);
        fRow.SetParameters(A_row, muRow, S_row, B_row);
      } else {
        // Fallback: baseline-subtracted weighted centroid on full window
        double wsum = 0.0, xw = 0.0;
        for (size_t k=0;k<x_row_fit.size();++k) { double w = std::max(0.0, q_row_fit[k] - B0_row); wsum += w; xw += w * x_row_fit[k]; }
        if (wsum > 0) { muRow = xw / wsum; haveFitRow = true; }
      }

      if (haveFitCol) {
        A_col = fitCol.Result().Parameter(0);
        muCol = fitCol.Result().Parameter(1);
        S_col = fitCol.Result().Parameter(2);
        B_col = fitCol.Result().Parameter(3);
        fCol.SetRange(yMin, yMax);
        fCol.SetParameters(A_col, muCol, S_col, B_col);
      } else {
        // Fallback: baseline-subtracted weighted centroid on full window
        double wsum = 0.0, yw = 0.0;
        for (size_t k=0;k<y_col_fit.size();++k) { double w = std::max(0.0, q_col_fit[k] - B0_col); wsum += w; yw += w * y_col_fit[k]; }
        if (wsum > 0) { muCol = yw / wsum; haveFitCol = true; }
      }
    }

    // Only proceed when BOTH reconstructions are valid, matching processing2D.C behavior
    if (!(haveFitRow && haveFitCol) || !IsFinite(muRow) || !IsFinite(muCol)) continue;

    // Add headroom so curves are visible
    double dataMaxRow = *std::max_element(q_row.begin(), q_row.end());
    double yMaxRow = 1.20 * dataMaxRow;
    if (A_row > 0.0) {
      double fitPeakRow = A_row + B_row;
      yMaxRow = 1.20 * std::max(dataMaxRow, fitPeakRow);
    }
    baseRowPtr->SetMaximum(yMaxRow);

    double dataMaxCol = *std::max_element(q_col.begin(), q_col.end());
    double yMaxCol = 1.20 * dataMaxCol;
    if (A_col > 0.0) {
      double fitPeakCol = A_col + B_col;
      yMaxCol = 1.20 * std::max(dataMaxCol, fitPeakCol);
    }
    baseColPtr->SetMaximum(yMaxCol);

    // Left pad: COLUMN plot
    pL.cd();
    baseColPtr->Draw(useWeights ? "AP" : "AP");
    if (A_col > 0.0) {
      fCol.SetNpx(600);
      fCol.SetLineWidth(2);
      fCol.SetLineColor(kRed+1);
      fCol.Draw("L SAME");
    }
    gPad->Update();
    double yPadMinC = gPad->GetUymin();
    double yPadMaxC = gPad->GetUymax();
    // Draw horizontal pixel-width rectangles centered at each data point (outline only)
    {
      const double halfHc = 0.015 * (yPadMaxC - yPadMinC);
      for (size_t k = 0; k < y_col.size(); ++k) {
        const double xc = y_col[k]; // note: x-axis is Y coordinate here
        const double yc = q_col[k];
        const double xlo = xc - 0.5 * pixelSize;
        const double xhi = xc + 0.5 * pixelSize;
        const double ylo = yc - halfHc;
        const double yhi = yc + halfHc;
        TBox* box = new TBox(xlo, ylo, xhi, yhi);
        box->SetFillStyle(0);
        box->SetLineColor(kGray+2);
        box->SetLineWidth(1);
        box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
    }
    TLine lineYtrue(y_true, yPadMinC, y_true, yPadMaxC);
    lineYtrue.SetLineStyle(2);
    lineYtrue.SetLineWidth(2);
    lineYtrue.SetLineColor(kBlack);
    lineYtrue.Draw("SAME");
    double fx1c = gPad->GetLeftMargin();
    double fy1c = gPad->GetBottomMargin();
    double fx2c = 1.0 - gPad->GetRightMargin();
    double fy2c = 1.0 - gPad->GetTopMargin();
    // Increase legend size to accommodate extra entries
    double legWc = 0.28, legHc = 0.30, insetc = 0.008;
    double ly2c = fy2c - insetc, ly1c = ly2c - legHc;
    // Left legend (top-left): line indicators
    double lx1c = fx1c + insetc, lx2c = lx1c + legWc;
    TLegend legCLeft(lx1c, ly1c, lx2c, ly2c, "", "NDC");
    legCLeft.SetBorderSize(0);
    legCLeft.SetFillStyle(0);
    legCLeft.SetTextSize(0.03);
    legCLeft.AddEntry(&lineYtrue, "y_{true}", "l");
    TLine lineYrec(muCol, yPadMinC, muCol, yPadMaxC);
    lineYrec.SetLineStyle(2);
    lineYrec.SetLineWidth(2);
    lineYrec.SetLineColor(kRed+1);
    lineYrec.Draw("SAME");
    legCLeft.AddEntry(&lineYrec, "y_{rec}", "l");
    legCLeft.Draw();
    // Right legend (top-right): numeric values
    double rx2c = fx2c - insetc, rx1c = rx2c - legWc;
    TLegend legCRight(rx1c, ly1c, rx2c, ly2c, "", "NDC");
    legCRight.SetBorderSize(0);
    legCRight.SetFillStyle(0);
    legCRight.SetTextSize(0.03);
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true} = %.4f mm", y_true), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{rec} = %.4f mm", muCol), "");
    legCRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", S_col), "");
    // Show pixel-centered delta (|y_true - y_px|) and reconstruction delta
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{px} = %.1f #mum", 1000.0*(y_true - y_px)), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{rec} = %.1f #mum", 1000.0*(y_true - muCol)), "");
    legCRight.Draw();

    // Right pad: ROW plot
    pR.cd();
    baseRowPtr->Draw(useWeights ? "AP" : "AP");
    if (A_row > 0.0) {
      fRow.SetNpx(600);
      fRow.SetLineWidth(2);
      fRow.SetLineColor(kRed+1);
      fRow.Draw("L SAME");
    }
    gPad->Update();
    double yPadMin = gPad->GetUymin();
    double yPadMax = gPad->GetUymax();
    // Draw horizontal pixel-width rectangles centered at each data point (outline only)
    {
      const double halfH = 0.015 * (yPadMax - yPadMin);
      for (size_t k = 0; k < x_row.size(); ++k) {
        const double xc = x_row[k];
        const double yc = q_row[k];
        const double xlo = xc - 0.5 * pixelSize;
        const double xhi = xc + 0.5 * pixelSize;
        const double ylo = yc - halfH;
        const double yhi = yc + halfH;
        TBox* box = new TBox(xlo, ylo, xhi, yhi);
        box->SetFillStyle(0);
        box->SetLineColor(kGray+2);
        box->SetLineWidth(1);
        box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
    }
    TLine lineXtrue(x_true, yPadMin, x_true, yPadMax);
    lineXtrue.SetLineStyle(2);
    lineXtrue.SetLineWidth(2);
    lineXtrue.SetLineColor(kBlack);
    lineXtrue.Draw("SAME");
    double fx1 = gPad->GetLeftMargin();
    double fy1 = gPad->GetBottomMargin();
    double fx2 = 1.0 - gPad->GetRightMargin();
    double fy2 = 1.0 - gPad->GetTopMargin();
    // Increase legend size to accommodate extra entries
    double legW = 0.28, legH = 0.30, inset = 0.008;
    double ly2 = fy2 - inset, ly1 = ly2 - legH;
    // Left legend (top-left): line indicators
    double lx1 = fx1 + inset, lx2 = lx1 + legW;
    TLegend legLeft(lx1, ly1, lx2, ly2, "", "NDC");
    legLeft.SetBorderSize(0);
    legLeft.SetFillStyle(0);
    legLeft.SetTextSize(0.03);
    legLeft.AddEntry(&lineXtrue, "x_{true}", "l");
    TLine lineXrec(muRow, yPadMin, muRow, yPadMax);
    lineXrec.SetLineStyle(2);
    lineXrec.SetLineWidth(2);
    lineXrec.SetLineColor(kRed+1);
    lineXrec.Draw("SAME");
    legLeft.AddEntry(&lineXrec, "x_{rec}", "l");
    legLeft.Draw();
    // Right legend (top-right): numeric values
    double rx2 = fx2 - inset, rx1 = rx2 - legW;
    TLegend legRight(rx1, ly1, rx2, ly2, "", "NDC");
    legRight.SetBorderSize(0);
    legRight.SetFillStyle(0);
    legRight.SetTextSize(0.03);
    legRight.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{rec} = %.4f mm", muRow), "");
    legRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", S_row), "");
    // Show pixel-centered delta (|x_true - x_px|) and reconstruction delta
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{px} = %.1f #mum", 1000.0*(x_true - x_px)), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec} = %.1f #mum", 1000.0*(x_true - muRow)), "");
    legRight.Draw();

    // Print page
    c.cd();
    c.Print("2Dfits.pdf");
    nPages++;
  }

  // Close PDF
  c.Print("2Dfits.pdf]");

  file->Close();
  delete file;

  ::Info("processing2D_plots", "Generated %lld pages (considered %lld events).", nPages, nConsidered);
  return 0;
}



// ROOT auto-exec wrappers so `root -l -b -q plotProcessing2D.C` works
int plotProcessing2D() {
  return processing2D_plots();
}

int plotProcessing2D(const char* filename, double errorPercentOfMax) {
  return processing2D_plots(filename, errorPercentOfMax);
}

int plotProcessing2D(Long64_t nRandomEvents) {
  return processing2D_plots("../build/epicChargeSharing.root", 5.0, nRandomEvents);
}

int plotProcessing2D(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents) {
  return processing2D_plots(filename, errorPercentOfMax, nRandomEvents);
}