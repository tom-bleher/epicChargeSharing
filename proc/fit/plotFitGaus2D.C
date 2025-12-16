// ROOT macro: plotFitGaus2D.C
// Produces a multi-page PDF previewing 2D Gaussian fits with enhanced
// diagnostics for visual assessment of fit quality:
//   - Data heatmap with 1σ/2σ/3σ model contours
//   - Normalized residuals (pulls) heatmap
//   - Pull histogram with Gaussian reference
//   - 1D row/column profiles with fit overlays
//   - Fit quality metrics and color-coded status

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TSystem.h>
#include <TGraph.h>
#include <TGraph2D.h>
#include <TF2.h>
#include <TF1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TROOT.h>
#include <TError.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TBox.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TMarker.h>
#include <TMath.h>
#include <TColor.h>
#include <TText.h>
#include <TPaveText.h>
#include <Math/MinimizerOptions.h>
#include <Fit/Fitter.h>
#include <Fit/BinData.h>
#include <Math/WrappedMultiTF1.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {
  // 2D Gaussian with constant offset:
  // A * exp(-0.5 * ((x-mux)^2/sigx^2 + (y-muy)^2/sigy^2)) + B
  double Gauss2DPlusB(double* xy, double* p) {
    const double A   = p[0];
    const double mx  = p[1];
    const double my  = p[2];
    const double sx  = p[3];
    const double sy  = p[4];
    const double B   = p[5];
    const double dx  = (xy[0] - mx) / sx;
    const double dy  = (xy[1] - my) / sy;
    return A * std::exp(-0.5 * (dx*dx + dy*dy)) + B;
  }

  inline bool IsFinite(double v) { return std::isfinite(v); }
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%).
// The same error is applied to all data points used in the fits for that event.
int processing3D_plots(const char* filename = "../build/epicChargeSharing.root",
                       double errorPercentOfMax = 5.0) {
  // Match processing3D configuration
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(400);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for read (try provided path, then fallback relative to this macro)
  TFile* file = TFile::Open(filename, "READ");
  if (!file || file->IsZombie()) {
    if (file) { file->Close(); delete file; file = nullptr; }
    TString macroDir = gSystem->DirName(__FILE__);
    TString fallback = macroDir + "/../../build/epicChargeSharing.root";
    file = TFile::Open(fallback, "READ");
    if (!file || file->IsZombie()) {
      ::Error("processing3D_plots", "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return 1;
    }
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing3D_plots", "Hits tree not found in file: %s", filename);
    file->Close();
    delete file;
    return 2;
  }

  // Pixel spacing/size: prefer metadata; fallback to inference/defaults
  double pixelSpacing = NAN;
  double pixelSize    = NAN;
  if (auto* spacingObj = dynamic_cast<TNamed*>(file->Get("GridPixelSpacing_mm"))) {
    try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) {}
  }
  if (auto* sizeObj = dynamic_cast<TNamed*>(file->Get("GridPixelSize_mm"))) {
    try { pixelSize = std::stod(sizeObj->GetTitle()); } catch (...) {}
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

  auto inferSpacingFromTree = [&](TTree* t) -> double {
    // First try PixelX/PixelY
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
    double gx = computeGap(xs);
    double gy = computeGap(ys);
    if (IsFinite(gx) && gx>0 && IsFinite(gy) && gy>0) return 0.5*(gx+gy);
    if (IsFinite(gx) && gx>0) return gx;
    if (IsFinite(gy) && gy>0) return gy;

    // Fallback: try NeighborhoodPixelX/Y from first entry
    t->ResetBranchAddresses();
    std::vector<double>* nbX = nullptr;
    std::vector<double>* nbY = nullptr;
    if (t->GetBranch("NeighborhoodPixelX")) t->SetBranchAddress("NeighborhoodPixelX", &nbX);
    if (t->GetBranch("NeighborhoodPixelY")) t->SetBranchAddress("NeighborhoodPixelY", &nbY);
    if (nbX || nbY) {
      t->GetEntry(0);
      xs.clear(); ys.clear();
      if (nbX) for (auto v : *nbX) if (IsFinite(v)) xs.push_back(v);
      if (nbY) for (auto v : *nbY) if (IsFinite(v)) ys.push_back(v);
      gx = computeGap(xs);
      gy = computeGap(ys);
      if (IsFinite(gx) && gx>0 && IsFinite(gy) && gy>0) return 0.5*(gx+gy);
      if (IsFinite(gx) && gx>0) return gx;
      if (IsFinite(gy) && gy>0) return gy;
    }
    return NAN;
  };

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
    tree->ResetBranchAddresses(); // Clean up after inference
  }
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing3D_plots", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  // Pixel size: if not available in metadata, fall back to nominal 0.1 mm
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.1; // mm (100 µm)
    ::Info("processing3D_plots", "Pixel size metadata missing. Falling back to nominal %.3f mm.", pixelSize);
  }

  // Branches
  double x_true = 0.0, y_true = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Fi = nullptr; // fractions 0..1

  tree->SetBranchAddress("TrueX", &x_true);
  tree->SetBranchAddress("TrueY", &y_true);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_true);
  // Try both "Fi" and "FiBlock" branch names for compatibility
  if (tree->GetBranch("Fi")) {
    tree->SetBranchAddress("Fi", &Fi);
  } else if (tree->GetBranch("FiBlock")) {
    tree->SetBranchAddress("FiBlock", &Fi);
  } else {
    ::Error("processing3D_plots", "Neither Fi nor FiBlock branch found in tree. Aborting.");
    file->Close();
    delete file;
    return 4;
  }

  // Style
  gROOT->SetBatch(true);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  TCanvas c("c", "2D Gaussian Fit Diagnostics", 1800, 1200);
  c.Divide(3, 2); // 3x2: Data+Contours, Pulls heatmap, Pull histogram
                  //      Row profile, Column profile, Fit summary

  // Open multipage PDF
  c.Print("fit3d.pdf[");

  // Model function (parameters will be set after fit)
  TF2 f2D("f2D", Gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);

  const Long64_t nEntries = tree->GetEntries();
  const Long64_t nEntriesToPlot = std::min<Long64_t>(nEntries, 100);
  Long64_t nPages = 0, nConsidered = 0;

  // Text drawer for metrics
  TLatex latex; latex.SetNDC(true); latex.SetTextSize(0.030);

  for (Long64_t i = 0; i < nEntriesToPlot; ++i) {
    tree->GetEntry(i);

    if (is_pixel_true || !Fi || Fi->empty()) continue;

    const size_t total = Fi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Build grid-centered arrays and track max for error scaling
    double qmaxNeighborhood = -1e300;
    double zminNeighborhood = 1e300;

    // Prepare TH2s aligned with grid
    const double xLo = x_px - (R + 0.5) * pixelSpacing;
    const double xHi = x_px + (R + 0.5) * pixelSpacing;
    const double yLo = y_px - (R + 0.5) * pixelSpacing;
    const double yHi = y_px + (R + 0.5) * pixelSpacing;

    TH2D hData("hData", Form("Event %lld: Data (F_{i}); x [mm]; y [mm]", i),
               N, xLo, xHi, N, yLo, yHi);
    TH2D hModel("hModel", "Model fit; x [mm]; y [mm]", N, xLo, xHi, N, yLo, yHi);
    TH2D hResid("hResid", "Residuals (data - model); x [mm]; y [mm]", N, xLo, xHi, N, yLo, yHi);
    TH2D hPull("hPull", "Normalized Residuals (pulls); x [mm]; y [mm]", N, xLo, xHi, N, yLo, yHi);
    TH1D hPullDist("hPullDist", "Pull Distribution; (data - model) / #sigma; Entries", 50, -5, 5);

    // Row/column profiles through pixel center
    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);

    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double f = (*Fi)[idx];
        if (!IsFinite(f) || f < 0) continue;
        const double x = x_px + di * pixelSpacing;
        const double y = y_px + dj * pixelSpacing;
        const int binx = hData.GetXaxis()->FindBin(x);
        const int biny = hData.GetYaxis()->FindBin(y);
        hData.SetBinContent(binx, biny, f);
        if (f > qmaxNeighborhood) qmaxNeighborhood = f;
        if (f < zminNeighborhood) zminNeighborhood = f;
        if (dj == 0) { x_row.push_back(x); q_row.push_back(f); }
        if (di == 0) { y_col.push_back(y); q_col.push_back(f); }
      }
    }

    if (x_row.size() < 3 || y_col.size() < 3) continue;
    nConsidered++;

    // Error model for weighting
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    // Initial parameters
    double zmax = -1e300; int idxMax = -1;
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double f = (*Fi)[idx];
        if (!IsFinite(f) || f < 0) continue;
        if (f > zmax) { zmax = f; idxMax = idx; }
      }
    }
    const double A0 = std::max(1e-18, zmax - zminNeighborhood);
    const double B0 = std::max(0.0, zminNeighborhood);
    const int diMax = (idxMax >= 0) ? (idxMax / N - R) : 0;
    const int djMax = (idxMax >= 0) ? (idxMax % N - R) : 0;
    const double mux0 = x_px + diMax * pixelSpacing;
    const double muy0 = y_px + djMax * pixelSpacing;

    // Low-contrast: use centroid; still display page without contours
    bool haveFit = false;
    double mux = NAN, muy = NAN, sx = std::max(0.25*pixelSpacing, 1e-6), sy = std::max(0.25*pixelSpacing, 1e-6);
    double A = A0, B = B0;

    if (A0 < 1e-6) {
      double wsum = 0.0, xw = 0.0, yw = 0.0;
      for (int di = -R; di <= R; ++di) {
        for (int dj = -R; dj <= R; ++dj) {
          const int idx = (di + R) * N + (dj + R);
          const double f = (*Fi)[idx];
          if (!IsFinite(f) || f < 0) continue;
          const double x = x_px + di * pixelSpacing;
          const double y = y_px + dj * pixelSpacing;
          const double w = std::max(0.0, f - B0);
          wsum += w; xw += w * x; yw += w * y;
        }
      }
      if (wsum > 0.0) { mux = xw/wsum; muy = yw/wsum; haveFit = true; }
    } else {
      // Seed sigmas from baseline-subtracted second moments
      auto sigmaSeed2D = [&](bool forX)->double {
        double wsum = 0.0, m = 0.0;
        for (int di = -R; di <= R; ++di) {
          for (int dj = -R; dj <= R; ++dj) {
            const int idx = (di + R) * N + (dj + R);
            const double f = (*Fi)[idx];
            if (!IsFinite(f) || f < 0) continue;
            const double coord = forX ? (x_px + di*pixelSpacing) : (y_px + dj*pixelSpacing);
            const double w = std::max(0.0, f - B0);
            wsum += w; m += w * coord;
          }
        }
        if (wsum <= 0.0) return std::max(0.25*pixelSpacing, 1e-6);
        m /= wsum;
        double var = 0.0;
        for (int di = -R; di <= R; ++di) {
          for (int dj = -R; dj <= R; ++dj) {
            const int idx = (di + R) * N + (dj + R);
            const double f = (*Fi)[idx];
            if (!IsFinite(f) || f < 0) continue;
            const double coord = forX ? (x_px + di*pixelSpacing) : (y_px + dj*pixelSpacing);
            const double d = coord - m;
            const double w = std::max(0.0, f - B0);
            var += w * d * d;
          }
        }
        var = (wsum > 0.0) ? (var / wsum) : 0.0;
        double s = std::sqrt(std::max(var, 1e-12));
        const double sigLoBound = std::max(1e-6, 0.02*pixelSpacing);
        const double sigHiBound = 3.0*pixelSpacing;
        if (s < sigLoBound) s = sigLoBound;
        if (s > sigHiBound) s = sigHiBound;
        return s;
      };
      sx = sigmaSeed2D(true);
      sy = sigmaSeed2D(false);

      // Fit using ROOT::Fit on full window
      const double muXLo = x_px - 0.5 * pixelSpacing;
      const double muXHi = x_px + 0.5 * pixelSpacing;
      const double muYLo = y_px - 0.5 * pixelSpacing;
      const double muYHi = y_px + 0.5 * pixelSpacing;
      const double sigLoBound = std::max(1e-6, 0.02*pixelSpacing);
      const double sigHiBound = 3.0*pixelSpacing;

      // Wrap model; use compact range covering all points
      TF2 fModel("fModel", Gauss2DPlusB, xLo, xHi, yLo, yHi, 6);
      ROOT::Math::WrappedMultiTF1 wModel(fModel, 2);

      // Build BinData
      std::vector<double> Xs; Xs.reserve(N*N);
      std::vector<double> Ys; Ys.reserve(N*N);
      std::vector<double> Zs; Zs.reserve(N*N);
      for (int di = -R; di <= R; ++di) {
        for (int dj = -R; dj <= R; ++dj) {
          const int idx = (di + R) * N + (dj + R);
          const double f = (*Fi)[idx];
          if (!IsFinite(f) || f < 0) continue;
          Xs.push_back(x_px + di * pixelSpacing);
          Ys.push_back(y_px + dj * pixelSpacing);
          Zs.push_back(f);
        }
      }
      const int nPts = static_cast<int>(Zs.size());
      ROOT::Fit::BinData data2D(nPts, 2);
      for (int k = 0; k < nPts; ++k) {
        const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
        double xy[2] = {Xs[k], Ys[k]};
        data2D.Add(xy, Zs[k], ey);
      }

      ROOT::Fit::Fitter fitter;
      fitter.Config().SetMinimizer("Minuit2", "Fumili2");
      fitter.Config().MinimizerOptions().SetStrategy(0);
      fitter.Config().MinimizerOptions().SetTolerance(1e-4);
      fitter.Config().MinimizerOptions().SetPrintLevel(0);
      fitter.SetFunction(wModel);

      // Parameter settings and limits
      fitter.Config().ParSettings(0).SetName("A");
      fitter.Config().ParSettings(1).SetName("mux");
      fitter.Config().ParSettings(2).SetName("muy");
      fitter.Config().ParSettings(3).SetName("sigx");
      fitter.Config().ParSettings(4).SetName("sigy");
      fitter.Config().ParSettings(5).SetName("B");
      fitter.Config().ParSettings(0).SetLowerLimit(0.0);
      fitter.Config().ParSettings(1).SetLimits(muXLo, muXHi);
      fitter.Config().ParSettings(2).SetLimits(muYLo, muYHi);
      fitter.Config().ParSettings(3).SetLimits(sigLoBound, sigHiBound);
      fitter.Config().ParSettings(4).SetLimits(sigLoBound, sigHiBound);
      fitter.Config().ParSettings(5).SetLowerLimit(0.0);

      fitter.Config().ParSettings(0).SetStepSize(1e-3);
      fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
      fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
      fitter.Config().ParSettings(3).SetStepSize(1e-4*pixelSpacing);
      fitter.Config().ParSettings(4).SetStepSize(1e-4*pixelSpacing);
      fitter.Config().ParSettings(5).SetStepSize(1e-3);

      // Seed
      fitter.Config().ParSettings(0).SetValue(A0);
      fitter.Config().ParSettings(1).SetValue(mux0);
      fitter.Config().ParSettings(2).SetValue(muy0);
      fitter.Config().ParSettings(3).SetValue(sx);
      fitter.Config().ParSettings(4).SetValue(sy);
      fitter.Config().ParSettings(5).SetValue(B0);

      haveFit = fitter.Fit(data2D);
      if (haveFit) {
        A = fitter.Result().Parameter(0);
        mux = fitter.Result().Parameter(1);
        muy = fitter.Result().Parameter(2);
        sx  = fitter.Result().Parameter(3);
        sy  = fitter.Result().Parameter(4);
        B   = fitter.Result().Parameter(5);
      } else {
        // Fallback: weighted centroid
        double wsum = 0.0, xw = 0.0, yw = 0.0;
        for (size_t k = 0; k < Zs.size(); ++k) {
          double w = std::max(0.0, Zs[k] - B0);
          wsum += w; xw += w * Xs[k]; yw += w * Ys[k];
        }
        if (wsum > 0) { mux = xw/wsum; muy = yw/wsum; haveFit = true; }
      }
    }

    // Prepare model TF2 with fitted parameters
    f2D.SetRange(xLo, xHi, yLo, yHi);
    f2D.SetParameters(A, mux, muy, sx, sy, B);

    // Fill model, residuals, and pulls on the same grid
    double chi2 = 0.0; int nPtsUsed = 0;
    double pullSum = 0.0, pullSum2 = 0.0;
    for (int ix = 1; ix <= N; ++ix) {
      const double xc = hData.GetXaxis()->GetBinCenter(ix);
      for (int iy = 1; iy <= N; ++iy) {
        const double yc = hData.GetYaxis()->GetBinCenter(iy);
        const double data = hData.GetBinContent(ix, iy);
        if (!IsFinite(data) || data < 0) continue;
        const double model = f2D.Eval(xc, yc);
        hModel.SetBinContent(ix, iy, model);
        const double r = data - model;
        hResid.SetBinContent(ix, iy, r);
        // Normalized residual (pull)
        const double pull = (uniformSigma > 0.0) ? (r / uniformSigma) : r;
        hPull.SetBinContent(ix, iy, pull);
        hPullDist.Fill(pull);
        pullSum += pull;
        pullSum2 += pull * pull;
        const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
        chi2 += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
        nPtsUsed++;
      }
    }
    const int nPar = 6;
    const int ndof = std::max(1, nPtsUsed - nPar);
    // Pull statistics
    const double pullMean = (nPtsUsed > 0) ? pullSum / nPtsUsed : 0.0;
    const double pullRMS = (nPtsUsed > 1) ? std::sqrt((pullSum2 - pullSum*pullSum/nPtsUsed) / (nPtsUsed - 1)) : 0.0;

    // Determine color scales
    double zDataMax = hData.GetMaximum();
    hData.SetMaximum(1.20 * zDataMax);
    hData.SetMinimum(0.0);
    // Residual range symmetric around zero
    double absRmax = 0.0;
    for (int b = 1; b <= hResid.GetSize() - 2; ++b) {
      absRmax = std::max(absRmax, std::abs(hResid.GetArray()[b]));
    }
    hResid.SetMaximum(+absRmax);
    hResid.SetMinimum(-absRmax);

    // Determine fit quality for color coding
    const double chi2ndof = static_cast<double>(chi2) / static_cast<double>(ndof);
    int fitQualityColor = kGreen+2;  // Good fit
    const char* fitQualityLabel = "GOOD";
    if (chi2ndof > 3.0) { fitQualityColor = kRed+1; fitQualityLabel = "POOR"; }
    else if (chi2ndof > 1.5) { fitQualityColor = kOrange+1; fitQualityLabel = "FAIR"; }

    // Draw Data + explicit sigma contours
    c.cd(1);
    gPad->SetRightMargin(0.14);
    gPad->SetLeftMargin(0.12);
    hData.SetTitle(Form("Event %lld: Data with 1#sigma/2#sigma/3#sigma contours", i));
    hData.GetXaxis()->SetTitleSize(0.045);
    hData.GetYaxis()->SetTitleSize(0.045);
    hData.Draw("COLZ");
    if (haveFit && A > 0.0) {
      // Draw explicit sigma level contours: f(x,y) = A*exp(-0.5*r^2) + B
      // At n-sigma: r^2 = n^2, so value = A*exp(-0.5*n^2) + B
      double contourLevels[3];
      contourLevels[0] = A * std::exp(-0.5 * 1.0) + B;  // 1σ (60.65% of peak)
      contourLevels[1] = A * std::exp(-0.5 * 4.0) + B;  // 2σ (13.53% of peak)
      contourLevels[2] = A * std::exp(-0.5 * 9.0) + B;  // 3σ (1.11% of peak)

      f2D.SetNpx(100);
      f2D.SetNpy(100);
      f2D.SetContour(3, contourLevels);
      f2D.SetLineWidth(2);

      // Draw each contour with different style
      TF2* f1sig = new TF2("f1sig", Gauss2DPlusB, xLo, xHi, yLo, yHi, 6);
      f1sig->SetParameters(A, mux, muy, sx, sy, B);
      f1sig->SetContour(1, &contourLevels[0]);
      f1sig->SetLineColor(kRed+1);
      f1sig->SetLineWidth(3);
      f1sig->SetLineStyle(1);
      f1sig->SetNpx(100); f1sig->SetNpy(100);
      f1sig->Draw("CONT3 SAME");
      f1sig->SetBit(kCanDelete);

      TF2* f2sig = new TF2("f2sig", Gauss2DPlusB, xLo, xHi, yLo, yHi, 6);
      f2sig->SetParameters(A, mux, muy, sx, sy, B);
      f2sig->SetContour(1, &contourLevels[1]);
      f2sig->SetLineColor(kOrange+1);
      f2sig->SetLineWidth(2);
      f2sig->SetLineStyle(2);
      f2sig->SetNpx(100); f2sig->SetNpy(100);
      f2sig->Draw("CONT3 SAME");
      f2sig->SetBit(kCanDelete);

      TF2* f3sig = new TF2("f3sig", Gauss2DPlusB, xLo, xHi, yLo, yHi, 6);
      f3sig->SetParameters(A, mux, muy, sx, sy, B);
      f3sig->SetContour(1, &contourLevels[2]);
      f3sig->SetLineColor(kYellow+1);
      f3sig->SetLineWidth(2);
      f3sig->SetLineStyle(3);
      f3sig->SetNpx(100); f3sig->SetNpy(100);
      f3sig->Draw("CONT3 SAME");
      f3sig->SetBit(kCanDelete);
    }
    // Markers: draw REC last so it is visible when overlapping the pixel center
    TMarker mtrue(x_true, y_true, 33); mtrue.SetMarkerColor(kBlack); mtrue.SetMarkerSize(1.6);
    TMarker mPix(x_px, y_px, 28); mPix.SetMarkerColor(kBlue+2); mPix.SetMarkerSize(1.4);
    TMarker mRec; if (IsFinite(mux) && IsFinite(muy)) { mRec = TMarker(mux, muy, 29); mRec.SetMarkerColor(kRed+1); mRec.SetMarkerSize(1.8); }
    mtrue.Draw();
    mPix.Draw();
    if (IsFinite(mux) && IsFinite(muy)) mRec.Draw();

    // Legend for markers and contours
    TLegend leg1(0.01, 0.72, 0.38, 0.92);
    leg1.SetBorderSize(0); leg1.SetFillStyle(0); leg1.SetTextSize(0.032);
    leg1.AddEntry(&mtrue, "(x_{true},y_{true})", "p");
    leg1.AddEntry(&mPix, "(x_{px},y_{px})", "p");
    if (IsFinite(mux) && IsFinite(muy)) leg1.AddEntry(&mRec, "(x_{rec},y_{rec})", "p");
    leg1.Draw();

    // Contour legend
    if (haveFit && A > 0.0) {
      TLegend legC(0.60, 0.01, 0.99, 0.18);
      legC.SetBorderSize(0); legC.SetFillStyle(0); legC.SetTextSize(0.028);
      TLine* l1 = new TLine(); l1->SetLineColor(kRed+1); l1->SetLineWidth(3); l1->SetLineStyle(1); l1->SetBit(kCanDelete);
      TLine* l2 = new TLine(); l2->SetLineColor(kOrange+1); l2->SetLineWidth(2); l2->SetLineStyle(2); l2->SetBit(kCanDelete);
      TLine* l3 = new TLine(); l3->SetLineColor(kYellow+1); l3->SetLineWidth(2); l3->SetLineStyle(3); l3->SetBit(kCanDelete);
      legC.AddEntry(l1, "1#sigma (60.7%)", "l");
      legC.AddEntry(l2, "2#sigma (13.5%)", "l");
      legC.AddEntry(l3, "3#sigma (1.1%)", "l");
      legC.Draw();
    }

    // Draw Pulls (normalized residuals) heatmap with diverging colormap
    c.cd(2);
    gPad->SetRightMargin(0.14);
    gPad->SetLeftMargin(0.12);
    // Symmetric pull range centered at zero
    double pullMax = std::max(3.0, std::abs(hPull.GetMaximum()));
    pullMax = std::max(pullMax, std::abs(hPull.GetMinimum()));
    hPull.SetMaximum(+pullMax);
    hPull.SetMinimum(-pullMax);
    hPull.SetTitle("Normalized Residuals (Pulls)");
    hPull.GetXaxis()->SetTitleSize(0.045);
    hPull.GetYaxis()->SetTitleSize(0.045);
    hPull.SetContour(100);
    // Use a diverging colormap (blue-white-red)
    gStyle->SetPalette(kTemperatureMap);
    hPull.Draw("COLZ");
    // Mark bins with |pull| > 2 as potentially problematic
    latex.SetTextSize(0.035);
    latex.DrawLatex(0.15, 0.92, Form("Pull mean: %.2f", pullMean));
    latex.DrawLatex(0.15, 0.87, Form("Pull RMS: %.2f", pullRMS));
    // Reset palette to default for other plots
    gStyle->SetPalette(kBird);

    // Draw Pull histogram (panel 3)
    c.cd(3);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    hPullDist.SetTitle("Pull Distribution");
    hPullDist.GetXaxis()->SetTitleSize(0.045);
    hPullDist.GetYaxis()->SetTitleSize(0.045);
    hPullDist.SetFillColor(kAzure-9);
    hPullDist.SetLineColor(kBlue+2);
    hPullDist.SetLineWidth(2);
    hPullDist.Draw("HIST");
    // Overlay reference Gaussian N(0,1)
    if (hPullDist.GetEntries() > 0) {
      double normFactor = hPullDist.GetEntries() * hPullDist.GetBinWidth(1);
      TF1* fGausRef = new TF1("fGausRef", "[0]*exp(-0.5*x*x)", -5, 5);
      fGausRef->SetParameter(0, normFactor / std::sqrt(2.0 * TMath::Pi()));
      fGausRef->SetLineColor(kRed+1);
      fGausRef->SetLineWidth(2);
      fGausRef->SetLineStyle(2);
      fGausRef->Draw("L SAME");
      fGausRef->SetBit(kCanDelete);
      TLegend legPull(0.60, 0.75, 0.94, 0.92);
      legPull.SetBorderSize(0); legPull.SetFillStyle(0); legPull.SetTextSize(0.032);
      legPull.AddEntry(&hPullDist, "Pulls", "f");
      legPull.AddEntry(fGausRef, "N(0,1) reference", "l");
      legPull.Draw();
    }
    // Add statistics
    latex.SetTextSize(0.035);
    latex.DrawLatex(0.15, 0.92, Form("#mu = %.3f", pullMean));
    latex.DrawLatex(0.15, 0.86, Form("#sigma = %.3f", pullRMS));
    // Quality indicator
    latex.SetTextColor(fitQualityColor);
    latex.DrawLatex(0.15, 0.80, Form("Fit: %s", fitQualityLabel));
    latex.SetTextColor(kBlack);

    // Row profile (y fixed at y_px) - Panel 4
    c.cd(4);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    std::vector<double> x_row_sorted = x_row; std::sort(x_row_sorted.begin(), x_row_sorted.end());
    TGraph gRow(static_cast<int>(x_row.size()));
    for (int k = 0; k < (int)x_row.size(); ++k) gRow.SetPoint(k, x_row[k], q_row[k]);
    gRow.SetTitle("Central row (y=y_{px}); x [mm]; F_{i}");
    gRow.SetMarkerStyle(20); gRow.SetMarkerSize(0.9); gRow.SetLineColor(kBlue+2);
    double maxRow = *std::max_element(q_row.begin(), q_row.end());
    double yMaxRowAuto = 1.20 * maxRow;
    if (haveFit && A > 0.0) {
      double Arow  = A * std::exp(-0.5 * std::pow((y_px - muy)/sy, 2.0));
      double Arow0 = A * std::exp(-0.5 * std::pow((0.0  - muy)/sy, 2.0));
      double peakRow = std::max(Arow + B, Arow0 + B);
      yMaxRowAuto = 1.20 * std::max(maxRow, peakRow);
    }
    gRow.SetMaximum(yMaxRowAuto);
    gRow.Draw("AP");
    // Draw simple rectangles indicating pixel width around each point
    {
      gPad->Update();
      double yPadMin = gPad->GetUymin();
      double yPadMax = gPad->GetUymax();
      const double halfH = 0.015 * (yPadMax - yPadMin);
      for (size_t k = 0; k < x_row.size(); ++k) {
        const double xc = x_row[k];
        const double yc = q_row[k];
        TBox* box = new TBox(xc - 0.5*pixelSize, yc - halfH, xc + 0.5*pixelSize, yc + halfH);
        box->SetFillStyle(0); box->SetLineColor(kGray+2); box->SetLineWidth(1); box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
    }
    // Overlay analytic 1D slices of the 3D Gaussian:
    //  - at y=y_px (matches central row data)
    //  - at y=0    (requested plane intersection)
    if (haveFit && A > 0.0) {
      double xMin = *std::min_element(x_row.begin(), x_row.end()) - 0.5*pixelSpacing;
      double xMax = *std::max_element(x_row.begin(), x_row.end()) + 0.5*pixelSpacing;
      double Arow  = A * std::exp(-0.5 * std::pow((y_px - muy)/sy, 2.0));
      TF1* fRowSlice   = new TF1("fRowSlice_yPx", "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]", xMin, xMax);
      fRowSlice->SetParameters(Arow, mux, sx, B);
      fRowSlice->SetNpx(600);
      fRowSlice->SetLineColor(kRed+1);
      fRowSlice->SetLineWidth(2);
      fRowSlice->Draw("L SAME");
      fRowSlice->SetBit(kCanDelete);
      // Also overlay slice at y=0
      double Arow0 = A * std::exp(-0.5 * std::pow((0.0 - muy)/sy, 2.0));
      TF1* fRowSlice0 = new TF1("fRowSlice_y0", "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]", xMin, xMax);
      fRowSlice0->SetParameters(Arow0, mux, sx, B);
      fRowSlice0->SetNpx(600);
      fRowSlice0->SetLineColor(kRed+2);
      fRowSlice0->SetLineWidth(2);
      fRowSlice0->SetLineStyle(7);
      fRowSlice0->Draw("L SAME");
      fRowSlice0->SetBit(kCanDelete);
      TLegend legSliceR(0.60, 0.80, 0.98, 0.94);
      legSliceR.SetBorderSize(0); legSliceR.SetFillStyle(0); legSliceR.SetTextSize(0.028);
      legSliceR.AddEntry(fRowSlice,  "3D slice @ y=y_{px}", "l");
      legSliceR.AddEntry(fRowSlice0, "3D slice @ y=0", "l");
      legSliceR.Draw();
    }
    // true and rec markers
    {
      gPad->Update(); double yPadMin = gPad->GetUymin(); double yPadMax = gPad->GetUymax();
      TLine lineXtrue(x_true, yPadMin, x_true, yPadMax); lineXtrue.SetLineStyle(2); lineXtrue.SetLineWidth(2); lineXtrue.SetLineColor(kBlack); lineXtrue.Draw("SAME");
      if (IsFinite(mux)) { TLine lineXrec(mux, yPadMin, mux, yPadMax); lineXrec.SetLineStyle(2); lineXrec.SetLineWidth(2); lineXrec.SetLineColor(kRed+1); lineXrec.Draw("SAME"); }
      TLegend legR(0.12, 0.78, 0.42, 0.94); legR.SetBorderSize(0); legR.SetFillStyle(0); legR.SetTextSize(0.030);
      legR.AddEntry((TObject*)nullptr, Form("|x_{true}-x_{px}|=%.1f #mum", 1000.0*std::abs(x_true - x_px)), "");
      if (IsFinite(mux)) legR.AddEntry((TObject*)nullptr, Form("|x_{true}-x_{rec}|=%.1f #mum", 1000.0*std::abs(x_true - mux)), "");
      legR.Draw();
    }

    // Column profile (x fixed at x_px) - Panel 5
    c.cd(5);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    std::vector<double> y_col_sorted = y_col; std::sort(y_col_sorted.begin(), y_col_sorted.end());
    TGraph gCol(static_cast<int>(y_col.size()));
    for (int k = 0; k < (int)y_col.size(); ++k) gCol.SetPoint(k, y_col[k], q_col[k]);
    gCol.SetTitle("Central column (x=x_{px}); y [mm]; F_{i}");
    gCol.SetMarkerStyle(21); gCol.SetMarkerSize(0.9); gCol.SetLineColor(kBlue+2);
    double maxCol = *std::max_element(q_col.begin(), q_col.end());
    double yMaxColAuto = 1.20 * maxCol;
    if (haveFit && A > 0.0) {
      double Acol  = A * std::exp(-0.5 * std::pow((x_px - mux)/sx, 2.0));
      double Acol0 = A * std::exp(-0.5 * std::pow((0.0  - mux)/sx, 2.0));
      double peakCol = std::max(Acol + B, Acol0 + B);
      yMaxColAuto = 1.20 * std::max(maxCol, peakCol);
    }
    gCol.SetMaximum(yMaxColAuto);
    gCol.Draw("AP");
    {
      gPad->Update();
      double yPadMin = gPad->GetUymin();
      double yPadMax = gPad->GetUymax();
      const double halfH = 0.015 * (yPadMax - yPadMin);
      for (size_t k = 0; k < y_col.size(); ++k) {
        const double xc = y_col[k];
        const double yc = q_col[k];
        TBox* box = new TBox(xc - 0.5*pixelSize, yc - halfH, xc + 0.5*pixelSize, yc + halfH);
        box->SetFillStyle(0); box->SetLineColor(kGray+2); box->SetLineWidth(1); box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
    }
    if (haveFit && A > 0.0) {
      double yMin = *std::min_element(y_col.begin(), y_col.end()) - 0.5*pixelSpacing;
      double yMax = *std::max_element(y_col.begin(), y_col.end()) + 0.5*pixelSpacing;
      double Acol  = A * std::exp(-0.5 * std::pow((x_px - mux)/sx, 2.0));
      TF1* fColSlice = new TF1("fColSlice_xPx", "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]", yMin, yMax);
      fColSlice->SetParameters(Acol, muy, sy, B);
      fColSlice->SetNpx(600);
      fColSlice->SetLineColor(kRed+1);
      fColSlice->SetLineWidth(2);
      fColSlice->Draw("L SAME");
      fColSlice->SetBit(kCanDelete);
      // Also overlay slice at x=0
      double Acol0 = A * std::exp(-0.5 * std::pow((0.0 - mux)/sx, 2.0));
      TF1* fColSlice0 = new TF1("fColSlice_x0", "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]", yMin, yMax);
      fColSlice0->SetParameters(Acol0, muy, sy, B);
      fColSlice0->SetNpx(600);
      fColSlice0->SetLineColor(kRed+2);
      fColSlice0->SetLineWidth(2);
      fColSlice0->SetLineStyle(7);
      fColSlice0->Draw("L SAME");
      fColSlice0->SetBit(kCanDelete);
      TLegend legSliceC(0.60, 0.80, 0.98, 0.94);
      legSliceC.SetBorderSize(0); legSliceC.SetFillStyle(0); legSliceC.SetTextSize(0.028);
      legSliceC.AddEntry(fColSlice,  "3D slice @ x=x_{px}", "l");
      legSliceC.AddEntry(fColSlice0, "3D slice @ x=0", "l");
      legSliceC.Draw();
    }
    {
      gPad->Update(); double yPadMin = gPad->GetUymin(); double yPadMax = gPad->GetUymax();
      TLine lineYtrue(y_true, yPadMin, y_true, yPadMax); lineYtrue.SetLineStyle(2); lineYtrue.SetLineWidth(2); lineYtrue.SetLineColor(kBlack); lineYtrue.Draw("SAME");
      if (IsFinite(muy)) { TLine lineYrec(muy, yPadMin, muy, yPadMax); lineYrec.SetLineStyle(2); lineYrec.SetLineWidth(2); lineYrec.SetLineColor(kRed+1); lineYrec.Draw("SAME"); }
      TLegend legColY(0.12, 0.78, 0.42, 0.94); legColY.SetBorderSize(0); legColY.SetFillStyle(0); legColY.SetTextSize(0.030);
      legColY.AddEntry((TObject*)nullptr, Form("|y_{true}-y_{px}|=%.1f #mum", 1000.0*std::abs(y_true - y_px)), "");
      if (IsFinite(muy)) legColY.AddEntry((TObject*)nullptr, Form("|y_{true}-y_{rec}|=%.1f #mum", 1000.0*std::abs(y_true - muy)), "");
      legColY.Draw();
    }

    // Fit Summary Panel (Panel 6)
    c.cd(6);
    gPad->SetLeftMargin(0.02);
    gPad->SetRightMargin(0.02);
    gPad->SetTopMargin(0.02);
    gPad->SetBottomMargin(0.02);

    TPaveText summaryBox(0.05, 0.05, 0.95, 0.95, "NDC");
    summaryBox.SetBorderSize(1);
    summaryBox.SetFillColor(kWhite);
    summaryBox.SetTextAlign(12);
    summaryBox.SetTextFont(42);
    summaryBox.SetTextSize(0.045);

    // Title with quality indicator
    TText* titleLine = summaryBox.AddText(Form("Event %lld - Fit Summary", i));
    titleLine->SetTextSize(0.055);
    titleLine->SetTextFont(62);

    summaryBox.AddText(" ");  // spacer

    // Fit quality with color
    TText* qualLine = summaryBox.AddText(Form("Fit Quality: %s (#chi^{2}/ndf = %.2f)", fitQualityLabel, chi2ndof));
    qualLine->SetTextColor(fitQualityColor);
    qualLine->SetTextSize(0.050);

    summaryBox.AddText(" ");  // spacer

    // Fit parameters
    summaryBox.AddText(Form("Amplitude A = %.4g", A));
    summaryBox.AddText(Form("Baseline B = %.4g", B));
    summaryBox.AddText(Form("#sigma_{x} = %.4f mm (%.1f #mum)", sx, sx*1000));
    summaryBox.AddText(Form("#sigma_{y} = %.4f mm (%.1f #mum)", sy, sy*1000));

    summaryBox.AddText(" ");  // spacer

    // Position reconstruction
    summaryBox.AddText(Form("#mu_{x} = %.4f mm, #mu_{y} = %.4f mm", mux, muy));

    summaryBox.AddText(" ");  // spacer

    // Reconstruction errors
    double errX_rec = IsFinite(mux) ? 1000.0*std::abs(x_true - mux) : NAN;
    double errY_rec = IsFinite(muy) ? 1000.0*std::abs(y_true - muy) : NAN;
    double errX_px = 1000.0*std::abs(x_true - x_px);
    double errY_px = 1000.0*std::abs(y_true - y_px);

    TText* errRecLine = summaryBox.AddText(Form("Rec. error: #Deltax = %.1f #mum, #Deltay = %.1f #mum", errX_rec, errY_rec));
    // Color code based on improvement
    if (IsFinite(errX_rec) && IsFinite(errY_rec)) {
      double errRec = std::sqrt(errX_rec*errX_rec + errY_rec*errY_rec);
      double errPx = std::sqrt(errX_px*errX_px + errY_px*errY_px);
      if (errRec < errPx * 0.8) errRecLine->SetTextColor(kGreen+2);
      else if (errRec > errPx * 1.2) errRecLine->SetTextColor(kRed+1);
    }

    summaryBox.AddText(Form("Pixel error: #Deltax = %.1f #mum, #Deltay = %.1f #mum", errX_px, errY_px));

    summaryBox.AddText(" ");  // spacer

    // Pull statistics assessment
    TText* pullLine = summaryBox.AddText(Form("Pull stats: #mu = %.2f, #sigma = %.2f", pullMean, pullRMS));
    if (std::abs(pullMean) < 0.3 && std::abs(pullRMS - 1.0) < 0.3) {
      pullLine->SetTextColor(kGreen+2);
    } else if (std::abs(pullMean) > 1.0 || std::abs(pullRMS - 1.0) > 0.5) {
      pullLine->SetTextColor(kRed+1);
    } else {
      pullLine->SetTextColor(kOrange+1);
    }

    // Interpretation guide
    summaryBox.AddText(" ");
    TText* guideLine = summaryBox.AddText("Good fit: #chi^{2}/ndf #approx 1, pull #mu #approx 0, #sigma #approx 1");
    guideLine->SetTextSize(0.035);
    guideLine->SetTextColor(kGray+2);

    summaryBox.Draw();

    c.Print("fit3d.pdf");
    nPages++;
  }

  c.Print("fit3d.pdf]");

  file->Close();
  delete file;

  ::Info("processing3D_plots", "Generated %lld pages (considered %lld events).", nPages, nConsidered);
  return 0;
}



// Convenience wrappers so the macro can be executed directly as:
//   root -l -b -q plotFitGaus2D.C
// They forward to processing3D_plots and try a few common default paths.
int plotFitGaus2D() {
  TString macroDir = gSystem->DirName(__FILE__);
  TString def = macroDir + "/../../build/epicChargeSharing.root";
  return processing3D_plots(def.Data(), 5.0);
}

int plotFitGaus2D(const char* filename) {
  return processing3D_plots(filename, 5.0);
}

int plotFitGaus2D(const char* filename, double errorPercentOfMax) {
  return processing3D_plots(filename, errorPercentOfMax);
}


