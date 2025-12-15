// ROOT macro: processing3D_plots.C
// Produces a multi-page PDF previewing 3D Gaussian fits (data heatmap with
// model contours, residuals heatmap, and 1D row/column profiles) to assess
// goodness-of-fit for non-pixel-pad trues.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph2D.h>
#include <TF2.h>
#include <TF1.h>
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
  tree->SetBranchAddress("Fi", &Fi);

  // Style
  gROOT->SetBatch(true);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  TCanvas c("c", "processing3D fit preview", 1200, 900);
  c.Divide(2, 2); // 2x2: Data+Contours, Residuals, Row profile, Column profile

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

    // Fill model and residuals on the same grid
    double chi2 = 0.0; int nPtsUsed = 0;
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
        const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
        chi2 += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
        nPtsUsed++;
      }
    }
    const int nPar = 6;
    const int ndof = std::max(1, nPtsUsed - nPar);

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

    // Draw Data + contours
    c.cd(1);
    gPad->SetRightMargin(0.12);
    hData.SetTitle(Form("Event %lld: Data with fit contours; F_{i} [unitless]", i));
    hData.Draw("COLZ");
    if (haveFit && A > 0.0) {
      f2D.SetNpx(80); // contour resolution in x
      f2D.SetNpy(80); // contour resolution in y
      f2D.SetLineColor(kRed+1);
      f2D.SetLineWidth(2);
      f2D.Draw("CONT3 SAME");
    }
    // Markers: draw REC last so it is visible when overlapping the pixel center
    TMarker mtrue(x_true, y_true, 33); mtrue.SetMarkerColor(kBlack); mtrue.SetMarkerSize(1.4);
    TMarker mPix(x_px, y_px, 28); mPix.SetMarkerColor(kBlue+2); mPix.SetMarkerSize(1.2);
    TMarker mRec; if (IsFinite(mux) && IsFinite(muy)) { mRec = TMarker(mux, muy, 29); mRec.SetMarkerColor(kRed+1); mRec.SetMarkerSize(1.5); }
    mtrue.Draw();
    mPix.Draw();
    if (IsFinite(mux) && IsFinite(muy)) mRec.Draw();
    TLegend leg1(0.12, 0.80, 0.46, 0.94);
    leg1.SetBorderSize(0); leg1.SetFillStyle(0); leg1.SetTextSize(0.030);
    leg1.AddEntry(&mtrue, "(x_{true},y_{true})", "p");
    leg1.AddEntry(&mPix, "(x_{px},y_{px})", "p");
    if (IsFinite(mux) && IsFinite(muy)) leg1.AddEntry(&mRec, "(x_{rec},y_{rec})", "p");
    leg1.Draw();

    // Metrics box
    c.cd(1);
    double chi2ndof = static_cast<double>(chi2) / static_cast<double>(ndof);
    latex.DrawLatex(0.50, 0.93, Form("#chi^{2}/ndof = %.3g / %d = %.3g", chi2, ndof, chi2ndof));
    latex.DrawLatex(0.50, 0.88, Form("A=%.3g, B=%.3g, #sigma_{x}=%.3g, #sigma_{y}=%.3g", A, B, sx, sy));
    latex.DrawLatex(0.50, 0.83, Form("|x_{true}-x_{rec}|=%.1f #mum, |y_{true}-y_{rec}|=%.1f #mum", 1000.0*std::abs(x_true - mux), 1000.0*std::abs(y_true - muy)));
    latex.DrawLatex(0.50, 0.78, Form("|x_{true}-x_{px}|=%.1f #mum, |y_{true}-y_{px}|=%.1f #mum", 1000.0*std::abs(x_true - x_px), 1000.0*std::abs(y_true - y_px)));

    // Draw Residuals heatmap
    c.cd(2);
    gPad->SetRightMargin(0.12);
    hResid.SetTitle("Residuals (data - model)");
    hResid.SetContour(100);
    hResid.Draw("COLZ");
    // Zero residual contour (approx by line at 0 via contours): use TF2 difference? Skip for simplicity

    // Row profile (y fixed at y_px)
    c.cd(3);
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

    // Column profile (x fixed at x_px)
    c.cd(4);
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
      TLegend legC(0.12, 0.78, 0.42, 0.94); legC.SetBorderSize(0); legC.SetFillStyle(0); legC.SetTextSize(0.030);
      legC.AddEntry((TObject*)nullptr, Form("|y_{true}-y_{px}|=%.1f #mum", 1000.0*std::abs(y_true - y_px)), "");
      if (IsFinite(muy)) legC.AddEntry((TObject*)nullptr, Form("|y_{true}-y_{rec}|=%.1f #mum", 1000.0*std::abs(y_true - muy)), "");
      legC.Draw();
    }

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
//   root -l -b -q plotProcessing3D.C
// They forward to processing3D_plots and try a few common default paths.
int plotProcessing3D() {
  TString macroDir = gSystem->DirName(__FILE__);
  TString def = macroDir + "/../../build/epicChargeSharing.root";
  return processing3D_plots(def.Data(), 5.0);
}

int plotProcessing3D(const char* filename) {
  return processing3D_plots(filename, 5.0);
}

int plotProcessing3D(const char* filename, double errorPercentOfMax) {
  return processing3D_plots(filename, errorPercentOfMax);
}


