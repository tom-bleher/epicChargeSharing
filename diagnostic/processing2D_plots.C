// ROOT macro: processing2D_plots.C
// Produces multi-page PDFs with row and column fits for non-pixel-pad hits
// Output files: row.pdf, column.pdf (each page corresponds to one event)

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TROOT.h>
#include <TError.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TLatex.h>
#include <TStyle.h>
#include <Math/MinimizerOptions.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>

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
int processing2D_plots(const char* filename = "../build/epicChargeSharingOutput.root",
                       double errorPercentOfMax = 5.0) {
  // Use Minuit2 by default (match processing2D)
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);

  // Open file for read
  TFile* file = TFile::Open(filename, "READ");
  if (!file || file->IsZombie()) {
    ::Error("processing2D_plots", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D_plots", "Hits tree not found in file: %s", filename);
    file->Close();
    delete file;
    return 2;
  }

  // Pixel spacing/size: prefer metadata; fallback to inference/defaults as in processing2D
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
    ::Error("processing2D_plots", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  // Pixel size: if not available in metadata, fall back to nominal 0.1 mm
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.1; // mm (100 µm)
    ::Info("processing2D_plots", "Pixel size metadata missing. Falling back to nominal %.3f mm.", pixelSize);
  }

  // Set up branches
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  std::vector<double>* Fi = nullptr; // use fractions for fitting

  tree->SetBranchAddress("x_hit", &x_hit);
  tree->SetBranchAddress("y_hit", &y_hit);
  tree->SetBranchAddress("x_px",  &x_px);
  tree->SetBranchAddress("y_px",  &y_px);
  tree->SetBranchAddress("is_pixel_hit", &is_pixel_hit);
  tree->SetBranchAddress("F_i", &Fi);

  // Prepare plotting objects
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  TCanvas c("c", "processing2D fits", 900, 700);

  // Open multipage PDFs
  c.Print("row.pdf[");
  c.Print("column.pdf[");

  // Functions for visualization (parameters will be set from the new fitter)
  TF1 fRow("fRow", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol", GaussPlusB, -1e9, 1e9, 4);

  const Long64_t nEntries = tree->GetEntries();
  const Long64_t nEntriesToPlot = std::min<Long64_t>(nEntries, 100);
  Long64_t nPagesRow = 0, nPagesCol = 0, nConsidered = 0;

  for (Long64_t i = 0; i < nEntriesToPlot; ++i) {
    tree->GetEntry(i);

    // Only non-pixel-pad hits with valid neighborhood (fractions 0..1)
    if (is_pixel_hit || !Fi || Fi->empty()) continue;

    const size_t total = Fi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Diagnostics: verify geometric non-pixel condition max(|dx|,|dy|) > size/2
    const double dxAbs = std::abs(x_hit - x_px);
    const double dyAbs = std::abs(y_hit - y_px);
    const double halfPixel = 0.5 * pixelSize;
    const double maxAbs = std::max(dxAbs, dyAbs);
    if (!(maxAbs > halfPixel)) {
      ::Warning("processing2D_plots",
                "Event %lld marked non-pixel but max(|dx|,|dy|)=%.3f µm <= halfPixel=%.3f µm (dx=%.3f µm, dy=%.3f µm).",
                i, 1000.0*maxAbs, 1000.0*halfPixel, 1000.0*dxAbs, 1000.0*dyAbs);
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
        const double f = (*Fi)[idx];
        if (!IsFinite(f) || f < 0) continue;
        if (f > qmaxNeighborhood) qmaxNeighborhood = f;
        // Correct mapping: di moves along X, dj moves along Y
        if (dj == 0) { // central row (vary X at fixed Y)
          const double x = x_px + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(f);
        }
        if (di == 0) { // central column (vary Y at fixed X)
          const double y = y_px + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(f);
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
    baseRowPtr->SetTitle(Form("Event %lld: Central row fit; x [mm]; Charge fraction F_i [unitless]", i));
    baseRowPtr->SetMarkerStyle(20);
    baseRowPtr->SetMarkerSize(0.9);
    baseRowPtr->SetLineColor(kBlue+1);

    baseColPtr->SetTitle(Form("Event %lld: Central column fit; y [mm]; Charge fraction F_i [unitless]", i));
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

    // Very low contrast: fallback to fast weighted centroids (skip fit)
    if (A0_row < 1e-6 && A0_col < 1e-6) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k]); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k]); wsumy += w; yw += w * y_col[k]; }
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

      // Chi2 objectives (weighted if uniformSigma > 0)
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

      haveFitRow = mRow->Minimize();
      haveFitCol = mCol->Minimize();

      if (haveFitRow) {
        A_row = mRow->X()[0]; muRow = mRow->X()[1]; S_row = mRow->X()[2]; B_row = mRow->X()[3];
        fRow.SetRange(xMin, xMax);
        fRow.SetParameters(A_row, muRow, S_row, B_row);
      } else {
        // Fallback: weighted centroid on trimmed window
        double wsum = 0.0, xw = 0.0;
        for (size_t k=0;k<x_row_fit.size();++k) { double w = std::max(0.0, q_row_fit[k]); wsum += w; xw += w * x_row_fit[k]; }
        if (wsum > 0) { muRow = xw / wsum; haveFitRow = true; }
      }

      if (haveFitCol) {
        A_col = mCol->X()[0]; muCol = mCol->X()[1]; S_col = mCol->X()[2]; B_col = mCol->X()[3];
        fCol.SetRange(yMin, yMax);
        fCol.SetParameters(A_col, muCol, S_col, B_col);
      } else {
        // Fallback: weighted centroid on trimmed window
        double wsum = 0.0, yw = 0.0;
        for (size_t k=0;k<y_col_fit.size();++k) { double w = std::max(0.0, q_col_fit[k]); wsum += w; yw += w * y_col_fit[k]; }
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

    // Draw row
    c.cd();
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
    TLine lineXhit(x_hit, yPadMin, x_hit, yPadMax);
    lineXhit.SetLineStyle(2);
    lineXhit.SetLineWidth(2);
    lineXhit.SetLineColor(kBlack);
    lineXhit.Draw("SAME");
    double fx1 = gPad->GetLeftMargin();
    double fy1 = gPad->GetBottomMargin();
    double fx2 = 1.0 - gPad->GetRightMargin();
    double fy2 = 1.0 - gPad->GetTopMargin();
    // Slightly increase legend height to accommodate extra entry (|x_hit - x_px|)
    double legW = 0.22, legH = 0.18, inset = 0.008;
    double lx1 = fx1 + inset, lx2 = lx1 + legW;
    double ly2 = fy2 - inset, ly1 = ly2 - legH;
    TLegend leg(lx1, ly1, lx2, ly2, "", "NDC");
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.SetTextSize(0.03);
    leg.AddEntry(&lineXhit, "x_{hit}", "l");
    TLine lineXrec(muRow, yPadMin, muRow, yPadMax);
    lineXrec.SetLineStyle(2);
    lineXrec.SetLineWidth(2);
    lineXrec.SetLineColor(kRed+1);
    lineXrec.Draw("SAME");
    leg.AddEntry(&lineXrec, "x_{rec}", "l");
    // New: show pixel-centered delta as requested (|x_hit - x_px|)
    leg.AddEntry((TObject*)nullptr, Form("|x_{hit}-x_{px}| = %.1f #mum", 1000.0*std::abs(x_hit - x_px)), "");
    leg.AddEntry((TObject*)nullptr, Form("|x_{hit}-x_{rec,2d}| = %.1f #mum", 1000.0*std::abs(x_hit - muRow)), "");
    leg.Draw();
    c.Print("row.pdf");
    nPagesRow++;

    // Draw column
    c.cd();
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
    TLine lineYhit(y_hit, yPadMinC, y_hit, yPadMaxC);
    lineYhit.SetLineStyle(2);
    lineYhit.SetLineWidth(2);
    lineYhit.SetLineColor(kBlack);
    lineYhit.Draw("SAME");
    double fx1c = gPad->GetLeftMargin();
    double fy1c = gPad->GetBottomMargin();
    double fx2c = 1.0 - gPad->GetRightMargin();
    double fy2c = 1.0 - gPad->GetTopMargin();
    // Slightly increase legend height to accommodate extra entry (|y_hit - y_px|)
    double legWc = 0.22, legHc = 0.18, insetc = 0.008;
    double lx1c = fx1c + insetc, lx2c = lx1c + legWc;
    double ly2c = fy2c - insetc, ly1c = ly2c - legHc;
    TLegend legC(lx1c, ly1c, lx2c, ly2c, "", "NDC");
    legC.SetBorderSize(0);
    legC.SetFillStyle(0);
    legC.SetTextSize(0.03);
    legC.AddEntry(&lineYhit, "y_{hit}", "l");
    TLine lineYrec(muCol, yPadMinC, muCol, yPadMaxC);
    lineYrec.SetLineStyle(2);
    lineYrec.SetLineWidth(2);
    lineYrec.SetLineColor(kRed+1);
    lineYrec.Draw("SAME");
    legC.AddEntry(&lineYrec, "y_{rec}", "l");
    // New: show pixel-centered delta as requested (|y_hit - y_px|)
    legC.AddEntry((TObject*)nullptr, Form("|y_{hit}-y_{px}| = %.1f #mum", 1000.0*std::abs(y_hit - y_px)), "");
    legC.AddEntry((TObject*)nullptr, Form("|y_{hit}-y_{rec,2d}| = %.1f #mum", 1000.0*std::abs(y_hit - muCol)), "");
    legC.Draw();
    c.Print("column.pdf");
    nPagesCol++;
  }

  // Close PDFs
  c.Print("row.pdf]");
  c.Print("column.pdf]");

  file->Close();
  delete file;

  ::Info("processing2D_plots", "Generated %lld row pages and %lld column pages (considered %lld events).", nPagesRow, nPagesCol, nConsidered);
  return 0;
}

