// ROOT macro: processing2D_plots.C
// Produces multi-page PDFs with row and column fits for non-pixel-pad hits
// Output files: row.pdf, column.pdf (each page corresponds to one event)

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph.h>
#include <TF1.h>
#include <TROOT.h>
#include <TError.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TLatex.h>
#include <TStyle.h>
#include <Math/MinimizerOptions.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

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

int processing2D_plots(const char* filename = "../build/epicChargeSharingOutput.root") {
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

  // Pixel spacing: prefer metadata; fallback to inference as in processing2D
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
    ::Error("processing2D_plots", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
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

  // Fitting functions reused per event
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

    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);

    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int irow = di + R;
        const int jcol = dj + R;
        const int idx  = irow * N + jcol;
        const double f = (*Fi)[idx];
        if (!IsFinite(f) || f < 0) continue;
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

    // Common initial params via robust seeding
    auto setupAndFit = [&](TGraph& g, TF1& f, const std::vector<double>& xs, const std::vector<double>& qs) -> int {
      auto minmax = std::minmax_element(qs.begin(), qs.end());
      const double A0 = std::max(1e-18, *minmax.second - *minmax.first);
      const double B0 = std::max(0.0, *minmax.first);
      int idxMax = std::distance(qs.begin(), std::max_element(qs.begin(), qs.end()));
      double mu0 = xs[idxMax];

      f.SetParameters(A0, mu0, std::max(0.25*pixelSpacing, 1e-6), B0);
      // Fractions are unitless in [0,1]
      f.SetParLimits(0, 0.0, 1.0);
      f.SetParLimits(2, 1e-6, 5.0*pixelSpacing);
      f.SetParLimits(3, 0.0, 1.0);

      int status = g.Fit(&f, "QNS");
      if (status != 0) {
        double wsum = 0, xw = 0;
        for (size_t k=0;k<xs.size();++k){ double w = std::max(0.0, qs[k]-B0); wsum += w; xw += w * xs[k]; }
        double muW = (wsum>0)? xw/wsum : mu0;
        f.SetParameters(A0, muW, std::max(0.5*pixelSpacing, 1e-6), B0);
        status = g.Fit(&f, "QNSR");
      }
      return status;
    };

    // Build graphs for BOTH row and column and perform fits with centroid fallback
    TGraph gRow(static_cast<int>(x_row.size()));
    for (int k = 0; k < gRow.GetN(); ++k) gRow.SetPoint(k, x_row[k], q_row[k]);
    TGraph gCol(static_cast<int>(y_col.size()));
    for (int k = 0; k < gCol.GetN(); ++k) gCol.SetPoint(k, y_col[k], q_col[k]);

    // Set titles and styles
    gRow.SetTitle(Form("Event %lld: Central row fit; x [mm]; Charge fraction F_i [unitless]", i));
    gRow.SetMarkerStyle(20);
    gRow.SetMarkerSize(0.9);
    gRow.SetLineColor(kBlue+1);

    gCol.SetTitle(Form("Event %lld: Central column fit; y [mm]; Charge fraction F_i [unitless]", i));
    gCol.SetMarkerStyle(21);
    gCol.SetMarkerSize(0.9);
    gCol.SetLineColor(kBlue+2);

    // Restrict function draw range to data spans and style for visibility
    auto [xminIt, xmaxIt] = std::minmax_element(x_row.begin(), x_row.end());
    double xmin = *xminIt, xmax = *xmaxIt;
    fRow.SetRange(xmin - 0.1*pixelSpacing, xmax + 0.1*pixelSpacing);
    fRow.SetNpx(600);
    fRow.SetLineWidth(2);

    auto [yminIt, ymaxIt] = std::minmax_element(y_col.begin(), y_col.end());
    double ymin = *yminIt, ymax = *ymaxIt;
    fCol.SetRange(ymin - 0.1*pixelSpacing, ymax + 0.1*pixelSpacing);
    fCol.SetNpx(600);
    fCol.SetLineWidth(2);

    // Initial fits
    int statusRow = setupAndFit(gRow, fRow, x_row, q_row);
    int statusCol = setupAndFit(gCol, fCol, y_col, q_col);

    bool okRow = (statusRow == 0);
    bool okCol = (statusCol == 0);
    double muRow = okRow ? fRow.GetParameter(1) : NAN;
    double muCol = okCol ? fCol.GetParameter(1) : NAN;

    // Centroid fallback identical to processing2D.C when fit fails
    if (!okRow) {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k]); wsum += w; xw += w * x_row[k]; }
      if (wsum > 0) { muRow = xw / wsum; okRow = true; }
    }
    if (!okCol) {
      double wsum = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k]); wsum += w; yw += w * y_col[k]; }
      if (wsum > 0) { muCol = yw / wsum; okCol = true; }
    }

    // Only proceed when BOTH reconstructions are valid, matching processing2D.C behavior
    if (!(okRow && okCol) || !IsFinite(muRow) || !IsFinite(muCol)) continue;

    // Add headroom so curves are visible
    double dataMaxRow = *std::max_element(q_row.begin(), q_row.end());
    double fitPeakRow = fRow.GetParameter(0) + fRow.GetParameter(3);
    double yMaxRow = 1.20 * std::max(dataMaxRow, fitPeakRow);
    gRow.SetMaximum(yMaxRow);

    double dataMaxCol = *std::max_element(q_col.begin(), q_col.end());
    double fitPeakCol = fCol.GetParameter(0) + fCol.GetParameter(3);
    double yMaxCol = 1.20 * std::max(dataMaxCol, fitPeakCol);
    gCol.SetMaximum(yMaxCol);

    // Draw row
    c.cd();
    gRow.Draw("AP");
    fRow.SetLineColor(kRed+1);
    fRow.Draw("L SAME");
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
    double legW = 0.22, legH = 0.14, inset = 0.008;
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
    leg.AddEntry((TObject*)nullptr, Form("|x_{hit}-x_{rec,2d}| = %.1f #mum", 1000.0*std::abs(x_hit - muRow)), "");
    leg.Draw();
    c.Print("row.pdf");
    nPagesRow++;

    // Draw column
    c.cd();
    gCol.Draw("AP");
    fCol.SetLineColor(kRed+1);
    fCol.Draw("L SAME");
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
    double legWc = 0.22, legHc = 0.14, insetc = 0.008;
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

