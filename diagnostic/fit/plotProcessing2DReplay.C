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
}

// Visually replay the fits performed by processing2D.C using the saved
// GaussRow/Col{A,Mu,Sigma,B} parameters and GaussRow/ColMaskRemoved masks.
// This mirrors the layout and styling of plotProcessing2D.C, but does no fitting.
// It draws the saved model curves and highlights removed points.
int processing2D_replay(const char* filename = "/home/tom/Desktop/Putza/epicChargeSharing/build/epicChargeSharing.root",
                        double errorPercentOfMax = 5.0,
                        Long64_t nRandomEvents = 100) {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for read (try provided path, then fallback to build/ path)
  TFile* file = TFile::Open(filename, "READ");
  if (!file || file->IsZombie()) {
    if (file) { file->Close(); delete file; file = nullptr; }
    TString fallback = "/home/tom/Desktop/Putza/epicChargeSharing/build/epicChargeSharing.root";
    file = TFile::Open(fallback, "READ");
    if (!file || file->IsZombie()) {
      ::Error("processing2D_replay", "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return 1;
    }
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D_replay", "Hits tree not found in file: %s", filename);
    file->Close();
    delete file;
    return 2;
  }

  // Pixel spacing/size/radius: prefer metadata; fallback to inference
  double pixelSpacing = NAN;
  double pixelSize    = NAN;
  int neighborhoodRadiusMeta = -1;
  if (auto* spacingObj = dynamic_cast<TNamed*>(file->Get("GridPixelSpacing_mm"))) {
    try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) {}
  }
  if (auto* sizeObj = dynamic_cast<TNamed*>(file->Get("GridPixelSize_mm"))) {
    try { pixelSize = std::stod(sizeObj->GetTitle()); } catch (...) {}
  }
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

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
  }
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing2D_replay", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.5 * pixelSpacing; // mm
  }

  // Decide which charge branch to use for plotting points. This only affects point y-values;
  // indices/geometry are identical across Q_f/Q_i/F_i.
  std::string chosenCharge;
  if (tree->GetBranch("Q_f")) chosenCharge = "Q_f";
  else if (tree->GetBranch("F_i")) chosenCharge = "F_i";
  else if (tree->GetBranch("Q_i")) chosenCharge = "Q_i";
  else {
    ::Error("processing2D_replay", "No charge branch found (tried Q_f, F_i, Q_i)");
    file->Close();
    delete file;
    return 4;
  }

  // Set up branches
  double x_true = 0.0, y_true = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Q = nullptr;
  // Saved fit parameters
  double rowA = NAN, rowMu = NAN, rowSig = NAN, rowB = NAN;
  double colA = NAN, colMu = NAN, colSig = NAN, colB = NAN;
  // Saved removal masks (0 = kept, 1 = removed)
  std::vector<int>* rowMask = nullptr;
  std::vector<int>* colMask = nullptr;

  tree->SetBranchAddress("TrueX", &x_true);
  tree->SetBranchAddress("TrueY", &y_true);
  tree->SetBranchAddress("PixelX",  &x_px);
  tree->SetBranchAddress("PixelY",  &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_true);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);

  bool haveRowA = tree->GetBranch("GaussRowA") != nullptr;
  bool haveRowMu = tree->GetBranch("GaussRowMu") != nullptr;
  bool haveRowSig = tree->GetBranch("GaussRowSigma") != nullptr;
  bool haveRowB = tree->GetBranch("GaussRowB") != nullptr;
  bool haveColA = tree->GetBranch("GaussColA") != nullptr;
  bool haveColMu = tree->GetBranch("GaussColMu") != nullptr;
  bool haveColSig = tree->GetBranch("GaussColSigma") != nullptr;
  bool haveColB = tree->GetBranch("GaussColB") != nullptr;
  bool haveAnyParams = haveRowMu && haveColMu; // at least mus to draw rec lines
  if (haveRowA) tree->SetBranchAddress("GaussRowA", &rowA);
  if (haveRowMu) tree->SetBranchAddress("GaussRowMu", &rowMu);
  if (haveRowSig) tree->SetBranchAddress("GaussRowSigma", &rowSig);
  if (haveRowB) tree->SetBranchAddress("GaussRowB", &rowB);
  if (haveColA) tree->SetBranchAddress("GaussColA", &colA);
  if (haveColMu) tree->SetBranchAddress("GaussColMu", &colMu);
  if (haveColSig) tree->SetBranchAddress("GaussColSigma", &colSig);
  if (haveColB) tree->SetBranchAddress("GaussColB", &colB);

  bool haveRowMask = tree->GetBranch("GaussRowMaskRemoved") != nullptr;
  bool haveColMask = tree->GetBranch("GaussColMaskRemoved") != nullptr;
  if (haveRowMask) tree->SetBranchAddress("GaussRowMaskRemoved", &rowMask);
  if (haveColMask) tree->SetBranchAddress("GaussColMaskRemoved", &colMask);

  // Prepare plotting objects
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  TCanvas c("c2up_replay", "processing2D replay", 1800, 700);
  TPad pL("pL", "column-left", 0.0, 0.0, 0.5, 1.0);
  TPad pR("pR", "row-right",  0.5, 0.0, 1.0, 1.0);
  pL.SetTicks(1,1);
  pR.SetTicks(1,1);
  pL.Draw();
  pR.Draw();

  // Open multipage PDF
  c.Print("2Dfits_replay.pdf[");

  TF1 fRow("fRow_replay", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol_replay", GaussPlusB, -1e9, 1e9, 4);

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

    if (is_pixel_true || !Q || Q->empty()) continue;

    const size_t total = Q->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Build full row/column samples from raw charges
    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);
    double qmaxNeighborhood = -1e300;
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx  = (di + R) * N + (dj + R);
        const double q = (*Q)[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        if (dj == 0) {
          const double x = x_px + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
        }
        if (di == 0) {
          const double y = y_px + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
        }
      }
    }
    if (x_row.size() < 3 || y_col.size() < 3) continue;

    // Determine which points were kept/removed per saved masks (if available)
    std::vector<char> rowKeptMask(x_row.size(), 1);
    std::vector<char> colKeptMask(y_col.size(), 1);
    if (haveRowMask && rowMask && !rowMask->empty() && rowMask->size() == x_row.size()) {
      for (size_t k=0;k<rowMask->size();++k) rowKeptMask[k] = ((*rowMask)[k] == 0) ? 1 : 0;
    }
    if (haveColMask && colMask && !colMask->empty() && colMask->size() == y_col.size()) {
      for (size_t k=0;k<colMask->size();++k) colKeptMask[k] = ((*colMask)[k] == 0) ? 1 : 0;
    }
    // If fewer than 3 points were kept, fall back to showing all points as kept
    auto countKept = [](const std::vector<char>& v){ size_t c=0; for(char b: v) if (b) ++c; return c; };
    if (countKept(rowKeptMask) < 3) std::fill(rowKeptMask.begin(), rowKeptMask.end(), 1);
    if (countKept(colKeptMask) < 3) std::fill(colKeptMask.begin(), colKeptMask.end(), 1);

    // Reconstruct the filtered sets used for fitting (kept points only)
    std::vector<double> x_row_fit, q_row_fit, y_col_fit, q_col_fit;
    x_row_fit.reserve(x_row.size()); q_row_fit.reserve(q_row.size());
    y_col_fit.reserve(y_col.size()); q_col_fit.reserve(q_col.size());
    for (size_t k=0;k<x_row.size();++k) if (rowKeptMask[k]) { x_row_fit.push_back(x_row[k]); q_row_fit.push_back(q_row[k]); }
    for (size_t k=0;k<y_col.size();++k) if (colKeptMask[k]) { y_col_fit.push_back(y_col[k]); q_col_fit.push_back(q_col[k]); }
    // Hard minimum for replay as well
    if ((int)x_row_fit.size() < 3) { x_row_fit = x_row; q_row_fit = q_row; }
    if ((int)y_col_fit.size() < 3) { y_col_fit = y_col; q_col_fit = q_col; }

    // Must have mus to draw recon lines. Prefer saved parameters when present.
    bool haveMuRow = haveRowMu && IsFinite(rowMu);
    bool haveMuCol = haveColMu && IsFinite(colMu);
    if (!(haveMuRow && haveMuCol)) continue;

    nConsidered++;

    // Optional uniform vertical uncertainty from percent-of-max
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    // Build graphs for kept points only (to mirror original drawing)
    bool useWeights = (uniformSigma > 0.0);
    TGraph* baseRowPtr = nullptr;
    TGraph* baseColPtr = nullptr;
    TGraph gRowPlain;
    TGraph gColPlain;
    TGraphErrors gRowErr;
    TGraphErrors gColErr;

    if (useWeights) {
      gRowErr = TGraphErrors(static_cast<int>(x_row_fit.size()));
      for (int k = 0; k < gRowErr.GetN(); ++k) {
        gRowErr.SetPoint(k, x_row_fit[k], q_row_fit[k]);
        gRowErr.SetPointError(k, 0.0, uniformSigma);
      }
      baseRowPtr = &gRowErr;
      gColErr = TGraphErrors(static_cast<int>(y_col_fit.size()));
      for (int k = 0; k < gColErr.GetN(); ++k) {
        gColErr.SetPoint(k, y_col_fit[k], q_col_fit[k]);
        gColErr.SetPointError(k, 0.0, uniformSigma);
      }
      baseColPtr = &gColErr;
    } else {
      gRowPlain = TGraph(static_cast<int>(x_row_fit.size()));
      for (int k = 0; k < gRowPlain.GetN(); ++k) gRowPlain.SetPoint(k, x_row_fit[k], q_row_fit[k]);
      baseRowPtr = &gRowPlain;
      gColPlain = TGraph(static_cast<int>(y_col_fit.size()));
      for (int k = 0; k < gColPlain.GetN(); ++k) gColPlain.SetPoint(k, y_col_fit[k], q_col_fit[k]);
      baseColPtr = &gColPlain;
    }

    baseRowPtr->SetTitle(Form("Event %lld: Central row fit; x [mm]; Noisy charge %s [C]", eventIndex, chosenCharge.c_str()));
    baseRowPtr->SetMarkerStyle(20);
    baseRowPtr->SetMarkerSize(0.9);
    baseRowPtr->SetLineColor(kBlue+1);
    baseColPtr->SetTitle(Form("Event %lld: Central column fit; y [mm]; Noisy charge %s [C]", eventIndex, chosenCharge.c_str()));
    baseColPtr->SetMarkerStyle(21);
    baseColPtr->SetMarkerSize(0.9);
    baseColPtr->SetLineColor(kBlue+2);

    // Determine Y-axis limits using all samples (kept+removed) and model baseline/peak
    double dataMinRow = *std::min_element(q_row.begin(), q_row.end());
    double dataMaxRow = *std::max_element(q_row.begin(), q_row.end());
    double yMinRow = dataMinRow;
    double yMaxRow = dataMaxRow;
    if (haveRowA && IsFinite(rowA) && haveRowB && IsFinite(rowB)) {
      yMaxRow = std::max(yMaxRow, rowA + rowB);
      yMinRow = std::min(yMinRow, rowB);
    }
    double padRow = 0.10 * (yMaxRow - yMinRow);
    if (!(padRow > 0)) padRow = 0.10 * std::max(std::abs(yMaxRow), 1.0);
    baseRowPtr->SetMinimum(yMinRow - padRow);
    baseRowPtr->SetMaximum(yMaxRow + padRow);

    double dataMinCol = *std::min_element(q_col.begin(), q_col.end());
    double dataMaxCol = *std::max_element(q_col.begin(), q_col.end());
    double yMinCol = dataMinCol;
    double yMaxCol = dataMaxCol;
    if (haveColA && IsFinite(colA) && haveColB && IsFinite(colB)) {
      yMaxCol = std::max(yMaxCol, colA + colB);
      yMinCol = std::min(yMinCol, colB);
    }
    double padCol = 0.10 * (yMaxCol - yMinCol);
    if (!(padCol > 0)) padCol = 0.10 * std::max(std::abs(yMaxCol), 1.0);
    baseColPtr->SetMinimum(yMinCol - padCol);
    baseColPtr->SetMaximum(yMaxCol + padCol);

    // Draw COLUMN (left)
    pL.cd();
    baseColPtr->Draw(useWeights ? "AP" : "AP");
    // Ensure x-axis covers the full neighborhood, not only kept points
    {
      auto minmaxY_all = std::minmax_element(y_col.begin(), y_col.end());
      const double yAxisMin = *minmaxY_all.first - 0.5 * pixelSpacing;
      const double yAxisMax = *minmaxY_all.second + 0.5 * pixelSpacing;
      baseColPtr->GetXaxis()->SetLimits(yAxisMin, yAxisMax);
    }
    // Draw saved fit curve if parameters exist
    bool didColFit = haveColA && haveColMu && haveColSig && haveColB && IsFinite(colA) && IsFinite(colMu) && IsFinite(colSig) && IsFinite(colB) && colSig > 0;
    if (didColFit) {
      auto minmaxY = std::minmax_element(y_col.begin(), y_col.end());
      const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
      const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
      fCol.SetRange(yMin, yMax);
      fCol.SetParameters(colA, colMu, colSig, colB);
      fCol.SetNpx(600);
      fCol.SetLineWidth(2);
      fCol.SetLineColor(kRed+1);
      fCol.Draw("L SAME");
    }
    gPad->Update();
    double yPadMinC = gPad->GetUymin();
    double yPadMaxC = gPad->GetUymax();
    // Pixel-width rectangles for all points, with low opacity for removed
    {
      const double halfHc = 0.015 * (yPadMaxC - yPadMinC);
      for (size_t k = 0; k < y_col.size(); ++k) {
        const double xc = y_col[k];
        const double yc = q_col[k];
        const double xlo = xc - 0.5 * pixelSize;
        const double xhi = xc + 0.5 * pixelSize;
        const double ylo = yc - halfHc;
        const double yhi = yc + halfHc;
        TBox* box = new TBox(xlo, ylo, xhi, yhi);
        box->SetFillStyle(0);
        if (k < colKeptMask.size() && colKeptMask[k]) { box->SetLineColor(kGray+2); }
        else { box->SetLineColorAlpha(kGray+2, 0.25); }
        box->SetLineWidth(1);
        box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
      // Draw red X markers on removed points for extra visibility
      for (size_t k = 0; k < y_col.size(); ++k) {
        if (!(k < colKeptMask.size() && colKeptMask[k])) {
          const double xc = y_col[k];
          const double yc = q_col[k];
          const double dx = 0.15 * pixelSize;
          TLine* l1 = new TLine(xc - dx, yc - 0.5*halfHc, xc + dx, yc + 0.5*halfHc);
          TLine* l2 = new TLine(xc - dx, yc + 0.5*halfHc, xc + dx, yc - 0.5*halfHc);
          l1->SetLineColor(kRed+1); l2->SetLineColor(kRed+1);
          l1->SetLineWidth(2); l2->SetLineWidth(2);
          l1->SetBit(kCanDelete); l2->SetBit(kCanDelete);
          l1->Draw("SAME"); l2->Draw("SAME");
        }
      }
    }
    TLine lineYtrue(y_true, yPadMinC, y_true, yPadMaxC);
    lineYtrue.SetLineStyle(2);
    lineYtrue.SetLineWidth(2);
    lineYtrue.SetLineColor(kBlack);
    lineYtrue.Draw("SAME");
    TLine lineYrec(colMu, yPadMinC, colMu, yPadMaxC);
    lineYrec.SetLineStyle(2);
    lineYrec.SetLineWidth(2);
    lineYrec.SetLineColor(kRed+1);
    lineYrec.Draw("SAME");

    double fx1c = gPad->GetLeftMargin();
    double fy1c = gPad->GetBottomMargin();
    double fx2c = 1.0 - gPad->GetRightMargin();
    double fy2c = 1.0 - gPad->GetTopMargin();
    double legWc = 0.28, legHc = 0.30, insetc = 0.008;
    double ly2c = fy2c - insetc, ly1c = ly2c - legHc;
    double lx1c = fx1c + insetc, lx2c = lx1c + legWc;
    TLegend legCLeft(lx1c, ly1c, lx2c, ly2c, "", "NDC");
    legCLeft.SetBorderSize(0);
    legCLeft.SetFillStyle(0);
    legCLeft.SetTextSize(0.03);
    legCLeft.AddEntry(&lineYtrue, "y_{true}", "l");
    legCLeft.AddEntry(&lineYrec,  "y_{rec}",  "l");
    legCLeft.Draw();
    double rx2c = fx2c - insetc, rx1c = rx2c - legWc;
    TLegend legCRight(rx1c, ly1c, rx2c, ly2c, "", "NDC");
    legCRight.SetBorderSize(0);
    legCRight.SetFillStyle(0);
    legCRight.SetTextSize(0.03);
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true} = %.4f mm", y_true), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{rec} = %.4f mm", colMu), "");
    if (didColFit) legCRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", colSig), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{px} = %.1f #mum", 1000.0*(y_true - y_px)), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{rec} = %.1f #mum", 1000.0*(y_true - colMu)), "");
    legCRight.Draw();

    // Draw ROW (right)
    pR.cd();
    baseRowPtr->Draw(useWeights ? "AP" : "AP");
    // Ensure x-axis covers the full neighborhood, not only kept points
    {
      auto minmaxX_all = std::minmax_element(x_row.begin(), x_row.end());
      const double xAxisMin = *minmaxX_all.first - 0.5 * pixelSpacing;
      const double xAxisMax = *minmaxX_all.second + 0.5 * pixelSpacing;
      baseRowPtr->GetXaxis()->SetLimits(xAxisMin, xAxisMax);
    }
    bool didRowFit = haveRowA && haveRowMu && haveRowSig && haveRowB && IsFinite(rowA) && IsFinite(rowMu) && IsFinite(rowSig) && IsFinite(rowB) && rowSig > 0;
    if (didRowFit) {
      auto minmaxX = std::minmax_element(x_row.begin(), x_row.end());
      const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
      const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
      fRow.SetRange(xMin, xMax);
      fRow.SetParameters(rowA, rowMu, rowSig, rowB);
      fRow.SetNpx(600);
      fRow.SetLineWidth(2);
      fRow.SetLineColor(kRed+1);
      fRow.Draw("L SAME");
    }
    gPad->Update();
    double yPadMin = gPad->GetUymin();
    double yPadMax = gPad->GetUymax();
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
        if (k < rowKeptMask.size() && rowKeptMask[k]) { box->SetLineColor(kGray+2); }
        else { box->SetLineColorAlpha(kGray+2, 0.25); }
        box->SetLineWidth(1);
        box->SetBit(kCanDelete);
        box->Draw("SAME L");
      }
      for (size_t k = 0; k < x_row.size(); ++k) {
        if (!(k < rowKeptMask.size() && rowKeptMask[k])) {
          const double xc = x_row[k];
          const double yc = q_row[k];
          const double dx = 0.15 * pixelSize;
          TLine* l1 = new TLine(xc - dx, yc - 0.5*halfH, xc + dx, yc + 0.5*halfH);
          TLine* l2 = new TLine(xc - dx, yc + 0.5*halfH, xc + dx, yc - 0.5*halfH);
          l1->SetLineColor(kRed+1); l2->SetLineColor(kRed+1);
          l1->SetLineWidth(2); l2->SetLineWidth(2);
          l1->SetBit(kCanDelete); l2->SetBit(kCanDelete);
          l1->Draw("SAME"); l2->Draw("SAME");
        }
      }
    }
    TLine lineXtrue(x_true, yPadMin, x_true, yPadMax);
    lineXtrue.SetLineStyle(2);
    lineXtrue.SetLineWidth(2);
    lineXtrue.SetLineColor(kBlack);
    lineXtrue.Draw("SAME");
    TLine lineXrec(rowMu, yPadMin, rowMu, yPadMax);
    lineXrec.SetLineStyle(2);
    lineXrec.SetLineWidth(2);
    lineXrec.SetLineColor(kRed+1);
    lineXrec.Draw("SAME");

    double fx1 = gPad->GetLeftMargin();
    double fy1 = gPad->GetBottomMargin();
    double fx2 = 1.0 - gPad->GetRightMargin();
    double fy2 = 1.0 - gPad->GetTopMargin();
    double legW = 0.28, legH = 0.30, inset = 0.008;
    double ly2 = fy2 - inset, ly1 = ly2 - legH;
    double lx1 = fx1 + inset, lx2 = lx1 + legW;
    TLegend legLeft(lx1, ly1, lx2, ly2, "", "NDC");
    legLeft.SetBorderSize(0);
    legLeft.SetFillStyle(0);
    legLeft.SetTextSize(0.03);
    legLeft.AddEntry(&lineXtrue, "x_{true}", "l");
    legLeft.AddEntry(&lineXrec,  "x_{rec}",  "l");
    legLeft.Draw();
    double rx2 = fx2 - inset, rx1 = rx2 - legW;
    TLegend legRight(rx1, ly1, rx2, ly2, "", "NDC");
    legRight.SetBorderSize(0);
    legRight.SetFillStyle(0);
    legRight.SetTextSize(0.03);
    legRight.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{rec} = %.4f mm", rowMu), "");
    if (didRowFit) legRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", rowSig), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{px} = %.1f #mum", 1000.0*(x_true - x_px)), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec} = %.1f #mum", 1000.0*(x_true - rowMu)), "");
    legRight.Draw();

    // Print page
    c.cd();
    c.Print("2Dfits_replay.pdf");
    nPages++;
  }

  c.Print("2Dfits_replay.pdf]");

  file->Close();
  delete file;

  ::Info("processing2D_replay", "Generated %lld pages (considered %lld events).", nPages, nConsidered);
  return 0;
}


// ROOT auto-exec wrappers so `root -l -b -q plotProcessing2DReplay.C` works
int plotProcessing2DReplay() {
  return processing2D_replay();
}

int plotProcessing2DReplay(const char* filename, double errorPercentOfMax) {
  return processing2D_replay(filename, errorPercentOfMax);
}

int plotProcessing2DReplay(Long64_t nRandomEvents) {
  return processing2D_replay("../build/epicChargeSharing.root", 5.0, nRandomEvents);
}

int plotProcessing2DReplay(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents) {
  return processing2D_replay(filename, errorPercentOfMax, nRandomEvents);
}


