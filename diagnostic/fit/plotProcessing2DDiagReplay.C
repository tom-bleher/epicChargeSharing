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
#include <utility>

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

// Visually replay the fits performed by processing2D.C (diagonals) using the saved
// GaussDiag(Main/Second){A,Mu,Sigma,B} parameters. Mirrors the layout/styling of
// plotProcessing2DReplay.C but for the two diagonals.
int processing2D_diag_replay(const char* filename = "/home/tom/Desktop/Putza/epicChargeSharing/build/epicChargeSharing.root",
                             double errorPercentOfMax = 5.0,
                             Long64_t nRandomEvents = 100,
                             bool plotQiOverlay = true) {
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
      ::Error("processing2D_diag_replay", "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return 1;
    }
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing2D_diag_replay", "Hits tree not found in file: %s", filename);
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
    ::Error("processing2D_diag_replay", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.5 * pixelSpacing; // mm
  }

  // Decide which charge branch to use for plotting points. This only affects point y-values.
  std::string chosenCharge;
  if (tree->GetBranch("Q_f")) chosenCharge = "Q_f";
  else if (tree->GetBranch("F_i")) chosenCharge = "F_i";
  else if (tree->GetBranch("Q_i")) chosenCharge = "Q_i";
  else {
    ::Error("processing2D_diag_replay", "No charge branch found (tried Q_f, F_i, Q_i)");
    file->Close();
    delete file;
    return 4;
  }

  // Set up branches
  double x_true = 0.0, y_true = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Q = nullptr;
  std::vector<double>* QiVec = nullptr; // Optional overlay source
  // Saved diagonal fit parameters
  double d1A = NAN, d1Mu = NAN, d1Sig = NAN, d1B = NAN;
  double d2A = NAN, d2Mu = NAN, d2Sig = NAN, d2B = NAN;

  tree->SetBranchAddress("TrueX", &x_true);
  tree->SetBranchAddress("TrueY", &y_true);
  tree->SetBranchAddress("PixelX",  &x_px);
  tree->SetBranchAddress("PixelY",  &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_true);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);
  const bool haveQiBranch = (tree->GetBranch("Q_i") != nullptr);
  if (haveQiBranch) {
    tree->SetBranchAddress("Q_i", &QiVec);
  }

  bool haveD1A = tree->GetBranch("GaussDiagMainA") != nullptr;
  bool haveD1Mu = tree->GetBranch("GaussDiagMainMu") != nullptr;
  bool haveD1Sig = tree->GetBranch("GaussDiagMainSigma") != nullptr;
  bool haveD1B = tree->GetBranch("GaussDiagMainB") != nullptr;
  bool haveD2A = tree->GetBranch("GaussDiagSecondA") != nullptr;
  bool haveD2Mu = tree->GetBranch("GaussDiagSecondMu") != nullptr;
  bool haveD2Sig = tree->GetBranch("GaussDiagSecondSigma") != nullptr;
  bool haveD2B = tree->GetBranch("GaussDiagSecondB") != nullptr;

  if (haveD1A) tree->SetBranchAddress("GaussDiagMainA", &d1A);
  if (haveD1Mu) tree->SetBranchAddress("GaussDiagMainMu", &d1Mu);
  if (haveD1Sig) tree->SetBranchAddress("GaussDiagMainSigma", &d1Sig);
  if (haveD1B) tree->SetBranchAddress("GaussDiagMainB", &d1B);
  if (haveD2A) tree->SetBranchAddress("GaussDiagSecondA", &d2A);
  if (haveD2Mu) tree->SetBranchAddress("GaussDiagSecondMu", &d2Mu);
  if (haveD2Sig) tree->SetBranchAddress("GaussDiagSecondSigma", &d2Sig);
  if (haveD2B) tree->SetBranchAddress("GaussDiagSecondB", &d2B);

  // Prepare plotting objects
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  TCanvas c("c2up_replay_diag", "processing2D diagonal replay", 1800, 700);
  TPad pL("pL", "main-diagonal-left", 0.0, 0.0, 0.5, 1.0);
  TPad pR("pR", "secondary-diagonal-right",  0.5, 0.0, 1.0, 1.0);
  pL.SetTicks(1,1);
  pR.SetTicks(1,1);
  pL.Draw();
  pR.Draw();

  // Open multipage PDF
  c.Print("2Dfits_diag_replay.pdf[");

  TF1 fD1("fDiag1_replay", GaussPlusB, -1e9, 1e9, 4);
  TF1 fD2("fDiag2_replay", GaussPlusB, -1e9, 1e9, 4);

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

    // Build main (dj=di) and secondary (dj=-di) diagonal samples
    std::vector<double> s_d1; s_d1.reserve(N);
    std::vector<double> q_d1; q_d1.reserve(N);
    std::vector<int> idx_d1; idx_d1.reserve(N);
    std::vector<double> s_d2; s_d2.reserve(N);
    std::vector<double> q_d2; q_d2.reserve(N);
    std::vector<int> idx_d2; idx_d2.reserve(N);

    double qmaxNeighborhood = -1e300;
    for (int k = -R; k <= R; ++k) {
      // main diagonal
      {
        const int idx = (k + R) * N + (k + R);
        const double q = (*Q)[idx];
        if (IsFinite(q) && q >= 0.0) {
          const double s = x_px + k * pixelSpacing; // use X as the diagonal coordinate
          s_d1.push_back(s);
          q_d1.push_back(q);
          idx_d1.push_back(idx);
          if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        }
      }
      // secondary diagonal
      {
        const int idx = (k + R) * N + (-k + R);
        const double q = (*Q)[idx];
        if (IsFinite(q) && q >= 0.0) {
          const double s = x_px + k * pixelSpacing; // use X as the diagonal coordinate
          s_d2.push_back(s);
          q_d2.push_back(q);
          idx_d2.push_back(idx);
          if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        }
      }
    }
    if (s_d1.size() < 3 || s_d2.size() < 3) continue;

    // Must have mus to draw recon lines. Prefer saved parameters when present.
    bool haveMuD1 = haveD1Mu && IsFinite(d1Mu);
    bool haveMuD2 = haveD2Mu && IsFinite(d2Mu);
    if (!(haveMuD1 && haveMuD2)) continue;

    nConsidered++;

    // Optional uniform vertical uncertainty from percent-of-max
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    // Build graphs for all points (no saved diag masks available)
    bool useWeights = (uniformSigma > 0.0);
    TGraph* baseD1Ptr = nullptr;
    TGraph* baseD2Ptr = nullptr;
    TGraph gD1Plain;
    TGraph gD2Plain;
    TGraphErrors gD1Err;
    TGraphErrors gD2Err;

    if (useWeights) {
      gD1Err = TGraphErrors(static_cast<int>(s_d1.size()));
      for (int k = 0; k < gD1Err.GetN(); ++k) {
        gD1Err.SetPoint(k, s_d1[k], q_d1[k]);
        gD1Err.SetPointError(k, 0.0, uniformSigma);
      }
      baseD1Ptr = &gD1Err;
      gD2Err = TGraphErrors(static_cast<int>(s_d2.size()));
      for (int k = 0; k < gD2Err.GetN(); ++k) {
        gD2Err.SetPoint(k, s_d2[k], q_d2[k]);
        gD2Err.SetPointError(k, 0.0, uniformSigma);
      }
      baseD2Ptr = &gD2Err;
    } else {
      gD1Plain = TGraph(static_cast<int>(s_d1.size()));
      for (int k = 0; k < gD1Plain.GetN(); ++k) gD1Plain.SetPoint(k, s_d1[k], q_d1[k]);
      baseD1Ptr = &gD1Plain;
      gD2Plain = TGraph(static_cast<int>(s_d2.size()));
      for (int k = 0; k < gD2Plain.GetN(); ++k) gD2Plain.SetPoint(k, s_d2[k], q_d2[k]);
      baseD2Ptr = &gD2Plain;
    }

    baseD1Ptr->SetTitle(Form("Event %lld: Main diagonal fit (d_{j}=d_{i}); s=x [mm]; Noisy charge %s [C]", eventIndex, chosenCharge.c_str()));
    baseD1Ptr->SetMarkerStyle(20);
    baseD1Ptr->SetMarkerSize(0.9);
    baseD1Ptr->SetLineColor(kBlue+1);
    baseD2Ptr->SetTitle(Form("Event %lld: Secondary diagonal fit (d_{j}=-d_{i}); s=x [mm]; Noisy charge %s [C]", eventIndex, chosenCharge.c_str()));
    baseD2Ptr->SetMarkerStyle(21);
    baseD2Ptr->SetMarkerSize(0.9);
    baseD2Ptr->SetLineColor(kBlue+2);

    // Determine Y-axis limits using samples and model baseline/peak
    auto minmaxD1 = std::minmax_element(q_d1.begin(), q_d1.end());
    double dataMinD1 = *minmaxD1.first;
    double dataMaxD1 = *minmaxD1.second;
    double yMinD1 = dataMinD1;
    double yMaxD1 = dataMaxD1;
    if (haveD1A && IsFinite(d1A) && haveD1B && IsFinite(d1B)) {
      yMaxD1 = std::max(yMaxD1, d1A + d1B);
      yMinD1 = std::min(yMinD1, d1B);
    }
    double padD1 = 0.10 * (yMaxD1 - yMinD1);
    if (!(padD1 > 0)) padD1 = 0.10 * std::max(std::abs(yMaxD1), 1.0);
    baseD1Ptr->SetMinimum(yMinD1 - padD1);
    baseD1Ptr->SetMaximum(yMaxD1 + padD1);

    auto minmaxD2 = std::minmax_element(q_d2.begin(), q_d2.end());
    double dataMinD2 = *minmaxD2.first;
    double dataMaxD2 = *minmaxD2.second;
    double yMinD2 = dataMinD2;
    double yMaxD2 = dataMaxD2;
    if (haveD2A && IsFinite(d2A) && haveD2B && IsFinite(d2B)) {
      yMaxD2 = std::max(yMaxD2, d2A + d2B);
      yMinD2 = std::min(yMinD2, d2B);
    }
    double padD2 = 0.10 * (yMaxD2 - yMinD2);
    if (!(padD2 > 0)) padD2 = 0.10 * std::max(std::abs(yMaxD2), 1.0);
    baseD2Ptr->SetMinimum(yMinD2 - padD2);
    baseD2Ptr->SetMaximum(yMaxD2 + padD2);

    // Draw MAIN DIAGONAL (left)
    pL.cd();
    baseD1Ptr->Draw(useWeights ? "AP" : "AP");
    // Ensure x-axis covers the full diagonal extent
    {
      auto minmaxS = std::minmax_element(s_d1.begin(), s_d1.end());
      const double sAxisMin = *minmaxS.first - 0.5 * pixelSpacing;
      const double sAxisMax = *minmaxS.second + 0.5 * pixelSpacing;
      baseD1Ptr->GetXaxis()->SetLimits(sAxisMin, sAxisMax);
    }
    bool didD1Fit = haveD1A && haveD1Mu && haveD1Sig && haveD1B && IsFinite(d1A) && IsFinite(d1Mu) && IsFinite(d1Sig) && IsFinite(d1B) && d1Sig > 0;
    if (didD1Fit) {
      auto minmaxS = std::minmax_element(s_d1.begin(), s_d1.end());
      const double sMin = *minmaxS.first - 0.5 * pixelSpacing;
      const double sMax = *minmaxS.second + 0.5 * pixelSpacing;
      fD1.SetRange(sMin, sMax);
      fD1.SetParameters(d1A, d1Mu, d1Sig, d1B);
      fD1.SetNpx(600);
      fD1.SetLineWidth(2);
      fD1.SetLineColor(kRed+1);
      fD1.Draw("L SAME");
    }
    gPad->Update();
    double yPadMin1 = gPad->GetUymin();
    double yPadMax1 = gPad->GetUymax();
    // Pixel-width rectangles for all points
    {
      const double halfH = 0.015 * (yPadMax1 - yPadMin1);
      for (size_t k = 0; k < s_d1.size(); ++k) {
        const double xc = s_d1[k];
        const double yc = q_d1[k];
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
    // Optional overlay of Q_i points
    bool drewD1Qi = false;
    TGraph gD1Qi;
    const bool canOverlayQi = plotQiOverlay && haveQiBranch && (chosenCharge != "Q_i");
    if (canOverlayQi && QiVec) {
      std::vector<std::pair<double,double>> d1Points;
      d1Points.reserve(idx_d1.size());
      for (size_t k = 0; k < idx_d1.size(); ++k) {
        const int idx = idx_d1[k];
        if (idx >= 0 && idx < static_cast<int>(QiVec->size())) {
          const double qqi = (*QiVec)[idx];
          if (IsFinite(qqi) && qqi >= 0.0) d1Points.emplace_back(s_d1[k], qqi);
        }
      }
      if (!d1Points.empty()) {
        gD1Qi = TGraph(static_cast<int>(d1Points.size()));
        for (int k = 0; k < gD1Qi.GetN(); ++k) gD1Qi.SetPoint(k, d1Points[k].first, d1Points[k].second);
        gD1Qi.SetMarkerStyle(24);
        gD1Qi.SetMarkerSize(1.1);
        gD1Qi.SetMarkerColor(kGreen+2);
        gD1Qi.SetLineColor(kGreen+2);
        gD1Qi.Draw("P SAME");
        drewD1Qi = true;
      }
    }
    TLine lineStrue1(x_true, yPadMin1, x_true, yPadMax1);
    lineStrue1.SetLineStyle(2);
    lineStrue1.SetLineWidth(2);
    lineStrue1.SetLineColor(kBlack);
    lineStrue1.Draw("SAME");
    TLine lineSrec1(d1Mu, yPadMin1, d1Mu, yPadMax1);
    lineSrec1.SetLineStyle(2);
    lineSrec1.SetLineWidth(2);
    lineSrec1.SetLineColor(kRed+1);
    lineSrec1.Draw("SAME");

    {
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
      legLeft.AddEntry(&lineStrue1, "x_{true}", "l");
      legLeft.AddEntry(&lineSrec1,  "x_{rec}",  "l");
      if (drewD1Qi) legLeft.AddEntry(&gD1Qi, "Q_{i} points", "p");
      legLeft.Draw();
      double rx2 = fx2 - inset, rx1 = rx2 - legW;
      TLegend legRight(rx1, ly1, rx2, ly2, "", "NDC");
      legRight.SetBorderSize(0);
      legRight.SetFillStyle(0);
      legRight.SetTextSize(0.03);
      legRight.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{rec} = %.4f mm", d1Mu), "");
      if (didD1Fit) legRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", d1Sig), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{px} = %.1f #mum", 1000.0*(x_true - x_px)), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec} = %.1f #mum", 1000.0*(x_true - d1Mu)), "");
      legRight.Draw();
    }

    // Draw SECONDARY DIAGONAL (right)
    pR.cd();
    baseD2Ptr->Draw(useWeights ? "AP" : "AP");
    {
      auto minmaxS = std::minmax_element(s_d2.begin(), s_d2.end());
      const double sAxisMin = *minmaxS.first - 0.5 * pixelSpacing;
      const double sAxisMax = *minmaxS.second + 0.5 * pixelSpacing;
      baseD2Ptr->GetXaxis()->SetLimits(sAxisMin, sAxisMax);
    }
    bool didD2Fit = haveD2A && haveD2Mu && haveD2Sig && haveD2B && IsFinite(d2A) && IsFinite(d2Mu) && IsFinite(d2Sig) && IsFinite(d2B) && d2Sig > 0;
    if (didD2Fit) {
      auto minmaxS = std::minmax_element(s_d2.begin(), s_d2.end());
      const double sMin = *minmaxS.first - 0.5 * pixelSpacing;
      const double sMax = *minmaxS.second + 0.5 * pixelSpacing;
      fD2.SetRange(sMin, sMax);
      fD2.SetParameters(d2A, d2Mu, d2Sig, d2B);
      fD2.SetNpx(600);
      fD2.SetLineWidth(2);
      fD2.SetLineColor(kRed+1);
      fD2.Draw("L SAME");
    }
    gPad->Update();
    double yPadMin2 = gPad->GetUymin();
    double yPadMax2 = gPad->GetUymax();
    {
      const double halfH = 0.015 * (yPadMax2 - yPadMin2);
      for (size_t k = 0; k < s_d2.size(); ++k) {
        const double xc = s_d2[k];
        const double yc = q_d2[k];
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
    bool drewD2Qi = false;
    TGraph gD2Qi;
    if (canOverlayQi && QiVec) {
      std::vector<std::pair<double,double>> d2Points;
      d2Points.reserve(idx_d2.size());
      for (size_t k = 0; k < idx_d2.size(); ++k) {
        const int idx = idx_d2[k];
        if (idx >= 0 && idx < static_cast<int>(QiVec->size())) {
          const double qqi = (*QiVec)[idx];
          if (IsFinite(qqi) && qqi >= 0.0) d2Points.emplace_back(s_d2[k], qqi);
        }
      }
      if (!d2Points.empty()) {
        gD2Qi = TGraph(static_cast<int>(d2Points.size()));
        for (int k = 0; k < gD2Qi.GetN(); ++k) gD2Qi.SetPoint(k, d2Points[k].first, d2Points[k].second);
        gD2Qi.SetMarkerStyle(24);
        gD2Qi.SetMarkerSize(1.1);
        gD2Qi.SetMarkerColor(kGreen+2);
        gD2Qi.SetLineColor(kGreen+2);
        gD2Qi.Draw("P SAME");
        drewD2Qi = true;
      }
    }
    TLine lineStrue2(x_true, yPadMin2, x_true, yPadMax2);
    lineStrue2.SetLineStyle(2);
    lineStrue2.SetLineWidth(2);
    lineStrue2.SetLineColor(kBlack);
    lineStrue2.Draw("SAME");
    TLine lineSrec2(d2Mu, yPadMin2, d2Mu, yPadMax2);
    lineSrec2.SetLineStyle(2);
    lineSrec2.SetLineWidth(2);
    lineSrec2.SetLineColor(kRed+1);
    lineSrec2.Draw("SAME");

    {
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
      legLeft.AddEntry(&lineStrue2, "x_{true}", "l");
      legLeft.AddEntry(&lineSrec2,  "x_{rec}",  "l");
      if (drewD2Qi) legLeft.AddEntry(&gD2Qi, "Q_{i} points", "p");
      legLeft.Draw();
      double rx2 = fx2 - inset, rx1 = rx2 - legW;
      TLegend legRight(rx1, ly1, rx2, ly2, "", "NDC");
      legRight.SetBorderSize(0);
      legRight.SetFillStyle(0);
      legRight.SetTextSize(0.03);
      legRight.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{rec} = %.4f mm", d2Mu), "");
      if (didD2Fit) legRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit} = %.3f mm", d2Sig), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{px} = %.1f #mum", 1000.0*(x_true - x_px)), "");
      legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec} = %.1f #mum", 1000.0*(x_true - d2Mu)), "");
      legRight.Draw();
    }

    // Print page
    c.cd();
    c.Print("2Dfits_diag_replay.pdf");
    nPages++;
  }

  c.Print("2Dfits_diag_replay.pdf]");

  file->Close();
  delete file;

  ::Info("processing2D_diag_replay", "Generated %lld pages (considered %lld events).", nPages, nConsidered);
  return 0;
}


// ROOT auto-exec wrappers so `root -l -b -q plotProcessing2DDiagReplay.C` works
int plotProcessing2DDiagReplay() {
  return processing2D_diag_replay();
}

int plotProcessing2DDiagReplay(const char* filename, double errorPercentOfMax) {
  return processing2D_diag_replay(filename, errorPercentOfMax);
}

int plotProcessing2DDiagReplay(Long64_t nRandomEvents) {
  return processing2D_diag_replay("../build/epicChargeSharing.root", 5.0, nRandomEvents);
}

int plotProcessing2DDiagReplay(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents) {
  return processing2D_diag_replay(filename, errorPercentOfMax, nRandomEvents);
}

int plotProcessing2DDiagReplay(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents, bool plotQiOverlay) {
  return processing2D_diag_replay(filename, errorPercentOfMax, nRandomEvents, plotQiOverlay);
}
