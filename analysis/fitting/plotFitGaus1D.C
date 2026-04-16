// plotFitGaus1D.C - Unified 1D Gaussian fitting and plotting
//
// Supports two modes:
//   - REPLAY MODE (default): Uses saved Gaussian fit parameters (Qf) from ROOT file.
//     Requires GaussRowA/Mu/Sigma/B and GaussColA/Mu/Sigma/B branches.
//   - FRESH MODE: Performs fitting at runtime (no saved params required).
//
// Default: Row/col replay with Qi overlay and Qi fitting enabled.
//
// Usage examples:
//   // Default: row/col replay with Qi fitting
//   root -l -b -q 'plotFitGaus1D.C+("myfile.root")'
//
//   // Row/col + diagonals with Qi fitting (replay)
//   root -l -b -q 'plotFitGaus1D.C+(true, true, true, true, "myfile.root")'
//
//   // Fresh fitting (no replay)
//   root -l -b -q 'plotFitGaus1D.C+(true, false, false, false, "myfile.root", 5.0, 100, false)'

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TList.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TAxis.h>
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
#include <memory>
#include <random>
#include <utility>
#include <limits>

// =============================================================================
// Configuration structure
// =============================================================================
struct FitGaus1DConfig {
  // What to plot
  bool plotRowCol = true;       // Plot row/column (central cross) fits
  bool plotDiagonals = false;   // Plot diagonal fits (main + secondary)

  // Fitting mode
  bool replayMode = true;       // true: use saved fit params; false: fresh fit

  // Qi fitting options (runtime fitting of Qi points for comparison)
  bool doQiFit = true;          // Refit Qi points and show green curves
  bool plotQiOverlay = true;    // Overlay Qi points on plots

  // Error handling for fits
  double errorPercentOfMax = 5.0;  // Vertical error as % of max charge

  // Event selection
  Long64_t nRandomEvents = 100;    // Number of events to plot

  // Output
  std::string outputPdf = "";      // Empty = auto-generate name

  // Input file
  std::string filename = "../build/epicChargeSharing.root";

  // Generate output filename based on settings
  std::string GetOutputPdf() const {
    if (!outputPdf.empty()) return outputPdf;
    std::string name = "1Dfits";
    if (plotRowCol && plotDiagonals) name += "_all";
    else if (plotDiagonals) name += "_diag";
    name += replayMode ? "_replay" : "_fresh";
    if (doQiFit) name += "_qifit";
    name += ".pdf";
    return name;
  }
};

// =============================================================================
// Common helper functions namespace
// =============================================================================
namespace FitGaus1DHelpers {

// 1D Gaussian with constant offset: A * exp(-0.5*((x-mu)/sigma)^2) + B
inline double GaussPlusB(double* x, double* p) {
  const double A     = p[0];
  const double mu    = p[1];
  const double sigma = p[2];
  const double B     = p[3];
  const double dx    = (x[0] - mu) / sigma;
  return A * std::exp(-0.5 * dx * dx) + B;
}

inline bool IsFinite(double v) { return std::isfinite(v); }

// Metadata helpers
inline double GetDoubleMetadata(TTree* tree, const char* key) {
  if (tree) {
    TList* info = tree->GetUserInfo();
    if (info) {
      if (auto* param = dynamic_cast<TParameter<double>*>(info->FindObject(key)))
        return param->GetVal();
    }
  }
  return NAN;
}

inline int GetIntMetadata(TTree* tree, const char* key) {
  if (tree) {
    TList* info = tree->GetUserInfo();
    if (info) {
      if (auto* param = dynamic_cast<TParameter<int>*>(info->FindObject(key)))
        return param->GetVal();
    }
  }
  return -1;
}

// Infer pixel spacing from tree data
inline double InferSpacingFromTree(TTree* t) {
  if (!t) return NAN;
  std::vector<double> xs, ys;
  xs.reserve(5000); ys.reserve(5000);
  double x_px_tmp = 0.0, y_px_tmp = 0.0;
  t->SetBranchAddress("PixelX", &x_px_tmp);
  t->SetBranchAddress("PixelY", &y_px_tmp);
  Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
  for (Long64_t i = 0; i < nToScan; ++i) {
    t->GetEntry(i);
    if (IsFinite(x_px_tmp)) xs.push_back(x_px_tmp);
    if (IsFinite(y_px_tmp)) ys.push_back(y_px_tmp);
  }
  auto computeGap = [](std::vector<double>& v) -> double {
    if (v.size() < 2) return NAN;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    if (v.size() < 2) return NAN;
    std::vector<double> gaps;
    for (size_t i = 1; i < v.size(); ++i) {
      double d = v[i] - v[i-1];
      if (d > 1e-9 && IsFinite(d)) gaps.push_back(d);
    }
    if (gaps.empty()) return NAN;
    std::nth_element(gaps.begin(), gaps.begin() + gaps.size()/2, gaps.end());
    return gaps[gaps.size()/2];
  };
  double gx = computeGap(xs), gy = computeGap(ys);
  if (IsFinite(gx) && gx > 0 && IsFinite(gy) && gy > 0) return 0.5 * (gx + gy);
  if (IsFinite(gx) && gx > 0) return gx;
  if (IsFinite(gy) && gy > 0) return gy;
  return NAN;
}

inline int InferRadiusFromTree(TTree* t) {
  if (!t) return -1;
  std::vector<double>* Qi_tmp = nullptr;
  if (t->GetBranch("Qi")) t->SetBranchAddress("Qi", &Qi_tmp);
  Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
  for (Long64_t i = 0; i < nToScan; ++i) {
    t->GetEntry(i);
    if (Qi_tmp && !Qi_tmp->empty()) {
      const size_t total = Qi_tmp->size();
      const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
      if (N >= 3 && N * N == static_cast<int>(total)) return (N - 1) / 2;
    }
  }
  return -1;
}

inline void SetupMinimizer() {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);
}

inline TFile* OpenFileWithFallback(const char* filename, const char* context) {
  TFile* file = TFile::Open(filename, "READ");
  if (!file || file->IsZombie()) {
    if (file) { file->Close(); delete file; file = nullptr; }
    TString fallback = "../../build/epicChargeSharing.root";
    file = TFile::Open(fallback, "READ");
    if (!file || file->IsZombie()) {
      ::Error(context, "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return nullptr;
    }
  }
  return file;
}

// 1D fit helper function (used only for Qi refitting at runtime)
inline bool Fit1DGaussian(const std::vector<double>& xs, const std::vector<double>& qs,
                          double pixelSpacing, double pixelSize, double x_px, double qmax,
                          int neighborhoodRadius, double uniformSigma,
                          double& outA, double& outMu, double& outSig, double& outB) {
  if (xs.size() < 3 || qs.size() < 3) return false;

  auto mm = std::minmax_element(qs.begin(), qs.end());
  double A0 = std::max(1e-18, *mm.second - *mm.first);
  double B0 = std::max(0.0, *mm.first);
  int idxMax = std::distance(qs.begin(), std::max_element(qs.begin(), qs.end()));
  double mu0 = xs[idxMax];

  const double sigLoBound = pixelSize;
  const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadius) * pixelSpacing);
  double sigInit = std::max(0.25 * pixelSpacing, sigLoBound);

  TF1 fLoc("fLoc", GaussPlusB, -1e9, 1e9, 4);
  fLoc.SetParameters(A0, mu0, sigInit, B0);
  ROOT::Math::WrappedMultiTF1 wModel(fLoc, 1);

  ROOT::Fit::BinData data(static_cast<int>(xs.size()), 1);
  for (size_t i = 0; i < xs.size(); ++i) {
    data.Add(xs[i], qs[i], uniformSigma > 0.0 ? uniformSigma : 1.0);
  }

  ROOT::Fit::Fitter fitter;
  fitter.Config().SetMinimizer("Minuit2", "Fumili2");
  fitter.Config().MinimizerOptions().SetStrategy(0);
  fitter.Config().MinimizerOptions().SetTolerance(1e-4);
  fitter.Config().MinimizerOptions().SetPrintLevel(0);
  fitter.SetFunction(wModel);

  const double AHi = std::max(1e-18, 2.0 * qmax);
  const double BHi = std::max(1e-18, qmax);
  const double muLo = x_px - 0.5 * pixelSpacing;
  const double muHi = x_px + 0.5 * pixelSpacing;

  fitter.Config().ParSettings(0).SetLimits(1e-18, AHi);
  fitter.Config().ParSettings(1).SetLimits(muLo, muHi);
  fitter.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
  fitter.Config().ParSettings(3).SetLimits(0.0, BHi);
  fitter.Config().ParSettings(0).SetValue(A0);
  fitter.Config().ParSettings(1).SetValue(mu0);
  fitter.Config().ParSettings(2).SetValue(sigInit);
  fitter.Config().ParSettings(3).SetValue(B0);

  bool ok = fitter.Fit(data);
  if (ok) {
    outA = fitter.Result().Parameter(0);
    outMu = fitter.Result().Parameter(1);
    outSig = fitter.Result().Parameter(2);
    outB = fitter.Result().Parameter(3);
    return true;
  }
  // Fallback: weighted centroid
  double wsum = 0.0, xw = 0.0;
  for (size_t i = 0; i < xs.size(); ++i) {
    double w = std::max(0.0, qs[i] - B0);
    wsum += w; xw += w * xs[i];
  }
  if (wsum > 0) {
    outA = A0; outB = B0; outSig = sigInit; outMu = xw / wsum;
    return true;
  }
  return false;
}

} // namespace FitGaus1DHelpers

using namespace FitGaus1DHelpers;

// =============================================================================
// Main unified plotting function (REPLAY MODE ONLY)
// =============================================================================
int plotFitGaus1DUnified(const FitGaus1DConfig& cfg) {
  SetupMinimizer();

  TFile* file = OpenFileWithFallback(cfg.filename.c_str(), "plotFitGaus1DUnified");
  if (!file) return 1;

  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("plotFitGaus1DUnified", "Hits tree not found");
    file->Close(); delete file;
    return 2;
  }

  // Get geometry parameters
  double pixelSpacing = GetDoubleMetadata(tree, "GridPixelSpacing_mm");
  double pixelSize = GetDoubleMetadata(tree, "GridPixelSize_mm");
  int neighborhoodRadius = GetIntMetadata(tree, "NeighborhoodRadius");

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0)
    pixelSpacing = InferSpacingFromTree(tree);
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("plotFitGaus1DUnified", "Pixel spacing unavailable");
    file->Close(); delete file;
    return 3;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0)
    pixelSize = 0.5 * pixelSpacing;
  if (neighborhoodRadius <= 0)
    neighborhoodRadius = InferRadiusFromTree(tree);
  if (neighborhoodRadius <= 0)
    neighborhoodRadius = 2;

  // Determine charge branch
  std::string chosenCharge;
  if (tree->GetBranch("Qf")) chosenCharge = "Qf";
  else if (tree->GetBranch("Fi")) chosenCharge = "Fi";
  else if (tree->GetBranch("Qi")) chosenCharge = "Qi";
  else {
    ::Error("plotFitGaus1DUnified", "No charge branch found");
    file->Close(); delete file;
    return 4;
  }

  // ==========================================================================
  // Check for required Gaussian fit branches (REPLAY MODE ONLY)
  // ==========================================================================
  bool haveRowColBranches = false;
  bool haveDiagBranches = false;

  if (cfg.replayMode) {
    if (cfg.plotRowCol) {
      if (!tree->GetBranch("GaussRowA") || !tree->GetBranch("GaussRowMu") ||
          !tree->GetBranch("GaussRowSigma") || !tree->GetBranch("GaussRowB")) {
        ::Error("plotFitGaus1DUnified",
                "Row fit branches not found (GaussRowA/Mu/Sigma/B). "
                "This file does not contain saved Gaussian fit parameters. "
                "Run the fitting step first or use fresh fitting mode (replayMode=false).");
        file->Close(); delete file;
        return 10;
      }
      if (!tree->GetBranch("GaussColA") || !tree->GetBranch("GaussColMu") ||
          !tree->GetBranch("GaussColSigma") || !tree->GetBranch("GaussColB")) {
        ::Error("plotFitGaus1DUnified",
                "Column fit branches not found (GaussColA/Mu/Sigma/B). "
                "This file does not contain saved Gaussian fit parameters. "
                "Run the fitting step first or use fresh fitting mode (replayMode=false).");
        file->Close(); delete file;
        return 11;
      }
      haveRowColBranches = true;
    }

    if (cfg.plotDiagonals) {
      if (!tree->GetBranch("GaussMDiagA") || !tree->GetBranch("GaussMDiagMu") ||
          !tree->GetBranch("GaussMDiagSigma") || !tree->GetBranch("GaussMDiagB")) {
        ::Error("plotFitGaus1DUnified",
                "Main diagonal fit branches not found (GaussMDiagA/Mu/Sigma/B). "
                "This file does not contain saved diagonal Gaussian fit parameters. "
                "Run the fitting step with diagonal fitting enabled or use fresh fitting mode.");
        file->Close(); delete file;
        return 12;
      }
      if (!tree->GetBranch("GaussSDiagA") || !tree->GetBranch("GaussSDiagMu") ||
          !tree->GetBranch("GaussSDiagSigma") || !tree->GetBranch("GaussSDiagB")) {
        ::Error("plotFitGaus1DUnified",
                "Secondary diagonal fit branches not found (GaussSDiagA/Mu/Sigma/B). "
                "This file does not contain saved diagonal Gaussian fit parameters. "
                "Run the fitting step with diagonal fitting enabled or use fresh fitting mode.");
        file->Close(); delete file;
        return 13;
      }
      haveDiagBranches = true;
    }
  }

  // Setup branches
  double x_true = 0.0, y_true = 0.0, x_px = 0.0, y_px = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Q = nullptr;
  std::vector<double>* QiVec = nullptr;

  // Saved fit parameters (required for replay)
  double rowA = NAN, rowMu = NAN, rowSig = NAN, rowB = NAN;
  double colA = NAN, colMu = NAN, colSig = NAN, colB = NAN;
  double d1A = NAN, d1Mu = NAN, d1Sig = NAN, d1B = NAN;
  double d2A = NAN, d2Mu = NAN, d2Sig = NAN, d2B = NAN;

  tree->SetBranchAddress("TrueX", &x_true);
  tree->SetBranchAddress("TrueY", &y_true);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_true);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);

  const bool haveQiBranch = tree->GetBranch("Qi") != nullptr;
  if (haveQiBranch) tree->SetBranchAddress("Qi", &QiVec);

  // Row/col saved params (only in replay mode)
  if (cfg.replayMode && cfg.plotRowCol) {
    tree->SetBranchAddress("GaussRowA", &rowA);
    tree->SetBranchAddress("GaussRowMu", &rowMu);
    tree->SetBranchAddress("GaussRowSigma", &rowSig);
    tree->SetBranchAddress("GaussRowB", &rowB);
    tree->SetBranchAddress("GaussColA", &colA);
    tree->SetBranchAddress("GaussColMu", &colMu);
    tree->SetBranchAddress("GaussColSigma", &colSig);
    tree->SetBranchAddress("GaussColB", &colB);
  }

  // Diagonal saved params (only in replay mode)
  if (cfg.replayMode && cfg.plotDiagonals) {
    tree->SetBranchAddress("GaussMDiagA", &d1A);
    tree->SetBranchAddress("GaussMDiagMu", &d1Mu);
    tree->SetBranchAddress("GaussMDiagSigma", &d1Sig);
    tree->SetBranchAddress("GaussMDiagB", &d1B);
    tree->SetBranchAddress("GaussSDiagA", &d2A);
    tree->SetBranchAddress("GaussSDiagMu", &d2Mu);
    tree->SetBranchAddress("GaussSDiagSigma", &d2Sig);
    tree->SetBranchAddress("GaussSDiagB", &d2B);
  }

  // Setup canvas
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);

  TCanvas c("c_unified", "1D Gaussian Fits", 1800, 700);
  TPad pL("pL", "left", 0.0, 0.0, 0.5, 1.0);
  TPad pR("pR", "right", 0.5, 0.0, 1.0, 1.0);
  pL.SetTicks(1, 1); pR.SetTicks(1, 1);
  pL.Draw(); pR.Draw();

  std::string outputPdf = cfg.GetOutputPdf();
  c.Print((outputPdf + "[").c_str());

  // Fit functions
  TF1 fLeft("fLeft", GaussPlusB, -1e9, 1e9, 4);
  TF1 fRight("fRight", GaussPlusB, -1e9, 1e9, 4);
  TF1 fLeftQi("fLeftQi", GaussPlusB, -1e9, 1e9, 4);
  TF1 fRightQi("fRightQi", GaussPlusB, -1e9, 1e9, 4);

  // Event loop setup
  const Long64_t nEntries = tree->GetEntries();
  std::vector<Long64_t> indices;
  indices.reserve(nEntries);
  for (Long64_t i = 0; i < nEntries; ++i) indices.push_back(i);
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::shuffle(indices.begin(), indices.end(), rng);

  Long64_t nPages = 0, nConsidered = 0;
  const Long64_t targetPages = (cfg.nRandomEvents < 0) ? 0 : cfg.nRandomEvents;

  for (Long64_t sampleIdx = 0; sampleIdx < nEntries && nPages < targetPages; ++sampleIdx) {
    const Long64_t eventIndex = indices[sampleIdx];
    tree->GetEntry(eventIndex);

    if (is_pixel_true || !Q || Q->empty()) continue;

    const size_t total = Q->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Build data arrays
    std::vector<double> x_row, q_row, y_col, q_col;
    std::vector<double> s_d1, q_d1, s_d2, q_d2;
    std::vector<int> rowIdx, colIdx, d1Idx, d2Idx;
    double qmaxNeighborhood = -1e300;

    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double q = (*Q)[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;

        // Row (central horizontal)
        if (dj == 0 && cfg.plotRowCol) {
          x_row.push_back(x_px + di * pixelSpacing);
          q_row.push_back(q);
          rowIdx.push_back(idx);
        }
        // Column (central vertical)
        if (di == 0 && cfg.plotRowCol) {
          y_col.push_back(y_px + dj * pixelSpacing);
          q_col.push_back(q);
          colIdx.push_back(idx);
        }
      }
      // Main diagonal (di = dj)
      if (cfg.plotDiagonals) {
        int idx1 = (di + R) * N + (di + R);
        if (idx1 >= 0 && idx1 < (int)total) {
          double q1 = (*Q)[idx1];
          if (IsFinite(q1) && q1 >= 0) {
            s_d1.push_back(x_px + di * pixelSpacing);
            q_d1.push_back(q1);
            d1Idx.push_back(idx1);
          }
        }
        // Secondary diagonal (di = -dj)
        int idx2 = (di + R) * N + (-di + R);
        if (idx2 >= 0 && idx2 < (int)total) {
          double q2 = (*Q)[idx2];
          if (IsFinite(q2) && q2 >= 0) {
            s_d2.push_back(x_px + di * pixelSpacing);
            q_d2.push_back(q2);
            d2Idx.push_back(idx2);
          }
        }
      }
    }

    // Check minimum points
    bool hasRowCol = cfg.plotRowCol && x_row.size() >= 3 && y_col.size() >= 3;
    bool hasDiag = cfg.plotDiagonals && s_d1.size() >= 3 && s_d2.size() >= 3;

    if (!hasRowCol && !hasDiag) continue;

    const double relErr = std::max(0.0, cfg.errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0) ? relErr * qmaxNeighborhood : 0.0;
    bool useWeights = (uniformSigma > 0.0);

    // Decide what to plot this page
    bool doRowColThisPage = hasRowCol;
    bool doDiagThisPage = hasDiag && !hasRowCol;
    if (hasRowCol && hasDiag) {
      doDiagThisPage = false;
    }

    // =========================================================================
    // ROW/COLUMN PLOTTING
    // =========================================================================
    if (doRowColThisPage) {
      // In replay mode, verify saved params are valid
      if (cfg.replayMode && (!IsFinite(rowMu) || !IsFinite(colMu))) continue;

      nConsidered++;

      // Build graphs
      TGraphErrors gRowErr, gColErr;
      TGraph gRowPlain, gColPlain;
      TGraph* baseRowPtr = nullptr;
      TGraph* baseColPtr = nullptr;

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

      baseRowPtr->SetTitle(Form("Event %lld: Row; x [mm]; %s [C]", eventIndex, chosenCharge.c_str()));
      baseRowPtr->SetMarkerStyle(20); baseRowPtr->SetMarkerSize(0.9); baseRowPtr->SetLineColor(kBlue+1);
      baseColPtr->SetTitle(Form("Event %lld: Column; y [mm]; %s [C]", eventIndex, chosenCharge.c_str()));
      baseColPtr->SetMarkerStyle(21); baseColPtr->SetMarkerSize(0.9); baseColPtr->SetLineColor(kBlue+2);

      // Get fit parameters (replay from saved or fresh fit)
      double fitRowA, fitRowMu, fitRowSig, fitRowB;
      double fitColA, fitColMu, fitColSig, fitColB;
      bool didRowFit = false, didColFit = false;

      if (cfg.replayMode) {
        // Use saved fit parameters
        fitRowA = rowA; fitRowMu = rowMu; fitRowSig = rowSig; fitRowB = rowB;
        fitColA = colA; fitColMu = colMu; fitColSig = colSig; fitColB = colB;
        didRowFit = IsFinite(rowA) && IsFinite(rowMu) && IsFinite(rowSig) && IsFinite(rowB) && rowSig > 0;
        didColFit = IsFinite(colA) && IsFinite(colMu) && IsFinite(colSig) && IsFinite(colB) && colSig > 0;
      } else {
        // Fresh fitting
        didRowFit = Fit1DGaussian(x_row, q_row, pixelSpacing, pixelSize, x_px, qmaxNeighborhood,
                                   neighborhoodRadius, uniformSigma, fitRowA, fitRowMu, fitRowSig, fitRowB);
        didColFit = Fit1DGaussian(y_col, q_col, pixelSpacing, pixelSize, y_px, qmaxNeighborhood,
                                   neighborhoodRadius, uniformSigma, fitColA, fitColMu, fitColSig, fitColB);
      }

      if (!didRowFit || !didColFit || !IsFinite(fitRowMu) || !IsFinite(fitColMu)) continue;

      // Qi fitting (runtime fitting for comparison)
      bool didRowFitQi = false, didColFitQi = false;
      double fitRowAQi, fitRowMuQi, fitRowSigQi, fitRowBQi;
      double fitColAQi, fitColMuQi, fitColSigQi, fitColBQi;
      double qmaxQi = -1e300;

      if (cfg.doQiFit && haveQiBranch && QiVec && !QiVec->empty()) {
        // Build Qi arrays
        std::vector<double> x_row_qi, q_row_qi, y_col_qi, q_col_qi;
        for (size_t k = 0; k < rowIdx.size(); ++k) {
          int idx = rowIdx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) {
              x_row_qi.push_back(x_row[k]);
              q_row_qi.push_back(qqi);
              if (qqi > qmaxQi) qmaxQi = qqi;
            }
          }
        }
        for (size_t k = 0; k < colIdx.size(); ++k) {
          int idx = colIdx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) {
              y_col_qi.push_back(y_col[k]);
              q_col_qi.push_back(qqi);
              if (qqi > qmaxQi) qmaxQi = qqi;
            }
          }
        }

        double uniformSigmaQi = (qmaxQi > 0 && relErr > 0.0) ? relErr * qmaxQi : 0.0;

        if (x_row_qi.size() >= 3) {
          didRowFitQi = Fit1DGaussian(x_row_qi, q_row_qi, pixelSpacing, pixelSize, x_px, qmaxQi,
                                       neighborhoodRadius, uniformSigmaQi, fitRowAQi, fitRowMuQi, fitRowSigQi, fitRowBQi);
        }
        if (y_col_qi.size() >= 3) {
          didColFitQi = Fit1DGaussian(y_col_qi, q_col_qi, pixelSpacing, pixelSize, y_px, qmaxQi,
                                       neighborhoodRadius, uniformSigmaQi, fitColAQi, fitColMuQi, fitColSigQi, fitColBQi);
        }
      }

      // Y-axis limits
      auto mmRow = std::minmax_element(q_row.begin(), q_row.end());
      auto mmCol = std::minmax_element(q_col.begin(), q_col.end());
      double yMinRow = *mmRow.first, yMaxRow = *mmRow.second;
      double yMinCol = *mmCol.first, yMaxCol = *mmCol.second;
      if (didRowFit) { yMaxRow = std::max(yMaxRow, fitRowA + fitRowB); yMinRow = std::min(yMinRow, fitRowB); }
      if (didColFit) { yMaxCol = std::max(yMaxCol, fitColA + fitColB); yMinCol = std::min(yMinCol, fitColB); }
      if (didRowFitQi) { yMaxRow = std::max(yMaxRow, fitRowAQi + fitRowBQi); }
      if (didColFitQi) { yMaxCol = std::max(yMaxCol, fitColAQi + fitColBQi); }
      double padRow = 0.10 * (yMaxRow - yMinRow);
      double padCol = 0.10 * (yMaxCol - yMinCol);
      baseRowPtr->SetMinimum(yMinRow - padRow); baseRowPtr->SetMaximum(yMaxRow + padRow);
      baseColPtr->SetMinimum(yMinCol - padCol); baseColPtr->SetMaximum(yMaxCol + padCol);

      // Draw Column (left pad)
      pL.cd();
      baseColPtr->Draw(useWeights ? "AP" : "AP");
      auto mmY = std::minmax_element(y_col.begin(), y_col.end());
      baseColPtr->GetXaxis()->SetLimits(*mmY.first - 0.5*pixelSpacing, *mmY.second + 0.5*pixelSpacing);

      if (didColFit) {
        fLeft.SetRange(*mmY.first - 0.5*pixelSpacing, *mmY.second + 0.5*pixelSpacing);
        fLeft.SetParameters(fitColA, fitColMu, fitColSig, fitColB);
        fLeft.SetNpx(600); fLeft.SetLineWidth(2); fLeft.SetLineColor(kRed+1);
        fLeft.Draw("L SAME");
      }
      if (didColFitQi) {
        fLeftQi.SetRange(*mmY.first - 0.5*pixelSpacing, *mmY.second + 0.5*pixelSpacing);
        fLeftQi.SetParameters(fitColAQi, fitColMuQi, fitColSigQi, fitColBQi);
        fLeftQi.SetNpx(600); fLeftQi.SetLineWidth(2); fLeftQi.SetLineStyle(2); fLeftQi.SetLineColor(kGreen+2);
        fLeftQi.Draw("L SAME");
      }

      // Qi overlay points
      TGraph gColQi;
      bool drewColQi = false;
      if (cfg.plotQiOverlay && haveQiBranch && QiVec && chosenCharge != "Qi") {
        std::vector<std::pair<double,double>> pts;
        for (size_t k = 0; k < colIdx.size(); ++k) {
          int idx = colIdx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) pts.emplace_back(y_col[k], qqi);
          }
        }
        if (!pts.empty()) {
          gColQi = TGraph(static_cast<int>(pts.size()));
          for (int k = 0; k < gColQi.GetN(); ++k) gColQi.SetPoint(k, pts[k].first, pts[k].second);
          gColQi.SetMarkerStyle(24); gColQi.SetMarkerSize(1.1);
          gColQi.SetMarkerColor(kGreen+2); gColQi.SetLineColor(kGreen+2);
          gColQi.Draw("P SAME");
          drewColQi = true;
        }
      }

      gPad->Update();
      double yPadMinC = gPad->GetUymin(), yPadMaxC = gPad->GetUymax();
      TLine lineYtrue(y_true, yPadMinC, y_true, yPadMaxC);
      lineYtrue.SetLineStyle(1); lineYtrue.SetLineWidth(3); lineYtrue.SetLineColor(kBlack);
      lineYtrue.Draw("SAME");
      TLine lineYrec(fitColMu, yPadMinC, fitColMu, yPadMaxC);
      lineYrec.SetLineStyle(2); lineYrec.SetLineWidth(2); lineYrec.SetLineColor(kRed+1);
      lineYrec.Draw("SAME");
      TLine lineYrecQi;
      if (didColFitQi && IsFinite(fitColMuQi)) {
        lineYrecQi = TLine(fitColMuQi, yPadMinC, fitColMuQi, yPadMaxC);
        lineYrecQi.SetLineStyle(2); lineYrecQi.SetLineWidth(2); lineYrecQi.SetLineColor(kGreen+2);
        lineYrecQi.Draw("SAME");
      }

      TLegend legCL(0.12, 0.62, 0.40, 0.88, "", "NDC");
      legCL.SetBorderSize(0); legCL.SetFillStyle(0); legCL.SetTextSize(0.028);
      legCL.AddEntry(&lineYtrue, "y_{true}", "l");
      legCL.AddEntry(&lineYrec, Form("y_{rec}(%s)", chosenCharge.c_str()), "l");
      if (didColFitQi) legCL.AddEntry(&lineYrecQi, "y_{rec}(Q_{i})", "l");
      if (drewColQi) legCL.AddEntry(&gColQi, "Q_{i} points", "p");
      legCL.Draw();

      TLegend legCR(0.58, 0.62, 0.88, 0.88, "", "NDC");
      legCR.SetBorderSize(0); legCR.SetFillStyle(0); legCR.SetTextSize(0.028);
      legCR.AddEntry((TObject*)nullptr, Form("y_{true} = %.4f mm", y_true), "");
      legCR.AddEntry((TObject*)nullptr, Form("y_{rec} = %.4f mm", fitColMu), "");
      legCR.AddEntry((TObject*)nullptr, Form("#Delta = %.1f #mum", 1000.0*(y_true - fitColMu)), "");
      if (didColFitQi) {
        legCR.AddEntry((TObject*)nullptr, Form("y_{rec}(Q_{i}) = %.4f mm", fitColMuQi), "");
        legCR.AddEntry((TObject*)nullptr, Form("#Delta(Q_{i}) = %.1f #mum", 1000.0*(y_true - fitColMuQi)), "");
      }
      legCR.Draw();

      // Draw Row (right pad)
      pR.cd();
      baseRowPtr->Draw(useWeights ? "AP" : "AP");
      auto mmX = std::minmax_element(x_row.begin(), x_row.end());
      baseRowPtr->GetXaxis()->SetLimits(*mmX.first - 0.5*pixelSpacing, *mmX.second + 0.5*pixelSpacing);

      if (didRowFit) {
        fRight.SetRange(*mmX.first - 0.5*pixelSpacing, *mmX.second + 0.5*pixelSpacing);
        fRight.SetParameters(fitRowA, fitRowMu, fitRowSig, fitRowB);
        fRight.SetNpx(600); fRight.SetLineWidth(2); fRight.SetLineColor(kRed+1);
        fRight.Draw("L SAME");
      }
      if (didRowFitQi) {
        fRightQi.SetRange(*mmX.first - 0.5*pixelSpacing, *mmX.second + 0.5*pixelSpacing);
        fRightQi.SetParameters(fitRowAQi, fitRowMuQi, fitRowSigQi, fitRowBQi);
        fRightQi.SetNpx(600); fRightQi.SetLineWidth(2); fRightQi.SetLineStyle(2); fRightQi.SetLineColor(kGreen+2);
        fRightQi.Draw("L SAME");
      }

      // Qi overlay points
      TGraph gRowQi;
      bool drewRowQi = false;
      if (cfg.plotQiOverlay && haveQiBranch && QiVec && chosenCharge != "Qi") {
        std::vector<std::pair<double,double>> pts;
        for (size_t k = 0; k < rowIdx.size(); ++k) {
          int idx = rowIdx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) pts.emplace_back(x_row[k], qqi);
          }
        }
        if (!pts.empty()) {
          gRowQi = TGraph(static_cast<int>(pts.size()));
          for (int k = 0; k < gRowQi.GetN(); ++k) gRowQi.SetPoint(k, pts[k].first, pts[k].second);
          gRowQi.SetMarkerStyle(24); gRowQi.SetMarkerSize(1.1);
          gRowQi.SetMarkerColor(kGreen+2); gRowQi.SetLineColor(kGreen+2);
          gRowQi.Draw("P SAME");
          drewRowQi = true;
        }
      }

      gPad->Update();
      double yPadMin = gPad->GetUymin(), yPadMax = gPad->GetUymax();
      TLine lineXtrue(x_true, yPadMin, x_true, yPadMax);
      lineXtrue.SetLineStyle(1); lineXtrue.SetLineWidth(3); lineXtrue.SetLineColor(kBlack);
      lineXtrue.Draw("SAME");
      TLine lineXrec(fitRowMu, yPadMin, fitRowMu, yPadMax);
      lineXrec.SetLineStyle(2); lineXrec.SetLineWidth(2); lineXrec.SetLineColor(kRed+1);
      lineXrec.Draw("SAME");
      TLine lineXrecQi;
      if (didRowFitQi && IsFinite(fitRowMuQi)) {
        lineXrecQi = TLine(fitRowMuQi, yPadMin, fitRowMuQi, yPadMax);
        lineXrecQi.SetLineStyle(2); lineXrecQi.SetLineWidth(2); lineXrecQi.SetLineColor(kGreen+2);
        lineXrecQi.Draw("SAME");
      }

      TLegend legRL(0.12, 0.62, 0.40, 0.88, "", "NDC");
      legRL.SetBorderSize(0); legRL.SetFillStyle(0); legRL.SetTextSize(0.028);
      legRL.AddEntry(&lineXtrue, "x_{true}", "l");
      legRL.AddEntry(&lineXrec, Form("x_{rec}(%s)", chosenCharge.c_str()), "l");
      if (didRowFitQi) legRL.AddEntry(&lineXrecQi, "x_{rec}(Q_{i})", "l");
      if (drewRowQi) legRL.AddEntry(&gRowQi, "Q_{i} points", "p");
      legRL.Draw();

      TLegend legRR(0.58, 0.62, 0.88, 0.88, "", "NDC");
      legRR.SetBorderSize(0); legRR.SetFillStyle(0); legRR.SetTextSize(0.028);
      legRR.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
      legRR.AddEntry((TObject*)nullptr, Form("x_{rec} = %.4f mm", fitRowMu), "");
      legRR.AddEntry((TObject*)nullptr, Form("#Delta = %.1f #mum", 1000.0*(x_true - fitRowMu)), "");
      if (didRowFitQi) {
        legRR.AddEntry((TObject*)nullptr, Form("x_{rec}(Q_{i}) = %.4f mm", fitRowMuQi), "");
        legRR.AddEntry((TObject*)nullptr, Form("#Delta(Q_{i}) = %.1f #mum", 1000.0*(x_true - fitRowMuQi)), "");
      }
      legRR.Draw();

      c.cd();
      c.Print(outputPdf.c_str());
      nPages++;
    }

    // =========================================================================
    // DIAGONAL PLOTTING
    // =========================================================================
    if (doDiagThisPage) {
      // In replay mode, verify saved params are valid
      if (cfg.replayMode && (!IsFinite(d1Mu) || !IsFinite(d2Mu))) continue;

      nConsidered++;

      TGraphErrors gD1Err, gD2Err;
      TGraph gD1Plain, gD2Plain;
      TGraph* baseD1Ptr = nullptr;
      TGraph* baseD2Ptr = nullptr;

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

      baseD1Ptr->SetTitle(Form("Event %lld: Main diag; s=x [mm]; %s [C]", eventIndex, chosenCharge.c_str()));
      baseD1Ptr->SetMarkerStyle(20); baseD1Ptr->SetMarkerSize(0.9); baseD1Ptr->SetLineColor(kBlue+1);
      baseD2Ptr->SetTitle(Form("Event %lld: Sec diag; s=x [mm]; %s [C]", eventIndex, chosenCharge.c_str()));
      baseD2Ptr->SetMarkerStyle(21); baseD2Ptr->SetMarkerSize(0.9); baseD2Ptr->SetLineColor(kBlue+2);

      // Get fit parameters (replay from saved or fresh fit)
      double fitD1A, fitD1Mu, fitD1Sig, fitD1B;
      double fitD2A, fitD2Mu, fitD2Sig, fitD2B;
      bool didD1Fit = false, didD2Fit = false;

      if (cfg.replayMode) {
        // Use saved fit parameters
        fitD1A = d1A; fitD1Mu = d1Mu; fitD1Sig = d1Sig; fitD1B = d1B;
        fitD2A = d2A; fitD2Mu = d2Mu; fitD2Sig = d2Sig; fitD2B = d2B;
        didD1Fit = IsFinite(d1A) && IsFinite(d1Mu) && IsFinite(d1Sig) && IsFinite(d1B) && d1Sig > 0;
        didD2Fit = IsFinite(d2A) && IsFinite(d2Mu) && IsFinite(d2Sig) && IsFinite(d2B) && d2Sig > 0;
      } else {
        // Fresh fitting
        didD1Fit = Fit1DGaussian(s_d1, q_d1, pixelSpacing, pixelSize, x_px, qmaxNeighborhood,
                                  neighborhoodRadius, uniformSigma, fitD1A, fitD1Mu, fitD1Sig, fitD1B);
        didD2Fit = Fit1DGaussian(s_d2, q_d2, pixelSpacing, pixelSize, x_px, qmaxNeighborhood,
                                  neighborhoodRadius, uniformSigma, fitD2A, fitD2Mu, fitD2Sig, fitD2B);
      }

      if (!didD1Fit || !didD2Fit || !IsFinite(fitD1Mu) || !IsFinite(fitD2Mu)) continue;

      // Qi fitting for diagonals (runtime fitting for comparison)
      bool didD1FitQi = false, didD2FitQi = false;
      double fitD1AQi, fitD1MuQi, fitD1SigQi, fitD1BQi;
      double fitD2AQi, fitD2MuQi, fitD2SigQi, fitD2BQi;
      double qmaxQiDiag = -1e300;

      if (cfg.doQiFit && haveQiBranch && QiVec && !QiVec->empty()) {
        std::vector<double> s_d1_qi, q_d1_qi, s_d2_qi, q_d2_qi;
        for (size_t k = 0; k < d1Idx.size(); ++k) {
          int idx = d1Idx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) {
              s_d1_qi.push_back(s_d1[k]);
              q_d1_qi.push_back(qqi);
              if (qqi > qmaxQiDiag) qmaxQiDiag = qqi;
            }
          }
        }
        for (size_t k = 0; k < d2Idx.size(); ++k) {
          int idx = d2Idx[k];
          if (idx >= 0 && idx < (int)QiVec->size()) {
            double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0) {
              s_d2_qi.push_back(s_d2[k]);
              q_d2_qi.push_back(qqi);
              if (qqi > qmaxQiDiag) qmaxQiDiag = qqi;
            }
          }
        }

        double uniformSigmaQi = (qmaxQiDiag > 0 && relErr > 0.0) ? relErr * qmaxQiDiag : 0.0;

        if (s_d1_qi.size() >= 3) {
          didD1FitQi = Fit1DGaussian(s_d1_qi, q_d1_qi, pixelSpacing, pixelSize, x_px, qmaxQiDiag,
                                      neighborhoodRadius, uniformSigmaQi, fitD1AQi, fitD1MuQi, fitD1SigQi, fitD1BQi);
        }
        if (s_d2_qi.size() >= 3) {
          didD2FitQi = Fit1DGaussian(s_d2_qi, q_d2_qi, pixelSpacing, pixelSize, x_px, qmaxQiDiag,
                                      neighborhoodRadius, uniformSigmaQi, fitD2AQi, fitD2MuQi, fitD2SigQi, fitD2BQi);
        }
      }

      // Y-axis limits
      auto mmD1 = std::minmax_element(q_d1.begin(), q_d1.end());
      auto mmD2 = std::minmax_element(q_d2.begin(), q_d2.end());
      double yMinD1 = *mmD1.first, yMaxD1 = *mmD1.second;
      double yMinD2 = *mmD2.first, yMaxD2 = *mmD2.second;
      if (didD1Fit) { yMaxD1 = std::max(yMaxD1, fitD1A + fitD1B); yMinD1 = std::min(yMinD1, fitD1B); }
      if (didD2Fit) { yMaxD2 = std::max(yMaxD2, fitD2A + fitD2B); yMinD2 = std::min(yMinD2, fitD2B); }
      double padD1 = 0.10 * (yMaxD1 - yMinD1);
      double padD2 = 0.10 * (yMaxD2 - yMinD2);
      baseD1Ptr->SetMinimum(yMinD1 - padD1); baseD1Ptr->SetMaximum(yMaxD1 + padD1);
      baseD2Ptr->SetMinimum(yMinD2 - padD2); baseD2Ptr->SetMaximum(yMaxD2 + padD2);

      // Draw main diagonal (left)
      pL.cd();
      baseD1Ptr->Draw(useWeights ? "AP" : "AP");
      auto mmS1 = std::minmax_element(s_d1.begin(), s_d1.end());
      baseD1Ptr->GetXaxis()->SetLimits(*mmS1.first - 0.5*pixelSpacing, *mmS1.second + 0.5*pixelSpacing);

      if (didD1Fit) {
        fLeft.SetRange(*mmS1.first - 0.5*pixelSpacing, *mmS1.second + 0.5*pixelSpacing);
        fLeft.SetParameters(fitD1A, fitD1Mu, fitD1Sig, fitD1B);
        fLeft.SetNpx(600); fLeft.SetLineWidth(2); fLeft.SetLineColor(kRed+1);
        fLeft.Draw("L SAME");
      }
      if (didD1FitQi) {
        fLeftQi.SetRange(*mmS1.first - 0.5*pixelSpacing, *mmS1.second + 0.5*pixelSpacing);
        fLeftQi.SetParameters(fitD1AQi, fitD1MuQi, fitD1SigQi, fitD1BQi);
        fLeftQi.SetNpx(600); fLeftQi.SetLineWidth(2); fLeftQi.SetLineStyle(2); fLeftQi.SetLineColor(kGreen+2);
        fLeftQi.Draw("L SAME");
      }

      gPad->Update();
      double yPadMin1 = gPad->GetUymin(), yPadMax1 = gPad->GetUymax();
      TLine lineStrue1(x_true, yPadMin1, x_true, yPadMax1);
      lineStrue1.SetLineStyle(1); lineStrue1.SetLineWidth(3); lineStrue1.SetLineColor(kBlack);
      lineStrue1.Draw("SAME");
      TLine lineSrec1(fitD1Mu, yPadMin1, fitD1Mu, yPadMax1);
      lineSrec1.SetLineStyle(2); lineSrec1.SetLineWidth(2); lineSrec1.SetLineColor(kRed+1);
      lineSrec1.Draw("SAME");
      TLine lineSrecQi1;
      if (didD1FitQi && IsFinite(fitD1MuQi)) {
        lineSrecQi1 = TLine(fitD1MuQi, yPadMin1, fitD1MuQi, yPadMax1);
        lineSrecQi1.SetLineStyle(2); lineSrecQi1.SetLineWidth(2); lineSrecQi1.SetLineColor(kGreen+2);
        lineSrecQi1.Draw("SAME");
      }

      TLegend legD1L(0.12, 0.65, 0.40, 0.88, "", "NDC");
      legD1L.SetBorderSize(0); legD1L.SetFillStyle(0); legD1L.SetTextSize(0.028);
      legD1L.AddEntry(&lineStrue1, "x_{true}", "l");
      legD1L.AddEntry(&lineSrec1, "x_{rec}", "l");
      if (didD1FitQi) legD1L.AddEntry(&lineSrecQi1, "x_{rec}(Q_{i})", "l");
      legD1L.Draw();

      TLegend legD1R(0.58, 0.65, 0.88, 0.88, "", "NDC");
      legD1R.SetBorderSize(0); legD1R.SetFillStyle(0); legD1R.SetTextSize(0.028);
      legD1R.AddEntry((TObject*)nullptr, Form("#Delta = %.1f #mum", 1000.0*(x_true - fitD1Mu)), "");
      if (didD1FitQi) legD1R.AddEntry((TObject*)nullptr, Form("#Delta(Q_{i}) = %.1f #mum", 1000.0*(x_true - fitD1MuQi)), "");
      legD1R.Draw();

      // Draw secondary diagonal (right)
      pR.cd();
      baseD2Ptr->Draw(useWeights ? "AP" : "AP");
      auto mmS2 = std::minmax_element(s_d2.begin(), s_d2.end());
      baseD2Ptr->GetXaxis()->SetLimits(*mmS2.first - 0.5*pixelSpacing, *mmS2.second + 0.5*pixelSpacing);

      if (didD2Fit) {
        fRight.SetRange(*mmS2.first - 0.5*pixelSpacing, *mmS2.second + 0.5*pixelSpacing);
        fRight.SetParameters(fitD2A, fitD2Mu, fitD2Sig, fitD2B);
        fRight.SetNpx(600); fRight.SetLineWidth(2); fRight.SetLineColor(kRed+1);
        fRight.Draw("L SAME");
      }
      if (didD2FitQi) {
        fRightQi.SetRange(*mmS2.first - 0.5*pixelSpacing, *mmS2.second + 0.5*pixelSpacing);
        fRightQi.SetParameters(fitD2AQi, fitD2MuQi, fitD2SigQi, fitD2BQi);
        fRightQi.SetNpx(600); fRightQi.SetLineWidth(2); fRightQi.SetLineStyle(2); fRightQi.SetLineColor(kGreen+2);
        fRightQi.Draw("L SAME");
      }

      gPad->Update();
      double yPadMin2 = gPad->GetUymin(), yPadMax2 = gPad->GetUymax();
      TLine lineStrue2(x_true, yPadMin2, x_true, yPadMax2);
      lineStrue2.SetLineStyle(1); lineStrue2.SetLineWidth(3); lineStrue2.SetLineColor(kBlack);
      lineStrue2.Draw("SAME");
      TLine lineSrec2(fitD2Mu, yPadMin2, fitD2Mu, yPadMax2);
      lineSrec2.SetLineStyle(2); lineSrec2.SetLineWidth(2); lineSrec2.SetLineColor(kRed+1);
      lineSrec2.Draw("SAME");
      TLine lineSrecQi2;
      if (didD2FitQi && IsFinite(fitD2MuQi)) {
        lineSrecQi2 = TLine(fitD2MuQi, yPadMin2, fitD2MuQi, yPadMax2);
        lineSrecQi2.SetLineStyle(2); lineSrecQi2.SetLineWidth(2); lineSrecQi2.SetLineColor(kGreen+2);
        lineSrecQi2.Draw("SAME");
      }

      TLegend legD2L(0.12, 0.65, 0.40, 0.88, "", "NDC");
      legD2L.SetBorderSize(0); legD2L.SetFillStyle(0); legD2L.SetTextSize(0.028);
      legD2L.AddEntry(&lineStrue2, "x_{true}", "l");
      legD2L.AddEntry(&lineSrec2, "x_{rec}", "l");
      if (didD2FitQi) legD2L.AddEntry(&lineSrecQi2, "x_{rec}(Q_{i})", "l");
      legD2L.Draw();

      TLegend legD2R(0.58, 0.65, 0.88, 0.88, "", "NDC");
      legD2R.SetBorderSize(0); legD2R.SetFillStyle(0); legD2R.SetTextSize(0.028);
      legD2R.AddEntry((TObject*)nullptr, Form("#Delta = %.1f #mum", 1000.0*(x_true - fitD2Mu)), "");
      if (didD2FitQi) legD2R.AddEntry((TObject*)nullptr, Form("#Delta(Q_{i}) = %.1f #mum", 1000.0*(x_true - fitD2MuQi)), "");
      legD2R.Draw();

      c.cd();
      c.Print(outputPdf.c_str());
      nPages++;
    }
  }

  c.Print((outputPdf + "]").c_str());
  file->Close(); delete file;

  ::Info("plotFitGaus1DUnified", "Generated %lld pages (considered %lld events) -> %s",
         nPages, nConsidered, outputPdf.c_str());
  return 0;
}

// =============================================================================
// Convenience wrapper functions
// =============================================================================

// Full control version
int plotFitGaus1D(bool plotRowCol = true,
                     bool plotDiagonals = false,
                     bool doQiFit = true,
                     bool plotQiOverlay = true,
                     const char* filename = "../build/epicChargeSharing.root",
                     double errorPercentOfMax = 5.0,
                     Long64_t nRandomEvents = 100,
                     bool replayMode = true,
                     const char* outputPdf = "") {
  FitGaus1DConfig cfg;
  cfg.plotRowCol = plotRowCol;
  cfg.plotDiagonals = plotDiagonals;
  cfg.doQiFit = doQiFit;
  cfg.plotQiOverlay = plotQiOverlay;
  cfg.filename = filename;
  cfg.errorPercentOfMax = errorPercentOfMax;
  cfg.nRandomEvents = nRandomEvents;
  cfg.replayMode = replayMode;
  cfg.outputPdf = outputPdf;
  return plotFitGaus1DUnified(cfg);
}

// Simple filename-only version (row/col replay with Qi fit)
int plotFitGaus1D(const char* filename) {
  return plotFitGaus1D(true, false, true, true, filename, 5.0, 100, true);
}

// Default: row/col replay with Qi overlay and fitting
int plotFitGaus1D() {
  return plotFitGaus1D(true, false, true, true);
}

// === Preset modes ===

// Fresh fitting (no replay)
int plotFitGaus1DFresh(const char* filename = "../build/epicChargeSharing.root",
                       Long64_t nEvents = 100) {
  return plotFitGaus1D(true, false, false, false, filename, 5.0, nEvents, false, "1Dfits_fresh.pdf");
}

// Row/col replay only (no Qi)
int plotFitGaus1DReplay(const char* filename = "../build/epicChargeSharing.root",
                        Long64_t nEvents = 100) {
  return plotFitGaus1D(true, false, false, true, filename, 5.0, nEvents, true, "1Dfits_replay.pdf");
}

// Row/col replay + Qi fit
int plotFitGaus1DReplayQiFit(const char* filename = "../build/epicChargeSharing.root",
                             Long64_t nEvents = 100) {
  return plotFitGaus1D(true, false, true, true, filename, 5.0, nEvents, true, "1Dfits_replay_qifit.pdf");
}

// Diagonals replay only (no Qi)
int plotFitGaus1DDiagReplay(const char* filename = "../build/epicChargeSharing.root",
                            Long64_t nEvents = 100) {
  return plotFitGaus1D(false, true, false, true, filename, 5.0, nEvents, true, "1Dfits_diag_replay.pdf");
}

// Diagonals replay + Qi fit
int plotFitGaus1DDiagReplayQiFit(const char* filename = "../build/epicChargeSharing.root",
                                 Long64_t nEvents = 100) {
  return plotFitGaus1D(false, true, true, true, filename, 5.0, nEvents, true, "1Dfits_diag_replay_qifit.pdf");
}

// Both row/col and diagonals with Qi fit (replay)
int plotFitGaus1DWithDiag(const char* filename = "../build/epicChargeSharing.root",
                          Long64_t nEvents = 100) {
  // Do row/col pages first
  plotFitGaus1D(true, false, true, true, filename, 5.0, nEvents, true, "1Dfits_rowcol_qifit.pdf");
  // Then diagonal pages
  return plotFitGaus1D(false, true, true, true, filename, 5.0, nEvents, true, "1Dfits_diag_qifit.pdf");
}
