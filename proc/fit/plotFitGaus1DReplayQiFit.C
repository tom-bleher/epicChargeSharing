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
#include <TMath.h>

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
#include <limits>

#include "../../src/ChargeUtils.h"

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

// Replay Q_f/F_i fits (from saved parameters) and additionally fit Q_i points
// exactly like diagnostic/fit/plotProcessing1D.C does, drawing dashed green
// Gaussian curves and vertical recon lines for Q_i with legend deltas.
int processing1D_replay_qifit(const char* filename =
                              "/home/tomble/epicChargeSharing/build/epicChargeSharing.root",
                              double errorPercentOfMax = 5.0,
                              Long64_t nRandomEvents = 200,
                              bool plotQiOverlayPts = true,
                              bool doQiFit = true,
                              bool useQiQnPercentErrors = false,
                              const char* outputPdfPath = nullptr) {
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
      ::Error("processing1D_replay_qifit", "Cannot open file: %s or fallback: %s", filename, fallback.Data());
      if (file) { file->Close(); delete file; }
      return 1;
    }
  }

  // Get tree
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing1D_replay_qifit", "Hits tree not found in file: %s", filename);
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
    ::Error("processing1D_replay_qifit", "Pixel spacing unavailable (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 3;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    pixelSize = 0.5 * pixelSpacing; // mm
  }

  // Decide which charge branch to use for plotting points
  std::string chosenCharge;
  if (tree->GetBranch("Q_f")) chosenCharge = "Q_f";
  else if (tree->GetBranch("F_i")) chosenCharge = "F_i";
  else if (tree->GetBranch("Q_i")) chosenCharge = "Q_i";
  else {
    ::Error("processing1D_replay_qifit", "No charge branch found (tried Q_f, F_i, Q_i)");
    file->Close();
    delete file;
    return 4;
  }

  // Set up branches
  double x_true = 0.0, y_true = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_true = kFALSE;
  std::vector<double>* Q = nullptr;
  std::vector<double>* QiVec = nullptr; // Optional overlay + fitting source
  std::vector<double>* QnVec = nullptr;
  bool enableQiQnErrors = useQiQnPercentErrors;
  // Saved fit parameters
  double rowA = NAN, rowMu = NAN, rowSig = NAN, rowB = NAN;
  double rowChi2 = NAN, rowNdf = NAN, rowProb = NAN;
  double colA = NAN, colMu = NAN, colSig = NAN, colB = NAN;
  double colChi2 = NAN, colNdf = NAN, colProb = NAN;
  // Saved removal masks (0 = kept, 1 = removed)
  std::vector<int>* rowMask = nullptr;
  std::vector<int>* colMask = nullptr;

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
  const bool haveQnBranch = (tree->GetBranch("Q_n") != nullptr);
  if (haveQnBranch) {
    tree->SetBranchAddress("Q_n", &QnVec);
  }
  if (enableQiQnErrors && (!haveQiBranch || !haveQnBranch)) {
    ::Warning("processing1D_replay_qifit", "Requested Q_i/Q_n vertical errors but required branches are missing. Falling back to percent-of-max uncertainty.");
    enableQiQnErrors = false;
  }

  bool haveRowA = tree->GetBranch("GaussRowA") != nullptr;
  bool haveRowMu = tree->GetBranch("GaussRowMu") != nullptr;
  bool haveRowSig = tree->GetBranch("GaussRowSigma") != nullptr;
  bool haveRowB = tree->GetBranch("GaussRowB") != nullptr;
  bool haveRowChi2 = tree->GetBranch("GaussRowChi2") != nullptr;
  bool haveRowNdf = tree->GetBranch("GaussRowNdf") != nullptr;
  bool haveRowProb = tree->GetBranch("GaussRowProb") != nullptr;
  bool haveColA = tree->GetBranch("GaussColA") != nullptr;
  bool haveColMu = tree->GetBranch("GaussColMu") != nullptr;
  bool haveColSig = tree->GetBranch("GaussColSigma") != nullptr;
  bool haveColB = tree->GetBranch("GaussColB") != nullptr;
  bool haveColChi2 = tree->GetBranch("GaussColChi2") != nullptr;
  bool haveColNdf = tree->GetBranch("GaussColNdf") != nullptr;
  bool haveColProb = tree->GetBranch("GaussColProb") != nullptr;
  bool haveAnyParams = haveRowMu && haveColMu; // at least mus to draw rec lines
  if (haveRowA) tree->SetBranchAddress("GaussRowA", &rowA);
  if (haveRowMu) tree->SetBranchAddress("GaussRowMu", &rowMu);
  if (haveRowSig) tree->SetBranchAddress("GaussRowSigma", &rowSig);
  if (haveRowB) tree->SetBranchAddress("GaussRowB", &rowB);
  if (haveRowChi2) tree->SetBranchAddress("GaussRowChi2", &rowChi2);
  if (haveRowNdf) tree->SetBranchAddress("GaussRowNdf", &rowNdf);
  if (haveRowProb) tree->SetBranchAddress("GaussRowProb", &rowProb);
  if (haveColA) tree->SetBranchAddress("GaussColA", &colA);
  if (haveColMu) tree->SetBranchAddress("GaussColMu", &colMu);
  if (haveColSig) tree->SetBranchAddress("GaussColSigma", &colSig);
  if (haveColB) tree->SetBranchAddress("GaussColB", &colB);
  if (haveColChi2) tree->SetBranchAddress("GaussColChi2", &colChi2);
  if (haveColNdf) tree->SetBranchAddress("GaussColNdf", &colNdf);
  if (haveColProb) tree->SetBranchAddress("GaussColProb", &colProb);

  bool haveRowMask = tree->GetBranch("GaussRowMaskRemoved") != nullptr;
  bool haveColMask = tree->GetBranch("GaussColMaskRemoved") != nullptr;
  if (haveRowMask) tree->SetBranchAddress("GaussRowMaskRemoved", &rowMask);
  if (haveColMask) tree->SetBranchAddress("GaussColMaskRemoved", &colMask);

  // Prepare plotting objects
  gROOT->SetBatch(true);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  TCanvas c("c2up_replay_qifit", "processing1D replay + Qi fit", 1800, 700);
  TPad pL("pL", "column-left", 0.0, 0.0, 0.5, 1.0);
  TPad pR("pR", "row-right",  0.5, 0.0, 1.0, 1.0);
  pL.SetTicks(1,1);
  pR.SetTicks(1,1);
  pL.Draw();
  pR.Draw();

  // Open multipage PDF
  const std::string outputPdf = (outputPdfPath && outputPdfPath[0] != '\0')
                                  ? outputPdfPath
                                  : "1Dfits_replay_qifit.pdf";

  c.Print((outputPdf + "[").c_str());

  TF1 fRow("fRow_replay", GaussPlusB, -1e9, 1e9, 4);
  TF1 fCol("fCol_replay", GaussPlusB, -1e9, 1e9, 4);
  TF1 fRowQi("fRow_qi", GaussPlusB, -1e9, 1e9, 4);
  TF1 fColQi("fCol_qi", GaussPlusB, -1e9, 1e9, 4);

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
    const bool haveQiQnForEvent = enableQiQnErrors &&
                                  QiVec && QnVec &&
                                  QiVec->size() == QnVec->size();

    const size_t total = Q->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) continue;
    const int R = (N - 1) / 2;

    // Build full row/column samples from raw charges (chosenCharge)
    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    std::vector<int> rowIdx, colIdx; // indices into flattened NxN neighborhood
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);
   rowIdx.reserve(N); colIdx.reserve(N);
    double qmaxNeighborhood = -1e300;
    double qmaxQiNeighborhood = -1e300;
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx  = (di + R) * N + (dj + R);
        const double q = (*Q)[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        if (haveQiQnForEvent && QiVec && idx >= 0 &&
            static_cast<size_t>(idx) < QiVec->size()) {
          const double qiCandidate = (*QiVec)[idx];
          if (IsFinite(qiCandidate) && qiCandidate > qmaxQiNeighborhood) {
            qmaxQiNeighborhood = qiCandidate;
          }
        }
        if (dj == 0) {
          const double x = x_px + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
          rowIdx.push_back(idx);
        }
        if (di == 0) {
          const double y = y_px + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
          colIdx.push_back(idx);
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

    // Reconstruct the filtered sets used for plotting (kept points only)
    std::vector<double> x_row_fit, q_row_fit, y_col_fit, q_col_fit;
    std::vector<int> rowIdx_fit, colIdx_fit;
    x_row_fit.reserve(x_row.size()); q_row_fit.reserve(q_row.size());
    y_col_fit.reserve(y_col.size()); q_col_fit.reserve(q_col.size());
    for (size_t k=0;k<x_row.size();++k) if (rowKeptMask[k]) { x_row_fit.push_back(x_row[k]); q_row_fit.push_back(q_row[k]); rowIdx_fit.push_back(rowIdx[k]); }
    for (size_t k=0;k<y_col.size();++k) if (colKeptMask[k]) { y_col_fit.push_back(y_col[k]); q_col_fit.push_back(q_col[k]); colIdx_fit.push_back(colIdx[k]); }
    if ((int)x_row_fit.size() < 3) { x_row_fit = x_row; q_row_fit = q_row; rowIdx_fit = rowIdx; }
    if ((int)y_col_fit.size() < 3) { y_col_fit = y_col; q_col_fit = q_col; colIdx_fit = colIdx; }

    std::vector<double> rowErrors;
    std::vector<double> colErrors;
    if (haveQiQnForEvent) {
      rowErrors.reserve(rowIdx_fit.size());
      for (int idx : rowIdx_fit) {
        double candidate = std::numeric_limits<double>::quiet_NaN();
        if (idx >= 0 && QiVec && QnVec &&
            static_cast<size_t>(idx) < QiVec->size() &&
            static_cast<size_t>(idx) < QnVec->size()) {
          candidate = ComputeQnQiPercent((*QiVec)[idx], (*QnVec)[idx], qmaxQiNeighborhood);
        }
        rowErrors.push_back(candidate);
      }
      colErrors.reserve(colIdx_fit.size());
      for (int idx : colIdx_fit) {
        double candidate = std::numeric_limits<double>::quiet_NaN();
        if (idx >= 0 && QiVec && QnVec &&
            static_cast<size_t>(idx) < QiVec->size() &&
            static_cast<size_t>(idx) < QnVec->size()) {
          candidate = ComputeQnQiPercent((*QiVec)[idx], (*QnVec)[idx], qmaxQiNeighborhood);
        }
        colErrors.push_back(candidate);
      }
    }

    // Must have mus to draw recon lines for saved Q_f/F_i parameters
    bool haveMuRow = haveRowMu && IsFinite(rowMu);
    bool haveMuCol = haveColMu && IsFinite(colMu);
    if (!(haveMuRow && haveMuCol)) continue;

    nConsidered++;

    // Optional uniform vertical uncertainty from percent-of-max for chosenCharge
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;
    auto selectError = [&](double candidate) -> double {
      if (std::isfinite(candidate) && candidate > 0.0) return candidate;
      if (uniformSigma > 0.0) return uniformSigma;
      return 1.0;
    };
    auto hasFinitePositive = [](const std::vector<double>& vals) {
      for (double v : vals) {
        if (std::isfinite(v) && v > 0.0) return true;
      }
      return false;
    };
    const bool rowHasErrorBars = (uniformSigma > 0.0) || (haveQiQnForEvent && hasFinitePositive(rowErrors));
    const bool colHasErrorBars = (uniformSigma > 0.0) || (haveQiQnForEvent && hasFinitePositive(colErrors));

    // Build graphs for kept points only (to mirror original drawing)
    TGraph* baseRowPtr = nullptr;
    TGraph* baseColPtr = nullptr;
    TGraph gRowPlain;
    TGraph gColPlain;
    TGraphErrors gRowErr;
    TGraphErrors gColErr;

    if (rowHasErrorBars) {
      gRowErr = TGraphErrors(static_cast<int>(x_row_fit.size()));
      for (int k = 0; k < gRowErr.GetN(); ++k) {
        gRowErr.SetPoint(k, x_row_fit[k], q_row_fit[k]);
        const double candidate = (haveQiQnForEvent && k < static_cast<int>(rowErrors.size())) ? rowErrors[k]
                                                                                              : std::numeric_limits<double>::quiet_NaN();
        gRowErr.SetPointError(k, 0.0, selectError(candidate));
      }
      baseRowPtr = &gRowErr;
    } else {
      gRowPlain = TGraph(static_cast<int>(x_row_fit.size()));
      for (int k = 0; k < gRowPlain.GetN(); ++k) gRowPlain.SetPoint(k, x_row_fit[k], q_row_fit[k]);
      baseRowPtr = &gRowPlain;
    }

    if (colHasErrorBars) {
      gColErr = TGraphErrors(static_cast<int>(y_col_fit.size()));
      for (int k = 0; k < gColErr.GetN(); ++k) {
        gColErr.SetPoint(k, y_col_fit[k], q_col_fit[k]);
        const double candidate = (haveQiQnForEvent && k < static_cast<int>(colErrors.size())) ? colErrors[k]
                                                                                              : std::numeric_limits<double>::quiet_NaN();
        gColErr.SetPointError(k, 0.0, selectError(candidate));
      }
      baseColPtr = &gColErr;
    } else {
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

    // =========================
    // Optional Q_i fits (match plotProcessing1D.C logic)
    // =========================
    bool haveQi = (haveQiBranch && QiVec);
    bool didRowFitQi = false, didColFitQi = false;
    double A_row_qi = NAN, B_row_qi = NAN, S_row_qi = NAN, muRowQi = NAN;
    double chi2_row_qi = NAN, ndf_row_qi = NAN, prob_row_qi = NAN;
    double A_col_qi = NAN, B_col_qi = NAN, S_col_qi = NAN, muColQi = NAN;
    double chi2_col_qi = NAN, ndf_col_qi = NAN, prob_col_qi = NAN;
    double qmaxNeighborhoodQi = qmaxQiNeighborhood;
    if (haveQi && doQiFit) {
      // compute qmax over full neighborhood for Qi
      for (int di = -R; di <= R; ++di) {
        for (int dj = -R; dj <= R; ++dj) {
          const int idx  = (di + R) * N + (dj + R);
          if (idx >= 0 && idx < (int)QiVec->size()) {
            const double qqi = (*QiVec)[idx];
            if (IsFinite(qqi) && qqi >= 0.0 && qqi > qmaxNeighborhoodQi) qmaxNeighborhoodQi = qqi;
          }
        }
      }

      // Build full Qi samples along row/col using full sets (no clipping)
      std::vector<double> x_row_qi, q_row_qi;
      std::vector<double> y_col_qi, q_col_qi;
      x_row_qi.reserve(rowIdx.size()); q_row_qi.reserve(rowIdx.size());
      y_col_qi.reserve(colIdx.size()); q_col_qi.reserve(colIdx.size());
      std::vector<double> rowErrorsQi;
      std::vector<double> colErrorsQi;
      if (haveQiQnForEvent) {
        rowErrorsQi.reserve(rowIdx.size());
        colErrorsQi.reserve(colIdx.size());
      }
      for (size_t k = 0; k < rowIdx.size(); ++k) {
        const int idx = rowIdx[k];
        if (idx >= 0 && idx < (int)QiVec->size()) {
          const double qqi = (*QiVec)[idx];
          if (IsFinite(qqi) && qqi >= 0.0) {
            x_row_qi.push_back(x_row[k]);
            q_row_qi.push_back(qqi);
            if (haveQiQnForEvent && QnVec && idx >= 0 && static_cast<size_t>(idx) < QnVec->size()) {
              rowErrorsQi.push_back(ComputeQnQiPercent(qqi, (*QnVec)[idx], qmaxNeighborhoodQi));
            } else if (haveQiQnForEvent) {
              rowErrorsQi.push_back(std::numeric_limits<double>::quiet_NaN());
            }
          }
        }
      }
      for (size_t k = 0; k < colIdx.size(); ++k) {
        const int idx = colIdx[k];
        if (idx >= 0 && idx < (int)QiVec->size()) {
          const double qqi = (*QiVec)[idx];
          if (IsFinite(qqi) && qqi >= 0.0) {
            y_col_qi.push_back(y_col[k]);
            q_col_qi.push_back(qqi);
            if (haveQiQnForEvent && QnVec && idx >= 0 && static_cast<size_t>(idx) < QnVec->size()) {
              colErrorsQi.push_back(ComputeQnQiPercent(qqi, (*QnVec)[idx], qmaxNeighborhoodQi));
            } else if (haveQiQnForEvent) {
              colErrorsQi.push_back(std::numeric_limits<double>::quiet_NaN());
            }
          }
        }
      }
      const double uniformSigmaQi = (qmaxNeighborhoodQi > 0 && relErr > 0.0)
                                  ? relErr * qmaxNeighborhoodQi
                                  : 0.0;

      auto fit1DQi = [&](const std::vector<double>& xs, const std::vector<double>& qs,
                          const std::vector<double>* errVals,
                          double muLo, double muHi,
                          double& outA, double& outMu, double& outSig, double& outB,
                          double& outChi2, double& outNdf, double& outProb) -> bool {
        outChi2 = std::numeric_limits<double>::quiet_NaN();
        outNdf  = std::numeric_limits<double>::quiet_NaN();
        outProb = std::numeric_limits<double>::quiet_NaN();
        if (xs.size() < 3 || qs.size() < 3) return false;
        auto mm = std::minmax_element(qs.begin(), qs.end());
        double A0 = std::max(1e-18, *mm.second - *mm.first);
        double B0 = std::max(0.0, *mm.first);
        int idxMax = std::distance(qs.begin(), std::max_element(qs.begin(), qs.end()));
        double mu0 = xs[idxMax];
        const double sigLoBound = pixelSize;
        const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
        auto sigmaSeed1D = [&](const std::vector<double>& xs_, const std::vector<double>& qs_, double B0_)->double {
          double wsum = 0.0, xw = 0.0;
          for (size_t i=0;i<xs_.size();++i) { double w = std::max(0.0, qs_[i] - B0_); wsum += w; xw += w * xs_[i]; }
          if (wsum <= 0.0) return std::max(0.25*pixelSpacing, 1e-6);
          const double mean = xw / wsum;
          double var = 0.0;
          for (size_t i=0;i<xs_.size();++i) { double w = std::max(0.0, qs_[i] - B0_); const double dx = xs_[i] - mean; var += w * dx * dx; }
          var = (wsum > 0.0) ? (var / wsum) : 0.0;
          double s = std::sqrt(std::max(var, 1e-12));
          if (s < sigLoBound) s = sigLoBound;
          if (s > sigHiBound) s = sigHiBound;
          return s;
        };
        double sigInit = sigmaSeed1D(xs, qs, B0);

        ROOT::Math::WrappedMultiTF1 wLoc(fRowQi, 1); // function signature is same; we will not use fRowQi object here
        TF1 fLoc("fQiLoc", GaussPlusB, -1e9, 1e9, 4);
        fLoc.SetParameters(A0, mu0, sigInit, B0);
        ROOT::Math::WrappedMultiTF1 wModel(fLoc, 1);
        ROOT::Fit::BinData data(static_cast<int>(xs.size()), 1);
        auto selectErrorQi = [&](double candidate)->double {
          if (std::isfinite(candidate) && candidate > 0.0) return candidate;
          if (uniformSigmaQi > 0.0) return uniformSigmaQi;
          return 1.0;
        };
        for (size_t i=0;i<xs.size();++i) {
          const double candidate = (errVals && i < errVals->size()) ? (*errVals)[i] : std::numeric_limits<double>::quiet_NaN();
          data.Add(xs[i], qs[i], selectErrorQi(candidate));
        }
        ROOT::Fit::Fitter fitter;
        fitter.Config().SetMinimizer("Minuit2", "Fumili2");
        fitter.Config().MinimizerOptions().SetStrategy(0);
        fitter.Config().MinimizerOptions().SetTolerance(1e-4);
        fitter.Config().MinimizerOptions().SetPrintLevel(0);
        fitter.SetFunction(wModel);
        const double AHi = std::max(1e-18, 2.0 * std::max(qmaxNeighborhoodQi, 0.0));
        const double BHi = std::max(1e-18, 1.0 * std::max(qmaxNeighborhoodQi, 0.0));
        fitter.Config().ParSettings(0).SetLimits(1e-18, AHi);
        fitter.Config().ParSettings(1).SetLimits(muLo, muHi);
        fitter.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
        fitter.Config().ParSettings(3).SetLimits(0.0, BHi);
        fitter.Config().ParSettings(0).SetStepSize(std::max(1e-18, 0.01 * A0));
        fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(3).SetStepSize(std::max(1e-18, 0.01 * std::max(B0, A0)));
        fitter.Config().ParSettings(0).SetValue(A0);
        fitter.Config().ParSettings(1).SetValue(mu0);
        fitter.Config().ParSettings(2).SetValue(sigInit);
        fitter.Config().ParSettings(3).SetValue(B0);
        bool ok = fitter.Fit(data);
        if (ok) {
          const auto& res = fitter.Result();
          outA = res.Parameter(0);
          outMu = res.Parameter(1);
          outSig = res.Parameter(2);
          outB = res.Parameter(3);
          outChi2 = res.Chi2();
          outNdf  = res.Ndf();
          outProb = (res.Ndf() > 0) ? TMath::Prob(res.Chi2(), res.Ndf()) : std::numeric_limits<double>::quiet_NaN();
          return true;
        }
        // Fallback: baseline-subtracted weighted centroid
        double wsum = 0.0, xw = 0.0;
        for (size_t i=0;i<xs.size();++i) { double w = std::max(0.0, qs[i] - B0); wsum += w; xw += w * xs[i]; }
        if (wsum > 0) { outA = A0; outB = B0; outSig = sigInit; outMu = xw / wsum; return true; }
        return false;
      };

      const double muXLo = x_px - 0.5 * pixelSpacing;
      const double muXHi = x_px + 0.5 * pixelSpacing;
      const double muYLo = y_px - 0.5 * pixelSpacing;
      const double muYHi = y_px + 0.5 * pixelSpacing;
      const bool rowErrorSizeMatch = haveQiQnForEvent && (rowErrorsQi.size() == x_row_qi.size());
      const bool colErrorSizeMatch = haveQiQnForEvent && (colErrorsQi.size() == y_col_qi.size());
      const std::vector<double>* rowErrPtrQi = rowErrorSizeMatch ? &rowErrorsQi : nullptr;
      const std::vector<double>* colErrPtrQi = colErrorSizeMatch ? &colErrorsQi : nullptr;
      if (x_row_qi.size() >= 3) {
        didRowFitQi = fit1DQi(x_row_qi, q_row_qi, rowErrPtrQi, muXLo, muXHi,
                               A_row_qi, muRowQi, S_row_qi, B_row_qi,
                               chi2_row_qi, ndf_row_qi, prob_row_qi);
      }
      if (y_col_qi.size() >= 3) {
        didColFitQi = fit1DQi(y_col_qi, q_col_qi, colErrPtrQi, muYLo, muYHi,
                               A_col_qi, muColQi, S_col_qi, B_col_qi,
                               chi2_col_qi, ndf_col_qi, prob_col_qi);
      }
    }

    // Determine Y-axis limits using all samples (kept+removed) and model baseline/peak
    double dataMinRow = *std::min_element(q_row.begin(), q_row.end());
    double dataMaxRow = *std::max_element(q_row.begin(), q_row.end());
    double yMinRow = dataMinRow;
    double yMaxRow = dataMaxRow;
    if (haveRowA && IsFinite(rowA) && haveRowB && IsFinite(rowB)) {
      yMaxRow = std::max(yMaxRow, rowA + rowB);
      yMinRow = std::min(yMinRow, rowB);
    }
    if (didRowFitQi && IsFinite(A_row_qi) && IsFinite(B_row_qi)) {
      yMaxRow = std::max(yMaxRow, A_row_qi + B_row_qi);
      yMinRow = std::min(yMinRow, B_row_qi);
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
    if (didColFitQi && IsFinite(A_col_qi) && IsFinite(B_col_qi)) {
      yMaxCol = std::max(yMaxCol, A_col_qi + B_col_qi);
      yMinCol = std::min(yMinCol, B_col_qi);
    }
    double padCol = 0.10 * (yMaxCol - yMinCol);
    if (!(padCol > 0)) padCol = 0.10 * std::max(std::abs(yMaxCol), 1.0);
    baseColPtr->SetMinimum(yMinCol - padCol);
    baseColPtr->SetMaximum(yMaxCol + padCol);

    // Draw COLUMN (left)
    pL.cd();
    baseColPtr->Draw("AP");
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
    // Draw Qi fit curve if available
    if (didColFitQi) {
      auto minmaxY = std::minmax_element(y_col.begin(), y_col.end());
      const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
      const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
      fColQi.SetRange(yMin, yMax);
      fColQi.SetParameters(A_col_qi, muColQi, S_col_qi, B_col_qi);
      fColQi.SetNpx(600);
      fColQi.SetLineWidth(2);
      fColQi.SetLineStyle(2);
      fColQi.SetLineColor(kGreen+2);
      fColQi.Draw("L SAME");
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
    // Optional overlay of Q_i points (kept subset)
    bool drewColQiPts = false;
    TGraph gColQiPts;
    const bool canOverlayQi = plotQiOverlayPts && haveQiBranch && (chosenCharge != "Q_i");
    if (canOverlayQi && QiVec) {
      std::vector<std::pair<double,double>> colPoints;
      colPoints.reserve(colIdx_fit.size());
      for (size_t k = 0; k < colIdx_fit.size(); ++k) {
        const int idxQi = colIdx_fit[k];
        if (idxQi >= 0 && idxQi < static_cast<int>(QiVec->size())) {
          const double qqi = (*QiVec)[idxQi];
          if (IsFinite(qqi) && qqi >= 0.0) colPoints.emplace_back(y_col_fit[k], qqi);
        }
      }
      if (!colPoints.empty()) {
        gColQiPts = TGraph(static_cast<int>(colPoints.size()));
        for (int k = 0; k < gColQiPts.GetN(); ++k) gColQiPts.SetPoint(k, colPoints[k].first, colPoints[k].second);
        gColQiPts.SetMarkerStyle(24);
        gColQiPts.SetMarkerSize(1.1);
        gColQiPts.SetMarkerColor(kGreen+2);
        gColQiPts.SetLineColor(kGreen+2);
        gColQiPts.Draw("P SAME");
        drewColQiPts = true;
      }
    }
    TLine lineYtrue(y_true, yPadMinC, y_true, yPadMaxC);
    lineYtrue.SetLineStyle(1);
    lineYtrue.SetLineWidth(3);
    lineYtrue.SetLineColor(kBlack);
    lineYtrue.Draw("SAME");
    TLine lineYrec(colMu, yPadMinC, colMu, yPadMaxC);
    lineYrec.SetLineStyle(2);
    lineYrec.SetLineWidth(2);
    lineYrec.SetLineColor(kRed+1);
    lineYrec.Draw("SAME");
    TLine lineYrecQi;
    if (didColFitQi && IsFinite(muColQi)) {
      lineYrecQi = TLine(muColQi, yPadMinC, muColQi, yPadMaxC);
      lineYrecQi.SetLineStyle(2);
      lineYrecQi.SetLineWidth(2);
      lineYrecQi.SetLineColor(kGreen+2);
      lineYrecQi.Draw("SAME");
    }

    double fx1c = gPad->GetLeftMargin();
    double fy1c = gPad->GetBottomMargin();
    double fx2c = 1.0 - gPad->GetRightMargin();
    double fy2c = 1.0 - gPad->GetTopMargin();
    double legWc = 0.28, legHc = 0.34, insetc = 0.008;
    double ly2c = fy2c - insetc, ly1c = ly2c - legHc;
    double lx1c = fx1c + insetc, lx2c = lx1c + legWc;
    TLegend legCLeft(lx1c, ly1c, lx2c, ly2c, "", "NDC");
    legCLeft.SetBorderSize(0);
    legCLeft.SetFillStyle(0);
    legCLeft.SetTextSize(0.03);
    {
      std::string basePtsLabel = (chosenCharge == "Q_f") ? "Q_{f} points"
                                 : (chosenCharge == "F_i") ? "F_{i} points"
                                 : "Q_{i} points";
      legCLeft.AddEntry(baseColPtr, basePtsLabel.c_str(), "p");
    }
    if (didColFit) legCLeft.AddEntry(&fCol, "Gaussian fit (Q_{f})", "l");
    if (didColFitQi) legCLeft.AddEntry(&fColQi, "Gaussian fit (Q_{i})", "l");
    legCLeft.AddEntry(&lineYtrue, "y_{true}", "l");
    legCLeft.AddEntry(&lineYrec,  "y_{rec}(Q_{f})",  "l");
    if (didColFitQi && IsFinite(muColQi)) legCLeft.AddEntry(&lineYrecQi, "y_{rec}(Q_{i})", "l");
    if (drewColQiPts) legCLeft.AddEntry(&gColQiPts, "Q_{i} points", "p");
    legCLeft.Draw();
    double rx2c = fx2c - insetc, rx1c = rx2c - legWc;
    TLegend legCRight(rx1c, ly1c, rx2c, ly2c, "", "NDC");
    legCRight.SetBorderSize(0);
    legCRight.SetFillStyle(0);
    legCRight.SetTextSize(0.03);
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true} = %.4f mm", y_true), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{rec}(Q_{f}) = %.4f mm", colMu), "");
    if (didColFit) legCRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit}(Q_{f}) = %.3f mm", colSig), "");
    if (didColFit && haveColChi2 && haveColNdf && IsFinite(colChi2) && IsFinite(colNdf) && colNdf > 0.0) {
      legCRight.AddEntry((TObject*)nullptr,
                         Form("#chi^{2}/ndf (Q_{f}) = %.2f/%d", colChi2, static_cast<int>(std::lround(colNdf))),
                         "");
      if (haveColProb && IsFinite(colProb)) {
        legCRight.AddEntry((TObject*)nullptr, Form("P(Q_{f}) = %.3g", colProb), "");
      }
    }
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{px} = %.1f #mum", 1000.0*(y_true - y_px)), "");
    legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{rec}(Q_{f}) = %.1f #mum", 1000.0*(y_true - colMu)), "");
    if (didColFitQi) {
      legCRight.AddEntry((TObject*)nullptr, Form("y_{rec}(Q_{i}) = %.4f mm", muColQi), "");
      legCRight.AddEntry((TObject*)nullptr, Form("#sigma_{Q_{i}} = %.3f mm", S_col_qi), "");
      if (IsFinite(chi2_col_qi) && IsFinite(ndf_col_qi) && ndf_col_qi > 0.0) {
        legCRight.AddEntry((TObject*)nullptr,
                           Form("#chi^{2}/ndf (Q_{i}) = %.2f/%d", chi2_col_qi, static_cast<int>(std::lround(ndf_col_qi))),
                           "");
        if (IsFinite(prob_col_qi)) {
          legCRight.AddEntry((TObject*)nullptr, Form("P(Q_{i}) = %.3g", prob_col_qi), "");
        }
      }
      legCRight.AddEntry((TObject*)nullptr, Form("y_{true}-y_{rec}(Q_{i}) = %.1f #mum", 1000.0*(y_true - muColQi)), "");
    }
    legCRight.Draw();

    // Draw ROW (right)
    pR.cd();
    baseRowPtr->Draw("AP");
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
    if (didRowFitQi) {
      auto minmaxX = std::minmax_element(x_row.begin(), x_row.end());
      const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
      const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
      fRowQi.SetRange(xMin, xMax);
      fRowQi.SetParameters(A_row_qi, muRowQi, S_row_qi, B_row_qi);
      fRowQi.SetNpx(600);
      fRowQi.SetLineWidth(2);
      fRowQi.SetLineStyle(2);
      fRowQi.SetLineColor(kGreen+2);
      fRowQi.Draw("L SAME");
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
    // Optional overlay of Q_i points (kept subset)
    bool drewRowQiPts = false;
    TGraph gRowQiPts;
    if (canOverlayQi && QiVec) {
      std::vector<std::pair<double,double>> rowPoints;
      rowPoints.reserve(rowIdx_fit.size());
      for (size_t k = 0; k < rowIdx_fit.size(); ++k) {
        const int idxQi = rowIdx_fit[k];
        if (idxQi >= 0 && idxQi < static_cast<int>(QiVec->size())) {
          const double qqi = (*QiVec)[idxQi];
          if (IsFinite(qqi) && qqi >= 0.0) rowPoints.emplace_back(x_row_fit[k], qqi);
        }
      }
      if (!rowPoints.empty()) {
        gRowQiPts = TGraph(static_cast<int>(rowPoints.size()));
        for (int k = 0; k < gRowQiPts.GetN(); ++k) gRowQiPts.SetPoint(k, rowPoints[k].first, rowPoints[k].second);
        gRowQiPts.SetMarkerStyle(24);
        gRowQiPts.SetMarkerSize(1.1);
        gRowQiPts.SetMarkerColor(kGreen+2);
        gRowQiPts.SetLineColor(kGreen+2);
        gRowQiPts.Draw("P SAME");
        drewRowQiPts = true;
      }
    }
    TLine lineXtrue(x_true, yPadMin, x_true, yPadMax);
    lineXtrue.SetLineStyle(1);
    lineXtrue.SetLineWidth(3);
    lineXtrue.SetLineColor(kBlack);
    lineXtrue.Draw("SAME");
    TLine lineXrec(rowMu, yPadMin, rowMu, yPadMax);
    lineXrec.SetLineStyle(2);
    lineXrec.SetLineWidth(2);
    lineXrec.SetLineColor(kRed+1);
    lineXrec.Draw("SAME");
    TLine lineXrecQi;
    if (didRowFitQi && IsFinite(muRowQi)) {
      lineXrecQi = TLine(muRowQi, yPadMin, muRowQi, yPadMax);
      lineXrecQi.SetLineStyle(2);
      lineXrecQi.SetLineWidth(2);
      lineXrecQi.SetLineColor(kGreen+2);
      lineXrecQi.Draw("SAME");
    }

    double fx1 = gPad->GetLeftMargin();
    double fy1 = gPad->GetBottomMargin();
    double fx2 = 1.0 - gPad->GetRightMargin();
    double fy2 = 1.0 - gPad->GetTopMargin();
    double legW = 0.28, legH = 0.34, inset = 0.008;
    double ly2 = fy2 - inset, ly1 = ly2 - legH;
    double lx1 = fx1 + inset, lx2 = lx1 + legW;
    TLegend legLeft(lx1, ly1, lx2, ly2, "", "NDC");
    legLeft.SetBorderSize(0);
    legLeft.SetFillStyle(0);
    legLeft.SetTextSize(0.03);
    {
      std::string basePtsLabel = (chosenCharge == "Q_f") ? "Q_{f} points"
                                 : (chosenCharge == "F_i") ? "F_{i} points"
                                 : "Q_{i} points";
      legLeft.AddEntry(baseRowPtr, basePtsLabel.c_str(), "p");
    }
    if (didRowFit) legLeft.AddEntry(&fRow, "Gaussian fit (Q_{f})", "l");
    if (didRowFitQi) legLeft.AddEntry(&fRowQi, "Gaussian fit (Q_{i})", "l");
    legLeft.AddEntry(&lineXtrue, "x_{true}", "l");
    legLeft.AddEntry(&lineXrec,  "x_{rec}(Q_{f})",  "l");
    if (didRowFitQi && IsFinite(muRowQi)) legLeft.AddEntry(&lineXrecQi,  "x_{rec}(Q_{i})",  "l");
    if (drewRowQiPts) legLeft.AddEntry(&gRowQiPts, "Q_{i} points", "p");
    legLeft.Draw();
    double rx2 = fx2 - inset, rx1 = rx2 - legW;
    TLegend legRight(rx1, ly1, rx2, ly2, "", "NDC");
    legRight.SetBorderSize(0);
    legRight.SetFillStyle(0);
    legRight.SetTextSize(0.03);
    legRight.AddEntry((TObject*)nullptr, Form("x_{true} = %.4f mm", x_true), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{rec}(Q_{f}) = %.4f mm", rowMu), "");
    if (didRowFit) legRight.AddEntry((TObject*)nullptr, Form("#sigma_{fit}(Q_{f}) = %.3f mm", rowSig), "");
    if (didRowFit && haveRowChi2 && haveRowNdf && IsFinite(rowChi2) && IsFinite(rowNdf) && rowNdf > 0.0) {
      legRight.AddEntry((TObject*)nullptr,
                        Form("#chi^{2}/ndf (Q_{f}) = %.2f/%d", rowChi2, static_cast<int>(std::lround(rowNdf))),
                        "");
      if (haveRowProb && IsFinite(rowProb)) {
        legRight.AddEntry((TObject*)nullptr, Form("P(Q_{f}) = %.3g", rowProb), "");
      }
    }
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{px} = %.1f #mum", 1000.0*(x_true - x_px)), "");
    legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec}(Q_{f}) = %.1f #mum", 1000.0*(x_true - rowMu)), "");
    if (didRowFitQi) {
      legRight.AddEntry((TObject*)nullptr, Form("x_{rec}(Q_{i}) = %.4f mm", muRowQi), "");
      legRight.AddEntry((TObject*)nullptr, Form("#sigma_{Q_{i}} = %.3f mm", S_row_qi), "");
      if (IsFinite(chi2_row_qi) && IsFinite(ndf_row_qi) && ndf_row_qi > 0.0) {
        legRight.AddEntry((TObject*)nullptr,
                          Form("#chi^{2}/ndf (Q_{i}) = %.2f/%d", chi2_row_qi, static_cast<int>(std::lround(ndf_row_qi))),
                          "");
        if (IsFinite(prob_row_qi)) {
          legRight.AddEntry((TObject*)nullptr, Form("P(Q_{i}) = %.3g", prob_row_qi), "");
        }
      }
      legRight.AddEntry((TObject*)nullptr, Form("x_{true}-x_{rec}(Q_{i}) = %.1f #mum", 1000.0*(x_true - muRowQi)), "");
    }
    legRight.Draw();

    // Print page
    c.cd();
    c.Print(outputPdf.c_str());
    nPages++;
  }

  c.Print((outputPdf + "]").c_str());

  file->Close();
  delete file;

  ::Info("processing1D_replay_qifit", "Generated %lld pages (considered %lld events).", nPages, nConsidered);
  return 0;
}


// ROOT auto-exec wrappers so `root -l -b -q plotFitGaus1DReplayQiFit.C` works
int plotFitGaus1DReplayQiFit() {
  return processing1D_replay_qifit();
}

int plotFitGaus1DReplayQiFit(const char* filename, double errorPercentOfMax, bool useQiQnPercentErrors = false) {
  return processing1D_replay_qifit(filename, errorPercentOfMax, 300, true, true, useQiQnPercentErrors);
}

int plotFitGaus1DReplayQiFit(Long64_t nRandomEvents, bool useQiQnPercentErrors = false) {
  return processing1D_replay_qifit("../build/epicChargeSharing.root", 0.0, nRandomEvents, true, true, useQiQnPercentErrors);
}

int plotFitGaus1DReplayQiFit(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents, bool useQiQnPercentErrors = false) {
  return processing1D_replay_qifit(filename, errorPercentOfMax, nRandomEvents, true, true, useQiQnPercentErrors);
}

int plotFitGaus1DReplayQiFit(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents, bool plotQiOverlayPts, bool doQiFit, bool useQiQnPercentErrors = false) {
  return processing1D_replay_qifit(filename, errorPercentOfMax, nRandomEvents, plotQiOverlayPts, doQiFit, useQiQnPercentErrors);
}

int plotFitGaus1DReplayQiFit(const char* filename, double errorPercentOfMax, Long64_t nRandomEvents, bool plotQiOverlayPts, bool doQiFit, bool useQiQnPercentErrors, const char* outputPdfPath) {
  return processing1D_replay_qifit(filename, errorPercentOfMax, nRandomEvents, plotQiOverlayPts, doQiFit, useQiQnPercentErrors, outputPdfPath);
}
