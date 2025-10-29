// ROOT macro: FitGaus1D.C
// Performs 1D Gaussian fits on central row and column of the charge neighborhood
// using Q_f (noisy charge per pixel) to reconstruct (x_rec_2d, y_rec_2d) and
// deltas, and appends them as new branches. Falls back to Q_i if Q_f is absent.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TROOT.h>
#include <TError.h>
#include <TMath.h>
#include <Math/MinimizerOptions.h>
// Minuit2 least-squares API
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

// Fast least-squares fitter API (required for Fumili2)
#include <Fit/Fitter.h>
#include <Fit/BinData.h>
#include <Fit/Chi2FCN.h>
#include <Math/WrappedMultiTF1.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <atomic>
#include <span>
#include <ROOT/TThreadExecutor.hxx>

#include "ChargeUtils.h"

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

  struct FlatVectorStore {
    void Initialize(size_t nEntries, size_t reservePerEntry = 0) {
      offsets.assign(nEntries, -1);
      sizes.assign(nEntries, 0);
      values.clear();
      if (reservePerEntry > 0) {
        values.reserve(nEntries * reservePerEntry);
      }
    }

    void Store(size_t index, const std::vector<double>& src) {
      if (src.empty()) {
        return;
      }
      offsets[index] = static_cast<int>(values.size());
      sizes[index] = static_cast<int>(src.size());
      values.insert(values.end(), src.begin(), src.end());
    }

    [[nodiscard]] std::span<const double> Get(size_t index) const {
      const int offset = offsets[index];
      const int size = sizes[index];
      if (offset < 0 || size <= 0) {
        return {};
      }
      return {values.data() + offset, static_cast<size_t>(size)};
    }

    void ClearValues() {
      values.clear();
    }

    std::vector<double> values;
    std::vector<int> offsets;
    std::vector<int> sizes;
  };

  struct FitWorkBuffers {
    std::vector<double> x_row;
    std::vector<double> q_row;
    std::vector<double> err_row;
    std::vector<double> y_col;
    std::vector<double> q_col;
    std::vector<double> err_col;
    std::vector<double> s_d1;
    std::vector<double> q_d1;
    std::vector<double> err_d1;
    std::vector<double> s_d2;
    std::vector<double> q_d2;
    std::vector<double> err_d2;

    void PrepareRowCol(int n, bool needErrors) {
      EnsureCapacity(x_row, n);
      EnsureCapacity(q_row, n);
      EnsureCapacity(y_col, n);
      EnsureCapacity(q_col, n);
      if (needErrors) {
        EnsureCapacity(err_row, n);
        EnsureCapacity(err_col, n);
      } else {
        err_row.clear();
        err_col.clear();
      }
    }

    void PrepareDiag(int n, bool needErrors) {
      EnsureCapacity(s_d1, n);
      EnsureCapacity(q_d1, n);
      EnsureCapacity(s_d2, n);
      EnsureCapacity(q_d2, n);
      if (needErrors) {
        EnsureCapacity(err_d1, n);
        EnsureCapacity(err_d2, n);
      } else {
        err_d1.clear();
        err_d2.clear();
      }
    }

   private:
    static void EnsureCapacity(std::vector<double>& v, int n) {
      if (n <= 0) {
        v.clear();
        return;
      }
      if (static_cast<int>(v.capacity()) < n) {
        v.reserve(n);
      }
      v.clear();
    }
  };
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%)
// of the event's maximum charge within the neighborhood. The same error is
// applied to all data points used in the fits for that event.
int FitGaus1D(const char* filename = "../build/epicChargeSharing.root",
                 double errorPercentOfMax = 5.0,
                 bool saveParamA = true,
                 bool saveParamMu = true,
                 bool saveParamSigma = true,
                 bool saveParamB = true,
                 const char* chargeBranch = "Q_f",
                 bool fitDiagonals = false,
                 bool saveDiagParamA = true,
                 bool saveDiagParamMu = true,
                 bool saveDiagParamSigma = true,
                 bool saveDiagParamB = true,
                 bool saveLineMeans = true,
                 bool useQnQiPercentErrors = true) {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("FitGaus1D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first to provide clearer error when wrong file is passed)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("FitGaus1D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)", filename);
    file->Close();
    delete file;
    return 3;
  }

  // Fetch metadata (pixel spacing, pixel size, neighborhood radius) with fallbacks
  double pixelSpacing = NAN;
  if (auto* spacingObj = dynamic_cast<TNamed*>(file->Get("GridPixelSpacing_mm"))) {
    try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) {}
  }
  double pixelSize = NAN;
  if (auto* sizeObj = dynamic_cast<TNamed*>(file->Get("GridPixelSize_mm"))) {
    try { pixelSize = std::stod(sizeObj->GetTitle()); } catch (...) {}
  }
  int neighborhoodRadiusMeta = -1;
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

  auto inferRadiusFromTree = [&](TTree* t, const std::string& preferred) -> int {
    // Prefer requested branch; fall back to Q_f, F_i, Q_i
    std::vector<double>* Q_tmp = nullptr;
    auto bind = [&](const char* b)->bool { if (t->GetBranch(b)) { t->SetBranchStatus(b, 1); t->SetBranchAddress(b, &Q_tmp); return true; } return false; };
    if (!preferred.empty() && bind(preferred.c_str())) {}
    else if (bind("Q_f")) {}
    else if (bind("F_i")) {}
    else if (bind("Q_i")) {}
    else return -1;
    Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
    for (Long64_t i=0;i<nToScan;++i) {
      t->GetEntry(i);
      if (Q_tmp && !Q_tmp->empty()) {
        const size_t total = Q_tmp->size();
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
    ::Error("FitGaus1D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 2;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    // Fallback: if pixel size metadata is missing, use half of pitch as a conservative lower bound
    pixelSize = 0.5 * pixelSpacing;
  }
  
  // Decide which charge branch to use
  std::string chosenCharge = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Q_f");
  auto hasBranch = [&](const char* b){ return tree->GetBranch(b) != nullptr; };
  if (!hasBranch(chosenCharge.c_str())) {
    if (hasBranch("Q_f")) chosenCharge = "Q_f";
    else if (hasBranch("F_i")) chosenCharge = "F_i";
    else if (hasBranch("Q_i")) chosenCharge = "Q_i";
    else {
      ::Error("FitGaus1D", "No charge branch found (requested '%s'). Tried Q_f, F_i, Q_i.", chargeBranch ? chargeBranch : "<null>");
      file->Close();
      delete file;
      return 4;
    }
  }
  if (neighborhoodRadiusMeta <= 0) {
    neighborhoodRadiusMeta = inferRadiusFromTree(tree, chosenCharge);
  }

  // Existing branches (inputs)
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  // Use Q_f (noisy) for fits; fall back to Q_i if Q_f absent
  std::vector<double>* Q = nullptr; // used for fits (charges in Coulombs)
  std::vector<double>* Qi = nullptr; // initial charge (for error model)
  std::vector<double>* Qn = nullptr; // noiseless charge (for error model)
  bool enableQiQnErrors = useQnQiPercentErrors;
  bool haveQiBranchForErrors = false;
  bool haveQnBranchForErrors = false;

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("TrueX", 1);
  tree->SetBranchStatus("TrueY", 1);
  tree->SetBranchStatus("PixelX", 1);
  tree->SetBranchStatus("PixelY", 1);
  tree->SetBranchStatus("isPixelHit", 1);
  // Enable only the chosen charge branch
  tree->SetBranchStatus(chosenCharge.c_str(), 1);
  if (enableQiQnErrors) {
    haveQiBranchForErrors = tree->GetBranch("Q_i") != nullptr;
    haveQnBranchForErrors = tree->GetBranch("Q_n") != nullptr;
    if (haveQiBranchForErrors && haveQnBranchForErrors) {
      tree->SetBranchStatus("Q_i", 1);
      tree->SetBranchStatus("Q_n", 1);
    } else {
      ::Warning("FitGaus1D", "Requested Q_i/Q_n vertical errors but required branches are missing. Falling back to percent-of-max uncertainty.");
      enableQiQnErrors = false;
    }
  }

  tree->SetBranchAddress("TrueX", &x_hit);
  tree->SetBranchAddress("TrueY", &y_hit);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
  tree->SetBranchAddress(chosenCharge.c_str(), &Q);
  if (enableQiQnErrors && haveQiBranchForErrors) {
    tree->SetBranchAddress("Q_i", &Qi);
  }
  if (enableQiQnErrors && haveQnBranchForErrors) {
    tree->SetBranchAddress("Q_n", &Qn);
  }

  // New branches (outputs).
  // Use NaN sentinel so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double ReconRowX = INVALID_VALUE;
  double ReconColY = INVALID_VALUE;
  double ReconTrueDeltaRowX = INVALID_VALUE;
  double ReconTrueDeltaColY = INVALID_VALUE;

  // 1D Gaussian fit parameters (row=x, col=y)
  double GaussRowA = INVALID_VALUE;
  double GaussRowMu = INVALID_VALUE;
  double GaussRowSigma = INVALID_VALUE;
  double GaussRowB = INVALID_VALUE;
  double GaussRowChi2 = INVALID_VALUE;
  double GaussRowNdf = INVALID_VALUE;
  double GaussRowProb = INVALID_VALUE;
  double GaussColA = INVALID_VALUE;
  double GaussColMu = INVALID_VALUE;
  double GaussColSigma = INVALID_VALUE;
  double GaussColB = INVALID_VALUE;
  double GaussColChi2 = INVALID_VALUE;
  double GaussColNdf = INVALID_VALUE;
  double GaussColProb = INVALID_VALUE;

  // Optional diagonal reconstructions and parameters
  double ReconMDiagX = INVALID_VALUE;   // main diagonal (dj = di)
  double ReconMDiagY = INVALID_VALUE;
  double ReconSDiagX  = INVALID_VALUE;   // secondary diagonal (dj = -di)
  double ReconSDiagY  = INVALID_VALUE;
  double GaussMDiagA = INVALID_VALUE, GaussMDiagMu = INVALID_VALUE, GaussMDiagSigma = INVALID_VALUE, GaussMDiagB = INVALID_VALUE;
  double GaussSDiagA = INVALID_VALUE, GaussSDiagMu = INVALID_VALUE, GaussSDiagSigma = INVALID_VALUE, GaussSDiagB = INVALID_VALUE;
  double GaussMDiagChi2 = INVALID_VALUE, GaussMDiagNdf = INVALID_VALUE, GaussMDiagProb = INVALID_VALUE;
  double GaussSDiagChi2 = INVALID_VALUE, GaussSDiagNdf = INVALID_VALUE, GaussSDiagProb = INVALID_VALUE;
  // Signed deltas for diagonals
  double ReconTrueDeltaMDiagX = INVALID_VALUE;
  double ReconTrueDeltaMDiagY = INVALID_VALUE;
  double ReconTrueDeltaSDiagX = INVALID_VALUE;
  double ReconTrueDeltaSDiagY = INVALID_VALUE;
  // Mean-of-lines recon (optional)
  double ReconMeanX = INVALID_VALUE;
  double ReconMeanY = INVALID_VALUE;
  double ReconTrueDeltaMeanX = INVALID_VALUE;
  double ReconTrueDeltaMeanY = INVALID_VALUE;

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

  TBranch* br_x_rec = ensureAndResetBranch("ReconRowX", &ReconRowX);
  TBranch* br_y_rec = ensureAndResetBranch("ReconColY", &ReconColY);
  // Commented out per request: do not save absolute-value delta branches
  // TBranch* br_dx    = ensureAndResetBranch("ReconTrueDeltaX", &rec_hit_delta_x_2d);
  // TBranch* br_dy    = ensureAndResetBranch("ReconTrueDeltaY", &rec_hit_delta_y_2d);
  TBranch* br_dx_signed = ensureAndResetBranch("ReconTrueDeltaRowX", &ReconTrueDeltaRowX);
  TBranch* br_dy_signed = ensureAndResetBranch("ReconTrueDeltaColY", &ReconTrueDeltaColY);
  // Parameter branches
  TBranch* br_row_A = nullptr;
  TBranch* br_row_mu = nullptr;
  TBranch* br_row_sigma = nullptr;
  TBranch* br_row_B = nullptr;
  TBranch* br_row_chi2 = nullptr;
  TBranch* br_row_ndf = nullptr;
  TBranch* br_row_prob = nullptr;
  TBranch* br_col_A = nullptr;
  TBranch* br_col_mu = nullptr;
  TBranch* br_col_sigma = nullptr;
  TBranch* br_col_B = nullptr;
  TBranch* br_col_chi2 = nullptr;
  TBranch* br_col_ndf = nullptr;
  TBranch* br_col_prob = nullptr;
  // Diagonal parameter/output branches
  TBranch* br_x_rec_diag_main = nullptr;
  TBranch* br_y_rec_diag_main = nullptr;
  TBranch* br_x_rec_diag_sec  = nullptr;
  TBranch* br_y_rec_diag_sec  = nullptr;
  TBranch* br_d1_A = nullptr;
  TBranch* br_d1_mu = nullptr;
  TBranch* br_d1_sigma = nullptr;
  TBranch* br_d1_B = nullptr;
  TBranch* br_d1_chi2 = nullptr;
  TBranch* br_d1_ndf = nullptr;
  TBranch* br_d1_prob = nullptr;
  TBranch* br_d2_A = nullptr;
  TBranch* br_d2_mu = nullptr;
  TBranch* br_d2_sigma = nullptr;
  TBranch* br_d2_B = nullptr;
  TBranch* br_d2_chi2 = nullptr;
  TBranch* br_d2_ndf = nullptr;
  TBranch* br_d2_prob = nullptr;
  // Diagonal signed delta branches
  TBranch* br_mdiag_dx_signed = nullptr;
  TBranch* br_mdiag_dy_signed = nullptr;
  TBranch* br_sdiag_dx_signed = nullptr;
  TBranch* br_sdiag_dy_signed = nullptr;
  // Mean-of-lines branches
  TBranch* br_x_mean_lines = nullptr;
  TBranch* br_y_mean_lines = nullptr;
  TBranch* br_dx_mean_signed = nullptr;
  TBranch* br_dy_mean_signed = nullptr;
  if (saveParamA) {
    br_row_A     = ensureAndResetBranch("GaussRowA", &GaussRowA);
    br_col_A     = ensureAndResetBranch("GaussColA", &GaussColA);
  }
  if (saveParamMu) {
    br_row_mu    = ensureAndResetBranch("GaussRowMu", &GaussRowMu);
    br_col_mu    = ensureAndResetBranch("GaussColMu", &GaussColMu);
  }
  if (saveParamSigma) {
    br_row_sigma = ensureAndResetBranch("GaussRowSigma", &GaussRowSigma);
    br_col_sigma = ensureAndResetBranch("GaussColSigma", &GaussColSigma);
  }
  if (saveParamB) {
    br_row_B     = ensureAndResetBranch("GaussRowB", &GaussRowB);
    br_col_B     = ensureAndResetBranch("GaussColB", &GaussColB);
  }
  br_row_chi2 = ensureAndResetBranch("GaussRowChi2", &GaussRowChi2);
  br_row_ndf  = ensureAndResetBranch("GaussRowNdf", &GaussRowNdf);
  br_row_prob = ensureAndResetBranch("GaussRowProb", &GaussRowProb);
  br_col_chi2 = ensureAndResetBranch("GaussColChi2", &GaussColChi2);
  br_col_ndf  = ensureAndResetBranch("GaussColNdf", &GaussColNdf);
  br_col_prob = ensureAndResetBranch("GaussColProb", &GaussColProb);

  // Create diagonal branches if requested
  if (fitDiagonals) {
    br_x_rec_diag_main = ensureAndResetBranch("ReconMDiagX", &ReconMDiagX);
    br_y_rec_diag_main = ensureAndResetBranch("ReconMDiagY", &ReconMDiagY);
    br_x_rec_diag_sec  = ensureAndResetBranch("ReconSDiagX", &ReconSDiagX);
    br_y_rec_diag_sec  = ensureAndResetBranch("ReconSDiagY", &ReconSDiagY);
    br_mdiag_dx_signed = ensureAndResetBranch("ReconTrueDeltaMDiagX", &ReconTrueDeltaMDiagX);
    br_mdiag_dy_signed = ensureAndResetBranch("ReconTrueDeltaMDiagY", &ReconTrueDeltaMDiagY);
    br_sdiag_dx_signed = ensureAndResetBranch("ReconTrueDeltaSDiagX", &ReconTrueDeltaSDiagX);
    br_sdiag_dy_signed = ensureAndResetBranch("ReconTrueDeltaSDiagY", &ReconTrueDeltaSDiagY);
    if (saveLineMeans) {
      br_x_mean_lines = ensureAndResetBranch("ReconMeanX", &ReconMeanX);
      br_y_mean_lines = ensureAndResetBranch("ReconMeanY", &ReconMeanY);
      br_dx_mean_signed = ensureAndResetBranch("ReconTrueDeltaMeanX", &ReconTrueDeltaMeanX);
      br_dy_mean_signed = ensureAndResetBranch("ReconTrueDeltaMeanY", &ReconTrueDeltaMeanY);
    }
    if (saveDiagParamA) {
      br_d1_A = ensureAndResetBranch("GaussMDiagA", &GaussMDiagA);
      br_d2_A = ensureAndResetBranch("GaussSDiagA", &GaussSDiagA);
    }
    if (saveDiagParamMu) {
      br_d1_mu = ensureAndResetBranch("GaussMDiagMu", &GaussMDiagMu);
      br_d2_mu = ensureAndResetBranch("GaussSDiagMu", &GaussSDiagMu);
    }
    if (saveDiagParamSigma) {
      br_d1_sigma = ensureAndResetBranch("GaussMDiagSigma", &GaussMDiagSigma);
      br_d2_sigma = ensureAndResetBranch("GaussSDiagSigma", &GaussSDiagSigma);
    }
    if (saveDiagParamB) {
      br_d1_B = ensureAndResetBranch("GaussMDiagB", &GaussMDiagB);
      br_d2_B = ensureAndResetBranch("GaussSDiagB", &GaussSDiagB);
    }
    br_d1_chi2 = ensureAndResetBranch("GaussMDiagChi2", &GaussMDiagChi2);
    br_d1_ndf  = ensureAndResetBranch("GaussMDiagNdf", &GaussMDiagNdf);
    br_d1_prob = ensureAndResetBranch("GaussMDiagProb", &GaussMDiagProb);
    br_d2_chi2 = ensureAndResetBranch("GaussSDiagChi2", &GaussSDiagChi2);
    br_d2_ndf  = ensureAndResetBranch("GaussSDiagNdf", &GaussSDiagNdf);
    br_d2_prob = ensureAndResetBranch("GaussSDiagProb", &GaussSDiagProb);
  }

  // Fitting function for 1D gaussian + const (locals created per-fit below)

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0;
  std::atomic<long long> nFitted{0};

  // Preload inputs sequentially to avoid ROOT I/O races while keeping memory contiguous
  std::vector<double> v_x_hit(nEntries), v_y_hit(nEntries);
  std::vector<double> v_x_px(nEntries), v_y_px(nEntries);
  std::vector<char> v_is_pixel(nEntries);
  std::vector<int> v_gridDim(nEntries, 0);
  std::vector<char> v_hasQiQn(nEntries, 0);

  const int approxNeighborSide = (neighborhoodRadiusMeta > 0)
                                   ? (2 * neighborhoodRadiusMeta + 1)
                                   : 5;
  const size_t approxNeighborSize = static_cast<size_t>(approxNeighborSide) * approxNeighborSide;

  FlatVectorStore chargeStore;
  chargeStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSize);
  FlatVectorStore qiStore;
  FlatVectorStore qnStore;
  if (enableQiQnErrors) {
    qiStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSize);
    qnStore.Initialize(static_cast<size_t>(nEntries), approxNeighborSize);
  }

  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    v_x_hit[i] = x_hit;
    v_y_hit[i] = y_hit;
    v_x_px[i]  = x_px;
    v_y_px[i]  = y_px;
    const bool pixelEvent = is_pixel_hit ? true : false;
    v_is_pixel[i] = pixelEvent ? 1 : 0;
    if (pixelEvent || !Q || Q->empty()) {
      continue;
    }

    chargeStore.Store(static_cast<size_t>(i), *Q);
    const int total = static_cast<int>(Q->size());
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N >= 3 && N * N == total) {
      v_gridDim[i] = N;
    } else {
      v_gridDim[i] = 0;
    }

    if (!enableQiQnErrors || !Qi || !Qn) {
      continue;
    }
    if (Qi->size() != Q->size() || Qn->size() != Q->size() || Qi->empty() || Qn->empty()) {
      continue;
    }
    qiStore.Store(static_cast<size_t>(i), *Qi);
    qnStore.Store(static_cast<size_t>(i), *Qn);
    if (v_gridDim[i] > 0) {
      v_hasQiQn[i] = 1;
    }
  }

  // Prepare output buffers
  std::vector<double> out_x_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_y_rec(nEntries, INVALID_VALUE);
  std::vector<double> out_dx_s(nEntries, INVALID_VALUE);
  std::vector<double> out_dy_s(nEntries, INVALID_VALUE);
  // Output buffers for fit parameters
  std::vector<double> out_row_A(nEntries, INVALID_VALUE);
  std::vector<double> out_row_mu(nEntries, INVALID_VALUE);
  std::vector<double> out_row_sigma(nEntries, INVALID_VALUE);
  std::vector<double> out_row_B(nEntries, INVALID_VALUE);
  std::vector<double> out_row_chi2(nEntries, INVALID_VALUE);
  std::vector<double> out_row_ndf(nEntries, INVALID_VALUE);
  std::vector<double> out_row_prob(nEntries, INVALID_VALUE);
  std::vector<double> out_col_A(nEntries, INVALID_VALUE);
  std::vector<double> out_col_mu(nEntries, INVALID_VALUE);
  std::vector<double> out_col_sigma(nEntries, INVALID_VALUE);
  std::vector<double> out_col_B(nEntries, INVALID_VALUE);
  std::vector<double> out_col_chi2(nEntries, INVALID_VALUE);
  std::vector<double> out_col_ndf(nEntries, INVALID_VALUE);
  std::vector<double> out_col_prob(nEntries, INVALID_VALUE);

  // Diagonal buffers
  std::vector<double> out_x_rec_diag_main, out_y_rec_diag_main, out_x_rec_diag_sec, out_y_rec_diag_sec;
  std::vector<double> out_d1_A, out_d1_mu, out_d1_sigma, out_d1_B, out_d1_chi2, out_d1_ndf, out_d1_prob;
  std::vector<double> out_d2_A, out_d2_mu, out_d2_sigma, out_d2_B, out_d2_chi2, out_d2_ndf, out_d2_prob;
  std::vector<double> out_mdiag_dx_s, out_mdiag_dy_s, out_sdiag_dx_s, out_sdiag_dy_s;
  // Mean-of-lines buffers
  std::vector<double> out_x_mean_lines, out_y_mean_lines, out_dx_mean_s, out_dy_mean_s;
  if (fitDiagonals) {
    out_x_rec_diag_main.assign(nEntries, INVALID_VALUE);
    out_y_rec_diag_main.assign(nEntries, INVALID_VALUE);
    out_x_rec_diag_sec.assign(nEntries, INVALID_VALUE);
    out_y_rec_diag_sec.assign(nEntries, INVALID_VALUE);
    out_d1_A.assign(nEntries, INVALID_VALUE);
    out_d1_mu.assign(nEntries, INVALID_VALUE);
    out_d1_sigma.assign(nEntries, INVALID_VALUE);
    out_d1_B.assign(nEntries, INVALID_VALUE);
    out_d1_chi2.assign(nEntries, INVALID_VALUE);
    out_d1_ndf.assign(nEntries, INVALID_VALUE);
    out_d1_prob.assign(nEntries, INVALID_VALUE);
    out_d2_A.assign(nEntries, INVALID_VALUE);
    out_d2_mu.assign(nEntries, INVALID_VALUE);
    out_d2_sigma.assign(nEntries, INVALID_VALUE);
    out_d2_B.assign(nEntries, INVALID_VALUE);
    out_d2_chi2.assign(nEntries, INVALID_VALUE);
    out_d2_ndf.assign(nEntries, INVALID_VALUE);
    out_d2_prob.assign(nEntries, INVALID_VALUE);
    out_mdiag_dx_s.assign(nEntries, INVALID_VALUE);
    out_mdiag_dy_s.assign(nEntries, INVALID_VALUE);
    out_sdiag_dx_s.assign(nEntries, INVALID_VALUE);
    out_sdiag_dy_s.assign(nEntries, INVALID_VALUE);
    if (saveLineMeans) {
      out_x_mean_lines.assign(nEntries, INVALID_VALUE);
      out_y_mean_lines.assign(nEntries, INVALID_VALUE);
      out_dx_mean_s.assign(nEntries, INVALID_VALUE);
      out_dy_mean_s.assign(nEntries, INVALID_VALUE);
    }
  }

  // Parallel computation over entries
  std::vector<int> indices(nEntries);
  std::iota(indices.begin(), indices.end(), 0);
  ROOT::TThreadExecutor exec; // uses ROOT IMT pool size by default
  // Suppress expected Minuit2 error spam during Fumili2 attempts; we'll fallback to MIGRAD if needed
  const int prevErrorLevel_FitGaus1D = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  exec.Foreach([&](int i){
    if (v_is_pixel[i] != 0) {
      return;
    }

    const auto QLoc = chargeStore.Get(static_cast<size_t>(i));
    if (QLoc.empty()) {
      return;
    }
    const int N = v_gridDim[i];
    if (N < 3 || static_cast<size_t>(N * N) != QLoc.size()) {
      return;
    }
    const int R = (N - 1) / 2;
    bool haveQiQnForEvent = enableQiQnErrors && v_hasQiQn[i] != 0;
    std::span<const double> QiLoc;
    std::span<const double> QnLoc;
    if (haveQiQnForEvent) {
      QiLoc = qiStore.Get(static_cast<size_t>(i));
      QnLoc = qnStore.Get(static_cast<size_t>(i));
      if (QiLoc.size() != QLoc.size() || QnLoc.size() != QLoc.size()) {
        haveQiQnForEvent = false;
      }
    }

    static thread_local FitWorkBuffers fitBuffers;
    fitBuffers.PrepareRowCol(N, haveQiQnForEvent);
    std::vector<double>& x_row = fitBuffers.x_row;
    std::vector<double>& q_row = fitBuffers.q_row;
    std::vector<double>& y_col = fitBuffers.y_col;
    std::vector<double>& q_col = fitBuffers.q_col;
    std::vector<double>& err_row = fitBuffers.err_row;
    std::vector<double>& err_col = fitBuffers.err_col;
    std::vector<double>* s_d1_ptr = nullptr;
    std::vector<double>* q_d1_ptr = nullptr;
    std::vector<double>* err_d1_ptr = nullptr;
    std::vector<double>* s_d2_ptr = nullptr;
    std::vector<double>* q_d2_ptr = nullptr;
    std::vector<double>* err_d2_ptr = nullptr;
    if (fitDiagonals) {
      fitBuffers.PrepareDiag(N, haveQiQnForEvent);
      s_d1_ptr = &fitBuffers.s_d1;
      q_d1_ptr = &fitBuffers.q_d1;
      err_d1_ptr = &fitBuffers.err_d1;
      s_d2_ptr = &fitBuffers.s_d2;
      q_d2_ptr = &fitBuffers.q_d2;
      err_d2_ptr = &fitBuffers.err_d2;
    }
    double qmaxNeighborhood = -1e300;
    double qmaxQiNeighborhood = -1e300;
    const double x_px_loc = v_x_px[i];
    const double y_px_loc = v_y_px[i];
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx  = (di + R) * N + (dj + R);
        const double q = QLoc[idx];
        if (!IsFinite(q) || q < 0) continue;
        double qiVal = 0.0;
        double errCandidate = std::numeric_limits<double>::quiet_NaN();
        if (haveQiQnForEvent) {
          qiVal = QiLoc[idx];
          if (std::isfinite(qiVal) && qiVal > qmaxQiNeighborhood) {
            qmaxQiNeighborhood = qiVal;
          }
          errCandidate = ComputeQnQiPercent(qiVal, QnLoc[idx], qmaxQiNeighborhood);
        }
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        if (dj == 0) {
          const double x = x_px_loc + di * pixelSpacing;
          x_row.push_back(x);
          q_row.push_back(q);
          if (haveQiQnForEvent) {
            err_row.push_back(errCandidate);
          }
        }
        if (di == 0) {
          const double y = y_px_loc + dj * pixelSpacing;
          y_col.push_back(y);
          q_col.push_back(q);
          if (haveQiQnForEvent) {
            err_col.push_back(errCandidate);
          }
        }
        if (fitDiagonals) {
          if (s_d1_ptr && di == dj) {
            const double s = x_px_loc + di * pixelSpacing;
            s_d1_ptr->push_back(s);
            q_d1_ptr->push_back(q);
            if (haveQiQnForEvent) {
              err_d1_ptr->push_back(errCandidate);
            }
          }
          if (s_d2_ptr && di == -dj) {
            const double s = x_px_loc + di * pixelSpacing;
            s_d2_ptr->push_back(s);
            q_d2_ptr->push_back(q);
            if (haveQiQnForEvent) {
              err_d2_ptr->push_back(errCandidate);
            }
          }
        }
      }
    }
    if (x_row.size() < 3 || y_col.size() < 3) {
      return;
    }

    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;
    auto selectError = [&](double candidate) -> double {
      if (std::isfinite(candidate) && candidate > 0.0) return candidate;
      if (uniformSigma > 0.0) return uniformSigma;
      return 1.0;
    };

    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    // Allow negative baseline seed if the data suggest it
    double B0_row = *minmaxRow.first;
    double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    double B0_col = *minmaxCol.first;

    // Low-contrast: fast centroid (relative to neighborhood max charge)
    const double contrastEps = (qmaxNeighborhood > 0.0) ? (1e-3 * qmaxNeighborhood) : 0.0;
    if (qmaxNeighborhood > 0.0 && A0_row < contrastEps && A0_col < contrastEps) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k] - B0_row); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k] - B0_col); wsumy += w; yw += w * y_col[k]; }
      if (wsumx > 0 && wsumy > 0) {
        const double xr = xw / wsumx;
        const double yr = yw / wsumy;
        out_x_rec[i] = xr;
        out_y_rec[i] = yr;
        out_dx_s[i] = (v_x_hit[i] - xr);
        out_dy_s[i] = (v_y_hit[i] - yr);
        nFitted.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }

    // Seeds
    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    // Constrain sigma to be within [pixel size, radius * pitch]
    const double sigLoBound = pixelSize;
    const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
    auto sigmaSeed1D = [&](const std::vector<double>& xs, const std::vector<double>& qs, double B0)->double {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); wsum += w; xw += w * xs[k]; }
      if (wsum <= 0.0) {
        double s = std::max(0.25*pixelSpacing, 1e-6);
        if (s < sigLoBound) s = sigLoBound;
        if (s > sigHiBound) s = sigHiBound;
        return s;
      }
      const double mean = xw / wsum;
      double var = 0.0;
      for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0); const double dx = xs[k] - mean; var += w * dx * dx; }
      var = (wsum > 0.0) ? (var / wsum) : 0.0;
      double s = std::sqrt(std::max(var, 1e-12));
      if (s < sigLoBound) s = sigLoBound;
      if (s > sigHiBound) s = sigHiBound;
      return s;
    };
    double sigInitRow = sigmaSeed1D(x_row, q_row, B0_row);
    double sigInitCol = sigmaSeed1D(y_col, q_col, B0_col);

    // sigInitRow/sigInitCol were computed above on unfiltered data. They are possibly
    // updated after clipping in the block above.

    TF1 fRowLoc("fRowLoc", GaussPlusB, -1e9, 1e9, 4);
    TF1 fColLoc("fColLoc", GaussPlusB, -1e9, 1e9, 4);
    fRowLoc.SetParameters(A0_row, mu0_row, sigInitRow, B0_row);
    fColLoc.SetParameters(A0_col, mu0_col, sigInitCol, B0_col);

    auto minmaxX = std::minmax_element(x_row.begin(), x_row.end());
    auto minmaxY = std::minmax_element(y_col.begin(), y_col.end());
    const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
    const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
    const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
    const double yMax = *minmaxY.second + 0.5 * pixelSpacing;
    fRowLoc.SetRange(xMin, xMax);
    fColLoc.SetRange(yMin, yMax);
    // Allow mean to cross pixel boundaries to avoid one-sided residuals
    const double muXLo = x_px_loc - 1.0 * pixelSpacing;
    const double muXHi = x_px_loc + 1.0 * pixelSpacing;
    const double muYLo = y_px_loc - 1.0 * pixelSpacing;
    const double muYHi = y_px_loc + 1.0 * pixelSpacing;
    fRowLoc.SetParLimits(1, muXLo, muXHi);
    fColLoc.SetParLimits(1, muYLo, muYHi);
    // Bounds for Q_i fits: A in (0, ~2*qmax], B in [-~qmax, ~qmax]
    const double AHi = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
    const double BHi = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
    fRowLoc.SetParLimits(0, 1e-18, AHi);
    fRowLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fRowLoc.SetParLimits(3, -BHi, BHi);
    fColLoc.SetParLimits(0, 1e-18, AHi);
    fColLoc.SetParLimits(2, sigLoBound, sigHiBound);
    fColLoc.SetParLimits(3, -BHi, BHi);

    ROOT::Math::WrappedMultiTF1 wRow(fRowLoc, 1);
    ROOT::Math::WrappedMultiTF1 wCol(fColLoc, 1);
    ROOT::Fit::BinData dataRow(static_cast<int>(x_row.size()), 1);
    ROOT::Fit::BinData dataCol(static_cast<int>(y_col.size()), 1);
    for (size_t k = 0; k < x_row.size(); ++k) {
      const double candidate = (haveQiQnForEvent && k < err_row.size()) ? err_row[k] : std::numeric_limits<double>::quiet_NaN();
      const double ey = selectError(candidate);
      dataRow.Add(x_row[k], q_row[k], ey);
    }
    for (size_t k = 0; k < y_col.size(); ++k) {
      const double candidate = (haveQiQnForEvent && k < err_col.size()) ? err_col[k] : std::numeric_limits<double>::quiet_NaN();
      const double ey = selectError(candidate);
      dataCol.Add(y_col[k], q_col[k], ey);
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
    // A in (0, ~2*qmax]
    fitRow.Config().ParSettings(0).SetLimits(1e-18, AHi);
    fitRow.Config().ParSettings(1).SetLimits(muXLo, muXHi);
    fitRow.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    // B in [-~qmax, ~qmax]
    fitRow.Config().ParSettings(3).SetLimits(-BHi, BHi);
    // A in (0, ~2*qmax]
    fitCol.Config().ParSettings(0).SetLimits(1e-18, AHi);
    fitCol.Config().ParSettings(1).SetLimits(muYLo, muYHi);
    fitCol.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
    // B in [-~qmax, ~qmax]
    fitCol.Config().ParSettings(3).SetLimits(-BHi, BHi);
    const double stepA_row = std::max(1e-18, 0.01 * A0_row);
    const double stepA_col = std::max(1e-18, 0.01 * A0_col);
    const double stepB_row = std::max(1e-18, 0.01 * std::max(std::abs(B0_row), A0_row));
    const double stepB_col = std::max(1e-18, 0.01 * std::max(std::abs(B0_col), A0_col));
    fitRow.Config().ParSettings(0).SetStepSize(stepA_row);
    fitRow.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitRow.Config().ParSettings(3).SetStepSize(stepB_row);
    fitCol.Config().ParSettings(0).SetStepSize(stepA_col);
    fitCol.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitCol.Config().ParSettings(3).SetStepSize(stepB_col);
    fitRow.Config().ParSettings(0).SetValue(A0_row);
    fitRow.Config().ParSettings(1).SetValue(mu0_row);
    fitRow.Config().ParSettings(2).SetValue(sigInitRow);
    fitRow.Config().ParSettings(3).SetValue(B0_row);
    fitCol.Config().ParSettings(0).SetValue(A0_col);
    fitCol.Config().ParSettings(1).SetValue(mu0_col);
    fitCol.Config().ParSettings(2).SetValue(sigInitCol);
    fitCol.Config().ParSettings(3).SetValue(B0_col);

    bool okRowFit = fitRow.Fit(dataRow);
    bool okColFit = fitCol.Fit(dataCol);
    // If Fumili2 fails, retry once with MIGRAD which is more robust
    if (!okRowFit) {
      fitRow.Config().SetMinimizer("Minuit2", "Migrad");
      fitRow.Config().MinimizerOptions().SetStrategy(1);
      fitRow.Config().MinimizerOptions().SetTolerance(1e-3);
      fitRow.Config().MinimizerOptions().SetPrintLevel(0);
      okRowFit = fitRow.Fit(dataRow);
    }
    if (!okColFit) {
      fitCol.Config().SetMinimizer("Minuit2", "Migrad");
      fitCol.Config().MinimizerOptions().SetStrategy(1);
      fitCol.Config().MinimizerOptions().SetTolerance(1e-3);
      fitCol.Config().MinimizerOptions().SetPrintLevel(0);
      okColFit = fitCol.Fit(dataCol);
    }
    // Save parameters only if the corresponding fit converged
    if (okRowFit) {
      out_row_A[i]     = fitRow.Result().Parameter(0);
      out_row_mu[i]    = fitRow.Result().Parameter(1);
      out_row_sigma[i] = fitRow.Result().Parameter(2);
      out_row_B[i]     = fitRow.Result().Parameter(3);
      out_row_chi2[i]  = fitRow.Result().Chi2();
      out_row_ndf[i]   = fitRow.Result().Ndf();
      out_row_prob[i]  = (fitRow.Result().Ndf() > 0) ? TMath::Prob(fitRow.Result().Chi2(), fitRow.Result().Ndf()) : INVALID_VALUE;
    }
    if (okColFit) {
      out_col_A[i]     = fitCol.Result().Parameter(0);
      out_col_mu[i]    = fitCol.Result().Parameter(1);
      out_col_sigma[i] = fitCol.Result().Parameter(2);
      out_col_B[i]     = fitCol.Result().Parameter(3);
      out_col_chi2[i]  = fitCol.Result().Chi2();
      out_col_ndf[i]   = fitCol.Result().Ndf();
      out_col_prob[i]  = (fitCol.Result().Ndf() > 0) ? TMath::Prob(fitCol.Result().Chi2(), fitCol.Result().Ndf()) : INVALID_VALUE;
    }
    double muX = NAN, muY = NAN;
    if (okRowFit) muX = fitRow.Result().Parameter(1);
    if (okColFit) muY = fitCol.Result().Parameter(1);
    if (!okRowFit) {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k] - B0_row); wsum += w; xw += w * x_row[k]; }
      if (wsum > 0) { muX = xw / wsum; }
    }
    if (!okColFit) {
      double wsum = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k] - B0_col); wsum += w; yw += w * y_col[k]; }
      if (wsum > 0) { muY = yw / wsum; }
    }
    const bool okRow = IsFinite(muX);
    const bool okCol = IsFinite(muY);
    if (okRow && okCol) {
      out_x_rec[i] = muX;
      out_y_rec[i] = muY;
      out_dx_s[i] = (v_x_hit[i] - muX);
      out_dy_s[i] = (v_y_hit[i] - muY);
      nFitted.fetch_add(1, std::memory_order_relaxed);
    }
    // =========================
    // Diagonal fits (optional)
    // =========================
    if (fitDiagonals) {
      auto& s_d1_vec = fitBuffers.s_d1;
      auto& q_d1_vec = fitBuffers.q_d1;
      auto& err_d1_vec = fitBuffers.err_d1;
      auto& s_d2_vec = fitBuffers.s_d2;
      auto& q_d2_vec = fitBuffers.q_d2;
      auto& err_d2_vec = fitBuffers.err_d2;
      auto fitDiag = [&](const std::vector<double>& s_vals_in,
                         const std::vector<double>& q_vals_in,
                         const std::vector<double>* err_vals_in,
                         double muLo, double muHi,
                         double& outA, double& outMu, double& outSig, double& outB,
                         double& outChi2, double& outNdf, double& outProb) -> bool {
        if (s_vals_in.size() < 3) return false;
        // Seeds from min/max
        auto mm = std::minmax_element(q_vals_in.begin(), q_vals_in.end());
        double A0 = std::max(1e-18, *mm.second - *mm.first);
        double B0 = *mm.first; // allow negative baseline seed
        int idxMax = std::distance(q_vals_in.begin(), std::max_element(q_vals_in.begin(), q_vals_in.end()));
        double mu0 = s_vals_in[idxMax];
        // Sigma seed and bounds like rows/cols
        const double sigLoBound = pixelSize;
        const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
        auto sigmaSeed1D_local = [&](const std::vector<double>& xs, const std::vector<double>& qs, double B0L)->double {
          double wsum = 0.0, xw = 0.0;
          for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0L); wsum += w; xw += w * xs[k]; }
          if (wsum <= 0.0) {
            double s = std::max(0.25*pixelSpacing, 1e-6);
            if (s < sigLoBound) s = sigLoBound;
            if (s > sigHiBound) s = sigHiBound;
            return s;
          }
          const double mean = xw / wsum;
          double var = 0.0;
          for (size_t k=0;k<xs.size();++k) { double w = std::max(0.0, qs[k] - B0L); const double dx = xs[k] - mean; var += w * dx * dx; }
          var = (wsum > 0.0) ? (var / wsum) : 0.0;
          double s = std::sqrt(std::max(var, 1e-12));
          if (s < sigLoBound) s = sigLoBound;
          if (s > sigHiBound) s = sigHiBound;
          return s;
        };
        double sigInit = sigmaSeed1D_local(s_vals_in, q_vals_in, B0);
        // Prepare wrapped TF1
        TF1 fLoc("fDiagLoc", GaussPlusB, -1e9, 1e9, 4);
        fLoc.SetParameters(A0, mu0, sigInit, B0);
        ROOT::Math::WrappedMultiTF1 wLoc(fLoc, 1);
        ROOT::Fit::BinData data(static_cast<int>(s_vals_in.size()), 1);
        for (size_t k=0;k<s_vals_in.size();++k) {
          const double candidate = (err_vals_in && k < err_vals_in->size()) ? (*err_vals_in)[k] : std::numeric_limits<double>::quiet_NaN();
          const double ey = selectError(candidate);
          data.Add(s_vals_in[k], q_vals_in[k], ey);
        }
        ROOT::Fit::Fitter fitter;
        fitter.Config().SetMinimizer("Minuit2", "Fumili2");
        fitter.Config().MinimizerOptions().SetStrategy(0);
        fitter.Config().MinimizerOptions().SetTolerance(1e-4);
        fitter.Config().MinimizerOptions().SetPrintLevel(0);
        fitter.SetFunction(wLoc);
        const double AHiL = std::max(1e-18, 2.0 * std::max(qmaxNeighborhood, 0.0));
        const double BHiL = std::max(1e-18, 1.0 * std::max(qmaxNeighborhood, 0.0));
        fitter.Config().ParSettings(0).SetLimits(1e-18, AHiL);
        fitter.Config().ParSettings(1).SetLimits(muLo, muHi);
        fitter.Config().ParSettings(2).SetLimits(sigLoBound, sigHiBound);
        fitter.Config().ParSettings(3).SetLimits(-BHiL, BHiL);
        fitter.Config().ParSettings(0).SetStepSize(std::max(1e-18, 0.01 * A0));
        fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
        fitter.Config().ParSettings(3).SetStepSize(std::max(1e-18, 0.01 * std::max(std::abs(B0), A0)));
        fitter.Config().ParSettings(0).SetValue(A0);
        fitter.Config().ParSettings(1).SetValue(mu0);
        fitter.Config().ParSettings(2).SetValue(sigInit);
        fitter.Config().ParSettings(3).SetValue(B0);
        bool ok = fitter.Fit(data);
        if (!ok) {
          fitter.Config().SetMinimizer("Minuit2", "Migrad");
          fitter.Config().MinimizerOptions().SetStrategy(1);
          fitter.Config().MinimizerOptions().SetTolerance(1e-3);
          fitter.Config().MinimizerOptions().SetPrintLevel(0);
          ok = fitter.Fit(data);
        }
        if (ok) {
          outA = fitter.Result().Parameter(0);
          outMu = fitter.Result().Parameter(1);
          outSig = fitter.Result().Parameter(2);
          outB = fitter.Result().Parameter(3);
          outChi2 = fitter.Result().Chi2();
          outNdf  = fitter.Result().Ndf();
          outProb = (fitter.Result().Ndf() > 0) ? TMath::Prob(fitter.Result().Chi2(), fitter.Result().Ndf()) : INVALID_VALUE;
          return true;
        }
        // Fallback: baseline-subtracted weighted centroid
        double wsum = 0.0, sw = 0.0;
        for (size_t k=0;k<s_vals_in.size();++k) { double w = std::max(0.0, q_vals_in[k] - B0); wsum += w; sw += w * s_vals_in[k]; }
        if (wsum > 0) {
          outA = A0; outB = B0; outSig = sigInit; outMu = sw / wsum;
          outChi2 = INVALID_VALUE;
          outNdf = INVALID_VALUE;
          outProb = INVALID_VALUE;
          return true;
        }
        return false;
      };

      const double muLo = x_px_loc - 1.0 * pixelSpacing;
      const double muHi = x_px_loc + 1.0 * pixelSpacing;
      double A1=INVALID_VALUE, mu1=INVALID_VALUE, S1=INVALID_VALUE, B1=INVALID_VALUE;
      double A2=INVALID_VALUE, mu2=INVALID_VALUE, S2=INVALID_VALUE, B2=INVALID_VALUE;
      const std::vector<double>* err_d1_fit = (haveQiQnForEvent && !err_d1_vec.empty()) ? &err_d1_vec : nullptr;
      const std::vector<double>* err_d2_fit = (haveQiQnForEvent && !err_d2_vec.empty()) ? &err_d2_vec : nullptr;
      double chi2_1 = INVALID_VALUE, ndf_1 = INVALID_VALUE, prob_1 = INVALID_VALUE;
      double chi2_2 = INVALID_VALUE, ndf_2 = INVALID_VALUE, prob_2 = INVALID_VALUE;
      bool ok1 = fitDiag(s_d1_vec, q_d1_vec, err_d1_fit, muLo, muHi, A1, mu1, S1, B1, chi2_1, ndf_1, prob_1);
      bool ok2 = fitDiag(s_d2_vec, q_d2_vec, err_d2_fit, muLo, muHi, A2, mu2, S2, B2, chi2_2, ndf_2, prob_2);
      if (ok1) {
        // Transform diagonal coordinate mu -> (x,y)
        const double dx = mu1 - x_px_loc;
        out_x_rec_diag_main[i] = mu1;
        out_y_rec_diag_main[i] = y_px_loc + dx; // main diag: dj = di
        out_mdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_main[i]);
        out_mdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_main[i]);
        if (saveDiagParamA) out_d1_A[i] = A1; if (saveDiagParamMu) out_d1_mu[i] = mu1; if (saveDiagParamSigma) out_d1_sigma[i] = S1; if (saveDiagParamB) out_d1_B[i] = B1;
        out_d1_chi2[i] = chi2_1;
        out_d1_ndf[i]  = ndf_1;
        out_d1_prob[i] = prob_1;
      }
      if (ok2) {
        const double dx = mu2 - x_px_loc;
        out_x_rec_diag_sec[i] = mu2;
        out_y_rec_diag_sec[i] = y_px_loc - dx; // secondary diag: dj = -di
        out_sdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_sec[i]);
        out_sdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_sec[i]);
        if (saveDiagParamA) out_d2_A[i] = A2; if (saveDiagParamMu) out_d2_mu[i] = mu2; if (saveDiagParamSigma) out_d2_sigma[i] = S2; if (saveDiagParamB) out_d2_B[i] = B2;
        out_d2_chi2[i] = chi2_2;
        out_d2_ndf[i]  = ndf_2;
        out_d2_prob[i] = prob_2;
      }
      // Compute mean-of-lines recon (row/diagonals for X, col/diagonals for Y)
      if (saveLineMeans) {
        int cntX = 0;
        double accX = 0.0;
        if (IsFinite(muX)) { accX += muX; cntX++; }
        if (ok1) { accX += mu1; cntX++; }
        if (ok2) { accX += mu2; cntX++; }
        if (cntX > 0) {
          out_x_mean_lines[i] = accX / cntX;
          out_dx_mean_s[i] = v_x_hit[i] - out_x_mean_lines[i];
        }

        int cntY = 0;
        double accY = 0.0;
        if (IsFinite(muY)) { accY += muY; cntY++; }
        if (ok1) { double dy1 = (mu1 - x_px_loc); accY += (y_px_loc + dy1); cntY++; }
        if (ok2) { double dy2 = (mu2 - x_px_loc); accY += (y_px_loc - dy2); cntY++; }
        if (cntY > 0) {
          out_y_mean_lines[i] = accY / cntY;
          out_dy_mean_s[i] = v_y_hit[i] - out_y_mean_lines[i];
        }
      }
    }
  }, indices);
  // Restore previous error level
  gErrorIgnoreLevel = prevErrorLevel_FitGaus1D;

  // Avoid re-reading original data while filling new branches
  tree->SetBranchStatus("*", 0);
  // Sequentially write outputs to the tree (thread-safe)
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i); // ensure correct entry numbering for branch fill
    ReconRowX = out_x_rec[i];
    ReconColY = out_y_rec[i];
    ReconTrueDeltaRowX = out_dx_s[i];
    ReconTrueDeltaColY = out_dy_s[i];
    GaussRowA = out_row_A[i];
    GaussRowMu = out_row_mu[i];
    GaussRowSigma = out_row_sigma[i];
    GaussRowB = out_row_B[i];
    GaussRowChi2 = out_row_chi2[i];
    GaussRowNdf = out_row_ndf[i];
    GaussRowProb = out_row_prob[i];
    GaussColA = out_col_A[i];
    GaussColMu = out_col_mu[i];
    GaussColSigma = out_col_sigma[i];
    GaussColB = out_col_B[i];
    GaussColChi2 = out_col_chi2[i];
    GaussColNdf = out_col_ndf[i];
    GaussColProb = out_col_prob[i];
    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx_signed->Fill();
    br_dy_signed->Fill();
    if (br_row_A) br_row_A->Fill();
    if (br_row_mu) br_row_mu->Fill();
    if (br_row_sigma) br_row_sigma->Fill();
    if (br_row_B) br_row_B->Fill();
    if (br_row_chi2) br_row_chi2->Fill();
    if (br_row_ndf) br_row_ndf->Fill();
    if (br_row_prob) br_row_prob->Fill();
    if (br_col_A) br_col_A->Fill();
    if (br_col_mu) br_col_mu->Fill();
    if (br_col_sigma) br_col_sigma->Fill();
    if (br_col_B) br_col_B->Fill();
    if (br_col_chi2) br_col_chi2->Fill();
    if (br_col_ndf) br_col_ndf->Fill();
    if (br_col_prob) br_col_prob->Fill();
    if (fitDiagonals) {
      ReconMDiagX = out_x_rec_diag_main.empty() ? INVALID_VALUE : out_x_rec_diag_main[i];
      ReconMDiagY = out_y_rec_diag_main.empty() ? INVALID_VALUE : out_y_rec_diag_main[i];
      ReconSDiagX  = out_x_rec_diag_sec.empty()  ? INVALID_VALUE : out_x_rec_diag_sec[i];
      ReconSDiagY  = out_y_rec_diag_sec.empty()  ? INVALID_VALUE : out_y_rec_diag_sec[i];
      if (br_x_rec_diag_main) br_x_rec_diag_main->Fill();
      if (br_y_rec_diag_main) br_y_rec_diag_main->Fill();
      if (br_x_rec_diag_sec)  br_x_rec_diag_sec->Fill();
      if (br_y_rec_diag_sec)  br_y_rec_diag_sec->Fill();
      ReconTrueDeltaMDiagX = out_mdiag_dx_s.empty()? INVALID_VALUE : out_mdiag_dx_s[i];
      ReconTrueDeltaMDiagY = out_mdiag_dy_s.empty()? INVALID_VALUE : out_mdiag_dy_s[i];
      ReconTrueDeltaSDiagX = out_sdiag_dx_s.empty()? INVALID_VALUE : out_sdiag_dx_s[i];
      ReconTrueDeltaSDiagY = out_sdiag_dy_s.empty()? INVALID_VALUE : out_sdiag_dy_s[i];
      if (br_mdiag_dx_signed) br_mdiag_dx_signed->Fill();
      if (br_mdiag_dy_signed) br_mdiag_dy_signed->Fill();
      if (br_sdiag_dx_signed) br_sdiag_dx_signed->Fill();
      if (br_sdiag_dy_signed) br_sdiag_dy_signed->Fill();
      if (saveLineMeans) {
        ReconMeanX = out_x_mean_lines.empty()? INVALID_VALUE : out_x_mean_lines[i];
        ReconMeanY = out_y_mean_lines.empty()? INVALID_VALUE : out_y_mean_lines[i];
        ReconTrueDeltaMeanX = out_dx_mean_s.empty()? INVALID_VALUE : out_dx_mean_s[i];
        ReconTrueDeltaMeanY = out_dy_mean_s.empty()? INVALID_VALUE : out_dy_mean_s[i];
        if (br_x_mean_lines) br_x_mean_lines->Fill();
        if (br_y_mean_lines) br_y_mean_lines->Fill();
        if (br_dx_mean_signed) br_dx_mean_signed->Fill();
        if (br_dy_mean_signed) br_dy_mean_signed->Fill();
      }
      GaussMDiagA = out_d1_A.empty()? INVALID_VALUE : out_d1_A[i];
      GaussMDiagMu = out_d1_mu.empty()? INVALID_VALUE : out_d1_mu[i];
      GaussMDiagSigma = out_d1_sigma.empty()? INVALID_VALUE : out_d1_sigma[i];
      GaussMDiagB = out_d1_B.empty()? INVALID_VALUE : out_d1_B[i];
      GaussMDiagChi2 = out_d1_chi2.empty()? INVALID_VALUE : out_d1_chi2[i];
      GaussMDiagNdf = out_d1_ndf.empty()? INVALID_VALUE : out_d1_ndf[i];
      GaussMDiagProb = out_d1_prob.empty()? INVALID_VALUE : out_d1_prob[i];
      GaussSDiagA = out_d2_A.empty()? INVALID_VALUE : out_d2_A[i];
      GaussSDiagMu = out_d2_mu.empty()? INVALID_VALUE : out_d2_mu[i];
      GaussSDiagSigma = out_d2_sigma.empty()? INVALID_VALUE : out_d2_sigma[i];
      GaussSDiagB = out_d2_B.empty()? INVALID_VALUE : out_d2_B[i];
      GaussSDiagChi2 = out_d2_chi2.empty()? INVALID_VALUE : out_d2_chi2[i];
      GaussSDiagNdf = out_d2_ndf.empty()? INVALID_VALUE : out_d2_ndf[i];
      GaussSDiagProb = out_d2_prob.empty()? INVALID_VALUE : out_d2_prob[i];
      if (br_d1_A && saveDiagParamA) br_d1_A->Fill();
      if (br_d1_mu && saveDiagParamMu) br_d1_mu->Fill();
      if (br_d1_sigma && saveDiagParamSigma) br_d1_sigma->Fill();
      if (br_d1_B && saveDiagParamB) br_d1_B->Fill();
      if (br_d1_chi2) br_d1_chi2->Fill();
      if (br_d1_ndf) br_d1_ndf->Fill();
      if (br_d1_prob) br_d1_prob->Fill();
      if (br_d2_A && saveDiagParamA) br_d2_A->Fill();
      if (br_d2_mu && saveDiagParamMu) br_d2_mu->Fill();
      if (br_d2_sigma && saveDiagParamSigma) br_d2_sigma->Fill();
      if (br_d2_B && saveDiagParamB) br_d2_B->Fill();
      if (br_d2_chi2) br_d2_chi2->Fill();
      if (br_d2_ndf) br_d2_ndf->Fill();
      if (br_d2_prob) br_d2_prob->Fill();
    }
    nProcessed++;
  }

  // Re-enable all branches to avoid persisting disabled-status to file
  tree->SetBranchStatus("*", 1);

  // Overwrite tree in file
  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();
  delete file;

  ::Info("FitGaus1D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
  return 0;
}
