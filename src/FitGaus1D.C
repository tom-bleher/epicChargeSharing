// ROOT macro: FitGaus1D.C
// Performs 1D Gaussian fits on central row and column of the charge neighborhood
// using Qf (noisy charge per pixel) to reconstruct (x_rec_2d, y_rec_2d) and
// deltas, and appends them as new branches. Falls back to Qi if Qf is absent.

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
#include <utility>
#include <optional>
#include <ROOT/TThreadExecutor.hxx>

#include "ChargeUtils.h"

namespace fitgaus1d
{
namespace detail
{
double InferPixelSpacingFromTree(TTree* tree);
int InferRadiusFromTree(TTree* tree, const std::string& preferredBranch);
}

struct Metadata
{
    double pixelSpacing{std::numeric_limits<double>::quiet_NaN()};
    double pixelSize{std::numeric_limits<double>::quiet_NaN()};
    int neighborhoodRadius{-1};
};

std::unique_ptr<TFile> OpenRootFile(const std::string& filename)
{
    auto file = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "UPDATE"));
    if (!file || file->IsZombie()) {
        ::Error("FitGaus1D", "Cannot open file: %s", filename.c_str());
        return nullptr;
    }
    return file;
}

TTree* LoadHitsTree(TFile& file, const std::string& filename)
{
    auto* tree = dynamic_cast<TTree*>(file.Get("Hits"));
    if (!tree) {
        ::Error("FitGaus1D",
                "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)",
                filename.c_str());
    }
    return tree;
}

std::optional<Metadata> ExtractMetadata(TFile& file,
                                        TTree& tree,
                                        const std::string& preferredBranch)
{
    Metadata meta;

    if (auto* spacingObj = dynamic_cast<TNamed*>(file.Get("GridPixelSpacing_mm"))) {
        try {
            meta.pixelSpacing = std::stod(spacingObj->GetTitle());
        } catch (...) {
        }
    }
    if (auto* sizeObj = dynamic_cast<TNamed*>(file.Get("GridPixelSize_mm"))) {
        try {
            meta.pixelSize = std::stod(sizeObj->GetTitle());
        } catch (...) {
        }
    }
    if (auto* radiusObj = dynamic_cast<TNamed*>(file.Get("NeighborhoodRadius"))) {
        try {
            meta.neighborhoodRadius = std::stoi(radiusObj->GetTitle());
        } catch (...) {
            try {
                meta.neighborhoodRadius =
                    static_cast<int>(std::lround(std::stod(radiusObj->GetTitle())));
            } catch (...) {
            }
        }
    }

    if (!std::isfinite(meta.pixelSpacing) || meta.pixelSpacing <= 0.0) {
        meta.pixelSpacing = detail::InferPixelSpacingFromTree(&tree);
    }
    if (!std::isfinite(meta.pixelSpacing) || meta.pixelSpacing <= 0.0) {
        ::Error("FitGaus1D",
                "Pixel spacing not available (metadata missing and inference failed). Aborting.");
        return std::nullopt;
    }
    if (!std::isfinite(meta.pixelSize) || meta.pixelSize <= 0.0) {
        meta.pixelSize = 0.5 * meta.pixelSpacing;
    }
    if (meta.neighborhoodRadius <= 0) {
        meta.neighborhoodRadius = detail::InferRadiusFromTree(&tree, preferredBranch);
    }

    return meta;
}

std::string ResolveChargeBranch(TTree& tree, const std::string& requestedBranch)
{
    auto hasBranch = [&](const std::string& name) {
        return !name.empty() && tree.GetBranch(name.c_str()) != nullptr;
    };

    if (hasBranch(requestedBranch)) {
        return requestedBranch;
    }
    if (hasBranch("Qf")) {
        return "Qf";
    }
    if (hasBranch("Fi")) {
        return "Fi";
    }
    if (hasBranch("Qi")) {
        return "Qi";
    }

    ::Error("FitGaus1D",
            "No charge branch found (requested '%s'). Tried Qf, Fi, Qi.",
            requestedBranch.c_str());
    return {};
}

} // namespace fitgaus1d

namespace fitgaus1d::detail {
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

  double InferPixelSpacingFromTree(TTree* tree) {
    if (!tree) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double inferred = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(5000);
    ys.reserve(5000);
    double x_px_tmp = 0.0;
    double y_px_tmp = 0.0;
    tree->SetBranchStatus("PixelX", 1);
    tree->SetBranchStatus("PixelY", 1);
    tree->SetBranchAddress("PixelX", &x_px_tmp);
    tree->SetBranchAddress("PixelY", &y_px_tmp);
    const Long64_t nEntries = std::min<Long64_t>(tree->GetEntries(), 50000);
    for (Long64_t i = 0; i < nEntries; ++i) {
      tree->GetEntry(i);
      if (IsFinite(x_px_tmp)) xs.push_back(x_px_tmp);
      if (IsFinite(y_px_tmp)) ys.push_back(y_px_tmp);
    }
    auto computeGap = [](std::vector<double>& values) -> double {
      if (values.size() < 2) return std::numeric_limits<double>::quiet_NaN();
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
      if (values.size() < 2) return std::numeric_limits<double>::quiet_NaN();
      std::vector<double> gaps;
      gaps.reserve(values.size());
      for (size_t i = 1; i < values.size(); ++i) {
        const double d = values[i] - values[i - 1];
        if (d > 1e-9 && IsFinite(d)) gaps.push_back(d);
      }
      if (gaps.empty()) return std::numeric_limits<double>::quiet_NaN();
      std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
      return gaps[gaps.size() / 2];
    };
    const double gx = computeGap(xs);
    const double gy = computeGap(ys);
    if (IsFinite(gx) && gx > 0 && IsFinite(gy) && gy > 0) {
      inferred = 0.5 * (gx + gy);
    } else if (IsFinite(gx) && gx > 0) {
      inferred = gx;
    } else if (IsFinite(gy) && gy > 0) {
      inferred = gy;
    }
    tree->ResetBranchAddresses();
    return inferred;
  }

  int InferRadiusFromTree(TTree* tree, const std::string& preferredBranch) {
    if (!tree) {
      return -1;
    }
    std::vector<double>* charges = nullptr;
    auto bind = [&](const char* branch) -> bool {
      if (!branch || tree->GetBranch(branch) == nullptr) {
        return false;
      }
      tree->SetBranchStatus(branch, 1);
      tree->SetBranchAddress(branch, &charges);
      return true;
    };
    bool bound = false;
    if (!preferredBranch.empty() && bind(preferredBranch.c_str())) {
      bound = true;
    } else if (bind("Qf")) {
      bound = true;
    } else if (bind("Fi")) {
      bound = true;
    } else if (bind("Qi")) {
      bound = true;
    }
    if (!bound) {
      return -1;
    }
    const Long64_t nEntries = std::min<Long64_t>(tree->GetEntries(), 50000);
    int inferredRadius = -1;
    for (Long64_t i = 0; i < nEntries; ++i) {
      tree->GetEntry(i);
      if (!charges || charges->empty()) continue;
      const int total = static_cast<int>(charges->size());
      const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
      if (N >= 3 && N * N == total) {
        inferredRadius = (N - 1) / 2;
        break;
      }
    }
    tree->ResetBranchAddresses();
    return inferredRadius;
  }

  inline double ComputeUniformSigma(double errorPercentOfMax, double qmax) {
    return charge_uncert::UniformPercentOfMax(errorPercentOfMax, qmax);
  }

  inline std::pair<double, bool> WeightedCentroid(const std::vector<double>& positions,
                                                  const std::vector<double>& charges,
                                                  double baseline) {
    if (positions.size() != charges.size() || positions.empty()) {
      return {std::numeric_limits<double>::quiet_NaN(), false};
    }
    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < positions.size(); ++i) {
      const double weight = std::max(0.0, charges[i] - baseline);
      if (weight <= 0.0) continue;
      weightTotal += weight;
      weightedSum += weight * positions[i];
    }
    if (weightTotal <= 0.0) {
      return {std::numeric_limits<double>::quiet_NaN(), false};
    }
    return {weightedSum / weightTotal, true};
  }

  inline double SeedSigma(const std::vector<double>& positions,
                          const std::vector<double>& charges,
                          double baseline,
                          double pixelSpacing,
                          double sigmaLo,
                          double sigmaHi) {
    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < positions.size(); ++i) {
      const double weight = std::max(0.0, charges[i] - baseline);
      if (weight <= 0.0) continue;
      weightTotal += weight;
      weightedSum += weight * positions[i];
    }
    double sigma = std::numeric_limits<double>::quiet_NaN();
    if (weightTotal > 0.0) {
      const double mean = weightedSum / weightTotal;
      double variance = 0.0;
      for (size_t i = 0; i < positions.size(); ++i) {
        const double weight = std::max(0.0, charges[i] - baseline);
        if (weight <= 0.0) continue;
        const double dx = positions[i] - mean;
        variance += weight * dx * dx;
      }
      variance = (weightTotal > 0.0) ? (variance / weightTotal) : 0.0;
      sigma = std::sqrt(std::max(variance, 1e-12));
    }
    if (!IsFinite(sigma) || sigma <= 0.0) {
      sigma = std::max(0.25 * pixelSpacing, 1e-6);
    }
    if (sigma < sigmaLo) sigma = sigmaLo;
    if (sigma > sigmaHi) sigma = sigmaHi;
    return sigma;
  }

  struct GaussFitResult {
    bool converged = false;
    double A = std::numeric_limits<double>::quiet_NaN();
    double mu = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    double B = std::numeric_limits<double>::quiet_NaN();
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    double ndf = std::numeric_limits<double>::quiet_NaN();
    double prob = std::numeric_limits<double>::quiet_NaN();
  };

  struct GaussFitConfig {
    double muLo;
    double muHi;
    double sigmaLo;
    double sigmaHi;
    double qmax;
    double pixelSpacing;
    double seedA;
    double seedMu;
    double seedSigma;
    double seedB;
  };

  GaussFitResult RunGaussianFit(const std::vector<double>& positions,
                                const std::vector<double>& charges,
                                const std::vector<double>* sigmaCandidates,
                                double uniformSigma,
                                const GaussFitConfig& cfg) {
    GaussFitResult result;
    if (positions.size() != charges.size() || positions.size() < 3) {
      return result;
    }

    TF1 fLoc("fGauss1D_helper", GaussPlusB, -1e9, 1e9, 4);
    auto minmaxPos = std::minmax_element(positions.begin(), positions.end());
    if (minmaxPos.first != positions.end()) {
      const double margin = 0.5 * cfg.pixelSpacing;
      const double rangeLo = (*minmaxPos.first) - margin;
      const double rangeHi = (*minmaxPos.second) + margin;
      fLoc.SetRange(rangeLo, rangeHi);
    }
    const double qmax = IsFinite(cfg.qmax) ? std::max(cfg.qmax, 0.0) : 0.0;
    const double amplitudeMax = std::max(1e-18, 2.0 * qmax);
    const double baselineMax = std::max(1e-18, qmax);
    fLoc.SetParameters(cfg.seedA, cfg.seedMu, cfg.seedSigma, cfg.seedB);
    fLoc.SetParLimits(0, 1e-18, amplitudeMax);
    fLoc.SetParLimits(1, cfg.muLo, cfg.muHi);
    fLoc.SetParLimits(2, cfg.sigmaLo, cfg.sigmaHi);
    fLoc.SetParLimits(3, -baselineMax, baselineMax);
    ROOT::Math::WrappedMultiTF1 wrapped(fLoc, 1);
    ROOT::Fit::BinData data(static_cast<int>(positions.size()), 1);
    const bool hasSigmaCandidates = sigmaCandidates && !sigmaCandidates->empty();
    for (size_t i = 0; i < positions.size(); ++i) {
      double candidate = std::numeric_limits<double>::quiet_NaN();
      if (hasSigmaCandidates && i < sigmaCandidates->size()) {
        candidate = (*sigmaCandidates)[i];
      }
      const double sigmaY = charge_uncert::SelectVerticalSigma(candidate, uniformSigma);
      data.Add(positions[i], charges[i], sigmaY);
    }
    ROOT::Fit::Fitter fitter;
    fitter.Config().SetMinimizer("Minuit2", "Fumili2");
    fitter.Config().MinimizerOptions().SetStrategy(0);
    fitter.Config().MinimizerOptions().SetTolerance(1e-4);
    fitter.Config().MinimizerOptions().SetPrintLevel(0);
    fitter.SetFunction(wrapped);
    fitter.Config().ParSettings(0).SetLimits(1e-18, amplitudeMax);
    fitter.Config().ParSettings(1).SetLimits(cfg.muLo, cfg.muHi);
    fitter.Config().ParSettings(2).SetLimits(cfg.sigmaLo, cfg.sigmaHi);
    fitter.Config().ParSettings(3).SetLimits(-baselineMax, baselineMax);
    const double stepA = std::max(1e-18, 0.01 * cfg.seedA);
    const double stepB = std::max(1e-18, 0.01 * std::max(std::abs(cfg.seedB), cfg.seedA));
    fitter.Config().ParSettings(0).SetStepSize(stepA);
    fitter.Config().ParSettings(1).SetStepSize(1e-4 * cfg.pixelSpacing);
    fitter.Config().ParSettings(2).SetStepSize(1e-4 * cfg.pixelSpacing);
    fitter.Config().ParSettings(3).SetStepSize(stepB);
    fitter.Config().ParSettings(0).SetValue(cfg.seedA);
    fitter.Config().ParSettings(1).SetValue(cfg.seedMu);
    fitter.Config().ParSettings(2).SetValue(cfg.seedSigma);
    fitter.Config().ParSettings(3).SetValue(cfg.seedB);
    bool ok = fitter.Fit(data);
    if (!ok) {
      fitter.Config().SetMinimizer("Minuit2", "Migrad");
      fitter.Config().MinimizerOptions().SetStrategy(1);
      fitter.Config().MinimizerOptions().SetTolerance(1e-3);
      fitter.Config().MinimizerOptions().SetPrintLevel(0);
      ok = fitter.Fit(data);
    }
    if (!ok) {
      return result;
    }
    result.converged = true;
    result.A = fitter.Result().Parameter(0);
    result.mu = fitter.Result().Parameter(1);
    result.sigma = fitter.Result().Parameter(2);
    result.B = fitter.Result().Parameter(3);
    result.chi2 = fitter.Result().Chi2();
    result.ndf = fitter.Result().Ndf();
    result.prob = (fitter.Result().Ndf() > 0)
                      ? TMath::Prob(fitter.Result().Chi2(), fitter.Result().Ndf())
                      : std::numeric_limits<double>::quiet_NaN();
    return result;
  }
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
                 const char* chargeBranch = "Qf",
                 bool fitDiagonals = false,
                 bool saveDiagParamA = true,
                 bool saveDiagParamMu = true,
                 bool saveDiagParamSigma = true,
                 bool saveDiagParamB = true,
                 bool saveLineMeans = true,
                 bool useQnQiPercentErrors = false,
                 bool useDistanceWeightedErrors = true,
                 double distanceErrorScalePixels = 1.5,
                 double distanceErrorExponent = 1.5,
                 double distanceErrorFloorPercent = 4.0,
                 double distanceErrorCapPercent = 10.0,
                 bool distanceErrorPreferTruthCenter = true,
                 bool distanceErrorPowerInverse = true) {
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  const bool distanceErrorsEnabled = useDistanceWeightedErrors;
  const double distScalePx = (std::isfinite(distanceErrorScalePixels) &&
                              distanceErrorScalePixels > 0.0)
                                 ? distanceErrorScalePixels
                                 : 1.5;
  const double distExponent = std::isfinite(distanceErrorExponent)
                                  ? std::max(1.0001, distanceErrorExponent)
                                  : 1.5;
  const double distFloorPct = std::isfinite(distanceErrorFloorPercent)
                                  ? std::max(0.0, distanceErrorFloorPercent)
                                  : 0.0;
  const double distCapPct = std::isfinite(distanceErrorCapPercent)
                                ? std::max(0.0, distanceErrorCapPercent)
                                : 0.0;
  const bool distPreferTruthCenter = distanceErrorPreferTruthCenter;
  const bool distPowerInverse = distanceErrorPowerInverse;

  using fitgaus1d::detail::FlatVectorStore;
  using fitgaus1d::detail::FitWorkBuffers;
  using fitgaus1d::detail::IsFinite;

  if (distanceErrorsEnabled && useQnQiPercentErrors) {
    ::Warning("FitGaus1D",
              "Distance-weighted uncertainties requested; ignoring Qn/Qi"
              " vertical uncertainty model.");
  }

  const std::string fileNameStr = filename ? std::string(filename) : std::string("../build/epicChargeSharing.root");
  auto fileHandle = fitgaus1d::OpenRootFile(fileNameStr);
  if (!fileHandle) {
    return 1;
  }

  TTree* tree = fitgaus1d::LoadHitsTree(*fileHandle, fileNameStr);
  if (!tree) {
    fileHandle->Close();
    return 3;
  }

  const std::string requestedBranch =
      (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Qf");

  const auto metadataOpt = fitgaus1d::ExtractMetadata(*fileHandle, *tree, requestedBranch);
  if (!metadataOpt) {
    fileHandle->Close();
    return 2;
  }
  fitgaus1d::Metadata metadata = *metadataOpt;

  const std::string chosenCharge = fitgaus1d::ResolveChargeBranch(*tree, requestedBranch);
  if (chosenCharge.empty()) {
    fileHandle->Close();
    return 4;
  }

  double pixelSpacing = metadata.pixelSpacing;
  double pixelSize = metadata.pixelSize;
  int neighborhoodRadiusMeta = metadata.neighborhoodRadius;

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = fitgaus1d::detail::InferPixelSpacingFromTree(tree);
  }
  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("FitGaus1D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    fileHandle->Close();
    return 2;
  }
  if (!IsFinite(pixelSize) || pixelSize <= 0) {
    // Fallback: if pixel size metadata is missing, use half of pitch as a conservative lower bound
    pixelSize = 0.5 * pixelSpacing;
  }

  if (neighborhoodRadiusMeta <= 0) {
    neighborhoodRadiusMeta = fitgaus1d::detail::InferRadiusFromTree(tree, chosenCharge);
  }

  auto distanceSigma = [&](double distance, double qmax) -> double {
    if (distPowerInverse) {
      return charge_uncert::DistancePowerSigmaInverse(distance,
                                                      qmax,
                                                      pixelSpacing,
                                                      distScalePx,
                                                      distExponent,
                                                      distFloorPct,
                                                      distCapPct);
    }
    return charge_uncert::DistancePowerSigma(distance,
                                             qmax,
                                             pixelSpacing,
                                             distScalePx,
                                             distExponent,
                                             distFloorPct,
                                             distCapPct);
  };
  // Existing branches (inputs)
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  // Use Qf (noisy) for fits; fall back to Qi if Qf absent
  std::vector<double>* Q = nullptr; // used for fits (charges in Coulombs)
  std::vector<double>* Qi = nullptr; // initial charge (for error model)
  std::vector<double>* Qn = nullptr; // noiseless charge (for error model)
  bool enableQiQnErrors = useQnQiPercentErrors && !distanceErrorsEnabled;
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
    haveQiBranchForErrors = tree->GetBranch("Qi") != nullptr;
    haveQnBranchForErrors = tree->GetBranch("Qn") != nullptr;
    if (haveQiBranchForErrors && haveQnBranchForErrors) {
      tree->SetBranchStatus("Qi", 1);
      tree->SetBranchStatus("Qn", 1);
    } else {
      ::Warning("FitGaus1D", "Requested Qi/Qn vertical errors but required branches are missing. Falling back to percent-of-max uncertainty.");
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
    tree->SetBranchAddress("Qi", &Qi);
  }
  if (enableQiQnErrors && haveQnBranchForErrors) {
    tree->SetBranchAddress("Qn", &Qn);
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
    const bool needRowColErrors = haveQiQnForEvent || distanceErrorsEnabled;
    fitBuffers.PrepareRowCol(N, needRowColErrors);
    static thread_local std::vector<double> dist_err_row;
    static thread_local std::vector<double> dist_err_col;
    static thread_local std::vector<double> dist_err_d1;
    static thread_local std::vector<double> dist_err_d2;

    dist_err_row.clear();
    dist_err_col.clear();
    if (fitDiagonals) {
      dist_err_d1.clear();
      dist_err_d2.clear();
    }

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
      const bool needDiagErrors = haveQiQnForEvent || distanceErrorsEnabled;
      fitBuffers.PrepareDiag(N, needDiagErrors);
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
          errCandidate = charge_uncert::QnQiScaled(qiVal, QnLoc[idx], qmaxQiNeighborhood);
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

    const double uniformSigma = fitgaus1d::detail::ComputeUniformSigma(errorPercentOfMax, qmaxNeighborhood);

    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    // Allow negative baseline seed if the data suggest it
    double B0_row = *minmaxRow.first;
    double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    double B0_col = *minmaxCol.first;

    const auto [rowCentroid, rowCentroidOk] = fitgaus1d::detail::WeightedCentroid(x_row, q_row, B0_row);
    const auto [colCentroid, colCentroidOk] = fitgaus1d::detail::WeightedCentroid(y_col, q_col, B0_col);

    // Low-contrast: fast centroid (relative to neighborhood max charge)
    const double contrastEps = (qmaxNeighborhood > 0.0) ? (1e-3 * qmaxNeighborhood) : 0.0;
    if (qmaxNeighborhood > 0.0 && A0_row < contrastEps && A0_col < contrastEps) {
      if (rowCentroidOk && colCentroidOk) {
        out_x_rec[i] = rowCentroid;
        out_y_rec[i] = colCentroid;
        out_dx_s[i] = (v_x_hit[i] - rowCentroid);
        out_dy_s[i] = (v_y_hit[i] - colCentroid);
        nFitted.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }

    // Seeds
    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    double centerX = mu0_row;
    double centerY = mu0_col;
    if (distanceErrorsEnabled) {
      if (distPreferTruthCenter) {
        if (IsFinite(v_x_hit[i])) {
          centerX = v_x_hit[i];
        }
        if (IsFinite(v_y_hit[i])) {
          centerY = v_y_hit[i];
        }
      }
      if (!IsFinite(centerX)) {
        centerX = mu0_row;
      }
      if (!IsFinite(centerX)) {
        centerX = x_px_loc;
      }
      if (!IsFinite(centerY)) {
        centerY = mu0_col;
      }
      if (!IsFinite(centerY)) {
        centerY = y_px_loc;
      }
    }

    bool rowDistanceApplied = false;
    bool colDistanceApplied = false;
    bool diagMainDistanceApplied = false;
    bool diagSecDistanceApplied = false;

    const bool canApplyDistance = distanceErrorsEnabled && IsFinite(pixelSpacing) &&
                                  pixelSpacing > 0.0 && IsFinite(qmaxNeighborhood) &&
                                  qmaxNeighborhood > 0.0 && IsFinite(centerX) &&
                                  IsFinite(centerY);

    if (canApplyDistance) {
      if (!x_row.empty()) {
        dist_err_row.reserve(x_row.size());
        bool anyFinite = false;
        for (size_t k = 0; k < x_row.size(); ++k) {
          double sigma = distanceSigma(std::abs(x_row[k] - centerX), qmaxNeighborhood);
          if (std::isfinite(sigma) && sigma > 0.0) {
            anyFinite = true;
            dist_err_row.push_back(sigma);
          } else {
            dist_err_row.push_back(std::numeric_limits<double>::quiet_NaN());
          }
        }
        if (anyFinite) {
          rowDistanceApplied = true;
        } else {
          dist_err_row.clear();
        }
      }
      if (!y_col.empty()) {
        dist_err_col.reserve(y_col.size());
        bool anyFinite = false;
        for (size_t k = 0; k < y_col.size(); ++k) {
          double sigma = distanceSigma(std::abs(y_col[k] - centerY), qmaxNeighborhood);
          if (std::isfinite(sigma) && sigma > 0.0) {
            anyFinite = true;
            dist_err_col.push_back(sigma);
          } else {
            dist_err_col.push_back(std::numeric_limits<double>::quiet_NaN());
          }
        }
        if (anyFinite) {
          colDistanceApplied = true;
        } else {
          dist_err_col.clear();
        }
      }
      if (fitDiagonals) {
        auto fillDiagErrors = [&](const std::vector<double>& coords,
                                  std::vector<double>& dest,
                                  bool& applied) {
          if (coords.empty()) {
            return;
          }
          dest.reserve(coords.size());
          bool anyFiniteDiag = false;
          for (double c : coords) {
            double sigma = distanceSigma(std::abs(c - centerX), qmaxNeighborhood);
            if (std::isfinite(sigma) && sigma > 0.0) {
              anyFiniteDiag = true;
              dest.push_back(sigma);
            } else {
              dest.push_back(std::numeric_limits<double>::quiet_NaN());
            }
          }
          if (anyFiniteDiag) {
            applied = true;
          } else {
            dest.clear();
          }
        };

        fillDiagErrors(fitBuffers.s_d1, dist_err_d1, diagMainDistanceApplied);
        fillDiagErrors(fitBuffers.s_d2, dist_err_d2, diagSecDistanceApplied);
      }
    }

    // Constrain sigma to be within [pixel size, radius * pitch]
    const double sigLoBound = pixelSize;
    const double sigHiBound = std::max(sigLoBound, static_cast<double>(neighborhoodRadiusMeta > 0 ? neighborhoodRadiusMeta : R) * pixelSpacing);
    const double sigInitRow = fitgaus1d::detail::SeedSigma(x_row, q_row, B0_row, pixelSpacing, sigLoBound, sigHiBound);
    const double sigInitCol = fitgaus1d::detail::SeedSigma(y_col, q_col, B0_col, pixelSpacing, sigLoBound, sigHiBound);

    const double muXLo = x_px_loc - 1.0 * pixelSpacing;
    const double muXHi = x_px_loc + 1.0 * pixelSpacing;
    const double muYLo = y_px_loc - 1.0 * pixelSpacing;
    const double muYHi = y_px_loc + 1.0 * pixelSpacing;

    const std::vector<double>* rowSigmaCandidates = nullptr;
    if (rowDistanceApplied && !dist_err_row.empty()) {
      rowSigmaCandidates = &dist_err_row;
    } else if (haveQiQnForEvent && !err_row.empty()) {
      rowSigmaCandidates = &err_row;
    }
    const std::vector<double>* colSigmaCandidates = nullptr;
    if (colDistanceApplied && !dist_err_col.empty()) {
      colSigmaCandidates = &dist_err_col;
    } else if (haveQiQnForEvent && !err_col.empty()) {
      colSigmaCandidates = &err_col;
    }

    fitgaus1d::detail::GaussFitConfig rowConfig{muXLo, muXHi, sigLoBound, sigHiBound,
                             qmaxNeighborhood, pixelSpacing, A0_row,
                             mu0_row, sigInitRow, B0_row};
    fitgaus1d::detail::GaussFitConfig colConfig{muYLo, muYHi, sigLoBound, sigHiBound,
                             qmaxNeighborhood, pixelSpacing, A0_col,
                             mu0_col, sigInitCol, B0_col};

    fitgaus1d::detail::GaussFitResult rowFit = fitgaus1d::detail::RunGaussianFit(x_row, q_row, rowSigmaCandidates,
                                           uniformSigma, rowConfig);
    fitgaus1d::detail::GaussFitResult colFit = fitgaus1d::detail::RunGaussianFit(y_col, q_col, colSigmaCandidates,
                                           uniformSigma, colConfig);

    if (rowFit.converged) {
      out_row_A[i] = rowFit.A;
      out_row_mu[i] = rowFit.mu;
      out_row_sigma[i] = rowFit.sigma;
      out_row_B[i] = rowFit.B;
      out_row_chi2[i] = rowFit.chi2;
      out_row_ndf[i] = rowFit.ndf;
      out_row_prob[i] = rowFit.prob;
    }
    if (colFit.converged) {
      out_col_A[i] = colFit.A;
      out_col_mu[i] = colFit.mu;
      out_col_sigma[i] = colFit.sigma;
      out_col_B[i] = colFit.B;
      out_col_chi2[i] = colFit.chi2;
      out_col_ndf[i] = colFit.ndf;
      out_col_prob[i] = colFit.prob;
    }

    double muX = rowFit.converged ? rowFit.mu : std::numeric_limits<double>::quiet_NaN();
    double muY = colFit.converged ? colFit.mu : std::numeric_limits<double>::quiet_NaN();
    if (!rowFit.converged && rowCentroidOk) {
      muX = rowCentroid;
    }
    if (!colFit.converged && colCentroidOk) {
      muY = colCentroid;
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
      const double muLo = x_px_loc - 1.0 * pixelSpacing;
      const double muHi = x_px_loc + 1.0 * pixelSpacing;
      const std::vector<double>* err_d1_fit = nullptr;
      if (diagMainDistanceApplied && !dist_err_d1.empty()) {
        err_d1_fit = &dist_err_d1;
      } else if (haveQiQnForEvent && !err_d1_vec.empty()) {
        err_d1_fit = &err_d1_vec;
      }
      const std::vector<double>* err_d2_fit = nullptr;
      if (diagSecDistanceApplied && !dist_err_d2.empty()) {
        err_d2_fit = &dist_err_d2;
      } else if (haveQiQnForEvent && !err_d2_vec.empty()) {
        err_d2_fit = &err_d2_vec;
      }

      bool ok1 = false;
      bool ok2 = false;
      double mu1 = INVALID_VALUE;
      double mu2 = INVALID_VALUE;
      fitgaus1d::detail::GaussFitResult diagMainFit;
      fitgaus1d::detail::GaussFitResult diagSecFit;

      if (s_d1_vec.size() >= 3) {
        auto mm = std::minmax_element(q_d1_vec.begin(), q_d1_vec.end());
        double A0 = std::max(1e-18, *mm.second - *mm.first);
        double B0 = *mm.first;
        int idxMax = std::distance(q_d1_vec.begin(), std::max_element(q_d1_vec.begin(), q_d1_vec.end()));
        double mu0 = s_d1_vec[idxMax];
        const double sigInit = fitgaus1d::detail::SeedSigma(s_d1_vec, q_d1_vec, B0, pixelSpacing, sigLoBound, sigHiBound);
        fitgaus1d::detail::GaussFitConfig cfgDiag{muLo, muHi, sigLoBound, sigHiBound,
                               qmaxNeighborhood, pixelSpacing, A0,
                               mu0, sigInit, B0};
        const auto centroid = fitgaus1d::detail::WeightedCentroid(s_d1_vec, q_d1_vec, B0);
        diagMainFit = fitgaus1d::detail::RunGaussianFit(s_d1_vec, q_d1_vec, err_d1_fit, uniformSigma, cfgDiag);
        if (diagMainFit.converged) {
          ok1 = true;
          mu1 = diagMainFit.mu;
        } else if (centroid.second) {
          ok1 = true;
          mu1 = centroid.first;
          diagMainFit.A = A0;
          diagMainFit.mu = centroid.first;
          diagMainFit.sigma = sigInit;
          diagMainFit.B = B0;
          diagMainFit.chi2 = INVALID_VALUE;
          diagMainFit.ndf = INVALID_VALUE;
          diagMainFit.prob = INVALID_VALUE;
        }
      }

      if (s_d2_vec.size() >= 3) {
        auto mm = std::minmax_element(q_d2_vec.begin(), q_d2_vec.end());
        double A0 = std::max(1e-18, *mm.second - *mm.first);
        double B0 = *mm.first;
        int idxMax = std::distance(q_d2_vec.begin(), std::max_element(q_d2_vec.begin(), q_d2_vec.end()));
        double mu0 = s_d2_vec[idxMax];
        const double sigInit = fitgaus1d::detail::SeedSigma(s_d2_vec, q_d2_vec, B0, pixelSpacing, sigLoBound, sigHiBound);
        fitgaus1d::detail::GaussFitConfig cfgDiag{muLo, muHi, sigLoBound, sigHiBound,
                               qmaxNeighborhood, pixelSpacing, A0,
                               mu0, sigInit, B0};
        const auto centroid = fitgaus1d::detail::WeightedCentroid(s_d2_vec, q_d2_vec, B0);
        diagSecFit = fitgaus1d::detail::RunGaussianFit(s_d2_vec, q_d2_vec, err_d2_fit, uniformSigma, cfgDiag);
        if (diagSecFit.converged) {
          ok2 = true;
          mu2 = diagSecFit.mu;
        } else if (centroid.second) {
          ok2 = true;
          mu2 = centroid.first;
          diagSecFit.A = A0;
          diagSecFit.mu = centroid.first;
          diagSecFit.sigma = sigInit;
          diagSecFit.B = B0;
          diagSecFit.chi2 = INVALID_VALUE;
          diagSecFit.ndf = INVALID_VALUE;
          diagSecFit.prob = INVALID_VALUE;
        }
      }

      if (ok1) {
        // Transform diagonal coordinate mu -> (x,y)
        const double dx = mu1 - x_px_loc;
        out_x_rec_diag_main[i] = mu1;
        out_y_rec_diag_main[i] = y_px_loc + dx; // main diag: dj = di
        out_mdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_main[i]);
        out_mdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_main[i]);
        if (saveDiagParamA) out_d1_A[i] = diagMainFit.A;
        if (saveDiagParamMu) out_d1_mu[i] = diagMainFit.mu;
        if (saveDiagParamSigma) out_d1_sigma[i] = diagMainFit.sigma;
        if (saveDiagParamB) out_d1_B[i] = diagMainFit.B;
        out_d1_chi2[i] = diagMainFit.chi2;
        out_d1_ndf[i]  = diagMainFit.ndf;
        out_d1_prob[i] = diagMainFit.prob;
      }
      if (ok2) {
        const double dx = mu2 - x_px_loc;
        out_x_rec_diag_sec[i] = mu2;
        out_y_rec_diag_sec[i] = y_px_loc - dx; // secondary diag: dj = -di
        out_sdiag_dx_s[i] = (v_x_hit[i] - out_x_rec_diag_sec[i]);
        out_sdiag_dy_s[i] = (v_y_hit[i] - out_y_rec_diag_sec[i]);
        if (saveDiagParamA) out_d2_A[i] = diagSecFit.A;
        if (saveDiagParamMu) out_d2_mu[i] = diagSecFit.mu;
        if (saveDiagParamSigma) out_d2_sigma[i] = diagSecFit.sigma;
        if (saveDiagParamB) out_d2_B[i] = diagSecFit.B;
        out_d2_chi2[i] = diagSecFit.chi2;
        out_d2_ndf[i]  = diagSecFit.ndf;
        out_d2_prob[i] = diagSecFit.prob;
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

  // Sequentially write outputs to the tree (thread-safe). Use LoadTree to
  // advance entries without triggering I/O on the original branches.
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->LoadTree(i);
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
  fileHandle->cd();
  tree->Write("", TObject::kOverwrite);
  fileHandle->Flush();
  fileHandle->Close();

  ::Info("FitGaus1D", "Processed %lld entries, fitted %lld.", nProcessed, (long long)nFitted.load());
  return 0;
}
