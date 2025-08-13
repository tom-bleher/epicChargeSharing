// ROOT macro: processing2D_parallel.C
// Parallel-friendly version of processing2D using RDataFrame with implicit MT.
// - Reads input tree read-only
// - Computes per-entry 1D Gaussian fits for central row/column in thread-local lambdas
// - Writes only the derived columns to a separate output file/tree (no in-place UPDATE)

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TROOT.h>
#include <TError.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>

namespace {
  inline bool IsFinite(double v) { return std::isfinite(v); }

  inline void TryEnableImplicitMT() {
    // EnableImplicitMT may not be available in older ROOT headers.
    // Call it via the interpreter to avoid compile-time dependency.
    if (gROOT) {
      gROOT->ProcessLine("ROOT::EnableImplicitMT();");
    }
  }

  struct Fit2DResult {
    double x_rec;
    double y_rec;
    double dx_abs;
    double dy_abs;
    double dx_signed;
    double dy_signed;
    double dx_sq;
    double dy_sq;
  };

  // Perform per-entry 1D Gaussian fits on the central row and column
  // Returns all outputs in a struct for convenient unpacking
  Fit2DResult Compute2DFits(
      double x_hit,
      double y_hit,
      double x_px,
      double y_px,
      bool is_pixel_hit,
      const std::vector<double>& Fi,
      double pixelSpacing,
      double errorPercentOfMax)
  {
    const double INVALID = std::numeric_limits<double>::quiet_NaN();
    Fit2DResult out{INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID};

    if (is_pixel_hit || Fi.empty()) {
      return out;
    }

    const size_t total = Fi.size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      return out;
    }
    const int R = (N - 1) / 2;

    std::vector<double> x_row, q_row;
    std::vector<double> y_col, q_col;
    x_row.reserve(N); q_row.reserve(N);
    y_col.reserve(N); q_col.reserve(N);

    double qmaxNeighborhood = -1e300;

    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int irow = di + R;
        const int jcol = dj + R;
        const int idx  = irow * N + jcol;
        const double q = Fi[idx];
        if (!IsFinite(q) || q < 0) continue;
        if (q > qmaxNeighborhood) qmaxNeighborhood = q;
        if (dj == 0) { x_row.push_back(x_px + di * pixelSpacing); q_row.push_back(q); }
        if (di == 0) { y_col.push_back(y_px + dj * pixelSpacing); q_col.push_back(q); }
      }
    }

    if (x_row.size() < 3 || y_col.size() < 3) {
      return out;
    }

    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (qmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * qmaxNeighborhood
                              : 0.0;

    auto minmaxRow = std::minmax_element(q_row.begin(), q_row.end());
    auto minmaxCol = std::minmax_element(q_col.begin(), q_col.end());
    const double A0_row = std::max(1e-18, *minmaxRow.second - *minmaxRow.first);
    const double B0_row = std::max(0.0, *minmaxRow.first);
    const double A0_col = std::max(1e-18, *minmaxCol.second - *minmaxCol.first);
    const double B0_col = std::max(0.0, *minmaxCol.first);

    if (A0_row < 1e-6 && A0_col < 1e-6) {
      double wsumx = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k]); wsumx += w; xw += w * x_row[k]; }
      double wsumy = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k]); wsumy += w; yw += w * y_col[k]; }
      if (wsumx > 0 && wsumy > 0) {
        out.x_rec = xw / wsumx;
        out.y_rec = yw / wsumy;
        out.dx_signed = (x_hit - out.x_rec);
        out.dy_signed = (y_hit - out.y_rec);
        out.dx_abs = std::abs(out.dx_signed);
        out.dy_abs = std::abs(out.dy_signed);
        out.dx_sq = out.dx_signed * out.dx_signed;
        out.dy_sq = out.dy_signed * out.dy_signed;
      }
      return out;
    }

    int idxMaxRow = std::distance(q_row.begin(), std::max_element(q_row.begin(), q_row.end()));
    int idxMaxCol = std::distance(q_col.begin(), std::max_element(q_col.begin(), q_col.end()));
    double mu0_row = x_row[idxMaxRow];
    double mu0_col = y_col[idxMaxCol];

    auto minmaxX = std::minmax_element(x_row.begin(), x_row.end());
    auto minmaxY = std::minmax_element(y_col.begin(), y_col.end());
    const double xMin = *minmaxX.first - 0.5 * pixelSpacing;
    const double xMax = *minmaxX.second + 0.5 * pixelSpacing;
    const double yMin = *minmaxY.first - 0.5 * pixelSpacing;
    const double yMax = *minmaxY.second + 0.5 * pixelSpacing;

    const double sigInit = std::max(0.25*pixelSpacing, 1e-6);
    const double sigLo   = std::max(1e-6, 0.05*pixelSpacing);
    const double sigHi   = 2.0*pixelSpacing;

    auto chi2Row = [&](const double* p) -> double {
      const double A = p[0], mu = p[1], sig = p[2], B = p[3];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (size_t k=0;k<q_row.size();++k) {
        const double dx = (x_row[k] - mu)/sig;
        const double model = A * std::exp(-0.5*dx*dx) + B;
        const double r = (q_row[k] - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };
    auto chi2Col = [&](const double* p) -> double {
      const double A = p[0], mu = p[1], sig = p[2], B = p[3];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (size_t k=0;k<q_col.size();++k) {
        const double dy = (y_col[k] - mu)/sig;
        const double model = A * std::exp(-0.5*dy*dy) + B;
        const double r = (q_col[k] - model);
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
    // Match default strategy used in the non-parallel macro
    mRow->SetStrategy(0);
    mCol->SetStrategy(0);

    ROOT::Math::Functor fRowChi2(chi2Row, 4);
    ROOT::Math::Functor fColChi2(chi2Col, 4);
    mRow->SetFunction(fRowChi2);
    mCol->SetFunction(fColChi2);

    mRow->SetLimitedVariable(0, "A", A0_row, 1e-3, 0.0, 1.0);
    mRow->SetLimitedVariable(1, "mu", mu0_row, 1e-4*pixelSpacing, xMin, xMax);
    mRow->SetLimitedVariable(2, "sigma", sigInit, 1e-4*pixelSpacing, sigLo, sigHi);
    mRow->SetLimitedVariable(3, "B", B0_row, 1e-3, 0.0, std::max(0.0, *minmaxRow.first));

    mCol->SetLimitedVariable(0, "A", A0_col, 1e-3, 0.0, 1.0);
    mCol->SetLimitedVariable(1, "mu", mu0_col, 1e-4*pixelSpacing, yMin, yMax);
    mCol->SetLimitedVariable(2, "sigma", sigInit, 1e-4*pixelSpacing, sigLo, sigHi);
    mCol->SetLimitedVariable(3, "B", B0_col, 1e-3, 0.0, std::max(0.0, *minmaxCol.first));

    bool okRow = mRow->Minimize();
    bool okCol = mCol->Minimize();

    double muX = okRow ? mRow->X()[1] : INVALID;
    double muY = okCol ? mCol->X()[1] : INVALID;

    if (!okRow) {
      double wsum = 0.0, xw = 0.0;
      for (size_t k=0;k<x_row.size();++k) { double w = std::max(0.0, q_row[k]); wsum += w; xw += w * x_row[k]; }
      if (wsum > 0) { muX = xw / wsum; okRow = true; }
    }
    if (!okCol) {
      double wsum = 0.0, yw = 0.0;
      for (size_t k=0;k<y_col.size();++k) { double w = std::max(0.0, q_col[k]); wsum += w; yw += w * y_col[k]; }
      if (wsum > 0) { muY = yw / wsum; okCol = true; }
    }

    if (okRow && okCol && IsFinite(muX) && IsFinite(muY)) {
      out.x_rec = muX;
      out.y_rec = muY;
      out.dx_signed = (x_hit - out.x_rec);
      out.dy_signed = (y_hit - out.y_rec);
      out.dx_abs = std::abs(out.dx_signed);
      out.dy_abs = std::abs(out.dy_signed);
      out.dx_sq = out.dx_signed * out.dx_signed;
      out.dy_sq = out.dy_signed * out.dy_signed;
    }
    return out;
  }
}

int processing2D_parallel(const char* filename = "../build/epicChargeSharingOutput.root",
                          double errorPercentOfMax = 5.0,
                          const char* outFilePath = nullptr)
{
  // Read pixel spacing metadata, with fallback inference if needed
  double pixelSpacing = std::numeric_limits<double>::quiet_NaN();
  {
    TFile* inF = TFile::Open(filename, "READ");
    if (!inF || inF->IsZombie()) {
      ::Error("processing2D_parallel", "Cannot open file: %s", filename);
      if (inF) { inF->Close(); delete inF; }
      return 1;
    }
    if (auto* spacingObj = dynamic_cast<TNamed*>(inF->Get("GridPixelSpacing_mm"))) {
      try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) { pixelSpacing = std::numeric_limits<double>::quiet_NaN(); }
    }
    // Fallback: infer spacing from x_px/y_px if needed
    if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
      TTree* tree = dynamic_cast<TTree*>(inF->Get("Hits"));
      if (!tree) {
        ::Error("processing2D_parallel", "Hits tree not found in file: %s", filename);
        inF->Close(); delete inF; return 3;
      }
      double x_px_tmp = 0.0, y_px_tmp = 0.0;
      tree->SetBranchAddress("x_px", &x_px_tmp);
      tree->SetBranchAddress("y_px", &y_px_tmp);
      std::vector<double> xs; xs.reserve(5000);
      std::vector<double> ys; ys.reserve(5000);
      Long64_t nToScan = std::min<Long64_t>(tree->GetEntries(), 50000);
      for (Long64_t i=0;i<nToScan;++i) {
        tree->GetEntry(i);
        if (IsFinite(x_px_tmp)) xs.push_back(x_px_tmp);
        if (IsFinite(y_px_tmp)) ys.push_back(y_px_tmp);
      }
      auto computeGap = [](std::vector<double>& v)->double{
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> gaps; gaps.reserve(v.size());
        for (size_t i=1;i<v.size();++i) {
          double d = v[i]-v[i-1];
          if (d > 1e-9 && IsFinite(d)) gaps.push_back(d);
        }
        if (gaps.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::nth_element(gaps.begin(), gaps.begin()+gaps.size()/2, gaps.end());
        return gaps[gaps.size()/2];
      };
      double gx = computeGap(xs);
      double gy = computeGap(ys);
      if (IsFinite(gx) && gx>0 && IsFinite(gy) && gy>0) pixelSpacing = 0.5*(gx+gy);
      else if (IsFinite(gx) && gx>0) pixelSpacing = gx;
      else if (IsFinite(gy) && gy>0) pixelSpacing = gy;
    }
    inF->Close();
    delete inF;
  }

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing2D_parallel", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    return 2;
  }

  // Derive output file path if not provided
  std::string outPath;
  if (outFilePath && std::string(outFilePath).size() > 0) {
    outPath = outFilePath;
  } else {
    std::string in(filename);
    auto pos = in.rfind('.');
    if (pos == std::string::npos) outPath = in + ".proc2d.root";
    else outPath = in.substr(0, pos) + ".proc2d.root";
  }

  // Enable implicit multi-threading via runtime call (portable across ROOT versions)
  TryEnableImplicitMT();

  // Build dataframe and define outputs
  ROOT::RDataFrame df("Hits", filename);

  auto base = df.Define("__fit2d__",
                       [pixelSpacing, errorPercentOfMax](double x_hit, double y_hit,
                                                         double x_px,  double y_px,
                                                         bool is_pixel_hit,
                                                         const std::vector<double>& Fi) {
                         Fit2DResult r = Compute2DFits(x_hit, y_hit, x_px, y_px, is_pixel_hit, Fi,
                                                       pixelSpacing, errorPercentOfMax);
                         ROOT::RVec<double> v(8);
                         v[0] = r.x_rec; v[1] = r.y_rec;
                         v[2] = r.dx_abs; v[3] = r.dy_abs;
                         v[4] = r.dx_signed; v[5] = r.dy_signed;
                         v[6] = r.dx_sq; v[7] = r.dy_sq;
                         return v;
                       }, {"x_hit","y_hit","x_px","y_px","is_pixel_hit","F_i"})
                 ;

  auto hasCol = [&](const char* name){
    auto names = base.GetColumnNames();
    return std::find(names.begin(), names.end(), std::string(name)) != names.end();
  };

  auto node = base;
  node = hasCol("x_rec_2d")                 ? node.Redefine("x_rec_2d",                 "__fit2d__[0]") : node.Define("x_rec_2d",                 "__fit2d__[0]");
  node = hasCol("y_rec_2d")                 ? node.Redefine("y_rec_2d",                 "__fit2d__[1]") : node.Define("y_rec_2d",                 "__fit2d__[1]");
  node = hasCol("rec_hit_delta_x_2d")       ? node.Redefine("rec_hit_delta_x_2d",       "__fit2d__[2]") : node.Define("rec_hit_delta_x_2d",       "__fit2d__[2]");
  node = hasCol("rec_hit_delta_y_2d")       ? node.Redefine("rec_hit_delta_y_2d",       "__fit2d__[3]") : node.Define("rec_hit_delta_y_2d",       "__fit2d__[3]");
  node = hasCol("rec_hit_signed_delta_x_2d")? node.Redefine("rec_hit_signed_delta_x_2d","__fit2d__[4]") : node.Define("rec_hit_signed_delta_x_2d","__fit2d__[4]");
  node = hasCol("rec_hit_signed_delta_y_2d")? node.Redefine("rec_hit_signed_delta_y_2d","__fit2d__[5]") : node.Define("rec_hit_signed_delta_y_2d","__fit2d__[5]");
  node = hasCol("rec_hit_delta_x_sq_2d")    ? node.Redefine("rec_hit_delta_x_sq_2d",    "__fit2d__[6]") : node.Define("rec_hit_delta_x_sq_2d",    "__fit2d__[6]");
  node = hasCol("rec_hit_delta_y_sq_2d")    ? node.Redefine("rec_hit_delta_y_sq_2d",    "__fit2d__[7]") : node.Define("rec_hit_delta_y_sq_2d",    "__fit2d__[7]");

  const std::vector<std::string> cols = {
    "x_rec_2d","y_rec_2d",
    "rec_hit_delta_x_2d","rec_hit_delta_y_2d",
    "rec_hit_signed_delta_x_2d","rec_hit_signed_delta_y_2d",
    "rec_hit_delta_x_sq_2d","rec_hit_delta_y_sq_2d"
  };

  // Snapshot only derived columns to a new file/tree
  auto res = node.Snapshot("Hits_proc2d", outPath.c_str(), cols);
  if (!res) {
    ::Error("processing2D_parallel", "Snapshot failed for output: %s", outPath.c_str());
    return 4;
  }

  ::Info("processing2D_parallel", "Wrote derived columns to %s (tree: Hits_proc2d)", outPath.c_str());
  return 0;
}

