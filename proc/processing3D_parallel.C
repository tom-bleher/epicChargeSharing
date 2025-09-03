// ROOT macro: processing3D_parallel.C
// Parallel-friendly version of processing3D using RDataFrame with implicit MT.
// - Reads input tree in read-only mode
// - Computes per-entry 2D Gaussian fit of the neighborhood in thread-local lambda
// - Writes only the derived columns to a separate output file/tree

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
    if (gROOT) {
      gROOT->ProcessLine("ROOT::EnableImplicitMT();");
    }
  }

  struct Fit3DResult {
    double x_rec;
    double y_rec;
    double dx_abs;
    double dy_abs;
  };

  Fit3DResult Compute3DFit(
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
    Fit3DResult out{INVALID, INVALID, INVALID, INVALID};
    if (is_pixel_hit || Fi.empty()) return out;

    const size_t total = Fi.size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) return out;
    const int R = (N - 1) / 2;

    struct P { double x,y,z; };
    std::vector<P> pts; pts.reserve(total);
    double zmin = 1e300, zmax = -1e300;
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double f = Fi[idx];
        if (!IsFinite(f) || f < 0) continue;
        const double x = x_px + di * pixelSpacing;
        const double y = y_px + dj * pixelSpacing;
        pts.push_back({x,y,f});
        if (f < zmin) zmin = f;
        if (f > zmax) zmax = f;
      }
    }
    if (pts.size() < 5) return out;

    // Seeds
    auto itMax = std::max_element(pts.begin(), pts.end(), [](const P& a, const P& b){ return a.z < b.z; });
    const double A0 = std::max(1e-18, zmax - zmin);
    const double B0 = std::max(0.0, zmin);
    const double mux0 = itMax->x;
    const double muy0 = itMax->y;

    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (zmax > 0 && relErr > 0.0) ? relErr * zmax : 0.0;

    double xMinR =  1e300, xMaxR = -1e300;
    double yMinR =  1e300, yMaxR = -1e300;
    for (const auto& p : pts) { xMinR = std::min(xMinR, p.x); xMaxR = std::max(xMaxR, p.x); yMinR = std::min(yMinR, p.y); yMaxR = std::max(yMaxR, p.y); }
    xMinR -= 0.5 * pixelSpacing; xMaxR += 0.5 * pixelSpacing;
    yMinR -= 0.5 * pixelSpacing; yMaxR += 0.5 * pixelSpacing;

    auto chi2 = [&](const double* p) -> double {
      const double A = p[0];
      const double mx = p[1];
      const double my = p[2];
      const double sx = p[3];
      const double sy = p[4];
      const double B = p[5];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (const auto& pt : pts) {
        const double dx = (pt.x - mx)/sx;
        const double dy = (pt.y - my)/sy;
        const double model = A * std::exp(-0.5*(dx*dx + dy*dy)) + B;
        const double r = (pt.z - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };

    const double sxInit = std::max(0.25*pixelSpacing, 1e-6);
    const double syInit = std::max(0.25*pixelSpacing, 1e-6);
    const double sLo    = std::max(1e-6, 0.05*pixelSpacing);
    const double sHi    = 2.0*pixelSpacing;

    std::unique_ptr<ROOT::Math::Minimizer> m(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
    m->SetMaxFunctionCalls(300);
    m->SetTolerance(1e-5);
    m->SetStrategy(0);
    ROOT::Math::Functor fChi2(chi2, 6);
    m->SetFunction(fChi2);

    m->SetLimitedVariable(0, "A", A0, 1e-3, 0.0, 1.0);
    // Constrain means to ±1/2 pitch about nearest pixel center
    const double muXLo = x_px - 0.5 * pixelSpacing;
    const double muXHi = x_px + 0.5 * pixelSpacing;
    const double muYLo = y_px - 0.5 * pixelSpacing;
    const double muYHi = y_px + 0.5 * pixelSpacing;
    m->SetLimitedVariable(1, "mux", mux0, 1e-4*pixelSpacing, muXLo, muXHi);
    m->SetLimitedVariable(2, "muy", muy0, 1e-4*pixelSpacing, muYLo, muYHi);
    m->SetLimitedVariable(3, "sigx", sxInit, 1e-4*pixelSpacing, sLo, sHi);
    m->SetLimitedVariable(4, "sigy", syInit, 1e-4*pixelSpacing, sLo, sHi);
    // Ensure non-negative baseline
    m->SetLimitedVariable(5, "B", B0, 1e-3, 0.0, std::max(0.0, zmin));

    bool ok = m->Minimize();
    if (ok) {
      const double* X = m->X();
      out.x_rec = X[1];
      out.y_rec = X[2];
      out.dx_abs = std::abs(x_hit - out.x_rec);
      out.dy_abs = std::abs(y_hit - out.y_rec);
      return out;
    }

    // Fallback: weighted centroid
    double wsum = 0.0, xw = 0.0, yw = 0.0;
    for (const auto& p : pts) {
      double w = std::max(0.0, p.z - B0);
      wsum += w; xw += w * p.x; yw += w * p.y;
    }
    if (wsum > 0) {
      out.x_rec = xw / wsum;
      out.y_rec = yw / wsum;
      out.dx_abs = std::abs(x_hit - out.x_rec);
      out.dy_abs = std::abs(y_hit - out.y_rec);
    }
    return out;
  }
}

int processing3D_parallel(const char* filename = "../build/epicChargeSharingOutput.root",
                          double errorPercentOfMax = 5.0,
                          const char* outFilePath = nullptr)
{
  // Read pixel spacing from metadata or infer
  double pixelSpacing = std::numeric_limits<double>::quiet_NaN();
  {
    TFile* inF = TFile::Open(filename, "READ");
    if (!inF || inF->IsZombie()) {
      ::Error("processing3D_parallel", "Cannot open file: %s", filename);
      if (inF) { inF->Close(); delete inF; }
      return 1;
    }
    if (auto* spacingObj = dynamic_cast<TNamed*>(inF->Get("GridPixelSpacing_mm"))) {
      try { pixelSpacing = std::stod(spacingObj->GetTitle()); } catch (...) { pixelSpacing = std::numeric_limits<double>::quiet_NaN(); }
    }
    if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
      TTree* tree = dynamic_cast<TTree*>(inF->Get("Hits"));
      if (!tree) { inF->Close(); delete inF; return 3; }
      double x_px_tmp = 0.0, y_px_tmp = 0.0;
      tree->SetBranchAddress("x_px", &x_px_tmp);
      tree->SetBranchAddress("y_px", &y_px_tmp);
      std::vector<double> xs; xs.reserve(5000);
      std::vector<double> ys; ys.reserve(5000);
      Long64_t nToScan = std::min<Long64_t>(tree->GetEntries(), 50000);
      for (Long64_t i=0;i<nToScan;++i) { tree->GetEntry(i); if (IsFinite(x_px_tmp)) xs.push_back(x_px_tmp); if (IsFinite(y_px_tmp)) ys.push_back(y_px_tmp); }
      auto computeGap = [](std::vector<double>& v)->double{
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end());
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> gaps; gaps.reserve(v.size());
        for (size_t i=1;i<v.size();++i) { double d = v[i]-v[i-1]; if (d > 1e-9 && IsFinite(d)) gaps.push_back(d); }
        if (gaps.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::nth_element(gaps.begin(), gaps.begin()+gaps.size()/2, gaps.end());
        return gaps[gaps.size()/2]; };
      double gx = computeGap(xs); double gy = computeGap(ys);
      if (IsFinite(gx) && gx>0 && IsFinite(gy) && gy>0) pixelSpacing = 0.5*(gx+gy);
      else if (IsFinite(gx) && gx>0) pixelSpacing = gx; else if (IsFinite(gy) && gy>0) pixelSpacing = gy;
    }
    inF->Close(); delete inF;
  }

  if (!IsFinite(pixelSpacing) || pixelSpacing <= 0) {
    ::Error("processing3D_parallel", "Pixel spacing not available. Aborting.");
    return 2;
  }

  std::string outPath;
  if (outFilePath && std::string(outFilePath).size() > 0) outPath = outFilePath; else {
    std::string in(filename); auto pos = in.rfind('.');
    outPath = (pos == std::string::npos) ? in + ".proc3d.root" : in.substr(0, pos) + ".proc3d.root"; }

  TryEnableImplicitMT();

  ROOT::RDataFrame df("Hits", filename);
  auto base = df.Define("__fit3d__",
                       [pixelSpacing, errorPercentOfMax](double x_hit, double y_hit,
                                                         double x_px,  double y_px,
                                                         bool is_pixel_hit,
                                                         const std::vector<double>& Fi){
                         Fit3DResult r = Compute3DFit(x_hit, y_hit, x_px, y_px, is_pixel_hit, Fi,
                                                      pixelSpacing, errorPercentOfMax);
                         ROOT::RVec<double> v(4);
                         v[0] = r.x_rec; v[1] = r.y_rec; v[2] = r.dx_abs; v[3] = r.dy_abs; return v; },
                       {"x_hit","y_hit","x_px","y_px","is_pixel_hit","F_i"});

  auto hasCol = [&](const char* name){
    auto names = base.GetColumnNames();
    return std::find(names.begin(), names.end(), std::string(name)) != names.end();
  };

  auto node = base;
  node = hasCol("x_rec_3d")           ? node.Redefine("x_rec_3d",           "__fit3d__[0]") : node.Define("x_rec_3d",           "__fit3d__[0]");
  node = hasCol("y_rec_3d")           ? node.Redefine("y_rec_3d",           "__fit3d__[1]") : node.Define("y_rec_3d",           "__fit3d__[1]");
  node = hasCol("rec_hit_delta_x_3d") ? node.Redefine("rec_hit_delta_x_3d", "__fit3d__[2]") : node.Define("rec_hit_delta_x_3d", "__fit3d__[2]");
  node = hasCol("rec_hit_delta_y_3d") ? node.Redefine("rec_hit_delta_y_3d", "__fit3d__[3]") : node.Define("rec_hit_delta_y_3d", "__fit3d__[3]");

  const std::vector<std::string> cols = {"x_rec_3d","y_rec_3d","rec_hit_delta_x_3d","rec_hit_delta_y_3d"};
  auto res = node.Snapshot("Hits_proc3d", outPath.c_str(), cols);
  if (!res) {
    ::Error("processing3D_parallel", "Snapshot failed for output: %s", outPath.c_str());
    return 4;
  }

  ::Info("processing3D_parallel", "Wrote derived columns to %s (tree: Hits_proc3d)", outPath.c_str());
  return 0;
}

