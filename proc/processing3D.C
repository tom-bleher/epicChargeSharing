// ROOT macro: processing3D.C
// Performs 2D Gaussian fit on the full charge neighborhood to reconstruct
// (x_rec_3d, y_rec_3d) and deltas, and appends them as new branches.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TNamed.h>
#include <TGraph2D.h>
#include <TGraph2DErrors.h>
#include <TF2.h>
#include <TROOT.h>
#include <TError.h>
#include <Math/MinimizerOptions.h>
// Minuit2 least-squares API
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <Math/Functor.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
  // 2D Gaussian with constant offset:
  // A * exp(-0.5 * ((x-mux)^2/sigx^2 + (y-muy)^2/sigy^2)) + B
  double Gauss2DPlusB(double* xy, double* p) {
    const double A   = p[0];
    const double mux = p[1];
    const double muy = p[2];
    const double sx  = p[3];
    const double sy  = p[4];
    const double B   = p[5];
    const double dx  = (xy[0] - mux) / sx;
    const double dy  = (xy[1] - muy) / sy;
    return A * std::exp(-0.5 * (dx*dx + dy*dy)) + B;
  }

  inline bool IsFinite(double v) { return std::isfinite(v); }
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%)
// of the event's maximum charge within the neighborhood. The same error is
// applied to all data points used in the fit for that event.
int processing3D(const char* filename = "../build/epicChargeSharingOutput.root",
                 double errorPercentOfMax = 5.0) {
  // Use Minuit2 by default
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
  // Slightly relax tolerance to speed up fits while keeping accuracy
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-5);
  // Limit function evaluations and reduce strategy for speed
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(300);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("processing3D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first for clearer diagnostics)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing3D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharingOutput.root?)", filename);
    file->Close();
    delete file;
    return 3;
  }

  // Fetch metadata (pixel spacing) with fallback to inference from x_px/y_px
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
    ::Error("processing3D", "Pixel spacing not available (metadata missing and inference failed). Aborting.");
    file->Close();
    delete file;
    return 2;
  }

  // Existing branches (inputs)
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;
  // Switch to fitting charge fractions (unitless 0..1)
  std::vector<double>* Fi = nullptr;

  // Speed up I/O: deactivate all branches, then enable only what we read
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("x_hit", 1);
  tree->SetBranchStatus("y_hit", 1);
  tree->SetBranchStatus("x_px", 1);
  tree->SetBranchStatus("y_px", 1);
  tree->SetBranchStatus("is_pixel_hit", 1);
  tree->SetBranchStatus("F_i", 1);

  tree->SetBranchAddress("x_hit", &x_hit);
  tree->SetBranchAddress("y_hit", &y_hit);
  tree->SetBranchAddress("x_px", &x_px);
  tree->SetBranchAddress("y_px", &y_px);
  tree->SetBranchAddress("is_pixel_hit", &is_pixel_hit);
  tree->SetBranchAddress("F_i", &Fi);

  // New branches (outputs).
  // Use NaN so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double x_rec_3d = INVALID_VALUE;
  double y_rec_3d = INVALID_VALUE;
  double rec_hit_delta_x_3d = INVALID_VALUE;
  double rec_hit_delta_y_3d = INVALID_VALUE;

  auto ensureAndResetBranch = [&](const char* name, double* addr) -> TBranch* {
    TBranch* br = tree->GetBranch(name);
    if (!br) {
      br = tree->Branch(name, addr);
    } else {
      tree->SetBranchAddress(name, addr);
      br = tree->GetBranch(name);
      if (br) {
        br->Reset();
        br->DropBaskets();
      }
    }
    return br;
  };

  TBranch* br_x_rec = ensureAndResetBranch("x_rec_3d", &x_rec_3d);
  TBranch* br_y_rec = ensureAndResetBranch("y_rec_3d", &y_rec_3d);
  TBranch* br_dx    = ensureAndResetBranch("rec_hit_delta_x_3d", &rec_hit_delta_x_3d);
  TBranch* br_dy    = ensureAndResetBranch("rec_hit_delta_y_3d", &rec_hit_delta_y_3d);

  // 2D fit function kept for reference. We use Minuit2 on a compact window.
  TF2 f2D("f2D", Gauss2DPlusB, -1e9, 1e9, -1e9, 1e9, 6);

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nProcessed = 0, nFitted = 0;

  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    nProcessed++;

    // Defaults: set to finite invalid sentinel
    x_rec_3d = y_rec_3d = INVALID_VALUE;
    rec_hit_delta_x_3d = rec_hit_delta_y_3d = INVALID_VALUE;

    if (is_pixel_hit || !Fi || Fi->empty()) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      continue;
    }

    const size_t total = Fi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      continue;
    }

    const int R = (N - 1) / 2;

    // Prepare TGraph2D points for valid pixels (Fi >= 0). Also track the max
    TGraph2D g2d;
    int p = 0;
    double zmaxNeighborhood = -1e300;
    for (int di = -R; di <= R; ++di) {
      for (int dj = -R; dj <= R; ++dj) {
        const int idx = (di + R) * N + (dj + R);
        const double f = (*Fi)[idx];
        if (!IsFinite(f) || f < 0) continue; // skip invalid or sentinel
        // Mapping: di indexes X, dj indexes Y
        const double x = x_px + di * pixelSpacing;
        const double y = y_px + dj * pixelSpacing;
        g2d.SetPoint(p++, x, y, f);
        if (f > zmaxNeighborhood) zmaxNeighborhood = f;
      }
    }

    if (g2d.GetN() < 5) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      continue;
    }

    // Initial guesses from data moments (seed around the max point)
    double zmin = 1e300, zmax = -1e300; int idxMax = 0;
    for (int k = 0; k < g2d.GetN(); ++k) {
      const double z = g2d.GetZ()[k];
      if (z < zmin) zmin = z;
      if (z > zmax) { zmax = z; idxMax = k; }
    }
    const double A0 = std::max(1e-18, zmax - zmin);
    const double B0 = std::max(0.0, zmin);
    const double mux0 = g2d.GetX()[idxMax];
    const double muy0 = g2d.GetY()[idxMax];

    // Optional uniform vertical uncertainty from percent-of-max
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (zmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * zmaxNeighborhood
                              : 0.0;

    f2D.SetParameters(A0, mux0, muy0, std::max(0.25*pixelSpacing, 1e-6), std::max(0.25*pixelSpacing, 1e-6), B0);
    // Fractions bounded in [0,1]
    f2D.SetParLimits(0, 0.0, 1.0);                    // A >= 0
    f2D.SetParLimits(3, std::max(1e-6, 0.05*pixelSpacing), 2.0*pixelSpacing);  // tighter sigmas
    f2D.SetParLimits(4, std::max(1e-6, 0.05*pixelSpacing), 2.0*pixelSpacing);
    f2D.SetParLimits(5, 0.0, std::max(0.0, zmin));     // B up to local minimum

    // Fit on FULL range covering all points present in the neighborhood
    double xMinR =  1e300, xMaxR = -1e300;
    double yMinR =  1e300, yMaxR = -1e300;
    for (int k = 0; k < g2d.GetN(); ++k) {
      xMinR = std::min(xMinR, g2d.GetX()[k]);
      xMaxR = std::max(xMaxR, g2d.GetX()[k]);
      yMinR = std::min(yMinR, g2d.GetY()[k]);
      yMaxR = std::max(yMaxR, g2d.GetY()[k]);
    }
    // Add small padding
    xMinR -= 0.5 * pixelSpacing; xMaxR += 0.5 * pixelSpacing;
    yMinR -= 0.5 * pixelSpacing; yMaxR += 0.5 * pixelSpacing;
    f2D.SetRange(xMinR, xMaxR, yMinR, yMaxR);
    // Constrain means inside full span, too
    f2D.SetParLimits(1, xMinR, xMaxR);
    f2D.SetParLimits(2, yMinR, yMaxR);

    // Minuit2 least-squares over the trimmed window
    auto chi2 = [&](const double* p) -> double {
      const double A = p[0];
      const double mx = p[1];
      const double my = p[2];
      const double sx = p[3];
      const double sy = p[4];
      const double B = p[5];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (int k = 0; k < g2d.GetN(); ++k) {
        const double xi = g2d.GetX()[k];
        const double yi = g2d.GetY()[k];
        const double dx = (xi - mx)/sx;
        const double dy = (yi - my)/sy;
        const double model = A * std::exp(-0.5*(dx*dx + dy*dy)) + B;
        const double r = (g2d.GetZ()[k] - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };

    std::unique_ptr<ROOT::Math::Minimizer> m(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
    m->SetMaxFunctionCalls(300);
    m->SetTolerance(1e-5);
    ROOT::Math::Functor fChi2(chi2, 6);
    m->SetFunction(fChi2);

    const double sxInit = std::max(0.25*pixelSpacing, 1e-6);
    const double syInit = std::max(0.25*pixelSpacing, 1e-6);
    const double sLo    = std::max(1e-6, 0.05*pixelSpacing);
    const double sHi    = 2.0*pixelSpacing;

    m->SetLimitedVariable(0, "A", A0, 1e-3, 0.0, 1.0);
    m->SetLimitedVariable(1, "mux", mux0, 1e-4*pixelSpacing, xMinR, xMaxR);
    m->SetLimitedVariable(2, "muy", muy0, 1e-4*pixelSpacing, yMinR, yMaxR);
    m->SetLimitedVariable(3, "sigx", sxInit, 1e-4*pixelSpacing, sLo, sHi);
    m->SetLimitedVariable(4, "sigy", syInit, 1e-4*pixelSpacing, sLo, sHi);
    m->SetLimitedVariable(5, "B", B0, 1e-3, 0.0, std::max(0.0, zmin));

    bool ok = m->Minimize();
    if (ok) {
      const double* X = m->X();
      x_rec_3d = X[1];
      y_rec_3d = X[2];
      rec_hit_delta_x_3d = std::abs(x_hit - x_rec_3d);
      rec_hit_delta_y_3d = std::abs(y_hit - y_rec_3d);
      nFitted++;
    } else {
      // Fallback: weighted centroid over the neighborhood
      double wsum = 0.0, xw = 0.0, yw = 0.0;
      for (int k = 0; k < g2d.GetN(); ++k) {
        double w = std::max(0.0, g2d.GetZ()[k] - B0);
        wsum += w;
        xw += w * g2d.GetX()[k];
        yw += w * g2d.GetY()[k];
      }
      if (wsum > 0) {
        x_rec_3d = xw / wsum;
        y_rec_3d = yw / wsum;
        rec_hit_delta_x_3d = std::abs(x_hit - x_rec_3d);
        rec_hit_delta_y_3d = std::abs(y_hit - y_rec_3d);
        nFitted++;
      } else {
        x_rec_3d = y_rec_3d = INVALID_VALUE;
        rec_hit_delta_x_3d = rec_hit_delta_y_3d = INVALID_VALUE;
      }
    }

    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx->Fill();
    br_dy->Fill();
  }

  // Overwrite tree in file
  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();
  delete file;

  ::Info("processing3D", "Processed %lld entries, fitted %lld.", nProcessed, nFitted);
  return 0;
}

