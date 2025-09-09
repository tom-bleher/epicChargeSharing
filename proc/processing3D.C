// ROOT macro: processing3D.C
// Performs 2D Gaussian fit on the full charge neighborhood to reconstruct
// deltas, and appends them as new branches.

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

  inline bool IsFinite3D(double v) { return std::isfinite(v); }
}

// errorPercentOfMax: vertical uncertainty as a percent (e.g. 5.0 means 5%)
// of the event's maximum charge within the neighborhood. The same error is
// applied to all data points used in the fit for that event.
int processing3D(const char* filename = "../build/epicChargeSharing.root",
                 double errorPercentOfMax = 5.0) {
  // Favor faster least-squares: Minuit2 + Fumili2
  ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-4);
  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(400);
  ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
  ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(0);

  // Open file for update
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("processing3D", "Cannot open file: %s", filename);
    return 1;
  }

  // Get tree (check first for clearer diagnostics)
  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("processing3D", "Hits tree not found in file: %s (did you pass the correct path, e.g. build/epicChargeSharing.root?)", filename);
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
    t->SetBranchAddress("PixelX", &x_px_tmp);
    t->SetBranchAddress("PixelY", &y_px_tmp);
    Long64_t nToScan = std::min<Long64_t>(t->GetEntries(), 50000);
    for (Long64_t i=0;i<nToScan;++i) {
      t->GetEntry(i);
      if (IsFinite3D(x_px_tmp)) xs.push_back(x_px_tmp);
      if (IsFinite3D(y_px_tmp)) ys.push_back(y_px_tmp);
    }
    auto computeGap = [](std::vector<double>& v)->double{
      if (v.size() < 2) return NAN;
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      if (v.size() < 2) return NAN;
      std::vector<double> gaps; gaps.reserve(v.size());
      for (size_t i=1;i<v.size();++i) {
        double d = v[i]-v[i-1];
        if (d > 1e-9 && IsFinite3D(d)) gaps.push_back(d);
      }
      if (gaps.empty()) return NAN;
      std::nth_element(gaps.begin(), gaps.begin()+gaps.size()/2, gaps.end());
      return gaps[gaps.size()/2];
    };
    double gx = computeGap(xs);
    double gy = computeGap(ys);
    if (IsFinite3D(gx) && gx>0 && IsFinite3D(gy) && gy>0) return 0.5*(gx+gy);
    if (IsFinite3D(gx) && gx>0) return gx;
    if (IsFinite3D(gy) && gy>0) return gy;
    return NAN;
  };

  if (!IsFinite3D(pixelSpacing) || pixelSpacing <= 0) {
    pixelSpacing = inferSpacingFromTree(tree);
  }
  if (!IsFinite3D(pixelSpacing) || pixelSpacing <= 0) {
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
  tree->SetBranchStatus("TrueX", 1);
  tree->SetBranchStatus("TrueY", 1);
  tree->SetBranchStatus("PixelX", 1);
  tree->SetBranchStatus("PixelY", 1);
  tree->SetBranchStatus("isPixelHit", 1);
  tree->SetBranchStatus("F_i", 1);

  tree->SetBranchAddress("TrueX", &x_hit);
  tree->SetBranchAddress("TrueY", &y_hit);
  tree->SetBranchAddress("PixelX", &x_px);
  tree->SetBranchAddress("PixelY", &y_px);
  tree->SetBranchAddress("isPixelHit", &is_pixel_hit);
  tree->SetBranchAddress("F_i", &Fi);

  // New branches (outputs).
  // Use NaN so invalid/unfitted entries are ignored in ROOT histograms.
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double x_rec_3d = INVALID_VALUE;
  double y_rec_3d = INVALID_VALUE;
  double rec_hit_delta_x_3d = INVALID_VALUE;
  double rec_hit_delta_y_3d = INVALID_VALUE;
  double rec_hit_delta_x_3d_signed = INVALID_VALUE;
  double rec_hit_delta_y_3d_signed = INVALID_VALUE;
  
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

  TBranch* br_x_rec = ensureAndResetBranch("ReconX_3D", &x_rec_3d);
  TBranch* br_y_rec = ensureAndResetBranch("ReconY_3D", &y_rec_3d);
  TBranch* br_dx    = ensureAndResetBranch("ReconTrueDeltaX_3D", &rec_hit_delta_x_3d);
  TBranch* br_dy    = ensureAndResetBranch("ReconTrueDeltaY_3D", &rec_hit_delta_y_3d);
  TBranch* br_dx_signed = ensureAndResetBranch("ReconTrueDeltaX_3D_Signed", &rec_hit_delta_x_3d_signed);
  TBranch* br_dy_signed = ensureAndResetBranch("ReconTrueDeltaY_3D_Signed", &rec_hit_delta_y_3d_signed);
  
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
    rec_hit_delta_x_3d_signed = rec_hit_delta_y_3d_signed = INVALID_VALUE;

    if (is_pixel_hit || !Fi || Fi->empty()) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_dx_signed->Fill();
      br_dy_signed->Fill();
      
      continue;
    }

    const size_t total = Fi->size();
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(total))));
    if (N * N != static_cast<int>(total) || N < 3) {
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_dx_signed->Fill();
      br_dy_signed->Fill();
      
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
        if (!IsFinite3D(f) || f < 0) continue; // skip invalid or sentinel
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

    // Low-contrast neighborhood: skip iterative fit, use weighted centroid
    if (A0 < 1e-6) {
      double wsum = 0.0, xw = 0.0, yw = 0.0;
      for (int k = 0; k < g2d.GetN(); ++k) {
        const double w = std::max(0.0, g2d.GetZ()[k] - B0);
        wsum += w;
        xw += w * g2d.GetX()[k];
        yw += w * g2d.GetY()[k];
      }
      if (wsum > 0.0) {
        x_rec_3d = xw / wsum;
        y_rec_3d = yw / wsum;
        rec_hit_delta_x_3d = std::abs(x_hit - x_rec_3d);
        rec_hit_delta_y_3d = std::abs(y_hit - y_rec_3d);
        rec_hit_delta_x_3d_signed = (x_hit - x_rec_3d);
        rec_hit_delta_y_3d_signed = (y_hit - y_rec_3d);
        nFitted++;
      }
      br_x_rec->Fill();
      br_y_rec->Fill();
      br_dx->Fill();
      br_dy->Fill();
      br_dx_signed->Fill();
      br_dy_signed->Fill();
      continue;
    }

    // Optional uniform vertical uncertainty from percent-of-max
    const double relErr = std::max(0.0, errorPercentOfMax) * 0.01;
    const double uniformSigma = (zmaxNeighborhood > 0 && relErr > 0.0)
                              ? relErr * zmaxNeighborhood
                              : 0.0;

    // Seed sigmas from baseline-subtracted second moments; loosen bounds
    const double sigLoBound = std::max(1e-6, 0.02*pixelSpacing);
    const double sigHiBound = 3.0*pixelSpacing;

    auto sigmaSeed2D = [&](bool forX)->double {
      double wsum = 0.0, m = 0.0;
      const int n = g2d.GetN();
      if (n <= 0) return std::max(0.25*pixelSpacing, 1e-6);
      // mean
      for (int k=0;k<n;++k) {
        const double w = std::max(0.0, g2d.GetZ()[k] - B0);
        const double coord = forX ? g2d.GetX()[k] : g2d.GetY()[k];
        wsum += w; m += w * coord;
      }
      if (wsum <= 0.0) return std::max(0.25*pixelSpacing, 1e-6);
      m /= wsum;
      double var = 0.0;
      for (int k=0;k<n;++k) {
        const double w = std::max(0.0, g2d.GetZ()[k] - B0);
        const double coord = forX ? g2d.GetX()[k] : g2d.GetY()[k];
        const double d = coord - m;
        var += w * d * d;
      }
      var = (wsum > 0.0) ? (var / wsum) : 0.0;
      double s = std::sqrt(std::max(var, 1e-12));
      if (s < sigLoBound) s = sigLoBound;
      if (s > sigHiBound) s = sigHiBound;
      return s;
    };

    const double sxInitMoment = sigmaSeed2D(true);
    const double syInitMoment = sigmaSeed2D(false);

    // Seed params retained for reference TF2, but Minimizer constraints used below
    f2D.SetParameters(A0, mux0, muy0, sxInitMoment, syInitMoment, B0);

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
    // Constrain means to ±1/2 pitch about nearest pixel center
    const double muXLo = x_px - 0.5 * pixelSpacing;
    const double muXHi = x_px + 0.5 * pixelSpacing;
    const double muYLo = y_px - 0.5 * pixelSpacing;
    const double muYHi = y_px + 0.5 * pixelSpacing;
    f2D.SetParLimits(1, muXLo, muXHi);
    f2D.SetParLimits(2, muYLo, muYHi);

    // Minuit2 least-squares over the trimmed window
    const double* Xarr = g2d.GetX();
    const double* Yarr = g2d.GetY();
    const double* Zarr = g2d.GetZ();
    const int nPts = g2d.GetN();
    auto chi2 = [&](const double* p) -> double {
      const double A = p[0];
      const double mx = p[1];
      const double my = p[2];
      const double sx = p[3];
      const double sy = p[4];
      const double B = p[5];
      const double invVar = (uniformSigma > 0.0) ? 1.0/(uniformSigma*uniformSigma) : 1.0;
      double sum = 0.0;
      for (int k = 0; k < nPts; ++k) {
        const double xi = Xarr[k];
        const double yi = Yarr[k];
        const double dx = (xi - mx)/sx;
        const double dy = (yi - my)/sy;
        const double model = A * std::exp(-0.5*(dx*dx + dy*dy)) + B;
        const double r = (Zarr[k] - model);
        sum += (uniformSigma > 0.0) ? (r*r*invVar) : (r*r);
      }
      return sum;
    };

    // Wrap model as TF1 for ROOT::Fit pipeline
    TF2 fModel("fModel", Gauss2DPlusB, xMinR, xMaxR, yMinR, yMaxR, 6);
    ROOT::Math::WrappedMultiTF1 wModel(fModel, 2);

    // Build BinData (Fumili2 requires FitMethodFunction via Chi2Function)
    ROOT::Fit::BinData data2D(nPts, 2);
    for (int k = 0; k < nPts; ++k) {
      const double ey = (uniformSigma > 0.0) ? uniformSigma : 1.0;
      double xy[2] = {Xarr[k], Yarr[k]};
      data2D.Add(xy, Zarr[k], ey);
    }
    ROOT::Fit::Fitter fitter;
    fitter.Config().SetMinimizer("Minuit2", "Fumili2");
    fitter.Config().MinimizerOptions().SetStrategy(0);
    fitter.Config().MinimizerOptions().SetTolerance(1e-4);
    fitter.Config().MinimizerOptions().SetPrintLevel(0);
    // Provide the model; Chi2 will be constructed from BinData internally
    fitter.SetFunction(wModel);

    // Parameter settings and limits
    fitter.Config().ParSettings(0).SetName("A");
    fitter.Config().ParSettings(1).SetName("mux");
    fitter.Config().ParSettings(2).SetName("muy");
    fitter.Config().ParSettings(3).SetName("sigx");
    fitter.Config().ParSettings(4).SetName("sigy");
    fitter.Config().ParSettings(5).SetName("B");
    fitter.Config().ParSettings(0).SetLowerLimit(0.0);
    fitter.Config().ParSettings(1).SetLimits(muXLo, muXHi);
    fitter.Config().ParSettings(2).SetLimits(muYLo, muYHi);
    fitter.Config().ParSettings(3).SetLimits(sigLoBound, sigHiBound);
    fitter.Config().ParSettings(4).SetLimits(sigLoBound, sigHiBound);
    fitter.Config().ParSettings(5).SetLowerLimit(0.0);

    fitter.Config().ParSettings(0).SetStepSize(1e-3);
    fitter.Config().ParSettings(1).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(2).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(3).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(4).SetStepSize(1e-4*pixelSpacing);
    fitter.Config().ParSettings(5).SetStepSize(1e-3);

    // Initial parameters (via ParSettings)
    fitter.Config().ParSettings(0).SetValue(A0);
    fitter.Config().ParSettings(1).SetValue(mux0);
    fitter.Config().ParSettings(2).SetValue(muy0);
    fitter.Config().ParSettings(3).SetValue(sxInitMoment);
    fitter.Config().ParSettings(4).SetValue(syInitMoment);
    fitter.Config().ParSettings(5).SetValue(B0);

    bool ok = fitter.Fit(data2D);
    if (ok) {
      x_rec_3d = fitter.Result().Parameter(1);
      y_rec_3d = fitter.Result().Parameter(2);
      rec_hit_delta_x_3d = std::abs(x_hit - x_rec_3d);
      rec_hit_delta_y_3d = std::abs(y_hit - y_rec_3d);
      rec_hit_delta_x_3d_signed = (x_hit - x_rec_3d);
      rec_hit_delta_y_3d_signed = (y_hit - y_rec_3d);
      
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
        rec_hit_delta_x_3d_signed = (x_hit - x_rec_3d);
        rec_hit_delta_y_3d_signed = (y_hit - y_rec_3d);
        
        nFitted++;
      } else {
        x_rec_3d = y_rec_3d = INVALID_VALUE;
        rec_hit_delta_x_3d = rec_hit_delta_y_3d = INVALID_VALUE;
        rec_hit_delta_x_3d_signed = rec_hit_delta_y_3d_signed = INVALID_VALUE;
        
      }
    }

    br_x_rec->Fill();
    br_y_rec->Fill();
    br_dx->Fill();
    br_dy->Fill();
    br_dx_signed->Fill();
    br_dy_signed->Fill();
    
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

