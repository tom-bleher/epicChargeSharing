#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TBox.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TPolyLine.h>
#include <TRandom3.h>
#include <TColor.h>
// TPaletteAxis forward-declared via Hist headers; no extra include needed

#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
  double ReadDoubleNamed(TFile* file, const char* key)
  {
    if (!file) throw std::runtime_error("Null TFile in ReadDoubleNamed");
    TObject* obj = file->Get(key);
    if (!obj) {
      std::ostringstream oss; oss << "Missing metadata object: '" << key << "'"; 
      throw std::runtime_error(oss.str());
    }
    TNamed* named = dynamic_cast<TNamed*>(obj);
    if (!named) {
      std::ostringstream oss; oss << "Object '" << key << "' is not a TNamed"; 
      throw std::runtime_error(oss.str());
    }
    const char* s = named->GetTitle();
    if (!s) {
      std::ostringstream oss; oss << "TNamed '" << key << "' has empty title"; 
      throw std::runtime_error(oss.str());
    }
    return std::atof(s);
  }

  int ReadIntNamed(TFile* file, const char* key)
  {
    double v = ReadDoubleNamed(file, key);
    return static_cast<int>(std::lround(v));
  }
}

// Selects a random 3x3 pixel region, overlays non-pixel hits, and renders
// comparisons between 2D and 3D recon:
//  (A) |mean3D - mean2D| and signed (mean3D - mean2D), where meanN = 0.5(|dx_N|+|dy_N|)
//  (B) Per-axis diffs: abs and signed for X and Y using |d?_3D| and |d?_2D|
// Absolute diffs use [0,max]; signed diffs use symmetric [-max,+max].
//
// Usage (from repo root):
//   root -l 'diagnostic/plotRandom3x3_2Dvs3D.C("epicChargeSharing.root")'
// or simply:
//   root -l diagnostic/plotRandom3x3_2Dvs3D.C
void plotRandom3x3_2Dvs3DCore(const char* rootFilePath = "epicChargeSharing.root",
                       const char* outImagePath = "random3x3_2dvs3d.svg",
                       bool saveImage = true,
                       Long64_t seed = 0,
                       int forceI0 = -1,
                       int forceJ0 = -1)
{
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  auto tryOpen = [](const char* path) -> TFile* {
    if (!path || !*path) return nullptr;
    if (gSystem->AccessPathName(path, kReadPermission)) return nullptr;
    TFile* tmp = TFile::Open(path, "READ");
    if (!tmp || tmp->IsZombie()) { if (tmp) { tmp->Close(); delete tmp; } return nullptr; }
    return tmp;
  };

  TFile* f = tryOpen(rootFilePath);
  if (!f) f = tryOpen("build/epicChargeSharing.root");
  if (!f) f = tryOpen("../build/epicChargeSharing.root");
  if (!f) f = tryOpen("../epicChargeSharing.root");
  if (!f) {
    std::cerr << "ERROR: Cannot open any ROOT file (tried provided path and common fallbacks)." << std::endl;
    return;
  }

  double pixelSizeMm         = 0.0;
  double pixelSpacingMm      = 0.0; // pitch
  double pixelCornerOffsetMm = 0.0;
  double detSizeMm           = 0.0;
  int    numPerSide          = 0;

  try {
    pixelSizeMm         = ReadDoubleNamed(f, "GridPixelSize_mm");
    pixelSpacingMm      = ReadDoubleNamed(f, "GridPixelSpacing_mm");
    pixelCornerOffsetMm = ReadDoubleNamed(f, "GridPixelCornerOffset_mm");
    detSizeMm           = ReadDoubleNamed(f, "GridDetectorSize_mm");
    numPerSide          = ReadIntNamed  (f, "GridNumBlocksPerSide");
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    f->Close();
    delete f;
    return;
  }

  const double halfDet = detSizeMm / 2.0;
  const double firstPixelCenter = -halfDet + pixelCornerOffsetMm + pixelSizeMm/2.0;
  const double halfPitch = 0.5 * pixelSpacingMm; // e.g. 0.25 mm when pitch=0.5 mm

  // Choose a random 3x3 block (indices i0..i2 and j0..j2)
  TRandom3 rng;
  if (seed == 0) seed = static_cast<Long64_t>(gSystem->Now());
  rng.SetSeed(seed);

  int i0 = (forceI0 >= 0 && forceI0 <= numPerSide-3) ? forceI0 : rng.Integer(numPerSide-2);
  int j0 = (forceJ0 >= 0 && forceJ0 <= numPerSide-3) ? forceJ0 : rng.Integer(numPerSide-2);
  const int i1 = i0 + 1;
  const int i2 = i0 + 2;
  const int j1 = j0 + 1;
  const int j2 = j0 + 2;

  const double xC0 = firstPixelCenter + i0 * pixelSpacingMm;
  const double xC1 = firstPixelCenter + i1 * pixelSpacingMm;
  const double xC2 = firstPixelCenter + i2 * pixelSpacingMm;
  const double yC0 = firstPixelCenter + j0 * pixelSpacingMm;
  const double yC1 = firstPixelCenter + j1 * pixelSpacingMm;
  const double yC2 = firstPixelCenter + j2 * pixelSpacingMm;

  // Rectangle borders at half-pitch from each of the outer centers, clipped to detector
  double xMin = std::max(-halfDet, xC0 - halfPitch);
  double xMax = std::min( halfDet, xC2 + halfPitch);
  double yMin = std::max(-halfDet, yC0 - halfPitch);
  double yMax = std::min( halfDet, yC2 + halfPitch);

  // Helper to draw base scene (frame, pads, window)
  auto drawBase = [&](const char* cname) -> TCanvas* {
    const int canvasSize = 900;
    TCanvas* canvas = new TCanvas(cname, "", canvasSize, canvasSize);
    canvas->SetLeftMargin(0.12);
    canvas->SetRightMargin(0.15); // like ROOT tutorial; reserve a gutter for palette
    canvas->SetTopMargin(0.15);
    canvas->SetBottomMargin(0.12);
    canvas->SetFixedAspectRatio();
    canvas->SetGrid(0,0);

    TH1* fr = canvas->DrawFrame(xMin, yMin, xMax, yMax, ";x [mm];y [mm]");
    fr->GetXaxis()->SetTitleOffset(1.2);
    fr->GetYaxis()->SetTitleOffset(1.4);
    canvas->cd();
    gPad->SetFixedAspectRatio();

    const int colorFill = kAzure + 1;
    const int colorLine = kBlue + 2;
    const double alpha  = 0.35;
    for (int ii = 0; ii < 3; ++ii) {
      const double xCenter = (ii == 0) ? xC0 : (ii == 1 ? xC1 : xC2);
      for (int jj = 0; jj < 3; ++jj) {
        const double yCenter = (jj == 0) ? yC0 : (jj == 1 ? yC1 : yC2);
        const double x1 = xCenter - pixelSizeMm/2.0;
        const double x2 = xCenter + pixelSizeMm/2.0;
        const double y1 = yCenter - pixelSizeMm/2.0;
        const double y2 = yCenter + pixelSizeMm/2.0;
        TBox* px = new TBox(x1, y1, x2, y2);
        px->SetFillColorAlpha(colorFill, alpha);
        px->SetLineColor(colorLine);
        px->SetLineWidth(1);
        px->Draw("same");
      }
    }

    TBox windowBox(xMin, yMin, xMax, yMax);
    windowBox.SetFillStyle(0);
    windowBox.SetLineColor(kBlack);
    windowBox.SetLineWidth(3);
    windowBox.Draw("same");
    return canvas;
  };

  // Pull hits and compute metrics
  std::vector<double> xs; xs.reserve(8192);
  std::vector<double> ys; ys.reserve(8192);
  std::vector<double> mean2d; mean2d.reserve(8192);
  std::vector<double> mean3d; mean3d.reserve(8192);
  std::vector<double> diffAbs; diffAbs.reserve(8192);
  std::vector<double> diffSigned; diffSigned.reserve(8192);
  std::vector<double> diffAbsX; diffAbsX.reserve(8192);
  std::vector<double> diffSignedX; diffSignedX.reserve(8192);
  std::vector<double> diffAbsY; diffAbsY.reserve(8192);
  std::vector<double> diffSignedY; diffSignedY.reserve(8192);

  {
    TTree* hitsTree = dynamic_cast<TTree*>(f->Get("Hits"));
    if (!hitsTree) {
      std::cerr << "WARNING: 'Hits' tree not found in ROOT file; skipping hit overlay." << std::endl;
    } else {
      double x_hit = 0.0, y_hit = 0.0;
      double dtx2d = 0.0, dty2d = 0.0; // ReconTrueDeltaX/Y (2D)
      double dtx3d = 0.0, dty3d = 0.0; // ReconTrueDeltaX/Y (3D)
      Bool_t is_pixel_hit = 0;
      hitsTree->SetBranchStatus("*", 0);
      hitsTree->SetBranchStatus("TrueX", 1);
      hitsTree->SetBranchStatus("TrueY", 1);
      hitsTree->SetBranchStatus("isPixelHit", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaX", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaY", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaX_3D", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaY_3D", 1);
      hitsTree->SetBranchAddress("TrueX", &x_hit);
      hitsTree->SetBranchAddress("TrueY", &y_hit);
      hitsTree->SetBranchAddress("isPixelHit", &is_pixel_hit);
      hitsTree->SetBranchAddress("ReconTrueDeltaX", &dtx2d);
      hitsTree->SetBranchAddress("ReconTrueDeltaY", &dty2d);
      hitsTree->SetBranchAddress("ReconTrueDeltaX_3D", &dtx3d);
      hitsTree->SetBranchAddress("ReconTrueDeltaY_3D", &dty3d);

      const Long64_t nEntries = hitsTree->GetEntries();
      for (Long64_t i = 0; i < nEntries; ++i) {
        hitsTree->GetEntry(i);
        if (!std::isfinite(x_hit) || !std::isfinite(y_hit)) continue;
        if (x_hit < xMin || x_hit > xMax || y_hit < yMin || y_hit > yMax) continue;
        if (is_pixel_hit) continue; // only non-pixel hits
        xs.push_back(x_hit);
        ys.push_back(y_hit);
        const double adx2 = std::abs(dtx2d);
        const double ady2 = std::abs(dty2d);
        const double adx3 = std::abs(dtx3d);
        const double ady3 = std::abs(dty3d);
        const double m2 = 0.5*(adx2 + ady2);
        const double m3 = 0.5*(adx3 + ady3);
        mean2d.push_back(m2);
        mean3d.push_back(m3);
        diffAbs.push_back(std::abs(m3 - m2));
        diffSigned.push_back(m3 - m2);
        // Per-axis differences between absolute deltas
        diffAbsX.push_back(std::abs(adx3 - adx2));
        diffSignedX.push_back(adx3 - adx2);
        diffAbsY.push_back(std::abs(ady3 - ady2));
        diffSignedY.push_back(ady3 - ady2);
      }
    }
  }

  auto computeRangeAbs = [&](const std::vector<double>& v, double fallbackMax) {
    double vmax = v.empty() ? fallbackMax : v.front();
    for (size_t i = 1; i < v.size(); ++i) { if (v[i] > vmax) vmax = v[i]; }
    if (vmax < 1e-12) vmax = fallbackMax; // fallback
    return std::pair<double,double>(0.0, vmax);
  };

  auto computeRangeSignedSym = [&](const std::vector<double>& v, double fallbackAbsMax) {
    double amax = v.empty() ? fallbackAbsMax : std::abs(v.front());
    for (size_t i = 1; i < v.size(); ++i) { double a = std::abs(v[i]); if (a > amax) amax = a; }
    if (amax < 1e-12) amax = fallbackAbsMax;
    return std::pair<double,double>(-amax, amax);
  };

  const auto diffAbsRange = computeRangeAbs(diffAbs, halfPitch);
  const auto diffSignedRange = computeRangeSignedSym(diffSigned, halfPitch);
  const auto diffAbsXRange = computeRangeAbs(diffAbsX, halfPitch);
  const auto diffSignedXRange = computeRangeSignedSym(diffSignedX, halfPitch);
  const auto diffAbsYRange = computeRangeAbs(diffAbsY, halfPitch);
  const auto diffSignedYRange = computeRangeSignedSym(diffSignedY, halfPitch);

  auto drawColoredPoints = [&](const char* cname, const char* zTitle,
                               const std::vector<double>& zs,
                               double zmin, double zmax,
                               const char* outfile) {
    TCanvas* c = drawBase(cname);
    const double dotRadiusMm = pixelSizeMm / 6.0;
    const int numSegments = 36;
    const int ncolors = TColor::GetNumberOfColors();

    // Create a dummy off-frame histogram just to get a palette with tick marks
    // We draw it with COLZ so ROOT creates a TPaletteAxis on the right margin.
    TH2D* paletteHist = new TH2D(Form("pal_%s", cname), "", 2, xMax+1, xMax+2, 2, yMin, yMax);
    paletteHist->SetMinimum(zmin);
    paletteHist->SetMaximum(zmax);
    paletteHist->GetZaxis()->SetTitle(zTitle);
    paletteHist->GetZaxis()->SetTitleSize(0.040);
    paletteHist->GetZaxis()->SetLabelSize(0.032);
    paletteHist->Draw("COLZ SAME");
    gPad->Update();
    TPaletteAxis* pal = (TPaletteAxis*)paletteHist->GetListOfFunctions()->FindObject("palette");
    if (pal) {
      pal->SetLabelSize(0.032);
      pal->SetTitleOffset(1.30);
      pal->SetBorderSize(1);
      // Place palette fully inside the right margin like hist102_TH2_contour_list.C
      const double rm = gPad->GetRightMargin();
      const double tm = gPad->GetTopMargin();
      const double bm = gPad->GetBottomMargin();
      const double x1 = std::min(0.99, 1.0 - rm + 0.005);
      const double x2 = std::min(0.99, x1 + 0.050);
      const double y1 = std::max(0.0, bm + 0.02);
      const double y2 = std::min(0.99, 1.0 - tm - 0.02);
      pal->SetX1NDC(x1);
      pal->SetX2NDC(x2);
      pal->SetY1NDC(y1);
      pal->SetY2NDC(y2);
    }

    // Draw colored hit markers
    for (size_t i = 0; i < xs.size() && i < zs.size(); ++i) {
      const double val = zs[i];
      double t = (val - zmin) / (zmax - zmin);
      if (t < 0.0) t = 0.0; if (t > 1.0) t = 1.0;
      const int ci = TColor::GetColorPalette(std::lround(t * (ncolors - 1)));

      std::vector<double> px(numSegments + 1);
      std::vector<double> py(numSegments + 1);
      for (int k = 0; k <= numSegments; ++k) {
        const double angle = (2.0*M_PI * k) / static_cast<double>(numSegments);
        px[k] = xs[i] + dotRadiusMm * std::cos(angle);
        py[k] = ys[i] + dotRadiusMm * std::sin(angle);
      }
      TPolyLine* circle = new TPolyLine(numSegments + 1, px.data(), py.data());
      circle->SetFillColor(ci);
      circle->SetFillStyle(1001);
      circle->SetLineColor(ci);
      circle->SetLineWidth(1);
      circle->Draw("f same");
    }

    // Minimal annotation
    TLatex lat;
    lat.SetTextSize(0.028);
    lat.SetNDC(true);
    lat.DrawLatex(0.16, 0.92, "");

    c->Modified();
    c->Update();
    if (saveImage && outfile && std::strlen(outfile) > 0) c->SaveAs(outfile);
  };

  // Derive output file name base
  std::string base = outImagePath ? outImagePath : "random3x3_2dvs3d.svg";
  std::string ext = ".svg";
  {
    size_t slash = base.find_last_of('/');
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
      ext = base.substr(dot);
      base = base.substr(0, dot);
    }
  }
  std::string outDiffAbs = base + "_diffabs" + ext;
  std::string outDiffSigned = base + "_diffsigned" + ext;
  std::string outDxDiffAbs = base + "_dx_diffabs" + ext;
  std::string outDxDiffSigned = base + "_dx_diffsigned" + ext;
  std::string outDyDiffAbs = base + "_dy_diffabs" + ext;
  std::string outDyDiffSigned = base + "_dy_diffsigned" + ext;

  // Draw with appropriate ranges
  drawColoredPoints("c_rand3x3_diff_abs",
                    "|mean_{3D} - mean_{2D}| [mm]",
                    diffAbs, diffAbsRange.first, diffAbsRange.second, outDiffAbs.c_str());

  drawColoredPoints("c_rand3x3_diff_signed",
                    "mean_{3D} - mean_{2D} [mm]",
                    diffSigned, diffSignedRange.first, diffSignedRange.second, outDiffSigned.c_str());

  // Per-axis: X differences
  drawColoredPoints("c_rand3x3_dx_diff_abs",
                    "abs(|#Delta x_{3D}| - |#Delta x|) [mm]",
                    diffAbsX, diffAbsXRange.first, diffAbsXRange.second, outDxDiffAbs.c_str());

  drawColoredPoints("c_rand3x3_dx_diff_signed",
                    "|#Delta x_{3D}| - |#Delta x| [mm]",
                    diffSignedX, diffSignedXRange.first, diffSignedXRange.second, outDxDiffSigned.c_str());

  // Per-axis: Y differences
  drawColoredPoints("c_rand3x3_dy_diff_abs",
                    "abs(|#Delta y_{3D}| - |#Delta y|) [mm]",
                    diffAbsY, diffAbsYRange.first, diffAbsYRange.second, outDyDiffAbs.c_str());

  drawColoredPoints("c_rand3x3_dy_diff_signed",
                    "|#Delta y_{3D}| - |#Delta y| [mm]",
                    diffSignedY, diffSignedYRange.first, diffSignedYRange.second, outDyDiffSigned.c_str());

  f->Close();
  delete f;
}

// Convenience wrappers so running `root -l -b -q plotRandom3x3_2Dvs3D.C` auto-executes
void plotRandom3x3_2Dvs3D() {
  const char* thisFile = __FILE__;
  std::string thisDir = gSystem->DirName(thisFile);
  std::string defaultPath = thisDir + "/../../../build/epicChargeSharing.root";
  plotRandom3x3_2Dvs3DCore(defaultPath.c_str());
}

void plotRandom3x3_2Dvs3D(const char* filename) {
  plotRandom3x3_2Dvs3DCore(filename, "random3x3_2dvs3d.svg", true);
}




