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
#include <TSystem.h>
#include <TEllipse.h>
#include <TPolyLine.h>
#include <TMarker.h>

#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>

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

// Draws the AC-LGAD pixel grid in the detector XY plane using the exact
// placement math used in the simulation. Parameters are read from metadata
// written into the ROOT file by the simulation (master or single-thread).
//
// Usage (from repo root):
//   root -l 'diagnostic/plotPixelGrid.C("epicChargeSharing.root")'
// or simply:
//   root -l diagnostic/plotPixelGrid.C
// which defaults to "epicChargeSharing.root" in the CWD and writes SVG.
void plotPixelGrid(const char* rootFilePath = "epicChargeSharing.root",
                   const char* outImagePath = "pixel_grid.svg",
                   bool saveImage = true)
{
  // Style
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  // Try to open the requested ROOT file first. If unavailable or invalid,
  // try common fallback locations used by this project.
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

  double pixelSizeMm         = 0.0; // pad side length
  double pixelSpacingMm      = 0.0; // center-to-center spacing
  double pixelCornerOffsetMm = 0.0; // gap from detector edge to first pad edge
  double detSizeMm           = 0.0; // detector square size
  int    numPerSide          = 0;   // number of pixels per side

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

  // Sanity checks to ensure perfect consistency with simulation placement math
  // Recompute numPerSide and required detector size using the same formulas
  {
    const double computedNumPerSideReal = std::round((detSizeMm - 2.0*pixelCornerOffsetMm - pixelSizeMm)/pixelSpacingMm + 1.0);
    const int    computedNumPerSide     = static_cast<int>(computedNumPerSideReal);
    if (computedNumPerSide != numPerSide) {
      std::cerr << "WARNING: Metadata inconsistency: computed numPerSide=" << computedNumPerSide
                << " differs from saved numPerSide=" << numPerSide << std::endl;
    }

    const double requiredDetSize = 2.0*pixelCornerOffsetMm + pixelSizeMm + (numPerSide-1)*pixelSpacingMm;
    const double detDiff = std::abs(requiredDetSize - detSizeMm);
    if (detDiff > 1e-9) {
      std::cerr << "WARNING: Metadata inconsistency: required detSize=" << requiredDetSize
                << " mm differs from saved detSize=" << detSizeMm << " mm (diff=" << detDiff << ")" << std::endl;
    }
  }

  // All parameters are in mm and must match simulation math
  const double halfDet = detSizeMm / 2.0;

  // First pixel center position used in simulation:
  // firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  const double firstPixelCenter = -halfDet + pixelCornerOffsetMm + pixelSizeMm/2.0;

  // Canvas and frame with square aspect ratio
  const int canvasSize = 900;
  TCanvas* c = new TCanvas("c_pixel_grid", "", canvasSize, canvasSize);
  // Make the drawable area square: equal margins and fixed aspect ratio
  c->SetLeftMargin(0.12);
  c->SetRightMargin(0.12);
  c->SetTopMargin(0.12);
  c->SetBottomMargin(0.12);
  c->SetFixedAspectRatio();
  c->SetGrid(0,0);

  // Axis frame with proper world coordinates installed on the pad
  // Use DrawFrame to establish the user coordinate system (in mm)
  TH1* frame = c->DrawFrame(-halfDet, -halfDet, halfDet, halfDet, ";x [mm];y [mm]");
  frame->GetXaxis()->SetTitleOffset(1.2);
  frame->GetYaxis()->SetTitleOffset(1.4);
  // Ensure 1:1 scaling at pad level so mm units map equally on both axes
  c->cd();
  gPad->SetFixedAspectRatio();

  // Emphasize and validate the corner-offset parameter visually and numerically
  const double firstPadEdge = firstPixelCenter - pixelSizeMm/2.0; // inner edge of first pad
  const double lastPixelCenter = firstPixelCenter + (numPerSide-1) * pixelSpacingMm;
  const double lastPadEdge = lastPixelCenter + pixelSizeMm/2.0;   // outer edge of last pad

  const double leftEdge   = -halfDet;
  const double rightEdge  =  halfDet;
  const double bottomEdge = -halfDet;
  const double topEdge    =  halfDet;

  const double leftGap   = firstPadEdge - leftEdge;
  const double bottomGap = firstPadEdge - bottomEdge;
  const double rightGap  = rightEdge - lastPadEdge;
  const double topGap    = topEdge - lastPadEdge;

  // Print strict checks to stdout
  auto almostEqual = [](double a, double b, double eps){ return std::abs(a-b) <= eps; };
  const double eps = 1e-9; // mm tolerance
  std::cout << "Corner offset metadata: " << pixelCornerOffsetMm << " mm" << std::endl;
  std::cout << "Gaps (mm): left=" << leftGap << ", right=" << rightGap
            << ", bottom=" << bottomGap << ", top=" << topGap << std::endl;
  if (!almostEqual(leftGap, pixelCornerOffsetMm, eps) ||
      !almostEqual(rightGap, pixelCornerOffsetMm, eps) ||
      !almostEqual(bottomGap, pixelCornerOffsetMm, eps) ||
      !almostEqual(topGap, pixelCornerOffsetMm, eps)) {
    std::cerr << "WARNING: One or more edge gaps differ from corner offset!" << std::endl;
  }

  // Detector outline (square)
  TBox detBox(-halfDet, -halfDet, halfDet, halfDet);
  detBox.SetFillStyle(0);
  detBox.SetLineColor(kBlack);
  detBox.SetLineWidth(2);
  detBox.Draw("same");

  // Draw pixel pads as squares centered at grid locations
  const int colorFill = kAzure + 1;
  const int colorLine = kBlue + 2;
  const double alpha  = 0.35; // semi-transparent fill

  for (int i = 0; i < numPerSide; ++i) {
    const double xCenter = firstPixelCenter + i * pixelSpacingMm;
    for (int j = 0; j < numPerSide; ++j) {
      const double yCenter = firstPixelCenter + j * pixelSpacingMm;

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

  {
    TTree* hitsTree = dynamic_cast<TTree*>(f->Get("Hits"));
    if (!hitsTree) {
      std::cerr << "WARNING: 'Hits' tree not found in ROOT file; skipping hit overlay." << std::endl;
    } else {
      double x_hit = 0.0, y_hit = 0.0;
      Bool_t is_pixel_hit = 0;
      // Speed up by disabling all branches then enabling needed ones
      hitsTree->SetBranchStatus("*", 0);
      hitsTree->SetBranchStatus("TrueX", 1);
      hitsTree->SetBranchStatus("TrueY", 1);
      hitsTree->SetBranchStatus("isPixelHit", 1);
      hitsTree->SetBranchAddress("TrueX", &x_hit);
      hitsTree->SetBranchAddress("TrueY", &y_hit);
      hitsTree->SetBranchAddress("isPixelHit", &is_pixel_hit);

      const double dotRadiusMm = pixelSizeMm / 6.0;
      const int numSegments = 64; // smooth enough for vector outputs
      const int colorHit0 = kRed + 1;
      const int colorHit1 = kGreen + 2;
      const Long64_t nEntries = hitsTree->GetEntries();
      for (Long64_t i = 0; i < nEntries; ++i) {
        hitsTree->GetEntry(i);
        if (!std::isfinite(x_hit) || !std::isfinite(y_hit)) continue;
        if (x_hit < -halfDet || x_hit > halfDet || y_hit < -halfDet || y_hit > halfDet) continue;

        // Build a filled circle polygon to avoid low-segment artifacts in certain outputs
        std::vector<double> xs(numSegments + 1);
        std::vector<double> ys(numSegments + 1);
        for (int k = 0; k <= numSegments; ++k) {
          const double angle = (2.0*M_PI * k) / static_cast<double>(numSegments);
          xs[k] = x_hit + dotRadiusMm * std::cos(angle);
          ys[k] = y_hit + dotRadiusMm * std::sin(angle);
        }
        TPolyLine* circle = new TPolyLine(numSegments + 1, xs.data(), ys.data());
        const int color = is_pixel_hit ? colorHit1 : colorHit0;
        circle->SetFillColor(color);
        circle->SetFillStyle(1001);
        circle->SetLineColor(color);
        circle->SetLineWidth(1);
        circle->Draw("f same");
      }

      // Legend showing classification colors
      TLegend* leg = new TLegend(0.70, 0.82, 0.94, 0.94);
      leg->SetBorderSize(0);
      leg->SetFillStyle(0);
      TMarker* m1 = new TMarker(0, 0, 20); m1->SetMarkerColor(colorHit1); m1->SetMarkerSize(1.2);
      TMarker* m0 = new TMarker(0, 0, 20); m0->SetMarkerColor(colorHit0); m0->SetMarkerSize(1.2);
      leg->AddEntry(m1, "is_pixel_hit = 1", "p");
      leg->AddEntry(m0, "is_pixel_hit = 0", "p");
      leg->Draw();
    }
  }

  // Annotation with parameters
  //std::ostringstream info;
  //info.setf(std::ios::fixed);
  //info.precision(3);
  //info << "Detector: " << detSizeMm << " mm\n"
    //   << "Pixels/side: " << numPerSide << " (total " << numPerSide * numPerSide << ")\n"
    //   << "Pixel size: " << pixelSizeMm << " mm\n"
    //  << "Spacing: " << pixelSpacingMm << " mm\n"
    //   << "Corner offset: " << pixelCornerOffsetMm << " mm";

  //TLatex lat;
  //lat.SetTextSize(0.032);
  //lat.SetNDC(true);
  //lat.DrawLatex(0.16, 0.94, "");
  //lat.SetTextSize(0.028);
  //lat.DrawLatex(0.16, 0.90, info.str().c_str());

  c->Modified();
  c->Update();

  if (saveImage && outImagePath && std::strlen(outImagePath) > 0) {
    c->SaveAs(outImagePath);
  }

  // Clean up file handle (keep canvas open for interactive usage)
  f->Close();
  delete f;
}

// Convenience wrappers so running `root -l -b -q plotHitsOnGrid.C` auto-executes
void plotHitsOnGrid() {
  const char* thisFile = __FILE__;
  std::string thisDir = gSystem->DirName(thisFile);
  std::string defaultPath = thisDir + "/../build/epicChargeSharing.root";
  plotPixelGrid(defaultPath.c_str());
}

void plotHitsOnGrid(const char* filename) {
  plotPixelGrid(filename);
}
