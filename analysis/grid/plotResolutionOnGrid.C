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

// Draws the AC-LGAD pixel grid in the detector XY plane and overlays ONLY
// non-pixel hits colored by reconstruction resolution using a palette bar.
// The geometry is taken from metadata written by the simulation so all
// placement exactly matches it.
//
// Usage (from repo root):
//   root -l 'diagnostic/plotResolutionOnGrid.C("epicChargeSharing.root")'
// or simply:
//   root -l diagnostic/plotResolutionOnGrid.C
//
// "metric" can be one of: "dx", "dy", "dmean" (default).
void plotResolutionOnGridCore(const char* rootFilePath = "epicChargeSharing.root",
                              const char* outImagePath = "resolution_on_grid.svg",
                              bool saveImage = true,
                              const char* metric = "dmean")
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

  const int canvasSize = 900;
  TCanvas* c = new TCanvas("c_resolution_grid", "", canvasSize, canvasSize);
  c->SetLeftMargin(0.12);
  c->SetRightMargin(0.12);
  c->SetTopMargin(0.12);
  c->SetBottomMargin(0.12);
  c->SetFixedAspectRatio();
  c->SetGrid(0,0);

  TH1* frame = c->DrawFrame(-halfDet, -halfDet, halfDet, halfDet, ";x [mm];y [mm]");
  frame->GetXaxis()->SetTitleOffset(1.2);
  frame->GetYaxis()->SetTitleOffset(1.4);
  c->cd();
  gPad->SetFixedAspectRatio();

  // Detector outline
  {
    TBox detBox(-halfDet, -halfDet, halfDet, halfDet);
    detBox.SetFillStyle(0);
    detBox.SetLineColor(kBlack);
    detBox.SetLineWidth(2);
    detBox.Draw("same");
  }

  // Draw all pixel pads
  {
    const int colorFill = kAzure + 1;
    const int colorLine = kBlue + 2;
    const double alpha  = 0.35;
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
  }

  // Collect non-pixel hits and compute coloring metric
  std::vector<double> xs; xs.reserve(32768);
  std::vector<double> ys; ys.reserve(32768);
  std::vector<double> zs; zs.reserve(32768);

  {
    TTree* hitsTree = dynamic_cast<TTree*>(f->Get("Hits"));
    if (!hitsTree) {
      std::cerr << "WARNING: 'Hits' tree not found in ROOT file; skipping hit overlay." << std::endl;
    } else {
      double x_hit = 0.0, y_hit = 0.0;
      double dtx = 0.0, dty = 0.0; // ReconTrueDeltaX/Y
      Bool_t is_pixel_hit = 0;
      hitsTree->SetBranchStatus("*", 0);
      hitsTree->SetBranchStatus("TrueX", 1);
      hitsTree->SetBranchStatus("TrueY", 1);
      hitsTree->SetBranchStatus("isPixelHit", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaX", 1);
      hitsTree->SetBranchStatus("ReconTrueDeltaY", 1);
      hitsTree->SetBranchAddress("TrueX", &x_hit);
      hitsTree->SetBranchAddress("TrueY", &y_hit);
      hitsTree->SetBranchAddress("isPixelHit", &is_pixel_hit);
      hitsTree->SetBranchAddress("ReconTrueDeltaX", &dtx);
      hitsTree->SetBranchAddress("ReconTrueDeltaY", &dty);

      const bool useDx    = (metric && std::strcmp(metric, "dx") == 0);
      const bool useDy    = (metric && std::strcmp(metric, "dy") == 0);
      const bool useDmean = (!useDx && !useDy); // default

      const Long64_t nEntries = hitsTree->GetEntries();
      for (Long64_t i = 0; i < nEntries; ++i) {
        hitsTree->GetEntry(i);
        if (!std::isfinite(x_hit) || !std::isfinite(y_hit)) continue;
        if (x_hit < -halfDet || x_hit > halfDet || y_hit < -halfDet || y_hit > halfDet) continue;
        if (is_pixel_hit) continue; // only non-pixel hits

        xs.push_back(x_hit);
        ys.push_back(y_hit);

        double zval = 0.0;
        if (useDx) {
          zval = std::abs(dtx);
        } else if (useDy) {
          zval = std::abs(dty);
        } else {
          zval = 0.5 * (std::abs(dtx) + std::abs(dty));
        }
        zs.push_back(zval);
      }
    }
  }

  // Determine color scale range [0, zmax]
  double zmin = 0.0;
  double zmax = 0.0;
  if (!zs.empty()) {
    for (size_t i = 0; i < zs.size(); ++i) if (zs[i] > zmax) zmax = zs[i];
  }
  if (zmax <= 1e-12) {
    // Fallback: use half pitch as a reasonable upper scale
    zmax = 0.5 * pixelSpacingMm;
  }

  // Create a palette axis on the right margin
  TH2D* paletteHist = nullptr;
  {
    paletteHist = new TH2D("pal_res_grid", "", 2, halfDet + 1.0, halfDet + 2.0, 2, -halfDet, halfDet);
    paletteHist->SetMinimum(zmin);
    paletteHist->SetMaximum(zmax);
    const char* zTitle = (metric && std::strcmp(metric, "dx") == 0) ? "|#Delta x| [mm]" :
                         (metric && std::strcmp(metric, "dy") == 0) ? "|#Delta y| [mm]" :
                         "(|#Delta x|+|#Delta y|)/2 [mm]";
    paletteHist->GetZaxis()->SetTitle(zTitle);
    paletteHist->GetZaxis()->SetTitleSize(0.040);
    paletteHist->GetZaxis()->SetLabelSize(0.032);
    paletteHist->Draw("COLZ SAME");
    gPad->Update();
    TPaletteAxis* pal = (TPaletteAxis*)paletteHist->GetListOfFunctions()->FindObject("palette");
    if (pal) {
      pal->SetLabelSize(0.032);
      paletteHist->GetZaxis()->SetLabelOffset(0.006);
      paletteHist->GetZaxis()->SetTitleSize(0.040);
      pal->SetTitleOffset(1.35);
      pal->SetBorderSize(1);
      const double rm = gPad->GetRightMargin();
      const double tm = gPad->GetTopMargin();
      const double bm = gPad->GetBottomMargin();
      const double x2 = 0.99;
      const double x1 = std::max(1.0 - rm + 0.006, x2 - 0.045);
      const double y1 = std::max(0.0, bm + 0.02);
      const double y2 = std::min(0.99, 1.0 - tm - 0.02);
      pal->SetX1NDC(x1);
      pal->SetX2NDC(x2);
      pal->SetY1NDC(y1);
      pal->SetY2NDC(y2);
    }
  }

  // Draw colored non-pixel hits
  if (!xs.empty()) {
    const double dotRadiusMm = pixelSizeMm / 6.0;
    const int numSegments = 36;
    const int ncolors = TColor::GetNumberOfColors();

    for (size_t i = 0; i < xs.size(); ++i) {
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
  } else {
    std::cerr << "WARNING: No non-pixel hits found to draw." << std::endl;
  }

  c->Modified();
  c->Update();
  if (saveImage && outImagePath && std::strlen(outImagePath) > 0) c->SaveAs(outImagePath);

  f->Close();
  delete f;
}

// Convenience wrappers so running `root -l -b -q plotResolutionOnGrid.C` auto-executes
void plotResolutionOnGrid() {
  const char* thisFile = __FILE__;
  std::string thisDir = gSystem->DirName(thisFile);
  std::string defaultPath = thisDir + "/../build/epicChargeSharing.root";
  plotResolutionOnGridCore(defaultPath.c_str());
}

void plotResolutionOnGrid(const char* filename) {
  plotResolutionOnGridCore(filename, "resolution_on_grid.svg", true, "dmean");
}

void plotResolutionOnGrid(const char* filename, const char* metric) {
  plotResolutionOnGridCore(filename, "resolution_on_grid.svg", true, metric);
}


