#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TBox.h>
#include <TH1D.h>
#include <TGraph.h>
#include <TLatex.h>
#include <TLine.h>
#include <TSystem.h>
#include <TROOT.h>

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

// 2D analogue of plotRowColResolution_3D.C
void plotRowColResolution_2DCore(const char* rootFilePath = "epicChargeSharing.root",
                                 const char* outRows = "row_resolution_2d.svg",
                                 const char* outCols = "col_resolution_2d.svg",
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
  const double halfPitch = 0.5 * pixelSpacingMm;
  const double sigmaPixel = pixelSizeMm / std::sqrt(12.0);

  // Choose random 3x3 block
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

  // Pre-collect hits (non-pixel hits only) that fall inside the 3x3 window
  std::vector<double> hitX;
  std::vector<double> hitY;
  std::vector<double> hitDX; // |Δx| 2D
  std::vector<double> hitDY; // |Δy| 2D
  {
    TTree* hitsTree = dynamic_cast<TTree*>(f->Get("Hits"));
    if (!hitsTree) {
      std::cerr << "ERROR: 'Hits' tree not found in ROOT file." << std::endl;
      f->Close();
      delete f;
      return;
    }
    double x_hit = 0.0, y_hit = 0.0;
    double dtx2d = 0.0, dty2d = 0.0; // 2D deltas are stored without _2D suffix in processing2D
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
    hitsTree->SetBranchAddress("ReconTrueDeltaX", &dtx2d);
    hitsTree->SetBranchAddress("ReconTrueDeltaY", &dty2d);

    const Long64_t nEntries = hitsTree->GetEntries();
    hitX.reserve(nEntries);
    hitY.reserve(nEntries);
    hitDX.reserve(nEntries);
    hitDY.reserve(nEntries);

    for (Long64_t i = 0; i < nEntries; ++i) {
      hitsTree->GetEntry(i);
      if (!std::isfinite(x_hit) || !std::isfinite(y_hit)) continue;
      if (x_hit < xMin || x_hit > xMax || y_hit < yMin || y_hit > yMax) continue;
      if (is_pixel_hit) continue; // only non-pixel hits participate in charge sharing
      hitX.push_back(x_hit);
      hitY.push_back(y_hit);
      hitDX.push_back(std::abs(dtx2d));
      hitDY.push_back(std::abs(dty2d));
    }
  }

  auto makeCanvas = [](const char* name) {
    TCanvas* c = new TCanvas(name, "", 1100, 900);
    c->Divide(2,2);
    return c;
  };

  auto draw1D = [&](TPad* pad,
                    const char* title,
                    const char* xTitle,
                    double axisMin,
                    double axisMax,
                    const std::vector<std::pair<double,double>>& pixelRanges,
                    const std::vector<double>& pos,
                    const std::vector<double>& val,
                    double sigmaPix,
                    double yMaxOverride = -1.0) {
    pad->cd();
    pad->SetGrid(0,0);
    const double yMinAx = 0.0;
    double yMaxAx = 0.0;
    for (double v : val) if (v > yMaxAx) yMaxAx = v;
    yMaxAx = std::max(yMaxAx, sigmaPix);
    if (yMaxOverride > 0.0) yMaxAx = std::max(yMaxAx, yMaxOverride);
    yMaxAx = (yMaxAx <= 0.0) ? (sigmaPix*1.6) : (yMaxAx*1.15);

    TH1D* fr = new TH1D(Form("fr_%s", pad->GetName()), "", 10, axisMin, axisMax);
    fr->SetMinimum(yMinAx);
    fr->SetMaximum(yMaxAx);
    fr->GetXaxis()->SetTitle(xTitle);
    fr->GetYaxis()->SetTitle("resolution [mm]");
    fr->GetXaxis()->SetTitleOffset(1.1);
    fr->GetYaxis()->SetTitleOffset(1.2);
    fr->Draw("AXIS");

    const int colorFill = kPink + 1;
    const double alpha  = 0.25;
    for (const auto& r : pixelRanges) {
      TBox* b = new TBox(r.first, yMinAx, r.second, yMaxAx);
      b->SetFillColorAlpha(colorFill, alpha);
      b->SetLineColor(colorFill);
      b->SetLineWidth(1);
      b->Draw("same");

      TLine* l = new TLine(r.first, sigmaPix, r.second, sigmaPix);
      l->SetLineColor(kBlack);
      l->SetLineWidth(2);
      l->Draw("same");
    }

    TGraph* gr = new TGraph(static_cast<int>(pos.size()));
    for (int i = 0; i < gr->GetN(); ++i) {
      gr->SetPoint(i, pos[i], val[i]);
    }
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(0.6);
    gr->SetMarkerColor(kBlue+2);
    gr->Draw("P SAME");

    TLatex latex;
    latex.SetNDC(true);
    latex.SetTextSize(0.040);
    latex.DrawLatex(0.16, 0.92, title);

    pad->Modified();
    pad->Update();
  };

  std::vector<std::pair<double,double>> pxRanges = {
    {xC0 - pixelSizeMm/2.0, xC0 + pixelSizeMm/2.0},
    {xC1 - pixelSizeMm/2.0, xC1 + pixelSizeMm/2.0},
    {xC2 - pixelSizeMm/2.0, xC2 + pixelSizeMm/2.0}
  };
  std::vector<std::pair<double,double>> pyRanges = {
    {yC0 - pixelSizeMm/2.0, yC0 + pixelSizeMm/2.0},
    {yC1 - pixelSizeMm/2.0, yC1 + pixelSizeMm/2.0},
    {yC2 - pixelSizeMm/2.0, yC2 + pixelSizeMm/2.0}
  };

  std::vector<double> rowX0, rowR0, rowX1, rowR1, rowX2, rowR2;
  std::vector<double> colY0, colR0, colY1, colR1, colY2, colR2;
  rowX0.reserve(hitX.size()); rowR0.reserve(hitX.size());
  rowX1.reserve(hitX.size()); rowR1.reserve(hitX.size());
  rowX2.reserve(hitX.size()); rowR2.reserve(hitX.size());
  colY0.reserve(hitY.size()); colR0.reserve(hitY.size());
  colY1.reserve(hitY.size()); colR1.reserve(hitY.size());
  colY2.reserve(hitY.size()); colR2.reserve(hitY.size());

  const double halfH = pixelSizeMm/2.0;

  for (size_t i = 0; i < hitX.size(); ++i) {
    const double x = hitX[i];
    const double y = hitY[i];
    const double rx = hitDX[i];
    const double ry = hitDY[i];

    if (y >= yC0 - halfH && y <= yC0 + halfH) { rowX0.push_back(x); rowR0.push_back(rx); }
    if (y >= yC1 - halfH && y <= yC1 + halfH) { rowX1.push_back(x); rowR1.push_back(rx); }
    if (y >= yC2 - halfH && y <= yC2 + halfH) { rowX2.push_back(x); rowR2.push_back(rx); }

    if (x >= xC0 - halfH && x <= xC0 + halfH) { colY0.push_back(y); colR0.push_back(ry); }
    if (x >= xC1 - halfH && x <= xC1 + halfH) { colY1.push_back(y); colR1.push_back(ry); }
    if (x >= xC2 - halfH && x <= xC2 + halfH) { colY2.push_back(y); colR2.push_back(ry); }
  }

  TCanvas* cRows = makeCanvas("c_row_resolution_2d");
  draw1D((TPad*)cRows->cd(1), Form("Row y=%.3f mm", yC0), "x [mm]", xMin, xMax, pxRanges, rowX0, rowR0, sigmaPixel);
  draw1D((TPad*)cRows->cd(2), Form("Row y=%.3f mm", yC1), "x [mm]", xMin, xMax, pxRanges, rowX1, rowR1, sigmaPixel);
  draw1D((TPad*)cRows->cd(3), Form("Row y=%.3f mm", yC2), "x [mm]", xMin, xMax, pxRanges, rowX2, rowR2, sigmaPixel);
  std::vector<double> rowXAll; rowXAll.reserve(rowX0.size()+rowX1.size()+rowX2.size());
  std::vector<double> rowRAll; rowRAll.reserve(rowR0.size()+rowR1.size()+rowR2.size());
  rowXAll.insert(rowXAll.end(), rowX0.begin(), rowX0.end());
  rowXAll.insert(rowXAll.end(), rowX1.begin(), rowX1.end());
  rowXAll.insert(rowXAll.end(), rowX2.begin(), rowX2.end());
  rowRAll.insert(rowRAll.end(), rowR0.begin(), rowR0.end());
  rowRAll.insert(rowRAll.end(), rowR1.begin(), rowR1.end());
  rowRAll.insert(rowRAll.end(), rowR2.begin(), rowR2.end());
  draw1D((TPad*)cRows->cd(4), "Rows (aggregate)", "x [mm]", xMin, xMax, pxRanges, rowXAll, rowRAll, sigmaPixel);
  cRows->Modified(); cRows->Update();
  if (saveImage && outRows && std::strlen(outRows) > 0) cRows->SaveAs(outRows);

  TCanvas* cCols = makeCanvas("c_col_resolution_2d");
  draw1D((TPad*)cCols->cd(1), Form("Column x=%.3f mm", xC0), "y [mm]", yMin, yMax, pyRanges, colY0, colR0, sigmaPixel);
  draw1D((TPad*)cCols->cd(2), Form("Column x=%.3f mm", xC1), "y [mm]", yMin, yMax, pyRanges, colY1, colR1, sigmaPixel);
  draw1D((TPad*)cCols->cd(3), Form("Column x=%.3f mm", xC2), "y [mm]", yMin, yMax, pyRanges, colY2, colR2, sigmaPixel);
  std::vector<double> colYAll; colYAll.reserve(colY0.size()+colY1.size()+colY2.size());
  std::vector<double> colRAll; colRAll.reserve(colR0.size()+colR1.size()+colR2.size());
  colYAll.insert(colYAll.end(), colY0.begin(), colY0.end());
  colYAll.insert(colYAll.end(), colY1.begin(), colY1.end());
  colYAll.insert(colYAll.end(), colY2.begin(), colY2.end());
  colRAll.insert(colRAll.end(), colR0.begin(), colR0.end());
  colRAll.insert(colRAll.end(), colR1.begin(), colR1.end());
  colRAll.insert(colRAll.end(), colR2.begin(), colR2.end());
  draw1D((TPad*)cCols->cd(4), "Columns (aggregate)", "y [mm]", yMin, yMax, pyRanges, colYAll, colRAll, sigmaPixel);
  cCols->Modified(); cCols->Update();
  if (saveImage && outCols && std::strlen(outCols) > 0) cCols->SaveAs(outCols);

  f->Close();
  delete f;
}

void plotRowColResolution_2D() {
  const char* thisFile = __FILE__;
  std::string thisDir = gSystem->DirName(thisFile);
  std::string defaultPath = thisDir + "/../build/epicChargeSharing.root";
  plotRowColResolution_2DCore(defaultPath.c_str());
}

void plotRowColResolution_2D(const char* filename) {
  plotRowColResolution_2DCore(filename, "row_resolution_2d.svg", "col_resolution_2d.svg", true);
}


