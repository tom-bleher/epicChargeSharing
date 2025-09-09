#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TH1D.h>
#include <TGraph.h>
#include <TLatex.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TRandom3.h>
#include <TPad.h>
#include <TLegend.h>

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

// Between-pixel bands (gaps) version of row/column resolution plot using 2D deltas
void plotRowColResolutionBetween_2DCore(const char* rootFilePath = "epicChargeSharing.root",
                                        const char* outRows = "row_resolution_between_2d.svg",
                                        const char* outCols = "col_resolution_between_2d.svg",
                                        bool saveImage = true,
                                        Long64_t seed = 0,
                                        int forceI0 = -1,
                                        int forceJ0 = -1,
                                        double edgeMarginMm = 0.1) // exclude ±100 µm near each pixel edge inside the gap
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

  const double halfDet   = detSizeMm / 2.0;
  const double firstC    = -halfDet + pixelCornerOffsetMm + pixelSizeMm/2.0;
  const double halfPitch = 0.5 * pixelSpacingMm;
  const double gap       = std::max(0.0, pixelSpacingMm - pixelSizeMm);
  const double halfGap   = std::max(1e-6, 0.5 * gap);
  const double halfBand  = std::max(1e-6, halfGap - std::max(0.0, edgeMarginMm));

  // Close / Far thresholds (in mm)
  const double closeMin = 0.050e-3 * 1e3; // 0.050 mm
  const double closeMax = 0.150e-3 * 1e3; // 0.150 mm
  const double farMin   = 0.150e-3 * 1e3; // 0.150 mm
  const double farMax   = 0.250e-3 * 1e3; // 0.250 mm

  TRandom3 rng;
  if (seed == 0) seed = static_cast<Long64_t>(gSystem->Now());
  rng.SetSeed(seed);

  int i0 = (forceI0 >= 0 && forceI0 <= numPerSide-3) ? forceI0 : rng.Integer(numPerSide-2);
  int j0 = (forceJ0 >= 0 && forceJ0 <= numPerSide-3) ? forceJ0 : rng.Integer(numPerSide-2);
  const int i1 = i0 + 1;
  const int i2 = i0 + 2;
  const int j1 = j0 + 1;
  const int j2 = j0 + 2;

  const double xC0 = firstC + i0 * pixelSpacingMm;
  const double xC1 = firstC + i1 * pixelSpacingMm;
  const double xC2 = firstC + i2 * pixelSpacingMm;
  const double yC0 = firstC + j0 * pixelSpacingMm;
  const double yC1 = firstC + j1 * pixelSpacingMm;
  const double yC2 = firstC + j2 * pixelSpacingMm;

  // Gap band centers
  const double yG0 = 0.5 * (yC0 + yC1);
  const double yG1 = 0.5 * (yC1 + yC2);
  const double xG0 = 0.5 * (xC0 + xC1);
  const double xG1 = 0.5 * (xC1 + xC2);

  double xMin = std::max(-halfDet, xC0 - halfPitch);
  double xMax = std::min( halfDet, xC2 + halfPitch);
  double yMin = std::max(-halfDet, yC0 - halfPitch);
  double yMax = std::min( halfDet, yC2 + halfPitch);

  // Collect hits inside 3x3 window (non-pixel) and sort into gap bands
  std::vector<double> hitX, hitY, hitDX, hitDY, hitPDX, hitPDY;
  {
    TTree* hitsTree = dynamic_cast<TTree*>(f->Get("Hits"));
    if (!hitsTree) {
      std::cerr << "ERROR: 'Hits' tree not found in ROOT file." << std::endl;
      f->Close();
      delete f;
      return;
    }
    double x_hit = 0.0, y_hit = 0.0;
    double dtx2d = 0.0, dty2d = 0.0; // 2D deltas
    double px_dx = 0.0, px_dy = 0.0; // PixelTrueDeltaX/Y
    Bool_t is_pixel_hit = 0;
    hitsTree->SetBranchStatus("*", 0);
    hitsTree->SetBranchStatus("TrueX", 1);
    hitsTree->SetBranchStatus("TrueY", 1);
    hitsTree->SetBranchStatus("isPixelHit", 1);
    hitsTree->SetBranchStatus("ReconTrueDeltaX", 1);
    hitsTree->SetBranchStatus("ReconTrueDeltaY", 1);
    hitsTree->SetBranchStatus("PixelTrueDeltaX", 1);
    hitsTree->SetBranchStatus("PixelTrueDeltaY", 1);
    hitsTree->SetBranchAddress("TrueX", &x_hit);
    hitsTree->SetBranchAddress("TrueY", &y_hit);
    hitsTree->SetBranchAddress("isPixelHit", &is_pixel_hit);
    hitsTree->SetBranchAddress("ReconTrueDeltaX", &dtx2d);
    hitsTree->SetBranchAddress("ReconTrueDeltaY", &dty2d);
    hitsTree->SetBranchAddress("PixelTrueDeltaX", &px_dx);
    hitsTree->SetBranchAddress("PixelTrueDeltaY", &px_dy);

    const Long64_t nEntries = hitsTree->GetEntries();
    hitX.reserve(nEntries);
    hitY.reserve(nEntries);
    hitDX.reserve(nEntries);
    hitDY.reserve(nEntries);
    hitPDX.reserve(nEntries);
    hitPDY.reserve(nEntries);

    for (Long64_t i = 0; i < nEntries; ++i) {
      hitsTree->GetEntry(i);
      if (!std::isfinite(x_hit) || !std::isfinite(y_hit)) continue;
      if (x_hit < xMin || x_hit > xMax || y_hit < yMin || y_hit > yMax) continue;
      if (is_pixel_hit) continue;
      hitX.push_back(x_hit);
      hitY.push_back(y_hit);
      hitDX.push_back(std::abs(dtx2d));
      hitDY.push_back(std::abs(dty2d));
      hitPDX.push_back(std::abs(px_dx));
      hitPDY.push_back(std::abs(px_dy));
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
                    const std::vector<double>& pos,
                    const std::vector<double>& val) {
    pad->cd();
    pad->SetGrid(0,0);
    const double yMinAx = 0.0;
    double yMaxAx = 0.0;
    for (double v : val) if (v > yMaxAx) yMaxAx = v;
    if (yMaxAx <= 0.0) yMaxAx = 0.1; // small default
    yMaxAx *= 1.15;

    TH1D* fr = new TH1D(Form("fr_%s", pad->GetName()), "", 10, axisMin, axisMax);
    fr->SetMinimum(yMinAx);
    fr->SetMaximum(yMaxAx);
    fr->GetXaxis()->SetTitle(xTitle);
    fr->GetYaxis()->SetTitle("resolution [mm]");
    fr->GetXaxis()->SetTitleOffset(1.1);
    fr->GetYaxis()->SetTitleOffset(1.2);
    fr->Draw("AXIS");

    TGraph* gr = new TGraph(static_cast<int>(pos.size()));
    for (int i = 0; i < gr->GetN(); ++i) gr->SetPoint(i, pos[i], val[i]);
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(0.6);
    gr->SetMarkerColor(kBlue+2);
    gr->Draw("P SAME");

    TLatex latex; latex.SetNDC(true); latex.SetTextSize(0.040);
    latex.DrawLatex(0.16, 0.92, title);
  };

  auto draw1Dcf = [&](TPad* pad,
                      const char* title,
                      const char* xTitle,
                      double axisMin,
                      double axisMax,
                      const std::vector<double>& posClose,
                      const std::vector<double>& valClose,
                      const std::vector<double>& posFar,
                      const std::vector<double>& valFar) {
    pad->cd();
    pad->SetGrid(0,0);
    const double yMinAx = 0.0;
    double yMaxAx = 0.0;
    for (double v : valClose) if (v > yMaxAx) yMaxAx = v;
    for (double v : valFar)   if (v > yMaxAx) yMaxAx = v;
    if (yMaxAx <= 0.0) yMaxAx = 0.1;
    yMaxAx *= 1.15;

    TH1D* fr = new TH1D(Form("fr_%s", pad->GetName()), "", 10, axisMin, axisMax);
    fr->SetMinimum(yMinAx);
    fr->SetMaximum(yMaxAx);
    fr->GetXaxis()->SetTitle(xTitle);
    fr->GetYaxis()->SetTitle("resolution [mm]");
    fr->GetXaxis()->SetTitleOffset(1.1);
    fr->GetYaxis()->SetTitleOffset(1.2);
    fr->Draw("AXIS");

    TGraph* grC = new TGraph(static_cast<int>(posClose.size()));
    for (int i = 0; i < grC->GetN(); ++i) grC->SetPoint(i, posClose[i], valClose[i]);
    grC->SetMarkerStyle(20);
    grC->SetMarkerSize(0.6);
    grC->SetMarkerColor(kGreen+2);
    grC->Draw("P SAME");

    TGraph* grF = new TGraph(static_cast<int>(posFar.size()));
    for (int i = 0; i < grF->GetN(); ++i) grF->SetPoint(i, posFar[i], valFar[i]);
    grF->SetMarkerStyle(21);
    grF->SetMarkerSize(0.6);
    grF->SetMarkerColor(kRed+1);
    grF->Draw("P SAME");

    TLatex latex; latex.SetNDC(true); latex.SetTextSize(0.040);
    latex.DrawLatex(0.16, 0.92, title);

    TLegend* leg = new TLegend(0.62, 0.80, 0.88, 0.92);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->AddEntry(grC, "close", "p");
    leg->AddEntry(grF, "far",   "p");
    leg->Draw();
  };

  // Sort hits into gap bands and close/far groups
  std::vector<double> rowX_a_close, rowR_a_close, rowX_a_far, rowR_a_far;
  std::vector<double> rowX_b_close, rowR_b_close, rowX_b_far, rowR_b_far;
  std::vector<double> colY_a_close, colR_a_close, colY_a_far, colR_a_far;
  std::vector<double> colY_b_close, colR_b_close, colY_b_far, colR_b_far;
  rowX_a_close.reserve(hitX.size()); rowR_a_close.reserve(hitX.size());
  rowX_a_far.reserve(hitX.size());   rowR_a_far.reserve(hitX.size());
  rowX_b_close.reserve(hitX.size()); rowR_b_close.reserve(hitX.size());
  rowX_b_far.reserve(hitX.size());   rowR_b_far.reserve(hitX.size());
  colY_a_close.reserve(hitY.size()); colR_a_close.reserve(hitY.size());
  colY_a_far.reserve(hitY.size());   colR_a_far.reserve(hitY.size());
  colY_b_close.reserve(hitY.size()); colR_b_close.reserve(hitY.size());
  colY_b_far.reserve(hitY.size());   colR_b_far.reserve(hitY.size());

  for (size_t i = 0; i < hitX.size(); ++i) {
    const double x  = hitX[i];
    const double y  = hitY[i];
    const double rx = hitDX[i];
    const double ry = hitDY[i];
    const double pdx = hitPDX[i];
    const double pdy = hitPDY[i];

    const bool isCloseX = (pdx >= closeMin && pdx <= closeMax);
    const bool isFarX   = (pdx >  farMin  && pdx <= farMax);
    const bool isCloseY = (pdy >= closeMin && pdy <= closeMax);
    const bool isFarY   = (pdy >  farMin  && pdy <= farMax);

    if (std::abs(y - yG0) <= halfBand) {
      if (isCloseX) { rowX_a_close.push_back(x); rowR_a_close.push_back(rx); }
      if (isFarX)   { rowX_a_far.push_back(x);   rowR_a_far.push_back(rx); }
    }
    if (std::abs(y - yG1) <= halfBand) {
      if (isCloseX) { rowX_b_close.push_back(x); rowR_b_close.push_back(rx); }
      if (isFarX)   { rowX_b_far.push_back(x);   rowR_b_far.push_back(rx); }
    }
    if (std::abs(x - xG0) <= halfBand) {
      if (isCloseY) { colY_a_close.push_back(y); colR_a_close.push_back(ry); }
      if (isFarY)   { colY_a_far.push_back(y);   colR_a_far.push_back(ry); }
    }
    if (std::abs(x - xG1) <= halfBand) {
      if (isCloseY) { colY_b_close.push_back(y); colR_b_close.push_back(ry); }
      if (isFarY)   { colY_b_far.push_back(y);   colR_b_far.push_back(ry); }
    }
  }

  // ROWS canvas: two gap rows + aggregate
  TCanvas* cRows = makeCanvas("c_row_resolution_between_2d");
  draw1Dcf((TPad*)cRows->cd(1), Form("Gap-row y=%.3f mm", yG0), "x [mm]", xMin, xMax,
           rowX_a_close, rowR_a_close, rowX_a_far, rowR_a_far);
  draw1Dcf((TPad*)cRows->cd(2), Form("Gap-row y=%.3f mm", yG1), "x [mm]", xMin, xMax,
           rowX_b_close, rowR_b_close, rowX_b_far, rowR_b_far);
  std::vector<double> rowX_close_all = rowX_a_close; rowX_close_all.insert(rowX_close_all.end(), rowX_b_close.begin(), rowX_b_close.end());
  std::vector<double> rowR_close_all = rowR_a_close; rowR_close_all.insert(rowR_close_all.end(), rowR_b_close.begin(), rowR_b_close.end());
  std::vector<double> rowX_far_all   = rowX_a_far;   rowX_far_all.insert(rowX_far_all.end(), rowX_b_far.begin(), rowX_b_far.end());
  std::vector<double> rowR_far_all   = rowR_a_far;   rowR_far_all.insert(rowR_far_all.end(), rowR_b_far.begin(), rowR_b_far.end());
  draw1Dcf((TPad*)cRows->cd(3), "Gap-rows (aggregate)", "x [mm]", xMin, xMax,
           rowX_close_all, rowR_close_all, rowX_far_all, rowR_far_all);
  cRows->Modified(); cRows->Update();
  if (saveImage && outRows && std::strlen(outRows) > 0) cRows->SaveAs(outRows);

  // COLUMNS canvas: two gap columns + aggregate
  TCanvas* cCols = makeCanvas("c_col_resolution_between_2d");
  draw1Dcf((TPad*)cCols->cd(1), Form("Gap-col x=%.3f mm", xG0), "y [mm]", yMin, yMax,
           colY_a_close, colR_a_close, colY_a_far, colR_a_far);
  draw1Dcf((TPad*)cCols->cd(2), Form("Gap-col x=%.3f mm", xG1), "y [mm]", yMin, yMax,
           colY_b_close, colR_b_close, colY_b_far, colR_b_far);
  std::vector<double> colY_close_all = colY_a_close; colY_close_all.insert(colY_close_all.end(), colY_b_close.begin(), colY_b_close.end());
  std::vector<double> colR_close_all = colR_a_close; colR_close_all.insert(colR_close_all.end(), colR_b_close.begin(), colR_b_close.end());
  std::vector<double> colY_far_all   = colY_a_far;   colY_far_all.insert(colY_far_all.end(), colY_b_far.begin(), colY_b_far.end());
  std::vector<double> colR_far_all   = colR_a_far;   colR_far_all.insert(colR_far_all.end(), colR_b_far.begin(), colR_b_far.end());
  draw1Dcf((TPad*)cCols->cd(3), "Gap-columns (aggregate)", "y [mm]", yMin, yMax,
           colY_close_all, colR_close_all, colY_far_all, colR_far_all);
  cCols->Modified(); cCols->Update();
  if (saveImage && outCols && std::strlen(outCols) > 0) cCols->SaveAs(outCols);

  f->Close();
  delete f;
}

void plotRowColResolutionBetween_2D() {
  const char* thisFile = __FILE__;
  std::string thisDir = gSystem->DirName(thisFile);
  std::string defaultPath = thisDir + "/../../../build/epicChargeSharing.root";
  plotRowColResolutionBetween_2DCore(defaultPath.c_str());
}

void plotRowColResolutionBetween_2D(const char* filename) {
  plotRowColResolutionBetween_2DCore(filename, "row_resolution_between_2d.svg", "col_resolution_between_2d.svg", true);
}