#include <TFile.h>
#include <TTree.h>
#include <TNamed.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TBox.h>
#include <TH1.h>
#include <TH2D.h>
#include <TColor.h>
#include <TLatex.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TPolyLine.h>
#include <TLegend.h>
#include <TLine.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
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

  TFile* TryOpenWithFallbacks(const char* rootFilePath)
  {
    auto tryOpen = [](const char* path) -> TFile* {
      if (!path || !*path) return nullptr;
      if (gSystem->AccessPathName(path, kReadPermission)) return nullptr;
      TFile* tmp = TFile::Open(path, "READ");
      if (!tmp || tmp->IsZombie()) { if (tmp) { tmp->Close(); delete tmp; } return nullptr; }
      return tmp;
    };

    TFile* f = tryOpen(rootFilePath);
    if (!f) f = tryOpen("../../build/epicChargeSharing.root");
    return f;
  }

  // Helper to draw a filled circle in mm coordinates (avoids TMarker size-in-NDC)
  void DrawCircleMm(double cx, double cy, double r, int colorFill, int colorLine, int nSeg = 64)
  {
    std::vector<double> xs(nSeg + 1);
    std::vector<double> ys(nSeg + 1);
    for (int k = 0; k <= nSeg; ++k) {
      const double ang = (2.0*M_PI * k) / static_cast<double>(nSeg);
      xs[k] = cx + r * std::cos(ang);
      ys[k] = cy + r * std::sin(ang);
    }
    TPolyLine* circle = new TPolyLine(nSeg + 1, xs.data(), ys.data());
    circle->SetFillColor(colorFill);
    circle->SetFillStyle(1001);
    circle->SetLineColor(colorLine);
    circle->SetLineWidth(1);
    circle->Draw("f same");
  }

  // Map a value in [vmin,vmax] to a ROOT palette color index
  int ValueToPaletteColor(double v, double vmin, double vmax, bool invert=false)
  {
    if (!(std::isfinite(v) && std::isfinite(vmin) && std::isfinite(vmax)) || vmax <= vmin) {
      return kGray + 1;
    }
    double t = std::clamp((v - vmin) / (vmax - vmin), 0.0, 1.0);
    if (invert) t = 1.0 - t;
    const int n = gStyle->GetNumberOfColors();
    const int idx = std::clamp(static_cast<int>(std::round(t * (n - 1))), 0, n - 1);
    return TColor::GetColorPalette(idx);
  }
  
  // Map human-friendly quantity selectors to internal dataKind strings
  // Accepts: "F_i", "Fi", "fraction" -> "fraction" and
  //          "Q_i", "Qi", "charge", "coulomb" -> "coulomb"
  const char* QuantityToDataKind(const char* quantity)
  {
    if (!quantity) return "fraction";
    std::string q = quantity;
    std::transform(q.begin(), q.end(), q.begin(), [](unsigned char c){ return std::tolower(c); });
    if (q == "f" || q == "fi" || q == "f_i" || q == "fraction" || q == "fractions") return "fraction";
    if (q == "q" || q == "qi" || q == "q_i" || q == "charge" || q == "coulomb") return "coulomb";
    return "fraction";
  }
}

// Draw a 5x5 charge neighborhood centered on the event's pixel center using real mm coordinates.
// dataKind: "fraction" (F_i or Q_/Q_f divided by Q_tot), "coulomb" (Q_i/Q_f or F_i * Q_tot), "distance" (distance to hit)
// chargeBranch: choose explicitly among "Q_f", "Q_i", or "F_i" (default "Q_f").
void plotChargeNeighborhood5x5(const char* rootFilePath = "epicChargeSharing.root",
                               Long64_t eventIndex = -1,
                               const char* dataKind = "coulomb",
                               const char* outImagePath = "",
                               const char* chargeBranch = "Q_f")
{
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPalette(kBird);

  TFile* f = TryOpenWithFallbacks(rootFilePath);
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

  TTree* tree = dynamic_cast<TTree*>(f->Get("Hits"));
  if (!tree) {
    std::cerr << "ERROR: 'Hits' tree not found in ROOT file." << std::endl;
    f->Close(); delete f; return;
  }

  // Inputs
  double trueX = std::numeric_limits<double>::quiet_NaN();
  double trueY = std::numeric_limits<double>::quiet_NaN();
  double pixelX = std::numeric_limits<double>::quiet_NaN();
  double pixelY = std::numeric_limits<double>::quiet_NaN();
  double edep   = std::numeric_limits<double>::quiet_NaN(); // [MeV]
  Bool_t isPixelHit = 0;

  std::vector<double>* Fi = nullptr;          // Charge fractions per neighborhood cell (size = N*N)
  std::vector<double>* Qi = nullptr;          // Induced charge per cell [C] (size = N*N)
  std::vector<double>* Qn = nullptr;          // Intermediate noisy charge per cell [C] (size = N*N)
  std::vector<double>* Qf = nullptr;          // Final noisy charge per cell [C] (size = N*N)

  // Speed up I/O: deactivate everything, then enable the needed branches if present
  tree->SetBranchStatus("*", 0);
  auto enableIf = [&](const char* b){ if (tree->GetBranch(b)) tree->SetBranchStatus(b, 1); };
  enableIf("TrueX"); enableIf("TrueY"); enableIf("PixelX"); enableIf("PixelY");
  enableIf("Edep"); enableIf("isPixelHit");
  enableIf("Q_f"); enableIf("Q_n"); enableIf("Q_i"); enableIf("F_i");

  if (tree->GetBranch("TrueX")) tree->SetBranchAddress("TrueX", &trueX);
  if (tree->GetBranch("TrueY")) tree->SetBranchAddress("TrueY", &trueY);
  if (tree->GetBranch("PixelX")) tree->SetBranchAddress("PixelX", &pixelX);
  if (tree->GetBranch("PixelY")) tree->SetBranchAddress("PixelY", &pixelY);
  if (tree->GetBranch("Edep")) tree->SetBranchAddress("Edep", &edep);
  if (tree->GetBranch("isPixelHit")) tree->SetBranchAddress("isPixelHit", &isPixelHit);
  if (tree->GetBranch("Q_f")) tree->SetBranchAddress("Q_f", &Qf);
  if (tree->GetBranch("Q_n")) tree->SetBranchAddress("Q_n", &Qn);
  if (tree->GetBranch("Q_i")) tree->SetBranchAddress("Q_i", &Qi);
  if (tree->GetBranch("F_i")) tree->SetBranchAddress("F_i", &Fi);

  // Select which neighborhood data we will use
  std::string kind = dataKind ? dataKind : "fraction";
  std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c){ return std::tolower(c); });
  std::string label = (kind == "coulomb") ? "Charge" : (kind == "distance" ? "Distance" : "Charge Fraction");
  std::string unit  = (kind == "coulomb") ? " C" : (kind == "distance" ? " mm" : "");
  if (!tree->GetBranch("Q_i") && !tree->GetBranch("F_i") && !tree->GetBranch("Q_f") && !tree->GetBranch("Q_n")) {
    std::cerr << "ERROR: Required neighborhood data branch 'Q_n', 'Q_i', 'Q_f', or 'F_i' not found in tree." << std::endl;
    f->Close(); delete f; return;
  }

  const Long64_t nEntries = tree->GetEntries();
  if (nEntries <= 0) { std::cerr << "ERROR: Hits tree is empty." << std::endl; f->Close(); delete f; return; }

  // If eventIndex < 0, pick the first event that has any valid neighborhood values
  Long64_t evt = eventIndex;
  if (evt < 0 || evt >= nEntries) {
    evt = -1;
    for (Long64_t i = 0; i < nEntries; ++i) {
      tree->GetEntry(i);
      const std::vector<double>* vptr = (kind == "coulomb") ? (Qi ? Qi : Fi) : (Fi ? Fi : Qi);
      if (!vptr) continue;
      if (vptr->empty()) continue;
      const bool hasValid = std::any_of(vptr->begin(), vptr->end(), [](double v){ return std::isfinite(v) && v != -999.0; });
      if (hasValid) { evt = i; break; }
    }
    if (evt < 0) evt = 0; // fallback
  }

  // Load the selected event
  tree->GetEntry(evt);
  // Choose which branch to visualize, honoring explicit request then falling back Q_f -> Q_i -> F_i
  std::string chosen = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Q_n");
  auto pickVec = [&]() -> const std::vector<double>* {
    if (chosen == "Q_n" && Qn) return Qn;
    if (chosen == "Q_f" && Qf) return Qf;
    if (chosen == "Q_i" && Qi) return Qi;
    if (chosen == "F_i" && Fi) return Fi;
    if (Qn) return Qn;
    if (Qf) return Qf;
    if (Qi) return Qi;
    if (Fi) return Fi;
    return nullptr;
  };
  const std::vector<double>* vecPtr = pickVec();
  const bool usingQf = (vecPtr == Qf && Qf != nullptr);
  const bool usingQn = (vecPtr == Qn && Qn != nullptr);
  const bool usingQi = (vecPtr == Qi && Qi != nullptr);
  if (!vecPtr || vecPtr->empty()) { std::cerr << "ERROR: Neighborhood vector is empty for event " << evt << std::endl; f->Close(); delete f; return; }

  // Determine grid dimension (expect 5x5 or larger).
  const size_t nVals = vecPtr->size();
  const int dim = static_cast<int>(std::lround(std::sqrt(static_cast<double>(nVals))));
  if (dim * dim != static_cast<int>(nVals)) {
    std::cerr << "ERROR: Neighborhood vector size (" << nVals << ") is not a perfect square." << std::endl;
    f->Close(); delete f; return;
  }
  if (dim < 5) {
    std::cerr << "ERROR: Neighborhood grid dimension is " << dim << ", need at least 5 for 5x5 visualization." << std::endl;
    f->Close(); delete f; return;
  }

  // Build transposed 2D grid so that i->Y (row), j->X (col) matches world coordinates
  std::vector<std::vector<double>> grid(dim, std::vector<double>(dim, std::numeric_limits<double>::quiet_NaN()));
  for (int di = 0; di < dim; ++di) {
    for (int dj = 0; dj < dim; ++dj) {
      const int idx = di * dim + dj; // stored as outer X(di), inner Y(dj)
      grid[dj][di] = (*vecPtr)[idx]; // transpose to grid[y][x]
    }
  }

  const int c = dim / 2; // center index
  const int i0 = c - 2;
  const int i1 = c + 2; // inclusive
  const int j0 = c - 2;
  const int j1 = c + 2; // inclusive

  // Precompute Q_tot (for coulomb) if Edep is available
  const double eChargeC = 1.602176634e-19; // Coulombs
  const double pairEnergyEV = 3.60;        // eV per e-h pair in Si
  const double MeV_to_eV = 1.0e6;
  double qTotalC = 0.0;
  if (std::isfinite(edep) && edep > 0) {
    qTotalC = edep * MeV_to_eV / pairEnergyEV * eChargeC; // [C]
  }

  // Optionally zero the fractions when there's no energy deposited (Python behavior for edep<=0)
  if (std::isfinite(edep) && edep <= 0 && kind == "fraction") {
    for (int i = i0; i <= i1; ++i) for (int j = j0; j <= j1; ++j) if (grid[i][j] >= 0.0) grid[i][j] = 0.0;
  }

  // Compute color mapping range on the 5x5 ROI for the chosen kind
  double vmin = 0.0, vmax = 1.0;
  if (kind == "fraction") {
    vmin = 0.0;
    vmax = 0.0;
    bool any = false;
    for (int i = i0; i <= i1; ++i) for (int j = j0; j <= j1; ++j) {
      const double raw = grid[i][j];
      if (!(std::isfinite(raw) && raw >= 0.0)) continue;
      double frac = 0.0;
      if (std::isfinite(edep) && edep <= 0) {
        frac = 0.0;
      } else if (usingQf || usingQn || usingQi) {
        if (qTotalC > 0) frac = raw / qTotalC; else continue;
      } else {
        frac = raw;
      }
      vmax = std::max(vmax, frac); any = true;
    }
    if (!any) vmax = 1.0;
  } else if (kind == "coulomb") {
    vmin = +std::numeric_limits<double>::infinity();
    vmax = -std::numeric_limits<double>::infinity();
    for (int i = i0; i <= i1; ++i) for (int j = j0; j <= j1; ++j) {
      const double raw = grid[i][j];
      if (!(std::isfinite(raw) && raw >= 0.0)) continue;
      const double q = (usingQf || usingQn || usingQi) ? raw : (raw * qTotalC);
      vmin = std::min(vmin, q);
      vmax = std::max(vmax, q);
    }
    if (!(vmax > vmin)) { vmin = 0.0; vmax = 1.0; }
  } else { // distance
    vmin = +std::numeric_limits<double>::infinity();
    vmax = -std::numeric_limits<double>::infinity();
    for (int i = i0; i <= i1; ++i) for (int j = j0; j <= j1; ++j) {
      const double fval = grid[i][j];
      if (!(std::isfinite(fval) && fval >= 0.0)) continue;
      const double rel_x = (j - c) * pixelSpacingMm; // j maps to X
      const double rel_y = (i - c) * pixelSpacingMm; // i maps to Y
      const double cx = pixelX + rel_x;
      const double cy = pixelY + rel_y;
      const double d = std::hypot(cx - trueX, cy - trueY);
      vmin = std::min(vmin, d);
      vmax = std::max(vmax, d);
    }
    if (!(vmax > vmin)) { vmin = 0.0; vmax = 1.0; }
  }

  // Canvas with frame focused on 5x5 neighborhood around the pixel center
  const double halfGridExtent = 2.5 * pixelSpacingMm; // 2 cells to edge + half cell for boundary
  const int canvasSize = 900;
  TCanvas* ccan = new TCanvas("c_charge_neighborhood5x5", "", canvasSize, canvasSize);
  ccan->SetLeftMargin(0.12);
  ccan->SetRightMargin(0.15); // reserve space for palette
  ccan->SetTopMargin(0.12);
  ccan->SetBottomMargin(0.12);
  ccan->SetFixedAspectRatio();

  const double xMin = pixelX - halfGridExtent;
  const double xMax = pixelX + halfGridExtent;
  const double yMin = pixelY - halfGridExtent;
  const double yMax = pixelY + halfGridExtent;
  TH1* frame = ccan->DrawFrame(xMin, yMin, xMax, yMax, ";x [mm];y [mm]");
  frame->GetXaxis()->SetTitleOffset(1.2);
  frame->GetYaxis()->SetTitleOffset(1.4);
  ccan->cd();
  gPad->SetFixedAspectRatio();

  // Draw colored blocks (size = pixelSpacing) for valid cells in 5x5 ROI
  const double alpha = 0.80;
  int nValid = 0;
  // Keep colors consistent with a visible palette: invert globally for distance,
  // and use non-inverted mapping below.
  const bool invertGlobalPalette = (kind == "distance");
  const bool invertPalette = false;
  if (invertGlobalPalette) { TColor::InvertPalette(); }

  // Create a dummy 2D hist to produce a palette axis matching [vmin,vmax]
  {
    std::string branchTitle;
    if (kind != "distance") {
      branchTitle = usingQn ? "Q_n" : (usingQf ? "Q_f" : (usingQi ? "Q_i" : "F_i"));
    }
    std::string zTitle = label + unit;
    if (!branchTitle.empty()) { zTitle += " ("; zTitle += branchTitle; zTitle += ")"; }
    TH2D* paletteHist = new TH2D("pal_charge_neighborhood5x5", "", 2, xMax+1, xMax+2, 2, yMin, yMax);
    paletteHist->SetMinimum(vmin);
    paletteHist->SetMaximum(vmax);
    paletteHist->GetZaxis()->SetTitle(zTitle.c_str());
    paletteHist->GetZaxis()->SetTitleSize(0.040);
    paletteHist->GetZaxis()->SetLabelSize(0.032);
    paletteHist->Draw("COLZ SAME");
    gPad->Update();
    TPaletteAxis* pal = (TPaletteAxis*)paletteHist->GetListOfFunctions()->FindObject("palette");
    if (pal) {
      pal->SetLabelSize(0.032);
      pal->SetTitleOffset(1.30);
      pal->SetBorderSize(1);
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
  }
  std::vector<std::vector<bool>> valid(dim, std::vector<bool>(dim, false));
  for (int i = i0; i <= i1; ++i) {
    for (int j = j0; j <= j1; ++j) {
      const double fval = grid[i][j];
      if (!(std::isfinite(fval)) || fval < 0.0) continue;
      valid[i][j] = true;
      ++nValid;

      const double rel_x = (j - c) * pixelSpacingMm; // j maps to X
      const double rel_y = (i - c) * pixelSpacingMm; // i maps to Y
      const double cx = pixelX + rel_x;
      const double cy = pixelY + rel_y;
      const double x1 = cx - pixelSpacingMm/2.0;
      const double x2 = cx + pixelSpacingMm/2.0;
      const double y1 = cy - pixelSpacingMm/2.0;
      const double y2 = cy + pixelSpacingMm/2.0;

      double shownVal = 0.0;
      if (kind == "fraction") {
        if (std::isfinite(edep) && edep <= 0) shownVal = 0.0;
        else shownVal = (usingQf || usingQi) ? (qTotalC > 0 ? fval / qTotalC : 0.0) : fval;
      } else if (kind == "coulomb") {
        shownVal = (usingQf || usingQi) ? fval : (fval * qTotalC);
      } else { // distance
        shownVal = std::hypot(cx - trueX, cy - trueY);
      }

      const int color = ValueToPaletteColor(shownVal, vmin, vmax, invertPalette);
      TBox* blk = new TBox(x1, y1, x2, y2);
      blk->SetFillColorAlpha(color, alpha);
      blk->SetLineColor(kBlack);
      blk->SetLineWidth(1);
      blk->Draw("same");

      // Value label centered slightly below the pixel center
      std::ostringstream vs;
      if (kind == "fraction") { vs.setf(std::ios::fixed); vs.precision(3); vs << shownVal; }
      else if (kind == "coulomb") { vs.setf(std::ios::scientific); vs.precision(2); vs << shownVal; }
      else { vs.setf(std::ios::fixed); vs.precision(3); vs << shownVal; }
      TLatex* txt = new TLatex(cx, cy - pixelSizeMm/2.0 - 0.08, vs.str().c_str());
      txt->SetTextAlign(22);
      txt->SetTextColor(kWhite);
      txt->SetTextSize(0.025);
      txt->Draw("same");
    }
  }

  // Draw the actual pixel squares (size = pixelSize) on top for the same valid positions
  for (int i = i0; i <= i1; ++i) {
    for (int j = j0; j <= j1; ++j) {
      if (!valid[i][j]) continue;
      const double rel_x = (j - c) * pixelSpacingMm;
      const double rel_y = (i - c) * pixelSpacingMm;
      const double cx = pixelX + rel_x;
      const double cy = pixelY + rel_y;
      const double x1 = cx - pixelSizeMm/2.0;
      const double x2 = cx + pixelSizeMm/2.0;
      const double y1 = cy - pixelSizeMm/2.0;
      const double y2 = cy + pixelSizeMm/2.0;

      TBox* px = new TBox(x1, y1, x2, y2);
      px->SetFillStyle(0);
      px->SetLineColor(kBlack);
      px->SetLineWidth(2);
      px->Draw("same");
    }
  }

  // Draw borders only where adjacent cell is invalid (to mimic Python grid lines)
  for (int i = i0; i <= i1; ++i) {
    for (int j = j0; j <= j1; ++j) {
      if (!valid[i][j]) continue;
      const double rel_x = (j - c) * pixelSpacingMm;
      const double rel_y = (i - c) * pixelSpacingMm;
      const double cx = pixelX + rel_x;
      const double cy = pixelY + rel_y;
      const double x1 = cx - pixelSpacingMm/2.0;
      const double x2 = cx + pixelSpacingMm/2.0;
      const double y1 = cy - pixelSpacingMm/2.0;
      const double y2 = cy + pixelSpacingMm/2.0;

      if (j - 1 < 0 || !valid[i][j-1]) { TLine* ln = new TLine(x1, y1, x1, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (j + 1 >= dim || !valid[i][j+1]) { TLine* ln = new TLine(x2, y1, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (i + 1 >= dim || !valid[i+1][j]) { TLine* ln = new TLine(x1, y1, x2, y1); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (i - 1 < 0 || !valid[i-1][j]) { TLine* ln = new TLine(x1, y2, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
    }
  }

  // Mark the true hit position
  const double hitDotRadius = pixelSizeMm / 6.0;
  DrawCircleMm(trueX, trueY, hitDotRadius, kRed+1, kRed+1);

  // Title and small legend
  {
    std::ostringstream ttl;
    ttl.setf(std::ios::fixed);
    ttl.precision(3);
    ttl << "5x5 Charge Neighborhood (" << label << unit << ")  |  Event " << evt
        << "\nHit: (" << trueX << ", " << trueY << ") mm,  Pixel: (" << pixelX << ", " << pixelY << ") mm";
    frame->SetTitle(ttl.str().c_str());

    TLegend* leg = new TLegend(0.14, 0.86, 0.54, 0.94);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->AddEntry((TObject*)nullptr, (std::string("Data: ") + label + unit).c_str(), "");
    leg->AddEntry((TObject*)nullptr, (std::string("isPixelHit = ") + (isPixelHit ? "1" : "0")).c_str(), "");
    if (std::isfinite(edep)) {
      std::ostringstream oss; oss.setf(std::ios::scientific); oss.precision(2);
      oss << "Edep = " << edep << " MeV";
      leg->AddEntry((TObject*)nullptr, oss.str().c_str(), "");
    }
    leg->Draw();
  }

  ccan->Modified();
  ccan->Update();

  // Keep canvas for interactive usage; close file handle
  if (invertGlobalPalette) { TColor::InvertPalette(); }
  f->Close();
  delete f;
}

void plotChargeNeighborhood9x9(const char* filename, Long64_t eventIndex) {
  // Backward compat: now draw 5x5 ROI
  plotChargeNeighborhood5x5(filename, eventIndex, "fraction", "", "Q_n");
}

void plotChargeNeighborhood9x9(const char* filename, Long64_t eventIndex, const char* dataKind) {
  // Backward compat: now draw 5x5 ROI
  plotChargeNeighborhood5x5(filename, eventIndex, dataKind, "", "Q_n");
}

// Backward-compatible wrappers with previous function name
void plotChargeNeighborhood5x5(const char* filename, Long64_t eventIndex) {
  plotChargeNeighborhood5x5(filename, eventIndex, "fraction", "", "Q_n");
}

void plotChargeNeighborhood5x5(const char* filename, Long64_t eventIndex, const char* dataKind) {
  plotChargeNeighborhood5x5(filename, eventIndex, dataKind, "", "Q_n");
}

// Forward declaration of PDF generator used by default entrypoint
int plotChargeNeighborhood5x5_pages(const char* rootFilePath,
                                    Long64_t nPages,
                                    const char* dataKind,
                                    const char* outPdfPath,
                                    const char* chargeBranch);

// Default entrypoint now generates a multi-page PDF (no single-event PNG)
void plotChargeNeighborhood() {
  plotChargeNeighborhood5x5_pages("epicChargeSharing.root", 100, "coulomb", "charge_neighborhoods.pdf", "Q_n");
}

// New entrypoints: pass N to generate a multi-page PDF (one event per page)
// Forward declaration (defined later in this file)
int plotChargeNeighborhood5x5_pages(const char* rootFilePath,
                                    Long64_t nPages,
                                    const char* dataKind,
                                    const char* outPdfPath,
                                    const char* chargeBranch);

void plotChargeNeighborhood(Long64_t nPages) {
  plotChargeNeighborhood5x5_pages("epicChargeSharing.root", nPages, "coulomb", "charge_neighborhoods.pdf", "Q_n");
}

void plotChargeNeighborhood(const char* rootFilePath, Long64_t nPages) {
  plotChargeNeighborhood5x5_pages(rootFilePath, nPages, "coulomb", "charge_neighborhoods.pdf", "Q_n");
}

// New overloads: choose between F_i (default) and Q_i via a friendly selector
void plotChargeNeighborhood(const char* quantity) {
  const char* kind = QuantityToDataKind(quantity); // "F_i"/"Q_i" -> "fraction"/"coulomb"
  plotChargeNeighborhood5x5_pages("epicChargeSharing.root", 100, kind, "charge_neighborhoods.pdf", "Q_n");
}

void plotChargeNeighborhood(Long64_t nPages, const char* quantity) {
  const char* kind = QuantityToDataKind(quantity);
  plotChargeNeighborhood5x5_pages("epicChargeSharing.root", nPages, kind, "charge_neighborhoods.pdf", "Q_n");
}

void plotChargeNeighborhood(const char* rootFilePath, Long64_t nPages, const char* quantity) {
  const char* kind = QuantityToDataKind(quantity);
  plotChargeNeighborhood5x5_pages(rootFilePath, nPages, kind, "charge_neighborhoods.pdf", "Q_n");
}

// Compute and draw mean neighborhood (5x5 ROI) across all non-inside-pixel events with energy deposition
// dataKind: "fraction", "coulomb", or "distance"
void plotChargeNeighborhoodMean5x5(const char* rootFilePath = "epicChargeSharing.root",
                                   const char* dataKind = "fraction",
                                   const char* outImagePath = "",
                                   const char* chargeBranch = "Q_f")
{
  gStyle->SetOptStat(0);
  gStyle->SetPalette(kBird);

  TFile* f = TryOpenWithFallbacks(rootFilePath);
  if (!f) { std::cerr << "ERROR: Cannot open any ROOT file." << std::endl; return; }

  double pixelSizeMm = 0.0, pixelSpacingMm = 0.0, detSizeMm = 0.0, pixelCornerOffsetMm = 0.0; int numPerSide = 0;
  try {
    pixelSizeMm         = ReadDoubleNamed(f, "GridPixelSize_mm");
    pixelSpacingMm      = ReadDoubleNamed(f, "GridPixelSpacing_mm");
    pixelCornerOffsetMm = ReadDoubleNamed(f, "GridPixelCornerOffset_mm");
    detSizeMm           = ReadDoubleNamed(f, "GridDetectorSize_mm");
    numPerSide          = ReadIntNamed  (f, "GridNumBlocksPerSide");
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl; f->Close(); delete f; return;
  }

  TTree* tree = dynamic_cast<TTree*>(f->Get("Hits"));
  if (!tree) { std::cerr << "ERROR: 'Hits' tree not found." << std::endl; f->Close(); delete f; return; }

  // Branches
  double trueX = 0.0, trueY = 0.0, pixelX = 0.0, pixelY = 0.0, edep = 0.0;
  Bool_t isPixelHit = 0;
  std::vector<double>* Fi = nullptr;
  std::vector<double>* Qi = nullptr;
  std::vector<double>* Qf = nullptr;

  tree->SetBranchStatus("*", 0);
  auto enableIf = [&](const char* b){ if (tree->GetBranch(b)) tree->SetBranchStatus(b, 1); };
  enableIf("TrueX"); enableIf("TrueY"); enableIf("PixelX"); enableIf("PixelY"); enableIf("Edep"); enableIf("isPixelHit"); enableIf("Q_i"); enableIf("Q_f"); enableIf("F_i");
  if (tree->GetBranch("TrueX")) tree->SetBranchAddress("TrueX", &trueX);
  if (tree->GetBranch("TrueY")) tree->SetBranchAddress("TrueY", &trueY);
  if (tree->GetBranch("PixelX")) tree->SetBranchAddress("PixelX", &pixelX);
  if (tree->GetBranch("PixelY")) tree->SetBranchAddress("PixelY", &pixelY);
  if (tree->GetBranch("Edep"))  tree->SetBranchAddress("Edep", &edep);
  if (tree->GetBranch("isPixelHit")) tree->SetBranchAddress("isPixelHit", &isPixelHit);
  if (tree->GetBranch("Q_i")) tree->SetBranchAddress("Q_i", &Qi);
  if (tree->GetBranch("Q_f")) tree->SetBranchAddress("Q_f", &Qf);
  if (tree->GetBranch("F_i")) tree->SetBranchAddress("F_i", &Fi);

  if (!Qi && !Qf && !Fi) { std::cerr << "ERROR: Missing 'Q_i', 'Q_f', or 'F_i' branch" << std::endl; f->Close(); delete f; return; }

  const Long64_t nEntries = tree->GetEntries();
  if (nEntries <= 0) { std::cerr << "ERROR: Empty 'Hits' tree" << std::endl; f->Close(); delete f; return; }

  // Determine dim using chosen branch preference with sensible fallbacks
  std::string chosen = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Q_f");
  auto pickVecPtr = [&]() -> const std::vector<double>* {
    if (chosen == "Q_f" && Qf) return Qf;
    if (chosen == "Q_i" && Qi) return Qi;
    if (chosen == "F_i" && Fi) return Fi;
    if (Qf) return Qf;
    if (Qi) return Qi;
    if (Fi) return Fi;
    return nullptr;
  };
  // Determine dim
  int dim = -1; Long64_t firstValidEvt = -1;
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    const std::vector<double>* vec = pickVecPtr();
    if (!vec || vec->empty()) continue;
    const int d = static_cast<int>(std::lround(std::sqrt(static_cast<double>(vec->size()))));
    if (d*d != static_cast<int>(vec->size())) continue;
    dim = d; firstValidEvt = i; break;
  }
  if (dim <= 0) { std::cerr << "ERROR: Unable to determine neighborhood dimension" << std::endl; f->Close(); delete f; return; }

  if (dim < 5) { std::cerr << "ERROR: Neighborhood grid dimension is " << dim << ", need at least 5" << std::endl; f->Close(); delete f; return; }

  const int c = dim/2; const int i0 = c - 2; const int i1 = c + 2; const int j0 = c - 2; const int j1 = c + 2;

  // Accumulators for mean over 5x5 ROI with NaN-like ignore via counts
  const int roiSize = 5;
  std::vector<double> sum(roiSize*roiSize, 0.0);
  std::vector<int>    cnt(roiSize*roiSize, 0);

  const double eChargeC = 1.602176634e-19;
  const double pairEnergyEV = 3.60;
  const double MeV_to_eV = 1.0e6;

  // Select only non-inside-pixel events with Edep>0
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    if (!std::isfinite(edep) || edep <= 0) continue;
    if (isPixelHit) continue;

    const double qTotalC = edep * MeV_to_eV / pairEnergyEV * eChargeC;
    // Accumulate only 5x5 ROI, transposing to grid[y][x]
    for (int di = i0; di <= i1; ++di) {
      for (int dj = j0; dj <= j1; ++dj) {
        const int idxFlat = di * dim + dj;
        const int ii = di - i0; // 0..4
        const int jj = dj - j0; // 0..4
        const int idxGrid = ii * roiSize + jj;

        std::string kind = dataKind ? dataKind : "fraction";
        std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c){ return std::tolower(c); });
        const std::vector<double>* vec = pickVecPtr();
        const bool usingQi = (vec == Qi && Qi != nullptr);
        const bool usingQf = (vec == Qf && Qf != nullptr);
        if (!vec || vec->empty()) continue;

        const double raw = (*vec)[idxFlat];
        if (!(std::isfinite(raw)) || raw < 0.0) continue; // ignore -999 sentinel and invalid

        double val = 0.0;
        if (kind == "fraction") {
          val = (usingQi || usingQf) ? (qTotalC > 0 ? raw / qTotalC : 0.0) : raw;
        } else if (kind == "coulomb") {
          val = (usingQi || usingQf) ? raw : (raw * qTotalC);
        } else {
          const double cx = pixelX + (dj - c) * pixelSpacingMm;
          const double cy = pixelY + (di - c) * pixelSpacingMm;
          val = std::hypot(cx - trueX, cy - trueY);
        }
        sum[idxGrid] += val;
        cnt[idxGrid] += 1;
      }
    }
  }

  // Build mean grid (5x5) and detect valid positions
  std::vector<std::vector<double>> meanGrid(roiSize, std::vector<double>(roiSize, std::numeric_limits<double>::quiet_NaN()));
  std::vector<std::vector<bool>>   valid(roiSize, std::vector<bool>(roiSize, false));
  int nValidCells = 0;
  for (int i = 0; i < roiSize; ++i) {
    for (int j = 0; j < roiSize; ++j) {
      const int idx = i * roiSize + j;
      if (cnt[idx] > 0) {
        meanGrid[i][j] = sum[idx] / static_cast<double>(cnt[idx]);
        valid[i][j] = true; ++nValidCells;
      }
    }
  }

  if (nValidCells <= 0) { std::cerr << "ERROR: No valid cells for mean calculation" << std::endl; f->Close(); delete f; return; }

  // Choose reference pixel center from firstValidEvt
  double refPixelX = 0.0, refPixelY = 0.0; Long64_t refEvt = firstValidEvt >= 0 ? firstValidEvt : 0;
  tree->GetEntry(refEvt);
  refPixelX = pixelX; refPixelY = pixelY;

  // Determine color range
  std::string kind = dataKind ? dataKind : "fraction";
  std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c){ return std::tolower(c); });
  double vmin = 0.0, vmax = 1.0;
  if (kind == "fraction") {
    vmin = 0.0; vmax = 0.0; bool any=false;
    for (int i = 0; i < roiSize; ++i) for (int j = 0; j < roiSize; ++j) if (valid[i][j]) { vmax = std::max(vmax, meanGrid[i][j]); any=true; }
    if (!any) vmax = 1.0;
  } else {
    vmin = +std::numeric_limits<double>::infinity();
    vmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < roiSize; ++i) for (int j = 0; j < roiSize; ++j) if (valid[i][j]) { vmin = std::min(vmin, meanGrid[i][j]); vmax = std::max(vmax, meanGrid[i][j]); }
    if (!(vmax > vmin)) { vmin = 0.0; vmax = 1.0; }
  }

  // Canvas
  const double halfGridExtent = 2.5 * pixelSpacingMm;
  const int canvasSize = 900;
  TCanvas* ccan = new TCanvas("c_charge_neighborhood5x5_mean", "", canvasSize, canvasSize);
  ccan->SetLeftMargin(0.12); ccan->SetRightMargin(0.15); ccan->SetTopMargin(0.12); ccan->SetBottomMargin(0.12);
  ccan->SetFixedAspectRatio();
  const double xMin = refPixelX - halfGridExtent, xMax = refPixelX + halfGridExtent;
  const double yMin = refPixelY - halfGridExtent, yMax = refPixelY + halfGridExtent;
  TH1* frame = ccan->DrawFrame(xMin, yMin, xMax, yMax, ";x [mm];y [mm]");
  frame->GetXaxis()->SetTitleOffset(1.2); frame->GetYaxis()->SetTitleOffset(1.4);
  ccan->cd(); gPad->SetFixedAspectRatio();

  const bool invertGlobalPalette = (kind == "distance");
  const bool invertPalette = false;
  if (invertGlobalPalette) { TColor::InvertPalette(); }

  // Palette axis reflecting [vmin,vmax] for the chosen kind
  {
    std::string branchTitle;
    if (kind != "distance") {
      if (chosen == "Q_f" && Qf) branchTitle = "Q_f";
      else if (chosen == "Q_i" && Qi) branchTitle = "Q_i";
      else if (chosen == "F_i" && Fi) branchTitle = "F_i";
      else if (Qf) branchTitle = "Q_f";
      else if (Qi) branchTitle = "Q_i";
      else if (Fi) branchTitle = "F_i";
    }
    std::string zTitle = (kind == "fraction" ? std::string("Charge Fraction") : (kind == "coulomb" ? std::string("Charge C") : std::string("Distance mm")));
    if (!branchTitle.empty()) zTitle += " (" + branchTitle + ")";
    TH2D* paletteHist = new TH2D("pal_charge_neighborhood5x5_mean", "", 2, xMax+1, xMax+2, 2, yMin, yMax);
    paletteHist->SetMinimum(vmin);
    paletteHist->SetMaximum(vmax);
    paletteHist->GetZaxis()->SetTitle(zTitle.c_str());
    paletteHist->GetZaxis()->SetTitleSize(0.040);
    paletteHist->GetZaxis()->SetLabelSize(0.032);
    paletteHist->Draw("COLZ SAME");
    gPad->Update();
    TPaletteAxis* pal = (TPaletteAxis*)paletteHist->GetListOfFunctions()->FindObject("palette");
    if (pal) {
      pal->SetLabelSize(0.032);
      pal->SetTitleOffset(1.30);
      pal->SetBorderSize(1);
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
  }
  const double alpha = 0.80;
  for (int i = 0; i < roiSize; ++i) {
    for (int j = 0; j < roiSize; ++j) {
      if (!valid[i][j]) continue;
      const double rel_x = (j - 2) * pixelSpacingMm;
      const double rel_y = (i - 2) * pixelSpacingMm;
      const double cx = refPixelX + rel_x;
      const double cy = refPixelY + rel_y;
      const double x1 = cx - pixelSpacingMm/2.0, x2 = cx + pixelSpacingMm/2.0;
      const double y1 = cy - pixelSpacingMm/2.0, y2 = cy + pixelSpacingMm/2.0;

      const double v = meanGrid[i][j];
      const int color = ValueToPaletteColor(v, vmin, vmax, invertPalette);
      TBox* blk = new TBox(x1, y1, x2, y2);
      blk->SetFillColorAlpha(color, alpha);
      blk->SetLineColor(kBlack); blk->SetLineWidth(1); blk->Draw("same");

      std::ostringstream vs;
      if (kind == "fraction") { vs.setf(std::ios::fixed); vs.precision(3); vs << v; }
      else if (kind == "coulomb") { vs.setf(std::ios::scientific); vs.precision(2); vs << v; }
      else { vs.setf(std::ios::fixed); vs.precision(3); vs << v; }
      TLatex* txt = new TLatex(cx, cy - pixelSizeMm/2.0 - 0.08, vs.str().c_str());
      txt->SetTextAlign(22); txt->SetTextColor(kWhite); txt->SetTextSize(0.025); txt->Draw("same");
    }
  }

  // Pixel squares and borders
  std::vector<std::vector<bool>> validMask = valid;
  for (int i = 0; i < roiSize; ++i) {
    for (int j = 0; j < roiSize; ++j) {
      if (!validMask[i][j]) continue;
      const double rel_x = (j - 2) * pixelSpacingMm;
      const double rel_y = (i - 2) * pixelSpacingMm;
      const double cx = refPixelX + rel_x; const double cy = refPixelY + rel_y;
      const double x1 = cx - pixelSizeMm/2.0, x2 = cx + pixelSizeMm/2.0;
      const double y1 = cy - pixelSizeMm/2.0, y2 = cy + pixelSizeMm/2.0;
      TBox* px = new TBox(x1, y1, x2, y2); px->SetFillStyle(0); px->SetLineColor(kBlack); px->SetLineWidth(2); px->Draw("same");

      if (j - 1 < 0 || !validMask[i][j-1]) { TLine* ln = new TLine(x1, y1, x1, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (j + 1 >= dim || !validMask[i][j+1]) { TLine* ln = new TLine(x2, y1, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (i + 1 >= dim || !validMask[i+1][j]) { TLine* ln = new TLine(x1, y1, x2, y1); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      if (i - 1 < 0 || !validMask[i-1][j]) { TLine* ln = new TLine(x1, y2, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
    }
  }

  // Title
  {
    std::ostringstream ttl; ttl.setf(std::ios::fixed); ttl.precision(3);
    //ttl << "Mean 5x5 Charge Neighborhood (" << (kind == "fraction" ? "Charge Fraction" : (kind == "coulomb" ? "Charge" : "Distance")) << (kind == "coulomb" ? " C" : (kind == "distance" ? " mm" : "")) << ")";
    frame->SetTitle(ttl.str().c_str());
  }

  ccan->Modified(); ccan->Update();
  if (outImagePath && std::strlen(outImagePath) > 0) {
    ccan->SaveAs(outImagePath);
  } else {
    std::string fname = Form("charge_neighborhood5x5_mean_%s.png", kind.c_str());
    ccan->SaveAs(fname.c_str());
  }

  if (invertGlobalPalette) { TColor::InvertPalette(); }
  f->Close(); delete f;
}

// Backward-compatible wrapper
void plotChargeNeighborhoodMean9x9(const char* rootFilePath = "epicChargeSharing.root",
                                   const char* dataKind = "fraction",
                                   const char* outImagePath = "")
{
  plotChargeNeighborhoodMean5x5(rootFilePath, dataKind, outImagePath);
}



// Generate a multi-page PDF with up to N per-event 5x5 charge neighborhood plots
// dataKind: "fraction", "coulomb", or "distance"
int plotChargeNeighborhood5x5_pages(const char* rootFilePath = "epicChargeSharing.root",
                                    Long64_t nPages = 100,
                                    const char* dataKind = "coulomb",
                                    const char* outPdfPath = "charge_neighborhoods.pdf",
                                    const char* chargeBranch = "Q_n")
{
  gROOT->SetBatch(true);
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPalette(kBird);

  TFile* f = TryOpenWithFallbacks(rootFilePath);
  if (!f) {
    std::cerr << "ERROR: Cannot open any ROOT file (tried provided path and common fallbacks)." << std::endl;
    return 1;
  }

  double pixelSizeMm         = 0.0;
  double pixelSpacingMm      = 0.0;
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
    f->Close(); delete f; return 2;
  }

  TTree* tree = dynamic_cast<TTree*>(f->Get("Hits"));
  if (!tree) { std::cerr << "ERROR: 'Hits' tree not found in ROOT file." << std::endl; f->Close(); delete f; return 3; }

  // Inputs
  double trueX = std::numeric_limits<double>::quiet_NaN();
  double trueY = std::numeric_limits<double>::quiet_NaN();
  double pixelX = std::numeric_limits<double>::quiet_NaN();
  double pixelY = std::numeric_limits<double>::quiet_NaN();
  double edep   = std::numeric_limits<double>::quiet_NaN();
  Bool_t isPixelHit = 0;
  std::vector<double>* Fi = nullptr;
  std::vector<double>* Qi = nullptr;
  std::vector<double>* Qf = nullptr;

  // Speed up I/O
  tree->SetBranchStatus("*", 0);
  auto enableIf = [&](const char* b){ if (tree->GetBranch(b)) tree->SetBranchStatus(b, 1); };
  enableIf("TrueX"); enableIf("TrueY"); enableIf("PixelX"); enableIf("PixelY");
  enableIf("Edep"); enableIf("isPixelHit"); enableIf("Q_i"); enableIf("Q_n"); enableIf("Q_f"); enableIf("F_i");
  if (tree->GetBranch("TrueX")) tree->SetBranchAddress("TrueX", &trueX);
  if (tree->GetBranch("TrueY")) tree->SetBranchAddress("TrueY", &trueY);
  if (tree->GetBranch("PixelX")) tree->SetBranchAddress("PixelX", &pixelX);
  if (tree->GetBranch("PixelY")) tree->SetBranchAddress("PixelY", &pixelY);
  if (tree->GetBranch("Edep")) tree->SetBranchAddress("Edep", &edep);
  if (tree->GetBranch("isPixelHit")) tree->SetBranchAddress("isPixelHit", &isPixelHit);
  if (tree->GetBranch("Q_i")) tree->SetBranchAddress("Q_i", &Qi);
  if (tree->GetBranch("Q_n")) tree->SetBranchAddress("Q_n", &Qf); // temp reuse below, we'll correct shortly
  if (tree->GetBranch("Q_f")) tree->SetBranchAddress("Q_f", &Qf);
  if (tree->GetBranch("F_i")) tree->SetBranchAddress("F_i", &Fi);
  if (!Qi && !Qf && !Fi) { std::cerr << "ERROR: Required neighborhood data branch 'Q_i', 'Q_f', or 'F_i' not found in tree." << std::endl; f->Close(); delete f; return 4; }

  const Long64_t nEntries = tree->GetEntries();
  if (nEntries <= 0) { std::cerr << "ERROR: Hits tree is empty." << std::endl; f->Close(); delete f; return 5; }

  // Prepare canvas and PDF
  const int canvasSize = 900;
  TCanvas ccan("c_charge_neighborhood5x5_pages", "", canvasSize, canvasSize);
  ccan.SetLeftMargin(0.12);
  ccan.SetRightMargin(0.15);
  ccan.SetTopMargin(0.12);
  ccan.SetBottomMargin(0.12);
  ccan.SetFixedAspectRatio();
  std::string pdfPath = (outPdfPath && std::strlen(outPdfPath) > 0) ? std::string(outPdfPath) : std::string("charge_neighborhoods.pdf");
  ccan.Print((pdfPath + "[").c_str());

  // Normalize dataKind
  std::string kind = dataKind ? dataKind : "coulomb";
  std::transform(kind.begin(), kind.end(), kind.begin(), [](unsigned char c){ return std::tolower(c); });
  std::string label = (kind == "coulomb") ? "Charge" : (kind == "distance" ? "Distance" : "Charge Fraction");
  std::string unit  = (kind == "coulomb") ? " C" : (kind == "distance" ? " mm" : "");

  // Keep colors consistent with palette: invert globally for distance so palette and boxes match
  const bool invertGlobalPalette = (kind == "distance");
  if (invertGlobalPalette) { TColor::InvertPalette(); }

  const double halfGridExtent = 2.5 * pixelSpacingMm;
  const double eChargeC = 1.602176634e-19; // Coulombs
  const double pairEnergyEV = 3.60;        // eV/pair in Si
  const double MeV_to_eV = 1.0e6;

  Long64_t pages = 0;
  for (Long64_t i = 0; i < nEntries && pages < nPages; ++i) {
    tree->GetEntry(i);
  std::string chosen = (chargeBranch && chargeBranch[0] != '\0') ? std::string(chargeBranch) : std::string("Q_n");
    const std::vector<double>* vec = nullptr;
  if (chosen == "Q_f" && Qf) vec = Qf;
  else if (chosen == "Q_n" && Qf) vec = Qf; // because we temp-bound Q_n to Qf above
    else if (chosen == "Q_i" && Qi) vec = Qi;
    else if (chosen == "F_i" && Fi) vec = Fi;
  else if (Qf) vec = Qf;
    else if (Qi) vec = Qi;
    else if (Fi) vec = Fi;
    const bool usingQi = (vec == Qi && Qi != nullptr);
    const bool usingQf = (vec == Qf && Qf != nullptr);
    if (!vec || vec->empty()) continue;

    const size_t nVals = vec->size();
    const int dim = static_cast<int>(std::lround(std::sqrt(static_cast<double>(nVals))));
    if (dim * dim != static_cast<int>(nVals) || dim < 5) continue;

    // Build transposed grid[y][x] from chosen source (Qi or Fi)
    std::vector<std::vector<double>> grid(dim, std::vector<double>(dim, std::numeric_limits<double>::quiet_NaN()));
    for (int di = 0; di < dim; ++di) {
      for (int dj = 0; dj < dim; ++dj) {
        const int idx = di * dim + dj;
        grid[dj][di] = (*vec)[idx];
      }
    }

    const int c = dim / 2;
    const int i0 = c - 2, i1 = c + 2;
    const int j0 = c - 2, j1 = c + 2;

    double qTotalC = 0.0;
    if (std::isfinite(edep) && edep > 0) {
      qTotalC = edep * MeV_to_eV / pairEnergyEV * eChargeC;
    }
    if (std::isfinite(edep) && edep <= 0 && kind == "fraction") {
      for (int ii = i0; ii <= i1; ++ii) for (int jj = j0; jj <= j1; ++jj) if (grid[ii][jj] >= 0.0) grid[ii][jj] = 0.0;
    }

    // Determine color mapping range in ROI
    double vmin = 0.0, vmax = 1.0;
    if (kind == "fraction") {
      vmin = 0.0; vmax = 0.0; bool any=false;
      for (int ii=i0; ii<=i1; ++ii) for (int jj=j0; jj<=j1; ++jj) {
        double raw = grid[ii][jj]; if (!(std::isfinite(raw) && raw >= 0.0)) continue;
        double frac = 0.0;
        if (std::isfinite(edep) && edep <= 0) frac = 0.0;
        else if (usingQi || usingQf) { if (qTotalC > 0) frac = raw / qTotalC; else continue; }
        else frac = raw;
        vmax = std::max(vmax, frac); any = true;
      }
      if (!any) vmax = 1.0;
    } else if (kind == "coulomb") {
      vmin = +std::numeric_limits<double>::infinity(); vmax = -std::numeric_limits<double>::infinity();
      for (int ii=i0; ii<=i1; ++ii) for (int jj=j0; jj<=j1; ++jj) {
        double raw = grid[ii][jj]; if (!(std::isfinite(raw) && raw >= 0.0)) continue;
        double q = (usingQi || usingQf) ? raw : (raw * qTotalC);
        vmin = std::min(vmin, q); vmax = std::max(vmax, q);
      }
      if (!(vmax > vmin)) { vmin = 0.0; vmax = 1.0; }
    } else { // distance
      vmin = +std::numeric_limits<double>::infinity(); vmax = -std::numeric_limits<double>::infinity();
      for (int ii=i0; ii<=i1; ++ii) for (int jj=j0; jj<=j1; ++jj) { double fval = grid[ii][jj]; if (!(std::isfinite(fval) && fval >= 0.0)) continue; double cx = pixelX + (jj - c) * pixelSpacingMm; double cy = pixelY + (ii - c) * pixelSpacingMm; double d = std::hypot(cx - trueX, cy - trueY); vmin = std::min(vmin, d); vmax = std::max(vmax, d); }
      if (!(vmax > vmin)) { vmin = 0.0; vmax = 1.0; }
    }

    ccan.Clear();
    const double xMin = pixelX - halfGridExtent;
    const double xMax = pixelX + halfGridExtent;
    const double yMin = pixelY - halfGridExtent;
    const double yMax = pixelY + halfGridExtent;
    TH1* frame = ccan.DrawFrame(xMin, yMin, xMax, yMax, ";x [mm];y [mm]");
    frame->GetXaxis()->SetTitleOffset(1.2);
    frame->GetYaxis()->SetTitleOffset(1.4);
    ccan.cd();
    gPad->SetFixedAspectRatio();

    // Add a color bar matching [vmin,vmax] and the chosen quantity label
    {
      std::string branchTitle;
      if (kind != "distance") {
        branchTitle = usingQf ? "Q_f" : (usingQi ? "Q_i" : "F_i");
      }
      std::string zTitle = label + unit;
      if (!branchTitle.empty()) zTitle += " (" + branchTitle + ")";
      TH2D* paletteHist = new TH2D(Form("pal_charge_neighborhood5x5_pages_%lld", i), "", 2, xMax+1, xMax+2, 2, yMin, yMax);
      paletteHist->SetMinimum(vmin);
      paletteHist->SetMaximum(vmax);
      paletteHist->GetZaxis()->SetTitle(zTitle.c_str());
      paletteHist->GetZaxis()->SetTitleSize(0.040);
      paletteHist->GetZaxis()->SetLabelSize(0.032);
      paletteHist->Draw("COLZ SAME");
      gPad->Update();
      TPaletteAxis* pal = (TPaletteAxis*)paletteHist->GetListOfFunctions()->FindObject("palette");
      if (pal) {
        pal->SetLabelSize(0.032);
        pal->SetTitleOffset(1.30);
        pal->SetBorderSize(1);
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
    }

    const bool invertPalette = false;
    const double alpha = 0.80;
    // Build ROI valid mask (5x5)
    const int roiSize = 5;
    std::vector<std::vector<bool>> valid(roiSize, std::vector<bool>(roiSize, false));
    for (int ii = 0; ii < roiSize; ++ii) {
      for (int jj = 0; jj < roiSize; ++jj) {
        const int gi = i0 + ii;
        const int gj = j0 + jj;
        const double fval = grid[gi][gj];
        if (std::isfinite(fval) && fval >= 0.0) valid[ii][jj] = true;
      }
    }

    // Draw colored blocks
    for (int ii = 0; ii < roiSize; ++ii) {
      for (int jj = 0; jj < roiSize; ++jj) {
        if (!valid[ii][jj]) continue;
        const int gi = i0 + ii;
        const int gj = j0 + jj;
        const double rel_x = (gj - c) * pixelSpacingMm;
        const double rel_y = (gi - c) * pixelSpacingMm;
        const double cx = pixelX + rel_x;
        const double cy = pixelY + rel_y;
        const double x1 = cx - pixelSpacingMm/2.0;
        const double x2 = cx + pixelSpacingMm/2.0;
        const double y1 = cy - pixelSpacingMm/2.0;
        const double y2 = cy + pixelSpacingMm/2.0;

        double shownVal = 0.0;
        if (kind == "fraction") {
          if (std::isfinite(edep) && edep <= 0) shownVal = 0.0;
          else shownVal = (usingQi || usingQf) ? (qTotalC > 0 ? grid[gi][gj] / qTotalC : 0.0) : grid[gi][gj];
        } else if (kind == "coulomb") {
          shownVal = (usingQi || usingQf) ? grid[gi][gj] : (grid[gi][gj] * qTotalC);
        } else shownVal = std::hypot(cx - trueX, cy - trueY);

        const int color = ValueToPaletteColor(shownVal, vmin, vmax, invertPalette);
        TBox* blk = new TBox(x1, y1, x2, y2);
        blk->SetFillColorAlpha(color, alpha);
        blk->SetLineColor(kBlack);
        blk->SetLineWidth(1);
        blk->Draw("same");

        std::ostringstream vs;
        if (kind == "fraction") { vs.setf(std::ios::fixed); vs.precision(3); vs << shownVal; }
        else if (kind == "coulomb") { vs.setf(std::ios::scientific); vs.precision(2); vs << shownVal; }
        else { vs.setf(std::ios::fixed); vs.precision(3); vs << shownVal; }
        TLatex* txt = new TLatex(cx, cy - pixelSizeMm/2.0 - 0.08, vs.str().c_str());
        txt->SetTextAlign(22);
        txt->SetTextColor(kWhite);
        txt->SetTextSize(0.025);
        txt->Draw("same");
      }
    }

    // Pixel squares
    for (int ii = 0; ii < roiSize; ++ii) {
      for (int jj = 0; jj < roiSize; ++jj) {
        if (!valid[ii][jj]) continue;
        const int gi = i0 + ii;
        const int gj = j0 + jj;
        const double rel_x = (gj - c) * pixelSpacingMm;
        const double rel_y = (gi - c) * pixelSpacingMm;
        const double cx = pixelX + rel_x;
        const double cy = pixelY + rel_y;
        const double x1 = cx - pixelSizeMm/2.0;
        const double x2 = cx + pixelSizeMm/2.0;
        const double y1 = cy - pixelSizeMm/2.0;
        const double y2 = cy + pixelSizeMm/2.0;
        TBox* px = new TBox(x1, y1, x2, y2);
        px->SetFillStyle(0);
        px->SetLineColor(kBlack);
        px->SetLineWidth(2);
        px->Draw("same");
      }
    }

    // Borders between valid/invalid cells in ROI
    for (int ii = 0; ii < roiSize; ++ii) {
      for (int jj = 0; jj < roiSize; ++jj) {
        if (!valid[ii][jj]) continue;
        const int gi = i0 + ii;
        const int gj = j0 + jj;
        const double rel_x = (gj - c) * pixelSpacingMm;
        const double rel_y = (gi - c) * pixelSpacingMm;
        const double cx = pixelX + rel_x;
        const double cy = pixelY + rel_y;
        const double x1 = cx - pixelSpacingMm/2.0;
        const double x2 = cx + pixelSpacingMm/2.0;
        const double y1 = cy - pixelSpacingMm/2.0;
        const double y2 = cy + pixelSpacingMm/2.0;

        // left
        if (jj - 1 < 0 || !valid[ii][jj-1]) { TLine* ln = new TLine(x1, y1, x1, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
        // right
        if (jj + 1 >= roiSize || !valid[ii][jj+1]) { TLine* ln = new TLine(x2, y1, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
        // bottom
        if (ii + 1 >= roiSize || !valid[ii+1][jj]) { TLine* ln = new TLine(x1, y1, x2, y1); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
        // top
        if (ii - 1 < 0 || !valid[ii-1][jj]) { TLine* ln = new TLine(x1, y2, x2, y2); ln->SetLineColor(kBlack); ln->SetLineWidth(1); ln->Draw("same"); }
      }
    }

    // Hit marker
    const double hitDotRadius = pixelSizeMm / 6.0;
    DrawCircleMm(trueX, trueY, hitDotRadius, kRed+1, kRed+1);

    // Title: just the event number
    frame->SetTitle(Form("Event %lld Charge Neighborhood", i));

    ccan.Modified();
    ccan.Update();
    ccan.Print(pdfPath.c_str());
    pages++;
  }

  ccan.Print((pdfPath + "]").c_str());
  if (invertGlobalPalette) { TColor::InvertPalette(); }
  f->Close(); delete f;
  ::Info("plotChargeNeighborhood5x5_pages", "Generated %lld pages to %s (scanned %lld events).", pages, pdfPath.c_str(), nEntries);
  return 0;
}

// Convenience wrappers
int plotChargeNeighborhoodPages(const char* rootFilePath = "epicChargeSharing.root",
                                Long64_t nPages = 100) {
  return plotChargeNeighborhood5x5_pages(rootFilePath, nPages, "coulomb", "charge_neighborhoods.pdf", "Q_f");
}

int plotChargeNeighborhoodPDF(const char* rootFilePath = "epicChargeSharing.root",
                              Long64_t nPages = 100,
                              const char* dataKind = "coulomb",
                              const char* outPdfPath = "charge_neighborhoods.pdf") {
  return plotChargeNeighborhood5x5_pages(rootFilePath, nPages, dataKind, outPdfPath, "Q_f");
}
