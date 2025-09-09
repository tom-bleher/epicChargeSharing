#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TBranch.h>
#include <TPaveStats.h>
#include <TLatex.h>
#include <TList.h>
#include <TText.h>
#include <TNamed.h>
#include <TDirectory.h>
#include <TROOT.h>

#include <string>
#include <iostream>
#include <cmath>

namespace {

std::string extractUnit(const std::string &title) {
  size_t start = title.find("[");
  size_t end = title.find("]");
  if (start != std::string::npos && end != std::string::npos && end > start) {
    return title.substr(start, end - start + 1);
  }
  return "";
}

std::string formatBranchLabel(const std::string &branchName) {
  std::string formatted = branchName;
  // Replace underscores with spaces for readability
  for (char &ch : formatted) {
    if (ch == '_') ch = ' ';
  }
  return formatted;
}

std::string getAxisTitle(const std::string &branchName,
                         const std::string &branchTitle,
                         double pixelPitchMm) {
  std::string unit = extractUnit(branchTitle);
  std::string formatted = formatBranchLabel(branchName);

  if (!unit.empty()) {
    return formatted + " " + unit + " (bin size: " + std::to_string(pixelPitchMm) + " mm)";
  }
  return formatted + " (bin size: " + std::to_string(pixelPitchMm) + " mm)";
}

bool ensureOutputDir(const std::string &outputDir) {
  void *dir = gSystem->OpenDirectory(outputDir.c_str());
  if (dir) {
    gSystem->FreeDirectory(dir);
    return true;
  }
  int status = gSystem->mkdir(outputDir.c_str(), kTRUE);
  return status == 0;
}

bool plotBranchWithPitch(TFile *file,
                         TTree *tree,
                         const std::string &branchName,
                         const std::string &outputDir,
                         double pixelPitchMm) {
  if (!file || !tree) return false;

  TBranch *branch = tree->GetBranch(branchName.c_str());
  if (!branch) {
    std::cerr << "Error: Could not find branch '" << branchName << "'" << std::endl;
    return false;
  }

  std::string branchTitle = branch->GetTitle();

  // Determine data range from the tree
  Double_t minVal = tree->GetMinimum(branchName.c_str());
  Double_t maxVal = tree->GetMaximum(branchName.c_str());
  if (!std::isfinite(minVal) || !std::isfinite(maxVal)) {
    std::cerr << "Error: Non-finite min/max for branch '" << branchName << "'" << std::endl;
    return false;
  }

  // Use pixel pitch as bin size, pad by half-pitch on both ends so bin centers align with pixel centers
  double paddedMin = minVal - 0.5 * pixelPitchMm;
  double paddedMax = maxVal + 0.5 * pixelPitchMm;
  if (paddedMax <= paddedMin) {
    paddedMax = paddedMin + pixelPitchMm;
  }
  int nbins = static_cast<int>(std::ceil((paddedMax - paddedMin) / pixelPitchMm));
  if (nbins < 1) nbins = 1;
  double xmin = paddedMin;
  double xmax = paddedMin + nbins * pixelPitchMm; // ensure exact alignment

  // Create unique temporary hist name; ROOT will own the filled one
  std::string histName = std::string("h_") + branchName + "_temp";
  TH1D *hist = new TH1D(histName.c_str(), branchTitle.c_str(), nbins, xmin, xmax);

  // Fill via Draw to honor TTree selection mechanism
  std::string drawCmd = branchName + ">>" + histName;
  tree->Draw(drawCmd.c_str(), "", "goff");

  TH1D *filledHist = dynamic_cast<TH1D *>(gDirectory->Get(histName.c_str()));
  if (!filledHist) {
    std::cerr << "Error: Could not retrieve filled histogram for '" << branchName << "'" << std::endl;
    delete hist;
    return false;
  }

  filledHist->SetLineColor(kBlue + 1);
  filledHist->SetFillColor(kBlue - 9);
  filledHist->SetFillStyle(1001);
  filledHist->SetLineWidth(2);
  filledHist->SetTitle(branchTitle.c_str());
  filledHist->SetName(branchName.c_str());
  filledHist->GetXaxis()->SetTitle(getAxisTitle(branchName, branchTitle, pixelPitchMm).c_str());
  filledHist->GetYaxis()->SetTitle("Entries");
  filledHist->GetXaxis()->SetTitleSize(0.045);
  filledHist->GetYaxis()->SetTitleSize(0.045);
  filledHist->GetXaxis()->SetLabelSize(0.04);
  filledHist->GetYaxis()->SetLabelSize(0.04);
  filledHist->GetXaxis()->SetTitleOffset(1.1);
  filledHist->GetYaxis()->SetTitleOffset(1.0);

  TCanvas *canvas = new TCanvas((std::string("c_") + branchName).c_str(), branchTitle.c_str(), 1600, 900);
  canvas->SetMargin(0.08, 0.03, 0.12, 0.08);

  gStyle->SetOptStat(1111);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatColor(0);
  gStyle->SetStatFont(42);
  gStyle->SetStatFontSize(0.032);
  gStyle->SetStatX(0.94);
  gStyle->SetStatY(0.94);
  gStyle->SetStatFormat("6.3g");

  filledHist->Draw("HIST");
  canvas->Update();

  // Prepend branch name in stats box
  if (TPaveStats *stats = dynamic_cast<TPaveStats *>(filledHist->FindObject("stats"))) {
    stats->SetName("mystats");
    if (TList *listOfLines = stats->GetListOfLines()) {
      if (TText *tconst = stats->GetLineWith(filledHist->GetName())) {
        listOfLines->Remove(tconst);
      }
      TLatex *header = new TLatex(0, 0, branchName.c_str());
      header->SetTextFont(42);
      header->SetTextSize(0.032);
      listOfLines->AddFirst(header);
      stats->SetOptStat(1111);
      canvas->Modified();
    }
  }

  std::string svgPath = outputDir + "/" + branchName + ".svg";
  canvas->SaveAs(svgPath.c_str());

  delete hist;
  delete canvas;
  return true;
}

} // namespace

// Usage from ROOT:
//   root -l -b -q 'analysis/root/vis/plot_x_y_px.C("/abs/path/to/epicChargeSharing.root", "histograms")'
// Defaults are chosen for this repository's layout.
void plot_x_y_px(const char *filename = 
                   "/home/tom/Desktop/Putza/epicChargeSharing/build/epicChargeSharing.root",
                 const char *outdir = "histograms") {
  std::string filePath(filename ? filename : "");
  std::string outputDir(outdir ? outdir : "histograms");

  if (!ensureOutputDir(outputDir)) {
    std::cerr << "Error: Could not create or access output directory '" << outputDir << "'" << std::endl;
    return;
  }

  TFile *file = TFile::Open(filePath.c_str(), "READ");
  if (!file || file->IsZombie()) {
    std::cerr << "Error: Could not open file '" << filePath << "'" << std::endl;
    if (file) file->Close();
    return;
  }

  TTree *tree = dynamic_cast<TTree *>(file->Get("Hits"));
  if (!tree) {
    std::cerr << "Error: Could not find tree 'Hits' in file" << std::endl;
    file->Close();
    return;
  }

  // Read pixel pitch from metadata (fallback to 0.1 mm)
  double pixelPitchMm = 0.1;
  if (TNamed *pixelSizeObj = dynamic_cast<TNamed *>(file->Get("GridPixelSize_mm"))) {
    try {
      pixelPitchMm = std::stod(std::string(pixelSizeObj->GetTitle()));
    } catch (...) {
      std::cerr << "Warning: Could not parse GridPixelSize_mm; using default 0.1 mm" << std::endl;
    }
  }

  // Plot x_px and y_px only, with binning aligned to pixel pitch
  auto branchOrAlt = [&](const std::string &preferred,
                         const std::string &alt) -> std::string {
    if (tree->GetBranch(preferred.c_str())) return preferred;
    if (!alt.empty() && tree->GetBranch(alt.c_str())) return alt;
    return preferred; // will fail in plot routine with clear message
  };

  std::string xBranch = branchOrAlt("x_px", "PixelX");
  std::string yBranch = branchOrAlt("y_px", "PixelY");

  bool okX = plotBranchWithPitch(file, tree, xBranch, outputDir, pixelPitchMm);
  bool okY = plotBranchWithPitch(file, tree, yBranch, outputDir, pixelPitchMm);

  std::cout << "Completed plotting. x_px: " << (okX ? "OK" : "FAIL")
            << ", y_px: " << (okY ? "OK" : "FAIL") << std::endl;

  file->Close();
}

