// ROOT macro: makeCombinedDeltas.C
// Creates two branches on the Hits tree that combine pixel and non‑pixel deltas:
//   combined_delta_x = is_pixel_hit ? |x_px - x_hit| : |x_rec - x_hit|
//   combined_delta_y = is_pixel_hit ? |y_px - y_hit| : |y_rec - y_hit|
// where x_rec,y_rec are taken from 2D reconstruction if available, else 3D.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TError.h>
#include <TROOT.h>

#include <cmath>
#include <limits>
#include <string>

namespace {
  inline bool IsFiniteD(double v) { return std::isfinite(v); }
}

int makeCombinedDeltas(const char* filename = "../build/epicChargeSharingOutput.root")
{
  TFile* file = TFile::Open(filename, "UPDATE");
  if (!file || file->IsZombie()) {
    ::Error("makeCombinedDeltas", "Cannot open file: %s", filename);
    return 1;
  }

  TTree* tree = dynamic_cast<TTree*>(file->Get("Hits"));
  if (!tree) {
    ::Error("makeCombinedDeltas", "Hits tree not found in file: %s", filename);
    file->Close();
    delete file;
    return 2;
  }

  // Inputs
  double x_hit = 0.0, y_hit = 0.0;
  double x_px  = 0.0, y_px  = 0.0;
  Bool_t is_pixel_hit = kFALSE;

  // Optional reconstructed positions (2D and 3D)
  double x_rec_2d = std::numeric_limits<double>::quiet_NaN();
  double y_rec_2d = std::numeric_limits<double>::quiet_NaN();
  double x_rec_3d = std::numeric_limits<double>::quiet_NaN();
  double y_rec_3d = std::numeric_limits<double>::quiet_NaN();

  // Activate only needed branches (missing ones are ignored safely)
  tree->SetBranchStatus("*", 0);
  tree->SetBranchStatus("x_hit", 1);
  tree->SetBranchStatus("y_hit", 1);
  tree->SetBranchStatus("x_px", 1);
  tree->SetBranchStatus("y_px", 1);
  tree->SetBranchStatus("is_pixel_hit", 1);
  // Optional reco branches
  tree->SetBranchStatus("x_rec_2d", 1);
  tree->SetBranchStatus("y_rec_2d", 1);
  tree->SetBranchStatus("x_rec_3d", 1);
  tree->SetBranchStatus("y_rec_3d", 1);

  tree->SetBranchAddress("x_hit", &x_hit);
  tree->SetBranchAddress("y_hit", &y_hit);
  tree->SetBranchAddress("x_px", &x_px);
  tree->SetBranchAddress("y_px", &y_px);
  tree->SetBranchAddress("is_pixel_hit", &is_pixel_hit);

  // Set optional addresses only if the branch exists
  if (tree->GetBranch("x_rec_2d")) tree->SetBranchAddress("x_rec_2d", &x_rec_2d);
  if (tree->GetBranch("y_rec_2d")) tree->SetBranchAddress("y_rec_2d", &y_rec_2d);
  if (tree->GetBranch("x_rec_3d")) tree->SetBranchAddress("x_rec_3d", &x_rec_3d);
  if (tree->GetBranch("y_rec_3d")) tree->SetBranchAddress("y_rec_3d", &y_rec_3d);

  // Outputs
  const double INVALID_VALUE = std::numeric_limits<double>::quiet_NaN();
  double combined_delta_x = INVALID_VALUE;
  double combined_delta_y = INVALID_VALUE;

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

  TBranch* br_dx = ensureAndResetBranch("combined_delta_x", &combined_delta_x);
  TBranch* br_dy = ensureAndResetBranch("combined_delta_y", &combined_delta_y);

  const Long64_t nEntries = tree->GetEntries();
  Long64_t nWritten = 0;

  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);

    // Default
    combined_delta_x = INVALID_VALUE;
    combined_delta_y = INVALID_VALUE;

    if (is_pixel_hit) {
      // Pixel hit: use nearest pixel center
      if (IsFiniteD(x_px) && IsFiniteD(x_hit)) combined_delta_x = std::abs(x_px - x_hit);
      if (IsFiniteD(y_px) && IsFiniteD(y_hit)) combined_delta_y = std::abs(y_px - y_hit);
    } else {
      // Non-pixel hit: prefer 2D reco, else 3D reco
      double xr = IsFiniteD(x_rec_2d) ? x_rec_2d : x_rec_3d;
      double yr = IsFiniteD(y_rec_2d) ? y_rec_2d : y_rec_3d;
      if (IsFiniteD(xr) && IsFiniteD(x_hit)) combined_delta_x = std::abs(xr - x_hit);
      if (IsFiniteD(yr) && IsFiniteD(y_hit)) combined_delta_y = std::abs(yr - y_hit);
    }

    br_dx->Fill();
    br_dy->Fill();
    nWritten++;
  }

  file->cd();
  tree->Write("", TObject::kOverwrite);
  file->Flush();
  file->Close();
  delete file;

  ::Info("makeCombinedDeltas", "Wrote %lld entries to combined_delta_{x,y}", nWritten);
  return 0;
}


