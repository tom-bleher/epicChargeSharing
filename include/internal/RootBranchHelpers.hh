#ifndef EPICCHARGESHARING_INTERNAL_ROOTBRANCHHELPERS_HH
#define EPICCHARGESHARING_INTERNAL_ROOTBRANCHHELPERS_HH

#include <vector>

#include <TBranch.h>
#include <TTree.h>

namespace rootutils {

struct BranchInfo {
  const char* name;
  double* value;
  bool enabled;
  TBranch** handle;
  const char* leaflist = nullptr;
};

inline TBranch* EnsureAndResetBranch(TTree* tree, const BranchInfo& info) {
  if (!tree || !info.name || !info.value || !info.handle) {
    return nullptr;
  }

  TBranch* branch = tree->GetBranch(info.name);
  if (!branch) {
    if (info.leaflist) {
      branch = tree->Branch(info.name, info.value, info.leaflist);
    } else {
      branch = tree->Branch(info.name, info.value);
    }
  } else {
    tree->SetBranchAddress(info.name, info.value);
    branch = tree->GetBranch(info.name);
    if (branch) {
      branch->Reset();
      branch->DropBaskets();
    }
  }
  tree->SetBranchStatus(info.name, 1);
  return branch;
}

inline void RegisterBranches(TTree* tree, std::vector<BranchInfo>& branches) {
  if (!tree) {
    return;
  }

  for (auto& info : branches) {
    if (!info.enabled) {
      continue;
    }
    if (!info.handle) {
      continue;
    }
    *info.handle = EnsureAndResetBranch(tree, info);
  }
}

inline void FillBranches(const std::vector<BranchInfo>& branches) {
  for (const auto& info : branches) {
    if (!info.enabled) {
      continue;
    }
    if (info.handle && *info.handle) {
      (*info.handle)->Fill();
    }
  }
}

}  // namespace rootutils

#endif  // EPICCHARGESHARING_INTERNAL_ROOTBRANCHHELPERS_HH


