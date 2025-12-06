/// \file RootHelpers.hh
/// \brief ROOT I/O helper utilities for thread-safe file operations.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_ROOT_HELPERS_HH
#define ECS_ROOT_HELPERS_HH

#include "globals.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include <TBranch.h>
#include <TTree.h>

class TFile;

namespace ECS {

// ============================================================================
// Branch Utilities
// ============================================================================

namespace RootUtils {

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
        branch = info.leaflist ? tree->Branch(info.name, info.value, info.leaflist)
                               : tree->Branch(info.name, info.value);
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
    if (!tree) return;
    for (auto& info : branches) {
        if (info.enabled && info.handle) {
            *info.handle = EnsureAndResetBranch(tree, info);
        }
    }
}

inline void FillBranches(const std::vector<BranchInfo>& branches) {
    for (const auto& info : branches) {
        if (info.enabled && info.handle && *info.handle) {
            (*info.handle)->Fill();
        }
    }
}

} // namespace RootUtils

// ============================================================================
// File Writer Helper
// ============================================================================

/// \brief Thread-safe ROOT file writer.
class RootFileWriter {
public:
    RootFileWriter();
    ~RootFileWriter();

    void Attach(TFile* file, TTree* tree, bool ownsObjects);
    TFile* File() const;
    TTree* Tree() const;

    bool SafeWrite(bool isMultithreaded, bool isWorker);
    bool Validate(const G4String& filename, bool* hasEntries);
    void Cleanup();
    void WriteMetadataSingleThread(G4double pixelSize, G4double pixelSpacing,
                                   G4double pixelCornerOffset, G4double detSize,
                                   G4int numBlocksPerSide, G4int neighborhoodRadius);

private:
    mutable std::mutex fMutex;
    TFile* fRootFile{nullptr};
    TTree* fTree{nullptr};
    bool fOwnsObjects{false};
};

// ============================================================================
// Worker Synchronization
// ============================================================================

/// \brief Synchronizes worker threads at end of run.
class WorkerSync {
public:
    static WorkerSync& Instance();

    void Reset(G4int totalWorkers);
    void SignalCompletion();
    void WaitForAll();

private:
    WorkerSync() = default;

    std::mutex fMutex;
    std::condition_variable fCv;
    std::atomic<int> fCompleted{0};
    int fTotal{0};
    bool fAllDone{false};
};

/// \brief Initialize ROOT for multithreaded use.
void InitializeROOTThreading();

/// \brief Generate worker-specific output filename.
G4String WorkerFileName(G4int threadId);

/// \brief Global mutex for ROOT I/O operations.
std::mutex& RootIOMutex();

} // namespace ECS

// Alias for ROOT macro compatibility (used by FitGaussian2D.C)
namespace rootutils = ECS::RootUtils;

#endif // ECS_ROOT_HELPERS_HH
