/// \file RootHelpers.hh
/// \brief ROOT I/O helper utilities for thread-safe file operations.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_ROOT_HELPERS_HH
#define ECS_ROOT_HELPERS_HH

// Check if we're being compiled with Geant4 or as a standalone ROOT macro
// ROOT macros define __CLING__ when interpreted
#if defined(__CLING__) || defined(__ROOTCLING__)
#define ECS_HAS_GEANT4 0
#include <string>
#else
#define ECS_HAS_GEANT4 1
#include "globals.hh"
#endif

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
    const char* name{nullptr};
    double* value{nullptr};
    bool enabled{false};
    TBranch** handle{nullptr};
    const char* leaflist = nullptr;
};

inline TBranch* EnsureAndResetBranch(TTree* tree, const BranchInfo& info) {
    if (!tree || !info.name || !info.value || !info.handle) {
        return nullptr;
    }

    TBranch* branch = tree->GetBranch(info.name);
    if (!branch) {
        // Branch doesn't exist - create it
        branch = info.leaflist ? tree->Branch(info.name, info.value, info.leaflist)
                               : tree->Branch(info.name, info.value);
    } else {
        // Branch exists - set address and clear for overwrite
        tree->SetBranchAddress(info.name, info.value);
        branch = tree->GetBranch(info.name);
        if (branch) {
            branch->Reset();        // Clear previous entries
            branch->DropBaskets();  // Drop old baskets to avoid mixing
        }
    }
    tree->SetBranchStatus(info.name, true);
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
// Geant4-specific utilities (only available when compiled with Geant4)
// ============================================================================

#if ECS_HAS_GEANT4

// ============================================================================
// File Writer Helper
// ============================================================================

/// \brief Thread-safe ROOT file writer.
class RootFileWriter {
public:
    RootFileWriter();
    ~RootFileWriter();

    void Attach(TFile* file, TTree* tree, bool ownsObjects);
    [[nodiscard]] TFile* File() const;
    [[nodiscard]] TTree* Tree() const;

    bool SafeWrite(bool isMultithreaded, bool isWorker);
    bool Validate(const G4String& filename, bool* hasEntries);
    void Cleanup();
    void WriteMetadataSingleThread(G4double pixelSize, G4double pixelSpacing,
                                   G4double gridOffset, G4double detSize,
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

#endif // ECS_HAS_GEANT4

} // namespace ECS

// Alias for ROOT macro compatibility (used by FitGaussian2D.C)
namespace rootutils = ECS::RootUtils;

#endif // ECS_ROOT_HELPERS_HH
