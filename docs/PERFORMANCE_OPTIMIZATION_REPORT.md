# Performance Optimization Report for epicChargeSharing

## Executive Summary

This report presents comprehensive optimization strategies for the epicChargeSharing Geant4/ROOT simulation. The recommendations are organized by impact level and implementation complexity, covering build configuration, Geant4-specific optimizations, ROOT I/O tuning, C++ code-level improvements, and advanced techniques for scaling.

**Key Findings:**
- Your codebase is already well-optimized with LTO, C++20, and fast EM physics
- The highest-impact quick win is fixing string comparisons in `SteppingAction.cc`
- ROOT I/O currently uses no compression (`SetCompressionLevel(0)`) - enabling ZSTD can reduce file sizes by 50-70%
- Several compute-intensive loops can benefit from SIMD vectorization and cache optimization

---

## Table of Contents

1. [Build System Optimizations](#1-build-system-optimizations)
2. [Geant4-Specific Optimizations](#2-geant4-specific-optimizations)
3. [ROOT I/O Optimizations](#3-root-io-optimizations)
4. [Code-Level Optimizations](#4-code-level-optimizations)
5. [Memory & Cache Optimizations](#5-memory--cache-optimizations)
6. [SIMD & Vectorization](#6-simd--vectorization)
7. [Distributed & Cluster Computing](#7-distributed--cluster-computing)
8. [Profiling & Benchmarking](#8-profiling--benchmarking)
9. [Geant4 11.3 New Features](#9-geant4-113-new-features)
10. [Implementation Priority Matrix](#10-implementation-priority-matrix)

---

## 1. Build System Optimizations

### 1.1 Current Strengths ✓
Your `CMakeLists.txt` already includes excellent optimizations:
- LTO (Link-Time Optimization) enabled for Release builds
- Release mode defaults with `-O3`
- `-funroll-loops` for release builds
- Optional `-ffast-math` and `-march=native` via `EC_FAST_MATH`
- Profiling support via `EC_PROFILE`

### 1.2 Profile-Guided Optimization (PGO)

PGO provides 10-20% speedup by optimizing based on actual runtime behavior:

```cmake
# Add to CMakeLists.txt
option(EC_PGO_GENERATE "Generate PGO profile data" OFF)
option(EC_PGO_USE "Use PGO profile data for optimization" OFF)

if(EC_PGO_GENERATE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND _project_compile_options -fprofile-generate=${CMAKE_BINARY_DIR}/pgo)
    list(APPEND _project_link_options -fprofile-generate=${CMAKE_BINARY_DIR}/pgo)
    message(STATUS "PGO: Profile generation enabled -> ${CMAKE_BINARY_DIR}/pgo")
elseif(EC_PGO_USE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND _project_compile_options
        -fprofile-use=${CMAKE_BINARY_DIR}/pgo
        -fprofile-correction)
    list(APPEND _project_link_options -fprofile-use=${CMAKE_BINARY_DIR}/pgo)
    message(STATUS "PGO: Using profile data for optimization")
endif()
```

**Usage workflow:**
```bash
# Step 1: Build with profiling instrumentation
cmake -DEC_PGO_GENERATE=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./epicChargeSharing run.mac  # Run representative workload (1000+ events)

# Step 2: Rebuild optimized with profile data
cmake -DEC_PGO_GENERATE=OFF -DEC_PGO_USE=ON ..
make -j$(nproc)
```

PGO is particularly effective because it optimizes branch prediction in your hot loops in `ComputeChargeFractions` and `ComputeFullGridFractions`.

### 1.3 Additional Compiler Flags

```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Better branch prediction layout
    list(APPEND _project_release_compile_options -fno-plt)

    # Enable OpenMP SIMD pragmas without full threading
    list(APPEND _project_release_compile_options -fopenmp-simd)

    # Disable errno checking for math functions (4x+ speedup for sqrt/log)
    list(APPEND _project_release_compile_options -fno-math-errno)

    # Allow reordering of floating-point operations
    list(APPEND _project_release_compile_options -ffp-contract=fast)
endif()
```

### 1.4 Architecture-Specific Tuning

```bash
# Identify your CPU architecture
gcc -march=native -Q --help=target | grep march

# Common architecture options:
# -march=znver4      # AMD Zen 4 (Ryzen 7000/EPYC Genoa)
# -march=znver3      # AMD Zen 3 (Ryzen 5000)
# -march=alderlake   # Intel 12th/13th Gen
# -march=skylake     # Intel 6th-10th Gen
# -march=neoverse-n1 # ARM Graviton2/Ampere Altra
```

**References:**
- [CMake Build Optimization](https://stackoverflow.com/questions/41361631/optimize-in-cmake-by-default)
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

---

## 2. Geant4-Specific Optimizations

### 2.1 Physics List Selection ✓

Your current configuration uses `G4EmStandardPhysics_option1` (EMV), which is the **fastest EM physics option**:

| Constructor | Speed | Accuracy | Best Use Case |
|-------------|-------|----------|---------------|
| `_option1` (EMV) ✓ | Fastest | Good | HEP, fast simulations |
| `_option2` (EMX) | Fast | Good | General purpose |
| `_option3` (EMY) | Medium | High | Medical, low-energy |
| `_option4` | Slow | Highest | Precision benchmarks |

**Your setup is optimal for speed.**

### 2.2 Production Cut Optimization

Your current 50 µm cut is quite fine. Consider region-specific cuts for better performance:

```cpp
// PhysicsList.cc - Region-specific cuts for optimal performance
void PhysicsList::SetCuts()
{
    // Fine cuts in sensitive silicon region
    G4Region* siliconRegion = G4RegionStore::GetInstance()->GetRegion("SiliconRegion");
    if (siliconRegion) {
        G4ProductionCuts* cuts = new G4ProductionCuts();
        cuts->SetProductionCut(10.0*CLHEP::micrometer, "gamma");
        cuts->SetProductionCut(10.0*CLHEP::micrometer, "e-");
        cuts->SetProductionCut(10.0*CLHEP::micrometer, "e+");
        siliconRegion->SetProductionCuts(cuts);
    }

    // Coarse cuts elsewhere for speed
    SetDefaultCutValue(1.0*CLHEP::millimeter);
}
```

**Impact:** Production cuts have roughly exponential impact on simulation time. Doubling the cut can reduce simulation time by 30-50% in high-energy scenarios.

### 2.3 SteppingAction Optimization ⚠️ **HIGH PRIORITY**

**Current Issue in `SteppingAction.cc:41`:**
```cpp
// SLOW: String comparison on EVERY step!
const G4String enteredName = postVol->GetLogicalVolume()->GetName();
if (enteredName == "logicBlock" || enteredName == "logicCube") {
```

String comparison costs ~100+ CPU cycles per step. With millions of steps per run, this is a significant overhead.

**Optimized implementation using pointer comparison:**

```cpp
// SteppingAction.hh - Add cached volume pointers
class SteppingAction : public G4UserSteppingAction {
private:
    const G4LogicalVolume* fLogicBlock = nullptr;
    const G4LogicalVolume* fLogicCube = nullptr;
    bool fVolumesCached = false;

    void CacheVolumes();
};

// SteppingAction.cc
#include "G4LogicalVolumeStore.hh"

void SteppingAction::CacheVolumes() {
    if (fVolumesCached) return;

    auto* lvStore = G4LogicalVolumeStore::GetInstance();
    fLogicBlock = lvStore->GetVolume("logicBlock", false);
    fLogicCube = lvStore->GetVolume("logicCube", false);
    fVolumesCached = true;
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step) {
    if (fFirstContactVolume != "NONE") return;

    CacheVolumes();  // One-time initialization

    G4StepPoint* postPoint = step->GetPostStepPoint();
    if (!postPoint || postPoint->GetStepStatus() != fGeomBoundary) return;

    G4VPhysicalVolume* postVol = postPoint->GetTouchableHandle()->GetVolume();
    if (!postVol) return;

    // FAST: Pointer comparison (~3 cycles vs ~100+ for string)
    const G4LogicalVolume* lv = postVol->GetLogicalVolume();
    if (lv == fLogicBlock || lv == fLogicCube) {
        fFirstContactVolume = lv->GetName();  // Only retrieve name when needed
        if (fEventAction) {
            fEventAction->RegisterFirstContact(postPoint->GetPosition());
        }
    }
}
```

**Expected speedup:** 10-50x for volume identification logic.

### 2.4 Use const References for G4ThreeVector

The Geant4 documentation recommends using const references for composite types:

```cpp
// SLOW: Copies G4ThreeVector
G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();

// FAST: Uses reference, avoids copy
const G4ThreeVector& pos = step->GetPreStepPoint()->GetPosition();
```

### 2.5 Scorer vs Sensitive Detector

Your current approach using `G4SDManager` and `G4THitsMap` is good. Key optimization:

```cpp
// EventAction.cc:338 - Cache the collection ID
void EventAction::CollectScorerData(const G4Event* event)
{
    // GOOD: You're caching edepID as static
    static G4int edepID = -1;
    if (edepID < 0) {
        if (auto* sdm = G4SDManager::GetSDMpointer()) {
            edepID = sdm->GetCollectionID("SiliconDetector/EnergyDeposit");
        }
    }
    // GetCollectionID() is expensive - caching is correct!
}
```

### 2.6 Multithreading Configuration

```cpp
// Optimal thread configuration in main()
#ifdef G4MULTITHREADED
    G4int nCores = G4Threading::G4GetNumberOfCores();
    // Physical cores only - hyperthreading provides no benefit for Geant4
    G4int nThreads = std::max(1, nCores - 1);  // Leave 1 for OS
    runManager->SetNumberOfThreads(nThreads);

    // Use MIXMAX for MT (guaranteed divergent sequences)
    CLHEP::HepRandom::setTheEngine(new CLHEP::MixMaxRng);
#endif
```

**References:**
- [Geant4 Performance Tips - CERN TWiki](https://twiki.cern.ch/twiki/bin/view/Geant4/Geant4PerformanceTips)
- [Geant4 EM Physics Guide](https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsListGuide/html/electromagnetic/emphyslist.html)
- [Geant4 Multithreading](https://geant4.web.cern.ch/documentation/pipelines/master/bftd_html/ForToolkitDeveloper/OOAnalysisDesign/Multithreading/mt.html)

---

## 3. ROOT I/O Optimizations

### 3.1 Current Issue: No Compression ⚠️

In `RunAction.cc:318`:
```cpp
rootFile->SetCompressionLevel(0);  // NO COMPRESSION!
```

This means your ROOT files are uncompressed, which:
- Increases disk I/O time
- Wastes storage space
- Slows down file merging

### 3.2 Recommended Compression Settings

```cpp
// RunAction.cc - Replace SetCompressionLevel(0) with:
#include "TFile.h"
#include "Compression.h"

// ZSTD: Best balance of speed and compression
rootFile->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
rootFile->SetCompressionLevel(4);  // 1-9, 4 is good balance

// Alternative: LZ4 for maximum write speed (less compression)
// rootFile->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kLZ4);
// rootFile->SetCompressionLevel(4);
```

| Algorithm | Write Speed | Read Speed | Compression Ratio |
|-----------|-------------|------------|-------------------|
| None (current) | Fastest | Fastest | 1.0x |
| LZ4 | Very Fast | Very Fast | 2-3x |
| **ZSTD** ✓ | Fast | Fast | **3-5x** |
| ZLIB | Slow | Medium | 3-4x |

### 3.3 TTree Configuration

Your current settings in `RunAction.cc:323-324`:
```cpp
tree->SetAutoSave(0);   // Disabled
tree->SetAutoFlush(0);  // Disabled
```

**Recommended settings:**
```cpp
// Enable AutoFlush for better read performance
tree->SetAutoFlush(50000);  // Flush every 50k entries

// Or use byte-based flushing (default is 30MB)
tree->SetAutoFlush(-30000000);  // Negative = bytes

// OptimizeBaskets after filling some entries
// Call this after ~1000 events to auto-tune buffer sizes
tree->OptimizeBaskets(256000000, 1.0, "");  // 256MB max memory
```

### 3.4 Basket Size for Vector Branches

Your vector branches use 256KB baskets (good!):
```cpp
constexpr Int_t bufsize = 256000;  // 256KB - already optimized ✓
```

For very large grids (full detector fractions), consider:
```cpp
// For branches with >10KB per entry
tree->SetBasketSize("FiGrid", 512000);
tree->SetBasketSize("QfGrid", 512000);
```

### 3.5 Future: Migrate to RNTuple

ROOT's RNTuple (production-ready in ROOT 6.36, Q2 2025) offers:
- 2-5x faster read throughput
- 20-35% smaller files
- Better multicore scalability (>100 threads tested)

```cpp
// Future-proof preparation
#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,36,0)
    #include "ROOT/RNTuple.hxx"
    // Use RNTuple API
#endif
```

### 3.6 RDataFrame for Analysis

For reading and analyzing your ROOT files:

```cpp
// Enable implicit multithreading
ROOT::EnableImplicitMT();

// Use RDataFrame instead of TTree loops
ROOT::RDataFrame df("Hits", "epicChargeSharing.root");

auto h = df.Filter("Edep > 0")
           .Define("deltaR", "sqrt(ReconTrueDeltaX*ReconTrueDeltaX + ReconTrueDeltaY*ReconTrueDeltaY)")
           .Histo1D("deltaR");
```

**References:**
- [ROOT TTree Manual](https://root.cern/manual/trees/)
- [RNTuple Update 2024](https://root.cern/blog/rntuple-update/)
- [RDataFrame Performance](https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_06029/epjconf_chep2018_06029.html)

---

## 4. Code-Level Optimizations

### 4.1 ChargeSharingCalculator: Duplicate Weight Computation

In `ComputeChargeFractions()` (lines ~510-575), weights are computed twice:

```cpp
// First computation (line ~388-396)
const G4double safeDistance = std::max(distance, d0p.minSafeDistance);
const G4double logValue = std::log(safeDistance * d0p.invLength);
G4double weight = (logValue > 0.0 && std::isfinite(logValue)) ? (alpha / logValue) : 0.0;

// ... later, RECOMPUTED (line ~519-528)
const G4double safeDistance = std::max(distance, d0p.minSafeDistance);  // DUPLICATE
const G4double logValue = std::log(safeDistance * d0p.invLength);       // DUPLICATE
```

**Solution:** Store original weights in the scratch buffer:

```cpp
// Modify WeightScratch to store both original and modified weights
struct WeightPair {
    G4double original;
    G4double modified;
};
std::vector<WeightPair> fWeightScratch;

// In ComputeChargeFractions:
fWeightScratch.push_back({weight, weight});  // Store original

// Later, for row/col sums, use fWeightScratch[i].original
```

### 4.2 Precompute Loop Invariants

In `ComputeFullGridFractions()`, move invariants outside loops:

```cpp
// Current: computed inside nested loops
const G4double alpha = CalcPixelAlphaSubtended(distance, pixelSize, pixelSize);

// Optimization: precompute constants
const G4double halfDiag = (pixelSize * 0.5) * 1.41421356237;  // sqrt(2)
const G4double invD0 = d0p.invLength;
const G4double betaInvMicron = chargeModel.beta * chargeModel.invMicron;
```

### 4.3 Use C++20 [[likely]]/[[unlikely]] Attributes

For predictable branches in hot paths:

```cpp
void SteppingAction::TrackVolumeInteractions(const G4Step* step) {
    if (fFirstContactVolume != "NONE") [[likely]] {
        return;  // Most steps don't change contact
    }

    // ... rest of function
}

// In ComputeChargeFractions
if (cell.gridIndex < 0) [[unlikely]] {
    continue;
}
```

### 4.4 Avoid std::vector Reallocations

Your code already reserves buffers (good!), but ensure consistency:

```cpp
// ChargeSharingCalculator.cc - ReserveBuffers()
void ChargeSharingCalculator::ReserveBuffers() {
    const std::size_t maxCells = static_cast<std::size_t>(fGridDim) * fGridDim;

    // Pre-allocate to avoid reallocations
    if (fResult.cells.capacity() < maxCells) {
        fResult.cells.reserve(maxCells);
    }
    if (fWeightScratch.capacity() < maxCells) {
        fWeightScratch.reserve(maxCells);
    }

    // Also reserve chargeBlock (always max 4)
    fResult.chargeBlock.reserve(4);
}
```

**References:**
- [Agner Fog's Optimization Manual](https://www.agner.org/optimize/optimizing_cpp.pdf)
- [C++ likely/unlikely attributes](https://en.cppreference.com/w/cpp/language/attributes/likely)

---

## 5. Memory & Cache Optimizations

### 5.1 False Sharing Prevention

In multithreaded scenarios, ensure thread-local data doesn't share cache lines:

```cpp
// Potential false sharing in RunAction if accessed from multiple threads
struct alignas(64) ThreadLocalData {  // 64-byte alignment = cache line
    G4double fEdep;
    G4double fTrueX;
    G4double fTrueY;
    // ... pad to 64 bytes if needed
    char padding[64 - 3*sizeof(G4double)];
};
```

For your current single-threaded per-worker design, this isn't critical, but good to know for future scaling.

### 5.2 Cache-Friendly Access Patterns

Your `Grid2D` template uses row-major storage (correct for C++):

```cpp
// Grid2D::operator() - Row-major access
const auto idx = static_cast<std::size_t>(row) * nCols + col;
```

Ensure loops iterate row-first:
```cpp
// GOOD: Sequential memory access
for (G4int i = 0; i < rows; ++i) {
    for (G4int j = 0; j < cols; ++j) {
        grid(i, j) = ...;  // Accesses grid[i*cols + j]
    }
}

// BAD: Strided access (cache misses)
for (G4int j = 0; j < cols; ++j) {
    for (G4int i = 0; i < rows; ++i) {
        grid(i, j) = ...;  // Jumps by 'cols' each iteration
    }
}
```

### 5.3 G4Allocator for Custom Hit Classes

If you create custom hit classes, use G4Allocator for memory pooling:

```cpp
class MyHit : public G4VHit {
public:
    inline void* operator new(size_t);
    inline void operator delete(void*);

private:
    static G4ThreadLocal G4Allocator<MyHit>* fAllocator;
};

inline void* MyHit::operator new(size_t) {
    if (!fAllocator) fAllocator = new G4Allocator<MyHit>;
    return (void*)fAllocator->MallocSingle();
}

inline void MyHit::operator delete(void* hit) {
    fAllocator->FreeSingle((MyHit*)hit);
}
```

**References:**
- [False Sharing in C++](https://medium.com/@techhara/speed-up-c-false-sharing-44b56fffe02b)
- [G4Allocator Documentation](https://apc.u-paris.fr/~franco/g4doxy4.10/html/class_g4_allocator.html)

---

## 6. SIMD & Vectorization

### 6.1 Enable Auto-Vectorization

Add compiler hints for vectorizable loops:

```cpp
// In hot loops like ComputeFullGridFractions
#pragma omp simd
for (G4int i = 0; i < rows; ++i) {
    // Loop body should be simple for vectorization
}
```

Compile with `-fopenmp-simd` to enable SIMD pragmas without full OpenMP threading.

### 6.2 Why Standard Math is Slow

Without `-fno-math-errno`, `std::sqrt` and `std::log` must set `errno` on error, preventing vectorization:

```cpp
// Standard sqrt: ~12-14 cycles, non-vectorized
double d = std::sqrt(x);

// With -fno-math-errno: vectorized, ~3-4 cycles per element
// Add to CMakeLists.txt:
list(APPEND _project_release_compile_options -fno-math-errno)
```

### 6.3 Consider Eigen for Matrix Operations

For intensive grid computations, Eigen provides auto-vectorized operations:

```cpp
#include <Eigen/Dense>

// Instead of manual Grid2D operations
Eigen::MatrixXd weights(rows, cols);
Eigen::MatrixXd fractions = weights / weights.sum();
```

Eigen auto-vectorizes with SSE/AVX/NEON and provides 2-10x speedups for dense matrix operations.

**References:**
- [Why C++ Math is Slow](https://hackernoon.com/why-math-functions-in-c-are-so-slow-nxz3155)
- [Eigen Library](https://eigen.tuxfamily.org/)
- [SIMD for C++ Developers](http://const.me/articles/simd/simd.pdf)

---

## 7. Distributed & Cluster Computing

### 7.1 MPI for Multi-Node Scaling

For cluster computing, Geant4 supports hybrid MPI+MT:

```cpp
// Example using G4MPImanager
#include "G4MPImanager.hh"
#include "G4MPIsession.hh"

int main(int argc, char** argv) {
    G4MPImanager* g4MPI = new G4MPImanager(argc, argv);
    G4MPIsession* session = g4MPI->GetMPIsession();

    // Each MPI rank runs MT workers
    G4MTRunManager* runManager = new G4MTRunManager;
    runManager->SetNumberOfThreads(G4Threading::G4GetNumberOfCores());

    // ... setup and run

    delete g4MPI;
}
```

### 7.2 Simple Multi-Process Parallelization

Without MPI, use GNU Parallel for embarrassingly parallel runs:

```bash
#!/bin/bash
# Run 8 independent simulations with different seeds
parallel -j 8 './epicChargeSharing -s {} run.mac' ::: $(seq 1 8)

# Merge output files
hadd -f epicChargeSharing_merged.root epicChargeSharing_*.root
```

### 7.3 Job Arrays on HPC Clusters

For SLURM clusters:
```bash
#!/bin/bash
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

./epicChargeSharing -s $SLURM_ARRAY_TASK_ID run.mac
```

**References:**
- [Geant4 MPI Extension](https://arxiv.org/abs/1605.01792)
- [ParGeant4](http://www.ccs.neu.edu/home/gene/pargeant4-cern.html)

---

## 8. Profiling & Benchmarking

### 8.1 Linux perf (Recommended)

```bash
# Record with call graph
perf record -g --call-graph dwarf ./epicChargeSharing run.mac

# Interactive analysis
perf report --hierarchy

# Flame graph (install flamegraph.pl)
perf script | stackcollapse-perf.pl | flamegraph.pl > profile.svg
```

### 8.2 Valgrind Cachegrind

```bash
# Simulate cache behavior
valgrind --tool=cachegrind ./epicChargeSharing run.mac

# View results
cg_annotate --auto=yes cachegrind.out.*

# Key metrics:
# D1mr: L1 data cache read misses
# DLmr: Last-level cache read misses (most expensive)
```

### 8.3 Intel VTune (if available)

```bash
vtune -collect hotspots ./epicChargeSharing run.mac
vtune -report hotspots -r r000hs
```

### 8.4 Built-in Timing

Add timing instrumentation:

```cpp
#include <chrono>
#include "G4Timer.hh"

// In RunAction
G4Timer fRunTimer;

void RunAction::BeginOfRunAction(const G4Run*) {
    fRunTimer.Start();
}

void RunAction::EndOfRunAction(const G4Run* run) {
    fRunTimer.Stop();
    G4int nEvents = run->GetNumberOfEvent();
    G4double wallTime = fRunTimer.GetRealElapsed();
    G4cout << "Performance: " << nEvents / wallTime << " events/second" << G4endl;
    G4cout << "Time per event: " << 1000.0 * wallTime / nEvents << " ms" << G4endl;
}
```

**References:**
- [Linux perf wiki](https://perf.wiki.kernel.org/)
- [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

---

## 9. Geant4 11.3 New Features

Released December 2024, Geant4 11.3 includes several performance improvements:

### 9.1 Parallel Geometry Voxelization

Enable multi-threaded geometry initialization:

```cpp
#include "G4GeometryManager.hh"

// In DetectorConstruction::Construct()
G4GeometryManager::GetInstance()->RequestParallelOptimisation(true, true);
```

This accelerates voxel tree construction for complex geometries.

### 9.2 Dynamic Particle Processing

New classes (`G4DynamicParticleIonisation`, `G4DynamicParticleMSC`) perform computations on-the-fly without particle definition lookups.

### 9.3 Optimized UrbanMsc Model

`G4UrbanMscModel` has an optimized step limitation algorithm in EM Opt3 physics.

### 9.4 Faster Cross-Section Calculations

`G4ChargeExchangeXS` now computes at element level instead of isotope level - several times faster.

### 9.5 Improved Task Distribution

`G4TaskRunManager` now distributes events more evenly across tasks, improving load balancing.

**References:**
- [Geant4 11.3 Release Notes](https://geant4.web.cern.ch/download/release-notes/notes-v11.3.0.html)
- [2025 Planned Features](https://geant4.web.cern.ch/news/2025-03-11-planned-dev)

---

## 10. Implementation Priority Matrix

| Optimization | Impact | Effort | Priority |
|--------------|--------|--------|----------|
| **SteppingAction pointer comparison** | High | Low | **P0** |
| **Enable ROOT compression (ZSTD)** | High | Low | **P0** |
| Enable `-fno-math-errno` | Medium-High | Low | **P0** |
| Profile-Guided Optimization | High | Medium | **P1** |
| Cache weight calculations | Medium | Medium | **P1** |
| TTree AutoFlush tuning | Medium | Low | **P1** |
| Region-specific production cuts | Medium | Medium | **P2** |
| [[likely]]/[[unlikely]] hints | Low-Medium | Low | **P2** |
| SIMD vectorization | Medium | High | **P3** |
| Migrate to RNTuple | High | High | **Future** |

### Quick Wins (Implement Today)

1. **SteppingAction fix** - Replace string comparison with pointer comparison
2. **Enable ZSTD compression** - Change `SetCompressionLevel(0)` to `SetCompressionAlgorithm(kZSTD)` + level 4
3. **Add `-fno-math-errno`** - 4x+ speedup for math functions
4. **Build with `-DEC_FAST_MATH=ON`** - Already supported in your CMake

### Medium-Term (This Week)

5. **Run profiling** with `perf` to identify actual hotspots
6. **Implement PGO** for 10-20% overall speedup
7. **Enable AutoFlush** for better ROOT file structure
8. **Add timing instrumentation** to measure improvements

### Long-Term (Future Releases)

9. Consider RNTuple when ROOT 6.36 is stable
10. Evaluate Eigen for matrix operations if compute-bound
11. Implement MPI support for cluster scaling

---

## Summary

Your codebase is already well-architected with good separation of concerns and reasonable defaults. The highest-impact improvements are:

| Change | Expected Impact | Lines to Change |
|--------|-----------------|-----------------|
| SteppingAction pointer comparison | 10-50x for volume ID | ~20 lines |
| Enable ZSTD compression | 50-70% smaller files | 2 lines |
| `-fno-math-errno` flag | 4x faster sqrt/log | 1 CMake line |
| PGO build | 10-20% overall | CMake + workflow |

**Recommended first step:** Run `perf record -g ./epicChargeSharing run.mac` and analyze the flame graph to confirm where time is actually spent before making changes.

---

## References & Further Reading

### Geant4
- [Geant4 Performance Tips](https://twiki.cern.ch/twiki/bin/view/Geant4/Geant4PerformanceTips)
- [Geant4 11.3 Release Notes](https://geant4.web.cern.ch/download/release-notes/notes-v11.3.0.html)
- [Geant4 2025 Planned Features](https://geant4.web.cern.ch/news/2025-03-11-planned-dev)
- [Geant4 Multithreading](https://geant4.web.cern.ch/documentation/pipelines/master/bftd_html/ForToolkitDeveloper/OOAnalysisDesign/Multithreading/mt.html)

### ROOT
- [ROOT TTree Manual](https://root.cern/manual/trees/)
- [RNTuple Update 2024](https://root.cern/blog/rntuple-update/)
- [RDataFrame Performance](https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_06029/epjconf_chep2018_06029.html)
- [ROOT I/O Compression](https://indico.cern.ch/event/697389/contributions/3062030/attachments/1712625/3007684/ROOT_I_O_compression_algorithms1.pdf)

### C++ Optimization
- [Agner Fog's Optimization Manual](https://www.agner.org/optimize/optimizing_cpp.pdf)
- [Data-Oriented Design (Mike Acton)](https://www.youtube.com/watch?v=rX0ItVEVjHc)
- [Why C++ Math Functions Are Slow](https://hackernoon.com/why-math-functions-in-c-are-so-slow-nxz3155)
- [False Sharing Prevention](https://medium.com/@techhara/speed-up-c-false-sharing-44b56fffe02b)
- [Eigen Linear Algebra Library](https://eigen.tuxfamily.org/)

### Profiling
- [Linux perf wiki](https://perf.wiki.kernel.org/)
- [Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

---

*Report generated: December 2025*
*Based on: Geant4 11.3, ROOT 6.x, GCC/Clang with C++20*
*Codebase analyzed: epicChargeSharing (AC-LGAD charge sharing simulation)*
