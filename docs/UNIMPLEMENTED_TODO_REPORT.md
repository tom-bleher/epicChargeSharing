# Unimplemented Features - Prioritized TODO Report

**Generated:** December 2025
**Based on:** All reports in `/docs/` directory

This report consolidates all recommendations from the documentation reports that have **not yet been implemented**, organized by priority based on impact to the program.

---

## Implementation Status Summary

| Report | Recommendations | Implemented | Pending |
|--------|-----------------|-------------|---------|
| PERFORMANCE_OPTIMIZATION_REPORT.md | 15 | 2 | 13 |
| METADATA_ANALYSIS_REPORT.md | 7 | 0 | 7 |
| FITTING_OPTIMIZATION_REPORT.md | 8 | 2 | 6 |
| ROOT_UPGRADES_REPORT.md | 5 | 0 | 5 |
| DEVELOPMENT_REPORT.md | 18 | 4 | 14 |
| **Total** | **53** | **8** | **45** |

### Already Implemented

- Config.hh refactoring with `ECS::Config` namespace structures
- History file created
- Thread-local `FitWorkBuffers` in fitting macros
- CMake Doxygen support (`make docs`)
- CMake testing infrastructure (`BUILD_TESTING` option)
- Doxygen-style comments in Config.hh
- EC_FAST_MATH option includes `-fno-math-errno`
- Some thread-local distance error vectors
- **SteppingAction pointer comparison** (implemented Dec 2025) - uses cached `G4LogicalVolume*` pointers instead of string comparison

---

## Priority 0 (Critical) - Implement Immediately

### ~~1. SteppingAction Pointer Comparison~~ ✅ IMPLEMENTED
**Status:** Completed December 2025
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 2.3)
**Impact:** 10-50x speedup for volume identification

Now uses cached `G4LogicalVolume*` pointers via `G4LogicalVolumeStore::GetVolume()` with lazy initialization, following the official Geant4 B1 example pattern.

---

### 2. Enable ROOT ZSTD Compression
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 3.2)
**Impact:** 50-70% smaller files, faster I/O
**Effort:** 2 lines
**File:** `src/RunAction.cc:318`

**Current Issue:**
```cpp
rootFile->SetCompressionLevel(0);  // NO COMPRESSION!
```

**Fix:**
```cpp
rootFile->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
rootFile->SetCompressionLevel(4);
```

---

### 3. Add Critical Metadata Fields
**Source:** METADATA_ANALYSIS_REPORT.md (Section 4.1)
**Impact:** Reproducibility, debugging, analysis
**Effort:** Low
**File:** `src/RootIO.cc`

**Missing Fields:**
| Field | Importance |
|-------|------------|
| `RunTimestamp` | High - ISO 8601 datetime |
| `TotalEvents` | High - Actual event count |
| `RandomSeed` | High - Critical for reproducibility |
| `SimulationVersion` | High - Git commit hash |
| `Geant4Version` | High - `G4VERSION_NUMBER` |
| `ROOTVersion` | Medium - `ROOT_VERSION_CODE` |
| `PrimaryParticle` | Medium - Type, energy config |

---

### 4. Enable Implicit Multithreading in Fitting Macros
**Source:** ROOT_UPGRADES_REPORT.md (Section 4)
**Impact:** 50-200% faster fitting
**Effort:** Very Low (1 line per file)
**Files:** `src/FitGaussian1D.C`, `src/FitGaussian2D.C`

**Fix:** Add at the beginning of each macro:
```cpp
ROOT::EnableImplicitMT();
```

---

### 5. Thread-Local TF1 Reuse
**Source:** FITTING_OPTIMIZATION_REPORT.md (Section 3.4)
**Impact:** 10-20% reduction in per-fit overhead
**Effort:** Low
**File:** `src/FitGaussian1D.C:474`

**Current Issue:**
```cpp
// SLOW: Creates new TF1 every fit
TF1 fLoc("fGauss1D_helper", GaussPlusB, -1e9, 1e9, 4);
```

**Fix:** Use `thread_local` to reuse TF1 across fits within same thread.

---

## Priority 1 (High) - Implement This Week

### 6. TTree AutoFlush Tuning
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 3.3)
**Impact:** Better read performance
**Effort:** Low
**File:** `src/RunAction.cc:323-324`

**Current:**
```cpp
tree->SetAutoSave(0);   // Disabled
tree->SetAutoFlush(0);  // Disabled
```

**Fix:**
```cpp
tree->SetAutoFlush(50000);  // Flush every 50k entries
```

---

### 7. Add Unit Testing Framework
**Source:** DEVELOPMENT_REPORT.md (Section 4.6)
**Impact:** Reliability, regression prevention
**Effort:** High
**Location:** Create `tests/` directory

**Required Tests:**
- `TestChargeSharingCalculator.cc` - Fraction sums, edge cases
- `TestGrid2D.cc` - Grid indexing, bounds
- `TestNeighborhoodBuffer.cc` - Buffer management
- `TestSimulationRun.cc` - Integration test
- `TestRootOutput.cc` - Output validation

---

### 8. Create Validation Framework
**Source:** DEVELOPMENT_REPORT.md (Section 5.3)
**Impact:** Physics verification
**Effort:** High

**Required Validations:**
- Total charge conservation across neighborhood
- Center hit symmetry
- Noise statistics validation
- Resolution scaling with pitch

---

### 9. Profile-Guided Optimization (PGO)
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 1.2)
**Impact:** 10-20% overall speedup
**Effort:** Medium
**File:** `CMakeLists.txt`

Add `EC_PGO_GENERATE` and `EC_PGO_USE` options.

---

### 10. Use TParameter for Typed Metadata
**Source:** METADATA_ANALYSIS_REPORT.md (Section 4.2)
**Impact:** Type preservation, no string parsing needed
**Effort:** Medium
**File:** `src/RootIO.cc`

**Current:** All values stored as strings via `TNamed`

**Fix:** Use `TParameter<T>` for numeric values.

---

### 11. Closed-Form Gaussian Seed
**Source:** FITTING_OPTIMIZATION_REPORT.md (Section 4.3)
**Impact:** 30-50% fewer iterations
**Effort:** Medium
**Files:** `src/FitGaussian1D.C`, `src/FitGaussian2D.C`

Implement Caruana's algorithm for linearized Gaussian estimation as a pre-fit seed.

---

### 12. Precompute Distance Errors
**Source:** FITTING_OPTIMIZATION_REPORT.md (Section 7.3)
**Impact:** Low-Medium, avoids redundant `std::pow` calls
**Effort:** Low

Compute all distance-based sigmas upfront before the fitting loop.

---

## Priority 2 (Medium) - Implement This Month

### 13. Extend ECS Namespace to All Code
**Source:** DEVELOPMENT_REPORT.md (Section 4.1)
**Impact:** Standards compliance, organization
**Effort:** Medium

**Status:** `ECS::Config` namespace exists but main classes still global.

Move all classes into proper namespace hierarchy:
- `ECS::DetectorConstruction`
- `ECS::Action::Run`, `ECS::Action::Event`, etc.
- `ECS::Charge::SharingCalculator`

---

### 14. Decompose RunAction.cc
**Source:** DEVELOPMENT_REPORT.md (Section 4.3)
**Impact:** Maintainability
**Effort:** Medium
**File:** `src/RunAction.cc` (1,021 lines)

Split into:
- `RunAction.cc` - Run lifecycle (~300 lines)
- `BranchConfigurator.cc` - Branch setup logic
- `WorkerSynchronizer.cc` - Thread coordination
- `PostProcessingRunner.cc` - Fit macro invocation

---

### 15. Region-Specific Production Cuts
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 2.2)
**Impact:** 30-50% simulation time reduction
**Effort:** Medium
**File:** `src/PhysicsList.cc`

Fine cuts (10 µm) in silicon region, coarse cuts (1 mm) elsewhere.

---

### 16. Add Runtime Configuration via G4Messenger
**Source:** DEVELOPMENT_REPORT.md (Section 4.4)
**Impact:** Flexibility, no recompilation needed
**Effort:** Medium

Add `G4GenericMessenger` support for:
- Pixel size/pitch
- Neighborhood radius
- Reconstruction model selection
- Noise parameters

---

### 17. Attach Metadata to TTree UserInfo
**Source:** METADATA_ANALYSIS_REPORT.md (Section 4.2B)
**Impact:** Metadata travels with TTree
**Effort:** Medium

Standard HEP practice used by ATLAS, CMS, LHCb.

---

### 18. Add [[likely]]/[[unlikely]] Hints
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 4.3)
**Impact:** Better branch prediction
**Effort:** Low
**Files:** Hot paths in `SteppingAction.cc`, `ChargeSharingCalculator.cc`

---

### 19. Multi-File Parallel Processing
**Source:** FITTING_OPTIMIZATION_REPORT.md (Section 8.2)
**Impact:** Faster batch processing
**Effort:** Low
**File:** `farm/run_gaussian_fits.py`

Use `ProcessPoolExecutor` to parallelize across files.

---

## Priority 3 (Low) - Future Releases

### 20. Migrate to RNTuple
**Source:** ROOT_UPGRADES_REPORT.md (Section 1)
**Impact:** 20-35% smaller files, 2x faster I/O
**Effort:** High
**Prerequisites:** ROOT 6.36+

---

### 21. Migrate Analysis to RDataFrame
**Source:** ROOT_UPGRADES_REPORT.md (Section 2)
**Impact:** Cleaner code, automatic parallelization
**Effort:** Medium

---

### 22. Implement Pluggable Charge Sharing Models
**Source:** DEVELOPMENT_REPORT.md (Section 5.1)
**Impact:** Extensibility
**Effort:** High

Create abstract `ModelBase` class with:
- `LogModel`
- `LinearModel`
- `DPCModel`
- `GaussianDiffusionModel` (new)
- `TemplateModel` (new)

---

### 23. Enhanced Noise Modeling
**Source:** DEVELOPMENT_REPORT.md (Section 5.2)
**Impact:** Physics accuracy
**Effort:** Medium

- Position-dependent noise (edge effects)
- Crosstalk modeling
- Temperature-dependent noise

---

### 24. Support Alternative Pixel Geometries
**Source:** DEVELOPMENT_REPORT.md (Section 5.5)
**Impact:** Flexibility
**Effort:** High

- Strip sensors (1D readout)
- Hexagonal pixels
- Cross-shaped electrodes

---

### 25. SIMD Vectorization
**Source:** PERFORMANCE_OPTIMIZATION_REPORT.md (Section 6)
**Impact:** 2-10x for matrix operations
**Effort:** High

Consider Eigen for `Grid2D` operations.

---

### 26. Doxygen Documentation Coverage
**Source:** DEVELOPMENT_REPORT.md (Section 4.5)
**Impact:** Documentation
**Effort:** Medium

Add `///` comments to all public interfaces beyond Config.hh.

---

### 27. JSON Sidecar for Configuration
**Source:** METADATA_ANALYSIS_REPORT.md (Section 4.3B)
**Impact:** Complex nested config support
**Effort:** Medium

---

### 28. FastPow for Common Exponents
**Source:** FITTING_OPTIMIZATION_REPORT.md (Section 7.2)
**Impact:** Low, avoids expensive `std::pow`
**Effort:** Low

Optimize for exp = 1.0, 1.5, 2.0.

---

### 29. Colorblind-Friendly Palettes
**Source:** ROOT_UPGRADES_REPORT.md (Section 6)
**Impact:** Publication quality
**Effort:** Low

Adopt new ROOT 6.34 palettes for plotting macros.

---

## Quick Reference - Top 10 by Impact/Effort Ratio

| Rank | Item | Expected Impact | Lines to Change | Status |
|------|------|-----------------|-----------------|--------|
| 1 | ROOT ZSTD Compression | 50-70% smaller files | 2 | Pending |
| 2 | ~~SteppingAction Pointer Comparison~~ | 10-50x for volume ID | ~20 | ✅ Done |
| 3 | Enable IMT in Fitting | 50-200% faster fitting | 1 per file | Pending |
| 4 | Thread-Local TF1 Reuse | 10-20% per-fit | ~15 | Pending |
| 5 | TTree AutoFlush | Better read perf | 1 | Pending |
| 6 | Critical Metadata Fields | Reproducibility | ~30 | Pending |
| 7 | [[likely]]/[[unlikely]] | Branch prediction | ~10 | Partial |
| 8 | Precompute Distance Errors | Avoid redundant pow | ~10 | Pending |
| 9 | Multi-File Parallel Fits | Faster batch | ~10 | Pending |
| 10 | Tolerance Tuning (1e-3) | 2-3x faster per fit | 1 | Pending |

---

## File Reference Index

| File | Pending Changes |
|------|-----------------|
| `src/SteppingAction.cc` | #1 Pointer comparison |
| `src/RunAction.cc` | #2 ZSTD, #6 AutoFlush, #14 Decompose |
| `src/RootIO.cc` | #3 Metadata, #10 TParameter, #17 UserInfo |
| `src/FitGaussian1D.C` | #4 IMT, #5 TF1 reuse, #11 Seed, #12 Errors |
| `src/FitGaussian2D.C` | #4 IMT, #5 TF1 reuse, #11 Seed, #12 Errors |
| `src/PhysicsList.cc` | #15 Production cuts |
| `src/ChargeSharingCalculator.cc` | #18 [[likely]], #22 Models |
| `CMakeLists.txt` | #9 PGO |
| `farm/run_gaussian_fits.py` | #19 Parallel |
| `tests/` | #7 Create directory and tests |

---

## Report Sources

1. **PERFORMANCE_OPTIMIZATION_REPORT.md** - Build/runtime optimizations
2. **METADATA_ANALYSIS_REPORT.md** - ROOT metadata improvements
3. **FITTING_OPTIMIZATION_REPORT.md** - Gaussian fitting performance
4. **ROOT_UPGRADES_REPORT.md** - ROOT 6.34+ features
5. **DEVELOPMENT_REPORT.md** - Architecture and standards

---

*This report should be updated as items are implemented.*
