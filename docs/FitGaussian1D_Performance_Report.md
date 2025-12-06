# Performance Analysis Report: FitGaussian1D.C

**Date**: December 2024
**Author**: Performance Analysis
**Subject**: Gaussian Fitting Performance Optimization for Charge Sharing Reconstruction

---

## Executive Summary

The `FitGaussian1D.C` fitting routine processes 50,000 events in approximately 6 minutes, achieving ~267 Gaussian fits per second. The observed 100% CPU usage across cores is **expected and intentional** — the code employs ROOT's implicit multithreading infrastructure. However, significant performance optimizations are achievable through algorithmic improvements and parameter tuning.

**Current Performance Metrics**:
- Events processed: 50,000
- Events fitted: 48,015 (96%)
- Wall-clock time: 6m 2.5s
- Total CPU time: 18m 14.9s
- Effective parallelization: ~3 cores

**Target Performance**: 30-60 seconds for 50,000 events (6-12x improvement)

---

## 1. Why Are CPU Cores at 100%?

### 1.1 Current Parallelization Architecture

The code explicitly enables parallel processing through ROOT's threading infrastructure:

```cpp
// src/FitGaussian1D.C:480 - Enable ROOT Implicit Multithreading
ROOT::EnableImplicitMT();

// src/FitGaussian1D.C:950 - Parallel execution over all entries
ROOT::TThreadExecutor exec;
exec.Foreach([&](int i){ /* fit logic */ }, indices);
```

**This behavior is correct and by design.** The timing output confirms effective parallel execution:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `real` | 6m 2.5s | Wall-clock time (what you experience) |
| `user` | 18m 14.9s | Total CPU time across all cores |
| `sys` | 0m 33.6s | Kernel/system overhead |
| Ratio (user/real) | ~3.0x | Effective core utilization |

### 1.2 Why Isn't Speedup Higher?

Several factors limit parallel scaling beyond ~3x:

1. **Minimizer overhead**: Each Gaussian fit involves iterative numerical optimization with convergence checks
2. **Thread-local object creation**: TF1 and Fitter objects have per-thread initialization costs
3. **Memory bandwidth contention**: Multiple threads compete for L3 cache and RAM access
4. **Sequential I/O phase**: Initial data loading is single-threaded (ROOT TTree limitation)
5. **Load imbalance**: Some events require more iterations than others

---

## 2. Current Bottlenecks Analysis

### 2.1 Primary Bottleneck: Numerical Minimization

Each event requires **2-4 Gaussian fits** (row, column, and optionally two diagonals), with each fit involving:

- **Fumili2 optimizer**: Up to 250 function evaluations per fit
- **Fallback to Migrad**: Additional iterations when Fumili2 fails
- **Parameter bound checking**: Enforced at each iteration
- **Gradient estimation**: Numerical differentiation overhead

**Estimated cost breakdown**:
```
Total fits: ~96,000 (48,015 events × 2 fits minimum)
Iterations per fit: 50-250
Function evaluations: 4.8M - 24M total
```

### 2.2 Overly Conservative Fit Configuration

The current minimizer settings prioritize precision over speed:

```cpp
// src/FitGaussian1D.C:482-486
ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2", "Fumili2");
ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);      // Very tight
ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(250); // High limit
ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
```

**Issues**:
- Tolerance of `1e-6` is unnecessarily precise for charge fraction reconstruction where physical uncertainties dominate
- 250 max function calls is excessive for 3-5 point polynomial-like fits
- Strategy 0 is appropriate but tolerance dominates convergence time

### 2.3 Sequential Data Loading Phase

```cpp
// src/FitGaussian1D.C:847-886 - Sequential I/O
for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);  // Single-threaded ROOT I/O
    v_x_hit[i] = x_hit;
    v_y_hit[i] = y_hit;
    // ... store remaining data ...
}
```

This phase processes all 50,000 entries sequentially before parallel fitting begins. While I/O is typically fast compared to fitting, it represents a serialization point.

### 2.4 Redundant Computations

Several quantities are recomputed per-event that could be cached:

```cpp
// Pixel position offsets computed repeatedly
const double x = x_px_loc + di * pixelSpacing;  // Line 1049
const double y = y_px_loc + dj * pixelSpacing;  // Line 1059
```

---

## 3. Optimization Plan

### Phase 1: Quick Wins (Estimated: 2-3x Speedup)

#### 3.1.1 Relax Minimizer Tolerance

**Current**: `1e-6` | **Proposed**: `1e-3`

```cpp
// Global default
ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-3);

// Per-fit configuration (src/FitGaussian1D.C:413)
fitter.Config().MinimizerOptions().SetTolerance(1e-3);  // was 1e-4
```

**Rationale**: Position reconstruction uncertainty from charge sharing is typically >1% of pixel pitch. A tolerance of `1e-3` provides sufficient precision while reducing iterations by 2-4x.

**Validation**: Compare `RMS(ReconRowX - TrueX)` before/after to confirm no degradation.

#### 3.1.2 Reduce Maximum Function Calls

**Current**: `250` | **Proposed**: `100`

```cpp
ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(100);
```

**Rationale**: Well-seeded 3-5 point Gaussian fits should converge in <50 iterations. The seeds from weighted centroid and max-element detection are typically excellent.

#### 3.1.3 Skip Fallback Minimizer for Low-Contrast Cases

Currently, failed Fumili2 fits always retry with Migrad:

```cpp
// src/FitGaussian1D.C:430-436
if (!ok) {
    fitter.Config().SetMinimizer("Minuit2", "Migrad");
    fitter.Config().MinimizerOptions().SetStrategy(1);
    ok = fitter.Fit(data);
}
```

**Proposed**: Add early exit for low-contrast events where centroid is sufficient:

```cpp
if (!ok) {
    // Check if signal is too weak for reliable Gaussian fit
    if (cfg.seedA < 0.01 * cfg.qmax) {
        return result;  // Centroid fallback will handle this
    }
    fitter.Config().SetMinimizer("Minuit2", "Migrad");
    // ... continue with fallback ...
}
```

### Phase 2: Algorithmic Improvements (Estimated: 3-5x Additional Speedup)

#### 3.2.1 Analytical Gaussian Fitting

For 3-5 point 1D data, an **analytical closed-form solution** exists using log-linearization. This eliminates iterative minimization entirely for most cases.

**Mathematical Basis**:

For `y = A·exp(-0.5·((x-μ)/σ)²) + B`, if B is estimated:

```
log(y - B) = log(A) - (x - μ)² / (2σ²)
           = log(A) - x²/(2σ²) + xμ/σ² - μ²/(2σ²)
           = c + bx + ax²
```

This is a quadratic polynomial solvable via 3×3 linear least squares:

```cpp
GaussFitResult AnalyticalGaussFit(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   double B_estimate) {
    GaussFitResult result;

    // Build normal equations for quadratic fit to log(y - B)
    // Solve [X'X]^(-1) X' log(y-B) for coefficients [c, b, a]
    // Extract: sigma² = -1/(2a), mu = -b/(2a), A = exp(c + mu²/(2σ²))

    // Fall back to numerical fit if:
    // - Any (y - B) <= 0 (log undefined)
    // - Condition number too high
    // - Extracted sigma² < 0

    return result;
}
```

**Implementation Notes**:
- Estimate B as minimum of y values (current approach)
- Use SVD or Cholesky decomposition for numerical stability
- Fall back to numerical Minuit2 fit when analytical fails

#### 3.2.2 Weighted Centroid as Primary Method

The code already implements weighted centroid as a fallback:

```cpp
// src/FitGaussian1D.C:292-310
inline std::pair<double, bool> WeightedCentroid(
    const std::vector<double>& positions,
    const std::vector<double>& charges,
    double baseline) {
    // ... computes sum(x_i * (q_i - baseline)) / sum(q_i - baseline)
}
```

**Proposed Enhancement**: Use centroid as the **primary method** for specific cases:

```cpp
// Use centroid instead of fit when:
// 1. Low contrast: (max - min) / max < 0.05
// 2. Few points: N < 4
// 3. Flat profile: all points within 10% of mean

if (shouldUseCentroid(q_row, qmaxNeighborhood)) {
    auto [centroid, ok] = WeightedCentroid(x_row, q_row, B0_row);
    if (ok) {
        out_x_rec[i] = centroid;
        // Skip Gaussian fit entirely
    }
}
```

#### 3.2.3 Precompute Pixel Position Offsets

**Current**: Computed per-event inside parallel loop

```cpp
const double x = x_px_loc + di * pixelSpacing;
```

**Proposed**: Precompute offset table once

```cpp
// Before parallel loop
const int maxR = neighborhoodRadiusMeta;
std::vector<double> pixelOffsets(2 * maxR + 1);
for (int di = -maxR; di <= maxR; ++di) {
    pixelOffsets[di + maxR] = di * pixelSpacing;
}

// Inside parallel loop
const double x = x_px_loc + pixelOffsets[di + R];
```

### Phase 3: Structural Optimizations (Estimated: 1.5-2x Additional Speedup)

#### 3.3.1 Batch I/O with RDataFrame

Replace sequential `GetEntry()` loop with ROOT's columnar data access:

```cpp
// Current approach (sequential)
for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    // ...
}

// Proposed: RDataFrame (parallelized I/O)
ROOT::RDataFrame df("Hits", filename);
auto results = df.Define("FitResult", [](double x, double y,
                                          const std::vector<double>& Q) {
    return performFit(x, y, Q);
}, {"TrueX", "TrueY", "Qf"});
```

**Benefits**:
- Automatic parallelization of I/O
- Better cache utilization through columnar access
- Eliminates manual data staging

#### 3.3.2 Reduce Memory Allocations in Hot Path

The `FitWorkBuffers` pattern is good but still involves clear/push_back cycles:

```cpp
// Current
x_row.clear();
for (...) {
    x_row.push_back(value);  // Potential reallocation
}

// Proposed: Pre-size and use indexing
x_row.resize(N);
size_t idx = 0;
for (...) {
    x_row[idx++] = value;  // No allocation
}
x_row.resize(idx);  // Trim to actual size
```

#### 3.3.3 SIMD Vectorization for Gaussian Evaluation

For analytical fitting or batch evaluation, use SIMD operations:

```cpp
#include <ROOT/RVec.hxx>

// Vectorized Gaussian evaluation
ROOT::RVec<double> GaussianBatch(const ROOT::RVec<double>& x,
                                  double A, double mu, double sigma, double B) {
    auto dx = (x - mu) / sigma;
    return A * ROOT::VecOps::exp(-0.5 * dx * dx) + B;
}
```

---

## 4. Implementation Priority Matrix

| Priority | Optimization | Implementation Effort | Expected Speedup | Risk |
|:--------:|--------------|:--------------------:|:----------------:|:----:|
| **1** | Relax tolerance (1e-6 → 1e-3) | 5 minutes | 1.5-2x | Low |
| **2** | Reduce max iterations (250 → 100) | 5 minutes | 1.2-1.5x | Low |
| **3** | Analytical Gaussian fit | 2-4 hours | 3-5x | Medium |
| **4** | Centroid for low-contrast | 30 minutes | 1.2x | Low |
| **5** | Skip Migrad fallback conditionally | 15 minutes | 1.1-1.3x | Low |
| **6** | Precompute pixel offsets | 15 minutes | 1.05x | None |
| **7** | RDataFrame batch I/O | 1-2 hours | 1.1-1.2x | Medium |
| **8** | SIMD vectorization | 2-3 hours | 1.2-1.5x | Medium |

### Cumulative Speedup Estimates

| Phase | Optimizations Applied | Estimated Time (50k events) |
|-------|----------------------|:---------------------------:|
| Current | None | 6 minutes |
| Phase 1 | Tolerance + iterations + fallback | 2-3 minutes |
| Phase 2 | + Analytical fit + centroid | 30-60 seconds |
| Phase 3 | + I/O + memory + SIMD | 20-40 seconds |

---

## 5. Validation Strategy

### 5.1 Accuracy Validation

Before deploying any optimization, validate that reconstruction quality is preserved:

```bash
# 1. Run with current settings, extract results
root -l -q 'FitGaussian1D.C("baseline.root")'

# 2. Apply optimizations, run again
root -l -q 'FitGaussian1D.C("optimized.root")'

# 3. Compare key metrics
root -l << 'EOF'
TFile f1("baseline.root");
TFile f2("optimized.root");
TTree* t1 = (TTree*)f1.Get("Hits");
TTree* t2 = (TTree*)f2.Get("Hits");

// Compare RMS of reconstruction residuals
t1->Draw("ReconTrueDeltaRowX>>h1(100,-0.01,0.01)");
t2->Draw("ReconTrueDeltaRowX>>h2(100,-0.01,0.01)");

std::cout << "Baseline RMS: " << h1->GetRMS() << std::endl;
std::cout << "Optimized RMS: " << h2->GetRMS() << std::endl;
EOF
```

**Acceptance Criteria**: RMS degradation < 5%

### 5.2 Performance Profiling

Use profiling tools to identify remaining hotspots:

```bash
# CPU profiling with perf
perf record -g root -l -q 'FitGaussian1D.C("test.root")'
perf report

# Detailed call graph with valgrind
valgrind --tool=callgrind root -l -q 'FitGaussian1D.C("test.root")'
kcachegrind callgrind.out.*
```

---

## 6. Recommended Implementation Roadmap

### Week 1: Quick Wins
1. **Day 1**: Implement tolerance and iteration limit changes
2. **Day 2**: Test and validate accuracy preservation
3. **Day 3**: Implement conditional fallback skip
4. **Day 4-5**: Benchmark and document results

### Week 2: Analytical Fitting
1. **Day 1-2**: Implement analytical Gaussian fitter
2. **Day 3**: Integration with existing code path
3. **Day 4**: Validation against numerical results
4. **Day 5**: Performance benchmarking

### Week 3: Structural Improvements
1. Evaluate RDataFrame migration feasibility
2. Implement memory allocation optimizations
3. Profile and identify remaining bottlenecks

---

## 7. Appendix: Code References

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Parallel executor | `src/FitGaussian1D.C` | 950-1452 | Main parallel loop |
| Minimizer config | `src/FitGaussian1D.C` | 482-486 | Global defaults |
| Per-fit config | `src/FitGaussian1D.C` | 409-428 | Fitter setup |
| Gaussian function | `src/FitGaussian1D.C` | 183-190 | `GaussPlusB` |
| Weighted centroid | `src/FitGaussian1D.C` | 292-310 | Fallback method |
| Data loading | `src/FitGaussian1D.C` | 847-886 | Sequential I/O |
| Fit execution | `src/FitGaussian1D.C` | 371-451 | `RunGaussianFit` |

---

## 8. Conclusion

The current implementation is correctly parallelized but uses unnecessarily conservative fitting parameters. By relaxing tolerance requirements and implementing analytical solutions for simple cases, a **6-12x performance improvement** is achievable while maintaining reconstruction accuracy.

The recommended first step is implementing the Phase 1 quick wins (tolerance and iteration limits), which require minimal code changes and provide immediate 2-3x speedup with negligible risk.
