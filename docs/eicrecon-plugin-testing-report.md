# EICrecon Plugin Testing: Issues & Solutions Report

## Executive Summary

Your EICrecon plugin (`chargeSharingRecon`) is correctly structured but **cannot be directly tested** because of a fundamental data format mismatch between your standalone simulation and EICrecon.

| Component | Your Simulation | EICrecon Expects |
|-----------|-----------------|------------------|
| Format | Custom ROOT TTree | PODIO/EDM4hep |
| Hit Type | `TrueX`, `TrueY`, `Edep` branches | `edm4hep::SimTrackerHit` |
| File Extension | `.root` | `.edm4hep.root` |
| Geometry | Manual configuration | DD4hep XML |

---

## Issue #1: Data Format Incompatibility

**Problem:** Your standalone simulation writes custom ROOT TTree format with branches like `TrueX`, `TrueY`, `Edep`. EICrecon plugins require `edm4hep::SimTrackerHit` collections in PODIO format.

### Method A: Write a Format Converter

Create a standalone tool that reads your TTree and writes EDM4hep.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Uses your existing simulation data; No changes to simulation code; Can batch-convert existing files |
| **Cons** | Extra build dependency (PODIO, EDM4hep); Converter must be maintained; cellID mapping is non-trivial |
| **Effort** | Medium (~200-300 lines of C++) |
| **Recommended** | Yes - Good for testing existing data |

```cpp
// Pseudocode
TTree* input = file->Get<TTree>("tree");
podio::ROOTWriter writer("output.edm4hep.root");
edm4hep::SimTrackerHitCollection hits;
for (event : input) {
    edm4hep::SimTrackerHit hit;
    hit.setPosition({trueX, trueY, trueZ});
    hit.setEDep(edep);
    hit.setCellID(computeCellID(nearestPixelI, nearestPixelJ));
    hits.push_back(hit);
}
writer.writeFrame(frame, "events");
```

---

### Method B: Add EDM4hep Output to Simulation

Modify your GEANT4 simulation to write EDM4hep format directly.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Native EDM4hep output; Single source of truth; Works with full EIC toolchain |
| **Cons** | Significant code changes; Adds PODIO/EDM4hep as build dependencies; May break existing analysis scripts |
| **Effort** | High (~500+ lines, CMake changes) |
| **Recommended** | Yes - Best long-term solution |

Changes required:
- Add `find_package(podio)`, `find_package(EDM4HEP)` to CMakeLists.txt
- Create `EDM4hepIO.cc` alongside `RootIO.cc`
- Write `edm4hep::SimTrackerHitCollection` in `EndOfEventAction`

---

### Method C: Use ePIC Simulation (npsim)

Generate test data using the official EIC simulation framework.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Official EIC format; Full DD4hep geometry integration; Realistic detector response |
| **Cons** | Different physics model than your simulation; Requires eic-shell environment; Less control over charge sharing parameters |
| **Effort** | Low (configuration only) |
| **Recommended** | Yes - Good for integration testing |

```bash
# In eic-shell
npsim --compactFile=$DETECTOR_PATH/epic.xml \
      -N=1000 \
      --enableGun --gun.particle="e-" \
      --gun.energy 10*GeV \
      --outputFile=test_simhits.edm4hep.root
```

---

## Issue #2: No Unit Test Infrastructure

**Problem:** The `BUILD_TESTING` CMake option references a non-existent `tests/` directory.

### Method A: Add Standalone Unit Tests

Create unit tests for the reconstruction algorithm independent of JANA2.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Fast execution; No framework dependencies; Tests core algorithm logic |
| **Cons** | Doesn't test JANA2 integration; Requires test framework setup |
| **Effort** | Medium |
| **Recommended** | Yes - Essential for algorithm validation |

```cpp
// tests/test_reconstructor.cc
TEST(ChargeSharingReconstructor, CenteredHitReturnsPixelCenter) {
    ChargeSharingConfig cfg;
    cfg.pixelSpacingMM = 0.5;
    ChargeSharingReconstructor recon;
    recon.configure(cfg, nullptr, nullptr);

    ChargeSharingReconstructor::Input input;
    input.hitPositionMM = {0.0, 0.0, 0.0};  // Hit at pixel center
    input.energyDepositGeV = 0.001;

    auto result = recon.process(input);
    EXPECT_NEAR(result.reconstructedPositionMM[0], 0.0, 0.001);
}
```

---

### Method B: Integration Tests with Mock Data

Create PODIO test fixtures for end-to-end testing.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Tests full JANA2 integration; Validates PODIO I/O; Catches interface issues |
| **Cons** | Complex setup; Requires PODIO infrastructure; Slower execution |
| **Effort** | High |
| **Recommended** | Optional - Nice to have |

---

## Issue #3: Validation Against Standalone Simulation

**Problem:** Need to verify EICrecon plugin produces same results as standalone.

### Method A: Parallel Comparison Script

Run both systems on equivalent inputs and compare outputs.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Definitive validation; Catches algorithm differences; Reuses existing analysis |
| **Cons** | Requires format conversion first; Complex setup |
| **Effort** | Medium |
| **Recommended** | Yes - Essential for validation |

```python
# compare_recon.py
import uproot
import numpy as np

standalone = uproot.open("epicChargeSharing.root:tree")
eicrecon = uproot.open("output.edm4eic.root:events")

# Compare reconstruction residuals
delta_standalone = standalone["ReconTrueDeltaX"].array()
# For EICrecon, compute: TrackerHit.position.x - SimTrackerHit.position.x
delta_eicrecon = compute_eicrecon_residuals(eicrecon)

print(f"Standalone σ: {np.std(delta_standalone):.4f} mm")
print(f"EICrecon σ:   {np.std(delta_eicrecon):.4f} mm")
assert np.isclose(np.std(delta_standalone), np.std(delta_eicrecon), rtol=0.01)
```

---

### Method B: Golden File Testing

Generate reference outputs and compare against future runs.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Catches regressions; Simple to implement; CI-friendly |
| **Cons** | Requires initial validated output; Brittle to intentional changes |
| **Effort** | Low |
| **Recommended** | Yes - Good for CI/CD |

---

## Issue #4: DD4hep Geometry Integration

**Problem:** Plugin expects DD4hep segmentation but your simulation uses manual geometry.

### Method A: Create Minimal DD4hep Geometry

Write a simple DD4hep XML file for your detector.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Full DD4hep integration; Proper cellID encoding; Works with EIC ecosystem |
| **Cons** | Learning curve for DD4hep XML; Must match simulation geometry exactly |
| **Effort** | Medium-High |
| **Recommended** | Yes - Required for production use |

```xml
<!-- simple_pixel_detector.xml -->
<lccdd>
  <define>
    <constant name="pixel_pitch" value="500*um"/>
    <constant name="pixel_size" value="100*um"/>
  </define>
  <readouts>
    <readout name="ChargeSharingRecon">
      <segmentation type="CartesianGridXY"
                    grid_size_x="pixel_pitch"
                    grid_size_y="pixel_pitch"/>
      <id>system:8,layer:4,x:16,y:16</id>
    </readout>
  </readouts>
</lccdd>
```

---

### Method B: Use Manual Configuration (Current)

Rely on parameter-based configuration without DD4hep.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Already implemented; Works standalone; No external dependencies |
| **Cons** | cellID decoding won't work; Limited ecosystem integration |
| **Effort** | None (current state) |
| **Recommended** | Acceptable for testing only |

---

## Recommended Action Plan

### Phase 1: Quick Validation (1-2 days)

1. **Build and load test**: Verify plugin compiles and registers with `eicrecon -L`
2. **Use npsim**: Generate test EDM4hep data with ePIC simulation
3. **Run plugin**: Process npsim output through your plugin
4. **Basic checks**: Verify output has reconstructed positions

### Phase 2: Algorithm Validation (3-5 days)

1. **Write converter** (Method A from Issue #1): TTree → EDM4hep
2. **Convert test data**: Process your existing simulation output
3. **Run comparison** (Method A from Issue #3): Compare residuals
4. **Add unit tests** (Method A from Issue #2): Test core algorithm

### Phase 3: Production Integration (1-2 weeks)

1. **Create DD4hep geometry** (Method A from Issue #4)
2. **Add EDM4hep output to simulation** (Method B from Issue #1)
3. **Full integration tests**: End-to-end with EIC toolchain

---

## Summary Table

| Issue | Best Method | Effort | Priority |
|-------|-------------|--------|----------|
| Format incompatibility | A: Write converter | Medium | **High** |
| No unit tests | A: Standalone tests | Medium | **High** |
| Validation | A: Comparison script | Medium | **High** |
| DD4hep geometry | B: Manual config (for now) | None | Medium |

---

## Appendix: Quick Reference Commands

### Build Plugin

```bash
cmake -S eicrecon -B build/eicrecon \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install
export EICrecon_MY=$(pwd)/eicrecon/install
```

### Verify Plugin Loads

```bash
eicrecon -L 2>&1 | grep -i chargeshar
```

### Run with npsim Data

```bash
eicrecon -Pplugins=chargeSharingRecon \
         -PChargeSharingRecon:readout=SiTrackerHits \
         -PChargeSharingRecon:neighborhoodRadius=2 \
         test_simhits.edm4hep.root \
         -Ppodio:output=recon_output.edm4eic.root
```

### Generate Test Data (in eic-shell)

```bash
npsim --compactFile=$DETECTOR_PATH/epic.xml \
      -N=1000 \
      --enableGun --gun.particle="e-" \
      --gun.energy 10*GeV \
      --outputFile=test_simhits.edm4hep.root
```

---

## References

- [EIC Tutorial: Reconstruction Algorithms in JANA2](https://eic.github.io/tutorial-jana2/aio/index.html)
- [Creating JANA2 Factories](https://eic.github.io/tutorial-jana2/04-factory/index.html)
- [Creating Custom Plugins](https://eic.github.io/tutorial-jana2/03-end-user-plugin/index.html)
- [GitHub - eic/EICrecon](https://github.com/eic/EICrecon)
- [npsim Documentation](https://github.com/eic/npsim)
- [ePIC Detector Integration](https://www.epic-eic.org/detector/detector_integration.html)
