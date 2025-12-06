# epicChargeSharing Naming Conventions Report

This document catalogs the naming conventions used throughout the codebase for functions, variables, classes, and other identifiers.

---

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `ECS` | Epic Charge Sharing - primary namespace for all simulation utilities |
| `ECS::Config` | Configuration structs and enums |
| `ECS::Internal` | Internal implementation details (backward compatibility) |
| `Constants` | Global simulation constants (backward compat wrapper around `ECS::Config`) |

---

## Classes

| Class | File | Purpose |
|-------|------|---------|
| `ActionInitialization` | ActionInitialization.hh | Geant4 user action initialization |
| `ChargeSharingCalculator` | ChargeSharingCalculator.hh | Core charge distribution algorithm (Tornago model) |
| `DetectorConstruction` | DetectorConstruction.hh | Geant4 geometry definition for AC-LGAD |
| `EventAction` | EventAction.hh | Per-event data collection and processing |
| `PhysicsList` | PhysicsList.hh | Geant4 physics processes configuration |
| `PrimaryGenerator` | PrimaryGenerator.hh | Particle gun configuration and vertex sampling |
| `RunAction` | RunAction.hh | Per-run initialization, ROOT file management |
| `SteppingAction` | SteppingAction.hh | Per-step energy deposition tracking |
| `NeighborhoodLayout` | NeighborhoodUtils.hh | Grid layout manager for pixel neighborhoods |
| `RootFileWriter` | RootHelpers.hh | Thread-safe ROOT file creation and management |
| `WorkerSync` | RootHelpers.hh | Multithreaded worker synchronization |
| `BranchConfigurator` | RootIO.hh | ROOT TTree branch setup |
| `TreeFiller` | RootIO.hh | Event data serialization to ROOT trees |
| `MetadataPublisher` | RootIO.hh | Simulation metadata writing |
| `PostProcessingRunner` | RootIO.hh | Post-run ROOT macro execution |

---

## Structs

### Configuration Structs (`ECS::Config`)

| Struct | Purpose |
|--------|---------|
| `DetectorGeometry` | Physical dimensions: detector size, pixel size, spacing, thickness |
| `PhysicsParameters` | Charge sharing model parameters: diffusion sigma, gain layer position |
| `NoiseModel` | Electronic noise parameters for charge collection |
| `ReconstructionConfig` | Position reconstruction algorithm settings |

### Data Structs

| Struct | Location | Purpose |
|--------|----------|---------|
| `SamplingWindow` | PrimaryGenerator.hh | Safe vertex sampling bounds |
| `GridGeometry` | ChargeSharingCalculator.hh | Pixel grid coordinate system |
| `ChargeData` | ChargeSharingCalculator.hh | Per-pixel charge accumulation |
| `EventSummaryData` | RootIO.hh | Aggregated event statistics |
| `EventRecord` | RootIO.hh | Complete event data for ROOT output |

---

## Enums

### `ECS::Config::PosReconModel`
Position reconstruction algorithm selection:
- `Log` - Logarithmic weighting
- `Linear` - Linear interpolation
- `DPC` - Discretized Positioning Circuit

### `ECS::Config::DenominatorMode`
Normalization mode for position reconstruction:
- `Neighborhood` - Use full neighborhood sum
- `ChargeBlock` - Use 2×2 charge block
- `RowCol` - Use row/column sums

### `ECS::Config::ChargeMode`
Charge source selection:
- `Induced` - Induced charge on electrodes
- `Collected` - Collected charge carriers

---

## Member Variable Convention

Following Geant4 conventions, member variables use the `f` prefix:

| Pattern | Example | Description |
|---------|---------|-------------|
| `fVariableName` | `fDetector` | Pointer/reference to detector |
| `fVariableName` | `fPixelSize` | Scalar value |
| `fVariableName` | `fChargeFractions` | Container (vector/array) |

### Key Member Variables by Class

**DetectorConstruction:**
- `fDetSize` - Detector size
- `fPixelSize` - Individual pixel size
- `fPixelSpacing` - Pixel pitch (center-to-center)
- `fPixelCornerOffset` - Offset to first pixel corner
- `fNeighborhoodRadius` - Neighborhood grid radius
- `fDetectorThickness` - Silicon thickness

**ChargeSharingCalculator:**
- `fGridGeometry` - Grid coordinate system
- `fChargeData` - Accumulated charge per pixel
- `fDiffusionSigma` - Charge diffusion parameter
- `fGainLayerZ` - Gain layer position

**EventAction:**
- `fTotalEnergyDeposit` - Cumulative energy deposition
- `fPrimaryPosition` - Primary vertex coordinates
- `fNeighborhoodChargeFractions` - Charge distribution array
- `fReconstructedPosition` - Reconstructed hit position

**RunAction:**
- `fRootWriter` - ROOT file manager
- `fWorkerSync` - Thread synchronization
- `fOutputTree` - ROOT TTree pointer
- `fBranchConfig` - Branch configuration

**PrimaryGenerator:**
- `fParticleGun` - Geant4 particle gun
- `fUseFixedPosition` - Fixed vs random sampling flag
- `fFixedX`, `fFixedY` - Fixed position coordinates
- `fSamplingWindow` - Cached sampling bounds

---

## Constants (`Constants::` namespace)

| Constant | Value | Description |
|----------|-------|-------------|
| `DETECTOR_SIZE` | 30.0 mm | Total detector extent |
| `PIXEL_SIZE` | 450 µm | Active pixel area |
| `PIXEL_SPACING` | 500 µm | Pixel pitch |
| `DETECTOR_THICKNESS` | 50 µm | Silicon thickness |
| `DIFFUSION_SIGMA` | 6.0 µm | Charge spread parameter |
| `GAIN_LAYER_Z` | -25 µm | Gain layer depth |
| `NEIGHBORHOOD_RADIUS` | 2 | Grid half-width (5×5 grid) |
| `PRIMARY_PARTICLE_Z_POSITION` | 5.0 mm | Gun position above detector |

---

## ROOT Branch Names

Output tree branches follow snake_case convention:

| Branch | Type | Description |
|--------|------|-------------|
| `event_id` | Int_t | Event number |
| `primary_x`, `primary_y` | Double_t | True vertex position |
| `reco_x`, `reco_y` | Double_t | Reconstructed position |
| `total_energy` | Double_t | Total energy deposit |
| `charge_fractions` | vector<Double_t> | Neighborhood charge array |
| `center_pixel_charge` | Double_t | Central pixel charge |
| `neighborhood_sum` | Double_t | Total neighborhood charge |

---

## Function Naming Patterns

| Pattern | Example | Usage |
|---------|---------|-------|
| `GetProperty()` | `GetPixelSize()` | Accessor methods |
| `SetProperty()` | `SetRadius()` | Mutator methods |
| `ComputeX()` | `ComputeChargeFraction()` | Calculation methods |
| `ConfigureX()` | `ConfigureMessenger()` | Setup methods |
| `ProcessX()` | `ProcessHits()` | Action methods |
| `IsCondition()` | `IsInsidePixel()` | Boolean queries |

---

## File Naming

| Pattern | Example | Content |
|---------|---------|---------|
| `ClassName.hh` | `RunAction.hh` | Class declaration |
| `ClassName.cc` | `RunAction.cc` | Class implementation |
| `UtilityName.hh` | `NeighborhoodUtils.hh` | Utility functions/classes |
| `script.C` | `FitGaus2D.C` | ROOT macro scripts |
| `script.mac` | `run.mac` | Geant4 macro commands |

---

## Suggested Naming Improvements

Based on the Tornago et al. paper (arXiv:2007.09528) terminology and standard physics conventions, the following naming changes are recommended to enhance clarity and alignment with established literature.

### Enum Values

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `PosReconModel::Log` | `PosReconModel::LogA` | Tornago paper uses "LogA" (Logarithmic Attenuation) consistently |
| `PosReconModel::Linear` | `PosReconModel::LinA` | Tornago paper uses "LinA" (Linear Attenuation) consistently |
| `ChargeMode::Patch` | `ChargeMode::Neighborhood` | More descriptive; matches "neighborhood" terminology used throughout |
| `ChargeMode::FullGrid` | `ChargeMode::FullMatrix` | "Matrix" aligns with RSD matrix terminology in the paper |

### Physics Parameters

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `d0ChargeSharing` | `d0` or `hitTransverseSize` | Paper uses d₀ = 1 µm as "transverse size of the hit" |
| `linearBetaNarrow` | `betaSmallPitch` | Clearer relation to pitch geometry |
| `linearBetaWide` | `betaLargePitch` | Clearer relation to pitch geometry |
| `amplificationFactor` | `gain` | Standard LGAD terminology; paper refers to "gain" |
| `PIXEL_SPACING` | `PIXEL_PITCH` | "Pitch" is the standard term in the paper (e.g., "pad-pitch geometry") |
| `PIXEL_SIZE` | `PAD_SIZE` or `METAL_PAD_SIZE` | Paper distinguishes "pad size" from "pitch" |

### Struct/Class Names

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `GridGeom` | `PixelGridGeometry` | More descriptive; avoids abbreviation |
| `HitInfo` | `ImpactPointInfo` | Paper uses "impact point" terminology |
| `ChargeMatrixSet` | `SignalFractionSet` | Paper uses "signal fraction" (F_i) rather than "charge matrix" |
| `D0Params` | `HitSizeParams` | More physically meaningful |

### Variable Names (Internal)

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `Fi` | `signalFraction` | F_i in paper is "fraction of total signal amplitude" |
| `Qi` | `chargeInduced` | Clearer physics meaning |
| `Qn` | `chargeWithNoise` | Explicit about noise contribution |
| `Qf` | `chargeFinal` | Explicit final processed value |
| `alpha` | `angleOfView` | Paper: "α is the angle of view of the pad" |
| `beta` | `attenuationFactor` | Paper: "β is the attenuation factor" |

### Constants

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `IONIZATION_ENERGY` | `PAIR_CREATION_ENERGY` | Standard silicon detector terminology (3.6 eV/pair) |
| `DETECTOR_WIDTH` | `DETECTOR_THICKNESS` | "Width" is confusing; paper uses "thickness" (50 µm) |
| `DPC_TOP_N_PIXELS` | `DPC_CORNER_PADS` | DPC uses exactly 4 corner pads by definition |

### Function Names

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `CalcPixelAlphaSubtended` | `ComputeAngleOfView` | Matches paper terminology |
| `ComputeChargeFractions` | `ComputeSignalFractions` | Paper uses "signal fraction" F_i |
| `CalcNearestPixel` | `FindNearestPad` | "Pad" is preferred RSD terminology |

### ROOT Branch Names

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `charge_fractions` | `signal_fractions` | Aligns with Tornago F_i terminology |
| `center_pixel_charge` | `center_pad_signal` | "Pad" and "signal" match paper conventions |
| `neighborhood_sum` | `total_signal_amplitude` | Paper: "A_tot = Σ A[i]" |

### File Names

| Current Name | Suggested Name | Rationale |
|--------------|----------------|-----------|
| `FitGaus1D.C` | `FitGaussian1D.C` | Avoid abbreviation |
| `FitGaus2D.C` | `FitGaussian2D.C` | Avoid abbreviation |

---

## Terminology Mapping: Code ↔ Tornago Paper

This table maps simulation code terms to the corresponding Tornago et al. paper definitions:

| Code Term | Paper Term | Paper Equation/Section |
|-----------|------------|------------------------|
| `signalFraction` / `Fi` | F_i | Eq. 4: F_i(α_i, d_i) |
| `d0` | d₀ | "transverse size of hit" = 1 µm |
| `distance` | d_i | distance from hit to pad i metal edge |
| `alpha` / `angleOfView` | α_i | angle of view of pad i |
| `beta` / `attenuationFactor` | β | LinA attenuation factor (Eq. 6) |
| `pitch` | pitch | pad center-to-center spacing |
| `padSize` / `metalSize` | pad size / metal | physical pad dimensions |
| `interpad` | pitch - metal | gap between pads |
| `DPC_Kx`, `DPC_Ky` | k_x, k_y | DPC coefficients (Fig. 7) |
| `gain` | gain | LGAD multiplication factor (8-25 typical) |
| `γ` (gamma) | γ | delay factor in Eq. 5 |
| `ζ` (zeta) | ζ | LinA delay factor in Eq. 7 |

---

## Priority Recommendations

### High Priority (Clarity & Paper Alignment)

1. **Rename `Log` → `LogA`** and **`Linear` → `LinA`** — Direct paper terminology
2. **Rename `PIXEL_SPACING` → `PIXEL_PITCH`** — Standard detector physics term
3. **Rename `Fi`, `Qi`, `Qn`, `Qf`** — Single-letter names lack context
4. **Rename `amplificationFactor` → `gain`** — Standard LGAD terminology

### Medium Priority (Consistency)

5. **Rename `GridGeom` → `PixelGridGeometry`** — Avoid abbreviations
7. **Rename `d0ChargeSharing` → `d0`** — Matches paper symbol

### Low Priority (Nice to Have)

8. **Rename `FitGaus*.C` → `FitGaussian*.C`** — Avoid abbreviation

---

## Summary

The codebase follows consistent conventions:
- **Namespaces**: `ECS::` for simulation code, `Constants::` for global values
- **Classes**: PascalCase (`ChargeSharingCalculator`)
- **Member variables**: f-prefix + PascalCase (`fPixelSize`)
- **Functions**: PascalCase (`ComputeChargeFraction`)
- **Constants**: SCREAMING_SNAKE_CASE (`PIXEL_SIZE`)
- **ROOT branches**: snake_case (`charge_fractions`)
- **Enums**: PascalCase for type, PascalCase for values (`PosReconModel::Log`)

---

## References

- Tornago, M. et al. "Resistive AC-Coupled Silicon Detectors: principles of operation and first results from a combined analysis of beam test and laser data." arXiv:2007.09528v4 (2021). [NIMA]
