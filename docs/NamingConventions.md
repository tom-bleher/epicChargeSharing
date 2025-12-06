# Naming Conventions

Code naming conventions used throughout epicChargeSharing.

## Summary

| Element | Convention | Example |
|---------|------------|---------|
| Namespaces | PascalCase | `ECS`, `Constants` |
| Classes | PascalCase | `ChargeSharingCalculator` |
| Member variables | f-prefix + PascalCase | `fPixelSize` |
| Functions | PascalCase | `ComputeChargeFraction` |
| Constants | SCREAMING_SNAKE | `PIXEL_SIZE` |
| ROOT branches | snake_case | `charge_fractions` |
| Enums | PascalCase | `PosReconModel::LogA` |

---

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `ECS` | Primary namespace for simulation |
| `ECS::Config` | Configuration structs and enums |
| `Constants` | Global simulation constants |

---

## Classes

| Class | Purpose |
|-------|---------|
| `DetectorConstruction` | GEANT4 geometry |
| `ChargeSharingCalculator` | Core charge distribution algorithm |
| `EventAction` | Per-event processing |
| `RunAction` | Run lifecycle, ROOT I/O |
| `PrimaryGenerator` | Particle gun configuration |

---

## Enums

**`ECS::Config::PosReconModel`**: `LogA`, `LinA`, `DPC`

**`ECS::Config::DenominatorMode`**: `Neighborhood`, `ChargeBlock`, `RowCol`

**`ECS::Config::ChargeMode`**: `Induced`, `Collected`

---

## Member Variables

Following GEANT4 conventions, use `f` prefix:

```cpp
fDetSize         // Detector size
fPixelSize       // Pixel dimensions
fPixelSpacing    // Pixel pitch
fChargeFractions // Charge data container
```

---

## Constants

```cpp
DETECTOR_SIZE      // 30.0 mm
PIXEL_SIZE         // 450 µm
PIXEL_PITCH        // 500 µm
NEIGHBORHOOD_RADIUS // 2 (5×5 grid)
```

---

## Function Patterns

| Pattern | Example | Usage |
|---------|---------|-------|
| `GetX()` | `GetPixelSize()` | Accessor |
| `SetX()` | `SetRadius()` | Mutator |
| `ComputeX()` | `ComputeChargeFraction()` | Calculation |
| `IsX()` | `IsInsidePixel()` | Boolean query |

---

## File Naming

| Pattern | Example |
|---------|---------|
| `ClassName.hh/.cc` | `RunAction.hh` |
| `UtilityName.hh` | `NeighborhoodUtils.hh` |
| `script.C` | `FitGaussian1D.C` |
| `script.mac` | `run.mac` |

---

## Code ↔ Tornago Paper Mapping

| Code | Paper | Reference |
|------|-------|-----------|
| `Fi` / `signalFraction` | F_i | Eq. 4 |
| `d0` | d₀ | transverse hit size (1 µm) |
| `d_i` | d_i | distance to pad |
| `alpha_i` | α_i | angle of view |
| `beta` | β | attenuation factor (LinA) |
| `pitch` | pitch | pad center-to-center |
| `gain` | gain | LGAD multiplication |

---

## Reference

M. Tornago et al., "Resistive AC-Coupled Silicon Detectors," [arXiv:2007.09528](https://arxiv.org/abs/2007.09528) (2021).
