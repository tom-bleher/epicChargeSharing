# Configuration Guide

This document provides a complete reference for all configuration parameters in epicChargeSharing.

## Quick Start

Edit `include/Config.hh` and modify the **USER SETTINGS** section at the top of the file. The most common changes are:

```cpp
// Choose reconstruction mode: Log, Linear, or DPC
inline constexpr Mode ACTIVE_MODE = Mode::Log;

// Detector geometry
inline const G4double PIXEL_SIZE  = 0.1 * mm;
inline const G4double PIXEL_PITCH = 0.5 * mm;

// Physics
inline constexpr G4double GAIN = 20.0;  // AC-LGAD gain (8-25)
```

After editing, rebuild:
```bash
cd build && make -j4
```

## Table of Contents

- [Mode Selection](#mode-selection)
- [Detector Geometry](#detector-geometry)
- [Physics Parameters](#physics-parameters)
- [Noise Model](#noise-model)
- [DPC Parameters](#dpc-parameters)
- [Linear Model Parameters](#linear-model-parameters)
- [Derived Settings](#derived-settings)
- [Runtime Configuration](#runtime-configuration)

---

## Mode Selection

The simulation supports three reconstruction modes selected via `ACTIVE_MODE`:

| Mode | Description | Fitting | Grid Storage |
|------|-------------|---------|--------------|
| `Log` | Logarithmic attenuation (Tornago Eq. 4) | 1D Gaussian | Full |
| `Linear` | Linear attenuation (Tornago Eq. 6) | 1D Gaussian | Full |
| `DPC` | Discretized Positioning Circuit | None | Minimal |

### Signal Models

| Model | Formula | Reference |
|-------|---------|-----------|
| **LogA** | `w_i = α_i / ln(d_i/d₀)` | Tornago et al. Eq. 4 |
| **LinA** | `w_i = α_i × exp(-β × d_i)` | Tornago et al. Eq. 6 |

### DPC Mode

When `ACTIVE_MODE = Mode::DPC`:
- Uses only the 4 closest pixels for position reconstruction
- No Gaussian fitting (position computed directly from charge ratios)
- Faster execution, smaller output files
- Use `DPC_CHARGE_MODEL` to select which signal model computes the charges

```cpp
inline constexpr Mode ACTIVE_MODE = Mode::DPC;
inline constexpr Mode DPC_CHARGE_MODEL = Mode::Log;  // Signal model for DPC
```

---

## Detector Geometry

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `DETECTOR_SIZE` | 30.0 | mm | Square sensor side length |
| `DETECTOR_WIDTH` | 0.05 | mm | Silicon substrate thickness |
| `PIXEL_SIZE` | 0.1 | mm | Pixel pad side length |
| `PIXEL_PITCH` | 0.5 | mm | Pixel center-to-center spacing |
| `PIXEL_CORNER_OFFSET` | 0.1 | mm | Distance from detector edge to first pixel |
| `NEIGHBORHOOD_RADIUS` | 2 | - | Charge sharing grid radius |

### Computed Values

```
Pixels per side = (DETECTOR_SIZE - 2 × PIXEL_CORNER_OFFSET) / PIXEL_PITCH + 1
                = (30 - 0.2) / 0.5 + 1 = 60 pixels

Total pixels    = 60 × 60 = 3,600 pixels

Neighborhood    = (2 × NEIGHBORHOOD_RADIUS + 1)² = 5 × 5 = 25 pixels
```

### Interpad Distance

```
interpad = PIXEL_PITCH - PIXEL_SIZE = 0.5 - 0.1 = 0.4 mm
```

---

## Physics Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `IONIZATION_ENERGY` | 3.6 | eV | Energy per electron-hole pair in silicon |
| `GAIN` | 20.0 | - | AC-LGAD amplification factor (typical: 8-25) |
| `D0` | 1.0 | µm | Transverse hit size parameter (Tornago Eq. 4) |
| `ELEMENTARY_CHARGE` | 1.602×10⁻¹⁹ | C | Electron charge |

### Charge Generation

```
Primary pairs   = (Energy deposit [MeV] × 10⁶) / IONIZATION_ENERGY
Amplified pairs = Primary pairs × GAIN
```

**Example**: 10 GeV electron deposits ~10 MeV → ~2.8M pairs → 56M amplified electrons

---

## Noise Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_GAIN_SIGMA_MIN` | 0.010 | Minimum per-pixel gain noise σ (1%) |
| `PIXEL_GAIN_SIGMA_MAX` | 0.050 | Maximum per-pixel gain noise σ (5%) |
| `NOISE_ELECTRON_COUNT` | 500.0 | Electronic noise in electrons |

### Noise Application

1. **Gain noise** (multiplicative, per-pixel):
   ```
   charge_noisy = charge × (1 + Gaussian(0, σ_gain))
   ```
   Each pixel has a random σ_gain between MIN and MAX, assigned at initialization.

2. **Electronic noise** (additive):
   ```
   σ_electronic = NOISE_ELECTRON_COUNT × ELEMENTARY_CHARGE
   charge_final = max(0, charge_noisy + Gaussian(0, σ_electronic))
   ```

---

## DPC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DPC_K_CALIBRATION` | 1.2 | k coefficient multiplier |
| `DPC_TOP_N_PIXELS` | 4 | Number of pixels for DPC (always 4) |

### DPC k Coefficient

Per Tornago et al. Section 3.4, the k coefficient is:
```
k = interpad × DPC_K_CALIBRATION
  = (PIXEL_PITCH - PIXEL_SIZE) × 1.2
  = 0.4 × 1.2 = 0.48 mm
```

Tune `DPC_K_CALIBRATION` to minimize spatial resolution (typically 1.2-1.5).

---

## Linear Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LINEAR_CHARGE_MODEL_BETA` | 0.001 | Attenuation coefficient β for LinA model |

Used when `ACTIVE_MODE = Mode::Linear` or `DPC_CHARGE_MODEL = Mode::Linear`.

---

## Derived Settings

These are computed automatically from `ACTIVE_MODE` and should not be edited:

| Setting | Value when Log | Value when Linear | Value when DPC |
|---------|----------------|-------------------|----------------|
| `IS_DPC_MODE` | false | false | true |
| `RECON_METHOD` | LogA | LinA | DPC |
| `SIGNAL_MODEL` | LogA | LinA | (from DPC_CHARGE_MODEL) |
| `FIT_GAUS_1D` | true | true | false |
| `FIT_GAUS_2D` | false | false | false |
| `STORE_FULL_GRID` | true | true | false |

---

## Runtime Configuration

### G4Messenger Commands

Some parameters can be changed at runtime before `/run/initialize`:

**Detector**:
```
/ecs/detector/pixelSize <value> <unit>
/ecs/detector/pixelPitch <value> <unit>
/ecs/detector/pixelCornerOffset <value> <unit>
```

**Particle Gun**:
```
/ecs/gun/fixedPosition <true|false>
/ecs/gun/fixedX <value>
/ecs/gun/fixedY <value>
```

### Example Macro

```
/control/verbose 0
/run/initialize

/gun/particle e-
/gun/energy 10 GeV

/ecs/gun/fixedPosition true
/ecs/gun/fixedX 0.125 mm
/ecs/gun/fixedY 0.125 mm
/run/beamOn 1000
```

---

## Configuration Examples

### High-Resolution Study

```cpp
// Config.hh USER SETTINGS
inline constexpr Mode ACTIVE_MODE = Mode::Log;
inline const G4double PIXEL_SIZE  = 0.05 * mm;   // Smaller pixels
inline const G4double PIXEL_PITCH = 0.25 * mm;   // Tighter pitch
inline constexpr G4int NEIGHBORHOOD_RADIUS = 3;  // Larger neighborhood
```

### Fast DPC Analysis

```cpp
// Config.hh USER SETTINGS
inline constexpr Mode ACTIVE_MODE = Mode::DPC;
inline constexpr Mode DPC_CHARGE_MODEL = Mode::Log;
```

### High Noise Study

```cpp
// Config.hh USER SETTINGS
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.02;   // 2%
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.10;   // 10%
inline constexpr G4double NOISE_ELECTRON_COUNT = 1000.0; // More noise
```

### Ideal Detector (No Noise)

```cpp
// Config.hh USER SETTINGS
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.0;
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.0;
inline constexpr G4double NOISE_ELECTRON_COUNT = 0.0;
```

---

## Internal Constants

These are not typically edited:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WORLD_SIZE` | 5.0 cm | Simulation world volume |
| `DETECTOR_Z_POSITION` | -1.0 cm | Z position of detector |
| `PIXEL_WIDTH` | 0.001 mm | Pixel pad thickness |
| `MAX_STEP_SIZE` | 20.0 µm | Maximum step in silicon |
| `PRIMARY_PARTICLE_Z_POSITION` | 0.0 cm | Particle gun Z position |
| `GEOMETRY_TOLERANCE` | 1.0 µm | Geometry overlap tolerance |
| `PRECISION_TOLERANCE` | 1.0 nm | Numerical precision tolerance |
| `OUT_OF_BOUNDS_FRACTION_SENTINEL` | -999.0 | Sentinel for invalid fractions |

---

## Namespace Structure

The configuration uses the `Constants` namespace with aliases for backward compatibility:

```cpp
namespace Constants { /* all config values */ }
namespace Config = Constants;           // Global alias
namespace ECS { namespace Config = ::Constants; }  // ECS namespace alias
```

All code should use `Constants::` for new code, but `Config::` and `ECS::Config::` work identically.
