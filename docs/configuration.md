# Configuration Guide

Complete reference for all configuration parameters in epicChargeSharing.

## Quick Start

Edit `include/Config.hh` and modify the **USER SETTINGS** section:

```cpp
// Reconstruction mode: Log, Linear, or DPC
inline constexpr Mode ACTIVE_MODE = Mode::Log;

// Detector geometry
inline const G4double PIXEL_SIZE  = 0.1 * mm;
inline const G4double PIXEL_PITCH = 0.5 * mm;

// Physics
inline constexpr G4double GAIN = 20.0;  // AC-LGAD gain (8-25)
```

After editing, rebuild: `cd build && make -j$(nproc)`

---

## Mode Selection

| Mode | Description | Fitting | Speed |
|------|-------------|---------|-------|
| `Log` | Logarithmic attenuation (Tornago Eq. 4) | 1D Gaussian | Slower |
| `Linear` | Linear attenuation (Tornago Eq. 6) | 1D Gaussian | Slower |
| `DPC` | Discretized Positioning Circuit | None | Fast |

**Signal Models**:

| Model | Formula |
|-------|---------|
| LogA | `w_i = α_i / ln(d_i/d₀)` |
| LinA | `w_i = α_i × exp(-β × d_i)` |

**DPC Mode**: Uses only 4 closest pixels for fast position reconstruction.

```cpp
inline constexpr Mode ACTIVE_MODE = Mode::DPC;
inline constexpr Mode DPC_CHARGE_MODEL = Mode::Log;  // Signal model for DPC
```

---

## Detector Geometry

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `DETECTOR_SIZE` | 30.0 | mm | Sensor side length |
| `DETECTOR_WIDTH` | 0.05 | mm | Silicon thickness |
| `PIXEL_SIZE` | 0.1 | mm | Pixel pad side length |
| `PIXEL_PITCH` | 0.5 | mm | Pixel center-to-center spacing |
| `PIXEL_CORNER_OFFSET` | 0.1 | mm | Edge to first pixel |
| `NEIGHBORHOOD_RADIUS` | 2 | - | Grid radius (2 = 5×5 = 25 pixels) |

**Computed values**:
```
Pixels per side = (DETECTOR_SIZE - 2 × PIXEL_CORNER_OFFSET) / PIXEL_PITCH + 1
Interpad gap = PIXEL_PITCH - PIXEL_SIZE
```

---

## Physics Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `IONIZATION_ENERGY` | 3.6 | eV | Energy per e-h pair in silicon |
| `GAIN` | 20.0 | - | AC-LGAD amplification (typical: 8-25) |
| `D0` | 1.0 | µm | Transverse hit size (Tornago Eq. 4) |

**Charge generation**:
```
Amplified electrons = (Energy deposit [eV] / 3.6) × GAIN
```

---

## Noise Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_GAIN_SIGMA_MIN` | 0.010 | Min per-pixel gain noise (1%) |
| `PIXEL_GAIN_SIGMA_MAX` | 0.050 | Max per-pixel gain noise (5%) |
| `NOISE_ELECTRON_COUNT` | 500.0 | Electronic noise in electrons |

**Noise application**:
```
Q_noisy = Q × (1 + gain_noise)          // Multiplicative
Q_final = max(0, Q_noisy + electronic_noise)  // Additive
```

---

## DPC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DPC_K_CALIBRATION` | 1.2 | k coefficient multiplier |
| `DPC_TOP_N_PIXELS` | 4 | Number of pixels (always 4) |

```
k = (PIXEL_PITCH - PIXEL_SIZE) × DPC_K_CALIBRATION
```

---

## Linear Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LINEAR_CHARGE_MODEL_BETA` | 0.001 | Attenuation coefficient β |

---

## Runtime Configuration

Some parameters can be changed via G4Messenger commands:

```
/ecs/detector/pixelSize <value> <unit>
/ecs/detector/pixelPitch <value> <unit>
/ecs/gun/fixedPosition <true|false>
/ecs/gun/fixedX <value>
/ecs/gun/fixedY <value>
```

**Example macro**:
```
/run/initialize
/gun/particle e-
/gun/energy 10 GeV
/ecs/gun/fixedPosition true
/ecs/gun/fixedX 0.125 mm
/run/beamOn 1000
```

---

## Configuration Examples

### High-Resolution Study
```cpp
inline constexpr Mode ACTIVE_MODE = Mode::Log;
inline const G4double PIXEL_SIZE  = 0.05 * mm;
inline const G4double PIXEL_PITCH = 0.25 * mm;
inline constexpr G4int NEIGHBORHOOD_RADIUS = 3;
```

### Fast DPC Analysis
```cpp
inline constexpr Mode ACTIVE_MODE = Mode::DPC;
inline constexpr Mode DPC_CHARGE_MODEL = Mode::Log;
```

### High Noise Study
```cpp
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.02;
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.10;
inline constexpr G4double NOISE_ELECTRON_COUNT = 1000.0;
```

### Ideal Detector (No Noise)
```cpp
inline constexpr G4double PIXEL_GAIN_SIGMA_MIN = 0.0;
inline constexpr G4double PIXEL_GAIN_SIGMA_MAX = 0.0;
inline constexpr G4double NOISE_ELECTRON_COUNT = 0.0;
```

---

## Internal Constants

These are typically not modified:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WORLD_SIZE` | 5.0 cm | Simulation world volume |
| `DETECTOR_Z_POSITION` | -1.0 cm | Detector Z position |
| `MAX_STEP_SIZE` | 20.0 µm | Maximum step in silicon |
| `OUT_OF_BOUNDS_SENTINEL` | -999.0 | Invalid fraction marker |

---

## Namespace Structure

```cpp
namespace Constants { /* all config values */ }
namespace Config = Constants;           // Alias
namespace ECS { namespace Config = ::Constants; }  // ECS alias
```

Use `Constants::` for new code; `Config::` and `ECS::Config::` are equivalent.
