# Physics Model

This document describes the physics underlying the epicChargeSharing simulation, based on the AC-LGAD charge sharing model from Tornago et al. (arXiv:2007.09528).

## Table of Contents

- [AC-LGAD Detector Principle](#ac-lgad-detector-principle)
- [Charge Generation](#charge-generation)
- [Charge Sharing Model](#charge-sharing-model)
- [Position Reconstruction](#position-reconstruction)
- [Noise Modeling](#noise-modeling)
- [Implementation Details](#implementation-details)

## AC-LGAD Detector Principle

### What is an AC-LGAD?

**AC-LGAD** (Resistive AC-Coupled Low Gain Avalanche Detector) is a silicon detector technology that combines:

1. **Low Gain Avalanche Diode (LGAD)**: Internal multiplication layer providing gain (~8-25×)
2. **AC Coupling**: Capacitive readout that spreads signal across multiple pads
3. **Resistive Layer**: Enables charge sharing between neighboring electrodes

```
                    Incident Particle
                           │
                           ▼
    ┌──────────────────────────────────────────┐
    │           Metal Pads (Al)                │  ← Readout electrodes
    ├──────────────────────────────────────────┤
    │        Resistive n+ Layer                │  ← Charge spreading
    ├──────────────────────────────────────────┤
    │           p-type Bulk                    │  ← Active volume
    ├──────────────────────────────────────────┤
    │         Multiplication Layer             │  ← Gain (~20×)
    ├──────────────────────────────────────────┤
    │           p+ Substrate                   │
    └──────────────────────────────────────────┘
```

### Why Charge Sharing?

In AC-LGADs, the signal induced on a metal pad depends on:

- Distance from the hit position to the pad
- Solid angle subtended by the pad
- Resistive layer properties

This allows **sub-pixel position resolution** by analyzing charge ratios.

## Charge Generation

### Ionization in Silicon

When a charged particle traverses silicon, it creates electron-hole pairs:

$$
N_{\text{pairs}} = \frac{E_{\text{dep}}}{\varepsilon}
$$

Where:

- $E_{\text{dep}}$ = Energy deposited (typically ~100 keV/µm for MIPs)
- $\varepsilon$ = 3.6 eV (ionization energy in silicon)

**Example**: 10 GeV electron depositing 10 MeV creates ~2.8 million e-h pairs.

### Signal Amplification

The LGAD gain layer provides internal amplification:

$$
N_{\text{amplified}} = N_{\text{pairs}} \times G
$$

Where:

- $G$ = Gain factor (default: 20)
- Typical range: 8-25 for AC-LGADs

### Total Charge

The total induced charge is:

$$
Q_{\text{total}} = N_{\text{amplified}} \times e = \frac{E_{\text{dep}}}{\varepsilon} \times G \times e
$$

Where $e = 1.602 \times 10^{-19}$ C.

## Charge Sharing Model

### Tornago Model (Eq. 4)

The fraction of total signal amplitude on pad $i$ is given by:

$$
F_i = \frac{\alpha_i / \ln(d_i / d_0)}{\sum_n \alpha_n / \ln(d_n / d_0)}
$$

Where:

- $F_i$ = Signal fraction on pad $i$ (normalized: $\sum F_i = 1$)
- $\alpha_i$ = Angle of view of pad $i$ from the hit position
- $d_i$ = Distance from hit to pad $i$ (to nearest edge)
- $d_0$ = Transverse hit size (characteristic length, ~1 µm)

### Physical Interpretation

**Angle of View ($\alpha_i$)**: The solid angle subtended by the pad:

```
           ┌────────────┐
           │   Pad i    │
           └──────┬─────┘
                  │ αᵢ
                  │/
            ──────●────── Hit position
```

**Distance Dependence**: The logarithmic term $\ln(d_i/d_0)$ models the attenuation of the induced signal with distance. Pads closer to the hit receive more signal.

### Linear Attenuation Model (LinA)

An alternative model uses exponential attenuation (Tornago Eq. 6):

$$
w_i = \alpha_i \times \exp(-\beta \times d_i)
$$

Where $\beta$ = Attenuation coefficient (default: 0.001-0.003 µm⁻¹).

The LinA model may better describe certain detector geometries.

## Position Reconstruction

### LogA/LinA Reconstruction (Chi-Square Fit)

For LogA and LinA methods, position is reconstructed by minimizing:

$$
\chi^2 = \sum_i \frac{(F_i^{\text{measured}} - F_i^{\text{model}}(x,y))^2}{\sigma_i^2}
$$

**Process**:

1. Measure charge fractions $F_i$ on each pad
2. Scan candidate positions $(x, y)$
3. Calculate model fractions $F_i^{\text{model}}$ at each position
4. Find $(x, y)$ that minimizes $\chi^2$

### DPC Reconstruction (Discretized Positioning Circuit)

DPC uses only the **4 closest pads** to the hit position (Tornago Section 3.4):

```
          │ Pad 1 │ Pad 2 │
          ├───────┼───────┤
          │ Pad 3 │●Pad 4 │  ← Hit near Pad 4
          └───────┴───────┘
```

**Position calculation**:

$$
x_{\text{reco}} = x_{\text{centroid}} + k_x \times R_x
$$

$$
y_{\text{reco}} = y_{\text{centroid}} + k_y \times R_y
$$

Where:

- $x_{\text{centroid}}$, $y_{\text{centroid}}$ = Center of the 4-pad block
- $k_x$, $k_y$ = Calibration constants
- $R_x$, $R_y$ = Charge imbalance ratios

**Charge ratios**:

$$
R_x = \frac{(Q_2 + Q_4) - (Q_1 + Q_3)}{Q_1 + Q_2 + Q_3 + Q_4}
$$

$$
R_y = \frac{(Q_1 + Q_2) - (Q_3 + Q_4)}{Q_1 + Q_2 + Q_3 + Q_4}
$$

### Method Comparison

| Method | Speed | Resolution | Use Case |
|--------|-------|------------|----------|
| **LogA** | Slow (fitting) | Best | High-precision studies |
| **LinA** | Slow (fitting) | Good | Alternative attenuation |
| **DPC** | Fast (direct) | Good | Real-time reconstruction |

## Noise Modeling

### Gain Noise (Multiplicative)

Each pixel has intrinsic gain variations:

$$
Q_{\text{noisy}} = Q_{\text{ideal}} \times (1 + \mathcal{N}(0, \sigma_{\text{gain}}))
$$

Where:

- $\sigma_{\text{gain}}$ = Per-pixel gain noise (randomly assigned between 1-5%)
- Different pixels have different noise levels

### Electronic Noise (Additive)

Electronic readout noise adds Gaussian fluctuations:

$$
Q_{\text{final}} = Q_{\text{noisy}} + \mathcal{N}(0, \sigma_{\text{electronic}})
$$

Where:

- $\sigma_{\text{electronic}} = N_{\text{electrons}} \times e$
- Default: 500 electrons (~80 fC)

### Combined Effect

$$
Q_{\text{final}} = \max(0, Q_{\text{ideal}} \times (1 + \delta_{\text{gain}}) + \delta_{\text{electronic}})
$$

The $\max(0, \ldots)$ ensures non-negative charges (physical constraint).

## Implementation Details

### Coordinate System

```
              Y
              ↑
              │
        ──────┼────── X
              │

    Pixel (0,0) at corner offset from detector edge
```

**Pixel center coordinates**:

$$
x_i = \text{pixelCornerOffset} + i \times \text{pitch}
$$

$$
y_j = \text{pixelCornerOffset} + j \times \text{pitch}
$$

### Neighborhood Definition

For neighborhood radius $r$, the grid is $(2r+1) \times (2r+1)$:

```
    r = 2 → 5×5 grid (25 pixels)

    ┌───┬───┬───┬───┬───┐
    │-2 │-2 │-2 │-2 │-2 │  Row offset from center
    │-2 │-1 │ 0 │+1 │+2 │  Column offset
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │   │ ● │   │   │  ← Center (nearest pixel)
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    └───┴───┴───┴───┴───┘
```

### Distance Calculation

Distance from hit position to pixel $i$:

$$
d_i = \sqrt{(\Delta x)^2 + (\Delta y)^2}
$$

Where $\Delta x = x_{\text{hit}} - x_{\text{pixel},i}$ and $\Delta y = y_{\text{hit}} - y_{\text{pixel},i}$.

Clamped to avoid singularity: $d_i = \max(d_i, d_0 \times \text{guardFactor})$

### Angle of View Calculation

Solid angle approximation for rectangular pads:

$$
\alpha_i \approx \frac{w_{\text{pixel}} \times h_{\text{pixel}}}{4\pi \times d_i^2}
$$

For more accurate calculation at close range:

$$
\alpha_i = \arctan\left(\frac{\text{pixelSize}/2}{d_i}\right)
$$

### Fraction Normalization Modes

**Neighborhood Mode**:

$$
F_i = \frac{w_i}{\sum_{\text{neighborhood}} w_n}
$$

**Row/Column Mode** (for 1D fitting):

$$
F_i^{\text{row}} = \frac{w_i}{\sum_{\text{same row}} w_n}, \quad F_i^{\text{col}} = \frac{w_i}{\sum_{\text{same col}} w_n}
$$

**Block Mode** (for DPC):

$$
F_i^{\text{block}} = \frac{w_i}{\sum_{\text{4 closest}} w_n}
$$

### Special Values

| Value | Meaning |
|-------|---------|
| `-999.0` | Out-of-bounds or invalid fraction |
| `NaN` | Uninitialized value |
| `-1` | Invalid pixel ID/index |

## Validation

### Expected Behavior

1. **Central pixel**: Highest charge fraction (~30-50% for typical geometry)
2. **Adjacent pixels**: Significant signal (~10-20%)
3. **Distant pixels**: Small but non-zero signal
4. **Sum of fractions**: Exactly 1.0 (by construction)

### Resolution Scaling

Theoretical position resolution scales as:

$$
\sigma_{\text{position}} \propto \frac{\text{pitch}}{\text{SNR}}
$$

Where:

- pitch = Pixel pitch (default: 500 µm)
- SNR = Signal-to-noise ratio

### DPC Resolution

For DPC reconstruction:

$$
\sigma_{\text{DPC}} \approx \frac{\text{pitch}}{4 \times \text{SNR}}
$$

Typical: ~10-50 µm for standard AC-LGAD configurations.

## References

1. **Tornago et al.** "Resistive AC-Coupled Silicon Detectors: principles of operation and first results from a combined analysis of beam test and laser data," [arXiv:2007.09528](https://arxiv.org/abs/2007.09528) (2021).

2. **LGAD Collaboration** papers on AC-LGAD development and characterization.

3. **GEANT4 Physics Reference Manual** for particle transport in silicon.
