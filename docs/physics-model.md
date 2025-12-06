# Physics Model

This document describes the physics underlying epicChargeSharing, based on the AC-LGAD charge sharing model from [Tornago et al. (arXiv:2007.09528)](https://arxiv.org/abs/2007.09528).

## AC-LGAD Detector Principle

**AC-LGAD** (Resistive AC-Coupled Low Gain Avalanche Detector) combines:

1. **LGAD gain layer**: Internal multiplication (~8-25×)
2. **AC coupling**: Capacitive readout spreading signal across pads
3. **Resistive layer**: Enables charge sharing between electrodes

```
                Incident Particle
                       │
    ┌──────────────────▼──────────────────┐
    │           Metal Pads (Al)           │  ← Readout electrodes
    ├─────────────────────────────────────┤
    │        Resistive n+ Layer           │  ← Charge spreading
    ├─────────────────────────────────────┤
    │           p-type Bulk               │  ← Active volume
    ├─────────────────────────────────────┤
    │         Multiplication Layer        │  ← Gain (~20×)
    └─────────────────────────────────────┘
```

The signal on each pad depends on distance from the hit, enabling **sub-pixel position resolution**.

---

## Charge Generation

### Ionization

Charged particles create electron-hole pairs in silicon:

$$N_{\text{pairs}} = \frac{E_{\text{dep}}}{\varepsilon}$$

where ε = 3.6 eV (ionization energy in silicon).

### Amplification

The LGAD gain layer provides internal amplification:

$$N_{\text{amplified}} = N_{\text{pairs}} \times G$$

where G = gain factor (default: 20, typical range: 8-25).

**Example**: 10 GeV electron depositing 10 MeV → ~2.8M pairs → 56M amplified electrons.

---

## Charge Sharing Model

### Tornago Model (Eq. 4)

The signal fraction on pad *i* is:

$$F_i = \frac{\alpha_i / \ln(d_i / d_0)}{\sum_n \alpha_n / \ln(d_n / d_0)}$$

where:

- $F_i$ = Signal fraction on pad *i* (normalized: $\sum F_i = 1$)
- $\alpha_i$ = Angle of view of pad *i* from hit position
- $d_i$ = Distance from hit to pad *i*
- $d_0$ = Transverse hit size (~1 µm)

### Linear Attenuation Model (LinA)

Alternative model using exponential attenuation (Tornago Eq. 6):

$$w_i = \alpha_i \times \exp(-\beta \times d_i)$$

where β = attenuation coefficient.

---

## Position Reconstruction

### LogA/LinA (Chi-Square Fit)

Position reconstructed by minimizing:

$$\chi^2 = \sum_i \frac{(F_i^{\text{measured}} - F_i^{\text{model}}(x,y))^2}{\sigma_i^2}$$

### DPC (Discretized Positioning Circuit)

Uses only the **4 closest pads** (Tornago Section 3.4):

$$x_{\text{reco}} = x_{\text{centroid}} + k_x \times R_x$$

where:

$$R_x = \frac{(Q_2 + Q_4) - (Q_1 + Q_3)}{Q_1 + Q_2 + Q_3 + Q_4}$$

### Method Comparison

| Method | Speed | Resolution | Use Case |
|--------|-------|------------|----------|
| LogA | Slow | Best | High-precision studies |
| LinA | Slow | Good | Alternative attenuation |
| DPC | Fast | Good | Real-time reconstruction |

---

## Noise Modeling

### Gain Noise (Multiplicative)

$$Q_{\text{noisy}} = Q_{\text{ideal}} \times (1 + \mathcal{N}(0, \sigma_{\text{gain}}))$$

where σ_gain is randomly assigned per pixel (1-5%).

### Electronic Noise (Additive)

$$Q_{\text{final}} = \max(0, Q_{\text{noisy}} + \mathcal{N}(0, \sigma_{\text{electronic}}))$$

where σ_electronic = 500 electrons (~80 fC) by default.

---

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

### Neighborhood Definition

For radius *r*, the grid is (2r+1) × (2r+1):

```
r = 2 → 5×5 grid (25 pixels)

┌───┬───┬───┬───┬───┐
│   │   │   │   │   │
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

### Special Values

| Value | Meaning |
|-------|---------|
| `-999.0` | Out-of-bounds or invalid fraction |
| `NaN` | Uninitialized value |
| `-1` | Invalid pixel ID/index |

---

## Validation

### Expected Behavior

- **Central pixel**: Highest charge fraction (~30-50%)
- **Adjacent pixels**: Significant signal (~10-20%)
- **Distant pixels**: Small but non-zero signal
- **Sum of fractions**: Exactly 1.0

### Resolution Scaling

$$\sigma_{\text{position}} \propto \frac{\text{pitch}}{\text{SNR}}$$

Typical DPC resolution: ~10-50 µm for standard AC-LGAD configurations.

---

## References

1. M. Tornago et al., "Resistive AC-Coupled Silicon Detectors: principles of operation and first results from a combined analysis of beam test and laser data," [arXiv:2007.09528](https://arxiv.org/abs/2007.09528) (2021).

2. GEANT4 Physics Reference Manual for particle transport in silicon.
