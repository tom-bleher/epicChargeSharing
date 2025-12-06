# ROOT Output Format

Reference for the ROOT file output structure produced by epicChargeSharing.

## Overview

The simulation outputs `epicChargeSharing.root` containing:

- **TTree `tree`**: Event-by-event data
- **TNamed objects**: Simulation metadata

---

## Configuration Controls

| Setting | Default | Description |
|---------|---------|-------------|
| `DENOMINATOR_MODE` | `Neighborhood` | Fraction normalization mode |
| `STORE_FULL_GRID` | `true` | Store full detector grid data |
| `NEIGHBORHOOD_RADIUS` | `2` | Grid radius (2 = 5×5 = 25 pixels) |

---

## Branch Categories

### Scalar Branches (Always Saved)

| Branch | Type | Unit | Description |
|--------|------|------|-------------|
| `TrueX`, `TrueY` | Double | mm | True hit position |
| `PixelX`, `PixelY` | Double | mm | Nearest pixel center |
| `Edep` | Double | MeV | Energy deposited |
| `PixelTrueDeltaX/Y` | Double | mm | Pixel - True position |
| `ReconX`, `ReconY` | Double | mm | Reconstructed position |
| `ReconTrueDeltaX/Y` | Double | mm | Recon - True position |

### Classification Branches

| Branch | Type | Description |
|--------|------|-------------|
| `isPixelHit` | Bool | Hit landed on a pixel |
| `NeighborhoodSize` | Int | Active cells in neighborhood |
| `NearestPixelI`, `J` | Int | Nearest pixel indices |
| `NearestPixelID` | Int | Nearest pixel global ID |

### Vector Branches (Neighborhood)

Size = 25 for radius=2 (5×5 grid).

| Branch | Type | Description |
|--------|------|-------------|
| `NeighborhoodPixelX/Y` | vector\<Double\> | Pixel center coordinates |
| `NeighborhoodPixelID` | vector\<Int\> | Pixel global IDs |
| `d_i` | vector\<Double\> | Distance from hit to each pixel |
| `alpha_i` | vector\<Double\> | Solid angle for each pixel |

---

## Mode-Specific Branches

### Neighborhood Mode (Default)

Fractions normalized by neighborhood sum: `F_i = w_i / Σ_neighborhood(w_n)`

| Branch | Description |
|--------|-------------|
| `Fi` | Signal fraction |
| `Qi` | Induced charge = `Fi × Q_total` |
| `Qn` | Charge after gain noise |
| `Qf` | Final charge after additive noise |

### RowCol Mode

Fractions normalized by row/column sums.

| Branch | Description |
|--------|-------------|
| `FiRow`, `FiCol` | Row/column-normalized fractions |
| `QiRow/Qn/Qf`, `QiCol/Qn/Qf` | Corresponding charges |

### ChargeBlock Mode

Fractions normalized by 4 closest pixels.

| Branch | Description |
|--------|-------------|
| `FiBlock` | Block-normalized fractions |
| `QiBlock`, `QnBlock`, `QfBlock` | Corresponding charges |

---

## Full Grid Branches

When `STORE_FULL_GRID = true`:

| Branch | Description |
|--------|-------------|
| `FullGridSide` | Pixels per side |
| `DistanceGrid` | Distance from hit to each pixel |
| `AlphaGrid` | Solid angle for each pixel |
| `PixelXGrid`, `PixelYGrid` | All pixel coordinates |
| `FiGrid`, `QiGrid`, `QnGrid`, `QfGrid` | Charges for entire grid |

---

## Charge Calculation

### Signal Fraction (Tornago Eq. 4)

```
F_i = w_i / Σ_n(w_n)    where w_i = α_i / ln(d_i/d_0)
```

### Charge Computation

```
Q_induced = F_i × Q_total
Q_noisy = Q_induced × (1 + gain_noise)
Q_final = max(0, Q_noisy + electronic_noise)
```

---

## Special Values

| Value | Meaning |
|-------|---------|
| `-999.0` | Out-of-bounds or invalid |
| `NaN` | Uninitialized |
| `-1` | Invalid pixel ID/index |

---

## Metadata

Stored as TNamed key-value pairs:

| Key | Description |
|-----|-------------|
| `GridPixelSize_mm` | Pixel size |
| `GridPixelSpacing_mm` | Pixel pitch |
| `NeighborhoodRadius` | Grid radius |
| `ChargeSharingModel` | LogA, LinA, or DPC |
| `DenominatorMode` | Fraction normalization |
| `Gain` | Amplification factor |
| `NoiseElectronCount` | Electronic noise level |

### Reading Metadata

```cpp
TFile* f = TFile::Open("epicChargeSharing.root");
TNamed* mode = (TNamed*)f->Get("DenominatorMode");
cout << mode->GetTitle() << endl;
```

---

## Example: Reading the File

```cpp
TFile* f = TFile::Open("epicChargeSharing.root");
TTree* tree = (TTree*)f->Get("tree");

Double_t trueX, trueY;
std::vector<Double_t>* Fi = nullptr;

tree->SetBranchAddress("TrueX", &trueX);
tree->SetBranchAddress("Fi", &Fi);

for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->GetEntry(i);
    // Process event...
}
```
