# ROOT File Output Branches Report

## Overview

The epicChargeSharing simulation outputs a ROOT file (`epicChargeSharing.root`) containing a TTree named `tree` with event-by-event data. The branches saved depend on configuration settings in `Config.hh`.

## Configuration Controls

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `DENOMINATOR_MODE` | `Config.hh:197` | `Neighborhood` | Controls which fraction/charge branches are saved |
| `STORE_FULL_GRID` | `Config.hh:222` | `true` | Whether to save full detector grid data |
| `NEIGHBORHOOD_RADIUS` | `Config.hh:221` | `2` | Neighborhood size (2 = 5x5 grid = 25 pixels) |

---

## Branch Categories

### 1. Scalar Branches (Always Saved)

These branches store one value per event.

| Branch | Type | Unit | Description |
|--------|------|------|-------------|
| `TrueX` | Double | mm | True hit X position |
| `TrueY` | Double | mm | True hit Y position |
| `PixelX` | Double | mm | Nearest pixel center X |
| `PixelY` | Double | mm | Nearest pixel center Y |
| `Edep` | Double | MeV | Energy deposited in event |
| `PixelTrueDeltaX` | Double | mm | `PixelX - TrueX` |
| `PixelTrueDeltaY` | Double | mm | `PixelY - TrueY` |
| `ReconX` | Double | mm | Reconstructed X position |
| `ReconY` | Double | mm | Reconstructed Y position |
| `ReconTrueDeltaX` | Double | mm | `ReconX - TrueX` |
| `ReconTrueDeltaY` | Double | mm | `ReconY - TrueY` |

### 2. Classification Branches (Always Saved)

| Branch | Type | Description |
|--------|------|-------------|
| `isPixelHit` | Bool | Whether hit landed on a pixel |
| `NeighborhoodSize` | Int | Number of active cells in neighborhood |
| `NearestPixelI` | Int | Row index of nearest pixel |
| `NearestPixelJ` | Int | Column index of nearest pixel |
| `NearestPixelID` | Int | Global ID of nearest pixel |

### 3. Common Vector Branches (Always Saved)

These are `std::vector<Double_t>` with size = neighborhood capacity (25 for radius=2).

| Branch | Description |
|--------|-------------|
| `NeighborhoodPixelX` | X coordinates of neighborhood pixel centers |
| `NeighborhoodPixelY` | Y coordinates of neighborhood pixel centers |
| `NeighborhoodPixelID` | Global IDs of neighborhood pixels (`std::vector<Int_t>`) |
| `d_i` | Distance from hit to each pixel center |
| `alpha_i` | Solid angle subtended by each pixel |

---

## 4. Mode-Specific Neighborhood Branches

**Controlled by:** `Constants::DENOMINATOR_MODE` in `Config.hh:197`

### Mode: `Neighborhood` (Default)

Fractions normalized by sum over all neighborhood pixels: `F_i = w_i / Σ_neighborhood(w_n)`

| Branch | Type | Description |
|--------|------|-------------|
| `Fi` | `vector<Double>` | Signal fraction (neighborhood denominator) |
| `Qi` | `vector<Double>` | Induced charge = `Fi * Q_total` |
| `Qn` | `vector<Double>` | Charge after gain noise |
| `Qf` | `vector<Double>` | Final charge after additive noise |

### Mode: `RowCol`

Fractions normalized by row/column sums: `F_i_row = w_i / Σ_row(w_n)`, `F_i_col = w_i / Σ_col(w_n)`

| Branch | Type | Description |
|--------|------|-------------|
| `FiRow` | `vector<Double>` | Signal fraction (row denominator) |
| `FiCol` | `vector<Double>` | Signal fraction (column denominator) |
| `QiRow` | `vector<Double>` | Row-based induced charge |
| `QnRow` | `vector<Double>` | Row-based charge after gain noise |
| `QfRow` | `vector<Double>` | Row-based final charge |
| `QiCol` | `vector<Double>` | Column-based induced charge |
| `QnCol` | `vector<Double>` | Column-based charge after gain noise |
| `QfCol` | `vector<Double>` | Column-based final charge |

### Mode: `ChargeBlock`

Fractions normalized by sum of 4 closest pixels: `F_i_block = w_i / Σ_4closest(w_n)`

| Branch | Type | Description |
|--------|------|-------------|
| `FiBlock` | `vector<Double>` | Signal fraction (4-pixel block denominator) |
| `QiBlock` | `vector<Double>` | Block-based induced charge |
| `QnBlock` | `vector<Double>` | Block-based charge after gain noise |
| `QfBlock` | `vector<Double>` | Block-based final charge |

---

## 5. Full Grid Branches

**Controlled by:** `Constants::STORE_FULL_GRID` in `Config.hh:222`

When `STORE_FULL_GRID = true`, additional branches store data for the entire detector pixel grid (not just the neighborhood). Vector size = `FullGridSide * FullGridSide`.

### Common Full Grid Branches (Always Saved when full grid enabled)

| Branch | Type | Description |
|--------|------|-------------|
| `FullGridSide` | Int | Number of pixels per side |
| `DistanceGrid` | `vector<Double>` | Distance from hit to each pixel |
| `AlphaGrid` | `vector<Double>` | Solid angle for each pixel |
| `PixelXGrid` | `vector<Double>` | X coordinates of all pixels |
| `PixelYGrid` | `vector<Double>` | Y coordinates of all pixels |

### Mode-Specific Full Grid Branches

#### Mode: `Neighborhood`

| Branch | Description |
|--------|-------------|
| `FiGrid` | Signal fractions (neighborhood denominator) |
| `QiGrid` | Induced charges |
| `QnGrid` | Charges after gain noise |
| `QfGrid` | Final charges |

#### Mode: `RowCol`

| Branch | Description |
|--------|-------------|
| `FiRowGrid` | Row-denominator fractions |
| `FiColGrid` | Column-denominator fractions |
| `QiRowGrid`, `QnRowGrid`, `QfRowGrid` | Row-based charges |
| `QiColGrid`, `QnColGrid`, `QfColGrid` | Column-based charges |

#### Mode: `ChargeBlock`

| Branch | Description |
|--------|-------------|
| `FiBlockGrid` | Block-denominator fractions |
| `QiBlockGrid`, `QnBlockGrid`, `QfBlockGrid` | Block-based charges |

---

## Charge Calculation Physics

### Signal Fraction (Tornago Eq. 4)
```
F_i = w_i / Σ_n(w_n)
```
where `w_i = α_i / ln(d_i/d_0)` is the weight for pixel i.

### Denominator Options
- **Neighborhood**: `Σ_n` over all pixels in the (2r+1)×(2r+1) neighborhood
- **Row**: `Σ_n` over pixels in the same row as pixel i
- **Column**: `Σ_n` over pixels in the same column as pixel i
- **Block**: `Σ_n` over the 4 pixels closest to the hit position

### Charge Computation
```
Q_induced = F_i × Q_total
Q_noisy = Q_induced × (1 + gain_noise)
Q_final = max(0, Q_noisy + additive_noise)
```

---

## Special Values

| Value | Meaning |
|-------|---------|
| `-999.0` | Sentinel for out-of-bounds or invalid fraction |
| `NaN` | Uninitialized distance/alpha values |
| `-1` | Invalid pixel ID or index |

---

## How to Change Mode

Edit `Config.hh` line 197:

```cpp
// For neighborhood mode (default):
inline constexpr ECS::Config::DenominatorMode DENOMINATOR_MODE = ECS::Config::DenominatorMode::Neighborhood;

// For row/column mode:
inline constexpr ECS::Config::DenominatorMode DENOMINATOR_MODE = ECS::Config::DenominatorMode::RowCol;

// For 4-pixel block mode:
inline constexpr ECS::Config::DenominatorMode DENOMINATOR_MODE = ECS::Config::DenominatorMode::ChargeBlock;
```

Then rebuild the project.

---

## Example: Reading the ROOT File

```cpp
TFile* f = TFile::Open("epicChargeSharing.root");
TTree* tree = (TTree*)f->Get("tree");

// Scalar data
Double_t trueX, trueY, edep;
tree->SetBranchAddress("TrueX", &trueX);
tree->SetBranchAddress("TrueY", &trueY);
tree->SetBranchAddress("Edep", &edep);

// Vector data (for Neighborhood mode)
std::vector<Double_t>* Fi = nullptr;
std::vector<Double_t>* Qi = nullptr;
tree->SetBranchAddress("Fi", &Fi);
tree->SetBranchAddress("Qi", &Qi);

for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->GetEntry(i);
    // Process event...
}
```

---

## Metadata

Simulation parameters are stored as `TNamed` key-value objects in the ROOT file. This allows analysis scripts to access the exact configuration used for each run.

### Metadata Fields

| Key | Type | Description |
|-----|------|-------------|
| `MetadataSchemaVersion` | String | Schema version (currently "2") |
| `GridPixelSize_mm` | Double | Pixel size in mm |
| `GridPixelSpacing_mm` | Double | Pixel pitch/spacing in mm |
| `GridPixelCornerOffset_mm` | Double | Offset from detector edge in mm |
| `GridDetectorSize_mm` | Double | Full detector size in mm |
| `GridNumBlocksPerSide` | Int | Number of pixels per side |
| `FullGridSide` | Int | Full grid dimension (if enabled) |
| `NeighborhoodRadius` | Int | Neighborhood radius (2 = 5x5) |
| `ChargeSharingModel` | String | Position reconstruction model: `LogA`, `LinA`, or `DPC` |
| `DenominatorMode` | String | Fraction denominator: `Neighborhood`, `RowCol`, or `ChargeBlock` |
| `ChargeSharingLinearBeta_per_um` | Double | β parameter for LinA/DPC models |
| `ChargeSharingPitch_mm` | Double | Pixel pitch for charge sharing |
| `ChargeSharingReferenceD0_microns` | Double | d₀ reference distance (Tornago Eq. 4) |
| `IonizationEnergy_eV` | Double | Energy per e-h pair (3.6 eV) |
| `Gain` | Double | AC-LGAD amplification factor |
| `ElementaryCharge_C` | Double | Electron charge in Coulombs |
| `NoisePixelGainSigmaMin` | Double | Minimum gain noise sigma |
| `NoisePixelGainSigmaMax` | Double | Maximum gain noise sigma |
| `NoiseElectronCount` | Double | Additive noise in electrons |
| `ChargeSharingEmitDistanceAlpha` | Bool | Whether d_i/α_i are stored |
| `ChargeSharingFullFractionsEnabled` | Bool | Whether full grid is stored |
| `PostProcessFitGaus1DEnabled` | Bool | 1D Gaussian fit enabled |
| `PostProcessFitGaus2DEnabled` | Bool | 2D Gaussian fit enabled |

### Reading Metadata

```cpp
TFile* f = TFile::Open("epicChargeSharing.root");

// Read a string metadata value
TNamed* mode = (TNamed*)f->Get("DenominatorMode");
if (mode) {
    std::cout << "Denominator Mode: " << mode->GetTitle() << std::endl;
}

// Read a numeric value
TNamed* d0 = (TNamed*)f->Get("ChargeSharingReferenceD0_microns");
if (d0) {
    double d0_value = std::stod(d0->GetTitle());
}
```

---

## File Location

Output file: `build/epicChargeSharing.root`
