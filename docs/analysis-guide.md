# Analysis Guide

This document describes how to analyze simulation output from epicChargeSharing, including built-in tools, Python scripts, and ROOT macros.

## Table of Contents

- [Output Overview](#output-overview)
- [Quick Analysis](#quick-analysis)
- [Python Analysis Scripts](#python-analysis-scripts)
- [ROOT Macros](#root-macros)
- [Position Sweep Studies](#position-sweep-studies)
- [Custom Analysis](#custom-analysis)
- [Visualization](#visualization)

## Output Overview

### ROOT File Structure

The simulation produces `epicChargeSharing.root` containing:

```
epicChargeSharing.root
├── tree (TTree)                    # Event-by-event data
│   ├── TrueX, TrueY, TrueZ        # True hit position
│   ├── PixelX, PixelY             # Nearest pixel center
│   ├── ReconX, ReconY             # Reconstructed position
│   ├── Edep                       # Energy deposit
│   ├── ReconTrueDeltaX/Y          # Reconstruction residuals
│   ├── Fi, Qi, Qn, Qf             # Charge fractions/charges
│   ├── d_i, alpha_i               # Distance/angle arrays
│   └── ...                        # See ROOT_OUTPUT_BRANCHES.md
│
└── Metadata (TNamed)              # Simulation parameters
    ├── ChargeSharingModel
    ├── DenominatorMode
    ├── Gain
    └── ...
```

### Key Branches for Analysis

| Branch | Type | Description |
|--------|------|-------------|
| `TrueX`, `TrueY` | Double | True particle position (mm) |
| `ReconX`, `ReconY` | Double | Reconstructed position (mm) |
| `ReconTrueDeltaX` | Double | X residual: ReconX - TrueX (mm) |
| `ReconTrueDeltaY` | Double | Y residual: ReconY - TrueY (mm) |
| `Fi` | vector<Double> | Signal fractions per pixel |
| `Qi` | vector<Double> | Induced charge per pixel |
| `d_i` | vector<Double> | Distance to each pixel (mm) |

## Quick Analysis

### Using ROOT Interactively

```bash
root -l epicChargeSharing.root
```

```cpp
// List branches
tree->Print();

// Plot reconstruction residuals
tree->Draw("ReconTrueDeltaX");
tree->Draw("ReconTrueDeltaY");

// 2D residual map
tree->Draw("ReconTrueDeltaY:ReconTrueDeltaX", "", "COLZ");

// Reconstruction residual vs true position
tree->Draw("ReconTrueDeltaX:TrueX", "", "COLZ");

// Central pixel charge fraction
tree->Draw("Fi[12]");  // Index 12 = center of 5x5 grid

// Resolution histogram with Gaussian fit
TH1F* h = new TH1F("h", "X Resolution", 100, -0.1, 0.1);
tree->Draw("ReconTrueDeltaX>>h");
h->Fit("gaus");
```

### Using PyROOT

```python
import ROOT

f = ROOT.TFile.Open("epicChargeSharing.root")
tree = f.Get("tree")

# Create histogram
h = ROOT.TH1F("res_x", "X Resolution;#DeltaX (mm);Events", 100, -0.1, 0.1)
tree.Draw("ReconTrueDeltaX>>res_x")

# Fit Gaussian
h.Fit("gaus")
fit = h.GetFunction("gaus")
sigma = fit.GetParameter(2)
print(f"Resolution: {sigma*1000:.1f} um")

# Save plot
c = ROOT.TCanvas()
h.Draw()
c.SaveAs("resolution.png")
```

### Using uproot (Pure Python)

```python
import uproot
import numpy as np
import matplotlib.pyplot as plt

# Open file
f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

# Read branches
true_x = tree["TrueX"].array(library="np")
recon_x = tree["ReconX"].array(library="np")
delta_x = recon_x - true_x

# Plot histogram
plt.hist(delta_x, bins=100, range=(-0.1, 0.1))
plt.xlabel("ΔX (mm)")
plt.ylabel("Events")
plt.savefig("resolution.png")
```

## Python Analysis Scripts

### Location

Analysis scripts are in the `farm/` directory:

```
farm/
├── Fi_x.py              # Charge fraction vs position
├── sigma_f_x.py         # Resolution vs position
├── sweep_x.py           # Generate position sweeps
├── sweep_analysis.py    # Batch sweep processing
└── run_gaussian_fits.py # Execute fit macros
```

### Dependencies

```bash
pip install uproot awkward numpy pandas matplotlib scipy openpyxl
```

### Fi_x.py - Charge Fraction Analysis

Analyzes how charge fractions vary with particle position.

**Usage**:
```bash
python3 farm/Fi_x.py --input-dir /path/to/sweep_files
```

**Options**:
| Option | Description |
|--------|-------------|
| `--input-dir` | Directory with sweep ROOT files |
| `--input-dir2` | Second directory for comparison |
| `--output-dir` | Output directory for plots |
| `--pixel-id` | Specific pixel ID to track |

**Outputs**:
- Excel workbook with mean fractions per position
- PNG plots of F_i vs x position
- Per-file histogram distributions (optional)

### sigma_f_x.py - Resolution Analysis

Fits Gaussians to reconstruction residuals and extracts resolution.

**Usage**:
```bash
python3 farm/sigma_f_x.py --input-dir /path/to/sweep_files
```

**Outputs**:
- Excel workbook: x position, σ, σ_error, std dev
- PNG plots with Gaussian fits
- Summary resolution vs position plot

### sweep_x.py - Position Sweep Generation

Generates macro files for systematic position scans.

**Usage**:
```bash
python3 farm/sweep_x.py --start -250 --end 250 --step 25 --output-dir sweeps
```

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--start` | -250 | Start position (µm) |
| `--end` | 250 | End position (µm) |
| `--step` | 25 | Step size (µm) |
| `--events` | 50000 | Events per position |

## ROOT Macros

### Location

ROOT macros are in `src/` and `proc/`:

```
src/
├── FitGaussian1D.C     # 1D Gaussian fitting
└── FitGaussian2D.C     # 2D Gaussian fitting

proc/
├── fit/                # Fitting macros
└── grid/               # Visualization macros
    ├── plotChargeNeighborhood.C
    ├── plotHitsOnGrid.C
    └── plotPixeldiscretization.C
```

### FitGaussian1D.C

Fits 1D Gaussians to reconstruction residuals.

**Usage**:
```bash
root -l 'src/FitGaussian1D.C("epicChargeSharing.root")'
```

**Features**:
- Fits ReconTrueDeltaX and ReconTrueDeltaY
- Outputs fit parameters (mean, sigma, error)
- Saves fit plots as PNG

### FitGaussian2D.C

Performs 2D Gaussian fitting to the residual distribution.

**Usage**:
```bash
root -l 'src/FitGaussian2D.C("epicChargeSharing.root")'
```

### plotChargeNeighborhood.C

Visualizes charge distribution in the neighborhood grid.

**Usage**:
```bash
root -l 'proc/grid/plotChargeNeighborhood.C("epicChargeSharing.root")'
```

**Output**: Heatmap of charge fractions across the 5×5 neighborhood.

### plotHitsOnGrid.C

Plots hit positions overlaid on the pixel grid.

**Usage**:
```bash
root -l 'proc/grid/plotHitsOnGrid.C("epicChargeSharing.root")'
```

## Position Sweep Studies

### Purpose

Position sweeps vary the particle gun position systematically to study:
- Resolution vs position within a pixel
- Charge sharing patterns
- Edge effects

### Workflow

1. **Generate sweep macros**:
   ```bash
   python3 farm/sweep_x.py --start -250 --end 250 --step 25
   ```

2. **Run simulations** (can be parallelized):
   ```bash
   for mac in sweeps/*.mac; do
       ./epicChargeSharing -m $mac &
   done
   wait
   ```

3. **Analyze results**:
   ```bash
   python3 farm/sigma_f_x.py --input-dir sweeps/
   python3 farm/Fi_x.py --input-dir sweeps/
   ```

### Batch Processing

For large sweeps on a computing cluster:

```bash
# Generate job scripts
python3 farm/sweep_analysis.py --mode generate --config sweep_config.yaml

# Submit jobs
./submit_jobs.sh

# Collect results
python3 farm/sweep_analysis.py --mode collect --input-dir results/
```

## Custom Analysis

### Reading Vector Branches

```python
import uproot
import awkward as ak

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

# Read vector branches (variable-length arrays)
Fi = tree["Fi"].array()  # Returns awkward array
d_i = tree["d_i"].array()

# Convert to numpy for specific event
event_0_fractions = ak.to_numpy(Fi[0])

# Filter valid fractions
valid = Fi[Fi > -900]  # Remove sentinel values
```

### Accessing Metadata

```python
import uproot

f = uproot.open("epicChargeSharing.root")

# List all objects
print(f.keys())

# Read metadata
model = f["ChargeSharingModel"]
print(f"Model: {model}")

gain = f["Gain"]
print(f"Gain: {gain}")
```

```cpp
// In ROOT
TFile* f = TFile::Open("epicChargeSharing.root");
TNamed* model = (TNamed*)f->Get("ChargeSharingModel");
cout << "Model: " << model->GetTitle() << endl;
```

### Computing Additional Quantities

```python
import uproot
import numpy as np

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

# Read data
true_x = tree["TrueX"].array(library="np")
true_y = tree["TrueY"].array(library="np")
recon_x = tree["ReconX"].array(library="np")
recon_y = tree["ReconY"].array(library="np")

# Compute 2D residual distance
delta_r = np.sqrt((recon_x - true_x)**2 + (recon_y - true_y)**2)

# Compute resolution (RMS)
resolution_x = np.std(recon_x - true_x)
resolution_y = np.std(recon_y - true_y)
resolution_r = np.std(delta_r)

print(f"σ_x = {resolution_x*1000:.1f} µm")
print(f"σ_y = {resolution_y*1000:.1f} µm")
print(f"σ_r = {resolution_r*1000:.1f} µm")
```

### Selecting Events

```python
import uproot
import numpy as np

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

# Read classification
is_pixel_hit = tree["isPixelHit"].array(library="np")
edep = tree["Edep"].array(library="np")

# Select pixel hits only
delta_x = tree["ReconTrueDeltaX"].array(library="np")
pixel_hits = delta_x[is_pixel_hit]

# Select high-energy events
high_e = delta_x[edep > 5.0]  # E > 5 MeV
```

## Visualization

### Charge Distribution Heatmap

```python
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

# Read charge fractions (5x5 = 25 values per event)
Fi = tree["Fi"].array()

# Average over all events
Fi_mean = ak.mean(Fi, axis=0)
Fi_2d = ak.to_numpy(Fi_mean).reshape(5, 5)

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(Fi_2d, cmap='hot', origin='lower')
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.set_title("Mean Charge Fraction Distribution")
plt.colorbar(im, label="Fraction")
plt.savefig("charge_heatmap.png")
```

### Resolution vs Position

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# After running sweep analysis
positions_um = [...]  # X positions in µm
resolutions_um = [...]  # Fitted σ in µm
errors_um = [...]  # Fit errors

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(positions_um, resolutions_um, yerr=errors_um,
            fmt='o', capsize=3)
ax.set_xlabel("X Position (µm)")
ax.set_ylabel("Resolution σ (µm)")
ax.set_title("Position Resolution vs Hit Position")
ax.grid(True, alpha=0.3)
plt.savefig("resolution_vs_position.png")
```

### Pixel Grid Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Detector parameters
pitch = 0.5  # mm
pixel_size = 0.1  # mm
n_pixels = 5

fig, ax = plt.subplots(figsize=(8, 8))

# Draw pixel grid
for i in range(n_pixels):
    for j in range(n_pixels):
        x = i * pitch
        y = j * pitch
        rect = patches.Rectangle(
            (x - pixel_size/2, y - pixel_size/2),
            pixel_size, pixel_size,
            linewidth=1, edgecolor='blue', facecolor='lightblue'
        )
        ax.add_patch(rect)

# Add hit position
ax.plot(1.0, 1.0, 'r*', markersize=15, label='Hit')

ax.set_xlim(-0.3, 2.3)
ax.set_ylim(-0.3, 2.3)
ax.set_aspect('equal')
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.legend()
plt.savefig("pixel_grid.png")
```

## Analysis Recipes

### Recipe 1: Basic Resolution Study

```bash
# 1. Run simulation
./epicChargeSharing -m macros/run.mac

# 2. Quick resolution check
root -l -e '
TFile f("epicChargeSharing.root");
TTree* t = (TTree*)f.Get("tree");
t->Draw("ReconTrueDeltaX>>h(100,-0.1,0.1)");
h->Fit("gaus");
'
```

### Recipe 2: Position-Dependent Resolution

```bash
# 1. Generate position sweep
python3 farm/sweep_x.py --start -250 --end 250 --step 25 --output sweeps/

# 2. Run all positions
for mac in sweeps/*.mac; do ./epicChargeSharing -m $mac; done

# 3. Analyze
python3 farm/sigma_f_x.py --input-dir sweeps/
```

### Recipe 3: Charge Sharing Pattern

```bash
# 1. Run at fixed position
./epicChargeSharing -m macros/run.mac

# 2. Plot neighborhood
root -l 'proc/grid/plotChargeNeighborhood.C("epicChargeSharing.root")'

# 3. Export data
python3 -c "
import uproot
import awkward as ak
f = uproot.open('epicChargeSharing.root')
Fi = f['tree']['Fi'].array()
print('Mean fractions:', ak.mean(Fi, axis=0))
"
```

### Recipe 4: Comparing Reconstruction Methods

```bash
# 1. Run with LogA (edit Config.hh, rebuild)
./epicChargeSharing -m macros/run.mac
mv epicChargeSharing.root logA.root

# 2. Run with DPC (edit Config.hh, rebuild)
./epicChargeSharing -m macros/run.mac
mv epicChargeSharing.root dpc.root

# 3. Compare
python3 -c "
import uproot
import numpy as np

for fname in ['logA.root', 'dpc.root']:
    f = uproot.open(fname)
    dx = f['tree']['ReconTrueDeltaX'].array(library='np')
    print(f'{fname}: σ = {np.std(dx)*1000:.1f} µm')
"
```

## Tips and Best Practices

1. **Check metadata first**: Verify simulation parameters before analysis
2. **Filter invalid values**: Use `Fi > -900` to exclude sentinel values
3. **Use uproot for large files**: More memory-efficient than PyROOT
4. **Parallelize sweeps**: Run multiple positions simultaneously
5. **Save intermediate results**: Export to CSV/Excel for later analysis
6. **Document analysis parameters**: Record bin sizes, cuts, fit ranges
