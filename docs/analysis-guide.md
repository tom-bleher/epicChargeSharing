# Analysis Guide

How to analyze simulation output from epicChargeSharing.

## Output Structure

### ROOT File Contents

```
epicChargeSharing.root
├── tree (TTree)                    # Event-by-event data
│   ├── TrueX, TrueY, TrueZ        # True hit position
│   ├── ReconX, ReconY             # Reconstructed position
│   ├── ReconTrueDeltaX/Y          # Residuals
│   ├── Fi, Qi, Qn, Qf             # Charge fractions/charges
│   └── d_i, alpha_i               # Distance/angle arrays
│
└── Metadata (TNamed)              # Simulation parameters
```

### Key Branches

| Branch | Type | Description |
|--------|------|-------------|
| `TrueX`, `TrueY` | Double | True particle position (mm) |
| `ReconX`, `ReconY` | Double | Reconstructed position (mm) |
| `ReconTrueDeltaX/Y` | Double | Residuals (mm) |
| `Fi` | vector | Signal fractions per pixel |
| `d_i` | vector | Distance to each pixel (mm) |

---

## Quick Analysis

### ROOT Interactive

```bash
root -l epicChargeSharing.root
```

```cpp
tree->Print();                                    // List branches
tree->Draw("ReconTrueDeltaX");                   // X residual
tree->Draw("ReconTrueDeltaY:ReconTrueDeltaX", "", "COLZ");  // 2D

// Resolution with fit
TH1F* h = new TH1F("h", "X Resolution", 100, -0.1, 0.1);
tree->Draw("ReconTrueDeltaX>>h");
h->Fit("gaus");
cout << "Resolution: " << h->GetFunction("gaus")->GetParameter(2)*1000 << " um" << endl;
```

### PyROOT

```python
import ROOT
f = ROOT.TFile.Open("epicChargeSharing.root")
tree = f.Get("tree")

h = ROOT.TH1F("res_x", "X Resolution;#DeltaX (mm);Events", 100, -0.1, 0.1)
tree.Draw("ReconTrueDeltaX>>res_x")
h.Fit("gaus")
print(f"Resolution: {h.GetFunction('gaus').GetParameter(2)*1000:.1f} um")
```

### Uproot (Pure Python)

```python
import uproot
import numpy as np

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

delta_x = tree["ReconTrueDeltaX"].array(library="np")
print(f"Resolution: {np.std(delta_x)*1000:.1f} um")
```

---

## Python Analysis Scripts

Located in `farm/`:

| Script | Purpose |
|--------|---------|
| `Fi_x.py` | Charge fraction vs position |
| `sigma_f_x.py` | Resolution vs position |
| `sweep_x.py` | Generate position sweep macros |
| `run_gaussian_fits.py` | Execute fit macros |

### Installation

```bash
pip install uproot awkward numpy pandas matplotlib scipy openpyxl
```

### Charge Fraction Analysis

```bash
python3 farm/Fi_x.py --input-dir /path/to/sweep_files
```

### Resolution Analysis

```bash
python3 farm/sigma_f_x.py --input-dir /path/to/sweep_files
```

---

## ROOT Macros

Located in `src/` and `proc/`:

```bash
# 1D Gaussian fit
root -l 'src/FitGaussian1D.C("epicChargeSharing.root")'

# 2D Gaussian fit
root -l 'src/FitGaussian2D.C("epicChargeSharing.root")'

# Charge neighborhood visualization
root -l 'proc/grid/plotChargeNeighborhood.C("epicChargeSharing.root")'
```

---

## Position Sweep Studies

### Workflow

```bash
# 1. Generate sweep macros
python3 farm/sweep_x.py --start -250 --end 250 --step 25

# 2. Run simulations
for mac in sweeps/*.mac; do
    ./epicChargeSharing -m $mac
done

# 3. Analyze results
python3 farm/sigma_f_x.py --input-dir sweeps/
python3 farm/Fi_x.py --input-dir sweeps/
```

---

## Advanced Analysis

### Reading Vector Branches

```python
import uproot
import awkward as ak

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

Fi = tree["Fi"].array()  # Awkward array
valid = Fi[Fi > -900]    # Remove sentinel values
```

### Accessing Metadata

```python
f = uproot.open("epicChargeSharing.root")
model = f["ChargeSharingModel"]
gain = f["Gain"]
```

```cpp
// ROOT
TFile* f = TFile::Open("epicChargeSharing.root");
TNamed* model = (TNamed*)f->Get("ChargeSharingModel");
cout << model->GetTitle() << endl;
```

### Custom Quantities

```python
import uproot
import numpy as np

f = uproot.open("epicChargeSharing.root")
tree = f["tree"]

true_x = tree["TrueX"].array(library="np")
true_y = tree["TrueY"].array(library="np")
recon_x = tree["ReconX"].array(library="np")
recon_y = tree["ReconY"].array(library="np")

# 2D residual
delta_r = np.sqrt((recon_x - true_x)**2 + (recon_y - true_y)**2)
print(f"σ_r = {np.std(delta_r)*1000:.1f} µm")
```

### Event Selection

```python
is_pixel_hit = tree["isPixelHit"].array(library="np")
edep = tree["Edep"].array(library="np")
delta_x = tree["ReconTrueDeltaX"].array(library="np")

# Pixel hits only
pixel_hits = delta_x[is_pixel_hit]

# High-energy events
high_e = delta_x[edep > 5.0]
```

---

## Visualization Examples

### Charge Distribution Heatmap

```python
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

f = uproot.open("epicChargeSharing.root")
Fi = f["tree"]["Fi"].array()

Fi_mean = ak.mean(Fi, axis=0)
Fi_2d = ak.to_numpy(Fi_mean).reshape(5, 5)

plt.imshow(Fi_2d, cmap='hot', origin='lower')
plt.colorbar(label="Fraction")
plt.savefig("charge_heatmap.png")
```

### Resolution vs Position

```python
import matplotlib.pyplot as plt

positions_um = [...]    # From sweep
resolutions_um = [...]  # Fitted σ

plt.errorbar(positions_um, resolutions_um, fmt='o')
plt.xlabel("X Position (µm)")
plt.ylabel("Resolution σ (µm)")
plt.savefig("resolution_vs_position.png")
```

---

## Quick Recipes

### Basic Resolution Study

```bash
./epicChargeSharing -m macros/run.mac
root -l -e 'TFile f("epicChargeSharing.root"); TTree* t=(TTree*)f.Get("tree"); t->Draw("ReconTrueDeltaX>>h(100,-0.1,0.1)"); h->Fit("gaus");'
```

### Comparing Reconstruction Methods

```bash
# Run with LogA, then DPC (edit Config.hh, rebuild between)
python3 -c "
import uproot
import numpy as np
for fname in ['logA.root', 'dpc.root']:
    f = uproot.open(fname)
    dx = f['tree']['ReconTrueDeltaX'].array(library='np')
    print(f'{fname}: σ = {np.std(dx)*1000:.1f} µm')
"
```

---

## Tips

1. **Check metadata first**: Verify simulation parameters before analysis
2. **Filter invalid values**: Use `Fi > -900` to exclude sentinels
3. **Use uproot for large files**: More memory-efficient than PyROOT
4. **Parallelize sweeps**: Run multiple positions simultaneously
5. **Save intermediate results**: Export to CSV/Excel for later use
