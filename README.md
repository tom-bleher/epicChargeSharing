# epicToy: LGAD Charge Sharing Simulation

A GEANT4-based Monte Carlo simulation for studying charge sharing and position reconstruction in AC-LGAD (Alternating Current Low Gain Avalanche Detector) pixel sensors.

## Physics Overview

### Detector Model
- **Geometry**: Pixelated silicon detector with aluminum readout pixels
- **Pixel Layout**: Configurable grid with 100 μm pixel size and 500 μm spacing
- **Active Volume**: 30×30 mm² silicon substrate (50 μm thickness)

### Charge Sharing Physics

When a particle deposits energy in the detector, the simulation models the following process:

1. **Electron Generation**: Number of electrons produced: \(N_e = E_{dep} / E_{ion}\)
2. **Amplification**: LGAD amplification factor of 20: \(N_e' = N_e \times 20\)
3. **Charge Distribution**: Total charge \(Q_{tot} = N_e' \times e\) is distributed across a 9×9 pixel neighborhood

#### Charge Fraction Calculation

For each pixel in the neighborhood, the charge fraction is calculated using:

\[
\alpha_i = \tan^{-1}\left[\frac{\ell/2 \times \sqrt{2}}{\ell/2 \times \sqrt{2} + d_i}\right]
\]

\[
F_i = \frac{\alpha_i \times \ln(d_i/d_0)^{-1}}{\sum_j \alpha_j \times \ln(d_j/d_0)^{-1}}
\]

Where:
- \(\ell\): pixel size
- \(d_i\): distance from hit to pixel center
- \(d_0\): 10 μm reference distance

### Position Reconstruction

The simulation implements 2D Gaussian fitting for position reconstruction:

#### Central Row/Column Fitting
- **X-direction**: Fit Gaussian to charge distribution along central row
- **Y-direction**: Fit Gaussian to charge distribution along central column

#### Diagonal Fitting
- **Main diagonal**: Fit along line with slope +1
- **Secondary diagonal**: Fit along line with slope -1

All fits use the form: \(y(x) = A \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) + B\)

## Build Requirements

- **GEANT4** (v11.0+) with Qt and OpenGL support
- **ROOT** (v6.20+) for data output
- **CMake** (v3.5+)
- **C++17** compatible compiler

## Installation

### Environment Setup
```bash
source ~/Geant4/geant4-v11.3.1-install/share/Geant4/geant4make/geant4make.sh
export QT_QPA_PLATFORM=xcb
```

### Build Process
```bash
mkdir build && cd build
cmake .. && make -j5
```

## Usage

### Interactive Mode (GUI)
```bash
./epicToy
```

### Batch Mode
```bash
./epicToy -m ../macros/run.mac
```

## Output Data Structure

The simulation generates ROOT files with the following branches:

### Event Information
| Branch | Type | Units | Description |
|--------|------|-------|-------------|
| `Edep` | Double | MeV | Energy deposited in detector |
| `TrueX/Y/Z` | Double | mm | True hit position |
| `PixelX/Y/Z` | Double | mm | Nearest pixel center |
| `PixelI/J` | Integer | - | Pixel grid indices |
| `PixelHit` | Boolean | - | Direct pixel hit flag |

### Charge Sharing Data (Non-Pixel Hits)
| Branch | Type | Description |
|--------|------|-------------|
| `GridNeighborhoodAngles` | Vector<Double> | Alpha angles for 9×9 grid |
| `GridNeighborhoodChargeFractions` | Vector<Double> | Charge fractions per pixel |
| `GridNeighborhoodCharge` | Vector<Double> | Actual charge values [C] |

### Gaussian Fit Results
| Branch | Type | Description |
|--------|------|-------------|
| `Fit2D_XCenter/YCenter` | Double | Fitted position from row/column |
| `Fit2D_XSigma/YSigma` | Double | Fitted width parameters |
| `Fit2D_XAmplitude/YAmplitude` | Double | Fitted amplitudes |
| `Fit2D_X*Err/Y*Err` | Double | Parameter uncertainties |
| `Fit2D_XChi2red/YChi2red` | Double | Reduced \(\chi^2\) values |
| `FitDiag_Main*/Sec*` | Double | Diagonal fit results |

### Delta Variables
| Branch | Type | Description |
|--------|------|-------------|
| `PixelTrueDeltaX/Y` | Double | Pixel center - true position |
| `GaussTrueDeltaX/Y` | Double | Fitted center - true position |

## Hit Classification

Events are classified into two categories:

1. **Pixel Hits** (`PixelHit = true`):
   - Direct hits on pixel active areas
   - Hits within \(d_0\) (10 μm) of pixel center
   - No charge sharing performed

2. **Non-Pixel Hits** (`PixelHit = false`):
   - Hits farther than \(d_0\) from pixel centers
   - Subject to charge sharing and Gaussian fitting

## Key Algorithms

### 2D Gaussian Fitting
- **Method**: Levenberg-Marquardt optimization
- **Convergence**: Analytical derivatives with damping factor adjustment
- **Error Estimation**: Curvature-based parameter uncertainties
- **Success Criteria**: Minimum 3 data points, stable convergence

### Neighborhood Construction
- **Grid Size**: 9×9 pixels (radius = 4)
- **Center**: Nearest pixel to true hit position
- **Boundary**: Rectangular grid aligned with detector axes

## Physics Validation

The simulation includes several validation mechanisms:
- Energy conservation checks
- Charge fraction normalization (\(\sum F_i = 1\))
- Geometric consistency verification
- Fit quality assessment (\(\chi^2/\text{NDF}\))

## Project Structure

```
epicToy/
├── src/                    # Implementation files
│   ├── DetectorConstruction.cc  # Detector geometry
│   ├── EventAction.cc           # Event processing & charge sharing
│   ├── RunAction.cc             # ROOT output management
│   ├── 2DGaussianFitCeres.cc         # Fitting algorithms
│   └── ...
├── include/                # Header files
├── macros/                 # GEANT4 macro files
└── build/                  # Build directory
```

## References

This simulation implements algorithms for LGAD charge sharing studies relevant to:
- ePIC detector development
- Timing detector position reconstruction
- Pixel sensor optimization studies