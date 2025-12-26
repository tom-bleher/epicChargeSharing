# epicChargeSharing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GEANT4-based simulation and reconstruction plugin for studying charge sharing position reconstruction in AC-LGAD detectors.

## Overview

epicChargeSharing simulates charge distribution across neighboring pixels in AC-LGAD (Resistive AC-Coupled Low Gain Avalanche Detector) sensors. It implements the analytical model from [Tornago et al. (arXiv:2007.09528)](https://arxiv.org/abs/2007.09528).

### Key Features

- **Analytical charge sharing models**: LogA (logarithmic attenuation), LinA (linear attenuation), and DPC (Discretized Positioning Circuit)
- **Noise simulation**: Per-pixel gain variations and additive electronic noise
- **Position reconstruction**: Built-in 1D and 2D Gaussian fitting
- **Multiple active pixel modes**: Neighborhood, row/column, and charge block configurations
- **Multithreaded execution**: Automatic ROOT file merging across threads
- **EIC integration**: Optional EDM4hep output and EICrecon plugin

### Output

The simulation produces `epicChargeSharing.root` containing:
- `Hits` TTree with per-event hit data, charge fractions, and reconstructed positions
- Typed run metadata in `Hits->GetUserInfo()` (no string parsing required)

## Quick Start

```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./epicChargeSharing -m ../macros/run.mac
```

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| GEANT4 | 11.0+ | Monte Carlo simulation |
| ROOT | 6.20+ | Data output and analysis |
| Eigen3 | 3.3+ | Matrix operations for fitting |
| CMake | 3.9+ | Build system |
| C++ Compiler | C++20 | GCC 10+, Clang 10+, or MSVC 2019+ |

Optional: EDM4hep for EIC integration (`-DWITH_EDM4HEP=ON`)

## Usage

```bash
# Batch mode (recommended)
./epicChargeSharing -m macros/run.mac

# Multithreaded (4 threads)
./epicChargeSharing -m macros/run.mac -t 4

# Interactive mode with visualization
./epicChargeSharing
```

## Configuration

Edit `include/Config.hh` to customize:
- Detector geometry (pixel size, pitch, neighborhood radius)
- Physics parameters (gain, ionization energy, hit size)
- Charge sharing model (LogA, LinA, DPC)
- Noise model (gain spread, electronic noise)
- Fitting options (1D/2D Gaussian, uncertainty models)

See the [documentation](https://tom-bleher.github.io/epicChargeSharing/) for details.

## EICrecon Plugin

The `chargeSharingRecon` plugin integrates charge-sharing reconstruction into the EIC software stack. It processes `edm4hep::SimTrackerHit` collections and outputs reconstructed `edm4eic::TrackerHit` positions with improved spatial resolution.

### Supported Detectors

| Detector | Input Collection | Output Collection |
|----------|-----------------|-------------------|
| B0 Tracker | `B0TrackerHits` | `B0ChargeSharingTrackerHits` |
| Lumi Spectrometer | `LumiSpecTrackerHits` | `LumiSpecTrackerChargeSharingHits` |

### Plugin Features

- **DD4hep integration**: Automatic geometry extraction from `CartesianGridXY`/`CartesianGridXZ` segmentations
- **Charge sharing models**: LogA (logarithmic) and LinA (linear) attenuation models
- **Position reconstruction**: Charge-weighted centroid and 1D/2D Gaussian fitting
- **Noise simulation**: Per-pixel gain variation and electronic noise injection
- **Monitoring output**: Residual histograms, correlation plots, and per-hit TTree for validation

### Quick Start

```bash
# Build inside eic-shell
cmake -S eicrecon -B build/eicrecon -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install

# Run reconstruction
export EICrecon_MY=$(pwd)/eicrecon/install
eicrecon -Pplugins=chargeSharingRecon -Ppodio:output_file=output.edm4hep.root input.edm4hep.root
```

See [eicrecon/README.md](eicrecon/README.md) for full configuration options and usage details.

## Documentation

Full documentation is available at **https://tom-bleher.github.io/epicChargeSharing/**

- [Getting Started](https://tom-bleher.github.io/epicChargeSharing/getting-started/) - Build and run instructions
- [Configuration](https://tom-bleher.github.io/epicChargeSharing/configuration/) - Compile-time and runtime options
- [Physics Model](https://tom-bleher.github.io/epicChargeSharing/physics-model/) - Charge sharing theory
- [Analysis Guide](https://tom-bleher.github.io/epicChargeSharing/analysis-guide/) - Post-processing workflows

## Project Structure

```
epicChargeSharing/
├── include/          # Header files
├── src/              # Source files
├── macros/           # GEANT4 macro scripts
├── eicrecon/         # EICrecon plugin for EIC integration
├── farm/             # Python analysis tools for parameter sweeps
├── proc/             # ROOT macros for visualization
└── docs/             # Documentation source
```

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff). GitHub provides formatted citations via the "Cite this repository" button.

Reference: M. Tornago et al., [Nucl. Instrum. Meth. A 1003 (2021) 165319](https://doi.org/10.1016/j.nima.2021.165319)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

Email at [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il) or open a [GitHub Issue](https://github.com/tom-bleher/epicChargeSharing/issues).
