# epicChargeSharing

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL--3.0--or--later-blue.svg)](LICENSE)

GEANT4-based simulation and reconstruction plugin for studying charge sharing position reconstruction in AC-LGAD detectors.

## Overview

epicChargeSharing simulates charge sharing in AC-LGAD sensors to provide reconstructed position.

### Key Features

- **Analytical charge sharing models**: LogA (logarithmic attenuation) and LinA (linear attenuation)
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
- Charge sharing model (LogA, LinA)
- Noise model (gain spread, electronic noise)
- Fitting options (1D/2D Gaussian, uncertainty models)

See the [documentation](https://tom-bleher.github.io/epicChargeSharing/) for details.

## EDM4hep Output (Standalone)

The standalone simulation can optionally write EDM4hep output files for validation and standalone analysis:

```bash
cmake -DWITH_EDM4HEP=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./epicChargeSharing -m ../macros/run.mac
# Produces: epicChargeSharing.edm4hep.root
```

**Important**: This output uses a simplified CellID encoding (`system:8|layer:4|x:16|y:16`) that is **not** identical to the DD4hep `BitFieldCoder` encoding used by `npsim`/`ddsim` in the EIC production pipeline. The standalone EDM4hep output is suitable for:

- Validating charge sharing algorithms independently
- Comparing standalone results with ROOT TTree output
- Prototyping analysis workflows

**For the EIC pipeline**, use the standard workflow instead:
1. **Simulation**: `npsim` (which produces DD4hep-compatible CellIDs)
2. **Reconstruction**: `eicrecon` with the `chargeSharingRecon` plugin (see below)

The standalone EDM4hep output should not be fed directly into `eicrecon` without accounting for the different CellID encoding.

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

Full documentation is published at <https://tom-bleher.github.io/epicChargeSharing/> and lives in the sibling [`epicChargeSharingDocs`](https://github.com/tom-bleher/epicChargeSharingDocs) repository.

- [Getting Started](https://tom-bleher.github.io/epicChargeSharing/getting-started/installation/) — install dependencies and build the simulation
- [User Guide](https://tom-bleher.github.io/epicChargeSharing/user-guide/running/) — running, configuration, output format, and analysis
- [Physics](https://tom-bleher.github.io/epicChargeSharing/physics/charge-sharing-models/) — LogA / LinA models, noise, position reconstruction
- [EICrecon Plugin](https://tom-bleher.github.io/epicChargeSharing/plugin/overview/) — building and running the `chargeSharingRecon` plugin
- [Reference](https://tom-bleher.github.io/epicChargeSharing/reference/architecture/) — code architecture, build options, CI, glossary

## Project Structure

```
epicChargeSharing/
├── include/          # Header files
├── src/              # Source files
├── core/             # Header-only physics core (shared with the plugin)
├── macros/           # GEANT4 macro scripts
├── eicrecon/         # EICrecon plugin for EIC integration
├── farm/             # Python analysis and parameter sweep tools
├── proc/             # ROOT macros and GUI for visualization
└── tests/            # Unit tests (build with -DBUILD_TESTING=ON)
```

User documentation lives in the sibling [`epicChargeSharingDocs`](https://github.com/tom-bleher/epicChargeSharingDocs) repository.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff). GitHub provides formatted citations via the "Cite this repository" button.

Reference: M. Tornago et al., [Nucl. Instrum. Meth. A 1003 (2021) 165319](https://doi.org/10.1016/j.nima.2021.165319)

## Development

```bash
# Run unit tests
cmake -DBUILD_TESTING=ON ..
make test_charge_sharing_core && ctest

# Static analysis (requires clang-tidy via brew install llvm)
make tidy

# Auto-format code
make format
```

Python analysis scripts require packages listed in `requirements.txt` (`pip install -r requirements.txt`).

## Contributing

Contributions are welcome. Please follow these guidelines:

### Code Style

- C++ naming: PascalCase classes, camelCase methods, `m_snake_case` members, `snake_case` locals, `SCREAMING_SNAKE_CASE` constants
- All source files must carry an SPDX `LGPL-3.0-or-later` header
- Run `make format` before submitting; `make format-check` is enforced in CI

### Building and Testing

```bash
# Standalone simulation
cmake --preset default && cmake --build build

# Unit tests
cmake --preset debug && ctest --test-dir build-debug --output-on-failure

# EICrecon plugin (requires eic-shell)
cmake -S eicrecon -B build/eicrecon -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install
```

### Workflow

1. Fork the repository and create a feature branch
2. Run `make tidy` and `make format-check` to check code quality
3. Run unit tests (see above)
4. Open a pull request against `main`

## Contact

Email at [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il) or open a [GitHub Issue](https://github.com/tom-bleher/epicChargeSharing/issues).
