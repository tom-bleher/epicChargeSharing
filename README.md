# epicChargeSharing

**GEANT4-based Monte Carlo simulation for studying charge sharing in AC-LGAD pixel detectors.**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![GEANT4](https://img.shields.io/badge/GEANT4-11.0%2B-green.svg)](https://geant4.web.cern.ch/)
[![ROOT](https://img.shields.io/badge/ROOT-6.20%2B-orange.svg)](https://root.cern/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

epicChargeSharing simulates charge distribution across neighboring pixels in AC-LGAD (Resistive AC-Coupled Low Gain Avalanche Detector) sensors. It implements the analytical model from [Tornago et al. (arXiv:2007.09528)](https://arxiv.org/abs/2007.09528).

### Features

- **Physics-based charge sharing** using the peer-reviewed Tornago model
- **Three reconstruction methods**: LogA, LinA, and DPC algorithms
- **Realistic noise modeling** with per-pixel gain variations
- **Multithreaded execution** with automatic file merging
- **ROOT output** with event data and simulation metadata

## Quick Start

```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./epicChargeSharing -m ../macros/run.mac
```

Output: `epicChargeSharing.root`

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first run |
| [Configuration](docs/configuration.md) | All parameters and options |
| [Physics Model](docs/physics-model.md) | Charge sharing physics |
| [Output Format](docs/ROOT_OUTPUT_BRANCHES.md) | ROOT file structure |
| [Analysis Guide](docs/analysis-guide.md) | Post-processing workflows |
| [Architecture](docs/architecture.md) | System design |

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| GEANT4 | 11.0+ | Monte Carlo simulation |
| ROOT | 6.20+ | Data output and analysis |
| CMake | 3.9+ | Build system |
| C++ Compiler | C++20 | GCC 10+, Clang 10+, or MSVC 2019+ |

## Usage

```bash
# Batch mode
./epicChargeSharing -m macros/run.mac

# Multithreaded
./epicChargeSharing -m macros/run.mac -t 4

# Interactive
./epicChargeSharing
```

## Reconstruction Methods

| Method | Description |
|--------|-------------|
| LogA | Logarithmic attenuation (Tornago Eq. 4) |
| LinA | Linear attenuation |
| DPC | Discretized Positioning Circuit |

## Citation

```bibtex
@misc{bleher2025epicchargesharing,
  author = {Tom Bleher and Igor Korover},
  title = {epicChargeSharing: GEANT4 Simulation for Charge Sharing in AC-LGAD Detectors},
  year = {2025},
  howpublished = {\url{https://github.com/tom-bleher/epicChargeSharing}}
}
```

Reference: M. Tornago et al., [arXiv:2007.09528](https://arxiv.org/abs/2007.09528)

## Contact

Email at [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il) or open Git Issue.
