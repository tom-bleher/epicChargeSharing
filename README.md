# epicChargeSharing

GEANT4-based simulation for studying charge sharing in AC-LGAD pixel detectors.

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![GEANT4](https://img.shields.io/badge/GEANT4-11.0%2B-green.svg)](https://geant4.web.cern.ch/)
[![ROOT](https://img.shields.io/badge/ROOT-6.20%2B-orange.svg)](https://root.cern/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

epicChargeSharing simulates charge distribution across neighboring pixels in AC-LGAD (Resistive AC-Coupled Low Gain Avalanche Detector) sensors.

## Quick Start

```bash
# Clone and build
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run simulation (10,000 events with 10 GeV electrons)
./epicChargeSharing -m ../macros/run.mac

# Output: epicChargeSharing.root
```

## Requirements

| Dependency | Version |
|------------|---------|
| GEANT4 | 11.0+ |
| ROOT | 6.20+ |
| CMake | 3.9+ |
| C++ Compiler | C++20 |

## Usage

```bash
# Batch mode (primary usage)
./epicChargeSharing -m macros/run.mac

# Interactive visualization
./epicChargeSharing
```

## Citation

If you use epicChargeSharing in your research, please cite:

```bibtex
@misc{bleher2025,
  author = {Tom Bleher and Igor Korover},
  title = {epicChargeSharing: GEANT4 Simulation for Charge Sharing in AC-LGAD Detectors},
  year = {2025},
  howpublished = {\url{https://github.com/tom-bleher/epicChargeSharing}}
}
```

## Contact

**Authors**: Tom Bleher, Igor Korover
**Email**: [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il)
**Issues**: [GitHub Issues](https://github.com/tom-bleher/epicChargeSharing/issues)

## Acknowledgments

- M. Tornago et al. for the AC-LGAD charge sharing model
