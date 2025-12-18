# epicChargeSharing

GEANT4-based simulation and reconstruction plugin for studying charge sharing position reconstruction in AC-LGAD detectors.

## Overview

epicChargeSharing simulates charge distribution across neighboring pixels in AC-LGAD (Resistive AC-Coupled Low Gain Avalanche Detector) sensors. It implements the analytical model from [Tornago et al. (arXiv:2007.09528)](https://arxiv.org/abs/2007.09528).

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
