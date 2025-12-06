# epicChargeSharing

GEANT4-based Monte Carlo simulation for studying spatial charge sharing in AC-LGAD pixel detectors.

---

## Overview

**epicChargeSharing** simulates how charge from particle interactions distributes across neighboring pixels in segmented AC-LGAD (Resistive AC-Coupled Low Gain Avalanche Detectors) sensors. It implements the analytical model from [Tornago et al. (arXiv:2007.09528)](https://arxiv.org/abs/2007.09528).

## Features

- **Physics-based model** — Implements peer-reviewed Tornago charge sharing model
- **Multiple reconstruction methods** — LogA, LinA, and DPC algorithms
- **ROOT integration** — Comprehensive TTree output with metadata
- **Multithreaded** — Native GEANT4 parallel processing
- **Configurable** — Modular geometry, physics, and noise parameters
- **Analysis tools** — Python scripts and ROOT macros included

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run simulation
./epicChargeSharing -m ../macros/run.mac
```

Output: `epicChargeSharing.root`

---

## Requirements

| Dependency | Version |
|------------|---------|
| GEANT4 | 11.0+ |
| ROOT | 6.20+ |
| CMake | 3.9+ |
| C++ | C++20 |

---

## Reconstruction Methods

| Method | Description | Speed |
|--------|-------------|-------|
| **LogA** | Logarithmic attenuation model | Slow |
| **LinA** | Linear attenuation model | Slow |
| **DPC** | Discretized Positioning Circuit | Fast |

The charge fraction on pixel $i$ follows:

$$
F_i = \frac{\alpha_i / \ln(d_i/d_0)}{\sum_n \alpha_n / \ln(d_n/d_0)}
$$

---

## Citation

```bibtex
@misc{bleher2025epicchargesharing,
  author = {Tom Bleher and Igor Korover},
  title = {epicChargeSharing: GEANT4 Simulation for AC-LGAD Charge Sharing},
  year = {2025},
  howpublished = {\url{https://github.com/tom-bleher/epicChargeSharing}}
}
```

**Reference:** M. Tornago et al., [arXiv:2007.09528](https://arxiv.org/abs/2007.09528)

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/tom-bleher/epicChargeSharing/issues)
- **Email**: [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il)
