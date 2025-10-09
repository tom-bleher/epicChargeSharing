# epicChargeSharing

A GEANT4-based simulation for spatial charge sharing studies in segmented pixel-pad sensors.

## Prerequisites

### System Requirements
- **GEANT4** v11.0+ (with Qt and OpenGL support)
- **ROOT** v6.20+ 
- **CMake** v3.9+
- **C++17** compatible compiler
  

## Installation

### Build
```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Batch Mode (required)
```bash
# Macro-driven run (required)
./epicChargeSharing -m ../macros/run.mac
```

## Repository Structure

    epicChargeSharing/
    |-- src/                      # Source implementation
    |   |-- DetectorConstruction.cc
    |   |-- EventAction.cc
    |   |-- RunAction.cc
    |   |-- fitGaus1D.C
    |   \-- fitGaus2D.C
    |-- include/                  # Header files
    |   |-- Constants.hh
    |   \-- ...
    |-- macros/                   # GEANT4 macro files
    |   \-- vis.mac               # Visualization setup
    |-- proc/                     # ROOT post-processing macros
    |   |-- fit/
    |   |   |-- plotFitGaus1D.C
    |   |   |-- plotFitGaus2D.C
    |   |   \-- ...
    |   |-- grid/
    |   |   |-- plotChargeNeighborhood.C
    |   |   \-- ...
    |   \-- gui/
    |       \-- ...
    |-- analysis/
    |   |-- python/
    |   \-- root/
    \-- CMakeLists.txt            # Build configuration

## Contact

For questions or contributions, please open an issue on GitHub or contact via [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il).

## Acknowledgements

The simulation utilizes the analytical model for signal pad sharing as presented in arXiv:2007.09528 by M. Tornago et al.

## Cite

```
@misc{bleher2025,
  author = {Tom Bleher, Igor Korover},
  title = {epicChargeSharing: Geant4 simulation to study charge sharing reconstruction in pixel detectors},
  year = {2025},
  howpublished = {\url{https://github.com/tom-bleher/epicChargeSharing}},
  note = {A GEANT4-based simulation for charge sharing studies in AC-LGAD pixel sensors with a focus on position reconstruction}
}
```