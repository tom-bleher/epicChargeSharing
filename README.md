# epicChargeSharingAnalysis

A GEANT4-based simulation for charge sharing studies in AC-LGAD pixel sensors.

## Prerequisites

### System Requirements
- **GEANT4** v11.0+ (with Qt and OpenGL support)
- **ROOT** v6.20+ 
- **CMake** v3.9+
- **C++17** compatible compiler
- **Ceres Solver** v2.0+ (non-linear optimization library)

## Installation

### Build
```bash
git clone https://github.com/tom-bleher/EpicChargeSharingAnalysis.git
cd EpicChargeSharingAnalysis
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Interactive Mode (GUI)
```bash
# Multi-threaded (default uses all CPU cores)
./epicChargeSharing

# Specify number of threads (single-threaded is -t 1)
./epicChargeSharing -t 4
```

### Batch Mode
```bash
# Multi-threaded with macro file (add -t flag to specify num of threads)
./epicChargeSharing -m ../macros/run.mac
```

### Command Line Options
- `-m {file}.mac` : Run in batch mode with specified macro file
- `-t {num}` : Set num of threads (default: all CPUs, use 1 for single-threaded)
- `-header` : Run in batch mode using constants from `include/Control.hh` (no macro file needed)

## Basic Configuration

The simulation behavior is controlled by constants in `include/Control.hh`:
- `PARTICLE_TYPE`: Type of particle to simulate (e.g., "e-", "mu-")
- `PARTICLE_ENERGY`: Particle energy in GeV
- `NUMBER_OF_EVENTS`: Number of events to simulate
- Various fitting and analysis options

### Farming

The simulation includes a farming system for automated parameter studies. It allows to run systematic sweeps across multiple parameter combinations automatically. The farmer takes a Cartesian product of specified parameters and runs the simulation for every combination. Edit `farm/control.yaml` to specify your parameter study:

```yaml
# Global simulation settings
simulation:
  name: "pitch_resolution_study"
  output_base_dir: "./pitch_resolution_study"
  
# Parameters to vary (Cartesian product)
varied_parameters:
  PIXEL_SPACING:
    values: [0.5, 0.1, 0.2]  # in mm

# Fixed parameters for all runs
constant_parameters:
  NUMBER_OF_EVENTS: 10000
  PARTICLE_TYPE: "e-"
  DETECTOR_SIZE: 30.0
  GAUSS_FIT: true
  LORENTZ_FIT: true
  # ...
```

After configuring, run
```bash
# Run with default configuration
python3 farm/farmer.py
```

## Repository Structure

```
EpicChargeSharingAnalysis/
├── src/                          # Source implementation
│   ├── DetectorConstruction.cc   # Detector geometry
│   ├── EventAction.cc            # Event processing & charge sharing
│   ├── GaussFit2D.cc            # 2D Gaussian fitting
│   ├── LorentzFit2D.cc          # 2D Lorentzian fitting
│   ├── PowerLorentzFit2D.cc     # 2D Power-Lorentz fitting
│   ├── *Fit3D.cc                # 3D fitting implementations
│   └── ...
├── include/                      # Header files
│   ├── Control.hh               # Main configuration constants
│   ├── Constants.hh             # Physics constants
│   └── ...
├── macros/                       # GEANT4 macro files
│   └── vis.mac                  # Visualization setup
├── analysis/                     # Analysis scripts
│   ├── python/                  # Python analysis tools
│   └── root/                    # ROOT analysis scripts
└── CMakeLists.txt               # Build configuration
```

## Contact

For questions or contributions, please open an issue on GitHub.

## Acknowledgements

The simulation utilizes the analytical model for signal pad sharing as presented in arXiv:2007.09528, M. Tornago et al.

## Cite

```
@misc{bleher2024epicchargesharinganalysis,
  author = {Tom Bleher, Igor Korover},
  title = {epicChargeSharingAnalysis: Geant4 simulation to study charge sharing reconstruction in pixel detectors},
  year = {2025},
  howpublished = {\url{https://github.com/tom-bleher/epicChargeSharingAnalysis}},
  note = {A GEANT4-based simulation for charge sharing studies in AC-LGAD pixel sensors with a focus on position reconstruction}
}
```