# epicChargeSharingAnalysis

A GEANT4-based simulation for charge sharing studies in AC-LGAD pixel sensors.

## Prerequisites

### System Requirements
- **GEANT4** v11.0+ (with Qt and OpenGL support)
- **ROOT** v6.20+ 
- **CMake** v3.9+
- **C++17** compatible compiler
- **Ceres Solver** v2.0+ (non-linear optimization library)

### Installing Ceres Solver

#### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev

# Download and build Ceres
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

#### CentOS/RHEL/Fedora
```bash
# Install dependencies
sudo yum install cmake glog-devel gflags-devel atlas-devel eigen3-devel suitesparse-devel
# Or for newer versions: sudo dnf install ...

# Build Ceres (same as above)
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

#### macOS
```bash
# Using Homebrew
brew install ceres-solver

# Or build from source (if needed)
brew install cmake eigen glog gflags suite-sparse
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Installation

### Environment Setup
```bash
# Source GEANT4 environment
source /path/to/geant4-install/share/Geant4/geant4make/geant4make.sh
export QT_QPA_PLATFORM=xcb  # For GUI mode
```

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
./epicChargeSharing
```

### Batch Mode
```bash
./epicChargeSharing -m ../macros/run.mac
```

## Repository Structure

```
EpicChargeSharingAnalysis/
├── src/                          # Source implementation
│   ├── DetectorConstruction.cc   # Detector geometry
│   ├── EventAction.cc            # Event processing & charge sharing
│   ├── 2DGaussCeres.cc     # Pos reconstruction
│   └── ...
├── include/                      # Header files
├── macros/                       # GEANT4 macro files
└── CMakeLists.txt               # Build configuration
```

## Contact

For questions or contributions, please open an issue on GitHub or contact me.

## TODO:

- Fix `vis.mac`