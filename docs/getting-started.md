# Getting Started

This guide walks you through installing, building, and running your first epicChargeSharing simulation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building](#building)
- [Your First Simulation](#your-first-simulation)
- [Understanding the Output](#understanding-the-output)
- [Next Steps](#next-steps)

## Prerequisites

### Required Software

| Software | Minimum Version | Installation |
|----------|-----------------|--------------|
| **GEANT4** | 11.0 | [geant4.web.cern.ch](https://geant4.web.cern.ch/) |
| **ROOT** | 6.20 | [root.cern](https://root.cern/) |
| **CMake** | 3.9 | Package manager or [cmake.org](https://cmake.org/) |
| **C++ Compiler** | C++20 support | GCC 10+, Clang 10+, or MSVC 2019+ |

### Installing GEANT4

```bash
# Download and extract GEANT4
wget https://geant4-data.web.cern.ch/releases/geant4-v11.2.0.tar.gz
tar xzf geant4-v11.2.0.tar.gz
cd geant4-v11.2.0

# Build
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/geant4 \
      -DGEANT4_USE_QT=ON \
      -DGEANT4_INSTALL_DATA=ON \
      -DGEANT4_USE_OPENGL_X11=ON \
      ..
make -j$(nproc)
sudo make install

# Source environment
source /opt/geant4/bin/geant4.sh
```

### Installing ROOT

```bash
# Using package manager (Ubuntu/Debian)
sudo apt install root-system

# Or from source
wget https://root.cern/download/root_v6.30.00.source.tar.gz
tar xzf root_v6.30.00.source.tar.gz
cd root-6.30.00
mkdir build && cd build
cmake ..
make -j$(nproc)
source bin/thisroot.sh
```

### Verifying Prerequisites

```bash
# Check GEANT4
geant4-config --version
# Should show: 11.x.x

# Check ROOT
root-config --version
# Should show: 6.xx/xx

# Check CMake
cmake --version
# Should show: cmake version 3.x.x

# Check compiler
g++ --version
# Should show: g++ (GCC) 10.x.x or higher
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
```

### Project Structure Overview

```
epicChargeSharing/
├── epicChargeSharing.cc    # Main program entry
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
├── src/                    # Source files
├── macros/                 # GEANT4 macro files
│   ├── run.mac             # Batch execution
│   └── vis.mac             # Visualization
├── farm/                   # Analysis scripts
├── proc/                   # ROOT macros
└── docs/                   # Documentation
```

## Building

### Standard Build

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build (use all available cores)
make -j$(nproc)
```

Expected output:
```
-- The CXX compiler identification is GNU 11.4.0
-- Found Geant4 11.2.0
-- Found ROOT 6.30/00
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/epicChargeSharing/build

[100%] Built target epicChargeSharing
```

### Build with Additional Options

```bash
# Debug build (for development)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# With unit tests
cmake -DBUILD_TESTING=ON ..
make -j$(nproc)
ctest

# With fast math optimizations
cmake -DEC_FAST_MATH=ON ..

# Generate Doxygen documentation
make docs
```

### Troubleshooting Build Issues

**GEANT4 not found**:
```bash
# Set GEANT4 path explicitly
cmake -DGeant4_DIR=/opt/geant4/lib/cmake/Geant4 ..
```

**ROOT not found**:
```bash
# Source ROOT environment first
source /path/to/root/bin/thisroot.sh
cmake ..
```

**C++20 not supported**:
```bash
# Use a newer compiler
export CXX=/usr/bin/g++-11
cmake ..
```

## Your First Simulation

### Step 1: Run the Simulation

```bash
# From the build directory
./epicChargeSharing -m ../macros/run.mac
```

You should see output like:
```
**************************************************************
 Geant4 version Name: geant4-11-02-patch-01    (16-February-2024)
**************************************************************

### Run 0 started.
--> Event 0
--> Event 1000
--> Event 2000
...
--> Event 9000
### Run 0 finished. 10000 events processed.
```

### Step 2: Check the Output

The simulation creates `epicChargeSharing.root`:

```bash
ls -la epicChargeSharing.root
# -rw-r--r-- 1 user user 5.2M Dec  5 12:00 epicChargeSharing.root
```

### Step 3: Quick Inspection

```bash
# Open ROOT and inspect
root -l epicChargeSharing.root
```

```cpp
// In ROOT prompt
.ls                              // List file contents
tree->Print()                    // Show all branches
tree->GetEntries()               // Number of events (should be 10000)
tree->Draw("ReconTrueDeltaX")    // Plot X residual
```

### Step 4: Simple Analysis

Create a quick resolution plot:

```cpp
// Still in ROOT
TH1F* h = new TH1F("h", "X Resolution;#DeltaX (mm);Events", 100, -0.1, 0.1);
tree->Draw("ReconTrueDeltaX>>h");
h->Fit("gaus");

// Get resolution
TF1* fit = h->GetFunction("gaus");
double sigma = fit->GetParameter(2);
cout << "Resolution: " << sigma*1000 << " um" << endl;
```

## Understanding the Output

### ROOT File Contents

```cpp
// List all objects in file
TFile f("epicChargeSharing.root");
f.ls();
```

Output:
```
TFile**         epicChargeSharing.root
 TFile*         epicChargeSharing.root
  KEY: TTree    tree;1  Simulation Results
  KEY: TNamed   ChargeSharingModel;1
  KEY: TNamed   DenominatorMode;1
  KEY: TNamed   Gain;1
  ...
```

### Key Branches Explained

| Branch | What it Contains |
|--------|-----------------|
| `TrueX`, `TrueY` | Where the particle actually hit (truth) |
| `ReconX`, `ReconY` | Where we reconstructed the hit |
| `ReconTrueDeltaX` | Difference: ReconX - TrueX |
| `Fi` | Charge fraction on each pixel (vector) |
| `isPixelHit` | Did particle hit a pixel directly? |

### Reading Metadata

```cpp
// Read simulation parameters
TNamed* model = (TNamed*)f.Get("ChargeSharingModel");
cout << "Model: " << model->GetTitle() << endl;

TNamed* gain = (TNamed*)f.Get("Gain");
cout << "Gain: " << gain->GetTitle() << endl;
```

## Command Line Options

```bash
./epicChargeSharing [options]

Options:
  -m <macro>    Run in batch mode with specified macro file (required)
  -t <threads>  Number of threads for parallel execution
```

### Examples

```bash
# Basic batch run
./epicChargeSharing -m ../macros/run.mac

# Multi-threaded (4 cores)
./epicChargeSharing -m ../macros/run.mac -t 4

# Interactive mode (no macro)
./epicChargeSharing
# Then type commands manually
```

## Modifying the Simulation

### Change Number of Events

Edit `macros/run.mac`:
```
/run/beamOn 100000    # Change from 10000 to 100000
```

### Change Particle Type

Edit `macros/run.mac`:
```
/gun/particle proton  # Instead of e-
/gun/energy 1 GeV     # Adjust energy
```

### Change Detector Configuration

Edit `include/Config.hh` and rebuild:
```cpp
// Change pixel pitch
inline const G4double PIXEL_PITCH = 0.4 * mm;  // Was 0.5 mm

// Change reconstruction method
inline constexpr ReconMethod RECON_METHOD = ReconMethod::LogA;  // Was DPC
```

Then rebuild:
```bash
cd build
make -j$(nproc)
```

## Next Steps

Now that you've run your first simulation, explore:

1. **[Configuration Guide](configuration.md)**: Learn all configurable parameters
2. **[Physics Model](physics-model.md)**: Understand the underlying physics
3. **[Analysis Guide](analysis-guide.md)**: Analyze your results
4. **[Architecture](architecture.md)**: Understand the code structure

### Try These Exercises

1. **Resolution Study**: Run with 100,000 events and measure the position resolution

2. **Particle Comparison**: Compare resolution for electrons vs protons

3. **Position Sweep**: Run at multiple fixed positions:
   ```bash
   # Generate sweep macros
   python3 farm/sweep_x.py --start -250 --end 250 --step 50
   ```

4. **Visualization**: Try interactive mode with visualization:
   ```bash
   ./epicChargeSharing
   /control/execute ../macros/vis.mac
   /run/beamOn 10
   ```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/tom-bleher/epicChargeSharing/issues)
- **Email**: [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il)

## Quick Reference Card

```bash
# Build
mkdir build && cd build && cmake .. && make -j$(nproc)

# Run simulation
./epicChargeSharing -m ../macros/run.mac

# Quick analysis
root -l epicChargeSharing.root
tree->Draw("ReconTrueDeltaX")

# Check resolution
TH1F h("h","",100,-0.1,0.1); tree->Draw("ReconTrueDeltaX>>h"); h.Fit("gaus")

# Python analysis
python3 ../farm/Fi_x.py --input-dir .
```
