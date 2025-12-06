# Getting Started

This guide covers installation, building, and running your first epicChargeSharing simulation.

## Prerequisites

### Required Software

| Software | Version | Notes |
|----------|---------|-------|
| GEANT4 | 11.0+ | [geant4.web.cern.ch](https://geant4.web.cern.ch/) |
| ROOT | 6.20+ | [root.cern](https://root.cern/) |
| CMake | 3.9+ | [cmake.org](https://cmake.org/) |
| C++ Compiler | C++20 | GCC 10+, Clang 10+, or MSVC 2019+ |

### Verify Installation

```bash
geant4-config --version   # Should show 11.x.x
root-config --version     # Should show 6.xx/xx
cmake --version           # Should show 3.x.x+
```

## Installation

### Clone and Build

```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build Options

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# With unit tests
cmake -DBUILD_TESTING=ON .. && make -j$(nproc) && ctest

# Generate API documentation
make docs
```

### Troubleshooting

**GEANT4 not found**:
```bash
cmake -DGeant4_DIR=/opt/geant4/lib/cmake/Geant4 ..
```

**ROOT not found**:
```bash
source /path/to/root/bin/thisroot.sh
cmake ..
```

## Running a Simulation

### Basic Execution

```bash
./epicChargeSharing -m ../macros/run.mac
```

Expected output:
```
**************************************************************
 Geant4 version Name: geant4-11-02-patch-01
**************************************************************
### Run 0 started.
--> Event 0
--> Event 1000
...
### Run 0 finished. 10000 events processed.
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-m <macro>` | Run in batch mode with specified macro |
| `-t <threads>` | Number of threads for parallel execution |

```bash
# Multithreaded execution
./epicChargeSharing -m ../macros/run.mac -t 4

# Interactive mode
./epicChargeSharing
```

## Understanding the Output

### ROOT File Structure

The simulation creates `epicChargeSharing.root`:

```bash
root -l epicChargeSharing.root
```

```cpp
.ls                           // List contents
tree->Print()                 // Show all branches
tree->GetEntries()            // Number of events
tree->Draw("ReconTrueDeltaX") // Plot X residual
```

### Key Branches

| Branch | Description |
|--------|-------------|
| `TrueX`, `TrueY` | True hit position |
| `ReconX`, `ReconY` | Reconstructed hit position |
| `ReconTrueDeltaX` | Reconstruction residual (ReconX - TrueX) |
| `Fi` | Charge fraction per pixel (vector) |
| `isPixelHit` | Whether particle hit a pixel directly |

### Quick Resolution Check

```cpp
// In ROOT
TH1F* h = new TH1F("h", "X Resolution;#DeltaX (mm);Events", 100, -0.1, 0.1);
tree->Draw("ReconTrueDeltaX>>h");
h->Fit("gaus");
cout << "Resolution: " << h->GetFunction("gaus")->GetParameter(2)*1000 << " um" << endl;
```

## Modifying the Simulation

### Change Event Count

Edit `macros/run.mac`:
```
/run/beamOn 100000   # Increase from 10000
```

### Change Particle/Energy

Edit `macros/run.mac`:
```
/gun/particle proton
/gun/energy 1 GeV
```

### Change Detector Configuration

Edit `include/Config.hh` and rebuild:
```cpp
inline const G4double PIXEL_PITCH = 0.4 * mm;  // Was 0.5 mm
inline constexpr ReconMethod RECON_METHOD = ReconMethod::LogA;
```

```bash
cd build && make -j$(nproc)
```

## Next Steps

- **[Configuration Guide](configuration.md)**: All configurable parameters
- **[Physics Model](physics-model.md)**: Underlying physics
- **[Analysis Guide](analysis-guide.md)**: Analyze your results
- **[Architecture](architecture.md)**: Code structure

## Quick Reference

```bash
# Build
mkdir build && cd build && cmake .. && make -j$(nproc)

# Run
./epicChargeSharing -m ../macros/run.mac

# Analyze
root -l epicChargeSharing.root -e 'tree->Draw("ReconTrueDeltaX")'
```
