# epicToy: ePIC Toy Detector Simulation

A Geant4-based simulation framework for testing particle interactions with silicon AC-LGAD sensors. This project provides a toy model of the ePIC detector with configurable pixel geometries and analysis tools.

## Project Overview

**epicToy** simulates particle interactions in a pixelated silicon detector with aluminum readout pixels. The simulation features:

- **Realistic AC-LGAD detector geometry** with configurable pixel layouts
- **Advanced particle tracking** with hit analysis and pixel mapping
- **ROOT-based data output** with comprehensive metadata
- **Interactive visualization** using Geant4's OpenGL interface
- **Python analysis toolkit** for data processing and visualization
- **Flexible configuration** through macro commands and messenger interfaces

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Modes](#usage-modes)
- [Configuration](#configuration)
- [Output Data](#output-data)
- [Analysis Tools](#analysis-tools)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

### Required Dependencies

- **[Geant4](https://geant4.web.cern.ch/)** (11.0+) with Qt and OpenGL support
- **[ROOT](https://root.cern/)** (6.20+) for data analysis
- **CMake** (3.5+)
- **C++ compiler** supporting C++17 or later
- **Qt5/Qt6** for GUI support

### Optional Dependencies

- **Python 3.7+** with the following packages for analysis:
  - `uproot` (ROOT file reading)
  - `matplotlib` (visualization)
  - `numpy` (numerical computing)
  - `pandas` (data analysis)

### System Setup

Before building, ensure Geant4 is properly sourced:

```bash
source ~/Geant4/geant4-v11.3.1-install/share/Geant4/geant4make/geant4make.sh
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/epicToy.git
cd epicToy
```

### 2. Build the Project

```bash
# Create and enter build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)
```

## Quick Start

### Interactive Mode (GUI)

```bash
cd build
./epicToy
```

This launches the simulation with:

- **OpenGL 3D visualization** window
- **Geant4 command interface** for real-time control
- **Interactive particle gun** for testing

### Batch Mode

```bash
cd build
./epicToy -m your_macro.mac
# or
./epicToy batch your_macro.mac
```

## 📁 Project Structure

```
epicToy/
├── 📂 src/                    # Source files
│   ├── DetectorConstruction.cc    # Detector geometry and materials
│   ├── EventAction.cc            # Per-event data processing
│   ├── RunAction.cc              # Run-level ROOT output management
│   ├── PrimaryGenerator.cc       # Particle beam configuration
│   ├── PhysicsList.cc            # Physics processes
│   ├── ActionInitialization.cc   # User action setup
│   ├── DetectorMessenger.cc      # UI commands for detector
│   └── SteppingAction.cc         # Step-by-step tracking
├── 📂 include/                # Header files
├── 📂 macros/                 # Geant4 macro files
│   └── vis.mac               # Default visualization setup
├── 📂 python/                 # Analysis toolkit
│   ├── proc/                 # Data processing scripts
│   │   ├── alpha_sim.py      # Alpha particle analysis
│   │   └── vis_sim.py        # Visualization tools
│   └── sim/                  # Simulation utilities
│       └── alpha_demo.py     # Demo analysis script
├── 📄 epicToy.cc              # Main application
├── 📄 CMakeLists.txt          # Build configuration
└── 📄 README.md               # This file
```

## Output Data

### ROOT Tree Structure

The simulation generates `epicToyOutput.root` with the following branches:

| Branch                           | Type            | Units   | Description                  |
| -------------------------------- | --------------- | ------- | ---------------------------- |
| `Edep`                           | Double          | MeV     | Energy deposited in detector |
| `TrueX`, `TrueY`, `TrueZ`           | Double          | mm      | Hit position coordinates     |
| `InitX`, `InitY`, `InitZ`        | Double          | mm      | Initial particle position    |
| `PixelX`, `PixelY`, `PixelZ`     | Double          | mm      | Nearest pixel center         |
| `PixelI`, `PixelJ`               | Integer         | -       | Pixel grid indices           |
| `PixelTrueDistance`                      | Double          | mm      | Distance to nearest pixel    |
| `PixelAlpha`                     | Double          | degrees | Angular size of pixel        |
| `PixelHit`                       | Boolean         | -       | Direct pixel hit flag        |
| `Grid9x9Angles`                  | Vector<Double>  | degrees | Angles to 9×9 pixel grid     |
| `Grid9x9PixelI`, `Grid9x9PixelJ` | Vector<Integer> | -       | 9×9 grid indices             |

### Metadata

Detector configuration is automatically saved as ROOT metadata:

- `GridPixelSize` - Final pixel dimensions
- `GridPixelSpacing` - Center-to-center spacing
- `GridDetectorSize` - Total detector dimensions
- `GridNumBlocksPerSide` - Pixel count per dimension