# epicToy: ePIC Toy Detector Simulation

A Geant4-based toy simulation of the ePIC detector. The project provides a simple framework to test particle interactions with silicon detectors.

## Prerequisites

- [Geant4](https://geant4.web.cern.ch/)
- [ROOT](https://root.cern/)
- CMake (3.5+)
- C++ compiler supporting C++17

## Building the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/epicToy.git
cd epicToy

# Create build directory
mkdir -p build
cd build

# Configure with CMake and build
cmake ..
make
```

## Running the Simulation

From the build directory:

```bash
./epicToy
```

This will start the simulation with a graphical user interface where you can interact with the detector and run simulations.

## Output

The simulation produces a ROOT file (`epicToyOutput.root`) containing a tree with the following information:
- Energy deposition
- Position coordinates (x, y, z) of hits
