# Architecture

This document describes the system architecture, component design, and data flow of the epicChargeSharing simulation.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Component Descriptions](#component-descriptions)
- [Data Flow](#data-flow)
- [Threading Model](#threading-model)
- [Namespace Organization](#namespace-organization)

## Overview

epicChargeSharing follows the standard GEANT4 extended example architecture with custom extensions for AC-LGAD charge sharing simulation. The system is organized into:

1. **Simulation Core**: GEANT4 user action classes
2. **Charge Calculation**: Physics-based charge distribution algorithms
3. **I/O Layer**: ROOT file output and metadata management
4. **Configuration**: Compile-time and runtime parameters

## System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              epicChargeSharing.cc           │
                    │                 (Main Entry)                │
                    └─────────────────┬───────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│ DetectorConstr. │      │   ActionInitializ.  │      │   PhysicsList   │
│   (Geometry)    │      │  (Action Factory)   │      │  (EM Physics)   │
└─────────────────┘      └──────────┬──────────┘      └─────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ PrimaryGenerator│      │    RunAction    │      │   EventAction   │
│ (Particle Gun)  │      │ (ROOT I/O, Run) │      │ (Event Process) │
└─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                           │
                                    ┌──────────────────────┴──────────────────────┐
                                    │                                             │
                                    ▼                                             ▼
                         ┌─────────────────────┐                      ┌─────────────────┐
                         │ ChargeSharingCalc.  │                      │  SteppingAction │
                         │ (Core Algorithm)    │                      │ (Track Steps)   │
                         └─────────────────────┘                      └─────────────────┘
```

## Component Descriptions

### DetectorConstruction

**File**: `include/DetectorConstruction.hh`, `src/DetectorConstruction.cc`

Builds the physical detector geometry:

| Element | Dimensions | Material |
|---------|------------|----------|
| World Volume | 5 cm cube | Air |
| Silicon Detector | 30×30×0.05 mm | Silicon |
| Pixel Pads | 0.1×0.1×0.001 mm | Aluminum |

**Key Features**:
- Configurable pixel grid geometry
- Per-pixel gain noise assignment
- Pixel finder for nearest-pixel lookup
- G4Messenger for runtime geometry changes

**Configuration Interface**:
```
/ecs/detector/pixelSize <value> <unit>
/ecs/detector/pixelPitch <value> <unit>
/ecs/detector/pixelCornerOffset <value> <unit>
```

### ChargeSharingCalculator

**File**: `include/ChargeSharingCalculator.hh`, `src/ChargeSharingCalculator.cc`

Core computation engine implementing the Tornago et al. charge sharing model.

**Computation Steps**:
1. Find nearest pixel to hit position
2. Define (2r+1)×(2r+1) neighborhood grid
3. Compute charge fractions F_i for each pixel
4. Apply noise models (gain + electronic)
5. Optionally compute full detector grid fractions

**Key Data Structures**:

```cpp
struct PixelGridGeometry {
    G4int nRows, nCols;      // Grid dimensions
    G4double pitchX, pitchY; // Pixel spacing
    G4double x0, y0;         // Grid origin
};

struct HitInfo {
    G4double trueX, trueY, trueZ;  // True position
    G4int pixRow, pixCol;           // Nearest pixel indices
    G4double pixCenterX, pixCenterY;// Pixel center
};

struct ChargeMatrixSet {
    Grid2D<G4double> signalFraction;     // F_i
    Grid2D<G4double> signalFractionRow;  // Row-normalized
    Grid2D<G4double> signalFractionCol;  // Col-normalized
    Grid2D<G4double> signalFractionBlock;// Block-normalized
    // + corresponding charge grids
};
```

**Signal Models**:

| Model | Formula | Use Case |
|-------|---------|----------|
| LogA | `w_i = α_i / ln(d_i/d₀)` | Standard charge sharing |
| LinA | `w_i = α_i × exp(-β × d_i)` | Linear attenuation |

### EventAction

**File**: `include/EventAction.hh`, `src/EventAction.cc`

Per-event processing and data collection:

**Workflow**:
```
BeginOfEventAction()
    └── Reset accumulators

[GEANT4 tracking loop]
    └── SteppingAction collects energy deposits

EndOfEventAction()
    ├── Determine hit position
    ├── Call ChargeSharingCalculator::Compute()
    ├── Perform position reconstruction (LogA/LinA/DPC)
    └── Fill ROOT tree via RunAction
```

**Position Reconstruction Methods**:

| Method | Algorithm |
|--------|-----------|
| LogA | Chi-square fit to logarithmic model |
| LinA | Chi-square fit to linear model |
| DPC | Direct calculation: `pos = centroid + K × charge_ratio` |

### RunAction

**File**: `include/RunAction.hh`, `src/RunAction.cc`

Run lifecycle and ROOT I/O management:

**Responsibilities**:
- ROOT file creation and TTree setup
- Branch configuration (scalar, vector, classification)
- Worker thread synchronization (multithreaded mode)
- Worker file merging after run completion
- Metadata publishing
- Post-processing macro invocation

**Thread Safety**:
- Master thread creates output file
- Worker threads write to separate files
- Files merged at EndOfRunAction

### PrimaryGenerator

**File**: `include/PrimaryGenerator.hh`, `src/PrimaryGenerator.cc`

Particle gun configuration:

**Modes**:
| Mode | Description |
|------|-------------|
| Fixed | Particles at specified (x,y) coordinates |
| Random | Uniform sampling within detector bounds |

**Smart Margins**: Automatically computes safe margins from edges based on neighborhood radius.

**G4Messenger Commands**:
```
/ecs/gun/fixedPosition <bool>
/ecs/gun/fixedX <value>
/ecs/gun/fixedY <value>
```

### PhysicsList

**File**: `include/PhysicsList.hh`, `src/PhysicsList.cc`

Physics process configuration:

- Standard EM physics (G4EmStandardPhysics)
- Step limiter for controlled tracking
- Maximum step size: 20 µm in silicon

### SteppingAction

**File**: `include/SteppingAction.hh`, `src/SteppingAction.cc`

Lightweight tracker for:
- First-contact volume identification
- Energy deposit accumulation
- Hit classification (pixel vs silicon)

## Data Flow

### Event Processing Pipeline

```
┌──────────────────┐
│  Primary Vertex  │
│   (x, y, z)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Particle Tracking│
│  Energy Deposits │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Hit Position     │
│ Determination    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│         ChargeSharingCalculator              │
│  ┌────────────────────────────────────────┐  │
│  │ 1. Find nearest pixel                  │  │
│  │ 2. Build neighborhood grid             │  │
│  │ 3. Compute distances d_i, angles α_i   │  │
│  │ 4. Calculate weights w_i (LogA/LinA)   │  │
│  │ 5. Normalize: F_i = w_i / Σw_n         │  │
│  │ 6. Apply gain noise                    │  │
│  │ 7. Apply electronic noise              │  │
│  └────────────────────────────────────────┘  │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Position Recon.  │
│ (LogA/LinA/DPC)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ROOT TTree      │
│    Output        │
└──────────────────┘
```

### ROOT Output Structure

```
epicChargeSharing.root
├── tree (TTree)
│   ├── Scalar Branches
│   │   ├── TrueX, TrueY, TrueZ
│   │   ├── PixelX, PixelY
│   │   ├── ReconX, ReconY
│   │   ├── Edep
│   │   └── Delta branches (residuals)
│   │
│   ├── Classification Branches
│   │   ├── isPixelHit
│   │   ├── NeighborhoodSize
│   │   └── NearestPixel indices
│   │
│   ├── Vector Branches (neighborhood)
│   │   ├── Fi, Qi, Qn, Qf
│   │   ├── d_i, alpha_i
│   │   └── Pixel coordinates
│   │
│   └── Full Grid Branches (optional)
│       └── Complete detector maps
│
└── Metadata (TNamed objects)
    ├── Geometry parameters
    ├── Physics settings
    ├── Noise configuration
    └── Reconstruction method
```

## Threading Model

epicChargeSharing supports GEANT4's multithreading when compiled with MT support.

### Architecture

```
┌─────────────────────────────────────────────────┐
│              G4MTRunManager                      │
│                 (Master)                         │
└───────────────────┬─────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌────────┐      ┌────────┐
│Worker 0│    │Worker 1│ ...  │Worker N│
│Thread  │    │Thread  │      │Thread  │
└────┬───┘    └────┬───┘      └────┬───┘
     │             │               │
     ▼             ▼               ▼
worker_0.root  worker_1.root  worker_N.root
     │             │               │
     └─────────────┼───────────────┘
                   │
                   ▼ (merge)
           epicChargeSharing.root
```

### Synchronization

**WorkerSync Class** (`RootHelpers.hh`):
- Atomic counter for worker registration
- Condition variables for ordering
- Mutex protection for file operations

```cpp
struct WorkerSync {
    std::atomic<int> workerCount{0};
    std::mutex mutex;
    std::condition_variable cv;

    void RegisterWorker();
    void WaitForAllWorkers();
};
```

### Thread-Local Storage

Each worker maintains:
- Own EventAction instance
- Own ChargeSharingCalculator
- Own ROOT file handle
- Own random number generator state

## Namespace Organization

```
ECS (Epic Charge Sharing)
├── Config
│   ├── SignalModel (LogA, LinA)
│   ├── ReconMethod (LogA, LinA, DPC)
│   ├── DenominatorMode (Neighborhood, ChargeBlock, RowCol)
│   ├── DetectorGeometry (struct)
│   ├── PhysicsParameters (struct)
│   ├── NoiseModel (struct)
│   ├── ReconstructionConfig (struct)
│   └── GlobalConfig (struct)
│
├── DetectorConstruction
├── ActionInitialization
├── PhysicsList
├── PrimaryGenerator
├── EventAction
├── RunAction
├── SteppingAction
├── ChargeSharingCalculator
│   ├── PixelGridGeometry
│   ├── HitInfo
│   ├── ChargeMode
│   ├── Grid2D<T>
│   ├── ChargeMatrixSet
│   ├── FullGridCharges
│   ├── PatchInfo
│   ├── PatchGridCharges
│   └── Result
│
└── (Utility classes)
    ├── NeighborhoodLayout
    ├── RootFileWriter
    ├── WorkerSync
    ├── BranchConfigurator
    ├── TreeFiller
    ├── MetadataPublisher
    └── PostProcessingRunner
```

### Backward Compatibility

Global aliases maintain compatibility with older code:

```cpp
// In global namespace
using DetectorConstruction = ECS::DetectorConstruction;
using ChargeSharingCalculator = ECS::ChargeSharingCalculator;
// etc.

namespace Constants {
    // Wraps ECS::Config values
    inline constexpr auto DETECTOR_SIZE = ECS::Config::DetectorGeometry::Default().detectorSize;
    // etc.
}
```

## File Dependencies

```
epicChargeSharing.cc
    ├── DetectorConstruction.hh
    ├── PhysicsList.hh
    └── ActionInitialization.hh
            └── PrimaryGenerator.hh
            └── RunAction.hh
            │       ├── RootIO.hh
            │       │       ├── RootHelpers.hh
            │       │       └── Config.hh
            │       └── NeighborhoodUtils.hh
            └── EventAction.hh
            │       └── ChargeSharingCalculator.hh
            │               └── Config.hh
            └── SteppingAction.hh
```

## Extension Points

### Adding New Reconstruction Methods

1. Add enum value in `Config.hh`:
   ```cpp
   enum class ReconMethod { LogA, LinA, DPC, NewMethod };
   ```

2. Implement reconstruction in `EventAction.cc`:
   ```cpp
   case ReconMethod::NewMethod:
       // Your algorithm
       break;
   ```

### Adding New Output Branches

1. Declare in `RunAction.hh`:
   ```cpp
   G4double fNewBranch;
   ```

2. Create branch in `RunAction::SetupBranches()`:
   ```cpp
   fTree->Branch("NewBranch", &fNewBranch);
   ```

3. Fill in `RunAction::FillTree()`:
   ```cpp
   fNewBranch = computedValue;
   ```

### Adding New Physics Parameters

1. Add to struct in `Config.hh`:
   ```cpp
   struct PhysicsParameters {
       G4double newParam = defaultValue;
   };
   ```

2. Add backward-compat alias:
   ```cpp
   namespace Constants {
       inline constexpr G4double NEW_PARAM = ...;
   }
   ```

3. Update metadata publishing in `RootIO.cc`
