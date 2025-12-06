# Architecture

System architecture, component design, and data flow of epicChargeSharing.

## Overview

epicChargeSharing follows standard GEANT4 architecture with extensions for AC-LGAD charge sharing simulation:

1. **Simulation Core**: GEANT4 user action classes
2. **Charge Calculation**: Physics-based charge distribution
3. **I/O Layer**: ROOT file output and metadata
4. **Configuration**: Compile-time and runtime parameters

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │       epicChargeSharing.cc      │
                    │          (Main Entry)           │
                    └───────────────┬─────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ DetectorConstr. │      │ ActionInitializ.│      │   PhysicsList   │
│   (Geometry)    │      │ (Action Factory)│      │  (EM Physics)   │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│PrimaryGenerator │    │    RunAction    │    │   EventAction   │
│ (Particle Gun)  │    │  (ROOT I/O)     │    │(Event Process)  │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                              ┌─────────────────────────┴──────────┐
                              ▼                                    ▼
                   ┌─────────────────────┐              ┌─────────────────┐
                   │ChargeSharingCalc.   │              │  SteppingAction │
                   │  (Core Algorithm)   │              │  (Track Steps)  │
                   └─────────────────────┘              └─────────────────┘
```

---

## Component Descriptions

### DetectorConstruction

**Files**: `DetectorConstruction.hh/.cc`

Builds detector geometry:

| Element | Dimensions | Material |
|---------|------------|----------|
| World | 5 cm cube | Air |
| Detector | 30×30×0.05 mm | Silicon |
| Pixels | 0.1×0.1×0.001 mm | Aluminum |

Features: Configurable pixel grid, per-pixel gain noise, G4Messenger commands.

### ChargeSharingCalculator

**Files**: `ChargeSharingCalculator.hh/.cc`

Core computation engine implementing the Tornago charge sharing model:

1. Find nearest pixel to hit position
2. Define (2r+1)×(2r+1) neighborhood grid
3. Compute charge fractions F_i for each pixel
4. Apply noise models
5. Optionally compute full detector grid

**Key Structures**:
```cpp
struct PixelGridGeometry { nRows, nCols, pitchX, pitchY, x0, y0 };
struct HitInfo { trueX, trueY, trueZ, pixRow, pixCol };
struct ChargeMatrixSet { signalFraction, signalFractionRow, ... };
```

### EventAction

**Files**: `EventAction.hh/.cc`

Per-event processing:

```
BeginOfEventAction() → Reset accumulators
[GEANT4 tracking] → SteppingAction collects energy
EndOfEventAction() → Compute charges → Reconstruct position → Fill tree
```

### RunAction

**Files**: `RunAction.hh/.cc`

Run lifecycle management:
- ROOT file creation and TTree setup
- Worker thread synchronization
- Worker file merging
- Metadata publishing

### PrimaryGenerator

**Files**: `PrimaryGenerator.hh/.cc`

Particle gun with fixed or random position modes:

```
/ecs/gun/fixedPosition <bool>
/ecs/gun/fixedX <value>
/ecs/gun/fixedY <value>
```

---

## Data Flow

```
┌──────────────────┐
│  Primary Vertex  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Particle Tracking│
│ (Energy Deposits)│
└────────┬─────────┘
         ▼
┌──────────────────────────────────────┐
│      ChargeSharingCalculator         │
│  1. Find nearest pixel               │
│  2. Build neighborhood grid          │
│  3. Compute weights w_i (LogA/LinA)  │
│  4. Normalize: F_i = w_i / Σw_n      │
│  5. Apply noise                      │
└────────┬─────────────────────────────┘
         ▼
┌──────────────────┐
│Position Recon.   │
│(LogA/LinA/DPC)   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  ROOT TTree      │
└──────────────────┘
```

---

## Threading Model

```
┌─────────────────────────────────────┐
│         G4MTRunManager (Master)     │
└───────────────────┬─────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌────────┐    ┌────────┐      ┌────────┐
│Worker 0│    │Worker 1│ ...  │Worker N│
└────┬───┘    └────┬───┘      └────┬───┘
     ▼             ▼               ▼
worker_0.root  worker_1.root  worker_N.root
     └─────────────┼───────────────┘
                   ▼ (merge)
           epicChargeSharing.root
```

**Synchronization**: Atomic counters, mutexes, condition variables via `WorkerSync` class.

---

## Namespace Organization

```
ECS (Epic Charge Sharing)
├── Config
│   ├── SignalModel (LogA, LinA)
│   ├── ReconMethod (LogA, LinA, DPC)
│   ├── DenominatorMode (Neighborhood, ChargeBlock, RowCol)
│   └── Geometry/Physics/Noise structs
│
├── DetectorConstruction
├── ChargeSharingCalculator
├── EventAction, RunAction, SteppingAction
├── PrimaryGenerator
│
└── Utilities
    ├── NeighborhoodLayout
    ├── RootFileWriter, WorkerSync
    ├── BranchConfigurator, TreeFiller
    └── MetadataPublisher
```

---

## Extension Points

### Adding New Reconstruction Methods

1. Add enum in `Config.hh`:
   ```cpp
   enum class ReconMethod { LogA, LinA, DPC, NewMethod };
   ```

2. Implement in `EventAction.cc`:
   ```cpp
   case ReconMethod::NewMethod:
       // Algorithm
       break;
   ```

### Adding New Output Branches

1. Declare in `RunAction.hh`:
   ```cpp
   G4double fNewBranch;
   ```

2. Create branch:
   ```cpp
   fTree->Branch("NewBranch", &fNewBranch);
   ```

3. Fill in `RunAction::FillTree()`.

### Adding New Physics Parameters

1. Add to struct in `Config.hh`
2. Add backward-compat alias in `Constants` namespace
3. Update metadata publishing in `RootIO.cc`
