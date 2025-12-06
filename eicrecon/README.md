# chargeSharingRecon Plugin

EICrecon plugin that adapts the charge-sharing model for the JANA2 reconstruction chain. Reads `edm4hep::SimTrackerHit` collections, applies charge sharing, and outputs reconstructed `edm4eic::TrackerHit` positions.

## Build

```bash
cmake -S eicrecon -B build/eicrecon \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install
```

## Environment

```bash
export EICrecon_MY=$(pwd)/eicrecon/install
```

## Usage

```bash
eicrecon -Pplugins=chargeSharingRecon \
         -PChargeSharingRecon:readout=YourReadoutName \
         -PChargeSharingRecon:neighborhoodRadius=2 \
         input.edm4hep.root \
         -Ppodio:output=output.edm4eic.root
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `readout` | string | DD4hep readout name for segmentation lookup |
| `minEDep` | float | Energy deposition threshold (GeV) |
| `neighborhoodRadius` | int | Neighborhood half-width (2 = 5×5 grid) |
| `pixelSpacingMM` | float | Manual pixel pitch override (mm) |
| `pixelSizeMM` | float | Manual pixel size override (mm) |
| `ionizationEnergyEV` | float | Energy per e-h pair (eV) |
| `amplificationFactor` | float | Gain factor |
| `d0Micron` | float | Transverse hit size (µm) |
| `emitNeighborDiagnostics` | bool | Output diagnostic data |

Geometry values are auto-populated from DD4hep `CartesianGridXY` segmentation if available.

## Output

The factory emits `edm4eic::TrackerHit` with positions computed as charge-weighted centroids.
