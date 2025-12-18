# chargeSharingRecon Plugin

EICrecon plugin that applies charge-sharing reconstruction to silicon tracker hits. Reads `edm4hep::SimTrackerHit` collections, applies a charge diffusion model, and outputs reconstructed `edm4eic::TrackerHit` positions with improved spatial resolution.

## Requirements

- EIC software stack (eic-shell)
- EICrecon with DD4hep, EDM4hep, EDM4eic support

## Build

Build inside the eic-shell environment:

```bash
# Enter eic-shell
./eic-shell

# Configure and build
cmake -S eicrecon -B build/eicrecon \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install
```

The plugin will be installed to `eicrecon/install/plugins/chargeSharingRecon.so`.

## Usage

### Set Environment

```bash
export EICrecon_MY=$(pwd)/eicrecon/install
```

### Run with EICrecon

Basic usage with default settings:

```bash
eicrecon -Pplugins=chargeSharingRecon \
         -Ppodio:output_file=output.edm4hep.root \
         input.edm4hep.root
```

### Full Reconstruction Chain Example

```bash
# 1. Run simulation with ddsim
source $EPIC_PATH/bin/thisepic.sh
ddsim --compactFile $DETECTOR_PATH/epic_craterlake.xml \
      --numberOfEvents 1000 \
      --enableGun \
      --gun.particle "e-" \
      --gun.energy "10*GeV" \
      --outputFile sim_output.edm4hep.root \
      --runType batch

# 2. Run reconstruction with charge sharing plugin
export EICrecon_MY=$(pwd)/eicrecon/install
eicrecon -Pplugins=chargeSharingRecon \
         -Ppodio:output_file=reco_output.edm4hep.root \
         -Pjana:nevents=1000 \
         sim_output.edm4hep.root
```

### Custom Parameters

Override default configuration:

```bash
eicrecon -Pplugins=chargeSharingRecon \
         -PChargeSharingRecon:neighborhoodRadius=3 \
         -PChargeSharingRecon:minEDep=0.0001 \
         -Ppodio:output_file=output.edm4hep.root \
         input.edm4hep.root
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `readout` | string | - | DD4hep readout name for segmentation lookup |
| `minEDep` | float | 0 | Energy deposition threshold (GeV) |
| `neighborhoodRadius` | int | 2 | Neighborhood half-width (2 = 5×5 grid) |
| `pixelSpacingMM` | float | - | Manual pixel pitch override (mm) |
| `pixelSizeMM` | float | - | Manual pixel size override (mm) |
| `ionizationEnergyEV` | float | 3.6 | Energy per e-h pair in silicon (eV) |
| `amplificationFactor` | float | 1.0 | Gain factor |
| `d0Micron` | float | 10 | Transverse charge cloud size (µm) |
| `emitNeighborDiagnostics` | bool | false | Output diagnostic neighbor data |

Geometry values are auto-populated from DD4hep `CartesianGridXY` segmentation when available.

## Input/Output Collections

| Direction | Collection Name | Type |
|-----------|-----------------|------|
| Input | `SiTrackerHits` (configurable) | `edm4hep::SimTrackerHit` |
| Output | `ChargeSharingTrackerHits` | `edm4eic::TrackerHit` |

## Algorithm

The plugin implements a charge-sharing model:

1. For each SimTrackerHit, identifies the hit pixel from DD4hep geometry
2. Calculates charge diffusion to neighboring pixels using a Gaussian model
3. Computes charge-weighted centroid position across the pixel neighborhood
4. Outputs TrackerHit with improved position estimate

## Troubleshooting

### Plugin not found

Ensure `EICrecon_MY` points to the install directory:
```bash
export EICrecon_MY=/path/to/eicrecon/install
ls $EICrecon_MY/plugins/chargeSharingRecon.so
```

### Parameter errors

Use relaxed parameter checking if needed:
```bash
eicrecon -Pjana:parameter_strictness=0 ...
```

### View available parameters

```bash
eicrecon -Pplugins=chargeSharingRecon -Pjana:show_all_parameters=1
```
