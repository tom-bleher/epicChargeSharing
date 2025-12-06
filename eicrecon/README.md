# chargeSharingRecon Plugin

This directory contains an out-of-tree EICrecon plugin that adapts the
charge-sharing model from this repository into the JANA2 reconstruction
chain. The plugin reads `edm4hep::SimTrackerHit` collections, applies the local
charge sharing model, and publishes reconstructed `edm4eic::TrackerHit`
positions.

## Build

```
cmake -S plugins/eicrecon -B build/eicrecon \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/plugins/eicrecon/install
cmake --build build/eicrecon --target install
```

## Environment

Expose the installed plugin to EICrecon via

```
export EICrecon_MY=$(pwd)/plugins/eicrecon/install
```

## Running inside EICrecon

```
eicrecon -Pplugins=chargeSharingRecon \
         -PChargeSharingRecon:readout=YourReadoutName \
         -PChargeSharingRecon:neighborhoodRadius=2 \
         your_input.edm4hep.root \
         -Ppodio:output=charge_sharing_recon.edm4eic.root
```

### Key Parameters

- `readout` — DD4hep readout used to look up the segmentation (string). The
  factory queries `DD4hep_service` for this readout on configuration and
  auto-populates the pixel pitch, size, offsets, and index bounds when a
  `CartesianGridXY` segmentation is available.
- `minEDep` — energy deposition threshold in GeV (float)
- `neighborhoodRadius` — neighborhood half-width (2 ➔ 5×5 grid)
- `pixelSpacingMM`, `pixelSizeMM`, `pixelCornerOffsetMM` — optional manual
  overrides for the geometry (mm). Leave unset to use DD4hep-derived values.
- `pixelsPerSide` — optional override for the inferred pixel count (int)
- `ionizationEnergyEV`, `amplificationFactor`, `d0Micron` — physics knobs
- `emitNeighborDiagnostics` — emit per-neighbor diagnostic payloads (bool)

Parameters may be supplied either on the command line (`-PChargeSharingRecon:*`) or via
a JANA configuration file.

## Output

The factory emits an `edm4eic::TrackerHit` collection whose positions are the
charge-weighted centroid computed by the charge-sharing model. Additional
neighbor information is retained internally for future diagnostic factories.

