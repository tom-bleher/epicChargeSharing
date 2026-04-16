# epicChargeSharing

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL--3.0--or--later-blue.svg)](LICENSE)

AC-LGAD charge-sharing reconstruction for the ePIC detector. This repository ships two coordinated artifacts:

1. A **production EICrecon plugin suite** (`eicrecon/`) that plugs into the ePIC reconstruction pipeline and provides `SimTrackerHit -> TrackerHit (+ MC association) -> Measurement2D` for the **B0 tracker** and **Luminosity Spectrometer**. This is what you load with `eicrecon -Pplugins=...`.
2. A **standalone Geant4 validation harness** (`epicChargeSharing.cc` + `src/` + `include/`) that exercises the shared physics library on a parametric pad grid, independent of DD4hep geometry. This is *not* an ePIC simulator; production simulation is done with `ddsim` / `npsim` on the ePIC compact XML.

Both paths consume the same header-only physics core under `core/` (LogA / LinA charge-sharing models, Gaussian position fitting, noise injection). That way the plugin and the harness are guaranteed to stay numerically consistent.

## EICrecon plugin (production path)

The plugin suite installs three `.so` libraries under `eicrecon/install/plugins/`:

| Plugin | Role |
|--------|------|
| `B0TRK_lgad_chargesharing.so` | B0 tracker: charge sharing + Gaussian clustering |
| `LumiSpec_lgad_chargesharing.so` | LumiSpec tracker: charge sharing only |
| `LGAD_chargesharing_benchmark.so` | Truth-residual histograms + TTree into `-Phistsfile=...` |

### Build

```bash
./eic-shell
cmake -S eicrecon -B build/eicrecon \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/eicrecon/install
cmake --build build/eicrecon --target install
```

### Run

```bash
export EICrecon_MY=$(pwd)/eicrecon/install
eicrecon \
    -Pplugins=B0TRK_lgad_chargesharing,LumiSpec_lgad_chargesharing,LGAD_chargesharing_benchmark \
    -Pjana:plugin_path=$EICrecon_MY/plugins \
    -Phistsfile=lgad_hists.root \
    -Ppodio:output_file=reco_output.edm4hep.root \
    sim_output.edm4hep.root
```

See [eicrecon/README.md](eicrecon/README.md) for the full per-parameter configuration table, per-detector output collection names, algorithm description, and benchmark TTree schema.

### Inputs and outputs

| Detector | Input `SimTrackerHit` | Output `TrackerHit` | Output `Measurement2D` |
|----------|------------------------|---------------------|-------------------------|
| B0TRK | `B0TrackerHits` | `B0TrackerChargeSharingHits` (+ `B0TrackerChargeSharingHitAssociations`) | `B0TrackerClusterHits` |
| LumiSpec | `LumiSpecTrackerHits` | `LumiSpecTrackerChargeSharingHits` (+ `LumiSpecTrackerChargeSharingHitAssociations`) | *(not registered; add in `src/detectors/LumiSpec/` when segmentation is ready)* |

## Standalone validation harness

The standalone harness is a Geant4 application that drives the same `core/` physics on a parametric rectangular pad grid so you can cross-check plugin outputs against a simplified reference.

Important: The standalone harness is **not** an ePIC simulator. For production simulation, use `ddsim` / `npsim` on `eic/epic`.

### Quick start

```bash
git clone https://github.com/tom-bleher/epicChargeSharing.git
cd epicChargeSharing
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./epicChargeSharing -m ../macros/run.mac
```

### Output

The harness produces `epicChargeSharing.root` containing:
- `Hits` TTree with per-event hit data, charge fractions, and reconstructed positions
- Typed run metadata in `Hits->GetUserInfo()` (no string parsing required)

EDM4hep output in the standalone harness is **off by default** (`-DWITH_EDM4HEP=OFF`). It can be re-enabled for cross-format validation, but the resulting CellIDs use a simplified `system:8|layer:4|x:16|y:16` encoding that is **not** wire-compatible with `ddsim` output. Never feed harness EDM4hep files into `eicrecon` with this plugin loaded -- use the plugin directly on `ddsim` output as described above.

### Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| GEANT4 | 11.0+ | Harness Monte Carlo |
| ROOT | 6.20+ | Data output and analysis |
| Eigen3 | 3.3+ | Fit matrix operations |
| CMake | 3.24+ | Build system |
| C++ Compiler | C++20 | GCC 10+, Clang 10+, or MSVC 2019+ |

EDM4hep / podio are only needed when `-DWITH_EDM4HEP=ON`.

## Repository layout

```
epicChargeSharing/
├── core/                     # Header-only physics + Gaussian fitter (shared)
│   └── include/chargesharing/{core,fit}/*.hh
├── eicrecon/                 # Production EICrecon plugin suite
│   ├── src/algorithms/       # LGADChargeSharingRecon + LGADGaussianClustering
│   ├── src/factories/        # JOmniFactory wrappers
│   ├── src/detectors/        # B0TRK/ + LumiSpec/ plugin libraries
│   ├── src/benchmarks/       # LGADChargeSharingMonitor JEventProcessor
│   └── src/tests/            # Catch2 unit tests
├── include/ + src/ + epicChargeSharing.cc   # Standalone Geant4 validation harness
├── macros/                   # Geant4 macros for the harness
├── farm/                     # Python sweep tooling
└── proc/                     # ROOT-based analysis GUIs
```

## Documentation

- [eicrecon/README.md](eicrecon/README.md) — plugin configuration, collection names, algorithm details
- [Full docs](https://tom-bleher.github.io/epicChargeSharing/) — physics, CI, reference
- Reference: M. Tornago et al., [Nucl. Instrum. Meth. A 1003 (2021) 165319](https://doi.org/10.1016/j.nima.2021.165319)

## Development

```bash
# Unit tests for the plugin (inside eic-shell)
cmake -S eicrecon -B build/eicrecon -DBUILD_TESTING=ON
cmake --build build/eicrecon
ctest --test-dir build/eicrecon --output-on-failure

# Unit tests for the core physics (host)
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Static analysis / formatting
make tidy
make format
```

Python analysis scripts require the packages in `requirements.txt` (`pip install -r requirements.txt`).

## Contributing

- Follow the code style: PascalCase classes, camelCase methods, `m_snake_case` members, `snake_case` locals, `SCREAMING_SNAKE_CASE` constants.
- Every source file must carry an SPDX `LGPL-3.0-or-later` header.
- `make format` before committing; `make format-check` is enforced in CI.

Workflow: fork, branch, run `make tidy` + `make format-check`, run the relevant test suite, open a PR against `main`.

## Citation

Metadata lives in [CITATION.cff](CITATION.cff). GitHub renders a formatted citation via the "Cite this repository" button.

## Contact

Email [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il) or open a [GitHub Issue](https://github.com/tom-bleher/epicChargeSharing/issues).
