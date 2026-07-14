# epicChargeSharing

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL--3.0--or--later-blue.svg)](LICENSE)

AC-LGAD charge-sharing reconstruction for the ePIC detector. This repository ships two coordinated artifacts:

1. A **prototype out-of-tree EICrecon plugin suite** (`eicrecon/`) that plugs into the ePIC reconstruction pipeline and provides `SimTrackerHit -> TrackerHit (+ MC association) -> Measurement2D` for the **B0 tracker** and **Luminosity Spectrometer**. This is what you load with `eicrecon -Pplugins=...`.
2. A **standalone Geant4 validation harness** (`epicChargeSharing.cc` + `src/` + `include/`) that exercises the shared physics library on a parametric pad grid, independent of DD4hep geometry. This is *not* an ePIC simulator; production simulation is done with `ddsim` / `npsim` on the ePIC compact XML.

Both paths consume the same compiled physics library under `core/` (LogA charge sharing, Gaussian position fitting, noise injection). That way the plugin and the harness are guaranteed to stay numerically consistent.

> **Status: prototype / R&D.** The plugin builds and runs, but its interfaces
> and estimators are being reworked toward upstream EICrecon's staged
> digitization (`SiliconChargeSharing` -> pulse chain -> clustering) before any
> upstreaming. Known open items: per-pad shared hits are not yet emitted as
> separate EDM objects, local/global frame handling on rotated modules, and
> the Gaussian2D estimator defects marked `[!mayfail]` in the unit tests. Do
> not use its output for physics conclusions yet.

## EICrecon plugin

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
├── core/                     # Compiled physics library + Gaussian fitter (shared)
│   ├── include/chargesharing/{core,fit}/*.hh
│   └── src/
├── eicrecon/                 # Out-of-tree EICrecon plugin suite (prototype)
│   ├── cmake/                # Shim locating upstream's jana_plugin.cmake macros
│   ├── src/algorithms/       # LGADChargeSharingRecon + LGADGaussianClustering
│   ├── src/factories/        # JOmniFactory wrappers
│   ├── src/detectors/        # B0TRK/ + LumiSpec/ plugin libraries
│   ├── src/benchmarks/       # LGADChargeSharingMonitor JEventProcessor
│   ├── src/tests/            # Catch2 unit tests
│   └── test/                 # End-to-end ddsim/eicrecon test scripts
├── include/ + src/ + epicChargeSharing.cc   # Standalone Geant4 validation harness
├── macros/                   # Geant4 macros for the harness
└── analysis/                 # Python/ROOT analysis: sweep/, fitting/, viz/, diagnostics/
```

## Documentation

- [eicrecon/README.md](eicrecon/README.md) — plugin configuration, collection names, algorithm details
- Reference: M. Tornago et al., [Nucl. Instrum. Meth. A 1003 (2021) 165319](https://doi.org/10.1016/j.nima.2021.165319) ([arXiv:2007.09528](https://arxiv.org/abs/2007.09528))

## Development

```bash
# Unit tests for the plugin (inside eic-shell)
cmake -S eicrecon -B build/eicrecon -DBUILD_TESTING=ON
cmake --build build/eicrecon
ctest --test-dir build/eicrecon --output-on-failure

# Static analysis / formatting (config matches eic/EICrecon: LLVM style, 100 col)
make tidy
make format
```

Two unit tests documenting known Gaussian2D estimator defects are tagged
`[!mayfail]`: they run and report, but do not fail the suite. They will be
re-enabled as strict once the estimator rework lands.

Python analysis scripts require the packages in `requirements.txt` (`pip install -r requirements.txt`).

## Continuous integration

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) builds
both artifacts inside the `eic_xl` container (CVMFS +
`eic/run-cvmfs-osg-eic-shell`, the same recipe `eic/epic` and `eic/EICrecon`
use) and runs the plugin's Catch2 suite. A `clang-format` gate is prepared but
disabled until a one-time whole-tree reformat commit is made.

## Contributing

- Follow the code style: PascalCase classes, camelCase methods, `m_snake_case` members, `snake_case` locals, `SCREAMING_SNAKE_CASE` constants; `.clang-format` / `.clang-tidy` are imported from `eic/EICrecon`.
- Every source file must carry an SPDX `LGPL-3.0-or-later` header (exception: the derived generator macros under `eicrecon/test/lumi/`, which carry upstream attribution instead).
- `make format` before committing.

Workflow: fork, branch, run `make tidy` + `make format-check`, run the relevant test suite, open a PR against `main`.

## Citation

Metadata lives in [CITATION.cff](CITATION.cff). GitHub renders a formatted citation via the "Cite this repository" button.

## Contact

Email [tombleher@tauex.tau.ac.il](mailto:tombleher@tauex.tau.ac.il) or open a [GitHub Issue](https://github.com/tom-bleher/epicChargeSharing/issues).
