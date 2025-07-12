#!/usr/bin/env python3
"""
Simulation Farmer
=================
This script reads a YAML control file (default: ``farm/control.yaml``) that
specifies which compile–time parameters should be varied for an AC-LGAD charge
sharing simulation.  For every Cartesian-product combination of the *varied
parameters* the farmer will:

1. Patch ``include/Constants.hh`` and ``include/Control.hh`` with the requested
   parameter values (falling back to *constant_parameters* when not varied).
2. Configure & build the C++ project using CMake, employing all available CPU
   cores (or the user-provided ``--jobs`` value).
3. Execute the resulting ``epicChargeSharing`` binary, passing it a macro file
   (default: ``macros/run.mac``).
4. Move the full build directory to a uniquely-named results folder inside the
   ``output_base_dir`` declared in the YAML (e.g. ``./pixel_size_study``).

The script restores the original header files after *every* combination so the
repository always returns to a clean state.
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover – helpful install hint
    sys.exit("PyYAML is required (pip install pyyaml).")

ROOT_DIR = Path(__file__).resolve().parent.parent  # project root
CONSTANTS_HEADER = ROOT_DIR / "include" / "Constants.hh"
CONTROL_HEADER = ROOT_DIR / "include" / "Control.hh"
DEFAULT_YAML = ROOT_DIR / "farm" / "control.yaml"

# ---------------------------------------------------------------------------
# YAML PARSING HELPERS
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return the resulting dictionary."""
    with path.open("r") as fp:
        return yaml.safe_load(fp)


def cartesian_product(varied: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list with one dict per Cartesian-product combination."""
    if not varied:
        return [{}]

    keys = list(varied.keys())
    values_pool = [varied[k]["values"] for k in keys]
    combos = []
    for combo in itertools.product(*values_pool):
        combos.append(dict(zip(keys, combo)))
    return combos

# ---------------------------------------------------------------------------
# HEADER PATCHING
# ---------------------------------------------------------------------------

_BOOL_PATTERN = re.compile(r"=\s*(true|false)")
_NUMERIC_PATTERN = re.compile(r"=\s*([0-9eE+\-\.]+)([^;]*);")


def _update_line(line: str, param: str, value: Any) -> str:
    """Replace the value of *param* in *line*; return the (possibly) new line."""
    if f" {param} " not in line:
        return line  # fast-path – param not on this line

    if "G4bool" in line:
        bool_val = "true" if str(value).lower() in ("true", "1", "yes") else "false"
        return _BOOL_PATTERN.sub(f"= {bool_val}", line)

    # Assume numeric otherwise (G4double / G4int …)
    m = _NUMERIC_PATTERN.search(line)
    if not m:
        return line  # pattern not as expected – leave untouched

    suffix = m.group(2)  # preserve unit multiplier (e.g. *mm)
    return _NUMERIC_PATTERN.sub(f"= {value}{suffix};", line, count=1)


def patch_header(header_path: Path, params: Dict[str, Any]) -> None:
    """Patch *header_path* in-place, updating any constants present in *params*."""
    lines = header_path.read_text().splitlines(keepends=False)
    new_lines = []
    for original in lines:
        updated = original
        for name, val in params.items():
            updated = _update_line(updated, name, val)
        new_lines.append(updated)
    header_path.write_text("\n".join(new_lines) + "\n")

# ---------------------------------------------------------------------------
# BUILD / RUN HELPERS
# ---------------------------------------------------------------------------

def cmake_configure(build_dir: Path) -> None:
    subprocess.run(["cmake", "-B", str(build_dir), "-S", str(ROOT_DIR)], check=True)


def cmake_build(build_dir: Path, jobs: int) -> None:
    subprocess.run(["cmake", "--build", str(build_dir), "--", f"-j{jobs}"], check=True)


def run_simulation(build_dir: Path, macro: Path) -> None:
    exe = build_dir / "epicChargeSharing"
    if not exe.is_file():
        raise FileNotFoundError(f"Simulation binary not found: {exe}")
    subprocess.run([str(exe), str(macro)], check=True)

# ---------------------------------------------------------------------------
# MAIN FARM LOGIC
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: C901 – top-level clarity
    parser = argparse.ArgumentParser(description="AC-LGAD simulation farmer")
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_YAML,
                        help="Path to YAML control file (default: farm/control.yaml)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                        help="CMake parallel build jobs (default: # logical cores)")
    parser.add_argument("--macro", type=Path, default=ROOT_DIR / "macros" / "run.mac",
                        help="Geant4 macro to pass to the simulation binary")
    parser.add_argument("--build-dir", type=Path, default=ROOT_DIR / "build",
                        help="Transient build directory used for each combination")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    sim_cfg: Dict[str, Any] = cfg.get("simulation", {})
    output_base_dir = Path(sim_cfg.get("output_base_dir", "./sim_output")).expanduser()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    varied_params: Dict[str, Any] = cfg.get("varied_parameters", {})
    constant_params: Dict[str, Any] = cfg.get("constant_parameters", {})

    combinations = cartesian_product(varied_params)
    print(f"📊  {len(combinations)} parameter combinations detected.")

    # Snapshot original headers so we can restore after each build
    orig_constants = CONSTANTS_HEADER.read_text()
    orig_control = CONTROL_HEADER.read_text()

    for idx, combo in enumerate(combinations, 1):
        print("\n────────────────────────────────────────────────────────")
        print(f"🚀  Combination {idx}/{len(combinations)} → {combo}")

        # Merge constants (combo has precedence over constant_params)
        run_params = constant_params.copy()
        run_params.update(combo)

        # Restore unmodified headers then patch
        CONSTANTS_HEADER.write_text(orig_constants)
        CONTROL_HEADER.write_text(orig_control)
        patch_header(CONSTANTS_HEADER, run_params)
        patch_header(CONTROL_HEADER, run_params)

        # Fresh build directory
        if args.build_dir.exists():
            shutil.rmtree(args.build_dir)
        cmake_configure(args.build_dir)
        cmake_build(args.build_dir, args.jobs)

        # Run simulation (Geant4 app is multi-threaded; max threads internal)
        run_simulation(args.build_dir, args.macro)

        # Results directory naming – join on "__" when multiple params
        label_parts = [f"{k}_{v}" for k, v in combo.items()]
        result_dir = output_base_dir / "__".join(label_parts)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Move the entire build directory for record-keeping
        shutil.move(str(args.build_dir), result_dir / "build")
        print(f"📦  Build + outputs moved → {result_dir.relative_to(Path.cwd())}")

    # Restore pristine headers at the end
    CONSTANTS_HEADER.write_text(orig_constants)
    CONTROL_HEADER.write_text(orig_control)
    print("\n✅  All simulations completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
