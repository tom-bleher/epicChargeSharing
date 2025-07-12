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
4. Move the final ROOT file and other essential artifacts to a uniquely-named 
   results folder inside the ``output_base_dir`` declared in the YAML (e.g. ``./pixel_size_study``).

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
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

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


def run_simulation(build_dir: Path, macro: Path, num_events: Optional[int] = None) -> None:
    exe = build_dir / "epicChargeSharing"
    if not exe.is_file():
        raise FileNotFoundError(f"Simulation binary not found: {exe}")
    
    # Run simulation from build directory so ROOT file is created there
    # Convert to absolute path for working directory change
    exe_abs = exe.resolve()
    
    if num_events is not None:
        # Use header mode when num_events is specified
        # This eliminates the need for macro files entirely
        print(f"Running {num_events} events using header mode")
        subprocess.run([str(exe_abs), "-header"], cwd=str(build_dir), check=True)
    else:
        # Traditional macro file mode (for compatibility)
        macro_abs = macro.resolve()
        print(f"Running simulation with macro: {macro}")
        subprocess.run([str(exe_abs), "-m", str(macro_abs)], cwd=str(build_dir), check=True)

def wait_for_file_merge_completion(build_dir: Path, timeout_seconds: int = 30) -> bool:
    """
    Wait for Geant4 multithreading to complete file merging.
    
    In multithreaded Geant4:
    1. Worker threads create epicChargeSharingOutput_t*.root files  
    2. Master thread merges VALID files into epicChargeSharingOutput.root
    3. Valid worker files are deleted, but INVALID/EMPTY files may remain
    
    We only need to wait for the final merged file to exist and be valid.
    Some empty worker files may be left behind intentionally.
    
    Returns True if final merged file exists and is valid, False on timeout.
    """
    final_file = build_dir / "epicChargeSharingOutput.root"
    start_time = time.time()
    
    print(f"Waiting for final merged file: {final_file.name}")
    
    while time.time() - start_time < timeout_seconds:
        if final_file.exists():
            # Check if the file is valid (has content)
            try:
                file_size = final_file.stat().st_size
                if file_size > 1024:  # At least 1KB - indicates real content
                    print(f"Final merged file created: {final_file.name} ({file_size} bytes)")
                    
                    # Give a moment for any final cleanup
                    time.sleep(2)
                    return True
                else:
                    print(f"Final file exists but too small ({file_size} bytes), waiting...")
            except Exception as e:
                print(f"Error checking file: {e}, waiting...")
        
        time.sleep(1)
    
    print(f"Timeout waiting for valid final merged file")
    return False


def move_simulation_results(build_dir: Path, result_dir: Path) -> None:
    """
    Move only the essential simulation results to the result directory.
    
    For multithreaded Geant4, we expect:
    - epicChargeSharingOutput.root (final merged file)
    - Log files and other build artifacts
    
    We explicitly avoid moving intermediate worker thread files that should 
    have been cleaned up automatically.
    """
    print(f"Files in build directory before moving:")
    build_files = list(build_dir.iterdir())
    for item in build_files:
        print(f"     {item.name}")
    
    # Check for the final merged ROOT file
    final_root_file = build_dir / "epicChargeSharingOutput.root"
    worker_files = [f for f in build_dir.glob("epicChargeSharingOutput_t*.root")]
    
    if final_root_file.exists():
        print(f"Found final merged ROOT file: {final_root_file.name}")
    else:
        print(f"ERROR: Final merged ROOT file not found!")
        
        # Check if we have worker files that weren't merged
        if worker_files:
            print(f"Found {len(worker_files)} unmerged worker files: {[f.name for f in worker_files]}")
            print("This suggests the multithreaded merge process failed")
            
            # Fallback: Check if ROOT file was created in project root (old behavior)
            project_root_file = ROOT_DIR / "epicChargeSharingOutput.root"
            if project_root_file.exists():
                print(f"🔍  Found ROOT file in project root, moving it to build directory")
                shutil.move(str(project_root_file), str(final_root_file))
            else:
                raise FileNotFoundError("No ROOT output file found in build directory or project root")
    
    if worker_files:
        print(f"Found {len(worker_files)} remaining worker files")
        print("These are likely empty/invalid files that weren't merged (normal behavior)")
        print("Moving only the final merged file")
    
    # Define which files to move (only essential results, not intermediate files)
    files_to_move = []
    
    # Always move the final ROOT file if it exists
    if final_root_file.exists():
        files_to_move.append(final_root_file)
    
    # Move log files and other essential build artifacts (but not worker ROOT files)
    for item in build_dir.iterdir():
        if item.is_file():
            # Move log files, binaries, but skip intermediate ROOT files
            if (item.suffix in ['.log', '.txt'] or 
                item.name == 'epicChargeSharing' or
                (item.suffix == '.root' and not item.name.startswith('epicChargeSharingOutput_t'))):
                if item != final_root_file:  # Don't add twice
                    files_to_move.append(item)
        elif item.is_dir():
            # Move subdirectories (like logs/ if they exist)
            files_to_move.append(item)
    
    # Move selected files to result directory
    print(f"Moving {len(files_to_move)} essential files to result directory:")
    for item in files_to_move:
        dest_path = result_dir / item.name
        if dest_path.exists():
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()
        shutil.move(str(item), str(dest_path))
        print(f"     Moved: {item.name}")
    
    # Clean up any remaining intermediate files in build directory
    remaining_worker_files = [f for f in build_dir.glob("epicChargeSharingOutput_t*.root")]
    if remaining_worker_files:
        print(f"Cleaning up {len(remaining_worker_files)} leftover worker files:")
        for worker_file in remaining_worker_files:
            try:
                worker_file.unlink()
                print(f"     Deleted: {worker_file.name}")
            except Exception as e:
                print(f"     Failed to delete {worker_file.name}: {e}")
    
    # Verify final result
    result_root_files = [f for f in result_dir.iterdir() if f.suffix == '.root']
    if result_root_files:
        print(f"Final ROOT file in result directory: {[f.name for f in result_root_files]}")
    else:
        raise FileNotFoundError("ERROR: No ROOT file found in result directory after move!")


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
    print(f"{len(combinations)} parameter combinations detected.")

    # Snapshot original headers so we can restore after each build
    orig_constants = CONSTANTS_HEADER.read_text()
    orig_control = CONTROL_HEADER.read_text()

    for idx, combo in enumerate(combinations, 1):
        print("\n────────────────────────────────────────────────────────")
        print(f"Combination {idx}/{len(combinations)} → {combo}")

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
        num_events = run_params.get("NUMBER_OF_EVENTS")
        run_simulation(args.build_dir, args.macro, num_events=num_events)

        # Wait for multithreaded file merging to complete
        print("Waiting for multithreaded file merging to complete...")
        if not wait_for_file_merge_completion(args.build_dir):
            print("Continuing despite merge timeout - files may need manual inspection")

        # Results directory naming – join on "__" when multiple params
        label_parts = [f"{k}_{v}" for k, v in combo.items()]
        result_dir = output_base_dir / "__".join(label_parts)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Move only essential simulation results (not intermediate worker files)
        move_simulation_results(args.build_dir, result_dir)
        
        # Remove the build directory (may contain leftover files)
        if args.build_dir.exists():
            try:
                shutil.rmtree(args.build_dir)
                print(f"Cleaned up build directory")
            except Exception as e:
                print(f"Could not remove build directory: {e}")
                print(f"(This is not critical - results are safely stored)")
        
        print(f"Results saved to → {result_dir.resolve().relative_to(Path.cwd())}")

    # Restore pristine headers at the end
    CONSTANTS_HEADER.write_text(orig_constants)
    CONTROL_HEADER.write_text(orig_control)
    print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
