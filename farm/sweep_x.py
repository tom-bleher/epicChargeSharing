#!/usr/bin/env python3
"""
Sweep particle gun X position across gap regions between pixels.

This script runs simulations at multiple X positions without recompiling,
using Geant4 macro commands to set the particle gun position at runtime.

The simulation is built once, then for each position:
1. A temporary macro file is generated with the position commands
2. The simulation is run with that macro
3. The output ROOT file is collected

This is much faster than the old approach of modifying source code and
recompiling for each position.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
# Resolve repository root relative to this script so the tool works anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
EXECUTABLE = BUILD_DIR / "epicChargeSharing"
BASE_MAC = REPO_ROOT / "macros" / "run.mac"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "sweep_x_runs"

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Pixel pitch = 500 µm (pixels centered at ..., -500, 0, +500, ... µm)
# Metal pad size = 150 µm (pads span ±75 µm from pixel center)
#
# For 3 pixels centered at -500, 0, +500 µm:
#   Metal pads at: [-575,-425], [-75,+75], [+425,+575] µm
#   Gap regions:   [-425,-75] and [+75,+425] µm
#
# We want positions that:
#   1. Stay in the gap regions (avoid hitting metal pads)
#   2. Are equally spaced
#   3. Get close to pixel edges from left and right

PIXEL_PITCH_UM = 500.0      # Pixel spacing in µm
METAL_PAD_SIZE_UM = 150.0   # Metal pad size in µm
HALF_PAD_UM = METAL_PAD_SIZE_UM / 2.0  # 75 µm
EDGE_MARGIN_UM = 3.0        # Margin from pad edge to avoid touching
STEP_SIZE_UM = 25.0         # Step size between positions

def calculate_gap_positions(
    pixel_pitch: float = PIXEL_PITCH_UM,
    half_pad: float = HALF_PAD_UM,
    edge_margin: float = EDGE_MARGIN_UM,
    step_size: float = STEP_SIZE_UM,
    n_pixels: int = 3,
    include_edges: bool = True,
) -> list:
    """Calculate scan positions in the gap regions between metal pads.

    Args:
        pixel_pitch: Distance between pixel centers in µm (default: 500)
        half_pad: Half of metal pad size in µm (default: 75)
        edge_margin: Margin from pad edge in µm (default: 3)
        step_size: Step size between positions in µm (default: 25)
        n_pixels: Number of pixels to span (default: 3)
        include_edges: Ensure both gap edges are included even if step doesn't align (default: True)

    Returns:
        List of positions in µm, sorted from most negative to most positive

    Example for 3 pixels with default geometry:
        Pixels at: -500, 0, +500 µm
        Pads at: [-575,-425], [-75,+75], [+425,+575] µm
        Gap 1: [-425+margin, -75-margin] = [-422, -78] µm
        Gap 2: [+75+margin, +425-margin] = [+78, +422] µm

    The positions are calculated to:
    1. Stay within gap regions (avoid metal pads)
    2. Be equally spaced within each gap
    3. Include positions very close to both left and right edges of each gap
    """
    positions = set()  # Use set to avoid duplicates

    # Calculate pixel centers (symmetric around 0)
    # For n_pixels=3: centers at -500, 0, +500 µm
    # For n_pixels=5: centers at -1000, -500, 0, +500, +1000 µm
    half_span = (n_pixels - 1) // 2
    pixel_centers = [i * pixel_pitch for i in range(-half_span, half_span + 1)]

    # Generate positions in each gap between adjacent pixels
    for i in range(len(pixel_centers) - 1):
        left_pixel = pixel_centers[i]
        right_pixel = pixel_centers[i + 1]

        # Gap boundaries (with margin from pad edges)
        gap_start = left_pixel + half_pad + edge_margin   # Right edge of left pad + margin
        gap_end = right_pixel - half_pad - edge_margin     # Left edge of right pad - margin

        # Always include edge positions if requested
        if include_edges:
            positions.add(round(gap_start, 1))
            positions.add(round(gap_end, 1))

        # Generate equally-spaced interior positions
        pos = gap_start + step_size
        while pos < gap_end - 0.1:  # Don't include gap_end (already added if include_edges)
            positions.add(round(pos, 1))
            pos += step_size

    return sorted(positions)

# Default positions: gap regions between 3 pixels, avoiding metal pads
POSITIONS = calculate_gap_positions()


def determine_output_dir(output_dir_arg: Optional[str]) -> Path:
    if output_dir_arg:
        candidate = Path(output_dir_arg)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = (DEFAULT_OUTPUT_BASE / timestamp).resolve()

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def read_file_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def generate_position_macro(x_um: float, n_events: int = 10000) -> str:
    """Generate a Geant4 macro that sets the particle gun to a fixed X position.

    Uses the /epic/gun/ messenger commands defined in PrimaryGenerator.cc:
    - /epic/gun/useFixedPosition true  (enable fixed position mode)
    - /epic/gun/fixedX <value> mm      (set X coordinate)
    - /epic/gun/fixedY 0 mm            (set Y to center of pad row)

    Args:
        x_um: X position in micrometers
        n_events: Number of events to simulate

    Returns:
        Macro file contents as a string
    """
    x_mm = x_um / 1000.0  # Convert µm to mm

    return f"""# Auto-generated macro for sweep position x = {x_um:.1f} µm
# Set verbosity
/control/verbose 0
/run/verbose 0
/event/verbose 0
/tracking/verbose 0

# Initialize
/run/initialize

# Set particle gun parameters
/gun/particle e-
/gun/energy 10 GeV

# Set fixed position mode with X = {x_um:.1f} µm ({x_mm:.4f} mm)
/epic/gun/useFixedPosition true
/epic/gun/fixedX {x_mm:.4f} mm
/epic/gun/fixedY 0 mm

# Run events
/run/beamOn {n_events}
"""


def run_cmd(cmd, cwd: Path = None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def configure_build_if_needed():
    BUILD_DIR.mkdir(exist_ok=True)
    if not (BUILD_DIR / "Makefile").exists():
        run_cmd(["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)])


def build_project():
    run_cmd(["cmake", "--build", str(BUILD_DIR), "-j"])


def run_simulation(macro_path: Path):
    """Run the simulation with the specified macro file."""
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Executable not found: {EXECUTABLE}")
    if not macro_path.exists():
        raise RuntimeError(f"Macro file not found: {macro_path}")
    # Multithreaded by default (let executable decide threads)
    run_cmd([str(EXECUTABLE), "-m", str(macro_path)])


def wait_for_file(path: Path, timeout_s: int = 120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(1)
    return path.exists()


def wait_for_root_close(path: Path, timeout_s: int = 60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            try:
                with open(path, "ab"):
                    return
            except Exception:
                time.sleep(0.5)
        else:
            time.sleep(0.5)


def find_output_root() -> Path:
    # After MT run, final merged file is epicChargeSharing.root in working dir
    candidates = [
        REPO_ROOT / "epicChargeSharing.root",
        BUILD_DIR / "epicChargeSharing.root",
    ]
    # Wait for merged file to appear
    for c in candidates:
        if wait_for_file(c, timeout_s=180):
            return c
    raise RuntimeError("Expected merged ROOT output file not found after run.")


def rename_output_to(target_name: Path):
    source = find_output_root()
    wait_for_root_close(source)
    target_name.parent.mkdir(parents=True, exist_ok=True)
    if target_name.exists():
        target_name.unlink()
    shutil.move(str(source), str(target_name))

def run_post_analysis(output_dir: Path):
    """Run post-sweep analysis scripts (Fi_x.py and sigma_f_x.py)."""
    print("\n" + "=" * 80)
    print("RUNNING POST-SWEEP ANALYSIS")
    print("=" * 80)

    # Ensure repo root is in sys.path for 'from farm import ...' to work
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # Import and run Fi_x.py
    try:
        print("\n[INFO] Running Fi_x.py analysis...")
        from farm import Fi_x
        Fi_x.main(["--input-dir", str(output_dir)])
        print("[OK] Fi_x.py completed successfully")
    except Exception as e:
        print(f"[WARN] Fi_x.py failed: {e}")

    # Import and run sigma_f_x.py
    try:
        print("\n[INFO] Running sigma_f_x.py analysis...")
        from farm import sigma_f_x
        sigma_f_x.main(output_dir)
        print("[OK] sigma_f_x.py completed successfully")
    except Exception as e:
        print(f"[WARN] sigma_f_x.py failed: {e}")

    print("\n" + "=" * 80)
    print("POST-SWEEP ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep x positions and collect output ROOT files into a dedicated directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Directory to store output ROOT files. If relative, resolved against the repository root. Default: timestamped folder under sweep_x_runs/.",
    )
    parser.add_argument(
        "--skip-analysis",
        dest="skip_analysis",
        action="store_true",
        help="Skip running post-sweep analysis scripts (Fi_x.py, sigma_f_x.py).",
    )
    parser.add_argument(
        "--n-events",
        dest="n_events",
        type=int,
        default=10000,
        help="Number of events per position.",
    )
    parser.add_argument(
        "--skip-build",
        dest="skip_build",
        action="store_true",
        help="Skip building (assume executable is already up to date).",
    )
    args = parser.parse_args()

    os.chdir(str(REPO_ROOT))

    output_dir = determine_output_dir(args.output_dir)
    print(f"Writing output ROOT files to {output_dir}")
    print(f"Positions to sweep: {len(POSITIONS)} points")
    print(f"Events per position: {args.n_events}")

    # Build once at the start (no recompilation needed for position changes!)
    if not args.skip_build:
        print("\n" + "=" * 80)
        print("BUILDING PROJECT (once)")
        print("=" * 80)
        configure_build_if_needed()
        build_project()
    else:
        print("\n[INFO] Skipping build (--skip-build specified)")

    # Create a temporary directory for macro files
    macro_dir = output_dir / "macros"
    macro_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)
    print("Using Geant4 macro commands to set position (no recompilation needed)")

    for i, x_um in enumerate(POSITIONS, 1):
        out_name = output_dir / (
            f"{int(x_um)}um.root" if float(x_um).is_integer() else f"{x_um}um.root"
        )

        print(f"\n=== [{i}/{len(POSITIONS)}] Position x = {x_um} µm ===")

        # Generate position-specific macro
        macro_content = generate_position_macro(x_um, args.n_events)
        macro_path = macro_dir / f"run_x{int(x_um) if float(x_um).is_integer() else x_um}um.mac"
        write_file_text(macro_path, macro_content)

        # Run simulation with macro
        run_simulation(macro_path)

        # Rename/move output (merged file)
        rename_output_to(out_name)

    # Run post-analysis scripts
    if not args.skip_analysis:
        run_post_analysis(output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
