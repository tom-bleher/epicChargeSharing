#!/usr/bin/env python3
"""
Run sweep_x.py with multiple fit configuration variants.

Configurations:
1. 1D LogA with 5% max vertical uncertainty
2. 2D LogA with 5% max vertical uncertainty
3. 1D without vertical uncertainty
4. 2D without vertical uncertainty
5. 1D with distance-based vertical uncertainty
6. 2D with distance-based vertical uncertainty
7. 1D with inverse distance vertical uncertainty
8. 2D with inverse distance vertical uncertainty

Each configuration modifies Config.hh, rebuilds, and runs the full sweep.
Results are saved to separate timestamped directories.

After all runs, generates comparison plots overlaying 1D vs 2D:
- 5% max vertical uncertainty
- No vertical uncertainty
- Distance-based vertical uncertainty
- Inverse distance vertical uncertainty
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "include" / "Config.hh"
SWEEP_SCRIPT = REPO_ROOT / "farm" / "sweep_x.py"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "sweep_fit_configs"


class FitConfig(NamedTuple):
    """Configuration for a single sweep run."""
    name: str                           # Short name for output directory
    description: str                    # Human-readable description
    fit_gaus_1d: bool                   # FIT_GAUS_1D setting
    fit_gaus_2d: bool                   # FIT_GAUS_2D setting
    use_vertical_uncertainties: bool   # FIT_USE_VERTICAL_UNCERTAINTIES
    use_distance_weighted: bool        # FIT_USE_DISTANCE_WEIGHTED_ERRORS
    distance_power_inverse: bool = True # FIT_DISTANCE_POWER_INVERSE (inverse vs direct)
    error_percent: float = 5.0         # FIT_ERROR_PERCENT_OF_MAX


# Define all configurations to run
CONFIGURATIONS = [
    #FitConfig(
        #name="1D_LogA_5pct_vert",
        #description="1D LogA with 5% max vertical uncertainty",
        #fit_gaus_1d=True,
        #fit_gaus_2d=False,
        #use_vertical_uncertainties=True,
        #use_distance_weighted=False,
        #error_percent=5.0,
    #),
    #FitConfig(
        #name="2D_LogA_5pct_vert",
        #description="2D LogA with 5% max vertical uncertainty",
        #fit_gaus_1d=False,
        #fit_gaus_2d=True,
        #use_vertical_uncertainties=True,
        #use_distance_weighted=False,
        #error_percent=5.0,
    #),
    #FitConfig(
        #name="1D_LogA_no_vert",
        #description="1D LogA without vertical uncertainty",
        #fit_gaus_1d=True,
        #fit_gaus_2d=False,
        #use_vertical_uncertainties=False,
        #use_distance_weighted=False,
    #),
    #FitConfig(
        #name="2D_LogA_no_vert",
        #description="2D LogA without vertical uncertainty",
        #fit_gaus_1d=False,
        #fit_gaus_2d=True,
        #use_vertical_uncertainties=False,
        #use_distance_weighted=False,
    #),
    FitConfig(
        name="1D_LogA_dist_vert",
        description="1D LogA with distance-based vertical uncertainty",
        fit_gaus_1d=True,
        fit_gaus_2d=False,
        use_vertical_uncertainties=True,
        use_distance_weighted=True,
        distance_power_inverse=False,
    ),
    FitConfig(
        name="2D_LogA_dist_vert",
        description="2D LogA with distance-based vertical uncertainty",
        fit_gaus_1d=False,
        fit_gaus_2d=True,
        use_vertical_uncertainties=True,
        use_distance_weighted=True,
        distance_power_inverse=False,
    ),
    FitConfig(
        name="1D_LogA_inv_dist_vert",
        description="1D LogA with inverse distance vertical uncertainty",
        fit_gaus_1d=True,
        fit_gaus_2d=False,
        use_vertical_uncertainties=True,
        use_distance_weighted=True,
        distance_power_inverse=True,
    ),
    FitConfig(
        name="2D_LogA_inv_dist_vert",
        description="2D LogA with inverse distance vertical uncertainty",
        fit_gaus_1d=False,
        fit_gaus_2d=True,
        use_vertical_uncertainties=True,
        use_distance_weighted=True,
        distance_power_inverse=True,
    ),
]

# Comparison groups: (plot_name, title, 1D_config_name, 2D_config_name)
COMPARISON_GROUPS = [
    #("5pct_vert", "5% Max Vertical Uncertainty", "1D_LogA_5pct_vert", "2D_LogA_5pct_vert"),
    #("no_vert", "No Vertical Uncertainty", "1D_LogA_no_vert", "2D_LogA_no_vert"),
    ("dist_vert", "Distance-Based Vertical Uncertainty", "1D_LogA_dist_vert", "2D_LogA_dist_vert"),
    ("inv_dist_vert", "Inverse Distance Vertical Uncertainty", "1D_LogA_inv_dist_vert", "2D_LogA_inv_dist_vert"),
]

# Branch to use for sigma_f comparison (primary reconstruction method)
PRIMARY_BRANCH = "ReconTrueDeltaRowX"

# Pixel geometry defaults (will be read from ROOT files if available)
DEFAULT_PIXEL_SPACING_MM = 0.5
DEFAULT_PIXEL_SIZE_MM = 0.15


# ============================================================================
# Comparison Plotting
# ============================================================================

def load_sigma_f_data(output_dir: Path, branch: str = PRIMARY_BRANCH) -> Optional[pd.DataFrame]:
    """Load sigma_f data from the Excel summary file."""
    excel_path = output_dir / "uproot_gaussian_out" / "gaussian_sigma_summary.xlsx"
    if not excel_path.exists():
        print(f"[WARN] Excel file not found: {excel_path}")
        return None

    try:
        df = pd.read_excel(excel_path, sheet_name=branch)
        return df
    except Exception as e:
        print(f"[WARN] Failed to load {excel_path}: {e}")
        return None


def draw_pixel_regions(
    ax: plt.Axes,
    pixel_spacing_mm: float,
    pixel_size_mm: float,
    x_range_mm: Tuple[float, float],
    color: str = "#d0d0d0",
    alpha: float = 0.5,
) -> None:
    """Draw gray vertical bands for pixel pad locations."""
    half_size = pixel_size_mm / 2.0
    x_min, x_max = x_range_mm

    max_n = int(np.ceil(max(abs(x_min), abs(x_max)) / pixel_spacing_mm)) + 1
    for n in range(-max_n, max_n + 1):
        center = n * pixel_spacing_mm
        left = center - half_size
        right = center + half_size
        if right >= x_min and left <= x_max:
            ax.axvspan(left, right, color=color, alpha=alpha, zorder=0)


def create_comparison_plot(
    batch_dir: Path,
    plot_name: str,
    title: str,
    config_1d: str,
    config_2d: str,
    branch: str = PRIMARY_BRANCH,
) -> Optional[Path]:
    """Create a comparison plot overlaying 1D and 2D sigma_f results."""

    # Load data for both configurations
    dir_1d = batch_dir / config_1d
    dir_2d = batch_dir / config_2d

    df_1d = load_sigma_f_data(dir_1d, branch)
    df_2d = load_sigma_f_data(dir_2d, branch)

    if df_1d is None or df_2d is None:
        print(f"[WARN] Missing data for comparison plot: {plot_name}")
        return None

    # Extract data
    x_1d = df_1d["x (um)"].values / 1000.0  # Convert to mm
    sigma_1d = df_1d["sigma_f (um)"].values
    err_1d = df_1d["d_sigma_f (um)"].values if "d_sigma_f (um)" in df_1d.columns else None

    x_2d = df_2d["x (um)"].values / 1000.0
    sigma_2d = df_2d["sigma_f (um)"].values
    err_2d = df_2d["d_sigma_f (um)"].values if "d_sigma_f (um)" in df_2d.columns else None

    # Sort by x
    sort_1d = np.argsort(x_1d)
    x_1d, sigma_1d = x_1d[sort_1d], sigma_1d[sort_1d]
    if err_1d is not None:
        err_1d = err_1d[sort_1d]

    sort_2d = np.argsort(x_2d)
    x_2d, sigma_2d = x_2d[sort_2d], sigma_2d[sort_2d]
    if err_2d is not None:
        err_2d = err_2d[sort_2d]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Determine x range
    x_min = min(np.nanmin(x_1d), np.nanmin(x_2d)) - 0.05
    x_max = max(np.nanmax(x_1d), np.nanmax(x_2d)) + 0.05

    # Draw pixel regions
    draw_pixel_regions(ax, DEFAULT_PIXEL_SPACING_MM, DEFAULT_PIXEL_SIZE_MM,
                      (x_min, x_max), color="#d0d0d0", alpha=0.5)

    # Plot 1D data
    ax.step(x_1d, sigma_1d, where="mid", color="#1f77b4", lw=2, label="1D Gaussian", zorder=2)
    ax.scatter(x_1d, sigma_1d, marker='o', s=40, color="#1f77b4",
               edgecolors='black', linewidths=0.5, zorder=4)
    if err_1d is not None and np.any(np.isfinite(err_1d)):
        ax.errorbar(x_1d, sigma_1d, yerr=err_1d, fmt='none',
                   capsize=2, color='#1f77b4', alpha=0.5, zorder=3)

    # Plot 2D data
    ax.step(x_2d, sigma_2d, where="mid", color="#d62728", lw=2, label="2D Gaussian", zorder=2)
    ax.scatter(x_2d, sigma_2d, marker='s', s=40, color="#d62728",
               edgecolors='black', linewidths=0.5, zorder=4)
    if err_2d is not None and np.any(np.isfinite(err_2d)):
        ax.errorbar(x_2d, sigma_2d, yerr=err_2d, fmt='none',
                   capsize=2, color='#d62728', alpha=0.5, zorder=3)

    # Labels and styling
    ax.set_xlabel("Track x position [mm]", fontsize=12)
    ax.set_ylabel(r"Position resolution $\sigma_f$ [µm]", fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Y-axis from 0
    y_max = max(np.nanmax(sigma_1d), np.nanmax(sigma_2d))
    ax.set_ylim(0, y_max * 1.15)

    ax.set_title(f"Position Resolution: 1D vs 2D Gaussian Fit\n{title}", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()

    # Save plot
    comparison_dir = batch_dir / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    out_path = comparison_dir / f"1D_vs_2D_{plot_name}.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def generate_comparison_plots(batch_dir: Path, results: Dict[str, bool]) -> List[Path]:
    """Generate all comparison plots for completed configurations."""
    print_banner("GENERATING COMPARISON PLOTS")

    generated = []
    for plot_name, title, config_1d, config_2d in COMPARISON_GROUPS:
        # Check if both configs completed successfully
        if not results.get(config_1d, False) or not results.get(config_2d, False):
            print(f"[SKIP] {plot_name}: Missing successful runs for {config_1d} and/or {config_2d}")
            continue

        print(f"[INFO] Creating comparison: {title}")
        out_path = create_comparison_plot(batch_dir, plot_name, title, config_1d, config_2d)
        if out_path:
            print(f"[OK] Saved: {out_path}")
            generated.append(out_path)

    return generated


# ============================================================================
# Config.hh Manipulation
# ============================================================================

def read_config() -> str:
    """Read the current Config.hh content."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return f.read()


def write_config(content: str) -> None:
    """Write new content to Config.hh."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def modify_bool_constant(content: str, name: str, value: bool) -> str:
    """Modify a bool constant in Config.hh."""
    bool_str = "true" if value else "false"
    # Match patterns like: inline constexpr G4bool NAME = true;
    pattern = rf"(inline\s+constexpr\s+G4bool\s+{name}\s*=\s*)(true|false)(\s*;)"
    replacement = rf"\g<1>{bool_str}\g<3>"
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"[WARN] Could not find constant {name} in Config.hh")
    return new_content


def modify_double_constant(content: str, name: str, value: float) -> str:
    """Modify a double constant in Config.hh."""
    # Match patterns like: inline constexpr G4double NAME = 5.0;
    pattern = rf"(inline\s+constexpr\s+G4double\s+{name}\s*=\s*)([\d.]+)(\s*;)"
    replacement = rf"\g<1>{value}\g<3>"
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"[WARN] Could not find constant {name} in Config.hh")
    return new_content


def apply_config(cfg: FitConfig) -> str:
    """Apply a FitConfig to Config.hh and return the original content."""
    original = read_config()
    modified = original

    # Apply settings
    modified = modify_bool_constant(modified, "FIT_GAUS_1D", cfg.fit_gaus_1d)
    modified = modify_bool_constant(modified, "FIT_GAUS_2D", cfg.fit_gaus_2d)
    modified = modify_bool_constant(modified, "FIT_USE_VERTICAL_UNCERTAINTIES", cfg.use_vertical_uncertainties)
    modified = modify_bool_constant(modified, "FIT_USE_DISTANCE_WEIGHTED_ERRORS", cfg.use_distance_weighted)
    modified = modify_bool_constant(modified, "FIT_DISTANCE_POWER_INVERSE", cfg.distance_power_inverse)
    modified = modify_double_constant(modified, "FIT_ERROR_PERCENT_OF_MAX", cfg.error_percent)

    write_config(modified)
    return original


def restore_config(original: str) -> None:
    """Restore Config.hh to its original content."""
    write_config(original)


# ============================================================================
# Sweep Execution
# ============================================================================

def run_sweep(output_dir: Path, n_events: int, skip_analysis: bool) -> bool:
    """Run sweep_x.py with the given output directory."""
    cmd = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--output-dir", str(output_dir),
        "--n-events", str(n_events),
    ]
    if skip_analysis:
        cmd.append("--skip-analysis")

    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def print_banner(text: str, char: str = "=") -> None:
    """Print a banner with the given text."""
    width = 80
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run sweep_x.py with multiple fit configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base directory for all output runs",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=10000,
        help="Number of events per position",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip post-sweep analysis",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=[c.name for c in CONFIGURATIONS] + ["all"],
        default=["all"],
        help="Which configurations to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running sweeps",
    )
    args = parser.parse_args()

    # Determine which configs to run
    if "all" in args.configs:
        configs_to_run = CONFIGURATIONS
    else:
        configs_to_run = [c for c in CONFIGURATIONS if c.name in args.configs]

    if not configs_to_run:
        print("[ERROR] No configurations selected")
        return 1

    # Create timestamp for this batch
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = Path(args.output_base)
    batch_dir = output_base / timestamp

    print_banner("FIT CONFIGURATION SWEEP")
    print(f"Timestamp: {timestamp}")
    print(f"Output base: {batch_dir}")
    print(f"Events per position: {args.n_events}")
    print(f"Configurations to run: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg.name}: {cfg.description}")

    if args.dry_run:
        print("\n[DRY RUN] Would modify Config.hh with these settings:")
        for cfg in configs_to_run:
            print(f"\n  {cfg.name}:")
            print(f"    FIT_GAUS_1D = {cfg.fit_gaus_1d}")
            print(f"    FIT_GAUS_2D = {cfg.fit_gaus_2d}")
            print(f"    FIT_USE_VERTICAL_UNCERTAINTIES = {cfg.use_vertical_uncertainties}")
            print(f"    FIT_USE_DISTANCE_WEIGHTED_ERRORS = {cfg.use_distance_weighted}")
            print(f"    FIT_DISTANCE_POWER_INVERSE = {cfg.distance_power_inverse}")
            print(f"    FIT_ERROR_PERCENT_OF_MAX = {cfg.error_percent}")
        return 0

    # Store original config to restore later
    original_config = read_config()

    results: Dict[str, bool] = {}

    try:
        for i, cfg in enumerate(configs_to_run, 1):
            print_banner(f"[{i}/{len(configs_to_run)}] {cfg.name}: {cfg.description}")

            # Apply configuration
            print("[INFO] Modifying Config.hh...")
            print(f"  FIT_GAUS_1D = {cfg.fit_gaus_1d}")
            print(f"  FIT_GAUS_2D = {cfg.fit_gaus_2d}")
            print(f"  FIT_USE_VERTICAL_UNCERTAINTIES = {cfg.use_vertical_uncertainties}")
            print(f"  FIT_USE_DISTANCE_WEIGHTED_ERRORS = {cfg.use_distance_weighted}")
            print(f"  FIT_DISTANCE_POWER_INVERSE = {cfg.distance_power_inverse}")
            print(f"  FIT_ERROR_PERCENT_OF_MAX = {cfg.error_percent}")
            apply_config(cfg)

            # Run sweep
            output_dir = batch_dir / cfg.name
            print(f"[INFO] Running sweep to {output_dir}")
            success = run_sweep(output_dir, args.n_events, args.skip_analysis)
            results[cfg.name] = success

            if success:
                print(f"[OK] {cfg.name} completed successfully")
            else:
                print(f"[FAIL] {cfg.name} failed")

    finally:
        # Always restore original config
        print("\n[INFO] Restoring original Config.hh...")
        restore_config(original_config)

    # Generate comparison plots if not skipping analysis
    comparison_plots = []
    if not args.skip_analysis:
        comparison_plots = generate_comparison_plots(batch_dir, results)

    # Print summary
    print_banner("SUMMARY")
    print(f"Results saved to: {batch_dir}")
    print()
    all_success = True
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_success = False

    if comparison_plots:
        print()
        print("Comparison plots:")
        for p in comparison_plots:
            print(f"  {p}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
