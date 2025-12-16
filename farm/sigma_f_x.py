#!/usr/bin/env python3
"""
Gaussian fitting and export utility using uproot.

Tasks:
 1) Read all ROOT files in Q_f directory.
 2) For branches:
    - ReconTrueDeltaRowX
    - ReconTrueDeltaSDiagX
    - ReconTrueDeltaMDiagX
    - ReconTrueDeltaX_2D
    - ReconTrueDeltaMeanX
    Fit a Gaussian to the histogram and extract sigma (sigma_f) and its fit error.
 3) Save a PNG with the histogram and the fitted Gaussian overlay for each file/branch.
 4) Create an Excel workbook with one sheet per branch containing columns:
      x (um), sigma_f (um), d_sigma_f (um), stdev (um)

Notes:
 - Assumes values in the branches are in millimeters (mm). We convert to micrometers (um)
   for the Excel export to match requested units.
 - Invalid values are assumed to be NaN (as produced by this codebase) and are filtered out.
 - Requires packages: uproot, numpy, scipy, matplotlib, pandas, openpyxl
"""

from __future__ import annotations

import os
import re
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uproot


# ---------------------------- Configuration ---------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SWEEP_RUNS_DIR = REPO_ROOT / "sweep_x_runs"
TREE_NAME = "Hits"


def find_latest_sweep_dir() -> pathlib.Path:
    """Find the most recent timestamped directory in sweep_x_runs."""
    if not SWEEP_RUNS_DIR.exists():
        raise FileNotFoundError(f"Sweep runs directory not found: {SWEEP_RUNS_DIR}")
    # Look for directories with timestamp pattern YYYYMMDD-HHMMSS
    candidates = [
        d for d in SWEEP_RUNS_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No sweep run directories found in {SWEEP_RUNS_DIR}")
    # Sort by name (timestamp format sorts chronologically)
    return sorted(candidates, key=lambda d: d.name)[-1]


def get_default_input_dir() -> pathlib.Path:
    """Get the default input directory (latest sweep run)."""
    return find_latest_sweep_dir()


def get_default_output_dir(input_dir: pathlib.Path) -> pathlib.Path:
    """Get the default output directory based on input directory."""
    return input_dir / "uproot_gaussian_out"

BRANCHES = [
    "ReconTrueDeltaRowX",
    "ReconTrueDeltaSDiagX",
    "ReconTrueDeltaMDiagX",
    "ReconTrueDeltaX_2D",
    "ReconTrueDeltaMeanX",
]

NUM_BINS = 80
FIGSIZE = (12, 8)
DPI = 300


# ------------------------------- Utilities -----------------------------------

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def read_metadata_from_root_file(root_path: pathlib.Path) -> Dict[str, float]:
    """Read grid metadata (pixel size, spacing) from a ROOT file.
    
    Returns a dict with keys like 'GridPixelSize_mm', 'GridPixelSpacing_mm'.
    Uses TParameter<double> objects stored in the tree's UserInfo.
    """
    metadata = {}
    try:
        with uproot.open(root_path) as f:
            if "Hits" not in f:
                return metadata
            # Try accessing via file-level keys for TParameter objects
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, 'member'):
                    try:
                        name = key.split(';')[0]
                        if name.startswith('Grid') or name == 'Gain':
                            val = obj.member('fVal')
                            metadata[name] = float(val)
                    except Exception:
                        pass
    except Exception:
        pass
    return metadata


def draw_pixel_regions(
    ax: plt.Axes,
    pixel_spacing_mm: float,
    pixel_size_mm: float,
    x_range_mm: Tuple[float, float] = (-0.8, 0.8),
    color: str = "lightgray",
    alpha: float = 0.5,
) -> None:
    """Draw gray vertical bands for pixel pad locations.

    Assumes pixels are centered at 0, ±pixel_spacing_mm, etc.
    Each pixel pad extends ±(pixel_size_mm/2) from its center.
    """
    half_size = pixel_size_mm / 2.0
    x_min, x_max = x_range_mm

    # Find all pixel centers in range
    max_n = int(np.ceil(max(abs(x_min), abs(x_max)) / pixel_spacing_mm)) + 1
    for n in range(-max_n, max_n + 1):
        center = n * pixel_spacing_mm
        left = center - half_size
        right = center + half_size
        if right >= x_min and left <= x_max:
            ax.axvspan(left, right, color=color, alpha=alpha, zorder=0)


def get_pixel_boundaries(
    pixel_spacing_mm: float,
    pixel_size_mm: float,
    x_range_mm: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Get list of (left_edge, right_edge) for all pixel regions in x_range."""
    half_size = pixel_size_mm / 2.0
    x_min, x_max = x_range_mm
    boundaries = []

    max_n = int(np.ceil(max(abs(x_min), abs(x_max)) / pixel_spacing_mm)) + 1
    for n in range(-max_n, max_n + 1):
        center = n * pixel_spacing_mm
        left = center - half_size
        right = center + half_size
        if right >= x_min and left <= x_max:
            boundaries.append((left, right))

    return sorted(boundaries, key=lambda b: b[0])


def add_binary_resolution_in_pixel_regions(
    x_vals_mm: np.ndarray,
    sigma_vals_um: np.ndarray,
    pixel_spacing_mm: float,
    pixel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add points with binary resolution (pixel_size/sqrt(12)) inside pixel regions.

    This makes step plots show the degraded binary resolution inside metal pads
    rather than interpolating across them.

    Returns:
        Tuple of (x_vals_mm, sigma_vals_um) with inserted pixel region points.
    """
    if len(x_vals_mm) == 0:
        return x_vals_mm, sigma_vals_um

    # Binary resolution in µm
    binary_res_um = (pixel_size_mm * 1000.0) / math.sqrt(12.0)

    x_min, x_max = np.nanmin(x_vals_mm), np.nanmax(x_vals_mm)
    boundaries = get_pixel_boundaries(pixel_spacing_mm, pixel_size_mm, (x_min, x_max))

    if not boundaries:
        return x_vals_mm, sigma_vals_um

    # Build new arrays with inserted pixel region points
    new_x = list(x_vals_mm)
    new_sigma = list(sigma_vals_um)

    # Small offset to place points just inside pixel boundaries
    eps = 0.001  # 1 µm in mm

    for left, right in boundaries:
        # Add points just inside the pixel boundaries with binary resolution
        # These create the flat horizontal line inside the pixel region
        left_inside = left + eps
        right_inside = right - eps

        # Only add if within data range and not duplicating existing points
        if left_inside >= x_min and left_inside <= x_max:
            if not any(abs(x - left_inside) < eps * 2 for x in new_x):
                new_x.append(left_inside)
                new_sigma.append(binary_res_um)

        if right_inside >= x_min and right_inside <= x_max:
            if not any(abs(x - right_inside) < eps * 2 for x in new_x):
                new_x.append(right_inside)
                new_sigma.append(binary_res_um)

    # Sort by x position
    sort_idx = np.argsort(new_x)
    return np.array(new_x)[sort_idx], np.array(new_sigma)[sort_idx]


def parse_x_um_from_filename(path: str) -> float:
    """Parse the particle gun x-position in micrometers from the file name.

    Accepts names like "-100um.root", "25um.root", or with prefixes.
    Returns NaN if no match.
    """
    fname = os.path.basename(path)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*um", fname, flags=re.IGNORECASE)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")


def gaussian_with_const(x: np.ndarray, A: float, mu: float, sigma: float, B: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-12)) ** 2) + B


@dataclass
class FitResult:
    mu: float
    sigma: float
    sigma_err: float
    popt: Tuple[float, float, float, float]
    pcov: np.ndarray
    entries: int
    mean: float
    stdev: float
    hist_x: np.ndarray
    hist_y: np.ndarray


def fit_gaussian_to_values(values_mm: np.ndarray, *, num_bins: int = NUM_BINS) -> FitResult:
    """Compute histogram and fit a Gaussian+constant to the counts.

    Returns sigma and its uncertainty from the covariance matrix. Also returns
    entries, mean, stdev computed directly from the values (population std).
    """
    data = np.asarray(values_mm, dtype=float)
    data = data[np.isfinite(data)]

    if data.size < 10:
        # Not enough statistics to fit
        return FitResult(float("nan"), float("nan"), float("nan"), (0.0, 0.0, 0.0, 0.0), np.full((4, 4), np.nan), 0, float("nan"), float("nan"), np.array([]), np.array([]))

    # Focus range for fit; separate full-range histogram for display/stat consistency
    mean_data = float(np.mean(data))
    stdev_data = float(np.std(data, ddof=0))
    fit_lo = float(np.percentile(data, 0.5))
    fit_hi = float(np.percentile(data, 99.5))
    if not np.isfinite(fit_lo) or not np.isfinite(fit_hi) or not (fit_hi > fit_lo):
        fit_lo, fit_hi = float(np.min(data)), float(np.max(data))
    if fit_hi <= fit_lo:
        if np.isfinite(stdev_data) and stdev_data > 0:
            fit_lo, fit_hi = mean_data - 5.0 * stdev_data, mean_data + 5.0 * stdev_data
        else:
            span = max(1e-4, abs(mean_data) + 1e-4)
            fit_lo, fit_hi = mean_data - span, mean_data + span

    # Full-range histogram (to include all data in the displayed entries)
    hist_lo, hist_hi = float(np.min(data)), float(np.max(data))
    hist_y, edges = np.histogram(data, bins=num_bins, range=(hist_lo, hist_hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Histogram statistics like ROOT (weighted by bin counts)
    entries = int(np.sum(hist_y))
    mean_hist = float(np.sum(centers * hist_y) / entries) if entries > 0 else float("nan")
    var_hist = float(np.sum(hist_y * (centers - mean_hist) ** 2) / entries) if entries > 0 else float("nan")
    stdev_hist = float(math.sqrt(var_hist)) if np.isfinite(var_hist) and var_hist >= 0 else float("nan")

    # Initial parameter guesses
    A0 = float(np.max(hist_y)) if np.max(hist_y) > 0 else 1.0
    mu0 = mean_hist
    sigma0 = stdev_hist if (np.isfinite(stdev_hist) and stdev_hist > 0) else (0.25 * (fit_hi - fit_lo) / 3.0)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(1e-3, 0.01 * (fit_hi - fit_lo))
    B0 = float(np.min(hist_y)) if hist_y.size > 0 else 0.0

    weights = np.sqrt(np.maximum(hist_y, 1.0))  # Poisson approximation

    try:
        popt, pcov = curve_fit(
            gaussian_with_const,
            centers,
            hist_y,
            p0=(A0, mu0, sigma0, B0),
            sigma=weights,
            absolute_sigma=True,
            bounds=([0.0, -np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        sigma_err = float(math.sqrt(abs(pcov[2, 2]))) if np.isfinite(pcov[2, 2]) else float("nan")
        mu_fit, sigma_fit = float(popt[1]), float(abs(popt[2]))
        return FitResult(mu_fit, sigma_fit, sigma_err, tuple(popt), pcov, entries, mean_hist, stdev_hist, centers, hist_y)
    except Exception:
        popt = (A0, mu0, sigma0, B0)
        pcov = np.full((4, 4), np.nan)
        return FitResult(mu0, sigma0, float("nan"), popt, pcov, entries, mean_hist, stdev_hist, centers, hist_y)


def save_plot(
    branch: str,
    file_title: str,
    centers: np.ndarray,
    counts: np.ndarray,
    popt: Tuple[float, float, float, float],
    entries: int,
    mean_val: float,
    stdev_val: float,
    sigma_f: float,
    out_png: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Histogram as step to match ROOT style
    ax.step(centers, counts, where="mid", color="#2b5fff", linewidth=1.2)

    # Overlay fitted Gaussian
    if centers.size > 0:
        xs = np.linspace(centers.min(), centers.max(), 800)
        ys = gaussian_with_const(xs, *popt)
        ax.plot(xs, ys, color="red", linewidth=1.5)

    ax.set_title(f"{branch}")
    ax.set_xlabel("Delta X [mm]")
    ax.set_ylabel("Entries")

    # Stats box similar to ROOT (use spaces, not tabs; add sigma_f)
    text = (
        f"Entries: {entries}\n"
        f"Mean: {mean_val:+.5f}\n"
        f"Std Dev: {stdev_val:.5f}\n"
        f"σ_f: {sigma_f:.5f}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )

    # Footer with file descriptor (x position)
    ax.text(0.01, 0.02, file_title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, alpha=0.7)

    ensure_dir(os.path.dirname(out_png))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def process_root_file(root_path: str, results: Dict[str, List[Tuple[float, float, float, float]]], output_base: pathlib.Path) -> None:
    """Process one ROOT file: fit specified branches and save plots.

    Updates `results` in-place: results[branch].append((x_um, sigma_um, d_sigma_um, stdev_um))
    """
    x_um = parse_x_um_from_filename(root_path)
    file_desc = f"x = {x_um:.0f} um" if np.isfinite(x_um) else os.path.basename(root_path)

    try:
        with uproot.open(root_path) as f:
            if TREE_NAME not in f:
                print(f"[WARN] Tree '{TREE_NAME}' not found in {root_path}. Skipping.")
                return
            tree = f[TREE_NAME]
            available = set(tree.keys())
            for branch in BRANCHES:
                if branch not in available:
                    print(f"[INFO] Branch '{branch}' not found in {os.path.basename(root_path)}. Skipping branch.")
                    continue

                arr = tree[branch].array(library="np")
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    print(f"[INFO] No valid data for {branch} in {os.path.basename(root_path)}")
                    continue

                fit = fit_gaussian_to_values(arr, num_bins=NUM_BINS)

                # Save plot
                out_dir = output_base / "plots" / branch
                out_png = out_dir / pathlib.Path(root_path).with_suffix(".png").name
                save_plot(
                    branch,
                    file_desc,
                    fit.hist_x,
                    fit.hist_y,
                    fit.popt,
                    arr.size,
                    float(np.mean(arr)) if arr.size > 0 else float("nan"),
                    float(np.std(arr, ddof=0)) if arr.size > 0 else float("nan"),
                    fit.sigma,
                    str(out_png),
                )

                # Convert mm -> um for export values
                sigma_um = float(fit.sigma * 1000.0)
                d_sigma_um = float(fit.sigma_err * 1000.0) if np.isfinite(fit.sigma_err) else float("nan")
                stdev_um = float(fit.stdev * 1000.0)
                results.setdefault(branch, []).append((x_um, sigma_um, d_sigma_um, stdev_um))

    except Exception as e:
        print(f"[ERROR] Failed to process '{root_path}': {e}")


def write_excel(results: Dict[str, List[Tuple[float, float, float, float]]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for branch, rows in results.items():
            if not rows:
                continue
            # Sort by x
            rows_sorted = sorted(rows, key=lambda t: (np.nan_to_num(t[0], nan=np.inf)))
            df = pd.DataFrame(rows_sorted, columns=["x (um)", "sigma_f (um)", "d_sigma_f (um)", "stdev (um)"])
            df.to_excel(writer, sheet_name=branch[:31], index=False)


def save_sigma_vs_x_scatter(
    results: Dict[str, List[Tuple[float, float, float, float]]],
    output_dir: pathlib.Path,
    pixel_spacing_mm: float = 0.5,
    pixel_size_mm: float = 0.15,
) -> None:
    """Create FNAL paper-style plots of sigma_f vs x for each branch.

    Style based on Tornago et al. / FNAL beam test publications:
    - Gray bands for pixel/metal pad regions
    - Step/histogram style lines for resolution data
    - Binary resolution (pixel_size/sqrt(12)) shown inside pixel regions
    - Clean axis labels and formatting
    """
    scatter_dir = output_dir / "scatter_plots"
    ensure_dir(str(scatter_dir))

    for branch, rows in results.items():
        if not rows:
            continue

        # Sort by x and extract data
        rows_sorted = sorted(rows, key=lambda t: (np.nan_to_num(t[0], nan=np.inf)))
        x_vals_um = np.array([r[0] for r in rows_sorted])
        sigma_vals = np.array([r[1] for r in rows_sorted])
        sigma_errs = np.array([r[2] for r in rows_sorted])

        # Filter out NaN values
        mask = np.isfinite(x_vals_um) & np.isfinite(sigma_vals)
        x_vals_um = x_vals_um[mask]
        sigma_vals = sigma_vals[mask]
        sigma_errs = sigma_errs[mask]

        if len(x_vals_um) == 0:
            continue

        # Convert to mm for x-axis
        x_vals_mm = x_vals_um / 1000.0

        # Add binary resolution points inside pixel regions
        x_vals_mm_extended, sigma_vals_extended = add_binary_resolution_in_pixel_regions(
            x_vals_mm, sigma_vals, pixel_spacing_mm, pixel_size_mm
        )

        # FNAL paper-style figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

        # Determine x range from data (extend slightly beyond data)
        x_min = np.nanmin(x_vals_mm) - 0.05
        x_max = np.nanmax(x_vals_mm) + 0.05

        # Draw gray pixel regions first (behind the data)
        draw_pixel_regions(ax, pixel_spacing_mm, pixel_size_mm,
                          x_range_mm=(x_min, x_max), color="#d0d0d0", alpha=0.6)

        # Plot as histogram-style step (FNAL paper style) with extended data
        ax.step(x_vals_mm_extended, sigma_vals_extended, where="mid", color="#1f77b4", lw=2,
                zorder=2, label="Simulation")

        # Add square markers at measurement points (FNAL paper style)
        ax.scatter(x_vals_mm, sigma_vals, marker='s', s=40, color="#1f77b4",
                   edgecolors='black', linewidths=0.5, zorder=4)

        # Add error bars for original data points only (not the inserted binary resolution points)
        has_errors = np.any(np.isfinite(sigma_errs))
        if has_errors:
            ax.errorbar(x_vals_mm, sigma_vals, yerr=sigma_errs, fmt='none',
                       capsize=2, color='#666666', zorder=3, alpha=0.7)

        # FNAL paper-style axis labels
        ax.set_xlabel("Track x position [mm]", fontsize=12)
        ax.set_ylabel(r"Position resolution [µm]", fontsize=12)
        ax.set_xlim(x_min, x_max)

        # Y-axis starts from 0, extends above max with margin
        y_max_data = np.nanmax(sigma_vals_extended)
        ax.set_ylim(0, y_max_data * 1.1)

        # Minimal grid like paper
        ax.grid(True, alpha=0.3, zorder=1, linestyle='-')
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Title and legend
        branch_name = branch.replace("ReconTrueDelta", "").replace("_2D", " (2D)")
        ax.set_title(f"Position resolution vs track position - {branch_name}", fontsize=12)
        ax.legend(loc='upper right', fontsize=10)

        fig.tight_layout()
        out_png = scatter_dir / f"sigma_vs_x_{branch}.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[OK] Saved plot: {out_png}")


def main(input_dir: pathlib.Path = None, output_dir: pathlib.Path = None) -> None:
    """Main entry point. Can be called directly with paths or via CLI."""
    import argparse

    # Handle CLI arguments if called without parameters
    if input_dir is None:
        parser = argparse.ArgumentParser(
            description="Gaussian fitting for reconstruction deltas vs. x position"
        )
        parser.add_argument(
            "--input-dir",
            dest="input_dir",
            type=str,
            default=None,
            help="Directory containing ROOT files (default: latest in sweep_x_runs)",
        )
        parser.add_argument(
            "--output-dir",
            dest="output_dir",
            type=str,
            default=None,
            help="Output directory (default: input_dir/uproot_gaussian_out)",
        )
        args = parser.parse_args()

        if args.input_dir:
            p = pathlib.Path(args.input_dir)
            input_dir = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        else:
            input_dir = get_default_input_dir()

        if args.output_dir:
            p = pathlib.Path(args.output_dir)
            output_dir = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        else:
            output_dir = get_default_output_dir(input_dir)
    elif output_dir is None:
        output_dir = get_default_output_dir(input_dir)

    ensure_dir(str(output_dir))

    # Gather ROOT files
    root_files = sorted(input_dir.glob("*.root"))
    if not root_files:
        print(f"[WARN] No ROOT files found in {input_dir}")
        return

    # Read pixel geometry metadata from first ROOT file
    pixel_spacing_mm = 0.5  # Default: 500 µm
    pixel_size_mm = 0.15    # Default: 150 µm
    if root_files:
        metadata = read_metadata_from_root_file(root_files[0])
        if "GridPixelSpacing_mm" in metadata:
            pixel_spacing_mm = metadata["GridPixelSpacing_mm"]
        if "GridPixelSize_mm" in metadata:
            pixel_size_mm = metadata["GridPixelSize_mm"]
        print(f"[INFO] Pixel geometry: spacing={pixel_spacing_mm:.3f} mm, size={pixel_size_mm:.3f} mm")

    results: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for path in root_files:
        print(f"[INFO] Processing {path.name}")
        process_root_file(str(path), results, output_dir)

    out_excel = output_dir / "gaussian_sigma_summary.xlsx"
    write_excel(results, str(out_excel))
    print(f"[OK] Wrote Excel: {out_excel}")
    print(f"[OK] PNG plots rooted at: {output_dir / 'plots'}")

    # Generate sigma vs x histogram plots
    save_sigma_vs_x_scatter(results, output_dir, 
                            pixel_spacing_mm=pixel_spacing_mm,
                            pixel_size_mm=pixel_size_mm)


if __name__ == "__main__":
    main()


