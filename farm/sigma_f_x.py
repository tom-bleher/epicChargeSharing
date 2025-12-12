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

    results: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for path in root_files:
        print(f"[INFO] Processing {path.name}")
        process_root_file(str(path), results, output_dir)

    out_excel = output_dir / "gaussian_sigma_summary.xlsx"
    write_excel(results, str(out_excel))
    print(f"[OK] Wrote Excel: {out_excel}")
    print(f"[OK] PNG plots rooted at: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()


