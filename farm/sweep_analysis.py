#!/usr/bin/env python3
"""
Unified sweep analysis pipeline.

This script combines the functionality of:
- sweep_x.py: Run simulations at different particle gun positions
- run_gaussian_fits.py: Generate PDFs of Q_i vs d Gaussian fits
- plotChargeNeighborhood.C: Generate charge neighborhood visualization PDFs
- sigma_f_x.py: Extract sigma_f from ReconTrueDeltaX distributions
- Fi_x.py: Extract F_i values per position

Workflow per position:
1. Configure simulation parameters (charge model, active pixel mode, beta)
2. Modify PrimaryGenerator.cc to set fFixedX
3. Build and run simulation
4. Save ROOT file with position-based name
5. Generate PDF of N individual Gaussian fits (Q_i vs d)
6. Generate charge neighborhood PDFs colored by:
   - Fi: Signal fraction (F_i)
   - Qi: Induced charge (Q_i)
   - Qn: Charge with noise (Q_n)
   - Qf: Final charge (Q_f)
   - distance: Distance from hit to pixel center
7. Fit ReconTrueDeltaX distribution to get sigma_f
8. Extract F_i values

Simulation configuration options:
- Charge sharing model: LogA (logarithmic, Tornago Eq.4), LinA (linear, Tornago Eq.6)
- Active pixel mode: Neighborhood (all pixels), ChargeBlock2x2/3x3 (highest F_i), RowCol/RowCol3x3 (cross pattern)
- Beta attenuation coefficient for linear model

Final outputs:
- Plot of sigma_F vs d
- Plot of F_i vs d (Tornago et al. Figure 5 style)
- Charge neighborhood visualization PDFs
- Excel summary files

Usage examples:
    # Run with default settings from sweep_config.yaml
    python sweep_analysis.py run

    # Run with a custom config file
    python sweep_analysis.py run --config my_config.yaml

    # Run with CLI overrides (CLI args take precedence over config)
    python sweep_analysis.py run --positions 0,50,-50,100,-100

    # Run with LogA charge model and ChargeBlock active pixel mode
    python sweep_analysis.py run --charge-model LogA --active-pixel-mode ChargeBlock2x2

    # Run with linear model and custom beta
    python sweep_analysis.py run --charge-model LinA --beta 0.005

    # Analyze existing ROOT files using config
    python sweep_analysis.py analyze /path/to/root/files --config sweep_config.yaml

Configuration file (sweep_config.yaml):
    All settings can be configured via YAML file. See sweep_config.yaml for the
    full schema with documentation. CLI arguments override config file values.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import yaml

try:
    import uproot
    import awkward as ak
except ImportError:
    print("ERROR: uproot and awkward are required. Install with: pip install uproot awkward")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
FARM_DIR = Path(__file__).resolve().parent
SRC_FILE = REPO_ROOT / "src" / "PrimaryGenerator.cc"
CONFIG_FILE = REPO_ROOT / "include" / "Config.hh"
BUILD_DIR = REPO_ROOT / "build"
EXECUTABLE = BUILD_DIR / "epicChargeSharing"
RUN_MAC = (BUILD_DIR / "run.mac") if (BUILD_DIR / "run.mac").exists() else (REPO_ROOT / "macros" / "run.mac")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "sweep_analysis_runs"
MACRO_PATH = REPO_ROOT / "proc" / "fit" / "plotFitGaus1DReplayQiFit.C"
NEIGHBORHOOD_MACRO_PATH = REPO_ROOT / "proc" / "grid" / "plotChargeNeighborhood.C"
DEFAULT_CONFIG_FILE = FARM_DIR / "sweep_config.yaml"

# Charge sharing model options
CHARGE_MODELS = ["LogA", "LinA"]
ACTIVE_PIXEL_MODES = ["Neighborhood", "ChargeBlock2x2", "ChargeBlock3x3", "RowCol", "RowCol3x3"]

# Default beta value for linear model
DEFAULT_BETA = 0.001

# Charge neighborhood visualization types
# Maps user-friendly names to (dataKind, chargeBranch) pairs for the ROOT macro
# - dataKind: "fraction" (shows Fi values), "coulomb" (shows charge values), "distance" (shows distance to hit)
# - chargeBranch: ROOT branch name to use ("Fi", "Qi", "Qn", "Qf")
NEIGHBORHOOD_TYPES = {
    "Fi": ("fraction", "Fi"),     # Signal fraction (from Fi branch)
    "Qi": ("coulomb", "Qi"),      # Induced charge (from Qi branch)
    "Qn": ("coulomb", "Qn"),      # Charge with noise (from Qn branch)
    "Qf": ("coulomb", "Qf"),      # Final charge (from Qf branch)
    "distance": ("distance", "Qn"),  # Distance to hit (uses Qn for valid cell detection)
}

# Default positions to sweep (micrometers)
DEFAULT_POSITIONS = [0, 25, -25, 50, -50, 75, -75, 100, -100, 125, -125, 150, -150, 175, -175]

# Regex to locate fFixedX assignment in PrimaryGenerator.cc
FIXED_X_LINE_REGEX = re.compile(
    r"^(\s*)fFixedX\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:\*\s*um)?\s*;\s*$"
)

# Branches to analyze for sigma_f
SIGMA_F_BRANCHES = [
    "ReconTrueDeltaRowX",
    "ReconTrueDeltaSDiagX",
    "ReconTrueDeltaMDiagX",
    "ReconTrueDeltaX_2D",
    "ReconTrueDeltaMeanX",
]

# Gaussian fit configuration for sigma_f
NUM_BINS = 80
FIGSIZE = (12, 8)
DPI = 300

# F_i extraction configuration
SENTINEL_INVALID_FRACTION = -999.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GaussianFitResult:
    """Result of a Gaussian fit to a distribution."""
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


@dataclass
class SigmaFResult:
    """Sigma_f results for a single position."""
    x_um: float
    branch_results: Dict[str, GaussianFitResult] = field(default_factory=dict)


@dataclass
class FiResult:
    """F_i results for a single position."""
    x_um: float
    pixel_id: int
    mean: float
    std: float
    count: int
    distance_um: Optional[float] = None
    coord_mm: Optional[Tuple[float, float]] = None


@dataclass
class PositionResult:
    """Complete results for a single particle gun position."""
    x_um: float
    root_file: Path
    gaussian_fits_pdf: Optional[Path] = None
    sigma_f: Optional[SigmaFResult] = None
    fi: Optional[FiResult] = None
    neighborhood_pdfs: Optional[Dict[str, Path]] = None  # Maps quantity name to PDF path
    distance_to_reference_pixel_um: Optional[float] = None  # |x_gun - x_pixel_center|


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Create directory and parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


def read_file_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_x_um_from_filename(path: Path) -> float:
    """Extract micrometer displacement from filename."""
    name = path.name
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*um", name, flags=re.IGNORECASE)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return result."""
    print(f"$ {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            print(result.stdout)
    else:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
    return result


# =============================================================================
# YAML Configuration Loading
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration loaded from YAML file."""
    positions: List[float] = field(default_factory=lambda: DEFAULT_POSITIONS.copy())
    charge_model: Optional[str] = None
    active_pixel_mode: Optional[str] = None
    beta: Optional[float] = None
    n_events: int = 200
    pixel_id: Optional[int] = None


def load_config(config_path: Optional[Path] = None) -> SweepConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config file.

    Returns:
        SweepConfig object with loaded settings.
    """
    config = SweepConfig()

    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE

    if not config_path.exists():
        print(f"[INFO] Config file not found: {config_path}, using defaults")
        return config

    print(f"[INFO] Loading config from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        return config

    if "positions" in yaml_data and yaml_data["positions"] is not None:
        config.positions = [float(x) for x in yaml_data["positions"]]
    if "charge_model" in yaml_data:
        config.charge_model = yaml_data["charge_model"]
    if "active_pixel_mode" in yaml_data:
        config.active_pixel_mode = yaml_data["active_pixel_mode"]
    if "beta" in yaml_data:
        config.beta = yaml_data["beta"]
    if "n_events" in yaml_data:
        config.n_events = int(yaml_data["n_events"])
    if "pixel_id" in yaml_data:
        config.pixel_id = yaml_data["pixel_id"]

    return config


# =============================================================================
# Config.hh Updater
# =============================================================================

# Regex patterns for updating Config.hh
# Note: Config.hh uses unqualified type names (inside Constants namespace)
CONFIG_POS_RECON_MODEL_REGEX = re.compile(
    r"^(\s*inline\s+constexpr\s+PosReconModel\s+POS_RECON_MODEL\s*=\s*)(?:ECS::Config::)?(?:Constants::)?PosReconModel::\w+\s*;"
)
CONFIG_ACTIVE_PIXEL_MODE_REGEX = re.compile(
    r"^(\s*inline\s+constexpr\s+ActivePixelMode\s+ACTIVE_PIXEL_MODE\s*=\s*)(?:ECS::Config::)?(?:Constants::)?ActivePixelMode::\w+\s*;"
)
CONFIG_BETA_REGEX = re.compile(
    r"^(\s*inline\s+constexpr\s+G4double\s+LINEAR_CHARGE_MODEL_BETA\s*=\s*)[\d.]+\s*;"
)


def update_config_hh(
    source_text: str,
    charge_model: Optional[str] = None,
    active_pixel_mode: Optional[str] = None,
    beta: Optional[float] = None,
) -> str:
    """Update simulation parameters in Config.hh source.

    Args:
        source_text: Current content of Config.hh
        charge_model: One of "LogA", "LinA" (or None to keep current)
        active_pixel_mode: One of "Neighborhood", "ChargeBlock2x2", "ChargeBlock3x3", "RowCol", "RowCol3x3" (or None to keep current)
        beta: Beta attenuation coefficient for linear model (or None to keep current)

    Returns:
        Updated Config.hh content
    """
    lines = source_text.splitlines()
    new_lines = []
    changes = []

    for line in lines:
        new_line = line

        # Update POS_RECON_MODEL
        if charge_model is not None:
            m = CONFIG_POS_RECON_MODEL_REGEX.match(line)
            if m:
                new_line = f"{m.group(1)}PosReconModel::{charge_model};"
                changes.append(f"POS_RECON_MODEL = {charge_model}")

        # Update ACTIVE_PIXEL_MODE
        if active_pixel_mode is not None:
            m = CONFIG_ACTIVE_PIXEL_MODE_REGEX.match(line)
            if m:
                new_line = f"{m.group(1)}ActivePixelMode::{active_pixel_mode};"
                changes.append(f"ACTIVE_PIXEL_MODE = {active_pixel_mode}")

        # Update BETA
        if beta is not None:
            m = CONFIG_BETA_REGEX.match(line)
            if m:
                new_line = f"{m.group(1)}{beta};"
                changes.append(f"LINEAR_CHARGE_MODEL_BETA = {beta}")

        new_lines.append(new_line)

    if changes:
        print(f"[INFO] Updated Config.hh: {', '.join(changes)}")

    return "\n".join(new_lines) + "\n"


def get_current_config_settings(source_text: str) -> Dict[str, str]:
    """Extract current configuration settings from Config.hh.

    Returns:
        Dictionary with keys: charge_model, active_pixel_mode, beta
    """
    settings = {}

    for line in source_text.splitlines():
        m = CONFIG_POS_RECON_MODEL_REGEX.match(line)
        if m:
            # Extract model name from the line
            match = re.search(r"PosReconModel::(\w+)", line)
            if match:
                settings["charge_model"] = match.group(1)

        m = CONFIG_ACTIVE_PIXEL_MODE_REGEX.match(line)
        if m:
            match = re.search(r"ActivePixelMode::(\w+)", line)
            if match:
                settings["active_pixel_mode"] = match.group(1)

        m = CONFIG_BETA_REGEX.match(line)
        if m:
            match = re.search(r"=\s*([\d.]+)", line)
            if match:
                settings["beta"] = match.group(1)

    return settings


# =============================================================================
# Simulation Runner (from sweep_x.py)
# =============================================================================

def update_primary_generator_x_um(source_text: str, x_um: float) -> str:
    """Update fFixedX value in PrimaryGenerator.cc source."""
    lines = source_text.splitlines()
    replaced = False
    new_lines = []
    for line in lines:
        m = FIXED_X_LINE_REGEX.match(line)
        if m:
            indent = m.group(1)
            value_str = f"{x_um:.1f}"
            newline = f"{indent}fFixedX = {value_str}*um;"
            new_lines.append(newline)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        raise RuntimeError(
            "Could not find fixed X assignment in PrimaryGenerator.cc (expected fFixedX = ...;)"
        )
    return "\n".join(new_lines) + ("\n" if source_text.endswith("\n") else "")


def configure_build_if_needed() -> None:
    """Run cmake configure if Makefile doesn't exist."""
    BUILD_DIR.mkdir(exist_ok=True)
    if not (BUILD_DIR / "Makefile").exists():
        run_cmd(["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR), "-DWITH_EDM4HEP=OFF"])


def build_project() -> None:
    """Build the project using cmake."""
    run_cmd(["cmake", "--build", str(BUILD_DIR), "-j"])


def run_simulation() -> None:
    """Run the Geant4 simulation."""
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Executable not found: {EXECUTABLE}")
    if not RUN_MAC.exists():
        raise RuntimeError(f"Run macro not found: {RUN_MAC}")
    run_cmd([str(EXECUTABLE), "-m", str(RUN_MAC)])


def wait_for_file(path: Path, timeout_s: int = 120) -> bool:
    """Wait for a file to appear."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(1)
    return path.exists()


def wait_for_root_close(path: Path, timeout_s: int = 60) -> None:
    """Wait for ROOT file to be closed/writable."""
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
    """Find the output ROOT file after simulation."""
    candidates = [
        REPO_ROOT / "epicChargeSharing.root",
        BUILD_DIR / "epicChargeSharing.root",
    ]
    for c in candidates:
        if wait_for_file(c, timeout_s=180):
            return c
    raise RuntimeError("Expected merged ROOT output file not found after run.")


def run_simulation_for_position(x_um: float, output_dir: Path, original_text: str) -> Path:
    """Run simulation for a specific x position and return output ROOT file path."""
    out_name = output_dir / (
        f"{int(x_um)}um.root" if float(x_um).is_integer() else f"{x_um}um.root"
    )

    print(f"\n{'='*60}")
    print(f"Running simulation for x = {x_um} um")
    print(f"{'='*60}")

    # Update source
    new_text = update_primary_generator_x_um(original_text, float(x_um))
    write_file_text(SRC_FILE, new_text)

    # Build
    build_project()

    # Run
    run_simulation()

    # Move output
    source = find_output_root()
    wait_for_root_close(source)
    out_name.parent.mkdir(parents=True, exist_ok=True)
    if out_name.exists():
        out_name.unlink()
    shutil.move(str(source), str(out_name))

    return out_name


# =============================================================================
# Gaussian Fits PDF Generation (from run_gaussian_fits.py)
# =============================================================================

DEFAULT_DISTANCE_ARGS = {
    "use_distance_weighted_errors": True,
    "distance_error_scale_pixels": 1.5,
    "distance_error_exponent": 1.5,
    "distance_error_floor_percent": 4.0,
    "distance_error_cap_percent": 10.0,
    "distance_error_prefer_truth_center": True,
    "distance_error_power_inverse": True,
}


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def generate_gaussian_fits_pdf(
    root_file: Path,
    output_pdf: Path,
    root_executable: str = "root",
    n_events: int = 200,
    error_percent: float = 0.0,
    use_qiqn_errors: bool = False,
    plot_qi_overlay: bool = True,
    do_qi_fit: bool = True,
    **distance_args,
) -> Optional[Path]:
    """Generate PDF of Gaussian fits using the ROOT macro."""
    if not MACRO_PATH.exists():
        print(f"[WARN] ROOT macro not found: {MACRO_PATH}")
        return None

    # Merge with defaults
    cfg = {**DEFAULT_DISTANCE_ARGS, **distance_args}

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    macro_call = (
        f"{MACRO_PATH.as_posix()}(\"{root_file.as_posix()}\", "
        f"{error_percent:.6f}, {n_events}, {_bool_str(plot_qi_overlay)}, {_bool_str(do_qi_fit)}, "
        f"{_bool_str(use_qiqn_errors)}, \"{output_pdf.as_posix()}\", "
        f"{_bool_str(cfg['use_distance_weighted_errors'])}, {cfg['distance_error_scale_pixels']:.6f}, "
        f"{cfg['distance_error_exponent']:.6f}, {cfg['distance_error_floor_percent']:.6f}, "
        f"{cfg['distance_error_cap_percent']:.6f}, {_bool_str(cfg['distance_error_prefer_truth_center'])}, "
        f"{_bool_str(cfg['distance_error_power_inverse'])})"
    )

    try:
        subprocess.run([root_executable, "-l", "-b", "-q", macro_call], check=True)
        return output_pdf
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] ROOT macro failed for {root_file} (exit code {exc.returncode})")
        return None


# =============================================================================
# Charge Neighborhood PDF Generation (from plotChargeNeighborhood.C)
# =============================================================================

def generate_neighborhood_pdf(
    root_file: Path,
    output_pdf: Path,
    quantity: str = "Qn",
    n_pages: int = 100,
    root_executable: str = "root",
) -> Optional[Path]:
    """Generate PDF of charge neighborhood visualizations using the ROOT macro.

    Args:
        root_file: Path to the ROOT file
        output_pdf: Path for the output PDF
        quantity: One of "Fi", "Qi", "Qn", "Qf", "distance"
        n_pages: Number of events to include
        root_executable: Path to ROOT executable

    Returns:
        Path to generated PDF, or None if failed
    """
    if not NEIGHBORHOOD_MACRO_PATH.exists():
        print(f"[WARN] Neighborhood macro not found: {NEIGHBORHOOD_MACRO_PATH}")
        return None

    # Map quantity to dataKind and chargeBranch
    if quantity not in NEIGHBORHOOD_TYPES:
        print(f"[WARN] Unknown neighborhood type '{quantity}', defaulting to 'Qn'")
        print(f"[INFO] Available types: {list(NEIGHBORHOOD_TYPES.keys())}")
        quantity = "Qn"

    data_kind, charge_branch = NEIGHBORHOOD_TYPES[quantity]

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Call plotChargeNeighborhood5x5_pages(rootFilePath, nPages, dataKind, outPdfPath, chargeBranch)
    macro_call = (
        f'{NEIGHBORHOOD_MACRO_PATH.as_posix()}('
        f'"{root_file.as_posix()}", '
        f'{n_pages}, '
        f'"{data_kind}", '
        f'"{output_pdf.as_posix()}", '
        f'"{charge_branch}")'
    )

    # We need to call the specific function
    full_call = f'plotChargeNeighborhood5x5_pages("{root_file.as_posix()}", {n_pages}, "{data_kind}", "{output_pdf.as_posix()}", "{charge_branch}")'

    try:
        # Load the macro and call the function
        cmd = [
            root_executable, "-l", "-b", "-q",
            f'{NEIGHBORHOOD_MACRO_PATH.as_posix()}+("{root_file.as_posix()}", {n_pages}, "{data_kind}", "{output_pdf.as_posix()}", "{charge_branch}")'
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if output_pdf.exists():
            return output_pdf
        else:
            print(f"[WARN] Neighborhood PDF not created: {output_pdf}")
            return None
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Neighborhood macro failed for {root_file}: {exc.stderr}")
        return None


def generate_all_neighborhood_pdfs(
    root_file: Path,
    output_dir: Path,
    quantities: Optional[List[str]] = None,
    n_pages: int = 100,
    root_executable: str = "root",
) -> Dict[str, Optional[Path]]:
    """Generate neighborhood PDFs for all specified quantities.

    Args:
        root_file: Path to the ROOT file
        output_dir: Directory for output PDFs
        quantities: List of quantities to generate (default: Fi, Qi, Qn, Qf, distance)
        n_pages: Number of events per PDF
        root_executable: Path to ROOT executable

    Returns:
        Dictionary mapping quantity name to output PDF path (or None if failed)
    """
    if quantities is None:
        quantities = ["Fi", "Qi", "Qn", "Qf", "distance"]

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for qty in quantities:
        pdf_name = f"neighborhood_{qty}_{root_file.stem}.pdf"
        pdf_path = output_dir / pdf_name
        print(f"[INFO] Generating neighborhood PDF for {qty}...")
        results[qty] = generate_neighborhood_pdf(
            root_file, pdf_path, qty, n_pages, root_executable
        )

    return results


# =============================================================================
# Sigma_f Extraction (from sigma_f_x.py)
# =============================================================================

def gaussian_with_const(x: np.ndarray, A: float, mu: float, sigma: float, B: float) -> np.ndarray:
    """1D Gaussian with constant offset."""
    return A * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-12)) ** 2) + B


def fit_gaussian_to_values(values_mm: np.ndarray, num_bins: int = NUM_BINS) -> GaussianFitResult:
    """Fit a Gaussian+constant to histogram of values."""
    data = np.asarray(values_mm, dtype=float)
    data = data[np.isfinite(data)]

    if data.size < 10:
        return GaussianFitResult(
            float("nan"), float("nan"), float("nan"),
            (0.0, 0.0, 0.0, 0.0), np.full((4, 4), np.nan),
            0, float("nan"), float("nan"), np.array([]), np.array([])
        )

    mean_data = float(np.mean(data))
    stdev_data = float(np.std(data, ddof=0))

    hist_lo, hist_hi = float(np.min(data)), float(np.max(data))
    hist_y, edges = np.histogram(data, bins=num_bins, range=(hist_lo, hist_hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    entries = int(np.sum(hist_y))
    mean_hist = float(np.sum(centers * hist_y) / entries) if entries > 0 else float("nan")
    var_hist = float(np.sum(hist_y * (centers - mean_hist) ** 2) / entries) if entries > 0 else float("nan")
    stdev_hist = float(math.sqrt(var_hist)) if np.isfinite(var_hist) and var_hist >= 0 else float("nan")

    # Initial guesses
    A0 = float(np.max(hist_y)) if np.max(hist_y) > 0 else 1.0
    mu0 = mean_hist
    sigma0 = stdev_hist if (np.isfinite(stdev_hist) and stdev_hist > 0) else 0.01
    B0 = float(np.min(hist_y)) if hist_y.size > 0 else 0.0

    weights = np.sqrt(np.maximum(hist_y, 1.0))

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
        return GaussianFitResult(mu_fit, sigma_fit, sigma_err, tuple(popt), pcov, entries, mean_hist, stdev_hist, centers, hist_y)
    except Exception:
        popt = (A0, mu0, sigma0, B0)
        pcov = np.full((4, 4), np.nan)
        return GaussianFitResult(mu0, sigma0, float("nan"), popt, pcov, entries, mean_hist, stdev_hist, centers, hist_y)


def save_sigma_f_plot(
    branch: str,
    x_um: float,
    fit: GaussianFitResult,
    out_png: Path,
    figsize: Tuple[int, int] = FIGSIZE,
    dpi: int = DPI,
) -> None:
    """Save histogram with Gaussian fit overlay for sigma_f."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.step(fit.hist_x, fit.hist_y, where="mid", color="#2b5fff", linewidth=1.2)

    if fit.hist_x.size > 0:
        xs = np.linspace(fit.hist_x.min(), fit.hist_x.max(), 800)
        ys = gaussian_with_const(xs, *fit.popt)
        ax.plot(xs, ys, color="red", linewidth=1.5)

    ax.set_title(f"{branch}")
    ax.set_xlabel("Delta X [mm]")
    ax.set_ylabel("Entries")

    text = (
        f"Entries: {fit.entries}\n"
        f"Mean: {fit.mean:+.5f}\n"
        f"Std Dev: {fit.stdev:.5f}\n"
        f"σ_f: {fit.sigma:.5f}"
    )
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )

    ax.text(0.01, 0.02, f"x = {x_um:.0f} um", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, alpha=0.7)

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def extract_sigma_f(
    root_file: Path,
    output_dir: Path,
    branches: List[str] = SIGMA_F_BRANCHES,
    num_bins: int = NUM_BINS,
    figsize: Tuple[int, int] = FIGSIZE,
    dpi: int = DPI,
) -> SigmaFResult:
    """Extract sigma_f from ReconTrueDelta distributions."""
    x_um = parse_x_um_from_filename(root_file)
    result = SigmaFResult(x_um=x_um)

    try:
        # Use MemmapSource for local files so ZSTD-compressed branches are readable
        # even if the filesystem backend selection changes.
        with uproot.open(root_file, handler=uproot.source.file.MemmapSource) as f:
            if "Hits" not in f:
                print(f"[WARN] Tree 'Hits' not found in {root_file}")
                return result

            tree = f["Hits"]
            available = set(tree.keys())

            for branch in branches:
                if branch not in available:
                    continue

                arr = tree[branch].array(library="np")
                arr = arr[np.isfinite(arr)]

                if arr.size == 0:
                    continue

                fit = fit_gaussian_to_values(arr, num_bins=num_bins)
                result.branch_results[branch] = fit

                # Save plot
                out_png = output_dir / "sigma_f_plots" / branch / f"{root_file.stem}.png"
                save_sigma_f_plot(branch, x_um, fit, out_png, figsize=figsize, dpi=dpi)

    except Exception as e:
        print(f"[ERROR] Failed to extract sigma_f from {root_file}: {e}")

    return result


# =============================================================================
# F_i Extraction (from Fi_x.py)
# =============================================================================

def infer_pixel_of_interest(
    root_files: Sequence[Path],
    requested_pixel_id: Optional[int],
    sample_events: int = 20000,
    step_size: str = "200 MB",
    prefer_right_pixel: bool = True,
) -> int:
    """Determine which global pixel ID to analyze.

    Args:
        root_files: List of ROOT files to sample from
        requested_pixel_id: If specified, use this pixel ID directly
        sample_events: Number of events to sample for inference
        step_size: Chunk size for reading ROOT files
        prefer_right_pixel: If True, prefer pixels with positive x-coordinate (right side)

    Returns:
        The pixel ID to use for analysis
    """
    if requested_pixel_id is not None:
        return int(requested_pixel_id)

    # Check for full-grid fractions and coordinates
    use_full_grid = False
    have_coords = False
    if root_files:
        try:
            with uproot.open(root_files[0], handler=uproot.source.file.MemmapSource) as first_file:
                if "Hits" in first_file:
                    key_set = {key.split(";")[0] for key in first_file["Hits"].keys()}
                    use_full_grid = {"F_all", "F_all_pixel_id"}.issubset(key_set)
                    have_coords = {"F_all_pixel_x"}.issubset(key_set)
        except Exception:
            use_full_grid = False

    accum_sum: Dict[int, float] = {}
    accum_count: Dict[int, int] = {}
    pixel_x_coords: Dict[int, float] = {}  # Store x-coordinates for each pixel
    full_sum: Optional[np.ndarray] = None
    full_count: Optional[np.ndarray] = None
    full_pixel_ids: Optional[np.ndarray] = None
    full_pixel_x: Optional[np.ndarray] = None

    # Prioritize files near x=0
    prioritized = sorted(
        root_files,
        key=lambda p: (
            math.isnan(parse_x_um_from_filename(p)),
            abs(parse_x_um_from_filename(p)) if not math.isnan(parse_x_um_from_filename(p)) else float("inf"),
        ),
    )

    remaining = sample_events
    for path in prioritized:
        if remaining <= 0:
            break
        try:
            with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
                if "Hits" not in f:
                    continue
                tree = f["Hits"]

                if use_full_grid:
                    branches_to_read = ["F_all", "F_all_pixel_id"]
                    if have_coords:
                        branches_to_read.append("F_all_pixel_x")
                    for arrays in tree.iterate(branches_to_read, step_size=step_size, library="ak"):
                        fractions = ak.Array(arrays["F_all"])
                        n_events = len(fractions)
                        if n_events == 0:
                            continue
                        if remaining < n_events:
                            fractions = fractions[:remaining]
                            n_events = len(fractions)
                        remaining -= n_events
                        if n_events == 0:
                            break
                        np_fractions = ak.to_numpy(fractions)
                        if np_fractions.ndim == 1:
                            np_fractions = np_fractions[np.newaxis, :]
                        if full_pixel_ids is None:
                            raw_ids = ak.to_numpy(arrays["F_all_pixel_id"][0])
                            full_pixel_ids = np.asarray(raw_ids, dtype=int)
                            full_sum = np.zeros_like(full_pixel_ids, dtype=float)
                            full_count = np.zeros_like(full_pixel_ids, dtype=int)
                            if have_coords:
                                raw_x = ak.to_numpy(arrays["F_all_pixel_x"][0])
                                full_pixel_x = np.asarray(raw_x, dtype=float)
                        cols = min(np_fractions.shape[1], full_pixel_ids.size)
                        if cols == 0:
                            continue
                        np_fractions = np.asarray(np_fractions[:, :cols], dtype=float)
                        np.nan_to_num(np_fractions, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        full_sum[:cols] += np.sum(np_fractions, axis=0)
                        full_count[:cols] += n_events
                        if remaining <= 0:
                            break
                else:
                    # Check if we have NeighborhoodPixelX
                    tree_keys = {key.split(";")[0] for key in tree.keys()}
                    have_neighborhood_x = "NeighborhoodPixelX" in tree_keys
                    neighborhood_branches = ["Fi", "NeighborhoodPixelID"]
                    if have_neighborhood_x:
                        neighborhood_branches.append("NeighborhoodPixelX")

                    for arrays in tree.iterate(neighborhood_branches, step_size=step_size, library="ak"):
                        fi = ak.Array(arrays["Fi"])
                        ids = ak.Array(arrays["NeighborhoodPixelID"])

                        if remaining < len(fi):
                            fi = fi[:remaining]
                            ids = ids[:remaining]
                        remaining -= len(fi)

                        fi = ak.fill_none(fi, np.nan)
                        ids = ak.fill_none(ids, -1)

                        mask_pixel = ids >= 0
                        if not ak.any(mask_pixel, axis=None):
                            if remaining <= 0:
                                break
                            continue

                        fi_pixel = fi[mask_pixel]
                        ids_pixel = ids[mask_pixel]

                        fi_flat = ak.to_numpy(ak.flatten(fi_pixel, axis=None))
                        ids_flat = ak.to_numpy(ak.flatten(ids_pixel, axis=None)).astype(int)

                        # Also get x-coordinates if available
                        if have_neighborhood_x:
                            px = ak.fill_none(ak.Array(arrays["NeighborhoodPixelX"]), np.nan)
                            px_pixel = px[mask_pixel]
                            px_flat = ak.to_numpy(ak.flatten(px_pixel, axis=None))

                        if fi_flat.size == 0:
                            if remaining <= 0:
                                break
                            continue

                        finite_mask = np.isfinite(fi_flat)
                        if not finite_mask.any():
                            if remaining <= 0:
                                break
                            continue

                        fi_flat = fi_flat[finite_mask]
                        ids_flat = ids_flat[finite_mask]
                        if have_neighborhood_x:
                            px_flat = px_flat[finite_mask]

                        valid_mask = (fi_flat != SENTINEL_INVALID_FRACTION) & (fi_flat >= 0.0)
                        if not valid_mask.any():
                            if remaining <= 0:
                                break
                            continue

                        fi_flat = fi_flat[valid_mask]
                        ids_flat = ids_flat[valid_mask]
                        if have_neighborhood_x:
                            px_flat = px_flat[valid_mask]

                        for i, (pid, val) in enumerate(zip(ids_flat, fi_flat)):
                            accum_sum[pid] = accum_sum.get(pid, 0.0) + float(val)
                            accum_count[pid] = accum_count.get(pid, 0) + 1
                            # Store x-coordinate (use first seen value)
                            if have_neighborhood_x and pid not in pixel_x_coords:
                                pixel_x_coords[pid] = float(px_flat[i])

                        if remaining <= 0:
                            break
        except Exception as exc:
            print(f"[WARN] Failed to inspect '{path}': {exc}")

    if use_full_grid:
        if full_pixel_ids is None or full_sum is None or full_count is None:
            raise RuntimeError("Unable to infer pixel of interest from full-grid fractions.")
        valid_mask = full_count > 0
        if not np.any(valid_mask):
            raise RuntimeError("No valid full-grid samples encountered.")
        means = np.full_like(full_sum, fill_value=-np.inf, dtype=float)
        means[valid_mask] = full_sum[valid_mask] / full_count[valid_mask]

        # If prefer_right_pixel and we have coordinates, filter to positive x
        if prefer_right_pixel and full_pixel_x is not None:
            right_mask = full_pixel_x > 0
            combined_mask = valid_mask & right_mask
            if np.any(combined_mask):
                # Find best among right-side pixels
                means_right = np.full_like(means, fill_value=-np.inf)
                means_right[combined_mask] = means[combined_mask]
                best_index = int(np.argmax(means_right))
                print(f"[INFO] Selected right-side pixel at x={full_pixel_x[best_index]*1000:.1f} µm")
                return int(full_pixel_ids[best_index])
            else:
                print("[WARN] No right-side pixels found, selecting from all pixels")

        best_index = int(np.argmax(means))
        return int(full_pixel_ids[best_index])

    if not accum_count:
        raise RuntimeError("Unable to infer pixel of interest: no valid F_i entries.")

    # If prefer_right_pixel and we have coordinates, filter to positive x
    if prefer_right_pixel and pixel_x_coords:
        right_pids = [pid for pid in accum_sum.keys() if pixel_x_coords.get(pid, 0) > 0]
        if right_pids:
            best_pid = max(right_pids, key=lambda pid: accum_sum[pid] / accum_count[pid])
            print(f"[INFO] Selected right-side pixel at x={pixel_x_coords[best_pid]*1000:.1f} µm")
            return int(best_pid)
        else:
            print("[WARN] No right-side pixels found, selecting from all pixels")

    best_pid = max(accum_sum.keys(), key=lambda pid: accum_sum[pid] / accum_count[pid])
    return int(best_pid)


def extract_fi_from_file(
    root_file: Path,
    pixel_id: int,
    step_size: str = "200 MB",
) -> FiResult:
    """Extract F_i statistics for a specific pixel from a ROOT file."""
    x_um = parse_x_um_from_filename(root_file)

    with uproot.open(root_file, handler=uproot.source.file.MemmapSource) as f:
        if "Hits" not in f:
            raise RuntimeError(f"Tree 'Hits' not found in {root_file}")

        tree = f["Hits"]
        tree_keys = {key.split(";")[0] for key in tree.keys()}
        use_full_grid = {"F_all", "F_all_pixel_id"}.issubset(tree_keys)
        have_full_coords = {"F_all_pixel_x", "F_all_pixel_y"}.issubset(tree_keys)

        values_sum = 0.0
        values_sumsq = 0.0
        values_count = 0
        coord_mm: Optional[Tuple[float, float]] = None

        if use_full_grid:
            branch_names = ["F_all", "F_all_pixel_id"]
            if have_full_coords:
                branch_names.extend(["F_all_pixel_x", "F_all_pixel_y"])

            mapping_idx: Optional[int] = None
            full_ids: Optional[np.ndarray] = None

            for arrays in tree.iterate(branch_names, step_size=step_size, library="ak"):
                fractions = ak.Array(arrays["F_all"])
                n_events = len(fractions)
                if n_events == 0:
                    continue
                np_fractions = ak.to_numpy(fractions)
                if np_fractions.ndim == 1:
                    np_fractions = np_fractions[np.newaxis, :]
                if full_ids is None:
                    raw_ids = ak.to_numpy(arrays["F_all_pixel_id"][0])
                    full_ids = np.asarray(raw_ids, dtype=int)
                    matches = np.where(full_ids == pixel_id)[0]
                    if matches.size == 0:
                        continue
                    mapping_idx = int(matches[0])
                    if have_full_coords and coord_mm is None:
                        px_vals = ak.to_numpy(arrays["F_all_pixel_x"][0])
                        py_vals = ak.to_numpy(arrays["F_all_pixel_y"][0])
                        if mapping_idx < len(px_vals) and mapping_idx < len(py_vals):
                            coord_mm = (float(px_vals[mapping_idx]), float(py_vals[mapping_idx]))
                if mapping_idx is None:
                    continue
                pixel_vals = np.asarray(np_fractions[:, mapping_idx], dtype=float)
                finite_mask = np.isfinite(pixel_vals)
                if not finite_mask.any():
                    continue
                pixel_vals = pixel_vals[finite_mask]
                valid_mask = pixel_vals >= 0.0
                if not valid_mask.any():
                    continue
                pixel_vals = pixel_vals[valid_mask]
                if pixel_vals.size == 0:
                    continue
                values_sum += float(np.sum(pixel_vals))
                values_sumsq += float(np.sum(pixel_vals * pixel_vals))
                values_count += int(pixel_vals.size)
        else:
            for arrays in tree.iterate(
                ["Fi", "NeighborhoodPixelID", "NeighborhoodPixelX", "NeighborhoodPixelY"],
                step_size=step_size,
                library="ak",
            ):
                fi = ak.fill_none(ak.Array(arrays["Fi"]), np.nan)
                ids = ak.fill_none(ak.Array(arrays["NeighborhoodPixelID"]), -1)

                mask_pixel = ids == pixel_id
                if not ak.any(mask_pixel, axis=None):
                    continue

                fi_selected = fi[mask_pixel]
                fi_flat = ak.to_numpy(ak.flatten(fi_selected, axis=None)).astype(float)
                if fi_flat.size == 0:
                    continue

                finite_mask = np.isfinite(fi_flat)
                if not finite_mask.any():
                    continue
                fi_flat = fi_flat[finite_mask]

                valid_mask = (fi_flat != SENTINEL_INVALID_FRACTION) & (fi_flat >= 0.0)
                if not valid_mask.any():
                    continue
                fi_flat = fi_flat[valid_mask]

                values_sum += float(np.sum(fi_flat))
                values_sumsq += float(np.sum(fi_flat * fi_flat))
                values_count += int(fi_flat.size)

    count = values_count
    mean = values_sum / count if count > 0 else float("nan")

    if count > 1:
        variance = (values_sumsq - count * mean * mean) / (count - 1)
        variance = max(variance, 0.0)
        std = math.sqrt(variance)
    else:
        std = float("nan")

    return FiResult(
        x_um=x_um,
        pixel_id=pixel_id,
        mean=mean,
        std=std,
        count=count,
        coord_mm=coord_mm,
    )


# =============================================================================
# Summary Plots
# =============================================================================

# Default pixel geometry for plots (can be overridden)
DEFAULT_PIXEL_SPACING_MM = 0.5   # 500 µm
DEFAULT_PIXEL_SIZE_MM = 0.15    # 150 µm


def draw_pixel_regions(
    ax: plt.Axes,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
    x_range_mm: Tuple[float, float] = (-0.8, 0.8),
    color: str = "#d0d0d0",
    alpha: float = 0.6,
) -> None:
    """Draw gray vertical bands for pixel pad locations (FNAL paper style).

    Assumes pixels are centered at 0, ±pixel_spacing_mm, ±2*pixel_spacing_mm, etc.
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


def add_fi_in_pixel_regions(
    x_vals_mm: np.ndarray,
    fi_vals: np.ndarray,
    pixel_spacing_mm: float,
    pixel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add points with F_i = 1.0 inside pixel regions.

    When a track hits directly on the metal pad, all charge goes to that pixel,
    so F_i = 1.0.

    Returns:
        Tuple of (x_vals_mm, fi_vals) with inserted pixel region points.
    """
    if len(x_vals_mm) == 0:
        return x_vals_mm, fi_vals

    x_min, x_max = np.nanmin(x_vals_mm), np.nanmax(x_vals_mm)
    boundaries = get_pixel_boundaries(pixel_spacing_mm, pixel_size_mm, (x_min, x_max))

    if not boundaries:
        return x_vals_mm, fi_vals

    # Build new arrays with inserted pixel region points
    new_x = list(x_vals_mm)
    new_fi = list(fi_vals)

    # Small offset to place points just inside pixel boundaries
    eps = 0.001  # 1 µm in mm

    for left, right in boundaries:
        # Add points just inside the pixel boundaries with F_i = 1.0
        left_inside = left + eps
        right_inside = right - eps

        # Only add if within data range and not duplicating existing points
        if left_inside >= x_min and left_inside <= x_max:
            if not any(abs(x - left_inside) < eps * 2 for x in new_x):
                new_x.append(left_inside)
                new_fi.append(1.0)

        if right_inside >= x_min and right_inside <= x_max:
            if not any(abs(x - right_inside) < eps * 2 for x in new_x):
                new_x.append(right_inside)
                new_fi.append(1.0)

    # Sort by x position
    sort_idx = np.argsort(new_x)
    return np.array(new_x)[sort_idx], np.array(new_fi)[sort_idx]


def save_pixel_grid_visualization(
    measurement_positions_um: Sequence[float],
    output_dir: Path,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
    n_cols: int = 3,
    n_rows: int = 2,
) -> Path:
    """Create a visualization of the pixel grid showing measurement positions.

    Shows the actual detector geometry with:
    - Metal pads as gray rectangles
    - Gap regions between pads
    - Red dots marking where measurements are taken
    - Proper scale with dimensions in µm

    Args:
        measurement_positions_um: List of x positions in µm where measurements are taken
        output_dir: Directory to save the output image
        pixel_spacing_mm: Distance between pixel centers in mm (default: 0.5 mm = 500 µm)
        pixel_size_mm: Size of metal pads in mm (default: 0.15 mm = 150 µm)
        n_cols: Number of pixel columns (default: 3)
        n_rows: Number of pixel rows (default: 2)

    Returns:
        Path to the saved image
    """
    ensure_dir(output_dir)

    # Convert to µm for easier visualization
    pixel_spacing_um = pixel_spacing_mm * 1000.0
    pixel_size_um = pixel_size_mm * 1000.0
    half_pad = pixel_size_um / 2.0

    # Calculate grid extent
    half_cols = (n_cols - 1) // 2
    half_rows = (n_rows - 1) // 2
    x_centers = [i * pixel_spacing_um for i in range(-half_cols, half_cols + 1)]
    y_centers = [j * pixel_spacing_um for j in range(-half_rows, half_rows + 1)]

    # Calculate figure bounds (add margin)
    margin_um = 100.0
    x_min = x_centers[0] - pixel_spacing_um / 2 - margin_um
    x_max = x_centers[-1] + pixel_spacing_um / 2 + margin_um
    y_min = y_centers[0] - pixel_spacing_um / 2 - margin_um
    y_max = y_centers[-1] + pixel_spacing_um / 2 + margin_um

    # Create figure with proper aspect ratio
    width_um = x_max - x_min
    height_um = y_max - y_min
    fig_width = 10
    fig_height = fig_width * height_um / width_um

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)

    # Draw pixel boundaries (dashed lines)
    for x in x_centers:
        ax.axvline(x - pixel_spacing_um / 2, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x + pixel_spacing_um / 2, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    for y in y_centers:
        ax.axhline(y - pixel_spacing_um / 2, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(y + pixel_spacing_um / 2, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    # Draw metal pads as gray rectangles
    from matplotlib.patches import Rectangle
    for x in x_centers:
        for y in y_centers:
            rect = Rectangle(
                (x - half_pad, y - half_pad),
                pixel_size_um,
                pixel_size_um,
                linewidth=1,
                edgecolor='#404040',
                facecolor='#d0d0d0',
                zorder=1,
            )
            ax.add_patch(rect)

    # Highlight the row where measurements are taken (y=0, the "pad row")
    # Draw a light red band across the measurement row
    measurement_row_y = 0  # Assuming measurements are in the row at y=0
    row_band_height = pixel_spacing_um
    ax.axhspan(
        measurement_row_y - row_band_height / 2,
        measurement_row_y + row_band_height / 2,
        color='#ffcccc',
        alpha=0.3,
        zorder=0,
        label='Measurement row'
    )

    # Draw gap regions (columns between pads) with light blue
    for i in range(len(x_centers) - 1):
        gap_left = x_centers[i] + half_pad
        gap_right = x_centers[i + 1] - half_pad
        ax.axvspan(
            gap_left, gap_right,
            color='#cce5ff',
            alpha=0.4,
            zorder=0,
        )

    # Draw measurement positions as red dots
    measurement_y = 0  # All measurements are at y=0 (upper pad row)
    xs = np.array(measurement_positions_um)
    ys = np.full_like(xs, measurement_y)
    ax.scatter(xs, ys, c='red', s=30, zorder=3, label='Measurement positions', edgecolors='darkred', linewidths=0.5)

    # Add dimension annotations
    # Show pixel pitch
    ax.annotate(
        '', xy=(x_centers[0], y_max - 30), xytext=(x_centers[1], y_max - 30),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
    )
    ax.text((x_centers[0] + x_centers[1]) / 2, y_max - 10, f'Pitch: {pixel_spacing_um:.0f} µm',
            ha='center', va='bottom', fontsize=9)

    # Show pad size
    ax.annotate(
        '', xy=(x_centers[1] - half_pad, y_centers[-1] + half_pad + 20),
        xytext=(x_centers[1] + half_pad, y_centers[-1] + half_pad + 20),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
    )
    ax.text(x_centers[1], y_centers[-1] + half_pad + 40, f'Pad: {pixel_size_um:.0f} µm',
            ha='center', va='bottom', fontsize=9)

    # Axis settings
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x position [µm]', fontsize=11)
    ax.set_ylabel('y position [µm]', fontsize=11)
    ax.set_title('Pixel Grid with Measurement Positions', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(False)

    # Legend
    ax.legend(loc='upper left', fontsize=9)

    # Add text annotation for number of positions
    n_positions = len(measurement_positions_um)
    ax.text(
        0.98, 0.02,
        f'{n_positions} measurement positions\nin gap regions',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    fig.tight_layout()
    out_path = output_dir / "pixel_grid_with_measurements.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return out_path


def save_sigma_f_vs_x_plot(
    results: List[PositionResult],
    output_dir: Path,
    branch: str = "ReconTrueDeltaRowX",
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
) -> Path:
    """FNAL paper-style plot of sigma_f vs track x position with gray pixel bands.

    Shows binary resolution (pixel_size/sqrt(12)) inside pixel regions where
    no measurements exist, and uses data-driven y-axis limits.
    """
    xs_um = []
    ys = []

    for r in results:
        if r.sigma_f is None or branch not in r.sigma_f.branch_results:
            continue
        fit = r.sigma_f.branch_results[branch]
        if not np.isfinite(fit.sigma):
            continue
        xs_um.append(r.x_um)
        ys.append(fit.sigma * 1000.0)  # mm -> um

    if not xs_um:
        print(f"[WARN] No valid sigma_f data for {branch}")
        return output_dir / f"sigma_f_vs_x_{branch}.png"

    # Sort by x position
    sort_idx = np.argsort(xs_um)
    xs_um = np.array(xs_um)[sort_idx]
    ys = np.array(ys)[sort_idx]
    xs_mm = xs_um / 1000.0

    # Add binary resolution points inside pixel regions
    xs_mm_extended, ys_extended = add_binary_resolution_in_pixel_regions(
        xs_mm, ys, pixel_spacing_mm, pixel_size_mm
    )

    # FNAL paper-style figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    # Determine x range
    x_min = np.nanmin(xs_mm) - 0.05
    x_max = np.nanmax(xs_mm) + 0.05

    # Draw gray pixel regions first (behind the data)
    draw_pixel_regions(ax, pixel_spacing_mm, pixel_size_mm, x_range_mm=(x_min, x_max))

    # Plot as histogram-style step (FNAL paper style) with extended data
    ax.step(xs_mm_extended, ys_extended, where="mid", color="#1f77b4", lw=2, zorder=2, label="Simulation")

    # Add square markers at measurement points (FNAL paper style)
    ax.scatter(xs_mm, ys, marker='s', s=40, color="#1f77b4",
               edgecolors='black', linewidths=0.5, zorder=4)

    # FNAL paper-style axis labels
    ax.set_xlabel("Track x position [mm]", fontsize=12)
    ax.set_ylabel(r"Position resolution [µm]", fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Y-axis starts from 0, extends above max with margin
    y_max_data = np.nanmax(ys_extended)
    ax.set_ylim(0, y_max_data * 1.1)

    # Minimal grid
    ax.grid(True, alpha=0.3, zorder=1, linestyle='-')
    ax.tick_params(axis='both', which='major', labelsize=10)

    branch_name = branch.replace("ReconTrueDelta", "").replace("_2D", " (2D)")
    ax.set_title(f"Position resolution vs track position - {branch_name}", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    ensure_dir(output_dir)
    out_path = output_dir / f"sigma_f_vs_x_{branch}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def save_fi_vs_x_plot(
    results: List[PositionResult],
    output_dir: Path,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
) -> Path:
    """FNAL paper-style plot of F_i vs track x position with gray pixel bands.

    Shows F_i = 1.0 inside pixel regions where all charge goes to that pixel.
    """
    xs_um = []
    ys = []

    for r in results:
        if r.fi is None or not np.isfinite(r.fi.mean):
            continue
        xs_um.append(r.x_um)
        ys.append(r.fi.mean)

    if not xs_um:
        print("[WARN] No valid F_i data")
        return output_dir / "fi_vs_x.png"

    # Sort by x position
    sort_idx = np.argsort(xs_um)
    xs_um = np.array(xs_um)[sort_idx]
    ys = np.array(ys)[sort_idx]
    xs_mm = xs_um / 1000.0

    # Add F_i = 1.0 points inside pixel regions
    xs_mm_extended, ys_extended = add_fi_in_pixel_regions(
        xs_mm, ys, pixel_spacing_mm, pixel_size_mm
    )

    # FNAL paper-style figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    # Determine x range
    x_min = np.nanmin(xs_mm) - 0.05
    x_max = np.nanmax(xs_mm) + 0.05

    # Draw gray pixel regions first (behind the data)
    draw_pixel_regions(ax, pixel_spacing_mm, pixel_size_mm, x_range_mm=(x_min, x_max))

    # Plot as histogram-style step (FNAL paper style) with extended data
    ax.step(xs_mm_extended, ys_extended, where="mid", color="#1f77b4", lw=2, zorder=2, label="Simulation")

    # Add square markers at measurement points (FNAL paper style)
    ax.scatter(xs_mm, ys, marker='s', s=40, color="#1f77b4",
               edgecolors='black', linewidths=0.5, zorder=4)

    # FNAL paper-style axis labels
    ax.set_xlabel("Track x position [mm]", fontsize=12)
    ax.set_ylabel(r"Signal fraction $F_i$", fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Data-driven y-axis limits with margin (F_i bounded by 0 and 1)
    y_min_data = np.nanmin(ys_extended)
    y_max_data = np.nanmax(ys_extended)
    y_margin = (y_max_data - y_min_data) * 0.1
    ax.set_ylim(max(0, y_min_data - y_margin), min(1.05, y_max_data + y_margin))

    # Minimal grid
    ax.grid(True, alpha=0.3, zorder=1, linestyle='-')
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_title("Signal fraction vs track position", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    ensure_dir(output_dir)
    out_path = output_dir / "fi_vs_x.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def save_sigma_f_vs_d_plot(
    results: List[PositionResult],
    output_dir: Path,
    branch: str = "ReconTrueDeltaRowX",
) -> Path:
    """Plot sigma_f vs distance for a specific branch.

    Distance is signed: d = x_particle_gun - x_pixel_center
    (Tornago et al. convention).
    """
    xs = []
    ys = []

    for r in results:
        if r.sigma_f is None or branch not in r.sigma_f.branch_results:
            continue
        fit = r.sigma_f.branch_results[branch]
        if not np.isfinite(fit.sigma):
            continue
        if r.distance_to_reference_pixel_um is None:
            continue
        xs.append(r.distance_to_reference_pixel_um)
        ys.append(fit.sigma * 1000.0)  # mm -> um

    if not xs:
        print(f"[WARN] No valid sigma_f data for {branch}")
        return output_dir / f"sigma_f_vs_d_{branch}.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    ax.scatter(xs, ys, color="#1f77b4", s=50, alpha=0.8)
    ax.set_xlabel(r"$d = x_{gun} - x_{pixel}$ [µm]")
    ax.set_ylabel(r"$\sigma_F$ [µm]")
    ax.set_title(f"$\\sigma_F$ vs distance ({branch})")
    ax.grid(True, linestyle="--", alpha=0.35)

    ensure_dir(output_dir)
    out_path = output_dir / f"sigma_f_vs_d_{branch}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def save_fi_vs_d_plot(
    results: List[PositionResult],
    output_dir: Path,
) -> Path:
    """Plot F_i vs distance in the style of Tornago et al. (Figure 5).

    Reference: M. Tornago et al., "Resistive AC-Coupled Silicon Detectors:
    principles of operation and first results from a combined analysis of
    beam test and laser data", arXiv:2007.09528v4

    The plot shows "Fractional amplitude seen in a pad as a function of the hit distance"
    with y-axis ranging from 0 to 1.

    Distance is signed: d = x_particle_gun - x_pixel_center
    """
    xs = []
    ys = []

    for r in results:
        if r.fi is None:
            continue
        if not np.isfinite(r.fi.mean):
            continue
        if r.distance_to_reference_pixel_um is None:
            continue
        xs.append(r.distance_to_reference_pixel_um)
        ys.append(r.fi.mean)

    if not xs:
        print("[WARN] No valid F_i data")
        return output_dir / "fi_vs_d.png"

    # Create figure in Tornago paper style
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

    # Scatter plot (no connecting lines, no error bars)
    ax.scatter(xs, ys, color='#2b5fff', s=50, alpha=0.8, label='Simulation')

    # Style matching Tornago paper Figure 5
    ax.set_xlabel(r"$d = x_{gun} - x_{pixel}$ [µm]", fontsize=12)
    ax.set_ylabel("Fractional amplitude", fontsize=12)
    ax.set_title("Fractional amplitude seen in a pad as a function of the hit distance",
                 fontsize=11)

    # Set y-axis to match paper (0 to 1)
    ax.set_ylim(0, 1.05)

    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add legend if we have data
    if len(xs) > 0:
        ax.legend(loc='upper right', fontsize=10)

    ensure_dir(output_dir)
    out_path = output_dir / "fi_vs_d.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def save_combined_summary_plot(
    results: List[PositionResult],
    output_dir: Path,
    sigma_f_branch: str = "ReconTrueDeltaRowX",
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM,
) -> Path:
    """FNAL paper-style combined plot with sigma_f and F_i vs track position.

    Both plots use gray pixel bands and step/histogram style.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)

    # Collect sigma_f data
    xs_sigma_um = []
    ys_sigma = []
    for r in results:
        if r.sigma_f is None or sigma_f_branch not in r.sigma_f.branch_results:
            continue
        fit = r.sigma_f.branch_results[sigma_f_branch]
        if not np.isfinite(fit.sigma):
            continue
        xs_sigma_um.append(r.x_um)
        ys_sigma.append(fit.sigma * 1000.0)

    # Collect F_i data
    xs_fi_um = []
    ys_fi = []
    for r in results:
        if r.fi is None or not np.isfinite(r.fi.mean):
            continue
        xs_fi_um.append(r.x_um)
        ys_fi.append(r.fi.mean)

    # Determine x range from all data
    all_xs_um = xs_sigma_um + xs_fi_um
    if all_xs_um:
        all_xs_mm = np.array(all_xs_um) / 1000.0
        x_min = np.nanmin(all_xs_mm) - 0.05
        x_max = np.nanmax(all_xs_mm) + 0.05
    else:
        x_min, x_max = -0.5, 0.5

    # sigma_f plot
    draw_pixel_regions(ax1, pixel_spacing_mm, pixel_size_mm, x_range_mm=(x_min, x_max))

    if xs_sigma_um:
        sort_idx = np.argsort(xs_sigma_um)
        xs_mm = np.array(xs_sigma_um)[sort_idx] / 1000.0
        ys = np.array(ys_sigma)[sort_idx]
        ax1.step(xs_mm, ys, where="mid", color="#1f77b4", lw=2, zorder=2, label="Simulation")

    ax1.set_xlabel("Track x position [mm]", fontsize=12)
    ax1.set_ylabel(r"Position resolution [µm]", fontsize=12)
    ax1.set_xlim(x_min, x_max)
    y_max = max(ys_sigma) * 1.15 if ys_sigma else 100
    ax1.set_ylim(0, max(y_max, 80))
    ax1.grid(True, linestyle="-", alpha=0.3, zorder=1)
    branch_name = sigma_f_branch.replace("ReconTrueDelta", "").replace("_2D", " (2D)")
    ax1.set_title(f"Position resolution - {branch_name}", fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)

    # F_i plot
    draw_pixel_regions(ax2, pixel_spacing_mm, pixel_size_mm, x_range_mm=(x_min, x_max))

    if xs_fi_um:
        sort_idx = np.argsort(xs_fi_um)
        xs_mm = np.array(xs_fi_um)[sort_idx] / 1000.0
        ys = np.array(ys_fi)[sort_idx]
        ax2.step(xs_mm, ys, where="mid", color="#1f77b4", lw=2, zorder=2, label="Simulation")

    ax2.set_xlabel("Track x position [mm]", fontsize=12)
    ax2.set_ylabel(r"Signal fraction $F_i$", fontsize=12)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="-", alpha=0.3, zorder=1)
    ax2.set_title("Signal fraction vs track position", fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)

    ensure_dir(output_dir)
    out_path = output_dir / "summary_sigma_f_and_fi_vs_x.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def write_summary_excel(
    results: List[PositionResult],
    output_dir: Path,
    sigma_f_branch: str = "ReconTrueDeltaRowX",
) -> Path:
    """Write summary Excel file with all results.

    Distance d is measured from particle gun position to reference pixel center:
    d = |x_particle_gun - x_pixel_center| (Tornago et al. convention).
    """
    records = []

    for r in sorted(results, key=lambda x: x.x_um):
        # Use distance to reference pixel if available, otherwise fall back to abs(x_um)
        distance = r.distance_to_reference_pixel_um if r.distance_to_reference_pixel_um is not None else abs(r.x_um)
        record = {
            "x_gun (um)": r.x_um,
            "d (um)": distance,
            "ROOT file": r.root_file.name if r.root_file else "",
        }

        # sigma_f data
        if r.sigma_f and sigma_f_branch in r.sigma_f.branch_results:
            fit = r.sigma_f.branch_results[sigma_f_branch]
            record["sigma_f (um)"] = fit.sigma * 1000.0
            record["d_sigma_f (um)"] = fit.sigma_err * 1000.0 if np.isfinite(fit.sigma_err) else float("nan")
        else:
            record["sigma_f (um)"] = float("nan")
            record["d_sigma_f (um)"] = float("nan")

        # F_i data
        if r.fi:
            record["F_i mean"] = r.fi.mean
            record["F_i std"] = r.fi.std
            record["F_i count"] = r.fi.count
            record["pixel_id"] = r.fi.pixel_id
        else:
            record["F_i mean"] = float("nan")
            record["F_i std"] = float("nan")
            record["F_i count"] = 0
            record["pixel_id"] = -1

        records.append(record)

    df = pd.DataFrame(records)
    ensure_dir(output_dir)
    out_path = output_dir / "sweep_analysis_summary.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)

    return out_path


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(
    positions: List[float],
    output_dir: Path,
    run_simulations: bool = True,
    root_executable: str = "root",
    n_fit_events: int = 200,
    pixel_id: Optional[int] = None,
    skip_gaussian_fits: bool = False,
    skip_sigma_f: bool = False,
    skip_fi: bool = False,
    charge_model: Optional[str] = None,
    active_pixel_mode: Optional[str] = None,
    beta: Optional[float] = None,
) -> List[PositionResult]:
    """Run the complete analysis pipeline.

    Args:
        positions: List of x positions in micrometers
        output_dir: Output directory for all results
        run_simulations: Whether to run new simulations or use existing files
        root_executable: Path to ROOT executable
        n_fit_events: Number of events for Gaussian fit PDFs
        pixel_id: Specific pixel ID to track (auto-detected if None)
        skip_gaussian_fits: Skip generating Gaussian fits PDFs
        skip_sigma_f: Skip sigma_f extraction
        skip_fi: Skip F_i extraction
        charge_model: Charge sharing model ("LogA" or "LinA")
        active_pixel_mode: Signal fraction mode ("Neighborhood", "ChargeBlock2x2", "ChargeBlock3x3", "RowCol", or "RowCol3x3")
        beta: Beta attenuation coefficient for linear model
    """
    results: List[PositionResult] = []

    ensure_dir(output_dir)
    root_files_dir = output_dir / "root_files"
    ensure_dir(root_files_dir)

    # Step 1: Run simulations (if enabled)
    if run_simulations:
        print("\n" + "=" * 80)
        print("STEP 1: Running simulations")
        print("=" * 80)

        os.chdir(str(REPO_ROOT))

        # Update Config.hh with simulation parameters if specified
        original_config_text = read_file_text(CONFIG_FILE)
        config_updated = False

        if any([charge_model, active_pixel_mode, beta]):
            print("\n[INFO] Updating simulation configuration...")
            current_settings = get_current_config_settings(original_config_text)
            print(f"[INFO] Current settings: {current_settings}")

            new_config_text = update_config_hh(
                original_config_text,
                charge_model=charge_model,
                active_pixel_mode=active_pixel_mode,
                beta=beta,
            )

            if new_config_text != original_config_text:
                write_file_text(CONFIG_FILE, new_config_text)
                config_updated = True
                print("[INFO] Config.hh updated, rebuild required")

        configure_build_if_needed()
        original_text = read_file_text(SRC_FILE)

        try:
            for x_um in positions:
                root_file = run_simulation_for_position(x_um, root_files_dir, original_text)
                results.append(PositionResult(x_um=x_um, root_file=root_file))
        finally:
            # Restore original source files
            try:
                write_file_text(SRC_FILE, original_text)
            except Exception as e:
                print(f"[WARN] Failed to restore original PrimaryGenerator.cc: {e}")

            # Restore original Config.hh if it was modified
            if config_updated:
                try:
                    write_file_text(CONFIG_FILE, original_config_text)
                    print("[INFO] Restored original Config.hh")
                except Exception as e:
                    print(f"[WARN] Failed to restore original Config.hh: {e}")
    else:
        # Load existing ROOT files
        print("\n" + "=" * 80)
        print("STEP 1: Loading existing ROOT files")
        print("=" * 80)

        root_files = sorted(root_files_dir.glob("*.root"))
        if not root_files:
            raise RuntimeError(f"No ROOT files found in {root_files_dir}")

        for rf in root_files:
            x_um = parse_x_um_from_filename(rf)
            results.append(PositionResult(x_um=x_um, root_file=rf))

        print(f"Found {len(results)} ROOT files")

    # Step 2: Generate Gaussian fits PDFs
    if not skip_gaussian_fits:
        print("\n" + "=" * 80)
        print("STEP 2: Generating Gaussian fits PDFs")
        print("=" * 80)

        gaussian_fits_dir = output_dir / "gaussian_fits"
        ensure_dir(gaussian_fits_dir)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Generating Gaussian fits PDF for {r.root_file.name}...")
            pdf_path = gaussian_fits_dir / f"{r.root_file.stem}.pdf"
            r.gaussian_fits_pdf = generate_gaussian_fits_pdf(
                r.root_file, pdf_path,
                root_executable=root_executable,
                n_events=n_fit_events,
            )

    # Step 2.5: Generate charge neighborhood PDFs
    if not skip_gaussian_fits:  # Use same flag as Gaussian fits
        print("\n" + "=" * 80)
        print("STEP 2.5: Generating charge neighborhood PDFs")
        print("=" * 80)

        neighborhood_dir = output_dir / "neighborhood_pdfs"
        ensure_dir(neighborhood_dir)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Generating neighborhood PDFs for {r.root_file.name}...")
            r.neighborhood_pdfs = generate_all_neighborhood_pdfs(
                r.root_file,
                neighborhood_dir,
                n_pages=min(n_fit_events, 100),
                root_executable=root_executable,
            )

    # Step 3: Extract sigma_f
    if not skip_sigma_f:
        print("\n" + "=" * 80)
        print("STEP 3: Extracting sigma_f from ReconTrueDeltaX distributions")
        print("=" * 80)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Extracting sigma_f from {r.root_file.name}...")
            r.sigma_f = extract_sigma_f(r.root_file, output_dir)

    # Step 4: Extract F_i and compute distance to reference pixel
    # The distance d = |x_particle_gun - x_pixel_center| is used for both F_i and sigma_f plots
    # (Tornago et al. Figure 5 style)
    reference_coord: Optional[Tuple[float, float]] = None

    if not skip_fi:
        print("\n" + "=" * 80)
        print("STEP 4: Extracting F_i values")
        print("=" * 80)

        # Infer pixel ID if not provided
        root_files = [r.root_file for r in results if r.root_file and r.root_file.exists()]
        if root_files:
            inferred_pixel_id = infer_pixel_of_interest(root_files, pixel_id)
            print(f"[INFO] Using pixel ID: {inferred_pixel_id}")

            for r in results:
                if r.root_file is None or not r.root_file.exists():
                    continue
                print(f"[INFO] Extracting F_i from {r.root_file.name}...")
                r.fi = extract_fi_from_file(r.root_file, inferred_pixel_id)

                # Set up reference pixel coordinate (for distance calculation)
                if r.fi.coord_mm is not None and reference_coord is None:
                    reference_coord = r.fi.coord_mm
                    print(f"[INFO] Reference pixel center: ({reference_coord[0]*1000:.1f}, {reference_coord[1]*1000:.1f}) µm")

    # Calculate signed distance from particle gun to reference pixel center for ALL results
    # This is used for both sigma_f and F_i plots (Tornago et al. Figure 5 convention)
    # Distance is signed: d = x_particle_gun - x_pixel_center
    if reference_coord is not None:
        pixel_x_mm, pixel_y_mm = reference_coord
        pixel_x_um = pixel_x_mm * 1000.0
        print(f"[INFO] Computing distances to reference pixel at x={pixel_x_um:.1f} µm")
        for r in results:
            # Signed distance: x_gun - x_pixel_center
            r.distance_to_reference_pixel_um = r.x_um - pixel_x_um
            # Also update r.fi.distance_um for backward compatibility
            if r.fi is not None:
                r.fi.distance_um = r.distance_to_reference_pixel_um
    else:
        # No reference coordinate available - fall back to x_um (assuming pixel at x=0)
        print("[WARN] No reference pixel coordinate available, using x_um as distance")
        for r in results:
            r.distance_to_reference_pixel_um = r.x_um

    # Step 5: Generate summary plots and Excel
    print("\n" + "=" * 80)
    print("STEP 5: Generating summary plots and Excel")
    print("=" * 80)

    plots_dir = output_dir / "summary_plots"
    ensure_dir(plots_dir)

    # Generate pixel grid visualization with measurement positions
    measurement_positions = [r.x_um for r in results]
    grid_viz_path = save_pixel_grid_visualization(measurement_positions, plots_dir)
    print(f"[INFO] Saved pixel grid visualization: {grid_viz_path}")

    # FNAL paper-style plots: resolution and F_i vs track position with gray pixel bands
    for branch in SIGMA_F_BRANCHES:
        has_data = any(
            r.sigma_f and branch in r.sigma_f.branch_results and np.isfinite(r.sigma_f.branch_results[branch].sigma)
            for r in results
        )
        if has_data:
            # FNAL paper-style: resolution vs track position with gray bands
            plot_path = save_sigma_f_vs_x_plot(results, plots_dir, branch)
            print(f"[INFO] Saved sigma_f vs x plot (FNAL style): {plot_path}")
            # Also save traditional distance plot
            plot_path = save_sigma_f_vs_d_plot(results, plots_dir, branch)
            print(f"[INFO] Saved sigma_f vs d plot: {plot_path}")

    # FNAL paper-style: F_i vs track position with gray bands
    fi_x_plot_path = save_fi_vs_x_plot(results, plots_dir)
    print(f"[INFO] Saved F_i vs x plot (FNAL style): {fi_x_plot_path}")

    # Traditional distance plot
    fi_plot_path = save_fi_vs_d_plot(results, plots_dir)
    print(f"[INFO] Saved F_i vs d plot: {fi_plot_path}")

    # Combined summary plot (FNAL style with gray bands)
    combined_plot_path = save_combined_summary_plot(results, plots_dir)
    print(f"[INFO] Saved combined summary plot: {combined_plot_path}")

    # Excel summary
    excel_path = write_summary_excel(results, output_dir)
    print(f"[INFO] Saved Excel summary: {excel_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    return results


def analyze_existing_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    root_executable: str = "root",
    n_fit_events: int = 200,
    pixel_id: Optional[int] = None,
    skip_gaussian_fits: bool = False,
    skip_sigma_f: bool = False,
    skip_fi: bool = False,
) -> List[PositionResult]:
    """Analyze an existing directory of ROOT files without running simulations."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = input_dir / f"analysis_{timestamp}"

    ensure_dir(output_dir)

    # Find ROOT files
    root_files = sorted(input_dir.glob("*.root"))
    if not root_files:
        raise RuntimeError(f"No ROOT files found in {input_dir}")

    print(f"Found {len(root_files)} ROOT files in {input_dir}")

    results: List[PositionResult] = []
    for rf in root_files:
        x_um = parse_x_um_from_filename(rf)
        results.append(PositionResult(x_um=x_um, root_file=rf))

    # Generate Gaussian fits PDFs
    if not skip_gaussian_fits:
        print("\n" + "=" * 80)
        print("Generating Gaussian fits PDFs")
        print("=" * 80)

        gaussian_fits_dir = output_dir / "gaussian_fits"
        ensure_dir(gaussian_fits_dir)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Generating Gaussian fits PDF for {r.root_file.name}...")
            pdf_path = gaussian_fits_dir / f"{r.root_file.stem}.pdf"
            r.gaussian_fits_pdf = generate_gaussian_fits_pdf(
                r.root_file, pdf_path,
                root_executable=root_executable,
                n_events=n_fit_events,
            )

    # Generate charge neighborhood PDFs
    if not skip_gaussian_fits:  # Use same flag as Gaussian fits
        print("\n" + "=" * 80)
        print("Generating charge neighborhood PDFs")
        print("=" * 80)

        neighborhood_dir = output_dir / "neighborhood_pdfs"
        ensure_dir(neighborhood_dir)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Generating neighborhood PDFs for {r.root_file.name}...")
            r.neighborhood_pdfs = generate_all_neighborhood_pdfs(
                r.root_file,
                neighborhood_dir,
                n_pages=min(n_fit_events, 100),
                root_executable=root_executable,
            )

    # Extract sigma_f
    if not skip_sigma_f:
        print("\n" + "=" * 80)
        print("Extracting sigma_f from ReconTrueDeltaX distributions")
        print("=" * 80)

        for r in results:
            if r.root_file is None or not r.root_file.exists():
                continue
            print(f"[INFO] Extracting sigma_f from {r.root_file.name}...")
            r.sigma_f = extract_sigma_f(r.root_file, output_dir)

    # Extract F_i and compute distance to reference pixel
    # The distance d = |x_particle_gun - x_pixel_center| is used for both F_i and sigma_f plots
    # (Tornago et al. Figure 5 style)
    reference_coord: Optional[Tuple[float, float]] = None

    if not skip_fi:
        print("\n" + "=" * 80)
        print("Extracting F_i values")
        print("=" * 80)

        root_file_list = [r.root_file for r in results if r.root_file and r.root_file.exists()]
        if root_file_list:
            inferred_pixel_id = infer_pixel_of_interest(root_file_list, pixel_id)
            print(f"[INFO] Using pixel ID: {inferred_pixel_id}")

            for r in results:
                if r.root_file is None or not r.root_file.exists():
                    continue
                print(f"[INFO] Extracting F_i from {r.root_file.name}...")
                r.fi = extract_fi_from_file(r.root_file, inferred_pixel_id)

                # Set up reference pixel coordinate (for distance calculation)
                if r.fi.coord_mm is not None and reference_coord is None:
                    reference_coord = r.fi.coord_mm
                    print(f"[INFO] Reference pixel center: ({reference_coord[0]*1000:.1f}, {reference_coord[1]*1000:.1f}) µm")

    # Calculate signed distance from particle gun to reference pixel center for ALL results
    # This is used for both sigma_f and F_i plots (Tornago et al. Figure 5 convention)
    # Distance is signed: d = x_particle_gun - x_pixel_center
    if reference_coord is not None:
        pixel_x_mm, pixel_y_mm = reference_coord
        pixel_x_um = pixel_x_mm * 1000.0
        print(f"[INFO] Computing distances to reference pixel at x={pixel_x_um:.1f} µm")
        for r in results:
            # Signed distance: x_gun - x_pixel_center
            r.distance_to_reference_pixel_um = r.x_um - pixel_x_um
            # Also update r.fi.distance_um for backward compatibility
            if r.fi is not None:
                r.fi.distance_um = r.distance_to_reference_pixel_um
    else:
        # No reference coordinate available - fall back to x_um (assuming pixel at x=0)
        print("[WARN] No reference pixel coordinate available, using x_um as distance")
        for r in results:
            r.distance_to_reference_pixel_um = r.x_um

    # Generate summary plots and Excel
    print("\n" + "=" * 80)
    print("Generating summary plots and Excel")
    print("=" * 80)

    plots_dir = output_dir / "summary_plots"
    ensure_dir(plots_dir)

    # Generate pixel grid visualization with measurement positions
    measurement_positions = [r.x_um for r in results]
    grid_viz_path = save_pixel_grid_visualization(measurement_positions, plots_dir)
    print(f"[INFO] Saved pixel grid visualization: {grid_viz_path}")

    # FNAL paper-style plots: resolution and F_i vs track position with gray pixel bands
    for branch in SIGMA_F_BRANCHES:
        has_data = any(
            r.sigma_f and branch in r.sigma_f.branch_results and np.isfinite(r.sigma_f.branch_results[branch].sigma)
            for r in results
        )
        if has_data:
            # FNAL paper-style: resolution vs track position with gray bands
            plot_path = save_sigma_f_vs_x_plot(results, plots_dir, branch)
            print(f"[INFO] Saved sigma_f vs x plot (FNAL style): {plot_path}")
            # Also save traditional distance plot
            plot_path = save_sigma_f_vs_d_plot(results, plots_dir, branch)
            print(f"[INFO] Saved sigma_f vs d plot: {plot_path}")

    # FNAL paper-style: F_i vs track position with gray bands
    fi_x_plot_path = save_fi_vs_x_plot(results, plots_dir)
    print(f"[INFO] Saved F_i vs x plot (FNAL style): {fi_x_plot_path}")

    # Traditional distance plot
    fi_plot_path = save_fi_vs_d_plot(results, plots_dir)
    print(f"[INFO] Saved F_i vs d plot: {fi_plot_path}")

    # Combined summary plot (FNAL style with gray bands)
    combined_plot_path = save_combined_summary_plot(results, plots_dir)
    print(f"[INFO] Saved combined summary plot: {combined_plot_path}")

    excel_path = write_summary_excel(results, output_dir)
    print(f"[INFO] Saved Excel summary: {excel_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified sweep analysis pipeline for charge sharing studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global config argument
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_FILE})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full pipeline command
    full_parser = subparsers.add_parser("run", help="Run full pipeline (simulate + analyze)")
    full_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for all results",
    )
    full_parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated list of x positions in um (e.g., '0,25,-25,50,-50')",
    )
    full_parser.add_argument(
        "--root-bin",
        type=str,
        default="root",
        help="ROOT executable path",
    )
    full_parser.add_argument(
        "--n-fit-events",
        type=int,
        default=200,
        help="Number of events to plot in Gaussian fits PDF",
    )
    full_parser.add_argument(
        "--pixel-id",
        type=int,
        default=None,
        help="Specific pixel ID to track for F_i (auto-detected if not specified)",
    )
    full_parser.add_argument(
        "--skip-gaussian-fits",
        action="store_true",
        help="Skip generating Gaussian fits PDFs",
    )
    full_parser.add_argument(
        "--skip-sigma-f",
        action="store_true",
        help="Skip sigma_f extraction",
    )
    full_parser.add_argument(
        "--skip-fi",
        action="store_true",
        help="Skip F_i extraction",
    )

    # Simulation configuration arguments
    full_parser.add_argument(
        "--charge-model",
        type=str,
        choices=CHARGE_MODELS,
        default=None,
        help="Charge sharing model: LogA (logarithmic attenuation, Tornago Eq.4) "
             "or LinA (linear attenuation, Tornago Eq.6). "
             "If not specified, uses current Config.hh setting.",
    )
    full_parser.add_argument(
        "--active-pixel-mode",
        type=str,
        choices=ACTIVE_PIXEL_MODES,
        default=None,
        help="Active pixel mode for charge fraction calculation: "
             "Neighborhood (all pixels), ChargeBlock2x2/3x3 (4/9 highest F_i), or RowCol/RowCol3x3 (cross pattern). "
             "If not specified, uses current Config.hh setting.",
    )
    full_parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta attenuation coefficient for linear charge model. "
             "Only used when --charge-model=LinA. "
             "If not specified, uses current Config.hh setting.",
    )

    # Analyze existing command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing ROOT files")
    analyze_parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing ROOT files to analyze",
    )
    analyze_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )
    analyze_parser.add_argument(
        "--root-bin",
        type=str,
        default="root",
        help="ROOT executable path",
    )
    analyze_parser.add_argument(
        "--n-fit-events",
        type=int,
        default=200,
        help="Number of events to plot in Gaussian fits PDF",
    )
    analyze_parser.add_argument(
        "--pixel-id",
        type=int,
        default=None,
        help="Specific pixel ID to track for F_i",
    )
    analyze_parser.add_argument(
        "--skip-gaussian-fits",
        action="store_true",
        help="Skip generating Gaussian fits PDFs",
    )
    analyze_parser.add_argument(
        "--skip-sigma-f",
        action="store_true",
        help="Skip sigma_f extraction",
    )
    analyze_parser.add_argument(
        "--skip-fi",
        action="store_true",
        help="Skip F_i extraction",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        # Load config from YAML file
        config_path = None
        if args.config:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = (REPO_ROOT / config_path).resolve()

        config = load_config(config_path)

        if args.command == "run":
            # CLI args override config values
            if args.positions:
                positions = [float(x.strip()) for x in args.positions.split(",")]
            else:
                positions = config.positions

            # Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
                if not output_dir.is_absolute():
                    output_dir = (REPO_ROOT / output_dir).resolve()
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_dir = DEFAULT_OUTPUT_BASE / timestamp

            # Merge CLI and config
            n_fit_events = args.n_fit_events if args.n_fit_events != 200 else config.n_events
            pixel_id = args.pixel_id if args.pixel_id is not None else config.pixel_id
            charge_model = args.charge_model if args.charge_model else config.charge_model
            active_pixel_mode = args.active_pixel_mode if args.active_pixel_mode else config.active_pixel_mode
            beta = args.beta if args.beta is not None else config.beta

            run_full_pipeline(
                positions=positions,
                output_dir=output_dir,
                run_simulations=True,
                root_executable=args.root_bin,
                n_fit_events=n_fit_events,
                pixel_id=pixel_id,
                skip_gaussian_fits=args.skip_gaussian_fits,
                skip_sigma_f=args.skip_sigma_f,
                skip_fi=args.skip_fi,
                charge_model=charge_model,
                active_pixel_mode=active_pixel_mode,
                beta=beta,
            )

        elif args.command == "analyze":
            input_dir = Path(args.input_dir)
            if not input_dir.is_absolute():
                input_dir = (REPO_ROOT / input_dir).resolve()

            output_dir = None
            if args.output_dir:
                output_dir = Path(args.output_dir)
                if not output_dir.is_absolute():
                    output_dir = (REPO_ROOT / output_dir).resolve()

            # Merge CLI and config
            n_fit_events = args.n_fit_events if args.n_fit_events != 200 else config.n_events
            pixel_id = args.pixel_id if args.pixel_id is not None else config.pixel_id

            analyze_existing_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                root_executable=args.root_bin,
                n_fit_events=n_fit_events,
                pixel_id=pixel_id,
                skip_gaussian_fits=args.skip_gaussian_fits,
                skip_sigma_f=args.skip_sigma_f,
                skip_fi=args.skip_fi,
            )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
