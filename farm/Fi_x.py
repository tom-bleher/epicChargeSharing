#!/usr/bin/env python3
"""Compute `F_i` charge fractions versus particle-gun x position.

Tailored for the `sweep_x` production, this utility extracts the F_i fraction
for a *single, fixed pixel of interest* from the first event at each particle-
gun x position.  Since F_i is computed from a formula and is constant (no noise),
there is no need to aggregate statistics over multiple events.

Key features
------------
- Robust pixel selection: we track the fraction for a specific *global* pixel
  ID using the `NeighborhoodPixelID` branch, so the neighborhood can shift
  around without losing the intended pixel.
- Automatic pixel detection prioritising the x ≈ 0 file (where the beam starts
  midway between two pads); we pick the dominant pixel and verify that the same
  pad is present across the sweep.
- Outputs: an Excel workbook with F_i per x position together with pixel
  coordinate drifts, and a consolidated PNG plotting F_i vs. x.
- Comparison mode: optionally process two directories (--input-dir2) to generate
  individual plots for each plus an overlay comparison plot.

Requirements
------------
Python packages: `uproot`, `awkward`, `numpy`, `pandas`, `matplotlib`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import pathlib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SWEEP_RUNS_DIR = REPO_ROOT / "sweep_x_runs"
SENTINEL_INVALID_FRACTION = -999.0


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
    return input_dir / "fi_vs_x"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if needed."""

    path.mkdir(parents=True, exist_ok=True)


def read_metadata_from_root_file(root_path: Path) -> Dict[str, float]:
    """Read grid metadata (pixel size, spacing) from a ROOT file.
    
    Returns a dict with keys like 'GridPixelSize_mm', 'GridPixelSpacing_mm'.
    Uses TParameter<double> objects stored in the tree's UserInfo.
    """
    metadata = {}
    try:
        with uproot.open(root_path) as f:
            if "Hits" not in f:
                return metadata
            tree = f["Hits"]
            user_info = tree.get("fUserInfo", None)
            if user_info is None:
                # Try accessing via the tree's user_info list
                try:
                    # uproot stores UserInfo as objects in tree directory
                    for key in f.keys():
                        obj = f[key]
                        if hasattr(obj, 'member'):
                            try:
                                # Check if it's a TParameter<double>
                                name = key.split(';')[0]
                                if name.startswith('Grid') or name == 'Gain':
                                    val = obj.member('fVal')
                                    metadata[name] = float(val)
                            except Exception:
                                pass
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
    
    Assumes pixels are centered at 0, ±pixel_spacing_mm, ±2*pixel_spacing_mm, etc.
    Each pixel pad extends ±(pixel_size_mm/2) from its center.
    """
    half_size = pixel_size_mm / 2.0
    x_min, x_max = x_range_mm
    
    # Find all pixel centers in range
    # Pixels at 0, ±spacing, ±2*spacing, ...
    max_n = int(np.ceil(max(abs(x_min), abs(x_max)) / pixel_spacing_mm)) + 1
    for n in range(-max_n, max_n + 1):
        center = n * pixel_spacing_mm
        left = center - half_size
        right = center + half_size
        # Only draw if it overlaps with the visible range
        if right >= x_min and left <= x_max:
            ax.axvspan(left, right, color=color, alpha=alpha, zorder=0)


def resolve_path(path: Optional[str], *, default: Path | str) -> Path:
    base = Path(default)
    if path is None:
        return base.resolve()
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def parse_x_um_from_filename(path: Path) -> float:
    """Extract micrometer displacement from file name, fall back to NaN."""

    name = path.name
    match = None
    for token in name.replace("-", " -").split():
        token = token.strip()
        if token.lower().endswith("um.root"):
            match = token[:-len("um.root")]
        elif token.lower().endswith("um"):
            match = token[:-len("um")]
        if match:
            try:
                return float(match)
            except ValueError:
                match = None
    # Second pass using numeric prefixes (e.g. "25um" without separator)
    import re

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*um", name, flags=re.IGNORECASE)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PixelSelection:
    pixel_id: int
    coord_mm: Optional[Tuple[float, float]] = None


@dataclass
class FileStats:
    path: Path
    x_um: float
    fi_value: float
    total_events: int
    coord_mm: Optional[Tuple[float, float]]
    offset_um: Optional[Tuple[float, float]] = None
    distance_um: Optional[float] = None


@dataclass
class RunResults:
    """Results from processing a single run directory."""
    input_dir: Path
    output_dir: Path
    pixel: PixelSelection
    stats: List[FileStats]
    excel_path: Path
    plot_path: Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def detect_fraction_branch(root_file: Path) -> str:
    """Detect which fraction branch name is available in the ROOT file.

    Returns the branch name to use for charge fractions. Supports:
    - 'Fi' (standard Neighborhood mode)
    - 'FiRow' (RowCol mode, row denominator)
    - 'FiCol' (RowCol mode, column denominator)
    - 'FiBlock' (RowCol3x3/Block mode)

    Falls back to 'Fi' if detection fails.
    """
    try:
        with uproot.open(root_file) as f:
            if "Hits" not in f:
                return "Fi"
            tree = f["Hits"]
            keys = {key.split(";")[0] for key in tree.keys()}

            # Check for RowCol mode branches first (more specific)
            if "FiRow" in keys:
                return "FiRow"
            if "FiCol" in keys:
                return "FiCol"
            # Block mode (RowCol3x3)
            if "FiBlock" in keys:
                return "FiBlock"
            # Standard neighborhood mode
            if "Fi" in keys:
                return "Fi"
    except Exception:
        pass
    return "Fi"


def iterate_tree_arrays(
    tree: uproot.behaviors.TBranch.HasBranches,
    branches: Sequence[str],
    *,
    step_size: str,
):
    """Yield awkward arrays for *branches* in *tree* with the given step size."""

    for arrays in tree.iterate(branches, step_size=step_size, library="ak"):
        yield arrays


def infer_pixel_of_interest(
    root_files: Sequence[Path],
    *,
    requested_pixel_id: Optional[int],
    sample_events: int,
    step_size: str,
) -> PixelSelection:
    """Determine which global pixel ID to analyse.

    If *requested_pixel_id* is provided it is returned directly (after basic
    validation).  Otherwise, inspect up to *sample_events* events across the
    files and choose the pixel with the highest mean fraction.
    """

    if requested_pixel_id is not None:
        return PixelSelection(pixel_id=int(requested_pixel_id))

    use_full_grid = False
    fraction_branch = "Fi"  # Default
    if root_files:
        try:
            with uproot.open(root_files[0]) as first_file:
                if "Hits" in first_file:
                    key_set = {key.split(";")[0] for key in first_file["Hits"].keys()}
                    use_full_grid = {"F_all", "F_all_pixel_id"}.issubset(key_set)
            # Detect which fraction branch to use
            fraction_branch = detect_fraction_branch(root_files[0])
        except Exception:
            use_full_grid = False

    accum_sum: Dict[int, float] = {}
    accum_count: Dict[int, int] = {}
    full_sum: Optional[np.ndarray] = None
    full_count: Optional[np.ndarray] = None
    full_pixel_ids: Optional[np.ndarray] = None

    prioritized = sorted(
        root_files,
        key=lambda p: (
            np.isnan(parse_x_um_from_filename(p)),
            abs(parse_x_um_from_filename(p)) if not np.isnan(parse_x_um_from_filename(p)) else float("inf"),
        ),
    )

    remaining = sample_events
    for path in prioritized:
        if remaining <= 0:
            break
        try:
            with uproot.open(path) as f:
                if "Hits" not in f:
                    continue
                tree = f["Hits"]
                if use_full_grid:
                    branches = ["F_all", "F_all_pixel_id"]
                    for arrays in iterate_tree_arrays(tree, branches, step_size=step_size):
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
                        assert full_sum is not None and full_count is not None and full_pixel_ids is not None
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
                    for arrays in iterate_tree_arrays(
                        tree,
                        [fraction_branch, "NeighborhoodPixelID"],
                        step_size=step_size,
                    ):
                        fi = ak.Array(arrays[fraction_branch])
                        ids = ak.Array(arrays["NeighborhoodPixelID"])

                        # Truncate chunk if we exceed desired sample size
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

                        valid_mask = (fi_flat != SENTINEL_INVALID_FRACTION) & (fi_flat >= 0.0)
                        if not valid_mask.any():
                            if remaining <= 0:
                                break
                            continue

                        fi_flat = fi_flat[valid_mask]
                        ids_flat = ids_flat[valid_mask]

                        for pid, val in zip(ids_flat, fi_flat):
                            accum_sum[pid] = accum_sum.get(pid, 0.0) + float(val)
                            accum_count[pid] = accum_count.get(pid, 0) + 1

                        if remaining <= 0:
                            break
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - diagnostic output only
            print(f"[WARN] Failed to inspect '{path}': {exc}")

    if use_full_grid:
        if full_pixel_ids is None or full_sum is None or full_count is None:
            raise RuntimeError("Unable to infer pixel of interest from full-grid fractions.")
        valid_mask = full_count > 0
        if not np.any(valid_mask):
            raise RuntimeError("No valid full-grid samples encountered while inferring pixel.")
        means = np.full_like(full_sum, fill_value=-np.inf, dtype=float)
        means[valid_mask] = full_sum[valid_mask] / full_count[valid_mask]
        best_index = int(np.argmax(means))
        return PixelSelection(pixel_id=int(full_pixel_ids[best_index]))

    if not accum_count:
        raise RuntimeError(
            "Unable to infer pixel of interest: no valid F_i entries were observed."
        )

    best_pid = max(accum_sum.keys(), key=lambda pid: accum_sum[pid] / accum_count[pid])
    return PixelSelection(pixel_id=int(best_pid))


def _first_pixel_coordinates(
    fi_chunk: ak.Array,
    ids_chunk: ak.Array,
    px_chunk: Optional[ak.Array],
    py_chunk: Optional[ak.Array],
    pixel_id: int,
) -> Optional[Tuple[float, float]]:
    """Extract the (x,y) coordinates for the first occurrence of *pixel_id*."""

    try:
        has_pixel = ak.to_numpy(ak.any(ids_chunk == pixel_id, axis=1))
    except Exception:
        return None

    idx_candidates = np.nonzero(has_pixel)[0]
    if idx_candidates.size == 0:
        return None

    evt_idx = int(idx_candidates[0])
    try:
        mask_evt = ak.to_list((ids_chunk == pixel_id)[evt_idx])
        fi_evt = ak.to_list(fi_chunk[evt_idx])
    except Exception:
        return None

    px_evt = ak.to_list(px_chunk[evt_idx]) if px_chunk is not None else None
    py_evt = ak.to_list(py_chunk[evt_idx]) if py_chunk is not None else None

    for k, flag in enumerate(mask_evt):
        if bool(flag):
            x = float(px_evt[k]) if px_evt is not None else float("nan")
            y = float(py_evt[k]) if py_evt is not None else float("nan")
            return (x, y)
    return None


def collect_file_stats(
    path: Path,
    pixel_id: int,
    *,
    step_size: str,
) -> FileStats:
    """Extract F_i for *pixel_id* from the first event in a ROOT file.

    Since F_i is formula-based and constant (no noise), we only need the first
    event's value rather than aggregating statistics.
    """

    # Detect which fraction branch to use before opening for iteration
    fraction_branch = detect_fraction_branch(path)

    with uproot.open(path) as f:
        if "Hits" not in f:
            raise RuntimeError(f"Tree 'Hits' not found in {path}")
        tree = f["Hits"]
        tree_keys = {key.split(";")[0] for key in tree.keys()}
        use_full_grid = {"F_all", "F_all_pixel_id"}.issubset(tree_keys)
        have_full_coords = {"F_all_pixel_x", "F_all_pixel_y"}.issubset(tree_keys)

        total_events = int(tree.num_entries)
        coord_mm: Optional[Tuple[float, float]] = None
        fi_value: Optional[float] = None

        if use_full_grid:
            branch_names = ["F_all", "F_all_pixel_id"]
            if have_full_coords:
                branch_names.extend(["F_all_pixel_x", "F_all_pixel_y"])

            # Read just the first event
            arrays = tree.arrays(branch_names, library="ak", entry_stop=1)
            if len(arrays) == 0:
                raise RuntimeError(f"No events in {path}")

            raw_ids = ak.to_numpy(arrays["F_all_pixel_id"][0])
            full_ids = np.asarray(raw_ids, dtype=int)
            matches = np.where(full_ids == pixel_id)[0]
            if matches.size == 0:
                raise RuntimeError(f"Pixel ID {pixel_id} not present in full-grid fractions for {path}")
            mapping_idx = int(matches[0])

            fractions = ak.to_numpy(arrays["F_all"])
            if fractions.ndim == 1:
                fractions = fractions[np.newaxis, :]
            fi_value = float(fractions[0, mapping_idx])

            if have_full_coords:
                px_vals = ak.to_numpy(arrays["F_all_pixel_x"][0])
                py_vals = ak.to_numpy(arrays["F_all_pixel_y"][0])
                if mapping_idx < len(px_vals) and mapping_idx < len(py_vals):
                    coord_mm = (float(px_vals[mapping_idx]), float(py_vals[mapping_idx]))

        else:
            # Read just the first event
            arrays = tree.arrays(
                [fraction_branch, "NeighborhoodPixelID", "NeighborhoodPixelX", "NeighborhoodPixelY"],
                library="ak",
                entry_stop=1,
            )
            if len(arrays) == 0:
                raise RuntimeError(f"No events in {path}")

            fi = ak.fill_none(ak.Array(arrays[fraction_branch]), np.nan)
            ids = ak.fill_none(ak.Array(arrays["NeighborhoodPixelID"]), -1)
            px = ak.Array(arrays["NeighborhoodPixelX"]) if "NeighborhoodPixelX" in arrays.fields else None
            py = ak.Array(arrays["NeighborhoodPixelY"]) if "NeighborhoodPixelY" in arrays.fields else None

            mask_pixel = ids == pixel_id
            if not ak.any(mask_pixel, axis=None):
                # Pixel not in neighborhood (e.g., scan position too far from target pixel)
                fi_value = float("nan")
            else:
                # Get the F_i value for this pixel from the first event
                fi_selected = fi[mask_pixel]
                fi_flat = ak.to_numpy(ak.flatten(fi_selected, axis=None)).astype(float)
                valid_mask = np.isfinite(fi_flat) & (fi_flat != SENTINEL_INVALID_FRACTION) & (fi_flat >= 0.0)
                if not valid_mask.any():
                    fi_value = float("nan")  # No valid fraction for this pixel
                else:
                    fi_value = float(fi_flat[valid_mask][0])

            # Get coordinates
            if px is not None and py is not None:
                coord_mm = _first_pixel_coordinates(fi, ids, px, py, pixel_id)

    return FileStats(
        path=path,
        x_um=parse_x_um_from_filename(path),
        fi_value=fi_value,
        total_events=total_events,
        coord_mm=coord_mm,
    )


def save_summary_plot(
    results: Sequence[FileStats],
    pixel: PixelSelection,
    out_dir: Path,
    *,
    label: Optional[str] = None,
    pixel_spacing_mm: float = 0.5,
    pixel_size_mm: float = 0.15,
) -> Path:
    """Save FNAL paper-style plot of Fi vs x with gray pixel regions.

    Style based on Tornago et al. / FNAL beam test publications:
    - Gray bands for pixel/metal pad regions
    - Step/histogram style lines
    - Clean axis labels: "Track x position [mm]"
    """
    ensure_dir(out_dir)

    # Use x_um (gun position) directly - convert to mm
    xs_um = np.array([r.x_um for r in results], dtype=float)
    xs_mm = xs_um / 1000.0  # Convert µm to mm
    ys = np.array([r.fi_value for r in results], dtype=float)

    # Sort by x position for proper step plot
    sort_idx = np.argsort(xs_mm)
    xs_mm = xs_mm[sort_idx]
    ys = ys[sort_idx]

    # FNAL paper-style figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Determine x range from data, extended slightly
    x_min = np.nanmin(xs_mm) - 0.05
    x_max = np.nanmax(xs_mm) + 0.05

    # Draw gray pixel regions first (behind the data)
    # Using lighter gray like the paper
    draw_pixel_regions(ax, pixel_spacing_mm, pixel_size_mm,
                       x_range_mm=(x_min, x_max), color="#d0d0d0", alpha=0.6)

    # Plot as histogram-style step (FNAL paper style)
    ax.step(xs_mm, ys, where="mid", color="#1f77b4", lw=2, zorder=2,
            label="Simulation")

    # FNAL paper-style axis labels
    ax.set_xlabel("Track x position [mm]", fontsize=12)
    ax.set_ylabel(r"Signal fraction $F_i$", fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Set y-axis range (F_i is between 0 and 1)
    ax.set_ylim(0, 1.05)

    # Minimal grid like paper
    ax.grid(True, linestyle="-", alpha=0.3, zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Title with pixel info
    coord_note = ""
    if pixel.coord_mm is not None:
        cx, cy = pixel.coord_mm
        coord_note = f" (pixel at x={cx:.3f} mm)"
    title = f"Signal fraction vs track position{coord_note}"
    if label:
        title = f"{title}\n{label}"
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    fig.tight_layout()

    suffix = f"_{label.replace(' ', '_')}" if label else ""
    out_path = out_dir / f"fi_vs_x_pixel{pixel.pixel_id}{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_overlay_plot(
    run1: RunResults,
    run2: RunResults,
    out_dir: Path,
) -> Path:
    """Create an overlay plot comparing two runs."""
    ensure_dir(out_dir)

    # Extract data for run 1
    xs1 = np.array([
        r.distance_um if (r.distance_um is not None and np.isfinite(r.distance_um)) else abs(r.x_um)
        for r in run1.stats
    ], dtype=float)
    ys1 = np.array([r.fi_value for r in run1.stats], dtype=float)

    # Extract data for run 2
    xs2 = np.array([
        r.distance_um if (r.distance_um is not None and np.isfinite(r.distance_um)) else abs(r.x_um)
        for r in run2.stats
    ], dtype=float)
    ys2 = np.array([r.fi_value for r in run2.stats], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Plot run 1
    label1 = run1.input_dir.name
    ax.plot(xs1, ys1, "o-", color="#d62728", lw=2, markersize=6, label=label1, alpha=0.8)

    # Plot run 2
    label2 = run2.input_dir.name
    ax.plot(xs2, ys2, "s-", color="#1f77b4", lw=2, markersize=6, label=label2, alpha=0.8)

    ax.set_xlabel("Distance from pixel center |Δx| [µm]")
    ax.set_ylabel("F_i (dimensionless)")

    coord_note = ""
    if run1.pixel.coord_mm is not None:
        cx, cy = run1.pixel.coord_mm
        coord_note = f" (pixel {run1.pixel.pixel_id}, x={cx:.3f} mm, y={cy:.3f} mm)"
    ax.set_title(f"F_i vs. distance comparison{coord_note}")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    # Create filename from both directory names
    name1 = run1.input_dir.name.replace("-", "").replace("_", "")[:12]
    name2 = run2.input_dir.name.replace("-", "").replace("_", "")[:12]
    out_path = out_dir / f"overlay_{name1}_vs_{name2}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_excel(results: Sequence[FileStats], pixel: PixelSelection, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    records = []
    for r in results:
        if r.offset_um is not None:
            dx_um, dy_um = r.offset_um
        else:
            dx_um = float("nan")
            dy_um = float("nan")
        records.append(
            {
                "gun x (um)": r.x_um,
                "distance |Δx| (um)": r.distance_um if r.distance_um is not None else float("nan"),
                "F_i": r.fi_value,
                "total events": r.total_events,
                "pixel Δx (um)": dx_um,
                "pixel Δy (um)": dy_um,
            }
        )

    df = pd.DataFrame(records).sort_values("gun x (um)")
    for col in ["distance |Δx| (um)", "F_i", "pixel Δx (um)", "pixel Δy (um)"]:
        df[col] = df[col].astype(float)

    out_path = out_dir / f"fi_vs_x_pixel{pixel.pixel_id}.xlsx"
    ensure_dir(out_path.parent)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Fi", index=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI handling
# ---------------------------------------------------------------------------


def list_root_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    files = sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".root"],
        key=lambda p: p.name,
    )
    if not files:
        raise RuntimeError(f"No ROOT files found in {directory}")
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate F_i fractions vs. particle-gun x position",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory containing ROOT files named by x position",
    )
    parser.add_argument(
        "--input-dir2",
        dest="input_dir2",
        type=str,
        default=None,
        help="Second directory for comparison (optional)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to store Excel and plots",
    )
    parser.add_argument(
        "--pixel-id",
        dest="pixel_id",
        type=int,
        default=None,
        help="Global pixel ID to track. If omitted, the dominant pixel is inferred",
    )
    parser.add_argument(
        "--sample-events",
        dest="sample_events",
        type=int,
        default=20000,
        help="Maximum number of events to inspect when inferring the pixel",
    )
    parser.add_argument(
        "--step-size",
        dest="step_size",
        type=str,
        default="200 MB",
        help="Iteration chunk size for uproot (advanced)",
    )
    return parser


def process_single_run(
    input_dir: Path,
    output_dir: Path,
    *,
    pixel_id: Optional[int],
    sample_events: int,
    step_size: str,
    label: Optional[str] = None,
) -> RunResults:
    """Process a single run directory and return results."""

    print(f"[INFO] Processing run: {input_dir}")
    ensure_dir(output_dir)

    root_files = list_root_files(input_dir)
    print(f"[INFO] Found {len(root_files)} ROOT files")

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

    pixel = infer_pixel_of_interest(
        root_files,
        requested_pixel_id=pixel_id,
        sample_events=max(1, int(sample_events)),
        step_size=step_size,
    )
    print(f"[INFO] Tracking pixel ID: {pixel.pixel_id}")

    stats: List[FileStats] = []
    reference_coord: Optional[Tuple[float, float]] = None

    for path in root_files:
        print(f"[INFO] Processing {path.name} …", end="")
        file_stats = collect_file_stats(
            path,
            pixel.pixel_id,
            step_size=step_size,
        )
        stats.append(file_stats)
        if file_stats.coord_mm is not None:
            if reference_coord is None:
                reference_coord = file_stats.coord_mm
                pixel.coord_mm = reference_coord
                file_stats.offset_um = (0.0, 0.0)
            else:
                dx = (file_stats.coord_mm[0] - reference_coord[0]) * 1000.0
                dy = (file_stats.coord_mm[1] - reference_coord[1]) * 1000.0
                file_stats.offset_um = (dx, dy)
                if max(abs(dx), abs(dy)) > 1e-3:
                    print(
                        f"\n[WARN] Pixel coordinate drift detected in {path.name}: Δx={dx:.3e} µm, Δy={dy:.3e} µm",
                        end="",
                    )
        if reference_coord is not None:
            ref_x_um = reference_coord[0] * 1000.0
            file_stats.distance_um = abs(file_stats.x_um - ref_x_um)
        dist_note = (
            f", |Δx|={file_stats.distance_um:.3f} µm"
            if file_stats.distance_um is not None and np.isfinite(file_stats.distance_um)
            else ""
        )
        print(f" done (F_i={file_stats.fi_value:.6f}{dist_note})")

    if not stats:
        raise RuntimeError("No statistics were collected; aborting.")

    if reference_coord is None and pixel.coord_mm is not None:
        reference_coord = pixel.coord_mm

    if reference_coord is not None:
        ref_x_um = reference_coord[0] * 1000.0
        for s in stats:
            if s.distance_um is None:
                s.distance_um = abs(s.x_um - ref_x_um)

    stats_sorted = sorted(stats, key=lambda s: s.x_um)

    excel_path = write_excel(stats_sorted, pixel, output_dir)
    plot_path = save_summary_plot(
        stats_sorted, pixel, output_dir, 
        label=label,
        pixel_spacing_mm=pixel_spacing_mm,
        pixel_size_mm=pixel_size_mm,
    )

    print(f"[INFO] Excel summary written to: {excel_path}")
    print(f"[INFO] Summary plot written to: {plot_path}")

    if pixel.coord_mm is not None:
        cx, cy = pixel.coord_mm
        print(f"[INFO] Pixel coordinates: x={cx:.6f} mm, y={cy:.6f} mm")

    return RunResults(
        input_dir=input_dir,
        output_dir=output_dir,
        pixel=pixel,
        stats=stats_sorted,
        excel_path=excel_path,
        plot_path=plot_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve input directory (use latest sweep run if not specified)
    if args.input_dir:
        p = Path(args.input_dir)
        if not p.is_absolute():
            input_dir = (REPO_ROOT / p).resolve()
        else:
            input_dir = p.resolve()
    else:
        input_dir = get_default_input_dir()

    input_dir2: Optional[Path] = None
    if args.input_dir2:
        # For second input dir, use relative resolution without default
        p = Path(args.input_dir2)
        if not p.is_absolute():
            input_dir2 = (REPO_ROOT / p).resolve()
        else:
            input_dir2 = p.resolve()

    # Resolve output directory (default to input_dir/fi_vs_x)
    if args.output_dir:
        p = Path(args.output_dir)
        if not p.is_absolute():
            base_output_dir = (REPO_ROOT / p).resolve()
        else:
            base_output_dir = p.resolve()
    else:
        base_output_dir = get_default_output_dir(input_dir)

    # Process first run
    print("=" * 80)
    print("PROCESSING RUN 1")
    print("=" * 80)
    output_dir1 = base_output_dir / "run1" if input_dir2 else base_output_dir
    run1 = process_single_run(
        input_dir,
        output_dir1,
        pixel_id=args.pixel_id,
        sample_events=args.sample_events,
        step_size=args.step_size,
        label=input_dir.name if input_dir2 else None,
    )

    # Process second run if provided
    if input_dir2:
        print()
        print("=" * 80)
        print("PROCESSING RUN 2")
        print("=" * 80)
        output_dir2 = base_output_dir / "run2"
        run2 = process_single_run(
            input_dir2,
            output_dir2,
            pixel_id=args.pixel_id,
            sample_events=args.sample_events,
            step_size=args.step_size,
            label=input_dir2.name,
        )

        # Create overlay plot
        print()
        print("=" * 80)
        print("CREATING OVERLAY PLOT")
        print("=" * 80)
        overlay_path = save_overlay_plot(run1, run2, base_output_dir)
        print(f"[INFO] Overlay plot written to: {overlay_path}")

    print()
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


