#!/usr/bin/env python3
"""Compute charge values (Q_i, Q_n, Q_f) versus particle-gun x position.

Tailored for the `sweep_x` production, this utility tracks raw charge values
for a *single, fixed pixel of interest* from every event, aggregates mean
charges at each particle-gun x position (50k shots per file), and exports
artefacts that are easy to plot or post-process.

Key features
------------
- Robust pixel selection: we track charges for a specific *global* pixel
  ID using the `NeighborhoodPixelID` branch, so the neighborhood can shift
  around without losing the intended pixel.
- Automatic pixel detection prioritising the x ≈ 0 file (where the beam starts
  midway between two pads); we pick the dominant pixel and verify that the same
  pad is present across the sweep.
- Outputs: an Excel workbook summarising mean/std per x position for Q_i, Q_n,
  and Q_f, together with pixel coordinate drifts and event coverage, a
  consolidated PNG plotting mean charges vs. x with error bars, and optional
  per-file histograms of the event-level distributions for each charge type.

Requirements
------------
Python packages: `uproot`, `awkward`, `numpy`, `pandas`, `matplotlib`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "sweep_x_runs" / "latest"  # Symlink or most recent run
DEFAULT_OUTPUT_BASE = REPO_ROOT / "proc" / "fit" / "q_vs_x"
SENTINEL_INVALID_CHARGE = -999.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if needed."""

    path.mkdir(parents=True, exist_ok=True)


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
class ChargeStats:
    mean: float
    std: float
    count: int
    values: Optional[np.ndarray] = None


@dataclass
class FileStats:
    path: Path
    x_um: float
    q_i: ChargeStats
    q_n: ChargeStats
    q_f: ChargeStats
    event_has_pixel: int
    total_events: int
    coord_mm: Optional[Tuple[float, float]]
    offset_um: Optional[Tuple[float, float]] = None
    distance_um: Optional[float] = None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


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
    files and choose the pixel with the highest mean Q_i (initial charge).
    """

    if requested_pixel_id is not None:
        return PixelSelection(pixel_id=int(requested_pixel_id))

    accum_sum: Dict[int, float] = {}
    accum_count: Dict[int, int] = {}

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
            with uproot.open(path) as f:
                if "Hits" not in f:
                    continue
                tree = f["Hits"]
                for arrays in iterate_tree_arrays(
                    tree,
                    ["Q_i", "NeighborhoodPixelID"],
                    step_size=step_size,
                ):
                    qi = ak.Array(arrays["Q_i"])
                    ids = ak.Array(arrays["NeighborhoodPixelID"])

                    # Truncate chunk if we exceed desired sample size
                    if remaining < len(qi):
                        qi = qi[:remaining]
                        ids = ids[:remaining]
                    remaining -= len(qi)

                    qi = ak.fill_none(qi, np.nan)
                    ids = ak.fill_none(ids, -1)

                    mask_pixel = ids >= 0
                    if not ak.any(mask_pixel, axis=None):
                        if remaining <= 0:
                            break
                        continue

                    qi_pixel = qi[mask_pixel]
                    ids_pixel = ids[mask_pixel]

                    qi_flat = ak.to_numpy(ak.flatten(qi_pixel, axis=None))
                    ids_flat = ak.to_numpy(ak.flatten(ids_pixel, axis=None)).astype(int)

                    if qi_flat.size == 0:
                        if remaining <= 0:
                            break
                        continue

                    finite_mask = np.isfinite(qi_flat)
                    if not finite_mask.any():
                        if remaining <= 0:
                            break
                        continue

                    qi_flat = qi_flat[finite_mask]
                    ids_flat = ids_flat[finite_mask]

                    valid_mask = (qi_flat != SENTINEL_INVALID_CHARGE) & (qi_flat >= 0.0)
                    if not valid_mask.any():
                        if remaining <= 0:
                            break
                        continue

                    qi_flat = qi_flat[valid_mask]
                    ids_flat = ids_flat[valid_mask]

                    for pid, val in zip(ids_flat, qi_flat):
                        accum_sum[pid] = accum_sum.get(pid, 0.0) + float(val)
                        accum_count[pid] = accum_count.get(pid, 0) + 1

                    if remaining <= 0:
                        break
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - diagnostic output only
            print(f"[WARN] Failed to inspect '{path}': {exc}")

    if not accum_count:
        raise RuntimeError(
            "Unable to infer pixel of interest: no valid Q_i entries were observed."
        )

    best_pid = max(accum_sum.keys(), key=lambda pid: accum_sum[pid] / accum_count[pid])
    return PixelSelection(pixel_id=int(best_pid))


def _first_pixel_coordinates(
    qi_chunk: ak.Array,
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
        qi_evt = ak.to_list(qi_chunk[evt_idx])
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


def _compute_stats(
    values_sum: float,
    values_sumsq: float,
    values_count: int,
    values_chunks: List[np.ndarray],
    capture_values: bool,
) -> ChargeStats:
    """Compute statistics from accumulated sums."""

    if capture_values and values_chunks:
        stored_values = np.concatenate(values_chunks)
    elif capture_values:
        stored_values = np.empty(0, dtype=float)
    else:
        stored_values = None

    count = values_count
    if count > 0:
        mean = values_sum / count
    else:
        mean = float("nan")

    if count > 1:
        variance = (values_sumsq - count * mean * mean) / (count - 1)
        variance = max(variance, 0.0)
        std = math.sqrt(variance)
    else:
        std = float("nan")

    return ChargeStats(
        mean=mean,
        std=std,
        count=count,
        values=stored_values,
    )


def collect_file_stats(
    path: Path,
    pixel_id: int,
    *,
    step_size: str,
    capture_values: bool,
) -> FileStats:
    """Gather aggregate statistics for *pixel_id* from a ROOT file."""

    with uproot.open(path) as f:
        if "Hits" not in f:
            raise RuntimeError(f"Tree 'Hits' not found in {path}")
        tree = f["Hits"]

        total_events = int(tree.num_entries)
        event_has_pixel = 0
        coord_mm: Optional[Tuple[float, float]] = None

        # Accumulators for Q_i, Q_n, Q_f
        qi_sum = 0.0
        qi_sumsq = 0.0
        qi_count = 0
        qi_chunks: List[np.ndarray] = [] if capture_values else []

        qn_sum = 0.0
        qn_sumsq = 0.0
        qn_count = 0
        qn_chunks: List[np.ndarray] = [] if capture_values else []

        qf_sum = 0.0
        qf_sumsq = 0.0
        qf_count = 0
        qf_chunks: List[np.ndarray] = [] if capture_values else []

        for arrays in iterate_tree_arrays(
            tree,
            [
                "Q_i",
                "Q_n",
                "Q_f",
                "NeighborhoodPixelID",
                "NeighborhoodPixelX",
                "NeighborhoodPixelY",
            ],
            step_size=step_size,
        ):
            qi = ak.fill_none(ak.Array(arrays["Q_i"]), np.nan)
            qn = ak.fill_none(ak.Array(arrays["Q_n"]), np.nan)
            qf = ak.fill_none(ak.Array(arrays["Q_f"]), np.nan)
            ids = ak.fill_none(ak.Array(arrays["NeighborhoodPixelID"]), -1)
            px = ak.Array(arrays["NeighborhoodPixelX"]) if "NeighborhoodPixelX" in arrays.fields else None
            py = ak.Array(arrays["NeighborhoodPixelY"]) if "NeighborhoodPixelY" in arrays.fields else None

            mask_pixel = ids == pixel_id

            try:
                per_event = ak.to_numpy(ak.any(mask_pixel, axis=1))
            except Exception:
                per_event = np.zeros(len(ids), dtype=bool)
            event_has_pixel += int(np.count_nonzero(per_event))

            if coord_mm is None and px is not None and py is not None:
                coord_mm = _first_pixel_coordinates(qi, ids, px, py, pixel_id)

            if not ak.any(mask_pixel, axis=None):
                continue

            # Process Q_i
            qi_selected = qi[mask_pixel]
            qi_flat = ak.to_numpy(ak.flatten(qi_selected, axis=None)).astype(float)
            if qi_flat.size > 0:
                finite_mask = np.isfinite(qi_flat)
                if finite_mask.any():
                    qi_flat = qi_flat[finite_mask]
                    valid_mask = (qi_flat != SENTINEL_INVALID_CHARGE) & (qi_flat >= 0.0)
                    if valid_mask.any():
                        qi_flat = qi_flat[valid_mask]
                        qi_sum += float(np.sum(qi_flat))
                        qi_sumsq += float(np.sum(qi_flat * qi_flat))
                        qi_count += int(qi_flat.size)
                        if capture_values:
                            qi_chunks.append(qi_flat)

            # Process Q_n
            qn_selected = qn[mask_pixel]
            qn_flat = ak.to_numpy(ak.flatten(qn_selected, axis=None)).astype(float)
            if qn_flat.size > 0:
                finite_mask = np.isfinite(qn_flat)
                if finite_mask.any():
                    qn_flat = qn_flat[finite_mask]
                    valid_mask = (qn_flat != SENTINEL_INVALID_CHARGE) & (qn_flat >= 0.0)
                    if valid_mask.any():
                        qn_flat = qn_flat[valid_mask]
                        qn_sum += float(np.sum(qn_flat))
                        qn_sumsq += float(np.sum(qn_flat * qn_flat))
                        qn_count += int(qn_flat.size)
                        if capture_values:
                            qn_chunks.append(qn_flat)

            # Process Q_f
            qf_selected = qf[mask_pixel]
            qf_flat = ak.to_numpy(ak.flatten(qf_selected, axis=None)).astype(float)
            if qf_flat.size > 0:
                finite_mask = np.isfinite(qf_flat)
                if finite_mask.any():
                    qf_flat = qf_flat[finite_mask]
                    valid_mask = (qf_flat != SENTINEL_INVALID_CHARGE) & (qf_flat >= 0.0)
                    if valid_mask.any():
                        qf_flat = qf_flat[valid_mask]
                        qf_sum += float(np.sum(qf_flat))
                        qf_sumsq += float(np.sum(qf_flat * qf_flat))
                        qf_count += int(qf_flat.size)
                        if capture_values:
                            qf_chunks.append(qf_flat)

    qi_stats = _compute_stats(qi_sum, qi_sumsq, qi_count, qi_chunks, capture_values)
    qn_stats = _compute_stats(qn_sum, qn_sumsq, qn_count, qn_chunks, capture_values)
    qf_stats = _compute_stats(qf_sum, qf_sumsq, qf_count, qf_chunks, capture_values)

    return FileStats(
        path=path,
        x_um=parse_x_um_from_filename(path),
        q_i=qi_stats,
        q_n=qn_stats,
        q_f=qf_stats,
        event_has_pixel=event_has_pixel,
        total_events=total_events,
        coord_mm=coord_mm,
    )


def save_histograms(stats: FileStats, pixel_id: int, out_dir: Path) -> None:
    """Save histograms for Q_i, Q_n, and Q_f."""

    ensure_dir(out_dir)
    
    charge_types = [
        ("Q_i", stats.q_i, "Initial charge"),
        ("Q_n", stats.q_n, "Neighborhood charge"),
        ("Q_f", stats.q_f, "Fitted charge"),
    ]

    for charge_name, charge_stats, charge_label in charge_types:
        if charge_stats.values is None or charge_stats.values.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        ax.hist(charge_stats.values, bins=50, color="#2b5fff", alpha=0.75, edgecolor="#1c3fa0")
        ax.set_xlabel(f"{charge_name} [electrons]")
        ax.set_ylabel("Entries")
        ax.set_title(
            f"{charge_label} distribution — x={stats.x_um:+.1f} µm, pixel {pixel_id}\n"
            f"N={charge_stats.count}, mean={charge_stats.mean:.1f} e⁻, σ={charge_stats.std:.1f} e⁻"
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()

        basename = f"hist_{charge_name}_x_{stats.x_um:+.1f}um_pixel{pixel_id}.png".replace("+", "plus").replace("-", "minus")
        fig.savefig(out_dir / basename)
        plt.close(fig)


def save_summary_plots(results: Sequence[FileStats], pixel: PixelSelection, out_dir: Path) -> List[Path]:
    """Generate summary plots for Q_i, Q_n, and Q_f versus x position."""

    ensure_dir(out_dir)
    out_paths = []

    xs = np.array([
        r.distance_um if (r.distance_um is not None and math.isfinite(r.distance_um)) else abs(r.x_um)
        for r in results
    ], dtype=float)

    charge_types = [
        ("Q_i", "Initial charge", "#d62728", "#ff9896"),
        ("Q_n", "Neighborhood charge", "#2ca02c", "#98df8a"),
        ("Q_f", "Fitted charge", "#1f77b4", "#aec7e8"),
    ]

    coord_note = ""
    if pixel.coord_mm is not None:
        cx, cy = pixel.coord_mm
        coord_note = f" (pixel {pixel.pixel_id}, x={cx:.3f} mm, y={cy:.3f} mm)"

    for charge_name, charge_label, color, ecolor in charge_types:
        ys = np.array([getattr(r, charge_name.lower()).mean for r in results], dtype=float)
        yerr = np.array([
            getattr(r, charge_name.lower()).std if math.isfinite(getattr(r, charge_name.lower()).std) else 0.0
            for r in results
        ], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=color, ecolor=ecolor, capsize=5, lw=2)
        ax.set_xlabel("Distance from pixel center |Δx| [µm]")
        ax.set_ylabel(f"Mean {charge_name} [electrons]")
        ax.set_title(f"{charge_label} vs. distance{coord_note}")
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()

        out_path = out_dir / f"{charge_name.lower()}_vs_x_pixel{pixel.pixel_id}.png"
        fig.savefig(out_path)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def write_excel(results: Sequence[FileStats], pixel: PixelSelection, out_dir: Path) -> Path:
    """Write Excel workbook with Q_i, Q_n, and Q_f statistics."""

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
                "mean Q_i": r.q_i.mean,
                "std Q_i": r.q_i.std,
                "samples Q_i": r.q_i.count,
                "mean Q_n": r.q_n.mean,
                "std Q_n": r.q_n.std,
                "samples Q_n": r.q_n.count,
                "mean Q_f": r.q_f.mean,
                "std Q_f": r.q_f.std,
                "samples Q_f": r.q_f.count,
                "events with pixel": r.event_has_pixel,
                "total events": r.total_events,
                "pixel Δx (um)": dx_um,
                "pixel Δy (um)": dy_um,
                "coverage": (r.event_has_pixel / r.total_events) if r.total_events else float("nan"),
            }
        )

    df = pd.DataFrame(records).sort_values("gun x (um)")
    for col in ["distance |Δx| (um)", "mean Q_i", "std Q_i", "mean Q_n", "std Q_n", 
                "mean Q_f", "std Q_f", "pixel Δx (um)", "pixel Δy (um)", "coverage"]:
        df[col] = df[col].astype(float)

    out_path = out_dir / f"q_vs_x_pixel{pixel.pixel_id}.xlsx"
    ensure_dir(out_path.parent)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Charges", index=False)
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
        description="Aggregate Q_i, Q_n, Q_f charges vs. particle-gun x position",
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
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to store Excel, plots, and histograms",
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
        "--skip-histograms",
        action="store_true",
        help="Do not save per-file charge histograms",
    )
    parser.add_argument(
        "--step-size",
        dest="step_size",
        type=str,
        default="200 MB",
        help="Iteration chunk size for uproot (advanced)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_path(args.input_dir, default=DEFAULT_INPUT_DIR)
    output_dir = resolve_path(args.output_dir, default=DEFAULT_OUTPUT_BASE / datetime.now().strftime("%Y%m%d-%H%M%S"))

    print(f"[INFO] Using input directory: {input_dir}")
    ensure_dir(output_dir)

    root_files = list_root_files(input_dir)
    print(f"[INFO] Found {len(root_files)} ROOT files")

    pixel = infer_pixel_of_interest(
        root_files,
        requested_pixel_id=args.pixel_id,
        sample_events=max(1, int(args.sample_events)),
        step_size=args.step_size,
    )
    print(f"[INFO] Tracking pixel ID: {pixel.pixel_id}")

    stats: List[FileStats] = []
    histogram_dir = output_dir / "histograms"
    reference_coord: Optional[Tuple[float, float]] = None

    for path in root_files:
        print(f"[INFO] Processing {path.name} …", end="")
        file_stats = collect_file_stats(
            path,
            pixel.pixel_id,
            step_size=args.step_size,
            capture_values=not args.skip_histograms,
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
        coverage = (
            file_stats.event_has_pixel / file_stats.total_events
            if file_stats.total_events > 0
            else float("nan")
        )
        if coverage < 0.999:
            print(
                f"\n[WARN] Pixel ID {pixel.pixel_id} present in only {coverage*100:.2f}% of events in {path.name}",
                end="",
            )
        if not args.skip_histograms:
            save_histograms(file_stats, pixel.pixel_id, histogram_dir)
        dist_note = (
            f", |Δx|={file_stats.distance_um:.3f} µm"
            if file_stats.distance_um is not None and math.isfinite(file_stats.distance_um)
            else ""
        )
        print(f" done (Q_i: N={file_stats.q_i.count}, mean={file_stats.q_i.mean:.1f}{dist_note})")

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
    plot_paths = save_summary_plots(stats_sorted, pixel, output_dir)

    print(f"[INFO] Excel summary written to: {excel_path}")
    for plot_path in plot_paths:
        print(f"[INFO] Summary plot written to: {plot_path}")

    if pixel.coord_mm is not None:
        cx, cy = pixel.coord_mm
        print(f"[INFO] Pixel coordinates: x={cx:.6f} mm, y={cy:.6f} mm")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

