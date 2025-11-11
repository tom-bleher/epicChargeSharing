#!/usr/bin/env python3
"""Interactive demo for distance-based vertical error parameters.

Loads Q_f neighborhoods from the EPIC charge sharing ROOT output and allows
interactive tweaking of the distance-based error model parameters defined in
`ChargeUtils.h::DistancePowerSigma`. The demo displays the center row (x) and
center column (y) charges for a random sample of events together with the
computed per-point σ values.

Usage:
    python proc/fit/distance_error_demo.py [--mode linear|log]
                                          [--file /path/to/file.root]
                                          [--sample-size N]
                                          [--seed SEED]
                                          [--prefer-truth-center/--no-prefer-truth-center]

Dependencies: uproot, awkward, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.widgets import Slider


LINEAR_DIR_DEFAULT = "/home/tomble/epicChargeSharing/sweep_x_runs/LinearB0.001"
LOG_DIR_DEFAULT = "/home/tomble/epicChargeSharing/sweep_x_runs/Log"


@dataclass
class FileMetadata:
    pixel_spacing: float
    pixel_size: float
    neighborhood_radius: int


@dataclass
class EventSample:
    file_path: str
    entry: int
    x_true: float
    y_true: float
    x_px: float
    y_px: float
    q_grid: np.ndarray  # Shape (N, N)
    metadata: FileMetadata

    @property
    def grid_dim(self) -> int:
        return self.q_grid.shape[0]


def _read_tnamed_title(obj: uproot.reading.ReadOnlyDirectory) -> Optional[float]:
    try:
        title = obj.member("fTitle")
    except Exception:
        return None
    if title is None:
        return None
    try:
        return float(title)
    except (TypeError, ValueError):
        try:
            return float(str(title))
        except Exception:
            return None


def _infer_pixel_spacing_from_tree(tree: uproot.behaviors.TTree.TTree, max_entries: int = 50_000) -> Optional[float]:
    """Infer pixel spacing from PixelX/Y coordinates similarly to FitGaus1D."""
    arrays = tree.arrays({"PixelX": "PixelX", "PixelY": "PixelY"}, entry_stop=max_entries, library="np")
    x_vals = arrays["PixelX"].astype(float)
    y_vals = arrays["PixelY"].astype(float)
    x_vals = x_vals[np.isfinite(x_vals)]
    y_vals = y_vals[np.isfinite(y_vals)]

    def compute_gap(values: np.ndarray) -> Optional[float]:
        if values.size < 2:
            return None
        uniq = np.unique(values)
        if uniq.size < 2:
            return None
        gaps = np.diff(uniq)
        gaps = gaps[(gaps > 1e-9) & np.isfinite(gaps)]
        if gaps.size == 0:
            return None
        median_idx = gaps.size // 2
        return float(np.partition(gaps, median_idx)[median_idx])

    gx = compute_gap(x_vals)
    gy = compute_gap(y_vals)
    if gx and gy:
        return 0.5 * (gx + gy)
    if gx:
        return gx
    if gy:
        return gy
    return None


def _load_metadata(file: uproot.reading.ReadOnlyDirectory) -> FileMetadata:
    spacing = None
    size = None
    radius = None

    if "GridPixelSpacing_mm" in file:
        spacing = _read_tnamed_title(file["GridPixelSpacing_mm"])
    if "GridPixelSize_mm" in file:
        size = _read_tnamed_title(file["GridPixelSize_mm"])
    if "NeighborhoodRadius" in file:
        radius_val = _read_tnamed_title(file["NeighborhoodRadius"])
        if radius_val is not None:
            radius = int(round(radius_val))

    tree = file["Hits"]

    if not spacing or spacing <= 0.0:
        spacing = _infer_pixel_spacing_from_tree(tree)
    if not spacing or spacing <= 0.0:
        raise RuntimeError("Unable to determine pixel spacing from metadata or tree contents.")

    if not size or size <= 0.0:
        size = 0.5 * spacing

    if not radius or radius <= 0:
        radius = _infer_radius_from_tree(tree)
        if not radius or radius <= 0:
            radius = 2  # default to 5x5 neighborhood if all else fails

    return FileMetadata(pixel_spacing=float(spacing), pixel_size=float(size), neighborhood_radius=int(radius))


def _infer_radius_from_tree(tree: uproot.behaviors.TTree.TTree, max_entries: int = 50_000) -> Optional[int]:
    """Infer neighborhood radius by inspecting Q_f jagged array lengths."""
    for arrays in tree.iterate(filter_name="Q_f", entry_stop=max_entries, library="ak", step_size="1000"):
        q_vals = arrays["Q_f"]
        for arr in q_vals:
            length = len(arr)
            if length <= 0:
                continue
            n = int(round(math.sqrt(length)))
            if n * n == length and n >= 3:
                return (n - 1) // 2
    return None


def _iter_non_pixel_events(
    tree: uproot.behaviors.TTree.TTree,
    require_branches: Sequence[str],
    chunk: int = 2_000,
) -> Iterable[Tuple[int, dict]]:
    """Yield (entry_index, arrays) for non-pixel-hit entries."""
    all_branches = {"TrueX", "TrueY", "PixelX", "PixelY", "isPixelHit"}
    all_branches.update(require_branches)
    branch_list = list(all_branches)

    entry_start = 0
    total_entries = tree.num_entries
    while entry_start < total_entries:
        entry_stop = min(entry_start + chunk, total_entries)
        arrays = tree.arrays(branch_list, entry_start=entry_start, entry_stop=entry_stop, library="ak")
        count = len(arrays["TrueX"])
        for offset in range(count):
            entry_idx = entry_start + offset
            if bool(arrays["isPixelHit"][offset]):
                continue
            yield entry_idx, {name: arrays[name][offset] for name in require_branches}
        entry_start = entry_stop


def _collect_events_from_file(
    path: str,
    sample_size: int,
    rng: random.Random,
) -> List[EventSample]:
    events: List[EventSample] = []
    with uproot.open(path) as file:
        metadata = _load_metadata(file)
        tree = file["Hits"]

        entries = list(_iter_non_pixel_events(tree, require_branches=["TrueX", "TrueY", "PixelX", "PixelY", "Q_f"]))
        rng.shuffle(entries)
        for entry_idx, arrays in entries:
            q_vals = np.asarray(arrays["Q_f"], dtype=float)
            if q_vals.size == 0:
                continue
            n_side = int(round(math.sqrt(q_vals.size)))
            if n_side * n_side != q_vals.size or n_side < 3:
                continue
            q_grid = q_vals.reshape((n_side, n_side))
            events.append(
                EventSample(
                    file_path=path,
                    entry=entry_idx,
                    x_true=float(arrays["TrueX"]),
                    y_true=float(arrays["TrueY"]),
                    x_px=float(arrays["PixelX"]),
                    y_px=float(arrays["PixelY"]),
                    q_grid=q_grid,
                    metadata=metadata,
                )
            )
            if len(events) >= sample_size:
                break
    return events


def _apply_sigma_bounds(sigma: np.ndarray, max_charge: float, floor_pct: float, cap_pct: float) -> np.ndarray:
    result = np.array(sigma, copy=True, dtype=float)
    finite_mask = np.isfinite(result) & (result > 0.0)
    if not np.isfinite(max_charge) or max_charge <= 0.0:
        result[:] = np.nan
        return result

    floor_abs = 0.0
    if np.isfinite(floor_pct) and floor_pct > 0.0:
        floor_abs = floor_pct * 0.01 * max_charge
    cap_abs = np.inf
    if np.isfinite(cap_pct) and cap_pct > 0.0:
        cap_abs = cap_pct * 0.01 * max_charge

    result[finite_mask & (result < floor_abs)] = floor_abs
    result[finite_mask & (result > cap_abs)] = cap_abs

    result[~np.isfinite(result) | (result <= 0.0)] = np.nan
    return result


def distance_power_sigma(
    distances: np.ndarray,
    max_charge: float,
    pixel_spacing: float,
    distance_scale_pixels: float,
    exponent: float,
    floor_percent: float,
    cap_percent: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if not np.isfinite(max_charge) or max_charge <= 0.0:
        return np.full_like(distances, np.nan)
    if not np.isfinite(pixel_spacing) or pixel_spacing <= 0.0:
        return np.full_like(distances, np.nan)
    scale_pixels = distance_scale_pixels
    if not np.isfinite(scale_pixels) or scale_pixels <= 0.0:
        scale_pixels = 1.0
    exponent = exponent if np.isfinite(exponent) else 1.0
    exponent = max(0.0, exponent)
    floor_percent = floor_percent if np.isfinite(floor_percent) else 0.0
    floor_percent = max(0.0, floor_percent)

    distance_scale = max(scale_pixels * pixel_spacing, 1e-12)
    ratio = np.abs(distances) / distance_scale
    base_sigma = floor_percent * 0.01 * max_charge * np.power(1.0 + ratio, exponent)
    return _apply_sigma_bounds(base_sigma, max_charge, floor_percent, cap_percent)


def _compute_line_data(
    event: EventSample,
    axis: str,
    center_prefers_truth: bool,
    scale_px: float,
    exponent: float,
    floor_pct: float,
    cap_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    grid = event.q_grid
    n = event.grid_dim
    offsets = np.arange(n) - n // 2
    pixel_spacing = event.metadata.pixel_spacing
    max_charge = np.nanmax(grid[np.isfinite(grid)]) if np.any(np.isfinite(grid)) else np.nan

    if axis == "row":
        charges = grid[:, n // 2]
        positions = event.x_px + offsets * pixel_spacing
        truth_center = event.x_true
        fallback_peak_idx = int(np.nanargmax(charges)) if np.any(np.isfinite(charges)) else n // 2
        fallback_peak = positions[fallback_peak_idx]
    elif axis == "col":
        charges = grid[n // 2, :]
        positions = event.y_px + offsets * pixel_spacing
        truth_center = event.y_true
        fallback_peak_idx = int(np.nanargmax(charges)) if np.any(np.isfinite(charges)) else n // 2
        fallback_peak = positions[fallback_peak_idx]
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    if center_prefers_truth and np.isfinite(truth_center):
        center = truth_center
    elif np.isfinite(fallback_peak):
        center = fallback_peak
    else:
        center = positions[n // 2]

    sigmas = distance_power_sigma(
        positions - center,
        max_charge=max_charge,
        pixel_spacing=pixel_spacing,
        distance_scale_pixels=scale_px,
        exponent=exponent,
        floor_percent=floor_pct,
        cap_percent=cap_pct,
    )
    return positions, charges, sigmas, center


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize distance-based vertical uncertainties for Q_f data.")
    parser.add_argument("--mode", choices=("linear", "log"), default="linear", help="Dataset family to sample from.")
    parser.add_argument("--linear-dir", default=LINEAR_DIR_DEFAULT, help="Path to LinearB0.001 directory.")
    parser.add_argument("--log-dir", default=LOG_DIR_DEFAULT, help="Path to Log directory.")
    parser.add_argument("--file", help="Explicit ROOT file to load (overrides --mode).")
    parser.add_argument("--sample-size", type=int, default=40, help="Number of events to preload for interactive browsing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for event sampling.")
    parser.add_argument(
        "--prefer-truth-center",
        dest="prefer_truth_center",
        action="store_true",
        default=True,
        help="Center distance weighting on (TrueX, TrueY) when available (default).",
    )
    parser.add_argument(
        "--no-prefer-truth-center",
        dest="prefer_truth_center",
        action="store_false",
        help="Use peak/pixel center for distance origin instead of truth coordinates.",
    )
    return parser


def _gather_events(args: argparse.Namespace) -> List[EventSample]:
    rng = random.Random(args.seed)

    if args.file:
        root_files = [args.file]
    else:
        base_dir = Path(args.linear_dir if args.mode == "linear" else args.log_dir)
        root_files = sorted(str(p) for p in base_dir.glob("*.root"))
        if not root_files:
            raise RuntimeError(f"No ROOT files found in {base_dir}")
    events: List[EventSample] = []
    needed = args.sample_size
    for root_path in root_files:
        remaining = needed - len(events)
        if remaining <= 0:
            break
        samples = _collect_events_from_file(root_path, remaining, rng)
        events.extend(samples)
    if not events:
        raise RuntimeError("Unable to collect any non-pixel events with Q_f data.")
    return events


def _format_event_label(event: EventSample) -> str:
    rel_path = os.path.relpath(event.file_path, start=os.getcwd())
    return f"{rel_path} [entry {event.entry}, N={event.grid_dim}]"


def _create_sliders(fig: plt.Figure, slider_specs: Sequence[Tuple[str, Tuple[float, float], float, float]]) -> List[Slider]:
    sliders: List[Slider] = []
    slider_height = 0.03
    bottom = 0.03
    left = 0.1
    width = 0.78
    for idx, (label, (vmin, vmax), valinit, valstep) in enumerate(slider_specs):
        ax = fig.add_axes([left, bottom + idx * (slider_height + 0.01), width, slider_height])
        sliders.append(Slider(ax=ax, label=label, valmin=vmin, valmax=vmax, valinit=valinit, valstep=valstep))
    return sliders


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    events = _gather_events(args)
    event_count = len(events)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.22, wspace=0.25)

    # Slider configuration: (label, (min,max), init, step)
    slider_specs = [
        ("event index", (0, event_count - 1), 0, 1),
        ("scale (pixels)", (0.1, 2.0), 0.5, 0.05),
        ("exponent", (0.0, 3.0), 1.5, 0.05),
        ("floor %", (0.0, 20.0), 2.0, 0.1),
        ("cap %", (0.0, 30.0), 10.0, 0.1),
    ]
    sliders = _create_sliders(fig, slider_specs)
    slider_event, slider_scale, slider_exp, slider_floor, slider_cap = sliders

    text_box = fig.text(0.02, 0.95, "", fontsize=10, verticalalignment="top")

    def update_plot(_: float) -> None:
        idx = int(round(slider_event.val))
        idx = max(0, min(event_count - 1, idx))
        event = events[idx]
        scale_px = float(slider_scale.val)
        exponent = float(slider_exp.val)
        floor_pct = float(slider_floor.val)
        cap_pct = float(slider_cap.val)

        axes_titles = ["Center row (x vs Q_f)", "Center column (y vs Q_f)"]
        for ax, axis_name, title in zip(axes, ("row", "col"), axes_titles):
            ax.clear()
            try:
                positions, charges, sigmas, center = _compute_line_data(
                    event,
                    axis=axis_name,
                    center_prefers_truth=args.prefer_truth_center,
                    scale_px=scale_px,
                    exponent=exponent,
                    floor_pct=floor_pct,
                    cap_pct=cap_pct,
                )
            except ValueError:
                ax.set_title(f"{title}\n(no data)")
                continue

            mask = np.isfinite(charges)
            if not np.any(mask):
                ax.set_title(f"{title}\n(no finite Q_f values)")
                continue

            pos = positions[mask]
            y = charges[mask]
            yerr = sigmas[mask]

            ax.errorbar(pos, y, yerr=yerr, fmt="o", color="#1f77b4", ecolor="#ff7f0e", capsize=4, label="Q_f with σ")
            ax.axvline(center, color="#d62728", linestyle="--", linewidth=1.5, label="distance origin")
            if args.prefer_truth_center and np.isfinite(event.x_true if axis_name == "row" else event.y_true):
                truth_val = event.x_true if axis_name == "row" else event.y_true
                ax.axvline(truth_val, color="#2ca02c", linestyle=":", linewidth=1.2, label="truth")

            ax.set_xlabel("position [mm]")
            ax.set_ylabel("charge [C]")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # Indicate sigma range.
            finite_sigmas = yerr[np.isfinite(yerr)]
            if finite_sigmas.size > 0:
                ax.text(
                    0.02,
                    0.95,
                    f"σ range: {finite_sigmas.min():.2e} – {finite_sigmas.max():.2e}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

        text_box.set_text(
            f"File: {_format_event_label(event)}\n"
            f"Pixel spacing: {event.metadata.pixel_spacing:.4f} mm | "
            f"Pixel size: {event.metadata.pixel_size:.4f} mm\n"
            f"Params → scale={scale_px:.2f}px, exponent={exponent:.2f}, floor={floor_pct:.1f}%, cap={cap_pct:.1f}%"
        )
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update_plot)

    update_plot(0.0)
    plt.show()


if __name__ == "__main__":
    main()


