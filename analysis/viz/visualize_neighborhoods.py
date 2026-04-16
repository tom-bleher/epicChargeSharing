#!/usr/bin/env python3
"""Visualize charge neighborhood grids for every event, highlighting threshold-activated pixels."""

import sys
import numpy as np
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

EVENTS_PER_ROW = 8
EVENTS_PER_COL = 6
EVENTS_PER_PAGE = EVENTS_PER_ROW * EVENTS_PER_COL
GRID_SIDE = 5  # 5x5 neighborhood
DETECTOR_HALF = 15.0  # detector extends from -15 to +15 mm

# Defaults — overridden by UserInfo metadata when available
DEFAULT_PITCH = 0.5  # mm
DEFAULT_PIXEL_SIZE = 0.10  # mm (metal pad width)


def read_metadata(root_file):
    """Read geometry metadata from UserInfo if available, else derive from data."""
    meta = {"pitch": DEFAULT_PITCH, "pixel_size": DEFAULT_PIXEL_SIZE}
    with uproot.open(root_file, minimal_ttree_metadata=False) as f:
        tree = f["Hits"]
        # Try UserInfo first
        try:
            user_info = tree.member("fUserInfo", none_if_missing=True)
            if user_info is not None:
                ui_map = {}
                for obj in user_info:
                    try:
                        ui_map[obj.member("fName")] = float(obj.member("fVal"))
                    except Exception:
                        pass
                if "GridPixelSpacing_mm" in ui_map:
                    meta["pitch"] = ui_map["GridPixelSpacing_mm"]
                if "GridPixelSize_mm" in ui_map:
                    meta["pixel_size"] = ui_map["GridPixelSize_mm"]
                if ui_map:
                    print(f"  Metadata from UserInfo: pitch={meta['pitch']}mm, pad={meta['pixel_size']}mm")
                    return meta
        except Exception:
            pass
        # Derive pitch from pixel coordinates
        try:
            px = tree["NeighborhoodPixelX"].array(entry_stop=50, library="ak")
            ns = tree["NeighborhoodSize"].array(entry_stop=50, library="ak")
            for i in range(min(50, len(ns))):
                n = int(ns[i])
                if n < 4:
                    continue
                side = int(np.sqrt(n))
                x = ak.to_numpy(px[i]).reshape(side, side)
                derived_pitch = abs(float(x[1, 0] - x[0, 0]))
                if derived_pitch > 0:
                    meta["pitch"] = derived_pitch
                    break
        except Exception:
            pass
    print(f"  Metadata (derived/default): pitch={meta['pitch']}mm, pad={meta['pixel_size']}mm")
    return meta


def load_data(root_file):
    with uproot.open(root_file) as f:
        tree = f["Hits"]
        branches = tree.arrays(
            ["Fi", "Q_ind", "Q_amp", "Q_meas", "NeighborhoodPixelX", "NeighborhoodPixelY",
             "TrueX", "TrueY", "PixelX", "PixelY", "EnergyDeposited", "NeighborhoodSize",
             "NearestPixelI", "NearestPixelJ", "isPixelHit", "hitWithinDetector"],
            library="ak",
        )
    return branches


def draw_event(ax, orig_idx, branches, meta):
    """Draw a single event's 5x5 neighborhood grid on the given axes."""
    nsize = int(branches["NeighborhoodSize"][orig_idx])
    pitch = meta["pitch"]
    pad_size = meta["pixel_size"]
    pad_frac = pad_size / pitch  # true physical fraction (e.g. 0.2)

    fi = ak.to_numpy(branches["Fi"][orig_idx])
    qf = ak.to_numpy(branches["Q_meas"][orig_idx])
    true_x = float(branches["TrueX"][orig_idx])
    true_y = float(branches["TrueY"][orig_idx])
    pix_x = ak.to_numpy(branches["NeighborhoodPixelX"][orig_idx])
    pix_y = ak.to_numpy(branches["NeighborhoodPixelY"][orig_idx])
    edep = float(branches["EnergyDeposited"][orig_idx])

    # The neighborhood may be incomplete (< GRID_SIDE^2 pixels) near detector edges.
    # Reshape into full grid, padding missing pixels with NaN.
    expected = GRID_SIDE * GRID_SIDE
    is_edge = nsize < expected

    if is_edge:
        # Pad arrays to full grid size with NaN/zero
        def pad(arr, fill=np.nan):
            out = np.full(expected, fill)
            out[:nsize] = arr[:nsize]
            return out
        fi = pad(fi)
        qf = pad(qf, 0.0)
        pix_x = pad(pix_x)
        pix_y = pad(pix_y)

    fi_grid = fi.reshape(GRID_SIDE, GRID_SIDE)
    qf_grid = qf.reshape(GRID_SIDE, GRID_SIDE)
    active_mask = qf_grid != 0.0
    n_active = int(np.count_nonzero(active_mask))

    # Pixel coordinates and hit position in grid units
    px_arr = pix_x.reshape(GRID_SIDE, GRID_SIDE)
    py_arr = pix_y.reshape(GRID_SIDE, GRID_SIDE)
    x_vals = px_arr[:, 0]
    y_vals = py_arr[0, :]
    pitch_x = x_vals[1] - x_vals[0] if len(x_vals) > 1 else pitch
    pitch_y = y_vals[1] - y_vals[0] if len(y_vals) > 1 else pitch

    # For edge events, detect which cells are missing (NaN pixel coordinates)
    missing_mask = np.isnan(px_arr) | np.isnan(py_arr)

    # --- Background: white for gap, cells drawn on top ---
    ax.set_facecolor("white")

    # --- Grid lines at pitch boundaries (thin, dotted) ---
    for i in range(GRID_SIDE + 1):
        ax.axvline(i - 0.5, color="#cccccc", lw=0.3, ls=":", zorder=0)
        ax.axhline(i - 0.5, color="#cccccc", lw=0.3, ls=":", zorder=0)

    # --- Draw pads ---
    vmax = fi.max() * 1.1 if fi.max() > 0 else 1
    cmap = plt.cm.YlOrRd
    gap_half = (1.0 - pad_frac) / 2.0

    for ix in range(GRID_SIDE):
        for iy in range(GRID_SIDE):
            pad_x = ix - 0.5 + gap_half
            pad_y = iy - 0.5 + gap_half
            fi_val = fi_grid[ix, iy]

            norm_val = fi_val / vmax if vmax > 0 else 0
            face_color = cmap(norm_val)

            if missing_mask[ix, iy]:
                # Missing pixel (beyond detector edge): hatched cell
                ax.add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1,
                             linewidth=0.4, edgecolor="#aaaaaa",
                             facecolor="none", hatch="//", zorder=1))
            elif active_mask[ix, iy]:
                # Active pixel: colored pad with blue border
                ax.add_patch(Rectangle((pad_x, pad_y), pad_frac, pad_frac,
                             linewidth=0.5, edgecolor="blue",
                             facecolor=face_color, zorder=3))
            else:
                # Inactive pixel (below threshold): light gray pad
                ax.add_patch(Rectangle((pad_x, pad_y), pad_frac, pad_frac,
                             linewidth=0.3, edgecolor="#bbbbbb",
                             facecolor="#f0f0f0", zorder=2))

    # Mark true hit position relative to grid
    hit_ix = (true_x - x_vals[0]) / pitch_x
    hit_iy = (true_y - y_vals[0]) / pitch_y
    ax.plot(hit_ix, hit_iy, marker="+", color="lime", markersize=6,
            markeredgewidth=1.2, zorder=5)

    # --- Red border on sides that have missing pixels ---
    if is_edge:
        lw = 1.5
        # Check which edges have missing pixels
        if np.any(missing_mask[0, :]):
            ax.axvline(-0.5, color="red", lw=lw, zorder=6)
        if np.any(missing_mask[-1, :]):
            ax.axvline(GRID_SIDE - 0.5, color="red", lw=lw, zorder=6)
        if np.any(missing_mask[:, 0]):
            ax.axhline(-0.5, color="red", lw=lw, zorder=6)
        if np.any(missing_mask[:, -1]):
            ax.axhline(GRID_SIDE - 0.5, color="red", lw=lw, zorder=6)

    ax.set_xlim(-0.5, GRID_SIDE - 0.5)
    ax.set_ylim(-0.5, GRID_SIDE - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    label = f"#{orig_idx}  {n_active}px  {edep:.3f}keV"
    if is_edge:
        label += "  EDGE"
    ax.set_title(label, fontsize=4.5, pad=1)


def main():
    root_file = sys.argv[1] if len(sys.argv) > 1 else "build/epicChargeSharing.root"
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else "neighborhood_grids.pdf"
    max_events = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"Loading {root_file}...")
    meta = read_metadata(root_file)
    branches = load_data(root_file)
    n_total = len(branches["EnergyDeposited"])

    # Build list of valid event indices (skip nsize==0 and NaN events)
    nsize = ak.to_numpy(branches["NeighborhoodSize"])
    valid_mask = nsize > 0
    # Also skip NaN events (nsize>0 but data is NaN — shouldn't happen, but guard)
    fi_first = ak.to_numpy(branches["Fi"][:, 0])
    valid_mask &= ~np.isnan(fi_first)

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    n_skipped = n_total - n_valid

    if max_events is not None:
        valid_indices = valid_indices[:max_events]
        n_valid = len(valid_indices)

    n_pages = (n_valid + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE

    # Separate normal vs edge events (incomplete neighborhoods)
    expected_size = GRID_SIDE * GRID_SIDE
    all_nsize = ak.to_numpy(branches["NeighborhoodSize"])
    normal_indices = valid_indices[all_nsize[valid_indices] == expected_size]
    edge_indices_all = np.where((valid_mask) & (all_nsize < expected_size))[0]

    if max_events is not None:
        normal_indices = normal_indices[:max_events]
    n_normal = len(normal_indices)
    n_edge = len(edge_indices_all)
    n_pages = (n_normal + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE

    print(f"  {n_total} total events, {n_skipped} skipped (no hit), {n_normal} normal + {n_edge} edge")
    print(f"  {n_pages} pages + 1 edge page")
    print(f"  Writing to {out_pdf}...", flush=True)

    page_title = (
        f"Charge Neighborhood Grids — "
        f"pitch={meta['pitch']}mm, pad={meta['pixel_size']}mm "
        f"({meta['pixel_size']/meta['pitch']*100:.0f}% fill)   "
        f"Blue = above threshold  |  + = true hit"
    )

    with PdfPages(out_pdf) as pdf:
        # --- Normal event pages ---
        for page in range(n_pages):
            start = page * EVENTS_PER_PAGE
            end = min(start + EVENTS_PER_PAGE, n_normal)
            page_indices = normal_indices[start:end]

            fig, axes = plt.subplots(EVENTS_PER_COL, EVENTS_PER_ROW,
                                     figsize=(11, 8.5),
                                     gridspec_kw={"hspace": 0.4, "wspace": 0.15})
            fig.suptitle(page_title, fontsize=7, y=0.98)

            for i, ax in enumerate(axes.flat):
                if i < len(page_indices):
                    draw_event(ax, int(page_indices[i]), branches, meta)
                else:
                    ax.set_visible(False)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            if (page + 1) % 100 == 0 or page == n_pages - 1:
                pct = 100 * (page + 1) / n_pages
                print(f"  Page {page+1}/{n_pages} ({pct:.0f}%)", flush=True)

        # --- Dedicated edge page (incomplete neighborhoods) ---
        edge_to_show = edge_indices_all[:EVENTS_PER_PAGE]
        fig, axes = plt.subplots(EVENTS_PER_COL, EVENTS_PER_ROW,
                                 figsize=(11, 8.5),
                                 gridspec_kw={"hspace": 0.4, "wspace": 0.15})
        fig.suptitle(
            f"EDGE EVENTS — Incomplete neighborhoods (< {expected_size} pixels)   "
            f"({n_edge} total in file, showing {len(edge_to_show)})",
            fontsize=7, y=0.98,
        )

        for i, ax in enumerate(axes.flat):
            if i < len(edge_to_show):
                draw_event(ax, int(edge_to_show[i]), branches, meta)
            else:
                ax.set_visible(False)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"Done: {out_pdf}")


if __name__ == "__main__":
    main()
