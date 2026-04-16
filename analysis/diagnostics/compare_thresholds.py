#!/usr/bin/env python3
"""Compare the same event across different threshold settings side by side.

Requires ROOT files generated with the same random seed so event N
corresponds to the same physical hit across all files.

Usage:
    python compare_thresholds.py <sweep_dir> [output.pdf] [max_events]
"""

import sys
import os
import re
import numpy as np
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

GRID_SIDE = 5
ROWS_PER_PAGE = 6  # events per page
DEFAULT_PITCH = 0.5
DEFAULT_PIXEL_SIZE = 0.10


def read_metadata(root_file):
    meta = {"pitch": DEFAULT_PITCH, "pixel_size": DEFAULT_PIXEL_SIZE}
    with uproot.open(root_file, minimal_ttree_metadata=False) as f:
        tree = f["Hits"]
        try:
            user_info = tree.member("fUserInfo", none_if_missing=True)
            if user_info is not None:
                for obj in user_info:
                    try:
                        name = obj.member("fName")
                        val = float(obj.member("fVal"))
                        if name == "GridPixelSpacing_mm":
                            meta["pitch"] = val
                        elif name == "GridPixelSize_mm":
                            meta["pixel_size"] = val
                    except Exception:
                        pass
        except Exception:
            pass
        # Derive pitch from data if not found
        if meta["pitch"] == DEFAULT_PITCH:
            try:
                px = tree["NeighborhoodPixelX"].array(entry_stop=50, library="ak")
                ns = tree["NeighborhoodSize"].array(entry_stop=50, library="ak")
                for i in range(min(50, len(ns))):
                    n = int(ns[i])
                    if n < 4:
                        continue
                    side = int(np.sqrt(n))
                    x = ak.to_numpy(px[i]).reshape(side, side)
                    derived = abs(float(x[1, 0] - x[0, 0]))
                    if derived > 0:
                        meta["pitch"] = derived
                        break
            except Exception:
                pass
    return meta


def load_data(root_file):
    with uproot.open(root_file) as f:
        tree = f["Hits"]
        return tree.arrays(
            ["Fi", "Qi", "Qn", "Qf", "NeighborhoodPixelX", "NeighborhoodPixelY",
             "TrueX", "TrueY", "PixelX", "PixelY", "EnergyDeposited", "NeighborhoodSize",
             "NearestPixelI", "NearestPixelJ", "isPixelHit", "hitWithinDetector"],
            library="ak",
        )


def draw_event(ax, orig_idx, branches, meta, title=None):
    """Draw a single event's 5x5 neighborhood grid."""
    nsize = int(branches["NeighborhoodSize"][orig_idx])
    pitch = meta["pitch"]
    pad_frac = meta["pixel_size"] / pitch

    fi = ak.to_numpy(branches["Fi"][orig_idx])
    qf = ak.to_numpy(branches["Qf"][orig_idx])
    true_x = float(branches["TrueX"][orig_idx])
    true_y = float(branches["TrueY"][orig_idx])
    pix_x = ak.to_numpy(branches["NeighborhoodPixelX"][orig_idx])
    pix_y = ak.to_numpy(branches["NeighborhoodPixelY"][orig_idx])
    edep = float(branches["EnergyDeposited"][orig_idx])

    expected = GRID_SIDE * GRID_SIDE
    is_edge = nsize < expected
    if is_edge:
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

    px_arr = pix_x.reshape(GRID_SIDE, GRID_SIDE)
    py_arr = pix_y.reshape(GRID_SIDE, GRID_SIDE)
    x_vals = px_arr[:, 0]
    y_vals = py_arr[0, :]
    pitch_x = x_vals[1] - x_vals[0] if len(x_vals) > 1 else pitch
    pitch_y = y_vals[1] - y_vals[0] if len(y_vals) > 1 else pitch
    missing_mask = np.isnan(px_arr) | np.isnan(py_arr)

    ax.set_facecolor("white")
    for i in range(GRID_SIDE + 1):
        ax.axvline(i - 0.5, color="#cccccc", lw=0.3, ls=":", zorder=0)
        ax.axhline(i - 0.5, color="#cccccc", lw=0.3, ls=":", zorder=0)

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
                ax.add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1,
                             linewidth=0.4, edgecolor="#aaaaaa",
                             facecolor="none", hatch="//", zorder=1))
            elif active_mask[ix, iy]:
                ax.add_patch(Rectangle((pad_x, pad_y), pad_frac, pad_frac,
                             linewidth=0.5, edgecolor="blue",
                             facecolor=face_color, zorder=3))
            else:
                ax.add_patch(Rectangle((pad_x, pad_y), pad_frac, pad_frac,
                             linewidth=0.3, edgecolor="#bbbbbb",
                             facecolor="#f0f0f0", zorder=2))

    hit_ix = (true_x - x_vals[0]) / pitch_x
    hit_iy = (true_y - y_vals[0]) / pitch_y
    ax.plot(hit_ix, hit_iy, marker="+", color="lime", markersize=5,
            markeredgewidth=1.0, zorder=5)

    if is_edge:
        lw = 1.5
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

    if title is not None:
        ax.set_title(title, fontsize=4.5, pad=1)


def main():
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "threshold_sweep"
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else "threshold_comparison.pdf"
    max_events = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    # Discover ROOT files and sort by sigma
    files = []
    for fname in sorted(os.listdir(sweep_dir)):
        m = re.match(r"threshold_(\d+\.\d+)sigma\.root$", fname)
        if m:
            sigma = float(m.group(1))
            files.append((sigma, os.path.join(sweep_dir, fname)))

    if not files:
        print(f"No threshold_*.root files found in {sweep_dir}")
        sys.exit(1)

    sigmas = [s for s, _ in files]
    n_cols = len(files)
    print(f"Found {n_cols} threshold files: {', '.join(f'{s}σ' for s in sigmas)}")

    # Load all files
    all_data = {}
    meta = None
    for sigma, path in files:
        print(f"  Loading {os.path.basename(path)}...")
        all_data[sigma] = load_data(path)
        if meta is None:
            meta = read_metadata(path)

    # Find valid event indices (use first file as reference)
    ref = all_data[sigmas[0]]
    nsize = ak.to_numpy(ref["NeighborhoodSize"])
    expected = GRID_SIDE * GRID_SIDE
    valid_mask = nsize == expected
    valid_indices = np.where(valid_mask)[0]

    if max_events is not None:
        valid_indices = valid_indices[:max_events]

    n_events = len(valid_indices)
    n_pages = (n_events + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE

    print(f"  {n_events} events to compare, {n_pages} pages")
    print(f"  Layout: {ROWS_PER_PAGE} rows × {n_cols} columns per page")
    print(f"  Writing to {out_pdf}...", flush=True)

    # Write metadata to companion text file
    txt_path = out_pdf.replace(".pdf", ".txt")
    with open(txt_path, "w") as tf:
        tf.write(f"Threshold Comparison\n")
        tf.write(f"====================\n")
        tf.write(f"Seed: same across all files (same hit positions & noise)\n")
        tf.write(f"Pitch: {meta['pitch']} mm\n")
        tf.write(f"Pad size: {meta['pixel_size']} mm ({meta['pixel_size']/meta['pitch']*100:.0f}% fill)\n")
        tf.write(f"Thresholds: {', '.join(f'{s}σ' for s in sigmas)}\n")
        tf.write(f"Events shown: {n_events}\n")
        tf.write(f"Pages: {n_pages}\n")
        tf.write(f"Layout: {ROWS_PER_PAGE} rows × {n_cols} columns per page\n")
        tf.write(f"\nLegend:\n")
        tf.write(f"  Columns = threshold σ (increasing left→right)\n")
        tf.write(f"  Blue border = pixel above threshold\n")
        tf.write(f"  Green + = true hit position\n")
        tf.write(f"  Small square = metal pad ({meta['pixel_size']}mm within {meta['pitch']}mm pitch)\n")
        tf.write(f"  Hatched = missing pixel (detector edge)\n")
        tf.write(f"\nFiles:\n")
        for sigma, path in files:
            tf.write(f"  {sigma}σ: {path}\n")
    print(f"  Metadata written to {txt_path}")

    # 16:9 presentation aspect ratio
    FIG_W, FIG_H = 16, 9

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            start = page * ROWS_PER_PAGE
            end = min(start + ROWS_PER_PAGE, n_events)
            page_events = valid_indices[start:end]
            n_rows = len(page_events)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(FIG_W, FIG_H),
                gridspec_kw={
                    "hspace": 0.12, "wspace": 0.06,
                    "left": 0.03, "right": 0.97,
                    "top": 0.95, "bottom": 0.02,
                },
                squeeze=False,
            )

            # Minimal column headers on first row only
            for col, sigma in enumerate(sigmas):
                axes[0, col].text(
                    0.5, 1.08, f"{sigma}σ",
                    transform=axes[0, col].transAxes,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

            for row, evt_idx in enumerate(page_events):
                evt_idx = int(evt_idx)

                for col, sigma in enumerate(sigmas):
                    ax = axes[row, col]
                    data = all_data[sigma]
                    n_active = int(np.count_nonzero(
                        ak.to_numpy(data["Qf"][evt_idx]) != 0
                    ))
                    draw_event(ax, evt_idx, data, meta, title=None)
                    # Small active-pixel count inside the grid
                    ax.text(
                        0.02, 0.98, f"{n_active}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=5, color="#555555", zorder=7,
                    )

                # Event number label on the left edge of first column
                axes[row, 0].text(
                    -0.08, 0.5, f"#{evt_idx}",
                    transform=axes[row, 0].transAxes,
                    ha="right", va="center", fontsize=5.5, color="#333333",
                    rotation=0,
                )

            # Hide unused rows on the last page
            if n_rows < ROWS_PER_PAGE:
                for r in range(n_rows, ROWS_PER_PAGE):
                    if r < axes.shape[0]:
                        for c in range(n_cols):
                            axes[r, c].set_visible(False)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            pct = 100 * (page + 1) / n_pages
            print(f"  Page {page+1}/{n_pages} ({pct:.0f}%)", flush=True)

    print(f"Done: {out_pdf}")


if __name__ == "__main__":
    main()
