import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse

# Gauss model for curve_fit

def gauss(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B


def load_branches(tree, branches, entry_start, entry_stop):
    """Utility to load selected branches for a range of entries."""
    data = {}
    for br in branches:
        data[br] = tree[br].array(library="np", entry_start=entry_start, entry_stop=entry_stop)
    return data


def build_line_masks(radius):
    """Return boolean masks (row, column, main diag, second diag) for flattened grid indices."""
    gsize = 2 * radius + 1
    # Precompute di, dj arrays for indices 0..gsize**2-1
    idx = np.arange(gsize ** 2)
    di = idx // gsize - radius
    dj = idx % gsize - radius
    row_mask = dj == 0
    col_mask = di == 0
    main_diag_mask = di == dj
    sec_diag_mask = di == -dj
    return row_mask, col_mask, main_diag_mask, sec_diag_mask, di, dj


def analyze_events(root_path, num_events=10, pixel_spacing=0.5):
    file = uproot.open(root_path)
    tree = file["Hits"]

    # Determine total events available
    total_entries = tree.num_entries
    num_events = min(num_events, total_entries)

    # Load required branches for first num_events entries
    branches = [
        "PixelX",
        "PixelY",
        "NeighborhoodCharges",
        "SelectedRadius",
    ]
    arrays = load_branches(tree, branches, 0, num_events)

    # If SelectedRadius might be missing (older files), fill with default 4
    if "SelectedRadius" not in arrays:
        arrays["SelectedRadius"] = np.full(num_events, 4, dtype=np.int32)

    for evt in range(num_events):
        pixel_x = arrays["PixelX"][evt]
        pixel_y = arrays["PixelY"][evt]
        charges = arrays["NeighborhoodCharges"][evt]
        radius = int(arrays["SelectedRadius"][evt]) if arrays["SelectedRadius"] is not None else 4

        if len(charges) == 0:
            print(f"Event {evt}: no charge data, skipping")
            continue

        gsize = 2 * radius + 1
        expected_len = gsize ** 2
        if len(charges) != expected_len:
            print(f"Event {evt}: unexpected charge array length {len(charges)} (expected {expected_len}), skipping")
            continue

        # Build masks and di/dj arrays
        row_mask, col_mask, main_diag_mask, sec_diag_mask, di, dj = build_line_masks(radius)

        # Convert to numpy array of float (Coulombs)
        charges_arr = np.asarray(charges, dtype=float)

        # Filter invalid markers (-999) and non-positive charges
        valid = (charges_arr > 0) & (charges_arr > -998)

        # X reconstruction: use row, main diag, sec diag
        mask_x = (row_mask | main_diag_mask | sec_diag_mask) & valid
        x_poss = pixel_x + di[mask_x] * pixel_spacing
        x_charges = charges_arr[mask_x]

        # Y reconstruction: use column, main diag, sec diag
        mask_y = (col_mask | main_diag_mask | sec_diag_mask) & valid
        y_poss = pixel_y + dj[mask_y] * pixel_spacing
        y_charges = charges_arr[mask_y]

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # X plot
        ax = axes[0]
        # Scatter for each line type with colors
        for name, msk, color in [
            ("Row", row_mask & valid, "tab:blue"),
            ("MainDiag", main_diag_mask & valid, "tab:orange"),
            ("SecDiag", sec_diag_mask & valid, "tab:green"),
        ]:
            ax.scatter(
                pixel_x + di[msk] * pixel_spacing,
                charges_arr[msk],
                label=name,
                alpha=0.7,
                s=20,
                color=color,
            )

        #  Gauss to combined x
        if len(x_poss) >= 4:
            try:
                A0 = x_charges.max() - x_charges.min()
                mu0 = pixel_x
                sigma0 = pixel_spacing * 0.5
                B0 = x_charges.min()
                popt, _ = curve_fit(
                    gauss,
                    x_poss,
                    x_charges,
                    p0=[A0, mu0, sigma0, B0],
                    bounds=([0, mu0 - pixel_spacing, 1e-6, 0], [np.inf, mu0 + pixel_spacing, np.inf, np.inf]),
                )
                xs = np.linspace(x_poss.min() - pixel_spacing, x_poss.max() + pixel_spacing, 300)
                ax.plot(xs, gauss(xs, *popt), "k--", label=f"Gauss fit (σ={popt[2]:.3f} mm)")
            except Exception as e:
                print(f"Event {evt}: Gauss fit X failed: {e}")

        ax.set_xlabel("X pos [mm]")
        ax.set_ylabel("Charge [C]")
        ax.set_title(f"Event {evt} - X lines")
        ax.legend(fontsize=8)

        # Y plot
        ay = axes[1]
        for name, msk, color in [
            ("Col", col_mask & valid, "tab:blue"),
            ("MainDiag", main_diag_mask & valid, "tab:orange"),
            ("SecDiag", sec_diag_mask & valid, "tab:green"),
        ]:
            ay.scatter(
                pixel_y + dj[msk] * pixel_spacing,
                charges_arr[msk],
                label=name,
                alpha=0.7,
                s=20,
                color=color,
            )

        if len(y_poss) >= 4:
            try:
                A0 = y_charges.max() - y_charges.min()
                mu0 = pixel_y
                sigma0 = pixel_spacing * 0.5
                B0 = y_charges.min()
                popt, _ = curve_fit(
                    gauss,
                    y_poss,
                    y_charges,
                    p0=[A0, mu0, sigma0, B0],
                    bounds=([0, mu0 - pixel_spacing, 1e-6, 0], [np.inf, mu0 + pixel_spacing, np.inf, np.inf]),
                )
                ys = np.linspace(y_poss.min() - pixel_spacing, y_poss.max() + pixel_spacing, 300)
                ay.plot(ys, gauss(ys, *popt), "k--", label=f"Gauss fit (σ={popt[2]:.3f} mm)")
            except Exception as e:
                print(f"Event {evt}: Gauss fit Y failed: {e}")

        ay.set_xlabel("Y pos [mm]")
        ay.set_ylabel("Charge [C]")
        ay.set_title(f"Event {evt} - Y lines")
        ay.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overplot charge lines and Gauss fit for EPIC simulation ROOT output.")
    parser.add_argument(
        "rootfile",
        type=str,
        default="epicChargeSharingOutput.root",
        help="Path to ROOT file produced by simulation (default: epicChargeSharingOutput.root)",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=10,
        help="Number of events to process (default: 10)",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=0.5,
        help="Pixel spacing in mm (default: 0.5 mm)",
    )

    args = parser.parse_args()

    analyze_events(args.rootfile, num_events=args.events, pixel_spacing=args.pixel_spacing) 