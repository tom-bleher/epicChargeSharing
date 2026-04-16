#!/usr/bin/env python3
"""Analyze threshold sweep: fit Gaussian to ReconTrueDelta X/Y, plot resolution vs threshold sigma."""

import sys
import os
import glob
import re
import numpy as np
import uproot
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_resolution(values, nbins=200, range_nsigma=5):
    """Fit a Gaussian to the core of the distribution, return (mu, sigma, sigma_err, fit_data)."""
    values = values[np.isfinite(values)]
    if len(values) < 50:
        return np.nan, np.nan, np.nan, None

    med = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    sig_est = iqr / 1.35
    if sig_est <= 0:
        sig_est = np.std(values)
    if sig_est <= 0:
        return np.nan, np.nan, np.nan, None

    lo, hi = med - range_nsigma * sig_est, med + range_nsigma * sig_est
    counts, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    try:
        popt, pcov = curve_fit(gaussian, centers, counts,
                               p0=[counts.max(), med, sig_est],
                               maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        fit_data = {"centers": centers, "counts": counts, "edges": edges, "popt": popt}
        return popt[1], abs(popt[2]), perr[2], fit_data
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, None


def plot_fit(fit_data, label, threshold_sigma, n_events, mu, sigma, sigma_err, out_path):
    """Save a plot of the histogram + Gaussian fit overlay."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = fit_data["centers"]
    counts = fit_data["counts"]
    popt = fit_data["popt"]
    edges = fit_data["edges"]

    ax.bar(centers, counts, width=(edges[1] - edges[0]), alpha=0.6, color="steelblue", label="Data")
    x_fine = np.linspace(centers[0], centers[-1], 500)
    ax.plot(x_fine, gaussian(x_fine, *popt), "r-", lw=2, label="Gaussian fit")
    ax.axvline(mu, color="red", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel(f"ReconTrueDelta{label} (µm)")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"Threshold {threshold_sigma}σ — {label} residual\n"
        f"µ = {mu:.2f} µm, σ = {sigma:.2f} ± {sigma_err:.2f} µm  (N = {n_events})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_file(root_path, threshold_sigma, plot_dir):
    """Extract resolution from a single ROOT file, save fit plots."""
    with uproot.open(root_path) as f:
        tree = f["Hits"]
        keys = tree.keys()

        results = {}
        for branch, label in [("ReconTrueDeltaX_2D", "X"), ("ReconTrueDeltaY_2D", "Y")]:
            if branch not in keys:
                print(f"  Warning: {branch} not found in {root_path}")
                results[label] = (np.nan, np.nan, np.nan)
                continue

            vals = tree[branch].array(library="np")
            vals = vals[np.isfinite(vals)]
            vals_um = vals * 1000.0

            mu, sigma, sigma_err, fit_data = fit_resolution(vals_um)
            results[label] = (mu, sigma, sigma_err)
            print(f"  {label}: mu={mu:.2f} µm, sigma={sigma:.2f} ± {sigma_err:.2f} µm  (N={len(vals)})")

            if fit_data is not None:
                plot_path = os.path.join(plot_dir, f"fit_{threshold_sigma}sigma_{label}.png")
                plot_fit(fit_data, label, threshold_sigma, len(vals), mu, sigma, sigma_err, plot_path)
                print(f"    -> {plot_path}")

        return results


def main():
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "threshold_sweep"
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sweep_dir, "resolution_vs_threshold.pdf")

    files = sorted(glob.glob(os.path.join(sweep_dir, "threshold_*sigma.root")))
    if not files:
        print(f"No threshold_*sigma.root files found in {sweep_dir}")
        sys.exit(1)

    plot_dir = os.path.join(sweep_dir, "fit_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Found {len(files)} threshold sweep files in {sweep_dir}")

    sigmas = []
    res_x, res_x_err = [], []
    res_y, res_y_err = [], []

    for fpath in files:
        match = re.search(r'threshold_([\d.]+)sigma', os.path.basename(fpath))
        if not match:
            continue
        threshold_sigma = float(match.group(1))
        print(f"\n{os.path.basename(fpath)} (threshold = {threshold_sigma}σ):")

        results = analyze_file(fpath, threshold_sigma, plot_dir)
        sigmas.append(threshold_sigma)
        res_x.append(results["X"][1])
        res_x_err.append(results["X"][2])
        res_y.append(results["Y"][1])
        res_y_err.append(results["Y"][2])

    sigmas = np.array(sigmas)
    res_x = np.array(res_x)
    res_x_err = np.array(res_x_err)
    res_y = np.array(res_y)
    res_y_err = np.array(res_y_err)

    # Summary plot: resolution vs threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigmas, res_x, yerr=res_x_err, fmt="o-", capsize=4, label="X resolution")
    ax.errorbar(sigmas, res_y, yerr=res_y_err, fmt="s-", capsize=4, label="Y resolution")
    ax.set_xlabel("Readout threshold (σ)")
    ax.set_ylabel("Position resolution σ (µm)")
    ax.set_title("Spatial Resolution vs Readout Threshold\n(2D Gaussian fit to ReconTrueDelta)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(sigmas.min() - 0.3, sigmas.max() + 0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150)
    print(f"\nSummary plot saved to {out_pdf}")

    # Also save as PNG
    png_path = out_pdf.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150)
    print(f"Summary plot saved to {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
