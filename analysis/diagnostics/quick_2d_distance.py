#!/usr/bin/env python3
"""Quick analysis: ReconTrueDistance_2D = sqrt(dx² + dy²) with Rayleigh fit."""

import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
from scipy.optimize import curve_fit

# Load data
f = uproot.open("build/epicChargeSharing.root")
tree = f["Hits"]
dx = tree["ReconTrueDeltaX_2D"].array(library="np")
dy = tree["ReconTrueDeltaY_2D"].array(library="np")

# Filter out NaN/inf
mask = np.isfinite(dx) & np.isfinite(dy)
dx = dx[mask]
dy = dy[mask]
print(f"Valid events: {len(dx)} out of {len(mask)}")

distance = np.sqrt(dx**2 + dy**2)

# Also compute individual sigmas for reference
sigma_x = np.std(dx)
sigma_y = np.std(dy)
print(f"sigma_x = {sigma_x*1e3:.2f} µm,  sigma_y = {sigma_y*1e3:.2f} µm")
print(f"Distance: mean = {np.mean(distance)*1e3:.2f} µm, median = {np.median(distance)*1e3:.2f} µm")

# --- Rayleigh fit ---
# Rayleigh PDF: f(r; sigma) = (r/sigma²) * exp(-r²/(2*sigma²))
# The MLE for sigma is: sigma = sqrt(sum(r²) / (2*N))
sigma_rayleigh_mle = np.sqrt(np.sum(distance**2) / (2 * len(distance)))
print(f"\nRayleigh MLE sigma = {sigma_rayleigh_mle*1e3:.2f} µm")
print(f"  → expected mean  = sigma*sqrt(pi/2) = {sigma_rayleigh_mle * np.sqrt(np.pi/2) * 1e3:.2f} µm")
print(f"  → expected mode  = sigma = {sigma_rayleigh_mle*1e3:.2f} µm")

# scipy Rayleigh fit for comparison (loc, scale)
loc_fit, scale_fit = rayleigh.fit(distance, floc=0)
print(f"scipy Rayleigh fit: loc={loc_fit*1e3:.2f} µm, scale(=sigma)={scale_fit*1e3:.2f} µm")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) 2D scatter of dx vs dy
ax = axes[0]
ax.scatter(dx * 1e3, dy * 1e3, s=0.3, alpha=0.3, rasterized=True)
circle = plt.Circle((0, 0), sigma_rayleigh_mle * 1e3, fill=False, color='red',
                     linestyle='--', linewidth=2, label=f'σ_Rayleigh = {sigma_rayleigh_mle*1e3:.1f} µm')
ax.add_patch(circle)
ax.set_xlabel("ΔX (µm)")
ax.set_ylabel("ΔY (µm)")
ax.set_title("Recon − True (2D)")
ax.set_aspect('equal')
ax.legend()
lim = max(np.percentile(np.abs(dx), 99), np.percentile(np.abs(dy), 99)) * 1e3 * 1.2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

# 2) Distance histogram + Rayleigh fit
ax = axes[1]
# Use µm for plotting
dist_um = distance * 1e3
n_bins = 100
counts, bin_edges, _ = ax.hist(dist_um, bins=n_bins, density=True, alpha=0.7,
                                color='steelblue', label='Data')
r_plot = np.linspace(0, bin_edges[-1], 500)
# Rayleigh PDF in µm units
sigma_um = sigma_rayleigh_mle * 1e3
rayleigh_pdf = (r_plot / sigma_um**2) * np.exp(-r_plot**2 / (2 * sigma_um**2))
ax.plot(r_plot, rayleigh_pdf, 'r-', linewidth=2,
        label=f'Rayleigh (σ={sigma_um:.1f} µm)')
ax.axvline(sigma_um, color='red', linestyle=':', alpha=0.5, label=f'mode = {sigma_um:.1f} µm')
ax.axvline(sigma_um * np.sqrt(np.pi/2), color='orange', linestyle=':', alpha=0.5,
           label=f'mean = {sigma_um * np.sqrt(np.pi/2):.1f} µm')
ax.set_xlabel("Distance |Recon − True| (µm)")
ax.set_ylabel("Probability density")
ax.set_title("ReconTrueDistance_2D")
ax.legend(fontsize=9)

# 3) Individual dx, dy histograms
ax = axes[2]
bins_1d = np.linspace(-lim, lim, 100)
ax.hist(dx * 1e3, bins=bins_1d, density=True, alpha=0.6, label=f'ΔX (σ={sigma_x*1e3:.1f} µm)')
ax.hist(dy * 1e3, bins=bins_1d, density=True, alpha=0.6, label=f'ΔY (σ={sigma_y*1e3:.1f} µm)')
# Gaussian overlay
from scipy.stats import norm
x_g = np.linspace(-lim, lim, 300)
ax.plot(x_g, norm.pdf(x_g, 0, sigma_x * 1e3), 'C0-', linewidth=2)
ax.plot(x_g, norm.pdf(x_g, 0, sigma_y * 1e3), 'C1-', linewidth=2)
ax.set_xlabel("Δ (µm)")
ax.set_ylabel("Probability density")
ax.set_title("Individual ΔX, ΔY")
ax.legend()

# Goodness-of-fit: KS test
from scipy.stats import kstest
ks_stat, ks_pval = kstest(distance, 'rayleigh', args=(0, sigma_rayleigh_mle))
fig.suptitle(f"2D Reconstruction Distance  |  Rayleigh σ = {sigma_um:.1f} µm  |  "
             f"KS p-value = {ks_pval:.4f}", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("build/recon_true_distance_2d.png", dpi=150, bbox_inches='tight')
print(f"\nKS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}")
print("Plot saved to build/recon_true_distance_2d.png")
plt.show()
