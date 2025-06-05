#!/usr/bin/env python3
"""
Statistical analysis of energy deposition and pixel hit data.
This script provides statistical analysis and visualization capabilities
for GEANT4 epicToy simulation data. All Gaussian fitting is performed
using ODRPACK95 Fortran library in the C++ code.

This module provides functions to create the following plots:
1. Chi-squared distribution with statistics
2. P-value distribution with statistics  
3. GaussTrueDistance distribution with statistics
4. Residual hitmap for fit analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uproot
import pandas as pd
import os
import argparse

def load_root_data(filename):
    """
    Load data from ROOT file and return as pandas DataFrame.
    
    Args:
        filename (str): Path to ROOT file
    
    Returns:
        pd.DataFrame: DataFrame containing all branch data
    """
    try:
        with uproot.open(filename) as file:
            tree = file['Hits']
            data = tree.arrays(library="pd")
            print(f"Loaded {len(data)} events from {filename}")
            return data
    except Exception as e:
        print(f"Error loading ROOT file {filename}: {e}")
        return None

def calculate_statistics(data):
    """
    Calculate mean, standard deviation, and median for given data.
    
    Args:
        data (array-like): Input data array
    
    Returns:
        tuple: (mean, std, median)
    """
    data_clean = data[~np.isnan(data)]  # Remove NaN values
    if len(data_clean) == 0:
        return 0, 0, 0
    
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    median_val = np.median(data_clean)
    
    return mean_val, std_val, median_val

def plot_chi2_distribution(data, output_dir="plots", save_format="png"):
    """
    Plot chi2red distribution along with mean, stdev, and median in the histogram's legend.
    
    Args:
        data (pd.DataFrame): DataFrame containing ROOT file data
        output_dir (str): Directory to save plots
        save_format (str): Format to save plots (png, pdf, svg)
    """
    plt.figure(figsize=(12, 8))
    
    # Get chi2red data (use all data version for comprehensive analysis)
    chi2_data = data['FitChi2red'].dropna()
    
    if len(chi2_data) == 0:
        print("No valid chi2red data found")
        return
    
    # Calculate statistics
    mean_val, std_val, median_val = calculate_statistics(chi2_data)
    
    # Create histogram
    plt.hist(chi2_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Set labels and title
    plt.xlabel('Chi-squared Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Chi-squared Values from Gaussian Fits', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Create comprehensive legend
    legend_text = f'Mean: {mean_val:.3f}\nStd Dev: {std_val:.3f}\nMedian: {median_val:.3f}\nN Events: {len(chi2_data)}'
    plt.text(0.65, 0.95, legend_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"chi2_distribution.{save_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"chi2red distribution plot saved to {filename}")

def plot_pvalue_distribution(data, output_dir="plots", save_format="png"):
    """
    Distribution of P-values along with mean, stdev, median in the histogram's legend.
    
    Args:
        data (pd.DataFrame): DataFrame containing ROOT file data
        output_dir (str): Directory to save plots
        save_format (str): Format to save plots (png, pdf, svg)
    """
    plt.figure(figsize=(12, 8))
    
    # Get P-value data (use all data version for comprehensive analysis)
    pvalue_data = data['FitPp'].dropna()
    
    if len(pvalue_data) == 0:
        print("No valid P-value data found")
        return
    
    # Calculate statistics
    mean_val, std_val, median_val = calculate_statistics(pvalue_data)
    
    # Create histogram
    plt.hist(pvalue_data, bins=50, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Add reference line for uniform distribution expectation
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Uniform Expectation')
    
    # Set labels and title
    plt.xlabel('P-value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of P-values from Gaussian Fits', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Create comprehensive legend
    legend_text = f'Mean: {mean_val:.3f}\nStd Dev: {std_val:.3f}\nMedian: {median_val:.3f}\nN Events: {len(pvalue_data)}'
    plt.text(0.65, 0.95, legend_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"pvalue_distribution.{save_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"P-value distribution plot saved to {filename}")

def plot_gauss_true_distance_distribution(data, output_dir="plots", save_format="png"):
    """
    Distribution of GaussTrueDistance along with mean, stdev, median in the histogram's legend.
    
    Args:
        data (pd.DataFrame): DataFrame containing ROOT file data
        output_dir (str): Directory to save plots
        save_format (str): Format to save plots (png, pdf, svg)
    """
    plt.figure(figsize=(12, 8))
    
    # Get GaussTrueDistance data
    distance_data = data['GaussTrueDistance'].dropna()
    
    if len(distance_data) == 0:
        print("No valid GaussTrueDistance data found")
        return
    
    # Calculate statistics
    mean_val, std_val, median_val = calculate_statistics(distance_data)
    
    # Create histogram
    plt.hist(distance_data, bins=50, alpha=0.7, density=True, color='mediumpurple', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Set labels and title
    plt.xlabel('Distance from Gaussian Center to True Position [mm]', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Distance from Gaussian Fit Center to True Hit Position', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Create comprehensive legend
    legend_text = f'Mean: {mean_val:.3f} mm\nStd Dev: {std_val:.3f} mm\nMedian: {median_val:.3f} mm\nN Events: {len(distance_data)}'
    plt.text(0.65, 0.95, legend_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"gauss_true_distance_distribution.{save_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"GaussTrueDistance distribution plot saved to {filename}")

def calculate_residuals(data):
    """
    Calculate residuals from charge sharing data and fit parameters.
    
    Args:
        data (pd.DataFrame): DataFrame containing ROOT file data
    
    Returns:
        tuple: (event_ids, data_point_indices, residuals)
    """
    event_ids = []
    data_point_indices = []
    residuals = []
    
    print("Calculating residuals from charge sharing data...")
    
    # Access data more efficiently to avoid the awkward array reshape issue
    successful_fits = data['FitSuccessful'].values
    event_id_vals = data['EventID'].values
    
    # Fit parameters
    fit_amplitude = data['FitAmplitude'].values
    fit_x0 = data['FitX0'].values
    fit_y0 = data['FitY0'].values
    fit_sigma_x = data['FitSigmaX'].values
    fit_sigma_y = data['FitSigmaY'].values
    fit_theta = data['FitTheta'].values
    fit_offset = data['FitOffset'].values
    
    # Process events one by one using index-based access
    for i in range(len(data)):
        # Skip if fit was not successful
        if not successful_fits[i]:
            continue
            
        # Get neighborhood data using iloc to avoid awkward array issues
        try:
            charge_values = data['GridNeighborhoodChargeValues'].iloc[i]
            pixel_i_list = data['GridNeighborhoodPixelI'].iloc[i]
            pixel_j_list = data['GridNeighborhoodPixelJ'].iloc[i]
        except Exception:
            continue
        
        if len(charge_values) == 0:
            continue
            
        if len(pixel_i_list) != len(charge_values) or len(pixel_j_list) != len(charge_values):
            continue
        
        # Convert pixel indices to physical coordinates
        # This is approximate - should use actual detector geometry parameters
        pixel_size = 0.1  # mm, should get from data
        pixel_spacing = 0.5  # mm, should get from data
        pixel_corner_offset = 0.1  # mm, should get from data
        
        for k in range(len(charge_values)):
            # Calculate physical coordinates from pixel indices
            x_coord = pixel_corner_offset + pixel_i_list[k] * pixel_spacing + pixel_size / 2
            y_coord = pixel_corner_offset + pixel_j_list[k] * pixel_spacing + pixel_size / 2
            
            # Calculate expected value from Gaussian fit
            cos_theta = np.cos(fit_theta[i])
            sin_theta = np.sin(fit_theta[i])
            
            # Rotate coordinates
            x_rot = cos_theta * (x_coord - fit_x0[i]) + sin_theta * (y_coord - fit_y0[i])
            y_rot = -sin_theta * (x_coord - fit_x0[i]) + cos_theta * (y_coord - fit_y0[i])
            
            # Calculate Gaussian value
            gaussian_val = (fit_amplitude[i] * 
                          np.exp(-0.5 * ((x_rot / fit_sigma_x[i])**2 + (y_rot / fit_sigma_y[i])**2)) + 
                          fit_offset[i])
            
            # Calculate residual
            residual = charge_values[k] - gaussian_val
            
            event_ids.append(event_id_vals[i])
            data_point_indices.append(k)
            residuals.append(residual)
    
    return np.array(event_ids), np.array(data_point_indices), np.array(residuals)

def plot_residual_hitmap(data, output_dir="plots", save_format="png"):
    """
    Plot residual hitmap: y axis event/fit ID number; x axis is the data point index; z axis is the residual.
    
    Args:
        data (pd.DataFrame): DataFrame containing ROOT file data
        output_dir (str): Directory to save plots
        save_format (str): Format to save plots (png, pdf, svg)
    """
    print("Creating residual hitmap...")
    
    # Calculate residuals
    event_ids, data_point_indices, residuals = calculate_residuals(data)
    
    if len(residuals) == 0:
        print("No residual data available for plotting")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Scatter plot of residuals
    scatter = ax1.scatter(data_point_indices, event_ids, c=residuals, cmap='RdBu_r', 
                         alpha=0.7, s=10, vmin=np.percentile(residuals, 5), 
                         vmax=np.percentile(residuals, 95))
    
    ax1.set_xlabel('Data Point Index', fontsize=12)
    ax1.set_ylabel('Event ID', fontsize=12)
    ax1.set_title('Residual Hitmap: Fit Residuals by Event and Data Point', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Residual (Data - Fit)', fontsize=11)
    
    # Plot 2: 2D histogram/heatmap
    # Create binned version for better visualization
    x_bins = np.linspace(data_point_indices.min(), data_point_indices.max(), 50)
    y_bins = np.linspace(event_ids.min(), event_ids.max(), 100)
    
    # Create 2D histogram of residuals
    hist, x_edges, y_edges = np.histogram2d(data_point_indices, event_ids, bins=[x_bins, y_bins], weights=residuals)
    counts, _, _ = np.histogram2d(data_point_indices, event_ids, bins=[x_bins, y_bins])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        hist_avg = np.divide(hist, counts, out=np.zeros_like(hist), where=counts!=0)
    
    # Plot heatmap
    im = ax2.imshow(hist_avg.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   vmin=np.nanpercentile(hist_avg, 5), vmax=np.nanpercentile(hist_avg, 95))
    
    ax2.set_xlabel('Data Point Index', fontsize=12)
    ax2.set_ylabel('Event ID', fontsize=12)
    ax2.set_title('Residual Heatmap: Average Residuals in Bins', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Average Residual (Data - Fit)', fontsize=11)
    
    # Add statistics text
    residual_stats = f'Residual Statistics:\nMean: {np.mean(residuals):.3e}\nStd: {np.std(residuals):.3e}\nN Points: {len(residuals)}'
    fig.text(0.02, 0.98, residual_stats, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"residual_hitmap.{save_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Residual hitmap saved to {filename}")

def create_all_plots(root_filename, output_dir="plots", save_format="png"):
    """
    Create all requested statistical plots from the ROOT file.
    
    Args:
        root_filename (str): Path to ROOT file
        output_dir (str): Directory to save plots
        save_format (str): Format to save plots (png, pdf, svg)
    """
    print(f"Loading data from {root_filename}...")
    data = load_root_data(root_filename)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Creating statistical plots in {output_dir}/")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all plots
    print("\n1. Creating chi-squared distribution plot...")
    plot_chi2_distribution(data, output_dir, save_format)
    
    print("\n2. Creating P-value distribution plot...")
    plot_pvalue_distribution(data, output_dir, save_format)
    
    print("\n3. Creating GaussTrueDistance distribution plot...")
    plot_gauss_true_distance_distribution(data, output_dir, save_format)
    
    print("\n4. Creating residual hitmap...")
    plot_residual_hitmap(data, output_dir, save_format)
    
    print(f"\nAll plots completed and saved to {output_dir}/")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description="Create statistical plots from epicToy ROOT file")
    parser.add_argument("root_file", help="Path to ROOT file")
    parser.add_argument("-o", "--output", default="plots", help="Output directory for plots")
    parser.add_argument("-f", "--format", default="png", choices=["png", "pdf", "svg"], 
                       help="Output format for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return
    
    create_all_plots(args.root_file, args.output, args.format)

if __name__ == "__main__":
    main()

