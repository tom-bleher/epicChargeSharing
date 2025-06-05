#!/usr/bin/env python3
"""
Post-processing plotting routine for Gaussian fit visualization of charge sharing in LGAD detectors.

This script creates plots for:
1. Gaussian curve estimation for central row (x-direction) with residuals
2. Gaussian curve estimation for central column (y-direction) with residuals

The plots show the fitted Gaussian curves overlaid on the actual charge data points 
from the neighborhood grid, along with residual plots showing fit quality.
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import argparse
from pathlib import Path
from scipy.stats import norm

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def gaussian_1d(x, amplitude, center, sigma, offset=0):
    """
    1D Gaussian function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amplitude: Gaussian amplitude
        center: Gaussian center (mean)
        sigma: Gaussian sigma (standard deviation)
        offset: Baseline offset
    
    Returns:
        Gaussian function values
    """
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2) + offset

def load_successful_fits(root_file):
    """
    Load data from ROOT file, filtering for successful 2D Gaussian fits.
    
    Args:
        root_file (str): Path to ROOT file
    
    Returns:
        dict: Dictionary containing arrays of fit data for successful fits
    """
    print(f"Loading data from {root_file}...")
    
    try:
        with uproot.open(root_file) as file:
            tree = file['Hits']
            
            # Load all relevant branches
            branches = [
                'TrueX', 'TrueY',  # True hit positions
                'PixelX', 'PixelY',  # Nearest pixel positions
                'IsPixelHit',  # Pixel hit classification
                'Fit2D_XCenter', 'Fit2D_XSigma', 'Fit2D_XAmplitude',
                'Fit2D_XCenterErr', 'Fit2D_XSigmaErr', 'Fit2D_XAmplitudeErr',
                'Fit2D_XChi2red', 'Fit2D_XNPoints',
                'Fit2D_YCenter', 'Fit2D_YSigma', 'Fit2D_YAmplitude',
                'Fit2D_YCenterErr', 'Fit2D_YSigmaErr', 'Fit2D_YAmplitudeErr',
                'Fit2D_YChi2red', 'Fit2D_YNPoints',
                'Fit2D_Successful',  # Whether 2D fitting was successful
                'NonPixel_GridNeighborhoodPixelI', 'NonPixel_GridNeighborhoodPixelJ',  # Pixel indices
                'NonPixel_GridNeighborhoodCharge',  # Charge values in Coulombs
                'NonPixel_GridNeighborhoodDistances'  # Distances from hit to pixels
            ]
            
            data = tree.arrays(branches, library="np")
            
            print(f"Total events loaded: {len(data['TrueX'])}")
            
            # Filter for successful non-pixel fits
            is_non_pixel = ~data['IsPixelHit'] 
            is_fit_successful = data['Fit2D_Successful']
            valid_mask = is_non_pixel & is_fit_successful
            
            print(f"Non-pixel events: {np.sum(is_non_pixel)}")
            print(f"Successful 2D fits: {np.sum(is_fit_successful)}")
            print(f"Valid events for plotting: {np.sum(valid_mask)}")
            
            if np.sum(valid_mask) == 0:
                print("Warning: No successful 2D fits found for non-pixel events!")
                return None
            
            # Extract valid data
            filtered_data = {}
            for key, values in data.items():
                if key.startswith('NonPixel_GridNeighborhood'):
                    # These are jagged arrays - keep them as is but filter by event
                    filtered_data[key] = values[valid_mask]
                else:
                    # Regular arrays - filter normally
                    filtered_data[key] = values[valid_mask]
            
            return filtered_data
            
    except Exception as e:
        print(f"Error loading ROOT file: {e}")
        return None

def extract_row_column_data(event_idx, data, neighborhood_radius=4):
    """
    Extract charge data for central row and column from neighborhood grid data.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (row_data, col_data) where each is (positions, charges) for central row/column
    """
    # Get neighborhood data for this event
    pixel_i = data['NonPixel_GridNeighborhoodPixelI'][event_idx]
    pixel_j = data['NonPixel_GridNeighborhoodPixelJ'][event_idx]
    charges = data['NonPixel_GridNeighborhoodCharge'][event_idx]
    
    # Get pixel positions (assuming they correspond to pixel centers)
    nearest_pixel_x = data['PixelX'][event_idx]
    nearest_pixel_y = data['PixelY'][event_idx]
    
    # Find the center pixel indices
    center_i = pixel_i[len(pixel_i)//2]  # Middle element should be center
    center_j = pixel_j[len(pixel_j)//2]
    
    # Grid size
    grid_size = 2 * neighborhood_radius + 1
    
    # Extract central row data (constant j, varying i)
    row_positions = []
    row_charges = []
    for idx, (i, j, charge) in enumerate(zip(pixel_i, pixel_j, charges)):
        if j == center_j and charge > 0:  # Central row, non-zero charge
            # Calculate actual pixel position (relative to nearest pixel)
            pixel_spacing = 0.5  # mm - this should match detector parameters
            x_pos = nearest_pixel_x + (i - center_i) * pixel_spacing
            row_positions.append(x_pos)
            row_charges.append(charge)
    
    # Extract central column data (constant i, varying j)
    col_positions = []
    col_charges = []
    for idx, (i, j, charge) in enumerate(zip(pixel_i, pixel_j, charges)):
        if i == center_i and charge > 0:  # Central column, non-zero charge
            # Calculate actual pixel position (relative to nearest pixel)
            y_pos = nearest_pixel_y + (j - center_j) * pixel_spacing
            col_positions.append(y_pos)
            col_charges.append(charge)
    
    # Sort by position
    if row_positions:
        row_sorted = sorted(zip(row_positions, row_charges))
        row_positions, row_charges = zip(*row_sorted)
    
    if col_positions:
        col_sorted = sorted(zip(col_positions, col_charges))
        col_positions, col_charges = zip(*col_sorted)
    
    return (np.array(row_positions), np.array(row_charges)), (np.array(col_positions), np.array(col_charges))

def calculate_residuals(positions, charges, fit_params):
    """
    Calculate residuals between data and fitted Gaussian.
    
    Args:
        positions (array): Position values
        charges (array): Charge values (data)
        fit_params (dict): Fitted parameters with keys 'center', 'sigma', 'amplitude'
    
    Returns:
        array: Residuals (data - fit)
    """
    if len(positions) == 0:
        return np.array([])
    
    fitted_values = gaussian_1d(positions, 
                               fit_params['amplitude'], 
                               fit_params['center'], 
                               fit_params['sigma'])
    
    return charges - fitted_values

def create_gauss_fit_plot(event_idx, data, output_dir="plots", show_event_info=True):
    """
    Create separate Gaussian fit plots for X and Y directions for a single event.
    Each direction gets its own figure with residuals on left, main plot on right.
    
    Args:
        event_idx (int): Event index to plot
        data (dict): Filtered data dictionary
        output_dir (str): Output directory for plots
        show_event_info (bool): Whether to show event information on plot
    
    Returns:
        str: Success message or error
    """
    try:
        # Extract row and column data
        (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get fit parameters for this event
        x_center = data['Fit2D_XCenter'][event_idx]
        x_sigma = data['Fit2D_XSigma'][event_idx]
        x_amplitude = data['Fit2D_XAmplitude'][event_idx]
        x_center_err = data['Fit2D_XCenterErr'][event_idx]
        x_sigma_err = data['Fit2D_XSigmaErr'][event_idx]
        x_chi2red = data['Fit2D_XChi2red'][event_idx]
        x_npoints = data['Fit2D_XNPoints'][event_idx]
        
        y_center = data['Fit2D_YCenter'][event_idx]
        y_sigma = data['Fit2D_YSigma'][event_idx]
        y_amplitude = data['Fit2D_YAmplitude'][event_idx]
        y_center_err = data['Fit2D_YCenterErr'][event_idx]
        y_sigma_err = data['Fit2D_YSigmaErr'][event_idx]
        y_chi2red = data['Fit2D_YChi2red'][event_idx]
        y_npoints = data['Fit2D_YNPoints'][event_idx]
        
        # True hit position for comparison
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ============================================
        # X-DIRECTION FIGURE
        # ============================================
        if len(row_pos) >= 3:
            fig_x = plt.figure(figsize=(16, 6))
            gs_x = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_x_res = fig_x.add_subplot(gs_x[0, 0])
            
            # Right panel: Main plot
            ax_x_main = fig_x.add_subplot(gs_x[0, 1])
            
            # Calculate fit parameters and residuals
            x_fit_params = {'center': x_center, 'sigma': x_sigma, 'amplitude': x_amplitude}
            residuals_x = calculate_residuals(row_pos, row_charges, x_fit_params)
            
            # Residuals plot (left panel)
            ax_x_res.errorbar(row_pos, residuals_x, fmt='bo', markersize=6, 
                             capsize=3, label='Fit residuals', alpha=0.8)
            ax_x_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_x_res.grid(True, alpha=0.3, linestyle=':')
            ax_x_res.set_xlabel(r'$X\mathrm{(mm)}$', fontsize=12)
            ax_x_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Gauss}(x_{\text{px}}) \mathrm{(C)}$', fontsize=12)
            ax_x_res.set_title('Residuals of Charge vs. Position X with Gaussian Fit', fontweight='bold')
            ax_x_res.legend()
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_x_main.errorbar(row_pos, row_charges, fmt='bo', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            x_fit_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_fit = gaussian_1d(x_fit_range, x_amplitude, x_center, x_sigma)
            
            # Plot fit curve
            fit_label = r'$y(x) = A \exp\!\Bigl(-\tfrac{(x - m)^2}{2\sigma^2}\Bigr)+ B$ fit'
            ax_x_main.plot(x_fit_range, y_fit, 'r-', linewidth=2.5, 
                          label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_x_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                             label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            ax_x_main.grid(True, alpha=0.3, linestyle=':')
            ax_x_main.set_xlabel(r'$X\mathrm{(mm)}$', fontsize=12)
            ax_x_main.set_ylabel(r'q\mathrm{(C))}', fontsize=12)
            ax_x_main.set_title('Charge vs. Row Position X with Gaussian Fit', fontweight='bold')
            ax_x_main.legend()
            
            # Add fit information text box
            fit_info = (f'μ = {x_center:.3f}±{x_center_err:.3f} mm\n'
                       f'σ = {x_sigma:.3f}±{x_sigma_err:.3f} mm\n' 
                       f'A = {x_amplitude:.2e} C\n'
                       f'χ²/ndf = {x_chi2red:.2f}\n'
                       f'N = {x_npoints}')
            ax_x_main.text(0.02, 0.98, fit_info, transform=ax_x_main.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Add overall title with event information
            if show_event_info:
                delta_x = x_center - true_x
                fig_x.suptitle(f'Event {event_idx}: X-Direction Gaussian Fit\n'
                              f'True X: {true_x:.3f} mm, Fitted X: {x_center:.3f} mm, '
                              f'ΔX: {delta_x:.3f} mm', 
                              fontsize=14, fontweight='bold')
            
            # Save X-direction plot
            filename_x = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_X.png')
            plt.savefig(filename_x, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # Y-DIRECTION FIGURE
        # ============================================
        if len(col_pos) >= 3:
            fig_y = plt.figure(figsize=(16, 6))
            gs_y = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_y_res = fig_y.add_subplot(gs_y[0, 0])
            
            # Right panel: Main plot
            ax_y_main = fig_y.add_subplot(gs_y[0, 1])
            
            # Calculate fit parameters and residuals
            y_fit_params = {'center': y_center, 'sigma': y_sigma, 'amplitude': y_amplitude}
            residuals_y = calculate_residuals(col_pos, col_charges, y_fit_params)
            
            # Residuals plot (left panel)
            ax_y_res.errorbar(col_pos, residuals_y, fmt='bo', markersize=6, 
                             capsize=3, label='Fit residuals', alpha=0.8)
            ax_y_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_y_res.grid(True, alpha=0.3, linestyle=':')
            ax_y_res.set_xlabel('Position Y [mm]', fontsize=12)
            ax_y_res.set_ylabel('Residuals (Data - Fit) [C]', fontsize=12)
            ax_y_res.set_title('Residuals of Charge vs. Position Y with Gaussian Fit', fontweight='bold')
            ax_y_res.legend()
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_y_main.errorbar(col_pos, col_charges, fmt='bo', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            y_fit_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_fit = gaussian_1d(y_fit_range, y_amplitude, y_center, y_sigma)
            
            # Plot fit curve
            fit_label = f'A·exp(-(y-μ)²/2σ²) fit'
            ax_y_main.plot(y_fit_range, y_fit, 'r-', linewidth=2.5, 
                          label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_y_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                             label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            ax_y_main.grid(True, alpha=0.3, linestyle=':')
            ax_y_main.set_xlabel('Position Y [mm]', fontsize=12)
            ax_y_main.set_ylabel('Charge [C]', fontsize=12)
            ax_y_main.set_title('Charge vs. Position Y with Gaussian Fit', fontweight='bold')
            ax_y_main.legend()
            
            # Add fit information text box
            fit_info = (f'μ = {y_center:.3f}±{y_center_err:.3f} mm\n'
                       f'σ = {y_sigma:.3f}±{y_sigma_err:.3f} mm\n' 
                       f'A = {y_amplitude:.2e} C\n'
                       f'χ²/ndf = {y_chi2red:.2f}\n'
                       f'N = {y_npoints}')
            ax_y_main.text(0.02, 0.98, fit_info, transform=ax_y_main.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Add overall title with event information
            if show_event_info:
                delta_y = y_center - true_y
                fig_y.suptitle(f'Event {event_idx}: Y-Direction Gaussian Fit\n'
                              f'True Y: {true_y:.3f} mm, Fitted Y: {y_center:.3f} mm, '
                              f'ΔY: {delta_y:.3f} mm', 
                              fontsize=14, fontweight='bold')
            
            # Save Y-direction plot
            filename_y = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_Y.png')
            plt.savefig(filename_y, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Generate success message
        created_plots = []
        if len(row_pos) >= 3:
            created_plots.append('X-direction')
        if len(col_pos) >= 3:
            created_plots.append('Y-direction')
        
        return f"Event {event_idx}: {' and '.join(created_plots)} plots saved"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating plot - {e}"

def create_summary_plots(data, output_dir="plots", max_events=50):
    """
    Create summary plots showing fit quality across multiple events.
    
    Args:
        data (dict): Filtered data dictionary
        output_dir (str): Output directory for plots
        max_events (int): Maximum number of events to plot individually
    
    Returns:
        str: Success message
    """
    try:
        n_events = len(data['TrueX'])
        n_to_plot = min(max_events, n_events)
        
        print(f"Creating summary plots for {n_to_plot} events...")
        
        # Arrays to collect data for summary statistics
        x_residuals_all = []
        y_residuals_all = []
        x_chi2_all = []
        y_chi2_all = []
        
        for i in range(n_to_plot):
            try:
                (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(i, data)
                
                # Collect X-direction data
                if len(row_pos) >= 3:
                    x_fit_params = {
                        'center': data['Fit2D_XCenter'][i],
                        'sigma': data['Fit2D_XSigma'][i],
                        'amplitude': data['Fit2D_XAmplitude'][i]
                    }
                    x_residuals = calculate_residuals(row_pos, row_charges, x_fit_params)
                    x_residuals_all.extend(x_residuals)
                    x_chi2_all.append(data['Fit2D_XChi2red'][i])
                
                # Collect Y-direction data
                if len(col_pos) >= 3:
                    y_fit_params = {
                        'center': data['Fit2D_YCenter'][i],
                        'sigma': data['Fit2D_YSigma'][i],
                        'amplitude': data['Fit2D_YAmplitude'][i]
                    }
                    y_residuals = calculate_residuals(col_pos, col_charges, y_fit_params)
                    y_residuals_all.extend(y_residuals)
                    y_chi2_all.append(data['Fit2D_YChi2red'][i])
                    
            except Exception as e:
                print(f"Warning: Error processing event {i}: {e}")
                continue
        
        # Create summary figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # X-direction residuals histogram
        if x_residuals_all:
            ax1.hist(x_residuals_all, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('X-Direction Fit Residuals Distribution', fontweight='bold')
            ax1.set_xlabel('Residual [C]')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(x_residuals_all)
            std_res = np.std(x_residuals_all)
            ax1.text(0.05, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}\nN: {len(x_residuals_all)}',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Y-direction residuals histogram
        if y_residuals_all:
            ax2.hist(y_residuals_all, bins=30, alpha=0.7, color='red', edgecolor='black')
            ax2.set_title('Y-Direction Fit Residuals Distribution', fontweight='bold')
            ax2.set_xlabel('Residual [C]')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(y_residuals_all)
            std_res = np.std(y_residuals_all)
            ax2.text(0.05, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}\nN: {len(y_residuals_all)}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Chi-squared distributions
        if x_chi2_all:
            ax3.hist(x_chi2_all, bins=20, alpha=0.7, color='blue', edgecolor='black', label='X-direction')
        if y_chi2_all:
            ax3.hist(y_chi2_all, bins=20, alpha=0.7, color='red', edgecolor='black', label='Y-direction')
        
        ax3.set_title('Reduced Chi-squared Distribution', fontweight='bold')
        ax3.set_xlabel('χ²/ndf')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Position accuracy plot
        true_x = data['TrueX'][:n_to_plot]
        true_y = data['TrueY'][:n_to_plot]
        fit_x = data['Fit2D_XCenter'][:n_to_plot]
        fit_y = data['Fit2D_YCenter'][:n_to_plot]
        
        distances = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
        ax4.hist(distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.set_title('Distance: Fitted Center to True Position', fontweight='bold')
        ax4.set_xlabel('Distance [mm]')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax4.text(0.05, 0.95, f'Mean: {mean_dist:.3f} mm\nStd: {std_dist:.3f} mm\nN: {len(distances)}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save summary plot
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, 'gauss_fit_summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Summary plots saved to {filename}"
        
    except Exception as e:
        return f"Error creating summary plots: {e}"

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Create Gaussian fit plots for charge sharing analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file with 2D Gaussian fit data")
    parser.add_argument("-o", "--output", default="gauss_plots", 
                       help="Output directory for plots")
    parser.add_argument("-n", "--num_events", type=int, default=10,
                       help="Number of individual events to plot")
    parser.add_argument("--summary_only", action="store_true",
                       help="Create only summary plots, skip individual events")
    parser.add_argument("--max_summary", type=int, default=100,
                       help="Maximum events to include in summary statistics")
    
    args = parser.parse_args()
    
    # Check if ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return 1
    
    # Load data
    data = load_successful_fits(args.root_file)
    if data is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Create summary plots
    print("\nCreating summary plots...")
    summary_result = create_summary_plots(data, args.output, args.max_summary)
    print(summary_result)
    
    if not args.summary_only:
        # Create individual event plots
        n_events = min(args.num_events, len(data['TrueX']))
        print(f"\nCreating individual plots for {n_events} events...")
        
        success_count = 0
        for i in range(n_events):
            result = create_gauss_fit_plot(i, data, args.output)
            if "Error" not in result:
                success_count += 1
            if i % 5 == 0 or "Error" in result:
                print(f"  {result}")
        
        print(f"\nSuccessfully created {success_count}/{n_events} individual plots")
    
    print(f"\nAll plots saved to: {args.output}/")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 