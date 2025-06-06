#!/usr/bin/env python3
"""
Post-processing plotting routine for Gaussian fit visualization of charge sharing in LGAD detectors.

This script creates plots for:
1. Gaussian curve estimation for central row (x-direction) with residuals
2. Gaussian curve estimation for central column (y-direction) with residuals
3. Gaussian curve estimation for main diagonal direction with residuals
4. Gaussian curve estimation for secondary diagonal direction with residuals
5. Overlay plots comparing different fitting approaches:
   - X-coordinate: Row fit + Main diagonal X fit + Secondary diagonal X fit
   - Y-coordinate: Column fit + Main diagonal Y fit + Secondary diagonal Y fit

The plots show the fitted Gaussian curves overlaid on the actual charge data points 
from the neighborhood grid, along with residual plots showing fit quality.
Individual directions get separate figures, and overlay plots compare all approaches.
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
                # Diagonal fit branches (4 separate fits: Main X, Main Y, Sec X, Sec Y)
                'FitDiag_MainXCenter', 'FitDiag_MainXSigma', 'FitDiag_MainXAmplitude',
                'FitDiag_MainXCenterErr', 'FitDiag_MainXSigmaErr', 'FitDiag_MainXAmplitudeErr',
                'FitDiag_MainXChi2red', 'FitDiag_MainXNPoints', 'FitDiag_MainXSuccessful',
                'FitDiag_MainYCenter', 'FitDiag_MainYSigma', 'FitDiag_MainYAmplitude',
                'FitDiag_MainYCenterErr', 'FitDiag_MainYSigmaErr', 'FitDiag_MainYAmplitudeErr',
                'FitDiag_MainYChi2red', 'FitDiag_MainYNPoints', 'FitDiag_MainYSuccessful',
                'FitDiag_SecXCenter', 'FitDiag_SecXSigma', 'FitDiag_SecXAmplitude',
                'FitDiag_SecXCenterErr', 'FitDiag_SecXSigmaErr', 'FitDiag_SecXAmplitudeErr',
                'FitDiag_SecXChi2red', 'FitDiag_SecXNPoints', 'FitDiag_SecXSuccessful',
                'FitDiag_SecYCenter', 'FitDiag_SecYSigma', 'FitDiag_SecYAmplitude',
                'FitDiag_SecYCenterErr', 'FitDiag_SecYSigmaErr', 'FitDiag_SecYAmplitudeErr',
                'FitDiag_SecYChi2red', 'FitDiag_SecYNPoints', 'FitDiag_SecYSuccessful',
                'FitDiag_Successful',  # Whether diagonal fitting was successful
                'NonPixel_GridNeighborhoodPixelI', 'NonPixel_GridNeighborhoodPixelJ',  # Pixel indices
                'NonPixel_GridNeighborhoodCharge',  # Charge values in Coulombs
                'NonPixel_GridNeighborhoodDistances'  # Distances from hit to pixels
            ]
            
            data = tree.arrays(branches, library="np")
            
            print(f"Total events loaded: {len(data['TrueX'])}")
            
            # Filter for successful non-pixel fits (both 2D and diagonal)
            is_non_pixel = ~data['IsPixelHit'] 
            is_fit_successful = data['Fit2D_Successful']
            is_diag_fit_successful = data['FitDiag_Successful']
            valid_mask = is_non_pixel & is_fit_successful & is_diag_fit_successful
            
            print(f"Non-pixel events: {np.sum(is_non_pixel)}")
            print(f"Successful 2D fits: {np.sum(is_fit_successful)}")
            print(f"Successful diagonal fits: {np.sum(is_diag_fit_successful)}")
            print(f"Valid events for plotting: {np.sum(valid_mask)}")
            
            if np.sum(valid_mask) == 0:
                print("Warning: No successful 2D and diagonal fits found for non-pixel events!")
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

def extract_diagonal_data(event_idx, data, neighborhood_radius=4):
    """
    Extract charge data for main and secondary diagonals from neighborhood grid data.
    This function matches the new 4-separate-fits approach: extracts X and Y coordinates
    separately for pixels on each diagonal line.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: ((main_x_pos, main_x_charges), (main_y_pos, main_y_charges), 
                (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges))
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
    pixel_spacing = 0.5  # mm - this should match detector parameters
    
    # Group pixels by diagonal lines
    # Main diagonal: pixels where (i - center_i) - (j - center_j) = constant
    # Secondary diagonal: pixels where (i - center_i) + (j - center_j) = constant
    
    main_diag_pixels = {}  # key: (i-j) difference, value: list of (x_pos, y_pos, charge)
    sec_diag_pixels = {}   # key: (i+j) sum, value: list of (x_pos, y_pos, charge)
    
    for idx, (i, j, charge) in enumerate(zip(pixel_i, pixel_j, charges)):
        if charge > 0:  # Only include pixels with charge
            # Calculate actual pixel position (relative to nearest pixel)
            x_pos = nearest_pixel_x + (i - center_i) * pixel_spacing
            y_pos = nearest_pixel_y + (j - center_j) * pixel_spacing
            
            # Group by diagonal lines
            main_key = (i - center_i) - (j - center_j)  # Main diagonal grouping
            sec_key = (i - center_i) + (j - center_j)   # Secondary diagonal grouping
            
            if main_key not in main_diag_pixels:
                main_diag_pixels[main_key] = []
            main_diag_pixels[main_key].append((x_pos, y_pos, charge))
            
            if sec_key not in sec_diag_pixels:
                sec_diag_pixels[sec_key] = []
            sec_diag_pixels[sec_key].append((x_pos, y_pos, charge))
    
    # Extract data for the main diagonal (typically the line passing through center)
    main_x_positions = []
    main_x_charges = []
    main_y_positions = []
    main_y_charges = []
    
    # Find the main diagonal line (key = 0, passing through center)
    if 0 in main_diag_pixels:
        for x_pos, y_pos, charge in main_diag_pixels[0]:
            main_x_positions.append(x_pos)
            main_x_charges.append(charge)
            main_y_positions.append(y_pos)
            main_y_charges.append(charge)
    
    # Extract data for the secondary diagonal (typically the line passing through center)  
    sec_x_positions = []
    sec_x_charges = []
    sec_y_positions = []
    sec_y_charges = []
    
    # Find the secondary diagonal line (key = 0, passing through center)
    if 0 in sec_diag_pixels:
        for x_pos, y_pos, charge in sec_diag_pixels[0]:
            sec_x_positions.append(x_pos)
            sec_x_charges.append(charge)
            sec_y_positions.append(y_pos)
            sec_y_charges.append(charge)
    
    # Sort by position for consistent plotting
    if main_x_positions:
        main_x_sorted = sorted(zip(main_x_positions, main_x_charges))
        main_x_positions, main_x_charges = zip(*main_x_sorted)
        
    if main_y_positions:
        main_y_sorted = sorted(zip(main_y_positions, main_y_charges))
        main_y_positions, main_y_charges = zip(*main_y_sorted)
    
    if sec_x_positions:
        sec_x_sorted = sorted(zip(sec_x_positions, sec_x_charges))
        sec_x_positions, sec_x_charges = zip(*sec_x_sorted)
        
    if sec_y_positions:
        sec_y_sorted = sorted(zip(sec_y_positions, sec_y_charges))
        sec_y_positions, sec_y_charges = zip(*sec_y_sorted)
    
    return ((np.array(main_x_positions), np.array(main_x_charges)), 
            (np.array(main_y_positions), np.array(main_y_charges)),
            (np.array(sec_x_positions), np.array(sec_x_charges)), 
            (np.array(sec_y_positions), np.array(sec_y_charges)))

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

def create_gauss_fit_plot(event_idx, data, output_dir="plots", show_event_info=False):
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
        # Extract row, column, and diagonal data
        (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3 and len(main_x_pos) < 3 and len(main_y_pos) < 3 and len(sec_x_pos) < 3 and len(sec_y_pos) < 3:
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
        
        # Get diagonal fit parameters (4 separate fits: Main X, Main Y, Sec X, Sec Y)
        # Main diagonal X fit (X vs Charge for pixels on main diagonal)
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx]
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_center_err = data['FitDiag_MainXCenterErr'][event_idx]
        main_diag_x_sigma_err = data['FitDiag_MainXSigmaErr'][event_idx]
        main_diag_x_chi2red = data['FitDiag_MainXChi2red'][event_idx]
        main_diag_x_npoints = data['FitDiag_MainXNPoints'][event_idx]
        main_diag_x_successful = data['FitDiag_MainXSuccessful'][event_idx]
        
        # Main diagonal Y fit (Y vs Charge for pixels on main diagonal)
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx]
        main_diag_y_center_err = data['FitDiag_MainYCenterErr'][event_idx]
        main_diag_y_sigma_err = data['FitDiag_MainYSigmaErr'][event_idx]
        main_diag_y_chi2red = data['FitDiag_MainYChi2red'][event_idx]
        main_diag_y_npoints = data['FitDiag_MainYNPoints'][event_idx]
        main_diag_y_successful = data['FitDiag_MainYSuccessful'][event_idx]
        
        # Secondary diagonal X fit (X vs Charge for pixels on secondary diagonal)
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_center_err = data['FitDiag_SecXCenterErr'][event_idx]
        sec_diag_x_sigma_err = data['FitDiag_SecXSigmaErr'][event_idx]
        sec_diag_x_chi2red = data['FitDiag_SecXChi2red'][event_idx]
        sec_diag_x_npoints = data['FitDiag_SecXNPoints'][event_idx]
        sec_diag_x_successful = data['FitDiag_SecXSuccessful'][event_idx]
        
        # Secondary diagonal Y fit (Y vs Charge for pixels on secondary diagonal)
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_center_err = data['FitDiag_SecYCenterErr'][event_idx]
        sec_diag_y_sigma_err = data['FitDiag_SecYSigmaErr'][event_idx]
        sec_diag_y_chi2red = data['FitDiag_SecYChi2red'][event_idx]
        sec_diag_y_npoints = data['FitDiag_SecYNPoints'][event_idx]
        sec_diag_y_successful = data['FitDiag_SecYSuccessful'][event_idx]
        
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
            ax_x_res.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(x_{\text{px}})\,(\mathrm{C})$', fontsize=12)
            ax_x_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Row)')
            ax_x_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_x_main.errorbar(row_pos, row_charges, fmt='bo', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            x_fit_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_fit = gaussian_1d(x_fit_range, x_amplitude, x_center, x_sigma)
            
            # Plot fit curve
            fit_label = r'$y(x) = A \exp\left(-\frac{(x - m)^2}{2\sigma^2}\right)+ B$'
            ax_x_main.plot(x_fit_range, y_fit, 'r-', linewidth=2.5, 
                          label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_x_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                             label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_x_main.axvline(x_center, color='orange', linestyle=':', linewidth=2, 
                             label=f'Fitted X = {x_center:.3f} mm', alpha=0.8)
            
            ax_x_main.grid(True, alpha=0.3, linestyle=':')
            ax_x_main.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_x_main.set_title('Pixel Charge vs. Central Coordinate (Row)')
            ax_x_main.legend(loc='upper left')
            
            '''
            # Add fit information text box
            fit_info = (f'μ = {x_center:.3f}±{x_center_err:.3f} mm\n'
                       f'σ = {x_sigma:.3f}±{x_sigma_err:.3f} mm\n' 
                       f'A = {x_amplitude:.2e} C\n'
                       f'χ²/ndf = {x_chi2red:.2f}\n'
                       f'N = {x_npoints}')
            ax_x_main.text(0.02, 0.98, fit_info, transform=ax_x_main.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            '''
            # Add overall title with event information
            if show_event_info:
                delta_x = x_center - true_x
                fig_x.suptitle(f'Event {event_idx}: X-Direction Gaussian Fit\n'
                              f'True X: {true_x:.3f} mm, Fitted X: {x_center:.3f} mm, '
                              f'ΔX: {delta_x:.3f} mm', 
                              fontsize=14)
            
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
            ax_y_res.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(y_{\text{px}})\,(\mathrm{C})$', fontsize=12)
            ax_y_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Column)')
            ax_y_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_y_main.errorbar(col_pos, col_charges, fmt='bo', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            y_fit_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_fit = gaussian_1d(y_fit_range, y_amplitude, y_center, y_sigma)
            
            # Plot fit curve
            fit_label = r'$y(x) = A \exp\left(-\frac{(x - m)^2}{2\sigma^2}\right)+ B$'
            ax_y_main.plot(y_fit_range, y_fit, 'r-', linewidth=2.5, 
                          label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_y_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                             label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_y_main.axvline(y_center, color='orange', linestyle=':', linewidth=2, 
                             label=f'Fitted Y = {y_center:.3f} mm', alpha=0.8)
            
            ax_y_main.grid(True, alpha=0.3, linestyle=':')
            ax_y_main.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_y_main.set_title('Pixel Charge vs. Central Coordinate (Column)')
            ax_y_main.legend(loc='upper left')
            '''
            # Add fit information text box
            fit_info = (f'μ = {y_center:.3f}±{y_center_err:.3f} mm\n'
                       f'σ = {y_sigma:.3f}±{y_sigma_err:.3f} mm\n' 
                       f'A = {y_amplitude:.2e} C\n'
                       f'χ²/ndf = {y_chi2red:.2f}\n'
                       f'N = {y_npoints}')
            ax_y_main.text(0.02, 0.98, fit_info, transform=ax_y_main.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            '''
            # Add overall title with event information
            if show_event_info:
                delta_y = y_center - true_y
                fig_y.suptitle(f'Event {event_idx}: Y-Direction Gaussian Fit\n'
                              f'True Y: {true_y:.3f} mm, Fitted Y: {y_center:.3f} mm, '
                              f'ΔY: {delta_y:.3f} mm', 
                              fontsize=14)
            
            # Save Y-direction plot
            filename_y = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_Y.png')
            plt.savefig(filename_y, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # MAIN DIAGONAL X FIGURE (X vs Charge for main diagonal pixels)
        # ============================================
        if len(main_x_pos) >= 3 and main_diag_x_successful:
            fig_main_x = plt.figure(figsize=(16, 6))
            gs_main_x = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_main_x_res = fig_main_x.add_subplot(gs_main_x[0, 0])
            
            # Right panel: Main plot
            ax_main_x_main = fig_main_x.add_subplot(gs_main_x[0, 1])
            
            # Calculate fit parameters and residuals
            main_x_fit_params = {'center': main_diag_x_center, 'sigma': main_diag_x_sigma, 'amplitude': main_diag_x_amplitude}
            residuals_main_x = calculate_residuals(main_x_pos, main_x_charges, main_x_fit_params)
            
            # Residuals plot (left panel)
            ax_main_x_res.errorbar(main_x_pos, residuals_main_x, fmt='bo', markersize=6, 
                                  capsize=3, label='Fit residuals', alpha=0.8)
            ax_main_x_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_main_x_res.grid(True, alpha=0.3, linestyle=':')
            ax_main_x_res.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_main_x_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(x_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_main_x_res.set_title('Residuals: Main Diagonal X vs Charge')
            ax_main_x_res.legend(loc='upper right')
            
            # Main plot (right panel)
            ax_main_x_main.errorbar(main_x_pos, main_x_charges, fmt='bo', markersize=8, 
                                   capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            main_x_fit_range = np.linspace(main_x_pos.min() - 0.1, main_x_pos.max() + 0.1, 200)
            main_x_y_fit = gaussian_1d(main_x_fit_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma)
            
            # Plot fit curve
            fit_label = r'$q(x) = A \exp\left(-\frac{(x - m)^2}{2\sigma^2}\right)+ B$'
            ax_main_x_main.plot(main_x_fit_range, main_x_y_fit, 'r-', linewidth=2.5, 
                               label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_main_x_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                                  label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_main_x_main.axvline(main_diag_x_center, color='orange', linestyle=':', linewidth=2, 
                                  label=f'Fitted X = {main_diag_x_center:.3f} mm', alpha=0.8)
            
            ax_main_x_main.grid(True, alpha=0.3, linestyle=':')
            ax_main_x_main.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_main_x_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_main_x_main.set_title('Main Diagonal: X Position vs Charge')
            ax_main_x_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_main_x = main_diag_x_center - true_x
                fig_main_x.suptitle(f'Event {event_idx}: Main Diagonal X Fit\n'
                                   f'True: {true_x:.3f} mm, Fitted: {main_diag_x_center:.3f} mm, '
                                   f'ΔX: {delta_main_x:.3f} mm', 
                                   fontsize=14)
            
            # Save main diagonal X plot
            filename_main_x = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_MainDiagX.png')
            plt.savefig(filename_main_x, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        # ============================================
        # MAIN DIAGONAL Y FIGURE (Y vs Charge for main diagonal pixels)
        # ============================================
        if len(main_y_pos) >= 3 and main_diag_y_successful:
            fig_main_y = plt.figure(figsize=(16, 6))
            gs_main_y = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_main_y_res = fig_main_y.add_subplot(gs_main_y[0, 0])
            
            # Right panel: Main plot
            ax_main_y_main = fig_main_y.add_subplot(gs_main_y[0, 1])
            
            # Calculate fit parameters and residuals
            main_y_fit_params = {'center': main_diag_y_center, 'sigma': main_diag_y_sigma, 'amplitude': main_diag_y_amplitude}
            residuals_main_y = calculate_residuals(main_y_pos, main_y_charges, main_y_fit_params)
            
            # Residuals plot (left panel)
            ax_main_y_res.errorbar(main_y_pos, residuals_main_y, fmt='bo', markersize=6, 
                                  capsize=3, label='Fit residuals', alpha=0.8)
            ax_main_y_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_main_y_res.grid(True, alpha=0.3, linestyle=':')
            ax_main_y_res.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_main_y_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(y_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_main_y_res.set_title('Residuals: Main Diagonal Y vs Charge')
            ax_main_y_res.legend(loc='upper right')
            
            # Main plot (right panel)
            ax_main_y_main.errorbar(main_y_pos, main_y_charges, fmt='bo', markersize=8, 
                                   capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            main_y_fit_range = np.linspace(main_y_pos.min() - 0.1, main_y_pos.max() + 0.1, 200)
            main_y_y_fit = gaussian_1d(main_y_fit_range, main_diag_y_amplitude, main_diag_y_center, main_diag_y_sigma)
            
            # Plot fit curve
            fit_label = r'$q(y) = A \exp\left(-\frac{(y - m)^2}{2\sigma^2}\right)+ B$'
            ax_main_y_main.plot(main_y_fit_range, main_y_y_fit, 'r-', linewidth=2.5, 
                               label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_main_y_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                                  label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_main_y_main.axvline(main_diag_y_center, color='orange', linestyle=':', linewidth=2, 
                                  label=f'Fitted Y = {main_diag_y_center:.3f} mm', alpha=0.8)
            
            ax_main_y_main.grid(True, alpha=0.3, linestyle=':')
            ax_main_y_main.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_main_y_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_main_y_main.set_title('Main Diagonal: Y Position vs Charge')
            ax_main_y_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_main_y = main_diag_y_center - true_y
                fig_main_y.suptitle(f'Event {event_idx}: Main Diagonal Y Fit\n'
                                   f'True: {true_y:.3f} mm, Fitted: {main_diag_y_center:.3f} mm, '
                                   f'ΔY: {delta_main_y:.3f} mm', 
                                   fontsize=14)
            
            # Save main diagonal Y plot
            filename_main_y = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_MainDiagY.png')
            plt.savefig(filename_main_y, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # SECONDARY DIAGONAL X FIGURE (X vs Charge for secondary diagonal pixels)
        # ============================================
        if len(sec_x_pos) >= 3 and sec_diag_x_successful:
            fig_sec_x = plt.figure(figsize=(16, 6))
            gs_sec_x = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_sec_x_res = fig_sec_x.add_subplot(gs_sec_x[0, 0])
            
            # Right panel: Main plot
            ax_sec_x_main = fig_sec_x.add_subplot(gs_sec_x[0, 1])
            
            # Calculate fit parameters and residuals
            sec_x_fit_params = {'center': sec_diag_x_center, 'sigma': sec_diag_x_sigma, 'amplitude': sec_diag_x_amplitude}
            residuals_sec_x = calculate_residuals(sec_x_pos, sec_x_charges, sec_x_fit_params)
            
            # Residuals plot (left panel)
            ax_sec_x_res.errorbar(sec_x_pos, residuals_sec_x, fmt='bo', markersize=6, 
                                 capsize=3, label='Fit residuals', alpha=0.8)
            ax_sec_x_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_sec_x_res.grid(True, alpha=0.3, linestyle=':')
            ax_sec_x_res.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_sec_x_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(x_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_sec_x_res.set_title('Residuals: Secondary Diagonal X vs Charge')
            ax_sec_x_res.legend(loc='upper right')
            
            # Main plot (right panel)
            ax_sec_x_main.errorbar(sec_x_pos, sec_x_charges, fmt='bo', markersize=8, 
                                  capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            sec_x_fit_range = np.linspace(sec_x_pos.min() - 0.1, sec_x_pos.max() + 0.1, 200)
            sec_x_y_fit = gaussian_1d(sec_x_fit_range, sec_diag_x_amplitude, sec_diag_x_center, sec_diag_x_sigma)
            
            # Plot fit curve
            fit_label = r'$q(x) = A \exp\left(-\frac{(x - m)^2}{2\sigma^2}\right)+ B$'
            ax_sec_x_main.plot(sec_x_fit_range, sec_x_y_fit, 'r-', linewidth=2.5, 
                              label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_sec_x_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                                 label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_sec_x_main.axvline(sec_diag_x_center, color='orange', linestyle=':', linewidth=2, 
                                 label=f'Fitted X = {sec_diag_x_center:.3f} mm', alpha=0.8)
            
            ax_sec_x_main.grid(True, alpha=0.3, linestyle=':')
            ax_sec_x_main.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_sec_x_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_sec_x_main.set_title('Secondary Diagonal: X Position vs Charge')
            ax_sec_x_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_sec_x = sec_diag_x_center - true_x
                fig_sec_x.suptitle(f'Event {event_idx}: Secondary Diagonal X Fit\n'
                                  f'True: {true_x:.3f} mm, Fitted: {sec_diag_x_center:.3f} mm, '
                                  f'ΔX: {delta_sec_x:.3f} mm', 
                                  fontsize=14)
            
            # Save secondary diagonal X plot
            filename_sec_x = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_SecDiagX.png')
            plt.savefig(filename_sec_x, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        # ============================================
        # SECONDARY DIAGONAL Y FIGURE (Y vs Charge for secondary diagonal pixels)
        # ============================================
        if len(sec_y_pos) >= 3 and sec_diag_y_successful:
            fig_sec_y = plt.figure(figsize=(16, 6))
            gs_sec_y = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_sec_y_res = fig_sec_y.add_subplot(gs_sec_y[0, 0])
            
            # Right panel: Main plot
            ax_sec_y_main = fig_sec_y.add_subplot(gs_sec_y[0, 1])
            
            # Calculate fit parameters and residuals
            sec_y_fit_params = {'center': sec_diag_y_center, 'sigma': sec_diag_y_sigma, 'amplitude': sec_diag_y_amplitude}
            residuals_sec_y = calculate_residuals(sec_y_pos, sec_y_charges, sec_y_fit_params)
            
            # Residuals plot (left panel)
            ax_sec_y_res.errorbar(sec_y_pos, residuals_sec_y, fmt='bo', markersize=6, 
                                 capsize=3, label='Fit residuals', alpha=0.8)
            ax_sec_y_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_sec_y_res.grid(True, alpha=0.3, linestyle=':')
            ax_sec_y_res.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_sec_y_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(y_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_sec_y_res.set_title('Residuals: Secondary Diagonal Y vs Charge')
            ax_sec_y_res.legend(loc='upper right')
            
            # Main plot (right panel)
            ax_sec_y_main.errorbar(sec_y_pos, sec_y_charges, fmt='bo', markersize=8, 
                                  capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            sec_y_fit_range = np.linspace(sec_y_pos.min() - 0.1, sec_y_pos.max() + 0.1, 200)
            sec_y_y_fit = gaussian_1d(sec_y_fit_range, sec_diag_y_amplitude, sec_diag_y_center, sec_diag_y_sigma)
            
            # Plot fit curve
            fit_label = r'$q(y) = A \exp\left(-\frac{(y - m)^2}{2\sigma^2}\right)+ B$'
            ax_sec_y_main.plot(sec_y_fit_range, sec_y_y_fit, 'r-', linewidth=2.5, 
                              label=fit_label, alpha=0.9)
            
            # Mark true position
            ax_sec_y_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                                 label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_sec_y_main.axvline(sec_diag_y_center, color='orange', linestyle=':', linewidth=2, 
                                 label=f'Fitted Y = {sec_diag_y_center:.3f} mm', alpha=0.8)
            
            ax_sec_y_main.grid(True, alpha=0.3, linestyle=':')
            ax_sec_y_main.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_sec_y_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_sec_y_main.set_title('Secondary Diagonal: Y Position vs Charge')
            ax_sec_y_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_sec_y = sec_diag_y_center - true_y
                fig_sec_y.suptitle(f'Event {event_idx}: Secondary Diagonal Y Fit\n'
                                  f'True: {true_y:.3f} mm, Fitted: {sec_diag_y_center:.3f} mm, '
                                  f'ΔY: {delta_sec_y:.3f} mm', 
                                  fontsize=14)
            
            # Save secondary diagonal Y plot
            filename_sec_y = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_SecDiagY.png')
            plt.savefig(filename_sec_y, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Generate success message
        created_plots = []
        if len(row_pos) >= 3:
            created_plots.append('X-direction')
        if len(col_pos) >= 3:
            created_plots.append('Y-direction')
        if len(main_x_pos) >= 3 and main_diag_x_successful:
            created_plots.append('Main-diagonal-X')
        if len(main_y_pos) >= 3 and main_diag_y_successful:
            created_plots.append('Main-diagonal-Y')
        if len(sec_x_pos) >= 3 and sec_diag_x_successful:
            created_plots.append('Secondary-diagonal-X')
        if len(sec_y_pos) >= 3 and sec_diag_y_successful:
            created_plots.append('Secondary-diagonal-Y')
        
        return f"Event {event_idx}: {' and '.join(created_plots)} plots saved"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating plot - {e}"

def create_overlay_fit_plots(event_idx, data, output_dir="plots", show_event_info=False):
    """
    Create overlay plots comparing all fitting approaches for X and Y coordinates.
    
    For X-coordinate: Row fit + Main diagonal X fit + Secondary diagonal X fit
    For Y-coordinate: Column fit + Main diagonal Y fit + Secondary diagonal Y fit
    
    Args:
        event_idx (int): Event index to plot
        data (dict): Filtered data dictionary
        output_dir (str): Output directory for plots
        show_event_info (bool): Whether to show event information on plot
    
    Returns:
        str: Success message or error
    """
    try:
        # Extract row, column, and diagonal data
        (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        # Get fit parameters for this event
        x_center = data['Fit2D_XCenter'][event_idx]
        x_sigma = data['Fit2D_XSigma'][event_idx]
        x_amplitude = data['Fit2D_XAmplitude'][event_idx]
        
        y_center = data['Fit2D_YCenter'][event_idx]
        y_sigma = data['Fit2D_YSigma'][event_idx]
        y_amplitude = data['Fit2D_YAmplitude'][event_idx]
        
        # Get diagonal fit parameters
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx]
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_successful = data['FitDiag_MainXSuccessful'][event_idx]
        
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx]
        main_diag_y_successful = data['FitDiag_MainYSuccessful'][event_idx]
        
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_successful = data['FitDiag_SecXSuccessful'][event_idx]
        
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_successful = data['FitDiag_SecYSuccessful'][event_idx]
        
        # True hit position for comparison
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ============================================
        # X-COORDINATE OVERLAY FIGURE
        # ============================================
        if len(row_pos) >= 3 or (len(main_x_pos) >= 3 and main_diag_x_successful) or (len(sec_x_pos) >= 3 and sec_diag_x_successful):
            fig_x_overlay = plt.figure(figsize=(14, 8))
            ax_x_overlay = fig_x_overlay.add_subplot(111)
            
            # Determine overall X range for plotting
            all_x_positions = []
            if len(row_pos) >= 3:
                all_x_positions.extend(row_pos)
            if len(main_x_pos) >= 3 and main_diag_x_successful:
                all_x_positions.extend(main_x_pos)
            if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                all_x_positions.extend(sec_x_pos)
            
            if all_x_positions:
                x_min, x_max = min(all_x_positions), max(all_x_positions)
                x_range = np.linspace(x_min - 0.2, x_max + 0.2, 300)
                
                # Plot Row data and fit (central row)
                if len(row_pos) >= 3:
                    ax_x_overlay.errorbar(row_pos, row_charges, fmt='o', markersize=8, 
                                        capsize=4, label='Row data', alpha=0.8, linewidth=1.5, 
                                        color='blue', markeredgecolor='darkblue')
                    
                    # Row fit curve
                    row_fit = gaussian_1d(x_range, x_amplitude, x_center, x_sigma)
                    ax_x_overlay.plot(x_range, row_fit, '-', linewidth=3, 
                                    label=f'Row fit (μ={x_center:.3f} mm)', 
                                    color='blue', alpha=0.9)
                
                # Plot Main diagonal X data and fit
                if len(main_x_pos) >= 3 and main_diag_x_successful:
                    ax_x_overlay.errorbar(main_x_pos, main_x_charges, fmt='s', markersize=8, 
                                        capsize=4, label='Main diag data', alpha=0.8, linewidth=1.5,
                                        color='red', markeredgecolor='darkred')
                    
                    # Main diagonal X fit curve
                    main_diag_x_fit = gaussian_1d(x_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma)
                    ax_x_overlay.plot(x_range, main_diag_x_fit, '--', linewidth=3, 
                                    label=f'Main diag fit (μ={main_diag_x_center:.3f} mm)', 
                                    color='red', alpha=0.9)
                
                # Plot Secondary diagonal X data and fit
                if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                    ax_x_overlay.errorbar(sec_x_pos, sec_x_charges, fmt='^', markersize=8, 
                                        capsize=4, label='Sec diag data', alpha=0.8, linewidth=1.5,
                                        color='purple', markeredgecolor='indigo')
                    
                    # Secondary diagonal X fit curve
                    sec_diag_x_fit = gaussian_1d(x_range, sec_diag_x_amplitude, sec_diag_x_center, sec_diag_x_sigma)
                    ax_x_overlay.plot(x_range, sec_diag_x_fit, ':', linewidth=3, 
                                    label=f'Sec diag fit (μ={sec_diag_x_center:.3f} mm)', 
                                    color='purple', alpha=0.9)
                
                # Mark fitted centers for each approach
                if len(row_pos) >= 3:
                    ax_x_overlay.axvline(x_center, color='blue', linestyle=':', linewidth=2, 
                                       label=f'Row fit center = {x_center:.3f} mm', alpha=0.7)
                
                if len(main_x_pos) >= 3 and main_diag_x_successful:
                    ax_x_overlay.axvline(main_diag_x_center, color='red', linestyle=':', linewidth=2, 
                                       label=f'Main diag center = {main_diag_x_center:.3f} mm', alpha=0.7)
                
                if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                    ax_x_overlay.axvline(sec_diag_x_center, color='purple', linestyle=':', linewidth=2, 
                                       label=f'Sec diag center = {sec_diag_x_center:.3f} mm', alpha=0.7)
                
                # Mark true position
                ax_x_overlay.axvline(true_x, color='green', linestyle='-', linewidth=3, 
                                   label=f'True X = {true_x:.3f} mm', alpha=0.8)
                
                ax_x_overlay.grid(True, alpha=0.3, linestyle=':')
                ax_x_overlay.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=14)
                ax_x_overlay.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=14)
                ax_x_overlay.set_title('X-Coordinate: Comparison of Row, Main Diagonal, and Secondary Diagonal Fits', fontsize=16)
                ax_x_overlay.legend(loc='best', fontsize=12)
                
                # Add overall title with event information
                if show_event_info:
                    fig_x_overlay.suptitle(f'Event {event_idx}: X-Coordinate Fit Comparison\n'
                                         f'True X: {true_x:.3f} mm', 
                                         fontsize=16)
                
                # Save X-coordinate overlay plot
                filename_x_overlay = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_X_overlay.png')
                plt.savefig(filename_x_overlay, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
        
        # ============================================
        # Y-COORDINATE OVERLAY FIGURE
        # ============================================
        if len(col_pos) >= 3 or (len(main_y_pos) >= 3 and main_diag_y_successful) or (len(sec_y_pos) >= 3 and sec_diag_y_successful):
            fig_y_overlay = plt.figure(figsize=(14, 8))
            ax_y_overlay = fig_y_overlay.add_subplot(111)
            
            # Determine overall Y range for plotting
            all_y_positions = []
            if len(col_pos) >= 3:
                all_y_positions.extend(col_pos)
            if len(main_y_pos) >= 3 and main_diag_y_successful:
                all_y_positions.extend(main_y_pos)
            if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                all_y_positions.extend(sec_y_pos)
            
            if all_y_positions:
                y_min, y_max = min(all_y_positions), max(all_y_positions)
                y_range = np.linspace(y_min - 0.2, y_max + 0.2, 300)
                
                # Plot Column data and fit (central column)
                if len(col_pos) >= 3:
                    ax_y_overlay.errorbar(col_pos, col_charges, fmt='o', markersize=8, 
                                        capsize=4, label='Column data', alpha=0.8, linewidth=1.5, 
                                        color='blue', markeredgecolor='darkblue')
                    
                    # Column fit curve
                    col_fit = gaussian_1d(y_range, y_amplitude, y_center, y_sigma)
                    ax_y_overlay.plot(y_range, col_fit, '-', linewidth=3, 
                                    label=f'Column fit (μ={y_center:.3f} mm)', 
                                    color='blue', alpha=0.9)
                
                # Plot Main diagonal Y data and fit
                if len(main_y_pos) >= 3 and main_diag_y_successful:
                    ax_y_overlay.errorbar(main_y_pos, main_y_charges, fmt='s', markersize=8, 
                                        capsize=4, label='Main diag data', alpha=0.8, linewidth=1.5,
                                        color='red', markeredgecolor='darkred')
                    
                    # Main diagonal Y fit curve
                    main_diag_y_fit = gaussian_1d(y_range, main_diag_y_amplitude, main_diag_y_center, main_diag_y_sigma)
                    ax_y_overlay.plot(y_range, main_diag_y_fit, '--', linewidth=3, 
                                    label=f'Main diag fit (μ={main_diag_y_center:.3f} mm)', 
                                    color='red', alpha=0.9)
                
                # Plot Secondary diagonal Y data and fit
                if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                    ax_y_overlay.errorbar(sec_y_pos, sec_y_charges, fmt='^', markersize=8, 
                                        capsize=4, label='Sec diag data', alpha=0.8, linewidth=1.5,
                                        color='purple', markeredgecolor='indigo')
                    
                    # Secondary diagonal Y fit curve
                    sec_diag_y_fit = gaussian_1d(y_range, sec_diag_y_amplitude, sec_diag_y_center, sec_diag_y_sigma)
                    ax_y_overlay.plot(y_range, sec_diag_y_fit, ':', linewidth=3, 
                                    label=f'Sec diag fit (μ={sec_diag_y_center:.3f} mm)', 
                                    color='purple', alpha=0.9)
                
                # Mark fitted centers for each approach
                if len(col_pos) >= 3:
                    ax_y_overlay.axvline(y_center, color='blue', linestyle=':', linewidth=2, 
                                       label=f'Column fit center = {y_center:.3f} mm', alpha=0.7)
                
                if len(main_y_pos) >= 3 and main_diag_y_successful:
                    ax_y_overlay.axvline(main_diag_y_center, color='red', linestyle=':', linewidth=2, 
                                       label=f'Main diag center = {main_diag_y_center:.3f} mm', alpha=0.7)
                
                if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                    ax_y_overlay.axvline(sec_diag_y_center, color='purple', linestyle=':', linewidth=2, 
                                       label=f'Sec diag center = {sec_diag_y_center:.3f} mm', alpha=0.7)
                
                # Mark true position
                ax_y_overlay.axvline(true_y, color='green', linestyle='-', linewidth=3, 
                                   label=f'True Y = {true_y:.3f} mm', alpha=0.8)
                
                ax_y_overlay.grid(True, alpha=0.3, linestyle=':')
                ax_y_overlay.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=14)
                ax_y_overlay.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=14)
                ax_y_overlay.set_title('Y-Coordinate: Comparison of Column, Main Diagonal, and Secondary Diagonal Fits', fontsize=16)
                ax_y_overlay.legend(loc='best', fontsize=12)
                
                # Add overall title with event information
                if show_event_info:
                    fig_y_overlay.suptitle(f'Event {event_idx}: Y-Coordinate Fit Comparison\n'
                                         f'True Y: {true_y:.3f} mm', 
                                         fontsize=16)
                
                # Save Y-coordinate overlay plot
                filename_y_overlay = os.path.join(output_dir, f'gauss_fit_event_{event_idx:04d}_Y_overlay.png')
                plt.savefig(filename_y_overlay, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
        
        # Generate success message for overlay plots
        created_overlays = []
        if len(row_pos) >= 3 or (len(main_x_pos) >= 3 and main_diag_x_successful) or (len(sec_x_pos) >= 3 and sec_diag_x_successful):
            created_overlays.append('X-overlay')
        if len(col_pos) >= 3 or (len(main_y_pos) >= 3 and main_diag_y_successful) or (len(sec_y_pos) >= 3 and sec_diag_y_successful):
            created_overlays.append('Y-overlay')
        
        return f"Event {event_idx}: {' and '.join(created_overlays)} plots saved"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating overlay plot - {e}"

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
        main_diag_residuals_all = []
        sec_diag_residuals_all = []
        x_chi2_all = []
        y_chi2_all = []
        main_diag_chi2_all = []
        sec_diag_chi2_all = []
        
        for i in range(n_to_plot):
            try:
                (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(i, data)
                (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(i, data)
                
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
                
                # Collect Main diagonal X data
                if len(main_x_pos) >= 3 and data['FitDiag_MainXSuccessful'][i]:
                    main_x_fit_params = {
                        'center': data['FitDiag_MainXCenter'][i],
                        'sigma': data['FitDiag_MainXSigma'][i],
                        'amplitude': data['FitDiag_MainXAmplitude'][i]
                    }
                    main_x_residuals = calculate_residuals(main_x_pos, main_x_charges, main_x_fit_params)
                    main_diag_residuals_all.extend(main_x_residuals)
                    main_diag_chi2_all.append(data['FitDiag_MainXChi2red'][i])
                
                # Collect Main diagonal Y data
                if len(main_y_pos) >= 3 and data['FitDiag_MainYSuccessful'][i]:
                    main_y_fit_params = {
                        'center': data['FitDiag_MainYCenter'][i],
                        'sigma': data['FitDiag_MainYSigma'][i],
                        'amplitude': data['FitDiag_MainYAmplitude'][i]
                    }
                    main_y_residuals = calculate_residuals(main_y_pos, main_y_charges, main_y_fit_params)
                    main_diag_residuals_all.extend(main_y_residuals)
                    main_diag_chi2_all.append(data['FitDiag_MainYChi2red'][i])
                
                # Collect Secondary diagonal X data
                if len(sec_x_pos) >= 3 and data['FitDiag_SecXSuccessful'][i]:
                    sec_x_fit_params = {
                        'center': data['FitDiag_SecXCenter'][i],
                        'sigma': data['FitDiag_SecXSigma'][i],
                        'amplitude': data['FitDiag_SecXAmplitude'][i]
                    }
                    sec_x_residuals = calculate_residuals(sec_x_pos, sec_x_charges, sec_x_fit_params)
                    sec_diag_residuals_all.extend(sec_x_residuals)
                    sec_diag_chi2_all.append(data['FitDiag_SecXChi2red'][i])
                
                # Collect Secondary diagonal Y data
                if len(sec_y_pos) >= 3 and data['FitDiag_SecYSuccessful'][i]:
                    sec_y_fit_params = {
                        'center': data['FitDiag_SecYCenter'][i],
                        'sigma': data['FitDiag_SecYSigma'][i],
                        'amplitude': data['FitDiag_SecYAmplitude'][i]
                    }
                    sec_y_residuals = calculate_residuals(sec_y_pos, sec_y_charges, sec_y_fit_params)
                    sec_diag_residuals_all.extend(sec_y_residuals)
                    sec_diag_chi2_all.append(data['FitDiag_SecYChi2red'][i])
                    
            except Exception as e:
                print(f"Warning: Error processing event {i}: {e}")
                continue
        
        # Create summary figure
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        
        # X-direction residuals histogram
        if x_residuals_all:
            ax1.hist(x_residuals_all, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('X-Direction Fit Residuals Distribution')
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
            ax2.set_title('Y-Direction Fit Residuals Distribution')
            ax2.set_xlabel('Residual [C]')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(y_residuals_all)
            std_res = np.std(y_residuals_all)
            ax2.text(0.05, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}\nN: {len(y_residuals_all)}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Main diagonal residuals histogram
        if main_diag_residuals_all:
            ax3.hist(main_diag_residuals_all, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax3.set_title('Main Diagonal Fit Residuals Distribution')
            ax3.set_xlabel('Residual [C]')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(main_diag_residuals_all)
            std_res = np.std(main_diag_residuals_all)
            ax3.text(0.05, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}\nN: {len(main_diag_residuals_all)}',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
        
        # Secondary diagonal residuals histogram
        if sec_diag_residuals_all:
            ax4.hist(sec_diag_residuals_all, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Secondary Diagonal Fit Residuals Distribution')
            ax4.set_xlabel('Residual [C]')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            mean_res = np.mean(sec_diag_residuals_all)
            std_res = np.std(sec_diag_residuals_all)
            ax4.text(0.05, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}\nN: {len(sec_diag_residuals_all)}',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.8))
        
        # Chi-squared distributions
        if x_chi2_all:
            ax5.hist(x_chi2_all, bins=20, alpha=0.7, color='blue', edgecolor='black', label='X-direction')
        if y_chi2_all:
            ax5.hist(y_chi2_all, bins=20, alpha=0.7, color='red', edgecolor='black', label='Y-direction')
        if main_diag_chi2_all:
            ax5.hist(main_diag_chi2_all, bins=20, alpha=0.7, color='purple', edgecolor='black', label='Main diagonal')
        if sec_diag_chi2_all:
            ax5.hist(sec_diag_chi2_all, bins=20, alpha=0.7, color='orange', edgecolor='black', label='Sec diagonal')
        
        ax5.set_title('Reduced Chi-squared Distribution')
        ax5.set_xlabel('χ²/ndf')
        ax5.set_ylabel('Frequency')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # Position accuracy plot - 2D Gaussian fits
        true_x = data['TrueX'][:n_to_plot]
        true_y = data['TrueY'][:n_to_plot]
        fit_x = data['Fit2D_XCenter'][:n_to_plot]
        fit_y = data['Fit2D_YCenter'][:n_to_plot]
        
        distances = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
        ax6.hist(distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax6.set_title('Distance: 2D Fitted Center to True Position')
        ax6.set_xlabel('Distance [mm]')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax6.text(0.05, 0.95, f'Mean: {mean_dist:.3f} mm\nStd: {std_dist:.3f} mm\nN: {len(distances)}',
                transform=ax6.transAxes, verticalalignment='top',
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
    parser.add_argument("--overlay_only", action="store_true",
                       help="Create only overlay plots, skip individual direction plots")
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
        overlay_success_count = 0
        for i in range(n_events):
            # Create individual plots for each direction (unless overlay_only is set)
            if not args.overlay_only:
                result = create_gauss_fit_plot(i, data, args.output)
                if "Error" not in result:
                    success_count += 1
                if i % 5 == 0 or "Error" in result:
                    print(f"  {result}")
            
            # Create overlay plots for comparison
            overlay_result = create_overlay_fit_plots(i, data, args.output)
            if "Error" not in overlay_result:
                overlay_success_count += 1
            if i % 5 == 0 or "Error" in overlay_result:
                print(f"  {overlay_result}")
        
        if not args.overlay_only:
            print(f"\nSuccessfully created {success_count}/{n_events} individual plots")
        print(f"Successfully created {overlay_success_count}/{n_events} overlay plots")
    
    print(f"\nAll plots saved to: {args.output}/")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 