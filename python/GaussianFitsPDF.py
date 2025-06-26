"""
ROOT File Gaussian Fits Visualization Tool

This tool reads ROOT output files from the GEANT4 charge sharing simulation
and creates PDF visualizations of the pre-computed Gaussian fits for row and column projections.

Key features:
1. Reads actual simulation data from ROOT files using uproot
2. Extracts neighborhood charge distributions and fitted Gaussian parameters  
3. Plots data points with fitted Gaussian curves and residuals
4. Creates comprehensive PDF reports for analysis verification
5. NO synthetic data or fitting - only visualization of existing results
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
import time

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def load_root_data(root_file, max_entries=None):
    """
    Load data from ROOT file created by the GEANT4 simulation.
    
    Args:
        root_file: Path to ROOT file
        max_entries: Maximum number of entries to load (None for all)
    
    Returns:
        Dictionary containing all relevant data
    """
    print(f"Loading data from ROOT file: {root_file}")
    
    try:
        file = uproot.open(root_file)
        
        # Find the tree (should be "Hits")
        tree_name = "Hits"
        if tree_name not in file:
            # Fallback search
            for key in file.keys():
                if hasattr(file[key], 'keys'):
                    tree_name = key
                    break
        
        tree = file[tree_name]
        
        # Load essential data
        data = {}
        
        # Basic position and pixel information
        data['TrueX'] = tree['TrueX'].array(library="np", entry_stop=max_entries)
        data['TrueY'] = tree['TrueY'].array(library="np", entry_stop=max_entries)
        data['PixelX'] = tree['PixelX'].array(library="np", entry_stop=max_entries)
        data['PixelY'] = tree['PixelY'].array(library="np", entry_stop=max_entries)
        data['IsPixelHit'] = tree['IsPixelHit'].array(library="np", entry_stop=max_entries)
        
        # Neighborhood grid data
        data['GridNeighborhoodCharges'] = tree['GridNeighborhoodCharges'].array(library="np", entry_stop=max_entries)
        
        # Row Gaussian fit results
        data['GaussFitRowCenter'] = tree['GaussFitRowCenter'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowStdev'] = tree['GaussFitRowStdev'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowAmplitude'] = tree['GaussFitRowAmplitude'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowVerticalOffset'] = tree['GaussFitRowVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowCenterErr'] = tree['GaussFitRowCenterErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowStdevErr'] = tree['GaussFitRowStdevErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowAmplitudeErr'] = tree['GaussFitRowAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowChi2red'] = tree['GaussFitRowChi2red'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowDOF'] = tree['GaussFitRowDOF'].array(library="np", entry_stop=max_entries)
        data['GaussFitRowChargeUncertainty'] = tree['GaussFitRowChargeUncertainty'].array(library="np", entry_stop=max_entries)
        
        # Column Gaussian fit results
        data['GaussFitColumnCenter'] = tree['GaussFitColumnCenter'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnStdev'] = tree['GaussFitColumnStdev'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnAmplitude'] = tree['GaussFitColumnAmplitude'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnVerticalOffset'] = tree['GaussFitColumnVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnCenterErr'] = tree['GaussFitColumnCenterErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnStdevErr'] = tree['GaussFitColumnStdevErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnAmplitudeErr'] = tree['GaussFitColumnAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnChi2red'] = tree['GaussFitColumnChi2red'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnDOF'] = tree['GaussFitColumnDOF'].array(library="np", entry_stop=max_entries)
        data['GaussFitColumnChargeUncertainty'] = tree['GaussFitColumnChargeUncertainty'].array(library="np", entry_stop=max_entries)
        
        # Try to load detector grid parameters from metadata
        try:
            if 'GridPixelSpacing_mm' in file:
                data['PixelSpacing'] = float(file['GridPixelSpacing_mm'].title.decode())
            else:
                data['PixelSpacing'] = 0.5  # Default fallback
        except:
            data['PixelSpacing'] = 0.5  # Default fallback
        
        n_events = len(data['TrueX'])
        print(f"Successfully loaded {n_events} events from ROOT file")
        
        # Filter for non-pixel hits only
        non_pixel_mask = ~data['IsPixelHit']
        n_non_pixel = np.sum(non_pixel_mask)
        
        if n_non_pixel > 0:
            print(f"Found {n_non_pixel} non-pixel hits out of {n_events} total events")
            # Apply filter to all arrays
            for key, values in data.items():
                if hasattr(values, '__len__') and len(values) == n_events:
                    data[key] = values[non_pixel_mask]
            
            return data
        else:
            print("Warning: No non-pixel hits found!")
            return data
        
    except Exception as e:
        print(f"Error loading ROOT file: {e}")
        return None

def extract_row_data(event_idx, data, grid_size=None):
    """
    Extract row (X-direction) charge data for a specific event.
    
    Args:
        event_idx: Event index
        data: Full data dictionary
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        x_positions, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        grid_charges = np.array(data['GridNeighborhoodCharges'][event_idx])
        
        if len(grid_charges) == 0:
            return None, None, False
        
        # Auto-detect grid size if not provided
        if grid_size is None:
            grid_size = int(np.sqrt(len(grid_charges)))
            if grid_size * grid_size != len(grid_charges):
                print(f"Warning: Grid charges length {len(grid_charges)} is not a perfect square")
                return None, None, False
        
        # Calculate neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        # Get central pixel position and spacing
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        pixel_spacing = data['PixelSpacing']
        
        # Find the central row (row index = grid_size // 2 = 4 for 9x9 grid)
        central_row_idx = radius
        
        # Extract pixels from the central row
        x_positions = []
        charges = []
        
        for i in range(len(grid_charges)):
            # Grid is stored column-major: col = i // grid_size, row = i % grid_size
            row = i % grid_size
            col = i // grid_size
            
            if row == central_row_idx:  # This is in the central row
                if grid_charges[i] > 0:  # Only include pixels with charge
                    # Calculate pixel position
                    # col ranges from 0 to grid_size-1, we want -radius to +radius
                    offset_x = col - radius
                    x_pos = pixel_x + offset_x * pixel_spacing
                    
                    x_positions.append(x_pos)
                    charges.append(grid_charges[i])
        
        if len(x_positions) < 3:
            return None, None, False
        
        # Sort by X position
        sorted_data = sorted(zip(x_positions, charges))
        x_positions, charges = zip(*sorted_data)
        
        return np.array(x_positions), np.array(charges), True
        
    except Exception as e:
        print(f"Error extracting row data for event {event_idx}: {e}")
        return None, None, False

def extract_column_data(event_idx, data, grid_size=None):
    """
    Extract column (Y-direction) charge data for a specific event.
    
    Args:
        event_idx: Event index
        data: Full data dictionary
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        y_positions, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        grid_charges = np.array(data['GridNeighborhoodCharges'][event_idx])
        
        if len(grid_charges) == 0:
            return None, None, False
        
        # Auto-detect grid size if not provided
        if grid_size is None:
            grid_size = int(np.sqrt(len(grid_charges)))
            if grid_size * grid_size != len(grid_charges):
                print(f"Warning: Grid charges length {len(grid_charges)} is not a perfect square")
                return None, None, False
        
        # Calculate neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        # Get central pixel position and spacing
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        pixel_spacing = data['PixelSpacing']
        
        # Find the central column (col index = grid_size // 2 = 4 for 9x9 grid)
        central_col_idx = radius
        
        # Extract pixels from the central column
        y_positions = []
        charges = []
        
        for i in range(len(grid_charges)):
            # Grid is stored column-major: col = i // grid_size, row = i % grid_size
            row = i % grid_size
            col = i // grid_size
            
            if col == central_col_idx:  # This is in the central column
                if grid_charges[i] > 0:  # Only include pixels with charge
                    # Calculate pixel position
                    # row ranges from 0 to grid_size-1, we want -radius to +radius
                    offset_y = row - radius
                    y_pos = pixel_y + offset_y * pixel_spacing
                    
                    y_positions.append(y_pos)
                    charges.append(grid_charges[i])
        
        if len(y_positions) < 3:
            return None, None, False
        
        # Sort by Y position
        sorted_data = sorted(zip(y_positions, charges))
        y_positions, charges = zip(*sorted_data)
        
        return np.array(y_positions), np.array(charges), True
        
    except Exception as e:
        print(f"Error extracting column data for event {event_idx}: {e}")
        return None, None, False

def gaussian_1d(x, amplitude, center, sigma, offset):
    """
    1D Gaussian function for plotting fitted curves.
    """
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2) + offset

def get_stored_charge_uncertainties(event_idx, data, direction):
    """
    Get the stored charge uncertainties from ROOT file for the specified direction.
    
    Args:
        event_idx: Event index
        data: Full data dictionary
        direction: 'row' or 'column'
    
    Returns:
        uncertainty: Single uncertainty value (5% of max charge for this event)
    """
    if direction.lower() == 'row':
        uncertainty = data['GaussFitRowChargeUncertainty'][event_idx]
    elif direction.lower() == 'column':
        uncertainty = data['GaussFitColumnChargeUncertainty'][event_idx]
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'row' or 'column'")
    
    return uncertainty

def calculate_charge_uncertainties(charges):
    """
    DEPRECATED: This function should not be used.
    Use get_stored_charge_uncertainties() instead to read uncertainties from ROOT file.
    """
    raise RuntimeError("calculate_charge_uncertainties() is deprecated. Use get_stored_charge_uncertainties() to read from ROOT file.")

def create_row_gaussian_plot(event_idx, data):
    """
    Create a plot showing the row Gaussian fit for a specific event.

    Returns:
        fig, success_flag
    """
    try:
        # Extract row data
        x_positions, charges, valid_data = extract_row_data(event_idx, data)
        
        if not valid_data:
            return None, False
        
        # Get fit parameters
        fit_center = data['GaussFitRowCenter'][event_idx]
        fit_sigma = data['GaussFitRowStdev'][event_idx]
        fit_amplitude = data['GaussFitRowAmplitude'][event_idx]
        fit_offset = data['GaussFitRowVerticalOffset'][event_idx]
        fit_center_err = data['GaussFitRowCenterErr'][event_idx]
        fit_sigma_err = data['GaussFitRowStdevErr'][event_idx]
        chi2_red = data['GaussFitRowChi2red'][event_idx]
        dof = data['GaussFitRowDOF'][event_idx]
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position
        true_x = data['TrueX'][event_idx]
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'row')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = gaussian_1d(x_positions, fit_amplitude, fit_center, fit_sigma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        x_min, x_max = x_positions.min() - 0.3, x_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(x_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8)
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_xlabel(r'$x_{pixel} (\mathrm{mm})$', fontsize=14)
        ax_left.set_ylabel(r'$Q_{pixel} - Q_{fit} (\mathrm{C})$', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Row Gaussian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(x_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Gaussian curve
        x_fit_range = np.linspace(x_min, x_max, 200)
        y_fit = gaussian_1d(x_fit_range, fit_amplitude, fit_center, fit_sigma, fit_offset)
        ax_right.plot(x_fit_range, y_fit, 'r-', linewidth=3, label='Gaussian Fit')
        
        # Mark true and fitted positions
        ax_right.axvline(true_x, color='green', linestyle='--', linewidth=3, alpha=0.8, label='$x_{true}$')
        ax_right.axvline(fit_center, color='red', linestyle=':', linewidth=3, alpha=0.8, label='$x_{fit}$')
        
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_xlabel(r'$x_{pixel} (\mathrm{mm})$', fontsize=14)
        ax_right.set_ylabel(r'$Q_{pixel} (\mathrm{C})$', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Row Gaussian Fit', fontsize=14, pad=20)
        ax_right.grid(True, alpha=0.3, linewidth=0.8)
        
        # Add fit information as text box
        delta_x = fit_center - true_x
        info_text = (f'$x_{{true}}$ = {true_x:.3f} mm\n'
                    f'$x_{{fit}}$ = {fit_center:.3f} ± {fit_center_err:.3f} mm\n'
                    f'$\\Delta x$ = $x_{{fit}} - x_{{true}}$ = {delta_x:.3f} mm\n'
                    f'$\\sigma$ = {fit_sigma:.3f} ± {fit_sigma_err:.3f} mm\n'
                    f'$\\chi^2/$DOF = {chi2_red:.3f}\n'
                    f'DOF = {dof}')
        
        ax_right.text(0.02, 0.98, info_text, transform=ax_right.transAxes, 
                     va='top', fontsize=10, bbox=dict(boxstyle='round', 
                     facecolor='white', alpha=0.8))
        
        ax_right.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig, True
        
    except Exception as e:
        print(f"Error creating row plot for event {event_idx}: {e}")
        return None, False

def create_column_gaussian_plot(event_idx, data):
    """
    Create a plot showing the column Gaussian fit for a specific event.
    
    Returns:
        fig, success_flag
    """
    try:
        # Extract column data
        y_positions, charges, valid_data = extract_column_data(event_idx, data)
        
        if not valid_data:
            return None, False
        
        # Get fit parameters
        fit_center = data['GaussFitColumnCenter'][event_idx]
        fit_sigma = data['GaussFitColumnStdev'][event_idx]
        fit_amplitude = data['GaussFitColumnAmplitude'][event_idx]
        fit_offset = data['GaussFitColumnVerticalOffset'][event_idx]
        fit_center_err = data['GaussFitColumnCenterErr'][event_idx]
        fit_sigma_err = data['GaussFitColumnStdevErr'][event_idx]
        chi2_red = data['GaussFitColumnChi2red'][event_idx]
        dof = data['GaussFitColumnDOF'][event_idx]
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position
        true_y = data['TrueY'][event_idx]
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'column')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = gaussian_1d(y_positions, fit_amplitude, fit_center, fit_sigma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        y_min, y_max = y_positions.min() - 0.3, y_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(y_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='blue', linestyle='-', linewidth=2, alpha=0.8)
        ax_left.set_xlim(y_min, y_max)
        ax_left.set_xlabel(r'$y_{pixel} (\mathrm{mm})$', fontsize=14)
        ax_left.set_ylabel(r'$Q_{pixel} - Q_{fit} (\mathrm{C})$', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Column Gaussian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(y_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Gaussian curve
        y_fit_range = np.linspace(y_min, y_max, 200)
        y_fit = gaussian_1d(y_fit_range, fit_amplitude, fit_center, fit_sigma, fit_offset)
        ax_right.plot(y_fit_range, y_fit, 'b-', linewidth=3, label='Gaussian Fit')
        
        # Mark true and fitted positions
        ax_right.axvline(true_y, color='green', linestyle='--', linewidth=3, alpha=0.8, label='$y_{true}$')
        ax_right.axvline(fit_center, color='blue', linestyle=':', linewidth=3, alpha=0.8, label='$y_{fit}$')
        
        ax_right.set_xlim(y_min, y_max)
        ax_right.set_xlabel(r'$y_{pixel} (\mathrm{mm})$', fontsize=14)
        ax_right.set_ylabel(r'$Q_{pixel} (\mathrm{C})$', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Column Gaussian Fit', fontsize=14, pad=20)
        ax_right.grid(True, alpha=0.3, linewidth=0.8)
        
        # Add fit information as text box
        delta_y = fit_center - true_y
        info_text = (f'$y_{{true}}$ = {true_y:.3f} mm\n'
                    f'$y_{{fit}}$ = {fit_center:.3f} ± {fit_center_err:.3f} mm\n'
                    f'$\\Delta y$ = $y_{{fit}} - y_{{true}}$ = {delta_y:.3f} mm\n'
                    f'$\\sigma$ = {fit_sigma:.3f} ± {fit_sigma_err:.3f} mm\n'
                    f'$\\chi^2/$DOF = {chi2_red:.3f}\n'
                    f'DOF = {dof}')
        
        ax_right.text(0.02, 0.98, info_text, transform=ax_right.transAxes, 
                     va='top', fontsize=10, bbox=dict(boxstyle='round', 
                     facecolor='white', alpha=0.8))
        
        ax_right.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig, True
        
    except Exception as e:
        print(f"Error creating column plot for event {event_idx}: {e}")
        return None, False

def create_gaussian_fit_pdfs(data, output_dir="plots", max_events=None):
    """
    Create PDF files with Gaussian fits visualization.
    
    Args:
        data: Data dictionary from ROOT file
        output_dir: Output directory for PDF files
        max_events: Maximum number of events to process
    
    Returns:
        tuple: (x_success_count, y_success_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    n_events = len(data['TrueX'])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    print(f"Creating Gaussian fit visualizations for {n_events} events")
    
    # Create output paths
    x_pdf_path = os.path.join(output_dir, "gaussian_fits_row.pdf")
    y_pdf_path = os.path.join(output_dir, "gaussian_fits_column.pdf")
    
    x_success_count = 0
    y_success_count = 0
    
    # Create X-direction (row) PDF
    print(f"Creating row Gaussian fits PDF...")
    with PdfPages(x_pdf_path) as pdf:
        for event_idx in range(n_events):
            fig, success = create_row_gaussian_plot(event_idx, data)
            if success:
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                x_success_count += 1
            if fig is not None:
                plt.close(fig)
            
            # Memory cleanup
            if event_idx % 100 == 0:
                gc.collect()
    
    # Create Y-direction (column) PDF
    print(f"Creating column Gaussian fits PDF...")
    with PdfPages(y_pdf_path) as pdf:
        for event_idx in range(n_events):
            fig, success = create_column_gaussian_plot(event_idx, data)
            if success:
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                y_success_count += 1
            if fig is not None:
                plt.close(fig)
            
            # Memory cleanup
            if event_idx % 100 == 0:
                gc.collect()
    
    print(f"PDF generation completed!")
    print(f"  Row fits visualized: {x_success_count}")
    print(f"  Row PDF saved to: {x_pdf_path}")
    print(f"  Column fits visualized: {y_success_count}")
    print(f"  Column PDF saved to: {y_pdf_path}")
    
    return x_success_count, y_success_count

def inspect_root_file(root_file):
    """
    Inspect the ROOT file to show available branches and basic statistics.
    """
    print(f"Inspecting ROOT file: {root_file}")
    
    try:
        file = uproot.open(root_file)
        print(f"Available objects in file: {list(file.keys())}")
        
        tree = file["Hits"]
        branches = tree.keys()
        
        print(f"\nTree 'Hits' contains {len(branches)} branches:")
        
        # Show Gaussian fitting branches
        gaussian_branches = [b for b in branches if 'GaussFit' in b]
        print(f"\nGaussian fitting branches ({len(gaussian_branches)}):")
        for i, branch in enumerate(sorted(gaussian_branches)):
            print(f"  {i+1:2d}: {branch}")
        
        # Show grid data branches
        grid_branches = [b for b in branches if 'Grid' in b]
        print(f"\nGrid data branches ({len(grid_branches)}):")
        for i, branch in enumerate(sorted(grid_branches)):
            print(f"  {i+1:2d}: {branch}")
        
        # Show basic statistics
        n_events = tree.num_entries
        is_pixel_hit = tree['IsPixelHit'].array(library="np")
        n_non_pixel = np.sum(~is_pixel_hit)
        
        print(f"\nBasic statistics:")
        print(f"  Total events: {n_events}")
        print(f"  Non-pixel hits: {n_non_pixel} ({100*n_non_pixel/n_events:.1f}%)")
        print(f"  Pixel hits: {n_events - n_non_pixel} ({100*(n_events - n_non_pixel)/n_events:.1f}%)")
        
        # Check for successful fits
        if 'GaussFitRowDOF' in branches:
            row_dof = tree['GaussFitRowDOF'].array(library="np")
            successful_row_fits = np.sum(row_dof > 0)
            print(f"  Successful row fits: {successful_row_fits} ({100*successful_row_fits/n_events:.1f}%)")
        
        if 'GaussFitColumnDOF' in branches:
            col_dof = tree['GaussFitColumnDOF'].array(library="np")
            successful_col_fits = np.sum(col_dof > 0)
            print(f"  Successful column fits: {successful_col_fits} ({100*successful_col_fits/n_events:.1f}%)")
        
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian fits from GEANT4 charge sharing simulation ROOT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file from GEANT4 simulation")
    parser.add_argument("-o", "--output", default="gaussian_fits", 
                       help="Output directory for PDF files")
    parser.add_argument("-n", "--num_events", type=int, default=None,
                       help="Maximum number of events to process (default: all events)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect ROOT file contents and exit")
    
    args = parser.parse_args()
    
    # Check if ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return 1
    
    # Inspect mode
    if args.inspect:
        inspect_root_file(args.root_file)
        return 0
    
    # Load data from ROOT file
    data = load_root_data(args.root_file, max_entries=args.max_entries)
    if data is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Check if we have Gaussian fitting data
    n_events = len(data['TrueX'])
    if n_events == 0:
        print("No events found in the data!")
        return 1
    
    # Count successful fits
    row_successful = np.sum(data['GaussFitRowDOF'] > 0)
    col_successful = np.sum(data['GaussFitColumnDOF'] > 0)
    
    print(f"Found {row_successful} successful row fits and {col_successful} successful column fits")
    
    if row_successful == 0 and col_successful == 0:
        print("No successful Gaussian fits found in the data!")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Create PDF visualizations
    x_success, y_success = create_gaussian_fit_pdfs(
        data, args.output, args.num_events
    )
    
    print(f"\nVisualization completed!")
    print(f"  Row fits visualized: {x_success}")
    print(f"  Column fits visualized: {y_success}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 