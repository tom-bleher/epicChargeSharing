"""
ROOT File Lorentz s Visualization Tool

This tool reads ROOT output files from the GEANT4 charge sharing simulation
and creates PDF visualizations of the pre-computed Lorentz fits for row and column projections.

Key features:
1. Reads actual simulation data from ROOT files using uproot
2. Extracts neighborhood charge distributions and fitted Lorentz parameters  
3. Plots data points with fitted Lorentz curves and residuals
4. Creates comprehensive PDF reports for analysis verification
5. NO synthetic data or fitting - only visualization of existing results
6. OPTIMIZED for parallel processing with multithreading support
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
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar
    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.count = 0
            if self.total > 0:
                print(f"Starting {self.desc}: 0/{self.total}")
        
        def update(self, n=1):
            self.count += n
            if self.total > 0 and self.count % max(1, self.total // 20) == 0:
                print(f"{self.desc}: {self.count}/{self.total} ({100*self.count/self.total:.1f}%)")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.total > 0:
                print(f"Completed {self.desc}: {self.count}/{self.total}")
    
    tqdm = FakeTqdm

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.default'] = 'regular'

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
        
        # Basic pos and pixel information
        data['TrueX'] = tree['TrueX'].array(library="np", entry_stop=max_entries)
        data['TrueY'] = tree['TrueY'].array(library="np", entry_stop=max_entries)
        data['PixelX'] = tree['PixelX'].array(library="np", entry_stop=max_entries)
        data['PixelY'] = tree['PixelY'].array(library="np", entry_stop=max_entries)
        data['IsPixelHit'] = tree['IsPixelHit'].array(library="np", entry_stop=max_entries)
        
        # Neighborhood grid data
        data['NeighborhoodCharges'] = tree['NeighborhoodCharges'].array(library="np", entry_stop=max_entries)
        
        # Row Lorentz fit results
        data['LorentzRowCenter'] = tree['LorentzRowCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzRowGamma'] = tree['LorentzRowGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzRowAmp'] = tree['LorentzRowAmp'].array(library="np", entry_stop=max_entries)
        data['LorentzRowVertOffset'] = tree['LorentzRowVertOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzRowCenterErr'] = tree['LorentzRowCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzRowGammaErr'] = tree['LorentzRowGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzRowAmpErr'] = tree['LorentzRowAmpErr'].array(library="np", entry_stop=max_entries)
        data['LorentzRowChi2red'] = tree['LorentzRowChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzRowDOF'] = tree['LorentzRowDOF'].array(library="np", entry_stop=max_entries)
        data['LorentzRowChargeErr'] = tree['LorentzRowChargeErr'].array(library="np", entry_stop=max_entries)
        
        # Col Lorentz fit results
        data['LorentzColCenter'] = tree['LorentzColCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzColGamma'] = tree['LorentzColGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzColAmp'] = tree['LorentzColAmp'].array(library="np", entry_stop=max_entries)
        data['LorentzColVertOffset'] = tree['LorentzColVertOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzColCenterErr'] = tree['LorentzColCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzColGammaErr'] = tree['LorentzColGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzColAmpErr'] = tree['LorentzColAmpErr'].array(library="np", entry_stop=max_entries)
        data['LorentzColChi2red'] = tree['LorentzColChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzColDOF'] = tree['LorentzColDOF'].array(library="np", entry_stop=max_entries)
        data['LorentzColChargeErr'] = tree['LorentzColChargeErr'].array(library="np", entry_stop=max_entries)
        
        # Try to load detector grid parameters from metadata
        try:
            if 'GridPixelSpacing_mm' in file:
                data['PixelSpacing'] = float(file['GridPixelSpacing_mm'].title.decode())
            else:
                data['PixelSpacing'] = 0.5  # Default fallback
        except:
            data['PixelSpacing'] = 0.5  # Default fallback
        
        n_events = len(data['TrueX'])
        print(f"Successly loaded {n_events} events from ROOT file")
        
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
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        x_poss, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        # Check if data is already subset to single event or is full dataset
        if isinstance(data['NeighborhoodCharges'], np.ndarray) and data['NeighborhoodCharges'].ndim == 1:
            # Single event data (already subset)
            grid_charges = data['NeighborhoodCharges']
            pixel_x = data['PixelX']
        elif hasattr(data['NeighborhoodCharges'], '__len__') and len(data['NeighborhoodCharges']) > event_idx:
            # Full dataset - extract for specific event
            grid_charges = np.array(data['NeighborhoodCharges'][event_idx])
            pixel_x = data['PixelX'][event_idx]
        else:
            return None, None, False
        
        if len(grid_charges) == 0:
            return None, None, False
        
        # Auto-detect grid size if not provided
        if grid_size is None:
            grid_size = int(np.sqrt(len(grid_charges)))
            if grid_size * grid_size != len(grid_charges):
                return None, None, False
        
        # Calc neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        pixel_spacing = data['PixelSpacing']
        
        # Find the central row (row index = grid_size // 2 = 4 for 9x9 grid)
        central_row_idx = radius
        
        # Vectorized extraction for better performance
        indices = np.arange(len(grid_charges))
        rows = indices % grid_size
        cols = indices // grid_size
        
        # Get central row mask
        central_row_mask = (rows == central_row_idx) & (grid_charges > 0)
        
        if np.sum(central_row_mask) < 3:
            return None, None, False
        
        # Extract data
        central_cols = cols[central_row_mask]
        central_charges = grid_charges[central_row_mask]
        
        # Calc poss
        offset_x = central_cols - radius
        x_poss = pixel_x + offset_x * pixel_spacing
        
        # Sort by X pos
        sort_indices = np.argsort(x_poss)
        x_poss = x_poss[sort_indices]
        charges = central_charges[sort_indices]
        
        return x_poss, charges, True
        
    except Exception as e:
        return None, None, False

def extract_column_data(event_idx, data, grid_size=None):
    """
    Extract column (Y-direction) charge data for a specific event.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        y_poss, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        # Check if data is already subset to single event or is full dataset
        if isinstance(data['NeighborhoodCharges'], np.ndarray) and data['NeighborhoodCharges'].ndim == 1:
            # Single event data (already subset)
            grid_charges = data['NeighborhoodCharges']
            pixel_y = data['PixelY']
        elif hasattr(data['NeighborhoodCharges'], '__len__') and len(data['NeighborhoodCharges']) > event_idx:
            # Full dataset - extract for specific event
            grid_charges = np.array(data['NeighborhoodCharges'][event_idx])
            pixel_y = data['PixelY'][event_idx]
        else:
            return None, None, False
        
        if len(grid_charges) == 0:
            return None, None, False
        
        # Auto-detect grid size if not provided
        if grid_size is None:
            grid_size = int(np.sqrt(len(grid_charges)))
            if grid_size * grid_size != len(grid_charges):
                return None, None, False
        
        # Calc neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        pixel_spacing = data['PixelSpacing']
        
        # Find the central column (col index = grid_size // 2 = 4 for 9x9 grid)
        central_col_idx = radius
        
        # Vectorized extraction for better performance
        indices = np.arange(len(grid_charges))
        rows = indices % grid_size
        cols = indices // grid_size
        
        # Get central column mask
        central_col_mask = (cols == central_col_idx) & (grid_charges > 0)
        
        if np.sum(central_col_mask) < 3:
            return None, None, False
        
        # Extract data
        central_rows = rows[central_col_mask]
        central_charges = grid_charges[central_col_mask]
        
        # Calc poss
        offset_y = central_rows - radius
        y_poss = pixel_y + offset_y * pixel_spacing
        
        # Sort by Y pos
        sort_indices = np.argsort(y_poss)
        y_poss = y_poss[sort_indices]
        charges = central_charges[sort_indices]
        
        return y_poss, charges, True
            
    except Exception as e:
        return None, None, False

def lorentz_1d(x, amp, center, gamma, offset):
    """
    1D Lorentz function for plotting fitted curves.
    """
    safe_gamma = max(abs(gamma), 1e-12)
    return amp / (1 + ((x - center) / safe_gamma)**2) + offset

def get_stored_charge_uncertainties(event_idx, data, direction):
    """
    Get the stored charge uncertainties from ROOT file for the specified direction.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        direction: 'row' or 'column'
    
    Returns:
        err: Single err value (5% of max charge for this event)
    """
    if direction.lower() == 'row':
        err_data = data['LorentzRowChargeErr']
    elif direction.lower() == 'column':
        err_data = data['LorentzColChargeErr']
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'row' or 'column'")
    
    # Handle both single event data and full dataset
    if np.isscalar(err_data):
        # Single event data (already subset)
        err = err_data
    elif hasattr(err_data, '__len__') and len(err_data) > event_idx:
        # Full dataset - extract for specific event
        err = err_data[event_idx]
    else:
        err = 0.05  # Default fallback
    
    return err

def calculate_charge_uncertainties(charges):
    """
    DEPRECATED: This function should not be used.
    Use get_stored_charge_uncertainties() instead to read uncertainties from ROOT file.
    """
    raise RuntimeError("calculate_charge_uncertainties() is deprecated. Use get_stored_charge_uncertainties() to read from ROOT file.")

def create_row_lorentz_plot(event_idx, data):
    """
    Create a plot showing the row Lorentz fit for a specific event.

    Returns:
        fig, success_flag
    """
    try:
        # Extract row data
        x_poss, charges, valid_data = extract_row_data(event_idx, data)
        
        if not valid_data:
            return None, False
        
        # Get fit parameters - handle both single event data and full dataset
        def get_param_value(param_data, idx):
            if np.isscalar(param_data):
                return float(param_data)
            elif hasattr(param_data, '__len__') and len(param_data) > idx:
                return float(param_data[idx])
            else:
                return 0.0
        
        fit_center = get_param_value(data['LorentzRowCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzRowGamma'], event_idx)
        fit_amp = get_param_value(data['LorentzRowAmp'], event_idx)
        fit_offset = get_param_value(data['LorentzRowVertOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzRowCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzRowGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzRowChi2red'], event_idx)
        dof = get_param_value(data['LorentzRowDOF'], event_idx)
        
        # Check if fit was success (DOF > 0 indicates success fit)
        if dof <= 0:
            return None, False
        
        # Get true pos
        true_x = get_param_value(data['TrueX'], event_idx)
        
        # Calc uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'row')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calc fitted curve and residuals
        fitted_charges = lorentz_1d(x_poss, fit_amp, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        x_min, x_max = x_poss.min() - 0.3, x_poss.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(x_poss, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_xlabel('x_pixel (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Row Lorentz Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(x_poss, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentz curve
        x_fit_range = np.linspace(x_min, x_max, 200)
        y_fit = lorentz_1d(x_fit_range, fit_amp, fit_center, fit_gamma, fit_offset)
        ax_right.plot(x_fit_range, y_fit, 'r-', linewidth=3, label='Lorentz ')
        
        # Mark true and fitted poss
        ax_right.axvline(true_x, color='green', linestyle='--', linewidth=3, alpha=0.8, label='x_true')
        ax_right.axvline(fit_center, color='red', linestyle=':', linewidth=3, alpha=0.8, label='x_fit')
        
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_xlabel('x_pixel (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Row Lorentz ', fontsize=14, pad=20)
        ax_right.grid(True, alpha=0.3, linewidth=0.8)
        
        # Add fit information as text box
        delta_x = fit_center - true_x
        info_text = (f'x_true = {true_x:.3f} mm\n'
                    f'x_fit = {fit_center:.3f} ± {fit_center_err:.3f} mm\n'
                    f'Δx = x_fit - x_true = {delta_x:.3f} mm\n'
                    f'γ = {fit_gamma:.3f} ± {fit_gamma_err:.3f} mm\n'
                    f'χ²/DOF = {chi2_red:.3f}\n'
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

def create_column_lorentz_plot(event_idx, data):
    """
    Create a plot showing the column Lorentz fit for a specific event.
    
    Returns:
        fig, success_flag
    """
    try:
        # Extract column data
        y_poss, charges, valid_data = extract_column_data(event_idx, data)
        
        if not valid_data:
            return None, False
        
        # Get fit parameters - handle both single event data and full dataset
        def get_param_value(param_data, idx):
            if np.isscalar(param_data):
                return float(param_data)
            elif hasattr(param_data, '__len__') and len(param_data) > idx:
                return float(param_data[idx])
            else:
                return 0.0
        
        fit_center = get_param_value(data['LorentzColCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzColGamma'], event_idx)
        fit_amp = get_param_value(data['LorentzColAmp'], event_idx)
        fit_offset = get_param_value(data['LorentzColVertOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzColCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzColGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzColChi2red'], event_idx)
        dof = get_param_value(data['LorentzColDOF'], event_idx)
        
        # Check if fit was success (DOF > 0 indicates success fit)
        if dof <= 0:
            return None, False
        
        # Get true pos
        true_y = get_param_value(data['TrueY'], event_idx)
        
        # Calc uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'column')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calc fitted curve and residuals
        fitted_charges = lorentz_1d(y_poss, fit_amp, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        y_min, y_max = y_poss.min() - 0.3, y_poss.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(y_poss, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(y_min, y_max)
        ax_left.set_xlabel('y_pixel (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Col Lorentz Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(y_poss, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentz curve
        y_fit_range = np.linspace(y_min, y_max, 200)
        y_fit = lorentz_1d(y_fit_range, fit_amp, fit_center, fit_gamma, fit_offset)
        ax_right.plot(y_fit_range, y_fit, 'b-', linewidth=3, label='Lorentz ')
        
        # Mark true and fitted poss
        ax_right.axvline(true_y, color='green', linestyle='--', linewidth=3, alpha=0.8, label='y_true')
        ax_right.axvline(fit_center, color='blue', linestyle=':', linewidth=3, alpha=0.8, label='y_fit')
        
        ax_right.set_xlim(y_min, y_max)
        ax_right.set_xlabel('y_pixel (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Col Lorentz ', fontsize=14, pad=20)
        ax_right.grid(True, alpha=0.3, linewidth=0.8)
        
        # Add fit information as text box
        delta_y = fit_center - true_y
        info_text = (f'y_true = {true_y:.3f} mm\n'
                    f'y_fit = {fit_center:.3f} ± {fit_center_err:.3f} mm\n'
                    f'Δy = y_fit - y_true = {delta_y:.3f} mm\n'
                    f'γ = {fit_gamma:.3f} ± {fit_gamma_err:.3f} mm\n'
                    f'χ²/DOF = {chi2_red:.3f}\n'
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

def _create_plot_worker(args):
    """
    Worker function for parallel plot generation.
    
    Args:
        args: Tuple of (event_idx, data_subset, plot_type)
    
    Returns:
        tuple: (event_idx, figure_object, success)
    """
    event_idx, data_subset, plot_type = args
    
    try:
        # Ensure matplotlib is properly configured for this thread
        import matplotlib
        matplotlib.use('Agg')  # Force non-interactive backend
        import matplotlib.pyplot as plt
        
        # Reconstruct data dictionary for this event
        data = {}
        for key, value in data_subset.items():
            data[key] = value
        
        if plot_type == 'row':
            fig, success = create_row_lorentz_plot(event_idx, data)
        elif plot_type == 'column':
            fig, success = create_column_lorentz_plot(event_idx, data)
        else:
            return event_idx, None, False
        
        if success and fig is not None:
            return event_idx, fig, True
        else:
            if fig is not None:
                plt.close(fig)
            return event_idx, None, False
            
    except Exception as e:
        print(f"Error in worker for event {event_idx}: {e}")
        return event_idx, None, False

def _prepare_data_subset(data, event_idx):
    """
    Prepare a minimal data subset for a single event to reduce memory overhead.
    
    Args:
        data: Full data dictionary
        event_idx: Event index
        
    Returns:
        dict: Minimal data subset for this event
    """
    subset = {}
    
    # List of all keys we need for plotting
    keys_needed = [
        'TrueX', 'TrueY', 'PixelX', 'PixelY', 'IsPixelHit',
        'NeighborhoodCharges', 'PixelSpacing',
        'LorentzRowCenter', 'LorentzRowGamma', 'LorentzRowAmp', 
        'LorentzRowVertOffset', 'LorentzRowCenterErr', 'LorentzRowGammaErr',
        'LorentzRowAmpErr', 'LorentzRowChi2red', 'LorentzRowDOF',
        'LorentzRowChargeErr',
        'LorentzColCenter', 'LorentzColGamma', 'LorentzColAmp',
        'LorentzColVertOffset', 'LorentzColCenterErr', 'LorentzColGammaErr',
        'LorentzColAmpErr', 'LorentzColChi2red', 'LorentzColDOF',
        'LorentzColChargeErr'
    ]
    
    for key in keys_needed:
        if key in data:
            if hasattr(data[key], '__len__') and len(data[key]) > event_idx:
                subset[key] = data[key][event_idx]
            else:
                subset[key] = data[key]  # Scalar values like PixelSpacing
    
    return subset

def create_lorentz_fit_pdfs(data, output_dir="plots", max_events=None, n_workers=None):
    """
    Create PDF files with Gauss fits visualization using parallel processing.
    
    Args:
        data: Data dictionary from ROOT file
        output_dir: Output directory for PDF files
        max_events: Maximum number of events to process
        n_workers: Number of worker processes (default: CPU count)
    
    Returns:
        tuple: (x_success_count, y_success_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    n_events = len(data['TrueX'])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    # Set batch size for processing
    batch_size = max(50, n_workers * 10)  # Process in batches to manage memory
    
    print(f"Creating Lorentz fit visualizations for {n_events} events using {n_workers} workers")
    
    # Create output paths
    x_pdf_path = os.path.join(output_dir, "lorentz_fits_row.pdf")
    y_pdf_path = os.path.join(output_dir, "lorentz_fits_column.pdf")
    
    # Pre-filter events with success fits to avoid processing invalid events
    row_valid_events = []
    col_valid_events = []
    
    for event_idx in range(n_events):
        if data['LorentzRowDOF'][event_idx] > 0:
            row_valid_events.append(event_idx)
        if data['LorentzColDOF'][event_idx] > 0:
            col_valid_events.append(event_idx)
    
    print(f"Found {len(row_valid_events)} valid row fits and {len(col_valid_events)} valid column fits")
    
    # Pre-calculate grid size to avoid repeated calculations
    if len(row_valid_events) > 0 or len(col_valid_events) > 0:
        sample_event = row_valid_events[0] if row_valid_events else col_valid_events[0]
        sample_grid = np.array(data['NeighborhoodCharges'][sample_event])
        grid_size = int(np.sqrt(len(sample_grid))) if len(sample_grid) > 0 else 9
        print(f"Detected grid size: {grid_size}x{grid_size}")
    else:
        grid_size = 9  # Default fallback
    
    x_success_count = 0
    y_success_count = 0
    
    # Process row fits
    if row_valid_events:
        print("Creating row Lorentz fits PDF...")
        
        # Prepare arguments for parallel processing
        row_args = []
        for event_idx in row_valid_events:
            data_subset = _prepare_data_subset(data, event_idx)
            row_args.append((event_idx, data_subset, 'row'))
        
        # Process in parallel with progress bar using threading
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_event = {executor.submit(_create_plot_worker, args): args[0] 
                             for args in row_args}
            
            # Collect results with progress bar
            results = {}
            with tqdm(total=len(row_args), desc="Generating row plots") as pbar:
                for future in as_completed(future_to_event):
                    event_idx = future_to_event[future]
                    try:
                        event_idx, fig, success = future.result()
                        results[event_idx] = (fig, success)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing event {event_idx}: {e}")
                        results[event_idx] = (None, False)
                        pbar.update(1)
        
        # Write results to PDF in order
        with PdfPages(x_pdf_path) as pdf:
            with tqdm(total=len(row_valid_events), desc="Writing row PDF") as pbar:
                for event_idx in row_valid_events:
                    fig, success = results.get(event_idx, (None, False))
                    if success and fig is not None:
                        pdf.savefig(fig, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        x_success_count += 1
                    pbar.update(1)
        
        # Cleanup
        del results
        del row_args
        gc.collect()
    
    # Process column fits
    if col_valid_events:
        print("Creating column Lorentz fits PDF...")
        
        # Prepare arguments for parallel processing
        col_args = []
        for event_idx in col_valid_events:
            data_subset = _prepare_data_subset(data, event_idx)
            col_args.append((event_idx, data_subset, 'column'))
        
        # Process in parallel with progress bar using threading
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_event = {executor.submit(_create_plot_worker, args): args[0] 
                             for args in col_args}
            
            # Collect results with progress bar
            results = {}
            with tqdm(total=len(col_args), desc="Generating column plots") as pbar:
                for future in as_completed(future_to_event):
                    event_idx = future_to_event[future]
                    try:
                        event_idx, fig, success = future.result()
                        results[event_idx] = (fig, success)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing event {event_idx}: {e}")
                        results[event_idx] = (None, False)
                        pbar.update(1)
        
        # Write results to PDF in order
        with PdfPages(y_pdf_path) as pdf:
            with tqdm(total=len(col_valid_events), desc="Writing column PDF") as pbar:
                for event_idx in col_valid_events:
                    fig, success = results.get(event_idx, (None, False))
                    if success and fig is not None:
                        pdf.savefig(fig, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        y_success_count += 1
                    pbar.update(1)
        
        # Cleanup
        del results
        del col_args
        gc.collect()
    
    print(f"PDF generation completed!")
    print(f"  Row fits visualized: {x_success_count}")
    print(f"  Row PDF saved to: {x_pdf_path}")
    print(f"  Col fits visualized: {y_success_count}")
    print(f"  Col PDF saved to: {y_pdf_path}")
    
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
        
        # Show Gauss fitting branches
        gauss_branches = [b for b in branches if 'Gauss' in b]
        print(f"\nGauss fitting branches ({len(gauss_branches)}):")
        for i, branch in enumerate(sorted(gauss_branches)):
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
        
        # Check for success fits
        if 'LorentzRowDOF' in branches:
            row_dof = tree['LorentzRowDOF'].array(library="np")
            success_row_fits = np.sum(row_dof > 0)
            print(f"  Success row fits: {success_row_fits} ({100*success_row_fits/n_events:.1f}%)")
        
        if 'LorentzColDOF' in branches:
            col_dof = tree['LorentzColDOF'].array(library="np")
            success_col_fits = np.sum(col_dof > 0)
            print(f"  Success column fits: {success_col_fits} ({100*success_col_fits/n_events:.1f}%)")
        
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Visualize Lorentz fits from GEANT4 charge sharing simulation ROOT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file from GEANT4 simulation")
    parser.add_argument("-o", "--output", default="lorentz_fits", 
                       help="Output directory for PDF files")
    parser.add_argument("-n", "--num_events", type=int, default=None,
                       help="Maximum number of events to process (default: all events)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker threads for parallel processing (default: CPU count)")
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
    
    # Check if we have Lorentz fitting data
    n_events = len(data['TrueX'])
    if n_events == 0:
        print("No events found in the data!")
        return 1
    
    # Count success fits
    row_success = np.sum(data['LorentzRowDOF'] > 0)
    col_success = np.sum(data['LorentzColDOF'] > 0)
    
    print(f"Found {row_success} success row fits and {col_success} success column fits")
    
    if row_success == 0 and col_success == 0:
        print("No success Lorentz fits found in the data!")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Create PDF visualizations
    x_success, y_success = create_lorentz_fit_pdfs(
        data, args.output, args.num_events, args.workers
    )
    
    print(f"\nVisualization completed!")
    print(f"  Row fits visualized: {x_success}")
    print(f"  Col fits visualized: {y_success}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 