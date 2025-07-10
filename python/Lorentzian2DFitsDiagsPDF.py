"""
ROOT File Diagonal Lorentzian Fits Visualization Tool

This tool reads ROOT output files from the GEANT4 charge sharing simulation
and creates PDF visualizations of the pre-computed diagonal Lorentzian fits.

Key features:
1. Reads actual simulation data from ROOT files using uproot
2. Extracts diagonal neighborhood charge distributions and fitted Lorentzian parameters  
3. Plots data points with fitted Lorentzian curves and residuals
4. Creates comprehensive PDF reports for main and secondary diagonal analysis
5. NO synthetic data or fitting - only visualization of existing results
6. OPTIMIZED for parallel processing with multithreading support

Generates 4 PDFs:
- MainDiagX: Main diagonal X-component fits
- MainDiagY: Main diagonal Y-component fits  
- SecDiagX: Secondary diagonal X-component fits
- SecDiagY: Secondary diagonal Y-component fits
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
        
        # Basic position and pixel information
        data['TrueX'] = tree['TrueX'].array(library="np", entry_stop=max_entries)
        data['TrueY'] = tree['TrueY'].array(library="np", entry_stop=max_entries)
        data['PixelX'] = tree['PixelX'].array(library="np", entry_stop=max_entries)
        data['PixelY'] = tree['PixelY'].array(library="np", entry_stop=max_entries)
        data['IsPixelHit'] = tree['IsPixelHit'].array(library="np", entry_stop=max_entries)
        
        # Neighborhood grid data
        data['GridNeighborhoodCharges'] = tree['GridNeighborhoodCharges'].array(library="np", entry_stop=max_entries)
        
        # Main Diagonal X fit results (using LorentzFit instead of GaussFit)
        data['LorentzFitMainDiagXCenter'] = tree['LorentzFitMainDiagXCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXGamma'] = tree['LorentzFitMainDiagXGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXAmplitude'] = tree['LorentzFitMainDiagXAmplitude'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXVerticalOffset'] = tree['LorentzFitMainDiagXVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXCenterErr'] = tree['LorentzFitMainDiagXCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXGammaErr'] = tree['LorentzFitMainDiagXGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXAmplitudeErr'] = tree['LorentzFitMainDiagXAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXChi2red'] = tree['LorentzFitMainDiagXChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagXDOF'] = tree['LorentzFitMainDiagXDOF'].array(library="np", entry_stop=max_entries)
        
        # Try to load charge uncertainty - if not available, we'll calculate it later
        try:
            data['LorentzFitMainDiagXChargeUncertainty'] = tree['LorentzFitMainDiagXChargeUncertainty'].array(library="np", entry_stop=max_entries)
        except:
            data['LorentzFitMainDiagXChargeUncertainty'] = None
        
        # Main Diagonal Y fit results
        data['LorentzFitMainDiagYCenter'] = tree['LorentzFitMainDiagYCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYGamma'] = tree['LorentzFitMainDiagYGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYAmplitude'] = tree['LorentzFitMainDiagYAmplitude'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYVerticalOffset'] = tree['LorentzFitMainDiagYVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYCenterErr'] = tree['LorentzFitMainDiagYCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYGammaErr'] = tree['LorentzFitMainDiagYGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYAmplitudeErr'] = tree['LorentzFitMainDiagYAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYChi2red'] = tree['LorentzFitMainDiagYChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzFitMainDiagYDOF'] = tree['LorentzFitMainDiagYDOF'].array(library="np", entry_stop=max_entries)
        
        # Try to load charge uncertainty - if not available, we'll calculate it later
        try:
            data['LorentzFitMainDiagYChargeUncertainty'] = tree['LorentzFitMainDiagYChargeUncertainty'].array(library="np", entry_stop=max_entries)
        except:
            data['LorentzFitMainDiagYChargeUncertainty'] = None
        
        # Secondary Diagonal X fit results (note: uses "SecondDiag" not "SecDiag")
        data['LorentzFitSecDiagXCenter'] = tree['LorentzFitSecondDiagXCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXGamma'] = tree['LorentzFitSecondDiagXGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXAmplitude'] = tree['LorentzFitSecondDiagXAmplitude'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXVerticalOffset'] = tree['LorentzFitSecondDiagXVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXCenterErr'] = tree['LorentzFitSecondDiagXCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXGammaErr'] = tree['LorentzFitSecondDiagXGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXAmplitudeErr'] = tree['LorentzFitSecondDiagXAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXChi2red'] = tree['LorentzFitSecondDiagXChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagXDOF'] = tree['LorentzFitSecondDiagXDOF'].array(library="np", entry_stop=max_entries)
        
        # Try to load charge uncertainty - if not available, we'll calculate it later
        try:
            data['LorentzFitSecDiagXChargeUncertainty'] = tree['LorentzFitSecDiagXChargeUncertainty'].array(library="np", entry_stop=max_entries)
        except:
            data['LorentzFitSecDiagXChargeUncertainty'] = None
        
        # Secondary Diagonal Y fit results (note: uses "SecondDiag" not "SecDiag")
        data['LorentzFitSecDiagYCenter'] = tree['LorentzFitSecondDiagYCenter'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYGamma'] = tree['LorentzFitSecondDiagYGamma'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYAmplitude'] = tree['LorentzFitSecondDiagYAmplitude'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYVerticalOffset'] = tree['LorentzFitSecondDiagYVerticalOffset'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYCenterErr'] = tree['LorentzFitSecondDiagYCenterErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYGammaErr'] = tree['LorentzFitSecondDiagYGammaErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYAmplitudeErr'] = tree['LorentzFitSecondDiagYAmplitudeErr'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYChi2red'] = tree['LorentzFitSecondDiagYChi2red'].array(library="np", entry_stop=max_entries)
        data['LorentzFitSecDiagYDOF'] = tree['LorentzFitSecondDiagYDOF'].array(library="np", entry_stop=max_entries)
        
        # Try to load charge uncertainty - if not available, we'll calculate it later
        try:
            data['LorentzFitSecDiagYChargeUncertainty'] = tree['LorentzFitSecDiagYChargeUncertainty'].array(library="np", entry_stop=max_entries)
        except:
            data['LorentzFitSecDiagYChargeUncertainty'] = None
        
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

# Copy the diagonal data extraction functions from Gaussian version (they're the same)
def extract_main_diagonal_data(event_idx, data, grid_size=None):
    """
    Extract main diagonal charge data for a specific event.
    Main diagonal goes from bottom-left to top-right: (0,0), (1,1), (2,2), etc.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        positions, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        # Check if data is already subset to single event or is full dataset
        if isinstance(data['GridNeighborhoodCharges'], np.ndarray) and data['GridNeighborhoodCharges'].ndim == 1:
            # Single event data (already subset)
            grid_charges = data['GridNeighborhoodCharges']
            pixel_x = data['PixelX']
            pixel_y = data['PixelY']
        elif hasattr(data['GridNeighborhoodCharges'], '__len__') and len(data['GridNeighborhoodCharges']) > event_idx:
            # Full dataset - extract for specific event
            grid_charges = np.array(data['GridNeighborhoodCharges'][event_idx])
            pixel_x = data['PixelX'][event_idx]
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
        
        # Calculate neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        pixel_spacing = data['PixelSpacing']
        
        # Extract main diagonal elements
        diagonal_indices = []
        diagonal_charges = []
        diagonal_positions = []
        
        for i in range(grid_size):
            # Main diagonal: (i, i) in (row, col) notation
            row = i
            col = i
            index = row * grid_size + col
            
            if index < len(grid_charges) and grid_charges[index] > 0:
                diagonal_indices.append(index)
                diagonal_charges.append(grid_charges[index])
                
                # Calculate position relative to pixel center
                offset_x = col - radius
                offset_y = row - radius
                pos_x = pixel_x + offset_x * pixel_spacing
                pos_y = pixel_y + offset_y * pixel_spacing
                
                # For diagonal, use distance along diagonal as position
                # Distance from center pixel along diagonal
                diagonal_distance = np.sqrt(offset_x**2 + offset_y**2) * np.sign(offset_x)
                diagonal_positions.append(diagonal_distance)
        
        if len(diagonal_charges) < 3:
            return None, None, False
        
        # Convert to numpy arrays and sort by diagonal position
        diagonal_positions = np.array(diagonal_positions)
        diagonal_charges = np.array(diagonal_charges)
        
        sort_indices = np.argsort(diagonal_positions)
        diagonal_positions = diagonal_positions[sort_indices]
        diagonal_charges = diagonal_charges[sort_indices]
        
        return diagonal_positions, diagonal_charges, True
        
    except Exception as e:
        return None, None, False

def extract_secondary_diagonal_data(event_idx, data, grid_size=None):
    """
    Extract secondary diagonal charge data for a specific event.
    Secondary diagonal goes from top-left to bottom-right: (0,N), (1,N-1), (2,N-2), etc.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        positions, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        # Check if data is already subset to single event or is full dataset
        if isinstance(data['GridNeighborhoodCharges'], np.ndarray) and data['GridNeighborhoodCharges'].ndim == 1:
            # Single event data (already subset)
            grid_charges = data['GridNeighborhoodCharges']
            pixel_x = data['PixelX']
            pixel_y = data['PixelY']
        elif hasattr(data['GridNeighborhoodCharges'], '__len__') and len(data['GridNeighborhoodCharges']) > event_idx:
            # Full dataset - extract for specific event
            grid_charges = np.array(data['GridNeighborhoodCharges'][event_idx])
            pixel_x = data['PixelX'][event_idx]
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
        
        # Calculate neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        pixel_spacing = data['PixelSpacing']
        
        # Extract secondary diagonal elements
        diagonal_indices = []
        diagonal_charges = []
        diagonal_positions = []
        
        for i in range(grid_size):
            # Secondary diagonal: (i, grid_size-1-i) in (row, col) notation
            row = i
            col = grid_size - 1 - i
            index = row * grid_size + col
            
            if index < len(grid_charges) and grid_charges[index] > 0:
                diagonal_indices.append(index)
                diagonal_charges.append(grid_charges[index])
                
                # Calculate position relative to pixel center
                offset_x = col - radius
                offset_y = row - radius
                pos_x = pixel_x + offset_x * pixel_spacing
                pos_y = pixel_y + offset_y * pixel_spacing
                
                # For secondary diagonal, use distance along diagonal as position
                # Distance from center pixel along secondary diagonal
                diagonal_distance = np.sqrt(offset_x**2 + offset_y**2) * np.sign(-offset_x)
                diagonal_positions.append(diagonal_distance)
        
        if len(diagonal_charges) < 3:
            return None, None, False
        
        # Convert to numpy arrays and sort by diagonal position
        diagonal_positions = np.array(diagonal_positions)
        diagonal_charges = np.array(diagonal_charges)
        
        sort_indices = np.argsort(diagonal_positions)
        diagonal_positions = diagonal_positions[sort_indices]
        diagonal_charges = diagonal_charges[sort_indices]
        
        return diagonal_positions, diagonal_charges, True
        
    except Exception as e:
        return None, None, False

def lorentzian_1d(x, amplitude, center, gamma, offset):
    """
    1D Lorentzian function for plotting fitted curves.
    
    Args:
        x: Position array
        amplitude: Peak amplitude
        center: Peak center position
        gamma: Half-width at half-maximum (HWHM)
        offset: Vertical offset
    
    Returns:
        Lorentzian curve values
    """
    return amplitude / (1 + ((x - center) / gamma)**2) + offset

def get_stored_charge_uncertainties(event_idx, data, diagonal_type):
    """
    Get the stored charge uncertainties from ROOT file for the specified diagonal type.
    If the stored uncertainties are not available, calculate fallback uncertainty as 5% of max charge.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
        diagonal_type: 'main_diag_x', 'main_diag_y', 'sec_diag_x', or 'sec_diag_y'
    
    Returns:
        uncertainty: Single uncertainty value (5% of max charge for this event)
    """
    diagonal_map = {
        'main_diag_x': 'LorentzFitMainDiagXChargeUncertainty',
        'main_diag_y': 'LorentzFitMainDiagYChargeUncertainty',
        'sec_diag_x': 'LorentzFitSecDiagXChargeUncertainty',
        'sec_diag_y': 'LorentzFitSecDiagYChargeUncertainty'
    }
    
    if diagonal_type.lower() not in diagonal_map:
        raise ValueError(f"Invalid diagonal_type: {diagonal_type}. Must be one of {list(diagonal_map.keys())}")
    
    uncertainty_data = data[diagonal_map[diagonal_type.lower()]]
    
    # Check if uncertainty data is available
    if uncertainty_data is None:
        # Calculate fallback uncertainty as 5% of max charge for this event
        # Extract diagonal data to get the charge values
        if diagonal_type.startswith('main_diag'):
            diagonal_positions, charges, valid_data = extract_main_diagonal_data(event_idx, data)
        else:
            diagonal_positions, charges, valid_data = extract_secondary_diagonal_data(event_idx, data)
            
        if valid_data and len(charges) > 0:
            uncertainty = 0.05 * np.max(charges)  # 5% of max charge
        else:
            uncertainty = 0.05  # Default fallback
    else:
        # Handle both single event data and full dataset
        if np.isscalar(uncertainty_data):
            # Single event data (already subset)
            uncertainty = uncertainty_data
        elif hasattr(uncertainty_data, '__len__') and len(uncertainty_data) > event_idx:
            # Full dataset - extract for specific event
            uncertainty = uncertainty_data[event_idx]
        else:
            uncertainty = 0.05  # Default fallback
    
    return uncertainty

def create_main_diagonal_x_plot(event_idx, data):
    """
    Create a plot showing the main diagonal X-component Lorentzian fit for a specific event.

    Returns:
        fig, success_flag
    """
    try:
        # Extract main diagonal data
        diagonal_positions, charges, valid_data = extract_main_diagonal_data(event_idx, data)
        
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
        
        fit_center = get_param_value(data['LorentzFitMainDiagXCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzFitMainDiagXGamma'], event_idx)
        fit_amplitude = get_param_value(data['LorentzFitMainDiagXAmplitude'], event_idx)
        fit_offset = get_param_value(data['LorentzFitMainDiagXVerticalOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzFitMainDiagXCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzFitMainDiagXGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzFitMainDiagXChi2red'], event_idx)
        dof = get_param_value(data['LorentzFitMainDiagXDOF'], event_idx)
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position and project onto main diagonal
        true_x = get_param_value(data['TrueX'], event_idx)
        true_y = get_param_value(data['TrueY'], event_idx)
        pixel_x = get_param_value(data['PixelX'], event_idx)
        pixel_y = get_param_value(data['PixelY'], event_idx)
        pixel_spacing = data['PixelSpacing']
        
        # Project true position onto main diagonal direction
        # Main diagonal has direction (1,1), so projection distance is:
        true_offset_x = (true_x - pixel_x) / pixel_spacing
        true_offset_y = (true_y - pixel_y) / pixel_spacing
        true_diagonal_pos = true_offset_x + true_offset_y
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'main_diag_x')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = lorentzian_1d(diagonal_positions, fit_amplitude, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        pos_min, pos_max = diagonal_positions.min() - 0.3, diagonal_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(diagonal_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(pos_min, pos_max)
        ax_left.set_xlabel('Main Diagonal Position (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Main Diagonal X Lorentzian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(diagonal_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentzian curve
        pos_fit_range = np.linspace(pos_min, pos_max, 200)
        y_fit = lorentzian_1d(pos_fit_range, fit_amplitude, fit_center, fit_gamma, fit_offset)
        ax_right.plot(pos_fit_range, y_fit, 'r-', linewidth=3, label='Lorentzian Fit')
        
        # Mark true and fitted positions (projected to diagonal)
        ax_right.axvline(true_diagonal_pos, color='green', linestyle='--', linewidth=3, alpha=0.8, label='true_pos')
        ax_right.axvline(fit_center, color='red', linestyle=':', linewidth=3, alpha=0.8, label='x_fit (main diag)')
        
        ax_right.set_xlim(pos_min, pos_max)
        ax_right.set_xlabel('Main Diagonal Position (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Main Diagonal X Lorentzian Fit', fontsize=14, pad=20)
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
        print(f"Error creating main diagonal X plot for event {event_idx}: {e}")
        return None, False

def create_main_diagonal_y_plot(event_idx, data):
    """
    Create a plot showing the main diagonal Y-component Lorentzian fit for a specific event.
    
    Returns:
        fig, success_flag
    """
    try:
        # Extract main diagonal data
        diagonal_positions, charges, valid_data = extract_main_diagonal_data(event_idx, data)
        
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
        
        fit_center = get_param_value(data['LorentzFitMainDiagYCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzFitMainDiagYGamma'], event_idx)
        fit_amplitude = get_param_value(data['LorentzFitMainDiagYAmplitude'], event_idx)
        fit_offset = get_param_value(data['LorentzFitMainDiagYVerticalOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzFitMainDiagYCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzFitMainDiagYGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzFitMainDiagYChi2red'], event_idx)
        dof = get_param_value(data['LorentzFitMainDiagYDOF'], event_idx)
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position and project onto main diagonal
        true_x = get_param_value(data['TrueX'], event_idx)
        true_y = get_param_value(data['TrueY'], event_idx)
        pixel_x = get_param_value(data['PixelX'], event_idx)
        pixel_y = get_param_value(data['PixelY'], event_idx)
        pixel_spacing = data['PixelSpacing']
        
        # Project true position onto main diagonal direction
        # Main diagonal has direction (1,1), so projection distance is:
        true_offset_x = (true_x - pixel_x) / pixel_spacing
        true_offset_y = (true_y - pixel_y) / pixel_spacing
        true_diagonal_pos = true_offset_x + true_offset_y
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'main_diag_y')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = lorentzian_1d(diagonal_positions, fit_amplitude, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        pos_min, pos_max = diagonal_positions.min() - 0.3, diagonal_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(diagonal_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(pos_min, pos_max)
        ax_left.set_xlabel('Main Diagonal Position (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Main Diagonal Y Lorentzian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(diagonal_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentzian curve
        pos_fit_range = np.linspace(pos_min, pos_max, 200)
        y_fit = lorentzian_1d(pos_fit_range, fit_amplitude, fit_center, fit_gamma, fit_offset)
        ax_right.plot(pos_fit_range, y_fit, 'b-', linewidth=3, label='Lorentzian Fit')
        
        # Mark true and fitted positions (projected to diagonal)
        ax_right.axvline(true_diagonal_pos, color='green', linestyle='--', linewidth=3, alpha=0.8, label='true_pos')
        ax_right.axvline(fit_center, color='blue', linestyle=':', linewidth=3, alpha=0.8, label='y_fit (main diag)')
        
        ax_right.set_xlim(pos_min, pos_max)
        ax_right.set_xlabel('Main Diagonal Position (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Main Diagonal Y Lorentzian Fit', fontsize=14, pad=20)
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
        print(f"Error creating main diagonal Y plot for event {event_idx}: {e}")
        return None, False

def create_secondary_diagonal_x_plot(event_idx, data):
    """
    Create a plot showing the secondary diagonal X-component Lorentzian fit for a specific event.

    Returns:
        fig, success_flag
    """
    try:
        # Extract secondary diagonal data
        diagonal_positions, charges, valid_data = extract_secondary_diagonal_data(event_idx, data)
        
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
        
        fit_center = get_param_value(data['LorentzFitSecDiagXCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzFitSecDiagXGamma'], event_idx)
        fit_amplitude = get_param_value(data['LorentzFitSecDiagXAmplitude'], event_idx)
        fit_offset = get_param_value(data['LorentzFitSecDiagXVerticalOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzFitSecDiagXCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzFitSecDiagXGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzFitSecDiagXChi2red'], event_idx)
        dof = get_param_value(data['LorentzFitSecDiagXDOF'], event_idx)
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position and project onto secondary diagonal
        true_x = get_param_value(data['TrueX'], event_idx)
        true_y = get_param_value(data['TrueY'], event_idx)
        pixel_x = get_param_value(data['PixelX'], event_idx)
        pixel_y = get_param_value(data['PixelY'], event_idx)
        pixel_spacing = data['PixelSpacing']
        
        # Project true position onto secondary diagonal direction
        # Secondary diagonal has direction (1,-1), so projection distance is:
        true_offset_x = (true_x - pixel_x) / pixel_spacing
        true_offset_y = (true_y - pixel_y) / pixel_spacing
        true_diagonal_pos = true_offset_x - true_offset_y
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'sec_diag_x')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = lorentzian_1d(diagonal_positions, fit_amplitude, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        pos_min, pos_max = diagonal_positions.min() - 0.3, diagonal_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(diagonal_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='orange', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(pos_min, pos_max)
        ax_left.set_xlabel('Secondary Diagonal Position (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Secondary Diagonal X Lorentzian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(diagonal_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentzian curve
        pos_fit_range = np.linspace(pos_min, pos_max, 200)
        y_fit = lorentzian_1d(pos_fit_range, fit_amplitude, fit_center, fit_gamma, fit_offset)
        ax_right.plot(pos_fit_range, y_fit, 'orange', linewidth=3, label='Lorentzian Fit')
        
        # Mark true and fitted positions (projected to diagonal)
        ax_right.axvline(true_diagonal_pos, color='green', linestyle='--', linewidth=3, alpha=0.8, label='true_pos')
        ax_right.axvline(fit_center, color='orange', linestyle=':', linewidth=3, alpha=0.8, label='x_fit (sec diag)')
        
        ax_right.set_xlim(pos_min, pos_max)
        ax_right.set_xlabel('Secondary Diagonal Position (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Secondary Diagonal X Lorentzian Fit', fontsize=14, pad=20)
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
        print(f"Error creating secondary diagonal X plot for event {event_idx}: {e}")
        return None, False

def create_secondary_diagonal_y_plot(event_idx, data):
    """
    Create a plot showing the secondary diagonal Y-component Lorentzian fit for a specific event.
    
    Returns:
        fig, success_flag
    """
    try:
        # Extract secondary diagonal data
        diagonal_positions, charges, valid_data = extract_secondary_diagonal_data(event_idx, data)
        
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
        
        fit_center = get_param_value(data['LorentzFitSecDiagYCenter'], event_idx)
        fit_gamma = get_param_value(data['LorentzFitSecDiagYGamma'], event_idx)
        fit_amplitude = get_param_value(data['LorentzFitSecDiagYAmplitude'], event_idx)
        fit_offset = get_param_value(data['LorentzFitSecDiagYVerticalOffset'], event_idx)
        fit_center_err = get_param_value(data['LorentzFitSecDiagYCenterErr'], event_idx)
        fit_gamma_err = get_param_value(data['LorentzFitSecDiagYGammaErr'], event_idx)
        chi2_red = get_param_value(data['LorentzFitSecDiagYChi2red'], event_idx)
        dof = get_param_value(data['LorentzFitSecDiagYDOF'], event_idx)
        
        # Check if fit was successful (DOF > 0 indicates successful fit)
        if dof <= 0:
            return None, False
        
        # Get true position and project onto secondary diagonal
        true_x = get_param_value(data['TrueX'], event_idx)
        true_y = get_param_value(data['TrueY'], event_idx)
        pixel_x = get_param_value(data['PixelX'], event_idx)
        pixel_y = get_param_value(data['PixelY'], event_idx)
        pixel_spacing = data['PixelSpacing']
        
        # Project true position onto secondary diagonal direction
        # Secondary diagonal has direction (1,-1), so projection distance is:
        true_offset_x = (true_x - pixel_x) / pixel_spacing
        true_offset_y = (true_y - pixel_y) / pixel_spacing
        true_diagonal_pos = true_offset_x - true_offset_y
        
        # Calculate uncertainties
        uncertainties = get_stored_charge_uncertainties(event_idx, data, 'sec_diag_y')
        # Create array of uncertainties - same value for all data points
        uncertainties = np.full_like(charges, uncertainties)
        
        # Calculate fitted curve and residuals
        fitted_charges = lorentzian_1d(diagonal_positions, fit_amplitude, fit_center, fit_gamma, fit_offset)
        residuals = charges - fitted_charges
        
        # Create figure with two panels
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine plot range
        pos_min, pos_max = diagonal_positions.min() - 0.3, diagonal_positions.max() + 0.3
        
        # LEFT PANEL: Residuals
        ax_left.errorbar(diagonal_positions, residuals, yerr=uncertainties,
                        fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                        elinewidth=1.5, alpha=0.8, label='Data')
        ax_left.axhline(y=0, color='purple', linestyle='--', linewidth=2, alpha=0.8)
        ax_left.set_xlim(pos_min, pos_max)
        ax_left.set_xlabel('Secondary Diagonal Position (mm)', fontsize=14)
        ax_left.set_ylabel('Q_pixel - Q_fit (C)', fontsize=14)
        ax_left.set_title(f'Event {event_idx}: Secondary Diagonal Y Lorentzian Residuals', fontsize=14, pad=20)
        ax_left.grid(True, alpha=0.3, linewidth=0.8)
        
        # RIGHT PANEL: Data and fitted curve
        ax_right.errorbar(diagonal_positions, charges, yerr=uncertainties,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5, 
                         elinewidth=1.5, alpha=0.8, label='Data Points')
        
        # Plot fitted Lorentzian curve
        pos_fit_range = np.linspace(pos_min, pos_max, 200)
        y_fit = lorentzian_1d(pos_fit_range, fit_amplitude, fit_center, fit_gamma, fit_offset)
        ax_right.plot(pos_fit_range, y_fit, 'purple', linewidth=3, label='Lorentzian Fit')
        
        # Mark true and fitted positions (projected to diagonal)
        ax_right.axvline(true_diagonal_pos, color='green', linestyle='--', linewidth=3, alpha=0.8, label='true_pos')
        ax_right.axvline(fit_center, color='purple', linestyle=':', linewidth=3, alpha=0.8, label='y_fit (sec diag)')
        
        ax_right.set_xlim(pos_min, pos_max)
        ax_right.set_xlabel('Secondary Diagonal Position (mm)', fontsize=14)
        ax_right.set_ylabel('Q_pixel (C)', fontsize=14)
        ax_right.set_title(f'Event {event_idx}: Secondary Diagonal Y Lorentzian Fit', fontsize=14, pad=20)
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
        print(f"Error creating secondary diagonal Y plot for event {event_idx}: {e}")
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
        
        if plot_type == 'main_diag_x':
            fig, success = create_main_diagonal_x_plot(event_idx, data)
        elif plot_type == 'main_diag_y':
            fig, success = create_main_diagonal_y_plot(event_idx, data)
        elif plot_type == 'sec_diag_x':
            fig, success = create_secondary_diagonal_x_plot(event_idx, data)
        elif plot_type == 'sec_diag_y':
            fig, success = create_secondary_diagonal_y_plot(event_idx, data)
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
        'GridNeighborhoodCharges', 'PixelSpacing',
        # Main diagonal X parameters
        'LorentzFitMainDiagXCenter', 'LorentzFitMainDiagXGamma', 'LorentzFitMainDiagXAmplitude', 
        'LorentzFitMainDiagXVerticalOffset', 'LorentzFitMainDiagXCenterErr', 'LorentzFitMainDiagXGammaErr',
        'LorentzFitMainDiagXAmplitudeErr', 'LorentzFitMainDiagXChi2red', 'LorentzFitMainDiagXDOF',
        'LorentzFitMainDiagXChargeUncertainty',
        # Main diagonal Y parameters
        'LorentzFitMainDiagYCenter', 'LorentzFitMainDiagYGamma', 'LorentzFitMainDiagYAmplitude',
        'LorentzFitMainDiagYVerticalOffset', 'LorentzFitMainDiagYCenterErr', 'LorentzFitMainDiagYGammaErr',
        'LorentzFitMainDiagYAmplitudeErr', 'LorentzFitMainDiagYChi2red', 'LorentzFitMainDiagYDOF',
        'LorentzFitMainDiagYChargeUncertainty',
        # Secondary diagonal X parameters
        'LorentzFitSecDiagXCenter', 'LorentzFitSecDiagXGamma', 'LorentzFitSecDiagXAmplitude', 
        'LorentzFitSecDiagXVerticalOffset', 'LorentzFitSecDiagXCenterErr', 'LorentzFitSecDiagXGammaErr',
        'LorentzFitSecDiagXAmplitudeErr', 'LorentzFitSecDiagXChi2red', 'LorentzFitSecDiagXDOF',
        'LorentzFitSecDiagXChargeUncertainty',
        # Secondary diagonal Y parameters
        'LorentzFitSecDiagYCenter', 'LorentzFitSecDiagYGamma', 'LorentzFitSecDiagYAmplitude',
        'LorentzFitSecDiagYVerticalOffset', 'LorentzFitSecDiagYCenterErr', 'LorentzFitSecDiagYGammaErr',
        'LorentzFitSecDiagYAmplitudeErr', 'LorentzFitSecDiagYChi2red', 'LorentzFitSecDiagYDOF',
        'LorentzFitSecDiagYChargeUncertainty'
    ]
    
    for key in keys_needed:
        if key in data:
            if hasattr(data[key], '__len__') and len(data[key]) > event_idx:
                subset[key] = data[key][event_idx]
            else:
                subset[key] = data[key]  # Scalar values like PixelSpacing
    
    return subset

def create_diagonal_lorentzian_fit_pdfs(data, output_dir="diagonal_lorentzian_plots", max_events=None, n_workers=None):
    """
    Create PDF files with diagonal Lorentzian fits visualization using parallel processing.
    
    Args:
        data: Data dictionary from ROOT file
        output_dir: Output directory for PDF files
        max_events: Maximum number of events to process
        n_workers: Number of worker processes (default: CPU count)
    
    Returns:
        tuple: (main_diag_x_count, main_diag_y_count, sec_diag_x_count, sec_diag_y_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    n_events = len(data['TrueX'])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Creating diagonal Lorentzian fit visualizations for {n_events} events using {n_workers} workers")
    
    # Create output paths
    main_diag_x_pdf = os.path.join(output_dir, "lorentzian_fits_main_diagonal_x.pdf")
    main_diag_y_pdf = os.path.join(output_dir, "lorentzian_fits_main_diagonal_y.pdf")
    sec_diag_x_pdf = os.path.join(output_dir, "lorentzian_fits_secondary_diagonal_x.pdf")
    sec_diag_y_pdf = os.path.join(output_dir, "lorentzian_fits_secondary_diagonal_y.pdf")
    
    # Pre-filter events with successful fits
    main_diag_x_valid = []
    main_diag_y_valid = []
    sec_diag_x_valid = []
    sec_diag_y_valid = []
    
    for event_idx in range(n_events):
        if data['LorentzFitMainDiagXDOF'][event_idx] > 0:
            main_diag_x_valid.append(event_idx)
        if data['LorentzFitMainDiagYDOF'][event_idx] > 0:
            main_diag_y_valid.append(event_idx)
        if data['LorentzFitSecDiagXDOF'][event_idx] > 0:
            sec_diag_x_valid.append(event_idx)
        if data['LorentzFitSecDiagYDOF'][event_idx] > 0:
            sec_diag_y_valid.append(event_idx)
    
    print(f"Found {len(main_diag_x_valid)} valid main diagonal X fits")
    print(f"Found {len(main_diag_y_valid)} valid main diagonal Y fits")
    print(f"Found {len(sec_diag_x_valid)} valid secondary diagonal X fits")
    print(f"Found {len(sec_diag_y_valid)} valid secondary diagonal Y fits")
    
    success_counts = {}
    
    # Define diagonal types and their corresponding data
    diagonal_types = [
        ('main_diag_x', main_diag_x_valid, main_diag_x_pdf, "Main Diagonal X"),
        ('main_diag_y', main_diag_y_valid, main_diag_y_pdf, "Main Diagonal Y"),
        ('sec_diag_x', sec_diag_x_valid, sec_diag_x_pdf, "Secondary Diagonal X"),
        ('sec_diag_y', sec_diag_y_valid, sec_diag_y_pdf, "Secondary Diagonal Y")
    ]
    
    for diag_type, valid_events, pdf_path, display_name in diagonal_types:
        if not valid_events:
            print(f"No valid {display_name} fits found, skipping...")
            success_counts[diag_type] = 0
            continue
            
        print(f"Creating {display_name} fits PDF...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for event_idx in valid_events:
            data_subset = _prepare_data_subset(data, event_idx)
            args_list.append((event_idx, data_subset, diag_type))
        
        # Process in parallel with progress bar using threading
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_event = {executor.submit(_create_plot_worker, args): args[0] 
                             for args in args_list}
            
            # Collect results with progress bar
            results = {}
            with tqdm(total=len(args_list), desc=f"Generating {display_name} plots") as pbar:
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
        success_count = 0
        with PdfPages(pdf_path) as pdf:
            with tqdm(total=len(valid_events), desc=f"Writing {display_name} PDF") as pbar:
                for event_idx in valid_events:
                    fig, success = results.get(event_idx, (None, False))
                    if success and fig is not None:
                        pdf.savefig(fig, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        success_count += 1
                    pbar.update(1)
        
        success_counts[diag_type] = success_count
        print(f"  {display_name} fits visualized: {success_count}")
        print(f"  {display_name} PDF saved to: {pdf_path}")
        
        # Cleanup
        del results
        del args_list
        gc.collect()
    
    print(f"\nDiagonal Lorentzian PDF generation completed!")
    return (success_counts.get('main_diag_x', 0), 
            success_counts.get('main_diag_y', 0),
            success_counts.get('sec_diag_x', 0), 
            success_counts.get('sec_diag_y', 0))

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
        
        # Show diagonal Lorentzian fitting branches
        diagonal_branches = [b for b in branches if 'LorentzFit' in b and 'Diag' in b]
        print(f"\nDiagonal Lorentzian fitting branches ({len(diagonal_branches)}):")
        for i, branch in enumerate(sorted(diagonal_branches)):
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
        
        # Check for successful diagonal fits
        diagonal_dofs = [
            ('LorentzFitMainDiagXDOF', 'Main Diagonal X'),
            ('LorentzFitMainDiagYDOF', 'Main Diagonal Y'),
            ('LorentzFitSecDiagXDOF', 'Secondary Diagonal X'),
            ('LorentzFitSecDiagYDOF', 'Secondary Diagonal Y')
        ]
        
        for dof_branch, name in diagonal_dofs:
            if dof_branch in branches:
                dof_data = tree[dof_branch].array(library="np")
                successful_fits = np.sum(dof_data > 0)
                print(f"  Successful {name} fits: {successful_fits} ({100*successful_fits/n_events:.1f}%)")
        
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Visualize diagonal Lorentzian fits from GEANT4 charge sharing simulation ROOT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file from GEANT4 simulation")
    parser.add_argument("-o", "--output", default="diagonal_lorentzian_fits", 
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
    
    # Check if we have diagonal Lorentzian fitting data
    n_events = len(data['TrueX'])
    if n_events == 0:
        print("No events found in the data!")
        return 1
    
    # Count successful fits
    main_diag_x_successful = np.sum(data['LorentzFitMainDiagXDOF'] > 0)
    main_diag_y_successful = np.sum(data['LorentzFitMainDiagYDOF'] > 0)
    sec_diag_x_successful = np.sum(data['LorentzFitSecDiagXDOF'] > 0)
    sec_diag_y_successful = np.sum(data['LorentzFitSecDiagYDOF'] > 0)
    
    print(f"Found {main_diag_x_successful} successful main diagonal X fits")
    print(f"Found {main_diag_y_successful} successful main diagonal Y fits")
    print(f"Found {sec_diag_x_successful} successful secondary diagonal X fits")
    print(f"Found {sec_diag_y_successful} successful secondary diagonal Y fits")
    
    total_successful = main_diag_x_successful + main_diag_y_successful + sec_diag_x_successful + sec_diag_y_successful
    if total_successful == 0:
        print("No successful diagonal Lorentzian fits found in the data!")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Create PDF visualizations
    main_diag_x_count, main_diag_y_count, sec_diag_x_count, sec_diag_y_count = create_diagonal_lorentzian_fit_pdfs(
        data, args.output, args.num_events, args.workers
    )
    
    print(f"\nVisualization completed!")
    print(f"  Main diagonal X fits visualized: {main_diag_x_count}")
    print(f"  Main diagonal Y fits visualized: {main_diag_y_count}")
    print(f"  Secondary diagonal X fits visualized: {sec_diag_x_count}")
    print(f"  Secondary diagonal Y fits visualized: {sec_diag_y_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
