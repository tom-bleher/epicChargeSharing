"""
ROOT File 3D Lorentz s Visualization Tool

This tool reads ROOT output files from the GEANT4 charge sharing simulation
and creates PDF visualizations of the pre-computed 3D Lorentz surface fits.

Key features:
1. Reads actual simulation data from ROOT files using uproot
2. Extracts neighborhood charge distributions and fitted 3D Lorentz parameters  
3. Creates 3D surface plots with fitted Lorentz surfaces and residuals
4. Creates comprehensive PDF reports for analysis verification
5. NO synthetic data or fitting - only visualization of existing results
6. OPTIMIZED for parallel processing with multithreading support
7. Shows both 3D surface plots and 2D contour projections
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
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
        
        # 3D Lorentz fit results
        data['Lorentz3DAmp'] = tree['3DLorentzAmp'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DCenterX'] = tree['3DLorentzCenterX'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DCenterY'] = tree['3DLorentzCenterY'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DGammaX'] = tree['3DLorentzGammaX'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DGammaY'] = tree['3DLorentzGammaY'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DVertOffset'] = tree['3DLorentzVertOffset'].array(library="np", entry_stop=max_entries)
        
        # Error estimates
        data['Lorentz3DAmpErr'] = tree['3DLorentzAmpErr'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DCenterXErr'] = tree['3DLorentzCenterXErr'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DCenterYErr'] = tree['3DLorentzCenterYErr'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DGammaXErr'] = tree['3DLorentzGammaXErr'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DGammaYErr'] = tree['3DLorentzGammaYErr'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DVertOffsetErr'] = tree['3DLorentzVertOffsetErr'].array(library="np", entry_stop=max_entries)
        
        #  quality metrics
        data['Lorentz3DChi2red'] = tree['3DLorentzChi2red'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DDOF'] = tree['3DLorentzDOF'].array(library="np", entry_stop=max_entries)
        data['Lorentz3DChargeErr'] = tree['3DLorentzChargeErr'].array(library="np", entry_stop=max_entries)
        
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

def extract_3d_data(event_idx, data, grid_size=None):
    """
    Extract 3D surface data for a specific event.
    
    Args:
        event_idx: Event index (0 if data is already subset to single event)
        data: Full data dictionary or single-event data subset
        grid_size: Size of neighborhood grid (auto-detected if None)
    
    Returns:
        x_coords, y_coords, charges, valid_data_flag
    """
    try:
        # Get neighborhood data for this event
        grid_data = data['NeighborhoodCharges']
        
        # Detect if this is subset data (scalar/single values) or full dataset (arrays)
        pixel_x_data = data['PixelX']
        is_subset_data = np.isscalar(pixel_x_data) or (hasattr(pixel_x_data, '__len__') and len(pixel_x_data) == 1)
        
        if is_subset_data:
            # Single event data (already subset) - use scalar values directly
            if hasattr(grid_data, '__len__'):
                grid_charges = np.array(grid_data)
            else:
                return None, None, None, False
            pixel_x = float(pixel_x_data) if np.isscalar(pixel_x_data) else float(pixel_x_data[0])
            pixel_y = float(data['PixelY']) if np.isscalar(data['PixelY']) else float(data['PixelY'][0])
        else:
            # Full dataset case - index with event_idx
            if hasattr(grid_data, '__len__') and len(grid_data) > event_idx:
                raw_grid = grid_data[event_idx]
                if hasattr(raw_grid, '__len__'):
                    grid_charges = np.array(raw_grid)
                else:
                    return None, None, None, False
                pixel_x = data['PixelX'][event_idx]
                pixel_y = data['PixelY'][event_idx]
            else:
                return None, None, None, False
        
        if len(grid_charges) == 0:
            return None, None, None, False
        
        # Auto-detect grid size if not provided
        if grid_size is None:
            grid_size = int(np.sqrt(len(grid_charges)))
            if grid_size * grid_size != len(grid_charges):
                return None, None, None, False
        
        # Calc neighborhood radius from grid size
        radius = grid_size // 2  # For 9x9 grid, radius = 4
        
        pixel_spacing = data['PixelSpacing']
        
        # Extract grid coordinates and charges
        indices = np.arange(len(grid_charges))
        rows = indices % grid_size
        cols = indices // grid_size
        
        # Filter out zero charges
        valid_mask = grid_charges > 0
        n_valid = np.sum(valid_mask)
        
        if n_valid < 6:  # Need at least 6 points for 3D fit (6 parameters)
            return None, None, None, False
        
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_charges = grid_charges[valid_mask]
        
        # Calc actual world coordinates
        offset_x = valid_cols - radius
        offset_y = valid_rows - radius
        x_coords = pixel_x + offset_x * pixel_spacing
        y_coords = pixel_y + offset_y * pixel_spacing
        
        return x_coords, y_coords, valid_charges, True
        
    except Exception as e:
        return None, None, None, False

def lorentz_3d(x, y, amp, center_x, center_y, gamma_x, gamma_y, offset):
    """
    3D Lorentz function for plotting fitted surfaces.
    z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2) + B
    """
    dx = x - center_x
    dy = y - center_y
    normalized_dx = dx / gamma_x
    normalized_dy = dy / gamma_y
    denominator = 1.0 + normalized_dx**2 + normalized_dy**2
    return amp / denominator + offset

def get_stored_3d_charge_uncertainties(event_idx, data):
    """
    Get the stored charge uncertainties from ROOT file for 3D fitting.
    
    Args:
        event_idx: Event index (ignored if data is already subset)
        data: Full data dictionary or single-event data subset
    
    Returns:
        err: Single err value (5% of max charge for this event)
    """
    err_data = data['Lorentz3DChargeErr']
    
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

def create_3d_lorentz_plot(event_idx, data):
    """
    Create a compact visualization of the 3D Lorentz fit for one event.

    The figure now contains ONLY TWO panels (vs. previous 4):
      1. Left  – 2-D scatter of residuals (Data − ) coloured by value,
                 making it immediately obvious where the fit under/over-estimates.
      2. Right – 3-D surface plot of the fitted Lorentz overlaid with the
                 measured charge distribution and markers for the true and
                 fitted hit poss.

    Both panels include auxiliary information (colourbars / text boxes) so that
    the viewer can quickly judge the goodness-of-fit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    success_flag : bool
    """
    try:
        # ----------------------------- Data extraction -----------------------------
        x_coords, y_coords, charges, valid_data = extract_3d_data(event_idx, data)
        if not valid_data:
            return None, False
        
        # Helper to seamlessly pick scalar or array value
        # For subset data, param will be scalar. For full data, param will be array.
        def _get(param, idx, default=0.0):
            try:
                if np.isscalar(param):
                    return float(param)
                elif hasattr(param, '__len__'):
                    if len(param) == 1:
                        # Single-element array from subset data
                        return float(param[0])
                    elif len(param) > idx:
                        # Multi-element array from full dataset
                        return float(param[idx])
                    else:
                        return default
                else:
                    return default
            except (TypeError, ValueError, IndexError):
                return default

        fit_amp = _get(data['Lorentz3DAmp'], event_idx)
        fit_cx  = _get(data['Lorentz3DCenterX'],   event_idx)
        fit_cy  = _get(data['Lorentz3DCenterY'],   event_idx)
        fit_gx  = _get(data['Lorentz3DGammaX'],    event_idx)
        fit_gy  = _get(data['Lorentz3DGammaY'],    event_idx)
        fit_off = _get(data['Lorentz3DVertOffset'], event_idx)

        # Retrieve uncertainties for displaying
        fit_cx_err = _get(data['Lorentz3DCenterXErr'], event_idx)
        fit_cy_err = _get(data['Lorentz3DCenterYErr'], event_idx)
        fit_gx_err = _get(data['Lorentz3DGammaXErr'],  event_idx)
        fit_gy_err = _get(data['Lorentz3DGammaYErr'],  event_idx)
        
        chi2_red = _get(data['Lorentz3DChi2red'], event_idx)
        dof      = _get(data['Lorentz3DDOF'],     event_idx)
        
        if dof <= 0:  # unsuccess fit according to stored metadata
            return None, False
        
        # True hit pos for reference
        true_x = _get(data['TrueX'], event_idx)
        true_y = _get(data['TrueY'], event_idx)

        # -----------------------------  evaluation -----------------------------
        fitted_charges = lorentz_3d(x_coords, y_coords,
                                        fit_amp, fit_cx, fit_cy, fit_gx, fit_gy, fit_off)
        residuals = charges - fitted_charges
        rms_res   = np.sqrt(np.mean(residuals**2))

        # Setup figure – two columns with constrained layout for symmetry ----------
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        # ----------------------------- Residual panel -----------------------------
        ax_res = fig.add_subplot(gs[0])
        max_abs_res = np.max(np.abs(residuals))
        sc = ax_res.scatter(x_coords, y_coords, c=residuals, cmap='RdBu',
                            vmin=-max_abs_res, vmax=max_abs_res, s=60, edgecolors='k', alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax_res, pad=0.01, label='Residual (C)')
        ax_res.axhline(true_y, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_res.axvline(true_x, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        # Add lines for reconstructed (fit) pos
        ax_res.axhline(fit_cy, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_res.axvline(fit_cx, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)

        # Legend showing meaning of guides
        import matplotlib.lines as mlines
        true_proxy = mlines.Line2D([], [], color='green', linestyle='--', linewidth=1.5, label='True (x, y)')
        fit_proxy  = mlines.Line2D([], [], color='blue', linestyle=':', linewidth=1.5, label=' (x, y)')
        ax_res.legend(handles=[true_proxy, fit_proxy], loc='upper right', fontsize=9, framealpha=0.9)
        ax_res.set_xlabel('X Pos (mm)', fontsize=12)
        ax_res.set_ylabel('Y Pos (mm)', fontsize=12)
        ax_res.set_title(f'Event {event_idx}: Residuals (Data − )', fontsize=14, pad=15)
        ax_res.set_aspect('equal', adjustable='box')
        ax_res.grid(True, alpha=0.3)

        # -----------------------------  panel ----------------------------------
        ax_fit = fig.add_subplot(gs[1], projection='3d')
        # ------------------------------------------------------------------
        # Build a *symmetric* square XY footprint so that the 3-D panel
        # visually occupies the same extent as the residual panel.  This
        # prevents the right-hand plot from looking "smaller" and keeps the
        # bottom corner (minimum Y) aligned with the left plot.
        # ------------------------------------------------------------------
        margin = 0.0  # extra space [mm] around the data, adjust if desired

        # Determine the data span
        x_center = 0.5 * (x_coords.max() + x_coords.min())
        y_center = 0.5 * (y_coords.max() + y_coords.min())

        # Largest half-span in either X or Y → ensures square footprint
        half_span = max(x_coords.max() - x_center,
                        x_center - x_coords.min(),
                        y_coords.max() - y_center,
                        y_center - y_coords.min()) + margin

        # Symmetric limits
        x_min, x_max = x_center - half_span, x_center + half_span
        y_min, y_max = y_center - half_span, y_center + half_span
        
        Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 60),
                             np.linspace(y_min, y_max, 60))
        Zg = lorentz_3d(Xg, Yg, fit_amp, fit_cx, fit_cy, fit_gx, fit_gy, fit_off)
        surf = ax_fit.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.6, linewidth=0)
        # Apply the same symmetric limits to the axes so the rendered view
        # exactly matches the generated grid extents.
        ax_fit.set_xlim(x_min, x_max)
        ax_fit.set_ylim(y_min, y_max)
        # Overlay measured charges
        ax_fit.scatter(x_coords, y_coords, charges, c='red', s=60, depthshade=True, label='Data')

        # Draw vert error bars if charge err is available (>0)
        err = get_stored_3d_charge_uncertainties(event_idx, data)
        if err and float(err) > 0:
            for xi, yi, zi in zip(x_coords, y_coords, charges):
                ax_fit.plot([xi, xi], [yi, yi], [zi - err, zi + err],
                            color='gray', alpha=0.5, linewidth=1)

            # Keep error bars silent in legend; only show Data
            ax_fit.legend(loc='upper right', fontsize=9, framealpha=0.9)
        else:
            ax_fit.legend(loc='upper right')

        ax_fit.set_xlabel('X Pos (mm)', fontsize=12)
        ax_fit.set_ylabel('Y Pos (mm)', fontsize=12)
        ax_fit.set_zlabel('Charge (C)', fontsize=12)
        ax_fit.set_title(f'Event {event_idx}: 3D Lorentz Surface ', fontsize=14, pad=15)
        # Use equal XY aspect and a balanced Z aspect for visual symmetry
        try:
            # Increase Z aspect so the 3-D axis occupies more vert space.
            # A cubic aspect (1,1,1) yields a visually larger 3-D plot whose
            # bottom edge aligns better with the left residual panel.
            ax_fit.set_box_aspect([1, 1, 1])  # Requires Matplotlib ≥3.3
            # Anchor the axis to the bottom of its slot to remove excess
            # whitespace below the 3-D plot.
            ax_fit.set_anchor("S")
        except Exception:
            pass
        ax_fit.view_init(elev=25, azim=45)  # pleasant viewing angle
        # (Legend already placed inside – no further call)

        # ---------------------- Info textbox -----------------------------------
        delta_x = fit_cx - true_x
        delta_y = fit_cy - true_y
        info_text = (
            f"x_true = {true_x:.3f} mm\n"
            f"x_fit = {fit_cx:.3f} ± {fit_cx_err:.3f} mm\n"
            f"Δx = x_fit - x_true = {delta_x:.3f} mm\n"
            f"γ_x = {fit_gx:.3f} ± {fit_gx_err:.3f} mm\n"
            f"y_true = {true_y:.3f} mm\n"
            f"y_fit = {fit_cy:.3f} ± {fit_cy_err:.3f} mm\n"
            f"Δy = y_fit - y_true = {delta_y:.3f} mm\n"
            f"γ_y = {fit_gy:.3f} ± {fit_gy_err:.3f} mm\n"
            f"χ²/DOF = {chi2_red:.3f}\n"
            f"DOF = {dof}"
        )

        # Place text box at left inside the 3D panel area (using Axes coords)
        ax_fit.text2D(0.02, 0.98, info_text, transform=ax_fit.transAxes,
                      va='top', ha='left', fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # constrained_layout handles spacing
        # plt.tight_layout()  # not needed with constrained_layout
        return fig, True
        
    except Exception as e:
        print(f"Error creating 3D Lorentz plot for event {event_idx}: {e}")
        return None, False

def _create_3d_lorentz_plot_worker(args):
    """
    Worker function for parallel 3D Lorentz plot generation.
    
    Args:
        args: Tuple of (event_idx, data_subset)
        
    Returns:
        tuple: (event_idx, figure_object, success)
    """
    event_idx, data_subset = args
    
    try:
        # Ensure matplotlib is properly configured for this thread
        import matplotlib
        matplotlib.use('Agg')  # Force non-interactive backend
        import matplotlib.pyplot as plt
        
        # Reconstruct data dictionary for this event
        data = {}
        for key, value in data_subset.items():
            data[key] = value
        
        # Use event_idx=0 since data is already subset to single event
        fig, success = create_3d_lorentz_plot(0, data)
        
        if success and fig is not None:
            return event_idx, fig, True
        else:
            if fig is not None:
                plt.close(fig)
            return event_idx, None, False
            
    except Exception as e:
        print(f"Error in 3D Lorentz worker for event {event_idx}: {e}")
        import traceback
        traceback.print_exc()
        return event_idx, None, False

def _prepare_3d_lorentz_data_subset(data, event_idx):
    """
    Prepare a minimal data subset for a single event to reduce memory overhead.
    
    Args:
        data: Full data dictionary
        event_idx: Event index
        
    Returns:
        dict: Minimal data subset for this event
    """
    subset = {}
    
    # List of all keys we need for 3D Lorentz plotting
    keys_needed = [
        'TrueX', 'TrueY', 'PixelX', 'PixelY', 'IsPixelHit',
        'NeighborhoodCharges', 'PixelSpacing',
        'Lorentz3DAmp', 'Lorentz3DCenterX', 'Lorentz3DCenterY',
        'Lorentz3DGammaX', 'Lorentz3DGammaY', 'Lorentz3DVertOffset',
        'Lorentz3DAmpErr', 'Lorentz3DCenterXErr', 'Lorentz3DCenterYErr',
        'Lorentz3DGammaXErr', 'Lorentz3DGammaYErr', 'Lorentz3DVertOffsetErr',
        'Lorentz3DChi2red', 'Lorentz3DDOF', 'Lorentz3DChargeErr'
    ]
    
    for key in keys_needed:
        if key in data:
            if key == 'PixelSpacing':
                # PixelSpacing is always scalar
                subset[key] = data[key]
            elif hasattr(data[key], '__len__') and len(data[key]) > event_idx:
                subset[key] = data[key][event_idx]
            else:
                subset[key] = data[key]  # Fallback for scalar values
    
    return subset

def create_3d_lorentz_fit_pdfs(data, output_dir="plots", max_events=None, n_workers=None):
    """
    Create PDF files with 3D Lorentz fits visualization using parallel processing.
    
    Args:
        data: Data dictionary from ROOT file
        output_dir: Output directory for PDF files
        max_events: Maximum number of events to process
        n_workers: Number of worker processes (default: CPU count)
    
    Returns:
        success_count: Number of success visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    n_events = len(data['TrueX'])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Creating 3D Lorentz fit visualizations for {n_events} events using {n_workers} workers")
    
    # Create output path
    pdf_path = os.path.join(output_dir, "lorentz_3d_fits.pdf")
    
    # Pre-filter events with success fits to avoid processing invalid events
    valid_events = []
    
    for event_idx in range(n_events):
        if data['Lorentz3DDOF'][event_idx] > 0:
            valid_events.append(event_idx)
    
    print(f"Found {len(valid_events)} valid 3D Lorentz fits")
    
    if not valid_events:
        print("No valid 3D Lorentz fits found!")
        return 0
    
    # Pre-calculate grid size to avoid repeated calculations
    sample_event = valid_events[0]
    sample_grid = np.array(data['NeighborhoodCharges'][sample_event])
    grid_size = int(np.sqrt(len(sample_grid))) if len(sample_grid) > 0 else 9
    print(f"Detected grid size: {grid_size}x{grid_size}")
    
    success_count = 0
    
    print("Creating 3D Lorentz fits PDF...")
    
    # Prepare arguments for parallel processing
    args_list = []
    for event_idx in valid_events:
        data_subset = _prepare_3d_lorentz_data_subset(data, event_idx)
        args_list.append((event_idx, data_subset))
    
    # Process in parallel with progress bar using threading
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_event = {executor.submit(_create_3d_lorentz_plot_worker, args): args[0] 
                         for args in args_list}
        
        # Collect results with progress bar
        results = {}
        with tqdm(total=len(args_list), desc="Generating 3D Lorentz plots") as pbar:
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
    with PdfPages(pdf_path) as pdf:
        with tqdm(total=len(valid_events), desc="Writing PDF") as pbar:
            for event_idx in valid_events:
                fig, success = results.get(event_idx, (None, False))
                if success and fig is not None:
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    success_count += 1
                pbar.update(1)
    
    # Cleanup
    del results
    del args_list
    gc.collect()
    
    print(f"3D Lorentz PDF generation completed!")
    print(f"  3D Lorentz fits visualized: {success_count}")
    print(f"  PDF saved to: {pdf_path}")
    
    return success_count

def export_single_event_svg(data, event_idx, output_path):
    """
    Export a single event as SVG file.
    
    Args:
        data: Data dictionary from ROOT file
        event_idx: Event index to export
        output_path: Output SVG file path
    
    Returns:
        success: Boolean indicating success
    """
    print(f"Exporting event {event_idx} as SVG...")
    
    # Check if the event has a valid fit
    if data['Lorentz3DDOF'][event_idx] <= 0:
        print(f"Event {event_idx} does not have a valid 3D Lorentz fit (DOF = {data['Lorentz3DDOF'][event_idx]})")
        return False
    
    # Create the plot using full dataset (don't subset for single event export)
    fig, success = create_3d_lorentz_plot(event_idx, data)
    
    if success and fig is not None:
        # Save as SVG
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Successly exported event {event_idx} to {output_path}")
        return True
    else:
        print(f"Failed to create plot for event {event_idx}")
        if fig is not None:
            plt.close(fig)
        return False

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
        
        # Show 3D Lorentz fitting branches
        lorentz_3d_branches = [b for b in branches if '3DLorentz' in b]
        print(f"\n3D Lorentz fitting branches ({len(lorentz_3d_branches)}):")
        for i, branch in enumerate(sorted(lorentz_3d_branches)):
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
        
        # Check for success 3D Lorentz fits
        if '3DLorentzDOF' in branches:
            fit_3d_dof = tree['3DLorentzDOF'].array(library="np")
            success_3d_fits = np.sum(fit_3d_dof > 0)
            print(f"  Success 3D Lorentz fits: {success_3d_fits} ({100*success_3d_fits/n_events:.1f}%)")
        
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Visualize 3D Lorentz fits from GEANT4 charge sharing simulation ROOT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file from GEANT4 simulation")
    parser.add_argument("-o", "--output", default="lorentz_3d_fits", 
                       help="Output directory for PDF files")
    parser.add_argument("-n", "--num_events", type=int, default=None,
                       help="Maximum number of events to process (default: all events)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker threads for parallel processing (default: CPU count)")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect ROOT file contents and exit")
    parser.add_argument("--svg_event", type=int, default=None,
                       help="Export single event as SVG (specify event index)")
    parser.add_argument("--svg_output", default="event_3d_lorentz_fit.svg",
                       help="Output filename for SVG export")
    
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
    
    # Check if we have 3D Lorentz fitting data
    n_events = len(data['TrueX'])
    if n_events == 0:
        print("No events found in the data!")
        return 1
    
    # Count success 3D Lorentz fits
    success_3d = np.sum(data['Lorentz3DDOF'] > 0)
    
    print(f"Found {success_3d} success 3D Lorentz fits")
    
    if success_3d == 0:
        print("No success 3D Lorentz fits found in the data!")
        return 1
    
    # SVG export mode
    if args.svg_event is not None:
        if args.svg_event >= n_events:
            print(f"Error: Event index {args.svg_event} is out of range (0-{n_events-1})")
            return 1
        
        success = export_single_event_svg(data, args.svg_event, args.svg_output)
        if success:
            print(f"SVG export completed successfully!")
            return 0
        else:
            print(f"SVG export failed!")
            return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Create PDF visualizations
    success_count = create_3d_lorentz_fit_pdfs(
        data, args.output, args.num_events, args.workers
    )
    
    print(f"\n3D Lorentz Visualization completed!")
    print(f"  3D Lorentz fits visualized: {success_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 