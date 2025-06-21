#!/usr/bin/env python3
"""
3D surface fitting visualization routine for Lorentzian and Power-Law Lorentzian fits in LGAD detectors.

This script creates comprehensive plots for:
1. 3D Lorentzian surface fits with color-coded contour plots
2. 3D Power-Law Lorentzian surface fits with color-coded contour plots  
3. 2D projections along vertical (Y=center) and horizontal (X=center) lines
4. Residual plots showing fit quality
5. Comparison plots between 3D Lorentzian and 3D Power-Law Lorentzian models
6. Best/worst fit analysis based on chi-squared values
7. High amplitude event analysis

The plots show fitted 3D surfaces overlaid on actual charge data from the neighborhood grid,
with contour plots for easy visualization of the charge distribution and fit quality.

AUTOMATIC UNCERTAINTY DETECTION:
The script automatically detects whether charge uncertainty branches are available in the ROOT file.
- If uncertainty branches are present and contain meaningful values, contour plots show uncertainty information
- If uncertainty branches are missing or contain only zeros, plots show data without uncertainty visualization

Special modes:
- Use --best_worst to plot the 5 best and 5 worst 3D fits based on chi-squared values
- Use --high_amplitudes N to plot the N events with highest 3D surface amplitudes

For large ROOT files, use --max_entries to limit the number of events processed for memory efficiency.
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os
import sys
import argparse
from pathlib import Path
from scipy.interpolate import griddata
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

def lorentzian_3d(x, y, amplitude, center_x, center_y, gamma_x, gamma_y, offset=0):
    """
    3D Lorentzian function for plotting fitted surfaces.
    
    Args:
        x, y: Independent variables (meshgrid)
        amplitude: Lorentzian amplitude
        center_x, center_y: Lorentzian centers
        gamma_x, gamma_y: Lorentzian gamma parameters (HWHM)
        offset: Baseline offset
    
    Returns:
        3D Lorentzian function values
    """
    # Robust parameter validation to match C++ implementation
    safe_gamma_x = max(abs(gamma_x), 1e-12)
    safe_gamma_y = max(abs(gamma_y), 1e-12)
    
    denominator = 1 + ((x - center_x) / safe_gamma_x)**2 + ((y - center_y) / safe_gamma_y)**2
    denominator = np.maximum(denominator, 1e-12)  # Prevent numerical issues
    
    return amplitude / denominator + offset

def power_lorentzian_3d(x, y, amplitude, center_x, center_y, gamma_x, gamma_y, beta, offset=0):
    """
    3D Power-Law Lorentzian function for plotting fitted surfaces.
    
    Args:
        x, y: Independent variables (meshgrid)
        amplitude: Power Lorentzian amplitude
        center_x, center_y: Power Lorentzian centers
        gamma_x, gamma_y: Power Lorentzian gamma parameters
        beta: Power exponent
        offset: Baseline offset
    
    Returns:
        3D Power-Law Lorentzian function values
    """
    # Robust parameter validation to match C++ implementation
    safe_gamma_x = max(abs(gamma_x), 1e-12)
    safe_gamma_y = max(abs(gamma_y), 1e-12)
    safe_beta = max(abs(beta), 0.1)
    
    denominator_base = 1 + ((x - center_x) / safe_gamma_x)**2 + ((y - center_y) / safe_gamma_y)**2
    denominator_base = np.maximum(denominator_base, 1e-12)  # Prevent numerical issues
    denominator = np.power(denominator_base, safe_beta)
    
    return amplitude / denominator + offset

def inspect_root_file_3d(root_file):
    """
    Inspect the ROOT file to see what 3D fitting branches are available.
    """
    print(f"Inspecting ROOT file for 3D fitting data: {root_file}")
    
    try:
        with uproot.open(root_file) as file:
            print(f"Available objects in file: {file.keys()}")
            
            # Try to find the Hits tree
            tree_name = None
            for key in file.keys():
                if 'Hits' in key or 'hits' in key:
                    tree_name = key
                    break
            
            if tree_name is None:
                for key in file.keys():
                    obj = file[key]
                    if hasattr(obj, 'keys'):
                        tree_name = key
                        break
            
            if tree_name is None:
                print("No suitable tree found in ROOT file!")
                return []
            
            print(f"Using tree: {tree_name}")
            tree = file[tree_name]
            branches = tree.keys()
            
            # Look for 3D fitting branches specifically
            branches_3d = [b for b in branches if '3D' in b]
            print(f"\nFound {len(branches_3d)} 3D fitting branches:")
            for i, branch in enumerate(sorted(branches_3d)):
                print(f"  {i+1:3d}: {branch}")
            
            return list(branches)
            
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")
        return []

def detect_3d_uncertainty_branches(data):
    """
    Detect whether 3D charge uncertainty branches are available in the data.
    """
    uncertainty_status = {
        '3d_lorentzian_uncertainty_available': False,
        '3d_power_lorentzian_uncertainty_available': False,
        'any_3d_uncertainties_available': False
    }
    
    # Check for 3D Lorentzian uncertainty branch
    if '3DLorentzianFitChargeUncertainty' in data:
        uncertainties = data['3DLorentzianFitChargeUncertainty']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['3d_lorentzian_uncertainty_available'] = True
    
    # Check for 3D Power-Law Lorentzian uncertainty branch
    if '3DPowerLorentzianFitChargeUncertainty' in data:
        uncertainties = data['3DPowerLorentzianFitChargeUncertainty']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['3d_power_lorentzian_uncertainty_available'] = True
    
    # Set overall flag
    uncertainty_status['any_3d_uncertainties_available'] = (
        uncertainty_status['3d_lorentzian_uncertainty_available'] or
        uncertainty_status['3d_power_lorentzian_uncertainty_available']
    )
    
    return uncertainty_status

def load_3d_fit_data(root_file, max_entries=None):
    """
    Load 3D fitting data from ROOT file with robust branch detection.
    """
    print(f"Loading 3D fit data from {root_file}...")
    if max_entries is not None:
        print(f"Limiting to first {max_entries} entries for large file handling")
    
    try:
        with uproot.open(root_file) as file:
            # Find the tree
            tree_name = None
            for key in file.keys():
                if 'Hits' in key or 'hits' in key:
                    tree_name = key
                    break
            
            if tree_name is None:
                for key in file.keys():
                    obj = file[key]
                    if hasattr(obj, 'keys'):
                        tree_name = key
                        break
            
            if tree_name is None:
                print("Error: No suitable tree found in ROOT file!")
                return None
                
            tree = file[tree_name]
            
            # Define expected 3D fitting branches with robust mapping
            branch_mapping = {
                # Basic hit information
                'TrueX': 'TrueX',
                'TrueY': 'TrueY',
                'PixelX': 'PixelX',
                'PixelY': 'PixelY',
                'IsPixelHit': 'IsPixelHit',
                
                # 3D Lorentzian fit results
                '3DLorentzianFitCenterX': '3DLorentzianFitCenterX',
                '3DLorentzianFitCenterY': '3DLorentzianFitCenterY',
                '3DLorentzianFitGammaX': '3DLorentzianFitGammaX',
                '3DLorentzianFitGammaY': '3DLorentzianFitGammaY',
                '3DLorentzianFitAmplitude': '3DLorentzianFitAmplitude',
                '3DLorentzianFitVerticalOffset': '3DLorentzianFitVerticalOffset',
                '3DLorentzianFitChi2red': '3DLorentzianFitChi2red',
                '3DLorentzianFitPp': '3DLorentzianFitPp',
                '3DLorentzianFitDOF': '3DLorentzianFitDOF',
                '3DLorentzianFitSuccessful': '3DLorentzianFitSuccessful',
                '3DLorentzianDeltaX': '3DLorentzianDeltaX',
                '3DLorentzianDeltaY': '3DLorentzianDeltaY',
                
                # 3D Power-Law Lorentzian fit results
                '3DPowerLorentzianFitCenterX': '3DPowerLorentzianFitCenterX',
                '3DPowerLorentzianFitCenterY': '3DPowerLorentzianFitCenterY',
                '3DPowerLorentzianFitGammaX': '3DPowerLorentzianFitGammaX',
                '3DPowerLorentzianFitGammaY': '3DPowerLorentzianFitGammaY',
                '3DPowerLorentzianFitBeta': '3DPowerLorentzianFitBeta',
                '3DPowerLorentzianFitAmplitude': '3DPowerLorentzianFitAmplitude',
                '3DPowerLorentzianFitVerticalOffset': '3DPowerLorentzianFitVerticalOffset',
                '3DPowerLorentzianFitChi2red': '3DPowerLorentzianFitChi2red',
                '3DPowerLorentzianFitPp': '3DPowerLorentzianFitPp',
                '3DPowerLorentzianFitDOF': '3DPowerLorentzianFitDOF',
                '3DPowerLorentzianFitSuccessful': '3DPowerLorentzianFitSuccessful',
                '3DPowerLorentzianDeltaX': '3DPowerLorentzianDeltaX',
                '3DPowerLorentzianDeltaY': '3DPowerLorentzianDeltaY',
                
                # Grid neighborhood data for actual charge distribution
                'GridNeighborhoodCharges': 'GridNeighborhoodCharges',
                'GridNeighborhoodDistances': 'GridNeighborhoodDistances',
                'GridNeighborhoodAngles': 'GridNeighborhoodAngles',
                
                # Charge uncertainties
                '3DLorentzianFitChargeUncertainty': '3DLorentzianFitChargeUncertainty',
                '3DPowerLorentzianFitChargeUncertainty': '3DPowerLorentzianFitChargeUncertainty'
            }
            
            # Load all available branches
            data = {}
            loaded_count = 0
            skipped_3d_count = 0
            
            for expected_name, actual_name in branch_mapping.items():
                if actual_name in tree.keys():
                    try:
                        if max_entries is not None:
                            data[expected_name] = tree[actual_name].array(library="np", entry_stop=max_entries)
                        else:
                            data[expected_name] = tree[actual_name].array(library="np")
                        loaded_count += 1
                        if loaded_count <= 10:
                            print(f"Loaded: {expected_name}")
                    except Exception as e:
                        print(f"Warning: Could not load {actual_name}: {e}")
                else:
                    if '3D' in expected_name:
                        skipped_3d_count += 1
                    else:
                        print(f"Warning: Branch {actual_name} not found for {expected_name}")
            
            if skipped_3d_count > 0:
                print(f"Note: Skipped {skipped_3d_count} 3D fitting branches (3D fitting may be disabled)")
            
            print(f"Successfully loaded {loaded_count} branches with {len(data['TrueX'])} events")
            
            # Detect uncertainty branch availability
            uncertainty_status = detect_3d_uncertainty_branches(data)
            data['_3d_uncertainty_status'] = uncertainty_status
            
            # Print uncertainty detection results
            if uncertainty_status['any_3d_uncertainties_available']:
                print("3D charge uncertainty branches detected:")
                if uncertainty_status['3d_lorentzian_uncertainty_available']:
                    print("  ✓ 3D Lorentzian uncertainties available")
                if uncertainty_status['3d_power_lorentzian_uncertainty_available']:
                    print("  ✓ 3D Power-Law Lorentzian uncertainties available")
            else:
                print("No 3D charge uncertainty branches detected")
            
            # Apply filtering for non-pixel hits if available
            if 'IsPixelHit' in data:
                is_non_pixel = ~data['IsPixelHit']
                print(f"Non-pixel events: {np.sum(is_non_pixel)}")
                
                # Filter all data to non-pixel events
                filtered_data = {}
                n_events = len(data['TrueX'])
                for key, values in data.items():
                    if key == '_3d_uncertainty_status':
                        filtered_data[key] = values
                    elif hasattr(values, '__len__') and len(values) == n_events:
                        filtered_data[key] = values[is_non_pixel]
                    else:
                        filtered_data[key] = values
                
                if np.sum(is_non_pixel) > 0:
                    return filtered_data
                else:
                    print("Warning: No non-pixel events found - returning all data")
                    return data
            else:
                print("Warning: IsPixelHit branch not found - returning all data")
                return data
            
    except Exception as e:
        print(f"Error loading ROOT file: {e}")
        return None

def extract_3d_neighborhood_data(event_idx, data, neighborhood_radius=4):
    """
    Extract charge data for the full 3D neighborhood grid.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (x_positions, y_positions, charge_values) for all pixels with charge > 0
    """
    
    if 'GridNeighborhoodCharges' in data and event_idx < len(data['GridNeighborhoodCharges']):
        try:
            # Extract raw grid data
            grid_charges = data['GridNeighborhoodCharges'][event_idx]
            
            # Get nearest pixel position for reference
            center_x = data['PixelX'][event_idx]
            center_y = data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract all pixels with charge > 0
            x_positions = []
            y_positions = []
            charge_values = []
            
            for i in range(grid_size):  # X direction (columns)
                for j in range(grid_size):  # Y direction (rows)
                    grid_idx = i * grid_size + j  # Column-major indexing
                    if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                        # Calculate actual position for this pixel
                        offset_i = i - neighborhood_radius  # -4 to +4 for X
                        offset_j = j - neighborhood_radius  # -4 to +4 for Y
                        x_pos = center_x + offset_i * pixel_spacing
                        y_pos = center_y + offset_j * pixel_spacing
                        
                        x_positions.append(x_pos)
                        y_positions.append(y_pos)
                        charge_values.append(grid_charges[grid_idx])
            
            x_positions = np.array(x_positions)
            y_positions = np.array(y_positions)
            charge_values = np.array(charge_values)
            
            print(f"Event {event_idx}: Extracted {len(x_positions)} total grid points with charge > 0")
            
            return x_positions, y_positions, charge_values
            
        except Exception as e:
            print(f"Warning: Could not extract 3D grid data for event {event_idx}: {e}")
            return np.array([]), np.array([]), np.array([])
    else:
        print(f"Warning: GridNeighborhoodCharges not available for event {event_idx}")
        return np.array([]), np.array([]), np.array([])

def create_3d_lorentzian_plot(event_idx, data, output_dir="plots"):
    """
    Create comprehensive 3D Lorentzian surface plot with contour visualization and 2D projections.
    """
    try:
        # Check if 3D Lorentzian data is available
        if '3DLorentzianFitCenterX' not in data:
            return f"Event {event_idx}: 3D Lorentzian fit data not available"
        
        # Extract 3D neighborhood data
        x_positions, y_positions, charge_values = extract_3d_neighborhood_data(event_idx, data)
        
        if len(x_positions) < 5:
            return f"Event {event_idx}: Not enough data points for 3D plotting"
        
        # Get 3D Lorentzian fit parameters
        center_x = data['3DLorentzianFitCenterX'][event_idx]
        center_y = data['3DLorentzianFitCenterY'][event_idx]
        gamma_x = data['3DLorentzianFitGammaX'][event_idx]
        gamma_y = data['3DLorentzianFitGammaY'][event_idx]
        amplitude = data['3DLorentzianFitAmplitude'][event_idx]
        offset = data.get('3DLorentzianFitVerticalOffset', [0])[event_idx] if '3DLorentzianFitVerticalOffset' in data else 0
        chi2red = data['3DLorentzianFitChi2red'][event_idx]
        dof = data.get('3DLorentzianFitDOF', [1])[event_idx] if '3DLorentzianFitDOF' in data else 1
        fit_successful = data.get('3DLorentzianFitSuccessful', [True])[event_idx] if '3DLorentzianFitSuccessful' in data else True
        
        # True and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        
        # Calculate deltas
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        delta_fit_x = center_x - true_x
        delta_fit_y = center_y - true_y
        
        # Create output directory
        lorentzian_3d_dir = os.path.join(output_dir, "3d_lorentzian")
        os.makedirs(lorentzian_3d_dir, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, hspace=0.35, wspace=0.35)
        
        # Create grid for surface plotting
        x_range = np.linspace(x_positions.min() - 0.2, x_positions.max() + 0.2, 50)
        y_range = np.linspace(y_positions.min() - 0.2, y_positions.max() + 0.2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate fitted surface
        Z_fit = lorentzian_3d(X, Y, amplitude, center_x, center_y, gamma_x, gamma_y, offset)
        
        # Interpolate actual data to grid for contour plotting
        Z_data = griddata((x_positions, y_positions), charge_values, (X, Y), method='cubic', fill_value=0)
        
        # 1. Main contour plot - Fitted surface
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = ax1.contourf(X, Y, Z_fit, levels=20, cmap='viridis', alpha=0.8)
        ax1.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='viridis', 
                   edgecolors='black', linewidth=0.5, label='Data points')
        ax1.axvline(true_x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax1.axhline(true_y, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax1.axvline(center_x, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center X')
        ax1.axhline(center_y, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center Y')
        ax1.set_xlabel('X Position [mm]')
        ax1.set_ylabel('Y Position [mm]')
        ax1.set_title('3D Lorentzian Fit Surface')
        plt.colorbar(contour1, ax=ax1, label='Charge [C]')
        ax1.legend(fontsize=8)
        
        # 2. Data contour plot
        ax2 = fig.add_subplot(gs[0, 1])
        contour2 = ax2.contourf(X, Y, Z_data, levels=20, cmap='viridis', alpha=0.8)
        ax2.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='viridis', 
                   edgecolors='black', linewidth=0.5, label='Data points')
        ax2.axvline(true_x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax2.axhline(true_y, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax2.set_xlabel('X Position [mm]')
        ax2.set_ylabel('Y Position [mm]')
        ax2.set_title('Actual Data Distribution')
        plt.colorbar(contour2, ax=ax2, label='Charge [C]')
        ax2.legend(fontsize=8)
        
        # 3. Residual plot
        ax3 = fig.add_subplot(gs[0, 2])
        Z_residual = Z_data - Z_fit
        # Mask out areas where we don't have data
        Z_residual_masked = np.ma.masked_where(np.abs(Z_data) < 1e-20, Z_residual)
        contour3 = ax3.contourf(X, Y, Z_residual_masked, levels=20, cmap='RdBu_r', alpha=0.8)
        ax3.scatter(x_positions, y_positions, c='black', s=20, alpha=0.7, label='Data points')
        ax3.axvline(true_x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax3.axhline(true_y, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax3.axvline(center_x, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center X')
        ax3.axhline(center_y, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center Y')
        ax3.set_xlabel('X Position [mm]')
        ax3.set_ylabel('Y Position [mm]')
        ax3.set_title('Residuals (Data - Fit)')
        plt.colorbar(contour3, ax=ax3, label='Charge Residual [C]')
        ax3.legend(fontsize=8)
        
        # 4. X projection (along Y=center_y) - Vertical zero line
        ax4 = fig.add_subplot(gs[1, 0])
        y_center_idx = np.argmin(np.abs(y_range - center_y))
        x_projection_fit = Z_fit[y_center_idx, :]
        
        # Get actual data points near center_y
        y_tolerance = 0.25  # mm tolerance for projection
        mask_y = np.abs(y_positions - center_y) < y_tolerance
        if np.any(mask_y):
            x_data_proj = x_positions[mask_y]
            charge_data_proj = charge_values[mask_y]
            # Sort by x position for clean plotting
            sort_idx = np.argsort(x_data_proj)
            x_data_proj = x_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax4.plot(x_data_proj, charge_data_proj, 'ko', markersize=6, label=f'Data (|Y-{center_y:.2f}|<{y_tolerance})')
        
        ax4.plot(x_range, x_projection_fit, 'r-', linewidth=2, label='3D Lorentzian Fit')
        ax4.axvline(true_x, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax4.axvline(center_x, color='red', linestyle=':', linewidth=2, alpha=0.8, label='Fit Center X')
        ax4.set_xlabel('X Position [mm]')
        ax4.set_ylabel('Charge [C]')
        ax4.set_title('X Projection (Vertical Zero Line)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Y projection (along X=center_x) - Horizontal zero line
        ax5 = fig.add_subplot(gs[1, 1])
        x_center_idx = np.argmin(np.abs(x_range - center_x))
        y_projection_fit = Z_fit[:, x_center_idx]
        
        # Get actual data points near center_x
        x_tolerance = 0.25  # mm tolerance for projection
        mask_x = np.abs(x_positions - center_x) < x_tolerance
        if np.any(mask_x):
            y_data_proj = y_positions[mask_x]
            charge_data_proj = charge_values[mask_x]
            # Sort by y position for clean plotting
            sort_idx = np.argsort(y_data_proj)
            y_data_proj = y_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax5.plot(y_data_proj, charge_data_proj, 'ko', markersize=6, label=f'Data (|X-{center_x:.2f}|<{x_tolerance})')
        
        ax5.plot(y_range, y_projection_fit, 'r-', linewidth=2, label='3D Lorentzian Fit')
        ax5.axvline(true_y, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax5.axvline(center_y, color='red', linestyle=':', linewidth=2, alpha=0.8, label='Fit Center Y')
        ax5.set_xlabel('Y Position [mm]')
        ax5.set_ylabel('Charge [C]')
        ax5.set_title('Y Projection (Horizontal Zero Line)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Fit parameters and statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        stats_text = f"""3D Lorentzian Fit Parameters:

Center X: {center_x:.4f} mm
Center Y: {center_y:.4f} mm
Gamma X: {gamma_x:.4f} mm
Gamma Y: {gamma_y:.4f} mm
Amplitude: {amplitude:.3e} C
Offset: {offset:.3e} C

Fit Statistics:
χ²/ν: {chi2red:.3f}
DOF: {dof}
Successful: {fit_successful}

Position Differences:
Δ Pixel X: {delta_pixel_x:.4f} mm
Δ Pixel Y: {delta_pixel_y:.4f} mm
Δ Fit X: {delta_fit_x:.4f} mm  
Δ Fit Y: {delta_fit_y:.4f} mm

True Position: ({true_x:.3f}, {true_y:.3f})
Pixel Position: ({pixel_x:.3f}, {pixel_y:.3f})
Fit Position: ({center_x:.3f}, {center_y:.3f})

Data Points: {len(x_positions)}"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 7. 3D surface plot
        ax7 = fig.add_subplot(gs[2, :], projection='3d')
        
        # Plot fitted surface with transparency
        surf = ax7.plot_surface(X, Y, Z_fit, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
        
        # Plot data points with color coding
        scatter = ax7.scatter(x_positions, y_positions, charge_values, c=charge_values, 
                            cmap='viridis', s=50, alpha=1.0, edgecolors='black', linewidth=0.5)
        
        # Add vertical lines for true and fit positions
        z_min = 0
        z_max = np.max([np.max(charge_values), np.max(Z_fit)]) * 1.1
        ax7.plot([true_x, true_x], [true_y, true_y], [z_min, z_max], 'r--', 
                linewidth=3, alpha=0.8, label='True Position')
        ax7.plot([center_x, center_x], [center_y, center_y], [z_min, z_max], 'w:', 
                linewidth=3, alpha=0.9, label='Fit Center')
        
        ax7.set_xlabel('X Position [mm]')
        ax7.set_ylabel('Y Position [mm]')
        ax7.set_zlabel('Charge [C]')
        ax7.set_title('3D Lorentzian Surface Fit')
        
        # Add colorbar for 3D plot
        cbar = plt.colorbar(surf, ax=ax7, shrink=0.8, label='Charge [C]', pad=0.1)
        ax7.legend(loc='upper left')
        
        plt.suptitle(f'Event {event_idx}: 3D Lorentzian Surface Fitting Analysis\n' +
                    f'χ²/ν = {chi2red:.3f}, Center = ({center_x:.3f}, {center_y:.3f}), ' +
                    f'Δ = ({delta_fit_x:.3f}, {delta_fit_y:.3f})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(lorentzian_3d_dir, f'event_{event_idx:04d}_3d_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: 3D Lorentzian plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating 3D Lorentzian plot - {e}"

def create_3d_power_lorentzian_plot(event_idx, data, output_dir="plots"):
    """
    Create comprehensive 3D Power-Law Lorentzian surface plot with contour visualization and 2D projections.
    """
    try:
        # Check if 3D Power-Law Lorentzian data is available
        if '3DPowerLorentzianFitCenterX' not in data:
            return f"Event {event_idx}: 3D Power-Law Lorentzian fit data not available"
        
        # Extract 3D neighborhood data
        x_positions, y_positions, charge_values = extract_3d_neighborhood_data(event_idx, data)
        
        if len(x_positions) < 5:
            return f"Event {event_idx}: Not enough data points for 3D plotting"
        
        # Get 3D Power-Law Lorentzian fit parameters
        center_x = data['3DPowerLorentzianFitCenterX'][event_idx]
        center_y = data['3DPowerLorentzianFitCenterY'][event_idx]
        gamma_x = data['3DPowerLorentzianFitGammaX'][event_idx]
        gamma_y = data['3DPowerLorentzianFitGammaY'][event_idx]
        beta = data['3DPowerLorentzianFitBeta'][event_idx]
        amplitude = data['3DPowerLorentzianFitAmplitude'][event_idx]
        offset = data.get('3DPowerLorentzianFitVerticalOffset', [0])[event_idx] if '3DPowerLorentzianFitVerticalOffset' in data else 0
        chi2red = data['3DPowerLorentzianFitChi2red'][event_idx]
        dof = data.get('3DPowerLorentzianFitDOF', [1])[event_idx] if '3DPowerLorentzianFitDOF' in data else 1
        fit_successful = data.get('3DPowerLorentzianFitSuccessful', [True])[event_idx] if '3DPowerLorentzianFitSuccessful' in data else True
        
        # True and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        
        # Calculate deltas
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        delta_fit_x = center_x - true_x
        delta_fit_y = center_y - true_y
        
        # Create output directory
        power_lorentzian_3d_dir = os.path.join(output_dir, "3d_power_lorentzian")
        os.makedirs(power_lorentzian_3d_dir, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, hspace=0.35, wspace=0.35)
        
        # Create grid for surface plotting
        x_range = np.linspace(x_positions.min() - 0.2, x_positions.max() + 0.2, 50)
        y_range = np.linspace(y_positions.min() - 0.2, y_positions.max() + 0.2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate fitted surface
        Z_fit = power_lorentzian_3d(X, Y, amplitude, center_x, center_y, gamma_x, gamma_y, beta, offset)
        
        # Interpolate actual data to grid for contour plotting
        Z_data = griddata((x_positions, y_positions), charge_values, (X, Y), method='cubic', fill_value=0)
        
        # 1. Main contour plot - Fitted surface
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = ax1.contourf(X, Y, Z_fit, levels=20, cmap='plasma', alpha=0.8)
        ax1.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='plasma', 
                   edgecolors='black', linewidth=0.5, label='Data points')
        ax1.axvline(true_x, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax1.axhline(true_y, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax1.axvline(center_x, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center X')
        ax1.axhline(center_y, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center Y')
        ax1.set_xlabel('X Position [mm]')
        ax1.set_ylabel('Y Position [mm]')
        ax1.set_title('3D Power-Law Lorentzian Fit Surface')
        plt.colorbar(contour1, ax=ax1, label='Charge [C]')
        ax1.legend(fontsize=8)
        
        # 2. Data contour plot
        ax2 = fig.add_subplot(gs[0, 1])
        contour2 = ax2.contourf(X, Y, Z_data, levels=20, cmap='plasma', alpha=0.8)
        ax2.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='plasma', 
                   edgecolors='black', linewidth=0.5, label='Data points')
        ax2.axvline(true_x, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax2.axhline(true_y, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax2.set_xlabel('X Position [mm]')
        ax2.set_ylabel('Y Position [mm]')
        ax2.set_title('Actual Data Distribution')
        plt.colorbar(contour2, ax=ax2, label='Charge [C]')
        ax2.legend(fontsize=8)
        
        # 3. Residual plot
        ax3 = fig.add_subplot(gs[0, 2])
        Z_residual = Z_data - Z_fit
        # Mask out areas where we don't have data
        Z_residual_masked = np.ma.masked_where(np.abs(Z_data) < 1e-20, Z_residual)
        contour3 = ax3.contourf(X, Y, Z_residual_masked, levels=20, cmap='RdBu_r', alpha=0.8)
        ax3.scatter(x_positions, y_positions, c='black', s=20, alpha=0.7, label='Data points')
        ax3.axvline(true_x, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax3.axhline(true_y, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax3.axvline(center_x, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center X')
        ax3.axhline(center_y, color='white', linestyle=':', linewidth=2, alpha=0.9, label='Fit Center Y')
        ax3.set_xlabel('X Position [mm]')
        ax3.set_ylabel('Y Position [mm]')
        ax3.set_title('Residuals (Data - Fit)')
        plt.colorbar(contour3, ax=ax3, label='Charge Residual [C]')
        ax3.legend(fontsize=8)
        
        # 4. X projection (along Y=center_y) - Vertical zero line
        ax4 = fig.add_subplot(gs[1, 0])
        y_center_idx = np.argmin(np.abs(y_range - center_y))
        x_projection_fit = Z_fit[y_center_idx, :]
        
        # Get actual data points near center_y
        y_tolerance = 0.25  # mm tolerance for projection
        mask_y = np.abs(y_positions - center_y) < y_tolerance
        if np.any(mask_y):
            x_data_proj = x_positions[mask_y]
            charge_data_proj = charge_values[mask_y]
            # Sort by x position for clean plotting
            sort_idx = np.argsort(x_data_proj)
            x_data_proj = x_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax4.plot(x_data_proj, charge_data_proj, 'ko', markersize=6, label=f'Data (|Y-{center_y:.2f}|<{y_tolerance})')
        
        ax4.plot(x_range, x_projection_fit, 'm-', linewidth=2, label='3D Power-Law Lorentzian Fit')
        ax4.axvline(true_x, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax4.axvline(center_x, color='magenta', linestyle=':', linewidth=2, alpha=0.8, label='Fit Center X')
        ax4.set_xlabel('X Position [mm]')
        ax4.set_ylabel('Charge [C]')
        ax4.set_title('X Projection (Vertical Zero Line)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Y projection (along X=center_x) - Horizontal zero line
        ax5 = fig.add_subplot(gs[1, 1])
        x_center_idx = np.argmin(np.abs(x_range - center_x))
        y_projection_fit = Z_fit[:, x_center_idx]
        
        # Get actual data points near center_x
        x_tolerance = 0.25  # mm tolerance for projection
        mask_x = np.abs(x_positions - center_x) < x_tolerance
        if np.any(mask_x):
            y_data_proj = y_positions[mask_x]
            charge_data_proj = charge_values[mask_x]
            # Sort by y position for clean plotting
            sort_idx = np.argsort(y_data_proj)
            y_data_proj = y_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax5.plot(y_data_proj, charge_data_proj, 'ko', markersize=6, label=f'Data (|X-{center_x:.2f}|<{x_tolerance})')
        
        ax5.plot(y_range, y_projection_fit, 'm-', linewidth=2, label='3D Power-Law Lorentzian Fit')
        ax5.axvline(true_y, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax5.axvline(center_y, color='magenta', linestyle=':', linewidth=2, alpha=0.8, label='Fit Center Y')
        ax5.set_xlabel('Y Position [mm]')
        ax5.set_ylabel('Charge [C]')
        ax5.set_title('Y Projection (Horizontal Zero Line)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Fit parameters and statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        stats_text = f"""3D Power-Law Lorentzian Parameters:

Center X: {center_x:.4f} mm
Center Y: {center_y:.4f} mm
Gamma X: {gamma_x:.4f} mm
Gamma Y: {gamma_y:.4f} mm
Beta: {beta:.4f}
Amplitude: {amplitude:.3e} C
Offset: {offset:.3e} C

Fit Statistics:
χ²/ν: {chi2red:.3f}
DOF: {dof}
Successful: {fit_successful}

Position Differences:
Δ Pixel X: {delta_pixel_x:.4f} mm
Δ Pixel Y: {delta_pixel_y:.4f} mm
Δ Fit X: {delta_fit_x:.4f} mm  
Δ Fit Y: {delta_fit_y:.4f} mm

True Position: ({true_x:.3f}, {true_y:.3f})
Pixel Position: ({pixel_x:.3f}, {pixel_y:.3f})
Fit Position: ({center_x:.3f}, {center_y:.3f})

Data Points: {len(x_positions)}"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 7. 3D surface plot
        ax7 = fig.add_subplot(gs[2, :], projection='3d')
        
        # Plot fitted surface with transparency
        surf = ax7.plot_surface(X, Y, Z_fit, cmap='plasma', alpha=0.7, linewidth=0, antialiased=True)
        
        # Plot data points with color coding
        scatter = ax7.scatter(x_positions, y_positions, charge_values, c=charge_values, 
                            cmap='plasma', s=50, alpha=1.0, edgecolors='black', linewidth=0.5)
        
        # Add vertical lines for true and fit positions
        z_min = 0
        z_max = np.max([np.max(charge_values), np.max(Z_fit)]) * 1.1
        ax7.plot([true_x, true_x], [true_y, true_y], [z_min, z_max], 'c--', 
                linewidth=3, alpha=0.8, label='True Position')
        ax7.plot([center_x, center_x], [center_y, center_y], [z_min, z_max], 'w:', 
                linewidth=3, alpha=0.9, label='Fit Center')
        
        ax7.set_xlabel('X Position [mm]')
        ax7.set_ylabel('Y Position [mm]')
        ax7.set_zlabel('Charge [C]')
        ax7.set_title('3D Power-Law Lorentzian Surface Fit')
        
        # Add colorbar for 3D plot
        cbar = plt.colorbar(surf, ax=ax7, shrink=0.8, label='Charge [C]', pad=0.1)
        ax7.legend(loc='upper left')
        
        plt.suptitle(f'Event {event_idx}: 3D Power-Law Lorentzian Surface Fitting Analysis\n' +
                    f'χ²/ν = {chi2red:.3f}, Beta = {beta:.3f}, Center = ({center_x:.3f}, {center_y:.3f}), ' +
                    f'Δ = ({delta_fit_x:.3f}, {delta_fit_y:.3f})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(power_lorentzian_3d_dir, f'event_{event_idx:04d}_3d_power_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: 3D Power-Law Lorentzian plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating 3D Power-Law Lorentzian plot - {e}"

def calculate_3d_fit_quality_metric(data, event_idx):
    """
    Calculate an overall 3D fit quality metric for an event.
    Lower values indicate better fits.
    """
    try:
        # Get chi-squared values for 3D fits
        lorentz_3d_chi2 = data.get('3DLorentzianFitChi2red', [float('inf')])[event_idx] if '3DLorentzianFitChi2red' in data else float('inf')
        power_3d_chi2 = data.get('3DPowerLorentzianFitChi2red', [float('inf')])[event_idx] if '3DPowerLorentzianFitChi2red' in data else float('inf')
        
        # Use the best (lowest) chi-squared value as the metric
        chi2_values = []
        for chi2 in [lorentz_3d_chi2, power_3d_chi2]:
            if np.isfinite(chi2) and chi2 > 0:
                chi2_values.append(chi2)
        
        if not chi2_values:
            return float('inf')
        
        return min(chi2_values)
        
    except Exception as e:
        print(f"Warning: Could not calculate 3D fit quality for event {event_idx}: {e}")
        return float('inf')

def find_best_worst_3d_fits(data, n_best=5, n_worst=5):
    """
    Find the events with the best and worst 3D fit quality.
    """
    print("Calculating 3D fit quality metrics for all events...")
    
    n_events = len(data['TrueX'])
    fit_metrics = []
    
    for i in range(n_events):
        metric = calculate_3d_fit_quality_metric(data, i)
        fit_metrics.append((i, metric))
    
    # Sort by fit quality (lower is better)
    fit_metrics.sort(key=lambda x: x[1])
    
    # Remove events with infinite chi2 (failed fits)
    valid_fits = [(idx, metric) for idx, metric in fit_metrics if np.isfinite(metric)]
    
    if len(valid_fits) == 0:
        print("Warning: No valid 3D fits found!")
        return [], [], []
    
    print(f"Found {len(valid_fits)} events with valid 3D fits out of {n_events} total events")
    
    # Get best fits (lowest chi2)
    best_fits = valid_fits[:n_best]
    best_indices = [idx for idx, metric in best_fits]
    
    # Get worst fits (highest chi2, but still finite)
    worst_fits = valid_fits[-n_worst:]
    worst_indices = [idx for idx, metric in worst_fits]
    
    print(f"Best 3D fits (lowest χ²):")
    for i, (idx, metric) in enumerate(best_fits):
        print(f"  {i+1}. Event {idx}: χ² = {metric:.3f}")
    
    print(f"Worst 3D fits (highest χ²):")
    for i, (idx, metric) in enumerate(worst_fits):
        print(f"  {i+1}. Event {idx}: χ² = {metric:.3f}")
    
    return best_indices, worst_indices, fit_metrics

def find_high_amplitude_3d_events(data, n_events=10):
    """
    Find events with the highest 3D surface amplitudes.
    """
    print("Finding events with highest 3D surface amplitudes...")
    
    n_total = len(data['TrueX'])
    amplitude_metrics = []
    
    for i in range(n_total):
        try:
            # Get 3D amplitudes
            lorentz_3d_amp = data.get('3DLorentzianFitAmplitude', [0])[i] if '3DLorentzianFitAmplitude' in data and i < len(data['3DLorentzianFitAmplitude']) else 0
            power_3d_amp = data.get('3DPowerLorentzianFitAmplitude', [0])[i] if '3DPowerLorentzianFitAmplitude' in data and i < len(data['3DPowerLorentzianFitAmplitude']) else 0
            
            # Use the maximum amplitude across all 3D fits
            max_amplitude = max(abs(lorentz_3d_amp), abs(power_3d_amp))
            
            # Also get chi2 values for quality assessment
            lorentz_3d_chi2 = data.get('3DLorentzianFitChi2red', [float('inf')])[i] if '3DLorentzianFitChi2red' in data else float('inf')
            power_3d_chi2 = data.get('3DPowerLorentzianFitChi2red', [float('inf')])[i] if '3DPowerLorentzianFitChi2red' in data else float('inf')
            avg_chi2 = np.mean([chi2 for chi2 in [lorentz_3d_chi2, power_3d_chi2] if np.isfinite(chi2)])
            if not np.isfinite(avg_chi2):
                avg_chi2 = float('inf')
            
            amplitude_metrics.append((i, max_amplitude, avg_chi2, lorentz_3d_amp, power_3d_amp))
            
        except Exception as e:
            print(f"Warning: Could not extract 3D amplitude data for event {i}: {e}")
            amplitude_metrics.append((i, 0, float('inf'), 0, 0))
    
    # Sort by amplitude (highest first)
    amplitude_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Get events with finite amplitudes
    valid_amps = [(idx, amp, chi2, l_amp, p_amp) for idx, amp, chi2, l_amp, p_amp in amplitude_metrics if amp > 0 and np.isfinite(amp)]
    
    if len(valid_amps) == 0:
        print("Warning: No events with valid 3D amplitudes found!")
        return [], []
    
    print(f"Found {len(valid_amps)} events with valid 3D amplitudes out of {n_total} total events")
    
    # Get highest amplitude events
    high_amp_events = valid_amps[:n_events]
    high_amp_indices = [idx for idx, amp, chi2, l_amp, p_amp in high_amp_events]
    
    print(f"Highest 3D amplitude events:")
    for i, (idx, amp, chi2, l_amp, p_amp) in enumerate(high_amp_events):
        print(f"  {i+1}. Event {idx}: Max Amp = {amp:.2e} C (χ² = {chi2:.3f})")
        print(f"      3D Lorentz: {l_amp:.2e} | 3D Power: {p_amp:.2e}")
    
    return high_amp_indices, amplitude_metrics

def create_3d_comparison_plot(event_idx, data, output_dir="plots"):
    """
    Create comparison plot between 3D Lorentzian and 3D Power-Law Lorentzian fits.
    """
    try:
        # Check if both 3D fitting models are available
        has_3d_lorentzian = '3DLorentzianFitCenterX' in data
        has_3d_power_lorentzian = '3DPowerLorentzianFitCenterX' in data
        
        if not has_3d_lorentzian and not has_3d_power_lorentzian:
            return f"Event {event_idx}: No 3D fitting data available"
        elif not (has_3d_lorentzian and has_3d_power_lorentzian):
            return f"Event {event_idx}: Need both 3D Lorentzian and Power-Law Lorentzian fits for comparison"
        
        # Extract 3D neighborhood data
        x_positions, y_positions, charge_values = extract_3d_neighborhood_data(event_idx, data)
        
        if len(x_positions) < 5:
            return f"Event {event_idx}: Not enough data points for 3D comparison plotting"
        
        # Get fit parameters for both models
        # 3D Lorentzian
        l_center_x = data['3DLorentzianFitCenterX'][event_idx]
        l_center_y = data['3DLorentzianFitCenterY'][event_idx]
        l_gamma_x = data['3DLorentzianFitGammaX'][event_idx]
        l_gamma_y = data['3DLorentzianFitGammaY'][event_idx]
        l_amplitude = data['3DLorentzianFitAmplitude'][event_idx]
        l_offset = data.get('3DLorentzianFitVerticalOffset', [0])[event_idx] if '3DLorentzianFitVerticalOffset' in data else 0
        l_chi2red = data['3DLorentzianFitChi2red'][event_idx]
        
        # 3D Power-Law Lorentzian
        p_center_x = data['3DPowerLorentzianFitCenterX'][event_idx]
        p_center_y = data['3DPowerLorentzianFitCenterY'][event_idx]
        p_gamma_x = data['3DPowerLorentzianFitGammaX'][event_idx]
        p_gamma_y = data['3DPowerLorentzianFitGammaY'][event_idx]
        p_beta = data['3DPowerLorentzianFitBeta'][event_idx]
        p_amplitude = data['3DPowerLorentzianFitAmplitude'][event_idx]
        p_offset = data.get('3DPowerLorentzianFitVerticalOffset', [0])[event_idx] if '3DPowerLorentzianFitVerticalOffset' in data else 0
        p_chi2red = data['3DPowerLorentzianFitChi2red'][event_idx]
        
        # Validate parameters and warn about unusual values
        max_charge = np.max(charge_values)
        unusual_params = []
        
        if abs(l_gamma_x) < 1e-6 or abs(l_gamma_y) < 1e-6:
            unusual_params.append(f"3D Lorentzian: very small gamma ({l_gamma_x:.2e}, {l_gamma_y:.2e})")
        if abs(l_amplitude) > 100 * max_charge:
            unusual_params.append(f"3D Lorentzian: amplitude {l_amplitude:.2e} >> max_charge {max_charge:.2e}")
        
        if abs(p_gamma_x) < 1e-6 or abs(p_gamma_y) < 1e-6:
            unusual_params.append(f"3D Power: very small gamma ({p_gamma_x:.2e}, {p_gamma_y:.2e})")
        if abs(p_amplitude) > 100 * max_charge:
            unusual_params.append(f"3D Power: amplitude {p_amplitude:.2e} >> max_charge {max_charge:.2e}")
        if p_beta < 0.1 or p_beta > 10:
            unusual_params.append(f"3D Power: extreme beta = {p_beta:.3f}")
        
        if unusual_params:
            print(f"Event {event_idx} unusual fit parameters detected:")
            for param in unusual_params:
                print(f"  - {param}")
        
        # True positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        # Create output directory
        comparison_3d_dir = os.path.join(output_dir, "3d_comparison")
        os.makedirs(comparison_3d_dir, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # Create grid for surface plotting
        x_range = np.linspace(x_positions.min() - 0.2, x_positions.max() + 0.2, 50)
        y_range = np.linspace(y_positions.min() - 0.2, y_positions.max() + 0.2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate fitted surfaces
        Z_lorentz = lorentzian_3d(X, Y, l_amplitude, l_center_x, l_center_y, l_gamma_x, l_gamma_y, l_offset)
        Z_power = power_lorentzian_3d(X, Y, p_amplitude, p_center_x, p_center_y, p_gamma_x, p_gamma_y, p_beta, p_offset)
        Z_data = griddata((x_positions, y_positions), charge_values, (X, Y), method='cubic', fill_value=0)
        
        # 1. Data contour plot
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = ax1.contourf(X, Y, Z_data, levels=20, cmap='viridis', alpha=0.8)
        ax1.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='viridis', 
                   edgecolors='black', linewidth=0.5, label='Data')
        ax1.axvline(true_x, color='white', linestyle='--', linewidth=2, alpha=0.9, label='True Position')
        ax1.axhline(true_y, color='white', linestyle='--', linewidth=2, alpha=0.9)
        ax1.set_xlabel('X Position [mm]')
        ax1.set_ylabel('Y Position [mm]')
        ax1.set_title('Actual Data Distribution')
        plt.colorbar(contour1, ax=ax1, label='Charge [C]')
        ax1.legend(fontsize=8)
        
        # 2. 3D Lorentzian fit
        ax2 = fig.add_subplot(gs[0, 1])
        contour2 = ax2.contourf(X, Y, Z_lorentz, levels=20, cmap='viridis', alpha=0.8)
        ax2.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='viridis', 
                   edgecolors='black', linewidth=0.5, alpha=0.7)
        ax2.axvline(l_center_x, color='red', linestyle=':', linewidth=2, alpha=0.9, label=f'3D Lorentz Center')
        ax2.axhline(l_center_y, color='red', linestyle=':', linewidth=2, alpha=0.9)
        ax2.axvline(true_x, color='white', linestyle='--', linewidth=2, alpha=0.9, label='True Position')
        ax2.axhline(true_y, color='white', linestyle='--', linewidth=2, alpha=0.9)
        ax2.set_xlabel('X Position [mm]')
        ax2.set_ylabel('Y Position [mm]')
        ax2.set_title(f'3D Lorentzian Fit (χ²/ν = {l_chi2red:.3f})')
        plt.colorbar(contour2, ax=ax2, label='Charge [C]')
        ax2.legend(fontsize=8)
        
        # 3. 3D Power-Law Lorentzian fit
        ax3 = fig.add_subplot(gs[0, 2])
        contour3 = ax3.contourf(X, Y, Z_power, levels=20, cmap='plasma', alpha=0.8)
        ax3.scatter(x_positions, y_positions, c=charge_values, s=50, cmap='plasma', 
                   edgecolors='black', linewidth=0.5, alpha=0.7)
        ax3.axvline(p_center_x, color='magenta', linestyle=':', linewidth=2, alpha=0.9, label=f'3D Power Center')
        ax3.axhline(p_center_y, color='magenta', linestyle=':', linewidth=2, alpha=0.9)
        ax3.axvline(true_x, color='white', linestyle='--', linewidth=2, alpha=0.9, label='True Position')
        ax3.axhline(true_y, color='white', linestyle='--', linewidth=2, alpha=0.9)
        ax3.set_xlabel('X Position [mm]')
        ax3.set_ylabel('Y Position [mm]')
        ax3.set_title(f'3D Power-Law Lorentzian (χ²/ν = {p_chi2red:.3f}, β = {p_beta:.3f})')
        plt.colorbar(contour3, ax=ax3, label='Charge [C]')
        ax3.legend(fontsize=8)
        
        # 4. X projections comparison
        ax4 = fig.add_subplot(gs[1, 0])
        # Use true_y position for fair comparison, not different fit centers
        true_y_center_idx = np.argmin(np.abs(y_range - true_y))
        
        x_proj_lorentz = Z_lorentz[true_y_center_idx, :]
        x_proj_power = Z_power[true_y_center_idx, :]
        
        # Get data points for comparison
        y_tolerance = 0.25
        mask_y = np.abs(y_positions - true_y) < y_tolerance
        if np.any(mask_y):
            x_data_proj = x_positions[mask_y]
            charge_data_proj = charge_values[mask_y]
            sort_idx = np.argsort(x_data_proj)
            x_data_proj = x_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax4.plot(x_data_proj, charge_data_proj, 'ko', markersize=6, label='Data')
        
        ax4.plot(x_range, x_proj_lorentz, 'r-', linewidth=2, label=f'3D Lorentzian (χ²={l_chi2red:.2f})')
        ax4.plot(x_range, x_proj_power, 'm--', linewidth=2, label=f'3D Power-Law (χ²={p_chi2red:.2f})')
        ax4.axvline(true_x, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True X')
        ax4.set_xlabel('X Position [mm]')
        ax4.set_ylabel('Charge [C]')
        ax4.set_title('X Projections Comparison')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Y projections comparison
        ax5 = fig.add_subplot(gs[1, 1])
        # Use true_x position for fair comparison, not different fit centers
        true_x_center_idx = np.argmin(np.abs(x_range - true_x))
        
        y_proj_lorentz = Z_lorentz[:, true_x_center_idx]
        y_proj_power = Z_power[:, true_x_center_idx]
        
        # Get data points for comparison
        x_tolerance = 0.25
        mask_x = np.abs(x_positions - true_x) < x_tolerance
        if np.any(mask_x):
            y_data_proj = y_positions[mask_x]
            charge_data_proj = charge_values[mask_x]
            sort_idx = np.argsort(y_data_proj)
            y_data_proj = y_data_proj[sort_idx]
            charge_data_proj = charge_data_proj[sort_idx]
            ax5.plot(y_data_proj, charge_data_proj, 'ko', markersize=6, label='Data')
        
        ax5.plot(y_range, y_proj_lorentz, 'r-', linewidth=2, label=f'3D Lorentzian (χ²={l_chi2red:.2f})')
        ax5.plot(y_range, y_proj_power, 'm--', linewidth=2, label=f'3D Power-Law (χ²={p_chi2red:.2f})')
        ax5.axvline(true_y, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Y')
        ax5.set_xlabel('Y Position [mm]')
        ax5.set_ylabel('Charge [C]')
        ax5.set_title('Y Projections Comparison')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Comparison statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Calculate position differences
        l_delta_x = l_center_x - true_x
        l_delta_y = l_center_y - true_y
        p_delta_x = p_center_x - true_x
        p_delta_y = p_center_y - true_y
        
        comparison_text = f"""3D Fit Comparison:

3D Lorentzian:
  Center: ({l_center_x:.3f}, {l_center_y:.3f})
  Gamma: ({l_gamma_x:.3f}, {l_gamma_y:.3f})
  Amplitude: {l_amplitude:.2e} C
  χ²/ν: {l_chi2red:.3f}
  Δ: ({l_delta_x:.3f}, {l_delta_y:.3f})

3D Power-Law Lorentzian:
  Center: ({p_center_x:.3f}, {p_center_y:.3f})
  Gamma: ({p_gamma_x:.3f}, {p_gamma_y:.3f})
  Beta: {p_beta:.3f}
  Amplitude: {p_amplitude:.2e} C
  χ²/ν: {p_chi2red:.3f}
  Δ: ({p_delta_x:.3f}, {p_delta_y:.3f})

True Position: ({true_x:.3f}, {true_y:.3f})

Best Fit: {'3D Lorentzian' if l_chi2red < p_chi2red else '3D Power-Law Lorentzian'}
Δχ²/ν: {abs(l_chi2red - p_chi2red):.3f}"""
        
        ax6.text(0.05, 0.95, comparison_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Event {event_idx}: 3D Surface Fitting Models Comparison\n' +
                    f'Lorentzian χ²/ν = {l_chi2red:.3f} vs Power-Law χ²/ν = {p_chi2red:.3f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_3d_dir, f'event_{event_idx:04d}_3d_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: 3D comparison plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating 3D comparison plot - {e}"

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Create comprehensive 3D surface fitting plots for charge sharing analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file with 3D fit data")
    parser.add_argument("-o", "--output", default="3d_fits_plots", 
                       help="Output directory for plots")
    parser.add_argument("-n", "--num_events", type=int, default=10,
                       help="Number of individual events to plot (ignored if --best_worst or --high_amplitudes is used)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file")
    parser.add_argument("--best_worst", action="store_true",
                       help="Plot the 5 best and 5 worst 3D fits based on chi-squared values")
    parser.add_argument("--high_amplitudes", type=int, metavar="N", default=None,
                       help="Plot the N events with highest 3D surface amplitudes (default: 10)")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect ROOT file branches and exit")
    
    args = parser.parse_args()
    
    # Check if ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return 1
    
    # Inspect mode
    if args.inspect:
        inspect_root_file_3d(args.root_file)
        return 0
    
    # Load data
    data = load_3d_fit_data(args.root_file, max_entries=args.max_entries)
    if data is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Check what 3D fitting data is available
    has_3d_lorentzian = '3DLorentzianFitCenterX' in data
    has_3d_power_lorentzian = '3DPowerLorentzianFitCenterX' in data
    
    if not has_3d_lorentzian and not has_3d_power_lorentzian:
        print("No 3D fitting data available in ROOT file!")
        print("Available data branches:")
        for key in sorted(data.keys()):
            if not key.startswith('_'):
                print(f"  - {key}")
        return 1
    
    available_3d_models = []
    if has_3d_lorentzian:
        available_3d_models.append('3D Lorentzian')
    if has_3d_power_lorentzian:
        available_3d_models.append('3D Power-Law Lorentzian')
    
    print(f"Available 3D fits: {', '.join(available_3d_models)}")
    
    if args.best_worst:
        # Create plots for best and worst 3D fits
        print("\nUsing best/worst 3D fit selection based on chi-squared values...")
        best_indices, worst_indices, all_metrics = find_best_worst_3d_fits(data)
        
        if not best_indices and not worst_indices:
            print("No valid 3D fits found for best/worst analysis!")
            return 1
        
        # Create subdirectories
        best_dir = os.path.join(args.output, "best_3d_fits")
        worst_dir = os.path.join(args.output, "worst_3d_fits")
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(worst_dir, exist_ok=True)
        
        success_count = 0
        
        # Plot best fits
        for i, event_idx in enumerate(best_indices):
            if has_3d_lorentzian:
                result = create_3d_lorentzian_plot(event_idx, data, best_dir)
                if "Error" not in result:
                    success_count += 1
                print(f"  Best fit {i+1} (Event {event_idx}): {result}")
            
            if has_3d_power_lorentzian:
                result = create_3d_power_lorentzian_plot(event_idx, data, best_dir)
                if "Error" not in result:
                    success_count += 1
            
            if has_3d_lorentzian and has_3d_power_lorentzian:
                result = create_3d_comparison_plot(event_idx, data, best_dir)
                if "Error" not in result:
                    success_count += 1
        
        # Plot worst fits
        for i, event_idx in enumerate(worst_indices):
            if has_3d_lorentzian:
                result = create_3d_lorentzian_plot(event_idx, data, worst_dir)
                if "Error" not in result:
                    success_count += 1
                print(f"  Worst fit {i+1} (Event {event_idx}): {result}")
            
            if has_3d_power_lorentzian:
                result = create_3d_power_lorentzian_plot(event_idx, data, worst_dir)
                if "Error" not in result:
                    success_count += 1
            
            if has_3d_lorentzian and has_3d_power_lorentzian:
                result = create_3d_comparison_plot(event_idx, data, worst_dir)
                if "Error" not in result:
                    success_count += 1
        
        print(f"\nTotal 3D plots created: {success_count}")
        
    elif args.high_amplitudes is not None:
        # Create plots for high amplitude 3D events
        n_high_amp = args.high_amplitudes if args.high_amplitudes > 0 else 10
        print(f"\nUsing high amplitude 3D event selection (top {n_high_amp} events)...")
        high_amp_indices, amplitude_metrics = find_high_amplitude_3d_events(data, n_high_amp)
        
        if not high_amp_indices:
            print("No valid high amplitude 3D events found!")
            return 1
        
        # Create subdirectory
        high_amp_dir = os.path.join(args.output, "high_3d_amplitudes")
        os.makedirs(high_amp_dir, exist_ok=True)
        
        success_count = 0
        
        for i, event_idx in enumerate(high_amp_indices):
            if has_3d_lorentzian:
                result = create_3d_lorentzian_plot(event_idx, data, high_amp_dir)
                if "Error" not in result:
                    success_count += 1
                print(f"  High amp {i+1} (Event {event_idx}): {result}")
            
            if has_3d_power_lorentzian:
                result = create_3d_power_lorentzian_plot(event_idx, data, high_amp_dir)
                if "Error" not in result:
                    success_count += 1
            
            if has_3d_lorentzian and has_3d_power_lorentzian:
                result = create_3d_comparison_plot(event_idx, data, high_amp_dir)
                if "Error" not in result:
                    success_count += 1
        
        print(f"\nTotal 3D plots created: {success_count}")
        
    else:
        # Create plots for specified number of events (original behavior)
        n_events = min(args.num_events, len(data['TrueX']))
        print(f"\nCreating 3D surface plots for first {n_events} events...")
        
        lorentzian_3d_success = 0
        power_lorentzian_3d_success = 0
        comparison_3d_success = 0
        
        for i in range(n_events):
            if has_3d_lorentzian:
                lorentz_result = create_3d_lorentzian_plot(i, data, args.output)
                if "Error" not in lorentz_result:
                    lorentzian_3d_success += 1
                if i % 5 == 0 or "Error" in lorentz_result:
                    print(f"  {lorentz_result}")
            
            if has_3d_power_lorentzian:
                power_result = create_3d_power_lorentzian_plot(i, data, args.output)
                if "Error" not in power_result:
                    power_lorentzian_3d_success += 1
                if i % 5 == 0 or "Error" in power_result:
                    print(f"  {power_result}")
            
            # Create comparison plot if both models are available
            if has_3d_lorentzian and has_3d_power_lorentzian:
                comparison_result = create_3d_comparison_plot(i, data, args.output)
                if "Error" not in comparison_result:
                    comparison_3d_success += 1
                if i % 5 == 0 or "Error" in comparison_result:
                    print(f"  {comparison_result}")
        
        print(f"\nResults:")
        if has_3d_lorentzian:
            print(f"  Successfully created {lorentzian_3d_success}/{n_events} 3D Lorentzian surface plots")
        if has_3d_power_lorentzian:
            print(f"  Successfully created {power_lorentzian_3d_success}/{n_events} 3D Power-Law Lorentzian surface plots")
        if has_3d_lorentzian and has_3d_power_lorentzian:
            print(f"  Successfully created {comparison_3d_success}/{n_events} 3D comparison plots")
        
        print(f"\nAll plots saved to: {args.output}/")
        if has_3d_lorentzian:
            print(f"  - 3D Lorentzian surfaces: {args.output}/3d_lorentzian/")
        if has_3d_power_lorentzian:
            print(f"  - 3D Power-Law Lorentzian surfaces: {args.output}/3d_power_lorentzian/")
        if has_3d_lorentzian and has_3d_power_lorentzian:
            print(f"  - 3D model comparisons: {args.output}/3d_comparison/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 