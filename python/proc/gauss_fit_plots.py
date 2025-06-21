#!/usr/bin/env python3
"""
Post-processing plotting routine for Gaussian, Lorentzian, and Power Lorentzian fit visualization of charge sharing in LGAD detectors.

This script creates plots for:
1. Gaussian and Lorentzian curve estimation for central row (x-direction) with residuals
2. Gaussian and Lorentzian curve estimation for central column (y-direction) with residuals
3. Gaussian curve estimation for main diagonal direction with residuals
4. Gaussian curve estimation for secondary diagonal direction with residuals
5. Comparison plots showing ALL fitting approaches overlaid:
   - X-direction: Row (Gaussian + Lorentzian) + Main Diagonal (Gaussian) + Secondary Diagonal (Gaussian)
   - Y-direction: Column (Gaussian + Lorentzian) + Main Diagonal (Gaussian) + Secondary Diagonal (Gaussian)
6. Model comparison plots:
   - Gaussian vs Lorentzian comparison plots
   - Lorentzian vs Power Lorentzian comparison plots (if available)
   - All models comparison plots (Gaussian + Lorentzian + Power Lorentzian if available)

The plots show fitted curves overlaid on actual charge data points from the neighborhood grid, 
along with residual plots showing fit quality. Individual directions get separate figures in their 
respective subdirectories, and comparison plots show different fitting approaches overlaid for comprehensive analysis.

AUTOMATIC UNCERTAINTY DETECTION:
The script automatically detects whether charge uncertainty branches are available in the ROOT file.
- If uncertainty branches are present and contain meaningful values, data points are plotted with error bars
- If uncertainty branches are missing or contain only zeros (when ENABLE_VERTICAL_CHARGE_UNCERTAINTIES=false), 
  data points are plotted without error bars
- This allows the same plotting script to work with ROOT files from both uncertainty-enabled and uncertainty-disabled simulations

Special modes:
- Use --best_worst to plot the 5 best and 5 worst fits based on chi-squared values
- Use --high_amplitudes N to plot the N events with highest amplitudes (useful for examining outliers)

For large ROOT files, use --max_entries to limit the number of events processed for memory efficiency.
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

def inspect_root_file(root_file):
    """
    Inspect the ROOT file to see what branches are available.
    
    Args:
        root_file (str): Path to ROOT file
    
    Returns:
        list: List of available branch names
    """
    print(f"Inspecting ROOT file: {root_file}")
    
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
                # Try to find any tree
                for key in file.keys():
                    obj = file[key]
                    if hasattr(obj, 'keys'):  # It's likely a tree
                        tree_name = key
                        break
            
            if tree_name is None:
                print("No suitable tree found in ROOT file!")
                return []
            
            print(f"Using tree: {tree_name}")
            tree = file[tree_name]
            branches = tree.keys()
            
            print(f"\nFound {len(branches)} branches:")
            for i, branch in enumerate(sorted(branches)):
                print(f"  {i+1:3d}: {branch}")
            
            return list(branches)
            
    except Exception as e:
        print(f"Error inspecting ROOT file: {e}")
        return []

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

def lorentzian_1d(x, amplitude, center, gamma, offset=0):
    """
    1D Lorentzian function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amplitude: Lorentzian amplitude
        center: Lorentzian center
        gamma: Lorentzian gamma (half-width at half-maximum, HWHM)
        offset: Baseline offset
    
    Returns:
        Lorentzian function values
    """
    return amplitude / (1 + ((x - center) / gamma)**2) + offset

def power_lorentzian_1d(x, amplitude, center, gamma, power, offset=0):
    """
    1D Power Lorentzian function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amplitude: Power Lorentzian amplitude
        center: Power Lorentzian center
        gamma: Power Lorentzian gamma (half-width at half-maximum, HWHM)
        power: Power parameter (alpha)
        offset: Baseline offset
    
    Returns:
        Power Lorentzian function values
    """
    # Power Lorentzian component
    # Standard form: 1 / (1 + ((x - center) / gamma)^2)^power
    return amplitude / (1.0 + ((x - center) / gamma)**2)**power + offset

def find_matching_branches(available_branches, expected_patterns):
    """
    Find branches that match expected patterns.
    
    Args:
        available_branches (list): List of available branch names
        expected_patterns (list): List of expected branch name patterns
    
    Returns:
        dict: Mapping from expected pattern to actual branch name (or None if not found)
    """
    branch_mapping = {}
    
    for pattern in expected_patterns:
        found = False
        # Try exact match first
        if pattern in available_branches:
            branch_mapping[pattern] = pattern
            found = True
        else:
            # Try case-insensitive and partial matches
            pattern_lower = pattern.lower()
            for branch in available_branches:
                branch_lower = branch.lower()
                if (pattern_lower in branch_lower or 
                    branch_lower in pattern_lower or
                    pattern_lower.replace('_', '') in branch_lower.replace('_', '') or
                    branch_lower.replace('_', '') in pattern_lower.replace('_', '')):
                    branch_mapping[pattern] = branch
                    found = True
                    break
        
        if not found:
            branch_mapping[pattern] = None
            print(f"Warning: Could not find branch matching pattern '{pattern}'")
    
    return branch_mapping

def detect_uncertainty_branches(data):
    """
    Detect whether charge uncertainty branches are available in the data.
    
    Args:
        data (dict): Data dictionary from ROOT file
        
    Returns:
        dict: Dictionary with flags indicating which uncertainty branches are available
    """
    uncertainty_status = {
        'gauss_row_uncertainty_available': False,
        'gauss_col_uncertainty_available': False,
        'lorentz_row_uncertainty_available': False,
        'lorentz_col_uncertainty_available': False,
        'any_uncertainties_available': False
    }
    
    # Check for Gaussian uncertainty branches
    if 'GaussFitRowChargeUncertainty' in data:
        # Check if the values are meaningful (not all zeros)
        uncertainties = data['GaussFitRowChargeUncertainty']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['gauss_row_uncertainty_available'] = True
            
    if 'GaussFitColumnChargeUncertainty' in data:
        uncertainties = data['GaussFitColumnChargeUncertainty']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['gauss_col_uncertainty_available'] = True
    
    # Check for Lorentzian uncertainty branches  
    if 'LorentzFitRowChargeUncertainty' in data:
        uncertainties = data['LorentzFitRowChargeUncertainty']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['lorentz_row_uncertainty_available'] = True
            
    if 'LorentzFitColumnChargeUncertainty' in data:
        uncertainties = data['LorentzFitColumnChargeUncertainty'] 
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            uncertainty_status['lorentz_col_uncertainty_available'] = True
    
    # Set overall flag
    uncertainty_status['any_uncertainties_available'] = (
        uncertainty_status['gauss_row_uncertainty_available'] or
        uncertainty_status['gauss_col_uncertainty_available'] or
        uncertainty_status['lorentz_row_uncertainty_available'] or
        uncertainty_status['lorentz_col_uncertainty_available']
    )
    
    return uncertainty_status

def load_successful_fits(root_file, max_entries=None):
    """
    Load data from ROOT file, with robust branch detection.
    
    Args:
        root_file (str): Path to ROOT file
        max_entries (int, optional): Maximum number of entries to load. If None, load all entries.
    
    Returns:
        dict: Dictionary containing arrays of available data
    """
    print(f"Loading data from {root_file}...")
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
                # Use the first tree-like object
                for key in file.keys():
                    obj = file[key]
                    if hasattr(obj, 'keys'):
                        tree_name = key
                        break
            
            if tree_name is None:
                print("Error: No suitable tree found in ROOT file!")
                return None
                
            tree = file[tree_name]
            
            # Create a mapping from expected names to actual branch names
            branch_mapping = {
                # Basic hit information
                'TrueX': 'TrueX',
                'TrueY': 'TrueY', 
                'PixelX': 'PixelX',
                'PixelY': 'PixelY',
                'IsPixelHit': 'IsPixelHit',
                
                # Gaussian fit results - Row/X direction
                'Fit2D_XCenter': 'GaussFitRowCenter',
                'Fit2D_XSigma': 'GaussFitRowStdev', 
                'Fit2D_XAmplitude': 'GaussFitRowAmplitude',
                'Fit2D_XVerticalOffset': 'GaussFitRowVerticalOffset',
                'Fit2D_XCenterErr': 'GaussFitRowCenterErr',
                'Fit2D_XSigmaErr': 'GaussFitRowStdevErr',
                'Fit2D_XAmplitudeErr': 'GaussFitRowAmplitudeErr',
                'Fit2D_XChi2red': 'GaussFitRowChi2red',
                'Fit2D_XNPoints': 'GaussFitRowDOF',  # DOF + parameters = NPoints
                
                # Gaussian fit results - Column/Y direction  
                'Fit2D_YCenter': 'GaussFitColumnCenter',
                'Fit2D_YSigma': 'GaussFitColumnStdev',
                'Fit2D_YAmplitude': 'GaussFitColumnAmplitude',
                'Fit2D_YVerticalOffset': 'GaussFitColumnVerticalOffset',
                'Fit2D_YCenterErr': 'GaussFitColumnCenterErr',
                'Fit2D_YSigmaErr': 'GaussFitColumnStdevErr',
                'Fit2D_YAmplitudeErr': 'GaussFitColumnAmplitudeErr',
                'Fit2D_YChi2red': 'GaussFitColumnChi2red',
                'Fit2D_YNPoints': 'GaussFitColumnDOF',
                
                # Lorentzian fit results - Row/X direction
                'Fit2D_Lorentz_XCenter': 'LorentzFitRowCenter',
                'Fit2D_Lorentz_XGamma': 'LorentzFitRowGamma',
                'Fit2D_Lorentz_XAmplitude': 'LorentzFitRowAmplitude',
                'Fit2D_Lorentz_XVerticalOffset': 'LorentzFitRowVerticalOffset',
                'Fit2D_Lorentz_XCenterErr': 'LorentzFitRowCenterErr',
                'Fit2D_Lorentz_XGammaErr': 'LorentzFitRowGammaErr',
                'Fit2D_Lorentz_XAmplitudeErr': 'LorentzFitRowAmplitudeErr',
                'Fit2D_Lorentz_XChi2red': 'LorentzFitRowChi2red',
                'Fit2D_Lorentz_XNPoints': 'LorentzFitRowDOF',
                
                # Lorentzian fit results - Column/Y direction
                'Fit2D_Lorentz_YCenter': 'LorentzFitColumnCenter',
                'Fit2D_Lorentz_YGamma': 'LorentzFitColumnGamma',
                'Fit2D_Lorentz_YAmplitude': 'LorentzFitColumnAmplitude',
                'Fit2D_Lorentz_YVerticalOffset': 'LorentzFitColumnVerticalOffset',
                'Fit2D_Lorentz_YCenterErr': 'LorentzFitColumnCenterErr',
                'Fit2D_Lorentz_YGammaErr': 'LorentzFitColumnGammaErr',
                'Fit2D_Lorentz_YAmplitudeErr': 'LorentzFitColumnAmplitudeErr',
                'Fit2D_Lorentz_YChi2red': 'LorentzFitColumnChi2red',
                'Fit2D_Lorentz_YNPoints': 'LorentzFitColumnDOF',
                
                # Power Lorentzian fit results - Row/X direction
                'Fit2D_PowerLorentz_XCenter': 'PowerLorentzFitRowCenter',
                'Fit2D_PowerLorentz_XGamma': 'PowerLorentzFitRowBeta',  # Beta is the shape parameter
                'Fit2D_PowerLorentz_XAmplitude': 'PowerLorentzFitRowAmplitude',
                'Fit2D_PowerLorentz_XPower': 'PowerLorentzFitRowLambda',  # Lambda is the power parameter
                'Fit2D_PowerLorentz_XCenterErr': 'PowerLorentzFitRowCenterErr',
                'Fit2D_PowerLorentz_XGammaErr': 'PowerLorentzFitRowBetaErr',
                'Fit2D_PowerLorentz_XAmplitudeErr': 'PowerLorentzFitRowAmplitudeErr',
                'Fit2D_PowerLorentz_XPowerErr': 'PowerLorentzFitRowLambdaErr',
                'Fit2D_PowerLorentz_XChi2red': 'PowerLorentzFitRowChi2red',
                'Fit2D_PowerLorentz_XNPoints': 'PowerLorentzFitRowDOF',
                
                # Power Lorentzian fit results - Column/Y direction
                'Fit2D_PowerLorentz_YCenter': 'PowerLorentzFitColumnCenter',
                'Fit2D_PowerLorentz_YGamma': 'PowerLorentzFitColumnBeta',  # Beta is the shape parameter
                'Fit2D_PowerLorentz_YAmplitude': 'PowerLorentzFitColumnAmplitude',
                'Fit2D_PowerLorentz_YPower': 'PowerLorentzFitColumnLambda',  # Lambda is the power parameter
                'Fit2D_PowerLorentz_YCenterErr': 'PowerLorentzFitColumnCenterErr',
                'Fit2D_PowerLorentz_YGammaErr': 'PowerLorentzFitColumnBetaErr',
                'Fit2D_PowerLorentz_YAmplitudeErr': 'PowerLorentzFitColumnAmplitudeErr',
                'Fit2D_PowerLorentz_YPowerErr': 'PowerLorentzFitColumnLambdaErr',
                'Fit2D_PowerLorentz_YChi2red': 'PowerLorentzFitColumnChi2red',
                'Fit2D_PowerLorentz_YNPoints': 'PowerLorentzFitColumnDOF',
                
                # Diagonal Gaussian fits - Main diagonal X
                'FitDiag_MainXCenter': 'GaussFitMainDiagXCenter',
                'FitDiag_MainXSigma': 'GaussFitMainDiagXStdev',
                'FitDiag_MainXAmplitude': 'GaussFitMainDiagXAmplitude',
                'FitDiag_MainXVerticalOffset': 'GaussFitMainDiagXVerticalOffset',
                'FitDiag_MainXCenterErr': 'GaussFitMainDiagXCenterErr',
                'FitDiag_MainXSigmaErr': 'GaussFitMainDiagXStdevErr',
                'FitDiag_MainXAmplitudeErr': 'GaussFitMainDiagXAmplitudeErr',
                'FitDiag_MainXChi2red': 'GaussFitMainDiagXChi2red',
                'FitDiag_MainXNPoints': 'GaussFitMainDiagXDOF',
                
                # Diagonal Gaussian fits - Main diagonal Y
                'FitDiag_MainYCenter': 'GaussFitMainDiagYCenter',
                'FitDiag_MainYSigma': 'GaussFitMainDiagYStdev',
                'FitDiag_MainYAmplitude': 'GaussFitMainDiagYAmplitude',
                'FitDiag_MainYVerticalOffset': 'GaussFitMainDiagYVerticalOffset',
                'FitDiag_MainYCenterErr': 'GaussFitMainDiagYCenterErr',
                'FitDiag_MainYSigmaErr': 'GaussFitMainDiagYStdevErr',
                'FitDiag_MainYAmplitudeErr': 'GaussFitMainDiagYAmplitudeErr',
                'FitDiag_MainYChi2red': 'GaussFitMainDiagYChi2red',
                'FitDiag_MainYNPoints': 'GaussFitMainDiagYDOF',
                
                # Diagonal Gaussian fits - Secondary diagonal X
                'FitDiag_SecXCenter': 'GaussFitSecondDiagXCenter',
                'FitDiag_SecXSigma': 'GaussFitSecondDiagXStdev',
                'FitDiag_SecXAmplitude': 'GaussFitSecondDiagXAmplitude',
                'FitDiag_SecXVerticalOffset': 'GaussFitSecondDiagXVerticalOffset',
                'FitDiag_SecXCenterErr': 'GaussFitSecondDiagXCenterErr',
                'FitDiag_SecXSigmaErr': 'GaussFitSecondDiagXStdevErr',
                'FitDiag_SecXAmplitudeErr': 'GaussFitSecondDiagXAmplitudeErr',
                'FitDiag_SecXChi2red': 'GaussFitSecondDiagXChi2red',
                'FitDiag_SecXNPoints': 'GaussFitSecondDiagXDOF',
                
                # Diagonal Gaussian fits - Secondary diagonal Y
                'FitDiag_SecYCenter': 'GaussFitSecondDiagYCenter',
                'FitDiag_SecYSigma': 'GaussFitSecondDiagYStdev',
                'FitDiag_SecYAmplitude': 'GaussFitSecondDiagYAmplitude',
                'FitDiag_SecYVerticalOffset': 'GaussFitSecondDiagYVerticalOffset',
                'FitDiag_SecYCenterErr': 'GaussFitSecondDiagYCenterErr',
                'FitDiag_SecYSigmaErr': 'GaussFitSecondDiagYStdevErr',
                'FitDiag_SecYAmplitudeErr': 'GaussFitSecondDiagYAmplitudeErr',
                'FitDiag_SecYChi2red': 'GaussFitSecondDiagYChi2red',
                'FitDiag_SecYNPoints': 'GaussFitSecondDiagYDOF',
                
                # Grid neighborhood data
                'NonPixel_GridNeighborhoodCharge': 'GridNeighborhoodCharges',
                'NonPixel_GridNeighborhoodDistances': 'GridNeighborhoodDistances',
                'NonPixel_GridNeighborhoodAngles': 'GridNeighborhoodAngles',
                'NonPixel_GridNeighborhoodChargeFractions': 'GridNeighborhoodChargeFractions',
                
                # Charge uncertainties (5% of max charge)
                'GaussFitRowChargeUncertainty': 'GaussFitRowChargeUncertainty',
                'GaussFitColumnChargeUncertainty': 'GaussFitColumnChargeUncertainty',
                'LorentzFitRowChargeUncertainty': 'LorentzFitRowChargeUncertainty',
                'LorentzFitColumnChargeUncertainty': 'LorentzFitColumnChargeUncertainty',
                
                # Nearest pixel positions
                'NearestPixelX': 'PixelX',
                'NearestPixelY': 'PixelY', 
                'NearestPixelZ': 'PixelZ'
            }
            
            # Load all available branches with mapping
            data = {}
            loaded_count = 0
            skipped_power_count = 0
            
            for expected_name, actual_name in branch_mapping.items():
                if actual_name in tree.keys():
                    try:
                        if max_entries is not None:
                            data[expected_name] = tree[actual_name].array(library="np", entry_stop=max_entries)
                        else:
                            data[expected_name] = tree[actual_name].array(library="np")
                        loaded_count += 1
                        if loaded_count <= 10:  # Only print first 10 to avoid spam
                            print(f"Loaded: {expected_name} -> {actual_name}")
                    except Exception as e:
                        print(f"Warning: Could not load {actual_name}: {e}")
                else:
                    # Suppress warnings for Power Lorentzian branches if they're expected to be missing
                    if 'PowerLorentz' in expected_name:
                        skipped_power_count += 1
                    else:
                        print(f"Warning: Branch {actual_name} not found for {expected_name}")
            
            if skipped_power_count > 0:
                print(f"Note: Skipped {skipped_power_count} Power Lorentzian branches (fitting disabled)")
            
            print(f"Successfully loaded {loaded_count} branches with {len(data['TrueX'])} events")
            
            # Detect uncertainty branch availability
            uncertainty_status = detect_uncertainty_branches(data)
            data['_uncertainty_status'] = uncertainty_status
            
            # Print uncertainty detection results
            if uncertainty_status['any_uncertainties_available']:
                print("Charge uncertainty branches detected:")
                if uncertainty_status['gauss_row_uncertainty_available']:
                    print("  ✓ Gaussian row uncertainties available")
                if uncertainty_status['gauss_col_uncertainty_available']:
                    print("  ✓ Gaussian column uncertainties available") 
                if uncertainty_status['lorentz_row_uncertainty_available']:
                    print("  ✓ Lorentzian row uncertainties available")
                if uncertainty_status['lorentz_col_uncertainty_available']:
                    print("  ✓ Lorentzian column uncertainties available")
            else:
                print("No charge uncertainty branches detected - plots will show data points without error bars")
            
            # Create success flags since they're not individual branches in this ROOT file
            # We'll check if the chi2 values are reasonable (non-zero and finite)
            n_events = len(data['TrueX'])
            
            # Create success flags based on available fit data
            data['Fit2D_Successful'] = np.ones(n_events, dtype=bool)  # Assume all successful for now
            data['Fit2D_Lorentz_Successful'] = np.ones(n_events, dtype=bool)
            data['Fit2D_PowerLorentz_Successful'] = np.ones(n_events, dtype=bool)
            data['FitDiag_Successful'] = np.ones(n_events, dtype=bool)
            data['FitDiag_MainXSuccessful'] = np.ones(n_events, dtype=bool)
            data['FitDiag_MainYSuccessful'] = np.ones(n_events, dtype=bool)
            data['FitDiag_SecXSuccessful'] = np.ones(n_events, dtype=bool)
            data['FitDiag_SecYSuccessful'] = np.ones(n_events, dtype=bool)
            
            # Apply basic filtering for non-pixel hits if available
            if 'IsPixelHit' in data:
                is_non_pixel = ~data['IsPixelHit']
                print(f"Non-pixel events: {np.sum(is_non_pixel)}")
                
                # Filter all data to non-pixel events
                filtered_data = {}
                for key, values in data.items():
                    if hasattr(values, '__len__') and len(values) == n_events:
                        # Regular arrays - filter normally
                        filtered_data[key] = values[is_non_pixel]
                    else:
                        # Keep as is (might be jagged arrays or constants)
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

def extract_row_column_data(event_idx, data, neighborhood_radius=4):
    """
    Extract charge data for central row and column from neighborhood grid data.
    Now properly extracts from the full 9x9 grid stored in GridNeighborhoodCharges branch.
    Also extracts charge uncertainties (5% of max charge) for error bars.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (row_data, col_data) where each is (positions, charges, uncertainties) for central row/column
    """
    
    # First try to use the raw neighborhood grid data (preferred method)
    if 'NonPixel_GridNeighborhoodCharge' in data and event_idx < len(data['NonPixel_GridNeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NonPixel_GridNeighborhoodCharge'][event_idx]
            grid_charge_fractions = data['NonPixel_GridNeighborhoodChargeFractions'][event_idx] if 'NonPixel_GridNeighborhoodChargeFractions' in data else None
            
            # Get nearest pixel position for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract central row (Y = center_y, varying X)
            row_positions = []
            row_charges = []
            
            # Central row corresponds to j = neighborhood_radius (middle of grid)
            center_row = neighborhood_radius
            for i in range(grid_size):  # i varies from 0 to 8 (X direction)
                grid_idx = i * grid_size + center_row  # Column-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calculate X position for this pixel
                    offset_i = i - neighborhood_radius  # -4 to +4
                    x_pos = center_x + offset_i * pixel_spacing
                    row_positions.append(x_pos)
                    row_charges.append(grid_charges[grid_idx])
            
            # Extract central column (X = center_x, varying Y)
            col_positions = []
            col_charges = []
            
            # Central column corresponds to i = neighborhood_radius (middle of grid)
            center_col = neighborhood_radius
            for j in range(grid_size):  # j varies from 0 to 8 (Y direction)
                grid_idx = center_col * grid_size + j  # Column-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calculate Y position for this pixel
                    offset_j = j - neighborhood_radius  # -4 to +4
                    y_pos = center_y + offset_j * pixel_spacing
                    col_positions.append(y_pos)
                    col_charges.append(grid_charges[grid_idx])
            
            row_positions = np.array(row_positions)
            row_charge_values = np.array(row_charges)
            col_positions = np.array(col_positions)
            col_charge_values = np.array(col_charges)
            
            # Extract charge uncertainties (5% of max charge for each direction) if available
            uncertainty_status = data.get('_uncertainty_status', {})
            
            if uncertainty_status.get('gauss_row_uncertainty_available', False):
                row_uncertainty = data.get('GaussFitRowChargeUncertainty', [0])[event_idx] if 'GaussFitRowChargeUncertainty' in data and event_idx < len(data['GaussFitRowChargeUncertainty']) else 0
            else:
                row_uncertainty = 0
                
            if uncertainty_status.get('gauss_col_uncertainty_available', False):
                col_uncertainty = data.get('GaussFitColumnChargeUncertainty', [0])[event_idx] if 'GaussFitColumnChargeUncertainty' in data and event_idx < len(data['GaussFitColumnChargeUncertainty']) else 0
            else:
                col_uncertainty = 0
            
            # Create uncertainty arrays (same uncertainty for all points in a line)
            row_uncertainties = np.full(len(row_positions), row_uncertainty)
            col_uncertainties = np.full(len(col_positions), col_uncertainty)
            
            print(f"Event {event_idx}: Extracted {len(row_positions)} row points, {len(col_positions)} col points from grid")
            if uncertainty_status.get('any_uncertainties_available', False):
                print(f"  Row uncertainty: {row_uncertainty:.2e} C, Col uncertainty: {col_uncertainty:.2e} C")
            else:
                print(f"  No uncertainties available - using data points without error bars")
            
        except Exception as e:
            print(f"Warning: Could not extract grid data for event {event_idx}: {e}")
            # Fall back to fit results
            row_positions, row_charge_values, row_uncertainties = extract_from_fit_results(event_idx, data, 'row')
            col_positions, col_charge_values, col_uncertainties = extract_from_fit_results(event_idx, data, 'col')
    else:
        # Fall back to fit results if grid data not available
        print(f"Warning: NonPixel_GridNeighborhoodCharge not available for event {event_idx}, using fit results")
        row_positions, row_charge_values, row_uncertainties = extract_from_fit_results(event_idx, data, 'row')
        col_positions, col_charge_values, col_uncertainties = extract_from_fit_results(event_idx, data, 'col')
    
    return (row_positions, row_charge_values, row_uncertainties), (col_positions, col_charge_values, col_uncertainties)


def extract_from_fit_results(event_idx, data, direction):
    """
    Extract data from fit results as fallback when grid data is not available.
    Now also extracts charge uncertainties.
    """
    try:
        if direction == 'row':
            if 'LorentzFitRowPixelCoords' in data and event_idx < len(data['LorentzFitRowPixelCoords']):
                coords = data['LorentzFitRowPixelCoords'][event_idx]
                charges = data['LorentzFitRowChargeValues'][event_idx]
                positions = np.array(coords) if hasattr(coords, '__len__') else np.array([])
                charge_values = np.array(charges) if hasattr(charges, '__len__') else np.array([])
            else:
                # Create synthetic row data
                center = data['Fit2D_XCenter'][event_idx] if 'Fit2D_XCenter' in data else data['PixelX'][event_idx]
                sigma = data['Fit2D_XSigma'][event_idx] if 'Fit2D_XSigma' in data else 0.5
                amplitude = data['Fit2D_XAmplitude'][event_idx] if 'Fit2D_XAmplitude' in data else 1e-12
                pixel_spacing = 0.5
                positions = np.array([center + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
                charge_values = amplitude * np.exp(-0.5 * ((positions - center) / sigma)**2)
            
            # Extract row uncertainty if available
            uncertainty_status = data.get('_uncertainty_status', {})
            if uncertainty_status.get('gauss_row_uncertainty_available', False):
                uncertainty = data.get('GaussFitRowChargeUncertainty', [0])[event_idx] if 'GaussFitRowChargeUncertainty' in data and event_idx < len(data['GaussFitRowChargeUncertainty']) else 0
            else:
                uncertainty = 0
            uncertainties = np.full(len(positions), uncertainty)
            
        else:  # column
            if 'LorentzFitColumnPixelCoords' in data and event_idx < len(data['LorentzFitColumnPixelCoords']):
                coords = data['LorentzFitColumnPixelCoords'][event_idx]
                charges = data['LorentzFitColumnChargeValues'][event_idx]
                positions = np.array(coords) if hasattr(coords, '__len__') else np.array([])
                charge_values = np.array(charges) if hasattr(charges, '__len__') else np.array([])
            else:
                # Create synthetic column data
                center = data['Fit2D_YCenter'][event_idx] if 'Fit2D_YCenter' in data else data['PixelY'][event_idx]
                sigma = data['Fit2D_YSigma'][event_idx] if 'Fit2D_YSigma' in data else 0.5
                amplitude = data['Fit2D_YAmplitude'][event_idx] if 'Fit2D_YAmplitude' in data else 1e-12
                pixel_spacing = 0.5
                positions = np.array([center + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
                charge_values = amplitude * np.exp(-0.5 * ((positions - center) / sigma)**2)
            
            # Extract column uncertainty if available
            uncertainty_status = data.get('_uncertainty_status', {})
            if uncertainty_status.get('gauss_col_uncertainty_available', False):
                uncertainty = data.get('GaussFitColumnChargeUncertainty', [0])[event_idx] if 'GaussFitColumnChargeUncertainty' in data and event_idx < len(data['GaussFitColumnChargeUncertainty']) else 0
            else:
                uncertainty = 0
            uncertainties = np.full(len(positions), uncertainty)
                
    except Exception as e:
        print(f"Warning: Could not extract {direction} data for event {event_idx}: {e}")
        positions = np.array([])
        charge_values = np.array([])
        uncertainties = np.array([])
    
    return positions, charge_values, uncertainties


def extract_full_grid_data(event_idx, data, neighborhood_radius=4):
    """
    Extract all pixels with charge from the full neighborhood grid for 2D visualization.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (x_positions, y_positions, charge_values) for all pixels with charge > 0
    """
    
    if 'NonPixel_GridNeighborhoodCharge' in data and event_idx < len(data['NonPixel_GridNeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NonPixel_GridNeighborhoodCharge'][event_idx]
            
            # Get nearest pixel position for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
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
            print(f"Warning: Could not extract full grid data for event {event_idx}: {e}")
            return np.array([]), np.array([]), np.array([])
    else:
        print(f"Warning: NonPixel_GridNeighborhoodCharge not available for event {event_idx}")
        return np.array([]), np.array([]), np.array([])


def extract_diagonal_data(event_idx, data, neighborhood_radius=4):
    """
    Extract charge data for main and secondary diagonals from neighborhood grid data.
    Now properly extracts from the full 9x9 grid when available.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: ((main_x_pos, main_x_charges), (main_y_pos, main_y_charges), 
                (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges))
    """
    
    # First try to extract diagonal data from the raw grid
    if 'NonPixel_GridNeighborhoodCharge' in data and event_idx < len(data['NonPixel_GridNeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NonPixel_GridNeighborhoodCharge'][event_idx]
            
            # Get nearest pixel position for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract main diagonal (i == j, top-left to bottom-right)
            main_x_positions = []
            main_x_charges = []
            main_y_positions = []
            main_y_charges = []
            
            for k in range(grid_size):  # k varies from 0 to 8
                i = k  # X index
                j = k  # Y index (same as X for main diagonal)
                grid_idx = i * grid_size + j  # Column-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calculate position for this diagonal pixel
                    offset_i = i - neighborhood_radius  # -4 to +4 for X
                    offset_j = j - neighborhood_radius  # -4 to +4 for Y
                    x_pos = center_x + offset_i * pixel_spacing
                    y_pos = center_y + offset_j * pixel_spacing
                    
                    # For diagonal plots, we need to project to 1D coordinate
                    # Main diagonal: use distance along diagonal from center
                    diag_coord = offset_i * pixel_spacing * np.sqrt(2)  # √2 × pitch per step
                    
                    main_x_positions.append(diag_coord)
                    main_x_charges.append(grid_charges[grid_idx])
                    main_y_positions.append(diag_coord)  # Same coordinate for both X and Y projections
                    main_y_charges.append(grid_charges[grid_idx])
            
            # Extract secondary diagonal (i + j == grid_size - 1, top-right to bottom-left)
            sec_x_positions = []
            sec_x_charges = []
            sec_y_positions = []
            sec_y_charges = []
            
            for k in range(grid_size):  # k varies from 0 to 8
                i = k  # X index
                j = grid_size - 1 - k  # Y index (complementary for secondary diagonal)
                grid_idx = i * grid_size + j  # Column-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calculate position for this diagonal pixel
                    offset_i = i - neighborhood_radius  # -4 to +4 for X
                    offset_j = j - neighborhood_radius  # -4 to +4 for Y
                    
                    # For secondary diagonal: use distance along diagonal from center
                    # Secondary diagonal runs from top-right to bottom-left
                    diag_coord = offset_i * pixel_spacing * np.sqrt(2)  # X coordinate along the diagonal
                    
                    sec_x_positions.append(diag_coord)
                    sec_x_charges.append(grid_charges[grid_idx])
                    sec_y_positions.append(diag_coord)
                    sec_y_charges.append(grid_charges[grid_idx])
            
            main_x_positions = np.array(main_x_positions)
            main_x_charge_values = np.array(main_x_charges)
            main_y_positions = np.array(main_y_positions)
            main_y_charge_values = np.array(main_y_charges)
            sec_x_positions = np.array(sec_x_positions)
            sec_x_charge_values = np.array(sec_x_charges)
            sec_y_positions = np.array(sec_y_positions)
            sec_y_charge_values = np.array(sec_y_charges)
            
            print(f"Event {event_idx}: Extracted {len(main_x_positions)} main diagonal points, {len(sec_x_positions)} secondary diagonal points")
            
        except Exception as e:
            print(f"Warning: Could not extract diagonal grid data for event {event_idx}: {e}")
            # Fall back to fit results
            return extract_diagonal_from_fit_results(event_idx, data)
    else:
        # Fall back to fit results if grid data not available
        print(f"Warning: NonPixel_GridNeighborhoodCharge not available for event {event_idx}, using fit results for diagonals")
        return extract_diagonal_from_fit_results(event_idx, data)
    
    return ((main_x_positions, main_x_charge_values), 
            (main_y_positions, main_y_charge_values),
            (sec_x_positions, sec_x_charge_values), 
            (sec_y_positions, sec_y_charge_values))


def extract_diagonal_from_fit_results(event_idx, data):
    """
    Extract diagonal data from fit results as fallback.
    """
    # Try to extract main diagonal data from Lorentzian fit results
    try:
        if 'LorentzFitMainDiagXPixelCoords' in data and event_idx < len(data['LorentzFitMainDiagXPixelCoords']):
            main_x_coords = data['LorentzFitMainDiagXPixelCoords'][event_idx]
            main_x_charges = data['LorentzFitMainDiagXChargeValues'][event_idx]
            main_x_positions = np.array(main_x_coords) if hasattr(main_x_coords, '__len__') else np.array([])
            main_x_charge_values = np.array(main_x_charges) if hasattr(main_x_charges, '__len__') else np.array([])
        else:
            # Create synthetic main diagonal X data
            center_x = data['FitDiag_MainXCenter'][event_idx] if 'FitDiag_MainXCenter' in data else data['PixelX'][event_idx]
            sigma = data['FitDiag_MainXSigma'][event_idx] if 'FitDiag_MainXSigma' in data else 0.5
            amplitude = data['FitDiag_MainXAmplitude'][event_idx] if 'FitDiag_MainXAmplitude' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            main_x_positions = np.array([center_x + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            main_x_charge_values = amplitude * np.exp(-0.5 * ((main_x_positions - center_x) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract main diagonal X data for event {event_idx}: {e}")
        main_x_positions = np.array([])
        main_x_charge_values = np.array([])
    
    try:
        if 'LorentzFitMainDiagYPixelCoords' in data and event_idx < len(data['LorentzFitMainDiagYPixelCoords']):
            main_y_coords = data['LorentzFitMainDiagYPixelCoords'][event_idx]
            main_y_charges = data['LorentzFitMainDiagYChargeValues'][event_idx]
            main_y_positions = np.array(main_y_coords) if hasattr(main_y_coords, '__len__') else np.array([])
            main_y_charge_values = np.array(main_y_charges) if hasattr(main_y_charges, '__len__') else np.array([])
        else:
            # Create synthetic main diagonal Y data
            center_y = data['FitDiag_MainYCenter'][event_idx] if 'FitDiag_MainYCenter' in data else data['PixelY'][event_idx]
            sigma = data['FitDiag_MainYSigma'][event_idx] if 'FitDiag_MainYSigma' in data else 0.5
            amplitude = data['FitDiag_MainYAmplitude'][event_idx] if 'FitDiag_MainYAmplitude' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            main_y_positions = np.array([center_y + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            main_y_charge_values = amplitude * np.exp(-0.5 * ((main_y_positions - center_y) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract main diagonal Y data for event {event_idx}: {e}")
        main_y_positions = np.array([])
        main_y_charge_values = np.array([])
    
    try:
        if 'LorentzFitSecondDiagXPixelCoords' in data and event_idx < len(data['LorentzFitSecondDiagXPixelCoords']):
            sec_x_coords = data['LorentzFitSecondDiagXPixelCoords'][event_idx]
            sec_x_charges = data['LorentzFitSecondDiagXChargeValues'][event_idx]
            sec_x_positions = np.array(sec_x_coords) if hasattr(sec_x_coords, '__len__') else np.array([])
            sec_x_charge_values = np.array(sec_x_charges) if hasattr(sec_x_charges, '__len__') else np.array([])
        else:
            # Create synthetic secondary diagonal X data
            center_x = data['FitDiag_SecXCenter'][event_idx] if 'FitDiag_SecXCenter' in data else data['PixelX'][event_idx]
            sigma = data['FitDiag_SecXSigma'][event_idx] if 'FitDiag_SecXSigma' in data else 0.5
            amplitude = data['FitDiag_SecXAmplitude'][event_idx] if 'FitDiag_SecXAmplitude' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            sec_x_positions = np.array([center_x + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            sec_x_charge_values = amplitude * np.exp(-0.5 * ((sec_x_positions - center_x) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract secondary diagonal X data for event {event_idx}: {e}")
        sec_x_positions = np.array([])
        sec_x_charge_values = np.array([])
    
    try:
        if 'LorentzFitSecondDiagYPixelCoords' in data and event_idx < len(data['LorentzFitSecondDiagYPixelCoords']):
            sec_y_coords = data['LorentzFitSecondDiagYPixelCoords'][event_idx]
            sec_y_charges = data['LorentzFitSecondDiagYChargeValues'][event_idx]
            sec_y_positions = np.array(sec_y_coords) if hasattr(sec_y_coords, '__len__') else np.array([])
            sec_y_charge_values = np.array(sec_y_charges) if hasattr(sec_y_charges, '__len__') else np.array([])
        else:
            # Create synthetic secondary diagonal Y data
            center_y = data['FitDiag_SecYCenter'][event_idx] if 'FitDiag_SecYCenter' in data else data['PixelY'][event_idx]
            sigma = data['FitDiag_SecYSigma'][event_idx] if 'FitDiag_SecYSigma' in data else 0.5
            amplitude = data['FitDiag_SecYAmplitude'][event_idx] if 'FitDiag_SecYAmplitude' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            sec_y_positions = np.array([center_y + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            sec_y_charge_values = amplitude * np.exp(-0.5 * ((sec_y_positions - center_y) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract secondary diagonal Y data for event {event_idx}: {e}")
        sec_y_positions = np.array([])
        sec_y_charge_values = np.array([])
    
    return ((main_x_positions, main_x_charge_values), 
            (main_y_positions, main_y_charge_values),
            (sec_x_positions, sec_x_charge_values), 
            (sec_y_positions, sec_y_charge_values))

def autoscale_axes(fig):
    """
    Auto-scale all axes in a figure for better visualization.
    """
    for ax in fig.get_axes():
        ax.relim()
        ax.autoscale_view()

def plot_data_points(ax, positions, charges, uncertainties, **kwargs):
    """
    Plot data points with or without error bars, depending on whether uncertainties are meaningful.
    
    Args:
        ax: Matplotlib axis object
        positions: X positions
        charges: Y values (charges)
        uncertainties: Uncertainty values
        **kwargs: Additional keyword arguments passed to plotting function
    
    Returns:
        matplotlib artist object
    """
    # Check if uncertainties are meaningful (not all zeros or very small)
    has_meaningful_uncertainties = (
        len(uncertainties) > 0 and 
        np.any(np.abs(uncertainties) > 1e-20) and
        np.any(np.abs(uncertainties) > 0.001 * np.max(np.abs(charges))) if len(charges) > 0 else False
    )
    
    if has_meaningful_uncertainties:
        # Use errorbar plot with uncertainties
        return ax.errorbar(positions, charges, yerr=uncertainties, **kwargs)
    else:
        # Use simple plot without error bars
        # Convert errorbar kwargs to plot kwargs
        plot_kwargs = dict(kwargs)
        if 'fmt' in plot_kwargs:
            plot_kwargs['marker'] = plot_kwargs['fmt'][1] if len(plot_kwargs['fmt']) > 1 else 'o'
            plot_kwargs['color'] = plot_kwargs['fmt'][0] if len(plot_kwargs['fmt']) > 0 and plot_kwargs['fmt'][0] != 'k' else 'black'
            plot_kwargs['linestyle'] = 'None'
            del plot_kwargs['fmt']
        if 'capsize' in plot_kwargs:
            del plot_kwargs['capsize']
        if 'markersize' not in plot_kwargs and 'marker' in plot_kwargs:
            plot_kwargs['markersize'] = 6
            
        return ax.plot(positions, charges, **plot_kwargs)

def calculate_residuals(positions, charges, fit_params, fit_type='gaussian'):
    """
    Calculate residuals between data and fitted function.
    
    Args:
        positions (array): Position values
        charges (array): Charge values (data)
        fit_params (dict): Fitted parameters with keys 'center', 'sigma'/'gamma'/'power', 'amplitude'
        fit_type (str): 'gaussian', 'lorentzian', or 'power_lorentzian'
    
    Returns:
        array: Residuals (data - fit)
    """
    if len(positions) == 0:
        return np.array([])
    
    if fit_type == 'gaussian':
        fitted_values = gaussian_1d(positions, 
                                   fit_params['amplitude'], 
                                   fit_params['center'], 
                                   fit_params['sigma'])
    elif fit_type == 'lorentzian':
        fitted_values = lorentzian_1d(positions, 
                                     fit_params['amplitude'], 
                                     fit_params['center'], 
                                     fit_params['gamma'])
    elif fit_type == 'power_lorentzian':
        fitted_values = power_lorentzian_1d(positions, 
                                           fit_params['amplitude'], 
                                           fit_params['center'], 
                                           fit_params['gamma'],
                                           fit_params['power'])
    else:
        raise ValueError("fit_type must be 'gaussian', 'lorentzian', or 'power_lorentzian'")
    
    return charges - fitted_values

def create_all_lorentzian_plot(event_idx, data, output_dir="plots"):
    """
    Create Lorentzian fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get Lorentzian fit parameters
        x_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
        x_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
        x_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
        x_lorentz_vertical_offset = data.get('Fit2D_Lorentz_XVerticalOffset', [0])[event_idx] if 'Fit2D_Lorentz_XVerticalOffset' in data else 0
        x_lorentz_chi2red = data['Fit2D_Lorentz_XChi2red'][event_idx]
        x_lorentz_dof = data.get('Fit2D_Lorentz_XNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        y_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
        y_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
        y_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
        y_lorentz_vertical_offset = data.get('Fit2D_Lorentz_YVerticalOffset', [0])[event_idx] if 'Fit2D_Lorentz_YVerticalOffset' in data else 0
        y_lorentz_chi2red = data['Fit2D_Lorentz_YChi2red'][event_idx]
        y_lorentz_dof = data.get('Fit2D_Lorentz_YNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        # Diagonal parameters (treat as Lorentzian with gamma = sigma)
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx] 
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_vertical_offset = data.get('FitDiag_MainXVerticalOffset', [0])[event_idx] if 'FitDiag_MainXVerticalOffset' in data else 0
        main_diag_x_chi2red = data['FitDiag_MainXChi2red'][event_idx]
        main_diag_x_dof = data.get('FitDiag_MainXNPoints', [0])[event_idx] - 4
        
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx]
        main_diag_y_vertical_offset = data.get('FitDiag_MainYVerticalOffset', [0])[event_idx] if 'FitDiag_MainYVerticalOffset' in data else 0
        main_diag_y_chi2red = data['FitDiag_MainYChi2red'][event_idx]
        main_diag_y_dof = data.get('FitDiag_MainYNPoints', [0])[event_idx] - 4
        
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_vertical_offset = data.get('FitDiag_SecXVerticalOffset', [0])[event_idx] if 'FitDiag_SecXVerticalOffset' in data else 0
        sec_diag_x_chi2red = data['FitDiag_SecXChi2red'][event_idx]
        sec_diag_x_dof = data.get('FitDiag_SecXNPoints', [0])[event_idx] - 4
        
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_vertical_offset = data.get('FitDiag_SecYVerticalOffset', [0])[event_idx] if 'FitDiag_SecYVerticalOffset' in data else 0
        sec_diag_y_chi2red = data['FitDiag_SecYChi2red'][event_idx]
        sec_diag_y_dof = data.get('FitDiag_SecYNPoints', [0])[event_idx] - 4
        
        # True positions and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calculate delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        lorentzian_dir = os.path.join(output_dir, "lorentzian")
        os.makedirs(lorentzian_dir, exist_ok=True)
        
        # Create all Lorentzian collage plot
        fig_lor_all = plt.figure(figsize=(20, 15))
        gs_lor_all = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        def plot_lorentzian_direction(ax, positions, charges, uncertainties, center, gamma, amplitude, vertical_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(positions) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, positions, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Lorentzian fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
            y_fit = lorentzian_1d(pos_range, amplitude, center, gamma, vertical_offset)
            ax.plot(pos_range, y_fit, 'r-', linewidth=2, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='red', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Lorentzian (χ²/ν = {chi2red:.2f})\n'
                          f'Δ pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n'
                          rf'${direction}_{{fit}} - {direction}_{{true}}$ = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            ax.set_xlabel('Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        # Plot all directions using the helper function
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot
        ax_row = fig_lor_all.add_subplot(gs_lor_all[0, 0])
        plot_lorentzian_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                                x_lorentz_center, x_lorentz_gamma, x_lorentz_amplitude, 
                                x_lorentz_chi2red, x_lorentz_dof, true_x, 'Row (X-direction)', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_lor_all.add_subplot(gs_lor_all[0, 1])
        plot_lorentzian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                y_lorentz_center, y_lorentz_gamma, y_lorentz_amplitude, 
                                y_lorentz_chi2red, y_lorentz_dof, true_y, 'Column (Y-direction)', 'y', delta_pixel_y)
        
        # Main diagonal X plot
        ax_main_x = fig_lor_all.add_subplot(gs_lor_all[1, 0])
        plot_lorentzian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, 
                                main_diag_x_chi2red, main_diag_x_dof, true_x, 'Main Diagonal X', 'x', delta_pixel_x)
        
        # Main diagonal Y plot
        ax_main_y = fig_lor_all.add_subplot(gs_lor_all[1, 1])
        plot_lorentzian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, 
                                main_diag_y_chi2red, main_diag_y_dof, true_y, 'Main Diagonal Y', 'y', delta_pixel_y)
        
        # Secondary diagonal X plot
        ax_sec_x = fig_lor_all.add_subplot(gs_lor_all[2, 0])
        plot_lorentzian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, 
                                sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'Secondary Diagonal X', 'x', delta_pixel_x)
        
        # Secondary diagonal Y plot
        ax_sec_y = fig_lor_all.add_subplot(gs_lor_all[2, 1])
        plot_lorentzian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, 
                                sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Secondary Diagonal Y', 'y', delta_pixel_y)
        
        
        plt.suptitle(f'Event {event_idx}: Lorentzian Fits (All Directions)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(lorentzian_dir, f'event_{event_idx:04d}_all_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Lorentzian collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Lorentzian plot - {e}"

def create_all_power_lorentzian_plot(event_idx, data, output_dir="plots"):
    """
    Create Power Lorentzian fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Check if Power Lorentzian fit parameters are available
        if 'Fit2D_PowerLorentz_XCenter' not in data:
            return f"Event {event_idx}: Power Lorentzian fit data not available"
        
        # Get Power Lorentzian fit parameters
        x_power_center = data['Fit2D_PowerLorentz_XCenter'][event_idx]
        x_power_gamma = data['Fit2D_PowerLorentz_XGamma'][event_idx]
        x_power_amplitude = data['Fit2D_PowerLorentz_XAmplitude'][event_idx]
        x_power_power = data.get('Fit2D_PowerLorentz_XPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_XPower' in data else 1.0
        x_power_chi2red = data['Fit2D_PowerLorentz_XChi2red'][event_idx]
        x_power_dof = data.get('Fit2D_PowerLorentz_XNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters
        
        y_power_center = data['Fit2D_PowerLorentz_YCenter'][event_idx]
        y_power_gamma = data['Fit2D_PowerLorentz_YGamma'][event_idx]
        y_power_amplitude = data['Fit2D_PowerLorentz_YAmplitude'][event_idx]
        y_power_power = data.get('Fit2D_PowerLorentz_YPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_YPower' in data else 1.0
        y_power_chi2red = data['Fit2D_PowerLorentz_YChi2red'][event_idx]
        y_power_dof = data.get('Fit2D_PowerLorentz_YNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters
        
        # True positions and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calculate delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        power_lorentzian_dir = os.path.join(output_dir, "power_lorentzian")
        os.makedirs(power_lorentzian_dir, exist_ok=True)
        
        # Create all Power Lorentzian collage plot
        fig_power_all = plt.figure(figsize=(20, 15))
        gs_power_all = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        def plot_power_lorentzian_direction(ax, positions, charges, uncertainties, center, gamma, amplitude, power, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(positions) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, positions, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Power Lorentzian fit
            pos_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
            y_fit = power_lorentzian_1d(pos_range, amplitude, center, gamma, power)
            ax.plot(pos_range, y_fit, 'm-', linewidth=2, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='magenta', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Power Lorentzian (χ²/ν = {chi2red:.2f})\n'
                          f'Power = {power:.2f}\n'
                          f'Δ pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n'
                          rf'${direction}_{{fit}} - {direction}_{{true}}$ = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            ax.set_xlabel('Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot
        ax_row = fig_power_all.add_subplot(gs_power_all[0, 0])
        plot_power_lorentzian_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                                      x_power_center, x_power_gamma, x_power_amplitude, x_power_power,
                                      x_power_chi2red, x_power_dof, true_x, 'Row (X-direction)', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_power_all.add_subplot(gs_power_all[0, 1])
        plot_power_lorentzian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                      y_power_center, y_power_gamma, y_power_amplitude, y_power_power,
                                      y_power_chi2red, y_power_dof, true_y, 'Column (Y-direction)', 'y', delta_pixel_y)
        
        # For diagonals, use Gaussian parameters (Power Lorentzian diagonals may not be implemented)
        main_diag_x_center = data.get('FitDiag_MainXCenter', [x_power_center])[event_idx]
        main_diag_x_sigma = data.get('FitDiag_MainXSigma', [x_power_gamma])[event_idx]
        main_diag_x_amplitude = data.get('FitDiag_MainXAmplitude', [x_power_amplitude])[event_idx]
        main_diag_x_chi2red = data.get('FitDiag_MainXChi2red', [1.0])[event_idx]
        main_diag_x_dof = data.get('FitDiag_MainXNPoints', [4])[event_idx] - 4
        
        # Main diagonal X plot (using Gaussian fit for diagonal, but labeled as approximation)
        ax_main_x = fig_power_all.add_subplot(gs_power_all[1, 0])
        plot_power_lorentzian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                      main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, 1.0,
                                      main_diag_x_chi2red, main_diag_x_dof, true_x, 'Main Diagonal X (approx)', 'x', delta_pixel_x)
        
        # Similar for other diagonals...
        main_diag_y_center = data.get('FitDiag_MainYCenter', [y_power_center])[event_idx]
        main_diag_y_sigma = data.get('FitDiag_MainYSigma', [y_power_gamma])[event_idx]
        main_diag_y_amplitude = data.get('FitDiag_MainYAmplitude', [y_power_amplitude])[event_idx]
        main_diag_y_chi2red = data.get('FitDiag_MainYChi2red', [1.0])[event_idx]
        main_diag_y_dof = data.get('FitDiag_MainYNPoints', [4])[event_idx] - 4
        
        ax_main_y = fig_power_all.add_subplot(gs_power_all[1, 1])
        plot_power_lorentzian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                      main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, 1.0,
                                      main_diag_y_chi2red, main_diag_y_dof, true_y, 'Main Diagonal Y (approx)', 'y', delta_pixel_y)
        
        sec_diag_x_center = data.get('FitDiag_SecXCenter', [x_power_center])[event_idx]
        sec_diag_x_sigma = data.get('FitDiag_SecXSigma', [x_power_gamma])[event_idx]
        sec_diag_x_amplitude = data.get('FitDiag_SecXAmplitude', [x_power_amplitude])[event_idx]
        sec_diag_x_chi2red = data.get('FitDiag_SecXChi2red', [1.0])[event_idx]
        sec_diag_x_dof = data.get('FitDiag_SecXNPoints', [4])[event_idx] - 4
        
        ax_sec_x = fig_power_all.add_subplot(gs_power_all[2, 0])
        plot_power_lorentzian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                      sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, 1.0,
                                      sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'Secondary Diagonal X (approx)', 'x', delta_pixel_x)
        
        sec_diag_y_center = data.get('FitDiag_SecYCenter', [y_power_center])[event_idx]
        sec_diag_y_sigma = data.get('FitDiag_SecYSigma', [y_power_gamma])[event_idx]
        sec_diag_y_amplitude = data.get('FitDiag_SecYAmplitude', [y_power_amplitude])[event_idx]
        sec_diag_y_chi2red = data.get('FitDiag_SecYChi2red', [1.0])[event_idx]
        sec_diag_y_dof = data.get('FitDiag_SecYNPoints', [4])[event_idx] - 4
        
        ax_sec_y = fig_power_all.add_subplot(gs_power_all[2, 1])
        plot_power_lorentzian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                      sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, 1.0,
                                      sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Secondary Diagonal Y (approx)', 'y', delta_pixel_y)
        
        plt.suptitle(f'Event {event_idx}: Power Lorentzian Fits (All Directions)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(power_lorentzian_dir, f'event_{event_idx:04d}_all_power_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Power Lorentzian collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Power Lorentzian plot - {e}"

def create_all_gaussian_plot(event_idx, data, output_dir="plots"):
    """
    Create Gaussian fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get Gaussian fit parameters
        x_gauss_center = data['Fit2D_XCenter'][event_idx]
        x_gauss_sigma = data['Fit2D_XSigma'][event_idx]
        x_gauss_amplitude = data['Fit2D_XAmplitude'][event_idx]
        x_gauss_vertical_offset = data.get('Fit2D_XVerticalOffset', [0])[event_idx] if 'Fit2D_XVerticalOffset' in data else 0
        x_gauss_chi2red = data['Fit2D_XChi2red'][event_idx]
        x_gauss_dof = data.get('Fit2D_XNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        y_gauss_center = data['Fit2D_YCenter'][event_idx]
        y_gauss_sigma = data['Fit2D_YSigma'][event_idx]
        y_gauss_amplitude = data['Fit2D_YAmplitude'][event_idx]
        y_gauss_vertical_offset = data.get('Fit2D_YVerticalOffset', [0])[event_idx] if 'Fit2D_YVerticalOffset' in data else 0
        y_gauss_chi2red = data['Fit2D_YChi2red'][event_idx]
        y_gauss_dof = data.get('Fit2D_YNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        # Diagonal parameters (using Gaussian fits)
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx] 
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_vertical_offset = data.get('FitDiag_MainXVerticalOffset', [0])[event_idx] if 'FitDiag_MainXVerticalOffset' in data else 0
        main_diag_x_chi2red = data['FitDiag_MainXChi2red'][event_idx]
        main_diag_x_dof = data.get('FitDiag_MainXNPoints', [0])[event_idx] - 4
        
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx]
        main_diag_y_vertical_offset = data.get('FitDiag_MainYVerticalOffset', [0])[event_idx] if 'FitDiag_MainYVerticalOffset' in data else 0
        main_diag_y_chi2red = data['FitDiag_MainYChi2red'][event_idx]
        main_diag_y_dof = data.get('FitDiag_MainYNPoints', [0])[event_idx] - 4
        
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_vertical_offset = data.get('FitDiag_SecXVerticalOffset', [0])[event_idx] if 'FitDiag_SecXVerticalOffset' in data else 0
        sec_diag_x_chi2red = data['FitDiag_SecXChi2red'][event_idx]
        sec_diag_x_dof = data.get('FitDiag_SecXNPoints', [0])[event_idx] - 4
        
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_vertical_offset = data.get('FitDiag_SecYVerticalOffset', [0])[event_idx] if 'FitDiag_SecYVerticalOffset' in data else 0
        sec_diag_y_chi2red = data['FitDiag_SecYChi2red'][event_idx]
        sec_diag_y_dof = data.get('FitDiag_SecYNPoints', [0])[event_idx] - 4
        
        # True positions and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calculate delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        gaussian_dir = os.path.join(output_dir, "gaussian")
        os.makedirs(gaussian_dir, exist_ok=True)
        
        # Create all Gaussian collage plot
        fig_gauss_all = plt.figure(figsize=(20, 15))
        gs_gauss_all = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        def plot_gaussian_direction(ax, positions, charges, uncertainties, center, sigma, amplitude, vertical_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(positions) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, positions, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Gaussian fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
            y_fit = gaussian_1d(pos_range, amplitude, center, sigma, vertical_offset)
            ax.plot(pos_range, y_fit, 'b-', linewidth=2, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='blue', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Gaussian (χ²/ν = {chi2red:.2f})\n'
                          f'Δ pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n'
                          rf'${direction}_{{fit}} - {direction}_{{true}}$ = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            ax.set_xlabel('Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        # Plot all directions using the helper function
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot
        ax_row = fig_gauss_all.add_subplot(gs_gauss_all[0, 0])
        plot_gaussian_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                              x_gauss_center, x_gauss_sigma, x_gauss_amplitude, x_gauss_vertical_offset,
                              x_gauss_chi2red, x_gauss_dof, true_x, 'Row (X-direction)', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_gauss_all.add_subplot(gs_gauss_all[0, 1])
        plot_gaussian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                              y_gauss_center, y_gauss_sigma, y_gauss_amplitude, y_gauss_vertical_offset,
                              y_gauss_chi2red, y_gauss_dof, true_y, 'Column (Y-direction)', 'y', delta_pixel_y)
        
        # Main diagonal X plot
        ax_main_x = fig_gauss_all.add_subplot(gs_gauss_all[1, 0])
        plot_gaussian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                              main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, main_diag_x_vertical_offset,
                              main_diag_x_chi2red, main_diag_x_dof, true_x, 'Main Diagonal X', 'x', delta_pixel_x)
        
        # Main diagonal Y plot
        ax_main_y = fig_gauss_all.add_subplot(gs_gauss_all[1, 1])
        plot_gaussian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                              main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, main_diag_y_vertical_offset,
                              main_diag_y_chi2red, main_diag_y_dof, true_y, 'Main Diagonal Y', 'y', delta_pixel_y)
        
        # Secondary diagonal X plot
        ax_sec_x = fig_gauss_all.add_subplot(gs_gauss_all[2, 0])
        plot_gaussian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                              sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, sec_diag_x_vertical_offset,
                              sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'Secondary Diagonal X', 'x', delta_pixel_x)
        
        # Secondary diagonal Y plot
        ax_sec_y = fig_gauss_all.add_subplot(gs_gauss_all[2, 1])
        plot_gaussian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                              sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, sec_diag_y_vertical_offset,
                              sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Secondary Diagonal Y', 'y', delta_pixel_y)
        
        plt.suptitle(f'Event {event_idx}: Gaussian Fits (All Directions)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(gaussian_dir, f'event_{event_idx:04d}_all_gaussian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Gaussian collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Gaussian plot - {e}"

def create_all_models_combined_plot(event_idx, data, output_dir="plots"):
    """
    Create combined Gaussian, Lorentzian, and Power Lorentzian fit plots for ALL directions in a single collage for one event.
    Automatically detects which models are available and shows all available fits.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Check which models are available
        has_gaussian = 'Fit2D_XCenter' in data
        has_lorentzian = 'Fit2D_Lorentz_XCenter' in data
        has_power_lorentzian = 'Fit2D_PowerLorentz_XCenter' in data
        
        available_models = []
        if has_gaussian:
            available_models.append('Gaussian')
        if has_lorentzian:
            available_models.append('Lorentzian')
        if has_power_lorentzian:
            available_models.append('Power Lorentzian')
        
        if not available_models:
            return f"Event {event_idx}: No fitting models available"
        
        print(f"Event {event_idx}: Available models: {', '.join(available_models)}")
        
        # Get Gaussian fit parameters if available
        if has_gaussian:
            x_gauss_center = data['Fit2D_XCenter'][event_idx]
            x_gauss_sigma = data['Fit2D_XSigma'][event_idx]
            x_gauss_amplitude = data['Fit2D_XAmplitude'][event_idx]
            x_gauss_chi2red = data['Fit2D_XChi2red'][event_idx]
            
            y_gauss_center = data['Fit2D_YCenter'][event_idx]
            y_gauss_sigma = data['Fit2D_YSigma'][event_idx]
            y_gauss_amplitude = data['Fit2D_YAmplitude'][event_idx]
            y_gauss_chi2red = data['Fit2D_YChi2red'][event_idx]
        
        # Get Lorentzian fit parameters if available
        if has_lorentzian:
            x_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
            x_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
            x_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
            x_lorentz_chi2red = data['Fit2D_Lorentz_XChi2red'][event_idx]
            
            y_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
            y_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
            y_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
            y_lorentz_chi2red = data['Fit2D_Lorentz_YChi2red'][event_idx]
        
        # Get Power Lorentzian fit parameters if available
        if has_power_lorentzian:
            x_power_center = data['Fit2D_PowerLorentz_XCenter'][event_idx]
            x_power_gamma = data['Fit2D_PowerLorentz_XGamma'][event_idx]
            x_power_amplitude = data['Fit2D_PowerLorentz_XAmplitude'][event_idx]
            x_power_power = data.get('Fit2D_PowerLorentz_XPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_XPower' in data else 1.0
            x_power_chi2red = data['Fit2D_PowerLorentz_XChi2red'][event_idx]
            
            y_power_center = data['Fit2D_PowerLorentz_YCenter'][event_idx]
            y_power_gamma = data['Fit2D_PowerLorentz_YGamma'][event_idx]
            y_power_amplitude = data['Fit2D_PowerLorentz_YAmplitude'][event_idx]
            y_power_power = data.get('Fit2D_PowerLorentz_YPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_YPower' in data else 1.0
            y_power_chi2red = data['Fit2D_PowerLorentz_YChi2red'][event_idx]
        
        # Diagonal parameters (using Gaussian fits)
        main_diag_x_center = data.get('FitDiag_MainXCenter', [0])[event_idx] if 'FitDiag_MainXCenter' in data else 0
        main_diag_x_sigma = data.get('FitDiag_MainXSigma', [0.1])[event_idx] if 'FitDiag_MainXSigma' in data else 0.1
        main_diag_x_amplitude = data.get('FitDiag_MainXAmplitude', [1e-12])[event_idx] if 'FitDiag_MainXAmplitude' in data else 1e-12
        main_diag_x_chi2red = data.get('FitDiag_MainXChi2red', [1.0])[event_idx] if 'FitDiag_MainXChi2red' in data else 1.0
        
        # True positions and pixel positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calculate delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        combined_dir = os.path.join(output_dir, "all_models_combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        # Create combined plot
        fig_combined = plt.figure(figsize=(20, 15))
        gs_combined = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        def plot_all_models_direction(ax, positions, charges, uncertainties, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all available fits."""
            if len(positions) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, positions, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot range for smooth curves
            pos_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
            
            legend_lines = []
            legend_text_parts = []
            
            # Plot Gaussian fit if available
            if has_gaussian and direction == 'x':
                gauss_fit = gaussian_1d(pos_range, x_gauss_amplitude, x_gauss_center, x_gauss_sigma)
                line = ax.plot(pos_range, gauss_fit, 'b-', linewidth=2, alpha=0.9, label='Gaussian')[0]
                legend_lines.append(line)
                ax.axvline(x_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = x_gauss_center - true_pos
                legend_text_parts.append(f'Gaussian: χ²/ν = {x_gauss_chi2red:.2f}, Δ = {gauss_diff:.3f}')
            elif has_gaussian and direction == 'y':
                gauss_fit = gaussian_1d(pos_range, y_gauss_amplitude, y_gauss_center, y_gauss_sigma)
                line = ax.plot(pos_range, gauss_fit, 'b-', linewidth=2, alpha=0.9, label='Gaussian')[0]
                legend_lines.append(line)
                ax.axvline(y_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = y_gauss_center - true_pos
                legend_text_parts.append(f'Gaussian: χ²/ν = {y_gauss_chi2red:.2f}, Δ = {gauss_diff:.3f}')
            
            # Plot Lorentzian fit if available
            if has_lorentzian and direction == 'x':
                lorentz_fit = lorentzian_1d(pos_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma)
                line = ax.plot(pos_range, lorentz_fit, 'r--', linewidth=2, alpha=0.9, label='Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(x_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = x_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentzian: χ²/ν = {x_lorentz_chi2red:.2f}, Δ = {lorentz_diff:.3f}')
            elif has_lorentzian and direction == 'y':
                lorentz_fit = lorentzian_1d(pos_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma)
                line = ax.plot(pos_range, lorentz_fit, 'r--', linewidth=2, alpha=0.9, label='Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(y_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = y_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentzian: χ²/ν = {y_lorentz_chi2red:.2f}, Δ = {lorentz_diff:.3f}')
            
            # Plot Power Lorentzian fit if available
            if has_power_lorentzian and direction == 'x':
                power_fit = power_lorentzian_1d(pos_range, x_power_amplitude, x_power_center, x_power_gamma, x_power_power)
                line = ax.plot(pos_range, power_fit, 'm:', linewidth=2, alpha=0.9, label='Power Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(x_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = x_power_center - true_pos
                legend_text_parts.append(f'Power Lorentzian: χ²/ν = {x_power_chi2red:.2f}, Δ = {power_diff:.3f}')
            elif has_power_lorentzian and direction == 'y':
                power_fit = power_lorentzian_1d(pos_range, y_power_amplitude, y_power_center, y_power_gamma, y_power_power)
                line = ax.plot(pos_range, power_fit, 'm:', linewidth=2, alpha=0.9, label='Power Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(y_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = y_power_center - true_pos
                legend_text_parts.append(f'Power Lorentzian: χ²/ν = {y_power_chi2red:.2f}, Δ = {power_diff:.3f}')
            
            # Add true position line
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Position')
            
            # Create legend text
            legend_text = '\n'.join(legend_text_parts)
            legend_text += f'\nΔ pixel {direction.upper()} = {delta_pixel:.3f} mm'
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
            
            ax.set_xlabel('Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot
        ax_row = fig_combined.add_subplot(gs_combined[0, 0])
        plot_all_models_direction(ax_row, row_pos, row_charges, row_uncertainties, true_x, 'Row (X-direction)', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_combined.add_subplot(gs_combined[0, 1])
        plot_all_models_direction(ax_col, col_pos, col_charges, col_uncertainties, true_y, 'Column (Y-direction)', 'y', delta_pixel_y)
        
        # For diagonal plots, use Gaussian parameters as approximation
        # Main diagonal X plot
        ax_main_x = fig_combined.add_subplot(gs_combined[1, 0])
        if len(main_x_pos) >= 3:
            plot_data_points(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, fmt='ko', markersize=6, capsize=3, alpha=0.8)
            pos_range = np.linspace(main_x_pos.min() - 0.1, main_x_pos.max() + 0.1, 200)
            diag_fit = gaussian_1d(pos_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma)
            ax_main_x.plot(pos_range, diag_fit, 'b-', linewidth=2, alpha=0.9, label='Gaussian (diagonal)')
            ax_main_x.axvline(true_x, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax_main_x.axvline(main_diag_x_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
            ax_main_x.legend()
        ax_main_x.set_title('Main Diagonal X')
        ax_main_x.set_xlabel('Position [mm]')
        ax_main_x.set_ylabel('Charge [C]')
        ax_main_x.grid(True, alpha=0.3)
        
        # Similar for other diagonal plots (simplified)
        ax_main_y = fig_combined.add_subplot(gs_combined[1, 1])
        ax_main_y.set_title('Main Diagonal Y')
        ax_main_y.grid(True, alpha=0.3)
        
        ax_sec_x = fig_combined.add_subplot(gs_combined[2, 0])
        ax_sec_x.set_title('Secondary Diagonal X')
        ax_sec_x.grid(True, alpha=0.3)
        
        ax_sec_y = fig_combined.add_subplot(gs_combined[2, 1])
        ax_sec_y.set_title('Secondary Diagonal Y')
        ax_sec_y.grid(True, alpha=0.3)
        
        models_str = "_".join([m.lower().replace(" ", "_") for m in available_models])
        plt.suptitle(f'Event {event_idx}: All Available Models ({", ".join(available_models)})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f'event_{event_idx:04d}_all_models_combined.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: Combined plot with {len(available_models)} models created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating combined models plot - {e}"

def calculate_fit_quality_metric(data, event_idx):
    """
    Calculate an overall fit quality metric for an event.
    Lower values indicate better fits.
    
    Args:
        data (dict): Data dictionary
        event_idx (int): Event index
    
    Returns:
        float: Combined fit quality metric (lower is better)
    """
    try:
        # Get chi-squared values for different fits
        x_gauss_chi2 = data.get('Fit2D_XChi2red', [float('inf')])[event_idx]
        y_gauss_chi2 = data.get('Fit2D_YChi2red', [float('inf')])[event_idx]
        x_lorentz_chi2 = data.get('Fit2D_Lorentz_XChi2red', [float('inf')])[event_idx]
        y_lorentz_chi2 = data.get('Fit2D_Lorentz_YChi2red', [float('inf')])[event_idx]
        x_power_chi2 = data.get('Fit2D_PowerLorentz_XChi2red', [float('inf')])[event_idx]
        y_power_chi2 = data.get('Fit2D_PowerLorentz_YChi2red', [float('inf')])[event_idx]
        
        # Get diagonal chi-squared values
        main_x_chi2 = data.get('FitDiag_MainXChi2red', [float('inf')])[event_idx]
        main_y_chi2 = data.get('FitDiag_MainYChi2red', [float('inf')])[event_idx]
        sec_x_chi2 = data.get('FitDiag_SecXChi2red', [float('inf')])[event_idx]
        sec_y_chi2 = data.get('FitDiag_SecYChi2red', [float('inf')])[event_idx]
        
        # Convert any invalid values to inf
        chi2_values = []
        for chi2 in [x_gauss_chi2, y_gauss_chi2, x_lorentz_chi2, y_lorentz_chi2, x_power_chi2, y_power_chi2,
                     main_x_chi2, main_y_chi2, sec_x_chi2, sec_y_chi2]:
            if np.isfinite(chi2) and chi2 > 0:
                chi2_values.append(chi2)
        
        if not chi2_values:
            return float('inf')
        
        # Use the average of all valid chi-squared values as the metric
        # Weight the main row/column fits more heavily
        weights = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1]  # Row/col Gauss/Lorentz/Power weighted more
        weighted_chi2 = []
        
        for i, chi2 in enumerate([x_gauss_chi2, y_gauss_chi2, x_lorentz_chi2, y_lorentz_chi2, x_power_chi2, y_power_chi2,
                                 main_x_chi2, main_y_chi2, sec_x_chi2, sec_y_chi2]):
            if np.isfinite(chi2) and chi2 > 0:
                weighted_chi2.extend([chi2] * weights[i])
        
        return np.mean(weighted_chi2) if weighted_chi2 else float('inf')
        
    except Exception as e:
        print(f"Warning: Could not calculate fit quality for event {event_idx}: {e}")
        return float('inf')

def find_best_worst_fits(data, n_best=5, n_worst=5):
    """
    Find the events with the best and worst fit quality.
    
    Args:
        data (dict): Data dictionary
        n_best (int): Number of best fits to find
        n_worst (int): Number of worst fits to find
    
    Returns:
        tuple: (best_indices, worst_indices, all_metrics)
    """
    print("Calculating fit quality metrics for all events...")
    
    n_events = len(data['TrueX'])
    fit_metrics = []
    
    for i in range(n_events):
        metric = calculate_fit_quality_metric(data, i)
        fit_metrics.append((i, metric))
    
    # Sort by fit quality (lower is better)
    fit_metrics.sort(key=lambda x: x[1])
    
    # Remove events with infinite chi2 (failed fits)
    valid_fits = [(idx, metric) for idx, metric in fit_metrics if np.isfinite(metric)]
    
    if len(valid_fits) == 0:
        print("Warning: No valid fits found!")
        return [], [], []
    
    print(f"Found {len(valid_fits)} events with valid fits out of {n_events} total events")
    
    # Get best fits (lowest chi2)
    best_fits = valid_fits[:n_best]
    best_indices = [idx for idx, metric in best_fits]
    
    # Get worst fits (highest chi2, but still finite)
    worst_fits = valid_fits[-n_worst:]
    worst_indices = [idx for idx, metric in worst_fits]
    
    print(f"Best fits (lowest χ²):")
    for i, (idx, metric) in enumerate(best_fits):
        print(f"  {i+1}. Event {idx}: χ² = {metric:.3f}")
    
    print(f"Worst fits (highest χ²):")
    for i, (idx, metric) in enumerate(worst_fits):
        print(f"  {i+1}. Event {idx}: χ² = {metric:.3f}")
    
    return best_indices, worst_indices, fit_metrics

def find_high_amplitude_events(data, n_events=10):
    """
    Find events with the highest amplitudes to examine potential outliers.
    
    Args:
        data (dict): Data dictionary
        n_events (int): Number of high amplitude events to find
    
    Returns:
        tuple: (high_amp_indices, amplitude_metrics)
    """
    print("Finding events with highest amplitudes...")
    
    n_total = len(data['TrueX'])
    amplitude_metrics = []
    
    for i in range(n_total):
        try:
            # Get Gaussian amplitudes for row and column fits
            x_gauss_amp = data.get('Fit2D_XAmplitude', [0])[i] if 'Fit2D_XAmplitude' in data and i < len(data['Fit2D_XAmplitude']) else 0
            y_gauss_amp = data.get('Fit2D_YAmplitude', [0])[i] if 'Fit2D_YAmplitude' in data and i < len(data['Fit2D_YAmplitude']) else 0
            
            # Get Lorentzian amplitudes for comparison
            x_lorentz_amp = data.get('Fit2D_Lorentz_XAmplitude', [0])[i] if 'Fit2D_Lorentz_XAmplitude' in data and i < len(data['Fit2D_Lorentz_XAmplitude']) else 0
            y_lorentz_amp = data.get('Fit2D_Lorentz_YAmplitude', [0])[i] if 'Fit2D_Lorentz_YAmplitude' in data and i < len(data['Fit2D_Lorentz_YAmplitude']) else 0
            
            # Get Power Lorentzian amplitudes for comparison
            x_power_amp = data.get('Fit2D_PowerLorentz_XAmplitude', [0])[i] if 'Fit2D_PowerLorentz_XAmplitude' in data and i < len(data['Fit2D_PowerLorentz_XAmplitude']) else 0
            y_power_amp = data.get('Fit2D_PowerLorentz_YAmplitude', [0])[i] if 'Fit2D_PowerLorentz_YAmplitude' in data and i < len(data['Fit2D_PowerLorentz_YAmplitude']) else 0
            
            # Use the maximum amplitude across all fits as the metric
            max_amplitude = max(abs(x_gauss_amp), abs(y_gauss_amp), abs(x_lorentz_amp), abs(y_lorentz_amp), abs(x_power_amp), abs(y_power_amp))
            
            # Also get chi2 values for quality assessment
            x_gauss_chi2 = data.get('Fit2D_XChi2red', [float('inf')])[i]
            y_gauss_chi2 = data.get('Fit2D_YChi2red', [float('inf')])[i]
            avg_chi2 = (x_gauss_chi2 + y_gauss_chi2) / 2.0 if np.isfinite(x_gauss_chi2) and np.isfinite(y_gauss_chi2) else float('inf')
            
            amplitude_metrics.append((i, max_amplitude, avg_chi2, x_gauss_amp, y_gauss_amp, x_lorentz_amp, y_lorentz_amp))
            
        except Exception as e:
            print(f"Warning: Could not extract amplitude data for event {i}: {e}")
            amplitude_metrics.append((i, 0, float('inf'), 0, 0, 0, 0))
    
    # Sort by amplitude (highest first)
    amplitude_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Get events with finite amplitudes
    valid_amps = [(idx, amp, chi2, x_g, y_g, x_l, y_l) for idx, amp, chi2, x_g, y_g, x_l, y_l in amplitude_metrics if amp > 0 and np.isfinite(amp)]
    
    if len(valid_amps) == 0:
        print("Warning: No events with valid amplitudes found!")
        return [], []
    
    print(f"Found {len(valid_amps)} events with valid amplitudes out of {n_total} total events")
    
    # Get highest amplitude events
    high_amp_events = valid_amps[:n_events]
    high_amp_indices = [idx for idx, amp, chi2, x_g, y_g, x_l, y_l in high_amp_events]
    
    print(f"Highest amplitude events:")
    for i, (idx, amp, chi2, x_g, y_g, x_l, y_l) in enumerate(high_amp_events):
        print(f"  {i+1}. Event {idx}: Max Amp = {amp:.2e} C (χ² = {chi2:.3f})")
        print(f"      Gauss: X={x_g:.2e}, Y={y_g:.2e} | Lorentz: X={x_l:.2e}, Y={y_l:.2e}")
    
    return high_amp_indices, amplitude_metrics

def create_best_worst_plots(data, output_dir="plots"):
    """
    Create plots for the 5 best and 5 worst fits.
    
    Args:
        data (dict): Data dictionary
        output_dir (str): Output directory for plots
    
    Returns:
        int: Number of successfully created plots
    """
    print("\nFinding best and worst fits...")
    best_indices, worst_indices, all_metrics = find_best_worst_fits(data)
    
    if not best_indices and not worst_indices:
        print("No valid fits found for best/worst analysis!")
        return 0
    
    # Create subdirectories for best and worst fits
    best_dir = os.path.join(output_dir, "best_fits")
    worst_dir = os.path.join(output_dir, "worst_fits")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)
    
    success_count = 0
    
    # Plot best fits
    print(f"\nCreating plots for {len(best_indices)} best fits...")
    for i, event_idx in enumerate(best_indices):
        # Create all three types of plots for each best fit
        
        # Lorentzian plot
        lorentz_result = create_all_lorentzian_plot(event_idx, data, best_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  Best fit {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gaussian plot
        gauss_result = create_all_gaussian_plot(event_idx, data, best_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentzian plot
        power_result = create_all_power_lorentzian_plot(event_idx, data, best_dir)
        if "Error" not in power_result:
            success_count += 1
        
        # Combined plot with all models
        combined_result = create_all_models_combined_plot(event_idx, data, best_dir)
        if "Error" not in combined_result:
            success_count += 1
    
    # Plot worst fits
    print(f"\nCreating plots for {len(worst_indices)} worst fits...")
    for i, event_idx in enumerate(worst_indices):
        # Create all three types of plots for each worst fit
        
        # Lorentzian plot
        lorentz_result = create_all_lorentzian_plot(event_idx, data, worst_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  Worst fit {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gaussian plot
        gauss_result = create_all_gaussian_plot(event_idx, data, worst_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentzian plot
        power_result = create_all_power_lorentzian_plot(event_idx, data, worst_dir)
        if "Error" not in power_result:
            success_count += 1
        
        # Combined plot with all models
        combined_result = create_all_models_combined_plot(event_idx, data, worst_dir)
        if "Error" not in combined_result:
            success_count += 1
    
    print(f"\nBest/worst fit plots saved to:")
    print(f"  - Best fits: {best_dir}/")
    print(f"  - Worst fits: {worst_dir}/")
    
    return success_count

def create_high_amplitude_plots(data, output_dir="plots", n_events=10):
    """
    Create plots for events with the highest amplitudes to examine potential outliers.
    
    Args:
        data (dict): Data dictionary
        output_dir (str): Output directory for plots
        n_events (int): Number of high amplitude events to plot
    
    Returns:
        int: Number of successfully created plots
    """
    print(f"\nFinding {n_events} highest amplitude events...")
    high_amp_indices, amplitude_metrics = find_high_amplitude_events(data, n_events)
    
    if not high_amp_indices:
        print("No valid high amplitude events found!")
        return 0
    
    # Create subdirectory for high amplitude fits
    high_amp_dir = os.path.join(output_dir, "high_amplitudes")
    os.makedirs(high_amp_dir, exist_ok=True)
    
    success_count = 0
    
    # Plot high amplitude events
    print(f"\nCreating plots for {len(high_amp_indices)} highest amplitude events...")
    for i, event_idx in enumerate(high_amp_indices):
        # Create all three types of plots for each high amplitude event
        
        # Lorentzian plot
        lorentz_result = create_all_lorentzian_plot(event_idx, data, high_amp_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  High amp {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gaussian plot
        gauss_result = create_all_gaussian_plot(event_idx, data, high_amp_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentzian plot
        power_result = create_all_power_lorentzian_plot(event_idx, data, high_amp_dir)
        if "Error" not in power_result:
            success_count += 1
        
        # Combined plot with all models
        combined_result = create_all_models_combined_plot(event_idx, data, high_amp_dir)
        if "Error" not in combined_result:
            success_count += 1
    
    print(f"\nHigh amplitude fit plots saved to: {high_amp_dir}/")
    
    return success_count

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Create Gaussian and Lorentzian fit plots for charge sharing analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file with 2D Gaussian and Lorentzian fit data")
    parser.add_argument("-o", "--output", default="gauss_lorentz_plots", 
                       help="Output directory for plots")
    parser.add_argument("-n", "--num_events", type=int, default=10,
                       help="Number of individual events to plot (ignored if --best_worst or --high_amplitudes is used)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file (for handling large files)")
    parser.add_argument("--best_worst", action="store_true",
                       help="Plot the 5 best and 5 worst fits based on chi-squared values instead of first N events")
    parser.add_argument("--high_amplitudes", type=int, metavar="N", default=None,
                       help="Plot the N events with highest amplitudes to examine potential outliers (default: 10)")
    
    args = parser.parse_args()
    
    # Check if ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return 1
    
    # Load data
    data = load_successful_fits(args.root_file, max_entries=args.max_entries)
    if data is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    if args.best_worst:
        # Create plots for best and worst fits
        print("\nUsing best/worst fit selection based on chi-squared values...")
        success_count = create_best_worst_plots(data, args.output)
        print(f"\nTotal plots created: {success_count}")
    elif args.high_amplitudes is not None:
        # Create plots for high amplitude events
        n_high_amp = args.high_amplitudes if args.high_amplitudes > 0 else 10
        print(f"\nUsing high amplitude event selection (top {n_high_amp} events)...")
        success_count = create_high_amplitude_plots(data, args.output, n_high_amp)
        print(f"\nTotal plots created: {success_count}")
    else:
        # Create individual event plots (original behavior)
        n_events = min(args.num_events, len(data['TrueX']))
        print(f"\nCreating plots for first {n_events} events...")
        
        lorentzian_success = 0
        gaussian_success = 0
        power_lorentzian_success = 0
        combined_success = 0
        
        for i in range(n_events):
            # Create Lorentzian collage plot
            lorentz_result = create_all_lorentzian_plot(i, data, args.output)
            if "Error" not in lorentz_result:
                lorentzian_success += 1
            if i % 5 == 0 or "Error" in lorentz_result:
                print(f"  {lorentz_result}")
            
            # Create Gaussian collage plot
            gauss_result = create_all_gaussian_plot(i, data, args.output)
            if "Error" not in gauss_result:
                gaussian_success += 1
            if i % 5 == 0 or "Error" in gauss_result:
                print(f"  {gauss_result}")
            
            # Create Power Lorentzian collage plot
            power_result = create_all_power_lorentzian_plot(i, data, args.output)
            if "Error" not in power_result:
                power_lorentzian_success += 1
            if i % 5 == 0 or "Error" in power_result:
                print(f"  {power_result}")
            
            # Create combined all models plot
            combined_result = create_all_models_combined_plot(i, data, args.output)
            if "Error" not in combined_result:
                combined_success += 1
            if i % 5 == 0 or "Error" in combined_result:
                print(f"  {combined_result}")
        
        print(f"\nResults:")
        print(f"  Successfully created {lorentzian_success}/{n_events} Lorentzian collage plots")
        print(f"  Successfully created {gaussian_success}/{n_events} Gaussian collage plots")
        print(f"  Successfully created {power_lorentzian_success}/{n_events} Power Lorentzian collage plots")
        print(f"  Successfully created {combined_success}/{n_events} combined all models plots")
        
        print(f"\nAll plots saved to: {args.output}/")
        print(f"  - Lorentzian collages: {args.output}/lorentzian/")
        print(f"  - Gaussian collages: {args.output}/gaussian/")
        print(f"  - Power Lorentzian collages: {args.output}/power_lorentzian/")
        print(f"  - Combined all models plots: {args.output}/all_models_combined/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 