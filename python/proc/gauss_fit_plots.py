#!/usr/bin/env python3
"""
Post-processing plotting routine for Gaussian and Lorentzian fit visualization of charge sharing in LGAD detectors.

This script creates plots for:
1. Gaussian and Lorentzian curve estimation for central row (x-direction) with residuals
2. Gaussian and Lorentzian curve estimation for central column (y-direction) with residuals
3. Gaussian curve estimation for main diagonal direction with residuals
4. Gaussian curve estimation for secondary diagonal direction with residuals
5. Comparison plots showing ALL fitting approaches overlaid:
   - X-direction: Row (Gaussian + Lorentzian) + Main Diagonal (Gaussian) + Secondary Diagonal (Gaussian)
   - Y-direction: Column (Gaussian + Lorentzian) + Main Diagonal (Gaussian) + Secondary Diagonal (Gaussian)

The plots show both fitted Gaussian and Lorentzian curves overlaid on the actual charge data points 
from the neighborhood grid, along with residual plots showing fit quality for both function types.
Individual directions get separate figures in their respective subdirectories, and comparison plots show all fitting approaches overlaid for comprehensive analysis.
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

def load_successful_fits(root_file):
    """
    Load data from ROOT file, with robust branch detection.
    
    Args:
        root_file (str): Path to ROOT file
    
    Returns:
        dict: Dictionary containing arrays of available data
    """
    print(f"Loading data from {root_file}...")
    
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
                'Fit2D_XCenterErr': 'GaussFitRowCenterErr',
                'Fit2D_XSigmaErr': 'GaussFitRowStdevErr',
                'Fit2D_XAmplitudeErr': 'GaussFitRowAmplitudeErr',
                'Fit2D_XChi2red': 'GaussFitRowChi2red',
                'Fit2D_XNPoints': 'GaussFitRowDOF',  # DOF + parameters = NPoints
                
                # Gaussian fit results - Column/Y direction  
                'Fit2D_YCenter': 'GaussFitColumnCenter',
                'Fit2D_YSigma': 'GaussFitColumnStdev',
                'Fit2D_YAmplitude': 'GaussFitColumnAmplitude', 
                'Fit2D_YCenterErr': 'GaussFitColumnCenterErr',
                'Fit2D_YSigmaErr': 'GaussFitColumnStdevErr',
                'Fit2D_YAmplitudeErr': 'GaussFitColumnAmplitudeErr',
                'Fit2D_YChi2red': 'GaussFitColumnChi2red',
                'Fit2D_YNPoints': 'GaussFitColumnDOF',
                
                # Lorentzian fit results - Row/X direction
                'Fit2D_Lorentz_XCenter': 'LorentzFitRowCenter',
                'Fit2D_Lorentz_XGamma': 'LorentzFitRowGamma',
                'Fit2D_Lorentz_XAmplitude': 'LorentzFitRowAmplitude',
                'Fit2D_Lorentz_XCenterErr': 'LorentzFitRowCenterErr',
                'Fit2D_Lorentz_XGammaErr': 'LorentzFitRowGammaErr',
                'Fit2D_Lorentz_XAmplitudeErr': 'LorentzFitRowAmplitudeErr',
                'Fit2D_Lorentz_XChi2red': 'LorentzFitRowChi2red',
                'Fit2D_Lorentz_XNPoints': 'LorentzFitRowDOF',
                
                # Lorentzian fit results - Column/Y direction
                'Fit2D_Lorentz_YCenter': 'LorentzFitColumnCenter',
                'Fit2D_Lorentz_YGamma': 'LorentzFitColumnGamma',
                'Fit2D_Lorentz_YAmplitude': 'LorentzFitColumnAmplitude',
                'Fit2D_Lorentz_YCenterErr': 'LorentzFitColumnCenterErr',
                'Fit2D_Lorentz_YGammaErr': 'LorentzFitColumnGammaErr',
                'Fit2D_Lorentz_YAmplitudeErr': 'LorentzFitColumnAmplitudeErr',
                'Fit2D_Lorentz_YChi2red': 'LorentzFitColumnChi2red',
                'Fit2D_Lorentz_YNPoints': 'LorentzFitColumnDOF',
                
                # Diagonal Gaussian fits - Main diagonal X
                'FitDiag_MainXCenter': 'GaussFitMainDiagXCenter',
                'FitDiag_MainXSigma': 'GaussFitMainDiagXStdev',
                'FitDiag_MainXAmplitude': 'GaussFitMainDiagXAmplitude',
                'FitDiag_MainXCenterErr': 'GaussFitMainDiagXCenterErr',
                'FitDiag_MainXSigmaErr': 'GaussFitMainDiagXStdevErr',
                'FitDiag_MainXAmplitudeErr': 'GaussFitMainDiagXAmplitudeErr',
                'FitDiag_MainXChi2red': 'GaussFitMainDiagXChi2red',
                'FitDiag_MainXNPoints': 'GaussFitMainDiagXDOF',
                
                # Diagonal Gaussian fits - Main diagonal Y
                'FitDiag_MainYCenter': 'GaussFitMainDiagYCenter',
                'FitDiag_MainYSigma': 'GaussFitMainDiagYStdev',
                'FitDiag_MainYAmplitude': 'GaussFitMainDiagYAmplitude',
                'FitDiag_MainYCenterErr': 'GaussFitMainDiagYCenterErr',
                'FitDiag_MainYSigmaErr': 'GaussFitMainDiagYStdevErr',
                'FitDiag_MainYAmplitudeErr': 'GaussFitMainDiagYAmplitudeErr',
                'FitDiag_MainYChi2red': 'GaussFitMainDiagYChi2red',
                'FitDiag_MainYNPoints': 'GaussFitMainDiagYDOF',
                
                # Diagonal Gaussian fits - Secondary diagonal X
                'FitDiag_SecXCenter': 'GaussFitSecondDiagXCenter',
                'FitDiag_SecXSigma': 'GaussFitSecondDiagXStdev',
                'FitDiag_SecXAmplitude': 'GaussFitSecondDiagXAmplitude',
                'FitDiag_SecXCenterErr': 'GaussFitSecondDiagXCenterErr',
                'FitDiag_SecXSigmaErr': 'GaussFitSecondDiagXStdevErr',
                'FitDiag_SecXAmplitudeErr': 'GaussFitSecondDiagXAmplitudeErr',
                'FitDiag_SecXChi2red': 'GaussFitSecondDiagXChi2red',
                'FitDiag_SecXNPoints': 'GaussFitSecondDiagXDOF',
                
                # Diagonal Gaussian fits - Secondary diagonal Y
                'FitDiag_SecYCenter': 'GaussFitSecondDiagYCenter',
                'FitDiag_SecYSigma': 'GaussFitSecondDiagYStdev',
                'FitDiag_SecYAmplitude': 'GaussFitSecondDiagYAmplitude',
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
                
                # Nearest pixel positions
                'NearestPixelX': 'NearestPixelX',
                'NearestPixelY': 'NearestPixelY',
                'NearestPixelZ': 'NearestPixelZ'
            }
            
            # Load all available branches with mapping
            data = {}
            loaded_count = 0
            
            for expected_name, actual_name in branch_mapping.items():
                if actual_name in tree.keys():
                    try:
                        data[expected_name] = tree[actual_name].array(library="np")
                        loaded_count += 1
                        if loaded_count <= 10:  # Only print first 10 to avoid spam
                            print(f"Loaded: {expected_name} -> {actual_name}")
                    except Exception as e:
                        print(f"Warning: Could not load {actual_name}: {e}")
                else:
                    print(f"Warning: Branch {actual_name} not found for {expected_name}")
            
            print(f"Successfully loaded {loaded_count} branches with {len(data['TrueX'])} events")
            
            # Create success flags since they're not individual branches in this ROOT file
            # We'll check if the chi2 values are reasonable (non-zero and finite)
            n_events = len(data['TrueX'])
            
            # Create success flags based on available fit data
            data['Fit2D_Successful'] = np.ones(n_events, dtype=bool)  # Assume all successful for now
            data['Fit2D_Lorentz_Successful'] = np.ones(n_events, dtype=bool)
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
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (row_data, col_data) where each is (positions, charges) for central row/column
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
            
            print(f"Event {event_idx}: Extracted {len(row_positions)} row points, {len(col_positions)} col points from grid")
            
        except Exception as e:
            print(f"Warning: Could not extract grid data for event {event_idx}: {e}")
            # Fall back to fit results
            row_positions, row_charge_values = extract_from_fit_results(event_idx, data, 'row')
            col_positions, col_charge_values = extract_from_fit_results(event_idx, data, 'col')
    else:
        # Fall back to fit results if grid data not available
        print(f"Warning: NonPixel_GridNeighborhoodCharge not available for event {event_idx}, using fit results")
        row_positions, row_charge_values = extract_from_fit_results(event_idx, data, 'row')
        col_positions, col_charge_values = extract_from_fit_results(event_idx, data, 'col')
    
    return (row_positions, row_charge_values), (col_positions, col_charge_values)


def extract_from_fit_results(event_idx, data, direction):
    """
    Extract data from fit results as fallback when grid data is not available.
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
                
    except Exception as e:
        print(f"Warning: Could not extract {direction} data for event {event_idx}: {e}")
        positions = np.array([])
        charge_values = np.array([])
    
    return positions, charge_values


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
                    diag_coord = offset_i * pixel_spacing  # Could also use offset_j * pixel_spacing
                    
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
                    diag_coord = offset_i * pixel_spacing  # X coordinate along the diagonal
                    
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

def calculate_residuals(positions, charges, fit_params, fit_type='gaussian'):
    """
    Calculate residuals between data and fitted function.
    
    Args:
        positions (array): Position values
        charges (array): Charge values (data)
        fit_params (dict): Fitted parameters with keys 'center', 'sigma'/'gamma', 'amplitude'
        fit_type (str): 'gaussian' or 'lorentzian'
    
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
    else:
        raise ValueError("fit_type must be 'gaussian' or 'lorentzian'")
    
    return charges - fitted_values

def create_gauss_fit_plot(event_idx, data, output_dir="plots", show_event_info=False):
    """
    Create separate Gaussian and Lorentzian fit plots for X and Y directions for a single event.
    Each direction gets its own figure with residuals on left, main plot on right.
    Saves Gaussian and Lorentzian plots to separate subdirectories.
    
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
        
        # Get Gaussian fit parameters for this event
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
        
        # Get Lorentzian fit parameters for this event
        x_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
        x_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
        x_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
        x_lorentz_center_err = data['Fit2D_Lorentz_XCenterErr'][event_idx]
        x_lorentz_gamma_err = data['Fit2D_Lorentz_XGammaErr'][event_idx]
        x_lorentz_chi2red = data['Fit2D_Lorentz_XChi2red'][event_idx]
        x_lorentz_npoints = data['Fit2D_Lorentz_XNPoints'][event_idx]
        
        y_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
        y_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
        y_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
        y_lorentz_center_err = data['Fit2D_Lorentz_YCenterErr'][event_idx]
        y_lorentz_gamma_err = data['Fit2D_Lorentz_YGammaErr'][event_idx]
        y_lorentz_chi2red = data['Fit2D_Lorentz_YChi2red'][event_idx]
        y_lorentz_npoints = data['Fit2D_Lorentz_YNPoints'][event_idx]
        
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
        
        # Create subdirectories for different fit types
        gaussian_dir = os.path.join(output_dir, "gaussian")
        lorentzian_dir = os.path.join(output_dir, "lorentzian")
        comparison_dir = os.path.join(output_dir, "comparison")
        
        os.makedirs(gaussian_dir, exist_ok=True)
        os.makedirs(lorentzian_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # ============================================
        # X-DIRECTION GAUSSIAN FIGURE
        # ============================================
        if len(row_pos) >= 3:
            fig_x = plt.figure(figsize=(16, 6))
            gs_x = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_x_res = fig_x.add_subplot(gs_x[0, 0])
            
            # Right panel: Main plot
            ax_x_main = fig_x.add_subplot(gs_x[0, 1])
            
            # Calculate fit parameters and residuals for Gaussian fit
            x_gauss_params = {'center': x_center, 'sigma': x_sigma, 'amplitude': x_amplitude}
            residuals_x_gauss = calculate_residuals(row_pos, row_charges, x_gauss_params, 'gaussian')
            
            # Residuals plot (left panel)
            ax_x_res.errorbar(row_pos, residuals_x_gauss, fmt='bo', markersize=6, 
                             capsize=3, label='Gaussian residuals', alpha=0.8)
            ax_x_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_x_res.grid(True, alpha=0.3, linestyle=':')
            ax_x_res.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(x_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_x_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Row)')
            ax_x_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_x_main.errorbar(row_pos, row_charges, fmt='ko', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            x_fit_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_gauss_fit = gaussian_1d(x_fit_range, x_amplitude, x_center, x_sigma)
            
            # Plot Gaussian fit curve
            gauss_label = r'$y(x) = A \exp\left(-\frac{(x - m)^2}{2\sigma^2}\right)+ B$'
            ax_x_main.plot(x_fit_range, y_gauss_fit, 'b-', linewidth=2.5, 
                          label=gauss_label, alpha=0.9)
            
            # Mark true position
            ax_x_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                             label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_x_main.axvline(x_center, color='blue', linestyle=':', linewidth=2, 
                             label=f'Fitted X = {x_center:.3f} mm', alpha=0.8)
            
            ax_x_main.grid(True, alpha=0.3, linestyle=':')
            ax_x_main.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_x_main.set_title('Pixel Charge vs. Central Coordinate (Row)')
            ax_x_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_x_gauss = x_center - true_x
                fig_x.suptitle(f'Event {event_idx}: X-Direction Gaussian Fit\n'
                              f'True X: {true_x:.3f} mm, Fitted: {x_center:.3f} mm (Δ={delta_x_gauss:.3f})', 
                              fontsize=14)
            
            # Save X-direction Gaussian plot
            autoscale_axes(fig_x)
            filename_x_gauss = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_X.png')
            plt.savefig(filename_x_gauss, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # X-DIRECTION LORENTZIAN FIGURE
        # ============================================
        if len(row_pos) >= 3:
            fig_x_lor = plt.figure(figsize=(16, 6))
            gs_x_lor = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_x_lor_res = fig_x_lor.add_subplot(gs_x_lor[0, 0])
            
            # Right panel: Main plot
            ax_x_lor_main = fig_x_lor.add_subplot(gs_x_lor[0, 1])
            
            # Calculate fit parameters and residuals for Lorentzian fit
            x_lorentz_params = {'center': x_lorentz_center, 'gamma': x_lorentz_gamma, 'amplitude': x_lorentz_amplitude}
            residuals_x_lorentz = calculate_residuals(row_pos, row_charges, x_lorentz_params, 'lorentzian')
            
            # Residuals plot (left panel)
            ax_x_lor_res.errorbar(row_pos, residuals_x_lorentz, fmt='rs', markersize=6, 
                                 capsize=3, label='Lorentzian residuals', alpha=0.8)
            ax_x_lor_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_x_lor_res.grid(True, alpha=0.3, linestyle=':')
            ax_x_lor_res.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_lor_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(x_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_x_lor_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Row)')
            ax_x_lor_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_x_lor_main.errorbar(row_pos, row_charges, fmt='ko', markersize=8, 
                                  capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Lorentzian
            x_fit_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_lorentz_fit = lorentzian_1d(x_fit_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma)
            
            # Plot Lorentzian fit curve
            lorentz_label = r'$y(x) = \frac{A}{1+\left(\frac{x-m}{\gamma}\right)^2} + B$'
            ax_x_lor_main.plot(x_fit_range, y_lorentz_fit, 'r-', linewidth=2.5, 
                              label=lorentz_label, alpha=0.9)
            
            # Mark true position
            ax_x_lor_main.axvline(true_x, color='green', linestyle='--', linewidth=2, 
                                 label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_x_lor_main.axvline(x_lorentz_center, color='red', linestyle=':', linewidth=2, 
                                 label=f'Fitted X = {x_lorentz_center:.3f} mm', alpha=0.8)
            
            ax_x_lor_main.grid(True, alpha=0.3, linestyle=':')
            ax_x_lor_main.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_x_lor_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_x_lor_main.set_title('Pixel Charge vs. Central Coordinate (Row)')
            ax_x_lor_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_x_lorentz = x_lorentz_center - true_x
                fig_x_lor.suptitle(f'Event {event_idx}: X-Direction Lorentzian Fit\n'
                                  f'True X: {true_x:.3f} mm, Fitted: {x_lorentz_center:.3f} mm (Δ={delta_x_lorentz:.3f})', 
                                  fontsize=14)
            
            # Save X-direction Lorentzian plot
            autoscale_axes(fig_x_lor)
            filename_x_lorentz = os.path.join(lorentzian_dir, f'lorentz_fit_event_{event_idx:04d}_X.png')
            plt.savefig(filename_x_lorentz, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # Y-DIRECTION GAUSSIAN FIGURE
        # ============================================
        if len(col_pos) >= 3:
            fig_y = plt.figure(figsize=(16, 6))
            gs_y = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_y_res = fig_y.add_subplot(gs_y[0, 0])
            
            # Right panel: Main plot
            ax_y_main = fig_y.add_subplot(gs_y[0, 1])
            
            # Calculate fit parameters and residuals for Gaussian fit
            y_gauss_params = {'center': y_center, 'sigma': y_sigma, 'amplitude': y_amplitude}
            residuals_y_gauss = calculate_residuals(col_pos, col_charges, y_gauss_params, 'gaussian')
            
            # Residuals plot (left panel)
            ax_y_res.errorbar(col_pos, residuals_y_gauss, fmt='bo', markersize=6, 
                             capsize=3, label='Gaussian residuals', alpha=0.8)
            ax_y_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_y_res.grid(True, alpha=0.3, linestyle=':')
            ax_y_res.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(y_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_y_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Column)')
            ax_y_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_y_main.errorbar(col_pos, col_charges, fmt='ko', markersize=8, 
                              capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Gaussian
            y_fit_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_gauss_fit = gaussian_1d(y_fit_range, y_amplitude, y_center, y_sigma)
            
            # Plot Gaussian fit curve
            gauss_label = r'$y(x) = A \exp\left(-\frac{(y - m)^2}{2\sigma^2}\right)+ B$'
            ax_y_main.plot(y_fit_range, y_gauss_fit, 'b-', linewidth=2.5, 
                          label=gauss_label, alpha=0.9)
            
            # Mark true position
            ax_y_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                             label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_y_main.axvline(y_center, color='blue', linestyle=':', linewidth=2, 
                             label=f'Fitted Y = {y_center:.3f} mm', alpha=0.8)
            
            ax_y_main.grid(True, alpha=0.3, linestyle=':')
            ax_y_main.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_y_main.set_title('Pixel Charge vs. Central Coordinate (Column)')
            ax_y_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_y_gauss = y_center - true_y
                fig_y.suptitle(f'Event {event_idx}: Y-Direction Gaussian Fit\n'
                              f'True Y: {true_y:.3f} mm, Fitted: {y_center:.3f} mm (Δ={delta_y_gauss:.3f})', 
                              fontsize=14)
            
            # Save Y-direction Gaussian plot
            autoscale_axes(fig_y)
            filename_y_gauss = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_Y.png')
            plt.savefig(filename_y_gauss, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # Y-DIRECTION LORENTZIAN FIGURE
        # ============================================
        if len(col_pos) >= 3:
            fig_y_lor = plt.figure(figsize=(16, 6))
            gs_y_lor = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            # Left panel: Residuals
            ax_y_lor_res = fig_y_lor.add_subplot(gs_y_lor[0, 0])
            
            # Right panel: Main plot
            ax_y_lor_main = fig_y_lor.add_subplot(gs_y_lor[0, 1])
            
            # Calculate fit parameters and residuals for Lorentzian fit
            y_lorentz_params = {'center': y_lorentz_center, 'gamma': y_lorentz_gamma, 'amplitude': y_lorentz_amplitude}
            residuals_y_lorentz = calculate_residuals(col_pos, col_charges, y_lorentz_params, 'lorentzian')
            
            # Residuals plot (left panel)
            ax_y_lor_res.errorbar(col_pos, residuals_y_lorentz, fmt='rs', markersize=6, 
                                 capsize=3, label='Lorentzian residuals', alpha=0.8)
            ax_y_lor_res.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax_y_lor_res.grid(True, alpha=0.3, linestyle=':')
            ax_y_lor_res.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_lor_res.set_ylabel(r'$q_{\mathrm{px}} - \mathrm{Fit}(y_{\mathrm{px}})\,(\mathrm{C})$', fontsize=12)
            ax_y_lor_res.set_title('Residuals of Pixel Charge vs. Central Coordinate (Column)')
            ax_y_lor_res.legend(loc='upper right')
            
            # Main plot (right panel)
            # Plot data points with error bars
            ax_y_lor_main.errorbar(col_pos, col_charges, fmt='ko', markersize=8, 
                                  capsize=4, label='Measured data', alpha=0.8, linewidth=1.5)
            
            # Create smooth curve for fitted Lorentzian
            y_fit_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_lorentz_fit = lorentzian_1d(y_fit_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma)
            
            # Plot Lorentzian fit curve
            lorentz_label = r'$y(x) = \frac{A}{1+\left(\frac{y-m}{\gamma}\right)^2} + B$'
            ax_y_lor_main.plot(y_fit_range, y_lorentz_fit, 'r-', linewidth=2.5, 
                              label=lorentz_label, alpha=0.9)
            
            # Mark true position
            ax_y_lor_main.axvline(true_y, color='green', linestyle='--', linewidth=2, 
                                 label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted center
            ax_y_lor_main.axvline(y_lorentz_center, color='red', linestyle=':', linewidth=2, 
                                 label=f'Fitted Y = {y_lorentz_center:.3f} mm', alpha=0.8)
            
            ax_y_lor_main.grid(True, alpha=0.3, linestyle=':')
            ax_y_lor_main.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=12)
            ax_y_lor_main.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=12)
            ax_y_lor_main.set_title('Pixel Charge vs. Central Coordinate (Column)')
            ax_y_lor_main.legend(loc='upper left')
            
            # Add overall title with event information
            if show_event_info:
                delta_y_lorentz = y_lorentz_center - true_y
                fig_y_lor.suptitle(f'Event {event_idx}: Y-Direction Lorentzian Fit\n'
                                  f'True Y: {true_y:.3f} mm, Fitted: {y_lorentz_center:.3f} mm (Δ={delta_y_lorentz:.3f})', 
                                  fontsize=14)
            
            # Save Y-direction Lorentzian plot
            autoscale_axes(fig_y_lor)
            filename_y_lorentz = os.path.join(lorentzian_dir, f'lorentz_fit_event_{event_idx:04d}_Y.png')
            plt.savefig(filename_y_lorentz, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # MAIN DIAGONAL X FIGURE (X vs Charge for main diagonal pixels) - GAUSSIAN ONLY
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
            autoscale_axes(fig_main_x)
            filename_main_x = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_MainDiagX.png')
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
            autoscale_axes(fig_main_y)
            filename_main_y = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_MainDiagY.png')
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
            autoscale_axes(fig_sec_x)
            filename_sec_x = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_SecDiagX.png')
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
            autoscale_axes(fig_sec_y)
            filename_sec_y = os.path.join(gaussian_dir, f'gauss_fit_event_{event_idx:04d}_SecDiagY.png')
            plt.savefig(filename_sec_y, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Generate success message
        gaussian_plots = []
        lorentzian_plots = []
        
        if len(row_pos) >= 3:
            gaussian_plots.append('X-direction')
            lorentzian_plots.append('X-direction')
        if len(col_pos) >= 3:
            gaussian_plots.append('Y-direction')
            lorentzian_plots.append('Y-direction')
        if len(main_x_pos) >= 3 and main_diag_x_successful:
            gaussian_plots.append('Main-diagonal-X')
        if len(main_y_pos) >= 3 and main_diag_y_successful:
            gaussian_plots.append('Main-diagonal-Y')
        if len(sec_x_pos) >= 3 and sec_diag_x_successful:
            gaussian_plots.append('Secondary-diagonal-X')
        if len(sec_y_pos) >= 3 and sec_diag_y_successful:
            gaussian_plots.append('Secondary-diagonal-Y')
        
        success_msg = f"Event {event_idx}: "
        if gaussian_plots:
            success_msg += f"Gaussian plots ({', '.join(gaussian_plots)}) "
        if lorentzian_plots:
            success_msg += f"and Lorentzian plots ({', '.join(lorentzian_plots)}) "
        success_msg += "saved to respective directories"
        
        return success_msg
        
    except Exception as e:
        return f"Event {event_idx}: Error creating plot - {e}"

def create_overlay_fit_plots(event_idx, data, output_dir="plots", show_event_info=False):
    """
    Create comparison plots showing ALL fitting approaches overlaid for each direction.
    
    X-direction: Row (Gaussian + Lorentzian), Main Diagonal X (Gaussian), Secondary Diagonal X (Gaussian)
    Y-direction: Column (Gaussian + Lorentzian), Main Diagonal Y (Gaussian), Secondary Diagonal Y (Gaussian)
    
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
        
        # True hit position for comparison
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        # Create comparison directory
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Get fit parameters for this event
        # Row/Column Gaussian fits
        row_gauss_center = data['Fit2D_XCenter'][event_idx]
        row_gauss_sigma = data['Fit2D_XSigma'][event_idx]
        row_gauss_amplitude = data['Fit2D_XAmplitude'][event_idx]
        row_gauss_chi2 = data['Fit2D_XChi2red'][event_idx]
        
        col_gauss_center = data['Fit2D_YCenter'][event_idx]
        col_gauss_sigma = data['Fit2D_YSigma'][event_idx]
        col_gauss_amplitude = data['Fit2D_YAmplitude'][event_idx]
        col_gauss_chi2 = data['Fit2D_YChi2red'][event_idx]
        
        # Row/Column Lorentzian fits
        row_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
        row_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
        row_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
        row_lorentz_chi2 = data['Fit2D_Lorentz_XChi2red'][event_idx]
        
        col_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
        col_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
        col_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
        col_lorentz_chi2 = data['Fit2D_Lorentz_YChi2red'][event_idx]
        
        # Diagonal Gaussian fits
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx]
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_chi2 = data['FitDiag_MainXChi2red'][event_idx]
        main_diag_x_successful = data['FitDiag_MainXSuccessful'][event_idx]
        
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx]
        main_diag_y_chi2 = data['FitDiag_MainYChi2red'][event_idx]
        main_diag_y_successful = data['FitDiag_MainYSuccessful'][event_idx]
        
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_chi2 = data['FitDiag_SecXChi2red'][event_idx]
        sec_diag_x_successful = data['FitDiag_SecXSuccessful'][event_idx]
        
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_chi2 = data['FitDiag_SecYChi2red'][event_idx]
        sec_diag_y_successful = data['FitDiag_SecYSuccessful'][event_idx]
        
        # ============================================
        # X-COORDINATE COMPARISON: ALL APPROACHES OVERLAID
        # ============================================
        if len(row_pos) >= 3:
            fig_x_comparison = plt.figure(figsize=(16, 8))
            ax_x_comparison = fig_x_comparison.add_subplot(111)
            
            # Determine overall X range for plotting from all data sources
            all_x_positions = list(row_pos)
            if len(main_x_pos) >= 3 and main_diag_x_successful:
                all_x_positions.extend(main_x_pos)
            if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                all_x_positions.extend(sec_x_pos)
            
            x_min, x_max = min(all_x_positions), max(all_x_positions)
            x_range = np.linspace(x_min - 0.3, x_max + 0.3, 400)
            
            # Plot Row data points and fits
            ax_x_comparison.errorbar(row_pos, row_charges, fmt='o', markersize=6, 
                                   capsize=3, label='Row data', alpha=0.7, linewidth=1.5, 
                                   color='darkblue', markeredgecolor='navy')
            
            # Row Gaussian fit
            row_gauss_fit = gaussian_1d(x_range, row_gauss_amplitude, row_gauss_center, row_gauss_sigma)
            ax_x_comparison.plot(x_range, row_gauss_fit, '-', linewidth=2.5, 
                               label=f'Row Gaussian (χ²={row_gauss_chi2:.2f})', 
                               color='blue', alpha=0.9)
            
            # Row Lorentzian fit
            row_lorentz_fit = lorentzian_1d(x_range, row_lorentz_amplitude, row_lorentz_center, row_lorentz_gamma)
            ax_x_comparison.plot(x_range, row_lorentz_fit, '--', linewidth=2.5, 
                               label=f'Row Lorentzian (χ²={row_lorentz_chi2:.2f})', 
                               color='red', alpha=0.9)
            
            # Main diagonal X data and fit (if available)
            if len(main_x_pos) >= 3 and main_diag_x_successful:
                ax_x_comparison.errorbar(main_x_pos, main_x_charges, fmt='s', markersize=6, 
                                       capsize=3, label='Main diag data', alpha=0.7, linewidth=1.5,
                                       color='darkgreen', markeredgecolor='forestgreen')
                
                main_diag_x_fit = gaussian_1d(x_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma)
                ax_x_comparison.plot(x_range, main_diag_x_fit, '-.', linewidth=2.5, 
                                   label=f'Main Diag Gaussian (χ²={main_diag_x_chi2:.2f})', 
                                   color='green', alpha=0.9)
            
            # Secondary diagonal X data and fit (if available)
            if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                ax_x_comparison.errorbar(sec_x_pos, sec_x_charges, fmt='^', markersize=6, 
                                       capsize=3, label='Sec diag data', alpha=0.7, linewidth=1.5,
                                       color='darkorange', markeredgecolor='orangered')
                
                sec_diag_x_fit = gaussian_1d(x_range, sec_diag_x_amplitude, sec_diag_x_center, sec_diag_x_sigma)
                ax_x_comparison.plot(x_range, sec_diag_x_fit, ':', linewidth=2.5, 
                                   label=f'Sec Diag Gaussian (χ²={sec_diag_x_chi2:.2f})', 
                                   color='orange', alpha=0.9)
            
            # Mark true position
            ax_x_comparison.axvline(true_x, color='black', linestyle='-', linewidth=3, 
                                  label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            # Mark fitted centers with vertical lines
            ax_x_comparison.axvline(row_gauss_center, color='blue', linestyle=':', linewidth=1.5, 
                                  alpha=0.6, label=f'Row Gauss center = {row_gauss_center:.3f} mm')
            ax_x_comparison.axvline(row_lorentz_center, color='red', linestyle=':', linewidth=1.5, 
                                  alpha=0.6, label=f'Row Lorentz center = {row_lorentz_center:.3f} mm')
            
            if len(main_x_pos) >= 3 and main_diag_x_successful:
                ax_x_comparison.axvline(main_diag_x_center, color='green', linestyle=':', linewidth=1.5, 
                                      alpha=0.6, label=f'Main Diag center = {main_diag_x_center:.3f} mm')
            
            if len(sec_x_pos) >= 3 and sec_diag_x_successful:
                ax_x_comparison.axvline(sec_diag_x_center, color='orange', linestyle=':', linewidth=1.5, 
                                      alpha=0.6, label=f'Sec Diag center = {sec_diag_x_center:.3f} mm')
            
            ax_x_comparison.grid(True, alpha=0.3, linestyle=':')
            ax_x_comparison.set_xlabel(r'$x_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=14)
            ax_x_comparison.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=14)
            ax_x_comparison.set_title('X-Direction: All Fitting Approaches Comparison', fontsize=16)
            ax_x_comparison.legend(loc='best', fontsize=10, ncol=2)
            
            # Add overall title with event information
            if show_event_info:
                delta_row_gauss = row_gauss_center - true_x
                delta_row_lorentz = row_lorentz_center - true_x
                fig_x_comparison.suptitle(f'Event {event_idx}: X-Direction All Fits Comparison\n'
                                        f'True: {true_x:.3f} mm, Row G: {row_gauss_center:.3f} (Δ={delta_row_gauss:.3f}), '
                                        f'Row L: {row_lorentz_center:.3f} (Δ={delta_row_lorentz:.3f})', 
                                        fontsize=14)
            
            # Save X-direction comparison plot
            autoscale_axes(fig_x_comparison)
            filename_x_comparison = os.path.join(comparison_dir, f'comparison_event_{event_idx:04d}_X_AllApproaches.png')
            plt.savefig(filename_x_comparison, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # Y-COORDINATE COMPARISON: ALL APPROACHES OVERLAID
        # ============================================
        if len(col_pos) >= 3:
            fig_y_comparison = plt.figure(figsize=(16, 8))
            ax_y_comparison = fig_y_comparison.add_subplot(111)
            
            # Determine overall Y range for plotting from all data sources
            all_y_positions = list(col_pos)
            if len(main_y_pos) >= 3 and main_diag_y_successful:
                all_y_positions.extend(main_y_pos)
            if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                all_y_positions.extend(sec_y_pos)
            
            y_min, y_max = min(all_y_positions), max(all_y_positions)
            y_range = np.linspace(y_min - 0.3, y_max + 0.3, 400)
            
            # Plot Column data points and fits
            ax_y_comparison.errorbar(col_pos, col_charges, fmt='o', markersize=6, 
                                   capsize=3, label='Column data', alpha=0.7, linewidth=1.5, 
                                   color='darkblue', markeredgecolor='navy')
            
            # Column Gaussian fit
            col_gauss_fit = gaussian_1d(y_range, col_gauss_amplitude, col_gauss_center, col_gauss_sigma)
            ax_y_comparison.plot(y_range, col_gauss_fit, '-', linewidth=2.5, 
                               label=f'Column Gaussian (χ²={col_gauss_chi2:.2f})', 
                               color='blue', alpha=0.9)
            
            # Column Lorentzian fit
            col_lorentz_fit = lorentzian_1d(y_range, col_lorentz_amplitude, col_lorentz_center, col_lorentz_gamma)
            ax_y_comparison.plot(y_range, col_lorentz_fit, '--', linewidth=2.5, 
                               label=f'Column Lorentzian (χ²={col_lorentz_chi2:.2f})', 
                               color='red', alpha=0.9)
            
            # Main diagonal Y data and fit (if available)
            if len(main_y_pos) >= 3 and main_diag_y_successful:
                ax_y_comparison.errorbar(main_y_pos, main_y_charges, fmt='s', markersize=6, 
                                       capsize=3, label='Main diag data', alpha=0.7, linewidth=1.5,
                                       color='darkgreen', markeredgecolor='forestgreen')
                
                main_diag_y_fit = gaussian_1d(y_range, main_diag_y_amplitude, main_diag_y_center, main_diag_y_sigma)
                ax_y_comparison.plot(y_range, main_diag_y_fit, '-.', linewidth=2.5, 
                                   label=f'Main Diag Gaussian (χ²={main_diag_y_chi2:.2f})', 
                                   color='green', alpha=0.9)
            
            # Secondary diagonal Y data and fit (if available)
            if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                ax_y_comparison.errorbar(sec_y_pos, sec_y_charges, fmt='^', markersize=6, 
                                       capsize=3, label='Sec diag data', alpha=0.7, linewidth=1.5,
                                       color='darkorange', markeredgecolor='orangered')
                
                sec_diag_y_fit = gaussian_1d(y_range, sec_diag_y_amplitude, sec_diag_y_center, sec_diag_y_sigma)
                ax_y_comparison.plot(y_range, sec_diag_y_fit, ':', linewidth=2.5, 
                                   label=f'Sec Diag Gaussian (χ²={sec_diag_y_chi2:.2f})', 
                                   color='orange', alpha=0.9)
            
            # Mark true position
            ax_y_comparison.axvline(true_y, color='black', linestyle='-', linewidth=3, 
                                  label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            # Mark fitted centers with vertical lines
            ax_y_comparison.axvline(col_gauss_center, color='blue', linestyle=':', linewidth=1.5, 
                                  alpha=0.6, label=f'Col Gauss center = {col_gauss_center:.3f} mm')
            ax_y_comparison.axvline(col_lorentz_center, color='red', linestyle=':', linewidth=1.5, 
                                  alpha=0.6, label=f'Col Lorentz center = {col_lorentz_center:.3f} mm')
            
            if len(main_y_pos) >= 3 and main_diag_y_successful:
                ax_y_comparison.axvline(main_diag_y_center, color='green', linestyle=':', linewidth=1.5, 
                                      alpha=0.6, label=f'Main Diag center = {main_diag_y_center:.3f} mm')
            
            if len(sec_y_pos) >= 3 and sec_diag_y_successful:
                ax_y_comparison.axvline(sec_diag_y_center, color='orange', linestyle=':', linewidth=1.5, 
                                      alpha=0.6, label=f'Sec Diag center = {sec_diag_y_center:.3f} mm')
            
            ax_y_comparison.grid(True, alpha=0.3, linestyle=':')
            ax_y_comparison.set_xlabel(r'$y_{\mathrm{px}}\,(\mathrm{mm})$', fontsize=14)
            ax_y_comparison.set_ylabel(r'$q_{\mathrm{px}}\,(\mathrm{C})$', fontsize=14)
            ax_y_comparison.set_title('Y-Direction: All Fitting Approaches Comparison', fontsize=16)
            ax_y_comparison.legend(loc='best', fontsize=10, ncol=2)
            
            # Add overall title with event information
            if show_event_info:
                delta_col_gauss = col_gauss_center - true_y
                delta_col_lorentz = col_lorentz_center - true_y
                fig_y_comparison.suptitle(f'Event {event_idx}: Y-Direction All Fits Comparison\n'
                                        f'True: {true_y:.3f} mm, Col G: {col_gauss_center:.3f} (Δ={delta_col_gauss:.3f}), '
                                        f'Col L: {col_lorentz_center:.3f} (Δ={delta_col_lorentz:.3f})', 
                                        fontsize=14)
            
            # Save Y-direction comparison plot
            autoscale_axes(fig_y_comparison)
            filename_y_comparison = os.path.join(comparison_dir, f'comparison_event_{event_idx:04d}_Y_AllApproaches.png')
            plt.savefig(filename_y_comparison, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Generate success message for comparison plots
        created_comparisons = []
        if len(row_pos) >= 3:
            created_comparisons.append('X-direction All-Approaches')
        if len(col_pos) >= 3:
            created_comparisons.append('Y-direction All-Approaches')
        
        if created_comparisons:
            return f"Event {event_idx}: {' and '.join(created_comparisons)} comparison plots saved to comparison directory"
        else:
            return f"Event {event_idx}: No comparison plots created (insufficient data)"
        
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
        autoscale_axes(fig)
        # Save summary plot
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, 'gauss_lorentz_fit_summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Summary plots saved to {filename}"
        
    except Exception as e:
        return f"Error creating summary plots: {e}"

def autoscale_axes(fig):
    """Rescale x and y limits for all axes in the given figure.

    This calls relim() and autoscale_view() on every Axes object so that the
    limits are always determined from the current artist data. It should be
    invoked after all plotting commands for a figure and right before the
    figure is saved.

    Args:
        fig (matplotlib.figure.Figure): Figure whose axes need rescaling.
    """
    for ax in fig.get_axes():
        ax.relim()
        ax.autoscale_view()

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
                       help="Number of individual events to plot")
    parser.add_argument("--summary_only", action="store_true",
                       help="Create only summary plots, skip individual events")
    parser.add_argument("--overlay_only", action="store_true",
                       help="Create only comparison plots (all approaches overlaid), skip individual direction plots")
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
            
            # Create comparison plots (Gaussian vs Lorentzian)
            comparison_result = create_overlay_fit_plots(i, data, args.output)
            if "Error" not in comparison_result:
                overlay_success_count += 1
            if i % 5 == 0 or "Error" in comparison_result:
                print(f"  {comparison_result}")
        
        if not args.overlay_only:
            print(f"\nSuccessfully created {success_count}/{n_events} individual plots")
        print(f"Successfully created {overlay_success_count}/{n_events} comparison plots")
    
    print(f"\nAll plots saved to: {args.output}/")
    print(f"  - Gaussian fits: {args.output}/gaussian/")
    print(f"  - Lorentzian fits: {args.output}/lorentzian/")
    print(f"  - Comparison plots (all approaches overlaid): {args.output}/comparison/")
    print(f"  - Summary plots: {args.output}/")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 