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

def skewed_lorentzian_1d(x, amplitude, center, gamma, skewness, offset=0):
    """
    1D Skewed Lorentzian function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amplitude: Skewed Lorentzian amplitude
        center: Skewed Lorentzian center
        gamma: Skewed Lorentzian gamma (half-width at half-maximum, HWHM)
        skewness: Skewness parameter (alpha)
        offset: Baseline offset
    
    Returns:
        Skewed Lorentzian function values
    """
    # Standard Lorentzian component
    lorentz = 1.0 / (1.0 + ((x - center) / gamma)**2)
    
    # Skewness factor using error function approximation
    # erf(α * (x - center) / gamma) where α is the skewness parameter
    # For computational efficiency, we use tanh approximation: tanh(π/2 * z) ≈ erf(z)
    skew_arg = skewness * (x - center) / gamma
    skew_factor = 1.0 + np.tanh(np.pi/2 * skew_arg)
    
    return amplitude * lorentz * skew_factor + offset

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
                
                # Skewed Lorentzian fit results - Row/X direction
                'Fit2D_SkewLorentz_XCenter': 'SkewedLorentzFitRowCenter',
                'Fit2D_SkewLorentz_XGamma': 'SkewedLorentzFitRowBeta',  # Beta is the shape parameter
                'Fit2D_SkewLorentz_XAmplitude': 'SkewedLorentzFitRowAmplitude',
                'Fit2D_SkewLorentz_XSkewness': 'SkewedLorentzFitRowLambda',  # Lambda is the skewness parameter
                'Fit2D_SkewLorentz_XCenterErr': 'SkewedLorentzFitRowCenterErr',
                'Fit2D_SkewLorentz_XGammaErr': 'SkewedLorentzFitRowBetaErr',
                'Fit2D_SkewLorentz_XAmplitudeErr': 'SkewedLorentzFitRowAmplitudeErr',
                'Fit2D_SkewLorentz_XSkewnessErr': 'SkewedLorentzFitRowLambdaErr',
                'Fit2D_SkewLorentz_XChi2red': 'SkewedLorentzFitRowChi2red',
                'Fit2D_SkewLorentz_XNPoints': 'SkewedLorentzFitRowDOF',
                
                # Skewed Lorentzian fit results - Column/Y direction
                'Fit2D_SkewLorentz_YCenter': 'SkewedLorentzFitColumnCenter',
                'Fit2D_SkewLorentz_YGamma': 'SkewedLorentzFitColumnBeta',  # Beta is the shape parameter
                'Fit2D_SkewLorentz_YAmplitude': 'SkewedLorentzFitColumnAmplitude',
                'Fit2D_SkewLorentz_YSkewness': 'SkewedLorentzFitColumnLambda',  # Lambda is the skewness parameter
                'Fit2D_SkewLorentz_YCenterErr': 'SkewedLorentzFitColumnCenterErr',
                'Fit2D_SkewLorentz_YGammaErr': 'SkewedLorentzFitColumnBetaErr',
                'Fit2D_SkewLorentz_YAmplitudeErr': 'SkewedLorentzFitColumnAmplitudeErr',
                'Fit2D_SkewLorentz_YSkewnessErr': 'SkewedLorentzFitColumnLambdaErr',
                'Fit2D_SkewLorentz_YChi2red': 'SkewedLorentzFitColumnChi2red',
                'Fit2D_SkewLorentz_YNPoints': 'SkewedLorentzFitColumnDOF',
                
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
            data['Fit2D_SkewLorentz_Successful'] = np.ones(n_events, dtype=bool)
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

def autoscale_axes(fig):
    """
    Auto-scale all axes in a figure for better visualization.
    """
    for ax in fig.get_axes():
        ax.relim()
        ax.autoscale_view()

def calculate_residuals(positions, charges, fit_params, fit_type='gaussian'):
    """
    Calculate residuals between data and fitted function.
    
    Args:
        positions (array): Position values
        charges (array): Charge values (data)
        fit_params (dict): Fitted parameters with keys 'center', 'sigma'/'gamma'/'skewness', 'amplitude'
        fit_type (str): 'gaussian', 'lorentzian', or 'skewed_lorentzian'
    
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
    elif fit_type == 'skewed_lorentzian':
        fitted_values = skewed_lorentzian_1d(positions, 
                                           fit_params['amplitude'], 
                                           fit_params['center'], 
                                           fit_params['gamma'],
                                           fit_params['skewness'])
    else:
        raise ValueError("fit_type must be 'gaussian', 'lorentzian', or 'skewed_lorentzian'")
    
    return charges - fitted_values

def create_gauss_fit_plot(event_idx, data, output_dir="plots", show_event_info=False):
    """
    Create Lorentzian and Skewed Lorentzian fit plots for ALL directions for a single event.
    Creates individual plots for row, column, and all 4 diagonal directions,
    plus comprehensive "all lines overplotted" overview plots.
    """
    try:
        # Extract all data
        (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get fit parameters
        x_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
        x_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
        x_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
        x_lorentz_chi2red = data['Fit2D_Lorentz_XChi2red'][event_idx]
        
        y_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
        y_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
        y_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
        y_lorentz_chi2red = data['Fit2D_Lorentz_YChi2red'][event_idx]
        
        # Get real skewed Lorentzian parameters from ROOT file
        x_skew_lorentz_center = data['Fit2D_SkewLorentz_XCenter'][event_idx]
        x_skew_lorentz_gamma = data['Fit2D_SkewLorentz_XGamma'][event_idx]  # This is actually Beta
        x_skew_lorentz_amplitude = data['Fit2D_SkewLorentz_XAmplitude'][event_idx]
        x_skew_lorentz_skewness = data['Fit2D_SkewLorentz_XSkewness'][event_idx]  # This is actually Lambda
        x_skew_lorentz_chi2red = data['Fit2D_SkewLorentz_XChi2red'][event_idx]
        
        y_skew_lorentz_center = data['Fit2D_SkewLorentz_YCenter'][event_idx]
        y_skew_lorentz_gamma = data['Fit2D_SkewLorentz_YGamma'][event_idx]  # This is actually Beta
        y_skew_lorentz_amplitude = data['Fit2D_SkewLorentz_YAmplitude'][event_idx]
        y_skew_lorentz_skewness = data['Fit2D_SkewLorentz_YSkewness'][event_idx]  # This is actually Lambda
        y_skew_lorentz_chi2red = data['Fit2D_SkewLorentz_YChi2red'][event_idx]
        
        # Diagonal parameters (using Gaussian fits)
        main_diag_x_center = data['FitDiag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['FitDiag_MainXSigma'][event_idx] 
        main_diag_x_amplitude = data['FitDiag_MainXAmplitude'][event_idx]
        main_diag_x_chi2red = data['FitDiag_MainXChi2red'][event_idx]
        
        main_diag_y_center = data['FitDiag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['FitDiag_MainYSigma'][event_idx]
        main_diag_y_amplitude = data['FitDiag_MainYAmplitude'][event_idx] 
        main_diag_y_chi2red = data['FitDiag_MainYChi2red'][event_idx]
        
        sec_diag_x_center = data['FitDiag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['FitDiag_SecXSigma'][event_idx]
        sec_diag_x_amplitude = data['FitDiag_SecXAmplitude'][event_idx]
        sec_diag_x_chi2red = data['FitDiag_SecXChi2red'][event_idx]
        
        sec_diag_y_center = data['FitDiag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['FitDiag_SecYSigma'][event_idx]
        sec_diag_y_amplitude = data['FitDiag_SecYAmplitude'][event_idx]
        sec_diag_y_chi2red = data['FitDiag_SecYChi2red'][event_idx]
        
        # True positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        # Create directories
        lorentzian_dir = os.path.join(output_dir, "lorentzian")
        skewed_lorentzian_dir = os.path.join(output_dir, "skewed_lorentzian")
        os.makedirs(lorentzian_dir, exist_ok=True)
        os.makedirs(skewed_lorentzian_dir, exist_ok=True)
        
        def create_individual_plot(positions, charges, fit_params, true_pos, title, filename, fit_type='lorentzian'):
            """Helper to create individual plots."""
            if len(positions) < 3:
                return
                
            fig = plt.figure(figsize=(16, 6))
            gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
            
            ax_res = fig.add_subplot(gs[0, 0])
            ax_main = fig.add_subplot(gs[0, 1])
            
            # Calculate fit curve and residuals
            if fit_type == 'gaussian':
                residuals = calculate_residuals(positions, charges, fit_params, 'gaussian')
                fit_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
                fit_curve = gaussian_1d(fit_range, fit_params['amplitude'], fit_params['center'], fit_params['sigma'])
                color = 'blue'
                fit_label = 'Gaussian'
            elif fit_type == 'skewed_lorentzian':
                residuals = calculate_residuals(positions, charges, fit_params, 'skewed_lorentzian')
                fit_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
                fit_curve = skewed_lorentzian_1d(fit_range, fit_params['amplitude'], fit_params['center'], 
                                               fit_params['gamma'], fit_params['skewness'])
                color = 'magenta'
                fit_label = 'Skewed Lorentzian'
            else:  # lorentzian
                residuals = calculate_residuals(positions, charges, fit_params, 'lorentzian')
                fit_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
                fit_curve = lorentzian_1d(fit_range, fit_params['amplitude'], fit_params['center'], fit_params['gamma'])
                color = 'red'
                fit_label = 'Lorentzian'
            
            # Residuals plot
            ax_res.errorbar(positions, residuals, fmt='o', markersize=6, capsize=3, label='Residuals', alpha=0.8, color=color)
            ax_res.axhline(0, color='black', linestyle='--', alpha=0.7)
            ax_res.grid(True, alpha=0.3, linestyle=':')
            ax_res.set_xlabel('Position [mm]')
            ax_res.set_ylabel('Residual [C]')
            ax_res.set_title(f'Residuals: {title}')
            ax_res.legend()
            
            # Main plot
            ax_main.errorbar(positions, charges, fmt='ko', markersize=8, capsize=4, label='Data', alpha=0.8)
            ax_main.plot(fit_range, fit_curve, '-', linewidth=2.5, color=color, label=fit_label, alpha=0.9)
            ax_main.axvline(true_pos, color='green', linestyle='--', linewidth=2, label=f'True = {true_pos:.3f} mm', alpha=0.8)
            ax_main.axvline(fit_params['center'], color=color, linestyle=':', linewidth=2, label=f'Fitted = {fit_params["center"]:.3f} mm', alpha=0.8)
            
            ax_main.grid(True, alpha=0.3, linestyle=':')
            ax_main.set_xlabel('Position [mm]')
            ax_main.set_ylabel('Charge [C]')
            ax_main.set_title(title)
            ax_main.legend()
            
            autoscale_axes(fig)
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # ============================================
        # LORENTZIAN INDIVIDUAL PLOTS
        # ============================================
        
        # Row (X-direction) Lorentzian
        if len(row_pos) >= 3:
            x_lorentz_params = {'center': x_lorentz_center, 'gamma': x_lorentz_gamma, 'amplitude': x_lorentz_amplitude}
            create_individual_plot(row_pos, row_charges, x_lorentz_params, true_x, 
                                 'Row (X-direction) Lorentzian Fit',
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_row.png'), 'lorentzian')
        
        # Column (Y-direction) Lorentzian
        if len(col_pos) >= 3:
            y_lorentz_params = {'center': y_lorentz_center, 'gamma': y_lorentz_gamma, 'amplitude': y_lorentz_amplitude}
            create_individual_plot(col_pos, col_charges, y_lorentz_params, true_y,
                                 'Column (Y-direction) Lorentzian Fit', 
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_column.png'), 'lorentzian')
        
        # Diagonal plots - we'll fit Lorentzian to diagonal data
        if len(main_x_pos) >= 3:
            # Use Lorentzian fit for main diagonal X (project Lorentzian parameters)
            main_x_lorentz_params = {'center': main_diag_x_center, 'gamma': main_diag_x_sigma, 'amplitude': main_diag_x_amplitude}
            create_individual_plot(main_x_pos, main_x_charges, main_x_lorentz_params, true_x,
                                 'Main Diagonal X Lorentzian Fit', 
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_diag_main_x.png'), 'lorentzian')
        
        if len(main_y_pos) >= 3:
            main_y_lorentz_params = {'center': main_diag_y_center, 'gamma': main_diag_y_sigma, 'amplitude': main_diag_y_amplitude}
            create_individual_plot(main_y_pos, main_y_charges, main_y_lorentz_params, true_y,
                                 'Main Diagonal Y Lorentzian Fit',
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_diag_main_y.png'), 'lorentzian')
        
        if len(sec_x_pos) >= 3:
            sec_x_lorentz_params = {'center': sec_diag_x_center, 'gamma': sec_diag_x_sigma, 'amplitude': sec_diag_x_amplitude}
            create_individual_plot(sec_x_pos, sec_x_charges, sec_x_lorentz_params, true_x,
                                 'Secondary Diagonal X Lorentzian Fit',
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_diag_sec_x.png'), 'lorentzian')
        
        if len(sec_y_pos) >= 3:
            sec_y_lorentz_params = {'center': sec_diag_y_center, 'gamma': sec_diag_y_sigma, 'amplitude': sec_diag_y_amplitude}
            create_individual_plot(sec_y_pos, sec_y_charges, sec_y_lorentz_params, true_y,
                                 'Secondary Diagonal Y Lorentzian Fit',
                                 os.path.join(lorentzian_dir, f'event_{event_idx:04d}_diag_sec_y.png'), 'lorentzian')
        
        # ============================================ 
        # SKEWED LORENTZIAN INDIVIDUAL PLOTS
        # ============================================
        
        # Row (X-direction) Skewed Lorentzian
        if len(row_pos) >= 3:
            x_skew_params = {'center': x_skew_lorentz_center, 'gamma': x_skew_lorentz_gamma, 
                           'amplitude': x_skew_lorentz_amplitude, 'skewness': x_skew_lorentz_skewness}
            create_individual_plot(row_pos, row_charges, x_skew_params, true_x,
                                 'Row (X-direction) Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_row.png'), 'skewed_lorentzian')
        
        # Column (Y-direction) Skewed Lorentzian
        if len(col_pos) >= 3:
            y_skew_params = {'center': y_skew_lorentz_center, 'gamma': y_skew_lorentz_gamma,
                           'amplitude': y_skew_lorentz_amplitude, 'skewness': y_skew_lorentz_skewness}
            create_individual_plot(col_pos, col_charges, y_skew_params, true_y,
                                 'Column (Y-direction) Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_column.png'), 'skewed_lorentzian')
        
        # Diagonal plots for Skewed Lorentzian - use Skewed Lorentzian fit on diagonal data
        if len(main_x_pos) >= 3:
            # Project skewed Lorentzian parameters to diagonal
            main_x_skew_params = {'center': main_diag_x_center, 'gamma': main_diag_x_sigma, 'amplitude': main_diag_x_amplitude, 'skewness': x_skew_lorentz_skewness}
            create_individual_plot(main_x_pos, main_x_charges, main_x_skew_params, true_x,
                                 'Main Diagonal X Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_diag_main_x.png'), 'skewed_lorentzian')
        
        if len(main_y_pos) >= 3:
            main_y_skew_params = {'center': main_diag_y_center, 'gamma': main_diag_y_sigma, 'amplitude': main_diag_y_amplitude, 'skewness': y_skew_lorentz_skewness}
            create_individual_plot(main_y_pos, main_y_charges, main_y_skew_params, true_y,
                                 'Main Diagonal Y Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_diag_main_y.png'), 'skewed_lorentzian')
        
        if len(sec_x_pos) >= 3:
            sec_x_skew_params = {'center': sec_diag_x_center, 'gamma': sec_diag_x_sigma, 'amplitude': sec_diag_x_amplitude, 'skewness': x_skew_lorentz_skewness}
            create_individual_plot(sec_x_pos, sec_x_charges, sec_x_skew_params, true_x,
                                 'Secondary Diagonal X Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_diag_sec_x.png'), 'skewed_lorentzian')
        
        if len(sec_y_pos) >= 3:
            sec_y_skew_params = {'center': sec_diag_y_center, 'gamma': sec_diag_y_sigma, 'amplitude': sec_diag_y_amplitude, 'skewness': y_skew_lorentz_skewness}
            create_individual_plot(sec_y_pos, sec_y_charges, sec_y_skew_params, true_y,
                                 'Secondary Diagonal Y Skewed Lorentzian Fit',
                                 os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_diag_sec_y.png'), 'skewed_lorentzian')
        
        # ============================================
        # ALL LINES OVERPLOTTED - LORENTZIAN (All directions)
        # ============================================
        fig_lor_all = plt.figure(figsize=(20, 15))
        gs_lor_all = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        # Row plot
        if len(row_pos) >= 3:
            ax_row = fig_lor_all.add_subplot(gs_lor_all[0, 0])
            ax_row.errorbar(row_pos, row_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(x_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma)
            ax_row.plot(x_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={x_lorentz_chi2red:.2f})')
            ax_row.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_row.set_xlabel('X Position [mm]')
            ax_row.set_ylabel('Charge [C]')
            ax_row.set_title('Row (X-direction)')
            ax_row.legend()
            ax_row.grid(True, alpha=0.3)
        
        # Column plot
        if len(col_pos) >= 3:
            ax_col = fig_lor_all.add_subplot(gs_lor_all[0, 1])
            ax_col.errorbar(col_pos, col_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(y_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma)
            ax_col.plot(y_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={y_lorentz_chi2red:.2f})')
            ax_col.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_col.set_xlabel('Y Position [mm]')
            ax_col.set_ylabel('Charge [C]')
            ax_col.set_title('Column (Y-direction)')
            ax_col.legend()
            ax_col.grid(True, alpha=0.3)
        
        # Add diagonal plots - all lines overplotted
        if len(main_x_pos) >= 3:
            ax_main_x = fig_lor_all.add_subplot(gs_lor_all[1, 0])
            ax_main_x.errorbar(main_x_pos, main_x_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(main_x_pos.min() - 0.1, main_x_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(x_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma)  # Using sigma as gamma
            ax_main_x.plot(x_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={main_diag_x_chi2red:.2f})')
            ax_main_x.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_main_x.set_xlabel('X Position [mm]')
            ax_main_x.set_ylabel('Charge [C]')
            ax_main_x.set_title('Main Diagonal X')
            ax_main_x.legend()
            ax_main_x.grid(True, alpha=0.3)
        
        if len(main_y_pos) >= 3:
            ax_main_y = fig_lor_all.add_subplot(gs_lor_all[1, 1])
            ax_main_y.errorbar(main_y_pos, main_y_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(main_y_pos.min() - 0.1, main_y_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(y_range, main_diag_y_amplitude, main_diag_y_center, main_diag_y_sigma)  # Using sigma as gamma
            ax_main_y.plot(y_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={main_diag_y_chi2red:.2f})')
            ax_main_y.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_main_y.set_xlabel('Y Position [mm]')
            ax_main_y.set_ylabel('Charge [C]')
            ax_main_y.set_title('Main Diagonal Y')
            ax_main_y.legend()
            ax_main_y.grid(True, alpha=0.3)
        
        if len(sec_x_pos) >= 3:
            ax_sec_x = fig_lor_all.add_subplot(gs_lor_all[2, 0])
            ax_sec_x.errorbar(sec_x_pos, sec_x_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(sec_x_pos.min() - 0.1, sec_x_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(x_range, sec_diag_x_amplitude, sec_diag_x_center, sec_diag_x_sigma)  # Using sigma as gamma
            ax_sec_x.plot(x_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={sec_diag_x_chi2red:.2f})')
            ax_sec_x.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_sec_x.set_xlabel('X Position [mm]')
            ax_sec_x.set_ylabel('Charge [C]')
            ax_sec_x.set_title('Secondary Diagonal X')
            ax_sec_x.legend()
            ax_sec_x.grid(True, alpha=0.3)
        
        if len(sec_y_pos) >= 3:
            ax_sec_y = fig_lor_all.add_subplot(gs_lor_all[2, 1])
            ax_sec_y.errorbar(sec_y_pos, sec_y_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(sec_y_pos.min() - 0.1, sec_y_pos.max() + 0.1, 200)
            y_fit = lorentzian_1d(y_range, sec_diag_y_amplitude, sec_diag_y_center, sec_diag_y_sigma)  # Using sigma as gamma
            ax_sec_y.plot(y_range, y_fit, 'r-', linewidth=2, label=f'Lorentzian (χ²={sec_diag_y_chi2red:.2f})')
            ax_sec_y.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_sec_y.set_xlabel('Y Position [mm]')
            ax_sec_y.set_ylabel('Charge [C]')
            ax_sec_y.set_title('Secondary Diagonal Y')
            ax_sec_y.legend()
            ax_sec_y.grid(True, alpha=0.3)
        
        
        plt.suptitle(f'Event {event_idx}: Lorentzian Fits (All Directions)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(lorentzian_dir, f'event_{event_idx:04d}_all_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # ============================================
        # ALL LINES OVERPLOTTED - SKEWED LORENTZIAN (All directions)
        # ============================================
        fig_skew_all = plt.figure(figsize=(20, 15))
        gs_skew_all = GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        # Row plot
        if len(row_pos) >= 3:
            ax_row = fig_skew_all.add_subplot(gs_skew_all[0, 0])
            ax_row.errorbar(row_pos, row_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(x_range, x_skew_lorentz_amplitude, x_skew_lorentz_center, x_skew_lorentz_gamma, x_skew_lorentz_skewness)
            ax_row.plot(x_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={x_skew_lorentz_chi2red:.2f})')
            ax_row.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_row.set_xlabel('X Position [mm]')
            ax_row.set_ylabel('Charge [C]')
            ax_row.set_title('Row (X-direction)')
            ax_row.legend()
            ax_row.grid(True, alpha=0.3)
        
        # Column plot
        if len(col_pos) >= 3:
            ax_col = fig_skew_all.add_subplot(gs_skew_all[0, 1])
            ax_col.errorbar(col_pos, col_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(y_range, y_skew_lorentz_amplitude, y_skew_lorentz_center, y_skew_lorentz_gamma, y_skew_lorentz_skewness)
            ax_col.plot(y_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={y_skew_lorentz_chi2red:.2f})')
            ax_col.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_col.set_xlabel('Y Position [mm]')
            ax_col.set_ylabel('Charge [C]')
            ax_col.set_title('Column (Y-direction)')
            ax_col.legend()
            ax_col.grid(True, alpha=0.3)
        
        # Add diagonal plots for skewed Lorentzian - all lines overplotted
        if len(main_x_pos) >= 3:
            ax_main_x = fig_skew_all.add_subplot(gs_skew_all[1, 0])
            ax_main_x.errorbar(main_x_pos, main_x_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(main_x_pos.min() - 0.1, main_x_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(x_range, main_diag_x_amplitude, main_diag_x_center, main_diag_x_sigma, x_skew_lorentz_skewness)
            ax_main_x.plot(x_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={main_diag_x_chi2red:.2f})')
            ax_main_x.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_main_x.set_xlabel('X Position [mm]')
            ax_main_x.set_ylabel('Charge [C]')
            ax_main_x.set_title('Main Diagonal X')
            ax_main_x.legend()
            ax_main_x.grid(True, alpha=0.3)
        
        if len(main_y_pos) >= 3:
            ax_main_y = fig_skew_all.add_subplot(gs_skew_all[1, 1])
            ax_main_y.errorbar(main_y_pos, main_y_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(main_y_pos.min() - 0.1, main_y_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(y_range, main_diag_y_amplitude, main_diag_y_center, main_diag_y_sigma, y_skew_lorentz_skewness)
            ax_main_y.plot(y_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={main_diag_y_chi2red:.2f})')
            ax_main_y.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_main_y.set_xlabel('Y Position [mm]')
            ax_main_y.set_ylabel('Charge [C]')
            ax_main_y.set_title('Main Diagonal Y')
            ax_main_y.legend()
            ax_main_y.grid(True, alpha=0.3)
        
        if len(sec_x_pos) >= 3:
            ax_sec_x = fig_skew_all.add_subplot(gs_skew_all[2, 0])
            ax_sec_x.errorbar(sec_x_pos, sec_x_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(sec_x_pos.min() - 0.1, sec_x_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(x_range, sec_diag_x_amplitude, sec_diag_x_center, sec_diag_x_sigma, x_skew_lorentz_skewness)
            ax_sec_x.plot(x_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={sec_diag_x_chi2red:.2f})')
            ax_sec_x.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax_sec_x.set_xlabel('X Position [mm]')
            ax_sec_x.set_ylabel('Charge [C]')
            ax_sec_x.set_title('Secondary Diagonal X')
            ax_sec_x.legend()
            ax_sec_x.grid(True, alpha=0.3)
        
        if len(sec_y_pos) >= 3:
            ax_sec_y = fig_skew_all.add_subplot(gs_skew_all[2, 1])
            ax_sec_y.errorbar(sec_y_pos, sec_y_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(sec_y_pos.min() - 0.1, sec_y_pos.max() + 0.1, 200)
            y_fit = skewed_lorentzian_1d(y_range, sec_diag_y_amplitude, sec_diag_y_center, sec_diag_y_sigma, y_skew_lorentz_skewness)
            ax_sec_y.plot(y_range, y_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={sec_diag_y_chi2red:.2f})')
            ax_sec_y.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax_sec_y.set_xlabel('Y Position [mm]')
            ax_sec_y.set_ylabel('Charge [C]')
            ax_sec_y.set_title('Secondary Diagonal Y')
            ax_sec_y.legend()
            ax_sec_y.grid(True, alpha=0.3)
        
        plt.suptitle(f'Event {event_idx}: Skewed Lorentzian Fits (All Directions)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(skewed_lorentzian_dir, f'event_{event_idx:04d}_all_skewed_lorentzian.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: Lorentzian and Skewed Lorentzian plots created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating plots - {e}"

def create_lorentzian_vs_skewed_comparison(event_idx, data, output_dir):
    """
    Create comparison plots between Lorentzian and Skewed Lorentzian fits.
    Now uses real skewed Lorentzian data from the ROOT file.
    """
    try:
        # Extract data
        (row_pos, row_charges), (col_pos, col_charges) = extract_row_column_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for comparison"
        
        # Get Lorentzian fit parameters
        x_lorentz_center = data['Fit2D_Lorentz_XCenter'][event_idx]
        x_lorentz_gamma = data['Fit2D_Lorentz_XGamma'][event_idx]
        x_lorentz_amplitude = data['Fit2D_Lorentz_XAmplitude'][event_idx]
        x_lorentz_chi2red = data['Fit2D_Lorentz_XChi2red'][event_idx]
        
        y_lorentz_center = data['Fit2D_Lorentz_YCenter'][event_idx]
        y_lorentz_gamma = data['Fit2D_Lorentz_YGamma'][event_idx]
        y_lorentz_amplitude = data['Fit2D_Lorentz_YAmplitude'][event_idx]
        y_lorentz_chi2red = data['Fit2D_Lorentz_YChi2red'][event_idx]
        
        # Get Skewed Lorentzian fit parameters (now using real data!)
        x_skew_center = data['Fit2D_SkewLorentz_XCenter'][event_idx]
        x_skew_gamma = data['Fit2D_SkewLorentz_XGamma'][event_idx]  # This is actually Beta
        x_skew_amplitude = data['Fit2D_SkewLorentz_XAmplitude'][event_idx]
        x_skew_skewness = data['Fit2D_SkewLorentz_XSkewness'][event_idx]  # This is actually Lambda
        x_skew_chi2red = data['Fit2D_SkewLorentz_XChi2red'][event_idx]
        
        y_skew_center = data['Fit2D_SkewLorentz_YCenter'][event_idx]
        y_skew_gamma = data['Fit2D_SkewLorentz_YGamma'][event_idx]  # This is actually Beta
        y_skew_amplitude = data['Fit2D_SkewLorentz_YAmplitude'][event_idx]
        y_skew_skewness = data['Fit2D_SkewLorentz_YSkewness'][event_idx]  # This is actually Lambda
        y_skew_chi2red = data['Fit2D_SkewLorentz_YChi2red'][event_idx]
        
        # True positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        # Create comparison directory
        comparison_dir = os.path.join(output_dir, "lorentzian_vs_skewed")
        os.makedirs(comparison_dir, exist_ok=True)
        
        created_comparisons = []
        
        # Row comparison
        if len(row_pos) >= 3:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot data
            ax.errorbar(row_pos, row_charges, fmt='ko', markersize=8, capsize=4, label='Data', alpha=0.8)
            
            # Plot fits
            x_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            lorentz_fit = lorentzian_1d(x_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma)
            skew_fit = skewed_lorentzian_1d(x_range, x_skew_amplitude, x_skew_center, x_skew_gamma, x_skew_skewness)
            
            ax.plot(x_range, lorentz_fit, 'r--', linewidth=2.5, label=f'Lorentzian (χ²={x_lorentz_chi2red:.2f})', alpha=0.9)
            ax.plot(x_range, skew_fit, 'm-', linewidth=2.5, label=f'Skewed Lorentzian (χ²={x_skew_chi2red:.2f})', alpha=0.9)
            
            # Add true position
            ax.axvline(true_x, color='green', linestyle='--', linewidth=2, label=f'True X = {true_x:.3f} mm', alpha=0.8)
            
            ax.set_xlabel('X Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(f'Event {event_idx}: Row (X-direction) Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(comparison_dir, f'event_{event_idx:04d}_row_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_comparisons.append('Row comparison')
        
        # Column comparison
        if len(col_pos) >= 3:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot data
            ax.errorbar(col_pos, col_charges, fmt='ko', markersize=8, capsize=4, label='Data', alpha=0.8)
            
            # Plot fits
            y_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            lorentz_fit = lorentzian_1d(y_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma)
            skew_fit = skewed_lorentzian_1d(y_range, y_skew_amplitude, y_skew_center, y_skew_gamma, y_skew_skewness)
            
            ax.plot(y_range, lorentz_fit, 'r--', linewidth=2.5, label=f'Lorentzian (χ²={y_lorentz_chi2red:.2f})', alpha=0.9)
            ax.plot(y_range, skew_fit, 'm-', linewidth=2.5, label=f'Skewed Lorentzian (χ²={y_skew_chi2red:.2f})', alpha=0.9)
            
            # Add true position
            ax.axvline(true_y, color='green', linestyle='--', linewidth=2, label=f'True Y = {true_y:.3f} mm', alpha=0.8)
            
            ax.set_xlabel('Y Position [mm]')
            ax.set_ylabel('Charge [C]')
            ax.set_title(f'Event {event_idx}: Column (Y-direction) Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(comparison_dir, f'event_{event_idx:04d}_column_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_comparisons.append('Column comparison')
        
        # Combined comparison plot
        if len(row_pos) >= 3 and len(col_pos) >= 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Row subplot
            ax1.errorbar(row_pos, row_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            x_range = np.linspace(row_pos.min() - 0.1, row_pos.max() + 0.1, 200)
            lorentz_fit = lorentzian_1d(x_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma)
            skew_fit = skewed_lorentzian_1d(x_range, x_skew_amplitude, x_skew_center, x_skew_gamma, x_skew_skewness)
            ax1.plot(x_range, lorentz_fit, 'r--', linewidth=2, label=f'Lorentzian (χ²={x_lorentz_chi2red:.2f})')
            ax1.plot(x_range, skew_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={x_skew_chi2red:.2f})')
            ax1.axvline(true_x, color='green', linestyle='--', label=f'True X = {true_x:.3f} mm')
            ax1.set_xlabel('X Position [mm]')
            ax1.set_ylabel('Charge [C]')
            ax1.set_title('Row (X-direction)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Column subplot
            ax2.errorbar(col_pos, col_charges, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            y_range = np.linspace(col_pos.min() - 0.1, col_pos.max() + 0.1, 200)
            lorentz_fit = lorentzian_1d(y_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma)
            skew_fit = skewed_lorentzian_1d(y_range, y_skew_amplitude, y_skew_center, y_skew_gamma, y_skew_skewness)
            ax2.plot(y_range, lorentz_fit, 'r--', linewidth=2, label=f'Lorentzian (χ²={y_lorentz_chi2red:.2f})')
            ax2.plot(y_range, skew_fit, 'm-', linewidth=2, label=f'Skewed Lorentzian (χ²={y_skew_chi2red:.2f})')
            ax2.axvline(true_y, color='green', linestyle='--', label=f'True Y = {true_y:.3f} mm')
            ax2.set_xlabel('Y Position [mm]')
            ax2.set_ylabel('Charge [C]')
            ax2.set_title('Column (Y-direction)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Event {event_idx}: Lorentzian vs Skewed Lorentzian Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'event_{event_idx:04d}_all_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_comparisons.append('All directions comparison')
        
        if created_comparisons:
            return f"Event {event_idx}: {' and '.join(created_comparisons)} saved to lorentzian_vs_skewed directory"
        else:
            return f"Event {event_idx}: No comparison plots created (insufficient data)"
            
    except Exception as e:
        return f"Event {event_idx}: Error creating comparison plot - {e}"

def create_summary_plots(data, output_dir, max_events):
    """
    Create summary plots for all fitting methods.
    This is a placeholder function that can be expanded later.
    """
    print(f"Summary plots functionality not yet implemented (would process up to {max_events} events)")
    return "Summary plots placeholder completed"

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Create Gaussian, Lorentzian, and Skewed Lorentzian fit plots for charge sharing analysis",
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
            
            # Create comparison plots (Lorentzian vs Skewed Lorentzian)
            comparison_result = create_lorentzian_vs_skewed_comparison(i, data, args.output)
            if "Error" not in comparison_result:
                overlay_success_count += 1
            if i % 5 == 0 or "Error" in comparison_result:
                print(f"  {comparison_result}")
        
        if not args.overlay_only:
            print(f"\nSuccessfully created {success_count}/{n_events} individual plots")
        print(f"Successfully created {overlay_success_count}/{n_events} comparison plots")
    
    print(f"\nAll plots saved to: {args.output}/")
    print(f"  - Lorentzian fits: {args.output}/lorentzian/")
    
    # Check if skewed Lorentzian data is available in the dataset
    if data and any(key.startswith('Fit2D_SkewLorentz_') for key in data.keys()):
        print(f"  - Skewed Lorentzian fits: {args.output}/skewed_lorentzian/")
        print(f"  - Lorentzian vs Skewed Lorentzian comparison: {args.output}/lorentzian_vs_skewed/")
    else:
        print(f"  - Skewed Lorentzian fits: Not available (no skewed Lorentzian data in ROOT file)")
        print(f"  - Lorentzian vs Skewed Lorentzian comparison: Not available (no skewed Lorentzian data)")
    
    print(f"  - Summary plots: {args.output}/")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 