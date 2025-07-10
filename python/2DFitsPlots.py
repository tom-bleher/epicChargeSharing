#!/usr/bin/env python3
"""
Publication-Quality Post-processing plotting routine for Gauss, Lorentz, and Power Lorentz 
fit visualization of charge sharing in LGAD detectors.

ENHANCED FOR SCIENTIFIC PUBLICATION:
- Professional LaTeX mathematical notation with proper raw strings
- High-resolution output (300 DPI) with publication-quality fonts
- Consistent professional color palette and styling
- Enhanced legends with proper mathematical symbols
- Professional grid and axis styling
- Optimized for scientific journals and presentations

Post-processing plotting routine for Gauss, Lorentz, and Power Lorentz fit visualization of charge sharing in LGAD detectors.

This script creates plots for:
1. Gauss and Lorentz curve estimation for central row (x-direction) with residuals
2. Gauss and Lorentz curve estimation for central column (y-direction) with residuals
3. Gauss curve estimation for main diagonal direction with residuals
4. Gauss curve estimation for secondary diagonal direction with residuals
5. Comparison plots showing ALL fitting approaches overlaid:
   - X-direction: Row (Gauss + Lorentz) + Main Diag (Gauss) + Secondary Diag (Gauss)
   - Y-direction: Col (Gauss + Lorentz) + Main Diag (Gauss) + Secondary Diag (Gauss)
6. Model comparison plots:
   - Gauss vs Lorentz comparison plots
   - Lorentz vs Power Lorentz comparison plots (if available)
   - All models comparison plots (Gauss + Lorentz + Power Lorentz if available)

The plots show fitted curves overlaid on actual charge data points from the neighborhood grid, 
along with residual plots showing fit quality. Individual directions get separate figures in their 
respective subdirectories, and comparison plots show different fitting approaches overlaid for comprehensive analysis.

AUTOMATIC UNCERTAINTY DETECTION:
The script automatically detects whether charge err branches are available in the ROOT file.
- If err branches are present and contain meaningful values, data points are plotted with error bars
- If err branches are missing or contain only zeros (when ENABLE_VERT_CHARGE_ERR=false), 
  data points are plotted without error bars
- This allows the same plotting script to work with ROOT files from both err-enabled and err-disabled simulations

Special modes:
- Use --best_worst to plot the 5 best and 5 worst fits based on chi-squared values
- Use --high_amps N to plot the N events with highest amps (useful for examining outliers)

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
plt.rcParams['figure.dpi'] = 300  # Higher DPI for publication quality
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'  # More professional for publications
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.fontsize'] = 10
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
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available on system
plt.rcParams['mathtext.fontset'] = 'stix'  # Professional math font

def configure_publication_style():
    """
    Configure matplotlib for publication-quality plots with enhanced aesthetics.
    """
    # Professional color palette for different fit types
    colors = {
        'gauss': '#1f77b4',      # Professional blue
        'lorentz': '#ff7f0e',    # Professional orange
        'power_lorentz': '#9467bd', # Professional purple
        'true_pos': '#2ca02c',  # Professional green
        'data_points': '#000000'     # Black for data points
    }
    
    # Enhanced line styles
    line_styles = {
        'gauss': '-',
        'lorentz': '--',
        'power_lorentz': ':',
        'true_pos': '--'
    }
    
    # Enhanced line widths
    line_widths = {
        'fit_curves': 2.5,
        'power_lorentz': 3.0,  # Slightly thicker for dotted line
        'true_pos': 2.5,
        'reference_lines': 1.5
    }
    
    return colors, line_styles, line_widths

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

def gauss_1d(x, amp, center, sigma, offset=0):
    """
    1D Gauss function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amp: Gauss amp
        center: Gauss center (mean)
        sigma: Gauss sigma (standard deviation)
        offset: Baseline offset
    
    Returns:
        Gauss function values
    """
    return amp * np.exp(-0.5 * ((x - center) / sigma)**2) + offset

def lorentz_1d(x, amp, center, gamma, offset=0):
    """
    1D Lorentz function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amp: Lorentz amp
        center: Lorentz center
        gamma: Lorentz gamma (half-width at half-maximum, HWHM)
        offset: Baseline offset
    
    Returns:
        Lorentz function values
    """
    return amp / (1 + ((x - center) / gamma)**2) + offset

def power_lorentz_1d(x, amp, center, gamma, power, offset=0):
    """
    1D Power Lorentz function for plotting fitted curves.
    
    Args:
        x: Independent variable
        amp: Power Lorentz amp
        center: Power Lorentz center
        gamma: Power Lorentz gamma (half-width at half-maximum, HWHM)
        power: Power parameter (alpha)
        offset: Baseline offset
    
    Returns:
        Power Lorentz function values
    """
    # Power Lorentz component
    # Standard form: 1 / (1 + ((x - center) / gamma)^2)^power
    return amp / (1.0 + ((x - center) / gamma)**2)**power + offset

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

def detect_err_branches(data):
    """
    Detect whether charge err branches are available in the data.
    
    Args:
        data (dict): Data dictionary from ROOT file
        
    Returns:
        dict: Dictionary with flags indicating which err branches are available
    """
    err_status = {
        'gauss_row_err_available': False,
        'gauss_col_err_available': False,
        'lorentz_row_err_available': False,
        'lorentz_col_err_available': False,
        'any_uncertainties_available': False
    }
    
    # Check for Gauss err branches
    if 'GaussRowChargeErr' in data:
        # Check if the values are meaningful (not all zeros)
        uncertainties = data['GaussRowChargeErr']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            err_status['gauss_row_err_available'] = True
            
    if 'GaussColChargeErr' in data:
        uncertainties = data['GaussColChargeErr']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            err_status['gauss_col_err_available'] = True
    
    # Check for Lorentz err branches  
    if 'LorentzRowChargeErr' in data:
        uncertainties = data['LorentzRowChargeErr']
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            err_status['lorentz_row_err_available'] = True
            
    if 'LorentzColChargeErr' in data:
        uncertainties = data['LorentzColChargeErr'] 
        if len(uncertainties) > 0 and np.any(np.abs(uncertainties) > 1e-20):
            err_status['lorentz_col_err_available'] = True
    
    # Set overall flag
    err_status['any_uncertainties_available'] = (
        err_status['gauss_row_err_available'] or
        err_status['gauss_col_err_available'] or
        err_status['lorentz_row_err_available'] or
        err_status['lorentz_col_err_available']
    )
    
    return err_status

def load_success_fits(root_file, max_entries=None):
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
                
                # Gauss fit results - Row/X direction
                '2D_XCenter': 'GaussRowCenter',
                '2D_XSigma': 'GaussRowSigma', 
                '2D_XAmp': 'GaussRowAmp',
                '2D_XVertOffset': 'GaussRowVertOffset',
                '2D_XCenterErr': 'GaussRowCenterErr',
                '2D_XSigmaErr': 'GaussRowSigmaErr',
                '2D_XAmpErr': 'GaussRowAmpErr',
                '2D_XVertOffsetErr': 'GaussRowVertOffsetErr',
                '2D_XChi2red': 'GaussRowChi2red',
                '2D_XNPoints': 'GaussRowDOF',  # DOF + parameters = NPoints
                
                # Gauss fit results - Col/Y direction  
                '2D_YCenter': 'GaussColCenter',
                '2D_YSigma': 'GaussColSigma',
                '2D_YAmp': 'GaussColAmp',
                '2D_YVertOffset': 'GaussColVertOffset',
                '2D_YCenterErr': 'GaussColCenterErr',
                '2D_YSigmaErr': 'GaussColSigmaErr',
                '2D_YAmpErr': 'GaussColAmpErr',
                '2D_YVertOffsetErr': 'GaussColVertOffsetErr',
                '2D_YChi2red': 'GaussColChi2red',
                '2D_YNPoints': 'GaussColDOF',
                
                # Lorentz fit results - Row/X direction
                '2D_Lorentz_XCenter': 'LorentzRowCenter',
                '2D_Lorentz_XGamma': 'LorentzRowGamma',
                '2D_Lorentz_XAmp': 'LorentzRowAmp',
                '2D_Lorentz_XVertOffset': 'LorentzRowVertOffset',
                '2D_Lorentz_XCenterErr': 'LorentzRowCenterErr',
                '2D_Lorentz_XGammaErr': 'LorentzRowGammaErr',
                '2D_Lorentz_XAmpErr': 'LorentzRowAmpErr',
                '2D_Lorentz_XVertOffsetErr': 'LorentzRowVertOffsetErr',
                '2D_Lorentz_XChi2red': 'LorentzRowChi2red',
                '2D_Lorentz_XNPoints': 'LorentzRowDOF',
                
                # Lorentz fit results - Col/Y direction
                '2D_Lorentz_YCenter': 'LorentzColCenter',
                '2D_Lorentz_YGamma': 'LorentzColGamma',
                '2D_Lorentz_YAmp': 'LorentzColAmp',
                '2D_Lorentz_YVertOffset': 'LorentzColVertOffset',
                '2D_Lorentz_YCenterErr': 'LorentzColCenterErr',
                '2D_Lorentz_YGammaErr': 'LorentzColGammaErr',
                '2D_Lorentz_YAmpErr': 'LorentzColAmpErr',
                '2D_Lorentz_YVertOffsetErr': 'LorentzColVertOffsetErr',
                '2D_Lorentz_YChi2red': 'LorentzColChi2red',
                '2D_Lorentz_YNPoints': 'LorentzColDOF',
                
                # Power Lorentz fit results - Row/X direction
                '2D_PowerLorentz_XCenter': 'PowerLorentzRowCenter',
                '2D_PowerLorentz_XGamma': 'PowerLorentzRowGamma',  # Gamma is the width parameter
                '2D_PowerLorentz_XAmp': 'PowerLorentzRowAmp',
                '2D_PowerLorentz_XPower': 'PowerLorentzRowBeta',  # Beta is the power exponent parameter
                '2D_PowerLorentz_XVertOffset': 'PowerLorentzRowVertOffset',
                '2D_PowerLorentz_XCenterErr': 'PowerLorentzRowCenterErr',
                '2D_PowerLorentz_XGammaErr': 'PowerLorentzRowGammaErr',
                '2D_PowerLorentz_XAmpErr': 'PowerLorentzRowAmpErr',
                '2D_PowerLorentz_XPowerErr': 'PowerLorentzRowBetaErr',
                '2D_PowerLorentz_XVertOffsetErr': 'PowerLorentzRowVertOffsetErr',
                '2D_PowerLorentz_XChi2red': 'PowerLorentzRowChi2red',
                '2D_PowerLorentz_XNPoints': 'PowerLorentzRowDOF',
                
                # Power Lorentz fit results - Col/Y direction
                '2D_PowerLorentz_YCenter': 'PowerLorentzColCenter',
                '2D_PowerLorentz_YGamma': 'PowerLorentzColGamma',  # Gamma is the width parameter
                '2D_PowerLorentz_YAmp': 'PowerLorentzColAmp',
                '2D_PowerLorentz_YPower': 'PowerLorentzColBeta',  # Beta is the power exponent parameter
                '2D_PowerLorentz_YVertOffset': 'PowerLorentzColVertOffset',
                '2D_PowerLorentz_YCenterErr': 'PowerLorentzColCenterErr',
                '2D_PowerLorentz_YGammaErr': 'PowerLorentzColGammaErr',
                '2D_PowerLorentz_YAmpErr': 'PowerLorentzColAmpErr',
                '2D_PowerLorentz_YPowerErr': 'PowerLorentzColBetaErr',
                '2D_PowerLorentz_YVertOffsetErr': 'PowerLorentzColVertOffsetErr',
                '2D_PowerLorentz_YChi2red': 'PowerLorentzColChi2red',
                '2D_PowerLorentz_YNPoints': 'PowerLorentzColDOF',
                
                # Diag Gauss fits - Main diagonal X
                'Diag_MainXCenter': 'GaussMainDiagXCenter',
                'Diag_MainXSigma': 'GaussMainDiagXSigma',
                'Diag_MainXAmp': 'GaussMainDiagXAmp',
                'Diag_MainXVertOffset': 'GaussMainDiagXVertOffset',
                'Diag_MainXCenterErr': 'GaussMainDiagXCenterErr',
                'Diag_MainXSigmaErr': 'GaussMainDiagXSigmaErr',
                'Diag_MainXAmpErr': 'GaussMainDiagXAmpErr',
                'Diag_MainXChi2red': 'GaussMainDiagXChi2red',
                'Diag_MainXNPoints': 'GaussMainDiagXDOF',
                
                # Diag Gauss fits - Main diagonal Y
                'Diag_MainYCenter': 'GaussMainDiagYCenter',
                'Diag_MainYSigma': 'GaussMainDiagYSigma',
                'Diag_MainYAmp': 'GaussMainDiagYAmp',
                'Diag_MainYVertOffset': 'GaussMainDiagYVertOffset',
                'Diag_MainYCenterErr': 'GaussMainDiagYCenterErr',
                'Diag_MainYSigmaErr': 'GaussMainDiagYSigmaErr',
                'Diag_MainYAmpErr': 'GaussMainDiagYAmpErr',
                'Diag_MainYChi2red': 'GaussMainDiagYChi2red',
                'Diag_MainYNPoints': 'GaussMainDiagYDOF',
                
                # Diag Gauss fits - Secondary diagonal X
                'Diag_SecXCenter': 'GaussSecDiagXCenter',
                'Diag_SecXSigma': 'GaussSecDiagXSigma',
                'Diag_SecXAmp': 'GaussSecDiagXAmp',
                'Diag_SecXVertOffset': 'GaussSecDiagXVertOffset',
                'Diag_SecXCenterErr': 'GaussSecDiagXCenterErr',
                'Diag_SecXSigmaErr': 'GaussSecDiagXSigmaErr',
                'Diag_SecXAmpErr': 'GaussSecDiagXAmpErr',
                'Diag_SecXChi2red': 'GaussSecDiagXChi2red',
                'Diag_SecXNPoints': 'GaussSecDiagXDOF',
                
                # Diag Gauss fits - Secondary diagonal Y
                'Diag_SecYCenter': 'GaussSecDiagYCenter',
                'Diag_SecYSigma': 'GaussSecDiagYSigma',
                'Diag_SecYAmp': 'GaussSecDiagYAmp',
                'Diag_SecYVertOffset': 'GaussSecDiagYVertOffset',
                'Diag_SecYCenterErr': 'GaussSecDiagYCenterErr',
                'Diag_SecYSigmaErr': 'GaussSecDiagYSigmaErr',
                'Diag_SecYAmpErr': 'GaussSecDiagYAmpErr',
                'Diag_SecYChi2red': 'GaussSecDiagYChi2red',
                'Diag_SecYNPoints': 'GaussSecDiagYDOF',
                
                # Diag Lorentz fits - Main diagonal X
                'Diag_Lorentz_MainXCenter': 'LorentzMainDiagXCenter',
                'Diag_Lorentz_MainXGamma': 'LorentzMainDiagXGamma',
                'Diag_Lorentz_MainXAmp': 'LorentzMainDiagXAmp',
                'Diag_Lorentz_MainXVertOffset': 'LorentzMainDiagXVertOffset',
                'Diag_Lorentz_MainXCenterErr': 'LorentzMainDiagXCenterErr',
                'Diag_Lorentz_MainXGammaErr': 'LorentzMainDiagXGammaErr',
                'Diag_Lorentz_MainXAmpErr': 'LorentzMainDiagXAmpErr',
                'Diag_Lorentz_MainXVertOffsetErr': 'LorentzMainDiagXVertOffsetErr',
                'Diag_Lorentz_MainXChi2red': 'LorentzMainDiagXChi2red',
                'Diag_Lorentz_MainXNPoints': 'LorentzMainDiagXDOF',
                
                # Diag Lorentz fits - Main diagonal Y
                'Diag_Lorentz_MainYCenter': 'LorentzMainDiagYCenter',
                'Diag_Lorentz_MainYGamma': 'LorentzMainDiagYGamma',
                'Diag_Lorentz_MainYAmp': 'LorentzMainDiagYAmp',
                'Diag_Lorentz_MainYVertOffset': 'LorentzMainDiagYVertOffset',
                'Diag_Lorentz_MainYCenterErr': 'LorentzMainDiagYCenterErr',
                'Diag_Lorentz_MainYGammaErr': 'LorentzMainDiagYGammaErr',
                'Diag_Lorentz_MainYAmpErr': 'LorentzMainDiagYAmpErr',
                'Diag_Lorentz_MainYVertOffsetErr': 'LorentzMainDiagYVertOffsetErr',
                'Diag_Lorentz_MainYChi2red': 'LorentzMainDiagYChi2red',
                'Diag_Lorentz_MainYNPoints': 'LorentzMainDiagYDOF',
                
                # Diag Lorentz fits - Secondary diagonal X
                'Diag_Lorentz_SecXCenter': 'LorentzSecDiagXCenter',
                'Diag_Lorentz_SecXGamma': 'LorentzSecDiagXGamma',
                'Diag_Lorentz_SecXAmp': 'LorentzSecDiagXAmp',
                'Diag_Lorentz_SecXVertOffset': 'LorentzSecDiagXVertOffset',
                'Diag_Lorentz_SecXCenterErr': 'LorentzSecDiagXCenterErr',
                'Diag_Lorentz_SecXGammaErr': 'LorentzSecDiagXGammaErr',
                'Diag_Lorentz_SecXAmpErr': 'LorentzSecDiagXAmpErr',
                'Diag_Lorentz_SecXVertOffsetErr': 'LorentzSecDiagXVertOffsetErr',
                'Diag_Lorentz_SecXChi2red': 'LorentzSecDiagXChi2red',
                'Diag_Lorentz_SecXNPoints': 'LorentzSecDiagXDOF',
                
                # Diag Lorentz fits - Secondary diagonal Y
                'Diag_Lorentz_SecYCenter': 'LorentzSecDiagYCenter',
                'Diag_Lorentz_SecYGamma': 'LorentzSecDiagYGamma',
                'Diag_Lorentz_SecYAmp': 'LorentzSecDiagYAmp',
                'Diag_Lorentz_SecYVertOffset': 'LorentzSecDiagYVertOffset',
                'Diag_Lorentz_SecYCenterErr': 'LorentzSecDiagYCenterErr',
                'Diag_Lorentz_SecYGammaErr': 'LorentzSecDiagYGammaErr',
                'Diag_Lorentz_SecYAmpErr': 'LorentzSecDiagYAmpErr',
                'Diag_Lorentz_SecYVertOffsetErr': 'LorentzSecDiagYVertOffsetErr',
                'Diag_Lorentz_SecYChi2red': 'LorentzSecDiagYChi2red',
                'Diag_Lorentz_SecYNPoints': 'LorentzSecDiagYDOF',
                
                # Diag Power Lorentz fits - Main diagonal X
                'Diag_PowerLorentz_MainXCenter': 'PowerLorentzMainDiagXCenter',
                'Diag_PowerLorentz_MainXGamma': 'PowerLorentzMainDiagXGamma',
                'Diag_PowerLorentz_MainXBeta': 'PowerLorentzMainDiagXBeta',
                'Diag_PowerLorentz_MainXAmp': 'PowerLorentzMainDiagXAmp',
                'Diag_PowerLorentz_MainXVertOffset': 'PowerLorentzMainDiagXVertOffset',
                'Diag_PowerLorentz_MainXCenterErr': 'PowerLorentzMainDiagXCenterErr',
                'Diag_PowerLorentz_MainXGammaErr': 'PowerLorentzMainDiagXGammaErr',
                'Diag_PowerLorentz_MainXBetaErr': 'PowerLorentzMainDiagXBetaErr',
                'Diag_PowerLorentz_MainXAmpErr': 'PowerLorentzMainDiagXAmpErr',
                'Diag_PowerLorentz_MainXVertOffsetErr': 'PowerLorentzMainDiagXVertOffsetErr',
                'Diag_PowerLorentz_MainXChi2red': 'PowerLorentzMainDiagXChi2red',
                'Diag_PowerLorentz_MainXNPoints': 'PowerLorentzMainDiagXDOF',
                
                # Diag Power Lorentz fits - Main diagonal Y
                'Diag_PowerLorentz_MainYCenter': 'PowerLorentzMainDiagYCenter',
                'Diag_PowerLorentz_MainYGamma': 'PowerLorentzMainDiagYGamma',
                'Diag_PowerLorentz_MainYBeta': 'PowerLorentzMainDiagYBeta',
                'Diag_PowerLorentz_MainYAmp': 'PowerLorentzMainDiagYAmp',
                'Diag_PowerLorentz_MainYVertOffset': 'PowerLorentzMainDiagYVertOffset',
                'Diag_PowerLorentz_MainYCenterErr': 'PowerLorentzMainDiagYCenterErr',
                'Diag_PowerLorentz_MainYGammaErr': 'PowerLorentzMainDiagYGammaErr',
                'Diag_PowerLorentz_MainYBetaErr': 'PowerLorentzMainDiagYBetaErr',
                'Diag_PowerLorentz_MainYAmpErr': 'PowerLorentzMainDiagYAmpErr',
                'Diag_PowerLorentz_MainYVertOffsetErr': 'PowerLorentzMainDiagYVertOffsetErr',
                'Diag_PowerLorentz_MainYChi2red': 'PowerLorentzMainDiagYChi2red',
                'Diag_PowerLorentz_MainYNPoints': 'PowerLorentzMainDiagYDOF',
                
                # Diag Power Lorentz fits - Secondary diagonal X
                'Diag_PowerLorentz_SecXCenter': 'PowerLorentzSecDiagXCenter',
                'Diag_PowerLorentz_SecXGamma': 'PowerLorentzSecDiagXGamma',
                'Diag_PowerLorentz_SecXBeta': 'PowerLorentzSecDiagXBeta',
                'Diag_PowerLorentz_SecXAmp': 'PowerLorentzSecDiagXAmp',
                'Diag_PowerLorentz_SecXVertOffset': 'PowerLorentzSecDiagXVertOffset',
                'Diag_PowerLorentz_SecXCenterErr': 'PowerLorentzSecDiagXCenterErr',
                'Diag_PowerLorentz_SecXGammaErr': 'PowerLorentzSecDiagXGammaErr',
                'Diag_PowerLorentz_SecXBetaErr': 'PowerLorentzSecDiagXBetaErr',
                'Diag_PowerLorentz_SecXAmpErr': 'PowerLorentzSecDiagXAmpErr',
                'Diag_PowerLorentz_SecXVertOffsetErr': 'PowerLorentzSecDiagXVertOffsetErr',
                'Diag_PowerLorentz_SecXChi2red': 'PowerLorentzSecDiagXChi2red',
                'Diag_PowerLorentz_SecXNPoints': 'PowerLorentzSecDiagXDOF',
                
                # Diag Power Lorentz fits - Secondary diagonal Y
                'Diag_PowerLorentz_SecYCenter': 'PowerLorentzSecDiagYCenter',
                'Diag_PowerLorentz_SecYGamma': 'PowerLorentzSecDiagYGamma',
                'Diag_PowerLorentz_SecYBeta': 'PowerLorentzSecDiagYBeta',
                'Diag_PowerLorentz_SecYAmp': 'PowerLorentzSecDiagYAmp',
                'Diag_PowerLorentz_SecYVertOffset': 'PowerLorentzSecDiagYVertOffset',
                'Diag_PowerLorentz_SecYCenterErr': 'PowerLorentzSecDiagYCenterErr',
                'Diag_PowerLorentz_SecYGammaErr': 'PowerLorentzSecDiagYGammaErr',
                'Diag_PowerLorentz_SecYBetaErr': 'PowerLorentzSecDiagYBetaErr',
                'Diag_PowerLorentz_SecYAmpErr': 'PowerLorentzSecDiagYAmpErr',
                'Diag_PowerLorentz_SecYVertOffsetErr': 'PowerLorentzSecDiagYVertOffsetErr',
                'Diag_PowerLorentz_SecYChi2red': 'PowerLorentzSecDiagYChi2red',
                'Diag_PowerLorentz_SecYNPoints': 'PowerLorentzSecDiagYDOF',
                
                # Grid neighborhood data
                'NeighborhoodCharge': 'NeighborhoodCharges',
                'NeighborhoodDistances': 'NeighborhoodDistances',
                'NeighborhoodAngles': 'NeighborhoodAngles',
                'NeighborhoodChargeFractions': 'NeighborhoodChargeFractions',
                
                # Charge uncertainties (5% of max charge)
                'GaussRowChargeErr': 'GaussRowChargeErr',
                'GaussColChargeErr': 'GaussColChargeErr',
                'LorentzRowChargeErr': 'LorentzRowChargeErr',
                'LorentzColChargeErr': 'LorentzColChargeErr',
                
                # Nearest pixel poss
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
                    # Suppress warnings for Power Lorentz branches if they're expected to be missing
                    if 'PowerLorentz' in expected_name:
                        skipped_power_count += 1
                    else:
                        print(f"Warning: Branch {actual_name} not found for {expected_name}")
            
            if skipped_power_count > 0:
                print(f"Note: Skipped {skipped_power_count} Power Lorentz branches (fitting disabled)")
            
            print(f"Successly loaded {loaded_count} branches with {len(data['TrueX'])} events")
            
            # Detect err branch availability
            err_status = detect_err_branches(data)
            data['_err_status'] = err_status
            
            # Print err detection results
            if err_status['any_uncertainties_available']:
                print("Charge err branches detected:")
                if err_status['gauss_row_err_available']:
                    print("  ✓ Gauss row uncertainties available")
                if err_status['gauss_col_err_available']:
                    print("  ✓ Gauss column uncertainties available") 
                if err_status['lorentz_row_err_available']:
                    print("  ✓ Lorentz row uncertainties available")
                if err_status['lorentz_col_err_available']:
                    print("  ✓ Lorentz column uncertainties available")
            else:
                print("No charge err branches detected - plots will show data points without error bars")
            
            # Create success flags since they're not individual branches in this ROOT file
            # We'll check if the chi2 values are reasonable (non-zero and finite)
            n_events = len(data['TrueX'])
            
            # Create success flags based on available fit data
            data['2D_Success'] = np.ones(n_events, dtype=bool)  # Assume all success for now
            data['2D_Lorentz_Success'] = np.ones(n_events, dtype=bool)
            data['2D_PowerLorentz_Success'] = np.ones(n_events, dtype=bool)
            data['Diag_Success'] = np.ones(n_events, dtype=bool)
            data['Diag_MainXSuccess'] = np.ones(n_events, dtype=bool)
            data['Diag_MainYSuccess'] = np.ones(n_events, dtype=bool)
            data['Diag_SecXSuccess'] = np.ones(n_events, dtype=bool)
            data['Diag_SecYSuccess'] = np.ones(n_events, dtype=bool)
            
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
    Now properly extracts from the full 9x9 grid stored in NeighborhoodCharges branch.
    Also extracts charge uncertainties (5% of max charge) for error bars.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (row_data, col_data) where each is (poss, charges, uncertainties) for central row/column
    """
    
    # First try to use the raw neighborhood grid data (preferred method)
    if 'NeighborhoodCharge' in data and event_idx < len(data['NeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NeighborhoodCharge'][event_idx]
            grid_charge_fractions = data['NeighborhoodChargeFractions'][event_idx] if 'NeighborhoodChargeFractions' in data else None
            
            # Get nearest pixel pos for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract central row (Y = center_y, varying X)
            row_poss = []
            row_charges = []
            
            # Central row corresponds to j = neighborhood_radius (middle of grid)
            center_row = neighborhood_radius
            for i in range(grid_size):  # i varies from 0 to 8 (X direction)
                grid_idx = i * grid_size + center_row  # Col-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calc X pos for this pixel
                    offset_i = i - neighborhood_radius  # -4 to +4
                    x_pos = center_x + offset_i * pixel_spacing
                    row_poss.append(x_pos)
                    row_charges.append(grid_charges[grid_idx])
            
            # Extract central column (X = center_x, varying Y)
            col_poss = []
            col_charges = []
            
            # Central column corresponds to i = neighborhood_radius (middle of grid)
            center_col = neighborhood_radius
            for j in range(grid_size):  # j varies from 0 to 8 (Y direction)
                grid_idx = center_col * grid_size + j  # Col-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calc Y pos for this pixel
                    offset_j = j - neighborhood_radius  # -4 to +4
                    y_pos = center_y + offset_j * pixel_spacing
                    col_poss.append(y_pos)
                    col_charges.append(grid_charges[grid_idx])
            
            row_poss = np.array(row_poss)
            row_charge_values = np.array(row_charges)
            col_poss = np.array(col_poss)
            col_charge_values = np.array(col_charges)
            
            # Extract charge uncertainties (5% of max charge for each direction) if available
            err_status = data.get('_err_status', {})
            
            if err_status.get('gauss_row_err_available', False):
                row_err = data.get('GaussRowChargeErr', [0])[event_idx] if 'GaussRowChargeErr' in data and event_idx < len(data['GaussRowChargeErr']) else 0
            else:
                row_err = 0
                
            if err_status.get('gauss_col_err_available', False):
                col_err = data.get('GaussColChargeErr', [0])[event_idx] if 'GaussColChargeErr' in data and event_idx < len(data['GaussColChargeErr']) else 0
            else:
                col_err = 0
            
            # Create err arrays (same err for all points in a line)
            row_uncertainties = np.full(len(row_poss), row_err)
            col_uncertainties = np.full(len(col_poss), col_err)
            
            print(f"Event {event_idx}: Extracted {len(row_poss)} row points, {len(col_poss)} col points from grid")
            if err_status.get('any_uncertainties_available', False):
                print(f"  Row err: {row_err:.2e} C, Col err: {col_err:.2e} C")
            else:
                print(f"  No uncertainties available - using data points without error bars")
            
        except Exception as e:
            print(f"Warning: Could not extract grid data for event {event_idx}: {e}")
            # Fall back to fit results
            row_poss, row_charge_values, row_uncertainties = extract_from_fit_results(event_idx, data, 'row')
            col_poss, col_charge_values, col_uncertainties = extract_from_fit_results(event_idx, data, 'col')
    else:
        # Fall back to fit results if grid data not available
        print(f"Warning: NeighborhoodCharge not available for event {event_idx}, using fit results")
        row_poss, row_charge_values, row_uncertainties = extract_from_fit_results(event_idx, data, 'row')
        col_poss, col_charge_values, col_uncertainties = extract_from_fit_results(event_idx, data, 'col')
    
    return (row_poss, row_charge_values, row_uncertainties), (col_poss, col_charge_values, col_uncertainties)


def extract_from_fit_results(event_idx, data, direction):
    """
    Extract data from fit results as fallback when grid data is not available.
    Now also extracts charge uncertainties.
    """
    try:
        if direction == 'row':
            if 'LorentzRowPixelCoords' in data and event_idx < len(data['LorentzRowPixelCoords']):
                coords = data['LorentzRowPixelCoords'][event_idx]
                charges = data['LorentzRowChargeValues'][event_idx]
                poss = np.array(coords) if hasattr(coords, '__len__') else np.array([])
                charge_values = np.array(charges) if hasattr(charges, '__len__') else np.array([])
            else:
                # Create synthetic row data
                center = data['2D_XCenter'][event_idx] if '2D_XCenter' in data else data['PixelX'][event_idx]
                sigma = data['2D_XSigma'][event_idx] if '2D_XSigma' in data else 0.5
                amp = data['2D_XAmp'][event_idx] if '2D_XAmp' in data else 1e-12
                pixel_spacing = 0.5
                poss = np.array([center + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
                charge_values = amp * np.exp(-0.5 * ((poss - center) / sigma)**2)
            
            # Extract row err if available
            err_status = data.get('_err_status', {})
            if err_status.get('gauss_row_err_available', False):
                err = data.get('GaussRowChargeErr', [0])[event_idx] if 'GaussRowChargeErr' in data and event_idx < len(data['GaussRowChargeErr']) else 0
            else:
                err = 0
            uncertainties = np.full(len(poss), err)
            
        else:  # column
            if 'LorentzColPixelCoords' in data and event_idx < len(data['LorentzColPixelCoords']):
                coords = data['LorentzColPixelCoords'][event_idx]
                charges = data['LorentzColChargeValues'][event_idx]
                poss = np.array(coords) if hasattr(coords, '__len__') else np.array([])
                charge_values = np.array(charges) if hasattr(charges, '__len__') else np.array([])
            else:
                # Create synthetic column data
                center = data['2D_YCenter'][event_idx] if '2D_YCenter' in data else data['PixelY'][event_idx]
                sigma = data['2D_YSigma'][event_idx] if '2D_YSigma' in data else 0.5
                amp = data['2D_YAmp'][event_idx] if '2D_YAmp' in data else 1e-12
                pixel_spacing = 0.5
                poss = np.array([center + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
                charge_values = amp * np.exp(-0.5 * ((poss - center) / sigma)**2)
            
            # Extract column err if available
            err_status = data.get('_err_status', {})
            if err_status.get('gauss_col_err_available', False):
                err = data.get('GaussColChargeErr', [0])[event_idx] if 'GaussColChargeErr' in data and event_idx < len(data['GaussColChargeErr']) else 0
            else:
                err = 0
            uncertainties = np.full(len(poss), err)
                
    except Exception as e:
        print(f"Warning: Could not extract {direction} data for event {event_idx}: {e}")
        poss = np.array([])
        charge_values = np.array([])
        uncertainties = np.array([])
    
    return poss, charge_values, uncertainties


def extract_full_grid_data(event_idx, data, neighborhood_radius=4):
    """
    Extract all pixels with charge from the full neighborhood grid for 2D visualization.
    
    Args:
        event_idx (int): Event index
        data (dict): Filtered data dictionary
        neighborhood_radius (int): Radius of neighborhood grid (default: 4 for 9x9)
    
    Returns:
        tuple: (x_poss, y_poss, charge_values) for all pixels with charge > 0
    """
    
    if 'NeighborhoodCharge' in data and event_idx < len(data['NeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NeighborhoodCharge'][event_idx]
            
            # Get nearest pixel pos for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract all pixels with charge > 0
            x_poss = []
            y_poss = []
            charge_values = []
            
            for i in range(grid_size):  # X direction (columns)
                for j in range(grid_size):  # Y direction (rows)
                    grid_idx = i * grid_size + j  # Col-major indexing
                    if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                        # Calc actual pos for this pixel
                        offset_i = i - neighborhood_radius  # -4 to +4 for X
                        offset_j = j - neighborhood_radius  # -4 to +4 for Y
                        x_pos = center_x + offset_i * pixel_spacing
                        y_pos = center_y + offset_j * pixel_spacing
                        
                        x_poss.append(x_pos)
                        y_poss.append(y_pos)
                        charge_values.append(grid_charges[grid_idx])
            
            x_poss = np.array(x_poss)
            y_poss = np.array(y_poss)
            charge_values = np.array(charge_values)
            
            print(f"Event {event_idx}: Extracted {len(x_poss)} total grid points with charge > 0")
            
            return x_poss, y_poss, charge_values
            
        except Exception as e:
            print(f"Warning: Could not extract full grid data for event {event_idx}: {e}")
            return np.array([]), np.array([]), np.array([])
    else:
        print(f"Warning: NeighborhoodCharge not available for event {event_idx}")
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
    if 'NeighborhoodCharge' in data and event_idx < len(data['NeighborhoodCharge']):
        try:
            # Extract raw grid data
            grid_charges = data['NeighborhoodCharge'][event_idx]
            
            # Get nearest pixel pos for reference
            center_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
            center_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
            
            # Grid parameters
            grid_size = 2 * neighborhood_radius + 1  # Should be 9 for radius 4
            pixel_spacing = 0.5  # mm
            
            # Extract main diagonal (i == j, top-left to bottom-right)
            main_x_poss = []
            main_x_charges = []
            main_y_poss = []
            main_y_charges = []
            
            for k in range(grid_size):  # k varies from 0 to 8
                i = k  # X index
                j = k  # Y index (same as X for main diagonal)
                grid_idx = i * grid_size + j  # Col-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calc pos for this diagonal pixel
                    offset_i = i - neighborhood_radius  # -4 to +4 for X
                    offset_j = j - neighborhood_radius  # -4 to +4 for Y
                    x_pos = center_x + offset_i * pixel_spacing
                    y_pos = center_y + offset_j * pixel_spacing
                    
                    # For diagonal plots, we need to project to 1D coordinate
                    # Main diagonal: use distance along diagonal from center
                    diag_coord = offset_i * pixel_spacing * np.sqrt(2)  # √2 × pitch per step
                    
                    main_x_poss.append(diag_coord)
                    main_x_charges.append(grid_charges[grid_idx])
                    main_y_poss.append(diag_coord)  # Same coordinate for both X and Y projections
                    main_y_charges.append(grid_charges[grid_idx])
            
            # Extract secondary diagonal (i + j == grid_size - 1, top-right to bottom-left)
            sec_x_poss = []
            sec_x_charges = []
            sec_y_poss = []
            sec_y_charges = []
            
            for k in range(grid_size):  # k varies from 0 to 8
                i = k  # X index
                j = grid_size - 1 - k  # Y index (complementary for secondary diagonal)
                grid_idx = i * grid_size + j  # Col-major indexing
                if grid_idx < len(grid_charges) and grid_charges[grid_idx] > 0:
                    # Calc pos for this diagonal pixel
                    offset_i = i - neighborhood_radius  # -4 to +4 for X
                    offset_j = j - neighborhood_radius  # -4 to +4 for Y
                    
                    # For secondary diagonal: use distance along diagonal from center
                    # Secondary diagonal runs from top-right to bottom-left
                    diag_coord = offset_i * pixel_spacing * np.sqrt(2)  # X coordinate along the diagonal
                    
                    sec_x_poss.append(diag_coord)
                    sec_x_charges.append(grid_charges[grid_idx])
                    sec_y_poss.append(diag_coord)
                    sec_y_charges.append(grid_charges[grid_idx])
            
            main_x_poss = np.array(main_x_poss)
            main_x_charge_values = np.array(main_x_charges)
            main_y_poss = np.array(main_y_poss)
            main_y_charge_values = np.array(main_y_charges)
            sec_x_poss = np.array(sec_x_poss)
            sec_x_charge_values = np.array(sec_x_charges)
            sec_y_poss = np.array(sec_y_poss)
            sec_y_charge_values = np.array(sec_y_charges)
            
            print(f"Event {event_idx}: Extracted {len(main_x_poss)} main diagonal points, {len(sec_x_poss)} secondary diagonal points")
            
        except Exception as e:
            print(f"Warning: Could not extract diagonal grid data for event {event_idx}: {e}")
            # Fall back to fit results
            return extract_diagonal_from_fit_results(event_idx, data)
    else:
        # Fall back to fit results if grid data not available
        print(f"Warning: NeighborhoodCharge not available for event {event_idx}, using fit results for diagonals")
        return extract_diagonal_from_fit_results(event_idx, data)
    
    return ((main_x_poss, main_x_charge_values), 
            (main_y_poss, main_y_charge_values),
            (sec_x_poss, sec_x_charge_values), 
            (sec_y_poss, sec_y_charge_values))


def extract_diagonal_from_fit_results(event_idx, data):
    """
    Extract diagonal data from fit results as fallback.
    """
    # Try to extract main diagonal data from Lorentz fit results
    try:
        if 'LorentzMainDiagXPixelCoords' in data and event_idx < len(data['LorentzMainDiagXPixelCoords']):
            main_x_coords = data['LorentzMainDiagXPixelCoords'][event_idx]
            main_x_charges = data['LorentzMainDiagXChargeValues'][event_idx]
            main_x_poss = np.array(main_x_coords) if hasattr(main_x_coords, '__len__') else np.array([])
            main_x_charge_values = np.array(main_x_charges) if hasattr(main_x_charges, '__len__') else np.array([])
        else:
            # Create synthetic main diagonal X data
            center_x = data['Diag_MainXCenter'][event_idx] if 'Diag_MainXCenter' in data else data['PixelX'][event_idx]
            sigma = data['Diag_MainXSigma'][event_idx] if 'Diag_MainXSigma' in data else 0.5
            amp = data['Diag_MainXAmp'][event_idx] if 'Diag_MainXAmp' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            main_x_poss = np.array([center_x + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            main_x_charge_values = amp * np.exp(-0.5 * ((main_x_poss - center_x) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract main diagonal X data for event {event_idx}: {e}")
        main_x_poss = np.array([])
        main_x_charge_values = np.array([])
    
    try:
        if 'LorentzMainDiagYPixelCoords' in data and event_idx < len(data['LorentzMainDiagYPixelCoords']):
            main_y_coords = data['LorentzMainDiagYPixelCoords'][event_idx]
            main_y_charges = data['LorentzMainDiagYChargeValues'][event_idx]
            main_y_poss = np.array(main_y_coords) if hasattr(main_y_coords, '__len__') else np.array([])
            main_y_charge_values = np.array(main_y_charges) if hasattr(main_y_charges, '__len__') else np.array([])
        else:
            # Create synthetic main diagonal Y data
            center_y = data['Diag_MainYCenter'][event_idx] if 'Diag_MainYCenter' in data else data['PixelY'][event_idx]
            sigma = data['Diag_MainYSigma'][event_idx] if 'Diag_MainYSigma' in data else 0.5
            amp = data['Diag_MainYAmp'][event_idx] if 'Diag_MainYAmp' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            main_y_poss = np.array([center_y + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            main_y_charge_values = amp * np.exp(-0.5 * ((main_y_poss - center_y) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract main diagonal Y data for event {event_idx}: {e}")
        main_y_poss = np.array([])
        main_y_charge_values = np.array([])
    
    try:
        if 'LorentzSecDiagXPixelCoords' in data and event_idx < len(data['LorentzSecDiagXPixelCoords']):
            sec_x_coords = data['LorentzSecDiagXPixelCoords'][event_idx]
            sec_x_charges = data['LorentzSecDiagXChargeValues'][event_idx]
            sec_x_poss = np.array(sec_x_coords) if hasattr(sec_x_coords, '__len__') else np.array([])
            sec_x_charge_values = np.array(sec_x_charges) if hasattr(sec_x_charges, '__len__') else np.array([])
        else:
            # Create synthetic secondary diagonal X data
            center_x = data['Diag_SecXCenter'][event_idx] if 'Diag_SecXCenter' in data else data['PixelX'][event_idx]
            sigma = data['Diag_SecXSigma'][event_idx] if 'Diag_SecXSigma' in data else 0.5
            amp = data['Diag_SecXAmp'][event_idx] if 'Diag_SecXAmp' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            sec_x_poss = np.array([center_x + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            sec_x_charge_values = amp * np.exp(-0.5 * ((sec_x_poss - center_x) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract secondary diagonal X data for event {event_idx}: {e}")
        sec_x_poss = np.array([])
        sec_x_charge_values = np.array([])
    
    try:
        if 'LorentzSecDiagYPixelCoords' in data and event_idx < len(data['LorentzSecDiagYPixelCoords']):
            sec_y_coords = data['LorentzSecDiagYPixelCoords'][event_idx]
            sec_y_charges = data['LorentzSecDiagYChargeValues'][event_idx]
            sec_y_poss = np.array(sec_y_coords) if hasattr(sec_y_coords, '__len__') else np.array([])
            sec_y_charge_values = np.array(sec_y_charges) if hasattr(sec_y_charges, '__len__') else np.array([])
        else:
            # Create synthetic secondary diagonal Y data
            center_y = data['Diag_SecYCenter'][event_idx] if 'Diag_SecYCenter' in data else data['PixelY'][event_idx]
            sigma = data['Diag_SecYSigma'][event_idx] if 'Diag_SecYSigma' in data else 0.5
            amp = data['Diag_SecYAmp'][event_idx] if 'Diag_SecYAmp' in data else 1e-12
            
            pixel_spacing = 0.5  # mm
            sec_y_poss = np.array([center_y + i * pixel_spacing for i in range(-4, 5)])  # Full 9 points
            sec_y_charge_values = amp * np.exp(-0.5 * ((sec_y_poss - center_y) / sigma)**2)
    except Exception as e:
        print(f"Warning: Could not extract secondary diagonal Y data for event {event_idx}: {e}")
        sec_y_poss = np.array([])
        sec_y_charge_values = np.array([])
    
    return ((main_x_poss, main_x_charge_values), 
            (main_y_poss, main_y_charge_values),
            (sec_x_poss, sec_x_charge_values), 
            (sec_y_poss, sec_y_charge_values))

def autoscale_axes(fig):
    """
    Auto-scale all axes in a figure for better visualization.
    """
    for ax in fig.get_axes():
        ax.relim()
        ax.autoscale_view()

def plot_data_points(ax, poss, charges, uncertainties, **kwargs):
    """
    Plot data points with or without error bars, depending on whether uncertainties are meaningful.
    
    Args:
        ax: Matplotlib axis object
        poss: X poss
        charges: Y values (charges)
        uncertainties: Err values
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
        return ax.errorbar(poss, charges, yerr=uncertainties, **kwargs)
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
            
        return ax.plot(poss, charges, **plot_kwargs)

def calculate_residuals(poss, charges, fit_params, fit_type='gauss'):
    """
    Calc residuals between data and fitted function.
    
    Args:
        poss (array): Pos values
        charges (array): Charge values (data)
        fit_params (dict): ted parameters with keys 'center', 'sigma'/'gamma'/'power', 'amp'
        fit_type (str): 'gauss', 'lorentz', or 'power_lorentz'
    
    Returns:
        array: Residuals (data - fit)
    """
    if len(poss) == 0:
        return np.array([])
    
    if fit_type == 'gauss':
        fitted_values = gauss_1d(poss, 
                                   fit_params['amp'], 
                                   fit_params['center'], 
                                   fit_params['sigma'])
    elif fit_type == 'lorentz':
        fitted_values = lorentz_1d(poss, 
                                     fit_params['amp'], 
                                     fit_params['center'], 
                                     fit_params['gamma'])
    elif fit_type == 'power_lorentz':
        fitted_values = power_lorentz_1d(poss, 
                                           fit_params['amp'], 
                                           fit_params['center'], 
                                           fit_params['gamma'],
                                           fit_params['power'])
    else:
        raise ValueError("fit_type must be 'gauss', 'lorentz', or 'power_lorentz'")
    
    return charges - fitted_values

def create_all_lorentz_plot(event_idx, data, output_dir="plots"):
    """
    Create Lorentz fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get Lorentz fit parameters
        x_lorentz_center = data['2D_Lorentz_XCenter'][event_idx]
        x_lorentz_gamma = data['2D_Lorentz_XGamma'][event_idx]
        x_lorentz_amp = data['2D_Lorentz_XAmp'][event_idx]
        x_lorentz_vert_offset = data.get('2D_Lorentz_XVertOffset', [0])[event_idx] if '2D_Lorentz_XVertOffset' in data else 0
        x_lorentz_chi2red = data['2D_Lorentz_XChi2red'][event_idx]
        x_lorentz_dof = data.get('2D_Lorentz_XNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        y_lorentz_center = data['2D_Lorentz_YCenter'][event_idx]
        y_lorentz_gamma = data['2D_Lorentz_YGamma'][event_idx]
        y_lorentz_amp = data['2D_Lorentz_YAmp'][event_idx]
        y_lorentz_vert_offset = data.get('2D_Lorentz_YVertOffset', [0])[event_idx] if '2D_Lorentz_YVertOffset' in data else 0
        y_lorentz_chi2red = data['2D_Lorentz_YChi2red'][event_idx]
        y_lorentz_dof = data.get('2D_Lorentz_YNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        # Diag parameters (treat as Lorentz with gamma = sigma)
        main_diag_x_center = data['Diag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['Diag_MainXSigma'][event_idx] 
        main_diag_x_amp = data['Diag_MainXAmp'][event_idx]
        main_diag_x_vert_offset = data.get('Diag_MainXVertOffset', [0])[event_idx] if 'Diag_MainXVertOffset' in data else 0
        main_diag_x_chi2red = data['Diag_MainXChi2red'][event_idx]
        main_diag_x_dof = data.get('Diag_MainXNPoints', [0])[event_idx] - 4
        
        main_diag_y_center = data['Diag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['Diag_MainYSigma'][event_idx]
        main_diag_y_amp = data['Diag_MainYAmp'][event_idx]
        main_diag_y_vert_offset = data.get('Diag_MainYVertOffset', [0])[event_idx] if 'Diag_MainYVertOffset' in data else 0
        main_diag_y_chi2red = data['Diag_MainYChi2red'][event_idx]
        main_diag_y_dof = data.get('Diag_MainYNPoints', [0])[event_idx] - 4
        
        sec_diag_x_center = data['Diag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['Diag_SecXSigma'][event_idx]
        sec_diag_x_amp = data['Diag_SecXAmp'][event_idx]
        sec_diag_x_vert_offset = data.get('Diag_SecXVertOffset', [0])[event_idx] if 'Diag_SecXVertOffset' in data else 0
        sec_diag_x_chi2red = data['Diag_SecXChi2red'][event_idx]
        sec_diag_x_dof = data.get('Diag_SecXNPoints', [0])[event_idx] - 4
        
        sec_diag_y_center = data['Diag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['Diag_SecYSigma'][event_idx]
        sec_diag_y_amp = data['Diag_SecYAmp'][event_idx]
        sec_diag_y_vert_offset = data.get('Diag_SecYVertOffset', [0])[event_idx] if 'Diag_SecYVertOffset' in data else 0
        sec_diag_y_chi2red = data['Diag_SecYChi2red'][event_idx]
        sec_diag_y_dof = data.get('Diag_SecYNPoints', [0])[event_idx] - 4
        
        # True poss and pixel poss
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calc delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        lorentz_dir = os.path.join(output_dir, "lorentz")
        os.makedirs(lorentz_dir, exist_ok=True)
        
        # Create all Lorentz collage plot with residuals
        fig_lor_all = plt.figure(figsize=(24, 15))
        gs_lor_all = GridSpec(3, 4, hspace=0.4, wspace=0.3)
        
        def plot_lorentz_direction(ax, poss, charges, uncertainties, center, gamma, amp, vert_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(poss) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return None
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, poss, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Lorentz fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(poss.min() - 0.1, poss.max() + 0.1, 200)
            y_fit = lorentz_1d(pos_range, amp, center, gamma, vert_offset)
            ax.plot(pos_range, y_fit, '-', color='#ff7f0e', linewidth=2.5, alpha=0.9)
            
            # Add vert lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='red', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calc fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Lorentz ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f' {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Calc residuals for residual plot
            fitted_values = lorentz_1d(poss, amp, center, gamma, vert_offset)
            residuals = charges - fitted_values
            
            return residuals
        
        def plot_residuals(ax, poss, residuals, true_pos, center, title, direction='x'):
            """Helper to plot residuals."""
            if len(poss) < 3 or residuals is None:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
                
            # Plot residuals
            ax.scatter(poss, residuals, c='black', s=20, alpha=0.7, marker='s')
            ax.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.8, label='Zero line')
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Pos')
            ax.axvline(center, color='red', linestyle=':', linewidth=2, alpha=0.8, label=' Center')
            
            # Calc RMS of residuals
            rms = np.sqrt(np.mean(residuals**2)) if len(residuals) > 0 else 0
            max_res = np.max(np.abs(residuals)) if len(residuals) > 0 else 0
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'Residuals (C)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add residual stats
            residual_text = (rf'RMS: {rms:.2e}' + '\n' + rf'Max: {max_res:.2e}')
            ax.text(0.02, 0.98, residual_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
        
        # Plot all directions using the helper function
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot and residuals
        ax_row = fig_lor_all.add_subplot(gs_lor_all[0, 0])
        row_residuals = plot_lorentz_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                                x_lorentz_center, x_lorentz_gamma, x_lorentz_amp, x_lorentz_vert_offset,
                                x_lorentz_chi2red, x_lorentz_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        ax_row_res = fig_lor_all.add_subplot(gs_lor_all[0, 1])
        plot_residuals(ax_row_res, row_pos, row_residuals, true_x, x_lorentz_center, 'X Row Residuals', 'x')
        
        # Col plot and residuals
        ax_col = fig_lor_all.add_subplot(gs_lor_all[0, 2])
        col_residuals = plot_lorentz_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                y_lorentz_center, y_lorentz_gamma, y_lorentz_amp, y_lorentz_vert_offset,
                                y_lorentz_chi2red, y_lorentz_dof, true_y, 'Y Col', 'y', delta_pixel_y)
        
        ax_col_res = fig_lor_all.add_subplot(gs_lor_all[0, 3])
        plot_residuals(ax_col_res, col_pos, col_residuals, true_y, y_lorentz_center, 'Y Col Residuals', 'y')
        
        # Main diagonal X plot and residuals
        ax_main_x = fig_lor_all.add_subplot(gs_lor_all[1, 0])
        main_x_residuals = plot_lorentz_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                main_diag_x_center, main_diag_x_sigma, main_diag_x_amp, main_diag_x_vert_offset,
                                main_diag_x_chi2red, main_diag_x_dof, true_x, 'X Main Diag', 'x', delta_pixel_x)
        
        ax_main_x_res = fig_lor_all.add_subplot(gs_lor_all[1, 1])
        plot_residuals(ax_main_x_res, main_x_pos, main_x_residuals, true_x, main_diag_x_center, 'X Main Diag Residuals', 'x')
        
        # Main diagonal Y plot and residuals
        ax_main_y = fig_lor_all.add_subplot(gs_lor_all[1, 2])
        main_y_residuals = plot_lorentz_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                main_diag_y_center, main_diag_y_sigma, main_diag_y_amp, main_diag_y_vert_offset,
                                main_diag_y_chi2red, main_diag_y_dof, true_y, 'Y Main Diag', 'y', delta_pixel_y)
        
        ax_main_y_res = fig_lor_all.add_subplot(gs_lor_all[1, 3])
        plot_residuals(ax_main_y_res, main_y_pos, main_y_residuals, true_y, main_diag_y_center, 'Y Main Diag Residuals', 'y')
        
        # Secondary diagonal X plot and residuals
        ax_sec_x = fig_lor_all.add_subplot(gs_lor_all[2, 0])
        sec_x_residuals = plot_lorentz_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amp, sec_diag_x_vert_offset,
                                sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'X Sec Diag', 'x', delta_pixel_x)
        
        ax_sec_x_res = fig_lor_all.add_subplot(gs_lor_all[2, 1])
        plot_residuals(ax_sec_x_res, sec_x_pos, sec_x_residuals, true_x, sec_diag_x_center, 'X Sec Diag Residuals', 'x')
        
        # Secondary diagonal Y plot and residuals
        ax_sec_y = fig_lor_all.add_subplot(gs_lor_all[2, 2])
        sec_y_residuals = plot_lorentz_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amp, sec_diag_y_vert_offset,
                                sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Y Sec Diag', 'y', delta_pixel_y)
        
        ax_sec_y_res = fig_lor_all.add_subplot(gs_lor_all[2, 3])
        plot_residuals(ax_sec_y_res, sec_y_pos, sec_y_residuals, true_y, sec_diag_y_center, 'Y Sec Diag Residuals', 'y')
        
        
        plt.suptitle(f'Event {event_idx}: Lorentz s & Residuals', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(lorentz_dir, f'event_{event_idx:04d}_all_lorentz.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Lorentz collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Lorentz plot - {e}"

def create_all_power_lorentz_plot(event_idx, data, output_dir="plots"):
    """
    Create Power Lorentz fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Check if Power Lorentz fit parameters are available
        if '2D_PowerLorentz_XCenter' not in data:
            return f"Event {event_idx}: Power Lorentz fit data not available"
        
        # Get Power Lorentz fit parameters
        x_power_center = data['2D_PowerLorentz_XCenter'][event_idx]
        x_power_gamma = data['2D_PowerLorentz_XGamma'][event_idx]
        x_power_amp = data['2D_PowerLorentz_XAmp'][event_idx]
        x_power_power = data.get('2D_PowerLorentz_XPower', [1.0])[event_idx] if '2D_PowerLorentz_XPower' in data else 1.0
        x_power_vert_offset = data.get('2D_PowerLorentz_XVertOffset', [0])[event_idx] if '2D_PowerLorentz_XVertOffset' in data else 0
        x_power_chi2red = data['2D_PowerLorentz_XChi2red'][event_idx]
        x_power_dof = data.get('2D_PowerLorentz_XNPoints', [0])[event_idx] - 5  # N - K, K=5 parameters (including offset)
        
        y_power_center = data['2D_PowerLorentz_YCenter'][event_idx]
        y_power_gamma = data['2D_PowerLorentz_YGamma'][event_idx]
        y_power_amp = data['2D_PowerLorentz_YAmp'][event_idx]
        y_power_power = data.get('2D_PowerLorentz_YPower', [1.0])[event_idx] if '2D_PowerLorentz_YPower' in data else 1.0
        y_power_vert_offset = data.get('2D_PowerLorentz_YVertOffset', [0])[event_idx] if '2D_PowerLorentz_YVertOffset' in data else 0
        y_power_chi2red = data['2D_PowerLorentz_YChi2red'][event_idx]
        y_power_dof = data.get('2D_PowerLorentz_YNPoints', [0])[event_idx] - 5  # N - K, K=5 parameters (including offset)
        
        # True poss and pixel poss
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calc delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        power_lorentz_dir = os.path.join(output_dir, "power_lorentz")
        os.makedirs(power_lorentz_dir, exist_ok=True)
        
        # Create all Power Lorentz collage plot with residuals
        fig_power_all = plt.figure(figsize=(24, 15))
        gs_power_all = GridSpec(3, 4, hspace=0.4, wspace=0.3)
        
        def plot_power_lorentz_direction(ax, poss, charges, uncertainties, center, gamma, amp, power, vert_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(poss) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return None
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, poss, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Power Lorentz fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(poss.min() - 0.1, poss.max() + 0.1, 200)
            y_fit = power_lorentz_1d(pos_range, amp, center, gamma, power, vert_offset)
            ax.plot(pos_range, y_fit, '-', color='#9467bd', linewidth=2.5, alpha=0.9)
            
            # Add vert lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='magenta', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calc fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Power Lorentz ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n'
                          f'Power = {power:.2f}\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f' {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Calc residuals for residual plot
            fitted_values = power_lorentz_1d(poss, amp, center, gamma, power, vert_offset)
            residuals = charges - fitted_values
            
            return residuals
        
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot and residuals
        ax_row = fig_power_all.add_subplot(gs_power_all[0, 0])
        row_residuals = plot_power_lorentz_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                                      x_power_center, x_power_gamma, x_power_amp, x_power_power, x_power_vert_offset,
                                      x_power_chi2red, x_power_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        ax_row_res = fig_power_all.add_subplot(gs_power_all[0, 1])
        plot_residuals(ax_row_res, row_pos, row_residuals, true_x, x_power_center, 'X Row Residuals', 'x')
        
        # Col plot and residuals
        ax_col = fig_power_all.add_subplot(gs_power_all[0, 2])
        col_residuals = plot_power_lorentz_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                      y_power_center, y_power_gamma, y_power_amp, y_power_power, y_power_vert_offset,
                                      y_power_chi2red, y_power_dof, true_y, 'Y Col', 'y', delta_pixel_y)
        
        ax_col_res = fig_power_all.add_subplot(gs_power_all[0, 3])
        plot_residuals(ax_col_res, col_pos, col_residuals, true_y, y_power_center, 'Y Col Residuals', 'y')
        
        # For diagonals, use Gauss parameters (Power Lorentz diagonals may not be implemented)
        main_diag_x_center = data.get('Diag_MainXCenter', [x_power_center])[event_idx]
        main_diag_x_sigma = data.get('Diag_MainXSigma', [x_power_gamma])[event_idx]
        main_diag_x_amp = data.get('Diag_MainXAmp', [x_power_amp])[event_idx]
        main_diag_x_chi2red = data.get('Diag_MainXChi2red', [1.0])[event_idx]
        main_diag_x_dof = data.get('Diag_MainXNPoints', [4])[event_idx] - 4
        
        # Main diagonal X plot (using Gauss fit for diagonal, but labeled as approximation)
        ax_main_x = fig_power_all.add_subplot(gs_power_all[1, 0])
        main_x_residuals = plot_power_lorentz_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                      main_diag_x_center, main_diag_x_sigma, main_diag_x_amp, 1.0, 0,
                                      main_diag_x_chi2red, main_diag_x_dof, true_x, 'Main Diag X (approx)', 'x', delta_pixel_x)
        
        ax_main_x_res = fig_power_all.add_subplot(gs_power_all[1, 1])
        plot_residuals(ax_main_x_res, main_x_pos, main_x_residuals, true_x, main_diag_x_center, 'Main Diag X Residuals', 'x')
        
        # Similar for other diagonals...
        main_diag_y_center = data.get('Diag_MainYCenter', [y_power_center])[event_idx]
        main_diag_y_sigma = data.get('Diag_MainYSigma', [y_power_gamma])[event_idx]
        main_diag_y_amp = data.get('Diag_MainYAmp', [y_power_amp])[event_idx]
        main_diag_y_chi2red = data.get('Diag_MainYChi2red', [1.0])[event_idx]
        main_diag_y_dof = data.get('Diag_MainYNPoints', [4])[event_idx] - 4
        
        ax_main_y = fig_power_all.add_subplot(gs_power_all[1, 2])
        main_y_residuals = plot_power_lorentz_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                      main_diag_y_center, main_diag_y_sigma, main_diag_y_amp, 1.0, 0,
                                      main_diag_y_chi2red, main_diag_y_dof, true_y, 'Main Diag Y (approx)', 'y', delta_pixel_y)
        
        ax_main_y_res = fig_power_all.add_subplot(gs_power_all[1, 3])
        plot_residuals(ax_main_y_res, main_y_pos, main_y_residuals, true_y, main_diag_y_center, 'Main Diag Y Residuals', 'y')
        
        sec_diag_x_center = data.get('Diag_SecXCenter', [x_power_center])[event_idx]
        sec_diag_x_sigma = data.get('Diag_SecXSigma', [x_power_gamma])[event_idx]
        sec_diag_x_amp = data.get('Diag_SecXAmp', [x_power_amp])[event_idx]
        sec_diag_x_chi2red = data.get('Diag_SecXChi2red', [1.0])[event_idx]
        sec_diag_x_dof = data.get('Diag_SecXNPoints', [4])[event_idx] - 4
        
        ax_sec_x = fig_power_all.add_subplot(gs_power_all[2, 0])
        sec_x_residuals = plot_power_lorentz_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                      sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amp, 1.0, 0,
                                      sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'Secondary Diag X (approx)', 'x', delta_pixel_x)
        
        ax_sec_x_res = fig_power_all.add_subplot(gs_power_all[2, 1])
        plot_residuals(ax_sec_x_res, sec_x_pos, sec_x_residuals, true_x, sec_diag_x_center, 'Secondary Diag X Residuals', 'x')
        
        sec_diag_y_center = data.get('Diag_SecYCenter', [y_power_center])[event_idx]
        sec_diag_y_sigma = data.get('Diag_SecYSigma', [y_power_gamma])[event_idx]
        sec_diag_y_amp = data.get('Diag_SecYAmp', [y_power_amp])[event_idx]
        sec_diag_y_chi2red = data.get('Diag_SecYChi2red', [1.0])[event_idx]
        sec_diag_y_dof = data.get('Diag_SecYNPoints', [4])[event_idx] - 4
        
        ax_sec_y = fig_power_all.add_subplot(gs_power_all[2, 2])
        sec_y_residuals = plot_power_lorentz_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                      sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amp, 1.0, 0,
                                      sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Secondary Diag Y (approx)', 'y', delta_pixel_y)
        
        ax_sec_y_res = fig_power_all.add_subplot(gs_power_all[2, 3])
        plot_residuals(ax_sec_y_res, sec_y_pos, sec_y_residuals, true_y, sec_diag_y_center, 'Secondary Diag Y Residuals', 'y')
        
        plt.suptitle(f'Event {event_idx}: Power Lorentz s & Residuals', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(power_lorentz_dir, f'event_{event_idx:04d}_all_power_lorentz.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Power Lorentz collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Power Lorentz plot - {e}"

def create_all_gauss_plot(event_idx, data, output_dir="plots"):
    """
    Create Gauss fit plots for ALL directions in a single collage for one event.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Get Gauss fit parameters
        x_gauss_center = data['2D_XCenter'][event_idx]
        x_gauss_sigma = data['2D_XSigma'][event_idx]
        x_gauss_amp = data['2D_XAmp'][event_idx]
        x_gauss_vert_offset = data.get('2D_XVertOffset', [0])[event_idx] if '2D_XVertOffset' in data else 0
        x_gauss_chi2red = data['2D_XChi2red'][event_idx]
        x_gauss_dof = data.get('2D_XNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        y_gauss_center = data['2D_YCenter'][event_idx]
        y_gauss_sigma = data['2D_YSigma'][event_idx]
        y_gauss_amp = data['2D_YAmp'][event_idx]
        y_gauss_vert_offset = data.get('2D_YVertOffset', [0])[event_idx] if '2D_YVertOffset' in data else 0
        y_gauss_chi2red = data['2D_YChi2red'][event_idx]
        y_gauss_dof = data.get('2D_YNPoints', [0])[event_idx] - 4  # N - K, K=4 parameters (including offset)
        
        # Diag parameters (using Gauss fits)
        main_diag_x_center = data['Diag_MainXCenter'][event_idx]
        main_diag_x_sigma = data['Diag_MainXSigma'][event_idx] 
        main_diag_x_amp = data['Diag_MainXAmp'][event_idx]
        main_diag_x_vert_offset = data.get('Diag_MainXVertOffset', [0])[event_idx] if 'Diag_MainXVertOffset' in data else 0
        main_diag_x_chi2red = data['Diag_MainXChi2red'][event_idx]
        main_diag_x_dof = data.get('Diag_MainXNPoints', [0])[event_idx] - 4
        
        main_diag_y_center = data['Diag_MainYCenter'][event_idx]
        main_diag_y_sigma = data['Diag_MainYSigma'][event_idx]
        main_diag_y_amp = data['Diag_MainYAmp'][event_idx]
        main_diag_y_vert_offset = data.get('Diag_MainYVertOffset', [0])[event_idx] if 'Diag_MainYVertOffset' in data else 0
        main_diag_y_chi2red = data['Diag_MainYChi2red'][event_idx]
        main_diag_y_dof = data.get('Diag_MainYNPoints', [0])[event_idx] - 4
        
        sec_diag_x_center = data['Diag_SecXCenter'][event_idx]
        sec_diag_x_sigma = data['Diag_SecXSigma'][event_idx]
        sec_diag_x_amp = data['Diag_SecXAmp'][event_idx]
        sec_diag_x_vert_offset = data.get('Diag_SecXVertOffset', [0])[event_idx] if 'Diag_SecXVertOffset' in data else 0
        sec_diag_x_chi2red = data['Diag_SecXChi2red'][event_idx]
        sec_diag_x_dof = data.get('Diag_SecXNPoints', [0])[event_idx] - 4
        
        sec_diag_y_center = data['Diag_SecYCenter'][event_idx]
        sec_diag_y_sigma = data['Diag_SecYSigma'][event_idx]
        sec_diag_y_amp = data['Diag_SecYAmp'][event_idx]
        sec_diag_y_vert_offset = data.get('Diag_SecYVertOffset', [0])[event_idx] if 'Diag_SecYVertOffset' in data else 0
        sec_diag_y_chi2red = data['Diag_SecYChi2red'][event_idx]
        sec_diag_y_dof = data.get('Diag_SecYNPoints', [0])[event_idx] - 4
        
        # True poss and pixel poss
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calc delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        gauss_dir = os.path.join(output_dir, "gauss")
        os.makedirs(gauss_dir, exist_ok=True)
        
        # Create all Gauss collage plot with residuals
        fig_gauss_all = plt.figure(figsize=(24, 15))
        gs_gauss_all = GridSpec(3, 4, hspace=0.4, wspace=0.3)
        
        def plot_gauss_direction(ax, poss, charges, uncertainties, center, sigma, amp, vert_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(poss) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return None
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, poss, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Gauss fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(poss.min() - 0.1, poss.max() + 0.1, 200)
            y_fit = gauss_1d(pos_range, amp, center, sigma, vert_offset)
            ax.plot(pos_range, y_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9)
            
            # Add vert lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='blue', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calc fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Gauss ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f' {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Calc residuals for residual plot
            fitted_values = gauss_1d(poss, amp, center, sigma, vert_offset)
            residuals = charges - fitted_values
            
            return residuals
        
        # Plot all directions using the helper function
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot and residuals
        ax_row = fig_gauss_all.add_subplot(gs_gauss_all[0, 0])
        row_residuals = plot_gauss_direction(ax_row, row_pos, row_charges, row_uncertainties, 
                              x_gauss_center, x_gauss_sigma, x_gauss_amp, x_gauss_vert_offset,
                              x_gauss_chi2red, x_gauss_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        ax_row_res = fig_gauss_all.add_subplot(gs_gauss_all[0, 1])
        plot_residuals(ax_row_res, row_pos, row_residuals, true_x, x_gauss_center, 'X Row Residuals', 'x')
        
        # Col plot and residuals
        ax_col = fig_gauss_all.add_subplot(gs_gauss_all[0, 2])
        col_residuals = plot_gauss_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                              y_gauss_center, y_gauss_sigma, y_gauss_amp, y_gauss_vert_offset,
                              y_gauss_chi2red, y_gauss_dof, true_y, 'Y Col', 'y', delta_pixel_y)
        
        ax_col_res = fig_gauss_all.add_subplot(gs_gauss_all[0, 3])
        plot_residuals(ax_col_res, col_pos, col_residuals, true_y, y_gauss_center, 'Y Col Residuals', 'y')
        
        # Main diagonal X plot and residuals
        ax_main_x = fig_gauss_all.add_subplot(gs_gauss_all[1, 0])
        main_x_residuals = plot_gauss_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                              main_diag_x_center, main_diag_x_sigma, main_diag_x_amp, main_diag_x_vert_offset,
                              main_diag_x_chi2red, main_diag_x_dof, true_x, 'X Main Diag', 'x', delta_pixel_x)
        
        ax_main_x_res = fig_gauss_all.add_subplot(gs_gauss_all[1, 1])
        plot_residuals(ax_main_x_res, main_x_pos, main_x_residuals, true_x, main_diag_x_center, 'X Main Diag Residuals', 'x')
        
        # Main diagonal Y plot and residuals
        ax_main_y = fig_gauss_all.add_subplot(gs_gauss_all[1, 2])
        main_y_residuals = plot_gauss_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                              main_diag_y_center, main_diag_y_sigma, main_diag_y_amp, main_diag_y_vert_offset,
                              main_diag_y_chi2red, main_diag_y_dof, true_y, 'Y Main Diag', 'y', delta_pixel_y)
        
        ax_main_y_res = fig_gauss_all.add_subplot(gs_gauss_all[1, 3])
        plot_residuals(ax_main_y_res, main_y_pos, main_y_residuals, true_y, main_diag_y_center, 'Y Main Diag Residuals', 'y')
        
        # Secondary diagonal X plot and residuals
        ax_sec_x = fig_gauss_all.add_subplot(gs_gauss_all[2, 0])
        sec_x_residuals = plot_gauss_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                              sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amp, sec_diag_x_vert_offset,
                              sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'X Sec Diag', 'x', delta_pixel_x)
        
        ax_sec_x_res = fig_gauss_all.add_subplot(gs_gauss_all[2, 1])
        plot_residuals(ax_sec_x_res, sec_x_pos, sec_x_residuals, true_x, sec_diag_x_center, 'X Sec Diag Residuals', 'x')
        
        # Secondary diagonal Y plot and residuals
        ax_sec_y = fig_gauss_all.add_subplot(gs_gauss_all[2, 2])
        sec_y_residuals = plot_gauss_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                              sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amp, sec_diag_y_vert_offset,
                              sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Y Sec Diag', 'y', delta_pixel_y)
        
        ax_sec_y_res = fig_gauss_all.add_subplot(gs_gauss_all[2, 3])
        plot_residuals(ax_sec_y_res, sec_y_pos, sec_y_residuals, true_y, sec_diag_y_center, 'Y Sec Diag Residuals', 'y')
        
        plt.suptitle(f'Event {event_idx}: Gauss s & Residuals', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(gauss_dir, f'event_{event_idx:04d}_all_gauss.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: All Gauss collage plot created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating all Gauss plot - {e}"

def create_all_models_combined_plot(event_idx, data, output_dir="plots"):
    """
    Create combined Gauss, Lorentz, and Power Lorentz fit plots for ALL directions in a single collage for one event.
    Automatically detects which models are available and shows all available fits.
    """
    try:
        # Extract all data
        (row_pos, row_charges, row_uncertainties), (col_pos, col_charges, col_uncertainties) = extract_row_column_data(event_idx, data)
        (main_x_pos, main_x_charges), (main_y_pos, main_y_charges), (sec_x_pos, sec_x_charges), (sec_y_pos, sec_y_charges) = extract_diagonal_data(event_idx, data)
        
        if len(row_pos) < 3 and len(col_pos) < 3:
            return f"Event {event_idx}: Not enough data points for plotting"
        
        # Check which models are available
        has_gauss = '2D_XCenter' in data
        has_lorentz = '2D_Lorentz_XCenter' in data
        has_power_lorentz = '2D_PowerLorentz_XCenter' in data
        
        available_models = []
        if has_gauss:
            available_models.append('Gauss')
        if has_lorentz:
            available_models.append('Lorentz')
        if has_power_lorentz:
            available_models.append('Power Lorentz')
        
        if not available_models:
            return f"Event {event_idx}: No fitting models available"
        
        print(f"Event {event_idx}: Available models: {', '.join(available_models)}")
        
        # Get Gauss fit parameters if available
        if has_gauss:
            x_gauss_center = data['2D_XCenter'][event_idx]
            x_gauss_sigma = data['2D_XSigma'][event_idx]
            x_gauss_amp = data['2D_XAmp'][event_idx]
            x_gauss_chi2red = data['2D_XChi2red'][event_idx]
            
            y_gauss_center = data['2D_YCenter'][event_idx]
            y_gauss_sigma = data['2D_YSigma'][event_idx]
            y_gauss_amp = data['2D_YAmp'][event_idx]
            y_gauss_chi2red = data['2D_YChi2red'][event_idx]
        
        # Get Lorentz fit parameters if available
        if has_lorentz:
            x_lorentz_center = data['2D_Lorentz_XCenter'][event_idx]
            x_lorentz_gamma = data['2D_Lorentz_XGamma'][event_idx]
            x_lorentz_amp = data['2D_Lorentz_XAmp'][event_idx]
            x_lorentz_chi2red = data['2D_Lorentz_XChi2red'][event_idx]
            
            y_lorentz_center = data['2D_Lorentz_YCenter'][event_idx]
            y_lorentz_gamma = data['2D_Lorentz_YGamma'][event_idx]
            y_lorentz_amp = data['2D_Lorentz_YAmp'][event_idx]
            y_lorentz_chi2red = data['2D_Lorentz_YChi2red'][event_idx]
        
        # Get Power Lorentz fit parameters if available
        if has_power_lorentz:
            x_power_center = data['2D_PowerLorentz_XCenter'][event_idx]
            x_power_gamma = data['2D_PowerLorentz_XGamma'][event_idx]
            x_power_amp = data['2D_PowerLorentz_XAmp'][event_idx]
            x_power_power = data.get('2D_PowerLorentz_XPower', [1.0])[event_idx] if '2D_PowerLorentz_XPower' in data else 1.0
            x_power_vert_offset = data.get('2D_PowerLorentz_XVertOffset', [0])[event_idx] if '2D_PowerLorentz_XVertOffset' in data else 0
            x_power_chi2red = data['2D_PowerLorentz_XChi2red'][event_idx]
            
            y_power_center = data['2D_PowerLorentz_YCenter'][event_idx]
            y_power_gamma = data['2D_PowerLorentz_YGamma'][event_idx]
            y_power_amp = data['2D_PowerLorentz_YAmp'][event_idx]
            y_power_power = data.get('2D_PowerLorentz_YPower', [1.0])[event_idx] if '2D_PowerLorentz_YPower' in data else 1.0
            y_power_vert_offset = data.get('2D_PowerLorentz_YVertOffset', [0])[event_idx] if '2D_PowerLorentz_YVertOffset' in data else 0
            y_power_chi2red = data['2D_PowerLorentz_YChi2red'][event_idx]
        
        # Diag parameters (using Gauss fits)
        main_diag_x_center = data.get('Diag_MainXCenter', [0])[event_idx] if 'Diag_MainXCenter' in data else 0
        main_diag_x_sigma = data.get('Diag_MainXSigma', [0.1])[event_idx] if 'Diag_MainXSigma' in data else 0.1
        main_diag_x_amp = data.get('Diag_MainXAmp', [1e-12])[event_idx] if 'Diag_MainXAmp' in data else 1e-12
        main_diag_x_chi2red = data.get('Diag_MainXChi2red', [1.0])[event_idx] if 'Diag_MainXChi2red' in data else 1.0
        
        # True poss and pixel poss
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['NearestPixelX'][event_idx] if 'NearestPixelX' in data else data['PixelX'][event_idx]
        pixel_y = data['NearestPixelY'][event_idx] if 'NearestPixelY' in data else data['PixelY'][event_idx]
        
        # Calc delta pixel values (pixel - true)
        delta_pixel_x = pixel_x - true_x
        delta_pixel_y = pixel_y - true_y
        
        # Create output directory
        combined_dir = os.path.join(output_dir, "all_models_combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        # Create combined plot with residuals
        fig_combined = plt.figure(figsize=(24, 15))
        gs_combined = GridSpec(3, 4, hspace=0.4, wspace=0.3)
        
        def plot_all_models_direction(ax, poss, charges, uncertainties, true_pos, title, direction='x', delta_pixel=0, diagonal_type=None):
            """Helper to plot one direction with all available fits."""
            if len(poss) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return None
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, poss, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot range for smooth curves
            pos_range = np.linspace(poss.min() - 0.1, poss.max() + 0.1, 200)
            
            legend_lines = []
            legend_text_parts = []
            residuals_dict = {}
            
            # Plot Gauss fit if available
            if has_gauss and direction == 'x':
                x_gauss_vert_offset = data.get('2D_XVertOffset', [0])[event_idx] if '2D_XVertOffset' in data else 0
                gauss_fit = gauss_1d(pos_range, x_gauss_amp, x_gauss_center, x_gauss_sigma, x_gauss_vert_offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gauss')[0]
                legend_lines.append(line)
                ax.axvline(x_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = x_gauss_center - true_pos
                legend_text_parts.append(f'Gauss: ' + r'$\chi^2/\nu$' + f' = {x_gauss_chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
                # Calc residuals
                gauss_fitted = gauss_1d(poss, x_gauss_amp, x_gauss_center, x_gauss_sigma, x_gauss_vert_offset)
                residuals_dict['gauss'] = charges - gauss_fitted
            elif has_gauss and direction == 'y':
                y_gauss_vert_offset = data.get('2D_YVertOffset', [0])[event_idx] if '2D_YVertOffset' in data else 0
                gauss_fit = gauss_1d(pos_range, y_gauss_amp, y_gauss_center, y_gauss_sigma, y_gauss_vert_offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gauss')[0]
                legend_lines.append(line)
                ax.axvline(y_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = y_gauss_center - true_pos
                legend_text_parts.append(f'Gauss: ' + r'$\chi^2/\nu$' + f' = {y_gauss_chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
                # Calc residuals
                gauss_fitted = gauss_1d(poss, y_gauss_amp, y_gauss_center, y_gauss_sigma, y_gauss_vert_offset)
                residuals_dict['gauss'] = charges - gauss_fitted
            
            # Plot Lorentz fit if available
            if has_lorentz and direction == 'x':
                x_lorentz_vert_offset = data.get('2D_Lorentz_XVertOffset', [0])[event_idx] if '2D_Lorentz_XVertOffset' in data else 0
                lorentz_fit = lorentz_1d(pos_range, x_lorentz_amp, x_lorentz_center, x_lorentz_gamma, x_lorentz_vert_offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentz')[0]
                legend_lines.append(line)
                ax.axvline(x_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = x_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentz: ' + r'$\chi^2/\nu$' + f' = {x_lorentz_chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
                # Calc residuals
                lorentz_fitted = lorentz_1d(poss, x_lorentz_amp, x_lorentz_center, x_lorentz_gamma, x_lorentz_vert_offset)
                residuals_dict['lorentz'] = charges - lorentz_fitted
            elif has_lorentz and direction == 'y':
                y_lorentz_vert_offset = data.get('2D_Lorentz_YVertOffset', [0])[event_idx] if '2D_Lorentz_YVertOffset' in data else 0
                lorentz_fit = lorentz_1d(pos_range, y_lorentz_amp, y_lorentz_center, y_lorentz_gamma, y_lorentz_vert_offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentz')[0]
                legend_lines.append(line)
                ax.axvline(y_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = y_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentz: ' + r'$\chi^2/\nu$' + f' = {y_lorentz_chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
                # Calc residuals
                lorentz_fitted = lorentz_1d(poss, y_lorentz_amp, y_lorentz_center, y_lorentz_gamma, y_lorentz_vert_offset)
                residuals_dict['lorentz'] = charges - lorentz_fitted
            
            # Plot Power Lorentz fit if available
            if has_power_lorentz and direction == 'x':
                power_fit = power_lorentz_1d(pos_range, x_power_amp, x_power_center, x_power_gamma, x_power_power, x_power_vert_offset)
                line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentz')[0]
                legend_lines.append(line)
                ax.axvline(x_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = x_power_center - true_pos
                legend_text_parts.append(f'Power Lorentz: ' + r'$\chi^2/\nu$' + f' = {x_power_chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
                # Calc residuals
                power_fitted = power_lorentz_1d(poss, x_power_amp, x_power_center, x_power_gamma, x_power_power, x_power_vert_offset)
                residuals_dict['power_lorentz'] = charges - power_fitted
            elif has_power_lorentz and direction == 'y':
                power_fit = power_lorentz_1d(pos_range, y_power_amp, y_power_center, y_power_gamma, y_power_power, y_power_vert_offset)
                line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentz')[0]
                legend_lines.append(line)
                ax.axvline(y_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = y_power_center - true_pos
                legend_text_parts.append(f'Power Lorentz: ' + r'$\chi^2/\nu$' + f' = {y_power_chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
                # Calc residuals
                power_fitted = power_lorentz_1d(poss, y_power_amp, y_power_center, y_power_gamma, y_power_power, y_power_vert_offset)
                residuals_dict['power_lorentz'] = charges - power_fitted
            
            # Add true pos line
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Pos')
            
            # Create legend text
            legend_text = '\n'.join(legend_text_parts)
            legend_text += '\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            return residuals_dict
        
        def plot_combined_residuals(ax, poss, residuals_dict, true_pos, title, direction='x'):
            """Helper to plot residuals from all available models."""
            if len(poss) < 3 or not residuals_dict:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
                
            # Plot residuals for each model
            colors = {'gauss': '#1f77b4', 'lorentz': '#ff7f0e', 'power_lorentz': '#9467bd'}
            markers = {'gauss': 'o', 'lorentz': 's', 'power_lorentz': '^'}
            
            for model, residuals in residuals_dict.items():
                if residuals is not None and len(residuals) > 0:
                    label = model.replace('_', ' ').title()
                    ax.scatter(poss, residuals, c=colors[model], s=20, alpha=0.7, 
                             marker=markers[model], label=f'{label} residuals')
            
            # Add reference lines
            ax.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.8)
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'Residuals (C)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # Create dummy uncertainties for diagonal data
        main_x_uncertainties = np.full(len(main_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        main_y_uncertainties = np.full(len(main_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        sec_x_uncertainties = np.full(len(sec_x_pos), row_uncertainties[0] if len(row_uncertainties) > 0 else 0)
        sec_y_uncertainties = np.full(len(sec_y_pos), col_uncertainties[0] if len(col_uncertainties) > 0 else 0)
        
        # Row plot and residuals
        ax_row = fig_combined.add_subplot(gs_combined[0, 0])
        row_residuals_dict = plot_all_models_direction(ax_row, row_pos, row_charges, row_uncertainties, true_x, 'X Row', 'x', delta_pixel_x)
        
        ax_row_res = fig_combined.add_subplot(gs_combined[0, 1])
        plot_combined_residuals(ax_row_res, row_pos, row_residuals_dict, true_x, 'X Row Residuals', 'x')
        
        # Col plot and residuals
        ax_col = fig_combined.add_subplot(gs_combined[0, 2])
        col_residuals_dict = plot_all_models_direction(ax_col, col_pos, col_charges, col_uncertainties, true_y, 'Y Col', 'y', delta_pixel_y)
        
        ax_col_res = fig_combined.add_subplot(gs_combined[0, 3])
        plot_combined_residuals(ax_col_res, col_pos, col_residuals_dict, true_y, 'Y Col Residuals', 'y')
        
        # Diag plots (all available diagonal fits)
        def plot_diagonal_direction(ax, poss, charges, uncertainties, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot diagonal direction with all available diagonal fits."""
            if len(poss) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, poss, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot range for smooth curves
            pos_range = np.linspace(poss.min() - 0.1, poss.max() + 0.1, 200)
            
            legend_lines = []
            legend_text_parts = []
            
            # Helper function to get diagonal parameters
            def get_diagonal_params(diag_type, direction, model):
                if model == 'gauss':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('Diag_MainXCenter', [np.nan])[event_idx] if 'Diag_MainXCenter' in data else np.nan
                        width = data.get('Diag_MainXSigma', [np.nan])[event_idx] if 'Diag_MainXSigma' in data else np.nan
                        amp = data.get('Diag_MainXAmp', [np.nan])[event_idx] if 'Diag_MainXAmp' in data else np.nan
                        offset = data.get('Diag_MainXVertOffset', [0])[event_idx] if 'Diag_MainXVertOffset' in data else 0
                        chi2red = data.get('Diag_MainXChi2red', [np.nan])[event_idx] if 'Diag_MainXChi2red' in data else np.nan
                        dof = data.get('Diag_MainXNPoints', [0])[event_idx] if 'Diag_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('Diag_MainYCenter', [np.nan])[event_idx] if 'Diag_MainYCenter' in data else np.nan
                        width = data.get('Diag_MainYSigma', [np.nan])[event_idx] if 'Diag_MainYSigma' in data else np.nan
                        amp = data.get('Diag_MainYAmp', [np.nan])[event_idx] if 'Diag_MainYAmp' in data else np.nan
                        offset = data.get('Diag_MainYVertOffset', [0])[event_idx] if 'Diag_MainYVertOffset' in data else 0
                        chi2red = data.get('Diag_MainYChi2red', [np.nan])[event_idx] if 'Diag_MainYChi2red' in data else np.nan
                        dof = data.get('Diag_MainYNPoints', [0])[event_idx] if 'Diag_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('Diag_SecXCenter', [np.nan])[event_idx] if 'Diag_SecXCenter' in data else np.nan
                        width = data.get('Diag_SecXSigma', [np.nan])[event_idx] if 'Diag_SecXSigma' in data else np.nan
                        amp = data.get('Diag_SecXAmp', [np.nan])[event_idx] if 'Diag_SecXAmp' in data else np.nan
                        offset = data.get('Diag_SecXVertOffset', [0])[event_idx] if 'Diag_SecXVertOffset' in data else 0
                        chi2red = data.get('Diag_SecXChi2red', [np.nan])[event_idx] if 'Diag_SecXChi2red' in data else np.nan
                        dof = data.get('Diag_SecXNPoints', [0])[event_idx] if 'Diag_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('Diag_SecYCenter', [np.nan])[event_idx] if 'Diag_SecYCenter' in data else np.nan
                        width = data.get('Diag_SecYSigma', [np.nan])[event_idx] if 'Diag_SecYSigma' in data else np.nan
                        amp = data.get('Diag_SecYAmp', [np.nan])[event_idx] if 'Diag_SecYAmp' in data else np.nan
                        offset = data.get('Diag_SecYVertOffset', [0])[event_idx] if 'Diag_SecYVertOffset' in data else 0
                        chi2red = data.get('Diag_SecYChi2red', [np.nan])[event_idx] if 'Diag_SecYChi2red' in data else np.nan
                        dof = data.get('Diag_SecYNPoints', [0])[event_idx] if 'Diag_SecYNPoints' in data else 0
                elif model == 'lorentz':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('Diag_Lorentz_MainXCenter', [np.nan])[event_idx] if 'Diag_Lorentz_MainXCenter' in data else np.nan
                        width = data.get('Diag_Lorentz_MainXGamma', [np.nan])[event_idx] if 'Diag_Lorentz_MainXGamma' in data else np.nan
                        amp = data.get('Diag_Lorentz_MainXAmp', [np.nan])[event_idx] if 'Diag_Lorentz_MainXAmp' in data else np.nan
                        offset = data.get('Diag_Lorentz_MainXVertOffset', [0])[event_idx] if 'Diag_Lorentz_MainXVertOffset' in data else 0
                        chi2red = data.get('Diag_Lorentz_MainXChi2red', [np.nan])[event_idx] if 'Diag_Lorentz_MainXChi2red' in data else np.nan
                        dof = data.get('Diag_Lorentz_MainXNPoints', [0])[event_idx] if 'Diag_Lorentz_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('Diag_Lorentz_MainYCenter', [np.nan])[event_idx] if 'Diag_Lorentz_MainYCenter' in data else np.nan
                        width = data.get('Diag_Lorentz_MainYGamma', [np.nan])[event_idx] if 'Diag_Lorentz_MainYGamma' in data else np.nan
                        amp = data.get('Diag_Lorentz_MainYAmp', [np.nan])[event_idx] if 'Diag_Lorentz_MainYAmp' in data else np.nan
                        offset = data.get('Diag_Lorentz_MainYVertOffset', [0])[event_idx] if 'Diag_Lorentz_MainYVertOffset' in data else 0
                        chi2red = data.get('Diag_Lorentz_MainYChi2red', [np.nan])[event_idx] if 'Diag_Lorentz_MainYChi2red' in data else np.nan
                        dof = data.get('Diag_Lorentz_MainYNPoints', [0])[event_idx] if 'Diag_Lorentz_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('Diag_Lorentz_SecXCenter', [np.nan])[event_idx] if 'Diag_Lorentz_SecXCenter' in data else np.nan
                        width = data.get('Diag_Lorentz_SecXGamma', [np.nan])[event_idx] if 'Diag_Lorentz_SecXGamma' in data else np.nan
                        amp = data.get('Diag_Lorentz_SecXAmp', [np.nan])[event_idx] if 'Diag_Lorentz_SecXAmp' in data else np.nan
                        offset = data.get('Diag_Lorentz_SecXVertOffset', [0])[event_idx] if 'Diag_Lorentz_SecXVertOffset' in data else 0
                        chi2red = data.get('Diag_Lorentz_SecXChi2red', [np.nan])[event_idx] if 'Diag_Lorentz_SecXChi2red' in data else np.nan
                        dof = data.get('Diag_Lorentz_SecXNPoints', [0])[event_idx] if 'Diag_Lorentz_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('Diag_Lorentz_SecYCenter', [np.nan])[event_idx] if 'Diag_Lorentz_SecYCenter' in data else np.nan
                        width = data.get('Diag_Lorentz_SecYGamma', [np.nan])[event_idx] if 'Diag_Lorentz_SecYGamma' in data else np.nan
                        amp = data.get('Diag_Lorentz_SecYAmp', [np.nan])[event_idx] if 'Diag_Lorentz_SecYAmp' in data else np.nan
                        offset = data.get('Diag_Lorentz_SecYVertOffset', [0])[event_idx] if 'Diag_Lorentz_SecYVertOffset' in data else 0
                        chi2red = data.get('Diag_Lorentz_SecYChi2red', [np.nan])[event_idx] if 'Diag_Lorentz_SecYChi2red' in data else np.nan
                        dof = data.get('Diag_Lorentz_SecYNPoints', [0])[event_idx] if 'Diag_Lorentz_SecYNPoints' in data else 0
                elif model == 'power_lorentz':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('Diag_PowerLorentz_MainXCenter', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainXCenter' in data else np.nan
                        width = data.get('Diag_PowerLorentz_MainXGamma', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainXGamma' in data else np.nan
                        amp = data.get('Diag_PowerLorentz_MainXAmp', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainXAmp' in data else np.nan
                        power = data.get('Diag_PowerLorentz_MainXBeta', [1.0])[event_idx] if 'Diag_PowerLorentz_MainXBeta' in data else 1.0
                        offset = data.get('Diag_PowerLorentz_MainXVertOffset', [0])[event_idx] if 'Diag_PowerLorentz_MainXVertOffset' in data else 0
                        chi2red = data.get('Diag_PowerLorentz_MainXChi2red', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainXChi2red' in data else np.nan
                        dof = data.get('Diag_PowerLorentz_MainXNPoints', [0])[event_idx] if 'Diag_PowerLorentz_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('Diag_PowerLorentz_MainYCenter', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainYCenter' in data else np.nan
                        width = data.get('Diag_PowerLorentz_MainYGamma', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainYGamma' in data else np.nan
                        amp = data.get('Diag_PowerLorentz_MainYAmp', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainYAmp' in data else np.nan
                        power = data.get('Diag_PowerLorentz_MainYBeta', [1.0])[event_idx] if 'Diag_PowerLorentz_MainYBeta' in data else 1.0
                        offset = data.get('Diag_PowerLorentz_MainYVertOffset', [0])[event_idx] if 'Diag_PowerLorentz_MainYVertOffset' in data else 0
                        chi2red = data.get('Diag_PowerLorentz_MainYChi2red', [np.nan])[event_idx] if 'Diag_PowerLorentz_MainYChi2red' in data else np.nan
                        dof = data.get('Diag_PowerLorentz_MainYNPoints', [0])[event_idx] if 'Diag_PowerLorentz_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('Diag_PowerLorentz_SecXCenter', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecXCenter' in data else np.nan
                        width = data.get('Diag_PowerLorentz_SecXGamma', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecXGamma' in data else np.nan
                        amp = data.get('Diag_PowerLorentz_SecXAmp', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecXAmp' in data else np.nan
                        power = data.get('Diag_PowerLorentz_SecXBeta', [1.0])[event_idx] if 'Diag_PowerLorentz_SecXBeta' in data else 1.0
                        offset = data.get('Diag_PowerLorentz_SecXVertOffset', [0])[event_idx] if 'Diag_PowerLorentz_SecXVertOffset' in data else 0
                        chi2red = data.get('Diag_PowerLorentz_SecXChi2red', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecXChi2red' in data else np.nan
                        dof = data.get('Diag_PowerLorentz_SecXNPoints', [0])[event_idx] if 'Diag_PowerLorentz_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('Diag_PowerLorentz_SecYCenter', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecYCenter' in data else np.nan
                        width = data.get('Diag_PowerLorentz_SecYGamma', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecYGamma' in data else np.nan
                        amp = data.get('Diag_PowerLorentz_SecYAmp', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecYAmp' in data else np.nan
                        power = data.get('Diag_PowerLorentz_SecYBeta', [1.0])[event_idx] if 'Diag_PowerLorentz_SecYBeta' in data else 1.0
                        offset = data.get('Diag_PowerLorentz_SecYVertOffset', [0])[event_idx] if 'Diag_PowerLorentz_SecYVertOffset' in data else 0
                        chi2red = data.get('Diag_PowerLorentz_SecYChi2red', [np.nan])[event_idx] if 'Diag_PowerLorentz_SecYChi2red' in data else np.nan
                        dof = data.get('Diag_PowerLorentz_SecYNPoints', [0])[event_idx] if 'Diag_PowerLorentz_SecYNPoints' in data else 0
                
                if model == 'power_lorentz':
                    return center, width, amp, offset, chi2red, dof, power
                else:
                    return center, width, amp, offset, chi2red, dof
            
            # Determine diagonal type from title
            if 'Main' in title:
                diag_type = 'main'
            else:
                diag_type = 'sec'
            
            # Plot Gauss diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'gauss')
            center, width, amp, offset, chi2red, dof = params
            if not np.isnan(center) and not np.isnan(width) and not np.isnan(amp) and dof > 0:
                gauss_fit = gauss_1d(pos_range, amp, center, width, offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gauss')[0]
                legend_lines.append(line)
                ax.axvline(center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = center - true_pos
                legend_text_parts.append(f'Gauss: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
            
            # Plot Lorentz diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'lorentz')
            center, width, amp, offset, chi2red, dof = params
            if not np.isnan(center) and not np.isnan(width) and not np.isnan(amp) and dof > 0:
                lorentz_fit = lorentz_1d(pos_range, amp, center, width, offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentz')[0]
                legend_lines.append(line)
                ax.axvline(center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = center - true_pos
                legend_text_parts.append(f'Lorentz: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
            
            # Plot Power Lorentz diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'power_lorentz')
            if len(params) == 7:  # power_lorentz returns 7 params
                center, width, amp, offset, chi2red, dof, power = params
                if not np.isnan(center) and not np.isnan(width) and not np.isnan(amp) and dof > 0:
                    power_fit = power_lorentz_1d(pos_range, amp, center, width, power, offset)
                    line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentz')[0]
                    legend_lines.append(line)
                    ax.axvline(center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                    power_diff = center - true_pos
                    legend_text_parts.append(f'Power Lorentz: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
            
            # Add true pos line
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Pos')
            
            # Create legend text
            if legend_text_parts:
                legend_text = '\n'.join(legend_text_parts)
                legend_text += '\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            else:
                legend_text = 'No success diagonal fits\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   vertalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # Main diagonal X plot and residuals
        ax_main_x = fig_combined.add_subplot(gs_combined[1, 0])
        main_x_residuals_dict = plot_all_models_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, true_x, 'X Main Diag', 'x', delta_pixel_x, diagonal_type='main')
        
        ax_main_x_res = fig_combined.add_subplot(gs_combined[1, 1])
        plot_combined_residuals(ax_main_x_res, main_x_pos, main_x_residuals_dict, true_x, 'X Main Diag Residuals', 'x')
        
        # Main diagonal Y plot and residuals
        ax_main_y = fig_combined.add_subplot(gs_combined[1, 2])
        main_y_residuals_dict = plot_all_models_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, true_y, 'Y Main Diag', 'y', delta_pixel_y, diagonal_type='main')
        
        ax_main_y_res = fig_combined.add_subplot(gs_combined[1, 3])
        plot_combined_residuals(ax_main_y_res, main_y_pos, main_y_residuals_dict, true_y, 'Y Main Diag Residuals', 'y')
        
        # Secondary diagonal X plot and residuals
        ax_sec_x = fig_combined.add_subplot(gs_combined[2, 0])
        sec_x_residuals_dict = plot_all_models_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, true_x, 'X Sec Diag', 'x', delta_pixel_x, diagonal_type='sec')
        
        ax_sec_x_res = fig_combined.add_subplot(gs_combined[2, 1])
        plot_combined_residuals(ax_sec_x_res, sec_x_pos, sec_x_residuals_dict, true_x, 'X Sec Diag Residuals', 'x')
        
        # Secondary diagonal Y plot and residuals
        ax_sec_y = fig_combined.add_subplot(gs_combined[2, 2])
        sec_y_residuals_dict = plot_all_models_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, true_y, 'Y Sec Diag', 'y', delta_pixel_y, diagonal_type='sec')
        
        ax_sec_y_res = fig_combined.add_subplot(gs_combined[2, 3])
        plot_combined_residuals(ax_sec_y_res, sec_y_pos, sec_y_residuals_dict, true_y, 'Y Sec Diag Residuals', 'y')
        
        models_str = "_".join([m.lower().replace(" ", "_") for m in available_models])
        plt.suptitle(f'Event {event_idx}: Combined Models & Residuals ({", ".join(available_models)})', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f'event_{event_idx:04d}_all_models_combined.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return f"Event {event_idx}: Combined plot with {len(available_models)} models created successfully"
        
    except Exception as e:
        return f"Event {event_idx}: Error creating combined models plot - {e}"

def calculate_fit_quality_metric(data, event_idx):
    """
    Calc an overall fit quality metric for an event.
    Lower values indicate better fits.
    
    Args:
        data (dict): Data dictionary
        event_idx (int): Event index
    
    Returns:
        float: Combined fit quality metric (lower is better)
    """
    try:
        # Get chi-squared values for different fits
        x_gauss_chi2 = data.get('2D_XChi2red', [float('inf')])[event_idx]
        y_gauss_chi2 = data.get('2D_YChi2red', [float('inf')])[event_idx]
        x_lorentz_chi2 = data.get('2D_Lorentz_XChi2red', [float('inf')])[event_idx]
        y_lorentz_chi2 = data.get('2D_Lorentz_YChi2red', [float('inf')])[event_idx]
        x_power_chi2 = data.get('2D_PowerLorentz_XChi2red', [float('inf')])[event_idx]
        y_power_chi2 = data.get('2D_PowerLorentz_YChi2red', [float('inf')])[event_idx]
        
        # Get diagonal chi-squared values
        main_x_chi2 = data.get('Diag_MainXChi2red', [float('inf')])[event_idx]
        main_y_chi2 = data.get('Diag_MainYChi2red', [float('inf')])[event_idx]
        sec_x_chi2 = data.get('Diag_SecXChi2red', [float('inf')])[event_idx]
        sec_y_chi2 = data.get('Diag_SecYChi2red', [float('inf')])[event_idx]
        
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

def find_high_amp_events(data, n_events=10):
    """
    Find events with the highest amps to examine potential outliers.
    
    Args:
        data (dict): Data dictionary
        n_events (int): Number of high amp events to find
    
    Returns:
        tuple: (high_amp_indices, amp_metrics)
    """
    print("Finding events with highest amps...")
    
    n_total = len(data['TrueX'])
    amp_metrics = []
    
    for i in range(n_total):
        try:
            # Get Gauss amps for row and column fits
            x_gauss_amp = data.get('2D_XAmp', [0])[i] if '2D_XAmp' in data and i < len(data['2D_XAmp']) else 0
            y_gauss_amp = data.get('2D_YAmp', [0])[i] if '2D_YAmp' in data and i < len(data['2D_YAmp']) else 0
            
            # Get Lorentz amps for comparison
            x_lorentz_amp = data.get('2D_Lorentz_XAmp', [0])[i] if '2D_Lorentz_XAmp' in data and i < len(data['2D_Lorentz_XAmp']) else 0
            y_lorentz_amp = data.get('2D_Lorentz_YAmp', [0])[i] if '2D_Lorentz_YAmp' in data and i < len(data['2D_Lorentz_YAmp']) else 0
            
            # Get Power Lorentz amps for comparison
            x_power_amp = data.get('2D_PowerLorentz_XAmp', [0])[i] if '2D_PowerLorentz_XAmp' in data and i < len(data['2D_PowerLorentz_XAmp']) else 0
            y_power_amp = data.get('2D_PowerLorentz_YAmp', [0])[i] if '2D_PowerLorentz_YAmp' in data and i < len(data['2D_PowerLorentz_YAmp']) else 0
            
            # Use the maximum amp across all fits as the metric
            max_amp = max(abs(x_gauss_amp), abs(y_gauss_amp), abs(x_lorentz_amp), abs(y_lorentz_amp), abs(x_power_amp), abs(y_power_amp))
            
            # Also get chi2 values for quality assessment
            x_gauss_chi2 = data.get('2D_XChi2red', [float('inf')])[i]
            y_gauss_chi2 = data.get('2D_YChi2red', [float('inf')])[i]
            avg_chi2 = (x_gauss_chi2 + y_gauss_chi2) / 2.0 if np.isfinite(x_gauss_chi2) and np.isfinite(y_gauss_chi2) else float('inf')
            
            amp_metrics.append((i, max_amp, avg_chi2, x_gauss_amp, y_gauss_amp, x_lorentz_amp, y_lorentz_amp, x_power_amp, y_power_amp))
            
        except Exception as e:
            print(f"Warning: Could not extract amp data for event {i}: {e}")
            amp_metrics.append((i, 0, float('inf'), 0, 0, 0, 0, 0, 0))
    
    # Sort by amp (highest first)
    amp_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Get events with finite amps
    valid_amps = [(idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p) for idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p in amp_metrics if amp > 0 and np.isfinite(amp)]
    
    if len(valid_amps) == 0:
        print("Warning: No events with valid amps found!")
        return [], []
    
    print(f"Found {len(valid_amps)} events with valid amps out of {n_total} total events")
    
    # Get highest amp events
    high_amp_events = valid_amps[:n_events]
    high_amp_indices = [idx for idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p in high_amp_events]
    
    print(f"Highest amp events:")
    for i, (idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p) in enumerate(high_amp_events):
        print(f"  {i+1}. Event {idx}: Max Amp = {amp:.2e} C (χ² = {chi2:.3f})")
        print(f"      Gauss: X={x_g:.2e}, Y={y_g:.2e} | Lorentz: X={x_l:.2e}, Y={y_l:.2e} | Power: X={x_p:.2e}, Y={y_p:.2e}")
    
    return high_amp_indices, amp_metrics

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
        
        # Lorentz plot
        lorentz_result = create_all_lorentz_plot(event_idx, data, best_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  Best fit {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gauss plot
        gauss_result = create_all_gauss_plot(event_idx, data, best_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentz plot
        power_result = create_all_power_lorentz_plot(event_idx, data, best_dir)
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
        
        # Lorentz plot
        lorentz_result = create_all_lorentz_plot(event_idx, data, worst_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  Worst fit {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gauss plot
        gauss_result = create_all_gauss_plot(event_idx, data, worst_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentz plot
        power_result = create_all_power_lorentz_plot(event_idx, data, worst_dir)
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

def create_high_amp_plots(data, output_dir="plots", n_events=10):
    """
    Create plots for events with the highest amps to examine potential outliers.
    
    Args:
        data (dict): Data dictionary
        output_dir (str): Output directory for plots
        n_events (int): Number of high amp events to plot
    
    Returns:
        int: Number of successfully created plots
    """
    print(f"\nFinding {n_events} highest amp events...")
    high_amp_indices, amp_metrics = find_high_amp_events(data, n_events)
    
    if not high_amp_indices:
        print("No valid high amp events found!")
        return 0
    
    # Create subdirectory for high amp fits
    high_amp_dir = os.path.join(output_dir, "high_amps")
    os.makedirs(high_amp_dir, exist_ok=True)
    
    success_count = 0
    
    # Plot high amp events
    print(f"\nCreating plots for {len(high_amp_indices)} highest amp events...")
    for i, event_idx in enumerate(high_amp_indices):
        # Create all three types of plots for each high amp event
        
        # Lorentz plot
        lorentz_result = create_all_lorentz_plot(event_idx, data, high_amp_dir)
        if "Error" not in lorentz_result:
            success_count += 1
        print(f"  High amp {i+1} (Event {event_idx}): {lorentz_result}")
        
        # Gauss plot
        gauss_result = create_all_gauss_plot(event_idx, data, high_amp_dir)
        if "Error" not in gauss_result:
            success_count += 1
        
        # Power Lorentz plot
        power_result = create_all_power_lorentz_plot(event_idx, data, high_amp_dir)
        if "Error" not in power_result:
            success_count += 1
        
        # Combined plot with all models
        combined_result = create_all_models_combined_plot(event_idx, data, high_amp_dir)
        if "Error" not in combined_result:
            success_count += 1
    
    print(f"\nHigh amp fit plots saved to: {high_amp_dir}/")
    
    return success_count

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Create Gauss, Lorentz, and Power Lorentz fit plots for charge sharing analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file with 2D Gauss, Lorentz, and Power Lorentz fit data")
    parser.add_argument("-o", "--output", default="gauss_lorentz_plots", 
                       help="Output directory for plots")
    parser.add_argument("-n", "--num_events", type=int, default=10,
                       help="Number of individual events to plot (ignored if --best_worst or --high_amps is used)")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum entries to load from ROOT file (for handling large files)")
    parser.add_argument("--best_worst", action="store_true",
                       help="Plot the 5 best and 5 worst fits based on chi-squared values instead of first N events")
    parser.add_argument("--high_amps", type=int, metavar="N", default=None,
                       help="Plot the N events with highest amps to examine potential outliers (default: 10)")
    
    args = parser.parse_args()
    
    # Check if ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file {args.root_file} not found!")
        return 1
    
    # Load data
    data = load_success_fits(args.root_file, max_entries=args.max_entries)
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
    elif args.high_amps is not None:
        # Create plots for high amp events
        n_high_amp = args.high_amps if args.high_amps > 0 else 10
        print(f"\nUsing high amp event selection (top {n_high_amp} events)...")
        success_count = create_high_amp_plots(data, args.output, n_high_amp)
        print(f"\nTotal plots created: {success_count}")
    else:
        # Create individual event plots (original behavior)
        n_events = min(args.num_events, len(data['TrueX']))
        print(f"\nCreating plots for first {n_events} events...")
        
        lorentz_success = 0
        gauss_success = 0
        power_lorentz_success = 0
        combined_success = 0
        
        for i in range(n_events):
            # Create Lorentz collage plot
            lorentz_result = create_all_lorentz_plot(i, data, args.output)
            if "Error" not in lorentz_result:
                lorentz_success += 1
            if i % 5 == 0 or "Error" in lorentz_result:
                print(f"  {lorentz_result}")
            
            # Create Gauss collage plot
            gauss_result = create_all_gauss_plot(i, data, args.output)
            if "Error" not in gauss_result:
                gauss_success += 1
            if i % 5 == 0 or "Error" in gauss_result:
                print(f"  {gauss_result}")
            
            # Create Power Lorentz collage plot
            power_result = create_all_power_lorentz_plot(i, data, args.output)
            if "Error" not in power_result:
                power_lorentz_success += 1
            if i % 5 == 0 or "Error" in power_result:
                print(f"  {power_result}")
            
            # Create combined all models plot
            combined_result = create_all_models_combined_plot(i, data, args.output)
            if "Error" not in combined_result:
                combined_success += 1
            if i % 5 == 0 or "Error" in combined_result:
                print(f"  {combined_result}")
        
        print(f"\nResults:")
        print(f"  Successly created {lorentz_success}/{n_events} Lorentz collage plots")
        print(f"  Successly created {gauss_success}/{n_events} Gauss collage plots")
        print(f"  Successly created {power_lorentz_success}/{n_events} Power Lorentz collage plots")
        print(f"  Successly created {combined_success}/{n_events} combined all models plots")
        
        print(f"\nAll plots saved to: {args.output}/")
        print(f"  - Lorentz collages: {args.output}/lorentz/")
        print(f"  - Gauss collages: {args.output}/gauss/")
        print(f"  - Power Lorentz collages: {args.output}/power_lorentz/")
        print(f"  - Combined all models plots: {args.output}/all_models_combined/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 