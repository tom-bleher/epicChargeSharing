#!/usr/bin/env python3
"""
Publication-Quality Post-processing plotting routine for Gaussian, Lorentzian, and Power Lorentzian 
fit visualization of charge sharing in LGAD detectors.

ENHANCED FOR SCIENTIFIC PUBLICATION:
- Professional LaTeX mathematical notation with proper raw strings
- High-resolution output (300 DPI) with publication-quality fonts
- Consistent professional color palette and styling
- Enhanced legends with proper mathematical symbols
- Professional grid and axis styling
- Optimized for scientific journals and presentations

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
        'gaussian': '#1f77b4',      # Professional blue
        'lorentzian': '#ff7f0e',    # Professional orange
        'power_lorentzian': '#9467bd', # Professional purple
        'true_position': '#2ca02c',  # Professional green
        'data_points': '#000000'     # Black for data points
    }
    
    # Enhanced line styles
    line_styles = {
        'gaussian': '-',
        'lorentzian': '--',
        'power_lorentzian': ':',
        'true_position': '--'
    }
    
    # Enhanced line widths
    line_widths = {
        'fit_curves': 2.5,
        'power_lorentzian': 3.0,  # Slightly thicker for dotted line
        'true_position': 2.5,
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
                'Fit2D_XVerticalOffsetErr': 'GaussFitRowVerticalOffsetErr',
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
                'Fit2D_YVerticalOffsetErr': 'GaussFitColumnVerticalOffsetErr',
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
                'Fit2D_Lorentz_XVerticalOffsetErr': 'LorentzFitRowVerticalOffsetErr',
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
                'Fit2D_Lorentz_YVerticalOffsetErr': 'LorentzFitColumnVerticalOffsetErr',
                'Fit2D_Lorentz_YChi2red': 'LorentzFitColumnChi2red',
                'Fit2D_Lorentz_YNPoints': 'LorentzFitColumnDOF',
                
                # Power Lorentzian fit results - Row/X direction
                'Fit2D_PowerLorentz_XCenter': 'PowerLorentzFitRowCenter',
                'Fit2D_PowerLorentz_XGamma': 'PowerLorentzFitRowGamma',  # Gamma is the width parameter
                'Fit2D_PowerLorentz_XAmplitude': 'PowerLorentzFitRowAmplitude',
                'Fit2D_PowerLorentz_XPower': 'PowerLorentzFitRowBeta',  # Beta is the power exponent parameter
                'Fit2D_PowerLorentz_XVerticalOffset': 'PowerLorentzFitRowVerticalOffset',
                'Fit2D_PowerLorentz_XCenterErr': 'PowerLorentzFitRowCenterErr',
                'Fit2D_PowerLorentz_XGammaErr': 'PowerLorentzFitRowGammaErr',
                'Fit2D_PowerLorentz_XAmplitudeErr': 'PowerLorentzFitRowAmplitudeErr',
                'Fit2D_PowerLorentz_XPowerErr': 'PowerLorentzFitRowBetaErr',
                'Fit2D_PowerLorentz_XVerticalOffsetErr': 'PowerLorentzFitRowVerticalOffsetErr',
                'Fit2D_PowerLorentz_XChi2red': 'PowerLorentzFitRowChi2red',
                'Fit2D_PowerLorentz_XNPoints': 'PowerLorentzFitRowDOF',
                
                # Power Lorentzian fit results - Column/Y direction
                'Fit2D_PowerLorentz_YCenter': 'PowerLorentzFitColumnCenter',
                'Fit2D_PowerLorentz_YGamma': 'PowerLorentzFitColumnGamma',  # Gamma is the width parameter
                'Fit2D_PowerLorentz_YAmplitude': 'PowerLorentzFitColumnAmplitude',
                'Fit2D_PowerLorentz_YPower': 'PowerLorentzFitColumnBeta',  # Beta is the power exponent parameter
                'Fit2D_PowerLorentz_YVerticalOffset': 'PowerLorentzFitColumnVerticalOffset',
                'Fit2D_PowerLorentz_YCenterErr': 'PowerLorentzFitColumnCenterErr',
                'Fit2D_PowerLorentz_YGammaErr': 'PowerLorentzFitColumnGammaErr',
                'Fit2D_PowerLorentz_YAmplitudeErr': 'PowerLorentzFitColumnAmplitudeErr',
                'Fit2D_PowerLorentz_YPowerErr': 'PowerLorentzFitColumnBetaErr',
                'Fit2D_PowerLorentz_YVerticalOffsetErr': 'PowerLorentzFitColumnVerticalOffsetErr',
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
                
                # Diagonal Lorentzian fits - Main diagonal X
                'FitDiag_Lorentz_MainXCenter': 'LorentzFitMainDiagXCenter',
                'FitDiag_Lorentz_MainXGamma': 'LorentzFitMainDiagXGamma',
                'FitDiag_Lorentz_MainXAmplitude': 'LorentzFitMainDiagXAmplitude',
                'FitDiag_Lorentz_MainXVerticalOffset': 'LorentzFitMainDiagXVerticalOffset',
                'FitDiag_Lorentz_MainXCenterErr': 'LorentzFitMainDiagXCenterErr',
                'FitDiag_Lorentz_MainXGammaErr': 'LorentzFitMainDiagXGammaErr',
                'FitDiag_Lorentz_MainXAmplitudeErr': 'LorentzFitMainDiagXAmplitudeErr',
                'FitDiag_Lorentz_MainXVerticalOffsetErr': 'LorentzFitMainDiagXVerticalOffsetErr',
                'FitDiag_Lorentz_MainXChi2red': 'LorentzFitMainDiagXChi2red',
                'FitDiag_Lorentz_MainXNPoints': 'LorentzFitMainDiagXDOF',
                
                # Diagonal Lorentzian fits - Main diagonal Y
                'FitDiag_Lorentz_MainYCenter': 'LorentzFitMainDiagYCenter',
                'FitDiag_Lorentz_MainYGamma': 'LorentzFitMainDiagYGamma',
                'FitDiag_Lorentz_MainYAmplitude': 'LorentzFitMainDiagYAmplitude',
                'FitDiag_Lorentz_MainYVerticalOffset': 'LorentzFitMainDiagYVerticalOffset',
                'FitDiag_Lorentz_MainYCenterErr': 'LorentzFitMainDiagYCenterErr',
                'FitDiag_Lorentz_MainYGammaErr': 'LorentzFitMainDiagYGammaErr',
                'FitDiag_Lorentz_MainYAmplitudeErr': 'LorentzFitMainDiagYAmplitudeErr',
                'FitDiag_Lorentz_MainYVerticalOffsetErr': 'LorentzFitMainDiagYVerticalOffsetErr',
                'FitDiag_Lorentz_MainYChi2red': 'LorentzFitMainDiagYChi2red',
                'FitDiag_Lorentz_MainYNPoints': 'LorentzFitMainDiagYDOF',
                
                # Diagonal Lorentzian fits - Secondary diagonal X
                'FitDiag_Lorentz_SecXCenter': 'LorentzFitSecondDiagXCenter',
                'FitDiag_Lorentz_SecXGamma': 'LorentzFitSecondDiagXGamma',
                'FitDiag_Lorentz_SecXAmplitude': 'LorentzFitSecondDiagXAmplitude',
                'FitDiag_Lorentz_SecXVerticalOffset': 'LorentzFitSecondDiagXVerticalOffset',
                'FitDiag_Lorentz_SecXCenterErr': 'LorentzFitSecondDiagXCenterErr',
                'FitDiag_Lorentz_SecXGammaErr': 'LorentzFitSecondDiagXGammaErr',
                'FitDiag_Lorentz_SecXAmplitudeErr': 'LorentzFitSecondDiagXAmplitudeErr',
                'FitDiag_Lorentz_SecXVerticalOffsetErr': 'LorentzFitSecondDiagXVerticalOffsetErr',
                'FitDiag_Lorentz_SecXChi2red': 'LorentzFitSecondDiagXChi2red',
                'FitDiag_Lorentz_SecXNPoints': 'LorentzFitSecondDiagXDOF',
                
                # Diagonal Lorentzian fits - Secondary diagonal Y
                'FitDiag_Lorentz_SecYCenter': 'LorentzFitSecondDiagYCenter',
                'FitDiag_Lorentz_SecYGamma': 'LorentzFitSecondDiagYGamma',
                'FitDiag_Lorentz_SecYAmplitude': 'LorentzFitSecondDiagYAmplitude',
                'FitDiag_Lorentz_SecYVerticalOffset': 'LorentzFitSecondDiagYVerticalOffset',
                'FitDiag_Lorentz_SecYCenterErr': 'LorentzFitSecondDiagYCenterErr',
                'FitDiag_Lorentz_SecYGammaErr': 'LorentzFitSecondDiagYGammaErr',
                'FitDiag_Lorentz_SecYAmplitudeErr': 'LorentzFitSecondDiagYAmplitudeErr',
                'FitDiag_Lorentz_SecYVerticalOffsetErr': 'LorentzFitSecondDiagYVerticalOffsetErr',
                'FitDiag_Lorentz_SecYChi2red': 'LorentzFitSecondDiagYChi2red',
                'FitDiag_Lorentz_SecYNPoints': 'LorentzFitSecondDiagYDOF',
                
                # Diagonal Power Lorentzian fits - Main diagonal X
                'FitDiag_PowerLorentz_MainXCenter': 'PowerLorentzFitMainDiagXCenter',
                'FitDiag_PowerLorentz_MainXGamma': 'PowerLorentzFitMainDiagXGamma',
                'FitDiag_PowerLorentz_MainXBeta': 'PowerLorentzFitMainDiagXBeta',
                'FitDiag_PowerLorentz_MainXAmplitude': 'PowerLorentzFitMainDiagXAmplitude',
                'FitDiag_PowerLorentz_MainXVerticalOffset': 'PowerLorentzFitMainDiagXVerticalOffset',
                'FitDiag_PowerLorentz_MainXCenterErr': 'PowerLorentzFitMainDiagXCenterErr',
                'FitDiag_PowerLorentz_MainXGammaErr': 'PowerLorentzFitMainDiagXGammaErr',
                'FitDiag_PowerLorentz_MainXBetaErr': 'PowerLorentzFitMainDiagXBetaErr',
                'FitDiag_PowerLorentz_MainXAmplitudeErr': 'PowerLorentzFitMainDiagXAmplitudeErr',
                'FitDiag_PowerLorentz_MainXVerticalOffsetErr': 'PowerLorentzFitMainDiagXVerticalOffsetErr',
                'FitDiag_PowerLorentz_MainXChi2red': 'PowerLorentzFitMainDiagXChi2red',
                'FitDiag_PowerLorentz_MainXNPoints': 'PowerLorentzFitMainDiagXDOF',
                
                # Diagonal Power Lorentzian fits - Main diagonal Y
                'FitDiag_PowerLorentz_MainYCenter': 'PowerLorentzFitMainDiagYCenter',
                'FitDiag_PowerLorentz_MainYGamma': 'PowerLorentzFitMainDiagYGamma',
                'FitDiag_PowerLorentz_MainYBeta': 'PowerLorentzFitMainDiagYBeta',
                'FitDiag_PowerLorentz_MainYAmplitude': 'PowerLorentzFitMainDiagYAmplitude',
                'FitDiag_PowerLorentz_MainYVerticalOffset': 'PowerLorentzFitMainDiagYVerticalOffset',
                'FitDiag_PowerLorentz_MainYCenterErr': 'PowerLorentzFitMainDiagYCenterErr',
                'FitDiag_PowerLorentz_MainYGammaErr': 'PowerLorentzFitMainDiagYGammaErr',
                'FitDiag_PowerLorentz_MainYBetaErr': 'PowerLorentzFitMainDiagYBetaErr',
                'FitDiag_PowerLorentz_MainYAmplitudeErr': 'PowerLorentzFitMainDiagYAmplitudeErr',
                'FitDiag_PowerLorentz_MainYVerticalOffsetErr': 'PowerLorentzFitMainDiagYVerticalOffsetErr',
                'FitDiag_PowerLorentz_MainYChi2red': 'PowerLorentzFitMainDiagYChi2red',
                'FitDiag_PowerLorentz_MainYNPoints': 'PowerLorentzFitMainDiagYDOF',
                
                # Diagonal Power Lorentzian fits - Secondary diagonal X
                'FitDiag_PowerLorentz_SecXCenter': 'PowerLorentzFitSecondDiagXCenter',
                'FitDiag_PowerLorentz_SecXGamma': 'PowerLorentzFitSecondDiagXGamma',
                'FitDiag_PowerLorentz_SecXBeta': 'PowerLorentzFitSecondDiagXBeta',
                'FitDiag_PowerLorentz_SecXAmplitude': 'PowerLorentzFitSecondDiagXAmplitude',
                'FitDiag_PowerLorentz_SecXVerticalOffset': 'PowerLorentzFitSecondDiagXVerticalOffset',
                'FitDiag_PowerLorentz_SecXCenterErr': 'PowerLorentzFitSecondDiagXCenterErr',
                'FitDiag_PowerLorentz_SecXGammaErr': 'PowerLorentzFitSecondDiagXGammaErr',
                'FitDiag_PowerLorentz_SecXBetaErr': 'PowerLorentzFitSecondDiagXBetaErr',
                'FitDiag_PowerLorentz_SecXAmplitudeErr': 'PowerLorentzFitSecondDiagXAmplitudeErr',
                'FitDiag_PowerLorentz_SecXVerticalOffsetErr': 'PowerLorentzFitSecondDiagXVerticalOffsetErr',
                'FitDiag_PowerLorentz_SecXChi2red': 'PowerLorentzFitSecondDiagXChi2red',
                'FitDiag_PowerLorentz_SecXNPoints': 'PowerLorentzFitSecondDiagXDOF',
                
                # Diagonal Power Lorentzian fits - Secondary diagonal Y
                'FitDiag_PowerLorentz_SecYCenter': 'PowerLorentzFitSecondDiagYCenter',
                'FitDiag_PowerLorentz_SecYGamma': 'PowerLorentzFitSecondDiagYGamma',
                'FitDiag_PowerLorentz_SecYBeta': 'PowerLorentzFitSecondDiagYBeta',
                'FitDiag_PowerLorentz_SecYAmplitude': 'PowerLorentzFitSecondDiagYAmplitude',
                'FitDiag_PowerLorentz_SecYVerticalOffset': 'PowerLorentzFitSecondDiagYVerticalOffset',
                'FitDiag_PowerLorentz_SecYCenterErr': 'PowerLorentzFitSecondDiagYCenterErr',
                'FitDiag_PowerLorentz_SecYGammaErr': 'PowerLorentzFitSecondDiagYGammaErr',
                'FitDiag_PowerLorentz_SecYBetaErr': 'PowerLorentzFitSecondDiagYBetaErr',
                'FitDiag_PowerLorentz_SecYAmplitudeErr': 'PowerLorentzFitSecondDiagYAmplitudeErr',
                'FitDiag_PowerLorentz_SecYVerticalOffsetErr': 'PowerLorentzFitSecondDiagYVerticalOffsetErr',
                'FitDiag_PowerLorentz_SecYChi2red': 'PowerLorentzFitSecondDiagYChi2red',
                'FitDiag_PowerLorentz_SecYNPoints': 'PowerLorentzFitSecondDiagYDOF',
                
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
            ax.plot(pos_range, y_fit, '-', color='#ff7f0e', linewidth=2.5, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='red', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Lorentzian ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
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
                                x_lorentz_center, x_lorentz_gamma, x_lorentz_amplitude, x_lorentz_vertical_offset,
                                x_lorentz_chi2red, x_lorentz_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_lor_all.add_subplot(gs_lor_all[0, 1])
        plot_lorentzian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                y_lorentz_center, y_lorentz_gamma, y_lorentz_amplitude, y_lorentz_vertical_offset,
                                y_lorentz_chi2red, y_lorentz_dof, true_y, 'Y Column', 'y', delta_pixel_y)
        
        # Main diagonal X plot
        ax_main_x = fig_lor_all.add_subplot(gs_lor_all[1, 0])
        plot_lorentzian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, main_diag_x_vertical_offset,
                                main_diag_x_chi2red, main_diag_x_dof, true_x, 'X Main Diag', 'x', delta_pixel_x)
        
        # Main diagonal Y plot
        ax_main_y = fig_lor_all.add_subplot(gs_lor_all[1, 1])
        plot_lorentzian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, main_diag_y_vertical_offset,
                                main_diag_y_chi2red, main_diag_y_dof, true_y, 'Y Main Diag', 'y', delta_pixel_y)
        
        # Secondary diagonal X plot
        ax_sec_x = fig_lor_all.add_subplot(gs_lor_all[2, 0])
        plot_lorentzian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, sec_diag_x_vertical_offset,
                                sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'X Sec Diag', 'x', delta_pixel_x)
        
        # Secondary diagonal Y plot
        ax_sec_y = fig_lor_all.add_subplot(gs_lor_all[2, 1])
        plot_lorentzian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, sec_diag_y_vertical_offset,
                                sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Y Sec Diag', 'y', delta_pixel_y)
        
        
        plt.suptitle(f'Event {event_idx}: Lorentzian Fits', fontsize=13)
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
        x_power_vertical_offset = data.get('Fit2D_PowerLorentz_XVerticalOffset', [0])[event_idx] if 'Fit2D_PowerLorentz_XVerticalOffset' in data else 0
        x_power_chi2red = data['Fit2D_PowerLorentz_XChi2red'][event_idx]
        x_power_dof = data.get('Fit2D_PowerLorentz_XNPoints', [0])[event_idx] - 5  # N - K, K=5 parameters (including offset)
        
        y_power_center = data['Fit2D_PowerLorentz_YCenter'][event_idx]
        y_power_gamma = data['Fit2D_PowerLorentz_YGamma'][event_idx]
        y_power_amplitude = data['Fit2D_PowerLorentz_YAmplitude'][event_idx]
        y_power_power = data.get('Fit2D_PowerLorentz_YPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_YPower' in data else 1.0
        y_power_vertical_offset = data.get('Fit2D_PowerLorentz_YVerticalOffset', [0])[event_idx] if 'Fit2D_PowerLorentz_YVerticalOffset' in data else 0
        y_power_chi2red = data['Fit2D_PowerLorentz_YChi2red'][event_idx]
        y_power_dof = data.get('Fit2D_PowerLorentz_YNPoints', [0])[event_idx] - 5  # N - K, K=5 parameters (including offset)
        
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
        
        def plot_power_lorentzian_direction(ax, positions, charges, uncertainties, center, gamma, amplitude, power, vertical_offset, chi2red, dof, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot one direction with all requested features."""
            if len(positions) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
            
            # Plot data with or without error bars (automatically detected)
            plot_data_points(ax, positions, charges, uncertainties, fmt='ko', markersize=6, capsize=3, label='Data', alpha=0.8)
            
            # Plot Power Lorentzian fit - NOW INCLUDING THE VERTICAL OFFSET!
            pos_range = np.linspace(positions.min() - 0.1, positions.max() + 0.1, 200)
            y_fit = power_lorentzian_1d(pos_range, amplitude, center, gamma, power, vertical_offset)
            ax.plot(pos_range, y_fit, '-', color='#9467bd', linewidth=2.5, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='magenta', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Power Lorentzian ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n'
                          f'Power = {power:.2f}\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
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
                                      x_power_center, x_power_gamma, x_power_amplitude, x_power_power, x_power_vertical_offset,
                                      x_power_chi2red, x_power_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_power_all.add_subplot(gs_power_all[0, 1])
        plot_power_lorentzian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                                      y_power_center, y_power_gamma, y_power_amplitude, y_power_power, y_power_vertical_offset,
                                      y_power_chi2red, y_power_dof, true_y, 'Y Column', 'y', delta_pixel_y)
        
        # For diagonals, use Gaussian parameters (Power Lorentzian diagonals may not be implemented)
        main_diag_x_center = data.get('FitDiag_MainXCenter', [x_power_center])[event_idx]
        main_diag_x_sigma = data.get('FitDiag_MainXSigma', [x_power_gamma])[event_idx]
        main_diag_x_amplitude = data.get('FitDiag_MainXAmplitude', [x_power_amplitude])[event_idx]
        main_diag_x_chi2red = data.get('FitDiag_MainXChi2red', [1.0])[event_idx]
        main_diag_x_dof = data.get('FitDiag_MainXNPoints', [4])[event_idx] - 4
        
        # Main diagonal X plot (using Gaussian fit for diagonal, but labeled as approximation)
        ax_main_x = fig_power_all.add_subplot(gs_power_all[1, 0])
        plot_power_lorentzian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                                      main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, 1.0, 0,
                                      main_diag_x_chi2red, main_diag_x_dof, true_x, 'Main Diagonal X (approx)', 'x', delta_pixel_x)
        
        # Similar for other diagonals...
        main_diag_y_center = data.get('FitDiag_MainYCenter', [y_power_center])[event_idx]
        main_diag_y_sigma = data.get('FitDiag_MainYSigma', [y_power_gamma])[event_idx]
        main_diag_y_amplitude = data.get('FitDiag_MainYAmplitude', [y_power_amplitude])[event_idx]
        main_diag_y_chi2red = data.get('FitDiag_MainYChi2red', [1.0])[event_idx]
        main_diag_y_dof = data.get('FitDiag_MainYNPoints', [4])[event_idx] - 4
        
        ax_main_y = fig_power_all.add_subplot(gs_power_all[1, 1])
        plot_power_lorentzian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                                      main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, 1.0, 0,
                                      main_diag_y_chi2red, main_diag_y_dof, true_y, 'Main Diagonal Y (approx)', 'y', delta_pixel_y)
        
        sec_diag_x_center = data.get('FitDiag_SecXCenter', [x_power_center])[event_idx]
        sec_diag_x_sigma = data.get('FitDiag_SecXSigma', [x_power_gamma])[event_idx]
        sec_diag_x_amplitude = data.get('FitDiag_SecXAmplitude', [x_power_amplitude])[event_idx]
        sec_diag_x_chi2red = data.get('FitDiag_SecXChi2red', [1.0])[event_idx]
        sec_diag_x_dof = data.get('FitDiag_SecXNPoints', [4])[event_idx] - 4
        
        ax_sec_x = fig_power_all.add_subplot(gs_power_all[2, 0])
        plot_power_lorentzian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                                      sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, 1.0, 0,
                                      sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'Secondary Diagonal X (approx)', 'x', delta_pixel_x)
        
        sec_diag_y_center = data.get('FitDiag_SecYCenter', [y_power_center])[event_idx]
        sec_diag_y_sigma = data.get('FitDiag_SecYSigma', [y_power_gamma])[event_idx]
        sec_diag_y_amplitude = data.get('FitDiag_SecYAmplitude', [y_power_amplitude])[event_idx]
        sec_diag_y_chi2red = data.get('FitDiag_SecYChi2red', [1.0])[event_idx]
        sec_diag_y_dof = data.get('FitDiag_SecYNPoints', [4])[event_idx] - 4
        
        ax_sec_y = fig_power_all.add_subplot(gs_power_all[2, 1])
        plot_power_lorentzian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                                      sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, 1.0, 0,
                                      sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Secondary Diagonal Y (approx)', 'y', delta_pixel_y)
        
        plt.suptitle(f'Event {event_idx}: Power Lorentzian Fits', fontsize=13)
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
            ax.plot(pos_range, y_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9)
            
            # Add vertical lines
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(center, color='blue', linestyle=':', linewidth=2, alpha=0.8)
            
            # Calculate fit-true difference
            fit_true_diff = center - true_pos
            
            # Create legend with chi2/dof, delta pixel, and fit-true difference
            legend_text = (f'Gaussian ' + r'($\chi^2/\nu$' + f' = {chi2red:.2f})\n' +
                          r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm\n'
                          f'Fit {direction.upper()} = {center:.3f} mm\n' +
                          r'$' + f'{direction}' + r'_{\mathrm{fit}} - ' + f'{direction}' + r'_{\mathrm{true}}$' + f' = {fit_true_diff:.3f} mm')
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
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
                              x_gauss_chi2red, x_gauss_dof, true_x, 'X Row', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_gauss_all.add_subplot(gs_gauss_all[0, 1])
        plot_gaussian_direction(ax_col, col_pos, col_charges, col_uncertainties, 
                              y_gauss_center, y_gauss_sigma, y_gauss_amplitude, y_gauss_vertical_offset,
                              y_gauss_chi2red, y_gauss_dof, true_y, 'Y Column', 'y', delta_pixel_y)
        
        # Main diagonal X plot
        ax_main_x = fig_gauss_all.add_subplot(gs_gauss_all[1, 0])
        plot_gaussian_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, 
                              main_diag_x_center, main_diag_x_sigma, main_diag_x_amplitude, main_diag_x_vertical_offset,
                              main_diag_x_chi2red, main_diag_x_dof, true_x, 'X Main Diag', 'x', delta_pixel_x)
        
        # Main diagonal Y plot
        ax_main_y = fig_gauss_all.add_subplot(gs_gauss_all[1, 1])
        plot_gaussian_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, 
                              main_diag_y_center, main_diag_y_sigma, main_diag_y_amplitude, main_diag_y_vertical_offset,
                              main_diag_y_chi2red, main_diag_y_dof, true_y, 'Y Main Diag', 'y', delta_pixel_y)
        
        # Secondary diagonal X plot
        ax_sec_x = fig_gauss_all.add_subplot(gs_gauss_all[2, 0])
        plot_gaussian_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, 
                              sec_diag_x_center, sec_diag_x_sigma, sec_diag_x_amplitude, sec_diag_x_vertical_offset,
                              sec_diag_x_chi2red, sec_diag_x_dof, true_x, 'X Sec Diag', 'x', delta_pixel_x)
        
        # Secondary diagonal Y plot
        ax_sec_y = fig_gauss_all.add_subplot(gs_gauss_all[2, 1])
        plot_gaussian_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, 
                              sec_diag_y_center, sec_diag_y_sigma, sec_diag_y_amplitude, sec_diag_y_vertical_offset,
                              sec_diag_y_chi2red, sec_diag_y_dof, true_y, 'Y Sec Diag', 'y', delta_pixel_y)
        
        plt.suptitle(f'Event {event_idx}: Gaussian Fits', fontsize=13)
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
            x_power_vertical_offset = data.get('Fit2D_PowerLorentz_XVerticalOffset', [0])[event_idx] if 'Fit2D_PowerLorentz_XVerticalOffset' in data else 0
            x_power_chi2red = data['Fit2D_PowerLorentz_XChi2red'][event_idx]
            
            y_power_center = data['Fit2D_PowerLorentz_YCenter'][event_idx]
            y_power_gamma = data['Fit2D_PowerLorentz_YGamma'][event_idx]
            y_power_amplitude = data['Fit2D_PowerLorentz_YAmplitude'][event_idx]
            y_power_power = data.get('Fit2D_PowerLorentz_YPower', [1.0])[event_idx] if 'Fit2D_PowerLorentz_YPower' in data else 1.0
            y_power_vertical_offset = data.get('Fit2D_PowerLorentz_YVerticalOffset', [0])[event_idx] if 'Fit2D_PowerLorentz_YVerticalOffset' in data else 0
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
        
        def plot_all_models_direction(ax, positions, charges, uncertainties, true_pos, title, direction='x', delta_pixel=0, diagonal_type=None):
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
                x_gauss_vertical_offset = data.get('Fit2D_XVerticalOffset', [0])[event_idx] if 'Fit2D_XVerticalOffset' in data else 0
                gauss_fit = gaussian_1d(pos_range, x_gauss_amplitude, x_gauss_center, x_gauss_sigma, x_gauss_vertical_offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gaussian')[0]
                legend_lines.append(line)
                ax.axvline(x_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = x_gauss_center - true_pos
                legend_text_parts.append(f'Gaussian: ' + r'$\chi^2/\nu$' + f' = {x_gauss_chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
            elif has_gaussian and direction == 'y':
                y_gauss_vertical_offset = data.get('Fit2D_YVerticalOffset', [0])[event_idx] if 'Fit2D_YVerticalOffset' in data else 0
                gauss_fit = gaussian_1d(pos_range, y_gauss_amplitude, y_gauss_center, y_gauss_sigma, y_gauss_vertical_offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gaussian')[0]
                legend_lines.append(line)
                ax.axvline(y_gauss_center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = y_gauss_center - true_pos
                legend_text_parts.append(f'Gaussian: ' + r'$\chi^2/\nu$' + f' = {y_gauss_chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
            
            # Plot Lorentzian fit if available
            if has_lorentzian and direction == 'x':
                x_lorentz_vertical_offset = data.get('Fit2D_Lorentz_XVerticalOffset', [0])[event_idx] if 'Fit2D_Lorentz_XVerticalOffset' in data else 0
                lorentz_fit = lorentzian_1d(pos_range, x_lorentz_amplitude, x_lorentz_center, x_lorentz_gamma, x_lorentz_vertical_offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(x_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = x_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentzian: ' + r'$\chi^2/\nu$' + f' = {x_lorentz_chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
            elif has_lorentzian and direction == 'y':
                y_lorentz_vertical_offset = data.get('Fit2D_Lorentz_YVerticalOffset', [0])[event_idx] if 'Fit2D_Lorentz_YVerticalOffset' in data else 0
                lorentz_fit = lorentzian_1d(pos_range, y_lorentz_amplitude, y_lorentz_center, y_lorentz_gamma, y_lorentz_vertical_offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(y_lorentz_center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = y_lorentz_center - true_pos
                legend_text_parts.append(f'Lorentzian: ' + r'$\chi^2/\nu$' + f' = {y_lorentz_chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
            
            # Plot Power Lorentzian fit if available
            if has_power_lorentzian and direction == 'x':
                power_fit = power_lorentzian_1d(pos_range, x_power_amplitude, x_power_center, x_power_gamma, x_power_power, x_power_vertical_offset)
                line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(x_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = x_power_center - true_pos
                legend_text_parts.append(f'Power Lorentzian: ' + r'$\chi^2/\nu$' + f' = {x_power_chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
            elif has_power_lorentzian and direction == 'y':
                power_fit = power_lorentzian_1d(pos_range, y_power_amplitude, y_power_center, y_power_gamma, y_power_power, y_power_vertical_offset)
                line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(y_power_center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                power_diff = y_power_center - true_pos
                legend_text_parts.append(f'Power Lorentzian: ' + r'$\chi^2/\nu$' + f' = {y_power_chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
            
            # Add true position line
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Position')
            
            # Create legend text
            legend_text = '\n'.join(legend_text_parts)
            legend_text += '\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
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
        plot_all_models_direction(ax_row, row_pos, row_charges, row_uncertainties, true_x, 'X Row', 'x', delta_pixel_x)
        
        # Column plot
        ax_col = fig_combined.add_subplot(gs_combined[0, 1])
        plot_all_models_direction(ax_col, col_pos, col_charges, col_uncertainties, true_y, 'Y Column', 'y', delta_pixel_y)
        
        # Diagonal plots (all available diagonal fits)
        def plot_diagonal_direction(ax, positions, charges, uncertainties, true_pos, title, direction='x', delta_pixel=0):
            """Helper to plot diagonal direction with all available diagonal fits."""
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
            
            # Helper function to get diagonal parameters
            def get_diagonal_params(diag_type, direction, model):
                if model == 'gaussian':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('FitDiag_MainXCenter', [np.nan])[event_idx] if 'FitDiag_MainXCenter' in data else np.nan
                        width = data.get('FitDiag_MainXSigma', [np.nan])[event_idx] if 'FitDiag_MainXSigma' in data else np.nan
                        amplitude = data.get('FitDiag_MainXAmplitude', [np.nan])[event_idx] if 'FitDiag_MainXAmplitude' in data else np.nan
                        offset = data.get('FitDiag_MainXVerticalOffset', [0])[event_idx] if 'FitDiag_MainXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_MainXChi2red', [np.nan])[event_idx] if 'FitDiag_MainXChi2red' in data else np.nan
                        dof = data.get('FitDiag_MainXNPoints', [0])[event_idx] if 'FitDiag_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('FitDiag_MainYCenter', [np.nan])[event_idx] if 'FitDiag_MainYCenter' in data else np.nan
                        width = data.get('FitDiag_MainYSigma', [np.nan])[event_idx] if 'FitDiag_MainYSigma' in data else np.nan
                        amplitude = data.get('FitDiag_MainYAmplitude', [np.nan])[event_idx] if 'FitDiag_MainYAmplitude' in data else np.nan
                        offset = data.get('FitDiag_MainYVerticalOffset', [0])[event_idx] if 'FitDiag_MainYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_MainYChi2red', [np.nan])[event_idx] if 'FitDiag_MainYChi2red' in data else np.nan
                        dof = data.get('FitDiag_MainYNPoints', [0])[event_idx] if 'FitDiag_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('FitDiag_SecXCenter', [np.nan])[event_idx] if 'FitDiag_SecXCenter' in data else np.nan
                        width = data.get('FitDiag_SecXSigma', [np.nan])[event_idx] if 'FitDiag_SecXSigma' in data else np.nan
                        amplitude = data.get('FitDiag_SecXAmplitude', [np.nan])[event_idx] if 'FitDiag_SecXAmplitude' in data else np.nan
                        offset = data.get('FitDiag_SecXVerticalOffset', [0])[event_idx] if 'FitDiag_SecXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_SecXChi2red', [np.nan])[event_idx] if 'FitDiag_SecXChi2red' in data else np.nan
                        dof = data.get('FitDiag_SecXNPoints', [0])[event_idx] if 'FitDiag_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('FitDiag_SecYCenter', [np.nan])[event_idx] if 'FitDiag_SecYCenter' in data else np.nan
                        width = data.get('FitDiag_SecYSigma', [np.nan])[event_idx] if 'FitDiag_SecYSigma' in data else np.nan
                        amplitude = data.get('FitDiag_SecYAmplitude', [np.nan])[event_idx] if 'FitDiag_SecYAmplitude' in data else np.nan
                        offset = data.get('FitDiag_SecYVerticalOffset', [0])[event_idx] if 'FitDiag_SecYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_SecYChi2red', [np.nan])[event_idx] if 'FitDiag_SecYChi2red' in data else np.nan
                        dof = data.get('FitDiag_SecYNPoints', [0])[event_idx] if 'FitDiag_SecYNPoints' in data else 0
                elif model == 'lorentzian':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('FitDiag_Lorentz_MainXCenter', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainXCenter' in data else np.nan
                        width = data.get('FitDiag_Lorentz_MainXGamma', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainXGamma' in data else np.nan
                        amplitude = data.get('FitDiag_Lorentz_MainXAmplitude', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainXAmplitude' in data else np.nan
                        offset = data.get('FitDiag_Lorentz_MainXVerticalOffset', [0])[event_idx] if 'FitDiag_Lorentz_MainXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_Lorentz_MainXChi2red', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainXChi2red' in data else np.nan
                        dof = data.get('FitDiag_Lorentz_MainXNPoints', [0])[event_idx] if 'FitDiag_Lorentz_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('FitDiag_Lorentz_MainYCenter', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainYCenter' in data else np.nan
                        width = data.get('FitDiag_Lorentz_MainYGamma', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainYGamma' in data else np.nan
                        amplitude = data.get('FitDiag_Lorentz_MainYAmplitude', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainYAmplitude' in data else np.nan
                        offset = data.get('FitDiag_Lorentz_MainYVerticalOffset', [0])[event_idx] if 'FitDiag_Lorentz_MainYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_Lorentz_MainYChi2red', [np.nan])[event_idx] if 'FitDiag_Lorentz_MainYChi2red' in data else np.nan
                        dof = data.get('FitDiag_Lorentz_MainYNPoints', [0])[event_idx] if 'FitDiag_Lorentz_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('FitDiag_Lorentz_SecXCenter', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecXCenter' in data else np.nan
                        width = data.get('FitDiag_Lorentz_SecXGamma', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecXGamma' in data else np.nan
                        amplitude = data.get('FitDiag_Lorentz_SecXAmplitude', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecXAmplitude' in data else np.nan
                        offset = data.get('FitDiag_Lorentz_SecXVerticalOffset', [0])[event_idx] if 'FitDiag_Lorentz_SecXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_Lorentz_SecXChi2red', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecXChi2red' in data else np.nan
                        dof = data.get('FitDiag_Lorentz_SecXNPoints', [0])[event_idx] if 'FitDiag_Lorentz_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('FitDiag_Lorentz_SecYCenter', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecYCenter' in data else np.nan
                        width = data.get('FitDiag_Lorentz_SecYGamma', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecYGamma' in data else np.nan
                        amplitude = data.get('FitDiag_Lorentz_SecYAmplitude', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecYAmplitude' in data else np.nan
                        offset = data.get('FitDiag_Lorentz_SecYVerticalOffset', [0])[event_idx] if 'FitDiag_Lorentz_SecYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_Lorentz_SecYChi2red', [np.nan])[event_idx] if 'FitDiag_Lorentz_SecYChi2red' in data else np.nan
                        dof = data.get('FitDiag_Lorentz_SecYNPoints', [0])[event_idx] if 'FitDiag_Lorentz_SecYNPoints' in data else 0
                elif model == 'power_lorentzian':
                    if diag_type == 'main' and direction == 'x':
                        center = data.get('FitDiag_PowerLorentz_MainXCenter', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainXCenter' in data else np.nan
                        width = data.get('FitDiag_PowerLorentz_MainXGamma', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainXGamma' in data else np.nan
                        amplitude = data.get('FitDiag_PowerLorentz_MainXAmplitude', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainXAmplitude' in data else np.nan
                        power = data.get('FitDiag_PowerLorentz_MainXBeta', [1.0])[event_idx] if 'FitDiag_PowerLorentz_MainXBeta' in data else 1.0
                        offset = data.get('FitDiag_PowerLorentz_MainXVerticalOffset', [0])[event_idx] if 'FitDiag_PowerLorentz_MainXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_PowerLorentz_MainXChi2red', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainXChi2red' in data else np.nan
                        dof = data.get('FitDiag_PowerLorentz_MainXNPoints', [0])[event_idx] if 'FitDiag_PowerLorentz_MainXNPoints' in data else 0
                    elif diag_type == 'main' and direction == 'y':
                        center = data.get('FitDiag_PowerLorentz_MainYCenter', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainYCenter' in data else np.nan
                        width = data.get('FitDiag_PowerLorentz_MainYGamma', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainYGamma' in data else np.nan
                        amplitude = data.get('FitDiag_PowerLorentz_MainYAmplitude', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainYAmplitude' in data else np.nan
                        power = data.get('FitDiag_PowerLorentz_MainYBeta', [1.0])[event_idx] if 'FitDiag_PowerLorentz_MainYBeta' in data else 1.0
                        offset = data.get('FitDiag_PowerLorentz_MainYVerticalOffset', [0])[event_idx] if 'FitDiag_PowerLorentz_MainYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_PowerLorentz_MainYChi2red', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_MainYChi2red' in data else np.nan
                        dof = data.get('FitDiag_PowerLorentz_MainYNPoints', [0])[event_idx] if 'FitDiag_PowerLorentz_MainYNPoints' in data else 0
                    elif diag_type == 'sec' and direction == 'x':
                        center = data.get('FitDiag_PowerLorentz_SecXCenter', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecXCenter' in data else np.nan
                        width = data.get('FitDiag_PowerLorentz_SecXGamma', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecXGamma' in data else np.nan
                        amplitude = data.get('FitDiag_PowerLorentz_SecXAmplitude', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecXAmplitude' in data else np.nan
                        power = data.get('FitDiag_PowerLorentz_SecXBeta', [1.0])[event_idx] if 'FitDiag_PowerLorentz_SecXBeta' in data else 1.0
                        offset = data.get('FitDiag_PowerLorentz_SecXVerticalOffset', [0])[event_idx] if 'FitDiag_PowerLorentz_SecXVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_PowerLorentz_SecXChi2red', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecXChi2red' in data else np.nan
                        dof = data.get('FitDiag_PowerLorentz_SecXNPoints', [0])[event_idx] if 'FitDiag_PowerLorentz_SecXNPoints' in data else 0
                    else:  # sec and y
                        center = data.get('FitDiag_PowerLorentz_SecYCenter', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecYCenter' in data else np.nan
                        width = data.get('FitDiag_PowerLorentz_SecYGamma', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecYGamma' in data else np.nan
                        amplitude = data.get('FitDiag_PowerLorentz_SecYAmplitude', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecYAmplitude' in data else np.nan
                        power = data.get('FitDiag_PowerLorentz_SecYBeta', [1.0])[event_idx] if 'FitDiag_PowerLorentz_SecYBeta' in data else 1.0
                        offset = data.get('FitDiag_PowerLorentz_SecYVerticalOffset', [0])[event_idx] if 'FitDiag_PowerLorentz_SecYVerticalOffset' in data else 0
                        chi2red = data.get('FitDiag_PowerLorentz_SecYChi2red', [np.nan])[event_idx] if 'FitDiag_PowerLorentz_SecYChi2red' in data else np.nan
                        dof = data.get('FitDiag_PowerLorentz_SecYNPoints', [0])[event_idx] if 'FitDiag_PowerLorentz_SecYNPoints' in data else 0
                
                if model == 'power_lorentzian':
                    return center, width, amplitude, offset, chi2red, dof, power
                else:
                    return center, width, amplitude, offset, chi2red, dof
            
            # Determine diagonal type from title
            if 'Main' in title:
                diag_type = 'main'
            else:
                diag_type = 'sec'
            
            # Plot Gaussian diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'gaussian')
            center, width, amplitude, offset, chi2red, dof = params
            if not np.isnan(center) and not np.isnan(width) and not np.isnan(amplitude) and dof > 0:
                gauss_fit = gaussian_1d(pos_range, amplitude, center, width, offset)
                line = ax.plot(pos_range, gauss_fit, '-', color='#1f77b4', linewidth=2.5, alpha=0.9, label='Gaussian')[0]
                legend_lines.append(line)
                ax.axvline(center, color='blue', linestyle=':', linewidth=1, alpha=0.8)
                gauss_diff = center - true_pos
                legend_text_parts.append(f'Gaussian: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {gauss_diff:.3f}')
            
            # Plot Lorentzian diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'lorentzian')
            center, width, amplitude, offset, chi2red, dof = params
            if not np.isnan(center) and not np.isnan(width) and not np.isnan(amplitude) and dof > 0:
                lorentz_fit = lorentzian_1d(pos_range, amplitude, center, width, offset)
                line = ax.plot(pos_range, lorentz_fit, '--', color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Lorentzian')[0]
                legend_lines.append(line)
                ax.axvline(center, color='red', linestyle=':', linewidth=1, alpha=0.8)
                lorentz_diff = center - true_pos
                legend_text_parts.append(f'Lorentzian: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {lorentz_diff:.3f}')
            
            # Plot Power Lorentzian diagonal fit if available
            params = get_diagonal_params(diag_type, direction, 'power_lorentzian')
            if len(params) == 7:  # power_lorentzian returns 7 params
                center, width, amplitude, offset, chi2red, dof, power = params
                if not np.isnan(center) and not np.isnan(width) and not np.isnan(amplitude) and dof > 0:
                    power_fit = power_lorentzian_1d(pos_range, amplitude, center, width, power, offset)
                    line = ax.plot(pos_range, power_fit, ':', color='#9467bd', linewidth=3.0, alpha=0.9, label='Power Lorentzian')[0]
                    legend_lines.append(line)
                    ax.axvline(center, color='magenta', linestyle=':', linewidth=1, alpha=0.8)
                    power_diff = center - true_pos
                    legend_text_parts.append(f'Power Lorentzian: ' + r'$\chi^2/\nu$' + f' = {chi2red:.2f}, ' + r'$\Delta$' + f' = {power_diff:.3f}')
            
            # Add true position line
            ax.axvline(true_pos, color='green', linestyle='--', linewidth=2, alpha=0.8, label='True Position')
            
            # Create legend text
            if legend_text_parts:
                legend_text = '\n'.join(legend_text_parts)
                legend_text += '\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            else:
                legend_text = 'No successful diagonal fits\n' + r'$\Delta$' + f' pixel {direction.upper()} = {delta_pixel:.3f} mm'
            
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
            
            # Set appropriate x-axis label based on direction
            if direction == 'x':
                ax.set_xlabel(r'$x_{\mathrm{px}}\ (\mathrm{mm})$')
            else:
                ax.set_xlabel(r'$y_{\mathrm{px}}\ (\mathrm{mm})$')
            ax.set_ylabel(r'$Q_{\mathrm{px}}\ (\mathrm{C})$')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        # Main diagonal X plot - use comprehensive plotting function
        ax_main_x = fig_combined.add_subplot(gs_combined[1, 0])
        plot_all_models_direction(ax_main_x, main_x_pos, main_x_charges, main_x_uncertainties, true_x, 'X Main Diag', 'x', delta_pixel_x, diagonal_type='main')
        
        # Main diagonal Y plot - use comprehensive plotting function
        ax_main_y = fig_combined.add_subplot(gs_combined[1, 1])
        plot_all_models_direction(ax_main_y, main_y_pos, main_y_charges, main_y_uncertainties, true_y, 'Y Main Diag', 'y', delta_pixel_y, diagonal_type='main')
        
        # Secondary diagonal X plot - use comprehensive plotting function
        ax_sec_x = fig_combined.add_subplot(gs_combined[2, 0])
        plot_all_models_direction(ax_sec_x, sec_x_pos, sec_x_charges, sec_x_uncertainties, true_x, 'X Sec Diag', 'x', delta_pixel_x, diagonal_type='sec')
        
        # Secondary diagonal Y plot - use comprehensive plotting function
        ax_sec_y = fig_combined.add_subplot(gs_combined[2, 1])
        plot_all_models_direction(ax_sec_y, sec_y_pos, sec_y_charges, sec_y_uncertainties, true_y, 'Y Sec Diag', 'y', delta_pixel_y, diagonal_type='sec')
        
        models_str = "_".join([m.lower().replace(" ", "_") for m in available_models])
        plt.suptitle(f'Event {event_idx}: Combined Models ({", ".join(available_models)})', fontsize=12)
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
            
            amplitude_metrics.append((i, max_amplitude, avg_chi2, x_gauss_amp, y_gauss_amp, x_lorentz_amp, y_lorentz_amp, x_power_amp, y_power_amp))
            
        except Exception as e:
            print(f"Warning: Could not extract amplitude data for event {i}: {e}")
            amplitude_metrics.append((i, 0, float('inf'), 0, 0, 0, 0, 0, 0))
    
    # Sort by amplitude (highest first)
    amplitude_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Get events with finite amplitudes
    valid_amps = [(idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p) for idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p in amplitude_metrics if amp > 0 and np.isfinite(amp)]
    
    if len(valid_amps) == 0:
        print("Warning: No events with valid amplitudes found!")
        return [], []
    
    print(f"Found {len(valid_amps)} events with valid amplitudes out of {n_total} total events")
    
    # Get highest amplitude events
    high_amp_events = valid_amps[:n_events]
    high_amp_indices = [idx for idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p in high_amp_events]
    
    print(f"Highest amplitude events:")
    for i, (idx, amp, chi2, x_g, y_g, x_l, y_l, x_p, y_p) in enumerate(high_amp_events):
        print(f"  {i+1}. Event {idx}: Max Amp = {amp:.2e} C (χ² = {chi2:.3f})")
        print(f"      Gauss: X={x_g:.2e}, Y={y_g:.2e} | Lorentz: X={x_l:.2e}, Y={y_l:.2e} | Power: X={x_p:.2e}, Y={y_p:.2e}")
    
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
        description="Create Gaussian, Lorentzian, and Power Lorentzian fit plots for charge sharing analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_file", help="Path to ROOT file with 2D Gaussian, Lorentzian, and Power Lorentzian fit data")
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