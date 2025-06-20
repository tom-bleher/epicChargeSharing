#!/usr/bin/env python3
"""
Spatial Resolution Calculator for PixelChargeSharingToy

This script reads simulated data from a ROOT file and calculates the spatial resolution
for various reconstruction methods by computing position residuals.

Spatial resolution is quantified as the standard deviation of position residuals:
- Residual: r_i = x_rec(i) - x_true(i)
- Mean residual (bias): r_bar = (1/N) * sum(r_i)
- Standard deviation: σ_res = sqrt((1/(N-1)) * sum((r_i - r_bar)^2))
- RMS (if unbiased): RMS = sqrt((1/N) * sum(r_i^2))

Usage:
    python calc_res.py [ROOT_FILE_PATH]
    
If no path is provided, defaults to ../build/epicToyOutput.root
"""

import sys
import os
import numpy as np
import argparse
from datetime import datetime

try:
    import uproot
except ImportError:
    print("Error: uproot not found. Please install with: pip install uproot awkward")
    sys.exit(1)


def calculate_resolution_stats(residuals, name):
    """
    Calculate resolution statistics from residuals array.
    
    Args:
        residuals: numpy array of residuals (reconstructed - true)
        name: string identifier for this measurement
        
    Returns:
        dict with statistics
    """
    n_events = len(residuals)
    if n_events == 0:
        return None
        
    # Calculate mean residual (bias)
    mean_residual = np.mean(residuals)
    
    # Calculate standard deviation (unbiased estimator)
    std_residual = np.std(residuals, ddof=1)  # ddof=1 for unbiased estimator
    
    # Calculate RMS (assuming zero mean)
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    # Additional statistics
    median_residual = np.median(residuals)
    min_residual = np.min(residuals)
    max_residual = np.max(residuals)
    
    return {
        'name': name,
        'n_events': n_events,
        'mean_bias': mean_residual,
        'std_dev': std_residual,
        'rms': rms_residual,
        'median': median_residual,
        'min': min_residual,
        'max': max_residual
    }


def read_root_data(root_file_path):
    """
    Read data from ROOT file and extract relevant branches.
    
    Args:
        root_file_path: path to ROOT file
        
    Returns:
        dict with numpy arrays for each branch
    """
    if not os.path.exists(root_file_path):
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")
    
    print(f"Reading ROOT file: {root_file_path}")
    
    # Open ROOT file with uproot
    try:
        with uproot.open(root_file_path) as root_file:
            # Get the tree
            if "Hits" not in root_file:
                raise RuntimeError("Cannot find 'Hits' tree in ROOT file")
            
            tree = root_file["Hits"]
            n_entries = tree.num_entries
            print(f"Found {n_entries} entries in tree")
            
            # List ALL available branches to debug what's actually there
            available_branches = list(tree.keys())
            print(f"Total available branches: {len(available_branches)}")
            
            # Look for diagonal branches specifically
            diagonal_branches = [b for b in available_branches if 'Diag' in b and 'Delta' in b]
            print(f"Found diagonal delta branches: {diagonal_branches}")
            
            # Branch names based on actual ROOT file structure
            branches = [
                'TrueX', 'TrueY',  # True positions
                'PixelTrueDeltaX', 'PixelTrueDeltaY',  # Digital readout deltas
                'GaussRowDeltaX', 'GaussColumnDeltaY',  # Gaussian row/column fit deltas
                'LorentzRowDeltaX', 'LorentzColumnDeltaY',  # Lorentzian row/column fit deltas
                
                # Main diagonal deltas
                'GaussMainDiagTransformedDeltaX', 'GaussMainDiagTransformedDeltaY',
                'LorentzMainDiagTransformedDeltaX', 'LorentzMainDiagTransformedDeltaY',
                
                # Secondary diagonal deltas (check if they exist)
                'GaussSecondDiagTransformedDeltaX', 'GaussSecondDiagTransformedDeltaY',
                'LorentzSecondDiagTransformedDeltaX', 'LorentzSecondDiagTransformedDeltaY',
                
                # Skewed Lorentzian deltas (check if they exist)
                'SkewedLorentzMainDiagTransformedDeltaX', 'SkewedLorentzMainDiagTransformedDeltaY',
                'SkewedLorentzSecondDiagTransformedDeltaX', 'SkewedLorentzSecondDiagTransformedDeltaY',
                
                # Mean estimators
                'GaussMeanTrueDeltaX', 'GaussMeanTrueDeltaY',
                'LorentzMeanTrueDeltaX', 'LorentzMeanTrueDeltaY',
                'SkewedLorentzMeanTrueDeltaX', 'SkewedLorentzMeanTrueDeltaY',
            ]
            
            # Read all branches at once
            print("Reading tree data...")
            print(f"Available branches: {sorted(available_branches)}")
            
            data = {}
            missing_branches = []
            for branch in branches:
                if branch in available_branches:
                    data[branch] = tree[branch].array(library="np")
                    print(f"  ✓ Read {branch}: {len(data[branch])} entries")
                else:
                    missing_branches.append(branch)
                    print(f"  ✗ Missing: {branch}")
                    data[branch] = np.zeros(n_entries)
            
            if missing_branches:
                print(f"\nMissing branches ({len(missing_branches)}):")
                for branch in missing_branches:
                    print(f"  - {branch}")
            
            print("ROOT file reading complete")
            return data
            
    except Exception as e:
        raise RuntimeError(f"Error reading ROOT file: {e}")


def calculate_all_resolutions(data):
    """
    Calculate spatial resolution for all reconstruction methods.
    
    The ROOT file already contains delta values (residuals = reconstructed - true),
    so we can directly use these for resolution calculation.
    
    Args:
        data: dict with branch data from ROOT file
        
    Returns:
        list of resolution statistics dictionaries
    """
    results = []
    
    # Define reconstruction methods to analyze
    # The ROOT file contains pre-calculated delta values (residuals)
    methods = [
        ('Digital Readout X', 'PixelTrueDeltaX'),
        ('Digital Readout Y', 'PixelTrueDeltaY'),
        ('Gaussian Row Fit X', 'GaussRowDeltaX'),
        ('Gaussian Column Fit Y', 'GaussColumnDeltaY'),
        ('Lorentzian Row Fit X', 'LorentzRowDeltaX'),
        ('Lorentzian Column Fit Y', 'LorentzColumnDeltaY'),
        
        # Main Diagonal Fits (slope +1: dx - dy = constant)
        ('Gaussian Main Diagonal Fit X', 'GaussMainDiagTransformedDeltaX'),
        ('Gaussian Main Diagonal Fit Y', 'GaussMainDiagTransformedDeltaY'),
        ('Lorentzian Main Diagonal Fit X', 'LorentzMainDiagTransformedDeltaX'),
        ('Lorentzian Main Diagonal Fit Y', 'LorentzMainDiagTransformedDeltaY'),
        
        # Secondary Diagonal Fits (slope -1: dx + dy = constant)
        ('Gaussian Secondary Diagonal Fit X', 'GaussSecondDiagTransformedDeltaX'),
        ('Gaussian Secondary Diagonal Fit Y', 'GaussSecondDiagTransformedDeltaY'),
        ('Lorentzian Secondary Diagonal Fit X', 'LorentzSecondDiagTransformedDeltaX'),
        ('Lorentzian Secondary Diagonal Fit Y', 'LorentzSecondDiagTransformedDeltaY'),
        
        # Skewed Lorentzian Main Diagonal Fits (slope +1: dx - dy = constant)
        ('Skewed Lorentzian Main Diagonal Fit X', 'SkewedLorentzMainDiagTransformedDeltaX'),
        ('Skewed Lorentzian Main Diagonal Fit Y', 'SkewedLorentzMainDiagTransformedDeltaY'),
        
        # Skewed Lorentzian Secondary Diagonal Fits (slope -1: dx + dy = constant)
        ('Skewed Lorentzian Secondary Diagonal Fit X', 'SkewedLorentzSecondDiagTransformedDeltaX'),
        ('Skewed Lorentzian Secondary Diagonal Fit Y', 'SkewedLorentzSecondDiagTransformedDeltaY'),
        
        # Mean Estimators (combined from all methods)
        ('Gaussian Mean Estimator X', 'GaussMeanTrueDeltaX'),
        ('Gaussian Mean Estimator Y', 'GaussMeanTrueDeltaY'),
        ('Lorentzian Mean Estimator X', 'LorentzMeanTrueDeltaX'),
        ('Lorentzian Mean Estimator Y', 'LorentzMeanTrueDeltaY'),
        ('Skewed Lorentzian Mean Estimator X', 'SkewedLorentzMeanTrueDeltaX'),
        ('Skewed Lorentzian Mean Estimator Y', 'SkewedLorentzMeanTrueDeltaY'),
    ]
    
    print("\nCalculating spatial resolution for all methods...")
    
    for method_name, delta_branch in methods:
        if delta_branch in data:
            # The residuals are already calculated in the ROOT file
            residuals = data[delta_branch]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(residuals)
            clean_residuals = residuals[valid_mask]
            
            if len(clean_residuals) > 0:
                stats = calculate_resolution_stats(clean_residuals, method_name)
                if stats:
                    results.append(stats)
                    print(f"  {method_name}: σ = {stats['std_dev']*1000:.1f} μm, "
                          f"bias = {stats['mean_bias']*1000:.3f} μm, "
                          f"RMS = {stats['rms']*1000:.1f} μm")
            else:
                print(f"  {method_name}: No valid data")
        else:
            print(f"  {method_name}: Branch '{delta_branch}' not found")
    
    return results


def save_results_to_file(results, output_file):
    """
    Save resolution results to a text file.
    
    Args:
        results: list of resolution statistics dictionaries
        output_file: path to output text file
    """
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w') as f:
        # Group results by category for better organization
        digital_results = [r for r in results if 'Digital' in r['name']]
        row_col_results = [r for r in results if ('Row Fit' in r['name'] or 'Column Fit' in r['name'])]
        main_diag_results = [r for r in results if 'Main Diagonal' in r['name'] and r['std_dev'] > 0]
        sec_diag_results = [r for r in results if 'Secondary Diagonal' in r['name'] and r['std_dev'] > 0]
        mean_results = [r for r in results if 'Mean Estimator' in r['name'] and r['std_dev'] > 0]
        
        # Write detailed results table with higher precision
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Method':<45} {'N Events':>10} {'Bias (μm)':>14} {'σ (μm)':>12} {'RMS (μm)':>12}\n")
        f.write("=" * 103 + "\n")
        
        # Digital readout section
        if digital_results:
            f.write("DIGITAL READOUT:\n")
            for result in sorted(digital_results, key=lambda x: x['name']):
                f.write(f"{result['name']:<45} "
                       f"{result['n_events']:>10,} "
                       f"{result['mean_bias']*1000:>14.3f} "
                       f"{result['std_dev']*1000:>12.2f} "
                       f"{result['rms']*1000:>12.2f}\n")
            f.write("-" * 103 + "\n")
        
        # Row/Column fits section
        if row_col_results:
            f.write("ROW & COLUMN FITS:\n")
            for result in sorted(row_col_results, key=lambda x: x['name']):
                f.write(f"{result['name']:<45} "
                       f"{result['n_events']:>10,} "
                       f"{result['mean_bias']*1000:>14.3f} "
                       f"{result['std_dev']*1000:>12.2f} "
                       f"{result['rms']*1000:>12.2f}\n")
            f.write("-" * 103 + "\n")
        
        # Main diagonal fits section
        if main_diag_results:
            f.write("MAIN DIAGONAL FITS (slope +1, dx-dy=constant):\n")
            for result in sorted(main_diag_results, key=lambda x: x['std_dev']):
                f.write(f"{result['name']:<45} "
                       f"{result['n_events']:>10,} "
                       f"{result['mean_bias']*1000:>14.3f} "
                       f"{result['std_dev']*1000:>12.2f} "
                       f"{result['rms']*1000:>12.2f}\n")
            f.write("-" * 103 + "\n")
        
        # Secondary diagonal fits section
        if sec_diag_results:
            f.write("SECONDARY DIAGONAL FITS (slope -1, dx+dy=constant):\n")
            for result in sorted(sec_diag_results, key=lambda x: x['std_dev']):
                f.write(f"{result['name']:<45} "
                       f"{result['n_events']:>10,} "
                       f"{result['mean_bias']*1000:>14.3f} "
                       f"{result['std_dev']*1000:>12.2f} "
                       f"{result['rms']*1000:>12.2f}\n")
            f.write("-" * 103 + "\n")
        
        # Mean estimators section
        if mean_results:
            f.write("MEAN ESTIMATORS (combined methods):\n")
            for result in sorted(mean_results, key=lambda x: x['std_dev']):
                f.write(f"{result['name']:<45} "
                       f"{result['n_events']:>10,} "
                       f"{result['mean_bias']*1000:>14.3f} "
                       f"{result['std_dev']*1000:>12.2f} "
                       f"{result['rms']*1000:>12.2f}\n")
            f.write("-" * 103 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate spatial resolution from PixelChargeSharingToy ROOT file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calc_res.py                                    # Use default path
  python calc_res.py /path/to/epicToyOutput.root       # Use specific file
  python calc_res.py --output my_results.txt           # Custom output file
        """
    )
    
    parser.add_argument('root_file', nargs='?', 
                       default='../build/epicToyOutput.root',
                       help='Path to ROOT file (default: ../build/epicToyOutput.root)')
    
    parser.add_argument('-o', '--output', 
                       default='spatial_resolution_results.txt',
                       help='Output text file name (default: spatial_resolution_results.txt)')
    
    args = parser.parse_args()
    
    try:
        # Read ROOT data
        data = read_root_data(args.root_file)
        
        # Calculate resolutions
        results = calculate_all_resolutions(data)
        
        if not results:
            print("Error: No valid resolution results calculated")
            return 1
        
        # Save results
        save_results_to_file(results, args.output)
        
        print(f"\nSpatial resolution analysis complete!")
        print(f"Results saved to: {args.output}")
        print(f"Analyzed {len(results)} reconstruction methods")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
