#!/usr/bin/env python3
"""
Position Resolution Visualization for PixelChargeSharingToy

This script reads simulated data from a ROOT file and creates position resolution plots
showing spatial resolution as a function of position, with gray regions indicating
pixel pad locations.

Usage:
    python resPos.py [ROOT_FILE_PATH]
    
    If no path is provided, defaults to ../../build/epicChargeSharingOutput.root
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

try:
    import uproot
    import awkward as ak
except ImportError:
    print("Error: uproot and awkward not found. Please install with: pip install uproot awkward")
    sys.exit(1)

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    print("Warning: joblib not found. Install with 'pip install joblib' for better performance")
    JOBLIB_AVAILABLE = False


def read_root_metadata(root_file_path):
    """
    Read detector configuration metadata from ROOT file.
    
    Returns:
        dict: Detector configuration parameters
    """
    try:
        root_file = uproot.open(root_file_path)
        
        try:
            # Try to read the actual metadata TNamed objects first
            try:
                metadata = {}
                
                # Try different ways to read the stored grid parameters
                metadata_names = [
                    ("GridPixelSize_mm", "pixel_size"),
                    ("GridPixelSpacing_mm", "pixel_spacing"),
                    ("GridPixelCornerOffset_mm", "pixel_corner_offset"),
                    ("GridDetectorSize_mm", "detector_size"),
                    ("GridNumBlocksPerSide", "num_pixels_per_side")
                ]
                
                for root_name, dict_key in metadata_names:
                    if root_name in root_file:
                        try:
                            # Try different methods to extract the value
                            obj = root_file[root_name]
                            value = None
                            
                            # Try different attribute names for TNamed objects
                            for attr in ['title', 'fTitle', 'GetTitle']:
                                if hasattr(obj, attr):
                                    if callable(getattr(obj, attr)):
                                        value = getattr(obj, attr)()
                                    else:
                                        value = getattr(obj, attr)
                                    break
                            
                            if value is not None:
                                if dict_key == "num_pixels_per_side":
                                    metadata[dict_key] = int(float(value))
                                else:
                                    metadata[dict_key] = float(value)
                        except Exception as e:
                            print(f"Warning: Could not read {root_name}: {e}")
                
                # If we got all the metadata, return it
                if len(metadata) == 5:
                    return metadata
                else:
                    print(f"Warning: Only found {len(metadata)} of 5 expected metadata entries")
                    
            except Exception as e:
                print(f"Warning: Could not read stored metadata: {e}")
            
            # Fallback: estimate from data if metadata reading failed
            if "Hits" in root_file:
                tree = root_file["Hits"]
                
                # Read a small sample to get typical values for pixel spacing calculation
                try:
                    x_data = tree["TrueX"].array(entry_stop=1000, library="np")  # type: ignore
                    y_data = tree["TrueY"].array(entry_stop=1000, library="np")  # type: ignore
                    arrays = {"TrueX": x_data, "TrueY": y_data}
                except:
                    # Fallback if individual array reading fails
                    arrays = {"TrueX": np.array([0.0]), "TrueY": np.array([0.0])}
                
                # Estimate pixel spacing from unique coordinate differences
                x_coords = arrays["TrueX"]
                y_coords = arrays["TrueY"]
                
                if len(x_coords) > 10:
                    x_unique = np.unique(np.round(x_coords, 6))
                    y_unique = np.unique(np.round(y_coords, 6))
                    
                    if len(x_unique) > 1:
                        x_diffs = np.diff(x_unique)
                        pixel_spacing = np.median(x_diffs[x_diffs > 0])
                    else:
                        pixel_spacing = 0.5  # mm default
                else:
                    pixel_spacing = 0.5  # mm default
                
                # Estimate detector size and pixel configuration
                x_range = np.max(x_coords) - np.min(x_coords)
                y_range = np.max(y_coords) - np.min(y_coords)
                det_size = max(x_range, y_range) * 1.2  # Add some margin
                
                # Estimate number of pixels
                num_pixels_per_side = int(np.round(det_size / pixel_spacing))
                
                # Default pixel size (typically smaller than spacing)
                pixel_size = pixel_spacing * 0.2  # 20% of spacing as default
                
                # Default corner offset
                pixel_corner_offset = pixel_spacing * 0.3
                
                return {
                    'pixel_size': pixel_size,
                    'pixel_spacing': pixel_spacing,
                    'pixel_corner_offset': pixel_corner_offset,
                    'detector_size': det_size,
                    'num_pixels_per_side': num_pixels_per_side
                }
            else:
                raise ValueError("No 'Hits' tree found in ROOT file")
        finally:
            root_file.close()
                    
    except Exception as e:
        print(f"Warning: Could not read metadata from ROOT file: {e}")
        # Return default values
        return {
            'pixel_size': 0.1,      # 100 microns
            'pixel_spacing': 0.5,   # 500 microns
            'pixel_corner_offset': 0.1,  # 100 microns
            'detector_size': 30.0,  # 30 mm
            'num_pixels_per_side': 60
        }


def read_simulation_data(root_file_path):
    """
    Read simulation data from ROOT file.
    
    Returns:
        dict: Simulation data arrays
    """
    if not os.path.exists(root_file_path):
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")
    
    print(f"Reading ROOT file: {root_file_path}")
    
    root_file = uproot.open(root_file_path)
    try:
        if "Hits" not in root_file:
            raise RuntimeError("Cannot find 'Hits' tree in ROOT file")
        
        tree = root_file["Hits"]
        n_entries = len(tree)
        print(f"Found {n_entries} entries in tree")
        
        # Define branches to read
        branches = [
            'TrueX', 'TrueY',  # True positions
            'PixelTrueDeltaX', 'PixelTrueDeltaY',  # Pixel center deltas
            'GaussRowDeltaX', 'GaussColDeltaY',  # Gauss fit deltas
        ]
        
        # Check which branches exist
        available_branches = tree.keys()
        existing_branches = [b for b in branches if b in available_branches]
        missing_branches = [b for b in branches if b not in available_branches]
        
        if missing_branches:
            print(f"Warning: Missing branches: {missing_branches}")
        
        if not existing_branches:
            raise RuntimeError("No required branches found in ROOT file")
        
        # Read the data using individual branch reads to avoid type issues
        data = {}
        for branch in existing_branches:
            try:
                data[branch] = tree[branch].array(library="np")
                print(f"  ✓ Read {branch}: {len(data[branch])} entries")
            except Exception as e:
                print(f"  ✗ Error reading {branch}: {e}")
                # Create dummy data for missing branches
                data[branch] = np.zeros(n_entries, dtype=np.float64)
        
        print(f"Successfully read {len(existing_branches)} branches")
        return data
    finally:
        try:
            root_file.close()
        except:
            pass  # Ignore close errors


def calculate_pixel_positions(metadata):
    """
    Calculate pixel center positions based on detector configuration.
    
    Returns:
        tuple: (x_pixel_centers, y_pixel_centers)
    """
    pixel_size = metadata['pixel_size']
    pixel_spacing = metadata['pixel_spacing']
    pixel_corner_offset = metadata['pixel_corner_offset']
    detector_size = metadata['detector_size']
    num_pixels = metadata['num_pixels_per_side']
    
    # Calculate first pixel position
    first_pixel_pos = -detector_size/2 + pixel_corner_offset + pixel_size/2
    
    # Generate pixel center positions
    pixel_positions = []
    for i in range(num_pixels):
        pos = first_pixel_pos + i * pixel_spacing
        pixel_positions.append(pos)
    
    return np.array(pixel_positions), np.array(pixel_positions)


def is_on_pixel_vectorized(x_arr, y_arr, pixel_x_centers, pixel_y_centers, pixel_size):
    """
    Vectorized version to determine if positions are on pixel pads.
    
    Args:
        x_arr, y_arr: Arrays of x,y positions
        pixel_x_centers, pixel_y_centers: Arrays of pixel center positions
        pixel_size: Size of pixel pads
    
    Returns:
        np.ndarray: Boolean array indicating which positions are on pixels
    """
    x_arr = np.asarray(x_arr)
    y_arr = np.asarray(y_arr)
    
    # Reshape for broadcasting
    x_expanded = x_arr[:, np.newaxis, np.newaxis]  # (n_points, 1, 1)
    y_expanded = y_arr[:, np.newaxis, np.newaxis]  # (n_points, 1, 1)
    
    px_expanded = pixel_x_centers[np.newaxis, :, np.newaxis]  # (1, n_px, 1)
    py_expanded = pixel_y_centers[np.newaxis, np.newaxis, :]  # (1, 1, n_py)
    
    # Check if within pixel bounds
    x_in_pixel = np.abs(x_expanded - px_expanded) <= pixel_size/2
    y_in_pixel = np.abs(y_expanded - py_expanded) <= pixel_size/2
    
    # Point is on pixel if it's within bounds of any pixel
    on_any_pixel = np.any(x_in_pixel & y_in_pixel, axis=(1, 2))
    
    return on_any_pixel


def process_single_bin(bin_idx, bins, true_pos, other_pos, gauss_delta, 
                      pixel_x_centers, pixel_y_centers, pixel_size, coordinate):
    """
    Process a single position bin for resolution calculation.
    Only calculate resolution for off-pixel regions using Gauss reconstruction.
    
    Returns:
        tuple: (bin_center, resolution, bin_count)
    """
    # Find events in this bin
    in_bin = (true_pos >= bins[bin_idx]) & (true_pos < bins[bin_idx+1])
    
    if np.sum(in_bin) < 5:  # Need minimum events for meaningful statistics
        bin_center = (bins[bin_idx] + bins[bin_idx+1]) / 2
        return bin_center, np.nan, 0
    
    bin_true_pos = true_pos[in_bin]
    bin_other_pos = other_pos[in_bin]
    bin_gauss_delta = gauss_delta[in_bin]
    
    # Vectorized pixel detection
    if coordinate.lower() == 'x':
        x_coords = bin_true_pos
        y_coords = bin_other_pos
    else:
        x_coords = bin_other_pos
        y_coords = bin_true_pos
    
    on_pixel_mask = is_on_pixel_vectorized(x_coords, y_coords, 
                                          pixel_x_centers, pixel_y_centers, pixel_size)
    
    # Only use OFF-pixel events (ignore on-pixel events as requested)
    off_pixel_mask = ~on_pixel_mask
    
    if not np.any(off_pixel_mask):
        # No off-pixel events in this bin
        bin_center = (bins[bin_idx] + bins[bin_idx+1]) / 2
        return bin_center, np.nan, 0
    
    # Use Gauss reconstruction for off-pixel events
    off_pixel_residuals = bin_gauss_delta[off_pixel_mask]
    valid_residuals = off_pixel_residuals[np.isfinite(off_pixel_residuals)]
    
    bin_center = (bins[bin_idx] + bins[bin_idx+1]) / 2
    
    if len(valid_residuals) >= 3:
        # Calculate standard deviation (same as res.py)
        resolution = np.std(valid_residuals, ddof=1)  # Sample standard deviation
        return bin_center, resolution, len(valid_residuals)
    else:
        return bin_center, np.nan, len(valid_residuals)


def calculate_resolution_vs_position(data, metadata, coordinate='x', n_bins=50):
    """
    Calculate position resolution as a function of position.
    Only for off-pixel regions using Gauss reconstruction.
    
    Returns:
        tuple: (bin_centers, resolutions, bin_counts)
    """
    if coordinate.lower() == 'x':
        true_pos = data['TrueX']
        gauss_delta = data.get('GaussRowDeltaX', np.zeros_like(true_pos))
    else:
        true_pos = data['TrueY']
        gauss_delta = data.get('GaussColDeltaY', np.zeros_like(true_pos))
    
    # Get other coordinate for pixel detection
    if coordinate.lower() == 'x':
        other_pos = data['TrueY']
    else:
        other_pos = data['TrueX']
    
    # Get pixel centers
    pixel_x_centers, pixel_y_centers = calculate_pixel_positions(metadata)
    
    # Create position bins
    pos_min = np.min(true_pos)
    pos_max = np.max(true_pos)
    pos_range = pos_max - pos_min
    margin = pos_range * 0.1
    
    bins = np.linspace(pos_min - margin, pos_max + margin, n_bins + 1)
    
    # Process bins
    resolutions = []
    bin_counts = []
    bin_centers = []
    
    for bin_idx in range(n_bins):
        bin_center, resolution, bin_count = process_single_bin(
            bin_idx, bins, true_pos, other_pos, gauss_delta, 
            pixel_x_centers, pixel_y_centers, metadata['pixel_size'], coordinate
        )
        resolutions.append(resolution)
        bin_counts.append(bin_count)
        bin_centers.append(bin_center)
    
    return np.array(bin_centers), np.array(resolutions), np.array(bin_counts)


def plot_resolution_vs_position(bin_centers, resolutions, bin_counts, metadata, 
                               coordinate='x', output_path=None):
    """
    Create position resolution plot similar to the attached reference plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get pixel information
    if coordinate.lower() == 'x':
        pixel_centers = calculate_pixel_positions(metadata)[0]
        coord_label = 'X'
    else:
        pixel_centers = calculate_pixel_positions(metadata)[1]
        coord_label = 'Y'
    
    pixel_size = metadata['pixel_size']
    pixel_spacing = metadata['pixel_spacing']
    
    # Plot gray regions for pixel pads
    for pixel_center in pixel_centers:
        pixel_left = pixel_center - pixel_size/2
        pixel_right = pixel_center + pixel_size/2
        ax.axvspan(pixel_left, pixel_right, alpha=0.3, color='gray', 
                  label='Pixel Pad' if pixel_center == pixel_centers[0] else "")
    
    # Plot resolution data (only for off-pixel regions)
    valid_mask = np.isfinite(resolutions) & (bin_counts >= 3)
    
    if np.any(valid_mask):
        ax.plot(bin_centers[valid_mask], resolutions[valid_mask] * 1000,  # Convert to microns
               'bo-', linewidth=2, markersize=6, label='Two strip observed', color='green')
        
        # Add error bars based on bin statistics
        resolution_errors = resolutions[valid_mask] / np.sqrt(2 * bin_counts[valid_mask])
        ax.errorbar(bin_centers[valid_mask], resolutions[valid_mask] * 1000,
                   yerr=resolution_errors * 1000, fmt='none', capsize=3, alpha=0.7, color='green')
    
    # Plot horizontal line at pitch/sqrt(12)
    digital_resolution = pixel_spacing / np.sqrt(12) * 1000  # Convert to microns
    ax.axhline(y=digital_resolution, color='red', linestyle='--', linewidth=2,
              label=f'Pitch / √12')
    
    # Add "Two strip expected" dashed line (at similar level to "Two strip observed")
    if np.any(valid_mask):
        expected_level = np.nanmean(resolutions[valid_mask]) * 1000
        ax.axhline(y=expected_level, color='green', linestyle='--', linewidth=2,
                  label='Two strip expected')
    
    # Formatting to match the reference plot
    ax.set_xlabel(f'Track {coordinate.lower()} position [mm]', fontsize=14)
    ax.set_ylabel('Position resolution [μm]', fontsize=14)
    ax.set_title(f'10 GeV proton beam', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set y-axis limits to match reference plot style
    if np.any(valid_mask):
        y_max = max(np.nanmax(resolutions[valid_mask]) * 1000 * 1.2, digital_resolution * 1.1)
    else:
        y_max = digital_resolution * 1.2
    ax.set_ylim(0, min(y_max, 160))  # Cap at 160 to match reference plot
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig, ax


def process_coordinate_data(data, metadata, coordinate, n_bins):
    """
    Process coordinate data for resolution calculation.
    
    Returns:
        tuple: (bin_centers, resolutions, bin_counts, coordinate)
    """
    print(f"Processing {coordinate.upper()} coordinate...")
    bin_centers, resolutions, bin_counts = calculate_resolution_vs_position(
        data, metadata, coordinate=coordinate, n_bins=n_bins)
    return bin_centers, resolutions, bin_counts, coordinate


def create_resolution_plot(bin_centers, resolutions, bin_counts, metadata, coordinate, output_dir):
    """
    Create and save resolution plot for a coordinate.
    
    Returns:
        tuple: (fig, ax, coordinate)
    """
    print(f"Creating {coordinate.upper()} position resolution plot...")
    output_path = output_dir / f'position_resolution_{coordinate.lower()}.png'
    fig, ax = plot_resolution_vs_position(
        bin_centers, resolutions, bin_counts, metadata, 
        coordinate=coordinate, output_path=output_path)
    return fig, ax, coordinate


def main():
    parser = argparse.ArgumentParser(
        description="Create position resolution plots from simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resPos.py                                           # Use default path
  python resPos.py /path/to/epicChargeSharingOutput.root    # Use specific file
  python resPos.py --output-dir ./plots                     # Custom output directory
        """
    )
    
    parser.add_argument('root_file', nargs='?', 
                       default='epicChargeSharingOutput.root',
                       help='Path to ROOT file (default: epicChargeSharingOutput.root)')
    
    parser.add_argument('--output-dir', '-o', 
                       default='./plots',
                       help='Output directory for plots (default: ./plots)')
    
    parser.add_argument('--bins', '-b', type=int, default=50,
                       help='Number of position bins (default: 50)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Read metadata and data
        print("Reading detector metadata...")
        metadata = read_root_metadata(args.root_file)
        print(f"Detector configuration:")
        print(f"  Pixel size: {metadata['pixel_size']*1000:.0f} μm")
        print(f"  Pixel spacing: {metadata['pixel_spacing']*1000:.0f} μm")
        print(f"  Detector size: {metadata['detector_size']:.1f} mm")
        print(f"  Number of pixels per side: {metadata['num_pixels_per_side']}")
        
        print("\nReading simulation data...")
        data = read_simulation_data(args.root_file)
        
        # Process X and Y coordinates simultaneously using ThreadPoolExecutor
        print("\nProcessing X and Y coordinates for resolution...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            x_future = executor.submit(process_coordinate_data, data, metadata, 'x', args.bins)
            y_future = executor.submit(process_coordinate_data, data, metadata, 'y', args.bins)
            
            x_bin_centers, x_resolutions, x_counts, x_coord = x_future.result()
            y_bin_centers, y_resolutions, y_counts, y_coord = y_future.result()
        
        # Create plots for X and Y
        print("\nCreating X position resolution plot...")
        x_fig, x_ax, x_coord = create_resolution_plot(x_bin_centers, x_resolutions, x_counts, metadata, x_coord, output_dir)
        
        print("Creating Y position resolution plot...")
        y_fig, y_ax, y_coord = create_resolution_plot(y_bin_centers, y_resolutions, y_counts, metadata, y_coord, output_dir)
        
        # Show plots
        plt.show()
        
        # Print summary statistics
        valid_x = np.isfinite(x_resolutions) & (x_counts >= 3)
        valid_y = np.isfinite(y_resolutions) & (y_counts >= 3)
        
        print(f"\n=== SUMMARY ===")
        if np.any(valid_x):
            print(f"X Resolution: {np.nanmean(x_resolutions[valid_x])*1000:.1f} ± {np.nanstd(x_resolutions[valid_x])*1000:.1f} μm")
        if np.any(valid_y):
            print(f"Y Resolution: {np.nanmean(y_resolutions[valid_y])*1000:.1f} ± {np.nanstd(y_resolutions[valid_y])*1000:.1f} μm")
        
        digital_res = metadata['pixel_spacing'] / np.sqrt(12) * 1000
        print(f"Digital Resolution (Pitch/√12): {digital_res:.1f} μm")
        
        print(f"\nPlots saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
