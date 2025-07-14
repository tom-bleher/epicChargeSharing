#!/usr/bin/env python3
"""
Pixel Hit Validator and Filter for EpicChargeSharingAnalysis

This script reads the ROOT file metadata, validates pixel hit classifications,
and removes GaussRowDeltaX and GaussColDeltaY values for events that are
incorrectly classified or have true positions within pixel areas.

Usage:
    python filter_pixel_hits.py [ROOT_FILE_PATH]
"""

import sys
import os
import numpy as np
import argparse
from pathlib import Path

try:
    import uproot
    import awkward as ak
except ImportError:
    print("Error: uproot and awkward not found. Please install with: pip install uproot awkward")
    sys.exit(1)


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
                    x_data = tree["TrueX"].array(entry_stop=1000, library="np")
                    y_data = tree["TrueY"].array(entry_stop=1000, library="np")
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


def load_simulation_data(root_file_path):
    """
    Load simulation data from ROOT file.
    
    Returns:
        dict: Simulation data arrays
    """
    if not os.path.exists(root_file_path):
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")
    
    print(f"Loading ROOT file: {root_file_path}")
    
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
            'PixelHit',  # Pixel hit flag
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
        
        # Read the data
        data = {}
        for branch in existing_branches:
            try:
                data[branch] = tree[branch].array(library="np")
                print(f"  ✓ Read {branch}: {len(data[branch])} entries")
            except Exception as e:
                print(f"  ✗ Error reading {branch}: {e}")
                # Create dummy data for missing branches
                data[branch] = np.zeros(n_entries, dtype=np.float64)
        
        # Handle missing branches
        for branch in missing_branches:
            if branch == 'PixelHit':
                data[branch] = np.zeros(n_entries, dtype=bool)
            else:
                data[branch] = np.zeros(n_entries, dtype=np.float64)
        
        print(f"Successfully loaded {len(existing_branches)} branches")
        return data
    finally:
        try:
            root_file.close()
        except:
            pass


def validate_and_filter_events(data, metadata, verbose=False):
    """
    Validate pixel hit classifications and filter Gauss delta values.
    
    Args:
        data: Dictionary containing simulation data
        metadata: Dictionary containing detector metadata
        verbose: Whether to print detailed validation information
    
    Returns:
        dict: Updated data with filtered Gauss delta values
    """
    # Get pixel positions
    pixel_x_centers, pixel_y_centers = calculate_pixel_positions(metadata)
    pixel_size = metadata['pixel_size']
    
    # Get arrays
    true_x = data['TrueX']
    true_y = data['TrueY']
    pixel_hit_flags = data.get('PixelHit', np.zeros(len(true_x), dtype=bool))
    gauss_row_delta_x = data.get('GaussRowDeltaX', np.zeros(len(true_x)))
    gauss_col_delta_y = data.get('GaussColDeltaY', np.zeros(len(true_x)))
    
    # Check which positions are actually on pixels
    actual_on_pixel = is_on_pixel_vectorized(true_x, true_y, pixel_x_centers, pixel_y_centers, pixel_size)
    
    # Statistics
    n_total = len(true_x)
    n_marked_pixel_hits = np.sum(pixel_hit_flags)
    n_actual_pixel_hits = np.sum(actual_on_pixel)
    n_correctly_classified = np.sum(pixel_hit_flags == actual_on_pixel)
    n_false_positives = np.sum(pixel_hit_flags & ~actual_on_pixel)  # Marked as pixel but not on pixel
    n_false_negatives = np.sum(~pixel_hit_flags & actual_on_pixel)  # Not marked as pixel but on pixel
    
    if verbose:
        print(f"\n=== PIXEL HIT VALIDATION STATISTICS ===")
        print(f"Total events: {n_total}")
        print(f"Marked as pixel hits: {n_marked_pixel_hits} ({100*n_marked_pixel_hits/n_total:.1f}%)")
        print(f"Actually on pixels: {n_actual_pixel_hits} ({100*n_actual_pixel_hits/n_total:.1f}%)")
        print(f"Correctly classified: {n_correctly_classified} ({100*n_correctly_classified/n_total:.1f}%)")
        print(f"False positives (marked pixel but not on pixel): {n_false_positives}")
        print(f"False negatives (not marked pixel but on pixel): {n_false_negatives}")
        print(f"========================================")
    
    # Find events that should NOT have Gauss delta values
    # These are events where:
    # 1. The event is actually on a pixel (regardless of what the flag says)
    # 2. OR the event is marked as a pixel hit in the ROOT file
    events_to_filter = actual_on_pixel | pixel_hit_flags
    
    # Count events with non-zero/non-NaN Gauss delta values that should be filtered
    has_gauss_x = np.isfinite(gauss_row_delta_x) & (gauss_row_delta_x != 0)
    has_gauss_y = np.isfinite(gauss_col_delta_y) & (gauss_col_delta_y != 0)
    
    n_gauss_x_to_filter = np.sum(events_to_filter & has_gauss_x)
    n_gauss_y_to_filter = np.sum(events_to_filter & has_gauss_y)
    
    if verbose:
        print(f"\n=== GAUSS DELTA FILTERING ===")
        print(f"Events to filter (pixel hits): {np.sum(events_to_filter)}")
        print(f"GaussRowDeltaX values to remove: {n_gauss_x_to_filter}")
        print(f"GaussColDeltaY values to remove: {n_gauss_y_to_filter}")
        print(f"=============================")
    
    # Create filtered data
    filtered_data = data.copy()
    
    # Set Gauss delta values to NaN for events that should be filtered
    filtered_gauss_row_delta_x = gauss_row_delta_x.copy()
    filtered_gauss_col_delta_y = gauss_col_delta_y.copy()
    
    filtered_gauss_row_delta_x[events_to_filter] = np.nan
    filtered_gauss_col_delta_y[events_to_filter] = np.nan
    
    # Update the data
    filtered_data['GaussRowDeltaX'] = filtered_gauss_row_delta_x
    filtered_data['GaussColDeltaY'] = filtered_gauss_col_delta_y
    
    # Add validation results
    filtered_data['ActualPixelHit'] = actual_on_pixel
    filtered_data['ValidationFlag'] = actual_on_pixel.astype(int)
    
    return filtered_data


def save_filtered_data(filtered_data, output_path, verbose=False):
    """
    Save filtered data to a new ROOT file.
    
    Args:
        filtered_data: Dictionary containing filtered simulation data
        output_path: Path for output ROOT file
        verbose: Whether to print detailed information
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert data to uproot format
        uproot_data = {}
        for key, value in filtered_data.items():
            if isinstance(value, np.ndarray):
                uproot_data[key] = value
            else:
                uproot_data[key] = np.array(value)
        
        # Write to ROOT file
        with uproot.recreate(output_path) as output_file:
            output_file["Hits"] = uproot_data
        
        if verbose:
            print(f"Filtered data saved to: {output_path}")
            print(f"Saved {len(uproot_data)} branches with {len(list(uproot_data.values())[0])} entries each")
        
    except Exception as e:
        print(f"Error saving filtered data: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Validate pixel hit classifications and filter Gauss delta values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_pixel_hits.py epicChargeSharingOutputIPS.root
  python filter_pixel_hits.py --output filtered_output.root epicChargeSharingOutputIPS.root
  python filter_pixel_hits.py --verbose epicChargeSharingOutputIPS.root
        """
    )
    
    parser.add_argument('root_file', 
                       help='Path to input ROOT file')
    
    parser.add_argument('--output', '-o', 
                       help='Path for output ROOT file (default: input_file_filtered.root)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.root_file):
        print(f"Error: Input file not found: {args.root_file}")
        return 1
    
    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.root_file)
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    try:
        # Read metadata
        print("Reading detector metadata...")
        metadata = read_root_metadata(args.root_file)
        
        if args.verbose:
            print(f"Detector configuration:")
            print(f"  Pixel size: {metadata['pixel_size']*1000:.0f} μm")
            print(f"  Pixel spacing: {metadata['pixel_spacing']*1000:.0f} μm")
            print(f"  Pixel corner offset: {metadata['pixel_corner_offset']*1000:.0f} μm")
            print(f"  Detector size: {metadata['detector_size']:.1f} mm")
            print(f"  Number of pixels per side: {metadata['num_pixels_per_side']}")
        
        # Load simulation data
        print("Loading simulation data...")
        data = load_simulation_data(args.root_file)
        
        # Validate and filter
        print("Validating pixel hits and filtering Gauss delta values...")
        filtered_data = validate_and_filter_events(data, metadata, verbose=args.verbose)
        
        # Save results
        print("Saving filtered data...")
        save_filtered_data(filtered_data, output_path, verbose=args.verbose)
        
        print(f"\n✓ Processing complete!")
        print(f"Input file: {args.root_file}")
        print(f"Output file: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 