#!/usr/bin/env python3
"""
Visualize hit positions (PosX, PosY) and their pixel approximations (PixelX, PixelY)
from a ROOT file using uproot and matplotlib.

Optimized for performance with multiprocessing and efficient plotting techniques.
"""

import uproot
import numpy as np
import matplotlib
# Use Agg backend for better performance when not in interactive mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
from pathlib import Path
import multiprocessing as mp
import time
from functools import partial
import argparse

# Set global matplotlib options for better performance
plt.rcParams['figure.dpi'] = 100  # Lower default DPI for faster rendering during development
plt.rcParams['path.simplify'] = True  # Simplify paths for better performance
plt.rcParams['path.simplify_threshold'] = 0.9  # Higher threshold = more simplification
plt.rcParams['agg.path.chunksize'] = 10000  # Larger chunks for better performance

def create_hit_positions_plot(data_dict, output_prefix=""):
    """Create hit positions plot in a separate process."""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(pos_x, pos_y, gridsize=50, cmap='viridis')
    ax.set_title("Hit Positions (PosX, PosY)", fontsize=14)
    ax.set_xlabel("PosX [mm]", fontsize=12)
    ax.set_ylabel("PosY [mm]", fontsize=12)
    ax.grid(alpha=0.3)
    fig.colorbar(hb, ax=ax, label="Count")
    plt.tight_layout()
    
    # Save with prefix if provided
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}hit_positions.png", dpi=300)
    fig.savefig(f"{prefix}hit_positions.svg", format='svg')
    fig.savefig(f"{prefix}hit_positions.pdf", format='pdf')
    plt.close(fig)
    return "Hit positions plot completed"

def create_pixel_positions_plot(data_dict, output_prefix=""):
    """Create pixel positions plot in a separate process."""
    pixel_x, pixel_y = data_dict['pixel_x'], data_dict['pixel_y']
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(pixel_x, pixel_y, gridsize=50, cmap='viridis')
    ax.set_title("Pixel Positions (PixelX, PixelY)", fontsize=14)
    ax.set_xlabel("PixelX [mm]", fontsize=12)
    ax.set_ylabel("PixelY [mm]", fontsize=12)
    ax.grid(alpha=0.3)
    fig.colorbar(hb, ax=ax, label="Count")
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}pixel_positions.png", dpi=300)
    fig.savefig(f"{prefix}pixel_positions.svg", format='svg')
    fig.savefig(f"{prefix}pixel_positions.pdf", format='pdf')
    plt.close(fig)
    return "Pixel positions plot completed"

def create_hit_pixel_mapping_plot(data_dict, output_prefix=""):
    """Create hit-to-pixel mapping plot in a separate process."""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    pixel_x, pixel_y = data_dict['pixel_x'], data_dict['pixel_y']
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(pos_x, pos_y, s=1, alpha=0.5, color='blue', label='Hits')
    ax.scatter(pixel_x, pixel_y, s=10, alpha=0.8, color='red', marker='s', label='Pixels')
    
    # Draw all connecting lines between hits and pixels
    # Use a line collection for much faster rendering of many lines
    from matplotlib.collections import LineCollection
    
    # Create a list of line segments
    n_points = len(pos_x)
    lines = [[(pos_x[i], pos_y[i]), (pixel_x[i], pixel_y[i])] for i in range(n_points)]
    
    # Create a line collection
    lc = LineCollection(lines, colors='black', alpha=0.5, linewidths=0.5)
    ax.add_collection(lc)
    
    ax.set_title("Hit to Pixel Mapping (All Hits)", fontsize=14)
    ax.set_xlabel("X [mm]", fontsize=12)
    ax.set_ylabel("Y [mm]", fontsize=12)
    ax.legend(loc='upper right')  # Explicitly set legend position
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}hit_pixel_mapping.png", dpi=300)
    fig.savefig(f"{prefix}hit_pixel_mapping.svg", format='svg')
    fig.savefig(f"{prefix}hit_pixel_mapping.pdf", format='pdf')
    plt.close(fig)
    return "Hit-to-pixel mapping plot completed"

def create_distance_histogram_plot(data_dict, output_prefix=""):
    """Create distance histogram plot in a separate process."""
    pixel_dist = data_dict['pixel_dist']
    
    mean_dist = np.mean(pixel_dist)
    median_dist = np.median(pixel_dist)
    max_dist = np.max(pixel_dist)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    hist = ax.hist(pixel_dist, bins=50, alpha=0.7, color='green', 
                 edgecolor='black', linewidth=0.5)
    ax.set_title("Distance from Hit to Nearest Pixel", fontsize=14)
    ax.set_xlabel("Distance [mm]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(alpha=0.3)
    
    stats_text = (f"Mean: {mean_dist:.4f} mm\n"
                f"Median: {median_dist:.4f} mm\n"
                f"Max: {max_dist:.4f} mm")
    
    ax.text(0.97, 0.97, stats_text, 
           transform=ax.transAxes, 
           verticalalignment='top', 
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}pixel_distance_histogram.png", dpi=300)
    fig.savefig(f"{prefix}pixel_distance_histogram.svg", format='svg')
    fig.savefig(f"{prefix}pixel_distance_histogram.pdf", format='pdf')
    plt.close(fig)
    return "Distance histogram plot completed"

def create_pixel_hit_status_plot(data_dict, output_prefix=""):
    """Create pixel hit status plot in a separate process."""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    pixel_hit = data_dict['pixel_hit']
    
    hits_on_pixels = pixel_hit == True
    hits_off_pixels = pixel_hit == False
    
    num_on_pixels = np.sum(hits_on_pixels)
    num_off_pixels = np.sum(hits_off_pixels)
    total_hits = len(pixel_hit)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    
    ax.scatter(pos_x[hits_on_pixels], pos_y[hits_on_pixels], s=5, alpha=0.7, 
             color='green', label=f'Hits on pixels ({num_on_pixels}, {num_on_pixels/total_hits*100:.1f}%)')
    ax.scatter(pos_x[hits_off_pixels], pos_y[hits_off_pixels], s=5, alpha=0.7, 
             color='red', label=f'Hits not on pixels ({num_off_pixels}, {num_off_pixels/total_hits*100:.1f}%)')
    
    ax.set_title("Hit Position by Pixel Hit Status", fontsize=14)
    ax.set_xlabel("X [mm]", fontsize=12)
    ax.set_ylabel("Y [mm]", fontsize=12)
    ax.legend(loc='upper right')  # Explicitly set legend position
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}hit_pixel_status.png", dpi=300)
    fig.savefig(f"{prefix}hit_pixel_status.svg", format='svg')
    fig.savefig(f"{prefix}hit_pixel_status.pdf", format='pdf')
    plt.close(fig)
    return "Pixel hit status plot completed"

def create_pixel_indices_heatmap(data_dict, output_prefix=""):
    """Create pixel indices heatmap in a separate process."""
    pixel_i, pixel_j = data_dict['pixel_i'], data_dict['pixel_j']
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    hist_2d = ax.hist2d(pixel_i, pixel_j, 
                      bins=[int(np.max(pixel_i))+1, int(np.max(pixel_j))+1], 
                      cmap='viridis')
    
    ax.set_title("Hit Distribution by Pixel Indices", fontsize=14)
    ax.set_xlabel("Pixel Index I", fontsize=12)
    ax.set_ylabel("Pixel Index J", fontsize=12)
    cbar = fig.colorbar(hist_2d[3], ax=ax, label="Hit Count")
    
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    plt.savefig(f"{prefix}pixel_indices_heatmap.png", dpi=300)
    plt.savefig(f"{prefix}pixel_indices_heatmap.svg", format='svg')
    plt.savefig(f"{prefix}pixel_indices_heatmap.pdf", format='pdf')
    plt.close(fig)
    return "Pixel indices heatmap completed"

def create_combined_plots(data_dict, output_prefix=""):
    """Create the combined plots in a separate process."""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    pixel_x, pixel_y = data_dict['pixel_x'], data_dict['pixel_y']
    pixel_dist = data_dict['pixel_dist']
    
    mean_dist = np.mean(pixel_dist)
    median_dist = np.median(pixel_dist)
    max_dist = np.max(pixel_dist)
    
    fig = plt.figure(figsize=(18, 16), dpi=150)
    
    # 1. Scatter plot showing original hit positions
    ax1 = fig.add_subplot(221)
    hb1 = ax1.hexbin(pos_x, pos_y, gridsize=50, cmap='viridis')
    ax1.set_title("Hit Positions (PosX, PosY)", fontsize=14)
    ax1.set_xlabel("PosX [mm]", fontsize=12)
    ax1.set_ylabel("PosY [mm]", fontsize=12)
    ax1.grid(alpha=0.3)
    fig.colorbar(hb1, ax=ax1, label="Count")
    
    # 2. Scatter plot showing pixel positions
    ax2 = fig.add_subplot(222)
    hb2 = ax2.hexbin(pixel_x, pixel_y, gridsize=50, cmap='viridis')
    ax2.set_title("Pixel Positions (PixelX, PixelY)", fontsize=14)
    ax2.set_xlabel("PixelX [mm]", fontsize=12)
    ax2.set_ylabel("PixelY [mm]", fontsize=12)
    ax2.grid(alpha=0.3)
    fig.colorbar(hb2, ax=ax2, label="Count")
    
    # 3. Scatter plot showing both hits and pixels
    ax3 = fig.add_subplot(223)
    ax3.scatter(pos_x, pos_y, s=1, alpha=0.5, color='blue', label='Hits')
    ax3.scatter(pixel_x, pixel_y, s=10, alpha=0.8, color='red', marker='s', label='Pixels')
    
    # Draw all connecting lines between hits and pixels using LineCollection
    # for much better performance
    from matplotlib.collections import LineCollection
    
    # Create a list of line segments
    n_points = len(pos_x)
    lines = [[(pos_x[i], pos_y[i]), (pixel_x[i], pixel_y[i])] for i in range(n_points)]
    
    # Create a line collection
    lc = LineCollection(lines, colors='black', alpha=0.5, linewidths=0.5)
    ax3.add_collection(lc)
    
    ax3.set_title("Hit to Pixel Mapping (All Hits)", fontsize=14)
    ax3.set_xlabel("X [mm]", fontsize=12)
    ax3.set_ylabel("Y [mm]", fontsize=12)
    ax3.legend(loc='upper right')  # Explicitly set legend position
    ax3.grid(alpha=0.3)
    
    # 4. Histogram of distances from hits to their nearest pixel
    ax4 = fig.add_subplot(224)
    hist = ax4.hist(pixel_dist, bins=50, alpha=0.7, color='green', 
                   edgecolor='black', linewidth=0.5)
    ax4.set_title("Distance from Hit to Nearest Pixel", fontsize=14)
    ax4.set_xlabel("Distance [mm]", fontsize=12)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.grid(alpha=0.3)
    
    # Add statistics to the distance histogram
    stats_text = (f"Mean: {mean_dist:.4f} mm\n"
                 f"Median: {median_dist:.4f} mm\n"
                 f"Max: {max_dist:.4f} mm")
    
    ax4.text(0.97, 0.97, stats_text, 
            transform=ax4.transAxes, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    plt.savefig(f"{prefix}hit_pixel_visualization.png", dpi=300)
    plt.savefig(f"{prefix}hit_pixel_visualization.svg", format='svg')
    plt.savefig(f"{prefix}hit_pixel_visualization.pdf", format='pdf')
    plt.close(fig)
    return "Combined plots completed"

def main():
    parser = argparse.ArgumentParser(description='Visualize hit positions and pixel approximations')
    parser.add_argument('-f', '--file', type=str, default="/home/tom/Desktop/epicToy/build/epicToyOutput.root",
                        help='Path to ROOT file')
    parser.add_argument('-o', '--output-dir', type=str, default="",
                        help='Output directory for plots')
    parser.add_argument('-j', '--jobs', type=int, default=0,
                        help='Number of parallel jobs (0=auto)')
    args = parser.parse_args()
    
    start_time = time.time()
    root_file = args.file
    
    # Check if file exists
    if not os.path.exists(root_file):
        print(f"Error: File {root_file} does not exist.")
        return 1
    
    print(f"Opening ROOT file: {root_file}")
    
    try:
        # Open the ROOT file
        with uproot.open(root_file) as file:
            # Access the tree named "Hits"
            if "Hits" not in file:
                print(f"Error: Tree 'Hits' not found in file. Available objects: {list(file.keys())}")
                return 1
            
            tree = file["Hits"]
            print(f"Tree successfully opened with {tree.num_entries} entries")
            
            # List available branches
            print("Available branches:", tree.keys())
            
            # Read the position and pixel data
            data = tree.arrays(["PosX", "PosY", "PixelX", "PixelY", "PixelDist", "PixelI", "PixelJ", "PixelHit"])
            
            # Convert awkward arrays to numpy arrays for plotting
            # Use ak.to_numpy() instead of np.array() to avoid deprecation warnings
            import awkward as ak
            pos_x = ak.to_numpy(data["PosX"])
            pos_y = ak.to_numpy(data["PosY"])
            pixel_x = ak.to_numpy(data["PixelX"])
            pixel_y = ak.to_numpy(data["PixelY"])
            pixel_dist = ak.to_numpy(data["PixelDist"])
            pixel_i = ak.to_numpy(data["PixelI"])
            pixel_hit = ak.to_numpy(data["PixelHit"])
            pixel_j = ak.to_numpy(data["PixelJ"])
            
            # Create a data dictionary to share with worker processes
            data_dict = {
                'pos_x': pos_x,
                'pos_y': pos_y,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'pixel_dist': pixel_dist,
                'pixel_i': pixel_i,
                'pixel_j': pixel_j,
                'pixel_hit': pixel_hit
            }
            
            print("Creating plots in parallel...")
            
            # Define output directory for plots
            output_dir = args.output_dir
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Change working directory to output directory
            original_dir = os.getcwd()
            os.chdir(output_dir)
            
            # Determine number of processes to use
            n_jobs = args.jobs if args.jobs > 0 else mp.cpu_count()
            
            # Create a multiprocessing pool with the specified number of workers
            pool = mp.Pool(processes=n_jobs)
            print(f"Using {n_jobs} parallel processes")
            
            # Define the plotting functions to run in parallel
            plot_tasks = [
                # (create_combined_plots, data_dict),  # Removed hit_pixel_visualization
                # (create_hit_positions_plot, data_dict),  # Removed hit_positions
                # (create_pixel_positions_plot, data_dict),  # Removed pixel_positions
                (create_hit_pixel_mapping_plot, data_dict),
                (create_distance_histogram_plot, data_dict),
                (create_pixel_hit_status_plot, data_dict),
                (create_pixel_indices_heatmap, data_dict)
            ]
            
            # Start all the plotting tasks in parallel
            results = []
            for func, data in plot_tasks:
                results.append(pool.apply_async(func, args=(data,)))
            
            # Wait for all tasks to complete
            for result in results:
                print(f"Plot completed: {result.get()}")
            
            # Close the pool and wait for workers to exit
            pool.close()
            pool.join()
            
            # Return to original directory
            os.chdir(original_dir)
            
            # Compute basic statistics for the report
            hits_on_pixels = np.sum(pixel_hit)
            total_hits = len(pixel_hit)
            hit_percentage = (hits_on_pixels / total_hits) * 100 if total_hits > 0 else 0
            
            # Calculate detector area (3x3 cm²)
            detector_area = 3.0 * 3.0  # cm²
            detector_area_mm2 = detector_area * 100  # convert to mm²
            
            # Calculate the ratio of hits on pixels to detector area
            hits_to_area_ratio = hits_on_pixels / detector_area_mm2  # hits per mm²
            
            # Print detailed summary statistics
            print("\n=== Hit Statistics Summary ===")
            print(f"Total events: {total_hits}")
            print(f"Hits on pixels: {hits_on_pixels} ({hit_percentage:.2f}%)")
            print(f"Hits not on pixels: {total_hits - hits_on_pixels} ({100 - hit_percentage:.2f}%)")
            print(f"Mean distance to nearest pixel: {np.mean(pixel_dist):.4f} mm")
            print(f"Median distance to nearest pixel: {np.median(pixel_dist):.4f} mm")
            print(f"Max distance to nearest pixel: {np.max(pixel_dist):.4f} mm")
            print(f"Min/Max PixelI: {min(pixel_i)}/{max(pixel_i)}")
            print(f"Min/Max PixelJ: {min(pixel_j)}/{max(pixel_j)}")
            print(f"Detector area: {detector_area} cm² ({detector_area_mm2} mm²)")
            print(f"Ratio of hits on pixels to detector area: {hits_to_area_ratio:.4f} hits/mm²")
            print("============================")
            
            # Print execution time
            end_time = time.time()
            print(f"\nExecution completed in {end_time - start_time:.2f} seconds")
            
            return 0
    
    except Exception as e:
        print(f"Error processing the ROOT file: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
