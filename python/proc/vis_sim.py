#!/usr/bin/env python3
"""
Visualize hit positions (TrueX, TrueY) and their pixel approximations (PixelX, PixelY)
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
import matplotlib.patches as patches

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
    ax.set_title("Hit Positions (TrueX, TrueY)", fontsize=14)
    ax.set_xlabel("TrueX [mm]", fontsize=12)
    ax.set_ylabel("TrueY [mm]", fontsize=12)
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
    ax1.set_title("Hit Positions (TrueX, TrueY)", fontsize=14)
    ax1.set_xlabel("TrueX [mm]", fontsize=12)
    ax1.set_ylabel("TrueY [mm]", fontsize=12)
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

def create_angle_visualization_plot(data_dict, grid_params, output_prefix=""):
    """Create angle visualization plot showing the complete theoretical pixel grid with angles to actual hit pixels"""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    pixel_x, pixel_y = data_dict['pixel_x'], data_dict['pixel_y']
    pixel_alpha = data_dict['pixel_alpha']
    pixel_hit = data_dict['pixel_hit']
    
    # Filter out hits that are on pixels (where alpha calculation doesn't apply)
    valid_indices = ~(pixel_hit == True)
    valid_pos_x = pos_x[valid_indices]
    valid_pos_y = pos_y[valid_indices]
    valid_pixel_x = pixel_x[valid_indices]
    valid_pixel_y = pixel_y[valid_indices]
    valid_alpha = pixel_alpha[valid_indices]
    
    # Also filter out NaN values (hits inside pixels)
    finite_indices = np.isfinite(valid_alpha)
    valid_pos_x = valid_pos_x[finite_indices]
    valid_pos_y = valid_pos_y[finite_indices]
    valid_pixel_x = valid_pixel_x[finite_indices]
    valid_pixel_y = valid_pixel_y[finite_indices]
    valid_alpha = valid_alpha[finite_indices]
    
    # Use grid parameters from ROOT metadata for complete grid
    pixel_size = grid_params['pixel_size']
    pixel_spacing = grid_params['pixel_spacing']
    pixel_corner_offset = grid_params['pixel_corner_offset']
    det_size = grid_params['detector_size']
    num_blocks_per_side = grid_params['num_blocks_per_side']

    # Calculate first pixel center position using the exact Geant4 formula
    first_pixel_pos = -det_size/2 + pixel_corner_offset + pixel_size/2
    
    # Get all unique pixel positions that actually received hits (for highlighting)
    all_actual_pixels = set(zip(data_dict['pixel_x'], data_dict['pixel_y']))
    actual_pixel_positions = set(all_actual_pixels)
    
    # Create figure with exact same size as alpha_demo.py
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw detector outline (exact same style as alpha_demo.py)
    detector = patches.Rectangle((-det_size/2, -det_size/2), det_size, det_size, 
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.7)
    ax.add_patch(detector)
    
    # Draw complete theoretical pixel grid (all possible pixels)
    print(f"Drawing complete theoretical grid: {num_blocks_per_side}×{num_blocks_per_side} = {num_blocks_per_side**2} pixels")
    print(f"Pixels that received hits: {len(actual_pixel_positions)}")
    
    for i in range(num_blocks_per_side):
        for j in range(num_blocks_per_side):
            # Calculate pixel CENTER position (matching Geant4 calculation)
            pixel_center_x = first_pixel_pos + i * pixel_spacing
            pixel_center_y = first_pixel_pos + j * pixel_spacing
            
            # Calculate pixel corner position for drawing (corner = center - size/2)
            x = pixel_center_x - pixel_size/2
            y = pixel_center_y - pixel_size/2
            
            # Check if this pixel actually received hits
            pixel_received_hits = (pixel_center_x, pixel_center_y) in actual_pixel_positions
            
            # Create pixel as a rectangle with different styling based on whether it received hits
            if pixel_received_hits:
                # Pixels that received hits: normal blue with lightblue fill
                pixel = patches.Rectangle((x, y), pixel_size, pixel_size,
                                        linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.8)
            else:
                # Pixels that didn't receive hits: lighter styling
                pixel = patches.Rectangle((x, y), pixel_size, pixel_size,
                                        linewidth=0.5, edgecolor='lightgray', facecolor='none', alpha=0.4)
            ax.add_patch(pixel)
    
    # Limit the number of points to visualize for performance
    max_points = min(100, len(valid_pos_x))  # Same as alpha_demo.py default
    if len(valid_pos_x) > max_points:
        indices = np.random.choice(len(valid_pos_x), max_points, replace=False)
        vis_pos_x = valid_pos_x[indices]
        vis_pos_y = valid_pos_y[indices]
        vis_pixel_x = valid_pixel_x[indices]
        vis_pixel_y = valid_pixel_y[indices]
        vis_alpha = valid_alpha[indices]
    else:
        vis_pos_x = valid_pos_x
        vis_pos_y = valid_pos_y
        vis_pixel_x = valid_pixel_x
        vis_pixel_y = valid_pixel_y
        vis_alpha = valid_alpha
    
    # Point radius (same as alpha_demo.py)
    point_radius = 0.01  # mm
    
    # Draw all the visualization elements for each point (same style as alpha_demo.py)
    for idx in range(len(vis_pos_x)):
        point_x, point_y = vis_pos_x[idx], vis_pos_y[idx]
        pix_x, pix_y = vis_pixel_x[idx], vis_pixel_y[idx]
        alpha = vis_alpha[idx]
        
        # Draw the point (exact same style as alpha_demo.py)
        point_patch = patches.Circle((point_x, point_y), radius=point_radius, color='black')
        ax.add_patch(point_patch)
        
        # Get pixel data (exact same variable names as alpha_demo.py)
        px = pix_x - pixel_size/2
        py = pix_y - pixel_size/2
        pwidth = pixel_size
        pheight = pixel_size
        
        # Highlight the closest pixel (exact same style as alpha_demo.py)
        pixel_highlight = patches.Rectangle((px, py), pwidth, pheight,
                                           linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(pixel_highlight)
        
        # Calculate pixel center and distance to point (exact same as alpha_demo.py)
        pixel_center_x = px + pwidth/2
        pixel_center_y = py + pheight/2
        distance = np.sqrt((point_x - pixel_center_x)**2 + (point_y - pixel_center_y)**2)
        
        # Set arc radius based on distance to pixel (exact same formula as alpha_demo.py)
        radius = min(max(distance * 0.3, 0.1), 0.5)  # Adaptive radius, min 0.1mm, max 0.5mm
        
        # Get pixel corners used to define the alpha angle (exact same as alpha_demo.py)
        corners = [
            (px, py),                    # bottom-left
            (px + pwidth, py),           # bottom-right
            (px + pwidth, py + pheight), # top-right
            (px, py + pheight)           # top-left
        ]
        
        # Draw lines directly from point to the corners that define the maximum angle
        # (exact same method as alpha_demo.py)
        angles_sorted = sorted([(np.arctan2(c[1] - point_y, c[0] - point_x), i) for i, c in enumerate(corners)])
        
        # Find the corners that define the alpha angle (exact same as alpha_demo.py)
        angle_diffs = []
        for i in range(len(angles_sorted)):
            next_idx = (i + 1) % len(angles_sorted)
            diff = angles_sorted[next_idx][0] - angles_sorted[i][0]
            if diff < 0:
                diff += 2 * np.pi
            angle_diffs.append(diff)
            
        max_diff_idx = angle_diffs.index(max(angle_diffs))
        corner1 = corners[angles_sorted[max_diff_idx][1]]
        corner2 = corners[angles_sorted[(max_diff_idx + 1) % len(angles_sorted)][1]]
        
        # Draw lines to the actual corners (exact same style as alpha_demo.py)
        line1 = ax.plot([point_x, corner1[0]], [point_y, corner1[1]], 'r-', linewidth=1.5)[0]
        line2 = ax.plot([point_x, corner2[0]], [point_y, corner2[1]], 'r-', linewidth=1.5)[0]
        
        # Calculate start and end angles for arc visualization (same as alpha_demo.py)
        start_angle = angles_sorted[max_diff_idx][0]
        end_angle = angles_sorted[(max_diff_idx + 1) % len(angles_sorted)][0]
        
        # Draw the angle arc in the complementary region (exact same as alpha_demo.py)
        # This shows the region that the pixel occupies in the field of view
        arc = patches.Arc((point_x, point_y), radius*2, radius*2, 
                         theta1=np.degrees(end_angle), 
                         theta2=np.degrees(start_angle) + 360 if start_angle > end_angle else np.degrees(start_angle) + 360,
                         linewidth=2, color='red')
        ax.add_patch(arc)
        
        # Draw dotted line to the pixel center (exact same style as alpha_demo.py)
        dist_line = ax.plot([point_x, pixel_center_x], 
                           [point_y, pixel_center_y], 
                           'k--', linewidth=1, alpha=0.6)[0]
    
    # Set plot properties exactly like alpha_demo.py
    margin = 0.5  # Exact same margin as alpha_demo.py
    ax.set_xlim(-det_size/2-margin, det_size/2+margin)
    ax.set_ylim(-det_size/2-margin, det_size/2+margin)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)  # Exact same grid style as alpha_demo.py
    ax.set_xlabel('X (mm)')  # Exact same label as alpha_demo.py
    ax.set_ylabel('Y (mm)')  # Exact same label as alpha_demo.py
    
    # Add a title to explain the visualization
    ax.set_title(f'Complete Detector Grid ({num_blocks_per_side}×{num_blocks_per_side} pixels)\nBlue: pixels with hits, Gray: no hits, Red: highlighted nearest pixels', 
                fontsize=10, pad=20)
    
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}angle_visualization.png", dpi=300)
    fig.savefig(f"{prefix}angle_visualization.svg", format='svg')
    fig.savefig(f"{prefix}angle_visualization.pdf", format='pdf')
    plt.close(fig)
    return "Angle visualization plot completed"

def create_alpha_histogram_plot(data_dict, output_prefix=""):
    """Create histogram of alpha angles similar to alpha_demo.py"""
    pixel_alpha = data_dict['pixel_alpha']
    pixel_hit = data_dict['pixel_hit']
    
    # Filter out hits that are on pixels and NaN values
    valid_indices = ~(pixel_hit == True)
    valid_alpha = pixel_alpha[valid_indices]
    finite_alpha = valid_alpha[np.isfinite(valid_alpha)]
    
    if len(finite_alpha) == 0:
        print("No valid alpha angles to plot histogram")
        return "No valid alpha angles"
    
    fig = plt.figure(figsize=(16, 9), dpi=150)  # 16:9 aspect ratio for presentations
    ax = fig.add_subplot(111)
    
    # Use 100 bins like ROOT default
    num_bins = 100
    
    # Set up the bins to cover the full range
    min_alpha = np.min(finite_alpha)
    max_alpha = np.max(finite_alpha)
    buffer = (max_alpha - min_alpha) * 0.05 if max_alpha > min_alpha else 1.0
    bins = np.linspace(min_alpha - buffer, max_alpha + buffer, num_bins)
    
    # Add reference lines at 90 and 180 degrees
    ax.axvline(x=90, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    ax.axvline(x=180, color='purple', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    
    # Plot histogram
    hist, bin_edges, patches = ax.hist(finite_alpha, bins=bins, alpha=0.7, color='blue', 
                                     edgecolor='black', linewidth=0.5,
                                     label=f'Alpha angles ({len(finite_alpha)} hits)')
    
    # Set labels and title
    ax.set_xlabel('Alpha Angle (degrees)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Histogram of Alpha Angles ({len(finite_alpha)} hits)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add text annotations for reference lines
    y_lim = ax.get_ylim()
    y_max = y_lim[1]
    y_middle = y_max * 0.5
    
    ax.text(90, y_middle, '90°', color='darkgreen', fontsize=12, 
           ha='center', va='center', weight='bold', 
           bbox=dict(facecolor='white', alpha=0.7, pad=1))
    ax.text(180, y_middle, '180°', color='purple', fontsize=12, 
           ha='center', va='center', weight='bold', 
           bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Add statistics text
    mean_alpha = np.mean(finite_alpha)
    median_alpha = np.median(finite_alpha)
    std_alpha = np.std(finite_alpha)
    
    stats_text = (f"Mean: {mean_alpha:.2f}°\n"
                 f"Median: {median_alpha:.2f}°\n"
                 f"Std Dev: {std_alpha:.2f}°")
    
    ax.text(0.97, 0.97, stats_text, 
           transform=ax.transAxes, 
           verticalalignment='top', 
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
           fontsize=11)
    
    ax.legend(loc='upper left')
    
    # Set x-axis limits
    x_min = max(0, min_alpha - buffer)
    x_max = min(max(max_alpha + buffer, 185), 360)
    ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}alpha_histogram.png", dpi=300)
    fig.savefig(f"{prefix}alpha_histogram.svg", format='svg')
    fig.savefig(f"{prefix}alpha_histogram.pdf", format='pdf')
    plt.close(fig)
    return "Alpha histogram plot completed"

def create_alpha_vs_distance_plot(data_dict, output_prefix=""):
    """Create scatter plot of alpha angle vs distance similar to alpha_demo.py"""
    pixel_alpha = data_dict['pixel_alpha']
    pixel_dist = data_dict['pixel_dist']
    pixel_hit = data_dict['pixel_hit']
    
    # Filter out hits that are on pixels and NaN values
    valid_indices = ~(pixel_hit == True)
    valid_alpha = pixel_alpha[valid_indices]
    valid_dist = pixel_dist[valid_indices]
    
    finite_indices = np.isfinite(valid_alpha)
    finite_alpha = valid_alpha[finite_indices]
    finite_dist = valid_dist[finite_indices]
    
    if len(finite_alpha) == 0:
        print("No valid alpha angles to plot scatter plot")
        return "No valid alpha angles"
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    
    # Create scatter plot
    scatter = ax.scatter(finite_dist, finite_alpha, s=20, alpha=0.7, 
                        color='blue', edgecolors='black', linewidth=0.3)
    
    ax.set_title(f'Alpha Angle vs Distance ({len(finite_alpha)} hits)', fontsize=14)
    ax.set_xlabel('Distance to Pixel Center [mm]', fontsize=12)
    ax.set_ylabel('Alpha Angle [degrees]', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis limits
    if len(finite_dist) > 0:
        ax.set_xlim(0, max(finite_dist) * 1.1)
        ax.set_ylim(0, max(finite_alpha) * 1.1)
    
    # Add statistics text
    if len(finite_alpha) > 0:
        corr_coef = np.corrcoef(finite_dist, finite_alpha)[0, 1]
        mean_alpha = np.mean(finite_alpha)
        mean_dist = np.mean(finite_dist)
        
        stats_text = (f"Points: {len(finite_alpha)}\n"
                     f"Mean α: {mean_alpha:.2f}°\n"
                     f"Mean dist: {mean_dist:.3f} mm\n"
                     f"Correlation: {corr_coef:.3f}")
        
        ax.text(0.97, 0.03, stats_text, 
               transform=ax.transAxes, 
               verticalalignment='bottom', 
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
               fontsize=10)
    
    plt.tight_layout()
    
    prefix = f"{output_prefix}_" if output_prefix else ""
    fig.savefig(f"{prefix}alpha_vs_distance.png", dpi=300)
    fig.savefig(f"{prefix}alpha_vs_distance.svg", format='svg')
    fig.savefig(f"{prefix}alpha_vs_distance.pdf", format='pdf')
    plt.close(fig)
    return "Alpha vs distance plot completed"

def verify_alpha_calculation(data_dict, output_prefix=""):
    """Verify that our alpha calculation matches the values stored in the ROOT file from Geant4 simulation"""
    pos_x, pos_y = data_dict['pos_x'], data_dict['pos_y']
    pixel_x, pixel_y = data_dict['pixel_x'], data_dict['pixel_y']
    pixel_alpha = data_dict['pixel_alpha']
    pixel_hit = data_dict['pixel_hit']
    
    # Filter out hits that are on pixels (where alpha calculation doesn't apply)
    valid_indices = ~(pixel_hit == True)
    valid_pos_x = pos_x[valid_indices]
    valid_pos_y = pos_y[valid_indices]
    valid_pixel_x = pixel_x[valid_indices]
    valid_pixel_y = pixel_y[valid_indices]
    valid_alpha = pixel_alpha[valid_indices]
    
    # Also filter out NaN values (hits inside pixels)
    finite_indices = np.isfinite(valid_alpha)
    valid_pos_x = valid_pos_x[finite_indices]
    valid_pos_y = valid_pos_y[finite_indices]
    valid_pixel_x = valid_pixel_x[finite_indices]
    valid_pixel_y = valid_pixel_y[finite_indices]
    valid_alpha = valid_alpha[finite_indices]
    
    # Parameters from the detector (matching Geant4 simulation)
    pixel_size = 0.1  # mm (100 μm)
    
    def calculate_alpha_python(hit_x, hit_y, pixel_center_x, pixel_center_y):
        """
        Calculate alpha angle using the exact same method as alpha_demo.py and Geant4 simulation
        """
        # Calculate the four corners of the pixel (same as in alpha_demo.py and Geant4)
        half_pixel = pixel_size / 2.0
        corners = [
            (pixel_center_x - half_pixel, pixel_center_y - half_pixel),  # bottom-left (0)
            (pixel_center_x + half_pixel, pixel_center_y - half_pixel),  # bottom-right (1)
            (pixel_center_x + half_pixel, pixel_center_y + half_pixel),  # top-right (2)
            (pixel_center_x - half_pixel, pixel_center_y + half_pixel)   # top-left (3)
        ]
        
        # Calculate angles to each corner from the hit point (same method as alpha_demo.py)
        angles = []
        for i, (corner_x, corner_y) in enumerate(corners):
            dx = corner_x - hit_x
            dy = corner_y - hit_y
            angle = np.arctan2(dy, dx)
            angles.append((angle, i))  # Store corner index with angle
        
        # Sort by angle
        angles.sort()
        sorted_angles = [a[0] for a in angles]
        
        # Calculate differences between consecutive angles
        angle_diffs = []
        for i in range(len(sorted_angles)):
            diff = sorted_angles[(i+1) % len(sorted_angles)] - sorted_angles[i]
            # Handle wrap-around (angles close to 2π)
            if diff < 0:
                diff += 2 * np.pi
            angle_diffs.append(diff)
        
        # The maximum angle is 2π minus the largest difference
        alpha = 2 * np.pi - max(angle_diffs)
        
        # Convert to degrees (same as Geant4 simulation)
        alpha_degrees = alpha * (180.0 / np.pi)
        
        return alpha_degrees
    
    # Calculate alpha for a subset of points for verification
    max_verify = min(100, len(valid_pos_x))
    if len(valid_pos_x) > max_verify:
        indices = np.random.choice(len(valid_pos_x), max_verify, replace=False)
        verify_pos_x = valid_pos_x[indices]
        verify_pos_y = valid_pos_y[indices]
        verify_pixel_x = valid_pixel_x[indices]
        verify_pixel_y = valid_pixel_y[indices]
        verify_alpha_stored = valid_alpha[indices]
    else:
        verify_pos_x = valid_pos_x
        verify_pos_y = valid_pos_y
        verify_pixel_x = valid_pixel_x
        verify_pixel_y = valid_pixel_y
        verify_alpha_stored = valid_alpha
    
    # Calculate alpha using our Python implementation
    python_alphas = []
    for i in range(len(verify_pos_x)):
        alpha_calc = calculate_alpha_python(verify_pos_x[i], verify_pos_y[i], 
                                          verify_pixel_x[i], verify_pixel_y[i])
        python_alphas.append(alpha_calc)
    
    python_alphas = np.array(python_alphas)
    
    # Compare the results
    differences = np.abs(python_alphas - verify_alpha_stored)
    max_diff = np.max(differences)
    mean_diff = np.mean(differences)
    rms_diff = np.sqrt(np.mean(differences**2))
    
    print(f"\n=== Alpha Calculation Verification ===")
    print(f"Verified {len(verify_pos_x)} alpha calculations")
    print(f"Maximum difference: {max_diff:.6f} degrees")
    print(f"Mean difference: {mean_diff:.6f} degrees")
    print(f"RMS difference: {rms_diff:.6f} degrees")
    
    if max_diff < 1e-10:  # Very small tolerance for floating point precision
        print("✓ VERIFICATION PASSED: Alpha calculations match perfectly!")
    elif max_diff < 1e-6:
        print("✓ VERIFICATION PASSED: Alpha calculations match within numerical precision")
    else:
        print("✗ VERIFICATION FAILED: Significant differences found")
        print("Sample comparisons (first 5):")
        for i in range(min(5, len(python_alphas))):
            print(f"  Point {i}: Stored={verify_alpha_stored[i]:.6f}°, "
                  f"Calculated={python_alphas[i]:.6f}°, "
                  f"Diff={differences[i]:.6f}°")
    
    print("======================================\n")
    
    # Create comparison plot
    if len(python_alphas) > 0:
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111)
        
        # Scatter plot comparing stored vs calculated values
        scatter = ax.scatter(verify_alpha_stored, python_alphas, s=20, alpha=0.7, 
                           color='blue', edgecolors='black', linewidth=0.3)
        
        # Add perfect correlation line (y=x)
        min_val = min(np.min(verify_alpha_stored), np.min(python_alphas))
        max_val = max(np.max(verify_alpha_stored), np.max(python_alphas))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Perfect correlation (y=x)')
        
        ax.set_title(f'Alpha Angle Verification ({len(python_alphas)} points)', fontsize=14)
        ax.set_xlabel('Stored Alpha (from Geant4) [degrees]', fontsize=12)
        ax.set_ylabel('Calculated Alpha (Python) [degrees]', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # Add statistics text
        stats_text = (f"Max diff: {max_diff:.2e}°\n"
                     f"Mean diff: {mean_diff:.2e}°\n"
                     f"RMS diff: {rms_diff:.2e}°")
        
        ax.text(0.03, 0.97, stats_text, 
               transform=ax.transAxes, 
               verticalalignment='top', 
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
               fontsize=10)
        
        plt.tight_layout()
        
        prefix = f"{output_prefix}_" if output_prefix else ""
        fig.savefig(f"{prefix}alpha_verification.png", dpi=300)
        fig.savefig(f"{prefix}alpha_verification.svg", format='svg')
        fig.savefig(f"{prefix}alpha_verification.pdf", format='pdf')
        plt.close(fig)
    
    return "Alpha verification completed"

def load_grid_parameters_from_root(root_file):
    """
    Load grid parameters from ROOT file metadata.
    Returns tuple: (pixel_size, pixel_spacing, pixel_corner_offset, detector_size, num_blocks_per_side)
    Returns None if metadata is not found.
    """
    try:
        # Try to read grid parameters from ROOT metadata (TNamed objects)
        pixel_size = float(root_file['GridPixelSize;1'].member('fTitle'))
        pixel_spacing = float(root_file['GridPixelSpacing;1'].member('fTitle'))
        pixel_corner_offset = float(root_file['GridPixelCornerOffset;1'].member('fTitle'))
        detector_size = float(root_file['GridDetectorSize;1'].member('fTitle'))
        num_blocks_per_side = int(root_file['GridNumBlocksPerSide;1'].member('fTitle'))
        
        print(f"Loaded grid parameters from ROOT metadata:")
        print(f"  Pixel size: {pixel_size} mm")
        print(f"  Pixel spacing: {pixel_spacing} mm")
        print(f"  Pixel corner offset: {pixel_corner_offset} mm")
        print(f"  Detector size: {detector_size} mm")
        print(f"  Number of blocks per side: {num_blocks_per_side}")
        
        return {
            'pixel_size': pixel_size,
            'pixel_spacing': pixel_spacing,
            'pixel_corner_offset': pixel_corner_offset,
            'detector_size': detector_size,
            'num_blocks_per_side': num_blocks_per_side
        }
    except KeyError as e:
        print(f"Grid metadata not found in ROOT file: {e}")
        return None
    except Exception as e:
        print(f"Error reading grid metadata: {e}")
        return None

def get_detector_parameters():
    """Get detector parameters, either from ROOT metadata or fallback to defaults"""
    # Default parameters (fallback values matching the actual simulation)
    default_params = {
        'pixel_size': 0.1,  # mm (100 μm)
        'pixel_spacing': 0.5,  # mm (500 μm)
        'pixel_corner_offset': -0.05,  # mm (ACTUAL value used in Geant4)
        'detector_size': 30.0,  # mm (3 cm)
        'num_blocks_per_side': 61  # This matches the actual ROOT data
    }
    
    return default_params

def main():
    parser = argparse.ArgumentParser(description='Visualize hit positions and pixel approximations')
    parser.add_argument('-f', '--file', type=str, default="/home/tom/Desktop/Cultural_Keys/epicToy/build/epicToyOutput.root",
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
            # Load grid parameters from ROOT metadata
            grid_params = load_grid_parameters_from_root(file)
            if grid_params is None:
                print("Using default grid parameters (fallback)")
                grid_params = get_detector_parameters()
            else:
                print("Successfully loaded grid parameters from ROOT metadata")
            
            print(f"Using grid parameters:")
            for key, value in grid_params.items():
                print(f"  {key}: {value}")
            
            # Access the tree named "Hits" (handle version numbers like "Hits;1")
            tree_name = None
            for key in file.keys():
                if key.startswith("Hits"):
                    tree_name = key
                    break
            
            if tree_name is None:
                print(f"Error: Tree 'Hits' not found in file. Available objects: {list(file.keys())}")
                return 1
            
            tree = file[tree_name]
            print(f"Tree '{tree_name}' successfully opened with {tree.num_entries} entries")
            
            # List available branches
            print("Available branches:", tree.keys())
            
            # Read the position and pixel data
            data = tree.arrays(["TrueX", "TrueY", "PixelX", "PixelY", "PixelDist", "PixelI", "PixelJ", "PixelHit", "PixelAlpha"])
            
            # Convert awkward arrays to numpy arrays for plotting
            # Use ak.to_numpy() instead of np.array() to avoid deprecation warnings
            import awkward as ak
            pos_x = ak.to_numpy(data["TrueX"])
            pos_y = ak.to_numpy(data["TrueY"])
            pixel_x = ak.to_numpy(data["PixelX"])
            pixel_y = ak.to_numpy(data["PixelY"])
            pixel_dist = ak.to_numpy(data["PixelDist"])
            pixel_i = ak.to_numpy(data["PixelI"])
            pixel_hit = ak.to_numpy(data["PixelHit"])
            pixel_j = ak.to_numpy(data["PixelJ"])
            pixel_alpha = ak.to_numpy(data["PixelAlpha"])
            
            # Create a data dictionary to share with worker processes
            data_dict = {
                'pos_x': pos_x,
                'pos_y': pos_y,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'pixel_dist': pixel_dist,
                'pixel_i': pixel_i,
                'pixel_j': pixel_j,
                'pixel_hit': pixel_hit,
                'pixel_alpha': pixel_alpha
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
                (create_pixel_indices_heatmap, data_dict),
                (partial(create_angle_visualization_plot, grid_params=grid_params), data_dict),
                (create_alpha_histogram_plot, data_dict),
                (create_alpha_vs_distance_plot, data_dict),
                #(verify_alpha_calculation, data_dict)
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
