#!/usr/bin/env python3
# filepath: /home/tom/Desktop/epicToy/python/calc_alpha.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math
import csv
import os
import time
import multiprocessing
from functools import partial
from matplotlib.path import Path
from matplotlib.widgets import Button

class PixelDetectorVisualizer:
    """
    Interactive visualization of a pixel detector with functionality to:
    - Place point particles in the detector (outside pixels)
    - Calculate angles between particles and pixels
    - Visualize these angles with arcs and lines
    """
    def __init__(self, histogram_bins=100):
        # Parameters from DetectorConstruction.cc (converting to mm)
        self.pixel_size = 0.1  # mm (100 μm)
        self.pixel_width = 0.001  # mm (1 μm) - not used in 2D visualization
        self.pixel_spacing = 0.5  # mm (500 μm)
        self.pixel_corner_offset = 0.1  # mm (100 μm)
        self.det_size = 30.0  # mm (3 cm)
        
        # Histogram configuration (similar to ROOT defaults)
        self.histogram_bins = histogram_bins  # Default to 100 bins like ROOT
        
        # Calculate number of pixels per side using the same formula as in Geant4
        self.num_blocks_per_side = round((self.det_size - 2*self.pixel_corner_offset - self.pixel_size)/self.pixel_spacing + 1)
        
        # Recalculate the corner offset for perfect centering (as done in Geant4)
        self.pixel_corner_offset = (self.det_size - (self.num_blocks_per_side-1)*self.pixel_spacing - self.pixel_size)/2
        
        # Create main figure and axis for detector visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax_info = self.fig.add_axes([0.05, 0.05, 0.3, 0.2])  # info panel
        self.ax_info.axis('off')
        
        # Add a button to save detector visualization
        self.save_detector_button_ax = self.fig.add_axes([0.7, 0.01, 0.25, 0.05])
        self.save_detector_button = Button(self.save_detector_button_ax, 'Save Detector Image')
        self.save_detector_button.on_clicked(lambda event: self.save_detector_visualization())
        
        # Create separate figure for alpha vs distance scatter plot
        self.scatter_fig, self.scatter_ax = plt.subplots(figsize=(10, 8))
        self.scatter_ax.set_title('Alpha Angle vs Distance (Combined Cases)')
        self.scatter_ax.set_xlabel('Distance (mm)')
        self.scatter_ax.set_ylabel('Alpha Angle (degrees)')
        self.scatter_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a button to save CSV data
        self.save_csv_button_ax = self.scatter_fig.add_axes([0.7, 0.01, 0.25, 0.05])
        self.save_csv_button = Button(self.save_csv_button_ax, 'Save to CSV')
        self.save_csv_button.on_clicked(lambda event: self.save_to_csv())
        
        # Store distance and alpha values for scatter plot
        self.distances = []
        self.alphas = []
        self.point_types = []  # 1 for same side case, 2 for adjacent sides case
        self.scatter_plot = None
        
        # Create separate figures for the two cases
        self.case1_fig, self.case1_ax = plt.subplots(figsize=(10, 8))
        self.case1_ax.set_title('Alpha Angle vs Distance (Same Side Case - 0 points)')
        self.case1_ax.set_xlabel('Distance (mm)')
        self.case1_ax.set_ylabel('Alpha Angle (degrees)')
        self.case1_ax.grid(True, linestyle='--', alpha=0.7)
        self.case1_scatter = None
        
        self.case2_fig, self.case2_ax = plt.subplots(figsize=(10, 8))
        self.case2_ax.set_title('Alpha Angle vs Distance (Adjacent Sides Case - 0 points)')
        self.case2_ax.set_xlabel('Distance (mm)')
        self.case2_ax.set_ylabel('Alpha Angle (degrees)')
        self.case2_ax.grid(True, linestyle='--', alpha=0.7)
        self.case2_scatter = None
        
        # Create figure for alpha angle histogram (wider for presentation format)
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(16, 9))  # 16:9 aspect ratio for presentations
        self.hist_ax.set_title('Histogram of Alpha Angles (0 points)', fontsize=14)
        self.hist_ax.set_xlabel('Alpha Angle (degrees)', fontsize=13)
        self.hist_ax.set_ylabel('Count', fontsize=13)
        self.hist_ax.tick_params(axis='both', which='major', labelsize=12)
        self.hist_ax.grid(True, linestyle='--', alpha=0.7)
        self.hist_bars = None
        
        # Set up buttons
        #self.new_points_button_ax = self.fig.add_axes([0.65, 0.05, 0.15, 0.05])
        #self.new_points_button = Button(self.new_points_button_ax, 'New Points')
        #self.new_points_button.on_clicked(self.generate_random_points)
        
        #self.clear_button_ax = self.fig.add_axes([0.8, 0.05, 0.15, 0.05])
        #self.clear_button = Button(self.clear_button_ax, 'Clear Points')
        #self.clear_button.on_clicked(self.clear_points)
        
        # Configuration for points
        self.num_random_points = 10000  # Number of random points to generate
        self.point_radius = 0.01    # Smaller point radius (mm)
        self.use_multiprocessing = True  # Enable multiprocessing for point generation
        self.num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        
        # Initialize data structures for tracking
        self.points = []
        self.all_pixel_positions = []  # Will store (x, y, width, height) for each pixel
        self.pixel_patches = []        # Store the rectangle patches for each pixel
        self.point_visualizations = [] # Store all visual elements for each point

        # Setup detector and pixels
        self.setup_detector()
        
        # Set plot properties - tight to the detector outline (just a small margin)
        margin = 0.5  # 0.5mm margin
        self.ax.set_xlim(-self.det_size/2-margin, self.det_size/2+margin)
        self.ax.set_ylim(-self.det_size/2-margin, self.det_size/2+margin)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        #self.ax.set_title('Pixel Detector with Alpha Angle Calculation')
        
        # Show instructions
        #self.update_info_text("Click anywhere in the detector (outside pixels) to place a point.")

    def setup_detector(self):
        """Set up detector and pixel grid"""
        # Draw detector (as a square)
        detector = patches.Rectangle((-self.det_size/2, -self.det_size/2), self.det_size, self.det_size, 
                                   linewidth=2, edgecolor='green', facecolor='none', alpha=0.7)
        self.ax.add_patch(detector)
        
        # Calculate the position of first pixel
        first_pixel_pos = -self.det_size/2 + self.pixel_corner_offset
        
        # Create pixel grid
        for i in range(self.num_blocks_per_side):
            for j in range(self.num_blocks_per_side):
                # Calculate pixel position
                x = first_pixel_pos + i * self.pixel_spacing
                y = first_pixel_pos + j * self.pixel_spacing
                
                # Store pixel position for later lookup
                self.all_pixel_positions.append((x, y, self.pixel_size, self.pixel_size))
                
                # Create pixel as a rectangle
                pixel = patches.Rectangle((x, y), self.pixel_size, self.pixel_size,
                                        linewidth=1, edgecolor='blue', facecolor='lightblue')
                self.pixel_patches.append(pixel)
                self.ax.add_patch(pixel)
        
        # Calculate and display key statistics
        total_pixel_area = self.num_blocks_per_side * self.num_blocks_per_side * self.pixel_size * self.pixel_size
        detector_area = self.det_size * self.det_size
        pixel_area_ratio = total_pixel_area / detector_area
        
        stats_text = (
            f"Statistics:\n"
            f"Total pixels: {self.num_blocks_per_side * self.num_blocks_per_side}\n"
            f"Pixel coverage: {pixel_area_ratio*100:.2f}%\n"
            f"Pixel area: {total_pixel_area:.3f} mm²\n"
            f"Detector area: {detector_area:.3f} mm²"
        )
        print(stats_text)

    def is_point_inside_pixel(self, point_x, point_y):
        """Check if the given point is inside any pixel"""
        for x, y, width, height in self.all_pixel_positions:
            if (x <= point_x <= x + width) and (y <= point_y <= y + height):
                return True
        return False
    
    def is_point_inside_detector(self, point_x, point_y):
        """Check if the given point is inside the detector"""
        half_size = self.det_size / 2
        return (-half_size <= point_x <= half_size) and (-half_size <= point_y <= half_size)

    def find_closest_pixel(self, point_x, point_y):
        """
        Find the closest pixel to the given point using the same algorithm as in Geant4 code.
        This matches the CalculateNearestPixel method from EventAction.cc
        """
        # Calculate the first pixel position (corner)
        first_pixel_pos = -self.det_size/2 + self.pixel_corner_offset + self.pixel_size/2
        
        # Calculate which pixel grid position is closest (i and j indices)
        i = round((point_x - first_pixel_pos) / self.pixel_spacing)
        j = round((point_y - first_pixel_pos) / self.pixel_spacing)
        
        # Clamp i and j to valid pixel indices
        i = max(0, min(i, self.num_blocks_per_side - 1))
        j = max(0, min(j, self.num_blocks_per_side - 1))
        
        # Calculate the actual pixel center position
        pixel_x = first_pixel_pos + i * self.pixel_spacing
        pixel_y = first_pixel_pos + j * self.pixel_spacing
        
        # Calculate distance from hit to pixel center
        distance = math.sqrt((point_x - pixel_x)**2 + (point_y - pixel_y)**2)
        
        # Convert i,j to linear index for our array
        pixel_idx = i * self.num_blocks_per_side + j
        
        return pixel_idx, distance

    def calculate_alpha(self, point_x, point_y, pixel_idx):
        """
        Calculate the maximum angle (alpha) from the point to the pixel
        without having lines intersect the pixel
        
        Returns:
        - alpha: maximum viewing angle
        - start_angle: starting angle of the view
        - end_angle: ending angle of the view
        - bisector_angle: middle of the viewing angle
        - point_type: 1 for same side case, 2 for adjacent sides case
        """
        if pixel_idx < 0 or pixel_idx >= len(self.all_pixel_positions):
            return 0, 0, 0, 0, 0
            
        # Get pixel coordinates
        px, py, pwidth, pheight = self.all_pixel_positions[pixel_idx]
        
        # Calculate pixel corners
        corners = [
            (px, py),                    # bottom-left (0)
            (px + pwidth, py),           # bottom-right (1)
            (px + pwidth, py + pheight), # top-right (2)
            (px, py + pheight)           # top-left (3)
        ]
        
        # Calculate angles to each corner from the point
        angles = []
        for i, (corner_x, corner_y) in enumerate(corners):
            dx = corner_x - point_x
            dy = corner_y - point_y
            angle = math.atan2(dy, dx)
            angles.append((angle, i))  # Store corner index with angle
        
        # Sort by angle
        angles.sort()
        sorted_angles = [a[0] for a in angles]
        sorted_indices = [a[1] for a in angles]
        
        # Calculate differences between consecutive angles
        angle_diffs = []
        for i in range(len(sorted_angles)):
            diff = sorted_angles[(i+1) % len(sorted_angles)] - sorted_angles[i]
            # Handle wrap-around (angles close to 2π)
            if diff < 0:
                diff += 2 * math.pi
            angle_diffs.append(diff)
        
        # The maximum angle is 2π minus the largest difference
        alpha = 2 * math.pi - max(angle_diffs)
        
        # Calculate the bisector angle (middle of the viewing angle)
        max_diff_idx = angle_diffs.index(max(angle_diffs))
        start_angle = sorted_angles[max_diff_idx]
        end_angle = sorted_angles[(max_diff_idx + 1) % len(sorted_angles)]
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        bisector_angle = (start_angle + end_angle) / 2
        
        # Determine corner indices that define the alpha angle
        corner1_idx = sorted_indices[max_diff_idx]
        corner2_idx = sorted_indices[(max_diff_idx + 1) % len(sorted_indices)]
        
        # Determine if this is a same side case (1) or adjacent sides case (2)
        # Same side pairs are (0,1), (1,2), (2,3), or (3,0)
        same_side_pairs = [(0,1), (1,2), (2,3), (3,0)]
        sorted_pair = tuple(sorted([corner1_idx, corner2_idx]))
        
        if sorted_pair in [(0,1), (1,2), (2,3), (0,3)]:
            point_type = 1  # Same side
        else:
            point_type = 2  # Adjacent sides
            
        return alpha, start_angle, end_angle, bisector_angle, point_type

    def draw_alpha_visualization(self, point_x, point_y, pixel_idx):
        """Draw visualization of the alpha angle and projection lines"""
        visuals = []
        
        if pixel_idx < 0 or pixel_idx >= len(self.all_pixel_positions):
            return visuals
        
        # Get pixel data
        px, py, pwidth, pheight = self.all_pixel_positions[pixel_idx]
        
        # Highlight the closest pixel
        pixel_highlight = patches.Rectangle((px, py), pwidth, pheight,
                                           linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(pixel_highlight)
        visuals.append(pixel_highlight)
        
        # Calculate alpha and related angles
        alpha, start_angle, end_angle, bisector_angle, point_type = self.calculate_alpha(point_x, point_y, pixel_idx)
        
        # Calculate pixel center and distance to point
        pixel_center_x = px + pwidth/2
        pixel_center_y = py + pheight/2
        distance = math.sqrt((point_x - pixel_center_x)**2 + (point_y - pixel_center_y)**2)
        
        # Set arc radius based on distance to pixel (between 20-40% of distance)
        radius = min(max(distance * 0.3, 0.1), 0.5)  # Adaptive radius, min 0.1mm, max 0.5mm
        
        # Draw the angle arc in the complementary region (the larger portion of the circle)
        # This shows the region that the pixel occupies in the field of view
        arc = patches.Arc((point_x, point_y), radius*2, radius*2, 
                         theta1=math.degrees(end_angle), 
                         theta2=math.degrees(start_angle) + 360 if start_angle > end_angle else math.degrees(start_angle) + 360,
                         linewidth=2, color='red')
        self.ax.add_patch(arc)
        visuals.append(arc)
        
        # Get pixel corners used to define the alpha angle
        corners = [
            (px, py),                    # bottom-left
            (px + pwidth, py),           # bottom-right
            (px + pwidth, py + pheight), # top-right
            (px, py + pheight)           # top-left
        ]
        
        # Draw lines directly from point to the corners that define the maximum angle
        max_diff_idx = -1
        angle_diffs = []
        angles_sorted = sorted([(math.atan2(c[1] - point_y, c[0] - point_x), i) for i, c in enumerate(corners)])
        
        # Find the corners that define the alpha angle
        for i in range(len(angles_sorted)):
            next_idx = (i + 1) % len(angles_sorted)
            diff = angles_sorted[next_idx][0] - angles_sorted[i][0]
            if diff < 0:
                diff += 2 * math.pi
            angle_diffs.append(diff)
            
        max_diff_idx = angle_diffs.index(max(angle_diffs))
        corner1 = corners[angles_sorted[max_diff_idx][1]]
        corner2 = corners[angles_sorted[(max_diff_idx + 1) % len(angles_sorted)][1]]
        
        # Draw lines to the actual corners
        line1 = self.ax.plot([point_x, corner1[0]], [point_y, corner1[1]], 'r-', linewidth=1.5)[0]
        line2 = self.ax.plot([point_x, corner2[0]], [point_y, corner2[1]], 'r-', linewidth=1.5)[0]
        visuals.append(line1)
        visuals.append(line2)
        
        # Calculate the complementary bisector angle (middle of the larger angle region)
        comp_bisector_angle = bisector_angle + math.pi
        if comp_bisector_angle > 2 * math.pi:
            comp_bisector_angle -= 2 * math.pi
            
        # Only show angle indicator text if we have <= 100 points
        if len(self.points) <= 100:
            radius_factor = 0.7  # Adjust as needed for text positioning
            #angle_text = self.ax.text(point_x + radius*radius_factor*math.cos(comp_bisector_angle),
                                    #point_y + radius*radius_factor*math.sin(comp_bisector_angle),
                                    #f'α = {math.degrees(alpha):.1f}°',
                                    #color='red', fontsize=10,
                                    #horizontalalignment='center',
                                    #verticalalignment='center')
            #svisuals.append(angle_text)
        
        # Draw dotted line to the pixel center
        dist_line = self.ax.plot([point_x, pixel_center_x], 
                                [point_y, pixel_center_y], 
                                'k--', linewidth=1, alpha=0.6)[0]
        visuals.append(dist_line)
        
        # Distance is already calculated above
        
        # Add text box with both distance and angle only if we have <= 100 points
        #if len(self.points) <= 100:
            #info_text = self.ax.text(point_x + 0.1, point_y + 0.1, 
                                #f'd = {distance:.2f} mm\nα = {math.degrees(alpha):.1f}°',
                                #fontsize=7, color='black',
                                #bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', pad=0.2),
                                #verticalalignment='bottom')
            #visuals.append(info_text)
        
        #return visuals

    def generate_random_points(self, event=None):
        """Generate random points in the detector (outside pixels) using multiprocessing"""
        self.clear_points()  # Clear existing points first
        
        # Use multiprocessing if enabled
        if self.use_multiprocessing and self.num_random_points > 1000:
            self._generate_points_multiprocessing()
        else:
            self._generate_points_single_process()
    
    def _generate_points_single_process(self):
        """Generate points using a single process (original method)"""
        points_added = 0
        batch_size = min(1000, self.num_random_points)  # Process in smaller batches
        total_attempts = 0
        max_total_attempts = self.num_random_points * 10  # Limit total attempts
        
        half_size = self.det_size / 2
        
        # Pre-compute pixel boundaries for faster checking
        pixel_bounds = []
        for x, y, width, height in self.all_pixel_positions:
            pixel_bounds.append((x, y, x + width, y + height))
        
        while points_added < self.num_random_points and total_attempts < max_total_attempts:
            # Generate multiple random positions at once (vectorized)
            xs = np.random.uniform(-half_size, half_size, batch_size)
            ys = np.random.uniform(-half_size, half_size, batch_size)
            
            for x, y in zip(xs, ys):
                total_attempts += 1
                
                # Quick check if inside detector
                if abs(x) > half_size or abs(y) > half_size:
                    continue
                
                # Check if not inside any pixel (optimized)
                if not self._is_point_inside_any_pixel(x, y, pixel_bounds):
                    self.add_point(x, y)
                    points_added += 1
                    
                    if points_added >= self.num_random_points:
                        break
            
            # Print progress every 10% steps
            if points_added > 0 and points_added % (self.num_random_points // 10) == 0:
                print(f"Progress: {points_added}/{self.num_random_points} points generated ({points_added/self.num_random_points*100:.1f}%)")
        
        print(f"Single-process: Generated {points_added} random points with {total_attempts} attempts")
    
    def _is_point_inside_any_pixel(self, x, y, pixel_bounds):
        """Optimized check if point is inside any pixel"""
        for xmin, ymin, xmax, ymax in pixel_bounds:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False
    
    def _generate_points_multiprocessing(self):
        """Generate points using multiple processes"""
        print(f"Using {self.num_processes} processes for point generation")
        
        # Calculate how many points each process should try to generate
        points_per_process = int(np.ceil(self.num_random_points / self.num_processes))
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Create partial function with fixed parameters
            worker_func = partial(
                self._worker_generate_points,
                points_to_generate=points_per_process,
                half_size=self.det_size/2,
                pixel_positions=self.all_pixel_positions
            )
            
            # Map the worker function to the processes
            results = pool.map(worker_func, range(self.num_processes))
        
        # Combine the results from all processes
        valid_points = []
        for process_points in results:
            valid_points.extend(process_points)
        
        # We might have more points than needed, so truncate
        valid_points = valid_points[:self.num_random_points]
        print(f"Multiprocessing: Generated {len(valid_points)} valid points")
        
        # Add the points to the visualization
        for x, y in valid_points:
            self.add_point(x, y)
    
    @staticmethod
    def _worker_generate_points(process_id, points_to_generate, half_size, pixel_positions):
        """Worker function for each process to generate points"""
        np.random.seed(42 + process_id)  # Different seed for each process
        
        valid_points = []
        max_attempts = points_to_generate * 10
        attempts = 0
        
        # Pre-compute pixel boundaries for faster checking
        pixel_bounds = [(x, y, x + w, y + h) for x, y, w, h in pixel_positions]
        
        while len(valid_points) < points_to_generate and attempts < max_attempts:
            # Generate multiple random positions at once (vectorized)
            batch_size = min(1000, points_to_generate - len(valid_points))
            xs = np.random.uniform(-half_size, half_size, batch_size)
            ys = np.random.uniform(-half_size, half_size, batch_size)
            
            for x, y in zip(xs, ys):
                attempts += 1
                
                # Check if point is inside detector (should always be true with our generation method)
                if abs(x) > half_size or abs(y) > half_size:
                    continue
                
                # Check if not inside any pixel
                is_inside_pixel = False
                for xmin, ymin, xmax, ymax in pixel_bounds:
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        is_inside_pixel = True
                        break
                
                if not is_inside_pixel:
                    valid_points.append((x, y))
                    
                    # If we've found enough points, stop
                    if len(valid_points) >= points_to_generate:
                        break
        
        return valid_points

    def add_point(self, x, y):
        """Add a point and calculate/visualize its alpha angle"""
        # Add point to the list
        self.points.append((x, y))
        
        # Find the closest pixel
        closest_pixel_idx, distance = self.find_closest_pixel(x, y)
        
        # Draw the point (with smaller radius)
        point_patch = patches.Circle((x, y), radius=self.point_radius, color='black')
        self.ax.add_patch(point_patch)
        
        # Draw alpha visualization
        visuals = self.draw_alpha_visualization(x, y, closest_pixel_idx)
        #visuals.append(point_patch)
        
        # Store all visuals associated with this point
        self.point_visualizations.append(visuals)
        
        # Calculate alpha and point type for info text
        alpha, _, _, _, point_type = self.calculate_alpha(x, y, closest_pixel_idx)
        
        # Add data points to scatter plot
        alpha_deg = math.degrees(alpha)
        self.distances.append(distance)
        self.alphas.append(alpha_deg)
        self.point_types.append(point_type)
        
        # Only update the plots periodically for better performance
        # Update on every point for first 100 points, then on every 1000th point
        update_threshold = 100 if len(self.points) <= 100 else 1000
        if len(self.points) == 1 or len(self.points) == self.num_random_points or len(self.points) % update_threshold == 0:
            self._update_scatter_plot()
            self._update_case_plots()
            self._update_histogram()
            self._update_histogram()
    
    def _update_scatter_plot(self):
        """Update the scatter plot with the current data (as a separate method for optimization)"""
        # Remove previous scatter plots if they exist
        if hasattr(self, 'scatter_plot_case1') and self.scatter_plot_case1:
            self.scatter_plot_case1.remove()
        if hasattr(self, 'scatter_plot_case2') and self.scatter_plot_case2:
            self.scatter_plot_case2.remove()
        
        # If there's a colorbar, remove it as we're using discrete colors with a legend
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
            delattr(self, 'colorbar')
        
        # Separate data by case type
        case1_indices = [i for i, t in enumerate(self.point_types) if t == 1]
        case2_indices = [i for i, t in enumerate(self.point_types) if t == 2]
        
        case1_distances = [self.distances[i] for i in case1_indices]
        case1_alphas = [self.alphas[i] for i in case1_indices]
        
        case2_distances = [self.distances[i] for i in case2_indices]
        case2_alphas = [self.alphas[i] for i in case2_indices]
        
        # Plot each case with a specific discrete color
        # Blue for Case 1 (Same Side) and Red for Case 2 (Adjacent Sides)
        self.scatter_plot_case1 = self.scatter_ax.scatter(
            case1_distances, 
            case1_alphas, 
            color='blue',
            label='Same Side Case', 
            s=50,
            alpha=0.7, 
            edgecolors='black'
        )
        
        self.scatter_plot_case2 = self.scatter_ax.scatter(
            case2_distances, 
            case2_alphas, 
            color='red', 
            label='Adjacent Sides Case', 
            s=50,
            alpha=0.7, 
            edgecolors='black'
        )
        
        # Always show the legend for clear distinction between the cases
        self.scatter_ax.legend(loc='upper right')
        
        # Update the title to show case counts
        total_points = len(self.distances)
        case1_count = len(case1_distances)
        case2_count = len(case2_distances)
        self.scatter_ax.set_title(f'Alpha Angle vs Distance (Combined: {total_points} points, ' +
                                 f'Same Side: {case1_count}, Adjacent Sides: {case2_count})')
        
        # Update axes limits
        if self.distances:
            self.scatter_ax.set_xlim(0, max(self.distances) * 1.1)
            self.scatter_ax.set_ylim(0, max(self.alphas) * 1.1)
        else:
            self.scatter_ax.set_xlim(0, 15)
            self.scatter_ax.set_ylim(0, 90)
            
        # Redraw the canvas
        self.scatter_fig.canvas.draw_idle()
            
    def _update_case_plots(self):
        """Update the separate case plots with the current data"""
        # Separate data by case
        case1_distances = [d for d, t in zip(self.distances, self.point_types) if t == 1]
        case1_alphas = [a for a, t in zip(self.alphas, self.point_types) if t == 1]
        
        case2_distances = [d for d, t in zip(self.distances, self.point_types) if t == 2]
        case2_alphas = [a for a, t in zip(self.alphas, self.point_types) if t == 2]
        
        # Update case 1 plot (Same Side Case - Blue)
        if hasattr(self, 'case1_scatter') and self.case1_scatter:
            self.case1_scatter.remove()
        
        if case1_distances:
            self.case1_scatter = self.case1_ax.scatter(
                case1_distances, 
                case1_alphas,
                color='blue', 
                s=50, 
                alpha=0.7, 
                edgecolors='black',
                label='Same Side Case'
            )
            self.case1_ax.set_xlim(0, max(case1_distances) * 1.1)
            self.case1_ax.set_ylim(0, max(case1_alphas) * 1.1)
        else:
            self.case1_ax.set_xlim(0, 15)
            self.case1_ax.set_ylim(0, 90)
        
        # Add case count to the title
        self.case1_ax.set_title(f'Alpha Angle vs Distance (Same Side Case - {len(case1_distances)} points)')
            
        # Update case 2 plot (Adjacent Sides Case - Red)
        if hasattr(self, 'case2_scatter') and self.case2_scatter:
            self.case2_scatter.remove()
            
        if case2_distances:
            self.case2_scatter = self.case2_ax.scatter(
                case2_distances, 
                case2_alphas,
                color='red', 
                s=50, 
                alpha=0.7, 
                edgecolors='black',
                label='Adjacent Sides Case'
            )
            self.case2_ax.set_xlim(0, max(case2_distances) * 1.1)
            self.case2_ax.set_ylim(0, max(case2_alphas) * 1.1)
        else:
            self.case2_ax.set_xlim(0, 15)
            self.case2_ax.set_ylim(0, 90)
            
        # Add case count to the title
        self.case2_ax.set_title(f'Alpha Angle vs Distance (Adjacent Sides Case - {len(case2_distances)} points)')
        
        # Redraw the canvases
        self.case1_fig.canvas.draw_idle()
        self.case2_fig.canvas.draw_idle()

    def _update_histogram(self):
        """Update the histogram of alpha angles"""
        # Clear previous histogram if it exists
        self.hist_ax.clear()
        self.hist_ax.grid(True, linestyle='--', alpha=0.7)
        self.hist_ax.set_xlabel('Alpha Angle (degrees)')
        self.hist_ax.set_ylabel('Count')
        
        # If we have no data, just return
        if not self.alphas:
            self.hist_ax.set_title('Histogram of Alpha Angles (0 points)', fontsize=14)
            self.hist_ax.tick_params(axis='both', which='major', labelsize=12)
            return
        
        # Use ROOT-like binning from the configured parameter
        # ROOT typically defaults to around 100 bins for better statistics visualization
        num_bins = self.histogram_bins
        
        # Set font sizes for better readability in presentations
        self.hist_ax.set_xlabel('Alpha Angle (degrees)', fontsize=13)
        self.hist_ax.set_ylabel('Count', fontsize=13)
        self.hist_ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Create separate histograms for each case
        case1_alphas = [a for a, t in zip(self.alphas, self.point_types) if t == 1]
        case2_alphas = [a for a, t in zip(self.alphas, self.point_types) if t == 2]
        
        # Set up the bins to cover the full range
        all_alphas = self.alphas.copy()
        if all_alphas:
            min_alpha = min(all_alphas)
            max_alpha = max(all_alphas)
            # Add a small buffer to the bins
            buffer = (max_alpha - min_alpha) * 0.05 if max_alpha > min_alpha else 1.0
            bins = np.linspace(min_alpha - buffer, max_alpha + buffer, num_bins)
        else:
            bins = num_bins
            min_alpha = 0
            max_alpha = 180
            
        # Add vertical lines at 90 and 180 degrees (common angular references in physics)
        # Add these before plotting the histogram so they appear behind the bars
        self.hist_ax.axvline(x=90, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
        self.hist_ax.axvline(x=180, color='purple', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
        
        # Calculate appropriate x-axis limits based on data
        x_min = max(0, min_alpha - buffer)  # Never go below 0 for angles
        x_max = min(max(max_alpha + buffer, 185), 360)  # Ensure 180° is visible but don't exceed 360°
        
        # Plot the histograms with transparency for overlap visibility
        # Use slightly thinner bars for the wider display
        hist_heights = []  # To track maximum height for y-axis scaling
        
        if case1_alphas:
            hist1, _, _ = self.hist_ax.hist(case1_alphas, bins=bins, alpha=0.7, color='blue', 
                                          label=f'Same Side Case ({len(case1_alphas)} points)',
                                          rwidth=0.9)  # Relative width of bars
            if len(hist1) > 0:
                hist_heights.append(max(hist1))
        
        if case2_alphas:
            hist2, _, _ = self.hist_ax.hist(case2_alphas, bins=bins, alpha=0.7, color='red', 
                                          label=f'Adjacent Sides Case ({len(case2_alphas)} points)',
                                          rwidth=0.9)  # Relative width of bars
            if len(hist2) > 0:
                hist_heights.append(max(hist2))
        
        # Add combined histogram as a line
        if all_alphas:
            # Calculate the histogram data without plotting
            hist_counts, bin_edges = np.histogram(all_alphas, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            if len(hist_counts) > 0:
                hist_heights.append(max(hist_counts))
            
            # Plot the line on top with steps
            self.hist_ax.step(bin_centers, hist_counts, where='mid', color='black', 
                            linestyle='--', linewidth=2, 
                            label=f'Combined ({len(all_alphas)} points)')
            
        # Adjust y-axis to fit the data with a small margin at the top
        if hist_heights:
            max_height = max(hist_heights)
            self.hist_ax.set_ylim(0, max_height * 1.15)  # Add 15% margin at the top
        
        # Add text annotations for the reference lines
        y_lim = self.hist_ax.get_ylim()
        y_max = y_lim[1]
        y_middle = y_max * 0.5  # Position text in the middle of the y-axis
        
        # Add text annotations directly on the plot for 90° and 180° lines
        self.hist_ax.text(90, y_middle, '90°', color='darkgreen', fontsize=12, 
                         ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=1))
        self.hist_ax.text(180, y_middle, '180°', color='purple', fontsize=12, 
                         ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        # Add legend without the reference lines and update title
        self.hist_ax.legend(loc='upper right')  # Position legend in upper right corner
        self.hist_ax.set_title(f'Histogram of Alpha Angles ({len(all_alphas)} points)', fontsize=14)
        
        # Set x-axis limits based on data
        self.hist_ax.set_xlim(x_min, x_max)
        
        # Adjust layout to fit the wider figure with proper spacing
        self.hist_fig.tight_layout(pad=2.0)
        
        # Redraw the canvas
        self.hist_fig.canvas.draw_idle()

    def clear_points(self, event=None):
        """Clear all points and visualizations"""
        # Remove all visual elements
        for visuals in self.point_visualizations:
            for item in visuals:
                if isinstance(item, patches.Patch):
                    item.remove()
                else:
                    # Handle other types like lines and texts
                    try:
                        item.remove()
                    except:
                        pass
        
        # Clear data structures
        self.points.clear()
        self.point_visualizations.clear()
        
        # Clear scatter plot data
        self.distances.clear()
        self.alphas.clear()
        self.point_types.clear()
        
        # Clear main scatter plot
        if hasattr(self, 'scatter_plot_case1') and self.scatter_plot_case1:
            self.scatter_plot_case1.remove()
            self.scatter_plot_case1 = None
        if hasattr(self, 'scatter_plot_case2') and self.scatter_plot_case2:
            self.scatter_plot_case2.remove()
            self.scatter_plot_case2 = None
        self.scatter_ax.set_title('Alpha Angle vs Distance (Combined Cases)')
        self.scatter_ax.set_xlim(0, 15)
        self.scatter_ax.set_ylim(0, 90)
        
        # Clear case 1 scatter plot
        if hasattr(self, 'case1_scatter') and self.case1_scatter:
            self.case1_scatter.remove()
            self.case1_scatter = None
        self.case1_ax.set_title('Alpha Angle vs Distance (Same Side Case - 0 points)')
        self.case1_ax.set_xlim(0, 15)
        self.case1_ax.set_ylim(0, 90)
        
        # Clear case 2 scatter plot
        if hasattr(self, 'case2_scatter') and self.case2_scatter:
            self.case2_scatter.remove()
            self.case2_scatter = None
        self.case2_ax.set_title('Alpha Angle vs Distance (Adjacent Sides Case - 0 points)')
        self.case2_ax.set_xlim(0, 15)
        self.case2_ax.set_ylim(0, 90)
        
        # Clear histogram
        self.hist_ax.clear()
        self.hist_ax.grid(True, linestyle='--', alpha=0.7)
        self.hist_ax.set_title('Histogram of Alpha Angles (0 points)', fontsize=14)
        self.hist_ax.set_xlabel('Alpha Angle (degrees)', fontsize=13)
        self.hist_ax.set_ylabel('Count', fontsize=13)
        self.hist_ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add the reference lines to empty histogram
        self.hist_ax.axvline(x=90, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
        self.hist_ax.axvline(x=180, color='purple', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
        
        # Add text annotations for reference lines
        self.hist_ax.text(90, 5, '90°', color='darkgreen', fontsize=12, 
                         ha='center', va='bottom', weight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=1))
        self.hist_ax.text(180, 5, '180°', color='purple', fontsize=12, 
                         ha='center', va='bottom', weight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=1))
                         
        # Set appropriate axis limits for empty plot
        self.hist_ax.set_xlim(0, 185)
        self.hist_ax.set_ylim(0, 10)
        
        # Redraw all canvases
        self.scatter_fig.canvas.draw_idle()
        self.case1_fig.canvas.draw_idle()
        self.case2_fig.canvas.draw_idle()
        self.hist_fig.canvas.draw_idle()
        self.hist_fig.canvas.draw_idle()
        
        # Update info text
        #self.update_info_text("All points cleared. Click in the detector to add points.")
        
        # Redraw
        self.fig.canvas.draw_idle()

    def update_info_text(self, text):
        """Update the information panel with new text"""
        self.ax_info.clear()
        self.ax_info.text(0.05, 0.95, text, 
                         transform=self.ax_info.transAxes,
                         verticalalignment='top')
        self.ax_info.axis('off')
        self.fig.canvas.draw_idle()

    def save_to_csv(self, filename=None):
        """Save alpha angle and distance data to CSV files and plots to image files"""
        if filename is None:
            # Generate a default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"alpha_distance_data_{timestamp}"
        else:
            # Remove extension if present to use as base filename
            filename_base = os.path.splitext(filename)[0]
        
        try:
            # Save combined data to CSV
            combined_filename = f"{filename_base}.csv"
            with open(combined_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Distance (mm)', 'Alpha Angle (degrees)', 'Point Type'])
                # Write data rows
                for d, a, t in zip(self.distances, self.alphas, self.point_types):
                    writer.writerow([d, a, t])
            
            print(f"Combined data saved to {os.path.abspath(combined_filename)}")
            
            # Save case 1 data (same side points)
            case1_filename = f"{filename_base}_case1_same_side.csv"
            case1_data = [(d, a) for d, a, t in zip(self.distances, self.alphas, self.point_types) if t == 1]
            with open(case1_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Distance (mm)', 'Alpha Angle (degrees)'])
                for d, a in case1_data:
                    writer.writerow([d, a])
            
            print(f"Case 1 data saved to {os.path.abspath(case1_filename)}")
            
            # Save case 2 data (adjacent sides points)
            case2_filename = f"{filename_base}_case2_adjacent_sides.csv"
            case2_data = [(d, a) for d, a, t in zip(self.distances, self.alphas, self.point_types) if t == 2]
            with open(case2_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Distance (mm)', 'Alpha Angle (degrees)'])
                for d, a in case2_data:
                    writer.writerow([d, a])
            
            print(f"Case 2 data saved to {os.path.abspath(case2_filename)}")
            
            # Save scatter plots to image files
            self._save_scatter_plots(filename_base)
            
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def _save_scatter_plots(self, filename_base):
        """Save all scatter plots to PNG, SVG, and PDF files"""
        try:
            # Make sure all scatter plots are up to date
            self._update_scatter_plot()
            self._update_case_plots()
            
            # Save combined plot
            self.scatter_fig.savefig(f"{filename_base}_combined.png", dpi=300)
            self.scatter_fig.savefig(f"{filename_base}_combined.svg", format='svg')
            self.scatter_fig.savefig(f"{filename_base}_combined.pdf", format='pdf')
            print(f"Combined scatter plot saved to {os.path.abspath(f'{filename_base}_combined.png/svg/pdf')}")
            
            # Save case 1 plot
            self.case1_fig.savefig(f"{filename_base}_case1_same_side.png", dpi=300)
            self.case1_fig.savefig(f"{filename_base}_case1_same_side.svg", format='svg')
            self.case1_fig.savefig(f"{filename_base}_case1_same_side.pdf", format='pdf')
            print(f"Case 1 scatter plot saved to {os.path.abspath(f'{filename_base}_case1_same_side.png/svg/pdf')}")
            
            # Save case 2 plot
            self.case2_fig.savefig(f"{filename_base}_case2_adjacent_sides.png", dpi=300)
            self.case2_fig.savefig(f"{filename_base}_case2_adjacent_sides.svg", format='svg')
            self.case2_fig.savefig(f"{filename_base}_case2_adjacent_sides.pdf", format='pdf')
            print(f"Case 2 scatter plot saved to {os.path.abspath(f'{filename_base}_case2_adjacent_sides.png/svg/pdf')}")
            
            # Save histogram plot
            self.hist_fig.savefig(f"{filename_base}_alpha_histogram.png", dpi=300)
            self.hist_fig.savefig(f"{filename_base}_alpha_histogram.svg", format='svg')
            self.hist_fig.savefig(f"{filename_base}_alpha_histogram.pdf", format='pdf')
            print(f"Alpha histogram plot saved to {os.path.abspath(f'{filename_base}_alpha_histogram.png/svg/pdf')}")
            
            return True
        except Exception as e:
            print(f"Error saving scatter plots: {e}")
            return False

    def save_detector_visualization(self, filename_base=None):
        """Save the detector visualization (grid with angle lines) to image files"""
        if filename_base is None:
            # Generate a default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"detector_visualization_{timestamp}"
        
        try:
            # Save in different formats
            self.fig.savefig(f"{filename_base}_detector.png", dpi=300)
            print(f"Detector visualization saved to {os.path.abspath(f'{filename_base}_detector.png')}")
            
            self.fig.savefig(f"{filename_base}_detector.svg", format='svg')
            print(f"Detector visualization saved to {os.path.abspath(f'{filename_base}_detector.svg')}")
            
            self.fig.savefig(f"{filename_base}_detector.pdf", format='pdf')
            print(f"Detector visualization saved to {os.path.abspath(f'{filename_base}_detector.pdf')}")
            
            return True
        except Exception as e:
            print(f"Error saving detector visualization: {e}")
            return False

    def run(self):
        """Run the interactive visualization"""
        plt.tight_layout()
        
        start_time = time.time()
        print("Generating random points...")
        
        # Generate initial set of random points automatically
        self.generate_random_points()
        
        # Make sure histogram is updated
        self._update_histogram()
        
        end_time = time.time()
        print(f"Point generation completed in {end_time - start_time:.2f} seconds")
        
        # Generate timestamp for filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_base = f"alpha_distance_data_{timestamp}"
        
        # Save data to CSV file
        self.save_to_csv(filename_base)
        
        # Save detector visualization
        self.save_detector_visualization(filename_base)
        
        plt.show()

def create_detector_visualization(histogram_bins=100):
    """
    Create and run the detector visualization
    
    Parameters:
    -----------
    histogram_bins : int, default=100
        Number of bins to use for histograms (ROOT-like default is 100)
    """
    visualizer = PixelDetectorVisualizer(histogram_bins=histogram_bins)
    visualizer.run()

if __name__ == "__main__":
    create_detector_visualization(histogram_bins=100)  # Use ROOT-like default of 100 bins