#!/usr/bin/env python3
"""
Grid charge sharing analyzer for individual hits from ROOT simulation data.
Shows neighborhood (9x9) pixel grid visualizations with charge distribution around specific hit positions.
Also generates ensemble averages and handles edge cases.
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import random

class RandomHitChargeGridGenerator:
    """
    Generator for neighborhood (9x9) grid charge sharing visualizations for random individual hits
    """
    
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.detector_params = {}
        self.load_data()
        
    def load_data(self):
        """Load data from ROOT file"""
        try:
            with uproot.open(self.filename) as file:
                # Load detector parameters from TNamed objects
                self.detector_params = {
                    'pixel_size': float(file['GridPixelSize'].member('fTitle')),
                    'pixel_spacing': float(file['GridPixelSpacing'].member('fTitle')), 
                    'pixel_corner_offset': float(file['GridPixelCornerOffset'].member('fTitle')),
                    'detector_size': float(file['GridDetectorSize'].member('fTitle')),
                    'num_blocks_per_side': int(file['GridNumBlocksPerSide'].member('fTitle'))
                }
                
                # Load hit data
                tree = file["Hits"]
                self.data = tree.arrays(library="np")
                
                print(f"Loaded detector parameters: {self.detector_params}")
                print(f"Loaded {len(self.data['TrueX'])} events")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def plot_single_neighborhood_charge_grid(self, event_idx, charge_type='fraction', save_individual=True, output_dir="", event_type="random"):
        """
        Plot focused neighborhood (9x9) grid for a single hit showing charge distribution
        Only draws pixels and blocks for valid positions (not outside detector bounds)
        
        Parameters:
        -----------
        charge_type : str
            'fraction' for charge fractions, 'value' for absolute charge values (electrons), 
            'coulomb' for charge in Coulombs, 'distance' for distances
        """
        if 'GridNeighborhoodChargeFractions' not in self.data:
            print(f"No neighborhood charge data found for event {event_idx}")
            return None
            
        # Get event data
        hit_x = self.data['TrueX'][event_idx]
        hit_y = self.data['TrueY'][event_idx]
        hit_z = self.data['TrueZ'][event_idx]
        edep = self.data['Edep'][event_idx]
        pixel_x = self.data['PixelX'][event_idx]
        pixel_y = self.data['PixelY'][event_idx]
        pixel_dist = self.data['PixelDist'][event_idx]
        pixel_hit = self.data['PixelHit'][event_idx]
        
        # Get charge data
        charge_fractions = self.data['GridNeighborhoodChargeFractions'][event_idx]
        charge_values = self.data['GridNeighborhoodChargeValues'][event_idx]
        charge_coulombs = self.data['GridNeighborhoodChargeCoulombs'][event_idx] if 'GridNeighborhoodChargeCoulombs' in self.data else None
        charge_distances = self.data['GridNeighborhoodDistances'][event_idx]
        
        # Select which data to display
        if charge_type == 'fraction':
            grid_data = np.array(charge_fractions).reshape(9, 9)
            data_label = 'Charge Fraction'
            data_unit = ''
            value_format = '.4f'
        elif charge_type == 'value':
            grid_data = np.array(charge_values).reshape(9, 9)
            data_label = 'Charge Value'
            data_unit = ' e⁻'
            value_format = '.0f'
        elif charge_type == 'coulomb':
            if charge_coulombs is None:
                raise ValueError("Coulomb charge data not available in this ROOT file")
            grid_data = np.array(charge_coulombs).reshape(9, 9)
            data_label = 'Charge'
            data_unit = ' Coulomb'
            value_format = '.2e'
        elif charge_type == 'distance':
            grid_data = np.array(charge_distances).reshape(9, 9)
            data_label = 'Distance'
            data_unit = ' mm'
            value_format = '.3f'
        else:
            raise ValueError("charge_type must be 'fraction', 'value', 'coulomb', or 'distance'")
        
        # Replace invalid values (-999.0) with NaN for proper display
        grid_data[grid_data == -999.0] = np.nan
        
        # For zero energy events, set all values to zero
        if edep <= 0:
            print(f"  No energy deposited - all charge values should be zero")
            grid_data[:] = 0.0
        
        # Get detector parameters
        pixel_size = self.detector_params['pixel_size']
        pixel_spacing = self.detector_params['pixel_spacing']
        
        # Create figure focused on 9x9 grid area
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Calculate the bounds of the 9x9 grid centered on PixelX, PixelY
        grid_extent = 4.5 * pixel_spacing  # 4.5 pixels in each direction from center
        margin = 0  # No margin around the grid
        
        ax.set_xlim(pixel_x - grid_extent, pixel_x + grid_extent)
        ax.set_ylim(pixel_y - grid_extent, pixel_y + grid_extent)
        ax.set_aspect('equal')
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Calculate valid data range for colormap
        valid_data = grid_data[~np.isnan(grid_data)]
        if len(valid_data) > 0:
            if charge_type == 'fraction':
                # For fractions, use fixed range 0 to max fraction to show relative distribution
                vmin, vmax = 0, np.max(valid_data)
            else:
                vmin, vmax = np.min(valid_data), np.max(valid_data)
        else:
            vmin, vmax = 0, 1
            
        # Create colormap for charge data
        if charge_type == 'distance':
            cmap = plt.cm.plasma_r  # Reverse plasma: closer distances are brighter
        else:
            cmap = plt.cm.plasma  # Plasma: higher charge is brighter
        
        # Track which grid positions have valid data for grid line drawing
        valid_positions = set()
        
        # First, draw the colored blocks (grid cells) behind the pixels - only for valid positions
        for i in range(9):
            for j in range(9):
                # Get data value for this grid position
                data_value = grid_data[i, j]
                
                # Only draw if data is valid (not NaN)
                if not np.isnan(data_value):
                    valid_positions.add((i, j))
                    
                    # Calculate pixel position relative to center (PixelX, PixelY is center)
                    rel_x = (j - 4) * pixel_spacing  # j maps to x direction, offset from center
                    rel_y = (i - 4) * pixel_spacing  # i maps to y direction, offset from center
                    
                    # Block center position (same as pixel center)
                    block_center_x = pixel_x + rel_x
                    block_center_y = pixel_y + rel_y
                    
                    # Calculate block corners for drawing (blocks are pixel_spacing sized)
                    block_corner_x = block_center_x - pixel_spacing/2
                    block_corner_y = block_center_y - pixel_spacing/2
                    
                    # Color block based on charge data value
                    normalized_value = (data_value - vmin) / (vmax - vmin) if vmax > vmin else 0
                    color = cmap(normalized_value)
                    
                    # Draw colored block (grid cell)
                    block_rect = patches.Rectangle((block_corner_x, block_corner_y), 
                                                 pixel_spacing, pixel_spacing,
                                                 linewidth=0.5, edgecolor='black', 
                                                 facecolor=color, alpha=0.7)
                    ax.add_patch(block_rect)
                    
                    # Position text below the pixel within the block
                    text_y = block_center_y - pixel_size/2 - 0.08  # Further below the pixel
                    
                    # Add data value text in the block below the pixel
                    if charge_type == 'fraction':
                        display_text = f'{data_value:.3f}'
                    elif charge_type == 'value':
                        if data_value >= 1000:
                            display_text = f'{data_value:.0f}'
                        else:
                            display_text = f'{data_value:.1f}'
                    elif charge_type == 'coulomb':
                        display_text = f'{data_value:.2e}'
                    else:  # distance
                        display_text = f'{data_value:.3f}'
                    
                    ax.text(block_center_x, text_y, display_text,
                           ha='center', va='center', fontsize=9, 
                           color='white', weight='bold')
        
        # Now draw the pixels on top with uniform fill color - only for valid positions
        for i in range(9):
            for j in range(9):
                # Only draw pixel if this position has valid data
                if (i, j) in valid_positions:
                    # Calculate pixel position relative to center (PixelX, PixelY is center)
                    rel_x = (j - 4) * pixel_spacing  # j maps to x direction, offset from center
                    rel_y = (i - 4) * pixel_spacing  # i maps to y direction, offset from center
                    
                    # Actual pixel center position
                    pixel_center_x = pixel_x + rel_x
                    pixel_center_y = pixel_y + rel_y
                    
                    # Calculate pixel corners for drawing
                    pixel_corner_x = pixel_center_x - pixel_size/2
                    pixel_corner_y = pixel_center_y - pixel_size/2
                    
                    # Draw pixel with uniform fill and edge color
                    pixel_rect = patches.Rectangle((pixel_corner_x, pixel_corner_y), 
                                                 pixel_size, pixel_size,
                                                 linewidth=1.5, edgecolor='black', 
                                                 facecolor='black', alpha=1.0)
                    ax.add_patch(pixel_rect)
        
        # Mark the actual hit position
        ax.plot(hit_x, hit_y, 'ro', markersize=8, markeredgewidth=4, 
                label=f'Hit Position ({hit_x:.2f}, {hit_y:.2f})')
        
        # Draw grid lines only where they separate existing blocks
        # Draw borders around each valid block
        for i, j in valid_positions:
            # Calculate block position relative to center
            rel_x = (j - 4) * pixel_spacing
            rel_y = (i - 4) * pixel_spacing
            
            # Block center position
            block_center_x = pixel_x + rel_x
            block_center_y = pixel_y + rel_y
            
            # Calculate block boundaries
            left = block_center_x - pixel_spacing/2
            right = block_center_x + pixel_spacing/2
            bottom = block_center_y - pixel_spacing/2
            top = block_center_y + pixel_spacing/2
            
            # Draw left border if no valid block to the left
            if (i, j-1) not in valid_positions:
                ax.plot([left, left], [bottom, top], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw right border if no valid block to the right
            if (i, j+1) not in valid_positions:
                ax.plot([right, right], [bottom, top], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw bottom border if no valid block below
            if (i+1, j) not in valid_positions:
                ax.plot([left, right], [bottom, bottom], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw top border if no valid block above
            if (i-1, j) not in valid_positions:
                ax.plot([left, right], [top, top], color='black', linewidth=0.8, alpha=0.9)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=f'{data_label}{data_unit}')
        cbar.ax.tick_params(labelsize=10)
        
        # Add concise scientific title
        hit_type = "Inside Pixel" if pixel_hit else "Outside Pixel"
        
        # Count invalid positions for edge cases
        invalid_positions = np.sum(np.array(charge_fractions) == -999.0) if not pixel_hit else 0
        
        if event_type == "edge-case" and invalid_positions > 0:
            title = f"Grid Charge Sharing Analysis: Event {event_idx} (Edge Case)\n" \
                    f"Hit: ({hit_x:.2f}, {hit_y:.2f}) mm, Edep: {edep:.3f} MeV, {len(valid_positions)}/81 positions inside detector"
        elif event_type == "inside-pixel":
            title = f"Grid Charge Sharing Analysis: Event {event_idx} (Inside Pixel)\n" \
                    f"Hit: ({hit_x:.2f}, {hit_y:.2f}) mm, Edep: {edep:.3f} MeV, All charge assigned to hit pixel"
        else:
            title = f"Grid Charge Sharing Analysis: Event {event_idx} (Random)\n" \
                    f"Hit: ({hit_x:.2f}, {hit_y:.2f}) mm, Edep: {edep:.3f} MeV, Distance: {pixel_dist:.3f} mm"
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Save individual plot if requested
        if save_individual:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'event_{event_idx}_9x9_charge_{charge_type}_{timestamp}.png'
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Event {event_idx} charge {charge_type} plot saved to {filename}")
        
        #plt.show()
        
        # Print statistics for this event
        if len(valid_data) > 0:
            if charge_type == 'fraction':
                total_fraction = np.sum(valid_data)
                print(f"  Event {event_idx}: {len(valid_data)} valid pixels, "
                      f"range {np.min(valid_data):.4f} to {np.max(valid_data):.4f}, "
                      f"total fraction: {total_fraction:.6f}, center: {grid_data[4, 4]:.4f}")
            elif charge_type == 'value':
                total_charge = np.sum(valid_data)
                print(f"  Event {event_idx}: {len(valid_data)} valid pixels, "
                      f"range {np.min(valid_data):.0f} to {np.max(valid_data):.0f} e⁻, "
                      f"total charge: {total_charge:.0f} e⁻, center: {grid_data[4, 4]:.0f} e⁻")
            elif charge_type == 'coulomb':
                total_charge = np.sum(valid_data)
                print(f"  Event {event_idx}: {len(valid_data)} valid pixels, "
                      f"range {np.min(valid_data):.2e} to {np.max(valid_data):.2e} C, "
                      f"total charge: {total_charge:.2e} C, center: {grid_data[4, 4]:.2e} C")
            else:  # distance
                print(f"  Event {event_idx}: {len(valid_data)} valid pixels, "
                      f"distance range {np.min(valid_data):.3f} to {np.max(valid_data):.3f} mm, "
                      f"center distance: {grid_data[4, 4]:.3f} mm")
        
        return fig, ax, grid_data
    
    def plot_mean_neighborhood_charge_grid(self, charge_type='fraction', save_plot=True, output_dir=""):
        """Plot mean neighborhood (9x9) charge grid averaged across all non-inside-pixel events
        Only draws pixels and blocks for positions that have valid data across events"""
        if 'GridNeighborhoodChargeFractions' not in self.data:
            print("No neighborhood charge data found for mean calculation")
            return None
            
        # Find all non-inside-pixel events with energy deposition
        pixel_hit = self.data['PixelHit']
        edep = self.data['Edep']
        valid_mask = (~pixel_hit) & (edep > 0)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("No valid events found for mean calculation")
            return None
            
        print(f"Calculating mean charge {charge_type} across {len(valid_indices)} valid events")
        
        # Select which data to average
        if charge_type == 'fraction':
            data_key = 'GridNeighborhoodChargeFractions'
            data_label = 'Mean Charge Fraction'
            data_unit = ''
        elif charge_type == 'value':
            data_key = 'GridNeighborhoodChargeValues'
            data_label = 'Mean Charge Value'
            data_unit = ' e⁻'
        elif charge_type == 'coulomb':
            if 'GridNeighborhoodChargeCoulombs' not in self.data:
                raise ValueError("Coulomb charge data not available in this ROOT file")
            data_key = 'GridNeighborhoodChargeCoulombs'
            data_label = 'Mean Charge'
            data_unit = ' C'
        elif charge_type == 'distance':
            data_key = 'GridNeighborhoodDistances'
            data_label = 'Mean Distance'
            data_unit = ' mm'
        else:
            raise ValueError("charge_type must be 'fraction', 'value', 'coulomb', or 'distance'")
        
        # Collect all grids for valid events
        all_grids = []
        valid_event_count = 0
        
        for event_idx in valid_indices:
            grid_data = self.data[data_key][event_idx]
            grid_array = np.array(grid_data).reshape(9, 9)
            
            # Replace invalid values (-999.0) with NaN
            grid_array[grid_array == -999.0] = np.nan
            
            # Skip events that have all NaN values
            if not np.all(np.isnan(grid_array)):
                all_grids.append(grid_array)
                valid_event_count += 1
        
        if len(all_grids) == 0:
            print("No valid charge data found for mean calculation")
            return None
            
        print(f"Using {valid_event_count} events with valid charge data for mean calculation")
        
        # Calculate mean, ignoring NaN values
        grids_array = np.array(all_grids)
        mean_grid = np.nanmean(grids_array, axis=0)
        
        # Calculate some representative positions for the plot
        # Use the first valid event for positioning reference
        ref_event_idx = valid_indices[0]
        pixel_x = self.data['PixelX'][ref_event_idx]
        pixel_y = self.data['PixelY'][ref_event_idx]
        
        # Get detector parameters
        pixel_size = self.detector_params['pixel_size']
        pixel_spacing = self.detector_params['pixel_spacing']
        
        # Create figure focused on neighborhood (9x9) grid area
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Calculate the bounds of the 9x9 grid centered on reference position
        grid_extent = 4.5 * pixel_spacing
        
        ax.set_xlim(pixel_x - grid_extent, pixel_x + grid_extent)
        ax.set_ylim(pixel_y - grid_extent, pixel_y + grid_extent)
        ax.set_aspect('equal')
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Calculate valid data range for colormap
        valid_mean_data = mean_grid[~np.isnan(mean_grid)]
        if len(valid_mean_data) > 0:
            if charge_type == 'fraction':
                vmin, vmax = 0, np.max(valid_mean_data)
            else:
                vmin, vmax = np.min(valid_mean_data), np.max(valid_mean_data)
        else:
            vmin, vmax = 0, 1
            
        # Create colormap for charge data
        if charge_type == 'distance':
            cmap = plt.cm.plasma_r  # Reverse plasma: closer distances are brighter
        else:
            cmap = plt.cm.plasma  # Plasma: higher charge is brighter
        
        # Track which grid positions have valid mean data for grid line drawing
        valid_positions = set()
        
        # Draw the colored blocks (grid cells) behind the pixels - only for valid positions
        for i in range(9):
            for j in range(9):
                # Get mean data value for this grid position
                mean_value = mean_grid[i, j]
                
                # Only draw if mean data is valid (not NaN)
                if not np.isnan(mean_value):
                    valid_positions.add((i, j))
                    
                    # Calculate pixel position relative to center
                    rel_x = (j - 4) * pixel_spacing
                    rel_y = (i - 4) * pixel_spacing
                    
                    # Block center position
                    block_center_x = pixel_x + rel_x
                    block_center_y = pixel_y + rel_y
                    
                    # Calculate block corners for drawing
                    block_corner_x = block_center_x - pixel_spacing/2
                    block_corner_y = block_center_y - pixel_spacing/2
                    
                    # Color block based on mean data value
                    normalized_value = (mean_value - vmin) / (vmax - vmin) if vmax > vmin else 0
                    color = cmap(normalized_value)
                    
                    # Draw colored block (grid cell)
                    block_rect = patches.Rectangle((block_corner_x, block_corner_y), 
                                                 pixel_spacing, pixel_spacing,
                                                 linewidth=0.5, edgecolor='black', 
                                                 facecolor=color, alpha=0.7)
                    ax.add_patch(block_rect)
                    
                    # Position text below the pixel within the block
                    text_y = block_center_y - pixel_size/2 - 0.08
                    
                    # Add mean value text in the block below the pixel
                    if charge_type == 'fraction':
                        display_text = f'{mean_value:.3f}'
                    elif charge_type == 'value':
                        if mean_value >= 1000:
                            display_text = f'{mean_value:.0f}'
                        else:
                            display_text = f'{mean_value:.1f}'
                    elif charge_type == 'coulomb':
                        display_text = f'{mean_value:.2e}'
                    else:  # distance
                        display_text = f'{mean_value:.3f}'
                    
                    ax.text(block_center_x, text_y, display_text,
                           ha='center', va='center', fontsize=9, 
                           color='white', weight='bold')
        
        # Draw the pixels on top with uniform fill color - only for valid positions
        for i in range(9):
            for j in range(9):
                # Only draw pixel if this position has valid mean data
                if (i, j) in valid_positions:
                    # Calculate pixel position relative to center
                    rel_x = (j - 4) * pixel_spacing
                    rel_y = (i - 4) * pixel_spacing
                    
                    # Actual pixel center position
                    pixel_center_x = pixel_x + rel_x
                    pixel_center_y = pixel_y + rel_y
                    
                    # Calculate pixel corners for drawing
                    pixel_corner_x = pixel_center_x - pixel_size/2
                    pixel_corner_y = pixel_center_y - pixel_size/2
                    
                    # Draw pixel with uniform fill and edge color
                    pixel_rect = patches.Rectangle((pixel_corner_x, pixel_corner_y), 
                                                 pixel_size, pixel_size,
                                                 linewidth=1.5, edgecolor='black', 
                                                 facecolor='black', alpha=1.0)
                    ax.add_patch(pixel_rect)
        
        # Draw grid lines only where they separate existing blocks
        # Draw borders around each valid block
        for i, j in valid_positions:
            # Calculate block position relative to center
            rel_x = (j - 4) * pixel_spacing
            rel_y = (i - 4) * pixel_spacing
            
            # Block center position
            block_center_x = pixel_x + rel_x
            block_center_y = pixel_y + rel_y
            
            # Calculate block boundaries
            left = block_center_x - pixel_spacing/2
            right = block_center_x + pixel_spacing/2
            bottom = block_center_y - pixel_spacing/2
            top = block_center_y + pixel_spacing/2
            
            # Draw left border if no valid block to the left
            if (i, j-1) not in valid_positions:
                ax.plot([left, left], [bottom, top], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw right border if no valid block to the right
            if (i, j+1) not in valid_positions:
                ax.plot([right, right], [bottom, top], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw bottom border if no valid block below
            if (i+1, j) not in valid_positions:
                ax.plot([left, right], [bottom, bottom], color='black', linewidth=0.8, alpha=0.9)
            
            # Draw top border if no valid block above
            if (i-1, j) not in valid_positions:
                ax.plot([left, right], [top, top], color='black', linewidth=0.8, alpha=0.9)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=f'{data_label}{data_unit}')
        cbar.ax.tick_params(labelsize=10)
        
        # Add title to distinguish from individual plots
        title = f'Mean Charge Sharing Distribution: Grid Analysis ({charge_type.title()})\n' \
                f'Ensemble Average (N = {valid_event_count} events, excluding inside-pixel hits)'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'mean_neighborhood_charge_{charge_type}_{valid_event_count}_events_{timestamp}.png'
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Mean charge {charge_type} grid plot saved to {filename}")
        
        #plt.show()
        
        # Print statistics
        if len(valid_mean_data) > 0:
            if charge_type == 'fraction':
                print(f"  Mean grid: {len(valid_mean_data)} valid positions, "
                      f"range {np.min(valid_mean_data):.4f} to {np.max(valid_mean_data):.4f}, "
                      f"center mean: {mean_grid[4, 4]:.4f}")
            elif charge_type == 'value':
                print(f"  Mean grid: {len(valid_mean_data)} valid positions, "
                      f"range {np.min(valid_mean_data):.0f} to {np.max(valid_mean_data):.0f} e⁻, "
                      f"center mean: {mean_grid[4, 4]:.0f} e⁻")
            elif charge_type == 'coulomb':
                print(f"  Mean grid: {len(valid_mean_data)} valid positions, "
                      f"range {np.min(valid_mean_data):.2e} to {np.max(valid_mean_data):.2e} C, "
                      f"center mean: {mean_grid[4, 4]:.2e} C")
            else:  # distance
                print(f"  Mean grid: {len(valid_mean_data)} valid positions, "
                      f"range {np.min(valid_mean_data):.3f} to {np.max(valid_mean_data):.3f} mm, "
                      f"center mean: {mean_grid[4, 4]:.3f} mm")
        
        return fig, ax, mean_grid, valid_event_count
    
    def generate_random_hits_charge_individual(self, num_events=3, charge_type='fraction', save_plots=True, output_dir="", seed=None, include_mean=True, include_edge_case=True):
        """Create individual neighborhood (9x9) charge grid visualizations for N random hits and optionally a mean plot"""
        if 'GridNeighborhoodChargeFractions' not in self.data:
            print("No neighborhood charge data found")
            return None
            
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Find events with energy deposition for charge sharing
        edep = self.data['Edep']
        energy_mask = edep > 0
        energy_indices = np.where(energy_mask)[0]
        
        if len(energy_indices) == 0:
            print("No events with energy deposition found for charge sharing analysis")
            return None
            
        # Select N random events from those with energy deposition
        if num_events > len(energy_indices):
            print(f"Requested {num_events} events, but only {len(energy_indices)} have energy deposition. Using all available.")
            num_events = len(energy_indices)
            
        random_events = random.sample(list(energy_indices), num_events)
        random_events.sort()  # Sort for easier reference
        
        print(f"Selected {num_events} random events with energy deposition: {random_events}")
        
        # Find an event where hit was inside pixel or closest to it
        inside_pixel_event = self.find_inside_pixel_event()
        
        # Find an edge case event where neighborhood (9x9) grid is incomplete
        edge_case_event = None
        if include_edge_case:
            edge_case_event = self.find_edge_case_event()
        
        # Combine all events, avoiding duplicates
        all_events = random_events.copy()
        
        if inside_pixel_event is not None and inside_pixel_event not in all_events:
            print(f"Adding inside-pixel event: {inside_pixel_event}")
            all_events.append(inside_pixel_event)
            
        if edge_case_event is not None and edge_case_event not in all_events:
            print(f"Adding edge case event: {edge_case_event}")
            all_events.append(edge_case_event)
            
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Process each event individually
        results = []
        for i, event_idx in enumerate(all_events):
            event_type = "random"
            if event_idx == inside_pixel_event:
                event_type = "inside-pixel"
            elif event_idx == edge_case_event:
                event_type = "edge-case"
                
            print(f"\nProcessing {event_type} event {event_idx} ({i+1}/{len(all_events)})...")
            fig, ax, charge_grid = self.plot_single_neighborhood_charge_grid(event_idx, 
                                                                   charge_type=charge_type,
                                                                   save_individual=save_plots, 
                                                                   output_dir=output_dir,
                                                                   event_type=event_type)
            results.append((event_idx, fig, ax, charge_grid))
        
        # Generate mean plot if requested
        if include_mean:
            print(f"\nGenerating mean charge {charge_type} plot...")
            fig, ax, mean_charge_grid, valid_event_count = self.plot_mean_neighborhood_charge_grid(charge_type=charge_type,
                                                                                         save_plot=save_plots, 
                                                                                         output_dir=output_dir)
            results.append((-1, fig, ax, mean_charge_grid))
        
        print(f"\n" + "="*60)
        print(f"CHARGE SHARING PROCESSING COMPLETE")
        print(f"="*60)
        print(f"Generated {len(all_events)} individual neighborhood (9x9) charge {charge_type} visualizations")
        print(f"Random events: {random_events}")
        if inside_pixel_event is not None:
            print(f"Inside-pixel/closest event: {inside_pixel_event}")
        if edge_case_event is not None:
            print(f"Edge case event: {edge_case_event}")
        if include_mean:
            print(f"Mean plot: averaged across all valid events")
        if save_plots:
            if output_dir:
                print(f"Plots saved to directory: {output_dir}")
            else:
                print(f"Plots saved to current directory")
        
        return results, all_events
    
    def find_edge_case_event(self):
        """Find an event where the hit is near the detector edge so neighborhood (9x9) grid is incomplete"""
        if 'GridNeighborhoodChargeFractions' not in self.data:
            print("GridNeighborhoodChargeFractions data not available for edge case search")
            return None
            
        # Get detector parameters
        detector_size = self.detector_params['detector_size']
        pixel_spacing = self.detector_params['pixel_spacing']
        
        # Calculate how close to edge a hit needs to be for incomplete neighborhood (9x9) grid
        grid_half_extent = 4 * pixel_spacing  # 4 pixels from center to edge of neighborhood grid
        edge_threshold = detector_size/2 - grid_half_extent
        
        print(f"Detector size: {detector_size} mm")
        print(f"Grid half extent: {grid_half_extent} mm") 
        print(f"Edge threshold: {edge_threshold} mm from detector center")
        
        # Find events where hit is close to any edge and has energy deposition
        hit_x = self.data['TrueX']
        hit_y = self.data['TrueY']
        edep = self.data['Edep']
        
        # Calculate distance from detector center and edges
        dist_from_center = np.sqrt(hit_x**2 + hit_y**2)
        
        # Check which events have hits near edges with energy deposition
        near_edge_mask = (dist_from_center > edge_threshold) & (edep > 0)
        near_edge_indices = np.where(near_edge_mask)[0]
        
        if len(near_edge_indices) == 0:
            print("No events found with hits near detector edges that have energy deposition")
            # If no clear edge cases, find the event closest to the edge with energy
            energy_mask = edep > 0
            energy_indices = np.where(energy_mask)[0]
            if len(energy_indices) > 0:
                farthest_idx = energy_indices[np.argmax(dist_from_center[energy_indices])]
                farthest_distance = dist_from_center[farthest_idx]
                print(f"Using event {farthest_idx} with maximum distance from center: {farthest_distance:.3f} mm")
                return int(farthest_idx)
            else:
                return None
        
        # From near-edge events, find one that actually has incomplete grid data
        best_edge_event = None
        max_incomplete_count = 0
        
        for event_idx in near_edge_indices:
            charge_fractions = self.data['GridNeighborhoodChargeFractions'][event_idx]
            charge_grid = np.array(charge_fractions).reshape(9, 9)
            
            # Count incomplete/invalid positions (should be -999.0 for out-of-bounds)
            incomplete_count = np.sum(charge_grid == -999.0)
            
            if incomplete_count > max_incomplete_count:
                max_incomplete_count = incomplete_count
                best_edge_event = event_idx
                
        if best_edge_event is not None:
            hit_x_val = hit_x[best_edge_event]
            hit_y_val = hit_y[best_edge_event]
            distance = dist_from_center[best_edge_event]
            print(f"Found edge case event {best_edge_event} at ({hit_x_val:.3f}, {hit_y_val:.3f}) mm")
            print(f"Distance from center: {distance:.3f} mm, incomplete positions: {max_incomplete_count}/81")
            return int(best_edge_event)
        else:
            # Fallback to the first near-edge event
            event_idx = near_edge_indices[0]
            hit_x_val = hit_x[event_idx]
            hit_y_val = hit_y[event_idx]
            distance = dist_from_center[event_idx]
            print(f"Using near-edge event {event_idx} at ({hit_x_val:.3f}, {hit_y_val:.3f}) mm")
            print(f"Distance from center: {distance:.3f} mm")
            return int(event_idx)

    def find_inside_pixel_event(self):
        """Find an event where the hit was inside a pixel, or the closest one with energy deposition"""
        if 'PixelHit' not in self.data or 'PixelDist' not in self.data:
            print("PixelHit or PixelDist data not available")
            return None
            
        pixel_hit = self.data['PixelHit']
        pixel_dist = self.data['PixelDist']
        edep = self.data['Edep']
        
        # First, check if there are any events where hit is inside pixel with energy deposition
        inside_pixel_mask = pixel_hit & (edep > 0)
        inside_pixel_indices = np.where(inside_pixel_mask)[0]
        
        if len(inside_pixel_indices) > 0:
            # Use the first inside-pixel event with energy
            event_idx = inside_pixel_indices[0]
            print(f"Found {len(inside_pixel_indices)} events with hits inside pixels and energy deposition")
            print(f"Using event {event_idx} (hit inside pixel)")
            return int(event_idx)
        else:
            # Find the event with minimum distance that has energy deposition
            energy_mask = edep > 0
            energy_indices = np.where(energy_mask)[0]
            
            if len(energy_indices) == 0:
                print("No events with energy deposition found")
                return None
                
            min_dist_idx = energy_indices[np.argmin(pixel_dist[energy_indices])]
            min_distance = pixel_dist[min_dist_idx]
            print(f"No events with hits inside pixels found")
            print(f"Using event {min_dist_idx} with minimum distance and energy deposition: {min_distance:.4f} mm")
            return int(min_dist_idx)

def create_random_charge_plots(root_filename, num_events=1, charge_type='fraction', output_dir="", seed=42, include_mean=True, include_edge_case=True):
    """
    Convenience function to create N random charge sharing plots from ROOT file
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file containing neighborhood (9x9) charge data
    num_events : int, optional
        Number of random events to visualize (default: 1)
    charge_type : str, optional
        Type of charge data to display: 'fraction', 'value', or 'distance' (default: 'fraction')
    output_dir : str, optional
        Directory to save plots (default: current directory)
    seed : int, optional
        Random seed for reproducible selection of events
    include_mean : bool, optional
        Whether to include a mean plot of all valid events (default: True)
    include_edge_case : bool, optional
        Whether to include an edge case where neighborhood (9x9) grid is incomplete (default: True)
    
    Returns:
    --------
    tuple : (results, selected_events)
    """
    generator = RandomHitChargeGridGenerator(root_filename)
    return generator.generate_random_hits_charge_individual(num_events=num_events,
                                                           charge_type=charge_type,
                                                           save_plots=True, 
                                                           output_dir=output_dir, 
                                                           seed=seed,
                                                           include_mean=include_mean,
                                                           include_edge_case=include_edge_case)

if __name__ == "__main__":
    # Configuration parameters
    NUM_EVENTS = 3  # Change this to control number of random events to visualize
    CHARGE_TYPES = ['fraction', 'coulomb']  # Generate both fraction and coulomb plots
    OUTPUT_DIR = "neighborhood_charge_plots"  # Directory to save plots (empty string for current directory)
    RANDOM_SEED = 2  # For reproducible results
    INCLUDE_MEAN = True  # Whether to include mean plot of all valid events
    INCLUDE_EDGE_CASE = True  # Whether to include edge case where neighborhood (9x9) grid is incomplete
    
    # Default ROOT file (can be changed)
    root_file = "epicToyOutput.root"
    
    # Check if file exists
    if not os.path.exists(root_file):
        print(f"Error: ROOT file '{root_file}' not found!")
        print("Make sure you have run the Geant4 simulation first.")
        exit(1)
    
    print("="*60)
    print(f"GENERATING {NUM_EVENTS} RANDOM HITS NEIGHBORHOOD (9x9) CHARGE GRIDS")
    print(f"CHARGE TYPES: {', '.join([ct.upper() for ct in CHARGE_TYPES])}")
    if INCLUDE_MEAN:
        print(f"+ MEAN CHARGE GRIDS OF ALL VALID EVENTS")
    if INCLUDE_EDGE_CASE:
        print("+ EDGE CASE WHERE NEIGHBORHOOD (9x9) GRID IS INCOMPLETE")
    print("="*60)
    
    try:
        # Generate visualizations for each charge type
        for charge_type in CHARGE_TYPES:
            print(f"\n{'='*40}")
            print(f"PROCESSING CHARGE TYPE: {charge_type.upper()}")
            print(f"{'='*40}")
            
            # Generate the visualization
            results, selected_events = create_random_charge_plots(root_file, 
                                                                num_events=NUM_EVENTS,
                                                                charge_type=charge_type,
                                                                output_dir=OUTPUT_DIR,
                                                                seed=RANDOM_SEED,
                                                                include_mean=INCLUDE_MEAN,
                                                                include_edge_case=INCLUDE_EDGE_CASE)
            
            print(f"\nCompleted {charge_type} plots for events: {selected_events}")
        
        print("\n" + "="*60)
        print("CHARGE SHARING VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Generated plots for {len(CHARGE_TYPES)} charge types: {', '.join(CHARGE_TYPES)}")
        print(f"Each type includes {NUM_EVENTS} individual neighborhood (9x9) charge plots")
        if INCLUDE_MEAN:
            print(f"Each type includes 1 mean neighborhood (9x9) charge plot averaged across all valid events")
        if INCLUDE_EDGE_CASE:
            print("Each type includes 1 edge case neighborhood (9x9) charge plot where grid extends beyond detector")
        print("Each event is displayed as a separate plot focused on its neighborhood (9x9) grid area.")
        print(f"All plots saved to directory: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback
        traceback.print_exc() 