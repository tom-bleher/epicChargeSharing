#!/usr/bin/env python3
"""
Visualization of 3D Gaussian fitting results from ROOT file.
Reads pre-computed Gaussian fit parameters from Geant4 simulation and creates plots.

UPDATED FOR NEW SEPARATED DATA STRUCTURE:
- Uses new NonPixel_ prefixed branch names for non-pixel hit data
- Uses new classification variables: IsPixelHit, IsWithinD0
- Updated to work with separated pixel/non-pixel data structure
- Gaussian fitting is only available for non-pixel hits (distance > D0)

FIXED TO MATCH ACTUAL C++ IMPLEMENTATION:
- Removed all references to "outlier removal" fits (not implemented in C++)
- Removed all references to R² (not calculated in C++, only Chi²/NDF available)
- Removed dual fit comparison functionality
- Updated to work with single MinuitGaussianFitter results only
- Fixed fit statistics to use actual ROOT branch names
- Simplified analysis to match available data

Recent updates:
- Updated for new separated pixel/non-pixel data structure
- Fixed to match actual C++ implementation (single fit only, no outliers, Chi²/NDF instead of R²)
- Added dynamic units for charge and residuals based on charge_type
- Units: 'fraction' → (fraction), 'value' → (keV), 'coulomb' → (C)
- Updated all colorbar labels and axis labels to include appropriate units
"""

# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import uproot
import os
from charge_sim import RandomHitChargeGridGenerator

class GaussianRootPlotter:
    """
    Create visualizations from pre-computed 3D Gaussian fit results stored in ROOT file
    """
    
    def __init__(self):
        pass
        
    def get_charge_units(self, charge_type):
        """
        Get appropriate units for charge and residuals based on charge type
        
        Parameters:
        -----------
        charge_type : str
            'fraction', 'value', or 'coulomb'
            
        Returns:
        --------
        charge_unit : str
            Unit string for charge
        residual_unit : str  
            Unit string for residuals (same as charge)
        """
        if charge_type == 'fraction':
            return '(fraction)', '(fraction)'
        elif charge_type == 'value':
            return '(keV)', '(keV)'
        elif charge_type == 'coulomb':
            return '(C)', '(C)'
        else:
            return '', ''
        
    def gaussian_3d(self, params, coords):
        """
        3D Gaussian function
        
        Parameters:
        -----------
        params : array-like
            [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
        coords : tuple
            (x, y) coordinate arrays
            
        Returns:
        --------
        z : array
            3D Gaussian values at given coordinates
        """
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = params
        x, y = coords
        
        # Rotation transformation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_rot = cos_theta * (x - x0) + sin_theta * (y - y0)
        y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0)
        
        # 3D Gaussian
        z = amplitude * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))) + offset
        
        return z

    def load_root_data(self, root_filename):
        """
        Load data from ROOT file using uproot
        """
        try:
            with uproot.open(root_filename) as file:
                tree = file["Hits"]
                
                # Load all relevant data
                data = {
                    # Position data
                    'TrueX': tree['TrueX'].array(library="np"),
                    'TrueY': tree['TrueY'].array(library="np"),
                    'PixelX': tree['PixelX'].array(library="np"),
                    'PixelY': tree['PixelY'].array(library="np"),
                    'PixelTrueDistance': tree['PixelTrueDistance'].array(library="np"),
                    
                    # NEW: Hit classification data
                    'IsPixelHit': tree['IsPixelHit'].array(library="np"),
                    'IsWithinD0': tree['IsWithinD0'].array(library="np"),
                    
                    # NEW: Pixel hit data (only meaningful for pixel hits)
                    'PixelHit_PixelAlpha': tree['PixelHit_PixelAlpha'].array(library="np"),
                    
                    # NEW: Non-pixel hit charge data (with NonPixel_ prefix)
                    'NonPixel_GridNeighborhoodChargeFractions': tree['NonPixel_GridNeighborhoodChargeFractions'].array(library="np"),
                    'NonPixel_GridNeighborhoodDistances': tree['NonPixel_GridNeighborhoodDistances'].array(library="np"),
                    'NonPixel_GridNeighborhoodCharge': tree['NonPixel_GridNeighborhoodCharge'].array(library="np"),
                    
                    # Event data
                    'EventID': tree['EventID'].array(library="np"),
                }
                
                # Load fit results - only single fit type available
                fit_branches = [
                    'FitAmplitude', 'FitX0', 'FitY0', 'FitSigmaX', 'FitSigmaY', 'FitTheta', 'FitOffset',
                    'FitAmplitudeErr', 'FitX0Err', 'FitY0Err', 'FitSigmaXErr', 'FitSigmaYErr', 'FitThetaErr', 'FitOffsetErr',
                    'FitChi2red', 'FitNDF', 'FitPp', 'FitNPoints', 'FitConstraintsSatisfied', 'FitResidualMean', 'FitResidualStd'
                ]
                
                fit_data_available = True
                missing_branches = []
                
                for branch in fit_branches:
                    try:
                        data[branch] = tree[branch].array(library="np")
                    except:
                        missing_branches.append(branch)
                        fit_data_available = False
                
                if not fit_data_available:
                    print(f"WARNING: Fit result branches not found in ROOT file: {missing_branches}")
                    print("This likely means Gaussian fitting was not enabled in the C++ simulation.")
                    print("Available branches:", list(tree.keys()))
                    print("\nTo enable Gaussian fitting:")
                    print("1. Make sure the C++ code is compiled with Minuit support")
                    print("2. Check that the EventAction is calling the Gaussian fitting")
                    print("3. Ensure sufficient energy deposition to trigger fitting")
                    
                    # Return data without fit results for basic visualization
                    for branch in fit_branches:
                        if branch not in data:
                            if 'Err' in branch or branch in ['FitNDF', 'FitPp', 'FitNPoints']:
                                data[branch] = np.zeros(len(data['EventID']))
                            elif branch == 'FitConstraintsSatisfied':
                                data[branch] = np.zeros(len(data['EventID']), dtype=bool)
                            else:
                                data[branch] = np.zeros(len(data['EventID']))
                
                # Load detector parameters from metadata
                detector_params = {}
                try:
                    pixel_size_obj = file["GridPixelSize"]
                    pixel_spacing_obj = file["GridPixelSpacing"]
                    detector_params['pixel_size'] = float(pixel_size_obj.title)
                    detector_params['pixel_spacing'] = float(pixel_spacing_obj.title)
                except:
                    print("Warning: Could not load detector parameters from ROOT file")
                    detector_params['pixel_size'] = 0.055  # Default values
                    detector_params['pixel_spacing'] = 0.055
                
                return data, detector_params
                
        except Exception as e:
            print(f"Error loading ROOT file: {e}")
            raise

    def get_charge_coordinates(self, event_idx, data, detector_params, charge_type='fraction'):
        """
        Extract charge coordinates for visualization using the same method as fit_gaussian.py
        """
        # Use the same extraction method as fit_gaussian.py
        from charge_sim import RandomHitChargeGridGenerator
        
        # Create a temporary filename - assume data came from the same ROOT file
        # This is a bit of a hack, but we need to recreate the same data extraction
        # For now, we'll fall back to the original method but match the processing exactly
        
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        pixel_spacing = detector_params['pixel_spacing']
        
        # Get charge data - updated to use new branch names with NonPixel_ prefix
        if charge_type == 'fraction':
            charge_data = data['NonPixel_GridNeighborhoodChargeFractions'][event_idx]
        elif charge_type == 'value':
            charge_data = data['NonPixel_GridNeighborhoodCharge'][event_idx]
        elif charge_type == 'coulomb':
            if 'NonPixel_GridNeighborhoodCharge' in data:
                charge_data = data['NonPixel_GridNeighborhoodCharge'][event_idx]
            else:
                raise ValueError("Coulomb charge data not available")
        else:
            raise ValueError("charge_type must be 'fraction', 'value', or 'coulomb'")
        
        # Reshape to 9x9 grid - exactly as in fit_gaussian.py
        grid_data = np.array(charge_data).reshape(9, 9)
        
        # Replace invalid values (-999.0) with NaN - exactly as in fit_gaussian.py
        grid_data[grid_data == -999.0] = np.nan
        
        # Create coordinate arrays - exactly as in fit_gaussian.py
        x_coords = []
        y_coords = []
        z_values = []
        
        for i in range(9):
            for j in range(9):
                if not np.isnan(grid_data[i, j]):
                    # Calculate actual position relative to pixel center - exactly as in fit_gaussian.py
                    rel_x = (j - 4) * pixel_spacing
                    rel_y = (i - 4) * pixel_spacing
                    
                    actual_x = pixel_x + rel_x
                    actual_y = pixel_y + rel_y
                    
                    x_coords.append(actual_x)
                    y_coords.append(actual_y)
                    z_values.append(grid_data[i, j])
        
        return np.array(x_coords), np.array(y_coords), np.array(z_values)

    def calculate_residuals(self, event_idx, data, detector_params, charge_type='fraction'):
        """
        Calculate residuals from stored fit parameters
        """
        # Get charge coordinates
        x, y, z = self.get_charge_coordinates(event_idx, data, detector_params, charge_type)
        
        # Get fit parameters
        fit_params = [
            data['FitAmplitude'][event_idx],
            data['FitX0'][event_idx],
            data['FitY0'][event_idx],
            data['FitSigmaX'][event_idx],
            data['FitSigmaY'][event_idx],
            data['FitTheta'][event_idx],
            data['FitOffset'][event_idx]
        ]
        
        # Calculate fitted values
        fitted_z = self.gaussian_3d(fit_params, (x, y))
        residuals = z - fitted_z
        
        return x, y, z, fitted_z, residuals

    def plot_simple_fit_results(self, event_idx, data, detector_params, charge_type='fraction', 
                               save_plot=True, output_dir="", root_filename=None):
        """
        Create simple 2-panel plot: Data+Fit and Residuals
        """
        # Check if fit was successful
        if not data['FitConstraintsSatisfied'][event_idx]:
            print(f"Warning: Fit was not successful for event {event_idx}")
            # Continue plotting even if ROOT fit failed
            # Data and generator-based extraction will still work
            fallback_used = True  # mark as fallback to adjust title later
        
        # Get fit parameters from ROOT
        root_fit_params = [
            data['FitAmplitude'][event_idx],
            data['FitX0'][event_idx],
            data['FitY0'][event_idx],
            data['FitSigmaX'][event_idx],
            data['FitSigmaY'][event_idx],
            data['FitTheta'][event_idx],
            data['FitOffset'][event_idx]
        ]
        
        # Validate fit parameters
        is_valid, warning_msg = self.validate_fit_parameters(root_fit_params, event_idx)
        
        if not is_valid:
            print(f"WARNING: {warning_msg}")
            
            # Try Python fitting as fallback if root_filename is provided
            if root_filename is not None:
                try:
                    x, y, z, fitted_z, residuals, fit_params, true_x, true_y, fit_info = self.fit_with_python_fallback(root_filename, event_idx, charge_type)
                    fallback_used = True
                except Exception as e:
                    print(f"Python fallback failed: {e}")
                    print("Proceeding with ROOT parameters despite issues...")
                    fallback_used = False
                    # Fall back to original method
                    x, y, z, pixel_x, pixel_y, true_x, true_y = self.extract_charge_data_like_fit_gaussian(root_filename, event_idx, charge_type)
                    fit_params = root_fit_params
                    fitted_z = self.gaussian_3d(fit_params, (x, y))
                    residuals = z - fitted_z
            else:
                print("No root_filename provided for Python fallback, proceeding with ROOT parameters...")
                fallback_used = False
                x, y, z, fitted_z, residuals = self.calculate_residuals(event_idx, data, detector_params, charge_type)
                true_x = data['TrueX'][event_idx]
                true_y = data['TrueY'][event_idx]
                fit_params = root_fit_params
        else:
            # Parameters are valid, use them as normal
            if not data['FitConstraintsSatisfied'][event_idx]:
                fallback_used = True  # unsuccessful fit, but we'll try to plot data anyway
            else:
                fallback_used = False
                
            if root_filename is not None:
                try:
                    # Extract charge data using the exact same method as fit_gaussian.py
                    x, y, z, pixel_x, pixel_y, true_x, true_y = self.extract_charge_data_like_fit_gaussian(root_filename, event_idx, charge_type)
                    
                    fit_params = root_fit_params
                    
                    # Calculate fitted values and residuals using the fit parameters from ROOT
                    fitted_z = self.gaussian_3d(fit_params, (x, y))
                    residuals = z - fitted_z
                    
                    # Print debug info to compare with fit_gaussian.py
                    print(f"DEBUG: Charge data range: {np.min(z):.6f} to {np.max(z):.6f}")
                    print(f"DEBUG: Number of data points: {len(z)}")
                    print(f"DEBUG: Fit parameters from ROOT: {fit_params}")
                    
                except ImportError:
                    print("Warning: charge_sim module not available, falling back to direct data extraction")
                    # Fall back to our existing method
                    x, y, z, fitted_z, residuals = self.calculate_residuals(event_idx, data, detector_params, charge_type)
                    true_x = data['TrueX'][event_idx]
                    true_y = data['TrueY'][event_idx]
                    fit_params = root_fit_params
            else:
                # Fall back to existing method if no root_filename provided
                x, y, z, fitted_z, residuals = self.calculate_residuals(event_idx, data, detector_params, charge_type)
                true_x = data['TrueX'][event_idx]
                true_y = data['TrueY'][event_idx]
                fit_params = root_fit_params
        
        # Create figure with 2 subplots (side by side) - matching fit_gaussian.py exactly
        plt.close('all')
        
        # Use GridSpec for precise layout control
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(25, 8))  # Slightly wider to better accommodate the proportions
        # Adjust width ratios: very narrow colorbar, left panel wider for square aspect, right panel for nice rectangle
        gs = GridSpec(1, 3, width_ratios=[0.03, 1.3, 1.0], wspace=0.0)  # No spacing between subplots
        
        # Create subplots with optimized sizes
        cax = fig.add_subplot(gs[0, 0])  # Colorbar space (leftmost)
        ax0 = fig.add_subplot(gs[0, 1])  # Left panel (middle)
        ax1 = fig.add_subplot(gs[0, 2])  # Right panel (rightmost)
        
        axs = [ax0, ax1]  # For compatibility with existing code
        
        plt.style.use('classic')
        
        fig.patch.set_facecolor('white')
        for ax in axs:
            ax.set_facecolor('white')
        
        # Remove any additional spacing
        fig.subplots_adjust(wspace=0.0)
        
        # Left panel: Data and fit as 2D contour plot - exact replication from fit_gaussian.py
        # Create grid for smooth surface plot
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_grid = np.linspace(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range, 50)
        y_grid = np.linspace(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_fitted_smooth = self.gaussian_3d(fit_params, (X_grid, Y_grid))
        
        # Debug: Check the range of the fitted surface
        print(f"DEBUG: Fitted surface range: {np.min(Z_fitted_smooth):.6f} to {np.max(Z_fitted_smooth):.6f}")
        
        # Plot fitted contours with better visibility
        contour = axs[0].contourf(X_grid, Y_grid, Z_fitted_smooth, levels=20, cmap='viridis', alpha=0.8)
        contour_lines = axs[0].contour(X_grid, Y_grid, Z_fitted_smooth, levels=12, colors='white', alpha=0.8, linewidths=0.8)
        
        # Plot data points with better contrast
        scatter = axs[0].scatter(x, y, c=z, s=120, cmap='viridis', edgecolors='white', linewidth=2, 
                               alpha=1.0, label='Data Points', zorder=5)
        
        # Add colorbar in the dedicated space
        charge_unit, residual_unit = self.get_charge_units(charge_type)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label(f'Charge {charge_unit}', fontsize=12, rotation=90, labelpad=15)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.label.set_horizontalalignment('center')
        cbar.ax.yaxis.tick_left()  # Move ticks to the left side
        
        # Position colorbar axis flush against contour plot
        pos0 = ax0.get_position()
        cbpos = cax.get_position()
        # Set cax to occupy its width immediately left of ax0
        cax.set_position([pos0.x0 - cbpos.width, pos0.y0, cbpos.width, pos0.height])
        
        # Add fit center with single + symbol
        axs[0].plot(fit_params[1], fit_params[2], '+', color='red', markersize=15, markeredgewidth=4, 
                   label='Fit Center')
        
        # Add TrueX, TrueY 
        axs[0].plot(true_x, true_y, 'x', color='orange', markersize=12, markeredgewidth=4, 
                   label='True Position')
            
        # Draw line from fit center to true position
        axs[0].plot([fit_params[1], true_x], [fit_params[2], true_y], 
                   'r--', linewidth=2, alpha=0.7, label='Distance')
        
        # Calculate and display distance
        distance = np.sqrt((fit_params[1] - true_x)**2 + (fit_params[2] - true_y)**2)
        
        # Get PixelTrueDistance from ROOT file for this event
        pixel_true_distance = data['PixelTrueDistance'][event_idx]
        
        # Add distance text annotation
        mid_x = (fit_params[1] + true_x) / 2
        mid_y = (fit_params[2] + true_y) / 2
        axs[0].annotate(f'{distance:.3f} mm', xy=(mid_x, mid_y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Add invisible line for PixelTrueDistance legend entry
        axs[0].plot([], [], ' ', label=f'Pixel-True Dist: {pixel_true_distance:.3f} mm')
        
        # Add title indicating fallback if used
        fit_status = data['FitConstraintsSatisfied'][event_idx]
        if fit_status:
            title = f'Event {event_idx} - Successful Fit (FitConstraintsSatisfied=True)'
            status_color = 'green'
        else:
            title = f'Event {event_idx} - Failed Fit (FitConstraintsSatisfied=False)'
            status_color = 'red'
        
        # Add title to the figure (not individual subplots to preserve layout)
        fig.suptitle(title, fontsize=16, fontweight='bold', color=status_color, y=0.95)
        
        # Add invisible line for fit status in legend
        axs[0].plot([], [], ' ', label=f'Fit Status: {"✓ Success" if fit_status else "✗ Failed"}')
        
        axs[0].set_xlabel('X (mm)', fontsize=14)
        axs[0].set_ylabel('Y (mm)', fontsize=14)
        axs[0].set_aspect('equal')  # Re-enabled now that we have proper space allocation
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper right')
        
        # Right panel: Residuals analysis - exact replication from fit_gaussian.py
        # Plot residuals vs radial distance from fit center
        r_distance = np.sqrt((x - fit_params[1])**2 + (y - fit_params[2])**2)
        
        # Estimate errors (use same approach as fit_gaussian.py)
        z_err = np.full_like(z, np.std(z) * 0.1)  # 10% of data std as default error
        
        axs[1].errorbar(r_distance, residuals, xerr=0, yerr=z_err, fmt='.b', markersize=10, 
                       ecolor='gray', capsize=4, alpha=0.8, label='Residuals')
        
        # Add zero reference line
        axs[1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        axs[1].set_xlabel('Radial Distance from Fit Center (mm)', fontsize=14)
        axs[1].set_ylabel(f'Fitted - Observed {residual_unit}', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        axs[1].ticklabel_format(style='plain', useOffset=False, axis='y')
        
        # Auto-adjust axis limits based on data ranges with appropriate margins
        # For x-axis: more spacious margin around radial distance
        x_margin = 0.15 * (np.max(r_distance) - np.min(r_distance)) if len(r_distance) > 1 else 0.2
        
        # For y-axis: account for error bars and make it more spacious
        residuals_with_err_min = np.min(residuals - z_err)
        residuals_with_err_max = np.max(residuals + z_err)
        y_range = residuals_with_err_max - residuals_with_err_min
        y_margin = 0.1 * y_range if y_range > 0 else 0.01  # 20% margin, more spacious
        
        axs[1].set_xlim(np.min(r_distance) - x_margin, np.max(r_distance) + x_margin)
        axs[1].set_ylim(residuals_with_err_min - y_margin, residuals_with_err_max + y_margin)
        
        # Adjust formatting for both axes
        for ax in axs:
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            ax.tick_params(labelsize=12)
        
        # Don't use tight_layout as it interferes with our precise GridSpec layout
        # plt.tight_layout()  # Removed to preserve equal subplot heights
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create descriptive filename based on fit status
            fit_status = data['FitConstraintsSatisfied'][event_idx]
            if fit_status:
                prefix = "successful_fit"
                status_desc = "success"
            else:
                prefix = "failed_fit"
                status_desc = "failed"
            
            filename = f'{prefix}_event_{event_idx}_{status_desc}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to {filename}")
        
        # plt.show()  # Commented out for non-interactive backend
        return fig

    def plot_3d_visualization(self, event_idx, data, detector_params, charge_type='fraction', 
                             save_plot=True, output_dir=""):
        """
        Create 3D visualization of the Gaussian fit results
        """
        # Check if fit was successful
        if not data['FitConstraintsSatisfied'][event_idx]:
            print(f"Warning: Fit was not successful for event {event_idx}")
            return None
        
        # Get coordinates and residuals
        x, y, z, fitted_z, residuals = self.calculate_residuals(event_idx, data, detector_params, charge_type)
        
        # Get fit parameters
        fit_params = [
            data['FitAmplitude'][event_idx],
            data['FitX0'][event_idx],
            data['FitY0'][event_idx],
            data['FitSigmaX'][event_idx],
            data['FitSigmaY'][event_idx],
            data['FitTheta'][event_idx],
            data['FitOffset'][event_idx]
        ]
        
        # Get true positions
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        # Create high-resolution grid for smooth surface
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_margin = 0.15 * x_range
        y_margin = 0.15 * y_range
        
        x_grid = np.linspace(np.min(x) - x_margin, np.max(x) + x_margin, 60)
        y_grid = np.linspace(np.min(y) - y_margin, np.max(y) + y_margin, 60)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_fitted_smooth = self.gaussian_3d(fit_params, (X_grid, Y_grid))
        
        # Get units for labels
        charge_unit, residual_unit = self.get_charge_units(charge_type)
        
        # Create figure with 4 3D subplots (2x2 layout)
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('white')
        
        # 1. Original Data with Error Bars
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_facecolor('white')
        
        scatter1 = ax1.scatter(x, y, z, c=z, cmap='plasma', s=100, alpha=0.9, 
                              edgecolors='black', linewidth=1, depthshade=True)
        
        # Mark true position and fit center
        true_z_proj = np.mean(z)
        ax1.scatter([true_x], [true_y], [true_z_proj], c='red', s=200, marker='x', 
                   linewidth=4, label='True Position')
        
        fit_z_proj = np.mean(z)
        ax1.scatter([fit_params[1]], [fit_params[2]], [fit_z_proj], c='orange', s=200, 
                   marker='+', linewidth=4, label='Fit Center')
        
        ax1.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax1.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax1.set_zlabel(f'Charge {charge_unit}', fontsize=12, labelpad=10)
        ax1.set_title('Original Data Points', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1, label=f'Charge {charge_unit}')
        
        # 2. Fitted 3D Gaussian Surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.set_facecolor('white')
        
        surface = ax2.plot_surface(X_grid, Y_grid, Z_fitted_smooth, cmap='plasma', alpha=0.8,
                                 linewidth=0, antialiased=True, rstride=2, cstride=2)
        
        # Mark positions on surface
        true_z_fit = self.gaussian_3d(fit_params, ([true_x], [true_y]))[0]
        ax2.scatter([true_x], [true_y], [true_z_fit], c='red', s=200, marker='x', 
                   linewidth=4, label='True Position')
        
        fit_z_center = self.gaussian_3d(fit_params, ([fit_params[1]], [fit_params[2]]))[0]
        ax2.scatter([fit_params[1]], [fit_params[2]], [fit_z_center], c='orange', s=200, 
                   marker='+', linewidth=4, label='Fit Center')
        
        ax2.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax2.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax2.set_zlabel(f'Fitted Charge {charge_unit}', fontsize=12, labelpad=10)
        ax2.set_title('Fitted 3D Gaussian Surface', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        plt.colorbar(surface, ax=ax2, shrink=0.6, pad=0.1, label=f'Fitted Charge {charge_unit}')
        
        # 3. Data Points on Fitted Surface
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.set_facecolor('white')
        
        surface3 = ax3.plot_surface(X_grid, Y_grid, Z_fitted_smooth, cmap='plasma', alpha=0.5,
                                  linewidth=0, antialiased=True, rstride=3, cstride=3)
        
        scatter3 = ax3.scatter(x, y, z, c='red', s=80, alpha=0.9, edgecolors='darkred', 
                             linewidth=1, label='Data Points', depthshade=True)
        
        # Draw lines from data to fit
        for i in range(len(x)):
            ax3.plot([x[i], x[i]], [y[i], y[i]], [z[i], fitted_z[i]], 
                    'gray', alpha=0.7, linewidth=1)
        
        ax3.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax3.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax3.set_zlabel(f'Charge {charge_unit}', fontsize=12, labelpad=10)
        ax3.set_title('Data Points on Fitted Surface', fontsize=14, pad=20)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        # 4. Residuals in 3D
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.set_facecolor('white')
        
        residual_scatter = ax4.scatter(x, y, residuals, c=residuals, cmap='RdBu_r', s=100, 
                                     alpha=0.9, edgecolors='black', linewidth=1)
        
        # Add zero plane
        zero_surface = ax4.plot_surface(X_grid, Y_grid, np.zeros_like(X_grid), alpha=0.3, 
                                       color='gray', linewidth=0)
        
        ax4.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax4.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax4.set_zlabel(f'Residuals {residual_unit}', fontsize=12, labelpad=10)
        ax4.set_title(f'Residuals (Std: {np.std(residuals):.6f})', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3)
        
        plt.colorbar(residual_scatter, ax=ax4, shrink=0.6, pad=0.1, label=f'Residuals {residual_unit}')
        
        # Set consistent viewing angles
        for ax in [ax1, ax2, ax3, ax4]:
            ax.view_init(elev=20, azim=45)
            ax.tick_params(labelsize=10)
        
        # Add overall title
        fig.suptitle(f'3D Gaussian Fit Visualization - Event {event_idx} (from ROOT)\n'
                    f'χ²/NDF = {data["FitChi2red"][event_idx]:.6f}', 
                    fontsize=16, y=0.95)
        
        plt.subplots_adjust(top=0.90)
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'3d_gaussian_root_event_{event_idx}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"3D visualization saved to {filename}")
        
        return fig

    def print_fit_summary(self, event_idx, data):
        """
        Print fit summary for an event
        """
        print(f"\n{'='*70}")
        print(f"GAUSSIAN FIT SUMMARY - EVENT {event_idx}")
        print(f"{'='*70}")
        
        fit_success = data['FitConstraintsSatisfied'][event_idx]
        
        if not fit_success:
            print("Fit was NOT successful!")
            return
        
        # Print fit parameters
        print(f"{'Parameter':<20} {'Value':<20} {'Error':<20}")
        print(f"{'-'*60}")
        
        param_names = [
            ('Amplitude', 'FitAmplitude', 'FitAmplitudeErr'),
            ('X Center [mm]', 'FitX0', 'FitX0Err'),
            ('Y Center [mm]', 'FitY0', 'FitY0Err'),
            ('Sigma X [mm]', 'FitSigmaX', 'FitSigmaXErr'),
            ('Sigma Y [mm]', 'FitSigmaY', 'FitSigmaYErr'),
            ('Rotation [rad]', 'FitTheta', 'FitThetaErr'),
            ('Offset', 'FitOffset', 'FitOffsetErr')
        ]
        
        for param_name, val_key, err_key in param_names:
            val = data[val_key][event_idx]
            err = data[err_key][event_idx]
            print(f"{param_name:<20} {val:<20.6f} {err:<20.6f}")
        
        print(f"\n{'='*70}")
        print("FIT STATISTICS")
        print(f"{'='*70}")
        
        # Statistics for fit
        stat_names = [
            ('Chi-squared', 'FitChi2red'),
            ('Degrees of Freedom', 'FitNDF'),
            ('Fit Probability', 'FitPp'),
            ('Data Points', 'FitNPoints'),
            ('Residual Mean', 'FitResidualMean'),
            ('Residual Std', 'FitResidualStd')
        ]
        
        for stat_name, key in stat_names:
            val = data[key][event_idx]
            if stat_name == 'Degrees of Freedom' or stat_name == 'Data Points':
                print(f"{stat_name:<20} {int(val)}")
            else:
                print(f"{stat_name:<20} {val:.6f}")
        
        # Reduced Chi-squared calculation (note: FitChi2red in ROOT is already chi2/ndf)
        chi2_per_ndf = data['FitChi2red'][event_idx]
        print(f"{'Chi²/NDF':<20} {chi2_per_ndf:.6f}")
        
        print(f"\n{'='*70}")
        print("POSITION COMPARISON")
        print(f"{'='*70}")
        
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        print(f"True Position:       ({true_x:.6f}, {true_y:.6f}) mm")
        
        fit_x = data['FitX0'][event_idx]
        fit_y = data['FitY0'][event_idx]
        distance = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
        print(f"Fitted Position:     ({fit_x:.6f}, {fit_y:.6f}) mm, Distance: {distance:.6f} mm")
        
        print(f"{'='*70}")

    def extract_charge_data_like_fit_gaussian(self, root_filename, event_idx, charge_type='fraction'):
        """
        Extract charge data using the exact same method as fit_gaussian.py
        This ensures we get the same data that would be used for Python fitting
        """
        from charge_sim import RandomHitChargeGridGenerator
        
        # Load data using the charge_sim module - exactly like fit_gaussian.py
        generator = RandomHitChargeGridGenerator(root_filename)
        
        # Get detector parameters
        pixel_spacing = generator.detector_params['pixel_spacing']
        
        # Get event data
        pixel_x = generator.data['PixelX'][event_idx]
        pixel_y = generator.data['PixelY'][event_idx]
        true_x = generator.data['TrueX'][event_idx]
        true_y = generator.data['TrueY'][event_idx]
        
        # Get charge data based on type - updated for new branch names
        if charge_type == 'fraction':
            charge_data = generator.data['NonPixel_GridNeighborhoodChargeFractions'][event_idx]
        elif charge_type == 'value':
            charge_data = generator.data['NonPixel_GridNeighborhoodCharge'][event_idx]
        elif charge_type == 'coulomb':
            if 'NonPixel_GridNeighborhoodCharge' not in generator.data:
                raise ValueError("Coulomb charge data not available in this ROOT file")
            charge_data = generator.data['NonPixel_GridNeighborhoodCharge'][event_idx]
        else:
            raise ValueError("charge_type must be 'fraction', 'value', or 'coulomb'")
        
        # Reshape to 9x9 grid - exactly like fit_gaussian.py
        grid_data = np.array(charge_data).reshape(9, 9)
        
        # Replace invalid values (-999.0) with NaN - exactly like fit_gaussian.py
        grid_data[grid_data == -999.0] = np.nan
        
        # Create coordinate arrays - exactly like fit_gaussian.py
        x_coords = []
        y_coords = []
        z_values = []
        
        for i in range(9):
            for j in range(9):
                if not np.isnan(grid_data[i, j]):
                    # Calculate actual position relative to pixel center - exactly like fit_gaussian.py
                    rel_x = (j - 4) * pixel_spacing
                    rel_y = (i - 4) * pixel_spacing
                    
                    actual_x = pixel_x + rel_x
                    actual_y = pixel_y + rel_y
                    
                    x_coords.append(actual_x)
                    y_coords.append(actual_y)
                    z_values.append(grid_data[i, j])
        
        return np.array(x_coords), np.array(y_coords), np.array(z_values), pixel_x, pixel_y, true_x, true_y

    def validate_fit_parameters(self, fit_params, event_idx):
        """
        Validate fit parameters to detect unreasonable values from ROOT file
        
        Parameters:
        -----------
        fit_params : list
            [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
        event_idx : int
            Event index for error reporting
            
        Returns:
        --------
        bool : True if parameters are reasonable, False otherwise
        str : Warning message if parameters are bad
        """
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = fit_params
        
        warnings = []
        
        # Check sigma values - should be reasonable for pixel detector
        # Pixel spacing is typically 0.5mm, so sigmas should be roughly 0.1-2.0 mm
        if sigma_x < 0.01 or sigma_x > 5.0:
            warnings.append(f"Sigma X ({sigma_x:.6f} mm) is unreasonable")
        
        if sigma_y < 0.01 or sigma_y > 5.0:
            warnings.append(f"Sigma Y ({sigma_y:.6f} mm) is unreasonable")
        
        # Check amplitude - should be positive and reasonable
        if amplitude <= 0:
            warnings.append(f"Amplitude ({amplitude:.6f}) is non-positive")
        
        if amplitude > 1.0:  # Charge fractions should be < 1
            warnings.append(f"Amplitude ({amplitude:.6f}) is too large for charge fraction")
        
        # Check offset - should be small and positive for charge fractions
        if offset < 0:
            warnings.append(f"Offset ({offset:.6f}) is negative")
        
        if offset > 0.5:  # Offset shouldn't be too large
            warnings.append(f"Offset ({offset:.6f}) is too large")
        
        is_valid = len(warnings) == 0
        warning_msg = f"Event {event_idx} has problematic fit parameters: " + "; ".join(warnings) if warnings else ""
        
        return is_valid, warning_msg

    def fit_with_python_fallback(self, root_filename, event_idx, charge_type='fraction'):
        """
        Use Python fitting as fallback when ROOT fit parameters are bad
        """
        try:
            from fit_gaussian import fit_charge_distribution_3d
            print(f"Using Python fitting as fallback for event {event_idx}")
            
            # Perform Python fitting
            python_results = fit_charge_distribution_3d(root_filename, event_idx, charge_type, 
                                                       save_plots=False)
            
            # Extract parameters in the same format
            py_params = python_results['parameters']
            fit_params = [
                py_params['amplitude'],
                py_params['x0'], 
                py_params['y0'],
                py_params['sigma_x'],
                py_params['sigma_y'],
                py_params['theta'],
                py_params['offset']
            ]
            
            # Extract charge data
            x, y, z, pixel_x, pixel_y, true_x, true_y = self.extract_charge_data_like_fit_gaussian(root_filename, event_idx, charge_type)
            
            # Calculate fitted values and residuals using Python fit parameters
            fitted_z = self.gaussian_3d(fit_params, (x, y))
            residuals = z - fitted_z
            
            print(f"Python fallback - R²: {python_results['fit_info']['r_squared']:.6f}")
            
            return x, y, z, fitted_z, residuals, fit_params, true_x, true_y, python_results['fit_info']
            
        except ImportError:
            print("Error: fit_gaussian module not available for Python fallback")
            raise
        except Exception as e:
            print(f"Error in Python fallback fitting: {e}")
            raise

    def plot_pixel_hit_info(self, event_idx, data, detector_params, charge_type='fraction',
                           save_plot=True, output_dir="", filename_prefix="pixel_hit"):
        """
        Create an information plot for pixel hits (no Gaussian fitting performed)
        
        This creates a simple plot showing the basic event information for pixel hits
        since Gaussian fitting is not performed for these events.
        """
        # Get basic event information
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        distance = data['PixelTrueDistance'][event_idx]
        is_within_d0 = data['IsWithinD0'][event_idx]
        edep = data['Edep'][event_idx]
        
        # Create a simple information plot
        plt.close('all')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Plot pixel center and true position
        ax.plot(pixel_x, pixel_y, 's', color='blue', markersize=15, markeredgewidth=2, 
               markerfacecolor='lightblue', label='Pixel Center')
        ax.plot(true_x, true_y, 'x', color='red', markersize=12, markeredgewidth=3, 
               label='True Hit Position')
        
        # Draw line between them
        ax.plot([pixel_x, true_x], [pixel_y, true_y], 'r--', linewidth=2, alpha=0.7, 
               label=f'Distance: {distance*1000:.1f} μm')
        
        # Draw pixel boundary (approximate)
        pixel_size = detector_params.get('pixel_size', 0.055)  # Default 55 μm
        pixel_half = pixel_size / 2
        
        # Draw pixel square
        pixel_corners_x = [pixel_x - pixel_half, pixel_x + pixel_half, 
                          pixel_x + pixel_half, pixel_x - pixel_half, pixel_x - pixel_half]
        pixel_corners_y = [pixel_y - pixel_half, pixel_y - pixel_half, 
                          pixel_y + pixel_half, pixel_y + pixel_half, pixel_y - pixel_half]
        ax.plot(pixel_corners_x, pixel_corners_y, 'b-', linewidth=2, alpha=0.8, label='Pixel Boundary')
        
        # Set axis labels and title
        ax.set_xlabel('X Position (mm)', fontsize=14)
        ax.set_ylabel('Y Position (mm)', fontsize=14)
        
        # Determine hit classification for title
        if is_within_d0:
            classification = f"Pixel Hit (distance ≤ D0: {distance*1000:.1f} μm ≤ 10 μm)"
            title_color = 'blue'
        else:
            classification = f"Pixel Hit (on pixel, distance: {distance*1000:.1f} μm)"
            title_color = 'green'
        
        title = f'Event {event_idx} - {classification}\nNo Gaussian Fitting Performed'
        ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
        
        # Add information text box
        info_text = f'''Event Information:
Energy Deposit: {edep:.6f} MeV
True Position: ({true_x:.6f}, {true_y:.6f}) mm
Pixel Center: ({pixel_x:.6f}, {pixel_y:.6f}) mm
Distance to Pixel: {distance*1000:.1f} μm
Within D0 (10 μm): {"Yes" if is_within_d0 else "No"}
Pixel Alpha: {data['PixelHit_PixelAlpha'][event_idx]:.3f}°'''

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightyellow', alpha=0.8))
        
        # Set equal aspect ratio and appropriate limits
        ax.set_aspect('equal')
        
        # Calculate reasonable plot limits
        center_x = (true_x + pixel_x) / 2
        center_y = (true_y + pixel_y) / 2
        range_x = max(abs(true_x - pixel_x), pixel_size * 2)
        range_y = max(abs(true_y - pixel_y), pixel_size * 2)
        margin = max(range_x, range_y) * 0.3
        
        ax.set_xlim(center_x - range_x/2 - margin, center_x + range_x/2 + margin)
        ax.set_ylim(center_y - range_y/2 - margin, center_y + range_y/2 + margin)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'{filename_prefix}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            # Note: Don't print save message here to avoid spam during batch processing
        
        return fig

def analyze_multiple_events_from_root(root_filename, event_indices=None, charge_type='fraction', 
                                    save_plots=True, output_dir="", plot_style='simple'):
    """
    Analyze multiple events from ROOT file and create plots
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file
    event_indices : list, optional
        List of event indices to analyze. If None, analyzes first 10 successful fits
    charge_type : str
        Type of charge data: 'fraction', 'value', or 'coulomb'
    save_plots : bool
        Whether to save plots
    output_dir : str
        Output directory for plots
    plot_style : str
        'simple', '3d', or 'both'
    """
    print(f"Loading data from ROOT file: {root_filename}")
    
    # Create plotter and load data
    plotter = GaussianRootPlotter()
    data, detector_params = plotter.load_root_data(root_filename)
    
    print(f"Loaded {len(data['EventID'])} events from ROOT file")
    
    # Find successful fits if no specific events requested
    if event_indices is None:
        successful_events = np.where(data['FitConstraintsSatisfied'])[0]
        if len(successful_events) == 0:
            print("No successful fits found in ROOT file!")
            return
        
        event_indices = successful_events[:min(10, len(successful_events))]
        print(f"Found {len(successful_events)} successful fits, analyzing first {len(event_indices)}")
    
    # Create output directory
    if save_plots and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each event
    results_summary = []
    
    for i, event_idx in enumerate(event_indices):
        print(f"\n{'='*50}")
        print(f"Processing event {event_idx} ({i+1}/{len(event_indices)})")
        print(f"{'='*50}")
        
        if not data['FitConstraintsSatisfied'][event_idx]:
            print(f"Skipping event {event_idx} - fit was not successful")
            continue
        
        # Print fit summary
        plotter.print_fit_summary(event_idx, data)
        
        # Create plots
        try:
            if plot_style in ['simple', 'both']:
                plotter.plot_simple_fit_results(event_idx, data, detector_params, charge_type, 
                                               save_plots, output_dir, root_filename)
            
            if plot_style in ['3d', 'both']:
                plotter.plot_3d_visualization(event_idx, data, detector_params, charge_type, 
                                            save_plots, output_dir)
            
            # Collect results for summary (using fit results)
            true_x = data['TrueX'][event_idx]
            true_y = data['TrueY'][event_idx]
            fit_x = data['FitX0'][event_idx]
            fit_y = data['FitY0'][event_idx]
            distance = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
            
            results_summary.append({
                'event_idx': event_idx,
                'chi2_per_ndf': data['FitChi2red'][event_idx],
                'distance_error': distance,
                'sigma_x': data['FitSigmaX'][event_idx],
                'sigma_y': data['FitSigmaY'][event_idx]
            })
            
        except Exception as e:
            print(f"Error processing event {event_idx}: {e}")
            continue
    
    # Print overall summary
    if results_summary:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY OF GAUSSIAN FITS FROM ROOT")
        print(f"{'='*70}")
        
        chi2_per_ndf_values = [r['chi2_per_ndf'] for r in results_summary]
        distance_errors = [r['distance_error'] for r in results_summary]
        sigma_x_values = [r['sigma_x'] for r in results_summary]
        sigma_y_values = [r['sigma_y'] for r in results_summary]
        
        print(f"Successfully processed {len(results_summary)} events")
        print(f"Average χ²/NDF:         {np.mean(chi2_per_ndf_values):.6f} ± {np.std(chi2_per_ndf_values):.6f}")
        print(f"Average Distance Error: {np.mean(distance_errors):.6f} ± {np.std(distance_errors):.6f} mm")
        print(f"Average σₓ:             {np.mean(sigma_x_values):.6f} ± {np.std(sigma_x_values):.6f} mm")
        print(f"Average σᵧ:             {np.mean(sigma_y_values):.6f} ± {np.std(sigma_y_values):.6f} mm")
        
        print(f"\nRange of Distance Errors: {np.min(distance_errors):.6f} to {np.max(distance_errors):.6f} mm")
        print(f"Range of χ²/NDF:         {np.min(chi2_per_ndf_values):.6f} to {np.max(chi2_per_ndf_values):.6f}")
    
    return results_summary

def plot_single_event_from_root(root_filename, event_idx, charge_type='fraction', 
                               save_plots=True, output_dir="", plot_style='both'):
    """
    Convenience function to plot a single event from ROOT file
    """
    # Create plotter and load data
    plotter = GaussianRootPlotter()
    data, detector_params = plotter.load_root_data(root_filename)
    
    if event_idx >= len(data['EventID']):
        print(f"Error: Event {event_idx} not found. Only {len(data['EventID'])} events in file.")
        return None
    
    if not data['FitConstraintsSatisfied'][event_idx]:
        print(f"Error: Fit was not successful for event {event_idx}")
        return None
    
    # Create output directory
    if save_plots and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print summary
    plotter.print_fit_summary(event_idx, data)
    
    # Create plots
    results = {}
    
    if plot_style in ['simple', 'both']:
        results['simple'] = plotter.plot_simple_fit_results(event_idx, data, detector_params, 
                                                           charge_type, save_plots, output_dir, root_filename)
    
    if plot_style in ['3d', 'both']:
        results['3d'] = plotter.plot_3d_visualization(event_idx, data, detector_params, 
                                                    charge_type, save_plots, output_dir)
    
    return results

def compare_python_vs_cpp_fits(root_filename, event_idx, charge_type='fraction'):
    """
    Compare Python-computed fit vs C++ ROOT file fit for the same event
    This requires both fit_gaussian.py and the ROOT file with C++ fits
    """
    print(f"Comparing Python vs C++ fits for event {event_idx}")
    
    # Get C++ results from ROOT file
    plotter = GaussianRootPlotter()
    data, detector_params = plotter.load_root_data(root_filename)
    
    if not data['FitConstraintsSatisfied'][event_idx]:
        print(f"C++ fit was not successful for event {event_idx}")
        return
    
    print("\nC++ FIT RESULTS (from ROOT file):")
    plotter.print_fit_summary(event_idx, data)
    
    # Try to get Python results (requires fit_gaussian.py)
    try:
        from fit_gaussian import fit_charge_distribution_3d
        print("\nPYTHON FIT RESULTS:")
        python_results = fit_charge_distribution_3d(root_filename, event_idx, charge_type, 
                                                   save_plots=False)
        
        # Compare key parameters
        print(f"\n{'='*60}")
        print("COMPARISON (C++ vs Python)")
        print(f"{'='*60}")
        
        cpFitPparams = {
            'amplitude': data['FitAmplitude'][event_idx],
            'x0': data['FitX0'][event_idx],
            'y0': data['FitY0'][event_idx],
            'sigma_x': data['FitSigmaX'][event_idx],
            'sigma_y': data['FitSigmaY'][event_idx],
            'theta': data['FitTheta'][event_idx],
            'offset': data['FitOffset'][event_idx],
            'chi2red': data['FitChi2red'][event_idx]
        }
        
        py_params = python_results['parameters']
        py_r_squared = python_results['fit_info']['r_squared']
        
        print(f"{'Parameter':<12} {'C++':<12} {'Python':<12} {'Difference':<12} {'Rel. Diff %':<12}")
        print("-" * 70)
        
        for param in ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']:
            cpp_val = cpFitPparams[param]
            py_val = py_params[param]
            diff = abs(cpp_val - py_val)
            rel_diff = diff / abs(py_val) * 100 if py_val != 0 else 0
            
            print(f"{param:<12} {cpp_val:<12.6f} {py_val:<12.6f} {diff:<12.6f} {rel_diff:<12.2f}")
        
        print(f"{'R-squared':<12} {cpFitPparams['chi2red']:<12.6f} {py_r_squared:<12.6f} "
              f"{abs(cpFitPparams['chi2red'] - py_r_squared):<12.6f} "
              f"{abs(cpFitPparams['chi2red'] - py_r_squared)/py_r_squared*100:<12.2f}")
        
    except ImportError:
        print("fit_gaussian.py not available for comparison")
    except Exception as e:
        print(f"Error running Python fit: {e}")

# COMMENTED OUT - outlier functionality removed from C++ code
# def plot_dual_fit_from_root(root_filename, event_idx, charge_type='fraction', 
#                            save_plots=True, output_dir=""):
#     """
#     Convenience function to plot dual fit comparison for a single event from ROOT file
#     COMMENTED OUT - outlier functionality removed
#     """
#     pass

# COMMENTED OUT - outlier functionality removed from C++ code
# def analyze_multiple_events_dual_fits(root_filename, event_indices=None, charge_type='fraction', 
#                                      save_plots=True, output_dir=""):
#     """
#     Analyze multiple events from ROOT file and create dual fit comparison plots
#     COMMENTED OUT - outlier functionality removed  
#     """
#     pass

# COMMENTED OUT - outlier functionality removed from C++ code
# def analyze_outlier_patterns_from_root(root_filename, event_idx, charge_type='fraction',
#                                       save_plots=True, output_dir=""):
#     """
#     Convenience function to analyze outlier patterns for a single event
#     COMMENTED OUT - outlier functionality removed
#     """
#     pass

# COMMENTED OUT - outlier functionality removed from C++ code
# def analyze_outlier_patterns_multiple_events(root_filename, event_indices=None, charge_type='fraction',
#                                             save_plots=True, output_dir=""):
#     """
#     Analyze outlier patterns across multiple events to find overall trends
#     COMMENTED OUT - outlier functionality removed
#     """
#     pass

def plot_all_events_from_root(root_filename, charge_type='fraction', 
                             save_plots=True, output_dir="all_events", plot_style='simple'):
    """
    Plot and save individual plots for ALL events in the ROOT file
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file
    charge_type : str
        Type of charge data: 'fraction', 'value', or 'coulomb'
    save_plots : bool
        Whether to save plots (should be True for this function)
    output_dir : str
        Output directory for plots
    plot_style : str
        'simple', '3d', or 'both'
        
    Returns:
    --------
    dict : Summary statistics of all events processed
    """
    print(f"Loading data from ROOT file: {root_filename}")
    
    # Create plotter and load data
    plotter = GaussianRootPlotter()
    data, detector_params = plotter.load_root_data(root_filename)
    
    total_events = len(data['EventID'])
    print(f"Found {total_events} events in ROOT file")
    
    if total_events == 0:
        print("No events found in ROOT file!")
        return {}
    
    # Create output directory
    if save_plots and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Analyze event types
    is_pixel_hit = data['IsPixelHit']
    is_within_d0 = data['IsWithinD0']
    fit_success = data['FitConstraintsSatisfied']
    
    # Categorize events
    pixel_hits = np.where(is_pixel_hit)[0]
    non_pixel_hits = np.where(~is_pixel_hit)[0]
    non_pixel_successful = np.where((~is_pixel_hit) & fit_success)[0]
    non_pixel_unsuccessful = np.where((~is_pixel_hit) & (~fit_success))[0]
    
    print(f"\nEvent classification:")
    print(f"  Pixel hits (on pixel or distance <= D0):     {len(pixel_hits)}")
    print(f"  Non-pixel hits (distance > D0):              {len(non_pixel_hits)}")
    print(f"    - Successful fits:                          {len(non_pixel_successful)}")
    print(f"    - Unsuccessful fits:                        {len(non_pixel_unsuccessful)}")
    
    # Statistics tracking
    stats = {
        'total_events': total_events,
        'pixel_hits': len(pixel_hits),
        'non_pixel_hits': len(non_pixel_hits),
        'successful_fits': len(non_pixel_successful),
        'unsuccessful_fits': len(non_pixel_unsuccessful),
        'plots_created': 0,
        'plots_failed': 0,
        'successful_fit_results': [],
        'processing_errors': []
    }
    
    # Process each event
    print(f"\nProcessing all {total_events} events...")
    for event_idx in range(total_events):
        try:
            # Progress indicator
            if (event_idx + 1) % 10 == 0 or event_idx == total_events - 1:
                print(f"  Processing event {event_idx + 1}/{total_events} ({(event_idx + 1)/total_events*100:.1f}%)")
            
            # Determine event type for filename
            is_pixel = is_pixel_hit[event_idx]
            has_successful_fit = fit_success[event_idx]
            distance = data['PixelTrueDistance'][event_idx]
            
            if is_pixel:
                event_type = "pixel_hit"
                status = f"dist_{distance*1000:.1f}um"  # Convert mm to microns
            else:
                if has_successful_fit:
                    event_type = "nonpixel_success"
                    chi2_ndf = data['FitChi2red'][event_idx] 
                    status = f"chi2ndf_{chi2_ndf:.3f}"
                else:
                    event_type = "nonpixel_failed"
                    status = "no_fit"
            
            # Create subdirectory for event type
            event_output_dir = os.path.join(output_dir, event_type) if output_dir else event_type
            if save_plots and not os.path.exists(event_output_dir):
                os.makedirs(event_output_dir)
            
            # Create descriptive filename
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"event_{event_idx:04d}_{event_type}_{status}"
            
            # Plot the event
            if is_pixel:
                # For pixel hits, create a basic data visualization
                # Since no Gaussian fitting is performed, we'll show the basic event info
                try:
                    # Create a simple info plot for pixel hits
                    fig = plotter.plot_pixel_hit_info(event_idx, data, detector_params, 
                                                    charge_type, save_plot=False, 
                                                    output_dir=event_output_dir,
                                                    filename_prefix=filename_base)
                    if fig is not None:
                        stats['plots_created'] += 1
                        
                        # Save with custom filename
                        if save_plots:
                            custom_filename = os.path.join(event_output_dir, f"{filename_base}.png")
                            fig.savefig(custom_filename, dpi=300, bbox_inches='tight', facecolor='white')
                        
                        plt.close(fig)  # Clean up memory
                except Exception as e:
                    print(f"    Warning: Could not create pixel hit plot for event {event_idx}: {e}")
                    # For pixel hits that can't be plotted, we'll skip but not count as error
                    pass
            else:
                # For non-pixel hits, use the existing plotting functions
                if plot_style in ['simple', 'both']:
                    try:
                        fig = plotter.plot_simple_fit_results(event_idx, data, detector_params, 
                                                            charge_type, save_plot=False, 
                                                            output_dir=event_output_dir, 
                                                            root_filename=root_filename)
                        if fig is not None:
                            stats['plots_created'] += 1
                            
                            # Save with custom filename
                            if save_plots:
                                custom_filename = os.path.join(event_output_dir, f"{filename_base}_simple.png")
                                fig.savefig(custom_filename, dpi=300, bbox_inches='tight', facecolor='white')
                            
                            plt.close(fig)  # Clean up memory
                    except Exception as e:
                        print(f"    Error creating simple plot for event {event_idx}: {e}")
                        stats['plots_failed'] += 1
                        stats['processing_errors'].append(f"Event {event_idx}: {str(e)}")
                
                if plot_style in ['3d', 'both'] and has_successful_fit:
                    try:
                        fig = plotter.plot_3d_visualization(event_idx, data, detector_params, 
                                                          charge_type, save_plot=False, 
                                                          output_dir=event_output_dir)
                        if fig is not None:
                            # Save with custom filename
                            if save_plots:
                                custom_filename = os.path.join(event_output_dir, f"{filename_base}_3d.png")
                                fig.savefig(custom_filename, dpi=300, bbox_inches='tight', facecolor='white')
                            
                            plt.close(fig)  # Clean up memory
                    except Exception as e:
                        print(f"    Error creating 3D plot for event {event_idx}: {e}")
                        stats['processing_errors'].append(f"Event {event_idx} 3D: {str(e)}")
            
            # Collect fit results for successful fits
            if has_successful_fit and not is_pixel:
                try:
                    true_x = data['TrueX'][event_idx]
                    true_y = data['TrueY'][event_idx]
                    fit_x = data['FitX0'][event_idx]
                    fit_y = data['FitY0'][event_idx]
                    distance_error = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
                    
                    stats['successful_fit_results'].append({
                        'event_idx': event_idx,
                        'chi2_per_ndf': data['FitChi2red'][event_idx],
                        'distance_error': distance_error,
                        'sigma_x': data['FitSigmaX'][event_idx],
                        'sigma_y': data['FitSigmaY'][event_idx],
                        'distance_to_pixel': distance
                    })
                except Exception as e:
                    print(f"    Warning: Could not collect fit statistics for event {event_idx}: {e}")
                    
        except Exception as e:
            print(f"    Error processing event {event_idx}: {e}")
            stats['plots_failed'] += 1
            stats['processing_errors'].append(f"Event {event_idx}: {str(e)}")
            continue
    
    # Print final summary
    print(f"\n{'='*70}")
    print("ALL EVENTS PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total events processed:        {stats['total_events']}")
    print(f"Plots successfully created:    {stats['plots_created']}")
    print(f"Plot creation failures:        {stats['plots_failed']}")
    print(f"Success rate:                  {stats['plots_created']/(stats['total_events'] - stats['pixel_hits'])*100:.1f}% (non-pixel events)")
    
    print(f"\nEvent breakdown:")
    print(f"  Pixel hits:                  {stats['pixel_hits']} (no Gaussian fitting)")
    print(f"  Non-pixel successful fits:   {stats['successful_fits']}")
    print(f"  Non-pixel unsuccessful fits: {stats['unsuccessful_fits']}")
    
    if stats['successful_fit_results']:
        successful_results = stats['successful_fit_results']
        chi2_values = [r['chi2_per_ndf'] for r in successful_results]
        distance_errors = [r['distance_error'] for r in successful_results]
        
        print(f"\nSuccessful fits statistics:")
        print(f"  Average χ²/NDF:              {np.mean(chi2_values):.6f} ± {np.std(chi2_values):.6f}")
        print(f"  Average distance error:      {np.mean(distance_errors):.6f} ± {np.std(distance_errors):.6f} mm")
        print(f"  Distance error range:        {np.min(distance_errors):.6f} to {np.max(distance_errors):.6f} mm")
    
    if stats['processing_errors']:
        print(f"\nFirst few processing errors:")
        for i, error in enumerate(stats['processing_errors'][:5]):
            print(f"  {i+1}. {error}")
        if len(stats['processing_errors']) > 5:
            print(f"  ... and {len(stats['processing_errors']) - 5} more errors")
    
    print(f"\nOutput directories created:")
    print(f"  {os.path.join(output_dir, 'pixel_hit')}     (pixel hits)")
    print(f"  {os.path.join(output_dir, 'nonpixel_success')}  (successful non-pixel fits)")
    print(f"  {os.path.join(output_dir, 'nonpixel_failed')}   (failed non-pixel fits)")
    
    return stats

if __name__ == "__main__":
    # Example usage - try multiple possible paths for the ROOT file
    possible_paths = [
        "epicToyOutput.root",                # When run from build/ directory
        "build/epicToyOutput.root",          # When run from epicToy/ directory
        "./build/epicToyOutput.root",        # When run from epicToy/ directory
        "../../build/epicToyOutput.root"     # When run from python/proc/
    ]
    
    root_file = None
    for path in possible_paths:
        if os.path.exists(path):
            root_file = path
            break
    
    if root_file is None:
        print("Error: ROOT file 'epicToyOutput.root' not found!")
        print("Looked in the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Make sure you have run the Geant4 simulation first.")
        exit(1)
    
    print(f"Using ROOT file: {root_file}")
    
    # Configuration
    CHARGE_TYPE = 'fraction'       # 'fraction', 'value', or 'coulomb'
    OUTPUT_DIR = "gaussian_fits"
    
    print("="*70)
    print("GAUSSIAN FIT VISUALIZATION FROM ROOT FILE")
    print("="*70)
    print("This script reads pre-computed Gaussian fit results from the")
    print("C++ Geant4 simulation and creates visualization plots.")
    print("NOTE: Updated for new separated pixel/non-pixel data structure.")
    print("Gaussian fitting is only performed for non-pixel hits (distance > D0).")
    print("="*70)
    
    # Ask user what type of analysis to perform
    print("\nAvailable analysis options:")
    print("1. Sample plots (3 good + 3 bad fits) [DEFAULT]")
    print("2. Plot ALL events in the ROOT file")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3) [1]: ").strip()
        if not choice:
            choice = "1"
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        exit(0)
    
    if choice == "3":
        print("Exiting...")
        exit(0)
    elif choice == "2":
        # Plot ALL events
        print("\n" + "="*70)
        print("PLOTTING ALL EVENTS IN ROOT FILE")
        print("="*70)
        print("This will create individual plots for EVERY event in the ROOT file.")
        print("Depending on the number of events, this may take a while and create many files.")
        
        # Confirm before proceeding
        try:
            confirm = input("\nDo you want to proceed? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user.")
            exit(0)
        
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            exit(0)
        
        # Plot all events
        try:
            stats = plot_all_events_from_root(
                root_file, 
                charge_type=CHARGE_TYPE,
                save_plots=True, 
                output_dir="all_events",
                plot_style='simple'
            )
            
            print(f"\n{'='*70}")
            print("ALL EVENTS ANALYSIS COMPLETE")
            print(f"{'='*70}")
            print("Check the 'all_events' directory for organized plots by event type:")
            print("  - all_events/pixel_hit/         (pixel hits - no Gaussian fitting)")
            print("  - all_events/nonpixel_success/  (successful non-pixel fits)")
            print("  - all_events/nonpixel_failed/   (failed non-pixel fits)")
            
        except Exception as e:
            print(f"Error during all events analysis: {e}")
            exit(1)
            
    else:  # choice == "1" or default
        # Original sample analysis (3 good + 3 bad fits)
        # Load data to check what's available
        plotter = GaussianRootPlotter()
        try:
            data, detector_params = plotter.load_root_data(root_file)
            
            print(f"\nLoaded {len(data['EventID'])} events from ROOT file")
            
            # NEW: Analyze hit classification
            is_pixel_hit = data['IsPixelHit']
            is_within_d0 = data['IsWithinD0']
            fit_success = data['FitConstraintsSatisfied']
            
            # Find different categories of events
            pixel_hits = np.where(is_pixel_hit)[0]
            non_pixel_hits = np.where(~is_pixel_hit)[0]
            
            # For non-pixel hits, find successful and unsuccessful fits
            non_pixel_successful = np.where((~is_pixel_hit) & fit_success)[0]
            non_pixel_unsuccessful = np.where((~is_pixel_hit) & (~fit_success))[0]
            
            total_events = len(data['EventID'])
            pixel_count = len(pixel_hits)
            non_pixel_count = len(non_pixel_hits)
            non_pixel_success_count = len(non_pixel_successful)
            non_pixel_fail_count = len(non_pixel_unsuccessful)
            
            print(f"\nHit classification:")
            print(f"  Pixel hits (on pixel or distance <= D0):     {pixel_count}/{total_events} ({pixel_count/total_events*100:.1f}%)")
            print(f"  Non-pixel hits (distance > D0):              {non_pixel_count}/{total_events} ({non_pixel_count/total_events*100:.1f}%)")
            
            print(f"\nGaussian fit results (non-pixel hits only):")
            print(f"  Successful fits:       {non_pixel_success_count}/{non_pixel_count} ({non_pixel_success_count/non_pixel_count*100:.1f}% of non-pixel hits)")
            print(f"  Unsuccessful fits:     {non_pixel_fail_count}/{non_pixel_count} ({non_pixel_fail_count/non_pixel_count*100:.1f}% of non-pixel hits)")
            
            # Select events to plot: 3 good fits and 3 bad fits from non-pixel hits
            events_to_plot_success = []
            events_to_plot_fail = []
            
            if len(non_pixel_successful) > 0:
                n_success = min(3, len(non_pixel_successful))
                events_to_plot_success = np.random.choice(non_pixel_successful, size=n_success, replace=False)
                print(f"\nSelected {n_success} successful non-pixel fits: {events_to_plot_success.tolist()}")
            else:
                print("\nNo successful non-pixel fits found!")
                
            if len(non_pixel_unsuccessful) > 0:
                n_fail = min(3, len(non_pixel_unsuccessful))
                events_to_plot_fail = np.random.choice(non_pixel_unsuccessful, size=n_fail, replace=False)
                print(f"Selected {n_fail} unsuccessful non-pixel fits: {events_to_plot_fail.tolist()}")
            else:
                print("No unsuccessful non-pixel fits found!")
            
            if len(events_to_plot_success) == 0 and len(events_to_plot_fail) == 0:
                print("No non-pixel events to plot!")
                print("Note: Gaussian fitting is only performed for non-pixel hits (distance > D0)")
                if pixel_count > 0:
                    print(f"Found {pixel_count} pixel hits, but these don't have Gaussian fits")
                exit(1)
            
            # Create output directory
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            # Plot successful fits
            if len(events_to_plot_success) > 0:
                print(f"\n{'='*70}")
                print("PLOTTING SUCCESSFUL GAUSSIAN FITS (Non-pixel hits with FitConstraintsSatisfied = True)")
                print(f"{'='*70}")
                
                for i, event_idx in enumerate(events_to_plot_success):
                    print(f"\nPlotting successful fit {i+1}/{len(events_to_plot_success)}: Event {event_idx}")
                    
                    # Print event classification info
                    distance = data['PixelTrueDistance'][event_idx]
                    print(f"  Event classification: Non-pixel hit (distance = {distance:.6f} mm > D0)")
                    print(f"  IsPixelHit: {data['IsPixelHit'][event_idx]}")
                    print(f"  IsWithinD0: {data['IsWithinD0'][event_idx]}")
                    
                    # Print fit summary for successful events
                    plotter.print_fit_summary(event_idx, data)
                    
                    # Create plot with indication that this is a successful fit
                    plot_single_event_from_root(root_file, event_idx, CHARGE_TYPE,
                                               save_plots=True, output_dir=OUTPUT_DIR,
                                               plot_style='simple')
            
            # Plot unsuccessful fits  
            if len(events_to_plot_fail) > 0:
                print(f"\n{'='*70}")
                print("PLOTTING UNSUCCESSFUL GAUSSIAN FITS (Non-pixel hits with FitConstraintsSatisfied = False)")
                print(f"{'='*70}")
                
                for i, event_idx in enumerate(events_to_plot_fail):
                    print(f"\nPlotting unsuccessful fit {i+1}/{len(events_to_plot_fail)}: Event {event_idx}")
                    print(f"WARNING: This event has FitConstraintsSatisfied = False")
                    
                    # Print event classification info
                    distance = data['PixelTrueDistance'][event_idx]
                    print(f"  Event classification: Non-pixel hit (distance = {distance:.6f} mm > D0)")
                    print(f"  IsPixelHit: {data['IsPixelHit'][event_idx]}")
                    print(f"  IsWithinD0: {data['IsWithinD0'][event_idx]}")
                    
                    # For unsuccessful fits, we'll still try to plot but with warnings
                    try:
                        # Create a simple plot showing the data even if fit failed
                        plotter.plot_simple_fit_results(event_idx, data, detector_params, CHARGE_TYPE, 
                                                       save_plot=True, output_dir=OUTPUT_DIR, 
                                                       root_filename=root_file)
                        
                        # Print basic event info
                        print(f"Event {event_idx} basic info:")
                        print(f"  True Position: ({data['TrueX'][event_idx]:.6f}, {data['TrueY'][event_idx]:.6f}) mm")
                        print(f"  Pixel Position: ({data['PixelX'][event_idx]:.6f}, {data['PixelY'][event_idx]:.6f}) mm")
                        print(f"  Pixel-True Distance: {data['PixelTrueDistance'][event_idx]:.6f} mm")
                        print(f"  Distance to Pixel Center: {data['PixelTrueDistance'][event_idx]:.6f} mm")
                        print(f"  Fit Parameters (may be unreliable):")
                        print(f"    Amplitude: {data['FitAmplitude'][event_idx]:.6f}")
                        print(f"    Center: ({data['FitX0'][event_idx]:.6f}, {data['FitY0'][event_idx]:.6f}) mm")
                        print(f"    Sigmas: ({data['FitSigmaX'][event_idx]:.6f}, {data['FitSigmaY'][event_idx]:.6f}) mm")
                        print(f"    Chi²/NDF: {data['FitChi2red'][event_idx]:.6f}")
                        
                    except Exception as e:
                        print(f"Error plotting unsuccessful event {event_idx}: {e}")
                        continue
            
            # Summary
            print(f"\n{'='*70}")
            print("SAMPLE PLOTTING COMPLETE")
            print(f"{'='*70}")
            print(f"Generated plots for:")
            if len(events_to_plot_success) > 0:
                print(f"  {len(events_to_plot_success)} successful non-pixel fits: {events_to_plot_success.tolist()}")
            if len(events_to_plot_fail) > 0:
                print(f"  {len(events_to_plot_fail)} unsuccessful non-pixel fits: {events_to_plot_fail.tolist()}")
            print(f"\nAll plots saved to: {OUTPUT_DIR}")
            print(f"\nFile naming convention:")
            print(f"  - Successful fits: 'successful_fit_event_<ID>_success_<timestamp>.png'")
            print(f"  - Failed fits: 'failed_fit_event_<ID>_failed_<timestamp>.png'")
            print(f"\nSuccessful fits show:")
            print(f"  - Reliable Gaussian fit parameters from C++ ROOT analysis")
            print(f"  - High-quality fit statistics and residuals")
            print(f"  - Accurate distance measurements")
            print(f"  - Green title indicating FitConstraintsSatisfied=True")
            print(f"  - Only non-pixel hits (distance > D0) are fitted")
            print(f"\nFailed fits show:")
            print(f"  - Raw data visualization (may use Python fallback fitting)")
            print(f"  - Unreliable fit parameters (marked with warnings)")
            print(f"  - Red title indicating FitConstraintsSatisfied=False")
            print(f"  - Indication of why the fit may have failed")
            print(f"\nData structure notes:")
            print(f"  - Pixel hits (distance <= D0 or on pixel): No Gaussian fitting performed")
            print(f"  - Non-pixel hits (distance > D0): Gaussian fitting attempted")
            print(f"  - Hit classification stored in IsPixelHit, IsWithinD0, PixelTrueDistance")
            print(f"  - Charge data for non-pixel hits stored with NonPixel_ prefix")
        
        except Exception as e:
            print(f"Error loading ROOT file: {e}")
            print("Make sure the ROOT file contains the required fit result branches.")
            exit(1)