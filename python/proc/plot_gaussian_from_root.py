#!/usr/bin/env python3
"""
Visualization of 3D Gaussian fitting results from ROOT file.
Reads pre-computed Gaussian fit parameters from Geant4 simulation and creates plots.
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
                    
                    # Charge data
                    'GridNeighborhoodChargeFractions': tree['GridNeighborhoodChargeFractions'].array(library="np"),
                    'GridNeighborhoodDistances': tree['GridNeighborhoodDistances'].array(library="np"),
                    'GridNeighborhoodChargeValues': tree['GridNeighborhoodChargeValues'].array(library="np"),
                    
                    # Fit results (all data - ONLY type available, outlier functionality removed)
                    'FitAmplitude_alldata': tree['FitAmplitude_alldata'].array(library="np"),
                    'FitX0_alldata': tree['FitX0_alldata'].array(library="np"),
                    'FitY0_alldata': tree['FitY0_alldata'].array(library="np"),
                    'FitSigmaX_alldata': tree['FitSigmaX_alldata'].array(library="np"),
                    'FitSigmaY_alldata': tree['FitSigmaY_alldata'].array(library="np"),
                    'FitTheta_alldata': tree['FitTheta_alldata'].array(library="np"),
                    'FitOffset_alldata': tree['FitOffset_alldata'].array(library="np"),
                    
                    # Fit errors (all data)
                    'FitAmplitudeErr_alldata': tree['FitAmplitudeErr_alldata'].array(library="np"),
                    'FitX0Err_alldata': tree['FitX0Err_alldata'].array(library="np"),
                    'FitY0Err_alldata': tree['FitY0Err_alldata'].array(library="np"),
                    'FitSigmaXErr_alldata': tree['FitSigmaXErr_alldata'].array(library="np"),
                    'FitSigmaYErr_alldata': tree['FitSigmaYErr_alldata'].array(library="np"),
                    'FitThetaErr_alldata': tree['FitThetaErr_alldata'].array(library="np"),
                    'FitOffsetErr_alldata': tree['FitOffsetErr_alldata'].array(library="np"),
                    
                    # Fit statistics (all data)
                    'FitChi2_alldata': tree['FitChi2_alldata'].array(library="np"),
                    'FitNDF_alldata': tree['FitNDF_alldata'].array(library="np"),
                    'FitProb_alldata': tree['FitProb_alldata'].array(library="np"),
                    'FitRSquared_alldata': tree['FitRSquared_alldata'].array(library="np"),
                    'FitNPoints_alldata': tree['FitNPoints_alldata'].array(library="np"),
                    'FitSuccessful_alldata': tree['FitSuccessful_alldata'].array(library="np"),
                    'FitResidualMean_alldata': tree['FitResidualMean_alldata'].array(library="np"),
                    'FitResidualStd_alldata': tree['FitResidualStd_alldata'].array(library="np"),
                    # 'FitNOutliersRemoved_alldata': tree['FitNOutliersRemoved_alldata'].array(library="np"), - COMMENTED OUT - outlier functionality removed
                    
                    # Event data
                    'EventID': tree['EventID'].array(library="np"),
                }
                
                # Try to load Coulomb data if available
                try:
                    data['GridNeighborhoodChargeCoulombs'] = tree['GridNeighborhoodChargeCoulombs'].array(library="np")
                except:
                    print("Warning: GridNeighborhoodChargeCoulombs not available in ROOT file")
                
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
        
        # Get charge data - exactly as in fit_gaussian.py extract_charge_data_for_fitting
        if charge_type == 'fraction':
            charge_data = data['GridNeighborhoodChargeFractions'][event_idx]
        elif charge_type == 'value':
            charge_data = data['GridNeighborhoodChargeValues'][event_idx]
        elif charge_type == 'coulomb':
            if 'GridNeighborhoodChargeCoulombs' in data:
                charge_data = data['GridNeighborhoodChargeCoulombs'][event_idx]
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
        Calculate residuals from stored fit parameters (ALL DATA only)
        """
        # Get charge coordinates
        x, y, z = self.get_charge_coordinates(event_idx, data, detector_params, charge_type)
        
        # Get fit parameters (using all-data fit since outlier functionality removed)
        fit_params = [
            data['FitAmplitude_alldata'][event_idx],
            data['FitX0_alldata'][event_idx],
            data['FitY0_alldata'][event_idx],
            data['FitSigmaX_alldata'][event_idx],
            data['FitSigmaY_alldata'][event_idx],
            data['FitTheta_alldata'][event_idx],
            data['FitOffset_alldata'][event_idx]
        ]
        
        # Calculate fitted values
        fitted_z = self.gaussian_3d(fit_params, (x, y))
        residuals = z - fitted_z
        
        return x, y, z, fitted_z, residuals

    def plot_simple_fit_results(self, event_idx, data, detector_params, charge_type='fraction', 
                               save_plot=True, output_dir="", root_filename=None):
        """
        Create simple 2-panel plot: Data+Fit and Residuals
        Now only works with ALL DATA fits (outlier functionality removed)
        """
        # Check if fit was successful (using all-data fit only)
        if not data['FitSuccessful_alldata'][event_idx]:
            print(f"Warning: Fit was not successful for event {event_idx}")
            return None
        
        # Get fit parameters from ROOT (all-data fit only)
        root_fit_params = [
            data['FitAmplitude_alldata'][event_idx],
            data['FitX0_alldata'][event_idx],
            data['FitY0_alldata'][event_idx],
            data['FitSigmaX_alldata'][event_idx],
            data['FitSigmaY_alldata'][event_idx],
            data['FitTheta_alldata'][event_idx],
            data['FitOffset_alldata'][event_idx]
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
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.style.use('classic')
        
        fig.patch.set_facecolor('white')
        for ax in axs:
            ax.set_facecolor('white')
        
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
        
        # Add colorbar with same height as plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('Charge', fontsize=12)
        
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
        
        # Add distance text annotation
        mid_x = (fit_params[1] + true_x) / 2
        mid_y = (fit_params[2] + true_y) / 2
        axs[0].annotate(f'{distance:.3f} mm', xy=(mid_x, mid_y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Add title indicating fallback if used
        title = f'Event {event_idx} - {"Python Fit (Fallback)" if fallback_used else "ROOT Fit"}'
        axs[0].set_title(title, fontsize=14)
        
        axs[0].set_xlabel('X (mm)', fontsize=14)
        axs[0].set_ylabel('Y (mm)', fontsize=14)
        axs[0].set_aspect('equal')
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
        axs[1].set_ylabel('Fitted - Observed', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        axs[1].ticklabel_format(style='plain', useOffset=False, axis='y')
        
        # Adjust formatting for both axes
        for ax in axs:
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = "python_fallback" if fallback_used else "gaussian_from_root"
            filename = f'{prefix}_simple_event_{event_idx}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Simple fit plot saved to {filename}")
        
        # plt.show()  # Commented out for non-interactive backend
        return fig

    def plot_3d_visualization(self, event_idx, data, detector_params, charge_type='fraction', 
                             save_plot=True, output_dir=""):
        """
        Create 3D visualization of the Gaussian fit results (ALL DATA only)
        """
        # Check if fit was successful (using all-data fit only)
        if not data['FitSuccessful_alldata'][event_idx]:
            print(f"Warning: Fit was not successful for event {event_idx}")
            return None
        
        # Get coordinates and residuals
        x, y, z, fitted_z, residuals = self.calculate_residuals(event_idx, data, detector_params, charge_type)
        
        # Get fit parameters (all-data fit only)
        fit_params = [
            data['FitAmplitude_alldata'][event_idx],
            data['FitX0_alldata'][event_idx],
            data['FitY0_alldata'][event_idx],
            data['FitSigmaX_alldata'][event_idx],
            data['FitSigmaY_alldata'][event_idx],
            data['FitTheta_alldata'][event_idx],
            data['FitOffset_alldata'][event_idx]
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
        ax1.set_zlabel('Charge', fontsize=12, labelpad=10)
        ax1.set_title('Original Data Points', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1, label='Charge')
        
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
        ax2.set_zlabel('Fitted Charge', fontsize=12, labelpad=10)
        ax2.set_title('Fitted 3D Gaussian Surface', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        plt.colorbar(surface, ax=ax2, shrink=0.6, pad=0.1, label='Fitted Charge')
        
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
        ax3.set_zlabel('Charge', fontsize=12, labelpad=10)
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
        ax4.set_zlabel('Residuals', fontsize=12, labelpad=10)
        ax4.set_title(f'Residuals (Std: {np.std(residuals):.6f})', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3)
        
        plt.colorbar(residual_scatter, ax=ax4, shrink=0.6, pad=0.1, label='Residuals')
        
        # Set consistent viewing angles
        for ax in [ax1, ax2, ax3, ax4]:
            ax.view_init(elev=20, azim=45)
            ax.tick_params(labelsize=10)
        
        # Add overall title
        fig.suptitle(f'3D Gaussian Fit Visualization - Event {event_idx} (from ROOT)\n'
                    f'R² = {data["FitRSquared_alldata"][event_idx]:.6f}, χ²/NDF = {data["FitChi2_alldata"][event_idx]/data["FitNDF_alldata"][event_idx]:.6f}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
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
        
        # plt.show()  # Commented out for non-interactive backend
        return fig

    def print_fit_summary(self, event_idx, data):
        """
        Print fit summary for an event (ALL DATA only - outlier functionality removed)
        """
        print(f"\n{'='*70}")
        print(f"GAUSSIAN FIT SUMMARY - EVENT {event_idx} (ALL DATA)")
        print(f"{'='*70}")
        
        all_data_success = data['FitSuccessful_alldata'][event_idx]
        
        if not all_data_success:
            print("Fit was NOT successful!")
            return
        
        # Print fit parameters for all-data fit only
        print(f"{'Parameter':<20} {'Value':<20} {'Error':<20}")
        print(f"{'-'*60}")
        
        param_names = [
            ('Amplitude', 'FitAmplitude_alldata', 'FitAmplitudeErr_alldata'),
            ('X Center [mm]', 'FitX0_alldata', 'FitX0Err_alldata'),
            ('Y Center [mm]', 'FitY0_alldata', 'FitY0Err_alldata'),
            ('Sigma X [mm]', 'FitSigmaX_alldata', 'FitSigmaXErr_alldata'),
            ('Sigma Y [mm]', 'FitSigmaY_alldata', 'FitSigmaYErr_alldata'),
            ('Rotation [rad]', 'FitTheta_alldata', 'FitThetaErr_alldata'),
            ('Offset', 'FitOffset_alldata', 'FitOffsetErr_alldata')
        ]
        
        for param_name, val_key, err_key in param_names:
            val = data[val_key][event_idx]
            err = data[err_key][event_idx]
            print(f"{param_name:<20} {val:<20.6f} {err:<20.6f}")
        
        print(f"\n{'='*70}")
        print("FIT STATISTICS (ALL DATA)")
        print(f"{'='*70}")
        
        # Statistics for all-data fit only
        stat_names = [
            ('R-squared', 'FitRSquared_alldata'),
            ('Chi-squared', 'FitChi2_alldata'),
            ('Degrees of Freedom', 'FitNDF_alldata'),
            ('Fit Probability', 'FitProb_alldata'),
            ('Data Points', 'FitNPoints_alldata'),
            ('Residual Mean', 'FitResidualMean_alldata'),
            ('Residual Std', 'FitResidualStd_alldata')
        ]
        
        for stat_name, all_data_key in stat_names:
            all_data_val = data[all_data_key][event_idx]
            if stat_name == 'Degrees of Freedom' or stat_name == 'Data Points':
                print(f"{stat_name:<20} {int(all_data_val)}")
            else:
                print(f"{stat_name:<20} {all_data_val:.6f}")
        
        # Reduced Chi-squared calculation
        chi2_red_all = data['FitChi2_alldata'][event_idx]/data['FitNDF_alldata'][event_idx] if data['FitNDF_alldata'][event_idx] > 0 else 0
        print(f"{'Reduced Chi-squared':<20} {chi2_red_all:.6f}")
        
        print(f"\n{'='*70}")
        print("POSITION COMPARISON")
        print(f"{'='*70}")
        
        true_x = data['TrueX'][event_idx]
        true_y = data['TrueY'][event_idx]
        
        print(f"True Position:       ({true_x:.6f}, {true_y:.6f}) mm")
        
        fit_x_all = data['FitX0_alldata'][event_idx]
        fit_y_all = data['FitY0_alldata'][event_idx]
        distance_all = np.sqrt((fit_x_all - true_x)**2 + (fit_y_all - true_y)**2)
        print(f"All Data Fit:        ({fit_x_all:.6f}, {fit_y_all:.6f}) mm, Distance: {distance_all:.6f} mm")
        
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
        
        # Get charge data based on type - exactly like fit_gaussian.py
        if charge_type == 'fraction':
            charge_data = generator.data['GridNeighborhoodChargeFractions'][event_idx]
        elif charge_type == 'value':
            charge_data = generator.data['GridNeighborhoodChargeValues'][event_idx]
        elif charge_type == 'coulomb':
            if 'GridNeighborhoodChargeCoulombs' not in generator.data:
                raise ValueError("Coulomb charge data not available in this ROOT file")
            charge_data = generator.data['GridNeighborhoodChargeCoulombs'][event_idx]
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

    def plot_dual_fit_comparison(self, event_idx, data, detector_params, charge_type='fraction', 
                                save_plot=True, output_dir="", root_filename=None):
        """
        Create dual comparison plot showing both fits: outliers removed vs all data
        """
        # Check if both fits were successful
        outliers_removed_success = data['FitSuccessful'][event_idx]
        all_data_success = data['FitSuccessful_alldata'][event_idx]
        
        if not (outliers_removed_success or all_data_success):
            print(f"Warning: Both fits failed for event {event_idx}")
            return None
        
        # Get charge coordinates using the same method as before
        if root_filename is not None:
            try:
                x, y, z, pixel_x, pixel_y, true_x, true_y = self.extract_charge_data_like_fit_gaussian(root_filename, event_idx, charge_type)
            except ImportError:
                print("Warning: charge_sim module not available, falling back to direct data extraction")
                x, y, z = self.get_charge_coordinates(event_idx, data, detector_params, charge_type)
                true_x = data['TrueX'][event_idx]
                true_y = data['TrueY'][event_idx]
        else:
            x, y, z = self.get_charge_coordinates(event_idx, data, detector_params, charge_type)
            true_x = data['TrueX'][event_idx]
            true_y = data['TrueY'][event_idx]
        
        # Get fit parameters for both fits
        outliers_removed_params = [
            data['FitAmplitude'][event_idx],
            data['FitX0'][event_idx],
            data['FitY0'][event_idx],
            data['FitSigmaX'][event_idx],
            data['FitSigmaY'][event_idx],
            data['FitTheta'][event_idx],
            data['FitOffset'][event_idx]
        ]
        
        all_data_params = [
            data['FitAmplitude_alldata'][event_idx],
            data['FitX0_alldata'][event_idx],
            data['FitY0_alldata'][event_idx],
            data['FitSigmaX_alldata'][event_idx],
            data['FitSigmaY_alldata'][event_idx],
            data['FitTheta_alldata'][event_idx],
            data['FitOffset_alldata'][event_idx]
        ]
        
        # Create figure with 2 subplots (side by side)
        plt.close('all')
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        plt.style.use('classic')
        
        fig.patch.set_facecolor('white')
        for ax in axs:
            ax.set_facecolor('white')
        
        # Common grid for smooth contour plots
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_grid = np.linspace(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range, 50)
        y_grid = np.linspace(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Left panel: Outliers Removed Fit
        if outliers_removed_success:
            Z_fitted_smooth = self.gaussian_3d(outliers_removed_params, (X_grid, Y_grid))
            
            contour = axs[0].contourf(X_grid, Y_grid, Z_fitted_smooth, levels=20, cmap='viridis', alpha=0.8)
            contour_lines = axs[0].contour(X_grid, Y_grid, Z_fitted_smooth, levels=12, colors='white', alpha=0.8, linewidths=0.8)
            
            # Plot data points
            scatter = axs[0].scatter(x, y, c=z, s=120, cmap='viridis', edgecolors='white', linewidth=2, 
                                   alpha=1.0, label='Data Points', zorder=5)
            
            # Add colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(contour, cax=cax)
            cbar.set_label('Charge', fontsize=12)
            
            # Add fit center and true position
            axs[0].plot(outliers_removed_params[1], outliers_removed_params[2], '+', color='red', markersize=15, 
                       markeredgewidth=4, label='Fit Center')
            axs[0].plot(true_x, true_y, 'x', color='orange', markersize=12, markeredgewidth=4, 
                       label='True Position')
            
            # Distance line and annotation
            distance_outliers = np.sqrt((outliers_removed_params[1] - true_x)**2 + (outliers_removed_params[2] - true_y)**2)
            axs[0].plot([outliers_removed_params[1], true_x], [outliers_removed_params[2], true_y], 
                       'r--', linewidth=2, alpha=0.7)
            
            mid_x = (outliers_removed_params[1] + true_x) / 2
            mid_y = (outliers_removed_params[2] + true_y) / 2
            axs[0].annotate(f'{distance_outliers:.3f} mm', xy=(mid_x, mid_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
            # Stats text
            n_outliers = data['FitNOutliersRemoved'][event_idx]
            r2 = data['FitRSquared'][event_idx]
            chi2_red = data['FitChi2'][event_idx] / data['FitNDF'][event_idx] if data['FitNDF'][event_idx] > 0 else 0
            
            stats_text = f'Outliers Removed: {n_outliers}\nR² = {r2:.4f}\nχ²/NDF = {chi2_red:.4f}'
            axs[0].text(0.02, 0.98, stats_text, transform=axs[0].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axs[0].text(0.5, 0.5, 'Outliers Removed\nFit Failed', transform=axs[0].transAxes, 
                       fontsize=16, ha='center', va='center', 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        axs[0].set_title('Outliers Removed Fit', fontsize=14)
        axs[0].set_xlabel('X (mm)', fontsize=14)
        axs[0].set_ylabel('Y (mm)', fontsize=14)
        axs[0].set_aspect('equal')
        axs[0].grid(True, alpha=0.3)
        if outliers_removed_success:
            axs[0].legend(loc='upper right')
        
        # Right panel: All Data Fit
        if all_data_success:
            Z_fitted_smooth_all = self.gaussian_3d(all_data_params, (X_grid, Y_grid))
            
            contour_all = axs[1].contourf(X_grid, Y_grid, Z_fitted_smooth_all, levels=20, cmap='plasma', alpha=0.8)
            contour_lines_all = axs[1].contour(X_grid, Y_grid, Z_fitted_smooth_all, levels=12, colors='white', alpha=0.8, linewidths=0.8)
            
            # Plot data points
            scatter_all = axs[1].scatter(x, y, c=z, s=120, cmap='plasma', edgecolors='white', linewidth=2, 
                                       alpha=1.0, label='Data Points', zorder=5)
            
            # Add colorbar
            divider_all = make_axes_locatable(axs[1])
            cax_all = divider_all.append_axes("right", size="5%", pad=0.1)
            cbar_all = plt.colorbar(contour_all, cax=cax_all)
            cbar_all.set_label('Charge', fontsize=12)
            
            # Add fit center and true position
            axs[1].plot(all_data_params[1], all_data_params[2], '+', color='red', markersize=15, 
                       markeredgewidth=4, label='Fit Center')
            axs[1].plot(true_x, true_y, 'x', color='orange', markersize=12, markeredgewidth=4, 
                       label='True Position')
            
            # Distance line and annotation
            distance_all = np.sqrt((all_data_params[1] - true_x)**2 + (all_data_params[2] - true_y)**2)
            axs[1].plot([all_data_params[1], true_x], [all_data_params[2], true_y], 
                       'r--', linewidth=2, alpha=0.7)
            
            mid_x_all = (all_data_params[1] + true_x) / 2
            mid_y_all = (all_data_params[2] + true_y) / 2
            axs[1].annotate(f'{distance_all:.3f} mm', xy=(mid_x_all, mid_y_all), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
            # Stats text
            r2_all = data['FitRSquared_alldata'][event_idx]
            chi2_red_all = data['FitChi2_alldata'][event_idx] / data['FitNDF_alldata'][event_idx] if data['FitNDF_alldata'][event_idx] > 0 else 0
            n_points_all = data['FitNPoints_alldata'][event_idx]
            
            stats_text_all = f'All Data: {n_points_all} points\nR² = {r2_all:.4f}\nχ²/NDF = {chi2_red_all:.4f}'
            axs[1].text(0.02, 0.98, stats_text_all, transform=axs[1].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axs[1].text(0.5, 0.5, 'All Data\nFit Failed', transform=axs[1].transAxes, 
                       fontsize=16, ha='center', va='center', 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        axs[1].set_title('All Data Fit', fontsize=14)
        axs[1].set_xlabel('X (mm)', fontsize=14)
        axs[1].set_ylabel('Y (mm)', fontsize=14)
        axs[1].set_aspect('equal')
        axs[1].grid(True, alpha=0.3)
        if all_data_success:
            axs[1].legend(loc='upper right')
        
        # Adjust formatting for both axes
        for ax in axs:
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            ax.tick_params(labelsize=12)
        
        # Overall title with comparison information
        fit_comparison_text = ""
        if outliers_removed_success and all_data_success:
            distance_improvement = distance_all - distance_outliers
            improvement_percent = (distance_improvement / distance_all * 100) if distance_all > 0 else 0
            fit_comparison_text = f"Distance Improvement: {distance_improvement:.3f} mm ({improvement_percent:+.1f}%)"
        
        fig.suptitle(f'Dual Fit Comparison - Event {event_idx}\n{fit_comparison_text}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'dual_fit_comparison_event_{event_idx}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Dual fit comparison plot saved to {filename}")
        
        # plt.show()  # Commented out for non-interactive backend
        return fig

    def analyze_outlier_patterns(self, event_idx, data, detector_params, charge_type='fraction', verbose=True):
        """
        Analyze patterns in outlier locations within the 9x9 grid
        """
        # Get charge coordinates and identify outliers
        x, y, z = self.get_charge_coordinates(event_idx, data, detector_params, charge_type)
        outlier_mask, outlier_info = self.identify_outliers_like_cpp(x, y, z, verbose=verbose)
        
        # Get pixel information
        pixel_x = data['PixelX'][event_idx]
        pixel_y = data['PixelY'][event_idx]
        pixel_spacing = detector_params['pixel_spacing']
        
        # Create 9x9 grid analysis
        grid_outliers = np.zeros((9, 9), dtype=bool)
        grid_values = np.full((9, 9), np.nan)
        grid_positions = np.full((9, 9, 2), np.nan)  # Store x,y positions
        
        # Map points to grid positions
        for i, (xi, yi, zi, is_outlier) in enumerate(zip(x, y, z, outlier_mask)):
            # Calculate grid indices (same logic as charge coordinate extraction)
            rel_x = xi - pixel_x
            rel_y = yi - pixel_y
            
            # Convert to grid indices
            j = int(round(rel_x / pixel_spacing)) + 4  # j is column (x direction)
            i_grid = int(round(rel_y / pixel_spacing)) + 4  # i is row (y direction)
            
            if 0 <= i_grid < 9 and 0 <= j < 9:
                grid_outliers[i_grid, j] = is_outlier
                grid_values[i_grid, j] = zi
                grid_positions[i_grid, j] = [xi, yi]
        
        # Analyze patterns
        analysis = {
            'total_points': len(z),
            'n_outliers': outlier_info['n_outliers'],
            'outlier_percentage': outlier_info['n_outliers'] / len(z) * 100,
            'grid_outliers': grid_outliers,
            'grid_values': grid_values,
            'grid_positions': grid_positions,
            'outlier_info': outlier_info
        }
        
        # Distance from center analysis
        center_distances = []
        edge_distances = []
        outlier_distances = []
        
        for i_grid in range(9):
            for j in range(9):
                if not np.isnan(grid_values[i_grid, j]):
                    # Distance from center of 9x9 grid
                    center_dist = np.sqrt((i_grid - 4)**2 + (j - 4)**2)
                    # Distance from edge (minimum distance to any edge)
                    edge_dist = min(i_grid, j, 8 - i_grid, 8 - j)
                    
                    center_distances.append(center_dist)
                    edge_distances.append(edge_dist)
                    
                    if grid_outliers[i_grid, j]:
                        outlier_distances.append(center_dist)
        
        analysis.update({
            'center_distances': np.array(center_distances),
            'edge_distances': np.array(edge_distances),
            'outlier_center_distances': np.array(outlier_distances) if outlier_distances else np.array([]),
        })
        
        # Pattern analysis
        if outlier_info['n_outliers'] > 0:
            # Check if outliers are more common at edges
            avg_center_dist_all = np.mean(center_distances)
            avg_center_dist_outliers = np.mean(outlier_distances) if len(outlier_distances) > 0 else 0
            
            # Count outliers by ring
            rings = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 0=center, 4=corners
            ring_totals = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            for i_grid in range(9):
                for j in range(9):
                    if not np.isnan(grid_values[i_grid, j]):
                        ring = int(max(abs(i_grid - 4), abs(j - 4)))  # Chebyshev distance
                        ring_totals[ring] += 1
                        if grid_outliers[i_grid, j]:
                            rings[ring] += 1
            
            analysis.update({
                'avg_center_dist_all': avg_center_dist_all,
                'avg_center_dist_outliers': avg_center_dist_outliers,
                'outliers_by_ring': rings,
                'totals_by_ring': ring_totals,
                'outlier_fraction_by_ring': {ring: rings[ring]/ring_totals[ring] if ring_totals[ring] > 0 else 0 
                                           for ring in rings.keys()}
            })
        
        if verbose:
            self.print_outlier_analysis(analysis, event_idx)
        
        return analysis

    def print_outlier_analysis(self, analysis, event_idx):
        """
        Print detailed outlier pattern analysis
        """
        print(f"\n{'='*60}")
        print(f"OUTLIER PATTERN ANALYSIS - EVENT {event_idx}")
        print(f"{'='*60}")
        
        print(f"Total data points: {analysis['total_points']}")
        print(f"Outliers found: {analysis['n_outliers']} ({analysis['outlier_percentage']:.1f}%)")
        print(f"Points remaining after outlier removal: {analysis['total_points'] - analysis['n_outliers']}")
        
        if analysis['n_outliers'] > 0:
            print(f"\nOUTLIER DETECTION STATISTICS:")
            info = analysis['outlier_info']
            print(f"Charge median: {info['median']:.6f}")
            print(f"MAD (Median Absolute Deviation): {info['mad']:.6f}")
            print(f"Outlier threshold: {info['threshold_used']:.6f}")
            print(f"Threshold multiplier: {info['outlier_threshold']:.1f} sigma")
            
            print(f"\nSPATIAL PATTERN ANALYSIS:")
            print(f"Average distance from grid center (all points): {analysis['avg_center_dist_all']:.2f}")
            print(f"Average distance from grid center (outliers): {analysis['avg_center_dist_outliers']:.2f}")
            
            if analysis['avg_center_dist_outliers'] > analysis['avg_center_dist_all']:
                print("→ Outliers tend to be FARTHER from center")
            else:
                print("→ Outliers tend to be CLOSER to center")
            
            print(f"\nOUTLIERS BY RING (distance from center):")
            print(f"{'Ring':<8} {'Count':<8} {'Total':<8} {'Fraction':<10} {'Description'}")
            print("-" * 50)
            
            ring_descriptions = {
                0: "Center pixel",
                1: "Adjacent to center", 
                2: "Second ring",
                3: "Third ring",
                4: "Corner pixels"
            }
            
            for ring in sorted(analysis['outliers_by_ring'].keys()):
                count = analysis['outliers_by_ring'][ring]
                total = analysis['totals_by_ring'][ring]
                fraction = analysis['outlier_fraction_by_ring'][ring]
                desc = ring_descriptions.get(ring, f"Ring {ring}")
                
                print(f"{ring:<8} {count:<8} {total:<8} {fraction:<10.3f} {desc}")
            
            # Find ring with highest outlier fraction
            max_ring = max(analysis['outlier_fraction_by_ring'].keys(), 
                          key=lambda k: analysis['outlier_fraction_by_ring'][k])
            max_fraction = analysis['outlier_fraction_by_ring'][max_ring]
            
            if max_fraction > 0:
                print(f"\nHighest outlier concentration: Ring {max_ring} ({ring_descriptions.get(max_ring, f'Ring {max_ring}')}) with {max_fraction:.1%}")
            
        else:
            print("\nNo outliers detected for this event.")
        
        print(f"{'='*60}")

    def plot_outlier_grid_visualization(self, event_idx, data, detector_params, charge_type='fraction',
                                       save_plot=True, output_dir=""):
        """
        Create visualization of the 9x9 grid showing outlier locations
        """
        analysis = self.analyze_outlier_patterns(event_idx, data, detector_params, charge_type, verbose=False)
        
        if analysis['n_outliers'] == 0:
            print(f"No outliers found for event {event_idx} - skipping grid visualization")
            return None
        
        # Create figure with multiple subplots
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.style.use('classic')
        fig.patch.set_facecolor('white')
        
        grid_values = analysis['grid_values']
        grid_outliers = analysis['grid_outliers']
        
        # Create masks for visualization
        valid_mask = ~np.isnan(grid_values)
        outlier_values = np.where(grid_outliers & valid_mask, grid_values, np.nan)
        clean_values = np.where(~grid_outliers & valid_mask, grid_values, np.nan)
        
        # Subplot 1: All data with outliers highlighted
        ax1 = axes[0, 0]
        im1 = ax1.imshow(grid_values, cmap='viridis', origin='lower', interpolation='nearest')
        
        # Overlay outlier markers
        outlier_positions = np.where(grid_outliers & valid_mask)
        if len(outlier_positions[0]) > 0:
            ax1.scatter(outlier_positions[1], outlier_positions[0], 
                       s=200, c='red', marker='x', linewidth=3, label=f'Outliers ({analysis["n_outliers"]})')
        
        ax1.set_title('All Data with Outliers Marked', fontsize=14)
        ax1.set_xlabel('Grid Column (X direction)', fontsize=12)
        ax1.set_ylabel('Grid Row (Y direction)', fontsize=12)
        ax1.legend()
        
        # Add grid lines and labels
        ax1.set_xticks(range(9))
        ax1.set_yticks(range(9))
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Charge Value')
        
        # Subplot 2: Only outliers
        ax2 = axes[0, 1]
        im2 = ax2.imshow(outlier_values, cmap='Reds', origin='lower', interpolation='nearest')
        ax2.set_title('Outliers Only', fontsize=14)
        ax2.set_xlabel('Grid Column (X direction)', fontsize=12)
        ax2.set_ylabel('Grid Row (Y direction)', fontsize=12)
        ax2.set_xticks(range(9))
        ax2.set_yticks(range(9))
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations for outlier values
        for i in range(9):
            for j in range(9):
                if grid_outliers[i, j] and valid_mask[i, j]:
                    ax2.text(j, i, f'{grid_values[i, j]:.4f}', 
                           ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        plt.colorbar(im2, ax=ax2, label='Outlier Charge Value')
        
        # Subplot 3: Clean data (outliers removed)
        ax3 = axes[1, 0]
        im3 = ax3.imshow(clean_values, cmap='viridis', origin='lower', interpolation='nearest')
        ax3.set_title('Clean Data (Outliers Removed)', fontsize=14)
        ax3.set_xlabel('Grid Column (X direction)', fontsize=12)
        ax3.set_ylabel('Grid Row (Y direction)', fontsize=12)
        ax3.set_xticks(range(9))
        ax3.set_yticks(range(9))
        ax3.grid(True, alpha=0.3)
        plt.colorbar(im3, ax=ax3, label='Charge Value')
        
        # Subplot 4: Distance from center analysis
        ax4 = axes[1, 1]
        
        # Create distance matrix
        distance_matrix = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                distance_matrix[i, j] = np.sqrt((i - 4)**2 + (j - 4)**2)
        
        im4 = ax4.imshow(distance_matrix, cmap='coolwarm', origin='lower', interpolation='nearest')
        
        # Overlay data points and outliers
        data_positions = np.where(valid_mask)
        ax4.scatter(data_positions[1], data_positions[0], 
                   s=100, c='black', marker='o', alpha=0.7, label='Data points')
        
        if len(outlier_positions[0]) > 0:
            ax4.scatter(outlier_positions[1], outlier_positions[0], 
                       s=200, c='red', marker='x', linewidth=3, label='Outliers')
        
        ax4.set_title('Distance from Center', fontsize=14)
        ax4.set_xlabel('Grid Column (X direction)', fontsize=12)
        ax4.set_ylabel('Grid Row (Y direction)', fontsize=12)
        ax4.set_xticks(range(9))
        ax4.set_yticks(range(9))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        plt.colorbar(im4, ax=ax4, label='Distance from Center')
        
        # Overall title with statistics
        outlier_info = analysis['outlier_info']
        fig.suptitle(f'Outlier Pattern Analysis - Event {event_idx}\n'
                    f'{analysis["n_outliers"]}/{analysis["total_points"]} outliers ({analysis["outlier_percentage"]:.1f}%) | '
                    f'Threshold: {outlier_info["outlier_threshold"]:.1f}σ | '
                    f'MAD: {outlier_info["mad"]:.6f}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'outlier_grid_analysis_event_{event_idx}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Outlier grid analysis plot saved to {filename}")
        
        # plt.show()  # Commented out for non-interactive backend
        return fig, analysis

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
        successful_events = np.where(data['FitSuccessful_alldata'])[0]
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
        
        if not data['FitSuccessful_alldata'][event_idx]:
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
            
            # Collect results for summary (using all-data fit results)
            true_x = data['TrueX'][event_idx]
            true_y = data['TrueY'][event_idx]
            fit_x = data['FitX0_alldata'][event_idx]
            fit_y = data['FitY0_alldata'][event_idx]
            distance = np.sqrt((fit_x - true_x)**2 + (fit_y - true_y)**2)
            
            results_summary.append({
                'event_idx': event_idx,
                'r_squared': data['FitRSquared_alldata'][event_idx],
                'chi2_reduced': data['FitChi2_alldata'][event_idx]/data['FitNDF_alldata'][event_idx],
                'distance_error': distance,
                'sigma_x': data['FitSigmaX_alldata'][event_idx],
                'sigma_y': data['FitSigmaY_alldata'][event_idx],
                'n_points': data['FitNPoints_alldata'][event_idx]
            })
            
        except Exception as e:
            print(f"Error processing event {event_idx}: {e}")
            continue
    
    # Print overall summary
    if results_summary:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY OF GAUSSIAN FITS FROM ROOT")
        print(f"{'='*70}")
        
        r_squared_values = [r['r_squared'] for r in results_summary]
        chi2_values = [r['chi2_reduced'] for r in results_summary]
        distance_errors = [r['distance_error'] for r in results_summary]
        sigma_x_values = [r['sigma_x'] for r in results_summary]
        sigma_y_values = [r['sigma_y'] for r in results_summary]
        
        print(f"Successfully processed {len(results_summary)} events")
        print(f"Average R-squared:      {np.mean(r_squared_values):.6f} ± {np.std(r_squared_values):.6f}")
        print(f"Average χ²/NDF:         {np.mean(chi2_values):.6f} ± {np.std(chi2_values):.6f}")
        print(f"Average Distance Error: {np.mean(distance_errors):.6f} ± {np.std(distance_errors):.6f} mm")
        print(f"Average σₓ:             {np.mean(sigma_x_values):.6f} ± {np.std(sigma_x_values):.6f} mm")
        print(f"Average σᵧ:             {np.mean(sigma_y_values):.6f} ± {np.std(sigma_y_values):.6f} mm")
        
        print(f"\nRange of Distance Errors: {np.min(distance_errors):.6f} to {np.max(distance_errors):.6f} mm")
        print(f"Range of R-squared:       {np.min(r_squared_values):.6f} to {np.max(r_squared_values):.6f}")
    
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
    
    if not data['FitSuccessful_alldata'][event_idx]:
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
    
    if not data['FitSuccessful_alldata'][event_idx]:
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
        
        cpp_params = {
            'amplitude': data['FitAmplitude_alldata'][event_idx],
            'x0': data['FitX0_alldata'][event_idx],
            'y0': data['FitY0_alldata'][event_idx],
            'sigma_x': data['FitSigmaX_alldata'][event_idx],
            'sigma_y': data['FitSigmaY_alldata'][event_idx],
            'theta': data['FitTheta_alldata'][event_idx],
            'offset': data['FitOffset_alldata'][event_idx],
            'r_squared': data['FitRSquared_alldata'][event_idx]
        }
        
        py_params = python_results['parameters']
        py_r_squared = python_results['fit_info']['r_squared']
        
        print(f"{'Parameter':<12} {'C++':<12} {'Python':<12} {'Difference':<12} {'Rel. Diff %':<12}")
        print("-" * 70)
        
        for param in ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']:
            cpp_val = cpp_params[param]
            py_val = py_params[param]
            diff = abs(cpp_val - py_val)
            rel_diff = diff / abs(py_val) * 100 if py_val != 0 else 0
            
            print(f"{param:<12} {cpp_val:<12.6f} {py_val:<12.6f} {diff:<12.6f} {rel_diff:<12.2f}")
        
        print(f"{'R-squared':<12} {cpp_params['r_squared']:<12.6f} {py_r_squared:<12.6f} "
              f"{abs(cpp_params['r_squared'] - py_r_squared):<12.6f} "
              f"{abs(cpp_params['r_squared'] - py_r_squared)/py_r_squared*100:<12.2f}")
        
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

def analyze_multiple_events_dual_fits(root_filename, event_indices=None, charge_type='fraction', 
                                     save_plots=True, output_dir=""):
    """
    Analyze multiple events from ROOT file and create dual fit comparison plots
    """
    print(f"Loading data from ROOT file: {root_filename}")
    
    # Create plotter and load data
    plotter = GaussianRootPlotter()
    data, detector_params = plotter.load_root_data(root_filename)
    
    print(f"Loaded {len(data['EventID'])} events from ROOT file")
    
    # Find events where at least one fit succeeded
    outliers_success = data['FitSuccessful']
    all_data_success = data['FitSuccessful_alldata']
    any_success = np.logical_or(outliers_success, all_data_success)
    successful_events = np.where(any_success)[0]
    
    if len(successful_events) == 0:
        print("No successful fits found in ROOT file!")
        return
    
    if event_indices is None:
        event_indices = successful_events[:min(10, len(successful_events))]
        print(f"Found {len(successful_events)} events with at least one successful fit, analyzing first {len(event_indices)}")
    
    # Create output directory
    if save_plots and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each event
    results_summary = []
    
    for i, event_idx in enumerate(event_indices):
        print(f"\n{'='*50}")
        print(f"Processing event {event_idx} ({i+1}/{len(event_indices)})")
        print(f"{'='*50}")
        
        outliers_removed_success = data['FitSuccessful'][event_idx]
        all_data_success = data['FitSuccessful_alldata'][event_idx]
        
        if not (outliers_removed_success or all_data_success):
            print(f"Skipping event {event_idx} - both fits failed")
            continue
        
        # Print fit summary
        plotter.print_fit_summary(event_idx, data)
        
        # Create dual comparison plot
        try:
            plotter.plot_dual_fit_comparison(event_idx, data, detector_params, charge_type, 
                                           save_plots, output_dir, root_filename)
            
            # Collect results for summary
            true_x = data['TrueX'][event_idx]
            true_y = data['TrueY'][event_idx]
            
            result_entry = {
                'event_idx': event_idx,
                'outliers_removed_success': outliers_removed_success,
                'all_data_success': all_data_success,
            }
            
            if outliers_removed_success:
                fit_x_outliers = data['FitX0'][event_idx]
                fit_y_outliers = data['FitY0'][event_idx]
                distance_outliers = np.sqrt((fit_x_outliers - true_x)**2 + (fit_y_outliers - true_y)**2)
                result_entry.update({
                    'r_squared_outliers': data['FitRSquared'][event_idx],
                    'chi2_reduced_outliers': data['FitChi2'][event_idx]/data['FitNDF'][event_idx],
                    'distance_error_outliers': distance_outliers,
                    'n_outliers_removed': data['FitNOutliersRemoved'][event_idx],
                    'n_points_outliers': data['FitNPoints'][event_idx]
                })
            
            if all_data_success:
                fit_x_all = data['FitX0_alldata'][event_idx]
                fit_y_all = data['FitY0_alldata'][event_idx]
                distance_all = np.sqrt((fit_x_all - true_x)**2 + (fit_y_all - true_y)**2)
                result_entry.update({
                    'r_squared_all': data['FitRSquared_alldata'][event_idx],
                    'chi2_reduced_all': data['FitChi2_alldata'][event_idx]/data['FitNDF_alldata'][event_idx],
                    'distance_error_all': distance_all,
                    'n_points_all': data['FitNPoints_alldata'][event_idx]
                })
            
            results_summary.append(result_entry)
            
        except Exception as e:
            print(f"Error processing event {event_idx}: {e}")
            continue
    
    # Print overall summary
    if results_summary:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY OF DUAL GAUSSIAN FITS FROM ROOT")
        print(f"{'='*80}")
        
        n_both_success = sum(1 for r in results_summary if r['outliers_removed_success'] and r['all_data_success'])
        n_outliers_only = sum(1 for r in results_summary if r['outliers_removed_success'] and not r['all_data_success'])
        n_all_data_only = sum(1 for r in results_summary if not r['outliers_removed_success'] and r['all_data_success'])
        
        print(f"Successfully processed {len(results_summary)} events")
        print(f"Both fits successful:      {n_both_success}")
        print(f"Only outliers fit successful: {n_outliers_only}")
        print(f"Only all-data fit successful: {n_all_data_only}")
        
        # Statistics for events where both fits succeeded
        if n_both_success > 0:
            both_success_results = [r for r in results_summary if r['outliers_removed_success'] and r['all_data_success']]
            
            outliers_distances = [r['distance_error_outliers'] for r in both_success_results]
            all_data_distances = [r['distance_error_all'] for r in both_success_results]
            improvements = [all_dist - out_dist for all_dist, out_dist in zip(all_data_distances, outliers_distances)]
            
            outliers_r2 = [r['r_squared_outliers'] for r in both_success_results]
            all_data_r2 = [r['r_squared_all'] for r in both_success_results]
            
            outliers_chi2 = [r['chi2_reduced_outliers'] for r in both_success_results]
            all_data_chi2 = [r['chi2_reduced_all'] for r in both_success_results]
            
            outliers_removed_counts = [r['n_outliers_removed'] for r in both_success_results]
            
            print(f"\nSTATISTICS FOR EVENTS WITH BOTH FITS SUCCESSFUL ({n_both_success} events):")
            print(f"Average outliers removed per event: {np.mean(outliers_removed_counts):.1f} ± {np.std(outliers_removed_counts):.1f}")
            print(f"Average distance error (outliers removed): {np.mean(outliers_distances):.6f} ± {np.std(outliers_distances):.6f} mm")
            print(f"Average distance error (all data):        {np.mean(all_data_distances):.6f} ± {np.std(all_data_distances):.6f} mm")
            print(f"Average improvement:                       {np.mean(improvements):.6f} ± {np.std(improvements):.6f} mm")
            print(f"Average R² (outliers removed):            {np.mean(outliers_r2):.6f} ± {np.std(outliers_r2):.6f}")
            print(f"Average R² (all data):                    {np.mean(all_data_r2):.6f} ± {np.std(all_data_r2):.6f}")
            print(f"Average χ²/NDF (outliers removed):        {np.mean(outliers_chi2):.6f} ± {np.std(outliers_chi2):.6f}")
            print(f"Average χ²/NDF (all data):                {np.mean(all_data_chi2):.6f} ± {np.std(all_data_chi2):.6f}")
            
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            negative_improvements = sum(1 for imp in improvements if imp < 0)
            no_change = sum(1 for imp in improvements if imp == 0)
            
            print(f"\nOUTLIER REMOVAL EFFECTIVENESS:")
            print(f"Improved fits:  {positive_improvements}/{n_both_success} ({positive_improvements/n_both_success*100:.1f}%)")
            print(f"Degraded fits:  {negative_improvements}/{n_both_success} ({negative_improvements/n_both_success*100:.1f}%)")
            print(f"No change:      {no_change}/{n_both_success} ({no_change/n_both_success*100:.1f}%)")
    
    return results_summary

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
    print("NOTE: Outlier functionality has been removed - only ALL DATA fits available.")
    print("="*70)
    
    # Load data to check what's available
    plotter = GaussianRootPlotter()
    try:
        data, detector_params = plotter.load_root_data(root_file)
        
        print(f"\nLoaded {len(data['EventID'])} events from ROOT file")
        
        # Find successful fits (all-data only)
        all_data_success = data['FitSuccessful_alldata']
        
        total_events = len(data['FitSuccessful_alldata'])
        all_data_count = np.sum(all_data_success)
        
        print(f"Fit success rates:")
        print(f"  All-data fits successful:       {all_data_count}/{total_events} ({all_data_count/total_events*100:.1f}%)")
        
        if all_data_count == 0:
            print("No successful fits found in ROOT file!")
            print("Make sure the Gaussian fitting was enabled in the C++ simulation.")
            exit(1)
        
        # Find events for demonstration
        success_events = np.where(all_data_success)[0]
        
        print(f"First 10 events with successful fits: {success_events[:10].tolist()}")
        
        # Demonstrate single event analysis
        demo_event = success_events[0]
        print(f"\nDemonstrating fit analysis for event {demo_event}:")
        
        # Single fit plot
        print(f"\n1. Creating fit visualization plot...")
        plot_single_event_from_root(root_file, demo_event, CHARGE_TYPE, 
                                   save_plots=True, output_dir=OUTPUT_DIR, 
                                   plot_style='simple')
        
        # Multiple events analysis
        print(f"\n2. Analyzing multiple events...")
        events_to_analyze = success_events[:min(5, len(success_events))]
        summary = analyze_multiple_events_from_root(root_file, events_to_analyze, CHARGE_TYPE, 
                                                  save_plots=True, output_dir=OUTPUT_DIR)
        
        print(f"\n{'='*70}")
        print("FIT ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Generated fit visualization plots from C++ Gaussian fits")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"\nFeatures available:")
        print(f"  - Shows fitted Gaussian surface with data points")
        print(f"  - Displays fit quality metrics (R², χ²/NDF)")
        print(f"  - Shows distance from true position")
        print(f"  - Creates residual analysis plots")
        print(f"  - Supports 3D visualization")
        print(f"\nUsage examples:")
        print(f"  # Single event simple plot:")
        print(f"  plot_single_event_from_root('{root_file}', {demo_event}, 'fraction', plot_style='simple')")
        print(f"  # Multiple events analysis:")
        print(f"  analyze_multiple_events_from_root('{root_file}', [{', '.join(map(str, events_to_analyze[:3]))}], 'fraction')")
    
    except Exception as e:
        print(f"Error loading ROOT file: {e}")
        print("Make sure the ROOT file contains the required fit result branches.")
        exit(1) 