#!/usr/bin/env python3
"""
3D Gaussian fitting using ODR (Orthogonal Distance Regression) for charge distribution data.
Fits 3D Gaussian functions to the grid charge sharing data from ROOT simulation files.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import odr
from scipy import stats
import uproot
import os
from charge_sim import RandomHitChargeGridGenerator

class Gaussian3DODR:
    """
    3D Gaussian fitting using Orthogonal Distance Regression (ODR)
    """
    
    def __init__(self):
        self.fit_result = None
        self.popt = None
        self.perr = None
        
    def gaussian_3d(self, params, coords):
        """
        3D Gaussian function for ODR fitting
        
        Parameters:
        -----------
        params : array-like
            [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
            amplitude: peak height
            x0, y0: center coordinates  
            sigma_x, sigma_y: standard deviations in rotated coordinate system
            theta: rotation angle in radians
            offset: background offset
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
    
    def fit_3d_gaussian_odr(self, x, y, z, z_err=None, initial_guess=None, verbose=True):
        """
        Fit 3D Gaussian using ODR with proper error propagation
        
        Parameters:
        -----------
        x, y, z : array-like
            Coordinate and data arrays
        z_err : array-like, optional
            Uncertainties in z values
        initial_guess : array-like, optional
            Initial parameter guess [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
        verbose : bool
            Print fit results
            
        Returns:
        --------
        dict : Fit results including parameters, errors, and statistics
        """
        # Flatten arrays if needed
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        
        if len(x) == 0:
            raise ValueError("No valid data points after removing NaN values")
        
        # Set up error estimates
        if z_err is None:
            # Estimate errors based on data noise
            z_err = np.full_like(z, np.std(z) * 0.1)  # 10% of data std as default error
        else:
            z_err = np.asarray(z_err).flatten()[valid_mask]
        
        # Initial parameter guess if not provided
        if initial_guess is None:
            amplitude_guess = np.max(z) - np.min(z)
            x0_guess = np.mean(x)
            y0_guess = np.mean(y)
            sigma_x_guess = np.std(x)
            sigma_y_guess = np.std(y)
            theta_guess = 0.0
            offset_guess = np.min(z)
            
            initial_guess = [amplitude_guess, x0_guess, y0_guess, 
                           sigma_x_guess, sigma_y_guess, theta_guess, offset_guess]
        
        if verbose:
            print(f"Initial guess: {initial_guess}")
            print(f"Data points: {len(x)}")
            print(f"Data range - X: [{np.min(x):.3f}, {np.max(x):.3f}], Y: [{np.min(y):.3f}, {np.max(y):.3f}], Z: [{np.min(z):.6f}, {np.max(z):.6f}]")
        
        # Set up ODR
        def odr_func(params, coords):
            return self.gaussian_3d(params, coords)
        
        # Create ODR model
        model = odr.Model(odr_func)
        
        # Create data object
        data = odr.RealData((x, y), z, sy=z_err)
        
        # Create ODR object
        odr_obj = odr.ODR(data, model, beta0=initial_guess)
        
        # Run ODR fit
        try:
            output = odr_obj.run()
            
            # Store results
            self.fit_result = output
            self.popt = output.beta
            self.perr = output.sd_beta
            
            # Calculate fit statistics
            fitted_z = self.gaussian_3d(self.popt, (x, y))
            residuals = z - fitted_z
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot)
            FitChi2red = ss_res / (len(z) - len(self.popt))
            
            # Prepare results dictionary
            param_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
            
            results = {
                'parameters': dict(zip(param_names, self.popt)),
                'parameter_errors': dict(zip(param_names, self.perr)),
                'fit_info': {
                    'FitChi2red': FitChi2red,
                    'FitPp': FitPp,
                    'residual_variance': output.res_var,
                    'sum_of_squares': output.sum_square,
                    'degrees_of_freedom': len(z) - len(self.popt),
                    'n_data_points': len(z)
                },
                'covariance_matrix': output.cov_beta,
                'fitted_values': fitted_z,
                'residuals': residuals,
                'data': {'x': x, 'y': y, 'z': z, 'z_err': z_err}
            }
            
            if verbose:
                self.print_fit_results(results)
            
            return results
            
        except Exception as e:
            print(f"ODR fitting failed: {e}")
            raise
    
    def print_fit_results(self, results):
        """Print formatted fit results"""
        print("\n" + "="*60)
        print("3D GAUSSIAN FIT RESULTS (ODR)")
        print("="*60)
        
        params = results['parameters']
        errors = results['parameter_errors']
        
        print(f"Amplitude:    {params['amplitude']:.6f} ± {errors['amplitude']:.6f}")
        print(f"X Center:     {params['x0']:.6f} ± {errors['x0']:.6f}")
        print(f"Y Center:     {params['y0']:.6f} ± {errors['y0']:.6f}")
        print(f"Sigma X:      {params['sigma_x']:.6f} ± {errors['sigma_x']:.6f}")
        print(f"Sigma Y:      {params['sigma_y']:.6f} ± {errors['sigma_y']:.6f}")
        print(f"Rotation:     {params['theta']:.6f} ± {errors['theta']:.6f} rad ({np.degrees(params['theta']):.2f}° ± {np.degrees(errors['theta']):.2f}°)")
        print(f"Offset:       {params['offset']:.6f} ± {errors['offset']:.6f}")
        
        print("\nFIT STATISTICS:")
        info = results['fit_info']
        print(f"R-squared:           {info['r_squared']:.6f}")
        print(f"Reduced Chi-squared: {info['FitChi2red']:.6f}")
        print(f"Residual Variance:   {info['residual_variance']:.6f}")
        print(f"Data Points:         {info['n_data_points']}")
        print(f"Degrees of Freedom:  {info['degrees_of_freedom']}")
        
    def plot_fit_results(self, results, save_plot=True, output_dir="", event_idx=None):
        """
        Create comprehensive scientific visualization of 3D Gaussian fit results
        Following scientific plotting standards with error bars, residuals, and statistics
        """
        # Set plotting style
        plt.style.use('classic')
        
        # Extract data
        x = results['data']['x']
        y = results['data']['y']
        z = results['data']['z']
        z_err = results['data']['z_err']
        fitted_z = results['fitted_values']
        residuals = results['residuals']
        
        # Extract fit parameters and statistics
        params = results['parameters']
        param_errors = results['parameter_errors']
        fit_info = results['fit_info']
        
        # Create grid for smooth surface plot
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_grid = np.linspace(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range, 50)
        y_grid = np.linspace(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_fitted_smooth = self.gaussian_3d(self.popt, (X_grid, Y_grid))
        
        # Create main figure with subplots (3x3 layout for comprehensive analysis)
        fig = plt.figure(figsize=(24, 18))
        fig.patch.set_facecolor('white')
        
        # 1. Original data with error bars (3D)
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1.set_facecolor('white')
        
        # 3D scatter with error bars (using stem plot for error representation)
        scatter1 = ax1.scatter(x, y, z, c=z, cmap='plasma', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add error bars in 3D (vertical lines showing uncertainty)
        for i in range(len(x)):
            ax1.plot([x[i], x[i]], [y[i], y[i]], [z[i]-z_err[i], z[i]+z_err[i]], 'k-', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        ax1.set_zlabel('Charge', fontsize=12)
        ax1.set_title('Original Data with Error Bars', fontsize=14)
        ax1.grid(True)
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='Charge')
        
        # 2. Fitted surface with data points (3D)
        ax2 = fig.add_subplot(3, 3, 2, projection='3d')
        ax2.set_facecolor('white')
        
        surface = ax2.plot_surface(X_grid, Y_grid, Z_fitted_smooth, cmap='plasma', alpha=0.6, 
                                 linewidth=0, antialiased=True)
        scatter2 = ax2.scatter(x, y, z, c='red', s=40, alpha=0.9, edgecolors='darkred', 
                             linewidth=1, label='Data Points')
        
        ax2.set_xlabel('X (mm)', fontsize=12)
        ax2.set_ylabel('Y (mm)', fontsize=12)
        ax2.set_zlabel('Charge', fontsize=12)
        ax2.set_title('3D Gaussian Fit', fontsize=14)
        ax2.grid(True)
        ax2.legend()
        plt.colorbar(surface, ax=ax2, shrink=0.6, label='Fitted Charge')
        
        # 3. Residuals (3D)
        ax3 = fig.add_subplot(3, 3, 3, projection='3d')
        ax3.set_facecolor('white')
        
        residual_scatter = ax3.scatter(x, y, residuals, c=residuals, cmap='RdBu_r', s=60, 
                                     edgecolors='black', linewidth=0.5)
        
        # Add horizontal plane at zero
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = np.zeros_like(xx)
        ax3.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
        
        ax3.set_xlabel('X (mm)', fontsize=12)
        ax3.set_ylabel('Y (mm)', fontsize=12)
        ax3.set_zlabel('Residuals', fontsize=12)
        ax3.set_title('Residuals Analysis (3D)', fontsize=14)
        ax3.grid(True)
        plt.colorbar(residual_scatter, ax=ax3, shrink=0.6, label='Residuals')
        
        # 4. 2D contour plot with data points and error bars
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.set_facecolor('white')
        
        contour = ax4.contourf(X_grid, Y_grid, Z_fitted_smooth, levels=15, cmap='plasma', alpha=0.8)
        contour_lines = ax4.contour(X_grid, Y_grid, Z_fitted_smooth, levels=10, colors='white', alpha=0.6, linewidths=1)
        
        # Add data points with error bars
        scatter4 = ax4.errorbar(x, y, xerr=0, yerr=0, fmt='o', color='red', markersize=6, 
                              markeredgecolor='darkred', markeredgewidth=1, 
                              ecolor='darkred', capsize=3, alpha=0.9, label='Data Points')
        
        ax4.set_xlabel('X (mm)', fontsize=12)
        ax4.set_ylabel('Y (mm)', fontsize=12)
        ax4.set_title('2D Contour Plot with Data', fontsize=14)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        plt.colorbar(contour, ax=ax4, label='Fitted Charge')
        
        # 5. Residuals vs X coordinate
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.set_facecolor('white')
        
        ax5.errorbar(x, residuals, xerr=0, yerr=z_err, fmt='.b', markersize=8, 
                    ecolor='gray', capsize=3, alpha=0.8, label='Residuals')
        ax5.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax5.axhline(np.mean(residuals), color='green', linestyle=':', linewidth=2, alpha=0.8, 
                   label=f'Mean: {np.mean(residuals):.6f}')
        
        ax5.set_xlabel('X (mm)', fontsize=12)
        ax5.set_ylabel('Residuals', fontsize=12)
        ax5.set_title('Residuals vs X Position', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.ticklabel_format(style='plain', useOffset=False, axis='y')
        
        # 6. Residuals vs Y coordinate
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.set_facecolor('white')
        
        ax6.errorbar(y, residuals, xerr=0, yerr=z_err, fmt='.b', markersize=8, 
                    ecolor='gray', capsize=3, alpha=0.8, label='Residuals')
        ax6.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax6.axhline(np.mean(residuals), color='green', linestyle=':', linewidth=2, alpha=0.8,
                   label=f'Mean: {np.mean(residuals):.6f}')
        
        ax6.set_xlabel('Y (mm)', fontsize=12)
        ax6.set_ylabel('Residuals', fontsize=12)
        ax6.set_title('Residuals vs Y Position', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.ticklabel_format(style='plain', useOffset=False, axis='y')
        
        # 7. Fitted vs Observed with error bars
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.set_facecolor('white')
        
        ax7.errorbar(z, fitted_z, xerr=z_err, yerr=0, fmt='o', markersize=6, 
                    color='blue', markeredgecolor='darkblue', markeredgewidth=1,
                    ecolor='gray', capsize=3, alpha=0.8, label='Data vs Fit')
        
        # Perfect fit line
        min_val = min(np.min(z), np.min(fitted_z))
        max_val = max(np.max(z), np.max(fitted_z))
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Fit')
        
        ax7.set_xlabel('Observed Charge', fontsize=12)
        ax7.set_ylabel('Fitted Charge', fontsize=12)
        ax7.set_title(f'Fitted vs Observed\nR² = {fit_info["r_squared"]}')
        ax7.set_aspect('equal')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        ax7.ticklabel_format(style='plain', useOffset=False)
        
        # 8. Residuals histogram with statistics
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.set_facecolor('white')
        
        n, bins, patches = ax8.hist(residuals, bins=15, alpha=0.7, color='skyblue', 
                                   edgecolor='black', density=True)
        ax8.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Zero')
        ax8.axvline(np.mean(residuals), color='green', linestyle=':', linewidth=2, alpha=0.8, 
                   label=f'Mean: {np.mean(residuals):.6f}')
        
        # Add Gaussian fit to residuals for normality check
        from scipy.stats import norm
        mu, sigma = norm.fit(residuals)
        x_hist = np.linspace(bins[0], bins[-1], 100)
        ax8.plot(x_hist, norm.pdf(x_hist, mu, sigma), 'orange', linewidth=2, alpha=0.8,
                label=f'Normal fit: μ={mu:.6f}, σ={sigma:.6f}')
        
        ax8.set_xlabel('Residuals', fontsize=12)
        ax8.set_ylabel('Density', fontsize=12)
        ax8.set_title(f'Residuals Distribution\nStd: {np.std(residuals):.6f}')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # 9. Fit parameters and statistics panel
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.set_facecolor('white')
        ax9.axis('off')  # Hide axes for text display
        
        # Create text summary
        param_text = "FIT PARAMETERS:\n\n"
        param_text += f"Amplitude: {params['amplitude']:.6f} ± {param_errors['amplitude']:.6f}\n"
        param_text += f"X Center: {params['x0']:.6f} ± {param_errors['x0']:.6f} mm\n"
        param_text += f"Y Center: {params['y0']:.6f} ± {param_errors['y0']:.6f} mm\n"
        param_text += f"σₓ: {params['sigma_x']:.6f} ± {param_errors['sigma_x']:.6f} mm\n"
        param_text += f"σᵧ: {params['sigma_y']:.6f} ± {param_errors['sigma_y']:.6f} mm\n"
        param_text += f"Rotation: {np.degrees(params['theta']):.2f}° ± {np.degrees(param_errors['theta']):.2f}°\n"
        param_text += f"Offset: {params['offset']:.6f} ± {param_errors['offset']:.6f}\n\n"
        
        param_text += "FIT STATISTICS:\n\n"
        param_text += f"R²: {fit_info['r_squared']:.6f}\n"
        param_text += f"χ²ᵣₑᵈ: {fit_info['FitChi2red']:.6f}\n"
        param_text += f"Data Points: {fit_info['n_data_points']}\n"
        param_text += f"DOF: {fit_info['degrees_of_freedom']}\n"
        param_text += f"Residual Std: {np.std(residuals):.6f}\n"
        
        # Calculate p-value from chi-squared
        from scipy.stats import chi2red
        chi_squared_stat = fit_info['FitChi2red'] * fit_info['degrees_of_freedom']
        FitPp = chi2red.sf(chi_squared_stat, fit_info['degrees_of_freedom'])
        param_text += f"P-value: {FitPp:.4e}\n"
        
        ax9.text(0.05, 0.95, param_text, transform=ax9.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if event_idx is not None:
                filename = f'3d_gaussian_fit_analysis_event_{event_idx}_{timestamp}.png'
            else:
                filename = f'3d_gaussian_fit_analysis_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Enhanced fit analysis saved to {filename}")
        
        return fig

    def plot_simple_fit_results(self, results, save_plot=True, output_dir="", event_idx=None, true_x=None, true_y=None):
        """
        Create a simple 2-panel plot similar to the 2D example: Data+Fit and Residuals
        Following the style of 2D_sci_fit.py
        """
        # Set plotting style
        plt.style.use('classic')
        
        # Extract data
        x = results['data']['x']
        y = results['data']['y']
        z = results['data']['z']
        z_err = results['data']['z_err']
        fitted_z = results['fitted_values']
        residuals = results['residuals']
        
        # Extract fit parameters and statistics
        params = results['parameters']
        param_errors = results['parameter_errors']
        fit_info = results['fit_info']
        
        # Calculate p-value from chi-squared
        chi_squared_stat = fit_info['FitChi2red'] * fit_info['degrees_of_freedom']
        FitPp = stats.chi2red.sf(chi_squared_stat, fit_info['degrees_of_freedom'])
        
        # Create figure with 2 subplots (side by side)
        plt.close('all')
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.style.use('classic')
        
        fig.patch.set_facecolor('white')
        for ax in axs:
            ax.set_facecolor('white')
        
        # Left panel: Data and fit as 2D contour plot
        # Create grid for smooth surface plot
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_grid = np.linspace(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range, 50)
        y_grid = np.linspace(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_fitted_smooth = self.gaussian_3d(self.popt, (X_grid, Y_grid))
        
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
        axs[0].plot(params['x0'], params['y0'], '+', color='red', markersize=15, markeredgewidth=4, 
                   label='Fit Center')
        
        # Add TrueX, TrueY if available
        if true_x is not None and true_y is not None:
            axs[0].plot(true_x, true_y, 'x', color='orange', markersize=12, markeredgewidth=4, 
                       label='True Position')
            
            # Draw line from fit center to true position
            axs[0].plot([params['x0'], true_x], [params['y0'], true_y], 
                       'r--', linewidth=2, alpha=0.7, label='Distance')
            
            # Calculate and display distance
            distance = np.sqrt((params['x0'] - true_x)**2 + (params['y0'] - true_y)**2)
            
            # Add distance text annotation
            mid_x = (params['x0'] + true_x) / 2
            mid_y = (params['y0'] + true_y) / 2
            axs[0].annotate(f'{distance:.3f} mm', xy=(mid_x, mid_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        axs[0].set_xlabel('X (mm)', fontsize=14)
        axs[0].set_ylabel('Y (mm)', fontsize=14)
        axs[0].set_aspect('equal')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper right')
        
        # Right panel: Residuals analysis
        # Plot residuals vs radial distance from fit center
        r_distance = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        
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
            if event_idx is not None:
                filename = f'3d_gaussian_simple_fit_event_{event_idx}_{timestamp}.png'
            else:
                filename = f'3d_gaussian_simple_fit_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Simple fit plot saved to {filename}")
        
        plt.show()
        return fig

    def plot_3d_fit_visualization(self, results, save_plot=True, output_dir="", event_idx=None, true_x=None, true_y=None):
        """
        Create separate focused 3D visualizations for fit quality assessment
        
        This method creates dedicated 3D plots to help visualize:
        1. Original data points with error bars
        2. Fitted 3D Gaussian surface
        3. Data points overlaid on fitted surface
        4. Residuals analysis in 3D
        
        Parameters:
        -----------
        results : dict
            Fit results from fit_3d_gaussian_odr
        save_plot : bool
            Whether to save the plots
        output_dir : str
            Directory to save plots
        event_idx : int, optional
            Event index for filename
        true_x, true_y : float, optional
            True position coordinates to mark on plots
        """
        # Set plotting style
        plt.style.use('classic')
        
        # Extract data
        x = results['data']['x']
        y = results['data']['y']
        z = results['data']['z']
        z_err = results['data']['z_err']
        fitted_z = results['fitted_values']
        residuals = results['residuals']
        
        # Extract fit parameters
        params = results['parameters']
        param_errors = results['parameter_errors']
        fit_info = results['fit_info']
        
        # Create high-resolution grid for smooth surface
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        x_margin = 0.15 * x_range
        y_margin = 0.15 * y_range
        
        x_grid = np.linspace(np.min(x) - x_margin, np.max(x) + x_margin, 60)
        y_grid = np.linspace(np.min(y) - y_margin, np.max(y) + y_margin, 60)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_fitted_smooth = self.gaussian_3d(self.popt, (X_grid, Y_grid))
        
        # Create figure with 4 3D subplots (2x2 layout)
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('white')
        
        # 1. Original Data with Error Bars
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_facecolor('white')
        
        # 3D scatter plot with color-coded heights
        scatter1 = ax1.scatter(x, y, z, c=z, cmap='plasma', s=100, alpha=0.9, 
                              edgecolors='black', linewidth=1, depthshade=True)
        
        # Add vertical error bars
        for i in range(len(x)):
            ax1.plot([x[i], x[i]], [y[i], y[i]], [z[i]-z_err[i], z[i]+z_err[i]], 
                    'k-', alpha=0.6, linewidth=2)
            # Add caps to error bars
            cap_size = 0.01 * (np.max(x) - np.min(x))
            ax1.plot([x[i]-cap_size, x[i]+cap_size], [y[i], y[i]], [z[i]-z_err[i], z[i]-z_err[i]], 
                    'k-', alpha=0.6, linewidth=2)
            ax1.plot([x[i]-cap_size, x[i]+cap_size], [y[i], y[i]], [z[i]+z_err[i], z[i]+z_err[i]], 
                    'k-', alpha=0.6, linewidth=2)
        
        # Mark true position if available
        if true_x is not None and true_y is not None:
            # Project true position to data plane
            true_z_proj = np.mean(z)  # Use mean height for projection
            ax1.scatter([true_x], [true_y], [true_z_proj], c='red', s=200, marker='x', 
                       linewidth=4, label='True Position')
        
        # Mark fit center
        fit_z_proj = np.mean(z)
        ax1.scatter([params['x0']], [params['y0']], [fit_z_proj], c='orange', s=200, 
                   marker='+', linewidth=4, label='Fit Center')
        
        ax1.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax1.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax1.set_zlabel('Charge', fontsize=12, labelpad=10)
        ax1.set_title('Original Data with Error Bars', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        if true_x is not None and true_y is not None:
            ax1.legend(loc='upper left')
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Charge', fontsize=12)
        
        # 2. Fitted 3D Gaussian Surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.set_facecolor('white')
        
        # Plot smooth fitted surface
        surface = ax2.plot_surface(X_grid, Y_grid, Z_fitted_smooth, cmap='plasma', alpha=0.8,
                                 linewidth=0, antialiased=True, rstride=2, cstride=2)
        
        # Add contour lines on the surface for better depth perception
        contours = ax2.contour(X_grid, Y_grid, Z_fitted_smooth, levels=10, colors='white', 
                             alpha=0.6, linewidths=1)
        
        # Mark fit center and true position
        if true_x is not None and true_y is not None:
            true_z_fit = self.gaussian_3d(self.popt, ([true_x], [true_y]))[0]
            ax2.scatter([true_x], [true_y], [true_z_fit], c='red', s=200, marker='x', 
                       linewidth=4, label='True Position')
        
        fit_z_center = self.gaussian_3d(self.popt, ([params['x0']], [params['y0']]))[0]
        ax2.scatter([params['x0']], [params['y0']], [fit_z_center], c='orange', s=200, 
                   marker='+', linewidth=4, label='Fit Center')
        
        ax2.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax2.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax2.set_zlabel('Fitted Charge', fontsize=12, labelpad=10)
        ax2.set_title('Fitted 3D Gaussian Surface', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        if true_x is not None and true_y is not None:
            ax2.legend(loc='upper left')
        
        # Add colorbar
        cbar2 = plt.colorbar(surface, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Fitted Charge', fontsize=12)
        
        # 3. Data Points on Fitted Surface (Combined View)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.set_facecolor('white')
        
        # Plot fitted surface with transparency
        surface3 = ax3.plot_surface(X_grid, Y_grid, Z_fitted_smooth, cmap='plasma', alpha=0.5,
                                  linewidth=0, antialiased=True, rstride=3, cstride=3)
        
        # Overlay data points
        scatter3 = ax3.scatter(x, y, z, c='red', s=80, alpha=0.9, edgecolors='darkred', 
                             linewidth=1, label='Data Points', depthshade=True)
        
        # Draw vertical lines from data points to fitted surface
        fitted_z_at_data = self.gaussian_3d(self.popt, (x, y))
        for i in range(len(x)):
            ax3.plot([x[i], x[i]], [y[i], y[i]], [z[i], fitted_z_at_data[i]], 
                    'gray', alpha=0.7, linewidth=1)
        
        # Mark positions
        if true_x is not None and true_y is not None:
            true_z_fit = self.gaussian_3d(self.popt, ([true_x], [true_y]))[0]
            ax3.scatter([true_x], [true_y], [true_z_fit], c='blue', s=200, marker='x', 
                       linewidth=4, label='True Position')
        
        ax3.scatter([params['x0']], [params['y0']], [fit_z_center], c='orange', s=200, 
                   marker='+', linewidth=4, label='Fit Center')
        
        ax3.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax3.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax3.set_zlabel('Charge', fontsize=12, labelpad=10)
        ax3.set_title('Data Points on Fitted Surface', fontsize=14, pad=20)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        # 4. Residuals in 3D
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.set_facecolor('white')
        
        # 3D residuals plot with color coding
        residual_scatter = ax4.scatter(x, y, residuals, c=residuals, cmap='RdBu_r', s=100, 
                                     alpha=0.9, edgecolors='black', linewidth=1)
        
        # Add zero plane for reference
        zero_surface = ax4.plot_surface(X_grid, Y_grid, np.zeros_like(X_grid), alpha=0.3, 
                                       color='gray', linewidth=0)
        
        # Draw vertical lines from zero plane to residuals
        for i in range(len(x)):
            ax4.plot([x[i], x[i]], [y[i], y[i]], [0, residuals[i]], 
                    'gray', alpha=0.7, linewidth=1)
        
        # Mark positions on zero plane
        if true_x is not None and true_y is not None:
            ax4.scatter([true_x], [true_y], [0], c='red', s=200, marker='x', 
                       linewidth=4, label='True Position')
        
        ax4.scatter([params['x0']], [params['y0']], [0], c='orange', s=200, marker='+', 
                   linewidth=4, label='Fit Center')
        
        ax4.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax4.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax4.set_zlabel('Residuals (Fitted - Observed)', fontsize=12, labelpad=10)
        ax4.set_title(f'Residuals Analysis\nStd: {np.std(residuals):.6f}', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        
        # Add colorbar for residuals
        cbar4 = plt.colorbar(residual_scatter, ax=ax4, shrink=0.6, pad=0.1)
        cbar4.set_label('Residuals', fontsize=12)
        
        # Set consistent viewing angles for better comparison
        for ax in [ax1, ax2, ax3, ax4]:
            ax.view_init(elev=20, azim=45)
            ax.tick_params(labelsize=10)
        
        # Add overall title with fit statistics
        fig.suptitle(f'3D Gaussian Fit Visualization - Event {event_idx if event_idx is not None else "N/A"}\n'
                    f'R² = {fit_info["r_squared"]:.6f}, χ²ᵣₑᵈ = {fit_info["FitChi2red"]:.6f}, ' +
                    f'Data Points = {fit_info["n_data_points"]}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if event_idx is not None:
                filename = f'3d_visualization_event_{event_idx}_{timestamp}.png'
            else:
                filename = f'3d_visualization_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"3D visualization saved to {filename}")
        
        plt.show()
        return fig

    def analyze_sub_neighborhoods(self, root_filename, event_idx, charge_type='fraction', save_plot=True, output_dir=""):
        """
        Analyze charge distribution in progressively smaller sub-neighborhoods
        
        This method extracts and fits 3D Gaussians to different neighborhood sizes
        from the same event: 9x9, 5x5, 3x3, and 2x2 grids, all centered on the same point.
        
        Parameters:
        -----------
        root_filename : str
            Path to ROOT file containing charge data
        event_idx : int
            Event index to analyze
        charge_type : str
            Type of charge data: 'fraction', 'value', or 'coulomb'
        save_plot : bool
            Whether to save the comparison plot
        output_dir : str
            Directory to save plots
            
        Returns:
        --------
        dict : Results for each neighborhood size with fit parameters and statistics
        """
        print(f"Analyzing sub-neighborhoods for event {event_idx} ({charge_type} data)")
        
        # Load data using the charge_sim module
        generator = RandomHitChargeGridGenerator(root_filename)
        
        # Get detector parameters
        pixel_spacing = generator.detector_params['pixel_spacing']
        
        # Get event data
        pixel_x = generator.data['PixelX'][event_idx]
        pixel_y = generator.data['PixelY'][event_idx]
        true_x = generator.data['TrueX'][event_idx]
        true_y = generator.data['TrueY'][event_idx]
        
        # Get charge data based on type
        if charge_type == 'fraction':
            charge_data = generator.data['GridNeighborhoodChargeFractions'][event_idx]
        elif charge_type == 'value':
            charge_data = generator.data['GridNeighborhoodChargeValues'][event_idx]
        elif charge_type == 'coulomb':
            if 'GridNeighborhoodCharge' not in generator.data:
                raise ValueError("Coulomb charge data not available in this ROOT file")
            charge_data = generator.data['GridNeighborhoodCharge'][event_idx]
        else:
            raise ValueError("charge_type must be 'fraction', 'value', or 'coulomb'")
        
        # Reshape to 9x9 grid and replace invalid values
        full_grid = np.array(charge_data).reshape(9, 9)
        full_grid[full_grid == -999.0] = np.nan
        
        # Define neighborhood sizes and their extraction ranges
        neighborhoods = {
            '9x9': {'size': 9, 'start': 0, 'end': 9},
            '5x5': {'size': 5, 'start': 2, 'end': 7},
            '3x3': {'size': 3, 'start': 3, 'end': 6},
            '2x2': {'size': 2, 'start': 3, 'end': 5}  # Note: 2x2 centered, so we take pixels [3:5, 3:5]
        }
        
        results = {}
        
        # Process each neighborhood size
        for name, params in neighborhoods.items():
            print(f"\nProcessing {name} neighborhood...")
            
            # Extract sub-grid
            start, end = params['start'], params['end']
            sub_grid = full_grid[start:end, start:end]
            
            # Create coordinate arrays for this sub-grid
            x_coords = []
            y_coords = []
            z_values = []
            
            grid_size = params['size']
            center_offset = (grid_size - 1) // 2
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if not np.isnan(sub_grid[i, j]):
                        # Calculate position relative to pixel center
                        rel_x = (j - center_offset) * pixel_spacing
                        rel_y = (i - center_offset) * pixel_spacing
                        
                        actual_x = pixel_x + rel_x
                        actual_y = pixel_y + rel_y
                        
                        x_coords.append(actual_x)
                        y_coords.append(actual_y)
                        z_values.append(sub_grid[i, j])
            
            # Convert to numpy arrays
            x = np.array(x_coords)
            y = np.array(y_coords)
            z = np.array(z_values)
            
            if len(x) < 4:  # Need at least 4 points for 7-parameter fit
                print(f"Warning: {name} has only {len(x)} valid points, skipping fit")
                results[name] = {
                    'data': {'x': x, 'y': y, 'z': z},
                    'fit_successful': False,
                    'n_points': len(x)
                }
                continue
            
            # Perform fit
            try:
                # Initial guess
                initial_guess = [
                    np.max(z) - np.min(z) if len(z) > 0 else 1.0,  # amplitude
                    pixel_x,                # x0
                    pixel_y,                # y0
                    np.std(x) if len(x) > 1 else pixel_spacing,    # sigma_x
                    np.std(y) if len(y) > 1 else pixel_spacing,    # sigma_y
                    0.0,                    # theta
                    np.min(z) if len(z) > 0 else 0.0               # offset
                ]
                
                fit_result = self.fit_3d_gaussian_odr(x, y, z, initial_guess=initial_guess, verbose=False)
                
                results[name] = {
                    'data': {'x': x, 'y': y, 'z': z},
                    'fit_result': fit_result,
                    'fit_successful': True,
                    'n_points': len(x),
                    'grid_data': sub_grid
                }
                
                print(f"{name}: R² = {fit_result['fit_info']['r_squared']:.4f}, "
                      f"σₓ = {fit_result['parameters']['sigma_x']:.4f} mm, "
                      f"σᵧ = {fit_result['parameters']['sigma_y']:.4f} mm")
                
            except Exception as e:
                print(f"Fit failed for {name}: {e}")
                results[name] = {
                    'data': {'x': x, 'y': y, 'z': z},
                    'fit_successful': False,
                    'n_points': len(x),
                    'grid_data': sub_grid,
                    'error': str(e)
                }
        
        # Create comparison visualization
        if save_plot:
            self.plot_sub_neighborhood_comparison(results, event_idx, charge_type, 
                                                pixel_x, pixel_y, true_x, true_y, 
                                                pixel_spacing, save_plot, output_dir)
        
        return results
    
    def plot_sub_neighborhood_comparison(self, results, event_idx, charge_type, 
                                       pixel_x, pixel_y, true_x, true_y, 
                                       pixel_spacing, save_plot=True, output_dir=""):
        """
        Create comprehensive comparison plot of different neighborhood sizes
        """
        # Set plotting style
        plt.style.use('classic')
        
        # Create figure with subplots for each neighborhood size
        fig = plt.figure(figsize=(24, 18))
        fig.patch.set_facecolor('white')
        
        # Neighborhood order for plotting
        neighborhood_order = ['9x9', '5x5', '3x3', '2x2']
        colors = ['plasma', 'viridis', 'cividis', 'magma']
        
        # Top row: Grid visualizations as heatmaps
        for i, (name, color) in enumerate(zip(neighborhood_order, colors)):
            if name not in results:
                continue
                
            ax = fig.add_subplot(3, 4, i+1)
            ax.set_facecolor('white')
            
            # Plot grid data as heatmap
            grid_data = results[name]['grid_data']
            
            # Create position arrays for the grid
            size = grid_data.shape[0]
            center_offset = (size - 1) // 2
            
            # Create coordinate arrays
            x_grid_pos = np.arange(size) - center_offset
            y_grid_pos = np.arange(size) - center_offset
            
            # Plot heatmap
            im = ax.imshow(grid_data, cmap=color, aspect='equal', 
                          extent=[x_grid_pos[0]-0.5, x_grid_pos[-1]+0.5, 
                                 y_grid_pos[-1]+0.5, y_grid_pos[0]-0.5])
            
            # Add text annotations for charge values
            for ii in range(size):
                for jj in range(size):
                    if not np.isnan(grid_data[ii, jj]):
                        text_color = 'white' if grid_data[ii, jj] > np.nanmean(grid_data) else 'black'
                        ax.text(jj-center_offset, ii-center_offset, f'{grid_data[ii, jj]:.3f}', 
                               ha='center', va='center', fontsize=8, color=text_color, weight='bold')
            
            # Mark center
            ax.plot(0, 0, '+', color='red', markersize=15, markeredgewidth=3)
            
            ax.set_title(f'{name} Grid\n{results[name]["n_points"]} valid points', fontsize=12)
            ax.set_xlabel('Pixel Units from Center', fontsize=10)
            ax.set_ylabel('Pixel Units from Center', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8, label='Charge')
        
        # Middle row: 2D scatter plots with fits
        for i, name in enumerate(neighborhood_order):
            if name not in results or not results[name]['fit_successful']:
                continue
                
            ax = fig.add_subplot(3, 4, i+5)
            ax.set_facecolor('white')
            
            # Get data
            data = results[name]['data']
            fit_result = results[name]['fit_result']
            x, y, z = data['x'], data['y'], data['z']
            
            # Create fitted surface for contour plot
            x_range = np.max(x) - np.min(x) if len(x) > 1 else pixel_spacing
            y_range = np.max(y) - np.min(y) if len(y) > 1 else pixel_spacing
            
            x_grid = np.linspace(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range, 30)
            y_grid = np.linspace(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range, 30)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            # Use the fitted parameters to create the surface
            fitted_params = [fit_result['parameters'][p] for p in 
                           ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']]
            Z_fitted = self.gaussian_3d(fitted_params, (X_grid, Y_grid))
            
            # Plot contours
            contour = ax.contourf(X_grid, Y_grid, Z_fitted, levels=15, cmap=colors[i], alpha=0.7)
            
            # Plot data points
            scatter = ax.scatter(x, y, c=z, s=80, cmap=colors[i], edgecolors='white', 
                               linewidth=2, zorder=5)
            
            # Mark fit center and true position
            ax.plot(fit_result['parameters']['x0'], fit_result['parameters']['y0'], 
                   '+', color='red', markersize=12, markeredgewidth=3, label='Fit Center')
            ax.plot(true_x, true_y, 'x', color='orange', markersize=10, 
                   markeredgewidth=3, label='True Position')
            
            # Calculate and display distance
            distance = np.sqrt((fit_result['parameters']['x0'] - true_x)**2 + 
                             (fit_result['parameters']['y0'] - true_y)**2)
            
            ax.set_title(f'{name} Fit\nR² = {fit_result["fit_info"]["r_squared"]:.4f}\n'
                        f'Distance = {distance:.4f} mm', fontsize=11)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            plt.colorbar(contour, ax=ax, shrink=0.8, label='Fitted Charge')
        
        # Bottom row: Fit parameter comparison
        ax_params = fig.add_subplot(3, 2, 5)
        ax_params.set_facecolor('white')
        
        # Extract parameters for successful fits
        successful_fits = {name: results[name] for name in neighborhood_order 
                          if name in results and results[name]['fit_successful']}
        
        if successful_fits:
            names = list(successful_fits.keys())
            n_points = [successful_fits[name]['n_points'] for name in names]
            r_squared = [successful_fits[name]['fit_result']['fit_info']['r_squared'] for name in names]
            sigma_x = [successful_fits[name]['fit_result']['parameters']['sigma_x'] for name in names]
            sigma_y = [successful_fits[name]['fit_result']['parameters']['sigma_y'] for name in names]
            distances = [np.sqrt((successful_fits[name]['fit_result']['parameters']['x0'] - true_x)**2 + 
                                (successful_fits[name]['fit_result']['parameters']['y0'] - true_y)**2) 
                        for name in names]
            
            x_pos = np.arange(len(names))
            
            # Plot fit quality metrics
            ax_params.plot(x_pos, r_squared, 'o-', label='R²', linewidth=2, markersize=8)
            ax_params.plot(x_pos, np.array(distances)*10, 's-', label='Distance×10 (mm)', linewidth=2, markersize=8)
            
            ax_params.set_xlabel('Neighborhood Size', fontsize=12)
            ax_params.set_ylabel('Fit Quality Metrics', fontsize=12)
            ax_params.set_title('Fit Quality vs Neighborhood Size', fontsize=14)
            ax_params.set_xticks(x_pos)
            ax_params.set_xticklabels(names)
            ax_params.grid(True, alpha=0.3)
            ax_params.legend()
            ax_params.set_ylim(0, 1.1)
        
        # Bottom right: Parameter evolution
        ax_sigma = fig.add_subplot(3, 2, 6)
        ax_sigma.set_facecolor('white')
        
        if successful_fits:
            ax_sigma.plot(x_pos, sigma_x, 'o-', label='σₓ (mm)', linewidth=2, markersize=8)
            ax_sigma.plot(x_pos, sigma_y, 's-', label='σᵧ (mm)', linewidth=2, markersize=8)
            ax_sigma.plot(x_pos, n_points, '^-', label='N points / 10', linewidth=2, markersize=8)
            
            ax_sigma.set_xlabel('Neighborhood Size', fontsize=12)
            ax_sigma.set_ylabel('Parameters', fontsize=12)
            ax_sigma.set_title('Gaussian Width vs Neighborhood Size', fontsize=14)
            ax_sigma.set_xticks(x_pos)
            ax_sigma.set_xticklabels(names)
            ax_sigma.grid(True, alpha=0.3)
            ax_sigma.legend()
        
        # Add overall title
        fig.suptitle(f'Sub-Neighborhood Analysis - Event {event_idx} ({charge_type})\n'
                    f'Pixel Center: ({pixel_x:.3f}, {pixel_y:.3f}) mm, '
                    f'True Position: ({true_x:.3f}, {true_y:.3f}) mm', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save plot if requested
        if save_plot:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'sub_neighborhoods_event_{event_idx}_{timestamp}.png'
            
            if output_dir:
                filename = os.path.join(output_dir, filename)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Sub-neighborhood analysis saved to {filename}")
        
        plt.show()
        return fig

def extract_charge_data_for_fitting(root_filename, event_idx, charge_type='fraction'):
    """
    Extract charge distribution data from ROOT file for 3D Gaussian fitting
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file
    event_idx : int
        Event index to extract data from
    charge_type : str
        Type of charge data: 'fraction', 'value', or 'coulomb'
        
    Returns:
    --------
    tuple : (x, y, z, x_center, y_center, true_x, true_y) coordinate and charge arrays
    """
    # Load data using the charge_sim module
    generator = RandomHitChargeGridGenerator(root_filename)
    
    # Get detector parameters
    pixel_spacing = generator.detector_params['pixel_spacing']
    
    # Get event data
    pixel_x = generator.data['PixelX'][event_idx]
    pixel_y = generator.data['PixelY'][event_idx]
    true_x = generator.data['TrueX'][event_idx]
    true_y = generator.data['TrueY'][event_idx]
    
    # Get charge data based on type
    if charge_type == 'fraction':
        charge_data = generator.data['GridNeighborhoodChargeFractions'][event_idx]
    elif charge_type == 'value':
        charge_data = generator.data['GridNeighborhoodChargeValues'][event_idx]
    elif charge_type == 'coulomb':
        if 'GridNeighborhoodCharge' not in generator.data:
            raise ValueError("Coulomb charge data not available in this ROOT file")
        charge_data = generator.data['GridNeighborhoodCharge'][event_idx]
    else:
        raise ValueError("charge_type must be 'fraction', 'value', or 'coulomb'")
    
    # Reshape to 9x9 grid
    grid_data = np.array(charge_data).reshape(9, 9)
    
    # Replace invalid values (-999.0) with NaN
    grid_data[grid_data == -999.0] = np.nan
    
    # Create coordinate arrays
    x_coords = []
    y_coords = []
    z_values = []
    
    for i in range(9):
        for j in range(9):
            if not np.isnan(grid_data[i, j]):
                # Calculate actual position relative to pixel center
                rel_x = (j - 4) * pixel_spacing
                rel_y = (i - 4) * pixel_spacing
                
                actual_x = pixel_x + rel_x
                actual_y = pixel_y + rel_y
                
                x_coords.append(actual_x)
                y_coords.append(actual_y)
                z_values.append(grid_data[i, j])
    
    return np.array(x_coords), np.array(y_coords), np.array(z_values), pixel_x, pixel_y, true_x, true_y

def fit_charge_distribution_3d(root_filename, event_idx, charge_type='fraction', save_plots=True, output_dir="", plot_style='both'):
    """
    Complete 3D Gaussian fitting workflow for charge distribution data
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file containing charge data
    event_idx : int
        Event index to fit
    charge_type : str
        Type of charge data to fit: 'fraction', 'value', or 'coulomb'
    save_plots : bool
        Whether to save visualization plots
    output_dir : str
        Directory to save plots
    plot_style : str
        Plot style: 'simple' (2-panel like 2D example), 'comprehensive' (9-panel analysis), 
        '3d' (focused 3D visualization), or 'both' (simple + comprehensive), 'all' (all three styles)
        
    Returns:
    --------
    dict : Complete fit results including parameters, errors, and statistics
    """
    print(f"Fitting 3D Gaussian to event {event_idx} ({charge_type} data)")
    
    # Extract data
    x, y, z, x_center, y_center, true_x, true_y = extract_charge_data_for_fitting(root_filename, event_idx, charge_type)
    
    print(f"Extracted {len(x)} valid data points")
    print(f"Charge range: {np.min(z):.6f} to {np.max(z):.6f}")
    
    # Create fitter and perform fit
    fitter = Gaussian3DODR()
    
    # Initial guess based on data
    initial_guess = [
        np.max(z) - np.min(z),  # amplitude
        x_center,               # x0 (use pixel center)
        y_center,               # y0 (use pixel center)
        np.std(x),              # sigma_x
        np.std(y),              # sigma_y
        0.0,                    # theta (no rotation initially)
        np.min(z)               # offset
    ]
    
    results = fitter.fit_3d_gaussian_odr(x, y, z, initial_guess=initial_guess)
    
    # Create visualization
    if save_plots:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if plot_style == 'simple':
            fitter.plot_simple_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
        elif plot_style == 'comprehensive':
            fitter.plot_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx)
        elif plot_style == '3d':
            fitter.plot_3d_fit_visualization(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
        elif plot_style == 'both':
            fitter.plot_simple_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
            fitter.plot_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx)
        elif plot_style == 'all':
            fitter.plot_simple_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
            fitter.plot_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx)
            fitter.plot_3d_fit_visualization(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
        else:
            print(f"Warning: Unknown plot_style '{plot_style}'. Using 'simple' style.")
            fitter.plot_simple_fit_results(results, save_plot=True, output_dir=output_dir, event_idx=event_idx, true_x=true_x, true_y=true_y)
    
    return results

def analyze_multiple_events(root_filename, event_indices, charge_type='fraction', save_plots=True, output_dir=""):
    """
    Fit 3D Gaussians to multiple events and compare results
    """
    all_results = {}
    
    print(f"Analyzing {len(event_indices)} events with 3D Gaussian fitting")
    
    for i, event_idx in enumerate(event_indices):
        print(f"\n{'='*50}")
        print(f"Processing event {event_idx} ({i+1}/{len(event_indices)})")
        print(f"{'='*50}")
        
        try:
            results = fit_charge_distribution_3d(root_filename, event_idx, charge_type, save_plots, output_dir)
            all_results[event_idx] = results
        except Exception as e:
            print(f"Failed to fit event {event_idx}: {e}")
            continue
    
    # Summary statistics
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL FITS")
        print(f"{'='*60}")
        
        # Collect parameter values
        param_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
        param_summary = {name: [] for name in param_names}
        r_squared_values = []
        
        for event_idx, results in all_results.items():
            for param in param_names:
                param_summary[param].append(results['parameters'][param])
            r_squared_values.append(results['fit_info']['r_squared'])
        
        # Print summary statistics
        print(f"Successfully fitted {len(all_results)} events")
        print(f"Average R-squared: {np.mean(r_squared_values):.6f} ± {np.std(r_squared_values):.6f}")
        
        for param in param_names:
            values = param_summary[param]
            print(f"{param:12s}: {np.mean(values):8.4f} ± {np.std(values):8.4f} (range: {np.min(values):8.4f} to {np.max(values):8.4f})")
    
    return all_results

def analyze_event_sub_neighborhoods(root_filename, event_idx, charge_type='fraction', save_plots=True, output_dir=""):
    """
    Convenience function to analyze charge distribution in progressively smaller sub-neighborhoods
    
    This function creates a complete analysis of how charge sharing and fit quality
    changes when looking at different neighborhood sizes (9x9, 5x5, 3x3, 2x2) for a single event.
    
    Parameters:
    -----------
    root_filename : str
        Path to ROOT file containing charge data
    event_idx : int
        Event index to analyze
    charge_type : str
        Type of charge data to analyze: 'fraction', 'value', or 'coulomb'
    save_plots : bool
        Whether to save visualization plots
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    dict : Complete analysis results for all neighborhood sizes
    """
    print(f"="*70)
    print(f"SUB-NEIGHBORHOOD ANALYSIS FOR EVENT {event_idx}")
    print(f"="*70)
    
    # Create output directory if needed
    if save_plots and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create fitter instance and run analysis
    fitter = Gaussian3DODR()
    results = fitter.analyze_sub_neighborhoods(root_filename, event_idx, charge_type, save_plots, output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF SUB-NEIGHBORHOOD ANALYSIS")
    print(f"{'='*70}")
    
    successful_fits = {name: results[name] for name in results 
                      if results[name]['fit_successful']}
    
    if successful_fits:
        print(f"Successfully fitted {len(successful_fits)} neighborhood sizes")
        print("\nFit Quality Summary:")
        print(f"{'Size':<6} {'Points':<7} {'R²':<8} {'σₓ (mm)':<10} {'σᵧ (mm)':<10} {'Distance (mm)':<12}")
        print("-" * 70)
        
        # Load true position for distance calculation
        from charge_sim import RandomHitChargeGridGenerator
        generator = RandomHitChargeGridGenerator(root_filename)
        true_x = generator.data['TrueX'][event_idx]
        true_y = generator.data['TrueY'][event_idx]
        
        for name in ['9x9', '5x5', '3x3', '2x2']:
            if name in successful_fits:
                fit_result = successful_fits[name]['fit_result']
                params = fit_result['parameters']
                distance = np.sqrt((params['x0'] - true_x)**2 + (params['y0'] - true_y)**2)
                
                print(f"{name:<6} {successful_fits[name]['n_points']:<7} "
                      f"{fit_result['fit_info']['r_squared']:<8.4f} "
                      f"{params['sigma_x']:<10.4f} "
                      f"{params['sigma_y']:<10.4f} "
                      f"{distance:<12.4f}")
        
        print(f"\nKey Observations:")
        print(f"- Neighborhood size affects fit quality and parameter estimates")
        print(f"- Smaller neighborhoods may have fewer valid data points")
        print(f"- Gaussian width parameters may change with neighborhood size")
        print(f"- Position accuracy may vary with available charge information")
    else:
        print("No successful fits obtained for any neighborhood size")
    
    return results

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
    EVENTS_TO_FIT = [10, 50, 100]  # Example event indices
    CHARGE_TYPE = 'fraction'       # 'fraction', 'value', or 'coulomb'
    OUTPUT_DIR = "3d_gaussian_fits"
    
    print("="*60)
    print("3D GAUSSIAN FITTING WITH ODR - ENHANCED PLOTTING")
    print("="*60)
    
    # Demonstrate simple plotting (like 2D example) for one event
    print(f"\nDemonstrating simple 2-panel plot for event {EVENTS_TO_FIT[0]}:")
    single_result = fit_charge_distribution_3d(root_file, EVENTS_TO_FIT[0], CHARGE_TYPE, 
                                             save_plots=True, output_dir=OUTPUT_DIR, 
                                             plot_style='simple')
    
    # Demonstrate dedicated 3D visualization for the same event
    print(f"\nDemonstrating dedicated 3D visualization for event {EVENTS_TO_FIT[0]}:")
    fit_charge_distribution_3d(root_file, EVENTS_TO_FIT[0], CHARGE_TYPE, 
                             save_plots=True, output_dir=OUTPUT_DIR, 
                             plot_style='3d')
    
    # Demonstrate sub-neighborhood analysis
    print(f"\nDemonstrating sub-neighborhood analysis for event {EVENTS_TO_FIT[0]}:")
    sub_neighborhood_results = analyze_event_sub_neighborhoods(root_file, EVENTS_TO_FIT[0], CHARGE_TYPE, 
                                                             save_plots=True, output_dir=OUTPUT_DIR)
    
    # Analyze multiple events with comprehensive plots
    print(f"\nAnalyzing {len(EVENTS_TO_FIT)} events with comprehensive plots:")
    results = analyze_multiple_events(root_file, EVENTS_TO_FIT, CHARGE_TYPE, save_plots=True, output_dir=OUTPUT_DIR)
    
    print(f"\nFitting complete. Results saved to {OUTPUT_DIR}")
    print("Plot styles generated:")
    print("  - Simple 2-panel plots (similar to 2D_sci_fit.py)")
    print("  - Comprehensive 9-panel analysis plots")
    print("  - Dedicated 3D visualization plots (NEW!)")
    print("    * Original data with 3D error bars")
    print("    * Fitted 3D Gaussian surface")
    print("    * Data points overlaid on fitted surface")
    print("    * 3D residuals analysis")
    print("  - Sub-neighborhood analysis plots (NEW!)")
    print("    * Compares 9x9, 5x5, 3x3, and 2x2 neighborhoods")
    print("    * Shows charge distribution heatmaps for each size")
    print("    * Fits Gaussians to each neighborhood size")
    print("    * Analyzes how fit quality changes with neighborhood size")
    print("  - All plots include error bars, residuals analysis, and statistical information")
    print("\nTo generate all plot styles for an event, use plot_style='all'")
    print(f"\nFor sub-neighborhood analysis of any event, use:")
    print(f"  analyze_event_sub_neighborhoods('{root_file}', event_idx, '{CHARGE_TYPE}')")
