#!/usr/bin/env python3
"""
Analyze 2D Gaussian fitting data to understand why fits are failing
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt

def analyze_fitting_data():
    # Read ROOT file
    root_file_path = "/home/tom/Desktop/Cultural_Keys/epicToy/build/epicToyOutput.root"
    
    with uproot.open(root_file_path) as file:
        tree = file['Hits']
        
        # Read relevant branches
        is_pixel_hit = tree['IsPixelHit'].array(library="np")
        charges = tree['NonPixel_GridNeighborhoodCharge'].array(library="np")
        charge_fractions = tree['NonPixel_GridNeighborhoodChargeFractions'].array(library="np")
        fit_successful = tree['Fit2D_Successful'].array(library="np")
        x_amplitude = tree['Fit2D_XAmplitude'].array(library="np")
        y_amplitude = tree['Fit2D_YAmplitude'].array(library="np")
        x_center = tree['Fit2D_XCenter'].array(library="np")
        y_center = tree['Fit2D_YCenter'].array(library="np")
        x_sigma = tree['Fit2D_XSigma'].array(library="np")
        y_sigma = tree['Fit2D_YSigma'].array(library="np")
        
        # Analyze non-pixel hits only
        non_pixel_mask = ~is_pixel_hit
        non_pixel_indices = np.where(non_pixel_mask)[0]
        
        print(f"Total events: {len(is_pixel_hit)}")
        print(f"Non-pixel events: {len(non_pixel_indices)}")
        print(f"Successful fits: {np.sum(fit_successful)}")
        
        # Look at a few specific events with charge data
        for i, event_idx in enumerate(non_pixel_indices[:5]):
            print(f"\n=== Event {event_idx} ===")
            event_charges = charges[event_idx]
            event_fractions = charge_fractions[event_idx]
            
            # Filter out invalid charges (-999.0)
            valid_charges = event_charges[event_charges > 0]
            valid_fractions = event_fractions[event_fractions > 0]
            
            print(f"Charges (Coulombs): {len(valid_charges)} valid points")
            if len(valid_charges) > 0:
                print(f"  Range: {np.min(valid_charges):.2e} to {np.max(valid_charges):.2e}")
                print(f"  Mean: {np.mean(valid_charges):.2e}")
                print(f"  Total charge: {np.sum(valid_charges):.2e}")
            
            print(f"Charge fractions: {len(valid_fractions)} valid points")
            if len(valid_fractions) > 0:
                print(f"  Range: {np.min(valid_fractions):.4f} to {np.max(valid_fractions):.4f}")
                print(f"  Sum: {np.sum(valid_fractions):.4f}")
                
            print(f"Fit successful: {fit_successful[event_idx]}")
            print(f"X amplitude: {x_amplitude[event_idx]:.2e}")
            print(f"Y amplitude: {y_amplitude[event_idx]:.2e}")
            print(f"X center: {x_center[event_idx]:.3f} mm")
            print(f"Y center: {y_center[event_idx]:.3f} mm")
            print(f"X sigma: {x_sigma[event_idx]:.3f} mm")
            print(f"Y sigma: {y_sigma[event_idx]:.3f} mm")
        
        # Check if the issue is with charge magnitudes
        all_valid_charges = []
        for event_idx in non_pixel_indices:
            event_charges = charges[event_idx]
            valid_charges = event_charges[event_charges > 0]
            all_valid_charges.extend(valid_charges)
        
        if all_valid_charges:
            all_valid_charges = np.array(all_valid_charges)
            print(f"\n=== Overall Charge Statistics ===")
            print(f"Total valid charge points: {len(all_valid_charges)}")
            print(f"Charge range: {np.min(all_valid_charges):.2e} to {np.max(all_valid_charges):.2e} Coulombs")
            print(f"Mean charge: {np.mean(all_valid_charges):.2e} Coulombs")
            print(f"Median charge: {np.median(all_valid_charges):.2e} Coulombs")
            
            # Convert to electrons for reference
            elementary_charge = 1.602176634e-19  # Coulombs
            charges_in_electrons = all_valid_charges / elementary_charge
            print(f"\nIn electron units:")
            print(f"Range: {np.min(charges_in_electrons):.0f} to {np.max(charges_in_electrons):.0f} electrons")
            print(f"Mean: {np.mean(charges_in_electrons):.0f} electrons")
            
        # Analyze fit amplitudes
        successful_fits = fit_successful & non_pixel_mask
        if np.any(successful_fits):
            successful_x_amp = x_amplitude[successful_fits]
            successful_y_amp = y_amplitude[successful_fits]
            print(f"\n=== Successful Fit Amplitudes ===")
            print(f"X amplitudes range: {np.min(successful_x_amp):.2e} to {np.max(successful_x_amp):.2e}")
            print(f"Y amplitudes range: {np.min(successful_y_amp):.2e} to {np.max(successful_y_amp):.2e}")
        else:
            print(f"\n=== No successful fits found! ===")
            failed_x_amp = x_amplitude[non_pixel_mask]
            failed_y_amp = y_amplitude[non_pixel_mask]
            print(f"Failed X amplitudes: {failed_x_amp[:10]}")  # First 10
            print(f"Failed Y amplitudes: {failed_y_amp[:10]}")

if __name__ == "__main__":
    analyze_fitting_data() 