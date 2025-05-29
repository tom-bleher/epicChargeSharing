#!/usr/bin/env python3
"""
Script to plot energy deposition (Edep) vs Z position for events.
This visualizes the step-by-step energy deposition pattern along the Z axis,
clearly distinguishing between pixel hits and detector body hits.
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import sys
from pathlib import Path

def read_detector_parameters_from_root(file):
    """
    Read detector parameters from ROOT file metadata.
    Returns a dictionary with all detector parameters.
    """
    params = {}
    
    # Fixed parameters from DetectorConstruction.cc
    params['detector_z'] = -10.0  # mm, detector center position
    params['detector_width'] = 0.05  # mm, 50 µm detector thickness
    params['pixel_width'] = 0.001  # mm, 1 µm pixel thickness
    
    # Try to read grid parameters from ROOT metadata
    try:
        # Read metadata TNamed objects
        pixel_size_obj = file.get("GridPixelSize")
        pixel_spacing_obj = file.get("GridPixelSpacing") 
        pixel_corner_offset_obj = file.get("GridPixelCornerOffset")
        detector_size_obj = file.get("GridDetectorSize")
        num_blocks_obj = file.get("GridNumBlocksPerSide")
        
        if all([pixel_size_obj, pixel_spacing_obj, pixel_corner_offset_obj, 
               detector_size_obj, num_blocks_obj]):
            # Extract values from TNamed objects (access fTitle member)
            params['pixel_size'] = float(pixel_size_obj.member('fTitle'))
            params['pixel_spacing'] = float(pixel_spacing_obj.member('fTitle'))
            params['pixel_corner_offset'] = float(pixel_corner_offset_obj.member('fTitle'))
            params['detector_size'] = float(detector_size_obj.member('fTitle'))
            params['num_blocks_per_side'] = int(num_blocks_obj.member('fTitle'))
            
            print("✓ Successfully read grid parameters from ROOT metadata")
        else:
            raise Exception("Some metadata objects not found")
            
    except Exception as e:
        print(f"WARNING: Could not read grid parameters from ROOT metadata: {e}")
        print("Using fallback default values")
        # Fallback values
        params['pixel_size'] = 0.1  # mm
        params['pixel_spacing'] = 0.5  # mm
        params['pixel_corner_offset'] = 0.1  # mm
        params['detector_size'] = 29.8  # mm
        params['num_blocks_per_side'] = 60
    
    # Calculate derived positions
    params['detector_front_surface_z'] = params['detector_z'] + params['detector_width']/2
    params['detector_back_surface_z'] = params['detector_z'] - params['detector_width']/2
    params['pixel_surface_z'] = params['detector_front_surface_z'] + params['pixel_width']/2
    
    return params

def select_representative_events(data):
    """
    Select one event with pixel hit and one without pixel hit for comparison.
    Returns (pixel_event_idx, detector_event_idx) or (None, None) if not found.
    """
    
    # Find events with energy deposition
    events_with_energy = np.where(data["Edep"] > 0)[0]
    
    if len(events_with_energy) == 0:
        return None, None
    
    # Check if PixelHit data is available
    if "PixelHit" not in data:
        print("WARNING: PixelHit data not available, selecting first two events with energy")
        if len(events_with_energy) >= 2:
            return events_with_energy[0], events_with_energy[1]
        else:
            return None, None
    
    # Find pixel hit events
    pixel_events = events_with_energy[data["PixelHit"][events_with_energy] == True]
    detector_events = events_with_energy[data["PixelHit"][events_with_energy] == False]
    
    pixel_event = pixel_events[0] if len(pixel_events) > 0 else None
    detector_event = detector_events[0] if len(detector_events) > 0 else None
    
    return pixel_event, detector_event

def plot_edep_vs_z(root_file_path, random_seed=42):
    """
    Plot energy deposition vs Z position comparing pixel hit vs detector body hit.
    
    Args:
        root_file_path: Path to the ROOT file
        random_seed: Random seed for reproducibility
    """
    print(f"Loading data from: {root_file_path}")
    
    try:
        with uproot.open(root_file_path) as file:
            # Read detector parameters
            params = read_detector_parameters_from_root(file)
            
            tree = file["Hits"]
            
            # Load all data
            print("Loading simulation data...")
            data = tree.arrays([
                "Edep", "TrueZ", "EventID", "PixelHit",
                "AllStepEnergyDeposition", "AllStepZPosition", "AllStepTime"
            ], library="np")
            
            n_total_events = len(data["Edep"])
            print(f"Total events in file: {n_total_events}")
            
            # Select representative events
            pixel_event_idx, detector_event_idx = select_representative_events(data)
            
            if pixel_event_idx is None and detector_event_idx is None:
                print("ERROR: Could not find events with energy deposition!")
                return False
            
            # Create the plot with better layout
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # Define colors and styles
            pixel_color = '#FF6B35'  # Orange-red for pixel hits
            detector_color = '#004E98'  # Dark blue for detector hits
            
            events_to_plot = []
            if pixel_event_idx is not None:
                events_to_plot.append(("Pixel Hit", pixel_event_idx, pixel_color, 'o'))
            if detector_event_idx is not None:
                events_to_plot.append(("Detector Body Hit", detector_event_idx, detector_color, 's'))
            
            print(f"\nSelected events for comparison:")
            for label, idx, color, marker in events_to_plot:
                event_id = data["EventID"][idx] if "EventID" in data else idx
                total_edep = data["Edep"][idx]
                true_z = data["TrueZ"][idx]
                pixel_hit = data["PixelHit"][idx] if "PixelHit" in data else "Unknown"
                print(f"  {label}: Event {event_id}, Edep = {total_edep*1000:.1f} keV, TrueZ = {true_z:.3f} mm, PixelHit = {pixel_hit}")
            
            # Plot cumulative energy deposition
            for label, event_idx, color, marker in events_to_plot:
                step_edeps = data["AllStepEnergyDeposition"][event_idx]
                step_z_positions = data["AllStepZPosition"][event_idx]
                total_edep = data["Edep"][event_idx]
                true_z = data["TrueZ"][event_idx]
                event_id = data["EventID"][event_idx] if "EventID" in data else event_idx
                
                # Cumulative energy deposition
                if len(step_z_positions) > 0:
                    # Sort by Z position for cumulative plot
                    sorted_indices = np.argsort(step_z_positions)
                    sorted_z = step_z_positions[sorted_indices]
                    sorted_edep = step_edeps[sorted_indices]
                    cumulative_edep = np.cumsum(sorted_edep)
                    
                    # Only plot points where energy is deposited
                    energy_indices = sorted_edep > 0
                    if np.any(energy_indices):
                        ax.plot(sorted_z[energy_indices], cumulative_edep[energy_indices] * 1000, 
                                marker=marker, color=color, alpha=0.9, linewidth=3, markersize=8,
                                markeredgecolor='white', markeredgewidth=1,
                                label=f'{label}')
                        
                        # Annotate final energy
                        final_energy = cumulative_edep[-1] * 1000
                        final_z = sorted_z[-1]
                        ax.annotate(f'{final_energy:.1f} keV', 
                                   xy=(final_z, final_energy),
                                   xytext=(15, 15), textcoords='offset points',
                                   fontsize=11, color=color,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                           edgecolor=color, alpha=0.8))
            
            # Add detector geometry visualization
            ax.axvline(params['detector_front_surface_z'], color='red', linestyle='-', alpha=0.8,
                      linewidth=3, label='Detector face')
            ax.axvline(params['detector_back_surface_z'], color='red', linestyle='-', alpha=0.8,
                      linewidth=3)
            
            ax.axvline(params['pixel_surface_z'], color='blue', linestyle='-', alpha=0.8,
                      linewidth=2, label='Pixel surface')
            
            ax.axvspan(params['detector_back_surface_z'], params['detector_front_surface_z'], 
                      alpha=0.15, color='red', label='Detector volume (Si)')
            
            ax.axvspan(params['detector_front_surface_z'], params['pixel_surface_z'], 
                      alpha=0.15, color='blue', label='Pixel layer (Al)')
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlabel('Z Position (mm)', fontsize=14)
            
            z_range = params['pixel_surface_z'] - params['detector_back_surface_z']
            z_padding = z_range * 0.3
            ax.set_xlim(params['detector_back_surface_z'] - z_padding, 
                       params['pixel_surface_z'] + z_padding)
            
            ax.set_ylabel('Cumulative Deposited Energy (keV)', fontsize=14)
            ax.set_title('Cumulative Energy Deposition Along Particle Path', 
                         fontsize=16, pad=20)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, framealpha=0.9)
            
            # Add particle direction indicator
            ax.text(0.02, 0.98, '← Particle Direction', transform=ax.transAxes, 
                    fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Print detailed geometry and event information
            print(f"\nDetector Geometry Summary:")
            print(f"  Detector center Z: {params['detector_z']:.3f} mm")
            print(f"  Detector thickness: {params['detector_width']:.3f} mm")
            print(f"  Pixel thickness: {params['pixel_width']:.3f} mm")
            print(f"  Front surface Z: {params['detector_front_surface_z']:.3f} mm")
            print(f"  Back surface Z: {params['detector_back_surface_z']:.3f} mm")
            print(f"  Pixel surface Z: {params['pixel_surface_z']:.3f} mm")
            
            print(f"\nEvent Analysis:")
            for label, event_idx, color, marker in events_to_plot:
                step_edeps = data["AllStepEnergyDeposition"][event_idx]
                step_z_positions = data["AllStepZPosition"][event_idx]
                total_edep = data["Edep"][event_idx]
                true_z = data["TrueZ"][event_idx]
                event_id = data["EventID"][event_idx] if "EventID" in data else event_idx
                
                n_total_steps = len(step_edeps)
                n_energy_steps = np.sum(step_edeps > 0)
                z_min, z_max = np.min(step_z_positions), np.max(step_z_positions)
                
                print(f"\n  {label} (Event {event_id}):")
                print(f"    Total energy: {total_edep*1000:.1f} keV")
                print(f"    True Z: {true_z:.3f} mm")
                print(f"    Total steps: {n_total_steps}")
                print(f"    Energy steps: {n_energy_steps}")
                print(f"    Z range: {z_min:.3f} to {z_max:.3f} mm")
                
                # Check if particle passed through different detector regions
                passed_pixel = np.any((step_z_positions >= params['detector_front_surface_z']) & 
                                    (step_z_positions <= params['pixel_surface_z']))
                passed_detector = np.any((step_z_positions >= params['detector_back_surface_z']) & 
                                       (step_z_positions <= params['detector_front_surface_z']))
                
                print(f"    Passed through pixel layer: {passed_pixel}")
                print(f"    Passed through detector: {passed_detector}")
            
            # Save the plot
            output_file = "energy_deposition_comparison.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"\n✓ Plot saved as '{output_file}'")
            
            # Show the plot
            plt.show()
            
    except Exception as e:
        print(f"ERROR: Failed to create plot: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Plot energy deposition vs Z position comparing pixel hits vs detector body hits')
    parser.add_argument('--file', '-f', default='epicToyOutput.root', 
                       help='ROOT file to analyze (default: epicToyOutput.root)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for event selection (default: 42)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"ERROR: File '{args.file}' not found!")
        sys.exit(1)
    
    success = plot_edep_vs_z(args.file, args.seed)
    
    if success:
        print("\n✓ Energy deposition comparison completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Energy deposition comparison failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
