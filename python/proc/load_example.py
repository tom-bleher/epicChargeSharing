#!/usr/bin/env python3
"""
Example script showing how to load and use the converted ROOT data
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    """Load the converted ROOT data from JSON file"""
    print("Loading epicToy data from JSON...")
    with open('epicToy_data.json', 'r') as f:
        data = json.load(f)
    print("✅ Data loaded successfully!")
    return data


def plot_energy_distribution(data):
    """Plot energy deposition distribution"""
    hits = data['Hits']
    edep = hits['Edep']
    
    plt.figure(figsize=(10, 6))
    plt.hist(edep, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Energy Deposition (MeV)')
    plt.ylabel('Frequency')
    plt.title('Energy Deposition Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_distribution.png', dpi=150)
    print("📊 Energy distribution plot saved as 'energy_distribution.png'")


def plot_hit_positions(data):
    """Plot hit positions in X-Y plane"""
    hits = data['Hits']
    x_pos = hits['TrueX']
    y_pos = hits['TrueY']
    edep = hits['Edep']
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_pos, y_pos, c=edep, cmap='viridis', alpha=0.6, s=1)
    plt.colorbar(scatter, label='Energy Deposition (MeV)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Hit Positions (colored by energy deposition)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hit_positions.png', dpi=150)
    print("📊 Hit positions plot saved as 'hit_positions.png'")


def analyze_pixel_hits(data):
    """Analyze pixel hit data"""
    hits = data['Hits']
    pixel_hits = hits['PixelHit']
    pixel_i = hits['PixelI']
    pixel_j = hits['PixelJ']
    edep = hits['Edep']
    
    # Filter only pixel hits
    hit_indices = [i for i, hit in enumerate(pixel_hits) if hit]
    
    print(f"\n🔍 Pixel Hit Analysis:")
    print(f"Total pixel hits: {len(hit_indices)}")
    
    if hit_indices:
        hit_energies = [edep[i] for i in hit_indices]
        hit_i_coords = [pixel_i[i] for i in hit_indices]
        hit_j_coords = [pixel_j[i] for i in hit_indices]
        
        print(f"Pixel hit energy stats:")
        print(f"  Mean: {np.mean(hit_energies):.6f} MeV")
        print(f"  Std:  {np.std(hit_energies):.6f} MeV")
        print(f"Pixel coordinate ranges:")
        print(f"  I: {min(hit_i_coords)} to {max(hit_i_coords)}")
        print(f"  J: {min(hit_j_coords)} to {max(hit_j_coords)}")


def analyze_grid_neighborhoods(data):
    """Analyze grid neighborhood data"""
    hits = data['Hits']
    neighborhoods = hits['GridNeighborhoodAngles']
    charge_fractions = hits['GridNeighborhoodChargeFractions']
    
    # Find valid neighborhoods (not -999)
    valid_neighborhoods = []
    valid_charges = []
    
    for i, (angles, charges) in enumerate(zip(neighborhoods, charge_fractions)):
        if len(angles) > 0 and angles[0] != -999.0:
            valid_neighborhoods.append(angles)
            valid_charges.append(charges)
    
    print(f"\n🔍 Grid Neighborhood Analysis:")
    print(f"Valid neighborhoods: {len(valid_neighborhoods)}")
    
    if valid_neighborhoods:
        neighborhood_sizes = [len(n) for n in valid_neighborhoods]
        print(f"Neighborhood sizes:")
        print(f"  Mean: {np.mean(neighborhood_sizes):.1f}")
        print(f"  Min:  {min(neighborhood_sizes)}")
        print(f"  Max:  {max(neighborhood_sizes)}")
        
        # Look at charge fractions
        all_charges = [charge for charges in valid_charges for charge in charges if charge != -999.0]
        if all_charges:
            print(f"Charge fraction stats:")
            print(f"  Mean: {np.mean(all_charges):.6f}")
            print(f"  Std:  {np.std(all_charges):.6f}")


def main():
    """Main function"""
    # Load the data
    data = load_data()
    
    # Print configuration
    print(f"\n⚙️ Configuration:")
    config_keys = ['GridPixelSize', 'GridPixelSpacing', 'GridDetectorSize', 'GridNumBlocksPerSide']
    for key in config_keys:
        if key in data:
            print(f"  {key}: {data[key]}")
    
    # Basic data info
    hits = data['Hits']
    print(f"\n📊 Data Overview:")
    print(f"  Total hits: {len(hits['Edep'])}")
    print(f"  Data keys: {list(hits.keys())}")
    
    # Perform analyses
    analyze_pixel_hits(data)
    analyze_grid_neighborhoods(data)
    
    # Create plots
    print(f"\n📈 Creating plots...")
    plot_energy_distribution(data)
    plot_hit_positions(data)
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 