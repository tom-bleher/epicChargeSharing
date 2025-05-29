#!/usr/bin/env python3
"""
ROOT to Python Dictionary Converter
Reads the epicToyOutput.root file and converts all data to Python dictionaries
"""

import uproot
import numpy as np
from pathlib import Path
import json


def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-serializable format
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def read_root_file(root_file_path):
    """
    Read ROOT file and convert all data to Python dictionary
    
    Args:
        root_file_path (str): Path to the ROOT file
        
    Returns:
        dict: Dictionary containing all ROOT file data
    """
    
    # Check if file exists
    if not Path(root_file_path).exists():
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")
    
    print(f"Reading ROOT file: {root_file_path}")
    
    # Open ROOT file
    with uproot.open(root_file_path) as root_file:
        data_dict = {}
        
        # Get all keys in the ROOT file
        root_keys = list(root_file.keys())
        print(f"Found keys in ROOT file: {root_keys}")
        
        # Process each key
        for key in root_keys:
            obj = root_file[key]
            key_name = key.split(';')[0]  # Remove version number
            
            if hasattr(obj, 'keys'):  # This is a TTree
                print(f"Processing TTree: {key_name}")
                tree_data = {}
                
                # Get all branches
                branches = list(obj.keys())
                print(f"  Branches: {branches}")
                
                # Read all data from the tree
                arrays = obj.arrays(library="np")
                
                # Convert to regular Python data types
                for branch_name, array in arrays.items():
                    # Convert everything to serializable format
                    tree_data[branch_name] = convert_to_serializable(array)
                
                data_dict[key_name] = tree_data
                print(f"  Converted {len(tree_data)} branches")
                
            else:
                # Handle other ROOT objects (histograms, parameters, etc.)
                print(f"Processing ROOT object: {key_name} (type: {type(obj)})")
                try:
                    # Handle TNamed objects specifically
                    if hasattr(obj, 'members'):
                        # For TNamed objects, extract the title which often contains the value
                        members = obj.members
                        if 'fTitle' in members:
                            title = members['fTitle']
                            name = members.get('fName', key_name)
                            
                            # Try to parse the title as a number if possible
                            try:
                                if '.' in title:
                                    data_dict[key_name] = float(title)
                                else:
                                    data_dict[key_name] = int(title)
                            except ValueError:
                                # If not a number, store as string
                                data_dict[key_name] = title
                            
                            print(f"  Extracted value: {data_dict[key_name]} (from title: '{title}')")
                        else:
                            data_dict[key_name] = str(members)
                    
                    # Try other methods
                    elif hasattr(obj, 'values'):
                        data_dict[key_name] = obj.values().tolist()
                    elif hasattr(obj, 'member'):
                        try:
                            data_dict[key_name] = obj.member('fVal')
                        except:
                            data_dict[key_name] = str(obj)
                    else:
                        data_dict[key_name] = str(obj)
                except Exception as e:
                    print(f"  Warning: Could not convert {key_name}: {e}")
                    data_dict[key_name] = f"<{type(obj).__name__}>"
    
    return data_dict


def save_dict_to_json(data_dict, output_path):
    """
    Save dictionary to JSON file
    
    Args:
        data_dict (dict): Dictionary to save
        output_path (str): Output JSON file path
    """
    print(f"Saving data to JSON: {output_path}")
    # Ensure everything is JSON serializable
    serializable_dict = convert_to_serializable(data_dict)
    with open(output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def print_data_summary(data_dict):
    """
    Print a summary of the converted data
    
    Args:
        data_dict (dict): Dictionary containing the data
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for key, value in data_dict.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    print(f"  {subkey}: list with {len(subvalue)} entries")
                    if len(subvalue) > 0:
                        first_elem = subvalue[0]
                        if isinstance(first_elem, list):
                            print(f"    (nested list, first entry has {len(first_elem)} elements)")
                        else:
                            print(f"    (type: {type(first_elem).__name__})")
                else:
                    print(f"  {subkey}: {type(subvalue).__name__}")
        else:
            print(f"  Type: {type(value).__name__}")
            if isinstance(value, list):
                print(f"  Length: {len(value)}")
            else:
                print(f"  Value: {value}")


def demonstrate_data_usage(data_dict):
    """
    Demonstrate how to use the converted ROOT data as a Python dictionary
    
    Args:
        data_dict (dict): Dictionary containing the ROOT data
    """
    print("\n" + "="*60)
    print("DATA USAGE DEMONSTRATION")
    print("="*60)
    
    if 'Hits' in data_dict:
        hits = data_dict['Hits']
        
        # Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"  Total number of hits: {len(hits['Edep'])}")
        
        # Energy deposition statistics
        edep = hits['Edep']
        print(f"  Energy deposition (MeV):")
        print(f"    Min: {min(edep):.6f}")
        print(f"    Max: {max(edep):.6f}")
        print(f"    Mean: {sum(edep)/len(edep):.6f}")
        
        # Pixel hit statistics
        pixel_hits = hits['PixelHit']
        hit_count = sum(pixel_hits)
        print(f"  Pixel hits: {hit_count}/{len(pixel_hits)} ({hit_count/len(pixel_hits)*100:.1f}%)")
        
        # Position ranges
        print(f"  Position ranges:")
        print(f"    X: {min(hits['TrueX']):.3f} to {max(hits['TrueX']):.3f}")
        print(f"    Y: {min(hits['TrueY']):.3f} to {max(hits['TrueY']):.3f}")
        print(f"    Z: {min(hits['TrueZ']):.3f} to {max(hits['TrueZ']):.3f}")
        
        # Grid neighborhood example
        neighborhoods = hits['GridNeighborhoodAngles']
        valid_neighborhoods = [n for n in neighborhoods if len(n) > 0 and n[0] != -999.0]
        print(f"  Valid grid neighborhoods: {len(valid_neighborhoods)}/{len(neighborhoods)}")
        
        if valid_neighborhoods:
            first_valid = valid_neighborhoods[0]
            print(f"    First valid neighborhood has {len(first_valid)} angles")
    
    # Configuration parameters
    print(f"\n⚙️  Configuration Parameters:")
    config_params = ['GridPixelSize', 'GridPixelSpacing', 'GridPixelCornerOffset', 
                     'GridDetectorSize', 'GridNumBlocksPerSide']
    for param in config_params:
        if param in data_dict:
            print(f"  {param}: {data_dict[param]}")
    
    print(f"\n✨ Example: Access first 5 energy depositions:")
    if 'Hits' in data_dict:
        first_5_edep = data_dict['Hits']['Edep'][:5]
        for i, edep in enumerate(first_5_edep):
            x = data_dict['Hits']['TrueX'][i]
            y = data_dict['Hits']['TrueY'][i]
            z = data_dict['Hits']['TrueZ'][i]
            print(f"    Hit {i+1}: E={edep:.6f} MeV at ({x:.3f}, {y:.3f}, {z:.3f})")


def main():
    """Main function to read ROOT file and convert to Python dictionary"""
    
    # Path to the ROOT file
    root_file_path = "/home/tom/Desktop/Cultural_Keys/epicToy/build/epicToyOutput.root"
    
    try:
        # Read ROOT file and convert to dictionary
        data_dict = read_root_file(root_file_path)
        
        # Print summary
        print_data_summary(data_dict)
        
        # Demonstrate usage
        demonstrate_data_usage(data_dict)
        
        # Optionally save to JSON file
        output_json = "epicToy_data.json"
        save_dict_to_json(data_dict, output_json)
        
        print(f"\n✅ Successfully converted ROOT file to Python dictionary!")
        print(f"📁 Data saved to: {output_json}")
        print(f"📖 You can now load the data with: data = json.load(open('{output_json}'))")
        
        return data_dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    data = main()
