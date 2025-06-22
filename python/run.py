#!/usr/bin/env python3
"""
Simple batch simulation runner for PixelChargeSharingToy
Runs simulations with and without charge uncertainties by modifying Constants.hh
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path
import time
from datetime import datetime

def modify_constants_file(enable_uncertainties=True):
    """Modify the Constants.hh file to enable/disable charge uncertainties."""
    constants_file = Path("include/Constants.hh")
    
    if not constants_file.exists():
        print(f"Error: {constants_file} not found!")
        return False
    
    # Read the file
    with open(constants_file, 'r') as f:
        content = f.read()
    
    # Replace the line
    old_line = "const G4bool ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = true;"
    new_line = f"const G4bool ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = {'true' if enable_uncertainties else 'false'};"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        # Try the false version too
        old_line_false = "const G4bool ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = false;"
        if old_line_false in content:
            content = content.replace(old_line_false, new_line)
        else:
            print("Error: Could not find ENABLE_VERTICAL_CHARGE_UNCERTAINTIES line to modify!")
            return False
    
    # Write back
    with open(constants_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Set ENABLE_VERTICAL_CHARGE_UNCERTAINTIES = {'true' if enable_uncertainties else 'false'}")
    return True

def build_project():
    """Build the project with cmake."""
    print("Building project...")
    
    # Create build directory if it doesn't exist
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Run cmake and make
    os.chdir("build")
    
    try:
        # Configure with cmake
        result = subprocess.run(["cmake", ".."], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"CMake failed: {result.stderr}")
            return False
        
        # Build with make
        result = subprocess.run(["make", "-j2"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Make failed: {result.stderr}")
            return False
        
        print("✓ Build successful")
        return True
        
    except Exception as e:
        print(f"Build error: {e}")
        return False
    finally:
        os.chdir("..")

def run_simulation(output_dir, macro_file="macros/run.mac"):
    """Run the simulation."""
    executable = Path("build/epicChargeSharing")
    
    if not executable.exists():
        print(f"Error: Executable {executable} not found!")
        return False
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory
    original_cwd = Path(os.getcwd())
    os.chdir(output_dir)
    
    try:
        # Run simulation
        cmd = [str(original_cwd / executable), str(original_cwd / macro_file)]
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Simulation completed in {duration:.1f} seconds")
        
        if result.returncode == 0:
            print("✓ Simulation successful")
            return True
        else:
            print(f"✗ Simulation failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Simulation error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """Main function."""
    print("PixelChargeSharingToy Batch Simulation Runner")
    print("=" * 50)
    
    base_dir = Path("results")
    
    with_uncertainties_dir = base_dir / "with_uncertainties"
    without_uncertainties_dir = base_dir / "without_uncertainties"
    
    total_start_time = time.time()
    
    # Run with uncertainties enabled
    print(f"\n{'='*50}")
    print("RUNNING WITH CHARGE UNCERTAINTIES")
    print(f"{'='*50}")
    
    if not modify_constants_file(enable_uncertainties=True):
        return 1
    
    if not build_project():
        return 1
    
    if not run_simulation(with_uncertainties_dir):
        return 1
    
    # Run with uncertainties disabled
    print(f"\n{'='*50}")
    print("RUNNING WITHOUT CHARGE UNCERTAINTIES")
    print(f"{'='*50}")
    
    if not modify_constants_file(enable_uncertainties=False):
        return 1
    
    if not build_project():
        return 1
    
    if not run_simulation(without_uncertainties_dir):
        return 1
    
    # Restore original state (with uncertainties enabled)
    modify_constants_file(enable_uncertainties=True)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Final summary
    print(f"\n{'='*50}")
    print("BATCH SIMULATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total time: {total_duration:.1f} seconds")
    print(f"Results saved in:")
    print(f"  With uncertainties:    {with_uncertainties_dir.absolute()}")
    print(f"  Without uncertainties: {without_uncertainties_dir.absolute()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
