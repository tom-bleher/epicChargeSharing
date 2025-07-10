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

def get_project_root():
    """Get the project root directory, handling different execution contexts."""
    current_dir = Path(os.getcwd())
    
    # If we're in the python directory, go up one level
    if current_dir.name == "python":
        return current_dir.parent
    
    # If we're in the project root, return current directory
    if (current_dir / "include" / "Constants.hh").exists():
        return current_dir
    
    # If we're in build directory, go up one level
    if current_dir.name == "build":
        return current_dir.parent
    
    # Otherwise, assume we're in project root
    return current_dir

def modify_constants_file(enable_uncertainties=True):
    """Modify the Constants.hh file to enable/disable charge uncertainties."""
    project_root = get_project_root()
    constants_file = project_root / "include" / "Constants.hh"
    
    print(f"Debug: Current working directory: {os.getcwd()}")
    print(f"Debug: Project root: {project_root}")
    print(f"Debug: Looking for constants file at: {constants_file}")
    print(f"Debug: Constants file exists: {constants_file.exists()}")
    
    if not constants_file.exists():
        print(f"Error: {constants_file} not found!")
        return False
    
    # Read the file
    with open(constants_file, 'r') as f:
        content = f.read()
    
    # Replace the line
    old_line = "const G4bool ENABLE_VERT_CHARGE_ERR = true;"
    new_line = f"const G4bool ENABLE_VERT_CHARGE_ERR = {'true' if enable_uncertainties else 'false'};"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        # Try the false version too
        old_line_false = "const G4bool ENABLE_VERT_CHARGE_ERR = false;"
        if old_line_false in content:
            content = content.replace(old_line_false, new_line)
        else:
            print("Error: Could not find ENABLE_VERT_CHARGE_ERR line to modify!")
            return False
    
    # Write back
    with open(constants_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Set ENABLE_VERT_CHARGE_ERR = {'true' if enable_uncertainties else 'false'}")
    return True

def build_project():
    """Build the project with cmake."""
    print("Building project...")
    
    project_root = get_project_root()
    build_dir = project_root / "build"
    build_dir.mkdir(exist_ok=True)
    
    print(f"Debug: Build directory: {build_dir}")
    
    # Run cmake and make
    original_cwd = Path(os.getcwd())
    os.chdir(build_dir)
    
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
        
        print("✓ Build success")
        return True
        
    except Exception as e:
        print(f"Build error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def run_simulation(output_dir, macro_file=None):
    """Run the simulation."""
    project_root = get_project_root()
    executable = project_root / "build" / "epicChargeSharing"
    
    if macro_file is None:
        macro_file = project_root / "macros" / "run.mac"
    else:
        macro_file = project_root / macro_file
    
    print(f"Debug: Executable path: {executable}")
    print(f"Debug: Executable exists: {executable.exists()}")
    print(f"Debug: Macro file: {macro_file}")
    print(f"Debug: Macro file exists: {macro_file.exists()}")
    
    if not executable.exists():
        print(f"Error: Executable {executable} not found!")
        return False
    
    if not macro_file.exists():
        print(f"Error: Macro file {macro_file} not found!")
        return False
    
    # Create output directory
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory
    original_cwd = Path(os.getcwd())
    os.chdir(output_dir)
    
    try:
        # Run simulation
        cmd = [str(executable), str(macro_file)]
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Simulation completed in {duration:.1f} seconds")
        
        if result.returncode == 0:
            print("✓ Simulation success")
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
    
    project_root = get_project_root()
    base_dir = project_root / "results"
    
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
