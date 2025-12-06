#!/usr/bin/env python3
"""
Wrapper script to run sweep_analysis.py with multiple configurations
for generating slide material.

Generates plots for:
- Results with ChargeBlock in Denominator (Log and Linear)
- Results with ChargeRow (RowCol) in Denominator (Log and Linear)

Each configuration produces:
- sigma_F vs d plots
- F_i vs d plots

Usage:
    python run_slides_sweep.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
SWEEP_SCRIPT = Path(__file__).parent / "sweep_analysis.py"
OUTPUT_BASE = Path(__file__).parent.parent / "slides_sweep_runs"

# Beta value for linear model
BETA_LINEAR = 0.001

# Configurations to run
# Each tuple: (name, charge_model, denominator_mode, beta)
CONFIGS = [
    ("ChargeBlock_Log", "LogA", "ChargeBlock", None),
    ("ChargeBlock_Linear", "LinA", "ChargeBlock", BETA_LINEAR),
    ("ChargeRow_Log", "LogA", "RowCol", None),
    ("ChargeRow_Linear", "LinA", "RowCol", BETA_LINEAR),
]


def run_sweep(name: str, charge_model: str, denominator_mode: str, beta: float | None) -> int:
    """Run sweep_analysis.py with the given configuration."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_BASE / f"{name}_{timestamp}"

    print("\n" + "=" * 80)
    print(f"Running configuration: {name}")
    print(f"  Charge model: {charge_model}")
    print(f"  Denominator mode: {denominator_mode}")
    print(f"  Beta: {beta}")
    print(f"  Output: {output_dir}")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "run",
        "--output-dir", str(output_dir),
        "--charge-model", charge_model,
        "--denominator-mode", denominator_mode,
    ]

    if beta is not None:
        cmd.extend(["--beta", str(beta)])

    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    print("=" * 80)
    print("Slides Sweep Analysis")
    print("=" * 80)
    print(f"Output base directory: {OUTPUT_BASE}")
    print(f"Configurations to run: {len(CONFIGS)}")
    print()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, charge_model, denominator_mode, beta in CONFIGS:
        try:
            ret = run_sweep(name, charge_model, denominator_mode, beta)
            if ret != 0:
                print(f"[ERROR] Configuration {name} failed with exit code {ret}")
                failed.append(name)
        except Exception as e:
            print(f"[ERROR] Configuration {name} failed with exception: {e}")
            failed.append(name)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configurations: {len(CONFIGS)}")
    print(f"Successful: {len(CONFIGS) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed configurations: {', '.join(failed)}")
    print(f"\nOutput directory: {OUTPUT_BASE}")
    print("\nGenerated plots for slides:")
    print("  - ChargeBlock (Log): sigma_F vs d, F_i vs d")
    print("  - ChargeBlock (Linear, beta=0.001): sigma_F vs d, F_i vs d")
    print("  - ChargeRow/RowCol (Log): sigma_F vs d, F_i vs d")
    print("  - ChargeRow/RowCol (Linear, beta=0.001): sigma_F vs d, F_i vs d")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
