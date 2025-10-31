#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
# Resolve repository root relative to this script so the tool works anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_FILE = REPO_ROOT / "src" / "PrimaryGenerator.cc"
BUILD_DIR = REPO_ROOT / "build"
EXECUTABLE = BUILD_DIR / "epicChargeSharing"
RUN_MAC = (BUILD_DIR / "run.mac") if (BUILD_DIR / "run.mac").exists() else (REPO_ROOT / "macros" / "run.mac")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "sweep_x_runs"

# Positions to sweep (micrometers)
# First group: [+-25, +-50, ..., +-175]
GROUP1 = [25, 50, 75, 100, 125, 150, 175]
# Second group: [+-325, +-350, ..., +-475]
GROUP2 = []#[325, 350, 375, 400, 425, 450, 475]
POSITIONS = [0]
for p in GROUP1:
    POSITIONS.extend([p, -p])
for p in GROUP2:
    POSITIONS.extend([p, -p])

# Regex pattern to locate the x position line in PrimaryGenerator.cc
# We expect a line like:     const G4double x = -25.0*um;
X_LINE_REGEX = re.compile(r"^(\s*)const\s+G4double\s+x\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\*\s*um\s*;\s*$")


def determine_output_dir(output_dir_arg: Optional[str]) -> Path:
    if output_dir_arg:
        candidate = Path(output_dir_arg)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = (DEFAULT_OUTPUT_BASE / timestamp).resolve()

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def read_file_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def update_primary_generator_x_um(source_text: str, x_um: float) -> str:
    lines = source_text.splitlines()
    replaced = False
    new_lines = []
    for line in lines:
        m = X_LINE_REGEX.match(line)
        if m:
            indent = m.group(1)
            value_str = f"{x_um:.1f}"
            newline = f"{indent}const G4double x = {value_str}*um;"
            new_lines.append(newline)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        raise RuntimeError("Could not find x position assignment line in PrimaryGenerator.cc")
    # Preserve trailing newline presence
    return "\n".join(new_lines) + ("\n" if source_text.endswith("\n") else "")


def run_cmd(cmd, cwd: Path = None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def configure_build_if_needed():
    BUILD_DIR.mkdir(exist_ok=True)
    if not (BUILD_DIR / "Makefile").exists():
        run_cmd(["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)])


def build_project():
    run_cmd(["cmake", "--build", str(BUILD_DIR), "-j"])


def run_simulation():
    if not EXECUTABLE.exists():
        raise RuntimeError(f"Executable not found: {EXECUTABLE}")
    if not RUN_MAC.exists():
        raise RuntimeError(f"Run macro not found: {RUN_MAC}")
    # Multithreaded by default (let executable decide threads); remove -t 1
    run_cmd([str(EXECUTABLE), "-m", str(RUN_MAC)])


def wait_for_file(path: Path, timeout_s: int = 120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(1)
    return path.exists()


def wait_for_root_close(path: Path, timeout_s: int = 60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            try:
                with open(path, "ab"):
                    return
            except Exception:
                time.sleep(0.5)
        else:
            time.sleep(0.5)


def find_output_root() -> Path:
    # After MT run, final merged file is epicChargeSharing.root in working dir
    candidates = [
        REPO_ROOT / "epicChargeSharing.root",
        BUILD_DIR / "epicChargeSharing.root",
    ]
    # Wait for merged file to appear
    for c in candidates:
        if wait_for_file(c, timeout_s=180):
            return c
    raise RuntimeError("Expected merged ROOT output file not found after run.")


def rename_output_to(target_name: Path):
    source = find_output_root()
    wait_for_root_close(source)
    target_name.parent.mkdir(parents=True, exist_ok=True)
    if target_name.exists():
        target_name.unlink()
    shutil.move(str(source), str(target_name))

def main():
    parser = argparse.ArgumentParser(
        description="Sweep x positions and collect output ROOT files into a dedicated directory."
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Directory to store output ROOT files. If relative, resolved against the repository root. Default: timestamped folder under sweep_x_runs/.",
    )
    args = parser.parse_args()

    os.chdir(str(REPO_ROOT))

    configure_build_if_needed()

    original_text = read_file_text(SRC_FILE)

    output_dir = determine_output_dir(args.output_dir)
    print(f"Writing output ROOT files to {output_dir}")

    try:
        for x_um in POSITIONS:
            out_name = output_dir / (
                f"{int(x_um)}um.root" if float(x_um).is_integer() else f"{x_um}um.root"
            )

            print(f"=== Running for x = {x_um} um (multithreaded) ===")

            # Update source
            new_text = update_primary_generator_x_um(original_text, float(x_um))
            write_file_text(SRC_FILE, new_text)

            # Build
            build_project()

            # Run
            run_simulation()

            # Rename/move output (merged file)
            rename_output_to(out_name)

    finally:
        # Restore original source
        try:
            write_file_text(SRC_FILE, original_text)
        except Exception as e:
            print(f"Warning: failed to restore original source: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
