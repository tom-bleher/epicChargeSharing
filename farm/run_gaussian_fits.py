#!/usr/bin/env python3

"""Batch runner for `plotFitGaus1DReplayQiFit` over multiple ROOT files.

This script iterates over the configured datasets, invokes the ROOT macro for
each `.root` file, and places the resulting PDFs under `gaussian_fits/<dataset>`.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_DISTANCE_ARGS = {
    "use_distance_weighted_errors": False,
    "distance_error_scale_pixels": 1.5,
    "distance_error_exponent": 1.5,
    "distance_error_floor_percent": 4.0,
    "distance_error_cap_percent": 10.0,
    "distance_error_prefer_truth_center": True,
    "distance_error_power_inverse": True,
}


def _dataset(distance_error_power_inverse: bool, **kwargs: object) -> Dict[str, object]:
    cfg: Dict[str, object] = {**DEFAULT_DISTANCE_ARGS, **kwargs}
    cfg["distance_error_power_inverse"] = distance_error_power_inverse
    return cfg


DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "sweep_x_runs/log": _dataset(
        distance_error_power_inverse=False,
        error_percent=0.0,
        use_qiqn_errors=False,
    ),
    "sweep_x_runs/logInv": _dataset(
        distance_error_power_inverse=True,
        error_percent=0.0,
        use_qiqn_errors=False,
    ),
    "sweep_x_runs/linearB0.001": _dataset(
        distance_error_power_inverse=False,
        error_percent=0.0,
        use_qiqn_errors=False,
        distance_error_exponent=1.0,
    ),
    "sweep_x_runs/linearB0.001Inv": _dataset(
        distance_error_power_inverse=True,
        error_percent=0.0,
        use_qiqn_errors=False,
        distance_error_exponent=1.0,
    ),
}


def _um_value_key(path: Path) -> float:
    stem = path.stem
    if stem.endswith("um"):
        stem = stem[:-2]
    try:
        return float(stem)
    except ValueError:
        return float("inf")


def _find_root_files(dataset_dir: Path) -> List[Path]:
    return sorted((p for p in dataset_dir.glob("*.root") if p.is_file()), key=_um_value_key)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def run_macro(
    root_executable: str,
    macro_path: Path,
    root_file: Path,
    output_pdf: Path,
    error_percent: float,
    use_qiqn_errors: bool,
    n_events: int,
    plot_qi_overlay: bool,
    do_qi_fit: bool,
    use_distance_weighted_errors: bool,
    distance_error_scale_pixels: float,
    distance_error_exponent: float,
    distance_error_floor_percent: float,
    distance_error_cap_percent: float,
    distance_error_prefer_truth_center: bool,
    distance_error_power_inverse: bool,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    macro_call = (
        f"{macro_path.as_posix()}(\"{root_file.as_posix()}\", "
        f"{error_percent:.6f}, {n_events}, {_bool_str(plot_qi_overlay)}, {_bool_str(do_qi_fit)}, "
        f"{_bool_str(use_qiqn_errors)}, \"{output_pdf.as_posix()}\", "
        f"{_bool_str(use_distance_weighted_errors)}, {distance_error_scale_pixels:.6f}, "
        f"{distance_error_exponent:.6f}, {distance_error_floor_percent:.6f}, "
        f"{distance_error_cap_percent:.6f}, {_bool_str(distance_error_prefer_truth_center)}, "
        f"{_bool_str(distance_error_power_inverse)})"
    )
    subprocess.run([root_executable, "-l", "-b", "-q", macro_call], check=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-bin",
        default="root",
        help="Root executable to use (default: %(default)s)",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=200,
        help="Number of random events/pages to plot (pass a negative value for all events)",
    )
    parser.add_argument(
        "--skip-qi-overlay",
        action="store_true",
        help="Disable plotting Q_i overlay points",
    )
    parser.add_argument(
        "--skip-qi-fit",
        action="store_true",
        help="Disable refitting Q_i curves",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_CONFIG),
        help="Specific dataset directories to process (defaults to all configured datasets)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    macro_path = project_root / "proc" / "fit" / "plotFitGaus1DReplayQiFit.C"
    gaussian_fits_dir = project_root / "gaussian_fits"

    missing_datasets = [d for d in args.datasets if d not in DATASET_CONFIG]
    if missing_datasets:
        print(f"Unknown dataset(s): {', '.join(missing_datasets)}", file=sys.stderr)
        return 1

    for dataset_name in args.datasets:
        dataset_dir = project_root / dataset_name
        if not dataset_dir.is_dir():
            print(f"Skipping {dataset_name}: directory not found at {dataset_dir}", file=sys.stderr)
            continue

        cfg = DATASET_CONFIG[dataset_name]
        error_percent = float(cfg["error_percent"])
        use_qiqn_errors = bool(cfg["use_qiqn_errors"])
        use_distance_weighted_errors = bool(cfg["use_distance_weighted_errors"])
        distance_error_scale_pixels = float(cfg["distance_error_scale_pixels"])
        distance_error_exponent = float(cfg["distance_error_exponent"])
        distance_error_floor_percent = float(cfg["distance_error_floor_percent"])
        distance_error_cap_percent = float(cfg["distance_error_cap_percent"])
        distance_error_prefer_truth_center = bool(cfg["distance_error_prefer_truth_center"])
        distance_error_power_inverse = bool(cfg["distance_error_power_inverse"])

        root_files = _find_root_files(dataset_dir)
        if not root_files:
            print(f"No ROOT files found in {dataset_dir}", file=sys.stderr)
            continue

        dataset_output_dir = gaussian_fits_dir / dataset_name

        print(f"Processing dataset {dataset_name} ({len(root_files)} files)...")
        for root_file in root_files:
            output_pdf = dataset_output_dir / f"{root_file.stem}.pdf"
            try:
                run_macro(
                    args.root_bin,
                    macro_path,
                    root_file.resolve(),
                    output_pdf,
                    error_percent,
                    use_qiqn_errors,
                    args.n_events,
                    not args.skip_qi_overlay,
                    not args.skip_qi_fit,
                    use_distance_weighted_errors,
                    distance_error_scale_pixels,
                    distance_error_exponent,
                    distance_error_floor_percent,
                    distance_error_cap_percent,
                    distance_error_prefer_truth_center,
                    distance_error_power_inverse,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"ROOT macro failed for {root_file} (exit code {exc.returncode}).",
                    file=sys.stderr,
                )
                return exc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

