#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2024-2026 Tom Bleher, Igor Korover
"""Truth-residual validator for the LGAD charge-sharing benchmark.

Reads the hist file produced by -Phistsfile=... (populated by the
LGAD_chargesharing_benchmark plugin) and enforces per-detector residual
thresholds on the (residualX, residualY) distributions.

The plugin writes two artifacts under the TDirectory LGADChargeSharing/:
  - per-detector histograms h<DetName>_residualX / _residualY / _residualR
  - a single TTree ``hits`` with scalar branches
    (trueX, reconX, residualX, residualY, residualR, detectorIndex, ...)

The validator prefers the TTree because it lets us filter by detector and
compute robust statistics, but falls back to the histograms if the tree is
missing.

Exit codes:
  0 - all checks pass
  1 - at least one check failed
  2 - file missing or unreadable
"""

import argparse
import math
import sys
from pathlib import Path


def _import_uproot():
    try:
        import uproot  # noqa: F401

        return uproot
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: uproot is required for benchmark validation ({exc})")
        print("       Install with: pip install uproot awkward numpy")
        sys.exit(2)


def _load_residuals_from_tree(uproot, path: Path, detector: str):
    """Return (residualX, residualY) numpy arrays filtered by detector name."""
    f = uproot.open(str(path))
    if "LGADChargeSharing/hits" not in f and "LGADChargeSharing" not in f:
        return None
    try:
        tree = f["LGADChargeSharing/hits"]
    except KeyError:
        return None

    branches = set(tree.keys())
    required = {"residualX", "residualY"}
    if not required.issubset(branches):
        return None

    arrays = tree.arrays(library="np")
    rx = arrays["residualX"]
    ry = arrays["residualY"]

    if "detectorIndex" in arrays:
        # DetectorConfig order in LGADChargeSharingMonitor.cc:
        #   0 = B0Tracker, 1 = LumiSpecTracker
        name_to_index = {"B0Tracker": 0, "LumiSpecTracker": 1}
        idx = name_to_index.get(detector)
        if idx is not None:
            mask = arrays["detectorIndex"] == idx
            rx = rx[mask]
            ry = ry[mask]
    return rx, ry


def _load_residual_rms_from_hist(uproot, path: Path, detector: str):
    """Return (rmsX, rmsY, nEntries) from the per-detector residual histograms."""
    f = uproot.open(str(path))
    hx_key = f"LGADChargeSharing/h{detector}_residualX"
    hy_key = f"LGADChargeSharing/h{detector}_residualY"
    if hx_key not in f or hy_key not in f:
        return None

    hx = f[hx_key]
    hy = f[hy_key]
    # Mean/std are exposed on uproot histogram objects as `.values()` + axis
    import numpy as np

    def _stats(h):
        values = h.values()
        edges = h.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        n = values.sum()
        if n <= 0:
            return float("nan"), 0
        mean = float((values * centers).sum() / n)
        var = float((values * (centers - mean) ** 2).sum() / n)
        return math.sqrt(max(var, 0.0)), int(n)

    rmsX, nX = _stats(hx)
    rmsY, nY = _stats(hy)
    return rmsX, rmsY, min(nX, nY)


def _robust_rms_mm(values):
    """Return the (mean, rms) of an array in mm, computed with numpy."""
    import numpy as np

    if len(values) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def validate(path: Path, detector: str, max_rms_x_um: float, max_rms_y_um: float,
             min_entries: int) -> int:
    uproot = _import_uproot()

    if not path.exists():
        print(f"  [FAIL] hist file missing: {path}")
        return 2

    header = f"[{detector}] {path.name}"
    print("=" * 72)
    print(header)
    print("=" * 72)

    tree_data = _load_residuals_from_tree(uproot, path, detector)
    failures = 0

    if tree_data is not None:
        rx, ry = tree_data
        n = min(len(rx), len(ry))
        meanX, rmsX_mm = _robust_rms_mm(rx)
        meanY, rmsY_mm = _robust_rms_mm(ry)
        rmsX_um = rmsX_mm * 1000.0
        rmsY_um = rmsY_mm * 1000.0
        source = "TTree"
    else:
        hist = _load_residual_rms_from_hist(uproot, path, detector)
        if hist is None:
            print(f"  [FAIL] Neither LGADChargeSharing/hits nor residual histograms found")
            return 1
        rmsX_mm, rmsY_mm, n = hist
        meanX = meanY = float("nan")
        rmsX_um = rmsX_mm * 1000.0
        rmsY_um = rmsY_mm * 1000.0
        source = "hists"

    print(f"  entries ({source}):       {n}")
    print(f"  residualX (um) mean/rms: {meanX*1000:+.2f} / {rmsX_um:.2f}")
    print(f"  residualY (um) mean/rms: {meanY*1000:+.2f} / {rmsY_um:.2f}")
    print(f"  thresholds (um):         <= {max_rms_x_um} (X), <= {max_rms_y_um} (Y)")
    print(f"  minimum entries:         >= {min_entries}")

    def _check(label: str, condition: bool, detail: str) -> None:
        nonlocal failures
        status = " OK " if condition else "FAIL"
        print(f"  [{status}] {label}: {detail}")
        if not condition:
            failures += 1

    _check("entries", n >= min_entries, f"n={n}")
    _check("rms(residualX)", math.isfinite(rmsX_um) and rmsX_um <= max_rms_x_um,
           f"rmsX = {rmsX_um:.2f} um")
    _check("rms(residualY)", math.isfinite(rmsY_um) and rmsY_um <= max_rms_y_um,
           f"rmsY = {rmsY_um:.2f} um")

    print()
    if failures:
        print(f"  SUMMARY: {failures} failure(s) for {detector}")
        return 1
    print(f"  SUMMARY: all checks passed for {detector}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histfile", required=True, help="Path to -Phistsfile output ROOT file")
    parser.add_argument("--detector", required=True,
                        choices=["B0Tracker", "LumiSpecTracker"],
                        help="Detector key written by LGADChargeSharingMonitor")
    parser.add_argument("--max-rms-x-um", type=float, default=150.0,
                        help="Maximum allowed RMS of residualX (microns)")
    parser.add_argument("--max-rms-y-um", type=float, default=150.0,
                        help="Maximum allowed RMS of residualY (microns)")
    parser.add_argument("--min-entries", type=int, default=50,
                        help="Minimum number of hits needed for the check")
    args = parser.parse_args()

    return validate(Path(args.histfile), args.detector, args.max_rms_x_um,
                    args.max_rms_y_um, args.min_entries)


if __name__ == "__main__":
    sys.exit(main())
