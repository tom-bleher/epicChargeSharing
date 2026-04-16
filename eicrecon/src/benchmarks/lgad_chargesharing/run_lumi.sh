#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2024-2026 Tom Bleher, Igor Korover
#
# Truth-residual benchmark for the LGAD charge-sharing reconstruction on the
# Luminosity Spectrometer tracker. Runs:
#   1. Generate BH e+e- pairs at the converter
#   2. ddsim Geant4 simulation
#   3. eicrecon with LumiSpec_lgad_chargesharing + LGAD_chargesharing_benchmark
#   4. validate.py to compute residual statistics and enforce thresholds
#
# Must be run inside eic-shell with the plugin built + installed.
# Usage: run_lumi.sh [NEVENTS] [OUTDIR]
set -e

NEVENTS=${1:-200}
OUTDIR=${2:-/tmp/lgad_chargesharing_bench/lumi}
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_DIR="${REPO_ROOT}/eicrecon/test"
COMPACT="${REPO_ROOT}/eicrecon/test_b0_lumi.xml"
PLUGIN_DIR="${REPO_ROOT}/eicrecon/install"

if [ ! -d "$PLUGIN_DIR/plugins" ]; then
    echo "ERROR: Plugin install directory not found at $PLUGIN_DIR/plugins"
    echo "       Build with: cmake --build build/eicrecon --target install"
    exit 1
fi

echo "=== LumiSpec LGAD charge-sharing benchmark ($NEVENTS events) ==="
echo "  Compact:  $COMPACT"
echo "  Output:   $OUTDIR"
echo ""

echo "=== Step 1: Generate $NEVENTS BH e+e- pairs ==="
python3 "${TEST_DIR}/lumi/gen_lumi_bh.py" \
    --nevents "$NEVENTS" \
    --output "$OUTDIR/lumi_bh.hepmc" \
    --Emin 1.0 --Emax 17.5

echo ""
echo "=== Step 2: ddsim simulation ==="
ddsim --compactFile "$COMPACT" \
      --numberOfEvents "$NEVENTS" \
      --inputFiles "$OUTDIR/lumi_bh.hepmc" \
      --outputFile "$OUTDIR/lumi_sim.edm4hep.root"

echo ""
echo "=== Step 3: EICrecon (LumiSpec plugin + benchmark monitor) ==="
export EICrecon_MY="$PLUGIN_DIR"
eicrecon \
    -Pplugins=LumiSpec_lgad_chargesharing,LGAD_chargesharing_benchmark \
    -Pjana:plugin_path="${PLUGIN_DIR}/plugins" \
    -Pnthreads=1 \
    -Pjana:nevents="$NEVENTS" \
    -Ppodio:output_file="$OUTDIR/lumi_reco.edm4hep.root" \
    -Ppodio:output_collections="LumiSpecTrackerChargeSharingHits,LumiSpecTrackerChargeSharingHitAssociations" \
    -Pdd4hep:xml_files="$COMPACT" \
    -Phistsfile="$OUTDIR/lumi_bench.root" \
    "$OUTDIR/lumi_sim.edm4hep.root"

echo ""
echo "=== Step 4: Validate residuals ==="
python3 "${SCRIPT_DIR}/validate.py" \
    --detector LumiSpecTracker \
    --histfile "$OUTDIR/lumi_bench.root" \
    --max-rms-x-um 200 \
    --max-rms-y-um 200 \
    --min-entries 50
