#!/bin/bash
# Sweep readout threshold from 2.0σ to 4.0σ, 10k events each
# Run from the build directory

set -e

NEVENTS=1000
OUTDIR="threshold_sweep"
SEED=20260415  # Fixed seed — same hits & noise across all threshold values
mkdir -p "$OUTDIR"

for SIGMA in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
    echo "=== Running threshold ${SIGMA}σ, ${NEVENTS} events, seed=${SEED} ==="

    # Generate macro on the fly
    cat > /tmp/threshold_run.mac <<EOF
/control/verbose 0
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/run/numberOfThreads 1
/run/initialize
/ecs/gun/useFixedPosition false
/ecs/gun/beamOvershoot 0.1
/ecs/fit/gaus1D false
/ecs/fit/gaus2D true
/gun/particle e-
/gun/energy 10 GeV
/ecs/noise/thresholdSigma ${SIGMA}
/random/setSeeds ${SEED} 0
/run/beamOn ${NEVENTS}
EOF

    ./epicChargeSharing -m /tmp/threshold_run.mac
    mv epicChargeSharing.root "$OUTDIR/threshold_${SIGMA}sigma.root"
    echo "=== Done: ${OUTDIR}/threshold_${SIGMA}sigma.root ==="
done

echo "All runs complete. Output in ${OUTDIR}/"
echo ""
echo "=== Running resolution analysis ==="
/tmp/eic_viz/bin/python3 ../analysis/diagnostics/analyze_threshold_sweep.py "$OUTDIR"
