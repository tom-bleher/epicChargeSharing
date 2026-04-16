#!/bin/bash
# Generate neighborhood grid PDFs for each threshold sweep file
# Run from build_docker/ inside the container

set -e

SWEEP_DIR="/eic/epicChargeSharing/build/threshold_sweep"
MACRO="/eic/epicChargeSharing/analysis/viz/plotChargeNeighborhood.C"
NEVENTS=100

for ROOT_FILE in "$SWEEP_DIR"/threshold_*sigma.root; do
    BASENAME=$(basename "$ROOT_FILE" .root)
    SIGMA=$(echo "$BASENAME" | sed 's/threshold_\(.*\)sigma/\1/')
    OUT_PDF="$SWEEP_DIR/${BASENAME}_grids.pdf"

    echo "=== Generating grids for ${SIGMA}σ (${NEVENTS} events) → ${OUT_PDF} ==="
    root -l -b -q "${MACRO}+(\"${ROOT_FILE}\", ${NEVENTS}, \"coulomb\", \"${OUT_PDF}\", \"Qn\")"
    echo "=== Done: ${OUT_PDF} ==="
done

echo "All grid PDFs generated in ${SWEEP_DIR}/"
