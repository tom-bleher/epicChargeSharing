#!/bin/bash
# B0 tracker charge sharing test chain:
# 1. Generate forward protons into B0 acceptance
# 2. Simulate with ddsim through full DD4hep geometry
# 3. Reconstruct with chargeSharingRecon plugin
# 4. Quick hit count check
set -e

NEVENTS=${1:-200}
OUTDIR=${2:-/tmp/cs_tests/b0}
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"
COMPACT="${TEST_DIR}/../test_b0_lumi.xml"
STEERING="${TEST_DIR}/steering.py"
PLUGIN_DIR="${TEST_DIR}/../install"

echo "=== B0 Tracker Test ($NEVENTS events) ==="
echo "  Compact:  $COMPACT"
echo "  Output:   $OUTDIR"
echo ""

echo "=== Step 1: Generate forward protons ==="
python3 "${SCRIPT_DIR}/gen_b0_particles.py" \
    --nevents "$NEVENTS" \
    --output "$OUTDIR/b0_gen.hepmc" \
    --particle proton \
    --Emin 50.0 --Emax 100.0

echo ""
echo "=== Step 2: ddsim simulation ==="
ddsim --compactFile "$COMPACT" \
      --numberOfEvents "$NEVENTS" \
      --inputFiles "$OUTDIR/b0_gen.hepmc" \
      --outputFile "$OUTDIR/b0_sim.edm4hep.root"

echo ""
echo "=== Step 3: EICrecon with B0TRK_lgad_chargesharing ==="
export EICrecon_MY="$PLUGIN_DIR"
eicrecon -Pplugins=B0TRK_lgad_chargesharing,LGAD_chargesharing_benchmark \
         -Pjana:plugin_path="${PLUGIN_DIR}/plugins" \
         -Pnthreads=1 \
         -Pjana:nevents="$NEVENTS" \
         -Ppodio:output_file="$OUTDIR/b0_reco.edm4hep.root" \
         -Ppodio:output_collections="B0TrackerChargeSharingHits,B0TrackerChargeSharingHitAssociations,B0TrackerClusterHits" \
         -Pdd4hep:xml_files="$COMPACT" \
         -Phistsfile="$OUTDIR/b0_monitor.root" \
         "$OUTDIR/b0_sim.edm4hep.root"

echo ""
echo "=== Step 4: Quick check ==="
python3 -c "
import uproot
f = uproot.open('$OUTDIR/b0_sim.edm4hep.root')
events = f['events']
for key in events.keys():
    if 'eDep' in key and 'Contributions' not in key and 'B0' in key:
        arr = events[key].array()
        total = sum(len(e) for e in arr)
        if total > 0:
            nhit_events = sum(1 for e in arr if len(e) > 0)
            print(f'  SIM  {key}: {total} hits in {nhit_events} events')

f2 = uproot.open('$OUTDIR/b0_reco.edm4hep.root')
events2 = f2['events']
for key in events2.keys():
    if 'position.x' in key and 'B0Tracker' in key:
        arr = events2[key].array()
        total = sum(len(e) for e in arr)
        nhit_events = sum(1 for e in arr if len(e) > 0)
        print(f'  RECO {key}: {total} hits in {nhit_events} events')
"
echo "=== B0 test done ==="
