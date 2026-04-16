#!/bin/bash
# Lumi spectrometer charge sharing test chain:
# 1. Generate BH e+e- pairs at converter
# 2. Simulate with ddsim through full DD4hep geometry
# 3. Reconstruct with chargeSharingRecon plugin
# 4. Quick hit count check
set -e

NEVENTS=${1:-200}
OUTDIR=${2:-/tmp/cs_tests/lumi}
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"
COMPACT="${TEST_DIR}/../test_b0_lumi.xml"
STEERING="${TEST_DIR}/steering.py"
PLUGIN_DIR="${TEST_DIR}/../install"

echo "=== Lumi Spectrometer Test ($NEVENTS events) ==="
echo "  Compact:  $COMPACT"
echo "  Output:   $OUTDIR"
echo ""

echo "=== Step 1: Generate $NEVENTS BH e+e- pairs ==="
python3 "${SCRIPT_DIR}/gen_lumi_bh.py" \
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
echo "=== Step 3: EICrecon with chargeSharingRecon ==="
export EICrecon_MY="$PLUGIN_DIR"
eicrecon -Pplugins=chargeSharingRecon \
         -Pnthreads=1 \
         -Pjana:nevents="$NEVENTS" \
         -Ppodio:output_file="$OUTDIR/lumi_reco.edm4hep.root" \
         -Ppodio:output_collections="LumiSpecTrackerChargeSharingHits,LumiSpecTrackerChargeSharingHitAssociations" \
         -Pdd4hep:xml_files="$COMPACT" \
         -Phistsfile="$OUTDIR/lumi_monitor.root" \
         "$OUTDIR/lumi_sim.edm4hep.root"

echo ""
echo "=== Step 4: Quick check ==="
python3 -c "
import uproot
f = uproot.open('$OUTDIR/lumi_sim.edm4hep.root')
events = f['events']
for key in events.keys():
    if 'eDep' in key and 'Contributions' not in key and 'Lumi' in key:
        arr = events[key].array()
        total = sum(len(e) for e in arr)
        if total > 0:
            nhit_events = sum(1 for e in arr if len(e) > 0)
            print(f'  SIM  {key}: {total} hits in {nhit_events} events')

f2 = uproot.open('$OUTDIR/lumi_reco.edm4hep.root')
events2 = f2['events']
for key in events2.keys():
    if 'position.x' in key and 'LumiSpec' in key:
        arr = events2[key].array()
        total = sum(len(e) for e in arr)
        nhit_events = sum(1 for e in arr if len(e) > 0)
        print(f'  RECO {key}: {total} hits in {nhit_events} events')
"
echo "=== Lumi test done ==="
