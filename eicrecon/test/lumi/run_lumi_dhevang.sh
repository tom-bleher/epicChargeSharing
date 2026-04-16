#!/bin/bash
# Lumi spectrometer test using dhevang's ROOT macro generators.
#
# This is the "full physics" pipeline from Analysis_epic:
#   1. lumi_particles.cxx  — generate BH photons (ROOT/HepMC3 macro)
#   2. PropagateAndConvert.cxx — propagate + convert to e+e- pairs
#   3. ddsim — Geant4 detector simulation
#   4. eicrecon — charge sharing reconstruction
#
# Requires ROOT with HepMC3 support (available in eic-shell nightly).
# abconv (beam effects) is optional — skipped if not found.
set -e

NEVENTS=${1:-200}
OUTDIR=${2:-/tmp/cs_tests/lumi_dhevang}
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"
COMPACT="${TEST_DIR}/../test_b0_lumi.xml"
STEERING="${TEST_DIR}/steering.py"
PLUGIN_DIR="${TEST_DIR}/../install"

echo "=== Lumi Dhevang Pipeline ($NEVENTS events) ==="

# Check ROOT + HepMC3 availability
if ! root -l -b -q -e 'gSystem->Load("libHepMC3"); std::cout << "HepMC3 OK" << std::endl;' 2>/dev/null | grep -q "HepMC3 OK"; then
    echo "ERROR: ROOT with HepMC3 support not available."
    echo "  This test requires eic-shell with ROOT compiled against HepMC3."
    echo "  Use run_lumi_test.sh instead (Python-only generators)."
    exit 1
fi

echo ""
echo "=== Step 1: Generate BH photons (lumi_particles.cxx) ==="
# Generate unconverted BH photons at IP
# Args: (n_events, flat=false, convert=false, displaceVertices=false, Egamma_start, Egamma_end, output)
root -l -b -q "${SCRIPT_DIR}/lumi_particles.cxx(${NEVENTS}, false, false, false, 1.0, 17.5, \"${OUTDIR}/bh_photons.hepmc\")"

echo ""
echo "=== Step 2: Check for abconv (beam effects) ==="
HEPMC_INPUT="${OUTDIR}/bh_photons.hepmc"
if command -v abconv &>/dev/null; then
    echo "  abconv found — applying beam effects"
    abconv "$HEPMC_INPUT" --plot-off -o "${OUTDIR}/bh_beam"
    HEPMC_INPUT="${OUTDIR}/bh_beam.hepmc"
else
    echo "  abconv not found — skipping beam effects (OK for charge sharing tests)"
fi

echo ""
echo "=== Step 3: Propagate + convert to e+e- (PropagateAndConvert.cxx) ==="
root -l -b -q "${SCRIPT_DIR}/PropagateAndConvert.cxx(\"${HEPMC_INPUT}\", \"${OUTDIR}/bh_electrons.hepmc\")"

echo ""
echo "=== Step 4: ddsim simulation ==="
ddsim --compactFile "$COMPACT" \
      --numberOfEvents "$NEVENTS" \
      --inputFiles "$OUTDIR/bh_electrons.hepmc" \
      --outputFile "$OUTDIR/lumi_sim.edm4hep.root"

echo ""
echo "=== Step 5: EICrecon with LumiSpec_lgad_chargesharing ==="
export EICrecon_MY="$PLUGIN_DIR"
eicrecon -Pplugins=LumiSpec_lgad_chargesharing,LGAD_chargesharing_benchmark \
         -Pjana:plugin_path="${PLUGIN_DIR}/plugins" \
         -Pnthreads=1 \
         -Pjana:nevents="$NEVENTS" \
         -Ppodio:output_file="$OUTDIR/lumi_reco.edm4hep.root" \
         -Ppodio:output_collections="LumiSpecTrackerChargeSharingHits,LumiSpecTrackerChargeSharingHitAssociations" \
         -Pdd4hep:xml_files="$COMPACT" \
         -Phistsfile="$OUTDIR/lumi_monitor.root" \
         "$OUTDIR/lumi_sim.edm4hep.root"

echo ""
echo "=== Step 6: Quick check ==="
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
echo "=== Lumi dhevang pipeline done ==="
