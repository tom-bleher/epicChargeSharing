#!/usr/bin/env python3
"""Validate charge sharing reconstruction output.

Runs tiered checks on simulation and reconstruction ROOT files:
  1. Smoke: output files exist, events tree present
  2. Sim hits: detector hit collections have > 0 entries
  3. Reco hits: charge sharing output collections have > 0 entries
  4. Efficiency: reco hit count >= sim hit count
  5. Associations: association count matches reco hit count

Usage:
    python3 validate.py --b0-dir /tmp/cs_tests/b0 --lumi-dir /tmp/cs_tests/lumi
    python3 validate.py --b0-dir /tmp/cs_tests/b0   # B0 only
    python3 validate.py --lumi-dir /tmp/cs_tests/lumi  # Lumi only

Exit code 0 = all pass, 1 = failures.
"""

import argparse
import os
import sys


def count_hits(events, pattern):
    """Count total hits in branches matching a pattern."""
    total = 0
    for key in events.keys():
        if pattern in key and "eDep" in key and "Contributions" not in key:
            arr = events[key].array()
            total += sum(len(e) for e in arr)
    return total


def count_reco_hits(events, collection_prefix):
    """Count reco hits by looking for position.x in a collection."""
    key = f"{collection_prefix}.position.x"
    if key not in events.keys():
        return 0
    arr = events[key].array()
    return sum(len(e) for e in arr)


def count_associations(events, collection_prefix):
    """Count associations by looking for weight field."""
    key = f"{collection_prefix}.weight"
    if key not in events.keys():
        # Try simHit index as fallback
        key = f"{collection_prefix}.simHit.index"
        if key not in events.keys():
            return -1  # not found
    arr = events[key].array()
    return sum(len(e) for e in arr)


class TestRunner:
    def __init__(self):
        self.results = []
        self.n_pass = 0
        self.n_fail = 0

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.n_pass += 1
        else:
            self.n_fail += 1
        self.results.append((name, status, detail))

    def print_results(self):
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        max_name = max(len(r[0]) for r in self.results) if self.results else 20
        for name, status, detail in self.results:
            marker = " OK " if status == "PASS" else "FAIL"
            line = f"  [{marker}] {name:<{max_name}}"
            if detail:
                line += f"  ({detail})"
            print(line)
        print("-" * 60)
        print(f"  {self.n_pass} passed, {self.n_fail} failed")
        print("=" * 60)

    @property
    def all_passed(self):
        return self.n_fail == 0


def validate_detector(runner, outdir, detector_name, sim_file, reco_file,
                      sim_pattern, reco_collection, assoc_collection):
    """Run all validation tiers for one detector."""
    import uproot

    prefix = f"{detector_name}"

    # Tier 1: Smoke — files exist
    sim_path = os.path.join(outdir, sim_file)
    reco_path = os.path.join(outdir, reco_file)
    runner.check(f"{prefix}: sim file exists", os.path.exists(sim_path), sim_path)
    runner.check(f"{prefix}: reco file exists", os.path.exists(reco_path), reco_path)

    if not os.path.exists(sim_path) or not os.path.exists(reco_path):
        runner.check(f"{prefix}: skipping remaining checks", False, "missing files")
        return

    # Open files
    sim_f = uproot.open(sim_path)
    reco_f = uproot.open(reco_path)

    runner.check(f"{prefix}: sim has events tree", "events" in sim_f)
    runner.check(f"{prefix}: reco has events tree", "events" in reco_f)

    if "events" not in sim_f or "events" not in reco_f:
        return

    sim_events = sim_f["events"]
    reco_events = reco_f["events"]

    # Tier 2: Sim hits
    n_sim = count_hits(sim_events, sim_pattern)
    runner.check(f"{prefix}: sim hits > 0", n_sim > 0, f"n_sim={n_sim}")

    # Tier 3: Reco hits
    n_reco = count_reco_hits(reco_events, reco_collection)
    runner.check(f"{prefix}: reco hits > 0", n_reco > 0, f"n_reco={n_reco}")

    # Tier 4: Efficiency (reco >= sim for 1:1 mapping)
    if n_sim > 0 and n_reco > 0:
        runner.check(f"{prefix}: reco >= sim hits",
                     n_reco >= n_sim,
                     f"n_reco={n_reco}, n_sim={n_sim}")

    # Tier 5: Associations
    n_assoc = count_associations(reco_events, assoc_collection)
    if n_assoc >= 0:
        runner.check(f"{prefix}: associations match reco",
                     n_assoc == n_reco,
                     f"n_assoc={n_assoc}, n_reco={n_reco}")
    else:
        runner.check(f"{prefix}: association collection found", False,
                     f"{assoc_collection} not in output")


def main():
    parser = argparse.ArgumentParser(description="Validate charge sharing reconstruction output")
    parser.add_argument("--b0-dir", help="Directory with B0 test outputs")
    parser.add_argument("--lumi-dir", help="Directory with Lumi test outputs")
    args = parser.parse_args()

    if not args.b0_dir and not args.lumi_dir:
        parser.error("At least one of --b0-dir or --lumi-dir is required")

    try:
        import uproot  # noqa: F401
    except ImportError:
        print("ERROR: uproot is required. Install with: pip install uproot awkward")
        sys.exit(1)

    runner = TestRunner()

    if args.b0_dir:
        validate_detector(
            runner, args.b0_dir,
            detector_name="B0",
            sim_file="b0_sim.edm4hep.root",
            reco_file="b0_reco.edm4hep.root",
            sim_pattern="B0Tracker",
            reco_collection="B0TrackerChargeSharingHits",
            assoc_collection="B0TrackerChargeSharingHitAssociations",
        )

    if args.lumi_dir:
        validate_detector(
            runner, args.lumi_dir,
            detector_name="Lumi",
            sim_file="lumi_sim.edm4hep.root",
            reco_file="lumi_reco.edm4hep.root",
            sim_pattern="LumiSpec",
            reco_collection="LumiSpecTrackerChargeSharingHits",
            assoc_collection="LumiSpecTrackerChargeSharingHitAssociations",
        )

    runner.print_results()
    sys.exit(0 if runner.all_passed else 1)


if __name__ == "__main__":
    main()
