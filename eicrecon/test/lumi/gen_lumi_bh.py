#!/usr/bin/env python3
"""Generate Bethe-Heitler bremsstrahlung e+e- pairs for lumi spectrometer testing.

Produces a HepMC3 ASCII file with converted BH photons (e+e- pairs) at the
converter position, ready for ddsim. Based on Dhevan Gangadharan's
lumi_particles.cxx from dhevang/Analysis_epic.

Usage (inside eic-shell):
    python3 gen_lumi_bh.py --nevents 100 --output lumi_bh.hepmc

Then feed to ddsim:
    ddsim --compactFile $COMPACT --inputFiles lumi_bh.hepmc --outputFile lumi_sim.edm4hep.root
"""

import argparse
import math
import numpy as np

# Physics constants
ELECTRON_MASS = 0.51099895e-3  # GeV
PROTON_MASS = 0.938272  # GeV
ELECTRON_PZ = -18.0  # GeV (18 GeV electron beam, going -z)
HADRON_PZ = 275.0  # GeV (275 GeV proton beam, going +z)
Z_ION = 1  # proton charge
PREFACTOR = 2.3179  # 4 * alpha * r_e^2 (mb)

# Converter position (new ePIC lumi design)
CONVERTER_Z = -41500.0  # mm — middle of magnets, where photons convert


def bh_energy_pdf(E, Ee, Ep):
    """Single-differential BH cross section dN/dE (Lifshitz QED, lab frame)."""
    if E <= 0 or E >= Ee:
        return 0.0
    ratio = (Ee - E) / (E * Ee)
    bracket1 = Ee / (Ee - E) + (Ee - E) / Ee - 2.0 / 3.0
    log_arg = 4.0 * Ep * Ee * (Ee - E) / (PROTON_MASS * ELECTRON_MASS * E)
    if log_arg <= 0:
        return 0.0
    bracket2 = math.log(log_arg) - 0.5
    return Z_ION**2 * PREFACTOR * ratio * bracket1 * bracket2


def bh_theta_pdf(theta, gamma_e):
    """BH photon angular distribution dN/dtheta (lab frame)."""
    inv_gamma = 1.0 / gamma_e
    return theta / (inv_gamma**2 + theta**2) ** 2


def sample_bh_energy(Emin, Emax, Ee, Ep, rng):
    """Sample BH photon energy using rejection sampling."""
    # Find approximate max of PDF for rejection
    E_test = np.linspace(Emin, Emax * 0.99, 1000)
    pdf_vals = np.array([bh_energy_pdf(e, Ee, Ep) for e in E_test])
    pdf_max = pdf_vals.max() * 1.1

    while True:
        E = rng.uniform(Emin, Emax)
        if rng.random() < bh_energy_pdf(E, Ee, Ep) / pdf_max:
            return E


def sample_bh_theta(gamma_e, rng):
    """Sample BH photon angle using rejection sampling."""
    theta_max = 10.0 / gamma_e  # well beyond the peak
    inv_gamma = 1.0 / gamma_e
    # Peak is at theta = inv_gamma / sqrt(3)
    pdf_max = bh_theta_pdf(inv_gamma / math.sqrt(3), gamma_e) * 1.1

    while True:
        theta = rng.uniform(0, theta_max)
        if rng.random() < bh_theta_pdf(theta, gamma_e) / pdf_max:
            return theta


def pdg_splitting(rng):
    """Sample electron energy fraction from PDG pair splitting function."""
    # P(x) = 1 - 4/3 * x*(1-x), peak at x=0 and x=1, min at x=0.5
    while True:
        x = rng.random()
        if rng.random() < 1.0 - 4.0 / 3.0 * x * (1.0 - x):
            return x


def write_hepmc3_event(f, event_num, particles):
    """Write one HepMC3 ASCII event."""
    # Header
    f.write(f"E {event_num} 0 {len(particles)}\n")
    f.write("U GEV MM\n")
    # Particles: status 1=final, 4=beam
    for i, (px, py, pz, e, pdg, status) in enumerate(particles):
        f.write(f"P {i+1} 0 {pdg} {px:.10e} {py:.10e} {pz:.10e} {e:.10e} 0 {status}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate BH e+e- pairs for lumi testing")
    parser.add_argument("--nevents", type=int, default=100)
    parser.add_argument("--output", default="lumi_bh.hepmc")
    parser.add_argument("--Emin", type=float, default=1.0, help="Min photon energy (GeV)")
    parser.add_argument("--Emax", type=float, default=17.5, help="Max photon energy (GeV)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat", action="store_true", help="Use flat energy spectrum")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    Ee = abs(ELECTRON_PZ)
    Ep = abs(HADRON_PZ)
    gamma_e = Ee / ELECTRON_MASS

    with open(args.output, "w") as f:
        f.write("HepMC::Version 3.02.06\n")
        f.write("HepMC::Asciiv3-START_EVENT_LISTING\n")

        n_written = 0
        for iev in range(args.nevents):
            # Sample photon kinematics
            if args.flat:
                E_gamma = rng.uniform(args.Emin, args.Emax)
                theta_gamma = math.pi  # straight backward
            else:
                E_gamma = sample_bh_energy(args.Emin, args.Emax, Ee, Ep, rng)
                theta_raw = sample_bh_theta(gamma_e, rng)
                theta_gamma = math.pi - theta_raw  # photons go -z

            phi_gamma = rng.uniform(0, 2 * math.pi)

            # Convert to e+e- pair (PDG splitting)
            x = pdg_splitting(rng)
            E_electron = E_gamma * x
            E_positron = E_gamma * (1.0 - x)

            if E_electron < 0.001 or E_positron < 0.001:
                continue

            p_electron = math.sqrt(max(0, E_electron**2 - ELECTRON_MASS**2))
            p_positron = math.sqrt(max(0, E_positron**2 - ELECTRON_MASS**2))

            # Both e+ and e- share the photon direction (collinear approximation)
            sx = math.sin(theta_gamma) * math.cos(phi_gamma)
            sy = math.sin(theta_gamma) * math.sin(phi_gamma)
            sz = math.cos(theta_gamma)

            # Propagate photon from IP to converter to get vertex position
            # photon direction: (sx, sy, sz) with sz < 0
            if sz != 0:
                t_prop = CONVERTER_Z / sz  # mm
                vx = sx * t_prop
                vy = sy * t_prop
                vz = CONVERTER_Z
            else:
                vx, vy, vz = 0, 0, CONVERTER_Z

            particles = [
                # Beam electron
                (0.0, 0.0, ELECTRON_PZ, math.sqrt(ELECTRON_PZ**2 + ELECTRON_MASS**2), 11, 4),
                # Beam proton
                (0.0, 0.0, HADRON_PZ, math.sqrt(HADRON_PZ**2 + PROTON_MASS**2), 2212, 4),
                # Final state electron (from conversion)
                (p_electron * sx, p_electron * sy, p_electron * sz, E_electron, 11, 1),
                # Final state positron (from conversion)
                (p_positron * sx, p_positron * sy, p_positron * sz, E_positron, -11, 1),
            ]

            # Write HepMC3 event with vertex at converter position
            f.write(f"E {n_written} 0 4\n")
            f.write("U GEV MM\n")
            f.write(f"A 0 GenCrossSection -1 -1 0 0\n")
            # Vertex at converter
            f.write(f"V -1 0 [{vx:.6e},{vy:.6e},{vz:.6e},0]\n")
            # Beam particles (incoming to vertex)
            f.write(f"P 1 -1 11 0 0 {ELECTRON_PZ:.10e} {math.sqrt(ELECTRON_PZ**2 + ELECTRON_MASS**2):.10e} {ELECTRON_MASS:.10e} 4\n")
            f.write(f"P 2 -1 2212 0 0 {HADRON_PZ:.10e} {math.sqrt(HADRON_PZ**2 + PROTON_MASS**2):.10e} {PROTON_MASS:.10e} 4\n")
            # Final state e+e-
            f.write(f"P 3 -1 11 {p_electron*sx:.10e} {p_electron*sy:.10e} {p_electron*sz:.10e} {E_electron:.10e} {ELECTRON_MASS:.10e} 1\n")
            f.write(f"P 4 -1 -11 {p_positron*sx:.10e} {p_positron*sy:.10e} {p_positron*sz:.10e} {E_positron:.10e} {ELECTRON_MASS:.10e} 1\n")

            n_written += 1

        f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")

    print(f"Generated {n_written} BH e+e- pair events -> {args.output}")
    print(f"  Energy range: [{args.Emin}, {args.Emax}] GeV")
    print(f"  Converter at z = {CONVERTER_Z} mm")


if __name__ == "__main__":
    main()
