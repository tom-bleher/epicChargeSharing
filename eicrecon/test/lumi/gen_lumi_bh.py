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
import pyhepmc

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

    with pyhepmc.open(args.output, "w") as f:
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
            if sz != 0:
                t_prop = CONVERTER_Z / sz  # mm
                vx = sx * t_prop
                vy = sy * t_prop
                vz = CONVERTER_Z
            else:
                vx, vy, vz = 0, 0, CONVERTER_Z

            evt = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
            evt.event_number = n_written

            # Vertex at converter position
            v = pyhepmc.GenVertex(pyhepmc.FourVector(vx, vy, vz, 0))
            evt.add_vertex(v)

            # Beam particles (incoming)
            beam_e = pyhepmc.GenParticle(
                pyhepmc.FourVector(0, 0, ELECTRON_PZ, math.sqrt(ELECTRON_PZ**2 + ELECTRON_MASS**2)),
                11, 4)
            beam_e.generated_mass = ELECTRON_MASS
            v.add_particle_in(beam_e)

            beam_p = pyhepmc.GenParticle(
                pyhepmc.FourVector(0, 0, HADRON_PZ, math.sqrt(HADRON_PZ**2 + PROTON_MASS**2)),
                2212, 4)
            beam_p.generated_mass = PROTON_MASS
            v.add_particle_in(beam_p)

            # Final state e+e-
            elec = pyhepmc.GenParticle(
                pyhepmc.FourVector(p_electron * sx, p_electron * sy, p_electron * sz, E_electron),
                11, 1)
            elec.generated_mass = ELECTRON_MASS
            v.add_particle_out(elec)

            posi = pyhepmc.GenParticle(
                pyhepmc.FourVector(p_positron * sx, p_positron * sy, p_positron * sz, E_positron),
                -11, 1)
            posi.generated_mass = ELECTRON_MASS
            v.add_particle_out(posi)

            f.write(evt)
            n_written += 1

    print(f"Generated {n_written} BH e+e- pair events -> {args.output}")
    print(f"  Energy range: [{args.Emin}, {args.Emax}] GeV")
    print(f"  Converter at z = {CONVERTER_Z} mm")


if __name__ == "__main__":
    main()
