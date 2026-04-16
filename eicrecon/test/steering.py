"""ddsim steering configuration for charge sharing tests.

Adapted from Dhevan Gangadharan's steeringGun.py (dhevang/Analysis_epic).
Configures physics list, magnetic field stepper, and sensitive detector actions
for tracker and calorimeter subsystems. No particle gun — input comes from
HepMC3 files.

Usage:
    ddsim --steeringFile steering.py --compactFile ... --inputFiles ... --outputFile ...
"""

from DDSim.DD4hepSimulation import DD4hepSimulation

SIM = DD4hepSimulation()

# No gun — we use HepMC3 input files exclusively
SIM.enableGun = False
SIM.enableG4Gun = False
SIM.enableG4GPS = False

# Physics list
SIM.physics.list = "FTFP_BERT"
SIM.physics.rangecut = 0.7  # mm (Geant4 default)

# Magnetic field stepper (ClassicalRK4 with tight tolerances for beamline)
SIM.field.stepper = "ClassicalRK4"
SIM.field.delta_chord = 0.25
SIM.field.delta_intersection = 0.001
SIM.field.delta_one_step = 0.01
SIM.field.eps_max = 0.001
SIM.field.eps_min = 5e-05
SIM.field.largest_step = 10000.0
SIM.field.min_chord_step = 0.01

# Tracker: weighted hit position combination
SIM.action.tracker = (
    "Geant4TrackerWeightedAction",
    {"HitPositionCombination": 2, "CollectSingleDeposits": False},
)

# Calorimeter: scintillator action
SIM.action.calo = "Geant4ScintillatorCalorimeterAction"

# Filters: 1 keV threshold for trackers, 0 for calorimeters
SIM.filter.tracker = "edep1kev"
SIM.filter.calo = "edep0"

# HepMC3 input
SIM.hepmc3.useHepMC3 = True

# Batch mode, moderate verbosity
SIM.runType = "batch"
SIM.printLevel = 3
