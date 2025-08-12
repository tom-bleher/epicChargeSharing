#ifndef CONTROL_HH
#define CONTROL_HH

#include "globals.hh"

namespace Control {
    
    // ========================
    // SIMULATION CONTROL PARAMETERS
    // ========================
    
    // Event generation control
    const G4int NUMBER_OF_EVENTS = 5000;         // Number of events to simulate
    const G4double PARTICLE_ENERGY = 10;        // Particle energy in GeV
    const G4String PARTICLE_TYPE = "e-";          // Particle type (e-, mu-, gamma, etc.)
    
} // namespace Control

#endif // CONTROL_HH 
