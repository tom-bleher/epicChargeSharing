#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList()
{
    // Set appropriate cut value for fine tracking in silicon detectors (10 microns)
    // This ensures good resolution for the pixel detector simulation
    SetDefaultCutValue(10.0*micrometer);
    
    // Use standard EM physics - simpler and no extra data files required
    RegisterPhysics(new G4EmStandardPhysics());
    
    // Add step limiter physics for fine control over step sizes
    RegisterPhysics(new G4StepLimiterPhysics());
}

PhysicsList::~PhysicsList()
{
    
}