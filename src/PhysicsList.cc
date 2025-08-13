#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList()
{
    // Fine tracking in silicon (10 μm cut) for pixel-pad resolution
    SetDefaultCutValue(10.0*micrometer);
    
    // Use standard EM physics - simpler and no extra data files required
    RegisterPhysics(new G4EmStandardPhysics());
    
    // Add step limiter physics for fine control over step sizes
    RegisterPhysics(new G4StepLimiterPhysics());
}

PhysicsList::~PhysicsList()
{
    
}