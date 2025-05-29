#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList()
{
    // Set very small cut value for fine tracking (1 micron)
    SetDefaultCutValue(1.0*micrometer);
    
    // Use standard EM physics - simpler and no extra data files required
    RegisterPhysics(new G4EmStandardPhysics());
    
    // Add step limiter physics for fine control over step sizes
    RegisterPhysics(new G4StepLimiterPhysics());
}

PhysicsList::~PhysicsList()
{
    
}