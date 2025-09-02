#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList()
{
    // Fine tracking in silicon (10 μm cut)
    SetDefaultCutValue(10.0*CLHEP::micrometer);
    
    // Standard EM physics
    RegisterPhysics(new G4EmStandardPhysics());
    
    // Step limiter physics
    RegisterPhysics(new G4StepLimiterPhysics());
}

PhysicsList::~PhysicsList()
{
    
}