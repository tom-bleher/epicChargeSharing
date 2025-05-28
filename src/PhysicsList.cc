#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"

PhysicsList::PhysicsList()
{
    // Set default cut value
    SetDefaultCutValue(0.1*mm);
    
    // Use standard EM physics - simpler and no extra data files required
    RegisterPhysics(new G4EmStandardPhysics());
}

PhysicsList::~PhysicsList()
{
    
}