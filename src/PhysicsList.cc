#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"

PhysicsList::PhysicsList()
{
    // EM Physics 
    RegisterPhysics(new G4EmStandardPhysics());
}

PhysicsList::~PhysicsList()
{
    
}