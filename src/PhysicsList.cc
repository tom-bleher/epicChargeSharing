#include "PhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList()
{
    SetDefaultCutValue(10.0*CLHEP::micrometer);
    
    RegisterPhysics(new G4EmStandardPhysics());
    
    RegisterPhysics(new G4StepLimiterPhysics());
}

PhysicsList::~PhysicsList()
{
    
}