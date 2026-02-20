/**
 * @file PhysicsList.cc
 * @brief Minimal EM physics with step limiter and default production cut.
 */
#include "PhysicsList.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList() {
    SetDefaultCutValue(50.0 * CLHEP::micrometer);

    RegisterPhysics(new G4EmStandardPhysics_option1());

    RegisterPhysics(new G4StepLimiterPhysics());
}