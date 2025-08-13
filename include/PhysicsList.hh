#ifndef PHYSICSLIST_HH
#define PHYSICSLIST_HH

#include "G4VModularPhysicsList.hh"

// Minimal EM physics with fine step-limiting for silicon to resolve
// pixel-pad scale effects in charge sharing.
class PhysicsList : public G4VModularPhysicsList
{
public:
    PhysicsList();
    ~PhysicsList();

};

#endif