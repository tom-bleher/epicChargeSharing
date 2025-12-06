/// \file PhysicsList.hh
/// \brief Definition of the ECS::PhysicsList class.
///
/// This file declares the PhysicsList class which configures the
/// physics processes for the AC-LGAD detector simulation.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_PHYSICS_LIST_HH
#define ECS_PHYSICS_LIST_HH

#include "G4VModularPhysicsList.hh"

namespace ECS {

/// \brief Physics list for AC-LGAD detector simulation.
///
/// Configures a minimal electromagnetic physics list with:
/// - Standard EM physics (G4EmStandardPhysics)
/// - Step limiter physics for controlled tracking
///
/// This physics list is optimized for silicon detector response
/// studies with high-energy charged particles.
class PhysicsList : public G4VModularPhysicsList
{
public:
    /// \brief Construct and register physics modules.
    PhysicsList();
    ~PhysicsList() override = default;
};

} // namespace ECS

// Backward compatibility alias
using PhysicsList = ECS::PhysicsList;

#endif // ECS_PHYSICS_LIST_HH