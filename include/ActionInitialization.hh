/// \file ActionInitialization.hh
/// \brief Definition of the ECS::ActionInitialization class.
///
/// This file declares the ActionInitialization class which creates and
/// registers all user action classes for the simulation.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_ACTION_INITIALIZATION_HH
#define ECS_ACTION_INITIALIZATION_HH

#include "G4VUserActionInitialization.hh"

namespace ECS {

// Forward declarations
class PrimaryGenerator;
class RunAction;
class EventAction;
class SteppingAction;
class DetectorConstruction;

/// \brief Initializes all user action classes.
///
/// This factory class creates and registers:
/// - PrimaryGenerator for particle gun configuration
/// - RunAction for run-level management and ROOT I/O
/// - EventAction for event-level processing
/// - SteppingAction for step-level tracking
///
/// For multithreaded operation, BuildForMaster() creates a master
/// RunAction while Build() creates per-worker instances.
class ActionInitialization : public G4VUserActionInitialization
{
public:
    /// \brief Construct with detector reference.
    /// \param detector Pointer to the detector construction (must outlive this object)
    explicit ActionInitialization(DetectorConstruction* detector);
    ~ActionInitialization() override = default;

    /// \brief Build actions for the master thread.
    void BuildForMaster() const override;

    /// \brief Build actions for worker threads.
    void Build() const override;

private:
    DetectorConstruction* fDetector;
    [[nodiscard]] RunAction* CreateRunAction() const;
};

} // namespace ECS

// Backward compatibility alias
using ActionInitialization = ECS::ActionInitialization;

#endif // ECS_ACTION_INITIALIZATION_HH