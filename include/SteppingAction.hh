// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/// \file SteppingAction.hh
/// \brief Definition of the ECS::SteppingAction class.
///
/// This file declares the SteppingAction class which tracks particle
/// steps and identifies first-contact volumes.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_STEPPING_ACTION_HH
#define ECS_STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

#include <vector>

class G4Step;
class G4LogicalVolume;

namespace ECS {

// Forward declarations
class EventAction;

/// \brief First contact volume type.
///
/// Identifies which volume type the particle first contacted.
enum class FirstContactType {
    None,   ///< No contact recorded yet
    Pixel,  ///< First contact was pixel aluminum (logicBlock)
    Silicon ///< First contact was silicon bulk (logicCube)
};

/// \brief Step-level tracking action.
///
/// This lightweight action class tracks particle steps to:
/// - Identify the first-contact volume (pixel aluminum vs silicon)
/// - Register first contact position with EventAction
///
/// The first-contact classification is used to determine whether
/// a hit should be classified as a direct pixel hit or a silicon hit.
///
/// \note Uses cached G4LogicalVolume pointers for fast volume identification
/// instead of string comparison. This follows Geant4 performance best practices
/// as documented in the official B1 example and CERN TWiki performance tips.
class SteppingAction : public G4UserSteppingAction {
public:
    /// \brief Construct with event action reference.
    /// \param eventAction Pointer to event action for registering contacts
    explicit SteppingAction(EventAction* eventAction);
    ~SteppingAction() override = default;

    /// \brief Process each simulation step.
    /// \param step The current step to process
    void UserSteppingAction(const G4Step* step) override;

    /// \brief Reset per-event state.
    void Reset();

    /// \brief Track volume boundary crossings.
    /// \param step The step containing volume information
    void TrackVolumeInteractions(const G4Step* step);

    /// \brief Per-step energy deposit record for Landau fluctuation studies.
    struct StepDeposit {
        G4ThreeVector position;  ///< Pre-step point position in sensitive volume
        G4double edep{0.0};      ///< Energy deposited in this step (Geant4 units)
        G4double time{0.0};      ///< Global time at pre-step point (ns)
    };

    /// \brief Get per-step energy deposits in the sensitive volume.
    [[nodiscard]] const std::vector<StepDeposit>& GetStepDeposits() const { return fStepDeposits; }

    /// \brief Get total energy deposited in the sensitive volume (replaces G4PSEnergyDeposit scorer).
    [[nodiscard]] G4double GetTotalEdep() const { return fTotalEdep; }

    /// \brief Check if first contact was with a pixel.
    /// \return true if first boundary entry was into pixel aluminum
    [[nodiscard]] G4bool FirstContactIsPixel() const { return fFirstContactType == FirstContactType::Pixel; }

    /// \brief Get accumulated path length through sensitive volume.
    [[nodiscard]] G4double GetPathLength() const { return fPathLengthInSensitive; }

private:
    /// \brief Cache volume pointers for fast lookup.
    ///
    /// Follows Geant4 B1 example pattern: lazy initialization of cached
    /// G4LogicalVolume pointers from G4LogicalVolumeStore. This avoids
    /// expensive string comparisons in the hot stepping action path.
    void CacheVolumes();

    /// \brief Accumulate step length when inside sensitive volume.
    void AccumulatePathLength(const G4Step* step);

    /// \brief Accumulate energy deposit and record per-step data in sensitive volume.
    void AccumulateEdep(const G4Step* step);

    EventAction* fEventAction;
    FirstContactType fFirstContactType = FirstContactType::None;
    G4double fPathLengthInSensitive{0.0}; ///< Accumulated path in sensitive volume
    G4double fTotalEdep{0.0};             ///< Total energy deposit in sensitive volume
    std::vector<StepDeposit> fStepDeposits; ///< Per-step deposits for Landau studies

    // Cached volume pointers for fast comparison (initialized lazily)
    const G4LogicalVolume* fLogicBlock = nullptr; ///< Pixel aluminum volume
    const G4LogicalVolume* fLogicCube = nullptr;  ///< Silicon bulk volume
    G4bool fVolumesCached = false;                ///< Whether volumes have been cached
};

} // namespace ECS

// Backward compatibility alias
using SteppingAction = ECS::SteppingAction;

#endif // ECS_STEPPING_ACTION_HH