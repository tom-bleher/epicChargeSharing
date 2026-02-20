/**
 * @file SteppingAction.cc
 * @brief Records first-contact volume transitions and forwards positions to `EventAction`.
 *
 * Uses cached G4LogicalVolume pointers for fast volume identification instead of
 * string comparison. This follows Geant4 performance best practices as documented
 * in the official B1 example and CERN TWiki performance tips:
 * https://twiki.cern.ch/twiki/bin/view/Geant4/Geant4PerformanceTips
 *
 * "User Stepping Action is the most important user code: it is called every single
 * step of the simulation, so if you really need it must be very small and fast.
 * Avoid any heavy operation if possible, especially combining two or more:
 * string comparisons, nested loops"
 */
#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4VTouchable.hh"

namespace ECS {

SteppingAction::SteppingAction(EventAction* eventAction)
    : fEventAction(eventAction)

{}

void SteppingAction::Reset() {
    fFirstContactType = FirstContactType::None;
    fPathLengthInSensitive = 0.0;
}

void SteppingAction::CacheVolumes() {
    // Lazy initialization pattern from Geant4 B1 example
    // Only called once per run, not per step
    if (fVolumesCached)
        return;

    auto* lvStore = G4LogicalVolumeStore::GetInstance();
    if (lvStore) {
        // GetVolume returns nullptr if not found, with optional warning
        fLogicBlock = lvStore->GetVolume("logicBlock", false);
        fLogicCube = lvStore->GetVolume("logicCube", false);
    }
    fVolumesCached = true;
}

void SteppingAction::TrackVolumeInteractions(const G4Step* step) {
    // Early exit if already recorded first-contact this event
    // Using [[likely]] attribute as most steps don't change contact
    if (fFirstContactType != FirstContactType::None) [[likely]] {
        return;
    }

    const G4StepPoint* postPoint = step->GetPostStepPoint();
    if (!postPoint || postPoint->GetStepStatus() != fGeomBoundary) {
        return; // Only care about boundary crossings
    }

    G4VPhysicalVolume const* postVol = postPoint->GetTouchableHandle()->GetVolume();
    if (!postVol) {
        return;
    }

    // Lazy initialization of cached volume pointers
    CacheVolumes();

    // FAST: Pointer comparison (~3 cycles vs ~100+ for string)
    // This is the key optimization recommended by Geant4 performance guidelines
    const G4LogicalVolume* lv = postVol->GetLogicalVolume();

    if (lv == fLogicBlock) {
        fFirstContactType = FirstContactType::Pixel;
        if (fEventAction) {
            fEventAction->RegisterFirstContact(postPoint->GetPosition(), postPoint->GetGlobalTime());
        }
    } else if (lv == fLogicCube) {
        fFirstContactType = FirstContactType::Silicon;
        if (fEventAction) {
            fEventAction->RegisterFirstContact(postPoint->GetPosition(), postPoint->GetGlobalTime());
        }
    }
}

void SteppingAction::AccumulatePathLength(const G4Step* step) {
    CacheVolumes();

    const G4StepPoint* prePoint = step->GetPreStepPoint();
    if (!prePoint)
        return;

    G4VPhysicalVolume const* preVol = prePoint->GetTouchableHandle()->GetVolume();
    if (!preVol)
        return;

    const G4LogicalVolume* lv = preVol->GetLogicalVolume();
    if (lv == fLogicCube || lv == fLogicBlock) {
        fPathLengthInSensitive += step->GetStepLength();
    }
}

void SteppingAction::UserSteppingAction(const G4Step* step) {
    TrackVolumeInteractions(step);
    AccumulatePathLength(step);
}

} // namespace ECS
