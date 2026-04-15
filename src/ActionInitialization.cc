// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

/**
 * @file ActionInitialization.cc
 * @brief Wires together user actions: primary generator, run/event/stepping actions.
 */
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "PrimaryGenerator.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"
#include "RuntimeConfig.hh"
#include <limits>

ActionInitialization::ActionInitialization(DetectorConstruction* detector) : fDetector(detector) {}

void ActionInitialization::BuildForMaster() const {
    SetUserAction(CreateRunAction());
}

void ActionInitialization::Build() const {
    SetUserAction(new PrimaryGenerator(fDetector));

    RunAction* runAction = CreateRunAction();
    SetUserAction(runAction);

    auto* eventAction = new EventAction(runAction, fDetector);
    SetUserAction(eventAction);

    if (fDetector) {
        eventAction->SetNeighborhoodRadius(fDetector->GetNeighborhoodRadius());
    }

    auto* steppingAction = new SteppingAction(eventAction);
    SetUserAction(steppingAction);

    eventAction->SetSteppingAction(steppingAction);
}

RunAction* ActionInitialization::CreateRunAction() const {
    auto* runAction = new RunAction();
    if (!fDetector) {
        return runAction;
    }

    runAction->SetDetectorGridParameters(fDetector->GetPixelSize(), fDetector->GetPixelSpacing(),
                                         fDetector->GetGridOffset(), fDetector->GetDetSize(),
                                         fDetector->GetNumBlocksPerSide());
    runAction->SetNeighborhoodRadiusMeta(fDetector->GetNeighborhoodRadius());
    runAction->SetGridPixelCenters(fDetector->GetPixelCenters());

    // Recon metadata (previously pushed via DetectorConstruction::SyncRunMetadata)
    const auto& rtConfig = ECS::RuntimeConfig::Instance();
    const auto reconMethod = (rtConfig.activeMode == 1) ? Constants::ReconMethod::LinA
                                                        : Constants::ReconMethod::LogA;
    G4double linearBeta = std::numeric_limits<G4double>::quiet_NaN();
    if (rtConfig.activeMode == 1) {
        linearBeta = fDetector->GetLinearChargeModelBeta();
    }
    runAction->SetPosReconMetadata(reconMethod, linearBeta, fDetector->GetPixelSpacing());

    return runAction;
}