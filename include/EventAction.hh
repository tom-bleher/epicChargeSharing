/// \file EventAction.hh
/// \brief Definition of the ECS::EventAction class.
///
/// This file declares the EventAction class which handles per-event
/// processing including hit classification and charge sharing computation.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_EVENT_ACTION_HH
#define ECS_EVENT_ACTION_HH

#include "ChargeSharingCalculator.hh"
#include "G4ThreeVector.hh"
#include "G4UserEventAction.hh"
#include "globals.hh"
#include "NeighborhoodUtils.hh"
#include "RootIO.hh"

#include <memory>
#include <vector>

class G4Event;
class G4GenericMessenger;

namespace ECS {

// Forward declarations
class RunAction;
class DetectorConstruction;
class SteppingAction;

/// \brief Event-level processing action.
///
/// Handles all per-event bookkeeping and data extraction:
/// - Collects energy deposit from sensitive detector scorer
/// - Determines hit position from first contact or geometric center
/// - Classifies hits as pixel or silicon based on contact volume
/// - Computes charge sharing fractions across pixel neighborhood
/// - Reconstructs position using configured model (LogA/LinA)
/// - Populates ROOT tree branches via RunAction
///
/// The charge sharing computation uses the ChargeSharingCalculator
/// to compute fractions for each pixel in a (2r+1)x(2r+1) neighborhood.
class EventAction : public G4UserEventAction {
public:
    EventAction(RunAction* runAction, DetectorConstruction* detector);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

    void SetSteppingAction(SteppingAction* steppingAction) { fSteppingAction = steppingAction; }

    void RegisterFirstContact(const G4ThreeVector& pos, G4double time) {
        fFirstContactPos = pos;
        fFirstContactTime = time;
        fHasFirstContactPos = true;
    }

    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);

    void SetNeighborhoodRadius(G4int radius) {
        fNeighborhoodRadius = radius;
        fChargeSharing.SetNeighborhoodRadius(radius);
        fNeighborhoodLayout.SetRadius(radius);
        EnsureNeighborhoodBuffers(fNeighborhoodLayout.TotalCells());
    }
    [[nodiscard]] G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }

    void CollectScorerData(const G4Event* event);

    void SetEmitDistanceAlpha(G4bool enabled);
    void SetComputeFullFractions(G4bool enabled);

private:
    [[nodiscard]] const G4ThreeVector& DetermineHitPosition() const;
    void UpdatePixelAndHitClassification(const G4ThreeVector& hitPos, G4ThreeVector& nearestPixel,
                                         G4bool& firstContactIsPixel, G4bool& geometricIsPixel,
                                         G4bool& isPixelHitCombined);
    const ChargeSharingCalculator::Result& ComputeChargeSharingForEvent(const G4ThreeVector& hitPos,
                                                                        G4double energyDeposit);
    void EnsureNeighborhoodBuffers(std::size_t totalCells);
    void UpdatePixelIndices(const ChargeSharingCalculator::Result& result, const G4ThreeVector& hitPos);
    void ReconstructPosition(const ChargeSharingCalculator::Result& result, const G4ThreeVector& hitPos);

    /// \brief Build EventSummaryData from current event state.
    [[nodiscard]] IO::EventSummaryData BuildEventSummary(G4double edep, const G4ThreeVector& hitPos,
                                                         const G4ThreeVector& nearestPixel, G4bool firstContactIsPixel,
                                                         G4bool geometricIsPixel, G4bool isPixelHitCombined) const;

    /// \brief Populate record fields from charge sharing result.
    void PopulateRecordFromChargeResult(IO::EventRecord& record, const ChargeSharingCalculator::Result& result) const;

    /// \brief Build default grid geometry when charge sharing was not computed.
    [[nodiscard]] ChargeSharingCalculator::PixelGridGeometry BuildDefaultGridGeometry() const;

    struct NeighborContext {
        G4double sigmaNoise{0.0};
    };

    [[nodiscard]] NeighborContext MakeNeighborContext() const;
    void PopulateNeighborCharges(const ChargeSharingCalculator::Result& result, const NeighborContext& context);

    RunAction* fRunAction{nullptr};
    DetectorConstruction* fDetector{nullptr};
    SteppingAction* fSteppingAction{nullptr};

    G4int fNeighborhoodRadius{Constants::NEIGHBORHOOD_RADIUS};

    G4ThreeVector fFirstContactPos;
    G4double fFirstContactTime{0.0};
    G4bool fHasFirstContactPos{false};

    G4int fPixelIndexI{-1};
    G4int fPixelIndexJ{-1};
    G4double fPixelTrueDeltaX{0.};
    G4double fPixelTrueDeltaY{0.};

    G4double fIonizationEnergy{Constants::IONIZATION_ENERGY};
    G4double fGain{Constants::GAIN};
    G4double fD0{Constants::D0};
    G4double fElementaryCharge{Constants::ELEMENTARY_CHARGE};

    G4double fScorerEnergyDeposit{0.0};

    ChargeSharingCalculator fChargeSharing;
    G4int fNearestPixelGlobalId{-1};
    NeighborhoodLayout fNeighborhoodLayout;
    std::vector<G4double> fNeighborhoodChargeNew;
    std::vector<G4double> fNeighborhoodChargeFinal;
    // Row-mode noisy charges
    std::vector<G4double> fNeighborhoodChargeNewRow;
    std::vector<G4double> fNeighborhoodChargeFinalRow;
    // Col-mode noisy charges
    std::vector<G4double> fNeighborhoodChargeNewCol;
    std::vector<G4double> fNeighborhoodChargeFinalCol;
    // Block-mode noisy charges
    std::vector<G4double> fNeighborhoodChargeNewBlock;
    std::vector<G4double> fNeighborhoodChargeFinalBlock;
    G4bool fEmitDistanceAlphaOutputs{false};
    G4bool fComputeFullFractions{false};
    std::unique_ptr<G4GenericMessenger> fMessenger;
};

} // namespace ECS

// Backward compatibility alias
using EventAction = ECS::EventAction;

#endif // ECS_EVENT_ACTION_HH
