#ifndef EVENTACTION_HH
#define EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"

#include "ChargeSharingCalculator.hh"

#include <memory>
#include <vector>

class RunAction;
class DetectorConstruction;
class SteppingAction;
class G4Event;
class G4GenericMessenger;

/**
 * Event-level bookkeeping and data extraction.
 * - Tracks first contact, classifies pixel hits, computes charge sharing
 *   in a (2r+1)x(2r+1) neighborhood around the nearest pixel.
 */
class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction, DetectorConstruction* detector);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

    void SetSteppingAction(SteppingAction* steppingAction) { fSteppingAction = steppingAction; }

    void RegisterFirstContact(const G4ThreeVector& pos)
    {
        fFirstContactPos = pos;
        fHasFirstContactPos = true;
    }

    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);

    void SetNeighborhoodRadius(G4int radius)
    {
        fNeighborhoodRadius = radius;
        fChargeSharing.SetNeighborhoodRadius(radius);
    }
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }

    void CollectScorerData(const G4Event* event);

    void SetEmitDistanceAlpha(G4bool enabled);

private:
    G4ThreeVector DetermineHitPosition() const;
    void UpdatePixelAndHitClassification(const G4ThreeVector& hitPos,
                                         G4ThreeVector& nearestPixel,
                                         G4bool& firstContactIsPixel,
                                         G4bool& geometricIsPixel,
                                         G4bool& isPixelHitCombined);
    void ComputeChargeSharingForEvent(const G4ThreeVector& hitPos, G4double energyDeposit);
    void EnsureNeighborhoodBuffers(std::size_t targetSize);

    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    SteppingAction* fSteppingAction;

    G4int fNeighborhoodRadius;

    G4ThreeVector fFirstContactPos;
    G4bool fHasFirstContactPos{false};

    G4int fPixelIndexI;
    G4int fPixelIndexJ;
    G4double fPixelTrueDeltaX;
    G4double fPixelTrueDeltaY;

    G4double fIonizationEnergy;
    G4double fAmplificationFactor;
    G4double fD0;
    G4double fElementaryCharge;

    G4double fScorerEnergyDeposit;

    ChargeSharingCalculator fChargeSharing;
    std::vector<G4double> fNeighborhoodChargeNew;
    std::vector<G4double> fNeighborhoodChargeFinal;
    G4bool fEmitDistanceAlphaOutputs{false};
    std::unique_ptr<G4GenericMessenger> fMessenger;
};

#endif // EVENTACTION_HH
