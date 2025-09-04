#ifndef EVENTACTION_HH
#define EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

class RunAction;
class DetectorConstruction;
class SteppingAction;

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
    
    void RegisterFirstContact(const G4ThreeVector& pos) { fFirstContactPos = pos; fHasFirstContactPos = true; }
    
    // Find center of nearest pixel to a given hit position (returns mm)
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
        
    G4double CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                         G4double pixelCenterX, G4double pixelCenterY,
                                         G4double pixelWidth, G4double pixelHeight);
    
    // Compute analytical charge sharing for current neighborhood
    void CalcNeighborhoodChargeSharing(const G4ThreeVector& hitPos);
    // Compute neighborhood pixel geometry and IDs matching fraction layout
    void CalcNeighborhoodPixelGeometryAndIDs(const G4ThreeVector& hitPos);
    
    // Neighborhood grid: r => (2r+1)x(2r+1); row-major flattening idx=(d_i+r)*(2r+1)+(d_j+r), d_i,d_j∈[-r,r].
    // F_i ∈ [0,1], sums to ~1 for in-bounds; OOB => F_i=sentinel, Q_i=0. Units: E_dep [MeV], Q_i [C].
    void SetNeighborhoodRadius(G4int radius) { fNeighborhoodRadius = radius; }
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    void CollectScorerData(const G4Event* event);

private:
    G4ThreeVector DetermineHitPosition() const;
    void UpdatePixelAndHitClassification(const G4ThreeVector& hitPos,
                                         G4ThreeVector& nearestPixel,
                                         G4bool& firstContactIsPixel,
                                         G4bool& geometricIsPixel,
                                         G4bool& isPixelHitCombined);
    void ComputeChargeSharingForEvent(const G4ThreeVector& hitPos);

    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    SteppingAction* fSteppingAction;
    
    G4int fNeighborhoodRadius;
    
      G4ThreeVector fFirstContactPos;
      G4bool       fHasFirstContactPos{false};
    
    G4int fPixelIndexI;
    G4int fPixelIndexJ;
    G4double fPixelTrueDeltaX;
    G4double fPixelTrueDeltaY;
    
    std::vector<G4double> fNeighborhoodChargeFractions;
    std::vector<G4double> fNeighborhoodCharge;
    
    // Neighborhood geometry (aligned to fractions vector sizing and ordering)
    std::vector<G4double> fNeighborhoodPixelX; // mm
    std::vector<G4double> fNeighborhoodPixelY; // mm
    std::vector<G4int>    fNeighborhoodPixelID; // global grid ID; -1 for OOB
    
    G4double fIonizationEnergy;
    G4double fAmplificationFactor;
    G4double fD0;
    G4double fElementaryCharge;
    
    G4double fScorerEnergyDeposit;
};

#endif