#ifndef EVENTACTION_HH
#define EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

class RunAction;
class DetectorConstruction;
class SteppingAction;

class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction, DetectorConstruction* detector);
    ~EventAction() override;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;
    
    void SetSteppingAction(SteppingAction* steppingAction) { fSteppingAction = steppingAction; }
    
    void RegisterFirstContact(const G4ThreeVector& pos) { fFirstContactPos = pos; fHasFirstContactPos = true; }
    
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
        
    G4double CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                         G4double pixelCenterX, G4double pixelCenterY,
                                         G4double pixelWidth, G4double pixelHeight);
    
    void CalcNeighborhoodChargeSharing(const G4ThreeVector& hitPos);
    
    // Neighborhood grid configuration: radius r => (2r+1)x(2r+1) cells.
    // Flattening: row-major with idx=(d_i+r)*(2r+1)+(d_j+r), d_i,d_j in [-r,r].
    // F_i in [0,1], sums to ~1 for in-bounds cells; out-of-bounds cells set
    // F_i=Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL, Q_i=0 (per igor.txt spec).
    // Units: e_dep in MeV; internal conversion to eV via CLHEP units; Q_i in Coulombs.
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
    
    G4double fIonizationEnergy;
    G4double fAmplificationFactor;
    G4double fD0;
    G4double fElementaryCharge;
    
    G4double fScorerEnergyDeposit;
};

#endif