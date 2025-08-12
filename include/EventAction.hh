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
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);
    
    // Method to set SteppingAction pointer for aluminum interaction tracking
    void SetSteppingAction(SteppingAction* steppingAction) { fSteppingAction = steppingAction; }
    
    // Method to accumulate silicon step positions (unweighted)
    void AddSiliconPos(const G4ThreeVector& pos);

      // Record the very first contact position (aluminum pixel or silicon)
      void RegisterFirstContact(const G4ThreeVector& pos) { fFirstContactPos = pos; fHasFirstContactPos = true; }
    
    // Method to calculate the nearest pixel pos
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
        
    // Method to calculate the angular size subtended by a pixel as seen from a hit point (2D calculation)
    G4double CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                         G4double pixelCenterX, G4double pixelCenterY,
                                         G4double pixelWidth, G4double pixelHeight);
    
    // Method to calculate charge sharing in the neighborhood (9x9) grid
    void CalcNeighborhoodChargeSharing();
    
    // Method to set neighborhood radius (default is 4 for 9x9 grid)
    void SetNeighborhoodRadius(G4int radius) { fNeighborhoodRadius = radius; }
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    // Scorer data collection methods
    void CollectScorerData(const G4Event* event);
    G4bool IsScorerDataValid() const { return fScorerDataValid; }

private:
    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    SteppingAction* fSteppingAction;
    
    // Neighborhood configuration
    G4int fNeighborhoodRadius;  // Radius of neighborhood grid (4 = 9x9, 3 = 7x7, etc.)
    
    G4ThreeVector fPos;  // Representative hit position (average of silicon step midpoints)
    G4int fNumPosSamples; // Number of silicon step samples accumulated
      // First-contact position and flag
      G4ThreeVector fFirstContactPos; // Position at the first volume entry (pixel or silicon)
      G4bool       fHasFirstContactPos{false};
    
    G4bool fHasHit;   // Flag to indicate if any energy was deposited
    
    // Pixel mapping information
    G4int fPixelIndexI;    // Pixel index in the X direction
    G4int fPixelIndexJ;    // Pixel index in the Y direction
    G4double fPixelTrueDeltaX; // Delta X from pixel center to hit (x_pixel - x_true) [mm]
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to hit (y_pixel - y_true) [mm]
    G4double fActualPixelDistance; // Actual distance from hit to pixel center (always calculated)
    G4bool fPixelHit;     // Flag to indicate if the hit was on a pixel
    
    // Neighborhood (9x9) grid charge sharing information (for non-pixel hits)
    std::vector<G4double> fNeighborhoodChargeFractions; // Charge fraction for each pixel in neighborhood grid
    std::vector<G4double> fNeighborhoodCharge;  // Actual charge value for each pixel in neighborhood grid (Coulombs)
    
    // Physics constants for charge sharing calculation
    G4double fIonizationEnergy;    // eV per electron-hole pair in silicon
    G4double fAmplificationFactor; // AC-LGAD amplification factor
    G4double fD0;                  // microns - reference distance for charge sharing
    G4double fElementaryCharge;    // Coulombs - elementary charge
    
    // Scorer data storage (authoritative energy deposition from MFD)
    G4double fScorerEnergyDeposit;
    G4int fScorerHitCount;
    G4bool fScorerDataValid;
    
    // Hit purity tracking
    G4bool fPureSiliconHit;
    G4bool fAluminumContaminated;
    G4bool fChargeCalculationEnabled;
};

#endif