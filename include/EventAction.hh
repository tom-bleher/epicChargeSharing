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
    
    // Method to accumulate energy deposition and pos
    void AddEdep(G4double edep, G4ThreeVector pos);
    
    // Method to set initial particle pos
    void SetInitialPos(const G4ThreeVector& pos);
    
    // Method to calculate the nearest pixel pos
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
    
    // Method to calculate the pixel alpha angle
    G4double CalcPixelAlpha(const G4ThreeVector& hitPos, G4int pixelI, G4int pixelJ);
    
    // Method to calculate angles from hit to neighborhood (9x9) grid around hit pixel
    void CalcNeighborhoodGridAngles(const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ);
    
    // Method to calculate the angular size subtended by a pixel as seen from a hit point (2D calculation)
    G4double CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                         G4double pixelCenterX, G4double pixelCenterY,
                                         G4double pixelWidth, G4double pixelHeight);
    
    // Method to calculate charge sharing in the neighborhood (9x9) grid
    void CalcNeighborhoodChargeSharing();
    
    // Method to set neighborhood radius (default is 4 for 9x9 grid)
    void SetNeighborhoodRadius(G4int radius) { fNeighborhoodRadius = radius; }
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    // Method to enable/disable automatic radius selection
    void SetAutoRadiusEnabled(G4bool enabled) { fAutoRadiusEnabled = enabled; }
    G4bool GetAutoRadiusEnabled() const { return fAutoRadiusEnabled; }
    
    // Method to set radius range for automatic selection
    void SetAutoRadiusRange(G4int minRadius, G4int maxRadius) { 
        fMinAutoRadius = minRadius; 
        fMaxAutoRadius = maxRadius; 
    }
    
    // Scorer data collection methods
    void CollectScorerData(const G4Event* event);
    void SetScorerData(G4double energy, G4int hits, G4bool valid);
    G4double GetScorerEnergyDeposit() const { return fScorerEnergyDeposit; }
    G4int GetScorerHitCount() const { return fScorerHitCount; }
    G4bool IsScorerDataValid() const { return fScorerDataValid; }
    
    // Validation methods
    void ValidateScorerDataIntegrity() const;
    
    // Exclusive Multi-Functional Detector validation methods
    void ValidateHitPurity();
    G4bool ShouldCalculateChargeSharing() const;
    void ConditionalChargeCalculation(const G4Event* event);
    
private:
    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    SteppingAction* fSteppingAction;
    
    // Neighborhood configuration
    G4int fNeighborhoodRadius;  // Radius of neighborhood grid (4 = 9x9, 3 = 7x7, etc.)
    
    G4double fEdep;   // Total energy depositionit in the event
    G4ThreeVector fPos;  // Pos of energy depositionit (weighted average)
    G4ThreeVector fInitialPos; // Initial particle pos
    G4bool fHasHit;   // Flag to indicate if any energy was deposited
    
    // Hit purity tracking variables for Multi-Functional Detector validation
    G4bool fPureSiliconHit;
    G4bool fAluminumContaminated;
    G4bool fChargeCalculationEnabled;
    
    // Pixel mapping information
    G4int fPixelIndexI;    // Pixel index in the X direction
    G4int fPixelIndexJ;    // Pixel index in the Y direction
    G4double fPixelTrueDeltaX; // Delta X from pixel center to hit (x_pixel - x_true) [mm]
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to hit (y_pixel - y_true) [mm]
    G4double fActualPixelDistance; // Actual distance from hit to pixel center (always calculated)
    G4bool fPixelHit;     // Flag to indicate if the hit was on a pixel
    
    // Neighborhood (9x9) grid angle information (for non-pixel hits)
    std::vector<G4double> fNeighborhoodAngles; // Angles from hit to each pixel in neighborhood grid
    
    // Neighborhood (9x9) grid charge sharing information (for non-pixel hits)
    std::vector<G4double> fNeighborhoodChargeFractions; // Charge fraction for each pixel in neighborhood grid
    std::vector<G4double> fNeighborhoodDistances;       // Distance from hit to each pixel center in neighborhood grid
    std::vector<G4double> fNeighborhoodCharge;  // Actual charge value for each pixel in neighborhood grid (Coulombs)
    
    // Physics constants for charge sharing calculation
    G4double fIonizationEnergy;    // eV per electron-hole pair in silicon
    G4double fAmplificationFactor; // AC-LGAD amplification factor
    G4double fD0;                  // microns - reference distance for charge sharing
    G4double fElementaryCharge;    // Coulombs - elementary charge
    
    // Automatic radius selection
    G4bool fAutoRadiusEnabled;
    G4int fMinAutoRadius;
    G4int fMaxAutoRadius;
    G4int fSelectedRadius;
    G4double fSelectedQuality;
    
    // Helper methods for automatic radius selection
    G4int SelectOptimalRadius(const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ);
    G4double EvaluateFitQuality(G4int radius, const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ);
    
    // Helper methods for exclusive Multi-Functional Detector validation
    void SetDefaultFittingResults();
    void PerformChargeShareFitting(const G4Event* event, const G4ThreeVector& nearestPixel);
    
    // Hit purity status getter methods for SteppingAction integration
    G4bool GetPureSiliconHit() const { return fPureSiliconHit; }
    G4bool GetAluminumContaminated() const { return fAluminumContaminated; }
    G4bool GetChargeCalculationEnabled() const { return fChargeCalculationEnabled; }
    G4bool GetHitPurityStatus() const { return fPureSiliconHit && !fAluminumContaminated; }
    
    // Comprehensive hit validation summary combining SteppingAction trajectory analysis with EventAction validation
    void GetHitValidationSummary(G4bool& pureSiliconHit, G4bool& aluminumContaminated, 
                                 G4bool& chargeCalculationEnabled, G4String& firstInteractionVolume,
                                 G4bool& hasAluminumInteraction, G4bool& hasAluminumPreContact) const;
    
    // Integration validation methods for SteppingAction connection
    G4bool IsSteppingActionConnected() const { return fSteppingAction != nullptr; }
    void ValidateSteppingActionIntegration() const;
    
    // Complete integration workflow demonstration
    void DemonstrateIntegrationWorkflow() const;
    
    // Scorer data storage
    G4double fScorerEnergyDeposit;
    G4int fScorerHitCount;
    G4bool fScorerDataValid;
};

#endif