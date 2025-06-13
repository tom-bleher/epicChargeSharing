#ifndef EVENTACTION_HH
#define EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

class RunAction;
class DetectorConstruction;

class EventAction : public G4UserEventAction
{
public:
    EventAction(RunAction* runAction, DetectorConstruction* detector);
    virtual ~EventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);
    
    // Method to accumulate energy deposition and position
    void AddEdep(G4double edep, G4ThreeVector position);
    
    // Method to set initial particle position
    void SetInitialPosition(const G4ThreeVector& position);
    
    // Method to calculate the nearest pixel position
    G4ThreeVector CalculateNearestPixel(const G4ThreeVector& position);
    
    // Method to calculate the pixel alpha angle
    G4double CalculatePixelAlpha(const G4ThreeVector& hitPosition, G4int pixelI, G4int pixelJ);
    
    // Method to calculate angles from hit to neighborhood (9x9) grid around hit pixel
    void CalculateNeighborhoodGridAngles(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ);
    
    // Method to calculate the angular size subtended by a pixel as seen from a hit point (2D calculation)
    G4double CalculatePixelAlphaSubtended(G4double hitX, G4double hitY,
                                         G4double pixelCenterX, G4double pixelCenterY,
                                         G4double pixelWidth, G4double pixelHeight);
    
    // Method to calculate charge sharing in the neighborhood (9x9) grid
    void CalculateNeighborhoodChargeSharing();
    
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
    
    // Method to get the selected radius for the current event
    G4int GetSelectedRadius() const { return fSelectedRadius; }
    
    // Method to get the fit quality for the selected radius
    G4double GetSelectedFitQuality() const { return fSelectedFitQuality; }
    
private:
    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    
    // Neighborhood configuration
    G4int fNeighborhoodRadius;  // Radius of neighborhood grid (4 = 9x9, 3 = 7x7, etc.)
    
    G4double fEdep;   // Total energy deposit in the event
    G4ThreeVector fPosition;  // Position of energy deposit (weighted average)
    G4ThreeVector fInitialPosition; // Initial particle position
    G4bool fHasHit;   // Flag to indicate if any energy was deposited
    
    // Pixel mapping information
    G4int fPixelIndexI;    // Pixel index in the X direction
    G4int fPixelIndexJ;    // Pixel index in the Y direction
    G4double fPixelTrueDeltaX; // Delta X from pixel center to hit (x_pixel - x_true) [mm]
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to hit (y_pixel - y_true) [mm]
    G4double fActualPixelDistance; // Actual distance from hit to pixel center (always calculated)
    G4bool fPixelHit;     // Flag to indicate if the hit was on a pixel
    
    // Neighborhood (9x9) grid angle information (for non-pixel hits)
    std::vector<G4double> fNonPixel_GridNeighborhoodAngles; // Angles from hit to each pixel in neighborhood grid
    
    // Neighborhood (9x9) grid charge sharing information (for non-pixel hits)
    std::vector<G4double> fNonPixel_GridNeighborhoodChargeFractions; // Charge fraction for each pixel in neighborhood grid
    std::vector<G4double> fNonPixel_GridNeighborhoodDistances;       // Distance from hit to each pixel center in neighborhood grid
    std::vector<G4double> fNonPixel_GridNeighborhoodCharge;  // Actual charge value for each pixel in neighborhood grid (Coulombs)
    
    // Constants for charge sharing calculation
    static constexpr G4double fIonizationEnergy = 3.6; // eV - typical for silicon
    static constexpr G4double fAmplificationFactor = 20.0; // AC-LGAD amplification factor
    static constexpr G4double fD0 = 10.0; // microns - reference distance for charge sharing
    static constexpr G4double fElementaryCharge = 1.602176634e-19; // Coulombs - elementary charge
    
    // Automatic radius selection
    G4bool fAutoRadiusEnabled;
    G4int fMinAutoRadius;
    G4int fMaxAutoRadius;
    G4int fSelectedRadius;
    G4double fSelectedFitQuality;
    
    // Method to perform automatic radius selection based on fit quality
    G4int SelectOptimalRadius(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ);
    
    // Method to evaluate fit quality for a given radius
    G4double EvaluateFitQuality(G4int radius, const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ);
    
    // Method to calculate residuals and goodness of fit metrics
    G4double CalculateGoodnessOfFit(const std::vector<double>& observed, const std::vector<double>& fitted);
};

#endif