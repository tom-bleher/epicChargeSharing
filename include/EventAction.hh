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
    
private:
    RunAction* fRunAction;
    DetectorConstruction* fDetector;
    G4double fEdep;   // Total energy deposit in the event
    G4ThreeVector fPosition;  // Position of energy deposit (weighted average)
    G4ThreeVector fInitialPosition; // Initial particle position
    G4bool fHasHit;   // Flag to indicate if any energy was deposited
    
    // Pixel mapping information
    G4int fPixelIndexI;    // Pixel index in the X direction
    G4int fPixelIndexJ;    // Pixel index in the Y direction
    G4double fPixelDistance; // Distance from hit to pixel center
    G4bool fPixelHit;     // Flag to indicate if the hit was on a pixel
    
    // Neighborhood (9x9) grid angle information
    std::vector<G4double> fGridNeighborhoodAngles; // Angles from hit to each pixel in neighborhood grid
    std::vector<G4int> fGridNeighborhoodPixelI;     // I indices of pixels in neighborhood grid
    std::vector<G4int> fGridNeighborhoodPixelJ;     // J indices of pixels in neighborhood grid
    
    // Neighborhood (9x9) grid charge sharing information
    std::vector<G4double> fGridNeighborhoodChargeFractions; // Charge fraction for each pixel in neighborhood grid
    std::vector<G4double> fGridNeighborhoodDistances;       // Distance from hit to each pixel center in neighborhood grid
    std::vector<G4double> fGridNeighborhoodChargeValues;    // Actual charge value for each pixel in neighborhood grid (electrons)
    std::vector<G4double> fGridNeighborhoodChargeCoulombs;  // Actual charge value for each pixel in neighborhood grid (Coulombs)
    
    // Constants for charge sharing calculation
    static constexpr G4double fIonizationEnergy = 3.6; // eV - typical for silicon
    static constexpr G4double fAmplificationFactor = 20.0; // AC-LGAD amplification factor
    static constexpr G4double fD0 = 10.0; // microns - reference distance for charge sharing
    static constexpr G4double fElementaryCharge = 1.602176634e-19; // Coulombs - elementary charge
};

#endif