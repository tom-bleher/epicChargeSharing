#ifndef EVENTACTION_HH
#define EVENTACTION_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

class RunAction;
class DetectorConstruction;
class MinuitGaussianFitter;

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
    
    void SetFinalParticleEnergy(G4double finalEnergy);
    
    // Method to add step-by-step energy deposition information
    void AddStepEnergyDeposition(G4double edep, G4double z, G4double time);
    
    // Method to add ALL step information (including non-energy depositing steps)
    void AddAllStepInfo(G4double edep, G4double z, G4double time);
    
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
    G4double fPixelTrueDist; // Distance from hit to pixel center
    G4bool fPixelHit;     // Flag to indicate if the hit was on a pixel
    
    // Neighborhood (9x9) grid angle information
    std::vector<G4double> fGridNeighborhoodAngles; // Angles from hit to each pixel in neighborhood grid
    std::vector<G4int> fGridNeighborhoodPixelI;     // I indices of pixels in neighborhood grid
    std::vector<G4int> fGridNeighborhoodPixelJ;     // J indices of pixels in neighborhood grid
    
    // Neighborhood (9x9) grid charge sharing information
    std::vector<G4double> fGridNeighborhoodChargeFractions; // Charge fraction for each pixel in neighborhood grid
    std::vector<G4double> fGridNeighborhoodDistances;       // Distance from hit to each pixel center in neighborhood grid
    std::vector<G4double> fGridNeighborhoodCharge;  // Actual charge value for each pixel in neighborhood grid (Coulombs)
    
    // Constants for charge sharing calculation
    static constexpr G4double fIonizationEnergy = 3.6; // eV - typical for silicon
    static constexpr G4double fAmplificationFactor = 20.0; // AC-LGAD amplification factor
    static constexpr G4double fD0 = 10.0; // microns - reference distance for charge sharing
    static constexpr G4double fElementaryCharge = 1.602176634e-19; // Coulombs - elementary charge
    
    // Additional tracking variables for enhanced ROOT output
    G4double fInitialParticleEnergy;   // Initial particle energy
    G4double fFinalParticleEnergy;     // Final particle energy  
    G4double fParticleMomentum;        // Particle momentum
    G4String fParticleName;            // Particle type name
    
    // Step-by-step energy deposition information
    std::vector<G4double> fStepEnergyDeposition;    // Energy deposited per step
    std::vector<G4double> fStepZPositions;       // Z position of each energy deposit
    std::vector<G4double> fStepTimes;    // Time of each energy deposit
    std::vector<G4int> fStepNumbers;        // Step number for each energy deposit
    
    // ALL step information (including non-energy depositing steps)
    std::vector<G4double> fAllStepEnergyDeposition;    // Energy deposited per step (including 0)
    std::vector<G4double> fAllStepZPositions;       // Z position of each step
    std::vector<G4double> fAllStepTimes;    // Time of each step
    std::vector<G4int> fAllStepNumbers;        // Step number for each step
    
    // 3D Gaussian fitter instance
    // Gaussian3DFitter* fGaussianFitter;

    // ROOT Minuit-based 3D Gaussian fitter instance
    MinuitGaussianFitter* fGaussianFitter;
};

#endif