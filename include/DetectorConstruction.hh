#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4NistManager.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
// Multi-Functional Detector includes
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PSNofStep.hh"

class EventAction;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    virtual ~DetectorConstruction();
    
    virtual G4VPhysicalVolume* Construct();
    void ConstructSDandField() override;
    
    // Method to set EventAction pointer for neighborhood configuration
    void SetEventAction(EventAction* eventAction) { fEventAction = eventAction; }
    
    // Set fixed pixel-pad corner offset (requires geometry rebuild)
    void SetPixelCornerOffset(G4double cornerOffset);
    
    // Method to set neighborhood radius
    void SetNeighborhoodRadius(G4int radius);
    
    // Method to get neighborhood radius  
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    // Getter methods for parameters needed to calculate nearest pixel
    G4double GetPixelSize() const { return fPixelSize; }
    G4double GetPixelSpacing() const { return fPixelSpacing; }
    G4double GetPixelCornerOffset() const { return fPixelCornerOffset; }
    G4double GetDetSize() const { return fDetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }
    G4ThreeVector GetDetectorPos() const; // Fixed position from Construct()
    
    // Method to save simulation parameters to a log file
    void SaveSimulationParameters(G4double totalPixelArea, G4double detectorArea, G4double pixelAreaRatio) const;
    
private:
    // Pixel parameters
    G4double fPixelSize;         // Size of each pixel
    G4double fPixelWidth;        // Width/thickness of each pixel
    G4double fPixelSpacing;      // Center-to-center spacing between pixels
    G4double fPixelCornerOffset; // Edge-most pixel distance from edge of detector
    
    // Detector parameters
    G4double fDetSize;           // Size of the detector
    G4double fDetWidth;          // Width/thickness of the detector
    
    G4int fNumBlocksPerSide;     // Number of pixels per row/column
    
    // Control flags
    G4bool fCheckOverlaps;       // Flag to check geometry overlaps
    
    // EventAction pointer for neighborhood configuration
    EventAction* fEventAction;
    
    // Neighborhood radius
    G4int fNeighborhoodRadius;
    
    G4LogicalVolume* fLogicSilicon;
};

#endif