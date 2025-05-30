#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Tubs.hh"
#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"

class DetectorMessenger;
class EventAction;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    virtual ~DetectorConstruction();
    
    virtual G4VPhysicalVolume* Construct();
    
    // Method to set EventAction pointer for neighborhood configuration
    void SetEventAction(EventAction* eventAction) { fEventAction = eventAction; }
    
    // Method to set the grid parameters
    void SetGridParameters(G4double pixelSize, G4double pixelSpacing, G4double pixelCornerOffset, G4int numPixels);
    
    // Method to set the corner offset directly (requires geometry rebuild)
    void SetPixelCornerOffset(G4double cornerOffset);
    
    // Method to set neighborhood radius
    void SetNeighborhoodRadius(G4int radius);
    
    // Getter method for DetectorMessenger
    DetectorMessenger* GetDetectorMessenger() const { return fDetectorMessenger; }
    
    // Getter methods for parameters needed to calculate nearest pixel
    G4double GetPixelSize() const { return fPixelSize; }
    G4double GetPixelSpacing() const { return fPixelSpacing; }
    G4double GetPixelCornerOffset() const { return fPixelCornerOffset; }
    G4double GetDetSize() const { return fdetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }
    G4ThreeVector GetDetectorPosition() const { return G4ThreeVector(0., 0., -1.0*cm); } // Fixed position from Construct()
    
    // Method to check if a position is within a pixel area
    G4bool IsPositionOnPixel(const G4ThreeVector& position) const;
    
    // Method to save simulation parameters to a log file
    void SaveSimulationParameters(G4double totalPixelArea, G4double detectorArea, G4double pixelAreaRatio) const;
    
    // Method to save grid parameters to a file for ROOT merging
    void SaveGridParametersToFile() const;
    
private:
    // Pixel parameters
    G4double fPixelSize;         // Size of each pixel
    G4double fPixelWidth;        // Width/thickness of each pixel
    G4double fPixelSpacing;      // Center-to-center spacing between pixels
    G4double fPixelCornerOffset; // Edge-most pixel distance from edge of detector
    
    // Detector parameters
    G4double fdetSize;           // Size of the detector
    G4double fdetWidth;          // Width/thickness of the detector
    
    G4int fNumBlocksPerSide;     // Number of pixels per row/column
    
    // Control flags
    G4bool fCheckOverlaps;       // Flag to check geometry overlaps
    
    // EventAction pointer for neighborhood configuration
    EventAction* fEventAction;
    
    // Messenger for UI commands
    DetectorMessenger* fDetectorMessenger;
};

#endif