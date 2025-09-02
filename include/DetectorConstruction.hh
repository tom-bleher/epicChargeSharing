#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"

class EventAction;
class G4LogicalVolume;
class G4Box;
class G4NistManager;
class G4PVPlacement;
class G4VisAttributes;
class G4MultiFunctionalDetector;
class G4VPrimitiveScorer;

// Builds AC-LGAD geometry and exposes parameters.
class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    virtual ~DetectorConstruction();
    
    virtual G4VPhysicalVolume* Construct();
    void ConstructSDandField() override;
    
    // Set EventAction
    void SetEventAction(EventAction* eventAction) { fEventAction = eventAction; }
    
    // Set fixed pixel corner offset
    void SetPixelCornerOffset(G4double cornerOffset);
    
    // Set neighborhood radius
    void SetNeighborhoodRadius(G4int radius);
    
    // Get neighborhood radius  
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    // Geometry getters
    G4double GetPixelSize() const { return fPixelSize; }
    G4double GetPixelSpacing() const { return fPixelSpacing; }
    G4double GetPixelCornerOffset() const { return fPixelCornerOffset; }
    G4double GetDetSize() const { return fDetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }
    G4ThreeVector GetDetectorPos() const; // Fixed position from Construct()
    
    // Save parameters to log
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