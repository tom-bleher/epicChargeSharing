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

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override;
    
    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;
    
    void SetEventAction(EventAction* eventAction) { fEventAction = eventAction; }
    
    void SetPixelCornerOffset(G4double cornerOffset);
    
    void SetNeighborhoodRadius(G4int radius);
    
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    
    G4double GetPixelSize() const { return fPixelSize; }
    G4double GetPixelSpacing() const { return fPixelSpacing; }
    G4double GetPixelCornerOffset() const { return fPixelCornerOffset; }
    G4double GetDetSize() const { return fDetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }
    G4ThreeVector GetDetectorPos() const;
    
    void SaveSimulationParameters(G4double totalPixelArea, G4double detectorArea, G4double pixelAreaRatio) const;
    
private:
    G4double fPixelSize;
    G4double fPixelWidth;
    G4double fPixelSpacing;
    G4double fPixelCornerOffset;
    
    G4double fDetSize;
    G4double fDetWidth;
    G4int fNumBlocksPerSide;

    EventAction* fEventAction;
    
    G4int fNeighborhoodRadius;

    G4LogicalVolume* fLogicSilicon;
};

#endif