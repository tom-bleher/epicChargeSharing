/**
 * @file DetectorConstruction.hh
 * @brief Declares `DetectorConstruction`, which builds the detector geometry and
 *        configures sensitive detectors and scoring for the silicon volume.
 */
#ifndef DETECTORCONSTRUCTION_HH
#define DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <mutex>

class EventAction;
class G4LogicalVolume;
class G4Box;
class G4NistManager;
class G4PVPlacement;
class G4VisAttributes;
class G4MultiFunctionalDetector;
class G4VPrimitiveScorer;
class G4Material;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    struct PixelLocation {
        G4ThreeVector center;
        G4int indexI{0};
        G4int indexJ{0};
        G4bool withinDetector{false};
    };

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
    G4double GetPixelPitch() const { return fPixelSpacing; }
    G4double GetLinearChargeModelBeta(G4double pitch) const;
    G4double GetPixelCornerOffset() const { return fPixelCornerOffset; }
    G4double GetDetSize() const { return fDetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }
    G4ThreeVector GetDetectorPos() const;
    PixelLocation FindNearestPixel(const G4ThreeVector& pos) const;
    // Noise sigma accessors (row-major global pixel id = i*N + j)
    G4double GetPixelGainSigma(G4int globalPixelId) const { return (globalPixelId >= 0 && globalPixelId < (G4int)fPixelGainSigmas.size()) ? fPixelGainSigmas[globalPixelId] : 0.0; }
    const std::vector<G4double>& GetPixelGainSigmas() const { return fPixelGainSigmas; }
    
    void SaveSimulationParameters(G4double totalPixelArea, G4double detectorArea, G4double pixelAreaRatio) const;
    
private:
    struct MaterialSet {
        G4Material* world;
        G4Material* silicon;
        G4Material* aluminum;
    };

    struct PixelGridStats {
        G4double totalPixelArea;
        G4double detectorArea;
        G4double coverage;
    };

    MaterialSet PrepareMaterials() const;
    G4VPhysicalVolume* BuildWorld(const MaterialSet& mats, G4bool checkOverlaps, G4LogicalVolume*& logicWorld);
    G4LogicalVolume* BuildSiliconDetector(G4LogicalVolume* logicWorld, const MaterialSet& mats, G4bool checkOverlaps, G4double originalDetSize);
    PixelGridStats ConfigurePixels(G4LogicalVolume* logicWorld, G4LogicalVolume* siliconLogical, const MaterialSet& mats, G4bool checkOverlaps);
    void InitializePixelGainSigmas();
    void WriteGeometryLog(const PixelGridStats& stats, G4double originalDetSize) const;
    void SyncRunMetadata();

    G4double fPixelSize;
    G4double fPixelWidth;
    G4double fPixelSpacing;
    G4double fPixelCornerOffset;
    
    G4double fDetSize;
    G4double fDetWidth;
    G4int fNumBlocksPerSide;

    mutable std::once_flag fLinearModelWarningFlag;

    EventAction* fEventAction;
    
    G4int fNeighborhoodRadius;

    G4LogicalVolume* fLogicSilicon;

    // Per-pixel multiplicative noise sigma table (initialized once during construction)
    std::vector<G4double> fPixelGainSigmas;
};

#endif // DETECTORCONSTRUCTION_HH
