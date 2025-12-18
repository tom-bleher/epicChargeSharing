/// \file DetectorConstruction.hh
/// \brief Definition of the ECS::DetectorConstruction class.
///
/// This file declares the DetectorConstruction class which builds the
/// AC-LGAD detector geometry including the silicon slab and aluminum
/// pixel pad array.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_DETECTOR_CONSTRUCTION_HH
#define ECS_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "Config.hh"
#include <memory>
#include <vector>
#include <mutex>

class G4LogicalVolume;
class G4GenericMessenger;
class G4Box;
class G4NistManager;
class G4PVPlacement;
class G4VisAttributes;
class G4MultiFunctionalDetector;
class G4VPrimitiveScorer;
class G4Material;

namespace ECS {

// Forward declarations within namespace
class EventAction;
class RunAction;

/// \brief Constructs the AC-LGAD detector geometry.
///
/// This class builds the complete detector geometry consisting of:
/// - World volume (air)
/// - Silicon detector slab with configurable dimensions
/// - Grid of aluminum pixel pads with configurable size and pitch
///
/// The class also manages per-pixel gain noise parameters and provides
/// methods for finding the nearest pixel to a given position.
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
    void SetRunAction(RunAction* runAction);
    
    void SetGridOffset(G4double offset);
    void SetPixelSize(G4double size);
    void SetPixelSpacing(G4double spacing);
    void SetNeighborhoodRadius(G4int radius);

    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }

    G4double GetPixelSize() const { return fPixelSize; }
    G4double GetPixelSpacing() const { return fPixelSpacing; }
    G4double GetPixelPitch() const { return fPixelSpacing; }
    G4double GetLinearChargeModelBeta() const { return Constants::LINEAR_CHARGE_MODEL_BETA; }
    G4double GetGridOffset() const { return fGridOffset; }
    G4double GetDetSize() const { return fDetSize; }
    G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }

    /// @brief Get minimum pixel index (DD4hep-style, can be negative for centered grid).
    G4int GetMinIndexX() const { return fMinIndexX; }
    G4int GetMinIndexY() const { return fMinIndexY; }
    G4int GetMaxIndexX() const { return fMaxIndexX; }
    G4int GetMaxIndexY() const { return fMaxIndexY; }
    const G4ThreeVector& GetDetectorPos() const { return fDetectorPos; }
    PixelLocation FindNearestPixel(const G4ThreeVector& pos) const;
    const std::vector<G4ThreeVector>& GetPixelCenters() const { return fPixelCenters; }
    const G4ThreeVector& GetPixelCenter(G4int globalPixelId) const;
    // Noise sigma accessors (row-major global pixel id = i*N + j)
    G4double GetPixelGainSigma(G4int globalPixelId) const { return (globalPixelId >= 0 && globalPixelId < (G4int)fPixelGainSigmas.size()) ? fPixelGainSigmas[globalPixelId] : 0.0; }
    const std::vector<G4double>& GetPixelGainSigmas() const { return fPixelGainSigmas; }
    
    
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
    void SyncRunMetadata();
    void SetupMessenger();

    G4double fPixelSize;
    G4double fPixelWidth;
    G4double fPixelSpacing;
    G4double fGridOffset;  ///< DD4hep-style grid offset (0 = centered grid)

    G4double fDetSize;
    G4double fDetWidth;
    G4int fNumBlocksPerSide;
    G4ThreeVector fDetectorPos;

    /// DD4hep-style index ranges (can be negative for centered grid)
    G4int fMinIndexX{0};
    G4int fMinIndexY{0};
    G4int fMaxIndexX{0};
    G4int fMaxIndexY{0};

    EventAction* fEventAction;
    RunAction* fRunAction{nullptr};
    
    G4int fNeighborhoodRadius;

    G4LogicalVolume* fLogicSilicon;

    // Per-pixel multiplicative noise sigma table (initialized once during construction)
    std::vector<G4double> fPixelGainSigmas;
    std::vector<G4ThreeVector> fPixelCenters;

    // UI messenger for runtime configuration
    std::unique_ptr<G4GenericMessenger> fMessenger;
};

} // namespace ECS

// Backward compatibility alias
using DetectorConstruction = ECS::DetectorConstruction;

#endif // ECS_DETECTOR_CONSTRUCTION_HH
