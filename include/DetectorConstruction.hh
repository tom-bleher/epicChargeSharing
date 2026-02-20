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

    [[nodiscard]] G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }

    [[nodiscard]] G4double GetPixelSize() const { return fPixelSize; }
    [[nodiscard]] G4double GetPixelSpacing() const { return fPixelSpacing; }
    [[nodiscard]] G4double GetPixelPitch() const { return fPixelSpacing; }
    [[nodiscard]] G4double GetLinearChargeModelBeta() const { return fLinearChargeModelBeta; }
    [[nodiscard]] G4double GetGridOffset() const { return fGridOffset; }
    [[nodiscard]] G4double GetDetSize() const { return fDetSize; }
    [[nodiscard]] G4int GetNumBlocksPerSide() const { return fNumBlocksPerSide; }

    /// @brief Get minimum pixel index (DD4hep-style, can be negative for centered grid).
    [[nodiscard]] G4int GetMinIndexX() const { return fMinIndexX; }
    [[nodiscard]] G4int GetMinIndexY() const { return fMinIndexY; }
    [[nodiscard]] G4int GetMaxIndexX() const { return fMaxIndexX; }
    [[nodiscard]] G4int GetMaxIndexY() const { return fMaxIndexY; }
    [[nodiscard]] const G4ThreeVector& GetDetectorPos() const { return fDetectorPos; }
    [[nodiscard]] PixelLocation FindNearestPixel(const G4ThreeVector& pos) const;
    [[nodiscard]] const std::vector<G4ThreeVector>& GetPixelCenters() const { return fPixelCenters; }
    [[nodiscard]] const G4ThreeVector& GetPixelCenter(G4int globalPixelId) const;
    // Noise sigma accessors (row-major global pixel id = i*N + j)
    [[nodiscard]] G4double GetPixelGainSigma(G4int globalPixelId) const { return (globalPixelId >= 0 && globalPixelId < (G4int)fPixelGainSigmas.size()) ? fPixelGainSigmas[globalPixelId] : 0.0; }
    [[nodiscard]] const std::vector<G4double>& GetPixelGainSigmas() const { return fPixelGainSigmas; }
    
    
private:
    struct MaterialSet {
        G4Material* world{nullptr};
        G4Material* silicon{nullptr};
        G4Material* aluminum{nullptr};
    };

    struct PixelGridStats {
        G4double totalPixelArea{0.0};
        G4double detectorArea{0.0};
        G4double coverage{0.0};
    };

    [[nodiscard]] MaterialSet PrepareMaterials() const;
    G4VPhysicalVolume* BuildWorld(const MaterialSet& mats, G4bool checkOverlaps, G4LogicalVolume*& logicWorld);
    G4LogicalVolume* BuildSiliconDetector(G4LogicalVolume* logicWorld, const MaterialSet& mats, G4bool checkOverlaps, G4double originalDetSize);
    PixelGridStats ConfigurePixels(G4LogicalVolume* logicWorld, G4LogicalVolume* siliconLogical, const MaterialSet& mats, G4bool checkOverlaps);
    void InitializePixelGainSigmas();
    void SyncRunMetadata();
    void SetupMessenger();

    G4double fPixelSize{0.0};
    G4double fPixelWidth{0.0};
    G4double fPixelSpacing{0.0};
    G4double fGridOffset{0.0};  ///< DD4hep-style grid offset (0 = centered grid)

    G4double fLinearChargeModelBeta{Constants::LINEAR_CHARGE_MODEL_BETA};

    G4double fDetSize{0.0};
    G4double fDetWidth{0.0};
    G4int fNumBlocksPerSide{0};
    G4ThreeVector fDetectorPos;

    /// DD4hep-style index ranges (can be negative for centered grid)
    G4int fMinIndexX{0};
    G4int fMinIndexY{0};
    G4int fMaxIndexX{0};
    G4int fMaxIndexY{0};

    EventAction* fEventAction{nullptr};
    RunAction* fRunAction{nullptr};
    
    G4int fNeighborhoodRadius{Constants::NEIGHBORHOOD_RADIUS};

    G4LogicalVolume* fLogicSilicon{nullptr};

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
