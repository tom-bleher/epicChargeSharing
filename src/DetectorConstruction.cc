/**
 * @file DetectorConstruction.cc
 * @brief Builds world, silicon detector, and pixel pads; attaches MFD and scorers.
 */
#include "DetectorConstruction.hh"

#include "Config.hh"
#include "EventAction.hh"
#include "RunAction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4LogicalVolume.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4UserLimits.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4VSensitiveDetector.hh"
#include "G4GenericMessenger.hh"
#include <G4ScoringManager.hh>

#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <limits>
#include <mutex>
#include <sstream>

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fPixelSize(Constants::PIXEL_SIZE),
      fPixelWidth(Constants::PIXEL_THICKNESS),
      fPixelSpacing(Constants::PIXEL_PITCH),
      fGridOffset(Constants::GRID_OFFSET),
      fDetSize(Constants::DETECTOR_SIZE),
      fDetWidth(Constants::DETECTOR_WIDTH),
      fNumBlocksPerSide(0),
      fDetectorPos(0., 0., Constants::DETECTOR_Z_POSITION),
      fMinIndexX(0),
      fMinIndexY(0),
      fMaxIndexX(0),
      fMaxIndexY(0),
      fEventAction(nullptr),
      fRunAction(nullptr),
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS),
      fLogicSilicon(nullptr)
{
    SetupMessenger();
}

DetectorConstruction::~DetectorConstruction() = default;

void DetectorConstruction::SetRunAction(RunAction* runAction)
{
    fRunAction = runAction;
    if (fRunAction) {
        SyncRunMetadata();
    }
}

void DetectorConstruction::SetGridOffset(G4double offset)
{
    G4cout << "[Detector] Grid offset set to " << offset / mm << " mm" << G4endl;
    fGridOffset = offset;

    if (G4RunManager* runManager = G4RunManager::GetRunManager()) {
        runManager->GeometryHasBeenModified();
    }
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    const G4bool checkOverlaps = true;
    const G4double originalDetSize = fDetSize;

    const MaterialSet materials = PrepareMaterials();

    G4LogicalVolume* logicWorld = nullptr;
    G4VPhysicalVolume* physWorld = BuildWorld(materials, checkOverlaps, logicWorld);

    G4LogicalVolume* siliconLogical =
        BuildSiliconDetector(logicWorld, materials, checkOverlaps, originalDetSize);

    const PixelGridStats gridStats =
        ConfigurePixels(logicWorld, siliconLogical, materials, checkOverlaps);

    SyncRunMetadata();

    static std::once_flag summaryFlag;
    std::call_once(summaryFlag, [&]() {
        const G4double coveragePercent = gridStats.coverage * 100.0;
        G4cout << "[Detector] Geometry prepared: size " << fDetSize / mm << " mm, pixels "
               << fNumBlocksPerSide << "x" << fNumBlocksPerSide << ", spacing "
               << fPixelSpacing / mm << " mm, coverage " << coveragePercent << "%" << G4endl;
    });

    return physWorld;
}

DetectorConstruction::MaterialSet DetectorConstruction::PrepareMaterials() const
{
    MaterialSet mats{};
    G4NistManager* nist = G4NistManager::Instance();
    mats.world = nist->FindOrBuildMaterial("G4_Galactic");
    mats.silicon = nist->FindOrBuildMaterial("G4_Si");
    mats.aluminum = nist->FindOrBuildMaterial("G4_Al");
    return mats;
}

G4VPhysicalVolume* DetectorConstruction::BuildWorld(const MaterialSet& mats,
                                                    G4bool checkOverlaps,
                                                    G4LogicalVolume*& logicWorld)
{
    auto* solidWorld =
        new G4Box("solidWorld", Constants::WORLD_SIZE, Constants::WORLD_SIZE, Constants::WORLD_SIZE);
    logicWorld = new G4LogicalVolume(solidWorld, mats.world, "logicWorld");

    auto* physWorld = new G4PVPlacement(
        nullptr,
        G4ThreeVector(0., 0., 0.),
        logicWorld,
        "physWorld",
        nullptr,
        false,
        0,
        checkOverlaps);

    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    return physWorld;
}

G4LogicalVolume* DetectorConstruction::BuildSiliconDetector(G4LogicalVolume* logicWorld,
                                                            const MaterialSet& mats,
                                                            G4bool checkOverlaps,
                                                            G4double originalDetSize)
{
    (void)originalDetSize;

    const G4ThreeVector& detectorPos = GetDetectorPos();

    auto* cubeVisAtt = new G4VisAttributes(G4Colour(0.7, 0.7, 0.7));
    cubeVisAtt->SetForceSolid(true);

    G4Box* detCube = new G4Box("detCube", fDetSize / 2, fDetSize / 2, fDetWidth / 2);
    auto* logicCube = new G4LogicalVolume(detCube, mats.silicon, "logicCube");
    logicCube->SetVisAttributes(cubeVisAtt);

    fLogicSilicon = logicCube;

    // DD4hep-style grid: compute index range based on detector size and pitch
    // Grid is centered at origin (detector center), indices can be negative
    //
    // For a detector of size D centered at 0, valid positions are [-D/2, +D/2]
    // Using DD4hep formula: index = floor((position + 0.5*pitch - offset) / pitch)
    //
    // Min index: position = -D/2 + pixelSize/2 (first pixel must fit inside detector)
    // Max index: position = +D/2 - pixelSize/2 (last pixel must fit inside detector)

    const G4double halfDet = fDetSize / 2.0;
    const G4double halfPad = fPixelSize / 2.0;

    // Compute index range using DD4hep formula
    // Ensure pixels fit within detector bounds
    fMinIndexX = Constants::PositionToIndex(-halfDet + halfPad, fPixelSpacing, fGridOffset);
    fMaxIndexX = Constants::PositionToIndex(+halfDet - halfPad, fPixelSpacing, fGridOffset);
    fMinIndexY = fMinIndexX;  // Square detector
    fMaxIndexY = fMaxIndexX;

    fNumBlocksPerSide = fMaxIndexX - fMinIndexX + 1;

    new G4PVPlacement(
        nullptr,
        detectorPos,
        logicCube,
        "physCube",
        logicWorld,
        false,
        0,
        checkOverlaps);

    return logicCube;
}

DetectorConstruction::PixelGridStats DetectorConstruction::ConfigurePixels(
    G4LogicalVolume* logicWorld,
    G4LogicalVolume* siliconLogical,
    const MaterialSet& mats,
    G4bool checkOverlaps)
{
    PixelGridStats stats{};

    auto* pixelBlock = new G4Box("pixelBlock", fPixelSize / 2, fPixelSize / 2, fPixelWidth / 2);
    auto* logicBlock = new G4LogicalVolume(pixelBlock, mats.aluminum, "logicBlock");

    auto* blockVisAtt = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0));
    blockVisAtt->SetForceSolid(true);
    logicBlock->SetVisAttributes(blockVisAtt);

    auto* stepLimit = new G4UserLimits(Constants::MAX_STEP_SIZE);
    siliconLogical->SetUserLimits(stepLimit);
    InitializePixelGainSigmas();

    const G4ThreeVector& detectorPos = GetDetectorPos();
    const G4double pixelZ = detectorPos.z() + fDetWidth / 2 + fPixelWidth / 2;

    // DD4hep-style grid: iterate over index range (can include negative indices)
    const G4int totalPixels = fNumBlocksPerSide * fNumBlocksPerSide;
    fPixelCenters.assign(static_cast<std::size_t>(std::max(0, totalPixels)),
                         G4ThreeVector(0., 0., 0.));

    G4int copyNo = 0;
    for (G4int i = fMinIndexX; i <= fMaxIndexX; ++i) {
        for (G4int j = fMinIndexY; j <= fMaxIndexY; ++j) {
            // DD4hep formula: position = index * pitch + offset
            const G4double pixelX = Constants::IndexToPosition(i, fPixelSpacing, fGridOffset);
            const G4double pixelY = Constants::IndexToPosition(j, fPixelSpacing, fGridOffset);

            // Convert from DD4hep index to flat array index
            // localI = i - fMinIndexX, localJ = j - fMinIndexY
            const G4int localI = i - fMinIndexX;
            const G4int localJ = j - fMinIndexY;
            const G4int globalId = localI * fNumBlocksPerSide + localJ;

            new G4PVPlacement(
                nullptr,
                G4ThreeVector(pixelX, pixelY, pixelZ),
                logicBlock,
                "physBlock",
                logicWorld,
                false,
                copyNo++,
                checkOverlaps);

            if (globalId >= 0 && globalId < totalPixels) {
                const auto idx = static_cast<std::size_t>(globalId);
                if (idx < fPixelCenters.size()) {
                    fPixelCenters[idx] = G4ThreeVector(pixelX, pixelY, pixelZ);
                }
            }
        }
    }

    if (G4VSensitiveDetector* pixelSensitiveDetector = logicBlock->GetSensitiveDetector()) {
        std::string message = "Aluminum pixels unexpectedly have sensitive detector '" +
                              pixelSensitiveDetector->GetName() + "' attached.";
        G4Exception("DetectorConstruction::ConfigurePixels",
                    "PixelSensitivityViolation",
                    FatalException,
                    message.c_str());
    }

    G4cout << "[Detector] Configured " << copyNo << " aluminum pixel pads (indices "
           << fMinIndexX << " to " << fMaxIndexX << ")" << G4endl;

    stats.totalPixelArea =
        static_cast<G4double>(fNumBlocksPerSide) * fNumBlocksPerSide * fPixelSize * fPixelSize;
    stats.detectorArea = fDetSize * fDetSize;
    stats.coverage = stats.totalPixelArea / stats.detectorArea;

    return stats;
}

void DetectorConstruction::InitializePixelGainSigmas()
{
    const G4int n = fNumBlocksPerSide;
    const G4int total = n * n;
    fPixelGainSigmas.clear();
    fPixelGainSigmas.reserve(total);
    const G4double minSigma = Constants::PIXEL_GAIN_SIGMA_MIN;
    const G4double maxSigma = Constants::PIXEL_GAIN_SIGMA_MAX;

    for (G4int idx = 0; idx < total; ++idx) {
        const G4double u = G4UniformRand();
        const G4double sigma = minSigma + (maxSigma - minSigma) * u;
        fPixelGainSigmas.push_back(sigma);
    }

}

void DetectorConstruction::SyncRunMetadata()
{
    if (!fRunAction) {
        return;
    }

    fRunAction->SetDetectorGridParameters(
        fPixelSize,
        fPixelSpacing,
        fGridOffset,
        fDetSize,
        fNumBlocksPerSide);
    fRunAction->SetGridPixelCenters(fPixelCenters);
    fRunAction->SetNeighborhoodRadiusMeta(fNeighborhoodRadius);

    const auto reconMethod = Constants::RECON_METHOD;
    G4double linearBeta = std::numeric_limits<G4double>::quiet_NaN();
    // Beta is only used when LinA signal model is active
    if (Constants::USES_LINEAR_SIGNAL) {
        linearBeta = GetLinearChargeModelBeta();
    }
    fRunAction->SetPosReconMetadata(reconMethod, linearBeta, fPixelSpacing);
}

DetectorConstruction::PixelLocation DetectorConstruction::FindNearestPixel(const G4ThreeVector& pos) const
{
    PixelLocation result{};

    const G4ThreeVector& detectorPos = GetDetectorPos();
    const G4ThreeVector relativePos = pos - detectorPos;

    // DD4hep formula: index = floor((position + 0.5*pitch - offset) / pitch)
    G4int i = Constants::PositionToIndex(relativePos.x(), fPixelSpacing, fGridOffset);
    G4int j = Constants::PositionToIndex(relativePos.y(), fPixelSpacing, fGridOffset);

    // Check if within valid index range
    result.withinDetector =
        (i >= fMinIndexX && i <= fMaxIndexX && j >= fMinIndexY && j <= fMaxIndexY);

    // Clamp to valid range
    i = std::max(fMinIndexX, std::min(i, fMaxIndexX));
    j = std::max(fMinIndexY, std::min(j, fMaxIndexY));

    // DD4hep formula: position = index * pitch + offset
    const G4double pixelX = Constants::IndexToPosition(i, fPixelSpacing, fGridOffset);
    const G4double pixelY = Constants::IndexToPosition(j, fPixelSpacing, fGridOffset);
    const G4double pixelZ = detectorPos.z() + fDetWidth / 2 + fPixelWidth / 2;

    result.center = {pixelX, pixelY, pixelZ};
    result.indexI = i;
    result.indexJ = j;
    return result;
}

const G4ThreeVector& DetectorConstruction::GetPixelCenter(G4int globalPixelId) const
{
    static const G4ThreeVector zero(0., 0., 0.);
    if (globalPixelId < 0) {
        return zero;
    }
    const auto idx = static_cast<std::size_t>(globalPixelId);
    if (idx >= fPixelCenters.size()) {
        return zero;
    }
    return fPixelCenters[idx];
}

void DetectorConstruction::ConstructSDandField()
{
    G4ScoringManager::GetScoringManager();

    auto* mfd = new G4MultiFunctionalDetector("SiliconDetector");
    G4SDManager::GetSDMpointer()->AddNewDetector(mfd);

    G4VPrimitiveScorer* energyScorer = new G4PSEnergyDeposit("EnergyDeposit");
    mfd->RegisterPrimitive(energyScorer);

    SetSensitiveDetector("logicCube", mfd);

    G4VSensitiveDetector* attachedDetector =
        fLogicSilicon ? fLogicSilicon->GetSensitiveDetector() : nullptr;
    if (attachedDetector != mfd) {
        G4Exception("DetectorConstruction::ConstructSDandField",
                    "SensitiveDetectorAttachmentFailed",
                    FatalException,
                    "Failed to attach Multi-Functional Detector to silicon volume.");
    }
}

void DetectorConstruction::SetNeighborhoodRadius(G4int radius)
{
    G4cout << "[Detector] Neighborhood radius set to " << radius << " ("
           << (2 * radius + 1) << "x" << (2 * radius + 1) << ")" << G4endl;
    fNeighborhoodRadius = radius;

    if (fEventAction) {
        fEventAction->SetNeighborhoodRadius(radius);
    }

    if (fRunAction) {
        fRunAction->SetNeighborhoodRadiusMeta(radius);
    }
}

void DetectorConstruction::SetPixelSize(G4double size)
{
    G4cout << "[Detector] Pixel size set to " << size / mm << " mm" << G4endl;
    fPixelSize = size;

    if (G4RunManager* runManager = G4RunManager::GetRunManager()) {
        runManager->GeometryHasBeenModified();
    }
}

void DetectorConstruction::SetPixelSpacing(G4double spacing)
{
    G4cout << "[Detector] Pixel spacing (pitch) set to " << spacing / mm << " mm" << G4endl;
    fPixelSpacing = spacing;

    if (G4RunManager* runManager = G4RunManager::GetRunManager()) {
        runManager->GeometryHasBeenModified();
    }
}

void DetectorConstruction::SetupMessenger()
{
    fMessenger = std::make_unique<G4GenericMessenger>(this, "/ecs/detector/",
                                                       "Detector geometry configuration");

    fMessenger->DeclareMethodWithUnit("pixelSize", "mm",
                                       &DetectorConstruction::SetPixelSize)
        .SetGuidance("Set the pixel pad size")
        .SetParameterName("size", false)
        .SetRange("size > 0")
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethodWithUnit("pixelSpacing", "mm",
                                       &DetectorConstruction::SetPixelSpacing)
        .SetGuidance("Set pixel center-to-center spacing (pitch)")
        .SetParameterName("spacing", false)
        .SetRange("spacing > 0")
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethodWithUnit("gridOffset", "mm",
                                       &DetectorConstruction::SetGridOffset)
        .SetGuidance("Set grid origin offset (DD4hep-style, 0 = centered)")
        .SetParameterName("offset", false)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethod("neighborhoodRadius",
                               &DetectorConstruction::SetNeighborhoodRadius)
        .SetGuidance("Set neighborhood radius for charge sharing (pixels)")
        .SetParameterName("radius", false)
        .SetRange("radius >= 0 && radius <= 10")
        .SetStates(G4State_PreInit, G4State_Idle);
}
