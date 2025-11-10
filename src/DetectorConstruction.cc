/**
 * @file DetectorConstruction.cc
 * @brief Builds world, silicon detector, and pixel pads; attaches MFD and scorers.
 */
#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "RunAction.hh"
#include "Constants.hh"

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
#include <G4ScoringManager.hh>

#include "Randomize.hh"

#include <cmath>
#include <string>
#include <limits>
#include <mutex>
#include <sstream>

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fPixelSize(Constants::PIXEL_SIZE),
      fPixelWidth(Constants::PIXEL_WIDTH),
      fPixelSpacing(Constants::PIXEL_SPACING),
      fPixelCornerOffset(Constants::PIXEL_CORNER_OFFSET),
      fDetSize(Constants::DETECTOR_SIZE),
      fDetWidth(Constants::DETECTOR_WIDTH),
      fNumBlocksPerSide(0),
      fEventAction(nullptr),
      fRunAction(nullptr),
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS),
      fLogicSilicon(nullptr)
{
}

DetectorConstruction::~DetectorConstruction() = default;

void DetectorConstruction::SetRunAction(RunAction* runAction)
{
    fRunAction = runAction;
    if (fRunAction) {
        SyncRunMetadata();
    }
}

void DetectorConstruction::SetPixelCornerOffset(G4double cornerOffset)
{
    G4cout << "[Detector] Pixel corner offset set to " << cornerOffset / mm << " mm" << G4endl;
    fPixelCornerOffset = cornerOffset;

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

    const G4ThreeVector detectorPos = GetDetectorPos();

    auto* cubeVisAtt = new G4VisAttributes(G4Colour(0.7, 0.7, 0.7));
    cubeVisAtt->SetForceSolid(true);

    G4Box* detCube = new G4Box("detCube", fDetSize / 2, fDetSize / 2, fDetWidth / 2);
    auto* logicCube = new G4LogicalVolume(detCube, mats.silicon, "logicCube");
    logicCube->SetVisAttributes(cubeVisAtt);

    fLogicSilicon = logicCube;

    fNumBlocksPerSide = static_cast<G4int>(
        std::round((fDetSize - 2 * fPixelCornerOffset - fPixelSize) / fPixelSpacing + 1));

    const G4double requiredDetSize =
        2 * fPixelCornerOffset + fPixelSize + (fNumBlocksPerSide - 1) * fPixelSpacing;

    if (std::abs(requiredDetSize - fDetSize) > Constants::GEOMETRY_TOLERANCE) {
        fDetSize = requiredDetSize;

        delete detCube;
        detCube = new G4Box("detCube", fDetSize / 2, fDetSize / 2, fDetWidth / 2);
        delete logicCube;
        logicCube = new G4LogicalVolume(detCube, mats.silicon, "logicCube");
        logicCube->SetVisAttributes(cubeVisAtt);
        fLogicSilicon = logicCube;
    }

    const G4double actualCornerOffset =
        (fDetSize - (fNumBlocksPerSide - 1) * fPixelSpacing - fPixelSize) / 2;
    if (std::abs(actualCornerOffset - fPixelCornerOffset) > Constants::PRECISION_TOLERANCE) {
        std::ostringstream oss;
        oss << "Corner offset calculation failed. Expected " << fPixelCornerOffset / mm
            << " mm but got " << actualCornerOffset / mm << " mm.";
        G4Exception("DetectorConstruction::BuildSiliconDetector",
                    "CornerOffsetMismatch",
                    FatalException,
                    oss.str().c_str());
    }

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

    const G4ThreeVector detectorPos = GetDetectorPos();
    const G4double pixelZ = detectorPos.z() + fDetWidth / 2 + fPixelWidth / 2;
    const G4double firstPixelPos = -fDetSize / 2 + fPixelCornerOffset + fPixelSize / 2;

    G4int copyNo = 0;
    for (G4int i = 0; i < fNumBlocksPerSide; ++i) {
        for (G4int j = 0; j < fNumBlocksPerSide; ++j) {
            const G4double pixelX = firstPixelPos + i * fPixelSpacing;
            const G4double pixelY = firstPixelPos + j * fPixelSpacing;

            new G4PVPlacement(
                nullptr,
                G4ThreeVector(pixelX, pixelY, pixelZ),
                logicBlock,
                "physBlock",
                logicWorld,
                false,
                copyNo++,
                checkOverlaps);
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

    G4cout << "[Detector] Configured " << copyNo << " aluminum pixel pads" << G4endl;

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
        fPixelCornerOffset,
        fDetSize,
        fNumBlocksPerSide);
    fRunAction->SetNeighborhoodRadiusMeta(fNeighborhoodRadius);

    const auto chargeModel = Constants::CHARGE_SHARING_MODEL;
    G4double linearBeta = std::numeric_limits<G4double>::quiet_NaN();
    if (chargeModel == Constants::ChargeSharingModel::Linear) {
        linearBeta = GetLinearChargeModelBeta(fPixelSpacing);
    }
    fRunAction->SetChargeSharingMetadata(chargeModel, linearBeta, fPixelSpacing);
}

G4ThreeVector DetectorConstruction::GetDetectorPos() const
{
    return G4ThreeVector(0., 0., Constants::DETECTOR_Z_POSITION);
}

DetectorConstruction::PixelLocation DetectorConstruction::FindNearestPixel(const G4ThreeVector& pos) const
{
    PixelLocation result{};

    const G4ThreeVector detectorPos = GetDetectorPos();
    const G4ThreeVector relativePos = pos - detectorPos;
    const G4double firstPixelPos = -fDetSize / 2 + fPixelCornerOffset + fPixelSize / 2;

    G4int i = static_cast<G4int>(std::round((relativePos.x() - firstPixelPos) / fPixelSpacing));
    G4int j = static_cast<G4int>(std::round((relativePos.y() - firstPixelPos) / fPixelSpacing));

    result.withinDetector =
        (i >= 0 && i < fNumBlocksPerSide && j >= 0 && j < fNumBlocksPerSide);

    i = std::max(0, std::min(i, fNumBlocksPerSide - 1));
    j = std::max(0, std::min(j, fNumBlocksPerSide - 1));

    const G4double pixelX = firstPixelPos + i * fPixelSpacing;
    const G4double pixelY = firstPixelPos + j * fPixelSpacing;
    const G4double pixelZ = detectorPos.z() + fDetWidth / 2 + fPixelWidth / 2;

    result.center = {pixelX, pixelY, pixelZ};
    result.indexI = i;
    result.indexJ = j;
    return result;
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

G4double DetectorConstruction::GetLinearChargeModelBeta(G4double pitch) const
{
    if (pitch >= Constants::LINEAR_CHARGE_MODEL_MIN_PITCH &&
        pitch <= Constants::LINEAR_CHARGE_MODEL_BOUNDARY_PITCH) {
        return Constants::LINEAR_CHARGE_MODEL_BETA_NARROW;
    }

    if (pitch > Constants::LINEAR_CHARGE_MODEL_BOUNDARY_PITCH &&
        pitch <= Constants::LINEAR_CHARGE_MODEL_MAX_PITCH) {
        return Constants::LINEAR_CHARGE_MODEL_BETA_WIDE;
    }

    std::call_once(fLinearModelWarningFlag,
                   [pitch]() {
                       std::ostringstream oss;
                       oss << "Linear charge sharing model: pixel pitch " << pitch / micrometer
                           << " um outside supported range ["
                           << Constants::LINEAR_CHARGE_MODEL_MIN_PITCH / micrometer
                           << ", "
                           << Constants::LINEAR_CHARGE_MODEL_MAX_PITCH / micrometer
                           << "]. Using narrow beta.";
                       const std::string message = oss.str();
                       G4Exception("DetectorConstruction::GetLinearChargeModelBeta",
                                   "LinearModelOutOfRange",
                                   JustWarning,
                                   message.c_str());
                   });

    return Constants::LINEAR_CHARGE_MODEL_BETA_NARROW;
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
