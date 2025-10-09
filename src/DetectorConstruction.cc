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

#include <fstream>
#include <cmath>
#include <ctime>
#include <string>
#include <filesystem>

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
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS),
      fLogicSilicon(nullptr)
{
}

DetectorConstruction::~DetectorConstruction() = default;

void DetectorConstruction::SetPixelCornerOffset(G4double cornerOffset)
{
    G4cout << "Setting pixel corner offset to: " << cornerOffset / um << " um" << G4endl;
    G4cout << "Note: This is now a fixed parameter - detector size will be adjusted if needed."
           << G4endl;
    fPixelCornerOffset = cornerOffset;

    if (G4RunManager* runManager = G4RunManager::GetRunManager()) {
        G4cout << "Requesting geometry update..." << G4endl;
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

    WriteGeometryLog(gridStats, originalDetSize);
    SaveSimulationParameters(gridStats.totalPixelArea,
                             gridStats.detectorArea,
                             gridStats.coverage);
    SyncRunMetadata();

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
        G4cout << "\n=== AUTOMATIC DETECTOR SIZE ADJUSTMENT ===\n"
               << "Original detector size: " << originalDetSize / mm << " mm\n"
               << "Calculated pixel grid requires: " << fNumBlocksPerSide << " x "
               << fNumBlocksPerSide << " pixels\n"
               << "Required detector size: " << requiredDetSize / mm << " mm\n"
               << "Pixel corner offset (fixed): " << fPixelCornerOffset / mm << " mm\n";

        fDetSize = requiredDetSize;

        G4cout << "-> Detector size adjusted to: " << fDetSize / mm << " mm\n"
               << "===========================================\n"
               << G4endl;

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
        G4cerr << "ERROR: Corner offset calculation failed!" << G4endl;
        G4cerr << "Expected: " << fPixelCornerOffset / mm << " mm, "
               << "Got: " << actualCornerOffset / mm << " mm" << G4endl;
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
    G4cout << "Step limiting: max step " << Constants::MAX_STEP_SIZE / um << " um" << G4endl;

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
        G4cerr << "ERROR: Aluminum pixels have sensitive detector attached!" << G4endl;
        G4cerr << "This violates the selective sensitivity requirement!" << G4endl;
        G4cerr << "Attached detector: " << pixelSensitiveDetector->GetName() << G4endl;
    } else {
        G4cout << "Aluminum pixel pads passive (no sensitive detector attached)" << G4endl;
        G4cout << "Pixel pad count: " << copyNo << " (" << fNumBlocksPerSide << " x "
               << fNumBlocksPerSide << ")" << G4endl;
    }

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

    G4cout << "Initialized per-pixel gain sigmas in [" << minSigma << ", " << maxSigma << "] with "
           << total << " entries" << G4endl;
}

void DetectorConstruction::WriteGeometryLog(const PixelGridStats& stats,
                                            G4double originalDetSize) const
{
    G4cout << "\nDetector configuration" << G4endl
           << "  size: " << fDetSize / mm << " mm x " << fDetSize / mm << " mm" << G4endl;
    if (std::abs(fDetSize - originalDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "  (adjusted from original " << originalDetSize / mm << " mm)" << G4endl;
    }
    G4cout << "  pixel corner offset (fixed): " << fPixelCornerOffset / mm << " mm" << G4endl
           << "  pixels: " << fNumBlocksPerSide << " x " << fNumBlocksPerSide << " ("
           << fNumBlocksPerSide * fNumBlocksPerSide << ")" << G4endl
           << "  pixel area (single): " << fPixelSize * fPixelSize / (mm * mm) << " mm^2"
           << G4endl
           << "  pixel area (total):  " << stats.totalPixelArea / (mm * mm) << " mm^2" << G4endl
           << "  detector area:       " << stats.detectorArea / (mm * mm) << " mm^2" << G4endl
           << "  coverage:            " << stats.coverage << G4endl;
}

void DetectorConstruction::SyncRunMetadata()
{
    if (auto* runManager = G4RunManager::GetRunManager()) {
        if (const auto* userRunAction = runManager->GetUserRunAction()) {
            if (auto* runAction = const_cast<RunAction*>(dynamic_cast<const RunAction*>(userRunAction))) {
                runAction->SetDetectorGridParameters(
                    fPixelSize,
                    fPixelSpacing,
                    fPixelCornerOffset,
                    fDetSize,
                    fNumBlocksPerSide);
                runAction->SetNeighborhoodRadiusMeta(fNeighborhoodRadius);

                G4cout << "Updated RunAction with final grid parameters:" << G4endl
                       << "  Final detector size: " << fDetSize / mm << " mm" << G4endl
                       << "  Fixed pixel corner offset: " << fPixelCornerOffset / mm << " mm" << G4endl
                       << "  Final number of blocks per side: " << fNumBlocksPerSide << G4endl;
            }
        }
    }
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

void DetectorConstruction::SaveSimulationParameters(G4double totalPixelArea,
                                                    G4double detectorArea,
                                                    G4double pixelAreaRatio) const
{
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));

    std::filesystem::path logsDir = std::filesystem::current_path() / "logs";
    std::error_code ec;
    if (!std::filesystem::create_directories(logsDir, ec) && ec) {
        G4cerr << "Warning: Could not create logs directory: " << ec.message() << G4endl;
    }

    std::filesystem::path filename =
        logsDir / ("simulation_params_" + std::string(timestamp) + ".log");

    std::ofstream paramFile(filename);
    if (!paramFile.is_open()) {
        G4cerr << "ERROR: Could not open file for saving simulation parameters: " << filename
               << G4endl;
        return;
    }

    char dateStr[100];
    std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    paramFile << "=========================================================" << std::endl;
    paramFile << "SIMULATION PARAMETERS" << std::endl;
    paramFile << "Generated on: " << dateStr << std::endl;
    paramFile << "=========================================================" << std::endl
              << std::endl;

    paramFile << "DETECTOR PARAMETERS" << std::endl;
    paramFile << "-----------------" << std::endl;
    paramFile << "Detector Size: " << fDetSize / mm << " mm" << std::endl;
    paramFile << "Detector Width/Thickness: " << fDetWidth / mm << " mm" << std::endl;
    paramFile << "Detector Area: " << detectorArea / (mm * mm) << " mm^2" << std::endl << std::endl;

    paramFile << "PIXEL PARAMETERS" << std::endl;
    paramFile << "---------------" << std::endl;
    paramFile << "Pixel Size: " << fPixelSize / mm << " mm" << std::endl;
    paramFile << "Pixel Width/Thickness: " << fPixelWidth / mm << " mm" << std::endl;
    paramFile << "Pixel Spacing (center-to-center): " << fPixelSpacing / mm << " mm" << std::endl;
    paramFile << "Pixel Corner Offset: " << fPixelCornerOffset / mm << " mm" << std::endl;
    paramFile << "Number of Pixels per Side: " << fNumBlocksPerSide << std::endl;
    paramFile << "Total Number of Pixels: " << fNumBlocksPerSide * fNumBlocksPerSide << std::endl;
    paramFile << "Single Pixel Area: " << (fPixelSize * fPixelSize) / (mm * mm) << " mm^2"
              << std::endl;
    paramFile << "Total Pixel Area: " << totalPixelArea / (mm * mm) << " mm^2" << std::endl
              << std::endl;

    paramFile << "DETECTOR STATISTICS" << std::endl;
    paramFile << "------------------" << std::endl;
    paramFile << "Pixel Area / Detector Area Ratio: " << pixelAreaRatio << std::endl;
    paramFile << "Pixel Coverage Percentage: " << pixelAreaRatio * 100.0 << " %" << std::endl;
    paramFile << "Pixel Area Fraction: " << pixelAreaRatio << std::endl;
    paramFile << std::endl;
    paramFile << "=========================================================" << std::endl;

    G4cout << "Simulation parameters saved to: " << filename << G4endl;
}

void DetectorConstruction::SetNeighborhoodRadius(G4int radius)
{
    G4cout << "Setting neighborhood radius to: " << radius << G4endl;
    G4cout << "This corresponds to a " << (2 * radius + 1) << "x" << (2 * radius + 1) << " grid"
           << G4endl;

    fNeighborhoodRadius = radius;

    if (fEventAction) {
        fEventAction->SetNeighborhoodRadius(radius);
        G4cout << "Updated EventAction with new neighborhood radius: " << radius << G4endl;
    } else {
        G4cout << "EventAction not yet available - radius will be set when EventAction is connected"
               << G4endl;
    }
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
    if (attachedDetector == mfd) {
        G4cout << "Multi-Functional Detector 'SiliconDetector' successfully attached to silicon "
                  "volume"
               << G4endl;
    } else {
        G4cerr << "ERROR: Failed to attach Multi-Functional Detector to silicon volume" << G4endl;
    }

    G4cout << "MFD: created, attached to silicon, primitives: " << mfd->GetNumberOfPrimitives()
           << G4endl;
}
