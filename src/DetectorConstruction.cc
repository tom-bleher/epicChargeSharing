#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "RunAction.hh"
#include "Constants.hh"
#include "G4RunManager.hh"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <string>
#include <filesystem>  // For portable directory creation

// Add step limiter includes
#include "G4UserLimits.hh"
#include "G4SDManager.hh"
#include <G4ScoringManager.hh>

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fPixelSize(Constants::PIXEL_SIZE),      // 100 microns - default value
      fPixelSpacing(Constants::PIXEL_SPACING),   // 500 microns - default value  
      fPixelCornerOffset(Constants::PIXEL_CORNER_OFFSET), // 100 microns - default value
      fDetSize(Constants::DETECTOR_SIZE),         // 30 mm - default value (may be adjusted)
      fDetWidth(Constants::DETECTOR_WIDTH),      // 50 microns thickness
      fPixelWidth(Constants::PIXEL_WIDTH),   // 1 micron thickness
      fNumBlocksPerSide(0),    // Will be calculated
      fCheckOverlaps(true),
      fEventAction(nullptr),   // Initialize EventAction pointer
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS)   // Default neighborhood radius for 9x9 grid
{
    // Values for pixel grid set at constants.hh
    
    // Will be calculated in Construct() based on symmetry constraint
    fNumBlocksPerSide = 0;
}

DetectorConstruction::~DetectorConstruction()
{
    

}



void DetectorConstruction::SetPixelCornerOffset(G4double cornerOffset)
{
    G4cout << "Setting pixel corner offset to: " << cornerOffset/um << " μm" << G4endl;
    G4cout << "Note: This is now a FIXED parameter - detector size will be adjusted if needed." << G4endl;
    fPixelCornerOffset = cornerOffset;
    
    // Trigger geometry reconstruction if we're already initialized
    G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager && runManager->GetRunManagerType() != G4RunManager::sequentialRM) {
        G4cout << "Requesting geometry update..." << G4endl;
        runManager->GeometryHasBeenModified();
    }
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4bool checkOverlaps = true;

    // Define materials
    G4NistManager *nist = G4NistManager::Instance();
    G4Material *worldMat = nist->FindOrBuildMaterial("G4_Galactic"); // World material
    G4Material *siliconMat = nist->FindOrBuildMaterial("G4_Si"); // Detector material
    G4Material *aluminumMat = nist->FindOrBuildMaterial("G4_Al"); // Pixel material
    
    // Create world volume
    G4Box *solidWorld = new G4Box("solidWorld", Constants::WORLD_SIZE, Constants::WORLD_SIZE, Constants::WORLD_SIZE);
    G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
    G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.),
                                                     logicWorld, "physWorld", 0, false, 0, checkOverlaps);

    // Create the main silicon detector
    G4Box* detCube = new G4Box("detCube", fDetSize/2, fDetSize/2, fDetWidth/2);
    // Store logical volume pointer for use in sensitive detector setup
    fLogicSilicon = new G4LogicalVolume(detCube, siliconMat, "logicCube");
    G4LogicalVolume* logicCube = fLogicSilicon; // alias for readability
    
    // Set visualization attributes for the detector (semi-transparent)
    // Note: G4 stores the pointer; keep ownership with Geant4 to ensure lifetime
    auto* cubeVisAtt = new G4VisAttributes(G4Colour(0.7, 0.7, 0.7, 0.5));
    logicCube->SetVisAttributes(cubeVisAtt);
    
    // Place the silicon detector at fixed pos
    G4ThreeVector detectorPos(0., 0., Constants::DETECTOR_Z_POSITION);
    
    // Store original detector size for comparison
    G4double originalDetSize = fDetSize;
    
    // Calc number of pixels that would fit with current parameters and FIXED corner offset
    fNumBlocksPerSide = static_cast<G4int>(std::round((fDetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1));
    
    // Calc the required detector size to accommodate the pixel grid with FIXED corner offset
    G4double requiredDetSize = 2*fPixelCornerOffset + fPixelSize + (fNumBlocksPerSide-1)*fPixelSpacing;
    
    // Update detector size if needed and notify user
    if (std::abs(requiredDetSize - fDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "\n=== AUTOMATIC DETECTOR SIZE ADJUSTMENT ===\n"
               << "Original detector size: " << originalDetSize/mm << " mm\n"
               << "Calcd pixel grid requires: " << fNumBlocksPerSide << "×" << fNumBlocksPerSide << " pixels\n"
               << "Required detector size: " << requiredDetSize/mm << " mm\n"
               << "Pixel corner offset (FIXED): " << fPixelCornerOffset/mm << " mm\n";
        
        // Update detector size
        fDetSize = requiredDetSize;
        
        G4cout << "✓ Detector size adjusted to: " << fDetSize/mm << " mm\n"
               << "==========================================" << G4endl;
        
        // Recreate the detector with the correct size
        delete detCube;
        detCube = new G4Box("detCube", fDetSize/2, fDetSize/2, fDetWidth/2);
        delete logicCube;
        logicCube = new G4LogicalVolume(detCube, siliconMat, "logicCube");
        logicCube->SetVisAttributes(cubeVisAtt);
        fLogicSilicon = logicCube; // keep pointer up-to-date
    }
    
    // Verify the corner offset calculation
    G4double actualCornerOffset = (fDetSize - (fNumBlocksPerSide-1)*fPixelSpacing - fPixelSize)/2;
    if (std::abs(actualCornerOffset - fPixelCornerOffset) > Constants::PRECISION_TOLERANCE) {
        G4cerr << "ERROR: Corner offset calculation failed!" << G4endl;
        G4cerr << "Expected: " << fPixelCornerOffset/mm << " mm, Got: " << actualCornerOffset/mm << " mm" << G4endl;
    }
    
    // Place the silicon detector at fixed pos (only once)
    new G4PVPlacement(0, detectorPos,
                      logicCube, "physCube", logicWorld, false, 0, checkOverlaps);
    
    // Create aluminum pixels on the detector surface
    G4Box *pixelBlock = new G4Box("pixelBlock", fPixelSize/2, fPixelSize/2, fPixelWidth/2);
    G4LogicalVolume *logicBlock = new G4LogicalVolume(pixelBlock, aluminumMat, "logicBlock");
    
    // Add step limiting for fine tracking - force steps to be at most 10 micrometers
    // Apply only to active/sensitive materials (silicon). Keep world and aluminum unrestricted.
    G4UserLimits* stepLimit = new G4UserLimits(Constants::MAX_STEP_SIZE);
    logicCube->SetUserLimits(stepLimit);
    
    G4cout << "✓ Step limiting enabled: maximum step size = 10 micrometers" << G4endl;
    
    // GEOMETRY UPDATE (2025-07): Pixel-pad aluminium must be the FIRST layer seen by the
    // incoming particle.  Therefore pixels are now placed on the FRONT face of the
    // silicon sensor (i.e. upstream along the particle direction).  SteppingAction
    // logic will mark events that traverse this metal layer before reaching the
    // silicon as aluminium-contaminated so that charge sharing is skipped.
    G4int copyNo = 0;
    G4double firstPixelPos = -fDetSize/2 + fPixelCornerOffset + fPixelSize/2;
    
    // Calculate Z position for the aluminium pads – FRONT face of the silicon
    // The primary particles start at +z and travel towards –z (momentum (0,0,-1)).
    // The silicon detector centre is at detectorPos.z(), so its front face is at
    //   detectorPos.z() + fDetWidth/2.  We add half the pad thickness to sit the
    // pads flush on top of that face.
    G4double pixelZ = detectorPos.z() + fDetWidth/2 + fPixelWidth/2;
    
    for (G4int i = 0; i < fNumBlocksPerSide; i++) {
        for (G4int j = 0; j < fNumBlocksPerSide; j++) {
            G4double pixelX = firstPixelPos + i * fPixelSpacing;
            G4double pixelY = firstPixelPos + j * fPixelSpacing;
            
            new G4PVPlacement(0, G4ThreeVector(pixelX, pixelY, pixelZ),
                             logicBlock, "physBlock", logicWorld, false, copyNo++, checkOverlaps);
        }
    }
    
    // Validate aluminum pixels remain passive (no sensitive detectors attached)
    G4VSensitiveDetector* pixelSensitiveDetector = logicBlock->GetSensitiveDetector();
    if (pixelSensitiveDetector == nullptr) {
        G4cout << "✓ Aluminum pixels confirmed passive - no sensitive detectors attached" << G4endl;
        G4cout << "✓ Total aluminum pixels placed: " << copyNo << " (" << fNumBlocksPerSide << "×" << fNumBlocksPerSide << ")" << G4endl;
    } else {
        G4cerr << "ERROR: Aluminum pixels have sensitive detector attached!" << G4endl;
        G4cerr << "This violates the selective sensitivity requirement!" << G4endl;
        G4cerr << "Attached detector: " << pixelSensitiveDetector->GetName() << G4endl;
    }
    
    // Set visualization attributes for pixels
    auto* blockVisAtt = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0));
    logicBlock->SetVisAttributes(blockVisAtt);
    
    // Set the world volume to be invisible
    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    
    // Calc and print the ratio of pixel area to detector area
    G4double totalPixelArea = fNumBlocksPerSide * fNumBlocksPerSide * fPixelSize * fPixelSize;
    G4double detectorArea = fDetSize * fDetSize;
    G4double pixelAreaRatio = totalPixelArea / detectorArea;
    
    G4cout << "\n=== FINAL DETECTOR CONFIGURATION ===\n"
           << "Detector Statistics:\n"
           << "  Final detector size: " << fDetSize/mm << " mm × " << fDetSize/mm << " mm\n";
    if (std::abs(fDetSize - originalDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "  (Adjusted from original: " << originalDetSize/mm << " mm)\n";
    }
    G4cout << "  Pixel corner offset (FIXED): " << fPixelCornerOffset/mm << " mm\n"
           << "  Total number of pixels: " << fNumBlocksPerSide * fNumBlocksPerSide << "\n"
           << "  Pixel grid: " << fNumBlocksPerSide << " × " << fNumBlocksPerSide << "\n"
           << "  Single pixel area: " << fPixelSize * fPixelSize / (mm*mm) << " mm²\n"
           << "  Total pixel area: " << totalPixelArea / (mm*mm) << " mm²\n"
           << "  Detector area: " << detectorArea / (mm*mm) << " mm²\n"
           << "  Pixel area / Detector area ratio: " << pixelAreaRatio << "\n"
           << "====================================" << G4endl;

    // Save all simulation parameters to a log file
    SaveSimulationParameters(totalPixelArea, detectorArea, pixelAreaRatio);

    // Update RunAction with the final grid parameters after geometry construction
    // This ensures the ROOT metadata contains the actual values used for pixel placement
    G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager) {
        RunAction* runAction = (RunAction*)runManager->GetUserRunAction();
        if (runAction) {
            runAction->SetDetectorGridParameters(
                fPixelSize,
                fPixelSpacing, 
                fPixelCornerOffset,  // This is now FIXED (not adjusted)
                fDetSize,            // This may have been adjusted
                fNumBlocksPerSide    // This is calculated based on final parameters
            );
            G4cout << "Updated RunAction with final grid parameters:" << G4endl;
            G4cout << "  Final Detector Size: " << fDetSize/mm << " mm" << G4endl;
            G4cout << "  Fixed Pixel Corner Offset: " << fPixelCornerOffset/mm << " mm" << G4endl;
            G4cout << "  Final Number of Blocks per Side: " << fNumBlocksPerSide << G4endl;
        }
    }

    return physWorld;
}

G4ThreeVector DetectorConstruction::GetDetectorPos() const
{
    return G4ThreeVector(0., 0., Constants::DETECTOR_Z_POSITION);
}

// Implementation of SaveSimulationParameters method
void DetectorConstruction::SaveSimulationParameters(G4double totalPixelArea, G4double detectorArea, G4double pixelAreaRatio) const
{
    // Get current time for file naming and log entry
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    
    // Create logs directory using portable std::filesystem
    std::filesystem::path logsDir = std::filesystem::current_path() / "logs";
    G4cout << "Creating logs directory at: " << logsDir << G4endl;
    
    std::error_code ec;
    if (!std::filesystem::create_directories(logsDir, ec)) {
        if (ec) {
            G4cerr << "Warning: Could not create logs directory: " << ec.message() << G4endl;
        }
        // Directory may already exist, which is fine
    }
    
    // Create filename with timestamp in logs directory
    std::filesystem::path filename = logsDir / ("simulation_params_" + std::string(timestamp) + ".log");
    
    // Open file for writing
    std::ofstream paramFile(filename);
    
    if (paramFile.is_open()) {
        // Write header with timestamp in human-readable format
        char dateStr[100];
        std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        
        paramFile << "=========================================================" << std::endl;
        paramFile << "EPIC TOY SIMULATION PARAMETERS" << std::endl;
        paramFile << "Generated on: " << dateStr << std::endl;
        paramFile << "=========================================================" << std::endl << std::endl;
        
        // Write detector parameters
        paramFile << "DETECTOR PARAMETERS" << std::endl;
        paramFile << "-----------------" << std::endl;
        paramFile << "Detector Size: " << fDetSize/mm << " mm" << std::endl;
        paramFile << "Detector Width/Thickness: " << fDetWidth/mm << " mm" << std::endl;
        paramFile << "Detector Area: " << detectorArea/(mm*mm) << " mm²" << std::endl << std::endl;
        
        // Write pixel parameters 
        paramFile << "PIXEL PARAMETERS" << std::endl;
        paramFile << "---------------" << std::endl;
        paramFile << "Pixel Size: " << fPixelSize/mm << " mm" << std::endl;
        paramFile << "Pixel Width/Thickness: " << fPixelWidth/mm << " mm" << std::endl;
        paramFile << "Pixel Spacing (center-to-center): " << fPixelSpacing/mm << " mm" << std::endl;
        paramFile << "Pixel Corner Offset: " << fPixelCornerOffset/mm << " mm" << std::endl;
        paramFile << "Number of Pixels per Side: " << fNumBlocksPerSide << std::endl;
        paramFile << "Total Number of Pixels: " << fNumBlocksPerSide * fNumBlocksPerSide << std::endl;
        paramFile << "Single Pixel Area: " << (fPixelSize * fPixelSize)/(mm*mm) << " mm²" << std::endl;
        paramFile << "Total Pixel Area: " << totalPixelArea/(mm*mm) << " mm²" << std::endl << std::endl;
        
        // Write detector statistics
        paramFile << "DETECTOR STATISTICS" << std::endl;
        paramFile << "------------------" << std::endl;
        paramFile << "Pixel Area / Detector Area Ratio: " << pixelAreaRatio << std::endl;
        paramFile << "Pixel Coverage Percentage: " << pixelAreaRatio * 100.0 << " %" << std::endl;
        paramFile << "Pixel Area Fraction: " << pixelAreaRatio << std::endl;
        
        // Write footer
        paramFile << std::endl;
        paramFile << "=========================================================" << std::endl;
        
        // Close file
        paramFile.close();
        
        G4cout << "Simulation parameters saved to: " << filename << G4endl;
    } else {
        G4cerr << "ERROR: Could not open file for saving simulation parameters: " << filename << G4endl;
    }
}

void DetectorConstruction::SetNeighborhoodRadius(G4int radius)
{
    G4cout << "Setting neighborhood radius to: " << radius << G4endl;
    G4cout << "This corresponds to a " << (2*radius + 1) << "x" << (2*radius + 1) << " grid" << G4endl;
    
    // Store the radius in DetectorConstruction
    fNeighborhoodRadius = radius;
    
    // Pass the radius to EventAction if it's available
    if (fEventAction) {
        fEventAction->SetNeighborhoodRadius(radius);
        G4cout << "Updated EventAction with new neighborhood radius: " << radius << G4endl;
    } else {
        G4cout << "EventAction not yet available - radius will be set when EventAction is connected" << G4endl;
    }
}



void DetectorConstruction::ConstructSDandField() {
    G4ScoringManager::GetScoringManager();  // Activate scoring manager

    // Create and register MultiFunctionalDetector with consistent name
    G4MultiFunctionalDetector* mfd = new G4MultiFunctionalDetector("SiliconDetector");
    G4SDManager::GetSDMpointer()->AddNewDetector(mfd);

    // Attach scorers
    G4VPrimitiveScorer* energyScorer = new G4PSEnergyDeposit("EnergyDeposit");
    mfd->RegisterPrimitive(energyScorer);

    G4VPrimitiveScorer* hitCountScorer = new G4PSNofStep("HitCount");
    mfd->RegisterPrimitive(hitCountScorer);

    // Attach to silicon volume only
    SetSensitiveDetector("logicCube", mfd);

    // Validate attachment
    G4VSensitiveDetector* attachedDetector = fLogicSilicon->GetSensitiveDetector();
    if (attachedDetector == mfd) {
        G4cout << "✓ Multi-Functional Detector 'SiliconDetector' successfully attached to silicon volume" << G4endl;
    } else {
        G4cerr << "ERROR: Failed to attach Multi-Functional Detector to silicon volume" << G4endl;
    }

    G4cout << "\n=== MULTI-FUNCTIONAL DETECTOR VALIDATION SUMMARY ===" << G4endl;
    G4cout << "✓ Multi-Functional Detector: Created and initialized" << G4endl;
    G4cout << "✓ Silicon Volume Sensitivity: Attached (logicCube)" << G4endl;
    G4cout << "✓ Primitive Scorers: " << mfd->GetNumberOfPrimitives() << " attached" << G4endl;
    G4cout << "✓ Selective Sensitivity: Requirements met" << G4endl;
    G4cout << "=================================================" << G4endl;
}
