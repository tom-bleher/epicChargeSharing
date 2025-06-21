#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
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
#include <memory>      // For std::unique_ptr

// Add step limiter includes
#include "G4UserLimits.hh"

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fPixelSize(Constants::DEFAULT_PIXEL_SIZE),      // 100 microns - default value
      fPixelSpacing(Constants::DEFAULT_PIXEL_SPACING),   // 500 microns - default value  
      fPixelCornerOffset(Constants::DEFAULT_PIXEL_CORNER_OFFSET), // 100 microns - default value
      fDetSize(Constants::DEFAULT_DETECTOR_SIZE),         // 30 mm - default value (may be adjusted)
      fDetWidth(Constants::DEFAULT_DETECTOR_WIDTH),      // 50 microns thickness
      fPixelWidth(Constants::DEFAULT_PIXEL_WIDTH),   // 1 micron thickness
      fNumBlocksPerSide(0),    // Will be calculated
      fCheckOverlaps(true),
      fEventAction(nullptr),   // Initialize EventAction pointer
      fDetectorMessenger(nullptr),
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS)   // Default neighborhood radius for 9x9 grid
{
    // Values for pixel grid set at constants.hh
    
    // Will be calculated in Construct() based on symmetry constraint
    fNumBlocksPerSide = 0;
    
    // Create the messenger
    fDetectorMessenger = new DetectorMessenger(this);
}

DetectorConstruction::~DetectorConstruction()
{
    delete fDetectorMessenger;
}

void DetectorConstruction::SetGridParameters(G4double pixelSize, G4double pixelSpacing, G4double pixelCornerOffset, G4int numPixels)
{
    // Update the parameters for pixel grid placement
    fPixelSize = pixelSize;
    fPixelSpacing = pixelSpacing;
    // NOTE: fPixelCornerOffset is now FIXED and not changed by this method
    
    // Store the original detector size for comparison
    G4double originalDetSize = fDetSize;
    
    // Calculate the number of pixels that would fit with current parameters
    fNumBlocksPerSide = static_cast<G4int>(std::round((fDetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1));
    
    // Calculate the required detector size to accommodate the pixel grid with FIXED corner offset
    G4double requiredDetSize = 2*fPixelCornerOffset + fPixelSize + (fNumBlocksPerSide-1)*fPixelSpacing;
    
    // Update detector size if it differs significantly (more than 1 μm tolerance)
    if (std::abs(requiredDetSize - fDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "\n=== DETECTOR SIZE ADJUSTMENT ===" << G4endl;
        G4cout << "Original detector size: " << originalDetSize/mm << " mm" << G4endl;
        G4cout << "Required detector size for " << fNumBlocksPerSide << "×" << fNumBlocksPerSide 
               << " pixel grid: " << requiredDetSize/mm << " mm" << G4endl;
        G4cout << "Pixel corner offset (FIXED): " << fPixelCornerOffset/mm << " mm" << G4endl;
        
        fDetSize = requiredDetSize;
        
        G4cout << "Detector size adjusted to: " << fDetSize/mm << " mm" << G4endl;
        G4cout << "=================================" << G4endl;
    }
    
    // Verify the calculation
    G4double actualCornerOffset = (fDetSize - (fNumBlocksPerSide-1)*fPixelSpacing - fPixelSize)/2;
    if (std::abs(actualCornerOffset - fPixelCornerOffset) > Constants::PRECISION_TOLERANCE) {
        G4cerr << "WARNING: Corner offset calculation mismatch!" << G4endl;
        G4cerr << "Expected: " << fPixelCornerOffset/mm << " mm, Got: " << actualCornerOffset/mm << " mm" << G4endl;
    }
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
    G4LogicalVolume* logicCube = new G4LogicalVolume(detCube, siliconMat, "logicCube");
    
    // Set visualization attributes for the detector (semi-transparent) - RAII
    auto cubeVisAtt = std::make_unique<G4VisAttributes>(G4Colour(0.7, 0.7, 0.7, 0.5)); // Grey, semi-transparent
    logicCube->SetVisAttributes(cubeVisAtt.get());
    
    // Place the silicon detector at fixed position
    G4ThreeVector detectorPosition(0., 0., Constants::DETECTOR_Z_POSITION);
    
    // Store original detector size for comparison
    G4double originalDetSize = fDetSize;
    
    // Calculate number of pixels that would fit with current parameters and FIXED corner offset
    fNumBlocksPerSide = static_cast<G4int>(std::round((fDetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1));
    
    // Calculate the required detector size to accommodate the pixel grid with FIXED corner offset
    G4double requiredDetSize = 2*fPixelCornerOffset + fPixelSize + (fNumBlocksPerSide-1)*fPixelSpacing;
    
    // Update detector size if needed and notify user
    if (std::abs(requiredDetSize - fDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "\n=== AUTOMATIC DETECTOR SIZE ADJUSTMENT ===" << G4endl;
        G4cout << "Original detector size: " << originalDetSize/mm << " mm" << G4endl;
        G4cout << "Calculated pixel grid requires: " << fNumBlocksPerSide << "×" << fNumBlocksPerSide << " pixels" << G4endl;
        G4cout << "Required detector size: " << requiredDetSize/mm << " mm" << G4endl;
        G4cout << "Pixel corner offset (FIXED): " << fPixelCornerOffset/mm << " mm" << G4endl;
        
        // Update detector size
        fDetSize = requiredDetSize;
        
        G4cout << "✓ Detector size adjusted to: " << fDetSize/mm << " mm" << G4endl;
        G4cout << "==========================================" << G4endl;
        
        // Recreate the detector with the correct size
        delete detCube;
        detCube = new G4Box("detCube", fDetSize/2, fDetSize/2, fDetWidth/2);
        delete logicCube;
        logicCube = new G4LogicalVolume(detCube, siliconMat, "logicCube");
        logicCube->SetVisAttributes(cubeVisAtt.get());
    }
    
    // Verify the corner offset calculation
    G4double actualCornerOffset = (fDetSize - (fNumBlocksPerSide-1)*fPixelSpacing - fPixelSize)/2;
    if (std::abs(actualCornerOffset - fPixelCornerOffset) > Constants::PRECISION_TOLERANCE) {
        G4cerr << "ERROR: Corner offset calculation failed!" << G4endl;
        G4cerr << "Expected: " << fPixelCornerOffset/mm << " mm, Got: " << actualCornerOffset/mm << " mm" << G4endl;
    }
    
    // Place the silicon detector at fixed position (only once)
    new G4PVPlacement(0, detectorPosition,
                      logicCube, "physCube", logicWorld, false, 0, checkOverlaps);
    
    // Create aluminum pixels on the detector surface
    G4Box *pixelBlock = new G4Box("pixelBlock", fPixelSize/2, fPixelSize/2, fPixelWidth/2);
    G4LogicalVolume *logicBlock = new G4LogicalVolume(pixelBlock, aluminumMat, "logicBlock");
    
    // Add step limiting for fine tracking - force steps to be at most 10 micrometers
    G4UserLimits* stepLimit = new G4UserLimits(Constants::MAX_STEP_SIZE);  // Max step size
    logicCube->SetUserLimits(stepLimit);   // Apply to detector volume
    logicBlock->SetUserLimits(stepLimit);  // Apply to pixel volumes
    logicWorld->SetUserLimits(stepLimit);  // Apply to world volume
    
    G4cout << "✓ Step limiting enabled: maximum step size = 10 micrometers" << G4endl;
    
    // Place pixels on the detector surface (front face)
    G4int copyNo = 0;
    G4double firstPixelPos = -fDetSize/2 + fPixelCornerOffset + fPixelSize/2;
    
    // Calculate z position for pixels - they should be on the detector surface
    G4double pixelZ = detectorPosition.z() + fDetWidth/2 + fPixelWidth/2;
    
    for (G4int i = 0; i < fNumBlocksPerSide; i++) {
        for (G4int j = 0; j < fNumBlocksPerSide; j++) {
            G4double pixelX = firstPixelPos + i * fPixelSpacing;
            G4double pixelY = firstPixelPos + j * fPixelSpacing;
            
            new G4PVPlacement(0, G4ThreeVector(pixelX, pixelY, pixelZ),
                             logicBlock, "physBlock", logicWorld, false, copyNo++, checkOverlaps);
        }
    }
    
    // Set visualization attributes for pixels - RAII
    auto blockVisAtt = std::make_unique<G4VisAttributes>(G4Colour(0.0, 0.0, 1.0)); // Blue color
    logicBlock->SetVisAttributes(blockVisAtt.get());
    
    // Set the world volume to be invisible
    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    
    // Calculate and print the ratio of pixel area to detector area
    G4double totalPixelArea = fNumBlocksPerSide * fNumBlocksPerSide * fPixelSize * fPixelSize;
    G4double detectorArea = fDetSize * fDetSize;
    G4double pixelAreaRatio = totalPixelArea / detectorArea;
    
    G4cout << "\n=== FINAL DETECTOR CONFIGURATION ===" << G4endl;
    G4cout << "Detector Statistics:" << G4endl;
    G4cout << "  Final detector size: " << fDetSize/mm << " mm × " << fDetSize/mm << " mm" << G4endl;
    if (std::abs(fDetSize - originalDetSize) > Constants::GEOMETRY_TOLERANCE) {
        G4cout << "  (Adjusted from original: " << originalDetSize/mm << " mm)" << G4endl;
    }
    G4cout << "  Pixel corner offset (FIXED): " << fPixelCornerOffset/mm << " mm" << G4endl;
    G4cout << "  Total number of pixels: " << fNumBlocksPerSide * fNumBlocksPerSide << G4endl;
    G4cout << "  Pixel grid: " << fNumBlocksPerSide << " × " << fNumBlocksPerSide << G4endl;
    G4cout << "  Single pixel area: " << fPixelSize * fPixelSize / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Total pixel area: " << totalPixelArea / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Detector area: " << detectorArea / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Pixel area / Detector area ratio: " << pixelAreaRatio << G4endl;
    G4cout << "====================================" << G4endl;

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

G4ThreeVector DetectorConstruction::GetDetectorPosition() const
{
    return G4ThreeVector(0., 0., Constants::DETECTOR_Z_POSITION);
}

// Implementation of IsPositionOnPixel method
G4bool DetectorConstruction::IsPositionOnPixel(const G4ThreeVector& position) const
{
    // Get the detector position
    G4ThreeVector detectorPosition = GetDetectorPosition();
    
    // Check if the hit is in the detector volume first
    G4double detHalfSize = fDetSize/2;
    G4double detHalfWidth = fDetWidth/2;
    
    // Check if hit is within the detector boundaries
    if (std::abs(position.x()) > detHalfSize || 
        std::abs(position.y()) > detHalfSize ||
        position.z() < detectorPosition.z() - detHalfWidth ||
        position.z() > detectorPosition.z() + detHalfWidth + fPixelWidth) {
        return false; // Outside detector volume
    }
    
    // Calculate the first pixel position (corner)
    G4double firstPixelPos = -fDetSize/2 + fPixelCornerOffset + fPixelSize/2;
    
    // Calculate which pixel grid position is closest (i and j indices)
    G4double normX = (position.x() - firstPixelPos) / fPixelSpacing;
    G4double normY = (position.y() - firstPixelPos) / fPixelSpacing;
    
    // Convert to pixel indices
    G4int i = std::round(normX);
    G4int j = std::round(normY);
    
    // Check if indices are within valid range
    if (i < 0 || i >= fNumBlocksPerSide || j < 0 || j >= fNumBlocksPerSide) {
        return false; // Outside pixel grid
    }
    
    // Calculate the actual pixel center position
    G4double pixelX = firstPixelPos + i * fPixelSpacing;
    G4double pixelY = firstPixelPos + j * fPixelSpacing;
    
    // Calculate distance from hit to pixel center
    G4double distanceX = std::abs(position.x() - pixelX);
    G4double distanceY = std::abs(position.y() - pixelY);
    
    // Check if the hit is within the pixel boundary (half the size in each direction)
    return (distanceX <= fPixelSize/2 && distanceY <= fPixelSize/2);
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

// Set automatic radius selection enabled
void DetectorConstruction::SetAutoRadiusEnabled(G4bool enabled)
{
    G4cout << "Setting automatic radius selection enabled: " << (enabled ? "Enabled" : "Disabled") << G4endl;
    
    // Pass the setting to EventAction if it's available
    if (fEventAction) {
        fEventAction->SetAutoRadiusEnabled(enabled);
        G4cout << "Updated EventAction with auto radius enabled: " << (enabled ? "Enabled" : "Disabled") << G4endl;
    } else {
        G4cout << "EventAction not yet available - auto radius enabled will be set when EventAction is connected" << G4endl;
    }
}

// Set minimum automatic radius
void DetectorConstruction::SetMinAutoRadius(G4int minRadius)
{
    G4cout << "Setting minimum automatic radius to: " << minRadius << G4endl;
    
    // Pass the setting to EventAction if it's available
    if (fEventAction) {
        fEventAction->SetAutoRadiusRange(minRadius, fEventAction->GetAutoRadiusEnabled() ? 
                                        Constants::MAX_AUTO_RADIUS : minRadius);
        G4cout << "Updated EventAction with minimum auto radius: " << minRadius << G4endl;
    } else {
        G4cout << "EventAction not yet available - minimum auto radius will be set when EventAction is connected" << G4endl;
    }
}

// Set maximum automatic radius
void DetectorConstruction::SetMaxAutoRadius(G4int maxRadius)
{
    G4cout << "Setting maximum automatic radius to: " << maxRadius << G4endl;
    
    // Pass the setting to EventAction if it's available
    if (fEventAction) {
        fEventAction->SetAutoRadiusRange(fEventAction->GetAutoRadiusEnabled() ? 
                                        Constants::MIN_AUTO_RADIUS : maxRadius, maxRadius);
        G4cout << "Updated EventAction with maximum auto radius: " << maxRadius << G4endl;
    } else {
        G4cout << "EventAction not yet available - maximum auto radius will be set when EventAction is connected" << G4endl;
    }
}

