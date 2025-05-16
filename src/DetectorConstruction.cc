#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <string>
#include <unistd.h> // For getcwd

DetectorConstruction::DetectorConstruction()
{
    // ————————————————————————
    // Parameters (all lengths are center–to–center except fPixelCornerOffset)
    // ————————————————————————
    // Pixels
    fPixelSize = 100*um;        // "pixel" side‐length
    fPixelWidth = 1*um;         // Width/thickness of each pixel
    fPixelSpacing = 500*um;     // (blue) center–to–center pitch
    fPixelCornerOffset = 100*um;  // (purple) from inner detector edge to first pixel edge

    // Detector 
    fdetSize = 3*cm;           // (green) outer‐square side length
    fdetWidth = 50*um;         // Width/thickness of the detector
    
    // Will be calculated in Construct() based on symmetry constraint
    fNumBlocksPerSide = 0;
    
    // Create the messenger
    fMessenger = new DetectorMessenger(this);
};

DetectorConstruction::~DetectorConstruction()
{
    delete fMessenger;
}

void DetectorConstruction::SetGridParameters(G4double pixelSize, G4double pixelSpacing, G4double pixelCornerOffset, G4int numPixels)
{
    // Update the parameters for pixel grid placement
    fPixelSize = pixelSize;
    fPixelSpacing = pixelSpacing;
    fPixelCornerOffset = pixelCornerOffset;

    // Always compute N as the nearest integer, then adjust offset for perfect centering
    fNumBlocksPerSide = static_cast<G4int>(std::round((fdetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1));
    fPixelCornerOffset = (fdetSize - (fNumBlocksPerSide-1)*fPixelSpacing - fPixelSize)/2;
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4bool checkOverlaps = true;

    // Define materials
    G4NistManager *nist = G4NistManager::Instance();
    G4Material *worldMat = nist->FindOrBuildMaterial("G4_Galactic"); // World material
    G4Material *siliconMat = nist->FindOrBuildMaterial("G4_Si"); // Detector material
    G4Material* SiliconO2Mat = nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE"); // Detector material
    G4Material *aluminumMat = nist->FindOrBuildMaterial("G4_Al"); // Pixel material
    
    // Create world volume
    G4Box *solidWorld = new G4Box("solidWorld", 5*cm, 5*cm, 5*cm);
    G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
    G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.),
                                                     logicWorld, "physWorld", 0, false, 0, checkOverlaps);

    // Use the detector size from constructor
    G4Box* detCube = new G4Box("detCube", fdetSize/2, fdetSize/2, fdetWidth/2);
    G4LogicalVolume* logicCube = new G4LogicalVolume(detCube, siliconMat, "logicCube");
    
    // Set visualization attributes for the cube (semi-transparent)
    G4VisAttributes* cubeVisAtt = new G4VisAttributes(G4Colour(0.7, 0.7, 0.7, 0.5)); // Grey, semi-transparent
    logicCube->SetVisAttributes(cubeVisAtt);
    
    // Place the cube
    new G4PVPlacement(0, G4ThreeVector(0., 0., -1.0*cm),
                      logicCube, "physCube", logicWorld, false, 0, checkOverlaps);

    // Create a single silicon block that we'll place multiple times
    // Use the pixel dimensions from constructor
    G4Box *pixelBlock = new G4Box("pixelBlock", fPixelSize/2, fPixelSize/2, fPixelWidth/2);
    G4LogicalVolume *logicBlock = new G4LogicalVolume(pixelBlock, aluminumMat, "logicBlock");

    // Calculate number of pixels using the symmetry constraint formula
    // N = (fdetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1
    fNumBlocksPerSide = static_cast<G4int>(std::round((fdetSize - 2*fPixelCornerOffset - fPixelSize)/fPixelSpacing + 1));
    fPixelCornerOffset = (fdetSize - (fNumBlocksPerSide-1)*fPixelSpacing - fPixelSize)/2;
    
    // Function to place blocks on a face
    auto placeBlocksOnFace = [&](G4double x, G4double y, G4double z, G4int normalAxis) {
        G4int copyNo = 0;
        
        // First pixel's center coordinates calculation
        G4double firstPixelPos = -fdetSize/2 + fPixelCornerOffset + fPixelSize/2;
        
        for (G4int i = 0; i < fNumBlocksPerSide; i++) {
            for (G4int j = 0; j < fNumBlocksPerSide; j++) {
                // Default positions (will be overwritten based on normalAxis)
                G4double xPos = x;
                G4double yPos = y;
                G4double zPos = z;
                
                // Calculate pixel center positions based on the specified formula:
                // x_i = -fdetSize/2 + fPixelCornerOffset + fPixelSize/2 + i*fPixelSpacing
                // y_j = -fdetSize/2 + fPixelCornerOffset + fPixelSize/2 + j*fPixelSpacing
                
                // For the z-normal faces (top/bottom), use i for x and j for y
                if (normalAxis == 3) {
                    xPos = firstPixelPos + i * fPixelSpacing;
                    yPos = firstPixelPos + j * fPixelSpacing;
                }
                // For the y-normal faces (front/back), use i for x and j for z
                else if (normalAxis == 2) {
                    xPos = firstPixelPos + i * fPixelSpacing;
                    zPos = firstPixelPos + j * fPixelSpacing;
                }
                // For the x-normal faces (left/right), use i for y and j for z
                else if (normalAxis == 1) {
                    yPos = firstPixelPos + i * fPixelSpacing;
                    zPos = firstPixelPos + j * fPixelSpacing;
                }
                
                new G4PVPlacement(0, G4ThreeVector(xPos, yPos, zPos),
                                 logicBlock, "physBlock", logicWorld, false, copyNo++, checkOverlaps);
            }
        }
    };
    
    // Place blocks on the detector face
    placeBlocksOnFace(0, 0, -1.0*cm, 3);  // z=-1.0*cm face, using axis=3 (z-normal)
    
    // Set visualization attributes
    G4VisAttributes* blockVisAtt = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0)); // Blue color
    logicBlock->SetVisAttributes(blockVisAtt);
    
    // Set the world volume to be invisible
    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    
    // Calculate and print the ratio of pixel area to detector area
    G4double totalPixelArea = fNumBlocksPerSide * fNumBlocksPerSide * fPixelSize * fPixelSize;
    G4double detectorArea = fdetSize * fdetSize;
    G4double pixelAreaRatio = totalPixelArea / detectorArea;
    G4cout << "\nDetector Statistics:" << G4endl;
    G4cout << "  Total number of pixels: " << fNumBlocksPerSide * fNumBlocksPerSide << G4endl;
    G4cout << "  Single pixel area: " << fPixelSize * fPixelSize / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Total pixel area: " << totalPixelArea / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Detector area: " << detectorArea / (mm*mm) << " mm²" << G4endl;
    G4cout << "  Pixel area / Detector area ratio: " << pixelAreaRatio << G4endl;

    // Save all simulation parameters to a log file
    SaveSimulationParameters(totalPixelArea, detectorArea, pixelAreaRatio);

    return physWorld;
}

// Implementation of IsPositionOnPixel method
G4bool DetectorConstruction::IsPositionOnPixel(const G4ThreeVector& position) const
{
    // Get the detector position
    G4ThreeVector detectorPosition = GetDetectorPosition();
    
    // Calculate the position relative to the detector face
    G4ThreeVector relativePos = position - detectorPosition;
    
    // For the z-normal face (top/bottom), only x and y matter for pixel position
    // Calculate the first pixel position (corner)
    G4double firstPixelPos = -fdetSize/2 + fPixelCornerOffset + fPixelSize/2;
    
    // Calculate which pixel grid position is closest (i and j indices)
    G4double normX = (relativePos.x() - firstPixelPos) / fPixelSpacing;
    G4double normY = (relativePos.y() - firstPixelPos) / fPixelSpacing;
    
    // Convert to pixel indices
    G4int i = std::round(normX);
    G4int j = std::round(normY);
    
    // Clamp i and j to valid pixel indices
    i = std::max(0, std::min(i, fNumBlocksPerSide - 1));
    j = std::max(0, std::min(j, fNumBlocksPerSide - 1));
    
    // Calculate the actual pixel center position
    G4double pixelX = firstPixelPos + i * fPixelSpacing;
    G4double pixelY = firstPixelPos + j * fPixelSpacing;
    
    // Calculate distance from hit to pixel center
    G4double distanceX = std::abs(relativePos.x() - pixelX);
    G4double distanceY = std::abs(relativePos.y() - pixelY);
    
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
    
    // Get the absolute path for logs directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        G4cerr << "ERROR: Could not get current working directory" << G4endl;
        return;
    }
    
    // Create logs directory with absolute path
    std::string logsDir = std::string(cwd) + "/logs";
    G4cout << "Creating logs directory at: " << logsDir << G4endl;
    system(("mkdir -p " + logsDir).c_str());
    
    // Create filename with timestamp in logs directory
    std::string filename = logsDir + "/simulation_params_" + std::string(timestamp) + ".log";
    
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
        paramFile << "Detector Size: " << fdetSize/mm << " mm" << std::endl;
        paramFile << "Detector Width/Thickness: " << fdetWidth/mm << " mm" << std::endl;
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

