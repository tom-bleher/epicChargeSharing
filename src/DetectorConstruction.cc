#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

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

    return physWorld;
}

