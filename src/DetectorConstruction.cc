#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

DetectorConstruction::DetectorConstruction()
{
    // Pixels
    fPixelSize = 100*um; // length and height of each pixel
    fPixelWidth = 1*um;
    fPixelSpacing = 500*um;   // spacing between center of pixels
    fPixelCornerOffset = 1*um;     // edge-most pixel distance from edge of detector

    // Detector 
    fdetSize = 3*cm; // length and height of each pixel
    fdetWidth = 50*um;
    
    // Will be calculated in Construct()
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
    fPixelSize = pixelSize;
    fPixelSpacing = pixelSpacing;
    fPixelCornerOffset = pixelCornerOffset;
    fNumBlocksPerSide = numPixels;
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

    // Calculate the available space for the grid, subtracting corner offsets from both sides
    G4double availableSpace = fdetSize - 2 * fPixelCornerOffset;
    
    // Calculate how many blocks can fit in the available space based on fixed spacing
    // Number of blocks = available space / spacing + 1
    fNumBlocksPerSide = static_cast<G4int>(availableSpace / fPixelSpacing) + 1;
    
    // Function to place blocks on a face
    auto placeBlocksOnFace = [&](G4double x, G4double y, G4double z, G4int normalAxis) {
        G4int copyNo = 0;
        
        // Starting position for the grid, adjusted by corner offset
        G4double startPos = -fdetSize/2 + fPixelCornerOffset;
        
        for (G4int i = 0; i < fNumBlocksPerSide; i++) {
            for (G4int j = 0; j < fNumBlocksPerSide; j++) {
                // Calculate positions based on fPixelSpacing
                G4double xPos = (normalAxis == 1) ? x : startPos + i * fPixelSpacing;
                G4double yPos = (normalAxis == 2) ? y : startPos + j * fPixelSpacing;
                G4double zPos = (normalAxis == 3) ? z : startPos + i * fPixelSpacing;
                
                // For the z-normal faces (top/bottom), use i for x and j for y
                if (normalAxis == 3) {
                    xPos = startPos + i * fPixelSpacing;
                    yPos = startPos + j * fPixelSpacing;
                }
                // For the y-normal faces (front/back), use i for x and j for z
                else if (normalAxis == 2) {
                    xPos = startPos + i * fPixelSpacing;
                    zPos = startPos + j * fPixelSpacing;
                }
                // For the x-normal faces (left/right), use i for y and j for z
                else if (normalAxis == 1) {
                    yPos = startPos + i * fPixelSpacing;
                    zPos = startPos + j * fPixelSpacing;
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

