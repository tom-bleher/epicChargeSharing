#ifndef CHARGESHARINGCALCULATOR_HH
#define CHARGESHARINGCALCULATOR_HH

#include "globals.hh"
#include "G4ThreeVector.hh"

#include <vector>

class DetectorConstruction;

/**
 * Encapsulates charge-sharing computations and reusable buffers for the
 * (2r+1)x(2r+1) pixel neighborhood around the hit pixel.
 */
class ChargeSharingCalculator
{
public:
    struct Result {
        G4ThreeVector nearestPixelCenter;
        G4int pixelIndexI{0};
        G4int pixelIndexJ{0};
        std::vector<G4double> fractions;
        std::vector<G4double> charges;
        std::vector<G4double> distances;
        std::vector<G4double> alphas;
        std::vector<G4double> pixelX;
        std::vector<G4double> pixelY;
        std::vector<G4int> pixelIds;
    };

    explicit ChargeSharingCalculator(const DetectorConstruction* detector = nullptr);

    void SetDetector(const DetectorConstruction* detector);
    void SetNeighborhoodRadius(G4int radius);
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }

    void ResetForEvent();

    const Result& Compute(const G4ThreeVector& hitPos,
                          G4double energyDeposit,
                          G4double ionizationEnergy,
                          G4double amplificationFactor,
                          G4double d0,
                          G4double elementaryCharge);

private:
    void ReserveBuffers();
    G4ThreeVector CalcNearestPixel(const G4ThreeVector& pos);
    G4double CalcPixelAlphaSubtended(G4double hitX,
                                     G4double hitY,
                                     G4double pixelCenterX,
                                     G4double pixelCenterY,
                                     G4double pixelWidth,
                                     G4double pixelHeight) const;
    void ComputeChargeFractions(const G4ThreeVector& hitPos,
                                G4double energyDeposit,
                                G4double ionizationEnergy,
                                G4double amplificationFactor,
                                G4double d0,
                                G4double elementaryCharge);
    void ComputeNeighborhoodGeometry();

    const DetectorConstruction* fDetector;
    G4int fNeighborhoodRadius;
    Result fResult;
    std::vector<G4double> fWeightGrid;
    std::vector<G4bool> fInBoundsGrid;
};

#endif // CHARGESHARINGCALCULATOR_HH
