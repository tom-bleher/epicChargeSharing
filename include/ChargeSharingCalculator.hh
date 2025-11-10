#ifndef CHARGESHARINGCALCULATOR_HH
#define CHARGESHARINGCALCULATOR_HH

#include "G4ThreeVector.hh"
#include "globals.hh"

#include <cstddef>
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
        struct NeighborCell {
            G4int gridIndex{-1};
            G4int globalPixelId{-1};
            G4ThreeVector center{0., 0., 0.};
            G4double fraction{0.0};
            G4double charge{0.0};
            G4double distance{0.0};
            G4double alpha{0.0};
        };

        G4ThreeVector nearestPixelCenter;
        G4int pixelIndexI{0};
        G4int pixelIndexJ{0};
        G4int gridRadius{0};
        G4int gridSide{1};
        std::size_t totalCells{0};
        std::vector<NeighborCell> cells;
        std::vector<G4double> fullFractions;
        std::vector<G4int> fullPixelIds;
        std::vector<G4double> fullPixelX;
        std::vector<G4double> fullPixelY;
        G4int fullGridSide{0};
        std::size_t fullTotalCells{0};
    };

    explicit ChargeSharingCalculator(const DetectorConstruction* detector = nullptr);

    void SetDetector(const DetectorConstruction* detector);
    void SetNeighborhoodRadius(G4int radius);
    G4int GetNeighborhoodRadius() const { return fNeighborhoodRadius; }
    void SetEmitDistanceAlpha(G4bool enabled) { fEmitDistanceAlpha = enabled; }
    G4bool GetEmitDistanceAlpha() const { return fEmitDistanceAlpha; }
    void SetComputeFullGridFractions(G4bool enabled);
    G4bool GetComputeFullGridFractions() const { return fComputeFullGridFractions; }

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
    G4double CalcPixelAlphaSubtended(G4double distance,
                                     G4double pixelWidth,
                                     G4double pixelHeight) const;
    void ComputeChargeFractions(const G4ThreeVector& hitPos,
                                G4double energyDeposit,
                                G4double ionizationEnergy,
                                G4double amplificationFactor,
                                G4double d0,
                                G4double elementaryCharge);
    void BuildOffsets();
    void EnsureFullGridBuffer();
    void ComputeFullGridFractions(const G4ThreeVector& hitPos,
                                  G4double d0,
                                  G4double pixelSize,
                                  G4double pixelSpacing,
                                  G4int numBlocksPerSide);

    const DetectorConstruction* fDetector;
    G4int fNeighborhoodRadius;
    Result fResult;
    std::vector<G4double> fWeightScratch;
    std::vector<G4double> fFullGridWeights;
    struct Offset { int di; int dj; int idx; };
    std::vector<Offset> fOffsets;
    int fGridDim{0};
    int fOffsetsDim{0};
    G4bool fEmitDistanceAlpha{false};
    G4bool fComputeFullGridFractions{false};
    G4bool fNeedsReset{true};
};

#endif // CHARGESHARINGCALCULATOR_HH
