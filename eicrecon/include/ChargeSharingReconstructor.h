#pragma once

#include "ChargeSharingConfig.h"
#include "ChargeSharingCore.hh"
#include "GaussianFit.hh"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace dd4hep {
namespace DDSegmentation {
class BitFieldCoder;
class CartesianGridXY;
}  // namespace DDSegmentation
}  // namespace dd4hep

namespace epic::chargesharing {

class ChargeSharingReconstructor {
 public:
  struct Input {
    std::array<double, 3> hitPositionMM{};                 ///< Truth position in mm
    std::optional<std::array<double, 3>> pixelHintMM{};    ///< Optional center of seed pixel (mm)
    double energyDepositGeV{0.0};                          ///< Energy in GeV
    std::uint64_t cellID{0};
  };

  struct NeighborData {
    double fraction{-999.0};
    double chargeC{0.0};
    double distanceMM{0.0};  ///< d_i: distance to pixel center (mm)
    double alphaRad{0.0};
    double pixelXMM{0.0};
    double pixelYMM{0.0};
    int pixelId{-1};
    int di{0};
    int dj{0};

    // Mode-specific fractions and charges for diagnostics
    double fractionRow{0.0};     ///< Fraction using row-only denominator
    double fractionCol{0.0};     ///< Fraction using col-only denominator
    double fractionBlock{0.0};   ///< Fraction using block denominator
    double chargeRowC{0.0};      ///< Charge using row fraction (Coulombs)
    double chargeColC{0.0};      ///< Charge using col fraction (Coulombs)
    double chargeBlockC{0.0};    ///< Charge using block fraction (Coulombs)
  };

  struct Result {
    std::array<double, 3> nearestPixelCenterMM{};
    std::array<double, 3> reconstructedPositionMM{};
    int pixelIndexI{0};
    int pixelIndexJ{0};
    double totalCollectedChargeC{0.0};
    std::vector<NeighborData> neighbors;

    // Fit results (if Gaussian fitting enabled)
    fit::GaussFit1DResult fitRowX;
    fit::GaussFit1DResult fitColY;
    fit::GaussFit2DResult fit2D;

    // ─────────────────────────── Diagnostic Metadata ───────────────────────────
    // Truth information (from input)
    std::array<double, 3> truthPositionMM{};    ///< Original hit position (mm)
    double inputEnergyDepositGeV{0.0};          ///< Input energy deposit (GeV)
    std::uint64_t inputCellID{0};               ///< Input cell ID

    // Reconstruction residuals
    double residualXMM() const { return reconstructedPositionMM[0] - truthPositionMM[0]; }
    double residualYMM() const { return reconstructedPositionMM[1] - truthPositionMM[1]; }

    // Summary charge statistics
    double maxNeighborChargeC{0.0};             ///< Max charge in neighborhood
    int numActiveNeighbors{0};                  ///< Number of pixels with charge > 0

    // Grid info
    int neighborhoodRadius{0};                  ///< Neighborhood radius used
    int neighborhoodGridSize{0};                ///< Total grid size (2*r+1)^2
  };

  void configure(const ChargeSharingConfig& cfg,
                 const dd4hep::DDSegmentation::CartesianGridXY* segmentation = nullptr,
                 const dd4hep::DDSegmentation::BitFieldCoder* decoder = nullptr);
  const ChargeSharingConfig& config() const { return m_cfg; }

  Result process(const Input& input);

 private:
  struct PixelLocation {
    std::array<double, 3> center;
    int indexI{0};
    int indexJ{0};
  };

  struct IndexBounds {
    int minIndex{0};
    int maxIndex{-1};

    bool hasBounds() const { return maxIndex >= minIndex; }
    bool contains(int value) const { return !hasBounds() || (value >= minIndex && value <= maxIndex); }
  };

  PixelLocation findNearestPixelFallback(const std::array<double, 3>& positionMM) const;
  PixelLocation findPixelFromSegmentation(std::uint64_t cellID) const;
  PixelLocation pixelLocationFromIndices(int indexI, int indexJ) const;
  bool isPixelIndexInBounds(int indexI, int indexJ) const;

  /// Perform position reconstruction using configured method
  std::array<double, 3> reconstructPosition(
      const core::NeighborhoodResult& neighborhood,
      double centerZ,
      fit::GaussFit1DResult& fitRowX,
      fit::GaussFit1DResult& fitColY,
      fit::GaussFit2DResult& fit2D) const;

  /// Reconstruct using charge-weighted centroid (fallback)
  std::array<double, 3> reconstructCentroid(
      const core::NeighborhoodResult& neighborhood,
      double centerZ) const;

  /// Reconstruct using 1D Gaussian fits on row and column
  std::array<double, 3> reconstructGaussian1D(
      const core::NeighborhoodResult& neighborhood,
      double centerZ,
      fit::GaussFit1DResult& fitRowX,
      fit::GaussFit1DResult& fitColY) const;

  /// Reconstruct using 2D Gaussian fit
  std::array<double, 3> reconstructGaussian2D(
      const core::NeighborhoodResult& neighborhood,
      double centerZ,
      fit::GaussFit2DResult& fit2D) const;

  ChargeSharingConfig m_cfg{};
  const dd4hep::DDSegmentation::CartesianGridXY* m_segmentation{nullptr};
  const dd4hep::DDSegmentation::BitFieldCoder* m_decoder{nullptr};
  bool m_haveSegmentation{false};
  IndexBounds m_boundsX{};
  IndexBounds m_boundsY{};
  mutable core::NoiseModel m_noiseModel{};  ///< Noise model for realistic charge simulation
};

}  // namespace epic::chargesharing
