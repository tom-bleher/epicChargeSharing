#pragma once

#include "ChargeSharingConfig.h"

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
    double distanceMM{0.0};
    double alphaRad{0.0};
    double pixelXMM{0.0};
    double pixelYMM{0.0};
    int pixelId{-1};
    int di{0};
    int dj{0};
  };

  struct Result {
    std::array<double, 3> nearestPixelCenterMM{};
    std::array<double, 3> reconstructedPositionMM{};
    int pixelIndexI{0};
    int pixelIndexJ{0};
    double totalCollectedChargeC{0.0};
    std::vector<NeighborData> neighbors;
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

  struct Offset {
    int di{0};
    int dj{0};
    std::size_t idx{0};
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
  double firstPixelCenterCoordinate(double detectorSize, double cornerOffset, double pixelSize) const;
  void ensureGrid();
  void resetBuffers();
  void rebuildOffsets();
  double calcPixelAlpha(double distanceMM, double pixelWidthMM, double pixelHeightMM) const;
  double linearModelBeta(double pitchMM) const;
  int linearizedPixelId(int indexI, int indexJ) const;

  ChargeSharingConfig m_cfg{};
  const dd4hep::DDSegmentation::CartesianGridXY* m_segmentation{nullptr};
  const dd4hep::DDSegmentation::BitFieldCoder* m_decoder{nullptr};
  bool m_haveSegmentation{false};
  IndexBounds m_boundsX{};
  IndexBounds m_boundsY{};
  int m_gridDim{0};
  std::vector<Offset> m_offsets;

  std::vector<double> m_weightGrid;
  std::vector<bool> m_inBounds;
  std::vector<double> m_fractions;
  std::vector<double> m_charges;
  std::vector<double> m_distances;
  std::vector<double> m_alphas;
  std::vector<double> m_pixelX;
  std::vector<double> m_pixelY;
  std::vector<int> m_pixelIds;
};

}  // namespace epic::chargesharing

