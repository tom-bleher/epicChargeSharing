/// \file NeighborhoodUtils.hh
/// \brief Utilities for managing pixel neighborhood grids.
///
/// \author Tom Bleher, Igor Korover
/// \date 2025

#ifndef ECS_NEIGHBORHOOD_UTILS_HH
#define ECS_NEIGHBORHOOD_UTILS_HH

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ECS {

/// \brief Compute side length from neighborhood radius.
inline std::size_t SideLengthFromRadius(int radius) {
    return static_cast<std::size_t>((2 * std::max(0, radius)) + 1);
}

/// \brief Compute total cells from neighborhood radius.
inline std::size_t TotalCellsFromRadius(int radius) {
    const auto side = SideLengthFromRadius(radius);
    return side * side;
}

/// \brief Neighborhood grid layout manager.
class NeighborhoodLayout {
public:
    NeighborhoodLayout() = default;
    explicit NeighborhoodLayout(int radius) { SetRadius(radius); }

    void SetRadius(int radius) {
        fRadius = std::max(0, radius);
        fSideLength = static_cast<int>(SideLengthFromRadius(fRadius));
        fTotalCells = TotalCellsFromRadius(fRadius);
    }

    [[nodiscard]] int Radius() const { return fRadius; }
    [[nodiscard]] int SideLength() const { return fSideLength; }
    [[nodiscard]] std::size_t TotalCells() const { return fTotalCells; }

private:
    int fRadius{0};
    int fSideLength{1};
    std::size_t fTotalCells{1};
};

/// \brief Resize vector and fill with value.
template <typename T, typename Value>
inline void ResizeAndFill(std::vector<T>& vec, std::size_t size, Value&& value) {
    if (vec.size() != size) {
        vec.assign(size, static_cast<T>(value));
    } else {
        std::fill(vec.begin(), vec.end(), static_cast<T>(value));
    }
}

/// \brief Fill existing vector with value.
template <typename T, typename Value>
inline void Fill(std::vector<T>& vec, Value&& value) {
    std::fill(vec.begin(), vec.end(), static_cast<T>(value));
}

} // namespace ECS

// Backward compatibility aliases
namespace ECS::Internal {
using Layout = ECS::NeighborhoodLayout;
using ECS::Fill;
using ECS::ResizeAndFill;
using ECS::SideLengthFromRadius;
using ECS::TotalCellsFromRadius;
} // namespace ECS::Internal
namespace neighbor = ECS::Internal;

#endif // ECS_NEIGHBORHOOD_UTILS_HH
