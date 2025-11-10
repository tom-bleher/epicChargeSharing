#ifndef INTERNAL_NEIGHBORHOODBUFFER_HH
#define INTERNAL_NEIGHBORHOODBUFFER_HH

#include <algorithm>
#include <cstddef>
#include <vector>

namespace neighbor
{

inline std::size_t SideLengthFromRadius(int radius)
{
    const int clamped = std::max(0, radius);
    return static_cast<std::size_t>(2 * clamped + 1);
}

inline std::size_t TotalCellsFromRadius(int radius)
{
    const auto side = SideLengthFromRadius(radius);
    return side * side;
}

class Layout
{
public:
    Layout() = default;

    explicit Layout(int radius)
    {
        SetRadius(radius);
    }

    void SetRadius(int radius)
    {
        const int clamped = std::max(0, radius);
        fRadius = clamped;
        fSideLength = static_cast<int>(SideLengthFromRadius(clamped));
        fTotalCells = TotalCellsFromRadius(clamped);
    }

    [[nodiscard]] int Radius() const { return fRadius; }
    [[nodiscard]] int SideLength() const { return fSideLength; }
    [[nodiscard]] std::size_t TotalCells() const { return fTotalCells; }

private:
    int fRadius{0};
    int fSideLength{1};
    std::size_t fTotalCells{1};
};

template <typename T, typename Value>
inline void ResizeAndFill(std::vector<T>& vec, std::size_t size, Value&& value)
{
    if (vec.size() != size) {
        vec.assign(size, static_cast<T>(value));
    } else {
        std::fill(vec.begin(), vec.end(), static_cast<T>(value));
    }
}

template <typename T, typename Value>
inline void Fill(std::vector<T>& vec, Value&& value)
{
    std::fill(vec.begin(), vec.end(), static_cast<T>(value));
}

} // namespace neighbor

#endif // INTERNAL_NEIGHBORHOODBUFFER_HH


