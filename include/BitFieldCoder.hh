/// \file BitFieldCoder.hh
/// \brief Standalone reimplementation of DD4hep's BitFieldCoder.
///
/// Encodes/decodes named fields into a 64-bit CellID using the same
/// algorithm and descriptor string format as dd4hep::DDSegmentation::BitFieldCoder.
/// This allows producing DD4hep-compatible CellIDs without linking against DD4hep.
///
/// Descriptor string format (comma-separated fields):
///   "name:width"          — auto-offset, placed after previous field
///   "name:offset:width"   — explicit bit offset
/// Negative width indicates a signed (two's complement) field.
///
/// Example (ePIC silicon barrel tracker):
///   "system:8,layer:4,module:12,sensor:2,x:32:-16,y:-16"
///
/// \see https://github.com/AIDASoft/DD4hep (DDCore/include/DDSegmentation/BitFieldCoder.h)

#ifndef ECS_BITFIELDCODER_HH
#define ECS_BITFIELDCODER_HH

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ECS {

using CellID  = std::uint64_t;
using FieldID = std::int64_t;

/// \brief A single named bit field within a 64-bit CellID.
class BitFieldElement {
public:
    BitFieldElement(const std::string& name, unsigned offset, int signedWidth)
        : m_name(name),
          m_offset(offset),
          m_width(static_cast<unsigned>(std::abs(signedWidth))),
          m_isSigned(signedWidth < 0) {

        if (m_offset > 63 || m_offset + m_width > 64)
            throw std::runtime_error("BitFieldElement '" + m_name + "': exceeds 64 bits");
        if (m_width == 0)
            throw std::runtime_error("BitFieldElement '" + m_name + "': zero width");

        m_mask = ((uint64_t{1} << m_width) - 1) << m_offset;

        if (m_isSigned) {
            m_minVal = -(FieldID{1} << (m_width - 1));
            m_maxVal =  (FieldID{1} << (m_width - 1)) - 1;
        } else {
            m_minVal = 0;
            m_maxVal = (FieldID{1} << m_width) - 1;
        }
    }

    /// Extract this field's value from a CellID.
    [[nodiscard]] FieldID value(CellID id) const {
        auto val = static_cast<FieldID>((id & m_mask) >> m_offset);
        if (m_isSigned && (val & (FieldID{1} << (m_width - 1))) != 0)
            val -= (FieldID{1} << m_width);  // two's complement sign extension
        return val;
    }

    /// Set this field's value in a CellID.
    void set(CellID& field, FieldID in) const {
        if (in < m_minVal || in > m_maxVal)
            throw std::runtime_error("BitFieldElement '" + m_name
                + "': value " + std::to_string(in) + " out of range ["
                + std::to_string(m_minVal) + ", " + std::to_string(m_maxVal) + "]");
        field &= ~m_mask;
        field |= (static_cast<CellID>(in) << m_offset) & m_mask;
    }

    [[nodiscard]] const std::string& name() const { return m_name; }
    [[nodiscard]] unsigned offset() const { return m_offset; }
    [[nodiscard]] unsigned width() const { return m_width; }
    [[nodiscard]] bool isSigned() const { return m_isSigned; }

private:
    std::string m_name;
    CellID      m_mask{};
    unsigned    m_offset{};
    unsigned    m_width{};
    FieldID     m_minVal{};
    FieldID     m_maxVal{};
    bool        m_isSigned{};
};

/// \brief Encodes/decodes named fields in a 64-bit CellID.
///
/// Standalone reimplementation of dd4hep::DDSegmentation::BitFieldCoder.
/// Parses the same descriptor string format and produces identical bit layouts.
class BitFieldCoder {
public:
    BitFieldCoder() = default;

    /// Construct from a DD4hep-style descriptor string.
    explicit BitFieldCoder(const std::string& descriptor) {
        if (!descriptor.empty())
            init(descriptor);
    }

    /// Get a field value by name.
    [[nodiscard]] FieldID get(CellID bitfield, const std::string& name) const {
        return m_fields.at(index(name)).value(bitfield);
    }

    /// Set a field value by name.
    void set(CellID& bitfield, const std::string& name, FieldID value) const {
        m_fields.at(index(name)).set(bitfield, value);
    }

    /// Get the descriptor string this coder was built from.
    [[nodiscard]] const std::string& descriptor() const { return m_descriptor; }

    /// Number of fields.
    [[nodiscard]] std::size_t size() const { return m_fields.size(); }

private:
    [[nodiscard]] std::size_t index(const std::string& name) const {
        for (std::size_t i = 0; i < m_fields.size(); ++i)
            if (m_fields[i].name() == name) return i;
        throw std::runtime_error("BitFieldCoder: unknown field '" + name + "'");
    }

    void init(const std::string& initString) {
        m_descriptor = initString;
        unsigned offset = 0;
        CellID usedBits = 0;

        for (const auto& desc : tokenize(initString, ',')) {
            auto parts = tokenize(desc, ':');

            std::string name;
            int width;
            unsigned thisOffset;

            if (parts.size() == 2) {
                name = parts[0];
                width = std::atoi(parts[1].c_str());
                thisOffset = offset;
                offset += static_cast<unsigned>(std::abs(width));
            } else if (parts.size() == 3) {
                name = parts[0];
                thisOffset = static_cast<unsigned>(std::atoi(parts[1].c_str()));
                width = std::atoi(parts[2].c_str());
                offset = thisOffset + static_cast<unsigned>(std::abs(width));
            } else {
                throw std::runtime_error(
                    "BitFieldCoder: invalid descriptor '" + desc + "'");
            }

            m_fields.emplace_back(name, thisOffset, width);

            CellID fieldMask = ((uint64_t{1} << static_cast<unsigned>(std::abs(width))) - 1) << thisOffset;
            if (usedBits & fieldMask)
                throw std::runtime_error(
                    "BitFieldCoder: overlapping bits for field '" + name + "'");
            usedBits |= fieldMask;
        }
    }

    static std::vector<std::string> tokenize(const std::string& s, char delim) {
        std::vector<std::string> tokens;
        std::string token;
        for (char c : s) {
            if (c == delim) {
                if (!token.empty()) { tokens.push_back(token); token.clear(); }
            } else if (c != ' ') {
                token += c;
            }
        }
        if (!token.empty()) tokens.push_back(token);
        return tokens;
    }

    std::vector<BitFieldElement> m_fields;
    std::string m_descriptor;
};

} // namespace ECS

#endif // ECS_BITFIELDCODER_HH
