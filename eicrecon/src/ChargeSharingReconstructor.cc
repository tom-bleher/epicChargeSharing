// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include "ChargeSharingReconstructor.h"

#include <DD4hep/Detector.h>
#include <DD4hep/Readout.h>
#include <DD4hep/Segmentations.h>
#include <DD4hep/Volumes.h>
#include <DDRec/CellIDPositionConverter.h>
#include <DDSegmentation/BitFieldCoder.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <DDSegmentation/CartesianGridXZ.h>
#include <Evaluator/DD4hepUnits.h>
#include <algorithms/geo.h>
#include <edm4eic/CovDiag3f.h>
#include <edm4hep/Vector3f.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace {

/// Helper struct for DD4hep constant lookups
struct DetectorConstants {
    std::string siliconThickness;
    std::string detectorSize;
    std::string pixelSize;
    std::string copperThickness;
};

/// Map readout names to their DD4hep constant prefixes
DetectorConstants getDetectorConstants(const std::string& readout) {
    DetectorConstants constants;

    if (readout.find("LumiSpec") != std::string::npos || readout.find("Lumi") != std::string::npos) {
        constants.siliconThickness = "LumiSpecTracker_Si_DZ";
        constants.detectorSize = "LumiSpecTracker_DXY";
        constants.pixelSize = "LumiSpecTracker_pixelSize";
        constants.copperThickness = "LumiSpecTracker_Cu_DZ";
    } else if (readout.find("B0") != std::string::npos) {
        constants.siliconThickness = "";
        constants.detectorSize = "";
        constants.pixelSize = "";
        constants.copperThickness = "";
    } else if (readout.find("TOF") != std::string::npos || readout.find("Tof") != std::string::npos) {
        constants.siliconThickness = "TOFBarrel_sensor_thickness";
        constants.detectorSize = "";
        constants.pixelSize = "";
    }

    return constants;
}

/// Try to extract sensor thickness from the sensitive detector's volume
double getSensorThicknessFromReadout(const dd4hep::Detector* detector, const std::string& readoutName) {
    if (!detector)
        return 0.0;

    try {
        dd4hep::Readout readout = detector->readout(readoutName);
        if (!readout.isValid())
            return 0.0;

        dd4hep::IDDescriptor idDesc = readout.idSpec();
        if (!idDesc.isValid())
            return 0.0;

        std::string detName = readoutName;
        if (detName.size() > 4 && detName.substr(detName.size() - 4) == "Hits") {
            detName = detName.substr(0, detName.size() - 4);
        }

        dd4hep::DetElement det = detector->detector(detName);
        if (!det.isValid()) {
            det = detector->detector(readoutName);
        }

        if (det.isValid() && det.placement().isValid()) {
            dd4hep::Volume vol = det.placement().volume();
            if (vol.isValid() && vol.solid().isValid()) {
                auto dims = vol.solid().dimensions();
                if (dims.size() >= 3) {
                    return 2.0 * dims[2] / dd4hep::mm;
                }
            }
        }
    } catch (...) {
        // Silently fail - will use fallback
    }

    return 0.0;
}

/// Get hardcoded fallback values for detectors without named DD4hep constants
void applyDetectorFallbacks(const std::string& readout, eicrecon::ChargeSharingConfig& cfg) {
    if (readout.find("B0") != std::string::npos) {
        cfg.detectorThicknessMM = 0.3; // SiliconOxide layer
    }
}

} // anonymous namespace

namespace eicrecon {

namespace core = epic::chargesharing::core;
namespace fit = epic::chargesharing::fit;

// ---------------------------------------------------------------------------
// init() -- one-time setup, replaces the old configure() method.
// Reads config from WithPodConfig::m_cfg and sets up DD4hep geometry.
// ---------------------------------------------------------------------------
void ChargeSharingReconstructor::init() {
    if (m_cfg.neighborhoodRadius < 0) {
        m_cfg.neighborhoodRadius = 0;
    }

    // Set effective Y values if not specified
    if (m_cfg.pixelSizeYMM <= 0.0) {
        m_cfg.pixelSizeYMM = m_cfg.pixelSizeMM;
    }
    if (m_cfg.pixelSpacingYMM <= 0.0) {
        m_cfg.pixelSpacingYMM = m_cfg.pixelSpacingMM;
    }

    // Configure noise model
    core::NoiseConfig noiseConfig;
    noiseConfig.enabled = m_cfg.noiseEnabled;
    noiseConfig.gainSigmaMin = m_cfg.noiseGainSigmaMin;
    noiseConfig.gainSigmaMax = m_cfg.noiseGainSigmaMax;
    noiseConfig.electronNoiseCount = m_cfg.noiseElectronCount;
    noiseConfig.elementaryCharge = m_cfg.elementaryChargeC;
    m_noise_model.setConfig(noiseConfig);
    if (m_cfg.noiseSeed != 0) {
        m_noise_model.setSeed(m_cfg.noiseSeed);
    }

    // -----------------------------------------------------------------------
    // DD4hep geometry setup -- read segmentation, decoder, and constants.
    // Uses algorithms::GeoSvc singleton (initialized by JANA2 framework).
    // -----------------------------------------------------------------------
    const dd4hep::DDSegmentation::BitFieldCoder* decoder = nullptr;

    const auto& geo = algorithms::GeoSvc::instance();
    const dd4hep::Detector* detector = geo.detector();
    m_converter = geo.cellIDPositionConverter();

    if (detector) {
        try {
            dd4hep::Readout readout = detector->readout(m_cfg.readout);
            dd4hep::Segmentation segmentation = readout.segmentation();

            if (!segmentation.isValid()) {
                warning("Readout '{}' has no valid segmentation; falling back to manual geometry", m_cfg.readout);
            } else {
                const dd4hep::DDSegmentation::CartesianGridXY* segImplXY = nullptr;
                const dd4hep::DDSegmentation::CartesianGridXZ* segImplXZ = nullptr;
                bool usingXZCoordinates = false;

                segImplXY = dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation.segmentation());

                if (segImplXY == nullptr) {
                    segImplXZ =
                        dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXZ*>(segmentation.segmentation());
                    if (segImplXZ != nullptr) {
                        usingXZCoordinates = true;
                    }
                }

                if (segImplXY == nullptr && segImplXZ == nullptr) {
                    warning("Segmentation for readout '{}' is '{}'; expected CartesianGridXY or CartesianGridXZ",
                            m_cfg.readout, segmentation.type());
                } else {
                    decoder = segmentation.decoder();
                    if (decoder == nullptr) {
                        warning("Segmentation for readout '{}' lacks a BitField decoder; neighbor bounds unavailable",
                                m_cfg.readout);
                    }

                    SegmentationConfig segCfg{};
                    segCfg.valid = (segImplXY != nullptr || segImplXZ != nullptr) && decoder != nullptr;

                    if (segImplXY != nullptr) {
                        segCfg.gridSizeXMM = segImplXY->gridSizeX();
                        segCfg.gridSizeYMM = segImplXY->gridSizeY();
                        segCfg.offsetXMM = segImplXY->offsetX();
                        segCfg.offsetYMM = segImplXY->offsetY();
                        segCfg.fieldNameX = segImplXY->fieldNameX();
                        segCfg.fieldNameY = segImplXY->fieldNameY();
                    } else {
                        segCfg.gridSizeXMM = segImplXZ->gridSizeX();
                        segCfg.gridSizeYMM = segImplXZ->gridSizeZ();
                        segCfg.offsetXMM = segImplXZ->offsetX();
                        segCfg.offsetYMM = segImplXZ->offsetZ();
                        segCfg.fieldNameX = segImplXZ->fieldNameX();
                        segCfg.fieldNameY = segImplXZ->fieldNameZ();
                    }

                    if (decoder != nullptr) {
                        const auto& fieldX = (*decoder)[segCfg.fieldNameX];
                        const auto& fieldY = (*decoder)[segCfg.fieldNameY];
                        segCfg.minIndexX = fieldX.minValue();
                        segCfg.maxIndexX = fieldX.maxValue();
                        segCfg.minIndexY = fieldY.minValue();
                        segCfg.maxIndexY = fieldY.maxValue();
                        if (fieldX.maxValue() >= fieldX.minValue()) {
                            segCfg.numCellsX = fieldX.maxValue() - fieldX.minValue() + 1;
                        }
                        if (fieldY.maxValue() >= fieldY.minValue()) {
                            segCfg.numCellsY = fieldY.maxValue() - fieldY.minValue() + 1;
                        }
                    }

                    const auto dims = segImplXY ? segImplXY->cellDimensions(0) : segImplXZ->cellDimensions(0);
                    if (!dims.empty()) {
                        segCfg.cellSizeXMM = dims[0];
                        if (dims.size() > 1) {
                            segCfg.cellSizeYMM = dims[1];
                        }
                    }
                    if (segCfg.cellSizeXMM <= 0.0) {
                        segCfg.cellSizeXMM = segCfg.gridSizeXMM;
                    }
                    if (segCfg.cellSizeYMM <= 0.0) {
                        segCfg.cellSizeYMM = segCfg.gridSizeYMM;
                    }

                    segCfg.useXZCoordinates = usingXZCoordinates;
                    m_cfg.segmentation = segCfg;

                    if (segCfg.valid) {
                        m_cfg.pixelSpacingMM = segCfg.gridSizeXMM;
                        m_cfg.pixelSpacingYMM = segCfg.gridSizeYMM;
                        m_cfg.pixelSizeMM = segCfg.cellSizeXMM;
                        m_cfg.pixelSizeYMM = segCfg.cellSizeYMM;
                        m_cfg.gridOffsetMM = segCfg.offsetXMM;
                        m_cfg.gridOffsetYMM = segCfg.offsetYMM;

                        if (m_cfg.pixelsPerSide <= 0 && segCfg.numCellsX > 0 &&
                            segCfg.numCellsX == segCfg.numCellsY) {
                            m_cfg.pixelsPerSide = segCfg.numCellsX;
                        }
                    }

                    info("Using DD4hep {} segmentation for readout '{}': pitch=({}, {}) mm, offset=({}, {}) "
                         "mm, cells=({}, {})",
                         usingXZCoordinates ? "CartesianGridXZ" : "CartesianGridXY", m_cfg.readout,
                         m_cfg.pixelSpacingMM, m_cfg.pixelSpacingYMM, m_cfg.gridOffsetMM, m_cfg.gridOffsetYMM,
                         segCfg.numCellsX, segCfg.numCellsY);
                }
            }

            // Read additional detector constants from DD4hep compact XML
            const auto detConstants = getDetectorConstants(m_cfg.readout);
            int constantsRead = 0;

            if (!detConstants.siliconThickness.empty()) {
                try {
                    double thickness = detector->constantAsDouble(detConstants.siliconThickness);
                    m_cfg.detectorThicknessMM = thickness / dd4hep::mm;
                    ++constantsRead;
                    debug("  Read {} = {} mm", detConstants.siliconThickness, m_cfg.detectorThicknessMM);
                } catch (const std::exception&) {
                }
            }

            if (!detConstants.detectorSize.empty()) {
                try {
                    double size = detector->constantAsDouble(detConstants.detectorSize);
                    m_cfg.detectorSizeMM = size / dd4hep::mm;
                    ++constantsRead;
                    debug("  Read {} = {} mm", detConstants.detectorSize, m_cfg.detectorSizeMM);
                } catch (const std::exception&) {
                }
            }

            if (!detConstants.pixelSize.empty()) {
                try {
                    double pixSize = detector->constantAsDouble(detConstants.pixelSize);
                    double pixSizeMM = pixSize / dd4hep::mm;
                    if (std::abs(pixSizeMM - m_cfg.pixelSizeMM) > 0.001) {
                        m_cfg.pixelSizeMM = pixSizeMM;
                        m_cfg.pixelSizeYMM = pixSizeMM;
                        ++constantsRead;
                        debug("  Read {} = {} mm (overriding segmentation)", detConstants.pixelSize, m_cfg.pixelSizeMM);
                    }
                } catch (const std::exception&) {
                }
            }

            if (!detConstants.copperThickness.empty()) {
                try {
                    double cuThickness = detector->constantAsDouble(detConstants.copperThickness);
                    m_cfg.pixelThicknessMM = cuThickness / dd4hep::mm;
                    ++constantsRead;
                    debug("  Read {} = {} mm", detConstants.copperThickness, m_cfg.pixelThicknessMM);
                } catch (const std::exception&) {
                }
            }

            if (constantsRead > 0) {
                info("Read {} DD4hep constants for '{}': Si thickness={} mm, det size={} mm", constantsRead,
                     m_cfg.readout, m_cfg.detectorThicknessMM, m_cfg.detectorSizeMM);
            } else {
                double volumeThickness = getSensorThicknessFromReadout(detector, m_cfg.readout);
                if (volumeThickness > 0.0) {
                    m_cfg.detectorThicknessMM = volumeThickness;
                    info("Read Si thickness from detector volume for '{}': {} mm", m_cfg.readout,
                         m_cfg.detectorThicknessMM);
                } else {
                    applyDetectorFallbacks(m_cfg.readout, m_cfg);
                    info("Using hardcoded fallbacks for '{}': Si thickness={} mm", m_cfg.readout,
                         m_cfg.detectorThicknessMM);
                }
            }

        } catch (const std::exception& ex) {
            warning("Failed to derive segmentation for readout '{}': {}", m_cfg.readout, ex.what());
        }
    } else {
        warning("DD4hep detector unavailable; using configured geometry parameters only");
    }

    // Store decoder and field names for CellID decoding in process()
    m_decoder = decoder;
    if (m_cfg.segmentation.valid) {
        m_field_name_x = m_cfg.segmentation.fieldNameX;
        m_field_name_y = m_cfg.segmentation.fieldNameY;
    }

    // Log final configuration after DD4hep overrides
    info("Final geometry configuration:");
    info("  Pixel: size={} mm, pitch={} mm, offset={} mm", m_cfg.pixelSizeMM, m_cfg.pixelSpacingMM,
         m_cfg.gridOffsetMM);
    info("  Detector: size={} mm, Si thickness={} mm, pixel thickness={} mm", m_cfg.detectorSizeMM,
         m_cfg.detectorThicknessMM, m_cfg.pixelThicknessMM);

    // -----------------------------------------------------------------------
    // Set up bounds from segmentation config
    // -----------------------------------------------------------------------
    if (m_cfg.segmentation.valid) {
        m_bounds_x.minIndex = m_cfg.segmentation.minIndexX;
        m_bounds_x.maxIndex = m_cfg.segmentation.maxIndexX;
        m_bounds_y.minIndex = m_cfg.segmentation.minIndexY;
        m_bounds_y.maxIndex = m_cfg.segmentation.maxIndexY;

        if (!m_bounds_x.hasBounds() && m_cfg.segmentation.numCellsX > 0) {
            m_bounds_x.minIndex = m_cfg.segmentation.minIndexX;
            m_bounds_x.maxIndex = m_cfg.segmentation.minIndexX + m_cfg.segmentation.numCellsX - 1;
        }
        if (!m_bounds_y.hasBounds() && m_cfg.segmentation.numCellsY > 0) {
            m_bounds_y.minIndex = m_cfg.segmentation.minIndexY;
            m_bounds_y.maxIndex = m_cfg.segmentation.minIndexY + m_cfg.segmentation.numCellsY - 1;
        }

        if (m_cfg.pixelsPerSide <= 0 && m_cfg.segmentation.numCellsX > 0 &&
            m_cfg.segmentation.numCellsX == m_cfg.segmentation.numCellsY) {
            m_cfg.pixelsPerSide = m_cfg.segmentation.numCellsX;
        }
    } else {
        if (m_cfg.pixelsPerSide <= 0 && m_cfg.pixelSpacingMM > 0.0) {
            const double approxPixels = m_cfg.detectorSizeMM / m_cfg.pixelSpacingMM;
            m_cfg.pixelsPerSide = static_cast<int>(std::round(approxPixels));
            if (m_cfg.pixelsPerSide < 1) {
                m_cfg.pixelsPerSide = 1;
            }
        }

        const int halfGrid = m_cfg.pixelsPerSide / 2;
        const int minIdx = -halfGrid;
        const int maxIdx = m_cfg.pixelsPerSide - halfGrid - 1;
        m_bounds_x = IndexBounds{minIdx, maxIdx};
        m_bounds_y = IndexBounds{minIdx, maxIdx};
    }

    // Re-apply effective Y values after DD4hep override may have changed them
    if (m_cfg.pixelSizeYMM <= 0.0) {
        m_cfg.pixelSizeYMM = m_cfg.pixelSizeMM;
    }
    if (m_cfg.pixelSpacingYMM <= 0.0) {
        m_cfg.pixelSpacingYMM = m_cfg.pixelSpacingMM;
    }
}

// ---------------------------------------------------------------------------
// process() -- collection-level interface (algorithms::Algorithm contract).
// Iterates over SimTrackerHits, runs per-hit reconstruction, outputs TrackerHits.
// ---------------------------------------------------------------------------
void ChargeSharingReconstructor::process(
    const ChargeSharingReconAlgorithm::Input& input,
    const ChargeSharingReconAlgorithm::Output& output) const {

    const auto [sim_hits] = input;
    auto [rec_hits, assocs] = output;

    const bool useXZ = m_cfg.segmentation.useXZCoordinates;

    // Position error defaults: pitch^2/12 (uniform distribution)
    const double pitchX = m_cfg.pixelSpacingMM;
    const double pitchY = m_cfg.effectivePixelSpacingYMM();
    const double defaultVarX = (pitchX * pitchX) / 12.0;
    const double defaultVarY = (pitchY * pitchY) / 12.0;
    const double sensorT = m_cfg.detectorThicknessMM;
    const double varZ = (sensorT * sensorT) / 12.0;

    for (const auto& hit : *sim_hits) {
        const double edep = hit.getEDep();
        if (edep < static_cast<double>(m_cfg.minEDepGeV)) {
            continue;
        }

        SingleHitInput singleInput{};
        const auto pos = hit.getPosition();

        // For XZ segmentation (e.g., B0 tracker), map Z -> local Y
        if (useXZ) {
            singleInput.hitPositionMM = {pos.x, pos.z, pos.y};
        } else {
            singleInput.hitPositionMM = {pos.x, pos.y, pos.z};
        }

        singleInput.energyDepositGeV = edep;
        singleInput.cellID = hit.getCellID();

        // Decode CellID to grid indices directly (preferred path)
        if (m_decoder != nullptr) {
            const int idxI = static_cast<int>(m_decoder->get(hit.getCellID(), m_field_name_x));
            const int idxJ = static_cast<int>(m_decoder->get(hit.getCellID(), m_field_name_y));
            singleInput.pixelIndexHint = std::pair<int, int>{idxI, idxJ};
        } else if (m_converter != nullptr) {
            const auto center = m_converter->position(hit.getCellID());
            if (useXZ) {
                singleInput.pixelHintMM = std::array<double, 3>{center.x(), center.z(), center.y()};
            } else {
                singleInput.pixelHintMM = std::array<double, 3>{center.x(), center.y(), center.z()};
            }
        }

        const auto result = processSingleHit(singleInput);

        // Map reconstructed position back to global coordinates
        edm4hep::Vector3f reconPosition;
        if (useXZ) {
            reconPosition = edm4hep::Vector3f{
                static_cast<float>(result.reconstructedPositionMM[0]),
                static_cast<float>(result.reconstructedPositionMM[2]),
                static_cast<float>(result.reconstructedPositionMM[1])};
        } else {
            reconPosition = edm4hep::Vector3f{static_cast<float>(result.reconstructedPositionMM[0]),
                                              static_cast<float>(result.reconstructedPositionMM[1]),
                                              static_cast<float>(result.reconstructedPositionMM[2])};
        }

        // Compute position error (variance) from fit results
        double varLocalX = defaultVarX;
        double varLocalY = defaultVarY;

        if (result.fitRowX.converged && std::isfinite(result.fitRowX.muError) && result.fitRowX.muError > 0.0) {
            varLocalX = result.fitRowX.muError * result.fitRowX.muError;
        }
        if (result.fitColY.converged && std::isfinite(result.fitColY.muError) && result.fitColY.muError > 0.0) {
            varLocalY = result.fitColY.muError * result.fitColY.muError;
        }

        if (result.fit2D.converged) {
            if (std::isfinite(result.fit2D.muXError) && result.fit2D.muXError > 0.0) {
                varLocalX = result.fit2D.muXError * result.fit2D.muXError;
            }
            if (std::isfinite(result.fit2D.muYError) && result.fit2D.muYError > 0.0) {
                varLocalY = result.fit2D.muYError * result.fit2D.muYError;
            }
        }

        edm4eic::CovDiag3f posError;
        if (useXZ) {
            posError = edm4eic::CovDiag3f{
                static_cast<float>(varLocalX),
                static_cast<float>(varZ),
                static_cast<float>(varLocalY)};
        } else {
            posError = edm4eic::CovDiag3f{
                static_cast<float>(varLocalX),
                static_cast<float>(varLocalY),
                static_cast<float>(varZ)};
        }

        rec_hits->create(
            hit.getCellID(),          // cellID
            reconPosition,            // position
            posError,                 // positionError
            hit.getTime(),            // time
            0.0f,                     // timeError
            static_cast<float>(edep), // edep
            0.0f                      // edepError
        );

        // Create MC-truth association linking SimTrackerHit to output.
        // rawHit is left unset because charge sharing reconstruction bypasses
        // the digitization stage (SimTrackerHit -> TrackerHit directly, no RawTrackerHit).
        // The association and TrackerHit collections share the same index ordering,
        // enabling downstream matching by collection index.
        auto assoc = assocs->create();
        assoc.setSimHit(hit);
        assoc.setWeight(1.0f);
    }
}

// ---------------------------------------------------------------------------
// processSingleHit() -- per-hit charge sharing reconstruction.
// This is the original process() logic, now called from the collection loop.
// ---------------------------------------------------------------------------
ChargeSharingReconstructor::SingleHitResult
ChargeSharingReconstructor::processSingleHit(const SingleHitInput& input) const {
    SingleHitResult result{};

    // Find the center pixel
    PixelLocation nearest{};
    if (input.pixelIndexHint.has_value()) {
        const auto& [idxI, idxJ] = *input.pixelIndexHint;
        nearest = pixelLocationFromIndices(idxI, idxJ);
    } else if (input.pixelHintMM.has_value()) {
        nearest = findNearestPixelFallback(*input.pixelHintMM);
    } else {
        nearest = findNearestPixelFallback(input.hitPositionMM);
    }

    result.nearestPixelCenterMM = nearest.center;
    result.pixelIndexI = nearest.indexI;
    result.pixelIndexJ = nearest.indexJ;

    const double hitX = input.hitPositionMM[0];
    const double hitY = input.hitPositionMM[1];

    // Configure neighborhood calculation
    core::NeighborhoodConfig neighborCfg;
    neighborCfg.signalModel = m_cfg.signalModel;
    neighborCfg.activeMode = m_cfg.activePixelMode;
    neighborCfg.radius = m_cfg.neighborhoodRadius;
    neighborCfg.pixelSizeMM = m_cfg.pixelSizeMM;
    neighborCfg.pixelSizeYMM = m_cfg.pixelSizeYMM;
    neighborCfg.pixelSpacingMM = m_cfg.pixelSpacingMM;
    neighborCfg.pixelSpacingYMM = m_cfg.pixelSpacingYMM;
    neighborCfg.d0Micron = m_cfg.d0Micron;
    neighborCfg.betaPerMicron = m_cfg.linearBetaPerMicron;
    neighborCfg.numPixelsX = m_cfg.pixelsPerSide;
    neighborCfg.numPixelsY = m_cfg.pixelsPerSide;
    neighborCfg.minIndexX = m_bounds_x.hasBounds() ? m_bounds_x.minIndex : 0;
    neighborCfg.minIndexY = m_bounds_y.hasBounds() ? m_bounds_y.minIndex : 0;

    // Calculate charge fractions using core algorithm
    core::NeighborhoodResult neighborhood = core::calculateNeighborhood(
        hitX, hitY, nearest.indexI, nearest.indexJ, nearest.center[0], nearest.center[1], neighborCfg);

    // Convert energy deposit to charge
    const double edepEV = input.energyDepositGeV * 1.0e9;
    const double numElectrons = (m_cfg.ionizationEnergyEV > 0.0) ? (edepEV / m_cfg.ionizationEnergyEV) : 0.0;
    const double totalChargeElectrons = numElectrons * m_cfg.amplificationFactor;
    const double totalChargeCoulombs = totalChargeElectrons * m_cfg.elementaryChargeC;

    // Compute charges in the neighborhood and apply noise if enabled
    for (auto& pixel : neighborhood.pixels) {
        if (pixel.inBounds) {
            double chargeC = pixel.fraction * totalChargeCoulombs;
            if (m_cfg.noiseEnabled) {
                chargeC = m_noise_model.applyNoise(chargeC);
            }
            pixel.charge = chargeC;

            pixel.chargeRow = pixel.fractionRow * totalChargeCoulombs;
            pixel.chargeCol = pixel.fractionCol * totalChargeCoulombs;
            pixel.chargeBlock = pixel.fractionBlock * totalChargeCoulombs;
        }
    }

    // Populate neighbor data for output
    result.neighbors.reserve(neighborhood.pixels.size());
    for (const auto& pixel : neighborhood.pixels) {
        if (!pixel.inBounds)
            continue;

        NeighborData neighbor{};
        neighbor.fraction = pixel.fraction;
        neighbor.chargeC = pixel.charge;
        neighbor.distanceMM = pixel.distance;
        neighbor.alphaRad = pixel.alpha;
        neighbor.pixelXMM = pixel.centerX;
        neighbor.pixelYMM = pixel.centerY;
        neighbor.pixelId = pixel.globalIndex;
        neighbor.di = pixel.di;
        neighbor.dj = pixel.dj;

        neighbor.fractionRow = pixel.fractionRow;
        neighbor.fractionCol = pixel.fractionCol;
        neighbor.fractionBlock = pixel.fractionBlock;
        neighbor.chargeRowC = pixel.chargeRow;
        neighbor.chargeColC = pixel.chargeCol;
        neighbor.chargeBlockC = pixel.chargeBlock;

        result.neighbors.push_back(neighbor);
    }

    result.totalCollectedChargeC = totalChargeCoulombs;

    // Store diagnostic metadata
    result.truthPositionMM = input.hitPositionMM;
    result.inputEnergyDepositGeV = input.energyDepositGeV;
    result.inputCellID = input.cellID;
    result.neighborhoodRadius = m_cfg.neighborhoodRadius;
    result.neighborhoodGridSize = (2 * m_cfg.neighborhoodRadius + 1) * (2 * m_cfg.neighborhoodRadius + 1);

    // Compute summary statistics
    result.maxNeighborChargeC = 0.0;
    result.numActiveNeighbors = 0;
    for (const auto& n : result.neighbors) {
        if (n.chargeC > 0.0) {
            result.numActiveNeighbors++;
            if (n.chargeC > result.maxNeighborChargeC) {
                result.maxNeighborChargeC = n.chargeC;
            }
        }
    }

    // Reconstruct position using configured method
    const double centerZ = nearest.center[2];
    result.reconstructedPositionMM =
        reconstructPosition(neighborhood, centerZ, result.fitRowX, result.fitColY, result.fit2D);

    return result;
}

// ---------------------------------------------------------------------------
// Position reconstruction methods (unchanged physics logic)
// ---------------------------------------------------------------------------

std::array<double, 3> ChargeSharingReconstructor::reconstructPosition(const core::NeighborhoodResult& neighborhood,
                                                                      double centerZ, fit::GaussFit1DResult& fitRowX,
                                                                      fit::GaussFit1DResult& fitColY,
                                                                      fit::GaussFit2DResult& fit2D) const {
    switch (m_cfg.reconMethod) {
        case core::ReconMethod::Gaussian1D:
            return reconstructGaussian1D(neighborhood, centerZ, fitRowX, fitColY);
        case core::ReconMethod::Gaussian2D:
            return reconstructGaussian2D(neighborhood, centerZ, fit2D);
        case core::ReconMethod::Centroid:
        default:
            return reconstructCentroid(neighborhood, centerZ);
    }
}

std::array<double, 3> ChargeSharingReconstructor::reconstructCentroid(const core::NeighborhoodResult& neighborhood,
                                                                      double centerZ) const {
    double weightedX = 0.0;
    double weightedY = 0.0;
    double weightSum = 0.0;

    for (const auto& pixel : neighborhood.pixels) {
        if (!pixel.inBounds || pixel.charge <= 0.0)
            continue;
        weightSum += pixel.charge;
        weightedX += pixel.charge * pixel.centerX;
        weightedY += pixel.charge * pixel.centerY;
    }

    if (weightSum > 0.0) {
        return {weightedX / weightSum, weightedY / weightSum, centerZ};
    }

    return {neighborhood.centerPixelX, neighborhood.centerPixelY, centerZ};
}

std::array<double, 3> ChargeSharingReconstructor::reconstructGaussian1D(const core::NeighborhoodResult& neighborhood,
                                                                        double centerZ, fit::GaussFit1DResult& fitRowX,
                                                                        fit::GaussFit1DResult& fitColY) const {
    auto rowSlice = neighborhood.getCenterRow();
    auto colSlice = neighborhood.getCenterCol();

    std::vector<double> rowPositions, rowCharges;
    rowPositions.reserve(rowSlice.size());
    rowCharges.reserve(rowSlice.size());

    double maxRowCharge = 0.0;
    for (const auto* pixel : rowSlice) {
        if (pixel->charge > 0.0) {
            rowPositions.push_back(pixel->centerX);
            rowCharges.push_back(pixel->charge);
            maxRowCharge = std::max(maxRowCharge, pixel->charge);
        }
    }

    std::vector<double> colPositions, colCharges;
    colPositions.reserve(colSlice.size());
    colCharges.reserve(colSlice.size());

    double maxColCharge = 0.0;
    for (const auto* pixel : colSlice) {
        if (pixel->charge > 0.0) {
            colPositions.push_back(pixel->centerY);
            colCharges.push_back(pixel->charge);
            maxColCharge = std::max(maxColCharge, pixel->charge);
        }
    }

    const double muRangeX = m_cfg.muRangeMM();
    const double muRangeY = m_cfg.muRangeMM();

    fit::DistanceWeightedErrorConfig distErrCfg;
    distErrCfg.enabled = m_cfg.fitUseDistanceWeightedErrors;
    distErrCfg.scalePixels = m_cfg.fitDistanceScalePixels;
    distErrCfg.exponent = m_cfg.fitDistanceExponent;
    distErrCfg.floorPercent = m_cfg.fitDistanceFloorPercent;
    distErrCfg.capPercent = m_cfg.fitDistanceCapPercent;
    distErrCfg.powerInverse = m_cfg.fitDistancePowerInverse;
    distErrCfg.pixelSpacing = m_cfg.pixelSpacingMM;
    distErrCfg.truthCenterX = neighborhood.centerPixelX;
    distErrCfg.truthCenterY = neighborhood.centerPixelY;
    distErrCfg.preferTruthCenter = m_cfg.fitDistancePreferTruthCenter;

    fit::GaussFit1DConfig rowConfig;
    rowConfig.muLo = neighborhood.centerPixelX - muRangeX;
    rowConfig.muHi = neighborhood.centerPixelX + muRangeX;
    rowConfig.sigmaLo = m_cfg.sigmaLoBound();
    rowConfig.sigmaHi = m_cfg.sigmaHiBound();
    rowConfig.qMax = maxRowCharge;
    rowConfig.pixelSpacing = m_cfg.pixelSpacingMM;
    rowConfig.errorPercent = m_cfg.fitErrorPercentOfMax;
    rowConfig.distanceErrorConfig = distErrCfg;
    rowConfig.centerPosition = neighborhood.centerPixelX;

    fit::GaussFit1DConfig colConfig;
    colConfig.muLo = neighborhood.centerPixelY - muRangeY;
    colConfig.muHi = neighborhood.centerPixelY + muRangeY;
    colConfig.sigmaLo = m_cfg.sigmaLoBound();
    colConfig.sigmaHi = m_cfg.sigmaHiBound();
    colConfig.qMax = maxColCharge;
    colConfig.pixelSpacing = m_cfg.pixelSpacingYMM;
    colConfig.errorPercent = m_cfg.fitErrorPercentOfMax;
    colConfig.distanceErrorConfig = distErrCfg;
    colConfig.distanceErrorConfig.pixelSpacing = m_cfg.pixelSpacingYMM;
    colConfig.centerPosition = neighborhood.centerPixelY;

    fitRowX = fit::fitGaussian1D(rowPositions, rowCharges, rowConfig);
    fitColY = fit::fitGaussian1D(colPositions, colCharges, colConfig);

    double reconX = neighborhood.centerPixelX;
    double reconY = neighborhood.centerPixelY;

    if (fitRowX.converged && std::isfinite(fitRowX.mu)) {
        reconX = fitRowX.mu;
    } else {
        auto [centroidX, ok] = fit::weightedCentroid(rowPositions, rowCharges, 0.0);
        if (ok)
            reconX = centroidX;
    }

    if (fitColY.converged && std::isfinite(fitColY.mu)) {
        reconY = fitColY.mu;
    } else {
        auto [centroidY, ok] = fit::weightedCentroid(colPositions, colCharges, 0.0);
        if (ok)
            reconY = centroidY;
    }

    return {reconX, reconY, centerZ};
}

std::array<double, 3> ChargeSharingReconstructor::reconstructGaussian2D(const core::NeighborhoodResult& neighborhood,
                                                                        double centerZ,
                                                                        fit::GaussFit2DResult& fit2D) const {
    std::vector<double> xPositions, yPositions, charges;
    xPositions.reserve(neighborhood.pixels.size());
    yPositions.reserve(neighborhood.pixels.size());
    charges.reserve(neighborhood.pixels.size());

    double maxCharge = 0.0;
    for (const auto& pixel : neighborhood.pixels) {
        if (pixel.inBounds && pixel.charge > 0.0) {
            xPositions.push_back(pixel.centerX);
            yPositions.push_back(pixel.centerY);
            charges.push_back(pixel.charge);
            maxCharge = std::max(maxCharge, pixel.charge);
        }
    }

    const double muRangeX = m_cfg.muRangeMM();
    const double muRangeY = m_cfg.muRangeMM();

    fit::DistanceWeightedErrorConfig distErrCfg;
    distErrCfg.enabled = m_cfg.fitUseDistanceWeightedErrors;
    distErrCfg.scalePixels = m_cfg.fitDistanceScalePixels;
    distErrCfg.exponent = m_cfg.fitDistanceExponent;
    distErrCfg.floorPercent = m_cfg.fitDistanceFloorPercent;
    distErrCfg.capPercent = m_cfg.fitDistanceCapPercent;
    distErrCfg.powerInverse = m_cfg.fitDistancePowerInverse;
    distErrCfg.pixelSpacing = m_cfg.pixelSpacingMM;
    distErrCfg.truthCenterX = neighborhood.centerPixelX;
    distErrCfg.truthCenterY = neighborhood.centerPixelY;
    distErrCfg.preferTruthCenter = m_cfg.fitDistancePreferTruthCenter;

    fit::GaussFit2DConfig fitCfg;
    fitCfg.muXLo = neighborhood.centerPixelX - muRangeX;
    fitCfg.muXHi = neighborhood.centerPixelX + muRangeX;
    fitCfg.muYLo = neighborhood.centerPixelY - muRangeY;
    fitCfg.muYHi = neighborhood.centerPixelY + muRangeY;
    fitCfg.sigmaLo = m_cfg.sigmaLoBound();
    fitCfg.sigmaHi = m_cfg.sigmaHiBound();
    fitCfg.qMax = maxCharge;
    fitCfg.pixelSpacing = m_cfg.pixelSpacingMM;
    fitCfg.errorPercent = m_cfg.fitErrorPercentOfMax;
    fitCfg.distanceErrorConfig = distErrCfg;

    fit2D = fit::fitGaussian2D(xPositions, yPositions, charges, fitCfg);

    double reconX = neighborhood.centerPixelX;
    double reconY = neighborhood.centerPixelY;

    if (fit2D.converged && std::isfinite(fit2D.muX) && std::isfinite(fit2D.muY)) {
        reconX = fit2D.muX;
        reconY = fit2D.muY;
    } else {
        return reconstructCentroid(neighborhood, centerZ);
    }

    return {reconX, reconY, centerZ};
}

ChargeSharingReconstructor::PixelLocation
ChargeSharingReconstructor::findNearestPixelFallback(const std::array<double, 3>& positionMM) const {
    const double offsetX = m_cfg.effectiveGridOffsetXMM();
    const double offsetY = m_cfg.effectiveGridOffsetYMM();

    int i = ChargeSharingConfig::positionToIndex(positionMM[0], m_cfg.pixelSpacingMM, offsetX);
    int j = ChargeSharingConfig::positionToIndex(positionMM[1], m_cfg.effectivePixelSpacingYMM(), offsetY);

    const int defaultMin = 0;
    const int defaultMax = std::max(0, m_cfg.pixelsPerSide - 1);
    const int minI = m_bounds_x.hasBounds() ? m_bounds_x.minIndex : defaultMin;
    const int maxI = m_bounds_x.hasBounds() ? m_bounds_x.maxIndex : defaultMax;
    const int minJ = m_bounds_y.hasBounds() ? m_bounds_y.minIndex : defaultMin;
    const int maxJ = m_bounds_y.hasBounds() ? m_bounds_y.maxIndex : defaultMax;

    i = std::clamp(i, minI, maxI);
    j = std::clamp(j, minJ, maxJ);

    return pixelLocationFromIndices(i, j);
}

ChargeSharingReconstructor::PixelLocation ChargeSharingReconstructor::pixelLocationFromIndices(int indexI,
                                                                                               int indexJ) const {
    PixelLocation loc{};
    loc.indexI = indexI;
    loc.indexJ = indexJ;

    const double pixelCenterZ =
        m_cfg.detectorZCenterMM + m_cfg.detectorThicknessMM / 2.0 + m_cfg.pixelThicknessMM / 2.0;

    const double pitchX = m_cfg.pixelSpacingMM;
    const double pitchY = m_cfg.effectivePixelSpacingYMM();
    const double offsetX = m_cfg.effectiveGridOffsetXMM();
    const double offsetY = m_cfg.effectiveGridOffsetYMM();

    loc.center = {ChargeSharingConfig::indexToPosition(indexI, pitchX, offsetX),
                  ChargeSharingConfig::indexToPosition(indexJ, pitchY, offsetY), pixelCenterZ};

    return loc;
}

bool ChargeSharingReconstructor::isPixelIndexInBounds(int indexI, int indexJ) const {
    return m_bounds_x.contains(indexI) && m_bounds_y.contains(indexJ);
}

} // namespace eicrecon
