// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2024-2026 Tom Bleher, Igor Korover

#include "LGADChargeSharingRecon.h"

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

constexpr double kElementaryChargeC = 1.602176634e-19;

/// Detector-specific DD4hep constants. Used only when segmentation does not
/// supply the value directly (e.g. sensor thickness). Must stay in sync with
/// the compact XML under `eic/epic`.
struct DetectorConstants {
    std::string siliconThickness;
    std::string detectorSize;
    std::string pixelSize;
    std::string copperThickness;
};

DetectorConstants getDetectorConstants(const std::string& readout) {
    DetectorConstants constants;
    if (readout.find("LumiSpec") != std::string::npos || readout.find("Lumi") != std::string::npos) {
        constants.siliconThickness = "LumiSpecTracker_Si_DZ";
        constants.detectorSize = "LumiSpecTracker_DXY";
        constants.pixelSize = "LumiSpecTracker_pixelSize";
        constants.copperThickness = "LumiSpecTracker_Cu_DZ";
    } else if (readout.find("B0") != std::string::npos) {
        // No named constants; use DD4hep volume dimensions / detector defaults.
    }
    return constants;
}

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
    }
    return 0.0;
}

void applyDetectorFallbacks(const std::string& readout, eicrecon::LGADChargeSharingRecon::Geometry& geom) {
    if (readout.find("B0") != std::string::npos) {
        geom.detectorThicknessMM = 0.3;
    }
}

} // namespace

namespace eicrecon {

namespace core = ::chargesharing::core;
namespace fit = ::chargesharing::fit;

// ---------------------------------------------------------------------------
// init()
// ---------------------------------------------------------------------------
void LGADChargeSharingRecon::init() {
    if (m_cfg.neighborhoodRadius < 0) {
        m_cfg.neighborhoodRadius = 0;
    }

    // Noise model setup
    core::NoiseConfig noiseConfig;
    noiseConfig.enabled = m_cfg.noiseEnabled;
    noiseConfig.gainSigmaMin = 0.01;
    noiseConfig.gainSigmaMax = 0.05;
    noiseConfig.electronNoiseCount = m_cfg.noiseElectronCount;
    noiseConfig.elementaryCharge = kElementaryChargeC;
    m_noise_model.setConfig(noiseConfig);

    if (m_skip_dd4hep_init) {
        info("LGADChargeSharingRecon: DD4hep init skipped (test geometry injected)");
        return;
    }

    // ------------------------------------------------------------------
    // DD4hep geometry lookup via algorithms::GeoSvc singleton.
    // Everything we need (pitch, pad size, offset, cell index bounds,
    // sensor thickness) comes from the readout segmentation + compact-XML
    // constants. The user does not configure these.
    // ------------------------------------------------------------------
    const auto& geo = algorithms::GeoSvc::instance();
    const dd4hep::Detector* detector = geo.detector();
    m_converter = geo.cellIDPositionConverter();

    if (!detector) {
        warning("DD4hep detector unavailable; LGADChargeSharingRecon will run with defaults");
        return;
    }

    try {
        dd4hep::Readout readout = detector->readout(m_cfg.readout);
        dd4hep::Segmentation segmentation = readout.segmentation();

        if (!segmentation.isValid()) {
            warning("Readout '{}' has no valid segmentation", m_cfg.readout);
        } else {
            const auto* segImplXY =
                dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation.segmentation());
            const dd4hep::DDSegmentation::CartesianGridXZ* segImplXZ = nullptr;
            if (segImplXY == nullptr) {
                segImplXZ =
                    dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXZ*>(segmentation.segmentation());
                m_geom.useXZCoordinates = (segImplXZ != nullptr);
            }

            if (segImplXY == nullptr && segImplXZ == nullptr) {
                warning("Segmentation for readout '{}' is '{}'; expected CartesianGridXY or CartesianGridXZ",
                        m_cfg.readout, segmentation.type());
            } else {
                m_decoder = segmentation.decoder();

                if (segImplXY != nullptr) {
                    m_geom.pixelSpacingXMM = segImplXY->gridSizeX() / dd4hep::mm;
                    m_geom.pixelSpacingYMM = segImplXY->gridSizeY() / dd4hep::mm;
                    m_geom.gridOffsetXMM = segImplXY->offsetX() / dd4hep::mm;
                    m_geom.gridOffsetYMM = segImplXY->offsetY() / dd4hep::mm;
                    m_geom.fieldNameX = segImplXY->fieldNameX();
                    m_geom.fieldNameY = segImplXY->fieldNameY();
                } else {
                    m_geom.pixelSpacingXMM = segImplXZ->gridSizeX() / dd4hep::mm;
                    m_geom.pixelSpacingYMM = segImplXZ->gridSizeZ() / dd4hep::mm;
                    m_geom.gridOffsetXMM = segImplXZ->offsetX() / dd4hep::mm;
                    m_geom.gridOffsetYMM = segImplXZ->offsetZ() / dd4hep::mm;
                    m_geom.fieldNameX = segImplXZ->fieldNameX();
                    m_geom.fieldNameY = segImplXZ->fieldNameZ();
                }

                if (m_decoder != nullptr) {
                    const auto& fx = (*m_decoder)[m_geom.fieldNameX];
                    const auto& fy = (*m_decoder)[m_geom.fieldNameY];
                    m_geom.minIndexX = fx.minValue();
                    m_geom.maxIndexX = fx.maxValue();
                    m_geom.minIndexY = fy.minValue();
                    m_geom.maxIndexY = fy.maxValue();
                    if (fx.maxValue() >= fx.minValue() && fy.maxValue() >= fy.minValue() &&
                        fx.maxValue() - fx.minValue() == fy.maxValue() - fy.minValue()) {
                        m_geom.pixelsPerSide = fx.maxValue() - fx.minValue() + 1;
                    }
                } else {
                    warning("Segmentation for readout '{}' lacks a BitField decoder; neighbour bounds unavailable",
                            m_cfg.readout);
                }

                // Cell dimensions (pad size) - DD4hep may report these too.
                const auto dims =
                    segImplXY ? segImplXY->cellDimensions(0) : segImplXZ->cellDimensions(0);
                if (dims.size() >= 1 && dims[0] > 0.0) {
                    m_geom.pixelSizeXMM = dims[0] / dd4hep::mm;
                }
                if (dims.size() >= 2 && dims[1] > 0.0) {
                    m_geom.pixelSizeYMM = dims[1] / dd4hep::mm;
                } else {
                    m_geom.pixelSizeYMM = m_geom.pixelSizeXMM;
                }

                info("Using DD4hep {} segmentation for readout '{}': pitch=({}, {}) mm, "
                     "offset=({}, {}) mm, cells/side={}",
                     m_geom.useXZCoordinates ? "CartesianGridXZ" : "CartesianGridXY", m_cfg.readout,
                     m_geom.pixelSpacingXMM, m_geom.pixelSpacingYMM, m_geom.gridOffsetXMM,
                     m_geom.gridOffsetYMM, m_geom.pixelsPerSide);
            }
        }

        // Optional per-detector constants for sensor thickness / pad size overrides.
        const auto detConstants = getDetectorConstants(m_cfg.readout);
        if (!detConstants.siliconThickness.empty()) {
            try {
                m_geom.detectorThicknessMM =
                    detector->constantAsDouble(detConstants.siliconThickness) / dd4hep::mm;
            } catch (const std::exception&) {
            }
        }
        if (!detConstants.pixelSize.empty()) {
            try {
                double p = detector->constantAsDouble(detConstants.pixelSize) / dd4hep::mm;
                if (p > 0.0) {
                    m_geom.pixelSizeXMM = p;
                    m_geom.pixelSizeYMM = p;
                }
            } catch (const std::exception&) {
            }
        }
        if (!detConstants.copperThickness.empty()) {
            try {
                m_geom.pixelThicknessMM =
                    detector->constantAsDouble(detConstants.copperThickness) / dd4hep::mm;
            } catch (const std::exception&) {
            }
        }

        if (detConstants.siliconThickness.empty() && detConstants.pixelSize.empty()) {
            double volumeThickness = getSensorThicknessFromReadout(detector, m_cfg.readout);
            if (volumeThickness > 0.0 && volumeThickness < 10.0) {
                m_geom.detectorThicknessMM = volumeThickness;
            } else {
                applyDetectorFallbacks(m_cfg.readout, m_geom);
            }
        }
    } catch (const std::exception& ex) {
        warning("Failed to derive segmentation for readout '{}': {}", m_cfg.readout, ex.what());
    }

    info("Final geometry: pitch=({}, {}) mm, pad=({}, {}) mm, Si thickness={} mm, pixels/side={}",
         m_geom.pixelSpacingXMM, m_geom.pixelSpacingYMM, m_geom.pixelSizeXMM, m_geom.pixelSizeYMM,
         m_geom.detectorThicknessMM, m_geom.pixelsPerSide);
}

// ---------------------------------------------------------------------------
// process()
// ---------------------------------------------------------------------------
void LGADChargeSharingRecon::process(const Input& input, const Output& output) const {
    const auto [sim_hits] = input;
    auto [rec_hits, assocs] = output;

    const bool useXZ = m_geom.useXZCoordinates;
    const double pitchX = m_geom.pixelSpacingXMM;
    const double pitchY = m_geom.pixelSpacingYMM;
    const double defaultVarX = (pitchX * pitchX) / 12.0;
    const double defaultVarY = (pitchY * pitchY) / 12.0;
    const double sensorT = m_geom.detectorThicknessMM;
    const double varZ = (sensorT * sensorT) / 12.0;

    for (const auto& hit : *sim_hits) {
        const double edep = hit.getEDep();
        if (edep < static_cast<double>(m_cfg.minEDepGeV)) {
            continue;
        }

        SingleHitInput singleInput{};
        const auto pos = hit.getPosition();
        if (useXZ) {
            // CartesianGridXZ: map (x, y, z) -> local (x, z, y) so that the
            // algorithm's local Y is the detector's Z coordinate.
            singleInput.hitPositionMM = {pos.x, pos.z, pos.y};
        } else {
            singleInput.hitPositionMM = {pos.x, pos.y, pos.z};
        }
        singleInput.energyDepositGeV = edep;
        singleInput.cellID = hit.getCellID();

        if (m_decoder != nullptr) {
            const int idxI = static_cast<int>(m_decoder->get(hit.getCellID(), m_geom.fieldNameX));
            const int idxJ = static_cast<int>(m_decoder->get(hit.getCellID(), m_geom.fieldNameY));
            singleInput.pixelIndexHint = std::pair<int, int>{idxI, idxJ};
        } else if (m_converter != nullptr) {
            const auto center = m_converter->position(hit.getCellID());
            if (useXZ) {
                singleInput.pixelHintMM = std::array<double, 3>{center.x(), center.z(), center.y()};
            } else {
                singleInput.pixelHintMM = std::array<double, 3>{center.x(), center.y(), center.z()};
            }
        }

        const auto result = processSingleHitImpl(singleInput);

        // Local sub-pad offset; apply to global cell-center position to get the
        // global reconstructed coordinates.
        const double dxLocal = result.reconstructedPositionMM[0] - result.nearestPixelCenterMM[0];
        const double dyLocal = result.reconstructedPositionMM[1] - result.nearestPixelCenterMM[1];

        edm4hep::Vector3f reconPosition;
        if (m_converter != nullptr) {
            const auto globalCenter = m_converter->position(hit.getCellID());
            const double gx = globalCenter.x() / dd4hep::mm;
            const double gy = globalCenter.y() / dd4hep::mm;
            const double gz = globalCenter.z() / dd4hep::mm;
            if (useXZ) {
                reconPosition = edm4hep::Vector3f{static_cast<float>(gx + dxLocal),
                                                   static_cast<float>(gy),
                                                   static_cast<float>(gz + dyLocal)};
            } else {
                reconPosition = edm4hep::Vector3f{static_cast<float>(gx + dxLocal),
                                                   static_cast<float>(gy + dyLocal),
                                                   static_cast<float>(gz)};
            }
        } else {
            if (useXZ) {
                reconPosition = edm4hep::Vector3f{static_cast<float>(result.reconstructedPositionMM[0]),
                                                   static_cast<float>(result.reconstructedPositionMM[2]),
                                                   static_cast<float>(result.reconstructedPositionMM[1])};
            } else {
                reconPosition = edm4hep::Vector3f{static_cast<float>(result.reconstructedPositionMM[0]),
                                                   static_cast<float>(result.reconstructedPositionMM[1]),
                                                   static_cast<float>(result.reconstructedPositionMM[2])};
            }
        }

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
            posError = edm4eic::CovDiag3f{static_cast<float>(varLocalX), static_cast<float>(varZ),
                                          static_cast<float>(varLocalY)};
        } else {
            posError = edm4eic::CovDiag3f{static_cast<float>(varLocalX), static_cast<float>(varLocalY),
                                          static_cast<float>(varZ)};
        }

        rec_hits->create(hit.getCellID(), reconPosition, posError, hit.getTime(), 0.0f,
                         static_cast<float>(edep), 0.0f);

        auto assoc = assocs->create();
        assoc.setSimHit(hit);
        assoc.setWeight(1.0f);
    }
}

// ---------------------------------------------------------------------------
// Per-hit processing (public for tests).
// ---------------------------------------------------------------------------
LGADChargeSharingRecon::SingleHitResult
LGADChargeSharingRecon::processSingleHit(const SingleHitInput& input) const {
    return processSingleHitImpl(input);
}

LGADChargeSharingRecon::SingleHitResult
LGADChargeSharingRecon::processSingleHitImpl(const SingleHitInput& input) const {
    SingleHitResult result{};

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
    result.pixelRowIndex = nearest.indexI;
    result.pixelColIndex = nearest.indexJ;

    const double hitX = input.hitPositionMM[0];
    const double hitY = input.hitPositionMM[1];

    core::NeighborhoodConfig neighborCfg;
    neighborCfg.signalModel = m_cfg.signalModel;
    neighborCfg.activeMode = m_cfg.activePixelMode;
    neighborCfg.radius = m_cfg.neighborhoodRadius;
    neighborCfg.pixelSizeMM = m_geom.pixelSizeXMM;
    neighborCfg.pixelSizeYMM = m_geom.pixelSizeYMM;
    neighborCfg.pixelSpacingMM = m_geom.pixelSpacingXMM;
    neighborCfg.pixelSpacingYMM = m_geom.pixelSpacingYMM;
    neighborCfg.d0Micron = m_cfg.d0Micron;
    neighborCfg.betaPerMicron = m_cfg.linearBetaPerMicron;
    neighborCfg.numPixelsX = m_geom.pixelsPerSide;
    neighborCfg.numPixelsY = m_geom.pixelsPerSide;
    neighborCfg.minIndexX = m_geom.hasBoundsX() ? m_geom.minIndexX : 0;
    neighborCfg.minIndexY = m_geom.hasBoundsY() ? m_geom.minIndexY : 0;

    core::NeighborhoodResult neighborhood = core::calculateNeighborhood(
        hitX, hitY, nearest.indexI, nearest.indexJ, nearest.center[0], nearest.center[1], neighborCfg);

    const double edepEV = input.energyDepositGeV * 1.0e9;
    const double numElectrons = (m_cfg.ionizationEnergyEV > 0.0) ? (edepEV / m_cfg.ionizationEnergyEV) : 0.0;
    const double totalChargeElectrons = numElectrons * m_cfg.amplificationFactor;
    const double totalChargeCoulombs = totalChargeElectrons * kElementaryChargeC;

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

    result.neighbors.reserve(neighborhood.pixels.size());
    for (const auto& pixel : neighborhood.pixels) {
        if (!pixel.inBounds)
            continue;
        NeighborData n{};
        n.fraction = pixel.fraction;
        n.chargeC = pixel.charge;
        n.distanceMM = pixel.distance;
        n.alphaRad = pixel.alpha;
        n.pixelXMM = pixel.centerX;
        n.pixelYMM = pixel.centerY;
        n.pixelId = pixel.globalIndex;
        n.di = pixel.di;
        n.dj = pixel.dj;
        result.neighbors.push_back(n);
    }

    result.totalCollectedChargeC = totalChargeCoulombs;
    result.truthPositionMM = input.hitPositionMM;
    result.inputEnergyDepositGeV = input.energyDepositGeV;
    result.inputCellID = input.cellID;
    result.neighborhoodRadius = m_cfg.neighborhoodRadius;
    result.neighborhoodGridSize =
        (2 * m_cfg.neighborhoodRadius + 1) * (2 * m_cfg.neighborhoodRadius + 1);

    result.maxNeighborChargeC = 0.0;
    result.numActiveNeighbors = 0;
    for (const auto& n : result.neighbors) {
        if (n.chargeC > 0.0) {
            ++result.numActiveNeighbors;
            if (n.chargeC > result.maxNeighborChargeC) {
                result.maxNeighborChargeC = n.chargeC;
            }
        }
    }

    const double centerZ = nearest.center[2];
    result.reconstructedPositionMM =
        reconstructPosition(neighborhood, centerZ, result.fitRowX, result.fitColY, result.fit2D);

    return result;
}

// ---------------------------------------------------------------------------
// Position reconstruction
// ---------------------------------------------------------------------------
std::array<double, 3>
LGADChargeSharingRecon::reconstructPosition(const core::NeighborhoodResult& neighborhood, double centerZ,
                                            fit::GaussFit1DResult& fitRowX, fit::GaussFit1DResult& fitColY,
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

std::array<double, 3>
LGADChargeSharingRecon::reconstructCentroid(const core::NeighborhoodResult& neighborhood,
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

std::array<double, 3>
LGADChargeSharingRecon::reconstructGaussian1D(const core::NeighborhoodResult& neighborhood, double centerZ,
                                              fit::GaussFit1DResult& fitRowX,
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

    fit::GaussFit1DConfig rowConfig;
    rowConfig.muLo = neighborhood.centerPixelX - muRangeXMM();
    rowConfig.muHi = neighborhood.centerPixelX + muRangeXMM();
    rowConfig.sigmaLo = sigmaLoBoundX();
    rowConfig.sigmaHi = sigmaHiBoundX();
    rowConfig.qMax = maxRowCharge;
    rowConfig.pixelSpacing = m_geom.pixelSpacingXMM;
    rowConfig.errorPercent = kFitErrorPercentOfMax;
    rowConfig.centerPosition = neighborhood.centerPixelX;

    fit::GaussFit1DConfig colConfig;
    colConfig.muLo = neighborhood.centerPixelY - muRangeYMM();
    colConfig.muHi = neighborhood.centerPixelY + muRangeYMM();
    colConfig.sigmaLo = sigmaLoBoundY();
    colConfig.sigmaHi = sigmaHiBoundY();
    colConfig.qMax = maxColCharge;
    colConfig.pixelSpacing = m_geom.pixelSpacingYMM;
    colConfig.errorPercent = kFitErrorPercentOfMax;
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

std::array<double, 3>
LGADChargeSharingRecon::reconstructGaussian2D(const core::NeighborhoodResult& neighborhood, double centerZ,
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

    fit::GaussFit2DConfig fitCfg;
    fitCfg.muXLo = neighborhood.centerPixelX - muRangeXMM();
    fitCfg.muXHi = neighborhood.centerPixelX + muRangeXMM();
    fitCfg.muYLo = neighborhood.centerPixelY - muRangeYMM();
    fitCfg.muYHi = neighborhood.centerPixelY + muRangeYMM();
    fitCfg.sigmaLo = std::min(sigmaLoBoundX(), sigmaLoBoundY());
    fitCfg.sigmaHi = std::max(sigmaHiBoundX(), sigmaHiBoundY());
    fitCfg.qMax = maxCharge;
    fitCfg.pixelSpacing = std::min(m_geom.pixelSpacingXMM, m_geom.pixelSpacingYMM);
    fitCfg.errorPercent = kFitErrorPercentOfMax;

    fit2D = fit::fitGaussian2D(xPositions, yPositions, charges, fitCfg);

    if (fit2D.converged && std::isfinite(fit2D.muX) && std::isfinite(fit2D.muY)) {
        return {fit2D.muX, fit2D.muY, centerZ};
    }
    return reconstructCentroid(neighborhood, centerZ);
}

// ---------------------------------------------------------------------------
// Pixel location helpers
// ---------------------------------------------------------------------------
LGADChargeSharingRecon::PixelLocation
LGADChargeSharingRecon::findNearestPixelFallback(const std::array<double, 3>& positionMM) const {
    int i = positionToIndex(positionMM[0], m_geom.pixelSpacingXMM, m_geom.gridOffsetXMM);
    int j = positionToIndex(positionMM[1], m_geom.pixelSpacingYMM, m_geom.gridOffsetYMM);

    const int defaultMin = 0;
    const int defaultMax = std::max(0, m_geom.pixelsPerSide - 1);
    const int minI = m_geom.hasBoundsX() ? m_geom.minIndexX : defaultMin;
    const int maxI = m_geom.hasBoundsX() ? m_geom.maxIndexX : defaultMax;
    const int minJ = m_geom.hasBoundsY() ? m_geom.minIndexY : defaultMin;
    const int maxJ = m_geom.hasBoundsY() ? m_geom.maxIndexY : defaultMax;

    i = std::clamp(i, minI, maxI);
    j = std::clamp(j, minJ, maxJ);
    return pixelLocationFromIndices(i, j);
}

LGADChargeSharingRecon::PixelLocation
LGADChargeSharingRecon::pixelLocationFromIndices(int indexI, int indexJ) const {
    PixelLocation loc{};
    loc.indexI = indexI;
    loc.indexJ = indexJ;
    const double pixelCenterZ =
        m_geom.detectorZCenterMM + m_geom.detectorThicknessMM / 2.0 + m_geom.pixelThicknessMM / 2.0;
    loc.center = {indexToPosition(indexI, m_geom.pixelSpacingXMM, m_geom.gridOffsetXMM),
                  indexToPosition(indexJ, m_geom.pixelSpacingYMM, m_geom.gridOffsetYMM), pixelCenterZ};
    return loc;
}

} // namespace eicrecon
