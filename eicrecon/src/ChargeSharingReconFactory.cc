// Include DD4hep headers first to resolve forward declarations
#include <DD4hep/Detector.h>
#include <DD4hep/Readout.h>
#include <DD4hep/Segmentations.h>
#include <DD4hep/Volumes.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <DDSegmentation/CartesianGridXZ.h>
#include <DDSegmentation/BitFieldCoder.h>
#include <Evaluator/DD4hepUnits.h>

#include "ChargeSharingReconFactory.h"

#include <edm4hep/Vector3f.h>
#include <edm4eic/CovDiag3f.h>

#include <array>
#include <exception>
#include <utility>
#include <vector>

namespace {

/// Helper struct for DD4hep constant lookups
struct DetectorConstants {
  std::string siliconThickness;   // e.g., "LumiSpecTracker_Si_DZ"
  std::string detectorSize;       // e.g., "LumiSpecTracker_DXY"
  std::string pixelSize;          // e.g., "LumiSpecTracker_pixelSize"
  std::string copperThickness;    // e.g., "LumiSpecTracker_Cu_DZ" (optional, for pixel backing)
};

/// Map readout names to their DD4hep constant prefixes
DetectorConstants getDetectorConstants(const std::string& readout) {
  DetectorConstants constants;

  if (readout.find("LumiSpec") != std::string::npos ||
      readout.find("Lumi") != std::string::npos) {
    // Luminosity Spectrometer Tracker
    constants.siliconThickness = "LumiSpecTracker_Si_DZ";
    constants.detectorSize = "LumiSpecTracker_DXY";
    constants.pixelSize = "LumiSpecTracker_pixelSize";
    constants.copperThickness = "LumiSpecTracker_Cu_DZ";
  } else if (readout.find("B0") != std::string::npos) {
    // B0 Tracker (far forward)
    // B0 defines geometry inline in B0_tracker.xml, not as named constants
    // Constants left empty - will use hardcoded fallbacks below
    constants.siliconThickness = "";
    constants.detectorSize = "";
    constants.pixelSize = "";
    constants.copperThickness = "";
  } else if (readout.find("TOF") != std::string::npos ||
             readout.find("Tof") != std::string::npos) {
    // Time-of-Flight detectors (also use AC-LGADs)
    constants.siliconThickness = "TOFBarrel_sensor_thickness";
    constants.detectorSize = "";
    constants.pixelSize = "";
  }

  return constants;
}

/// Try to extract sensor thickness from the sensitive detector's volume
/// Returns thickness in mm, or 0 if not found
double getSensorThicknessFromReadout(const dd4hep::Detector* detector, const std::string& readoutName) {
  if (!detector) return 0.0;

  try {
    dd4hep::Readout readout = detector->readout(readoutName);
    if (!readout.isValid()) return 0.0;

    // Get the sensitive detector's ID descriptor which links to the detector element
    dd4hep::IDDescriptor idDesc = readout.idSpec();
    if (!idDesc.isValid()) return 0.0;

    // Try to find the detector element by looking for common naming patterns
    // The readout name often corresponds to or contains the detector name
    std::string detName = readoutName;
    // Remove common suffixes like "Hits"
    if (detName.size() > 4 && detName.substr(detName.size() - 4) == "Hits") {
      detName = detName.substr(0, detName.size() - 4);
    }

    dd4hep::DetElement det = detector->detector(detName);
    if (!det.isValid()) {
      // Try with the full readout name
      det = detector->detector(readoutName);
    }

    if (det.isValid() && det.placement().isValid()) {
      dd4hep::Volume vol = det.placement().volume();
      if (vol.isValid() && vol.solid().isValid()) {
        // Get dimensions from the solid - for a Box this returns [dX, dY, dZ] (half-lengths)
        auto dims = vol.solid().dimensions();
        if (dims.size() >= 3) {
          // Return the Z dimension (full thickness = 2 * half-length)
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
void applyDetectorFallbacks(const std::string& readout, epic::chargesharing::ChargeSharingConfig& cfg) {
  if (readout.find("B0") != std::string::npos) {
    // B0 Tracker values from epic/compact/far_forward/B0_tracker.xml:
    // - Silicon (SiliconOxide sensitive): 0.3 mm
    // - Grid size from segmentation: 0.070 mm (read automatically)
    // Note: Copper backing (0.715 mm) is mechanical support, not used in charge sharing
    cfg.detectorThicknessMM = 0.3;    // SiliconOxide layer
    // detectorSizeMM not applicable - B0 uses trapezoidal modules
  }
}

}  // anonymous namespace

namespace epic::chargesharing {

namespace {

// Convert integer parameter to SignalModel enum
SignalModel toSignalModel(int value) {
  switch (value) {
    case 1: return SignalModel::LinA;
    case 0:
    default: return SignalModel::LogA;
  }
}

// Convert integer parameter to ActivePixelMode enum
ActivePixelMode toActivePixelMode(int value) {
  switch (value) {
    case 1: return ActivePixelMode::RowCol;
    case 2: return ActivePixelMode::RowCol3x3;
    case 3: return ActivePixelMode::ChargeBlock2x2;
    case 4: return ActivePixelMode::ChargeBlock3x3;
    case 0:
    default: return ActivePixelMode::Neighborhood;
  }
}

// Convert integer parameter to ReconMethod enum
core::ReconMethod toReconMethod(int value) {
  switch (value) {
    case 0: return core::ReconMethod::Centroid;
    case 2: return core::ReconMethod::Gaussian2D;
    case 1:
    default: return core::ReconMethod::Gaussian1D;
  }
}

}  // namespace

void ChargeSharingReconFactory::Configure() {
  ChargeSharingConfig cfg{};

  // Mode selection (convert integer parameters to enums)
  cfg.signalModel = toSignalModel(m_signalModelValue);
  cfg.activePixelMode = toActivePixelMode(m_activePixelModeValue);
  cfg.reconMethod = toReconMethod(m_reconMethodValue);

  // DD4hep integration
  cfg.readout = m_readout();
  cfg.minEDepGeV = m_minEDep();

  // Detector geometry
  cfg.neighborhoodRadius = m_neighborhoodRadius();
  cfg.pixelSizeMM = m_pixelSize();
  cfg.pixelSizeYMM = m_pixelSize();  // Same as X by default
  cfg.pixelSpacingMM = m_pixelSpacing();
  cfg.pixelSpacingYMM = m_pixelSpacing();  // Same as X by default
  cfg.gridOffsetMM = m_gridOffset();
  cfg.gridOffsetYMM = m_gridOffset();  // Same as X by default
  cfg.detectorSizeMM = m_detectorSize();
  cfg.detectorThicknessMM = m_detectorThickness();
  cfg.pixelThicknessMM = m_pixelThickness();
  cfg.detectorZCenterMM = m_detectorZCenter();
  cfg.pixelsPerSide = m_pixelsPerSide();

  // Physics
  cfg.ionizationEnergyEV = m_ionizationEnergy();
  cfg.amplificationFactor = m_amplificationFactor();
  cfg.elementaryChargeC = m_elementaryCharge();
  cfg.d0Micron = m_d0Micron();
  cfg.linearBetaPerMicron = m_linearBeta();

  // Fitting
  cfg.fitErrorPercentOfMax = m_fitErrorPercent();
  cfg.fitUseVerticalUncertainties = m_fitUseVerticalUncertainties();

  // Diagnostics
  cfg.emitNeighborDiagnostics = m_emitNeighborDiagnostics();

  // Log initial configuration (will be updated with DD4hep values below)
  if (auto log = logger()) {
    const char* modelNames[] = {"LogA", "LinA"};
    const char* activeNames[] = {"Neighborhood", "RowCol", "RowCol3x3", "ChargeBlock2x2", "ChargeBlock3x3"};
    const char* reconNames[] = {"Centroid", "Gaussian1D", "Gaussian2D"};

    log->info("ChargeSharingRecon configuration (defaults, may be overridden by DD4hep):");
    log->info("  Readout: {}", cfg.readout);
    log->info("  Signal model: {}", modelNames[static_cast<int>(cfg.signalModel)]);
    log->info("  Active pixel mode: {}", activeNames[static_cast<int>(cfg.activePixelMode)]);
    log->info("  Reconstruction method: {}", reconNames[static_cast<int>(cfg.reconMethod)]);
    log->info("  Neighborhood radius: {} ({}x{} grid)", cfg.neighborhoodRadius,
              2*cfg.neighborhoodRadius+1, 2*cfg.neighborhoodRadius+1);
    log->info("  Physics: d0={} um, gain={}, ionization={} eV",
              cfg.d0Micron, cfg.amplificationFactor, cfg.ionizationEnergyEV);
  }

  const dd4hep::DDSegmentation::CartesianGridXY* segImplXY = nullptr;
  const dd4hep::DDSegmentation::CartesianGridXZ* segImplXZ = nullptr;
  const dd4hep::DDSegmentation::BitFieldCoder* decoder = nullptr;
  bool usingXZCoordinates = false;

  auto& dd4hep_service = m_dd4hep();
  const dd4hep::Detector* detector = dd4hep_service.detector();
  if (detector) {
    try {
      dd4hep::Readout readout = detector->readout(cfg.readout);
      dd4hep::Segmentation segmentation = readout.segmentation();

      if (!segmentation.isValid()) {
        if (auto log = logger()) {
          log->warn("Readout '{}' has no valid segmentation; falling back to manual geometry", cfg.readout);
        }
      } else {
        // Try CartesianGridXY first
        segImplXY = dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation.segmentation());

        // If not XY, try CartesianGridXZ (used by B0 tracker)
        if (segImplXY == nullptr) {
          segImplXZ = dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXZ*>(segmentation.segmentation());
          if (segImplXZ != nullptr) {
            usingXZCoordinates = true;
          }
        }

        if (segImplXY == nullptr && segImplXZ == nullptr) {
          if (auto log = logger()) {
            log->warn("Segmentation for readout '{}' is '{}'; expected CartesianGridXY or CartesianGridXZ",
                      cfg.readout, segmentation.type());
          }
        } else {
          decoder = segmentation.decoder();
          if (decoder == nullptr) {
            if (auto log = logger()) {
              log->warn("Segmentation for readout '{}' lacks a BitField decoder; neighbor bounds unavailable",
                        cfg.readout);
            }
          }

          SegmentationConfig segCfg{};
          segCfg.valid = (segImplXY != nullptr || segImplXZ != nullptr) && decoder != nullptr;

          // Extract grid parameters based on segmentation type
          if (segImplXY != nullptr) {
            segCfg.gridSizeXMM = segImplXY->gridSizeX();
            segCfg.gridSizeYMM = segImplXY->gridSizeY();
            segCfg.offsetXMM = segImplXY->offsetX();
            segCfg.offsetYMM = segImplXY->offsetY();
            segCfg.fieldNameX = segImplXY->fieldNameX();
            segCfg.fieldNameY = segImplXY->fieldNameY();
          } else {
            // CartesianGridXZ: map Z -> "Y" in our local coordinate system
            segCfg.gridSizeXMM = segImplXZ->gridSizeX();
            segCfg.gridSizeYMM = segImplXZ->gridSizeZ();  // Z maps to local Y
            segCfg.offsetXMM = segImplXZ->offsetX();
            segCfg.offsetYMM = segImplXZ->offsetZ();      // Z offset maps to local Y offset
            segCfg.fieldNameX = segImplXZ->fieldNameX();
            segCfg.fieldNameY = segImplXZ->fieldNameZ();  // Z field maps to local Y field
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

          // Get cell dimensions
          const auto dims = segImplXY ? segImplXY->cellDimensions(0)
                                      : segImplXZ->cellDimensions(0);
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

          // Store coordinate system flag in config
          segCfg.useXZCoordinates = usingXZCoordinates;
          cfg.segmentation = segCfg;

          if (segCfg.valid) {
            // DD4hep-style grid parameters
            cfg.pixelSpacingMM = segCfg.gridSizeXMM;
            cfg.pixelSpacingYMM = segCfg.gridSizeYMM;
            cfg.pixelSizeMM = segCfg.cellSizeXMM;
            cfg.pixelSizeYMM = segCfg.cellSizeYMM;
            cfg.gridOffsetMM = segCfg.offsetXMM;
            cfg.gridOffsetYMM = segCfg.offsetYMM;

            if (cfg.pixelsPerSide <= 0 && segCfg.numCellsX > 0 && segCfg.numCellsX == segCfg.numCellsY) {
              cfg.pixelsPerSide = segCfg.numCellsX;
            }
          }

          if (auto log = logger()) {
            log->info("Using DD4hep {} segmentation for readout '{}': pitch=({}, {}) mm, offset=({}, {}) mm, cells=({}, {})",
                      usingXZCoordinates ? "CartesianGridXZ" : "CartesianGridXY",
                      cfg.readout, cfg.pixelSpacingMM, cfg.pixelSpacingYMM,
                      cfg.gridOffsetMM, cfg.gridOffsetYMM,
                      segCfg.numCellsX, segCfg.numCellsY);
          }
        }
      }

      // ─────────────────────────────────────────────────────────────────────────
      // Read additional detector constants from DD4hep compact XML
      // These provide sensor-specific parameters not available from segmentation
      // ─────────────────────────────────────────────────────────────────────────
      const auto detConstants = getDetectorConstants(cfg.readout);
      int constantsRead = 0;

      // Try to read silicon thickness
      if (!detConstants.siliconThickness.empty()) {
        try {
          double thickness = detector->constantAsDouble(detConstants.siliconThickness);
          cfg.detectorThicknessMM = thickness / dd4hep::mm;
          ++constantsRead;
          if (auto log = logger()) {
            log->debug("  Read {} = {} mm", detConstants.siliconThickness, cfg.detectorThicknessMM);
          }
        } catch (const std::exception&) {
          // Constant not found, use configured default
        }
      }

      // Try to read detector size
      if (!detConstants.detectorSize.empty()) {
        try {
          double size = detector->constantAsDouble(detConstants.detectorSize);
          cfg.detectorSizeMM = size / dd4hep::mm;
          ++constantsRead;
          if (auto log = logger()) {
            log->debug("  Read {} = {} mm", detConstants.detectorSize, cfg.detectorSizeMM);
          }
        } catch (const std::exception&) {
          // Constant not found, use configured default
        }
      }

      // Try to read pixel size (may override segmentation-derived value)
      if (!detConstants.pixelSize.empty()) {
        try {
          double pixSize = detector->constantAsDouble(detConstants.pixelSize);
          double pixSizeMM = pixSize / dd4hep::mm;
          // Only override if significantly different from segmentation
          if (std::abs(pixSizeMM - cfg.pixelSizeMM) > 0.001) {
            cfg.pixelSizeMM = pixSizeMM;
            cfg.pixelSizeYMM = pixSizeMM;
            ++constantsRead;
            if (auto log = logger()) {
              log->debug("  Read {} = {} mm (overriding segmentation)", detConstants.pixelSize, cfg.pixelSizeMM);
            }
          }
        } catch (const std::exception&) {
          // Constant not found, use segmentation value
        }
      }

      // Try to read copper/pixel backing thickness
      if (!detConstants.copperThickness.empty()) {
        try {
          double cuThickness = detector->constantAsDouble(detConstants.copperThickness);
          cfg.pixelThicknessMM = cuThickness / dd4hep::mm;
          ++constantsRead;
          if (auto log = logger()) {
            log->debug("  Read {} = {} mm", detConstants.copperThickness, cfg.pixelThicknessMM);
          }
        } catch (const std::exception&) {
          // Constant not found, use configured default
        }
      }

      if (constantsRead > 0) {
        if (auto log = logger()) {
          log->info("Read {} DD4hep constants for '{}': Si thickness={} mm, det size={} mm",
                    constantsRead, cfg.readout, cfg.detectorThicknessMM, cfg.detectorSizeMM);
        }
      } else {
        // Try to read thickness from detector volume hierarchy
        double volumeThickness = getSensorThicknessFromReadout(detector, cfg.readout);
        if (volumeThickness > 0.0) {
          cfg.detectorThicknessMM = volumeThickness;
          if (auto log = logger()) {
            log->info("Read Si thickness from detector volume for '{}': {} mm",
                      cfg.readout, cfg.detectorThicknessMM);
          }
        } else {
          // Apply hardcoded fallbacks for detectors without named constants
          applyDetectorFallbacks(cfg.readout, cfg);
          if (auto log = logger()) {
            log->info("Using hardcoded fallbacks for '{}': Si thickness={} mm",
                      cfg.readout, cfg.detectorThicknessMM);
          }
        }
      }

    } catch (const std::exception& ex) {
      if (auto log = logger()) {
        log->warn("Failed to derive segmentation for readout '{}': {}", cfg.readout, ex.what());
      }
    }
  } else if (auto log = logger()) {
    log->warn("DD4hep detector unavailable; using configured geometry parameters only");
  }

  // Log final configuration after DD4hep overrides
  if (auto log = logger()) {
    log->info("Final geometry configuration:");
    log->info("  Pixel: size={} mm, pitch={} mm, offset={} mm",
              cfg.pixelSizeMM, cfg.pixelSpacingMM, cfg.gridOffsetMM);
    log->info("  Detector: size={} mm, Si thickness={} mm, pixel thickness={} mm",
              cfg.detectorSizeMM, cfg.detectorThicknessMM, cfg.pixelThicknessMM);
  }

  // Store decoder and field names for CellID decoding in Process()
  m_decoder = decoder;
  if (cfg.segmentation.valid) {
    m_fieldNameX = cfg.segmentation.fieldNameX;
    m_fieldNameY = cfg.segmentation.fieldNameY;
  }

  m_reconstructor.configure(cfg);
}

void ChargeSharingReconFactory::Process(int32_t /*runNumber*/, uint64_t /*eventNumber*/) {
  const auto* simhits = m_in_simhits();
  auto* output = m_out_hits().get();

  auto& dd4hep_service = m_dd4hep();
  auto converter = dd4hep_service.converter();

  const auto& cfg = m_reconstructor.config();
  const bool useXZ = cfg.segmentation.useXZCoordinates;

  // Position error defaults: pitch^2/12 (uniform distribution), computed once per event
  const double pitchX = cfg.pixelSpacingMM;
  const double pitchY = cfg.effectivePixelSpacingYMM();
  const double defaultVarX = (pitchX * pitchX) / 12.0;
  const double defaultVarY = (pitchY * pitchY) / 12.0;
  const double sensorT = cfg.detectorThicknessMM;
  const double varZ = (sensorT * sensorT) / 12.0;

  for (const auto& hit : *simhits) {
    const double edep = hit.getEDep();
    if (edep < static_cast<double>(m_minEDep())) {
      continue;
    }

    ChargeSharingReconstructor::Input input{};
    const auto pos = hit.getPosition();

    // For XZ segmentation (e.g., B0 tracker), map Z -> local Y for charge sharing
    // The charge sharing algorithm works in a local 2D coordinate system
    if (useXZ) {
      input.hitPositionMM = {pos.x, pos.z, pos.y};  // X stays, Z->localY, Y->localZ
    } else {
      input.hitPositionMM = {pos.x, pos.y, pos.z};  // Standard XY segmentation
    }

    input.energyDepositGeV = edep;
    input.cellID = hit.getCellID();

    // Decode CellID to grid indices directly (preferred path)
    if (m_decoder != nullptr) {
      const int idxI = static_cast<int>(m_decoder->get(hit.getCellID(), m_fieldNameX));
      const int idxJ = static_cast<int>(m_decoder->get(hit.getCellID(), m_fieldNameY));
      input.pixelIndexHint = std::pair<int,int>{idxI, idxJ};
    } else if (converter) {
      // Fallback: use converter position as pixel hint
      const auto center = converter->position(hit.getCellID());
      if (useXZ) {
        input.pixelHintMM = std::array<double, 3>{center.x(), center.z(), center.y()};
      } else {
        input.pixelHintMM = std::array<double, 3>{center.x(), center.y(), center.z()};
      }
    }

    const auto result = m_reconstructor.process(input);

    // Map reconstructed position back to global coordinates
    edm4hep::Vector3f reconPosition;
    if (useXZ) {
      // Map back: localX->X, localY->Z, localZ->Y
      reconPosition = edm4hep::Vector3f{
          static_cast<float>(result.reconstructedPositionMM[0]),  // X
          static_cast<float>(result.reconstructedPositionMM[2]),  // Y (was localZ)
          static_cast<float>(result.reconstructedPositionMM[1])   // Z (was localY)
      };
    } else {
      reconPosition = edm4hep::Vector3f{
          static_cast<float>(result.reconstructedPositionMM[0]),
          static_cast<float>(result.reconstructedPositionMM[1]),
          static_cast<float>(result.reconstructedPositionMM[2])
      };
    }

    // Compute position error (variance) from fit results.
    // CovDiag3f stores variances (sigma^2), matching the EICrecon convention.
    // Fallback: pitch^2/12 (uniform distribution over pixel), as in TrackerHitReconstruction.
    double varLocalX = defaultVarX;
    double varLocalY = defaultVarY;

    // 1D fit results (populated when reconMethod == Gaussian1D)
    if (result.fitRowX.converged &&
        std::isfinite(result.fitRowX.muError) && result.fitRowX.muError > 0.0) {
      varLocalX = result.fitRowX.muError * result.fitRowX.muError;
    }
    if (result.fitColY.converged &&
        std::isfinite(result.fitColY.muError) && result.fitColY.muError > 0.0) {
      varLocalY = result.fitColY.muError * result.fitColY.muError;
    }

    // 2D fit results (populated when reconMethod == Gaussian2D)
    if (result.fit2D.converged) {
      if (std::isfinite(result.fit2D.muXError) && result.fit2D.muXError > 0.0) {
        varLocalX = result.fit2D.muXError * result.fit2D.muXError;
      }
      if (std::isfinite(result.fit2D.muYError) && result.fit2D.muYError > 0.0) {
        varLocalY = result.fit2D.muYError * result.fit2D.muYError;
      }
    }

    // Map covariance to global coordinates (must match position mapping above)
    edm4eic::CovDiag3f posError;
    if (useXZ) {
      // Local X->Global X, Local Y->Global Z, Local Z->Global Y
      posError = edm4eic::CovDiag3f{
          static_cast<float>(varLocalX),   // xx: Global X (from local X fit)
          static_cast<float>(varZ),        // yy: Global Y (= local Z, through-sensor)
          static_cast<float>(varLocalY)    // zz: Global Z (= local Y fit)
      };
    } else {
      posError = edm4eic::CovDiag3f{
          static_cast<float>(varLocalX),   // xx: Global X
          static_cast<float>(varLocalY),   // yy: Global Y
          static_cast<float>(varZ)         // zz: Global Z (through-sensor)
      };
    }

    // Create the output TrackerHit using the collection's create() method
    output->create(
        hit.getCellID(),                                                          // cellID
        reconPosition,                                                            // position
        posError,                                                                 // positionError
        hit.getTime(),                                                             // time
        0.0f,                                                                      // timeError
        static_cast<float>(edep),                                                  // edep
        0.0f                                                                       // edepError
    );
  }
}

}  // namespace epic::chargesharing
