// Include DD4hep headers first to resolve forward declarations
#include <DD4hep/Detector.h>
#include <DD4hep/Readout.h>
#include <DD4hep/Segmentations.h>
#include <DDSegmentation/CartesianGridXY.h>
#include <DDSegmentation/BitFieldCoder.h>

#include "ChargeSharingReconFactory.h"

#include <edm4hep/Vector3f.h>
#include <edm4eic/CovDiag3f.h>

#include <array>
#include <exception>
#include <utility>
#include <vector>

namespace epic::chargesharing {

void ChargeSharingReconFactory::Configure() {
  ChargeSharingConfig cfg{};
  cfg.readout = m_readout();
  cfg.minEDepGeV = m_minEDep();
  cfg.neighborhoodRadius = m_neighborhoodRadius();
  cfg.pixelSizeMM = m_pixelSize();
  cfg.pixelSizeYMM = m_pixelSize();
  cfg.pixelSpacingMM = m_pixelSpacing();
  cfg.pixelSpacingYMM = m_pixelSpacing();
  cfg.pixelCornerOffsetMM = m_pixelCornerOffset();
  cfg.detectorSizeMM = m_detectorSize();
  cfg.detectorThicknessMM = m_detectorThickness();
  cfg.pixelThicknessMM = m_pixelThickness();
  cfg.detectorZCenterMM = m_detectorZCenter();
  cfg.pixelsPerSide = m_pixelsPerSide();
  cfg.ionizationEnergyEV = m_ionizationEnergy();
  cfg.amplificationFactor = m_amplificationFactor();
  cfg.elementaryChargeC = m_elementaryCharge();
  cfg.d0Micron = m_d0Micron();
  cfg.emitNeighborDiagnostics = m_emitNeighborDiagnostics();

  const dd4hep::DDSegmentation::CartesianGridXY* segImpl = nullptr;
  const dd4hep::DDSegmentation::BitFieldCoder* decoder = nullptr;

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
        segImpl = dynamic_cast<const dd4hep::DDSegmentation::CartesianGridXY*>(segmentation.segmentation());
        if (segImpl == nullptr) {
          if (auto log = logger()) {
            log->warn("Segmentation for readout '{}' is '{}'; expected CartesianGridXY", cfg.readout,
                      segmentation.type());
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
          segCfg.valid = segImpl != nullptr && decoder != nullptr;
          segCfg.gridSizeXMM = segImpl->gridSizeX();
          segCfg.gridSizeYMM = segImpl->gridSizeY();
          segCfg.offsetXMM = segImpl->offsetX();
          segCfg.offsetYMM = segImpl->offsetY();
          segCfg.fieldNameX = segImpl->fieldNameX();
          segCfg.fieldNameY = segImpl->fieldNameY();

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

          const auto dims = segImpl->cellDimensions(0);
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

          cfg.segmentation = segCfg;

          if (segCfg.valid) {
            cfg.pixelSpacingMM = segCfg.gridSizeXMM;
            cfg.pixelSpacingYMM = segCfg.gridSizeYMM;
            cfg.pixelSizeMM = segCfg.cellSizeXMM;
            cfg.pixelSizeYMM = segCfg.cellSizeYMM;

            if (cfg.pixelsPerSide <= 0 && segCfg.numCellsX > 0 && segCfg.numCellsX == segCfg.numCellsY) {
              cfg.pixelsPerSide = segCfg.numCellsX;
            }
          }

          if (auto log = logger()) {
            log->info("Using DD4hep segmentation for readout '{}': pitch=({}, {}) mm, cells=({}, {})",
                      cfg.readout, cfg.pixelSpacingMM, cfg.pixelSpacingYMM,
                      segCfg.numCellsX, segCfg.numCellsY);
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

  m_reconstructor.configure(cfg, segImpl, decoder);
}

void ChargeSharingReconFactory::Process(int32_t /*runNumber*/, uint64_t /*eventNumber*/) {
  const auto* simhits = m_in_simhits();
  auto* output = m_out_hits().get();

  auto& dd4hep_service = m_dd4hep();
  auto converter = dd4hep_service.converter();

  for (const auto& hit : *simhits) {
    const double edep = hit.getEDep();
    if (edep < static_cast<double>(m_minEDep())) {
      continue;
    }

    ChargeSharingReconstructor::Input input{};
    const auto pos = hit.getPosition();
    input.hitPositionMM = {pos.x, pos.y, pos.z};
    input.energyDepositGeV = edep;
    input.cellID = hit.getCellID();

    if (converter) {
      const auto center = converter->position(hit.getCellID());
      input.pixelHintMM = std::array<double, 3>{center.x(), center.y(), center.z()};
    }

    const auto result = m_reconstructor.process(input);

    // Create the output TrackerHit using the collection's create() method
    output->create(
        hit.getCellID(),                                                          // cellID
        edm4hep::Vector3f{static_cast<float>(result.reconstructedPositionMM[0]),  // position
                          static_cast<float>(result.reconstructedPositionMM[1]),
                          static_cast<float>(result.reconstructedPositionMM[2])},
        edm4eic::CovDiag3f{0.0f, 0.0f, 0.0f},                                      // positionError (TODO: compute)
        hit.getTime(),                                                             // time
        0.0f,                                                                      // timeError
        static_cast<float>(edep),                                                  // edep
        0.0f                                                                       // edepError
    );
  }
}

}  // namespace epic::chargesharing
