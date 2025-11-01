#include "ChargeSharingReconFactory.h"

#include <DD4hep/Detector.h>
#include <DD4hep/Readout.h>
#include <DD4hep/Segmentations.h>
#include <DD4hep/CartesianGridXY.h>
#include <DDSegmentation/BitFieldCoder.h>
#include <podio/Vec3f.h>

#include <array>
#include <exception>
#include <utility>
#include <vector>

namespace epic::chargesharing {

ChargeSharingReconFactory::ChargeSharingReconFactory() {
  SetTag("ChargeSharingRecon");
}

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

  if (m_dd4hep()) {
    try {
      const auto* detector = m_dd4hep().detector();
      if (detector != nullptr) {
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
      }
    } catch (const std::exception& ex) {
      if (auto log = logger()) {
        log->warn("Failed to derive segmentation for readout '{}': {}", cfg.readout, ex.what());
      }
    }
  } else if (auto log = logger()) {
    log->warn("DD4hep service unavailable; using configured geometry parameters only");
  }

  m_reconstructor.configure(cfg, segImpl, decoder);
}

void ChargeSharingReconFactory::Process(int32_t /*runNumber*/, uint64_t /*eventNumber*/) {
  const auto& simhits = m_in_simhits();

  auto converter = m_dd4hep() ? m_dd4hep().converter() : nullptr;

  std::vector<edm4eic::TrackerHit> outHits;
  outHits.reserve(simhits.size());

  for (const auto& hit : simhits) {
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

    edm4eic::TrackerHit reco;
    reco.setCellID(hit.getCellID());
    reco.setEDep(static_cast<float>(edep));
    reco.setTime(hit.getTime());
    reco.setPosition(podio::Vec3f{static_cast<float>(result.reconstructedPositionMM[0]),
                                   static_cast<float>(result.reconstructedPositionMM[1]),
                                   static_cast<float>(result.reconstructedPositionMM[2])});

    outHits.push_back(std::move(reco));
  }

  m_out_hits().assign(outHits.begin(), outHits.end());
}

}  // namespace epic::chargesharing

