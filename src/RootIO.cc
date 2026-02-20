/// \file RootIO.cc
/// \brief Implementation of ROOT I/O classes.

#include "RootIO.hh"
#include "Config.hh"
#include "GaussianFitter.hh"

#include "G4Exception.hh"
#include "G4ios.hh"

#include "TFile.h"
#include "TList.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TROOT.h"
#include "TString.h"
#include "TTree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace ECS::IO {

// ============================================================================
// BranchConfigurator
// ============================================================================

void BranchConfigurator::ConfigureCoreBranches(TTree* tree, const ScalarBuffers& scalars,
                                                const ClassificationBuffers& classification,
                                                const VectorBuffers& vectors,
                                                Config::ActivePixelMode mode,
                                                Config::PosReconModel reconModel) {
    if (!tree) return;
    ConfigureScalarBranches(tree, scalars, reconModel);
    ConfigureClassificationBranches(tree, classification);
    ConfigureVectorBranches(tree, vectors, mode);
    ConfigureNeighborhoodBranches(tree, vectors.pixelID);
}

void BranchConfigurator::ConfigureScalarBranches(TTree* tree, const ScalarBuffers& buffers,
                                                  [[maybe_unused]] Config::PosReconModel reconModel) {
    if (!tree) return;

    struct BranchDef { const char* name; G4double* addr; const char* leaf; };

    // Core branches (always present)
    const std::array<BranchDef, 7> coreBranches{{
        {"TrueX", buffers.trueX, "TrueX/D"},
        {"TrueY", buffers.trueY, "TrueY/D"},
        {"PixelX", buffers.pixelX, "PixelX/D"},
        {"PixelY", buffers.pixelY, "PixelY/D"},
        {"Edep", buffers.edep, "Edep/D"},
        {"PixelTrueDeltaX", buffers.pixelTrueDeltaX, "PixelTrueDeltaX/D"},
        {"PixelTrueDeltaY", buffers.pixelTrueDeltaY, "PixelTrueDeltaY/D"},
    }};

    for (const auto& def : coreBranches) {
        if (def.addr) tree->Branch(def.name, def.addr, def.leaf);
    }

}

void BranchConfigurator::ConfigureClassificationBranches(TTree* tree,
                                                          const ClassificationBuffers& buffers) {
    if (!tree) return;
    if (buffers.isPixelHit)
        tree->Branch("isPixelHit", buffers.isPixelHit, "isPixelHit/O");
    if (buffers.neighborhoodActiveCells)
        tree->Branch("NeighborhoodSize", buffers.neighborhoodActiveCells, "NeighborhoodSize/I");
    if (buffers.nearestPixelI)
        tree->Branch("NearestPixelI", buffers.nearestPixelI, "NearestPixelI/I");
    if (buffers.nearestPixelJ)
        tree->Branch("NearestPixelJ", buffers.nearestPixelJ, "NearestPixelJ/I");
    if (buffers.nearestPixelGlobalId)
        tree->Branch("NearestPixelID", buffers.nearestPixelGlobalId, "NearestPixelID/I");
}

void BranchConfigurator::ConfigureVectorBranches(TTree* tree, const VectorBuffers& buffers,
                                                  Config::ActivePixelMode mode) {
    if (!tree) return;

    auto addBranch = [&](const char* name, std::vector<G4double>* vec) {
        if (vec) tree->Branch(name, vec, kDefaultBufferSize, kSplitLevel);
    };

    // Common branches (always saved)
    addBranch("NeighborhoodPixelX", buffers.pixelX);
    addBranch("NeighborhoodPixelY", buffers.pixelY);
    addBranch("d_i", buffers.distance);
    addBranch("alpha_i", buffers.alpha);

    // Mode-specific branches
    switch (mode) {
        case Config::ActivePixelMode::Neighborhood:
            addBranch("Fi", buffers.chargeFractions);
            addBranch("Qi", buffers.charge);
            addBranch("Qn", buffers.chargeNew);
            addBranch("Qf", buffers.chargeFinal);
            break;

        case Config::ActivePixelMode::RowCol:
        case Config::ActivePixelMode::RowCol3x3:
            addBranch("FiRow", buffers.chargeFractionsRow);
            addBranch("FiCol", buffers.chargeFractionsCol);
            addBranch("QiRow", buffers.chargeRow);
            addBranch("QnRow", buffers.chargeNewRow);
            addBranch("QfRow", buffers.chargeFinalRow);
            addBranch("QiCol", buffers.chargeCol);
            addBranch("QnCol", buffers.chargeNewCol);
            addBranch("QfCol", buffers.chargeFinalCol);
            break;

        case Config::ActivePixelMode::ChargeBlock2x2:
        case Config::ActivePixelMode::ChargeBlock3x3:
            addBranch("FiBlock", buffers.chargeFractionsBlock);
            addBranch("QiBlock", buffers.chargeBlock);
            addBranch("QnBlock", buffers.chargeNewBlock);
            addBranch("QfBlock", buffers.chargeFinalBlock);
            break;
    }
}

void BranchConfigurator::ConfigureNeighborhoodBranches(TTree* tree, std::vector<G4int>* pixelID) {
    if (tree && pixelID) {
        tree->Branch("NeighborhoodPixelID", pixelID, kDefaultBufferSize, kSplitLevel);
    }
}

bool BranchConfigurator::ConfigureFullGridBranches(TTree* tree, const FullGridBuffers& buffers,
                                                    Config::ActivePixelMode mode) {
    if (!tree) return false;
    bool configured = false;

    // Use larger basket size for full grid branches which can have >10KB per entry
    // (e.g., 100x100 grid = 10000 doubles = 80KB per entry)
    auto addBranch = [&](const char* name, auto* vec) {
        if (vec) { tree->Branch(name, vec, kLargeVectorBufferSize, kSplitLevel); configured = true; }
    };

    // Common branches (always saved)
    addBranch("DistanceGrid", buffers.distance);
    addBranch("AlphaGrid", buffers.alpha);
    addBranch("PixelXGrid", buffers.pixelX);
    addBranch("PixelYGrid", buffers.pixelY);

    // Mode-specific branches
    switch (mode) {
        case Config::ActivePixelMode::Neighborhood:
            addBranch("FiGrid", buffers.fi);
            addBranch("QiGrid", buffers.qi);
            addBranch("QnGrid", buffers.qn);
            addBranch("QfGrid", buffers.qf);
            break;

        case Config::ActivePixelMode::RowCol:
        case Config::ActivePixelMode::RowCol3x3:
            addBranch("FiRowGrid", buffers.fiRow);
            addBranch("FiColGrid", buffers.fiCol);
            addBranch("QiRowGrid", buffers.qiRow);
            addBranch("QnRowGrid", buffers.qnRow);
            addBranch("QfRowGrid", buffers.qfRow);
            addBranch("QiColGrid", buffers.qiCol);
            addBranch("QnColGrid", buffers.qnCol);
            addBranch("QfColGrid", buffers.qfCol);
            break;

        case Config::ActivePixelMode::ChargeBlock2x2:
        case Config::ActivePixelMode::ChargeBlock3x3:
            addBranch("FiBlockGrid", buffers.fiBlock);
            addBranch("QiBlockGrid", buffers.qiBlock);
            addBranch("QnBlockGrid", buffers.qnBlock);
            addBranch("QfBlockGrid", buffers.qfBlock);
            break;
    }

    if (buffers.gridSide) {
        tree->Branch("FullGridSide", buffers.gridSide, "FullGridSide/I");
        configured = true;
    }
    return configured;
}

// ============================================================================
// TreeFiller
// ============================================================================

TreeFiller::TreeFiller() : fNeighborhoodLayout(Constants::NEIGHBORHOOD_RADIUS) {}

void TreeFiller::SetNeighborhoodRadius(G4int radius) {
    fNeighborhoodLayout.SetRadius(radius);
}

bool TreeFiller::Fill(TTree* tree, const EventRecord& record, std::mutex* treeMutex) {
    if (!tree) return false;

    std::unique_lock<std::mutex> lock;
    if (treeMutex) lock = std::unique_lock<std::mutex>(*treeMutex);

    UpdateSummaryScalars(record);

    const std::size_t requestedCells = record.totalGridCells > 0
        ? static_cast<std::size_t>(record.totalGridCells)
        : std::max<std::size_t>(1, fNeighborhoodLayout.TotalCells());

    PrepareNeighborhoodStorage(requestedCells);
    PopulateNeighborhoodFromRecord(record);
    PopulateFullFractionsFromRecord(record);

    if (tree->Fill() < 0) {
        G4Exception("TreeFiller::Fill", "TreeFillFailure", FatalException,
                    "TTree::Fill() reported an error.");
        return false;
    }
    return true;
}

void TreeFiller::UpdateSummaryScalars(const EventRecord& record) {
    fEdep = record.summary.edep;
    fTrueX = record.summary.hitX;
    fTrueY = record.summary.hitY;
    fPixelX = record.summary.nearestPixelX;
    fPixelY = record.summary.nearestPixelY;
    fPixelTrueDeltaX = record.summary.pixelTrueDeltaX;
    fPixelTrueDeltaY = record.summary.pixelTrueDeltaY;
    fIsPixelHit = record.summary.isPixelHitCombined;
    fNearestPixelI = record.nearestPixelI;
    fNearestPixelJ = record.nearestPixelJ;
    fNearestPixelGlobalId = record.nearestPixelGlobalId;
}

void TreeFiller::PrepareNeighborhoodStorage(std::size_t requestedCells) {
    const std::size_t capacity = std::max<std::size_t>(1, requestedCells);
    fNeighborhoodCapacity = capacity;
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();

    // Fractions
    ResizeAndFill(fNeighborhoodChargeFractions, capacity, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    ResizeAndFill(fNeighborhoodChargeFractionsRow, capacity, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    ResizeAndFill(fNeighborhoodChargeFractionsCol, capacity, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    ResizeAndFill(fNeighborhoodChargeFractionsBlock, capacity, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    // Neighborhood charges
    ResizeAndFill(fNeighborhoodCharge, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeNew, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeFinal, capacity, 0.0);
    // Row-mode charges
    ResizeAndFill(fNeighborhoodChargeRow, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeNewRow, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeFinalRow, capacity, 0.0);
    // Col-mode charges
    ResizeAndFill(fNeighborhoodChargeCol, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeNewCol, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeFinalCol, capacity, 0.0);
    // Block-mode charges
    ResizeAndFill(fNeighborhoodChargeBlock, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeNewBlock, capacity, 0.0);
    ResizeAndFill(fNeighborhoodChargeFinalBlock, capacity, 0.0);
    // Geometry
    ResizeAndFill(fNeighborhoodDistance, capacity, nan);
    ResizeAndFill(fNeighborhoodAlpha, capacity, nan);
    ResizeAndFill(fNeighborhoodPixelX, capacity, nan);
    ResizeAndFill(fNeighborhoodPixelY, capacity, nan);
    ResizeAndFill(fNeighborhoodPixelID, capacity, -1);
    fNeighborhoodActiveCells = 0;
}

void TreeFiller::PopulateNeighborhoodFromRecord(const EventRecord& record) {
    auto copySpan = [](const auto& source, auto& target) {
        const std::size_t n = std::min(target.size(), source.size());
        if (n > 0) std::copy_n(source.begin(), n, target.begin());
    };

    copySpan(record.neighborChargesNew, fNeighborhoodChargeNew);
    copySpan(record.neighborChargesFinal, fNeighborhoodChargeFinal);
    // Row/Col/Block mode noisy charges
    copySpan(record.neighborChargesNewRow, fNeighborhoodChargeNewRow);
    copySpan(record.neighborChargesFinalRow, fNeighborhoodChargeFinalRow);
    copySpan(record.neighborChargesNewCol, fNeighborhoodChargeNewCol);
    copySpan(record.neighborChargesFinalCol, fNeighborhoodChargeFinalCol);
    copySpan(record.neighborChargesNewBlock, fNeighborhoodChargeNewBlock);
    copySpan(record.neighborChargesFinalBlock, fNeighborhoodChargeFinalBlock);

    std::size_t activeCells = 0;
    for (const auto& cell : record.neighborCells) {
        if (cell.gridIndex < 0) continue;
        const auto idx = static_cast<std::size_t>(cell.gridIndex);
        if (idx >= fNeighborhoodCapacity) continue;

        // Fractions
        fNeighborhoodChargeFractions[idx] = cell.fraction;
        fNeighborhoodChargeFractionsRow[idx] = cell.fractionRow;
        fNeighborhoodChargeFractionsCol[idx] = cell.fractionCol;
        fNeighborhoodChargeFractionsBlock[idx] = cell.fractionBlock;
        // Neighborhood charge
        fNeighborhoodCharge[idx] = cell.charge;
        // Row-mode charge (base charge from row fraction; noise applied later if needed)
        fNeighborhoodChargeRow[idx] = cell.chargeRow;
        // Col-mode charge
        fNeighborhoodChargeCol[idx] = cell.chargeCol;
        // Block-mode charge
        fNeighborhoodChargeBlock[idx] = cell.chargeBlock;
        // Geometry
        fNeighborhoodPixelX[idx] = cell.center.x();
        fNeighborhoodPixelY[idx] = cell.center.y();
        fNeighborhoodPixelID[idx] = cell.globalPixelId;
        if (record.includeDistanceAlpha) {
            fNeighborhoodDistance[idx] = cell.distance;
            fNeighborhoodAlpha[idx] = cell.alpha;
        }
        ++activeCells;
    }
    fNeighborhoodActiveCells = static_cast<G4int>(activeCells);
}

void TreeFiller::PopulateFullFractionsFromRecord(const EventRecord& record) {
    if (!fStoreFullFractions) return;

    const G4int gridSide = (record.fullGridCols > 0) ? record.fullGridCols : fGridNumBlocksPerSide;
    if (!EnsureFullFractionBuffer(gridSide)) return;
    if (record.fullGridCols > 0) fFullGridSide = record.fullGridCols;

    auto copyOrZero = [](const std::span<const G4double>& src, std::vector<G4double>& dst) {
        if (dst.empty()) return;
        if (src.empty()) { std::fill(dst.begin(), dst.end(), 0.0); return; }
        const std::size_t n = std::min(dst.size(), src.size());
        std::copy_n(src.begin(), n, dst.begin());
        if (n < dst.size()) std::fill(dst.begin() + n, dst.end(), 0.0);
    };

    // Fractions
    copyOrZero(record.fullFi, fFullFi);
    copyOrZero(record.fullFiRow, fFullFiRow);
    copyOrZero(record.fullFiCol, fFullFiCol);
    copyOrZero(record.fullFiBlock, fFullFiBlock);
    // Neighborhood charges
    copyOrZero(record.fullQi, fFullQi);
    copyOrZero(record.fullQn, fFullQn);
    copyOrZero(record.fullQf, fFullQf);
    // Row-mode charges
    copyOrZero(record.fullQiRow, fFullQiRow);
    copyOrZero(record.fullQnRow, fFullQnRow);
    copyOrZero(record.fullQfRow, fFullQfRow);
    // Col-mode charges
    copyOrZero(record.fullQiCol, fFullQiCol);
    copyOrZero(record.fullQnCol, fFullQnCol);
    copyOrZero(record.fullQfCol, fFullQfCol);
    // Block-mode charges
    copyOrZero(record.fullQiBlock, fFullQiBlock);
    copyOrZero(record.fullQnBlock, fFullQnBlock);
    copyOrZero(record.fullQfBlock, fFullQfBlock);
    // Geometry
    copyOrZero(record.fullDistance, fFullDistance);
    copyOrZero(record.fullAlpha, fFullAlpha);
    copyOrZero(record.fullPixelX, fFullPixelXGrid);
    copyOrZero(record.fullPixelY, fFullPixelYGrid);
}

bool TreeFiller::EnsureFullFractionBuffer(G4int gridSide) {
    if (gridSide >= 0) fFullGridSide = gridSide;
    const G4int side = (fFullGridSide > 0) ? fFullGridSide : fGridNumBlocksPerSide;
    const G4int numBlocks = std::max(0, side);
    const std::size_t totalPixels = static_cast<std::size_t>(numBlocks) * numBlocks;

    if (totalPixels == 0) {
        // Fractions
        fFullFi.clear(); fFullFiRow.clear(); fFullFiCol.clear(); fFullFiBlock.clear();
        // Neighborhood charges
        fFullQi.clear(); fFullQn.clear(); fFullQf.clear();
        // Row-mode charges
        fFullQiRow.clear(); fFullQnRow.clear(); fFullQfRow.clear();
        // Col-mode charges
        fFullQiCol.clear(); fFullQnCol.clear(); fFullQfCol.clear();
        // Block-mode charges
        fFullQiBlock.clear(); fFullQnBlock.clear(); fFullQfBlock.clear();
        // Geometry
        fFullDistance.clear(); fFullAlpha.clear();
        fFullPixelXGrid.clear(); fFullPixelYGrid.clear();
        fFullGridSide = 0;
        return false;
    }

    fFullGridSide = numBlocks;
    auto ensure = [totalPixels](auto& vec) {
        if (vec.size() != totalPixels) vec.assign(totalPixels, 0.0);
        else std::fill(vec.begin(), vec.end(), 0.0);
    };

    // Fractions
    ensure(fFullFi); ensure(fFullFiRow); ensure(fFullFiCol); ensure(fFullFiBlock);
    // Neighborhood charges
    ensure(fFullQi); ensure(fFullQn); ensure(fFullQf);
    // Row-mode charges
    ensure(fFullQiRow); ensure(fFullQnRow); ensure(fFullQfRow);
    // Col-mode charges
    ensure(fFullQiCol); ensure(fFullQnCol); ensure(fFullQfCol);
    // Block-mode charges
    ensure(fFullQiBlock); ensure(fFullQnBlock); ensure(fFullQfBlock);
    // Geometry
    ensure(fFullDistance); ensure(fFullAlpha);
    ensure(fFullPixelXGrid); ensure(fFullPixelYGrid);
    return true;
}

// ============================================================================
// MetadataPublisher
// ============================================================================

std::string MetadataPublisher::ModelToString(Config::PosReconModel model) {
    switch (model) {
        case Config::PosReconModel::LinA: return "LinA";
        case Config::PosReconModel::LogA:
        default: return "LogA";
    }
}

std::string MetadataPublisher::SignalModelToString(Config::SignalModel model) {
    switch (model) {
        case Config::SignalModel::LinA: return "LinA";
        case Config::SignalModel::LogA:
        default: return "LogA";
    }
}

static std::string ActivePixelModeToString(Config::ActivePixelMode mode) {
    switch (mode) {
        case Config::ActivePixelMode::Neighborhood: return "Neighborhood";
        case Config::ActivePixelMode::ChargeBlock2x2: return "ChargeBlock2x2";
        case Config::ActivePixelMode::ChargeBlock3x3: return "ChargeBlock3x3";
        case Config::ActivePixelMode::RowCol: return "RowCol";
        case Config::ActivePixelMode::RowCol3x3: return "RowCol3x3";
        default: return "Neighborhood";
    }
}

MetadataPublisher::EntryList MetadataPublisher::CollectEntries() const {
    EntryList entries;
    entries.reserve(20);

    // Helper lambdas for typed entry creation
    auto addString = [&](const std::string& key, const std::string& val) {
        entries.emplace_back(key, MetaValue{val});
    };
    auto addDouble = [&](const std::string& key, G4double val) {
        entries.emplace_back(key, MetaValue{val});
    };
    auto addInt = [&](const std::string& key, G4int val) {
        entries.emplace_back(key, MetaValue{val});
    };

    // Simulation info (timestamp and versions)
    if (!fSimulation.timestamp.empty()) addString("SimulationTimestamp", fSimulation.timestamp);
    if (!fSimulation.geant4Version.empty()) addString("Geant4Version", fSimulation.geant4Version);
    if (!fSimulation.rootVersion.empty()) addString("ROOTVersion", fSimulation.rootVersion);

    // Grid parameters (doubles and ints)
    if (fGrid.pixelSize > 0.0) addDouble("GridPixelSize_mm", fGrid.pixelSize);
    if (fGrid.pixelSpacing > 0.0) addDouble("GridPixelSpacing_mm", fGrid.pixelSpacing);
    addDouble("GridOffset_mm", fGrid.gridOffset);  // DD4hep-style grid offset
    if (fGrid.detectorSize > 0.0) addDouble("GridDetectorSize_mm", fGrid.detectorSize);
    if (fGrid.detectorThickness > 0.0) addDouble("DetectorThickness_mm", fGrid.detectorThickness);
    if (fGrid.interpadGap > 0.0) addDouble("InterpadGap_mm", fGrid.interpadGap);
    if (fGrid.numBlocksPerSide > 0) addInt("GridNumBlocksPerSide", fGrid.numBlocksPerSide);
    if (fGrid.storeFullFractions && fGrid.fullGridSide > 0) addInt("FullGridSide", fGrid.fullGridSide);
    if (fGrid.neighborhoodRadius >= 0) addInt("NeighborhoodRadius", fGrid.neighborhoodRadius);

    // Model parameters (strings for enums, doubles for numeric)
    addString("SignalModel", SignalModelToString(fModel.signalModel));
    addString("ReconMethod", ModelToString(fModel.model));
    addString("ActivePixelMode", ActivePixelModeToString(fModel.activePixelMode));
    // Beta is only used when LinA signal model is active
    if (fModel.signalModel == Config::SignalModel::LinA && std::isfinite(fModel.beta)) {
        addDouble("ChargeSharingLinearBeta_per_um", fModel.beta);
    }

    // Physics parameters (doubles)
    addDouble("ChargeSharingReferenceD0_microns", fPhysics.d0);
    addDouble("IonizationEnergy_eV", fPhysics.ionizationEnergy);
    addDouble("Gain", fPhysics.gain);

    // Noise parameters (doubles)
    addDouble("NoisePixelGainSigmaMin", fNoise.gainSigmaMin);
    addDouble("NoisePixelGainSigmaMax", fNoise.gainSigmaMax);
    addDouble("NoiseElectronCount", fNoise.electronCount);

    return entries;
}

void MetadataPublisher::WriteEntriesToUserInfo(TTree* tree, const EntryList& entries) {
    if (!tree) {
        G4Exception("MetadataPublisher::WriteEntriesToUserInfo", "InvalidTree", FatalException,
                    "Unable to write metadata because the TTree pointer is null.");
        return;
    }

    TList* userInfo = tree->GetUserInfo();
    for (const auto& [key, value] : entries) {
        std::visit([&key, userInfo](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            // Objects must be heap-allocated - TTree takes ownership
            if constexpr (std::is_same_v<T, G4double>) {
                userInfo->Add(new TParameter<double>(key.c_str(), val));
            } else if constexpr (std::is_same_v<T, G4int>) {
                userInfo->Add(new TParameter<int>(key.c_str(), val));
            } else if constexpr (std::is_same_v<T, G4bool>) {
                userInfo->Add(new TParameter<bool>(key.c_str(), val));
            } else if constexpr (std::is_same_v<T, std::string>) {
                userInfo->Add(new TNamed(key.c_str(), val.c_str()));
            }
        }, value);
    }
}

void MetadataPublisher::WriteToTree(TTree* tree, std::mutex* ioMutex) const {
    if (!tree) return;

    EntryList entries = CollectEntries();
    if (entries.empty()) return;

    if (ioMutex) {
        std::lock_guard<std::mutex> lock(*ioMutex);
        WriteEntriesToUserInfo(tree, entries);
    } else {
        WriteEntriesToUserInfo(tree, entries);
    }
}

// ============================================================================
// PostProcessingRunner
// ============================================================================

std::string PostProcessingRunner::NormalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

void PostProcessingRunner::EnsureBatchMode() {
    if (!fBatchModeSet) {
        gROOT->SetBatch(true);
        fBatchModeSet = true;
    }
}

void PostProcessingRunner::ConfigureIncludePaths() {
    if (fIncludePathsConfigured || fConfig.sourceDir.empty()) return;

    TString sourceDir = TString(NormalizePath(fConfig.sourceDir).c_str());
    gROOT->ProcessLine(TString::Format(".include %s/include", sourceDir.Data()));
    gROOT->ProcessLine(TString::Format(".include %s/src", sourceDir.Data()));
    fIncludePathsConfigured = true;
}

bool PostProcessingRunner::RunMacro(const std::string& macroPath, const std::string& entryPoint,
                                     const std::string& rootFile) {
    EnsureBatchMode();
    ConfigureIncludePaths();

    TString normalizedMacro = TString(NormalizePath(macroPath).c_str());
    TString normalizedRoot = TString(NormalizePath(rootFile).c_str());

    G4cout << "[PostProcessingRunner] Running macro " << entryPoint << "..." << G4endl;

    gROOT->ProcessLine(TString::Format(".L %s", normalizedMacro.Data()));
    gROOT->ProcessLine(TString::Format("%s(\"%s\")", entryPoint.c_str(), normalizedRoot.Data()));

    return true;
}

bool PostProcessingRunner::Run() {
    if (!fConfig.runFitGaus1D && !fConfig.runFitGaus2D) return false;

    EnsureBatchMode();

    bool executed = false;

    if (fConfig.runFitGaus1D) {
        G4cout << "[PostProcessingRunner] Running FitGaussian1D..." << G4endl;
        ECS::Fit::FitGaussian1D(fConfig.rootFileName.c_str());
        executed = true;
    }
    if (fConfig.runFitGaus2D) {
        G4cout << "[PostProcessingRunner] Running FitGaussian2D..." << G4endl;
        ECS::Fit::FitGaussian2D(fConfig.rootFileName.c_str());
        executed = true;
    }

    if (executed) {
        G4cout << "[PostProcessingRunner] Post-processing fits finished" << G4endl;
    }
    return executed;
}

} // namespace ECS::IO
