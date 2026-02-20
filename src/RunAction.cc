/**
 * @file RunAction.cc
 * @brief Run lifecycle management with dedicated helpers for ROOT I/O and worker synchronisation.
 */
#include "RunAction.hh"

#include "Config.hh"
#include "DetectorConstruction.hh"
#include "RootHelpers.hh"
#include "RuntimeConfig.hh"

#include "G4Exception.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

#include "Compression.h"
#include "RVersion.h"
#include "TError.h"
#include "TFile.h"
#include "TFileMerger.h"
#include "TThread.h"
#include "TTree.h"

#include "G4Version.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace {
std::once_flag gRootInitFlag;
} // namespace

// Anonymous namespace for internal helpers removed - functionality moved to ECS::IO::MetadataPublisher

RunAction::RunAction()
    : fRootWriter(std::make_unique<ECS::RootFileWriter>()),

      fNeighborhoodLayout(Constants::NEIGHBORHOOD_RADIUS),

      fChargeSharingBeta(std::numeric_limits<G4double>::quiet_NaN()),

      fStoreFullFractions(Constants::STORE_FULL_GRID),

      fEDM4hepWriter(IO::MakeEDM4hepWriter())

{
    if (fEDM4hepWriter && fEDM4hepWriter->IsEnabled()) {
        fWriteEDM4hep = true;
    }
}

RunAction::~RunAction() {
    CleanupRootObjects();
}

std::unique_lock<std::mutex> RunAction::MakeTreeLock() {
#ifdef G4MULTITHREADED
    if (G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread()) {
        return std::unique_lock<std::mutex>(fTreeMutex);
    }
#endif
    return {};
}

TFile* RunAction::GetRootFile() const {
    return fRootWriter ? fRootWriter->File() : nullptr;
}

TTree* RunAction::GetTree() const {
    return fRootWriter ? fRootWriter->Tree() : nullptr;
}

void RunAction::EnsureBranchBuffersInitialized() {
    const G4int radius = (fGridNeighborhoodRadius >= 0) ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS;
    fNeighborhoodLayout.SetRadius(radius);
    const std::size_t desiredCapacity = std::max<std::size_t>(1, fNeighborhoodLayout.TotalCells());

    if (desiredCapacity != fNeighborhoodCapacity || fNeighborhoodChargeFractions.size() != desiredCapacity) {
        fNeighborhoodCapacity = desiredCapacity;
        const auto reserveVec = [desiredCapacity](auto& vec) {
            vec.clear();
            vec.reserve(desiredCapacity);
        };
        reserveVec(fNeighborhoodChargeFractions);
        reserveVec(fNeighborhoodCharge);
        reserveVec(fNeighborhoodChargeNew);
        reserveVec(fNeighborhoodChargeFinal);
        reserveVec(fNeighborhoodDistance);
        reserveVec(fNeighborhoodAlpha);
        reserveVec(fNeighborhoodPixelX);
        reserveVec(fNeighborhoodPixelY);
        reserveVec(fNeighborhoodPixelID);
    }

    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    ECS::ResizeAndFill(fNeighborhoodChargeFractions, fNeighborhoodCapacity, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    ECS::ResizeAndFill(fNeighborhoodCharge, fNeighborhoodCapacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeNew, fNeighborhoodCapacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeFinal, fNeighborhoodCapacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodDistance, fNeighborhoodCapacity, nan);
    ECS::ResizeAndFill(fNeighborhoodAlpha, fNeighborhoodCapacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelX, fNeighborhoodCapacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelY, fNeighborhoodCapacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelID, fNeighborhoodCapacity, -1);
    fNeighborhoodActiveCells = 0;

    if (fStoreFullFractions) {
        EnsureFullFractionBuffer(fGridNumBlocksPerSide);
    } else {
        fFullFi.clear();
        fFullQi.clear();
        fFullQn.clear();
        fFullQf.clear();
        fFullDistance.clear();
        fFullAlpha.clear();
        fFullPixelXGrid.clear();
        fFullPixelYGrid.clear();
        fFullGridSide = 0;
    }
    fFullFractionsBranchInitialized = false;
}

bool RunAction::EnsureFullFractionBuffer(G4int gridSide) {
    if (gridSide >= 0) {
        fFullGridSide = gridSide;
    }
    const G4int side = (fFullGridSide > 0) ? fFullGridSide : fGridNumBlocksPerSide;
    const G4int numBlocks = std::max(0, side);
    const std::size_t totalPixels = static_cast<std::size_t>(numBlocks) * static_cast<std::size_t>(numBlocks);
    if (totalPixels == 0U) {
        // Fractions
        fFullFi.clear();
        fFullFiRow.clear();
        fFullFiCol.clear();
        fFullFiBlock.clear();
        // Neighborhood charges
        fFullQi.clear();
        fFullQn.clear();
        fFullQf.clear();
        // Row-mode charges
        fFullQiRow.clear();
        fFullQnRow.clear();
        fFullQfRow.clear();
        // Col-mode charges
        fFullQiCol.clear();
        fFullQnCol.clear();
        fFullQfCol.clear();
        // Block-mode charges
        fFullQiBlock.clear();
        fFullQnBlock.clear();
        fFullQfBlock.clear();
        // Geometry
        fFullDistance.clear();
        fFullAlpha.clear();
        fFullPixelXGrid.clear();
        fFullPixelYGrid.clear();
        fFullGridSide = 0;
        return false;
    }
    fFullGridSide = numBlocks;
    const auto ensure = [totalPixels](auto& vec) {
        if (vec.size() != totalPixels) {
            vec.assign(totalPixels, 0.0);
        } else {
            std::fill(vec.begin(), vec.end(), 0.0);
        }
    };

    // Fractions
    ensure(fFullFi);
    ensure(fFullFiRow);
    ensure(fFullFiCol);
    ensure(fFullFiBlock);
    // Neighborhood charges
    ensure(fFullQi);
    ensure(fFullQn);
    ensure(fFullQf);
    // Row-mode charges
    ensure(fFullQiRow);
    ensure(fFullQnRow);
    ensure(fFullQfRow);
    // Col-mode charges
    ensure(fFullQiCol);
    ensure(fFullQnCol);
    ensure(fFullQfCol);
    // Block-mode charges
    ensure(fFullQiBlock);
    ensure(fFullQnBlock);
    ensure(fFullQfBlock);
    // Geometry
    ensure(fFullDistance);
    ensure(fFullAlpha);
    ensure(fFullPixelXGrid);
    ensure(fFullPixelYGrid);
    return true;
}

void RunAction::BeginOfRunAction(const G4Run* run) {
    std::call_once(gRootInitFlag, ECS::InitializeROOTThreading);

    EnsureBranchBuffersInitialized();

    if (!run) {
        G4Exception("RunAction::BeginOfRunAction", "InvalidRunObject", FatalException,
                    "Received null run object in BeginOfRunAction.");
        return;
    }

    const ThreadContext context = BuildThreadContext(run);

    LogBeginRun(context);

    if (context.multithreaded && context.master) {
        ECS::WorkerSync::Instance().Reset(context.totalWorkers);
    }

    const G4String fileName = DetermineOutputFileName(context);

    if (!context.multithreaded || context.worker) {
        InitializeRootOutputs(context, fileName);
    } else {
        fRootWriter->Attach(nullptr, nullptr, false);
    }
}

void RunAction::EndOfRunAction(const G4Run* run) {
    if (!run) {
        G4Exception("RunAction::EndOfRunAction", "InvalidRunObject", FatalException,
                    "Received null run object in EndOfRunAction.");
        return;
    }

    const ThreadContext context = BuildThreadContext(run);

    if (!context.multithreaded || context.worker) {
        HandleWorkerEndOfRun(context, run);
        return;
    }

    WaitForAllWorkersToComplete();
    HandleMasterEndOfRun(context, run);
}

RunAction::ThreadContext RunAction::BuildThreadContext(const G4Run* run) {
    ThreadContext context;
    context.multithreaded = G4Threading::IsMultithreadedApplication();
#ifdef G4MULTITHREADED
    context.worker = context.multithreaded && G4Threading::IsWorkerThread();
    context.master = context.multithreaded && !context.worker;
    context.threadId = context.worker ? G4Threading::G4GetThreadId() : -1;
    context.totalWorkers = context.multithreaded ? G4Threading::GetNumberOfRunningWorkerThreads() : 0;
#else
    context.worker = false;
    context.master = false;
    context.threadId = -1;
    context.totalWorkers = 0;
#endif
    context.runId = run ? run->GetRunID() : -1;
    if (!context.multithreaded) {
        context.master = false;
        context.worker = false;
    }
    return context;
}

void RunAction::LogBeginRun(const ThreadContext& context) {
    const G4int runId = context.runId;
    if (context.multithreaded) {
        if (context.worker) {
            G4cout << "[RunAction] Worker " << context.threadId << " beginning run " << runId << G4endl;
        } else {
            G4cout << "[RunAction] Master beginning run " << runId << " with " << context.totalWorkers << " workers"
                   << G4endl;
        }
    } else {
        G4cout << "[RunAction] Beginning run " << runId << " (single-threaded)" << G4endl;
    }
}

G4String RunAction::DetermineOutputFileName(const ThreadContext& context) {
    if (!context.multithreaded) {
        return "epicChargeSharing.root";
    }
    if (context.worker) {
        return ECS::WorkerFileName(context.threadId);
    }
    return "epicChargeSharing.root";
}

void RunAction::InitializeRootOutputs(const ThreadContext& context, const G4String& fileName) {
    (void)context;

    std::unique_lock<std::mutex> const rootLock(ECS::RootIOMutex());
    auto* rootFile = new TFile(fileName.c_str(), "RECREATE");
    if (!rootFile || rootFile->IsZombie()) {
        G4Exception("RunAction::InitializeRootOutputs", "RootFileOpenFailure", FatalException,
                    ("Unable to open ROOT file " + std::string(fileName)).c_str());
    }
    rootFile->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
    rootFile->SetCompressionLevel(4);
    rootFile->cd();

    auto* tree = new TTree("Hits", "AC-LGAD charge sharing hits");
    tree->SetDirectory(rootFile);
    tree->SetAutoSave(0);
    tree->SetAutoFlush(50000);
    TTree::SetMaxTreeSize(static_cast<Long64_t>(10LL) * 1024LL * 1024LL * 1024LL);
    ConfigureCoreBranches(tree);

    fRootWriter->Attach(rootFile, tree, true);

    if (fStoreFullFractions) {
        ConfigureFullFractionBranch(true);
    }

    // Initialize EDM4hep output if enabled
    if (fWriteEDM4hep && fEDM4hepWriter) {
        // Derive EDM4hep filename from ROOT filename
        std::string edm4hepFileName = fileName;
        const std::size_t dotPos = edm4hepFileName.rfind('.');
        if (dotPos != std::string::npos) {
            edm4hepFileName = edm4hepFileName.substr(0, dotPos);
        }
        edm4hepFileName += ".edm4hep.root";

        fEDM4hepConfig.filename = edm4hepFileName;
        fEDM4hepWriter->Configure(fEDM4hepConfig);
        if (!fEDM4hepWriter->Open(edm4hepFileName)) {
            G4cout << "[RunAction] Warning: Failed to open EDM4hep file " << edm4hepFileName
                   << ". EDM4hep output disabled for this run." << G4endl;
        }
    }
}

void RunAction::ConfigureCoreBranches(TTree* tree) {
    if (!tree) {
        return;
    }

    // Build buffer structures for BranchConfigurator
    const IO::BranchConfigurator::ScalarBuffers scalars{.trueX = &fTrueX,
                                                        .trueY = &fTrueY,
                                                        .pixelX = &fPixelX,
                                                        .pixelY = &fPixelY,
                                                        .edep = &fEdep,
                                                        .pixelTrueDeltaX = &fPixelTrueDeltaX,
                                                        .pixelTrueDeltaY = &fPixelTrueDeltaY};

    const IO::BranchConfigurator::ClassificationBuffers classification{.isPixelHit = &fIsPixelHit,
                                                                       .neighborhoodActiveCells =
                                                                           &fNeighborhoodActiveCells,
                                                                       .nearestPixelI = &fNearestPixelI,
                                                                       .nearestPixelJ = &fNearestPixelJ,
                                                                       .nearestPixelGlobalId = &fNearestPixelGlobalId};

    IO::BranchConfigurator::VectorBuffers const vectors{.chargeFractions = &fNeighborhoodChargeFractions,
                                                        .chargeFractionsRow = &fNeighborhoodChargeFractionsRow,
                                                        .chargeFractionsCol = &fNeighborhoodChargeFractionsCol,
                                                        .chargeFractionsBlock = &fNeighborhoodChargeFractionsBlock,
                                                        .charge = &fNeighborhoodCharge,
                                                        .chargeNew = &fNeighborhoodChargeNew,
                                                        .chargeFinal = &fNeighborhoodChargeFinal,
                                                        .chargeRow = &fNeighborhoodChargeRow,
                                                        .chargeNewRow = &fNeighborhoodChargeNewRow,
                                                        .chargeFinalRow = &fNeighborhoodChargeFinalRow,
                                                        .chargeCol = &fNeighborhoodChargeCol,
                                                        .chargeNewCol = &fNeighborhoodChargeNewCol,
                                                        .chargeFinalCol = &fNeighborhoodChargeFinalCol,
                                                        .chargeBlock = &fNeighborhoodChargeBlock,
                                                        .chargeNewBlock = &fNeighborhoodChargeNewBlock,
                                                        .chargeFinalBlock = &fNeighborhoodChargeFinalBlock,
                                                        .pixelX = &fNeighborhoodPixelX,
                                                        .pixelY = &fNeighborhoodPixelY,
                                                        .distance = &fNeighborhoodDistance,
                                                        .alpha = &fNeighborhoodAlpha,
                                                        .pixelID = &fNeighborhoodPixelID};

    // Delegate to BranchConfigurator
    const auto mode = static_cast<ECS::Config::ActivePixelMode>(fActivePixelMode);
    const auto reconModel = static_cast<ECS::Config::PosReconModel>(fPosReconModel);
    fBranchConfigurator.ConfigureCoreBranches(tree, scalars, classification, vectors, mode, reconModel);
}

void RunAction::HandleWorkerEndOfRun(const ThreadContext& context, const G4Run* run) {
    const G4int nofEvents = run ? run->GetNumberOfEvent() : 0;

    bool wroteOutput = false;
    if (nofEvents > 0) {
        // In single-threaded mode, attach metadata before writing so the tree streamer
        // persists it for post-processing macros.
        if (!context.multithreaded) {
            WriteMetadataToTree(fRootWriter ? fRootWriter->Tree() : nullptr);
        }

        if (SafeWriteRootFile()) {
            wroteOutput = true;
            if (context.multithreaded) {
                G4cout << "[RunAction] Worker " << context.threadId << " stored " << nofEvents << " events" << G4endl;
            } else {
                G4cout << "[RunAction] Stored " << nofEvents << " events" << G4endl;
            }
        } else {
            G4Exception("RunAction::HandleWorkerEndOfRun", "RootFileWriteFailure", FatalException,
                        "SafeWriteRootFile() reported failure.");
        }
    }

    // Close EDM4hep output
    if (fEDM4hepWriter && fEDM4hepWriter->IsOpen()) {
        fEDM4hepWriter->Close();
    }

    CleanupRootObjects();

    if (!context.multithreaded && wroteOutput) {
        RunPostProcessingFits();
    }

    if (context.worker) {
        SignalWorkerCompletion();
    }
}

void RunAction::HandleMasterEndOfRun(const ThreadContext& context, const G4Run* /*run*/) {
    const std::vector<G4String> workerFiles = CollectWorkerFileNames(context.totalWorkers);
    const std::vector<G4String> existingFiles = FilterExistingWorkerFiles(workerFiles);

    if (existingFiles.empty()) {
        G4cout << "[RunAction] No worker ROOT files found to merge" << G4endl;
        return;
    }

    if (!MergeWorkerFilesAndPublishMetadata(existingFiles)) {
        return;
    }

    bool mergedHasEntries = false;
    if (!ValidateRootFile("epicChargeSharing.root", &mergedHasEntries)) {
        return;
    }

    for (const auto& file : existingFiles) {
        std::error_code ec;
        if (!std::filesystem::remove(file.c_str(), ec) && ec) {
            continue;
        }
    }

    G4cout << "[RunAction] Merged " << existingFiles.size() << " worker files into epicChargeSharing.root" << G4endl;

    if (!mergedHasEntries) {
        G4cout << "[RunAction] Merged ROOT file has no entries; skipping post-processing" << G4endl;
        return;
    }

    RunPostProcessingFits();
}

std::vector<G4String> RunAction::CollectWorkerFileNames(G4int totalWorkers) {
    std::vector<G4String> workerFiles;
    workerFiles.reserve(std::max<G4int>(0, totalWorkers));
    for (G4int tid = 0; tid < totalWorkers; ++tid) {
        workerFiles.push_back(ECS::WorkerFileName(tid));
    }
    return workerFiles;
}

std::vector<G4String> RunAction::FilterExistingWorkerFiles(const std::vector<G4String>& workerFiles) {
    std::vector<G4String> existingFiles;
    existingFiles.reserve(workerFiles.size());
    for (const auto& wf : workerFiles) {
        const std::filesystem::path path(wf.c_str());
        if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
            existingFiles.push_back(wf);
        }
    }
    return existingFiles;
}

bool RunAction::MergeWorkerFilesAndPublishMetadata(const std::vector<G4String>& existingFiles) {
    if (existingFiles.empty()) {
        return false;
    }

    auto mergeFiles = [](const std::vector<G4String>& inputs, const G4String& output) -> bool {
        if (inputs.empty()) {
            return false;
        }

        TFileMerger merger(kFALSE);
        merger.SetFastMethod(kFALSE);
        merger.SetNotrees(kFALSE);
        if (!merger.OutputFile(output.c_str(), "RECREATE")) {
            G4Exception("RunAction::MergeWorkerFilesAndPublishMetadata", "OutputFileFailure", FatalException,
                        "Unable to open output file for ROOT merge.");
            return false;
        }

        bool added = false;
        for (const auto& file : inputs) {
            if (std::filesystem::exists(file.c_str())) {
                if (merger.AddFile(file.c_str())) {
                    added = true;
                }
            }
        }

        if (!added) {
            return false;
        }

        return merger.Merge();
    };

    bool mergeOk = false;
    {
        std::lock_guard<std::mutex> const ioLock(ECS::RootIOMutex());
        mergeOk = mergeFiles(existingFiles, "epicChargeSharing.root");
        if (mergeOk) {
            std::unique_ptr<TFile> mergedFile(TFile::Open("epicChargeSharing.root", "UPDATE"));
            if (mergedFile && !mergedFile->IsZombie()) {
                auto* tree = mergedFile->Get<TTree>("Hits");
                if (tree) {
                    IO::MetadataPublisher::EntryList const entries = CollectMetadataEntries();
                    if (!entries.empty()) {
                        IO::MetadataPublisher::WriteEntriesToUserInfo(tree, entries);
                    }
                    tree->Write("", TObject::kOverwrite);
                }
                mergedFile->Close();
            } else {
                G4Exception("RunAction::MergeWorkerFilesAndPublishMetadata", "MergedFileOpenFailure", FatalException,
                            "Failed to open merged ROOT file for metadata update.");
                mergeOk = false;
            }
        }
    }

    return mergeOk;
}

void RunAction::UpdateSummaryScalars(const EventRecord& record) {
    fEdep = record.summary.edep;
    fTrueX = record.summary.hitX;
    fTrueY = record.summary.hitY;
    (void)record.summary.hitZ;
    fPixelX = record.summary.nearestPixelX;
    fPixelY = record.summary.nearestPixelY;
    fPixelTrueDeltaX = record.summary.pixelTrueDeltaX;
    fPixelTrueDeltaY = record.summary.pixelTrueDeltaY;
    fFirstContactIsPixel = record.summary.firstContactIsPixel;
    fGeometricIsPixel = record.summary.geometricIsPixel;
    fIsPixelHit = record.summary.isPixelHitCombined;

    fNearestPixelI = record.nearestPixelI;
    fNearestPixelJ = record.nearestPixelJ;
    fNearestPixelGlobalId = record.nearestPixelGlobalId;
}

void RunAction::PrepareNeighborhoodStorage(std::size_t requestedCells) {
    const std::size_t capacity = std::max<std::size_t>(1, requestedCells);
    fNeighborhoodCapacity = capacity;
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    const G4double sentinelFrac = Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL;

    // Fractions
    ECS::ResizeAndFill(fNeighborhoodChargeFractions, capacity, sentinelFrac);
    ECS::ResizeAndFill(fNeighborhoodChargeFractionsRow, capacity, sentinelFrac);
    ECS::ResizeAndFill(fNeighborhoodChargeFractionsCol, capacity, sentinelFrac);
    ECS::ResizeAndFill(fNeighborhoodChargeFractionsBlock, capacity, sentinelFrac);
    // Neighborhood charges
    ECS::ResizeAndFill(fNeighborhoodCharge, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeNew, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeFinal, capacity, 0.0);
    // Row-mode charges
    ECS::ResizeAndFill(fNeighborhoodChargeRow, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeNewRow, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeFinalRow, capacity, 0.0);
    // Col-mode charges
    ECS::ResizeAndFill(fNeighborhoodChargeCol, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeNewCol, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeFinalCol, capacity, 0.0);
    // Block-mode charges
    ECS::ResizeAndFill(fNeighborhoodChargeBlock, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeNewBlock, capacity, 0.0);
    ECS::ResizeAndFill(fNeighborhoodChargeFinalBlock, capacity, 0.0);
    // Geometry
    ECS::ResizeAndFill(fNeighborhoodDistance, capacity, nan);
    ECS::ResizeAndFill(fNeighborhoodAlpha, capacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelX, capacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelY, capacity, nan);
    ECS::ResizeAndFill(fNeighborhoodPixelID, capacity, -1);
    fNeighborhoodActiveCells = 0;
}

void RunAction::PopulateNeighborhoodFromRecord(const EventRecord& record) {
    auto copyCharges = [](const auto& src, auto& dst) {
        const std::size_t n = std::min(dst.size(), src.size());
        if (n > 0)
            std::copy_n(src.begin(), n, dst.begin());
    };

    // Neighborhood-mode noisy charges
    copyCharges(record.neighborChargesNew, fNeighborhoodChargeNew);
    copyCharges(record.neighborChargesFinal, fNeighborhoodChargeFinal);
    // Row-mode noisy charges
    copyCharges(record.neighborChargesNewRow, fNeighborhoodChargeNewRow);
    copyCharges(record.neighborChargesFinalRow, fNeighborhoodChargeFinalRow);
    // Col-mode noisy charges
    copyCharges(record.neighborChargesNewCol, fNeighborhoodChargeNewCol);
    copyCharges(record.neighborChargesFinalCol, fNeighborhoodChargeFinalCol);
    // Block-mode noisy charges
    copyCharges(record.neighborChargesNewBlock, fNeighborhoodChargeNewBlock);
    copyCharges(record.neighborChargesFinalBlock, fNeighborhoodChargeFinalBlock);

    std::size_t activeCells = 0;
    for (const auto& cell : record.neighborCells) {
        if (cell.gridIndex < 0) {
            continue;
        }
        const auto idx = static_cast<std::size_t>(cell.gridIndex);
        if (idx >= fNeighborhoodCapacity) {
            continue;
        }
        // Fractions
        fNeighborhoodChargeFractions[idx] = cell.fraction;
        fNeighborhoodChargeFractionsRow[idx] = cell.fractionRow;
        fNeighborhoodChargeFractionsCol[idx] = cell.fractionCol;
        fNeighborhoodChargeFractionsBlock[idx] = cell.fractionBlock;
        // Neighborhood charge
        fNeighborhoodCharge[idx] = cell.charge;
        // Mode-specific charges
        fNeighborhoodChargeRow[idx] = cell.chargeRow;
        fNeighborhoodChargeCol[idx] = cell.chargeCol;
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

void RunAction::PopulateFullFractionsFromRecord(const EventRecord& record) {
    if (!fStoreFullFractions) {
        return;
    }

    const G4int gridSide = (record.fullGridCols > 0) ? record.fullGridCols : fGridNumBlocksPerSide;
    if (!EnsureFullFractionBuffer(gridSide)) {
        return;
    }

    if (record.fullGridCols > 0) {
        fFullGridSide = record.fullGridCols;
    }

    const auto copyOrZero = [](const std::span<const G4double>& source, std::vector<G4double>& target) {
        if (target.empty()) {
            return;
        }
        if (source.empty()) {
            std::ranges::fill(target, 0.0);
            return;
        }
        const std::size_t n = std::min<std::size_t>(target.size(), source.size());
        std::copy_n(source.begin(), n, target.begin());
        if (n < target.size()) {
            std::fill(target.begin() + static_cast<std::ptrdiff_t>(n), target.end(), 0.0);
        }
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

void RunAction::RunPostProcessingFits() {
    // Delegate to PostProcessingRunner helper
    IO::PostProcessingRunner::Config config;
    config.runFitGaus1D = Constants::FIT_GAUS_1D;
    config.runFitGaus2D = Constants::FIT_GAUS_2D;
    config.sourceDir = PROJECT_SOURCE_DIR;
    config.rootFileName = "epicChargeSharing.root";

    fPostProcessingRunner.SetConfig(config);
    fPostProcessingRunner.Run();
}

void RunAction::ResetSynchronization() {
    if (G4Threading::IsMultithreadedApplication()) {
        ECS::WorkerSync::Instance().Reset(G4Threading::GetNumberOfRunningWorkerThreads());
    }
}

void RunAction::SignalWorkerCompletion() {
    if (G4Threading::IsMultithreadedApplication()) {
        ECS::WorkerSync::Instance().SignalCompletion();
    }
}

void RunAction::WaitForAllWorkersToComplete() {
    if (G4Threading::IsMultithreadedApplication()) {
        ECS::WorkerSync::Instance().WaitForAll();
    }
}

bool RunAction::SafeWriteRootFile() {
    const bool isMT = G4Threading::IsMultithreadedApplication();
    const bool isWorker = G4Threading::IsWorkerThread();
    return fRootWriter->SafeWrite(isMT, isWorker);
}

bool RunAction::ValidateRootFile(const G4String& filename, bool* hasEntries) {
    return fRootWriter->Validate(filename, hasEntries);
}

void RunAction::CleanupRootObjects() {
    fRootWriter->Cleanup();
}

void RunAction::SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, G4double gridOffset,
                                          G4double detSize, G4int numBlocksPerSide) {
    fGridPixelSize = pixelSize;
    fGridPixelSpacing = pixelSpacing;
    fGridOffset = gridOffset;
    fGridDetSize = detSize;
    fGridNumBlocksPerSide = numBlocksPerSide;
    fFullGridSide = numBlocksPerSide;
    fChargeSharingPitch = pixelSpacing;
}

void RunAction::SetNeighborhoodRadiusMeta(G4int radius) {
    fGridNeighborhoodRadius = radius;
    fNeighborhoodLayout.SetRadius(radius);

    if (GetTree()) {
        return;
    }

    fNeighborhoodCapacity = 0;
    fNeighborhoodChargeFractions.clear();
    fNeighborhoodCharge.clear();
    fNeighborhoodChargeNew.clear();
    fNeighborhoodChargeFinal.clear();
    fNeighborhoodDistance.clear();
    fNeighborhoodAlpha.clear();
    fNeighborhoodPixelX.clear();
    fNeighborhoodPixelY.clear();
    fNeighborhoodPixelID.clear();
    fNeighborhoodActiveCells = 0;
}

void RunAction::SetPosReconMetadata(Constants::PosReconModel model, G4double betaPerMicron, G4double pitch) {
    fPosReconModel = model;
    // Beta is only used when LinA signal model is active
    if (Constants::USES_LINEAR_SIGNAL && std::isfinite(betaPerMicron)) {
        fChargeSharingBeta = betaPerMicron;
    } else {
        fChargeSharingBeta = std::numeric_limits<G4double>::quiet_NaN();
    }

    if (pitch > 0.0) {
        fChargeSharingPitch = pitch;
    }
}

void RunAction::SetChargeSharingDistanceAlphaMeta(G4bool enabled) {
    fEmitDistanceAlphaMeta = enabled;
}

void RunAction::SetGridPixelCenters(const std::vector<G4ThreeVector>& centers) {
    const std::size_t count = centers.size();
    fGridPixelID.resize(count);
    fGridPixelX.resize(count);
    fGridPixelY.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& center = centers[i];
        fGridPixelID[i] = static_cast<G4int>(i);
        fGridPixelX[i] = center.x();
        fGridPixelY[i] = center.y();
    }
}

void RunAction::ConfigureFullFractionBranch(G4bool enable) {
    fStoreFullFractions = enable;
    if (!enable) {
        return;
    }

    auto treeLock = MakeTreeLock();
    auto* tree = GetTree();
    if (!tree) {
        return;
    }

    if (fFullFractionsBranchInitialized) {
        return;
    }

    if (!EnsureFullFractionBuffer(fGridNumBlocksPerSide)) {
        return;
    }

    // Build buffer structure for BranchConfigurator
    IO::BranchConfigurator::FullGridBuffers const buffers{.fi = &fFullFi,
                                                          .fiRow = &fFullFiRow,
                                                          .fiCol = &fFullFiCol,
                                                          .fiBlock = &fFullFiBlock,
                                                          .qi = &fFullQi,
                                                          .qn = &fFullQn,
                                                          .qf = &fFullQf,
                                                          .qiRow = &fFullQiRow,
                                                          .qnRow = &fFullQnRow,
                                                          .qfRow = &fFullQfRow,
                                                          .qiCol = &fFullQiCol,
                                                          .qnCol = &fFullQnCol,
                                                          .qfCol = &fFullQfCol,
                                                          .qiBlock = &fFullQiBlock,
                                                          .qnBlock = &fFullQnBlock,
                                                          .qfBlock = &fFullQfBlock,
                                                          .distance = &fFullDistance,
                                                          .alpha = &fFullAlpha,
                                                          .pixelX = &fFullPixelXGrid,
                                                          .pixelY = &fFullPixelYGrid,
                                                          .gridSide = &fFullGridSide};

    // Delegate to BranchConfigurator
    const auto mode = static_cast<ECS::Config::ActivePixelMode>(fActivePixelMode);
    fFullFractionsBranchInitialized = fBranchConfigurator.ConfigureFullGridBranches(tree, buffers, mode);
}

ECS::IO::MetadataPublisher RunAction::BuildMetadataPublisher() const {
    ECS::IO::MetadataPublisher publisher;
    const auto& rtConfig = ECS::RuntimeConfig::Instance();

    // Simulation info (timestamp and versions)
    ECS::IO::MetadataPublisher::SimulationMetadata sim;
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
        sim.timestamp = oss.str();
    }
    sim.geant4Version = G4Version;
    sim.rootVersion = ROOT_RELEASE;
    publisher.SetSimulationMetadata(sim);

    // Grid geometry metadata
    ECS::IO::MetadataPublisher::GridMetadata grid;
    grid.pixelSize = fGridPixelSize;
    grid.pixelSpacing = fGridPixelSpacing;
    grid.gridOffset = fGridOffset;
    grid.detectorSize = fGridDetSize;
    grid.detectorThickness = rtConfig.detectorWidth / CLHEP::mm;
    grid.interpadGap = (fGridPixelSpacing - fGridPixelSize);
    grid.numBlocksPerSide = fGridNumBlocksPerSide;
    grid.neighborhoodRadius = fGridNeighborhoodRadius;
    grid.fullGridSide = fFullGridSide;
    grid.storeFullFractions = fStoreFullFractions;
    publisher.SetGridMetadata(grid);

    // Charge sharing model metadata
    ECS::IO::MetadataPublisher::ModelMetadata model;
    model.signalModel = Constants::SIGNAL_MODEL;
    model.model = static_cast<ECS::Config::PosReconModel>(fPosReconModel);
    model.activePixelMode = static_cast<ECS::Config::ActivePixelMode>(fActivePixelMode);
    model.beta = fChargeSharingBeta;
    publisher.SetModelMetadata(model);

    // Physics constants metadata (from runtime config for sweep support)
    ECS::IO::MetadataPublisher::PhysicsMetadata physics;
    physics.d0 = rtConfig.d0;
    physics.ionizationEnergy = rtConfig.ionizationEnergy;
    physics.gain = rtConfig.gain;
    publisher.SetPhysicsMetadata(physics);

    // Noise model metadata (from runtime config for sweep support)
    ECS::IO::MetadataPublisher::NoiseMetadata noise;
    noise.gainSigmaMin = rtConfig.pixelGainSigmaMin;
    noise.gainSigmaMax = rtConfig.pixelGainSigmaMax;
    noise.electronCount = rtConfig.noiseElectronCount;
    publisher.SetNoiseMetadata(noise);

    return publisher;
}

void RunAction::WriteMetadataToTree(TTree* tree) const {
    if (!tree) {
        return;
    }

    const IO::MetadataPublisher publisher = BuildMetadataPublisher();
    std::mutex& ioMutex = ECS::RootIOMutex();
    publisher.WriteToTree(tree, &ioMutex);
}

ECS::IO::MetadataPublisher::EntryList RunAction::CollectMetadataEntries() const {
    return BuildMetadataPublisher().CollectEntries();
}

void RunAction::FillTree(const EventRecord& record, std::uint64_t eventNumber, G4int runNumber) {
    auto treeLock = MakeTreeLock();
    auto* tree = GetTree();
    if (!tree) {
        return;
    }

    if (fStoreFullFractions && !fFullFractionsBranchInitialized) {
        ConfigureFullFractionBranch(true);
    }

    UpdateSummaryScalars(record);

    const std::size_t requestedCells = record.totalGridCells > 0
                                           ? static_cast<std::size_t>(record.totalGridCells)
                                           : std::max<std::size_t>(1, fNeighborhoodLayout.TotalCells());

    PrepareNeighborhoodStorage(requestedCells);
    PopulateNeighborhoodFromRecord(record);
    PopulateFullFractionsFromRecord(record);

    if (tree->Fill() < 0) {
        G4Exception("RunAction::FillTree", "TreeFillFailure", FatalException, "TTree::Fill() reported an error.");
    }

    // Write to EDM4hep output if enabled
    if (fWriteEDM4hep && fEDM4hepWriter && fEDM4hepWriter->IsOpen()) {
        fEDM4hepWriter->WriteEvent(record, eventNumber, runNumber);
    }
}

void RunAction::SetEDM4hepEnabled(G4bool enabled) {
    fWriteEDM4hep = enabled;
    if (fEDM4hepWriter) {
        fEDM4hepConfig.enabled = enabled;
        fEDM4hepWriter->Configure(fEDM4hepConfig);
    }
}

G4bool RunAction::IsEDM4hepEnabled() const {
    return fWriteEDM4hep && fEDM4hepWriter && fEDM4hepWriter->IsEnabled();
}
