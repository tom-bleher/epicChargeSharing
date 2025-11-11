/**
 * @file RunAction.cc
 * @brief Run lifecycle management with dedicated helpers for ROOT I/O and worker synchronisation.
 */
#include "RunAction.hh"

#include "Constants.hh"
#include "DetectorConstruction.hh"
#include "internal/RunActionSupport.hh"

#include "G4RunManager.hh"
#include "G4Exception.hh"
#include "G4Threading.hh"
#include "G4SystemOfUnits.hh"

#include "TFile.h"
#include "TTree.h"
#include "TNamed.h"
#include "TError.h"
#include "TROOT.h"
#include "TThread.h"
#include "TFileMerger.h"
#include "RVersion.h"
#include "TString.h"
#include "TSystem.h"

#include <algorithm>
#include <cstddef>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace
{
std::once_flag gRootInitFlag;
} // namespace

using runaction::RootFileWriterHelper;
namespace support = runaction::support;

namespace
{
using MetadataEntries = std::vector<std::pair<std::string, std::string>>;

std::string BoolToString(bool value)
{
    return value ? "true" : "false";
}

void WriteMetadataEntriesUnlocked(TFile* file, const MetadataEntries& entries)
{
    if (!file || file->IsZombie()) {
        G4Exception("RunAction::WriteMetadataEntriesUnlocked",
                    "InvalidRootFile",
                    FatalException,
                    "Unable to write metadata because the ROOT file handle is invalid.");
        return;
    }

    file->cd();
    for (const auto& entry : entries) {
        TNamed meta(entry.first.c_str(), entry.second.c_str());
        meta.Write("", TObject::kOverwrite);
    }
    file->Flush();
}
} // namespace

RunAction::RunAction()
    : G4UserRunAction(),
      fRootWriter(std::make_unique<RootFileWriterHelper>()),
      fTrueX(0.0),
      fTrueY(0.0),
      fPixelX(0.0),
      fPixelY(0.0),
      fEdep(0.0),
      fPixelTrueDeltaX(0.0),
      fPixelTrueDeltaY(0.0),
      fNeighborhoodLayout(Constants::NEIGHBORHOOD_RADIUS),
      fGridPixelSize(0.0),
      fGridPixelSpacing(0.0),
      fGridPixelCornerOffset(0.0),
      fGridDetSize(0.0),
      fGridNumBlocksPerSide(0),
      fGridNeighborhoodRadius(0),
      fChargeSharingModel(Constants::CHARGE_SHARING_MODEL),
      fChargeSharingBeta(std::numeric_limits<G4double>::quiet_NaN()),
      fChargeSharingPitch(0.0),
      fEmitDistanceAlphaMeta(false),
      fStoreFullFractions(Constants::STORE_FULL_GRID),
      fNearestPixelI(-1),
      fNearestPixelJ(-1),
      fNearestPixelGlobalId(-1)
{
}

RunAction::~RunAction()
{
    CleanupRootObjects();
}

std::unique_lock<std::mutex> RunAction::MakeTreeLock()
{
#ifdef G4MULTITHREADED
    if (G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread()) {
        return std::unique_lock<std::mutex>(fTreeMutex);
    }
#endif
    return std::unique_lock<std::mutex>();
}

TFile* RunAction::GetRootFile() const
{
    return fRootWriter ? fRootWriter->File() : nullptr;
}

TTree* RunAction::GetTree() const
{
    return fRootWriter ? fRootWriter->Tree() : nullptr;
}

void RunAction::EnsureBranchBuffersInitialized()
{
    const G4int radius =
        (fGridNeighborhoodRadius >= 0) ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS;
    fNeighborhoodLayout.SetRadius(radius);
    const std::size_t desiredCapacity =
        std::max<std::size_t>(1, fNeighborhoodLayout.TotalCells());

    if (desiredCapacity != fNeighborhoodCapacity ||
        fNeighborhoodChargeFractions.size() != desiredCapacity) {
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
    neighbor::ResizeAndFill(fNeighborhoodChargeFractions,
                            fNeighborhoodCapacity,
                            Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    neighbor::ResizeAndFill(fNeighborhoodCharge, fNeighborhoodCapacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeNew, fNeighborhoodCapacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinal, fNeighborhoodCapacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodDistance, fNeighborhoodCapacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodAlpha, fNeighborhoodCapacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelX, fNeighborhoodCapacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelY, fNeighborhoodCapacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelID, fNeighborhoodCapacity, -1);
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

bool RunAction::EnsureFullFractionBuffer(G4int gridSide)
{
    if (gridSide >= 0) {
        fFullGridSide = gridSide;
    }
    const G4int side = (fFullGridSide > 0) ? fFullGridSide : fGridNumBlocksPerSide;
    const G4int numBlocks = std::max(0, side);
    const std::size_t totalPixels =
        static_cast<std::size_t>(numBlocks) * static_cast<std::size_t>(numBlocks);
    if (totalPixels == 0U) {
        fFullFi.clear();
        fFullQi.clear();
        fFullQn.clear();
        fFullQf.clear();
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

    ensure(fFullFi);
    ensure(fFullQi);
    ensure(fFullQn);
    ensure(fFullQf);
    ensure(fFullDistance);
    ensure(fFullAlpha);
    ensure(fFullPixelXGrid);
    ensure(fFullPixelYGrid);
    return true;
}

void RunAction::BeginOfRunAction(const G4Run* run)
{
    std::call_once(gRootInitFlag, support::InitializeROOTThreading);

    EnsureBranchBuffersInitialized();

    if (!run) {
        G4Exception("RunAction::BeginOfRunAction",
                    "InvalidRunObject",
                    FatalException,
                    "Received null run object in BeginOfRunAction.");
        return;
    }

    const ThreadContext context = BuildThreadContext(run);

    LogBeginRun(context);

    if (context.multithreaded && context.master) {
        support::WorkerSyncHelper::Instance().Reset(context.totalWorkers);
    }

    const G4String fileName = DetermineOutputFileName(context);

    if (!context.multithreaded || context.worker) {
        InitializeRootOutputs(context, fileName);
    } else {
        fRootWriter->Attach(nullptr, nullptr, false);
    }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    if (!run) {
        G4Exception("RunAction::EndOfRunAction",
                    "InvalidRunObject",
                    FatalException,
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

RunAction::ThreadContext RunAction::BuildThreadContext(const G4Run* run) const
{
    ThreadContext context;
    context.multithreaded = G4Threading::IsMultithreadedApplication();
#ifdef G4MULTITHREADED
    context.worker = context.multithreaded && G4Threading::IsWorkerThread();
    context.master = context.multithreaded && !context.worker;
    context.threadId = context.worker ? G4Threading::G4GetThreadId() : -1;
    context.totalWorkers =
        context.multithreaded ? G4Threading::GetNumberOfRunningWorkerThreads() : 0;
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

void RunAction::LogBeginRun(const ThreadContext& context) const
{
    const G4int runId = context.runId;
    if (context.multithreaded) {
        if (context.worker) {
            G4cout << "[RunAction] Worker " << context.threadId << " beginning run " << runId
                   << G4endl;
        } else {
            G4cout << "[RunAction] Master beginning run " << runId << " with "
                   << context.totalWorkers << " workers" << G4endl;
        }
    } else {
        G4cout << "[RunAction] Beginning run " << runId << " (single-threaded)" << G4endl;
    }
}

G4String RunAction::DetermineOutputFileName(const ThreadContext& context) const
{
    if (!context.multithreaded) {
        return "epicChargeSharing.root";
    }
    if (context.worker) {
        return support::WorkerFileName(context.threadId);
    }
    return "epicChargeSharing.root";
}

void RunAction::InitializeRootOutputs(const ThreadContext& context, const G4String& fileName)
{
    (void)context;

    std::unique_lock<std::mutex> rootLock(support::RootIOMutex());
    auto* rootFile = new TFile(fileName.c_str(), "RECREATE");
    if (!rootFile || rootFile->IsZombie()) {
        G4Exception("RunAction::InitializeRootOutputs",
                    "RootFileOpenFailure",
                    FatalException,
                    ("Unable to open ROOT file " + std::string(fileName)).c_str());
    }
    rootFile->SetCompressionLevel(0);
    rootFile->cd();

    auto* tree = new TTree("Hits", "AC-LGAD charge sharing hits");
    tree->SetDirectory(rootFile);
    tree->SetAutoSave(0);
    tree->SetAutoFlush(0);
    tree->SetMaxTreeSize(static_cast<Long64_t>(10LL) * 1024LL * 1024LL * 1024LL);
    ConfigureCoreBranches(tree);

    fRootWriter->Attach(rootFile, tree, true);

    if (fStoreFullFractions) {
        ConfigureFullFractionBranch(true);
    }
}

void RunAction::ConfigureCoreBranches(TTree* tree)
{
    if (!tree) {
        return;
    }

    ConfigureScalarBranches(tree);
    ConfigureClassificationBranches(tree);
    ConfigureVectorBranches(tree);
    ConfigureNeighborhoodBranches(tree);
}

void RunAction::ConfigureScalarBranches(TTree* tree)
{
    struct ScalarBranchDef
    {
        const char* name;
        double* address;
        const char* leaf;
    };
    const std::array<ScalarBranchDef, 7> scalarBranches{{
        {"TrueX", &fTrueX, "TrueX/D"},
        {"TrueY", &fTrueY, "TrueY/D"},
        {"PixelX", &fPixelX, "PixelX/D"},
        {"PixelY", &fPixelY, "PixelY/D"},
        {"Edep", &fEdep, "Edep/D"},
        {"PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D"},
        {"PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D"},
    }};
    for (const auto& def : scalarBranches) {
        tree->Branch(def.name, def.address, def.leaf);
    }
}

void RunAction::ConfigureClassificationBranches(TTree* tree)
{
    tree->Branch("isPixelHit", &fIsPixelHit, "isPixelHit/O");
    tree->Branch("NeighborhoodSize", &fNeighborhoodActiveCells, "NeighborhoodSize/I");
    tree->Branch("NearestPixelI", &fNearestPixelI, "NearestPixelI/I");
    tree->Branch("NearestPixelJ", &fNearestPixelJ, "NearestPixelJ/I");
    tree->Branch("NearestPixelID", &fNearestPixelGlobalId, "NearestPixelID/I");
}

void RunAction::ConfigureVectorBranches(TTree* tree)
{
    constexpr Int_t bufsize = 256000;
    const Int_t splitLevel = 0;
    const std::array<std::pair<const char*, std::vector<G4double>*>, 8> vectorDoubleBranches{{
        {"Fi", &fNeighborhoodChargeFractions},
        {"Qi", &fNeighborhoodCharge},
        {"Qn", &fNeighborhoodChargeNew},
        {"Qf", &fNeighborhoodChargeFinal},
        {"NeighborhoodPixelX", &fNeighborhoodPixelX},
        {"NeighborhoodPixelY", &fNeighborhoodPixelY},
        {"NeighborhoodDistance", &fNeighborhoodDistance},
        {"NeighborhoodAlpha", &fNeighborhoodAlpha},
    }};
    for (const auto& entry : vectorDoubleBranches) {
        tree->Branch(entry.first, entry.second, bufsize, splitLevel);
    }
}

void RunAction::ConfigureNeighborhoodBranches(TTree* tree)
{
    constexpr Int_t bufsize = 256000;
    const Int_t splitLevel = 0;
    tree->Branch("NeighborhoodPixelID", &fNeighborhoodPixelID, bufsize, splitLevel);
}

void RunAction::HandleWorkerEndOfRun(const ThreadContext& context, const G4Run* run)
{
    const G4int nofEvents = run ? run->GetNumberOfEvent() : 0;

    bool wroteOutput = false;
    if (nofEvents > 0) {
        if (SafeWriteRootFile()) {
            wroteOutput = true;
            if (context.multithreaded) {
                G4cout << "[RunAction] Worker " << context.threadId << " stored " << nofEvents
                       << " events" << G4endl;
            } else {
                G4cout << "[RunAction] Stored " << nofEvents << " events" << G4endl;
            }
            if (!context.multithreaded) {
                WriteMetadataToFile(fRootWriter ? fRootWriter->File() : nullptr);
            }
        } else {
            G4Exception("RunAction::HandleWorkerEndOfRun",
                        "RootFileWriteFailure",
                        FatalException,
                        "SafeWriteRootFile() reported failure.");
        }
    }

    CleanupRootObjects();

    if (!context.multithreaded && wroteOutput) {
        RunPostProcessingFits();
    }

    if (context.worker) {
        SignalWorkerCompletion();
    }
}

void RunAction::HandleMasterEndOfRun(const ThreadContext& context, const G4Run* /*run*/)
{
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

    G4cout << "[RunAction] Merged " << existingFiles.size()
           << " worker files into epicChargeSharing.root" << G4endl;

    if (!mergedHasEntries) {
        G4cout << "[RunAction] Merged ROOT file has no entries; skipping post-processing" << G4endl;
        return;
    }

    RunPostProcessingFits();
}

std::vector<G4String> RunAction::CollectWorkerFileNames(G4int totalWorkers) const
{
    std::vector<G4String> workerFiles;
    workerFiles.reserve(std::max<G4int>(0, totalWorkers));
    for (G4int tid = 0; tid < totalWorkers; ++tid) {
        workerFiles.push_back(support::WorkerFileName(tid));
    }
    return workerFiles;
}

std::vector<G4String> RunAction::FilterExistingWorkerFiles(const std::vector<G4String>& workerFiles) const
{
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

bool RunAction::MergeWorkerFilesAndPublishMetadata(const std::vector<G4String>& existingFiles)
{
    if (existingFiles.empty()) {
        return false;
    }

    auto mergeFiles = [](const std::vector<G4String>& inputs,
                         const G4String& output) -> bool {
        if (inputs.empty()) {
            return false;
        }

        TFileMerger merger(kFALSE);
        merger.SetFastMethod(kFALSE);
        merger.SetNotrees(kFALSE);
        if (!merger.OutputFile(output.c_str(), "RECREATE")) {
            G4Exception("RunAction::MergeWorkerFilesAndPublishMetadata",
                        "OutputFileFailure",
                        FatalException,
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
        std::lock_guard<std::mutex> ioLock(support::RootIOMutex());
        mergeOk = mergeFiles(existingFiles, "epicChargeSharing.root");
        if (mergeOk) {
            std::unique_ptr<TFile> mergedFile(TFile::Open("epicChargeSharing.root", "UPDATE"));
            if (mergedFile && !mergedFile->IsZombie()) {
                MetadataEntries entries = CollectMetadataEntries();
                if (!entries.empty()) {
                    WriteMetadataEntriesUnlocked(mergedFile.get(), entries);
                }
                mergedFile->Close();
            } else {
                G4Exception("RunAction::MergeWorkerFilesAndPublishMetadata",
                            "MergedFileOpenFailure",
                            FatalException,
                            "Failed to open merged ROOT file for metadata update.");
                mergeOk = false;
            }
        }
    }

    return mergeOk;
}

void RunAction::UpdateSummaryScalars(const EventRecord& record)
{
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

void RunAction::PrepareNeighborhoodStorage(std::size_t requestedCells)
{
    const std::size_t capacity = std::max<std::size_t>(1, requestedCells);
    fNeighborhoodCapacity = capacity;
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    neighbor::ResizeAndFill(fNeighborhoodChargeFractions,
                            capacity,
                            Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    neighbor::ResizeAndFill(fNeighborhoodCharge, capacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeNew, capacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodChargeFinal, capacity, 0.0);
    neighbor::ResizeAndFill(fNeighborhoodDistance, capacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodAlpha, capacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelX, capacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelY, capacity, nan);
    neighbor::ResizeAndFill(fNeighborhoodPixelID, capacity, -1);
    fNeighborhoodActiveCells = 0;
}

void RunAction::PopulateNeighborhoodFromRecord(const EventRecord& record)
{
    const std::size_t copyNew =
        std::min<std::size_t>(fNeighborhoodChargeNew.size(), record.neighborChargesNew.size());
    if (copyNew > 0) {
        std::copy_n(record.neighborChargesNew.begin(), copyNew, fNeighborhoodChargeNew.begin());
    }

    const std::size_t copyFinal =
        std::min<std::size_t>(fNeighborhoodChargeFinal.size(), record.neighborChargesFinal.size());
    if (copyFinal > 0) {
        std::copy_n(record.neighborChargesFinal.begin(),
                    copyFinal,
                    fNeighborhoodChargeFinal.begin());
    }

    std::size_t activeCells = 0;
    for (const auto& cell : record.neighborCells) {
        if (cell.gridIndex < 0) {
            continue;
        }
        const auto idx = static_cast<std::size_t>(cell.gridIndex);
        if (idx >= fNeighborhoodCapacity) {
            continue;
        }
        fNeighborhoodChargeFractions[idx] = cell.fraction;
        fNeighborhoodCharge[idx] = cell.charge;
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

void RunAction::PopulateFullFractionsFromRecord(const EventRecord& record)
{
    if (!fStoreFullFractions) {
        return;
    }

    const G4int gridSide =
        (record.fullGridCols > 0) ? record.fullGridCols : fGridNumBlocksPerSide;
    if (!EnsureFullFractionBuffer(gridSide)) {
        return;
    }

    if (record.fullGridCols > 0) {
        fFullGridSide = record.fullGridCols;
    }

    const auto copyOrZero = [](const std::span<const G4double>& source,
                               std::vector<G4double>& target) {
        if (target.empty()) {
            return;
        }
        if (source.empty()) {
            std::fill(target.begin(), target.end(), 0.0);
            return;
        }
        const std::size_t n = std::min<std::size_t>(target.size(), source.size());
        std::copy_n(source.begin(), n, target.begin());
        if (n < target.size()) {
            std::fill(target.begin() + static_cast<std::ptrdiff_t>(n), target.end(), 0.0);
        }
    };

    copyOrZero(record.fullFi, fFullFi);
    copyOrZero(record.fullQi, fFullQi);
    copyOrZero(record.fullQn, fFullQn);
    copyOrZero(record.fullQf, fFullQf);
    copyOrZero(record.fullDistance, fFullDistance);
    copyOrZero(record.fullAlpha, fFullAlpha);
    copyOrZero(record.fullPixelX, fFullPixelXGrid);
    copyOrZero(record.fullPixelY, fFullPixelYGrid);
}

void RunAction::RunPostProcessingFits()
{
    // Optionally execute ROOT macros that write reconstructed branches into epicChargeSharing.root
    if (!Constants::FIT_GAUS_1D && !Constants::FIT_GAUS_2D) {
        return;
    }

    // Ensure ROOT is in batch mode to avoid any UI attempts
    gROOT->SetBatch(true);

    // Build POSIX-style paths so the ROOT interpreter handles them consistently on Windows
    TString sourceDir = TString(PROJECT_SOURCE_DIR);
    sourceDir.ReplaceAll("\\", "/");
    const TString macro1D = sourceDir + "/src/FitGaus1D.C";
    const TString macro2D = sourceDir + "/src/FitGaus2D.C";
    const TString rootFile = "epicChargeSharing.root"; // produced in current working directory

    bool executed = false;
    auto runMacro = [&](const TString& macroPath, const char* entryPoint) {
        // Load the macro (interpreted). Using interpreted mode avoids platform-specific ACLiC complications.
        G4cout << "[RunAction] Running post-processing macro " << entryPoint << "..." << G4endl;
        gROOT->ProcessLine(TString::Format(".L %s", macroPath.Data()));
        // Call the entry point with just the filename, relying on macro defaults for other parameters.
        const TString call = TString::Format("%s(\"%s\")", entryPoint, rootFile.Data());
        gROOT->ProcessLine(call);
        executed = true;
    };

    // 1D row/column fits
    if (Constants::FIT_GAUS_1D) {
        runMacro(macro1D, "FitGaus1D");
    }
    // 2D neighborhood fit
    if (Constants::FIT_GAUS_2D) {
        runMacro(macro2D, "FitGaus2D");
    }

    if (executed) {
        G4cout << "[RunAction] Post-processing fits finished" << G4endl;
    }
}

void RunAction::ResetSynchronization()
{
    if (G4Threading::IsMultithreadedApplication()) {
        support::WorkerSyncHelper::Instance().Reset(G4Threading::GetNumberOfRunningWorkerThreads());
    }
}

void RunAction::SignalWorkerCompletion()
{
    if (G4Threading::IsMultithreadedApplication()) {
        support::WorkerSyncHelper::Instance().SignalWorkerCompletion();
    }
}

void RunAction::WaitForAllWorkersToComplete()
{
    if (G4Threading::IsMultithreadedApplication()) {
        support::WorkerSyncHelper::Instance().WaitForAllWorkers();
    }
}

bool RunAction::SafeWriteRootFile()
{
    const bool isMT = G4Threading::IsMultithreadedApplication();
    const bool isWorker = G4Threading::IsWorkerThread();
    return fRootWriter->SafeWrite(isMT, isWorker);
}

bool RunAction::ValidateRootFile(const G4String& filename, bool* hasEntries)
{
    return fRootWriter->Validate(filename, hasEntries);
}

void RunAction::CleanupRootObjects()
{
    fRootWriter->Cleanup();
}

void RunAction::SetDetectorGridParameters(G4double pixelSize,
                                          G4double pixelSpacing,
                                          G4double pixelCornerOffset,
                                          G4double detSize,
                                          G4int numBlocksPerSide)
{
    fGridPixelSize = pixelSize;
    fGridPixelSpacing = pixelSpacing;
    fGridPixelCornerOffset = pixelCornerOffset;
    fGridDetSize = detSize;
    fGridNumBlocksPerSide = numBlocksPerSide;
    fFullGridSide = numBlocksPerSide;
    fChargeSharingPitch = pixelSpacing;
}

void RunAction::SetNeighborhoodRadiusMeta(G4int radius)
{
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

void RunAction::SetChargeSharingMetadata(Constants::ChargeSharingModel model,
                                         G4double betaPerMicron,
                                         G4double pitch)
{
    fChargeSharingModel = model;
    if (model == Constants::ChargeSharingModel::Linear && std::isfinite(betaPerMicron)) {
        fChargeSharingBeta = betaPerMicron;
    } else {
        fChargeSharingBeta = std::numeric_limits<G4double>::quiet_NaN();
    }

    if (pitch > 0.0) {
        fChargeSharingPitch = pitch;
    }
}

void RunAction::SetChargeSharingDistanceAlphaMeta(G4bool enabled)
{
    fEmitDistanceAlphaMeta = enabled;
}

void RunAction::SetGridPixelCenters(const std::vector<G4ThreeVector>& centers)
{
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

void RunAction::ConfigureFullFractionBranch(G4bool enable)
{
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

    constexpr Int_t bufsize = 256000; // accommodate full-grid payload comfortably
    const Int_t splitLevel = 0;       // disable splitting for large std::vector payloads
    tree->Branch("FiGrid", &fFullFi, bufsize, splitLevel);
    tree->Branch("QiGrid", &fFullQi, bufsize, splitLevel);
    tree->Branch("QnGrid", &fFullQn, bufsize, splitLevel);
    tree->Branch("QfGrid", &fFullQf, bufsize, splitLevel);
    tree->Branch("DistanceGrid", &fFullDistance, bufsize, splitLevel);
    tree->Branch("AlphaGrid", &fFullAlpha, bufsize, splitLevel);
    tree->Branch("PixelXGrid", &fFullPixelXGrid, bufsize, splitLevel);
    tree->Branch("PixelYGrid", &fFullPixelYGrid, bufsize, splitLevel);
    tree->Branch("FullGridSide", &fFullGridSide, "FullGridSide/I");
    fFullFractionsBranchInitialized = true;
}

void RunAction::WriteMetadataToFile(TFile* file) const
{
    if (!file) {
        return;
    }

    MetadataEntries entries = CollectMetadataEntries();
    if (entries.empty()) {
        return;
    }

    std::lock_guard<std::mutex> ioLock(support::RootIOMutex());
    WriteMetadataEntriesUnlocked(file, entries);
}

std::vector<std::pair<std::string, std::string>> RunAction::CollectMetadataEntries() const
{
    MetadataEntries entries;
    entries.reserve(20);

    auto addPair = [&](const std::string& key, const std::string& value) {
        entries.emplace_back(key, value);
    };
    auto addDouble = [&](const std::string& key, G4double value, const char* fmt = "%.6f") {
        addPair(key, std::string(Form(fmt, value)));
    };
    auto addInt = [&](const std::string& key, G4int value) {
        addPair(key, std::string(Form("%d", value)));
    };

    addPair("MetadataSchemaVersion", "2");

    if (fGridPixelSize > 0.0) {
        addDouble("GridPixelSize_mm", fGridPixelSize);
    }
    if (fGridPixelSpacing > 0.0) {
        addDouble("GridPixelSpacing_mm", fGridPixelSpacing);
    }
    if (fGridPixelCornerOffset >= 0.0) {
        addDouble("GridPixelCornerOffset_mm", fGridPixelCornerOffset);
    }
    if (fGridDetSize > 0.0) {
        addDouble("GridDetectorSize_mm", fGridDetSize);
    }
    if (fGridNumBlocksPerSide > 0) {
        addInt("GridNumBlocksPerSide", fGridNumBlocksPerSide);
    }
    if (fStoreFullFractions && fFullGridSide > 0) {
        addInt("FullGridSide", fFullGridSide);
    }
    if (fGridNeighborhoodRadius >= 0) {
        addInt("NeighborhoodRadius", fGridNeighborhoodRadius);
    }

    const std::string modelStr =
        (fChargeSharingModel == Constants::ChargeSharingModel::Linear) ? "Linear" : "Log";
    addPair("ChargeSharingModel", modelStr);

    if (fChargeSharingModel == Constants::ChargeSharingModel::Linear &&
        std::isfinite(fChargeSharingBeta)) {
        addDouble("ChargeSharingLinearBeta_per_um", fChargeSharingBeta);
    }

    if (fChargeSharingPitch > 0.0) {
        addDouble("ChargeSharingPitch_mm", fChargeSharingPitch);
    }

    addDouble("ChargeSharingReferenceD0_microns", Constants::D0_CHARGE_SHARING);
    addDouble("IonizationEnergy_eV", Constants::IONIZATION_ENERGY);
    addDouble("AmplificationFactor", Constants::AMPLIFICATION_FACTOR);
    addPair("ElementaryCharge_C", std::string(Form("%.9e", Constants::ELEMENTARY_CHARGE)));
    addDouble("NoisePixelGainSigmaMin", Constants::PIXEL_GAIN_SIGMA_MIN);
    addDouble("NoisePixelGainSigmaMax", Constants::PIXEL_GAIN_SIGMA_MAX);
    addDouble("NoiseElectronCount", Constants::NOISE_ELECTRON_COUNT);
    addPair("ChargeSharingEmitDistanceAlpha", BoolToString(fEmitDistanceAlphaMeta));
    addPair("ChargeSharingFullFractionsEnabled", BoolToString(fStoreFullFractions));
    addPair("PostProcessFitGaus1DEnabled", BoolToString(Constants::FIT_GAUS_1D));
    addPair("PostProcessFitGaus2DEnabled", BoolToString(Constants::FIT_GAUS_2D));

    return entries;
}

void RunAction::FillTree(const EventRecord& record)
{
    auto treeLock = MakeTreeLock();
    auto* tree = GetTree();
    if (!tree) {
        return;
    }

    if (fStoreFullFractions && !fFullFractionsBranchInitialized) {
        ConfigureFullFractionBranch(true);
    }

    UpdateSummaryScalars(record);

    const std::size_t requestedCells =
        record.totalGridCells > 0 ? static_cast<std::size_t>(record.totalGridCells)
                                  : std::max<std::size_t>(1, fNeighborhoodLayout.TotalCells());

    PrepareNeighborhoodStorage(requestedCells);
    PopulateNeighborhoodFromRecord(record);
    PopulateFullFractionsFromRecord(record);

    if (tree->Fill() < 0) {
        G4Exception("RunAction::FillTree",
                    "TreeFillFailure",
                    FatalException,
                    "TTree::Fill() reported an error.");
    }
}
