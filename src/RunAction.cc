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
      fGridPixelSize(0.0),
      fGridPixelSpacing(0.0),
      fGridPixelCornerOffset(0.0),
      fGridDetSize(0.0),
      fGridNumBlocksPerSide(0),
      fGridNeighborhoodRadius(0),
      fChargeSharingModel(Constants::CHARGE_SHARING_MODEL),
      fChargeSharingBeta(std::numeric_limits<G4double>::quiet_NaN()),
      fChargeSharingPitch(0.0),
      fEmitDistanceAlphaMeta(false)
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

void RunAction::EnsureVectorSized(std::vector<G4double>& vec, G4double initValue) const
{
    if (static_cast<G4int>(vec.size()) != fNeighborhoodCapacity) {
        vec.assign(std::max(0, fNeighborhoodCapacity), initValue);
    } else {
        std::fill(vec.begin(), vec.end(), initValue);
    }
}

void RunAction::EnsureVectorSized(std::vector<G4int>& vec, G4int initValue) const
{
    if (static_cast<G4int>(vec.size()) != fNeighborhoodCapacity) {
        vec.assign(std::max(0, fNeighborhoodCapacity), initValue);
    } else {
        std::fill(vec.begin(), vec.end(), initValue);
    }
}

void RunAction::EnsureBranchBuffersInitialized()
{
    G4int radius = fGridNeighborhoodRadius >= 0 ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS;
    if (radius < 0) {
        radius = 0;
    }
    const G4int desiredCapacity = std::max(1, (2 * radius + 1) * (2 * radius + 1));
    if (desiredCapacity != fNeighborhoodCapacity ||
        static_cast<G4int>(fNeighborhoodChargeFractions.size()) != desiredCapacity) {
        fNeighborhoodCapacity = desiredCapacity;
        fNeighborhoodChargeFractions.clear();
        fNeighborhoodCharge.clear();
        fNeighborhoodChargeNew.clear();
        fNeighborhoodChargeFinal.clear();
        fNeighborhoodDistance.clear();
        fNeighborhoodAlpha.clear();
        fNeighborhoodPixelX.clear();
        fNeighborhoodPixelY.clear();
        fNeighborhoodPixelID.clear();
    }

    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    EnsureVectorSized(fNeighborhoodChargeFractions, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    EnsureVectorSized(fNeighborhoodCharge, 0.0);
    EnsureVectorSized(fNeighborhoodChargeNew, 0.0);
    EnsureVectorSized(fNeighborhoodChargeFinal, 0.0);
    EnsureVectorSized(fNeighborhoodDistance, nan);
    EnsureVectorSized(fNeighborhoodAlpha, nan);
    EnsureVectorSized(fNeighborhoodPixelX, nan);
    EnsureVectorSized(fNeighborhoodPixelY, nan);
    EnsureVectorSized(fNeighborhoodPixelID, -1);
    fNeighborhoodActiveCells = 0;
}

void RunAction::BeginOfRunAction(const G4Run* run)
{
    std::call_once(gRootInitFlag, support::InitializeROOTThreading);

    EnsureBranchBuffersInitialized();

    const bool isMT = G4Threading::IsMultithreadedApplication();
    const bool isWorker = G4Threading::IsWorkerThread();

    if (!run) {
        G4Exception("RunAction::BeginOfRunAction",
                    "InvalidRunObject",
                    FatalException,
                    "Received null run object in BeginOfRunAction.");
        return;
    }

    const G4int runId = run->GetRunID();
    if (isMT) {
        if (isWorker) {
            const G4int threadId = G4Threading::G4GetThreadId();
            G4cout << "[RunAction] Worker " << threadId << " beginning run " << runId << G4endl;
        } else {
            G4cout << "[RunAction] Master beginning run " << runId << " with "
                   << G4Threading::GetNumberOfRunningWorkerThreads() << " workers" << G4endl;
        }
    } else {
        G4cout << "[RunAction] Beginning run " << runId << " (single-threaded)" << G4endl;
    }

    if (isMT && !isWorker) {
        support::WorkerSyncHelper::Instance().Reset(G4Threading::GetNumberOfRunningWorkerThreads());
    }

    G4String fileName;
    if (isMT) {
        if (isWorker) {
            const G4int threadId = G4Threading::G4GetThreadId();
            fileName = support::WorkerFileName(threadId);
        } else {
            fileName = "epicChargeSharing.root";
        }
    } else {
        fileName = "epicChargeSharing.root";
    }

    if (!isMT || isWorker) {
        std::unique_lock<std::mutex> rootInitLock(support::RootIOMutex());
        auto* rootFile = new TFile(fileName.c_str(), "RECREATE");
        if (!rootFile || rootFile->IsZombie()) {
            G4Exception("RunAction::BeginOfRunAction",
                        "RootFileOpenFailure",
                        FatalException,
                        ("Unable to open ROOT file " + fileName).c_str());
        }
        rootFile->SetCompressionLevel(0);

        auto* tree = new TTree("Hits", "AC-LGAD charge sharing hits");

        tree->Branch("TrueX", &fTrueX, "TrueX/D");
        tree->Branch("TrueY", &fTrueY, "TrueY/D");
        tree->Branch("PixelX", &fPixelX, "PixelX/D");
        tree->Branch("PixelY", &fPixelY, "PixelY/D");
        tree->Branch("Edep", &fEdep, "Edep/D");
        tree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "PixelTrueDeltaX/D");
        tree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "PixelTrueDeltaY/D");
        tree->Branch("isPixelHit", &fIsPixelHit, "isPixelHit/O");
        tree->Branch("NeighborhoodSize", &fNeighborhoodActiveCells, "NeighborhoodSize/I");

        tree->Branch("F_i", &fNeighborhoodChargeFractions);
        tree->Branch("Q_i", &fNeighborhoodCharge);
        tree->Branch("Q_n", &fNeighborhoodChargeNew);
        tree->Branch("Q_f", &fNeighborhoodChargeFinal);
        // tree->Branch("d_i", &fNeighborhoodDistance);
        // tree->Branch("alpha_i", &fNeighborhoodAlpha);
        tree->Branch("NeighborhoodPixelX", &fNeighborhoodPixelX);
        tree->Branch("NeighborhoodPixelY", &fNeighborhoodPixelY);
        tree->Branch("NeighborhoodPixelID", &fNeighborhoodPixelID);

        fRootWriter->Attach(rootFile, tree, true);
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

    const bool isMT = G4Threading::IsMultithreadedApplication();
    const bool isWorker = G4Threading::IsWorkerThread();

    const G4int nofEvents = run->GetNumberOfEvent();

    if (!isMT || isWorker) {
        bool wroteOutput = false;
        if (nofEvents > 0) {
            if (SafeWriteRootFile()) {
                wroteOutput = true;
                if (isMT) {
                    const G4int threadId = G4Threading::G4GetThreadId();
                    G4cout << "[RunAction] Worker " << threadId << " stored " << nofEvents
                           << " events" << G4endl;
                } else {
                    G4cout << "[RunAction] Stored " << nofEvents << " events" << G4endl;
                }
                if (!isMT) {
                    WriteMetadataToFile(fRootWriter ? fRootWriter->File() : nullptr);
                }
            } else {
                G4Exception("RunAction::EndOfRunAction",
                            "RootFileWriteFailure",
                            FatalException,
                            "SafeWriteRootFile() reported failure.");
            }
        }

        CleanupRootObjects();
        if (!isMT && wroteOutput) {
            // Single-threaded: run post-processing fits once the file is closed
            RunPostProcessingFits();
        }
        if (isMT) {
            SignalWorkerCompletion();
        }
        return;
    }

    // Master thread in MT mode
    WaitForAllWorkersToComplete();

    const G4int totalWorkers = G4Threading::GetNumberOfRunningWorkerThreads();
    std::vector<G4String> workerFiles;
    workerFiles.reserve(totalWorkers);
    for (G4int tid = 0; tid < totalWorkers; ++tid) {
        workerFiles.push_back(support::WorkerFileName(tid));
    }

    std::vector<G4String> existingFiles;
    existingFiles.reserve(workerFiles.size());
    for (const auto& wf : workerFiles) {
        if (std::filesystem::exists(wf.c_str()) && std::filesystem::file_size(wf.c_str()) > 0) {
            existingFiles.push_back(wf);
        }
    }

    if (existingFiles.empty()) {
        G4cout << "[RunAction] No worker ROOT files found to merge" << G4endl;
        return;
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
            G4Exception("RunAction::mergeFiles",
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
                G4Exception("RunAction::EndOfRunAction",
                            "MergedFileOpenFailure",
                            FatalException,
                            "Failed to open merged ROOT file for metadata update.");
            }
        }
    }

    if (!mergeOk) {
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

    // Master thread: run post-processing fits on merged output
    RunPostProcessingFits();
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
    fChargeSharingPitch = pixelSpacing;
}

void RunAction::SetNeighborhoodRadiusMeta(G4int radius)
{
    fGridNeighborhoodRadius = radius;

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

    const std::size_t capacity = fNeighborhoodCapacity > 0 ? static_cast<std::size_t>(fNeighborhoodCapacity) : 0;
    if (capacity > 0U) {
        const std::size_t fractionCount = std::min({capacity,
                                                    record.neighborFractions.size(),
                                                    record.neighborCharges.size()});

        EnsureVectorSized(fNeighborhoodChargeFractions, Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
        EnsureVectorSized(fNeighborhoodCharge, 0.0);
        EnsureVectorSized(fNeighborhoodChargeNew, 0.0);
        EnsureVectorSized(fNeighborhoodChargeFinal, 0.0);
        const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
        EnsureVectorSized(fNeighborhoodDistance, nan);
        EnsureVectorSized(fNeighborhoodAlpha, nan);
        EnsureVectorSized(fNeighborhoodPixelX, nan);
        EnsureVectorSized(fNeighborhoodPixelY, nan);
        EnsureVectorSized(fNeighborhoodPixelID, -1);

        if (fractionCount > 0) {
            std::copy_n(record.neighborFractions.begin(), fractionCount, fNeighborhoodChargeFractions.begin());
            std::copy_n(record.neighborCharges.begin(), fractionCount, fNeighborhoodCharge.begin());
        }

        const std::size_t newCount = std::min(capacity, record.neighborChargesNew.size());
        if (newCount > 0) {
            std::copy_n(record.neighborChargesNew.begin(), newCount, fNeighborhoodChargeNew.begin());
        }

        const std::size_t finalCount = std::min(capacity, record.neighborChargesFinal.size());
        if (finalCount > 0) {
            std::copy_n(record.neighborChargesFinal.begin(), finalCount, fNeighborhoodChargeFinal.begin());
        }

        if (record.includeDistanceAlpha) {
            const std::size_t distCount = std::min(capacity, record.neighborDistances.size());
            const std::size_t alphaCount = std::min(capacity, record.neighborAlphas.size());
            if (distCount > 0) {
                std::copy_n(record.neighborDistances.begin(), distCount, fNeighborhoodDistance.begin());
            }
            if (alphaCount > 0) {
                std::copy_n(record.neighborAlphas.begin(), alphaCount, fNeighborhoodAlpha.begin());
            }
        }

        const std::size_t pixelCount = std::min({capacity,
                                                 record.neighborPixelX.size(),
                                                 record.neighborPixelY.size(),
                                                 record.neighborPixelIds.size()});
        if (pixelCount > 0) {
            std::copy_n(record.neighborPixelX.begin(), pixelCount, fNeighborhoodPixelX.begin());
            std::copy_n(record.neighborPixelY.begin(), pixelCount, fNeighborhoodPixelY.begin());
            std::copy_n(record.neighborPixelIds.begin(), pixelCount, fNeighborhoodPixelID.begin());
        }

        fNeighborhoodActiveCells = static_cast<G4int>(fractionCount);
    } else {
        fNeighborhoodActiveCells = 0;
    }

    if (tree->Fill() < 0) {
        G4Exception("RunAction::FillTree",
                    "TreeFillFailure",
                    FatalException,
                    "TTree::Fill() reported an error.");
    }
}
