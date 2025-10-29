/**
 * @file RunAction.cc
 * @brief Run lifecycle management with dedicated helpers for ROOT I/O and worker synchronisation.
 */
#include "RunAction.hh"

#include "Constants.hh"
#include "DetectorConstruction.hh"

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
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace
{
class WorkerSyncHelper
{
public:
    static WorkerSyncHelper& Instance()
    {
        static WorkerSyncHelper instance;
        return instance;
    }

    void Reset(G4int totalWorkers)
    {
        std::lock_guard<std::mutex> lock(fMutex);
        fWorkersCompleted.store(0);
        fTotalWorkers = std::max(0, totalWorkers);
        fAllWorkersCompleted = (fTotalWorkers == 0);
    }

    void SignalWorkerCompletion()
    {
        std::unique_lock<std::mutex> lock(fMutex);
        const int completed = ++fWorkersCompleted;
        if (completed >= fTotalWorkers && !fAllWorkersCompleted) {
            fAllWorkersCompleted = true;
            lock.unlock();
            fCv.notify_all();
        }
    }

    void WaitForAllWorkers()
    {
        std::unique_lock<std::mutex> lock(fMutex);
        if (fTotalWorkers == 0) {
            return;
        }
        fCv.wait(lock, [this]() { return fAllWorkersCompleted; });
    }

private:
    WorkerSyncHelper() = default;

    std::mutex fMutex;
    std::condition_variable fCv;
    std::atomic<int> fWorkersCompleted{0};
    int fTotalWorkers{0};
    bool fAllWorkersCompleted{false};
};

static std::once_flag gRootInitFlag;
static std::mutex gRootIOMutex;

void InitializeROOTThreading()
{
    // Keep ROOT in batch mode; avoid enabling ROOT implicit MT or TThread here
    // to reduce chances of interpreter/merger thread-safety issues.
    gROOT->SetBatch(true);
}

G4String WorkerFileName(G4int threadId)
{
    std::ostringstream oss;
    oss << "epicChargeSharing_t" << threadId << ".root";
    return oss.str();
}

} // namespace

// Move RootFileWriterHelper to global scope so it matches the forward
// declaration in include/RunAction.hh and is a complete type in this TU.
class RootFileWriterHelper
{
public:
    RootFileWriterHelper() = default;
    ~RootFileWriterHelper() { Cleanup(); }

    void Attach(TFile* file, TTree* tree, bool ownsObjects)
    {
        std::lock_guard<std::mutex> lock(fMutex);
        fRootFile = file;
        fTree = tree;
        fOwnsObjects = ownsObjects;
    }

    TFile* File() const { return fRootFile; }
    TTree* Tree() const { return fTree; }

    bool SafeWrite(bool /*isMultithreaded*/, bool isWorker)
    {
        std::lock_guard<std::mutex> globalLock(gRootIOMutex);
        std::unique_lock<std::mutex> lock(fMutex, std::defer_lock);
        if (!isWorker) {
            lock.lock();
        }

        if (!fRootFile || !fTree || fRootFile->IsZombie()) {
            G4cerr << "RunAction: Cannot write - invalid ROOT file or tree" << G4endl;
            return false;
        }

        try {
            fTree->FlushBaskets();
            fRootFile->cd();
            fTree->Write("", TObject::kOverwrite);
            fRootFile->Flush();
            return true;
        } catch (const std::exception& e) {
            G4cerr << "RunAction: Exception writing ROOT file: " << e.what() << G4endl;
            return false;
        }
    }

    bool Validate(const G4String& filename, bool* hasEntries)
    {
        std::lock_guard<std::mutex> globalLock(gRootIOMutex);
        if (filename.empty()) {
            G4cerr << "RunAction: Error - Empty filename provided for validation" << G4endl;
            return false;
        }

        TFile* testFile = nullptr;
        try {
            testFile = TFile::Open(filename.c_str(), "READ");
            if (!testFile || testFile->IsZombie()) {
                G4cerr << "RunAction: Error - Cannot open or corrupted file: " << filename << G4endl;
                delete testFile;
                return false;
            }

            auto* testTree = dynamic_cast<TTree*>(testFile->Get("Hits"));
            if (!testTree) {
                G4cerr << "RunAction: Error - No 'Hits' tree found in file: " << filename << G4endl;
                testFile->Close();
                delete testFile;
                return false;
            }

            const Long64_t entryCount = testTree->GetEntries();
            if (hasEntries) {
                *hasEntries = (entryCount > 0);
            }
            if (entryCount <= 0) {
                G4cerr << "RunAction: Warning - Empty tree in file: " << filename << G4endl;
            }
            testFile->Close();
            delete testFile;
            return true;
        } catch (const std::exception& e) {
            G4cerr << "RunAction: Exception during file validation: " << e.what() << G4endl;
            if (testFile) {
                testFile->Close();
                delete testFile;
            }
            return false;
        }
    }

    void Cleanup()
    {
        std::lock_guard<std::mutex> globalLock(gRootIOMutex);
        std::lock_guard<std::mutex> lock(fMutex);
        if (fRootFile) {
            if (fRootFile->IsOpen()) {
                fRootFile->Close();
            }
            if (fOwnsObjects) {
                delete fRootFile;
            }
        }
        fRootFile = nullptr;
        fTree = nullptr;
        fOwnsObjects = false;
    }

    void WriteMetadataSingleThread(G4double pixelSize,
                                   G4double pixelSpacing,
                                   G4double pixelCornerOffset,
                                   G4double detSize,
                                   G4int numBlocksPerSide,
                                   G4int neighborhoodRadius)
    {
        std::lock_guard<std::mutex> globalLock(gRootIOMutex);
        std::lock_guard<std::mutex> lock(fMutex);
        if (!fRootFile || fRootFile->IsZombie()) {
            return;
        }
        fRootFile->cd();
        TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", pixelSize));
        TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", pixelSpacing));
        TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm", Form("%.6f", pixelCornerOffset));
        TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", detSize));
        TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", numBlocksPerSide));
        TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", neighborhoodRadius));

        pixelSizeMeta.Write("", TObject::kOverwrite);
        pixelSpacingMeta.Write("", TObject::kOverwrite);
        pixelCornerOffsetMeta.Write("", TObject::kOverwrite);
        detSizeMeta.Write("", TObject::kOverwrite);
        numBlocksMeta.Write("", TObject::kOverwrite);
        neighborhoodRadiusMeta.Write("", TObject::kOverwrite);
    }

private:
    mutable std::mutex fMutex;
    TFile* fRootFile{nullptr};
    TTree* fTree{nullptr};
    bool fOwnsObjects{false};
};

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
      fGridNumBlocksPerSide(0)
{
}

RunAction::~RunAction()
{
    CleanupRootObjects();
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
    std::call_once(gRootInitFlag, InitializeROOTThreading);

    EnsureBranchBuffersInitialized();

    const bool isMT = G4Threading::IsMultithreadedApplication();
    const bool isWorker = G4Threading::IsWorkerThread();

    if (!run) {
        G4cerr << "RunAction: Error - Invalid run object in BeginOfRunAction" << G4endl;
        return;
    }

    if (isMT && !isWorker) {
        WorkerSyncHelper::Instance().Reset(G4Threading::GetNumberOfRunningWorkerThreads());
    }

    G4String fileName;
    if (isMT) {
        if (isWorker) {
            const G4int threadId = G4Threading::G4GetThreadId();
            fileName = WorkerFileName(threadId);
        } else {
            fileName = "epicChargeSharing.root";
        }
    } else {
        fileName = "epicChargeSharing.root";
    }

    if (!isMT || isWorker) {
        std::unique_lock<std::mutex> rootInitLock(gRootIOMutex);
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
        G4cerr << "RunAction: Error - Invalid run object in EndOfRunAction" << G4endl;
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
                G4cout << "RunAction: Successfully wrote ROOT file with " << nofEvents << " events"
                       << G4endl;
                if (!isMT) {
                    fRootWriter->WriteMetadataSingleThread(fGridPixelSize,
                                                           fGridPixelSpacing,
                                                           fGridPixelCornerOffset,
                                                           fGridDetSize,
                                                           fGridNumBlocksPerSide,
                                                           fGridNeighborhoodRadius > 0
                                                               ? fGridNeighborhoodRadius
                                                               : Constants::NEIGHBORHOOD_RADIUS);
                }
            } else {
                G4cerr << "RunAction: Failed to write ROOT file" << G4endl;
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
        workerFiles.push_back(WorkerFileName(tid));
    }

    std::vector<G4String> existingFiles;
    existingFiles.reserve(workerFiles.size());
    for (const auto& wf : workerFiles) {
        if (std::filesystem::exists(wf.c_str()) && std::filesystem::file_size(wf.c_str()) > 0) {
            existingFiles.push_back(wf);
        }
    }

    if (existingFiles.empty()) {
        G4cout << "RunAction: No worker ROOT files found after MT run; skipping merge" << G4endl;
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
            G4cerr << "RunAction: Unable to set output file " << output << " for merger" << G4endl;
            return false;
        }

        bool added = false;
        for (const auto& file : inputs) {
            if (std::filesystem::exists(file.c_str())) {
                if (merger.AddFile(file.c_str())) {
                    added = true;
                } else {
                    G4cerr << "RunAction: Failed to queue " << file << " for merge" << G4endl;
                }
            }
        }

        if (!added) {
            G4cerr << "RunAction: No readable worker files were added to the merger" << G4endl;
            return false;
        }

        return merger.Merge();
    };

    bool mergeOk = false;
    {
        std::lock_guard<std::mutex> ioLock(gRootIOMutex);
        mergeOk = mergeFiles(existingFiles, "epicChargeSharing.root");
        if (mergeOk && fGridPixelSize > 0) {
            std::unique_ptr<TFile> mergedFile(TFile::Open("epicChargeSharing.root", "UPDATE"));
            if (mergedFile && !mergedFile->IsZombie()) {
                mergedFile->cd();
                TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", fGridPixelSize));
                TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing));
                TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm",
                                             Form("%.6f", fGridPixelCornerOffset));
                TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", fGridDetSize));
                TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
                const G4int radiusValue =
                    fGridNeighborhoodRadius > 0 ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS;
                TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", radiusValue));

                pixelSizeMeta.Write("", TObject::kOverwrite);
                pixelSpacingMeta.Write("", TObject::kOverwrite);
                pixelCornerOffsetMeta.Write("", TObject::kOverwrite);
                detSizeMeta.Write("", TObject::kOverwrite);
                numBlocksMeta.Write("", TObject::kOverwrite);
                neighborhoodRadiusMeta.Write("", TObject::kOverwrite);

                mergedFile->Flush();
                mergedFile->Close();
            } else {
                G4cerr << "RunAction: Failed to open merged ROOT file for metadata update" << G4endl;
            }
        }
    }

    if (!mergeOk) {
        G4cerr << "RunAction: ROOT file merge failed" << G4endl;
        return;
    }

    bool mergedHasEntries = false;
    if (!ValidateRootFile("epicChargeSharing.root", &mergedHasEntries)) {
        G4cerr << "RunAction: Merged ROOT file validation failed" << G4endl;
        return;
    }

    G4cout << "RunAction: Merged ROOT file written successfully" << G4endl;

    for (const auto& file : existingFiles) {
        std::error_code ec;
        if (std::filesystem::remove(file.c_str(), ec)) {
            G4cout << "RunAction: Removed worker file " << file << G4endl;
        } else if (ec) {
            G4cerr << "RunAction: Could not remove worker file " << file << " (" << ec.message()
                   << ")" << G4endl;
        }
    }

    if (!mergedHasEntries) {
        G4cout << "RunAction: Merged ROOT file has no entries; skipping post-processing fits"
               << G4endl;
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

    auto runMacro = [&](const TString& macroPath, const char* entryPoint) {
        // Load the macro (interpreted). Using interpreted mode avoids platform-specific ACLiC complications.
        gROOT->ProcessLine(TString::Format(".L %s", macroPath.Data()));
        // Call the entry point with just the filename, relying on macro defaults for other parameters.
        const TString call = TString::Format("%s(\"%s\")", entryPoint, rootFile.Data());
        const Long_t status = gROOT->ProcessLine(call);
        if (status != 0) {
            G4cout << "Post-processing: call '" << entryPoint << "' returned status " << status << G4endl;
        }
    };

    // 1D row/column fits
    if (Constants::FIT_GAUS_1D) {
        runMacro(macro1D, "FitGaus1D");
    }
    // 2D neighborhood fit
    if (Constants::FIT_GAUS_2D) {
        runMacro(macro2D, "FitGaus2D");
    }
}

void RunAction::ResetSynchronization()
{
    if (G4Threading::IsMultithreadedApplication()) {
        WorkerSyncHelper::Instance().Reset(G4Threading::GetNumberOfRunningWorkerThreads());
    }
}

void RunAction::SignalWorkerCompletion()
{
    if (G4Threading::IsMultithreadedApplication()) {
        WorkerSyncHelper::Instance().SignalWorkerCompletion();
    }
}

void RunAction::WaitForAllWorkersToComplete()
{
    if (G4Threading::IsMultithreadedApplication()) {
        WorkerSyncHelper::Instance().WaitForAllWorkers();
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

void RunAction::SetEventData(G4double edep, G4double x, G4double y, G4double z)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fEdep = edep;
    fTrueX = x;
    fTrueY = y;
    (void)z;
}

void RunAction::UpdateEventSummary(const EventSummaryData& summary)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fEdep = summary.edep;
    fTrueX = summary.hitX;
    fTrueY = summary.hitY;
    (void)summary.hitZ;
    fPixelX = summary.nearestPixelX;
    fPixelY = summary.nearestPixelY;
    fPixelTrueDeltaX = summary.pixelTrueDeltaX;
    fPixelTrueDeltaY = summary.pixelTrueDeltaY;
    fFirstContactIsPixel = summary.firstContactIsPixel;
    fGeometricIsPixel = summary.geometricIsPixel;
    fIsPixelHit = summary.isPixelHitCombined;
}

void RunAction::SetNearestPixelPos(G4double x, G4double y)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fPixelX = x;
    fPixelY = y;
}

void RunAction::SetFirstContactIsPixel(G4bool v)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fFirstContactIsPixel = v;
}

void RunAction::SetGeometricIsPixel(G4bool v)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fGeometricIsPixel = v;
}

void RunAction::SetIsPixelHitCombined(G4bool v)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fIsPixelHit = v;
}

void RunAction::SetPixelClassification(G4bool isPixelHit,
                                       G4double pixelTrueDeltaX,
                                       G4double pixelTrueDeltaY)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    fIsPixelHit = isPixelHit;
    fPixelTrueDeltaX = pixelTrueDeltaX;
    fPixelTrueDeltaY = pixelTrueDeltaY;
}

void RunAction::SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                                          const std::vector<G4double>& chargeCoulombs)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodCapacity <= 0) {
        return;
    }

    const std::size_t capacity = static_cast<std::size_t>(fNeighborhoodCapacity);
    if (chargeFractions.size() > capacity || chargeCoulombs.size() > capacity) {
        static std::once_flag warnFlag;
        std::call_once(warnFlag, []() {
            G4cerr << "RunAction: Neighborhood data exceeds fixed branch capacity; truncating to configured radius." << G4endl;
        });
    }

    const std::size_t copyCount = std::min({chargeFractions.size(), chargeCoulombs.size(), capacity});
    std::fill(fNeighborhoodChargeFractions.begin(), fNeighborhoodChargeFractions.end(),
              Constants::OUT_OF_BOUNDS_FRACTION_SENTINEL);
    std::fill(fNeighborhoodCharge.begin(), fNeighborhoodCharge.end(), 0.0);

    if (copyCount > 0) {
        std::copy_n(chargeFractions.begin(), copyCount, fNeighborhoodChargeFractions.begin());
        std::copy_n(chargeCoulombs.begin(), copyCount, fNeighborhoodCharge.begin());
    }

    fNeighborhoodActiveCells = static_cast<G4int>(copyCount);
}

void RunAction::SetNeighborhoodChargeNewData(const std::vector<G4double>& chargeCoulombsNew)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodCapacity <= 0) {
        return;
    }

    const std::size_t capacity = static_cast<std::size_t>(fNeighborhoodCapacity);
    const std::size_t copyCount = std::min<std::size_t>(chargeCoulombsNew.size(), capacity);

    std::fill(fNeighborhoodChargeNew.begin(), fNeighborhoodChargeNew.end(), 0.0);
    if (copyCount > 0) {
        std::copy_n(chargeCoulombsNew.begin(), copyCount, fNeighborhoodChargeNew.begin());
    }
}

void RunAction::SetNeighborhoodChargeFinalData(const std::vector<G4double>& chargeCoulombsFinal)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodCapacity <= 0) {
        return;
    }

    const std::size_t capacity = static_cast<std::size_t>(fNeighborhoodCapacity);
    const std::size_t copyCount = std::min<std::size_t>(chargeCoulombsFinal.size(), capacity);

    std::fill(fNeighborhoodChargeFinal.begin(), fNeighborhoodChargeFinal.end(), 0.0);
    if (copyCount > 0) {
        std::copy_n(chargeCoulombsFinal.begin(), copyCount, fNeighborhoodChargeFinal.begin());
    }
}

void RunAction::SetNeighborhoodDistanceAlphaData(const std::vector<G4double>& distances,
                                                 const std::vector<G4double>& alphas)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodCapacity <= 0) {
        return;
    }

    const std::size_t capacity = static_cast<std::size_t>(fNeighborhoodCapacity);
    const std::size_t distCopy = std::min<std::size_t>(distances.size(), capacity);
    const std::size_t alphaCopy = std::min<std::size_t>(alphas.size(), capacity);

    if (distCopy == 0 && alphaCopy == 0) {
        return;
    }

    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();
    std::fill(fNeighborhoodDistance.begin(), fNeighborhoodDistance.end(), nan);
    std::fill(fNeighborhoodAlpha.begin(), fNeighborhoodAlpha.end(), nan);
    if (distCopy > 0) {
        std::copy_n(distances.begin(), distCopy, fNeighborhoodDistance.begin());
    }
    if (alphaCopy > 0) {
        std::copy_n(alphas.begin(), alphaCopy, fNeighborhoodAlpha.begin());
    }
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
}

void RunAction::SetNeighborhoodRadiusMeta(G4int radius)
{
    fGridNeighborhoodRadius = radius;

    if (GetTree()) {
        static std::once_flag warnFlag;
        std::call_once(warnFlag, []() {
            G4cerr << "RunAction: Neighborhood radius changed after tree creation; fixed-size branches retain original capacity." << G4endl;
        });
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

void RunAction::SetNeighborhoodPixelData(const std::vector<G4double>& xs,
                                         const std::vector<G4double>& ys,
                                         const std::vector<G4int>& ids)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodCapacity <= 0) {
        return;
    }

    const std::size_t capacity = static_cast<std::size_t>(fNeighborhoodCapacity);
    const std::size_t copyCount = std::min({xs.size(), ys.size(), static_cast<std::size_t>(ids.size()), capacity});
    const G4double nan = std::numeric_limits<G4double>::quiet_NaN();

    std::fill(fNeighborhoodPixelX.begin(), fNeighborhoodPixelX.end(), nan);
    std::fill(fNeighborhoodPixelY.begin(), fNeighborhoodPixelY.end(), nan);
    std::fill(fNeighborhoodPixelID.begin(), fNeighborhoodPixelID.end(), -1);

    if (copyCount > 0) {
        std::copy_n(xs.begin(), copyCount, fNeighborhoodPixelX.begin());
        std::copy_n(ys.begin(), copyCount, fNeighborhoodPixelY.begin());
        std::copy_n(ids.begin(), copyCount, fNeighborhoodPixelID.begin());
    }
}

void RunAction::FillTree()
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    auto* tree = GetTree();
    if (!tree) {
        return;
    }
    if (tree->Fill() < 0) {
        G4cerr << "RunAction: Tree Fill() returned error" << G4endl;
    }
}
