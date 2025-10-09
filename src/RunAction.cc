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

#include <atomic>
#include <filesystem>
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

    bool Validate(const G4String& filename)
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

            const bool isValid = testTree->GetEntries() > 0;
            if (!isValid) {
                G4cerr << "RunAction: Warning - Empty tree in file: " << filename << G4endl;
            }
            testFile->Close();
            delete testFile;
            return isValid;
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

void RunAction::BeginOfRunAction(const G4Run* run)
{
    std::call_once(gRootInitFlag, InitializeROOTThreading);

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
        tree->Branch("FirstContactIsPixel", &fFirstContactIsPixel, "FirstContactIsPixel/O");
        tree->Branch("GeometricIsPixel", &fGeometricIsPixel, "GeometricIsPixel/O");
        tree->Branch("isPixelHit", &fIsPixelHit, "isPixelHit/O");
        tree->Branch("F_i", &fNeighborhoodChargeFractions);
        tree->Branch("Q_i", &fNeighborhoodCharge);
        tree->Branch("Q_n", &fNeighborhoodChargeNew);
        tree->Branch("Q_f", &fNeighborhoodChargeFinal);
        tree->Branch("d_i", &fNeighborhoodDistance);
        tree->Branch("alpha_i", &fNeighborhoodAlpha);
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
        if (nofEvents > 0 && SafeWriteRootFile()) {
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
        } else if (nofEvents > 0) {
            G4cerr << "RunAction: Failed to write ROOT file" << G4endl;
        }

        CleanupRootObjects();
        if (!isMT) {
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

    if (!ValidateRootFile("epicChargeSharing.root")) {
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

bool RunAction::ValidateRootFile(const G4String& filename)
{
    return fRootWriter->Validate(filename);
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
    if (fNeighborhoodChargeFractions.capacity() < chargeFractions.size()) {
        fNeighborhoodChargeFractions.reserve(chargeFractions.size());
    }
    if (fNeighborhoodCharge.capacity() < chargeCoulombs.size()) {
        fNeighborhoodCharge.reserve(chargeCoulombs.size());
    }
    fNeighborhoodChargeFractions.assign(chargeFractions.begin(), chargeFractions.end());
    fNeighborhoodCharge.assign(chargeCoulombs.begin(), chargeCoulombs.end());
}

void RunAction::SetNeighborhoodChargeNewData(const std::vector<G4double>& chargeCoulombsNew)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodChargeNew.capacity() < chargeCoulombsNew.size()) {
        fNeighborhoodChargeNew.reserve(chargeCoulombsNew.size());
    }
    fNeighborhoodChargeNew.assign(chargeCoulombsNew.begin(), chargeCoulombsNew.end());
}

void RunAction::SetNeighborhoodChargeFinalData(const std::vector<G4double>& chargeCoulombsFinal)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodChargeFinal.capacity() < chargeCoulombsFinal.size()) {
        fNeighborhoodChargeFinal.reserve(chargeCoulombsFinal.size());
    }
    fNeighborhoodChargeFinal.assign(chargeCoulombsFinal.begin(), chargeCoulombsFinal.end());
}

void RunAction::SetNeighborhoodDistanceAlphaData(const std::vector<G4double>& distances,
                                                 const std::vector<G4double>& alphas)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodDistance.capacity() < distances.size()) {
        fNeighborhoodDistance.reserve(distances.size());
    }
    if (fNeighborhoodAlpha.capacity() < alphas.size()) {
        fNeighborhoodAlpha.reserve(alphas.size());
    }
    fNeighborhoodDistance.assign(distances.begin(), distances.end());
    fNeighborhoodAlpha.assign(alphas.begin(), alphas.end());
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

void RunAction::SetNeighborhoodPixelData(const std::vector<G4double>& xs,
                                         const std::vector<G4double>& ys,
                                         const std::vector<G4int>& ids)
{
    std::lock_guard<std::mutex> lock(fTreeMutex);
    if (fNeighborhoodPixelX.capacity() < xs.size()) {
        fNeighborhoodPixelX.reserve(xs.size());
    }
    if (fNeighborhoodPixelY.capacity() < ys.size()) {
        fNeighborhoodPixelY.reserve(ys.size());
    }
    if (fNeighborhoodPixelID.capacity() < ids.size()) {
        fNeighborhoodPixelID.reserve(ids.size());
    }
    fNeighborhoodPixelX.assign(xs.begin(), xs.end());
    fNeighborhoodPixelY.assign(ys.begin(), ys.end());
    fNeighborhoodPixelID.assign(ids.begin(), ids.end());
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
