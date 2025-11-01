#include "internal/RunActionSupport.hh"

#include "G4Exception.hh"

#include "TError.h"
#include "TFile.h"
#include "TNamed.h"
#include "TObject.h"
#include "TROOT.h"
#include "TString.h"
#include "TTree.h"

#include <algorithm>
#include <sstream>

namespace runaction
{

namespace
{
std::mutex& RootIOMutexInstance()
{
    static std::mutex mutex;
    return mutex;
}
} // namespace

RootFileWriterHelper::RootFileWriterHelper() = default;

RootFileWriterHelper::~RootFileWriterHelper()
{
    Cleanup();
}

void RootFileWriterHelper::Attach(TFile* file, TTree* tree, bool ownsObjects)
{
    std::lock_guard<std::mutex> lock(fMutex);
    fRootFile = file;
    fTree = tree;
    fOwnsObjects = ownsObjects;
}

TFile* RootFileWriterHelper::File() const
{
    return fRootFile;
}

TTree* RootFileWriterHelper::Tree() const
{
    return fTree;
}

bool RootFileWriterHelper::SafeWrite(bool /*isMultithreaded*/, bool isWorker)
{
    std::lock_guard<std::mutex> globalLock(support::RootIOMutex());
    std::unique_lock<std::mutex> lock(fMutex, std::defer_lock);
    if (!isWorker) {
        lock.lock();
    }

    if (!fRootFile || !fTree || fRootFile->IsZombie()) {
        G4Exception("RootFileWriterHelper::SafeWrite",
                    "InvalidRootFile",
                    FatalException,
                    "Cannot write because the ROOT file or tree handle is invalid.");
        return false;
    }

    try {
        fTree->FlushBaskets();
        fRootFile->cd();
        fTree->Write("", TObject::kOverwrite);
        fRootFile->Flush();
        return true;
    } catch (const std::exception& e) {
        G4Exception("RootFileWriterHelper::SafeWrite",
                    "RootWriteException",
                    FatalException,
                    e.what());
        return false;
    }
}

bool RootFileWriterHelper::Validate(const G4String& filename, bool* hasEntries)
{
    std::lock_guard<std::mutex> globalLock(support::RootIOMutex());
    if (filename.empty()) {
        G4Exception("RootFileWriterHelper::Validate",
                    "EmptyFilename",
                    FatalException,
                    "Empty filename provided for validation.");
        return false;
    }

    TFile* testFile = nullptr;
    try {
        testFile = TFile::Open(filename.c_str(), "READ");
        if (!testFile || testFile->IsZombie()) {
            G4Exception("RootFileWriterHelper::Validate",
                        "FileOpenFailure",
                        FatalException,
                        ("Cannot open or corrupted file: " + filename).c_str());
            delete testFile;
            return false;
        }

        auto* testTree = dynamic_cast<TTree*>(testFile->Get("Hits"));
        if (!testTree) {
            G4Exception("RootFileWriterHelper::Validate",
                        "MissingHitsTree",
                        FatalException,
                        ("No 'Hits' tree found in file: " + filename).c_str());
            testFile->Close();
            delete testFile;
            return false;
        }

        const Long64_t entryCount = testTree->GetEntries();
        if (hasEntries) {
            *hasEntries = (entryCount > 0);
        }
        (void)entryCount;
        testFile->Close();
        delete testFile;
        return true;
    } catch (const std::exception& e) {
        G4Exception("RootFileWriterHelper::Validate",
                    "ValidationException",
                    FatalException,
                    e.what());
        if (testFile) {
            testFile->Close();
            delete testFile;
        }
        return false;
    }
}

void RootFileWriterHelper::Cleanup()
{
    std::lock_guard<std::mutex> globalLock(support::RootIOMutex());
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

void RootFileWriterHelper::WriteMetadataSingleThread(G4double pixelSize,
                                                     G4double pixelSpacing,
                                                     G4double pixelCornerOffset,
                                                     G4double detSize,
                                                     G4int numBlocksPerSide,
                                                     G4int neighborhoodRadius)
{
    std::lock_guard<std::mutex> globalLock(support::RootIOMutex());
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

namespace support
{

WorkerSyncHelper& WorkerSyncHelper::Instance()
{
    static WorkerSyncHelper instance;
    return instance;
}

void WorkerSyncHelper::Reset(G4int totalWorkers)
{
    std::lock_guard<std::mutex> lock(fMutex);
    fWorkersCompleted.store(0);
    fTotalWorkers = std::max(0, totalWorkers);
    fAllWorkersCompleted = (fTotalWorkers == 0);
}

void WorkerSyncHelper::SignalWorkerCompletion()
{
    std::unique_lock<std::mutex> lock(fMutex);
    const int completed = ++fWorkersCompleted;
    if (completed >= fTotalWorkers && !fAllWorkersCompleted) {
        fAllWorkersCompleted = true;
        lock.unlock();
        fCv.notify_all();
    }
}

void WorkerSyncHelper::WaitForAllWorkers()
{
    std::unique_lock<std::mutex> lock(fMutex);
    if (fTotalWorkers == 0) {
        return;
    }
    fCv.wait(lock, [this]() { return fAllWorkersCompleted; });
}

void InitializeROOTThreading()
{
    gROOT->SetBatch(true);
}

G4String WorkerFileName(G4int threadId)
{
    std::ostringstream oss;
    oss << "epicChargeSharing_t" << threadId << ".root";
    return oss.str();
}

std::mutex& RootIOMutex()
{
    return RootIOMutexInstance();
}

} // namespace support

} // namespace runaction

