/// \file RootHelpers.cc
/// \brief Implementation of ROOT helper classes.

#include "RootHelpers.hh"

#include "G4Exception.hh"

#include "TError.h"
#include "TFile.h"
#include "TNamed.h"
#include "TObject.h"
#include "TROOT.h"
#include "TString.h"
#include "TTree.h"

#include <sstream>

namespace ECS {

namespace {
std::mutex& RootIOMutexInstance() {
    static std::mutex mutex;
    return mutex;
}
} // namespace

// ============================================================================
// RootFileWriter
// ============================================================================

RootFileWriter::RootFileWriter() = default;

RootFileWriter::~RootFileWriter() {
    Cleanup();
}

void RootFileWriter::Attach(TFile* file, TTree* tree, bool ownsObjects) {
    std::lock_guard<std::mutex> lock(fMutex);
    fRootFile = file;
    fTree = tree;
    fOwnsObjects = ownsObjects;
}

TFile* RootFileWriter::File() const {
    return fRootFile;
}

TTree* RootFileWriter::Tree() const {
    return fTree;
}

bool RootFileWriter::SafeWrite(bool /*isMultithreaded*/, bool isWorker) {
    std::lock_guard<std::mutex> globalLock(RootIOMutex());
    std::unique_lock<std::mutex> lock(fMutex, std::defer_lock);
    if (!isWorker) {
        lock.lock();
    }

    if (!fRootFile || !fTree || fRootFile->IsZombie()) {
        G4Exception("RootFileWriter::SafeWrite", "InvalidRootFile", FatalException,
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
        G4Exception("RootFileWriter::SafeWrite", "RootWriteException", FatalException, e.what());
        return false;
    }
}

bool RootFileWriter::Validate(const G4String& filename, bool* hasEntries) {
    std::lock_guard<std::mutex> globalLock(RootIOMutex());
    if (filename.empty()) {
        G4Exception("RootFileWriter::Validate", "EmptyFilename", FatalException,
                    "Empty filename provided for validation.");
        return false;
    }

    TFile* testFile = nullptr;
    try {
        testFile = TFile::Open(filename.c_str(), "READ");
        if (!testFile || testFile->IsZombie()) {
            G4Exception("RootFileWriter::Validate", "FileOpenFailure", FatalException,
                        ("Cannot open or corrupted file: " + filename).c_str());
            delete testFile;
            return false;
        }

        auto* testTree = dynamic_cast<TTree*>(testFile->Get("Hits"));
        if (!testTree) {
            G4Exception("RootFileWriter::Validate", "MissingHitsTree", FatalException,
                        ("No 'Hits' tree found in file: " + filename).c_str());
            testFile->Close();
            delete testFile;
            return false;
        }

        if (hasEntries) {
            *hasEntries = (testTree->GetEntries() > 0);
        }
        testFile->Close();
        delete testFile;
        return true;
    } catch (const std::exception& e) {
        G4Exception("RootFileWriter::Validate", "ValidationException", FatalException, e.what());
        if (testFile) {
            testFile->Close();
            delete testFile;
        }
        return false;
    }
}

void RootFileWriter::Cleanup() {
    std::lock_guard<std::mutex> globalLock(RootIOMutex());
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

void RootFileWriter::WriteMetadataSingleThread(G4double pixelSize, G4double pixelSpacing,
                                                G4double pixelCornerOffset, G4double detSize,
                                                G4int numBlocksPerSide, G4int neighborhoodRadius) {
    std::lock_guard<std::mutex> globalLock(RootIOMutex());
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

// ============================================================================
// WorkerSync
// ============================================================================

WorkerSync& WorkerSync::Instance() {
    static WorkerSync instance;
    return instance;
}

void WorkerSync::Reset(G4int totalWorkers) {
    std::lock_guard<std::mutex> lock(fMutex);
    fCompleted.store(0);
    fTotal = std::max(0, totalWorkers);
    fAllDone = (fTotal == 0);
}

void WorkerSync::SignalCompletion() {
    std::unique_lock<std::mutex> lock(fMutex);
    const int completed = ++fCompleted;
    if (completed >= fTotal && !fAllDone) {
        fAllDone = true;
        lock.unlock();
        fCv.notify_all();
    }
}

void WorkerSync::WaitForAll() {
    std::unique_lock<std::mutex> lock(fMutex);
    if (fTotal == 0) {
        return;
    }
    fCv.wait(lock, [this]() { return fAllDone; });
}

// ============================================================================
// Free Functions
// ============================================================================

void InitializeROOTThreading() {
    gROOT->SetBatch(true);
}

G4String WorkerFileName(G4int threadId) {
    std::ostringstream oss;
    oss << "epicChargeSharing_t" << threadId << ".root";
    return oss.str();
}

std::mutex& RootIOMutex() {
    return RootIOMutexInstance();
}

} // namespace ECS
