#ifndef INTERNAL_RUNACTIONSUPPORT_HH
#define INTERNAL_RUNACTIONSUPPORT_HH

#include "globals.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>

class TFile;
class TTree;

namespace runaction
{

class RootFileWriterHelper
{
public:
    RootFileWriterHelper();
    ~RootFileWriterHelper();

    void Attach(TFile* file, TTree* tree, bool ownsObjects);
    TFile* File() const;
    TTree* Tree() const;

    bool SafeWrite(bool isMultithreaded, bool isWorker);
    bool Validate(const G4String& filename, bool* hasEntries);
    void Cleanup();
    void WriteMetadataSingleThread(G4double pixelSize,
                                   G4double pixelSpacing,
                                   G4double pixelCornerOffset,
                                   G4double detSize,
                                   G4int numBlocksPerSide,
                                   G4int neighborhoodRadius);

private:
    mutable std::mutex fMutex;
    TFile* fRootFile{nullptr};
    TTree* fTree{nullptr};
    bool fOwnsObjects{false};
};

namespace support
{

class WorkerSyncHelper
{
public:
    static WorkerSyncHelper& Instance();

    void Reset(G4int totalWorkers);
    void SignalWorkerCompletion();
    void WaitForAllWorkers();

private:
    WorkerSyncHelper() = default;

    std::mutex fMutex;
    std::condition_variable fCv;
    std::atomic<int> fWorkersCompleted{0};
    int fTotalWorkers{0};
    bool fAllWorkersCompleted{false};
};

void InitializeROOTThreading();
G4String WorkerFileName(G4int threadId);
std::mutex& RootIOMutex();

} // namespace support

} // namespace runaction

#endif // INTERNAL_RUNACTIONSUPPORT_HH

