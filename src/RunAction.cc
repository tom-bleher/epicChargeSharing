/**
 * @file RunAction.cc
 * @brief Manages run lifecycle, ROOT I/O (thread-safe), and post-run merging/processing.
 */
#include "RunAction.hh"
#include "Constants.hh"
#include "Control.hh"
#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdio>
#include <filesystem>

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

std::mutex RunAction::fRootMutex;
std::atomic<int> RunAction::fWorkersCompleted{0};
std::atomic<int> RunAction::fTotalWorkers{0};
std::condition_variable RunAction::fWorkerCompletionCV;
std::mutex RunAction::fSyncMutex;
std::atomic<bool> RunAction::fAllWorkersCompleted{false};

static std::once_flag gRootInitFlag;

static void InitializeROOTThreading() {
    if (G4Threading::IsMultithreadedApplication()) {
        // Initialize ROOT threading support
        TThread::Initialize();
        gROOT->SetBatch(true); // Ensure batch mode for MT
        
        // Additional ROOT threading safety settings
        gErrorIgnoreLevel = kWarning; // Suppress minor ROOT warnings in MT mode
        
        // Enable ROOT thread safety if available - use different methods for different versions
        #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
        try {
            // ROOT 6.18+ supports implicit multi-threading
            if (!ROOT::IsImplicitMTEnabled()) {
                ROOT::EnableImplicitMT();
                G4cout << "ROOT: implicit MT enabled" << G4endl;
            }
        } catch (...) {
            G4cout << "ROOT multi-threading not available in this version" << G4endl;
        }
        #else
        G4cout << "ROOT version < 6.18, using basic threading support" << G4endl;
        #endif
        
        G4cout << "ROOT threading: initialized" << G4endl;
    }
}

static void RunPostProcessingMacros(const G4String& rootFilePath)
{
    if (!(Constants::RUN_PROCESSING_2D || Constants::RUN_PROCESSING_3D)) {
        return;
    }

    G4cout << "Starting fitting simulated data in " << rootFilePath << G4endl;

    // Ensure ROOT implicit MT is enabled for parallel fitting inside macros
    #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
    try {
        if (!ROOT::IsImplicitMTEnabled()) {
            ROOT::EnableImplicitMT();
            G4cout << "ROOT: implicit MT enabled for post-processing" << G4endl;
        }
    } catch (...) {
        // ignore if not available
    }
    #endif

    // Build ACLiC outputs in a local cache under build/ to avoid writing into source tree
    if (gSystem) {
        gSystem->mkdir("proc_cache", true);
        gSystem->SetBuildDir("proc_cache");
    }

    auto runMacro = [&](const char* macroPath, const char* funcName) {
        if (!macroPath || !funcName) return;
        // Force recompile each run ("++") to ensure fresh .so and corresponding .pcm are produced
        const G4String loadCmdRebuild = Form(".L %s++", macroPath);
        gROOT->ProcessLine(loadCmdRebuild.c_str());
        const G4String callCmd = Form("%s(\"%s\", %.9g);", funcName, rootFilePath.c_str(), Constants::PROCESSING_ERROR_PERCENT);
        gROOT->ProcessLine(callCmd.c_str());
    };

    if (Constants::RUN_PROCESSING_2D) {
        const G4String macroPath2D = G4String(PROJECT_SOURCE_DIR) + "/proc/processing2D.C";
        runMacro(macroPath2D.c_str(), "processing2D");
    }
    if (Constants::RUN_PROCESSING_3D) {
        const G4String macroPath3D = G4String(PROJECT_SOURCE_DIR) + "/proc/processing3D.C";
        runMacro(macroPath3D.c_str(), "processing3D");
    }
}

RunAction::RunAction()
: G4UserRunAction(),
  fRootFile(nullptr),
  fTree(nullptr),
  fTrueX(0),
  fTrueY(0),
  fPixelX(0),
  fPixelY(0),
  fEdep(0),
  fPixelTrueDeltaX(0),
  fPixelTrueDeltaY(0)

{
    fGridPixelSize = 0.0;
    fGridPixelSpacing = 0.0;
    fGridPixelCornerOffset = 0.0;
    fGridDetSize = 0.0;
    fGridNumBlocksPerSide = 0;
    fGridNeighborhoodRadius = 0;
}

RunAction::~RunAction() = default;

void RunAction::BeginOfRunAction(const G4Run* run)
{ 
    std::call_once(gRootInitFlag, InitializeROOTThreading);
    
    if (!G4Threading::IsWorkerThread()) {
        ResetSynchronization();
    }
    
    if (!run) {
        G4cerr << "RunAction: Error - Invalid run object in BeginOfRunAction" << G4endl;
        return;
    }
    
    // Synchronize detector grid parameters from the constructed geometry
    // Ensures worker threads also get the final adjusted geometry parameters
    if (auto* runMgr = G4RunManager::GetRunManager()) {
        auto* userDet = runMgr->GetUserDetectorConstruction();
        if (auto* det = dynamic_cast<const DetectorConstruction*>(userDet)) {
            SetDetectorGridParameters(
                det->GetPixelSize(),
                det->GetPixelSpacing(),
                det->GetPixelCornerOffset(),
                det->GetDetSize(),
                det->GetNumBlocksPerSide()
            );
            SetNeighborhoodRadiusMeta(det->GetNeighborhoodRadius());
        }
    }
    
    G4String fileName;
    if (G4Threading::IsMultithreadedApplication()) {
        if (G4Threading::IsWorkerThread()) {
            // Worker thread: create unique file for this thread
            G4int threadId = G4Threading::G4GetThreadId();
            std::ostringstream oss;
            oss << "epicChargeSharing_t" << threadId << ".root";
            fileName = oss.str();
        } else {
            // Master thread: this file will be created during merge
            fileName = "epicChargeSharing.root";
        }
    } else {
        // Single-threaded mode
        fileName = "epicChargeSharing.root";
    }
    
    // Only create ROOT file for worker threads or single-threaded mode
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        // Global lock only needed on master; workers have independent files
        const bool needGlobalLock = G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread();
        std::unique_ptr<std::lock_guard<std::mutex>> maybeLock;
        if (needGlobalLock) {
            maybeLock.reset(new std::lock_guard<std::mutex>(fRootMutex));
        }
        
        // Create the ROOT file with optimized settings
        fRootFile = new TFile(fileName.c_str(), "RECREATE", "", 1); // default compression setting; adjust below
        
        if (fRootFile->IsZombie()) {
            G4cerr << "Cannot create ROOT file: " << fileName << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        
        // Set auto-flush and auto-save for better performance and safety
        if (G4Threading::IsMultithreadedApplication() && G4Threading::IsWorkerThread()) {
            // Workers: maximize throughput (bigger baskets, no compression)
            fRootFile->SetCompressionLevel(0);
        } else {
            fRootFile->SetCompressionLevel(1);
        }
        
        G4cout << "ROOT file: " << fileName << G4endl;
        
        // Create the ROOT tree with optimized settings
        fTree = new TTree("Hits", "Particle hits and fitting results");
        if (!fTree) {
            G4cerr << "RunAction: Error - Failed to create ROOT tree" << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        if (G4Threading::IsMultithreadedApplication() && G4Threading::IsWorkerThread()) {
            fTree->SetAutoFlush(100000);   // Larger baskets for worker throughput
            fTree->SetAutoSave(200000);
        } else {
            fTree->SetAutoFlush(10000);
            fTree->SetAutoSave(50000);
        }
        
        // Create branches
        // =============================================
        // HITS BRANCHES
        // =============================================
        fTree->Branch("TrueX", &fTrueX, "x_hit/D")->SetTitle("True Position X [mm]");
        fTree->Branch("TrueY", &fTrueY, "y_hit/D")->SetTitle("True Position Y [mm]");
        fTree->Branch("PixelX", &fPixelX, "x_px/D")->SetTitle("Nearest Pixel Center X [mm]");
        fTree->Branch("PixelY", &fPixelY, "y_px/D")->SetTitle("Nearest Pixel Center Y [mm]");
        fTree->Branch("Edep", &fEdep, "e_dep/D")->SetTitle("Energy deposit in silicon [MeV]");
        {
            TBranch* b = fTree->Branch("isPixelHit", &fIsPixelHit, "isPixelHit/O");
            if (b) b->SetTitle("Geometric or First Contact Pixel Hit");
        }
        fTree->Branch("PixelTrueDeltaX", &fPixelTrueDeltaX, "px_hit_delta_x/D")->SetTitle("|x_hit - x_px| [mm]");
        fTree->Branch("PixelTrueDeltaY", &fPixelTrueDeltaY, "px_hit_delta_y/D")->SetTitle("|y_hit - y_px| [mm]"); 
        fTree->Branch("F_i", &fNeighborhoodChargeFractions)->SetTitle("Charge Fractions F_i for Neighborhood Grid Pixels");
        //fTree->Branch("Q_i", &fNeighborhoodCharge)->SetTitle("Induced charge per pixel Q_i = F_i * Q_tot [C]");
        
        // Full-grid pixel geometry and IDs (constant per run/thread)
        fTree->Branch("GridPixelX", &fGridPixelX)->SetTitle("Full-grid pixel centers X [mm] (row-major, size N^2)");
        fTree->Branch("GridPixelY", &fGridPixelY)->SetTitle("Full-grid pixel centers Y [mm] (row-major, size N^2)");
        fTree->Branch("GridPixelID", &fGridPixelID)->SetTitle("Full-grid pixel IDs (row-major i*N + j, size N^2)");

        // Neighborhood pixel geometry and IDs
        fTree->Branch("NeighborhoodPixelX", &fNeighborhoodPixelX)->SetTitle("Neighborhood pixel centers X [mm] (row-major, size (2r+1)^2)");
        fTree->Branch("NeighborhoodPixelY", &fNeighborhoodPixelY)->SetTitle("Neighborhood pixel centers Y [mm] (row-major, size (2r+1)^2)");
        fTree->Branch("NeighborhoodPixelID", &fNeighborhoodPixelID)->SetTitle("Neighborhood pixel IDs (global grid IDs; -1 for OOB)");
        
        G4cout << "ROOT tree 'Hits': " << fTree->GetNbranches() << " branches" << G4endl;
    
    }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    // Safety check for valid run
    if (!run) {
        G4cerr << "RunAction: Error - Invalid run object in EndOfRunAction" << G4endl;
        return;
    }
    
    G4int nofEvents = run->GetNumberOfEvent();
    G4String fileName = "";
    G4int nEntries = 0;
    
    // Worker threads: Write their individual files safely
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        
        if (fRootFile && !fRootFile->IsZombie()) {
            fileName = fRootFile->GetName();
        }
        
        if (fTree) {
            nEntries = fTree->GetEntries();
        }
        
        if (fRootFile && fTree && nofEvents > 0) {
            G4cout << "Worker thread writing ROOT file with " << nEntries 
                   << " entries from " << nofEvents << " events" << G4endl;
            
            // Use the new safe write method
            if (SafeWriteRootFile()) {
            G4cout << "Worker thread: Successfully wrote " << fileName << G4endl;
                // In single-threaded mode, run post-processing now on the produced file
                if (!G4Threading::IsMultithreadedApplication()) {
                    RunPostProcessingMacros(fileName);
                }
            } else {
                G4cerr << "Worker thread: Failed to write " << fileName << G4endl;
            }
        }
        
        // Clean up worker ROOT objects
        CleanupRootObjects();
        
        // Signal completion to master thread
        SignalWorkerCompletion();
        
        return; // Worker threads are done
    }
    
    // Master thread: Wait for workers then merge files
    G4cout << "Master thread: Waiting for all worker threads to complete..." << G4endl;
    
    // Use the new robust synchronization
    WaitForAllWorkersToComplete();
    
    // Now perform the robust file merging
    if (G4Threading::IsMultithreadedApplication()) {
        G4cout << "Master thread: Starting robust file merging..." << G4endl;
        
        try {
            // Use separate lock scope for merging
            std::lock_guard<std::mutex> lock(fRootMutex);
            
            // Be conservative: disable ROOT implicit MT during merge to avoid crashes in IO
            #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
            bool wasIMT = false;
            try {
                wasIMT = ROOT::IsImplicitMTEnabled();
                if (wasIMT) {
                    ROOT::DisableImplicitMT();
                    G4cout << "ROOT: implicit MT disabled for merge" << G4endl;
                }
            } catch (...) {
                // ignore if not available
            }
            #endif
            
            G4int nThreads = fTotalWorkers.load();
            std::vector<G4String> workerFileNames;
            workerFileNames.reserve(nThreads);
            for (G4int i = 0; i < nThreads; i++) {
                std::ostringstream oss;
                oss << "epicChargeSharing_t" << i << ".root";
                workerFileNames.push_back(oss.str());
            }

            // Quick filter: keep only existing, non-empty files (no ROOT open)
            std::vector<G4String> existingFiles;
            existingFiles.reserve(workerFileNames.size());
            for (const auto& wf : workerFileNames) {
                try {
                    if (std::filesystem::exists(wf.c_str()) && std::filesystem::file_size(wf.c_str()) > 0) {
                        existingFiles.push_back(wf);
                    }
                } catch (...) {
                    // ignore and skip problematic paths
                }
            }

            if (existingFiles.empty()) {
                G4cerr << "Master thread: No worker files found for merging!" << G4endl;
                // Re-enable IMT if it was previously on
                #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
                try {
                    if (wasIMT) {
                        ROOT::EnableImplicitMT();
                        G4cout << "ROOT: implicit MT re-enabled after merge" << G4endl;
                    }
                } catch (...) {}
                #endif
                return;
            }

            // Helper: merge files with fast path and fallback
            auto mergeFiles = [&](const std::vector<G4String>& inputs, const G4String& output) -> bool {
                auto doMerge = [&](Bool_t fast) -> Bool_t {
                    TFileMerger merger(kFALSE);
                    merger.SetFastMethod(fast);
                    merger.SetNotrees(kFALSE);
                    if (!merger.OutputFile(output.c_str(), "RECREATE", 1)) {
                        G4cerr << "Master thread: Failed to set output file for merger!" << G4endl;
                        return kFALSE;
                    }
                    Int_t added = 0;
                    for (const auto& in : inputs) {
                        // Add only files that still exist at this moment
                        if (std::filesystem::exists(in.c_str())) {
                            if (merger.AddFile(in.c_str())) {
                                added++;
                            } else {
                                G4cerr << "Master thread: Skipped unreadable file " << in << G4endl;
                            }
                        }
                    }
                    if (added == 0) {
                        G4cerr << "Master thread: No readable input files for merge to " << output << G4endl;
                        return kFALSE;
                    }
                    return merger.Merge();
                };

                Bool_t ok = kFALSE;
                try { ok = doMerge(kTRUE); } catch (...) { ok = kFALSE; }
                if (!ok) {
                    G4cout << "Master thread: Fast merge failed; retrying with safe merge..." << G4endl;
                    try { ok = doMerge(kFALSE); } catch (...) { ok = kFALSE; }
                }
                return ok;
            };

            // Chunked merge to limit simultaneously opened files
            const std::size_t kBatchSize = 16; // limit open files and memory usage
            std::vector<G4String> chunkOutputs;

            if (existingFiles.size() > kBatchSize) {
                for (std::size_t offset = 0, chunkIdx = 0; offset < existingFiles.size(); offset += kBatchSize, ++chunkIdx) {
                    const std::size_t end = std::min(offset + kBatchSize, existingFiles.size());
                    std::vector<G4String> chunk(existingFiles.begin() + offset, existingFiles.begin() + end);
                    G4String chunkName = Form("epicChargeSharing_chunk_%zu.root", chunkIdx);
                    if (!mergeFiles(chunk, chunkName)) {
                        G4cerr << "Master thread: Chunk merge failed" << G4endl;
                        // Attempt to clean partial chunk
                        try { std::filesystem::remove(chunkName.c_str()); } catch (...) {}
                        #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
                        try {
                            if (wasIMT) {
                                ROOT::EnableImplicitMT();
                                G4cout << "ROOT: implicit MT re-enabled after merge" << G4endl;
                            }
                        } catch (...) {}
                        #endif
                        return;
                    }
                    chunkOutputs.push_back(chunkName);
                }
            }

            const bool usedChunks = !chunkOutputs.empty();
            const std::vector<G4String>& finalInputs = usedChunks ? chunkOutputs : existingFiles;

            G4cout << "Master thread: Merging " << finalInputs.size() << " file(s) into final output" << G4endl;

            if (!mergeFiles(finalInputs, "epicChargeSharing.root")) {
                G4cerr << "Master thread: File merging failed!" << G4endl;
                // cleanup chunks if any
                for (const auto& f : chunkOutputs) { try { std::filesystem::remove(f.c_str()); } catch (...) {} }
                #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
                try {
                    if (wasIMT) {
                        ROOT::EnableImplicitMT();
                        G4cout << "ROOT: implicit MT re-enabled after merge" << G4endl;
                    }
                } catch (...) {}
                #endif
                return;
            }

            G4cout << "Master thread: File merging completed successfully" << G4endl;

            // Remove temporary chunk files
            for (const auto& f : chunkOutputs) { try { std::filesystem::remove(f.c_str()); } catch (...) {} }
            
            // Add metadata to the merged file
            // NOTE: This is the ONLY place metadata should be written to avoid duplicates
            // Worker threads write only their tree data, master adds metadata once to final file
            if (fGridPixelSize > 0) {
                TFile* mergedFile = TFile::Open("epicChargeSharing.root", "UPDATE");
                if (mergedFile && !mergedFile->IsZombie()) {
                    mergedFile->cd();
                    
                    TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", fGridPixelSize));
                    TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing));
                    TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset));
                    TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", fGridDetSize));
                    TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
                    TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", fGridNeighborhoodRadius > 0 ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS));
                    
                    pixelSizeMeta.Write();
                    pixelSpacingMeta.Write();
                    pixelCornerOffsetMeta.Write();
                    detSizeMeta.Write();
                    numBlocksMeta.Write();
                    neighborhoodRadiusMeta.Write();
                    
                    mergedFile->Close();
                    delete mergedFile;
                    
                    G4cout << "Master thread: Saved detector grid metadata to merged file" << G4endl;
                } else {
                    G4cerr << "Master thread: Failed to open merged file for metadata" << G4endl;
                }
            }
            
            // Verify the merged file
            TFile* verifyFile = TFile::Open("epicChargeSharing.root", "READ");
            if (verifyFile && !verifyFile->IsZombie()) {
                TTree* verifyTree = (TTree*)verifyFile->Get("Hits");
                if (verifyTree) {
                    G4cout << "Master thread: Successfully created merged file with " 
                           << verifyTree->GetEntries() << " entries" << G4endl;
                }
                verifyFile->Close();
                delete verifyFile;
            } else {
                G4cerr << "Master thread: Failed to verify merged file" << G4endl;
            }
            
            // Clean up worker files after success merge
            for (const auto& file : existingFiles) {
                try {
                    if (std::filesystem::exists(file.c_str())) {
                        if (std::filesystem::remove(file.c_str())) {
                            G4cout << "Master thread: Cleaned up " << file << G4endl;
                        } else {
                            G4cerr << "Master thread: Failed to clean up " << file << G4endl;
                        }
                    }
                } catch (...) {}
            }

            // Re-enable IMT before running post-processing macros (they can use IMT)
            #if ROOT_VERSION_CODE >= ROOT_VERSION(6,18,0)
            try {
                if (wasIMT) {
                    ROOT::EnableImplicitMT();
                    G4cout << "ROOT: implicit MT re-enabled after merge" << G4endl;
                }
            } catch (...) {}
            #endif

            // After merging and metadata write, run post-processing macros on the final file
            RunPostProcessingMacros("epicChargeSharing.root");
            
        } catch (const std::exception& e) {
            G4cerr << "Master thread: Exception during robust file merging: " << e.what() << G4endl;
        }
    }
    
    G4cout << "Master thread: File operations completed" << G4endl;
}

void RunAction::SetEventData(G4double edep, G4double x, G4double y, G4double z) 
{
    // Store energy deposition in MeV (Geant4 internal energy unit is MeV)
    fEdep = edep;
    
    // Store positions in mm (Geant4 internal length unit is mm)
    fTrueX = x;
    fTrueY = y;
}

void RunAction::SetPixelClassification(G4bool isPixelHit, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY)
{
    // Store the classification and delta values from pixel center to true position
    fIsPixelHit = isPixelHit;
    fPixelTrueDeltaX = pixelTrueDeltaX;
    fPixelTrueDeltaY = pixelTrueDeltaY;
}

void RunAction::SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                                 const std::vector<G4double>& chargeCoulombs)
{
    // Store the neighborhood (9x9) grid charge sharing data for non-pixel hits
    fNeighborhoodChargeFractions = chargeFractions;
    fNeighborhoodCharge = chargeCoulombs;
}

void RunAction::FillTree()
{
    if (!fTree || !fRootFile || fRootFile->IsZombie()) {
        G4cerr << "Error: Invalid ROOT file or tree in FillTree()" << G4endl;
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(fRootMutex);
        
        // Fill the tree with all current data (including scorer data)
        G4int fillResult = fTree->Fill();
        
        // Validate successful tree filling
        if (fillResult < 0) {
            G4cerr << "Error: Tree Fill() returned error code " << fillResult << G4endl;
            return;
        }

        
    } catch (const std::exception& e) {
        G4cerr << "Exception in FillTree: " << e.what() << G4endl;
    }
}

void RunAction::SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                           G4double pixelCornerOffset, G4double detSize, 
                                           G4int numBlocksPerSide)
{
    // Safety check for valid parameters
    if (pixelSize <= 0 || pixelSpacing <= 0 || detSize <= 0 || numBlocksPerSide <= 0) {
        G4cerr << "RunAction: Error - Invalid detector grid parameters provided" << G4endl;
        return;
    }
    
    // Store the detector grid parameters for saving to ROOT metadata
    fGridPixelSize = pixelSize;
    fGridPixelSpacing = pixelSpacing;
    fGridPixelCornerOffset = pixelCornerOffset;
    fGridDetSize = detSize;
    fGridNumBlocksPerSide = numBlocksPerSide;
    
    G4cout << "RunAction: Detector grid parameters set:" << G4endl;
    G4cout << "  Pixel Size: " << fGridPixelSize << " mm" << G4endl;
    G4cout << "  Pixel Spacing: " << fGridPixelSpacing << " mm" << G4endl;
    G4cout << "  Pixel Corner Offset: " << fGridPixelCornerOffset << " mm" << G4endl;
    G4cout << "  Detector Size: " << fGridDetSize << " mm" << G4endl;
    G4cout << "  Number of Blocks per Side: " << fGridNumBlocksPerSide << G4endl;

    // Populate full-grid pixel IDs (row-major with i major)
    fGridPixelID.clear();
    fGridPixelX.clear();
    fGridPixelY.clear();
    const G4int n = fGridNumBlocksPerSide;
    if (n > 0) {
        const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
        fGridPixelID.reserve(total);
        fGridPixelX.reserve(total);
        fGridPixelY.reserve(total);

        const G4double firstPixelPos = -fGridDetSize/2 + fGridPixelCornerOffset + fGridPixelSize/2;
        for (G4int i = 0; i < n; ++i) {
            for (G4int j = 0; j < n; ++j) {
                const G4int id = i * n + j; // row-major: x-index major
                fGridPixelID.push_back(id);
                const G4double x = firstPixelPos + i * fGridPixelSpacing;
                const G4double y = firstPixelPos + j * fGridPixelSpacing;
                fGridPixelX.push_back(x);
                fGridPixelY.push_back(y);
            }
        }
    }
}

void RunAction::SetNeighborhoodPixelData(const std::vector<G4double>& xs,
                                  const std::vector<G4double>& ys,
                                  const std::vector<G4int>& ids)
{
    fNeighborhoodPixelX = xs;
    fNeighborhoodPixelY = ys;
    fNeighborhoodPixelID = ids;
}

// =============================================
// THREAD SYNCHRONIZATION METHODS
// =============================================

void RunAction::ResetSynchronization()
{
    std::lock_guard<std::mutex> lock(fSyncMutex);
    fWorkersCompleted = 0;
    fTotalWorkers = 0;
    fAllWorkersCompleted = false;
    
    if (G4Threading::IsMultithreadedApplication()) {
        fTotalWorkers = G4Threading::GetNumberOfRunningWorkerThreads();
    }
    
    G4cout << "RunAction: Reset synchronization for " << fTotalWorkers.load() << " worker threads" << G4endl;
}

void RunAction::SignalWorkerCompletion()
{
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        std::unique_lock<std::mutex> lock(fSyncMutex);
        fWorkersCompleted++;
        
        G4cout << "RunAction: Worker thread completed (" << fWorkersCompleted.load() 
               << "/" << fTotalWorkers.load() << ")" << G4endl;
        
        if (fWorkersCompleted >= fTotalWorkers) {
            fAllWorkersCompleted = true;
            lock.unlock();
            fWorkerCompletionCV.notify_all();
            G4cout << "RunAction: All worker threads completed, notifying master" << G4endl;
        }
    }
}

void RunAction::WaitForAllWorkersToComplete()
{
    if (G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread()) {
        std::unique_lock<std::mutex> lock(fSyncMutex);
        
        G4cout << "RunAction: Master thread waiting for " << fTotalWorkers.load() << " workers to complete..." << G4endl;
        
        // Wait for all workers to complete with timeout
        auto timeout = std::chrono::seconds(30); // 30 Sec timeout
        bool completed = fWorkerCompletionCV.wait_for(lock, timeout, []() {
            return fAllWorkersCompleted.load();
        });
        
        if (completed) {
            G4cout << "RunAction: All workers completed successfully" << G4endl;
        } else {
            G4cerr << "RunAction: Warning - Timeout waiting for workers to complete!" << G4endl;
        }
    }
}

// =============================================
// SAFE ROOT FILE OPERATIONS
// =============================================

bool RunAction::ValidateRootFile(const G4String& filename)
{
    if (filename.empty()) {
        G4cerr << "RunAction: Error - Empty filename provided for validation" << G4endl;
        return false;
    }
    
    TFile* testFile = nullptr;
    try {
        testFile = TFile::Open(filename.c_str(), "READ");
        if (!testFile || testFile->IsZombie()) {
            G4cerr << "RunAction: Error - Cannot open or corrupted file: " << filename << G4endl;
            if (testFile) delete testFile;
            return false;
        }
        
        TTree* testTree = (TTree*)testFile->Get("Hits");
        if (!testTree) {
            G4cerr << "RunAction: Error - No 'Hits' tree found in file: " << filename << G4endl;
            testFile->Close();
            delete testFile;
            return false;
        }
        
        bool isValid = testTree->GetEntries() > 0;
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

bool RunAction::SafeWriteRootFile()
{
    // Global lock only for master-thread operations; workers write independently
    const bool needGlobalLock = G4Threading::IsMultithreadedApplication() && !G4Threading::IsWorkerThread();
    std::unique_ptr<std::lock_guard<std::mutex>> maybeLock;
    if (needGlobalLock) {
        maybeLock.reset(new std::lock_guard<std::mutex>(fRootMutex));
    }
    
    if (!fRootFile || !fTree || fRootFile->IsZombie()) {
        G4cerr << "RunAction: Cannot write - invalid ROOT file or tree" << G4endl;
        return false;
    }
    
    try {
        // In single-threaded mode, write metadata here since there's no merging
        // In multi-threaded mode, metadata is written only by master thread after merging
        if (!G4Threading::IsMultithreadedApplication() && fGridPixelSize > 0) {
            fRootFile->cd();
            
            // Create and write metadata objects
            TNamed pixelSizeMeta("GridPixelSize_mm", Form("%.6f", fGridPixelSize));
            TNamed pixelSpacingMeta("GridPixelSpacing_mm", Form("%.6f", fGridPixelSpacing));
            TNamed pixelCornerOffsetMeta("GridPixelCornerOffset_mm", Form("%.6f", fGridPixelCornerOffset));
            TNamed detSizeMeta("GridDetectorSize_mm", Form("%.6f", fGridDetSize));
            TNamed numBlocksMeta("GridNumBlocksPerSide", Form("%d", fGridNumBlocksPerSide));
            TNamed neighborhoodRadiusMeta("NeighborhoodRadius", Form("%d", fGridNeighborhoodRadius > 0 ? fGridNeighborhoodRadius : Constants::NEIGHBORHOOD_RADIUS));
            
            pixelSizeMeta.Write();
            pixelSpacingMeta.Write();
            pixelCornerOffsetMeta.Write();
            detSizeMeta.Write();
            numBlocksMeta.Write();
            neighborhoodRadiusMeta.Write();
            
            G4cout << "RunAction: Saved detector grid metadata to single-threaded file" << G4endl;
        }
        
        // Make sure all in-memory baskets are flushed only once at end-of-run
        fTree->FlushBaskets();   // replaces thousands of small AutoSave flushes
        
        // Write the full tree in a single operation (metadata added above for single-threaded, by master for multi-threaded)
        fRootFile->cd();
        fTree->Write();
        // Final flush for the file header / directory structure
        fRootFile->Flush();
        
        G4cout << "RunAction: Successfully wrote ROOT file with " << fTree->GetEntries() << " entries" << G4endl;
        return true;
        
    } catch (const std::exception& e) {
        G4cerr << "RunAction: Exception writing ROOT file: " << e.what() << G4endl;
        return false;
    }
}

void RunAction::CleanupRootObjects()
{
    std::lock_guard<std::mutex> lock(fRootMutex);
    
    try {
        if (fRootFile) {
            if (fRootFile->IsOpen() && !fRootFile->IsZombie()) {
                fRootFile->Close();
            }
            delete fRootFile;
            fRootFile = nullptr;
            fTree = nullptr; // Tree is owned by file
            G4cout << "RunAction: Successfully cleaned up ROOT objects" << G4endl;
        }
    } catch (const std::exception& e) {
        G4cerr << "RunAction: Exception during ROOT cleanup: " << e.what() << G4endl;
        // Force cleanup even if exception occurred
        fRootFile = nullptr;
        fTree = nullptr;
    }
}
