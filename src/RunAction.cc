#include "RunAction.hh"
#include "Constants.hh"
#include "Control.hh"

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdio>

#include "TFile.h"
#include "TTree.h"
#include "TNamed.h"
#include "TError.h"
#include "TROOT.h"
#include "TThread.h"
#include "TFileMerger.h"
#include "RVersion.h"

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

    auto runMacro = [&](const char* macroPath, const char* funcName) {
        if (!macroPath || !funcName) return;
        const G4String loadCmd = Form(".L %s+", macroPath);
        gROOT->ProcessLine(loadCmd.c_str());
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
}

RunAction::~RunAction()
{
}

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
    
    G4String fileName;
    if (G4Threading::IsMultithreadedApplication()) {
        if (G4Threading::IsWorkerThread()) {
            // Worker thread: create unique file for this thread
            G4int threadId = G4Threading::G4GetThreadId();
            std::ostringstream oss;
            oss << "epicChargeSharingOutput_t" << threadId << ".root";
            fileName = oss.str();
        } else {
            // Master thread: this file will be created during merge
            fileName = "epicChargeSharingOutput.root";
        }
    } else {
        // Single-threaded mode
        fileName = "epicChargeSharingOutput.root";
    }
    
    // Only create ROOT file for worker threads or single-threaded mode
    if (!G4Threading::IsMultithreadedApplication() || G4Threading::IsWorkerThread()) {
        // Lock mutex during ROOT file operations
        std::lock_guard<std::mutex> lock(fRootMutex);
        
        // Create the ROOT file with optimized settings
        fRootFile = new TFile(fileName.c_str(), "RECREATE", "", 1); // Low compression for speed
        
        if (fRootFile->IsZombie()) {
            G4cerr << "Cannot create ROOT file: " << fileName << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        
        // Set auto-flush and auto-save for better performance and safety
        fRootFile->SetCompressionLevel(1);
        
        G4cout << "ROOT file: " << fileName << G4endl;
        
        // Create the ROOT tree with optimized settings
        fTree = new TTree("Hits", "Particle hits and fitting results");
        if (!fTree) {
            G4cerr << "RunAction: Error - Failed to create ROOT tree" << G4endl;
            delete fRootFile;
            fRootFile = nullptr;
            return;
        }
        fTree->SetAutoFlush(10000);  // Flush every 10k entries
        fTree->SetAutoSave(50000);   // Save every 50k entries
        
        // Create branches
        // =============================================
        // HITS BRANCHES
        // =============================================
        fTree->Branch("x_hit", &fTrueX, "x_hit/D")->SetTitle("True Position X [mm]");
        fTree->Branch("y_hit", &fTrueY, "y_hit/D")->SetTitle("True Position Y [mm]");
        fTree->Branch("x_px", &fPixelX, "x_px/D")->SetTitle("Nearest Pixel Center X [mm]");
        fTree->Branch("y_px", &fPixelY, "y_px/D")->SetTitle("Nearest Pixel Center Y [mm]");
        // Energy branches use lowercase names per project guide
        fTree->Branch("e_dep", &fEdep, "e_dep/D")->SetTitle("Energy deposit in silicon [MeV]");
        // Pixel-pad classification flags
        fTree->Branch("first_contact_is_pixel", &fFirstContactIsPixel, "first_contact_is_pixel/O");
        fTree->Branch("geometric_is_pixel", &fGeometricIsPixel, "geometric_is_pixel/O");
        fTree->Branch("is_pixel_hit", &fIsPixelHit, "is_pixel_hit/O");
        fTree->Branch("px_hit_delta_x", &fPixelTrueDeltaX, "px_hit_delta_x/D")->SetTitle("|x_hit - x_px| [mm]");
        fTree->Branch("px_hit_delta_y", &fPixelTrueDeltaY, "px_hit_delta_y/D")->SetTitle("|y_hit - y_px| [mm]");
        
        // Neighborhood charge sharing results
        fTree->Branch("F_i", &fNeighborhoodChargeFractions)->SetTitle("Charge Fractions F_i for Neighborhood Grid Pixels");
        fTree->Branch("Q_i", &fNeighborhoodCharge)->SetTitle("Induced charge per pixel Q_i = F_i * Q_tot [C]");
        
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
            
            G4int nThreads = fTotalWorkers.load();
            std::vector<G4String> workerFileNames;
            std::vector<G4String> validFiles;
            
            // Generate expected worker file names
            for (G4int i = 0; i < nThreads; i++) {
                std::ostringstream oss;
                oss << "epicChargeSharingOutput_t" << i << ".root";
                workerFileNames.push_back(oss.str());
            }
            
            // Validate all worker files with enhanced checking
            for (const auto& workerFile : workerFileNames) {
                if (ValidateRootFile(workerFile)) {
                    validFiles.push_back(workerFile);
                    G4cout << "Master thread: Validated worker file " << workerFile << G4endl;
                } else {
                    G4cerr << "Master thread: Invalid or missing worker file " << workerFile << G4endl;
                }
            }
            
            if (validFiles.empty()) {
                G4cerr << "Master thread: No valid worker files found for merging!" << G4endl;
                return;
            }
            
            // Count total entries for verification
            G4int totalEntries = 0;
            for (const auto& validFile : validFiles) {
                TFile* testFile = TFile::Open(validFile.c_str(), "READ");
                if (testFile && !testFile->IsZombie()) {
                    TTree* testTree = (TTree*)testFile->Get("Hits");
                    if (testTree) {
                        totalEntries += testTree->GetEntries();
                    }
                    testFile->Close();
                    delete testFile;
                }
            }
            
            G4cout << "Master thread: Merging " << validFiles.size() 
                   << " files with total " << totalEntries << " entries" << G4endl;
            
            // Use ROOT's TFileMerger for robust and thread-safe file merging
            TFileMerger merger(kFALSE); // kFALSE = don't print progress
            merger.SetFastMethod(kTRUE);
            merger.SetNotrees(kFALSE);
            
            // Set output file
            if (!merger.OutputFile("epicChargeSharingOutput.root", "RECREATE", 1)) {
                G4cerr << "Master thread: Failed to set output file for merger!" << G4endl;
                return;
            }
            
            // Add all valid worker files to merger
            for (const auto& validFile : validFiles) {
                if (!merger.AddFile(validFile.c_str())) {
                    G4cerr << "Master thread: Failed to add " << validFile << " to merger" << G4endl;
                } else {
                    G4cout << "Master thread: Added " << validFile << " to merger" << G4endl;
                }
            }
            
            // Perform the merge
            Bool_t mergeResult = merger.Merge();
            if (!mergeResult) {
                G4cerr << "Master thread: File merging failed!" << G4endl;
                return;
            }
            
            G4cout << "Master thread: File merging completed successfully" << G4endl;
            
            // Add metadata to the merged file
            // NOTE: This is the ONLY place metadata should be written to avoid duplicates
            // Worker threads write only their tree data, master adds metadata once to final file
            if (fGridPixelSize > 0) {
                TFile* mergedFile = TFile::Open("epicChargeSharingOutput.root", "UPDATE");
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
            TFile* verifyFile = TFile::Open("epicChargeSharingOutput.root", "READ");
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
            for (const auto& file : validFiles) {
                if (std::remove(file.c_str()) == 0) {
                    G4cout << "Master thread: Cleaned up " << file << G4endl;
                } else {
                    G4cerr << "Master thread: Failed to clean up " << file << G4endl;
                }
            }

            // After merging and metadata write, run post-processing macros on the final file
            RunPostProcessingMacros("epicChargeSharingOutput.root");
            
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

void RunAction::SetNearestPixelPos(G4double x, G4double y)
{
    // Store nearest pixel center coordinates
    fPixelX = x;
    fPixelY = y;
}

// Initial energy is not persisted per igor.txt; no-op removed

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
    std::lock_guard<std::mutex> lock(fRootMutex);
    
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
