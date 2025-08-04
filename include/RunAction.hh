#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <vector>
#include <string>

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "G4Threading.hh"
#include <mutex>
#include <atomic>
#include <condition_variable>

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    virtual ~RunAction();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    // Access methods for ROOT objects
    TFile* GetRootFile() const { return fRootFile; }
    TTree* GetTree() const { return fTree; }
    
    // Thread synchronization for ROOT file operations
    static void WaitForAllWorkersToComplete();
    static void SignalWorkerCompletion();
    static void ResetSynchronization();
    
    // Safe ROOT file operations
    bool SafeWriteRootFile();
    bool ValidateRootFile(const G4String& filename);
    void CleanupRootObjects();
    
    // Auto-save mechanism
    void EnableAutoSave(G4int interval = 1000);
    void DisableAutoSave();
    void PerformAutoSave();
    
    // Variables for the branch (edep [MeV], positions [mm])
    void SetEventData(G4double edep, G4double x, G4double y, G4double z);
    
    // Method to set initial particle gun position [mm]
    void SetInitialPos(G4double x, G4double y, G4double z);
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPos(G4double x, G4double y);
    
    // Method to set initial particle energy [MeV]
    void SetInitialEnergy(G4double energy);
    
    // Method to set pixel hit status
    void SetPixelHitStatus(G4bool isPixelHit);
    
    // Method to set pixel classification data (hit status and delta values)
    void SetPixelClassification(G4bool isWithinD0, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY);
    
    // Method to set neighborhood (9x9) grid angle data for non-pixel hits
    void SetNeighborhoodGridData(const std::vector<G4double>& angles);
    
    // Method to set neighborhood (9x9) grid charge sharing data for non-pixel hits
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& distances,
                          const std::vector<G4double>& chargeValues,
                          const std::vector<G4double>& chargeCoulombs);
    
    // Method to set detector grid parameters for saving to ROOT
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    
    // Method to set 2D Gauss fit results (central row and column fitting)
    void Set2DGaussResults(G4double x_center, G4double x_sigma, G4double x_amp,
                                G4double x_center_err, G4double x_sigma_err, G4double x_amp_err,
                                G4double x_vert_offset, G4double x_vert_offset_err,
                                G4double x_chi2red, G4double x_pp, G4int x_dof,
                                G4double y_center, G4double y_sigma, G4double y_amp,
                                G4double y_center_err, G4double y_sigma_err, G4double y_amp_err,
                                G4double y_vert_offset, G4double y_vert_offset_err,
                                G4double y_chi2red, G4double y_pp, G4int y_dof,
                                G4double x_charge_err, G4double y_charge_err,
                                G4bool fit_success);
    
    // Method to set diagonal Gauss fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    void SetDiagGaussResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_amp,
                                      G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_amp_err,
                                      G4double main_diag_x_vert_offset, G4double main_diag_x_vert_offset_err,
                                      G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_success,
                                      G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_amp,
                                      G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_amp_err,
                                      G4double main_diag_y_vert_offset, G4double main_diag_y_vert_offset_err,
                                      G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_success,
                                      G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_amp,
                                      G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_amp_err,
                                      G4double sec_diag_x_vert_offset, G4double sec_diag_x_vert_offset_err,
                                      G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_success,
                                      G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_amp,
                                      G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_amp_err,
                                      G4double sec_diag_y_vert_offset, G4double sec_diag_y_vert_offset_err,
                                      G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_success,
                                                                             G4bool fit_success);

    // Method to set 3D Gauss fit results
    void Set3DGaussResults(G4double center_x, G4double center_y, G4double sigma_x, G4double sigma_y, G4double amp, G4double vert_offset,
                                        G4double center_x_err, G4double center_y_err, G4double sigma_x_err, G4double sigma_y_err, G4double amp_err, G4double vert_offset_err,
                                        G4double chi2red, G4double pp, G4int dof,
                                        G4double charge_err,
                                        G4bool fit_success);
    
    // Method to set 2D Lorentz fit results (central row and column fitting)
    void Set2DLorentzResults(G4double x_center, G4double x_gamma, G4double x_amp,
                                  G4double x_center_err, G4double x_gamma_err, G4double x_amp_err,
                                  G4double x_vert_offset, G4double x_vert_offset_err,
                                  G4double x_chi2red, G4double x_pp, G4int x_dof,
                                  G4double y_center, G4double y_gamma, G4double y_amp,
                                  G4double y_center_err, G4double y_gamma_err, G4double y_amp_err,
                                  G4double y_vert_offset, G4double y_vert_offset_err,
                                  G4double y_chi2red, G4double y_pp, G4int y_dof,
                                  G4double x_charge_err, G4double y_charge_err,
                                  G4bool fit_success);
    

    
    // Method to set diagonal Lorentz fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    void SetDiagLorentzResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_amp,
                                        G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_amp_err,
                                        G4double main_diag_x_vert_offset, G4double main_diag_x_vert_offset_err,
                                        G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_success,
                                        G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_amp,
                                        G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_amp_err,
                                        G4double main_diag_y_vert_offset, G4double main_diag_y_vert_offset_err,
                                        G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_success,
                                        G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_amp,
                                        G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_amp_err,
                                        G4double sec_diag_x_vert_offset, G4double sec_diag_x_vert_offset_err,
                                        G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_success,
                                        G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_amp,
                                        G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_amp_err,
                                        G4double sec_diag_y_vert_offset, G4double sec_diag_y_vert_offset_err,
                                        G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_success,
                                        G4bool fit_success);


    
    // Fill the ROOT tree with current event data
    void FillTree();
    
    // Method to store automatic radius selection results
    void SetAutoRadiusResults(G4int selectedRadius);
    
    // Method to set scorer data from Multi-Functional Detector
    void SetScorerData(G4double energyDeposit, G4int hitCount, G4bool dataValid);
    
    // Method to validate scorer data before tree storage
    void ValidateScorerDataForTreeStorage();
    
    // Method to verify scorer data is written to ROOT tree
    void VerifyScorerDataInTree();
    
    // Method to set hit purity tracking data from EventAction
    void SetHitPurityData(G4bool pureSiliconHit, G4bool aluminumContaminated, G4bool chargeCalculationEnabled);
 
private:
    // =============================================
    // COORDINATE TRANSFORMATION HELPER METHODS
    // =============================================
    
    // Apply rotation matrix transformation for diagonal coordinates
    void TransformDiagCoords(G4double x_prime, G4double y_prime, G4double theta_deg, 
                                      G4double& x_transformed, G4double& y_transformed);
    
    // Calc transformed coordinates for all diagonal fits
    void CalcTransformedDiagCoords();
    
    // Calc mean estimations from all fitting methods
    void CalcMeanEstimations();
    
    // Helper functions to organize branch creation
    void CreateHitsBranches();
    void CreateGaussBranches();
    void CreateLorentzBranches();
    void Create3DBranches();
    void CreateNeighborhoodBranches();
    void CreateMetadataBranches();

    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Thread synchronization for robust file operations
    static std::atomic<int> fWorkersCompleted;
    static std::atomic<int> fTotalWorkers;
    static std::condition_variable fWorkerCompletionCV;
    static std::mutex fSyncMutex;
    static std::atomic<bool> fAllWorkersCompleted;
    
    // Auto-save mechanism
    G4bool fAutoSaveEnabled;
    G4int fAutoSaveInterval;
    G4int fEventsSinceLastSave;
    
    // Initial particle gun position
    G4double fInitialX, fInitialY, fInitialZ;
    
    // Nearest pixel position
    G4double fNearestPixelX, fNearestPixelY;

    // =============================================
    // HITS DATA VARIABLES
    // =============================================
    G4double fTrueX;   // True Hit pos X [mm]
    G4double fTrueY;   // True Hit pos Y [mm]
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fEdep;   // Energy depositionit [MeV]
    G4double fPixelTrueDeltaX; // Delta X from pixel center to true pos [mm] (x_pixel - x_true)
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to true pos [mm] (y_pixel - y_true)
    
    // Delta variables for estimations vs true pos
    G4double fGaussRowDeltaX;
    G4double fGaussColDeltaY;
    G4double fLorentzRowDeltaX;
    G4double fLorentzColDeltaY;
    
    // 3D fitting delta variables
    G4double f3DLorentzDeltaX;          // Delta X from 3D Lorentz fit to True Pos [mm]
    G4double f3DLorentzDeltaY;          // Delta Y from 3D Lorentz fit to True Pos [mm]
    G4double f3DGaussDeltaX;            // Delta X from 3D Gauss fit to True Pos [mm]
    G4double f3DGaussDeltaY;            // Delta Y from 3D Gauss fit to True Pos [mm]

    // =============================================
    // TRANSFORMED DIAG COORDINATES (ROTATION MATRIX)
    // =============================================
    
    // Transformed main diagonal coordinates (θ = 45°)
    G4double fGaussMainDiagTransformedX;      // Transformed X from main diagonal (x' -> x)
    G4double fGaussMainDiagTransformedY;      // Transformed Y from main diagonal (y' -> y)
    G4double fLorentzMainDiagTransformedX;    // Transformed X from main diagonal (x' -> x)
    G4double fLorentzMainDiagTransformedY;    // Transformed Y from main diagonal (y' -> y)
    
    // Transformed secondary diagonal coordinates (θ = -45°)
    G4double fGaussSecDiagTransformedX;    // Transformed X from secondary diagonal (x' -> x)
    G4double fGaussSecDiagTransformedY;    // Transformed Y from secondary diagonal (y' -> y)
    G4double fLorentzSecDiagTransformedX;  // Transformed X from secondary diagonal (x' -> x)
    G4double fLorentzSecDiagTransformedY;  // Transformed Y from secondary diagonal (y' -> y)

    // Delta values for transformed coordinates vs true pos
    G4double fGaussMainDiagTransformedDeltaX;   // x_transformed - x_true (main diagonal)
    G4double fGaussMainDiagTransformedDeltaY;   // y_transformed - y_true (main diagonal)
    G4double fGaussSecDiagTransformedDeltaX; // x_transformed - x_true (secondary diagonal)
    G4double fGaussSecDiagTransformedDeltaY; // y_transformed - y_true (secondary diagonal)
    G4double fLorentzMainDiagTransformedDeltaX;   // x_transformed - x_true (main diagonal)
    G4double fLorentzMainDiagTransformedDeltaY;   // y_transformed - y_true (main diagonal)
    G4double fLorentzSecDiagTransformedDeltaX; // x_transformed - x_true (secondary diagonal)
    G4double fLorentzSecDiagTransformedDeltaY; // y_transformed - y_true (secondary diagonal)

    // =============================================
    // MEAN ESTIMATIONS FROM ALL FIT METHODS
    // =============================================
    
    // Mean of all X coordinate estimations (row, transformed diagonals)
    G4double fGaussMeanTrueDeltaX;   // Mean delta X from all Gauss estimation methods to True Pos [mm]
    G4double fGaussMeanTrueDeltaY;   // Mean delta Y from all Gauss estimation methods to True Pos [mm]
    G4double fLorentzMeanTrueDeltaX; // Mean delta X from all Lorentz estimation methods to True Pos [mm]
    G4double fLorentzMeanTrueDeltaY; // Mean delta Y from all Lorentz estimation methods to True Pos [mm]

    // =============================================
    // AUTOMATIC RADIUS SELECTION VARIABLES
    // =============================================
    
    G4int fSelectedRadius;          // Automatically selected radius for this event

    // =============================================
    // GAUSS FITS VARIABLES
    // =============================================
    
    // GaussRow/GaussRowX
    G4double fGaussRowAmp;
    G4double fGaussRowAmpErr;
    G4double fGaussRowSigma;
    G4double fGaussRowSigmaErr;
    G4double fGaussRowVertOffset;
    G4double fGaussRowVertOffsetErr;
    G4double fGaussRowCenter;
    G4double fGaussRowCenterErr;
    G4double fGaussRowChi2red;
    G4double fGaussRowPp;
    G4int fGaussRowDOF;
    
    // Charge uncertainties (5% of max charge for Gauss fits)
    G4double fGaussRowChargeErr;     // Row charge uncertainty (5% of max charge)
    
    // GaussCol/GaussColY
    G4double fGaussColAmp;
    G4double fGaussColAmpErr;
    G4double fGaussColSigma;
    G4double fGaussColSigmaErr;
    G4double fGaussColVertOffset;
    G4double fGaussColVertOffsetErr;
    G4double fGaussColCenter;
    G4double fGaussColCenterErr;
    G4double fGaussColChi2red;
    G4double fGaussColPp;
    G4int fGaussColDOF;
    
    // Charge err for Gauss column fit
    G4double fGaussColChargeErr;  // Col charge uncertainty (5% of max charge)
    
    // GaussMainDiag/GaussMainDiagX
    G4double fGaussMainDiagXAmp;
    G4double fGaussMainDiagXAmpErr;
    G4double fGaussMainDiagXSigma;
    G4double fGaussMainDiagXSigmaErr;
    G4double fGaussMainDiagXVertOffset;
    G4double fGaussMainDiagXVertOffsetErr;
    G4double fGaussMainDiagXCenter;
    G4double fGaussMainDiagXCenterErr;
    G4double fGaussMainDiagXChi2red;
    G4double fGaussMainDiagXPp;
    G4int fGaussMainDiagXDOF;
    
    // GaussMainDiag/GaussMainDiagY
    G4double fGaussMainDiagYAmp;
    G4double fGaussMainDiagYAmpErr;
    G4double fGaussMainDiagYSigma;
    G4double fGaussMainDiagYSigmaErr;
    G4double fGaussMainDiagYVertOffset;
    G4double fGaussMainDiagYVertOffsetErr;
    G4double fGaussMainDiagYCenter;
    G4double fGaussMainDiagYCenterErr;
    G4double fGaussMainDiagYChi2red;
    G4double fGaussMainDiagYPp;
    G4int fGaussMainDiagYDOF;
    
    // GaussSecDiag/GaussSecDiagX
    G4double fGaussSecDiagXAmp;
    G4double fGaussSecDiagXAmpErr;
    G4double fGaussSecDiagXSigma;
    G4double fGaussSecDiagXSigmaErr;
    G4double fGaussSecDiagXVertOffset;
    G4double fGaussSecDiagXVertOffsetErr;
    G4double fGaussSecDiagXCenter;
    G4double fGaussSecDiagXCenterErr;
    G4double fGaussSecDiagXChi2red;
    G4double fGaussSecDiagXPp;
    G4int fGaussSecDiagXDOF;
    
    // GaussSecDiag/GaussSecDiagY
    G4double fGaussSecDiagYAmp;
    G4double fGaussSecDiagYAmpErr;
    G4double fGaussSecDiagYSigma;
    G4double fGaussSecDiagYSigmaErr;
    G4double fGaussSecDiagYVertOffset;
    G4double fGaussSecDiagYVertOffsetErr;
    G4double fGaussSecDiagYCenter;
    G4double fGaussSecDiagYCenterErr;
    G4double fGaussSecDiagYChi2red;
    G4double fGaussSecDiagYPp;
    G4int fGaussSecDiagYDOF;

    // =============================================
    // LORENTZ FITS VARIABLES
    // =============================================
    
    // LorentzRow/LorentzRowX
    G4double fLorentzRowAmp;
    G4double fLorentzRowAmpErr;
    G4double fLorentzRowGamma;
    G4double fLorentzRowGammaErr;
    G4double fLorentzRowVertOffset;
    G4double fLorentzRowVertOffsetErr;
    G4double fLorentzRowCenter;
    G4double fLorentzRowCenterErr;
    G4double fLorentzRowChi2red;
    G4double fLorentzRowPp;
    G4int fLorentzRowDOF;
    
    // Charge err for Lorentz row fit  
    G4double fLorentzRowChargeErr;   // Row charge uncertainty (5% of max charge)
    
    // LorentzCol/LorentzColY
    G4double fLorentzColAmp;
    G4double fLorentzColAmpErr;
    G4double fLorentzColGamma;
    G4double fLorentzColGammaErr;
    G4double fLorentzColVertOffset;
    G4double fLorentzColVertOffsetErr;
    G4double fLorentzColCenter;
    G4double fLorentzColCenterErr;
    G4double fLorentzColChi2red;
    G4double fLorentzColPp;
    G4int fLorentzColDOF;
    
    // Charge err for Lorentz column fit
    G4double fLorentzColChargeErr; // Column charge uncertainty (5% of max charge)
    
    // LorentzMainDiag/LorentzMainDiagX
    G4double fLorentzMainDiagXAmp;
    G4double fLorentzMainDiagXAmpErr;
    G4double fLorentzMainDiagXGamma;
    G4double fLorentzMainDiagXGammaErr;
    G4double fLorentzMainDiagXVertOffset;
    G4double fLorentzMainDiagXVertOffsetErr;
    G4double fLorentzMainDiagXCenter;
    G4double fLorentzMainDiagXCenterErr;
    G4double fLorentzMainDiagXChi2red;
    G4double fLorentzMainDiagXPp;
    G4int fLorentzMainDiagXDOF;
    
    // LorentzMainDiag/LorentzMainDiagY
    G4double fLorentzMainDiagYAmp;
    G4double fLorentzMainDiagYAmpErr;
    G4double fLorentzMainDiagYGamma;
    G4double fLorentzMainDiagYGammaErr;
    G4double fLorentzMainDiagYVertOffset;
    G4double fLorentzMainDiagYVertOffsetErr;
    G4double fLorentzMainDiagYCenter;
    G4double fLorentzMainDiagYCenterErr;
    G4double fLorentzMainDiagYChi2red;
    G4double fLorentzMainDiagYPp;
    G4int fLorentzMainDiagYDOF;
    
    // LorentzSecDiag/LorentzSecDiagX
    G4double fLorentzSecDiagXAmp;
    G4double fLorentzSecDiagXAmpErr;
    G4double fLorentzSecDiagXGamma;
    G4double fLorentzSecDiagXGammaErr;
    G4double fLorentzSecDiagXVertOffset;
    G4double fLorentzSecDiagXVertOffsetErr;
    G4double fLorentzSecDiagXCenter;
    G4double fLorentzSecDiagXCenterErr;
    G4double fLorentzSecDiagXChi2red;
    G4double fLorentzSecDiagXPp;
    G4int fLorentzSecDiagXDOF;
    
    // LorentzSecDiag/LorentzSecDiagY
    G4double fLorentzSecDiagYAmp;
    G4double fLorentzSecDiagYAmpErr;
    G4double fLorentzSecDiagYGamma;
    G4double fLorentzSecDiagYGammaErr;
    G4double fLorentzSecDiagYVertOffset;
    G4double fLorentzSecDiagYVertOffsetErr;
    G4double fLorentzSecDiagYCenter;
    G4double fLorentzSecDiagYCenterErr;
    G4double fLorentzSecDiagYChi2red;
    G4double fLorentzSecDiagYPp;
    G4int fLorentzSecDiagYDOF;

    // =============================================
    // 3D GAUSS FITS VARIABLES
    // =============================================
    G4double f3DGaussAmp;
    G4double f3DGaussAmpErr;
    G4double f3DGaussSigmaX;
    G4double f3DGaussSigmaXErr;
    G4double f3DGaussSigmaY;
    G4double f3DGaussSigmaYErr;
    G4double f3DGaussVertOffset;
    G4double f3DGaussVertOffsetErr;
    G4double f3DGaussCenterX;
    G4double f3DGaussCenterXErr;
    G4double f3DGaussCenterY;
    G4double f3DGaussCenterYErr;
    G4double f3DGaussChi2red;
    G4double f3DGaussPp;
    G4int f3DGaussDOF;
    G4double f3DGaussChargeErr;
    G4bool f3DGaussSuccess;

    // =============================================
    // 3D LORENTZ FITS VARIABLES
    // =============================================
    G4double f3DLorentzAmp;
    G4double f3DLorentzAmpErr;
    G4double f3DLorentzGammaX;
    G4double f3DLorentzGammaXErr;
    G4double f3DLorentzGammaY;
    G4double f3DLorentzGammaYErr;
    G4double f3DLorentzVertOffset;
    G4double f3DLorentzVertOffsetErr;
    G4double f3DLorentzCenterX;
    G4double f3DLorentzCenterXErr;
    G4double f3DLorentzCenterY;
    G4double f3DLorentzCenterYErr;
    G4double f3DLorentzChi2red;
    G4double f3DLorentzPp;
    G4int f3DLorentzDOF;
    G4double f3DLorentzChargeErr;
    G4bool f3DLorentzSuccess;

    // Legacy variables that may still be used
    G4bool fIsPixelHit;  // True if hit is on pixel OR distance <= D0
    
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    std::vector<G4double> fNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4double> fNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels

    // Variables for particle information (reduced set)
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
    
    // Scorer data variables
    G4double fScorerEnergyDeposit;  // Energy deposit from Multi-Functional Detector [MeV]
    G4int fScorerHitCount;          // Hit count from Multi-Functional Detector
    G4bool fScorerDataValid;        // Validation flag for scorer data
    
    // Hit purity tracking variables for Multi-Functional Detector validation
    G4bool fPureSiliconHit;         // True if hit is purely in silicon (no aluminum contamination)
    G4bool fAluminumContaminated;   // True if hit has aluminum contamination
    G4bool fChargeCalculationEnabled; // True if charge sharing calculation was enabled
};

#endif