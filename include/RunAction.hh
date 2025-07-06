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
    void SetEventData(G4double edep, G4double x, G4double y);
    
    // Method to set initial particle gun position [mm]
    void SetInitialPosition(G4double x, G4double y, G4double z);
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPosition(G4double x, G4double y);
    
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
    
    // Method to set 2D Gaussian fit results (central row and column fitting)
    void Set2DGaussianFitResults(G4double x_center, G4double x_sigma, G4double x_amplitude,
                                G4double x_center_err, G4double x_sigma_err, G4double x_amplitude_err,
                                G4double x_vertical_offset, G4double x_vertical_offset_err,
                                G4double x_chi2red, G4double x_pp, G4int x_dof,
                                G4double y_center, G4double y_sigma, G4double y_amplitude,
                                G4double y_center_err, G4double y_sigma_err, G4double y_amplitude_err,
                                G4double y_vertical_offset, G4double y_vertical_offset_err,
                                G4double y_chi2red, G4double y_pp, G4int y_dof,
                                G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                G4bool fit_successful);
    
    // Method to set diagonal Gaussian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    void SetDiagonalGaussianFitResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_amplitude,
                                      G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_amplitude_err,
                                      G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                      G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                      G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_amplitude,
                                      G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_amplitude_err,
                                      G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                      G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                      G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_amplitude,
                                      G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_amplitude_err,
                                      G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                      G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                      G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_amplitude,
                                      G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_amplitude_err,
                                      G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                      G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                      G4bool fit_successful);
    
    // Method to set 2D Lorentzian fit results (central row and column fitting)
    void Set2DLorentzianFitResults(G4double x_center, G4double x_gamma, G4double x_amplitude,
                                  G4double x_center_err, G4double x_gamma_err, G4double x_amplitude_err,
                                  G4double x_vertical_offset, G4double x_vertical_offset_err,
                                  G4double x_chi2red, G4double x_pp, G4int x_dof,
                                  G4double y_center, G4double y_gamma, G4double y_amplitude,
                                  G4double y_center_err, G4double y_gamma_err, G4double y_amplitude_err,
                                  G4double y_vertical_offset, G4double y_vertical_offset_err,
                                  G4double y_chi2red, G4double y_pp, G4int y_dof,
                                  G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                  G4bool fit_successful);
    

    
    // Method to set diagonal Lorentzian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    void SetDiagonalLorentzianFitResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_amplitude,
                                        G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_amplitude_err,
                                        G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                        G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                        G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_amplitude,
                                        G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_amplitude_err,
                                        G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                        G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                        G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_amplitude,
                                        G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_amplitude_err,
                                        G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                        G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                        G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_amplitude,
                                        G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_amplitude_err,
                                        G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                        G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                        G4bool fit_successful);

    // Method to set 2D Power-Law Lorentzian fit results (central row and column fitting)
    // Model: y(x) = A / (1 + ((x-m)/gamma)^2)^beta + B
    void Set2DPowerLorentzianFitResults(G4double x_center, G4double x_gamma, G4double x_beta, G4double x_amplitude,
                                        G4double x_center_err, G4double x_gamma_err, G4double x_beta_err, G4double x_amplitude_err,
                                        G4double x_vertical_offset, G4double x_vertical_offset_err,
                                        G4double x_chi2red, G4double x_pp, G4int x_dof,
                                        G4double y_center, G4double y_gamma, G4double y_beta, G4double y_amplitude,
                                        G4double y_center_err, G4double y_gamma_err, G4double y_beta_err, G4double y_amplitude_err,
                                        G4double y_vertical_offset, G4double y_vertical_offset_err,
                                        G4double y_chi2red, G4double y_pp, G4int y_dof,
                                        G4double x_charge_uncertainty, G4double y_charge_uncertainty,
                                        G4bool fit_successful);
    
    // Method to set diagonal Power-Law Lorentzian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    // Model: y(x) = A / (1 + ((x-m)/gamma)^2)^beta + B
    void SetDiagonalPowerLorentzianFitResults(G4double main_diag_x_center, G4double main_diag_x_gamma, G4double main_diag_x_beta, G4double main_diag_x_amplitude,
                                              G4double main_diag_x_center_err, G4double main_diag_x_gamma_err, G4double main_diag_x_beta_err, G4double main_diag_x_amplitude_err,
                                              G4double main_diag_x_vertical_offset, G4double main_diag_x_vertical_offset_err,
                                              G4double main_diag_x_chi2red, G4double main_diag_x_pp, G4int main_diag_x_dof, G4bool main_diag_x_fit_successful,
                                              G4double main_diag_y_center, G4double main_diag_y_gamma, G4double main_diag_y_beta, G4double main_diag_y_amplitude,
                                              G4double main_diag_y_center_err, G4double main_diag_y_gamma_err, G4double main_diag_y_beta_err, G4double main_diag_y_amplitude_err,
                                              G4double main_diag_y_vertical_offset, G4double main_diag_y_vertical_offset_err,
                                              G4double main_diag_y_chi2red, G4double main_diag_y_pp, G4int main_diag_y_dof, G4bool main_diag_y_fit_successful,
                                              G4double sec_diag_x_center, G4double sec_diag_x_gamma, G4double sec_diag_x_beta, G4double sec_diag_x_amplitude,
                                              G4double sec_diag_x_center_err, G4double sec_diag_x_gamma_err, G4double sec_diag_x_beta_err, G4double sec_diag_x_amplitude_err,
                                              G4double sec_diag_x_vertical_offset, G4double sec_diag_x_vertical_offset_err,
                                              G4double sec_diag_x_chi2red, G4double sec_diag_x_pp, G4int sec_diag_x_dof, G4bool sec_diag_x_fit_successful,
                                              G4double sec_diag_y_center, G4double sec_diag_y_gamma, G4double sec_diag_y_beta, G4double sec_diag_y_amplitude,
                                              G4double sec_diag_y_center_err, G4double sec_diag_y_gamma_err, G4double sec_diag_y_beta_err, G4double sec_diag_y_amplitude_err,
                                              G4double sec_diag_y_vertical_offset, G4double sec_diag_y_vertical_offset_err,
                                              G4double sec_diag_y_chi2red, G4double sec_diag_y_pp, G4int sec_diag_y_dof, G4bool sec_diag_y_fit_successful,
                                              G4bool fit_successful);

    // =============================================
    // 3D FITTING RESULTS SETTER METHODS
    // =============================================
    
    // Method to set 3D Lorentzian fit results (entire neighborhood surface fitting)
    // Model: z(x,y) = A / (1 + ((x-mx)/γx)^2 + ((y-my)/γy)^2) + B
    void Set3DLorentzianFitResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double amplitude, G4double vertical_offset,
                                   G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double amplitude_err, G4double vertical_offset_err,
                                   G4double chi2red, G4double pp, G4int dof,
                                   G4double charge_uncertainty,
                                   G4bool fit_successful);
    
    // Method to set 3D Gaussian fit results (entire neighborhood surface fitting)
    // Model: z(x,y) = A * exp(-((x-mx)^2/(2*σx^2) + (y-my)^2/(2*σy^2))) + B
    void Set3DGaussianFitResults(G4double center_x, G4double center_y, G4double sigma_x, G4double sigma_y, G4double amplitude, G4double vertical_offset,
                                 G4double center_x_err, G4double center_y_err, G4double sigma_x_err, G4double sigma_y_err, G4double amplitude_err, G4double vertical_offset_err,
                                 G4double chi2red, G4double pp, G4int dof,
                                 G4double charge_uncertainty,
                                 G4bool fit_successful);
    
    // Method to set 3D Power-Law Lorentzian fit results (entire neighborhood surface fitting)
    // Model: z(x,y) = A / (1 + ((x-mx)/γx)^2 + ((y-my)/γy)^2)^β + B
    void Set3DPowerLorentzianFitResults(G4double center_x, G4double center_y, G4double gamma_x, G4double gamma_y, G4double beta, G4double amplitude, G4double vertical_offset,
                                        G4double center_x_err, G4double center_y_err, G4double gamma_x_err, G4double gamma_y_err, G4double beta_err, G4double amplitude_err, G4double vertical_offset_err,
                                        G4double chi2red, G4double pp, G4int dof,
                                        G4double charge_uncertainty,
                                        G4bool fit_successful);
    
    // Fill the ROOT tree with current event data
    void FillTree();
    
    // Method to store automatic radius selection results
    void SetAutoRadiusResults(G4int selectedRadius);

private:
    // =============================================
    // COORDINATE TRANSFORMATION HELPER METHODS
    // =============================================
    
    // Apply rotation matrix transformation for diagonal coordinates
    void TransformDiagonalCoordinates(G4double x_prime, G4double y_prime, G4double theta_deg, 
                                      G4double& x_transformed, G4double& y_transformed);
    
    // Calculate transformed coordinates for all diagonal fits
    void CalculateTransformedDiagonalCoordinates();
    
    // Calculate mean estimations from all fitting methods
    void CalculateMeanEstimations();
    
    // Helper functions to organize branch creation
    void CreateHitsBranches();
    void CreateGaussianFitBranches();
    void CreateLorentzianFitBranches();
    void CreatePowerLorentzianFitBranches();
    void Create3DFitBranches();
    void CreateGridNeighborhoodBranches();
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
    
    // =============================================
    // HITS DATA VARIABLES
    // =============================================
    G4double fTrueX;   // True Hit position X [mm]
    G4double fTrueY;   // True Hit position Y [mm]
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fEdep;   // Energy deposit [MeV]
    G4double fPixelTrueDeltaX; // Delta X from pixel center to true position [mm] (x_pixel - x_true)
    G4double fPixelTrueDeltaY; // Delta Y from pixel center to true position [mm] (y_pixel - y_true)
    
    // Delta variables for estimations vs true position
    G4double fGaussRowDeltaX;
    G4double fGaussColumnDeltaY;
    G4double fLorentzRowDeltaX;
    G4double fLorentzColumnDeltaY;
    G4double fPowerLorentzRowDeltaX;
    G4double fPowerLorentzColumnDeltaY;
    
    // 3D fitting delta variables
    G4double f3DLorentzianDeltaX;          // Delta X from 3D Lorentzian fit to True Position [mm]
    G4double f3DLorentzianDeltaY;          // Delta Y from 3D Lorentzian fit to True Position [mm]
    G4double f3DGaussianDeltaX;            // Delta X from 3D Gaussian fit to True Position [mm]
    G4double f3DGaussianDeltaY;            // Delta Y from 3D Gaussian fit to True Position [mm]
    G4double f3DPowerLorentzianDeltaX;     // Delta X from 3D Power-Law Lorentzian fit to True Position [mm]
    G4double f3DPowerLorentzianDeltaY;     // Delta Y from 3D Power-Law Lorentzian fit to True Position [mm]

    // =============================================
    // TRANSFORMED DIAGONAL COORDINATES (ROTATION MATRIX)
    // =============================================
    
    // Transformed main diagonal coordinates (θ = 45°)
    G4double fGaussMainDiagTransformedX;      // Transformed X from main diagonal (x' -> x)
    G4double fGaussMainDiagTransformedY;      // Transformed Y from main diagonal (y' -> y)
    G4double fLorentzMainDiagTransformedX;    // Transformed X from main diagonal (x' -> x)
    G4double fLorentzMainDiagTransformedY;    // Transformed Y from main diagonal (y' -> y)
    
    // Transformed secondary diagonal coordinates (θ = -45°)
    G4double fGaussSecondDiagTransformedX;    // Transformed X from secondary diagonal (x' -> x)
    G4double fGaussSecondDiagTransformedY;    // Transformed Y from secondary diagonal (y' -> y)
    G4double fLorentzSecondDiagTransformedX;  // Transformed X from secondary diagonal (x' -> x)
    G4double fLorentzSecondDiagTransformedY;  // Transformed Y from secondary diagonal (y' -> y)
    G4double fPowerLorentzMainDiagTransformedX;    // Transformed X from Power-Law Lorentzian main diagonal (x' -> x)
    G4double fPowerLorentzMainDiagTransformedY;    // Transformed Y from Power-Law Lorentzian main diagonal (y' -> y)
    G4double fPowerLorentzSecondDiagTransformedX;  // Transformed X from Power-Law Lorentzian secondary diagonal (x' -> x)
    G4double fPowerLorentzSecondDiagTransformedY;  // Transformed Y from Power-Law Lorentzian secondary diagonal (y' -> y)
    
    // Delta values for transformed coordinates vs true position
    G4double fGaussMainDiagTransformedDeltaX;   // x_transformed - x_true (main diagonal)
    G4double fGaussMainDiagTransformedDeltaY;   // y_transformed - y_true (main diagonal)
    G4double fGaussSecondDiagTransformedDeltaX; // x_transformed - x_true (secondary diagonal)
    G4double fGaussSecondDiagTransformedDeltaY; // y_transformed - y_true (secondary diagonal)
    G4double fLorentzMainDiagTransformedDeltaX;   // x_transformed - x_true (main diagonal)
    G4double fLorentzMainDiagTransformedDeltaY;   // y_transformed - y_true (main diagonal)
    G4double fLorentzSecondDiagTransformedDeltaX; // x_transformed - x_true (secondary diagonal)
    G4double fLorentzSecondDiagTransformedDeltaY; // y_transformed - y_true (secondary diagonal)
    G4double fPowerLorentzMainDiagTransformedDeltaX;   // x_transformed - x_true (Power-Law Lorentzian main diagonal)
    G4double fPowerLorentzMainDiagTransformedDeltaY;   // y_transformed - y_true (Power-Law Lorentzian main diagonal)
    G4double fPowerLorentzSecondDiagTransformedDeltaX; // x_transformed - x_true (Power-Law Lorentzian secondary diagonal)
    G4double fPowerLorentzSecondDiagTransformedDeltaY; // y_transformed - y_true (Power-Law Lorentzian secondary diagonal)

    // =============================================
    // MEAN ESTIMATIONS FROM ALL FITTING METHODS
    // =============================================
    
    // Mean of all X coordinate estimations (row, transformed diagonals)
    G4double fGaussMeanTrueDeltaX;   // Mean delta X from all Gaussian estimation methods to True Position [mm]
    G4double fGaussMeanTrueDeltaY;   // Mean delta Y from all Gaussian estimation methods to True Position [mm]
    G4double fLorentzMeanTrueDeltaX; // Mean delta X from all Lorentzian estimation methods to True Position [mm]
    G4double fLorentzMeanTrueDeltaY; // Mean delta Y from all Lorentzian estimation methods to True Position [mm]
    G4double fPowerLorentzMeanTrueDeltaX; // Mean delta X from all Power-Law Lorentzian estimation methods to True Position [mm]
    G4double fPowerLorentzMeanTrueDeltaY; // Mean delta Y from all Power-Law Lorentzian estimation methods to True Position [mm]

    // =============================================
    // AUTOMATIC RADIUS SELECTION VARIABLES
    // =============================================
    
    G4int fSelectedRadius;          // Automatically selected radius for this event

    // =============================================
    // GAUSSIAN FITS VARIABLES
    // =============================================
    
    // GaussFitRow/GaussFitRowX
    G4double fGaussFitRowAmplitude;
    G4double fGaussFitRowAmplitudeErr;
    G4double fGaussFitRowStdev;
    G4double fGaussFitRowStdevErr;
    G4double fGaussFitRowVerticalOffset;
    G4double fGaussFitRowVerticalOffsetErr;
    G4double fGaussFitRowCenter;
    G4double fGaussFitRowCenterErr;
    G4double fGaussFitRowChi2red;
    G4double fGaussFitRowPp;
    G4int fGaussFitRowDOF;
    
    // Charge uncertainties (5% of max charge for Gaussian fits)
    G4double fGaussFitRowChargeUncertainty;     // Row charge uncertainty (5% of max charge)
    
    // GaussFitColumn/GaussFitColumnY
    G4double fGaussFitColumnAmplitude;
    G4double fGaussFitColumnAmplitudeErr;
    G4double fGaussFitColumnStdev;
    G4double fGaussFitColumnStdevErr;
    G4double fGaussFitColumnVerticalOffset;
    G4double fGaussFitColumnVerticalOffsetErr;
    G4double fGaussFitColumnCenter;
    G4double fGaussFitColumnCenterErr;
    G4double fGaussFitColumnChi2red;
    G4double fGaussFitColumnPp;
    G4int fGaussFitColumnDOF;
    
    // Charge uncertainty for Gaussian column fit
    G4double fGaussFitColumnChargeUncertainty;  // Column charge uncertainty (5% of max charge)
    
    // GaussFitMainDiag/GaussFitMainDiagX
    G4double fGaussFitMainDiagXAmplitude;
    G4double fGaussFitMainDiagXAmplitudeErr;
    G4double fGaussFitMainDiagXStdev;
    G4double fGaussFitMainDiagXStdevErr;
    G4double fGaussFitMainDiagXVerticalOffset;
    G4double fGaussFitMainDiagXVerticalOffsetErr;
    G4double fGaussFitMainDiagXCenter;
    G4double fGaussFitMainDiagXCenterErr;
    G4double fGaussFitMainDiagXChi2red;
    G4double fGaussFitMainDiagXPp;
    G4int fGaussFitMainDiagXDOF;
    
    // GaussFitMainDiag/GaussFitMainDiagY
    G4double fGaussFitMainDiagYAmplitude;
    G4double fGaussFitMainDiagYAmplitudeErr;
    G4double fGaussFitMainDiagYStdev;
    G4double fGaussFitMainDiagYStdevErr;
    G4double fGaussFitMainDiagYVerticalOffset;
    G4double fGaussFitMainDiagYVerticalOffsetErr;
    G4double fGaussFitMainDiagYCenter;
    G4double fGaussFitMainDiagYCenterErr;
    G4double fGaussFitMainDiagYChi2red;
    G4double fGaussFitMainDiagYPp;
    G4int fGaussFitMainDiagYDOF;
    
    // GaussFitSecondDiag/GaussFitSecondDiagX
    G4double fGaussFitSecondDiagXAmplitude;
    G4double fGaussFitSecondDiagXAmplitudeErr;
    G4double fGaussFitSecondDiagXStdev;
    G4double fGaussFitSecondDiagXStdevErr;
    G4double fGaussFitSecondDiagXVerticalOffset;
    G4double fGaussFitSecondDiagXVerticalOffsetErr;
    G4double fGaussFitSecondDiagXCenter;
    G4double fGaussFitSecondDiagXCenterErr;
    G4double fGaussFitSecondDiagXChi2red;
    G4double fGaussFitSecondDiagXPp;
    G4int fGaussFitSecondDiagXDOF;
    
    // GaussFitSecondDiag/GaussFitSecondDiagY
    G4double fGaussFitSecondDiagYAmplitude;
    G4double fGaussFitSecondDiagYAmplitudeErr;
    G4double fGaussFitSecondDiagYStdev;
    G4double fGaussFitSecondDiagYStdevErr;
    G4double fGaussFitSecondDiagYVerticalOffset;
    G4double fGaussFitSecondDiagYVerticalOffsetErr;
    G4double fGaussFitSecondDiagYCenter;
    G4double fGaussFitSecondDiagYCenterErr;
    G4double fGaussFitSecondDiagYChi2red;
    G4double fGaussFitSecondDiagYPp;
    G4int fGaussFitSecondDiagYDOF;

    // =============================================
    // LORENTZIAN FITS VARIABLES
    // =============================================
    
    // LorentzFitRow/LorentzFitRowX
    G4double fLorentzFitRowAmplitude;
    G4double fLorentzFitRowAmplitudeErr;
    G4double fLorentzFitRowGamma;
    G4double fLorentzFitRowGammaErr;
    G4double fLorentzFitRowVerticalOffset;
    G4double fLorentzFitRowVerticalOffsetErr;
    G4double fLorentzFitRowCenter;
    G4double fLorentzFitRowCenterErr;
    G4double fLorentzFitRowChi2red;
    G4double fLorentzFitRowPp;
    G4int fLorentzFitRowDOF;
    
    // Charge uncertainty for Lorentzian row fit  
    G4double fLorentzFitRowChargeUncertainty;   // Row charge uncertainty (5% of max charge)
    
    // LorentzFitColumn/LorentzFitColumnY
    G4double fLorentzFitColumnAmplitude;
    G4double fLorentzFitColumnAmplitudeErr;
    G4double fLorentzFitColumnGamma;
    G4double fLorentzFitColumnGammaErr;
    G4double fLorentzFitColumnVerticalOffset;
    G4double fLorentzFitColumnVerticalOffsetErr;
    G4double fLorentzFitColumnCenter;
    G4double fLorentzFitColumnCenterErr;
    G4double fLorentzFitColumnChi2red;
    G4double fLorentzFitColumnPp;
    G4int fLorentzFitColumnDOF;
    
    // Charge uncertainty for Lorentzian column fit
    G4double fLorentzFitColumnChargeUncertainty; // Column charge uncertainty (5% of max charge)
    
    // LorentzFitMainDiag/LorentzFitMainDiagX
    G4double fLorentzFitMainDiagXAmplitude;
    G4double fLorentzFitMainDiagXAmplitudeErr;
    G4double fLorentzFitMainDiagXGamma;
    G4double fLorentzFitMainDiagXGammaErr;
    G4double fLorentzFitMainDiagXVerticalOffset;
    G4double fLorentzFitMainDiagXVerticalOffsetErr;
    G4double fLorentzFitMainDiagXCenter;
    G4double fLorentzFitMainDiagXCenterErr;
    G4double fLorentzFitMainDiagXChi2red;
    G4double fLorentzFitMainDiagXPp;
    G4int fLorentzFitMainDiagXDOF;
    
    // LorentzFitMainDiag/LorentzFitMainDiagY
    G4double fLorentzFitMainDiagYAmplitude;
    G4double fLorentzFitMainDiagYAmplitudeErr;
    G4double fLorentzFitMainDiagYGamma;
    G4double fLorentzFitMainDiagYGammaErr;
    G4double fLorentzFitMainDiagYVerticalOffset;
    G4double fLorentzFitMainDiagYVerticalOffsetErr;
    G4double fLorentzFitMainDiagYCenter;
    G4double fLorentzFitMainDiagYCenterErr;
    G4double fLorentzFitMainDiagYChi2red;
    G4double fLorentzFitMainDiagYPp;
    G4int fLorentzFitMainDiagYDOF;
    
    // LorentzFitSecondDiag/LorentzFitSecondDiagX
    G4double fLorentzFitSecondDiagXAmplitude;
    G4double fLorentzFitSecondDiagXAmplitudeErr;
    G4double fLorentzFitSecondDiagXGamma;
    G4double fLorentzFitSecondDiagXGammaErr;
    G4double fLorentzFitSecondDiagXVerticalOffset;
    G4double fLorentzFitSecondDiagXVerticalOffsetErr;
    G4double fLorentzFitSecondDiagXCenter;
    G4double fLorentzFitSecondDiagXCenterErr;
    G4double fLorentzFitSecondDiagXChi2red;
    G4double fLorentzFitSecondDiagXPp;
    G4int fLorentzFitSecondDiagXDOF;
    
    // LorentzFitSecondDiag/LorentzFitSecondDiagY
    G4double fLorentzFitSecondDiagYAmplitude;
    G4double fLorentzFitSecondDiagYAmplitudeErr;
    G4double fLorentzFitSecondDiagYGamma;
    G4double fLorentzFitSecondDiagYGammaErr;
    G4double fLorentzFitSecondDiagYVerticalOffset;
    G4double fLorentzFitSecondDiagYVerticalOffsetErr;
    G4double fLorentzFitSecondDiagYCenter;
    G4double fLorentzFitSecondDiagYCenterErr;
    G4double fLorentzFitSecondDiagYChi2red;
    G4double fLorentzFitSecondDiagYPp;
    G4int fLorentzFitSecondDiagYDOF;

    // =============================================
    // POWER-LAW LORENTZIAN FITS VARIABLES
    // =============================================
    
    // PowerLorentzFitRow/PowerLorentzFitRowX
    G4double fPowerLorentzFitRowAmplitude;
    G4double fPowerLorentzFitRowAmplitudeErr;
    G4double fPowerLorentzFitRowBeta;
    G4double fPowerLorentzFitRowBetaErr;
    G4double fPowerLorentzFitRowGamma;
    G4double fPowerLorentzFitRowGammaErr;
    G4double fPowerLorentzFitRowVerticalOffset;
    G4double fPowerLorentzFitRowVerticalOffsetErr;
    G4double fPowerLorentzFitRowCenter;
    G4double fPowerLorentzFitRowCenterErr;
    G4double fPowerLorentzFitRowChi2red;
    G4double fPowerLorentzFitRowPp;
    G4int fPowerLorentzFitRowDOF;
    
    // PowerLorentzFitColumn/PowerLorentzFitColumnY
    G4double fPowerLorentzFitColumnAmplitude;
    G4double fPowerLorentzFitColumnAmplitudeErr;
    G4double fPowerLorentzFitColumnBeta;
    G4double fPowerLorentzFitColumnBetaErr;
    G4double fPowerLorentzFitColumnGamma;
    G4double fPowerLorentzFitColumnGammaErr;
    G4double fPowerLorentzFitColumnVerticalOffset;
    G4double fPowerLorentzFitColumnVerticalOffsetErr;
    G4double fPowerLorentzFitColumnCenter;
    G4double fPowerLorentzFitColumnCenterErr;
    G4double fPowerLorentzFitColumnChi2red;
    G4double fPowerLorentzFitColumnPp;
    G4int fPowerLorentzFitColumnDOF;
    
    // Charge uncertainties for Power-Law Lorentzian fits (5% of max charge)
    G4double fPowerLorentzFitRowChargeUncertainty;    // Row charge uncertainty (5% of max charge)
    G4double fPowerLorentzFitColumnChargeUncertainty; // Column charge uncertainty (5% of max charge)
    
    // PowerLorentzFitMainDiag/PowerLorentzFitMainDiagX
    G4double fPowerLorentzFitMainDiagXAmplitude;
    G4double fPowerLorentzFitMainDiagXAmplitudeErr;
    G4double fPowerLorentzFitMainDiagXBeta;
    G4double fPowerLorentzFitMainDiagXBetaErr;
    G4double fPowerLorentzFitMainDiagXGamma;
    G4double fPowerLorentzFitMainDiagXGammaErr;
    G4double fPowerLorentzFitMainDiagXVerticalOffset;
    G4double fPowerLorentzFitMainDiagXVerticalOffsetErr;
    G4double fPowerLorentzFitMainDiagXCenter;
    G4double fPowerLorentzFitMainDiagXCenterErr;
    G4double fPowerLorentzFitMainDiagXChi2red;
    G4double fPowerLorentzFitMainDiagXPp;
    G4int fPowerLorentzFitMainDiagXDOF;
    
    // PowerLorentzFitMainDiag/PowerLorentzFitMainDiagY
    G4double fPowerLorentzFitMainDiagYAmplitude;
    G4double fPowerLorentzFitMainDiagYAmplitudeErr;
    G4double fPowerLorentzFitMainDiagYBeta;
    G4double fPowerLorentzFitMainDiagYBetaErr;
    G4double fPowerLorentzFitMainDiagYGamma;
    G4double fPowerLorentzFitMainDiagYGammaErr;
    G4double fPowerLorentzFitMainDiagYVerticalOffset;
    G4double fPowerLorentzFitMainDiagYVerticalOffsetErr;
    G4double fPowerLorentzFitMainDiagYCenter;
    G4double fPowerLorentzFitMainDiagYCenterErr;
    G4double fPowerLorentzFitMainDiagYChi2red;
    G4double fPowerLorentzFitMainDiagYPp;
    G4int fPowerLorentzFitMainDiagYDOF;
    
    // PowerLorentzFitSecondDiag/PowerLorentzFitSecondDiagX
    G4double fPowerLorentzFitSecondDiagXAmplitude;
    G4double fPowerLorentzFitSecondDiagXAmplitudeErr;
    G4double fPowerLorentzFitSecondDiagXBeta;
    G4double fPowerLorentzFitSecondDiagXBetaErr;
    G4double fPowerLorentzFitSecondDiagXGamma;
    G4double fPowerLorentzFitSecondDiagXGammaErr;
    G4double fPowerLorentzFitSecondDiagXVerticalOffset;
    G4double fPowerLorentzFitSecondDiagXVerticalOffsetErr;
    G4double fPowerLorentzFitSecondDiagXCenter;
    G4double fPowerLorentzFitSecondDiagXCenterErr;
    G4double fPowerLorentzFitSecondDiagXChi2red;
    G4double fPowerLorentzFitSecondDiagXPp;
    G4int fPowerLorentzFitSecondDiagXDOF;
    
    // PowerLorentzFitSecondDiag/PowerLorentzFitSecondDiagY
    G4double fPowerLorentzFitSecondDiagYAmplitude;
    G4double fPowerLorentzFitSecondDiagYAmplitudeErr;
    G4double fPowerLorentzFitSecondDiagYBeta;
    G4double fPowerLorentzFitSecondDiagYBetaErr;
    G4double fPowerLorentzFitSecondDiagYGamma;
    G4double fPowerLorentzFitSecondDiagYGammaErr;
    G4double fPowerLorentzFitSecondDiagYVerticalOffset;
    G4double fPowerLorentzFitSecondDiagYVerticalOffsetErr;
    G4double fPowerLorentzFitSecondDiagYCenter;
    G4double fPowerLorentzFitSecondDiagYCenterErr;
    G4double fPowerLorentzFitSecondDiagYChi2red;
    G4double fPowerLorentzFitSecondDiagYPp;
    G4int fPowerLorentzFitSecondDiagYDOF;

    // =============================================
    // 3D LORENTZIAN FITS VARIABLES
    // =============================================
    
    // 3D Lorentzian fit parameters: z(x,y) = A / (1 + ((x-mx)/γx)^2 + ((y-my)/γy)^2) + B
    G4double f3DLorentzianFitCenterX;            // mx parameter (X center)
    G4double f3DLorentzianFitCenterY;            // my parameter (Y center)
    G4double f3DLorentzianFitGammaX;             // γx parameter (X width/HWHM)
    G4double f3DLorentzianFitGammaY;             // γy parameter (Y width/HWHM)
    G4double f3DLorentzianFitAmplitude;          // A parameter (peak amplitude)
    G4double f3DLorentzianFitVerticalOffset;     // B parameter (baseline)
    
    // 3D Lorentzian fit parameter errors
    G4double f3DLorentzianFitCenterXErr;         
    G4double f3DLorentzianFitCenterYErr;         
    G4double f3DLorentzianFitGammaXErr;          
    G4double f3DLorentzianFitGammaYErr;          
    G4double f3DLorentzianFitAmplitudeErr;       
    G4double f3DLorentzianFitVerticalOffsetErr;  
    
    // 3D Lorentzian fit statistics
    G4double f3DLorentzianFitChi2red;            // Reduced Chi-squared
    G4double f3DLorentzianFitPp;                 // P-value  
    G4int f3DLorentzianFitDOF;                   // Degrees of Freedom
    G4double f3DLorentzianFitChargeUncertainty;  // Charge uncertainty (5% of max charge)
    G4bool f3DLorentzianFitSuccessful;           // Fit success flag

    // =============================================
    // 3D GAUSSIAN FITS VARIABLES
    // =============================================
    
    // 3D Gaussian fit parameters: z(x,y) = A * exp(-((x-mx)^2/(2*σx^2) + (y-my)^2/(2*σy^2))) + B
    G4double f3DGaussianFitCenterX;            // mx parameter (X center)
    G4double f3DGaussianFitCenterY;            // my parameter (Y center)
    G4double f3DGaussianFitSigmaX;             // σx parameter (X standard deviation)
    G4double f3DGaussianFitSigmaY;             // σy parameter (Y standard deviation)
    G4double f3DGaussianFitAmplitude;          // A parameter (peak amplitude)
    G4double f3DGaussianFitVerticalOffset;     // B parameter (baseline)
    
    // 3D Gaussian fit parameter errors
    G4double f3DGaussianFitCenterXErr;         
    G4double f3DGaussianFitCenterYErr;         
    G4double f3DGaussianFitSigmaXErr;          
    G4double f3DGaussianFitSigmaYErr;          
    G4double f3DGaussianFitAmplitudeErr;       
    G4double f3DGaussianFitVerticalOffsetErr;  
    
    // 3D Gaussian fit statistics
    G4double f3DGaussianFitChi2red;            // Reduced Chi-squared
    G4double f3DGaussianFitPp;                 // P-value  
    G4int f3DGaussianFitDOF;                   // Degrees of Freedom
    G4double f3DGaussianFitChargeUncertainty;  // Charge uncertainty (5% of max charge)
    G4bool f3DGaussianFitSuccessful;           // Fit success flag

    // =============================================
    // 3D POWER-LAW LORENTZIAN FITS VARIABLES
    // =============================================
    
    // 3D Power-Law Lorentzian fit parameters: z(x,y) = A / (1 + ((x-mx)/γx)^2 + ((y-my)/γy)^2)^β + B
    G4double f3DPowerLorentzianFitCenterX;            // mx parameter (X center)
    G4double f3DPowerLorentzianFitCenterY;            // my parameter (Y center)
    G4double f3DPowerLorentzianFitGammaX;             // γx parameter (X width)
    G4double f3DPowerLorentzianFitGammaY;             // γy parameter (Y width)
    G4double f3DPowerLorentzianFitBeta;               // β parameter (power exponent)  
    G4double f3DPowerLorentzianFitAmplitude;          // A parameter (peak amplitude)
    G4double f3DPowerLorentzianFitVerticalOffset;     // B parameter (baseline)
    
    // 3D Power-Law Lorentzian fit parameter errors
    G4double f3DPowerLorentzianFitCenterXErr;         
    G4double f3DPowerLorentzianFitCenterYErr;         
    G4double f3DPowerLorentzianFitGammaXErr;          
    G4double f3DPowerLorentzianFitGammaYErr;          
    G4double f3DPowerLorentzianFitBetaErr;            
    G4double f3DPowerLorentzianFitAmplitudeErr;       
    G4double f3DPowerLorentzianFitVerticalOffsetErr;  
    
    // 3D Power-Law Lorentzian fit statistics
    G4double f3DPowerLorentzianFitChi2red;            // Reduced Chi-squared
    G4double f3DPowerLorentzianFitPp;                 // P-value  
    G4int f3DPowerLorentzianFitDOF;                   // Degrees of Freedom
    G4double f3DPowerLorentzianFitChargeUncertainty;  // Charge uncertainty (5% of max charge)
    G4bool f3DPowerLorentzianFitSuccessful;           // Fit success flag

    // Legacy variables that may still be used
    G4bool fIsPixelHit;  // True if hit is on pixel OR distance <= D0
    
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    std::vector<G4double> fNonPixel_GridNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4double> fNonPixel_GridNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNonPixel_GridNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fNonPixel_GridNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels

    // Variables for particle information (reduced set)
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
};

#endif