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
    
    // Variables for the branch (edep [MeV], positions [mm])
    void SetEventData(G4double edep, G4double x, G4double y, G4double z);
    
    // Method to set initial particle gun position [mm]
    void SetInitialPosition(G4double x, G4double y, G4double z);
    
    // Method to set nearest pixel position [mm]
    void SetNearestPixelPosition(G4double x, G4double y, G4double z);
    


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
    
    // Fill the ROOT tree with current event data
    void FillTree();

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // =============================================
    // HITS DATA VARIABLES
    // =============================================
    G4double fTrueX;   // True Hit position X [mm]
    G4double fTrueY;   // True Hit position Y [mm]
    G4double fTrueZ;   // True Hit position Z [mm]
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
    G4double fGaussMainDiagDeltaX;
    G4double fGaussMainDiagDeltaY;
    G4double fGaussSecondDiagDeltaX;
    G4double fGaussSecondDiagDeltaY;
    G4double fLorentzRowDeltaX;
    G4double fLorentzColumnDeltaY;
    G4double fLorentzMainDiagDeltaX;
    G4double fLorentzMainDiagDeltaY;
    G4double fLorentzSecondDiagDeltaX;
    G4double fLorentzSecondDiagDeltaY;

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

    // Legacy variables that may still be used
    G4double fPixelZ; // Nearest to hit pixel center Z [mm]
    G4bool fIsPixelHit;  // True if hit is on pixel OR distance <= D0
    
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    std::vector<G4double> fNonPixel_GridNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4double> fNonPixel_GridNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNonPixel_GridNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fNonPixel_GridNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels

    // Variables for particle information (reduced set)
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    G4double fMomentum;             // Particle momentum [MeV/c]
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
};

#endif