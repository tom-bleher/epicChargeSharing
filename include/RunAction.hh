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
    
    // Method to set pixel indices and distance
    void SetPixelIndices(G4int i, G4int j, G4double distance);
    
    // Method to set pixel alpha angle
    void SetPixelAlpha(G4double alpha);

    // Method to set pixel hit status
    void SetPixelHitStatus(G4bool isPixelHit);
    
    // Method to set pixel classification data (hit status and delta values)
    void SetPixelClassification(G4bool isWithinD0, G4double pixelTrueDeltaX, G4double pixelTrueDeltaY);
    
    // Method to set neighborhood (9x9) grid angle data for non-pixel hits
    void SetNeighborhoodGridData(const std::vector<G4double>& angles, 
                        const std::vector<G4int>& pixelI, 
                        const std::vector<G4int>& pixelJ);
    
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
                                G4double x_chi2red, G4int x_npoints,
                                G4double y_center, G4double y_sigma, G4double y_amplitude,
                                G4double y_center_err, G4double y_sigma_err, G4double y_amplitude_err,
                                G4double y_chi2red, G4int y_npoints,
                                G4bool fit_successful);
    
    // Method to set diagonal Gaussian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    void SetDiagonalGaussianFitResults(G4double main_diag_x_center, G4double main_diag_x_sigma, G4double main_diag_x_amplitude,
                                      G4double main_diag_x_center_err, G4double main_diag_x_sigma_err, G4double main_diag_x_amplitude_err,
                                      G4double main_diag_x_chi2red, G4int main_diag_x_npoints, G4bool main_diag_x_fit_successful,
                                      G4double main_diag_y_center, G4double main_diag_y_sigma, G4double main_diag_y_amplitude,
                                      G4double main_diag_y_center_err, G4double main_diag_y_sigma_err, G4double main_diag_y_amplitude_err,
                                      G4double main_diag_y_chi2red, G4int main_diag_y_npoints, G4bool main_diag_y_fit_successful,
                                      G4double sec_diag_x_center, G4double sec_diag_x_sigma, G4double sec_diag_x_amplitude,
                                      G4double sec_diag_x_center_err, G4double sec_diag_x_sigma_err, G4double sec_diag_x_amplitude_err,
                                      G4double sec_diag_x_chi2red, G4int sec_diag_x_npoints, G4bool sec_diag_x_fit_successful,
                                      G4double sec_diag_y_center, G4double sec_diag_y_sigma, G4double sec_diag_y_amplitude,
                                      G4double sec_diag_y_center_err, G4double sec_diag_y_sigma_err, G4double sec_diag_y_amplitude_err,
                                      G4double sec_diag_y_chi2red, G4int sec_diag_y_npoints, G4bool sec_diag_y_fit_successful,
                                      G4bool fit_successful);
    
    // Fill the ROOT tree with current event data
    void FillTree();

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // =============================================
    // COMMON VARIABLES (stored for all events)
    // =============================================
    G4double fEdep;   // Energy deposit [MeV]
    G4double fTrueX;   // True Hit position X [mm]
    G4double fTrueY;   // True Hit position Y [mm]
    G4double fTrueZ;   // True Hit position Z [mm]
    
    // Variables for initial particle gun position
    G4double fInitX;  // Initial X [mm]
    G4double fInitY;  // Initial Y [mm]
    G4double fInitZ;  // Initial Z [mm]
    
    // Variables for nearest pixel center position
    G4double fPixelX; // Nearest to hit pixel center X [mm]
    G4double fPixelY; // Nearest to hit pixel center Y [mm]
    G4double fPixelZ; // Nearest to hit pixel center Z [mm]
    
    // Variables for pixel mapping
    G4int fPixelI;    // Pixel index in X direction
    G4int fPixelJ;    // Pixel index in Y direction
    G4double fPixelTrueDeltaX; // Delta X from hit to pixel center [mm] (x_pixel - x_true)
    G4double fPixelTrueDeltaY; // Delta Y from hit to pixel center [mm] (y_pixel - y_true)
    
    // =============================================
    // HIT CLASSIFICATION VARIABLES
    // =============================================
    G4bool fIsPixelHit;  // True if hit is on pixel OR distance <= D0

    // =============================================
    // PIXEL HIT DATA (distance <= D0 or on pixel)
    // =============================================
    G4double fPixelHit_PixelAlpha; // Angular size of pixel from hit position [deg]
    
    // =============================================
    // NON-PIXEL HIT DATA (distance > D0 and not on pixel)
    // =============================================
    
    // Variables for neighborhood (9x9) grid angle data
    std::vector<G4double> fNonPixel_GridNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4int> fNonPixel_GridNeighborhoodPixelI;     // I indices of neighborhood grid pixels
    std::vector<G4int> fNonPixel_GridNeighborhoodPixelJ;     // J indices of neighborhood grid pixels
    
    // Variables for neighborhood (9x9) grid charge sharing data
    std::vector<G4double> fNonPixel_GridNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fNonPixel_GridNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fNonPixel_GridNeighborhoodCharge;       // Charge values in Coulombs for neighborhood grid pixels
    
    // Variables for 3D Gaussian fit results
    G4double fNonPixel_FitAmplitude;         // Fitted amplitude
    G4double fNonPixel_FitX0;               // Fitted X center [mm]
    G4double fNonPixel_FitY0;               // Fitted Y center [mm]
    G4double fNonPixel_FitSigmaX;           // Fitted sigma X [mm]
    G4double fNonPixel_FitSigmaY;           // Fitted sigma Y [mm]
    G4double fNonPixel_FitTheta;            // Fitted rotation angle [rad]
    G4double fNonPixel_FitOffset;           // Fitted offset
    
    G4double fNonPixel_FitAmplitudeErr;     // Error in amplitude
    G4double fNonPixel_FitX0Err;           // Error in X center [mm]
    G4double fNonPixel_FitY0Err;           // Error in Y center [mm]
    G4double fNonPixel_FitSigmaXErr;       // Error in sigma X [mm]
    G4double fNonPixel_FitSigmaYErr;       // Error in sigma Y [mm]
    G4double fNonPixel_FitThetaErr;        // Error in rotation angle [rad]
    G4double fNonPixel_FitOffsetErr;       // Error in offset
    
    G4double fNonPixel_FitChi2;            // Chi-squared value
    G4double fNonPixel_FitNDF;             // Number of degrees of freedom
    G4double fNonPixel_FitChi2red;            // Reduced chi-squared (chi2red/NDF)
    G4double fNonPixel_FitPp;            // Fit probability (P-value)
    G4int fNonPixel_FitNPoints;            // Number of points used in fit
    G4double fNonPixel_FitResidualMean;    // Mean of residuals
    G4double fNonPixel_FitResidualStd;     // Standard deviation of residuals
    
    // Enhanced robustness metrics
    G4bool fNonPixel_FitConstraintsSatisfied; // Whether geometric constraints were satisfied
    
    // Additional variables for convenient access to Gaussian center and distance calculation
    G4double fNonPixel_GaussTrueDeltaX;  // Delta X from Gaussian center to true position [mm] (x_gauss - x_true)
    G4double fNonPixel_GaussTrueDeltaY;  // Delta Y from Gaussian center to true position [mm] (y_gauss - y_true)
    
    // Variables for 2D Gaussian fit results (central row and column fitting)
    G4double fNonPixel_Fit2D_XCenter;        // Fitted X center from central row [mm]
    G4double fNonPixel_Fit2D_XSigma;         // Fitted X sigma from central row [mm]
    G4double fNonPixel_Fit2D_XAmplitude;     // Fitted X amplitude from central row
    G4double fNonPixel_Fit2D_XCenterErr;     // Error in fitted X center [mm]
    G4double fNonPixel_Fit2D_XSigmaErr;      // Error in fitted X sigma [mm]
    G4double fNonPixel_Fit2D_XAmplitudeErr;  // Error in fitted X amplitude
    G4double fNonPixel_Fit2D_XChi2red;       // Reduced chi-squared for X fit
    G4int fNonPixel_Fit2D_XNPoints;          // Number of points used in X fit
    
    G4double fNonPixel_Fit2D_YCenter;        // Fitted Y center from central column [mm]
    G4double fNonPixel_Fit2D_YSigma;         // Fitted Y sigma from central column [mm]
    G4double fNonPixel_Fit2D_YAmplitude;     // Fitted Y amplitude from central column
    G4double fNonPixel_Fit2D_YCenterErr;     // Error in fitted Y center [mm]
    G4double fNonPixel_Fit2D_YSigmaErr;      // Error in fitted Y sigma [mm]
    G4double fNonPixel_Fit2D_YAmplitudeErr;  // Error in fitted Y amplitude
    G4double fNonPixel_Fit2D_YChi2red;       // Reduced chi-squared for Y fit
    G4int fNonPixel_Fit2D_YNPoints;          // Number of points used in Y fit
    
    G4bool fNonPixel_Fit2D_Successful;       // Whether 2D fitting was successful
    
    // Variables for diagonal Gaussian fit results (4 separate fits: Main X, Main Y, Sec X, Sec Y)
    // Main diagonal X fit (X vs Charge for pixels on main diagonal)
    G4double fNonPixel_FitDiag_MainXCenter;        // Fitted X center from main diagonal X fit [mm]
    G4double fNonPixel_FitDiag_MainXSigma;         // Fitted X sigma from main diagonal X fit [mm]
    G4double fNonPixel_FitDiag_MainXAmplitude;     // Fitted X amplitude from main diagonal X fit
    G4double fNonPixel_FitDiag_MainXCenterErr;     // Error in fitted X center from main diagonal X fit [mm]
    G4double fNonPixel_FitDiag_MainXSigmaErr;      // Error in fitted X sigma from main diagonal X fit [mm]
    G4double fNonPixel_FitDiag_MainXAmplitudeErr;  // Error in fitted X amplitude from main diagonal X fit
    G4double fNonPixel_FitDiag_MainXChi2red;       // Reduced chi-squared for main diagonal X fit
    G4int fNonPixel_FitDiag_MainXNPoints;          // Number of points used in main diagonal X fit
    G4bool fNonPixel_FitDiag_MainXSuccessful;      // Whether main diagonal X fitting was successful
    
    // Main diagonal Y fit (Y vs Charge for pixels on main diagonal)
    G4double fNonPixel_FitDiag_MainYCenter;        // Fitted Y center from main diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_MainYSigma;         // Fitted Y sigma from main diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_MainYAmplitude;     // Fitted Y amplitude from main diagonal Y fit
    G4double fNonPixel_FitDiag_MainYCenterErr;     // Error in fitted Y center from main diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_MainYSigmaErr;      // Error in fitted Y sigma from main diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_MainYAmplitudeErr;  // Error in fitted Y amplitude from main diagonal Y fit
    G4double fNonPixel_FitDiag_MainYChi2red;       // Reduced chi-squared for main diagonal Y fit
    G4int fNonPixel_FitDiag_MainYNPoints;          // Number of points used in main diagonal Y fit
    G4bool fNonPixel_FitDiag_MainYSuccessful;      // Whether main diagonal Y fitting was successful
    
    // Secondary diagonal X fit (X vs Charge for pixels on secondary diagonal)
    G4double fNonPixel_FitDiag_SecXCenter;         // Fitted X center from secondary diagonal X fit [mm]
    G4double fNonPixel_FitDiag_SecXSigma;          // Fitted X sigma from secondary diagonal X fit [mm]
    G4double fNonPixel_FitDiag_SecXAmplitude;      // Fitted X amplitude from secondary diagonal X fit
    G4double fNonPixel_FitDiag_SecXCenterErr;      // Error in fitted X center from secondary diagonal X fit [mm]
    G4double fNonPixel_FitDiag_SecXSigmaErr;       // Error in fitted X sigma from secondary diagonal X fit [mm]
    G4double fNonPixel_FitDiag_SecXAmplitudeErr;   // Error in fitted X amplitude from secondary diagonal X fit
    G4double fNonPixel_FitDiag_SecXChi2red;        // Reduced chi-squared for secondary diagonal X fit
    G4int fNonPixel_FitDiag_SecXNPoints;           // Number of points used in secondary diagonal X fit
    G4bool fNonPixel_FitDiag_SecXSuccessful;       // Whether secondary diagonal X fitting was successful
    
    // Secondary diagonal Y fit (Y vs Charge for pixels on secondary diagonal)
    G4double fNonPixel_FitDiag_SecYCenter;         // Fitted Y center from secondary diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_SecYSigma;          // Fitted Y sigma from secondary diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_SecYAmplitude;      // Fitted Y amplitude from secondary diagonal Y fit
    G4double fNonPixel_FitDiag_SecYCenterErr;      // Error in fitted Y center from secondary diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_SecYSigmaErr;       // Error in fitted Y sigma from secondary diagonal Y fit [mm]
    G4double fNonPixel_FitDiag_SecYAmplitudeErr;   // Error in fitted Y amplitude from secondary diagonal Y fit
    G4double fNonPixel_FitDiag_SecYChi2red;        // Reduced chi-squared for secondary diagonal Y fit
    G4int fNonPixel_FitDiag_SecYNPoints;           // Number of points used in secondary diagonal Y fit
    G4bool fNonPixel_FitDiag_SecYSuccessful;       // Whether secondary diagonal Y fitting was successful
    
    G4bool fNonPixel_FitDiag_Successful;           // Whether diagonal fitting was successful
    
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