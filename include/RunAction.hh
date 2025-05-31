#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <vector>

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
    
    // Method to set pixel hit flag
    void SetPixelHit(G4bool hit);
    
    // Method to set neighborhood (9x9) grid angle data
    void SetNeighborhoodGridData(const std::vector<G4double>& angles, 
                        const std::vector<G4int>& pixelI, 
                        const std::vector<G4int>& pixelJ);
    
    // Method to set neighborhood (9x9) grid charge sharing data
    void SetNeighborhoodChargeData(const std::vector<G4double>& chargeFractions,
                          const std::vector<G4double>& distances,
                          const std::vector<G4double>& chargeValues,
                          const std::vector<G4double>& chargeCoulombs);
    
    // Method to set detector grid parameters for saving to ROOT
    void SetDetectorGridParameters(G4double pixelSize, G4double pixelSpacing, 
                                   G4double pixelCornerOffset, G4double detSize, 
                                   G4int numBlocksPerSide);
    
    // Method to set particle information
    void SetParticleInfo(G4int eventID, G4double initialEnergy, G4double finalEnergy, 
                        G4double momentum, const G4String& particleName);
    
    // Method to set step-by-step energy deposition information
    void SetStepEnergyDeposition(const std::vector<G4double>& stepEdep,
                                const std::vector<G4double>& stepZ,
                                const std::vector<G4double>& stepTime);
    
    // Method to set ALL step information (including non-energy depositing steps)
    void SetAllStepInfo(const std::vector<G4double>& stepEdep,
                       const std::vector<G4double>& stepZ,
                       const std::vector<G4double>& stepTime);
    
    // Method to set 3D Gaussian fit results
    void SetGaussianFitResults(G4double amplitude, G4double x0, G4double y0,
                              G4double sigma_x, G4double sigma_y, G4double theta, G4double offset,
                              G4double amplitude_err, G4double x0_err, G4double y0_err,
                              G4double sigma_x_err, G4double sigma_y_err, G4double theta_err, G4double offset_err,
                              G4double chi2, G4double ndf, G4double prob,
                              G4int n_points,
                              G4double residual_mean, G4double residual_std,
                              G4bool constraints_satisfied,
                              G4double center_distance_from_detector_edge, G4double min_distance_to_pixel,
                              G4int fit_attempt_number);
    
    // Fill the ROOT tree with current event data
    void FillTree();

private:
    TFile* fRootFile;
    TTree* fTree;
    
    // Thread-safety mutex for ROOT operations
    static std::mutex fRootMutex;
    
    // Variables for data storage with fixed units
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
    G4double fPixelDist; // Distance from hit to pixel center [mm]
    G4double fPixelAlpha; // Angular size of pixel from hit position [deg]
    G4bool fPixelHit;  // Flag to indicate if hit was on a pixel
    
    // Variables for neighborhood (9x9) grid angle data
    std::vector<G4double> fGridNeighborhoodAngles; // Angles from hit to neighborhood grid pixels [deg]
    std::vector<G4int> fGridNeighborhoodPixelI;     // I indices of neighborhood grid pixels
    std::vector<G4int> fGridNeighborhoodPixelJ;     // J indices of neighborhood grid pixels
    
    // Variables for neighborhood (9x9) grid charge sharing data
    std::vector<G4double> fGridNeighborhoodChargeFractions; // Charge fractions for neighborhood grid pixels
    std::vector<G4double> fGridNeighborhoodDistances;         // Distances from hit to neighborhood grid pixels [mm]
    std::vector<G4double> fGridNeighborhoodChargeValues;        // Charge values for neighborhood grid pixels (electrons)
    std::vector<G4double> fGridNeighborhoodChargeCoulombs;       // Charge values in Coulombs for neighborhood grid pixels
    
    // Variables for detector grid parameters (stored as ROOT metadata)
    G4double fGridPixelSize;        // Pixel size [mm]
    G4double fGridPixelSpacing;     // Pixel spacing [mm]  
    G4double fGridPixelCornerOffset; // Pixel corner offset [mm]
    G4double fGridDetSize;          // Detector size [mm]
    G4int fGridNumBlocksPerSide;    // Number of blocks per side
    
    // Variables for particle information
    G4int fEventID;                 // Event ID
    G4double fInitialEnergy;        // Initial particle energy [MeV]
    G4double fFinalEnergy;          // Final particle energy [MeV]
    G4double fMomentum;             // Particle momentum [MeV/c]
    std::string fParticleName;      // Particle type name
    
    // Variables for step-by-step energy deposition information
    std::vector<G4double> fStepEdepVec;    // Energy deposited per step [MeV]
    std::vector<G4double> fStepZVec;       // Z position of each energy deposit [mm]
    std::vector<G4double> fStepTimeVec;    // Time of each energy deposit [ns]
    
    // Variables for ALL step information (including non-energy depositing steps)
    std::vector<G4double> fAllStepEdepVec;    // Energy deposited per step (including 0) [MeV]
    std::vector<G4double> fAllStepZVec;       // Z position of each step [mm]
    std::vector<G4double> fAllStepTimeVec;    // Time of each step [ns]
    std::vector<G4double> fAllStepLenVec;     // Length of each step [mm]
    std::vector<G4int> fAllStepNumVec;        // Step number for each step
    
    // Variables for 3D Gaussian fit results
    G4double fFitAmplitude;         // Fitted amplitude
    G4double fFitX0;               // Fitted X center [mm]
    G4double fFitY0;               // Fitted Y center [mm]
    G4double fFitSigmaX;           // Fitted sigma X [mm]
    G4double fFitSigmaY;           // Fitted sigma Y [mm]
    G4double fFitTheta;            // Fitted rotation angle [rad]
    G4double fFitOffset;           // Fitted offset
    
    G4double fFitAmplitudeErr;     // Error in amplitude
    G4double fFitX0Err;           // Error in X center [mm]
    G4double fFitY0Err;           // Error in Y center [mm]
    G4double fFitSigmaXErr;       // Error in sigma X [mm]
    G4double fFitSigmaYErr;       // Error in sigma Y [mm]
    G4double fFitThetaErr;        // Error in rotation angle [rad]
    G4double fFitOffsetErr;       // Error in offset
    
    G4double fFitChi2;            // Chi-squared value
    G4double fFitNDF;             // Number of degrees of freedom
    G4double fFitProb;            // Fit probability
    G4int fFitNPoints;            // Number of points used in fit
    G4double fFitResidualMean;    // Mean of residuals
    G4double fFitResidualStd;     // Standard deviation of residuals
    
    // Enhanced robustness metrics
    G4bool fFitConstraintsSatisfied; // Whether geometric constraints were satisfied
    G4double fFitCenterDistFromEdge; // Distance from fit center to detector edge [mm]
    G4double fFitMinDistToPixel;  // Minimum distance from fit center to any pixel [mm]
    G4int fFitAttemptNumber;      // Which fitting attempt succeeded (1-based)
    
    // Additional variables for convenient access to Gaussian center and distance calculation
    G4double fGaussX;             // Gaussian X center [mm] (alias for fFitX0)
    G4double fGaussY;             // Gaussian Y center [mm] (alias for fFitY0)  
    G4double fGaussTrueDistance;  // Distance from Gaussian center to true position [mm]
};

#endif