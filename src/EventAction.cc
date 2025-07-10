#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "Constants.hh"
#include "CrashHandler.hh"
#include "SimulationLogger.hh"
#include "2DGaussianFitCeres.hh"
#include "2DLorentzianFitCeres.hh"
#include "2DPowerLorentzianFitCeres.hh"
#include "3DGaussianFitCeres.hh"
#include "3DLorentzianFitCeres.hh"
#include "3DPowerLorentzianFitCeres.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Alpha calculation method: ANALYTICAL
// This implementation uses the analytical formula for calculating the alpha angle:
// α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
// where:
//   l = side length of the pixel pad (pixel_size)
//   d = distance from event hit to center of pixel pad
// See Page 9: https://indico.cern.ch/event/813597/contributions/3727782/attachments/1989546/3540780/TREDI_Cartiglia.pdf

EventAction::EventAction(RunAction* runAction, DetectorConstruction* detector)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetector(detector),
  fNeighborhoodRadius(4), // Default to 9x9 grid (radius 4)
  fEdep(0.),
  fPosition(G4ThreeVector(0.,0.,0.)),
  fInitialPosition(G4ThreeVector(0.,0.,0.)),
  fHasHit(false),
  fPixelIndexI(-1),
  fPixelIndexJ(-1),
  fPixelTrueDeltaX(0),
  fPixelTrueDeltaY(0),
  fActualPixelDistance(-1.),
  fPixelHit(false),
  fAutoRadiusEnabled(Constants::ENABLE_AUTO_RADIUS),
  fMinAutoRadius(Constants::MIN_AUTO_RADIUS),
  fMaxAutoRadius(Constants::MAX_AUTO_RADIUS),
  fSelectedRadius(4),
  fSelectedFitQuality(0.0),
  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE)
{ 
  G4cout << "EventAction: Using 2D Gaussian fitting for central row and column" << G4endl;
}

EventAction::~EventAction()
{ 
  // No 3D Gaussian fitter to clean up
}

void EventAction::BeginOfEventAction(const G4Event* event)
{
  // Log event start
  SimulationLogger* logger = SimulationLogger::GetInstance();
  if (logger) {
    logger->LogEventStart(event->GetEventID());
  }
  
  // Reset per-event variables
  fEdep = 0.;
  fPosition = G4ThreeVector(0.,0.,0.);
  fHasHit = false;
  
  // Initialize particle position - this will be updated when the primary vertex is created
  fInitialPosition = G4ThreeVector(0.,0.,0.);
  
  // Reset pixel mapping variables
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDeltaX = 0;
  fPixelTrueDeltaY = 0;
  fActualPixelDistance = -1.;
  fPixelHit = false;
  
  // Reset neighborhood (9x9) grid angle data
  fNonPixel_GridNeighborhoodAngles.clear();
  
  // Reset neighborhood (9x9) grid charge sharing data
  fNonPixel_GridNeighborhoodChargeFractions.clear();
  fNonPixel_GridNeighborhoodDistances.clear();
  fNonPixel_GridNeighborhoodCharge.clear();
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // Get the primary vertex position and energy from the event
  if (event->GetPrimaryVertex()) {
    G4ThreeVector primaryPos = event->GetPrimaryVertex()->GetPosition();
    // Update the initial position
    fInitialPosition = primaryPos;
    
    // Get the initial particle energy (kinetic energy)
    G4PrimaryParticle* primaryParticle = event->GetPrimaryVertex()->GetPrimary();
    if (primaryParticle) {
      G4double initialKineticEnergy = primaryParticle->GetKineticEnergy();
      // Store the initial energy in the RunAction for ROOT output
      // Convert from Geant4 internal units (MeV) to MeV for storage
      fRunAction->SetInitialEnergy(initialKineticEnergy);
    }
  }
  
  // Calculate and store nearest pixel position (this calculates fActualPixelDistance)
  G4ThreeVector nearestPixel = CalculateNearestPixel(fPosition);
  
  // Calculate and pass pixel angular size to RunAction
  G4double pixelAlpha = CalculatePixelAlpha(fPosition, fPixelIndexI, fPixelIndexJ);
  
  // Determine if hit is a pixel hit: only based on whether it's on pixel surface
  G4bool isPixelHit = fPixelHit;
  
  // ENERGY DEPOSITION LOGIC:
  // - Energy is summed only while particle travels through detector volume (handled in SteppingAction)
  // - For pixel hits: set energy deposition to zero (per user requirement)  
  // - For non-pixel hits: use energy deposited inside detector during particle passage
  G4double finalEdep = fEdep;
  if (isPixelHit) {
    finalEdep = 0.0; // Set to zero for pixel hits (per user requirement)
    // G4cout << "EventAction: Setting energy deposition to zero for pixel hit (event " << eventID << ")" << G4endl;
  } else {
    // For non-pixel hits, use the energy deposited inside the detector (already accumulated in fEdep from SteppingAction)
    // G4cout << "EventAction: Using detector energy deposition " << finalEdep/MeV 
    //        << " MeV for non-pixel hit (event " << eventID << ")" << G4endl;
  }
  
  // Pass classification data to RunAction
  fRunAction->SetPixelClassification(isPixelHit, fPixelTrueDeltaX, fPixelTrueDeltaY);
  fRunAction->SetPixelHitStatus(isPixelHit);
  
  // Update the event data with the corrected energy deposition
  fRunAction->SetEventData(finalEdep, fPosition.x(), fPosition.y(), fPosition.z());
  fRunAction->SetInitialPosition(fInitialPosition.x(), fInitialPosition.y(), fInitialPosition.z());
  
  // Set nearest pixel position in RunAction  
  fRunAction->SetNearestPixelPosition(nearestPixel.x(), nearestPixel.y());
  
  // Only calculate and store pixel-specific data for pixel hits (on pixel surface)
  if (isPixelHit) {
    
    // Clear non-pixel data for pixel hits
    fNonPixel_GridNeighborhoodAngles.clear();
    fNonPixel_GridNeighborhoodChargeFractions.clear();
    fNonPixel_GridNeighborhoodDistances.clear();
    fNonPixel_GridNeighborhoodCharge.clear();
  } else {
    // Non-pixel hit: determine optimal radius and calculate neighborhood grid data
    if (fAutoRadiusEnabled) {
      // Perform automatic radius selection based on fit quality
      fSelectedRadius = SelectOptimalRadius(fPosition, fPixelIndexI, fPixelIndexJ);
      fNeighborhoodRadius = fSelectedRadius;
      
      G4cout << "EventAction: Auto-selected radius " << fSelectedRadius 
             << " with fit quality " << fSelectedFitQuality << G4endl;
    } else {
      // Use fixed radius
      fSelectedRadius = fNeighborhoodRadius;
      fSelectedFitQuality = 0.0; // Not evaluated
    }
    
    // Calculate neighborhood grid data with selected radius
    CalculateNeighborhoodGridAngles(fPosition, fPixelIndexI, fPixelIndexJ);
    CalculateNeighborhoodChargeSharing();
  }
  
  // Pass neighborhood grid data to RunAction (will be empty for pixel hits)
  fRunAction->SetNeighborhoodGridData(fNonPixel_GridNeighborhoodAngles);
  fRunAction->SetNeighborhoodChargeData(fNonPixel_GridNeighborhoodChargeFractions, fNonPixel_GridNeighborhoodDistances, fNonPixel_GridNeighborhoodCharge, fNonPixel_GridNeighborhoodCharge);
  
  // Pass automatic radius selection results to RunAction
  fRunAction->SetAutoRadiusResults(fSelectedRadius);
  
  // Perform 2D Gaussian fitting on charge distribution data (central row and column)
  // Only fit for non-pixel hits (not on pixel surface)
  G4bool shouldPerformFit = !isPixelHit && !fNonPixel_GridNeighborhoodChargeFractions.empty();
  
  if (shouldPerformFit) {
    // Extract coordinates and charge values for fitting
    std::vector<double> x_coords, y_coords, charge_values;
    
    // Get detector parameters for coordinate calculation
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    
    // Convert grid indices to actual coordinates
    // The neighborhood grid is a systematic 9x9 grid around the center pixel
    G4int gridSize = 2 * fNeighborhoodRadius + 1; // Should be 9 for radius 4
    for (size_t i = 0; i < fNonPixel_GridNeighborhoodChargeFractions.size(); ++i) {
      if (fNonPixel_GridNeighborhoodChargeFractions[i] > 0) { // Only include pixels with charge
        // Calculate grid position from array index
        // The grid is stored in column-major order: i = col * gridSize + row
        // because di (X) is outer loop, dj (Y) is inner loop in charge calculation
        G4int col = i / gridSize;  // di (X) was outer loop
        G4int row = i % gridSize;  // dj (Y) was inner loop
        
        // Convert grid position to pixel offset from center
        G4int offsetI = col - fNeighborhoodRadius; // -4 to +4 for 9x9 grid (X offset)
        G4int offsetJ = row - fNeighborhoodRadius; // -4 to +4 for 9x9 grid (Y offset)
        
        // Calculate actual position
        G4double x_pos = nearestPixel.x() + offsetI * pixelSpacing;
        G4double y_pos = nearestPixel.y() + offsetJ * pixelSpacing;
        
        x_coords.push_back(x_pos);
        y_coords.push_back(y_pos);
        // Use actual charge values (in Coulombs) instead of fractions for fitting
        charge_values.push_back(fNonPixel_GridNeighborhoodCharge[i]);
      }
    }
    
    // ===============================================
    // GAUSSIAN FITTING (conditionally enabled)
    // ===============================================
    
    // Perform 2D fitting if we have enough data points and Gaussian fitting is enabled
    if (x_coords.size() >= 3 && Constants::ENABLE_GAUSSIAN_FITTING && Constants::ENABLE_2D_FITTING) { // Need at least 3 points for 1D Gaussian fit
      // Perform 2D Gaussian fitting using the Ceres Solver implementation
      GaussianFit2DResultsCeres fitResults = Fit2DGaussianCeres(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, 
        false, // verbose=false for production
        false); // enable_outlier_filtering
      
      if (fitResults.fit_successful) {
        // Removed verbose debug output for cleaner simulation logs
      }
      
      // Pass 2D fit results to RunAction
      fRunAction->Set2DGaussianFitResults(
        fitResults.x_center, fitResults.x_sigma, fitResults.x_amplitude,
        fitResults.x_center_err, fitResults.x_sigma_err, fitResults.x_amplitude_err,
        fitResults.x_vertical_offset, fitResults.x_vertical_offset_err,
        fitResults.x_chi2red, fitResults.x_pp, fitResults.x_dof,
        fitResults.y_center, fitResults.y_sigma, fitResults.y_amplitude,
        fitResults.y_center_err, fitResults.y_sigma_err, fitResults.y_amplitude_err,
        fitResults.y_vertical_offset, fitResults.y_vertical_offset_err,
        fitResults.y_chi2red, fitResults.y_pp, fitResults.y_dof,
        fitResults.x_charge_uncertainty, fitResults.y_charge_uncertainty,
        fitResults.fit_successful);
      
      // Log Gaussian fitting results to SimulationLogger
      SimulationLogger* logger = SimulationLogger::GetInstance();
      if (logger) {
        logger->LogGaussianFitResults(event->GetEventID(), fitResults);
      }
        
        // Perform diagonal fitting if 2D fitting was performed and successful and diagonal fitting is enabled
        if (fitResults.fit_successful && Constants::ENABLE_DIAGONAL_FITTING) {
          // Perform diagonal Gaussian fitting using the Ceres Solver implementation
          DiagonalFitResultsCeres diagResults = FitDiagonalGaussianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (diagResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass diagonal fit results to RunAction
        fRunAction->SetDiagonalGaussianFitResults(
          diagResults.main_diag_x_center, diagResults.main_diag_x_sigma, diagResults.main_diag_x_amplitude,
          diagResults.main_diag_x_center_err, diagResults.main_diag_x_sigma_err, diagResults.main_diag_x_amplitude_err,
          diagResults.main_diag_x_vertical_offset, diagResults.main_diag_x_vertical_offset_err,
          diagResults.main_diag_x_chi2red, diagResults.main_diag_x_pp, diagResults.main_diag_x_dof, diagResults.main_diag_x_fit_successful,
          diagResults.main_diag_y_center, diagResults.main_diag_y_sigma, diagResults.main_diag_y_amplitude,
          diagResults.main_diag_y_center_err, diagResults.main_diag_y_sigma_err, diagResults.main_diag_y_amplitude_err,
          diagResults.main_diag_y_vertical_offset, diagResults.main_diag_y_vertical_offset_err,
          diagResults.main_diag_y_chi2red, diagResults.main_diag_y_pp, diagResults.main_diag_y_dof, diagResults.main_diag_y_fit_successful,
          diagResults.sec_diag_x_center, diagResults.sec_diag_x_sigma, diagResults.sec_diag_x_amplitude,
          diagResults.sec_diag_x_center_err, diagResults.sec_diag_x_sigma_err, diagResults.sec_diag_x_amplitude_err,
          diagResults.sec_diag_x_vertical_offset, diagResults.sec_diag_x_vertical_offset_err,
          diagResults.sec_diag_x_chi2red, diagResults.sec_diag_x_pp, diagResults.sec_diag_x_dof, diagResults.sec_diag_x_fit_successful,
          diagResults.sec_diag_y_center, diagResults.sec_diag_y_sigma, diagResults.sec_diag_y_amplitude,
          diagResults.sec_diag_y_center_err, diagResults.sec_diag_y_sigma_err, diagResults.sec_diag_y_amplitude_err,
          diagResults.sec_diag_y_vertical_offset, diagResults.sec_diag_y_vertical_offset_err,
          diagResults.sec_diag_y_chi2red, diagResults.sec_diag_y_pp, diagResults.sec_diag_y_dof, diagResults.sec_diag_y_fit_successful,
          diagResults.fit_successful);
      } else {
        // Set default diagonal fit values when 2D fitting failed
        fRunAction->SetDiagonalGaussianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amplitude, center_err, sigma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
      
      // ===============================================
      // LORENTZIAN FITTING (conditionally enabled)
      // ===============================================
      
      // Perform 2D Lorentzian fitting if we have enough data points and Lorentzian fitting is enabled
      if (x_coords.size() >= 3 && Constants::ENABLE_LORENTZIAN_FITTING && Constants::ENABLE_2D_FITTING) { // Need at least 3 points for 1D Lorentzian fit
        // Perform 2D Lorentzian fitting using the Ceres Solver implementation
        LorentzianFit2DResultsCeres lorentzFitResults = Fit2DLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentzFitResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass 2D Lorentzian fit results to RunAction
        fRunAction->Set2DLorentzianFitResults(
          lorentzFitResults.x_center, lorentzFitResults.x_gamma, lorentzFitResults.x_amplitude,
          lorentzFitResults.x_center_err, lorentzFitResults.x_gamma_err, lorentzFitResults.x_amplitude_err,
          lorentzFitResults.x_vertical_offset, lorentzFitResults.x_vertical_offset_err,
          lorentzFitResults.x_chi2red, lorentzFitResults.x_pp, lorentzFitResults.x_dof,
          lorentzFitResults.y_center, lorentzFitResults.y_gamma, lorentzFitResults.y_amplitude,
          lorentzFitResults.y_center_err, lorentzFitResults.y_gamma_err, lorentzFitResults.y_amplitude_err,
          lorentzFitResults.y_vertical_offset, lorentzFitResults.y_vertical_offset_err,
          lorentzFitResults.y_chi2red, lorentzFitResults.y_pp, lorentzFitResults.y_dof,
          lorentzFitResults.x_charge_uncertainty, lorentzFitResults.y_charge_uncertainty,
          lorentzFitResults.fit_successful);
        
        // Log Lorentzian fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->LogLorentzianFitResults(event->GetEventID(), lorentzFitResults);
        }
          
        // Note: Lorentzian charge error data was removed as per user request
          
        // Perform diagonal Lorentzian fitting if 2D fitting was performed and successful and diagonal fitting is enabled
        if (lorentzFitResults.fit_successful && Constants::ENABLE_DIAGONAL_FITTING) {
          // Perform diagonal Lorentzian fitting using the Ceres Solver implementation
          DiagonalLorentzianFitResultsCeres lorentzDiagResults = FitDiagonalLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentzDiagResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass diagonal Lorentzian fit results to RunAction
        fRunAction->SetDiagonalLorentzianFitResults(
          lorentzDiagResults.main_diag_x_center, lorentzDiagResults.main_diag_x_gamma, lorentzDiagResults.main_diag_x_amplitude,
          lorentzDiagResults.main_diag_x_center_err, lorentzDiagResults.main_diag_x_gamma_err, lorentzDiagResults.main_diag_x_amplitude_err,
          lorentzDiagResults.main_diag_x_vertical_offset, lorentzDiagResults.main_diag_x_vertical_offset_err,
          lorentzDiagResults.main_diag_x_chi2red, lorentzDiagResults.main_diag_x_pp, lorentzDiagResults.main_diag_x_dof, lorentzDiagResults.main_diag_x_fit_successful,
          lorentzDiagResults.main_diag_y_center, lorentzDiagResults.main_diag_y_gamma, lorentzDiagResults.main_diag_y_amplitude,
          lorentzDiagResults.main_diag_y_center_err, lorentzDiagResults.main_diag_y_gamma_err, lorentzDiagResults.main_diag_y_amplitude_err,
          lorentzDiagResults.main_diag_y_vertical_offset, lorentzDiagResults.main_diag_y_vertical_offset_err,
          lorentzDiagResults.main_diag_y_chi2red, lorentzDiagResults.main_diag_y_pp, lorentzDiagResults.main_diag_y_dof, lorentzDiagResults.main_diag_y_fit_successful,
          lorentzDiagResults.sec_diag_x_center, lorentzDiagResults.sec_diag_x_gamma, lorentzDiagResults.sec_diag_x_amplitude,
          lorentzDiagResults.sec_diag_x_center_err, lorentzDiagResults.sec_diag_x_gamma_err, lorentzDiagResults.sec_diag_x_amplitude_err,
          lorentzDiagResults.sec_diag_x_vertical_offset, lorentzDiagResults.sec_diag_x_vertical_offset_err,
          lorentzDiagResults.sec_diag_x_chi2red, lorentzDiagResults.sec_diag_x_pp, lorentzDiagResults.sec_diag_x_dof, lorentzDiagResults.sec_diag_x_fit_successful,
          lorentzDiagResults.sec_diag_y_center, lorentzDiagResults.sec_diag_y_gamma, lorentzDiagResults.sec_diag_y_amplitude,
          lorentzDiagResults.sec_diag_y_center_err, lorentzDiagResults.sec_diag_y_gamma_err, lorentzDiagResults.sec_diag_y_amplitude_err,
          lorentzDiagResults.sec_diag_y_vertical_offset, lorentzDiagResults.sec_diag_y_vertical_offset_err,
          lorentzDiagResults.sec_diag_y_chi2red, lorentzDiagResults.sec_diag_y_pp, lorentzDiagResults.sec_diag_y_dof, lorentzDiagResults.sec_diag_y_fit_successful,
          lorentzDiagResults.fit_successful);
          
        // Note: Diagonal Lorentzian charge error data was removed as per user request
      } else {
        // Set default diagonal Lorentzian fit values when 2D fitting failed
        fRunAction->SetDiagonalLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
        
    } else {
      // Not enough data points for Lorentzian fitting or Lorentzian fitting is disabled
      if (Constants::ENABLE_LORENTZIAN_FITTING) {
        fRunAction->Set2DLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
          false); // fit_successful = false
        
        // Set default diagonal Lorentzian fit values when not enough data points
        fRunAction->SetDiagonalLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
    }
      
      // ===============================================
      // POWER-LAW LORENTZIAN FITTING (conditionally enabled)
      // ===============================================
      
      // Perform 2D Power-Law Lorentzian fitting if we have enough data points and Power-Law Lorentzian fitting is enabled
      if (x_coords.size() >= 3 && Constants::ENABLE_POWER_LORENTZIAN_FITTING && Constants::ENABLE_2D_FITTING) { // Need at least 3 points for Power-Law Lorentzian fit
        // Perform 2D Power-Law Lorentzian fitting using the Ceres Solver implementation
        PowerLorentzianFit2DResultsCeres powerLorentzFitResults = Fit2DPowerLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentzFitResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass 2D Power-Law Lorentzian fit results to RunAction
        fRunAction->Set2DPowerLorentzianFitResults(
          powerLorentzFitResults.x_center, powerLorentzFitResults.x_gamma, powerLorentzFitResults.x_beta, powerLorentzFitResults.x_amplitude,
          powerLorentzFitResults.x_center_err, powerLorentzFitResults.x_gamma_err, powerLorentzFitResults.x_beta_err, powerLorentzFitResults.x_amplitude_err,
          powerLorentzFitResults.x_vertical_offset, powerLorentzFitResults.x_vertical_offset_err,
          powerLorentzFitResults.x_chi2red, powerLorentzFitResults.x_pp, powerLorentzFitResults.x_dof,
          powerLorentzFitResults.y_center, powerLorentzFitResults.y_gamma, powerLorentzFitResults.y_beta, powerLorentzFitResults.y_amplitude,
          powerLorentzFitResults.y_center_err, powerLorentzFitResults.y_gamma_err, powerLorentzFitResults.y_beta_err, powerLorentzFitResults.y_amplitude_err,
          powerLorentzFitResults.y_vertical_offset, powerLorentzFitResults.y_vertical_offset_err,
          powerLorentzFitResults.y_chi2red, powerLorentzFitResults.y_pp, powerLorentzFitResults.y_dof,
          powerLorentzFitResults.x_charge_uncertainty, powerLorentzFitResults.y_charge_uncertainty,
          powerLorentzFitResults.fit_successful);
        
        // Log Power Lorentzian fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->LogPowerLorentzianFitResults(event->GetEventID(), powerLorentzFitResults);
        }
          
        // Perform diagonal Power-Law Lorentzian fitting if 2D fitting was performed and successful and diagonal fitting is enabled
        if (powerLorentzFitResults.fit_successful && Constants::ENABLE_DIAGONAL_FITTING) {
          // Perform diagonal Power-Law Lorentzian fitting using the Ceres Solver implementation
          DiagonalPowerLorentzianFitResultsCeres powerLorentzDiagResults = FitDiagonalPowerLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentzDiagResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass diagonal Power-Law Lorentzian fit results to RunAction
        fRunAction->SetDiagonalPowerLorentzianFitResults(
          powerLorentzDiagResults.main_diag_x_center, powerLorentzDiagResults.main_diag_x_gamma, powerLorentzDiagResults.main_diag_x_beta, powerLorentzDiagResults.main_diag_x_amplitude,
          powerLorentzDiagResults.main_diag_x_center_err, powerLorentzDiagResults.main_diag_x_gamma_err, powerLorentzDiagResults.main_diag_x_beta_err, powerLorentzDiagResults.main_diag_x_amplitude_err,
          powerLorentzDiagResults.main_diag_x_vertical_offset, powerLorentzDiagResults.main_diag_x_vertical_offset_err,
          powerLorentzDiagResults.main_diag_x_chi2red, powerLorentzDiagResults.main_diag_x_pp, powerLorentzDiagResults.main_diag_x_dof, powerLorentzDiagResults.main_diag_x_fit_successful,
          powerLorentzDiagResults.main_diag_y_center, powerLorentzDiagResults.main_diag_y_gamma, powerLorentzDiagResults.main_diag_y_beta, powerLorentzDiagResults.main_diag_y_amplitude,
          powerLorentzDiagResults.main_diag_y_center_err, powerLorentzDiagResults.main_diag_y_gamma_err, powerLorentzDiagResults.main_diag_y_beta_err, powerLorentzDiagResults.main_diag_y_amplitude_err,
          powerLorentzDiagResults.main_diag_y_vertical_offset, powerLorentzDiagResults.main_diag_y_vertical_offset_err,
          powerLorentzDiagResults.main_diag_y_chi2red, powerLorentzDiagResults.main_diag_y_pp, powerLorentzDiagResults.main_diag_y_dof, powerLorentzDiagResults.main_diag_y_fit_successful,
          powerLorentzDiagResults.sec_diag_x_center, powerLorentzDiagResults.sec_diag_x_gamma, powerLorentzDiagResults.sec_diag_x_beta, powerLorentzDiagResults.sec_diag_x_amplitude,
          powerLorentzDiagResults.sec_diag_x_center_err, powerLorentzDiagResults.sec_diag_x_gamma_err, powerLorentzDiagResults.sec_diag_x_beta_err, powerLorentzDiagResults.sec_diag_x_amplitude_err,
          powerLorentzDiagResults.sec_diag_x_vertical_offset, powerLorentzDiagResults.sec_diag_x_vertical_offset_err,
          powerLorentzDiagResults.sec_diag_x_chi2red, powerLorentzDiagResults.sec_diag_x_pp, powerLorentzDiagResults.sec_diag_x_dof, powerLorentzDiagResults.sec_diag_x_fit_successful,
          powerLorentzDiagResults.sec_diag_y_center, powerLorentzDiagResults.sec_diag_y_gamma, powerLorentzDiagResults.sec_diag_y_beta, powerLorentzDiagResults.sec_diag_y_amplitude,
          powerLorentzDiagResults.sec_diag_y_center_err, powerLorentzDiagResults.sec_diag_y_gamma_err, powerLorentzDiagResults.sec_diag_y_beta_err, powerLorentzDiagResults.sec_diag_y_amplitude_err,
          powerLorentzDiagResults.sec_diag_y_vertical_offset, powerLorentzDiagResults.sec_diag_y_vertical_offset_err,
          powerLorentzDiagResults.sec_diag_y_chi2red, powerLorentzDiagResults.sec_diag_y_pp, powerLorentzDiagResults.sec_diag_y_dof, powerLorentzDiagResults.sec_diag_y_fit_successful,
          powerLorentzDiagResults.fit_successful);

      } else {
        // Set default diagonal Power-Law Lorentzian fit values when 2D fitting failed
        fRunAction->SetDiagonalPowerLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amplitude, center_err, gamma_err, beta_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
        
    } else {
      // Not enough data points for Power-Law Lorentzian fitting or Power-Law Lorentzian fitting is disabled
      if (Constants::ENABLE_POWER_LORENTZIAN_FITTING) {
        fRunAction->Set2DPowerLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, beta, amplitude, center_err, gamma_err, beta_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
          false); // fit_successful = false
        
        // Set default diagonal Power-Law Lorentzian fit values when not enough data points
        fRunAction->SetDiagonalPowerLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amplitude, center_err, gamma_err, beta_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
    }
      
      // ===============================================
      // 3D LORENTZIAN FITTING (conditionally enabled)
      // ===============================================
      
      // Perform 3D Lorentzian fitting if we have enough data points and 3D Lorentzian fitting is enabled
      if (x_coords.size() >= 6 && Constants::ENABLE_3D_LORENTZIAN_FITTING) { // Need at least 6 points for 3D Lorentzian fit
        // Perform 3D Lorentzian fitting using the Ceres Solver implementation
        LorentzianFit3DResultsCeres lorentz3DFitResults = Fit3DLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentz3DFitResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass 3D Lorentzian fit results to RunAction
        fRunAction->Set3DLorentzianFitResults(
          lorentz3DFitResults.center_x, lorentz3DFitResults.center_y, 
          lorentz3DFitResults.gamma_x, lorentz3DFitResults.gamma_y, 
          lorentz3DFitResults.amplitude, lorentz3DFitResults.vertical_offset,
          lorentz3DFitResults.center_x_err, lorentz3DFitResults.center_y_err,
          lorentz3DFitResults.gamma_x_err, lorentz3DFitResults.gamma_y_err,
          lorentz3DFitResults.amplitude_err, lorentz3DFitResults.vertical_offset_err,
          lorentz3DFitResults.chi2red, lorentz3DFitResults.pp, lorentz3DFitResults.dof,
          lorentz3DFitResults.charge_uncertainty,
          lorentz3DFitResults.fit_successful);
        
        // Log 3D Lorentzian fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DLorentzianFitResults(event->GetEventID(), lorentz3DFitResults);
        }
        
    } else {
      // Not enough data points for 3D Lorentzian fitting or 3D Lorentzian fitting is disabled
      if (Constants::ENABLE_3D_LORENTZIAN_FITTING) {
        fRunAction->Set3DLorentzianFitResults(
          0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amplitude, vertical_offset
          0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, amplitude_err, vertical_offset_err
          0, 0, 0,           // chi2red, pp, dof
          0,                 // charge_uncertainty
          false);            // fit_successful = false
      }
    }
      
      // ===============================================
      // 3D GAUSSIAN FITTING (conditionally enabled)
      // ===============================================
      
      // Perform 3D Gaussian fitting if we have enough data points and 3D Gaussian fitting is enabled
      if (x_coords.size() >= 6 && Constants::ENABLE_3D_GAUSSIAN_FITTING) { // Need at least 6 points for 3D Gaussian fit
        // Perform 3D Gaussian fitting using the Ceres Solver implementation
        GaussianFit3DResultsCeres gauss3DFitResults = Fit3DGaussianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (gauss3DFitResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass 3D Gaussian fit results to RunAction
        fRunAction->Set3DGaussianFitResults(
          gauss3DFitResults.center_x, gauss3DFitResults.center_y, 
          gauss3DFitResults.sigma_x, gauss3DFitResults.sigma_y, 
          gauss3DFitResults.amplitude, gauss3DFitResults.vertical_offset,
          gauss3DFitResults.center_x_err, gauss3DFitResults.center_y_err,
          gauss3DFitResults.sigma_x_err, gauss3DFitResults.sigma_y_err,
          gauss3DFitResults.amplitude_err, gauss3DFitResults.vertical_offset_err,
          gauss3DFitResults.chi2red, gauss3DFitResults.pp, gauss3DFitResults.dof,
          gauss3DFitResults.charge_uncertainty,
          gauss3DFitResults.fit_successful);
        
        // Log 3D Gaussian fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DGaussianFitResults(event->GetEventID(), gauss3DFitResults);
        }
        
    } else {
      // Not enough data points for 3D Gaussian fitting or 3D Gaussian fitting is disabled
      if (Constants::ENABLE_3D_GAUSSIAN_FITTING) {
        fRunAction->Set3DGaussianFitResults(
          0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amplitude, vertical_offset
          0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, sigma_x_err, sigma_y_err, amplitude_err, vertical_offset_err
          0, 0, 0,           // chi2red, pp, dof
          0,                 // charge_uncertainty
          false);            // fit_successful = false
      }
    }
      
      // ===============================================
      // 3D POWER-LAW LORENTZIAN FITTING (conditionally enabled)
      // ===============================================
      
      // Perform 3D Power-Law Lorentzian fitting if we have enough data points and 3D Power-Law Lorentzian fitting is enabled
      if (x_coords.size() >= 7 && Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) { // Need at least 7 points for 3D Power-Law Lorentzian fit
        // Perform 3D Power-Law Lorentzian fitting using the Ceres Solver implementation
        PowerLorentzianFit3DResultsCeres powerLorentz3DFitResults = Fit3DPowerLorentzianCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentz3DFitResults.fit_successful) {
          // Removed verbose debug output for cleaner simulation logs
        }
        
        // Pass 3D Power-Law Lorentzian fit results to RunAction
        fRunAction->Set3DPowerLorentzianFitResults(
          powerLorentz3DFitResults.center_x, powerLorentz3DFitResults.center_y,
          powerLorentz3DFitResults.gamma_x, powerLorentz3DFitResults.gamma_y,
          powerLorentz3DFitResults.beta, powerLorentz3DFitResults.amplitude, 
          powerLorentz3DFitResults.vertical_offset,
          powerLorentz3DFitResults.center_x_err, powerLorentz3DFitResults.center_y_err,
          powerLorentz3DFitResults.gamma_x_err, powerLorentz3DFitResults.gamma_y_err,
          powerLorentz3DFitResults.beta_err, powerLorentz3DFitResults.amplitude_err, 
          powerLorentz3DFitResults.vertical_offset_err,
          powerLorentz3DFitResults.chi2red, powerLorentz3DFitResults.pp, powerLorentz3DFitResults.dof,
          powerLorentz3DFitResults.charge_uncertainty,
          powerLorentz3DFitResults.fit_successful);
        
        // Log 3D Power Lorentzian fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DPowerLorentzianFitResults(event->GetEventID(), powerLorentz3DFitResults);
        }
        
    } else {
      // Not enough data points for 3D Power-Law Lorentzian fitting or 3D Power-Law Lorentzian fitting is disabled
      if (Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) {
        fRunAction->Set3DPowerLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amplitude, vertical_offset
          0, 0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, beta_err, amplitude_err, vertical_offset_err
          0, 0, 0,              // chi2red, pp, dof
          0,                    // charge_uncertainty
          false);               // fit_successful = false
      }
    }

        
    } else {
      // Not enough data points for fitting or Gaussian fitting is disabled
      if (Constants::ENABLE_GAUSSIAN_FITTING) {
        fRunAction->Set2DGaussianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, sigma, amplitude, center_err, sigma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
          false); // fit_successful = false
        
        // Set default diagonal fit values when not enough data points
        fRunAction->SetDiagonalGaussianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amplitude, center_err, sigma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
      
      // Not enough data points for Lorentzian fitting or Lorentzian fitting is disabled
      if (Constants::ENABLE_LORENTZIAN_FITTING) {
        fRunAction->Set2DLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
          false); // fit_successful = false
        
        // Set default diagonal Lorentzian fit values when not enough data points
        fRunAction->SetDiagonalLorentzianFitResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_successful = false
      }
    }
  } else {
    // Skip fitting due to conditions not met
    if (isPixelHit) {
      // Removed verbose debug output for cleaner simulation logs
    } else if (fNonPixel_GridNeighborhoodChargeFractions.empty()) {
      // Removed verbose debug output for cleaner simulation logs
    }
    
    // Set default values (no fitting performed or fitting is disabled)
    if (Constants::ENABLE_GAUSSIAN_FITTING) {
      fRunAction->Set2DGaussianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, sigma, amplitude, center_err, sigma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
        false); // fit_successful = false
      
      // Set default diagonal fit values when fitting is skipped
      fRunAction->SetDiagonalGaussianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amplitude, center_err, sigma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_successful = false
    }
    
    // Set default Lorentzian values (no fitting performed or Lorentzian fitting is disabled)
    if (Constants::ENABLE_LORENTZIAN_FITTING) {
      fRunAction->Set2DLorentzianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
        false); // fit_successful = false
      
      // Set default diagonal Lorentzian fit values when fitting is skipped
      fRunAction->SetDiagonalLorentzianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amplitude, center_err, gamma_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_successful = false
    }
    
    // Set default Power-Law Lorentzian values (no fitting performed or Power-Law Lorentzian fitting is disabled)
    if (Constants::ENABLE_POWER_LORENTZIAN_FITTING) {
      fRunAction->Set2DPowerLorentzianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, beta, amplitude, center_err, gamma_err, beta_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_uncertainty, y_charge_uncertainty)
        false); // fit_successful = false
      
      // Set default diagonal Power-Law Lorentzian fit values when fitting is skipped
      fRunAction->SetDiagonalPowerLorentzianFitResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amplitude, center_err, gamma_err, beta_err, amplitude_err, vertical_offset, vertical_offset_err, chi2red, pp, dof, fit_successful)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_successful = false
    }
    
    // Set default 3D Lorentzian values (no fitting performed or 3D Lorentzian fitting is disabled)
    if (Constants::ENABLE_3D_LORENTZIAN_FITTING) {
      fRunAction->Set3DLorentzianFitResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amplitude, vertical_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, amplitude_err, vertical_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_uncertainty
        false);            // fit_successful = false
    }
    
    // Set default 3D Gaussian values (no fitting performed or 3D Gaussian fitting is disabled)
    if (Constants::ENABLE_3D_GAUSSIAN_FITTING) {
      fRunAction->Set3DGaussianFitResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amplitude, vertical_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, sigma_x_err, sigma_y_err, amplitude_err, vertical_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_uncertainty
        false);            // fit_successful = false
    }
    
    // Set default 3D Power-Law Lorentzian values (no fitting performed or 3D Power-Law Lorentzian fitting is disabled)
    if (Constants::ENABLE_3D_POWER_LORENTZIAN_FITTING) {
      fRunAction->Set3DPowerLorentzianFitResults(
        0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amplitude, vertical_offset
        0, 0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, beta_err, amplitude_err, vertical_offset_err
        0, 0, 0,              // chi2red, pp, dof
        0,                    // charge_uncertainty
        false);               // fit_successful = false
    }

  }
  
  fRunAction->FillTree();
  
  // Log event end
  G4int eventID = event->GetEventID();
  SimulationLogger* logger = SimulationLogger::GetInstance();
  if (logger) {
    logger->LogEventEnd(eventID);
  }
  
  // Update crash recovery progress tracking - only every 100 events to reduce mutex contention
  // The auto-save functionality in CrashHandler will still work at its configured intervals
  if (eventID % 100 == 0) {
    CrashHandler::GetInstance().UpdateProgress(eventID);
  }
}

void EventAction::AddEdep(G4double edep, G4ThreeVector position)
{
  // Accumulate energy deposited while particle travels through detector volume
  // Energy weighted position calculation
  if (edep > 0) {
    if (!fHasHit) {
      fPosition = position * edep;
      fEdep = edep;  // First energy deposit in detector
      fHasHit = true;
    } else {
      // Weight position by energy deposition and sum total energy
      fPosition = (fPosition * fEdep + position * edep) / (fEdep + edep);
      fEdep += edep;  // Accumulate total energy deposited in detector
    }
  }
}

// Implementation of the new method to set the initial position
void EventAction::SetInitialPosition(const G4ThreeVector& position)
{
  fInitialPosition = position;
}

// Implementation of the nearest pixel calculation method
G4ThreeVector EventAction::CalculateNearestPixel(const G4ThreeVector& position)
{
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the position relative to the detector face
  G4ThreeVector relativePos = position - detectorPosition;
  
  // For the AC-LGAD, pixels are on the front surface (z > detector z)
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calculate which pixel grid position is closest (i and j indices)
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  // Clamp i and j to valid pixel indices
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calculate the actual pixel center position
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  // Pixels are on the detector front surface
  G4double pixelZ = detectorPosition.z() + Constants::DEFAULT_DETECTOR_WIDTH/2 + Constants::DEFAULT_PIXEL_WIDTH/2; // detector half-width + pixel half-width
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calculate and store distance from hit to pixel center (2D distance in detector plane)
  G4double dx = position.x() - pixelX;
  G4double dy = position.y() - pixelY;
  fActualPixelDistance = std::sqrt(dx*dx + dy*dy);
  
  // Calculate and store delta values (pixel center - true position)
  fPixelTrueDeltaX = pixelX - position.x();
  fPixelTrueDeltaY = pixelY - position.y();
  
  // Determine if the hit was on a pixel using the detector's method
  fPixelHit = fDetector->IsPositionOnPixel(position);
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Calculate the angular size of pixel from hit position
G4double EventAction::CalculatePixelAlpha(const G4ThreeVector& hitPosition, G4int pixelI, G4int pixelJ)
{
  // Check if the hit is inside the pixel. If so, return NaN to indicate no alpha calculation
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    return std::numeric_limits<G4double>::quiet_NaN(); // Return NaN for hits inside pixels (no alpha calculation needed)
  }

  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calculate the center position of the specified pixel
  G4double pixelCenterX = firstPixelPos + pixelI * pixelSpacing;
  G4double pixelCenterY = firstPixelPos + pixelJ * pixelSpacing;
  G4double pixelCenterZ = detectorPosition.z();
  
  // Calculate distance from hit position to pixel center (2D distance in XY plane)
  G4double dx = hitPosition.x() - pixelCenterX;
  G4double dy = hitPosition.y() - pixelCenterY;
  G4double d = std::sqrt(dx*dx + dy*dy);
  
  // Use the pixel size as l (side of the pixel pad)
  G4double l = pixelSize;
  
  // Apply the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
  G4double numerator = (l/2.0) * std::sqrt(2.0);
  G4double denominator = numerator + d;
  
  // Handle edge case where denominator could be very small
  G4double alpha;
  if (denominator < Constants::MIN_DENOMINATOR_VALUE) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  // Convert to degrees for storage
  G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
  
  return alphaInDegrees;
}

// Calculate angles from hit position to all pixels in a neighborhood grid around the hit pixel
void EventAction::CalculateNeighborhoodGridAngles(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ)
{
  // Clear previous data
  fNonPixel_GridNeighborhoodAngles.clear();
  
  // Check if hit is inside a pixel - if so, all angles should be invalid
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(hitPosition);
  if (isInsidePixel) {
    // Fill all positions with NaN for inside-pixel hits
    G4int gridSize = 2 * fNeighborhoodRadius + 1;
    G4int totalPixels = gridSize * gridSize;
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        // Calculate the pixel indices for this grid position
        G4int gridPixelI = hitPixelI + di;
        G4int gridPixelJ = hitPixelJ + dj;
        
        // Store NaN angle
        fNonPixel_GridNeighborhoodAngles.push_back(std::numeric_limits<G4double>::quiet_NaN()); // Use same NaN as elsewhere
      }
    }
    return; // Exit early for inside-pixel hits
  }
  
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = hitPixelI + di;
      G4int gridPixelJ = hitPixelJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNonPixel_GridNeighborhoodAngles.push_back(-999.0); // Invalid angle marker
        continue;
      }
      
      // Calculate the center position of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calculate the alpha angle for this pixel using the same algorithm as the Python demo
      G4double alpha = CalculatePixelAlphaSubtended(hitPosition.x(), hitPosition.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Convert to degrees
      G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
      
      // Store the results
      fNonPixel_GridNeighborhoodAngles.push_back(alphaInDegrees);
    }
  }
}

// Calculate the angular size subtended by a pixel as seen from a hit point (2D calculation)
// This now uses the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
G4double EventAction::CalculatePixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calculate distance from hit position to pixel center (2D distance in XY plane)
  G4double dx = hitX - pixelCenterX;
  G4double dy = hitY - pixelCenterY;
  G4double d = std::sqrt(dx*dx + dy*dy);
  
  // Use the pixel size as l (side of the pixel pad)
  // For simplicity, use the average of width and height if they differ
  G4double l = (pixelWidth + pixelHeight) / 2.0;
  
  // Apply the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
  G4double numerator = (l/2.0) * std::sqrt(2.0);
  G4double denominator = numerator + d;
  
  // Handle edge case where denominator could be very small
  G4double alpha;
  if (denominator < Constants::MIN_DENOMINATOR_VALUE) {
    alpha = CLHEP::pi/2.0;  // Maximum possible angle (90 degrees)
  } else {
    alpha = std::atan(numerator / denominator);
  }
  
  return alpha; // Return in radians
}

// Calculate charge sharing for pixels in a neighborhood grid around the hit pixel
void EventAction::CalculateNeighborhoodChargeSharing()
{
  // Clear previous data
  fNonPixel_GridNeighborhoodChargeFractions.clear();
  fNonPixel_GridNeighborhoodDistances.clear();
  fNonPixel_GridNeighborhoodCharge.clear();
  
  // Check if no energy was deposited
  if (fEdep <= 0) {
    // Fill all positions with zero for no-energy events
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        fNonPixel_GridNeighborhoodChargeFractions.push_back(0.0);
        fNonPixel_GridNeighborhoodDistances.push_back(-999.0);
        fNonPixel_GridNeighborhoodCharge.push_back(0.0);
      }
    }
    return;
  }
  
  // Check if hit is inside a pixel - if so, assign zero charge (per user requirement)
  G4bool isInsidePixel = fDetector->IsPositionOnPixel(fPosition);
  if (isInsidePixel) {
    // For pixel hits, energy deposition should be zero (per user requirement)
    // Therefore, charge is also zero
    G4double totalCharge = 0.0;
    
    // Fill all positions, giving all charge to the hit pixel and zero to others
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this is the pixel that was hit
        if (di == 0 && dj == 0) {
          // This is the center pixel (the one that was hit) - but charge is zero for pixel hits
          fNonPixel_GridNeighborhoodChargeFractions.push_back(0.0);
          fNonPixel_GridNeighborhoodCharge.push_back(0.0);
          fNonPixel_GridNeighborhoodDistances.push_back(0.0); // Distance to center of hit pixel is effectively zero
        } else if (gridPixelI >= 0 && gridPixelI < fDetector->GetNumBlocksPerSide() && 
                   gridPixelJ >= 0 && gridPixelJ < fDetector->GetNumBlocksPerSide()) {
          // This is a valid pixel in the detector but not the hit pixel
          fNonPixel_GridNeighborhoodChargeFractions.push_back(0.0);
          fNonPixel_GridNeighborhoodCharge.push_back(0.0);
          
        // Calculate distance to this pixel center for completeness
        G4double pixelSize = fDetector->GetPixelSize();
        G4double pixelSpacing = fDetector->GetPixelSpacing();
        G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
        G4double detSize = fDetector->GetDetSize();
        G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
        
        G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
        G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
        G4double dx = fPosition.x() - pixelCenterX;
        G4double dy = fPosition.y() - pixelCenterY;
        G4double distance = std::sqrt(dx*dx + dy*dy);
          fNonPixel_GridNeighborhoodDistances.push_back(distance);
        } else {
          // This pixel is outside the detector bounds
          fNonPixel_GridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fNonPixel_GridNeighborhoodDistances.push_back(-999.0);
          fNonPixel_GridNeighborhoodCharge.push_back(0.0);
        }
      }
    }
    return;
  }
  
  // Convert energy deposit to number of electrons
  // fEdep is in MeV, fIonizationEnergy is in eV
  // Use explicit CLHEP units instead of manual conversion
  G4double edepInEV = fEdep * MeV / eV; // Convert MeV to eV using CLHEP units
  G4double numElectrons = edepInEV / fIonizationEnergy;
  
  // Apply AC-LGAD amplification
  G4double totalCharge = numElectrons * fAmplificationFactor;
  
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPosition = fDetector->GetDetectorPosition();
  
  // Calculate the first pixel position (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // D0 constant for charge sharing formula (convert to consistent units)
  // fD0 is in microns, distances are in mm, so convert using CLHEP units
  G4double d0_mm = fD0 * micrometer / mm; // Convert microns to mm using CLHEP units
  
  // Proceed with charge sharing calculation for non-pixel hits
  // First pass: collect valid pixels and calculate weights
  std::vector<G4double> weights;
  std::vector<G4double> distances;
  std::vector<G4double> angles;
  std::vector<G4int> validPixelI;
  std::vector<G4int> validPixelJ;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNonPixel_GridNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
        fNonPixel_GridNeighborhoodDistances.push_back(-999.0);
        fNonPixel_GridNeighborhoodCharge.push_back(0.0);
        continue;
      }
      
      // Calculate the center position of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calculate the distance from the hit to the pixel center (in mm)
      G4double dx = fPosition.x() - pixelCenterX;
      G4double dy = fPosition.y() - pixelCenterY;
      G4double distance = std::sqrt(dx*dx + dy*dy);
      
      // Calculate the alpha angle for this pixel using the same algorithm as elsewhere
      G4double alpha = CalculatePixelAlphaSubtended(fPosition.x(), fPosition.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Store data for valid pixels
      distances.push_back(distance);
      angles.push_back(alpha);
      validPixelI.push_back(gridPixelI);
      validPixelJ.push_back(gridPixelJ);
      
      // Calculate weight according to formula: α_i * ln(d_i/d_0)^(-1)
      // Handle the case where distance might be very small or zero
      G4double weight = 0.0;
      if (distance > d0_mm) {
        // Clamp the logarithm to avoid infinite weights when d ≈ d0
        G4double logArg = distance / d0_mm;
        G4double logValue = std::log(logArg);
        // Clamp log value to avoid division by very small numbers
        logValue = std::max(logValue, Constants::MIN_LOG_VALUE);
        weight = alpha * (1.0 / logValue);
      } else if (distance > 0) {
        // For very small distances, use a large weight
        weight = alpha * Constants::ALPHA_WEIGHT_MULTIPLIER; // Large weight for very close pixels
      } else {
        // Distance is zero (hit exactly on pixel center), give maximum weight
        weight = alpha * Constants::ALPHA_WEIGHT_MULTIPLIER;
      }
      
      weights.push_back(weight);
    }
  }
  
  // Calculate total weight
  G4double totalWeight = 0.0;
  for (G4double weight : weights) {
    totalWeight += weight;
  }
  
  // Second pass: calculate charge fractions and values
  size_t validIndex = 0;
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calculate the pixel indices for this grid position
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Already stored invalid data in first pass
        continue;
      }
      
      // Calculate charge fraction and value
      G4double chargeFraction = 0.0;
      G4double chargeValue = 0.0;
      
      if (totalWeight > 0) {
        chargeFraction = weights[validIndex] / totalWeight;
        chargeValue = chargeFraction * totalCharge;
      }
      
      fNonPixel_GridNeighborhoodChargeFractions.push_back(chargeFraction);
      fNonPixel_GridNeighborhoodDistances.push_back(distances[validIndex]);
      fNonPixel_GridNeighborhoodCharge.push_back(chargeValue * fElementaryCharge);
      
      validIndex++;
    }
  }
}

// Perform automatic radius selection based on fit quality
G4int EventAction::SelectOptimalRadius(const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ)
{
  G4double bestFitQuality = -1.0;
  G4int bestRadius = fNeighborhoodRadius; // Default fallback
  
  // Test different radii from min to max
  for (G4int testRadius = fMinAutoRadius; testRadius <= fMaxAutoRadius; testRadius++) {
    G4double fitQuality = EvaluateFitQuality(testRadius, hitPosition, hitPixelI, hitPixelJ);
    
    if (fitQuality > bestFitQuality) {
      bestFitQuality = fitQuality;
      bestRadius = testRadius;
    }
  }
  
  // Store the best fit quality found
  fSelectedFitQuality = bestFitQuality;
  
  return bestRadius;
}

// Evaluate fit quality for a given radius
G4double EventAction::EvaluateFitQuality(G4int radius, const G4ThreeVector& hitPosition, G4int hitPixelI, G4int hitPixelJ)
{
  // Temporarily save current radius
  G4int originalRadius = fNeighborhoodRadius;
  fNeighborhoodRadius = radius;
  
  // Clear and recalculate neighborhood data with test radius
  std::vector<G4double> tempAngles = fNonPixel_GridNeighborhoodAngles;
  std::vector<G4double> tempChargeFractions = fNonPixel_GridNeighborhoodChargeFractions;
  std::vector<G4double> tempDistances = fNonPixel_GridNeighborhoodDistances;
  std::vector<G4double> tempCharge = fNonPixel_GridNeighborhoodCharge;
  
  // Calculate neighborhood data with test radius
  CalculateNeighborhoodGridAngles(hitPosition, hitPixelI, hitPixelJ);
  CalculateNeighborhoodChargeSharing();
  
  // Check if we have enough data points for fitting
  G4int validPoints = 0;
  for (size_t i = 0; i < fNonPixel_GridNeighborhoodChargeFractions.size(); ++i) {
    if (fNonPixel_GridNeighborhoodChargeFractions[i] > 0) {
      validPoints++;
    }
  }
  
  if (validPoints < Constants::MIN_POINTS_FOR_FIT) {
    // Restore original data and radius
    fNeighborhoodRadius = originalRadius;
    fNonPixel_GridNeighborhoodAngles = tempAngles;
    fNonPixel_GridNeighborhoodChargeFractions = tempChargeFractions;
    fNonPixel_GridNeighborhoodDistances = tempDistances;
    fNonPixel_GridNeighborhoodCharge = tempCharge;
    return 0.0; // Poor quality due to insufficient data
  }
  
  // Extract data for fitting
  std::vector<double> x_coords, y_coords, charge_values;
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4int gridSize = 2 * radius + 1;
  G4ThreeVector nearestPixel = CalculateNearestPixel(hitPosition);
  
  for (size_t i = 0; i < fNonPixel_GridNeighborhoodChargeFractions.size(); ++i) {
    if (fNonPixel_GridNeighborhoodChargeFractions[i] > 0) {
      G4int col = i / gridSize;  // di (X) was outer loop
      G4int row = i % gridSize;  // dj (Y) was inner loop
      
      G4int offsetI = col - radius; // X offset from center pixel
      G4int offsetJ = row - radius; // Y offset from center pixel
      
      G4double x_pos = nearestPixel.x() + offsetI * pixelSpacing;
      G4double y_pos = nearestPixel.y() + offsetJ * pixelSpacing;
      
      x_coords.push_back(x_pos);
      y_coords.push_back(y_pos);
      charge_values.push_back(fNonPixel_GridNeighborhoodCharge[i]);
    }
  }
  
  G4double fitQuality = 0.0;
  
  if (x_coords.size() >= Constants::MIN_POINTS_FOR_FIT) {
    // Perform quick Gaussian fit evaluation
    GaussianFit2DResultsCeres fitResults = Fit2DGaussianCeres(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, 
      false, // verbose=false
      false); // enable_outlier_filtering
    
    if (fitResults.fit_successful) {
      // Calculate fit quality based on residuals and fit parameters
      // Use reduced chi-squared values and goodness of fit metrics
      G4double rowQuality = (fitResults.x_chi2red > 0) ? 1.0 / (1.0 + fitResults.x_chi2red) : 0.0;
      G4double colQuality = (fitResults.y_chi2red > 0) ? 1.0 / (1.0 + fitResults.y_chi2red) : 0.0;
      
      // Average the quality from row and column fits
      fitQuality = (rowQuality + colQuality) / 2.0;
      
      // Apply penalty for outliers based on fit probability
      G4double probPenalty = (fitResults.x_pp + fitResults.y_pp) / 2.0;
      fitQuality *= probPenalty;
      
      // Clamp to [0, 1] range
      fitQuality = std::max(0.0, std::min(1.0, fitQuality));
    }
  }
  
  // Restore original data and radius
  fNeighborhoodRadius = originalRadius;
  fNonPixel_GridNeighborhoodAngles = tempAngles;
  fNonPixel_GridNeighborhoodChargeFractions = tempChargeFractions;
  fNonPixel_GridNeighborhoodDistances = tempDistances;
  fNonPixel_GridNeighborhoodCharge = tempCharge;
  
  return fitQuality;
}