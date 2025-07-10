#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "Constants.hh"
#include "CrashHandler.hh"
#include "SimulationLogger.hh"
#include "2DGaussCeres.hh"
#include "2DLorentzCeres.hh"
#include "2DPowerLorentzCeres.hh"
#include "3DGaussCeres.hh"
#include "3DLorentzCeres.hh"
#include "3DPowerLorentzCeres.hh"

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
  fPos(G4ThreeVector(0.,0.,0.)),
  fInitialPos(G4ThreeVector(0.,0.,0.)),
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
  fSelectedQuality(0.0),
  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE)
{ 
  G4cout << "EventAction: Using 2D Gauss fitting for central row and column" << G4endl;
}

EventAction::~EventAction()
{ 
  // No 3D Gauss fitter to clean up
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
  fPos = G4ThreeVector(0.,0.,0.);
  fHasHit = false;
  
  // Initialize particle pos - this will be updated when the primary vertex is created
  fInitialPos = G4ThreeVector(0.,0.,0.);
  
  // Reset pixel mapping variables
  fPixelIndexI = -1;
  fPixelIndexJ = -1;
  fPixelTrueDeltaX = 0;
  fPixelTrueDeltaY = 0;
  fActualPixelDistance = -1.;
  fPixelHit = false;
  
  // Reset neighborhood (9x9) grid angle data
  fNeighborhoodAngles.clear();
  
  // Reset neighborhood (9x9) grid charge sharing data
  fNeighborhoodChargeFractions.clear();
  fNeighborhoodDistances.clear();
  fNeighborhoodCharge.clear();
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // Get the primary vertex pos and energy from the event
  if (event->GetPrimaryVertex()) {
    G4ThreeVector primaryPos = event->GetPrimaryVertex()->GetPosition();
    // Update the initial pos
    fInitialPos = primaryPos;
    
    // Get the initial particle energy (kinetic energy)
    G4PrimaryParticle* primaryParticle = event->GetPrimaryVertex()->GetPrimary();
    if (primaryParticle) {
      G4double initialKineticEnergy = primaryParticle->GetKineticEnergy();
      // Store the initial energy in the RunAction for ROOT output
      // Convert from Geant4 internal units (MeV) to MeV for storage
      fRunAction->SetInitialEnergy(initialKineticEnergy);
    }
  }
  
  // Calc and store nearest pixel pos (this calculates fActualPixelDistance)
  G4ThreeVector nearestPixel = CalcNearestPixel(fPos);
  
  // Calc and pass pixel angular size to RunAction
  G4double pixelAlpha = CalcPixelAlpha(fPos, fPixelIndexI, fPixelIndexJ);
  
  // Determine if hit is a pixel hit: only based on whether it's on pixel surface
  G4bool isPixelHit = fPixelHit;
  
  // ENERGY DEPOSITION LOGIC:
  // - Energy is summed only while particle travels through detector volume (handled in SteppingAction)
  // - For pixel hits: set energy deposition to zero (per user requirement)  
  // - For non-pixel hits: use energy deposited inside detector during particle passage
  G4double finalEdep = fEdep;
  if (isPixelHit) {
    finalEdep = 0.0; // Set to zero for pixel hits (per user requirement)
  } else {
    // For non-pixel hits, use the energy deposited inside the detector (already accumulated in fEdep from SteppingAction)
  }
  
  // Pass classification data to RunAction
  fRunAction->SetPixelClassification(isPixelHit, fPixelTrueDeltaX, fPixelTrueDeltaY);
  fRunAction->SetPixelHitStatus(isPixelHit);
  
  // Update the event data with the corrected energy deposition
  fRunAction->SetEventData(finalEdep, fPos.x(), fPos.y(), fPos.z());
  fRunAction->SetInitialPos(fInitialPos.x(), fInitialPos.y(), fInitialPos.z());
  
  // Set nearest pixel pos in RunAction  
  fRunAction->SetNearestPixelPos(nearestPixel.x(), nearestPixel.y());
  
  // Only calculate and store pixel-specific data for pixel hits (on pixel surface)
  if (isPixelHit) {
    
    // Clear non-pixel data for pixel hits
    fNeighborhoodAngles.clear();
    fNeighborhoodChargeFractions.clear();
    fNeighborhoodDistances.clear();
    fNeighborhoodCharge.clear();
  } else {
    // Non-pixel hit: determine optimal radius and calculate neighborhood grid data
    if (fAutoRadiusEnabled) {
      // Perform automatic radius selection based on fit quality
      fSelectedRadius = SelectOptimalRadius(fPos, fPixelIndexI, fPixelIndexJ);
      fNeighborhoodRadius = fSelectedRadius;
      
      G4cout << "EventAction: Auto-selected radius " << fSelectedRadius 
             << " with fit quality " << fSelectedQuality << G4endl;
    } else {
      // Use fixed radius
      fSelectedRadius = fNeighborhoodRadius;
      fSelectedQuality = 0.0; // Not evaluated
    }
    
    // Calc neighborhood grid data with selected radius
    CalcNeighborhoodGridAngles(fPos, fPixelIndexI, fPixelIndexJ);
    CalcNeighborhoodChargeSharing();
  }
  
  // Pass neighborhood grid data to RunAction (will be empty for pixel hits)
  fRunAction->SetNeighborhoodGridData(fNeighborhoodAngles);
  fRunAction->SetNeighborhoodChargeData(fNeighborhoodChargeFractions, fNeighborhoodDistances, fNeighborhoodCharge, fNeighborhoodCharge);
  
  // Pass automatic radius selection results to RunAction
  fRunAction->SetAutoRadiusResults(fSelectedRadius);
  
  // Perform 2D Gauss fitting on charge distribution data (central row and column)
  // Only fit for non-pixel hits (not on pixel surface)
  G4bool shouldPerform = !isPixelHit && !fNeighborhoodChargeFractions.empty();
  
  if (shouldPerform) {
    // Extract coordinates and charge values for fitting
    std::vector<double> x_coords, y_coords, charge_values;
    
    // Get detector parameters for coordinate calculation
    G4double pixelSpacing = fDetector->GetPixelSpacing();
    
    // Convert grid indices to actual coordinates
    // The neighborhood grid is a systematic 9x9 grid around the center pixel
    G4int gridSize = 2 * fNeighborhoodRadius + 1; // Should be 9 for radius 4
    for (size_t i = 0; i < fNeighborhoodChargeFractions.size(); ++i) {
      if (fNeighborhoodChargeFractions[i] > 0) { // Only include pixels with charge
        // Calc grid pos from array index
        // The grid is stored in column-major order: i = col * gridSize + row
        // because di (X) is outer loop, dj (Y) is inner loop in charge calculation
        G4int col = i / gridSize;  // di (X) was outer loop
        G4int row = i % gridSize;  // dj (Y) was inner loop
        
        // Convert grid pos to pixel offset from center
        G4int offsetI = col - fNeighborhoodRadius; // -4 to +4 for 9x9 grid (X offset)
        G4int offsetJ = row - fNeighborhoodRadius; // -4 to +4 for 9x9 grid (Y offset)
        
        // Calc actual pos
        G4double x_pos = nearestPixel.x() + offsetI * pixelSpacing;
        G4double y_pos = nearestPixel.y() + offsetJ * pixelSpacing;
        
        x_coords.push_back(x_pos);
        y_coords.push_back(y_pos);
        // Use actual charge values (in Coulombs) instead of fractions for fitting
        charge_values.push_back(fNeighborhoodCharge[i]);
      }
    }
    
    // ===============================================
    // GAUSS FIT (conditionally enabled)
    // ===============================================
    
    // Perform 2D fitting if we have enough data points and Gauss fitting is enabled
    if (x_coords.size() >= 3 && Constants::ENABLE_GAUSS_FIT && Constants::ENABLE_ROWCOL_FIT) { // Need at least 3 points for 1D Gauss fit
      // Perform 2D Gauss fitting using the Ceres Solver implementation
      Gauss2DResultsCeres fitResults = GaussCeres2D(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, 
        false, // verbose=false for production
        false); // enable_outlier_filtering
      
      if (fitResults.fit_success) {
      }
      
      // Pass 2D fit results to RunAction
      fRunAction->Set2DGaussResults(
        fitResults.x_center, fitResults.x_sigma, fitResults.x_amp,
        fitResults.x_center_err, fitResults.x_sigma_err, fitResults.x_amp_err,
        fitResults.x_vert_offset, fitResults.x_vert_offset_err,
        fitResults.x_chi2red, fitResults.x_pp, fitResults.x_dof,
        fitResults.y_center, fitResults.y_sigma, fitResults.y_amp,
        fitResults.y_center_err, fitResults.y_sigma_err, fitResults.y_amp_err,
        fitResults.y_vert_offset, fitResults.y_vert_offset_err,
        fitResults.y_chi2red, fitResults.y_pp, fitResults.y_dof,
        fitResults.x_charge_err, fitResults.y_charge_err,
        fitResults.fit_success);
      
      // Log Gauss fitting results to SimulationLogger
      SimulationLogger* logger = SimulationLogger::GetInstance();
      if (logger) {
        logger->LogGaussResults(event->GetEventID(), fitResults);
      }
        
        // Perform diagonal fitting if 2D fitting was performed and success and diagonal fitting is enabled
        if (fitResults.fit_success && Constants::ENABLE_DIAG_FIT) {
          // Perform diagonal Gauss fitting using the Ceres Solver implementation
          DiagResultsCeres diagResults = DiagGaussCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (diagResults.fit_success) {
        }
        
        // Pass diagonal fit results to RunAction
        fRunAction->SetDiagGaussResults(
          diagResults.main_diag_x_center, diagResults.main_diag_x_sigma, diagResults.main_diag_x_amp,
          diagResults.main_diag_x_center_err, diagResults.main_diag_x_sigma_err, diagResults.main_diag_x_amp_err,
          diagResults.main_diag_x_vert_offset, diagResults.main_diag_x_vert_offset_err,
          diagResults.main_diag_x_chi2red, diagResults.main_diag_x_pp, diagResults.main_diag_x_dof, diagResults.main_diag_x_fit_success,
          diagResults.main_diag_y_center, diagResults.main_diag_y_sigma, diagResults.main_diag_y_amp,
          diagResults.main_diag_y_center_err, diagResults.main_diag_y_sigma_err, diagResults.main_diag_y_amp_err,
          diagResults.main_diag_y_vert_offset, diagResults.main_diag_y_vert_offset_err,
          diagResults.main_diag_y_chi2red, diagResults.main_diag_y_pp, diagResults.main_diag_y_dof, diagResults.main_diag_y_fit_success,
          diagResults.sec_diag_x_center, diagResults.sec_diag_x_sigma, diagResults.sec_diag_x_amp,
          diagResults.sec_diag_x_center_err, diagResults.sec_diag_x_sigma_err, diagResults.sec_diag_x_amp_err,
          diagResults.sec_diag_x_vert_offset, diagResults.sec_diag_x_vert_offset_err,
          diagResults.sec_diag_x_chi2red, diagResults.sec_diag_x_pp, diagResults.sec_diag_x_dof, diagResults.sec_diag_x_fit_success,
          diagResults.sec_diag_y_center, diagResults.sec_diag_y_sigma, diagResults.sec_diag_y_amp,
          diagResults.sec_diag_y_center_err, diagResults.sec_diag_y_sigma_err, diagResults.sec_diag_y_amp_err,
          diagResults.sec_diag_y_vert_offset, diagResults.sec_diag_y_vert_offset_err,
          diagResults.sec_diag_y_chi2red, diagResults.sec_diag_y_pp, diagResults.sec_diag_y_dof, diagResults.sec_diag_y_fit_success,
          diagResults.fit_success);
      } else {
        // Set default diagonal fit values when 2D fitting failed
        fRunAction->SetDiagGaussResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amp, center_err, sigma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
      
      // ===============================================
      // LORENTZ FIT (conditionally enabled)
      // ===============================================
      
      // Perform 2D Lorentz fitting if we have enough data points and Lorentz fitting is enabled
      if (x_coords.size() >= 3 && Constants::ENABLE_LORENTZ_FIT && Constants::ENABLE_ROWCOL_FIT) { // Need at least 3 points for 1D Lorentz fit
        // Perform 2D Lorentz fitting using the Ceres Solver implementation
        Lorentz2DResultsCeres lorentzResults = LorentzCeres2D(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentzResults.fit_success) {
        }
        
        // Pass 2D Lorentz fit results to RunAction
        fRunAction->Set2DLorentzResults(
          lorentzResults.x_center, lorentzResults.x_gamma, lorentzResults.x_amp,
          lorentzResults.x_center_err, lorentzResults.x_gamma_err, lorentzResults.x_amp_err,
          lorentzResults.x_vert_offset, lorentzResults.x_vert_offset_err,
          lorentzResults.x_chi2red, lorentzResults.x_pp, lorentzResults.x_dof,
          lorentzResults.y_center, lorentzResults.y_gamma, lorentzResults.y_amp,
          lorentzResults.y_center_err, lorentzResults.y_gamma_err, lorentzResults.y_amp_err,
          lorentzResults.y_vert_offset, lorentzResults.y_vert_offset_err,
          lorentzResults.y_chi2red, lorentzResults.y_pp, lorentzResults.y_dof,
          lorentzResults.x_charge_err, lorentzResults.y_charge_err,
          lorentzResults.fit_success);
        
        // Log Lorentz fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->LogLorentzResults(event->GetEventID(), lorentzResults);
        }
          
        // Note: Lorentz charge error data was removed as per user request
          
        // Perform diagonal Lorentz fitting if 2D fitting was performed and success and diagonal fitting is enabled
        if (lorentzResults.fit_success && Constants::ENABLE_DIAG_FIT) {
          // Perform diagonal Lorentz fitting using the Ceres Solver implementation
          DiagLorentzResultsCeres lorentzDiagResults = DiagLorentzCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentzDiagResults.fit_success) {
        }
        
        // Pass diagonal Lorentz fit results to RunAction
        fRunAction->SetDiagLorentzResults(
          lorentzDiagResults.main_diag_x_center, lorentzDiagResults.main_diag_x_gamma, lorentzDiagResults.main_diag_x_amp,
          lorentzDiagResults.main_diag_x_center_err, lorentzDiagResults.main_diag_x_gamma_err, lorentzDiagResults.main_diag_x_amp_err,
          lorentzDiagResults.main_diag_x_vert_offset, lorentzDiagResults.main_diag_x_vert_offset_err,
          lorentzDiagResults.main_diag_x_chi2red, lorentzDiagResults.main_diag_x_pp, lorentzDiagResults.main_diag_x_dof, lorentzDiagResults.main_diag_x_fit_success,
          lorentzDiagResults.main_diag_y_center, lorentzDiagResults.main_diag_y_gamma, lorentzDiagResults.main_diag_y_amp,
          lorentzDiagResults.main_diag_y_center_err, lorentzDiagResults.main_diag_y_gamma_err, lorentzDiagResults.main_diag_y_amp_err,
          lorentzDiagResults.main_diag_y_vert_offset, lorentzDiagResults.main_diag_y_vert_offset_err,
          lorentzDiagResults.main_diag_y_chi2red, lorentzDiagResults.main_diag_y_pp, lorentzDiagResults.main_diag_y_dof, lorentzDiagResults.main_diag_y_fit_success,
          lorentzDiagResults.sec_diag_x_center, lorentzDiagResults.sec_diag_x_gamma, lorentzDiagResults.sec_diag_x_amp,
          lorentzDiagResults.sec_diag_x_center_err, lorentzDiagResults.sec_diag_x_gamma_err, lorentzDiagResults.sec_diag_x_amp_err,
          lorentzDiagResults.sec_diag_x_vert_offset, lorentzDiagResults.sec_diag_x_vert_offset_err,
          lorentzDiagResults.sec_diag_x_chi2red, lorentzDiagResults.sec_diag_x_pp, lorentzDiagResults.sec_diag_x_dof, lorentzDiagResults.sec_diag_x_fit_success,
          lorentzDiagResults.sec_diag_y_center, lorentzDiagResults.sec_diag_y_gamma, lorentzDiagResults.sec_diag_y_amp,
          lorentzDiagResults.sec_diag_y_center_err, lorentzDiagResults.sec_diag_y_gamma_err, lorentzDiagResults.sec_diag_y_amp_err,
          lorentzDiagResults.sec_diag_y_vert_offset, lorentzDiagResults.sec_diag_y_vert_offset_err,
          lorentzDiagResults.sec_diag_y_chi2red, lorentzDiagResults.sec_diag_y_pp, lorentzDiagResults.sec_diag_y_dof, lorentzDiagResults.sec_diag_y_fit_success,
          lorentzDiagResults.fit_success);
          
        // Note: Diag Lorentz charge error data was removed as per user request
      } else {
        // Set default diagonal Lorentz fit values when 2D fitting failed
        fRunAction->SetDiagLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
        
    } else {
      // Not enough data points for Lorentz fitting or Lorentz fitting is disabled
      if (Constants::ENABLE_LORENTZ_FIT) {
        fRunAction->Set2DLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
          false); // fit_success = false
        
        // Set default diagonal Lorentz fit values when not enough data points
        fRunAction->SetDiagLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
    }
      
      // ===============================================
      // POWER-LAW LORENTZ FIT (conditionally enabled)
      // ===============================================
      
      // Perform 2D Power-Law Lorentz fitting if we have enough data points and Power-Law Lorentz fitting is enabled
      if (x_coords.size() >= 3 && Constants::ENABLE_POWER_LORENTZ_FIT && Constants::ENABLE_ROWCOL_FIT) { // Need at least 3 points for Power-Law Lorentz fit
        // Perform 2D Power-Law Lorentz fitting using the Ceres Solver implementation
        PowerLorentz2DResultsCeres powerLorentzResults = PowerLorentzCeres2D(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentzResults.fit_success) {
        }
        
        // Pass 2D Power-Law Lorentz fit results to RunAction
        fRunAction->Set2DPowerLorentzResults(
          powerLorentzResults.x_center, powerLorentzResults.x_gamma, powerLorentzResults.x_beta, powerLorentzResults.x_amp,
          powerLorentzResults.x_center_err, powerLorentzResults.x_gamma_err, powerLorentzResults.x_beta_err, powerLorentzResults.x_amp_err,
          powerLorentzResults.x_vert_offset, powerLorentzResults.x_vert_offset_err,
          powerLorentzResults.x_chi2red, powerLorentzResults.x_pp, powerLorentzResults.x_dof,
          powerLorentzResults.y_center, powerLorentzResults.y_gamma, powerLorentzResults.y_beta, powerLorentzResults.y_amp,
          powerLorentzResults.y_center_err, powerLorentzResults.y_gamma_err, powerLorentzResults.y_beta_err, powerLorentzResults.y_amp_err,
          powerLorentzResults.y_vert_offset, powerLorentzResults.y_vert_offset_err,
          powerLorentzResults.y_chi2red, powerLorentzResults.y_pp, powerLorentzResults.y_dof,
          powerLorentzResults.x_charge_err, powerLorentzResults.y_charge_err,
          powerLorentzResults.fit_success);
        
        // Log Power Lorentz fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->LogPowerLorentzResults(event->GetEventID(), powerLorentzResults);
        }
          
        // Perform diagonal Power-Law Lorentz fitting if 2D fitting was performed and success and diagonal fitting is enabled
        if (powerLorentzResults.fit_success && Constants::ENABLE_DIAG_FIT) {
          // Perform diagonal Power-Law Lorentz fitting using the Ceres Solver implementation
          DiagPowerLorentzResultsCeres powerLorentzDiagResults = DiagPowerLorentzCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentzDiagResults.fit_success) {
        }
        
        // Pass diagonal Power-Law Lorentz fit results to RunAction
        fRunAction->SetDiagPowerLorentzResults(
          powerLorentzDiagResults.main_diag_x_center, powerLorentzDiagResults.main_diag_x_gamma, powerLorentzDiagResults.main_diag_x_beta, powerLorentzDiagResults.main_diag_x_amp,
          powerLorentzDiagResults.main_diag_x_center_err, powerLorentzDiagResults.main_diag_x_gamma_err, powerLorentzDiagResults.main_diag_x_beta_err, powerLorentzDiagResults.main_diag_x_amp_err,
          powerLorentzDiagResults.main_diag_x_vert_offset, powerLorentzDiagResults.main_diag_x_vert_offset_err,
          powerLorentzDiagResults.main_diag_x_chi2red, powerLorentzDiagResults.main_diag_x_pp, powerLorentzDiagResults.main_diag_x_dof, powerLorentzDiagResults.main_diag_x_fit_success,
          powerLorentzDiagResults.main_diag_y_center, powerLorentzDiagResults.main_diag_y_gamma, powerLorentzDiagResults.main_diag_y_beta, powerLorentzDiagResults.main_diag_y_amp,
          powerLorentzDiagResults.main_diag_y_center_err, powerLorentzDiagResults.main_diag_y_gamma_err, powerLorentzDiagResults.main_diag_y_beta_err, powerLorentzDiagResults.main_diag_y_amp_err,
          powerLorentzDiagResults.main_diag_y_vert_offset, powerLorentzDiagResults.main_diag_y_vert_offset_err,
          powerLorentzDiagResults.main_diag_y_chi2red, powerLorentzDiagResults.main_diag_y_pp, powerLorentzDiagResults.main_diag_y_dof, powerLorentzDiagResults.main_diag_y_fit_success,
          powerLorentzDiagResults.sec_diag_x_center, powerLorentzDiagResults.sec_diag_x_gamma, powerLorentzDiagResults.sec_diag_x_beta, powerLorentzDiagResults.sec_diag_x_amp,
          powerLorentzDiagResults.sec_diag_x_center_err, powerLorentzDiagResults.sec_diag_x_gamma_err, powerLorentzDiagResults.sec_diag_x_beta_err, powerLorentzDiagResults.sec_diag_x_amp_err,
          powerLorentzDiagResults.sec_diag_x_vert_offset, powerLorentzDiagResults.sec_diag_x_vert_offset_err,
          powerLorentzDiagResults.sec_diag_x_chi2red, powerLorentzDiagResults.sec_diag_x_pp, powerLorentzDiagResults.sec_diag_x_dof, powerLorentzDiagResults.sec_diag_x_fit_success,
          powerLorentzDiagResults.sec_diag_y_center, powerLorentzDiagResults.sec_diag_y_gamma, powerLorentzDiagResults.sec_diag_y_beta, powerLorentzDiagResults.sec_diag_y_amp,
          powerLorentzDiagResults.sec_diag_y_center_err, powerLorentzDiagResults.sec_diag_y_gamma_err, powerLorentzDiagResults.sec_diag_y_beta_err, powerLorentzDiagResults.sec_diag_y_amp_err,
          powerLorentzDiagResults.sec_diag_y_vert_offset, powerLorentzDiagResults.sec_diag_y_vert_offset_err,
          powerLorentzDiagResults.sec_diag_y_chi2red, powerLorentzDiagResults.sec_diag_y_pp, powerLorentzDiagResults.sec_diag_y_dof, powerLorentzDiagResults.sec_diag_y_fit_success,
          powerLorentzDiagResults.fit_success);

      } else {
        // Set default diagonal Power-Law Lorentz fit values when 2D fitting failed
        fRunAction->SetDiagPowerLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amp, center_err, gamma_err, beta_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
        
    } else {
      // Not enough data points for Power-Law Lorentz fitting or Power-Law Lorentz fitting is disabled
      if (Constants::ENABLE_POWER_LORENTZ_FIT) {
        fRunAction->Set2DPowerLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, beta, amp, center_err, gamma_err, beta_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
          false); // fit_success = false
        
        // Set default diagonal Power-Law Lorentz fit values when not enough data points
        fRunAction->SetDiagPowerLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amp, center_err, gamma_err, beta_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
    }
      
      // ===============================================
      // 3D LORENTZ FIT (conditionally enabled)
      // ===============================================
      
      // Perform 3D Lorentz fitting if we have enough data points and 3D Lorentz fitting is enabled
      if (x_coords.size() >= 6 && Constants::ENABLE_3D_LORENTZ_FIT) { // Need at least 6 points for 3D Lorentz fit
        // Perform 3D Lorentz fitting using the Ceres Solver implementation
        Lorentz3DResultsCeres lorentz3DResults = LorentzCeres3D(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (lorentz3DResults.fit_success) {
        }
        
        // Pass 3D Lorentz fit results to RunAction
        fRunAction->Set3DLorentzResults(
          lorentz3DResults.center_x, lorentz3DResults.center_y, 
          lorentz3DResults.gamma_x, lorentz3DResults.gamma_y, 
          lorentz3DResults.amp, lorentz3DResults.vert_offset,
          lorentz3DResults.center_x_err, lorentz3DResults.center_y_err,
          lorentz3DResults.gamma_x_err, lorentz3DResults.gamma_y_err,
          lorentz3DResults.amp_err, lorentz3DResults.vert_offset_err,
          lorentz3DResults.chi2red, lorentz3DResults.pp, lorentz3DResults.dof,
          lorentz3DResults.charge_err,
          lorentz3DResults.fit_success);
        
        // Log 3D Lorentz fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DLorentzResults(event->GetEventID(), lorentz3DResults);
        }
        
    } else {
      // Not enough data points for 3D Lorentz fitting or 3D Lorentz fitting is disabled
      if (Constants::ENABLE_3D_LORENTZ_FIT) {
        fRunAction->Set3DLorentzResults(
          0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amp, vert_offset
          0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, amp_err, vert_offset_err
          0, 0, 0,           // chi2red, pp, dof
          0,                 // charge_err
          false);            // fit_success = false
      }
    }
      
      // ===============================================
      // 3D GAUSS FIT (conditionally enabled)
      // ===============================================
      
      // Perform 3D Gauss fitting if we have enough data points and 3D Gauss fitting is enabled
      if (x_coords.size() >= 6 && Constants::ENABLE_3D_GAUSS_FIT) { // Need at least 6 points for 3D Gauss fit
        // Perform 3D Gauss fitting using the Ceres Solver implementation
        Gauss3DResultsCeres gauss3DResults = GaussCeres3D(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (gauss3DResults.fit_success) {
        }
        
        // Pass 3D Gauss fit results to RunAction
        fRunAction->Set3DGaussResults(
          gauss3DResults.center_x, gauss3DResults.center_y, 
          gauss3DResults.sigma_x, gauss3DResults.sigma_y, 
          gauss3DResults.amp, gauss3DResults.vert_offset,
          gauss3DResults.center_x_err, gauss3DResults.center_y_err,
          gauss3DResults.sigma_x_err, gauss3DResults.sigma_y_err,
          gauss3DResults.amp_err, gauss3DResults.vert_offset_err,
          gauss3DResults.chi2red, gauss3DResults.pp, gauss3DResults.dof,
          gauss3DResults.charge_err,
          gauss3DResults.fit_success);
        
        // Log 3D Gauss fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DGaussResults(event->GetEventID(), gauss3DResults);
        }
        
    } else {
      // Not enough data points for 3D Gauss fitting or 3D Gauss fitting is disabled
      if (Constants::ENABLE_3D_GAUSS_FIT) {
        fRunAction->Set3DGaussResults(
          0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amp, vert_offset
          0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, sigma_x_err, sigma_y_err, amp_err, vert_offset_err
          0, 0, 0,           // chi2red, pp, dof
          0,                 // charge_err
          false);            // fit_success = false
      }
    }
      
      // ===============================================
      // 3D POWER-LAW LORENTZ FIT (conditionally enabled)
      // ===============================================
      
      // Perform 3D Power-Law Lorentz fitting if we have enough data points and 3D Power-Law Lorentz fitting is enabled
      if (x_coords.size() >= 7 && Constants::ENABLE_3D_POWER_LORENTZ_FIT) { // Need at least 7 points for 3D Power-Law Lorentz fit
        // Perform 3D Power-Law Lorentz fitting using the Ceres Solver implementation
        PowerLorentz3DResultsCeres powerLorentz3DResults = PowerLorentzCeres3D(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (powerLorentz3DResults.fit_success) {
        }
        
        // Pass 3D Power-Law Lorentz fit results to RunAction
        fRunAction->Set3DPowerLorentzResults(
          powerLorentz3DResults.center_x, powerLorentz3DResults.center_y,
          powerLorentz3DResults.gamma_x, powerLorentz3DResults.gamma_y,
          powerLorentz3DResults.beta, powerLorentz3DResults.amp, 
          powerLorentz3DResults.vert_offset,
          powerLorentz3DResults.center_x_err, powerLorentz3DResults.center_y_err,
          powerLorentz3DResults.gamma_x_err, powerLorentz3DResults.gamma_y_err,
          powerLorentz3DResults.beta_err, powerLorentz3DResults.amp_err, 
          powerLorentz3DResults.vert_offset_err,
          powerLorentz3DResults.chi2red, powerLorentz3DResults.pp, powerLorentz3DResults.dof,
          powerLorentz3DResults.charge_err,
          powerLorentz3DResults.fit_success);
        
        // Log 3D Power Lorentz fitting results to SimulationLogger
        SimulationLogger* logger = SimulationLogger::GetInstance();
        if (logger) {
          logger->Log3DPowerLorentzResults(event->GetEventID(), powerLorentz3DResults);
        }
        
    } else {
      // Not enough data points for 3D Power-Law Lorentz fitting or 3D Power-Law Lorentz fitting is disabled
      if (Constants::ENABLE_3D_POWER_LORENTZ_FIT) {
        fRunAction->Set3DPowerLorentzResults(
          0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amp, vert_offset
          0, 0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, beta_err, amp_err, vert_offset_err
          0, 0, 0,              // chi2red, pp, dof
          0,                    // charge_err
          false);               // fit_success = false
      }
    }

        
    } else {
      // Not enough data points for fitting or Gauss fitting is disabled
      if (Constants::ENABLE_GAUSS_FIT) {
        fRunAction->Set2DGaussResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, sigma, amp, center_err, sigma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
          false); // fit_success = false
        
        // Set default diagonal fit values when not enough data points
        fRunAction->SetDiagGaussResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amp, center_err, sigma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
      
      // Not enough data points for Lorentz fitting or Lorentz fitting is disabled
      if (Constants::ENABLE_LORENTZ_FIT) {
        fRunAction->Set2DLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
          0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
          false); // fit_success = false
        
        // Set default diagonal Lorentz fit values when not enough data points
        fRunAction->SetDiagLorentzResults(
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
          false); // fit_success = false
      }
    }
  } else {
    // Skip fitting due to conditions not met
    if (isPixelHit) {
    } else if (fNeighborhoodChargeFractions.empty()) {
    }
    
    // Set default values (no fitting performed or fitting is disabled)
    if (Constants::ENABLE_GAUSS_FIT) {
      fRunAction->Set2DGaussResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, sigma, amp, center_err, sigma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
        false); // fit_success = false
      
      // Set default diagonal fit values when fitting is skipped
      fRunAction->SetDiagGaussResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, sigma, amp, center_err, sigma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_success = false
    }
    
    // Set default Lorentz values (no fitting performed or Lorentz fitting is disabled)
    if (Constants::ENABLE_LORENTZ_FIT) {
      fRunAction->Set2DLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
        false); // fit_success = false
      
      // Set default diagonal Lorentz fit values when fitting is skipped
      fRunAction->SetDiagLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, amp, center_err, gamma_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_success = false
    }
    
    // Set default Power-Law Lorentz values (no fitting performed or Power-Law Lorentz fitting is disabled)
    if (Constants::ENABLE_POWER_LORENTZ_FIT) {
      fRunAction->Set2DPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters (center, gamma, beta, amp, center_err, gamma_err, beta_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
        0, 0,  // charge uncertainties (x_charge_err, y_charge_err)
        false); // fit_success = false
      
      // Set default diagonal Power-Law Lorentz fit values when fitting is skipped
      fRunAction->SetDiagPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters (center, gamma, beta, amp, center_err, gamma_err, beta_err, amp_err, vert_offset, vert_offset_err, chi2red, pp, dof, fit_success)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
        false); // fit_success = false
    }
    
    // Set default 3D Lorentz values (no fitting performed or 3D Lorentz fitting is disabled)
    if (Constants::ENABLE_3D_LORENTZ_FIT) {
      fRunAction->Set3DLorentzResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amp, vert_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, amp_err, vert_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_err
        false);            // fit_success = false
    }
    
    // Set default 3D Gauss values (no fitting performed or 3D Gauss fitting is disabled)
    if (Constants::ENABLE_3D_GAUSS_FIT) {
      fRunAction->Set3DGaussResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amp, vert_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, sigma_x_err, sigma_y_err, amp_err, vert_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_err
        false);            // fit_success = false
    }
    
    // Set default 3D Power-Law Lorentz values (no fitting performed or 3D Power-Law Lorentz fitting is disabled)
    if (Constants::ENABLE_3D_POWER_LORENTZ_FIT) {
      fRunAction->Set3DPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amp, vert_offset
        0, 0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, beta_err, amp_err, vert_offset_err
        0, 0, 0,              // chi2red, pp, dof
        0,                    // charge_err
        false);               // fit_success = false
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

void EventAction::AddEdep(G4double edep, G4ThreeVector pos)
{
  // Accumulate energy deposited while particle travels through detector volume
  // Energy weighted pos calculation
  if (edep > 0) {
    if (!fHasHit) {
      fPos = pos * edep;
      fEdep = edep;  // First energy depositionit in detector
      fHasHit = true;
    } else {
      // Weight pos by energy deposition and sum total energy
      fPos = (fPos * fEdep + pos * edep) / (fEdep + edep);
      fEdep += edep;  // Accumulate total energy deposited in detector
    }
  }
}

// Implementation of the new method to set the initial pos
void EventAction::SetInitialPos(const G4ThreeVector& pos)
{
  fInitialPos = pos;
}

// Implementation of the nearest pixel calculation method
G4ThreeVector EventAction::CalcNearestPixel(const G4ThreeVector& pos)
{
  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the pos relative to the detector face
  G4ThreeVector relativePos = pos - detectorPos;
  
  // For the AC-LGAD, pixels are on the front surface (z > detector z)
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calc which pixel grid pos is closest (i and j indices)
  G4int i = std::round((relativePos.x() - firstPixelPos) / pixelSpacing);
  G4int j = std::round((relativePos.y() - firstPixelPos) / pixelSpacing);
  
  // Clamp i and j to valid pixel indices
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calc the actual pixel center pos
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  // Pixels are on the detector front surface
  G4double pixelZ = detectorPos.z() + Constants::DEFAULT_DETECTOR_WIDTH/2 + Constants::DEFAULT_PIXEL_WIDTH/2; // detector half-width + pixel half-width
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calc and store distance from hit to pixel center (2D distance in detector plane)
  G4double dx = pos.x() - pixelX;
  G4double dy = pos.y() - pixelY;
  fActualPixelDistance = std::sqrt(dx*dx + dy*dy);
  
  // Calc and store delta values (pixel center - true pos)
  fPixelTrueDeltaX = pixelX - pos.x();
  fPixelTrueDeltaY = pixelY - pos.y();
  
  // Determine if the hit was on a pixel using the detector's method
  fPixelHit = fDetector->IsPosOnPixel(pos);
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Calc the angular size of pixel from hit pos
G4double EventAction::CalcPixelAlpha(const G4ThreeVector& hitPos, G4int pixelI, G4int pixelJ)
{
  // Check if the hit is inside the pixel. If so, return NaN to indicate no alpha calculation
  G4bool isInsidePixel = fDetector->IsPosOnPixel(hitPos);
  if (isInsidePixel) {
    return std::numeric_limits<G4double>::quiet_NaN(); // Return NaN for hits inside pixels (no alpha calculation needed)
  }

  // Get detector parameters
  G4double pixelSize = fDetector->GetPixelSize();
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
  G4double detSize = fDetector->GetDetSize();
  G4int numBlocksPerSide = fDetector->GetNumBlocksPerSide();
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Calc the center pos of the specified pixel
  G4double pixelCenterX = firstPixelPos + pixelI * pixelSpacing;
  G4double pixelCenterY = firstPixelPos + pixelJ * pixelSpacing;
  G4double pixelCenterZ = detectorPos.z();
  
  // Calc distance from hit pos to pixel center (2D distance in XY plane)
  G4double dx = hitPos.x() - pixelCenterX;
  G4double dy = hitPos.y() - pixelCenterY;
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

// Calc angles from hit pos to all pixels in a neighborhood grid around the hit pixel
void EventAction::CalcNeighborhoodGridAngles(const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ)
{
  // Clear previous data
  fNeighborhoodAngles.clear();
  
  // Check if hit is inside a pixel - if so, all angles should be invalid
  G4bool isInsidePixel = fDetector->IsPosOnPixel(hitPos);
  if (isInsidePixel) {
    // Fill all poss with NaN for inside-pixel hits
    G4int gridSize = 2 * fNeighborhoodRadius + 1;
    G4int totalPixels = gridSize * gridSize;
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        // Calc the pixel indices for this grid pos
        G4int gridPixelI = hitPixelI + di;
        G4int gridPixelJ = hitPixelJ + dj;
        
        // Store NaN angle
        fNeighborhoodAngles.push_back(std::numeric_limits<G4double>::quiet_NaN()); // Use same NaN as elsewhere
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
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
  G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
  
  // Define the neighborhood grid: fNeighborhoodRadius pixels in each direction from the center
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = hitPixelI + di;
      G4int gridPixelJ = hitPixelJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNeighborhoodAngles.push_back(-999.0); // Invalid angle marker
        continue;
      }
      
      // Calc the center pos of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calc the alpha angle for this pixel using the same algorithm as the Python demo
      G4double alpha = CalcPixelAlphaSubtended(hitPos.x(), hitPos.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Convert to degrees
      G4double alphaInDegrees = alpha * (180.0 / CLHEP::pi);
      
      // Store the results
      fNeighborhoodAngles.push_back(alphaInDegrees);
    }
  }
}

// Calc the angular size subtended by a pixel as seen from a hit point (2D calculation)
// This now uses the analytical formula: α = tan^(-1) [(l/2 * √2) / (l/2 * √2 + d)]
G4double EventAction::CalcPixelAlphaSubtended(G4double hitX, G4double hitY,
                                                  G4double pixelCenterX, G4double pixelCenterY,
                                                  G4double pixelWidth, G4double pixelHeight)
{
  // Calc distance from hit pos to pixel center (2D distance in XY plane)
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

// Calc charge sharing for pixels in a neighborhood grid around the hit pixel
void EventAction::CalcNeighborhoodChargeSharing()
{
  // Clear previous data
  fNeighborhoodChargeFractions.clear();
  fNeighborhoodDistances.clear();
  fNeighborhoodCharge.clear();
  
  // Check if no energy was deposited
  if (fEdep <= 0) {
    // Fill all poss with zero for no-energy events
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        fNeighborhoodChargeFractions.push_back(0.0);
        fNeighborhoodDistances.push_back(-999.0);
        fNeighborhoodCharge.push_back(0.0);
      }
    }
    return;
  }
  
  // Check if hit is inside a pixel - if so, assign zero charge (per user requirement)
  G4bool isInsidePixel = fDetector->IsPosOnPixel(fPos);
  if (isInsidePixel) {
    // For pixel hits, energy deposition should be zero (per user requirement)
    // Therefore, charge is also zero
    G4double totalCharge = 0.0;
    
    // Fill all poss, giving all charge to the hit pixel and zero to others
    for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
      for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
        G4int gridPixelI = fPixelIndexI + di;
        G4int gridPixelJ = fPixelIndexJ + dj;
        
        // Check if this is the pixel that was hit
        if (di == 0 && dj == 0) {
          // This is the center pixel (the one that was hit) - but charge is zero for pixel hits
          fNeighborhoodChargeFractions.push_back(0.0);
          fNeighborhoodCharge.push_back(0.0);
          fNeighborhoodDistances.push_back(0.0); // Distance to center of hit pixel is effectively zero
        } else if (gridPixelI >= 0 && gridPixelI < fDetector->GetNumBlocksPerSide() && 
                   gridPixelJ >= 0 && gridPixelJ < fDetector->GetNumBlocksPerSide()) {
          // This is a valid pixel in the detector but not the hit pixel
          fNeighborhoodChargeFractions.push_back(0.0);
          fNeighborhoodCharge.push_back(0.0);
          
        // Calc distance to this pixel center for completeness
        G4double pixelSize = fDetector->GetPixelSize();
        G4double pixelSpacing = fDetector->GetPixelSpacing();
        G4double pixelCornerOffset = fDetector->GetPixelCornerOffset();
        G4double detSize = fDetector->GetDetSize();
        G4double firstPixelPos = -detSize/2 + pixelCornerOffset + pixelSize/2;
        
        G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
        G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
        G4double dx = fPos.x() - pixelCenterX;
        G4double dy = fPos.y() - pixelCenterY;
        G4double distance = std::sqrt(dx*dx + dy*dy);
          fNeighborhoodDistances.push_back(distance);
        } else {
          // This pixel is outside the detector bounds
          fNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
          fNeighborhoodDistances.push_back(-999.0);
          fNeighborhoodCharge.push_back(0.0);
        }
      }
    }
    return;
  }
  
  // Convert energy depositionit to number of electrons
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
  G4ThreeVector detectorPos = fDetector->GetDetectorPos();
  
  // Calc the first pixel pos (corner)
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
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within the detector bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Store invalid data for out-of-bounds pixels
        fNeighborhoodChargeFractions.push_back(-999.0); // Invalid marker
        fNeighborhoodDistances.push_back(-999.0);
        fNeighborhoodCharge.push_back(0.0);
        continue;
      }
      
      // Calc the center pos of this grid pixel
      G4double pixelCenterX = firstPixelPos + gridPixelI * pixelSpacing;
      G4double pixelCenterY = firstPixelPos + gridPixelJ * pixelSpacing;
      
      // Calc the distance from the hit to the pixel center (in mm)
      G4double dx = fPos.x() - pixelCenterX;
      G4double dy = fPos.y() - pixelCenterY;
      G4double distance = std::sqrt(dx*dx + dy*dy);
      
      // Calc the alpha angle for this pixel using the same algorithm as elsewhere
      G4double alpha = CalcPixelAlphaSubtended(fPos.x(), fPos.y(), 
                                                   pixelCenterX, pixelCenterY, 
                                                   pixelSize, pixelSize);
      
      // Store data for valid pixels
      distances.push_back(distance);
      angles.push_back(alpha);
      validPixelI.push_back(gridPixelI);
      validPixelJ.push_back(gridPixelJ);
      
      // Calc weight according to formula: α_i * ln(d_i/d_0)^(-1)
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
  
  // Calc total weight
  G4double totalWeight = 0.0;
  for (G4double weight : weights) {
    totalWeight += weight;
  }
  
  // Sec pass: calculate charge fractions and values
  size_t validIndex = 0;
  for (G4int di = -fNeighborhoodRadius; di <= fNeighborhoodRadius; di++) {
    for (G4int dj = -fNeighborhoodRadius; dj <= fNeighborhoodRadius; dj++) {
      // Calc the pixel indices for this grid pos
      G4int gridPixelI = fPixelIndexI + di;
      G4int gridPixelJ = fPixelIndexJ + dj;
      
      // Check if this pixel is within bounds
      if (gridPixelI < 0 || gridPixelI >= numBlocksPerSide || 
          gridPixelJ < 0 || gridPixelJ >= numBlocksPerSide) {
        // Already stored invalid data in first pass
        continue;
      }
      
      // Calc charge fraction and value
      G4double chargeFraction = 0.0;
      G4double chargeValue = 0.0;
      
      if (totalWeight > 0) {
        chargeFraction = weights[validIndex] / totalWeight;
        chargeValue = chargeFraction * totalCharge;
      }
      
      fNeighborhoodChargeFractions.push_back(chargeFraction);
      fNeighborhoodDistances.push_back(distances[validIndex]);
      fNeighborhoodCharge.push_back(chargeValue * fElementaryCharge);
      
      validIndex++;
    }
  }
}

// Perform automatic radius selection based on fit quality
G4int EventAction::SelectOptimalRadius(const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ)
{
  G4double bestQuality = -1.0;
  G4int bestRadius = fNeighborhoodRadius; // Default fallback
  
  // Test different radii from min to max
  for (G4int testRadius = fMinAutoRadius; testRadius <= fMaxAutoRadius; testRadius++) {
    G4double fitQuality = EvaluateFitQuality(testRadius, hitPos, hitPixelI, hitPixelJ);
    
    if (fitQuality > bestQuality) {
      bestQuality = fitQuality;
      bestRadius = testRadius;
    }
  }
  
  // Store the best fit quality found
  fSelectedQuality = bestQuality;
  
  return bestRadius;
}

// Evaluate fit quality for a given radius
G4double EventAction::EvaluateFitQuality(G4int radius, const G4ThreeVector& hitPos, G4int hitPixelI, G4int hitPixelJ)
{
  // Temporarily save current radius
  G4int originalRadius = fNeighborhoodRadius;
  fNeighborhoodRadius = radius;
  
  // Clear and recalculate neighborhood data with test radius
  std::vector<G4double> tempAngles = fNeighborhoodAngles;
  std::vector<G4double> tempChargeFractions = fNeighborhoodChargeFractions;
  std::vector<G4double> tempDistances = fNeighborhoodDistances;
  std::vector<G4double> tempCharge = fNeighborhoodCharge;
  
  // Calc neighborhood data with test radius
  CalcNeighborhoodGridAngles(hitPos, hitPixelI, hitPixelJ);
  CalcNeighborhoodChargeSharing();
  
  // Check if we have enough data points for fitting
  G4int validPoints = 0;
  for (size_t i = 0; i < fNeighborhoodChargeFractions.size(); ++i) {
    if (fNeighborhoodChargeFractions[i] > 0) {
      validPoints++;
    }
  }
  
  if (validPoints < Constants::MIN_POINTS_FOR_FIT) {
    // Restore original data and radius
    fNeighborhoodRadius = originalRadius;
    fNeighborhoodAngles = tempAngles;
    fNeighborhoodChargeFractions = tempChargeFractions;
    fNeighborhoodDistances = tempDistances;
    fNeighborhoodCharge = tempCharge;
    return 0.0; // Poor quality due to insufficient data
  }
  
  // Extract data for fitting
  std::vector<double> x_coords, y_coords, charge_values;
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  G4int gridSize = 2 * radius + 1;
  G4ThreeVector nearestPixel = CalcNearestPixel(hitPos);
  
  for (size_t i = 0; i < fNeighborhoodChargeFractions.size(); ++i) {
    if (fNeighborhoodChargeFractions[i] > 0) {
      G4int col = i / gridSize;  // di (X) was outer loop
      G4int row = i % gridSize;  // dj (Y) was inner loop
      
      G4int offsetI = col - radius; // X offset from center pixel
      G4int offsetJ = row - radius; // Y offset from center pixel
      
      G4double x_pos = nearestPixel.x() + offsetI * pixelSpacing;
      G4double y_pos = nearestPixel.y() + offsetJ * pixelSpacing;
      
      x_coords.push_back(x_pos);
      y_coords.push_back(y_pos);
      charge_values.push_back(fNeighborhoodCharge[i]);
    }
  }
  
  G4double fitQuality = 0.0;
  
  if (x_coords.size() >= Constants::MIN_POINTS_FOR_FIT) {
    // Perform quick Gauss fit evaluation
    Gauss2DResultsCeres fitResults = GaussCeres2D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, 
      false, // verbose=false
      false); // enable_outlier_filtering
    
    if (fitResults.fit_success) {
      // Calc fit quality based on residuals and fit parameters
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
  fNeighborhoodAngles = tempAngles;
  fNeighborhoodChargeFractions = tempChargeFractions;
  fNeighborhoodDistances = tempDistances;
  fNeighborhoodCharge = tempCharge;
  
  return fitQuality;
}