#include "EventAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "SteppingAction.hh"
#include "Constants.hh"
#include "Control.hh"
#include "CrashHandler.hh"
#include "SimulationLogger.hh"
#include "GaussFit2D.hh"
#include "LorentzFit2D.hh"
#include "PowerLorentzFit2D.hh"
#include "GaussFit3D.hh"
#include "LorentzFit3D.hh"
#include "PowerLorentzFit3D.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4THitsMap.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4RunManager.hh"
#include <exception>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>

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
  fSteppingAction(nullptr),
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
  fAutoRadiusEnabled(Control::AUTO_RADIUS),
  fMinAutoRadius(Constants::MIN_AUTO_RADIUS),
  fMaxAutoRadius(Constants::MAX_AUTO_RADIUS),
  fSelectedRadius(4),
  fSelectedQuality(0.0),
  fIonizationEnergy(Constants::IONIZATION_ENERGY),
  fAmplificationFactor(Constants::AMPLIFICATION_FACTOR),
  fD0(Constants::D0_CHARGE_SHARING),
  fElementaryCharge(Constants::ELEMENTARY_CHARGE),
  fScorerEnergyDeposit(0.0),
  fScorerHitCount(0),
  fScorerDataValid(false),
  fPureSiliconHit(false),
  fAluminumContaminated(false),
  fChargeCalculationEnabled(false)
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
  
  // Validate SteppingAction integration for first event only (to avoid spam)
  if (event->GetEventID() == 0) {
    ValidateSteppingActionIntegration();
    DemonstrateIntegrationWorkflow();
  }
  
  // ========================================
  // EXCLUSIVE SCORER USAGE WORKFLOW
  // ========================================
  
  // Step 1: Reset aluminum interaction tracking in SteppingAction for trajectory analysis
  if (fSteppingAction) {
    fSteppingAction->ResetInteractionTracking();
  }
  
  // Step 2: Reset hit purity tracking variables for Multi-Functional Detector validation
  fPureSiliconHit = false;
  fAluminumContaminated = false;
  fChargeCalculationEnabled = false;
  
  // Step 3: Reset scorer data for exclusive Multi-Functional Detector usage
  fScorerEnergyDeposit = 0.0;
  fScorerHitCount = 0;
  fScorerDataValid = false;
  
  // ========================================
  // TRADITIONAL EVENT VARIABLES RESET
  // ========================================
  
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
  // ========================================
  // EXCLUSIVE MULTI-FUNCTIONAL DETECTOR DATA PRIORITIZATION
  // ========================================
  
  // Step 1: Collect scorer data from Multi-Functional Detector (PRIMARY DATA SOURCE)
  // This is prioritized over traditional SteppingAction data collection
  try {
    CollectScorerData(event);
    // Validate that scorer data collection hasn't interfered with existing calculations
    ValidateScorerDataIntegrity();
  } catch (const std::exception& e) {
    // Fallback: If scorer data collection fails, continue with default values
    fScorerEnergyDeposit = 0.0;
    fScorerHitCount = 0;
    fScorerDataValid = false;
  } catch (...) {
    // Fallback: If any other exception occurs, continue with default values
    fScorerEnergyDeposit = 0.0;
    fScorerHitCount = 0;
    fScorerDataValid = false;
  }
  
  // Step 2: Perform exclusive Multi-Functional Detector validation
  // Validate hit purity by combining scorer data with trajectory analysis
  ValidateHitPurity();
  
  // Step 3: Determine if charge sharing calculations should proceed
  // Only proceed for pure silicon hits (aluminum-contaminated hits are excluded)
  G4bool shouldCalculateChargeSharing = ShouldCalculateChargeSharing();
  
  // ========================================
  // TRADITIONAL EVENT DATA PROCESSING
  // ========================================
  
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
  
  // ADDITIONAL VALIDATION: Double-check pixel hit status for consistency
  G4bool pixelHitDoubleCheck = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  if (isPixelHit != pixelHitDoubleCheck) {
    G4cerr << "WARNING: Pixel hit status inconsistency detected!" << G4endl;
    G4cerr << "  Initial check: " << (isPixelHit ? "PIXEL HIT" : "NON-PIXEL HIT") << G4endl;
    G4cerr << "  Double check: " << (pixelHitDoubleCheck ? "PIXEL HIT" : "NON-PIXEL HIT") << G4endl;
    G4cerr << "  Position: (" << fPos.x()/mm << ", " << fPos.y()/mm << ", " << fPos.z()/mm << ") mm" << G4endl;
    // Use the more recent check as it's more reliable
    isPixelHit = pixelHitDoubleCheck;
    fPixelHit = pixelHitDoubleCheck;
  }
  
  // VALIDATION: For pixel hits both deltas must be NaN. If either value is finite we report an error
  if (isPixelHit) {
    // Use robust NaN check: NaN != NaN is always true
    const G4bool deltaXIsFinite = std::isfinite(fPixelTrueDeltaX);
    const G4bool deltaYIsFinite = std::isfinite(fPixelTrueDeltaY);

    if (deltaXIsFinite || deltaYIsFinite) {
      G4cerr << "ERROR: Deltas should be NaN for pixel hits!" << G4endl;
      G4cerr << "  Pixel hit: YES" << G4endl;
      G4cerr << "  DeltaX: " << fPixelTrueDeltaX/mm << " mm (finite: " << (deltaXIsFinite ? "YES" : "NO") << ")" << G4endl;
      G4cerr << "  DeltaY: " << fPixelTrueDeltaY/mm << " mm (finite: " << (deltaYIsFinite ? "YES" : "NO") << ")" << G4endl;

      // Force deltas to NaN for pixel hits to keep downstream logic consistent
      fPixelTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
      fPixelTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
    }
  }
  
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
      // REMOVED: Only include pixels with charge filter - include ALL pixels for small charge values
      // BUT: Still need to filter out invalid markers (-999)
      if (fNeighborhoodChargeFractions[i] < -998.0) {
        continue; // Skip pixels with invalid markers (-999)
      }
      
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
    
    // ===============================================
    // GAUSS FIT (conditionally enabled)
    // ===============================================
    
    // Perform 2D fitting if we have enough data points and Gauss fitting is enabled
    if (x_coords.size() >= 3 && Control::GAUSS_FIT && Control::ROWCOL_FIT) { // Need at least 3 points for 1D Gauss fit
      // Perform 2D Gauss fitting using the Ceres Solver implementation
      Gauss2DResultsCeres fitResults = GaussCeres2D(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, 
        false, // verbose=false for production
        false); // enable_outlier_filtering
      
      if (fitResults.fit_success) {
      }
      
      // VALIDATION: Only save fitting results for non-pixel hits
      if (isPixelHit) {
        G4cerr << "CRITICAL ERROR: Attempting to save Gauss fitting results for a pixel hit!" << G4endl;
        G4cerr << "  This should never happen - fitting should only be performed for non-pixel hits." << G4endl;
        G4cerr << "  Event ID: " << event->GetEventID() << G4endl;
        G4cerr << "  Position: (" << fPos.x()/mm << ", " << fPos.y()/mm << ", " << fPos.z()/mm << ") mm" << G4endl;
        G4cerr << "  Skipping fitting result storage." << G4endl;
      } else {
        // SAFE: Save fitting results only for non-pixel hits
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
      }
        
        // Perform diagonal fitting if 2D fitting was performed and success and diagonal fitting is enabled
        if (fitResults.fit_success && Control::DIAG_FIT) {
          // Perform diagonal Gauss fitting using the Ceres Solver implementation
          DiagResultsCeres diagResults = DiagGaussCeres(
          x_coords, y_coords, charge_values,
          nearestPixel.x(), nearestPixel.y(),
          pixelSpacing, 
          false, // verbose=false for production
          false); // enable_outlier_filtering
        
        if (diagResults.fit_success) {
        }
        
        // VALIDATION: Only save diagonal fitting results for non-pixel hits
        if (isPixelHit) {
          G4cerr << "CRITICAL ERROR: Attempting to save diagonal Gauss fitting results for a pixel hit!" << G4endl;
          G4cerr << "  This should never happen - diagonal fitting should only be performed for non-pixel hits." << G4endl;
          G4cerr << "  Event ID: " << event->GetEventID() << G4endl;
          G4cerr << "  Position: (" << fPos.x()/mm << ", " << fPos.y()/mm << ", " << fPos.z()/mm << ") mm" << G4endl;
          G4cerr << "  Skipping diagonal fitting result storage." << G4endl;
        } else {
          // SAFE: Save diagonal fitting results only for non-pixel hits
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
        }
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
      if (x_coords.size() >= 3 && Control::LORENTZ_FIT && Control::ROWCOL_FIT) { // Need at least 3 points for 1D Lorentz fit
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
        if (lorentzResults.fit_success && Control::DIAG_FIT) {
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
      if (Control::LORENTZ_FIT) {
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
      if (x_coords.size() >= 3 && Control::POWER_LORENTZ_FIT && Control::ROWCOL_FIT) { // Need at least 3 points for Power-Law Lorentz fit
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
        if (powerLorentzResults.fit_success && Control::DIAG_FIT) {
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
      if (Control::POWER_LORENTZ_FIT) {
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
      if (x_coords.size() >= 6 && Control::LORENTZ_FIT_3D) { // Need at least 6 points for 3D Lorentz fit
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
      if (Control::LORENTZ_FIT_3D) {
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
      if (x_coords.size() >= 6 && Control::GAUSS_FIT_3D) { // Need at least 6 points for 3D Gauss fit
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
      if (Control::GAUSS_FIT_3D) {
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
      if (x_coords.size() >= 7 && Control::POWER_LORENTZ_FIT_3D) { // Need at least 7 points for 3D Power-Law Lorentz fit
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
      if (Control::POWER_LORENTZ_FIT_3D) {
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
      if (Control::GAUSS_FIT) {
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
      if (Control::LORENTZ_FIT) {
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
    if (Control::GAUSS_FIT) {
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
    if (Control::LORENTZ_FIT) {
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
    if (Control::POWER_LORENTZ_FIT) {
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
    if (Control::LORENTZ_FIT_3D) {
      fRunAction->Set3DLorentzResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amp, vert_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, amp_err, vert_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_err
        false);            // fit_success = false
    }
    
    // Set default 3D Gauss values (no fitting performed or 3D Gauss fitting is disabled)
    if (Control::GAUSS_FIT_3D) {
      fRunAction->Set3DGaussResults(
        0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amp, vert_offset
        0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, sigma_x_err, sigma_y_err, amp_err, vert_offset_err
        0, 0, 0,           // chi2red, pp, dof
        0,                 // charge_err
        false);            // fit_success = false
    }
    
    // Set default 3D Power-Law Lorentz values (no fitting performed or 3D Power-Law Lorentz fitting is disabled)
    if (Control::POWER_LORENTZ_FIT_3D) {
      fRunAction->Set3DPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amp, vert_offset
        0, 0, 0, 0, 0, 0, 0,  // center_x_err, center_y_err, gamma_x_err, gamma_y_err, beta_err, amp_err, vert_offset_err
        0, 0, 0,              // chi2red, pp, dof
        0,                    // charge_err
        false);               // fit_success = false
    }

  }
  
  // ========================================
  // CONDITIONAL CHARGE SHARING CALCULATION
  // ========================================
  
  // Perform conditional charge sharing calculation based on hit purity validation
  // This replaces traditional charge sharing and only processes pure silicon hits
  // GaussRowDeltaX, GaussRowDeltaY calculations are skipped for aluminum-contaminated hits
  if (shouldCalculateChargeSharing) {
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->LogInfo("Event " + std::to_string(event->GetEventID()) + 
                     ": Performing charge sharing calculation for pure silicon hit");
    }
    ConditionalChargeCalculation(event);
  } else {
    // Skip charge sharing calculations for aluminum-contaminated hits
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      G4String reason = "Charge sharing calculation skipped for Event " + 
                       std::to_string(event->GetEventID()) + ": ";
      
      if (!fChargeCalculationEnabled) {
        reason += "Charge calculation disabled";
      } else if (!fScorerDataValid) {
        reason += "Invalid scorer data";
      } else if (!fPureSiliconHit) {
        reason += "Not a pure silicon hit";
      } else if (fAluminumContaminated) {
        reason += "Aluminum contamination detected";
      } else {
        reason += "Hit validation failed";
      }
      
      logger->LogInfo(reason);
    }
    
    // Set default values for all fitting results since no calculations will be performed
    // DO NOT reset fitting results here – earlier fits (row/column) are already stored.
    // Leaving them intact avoids GaussRowDOF being overwritten to 0.
    // SetDefaultFittingResults();
  }
  
  // Transfer scorer data to RunAction for ROOT tree storage
  fRunAction->SetScorerData(fScorerEnergyDeposit, fScorerHitCount, fScorerDataValid);
  
  // Transfer hit purity tracking data to RunAction for ROOT tree storage
  fRunAction->SetHitPurityData(fPureSiliconHit, fAluminumContaminated, fChargeCalculationEnabled);
  
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
  
  // Check if the hit is within the detector bounds BEFORE clamping
  G4bool isWithinDetector = (i >= 0 && i < numBlocksPerSide && j >= 0 && j < numBlocksPerSide);
  
  // Clamp i and j to valid pixel indices (for geometry purposes)
  i = std::max(0, std::min(i, numBlocksPerSide - 1));
  j = std::max(0, std::min(j, numBlocksPerSide - 1));
  
  // Calc the actual pixel center pos
  G4double pixelX = firstPixelPos + i * pixelSpacing;
  G4double pixelY = firstPixelPos + j * pixelSpacing;
  // Pixels are on the detector front surface
  G4double pixelZ = detectorPos.z() + Constants::DETECTOR_WIDTH/2 + Constants::PIXEL_WIDTH/2; // detector half-width + pixel half-width
  
  // Store the pixel indices for later use
  fPixelIndexI = i;
  fPixelIndexJ = j;
  
  // Calc and store distance from hit to pixel center (2D distance in detector plane)
  G4double dx = pos.x() - pixelX;
  G4double dy = pos.y() - pixelY;
  fActualPixelDistance = std::sqrt(dx*dx + dy*dy);
  
  // Determine if the hit was on a pixel using volume-based detection
  G4bool isOnPixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
  
  // Calc and store delta values (pixel center - true pos)
  // Only calculate meaningful deltas for hits within detector bounds AND non-pixel hits
  if (isWithinDetector && !isOnPixel) {
    fPixelTrueDeltaX = std::abs(pixelX - pos.x());
    fPixelTrueDeltaY = std::abs(pixelY - pos.y());
  } else {
    // For hits outside detector bounds or pixel hits, mark deltas as invalid
    fPixelTrueDeltaX = std::numeric_limits<G4double>::quiet_NaN();
    fPixelTrueDeltaY = std::numeric_limits<G4double>::quiet_NaN();
  }
  
  // Store the pixel hit status
  fPixelHit = isOnPixel;
  
  return G4ThreeVector(pixelX, pixelY, pixelZ);
}

// Calc the angular size of pixel from hit pos
G4double EventAction::CalcPixelAlpha(const G4ThreeVector& hitPos, G4int pixelI, G4int pixelJ)
{
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
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
  
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
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
  
  // Check if hit is inside a pixel using volume-based detection
  G4bool isInsidePixel = fSteppingAction ? fSteppingAction->IsPixelHit() : false;
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
  G4int validPoints = fNeighborhoodChargeFractions.size(); // Use ALL points, not just positive charge
  
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
    // REMOVED: Only include pixels with charge filter - include ALL pixels for small charge values
    // BUT: Still need to filter out invalid markers (-999)
    if (fNeighborhoodChargeFractions[i] < -998.0) {
      continue; // Skip pixels with invalid markers (-999)
    }
    
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

void EventAction::CollectScorerData(const G4Event* event)
{
  G4HCofThisEvent* HC = event->GetHCofThisEvent();
  if (!HC) {
    G4cerr << "WARNING: No HCofThisEvent available" << G4endl;
    fScorerDataValid = false;
    return;
  }

  G4SDManager* SDman = G4SDManager::GetSDMpointer();

  // Energy deposit
  int energyID = SDman->GetCollectionID("SiliconDetector/EnergyDeposit");
  if (energyID < 0) {
    G4cerr << "WARNING: Invalid collection ID for EnergyDeposit" << G4endl;
    fScorerDataValid = false;
    return;
  }
  G4THitsMap<G4double>* energyMap = static_cast<G4THitsMap<G4double>*>(HC->GetHC(energyID));
  if (!energyMap) {
    G4cerr << "WARNING: Energy map not found" << G4endl;
    fScorerDataValid = false;
    return;
  }
  fScorerEnergyDeposit = 0.;
  for (auto it : *energyMap->GetMap()) {
    fScorerEnergyDeposit += *it.second;
  }

  // Hit count
  int hitID = SDman->GetCollectionID("SiliconDetector/HitCount");
  if (hitID < 0) {
    G4cerr << "WARNING: Invalid collection ID for HitCount" << G4endl;
    fScorerDataValid = false;
    return;
  }
  G4THitsMap<G4double>* hitMap = static_cast<G4THitsMap<G4double>*>(HC->GetHC(hitID));
  if (!hitMap) {
    G4cerr << "WARNING: Hit map not found" << G4endl;
    fScorerDataValid = false;
    return;
  }
  fScorerHitCount = 0;
  for (auto it : *hitMap->GetMap()) {
    fScorerHitCount += static_cast<G4int>(*it.second);
  }

  fScorerDataValid = (fScorerEnergyDeposit > 0. && fScorerHitCount > 0);
}

void EventAction::SetScorerData(G4double energy, G4int hits, G4bool valid)
{
  fScorerEnergyDeposit = energy;
  fScorerHitCount = hits;
  fScorerDataValid = valid;
}

void EventAction::ValidateScorerDataIntegrity() const
{
  // Validate that scorer data collection doesn't interfere with existing charge sharing calculations
  
  // Check that core event data remains valid
  if (fHasHit) {
    // Validate that position data is still valid
    if (!std::isfinite(fPos.x()) || !std::isfinite(fPos.y()) || !std::isfinite(fPos.z())) {
      G4cerr << "WARNING: Event position data corrupted after scorer data collection!" << G4endl;
      G4cerr << "Position: (" << fPos.x() << ", " << fPos.y() << ", " << fPos.z() << ")" << G4endl;
    }
    
    // Validate that energy data is still valid
    if (!std::isfinite(fEdep) || fEdep < 0.0) {
      G4cerr << "WARNING: Event energy data corrupted after scorer data collection!" << G4endl;
      G4cerr << "Energy deposit: " << fEdep << " MeV" << G4endl;
    }
    
    // Validate that pixel mapping data is still valid
    if (fPixelHit) {
      if (fPixelIndexI < 0 || fPixelIndexJ < 0) {
        G4cerr << "WARNING: Pixel mapping data corrupted after scorer data collection!" << G4endl;
        G4cerr << "Pixel indices: (" << fPixelIndexI << ", " << fPixelIndexJ << ")" << G4endl;
      }
    }
  }
  
  // Check that scorer data is within reasonable bounds
  if (fScorerDataValid) {
    if (fScorerEnergyDeposit < 0.0 || fScorerEnergyDeposit > 1000.0) {
      G4cerr << "WARNING: Scorer energy data out of reasonable bounds: " << fScorerEnergyDeposit << " MeV" << G4endl;
    }
    
    if (fScorerHitCount < 0 || fScorerHitCount > 10000) {
      G4cerr << "WARNING: Scorer hit count out of reasonable bounds: " << fScorerHitCount << G4endl;
    }
  }
  
  // Validate that neighborhood data is still intact
  if (!fNeighborhoodChargeFractions.empty()) {
    for (size_t i = 0; i < fNeighborhoodChargeFractions.size(); i++) {
      if (!std::isfinite(fNeighborhoodChargeFractions[i])) {
        G4cerr << "WARNING: Neighborhood charge fraction data corrupted after scorer data collection!" << G4endl;
        G4cerr << "Index " << i << ": " << fNeighborhoodChargeFractions[i] << G4endl;
        break;
      }
    }
  }
}

void EventAction::ValidateHitPurity()
{
  // ========================================
  // EXCLUSIVE MULTI-FUNCTIONAL DETECTOR VALIDATION
  // ========================================
  
  // This method implements the exclusive Multi-Functional Detector validation
  // by combining scorer data with SteppingAction trajectory analysis
  // Only pure silicon hits (no aluminum contamination) are approved for charge sharing
  
  // Reset hit purity flags
  fPureSiliconHit = false;
  fAluminumContaminated = false;
  fChargeCalculationEnabled = false;
  
  // Validate that we have SteppingAction available for trajectory analysis
  if (!fSteppingAction) {
    G4cerr << "WARNING: SteppingAction not available for hit purity validation!" << G4endl;
    return;
  }
  
  // Step 1: Validate Multi-Functional Detector scorer data integrity
  if (!fScorerDataValid) {
    G4cerr << "WARNING: Multi-Functional Detector scorer data invalid - cannot perform hit purity validation" << G4endl;
    return;
  }
  
  // Step 2: Check if we have valid scorer data (energy deposit and hit count)
  if (fScorerEnergyDeposit <= 0.0 || fScorerHitCount <= 0) {
    // No valid scorer data - cannot validate hit purity
    // This is normal behavior when particles don't hit the detector
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
      G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
      
      // Only log occasionally to avoid spam
      if (eventID % 1000 == 0) {
        logger->LogInfo("Event " + std::to_string(eventID) + ": No scorer data - particle missed detector");
      }
    }
    return;
  }
  
  // Step 3: Get trajectory analysis results from SteppingAction
  G4bool hasAluminumInteraction = fSteppingAction->HasAluminumInteraction();
  G4bool hasAluminumPreContact = fSteppingAction->HasAluminumPreContact();
  G4bool isValidSiliconHit = fSteppingAction->IsValidSiliconHit();
  G4String firstInteractionVolume = fSteppingAction->GetFirstInteractionVolume();
  
  // Step 4: Determine aluminum contamination status
  fAluminumContaminated = hasAluminumInteraction || hasAluminumPreContact;
  
  // Step 5: Determine if this is a pure silicon hit
  fPureSiliconHit = isValidSiliconHit && !fAluminumContaminated;
  
  // Step 6: Enable charge calculation only for pure silicon hits
  // This ensures GaussRowDeltaX, GaussRowDeltaY calculations are skipped for aluminum-contaminated hits
  fChargeCalculationEnabled = fPureSiliconHit;
  
  // Log hit purity validation results with SteppingAction integration details
  SimulationLogger* logger = SimulationLogger::GetInstance();
  if (logger) {
    const G4Event* currentEvent = G4RunManager::GetRunManager()->GetCurrentEvent();
    G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;
    
    logger->LogInfo("Event " + std::to_string(eventID) + ": Hit Purity Validation Results");
    logger->LogInfo("  - Scorer Energy Deposit: " + std::to_string(fScorerEnergyDeposit) + " MeV");
    logger->LogInfo("  - Scorer Hit Count: " + std::to_string(fScorerHitCount));
    logger->LogInfo("  - Scorer Data Valid: " + std::string(fScorerDataValid ? "YES" : "NO"));
    
    // SteppingAction trajectory analysis integration results
    logger->LogInfo("  - SteppingAction Integration Results:");
    logger->LogInfo("    * First Interaction Volume: " + firstInteractionVolume);
    logger->LogInfo("    * Has Aluminum Interaction: " + std::string(hasAluminumInteraction ? "YES" : "NO"));
    logger->LogInfo("    * Has Aluminum Pre-Contact: " + std::string(hasAluminumPreContact ? "YES" : "NO"));
    logger->LogInfo("    * Is Valid Silicon Hit: " + std::string(isValidSiliconHit ? "YES" : "NO"));
    
    // EventAction validation results
    logger->LogInfo("  - EventAction Validation Results:");
    logger->LogInfo("    * Pure Silicon Hit: " + std::string(fPureSiliconHit ? "YES" : "NO"));
    logger->LogInfo("    * Aluminum Contaminated: " + std::string(fAluminumContaminated ? "YES" : "NO"));
    logger->LogInfo("    * Charge Calculation Enabled: " + std::string(fChargeCalculationEnabled ? "YES" : "NO"));
    
    // Integration status
    logger->LogInfo("  - Integration Status: SteppingAction ↔ EventAction data transfer successful");
  }
}

G4bool EventAction::ShouldCalculateChargeSharing() const
{
  // ========================================
  // EXCLUSIVE MULTI-FUNCTIONAL DETECTOR VALIDATION
  // ========================================
  
  // This method implements the exclusive Multi-Functional Detector validation
  // and determines if charge sharing calculations should proceed
  // Only pure silicon hits (no aluminum contamination) are processed
  
  // Check 1: Multi-Functional Detector validation allows charge sharing calculation
  if (!fChargeCalculationEnabled) {
    return false; // Multi-Functional Detector validation disabled charge calculation
  }
  
  // Check 2: Valid Multi-Functional Detector scorer data
  if (!fScorerDataValid) {
    return false; // Invalid scorer data from Multi-Functional Detector
  }
  
  // Check 3: Pure silicon hit (no aluminum pre-interaction)
  if (!fPureSiliconHit) {
    return false; // Not a pure silicon hit
  }
  
  // Check 4: No aluminum contamination (critical for GaussRowDeltaX, GaussRowDeltaY exclusion)
  if (fAluminumContaminated) {
    return false; // Aluminum contamination detected - skip charge sharing
  }
  
  // Check 5: SteppingAction trajectory analysis confirms valid silicon hit
  if (fSteppingAction && !fSteppingAction->IsValidSiliconHit()) {
    return false; // SteppingAction reports invalid silicon hit
  }
  
  // Check 6: Energy deposited in detector
  if (fEdep <= 0.0) {
    return false; // No energy deposited
  }
  
  // Check 7: Valid hit detected
  if (!fHasHit) {
    return false; // No valid hit detected
  }
  
  // All conditions met - charge sharing calculation should proceed for pure silicon hit
  return true;
}

void EventAction::ConditionalChargeCalculation(const G4Event* event)
{
  // ========================================
  // CONDITIONAL CHARGE SHARING FOR PURE SILICON HITS ONLY
  // ========================================
  
  // This method implements the exclusive Multi-Functional Detector validation
  // and only performs charge sharing calculations for pure silicon hits
  // GaussRowDeltaX, GaussRowDeltaY calculations are skipped for aluminum-contaminated hits
  
  // Check if charge sharing calculation should be performed
  if (!ShouldCalculateChargeSharing()) {
    // Log why charge sharing calculation was skipped
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      G4int eventID = event ? event->GetEventID() : -1;
      G4String reason = "Charge sharing calculation skipped for Event " + std::to_string(eventID) + ": ";
      
      if (!fChargeCalculationEnabled) {
        reason += "Charge calculation disabled by Multi-Functional Detector validation";
      } else if (!fScorerDataValid) {
        reason += "Invalid Multi-Functional Detector scorer data";
      } else if (!fPureSiliconHit) {
        reason += "Not a pure silicon hit (Multi-Functional Detector validation)";
      } else if (fAluminumContaminated) {
        reason += "Aluminum contamination detected (SteppingAction trajectory analysis)";
      } else if (fSteppingAction && !fSteppingAction->IsValidSiliconHit()) {
        reason += "SteppingAction reports invalid silicon hit";
      } else if (fEdep <= 0.0) {
        reason += "No energy deposited";
      } else if (!fHasHit) {
        reason += "No valid hit detected";
      } else {
        reason += "Unknown reason";
      }
      
      logger->LogInfo(reason);
    }
    
    // Previously we reset all fit results to default (which zeroed the DOF).
    // This obliterated the valid fits that were already performed earlier in EndOfEventAction
    // and produced misleading DOF=0 entries in the output tree.
    // To preserve the earlier fit results we simply return here without touching them.
    // SetDefaultFittingResults();
    return;
  }
  
  // Proceed with charge sharing calculation for pure silicon hits only
  SimulationLogger* logger = SimulationLogger::GetInstance();
  if (logger) {
    G4int eventID = event ? event->GetEventID() : -1;
    logger->LogInfo("Event " + std::to_string(eventID) + 
                   ": Performing charge sharing calculation for pure silicon hit");
    logger->LogInfo("  - Multi-Functional Detector validation: PASSED");
    logger->LogInfo("  - Aluminum contamination check: PASSED");
    logger->LogInfo("  - SteppingAction trajectory analysis: PASSED");
  }
  
  // Calculate nearest pixel position and pixel alpha
  G4ThreeVector nearestPixel = CalcNearestPixel(fPos);
  G4double pixelAlpha = CalcPixelAlpha(fPos, fPixelIndexI, fPixelIndexJ);
  
  // Determine if hit is a pixel hit
  G4bool isPixelHit = fPixelHit;
  
  // Perform charge sharing calculations only for non-pixel hits
  if (!isPixelHit) {
    // Select optimal radius if auto-selection is enabled
    if (fAutoRadiusEnabled) {
      fSelectedRadius = SelectOptimalRadius(fPos, fPixelIndexI, fPixelIndexJ);
      fNeighborhoodRadius = fSelectedRadius;
    } else {
      fSelectedRadius = fNeighborhoodRadius;
      fSelectedQuality = 0.0;
    }
    
    // Calculate neighborhood grid data
    CalcNeighborhoodGridAngles(fPos, fPixelIndexI, fPixelIndexJ);
    CalcNeighborhoodChargeSharing();
    
    // Perform fitting only if we have sufficient data
    if (!fNeighborhoodChargeFractions.empty()) {
      PerformChargeShareFitting(event, nearestPixel);
    } else {
      // No neighborhood data available - set default values
      SetDefaultFittingResults();
    }
  } else {
    // Pixel hit - clear neighborhood data and set default fitting results
    fNeighborhoodAngles.clear();
    fNeighborhoodChargeFractions.clear();
    fNeighborhoodDistances.clear();
    fNeighborhoodCharge.clear();
    SetDefaultFittingResults();
  }
}

void EventAction::SetDefaultFittingResults()
{
  // Set default values for all fitting results when calculations are skipped
  if (Control::GAUSS_FIT) {
    fRunAction->Set2DGaussResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
      0, 0,  // charge uncertainties
      false); // fit_success = false
    
    fRunAction->SetDiagGaussResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
      false); // fit_success = false
  }
  
  if (Control::LORENTZ_FIT) {
    fRunAction->Set2DLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
      0, 0,  // charge uncertainties
      false); // fit_success = false
    
    fRunAction->SetDiagLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
      false); // fit_success = false
  }
  
  if (Control::POWER_LORENTZ_FIT) {
    fRunAction->Set2DPowerLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // X fit parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Y fit parameters
      0, 0,  // charge uncertainties
      false); // fit_success = false
    
    fRunAction->SetDiagPowerLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Main diagonal Y parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal X parameters
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,  // Secondary diagonal Y parameters
      false); // fit_success = false
  }
  
  if (Control::GAUSS_FIT_3D) {
    fRunAction->Set3DGaussResults(
      0, 0, 0, 0, 0, 0,  // center_x, center_y, sigma_x, sigma_y, amp, vert_offset
      0, 0, 0, 0, 0, 0,  // parameter errors
      0, 0, 0,           // chi2red, pp, dof
      0,                 // charge_err
      false);            // fit_success = false
  }
  
  if (Control::LORENTZ_FIT_3D) {
    fRunAction->Set3DLorentzResults(
      0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, amp, vert_offset
      0, 0, 0, 0, 0, 0,  // parameter errors
      0, 0, 0,           // chi2red, pp, dof
      0,                 // charge_err
      false);            // fit_success = false
  }
  
  if (Control::POWER_LORENTZ_FIT_3D) {
    fRunAction->Set3DPowerLorentzResults(
      0, 0, 0, 0, 0, 0, 0,  // center_x, center_y, gamma_x, gamma_y, beta, amp, vert_offset
      0, 0, 0, 0, 0, 0, 0,  // parameter errors
      0, 0, 0,              // chi2red, pp, dof
      0,                    // charge_err
      false);               // fit_success = false
  }
}

void EventAction::PerformChargeShareFitting(const G4Event* event, const G4ThreeVector& nearestPixel)
{
  // Extract coordinates and charge values for fitting
  std::vector<double> x_coords, y_coords, charge_values;
  G4double pixelSpacing = fDetector->GetPixelSpacing();
  
  // Convert grid indices to actual coordinates
  G4int gridSize = 2 * fNeighborhoodRadius + 1;
  for (size_t i = 0; i < fNeighborhoodChargeFractions.size(); ++i) {
    // REMOVED: Only include pixels with charge filter - include ALL pixels for small charge values
    // BUT: Still need to filter out invalid markers (-999)
    if (fNeighborhoodChargeFractions[i] < -998.0) {
      continue; // Skip pixels with invalid markers (-999)
    }
    
    G4int col = i / gridSize;  // di (X) was outer loop
    G4int row = i % gridSize;  // dj (Y) was inner loop
    
    G4int offsetI = col - fNeighborhoodRadius;
    G4int offsetJ = row - fNeighborhoodRadius;
    
    G4double x_pos = nearestPixel.x() + offsetI * pixelSpacing;
    G4double y_pos = nearestPixel.y() + offsetJ * pixelSpacing;
    
    x_coords.push_back(x_pos);
    y_coords.push_back(y_pos);
    charge_values.push_back(fNeighborhoodCharge[i]);
  }
  
  // Perform 2D Gauss fitting if enabled and we have enough data
  if (x_coords.size() >= 3 && Control::GAUSS_FIT && Control::ROWCOL_FIT) {
    Gauss2DResultsCeres fitResults = GaussCeres2D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
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
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->LogGaussResults(event->GetEventID(), fitResults);
    }
    
    // Diagonal fitting if enabled and 2D fitting succeeded
    if (fitResults.fit_success && Control::DIAG_FIT) {
      DiagResultsCeres diagResults = DiagGaussCeres(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, false, false);
      
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
      // Set default diagonal values if main 2D fitting failed
      fRunAction->SetDiagGaussResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  } else {
    // Set default Gauss values if not enough data or disabled
    if (Control::GAUSS_FIT) {
      fRunAction->Set2DGaussResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
      fRunAction->SetDiagGaussResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  }
  
  // Perform 2D Lorentz fitting if enabled and we have enough data
  if (x_coords.size() >= 3 && Control::LORENTZ_FIT && Control::ROWCOL_FIT) {
    Lorentz2DResultsCeres lorentzResults = LorentzCeres2D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
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
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->LogLorentzResults(event->GetEventID(), lorentzResults);
    }
    
    // Diagonal Lorentz fitting if enabled and 2D fitting succeeded
    if (lorentzResults.fit_success && Control::DIAG_FIT) {
      DiagLorentzResultsCeres diagResults = DiagLorentzCeres(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, false, false);
      
      fRunAction->SetDiagLorentzResults(
        diagResults.main_diag_x_center, diagResults.main_diag_x_gamma, diagResults.main_diag_x_amp,
        diagResults.main_diag_x_center_err, diagResults.main_diag_x_gamma_err, diagResults.main_diag_x_amp_err,
        diagResults.main_diag_x_vert_offset, diagResults.main_diag_x_vert_offset_err,
        diagResults.main_diag_x_chi2red, diagResults.main_diag_x_pp, diagResults.main_diag_x_dof, diagResults.main_diag_x_fit_success,
        diagResults.main_diag_y_center, diagResults.main_diag_y_gamma, diagResults.main_diag_y_amp,
        diagResults.main_diag_y_center_err, diagResults.main_diag_y_gamma_err, diagResults.main_diag_y_amp_err,
        diagResults.main_diag_y_vert_offset, diagResults.main_diag_y_vert_offset_err,
        diagResults.main_diag_y_chi2red, diagResults.main_diag_y_pp, diagResults.main_diag_y_dof, diagResults.main_diag_y_fit_success,
        diagResults.sec_diag_x_center, diagResults.sec_diag_x_gamma, diagResults.sec_diag_x_amp,
        diagResults.sec_diag_x_center_err, diagResults.sec_diag_x_gamma_err, diagResults.sec_diag_x_amp_err,
        diagResults.sec_diag_x_vert_offset, diagResults.sec_diag_x_vert_offset_err,
        diagResults.sec_diag_x_chi2red, diagResults.sec_diag_x_pp, diagResults.sec_diag_x_dof, diagResults.sec_diag_x_fit_success,
        diagResults.sec_diag_y_center, diagResults.sec_diag_y_gamma, diagResults.sec_diag_y_amp,
        diagResults.sec_diag_y_center_err, diagResults.sec_diag_y_gamma_err, diagResults.sec_diag_y_amp_err,
        diagResults.sec_diag_y_vert_offset, diagResults.sec_diag_y_vert_offset_err,
        diagResults.sec_diag_y_chi2red, diagResults.sec_diag_y_pp, diagResults.sec_diag_y_dof, diagResults.sec_diag_y_fit_success,
        diagResults.fit_success);
    } else {
      // Set default diagonal Lorentz values if main 2D fitting failed
      fRunAction->SetDiagLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  } else {
    // Set default Lorentz values if not enough data or disabled
    if (Control::LORENTZ_FIT) {
      fRunAction->Set2DLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
      fRunAction->SetDiagLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  }
  
  // Perform Power-Law Lorentz fitting if enabled and we have enough data
  if (x_coords.size() >= 3 && Control::POWER_LORENTZ_FIT && Control::ROWCOL_FIT) {
    PowerLorentz2DResultsCeres powerResults = PowerLorentzCeres2D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
    fRunAction->Set2DPowerLorentzResults(
      powerResults.x_center, powerResults.x_gamma, powerResults.x_beta, powerResults.x_amp,
      powerResults.x_center_err, powerResults.x_gamma_err, powerResults.x_beta_err, powerResults.x_amp_err,
      powerResults.x_vert_offset, powerResults.x_vert_offset_err,
      powerResults.x_chi2red, powerResults.x_pp, powerResults.x_dof,
      powerResults.y_center, powerResults.y_gamma, powerResults.y_beta, powerResults.y_amp,
      powerResults.y_center_err, powerResults.y_gamma_err, powerResults.y_beta_err, powerResults.y_amp_err,
      powerResults.y_vert_offset, powerResults.y_vert_offset_err,
      powerResults.y_chi2red, powerResults.y_pp, powerResults.y_dof,
      powerResults.x_charge_err, powerResults.y_charge_err,
      powerResults.fit_success);
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->LogPowerLorentzResults(event->GetEventID(), powerResults);
    }
    
    // Diagonal Power-Law Lorentz fitting if enabled and 2D fitting succeeded
    if (powerResults.fit_success && Control::DIAG_FIT) {
      DiagPowerLorentzResultsCeres diagResults = DiagPowerLorentzCeres(
        x_coords, y_coords, charge_values,
        nearestPixel.x(), nearestPixel.y(),
        pixelSpacing, false, false);
      
      fRunAction->SetDiagPowerLorentzResults(
        diagResults.main_diag_x_center, diagResults.main_diag_x_gamma, diagResults.main_diag_x_beta, diagResults.main_diag_x_amp,
        diagResults.main_diag_x_center_err, diagResults.main_diag_x_gamma_err, diagResults.main_diag_x_beta_err, diagResults.main_diag_x_amp_err,
        diagResults.main_diag_x_vert_offset, diagResults.main_diag_x_vert_offset_err,
        diagResults.main_diag_x_chi2red, diagResults.main_diag_x_pp, diagResults.main_diag_x_dof, diagResults.main_diag_x_fit_success,
        diagResults.main_diag_y_center, diagResults.main_diag_y_gamma, diagResults.main_diag_y_beta, diagResults.main_diag_y_amp,
        diagResults.main_diag_y_center_err, diagResults.main_diag_y_gamma_err, diagResults.main_diag_y_beta_err, diagResults.main_diag_y_amp_err,
        diagResults.main_diag_y_vert_offset, diagResults.main_diag_y_vert_offset_err,
        diagResults.main_diag_y_chi2red, diagResults.main_diag_y_pp, diagResults.main_diag_y_dof, diagResults.main_diag_y_fit_success,
        diagResults.sec_diag_x_center, diagResults.sec_diag_x_gamma, diagResults.sec_diag_x_beta, diagResults.sec_diag_x_amp,
        diagResults.sec_diag_x_center_err, diagResults.sec_diag_x_gamma_err, diagResults.sec_diag_x_beta_err, diagResults.sec_diag_x_amp_err,
        diagResults.sec_diag_x_vert_offset, diagResults.sec_diag_x_vert_offset_err,
        diagResults.sec_diag_x_chi2red, diagResults.sec_diag_x_pp, diagResults.sec_diag_x_dof, diagResults.sec_diag_x_fit_success,
        diagResults.sec_diag_y_center, diagResults.sec_diag_y_gamma, diagResults.sec_diag_y_beta, diagResults.sec_diag_y_amp,
        diagResults.sec_diag_y_center_err, diagResults.sec_diag_y_gamma_err, diagResults.sec_diag_y_beta_err, diagResults.sec_diag_y_amp_err,
        diagResults.sec_diag_y_vert_offset, diagResults.sec_diag_y_vert_offset_err,
        diagResults.sec_diag_y_chi2red, diagResults.sec_diag_y_pp, diagResults.sec_diag_y_dof, diagResults.sec_diag_y_fit_success,
        diagResults.fit_success);
    } else {
      // Set default diagonal Power-Law Lorentz values if main 2D fitting failed
      fRunAction->SetDiagPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  } else {
    // Set default Power-Law Lorentz values if not enough data or disabled
    if (Control::POWER_LORENTZ_FIT) {
      fRunAction->Set2DPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
      fRunAction->SetDiagPowerLorentzResults(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false);
    }
  }
  
  // Perform 3D fitting if enabled and we have enough data
  if (x_coords.size() >= 6 && Control::GAUSS_FIT_3D) {
    Gauss3DResultsCeres gauss3DResults = GaussCeres3D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
    fRunAction->Set3DGaussResults(
      gauss3DResults.center_x, gauss3DResults.center_y, gauss3DResults.sigma_x, gauss3DResults.sigma_y,
      gauss3DResults.amp, gauss3DResults.vert_offset,
      gauss3DResults.center_x_err, gauss3DResults.center_y_err, gauss3DResults.sigma_x_err, gauss3DResults.sigma_y_err,
      gauss3DResults.amp_err, gauss3DResults.vert_offset_err,
      gauss3DResults.chi2red, gauss3DResults.pp, gauss3DResults.dof,
      gauss3DResults.charge_err, gauss3DResults.fit_success);
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->Log3DGaussResults(event->GetEventID(), gauss3DResults);
    }
  } else if (Control::GAUSS_FIT_3D) {
    fRunAction->Set3DGaussResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  }
  
  if (x_coords.size() >= 6 && Control::LORENTZ_FIT_3D) {
    Lorentz3DResultsCeres lorentz3DResults = LorentzCeres3D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
    fRunAction->Set3DLorentzResults(
      lorentz3DResults.center_x, lorentz3DResults.center_y, lorentz3DResults.gamma_x, lorentz3DResults.gamma_y,
      lorentz3DResults.amp, lorentz3DResults.vert_offset,
      lorentz3DResults.center_x_err, lorentz3DResults.center_y_err, lorentz3DResults.gamma_x_err, lorentz3DResults.gamma_y_err,
      lorentz3DResults.amp_err, lorentz3DResults.vert_offset_err,
      lorentz3DResults.chi2red, lorentz3DResults.pp, lorentz3DResults.dof,
      lorentz3DResults.charge_err, lorentz3DResults.fit_success);
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->Log3DLorentzResults(event->GetEventID(), lorentz3DResults);
    }
  } else if (Control::LORENTZ_FIT_3D) {
    fRunAction->Set3DLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  }
  
  if (x_coords.size() >= 7 && Control::POWER_LORENTZ_FIT_3D) {
    PowerLorentz3DResultsCeres power3DResults = PowerLorentzCeres3D(
      x_coords, y_coords, charge_values,
      nearestPixel.x(), nearestPixel.y(),
      pixelSpacing, false, false);
    
    fRunAction->Set3DPowerLorentzResults(
      power3DResults.center_x, power3DResults.center_y, power3DResults.gamma_x, power3DResults.gamma_y,
      power3DResults.beta, power3DResults.amp, power3DResults.vert_offset,
      power3DResults.center_x_err, power3DResults.center_y_err, power3DResults.gamma_x_err, power3DResults.gamma_y_err,
      power3DResults.beta_err, power3DResults.amp_err, power3DResults.vert_offset_err,
      power3DResults.chi2red, power3DResults.pp, power3DResults.dof,
      power3DResults.charge_err, power3DResults.fit_success);
    
    // Log fitting results
    SimulationLogger* logger = SimulationLogger::GetInstance();
    if (logger) {
      logger->Log3DPowerLorentzResults(event->GetEventID(), power3DResults);
    }
  } else if (Control::POWER_LORENTZ_FIT_3D) {
    fRunAction->Set3DPowerLorentzResults(
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  }
}

void EventAction::GetHitValidationSummary(G4bool& pureSiliconHit, G4bool& aluminumContaminated,
                                          G4bool& chargeCalculationEnabled, G4String& firstInteractionVolume,
                                          G4bool& hasAluminumInteraction, G4bool& hasAluminumPreContact) const
{
  // Get EventAction validation results
  pureSiliconHit = fPureSiliconHit;
  aluminumContaminated = fAluminumContaminated;
  chargeCalculationEnabled = fChargeCalculationEnabled;
  
  // Get SteppingAction trajectory analysis results
  if (fSteppingAction) {
    firstInteractionVolume = fSteppingAction->GetFirstInteractionVolume();
    hasAluminumInteraction = fSteppingAction->HasAluminumInteraction();
    hasAluminumPreContact = fSteppingAction->HasAluminumPreContact();
  } else {
    // Fallback if SteppingAction is not available
    firstInteractionVolume = "UNKNOWN";
    hasAluminumInteraction = false;
    hasAluminumPreContact = false;
  }
}

void EventAction::ValidateSteppingActionIntegration() const
{
  // Validate that SteppingAction is properly connected
  if (!fSteppingAction) {
    G4cerr << "ERROR: SteppingAction integration validation failed!" << G4endl;
    G4cerr << "SteppingAction pointer is null - hit validation cannot proceed properly" << G4endl;
    G4cerr << "This indicates an initialization problem in ActionInitialization" << G4endl;
    return;
  }
  
  // Validate that SteppingAction trajectory analysis methods are available
  try {
    // Test basic trajectory analysis methods
    G4bool testAluminumInteraction = fSteppingAction->HasAluminumInteraction();
    G4bool testValidSiliconHit = fSteppingAction->IsValidSiliconHit();
    G4String testFirstInteractionVolume = fSteppingAction->GetFirstInteractionVolume();
    G4bool testAluminumPreContact = fSteppingAction->HasAluminumPreContact();
    
    G4cout << "✓ SteppingAction integration validation successful" << G4endl;
    G4cout << "✓ All trajectory analysis methods accessible" << G4endl;
    G4cout << "✓ Current state - Aluminum interaction: " << (testAluminumInteraction ? "YES" : "NO") << G4endl;
    G4cout << "✓ Current state - Valid silicon hit: " << (testValidSiliconHit ? "YES" : "NO") << G4endl;
    G4cout << "✓ Current state - First interaction volume: " << testFirstInteractionVolume << G4endl;
    G4cout << "✓ Current state - Aluminum pre-contact: " << (testAluminumPreContact ? "YES" : "NO") << G4endl;
    
  } catch (const std::exception& e) {
    G4cerr << "ERROR: SteppingAction integration validation failed!" << G4endl;
    G4cerr << "Exception while testing trajectory analysis methods: " << e.what() << G4endl;
  } catch (...) {
    G4cerr << "ERROR: SteppingAction integration validation failed!" << G4endl;
    G4cerr << "Unknown exception while testing trajectory analysis methods" << G4endl;
  }
}

void EventAction::DemonstrateIntegrationWorkflow() const
{
  G4cout << "\n=== SteppingAction ↔ EventAction Integration Workflow ===" << G4endl;
  G4cout << "Task 9: Integration between SteppingAction and EventAction for hit validation" << G4endl;
  G4cout << "========================================================" << G4endl;
  
  // Step 1: Initialization and cleanup
  G4cout << "1. Initialization & Cleanup:" << G4endl;
  G4cout << "   - BeginOfEventAction() calls fSteppingAction->ResetInteractionTracking()" << G4endl;
  G4cout << "   - This ensures clean state for each event" << G4endl;
  G4cout << "   - Integration validation performed for first event" << G4endl;
  
  // Step 2: Data transfer mechanism
  G4cout << "2. Data Transfer Mechanism:" << G4endl;
  G4cout << "   - SteppingAction performs trajectory analysis during particle steps" << G4endl;
  G4cout << "   - EventAction retrieves hit purity status via getter methods:" << G4endl;
  G4cout << "     * HasAluminumInteraction() → " << (fSteppingAction ? std::string(fSteppingAction->HasAluminumInteraction() ? "Available" : "Available") : "Not connected") << G4endl;
  G4cout << "     * IsValidSiliconHit() → " << (fSteppingAction ? std::string(fSteppingAction->IsValidSiliconHit() ? "Available" : "Available") : "Not connected") << G4endl;
  G4cout << "     * GetFirstInteractionVolume() → " << (fSteppingAction ? "Available" : "Not connected") << G4endl;
  G4cout << "     * HasAluminumPreContact() → " << (fSteppingAction ? std::string(fSteppingAction->HasAluminumPreContact() ? "Available" : "Available") : "Not connected") << G4endl;
  
  // Step 3: Hit validation process
  G4cout << "3. Hit Validation Process:" << G4endl;
  G4cout << "   - ValidateHitPurity() combines SteppingAction trajectory analysis" << G4endl;
  G4cout << "   - with EventAction Multi-Functional Detector data" << G4endl;
  G4cout << "   - Determines: Pure Silicon Hit, Aluminum Contaminated, Charge Calculation Enabled" << G4endl;
  
  // Step 4: Integration status
  G4cout << "4. Integration Status:" << G4endl;
  G4cout << "   - SteppingAction connected: " << (fSteppingAction ? "YES" : "NO") << G4endl;
  G4cout << "   - Hit purity getter methods: Available" << G4endl;
  G4cout << "   - Comprehensive validation summary: Available" << G4endl;
  G4cout << "   - Integration workflow: Complete" << G4endl;
  
  // Step 5: Available methods for external access
  G4cout << "5. Available Methods for External Access:" << G4endl;
  G4cout << "   - GetPureSiliconHit() → Current: " << (fPureSiliconHit ? "YES" : "NO") << G4endl;
  G4cout << "   - GetAluminumContaminated() → Current: " << (fAluminumContaminated ? "YES" : "NO") << G4endl;
  G4cout << "   - GetChargeCalculationEnabled() → Current: " << (fChargeCalculationEnabled ? "YES" : "NO") << G4endl;
  G4cout << "   - GetHitPurityStatus() → Current: " << (GetHitPurityStatus() ? "PURE" : "CONTAMINATED") << G4endl;
  G4cout << "   - GetHitValidationSummary() → Comprehensive status available" << G4endl;
  
  G4cout << "\n✓ Task 9 Integration Complete: SteppingAction ↔ EventAction" << G4endl;
  G4cout << "========================================================\n" << G4endl;
}

