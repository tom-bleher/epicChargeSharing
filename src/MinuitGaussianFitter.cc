#include "MinuitGaussianFitter.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ROOT includes
#include "TMath.h"
#include "TMinuit.h"

// Static member initialization
std::vector<G4double> MinuitGaussianFitter::fStaticXCoords;
std::vector<G4double> MinuitGaussianFitter::fStaticYCoords;
std::vector<G4double> MinuitGaussianFitter::fStaticZValues;
std::vector<G4double> MinuitGaussianFitter::fStaticZErrors;
const MinuitGaussianFitter* MinuitGaussianFitter::fStaticInstance = nullptr;

MinuitGaussianFitter::MinuitGaussianFitter(const DetectorGeometry& detector_geometry) 
    : fDetectorGeometry(detector_geometry), fMinuit(nullptr),
      fCenterPixelX(0.0), fCenterPixelY(0.0), fConstrainToCenterPixel(false)
{
    // Initialize Minuit with 7 parameters
    try {
        fMinuit = new TMinuit(fNParams);
        if (fMinuit) {
            fMinuit->SetFCN(MinuitFcn);
            fMinuit->SetPrintLevel(-1); // Suppress output by default
        } else {
            G4cout << "Warning: Failed to create TMinuit object" << G4endl;
        }
    } catch (const std::exception& e) {
        G4cout << "Warning: Exception while creating TMinuit: " << e.what() << G4endl;
        fMinuit = nullptr;
    } catch (...) {
        G4cout << "Warning: Unknown exception while creating TMinuit" << G4endl;
        fMinuit = nullptr;
    }
}

MinuitGaussianFitter::~MinuitGaussianFitter()
{
    // Clear static instance pointer to avoid dangling references
    if (fStaticInstance == this) {
        fStaticInstance = nullptr;
    }
    
    // Clear static data vectors to avoid memory issues
    fStaticXCoords.clear();
    fStaticYCoords.clear();
    fStaticZValues.clear();
    fStaticZErrors.clear();
    
    // Thread-safe cleanup of TMinuit object
    if (fMinuit) {
        try {
            // Reset any function callback to avoid issues during destruction
            fMinuit->SetFCN(nullptr);
            
            // Clear Minuit's internal state
            fMinuit->mncler();
            
            // Delete the TMinuit object
            delete fMinuit;
        } catch (...) {
            // Ignore any exceptions during cleanup to prevent crashes
            // This is acceptable since we're in a destructor
        }
        fMinuit = nullptr;
    }
}

void MinuitGaussianFitter::SetDetectorGeometry(const DetectorGeometry& geometry)
{
    fDetectorGeometry = geometry;
}

G4double MinuitGaussianFitter::Gaussian3DFunction(G4double x, G4double y, const G4double* params)
{
    /*
     * 3D Gaussian function with rotation
     * Parameters:
     * [0] = amplitude
     * [1] = x0 (center x)
     * [2] = y0 (center y)
     * [3] = sigma_x
     * [4] = sigma_y
     * [5] = theta (rotation angle in radians)
     * [6] = offset
     */
    
    const G4double amplitude = params[0];
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    const G4double sigma_x = params[3];
    const G4double sigma_y = params[4];
    const G4double theta = params[5];
    const G4double offset = params[6];
    
    // Avoid division by zero
    if (sigma_x <= 0 || sigma_y <= 0) {
        return offset;
    }
    
    // Rotation transformation
    const G4double cos_theta = TMath::Cos(theta);
    const G4double sin_theta = TMath::Sin(theta);
    
    const G4double x_rot = cos_theta * (x - x0) + sin_theta * (y - y0);
    const G4double y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0);
    
    // 3D Gaussian
    const G4double exponent = -(x_rot*x_rot / (2.0 * sigma_x*sigma_x) + 
                               y_rot*y_rot / (2.0 * sigma_y*sigma_y));
    
    return amplitude * TMath::Exp(exponent) + offset;
}

void MinuitGaussianFitter::MinuitFcn(G4int& npar, G4double* gin, G4double& f, G4double* par, G4int iflag)
{
    // Minuit callback function to calculate chi-squared
    G4double chi2 = 0.0;
    
    // Safety checks for static data
    if (fStaticXCoords.empty() || fStaticYCoords.empty() || fStaticZValues.empty()) {
        f = 1e10; // Return large chi2 for invalid data
        return;
    }
    
    if (fStaticXCoords.size() != fStaticYCoords.size() || 
        fStaticXCoords.size() != fStaticZValues.size()) {
        f = 1e10; // Return large chi2 for inconsistent data
        return;
    }
    
    // Safety check for parameters
    if (!par) {
        f = 1e10;
        return;
    }
    
    for (size_t i = 0; i < fStaticXCoords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(fStaticXCoords[i], fStaticYCoords[i], par);
        const G4double residual = fStaticZValues[i] - z_fit;
        const G4double error = (!fStaticZErrors.empty() && i < fStaticZErrors.size()) ? fStaticZErrors[i] : 1.0;
        
        if (error > 0) {
            chi2 += (residual * residual) / (error * error);
        }
    }
    
    // Add penalties for constraint violations if instance is available
    if (fStaticInstance) {
        G4double penalty = 0.0;
        const G4double penalty_factor = Constants::CONSTRAINT_PENALTY;
        
        // Check constraints and add penalties
        const G4double x0 = par[1];
        const G4double y0 = par[2];
        
        // Penalty for being outside detector bounds
        const G4double half_det = fStaticInstance->fDetectorGeometry.detector_size / 2.0;
        if (TMath::Abs(x0) > half_det) penalty += penalty_factor * (TMath::Abs(x0) - half_det) * 100.0;
        if (TMath::Abs(y0) > half_det) penalty += penalty_factor * (TMath::Abs(y0) - half_det) * 100.0;
        
        // Apply different constraint penalty based on center pixel constraint mode
        if (fStaticInstance->fConstrainToCenterPixel) {
            // Penalty for being outside center pixel bounds
            G4double bounds[4];
            fStaticInstance->GetPixelBounds(fStaticInstance->fCenterPixelX, fStaticInstance->fCenterPixelY, bounds);
            
            G4double pixel_penalty = 0.0;
            if (x0 < bounds[0]) pixel_penalty += (bounds[0] - x0);
            if (x0 > bounds[1]) pixel_penalty += (x0 - bounds[1]);
            if (y0 < bounds[2]) pixel_penalty += (bounds[2] - y0);
            if (y0 > bounds[3]) pixel_penalty += (y0 - bounds[3]);
            
            penalty += penalty_factor * pixel_penalty * 1000.0; // Strong penalty for leaving center pixel
        } else {
            // Original penalty for being too close to pixel center or inside pixel area
            const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
            if (fStaticInstance->IsPointInsidePixelZone(x0, y0, d0)) {
                G4double min_dist = fStaticInstance->CalculateMinDistanceToPixelCenter(x0, y0);
                if (min_dist < d0) {
                    penalty += penalty_factor * (d0 - min_dist) * 100.0;
                } else {
                    // Inside pixel area but outside d0 radius - still penalize
                    penalty += penalty_factor * 100.0;
                }
            }
        }
        
        chi2 += penalty;
    }
    
    // Ensure chi2 is finite and positive
    if (!TMath::Finite(chi2) || chi2 < 0) {
        chi2 = 1e10;
    }
    
    f = chi2;
}

G4bool MinuitGaussianFitter::IsPointInsidePixelZone(G4double x, G4double y, G4double min_distance) const
{
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Check if point is inside detector bounds
    if (TMath::Abs(x) > half_det || TMath::Abs(y) > half_det) {
        return false; // Outside detector, not in pixel zone
    }
    
    // Calculate the first pixel position
    const G4double first_pixel_pos = -half_det + fDetectorGeometry.pixel_corner_offset + fDetectorGeometry.pixel_size / 2.0;
    
    // Check each pixel center distance
    for (G4int i = 0; i < fDetectorGeometry.num_blocks_per_side; i++) {
        for (G4int j = 0; j < fDetectorGeometry.num_blocks_per_side; j++) {
            const G4double pixel_center_x = first_pixel_pos + i * fDetectorGeometry.pixel_spacing;
            const G4double pixel_center_y = first_pixel_pos + j * fDetectorGeometry.pixel_spacing;
            
            // Calculate distance from point to pixel center
            const G4double dx = x - pixel_center_x;
            const G4double dy = y - pixel_center_y;
            const G4double distance = TMath::Sqrt(dx*dx + dy*dy);
            
            // Check if within the specified minimum distance from pixel center
            if (distance < min_distance) {
                return true; // Too close to pixel center
            }
            
            // Also check if inside the pixel area itself
            const G4double pixel_half_size = fDetectorGeometry.pixel_size / 2.0;
            if (TMath::Abs(dx) <= pixel_half_size && TMath::Abs(dy) <= pixel_half_size) {
                return true; // Inside pixel area
            }
        }
    }
    
    return false; // Not in any pixel zone
}

G4double MinuitGaussianFitter::CalculateMinDistanceToPixel(G4double x, G4double y) const
{
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Check if point is inside detector bounds
    if (TMath::Abs(x) > half_det || TMath::Abs(y) > half_det) {
        return 0.0; // Outside detector
    }
    
    G4double min_distance = 1e6; // Large initial value
    const G4double first_pixel_pos = -half_det + fDetectorGeometry.pixel_corner_offset + fDetectorGeometry.pixel_size / 2.0;
    const G4double pixel_half_size = fDetectorGeometry.pixel_size / 2.0;
    
    // Check distance to each pixel
    for (G4int i = 0; i < fDetectorGeometry.num_blocks_per_side; i++) {
        for (G4int j = 0; j < fDetectorGeometry.num_blocks_per_side; j++) {
            const G4double pixel_center_x = first_pixel_pos + i * fDetectorGeometry.pixel_spacing;
            const G4double pixel_center_y = first_pixel_pos + j * fDetectorGeometry.pixel_spacing;
            
            // Calculate distance to pixel edge (rectangular)
            const G4double dx = TMath::Max(0.0, TMath::Abs(x - pixel_center_x) - pixel_half_size);
            const G4double dy = TMath::Max(0.0, TMath::Abs(y - pixel_center_y) - pixel_half_size);
            const G4double distance = TMath::Sqrt(dx*dx + dy*dy);
            
            min_distance = TMath::Min(min_distance, distance);
        }
    }
    
    return min_distance;
}

G4double MinuitGaussianFitter::CalculateMinDistanceToPixelCenter(G4double x, G4double y) const
{
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Check if point is inside detector bounds
    if (TMath::Abs(x) > half_det || TMath::Abs(y) > half_det) {
        return 1e6; // Outside detector, return large distance
    }
    
    G4double min_distance = 1e6; // Large initial value
    const G4double first_pixel_pos = -half_det + fDetectorGeometry.pixel_corner_offset + fDetectorGeometry.pixel_size / 2.0;
    
    // Check distance to each pixel center
    for (G4int i = 0; i < fDetectorGeometry.num_blocks_per_side; i++) {
        for (G4int j = 0; j < fDetectorGeometry.num_blocks_per_side; j++) {
            const G4double pixel_center_x = first_pixel_pos + i * fDetectorGeometry.pixel_spacing;
            const G4double pixel_center_y = first_pixel_pos + j * fDetectorGeometry.pixel_spacing;
            
            // Calculate distance to pixel center
            const G4double dx = x - pixel_center_x;
            const G4double dy = y - pixel_center_y;
            const G4double distance = TMath::Sqrt(dx*dx + dy*dy);
            
            min_distance = TMath::Min(min_distance, distance);
        }
    }
    
    return min_distance;
}

void MinuitGaussianFitter::GetPixelBounds(G4double center_x, G4double center_y, G4double* bounds) const
{
    const G4double pixel_half_size = fDetectorGeometry.pixel_size / 2.0;
    bounds[0] = center_x - pixel_half_size; // x_min
    bounds[1] = center_x + pixel_half_size; // x_max
    bounds[2] = center_y - pixel_half_size; // y_min
    bounds[3] = center_y + pixel_half_size; // y_max
}

void MinuitGaussianFitter::SetCenterPixelPosition(G4double center_x, G4double center_y)
{
    fCenterPixelX = center_x;
    fCenterPixelY = center_y;
    fConstrainToCenterPixel = true;
}

G4bool MinuitGaussianFitter::CheckConstraints(const G4double* params, G4bool verbose) const
{
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    
    // Check 1: Center must be within detector bounds
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    if (TMath::Abs(x0) > half_det || TMath::Abs(y0) > half_det) {
        if (verbose) {
            G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                   << ") outside detector bounds ±" << half_det << " mm" << G4endl;
        }
        return false;
    }
    
    // Check 2: If center pixel constraint is enabled, center must be within center pixel bounds
    if (fConstrainToCenterPixel) {
        G4double bounds[4];
        GetPixelBounds(fCenterPixelX, fCenterPixelY, bounds);
        
        if (x0 < bounds[0] || x0 > bounds[1] || y0 < bounds[2] || y0 > bounds[3]) {
            if (verbose) {
                G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                       << ") outside center pixel bounds [" << bounds[0] << ", " << bounds[1] 
                       << "] × [" << bounds[2] << ", " << bounds[3] << "] mm" << G4endl;
            }
            return false;
        }
    } else {
        // Check 2 (original): Center must be outside pixel area and d0 (10 micron) radius from pixel center
        const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
        if (IsPointInsidePixelZone(x0, y0, d0)) {
            if (verbose) {
                G4double min_dist = CalculateMinDistanceToPixelCenter(x0, y0);
                G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                       << ") too close to pixel center (min distance: " << min_dist 
                       << " mm, required: " << d0 << " mm)" << G4endl;
            }
            return false;
        }
    }
    
    return true;
}

void MinuitGaussianFitter::CalculateInitialGuess(const std::vector<G4double>& x_coords,
                                                 const std::vector<G4double>& y_coords,
                                                 const std::vector<G4double>& z_values,
                                                 G4double* initialParams,
                                                 G4int strategy)
{
    if (x_coords.empty() || y_coords.empty() || z_values.empty()) {
        // Set default values if no data
        initialParams[0] = 1.0;  // amplitude
        initialParams[1] = 0.0;  // x0
        initialParams[2] = 0.0;  // y0
        initialParams[3] = Constants::DEFAULT_SIGMA_ESTIMATE;  // sigma_x
        initialParams[4] = Constants::DEFAULT_SIGMA_ESTIMATE;  // sigma_y
        initialParams[5] = 0.0;  // theta
        initialParams[6] = 0.0;  // offset
        return;
    }
    
    // Calculate basic statistics
    const G4double z_min = *std::min_element(z_values.begin(), z_values.end());
    const G4double z_max = *std::max_element(z_values.begin(), z_values.end());
    
    G4double weighted_x = 0.0, weighted_y = 0.0, sum_z = 0.0;
    
    // Different strategies for initial guess
    switch (strategy) {
        case 0: // Weighted center of mass (default)
            for (size_t i = 0; i < x_coords.size(); ++i) {
                if (z_values[i] > z_min) {
                    G4double weight = z_values[i] - z_min;
                    sum_z += weight;
                    weighted_x += x_coords[i] * weight;
                    weighted_y += y_coords[i] * weight;
                }
            }
            if (sum_z > 0) {
                weighted_x /= sum_z;
                weighted_y /= sum_z;
            }
            break;
            
        case 1: // Maximum value position
            {
                size_t max_idx = std::distance(z_values.begin(), std::max_element(z_values.begin(), z_values.end()));
                weighted_x = x_coords[max_idx];
                weighted_y = y_coords[max_idx];
            }
            break;
            
        case 2: // Geometric center of data
            weighted_x = std::accumulate(x_coords.begin(), x_coords.end(), 0.0) / x_coords.size();
            weighted_y = std::accumulate(y_coords.begin(), y_coords.end(), 0.0) / y_coords.size();
            break;
            
        default:
            strategy = 0; // Fall back to default
            break;
    }
    
    // Ensure initial center is in allowed region
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    weighted_x = TMath::Max(-half_det * 0.8, TMath::Min(half_det * 0.8, weighted_x));
    weighted_y = TMath::Max(-half_det * 0.8, TMath::Min(half_det * 0.8, weighted_y));
    
    // If initial guess is in pixel zone, move it to center
    const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
    if (IsPointInsidePixelZone(weighted_x, weighted_y, d0)) {
        weighted_x = 0.0;
        weighted_y = 0.0;
    }
    
    // Calculate initial sigma estimates
    G4double sigma_x_est = Constants::DEFAULT_SIGMA_ESTIMATE; // Default 100 microns
    G4double sigma_y_est = Constants::DEFAULT_SIGMA_ESTIMATE;
    
    // Set initial parameters
    initialParams[0] = z_max - z_min;  // amplitude
    initialParams[1] = weighted_x;     // x0
    initialParams[2] = weighted_y;     // y0
    initialParams[3] = sigma_x_est;    // sigma_x
    initialParams[4] = sigma_y_est;    // sigma_y
    initialParams[5] = 0.0;            // theta
    initialParams[6] = z_min;          // offset
}

void MinuitGaussianFitter::CalculateResidualStats(const std::vector<G4double>& x_coords,
                                                  const std::vector<G4double>& y_coords,
                                                  const std::vector<G4double>& z_values,
                                                  const G4double* fitParams,
                                                  G4double& mean, G4double& std_dev)
{
    if (x_coords.empty()) {
        mean = 0.0;
        std_dev = 0.0;
        return;
    }
    
    // Calculate residuals
    std::vector<G4double> residuals;
    residuals.reserve(x_coords.size());
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], fitParams);
        residuals.push_back(z_values[i] - z_fit);
    }
    
    // Calculate mean
    mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    
    // Calculate standard deviation
    G4double variance = 0.0;
    for (const G4double& residual : residuals) {
        variance += (residual - mean) * (residual - mean);
    }
    variance /= residuals.size();
    std_dev = TMath::Sqrt(variance);
}

MinuitGaussianFitter::FitResults MinuitGaussianFitter::FitGaussian3D(const std::vector<G4double>& x_coords,
                                                                     const std::vector<G4double>& y_coords,
                                                                     const std::vector<G4double>& z_values,
                                                                     const std::vector<G4double>& z_errors,
                                                                     G4bool verbose)
{
    FitResults results;
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "MinuitGaussianFitter::FitGaussian3D - Error: Inconsistent input array sizes" << G4endl;
        }
        return results;
    }
    
    if (x_coords.size() < 7) { // Need at least as many points as parameters
        if (verbose) {
            G4cout << "MinuitGaussianFitter::FitGaussian3D - Error: Insufficient data points (" 
                   << x_coords.size() << " < 7)" << G4endl;
        }
        return results;
    }
    
    // Early safety check for TMinuit object
    if (!fMinuit) {
        if (verbose) {
            G4cout << "MinuitGaussianFitter::FitGaussian3D - Error: TMinuit object is null" << G4endl;
        }
        return results;
    }
    
    if (verbose) {
        G4cout << "\n=== Minuit Gaussian3D Fitting ===" << G4endl;
        G4cout << "Data points: " << x_coords.size() << G4endl;
        fMinuit->SetPrintLevel(0); // Some output for verbose mode
    } else {
        fMinuit->SetPrintLevel(-1); // Suppress output
    }
    
    results.fit_type = ALL_DATA;
    results.n_points = x_coords.size();
    
    // Set static data for Minuit callback - ensure thread safety
    // Note: This is not truly thread-safe, but Geant4's multithreading model
    // ensures each worker thread has its own EventAction with its own fitter
    fStaticXCoords = x_coords;
    fStaticYCoords = y_coords;
    fStaticZValues = z_values;
    fStaticZErrors = z_errors;
    fStaticInstance = this;
    
    // Try multiple fitting strategies
    G4bool fit_found = false;
    G4double best_chi2 = 1e10;
    G4double best_params[fNParams];
    G4double best_errors[fNParams];
    
    for (G4int attempt = 0; attempt < fMaxFitAttempts && !fit_found; ++attempt) {
        if (verbose) {
            G4cout << "\n--- Fit Attempt " << (attempt + 1) << " ---" << G4endl;
        }
        
        try {
            // Safety check before each fitting attempt
            if (!fMinuit) {
                if (verbose) {
                    G4cout << "  Error: TMinuit object became null during fitting" << G4endl;
                }
                break;
            }
            
            // Clear previous fit
            fMinuit->mncler();
            
            // Calculate initial parameter estimates
            G4double fitParams[fNParams];
            CalculateInitialGuess(x_coords, y_coords, z_values, fitParams, attempt);
            
            if (verbose) {
                G4cout << "Initial parameters (strategy " << attempt << "):" << G4endl;
                G4cout << "  Amplitude: " << fitParams[0] << G4endl;
                G4cout << "  Center: (" << fitParams[1] << ", " << fitParams[2] << ") mm" << G4endl;
                G4cout << "  Sigma: (" << fitParams[3] << ", " << fitParams[4] << ") mm" << G4endl;
                G4cout << "  Theta: " << fitParams[5] << " rad" << G4endl;
                G4cout << "  Offset: " << fitParams[6] << G4endl;
            }
            
            // Define parameters without parameter value constraints
            G4int ierflg = 0;
            
            // Only constrain center coordinates to detector bounds
            const G4double half_det = fDetectorGeometry.detector_size / 2.0;
            
            fMinuit->mnparm(0, "amplitude", fitParams[0], TMath::Abs(fitParams[0]) * Constants::DEFAULT_AMPLITUDE_FRACTION, 0, 0, ierflg);
            fMinuit->mnparm(1, "x0", fitParams[1], Constants::POSITION_STEP_SIZE, -half_det, half_det, ierflg);
            fMinuit->mnparm(2, "y0", fitParams[2], Constants::POSITION_STEP_SIZE, -half_det, half_det, ierflg);
            fMinuit->mnparm(3, "sigma_x", fitParams[3], Constants::SIGMA_STEP_SIZE, 0, 0, ierflg);
            fMinuit->mnparm(4, "sigma_y", fitParams[4], Constants::SIGMA_STEP_SIZE, 0, 0, ierflg);
            fMinuit->mnparm(5, "theta", fitParams[5], Constants::ANGLE_STEP_SIZE, -TMath::Pi(), TMath::Pi(), ierflg);
            fMinuit->mnparm(6, "offset", fitParams[6], TMath::Abs(fitParams[6]) * Constants::DEFAULT_AMPLITUDE_FRACTION, 0, 0, ierflg);
            
            // Set strategy (0=fast, 1=default, 2=reliable)
            fMinuit->SetMaxIterations(1000);
            G4double strategy_param = static_cast<G4double>(attempt);
            fMinuit->mnexcm("SET STR", &strategy_param, 1, ierflg);
            
            // Minimize
            fMinuit->mnexcm("MIGRAD", 0, 0, ierflg);
            
            // Get results
            G4double final_params[fNParams];
            G4double param_errors[fNParams];
            for (G4int i = 0; i < fNParams; ++i) {
                fMinuit->GetParameter(i, final_params[i], param_errors[i]);
            }
            
            // Get fit statistics
            G4double fmin, fedm, errdef;
            G4int npari, nparx, istat;
            fMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
            
            // Check constraints
            G4bool constraints_ok = CheckConstraints(final_params, verbose);
            
            if (verbose) {
                G4cout << "  Final chi2: " << fmin << G4endl;
                G4cout << "  Fit status: " << istat << " (0-3 acceptable for convergence)" << G4endl;
                G4cout << "  Constraints satisfied: " << (constraints_ok ? "Yes" : "No") << G4endl;
            }
            
            // Accept fit if constraints are satisfied and fit converged
            if (constraints_ok && (istat >= 0 && istat <= 3) && fmin < best_chi2 && fmin > 0) {
                best_chi2 = fmin;
                for (G4int i = 0; i < fNParams; ++i) {
                    best_params[i] = final_params[i];
                    best_errors[i] = param_errors[i];
                }
                results.constraints_satisfied = true;
                fit_found = true;
                
                if (verbose) {
                    G4cout << "  ✓ Fit accepted!" << G4endl;
                }
            } else if (verbose) {
                G4cout << "  ✗ Fit rejected (constraints: " << constraints_ok 
                       << ", status: " << istat << ", chi2: " << fmin << ")" << G4endl;
            }
            
        } catch (const std::exception& e) {
            if (verbose) {
                G4cout << "  Exception during fitting attempt " << (attempt + 1) << ": " << e.what() << G4endl;
            }
        } catch (...) {
            if (verbose) {
                G4cout << "  Unknown exception during fitting attempt " << (attempt + 1) << G4endl;
            }
        }
    }
    
    // Clear static data after fitting to prevent issues
    fStaticXCoords.clear();
    fStaticYCoords.clear();
    fStaticZValues.clear();
    fStaticZErrors.clear();
    fStaticInstance = nullptr;
    
    if (!fit_found) {
        if (verbose) {
            G4cout << "\nAll fitting attempts failed!" << G4endl;
        }
        return results;
    }
    
    // Store final results
    results.amplitude = best_params[0];
    results.x0 = best_params[1];
    results.y0 = best_params[2];
    results.sigma_x = best_params[3];
    results.sigma_y = best_params[4];
    results.theta = best_params[5];
    results.offset = best_params[6];
    
    // Store parameter errors
    results.amplitude_err = best_errors[0];
    results.x0_err = best_errors[1];
    results.y0_err = best_errors[2];
    results.sigma_x_err = best_errors[3];
    results.sigma_y_err = best_errors[4];
    results.theta_err = best_errors[5];
    results.offset_err = best_errors[6];
    
    // Calculate statistics
    results.chi2red = best_chi2;
    results.ndf = results.n_points - fNParams;
    results.Pp = (results.ndf > 0) ? TMath::Prob(best_chi2, results.ndf) : 0.0;
    
    // Calculate residual statistics
    CalculateResidualStats(x_coords, y_coords, z_values, best_params, 
                         results.residual_mean, results.residual_std);
    
    if (verbose) {
        G4cout << "\n=== Final Minuit Fit Results ===" << G4endl;
        G4cout << "Parameters:" << G4endl;
        G4cout << "  Amplitude: " << results.amplitude << " ± " << results.amplitude_err << G4endl;
        G4cout << "  Center: (" << results.x0 << " ± " << results.x0_err << ", " 
               << results.y0 << " ± " << results.y0_err << ") mm" << G4endl;
        G4cout << "  Sigma: (" << results.sigma_x << " ± " << results.sigma_x_err << ", " 
               << results.sigma_y << " ± " << results.sigma_y_err << ") mm" << G4endl;
        G4cout << "  Theta: " << results.theta << " ± " << results.theta_err << " rad ("
               << results.theta * 180.0 / TMath::Pi() << "° ± " 
               << results.theta_err * 180.0 / TMath::Pi() << "°)" << G4endl;
        G4cout << "  Offset: " << results.offset << " ± " << results.offset_err << G4endl;
        G4cout << "Statistics:" << G4endl;
        G4cout << "  Chi2/NDF: " << results.chi2red << "/" << results.ndf << G4endl;
        G4cout << "  Probability: " << results.Pp << G4endl;
        G4cout << "  Data points: " << results.n_points << G4endl;
        G4cout << "===============================" << G4endl;
    }
    
    return results;
} 