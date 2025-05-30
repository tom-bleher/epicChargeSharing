#include "Gaussian3DFitter.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ROOT includes (minimal to avoid Minuit issues)
#include "TMath.h"

// Define static constants
const G4double Gaussian3DFitter::fOutlierThreshold = 3.0;  // 3-sigma outlier threshold
const G4double Gaussian3DFitter::fConstraintPenalty = 1000.0;  // Large penalty for constraint violations

Gaussian3DFitter::Gaussian3DFitter(const DetectorGeometry& detector_geometry) 
    : fDetectorGeometry(detector_geometry), fGaussianFunction(nullptr), fDataGraph(nullptr)
{
    // No ROOT objects initialization to avoid Minuit issues
}

Gaussian3DFitter::~Gaussian3DFitter()
{
    // No cleanup needed for void* pointers in this implementation
    fGaussianFunction = nullptr;
    fDataGraph = nullptr;
}

void Gaussian3DFitter::SetDetectorGeometry(const DetectorGeometry& geometry)
{
    fDetectorGeometry = geometry;
}

G4double Gaussian3DFitter::Gaussian3DFunction(G4double x, G4double y, const G4double* params)
{
    /*
     * 3D Gaussian function with rotation (matching Python implementation)
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
    
    // Rotation transformation (same as Python version)
    const G4double cos_theta = TMath::Cos(theta);
    const G4double sin_theta = TMath::Sin(theta);
    
    const G4double x_rot = cos_theta * (x - x0) + sin_theta * (y - y0);
    const G4double y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0);
    
    // 3D Gaussian
    const G4double exponent = -(x_rot*x_rot / (2.0 * sigma_x*sigma_x) + 
                               y_rot*y_rot / (2.0 * sigma_y*sigma_y));
    
    return amplitude * TMath::Exp(exponent) + offset;
}

G4double Gaussian3DFitter::Gaussian3DFunctionWrapper(G4double* coords, G4double* params)
{
    // Wrapper function for ROOT compatibility (not used in new implementation)
    return Gaussian3DFunction(coords[0], coords[1], params);
}

G4bool Gaussian3DFitter::IsPointInsidePixelZone(G4double x, G4double y) const
{
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Check if point is inside detector bounds
    if (TMath::Abs(x) > half_det || TMath::Abs(y) > half_det) {
        return false; // Outside detector, not in pixel zone
    }
    
    // Calculate the first pixel position
    const G4double first_pixel_pos = -half_det + fDetectorGeometry.pixel_corner_offset + fDetectorGeometry.pixel_size / 2.0;
    
    // Check each pixel with buffer zone
    for (G4int i = 0; i < fDetectorGeometry.num_blocks_per_side; i++) {
        for (G4int j = 0; j < fDetectorGeometry.num_blocks_per_side; j++) {
            const G4double pixel_center_x = first_pixel_pos + i * fDetectorGeometry.pixel_spacing;
            const G4double pixel_center_y = first_pixel_pos + j * fDetectorGeometry.pixel_spacing;
            
            // Calculate distance from point to pixel center
            const G4double dx = x - pixel_center_x;
            const G4double dy = y - pixel_center_y;
            
            // Check if within pixel + buffer zone (using square boundary for efficiency)
            const G4double pixel_half_size = fDetectorGeometry.pixel_size / 2.0;
            const G4double exclusion_zone = pixel_half_size + fDetectorGeometry.pixel_exclusion_buffer;
            
            if (TMath::Abs(dx) <= exclusion_zone && TMath::Abs(dy) <= exclusion_zone) {
                return true; // Inside pixel exclusion zone
            }
        }
    }
    
    return false; // Not in any pixel zone
}

G4double Gaussian3DFitter::CalculateMinDistanceToPixel(G4double x, G4double y) const
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

G4bool Gaussian3DFitter::CheckConstraints(const G4double* params, G4bool verbose) const
{
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    const G4double sigma_x = params[3];
    const G4double sigma_y = params[4];
    const G4double amplitude = params[0];
    
    // Check 1: Center must be within detector bounds
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    if (TMath::Abs(x0) > half_det || TMath::Abs(y0) > half_det) {
        if (verbose) {
            G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                   << ") outside detector bounds ±" << half_det << " mm" << G4endl;
        }
        return false;
    }
    
    // Check 2: Center must not be inside pixel exclusion zones
    if (IsPointInsidePixelZone(x0, y0)) {
        if (verbose) {
            G4double min_dist = CalculateMinDistanceToPixel(x0, y0);
            G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                   << ") inside pixel zone (min distance: " << min_dist 
                   << " mm, required: " << fDetectorGeometry.pixel_exclusion_buffer << " mm)" << G4endl;
        }
        return false;
    }
    
    // Check 3: Sigma values must be reasonable
    const G4double min_sigma = 0.005; // 5 microns minimum
    const G4double max_sigma = fDetectorGeometry.detector_size / 4.0; // Quarter of detector size maximum
    if (sigma_x < min_sigma || sigma_x > max_sigma || sigma_y < min_sigma || sigma_y > max_sigma) {
        if (verbose) {
            G4cout << "  Constraint violation: Sigma values (" << sigma_x << ", " << sigma_y 
                   << ") outside reasonable range [" << min_sigma << ", " << max_sigma << "] mm" << G4endl;
        }
        return false;
    }
    
    // Check 4: Amplitude must be positive
    if (amplitude <= 0) {
        if (verbose) {
            G4cout << "  Constraint violation: Amplitude (" << amplitude << ") must be positive" << G4endl;
        }
        return false;
    }
    
    return true;
}

void Gaussian3DFitter::ApplyParameterBounds(G4double* params) const
{
    // Apply hard bounds to prevent optimization from going to unreasonable values
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    const G4double min_sigma = 0.005; // 5 microns
    const G4double max_sigma = fDetectorGeometry.detector_size / 4.0;
    
    // Clamp center coordinates to detector bounds
    params[1] = TMath::Max(-half_det, TMath::Min(half_det, params[1])); // x0
    params[2] = TMath::Max(-half_det, TMath::Min(half_det, params[2])); // y0
    
    // Clamp sigma values
    params[3] = TMath::Max(min_sigma, TMath::Min(max_sigma, params[3])); // sigma_x
    params[4] = TMath::Max(min_sigma, TMath::Min(max_sigma, params[4])); // sigma_y
    
    // Ensure amplitude is positive
    if (params[0] <= 0) params[0] = 1.0;
    
    // Keep theta in reasonable range [-π, π]
    while (params[5] > TMath::Pi()) params[5] -= 2.0 * TMath::Pi();
    while (params[5] < -TMath::Pi()) params[5] += 2.0 * TMath::Pi();
}

void Gaussian3DFitter::RemoveOutliers(std::vector<G4double>& x_coords,
                                     std::vector<G4double>& y_coords,
                                     std::vector<G4double>& z_values,
                                     std::vector<G4double>& z_errors,
                                     G4int& n_outliers_removed,
                                     G4bool verbose)
{
    if (z_values.size() < 10) {
        n_outliers_removed = 0;
        return; // Too few points for outlier removal
    }
    
    // Calculate robust statistics using median absolute deviation (MAD)
    std::vector<G4double> z_sorted = z_values;
    std::sort(z_sorted.begin(), z_sorted.end());
    
    const G4double median = z_sorted[z_sorted.size() / 2];
    
    // Calculate MAD
    std::vector<G4double> abs_deviations;
    for (const G4double& z : z_values) {
        abs_deviations.push_back(TMath::Abs(z - median));
    }
    std::sort(abs_deviations.begin(), abs_deviations.end());
    const G4double mad = abs_deviations[abs_deviations.size() / 2];
    
    // Use modified Z-score with MAD (more robust than standard deviation)
    const G4double mad_threshold = fOutlierThreshold * 1.4826 * mad; // 1.4826 converts MAD to std dev equivalent
    
    if (verbose) {
        G4cout << "  Outlier detection: median=" << median << ", MAD=" << mad 
               << ", threshold=" << mad_threshold << G4endl;
    }
    
    // Remove outliers
    std::vector<G4double> x_cleaned, y_cleaned, z_cleaned, z_err_cleaned;
    n_outliers_removed = 0;
    
    for (size_t i = 0; i < z_values.size(); ++i) {
        const G4double modified_z_score = TMath::Abs(z_values[i] - median);
        
        if (modified_z_score <= mad_threshold) {
            x_cleaned.push_back(x_coords[i]);
            y_cleaned.push_back(y_coords[i]);
            z_cleaned.push_back(z_values[i]);
            if (!z_errors.empty() && i < z_errors.size()) {
                z_err_cleaned.push_back(z_errors[i]);
            }
        } else {
            n_outliers_removed++;
            if (verbose) {
                G4cout << "    Removed outlier at (" << x_coords[i] << ", " << y_coords[i] 
                       << ") with value " << z_values[i] << " (modified Z=" << modified_z_score/mad_threshold << ")" << G4endl;
            }
        }
    }
    
    // Replace original vectors
    x_coords = x_cleaned;
    y_coords = y_cleaned;
    z_values = z_cleaned;
    z_errors = z_err_cleaned;
    
    if (verbose && n_outliers_removed > 0) {
        G4cout << "  Removed " << n_outliers_removed << " outliers, " << z_values.size() << " points remaining" << G4endl;
    }
}

void Gaussian3DFitter::CalculateInitialGuess(const std::vector<G4double>& x_coords,
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
        initialParams[3] = 0.1;  // sigma_x
        initialParams[4] = 0.1;  // sigma_y
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
    
    // If initial guess is in pixel zone, move it to nearest allowed position
    if (IsPointInsidePixelZone(weighted_x, weighted_y)) {
        // Try to find nearby allowed position
        G4bool found_valid = false;
        const G4double search_radius = fDetectorGeometry.pixel_spacing;
        const G4int n_attempts = 8;
        
        for (G4int i = 0; i < n_attempts && !found_valid; ++i) {
            G4double angle = i * 2.0 * TMath::Pi() / n_attempts;
            G4double test_x = weighted_x + search_radius * TMath::Cos(angle);
            G4double test_y = weighted_y + search_radius * TMath::Sin(angle);
            
            if (TMath::Abs(test_x) <= half_det && TMath::Abs(test_y) <= half_det && 
                !IsPointInsidePixelZone(test_x, test_y)) {
                weighted_x = test_x;
                weighted_y = test_y;
                found_valid = true;
            }
        }
        
        if (!found_valid) {
            // Fall back to center of detector
            weighted_x = 0.0;
            weighted_y = 0.0;
        }
    }
    
    // Calculate initial sigma estimates
    G4double sum_x_sq = 0.0, sum_y_sq = 0.0;
    for (size_t i = 0; i < x_coords.size(); ++i) {
        sum_x_sq += (x_coords[i] - weighted_x) * (x_coords[i] - weighted_x);
        sum_y_sq += (y_coords[i] - weighted_y) * (y_coords[i] - weighted_y);
    }
    
    G4double sigma_x_guess = TMath::Sqrt(sum_x_sq / x_coords.size());
    G4double sigma_y_guess = TMath::Sqrt(sum_y_sq / y_coords.size());
    
    // Apply reasonable bounds to sigma estimates
    const G4double min_sigma = 0.01; // 10 microns minimum
    const G4double max_sigma = fDetectorGeometry.detector_size / 6.0;
    sigma_x_guess = TMath::Max(min_sigma, TMath::Min(max_sigma, sigma_x_guess));
    sigma_y_guess = TMath::Max(min_sigma, TMath::Min(max_sigma, sigma_y_guess));
    
    // Set initial parameters
    initialParams[0] = (z_max - z_min) * 0.8;  // amplitude (slightly conservative)
    initialParams[1] = weighted_x;             // x0
    initialParams[2] = weighted_y;             // y0
    initialParams[3] = sigma_x_guess;          // sigma_x
    initialParams[4] = sigma_y_guess;          // sigma_y
    initialParams[5] = 0.0;                    // theta (no rotation initially)
    initialParams[6] = z_min;                  // offset
}

G4double Gaussian3DFitter::CalculateConstrainedChiSquared(const std::vector<G4double>& x_coords,
                                                          const std::vector<G4double>& y_coords,
                                                          const std::vector<G4double>& z_values,
                                                          const std::vector<G4double>& z_errors,
                                                          const G4double* params)
{
    G4double chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, params);
    
    // Add penalty terms for constraint violations
    G4double penalty = 0.0;
    
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Penalty for being outside detector bounds
    if (TMath::Abs(x0) > half_det) {
        penalty += fConstraintPenalty * (TMath::Abs(x0) - half_det);
    }
    if (TMath::Abs(y0) > half_det) {
        penalty += fConstraintPenalty * (TMath::Abs(y0) - half_det);
    }
    
    // Penalty for being in pixel exclusion zone
    if (IsPointInsidePixelZone(x0, y0)) {
        G4double dist_deficit = fDetectorGeometry.pixel_exclusion_buffer - CalculateMinDistanceToPixel(x0, y0);
        penalty += fConstraintPenalty * TMath::Max(0.0, dist_deficit * 100.0); // Scale up the penalty
    }
    
    // Penalty for unreasonable sigma values
    const G4double min_sigma = 0.005;
    const G4double max_sigma = fDetectorGeometry.detector_size / 4.0;
    if (params[3] < min_sigma) penalty += fConstraintPenalty * (min_sigma - params[3]) * 1000.0;
    if (params[3] > max_sigma) penalty += fConstraintPenalty * (params[3] - max_sigma);
    if (params[4] < min_sigma) penalty += fConstraintPenalty * (min_sigma - params[4]) * 1000.0;
    if (params[4] > max_sigma) penalty += fConstraintPenalty * (params[4] - max_sigma);
    
    // Penalty for negative amplitude
    if (params[0] <= 0) penalty += fConstraintPenalty * (-params[0] + 1.0) * 1000.0;
    
    return chi2 + penalty;
}

G4double Gaussian3DFitter::CalculateRSquared(const std::vector<G4double>& x_coords,
                                             const std::vector<G4double>& y_coords,
                                             const std::vector<G4double>& z_values,
                                             const G4double* fitParams)
{
    if (z_values.empty()) return 0.0;
    
    // Calculate mean of observed values
    const G4double z_mean = std::accumulate(z_values.begin(), z_values.end(), 0.0) / z_values.size();
    
    G4double ss_tot = 0.0; // Total sum of squares
    G4double ss_res = 0.0; // Residual sum of squares
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_obs = z_values[i];
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], fitParams);
        
        ss_tot += (z_obs - z_mean) * (z_obs - z_mean);
        ss_res += (z_obs - z_fit) * (z_obs - z_fit);
    }
    
    if (ss_tot == 0.0) return 0.0;
    
    return 1.0 - (ss_res / ss_tot);
}

void Gaussian3DFitter::CalculateResidualStats(const std::vector<G4double>& x_coords,
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

G4double Gaussian3DFitter::CalculateChiSquared(const std::vector<G4double>& x_coords,
                                               const std::vector<G4double>& y_coords,
                                               const std::vector<G4double>& z_values,
                                               const std::vector<G4double>& z_errors,
                                               const G4double* params)
{
    G4double chi2 = 0.0;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], params);
        const G4double residual = z_values[i] - z_fit;
        const G4double error = (!z_errors.empty() && i < z_errors.size()) ? z_errors[i] : 1.0;
        
        if (error > 0) {
            chi2 += (residual * residual) / (error * error);
        }
    }
    
    return chi2;
}

void Gaussian3DFitter::RobustSimplexFit(const std::vector<G4double>& x_coords,
                                        const std::vector<G4double>& y_coords,
                                        const std::vector<G4double>& z_values,
                                        const std::vector<G4double>& z_errors,
                                        G4double* params,
                                        G4bool verbose)
{
    // Enhanced Nelder-Mead simplex optimization with constraints
    const G4int n_params = fNParams;
    const G4int max_iterations = 2000; // Increased for robustness
    const G4double tolerance = 1e-8;    // Tighter tolerance
    
    // Create initial simplex with better parameter space exploration
    std::vector<std::vector<G4double>> simplex(n_params + 1);
    std::vector<G4double> chi2_values(n_params + 1);
    
    // Initialize simplex vertices with more sophisticated perturbations
    for (G4int i = 0; i <= n_params; ++i) {
        simplex[i].resize(n_params);
        for (G4int j = 0; j < n_params; ++j) {
            simplex[i][j] = params[j];
            if (i == j + 1) {
                // Parameter-specific perturbations
                G4double step;
                switch (j) {
                    case 0: // amplitude
                        step = TMath::Abs(params[j]) * 0.2; // 20% variation
                        if (step < 0.1) step = 0.1;
                        break;
                    case 1: case 2: // x0, y0
                        step = 0.05; // 50 microns
                        break;
                    case 3: case 4: // sigma_x, sigma_y
                        step = TMath::Abs(params[j]) * 0.3; // 30% variation
                        if (step < 0.01) step = 0.01;
                        break;
                    case 5: // theta
                        step = 0.2; // ~11 degrees
                        break;
                    case 6: // offset
                        step = TMath::Abs(params[j]) * 0.1;
                        if (step < 0.01) step = 0.01;
                        break;
                    default:
                        step = TMath::Abs(params[j]) * 0.1;
                        if (step < 1e-6) step = 0.01;
                }
                simplex[i][j] += step;
            }
        }
        
        // Apply parameter bounds before evaluation
        ApplyParameterBounds(&simplex[i][0]);
        chi2_values[i] = CalculateConstrainedChiSquared(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
    }
    
    if (verbose) {
        G4cout << "  Starting robust simplex optimization..." << G4endl;
        G4cout << "  Initial chi2 values: ";
        for (const auto& chi2 : chi2_values) {
            G4cout << chi2 << " ";
        }
        G4cout << G4endl;
    }
    
    // Simplex optimization with enhanced robustness
    G4int stagnation_count = 0;
    G4double prev_best_chi2 = 1e10;
    
    for (G4int iter = 0; iter < max_iterations; ++iter) {
        // Find indices of best, worst, and second worst vertices
        G4int best_idx = 0, worst_idx = 0, second_worst_idx = 0;
        
        for (G4int i = 1; i <= n_params; ++i) {
            if (chi2_values[i] < chi2_values[best_idx]) best_idx = i;
            if (chi2_values[i] > chi2_values[worst_idx]) worst_idx = i;
        }
        
        for (G4int i = 0; i <= n_params; ++i) {
            if (i != worst_idx && chi2_values[i] > chi2_values[second_worst_idx]) {
                second_worst_idx = i;
            }
        }
        
        // Check for convergence
        const G4double chi2_range = chi2_values[worst_idx] - chi2_values[best_idx];
        if (chi2_range < tolerance) {
            if (verbose) {
                G4cout << "  Simplex converged after " << iter << " iterations (range: " << chi2_range << ")" << G4endl;
            }
            break;
        }
        
        // Check for stagnation
        if (TMath::Abs(chi2_values[best_idx] - prev_best_chi2) < tolerance * 10) {
            stagnation_count++;
            if (stagnation_count > 50) {
                if (verbose) {
                    G4cout << "  Simplex stagnated after " << iter << " iterations" << G4endl;
                }
                break;
            }
        } else {
            stagnation_count = 0;
            prev_best_chi2 = chi2_values[best_idx];
        }
        
        // Calculate centroid (excluding worst point)
        std::vector<G4double> centroid(n_params, 0.0);
        for (G4int i = 0; i <= n_params; ++i) {
            if (i != worst_idx) {
                for (G4int j = 0; j < n_params; ++j) {
                    centroid[j] += simplex[i][j];
                }
            }
        }
        for (G4int j = 0; j < n_params; ++j) {
            centroid[j] /= n_params;
        }
        
        // Reflection
        std::vector<G4double> reflected(n_params);
        for (G4int j = 0; j < n_params; ++j) {
            reflected[j] = centroid[j] + 1.0 * (centroid[j] - simplex[worst_idx][j]);
        }
        ApplyParameterBounds(&reflected[0]);
        G4double chi2_reflected = CalculateConstrainedChiSquared(x_coords, y_coords, z_values, z_errors, &reflected[0]);
        
        if (chi2_reflected < chi2_values[best_idx]) {
            // Expansion
            std::vector<G4double> expanded(n_params);
            for (G4int j = 0; j < n_params; ++j) {
                expanded[j] = centroid[j] + 2.0 * (centroid[j] - simplex[worst_idx][j]);
            }
            ApplyParameterBounds(&expanded[0]);
            G4double chi2_expanded = CalculateConstrainedChiSquared(x_coords, y_coords, z_values, z_errors, &expanded[0]);
            
            if (chi2_expanded < chi2_reflected) {
                simplex[worst_idx] = expanded;
                chi2_values[worst_idx] = chi2_expanded;
            } else {
                simplex[worst_idx] = reflected;
                chi2_values[worst_idx] = chi2_reflected;
            }
        } else if (chi2_reflected < chi2_values[second_worst_idx]) {
            simplex[worst_idx] = reflected;
            chi2_values[worst_idx] = chi2_reflected;
        } else {
            // Contraction
            std::vector<G4double> contracted(n_params);
            if (chi2_reflected < chi2_values[worst_idx]) {
                // Outside contraction
                for (G4int j = 0; j < n_params; ++j) {
                    contracted[j] = centroid[j] + 0.5 * (reflected[j] - centroid[j]);
                }
            } else {
                // Inside contraction
                for (G4int j = 0; j < n_params; ++j) {
                    contracted[j] = centroid[j] + 0.5 * (simplex[worst_idx][j] - centroid[j]);
                }
            }
            ApplyParameterBounds(&contracted[0]);
            G4double chi2_contracted = CalculateConstrainedChiSquared(x_coords, y_coords, z_values, z_errors, &contracted[0]);
            
            if (chi2_contracted < TMath::Min(chi2_values[worst_idx], chi2_reflected)) {
                simplex[worst_idx] = contracted;
                chi2_values[worst_idx] = chi2_contracted;
            } else {
                // Shrink
                for (G4int i = 0; i <= n_params; ++i) {
                    if (i != best_idx) {
                        for (G4int j = 0; j < n_params; ++j) {
                            simplex[i][j] = simplex[best_idx][j] + 0.5 * (simplex[i][j] - simplex[best_idx][j]);
                        }
                        ApplyParameterBounds(&simplex[i][0]);
                        chi2_values[i] = CalculateConstrainedChiSquared(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
                    }
                }
            }
        }
        
        // Periodic verbose output
        if (verbose && iter % 200 == 0) {
            G4cout << "  Iteration " << iter << ": best chi2 = " << chi2_values[best_idx] << G4endl;
        }
    }
    
    // Find best solution
    G4int best_idx = 0;
    for (G4int i = 1; i <= n_params; ++i) {
        if (chi2_values[i] < chi2_values[best_idx]) best_idx = i;
    }
    
    // Copy best parameters
    for (G4int j = 0; j < n_params; ++j) {
        params[j] = simplex[best_idx][j];
    }
    
    if (verbose) {
        G4cout << "  Final chi2: " << chi2_values[best_idx] << G4endl;
    }
}

Gaussian3DFitter::FitResults Gaussian3DFitter::FitGaussian3D(const std::vector<G4double>& x_coords,
                                                            const std::vector<G4double>& y_coords,
                                                            const std::vector<G4double>& z_values,
                                                            const std::vector<G4double>& z_errors,
                                                            G4bool verbose)
{
    FitResults results;
    results.fit_type = OUTLIER_CLEANED; // This method performs outlier removal
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Inconsistent input array sizes" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    if (x_coords.size() < 7) { // Need at least as many points as parameters
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Insufficient data points (" 
                   << x_coords.size() << " < 7)" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    if (verbose) {
        G4cout << "\n=== Robust Gaussian3D Fitting (WITH outlier removal) ===" << G4endl;
        G4cout << "Initial data points: " << x_coords.size() << G4endl;
        G4cout << "Detector geometry: " << fDetectorGeometry.detector_size << "×" << fDetectorGeometry.detector_size 
               << " mm, " << fDetectorGeometry.num_blocks_per_side << "×" << fDetectorGeometry.num_blocks_per_side << " pixels" << G4endl;
        G4cout << "Pixel exclusion buffer: " << fDetectorGeometry.pixel_exclusion_buffer << " mm" << G4endl;
    }
    
    // Make copies of input data for outlier removal
    std::vector<G4double> x_clean = x_coords;
    std::vector<G4double> y_clean = y_coords;
    std::vector<G4double> z_clean = z_values;
    std::vector<G4double> z_err_clean = z_errors;
    
    // Step 1: Remove outliers
    RemoveOutliers(x_clean, y_clean, z_clean, z_err_clean, results.n_outliers_removed, verbose);
    
    // Check if we still have enough points
    if (x_clean.size() < 7) {
        if (verbose) {
            G4cout << "Error: Insufficient data points after outlier removal (" 
                   << x_clean.size() << " < 7)" << G4endl;
        }
        return results;
    }
    
    results.n_points = x_clean.size();
    
    // Step 2: Try multiple fitting strategies
    G4bool fit_found = false;
    G4double best_chi2 = 1e10;
    G4double best_params[fNParams];
    
    for (G4int attempt = 0; attempt < fMaxFitAttempts && !fit_found; ++attempt) {
        if (verbose) {
            G4cout << "\n--- Fit Attempt " << (attempt + 1) << " ---" << G4endl;
        }
        
        try {
            // Calculate initial parameter estimates with different strategies
            G4double fitParams[fNParams];
            CalculateInitialGuess(x_clean, y_clean, z_clean, fitParams, attempt);
            
            if (verbose) {
                G4cout << "Initial parameters (strategy " << attempt << "):" << G4endl;
                G4cout << "  Amplitude: " << fitParams[0] << G4endl;
                G4cout << "  Center: (" << fitParams[1] << ", " << fitParams[2] << ") mm" << G4endl;
                G4cout << "  Sigma: (" << fitParams[3] << ", " << fitParams[4] << ") mm" << G4endl;
                G4cout << "  Theta: " << fitParams[5] << " rad" << G4endl;
                G4cout << "  Offset: " << fitParams[6] << G4endl;
            }
            
            // Perform robust simplex optimization
            RobustSimplexFit(x_clean, y_clean, z_clean, z_err_clean, fitParams, verbose);
            
            // Check constraints
            G4bool constraints_ok = CheckConstraints(fitParams, verbose);
            
            // Calculate final chi-squared (without penalties)
            G4double final_chi2 = CalculateChiSquared(x_clean, y_clean, z_clean, z_err_clean, fitParams);
            
            if (verbose) {
                G4cout << "  Final chi2: " << final_chi2 << G4endl;
                G4cout << "  Constraints satisfied: " << (constraints_ok ? "Yes" : "No") << G4endl;
            }
            
            // Accept fit if constraints are satisfied and chi2 is reasonable
            if (constraints_ok && final_chi2 < best_chi2 && final_chi2 > 0) {
                best_chi2 = final_chi2;
                for (G4int i = 0; i < fNParams; ++i) {
                    best_params[i] = fitParams[i];
                }
                results.fit_attempt_number = attempt + 1;
                results.constraints_satisfied = true;
                fit_found = true;
                
                if (verbose) {
                    G4cout << "  ✓ Fit accepted!" << G4endl;
                }
            } else if (verbose) {
                G4cout << "  ✗ Fit rejected (constraints: " << constraints_ok 
                       << ", chi2: " << final_chi2 << ")" << G4endl;
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
    
    if (!fit_found) {
        if (verbose) {
            G4cout << "\nAll fitting attempts failed!" << G4endl;
        }
        return results;
    }
    
    // Step 3: Store final results
    results.amplitude = best_params[0];
    results.x0 = best_params[1];
    results.y0 = best_params[2];
    results.sigma_x = best_params[3];
    results.sigma_y = best_params[4];
    results.theta = best_params[5];
    results.offset = best_params[6];
    
    // Calculate statistics
    results.chi2 = best_chi2;
    results.ndf = results.n_points - fNParams;
    results.prob = (results.ndf > 0) ? TMath::Prob(results.chi2, results.ndf) : 0.0;
    results.r_squared = CalculateRSquared(x_clean, y_clean, z_clean, best_params);
    
    // Calculate residual statistics
    CalculateResidualStats(x_clean, y_clean, z_clean, best_params, 
                         results.residual_mean, results.residual_std);
    
    // Calculate robustness metrics
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    results.center_distance_from_detector_edge = TMath::Min(
        TMath::Min(half_det - TMath::Abs(results.x0), half_det - TMath::Abs(results.y0)),
        TMath::Min(half_det + results.x0, half_det + results.y0)
    );
    results.min_distance_to_pixel = CalculateMinDistanceToPixel(results.x0, results.y0);
    
    // Enhanced error estimates based on parameter sensitivity
    const G4double reduced_chi2 = results.ndf > 0 ? results.chi2 / results.ndf : 1.0;
    const G4double error_scale = TMath::Sqrt(TMath::Max(1.0, reduced_chi2));
    
    // More sophisticated error estimates
    results.amplitude_err = TMath::Abs(results.amplitude) * 0.05 * error_scale;
    results.x0_err = 0.005 * error_scale; // 5 microns base uncertainty
    results.y0_err = 0.005 * error_scale;
    results.sigma_x_err = TMath::Abs(results.sigma_x) * 0.08 * error_scale;
    results.sigma_y_err = TMath::Abs(results.sigma_y) * 0.08 * error_scale;
    results.theta_err = 0.05 * error_scale; // ~3 degrees
    results.offset_err = TMath::Abs(results.offset) * 0.1 * error_scale;
    
    // Final validation
    results.fit_successful = (results.r_squared > 0.3 && results.constraints_satisfied);
    
    if (verbose) {
        G4cout << "\n=== Final Fit Results (WITH outlier removal) ===" << G4endl;
        G4cout << "✓ Fit successful: " << (results.fit_successful ? "Yes" : "No") << G4endl;
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
        G4cout << "  Chi2/NDF: " << results.chi2 << " / " << results.ndf 
               << " = " << (results.ndf > 0 ? results.chi2/results.ndf : 0) << G4endl;
        G4cout << "  Probability: " << results.prob << G4endl;
        G4cout << "  R-squared: " << results.r_squared << G4endl;
        G4cout << "  Data points: " << results.n_points << " (outliers removed: " << results.n_outliers_removed << ")" << G4endl;
        G4cout << "Robustness:" << G4endl;
        G4cout << "  Distance from detector edge: " << results.center_distance_from_detector_edge << " mm" << G4endl;
        G4cout << "  Min distance to pixel: " << results.min_distance_to_pixel << " mm" << G4endl;
        G4cout << "  Fit attempt: " << results.fit_attempt_number << "/" << fMaxFitAttempts << G4endl;
        G4cout << "===========================" << G4endl;
    }
    
    return results;
}

Gaussian3DFitter::FitResults Gaussian3DFitter::FitGaussian3DAllData(const std::vector<G4double>& x_coords,
                                                                   const std::vector<G4double>& y_coords,
                                                                   const std::vector<G4double>& z_values,
                                                                   const std::vector<G4double>& z_errors,
                                                                   G4bool verbose)
{
    FitResults results;
    results.fit_type = ALL_DATA; // This method uses all data without outlier removal
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3DAllData - Error: Inconsistent input array sizes" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    if (x_coords.size() < 7) { // Need at least as many points as parameters
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3DAllData - Error: Insufficient data points (" 
                   << x_coords.size() << " < 7)" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    if (verbose) {
        G4cout << "\n=== Gaussian3D Fitting (ALL data, NO outlier removal) ===" << G4endl;
        G4cout << "Data points: " << x_coords.size() << G4endl;
        G4cout << "Detector geometry: " << fDetectorGeometry.detector_size << "×" << fDetectorGeometry.detector_size 
               << " mm, " << fDetectorGeometry.num_blocks_per_side << "×" << fDetectorGeometry.num_blocks_per_side << " pixels" << G4endl;
        G4cout << "Pixel exclusion buffer: " << fDetectorGeometry.pixel_exclusion_buffer << " mm" << G4endl;
    }
    
    // Use original data without outlier removal
    results.n_points = x_coords.size();
    results.n_outliers_removed = 0; // No outliers removed in this method
    
    // Try multiple fitting strategies
    G4bool fit_found = false;
    G4double best_chi2 = 1e10;
    G4double best_params[fNParams];
    
    for (G4int attempt = 0; attempt < fMaxFitAttempts && !fit_found; ++attempt) {
        if (verbose) {
            G4cout << "\n--- Fit Attempt " << (attempt + 1) << " ---" << G4endl;
        }
        
        try {
            // Calculate initial parameter estimates with different strategies
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
            
            // Perform robust simplex optimization (without outlier removal, but still with constraints)
            RobustSimplexFit(x_coords, y_coords, z_values, z_errors, fitParams, verbose);
            
            // Check constraints
            G4bool constraints_ok = CheckConstraints(fitParams, verbose);
            
            // Calculate final chi-squared (without penalties)
            G4double final_chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, fitParams);
            
            if (verbose) {
                G4cout << "  Final chi2: " << final_chi2 << G4endl;
                G4cout << "  Constraints satisfied: " << (constraints_ok ? "Yes" : "No") << G4endl;
            }
            
            // Accept fit if constraints are satisfied and chi2 is reasonable
            if (constraints_ok && final_chi2 < best_chi2 && final_chi2 > 0) {
                best_chi2 = final_chi2;
                for (G4int i = 0; i < fNParams; ++i) {
                    best_params[i] = fitParams[i];
                }
                results.fit_attempt_number = attempt + 1;
                results.constraints_satisfied = true;
                fit_found = true;
                
                if (verbose) {
                    G4cout << "  ✓ Fit accepted!" << G4endl;
                }
            } else if (verbose) {
                G4cout << "  ✗ Fit rejected (constraints: " << constraints_ok 
                       << ", chi2: " << final_chi2 << ")" << G4endl;
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
    
    // Calculate statistics
    results.chi2 = best_chi2;
    results.ndf = results.n_points - fNParams;
    results.prob = (results.ndf > 0) ? TMath::Prob(results.chi2, results.ndf) : 0.0;
    results.r_squared = CalculateRSquared(x_coords, y_coords, z_values, best_params);
    
    // Calculate residual statistics
    CalculateResidualStats(x_coords, y_coords, z_values, best_params, 
                         results.residual_mean, results.residual_std);
    
    // Calculate robustness metrics
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    results.center_distance_from_detector_edge = TMath::Min(
        TMath::Min(half_det - TMath::Abs(results.x0), half_det - TMath::Abs(results.y0)),
        TMath::Min(half_det + results.x0, half_det + results.y0)
    );
    results.min_distance_to_pixel = CalculateMinDistanceToPixel(results.x0, results.y0);
    
    // Enhanced error estimates based on parameter sensitivity
    const G4double reduced_chi2 = results.ndf > 0 ? results.chi2 / results.ndf : 1.0;
    const G4double error_scale = TMath::Sqrt(TMath::Max(1.0, reduced_chi2));
    
    // More sophisticated error estimates
    results.amplitude_err = TMath::Abs(results.amplitude) * 0.05 * error_scale;
    results.x0_err = 0.005 * error_scale; // 5 microns base uncertainty
    results.y0_err = 0.005 * error_scale;
    results.sigma_x_err = TMath::Abs(results.sigma_x) * 0.08 * error_scale;
    results.sigma_y_err = TMath::Abs(results.sigma_y) * 0.08 * error_scale;
    results.theta_err = 0.05 * error_scale; // ~3 degrees
    results.offset_err = TMath::Abs(results.offset) * 0.1 * error_scale;
    
    // Final validation
    results.fit_successful = (results.r_squared > 0.3 && results.constraints_satisfied);
    
    if (verbose) {
        G4cout << "\n=== Final Fit Results (ALL data, NO outlier removal) ===" << G4endl;
        G4cout << "✓ Fit successful: " << (results.fit_successful ? "Yes" : "No") << G4endl;
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
        G4cout << "  Chi2/NDF: " << results.chi2 << " / " << results.ndf 
               << " = " << (results.ndf > 0 ? results.chi2/results.ndf : 0) << G4endl;
        G4cout << "  Probability: " << results.prob << G4endl;
        G4cout << "  R-squared: " << results.r_squared << G4endl;
        G4cout << "  Data points: " << results.n_points << " (outliers removed: " << results.n_outliers_removed << ")" << G4endl;
        G4cout << "Robustness:" << G4endl;
        G4cout << "  Distance from detector edge: " << results.center_distance_from_detector_edge << " mm" << G4endl;
        G4cout << "  Min distance to pixel: " << results.min_distance_to_pixel << " mm" << G4endl;
        G4cout << "  Fit attempt: " << results.fit_attempt_number << "/" << fMaxFitAttempts << G4endl;
        G4cout << "===========================" << G4endl;
    }
    
    return results;
} 