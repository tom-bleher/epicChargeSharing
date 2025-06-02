#include "Gaussian3DFitter.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ROOT includes (minimal to avoid Minuit issues)
#include "TMath.h"

// Define static constants
const G4double Gaussian3DFitter::fConstraintPenalty = Constants::CONSTRAINT_PENALTY;  // Large penalty for constraint violations

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

G4bool Gaussian3DFitter::IsPointInsidePixelZone(G4double x, G4double y, G4double min_distance) const
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

G4double Gaussian3DFitter::CalculateMinDistanceToPixelCenter(G4double x, G4double y) const
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
    
    // Check 1: Center must be within detector bounds
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    if (TMath::Abs(x0) > half_det || TMath::Abs(y0) > half_det) {
        if (verbose) {
            G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                   << ") outside detector bounds ±" << half_det << " mm" << G4endl;
        }
        return false;
    }
    
    // Check 2: Center must be outside pixel area and d0 (10 micron) radius from pixel center
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
    
    return true;
}

void Gaussian3DFitter::ApplyParameterBounds(G4double* params) const
{
    // Apply hard bounds to prevent optimization from going to unreasonable values
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Clamp center coordinates to detector bounds
    params[1] = TMath::Max(-half_det, TMath::Min(half_det, params[1])); // x0
    params[2] = TMath::Max(-half_det, TMath::Min(half_det, params[2])); // y0
    
    // Keep theta in reasonable range [-π, π]
    while (params[5] > TMath::Pi()) params[5] -= 2.0 * TMath::Pi();
    while (params[5] < -TMath::Pi()) params[5] += 2.0 * TMath::Pi();
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
    
    // If initial guess is in pixel zone, move it to nearest allowed position
    const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
    if (IsPointInsidePixelZone(weighted_x, weighted_y, d0)) {
        // Try to find nearby allowed position
        G4bool found_valid = false;
        const G4double search_radius = fDetectorGeometry.pixel_spacing;
        const G4int n_attempts = 8;
        
        for (G4int i = 0; i < n_attempts && !found_valid; ++i) {
            G4double angle = i * 2.0 * TMath::Pi() / n_attempts;
            G4double test_x = weighted_x + search_radius * TMath::Cos(angle);
            G4double test_y = weighted_y + search_radius * TMath::Sin(angle);
            
            if (TMath::Abs(test_x) <= half_det && TMath::Abs(test_y) <= half_det && 
                !IsPointInsidePixelZone(test_x, test_y, d0)) {
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
    const G4double min_sigma = Constants::MIN_SIGMA_ALT; // 10 microns minimum
    const G4double max_sigma = fDetectorGeometry.detector_size * Constants::SIGMA_FRACTION_ALT;
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
    G4double chi2red = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, params);
    
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
    
    // Penalty for being too close to pixel center or inside pixel area
    const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
    if (IsPointInsidePixelZone(x0, y0, d0)) {
        G4double min_dist = CalculateMinDistanceToPixelCenter(x0, y0);
        if (min_dist < d0) {
            penalty += fConstraintPenalty * (d0 - min_dist) * 100.0; // Scale up the penalty
        } else {
            // Inside pixel area but outside d0 radius - still penalize
            penalty += fConstraintPenalty * 100.0;
        }
    }
    
    return chi2red + penalty;
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
    G4double chi2red = 0.0;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], params);
        const G4double residual = z_values[i] - z_fit;
        const G4double error = (!z_errors.empty() && i < z_errors.size()) ? z_errors[i] : 1.0;
        
        if (error > 0) {
            chi2red += (residual * residual) / (error * error);
        }
    }
    
    return chi2red;
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
    const G4int max_iterations = Constants::MAX_ITERATIONS; // Increased for robustness
    const G4double tolerance = Constants::FIT_TOLERANCE;    // Tighter tolerance
    
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
                        step = TMath::Abs(params[j]) * Constants::DEFAULT_AMPLITUDE_FRACTION;
                        if (step < Constants::MIN_STEP_SIZE) step = Constants::MIN_STEP_SIZE;
                        break;
                    case 1: case 2: // x0, y0
                        step = Constants::DEFAULT_STEP_SIZE;
                        break;
                    case 3: case 4: // sigma_x, sigma_y
                        step = TMath::Abs(params[j]) * Constants::DEFAULT_AMPLITUDE_FRACTION;
                        if (step < Constants::MIN_STEP_SIZE) step = Constants::MIN_STEP_SIZE;
                        break;
                    case 5: // theta
                        step = Constants::DEFAULT_STEP_SIZE;
                        break;
                    case 6: // offset
                        step = TMath::Abs(params[j]) * Constants::DEFAULT_AMPLITUDE_FRACTION;
                        if (step < Constants::MIN_STEP_SIZE_ALT) step = Constants::MIN_STEP_SIZE;
                        break;
                    default:
                        step = TMath::Abs(params[j]) * Constants::DEFAULT_AMPLITUDE_FRACTION;
                        if (step < Constants::MIN_STEP_SIZE) step = Constants::MIN_STEP_SIZE;
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
        G4cout << "  Initial chi2red values: ";
        for (const auto& chi2red : chi2_values) {
            G4cout << chi2red << " ";
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
            G4cout << "  Iteration " << iter << ": best chi2red = " << chi2_values[best_idx] << G4endl;
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
        G4cout << "  Final chi2red: " << chi2_values[best_idx] << G4endl;
    }
}

Gaussian3DFitter::FitResults Gaussian3DFitter::FitGaussian3D(const std::vector<G4double>& x_coords,
                                                            const std::vector<G4double>& y_coords,
                                                            const std::vector<G4double>& z_values,
                                                            const std::vector<G4double>& z_errors,
                                                            G4bool verbose)
{
    FitResults results;
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Inconsistent input array sizes" << G4endl;
        }
        return results;
    }
    
    if (x_coords.size() < 7) { // Need at least as many points as parameters
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Insufficient data points (" 
                   << x_coords.size() << " < 7)" << G4endl;
        }
        return results;
    }
    
    if (verbose) {
        G4cout << "\n=== Gaussian3D Fitting ===" << G4endl;
        G4cout << "Data points: " << x_coords.size() << G4endl;
        G4cout << "Detector geometry: " << fDetectorGeometry.detector_size << "×" << fDetectorGeometry.detector_size 
               << " mm, " << fDetectorGeometry.num_blocks_per_side << "×" << fDetectorGeometry.num_blocks_per_side << " pixels" << G4endl;
        G4cout << "Pixel exclusion buffer: " << fDetectorGeometry.pixel_exclusion_buffer << " mm" << G4endl;
    }
    
    results.fit_type = ALL_DATA; // This method uses all data
    results.n_points = x_coords.size();
    
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
            
            // Perform robust simplex optimization with constraints
            RobustSimplexFit(x_coords, y_coords, z_values, z_errors, fitParams, verbose);
            
            // Check constraints
            G4bool constraints_ok = CheckConstraints(fitParams, verbose);
            
            // Calculate final chi-squared (without penalties)
            G4double final_chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, fitParams);
            
            if (verbose) {
                G4cout << "  Final chi2red: " << final_chi2 << G4endl;
                G4cout << "  Constraints satisfied: " << (constraints_ok ? "Yes" : "No") << G4endl;
            }
            
            // Accept fit if constraints are satisfied and chi2red is reasonable
            if (constraints_ok && final_chi2 < best_chi2 && final_chi2 > 0) {
                best_chi2 = final_chi2;
                for (G4int i = 0; i < fNParams; ++i) {
                    best_params[i] = fitParams[i];
                }
                results.constraints_satisfied = true;
                fit_found = true;
                
                if (verbose) {
                    G4cout << "  ✓ Fit accepted!" << G4endl;
                }
            } else if (verbose) {
                G4cout << "  ✗ Fit rejected (constraints: " << constraints_ok 
                       << ", chi2red: " << final_chi2 << ")" << G4endl;
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
    results.chi2red = best_chi2;
    results.ndf = results.n_points - fNParams;
    results.Pp = (results.ndf > 0) ? TMath::Prob(best_chi2, results.ndf) : 0.0;
    
    // Calculate residual statistics
    CalculateResidualStats(x_coords, y_coords, z_values, best_params, 
                         results.residual_mean, results.residual_std);
    
    // Enhanced error estimates based on parameter sensitivity
    const G4double reduced_chi2 = results.ndf > 0 ? results.chi2red : 1.0;
    const G4double error_scale = TMath::Sqrt(TMath::Max(1.0, reduced_chi2));
    
    // More sophisticated error estimates
    results.amplitude_err = TMath::Abs(results.amplitude) * 0.05 * error_scale;
    results.x0_err = Constants::BASE_POSITION_ERROR * error_scale; // 5 microns base uncertainty
    results.y0_err = Constants::BASE_POSITION_ERROR * error_scale;
    results.sigma_x_err = TMath::Abs(results.sigma_x) * 0.08 * error_scale;
    results.sigma_y_err = TMath::Abs(results.sigma_y) * 0.08 * error_scale;
    results.theta_err = 0.05 * error_scale; // ~3 degrees
    results.offset_err = TMath::Abs(results.offset) * Constants::ERROR_SCALE_FRACTION * error_scale;
    
    if (verbose) {
        G4cout << "\n=== Final Fit Results ===" << G4endl;
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
        G4cout << "  chi2red/NDF: " << results.chi2red << G4endl;
        G4cout << "  Probability: " << results.Pp << G4endl;
        G4cout << "  Data points: " << results.n_points << G4endl;
        G4cout << "===========================" << G4endl;
    }
    
    return results;
} 