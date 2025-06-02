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
    : fDetectorGeometry(detector_geometry), fGaussianFunction(nullptr), fDataGraph(nullptr),
      fCenterPixelX(0.0), fCenterPixelY(0.0), fConstrainToCenterPixel(false),
      fNeighborhoodRadius(Constants::NEIGHBORHOOD_RADIUS), fUseRedAreaConstraints(false)
{
    // Initialize with detector geometry constraints
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
    
    // Check 2: Different constraint modes
    if (fUseRedAreaConstraints && fConstrainToCenterPixel) {
        // New red area constraints: center must be in the "red area"
        if (!IsPointInRedArea(x0, y0)) {
            if (verbose) {
                G4cout << "  Constraint violation: Center (" << x0 << ", " << y0 
                       << ") outside red area (neighborhood: " << fNeighborhoodRadius 
                       << " pixels around (" << fCenterPixelX << ", " << fCenterPixelY << "))" << G4endl;
            }
            return false;
        }
    } else if (fConstrainToCenterPixel) {
        // Original center pixel constraint: center must be within center pixel bounds
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
        // Original constraint: Center must be outside pixel area and d0 (10 micron) radius from pixel center
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

void Gaussian3DFitter::CalculatePhysicalInitialGuess(const std::vector<G4double>& x_coords,
                                                     const std::vector<G4double>& y_coords,
                                                     const std::vector<G4double>& z_values,
                                                     G4double* initialParams,
                                                     G4int strategy)
{
    if (x_coords.empty() || y_coords.empty() || z_values.empty()) {
        // Set physically reasonable default values
        initialParams[0] = 1.0;      // amplitude
        initialParams[1] = 0.0;      // x0
        initialParams[2] = 0.0;      // y0
        initialParams[3] = 0.2*mm;   // sigma_x - 200 microns default
        initialParams[4] = 0.2*mm;   // sigma_y - 200 microns default
        initialParams[5] = 0.0;      // theta
        initialParams[6] = 0.0;      // offset
        return;
    }
    
    // Calculate basic statistics
    const G4double z_min = *std::min_element(z_values.begin(), z_values.end());
    const G4double z_max = *std::max_element(z_values.begin(), z_values.end());
    
    G4double weighted_x = 0.0, weighted_y = 0.0, sum_z = 0.0;
    
    // Enhanced strategies for initial guess with better center estimation
    switch (strategy) {
        case 0: // Weighted center of mass with threshold
            {
                const G4double threshold = z_min + 0.3 * (z_max - z_min);
                for (size_t i = 0; i < x_coords.size(); ++i) {
                    if (z_values[i] > threshold) {
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
            }
            break;
            
        case 1: // Top 3 highest values centroid
            {
                std::vector<size_t> indices(z_values.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + 3, indices.end(),
                                [&z_values](size_t a, size_t b) { return z_values[a] > z_values[b]; });
                
                for (G4int i = 0; i < 3 && i < static_cast<G4int>(indices.size()); ++i) {
                    size_t idx = indices[i];
                    weighted_x += x_coords[idx];
                    weighted_y += y_coords[idx];
                }
                weighted_x /= 3.0;
                weighted_y /= 3.0;
            }
            break;
            
        case 2: // Maximum value position with small offset
            {
                size_t max_idx = std::distance(z_values.begin(), std::max_element(z_values.begin(), z_values.end()));
                weighted_x = x_coords[max_idx];
                weighted_y = y_coords[max_idx];
                
                // Add small random offset to avoid getting stuck
                G4double offset_radius = 0.02*mm * strategy;
                G4double angle = 2.0 * TMath::Pi() * (strategy % 4) / 4.0;
                weighted_x += offset_radius * TMath::Cos(angle);
                weighted_y += offset_radius * TMath::Sin(angle);
            }
            break;
            
        default:
            // Geometric center
            weighted_x = std::accumulate(x_coords.begin(), x_coords.end(), 0.0) / x_coords.size();
            weighted_y = std::accumulate(y_coords.begin(), y_coords.end(), 0.0) / y_coords.size();
            break;
    }
    
    // Constrain center to detector bounds and handle constraint zones
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    weighted_x = TMath::Max(-half_det * 0.9, TMath::Min(half_det * 0.9, weighted_x));
    weighted_y = TMath::Max(-half_det * 0.9, TMath::Min(half_det * 0.9, weighted_y));
    
    // Handle constraint zones (red area or pixel exclusion)
    if (fConstrainToCenterPixel) {
        // Ensure initial guess is in red area
        if (!IsPointInRedArea(weighted_x, weighted_y)) {
            // Find a valid point in red area
            std::vector<std::pair<G4double, G4double>> red_samples;
            G4int n_samples = GenerateRedAreaSamples(red_samples, 10);
            if (n_samples > 0) {
                // Use the first valid sample or one close to our guess
                G4double best_dist = 1e10;
                size_t best_idx = 0;
                for (size_t i = 0; i < red_samples.size(); ++i) {
                    G4double dx = red_samples[i].first - weighted_x;
                    G4double dy = red_samples[i].second - weighted_y;
                    G4double dist = TMath::Sqrt(dx*dx + dy*dy);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                weighted_x = red_samples[best_idx].first;
                weighted_y = red_samples[best_idx].second;
            }
        }
    } else {
        // Handle pixel exclusion zones
        const G4double d0 = 0.01*mm;
        if (IsPointInsidePixelZone(weighted_x, weighted_y, d0)) {
            // Move to allowed position
            G4bool found_valid = false;
            const G4double search_radius = fDetectorGeometry.pixel_spacing * 0.7;
            
            for (G4int angle_step = 0; angle_step < 8 && !found_valid; ++angle_step) {
                G4double angle = angle_step * TMath::Pi() / 4.0;
                G4double test_x = weighted_x + search_radius * TMath::Cos(angle);
                G4double test_y = weighted_y + search_radius * TMath::Sin(angle);
                
                if (TMath::Abs(test_x) <= half_det && TMath::Abs(test_y) <= half_det && 
                    !IsPointInsidePixelZone(test_x, test_y, d0)) {
                    weighted_x = test_x;
                    weighted_y = test_y;
                    found_valid = true;
                }
            }
        }
    }
    
    // Calculate physically reasonable sigma estimates based on data spread
    G4double data_span_x = 0.0, data_span_y = 0.0;
    if (x_coords.size() > 1) {
        auto x_minmax = std::minmax_element(x_coords.begin(), x_coords.end());
        auto y_minmax = std::minmax_element(y_coords.begin(), y_coords.end());
        data_span_x = *x_minmax.second - *x_minmax.first;
        data_span_y = *y_minmax.second - *y_minmax.first;
    }
    
    // Use data spread to estimate sigma, but ensure physical bounds
    G4double sigma_x_guess = TMath::Max(data_span_x * 0.3, 0.1*mm); // At least 100 microns
    G4double sigma_y_guess = TMath::Max(data_span_y * 0.3, 0.1*mm);
    
    // Apply physical bounds for sigma
    const G4double min_physical_sigma = 0.05*mm;  // 50 microns minimum
    const G4double max_physical_sigma = fDetectorGeometry.detector_size * 0.3; // 30% of detector max
    
    sigma_x_guess = TMath::Max(min_physical_sigma, TMath::Min(max_physical_sigma, sigma_x_guess));
    sigma_y_guess = TMath::Max(min_physical_sigma, TMath::Min(max_physical_sigma, sigma_y_guess));
    
    // Add strategy-dependent variation to sigma
    G4double sigma_factor = 1.0 + 0.3 * strategy; // 1.0, 1.3, 1.6, 1.9...
    sigma_x_guess *= sigma_factor;
    sigma_y_guess *= sigma_factor;
    
    // Ensure we stay within bounds
    sigma_x_guess = TMath::Min(sigma_x_guess, max_physical_sigma);
    sigma_y_guess = TMath::Min(sigma_y_guess, max_physical_sigma);
    
    // Set initial parameters
    initialParams[0] = (z_max - z_min) * 0.9;  // amplitude
    initialParams[1] = weighted_x;             // x0
    initialParams[2] = weighted_y;             // y0
    initialParams[3] = sigma_x_guess;          // sigma_x
    initialParams[4] = sigma_y_guess;          // sigma_y
    initialParams[5] = 0.0;                    // theta
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
    const G4double sigma_x = params[3];
    const G4double sigma_y = params[4];
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    
    // Penalty for being outside detector bounds
    if (TMath::Abs(x0) > half_det) {
        penalty += fConstraintPenalty * (TMath::Abs(x0) - half_det);
    }
    if (TMath::Abs(y0) > half_det) {
        penalty += fConstraintPenalty * (TMath::Abs(y0) - half_det);
    }
    
    // Add strong penalties for unphysical sigma values
    const G4double min_physical_sigma = 0.05*mm; // 50 microns minimum for physical Gaussian
    const G4double max_physical_sigma = fDetectorGeometry.detector_size * 0.4; // Max 40% of detector
    
    if (sigma_x < min_physical_sigma) {
        penalty += fConstraintPenalty * 1000.0 * (min_physical_sigma - sigma_x);
    }
    if (sigma_y < min_physical_sigma) {
        penalty += fConstraintPenalty * 1000.0 * (min_physical_sigma - sigma_y);
    }
    if (sigma_x > max_physical_sigma) {
        penalty += fConstraintPenalty * 100.0 * (sigma_x - max_physical_sigma);
    }
    if (sigma_y > max_physical_sigma) {
        penalty += fConstraintPenalty * 100.0 * (sigma_y - max_physical_sigma);
    }
    
    // Apply different constraint penalty based on center pixel constraint mode
    if (fConstrainToCenterPixel) {
        // Use red area constraints when center pixel is set
        if (!IsPointInRedArea(x0, y0)) {
            penalty += fConstraintPenalty * 1000.0; // Very strong penalty for leaving red area
        }
    } else {
        // Original penalty for being too close to pixel center or inside pixel area
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
    
    // If center pixel constraints are enabled, automatically use red area constraints
    if (fConstrainToCenterPixel) {
        if (verbose) {
            G4cout << "Center pixel constraint detected - using red area constrained fitting" << G4endl;
        }
        return FitGaussian3DWithRedAreaConstraints(x_coords, y_coords, z_values, z_errors, verbose);
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
    
    // Try multiple fitting strategies with improved initial guesses
    G4bool fit_found = false;
    G4double best_chi2 = 1e10;
    G4double best_params[fNParams];
    
    for (G4int attempt = 0; attempt < fMaxFitAttempts && !fit_found; ++attempt) {
        if (verbose) {
            G4cout << "\n--- Fit Attempt " << (attempt + 1) << " ---" << G4endl;
        }
        
        try {
            // Calculate improved initial parameter estimates
            G4double fitParams[fNParams];
            CalculatePhysicalInitialGuess(x_coords, y_coords, z_values, fitParams, attempt);
            
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
            
            // Additional physics check - ensure reasonable sigma values
            G4bool physics_ok = (fitParams[3] > 0.05*mm && fitParams[4] > 0.05*mm && 
                                fitParams[3] < fDetectorGeometry.detector_size * 0.4 && 
                                fitParams[4] < fDetectorGeometry.detector_size * 0.4);
            
            // Calculate final chi-squared (without penalties)
            G4double final_chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, fitParams);
            
            if (verbose) {
                G4cout << "  Final chi2red: " << final_chi2 << G4endl;
                G4cout << "  Constraints satisfied: " << (constraints_ok ? "Yes" : "No") << G4endl;
                G4cout << "  Physics check passed: " << (physics_ok ? "Yes" : "No") << G4endl;
            }
            
            // Accept fit if constraints are satisfied, physics is reasonable, and chi2red is good
            if (constraints_ok && physics_ok && final_chi2 < best_chi2 && final_chi2 > 0) {
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
                       << ", physics: " << physics_ok 
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

void Gaussian3DFitter::GetPixelBounds(G4double center_x, G4double center_y, G4double* bounds) const
{
    const G4double pixel_half_size = fDetectorGeometry.pixel_size / 2.0;
    bounds[0] = center_x - pixel_half_size; // x_min
    bounds[1] = center_x + pixel_half_size; // x_max
    bounds[2] = center_y - pixel_half_size; // y_min
    bounds[3] = center_y + pixel_half_size; // y_max
}

void Gaussian3DFitter::SetCenterPixelPosition(G4double center_x, G4double center_y)
{
    fCenterPixelX = center_x;
    fCenterPixelY = center_y;
    fConstrainToCenterPixel = true;
}

void Gaussian3DFitter::SetNeighborhoodRadius(G4int radius)
{
    fNeighborhoodRadius = radius;
}

void Gaussian3DFitter::SetUseRedAreaConstraints(G4bool enable)
{
    fUseRedAreaConstraints = enable;
}

void Gaussian3DFitter::CalculateNeighborhoodBounds(G4double* bounds) const
{
    // Calculate the boundaries of the neighborhood around the center pixel
    // bounds[0] = x_min, bounds[1] = x_max, bounds[2] = y_min, bounds[3] = y_max
    
    const G4double half_neighborhood = fNeighborhoodRadius * fDetectorGeometry.pixel_spacing;
    
    bounds[0] = fCenterPixelX - half_neighborhood; // x_min
    bounds[1] = fCenterPixelX + half_neighborhood; // x_max
    bounds[2] = fCenterPixelY - half_neighborhood; // y_min
    bounds[3] = fCenterPixelY + half_neighborhood; // y_max
    
    // Constrain to detector bounds
    const G4double half_det = fDetectorGeometry.detector_size / 2.0;
    bounds[0] = TMath::Max(bounds[0], -half_det);
    bounds[1] = TMath::Min(bounds[1], half_det);
    bounds[2] = TMath::Max(bounds[2], -half_det);
    bounds[3] = TMath::Min(bounds[3], half_det);
}

G4bool Gaussian3DFitter::IsPointInNeighborhood(G4double x, G4double y) const
{
    G4double bounds[4];
    CalculateNeighborhoodBounds(bounds);
    
    return (x >= bounds[0] && x <= bounds[1] && y >= bounds[2] && y <= bounds[3]);
}

G4bool Gaussian3DFitter::IsPointInRedArea(G4double x, G4double y) const
{
    // The "red area" is the valid region where Gaussian center can be placed:
    // 1. Must be within neighborhood boundary
    // 2. Must be outside d0 (10 micron) radius from any pixel center
    // 3. Must be outside any pixel area
    
    // Check if point is within neighborhood boundary
    if (!IsPointInNeighborhood(x, y)) {
        return false;
    }
    
    // Check if point is too close to any pixel center or inside any pixel
    const G4double d0 = 0.01*mm; // 10 micron minimum distance from pixel center
    if (IsPointInsidePixelZone(x, y, d0)) {
        return false;
    }
    
    return true;
}

G4bool Gaussian3DFitter::GenerateQuadrantGuess(G4int quadrant, G4double& x_guess, G4double& y_guess) const
{
    // Generate initial guess within specified quadrant of red area
    // quadrant: 0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right
    
    G4double bounds[4];
    CalculateNeighborhoodBounds(bounds);
    
    // Divide the neighborhood into quadrants
    const G4double center_x = fCenterPixelX;
    const G4double center_y = fCenterPixelY;
    
    G4double quad_x_min, quad_x_max, quad_y_min, quad_y_max;
    
    switch (quadrant) {
        case 0: // top-right
            quad_x_min = center_x;
            quad_x_max = bounds[1];
            quad_y_min = center_y;
            quad_y_max = bounds[3];
            break;
        case 1: // top-left
            quad_x_min = bounds[0];
            quad_x_max = center_x;
            quad_y_min = center_y;
            quad_y_max = bounds[3];
            break;
        case 2: // bottom-left
            quad_x_min = bounds[0];
            quad_x_max = center_x;
            quad_y_min = bounds[2];
            quad_y_max = center_y;
            break;
        case 3: // bottom-right
            quad_x_min = center_x;
            quad_x_max = bounds[1];
            quad_y_min = bounds[2];
            quad_y_max = center_y;
            break;
        default:
            return false;
    }
    
    // Try to find a valid point in this quadrant
    const G4int max_attempts = 50;
    const G4double d0 = 0.01*mm;
    
    for (G4int attempt = 0; attempt < max_attempts; ++attempt) {
        // Generate random point in quadrant
        x_guess = quad_x_min + (quad_x_max - quad_x_min) * (G4double)rand() / RAND_MAX;
        y_guess = quad_y_min + (quad_y_max - quad_y_min) * (G4double)rand() / RAND_MAX;
        
        // Check if point is in red area
        if (IsPointInRedArea(x_guess, y_guess)) {
            return true;
        }
    }
    
    // If no valid random point found, try systematic grid search
    const G4int grid_points = 10;
    const G4double dx = (quad_x_max - quad_x_min) / grid_points;
    const G4double dy = (quad_y_max - quad_y_min) / grid_points;
    
    for (G4int i = 1; i < grid_points; ++i) {
        for (G4int j = 1; j < grid_points; ++j) {
            x_guess = quad_x_min + i * dx;
            y_guess = quad_y_min + j * dy;
            
            if (IsPointInRedArea(x_guess, y_guess)) {
                return true;
            }
        }
    }
    
    return false; // No valid point found in this quadrant
}

G4int Gaussian3DFitter::GenerateRedAreaSamples(std::vector<std::pair<G4double, G4double>>& initial_guesses, G4int max_points) const
{
    initial_guesses.clear();
    
    // Strategy 1: Systematic grid sampling
    std::vector<std::pair<G4double, G4double>> systematic_samples;
    G4int systematic_count = SampleRedAreaSystematically(systematic_samples, 15);
    
    // Strategy 2: Adaptive sampling based on data distribution  
    std::vector<std::pair<G4double, G4double>> adaptive_samples;
    // Note: adaptive sampling will be called from the main fitting function with data
    
    // Strategy 3: Enhanced quadrant sampling
    std::vector<std::pair<G4double, G4double>> quadrant_samples;
    for (G4int quad = 0; quad < 4; ++quad) {
        // Multiple samples per quadrant for better coverage
        for (G4int attempt = 0; attempt < 3; ++attempt) {
            G4double x_guess, y_guess;
            if (GenerateQuadrantGuess(quad, x_guess, y_guess)) {
                quadrant_samples.push_back(std::make_pair(x_guess, y_guess));
            }
        }
    }
    
    // Strategy 4: Radial sampling around center pixel
    const G4double center_x = fCenterPixelX;
    const G4double center_y = fCenterPixelY;
    const G4int n_radial_rings = 3;
    const G4int points_per_ring = 8;
    
    for (G4int ring = 1; ring <= n_radial_rings; ++ring) {
        G4double radius = ring * fDetectorGeometry.pixel_spacing * 0.3; // Smaller radii for dense sampling
        for (G4int point = 0; point < points_per_ring; ++point) {
            G4double angle = 2.0 * TMath::Pi() * point / points_per_ring;
            G4double test_x = center_x + radius * TMath::Cos(angle);
            G4double test_y = center_y + radius * TMath::Sin(angle);
            
            if (IsPointInRedArea(test_x, test_y)) {
                quadrant_samples.push_back(std::make_pair(test_x, test_y));
            }
        }
    }
    
    // Combine all sampling strategies
    initial_guesses.insert(initial_guesses.end(), systematic_samples.begin(), systematic_samples.end());
    initial_guesses.insert(initial_guesses.end(), quadrant_samples.begin(), quadrant_samples.end());
    
    // Remove duplicates and limit to max_points
    if (initial_guesses.size() > static_cast<size_t>(max_points)) {
        // Keep the most diverse set of points
        std::vector<std::pair<G4double, G4double>> filtered_guesses;
        filtered_guesses.reserve(max_points);
        
        const G4double min_separation = fDetectorGeometry.pixel_size * 0.5; // Minimum separation between points
        
        for (const auto& candidate : initial_guesses) {
            G4bool too_close = false;
            for (const auto& existing : filtered_guesses) {
                G4double dx = candidate.first - existing.first;
                G4double dy = candidate.second - existing.second;
                if (TMath::Sqrt(dx*dx + dy*dy) < min_separation) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                filtered_guesses.push_back(candidate);
                if (filtered_guesses.size() >= static_cast<size_t>(max_points)) {
                    break;
                }
            }
        }
        
        initial_guesses = filtered_guesses;
    }
    
    return initial_guesses.size();
}

G4int Gaussian3DFitter::SampleRedAreaSystematically(std::vector<std::pair<G4double, G4double>>& samples, G4int grid_resolution) const
{
    samples.clear();
    
    G4double bounds[4];
    CalculateNeighborhoodBounds(bounds);
    
    const G4double dx = (bounds[1] - bounds[0]) / grid_resolution;
    const G4double dy = (bounds[3] - bounds[2]) / grid_resolution;
    
    // Systematic grid sampling of the neighborhood bounds
    for (G4int i = 0; i <= grid_resolution; ++i) {
        for (G4int j = 0; j <= grid_resolution; ++j) {
            G4double test_x = bounds[0] + i * dx;
            G4double test_y = bounds[2] + j * dy;
            
            if (IsPointInRedArea(test_x, test_y)) {
                samples.push_back(std::make_pair(test_x, test_y));
            }
        }
    }
    
    return samples.size();
}

G4int Gaussian3DFitter::SampleRedAreaAdaptively(std::vector<std::pair<G4double, G4double>>& samples,
                                                const std::vector<G4double>& x_coords,
                                                const std::vector<G4double>& y_coords,
                                                const std::vector<G4double>& z_values,
                                                G4int num_samples) const
{
    samples.clear();
    
    if (x_coords.empty() || y_coords.empty() || z_values.empty()) {
        return 0;
    }
    
    // Find the region with highest charge density for adaptive guidance
    G4double weighted_x = 0.0, weighted_y = 0.0, total_weight = 0.0;
    const G4double z_min = *std::min_element(z_values.begin(), z_values.end());
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        if (z_values[i] > z_min) {
            G4double weight = z_values[i] - z_min;
            total_weight += weight;
            weighted_x += x_coords[i] * weight;
            weighted_y += y_coords[i] * weight;
        }
    }
    
    if (total_weight > 0) {
        weighted_x /= total_weight;
        weighted_y /= total_weight;
    } else {
        weighted_x = fCenterPixelX;
        weighted_y = fCenterPixelY;
    }
    
    // Generate samples around the weighted center and other promising locations
    const G4double search_radius = fDetectorGeometry.pixel_spacing;
    const G4int rings = 2;
    const G4int points_per_ring = num_samples / (rings + 1);
    
    // Center point
    if (IsPointInRedArea(weighted_x, weighted_y)) {
        samples.push_back(std::make_pair(weighted_x, weighted_y));
    }
    
    // Radial sampling around weighted center
    for (G4int ring = 1; ring <= rings; ++ring) {
        G4double radius = ring * search_radius * 0.5;
        for (G4int point = 0; point < points_per_ring; ++point) {
            G4double angle = 2.0 * TMath::Pi() * point / points_per_ring;
            G4double test_x = weighted_x + radius * TMath::Cos(angle);
            G4double test_y = weighted_y + radius * TMath::Sin(angle);
            
            if (IsPointInRedArea(test_x, test_y)) {
                samples.push_back(std::make_pair(test_x, test_y));
            }
        }
    }
    
    // Also add samples around high-value data points
    std::vector<size_t> high_value_indices;
    const G4double z_max = *std::max_element(z_values.begin(), z_values.end());
    const G4double threshold = z_min + 0.7 * (z_max - z_min);
    
    for (size_t i = 0; i < z_values.size(); ++i) {
        if (z_values[i] > threshold) {
            high_value_indices.push_back(i);
        }
    }
    
    // Sample around top 3 high-value points
    const G4int max_high_points = TMath::Min(3, static_cast<G4int>(high_value_indices.size()));
    for (G4int idx = 0; idx < max_high_points; ++idx) {
        size_t data_idx = high_value_indices[idx];
        G4double data_x = x_coords[data_idx];
        G4double data_y = y_coords[data_idx];
        
        // Small radius around each high-value point
        for (G4int angle_step = 0; angle_step < 4; ++angle_step) {
            G4double angle = angle_step * TMath::Pi() / 2.0;
            G4double offset_radius = fDetectorGeometry.pixel_size * 0.8;
            G4double test_x = data_x + offset_radius * TMath::Cos(angle);
            G4double test_y = data_y + offset_radius * TMath::Sin(angle);
            
            if (IsPointInRedArea(test_x, test_y)) {
                samples.push_back(std::make_pair(test_x, test_y));
            }
        }
    }
    
    return samples.size();
}

G4double Gaussian3DFitter::AnalyzeRedAreaCoverage(G4bool verbose) const
{
    if (!fConstrainToCenterPixel) {
        if (verbose) {
            G4cout << "Red area analysis requires center pixel to be set" << G4endl;
        }
        return 0.0;
    }
    
    G4double bounds[4];
    CalculateNeighborhoodBounds(bounds);
    
    const G4double total_neighborhood_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
    
    // High-resolution sampling to estimate red area coverage
    const G4int analysis_resolution = 50;
    const G4double dx = (bounds[1] - bounds[0]) / analysis_resolution;
    const G4double dy = (bounds[3] - bounds[2]) / analysis_resolution;
    
    G4int total_points = 0;
    G4int red_area_points = 0;
    G4int pixel_area_points = 0;
    G4int d0_zone_points = 0;
    G4int outside_neighborhood_points = 0;
    
    for (G4int i = 0; i <= analysis_resolution; ++i) {
        for (G4int j = 0; j <= analysis_resolution; ++j) {
            total_points++;
            G4double test_x = bounds[0] + i * dx;
            G4double test_y = bounds[2] + j * dy;
            
            // Check different zone classifications
            if (!IsPointInNeighborhood(test_x, test_y)) {
                outside_neighborhood_points++;
                continue;
            }
            
            const G4double d0 = 0.01*mm; // 10 micron buffer
            if (IsPointInsidePixelZone(test_x, test_y, 0.0)) {
                pixel_area_points++;
            } else if (IsPointInsidePixelZone(test_x, test_y, d0)) {
                d0_zone_points++;
            } else if (IsPointInRedArea(test_x, test_y)) {
                red_area_points++;
            }
        }
    }
    
    const G4double point_area = dx * dy;
    const G4double red_area_estimate = red_area_points * point_area;
    const G4double pixel_area_total = pixel_area_points * point_area;
    const G4double d0_zone_area = d0_zone_points * point_area;
    
    if (verbose) {
        G4cout << "\n=== RED AREA COVERAGE ANALYSIS ===" << G4endl;
        G4cout << "Neighborhood bounds: [" << bounds[0] << ", " << bounds[1] 
               << "] × [" << bounds[2] << ", " << bounds[3] << "] mm" << G4endl;
        G4cout << "Total neighborhood area: " << total_neighborhood_area << " mm²" << G4endl;
        G4cout << "\nArea breakdown (estimated):" << G4endl;
        G4cout << "  Red area (valid for fitting): " << red_area_estimate << " mm² (" 
               << (red_area_estimate / total_neighborhood_area * 100.0) << "%)" << G4endl;
        G4cout << "  Pixel areas: " << pixel_area_total << " mm² (" 
               << (pixel_area_total / total_neighborhood_area * 100.0) << "%)" << G4endl;
        G4cout << "  d0 exclusion zones: " << d0_zone_area << " mm² (" 
               << (d0_zone_area / total_neighborhood_area * 100.0) << "%)" << G4endl;
        G4cout << "\nSampling statistics (resolution " << analysis_resolution << "×" << analysis_resolution << "):" << G4endl;
        G4cout << "  Total sample points: " << total_points << G4endl;
        G4cout << "  Red area points: " << red_area_points << " (" 
               << (red_area_points * 100.0 / total_points) << "%)" << G4endl;
        G4cout << "  Pixel area points: " << pixel_area_points << " (" 
               << (pixel_area_points * 100.0 / total_points) << "%)" << G4endl;
        G4cout << "  d0 zone points: " << d0_zone_points << " (" 
               << (d0_zone_points * 100.0 / total_points) << "%)" << G4endl;
        G4cout << "  Outside neighborhood: " << outside_neighborhood_points << " (" 
               << (outside_neighborhood_points * 100.0 / total_points) << "%)" << G4endl;
        
        // Analysis and recommendations
        G4cout << "\nAnalysis:" << G4endl;
        const G4double red_fraction = red_area_estimate / total_neighborhood_area;
        if (red_fraction < 0.1) {
            G4cout << "  ⚠ WARNING: Red area is very small (" << (red_fraction * 100.0) 
                   << "%). Consider adjusting constraints." << G4endl;
        } else if (red_fraction < 0.3) {
            G4cout << "  ⚠ Red area is limited (" << (red_fraction * 100.0) 
                   << "%). Comprehensive sampling is important." << G4endl;
        } else {
            G4cout << "  ✓ Red area coverage is reasonable (" << (red_fraction * 100.0) << "%)." << G4endl;
        }
        
        G4cout << "======================================" << G4endl;
    }
    
    return red_area_estimate;
}

G4double Gaussian3DFitter::CalculateGaussianTrueDistance(const std::vector<G4double>& x_coords,
                                                        const std::vector<G4double>& y_coords,
                                                        const std::vector<G4double>& z_values,
                                                        const G4double* params) const
{
    // Calculate a distance metric that represents how well the Gaussian fits the true data
    // This could be based on chi-squared, residual statistics, or other quality metrics
    
    if (x_coords.empty()) {
        return 1e10; // Large distance for invalid data
    }
    
    // Calculate chi-squared as a basis for true distance
    G4double chi2 = 0.0;
    G4double sum_residuals_sq = 0.0;
    G4double sum_values = 0.0;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], params);
        const G4double residual = z_values[i] - z_fit;
        
        chi2 += residual * residual;
        sum_residuals_sq += residual * residual;
        sum_values += z_values[i];
    }
    
    // Normalize by number of points and average value
    const G4double mean_value = sum_values / x_coords.size();
    const G4double normalized_chi2 = chi2 / (x_coords.size() * TMath::Max(1.0, mean_value));
    
    // Add penalty for constraint violations
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    
    G4double constraint_penalty = 0.0;
    if (!IsPointInRedArea(x0, y0)) {
        constraint_penalty = 1000.0; // Large penalty for being outside red area
    }
    
    return normalized_chi2 + constraint_penalty;
}

Gaussian3DFitter::FitResults Gaussian3DFitter::FitGaussian3DWithRedAreaConstraints(const std::vector<G4double>& x_coords,
                                                                                   const std::vector<G4double>& y_coords,
                                                                                   const std::vector<G4double>& z_values,
                                                                                   const std::vector<G4double>& z_errors,
                                                                                   G4bool verbose)
{
    FitResults best_results;
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3DWithRedAreaConstraints - Error: Inconsistent input array sizes" << G4endl;
        }
        return best_results;
    }
    
    if (x_coords.size() < 7) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3DWithRedAreaConstraints - Error: Insufficient data points (" 
                   << x_coords.size() << " < 7)" << G4endl;
        }
        return best_results;
    }
    
    if (!fConstrainToCenterPixel) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3DWithRedAreaConstraints - Error: Center pixel not set" << G4endl;
        }
        return best_results;
    }
    
    if (verbose) {
        G4cout << "\n=== Enhanced Gaussian3D Fitting with Comprehensive Red Area Search ===" << G4endl;
        G4cout << "Data points: " << x_coords.size() << G4endl;
        G4cout << "Center pixel: (" << fCenterPixelX << ", " << fCenterPixelY << ") mm" << G4endl;
        G4cout << "Neighborhood radius: " << fNeighborhoodRadius << " pixels" << G4endl;
        
        G4double bounds[4];
        CalculateNeighborhoodBounds(bounds);
        G4cout << "Neighborhood bounds: [" << bounds[0] << ", " << bounds[1] 
               << "] × [" << bounds[2] << ", " << bounds[3] << "] mm" << G4endl;
        
        // Analyze red area coverage
        AnalyzeRedAreaCoverage(true);
    }
    
    // Enable red area constraints for this fitting
    G4bool original_red_area_flag = fUseRedAreaConstraints;
    fUseRedAreaConstraints = true;
    
    // Generate comprehensive set of initial guesses
    std::vector<std::pair<G4double, G4double>> systematic_samples;
    std::vector<std::pair<G4double, G4double>> adaptive_samples;
    
    // Get systematic samples
    G4int systematic_count = GenerateRedAreaSamples(systematic_samples, 24);
    
    // Get adaptive samples based on data distribution
    G4int adaptive_count = SampleRedAreaAdaptively(adaptive_samples, x_coords, y_coords, z_values, 12);
    
    // Combine all initial guesses
    std::vector<std::pair<G4double, G4double>> all_initial_guesses;
    all_initial_guesses.insert(all_initial_guesses.end(), systematic_samples.begin(), systematic_samples.end());
    all_initial_guesses.insert(all_initial_guesses.end(), adaptive_samples.begin(), adaptive_samples.end());
    
    if (verbose) {
        G4cout << "Generated " << systematic_count << " systematic samples and " 
               << adaptive_count << " adaptive samples" << G4endl;
        G4cout << "Total initial guesses: " << all_initial_guesses.size() << G4endl;
    }
    
    if (all_initial_guesses.empty()) {
        if (verbose) {
            G4cout << "No valid initial guesses found in red area!" << G4endl;
        }
        fUseRedAreaConstraints = original_red_area_flag;
        return best_results;
    }
    
    G4double best_true_distance = 1e10;
    G4bool any_fit_found = false;
    G4int successful_fits = 0;
    G4int fit_attempt = 0;
    
    // Multi-start optimization with all initial guesses
    for (const auto& initial_guess : all_initial_guesses) {
        fit_attempt++;
        
        if (verbose) {
            G4cout << "\n--- Fit Attempt " << fit_attempt << "/" << all_initial_guesses.size() 
                   << " at (" << initial_guess.first << ", " << initial_guess.second << ") ---" << G4endl;
        }
        
        // Set up fit parameters with current initial guess
        G4double fitParams[fNParams];
        CalculateInitialGuess(x_coords, y_coords, z_values, fitParams, 0);
        
        // Override center coordinates with current initial guess
        fitParams[1] = initial_guess.first;  // x0
        fitParams[2] = initial_guess.second; // y0
        
        try {
            // Perform fitting with red area constraints
            RobustSimplexFitWithRedAreaConstraints(x_coords, y_coords, z_values, z_errors, fitParams, false); // Suppress verbose for individual fits
            
            // Verify final result is in red area
            if (!IsPointInRedArea(fitParams[1], fitParams[2])) {
                if (verbose) {
                    G4cout << "  ✗ Final result outside red area: (" << fitParams[1] << ", " << fitParams[2] << ")" << G4endl;
                }
                continue;
            }
            
            successful_fits++;
            
            // Calculate quality metrics
            G4double true_distance = CalculateGaussianTrueDistance(x_coords, y_coords, z_values, fitParams);
            G4double chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, fitParams);
            
            if (verbose) {
                G4cout << "  ✓ Valid fit - Center: (" << fitParams[1] << ", " << fitParams[2] << ") mm" << G4endl;
                G4cout << "    True distance: " << true_distance << ", Chi2: " << chi2 << G4endl;
            }
            
            // Check if this is the best fit so far
            if (true_distance < best_true_distance) {
                best_true_distance = true_distance;
                any_fit_found = true;
                
                // Store results
                best_results.fit_type = ALL_DATA;
                best_results.n_points = x_coords.size();
                best_results.amplitude = fitParams[0];
                best_results.x0 = fitParams[1];
                best_results.y0 = fitParams[2];
                best_results.sigma_x = fitParams[3];
                best_results.sigma_y = fitParams[4];
                best_results.theta = fitParams[5];
                best_results.offset = fitParams[6];
                best_results.constraints_satisfied = true;
                
                // Calculate detailed statistics
                best_results.chi2red = chi2;
                best_results.ndf = best_results.n_points - fNParams;
                best_results.Pp = (best_results.ndf > 0) ? TMath::Prob(chi2, best_results.ndf) : 0.0;
                
                // Calculate residual statistics
                CalculateResidualStats(x_coords, y_coords, z_values, fitParams, 
                                     best_results.residual_mean, best_results.residual_std);
                
                // Error estimates
                const G4double reduced_chi2 = best_results.ndf > 0 ? best_results.chi2red / best_results.ndf : 1.0;
                const G4double error_scale = TMath::Sqrt(TMath::Max(1.0, reduced_chi2));
                
                best_results.amplitude_err = TMath::Abs(best_results.amplitude) * 0.05 * error_scale;
                best_results.x0_err = Constants::BASE_POSITION_ERROR * error_scale;
                best_results.y0_err = Constants::BASE_POSITION_ERROR * error_scale;
                best_results.sigma_x_err = TMath::Abs(best_results.sigma_x) * 0.08 * error_scale;
                best_results.sigma_y_err = TMath::Abs(best_results.sigma_y) * 0.08 * error_scale;
                best_results.theta_err = 0.05 * error_scale;
                best_results.offset_err = TMath::Abs(best_results.offset) * Constants::ERROR_SCALE_FRACTION * error_scale;
                
                if (verbose) {
                    G4cout << "    ★ NEW BEST FIT ★ (attempt " << fit_attempt << ")" << G4endl;
                }
            }
            
        } catch (const std::exception& e) {
            if (verbose) {
                G4cout << "  Exception in attempt " << fit_attempt << ": " << e.what() << G4endl;
            }
        } catch (...) {
            if (verbose) {
                G4cout << "  Unknown exception in attempt " << fit_attempt << G4endl;
            }
        }
    }
    
    // Restore original red area constraints
    fUseRedAreaConstraints = original_red_area_flag;
    
    if (!any_fit_found) {
        if (verbose) {
            G4cout << "\nNo valid fits found in " << fit_attempt << " attempts!" << G4endl;
            G4cout << "Successful fits: " << successful_fits << "/" << fit_attempt << G4endl;
        }
        return best_results; // Return empty results
    }
    
    if (verbose) {
        G4cout << "\n=== OPTIMAL FIT RESULTS (Enhanced Red Area Search) ===" << G4endl;
        G4cout << "Search summary:" << G4endl;
        G4cout << "  Total attempts: " << fit_attempt << G4endl;
        G4cout << "  Successful fits: " << successful_fits << G4endl;
        G4cout << "  Best true distance: " << best_true_distance << G4endl;
        G4cout << "\nBest fit parameters:" << G4endl;
        G4cout << "  Amplitude: " << best_results.amplitude << " ± " << best_results.amplitude_err << G4endl;
        G4cout << "  Center: (" << best_results.x0 << " ± " << best_results.x0_err << ", " 
               << best_results.y0 << " ± " << best_results.y0_err << ") mm" << G4endl;
        G4cout << "  Sigma: (" << best_results.sigma_x << " ± " << best_results.sigma_x_err << ", " 
               << best_results.sigma_y << " ± " << best_results.sigma_y_err << ") mm" << G4endl;
        G4cout << "  Theta: " << best_results.theta << " ± " << best_results.theta_err << " rad ("
               << best_results.theta * 180.0 / TMath::Pi() << "° ± " 
               << best_results.theta_err * 180.0 / TMath::Pi() << "°)" << G4endl;
        G4cout << "  Offset: " << best_results.offset << " ± " << best_results.offset_err << G4endl;
        G4cout << "\nStatistics:" << G4endl;
        G4cout << "  Chi2/NDF: " << best_results.chi2red << "/" << best_results.ndf 
               << " = " << (best_results.ndf > 0 ? best_results.chi2red / best_results.ndf : 0) << G4endl;
        G4cout << "  Probability: " << best_results.Pp << G4endl;
        G4cout << "  Residual mean: " << best_results.residual_mean << G4endl;
        G4cout << "  Residual std: " << best_results.residual_std << G4endl;
        G4cout << "=========================================================" << G4endl;
    }
    
    return best_results;
}

void Gaussian3DFitter::RobustSimplexFitWithRedAreaConstraints(const std::vector<G4double>& x_coords,
                                                              const std::vector<G4double>& y_coords,
                                                              const std::vector<G4double>& z_values,
                                                              const std::vector<G4double>& z_errors,
                                                              G4double* params,
                                                              G4bool verbose)
{
    // Modified simplex optimization that enforces red area constraints
    const G4int n_params = fNParams;
    const G4int max_iterations = Constants::MAX_ITERATIONS;
    const G4double tolerance = Constants::FIT_TOLERANCE;
    
    // Create initial simplex with better parameter space exploration
    std::vector<std::vector<G4double>> simplex(n_params + 1);
    std::vector<G4double> chi2_values(n_params + 1);
    
    // Initialize simplex vertices
    for (G4int i = 0; i <= n_params; ++i) {
        simplex[i].resize(n_params);
        for (G4int j = 0; j < n_params; ++j) {
            simplex[i][j] = params[j];
            if (i == j + 1) {
                // Parameter-specific perturbations (smaller for position parameters)
                G4double step;
                switch (j) {
                    case 1: case 2: // x0, y0 - smaller steps to stay in red area
                        step = 0.02*mm; // 20 micron steps
                        break;
                    default:
                        step = TMath::Abs(params[j]) * 0.1;
                        if (step < 0.01*mm) step = 0.01*mm;
                }
                simplex[i][j] += step;
            }
        }
        
        // Ensure center coordinates stay in red area
        if (!IsPointInRedArea(simplex[i][1], simplex[i][2])) {
            // Project back to red area
            G4double x_proj = simplex[i][1];
            G4double y_proj = simplex[i][2];
            
            // Simple projection - find nearest valid point
            const G4int search_attempts = 20;
            G4bool found_valid = false;
            
            for (G4int attempt = 0; attempt < search_attempts && !found_valid; ++attempt) {
                G4double search_radius = 0.05*mm * (attempt + 1);
                for (G4int angle_step = 0; angle_step < 8; ++angle_step) {
                    G4double angle = angle_step * TMath::Pi() / 4.0;
                    G4double test_x = params[1] + search_radius * TMath::Cos(angle);
                    G4double test_y = params[2] + search_radius * TMath::Sin(angle);
                    
                    if (IsPointInRedArea(test_x, test_y)) {
                        simplex[i][1] = test_x;
                        simplex[i][2] = test_y;
                        found_valid = true;
                        break;
                    }
                }
            }
            
            if (!found_valid) {
                // Fall back to original center
                simplex[i][1] = params[1];
                simplex[i][2] = params[2];
            }
        }
        
        chi2_values[i] = CalculateConstrainedChiSquaredWithRedArea(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
    }
    
    // Simplex optimization with red area constraints
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
            break;
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
        
        // Ensure reflected point satisfies red area constraint for center coordinates
        if (!IsPointInRedArea(reflected[1], reflected[2])) {
            // Project to red area or use a valid nearby point
            reflected[1] = centroid[1];
            reflected[2] = centroid[2];
        }
        
        ApplyParameterBounds(&reflected[0]);
        G4double chi2_reflected = CalculateConstrainedChiSquaredWithRedArea(x_coords, y_coords, z_values, z_errors, &reflected[0]);
        
        if (chi2_reflected < chi2_values[best_idx]) {
            // Expansion
            std::vector<G4double> expanded(n_params);
            for (G4int j = 0; j < n_params; ++j) {
                expanded[j] = centroid[j] + 2.0 * (centroid[j] - simplex[worst_idx][j]);
            }
            
            // Ensure expanded point satisfies red area constraint
            if (!IsPointInRedArea(expanded[1], expanded[2])) {
                expanded[1] = reflected[1];
                expanded[2] = reflected[2];
            }
            
            ApplyParameterBounds(&expanded[0]);
            G4double chi2_expanded = CalculateConstrainedChiSquaredWithRedArea(x_coords, y_coords, z_values, z_errors, &expanded[0]);
            
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
            
            // Ensure contracted point satisfies red area constraint
            if (!IsPointInRedArea(contracted[1], contracted[2])) {
                contracted[1] = centroid[1];
                contracted[2] = centroid[2];
            }
            
            ApplyParameterBounds(&contracted[0]);
            G4double chi2_contracted = CalculateConstrainedChiSquaredWithRedArea(x_coords, y_coords, z_values, z_errors, &contracted[0]);
            
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
                        
                        // Ensure shrunken point satisfies red area constraint
                        if (!IsPointInRedArea(simplex[i][1], simplex[i][2])) {
                            simplex[i][1] = simplex[best_idx][1];
                            simplex[i][2] = simplex[best_idx][2];
                        }
                        
                        ApplyParameterBounds(&simplex[i][0]);
                        chi2_values[i] = CalculateConstrainedChiSquaredWithRedArea(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
                    }
                }
            }
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
}

G4double Gaussian3DFitter::CalculateConstrainedChiSquaredWithRedArea(const std::vector<G4double>& x_coords,
                                                                     const std::vector<G4double>& y_coords,
                                                                     const std::vector<G4double>& z_values,
                                                                     const std::vector<G4double>& z_errors,
                                                                     const G4double* params)
{
    G4double chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, params);
    
    // Add large penalty if center is outside red area
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    
    if (!IsPointInRedArea(x0, y0)) {
        chi2 += fConstraintPenalty * 1000.0; // Very large penalty
    }
    
    return chi2;
} 