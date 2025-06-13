#include "2DLorentzianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <sstream>
#include <limits>
#include <numeric>
#include <set>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Thread-safe mutex for Ceres operations
static std::mutex gCeresLorentzianFitMutex;
static std::atomic<int> gGlobalCeresLorentzianFitCounter{0};

// Use shared Google logging initialization
void InitializeCeresLorentzian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Enhanced cost function with pixel integration for Lorentzian (analytical integration with horizontal errors)
struct PixelIntegratedLorentzianCostFunction {
    PixelIntegratedLorentzianCostFunction(double pixel_center_x, double pixel_y, double pixel_size, 
                                        double weight = 1.0, double horizontal_uncertainty = 0.0) 
        : pixel_center_x_(pixel_center_x), pixel_y_(pixel_y), pixel_size_(pixel_size), 
          weight_(weight), horizontal_uncertainty_(horizontal_uncertainty) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = m (center)
        // params[2] = gamma (HWHM)
        // params[3] = B (baseline)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& gamma = params[2];
        const T& B = params[3];
        
        // Robust handling of gamma
        T safe_gamma = ceres::abs(gamma);
        if (safe_gamma < T(1e-12)) {
            safe_gamma = T(1e-12);
        }
        
        // Include horizontal uncertainty in effective gamma for pixel integration
        // For Lorentzian: horizontal uncertainty affects the width parameter
        T effective_gamma = safe_gamma;
        if (horizontal_uncertainty_ > 0) {
            T horizontal_gamma = T(horizontal_uncertainty_);
            // Combine fit gamma with horizontal positional uncertainty
            // For Lorentzian, uncertainties add in quadrature for the width parameter
            effective_gamma = ceres::sqrt(safe_gamma * safe_gamma + horizontal_gamma * horizontal_gamma);
        }
        
        // Pixel integration using analytical formula for Lorentzian integral with effective gamma
        // ∫[x-Δx/2 to x+Δx/2] A / (1 + ((t-m)/γ_eff)²) dt + B*Δx
        
        T half_pixel = T(pixel_size_) / T(2.0);
        T x_left = T(pixel_center_x_) - half_pixel;
        T x_right = T(pixel_center_x_) + half_pixel;
        
        // Analytical integral: A * γ_eff * [arctan((x-m)/γ_eff)]
        T left_arg = (x_left - m) / effective_gamma;
        T right_arg = (x_right - m) / effective_gamma;
        T arctan_left = ceres::atan(left_arg);
        T arctan_right = ceres::atan(right_arg);
        
        T integrated_lorentzian = A * effective_gamma * (arctan_right - arctan_left);
        T baseline_contribution = B * T(pixel_size_);
        
        T predicted = integrated_lorentzian + baseline_contribution;
        
        // Apply additional horizontal error weighting to residual
        T effective_weight = T(weight_);
        if (horizontal_uncertainty_ > 0) {
            // Reduce weight for pixels with larger horizontal uncertainty
            // This implements the "horizontal error" weighting for Lorentzian distributions
            T uncertainty_factor = T(1.0) + T(horizontal_uncertainty_) / T(pixel_size_);
            effective_weight /= uncertainty_factor;
        }
        
        // Weighted residual with horizontal error correction
        residual[0] = effective_weight * (predicted - T(pixel_y_));
        
        return true;
    }
    
    static ceres::CostFunction* Create(double pixel_center_x, double pixel_y, double pixel_size, 
                                     double weight = 1.0, double horizontal_uncertainty = 0.0) {
        return (new ceres::AutoDiffCostFunction<PixelIntegratedLorentzianCostFunction, 1, 4>(
            new PixelIntegratedLorentzianCostFunction(pixel_center_x, pixel_y, pixel_size, weight, horizontal_uncertainty)));
    }
    
private:
    const double pixel_center_x_;
    const double pixel_y_;
    const double pixel_size_;
    const double weight_;
    const double horizontal_uncertainty_;
};

// Enhanced cost function for Lorentzian fitting with numerical stability
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B
struct LorentzianCostFunction {
    LorentzianCostFunction(double x, double y, double weight = 1.0) 
        : x_(x), y_(y), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = m (center)
        // params[2] = gamma (HWHM)
        // params[3] = B (baseline)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& gamma = params[2];
        const T& B = params[3];
        
        // Robust handling of gamma (prevent division by zero)
        T safe_gamma = ceres::abs(gamma);
        if (safe_gamma < T(1e-12)) {
            safe_gamma = T(1e-12);
        }
        
        // Lorentzian function: y(x) = A / (1 + ((x - m) / γ)^2) + B
        T dx = x_ - m;
        T normalized_dx = dx / safe_gamma;
        T denominator = T(1.0) + normalized_dx * normalized_dx;
        
        // Prevent numerical issues with very small denominators
        if (denominator < T(1e-12)) {
            denominator = T(1e-12);
        }
        
        T predicted = A / denominator + B;
        
        // Weighted residual for robust fitting
        residual[0] = T(weight_) * (predicted - T(y_));
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<LorentzianCostFunction, 1, 4>(
            new LorentzianCostFunction(x, y, weight)));
    }
    
private:
    const double x_;
    const double y_;
    const double weight_;
};

// Parameter estimation structures for Lorentzian
struct LorentzianParameterEstimates {
    double amplitude;
    double center;
    double gamma;
    double baseline;
    double amplitude_err;
    double center_err;
    double gamma_err;
    double baseline_err;
    bool valid;
    int method_used;
};

// Robust statistics calculations (reusing from Gaussian implementation)
struct DataStatistics {
    double mean, median, std_dev, mad;
    double q25, q75, min_val, max_val;
    double weighted_mean, total_weight;
    bool valid;
};

DataStatistics CalculateRobustStatisticsLorentzian(const std::vector<double>& x_vals, 
                                                   const std::vector<double>& y_vals) {
    DataStatistics stats;
    stats.valid = false;
    
    if (x_vals.size() != y_vals.size() || x_vals.empty()) {
        return stats;
    }
    
    // Basic statistics
    stats.min_val = *std::min_element(y_vals.begin(), y_vals.end());
    stats.max_val = *std::max_element(y_vals.begin(), y_vals.end());
    
    // Mean and standard deviation
    stats.mean = std::accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();
    
    double variance = 0.0;
    for (double val : y_vals) {
        variance += (val - stats.mean) * (val - stats.mean);
    }
    stats.std_dev = std::sqrt(variance / y_vals.size());
    
    // Median and quartiles
    std::vector<double> sorted_y = y_vals;
    std::sort(sorted_y.begin(), sorted_y.end());
    
    size_t n = sorted_y.size();
    if (n % 2 == 0) {
        stats.median = (sorted_y[n/2 - 1] + sorted_y[n/2]) / 2.0;
    } else {
        stats.median = sorted_y[n/2];
    }
    
    stats.q25 = sorted_y[n/4];
    stats.q75 = sorted_y[3*n/4];
    
    // Median Absolute Deviation
    std::vector<double> abs_deviations;
    for (double val : y_vals) {
        abs_deviations.push_back(std::abs(val - stats.median));
    }
    std::sort(abs_deviations.begin(), abs_deviations.end());
    stats.mad = abs_deviations[n/2] * 1.4826;
    
    // Weighted statistics
    stats.weighted_mean = 0.0;
    stats.total_weight = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - stats.q25);
        if (weight > 0) {
            stats.weighted_mean += x_vals[i] * weight;
            stats.total_weight += weight;
        }
    }
    
    if (stats.total_weight > 0) {
        stats.weighted_mean /= stats.total_weight;
    } else {
        stats.weighted_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
    }
    
    stats.valid = true;
    return stats;
} 

// Parameter estimation for Lorentzian distributions
LorentzianParameterEstimates EstimateLorentzianParameters(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    LorentzianParameterEstimates estimates;
    estimates.valid = false;
    estimates.method_used = 0;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return estimates;
    }
    
    DataStatistics stats = CalculateRobustStatisticsLorentzian(x_vals, y_vals);
    if (!stats.valid) {
        return estimates;
    }
    
    if (verbose) {
        std::cout << "Lorentzian data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", weighted_mean=" << stats.weighted_mean << std::endl;
    }
    
    // Method 1: Physics-based estimation for charge distributions
    estimates.center = stats.weighted_mean;
    estimates.baseline = std::min(stats.min_val, stats.q25);
    estimates.amplitude = stats.max_val - estimates.baseline;
    
    // For Lorentzian: gamma (HWHM) estimation based on charge spread
    // Lorentzian tails are wider than Gaussian, so use larger initial gamma
    double distance_spread = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - estimates.baseline);
        if (weight > 0.1 * estimates.amplitude) {
            double dx = x_vals[i] - estimates.center;
            distance_spread += weight * dx * dx;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0) {
        // For Lorentzian, gamma ≈ sqrt(2*sigma^2) where sigma is from Gaussian equivalent
        estimates.gamma = std::sqrt(2.0 * distance_spread / weight_sum);
    } else {
        estimates.gamma = pixel_spacing * 0.7; // Larger default for Lorentzian
    }
    
    // Apply physics-based bounds (Lorentzian has wider tails)
    estimates.gamma = std::max(pixel_spacing * 0.3, std::min(pixel_spacing * 3.0, estimates.gamma));
    estimates.amplitude = std::max(estimates.amplitude, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amplitude > 0 && estimates.gamma > 0 && 
        !std::isnan(estimates.center) && !std::isnan(estimates.amplitude) && 
        !std::isnan(estimates.gamma) && !std::isnan(estimates.baseline)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Lorentzian Method 1 (Physics-based): A=" << estimates.amplitude 
                     << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center = stats.median;
    estimates.baseline = stats.q25;
    estimates.amplitude = stats.q75 - stats.q25;
    estimates.gamma = std::max(stats.mad, pixel_spacing * 0.5);
    
    if (estimates.amplitude > 0 && estimates.gamma > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Lorentzian Method 2 (Robust statistical): A=" << estimates.amplitude 
                     << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center = center_estimate;
    estimates.baseline = 0.0;
    estimates.amplitude = stats.max_val;
    estimates.gamma = pixel_spacing * 0.7;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "Lorentzian Method 3 (Conservative fallback): A=" << estimates.amplitude 
                 << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                 << ", B=" << estimates.baseline << std::endl;
    }
    
    return estimates;
}

// Enhanced weighting strategies for spatial uncertainty and bias reduction (Lorentzian) with "horizontal errors"
// Implements the specific techniques requested:
// 1. Downweight central pixel: Assign larger uncertainty to highest-charge pixel
// 2. Distance-based weights: w_i ∝ 1/(1 + d_i/d₀) where d_i is distance from current estimate
// 3. Robust losses: Use Cauchy/Huber loss functions to moderate single pixel influence
// 4. Pixel integration: Model pixel response by integrating over pixel area
// 5. Spatial error maps: Position-dependent weighting based on reconstruction error maps
struct LorentzianWeightingConfig {
    // Core horizontal error techniques
    bool enable_central_pixel_downweight = true;      // Technique #1: Downweight central pixel
    bool enable_distance_based_weights = true;        // Technique #2: Distance-based weighting
    bool enable_robust_losses = true;                 // Technique #3: Robust loss functions
    bool enable_pixel_integration = true;             // Technique #4: Pixel integration model
    bool enable_spatial_error_maps = true;            // Technique #5: Spatial error maps
    
    // Advanced weighting features for enhanced performance
    bool enable_spatial_uncertainty = true;           // Spatial uncertainty from pixel finite size
    bool enable_adaptive_central_weighting = true;    // Adaptive central pixel downweighting
    bool enable_edge_pixel_upweighting = true;        // Give edge pixels higher weight for position info
    bool enable_robust_profile_weighting = true;      // Weight based on expected vs. measured profile
    bool enable_iterative_reweighting = true;         // Iteratively update weights during fitting
    bool enable_horizontal_error_correction = true;   // Implement horizontal errors from pixel integration
    bool enable_systematic_bias_correction = true;    // Correct for systematic reconstruction bias
    bool enable_adaptive_robust_losses = true;        // Adaptively select robust loss functions
    bool enable_charge_weighted_uncertainty = true;   // Weight uncertainties by charge magnitude
    bool enable_pixel_edge_uncertainty = true;        // Additional uncertainty for pixels near edges
    bool enable_correlation_weighting = true;         // Weight based on inter-pixel correlations
    bool enable_dynamic_loss_switching = true;        // Dynamically switch loss functions based on residuals
    
    // Horizontal error parameters (technique #1: central pixel downweighting)
    double central_pixel_weight_factor = 0.10;        // Reduce central pixel weight to 10% (more aggressive for Lorentzian)
    double central_downweight_threshold = 2.2;        // Charge concentration threshold for adaptive downweighting
    double max_central_pixel_uncertainty = 7.5;      // Maximum uncertainty multiplier for central pixel
    
    // Distance-based weighting parameters (technique #2)
    double distance_scale_d0 = 40.0;                 // d₀ parameter: w_i ∝ 1/(1 + d_i/d₀) [μm] (tighter)
    double distance_weight_cap = 6.0;                // Maximum distance-based weight multiplier
    
    // Robust loss parameters (technique #3)
    double robust_threshold_factor = 0.10;           // Threshold for robust loss functions (more aggressive for Lorentzian)
    double dynamic_loss_threshold = 1.8;             // Threshold for dynamic loss function switching [sigma]
    
    // Pixel integration parameters (technique #4)
    double horizontal_error_scale = 0.50;            // Scale factor for horizontal errors (fraction of pixel size)
    double spatial_uncertainty_factor = 0.48;       // Pixel size uncertainty factor for horizontal errors
    double pixel_edge_uncertainty_factor = 0.85;    // Additional uncertainty for pixels near edges
    
    // Spatial error map parameters (technique #5)
    double spatial_error_map_strength = 0.28;       // Strength of spatial error map corrections
    double position_dependent_bias_scale = 0.22;    // Scale for position-dependent bias corrections
    
    // Additional parameters for enhanced performance
    double edge_pixel_boost_factor = 1.8;           // Boost factor for edge pixels (higher for Lorentzian)
    double charge_uncertainty_floor = 0.03;         // Minimum relative charge uncertainty (3%)
    double systematic_bias_strength = 0.32;         // Strength of systematic bias correction (stronger for Lorentzian)
    double correlation_radius = 1.6;                // Radius for inter-pixel correlation weighting [pixels] (larger for Lorentzian tails)
    double adaptive_weight_update_rate = 0.65;      // Rate for iterative weight updates (slightly lower for Lorentzian stability)
};

// Calculate weights for Lorentzian fitting with enhanced spatial uncertainty handling
std::vector<double> CalculateLorentzianDataWeights(const std::vector<double>& x_vals,
                                                  const std::vector<double>& y_vals,
                                                  const LorentzianParameterEstimates& estimates,
                                                  double pixel_spacing = 50.0,
                                                  const LorentzianWeightingConfig& config = LorentzianWeightingConfig()) {
    std::vector<double> weights(x_vals.size(), 1.0);
    
    DataStatistics stats = CalculateRobustStatisticsLorentzian(x_vals, y_vals);
    if (!stats.valid) {
        return weights;
    }
    
    // Find the highest charge pixel (central pixel candidate) and charge statistics
    size_t max_charge_idx = 0;
    double max_charge = y_vals[0];
    double total_charge = 0.0;
    for (size_t i = 0; i < y_vals.size(); ++i) {
        total_charge += y_vals[i];
        if (y_vals[i] > max_charge) {
            max_charge = y_vals[i];
            max_charge_idx = i;
        }
    }
    double mean_charge = total_charge / y_vals.size();
    
    // Identify edge pixels (those farthest from center estimate)
    std::vector<std::pair<double, size_t>> distance_indices;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double dx = std::abs(x_vals[i] - estimates.center);
        distance_indices.push_back(std::make_pair(dx, i));
    }
    std::sort(distance_indices.begin(), distance_indices.end(), std::greater<std::pair<double, size_t>>());
    
    // Mark edge pixels (top 30% by distance)
    std::set<size_t> edge_pixel_indices;
    size_t num_edge_pixels = std::max(1, static_cast<int>(x_vals.size() * 0.3));
    for (size_t i = 0; i < num_edge_pixels && i < distance_indices.size(); ++i) {
        edge_pixel_indices.insert(distance_indices[i].second);
    }
    
    for (size_t i = 0; i < y_vals.size(); ++i) {
        double weight = 1.0;
        
        // 1. Base weight by charge magnitude (higher charge = higher weight)
        double charge_weight = std::sqrt(std::max(0.0, y_vals[i] - estimates.baseline));
        charge_weight = std::max(0.1, std::min(10.0, charge_weight / stats.mad));
        
        // 2. Enhanced central pixel downweighting strategy
        if (config.enable_central_pixel_downweight && i == max_charge_idx) {
            if (config.enable_adaptive_central_weighting) {
                // Adaptive downweighting based on charge concentration
                double charge_concentration = max_charge / mean_charge;
                double adaptive_factor = config.central_pixel_weight_factor;
                if (charge_concentration > config.central_downweight_threshold) {
                    // More aggressive downweighting for highly concentrated charge
                    adaptive_factor *= 0.75; // Balanced downweighting for Lorentzian
                }
                charge_weight *= adaptive_factor;
            } else {
                charge_weight *= config.central_pixel_weight_factor;
            }
        }
        
        // 3. Enhanced distance-based weighting: w_i ∝ 1/(1 + d_i/d₀) for center estimation
        double distance_weight = 1.0;
        if (config.enable_distance_based_weights && estimates.gamma > 0) {
            double dx = std::abs(x_vals[i] - estimates.center);
            
            // Implement the exact formula: w_i ∝ 1/(1 + d_i/d₀)
            // This naturally gives more weight to pixels closer to the current center estimate
            // which helps with convergence while still allowing position refinement
            distance_weight = 1.0 / (1.0 + dx / config.distance_scale_d0);
            
            // Apply configurable cap to prevent extreme weighting
            distance_weight = std::min(config.distance_weight_cap, distance_weight);
        }
        
        // 4. Edge pixel upweighting for enhanced position sensitivity
        double edge_weight = 1.0;
        if (config.enable_edge_pixel_upweighting && edge_pixel_indices.count(i) > 0) {
            edge_weight = config.edge_pixel_boost_factor;
        }
        
        // 4. Spatial uncertainty and "horizontal errors": account for pixel integration area
        double spatial_weight = 1.0;
        if (config.enable_spatial_uncertainty) {
            // Pixels have inherent spatial uncertainty due to finite size
            // Add uncertainty proportional to pixel size - this is the "horizontal error"
            double spatial_gamma = config.spatial_uncertainty_factor * pixel_spacing;
            double total_gamma = std::sqrt(estimates.gamma * estimates.gamma + spatial_gamma * spatial_gamma);
            
            // Weight by inverse of total uncertainty
            if (total_gamma > 0) {
                spatial_weight = estimates.gamma / total_gamma;
            }
        }
        
        // 5. Enhanced horizontal error correction for pixel integration
        double horizontal_error_weight = 1.0;
        if (config.enable_horizontal_error_correction) {
            // Calculate horizontal error based on pixel position relative to estimated center
            double dx = std::abs(x_vals[i] - estimates.center);
            double pixel_edge_distance = pixel_spacing / 2.0;
            
            // Pixels closer to edges have larger horizontal uncertainty
            double horizontal_error = config.horizontal_error_scale * pixel_spacing;
            
            // If the pixel is at the edge relative to the fitted center, increase uncertainty
            if (dx > pixel_edge_distance) {
                horizontal_error *= (1.0 + (dx - pixel_edge_distance) / pixel_spacing);
            }
            
            // Weight inversely proportional to horizontal error
            horizontal_error_weight = 1.0 / (1.0 + horizontal_error / pixel_spacing);
        }
        
        // 5. Enhanced robust weighting based on expected Lorentzian profile
        double profile_weight = 1.0;
        if (config.enable_robust_profile_weighting && estimates.gamma > 0) {
            double dx = std::abs(x_vals[i] - estimates.center);
            double normalized_dx = dx / estimates.gamma;
            double expected_value = estimates.amplitude / (1.0 + normalized_dx * normalized_dx) + estimates.baseline;
            
            // Calculate normalized residual with robust scaling
            double residual = std::abs(y_vals[i] - expected_value);
            double expected_error = std::max(1.0, std::sqrt(expected_value)); // Poisson-like error model
            double normalized_residual = residual / expected_error;
            
            // Apply robust weighting: reduce influence of outliers (less aggressive for Lorentzian wide tails)
            profile_weight = 1.0 / (1.0 + 0.8 * normalized_residual * normalized_residual);
            profile_weight = std::max(0.1, profile_weight); // Keep reasonable minimum
        }
        
        // 6. Spatial error maps for position-dependent weighting (Lorentzian-specific)
        double spatial_error_weight = 1.0;
        if (config.enable_spatial_error_maps) {
            // Model systematic position-dependent reconstruction errors for Lorentzian distributions
            double dx = x_vals[i] - estimates.center;
            double pixel_position_error = 0.0;
            
            // Lorentzian has wider tails, so position error grows more slowly
            double normalized_distance = std::abs(dx) / (pixel_spacing / 2.0);
            if (normalized_distance < 1.0) {
                // Quadratic increase in error near pixel edges (less aggressive than Gaussian)
                pixel_position_error = config.position_dependent_bias_scale * 
                                     normalized_distance * normalized_distance * pixel_spacing * 0.8;
            } else {
                // Linear increase for pixels beyond immediate neighbors (gentler for Lorentzian)
                pixel_position_error = config.position_dependent_bias_scale * 
                                     (1.0 + (normalized_distance - 1.0) * 0.6) * pixel_spacing;
            }
            
            spatial_error_weight = 1.0 / (1.0 + config.spatial_error_map_strength * 
                                         pixel_position_error / pixel_spacing);
        }
        
        // 7. Inter-pixel correlation weighting (adapted for Lorentzian tails)
        double correlation_weight = 1.0;
        if (config.enable_correlation_weighting) {
            // Weight based on local charge distribution consistency
            double local_charge_variance = 0.0;
            double local_charge_sum = 0.0;
            int neighbor_count = 0;
            
            // Check neighboring pixels within correlation radius (larger for Lorentzian)
            for (size_t j = 0; j < y_vals.size(); ++j) {
                if (i == j) continue;
                double distance = std::abs(x_vals[j] - x_vals[i]) / pixel_spacing;
                if (distance <= config.correlation_radius) {
                    local_charge_sum += y_vals[j];
                    neighbor_count++;
                }
            }
            
            if (neighbor_count > 0) {
                double local_mean = local_charge_sum / neighbor_count;
                for (size_t j = 0; j < y_vals.size(); ++j) {
                    if (i == j) continue;
                    double distance = std::abs(x_vals[j] - x_vals[i]) / pixel_spacing;
                    if (distance <= config.correlation_radius) {
                        double deviation = y_vals[j] - local_mean;
                        local_charge_variance += deviation * deviation;
                    }
                }
                local_charge_variance /= neighbor_count;
                
                // For Lorentzian: be more tolerant of variance due to wider tails
                double variance_factor = 1.0 / (1.0 + 0.8 * local_charge_variance / (mean_charge * mean_charge));
                correlation_weight = 0.6 + 0.4 * variance_factor; // Range [0.6, 1.0] - more tolerant
            }
        }
        
        // Combine all weighting factors including horizontal error correction
        weight = charge_weight * distance_weight * spatial_weight * profile_weight * 
                edge_weight * horizontal_error_weight * spatial_error_weight * correlation_weight;
        
        // Ensure weight is reasonable and apply final normalization
        weights[i] = std::max(0.01, std::min(100.0, weight));
    }
    
    return weights;
} 

// Outlier filtering for Lorentzian fitting (adapted from Gaussian version)
std::pair<std::vector<double>, std::vector<double>> FilterLorentzianOutliers(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double sigma_threshold = 2.5,
    bool verbose = false) {
    
    std::vector<double> filtered_x, filtered_y;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return std::make_pair(filtered_x, filtered_y);
    }
    
    DataStatistics stats = CalculateRobustStatisticsLorentzian(x_vals, y_vals);
    if (!stats.valid) {
        return std::make_pair(x_vals, y_vals);
    }
    
    // Use MAD-based outlier detection
    double outlier_threshold = stats.median + sigma_threshold * stats.mad;
    double lower_threshold = stats.median - sigma_threshold * stats.mad;
    
    int outliers_removed = 0;
    for (size_t i = 0; i < y_vals.size(); ++i) {
        if (y_vals[i] >= lower_threshold && y_vals[i] <= outlier_threshold) {
            filtered_x.push_back(x_vals[i]);
            filtered_y.push_back(y_vals[i]);
        } else {
            outliers_removed++;
        }
    }
    
    // Use lenient filtering if too many outliers removed
    if (filtered_x.size() < x_vals.size() / 2) {
        if (verbose) {
            std::cout << "Too many Lorentzian outliers detected (" << outliers_removed 
                     << "), using lenient filtering" << std::endl;
        }
        
        filtered_x.clear();
        filtered_y.clear();
        
        double extreme_threshold = stats.median + 4.0 * stats.mad;
        double extreme_lower = stats.median - 4.0 * stats.mad;
        
        for (size_t i = 0; i < y_vals.size(); ++i) {
            if (y_vals[i] >= extreme_lower && y_vals[i] <= extreme_threshold) {
                filtered_x.push_back(x_vals[i]);
                filtered_y.push_back(y_vals[i]);
            }
        }
    }
    
    if (filtered_x.size() < 4) {
        if (verbose) {
            std::cout << "Warning: After Lorentzian outlier filtering, only " << filtered_x.size() 
                     << " points remain" << std::endl;
        }
        return std::make_pair(x_vals, y_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " Lorentzian outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_pair(filtered_x, filtered_y);
}

// Core Lorentzian fitting function using Ceres Solver
bool FitLorentzianCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    double& fit_amplitude,
    double& fit_center,
    double& fit_gamma,
    double& fit_vertical_offset,
    double& fit_amplitude_err,
    double& fit_center_err,
    double& fit_gamma_err,
    double& fit_vertical_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        if (verbose) {
            std::cout << "Insufficient data points for Lorentzian fitting" << std::endl;
        }
        return false;
    }
    
    // Multiple outlier filtering strategies
    std::vector<std::pair<std::vector<double>, std::vector<double>>> filtered_datasets;
    
    if (enable_outlier_filtering) {
        auto conservative_data = FilterLorentzianOutliers(x_vals, y_vals, 2.5, verbose);
        if (conservative_data.first.size() >= 4) {
            filtered_datasets.push_back(conservative_data);
        }
        
        auto lenient_data = FilterLorentzianOutliers(x_vals, y_vals, 3.0, verbose);
        if (lenient_data.first.size() >= 4) {
            filtered_datasets.push_back(lenient_data);
        }
    }
    
    // Always include original data as fallback
    filtered_datasets.push_back(std::make_pair(x_vals, y_vals));
    
    if (verbose) {
        std::cout << "Lorentzian outlier filtering " << (enable_outlier_filtering ? "enabled" : "disabled") 
                 << ", testing " << filtered_datasets.size() << " datasets" << std::endl;
    }
    
    // Try each filtered dataset
    for (size_t dataset_idx = 0; dataset_idx < filtered_datasets.size(); ++dataset_idx) {
        std::vector<double> clean_x = filtered_datasets[dataset_idx].first;
        std::vector<double> clean_y = filtered_datasets[dataset_idx].second;
        
        if (clean_x.size() < 4) continue;
        
        if (verbose) {
            std::cout << "Trying Lorentzian dataset " << dataset_idx << " with " << clean_x.size() << " points" << std::endl;
        }
        
        // Get parameter estimates
        LorentzianParameterEstimates estimates = EstimateLorentzianParameters(clean_x, clean_y, center_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "Lorentzian parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calculate data weights
        LorentzianWeightingConfig config;
        std::vector<double> data_weights = CalculateLorentzianDataWeights(clean_x, clean_y, estimates, pixel_spacing, config);
        
        // Multiple fitting configurations
        struct LorentzianFittingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        std::vector<LorentzianFittingConfig> configs;
        
        LorentzianFittingConfig config1;
        config1.linear_solver = ceres::DENSE_QR;
        config1.trust_region = ceres::LEVENBERG_MARQUARDT;
        config1.function_tolerance = 1e-15;
        config1.gradient_tolerance = 1e-15;
        config1.max_iterations = 2000;
        config1.loss_function = "HUBER";
        config1.loss_parameter = estimates.amplitude * config.robust_threshold_factor;
        configs.push_back(config1);
        
        LorentzianFittingConfig config2;
        config2.linear_solver = ceres::DENSE_QR;
        config2.trust_region = ceres::LEVENBERG_MARQUARDT;
        config2.function_tolerance = 1e-12;
        config2.gradient_tolerance = 1e-12;
        config2.max_iterations = 1500;
        config2.loss_function = "CAUCHY";
        config2.loss_parameter = estimates.amplitude * config.robust_threshold_factor * 1.6;
        configs.push_back(config2);
        
        LorentzianFittingConfig config3;
        config3.linear_solver = ceres::DENSE_QR;
        config3.trust_region = ceres::DOGLEG;
        config3.function_tolerance = 1e-10;
        config3.gradient_tolerance = 1e-10;
        config3.max_iterations = 1000;
        config3.loss_function = "NONE";
        config3.loss_parameter = 0.0;
        configs.push_back(config3);
        
        LorentzianFittingConfig config4;
        config4.linear_solver = ceres::DENSE_NORMAL_CHOLESKY;
        config4.trust_region = ceres::LEVENBERG_MARQUARDT;
        config4.function_tolerance = 1e-12;
        config4.gradient_tolerance = 1e-12;
        config4.max_iterations = 1500;
        config4.loss_function = "HUBER";
        config4.loss_parameter = estimates.amplitude * config.robust_threshold_factor * 1.3;
        configs.push_back(config4);
        
        LorentzianFittingConfig config5;
        config5.linear_solver = ceres::SPARSE_NORMAL_CHOLESKY;
        config5.trust_region = ceres::LEVENBERG_MARQUARDT;
        config5.function_tolerance = 1e-12;
        config5.gradient_tolerance = 1e-12;
        config5.max_iterations = 1200;
        config5.loss_function = "CAUCHY";
        config5.loss_parameter = estimates.amplitude * config.robust_threshold_factor * 2.2;
        configs.push_back(config5);
        
        for (const auto& config : configs) {
            // Set up parameter array
            double parameters[4];
            parameters[0] = estimates.amplitude;
            parameters[1] = estimates.center;
            parameters[2] = estimates.gamma;
            parameters[3] = estimates.baseline;
            
            // Build the problem
            ceres::Problem problem;
            
            // Add residual blocks with enhanced horizontal error handling
            LorentzianWeightingConfig weight_config;
            for (size_t i = 0; i < clean_x.size(); ++i) {
                ceres::CostFunction* cost_function;
                
                // Technique #4: Choose between pixel integration and point evaluation
                if (weight_config.enable_pixel_integration) {
                    // Calculate pixel-specific horizontal uncertainty for this pixel
                    double horizontal_uncertainty = 0.0;
                    if (weight_config.enable_horizontal_error_correction) {
                        // Base horizontal error from pixel size
                        horizontal_uncertainty = weight_config.horizontal_error_scale * pixel_spacing;
                        
                        // Enhanced pixel-specific horizontal error calculation for Lorentzian distributions
                        double dx = std::abs(clean_x[i] - estimates.center);
                        double pixel_edge_distance = pixel_spacing / 2.0;
                        
                        // Lorentzian has wider tails, so adjust uncertainty more gradually
                        if (dx > pixel_edge_distance) {
                            double edge_factor = 1.0 + 0.8 * (dx - pixel_edge_distance) / pixel_spacing;
                            horizontal_uncertainty *= edge_factor;
                        }
                        
                        // Charge-dependent horizontal uncertainty (adapted for Lorentzian)
                        if (weight_config.enable_charge_weighted_uncertainty) {
                            DataStatistics stats = CalculateRobustStatisticsLorentzian(clean_x, clean_y);
                            double charge_factor = 1.0 / std::sqrt(std::max(0.1, clean_y[i] / stats.mean));
                            horizontal_uncertainty *= charge_factor * 0.9; // Slightly less aggressive for Lorentzian
                        }
                        
                        // Position-dependent horizontal error from spatial error maps (adapted for wider tails)
                        if (weight_config.enable_spatial_error_maps) {
                            double position_error = weight_config.position_dependent_bias_scale * 
                                                   std::abs(dx) / pixel_spacing * pixel_spacing * 0.8; // Gentler for Lorentzian
                            horizontal_uncertainty += position_error;
                        }
                    }
                    
                    cost_function = PixelIntegratedLorentzianCostFunction::Create(
                        clean_x[i], clean_y[i], pixel_spacing, data_weights[i], horizontal_uncertainty);
                } else {
                    cost_function = LorentzianCostFunction::Create(
                        clean_x[i], clean_y[i], data_weights[i]);
                }
                
                // Technique #3: Enhanced robust loss functions with dynamic switching for Lorentzian
                ceres::LossFunction* loss_function = nullptr;
                if (weight_config.enable_robust_losses) {
                    if (config.loss_function == "HUBER") {
                        loss_function = new ceres::HuberLoss(config.loss_parameter);
                    } else if (config.loss_function == "CAUCHY") {
                        loss_function = new ceres::CauchyLoss(config.loss_parameter);
                    }
                    
                    // Technique: Dynamic loss switching optimized for Lorentzian distributions
                    if (weight_config.enable_dynamic_loss_switching) {
                        // For central high-charge pixels, use robust loss but less aggressively than Gaussian
                        size_t max_charge_idx = std::distance(clean_y.begin(), 
                            std::max_element(clean_y.begin(), clean_y.end()));
                        
                        DataStatistics stats = CalculateRobustStatisticsLorentzian(clean_x, clean_y);
                        if (i == max_charge_idx && clean_y[i] > stats.mean * weight_config.central_downweight_threshold) {
                            // Use moderately aggressive robust loss for central pixel (Lorentzian has natural wide tails)
                            double stronger_parameter = config.loss_parameter * 0.7; // Less aggressive than Gaussian
                            if (config.loss_function == "HUBER") {
                                delete loss_function;
                                loss_function = new ceres::HuberLoss(stronger_parameter);
                            } else if (config.loss_function == "CAUCHY") {
                                delete loss_function;
                                loss_function = new ceres::CauchyLoss(stronger_parameter);
                            }
                        }
                    }
                }
                
                problem.AddResidualBlock(cost_function, loss_function, parameters);
            }
            
            // Set bounds
            double amp_min = std::max(1e-20, estimates.amplitude * 0.01);
            double amp_max = estimates.amplitude * 100.0;
            problem.SetParameterLowerBound(parameters, 0, amp_min);
            problem.SetParameterUpperBound(parameters, 0, amp_max);
            
            double center_range = pixel_spacing * 3.0;
            problem.SetParameterLowerBound(parameters, 1, estimates.center - center_range);
            problem.SetParameterUpperBound(parameters, 1, estimates.center + center_range);
            
            problem.SetParameterLowerBound(parameters, 2, pixel_spacing * 0.1);
            problem.SetParameterUpperBound(parameters, 2, pixel_spacing * 5.0);
            
            double baseline_range = std::max(estimates.amplitude * 0.5, std::abs(estimates.baseline) * 2.0);
            problem.SetParameterLowerBound(parameters, 3, estimates.baseline - baseline_range);
            problem.SetParameterUpperBound(parameters, 3, estimates.baseline + baseline_range);
            
            // Configure solver
            ceres::Solver::Options options;
            options.linear_solver_type = config.linear_solver;
            options.minimizer_type = ceres::TRUST_REGION;
            options.trust_region_strategy_type = config.trust_region;
            options.function_tolerance = config.function_tolerance;
            options.gradient_tolerance = config.gradient_tolerance;
            options.parameter_tolerance = 1e-15;
            options.max_num_iterations = config.max_iterations;
            options.max_num_consecutive_invalid_steps = 50;
            options.use_nonmonotonic_steps = true;
            options.minimizer_progress_to_stdout = false;
            
            // Solve
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            // Validate results
            bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                                  summary.termination_type == ceres::USER_SUCCESS) &&
                                 parameters[0] > 0 && parameters[2] > 0 &&
                                 !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                 !std::isnan(parameters[2]) && !std::isnan(parameters[3]);
            
            if (fit_successful) {
                // Extract results
                fit_amplitude = parameters[0];
                fit_center = parameters[1];
                fit_gamma = std::abs(parameters[2]);
                fit_vertical_offset = parameters[3];
                
                // Calculate uncertainties
                bool cov_success = false;
                
                std::vector<std::pair<ceres::CovarianceAlgorithmType, double>> cov_configs = {
                    {ceres::DENSE_SVD, 1e-14},
                    {ceres::DENSE_SVD, 1e-12},
                    {ceres::DENSE_SVD, 1e-10},
                    {ceres::SPARSE_QR, 1e-12}
                };
                
                for (const auto& cov_config : cov_configs) {
                    ceres::Covariance::Options cov_options;
                    cov_options.algorithm_type = cov_config.first;
                    cov_options.min_reciprocal_condition_number = cov_config.second;
                    cov_options.null_space_rank = 2;
                    cov_options.apply_loss_function = true;
                    
                    ceres::Covariance covariance(cov_options);
                    std::vector<std::pair<const double*, const double*>> covariance_blocks;
                    covariance_blocks.push_back(std::make_pair(parameters, parameters));
                    
                    if (covariance.Compute(covariance_blocks, &problem)) {
                        double covariance_matrix[16];
                        if (covariance.GetCovarianceBlock(parameters, parameters, covariance_matrix)) {
                            fit_amplitude_err = std::sqrt(std::abs(covariance_matrix[0]));
                            fit_center_err = std::sqrt(std::abs(covariance_matrix[5]));
                            fit_gamma_err = std::sqrt(std::abs(covariance_matrix[10]));
                            fit_vertical_offset_err = std::sqrt(std::abs(covariance_matrix[15]));
                            
                            if (!std::isnan(fit_amplitude_err) && !std::isnan(fit_center_err) &&
                                !std::isnan(fit_gamma_err) && !std::isnan(fit_vertical_offset_err) &&
                                fit_amplitude_err < 10.0 * fit_amplitude &&
                                fit_center_err < 5.0 * pixel_spacing) {
                                cov_success = true;
                                break;
                            }
                        }
                    }
                }
                
                // Fallback uncertainty estimation
                if (!cov_success) {
                    DataStatistics data_stats = CalculateRobustStatisticsLorentzian(clean_x, clean_y);
                    fit_amplitude_err = std::max(0.02 * fit_amplitude, 0.1 * data_stats.mad);
                    fit_center_err = std::max(0.02 * pixel_spacing, fit_gamma / 10.0);
                    fit_gamma_err = std::max(0.05 * fit_gamma, 0.01 * pixel_spacing);
                    fit_vertical_offset_err = std::max(0.1 * std::abs(fit_vertical_offset), 0.05 * data_stats.mad);
                }
                
                // Calculate reduced chi-squared
                double chi2 = summary.final_cost;
                int dof = std::max(1, static_cast<int>(clean_x.size()) - 4);
                chi2_reduced = chi2 / dof;
                
                if (verbose) {
                    std::cout << "Successful Lorentzian fit with config " << &config - &configs[0] 
                             << ", dataset " << dataset_idx 
                             << ": A=" << fit_amplitude << "±" << fit_amplitude_err
                             << ", m=" << fit_center << "±" << fit_center_err
                             << ", gamma=" << fit_gamma << "±" << fit_gamma_err
                             << ", B=" << fit_vertical_offset << "±" << fit_vertical_offset_err
                             << ", chi2red=" << chi2_reduced << std::endl;
                }
                
                return true;
            } else if (verbose) {
                std::cout << "Lorentzian fit failed with config " << &config - &configs[0] 
                         << ": " << summary.BriefReport() << std::endl;
            }
        }
    }
    
    if (verbose) {
        std::cout << "All Lorentzian fitting strategies failed" << std::endl;
    }
    return false;
} 

LorentzianFit2DResultsCeres Fit2DLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    LorentzianFit2DResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeresLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit2DLorentzianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Fit2DLorentzianCeres: Error - need at least 4 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 2D Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // Create maps to group data by rows and columns
    std::map<double, std::vector<std::pair<double, double>>> rows_data; // y -> [(x, charge), ...]
    std::map<double, std::vector<std::pair<double, double>>> cols_data; // x -> [(y, charge), ...]
    
    // Group data points by rows and columns (within pixel spacing tolerance)
    const double tolerance = pixel_spacing * 0.1; // 10% tolerance for grouping
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x = x_coords[i];
        double y = y_coords[i];
        double charge = charge_values[i];
        
        if (charge <= 0) continue; // Skip non-positive charges
        
        // Find or create row
        bool found_row = false;
        for (auto& row_pair : rows_data) {
            if (std::abs(row_pair.first - y) < tolerance) {
                row_pair.second.push_back(std::make_pair(x, charge));
                found_row = true;
                break;
            }
        }
        if (!found_row) {
            rows_data[y].push_back(std::make_pair(x, charge));
        }
        
        // Find or create column
        bool found_col = false;
        for (auto& col_pair : cols_data) {
            if (std::abs(col_pair.first - x) < tolerance) {
                col_pair.second.push_back(std::make_pair(y, charge));
                found_col = true;
                break;
            }
        }
        if (!found_col) {
            cols_data[x].push_back(std::make_pair(y, charge));
        }
    }
    
    // Find the row and column closest to the center estimates
    double best_row_y = center_y_estimate;
    double min_row_dist = std::numeric_limits<double>::max();
    for (const auto& row_pair : rows_data) {
        double dist = std::abs(row_pair.first - center_y_estimate);
        if (dist < min_row_dist && row_pair.second.size() >= 4) {
            min_row_dist = dist;
            best_row_y = row_pair.first;
        }
    }
    
    double best_col_x = center_x_estimate;
    double min_col_dist = std::numeric_limits<double>::max();
    for (const auto& col_pair : cols_data) {
        double dist = std::abs(col_pair.first - center_x_estimate);
        if (dist < min_col_dist && col_pair.second.size() >= 4) {
            min_col_dist = dist;
            best_col_x = col_pair.first;
        }
    }
    
    bool x_fit_success = false;
    bool y_fit_success = false;
    
    // Fit X direction (central row)
    if (rows_data.find(best_row_y) != rows_data.end() && rows_data[best_row_y].size() >= 4) {
        auto& row_data = rows_data[best_row_y];
        
        // Sort by X coordinate
        std::sort(row_data.begin(), row_data.end());
        
        // Create vectors for fitting
        std::vector<double> x_vals, y_vals;
        std::vector<double> row_x_coords, row_y_coords;
        for (const auto& point : row_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
            row_x_coords.push_back(point.first);
            row_y_coords.push_back(best_row_y);  // Y coordinate is constant for row
        }
        
        if (verbose) {
            std::cout << "Fitting Lorentzian X direction with " << x_vals.size() << " points" << std::endl;
        }
        
        x_fit_success = FitLorentzianCeres(
            x_vals, y_vals, center_x_estimate, pixel_spacing,
            result.x_amplitude, result.x_center, result.x_gamma, result.x_vertical_offset,
            result.x_amplitude_err, result.x_center_err, result.x_gamma_err, result.x_vertical_offset_err,
            result.x_chi2red, verbose, enable_outlier_filtering);
        
        // Calculate DOF and p-value
        result.x_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.x_pp = (result.x_chi2red > 0) ? 1.0 - std::min(1.0, result.x_chi2red / 10.0) : 0.0;
        
        // Calculate 3x3 charge errors for central row
        std::vector<double> x_row_errors = Calculate3x3ChargeErrors(
            row_x_coords, row_y_coords, y_vals,
            x_coords, y_coords, charge_values,
            pixel_spacing, verbose);
        
        // Store data for ROOT analysis
        result.x_row_pixel_coords = x_vals;
        result.x_row_charge_values = y_vals;
        result.x_row_charge_errors = x_row_errors;
    }
    
    // Fit Y direction (central column)
    if (cols_data.find(best_col_x) != cols_data.end() && cols_data[best_col_x].size() >= 4) {
        auto& col_data = cols_data[best_col_x];
        
        // Sort by Y coordinate
        std::sort(col_data.begin(), col_data.end());
        
        // Create vectors for fitting
        std::vector<double> x_vals, y_vals;
        std::vector<double> col_x_coords, col_y_coords;
        for (const auto& point : col_data) {
            x_vals.push_back(point.first); // Y coordinate
            y_vals.push_back(point.second); // charge
            col_x_coords.push_back(best_col_x);  // X coordinate is constant for column
            col_y_coords.push_back(point.first); // Y coordinate
        }
        
        if (verbose) {
            std::cout << "Fitting Lorentzian Y direction with " << x_vals.size() << " points" << std::endl;
        }
        
        y_fit_success = FitLorentzianCeres(
            x_vals, y_vals, center_y_estimate, pixel_spacing,
            result.y_amplitude, result.y_center, result.y_gamma, result.y_vertical_offset,
            result.y_amplitude_err, result.y_center_err, result.y_gamma_err, result.y_vertical_offset_err,
            result.y_chi2red, verbose, enable_outlier_filtering);
        
        // Calculate DOF and p-value
        result.y_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.y_pp = (result.y_chi2red > 0) ? 1.0 - std::min(1.0, result.y_chi2red / 10.0) : 0.0;
        
        // Calculate 3x3 charge errors for central column
        std::vector<double> y_col_errors = Calculate3x3ChargeErrors(
            col_x_coords, col_y_coords, y_vals,
            x_coords, y_coords, charge_values,
            pixel_spacing, verbose);
        
        // Store data for ROOT analysis
        result.y_col_pixel_coords = x_vals;  // Y coordinates
        result.y_col_charge_values = y_vals;
        result.y_col_charge_errors = y_col_errors;
    }
    
    // Set overall success status
    result.fit_successful = x_fit_success && y_fit_success;
    
    if (verbose) {
        std::cout << "2D Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

DiagonalLorentzianFitResultsCeres FitDiagonalLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    DiagonalLorentzianFitResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeresLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size() || x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Diagonal Lorentzian fit (Ceres): Invalid input data size" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting Diagonal Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // Tolerance for grouping pixels into diagonals
    double tolerance = pixel_spacing * 0.1;
    
    // Group pixels by diagonal lines
    std::map<double, std::vector<std::pair<double, double>>> main_diagonal_data;
    std::map<double, std::vector<std::pair<double, double>>> sec_diagonal_data;
    
    // Group data points by diagonal
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x = x_coords[i];
        double y = y_coords[i];
        double charge = charge_values[i];
        
        if (charge <= 0) continue; // Skip non-positive charges
        
        // Main diagonal: constant value of (x - y)
        double main_diag_id = x - y;
        
        // Secondary diagonal: constant value of (x + y)  
        double sec_diag_id = x + y;
        
        // Find or create main diagonal - store both x and y coordinates with charge
        bool found_main_diag = false;
        for (auto& diag_pair : main_diagonal_data) {
            if (std::abs(diag_pair.first - main_diag_id) < tolerance) {
                diag_pair.second.push_back(std::make_pair(x, charge));
                diag_pair.second.push_back(std::make_pair(y, charge));
                found_main_diag = true;
                break;
            }
        }
        if (!found_main_diag) {
            main_diagonal_data[main_diag_id].push_back(std::make_pair(x, charge));
            main_diagonal_data[main_diag_id].push_back(std::make_pair(y, charge));
        }
        
        // Find or create secondary diagonal - store both x and y coordinates with charge
        bool found_sec_diag = false;
        for (auto& diag_pair : sec_diagonal_data) {
            if (std::abs(diag_pair.first - sec_diag_id) < tolerance) {
                diag_pair.second.push_back(std::make_pair(x, charge));
                diag_pair.second.push_back(std::make_pair(y, charge));
                found_sec_diag = true;
                break;
            }
        }
        if (!found_sec_diag) {
            sec_diagonal_data[sec_diag_id].push_back(std::make_pair(x, charge));
            sec_diagonal_data[sec_diag_id].push_back(std::make_pair(y, charge));
        }
    }
    
    // Find the main diagonal closest to the center estimates
    double center_main_diag_id = center_x_estimate - center_y_estimate;
    double best_main_diag_id = center_main_diag_id;
    double min_main_diag_dist = std::numeric_limits<double>::max();
    for (const auto& diag_pair : main_diagonal_data) {
        double dist = std::abs(diag_pair.first - center_main_diag_id);
        if (dist < min_main_diag_dist && diag_pair.second.size() >= 8) {
            min_main_diag_dist = dist;
            best_main_diag_id = diag_pair.first;
        }
    }
    
    // Find the secondary diagonal closest to the center estimates
    double center_sec_diag_id = center_x_estimate + center_y_estimate;
    double best_sec_diag_id = center_sec_diag_id;
    double min_sec_diag_dist = std::numeric_limits<double>::max();
    for (const auto& diag_pair : sec_diagonal_data) {
        double dist = std::abs(diag_pair.first - center_sec_diag_id);
        if (dist < min_sec_diag_dist && diag_pair.second.size() >= 8) {
            min_sec_diag_dist = dist;
            best_sec_diag_id = diag_pair.first;
        }
    }
    
    bool main_diag_x_success = false;
    bool main_diag_y_success = false;
    bool sec_diag_x_success = false;
    bool sec_diag_y_success = false;
    
    // Fit main diagonal X direction
    if (main_diagonal_data.find(best_main_diag_id) != main_diagonal_data.end() && 
        main_diagonal_data[best_main_diag_id].size() >= 8) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Extract X coordinates and charges (every other pair starting from index 0)
        std::vector<double> x_vals, charge_vals;
        for (size_t i = 0; i < main_diag_data.size(); i += 2) {
            x_vals.push_back(main_diag_data[i].first);
            charge_vals.push_back(main_diag_data[i].second);
        }
        
        // Sort by X coordinate
        std::vector<std::pair<double, double>> x_charge_pairs;
        for (size_t i = 0; i < x_vals.size(); ++i) {
            x_charge_pairs.push_back(std::make_pair(x_vals[i], charge_vals[i]));
        }
        std::sort(x_charge_pairs.begin(), x_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> x_sorted, charge_x_sorted;
        for (const auto& pair : x_charge_pairs) {
            x_sorted.push_back(pair.first);
            charge_x_sorted.push_back(pair.second);
        }
        
        if (x_sorted.size() >= 4) {
            if (verbose) {
                std::cout << "Fitting main diagonal Lorentzian X with " << x_sorted.size() << " points" << std::endl;
            }
            
            main_diag_x_success = FitLorentzianCeres(
                x_sorted, charge_x_sorted, center_x_estimate, pixel_spacing,
                result.main_diag_x_amplitude, result.main_diag_x_center, result.main_diag_x_gamma, result.main_diag_x_vertical_offset,
                result.main_diag_x_amplitude_err, result.main_diag_x_center_err, result.main_diag_x_gamma_err, result.main_diag_x_vertical_offset_err,
                result.main_diag_x_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.main_diag_x_dof = std::max(1, static_cast<int>(x_sorted.size()) - 4);
            result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
            result.main_diag_x_fit_successful = main_diag_x_success;
            
            // Calculate 3x3 charge errors for main diagonal X
            // Create corresponding Y coordinates for each X coordinate on the main diagonal
            std::vector<double> main_diag_x_coords, main_diag_y_coords;
            for (size_t i = 0; i < x_sorted.size(); ++i) {
                main_diag_x_coords.push_back(x_sorted[i]);
                // For main diagonal: y = x - diagonal_offset
                main_diag_y_coords.push_back(x_sorted[i] - best_main_diag_id);
            }
            
            std::vector<double> main_diag_x_errors = Calculate3x3ChargeErrors(
                main_diag_x_coords, main_diag_y_coords, charge_x_sorted,
                x_coords, y_coords, charge_values,
                pixel_spacing, verbose);
            
            // Store data for ROOT analysis
            result.main_diag_x_pixel_coords = x_sorted;
            result.main_diag_x_charge_values = charge_x_sorted;
            result.main_diag_x_charge_errors = main_diag_x_errors;
        }
    }
    
    // Fit main diagonal Y direction
    if (main_diagonal_data.find(best_main_diag_id) != main_diagonal_data.end() && 
        main_diagonal_data[best_main_diag_id].size() >= 8) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Extract Y coordinates and charges (every other pair starting from index 1)
        std::vector<double> y_vals, charge_vals;
        for (size_t i = 1; i < main_diag_data.size(); i += 2) {
            y_vals.push_back(main_diag_data[i].first);
            charge_vals.push_back(main_diag_data[i].second);
        }
        
        // Sort by Y coordinate
        std::vector<std::pair<double, double>> y_charge_pairs;
        for (size_t i = 0; i < y_vals.size(); ++i) {
            y_charge_pairs.push_back(std::make_pair(y_vals[i], charge_vals[i]));
        }
        std::sort(y_charge_pairs.begin(), y_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> y_sorted, charge_y_sorted;
        for (const auto& pair : y_charge_pairs) {
            y_sorted.push_back(pair.first);
            charge_y_sorted.push_back(pair.second);
        }
        
        if (y_sorted.size() >= 4) {
            if (verbose) {
                std::cout << "Fitting main diagonal Lorentzian Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            main_diag_y_success = FitLorentzianCeres(
                y_sorted, charge_y_sorted, center_y_estimate, pixel_spacing,
                result.main_diag_y_amplitude, result.main_diag_y_center, result.main_diag_y_gamma, result.main_diag_y_vertical_offset,
                result.main_diag_y_amplitude_err, result.main_diag_y_center_err, result.main_diag_y_gamma_err, result.main_diag_y_vertical_offset_err,
                result.main_diag_y_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.main_diag_y_dof = std::max(1, static_cast<int>(y_sorted.size()) - 4);
            result.main_diag_y_pp = (result.main_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_y_chi2red / 10.0) : 0.0;
            result.main_diag_y_fit_successful = main_diag_y_success;
            
            // Calculate 3x3 charge errors for main diagonal Y
            // Create corresponding X coordinates for each Y coordinate on the main diagonal
            std::vector<double> main_diag_x_coords_y, main_diag_y_coords_y;
            for (size_t i = 0; i < y_sorted.size(); ++i) {
                // For main diagonal: x = y + diagonal_offset
                main_diag_x_coords_y.push_back(y_sorted[i] + best_main_diag_id);
                main_diag_y_coords_y.push_back(y_sorted[i]);
            }
            
            std::vector<double> main_diag_y_errors = Calculate3x3ChargeErrors(
                main_diag_x_coords_y, main_diag_y_coords_y, charge_y_sorted,
                x_coords, y_coords, charge_values,
                pixel_spacing, verbose);
            
            // Store data for ROOT analysis
            result.main_diag_y_pixel_coords = y_sorted;
            result.main_diag_y_charge_values = charge_y_sorted;
            result.main_diag_y_charge_errors = main_diag_y_errors;
        }
    }
    
    // Fit secondary diagonal X direction
    if (sec_diagonal_data.find(best_sec_diag_id) != sec_diagonal_data.end() && 
        sec_diagonal_data[best_sec_diag_id].size() >= 8) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Extract X coordinates and charges (every other pair starting from index 0)
        std::vector<double> x_vals, charge_vals;
        for (size_t i = 0; i < sec_diag_data.size(); i += 2) {
            x_vals.push_back(sec_diag_data[i].first);
            charge_vals.push_back(sec_diag_data[i].second);
        }
        
        // Sort by X coordinate
        std::vector<std::pair<double, double>> x_charge_pairs;
        for (size_t i = 0; i < x_vals.size(); ++i) {
            x_charge_pairs.push_back(std::make_pair(x_vals[i], charge_vals[i]));
        }
        std::sort(x_charge_pairs.begin(), x_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> x_sorted, charge_x_sorted;
        for (const auto& pair : x_charge_pairs) {
            x_sorted.push_back(pair.first);
            charge_x_sorted.push_back(pair.second);
        }
        
        if (x_sorted.size() >= 4) {
            if (verbose) {
                std::cout << "Fitting secondary diagonal Lorentzian X with " << x_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_x_success = FitLorentzianCeres(
                x_sorted, charge_x_sorted, center_x_estimate, pixel_spacing,
                result.sec_diag_x_amplitude, result.sec_diag_x_center, result.sec_diag_x_gamma, result.sec_diag_x_vertical_offset,
                result.sec_diag_x_amplitude_err, result.sec_diag_x_center_err, result.sec_diag_x_gamma_err, result.sec_diag_x_vertical_offset_err,
                result.sec_diag_x_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.sec_diag_x_dof = std::max(1, static_cast<int>(x_sorted.size()) - 4);
            result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
            result.sec_diag_x_fit_successful = sec_diag_x_success;
            
            // Calculate 3x3 charge errors for secondary diagonal X
            // Create corresponding Y coordinates for each X coordinate on the secondary diagonal
            std::vector<double> sec_diag_x_coords, sec_diag_y_coords;
            for (size_t i = 0; i < x_sorted.size(); ++i) {
                sec_diag_x_coords.push_back(x_sorted[i]);
                // For secondary diagonal: y = -x + diagonal_offset
                sec_diag_y_coords.push_back(best_sec_diag_id - x_sorted[i]);
            }
            
            std::vector<double> sec_diag_x_errors = Calculate3x3ChargeErrors(
                sec_diag_x_coords, sec_diag_y_coords, charge_x_sorted,
                x_coords, y_coords, charge_values,
                pixel_spacing, verbose);
            
            // Store data for ROOT analysis
            result.sec_diag_x_pixel_coords = x_sorted;
            result.sec_diag_x_charge_values = charge_x_sorted;
            result.sec_diag_x_charge_errors = sec_diag_x_errors;
        }
    }
    
    // Fit secondary diagonal Y direction
    if (sec_diagonal_data.find(best_sec_diag_id) != sec_diagonal_data.end() && 
        sec_diagonal_data[best_sec_diag_id].size() >= 8) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Extract Y coordinates and charges (every other pair starting from index 1)
        std::vector<double> y_vals, charge_vals;
        for (size_t i = 1; i < sec_diag_data.size(); i += 2) {
            y_vals.push_back(sec_diag_data[i].first);
            charge_vals.push_back(sec_diag_data[i].second);
        }
        
        // Sort by Y coordinate
        std::vector<std::pair<double, double>> y_charge_pairs;
        for (size_t i = 0; i < y_vals.size(); ++i) {
            y_charge_pairs.push_back(std::make_pair(y_vals[i], charge_vals[i]));
        }
        std::sort(y_charge_pairs.begin(), y_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> y_sorted, charge_y_sorted;
        for (const auto& pair : y_charge_pairs) {
            y_sorted.push_back(pair.first);
            charge_y_sorted.push_back(pair.second);
        }
        
        if (y_sorted.size() >= 4) {
            if (verbose) {
                std::cout << "Fitting secondary diagonal Lorentzian Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_y_success = FitLorentzianCeres(
                y_sorted, charge_y_sorted, center_y_estimate, pixel_spacing,
                result.sec_diag_y_amplitude, result.sec_diag_y_center, result.sec_diag_y_gamma, result.sec_diag_y_vertical_offset,
                result.sec_diag_y_amplitude_err, result.sec_diag_y_center_err, result.sec_diag_y_gamma_err, result.sec_diag_y_vertical_offset_err,
                result.sec_diag_y_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.sec_diag_y_dof = std::max(1, static_cast<int>(y_sorted.size()) - 4);
            result.sec_diag_y_pp = (result.sec_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_y_chi2red / 10.0) : 0.0;
            result.sec_diag_y_fit_successful = sec_diag_y_success;
            
            // Calculate 3x3 charge errors for secondary diagonal Y
            // Create corresponding X coordinates for each Y coordinate on the secondary diagonal
            std::vector<double> sec_diag_x_coords_y, sec_diag_y_coords_y;
            for (size_t i = 0; i < y_sorted.size(); ++i) {
                // For secondary diagonal: x = -y + diagonal_offset
                sec_diag_x_coords_y.push_back(best_sec_diag_id - y_sorted[i]);
                sec_diag_y_coords_y.push_back(y_sorted[i]);
            }
            
            std::vector<double> sec_diag_y_errors = Calculate3x3ChargeErrors(
                sec_diag_x_coords_y, sec_diag_y_coords_y, charge_y_sorted,
                x_coords, y_coords, charge_values,
                pixel_spacing, verbose);
            
            // Store data for ROOT analysis
            result.sec_diag_y_pixel_coords = y_sorted;
            result.sec_diag_y_charge_values = charge_y_sorted;
            result.sec_diag_y_charge_errors = sec_diag_y_errors;
        }
    }
    
    // Set overall success status
    result.fit_successful = main_diag_x_success && main_diag_y_success && sec_diag_x_success && sec_diag_y_success;
    
    if (verbose) {
        std::cout << "Diagonal Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (Main X: " << (main_diag_x_success ? "OK" : "FAIL") 
                 << ", Main Y: " << (main_diag_y_success ? "OK" : "FAIL")
                 << ", Sec X: " << (sec_diag_x_success ? "OK" : "FAIL") 
                 << ", Sec Y: " << (sec_diag_y_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
} 

// Standalone outlier removal function with boolean control for Lorentzian fitting
LorentzianOutlierRemovalResult RemoveLorentzianOutliers(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& charge_values,
    bool enable_outlier_removal,
    double sigma_threshold,
    bool verbose) {
    
    LorentzianOutlierRemovalResult result;
    result.outliers_removed = 0;
    result.filtering_applied = enable_outlier_removal;
    result.success = false;
    
    // Input validation
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.empty()) {
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Error - empty input vectors" << std::endl;
        }
        return result;
    }
    
    // If outlier removal is disabled, return original data
    if (!enable_outlier_removal) {
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.success = true;
        
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Outlier removal disabled, returning original data (" 
                     << x_coords.size() << " points)" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "RemoveLorentzianOutliers: Starting outlier removal on " << x_coords.size() 
                 << " points with sigma threshold " << sigma_threshold << std::endl;
    }
    
    // Calculate robust statistics for charge values
    DataStatistics stats = CalculateRobustStatisticsLorentzian(x_coords, charge_values);
    if (!stats.valid) {
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Failed to calculate statistics, returning original data" << std::endl;
        }
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.success = true;
        return result;
    }
    
    // Use MAD-based outlier detection (more robust than standard deviation)
    // For Lorentzian distributions, we might want to be more lenient with outlier detection
    // since Lorentzian has wider tails than Gaussian
    double outlier_threshold = stats.median + sigma_threshold * stats.mad;
    double lower_threshold = stats.median - sigma_threshold * stats.mad;
    
    // Filter outliers
    for (size_t i = 0; i < charge_values.size(); ++i) {
        // Keep points that are within the robust bounds and have positive charge
        if (charge_values[i] >= lower_threshold && 
            charge_values[i] <= outlier_threshold && 
            charge_values[i] > 0) {
            result.filtered_x_coords.push_back(x_coords[i]);
            result.filtered_y_coords.push_back(y_coords[i]);
            result.filtered_charge_values.push_back(charge_values[i]);
        } else {
            result.outliers_removed++;
        }
    }
    
    // If too many outliers were removed, use more lenient filtering
    // For Lorentzian, use even more lenient thresholds due to wider tails
    if (result.filtered_x_coords.size() < x_coords.size() / 2) {
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Too many outliers detected (" << result.outliers_removed 
                     << "), using lenient filtering (5-sigma for Lorentzian)" << std::endl;
        }
        
        result.filtered_x_coords.clear();
        result.filtered_y_coords.clear();
        result.filtered_charge_values.clear();
        result.outliers_removed = 0;
        
        // Use 5-sigma threshold for lenient filtering (more lenient than Gaussian due to wide tails)
        double lenient_threshold = stats.median + 5.0 * stats.mad;
        double lenient_lower = stats.median - 5.0 * stats.mad;
        
        for (size_t i = 0; i < charge_values.size(); ++i) {
            if (charge_values[i] >= lenient_lower && 
                charge_values[i] <= lenient_threshold && 
                charge_values[i] > 0) {
                result.filtered_x_coords.push_back(x_coords[i]);
                result.filtered_y_coords.push_back(y_coords[i]);
                result.filtered_charge_values.push_back(charge_values[i]);
            } else {
                result.outliers_removed++;
            }
        }
    }
    
    // Final validation - ensure we have enough points remaining
    if (result.filtered_x_coords.size() < 3) {
        if (verbose) {
            std::cout << "RemoveLorentzianOutliers: Warning - only " << result.filtered_x_coords.size() 
                     << " points remain after filtering, returning original data" << std::endl;
        }
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
    }
    
    result.success = true;
    
    if (verbose) {
        std::cout << "RemoveLorentzianOutliers: Successfully filtered data - removed " << result.outliers_removed 
                 << " outliers, " << result.filtered_x_coords.size() << " points remaining" << std::endl;
    }
    
    return result;
} 

// Function to calculate charge errors based on 3x3 neighborhood around each pixel
std::vector<double> Calculate3x3ChargeErrors(
    const std::vector<double>& line_x_coords,
    const std::vector<double>& line_y_coords,
    const std::vector<double>& line_charge_values,
    const std::vector<double>& all_x_coords,
    const std::vector<double>& all_y_coords,
    const std::vector<double>& all_charge_values,
    double pixel_spacing,
    bool verbose) {
    
    std::vector<double> charge_errors;
    
    // Input validation
    if (line_x_coords.size() != line_y_coords.size() || 
        line_x_coords.size() != line_charge_values.size() ||
        all_x_coords.size() != all_y_coords.size() ||
        all_x_coords.size() != all_charge_values.size()) {
        if (verbose) {
            std::cout << "Calculate3x3ChargeErrors: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return charge_errors;
    }
    
    if (line_x_coords.empty() || all_x_coords.empty()) {
        if (verbose) {
            std::cout << "Calculate3x3ChargeErrors: Error - empty input vectors" << std::endl;
        }
        return charge_errors;
    }
    
    if (verbose) {
        std::cout << "Calculate3x3ChargeErrors: Calculating errors for " << line_x_coords.size() 
                 << " pixels using " << all_x_coords.size() << " total pixels" << std::endl;
    }
    
    // For each pixel on the line, find its 3x3 neighborhood and calculate error
    for (size_t i = 0; i < line_x_coords.size(); ++i) {
        double center_x = line_x_coords[i];
        double center_y = line_y_coords[i];
        
        // Find all pixels within 3x3 neighborhood (1.5 pixel spacing radius)
        std::vector<double> neighborhood_charges;
        double search_radius = 1.5 * pixel_spacing;
        
        // Add the center pixel itself
        neighborhood_charges.push_back(line_charge_values[i]);
        
        // Search for neighboring pixels in the full dataset
        for (size_t j = 0; j < all_x_coords.size(); ++j) {
            double dx = std::abs(all_x_coords[j] - center_x);
            double dy = std::abs(all_y_coords[j] - center_y);
            
            // Skip the center pixel (already added)
            if (dx < pixel_spacing * 0.1 && dy < pixel_spacing * 0.1) {
                continue;
            }
            
            // Check if pixel is within 3x3 neighborhood
            if (dx <= search_radius && dy <= search_radius) {
                neighborhood_charges.push_back(all_charge_values[j]);
            }
        }
        
        // Calculate error as (q_max - q_min) / sqrt(12)
        double charge_error = 0.0;
        
        if (neighborhood_charges.size() >= 2) {
            double q_max = *std::max_element(neighborhood_charges.begin(), neighborhood_charges.end());
            double q_min = *std::min_element(neighborhood_charges.begin(), neighborhood_charges.end());
            
            // Calculate error: (q_max - q_min) / sqrt(12)
            charge_error = (q_max - q_min) / std::sqrt(12.0);
            
            if (verbose) {
                std::cout << "Pixel (" << center_x << ", " << center_y << "): " 
                         << neighborhood_charges.size() << " neighbors, "
                         << "q_min=" << q_min << ", q_max=" << q_max 
                         << ", error=" << charge_error << std::endl;
            }
        } else {
            // Fallback: use a fraction of the pixel's own charge as error
            charge_error = std::max(0.1 * line_charge_values[i], 1.0);
            
            if (verbose) {
                std::cout << "Pixel (" << center_x << ", " << center_y << "): insufficient neighbors (" 
                         << neighborhood_charges.size() << "), using fallback error=" << charge_error << std::endl;
            }
        }
        
        // Ensure error is positive and reasonable
        charge_error = std::max(charge_error, 0.01 * line_charge_values[i]);
        charge_error = std::min(charge_error, 10.0 * line_charge_values[i]);
        
        charge_errors.push_back(charge_error);
    }
    
    if (verbose) {
        std::cout << "Calculate3x3ChargeErrors: Calculated " << charge_errors.size() << " charge errors" << std::endl;
    }
    
    return charge_errors;
} 