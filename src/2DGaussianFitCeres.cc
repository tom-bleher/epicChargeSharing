#include "2DGaussianFitCeres.hh"
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
static std::mutex gCeresFitMutex;
static std::atomic<int> gGlobalCeresFitCounter{0};

// Use shared Google logging initialization
void InitializeCeres() {
    CeresLoggingInitializer::InitializeOnce();
}

// Enhanced cost function with pixel integration for more accurate modeling with horizontal errors
struct PixelIntegratedGaussianCostFunction {
    PixelIntegratedGaussianCostFunction(double pixel_center_x, double pixel_y, double pixel_size, 
                                      double weight = 1.0, double horizontal_uncertainty = 0.0) 
        : pixel_center_x_(pixel_center_x), pixel_y_(pixel_y), pixel_size_(pixel_size), 
          weight_(weight), horizontal_uncertainty_(horizontal_uncertainty) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = m (center)
        // params[2] = sigma (width)
        // params[3] = B (offset)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& sigma = params[2];
        const T& B = params[3];
        
        // Robust handling of sigma
        T safe_sigma = ceres::abs(sigma);
        if (safe_sigma < T(1e-12)) {
            safe_sigma = T(1e-12);
        }
        
        // Include horizontal uncertainty in effective sigma for pixel integration
        // This models the fact that the true hit position within the pixel is uncertain
        T effective_sigma = safe_sigma;
        if (horizontal_uncertainty_ > 0) {
            T horizontal_sigma = T(horizontal_uncertainty_);
            // Combine fit sigma with horizontal positional uncertainty in quadrature
            effective_sigma = ceres::sqrt(safe_sigma * safe_sigma + horizontal_sigma * horizontal_sigma);
        }
        
        // Pixel integration using analytic formula for Gaussian integral with effective sigma
        // ∫[x-Δx/2 to x+Δx/2] A*exp(-(t-m)²/(2σ_eff²)) dt + B*Δx
        
        T half_pixel = T(pixel_size_) / T(2.0);
        T x_left = T(pixel_center_x_) - half_pixel;
        T x_right = T(pixel_center_x_) + half_pixel;
        
        // Use error function for analytic integration with effective sigma
        T sqrt_2_sigma = T(1.41421356237) * effective_sigma;
        T erf_left = ceres::erf((x_left - m) / sqrt_2_sigma);
        T erf_right = ceres::erf((x_right - m) / sqrt_2_sigma);
        
        // Integrated Gaussian: A * σ_eff * √(2π) * [erf(right) - erf(left)] / 2 + B * pixel_size
        T sqrt_2pi_sigma = T(2.50662827463) * effective_sigma;
        T integrated_gaussian = A * sqrt_2pi_sigma * (erf_right - erf_left) / T(2.0);
        T baseline_contribution = B * T(pixel_size_);
        
        T predicted = integrated_gaussian + baseline_contribution;
        
        // Apply additional horizontal error weighting to residual
        T effective_weight = T(weight_);
        if (horizontal_uncertainty_ > 0) {
            // Reduce weight for pixels with larger horizontal uncertainty
            // This implements the "horizontal error" weighting described in the request
            T uncertainty_factor = T(1.0) + T(horizontal_uncertainty_) / T(pixel_size_);
            effective_weight /= uncertainty_factor;
        }
        
        // Weighted residual with horizontal error correction
        residual[0] = effective_weight * (predicted - T(pixel_y_));
        
        return true;
    }
    
    static ceres::CostFunction* Create(double pixel_center_x, double pixel_y, double pixel_size, 
                                     double weight = 1.0, double horizontal_uncertainty = 0.0) {
        return (new ceres::AutoDiffCostFunction<PixelIntegratedGaussianCostFunction, 1, 4>(
            new PixelIntegratedGaussianCostFunction(pixel_center_x, pixel_y, pixel_size, weight, horizontal_uncertainty)));
    }
    
private:
    const double pixel_center_x_;
    const double pixel_y_;
    const double pixel_size_;
    const double weight_;
    const double horizontal_uncertainty_;
};

// Enhanced cost function for Gaussian fitting with better numerical stability
struct GaussianCostFunction {
    GaussianCostFunction(double x, double y, double weight = 1.0) 
        : x_(x), y_(y), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = m (center)
        // params[2] = sigma (width)
        // params[3] = B (offset)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& sigma = params[2];
        const T& B = params[3];
        
        // Robust handling of sigma
        T safe_sigma = ceres::abs(sigma);
        if (safe_sigma < T(1e-12)) {
            safe_sigma = T(1e-12);
        }
        
        // Gaussian function with robust exponent calculation
        T dx = x_ - m;
        T exponent = -(dx * dx) / (T(2.0) * safe_sigma * safe_sigma);
        
        // Prevent numerical overflow/underflow with robust bounds
        if (exponent < T(-200.0)) {
            exponent = T(-200.0);
        } else if (exponent > T(200.0)) {
            exponent = T(200.0);
        }
        
        T predicted = A * ceres::exp(exponent) + B;
        
        // Weighted residual for robust fitting
        residual[0] = T(weight_) * (predicted - T(y_));
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<GaussianCostFunction, 1, 4>(
            new GaussianCostFunction(x, y, weight)));
    }
    
private:
    const double x_;
    const double y_;
    const double weight_;
};

// Advanced parameter estimation with physics-based initialization
struct ParameterEstimates {
    double amplitude;
    double center;
    double sigma;
    double offset;
    double amplitude_err;
    double center_err;
    double sigma_err;
    double offset_err;
    bool valid;
    int method_used; // Track which estimation method was successful
};

// Robust statistics calculations
struct DataStatistics {
    double mean;
    double median;
    double std_dev;
    double mad; // Median Absolute Deviation
    double q25, q75; // Quartiles
    double min_val, max_val;
    double weighted_mean;
    double total_weight;
    double robust_center; // Improved center estimate
    bool valid;
};

DataStatistics CalculateRobustStatistics(const std::vector<double>& x_vals, 
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
    
    // Fast median-of-absolute-deviations using nth_element (O(n))
    std::vector<double> abs_deviations;
    abs_deviations.reserve(y_vals.size());
    for (double val : y_vals) {
        abs_deviations.push_back(std::abs(val - stats.median));
    }

    std::nth_element(abs_deviations.begin(), abs_deviations.begin() + n/2, abs_deviations.end());
    double mad_raw = abs_deviations[n/2];

    // If even number of elements, average the two central values for better robustness
    if (n % 2 == 0) {
        double second_val;
        std::nth_element(abs_deviations.begin(), abs_deviations.begin() + n/2 - 1, abs_deviations.end());
        second_val = abs_deviations[n/2 - 1];
        mad_raw = 0.5 * (mad_raw + second_val);
    }

    stats.mad = mad_raw * 1.4826; // Consistency factor for normal distribution
    
    // ------------------------------------------------------------------------------------
    // Numerical stability safeguard: ensure MAD is positive and finite
    // A zero or non-finite MAD leads to divide-by-zero or NaN weights which break the fit.
    // If MAD is too small (or not finite), fall back to the standard deviation or
    // a small positive epsilon value.
    // ------------------------------------------------------------------------------------
    if (!std::isfinite(stats.mad) || stats.mad < 1e-12) {
        stats.mad = (std::isfinite(stats.std_dev) && stats.std_dev > 1e-12) ?
                    stats.std_dev : 1e-12;
    }
    
    // Weighted statistics (weight by charge value for position estimation)
    stats.weighted_mean = 0.0;
    stats.total_weight = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - stats.q25); // Use values above Q1 as weights
        if (weight > 0) {
            stats.weighted_mean += x_vals[i] * weight;
            stats.total_weight += weight;
        }
    }
    
    if (stats.total_weight > 0) {
        stats.weighted_mean /= stats.total_weight;
        stats.robust_center = stats.weighted_mean;
    } else {
        stats.weighted_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
        stats.robust_center = stats.weighted_mean;
    }
    
    stats.valid = true;
    return stats;
}

// Multiple parameter estimation strategies
ParameterEstimates EstimateGaussianParameters(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    ParameterEstimates estimates;
    estimates.valid = false;
    estimates.method_used = 0;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 3) {
        return estimates;
    }
    
    // Calculate robust statistics
    DataStatistics stats = CalculateRobustStatistics(x_vals, y_vals);
    if (!stats.valid) {
        return estimates;
    }
    
    if (verbose) {
        std::cout << "Data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", MAD=" << stats.mad 
                 << ", weighted_mean=" << stats.weighted_mean << std::endl;
    }
    
    // Method 1: Physics-based estimation for charge distributions
    // For LGAD charge sharing, the peak should be near the weighted centroid
    estimates.center = stats.weighted_mean;
    estimates.offset = std::min(stats.min_val, stats.q25); // Use conservative offset
    estimates.amplitude = stats.max_val - estimates.offset;
    
    // Physics-based sigma estimation: charge spread should be related to pixel spacing
    // For LGAD detectors, typical charge spread is 0.3-0.8 pixel spacings
    double distance_spread = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - estimates.offset);
        if (weight > 0.1 * estimates.amplitude) { // Only use significant charges
            double dx = x_vals[i] - estimates.center;
            distance_spread += weight * dx * dx;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0) {
        estimates.sigma = std::sqrt(distance_spread / weight_sum);
    } else {
        estimates.sigma = pixel_spacing * 0.5; // Default fallback
    }
    
    // Apply physics-based bounds
    estimates.sigma = std::max(pixel_spacing * 0.2, std::min(pixel_spacing * 2.0, estimates.sigma));
    estimates.amplitude = std::max(estimates.amplitude, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amplitude > 0 && estimates.sigma > 0 && 
        !std::isnan(estimates.center) && !std::isnan(estimates.amplitude) && 
        !std::isnan(estimates.sigma) && !std::isnan(estimates.offset)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Method 1 (Physics-based): A=" << estimates.amplitude 
                     << ", m=" << estimates.center << ", sigma=" << estimates.sigma 
                     << ", B=" << estimates.offset << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center = stats.median; // More robust than weighted mean
    estimates.offset = stats.q25;
    estimates.amplitude = stats.q75 - stats.q25; // Inter-quartile range
    estimates.sigma = std::max(stats.mad, pixel_spacing * 0.3);
    
    if (estimates.amplitude > 0 && estimates.sigma > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Method 2 (Robust statistical): A=" << estimates.amplitude 
                     << ", m=" << estimates.center << ", sigma=" << estimates.sigma 
                     << ", B=" << estimates.offset << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center = center_estimate;
    estimates.offset = 0.0;
    estimates.amplitude = stats.max_val;
    estimates.sigma = pixel_spacing * 0.5;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "Method 3 (Conservative fallback): A=" << estimates.amplitude 
                 << ", m=" << estimates.center << ", sigma=" << estimates.sigma 
                 << ", B=" << estimates.offset << std::endl;
    }
    
    return estimates;
}

// Enhanced outlier filtering with multiple strategies
std::pair<std::vector<double>, std::vector<double>> FilterOutliers(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double sigma_threshold = 2.5,
    bool verbose = false) {
    
    std::vector<double> filtered_x, filtered_y;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 3) {
        return std::make_pair(filtered_x, filtered_y);
    }
    
    DataStatistics stats = CalculateRobustStatistics(x_vals, y_vals);
    if (!stats.valid) {
        return std::make_pair(x_vals, y_vals); // Return original if stats fail
    }
    
    // Use MAD-based outlier detection (more robust than standard deviation)
    double outlier_threshold = stats.median + sigma_threshold * stats.mad;
    double lower_threshold = stats.median - sigma_threshold * stats.mad;
    
    int outliers_removed = 0;
    for (size_t i = 0; i < y_vals.size(); ++i) {
        // Keep points that are within the robust bounds
        if (y_vals[i] >= lower_threshold && y_vals[i] <= outlier_threshold) {
            filtered_x.push_back(x_vals[i]);
            filtered_y.push_back(y_vals[i]);
        } else {
            outliers_removed++;
        }
    }
    
    // If too many outliers were removed, use a more lenient approach
    if (filtered_x.size() < x_vals.size() / 2) {
        if (verbose) {
            std::cout << "Too many outliers detected (" << outliers_removed 
                     << "), using lenient filtering" << std::endl;
        }
        
        filtered_x.clear();
        filtered_y.clear();
        
        // Just remove extreme outliers (beyond 4-sigma)
        double extreme_threshold = stats.median + 4.0 * stats.mad;
        double extreme_lower = stats.median - 4.0 * stats.mad;
        
        for (size_t i = 0; i < y_vals.size(); ++i) {
            if (y_vals[i] >= extreme_lower && y_vals[i] <= extreme_threshold) {
                filtered_x.push_back(x_vals[i]);
                filtered_y.push_back(y_vals[i]);
            }
        }
    }
    
    // Ensure we have enough points for fitting
    if (filtered_x.size() < 4) {
        if (verbose) {
            std::cout << "Warning: After outlier filtering, only " << filtered_x.size() 
                     << " points remain" << std::endl;
        }
        return std::make_pair(x_vals, y_vals); // Return original data
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_pair(filtered_x, filtered_y);
}

// Enhanced weighting strategies for spatial uncertainty and bias reduction (Gaussian) with "horizontal errors"
// Implements the specific techniques requested:
// 1. Downweight central pixel: Assign larger uncertainty to highest-charge pixel
// 2. Distance-based weights: w_i ∝ 1/(1 + d_i/d₀) where d_i is distance from current estimate
// 3. Robust losses: Use Cauchy/Huber loss functions to moderate single pixel influence
// 4. Pixel integration: Model pixel response by integrating over pixel area
// 5. Spatial error maps: Position-dependent weighting based on reconstruction error maps
struct WeightingConfig {
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
    double central_pixel_weight_factor = 0.08;        // Reduce central pixel weight to 8% (most aggressive for Gaussian)
    double central_downweight_threshold = 1.8;        // Charge concentration threshold for adaptive downweighting
    double max_central_pixel_uncertainty = 10.0;     // Maximum uncertainty multiplier for central pixel
    
    // Distance-based weighting parameters (technique #2)
    double distance_scale_d0 = 10.0;                 // d₀ parameter: w_i ∝ 1/(1 + d_i/d₀) [μm] (physics d₀ = 10 µm)
    double distance_weight_cap = 8.0;                // Maximum distance-based weight multiplier
    
    // Robust loss parameters (technique #3)
    double robust_threshold_factor = 0.06;           // Threshold for robust loss functions (most aggressive for Gaussian)
    double dynamic_loss_threshold = 2.0;             // Threshold for dynamic loss function switching [sigma]
    
    // Pixel integration parameters (technique #4)
    double horizontal_error_scale = 0.6;             // Scale factor for horizontal errors (fraction of pixel size)
    double spatial_uncertainty_factor = 0.5;        // Pixel size uncertainty factor for horizontal errors
    double pixel_edge_uncertainty_factor = 1.0;     // Additional uncertainty for pixels near edges
    double pixel_integration_samples = 5;           // Number of samples for pixel integration (when not using analytical)
    
    // Spatial error map parameters (technique #5)
    double spatial_error_map_strength = 0.3;        // Strength of spatial error map corrections
    double position_dependent_bias_scale = 0.25;    // Scale for position-dependent bias corrections
    
    // Additional parameters for enhanced performance
    double edge_pixel_boost_factor = 2.0;           // Boost factor for edge pixels (increased for better position info)
    double charge_uncertainty_floor = 0.02;         // Minimum relative charge uncertainty (2%)
    double systematic_bias_strength = 0.4;          // Strength of systematic bias correction (strongest for Gaussian)
    double correlation_radius = 1.5;                // Radius for inter-pixel correlation weighting [pixels]
    double adaptive_weight_update_rate = 0.7;       // Rate for iterative weight updates
};

// Calculate weights for data points with enhanced spatial uncertainty handling
std::vector<double> CalculateDataWeights(const std::vector<double>& x_vals,
                                       const std::vector<double>& y_vals,
                                       const ParameterEstimates& estimates,
                                       double pixel_spacing = 50.0,
                                       const WeightingConfig& config = WeightingConfig()) {
    std::vector<double> weights(x_vals.size(), 1.0);
    
    DataStatistics stats = CalculateRobustStatistics(x_vals, y_vals);
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
        double charge_weight = std::sqrt(std::max(0.0, y_vals[i] - estimates.offset));
        charge_weight = std::max(0.1, std::min(10.0, charge_weight / stats.mad));
        
        // 2. Enhanced central pixel downweighting strategy with systematic bias correction
        if (config.enable_central_pixel_downweight && i == max_charge_idx) {
            if (config.enable_adaptive_central_weighting) {
                // Adaptive downweighting based on charge concentration
                double charge_concentration = max_charge / mean_charge;
                double adaptive_factor = config.central_pixel_weight_factor;
                if (charge_concentration > config.central_downweight_threshold) {
                    // More aggressive downweighting for highly concentrated charge
                    adaptive_factor *= 0.6; // Stronger downweighting as requested
                    
                    // Apply systematic bias correction: central pixels tend to pull fits inward
                    if (config.enable_systematic_bias_correction) {
                        double bias_correction = 1.0 - config.systematic_bias_strength * 
                                               (charge_concentration - config.central_downweight_threshold);
                        adaptive_factor *= std::max(0.1, bias_correction);
                    }
                }
                charge_weight *= adaptive_factor;
                
                // Additional uncertainty for central pixel to implement "horizontal errors"
                if (config.enable_charge_weighted_uncertainty) {
                    double uncertainty_multiplier = 1.0 + (charge_concentration - 1.0) * 0.5;
                    uncertainty_multiplier = std::min(config.max_central_pixel_uncertainty, uncertainty_multiplier);
                    charge_weight /= uncertainty_multiplier; // Larger uncertainty = smaller weight
                }
            } else {
                charge_weight *= config.central_pixel_weight_factor;
            }
        }
        
        // 3. Enhanced distance-based weighting: w_i ∝ 1/(1 + d_i/d₀) for center estimation
        double distance_weight = 1.0;
        if (config.enable_distance_based_weights && estimates.sigma > 0) {
            double dx = std::abs(x_vals[i] - estimates.center);
            double d0_scaled = config.distance_scale_d0 * (pixel_spacing / 50.0);
            distance_weight = 1.0 / (1.0 + dx / d0_scaled);
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
            double spatial_sigma = config.spatial_uncertainty_factor * pixel_spacing;
            double total_sigma = std::sqrt(estimates.sigma * estimates.sigma + spatial_sigma * spatial_sigma);
            
            // Weight by inverse of total uncertainty
            if (total_sigma > 0) {
                spatial_weight = estimates.sigma / total_sigma;
            }
        }
        
        // 5. Enhanced horizontal error correction for pixel integration with edge effects
        double horizontal_error_weight = 1.0;
        if (config.enable_horizontal_error_correction) {
            // Calculate horizontal error based on pixel position relative to estimated center
            double dx = std::abs(x_vals[i] - estimates.center);
            double pixel_edge_distance = pixel_spacing / 2.0;
            
            // Base horizontal uncertainty from pixel size (the fundamental "horizontal error")
            double horizontal_error = config.horizontal_error_scale * pixel_spacing;
            
            // Additional uncertainty for pixels near pixel edges
            if (config.enable_pixel_edge_uncertainty) {
                double edge_proximity = std::max(0.0, pixel_edge_distance - (dx - pixel_edge_distance));
                if (edge_proximity > 0) {
                    double edge_factor = 1.0 + config.pixel_edge_uncertainty_factor * 
                                        (edge_proximity / pixel_edge_distance);
                    horizontal_error *= edge_factor;
                }
            }
            
            // If the pixel is at the edge relative to the fitted center, increase uncertainty
            if (dx > pixel_edge_distance) {
                horizontal_error *= (1.0 + (dx - pixel_edge_distance) / pixel_spacing);
            }
            
            // Charge-dependent horizontal error: higher charge pixels have more precise position info
            if (config.enable_charge_weighted_uncertainty) {
                double charge_factor = std::max(config.charge_uncertainty_floor, 
                                               1.0 / std::sqrt(y_vals[i] / mean_charge));
                horizontal_error *= charge_factor;
            }
            
            // Weight inversely proportional to horizontal error
            horizontal_error_weight = 1.0 / (1.0 + horizontal_error / pixel_spacing);
        }
        
        // 6. Spatial error maps for position-dependent weighting
        double spatial_error_weight = 1.0;
        if (config.enable_spatial_error_maps) {
            // Model systematic position-dependent reconstruction errors
            double dx = x_vals[i] - estimates.center;
            double pixel_position_error = 0.0;
            
            // Near pixel center: minimal error
            // Near pixel edges: increased error due to charge sharing
            double normalized_distance = std::abs(dx) / (pixel_spacing / 2.0);
            if (normalized_distance < 1.0) {
                // Quadratic increase in error near pixel edges
                pixel_position_error = config.position_dependent_bias_scale * 
                                     normalized_distance * normalized_distance * pixel_spacing;
            } else {
                // Linear increase for pixels beyond immediate neighbors
                pixel_position_error = config.position_dependent_bias_scale * 
                                     (1.0 + (normalized_distance - 1.0)) * pixel_spacing;
            }
            
            spatial_error_weight = 1.0 / (1.0 + config.spatial_error_map_strength * 
                                         pixel_position_error / pixel_spacing);
        }
        
        // 7. Inter-pixel correlation weighting
        double correlation_weight = 1.0;
        if (config.enable_correlation_weighting) {
            // Weight based on local charge distribution consistency
            double local_charge_variance = 0.0;
            double local_charge_sum = 0.0;
            int neighbor_count = 0;
            
            // Check neighboring pixels within correlation radius
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
                
                // Higher variance = less reliable, lower weight
                double variance_factor = 1.0 / (1.0 + local_charge_variance / (mean_charge * mean_charge));
                correlation_weight = 0.5 + 0.5 * variance_factor; // Range [0.5, 1.0]
            }
        }
        
        // 5. Enhanced robust weighting based on expected Gaussian profile
        double profile_weight = 1.0;
        if (config.enable_robust_profile_weighting && estimates.sigma > 0) {
            double dx = std::abs(x_vals[i] - estimates.center);
            double expected_value = estimates.amplitude * std::exp(-0.5 * dx * dx / (estimates.sigma * estimates.sigma)) + estimates.offset;
            
            // Calculate normalized residual with robust scaling
            double residual = std::abs(y_vals[i] - expected_value);
            double expected_error = std::max(1.0, std::sqrt(expected_value)); // Poisson-like error model
            double normalized_residual = residual / expected_error;
            
            // Apply robust weighting: reduce influence of outliers
            profile_weight = 1.0 / (1.0 + normalized_residual * normalized_residual);
            profile_weight = std::max(0.05, profile_weight); // Allow more aggressive downweighting
        }
        
        // ---- Analytic charge-sharing base weight ---------------------------------
        const double dx_distance = std::abs(x_vals[i] - estimates.center);
        const double l_pixel     = pixel_spacing;                       // approximate pad length
        const double num_cs      = (l_pixel * 0.5) * 1.41421356237;    // (l/2)*√2
        const double denom_cs    = num_cs + dx_distance;
        const double alpha_cs    = std::atan(num_cs / std::max(1e-12, denom_cs)); // solid angle term
        // Scale d0 with pixel spacing so that d0 remains 0.2·pitch for any line orientation
        const double d0_cs       = std::max(1e-6, config.distance_scale_d0 * (pixel_spacing / 50.0));
        double       log_term    = std::log(std::max(1e-6, dx_distance) / d0_cs);
        if (std::abs(log_term) < 1e-6) log_term = (log_term < 0 ? -1e-6 : 1e-6);
        const double cs_weight   = alpha_cs / std::abs(log_term);
        // -------------------------------------------------------------------------
        
        // Combine all weighting factors (analytic charge-sharing base + modifiers)
        weight = cs_weight * charge_weight * spatial_weight * profile_weight * 
                edge_weight * horizontal_error_weight * spatial_error_weight * correlation_weight;
        
        // Ensure weight is reasonable and apply final normalization
        weights[i] = std::max(0.01, std::min(100.0, weight));
    }
    
    return weights;
}

// ========================================================================================================
// HORIZONTAL ERROR TECHNIQUES IMPLEMENTATION - Core Gaussian fitting function using Ceres Solver
// ========================================================================================================
// 
// This function implements the five core horizontal error techniques for spatial uncertainty reduction:
// 
// 1. CENTRAL PIXEL DOWNWEIGHTING
//    - Reduces central pixel weight to 8% (most aggressive for sharp Gaussian peaks)
//    - Uses adaptive thresholds based on charge concentration (threshold: 2.0)
//    - Implements ScaledLoss for maximum central pixel suppression
//    - Prevents highest-charge pixel from dominating position reconstruction
// 
// 2. DISTANCE-BASED WEIGHTING
//    - Formula: w_i ∝ 1/(1 + d_i/d₀) where d₀ = 10μm (physics d₀ value)
//    - Gives more weight to pixels closer to current center estimate
//    - Stabilizes convergence while allowing position refinement
//    - Caps maximum weight at 8x to prevent extreme values
// 
// 3. ROBUST LOSS FUNCTIONS
//    - Uses Cauchy and Huber losses with threshold factor 0.06 (most aggressive)
//    - Dynamic switching: stronger losses (50% threshold) for central pixels
//    - Prevents single pixel outliers from dominating the fit
//    - Additional ScaledLoss for maximum central pixel suppression
// 
// 4. PIXEL INTEGRATION MODEL
//    - Analytical integration using error function: A*σ_eff*√(2π)*[erf(right)-erf(left)]/2
//    - Includes horizontal uncertainty in effective sigma: σ_eff = √(σ² + σ_h²)
//    - Models pixel response over finite area instead of point sampling
//    - Horizontal error scale: 60% of pixel size
// 
// 5. SPATIAL ERROR MAPS
//    - Position-dependent weighting based on systematic error patterns
//    - Quadratic error increase near pixel edges due to charge sharing
//    - Linear error growth for pixels beyond immediate neighbors
//    - Bias correction scale: 25% optimized for Gaussian distributions
// 
// ADVANCED FEATURES:
// - Edge pixel upweighting (2.0x boost) for better position sensitivity
// - Charge-weighted uncertainty (higher charge = more precise position)
// - Inter-pixel correlation weighting (radius: 1.5 pixels)
// - Systematic bias correction (strength: 40%)
// - Multi-strategy outlier filtering with progressive fallbacks
// - Thread-safe implementation with comprehensive error handling
//
// Ultra-robust Gaussian fitting with multiple strategies
bool FitGaussianCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    double& fit_amplitude,
    double& fit_center,
    double& fit_sigma,
    double& fit_offset,
    double& fit_amplitude_err,
    double& fit_center_err,
    double& fit_sigma_err,
    double& fit_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        if (verbose) {
            std::cout << "Insufficient data points for Gaussian fitting" << std::endl;
        }
        return false;
    }
    
    // Multiple outlier filtering strategies (conditional based on user preference)
    std::vector<std::pair<std::vector<double>, std::vector<double>>> filtered_datasets;
    
    if (enable_outlier_filtering) {
        // Strategy 1: Conservative filtering (2.5-sigma)
        auto conservative_data = FilterOutliers(x_vals, y_vals, 2.5, verbose);
        if (conservative_data.first.size() >= 4) {
            filtered_datasets.push_back(conservative_data);
        }
        
        // Strategy 2: Lenient filtering (3.0-sigma)
        auto lenient_data = FilterOutliers(x_vals, y_vals, 3.0, verbose);
        if (lenient_data.first.size() >= 4) {
            filtered_datasets.push_back(lenient_data);
        }
    }
    
    // Strategy 3: No filtering (use all data) - always available as fallback
    filtered_datasets.push_back(std::make_pair(x_vals, y_vals));
    
    if (verbose) {
        std::cout << "Outlier filtering " << (enable_outlier_filtering ? "enabled" : "disabled") 
                 << ", testing " << filtered_datasets.size() << " datasets" << std::endl;
    }
    
    // Try each filtered dataset
    for (size_t dataset_idx = 0; dataset_idx < filtered_datasets.size(); ++dataset_idx) {
        std::vector<double> clean_x = filtered_datasets[dataset_idx].first;
        std::vector<double> clean_y = filtered_datasets[dataset_idx].second;
        
        if (clean_x.size() < 4) continue;
        
        if (verbose) {
            std::cout << "Trying dataset " << dataset_idx << " with " << clean_x.size() << " points" << std::endl;
        }
        
        // Get parameter estimates
        ParameterEstimates estimates = EstimateGaussianParameters(clean_x, clean_y, center_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "Parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calculate data weights
        WeightingConfig config;
        std::vector<double> data_weights = CalculateDataWeights(clean_x, clean_y, estimates, pixel_spacing, config);
        
        // Multiple fitting attempts with different configurations
        struct FittingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        // Build solver configurations once per dataset so parameters reflect current estimates
        const std::vector<FittingConfig> configs = {
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-15, 1e-15, 800, "NONE", 0.0},            // cheap first pass
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-15, 1e-15, 1500, "HUBER", estimates.amplitude * config.robust_threshold_factor},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amplitude * config.robust_threshold_factor * 1.8},
            {ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amplitude * config.robust_threshold_factor * 1.4},
            {ceres::SPARSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1200, "CAUCHY", estimates.amplitude * config.robust_threshold_factor * 2.5}
        };
        
        // Progressive solver escalation
        for (const auto& config : configs) {
            // Set up parameter array with estimates
            double parameters[4];
            parameters[0] = estimates.amplitude;
            parameters[1] = estimates.center;
            parameters[2] = estimates.sigma;
            parameters[3] = estimates.offset;
            
            // Build the problem
            ceres::Problem problem;
            
            // Add residual blocks with weights and loss functions
            WeightingConfig weight_config;
            
            // Calculate robust statistics for weighting and outlier detection
            DataStatistics stats = CalculateRobustStatistics(clean_x, clean_y);
            
            // Find the highest charge pixel for central pixel downweighting
            size_t max_charge_idx = 0;
            double max_charge = clean_y[0];
            for (size_t i = 0; i < clean_y.size(); ++i) {
                if (clean_y[i] > max_charge) {
                    max_charge = clean_y[i];
                    max_charge_idx = i;
                }
            }
            
            // Container to own loss functions so their lifetime exceeds Problem::Solve
            std::vector<std::unique_ptr<ceres::LossFunction>> owned_losses;
            
            for (size_t i = 0; i < clean_x.size(); ++i) {
                ceres::CostFunction* cost_function;
                ceres::LossFunction* loss_function = nullptr;
                
                // Calculate pixel-specific horizontal uncertainty for this pixel (needed for both pixel integration and dynamic loss switching)
                double horizontal_uncertainty = 0.0;
                if (weight_config.enable_horizontal_error_correction) {
                    // Base horizontal error from pixel size
                    horizontal_uncertainty = weight_config.horizontal_error_scale * pixel_spacing;
                    
                    // Enhanced pixel-specific horizontal error calculation
                    double dx = std::abs(clean_x[i] - estimates.center);
                    double pixel_edge_distance = pixel_spacing / 2.0;
                    
                    // Increase uncertainty for pixels farther from center
                    if (dx > pixel_edge_distance) {
                        double edge_factor = 1.0 + (dx - pixel_edge_distance) / pixel_spacing;
                        horizontal_uncertainty *= edge_factor;
                    }
                    
                    // Charge-dependent horizontal uncertainty
                    if (weight_config.enable_charge_weighted_uncertainty) {
                        double charge_factor = 1.0 / std::sqrt(std::max(0.1, clean_y[i] / stats.mean));
                        horizontal_uncertainty *= charge_factor;
                    }
                    
                    // Position-dependent horizontal error from spatial error maps
                    if (weight_config.enable_spatial_error_maps) {
                        double position_error = weight_config.position_dependent_bias_scale * 
                                               std::abs(dx) / pixel_spacing * pixel_spacing;
                        horizontal_uncertainty += position_error;
                    }
                }
                
                // Technique #4: Choose between pixel integration and point evaluation
                if (weight_config.enable_pixel_integration) {
                    cost_function = PixelIntegratedGaussianCostFunction::Create(
                        clean_x[i], clean_y[i], pixel_spacing, data_weights[i], horizontal_uncertainty);
                } else {
                    cost_function = GaussianCostFunction::Create(
                        clean_x[i], clean_y[i], data_weights[i]);
                }
                
                // Technique #3: Enhanced robust loss functions with dynamic switching
                if (weight_config.enable_dynamic_loss_switching) {
                    // For central high-charge pixels, use stronger robust loss to reduce dominance
                    if (i == max_charge_idx && clean_y[i] > stats.mean * weight_config.central_downweight_threshold) {
                        // Use more aggressive robust loss for central pixel
                        double stronger_parameter = config.loss_parameter * 0.5;
                        if (config.loss_function == "HUBER") {
                            owned_losses.emplace_back(std::make_unique<ceres::HuberLoss>(stronger_parameter));
                            loss_function = owned_losses.back().get();
                        } else if (config.loss_function == "CAUCHY") {
                            owned_losses.emplace_back(std::make_unique<ceres::CauchyLoss>(stronger_parameter));
                            loss_function = owned_losses.back().get();
                        }
                    }
                }
                
                // Add residual block with appropriate loss function (nullptr if no loss function)
                problem.AddResidualBlock(cost_function, loss_function, parameters);
            }
            
            // Set robust bounds based on physics and estimates
            double amp_min = std::max(1e-20, estimates.amplitude * 0.01);

            double max_charge_val = *std::max_element(clean_y.begin(), clean_y.end());
            double physics_amp_max = max_charge_val * 1.5;
            double algo_amp_max = estimates.amplitude * 100.0;
            double amp_max = std::min(physics_amp_max, algo_amp_max);

            problem.SetParameterLowerBound(parameters, 0, amp_min);
            problem.SetParameterUpperBound(parameters, 0, amp_max);
            
            double center_range = pixel_spacing * 3.0;
            problem.SetParameterLowerBound(parameters, 1, estimates.center - center_range);
            problem.SetParameterUpperBound(parameters, 1, estimates.center + center_range);
            
            problem.SetParameterLowerBound(parameters, 2, pixel_spacing * 0.05);
            problem.SetParameterUpperBound(parameters, 2, pixel_spacing * 3.0);
            
            double offset_range = std::max(estimates.amplitude * 0.5, std::abs(estimates.offset) * 2.0);
            problem.SetParameterLowerBound(parameters, 3, estimates.offset - offset_range);
            problem.SetParameterUpperBound(parameters, 3, estimates.offset + offset_range);
            
            // Configure solver with robust settings
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
                fit_sigma = std::abs(parameters[2]);
                fit_offset = parameters[3];
                
                // Calculate uncertainties using robust methods
                bool cov_success = false;
                
                // Try covariance calculation with multiple settings
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
                    cov_options.null_space_rank = 2; // Allow for rank deficiency
                    cov_options.apply_loss_function = true;
                    
                    ceres::Covariance covariance(cov_options);
                    std::vector<std::pair<const double*, const double*>> covariance_blocks;
                    covariance_blocks.push_back(std::make_pair(parameters, parameters));
                    
                    if (covariance.Compute(covariance_blocks, &problem)) {
                        double covariance_matrix[16];
                        if (covariance.GetCovarianceBlock(parameters, parameters, covariance_matrix)) {
                            fit_amplitude_err = std::sqrt(std::abs(covariance_matrix[0]));
                            fit_center_err = std::sqrt(std::abs(covariance_matrix[5]));
                            fit_sigma_err = std::sqrt(std::abs(covariance_matrix[10]));
                            fit_offset_err = std::sqrt(std::abs(covariance_matrix[15]));
                            
                            // Validate uncertainties
                            if (!std::isnan(fit_amplitude_err) && !std::isnan(fit_center_err) &&
                                !std::isnan(fit_sigma_err) && !std::isnan(fit_offset_err) &&
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
                    DataStatistics data_stats = CalculateRobustStatistics(clean_x, clean_y);
                    fit_amplitude_err = std::max(0.02 * fit_amplitude, 0.1 * data_stats.mad);
                    fit_center_err = std::max(0.02 * pixel_spacing, fit_sigma / 10.0);
                    fit_sigma_err = std::max(0.05 * fit_sigma, 0.01 * pixel_spacing);
                    fit_offset_err = std::max(0.1 * std::abs(fit_offset), 0.05 * data_stats.mad);
                }
                
                // Calculate reduced chi-squared
                // Ceres reports 0.5 * Σ r_i^2, so multiply by 2 to obtain χ².
                double chi2 = summary.final_cost * 2.0;
                int dof = std::max(1, static_cast<int>(clean_x.size()) - 4);
                chi2_reduced = chi2 / dof;
                
                if (verbose) {
                    std::cout << "Successful fit with config " << &config - &configs[0] 
                             << ", dataset " << dataset_idx 
                             << ": A=" << fit_amplitude << "±" << fit_amplitude_err
                             << ", m=" << fit_center << "±" << fit_center_err
                             << ", sigma=" << fit_sigma << "±" << fit_sigma_err
                             << ", B=" << fit_offset << "±" << fit_offset_err
                             << ", chi2red=" << chi2_reduced << std::endl;
                }
                
                return true;
            } else if (verbose) {
                std::cout << "Fit failed with config " << &config - &configs[0] 
                         << ": " << summary.BriefReport() << std::endl;
            }
        }
    }
    
    if (verbose) {
        std::cout << "All fitting strategies failed" << std::endl;
    }
    return false;
}

GaussianFit2DResultsCeres Fit2DGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    GaussianFit2DResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresFitMutex);
    
    // Initialize Ceres logging
    InitializeCeres();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit2DGaussianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Fit2DGaussianCeres: Error - need at least 4 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 2D Gaussian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
        for (const auto& point : row_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting X direction with " << x_vals.size() << " points" << std::endl;
        }
        
        x_fit_success = FitGaussianCeres(
            x_vals, y_vals, center_x_estimate, pixel_spacing,
            result.x_amplitude, result.x_center, result.x_sigma, result.x_vertical_offset,
            result.x_amplitude_err, result.x_center_err, result.x_sigma_err, result.x_vertical_offset_err,
            result.x_chi2red, verbose, enable_outlier_filtering);
        
        // Calculate DOF and p-value
        result.x_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.x_pp = (result.x_chi2red > 0) ? 1.0 - std::min(1.0, result.x_chi2red / 10.0) : 0.0; // Simple p-value approximation
    }
    
    // Fit Y direction (central column)
    if (cols_data.find(best_col_x) != cols_data.end() && cols_data[best_col_x].size() >= 4) {
        auto& col_data = cols_data[best_col_x];
        
        // Sort by Y coordinate
        std::sort(col_data.begin(), col_data.end());
        
        // Create vectors for fitting
        std::vector<double> x_vals, y_vals;
        for (const auto& point : col_data) {
            x_vals.push_back(point.first); // Y coordinate
            y_vals.push_back(point.second); // charge
        }
        
        if (verbose) {
            std::cout << "Fitting Y direction with " << x_vals.size() << " points" << std::endl;
        }
        
        y_fit_success = FitGaussianCeres(
            x_vals, y_vals, center_y_estimate, pixel_spacing,
            result.y_amplitude, result.y_center, result.y_sigma, result.y_vertical_offset,
            result.y_amplitude_err, result.y_center_err, result.y_sigma_err, result.y_vertical_offset_err,
            result.y_chi2red, verbose, enable_outlier_filtering);
        
        // Calculate DOF and p-value
        result.y_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.y_pp = (result.y_chi2red > 0) ? 1.0 - std::min(1.0, result.y_chi2red / 10.0) : 0.0; // Simple p-value approximation
    }
    
    // Set overall success status
    result.fit_successful = x_fit_success && y_fit_success;
    
    if (verbose) {
        std::cout << "2D Gaussian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

DiagonalFitResultsCeres FitDiagonalGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    DiagonalFitResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresFitMutex);
    
    // Initialize Ceres logging
    InitializeCeres();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size() || x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Diagonal Gaussian fit (Ceres): Invalid input data size" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting Diagonal Gaussian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // Tolerance for grouping pixels into diagonals
    double tolerance = pixel_spacing * 0.1;
    
    // Effective centre-to-centre pitch along a ±45° diagonal is √2 times the horizontal pitch.
    const double diag_pixel_spacing = pixel_spacing * 1.41421356237; // ≈ √2
    
    // Group pixels by diagonal lines
    std::map<double, std::vector<std::pair<double, double>>> main_diagonal_data; // main_diag_id -> [(x, charge), (y, charge)]
    std::map<double, std::vector<std::pair<double, double>>> sec_diagonal_data;  // sec_diag_id -> [(x, charge), (y, charge)]
    
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
        if (dist < min_main_diag_dist && diag_pair.second.size() >= 8) { // Need at least 4 points for each x and y fit
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
        if (dist < min_sec_diag_dist && diag_pair.second.size() >= 8) { // Need at least 4 points for each x and y fit
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
                std::cout << "Fitting main diagonal X with " << x_sorted.size() << " points" << std::endl;
            }
            
            main_diag_x_success = FitGaussianCeres(
                x_sorted, charge_x_sorted, 0.0, diag_pixel_spacing,
                result.main_diag_x_amplitude, result.main_diag_x_center, result.main_diag_x_sigma, result.main_diag_x_vertical_offset,
                result.main_diag_x_amplitude_err, result.main_diag_x_center_err, result.main_diag_x_sigma_err, result.main_diag_x_vertical_offset_err,
                result.main_diag_x_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.main_diag_x_dof = std::max(1, static_cast<int>(x_sorted.size()) - 4);
            result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
            result.main_diag_x_fit_successful = main_diag_x_success;
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
                std::cout << "Fitting main diagonal Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            main_diag_y_success = FitGaussianCeres(
                y_sorted, charge_y_sorted, 0.0, diag_pixel_spacing,
                result.main_diag_y_amplitude, result.main_diag_y_center, result.main_diag_y_sigma, result.main_diag_y_vertical_offset,
                result.main_diag_y_amplitude_err, result.main_diag_y_center_err, result.main_diag_y_sigma_err, result.main_diag_y_vertical_offset_err,
                result.main_diag_y_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.main_diag_y_dof = std::max(1, static_cast<int>(y_sorted.size()) - 4);
            result.main_diag_y_pp = (result.main_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_y_chi2red / 10.0) : 0.0;
            result.main_diag_y_fit_successful = main_diag_y_success;
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
                std::cout << "Fitting secondary diagonal X with " << x_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_x_success = FitGaussianCeres(
                x_sorted, charge_x_sorted, 0.0, diag_pixel_spacing,
                result.sec_diag_x_amplitude, result.sec_diag_x_center, result.sec_diag_x_sigma, result.sec_diag_x_vertical_offset,
                result.sec_diag_x_amplitude_err, result.sec_diag_x_center_err, result.sec_diag_x_sigma_err, result.sec_diag_x_vertical_offset_err,
                result.sec_diag_x_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.sec_diag_x_dof = std::max(1, static_cast<int>(x_sorted.size()) - 4);
            result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
            result.sec_diag_x_fit_successful = sec_diag_x_success;
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
                std::cout << "Fitting secondary diagonal Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_y_success = FitGaussianCeres(
                y_sorted, charge_y_sorted, 0.0, diag_pixel_spacing,
                result.sec_diag_y_amplitude, result.sec_diag_y_center, result.sec_diag_y_sigma, result.sec_diag_y_vertical_offset,
                result.sec_diag_y_amplitude_err, result.sec_diag_y_center_err, result.sec_diag_y_sigma_err, result.sec_diag_y_vertical_offset_err,
                result.sec_diag_y_chi2red, verbose, enable_outlier_filtering);
            
            // Calculate DOF and p-value
            result.sec_diag_y_dof = std::max(1, static_cast<int>(y_sorted.size()) - 4);
            result.sec_diag_y_pp = (result.sec_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_y_chi2red / 10.0) : 0.0;
            result.sec_diag_y_fit_successful = sec_diag_y_success;
        }
    }
    
    // Set overall success status
    result.fit_successful = main_diag_x_success && main_diag_y_success && sec_diag_x_success && sec_diag_y_success;
    
    if (verbose) {
        std::cout << "Diagonal Gaussian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (Main X: " << (main_diag_x_success ? "OK" : "FAIL") 
                 << ", Main Y: " << (main_diag_y_success ? "OK" : "FAIL")
                 << ", Sec X: " << (sec_diag_x_success ? "OK" : "FAIL") 
                 << ", Sec Y: " << (sec_diag_y_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

// Standalone outlier removal function with boolean control
OutlierRemovalResult RemoveOutliers(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& charge_values,
    bool enable_outlier_removal,
    double sigma_threshold,
    bool verbose) {
    
    OutlierRemovalResult result;
    result.outliers_removed = 0;
    result.filtering_applied = enable_outlier_removal;
    result.success = false;
    
    // Input validation
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "RemoveOutliers: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.empty()) {
        if (verbose) {
            std::cout << "RemoveOutliers: Error - empty input vectors" << std::endl;
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
            std::cout << "RemoveOutliers: Outlier removal disabled, returning original data (" 
                     << x_coords.size() << " points)" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "RemoveOutliers: Starting outlier removal on " << x_coords.size() 
                 << " points with sigma threshold " << sigma_threshold << std::endl;
    }
    
    // Calculate robust statistics for charge values
    DataStatistics stats = CalculateRobustStatistics(x_coords, charge_values);
    if (!stats.valid) {
        if (verbose) {
            std::cout << "RemoveOutliers: Failed to calculate statistics, returning original data" << std::endl;
        }
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.success = true;
        return result;
    }
    
    // Use MAD-based outlier detection (more robust than standard deviation)
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
    if (result.filtered_x_coords.size() < x_coords.size() / 2) {
        if (verbose) {
            std::cout << "RemoveOutliers: Too many outliers detected (" << result.outliers_removed 
                     << "), using lenient filtering (4-sigma)" << std::endl;
        }
        
        result.filtered_x_coords.clear();
        result.filtered_y_coords.clear();
        result.filtered_charge_values.clear();
        result.outliers_removed = 0;
        
        // Use 4-sigma threshold for lenient filtering
        double lenient_threshold = stats.median + 4.0 * stats.mad;
        double lenient_lower = stats.median - 4.0 * stats.mad;
        
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
            std::cout << "RemoveOutliers: Warning - only " << result.filtered_x_coords.size() 
                     << " points remain after filtering, returning original data" << std::endl;
        }
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
    }
    
    result.success = true;
    
    if (verbose) {
        std::cout << "RemoveOutliers: Successfully filtered data - removed " << result.outliers_removed 
                 << " outliers, " << result.filtered_x_coords.size() << " points remaining" << std::endl;
    }
    
    return result;
} 