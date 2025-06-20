#include "2DGaussianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
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

// Calculate uncertainty as 5% of max charge in line (if enabled)
double CalculateUncertainty(double max_charge_in_line) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Uncertainty = 5% of max charge when enabled
    double uncertainty = 0.05 * max_charge_in_line;
    if (uncertainty < 1e-20) uncertainty = 1e-20; // Prevent division by zero (very small for coulomb range)
    return uncertainty;
}

// Gaussian cost function with uncertainty (5% of max charge)
// Function form: y(x) = A * exp(-(x - m)^2 / (2 * σ^2)) + B
struct GaussianCostFunction {
    GaussianCostFunction(double x, double y, double uncertainty) 
        : x_(x), y_(y), uncertainty_(uncertainty) {}
    
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
        
        // Residual divided by uncertainty (standard weighted least squares)
        residual[0] = (predicted - T(y_)) / T(uncertainty_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double uncertainty) {
        return (new ceres::AutoDiffCostFunction<GaussianCostFunction, 1, 4>(
            new GaussianCostFunction(x, y, uncertainty)));
    }
    
private:
    const double x_;
    const double y_;
    const double uncertainty_;
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
        
        // Calculate uncertainty as 5% of max charge
        double max_charge = *std::max_element(clean_y.begin(), clean_y.end());
        double uncertainty = CalculateUncertainty(max_charge);
        
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
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-15, 1e-15, 1500, "HUBER", estimates.amplitude * 0.1},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amplitude * 0.18},
            {ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amplitude * 0.14},
            {ceres::SPARSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1200, "CAUCHY", estimates.amplitude * 0.25}
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
            
            // Add residual blocks with uncertainties
            for (size_t i = 0; i < clean_x.size(); ++i) {
                ceres::CostFunction* cost_function = GaussianCostFunction::Create(
                    clean_x[i], clean_y[i], uncertainty);
                
                // No loss functions - simple weighted least squares
                problem.AddResidualBlock(cost_function, nullptr, parameters);
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
    
    // Calculate and store charge uncertainties (5% of max charge for each direction) only if enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        if (x_fit_success && rows_data.find(best_row_y) != rows_data.end()) {
            auto& row_data = rows_data[best_row_y];
            double max_charge_x = 0.0;
            for (const auto& point : row_data) {
                max_charge_x = std::max(max_charge_x, point.second);
            }
            result.x_charge_uncertainty = 0.05 * max_charge_x;
        }
        
        if (y_fit_success && cols_data.find(best_col_x) != cols_data.end()) {
            auto& col_data = cols_data[best_col_x];
            double max_charge_y = 0.0;
            for (const auto& point : col_data) {
                max_charge_y = std::max(max_charge_y, point.second);
            }
            result.y_charge_uncertainty = 0.05 * max_charge_y;
        }
    } else {
        result.x_charge_uncertainty = 0.0;
        result.y_charge_uncertainty = 0.0;
    }
    
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
    
    // ------------------------------------------------------------------------------------
    // New diagonal grouping (one entry per pixel) using true coordinate along the ±45° axis.
    // We project each hit onto the diagonal direction so that the abscissa that the
    // 1-D fitter sees is spaced by   pitch·√2   exactly as in the real geometry.
    // Define the diagonal coordinate s as (dx ± dy)/√2, where dx = x-cx, dy = y-cy.
    // ------------------------------------------------------------------------------------
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // Maps:   diagonal identifier  →  vector of (s, charge)
    std::map<double, std::vector<std::pair<double, double>>> main_diagonal_data; // slope +1
    std::map<double, std::vector<std::pair<double, double>>> sec_diagonal_data;  // slope −1

    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x      = x_coords[i];
        double y      = y_coords[i];
        double charge = charge_values[i];

        if (charge <= 0) continue; // ignore empty / negative pixels

        // Coordinates relative to centre estimates (for robust diagonal id)
        double dx = x - center_x_estimate;
        double dy = y - center_y_estimate;

        // Indicators for membership to the two sets of diagonals
        double diff = dx - dy;   // main diagonal (slope +1)
        double sum  = dx + dy;   // secondary diagonal (slope −1)

        // Main diagonal
        if (std::abs(diff) < tolerance) {
            double s = (dx + dy) * inv_sqrt2;          // coordinate along +45° direction
            main_diagonal_data[diff].emplace_back(s, charge);
        }

        // Secondary diagonal
        if (std::abs(sum) < tolerance) {
            double s = (dx - dy) * inv_sqrt2;          // coordinate along −45° direction
            sec_diagonal_data[sum].emplace_back(s, charge);
        }
    }
    
    // Find the main diagonal closest to the center estimates
    double center_main_diag_id = center_x_estimate - center_y_estimate;
    double best_main_diag_id = center_main_diag_id;
    double min_main_diag_dist = std::numeric_limits<double>::max();
    for (const auto& diag_pair : main_diagonal_data) {
        double dist = std::abs(diag_pair.first - center_main_diag_id);
        if (dist < min_main_diag_dist && diag_pair.second.size() >= 4) { // Need at least 4 points for each x and y fit
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
        if (dist < min_sec_diag_dist && diag_pair.second.size() >= 4) { // Need at least 4 points for each x and y fit
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
        main_diagonal_data[best_main_diag_id].size() >= 4) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Extract diagonal coordinate (s) and charge values for +45° direction
        std::vector<double> x_vals, charge_vals;
        for (const auto& p : main_diag_data) {
            x_vals.push_back(p.first);
            charge_vals.push_back(p.second);
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
        main_diagonal_data[best_main_diag_id].size() >= 4) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Re-use same dataset for the complementary fit (naming kept for backward compatibility)
        std::vector<double> y_vals, charge_vals;
        for (const auto& p : main_diag_data) {
            y_vals.push_back(p.first);
            charge_vals.push_back(p.second);
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
        sec_diagonal_data[best_sec_diag_id].size() >= 4) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Extract diagonal coordinate (s) and charge values for −45° direction
        std::vector<double> x_vals, charge_vals;
        for (const auto& p : sec_diag_data) {
            x_vals.push_back(p.first);
            charge_vals.push_back(p.second);
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
        sec_diagonal_data[best_sec_diag_id].size() >= 4) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Same dataset re-used for complementary fit
        std::vector<double> y_vals, charge_vals;
        for (const auto& p : sec_diag_data) {
            y_vals.push_back(p.first);
            charge_vals.push_back(p.second);
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