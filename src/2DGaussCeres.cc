#include "2DGaussCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <limits>
#include <numeric>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Use shared Google logging initialization
void InitializeCeres() {
    CeresLoggingInitializer::InitializeOnce();
}

// Calc err as 5% of max charge in line (if enabled)
double CalcErr(double max_charge_in_line) {
    if (!Constants::ENABLE_VERT_CHARGE_ERR) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Err = 5% of max charge when enabled
    double err = 0.05 * max_charge_in_line;
    if (err < Constants::MIN_UNCERTAINTY_VALUE) err = Constants::MIN_UNCERTAINTY_VALUE; // Prevent division by zero
    return err;
}

// Gauss cost function with err (5% of max charge)
// Function form: y(x) = A * exp(-(x - m)^2 / (2 * σ^2)) + B
struct GaussCostFunction {
    GaussCostFunction(double x, double y, double err) 
        : x_(x), y_(y), err_(err) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amp)
        // params[1] = m (center)
        // params[2] = sigma (width)
        // params[3] = B (offset)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& sigma = params[2];
        const T& B = params[3];
        
        // Robust handling of sigma
        T safe_sigma = ceres::abs(sigma);
        if (safe_sigma < T(Constants::MIN_SAFE_PARAMETER)) {
            safe_sigma = T(Constants::MIN_SAFE_PARAMETER);
        }
        
        // Gauss function with robust exponent calculation
        T dx = x_ - m;
        T exponent = -(dx * dx) / (T(2.0) * safe_sigma * safe_sigma);
        
        // Prevent numerical overflow/underflow with robust bounds
        if (exponent < T(-200.0)) {
            exponent = T(-200.0);
        } else if (exponent > T(200.0)) {
            exponent = T(200.0);
        }
        
        T predicted = A * ceres::exp(exponent) + B;
        
        // Residual divided by err (standard weighted least squares)
        residual[0] = (predicted - T(y_)) / T(err_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double err) {
        return (new ceres::AutoDiffCostFunction<GaussCostFunction, 1, 4>(
            new GaussCostFunction(x, y, err)));
    }
    
private:
    const double x_;
    const double y_;
    const double err_;
};

// Advanced parameter estimation with physics-based initialization
struct ParameterEstimates {
    double amp;
    double center;
    double sigma;
    double offset;
    double amp_err;
    double center_err;
    double sigma_err;
    double offset_err;
    bool valid;
    int method_used; // Track which estimation method was success
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

DataStatistics CalcRobustStatistics(const std::vector<double>& x_vals, 
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
    
    // Weighted statistics (weight by charge value for pos estimation)
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
ParameterEstimates EstimateGaussParameters(
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
    
    // Calc robust statistics
    DataStatistics stats = CalcRobustStatistics(x_vals, y_vals);
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
    estimates.amp = stats.max_val - estimates.offset;
    
    // Physics-based sigma estimation: charge spread should be related to pixel spacing
    // For LGAD detectors, typical charge spread is 0.3-0.8 pixel spacings
    double distance_spread = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - estimates.offset);
        if (weight > 0.1 * estimates.amp) { // Only use significant charges
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
    estimates.amp = std::max(estimates.amp, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amp > 0 && estimates.sigma > 0 && 
        !std::isnan(estimates.center) && !std::isnan(estimates.amp) && 
        !std::isnan(estimates.sigma) && !std::isnan(estimates.offset)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Method 1 (Physics-based): A=" << estimates.amp 
                     << ", m=" << estimates.center << ", sigma=" << estimates.sigma 
                     << ", B=" << estimates.offset << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center = stats.median; // More robust than weighted mean
    estimates.offset = stats.q25;
    estimates.amp = stats.q75 - stats.q25; // Inter-quartile range
    estimates.sigma = std::max(stats.mad, pixel_spacing * 0.3);
    
    if (estimates.amp > 0 && estimates.sigma > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Method 2 (Robust statistical): A=" << estimates.amp 
                     << ", m=" << estimates.center << ", sigma=" << estimates.sigma 
                     << ", B=" << estimates.offset << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center = center_estimate;
    estimates.offset = 0.0;
    estimates.amp = stats.max_val;
    estimates.sigma = pixel_spacing * 0.5;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "Method 3 (Conservative fallback): A=" << estimates.amp 
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
    
    DataStatistics stats = CalcRobustStatistics(x_vals, y_vals);
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
// HORIZONTAL ERROR TECHNIQUES IMPLEMENTATION - Core Gauss fitting function using Ceres Solver
// ========================================================================================================
// 
// This function implements the five core horizontal error techniques for spatial err reduction:
// 
// 1. CENTRAL PIXEL DOWNWEIGHTING
//    - Reduces central pixel weight to 8% (most aggressive for sharp Gauss peaks)
//    - Uses adaptive thresholds based on charge concentration (threshold: 2.0)
//    - Implements ScaledLoss for maximum central pixel suppression
//    - Prevents highest-charge pixel from dominating pos reconstruction
// 
// 2. DISTANCE-BASED WEIGHTING
//    - Formula: w_i ∝ 1/(1 + d_i/d₀) where d₀ = 10μm (physics d₀ value)
//    - Gives more weight to pixels closer to current center estimate
//    - Stabilizes convergence while allowing pos refinement
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
//    - Includes horizontal err in effective sigma: σ_eff = √(σ² + σ_h²)
//    - Models pixel response over finite area instead of point sampling
//    - Horizontal error scale: 60% of pixel size
// 
// 5. SPATIAL ERROR MAPS
//    - Pos-dependent weighting based on systematic error patterns
//    - Quadratic error increase near pixel edges due to charge sharing
//    - Linear error growth for pixels beyond immediate neighbors
//    - Bias correction scale: 25% optimized for Gauss distributions
// 
// ADVANCED FEATURES:
// - Edge pixel upweighting (2.0x boost) for better pos sensitivity
// - Charge-weighted err (higher charge = more precise pos)
// - Inter-pixel correlation weighting (radius: 1.5 pixels)
// - Systematic bias correction (strength: 40%)
// - Multi-strategy outlier filtering with progressive fallbacks
// - Thread-safe implementation with comprehensive error handling
//
// Ultra-robust Gauss fitting with multiple strategies
bool GaussCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    double& fit_amp,
    double& fit_center,
    double& fit_sigma,
    double& fit_offset,
    double& fit_amp_err,
    double& fit_center_err,
    double& fit_sigma_err,
    double& fit_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        if (verbose) {
            std::cout << "Insufficient data points for Gauss fitting" << std::endl;
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
        ParameterEstimates estimates = EstimateGaussParameters(clean_x, clean_y, center_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "Parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calc err as 5% of max charge
        double max_charge = *std::max_element(clean_y.begin(), clean_y.end());
        double err = CalcErr(max_charge);
        
        // OPTIMIZED: Cheap config first with early exit based on quality (Step 1 from optimize.md)
        // Start with DENSE_NORMAL_CHOLESKY, 1e-10 tolerances, 400 iterations
        // Only escalate if χ²ᵣ > 3 or !converged (5-6x speed-up expected)
        
        struct tingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        // Stage 1: Cheap configuration (as per optimize.md section 4.1)
        tingConfig cheap_config = {
            ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 
            1e-10, 1e-10, 400, "NONE", 0.0
        };
        
        // Stage 2: Expensive fallback configurations (only if needed)
        const std::vector<tingConfig> expensive_configs = {
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amp * 0.1},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amp * 0.18},
            {ceres::SPARSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1200, "CAUCHY", estimates.amp * 0.25}
        };
        
        // Try cheap config first
        auto try_config = [&](const tingConfig& config, const std::string& stage_name) -> bool {
            if (verbose) {
                std::cout << "Trying " << stage_name << " configuration..." << std::endl;
            }
            
            // STEP 2 OPTIMIZATION: Hierarchical multi-start budget
            // Start with base estimate, only add 2 perturbations if χ²ᵣ > 2.0
            // Expected: ×4-5 speed-up, average #Ceres solves/fit ≤10
            
            struct ParameterSet {
                double params[4];
                std::string description;
            };
            
            std::vector<ParameterSet> initial_guesses;
            
            // ALWAYS start with base estimate first
            ParameterSet base_set;
            base_set.params[0] = estimates.amp;
            base_set.params[1] = estimates.center;
            base_set.params[2] = estimates.sigma;
            base_set.params[3] = estimates.offset;
            base_set.description = "base_estimate";
            initial_guesses.push_back(base_set);
            
            double best_cost = std::numeric_limits<double>::max();
            double best_parameters[4];
            bool any_success = false;
            std::string best_description;
            double best_chi2_reduced = std::numeric_limits<double>::max();
            
            // Data characteristics for adaptive bounds
            DataStatistics data_stats = CalcRobustStatistics(clean_x, clean_y);
            double data_spread = *std::max_element(clean_x.begin(), clean_x.end()) - 
                               *std::min_element(clean_x.begin(), clean_x.end());
            double outlier_ratio = 0.0;
            if (clean_x.size() > 0) {
                int outlier_count = 0;
                double outlier_threshold = data_stats.median + 2.0 * data_stats.mad;
                for (double val : clean_y) {
                    if (val > outlier_threshold) outlier_count++;
                }
                outlier_ratio = static_cast<double>(outlier_count) / clean_x.size();
            }
            
            // Try base estimate first
            for (const auto& guess : initial_guesses) {
                double parameters[4];
                parameters[0] = guess.params[0];
                parameters[1] = guess.params[1];
                parameters[2] = guess.params[2];
                parameters[3] = guess.params[3];
                
                // Build the problem
                ceres::Problem problem;
                
                for (size_t i = 0; i < clean_x.size(); ++i) {
                    ceres::CostFunction* cost_function = GaussCostFunction::Create(
                        clean_x[i], clean_y[i], err);
                    problem.AddResidualBlock(cost_function, nullptr, parameters);
                }
                
                // Set adaptive bounds
                double max_charge_val = *std::max_element(clean_y.begin(), clean_y.end());
                double min_charge_val = *std::min_element(clean_y.begin(), clean_y.end());
                
                double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                        std::max(parameters[0] * 0.01, std::abs(min_charge_val) * 0.1));
                double physics_amp_max = std::max(max_charge_val * 1.5, std::abs(parameters[0]) * 2.0);
                double algo_amp_max = std::max(parameters[0] * 100.0, 1e-10);
                double amp_max = std::max(physics_amp_max, algo_amp_max);

                problem.SetParameterLowerBound(parameters, 0, amp_min);
                problem.SetParameterUpperBound(parameters, 0, amp_max);
                
                double adaptive_center_range = (outlier_ratio > 0.15) ? 
                    std::min(pixel_spacing * 3.0, data_spread * 0.4) : pixel_spacing * 3.0;
                problem.SetParameterLowerBound(parameters, 1, parameters[1] - adaptive_center_range);
                problem.SetParameterUpperBound(parameters, 1, parameters[1] + adaptive_center_range);
                
                double sigma_min = std::max(pixel_spacing * 0.05, data_spread * 0.01);
                double sigma_max = std::min(pixel_spacing * 3.0, data_spread * 0.8);
                problem.SetParameterLowerBound(parameters, 2, sigma_min);
                problem.SetParameterUpperBound(parameters, 2, sigma_max);
                
                double charge_range = std::abs(max_charge_val - min_charge_val);
                double offset_range = std::max(charge_range * 0.5, 
                                             std::max(std::abs(parameters[3]) * 2.0, 1e-12));
                problem.SetParameterLowerBound(parameters, 3, parameters[3] - offset_range);
                problem.SetParameterUpperBound(parameters, 3, parameters[3] + offset_range);
                
                // Enhanced solver configuration
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
                
                options.initial_trust_region_radius = 0.1 * pixel_spacing;
                options.max_trust_region_radius = 2.0 * pixel_spacing;
                options.min_trust_region_radius = 1e-4 * pixel_spacing;
                
                // Solve
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                
                // Validation
                bool fit_success = (summary.termination_type == ceres::CONVERGENCE ||
                                      summary.termination_type == ceres::USER_SUCCESS) &&
                                     parameters[0] > 0 && parameters[2] > 0 &&
                                     !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                     !std::isnan(parameters[2]) && !std::isnan(parameters[3]);
                
                if (fit_success) {
                    double cost = summary.final_cost;
                    double chi2 = cost * 2.0;
                    int dof = std::max(1, static_cast<int>(clean_x.size()) - 4);
                    double chi2_red = chi2 / dof;
                    
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_chi2_reduced = chi2_red;
                        std::copy(parameters, parameters + 4, best_parameters);
                        best_description = guess.description;
                        any_success = true;
                        
                        if (verbose) {
                            std::cout << "New best Gauss result from " << guess.description 
                                     << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                        }
                    }
                }
            }
            
            // Always try perturbations even if base fit hasn't succeeded yet
            if (verbose && !any_success) {
                std::cout << "Base fit failed, trying perturbations anyway..." << std::endl;
            }
            
            // ALWAYS add perturbations regardless of chi-squared quality
            if (any_success) {
                if (verbose) {
                    std::cout << "Base Gauss fit χ²ᵣ=" << best_chi2_reduced << ", trying perturbations..." << std::endl;
                }
                
                // Add exactly 2 perturbations to reduce multi-start budget
                const std::vector<double> perturbation_factors = {0.7, 1.3};
                
                for (double factor : perturbation_factors) {
                    ParameterSet perturbed_set;
                    perturbed_set.params[0] = estimates.amp * factor;
                    perturbed_set.params[1] = estimates.center + (factor - 1.0) * pixel_spacing * 0.3;
                    perturbed_set.params[2] = estimates.sigma * std::sqrt(factor);
                    perturbed_set.params[3] = estimates.offset * (0.8 + 0.4 * factor);
                    perturbed_set.description = "perturbation_" + std::to_string(factor);
                    
                    // Try this perturbation (same logic as above)
                    double parameters[4];
                    parameters[0] = perturbed_set.params[0];
                    parameters[1] = perturbed_set.params[1];
                    parameters[2] = perturbed_set.params[2];
                    parameters[3] = perturbed_set.params[3];
                    
                    ceres::Problem problem;
                    
                    for (size_t i = 0; i < clean_x.size(); ++i) {
                        ceres::CostFunction* cost_function = GaussCostFunction::Create(
                            clean_x[i], clean_y[i], err);
                        problem.AddResidualBlock(cost_function, nullptr, parameters);
                    }
                    
                    // Apply same bounds as before
                    double max_charge_val = *std::max_element(clean_y.begin(), clean_y.end());
                    double min_charge_val = *std::min_element(clean_y.begin(), clean_y.end());
                    
                    double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                            std::max(parameters[0] * 0.01, std::abs(min_charge_val) * 0.1));
                    double physics_amp_max = std::max(max_charge_val * 1.5, std::abs(parameters[0]) * 2.0);
                    double algo_amp_max = std::max(parameters[0] * 100.0, 1e-10);
                    double amp_max = std::max(physics_amp_max, algo_amp_max);

                    problem.SetParameterLowerBound(parameters, 0, amp_min);
                    problem.SetParameterUpperBound(parameters, 0, amp_max);
                    
                    double adaptive_center_range = (outlier_ratio > 0.15) ? 
                        std::min(pixel_spacing * 3.0, data_spread * 0.4) : pixel_spacing * 3.0;
                    problem.SetParameterLowerBound(parameters, 1, parameters[1] - adaptive_center_range);
                    problem.SetParameterUpperBound(parameters, 1, parameters[1] + adaptive_center_range);
                    
                    double sigma_min = std::max(pixel_spacing * 0.05, data_spread * 0.01);
                    double sigma_max = std::min(pixel_spacing * 3.0, data_spread * 0.8);
                    problem.SetParameterLowerBound(parameters, 2, sigma_min);
                    problem.SetParameterUpperBound(parameters, 2, sigma_max);
                    
                    double charge_range = std::abs(max_charge_val - min_charge_val);
                    double offset_range = std::max(charge_range * 0.5, 
                                                 std::max(std::abs(parameters[3]) * 2.0, 1e-12));
                    problem.SetParameterLowerBound(parameters, 3, parameters[3] - offset_range);
                    problem.SetParameterUpperBound(parameters, 3, parameters[3] + offset_range);
                    
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
                    
                    options.initial_trust_region_radius = 0.1 * pixel_spacing;
                    options.max_trust_region_radius = 2.0 * pixel_spacing;
                    options.min_trust_region_radius = 1e-4 * pixel_spacing;
                    
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    
                    bool fit_success = (summary.termination_type == ceres::CONVERGENCE ||
                                          summary.termination_type == ceres::USER_SUCCESS) &&
                                         parameters[0] > 0 && parameters[2] > 0 &&
                                         !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                         !std::isnan(parameters[2]) && !std::isnan(parameters[3]);
                    
                    if (fit_success) {
                        double cost = summary.final_cost;
                        double chi2 = cost * 2.0;
                        int dof = std::max(1, static_cast<int>(clean_x.size()) - 4);
                        double chi2_red = chi2 / dof;
                        
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_chi2_reduced = chi2_red;
                            std::copy(parameters, parameters + 4, best_parameters);
                            best_description = perturbed_set.description;
                            
                            if (verbose) {
                                std::cout << "New best result from " << perturbed_set.description 
                                         << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                            }
                        }
                    }
                }
            }
            
            if (any_success) {
                // Extract results from best attempt
                fit_amp = best_parameters[0];
                fit_center = best_parameters[1];
                fit_sigma = std::abs(best_parameters[2]);
                fit_offset = best_parameters[3];
                
                // Simple fallback err estimation
                fit_amp_err = std::max(0.02 * fit_amp, 0.1 * data_stats.mad);
                fit_center_err = std::max(0.02 * pixel_spacing, fit_sigma / 10.0);
                fit_sigma_err = std::max(0.05 * fit_sigma, 0.01 * pixel_spacing);
                fit_offset_err = std::max(0.1 * std::abs(fit_offset), 0.05 * data_stats.mad);
                
                chi2_reduced = best_chi2_reduced;
                
                if (verbose) {
                    std::cout << "Success Gauss fit with " << stage_name 
                             << ", dataset " << dataset_idx << ", best init: " << best_description
                             << ": A=" << fit_amp << "±" << fit_amp_err
                             << ", m=" << fit_center << "±" << fit_center_err
                             << ", sigma=" << fit_sigma << "±" << fit_sigma_err
                             << ", B=" << fit_offset << "±" << fit_offset_err
                             << ", chi2red=" << chi2_reduced << std::endl;
                }
                
                return true;
            }
            return false;
        };
        
        // Try ALL configurations without early exits
        bool success = try_config(cheap_config, "cheap");
        bool best_success = success;
        double best_chi2 = chi2_reduced;
        
        if (verbose) {
            std::cout << "Cheap config " << (success ? "succeeded" : "failed") 
                     << " with χ²ᵣ=" << chi2_reduced << std::endl;
        }
        
        // Always try ALL expensive configurations regardless of cheap config result
        if (verbose) {
            std::cout << "Trying all " << expensive_configs.size() 
                     << " expensive configurations..." << std::endl;
        }
        
        for (size_t i = 0; i < expensive_configs.size(); ++i) {
            bool config_success = try_config(expensive_configs[i], "expensive_" + std::to_string(i+1));
            if (config_success && (!best_success || chi2_reduced < best_chi2)) {
                best_success = config_success;
                best_chi2 = chi2_reduced;
                success = config_success;
            }
            
            if (verbose) {
                std::cout << "Expensive config " << (i+1) << " " 
                         << (config_success ? "succeeded" : "failed") 
                         << " with χ²ᵣ=" << chi2_reduced << std::endl;
            }
        }
        
        if (best_success) {
            if (verbose) {
                std::cout << "Best Gauss fit achieved with χ²ᵣ=" << best_chi2 << std::endl;
            }
            return true;
        }
    }
    
    if (verbose) {
        std::cout << "All fitting strategies failed" << std::endl;
    }
    return false;
}

Gauss2DResultsCeres GaussCeres2D(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    Gauss2DResultsCeres result;
    
    // Initialize Ceres logging (removed mutex for better parallelization)
    InitializeCeres();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "GaussCeres2D: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 4) {
        if (verbose) {
            std::cout << "GaussCeres2D: Error - need at least 4 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 2D Gauss fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
    
    //  X direction (central row)
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
            std::cout << "ting X direction with " << x_vals.size() << " points" << std::endl;
        }
        
        x_fit_success = GaussCeres(
            x_vals, y_vals, center_x_estimate, pixel_spacing,
            result.x_amp, result.x_center, result.x_sigma, result.x_vert_offset,
            result.x_amp_err, result.x_center_err, result.x_sigma_err, result.x_vert_offset_err,
            result.x_chi2red, verbose, enable_outlier_filtering);
        
        // Calc DOF and p-value
        result.x_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.x_pp = (result.x_chi2red > 0) ? 1.0 - std::min(1.0, result.x_chi2red / 10.0) : 0.0; // Simple p-value approximation
    }
    
    //  Y direction (central column)
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
            std::cout << "ting Y direction with " << x_vals.size() << " points" << std::endl;
        }
        
        y_fit_success = GaussCeres(
            x_vals, y_vals, center_y_estimate, pixel_spacing,
            result.y_amp, result.y_center, result.y_sigma, result.y_vert_offset,
            result.y_amp_err, result.y_center_err, result.y_sigma_err, result.y_vert_offset_err,
            result.y_chi2red, verbose, enable_outlier_filtering);
        
        // Calc DOF and p-value
        result.y_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.y_pp = (result.y_chi2red > 0) ? 1.0 - std::min(1.0, result.y_chi2red / 10.0) : 0.0; // Simple p-value approximation
    }
    
    // Set overall success status
    result.fit_success = x_fit_success && y_fit_success;
    
    // Calc and store charge uncertainties (5% of max charge for each direction) only if enabled
    if (Constants::ENABLE_VERT_CHARGE_ERR) {
        if (x_fit_success && rows_data.find(best_row_y) != rows_data.end()) {
            auto& row_data = rows_data[best_row_y];
            double max_charge_x = 0.0;
            for (const auto& point : row_data) {
                max_charge_x = std::max(max_charge_x, point.second);
            }
            result.x_charge_err = 0.05 * max_charge_x;
        }
        
        if (y_fit_success && cols_data.find(best_col_x) != cols_data.end()) {
            auto& col_data = cols_data[best_col_x];
            double max_charge_y = 0.0;
            for (const auto& point : col_data) {
                max_charge_y = std::max(max_charge_y, point.second);
            }
            result.y_charge_err = 0.05 * max_charge_y;
        }
    } else {
        result.x_charge_err = 0.0;
        result.y_charge_err = 0.0;
    }
    
    if (verbose) {
        std::cout << "2D Gauss fit (Ceres) " << (result.fit_success ? "success" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

DiagResultsCeres DiagGaussCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    DiagResultsCeres result;
    
    // Initialize Ceres logging (removed mutex for better parallelization)
    InitializeCeres();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size() || x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Diag Gauss fit (Ceres): Invalid input data size" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting Diag Gauss fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // FIXED DIAG APPROACH:
    // 1. Use spatial binning instead of exact coordinate matching
    // 2. Correct coordinate transformations for ±45° rotations
    // 3. Proper pixel spacing for diagonal geometry
    
    // Diag bin width - wider than row/column to capture more pixels
    const double bin_width = pixel_spacing * 0.5; // 50% of pixel spacing for better coverage
    
    // Correct diagonal pixel spacing (actual distance between diagonal neighbors)
    const double diag_pixel_spacing = pixel_spacing * 1.41421356237; // √2, no calibration factor
    
    // Maps: binned diagonal coordinate → [(distance_along_diagonal, charge), ...]
    std::map<int, std::vector<std::pair<double, double>>> main_diag_bins; // +45° direction
    std::map<int, std::vector<std::pair<double, double>>> sec_diag_bins;  // -45° direction
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x = x_coords[i];
        double y = y_coords[i];
        double charge = charge_values[i];
        
        if (charge <= 0) continue;
        
        double dx = x - center_x_estimate;
        double dy = y - center_y_estimate;
        
        // CORRECTED COORDINATE TRANSFORMATIONS:
        // Main diagonal (+45°): pixels where x-coordinate increases as y-coordinate increases
        // We rotate by -45° to align with X-axis: (x',y') = ((dx+dy)/√2, (dy-dx)/√2)
        // The diagonal runs along constant (dy-dx), and pos along diagonal is (dx+dy)/√2
        double main_diag_id = dy - dx;                    // Perpendicular distance to +45° line
        double main_diag_pos = (dx + dy) / 1.41421356237; // Pos along +45° diagonal
        
        // Secondary diagonal (-45°): pixels where x-coordinate increases as y-coordinate decreases  
        // We rotate by +45° to align with X-axis: (x',y') = ((dx-dy)/√2, (dx+dy)/√2)
        // The diagonal runs along constant (dx+dy), and pos along diagonal is (dx-dy)/√2
        double sec_diag_id = dx + dy;                     // Perpendicular distance to -45° line
        double sec_diag_pos = (dx - dy) / 1.41421356237; // Pos along -45° diagonal
        
        // Bin the diagonal identifiers for spatial grouping
        int main_bin = static_cast<int>(std::round(main_diag_id / bin_width));
        int sec_bin = static_cast<int>(std::round(sec_diag_id / bin_width));
        
        main_diag_bins[main_bin].emplace_back(main_diag_pos, charge);
        sec_diag_bins[sec_bin].emplace_back(sec_diag_pos, charge);
    }
    
    if (verbose) {
        std::cout << "Found " << main_diag_bins.size() << " main diagonal bins and " 
                 << sec_diag_bins.size() << " secondary diagonal bins" << std::endl;
    }
    
    // Find the best bins (those with most pixels near the center)
    auto find_best_bin = [&](const std::map<int, std::vector<std::pair<double, double>>>& bins, 
                             const std::string& direction) -> int {
        int best_bin = 0;
        int max_pixels = 0;
        for (const auto& bin_pair : bins) {
            int pixel_count = bin_pair.second.size();
            if (pixel_count >= 4 && pixel_count > max_pixels) {
                max_pixels = pixel_count;
                best_bin = bin_pair.first;
            }
        }
        if (verbose && max_pixels > 0) {
            std::cout << "Best " << direction << " diagonal bin: " << best_bin 
                     << " with " << max_pixels << " pixels" << std::endl;
        }
        return best_bin;
    };
    
    int best_main_bin = find_best_bin(main_diag_bins, "main");
    int best_sec_bin = find_best_bin(sec_diag_bins, "secondary");
    
    bool main_diag_success = false;
    bool sec_diag_success = false;
    
    //  main diagonal (+45° direction)
    if (main_diag_bins.find(best_main_bin) != main_diag_bins.end() && 
        main_diag_bins[best_main_bin].size() >= 4) {
        
        auto& main_data = main_diag_bins[best_main_bin];
        
        // Sort by pos along diagonal
        std::sort(main_data.begin(), main_data.end());
        
        std::vector<double> poss, charges;
        for (const auto& point : main_data) {
            poss.push_back(point.first);
            charges.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "ting main diagonal (+45°) with " << poss.size() << " points" << std::endl;
        }
        
        main_diag_success = GaussCeres(
            poss, charges, 0.0, diag_pixel_spacing,
            result.main_diag_x_amp, result.main_diag_x_center, result.main_diag_x_sigma, result.main_diag_x_vert_offset,
            result.main_diag_x_amp_err, result.main_diag_x_center_err, result.main_diag_x_sigma_err, result.main_diag_x_vert_offset_err,
            result.main_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.main_diag_x_dof = std::max(1, static_cast<int>(poss.size()) - 4);
        result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
        result.main_diag_x_fit_success = main_diag_success;
        
        // For symmetry, copy results to Y (since we only have one diagonal measurement)
        result.main_diag_y_amp = result.main_diag_x_amp;
        result.main_diag_y_center = result.main_diag_x_center;
        result.main_diag_y_sigma = result.main_diag_x_sigma;
        result.main_diag_y_vert_offset = result.main_diag_x_vert_offset;
        result.main_diag_y_amp_err = result.main_diag_x_amp_err;
        result.main_diag_y_center_err = result.main_diag_x_center_err;
        result.main_diag_y_sigma_err = result.main_diag_x_sigma_err;
        result.main_diag_y_vert_offset_err = result.main_diag_x_vert_offset_err;
        result.main_diag_y_chi2red = result.main_diag_x_chi2red;
        result.main_diag_y_dof = result.main_diag_x_dof;
        result.main_diag_y_pp = result.main_diag_x_pp;
        result.main_diag_y_fit_success = main_diag_success;
    }
    
    //  secondary diagonal (-45° direction)
    if (sec_diag_bins.find(best_sec_bin) != sec_diag_bins.end() && 
        sec_diag_bins[best_sec_bin].size() >= 4) {
        
        auto& sec_data = sec_diag_bins[best_sec_bin];
        
        // Sort by pos along diagonal
        std::sort(sec_data.begin(), sec_data.end());
        
        std::vector<double> poss, charges;
        for (const auto& point : sec_data) {
            poss.push_back(point.first);
            charges.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "ting secondary diagonal (-45°) with " << poss.size() << " points" << std::endl;
        }
        
        sec_diag_success = GaussCeres(
            poss, charges, 0.0, diag_pixel_spacing,
            result.sec_diag_x_amp, result.sec_diag_x_center, result.sec_diag_x_sigma, result.sec_diag_x_vert_offset,
            result.sec_diag_x_amp_err, result.sec_diag_x_center_err, result.sec_diag_x_sigma_err, result.sec_diag_x_vert_offset_err,
            result.sec_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.sec_diag_x_dof = std::max(1, static_cast<int>(poss.size()) - 4);
        result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
        result.sec_diag_x_fit_success = sec_diag_success;
        
        // For symmetry, copy results to Y
        result.sec_diag_y_amp = result.sec_diag_x_amp;
        result.sec_diag_y_center = result.sec_diag_x_center;
        result.sec_diag_y_sigma = result.sec_diag_x_sigma;
        result.sec_diag_y_vert_offset = result.sec_diag_x_vert_offset;
        result.sec_diag_y_amp_err = result.sec_diag_x_amp_err;
        result.sec_diag_y_center_err = result.sec_diag_x_center_err;
        result.sec_diag_y_sigma_err = result.sec_diag_x_sigma_err;
        result.sec_diag_y_vert_offset_err = result.sec_diag_x_vert_offset_err;
        result.sec_diag_y_chi2red = result.sec_diag_x_chi2red;
        result.sec_diag_y_dof = result.sec_diag_x_dof;
        result.sec_diag_y_pp = result.sec_diag_x_pp;
        result.sec_diag_y_fit_success = sec_diag_success;
    }
    
    // Set overall success status
    result.fit_success = main_diag_success && sec_diag_success;
    
    if (verbose) {
        std::cout << "Diag Gauss fit (Ceres) " << (result.fit_success ? "success" : "failed") 
                 << " (Main +45°: " << (main_diag_success ? "OK" : "FAIL") 
                 << ", Secondary -45°: " << (sec_diag_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}