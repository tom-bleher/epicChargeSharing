#include "3DGaussianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
// Removed mutex include - no longer needed for parallelization
#include <atomic>
#include <limits>
#include <numeric>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Thread counter for debugging (removed mutex for better parallelization)
static std::atomic<int> gGlobalCeres3DGaussianFitCounter{0};

// Use shared Google logging initialization
void InitializeCeres3DGaussian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Calculate uncertainty as 5% of max charge in neighborhood (if enabled)
double Calculate3DGaussianUncertainty(double max_charge_in_neighborhood) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Uncertainty = 5% of max charge when enabled
    double uncertainty = 0.05 * max_charge_in_neighborhood;
    if (uncertainty < Constants::MIN_UNCERTAINTY_VALUE) uncertainty = Constants::MIN_UNCERTAINTY_VALUE; // Prevent division by zero
    return uncertainty;
}

// 3D Gaussian cost function with uncertainty (5% of max charge)
// Function form: z(x,y) = A * exp(-((x - mx)^2 / (2 * σx^2) + (y - my)^2 / (2 * σy^2))) + B  
struct Gaussian3DCostFunction {
    Gaussian3DCostFunction(double x, double y, double z, double uncertainty) 
        : x_(x), y_(y), z_(z), uncertainty_(uncertainty) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = mx (X center)
        // params[2] = my (Y center)
        // params[3] = sigma_x (X width/standard deviation)
        // params[4] = sigma_y (Y width/standard deviation)
        // params[5] = B (baseline)
        
        const T& A = params[0];
        const T& mx = params[1];
        const T& my = params[2];
        const T& sigma_x = params[3];
        const T& sigma_y = params[4];
        const T& B = params[5];
        
        // Robust handling of sigma parameters (prevent division by zero)
        T safe_sigma_x = ceres::abs(sigma_x);
        T safe_sigma_y = ceres::abs(sigma_y);
        if (safe_sigma_x < T(1e-12)) {
            safe_sigma_x = T(1e-12);
        }
        if (safe_sigma_y < T(1e-12)) {
            safe_sigma_y = T(1e-12);
        }
        
        // 3D Gaussian function: z(x,y) = A * exp(-((x - mx)^2 / (2 * σx^2) + (y - my)^2 / (2 * σy^2))) + B
        T dx = x_ - mx;
        T dy = y_ - my;
        T normalized_dx = dx / safe_sigma_x;
        T normalized_dy = dy / safe_sigma_y;
        T exponent = -(normalized_dx * normalized_dx + normalized_dy * normalized_dy) / T(2.0);
        
        // Prevent numerical issues with very large/small exponents
        if (exponent < T(-200.0)) {
            exponent = T(-200.0);
        } else if (exponent > T(200.0)) {
            exponent = T(200.0);
        }
        
        T predicted = A * ceres::exp(exponent) + B;
        
        // Residual divided by uncertainty (standard weighted least squares)
        residual[0] = (predicted - T(z_)) / T(uncertainty_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double z, double uncertainty) {
        return (new ceres::AutoDiffCostFunction<Gaussian3DCostFunction, 1, 6>(
            new Gaussian3DCostFunction(x, y, z, uncertainty)));
    }
    
private:
    const double x_;
    const double y_;
    const double z_;
    const double uncertainty_;
};

// Parameter estimation structures for 3D Gaussian
struct Gaussian3DParameterEstimates {
    double amplitude;
    double center_x;
    double center_y;
    double sigma_x;
    double sigma_y;
    double baseline;
    double amplitude_err;
    double center_x_err;
    double center_y_err;
    double sigma_x_err;
    double sigma_y_err;
    double baseline_err;
    bool valid;
    int method_used;
};

// Robust statistics calculations for 3D data (reuse from existing code)
struct Data3DStatistics {
    double mean;
    double median;
    double std_dev;
    double mad; // Median Absolute Deviation
    double q25, q75; // Quartiles
    double min_val, max_val;
    double weighted_mean_x;
    double weighted_mean_y;
    double total_weight;
    double robust_center_x; // Improved X center estimate
    double robust_center_y; // Improved Y center estimate
    bool valid;
};

static Data3DStatistics CalculateRobust3DStatistics(const std::vector<double>& x_vals, 
                                                    const std::vector<double>& y_vals,
                                                    const std::vector<double>& z_vals) {
    Data3DStatistics stats;
    stats.valid = false;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.empty()) {
        return stats;
    }
    
    // Basic statistics on Z values (charge)
    stats.min_val = *std::min_element(z_vals.begin(), z_vals.end());
    stats.max_val = *std::max_element(z_vals.begin(), z_vals.end());
    
    // Mean and standard deviation
    stats.mean = std::accumulate(z_vals.begin(), z_vals.end(), 0.0) / z_vals.size();
    
    double variance = 0.0;
    for (double val : z_vals) {
        variance += (val - stats.mean) * (val - stats.mean);
    }
    stats.std_dev = std::sqrt(variance / z_vals.size());
    
    // Median and quartiles
    std::vector<double> sorted_z = z_vals;
    std::sort(sorted_z.begin(), sorted_z.end());
    
    size_t n = sorted_z.size();
    if (n % 2 == 0) {
        stats.median = (sorted_z[n/2 - 1] + sorted_z[n/2]) / 2.0;
    } else {
        stats.median = sorted_z[n/2];
    }
    
    stats.q25 = sorted_z[n/4];
    stats.q75 = sorted_z[3*n/4];
    
    // Median Absolute Deviation
    std::vector<double> abs_deviations;
    for (double val : z_vals) {
        abs_deviations.push_back(std::abs(val - stats.median));
    }
    std::sort(abs_deviations.begin(), abs_deviations.end());
    stats.mad = abs_deviations[n/2] * 1.4826;
    
    // Numerical stability safeguard
    if (!std::isfinite(stats.mad) || stats.mad < 1e-12) {
        stats.mad = (std::isfinite(stats.std_dev) && stats.std_dev > 1e-12) ?
                    stats.std_dev : 1e-12;
    }
    
    // Weighted statistics for center estimation (weight by charge value)
    stats.weighted_mean_x = 0.0;
    stats.weighted_mean_y = 0.0;
    stats.total_weight = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, z_vals[i] - stats.q25);
        if (weight > 0) {
            stats.weighted_mean_x += x_vals[i] * weight;
            stats.weighted_mean_y += y_vals[i] * weight;
            stats.total_weight += weight;
        }
    }
    
    if (stats.total_weight > 0) {
        stats.weighted_mean_x /= stats.total_weight;
        stats.weighted_mean_y /= stats.total_weight;
        stats.robust_center_x = stats.weighted_mean_x;
        stats.robust_center_y = stats.weighted_mean_y;
    } else {
        stats.weighted_mean_x = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
        stats.weighted_mean_y = std::accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();
        stats.robust_center_x = stats.weighted_mean_x;
        stats.robust_center_y = stats.weighted_mean_y;
    }
    
    stats.valid = true;
    return stats;
}

// Parameter estimation for 3D Gaussian distributions
static Gaussian3DParameterEstimates Estimate3DGaussianParameters(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    Gaussian3DParameterEstimates estimates;
    estimates.valid = false;
    estimates.method_used = 0;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 6) {
        return estimates;
    }
    
    Data3DStatistics stats = CalculateRobust3DStatistics(x_vals, y_vals, z_vals);
    if (!stats.valid) {
        return estimates;
    }
    
    if (verbose) {
        std::cout << "3D Gaussian data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", weighted_mean_x=" << stats.weighted_mean_x 
                 << ", weighted_mean_y=" << stats.weighted_mean_y << std::endl;
    }
    
    // Method 1: Physics-based estimation for charge distributions
    estimates.center_x = stats.weighted_mean_x;
    estimates.center_y = stats.weighted_mean_y;
    estimates.baseline = std::min(stats.min_val, stats.q25);
    estimates.amplitude = stats.max_val - estimates.baseline;
    
    // For 3D Gaussian: sigma estimation based on charge spread in both directions
    double distance_spread_x = 0.0;
    double distance_spread_y = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, z_vals[i] - estimates.baseline);
        if (weight > 0.1 * estimates.amplitude) {
            double dx = x_vals[i] - estimates.center_x;
            double dy = y_vals[i] - estimates.center_y;
            distance_spread_x += weight * dx * dx;
            distance_spread_y += weight * dy * dy;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0) {
        // For Gaussian, sigma is the standard deviation
        estimates.sigma_x = std::sqrt(distance_spread_x / weight_sum);
        estimates.sigma_y = std::sqrt(distance_spread_y / weight_sum);
    } else {
        estimates.sigma_x = pixel_spacing * 0.5; // Default for Gaussian (narrower than Lorentzian)
        estimates.sigma_y = pixel_spacing * 0.5;
    }
    
    // Apply physics-based bounds (Gaussian has narrower profiles than Lorentzian)
    estimates.sigma_x = std::max(pixel_spacing * 0.2, std::min(pixel_spacing * 2.0, estimates.sigma_x));
    estimates.sigma_y = std::max(pixel_spacing * 0.2, std::min(pixel_spacing * 2.0, estimates.sigma_y));
    estimates.amplitude = std::max(estimates.amplitude, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amplitude > 0 && estimates.sigma_x > 0 && estimates.sigma_y > 0 &&
        !std::isnan(estimates.center_x) && !std::isnan(estimates.center_y) && 
        !std::isnan(estimates.amplitude) && !std::isnan(estimates.sigma_x) && 
        !std::isnan(estimates.sigma_y) && !std::isnan(estimates.baseline)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "3D Gaussian Method 1 (Physics-based): A=" << estimates.amplitude 
                     << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                     << ", sigma_x=" << estimates.sigma_x << ", sigma_y=" << estimates.sigma_y
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center_x = stats.robust_center_x;
    estimates.center_y = stats.robust_center_y;
    estimates.baseline = stats.q25;
    estimates.amplitude = stats.q75 - stats.q25;
    estimates.sigma_x = std::max(stats.mad, pixel_spacing * 0.4);
    estimates.sigma_y = std::max(stats.mad, pixel_spacing * 0.4);
    
    if (estimates.amplitude > 0 && estimates.sigma_x > 0 && estimates.sigma_y > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "3D Gaussian Method 2 (Robust statistical): A=" << estimates.amplitude 
                     << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                     << ", sigma_x=" << estimates.sigma_x << ", sigma_y=" << estimates.sigma_y
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center_x = center_x_estimate;
    estimates.center_y = center_y_estimate;
    estimates.baseline = 0.0;
    estimates.amplitude = stats.max_val;
    estimates.sigma_x = pixel_spacing * 0.5;
    estimates.sigma_y = pixel_spacing * 0.5;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "3D Gaussian Method 3 (Conservative fallback): A=" << estimates.amplitude 
                 << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                 << ", sigma_x=" << estimates.sigma_x << ", sigma_y=" << estimates.sigma_y
                 << ", B=" << estimates.baseline << std::endl;
    }
    
    return estimates;
}

// Outlier filtering for 3D Gaussian fitting (adapted from 3D Lorentzian version)
static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> Filter3DGaussianOutliers(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double sigma_threshold = 2.5,
    bool verbose = false) {
    
    std::vector<double> filtered_x, filtered_y, filtered_z;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 6) {
        return std::make_tuple(filtered_x, filtered_y, filtered_z);
    }
    
    Data3DStatistics stats = CalculateRobust3DStatistics(x_vals, y_vals, z_vals);
    if (!stats.valid) {
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    // Use MAD-based outlier detection
    double outlier_threshold = stats.median + sigma_threshold * stats.mad;
    double lower_threshold = stats.median - sigma_threshold * stats.mad;
    
    int outliers_removed = 0;
    for (size_t i = 0; i < z_vals.size(); ++i) {
        if (z_vals[i] >= lower_threshold && z_vals[i] <= outlier_threshold) {
            filtered_x.push_back(x_vals[i]);
            filtered_y.push_back(y_vals[i]);
            filtered_z.push_back(z_vals[i]);
        } else {
            outliers_removed++;
        }
    }
    
    // Use lenient filtering if too many outliers removed
    if (filtered_x.size() < x_vals.size() / 2) {
        if (verbose) {
            std::cout << "Too many 3D Gaussian outliers detected (" << outliers_removed 
                     << "), using lenient filtering" << std::endl;
        }
        
        filtered_x.clear();
        filtered_y.clear();
        filtered_z.clear();
        
        double extreme_threshold = stats.median + 5.0 * stats.mad;
        double extreme_lower = stats.median - 5.0 * stats.mad;
        
        for (size_t i = 0; i < z_vals.size(); ++i) {
            if (z_vals[i] >= extreme_lower && z_vals[i] <= extreme_threshold) {
                filtered_x.push_back(x_vals[i]);
                filtered_y.push_back(y_vals[i]);
                filtered_z.push_back(z_vals[i]);
            }
        }
    }
    
    if (filtered_x.size() < 6) {
        if (verbose) {
            std::cout << "Warning: After 3D Gaussian outlier filtering, only " << filtered_x.size() 
                     << " points remain" << std::endl;
        }
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " 3D Gaussian outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_tuple(filtered_x, filtered_y, filtered_z);
}

// Core 3D Gaussian fitting function using Ceres Solver
bool Fit3DGaussianCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    double& fit_amplitude,
    double& fit_center_x,
    double& fit_center_y,
    double& fit_sigma_x,
    double& fit_sigma_y,
    double& fit_vertical_offset,
    double& fit_amplitude_err,
    double& fit_center_x_err,
    double& fit_center_y_err,
    double& fit_sigma_x_err,
    double& fit_sigma_y_err,
    double& fit_vertical_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 6) {
        if (verbose) {
            std::cout << "Insufficient data points for 3D Gaussian fitting" << std::endl;
        }
        return false;
    }
    
    // Multiple outlier filtering strategies
    std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> filtered_datasets;
    
    if (enable_outlier_filtering) {
        auto conservative_data = Filter3DGaussianOutliers(x_vals, y_vals, z_vals, 2.5, verbose);
        if (std::get<0>(conservative_data).size() >= 6) {
            filtered_datasets.push_back(conservative_data);
        }
        
        auto lenient_data = Filter3DGaussianOutliers(x_vals, y_vals, z_vals, 3.0, verbose);
        if (std::get<0>(lenient_data).size() >= 6) {
            filtered_datasets.push_back(lenient_data);
        }
    }
    
    // Always include original data as fallback
    filtered_datasets.push_back(std::make_tuple(x_vals, y_vals, z_vals));
    
    if (verbose) {
        std::cout << "3D Gaussian outlier filtering " << (enable_outlier_filtering ? "enabled" : "disabled") 
                 << ", testing " << filtered_datasets.size() << " datasets" << std::endl;
    }
    
    // Try each filtered dataset
    for (size_t dataset_idx = 0; dataset_idx < filtered_datasets.size(); ++dataset_idx) {
        std::vector<double> clean_x = std::get<0>(filtered_datasets[dataset_idx]);
        std::vector<double> clean_y = std::get<1>(filtered_datasets[dataset_idx]);
        std::vector<double> clean_z = std::get<2>(filtered_datasets[dataset_idx]);
        
        if (clean_x.size() < 6) continue;
        
        if (verbose) {
            std::cout << "Trying 3D Gaussian dataset " << dataset_idx << " with " << clean_x.size() << " points" << std::endl;
        }
        
        // Get parameter estimates
        Gaussian3DParameterEstimates estimates = Estimate3DGaussianParameters(clean_x, clean_y, clean_z, center_x_estimate, center_y_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "3D Gaussian parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calculate uncertainty as 5% of max charge
        double max_charge = *std::max_element(clean_z.begin(), clean_z.end());
        double uncertainty = Calculate3DGaussianUncertainty(max_charge);
        
        // OPTIMIZED: Cheap config first with early exit based on quality (Step 1 from optimize.md)
        struct Gaussian3DFittingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        // Stage 1: Cheap configuration (prefer DENSE_NORMAL_CHOLESKY for 3D)
        Gaussian3DFittingConfig cheap_config = {
            ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 
            1e-10, 1e-10, 400, "NONE", 0.0
        };
        
        // Stage 2: Expensive fallback configurations
        const std::vector<Gaussian3DFittingConfig> expensive_configs = {
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amplitude * 0.1},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amplitude * 0.16}
        };
        
        // Try cheap config first
        auto try_config = [&](const Gaussian3DFittingConfig& config, const std::string& stage_name) -> bool {
            if (verbose) {
                std::cout << "Trying 3D Gaussian " << stage_name << " configuration..." << std::endl;
            }
            // STEP 2 OPTIMIZATION: Hierarchical multi-start budget for 3D Gaussian
            // Start with base estimate, only add 2 perturbations if χ²ᵣ > 0.5
            // Expected: ×4-5 speed-up, average #Ceres solves/fit ≤10
            
            struct ParameterSet {
                double params[6];
                std::string description;
            };
            
            std::vector<ParameterSet> initial_guesses;
            
            // ALWAYS start with base estimate first
            ParameterSet base_set;
            base_set.params[0] = estimates.amplitude;
            base_set.params[1] = estimates.center_x;
            base_set.params[2] = estimates.center_y;
            base_set.params[3] = estimates.sigma_x;
            base_set.params[4] = estimates.sigma_y;
            base_set.params[5] = estimates.baseline;
            base_set.description = "base_estimate";
            initial_guesses.push_back(base_set);
            
            double best_cost = std::numeric_limits<double>::max();
            double best_parameters[6];
            bool any_success = false;
            std::string best_description;
            double best_chi2_reduced = std::numeric_limits<double>::max();
            
            // Data characteristics for adaptive bounds
            Data3DStatistics data_stats = CalculateRobust3DStatistics(clean_x, clean_y, clean_z);
            double data_spread_x = *std::max_element(clean_x.begin(), clean_x.end()) - 
                                 *std::min_element(clean_x.begin(), clean_x.end());
            double data_spread_y = *std::max_element(clean_y.begin(), clean_y.end()) - 
                                 *std::min_element(clean_y.begin(), clean_y.end());
            double outlier_ratio = 0.0;
            if (clean_x.size() > 0) {
                int outlier_count = 0;
                double outlier_threshold = data_stats.median + 2.0 * data_stats.mad;
                for (double val : clean_z) {
                    if (val > outlier_threshold) outlier_count++;
                }
                outlier_ratio = static_cast<double>(outlier_count) / clean_x.size();
            }
            
            // Try base estimate first
            for (const auto& guess : initial_guesses) {
                double parameters[6];
                parameters[0] = guess.params[0];
                parameters[1] = guess.params[1];
                parameters[2] = guess.params[2];
                parameters[3] = guess.params[3];
                parameters[4] = guess.params[4];
                parameters[5] = guess.params[5];
            
                ceres::Problem problem;
                
                for (size_t i = 0; i < clean_x.size(); ++i) {
                    ceres::CostFunction* cost_function = Gaussian3DCostFunction::Create(
                        clean_x[i], clean_y[i], clean_z[i], uncertainty);
                    problem.AddResidualBlock(cost_function, nullptr, parameters);
                }
                
                // Set adaptive bounds
                double max_charge_val = *std::max_element(clean_z.begin(), clean_z.end());
                double min_charge_val = *std::min_element(clean_z.begin(), clean_z.end());
                
                double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                        std::max(parameters[0] * 0.01, std::abs(min_charge_val) * 0.1));
                double physics_amp_max = std::max(max_charge_val * 1.5, std::abs(parameters[0]) * 2.0);
                double algo_amp_max = std::max(parameters[0] * 100.0, 1e-10);
                double amp_max = std::max(physics_amp_max, algo_amp_max);

                problem.SetParameterLowerBound(parameters, 0, amp_min);
                problem.SetParameterUpperBound(parameters, 0, amp_max);
                
                double adaptive_center_range_x = (outlier_ratio > 0.15) ? 
                    std::min(pixel_spacing * 3.0, data_spread_x * 0.4) : pixel_spacing * 3.0;
                double adaptive_center_range_y = (outlier_ratio > 0.15) ? 
                    std::min(pixel_spacing * 3.0, data_spread_y * 0.4) : pixel_spacing * 3.0;
                    
                problem.SetParameterLowerBound(parameters, 1, parameters[1] - adaptive_center_range_x);
                problem.SetParameterUpperBound(parameters, 1, parameters[1] + adaptive_center_range_x);
                problem.SetParameterLowerBound(parameters, 2, parameters[2] - adaptive_center_range_y);
                problem.SetParameterUpperBound(parameters, 2, parameters[2] + adaptive_center_range_y);
                
                double sigma_min_x = std::max(pixel_spacing * 0.05, data_spread_x * 0.01);
                double sigma_max_x = std::min(pixel_spacing * 3.0, data_spread_x * 0.8);
                double sigma_min_y = std::max(pixel_spacing * 0.05, data_spread_y * 0.01);
                double sigma_max_y = std::min(pixel_spacing * 3.0, data_spread_y * 0.8);
                
                problem.SetParameterLowerBound(parameters, 3, sigma_min_x);
                problem.SetParameterUpperBound(parameters, 3, sigma_max_x);
                problem.SetParameterLowerBound(parameters, 4, sigma_min_y);
                problem.SetParameterUpperBound(parameters, 4, sigma_max_y);
                
                double charge_range = std::abs(max_charge_val - min_charge_val);
                double baseline_range = std::max(charge_range * 0.5, 
                                               std::max(std::abs(parameters[5]) * 2.0, 1e-12));
                problem.SetParameterLowerBound(parameters, 5, parameters[5] - baseline_range);
                problem.SetParameterUpperBound(parameters, 5, parameters[5] + baseline_range);
            
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
            
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                
                bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                                      summary.termination_type == ceres::USER_SUCCESS) &&
                                     parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 &&
                                     !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                     !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                                     !std::isnan(parameters[4]) && !std::isnan(parameters[5]);
                
                if (fit_successful) {
                    double cost = summary.final_cost;
                    double chi2 = cost * 2.0;
                    int dof = std::max(1, static_cast<int>(clean_x.size()) - 6);
                    double chi2_red = chi2 / dof;
                    
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_chi2_reduced = chi2_red;
                        std::copy(parameters, parameters + 6, best_parameters);
                        best_description = guess.description;
                        any_success = true;
                        
                        if (verbose) {
                            std::cout << "New best 3D Gaussian result from " << guess.description 
                                     << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                        }
                    }
                }
            }
            
            // HIERARCHICAL DECISION: Only add perturbations if χ²ᵣ > 0.5
            // Updated threshold based on actual data: χ²ᵣ > 0.5 (around median-to-P75 range)
            if (any_success && best_chi2_reduced > 0.5) {
                if (verbose) {
                    std::cout << "Base 3D Gaussian fit χ²ᵣ=" << best_chi2_reduced << " > 0.5, trying 2 perturbations..." << std::endl;
                }
                
                // Add exactly 2 perturbations for 3D case
                const std::vector<double> perturbation_factors = {0.7, 1.3};
                
                for (double factor : perturbation_factors) {
                    ParameterSet perturbed_set;
                    perturbed_set.params[0] = estimates.amplitude * factor;
                    perturbed_set.params[1] = estimates.center_x + (factor - 1.0) * pixel_spacing * 0.3;
                    perturbed_set.params[2] = estimates.center_y + (factor - 1.0) * pixel_spacing * 0.3;
                    perturbed_set.params[3] = estimates.sigma_x * std::sqrt(factor);
                    perturbed_set.params[4] = estimates.sigma_y * std::sqrt(factor);
                    perturbed_set.params[5] = estimates.baseline * (0.8 + 0.4 * factor);
                    perturbed_set.description = "3d_gaussian_perturbation_" + std::to_string(factor);
                    
                    // Try this perturbation (same logic as above but abbreviated for space)
                    double parameters[6];
                    std::copy(perturbed_set.params, perturbed_set.params + 6, parameters);
                    
                    ceres::Problem problem;
                    
                    for (size_t i = 0; i < clean_x.size(); ++i) {
                        ceres::CostFunction* cost_function = Gaussian3DCostFunction::Create(
                            clean_x[i], clean_y[i], clean_z[i], uncertainty);
                        problem.AddResidualBlock(cost_function, nullptr, parameters);
                    }
                    
                    // Apply same bounds as before (abbreviated)
                    double max_charge_val = *std::max_element(clean_z.begin(), clean_z.end());
                    double min_charge_val = *std::min_element(clean_z.begin(), clean_z.end());
                    
                    double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                            std::max(parameters[0] * 0.01, std::abs(min_charge_val) * 0.1));
                    double amp_max = std::max(max_charge_val * 1.5, std::max(parameters[0] * 100.0, 1e-10));
                    problem.SetParameterLowerBound(parameters, 0, amp_min);
                    problem.SetParameterUpperBound(parameters, 0, amp_max);
                    
                    // Redeclare adaptive ranges for perturbation section
                    double adaptive_center_range_x = (outlier_ratio > 0.15) ? 
                        std::min(pixel_spacing * 3.0, data_spread_x * 0.4) : pixel_spacing * 3.0;
                    double adaptive_center_range_y = (outlier_ratio > 0.15) ? 
                        std::min(pixel_spacing * 3.0, data_spread_y * 0.4) : pixel_spacing * 3.0;
                    
                    problem.SetParameterLowerBound(parameters, 1, parameters[1] - adaptive_center_range_x);
                    problem.SetParameterUpperBound(parameters, 1, parameters[1] + adaptive_center_range_x);
                    problem.SetParameterLowerBound(parameters, 2, parameters[2] - adaptive_center_range_y);
                    problem.SetParameterUpperBound(parameters, 2, parameters[2] + adaptive_center_range_y);
                    
                    double sigma_min_x = std::max(pixel_spacing * 0.05, data_spread_x * 0.01);
                    double sigma_max_x = std::min(pixel_spacing * 3.0, data_spread_x * 0.8);
                    double sigma_min_y = std::max(pixel_spacing * 0.05, data_spread_y * 0.01);
                    double sigma_max_y = std::min(pixel_spacing * 3.0, data_spread_y * 0.8);
                    
                    problem.SetParameterLowerBound(parameters, 3, sigma_min_x);
                    problem.SetParameterUpperBound(parameters, 3, sigma_max_x);
                    problem.SetParameterLowerBound(parameters, 4, sigma_min_y);
                    problem.SetParameterUpperBound(parameters, 4, sigma_max_y);
                    
                    double charge_range = std::abs(max_charge_val - min_charge_val);
                    double baseline_range = std::max(charge_range * 0.5, 
                                                   std::max(std::abs(parameters[5]) * 2.0, 1e-12));
                    problem.SetParameterLowerBound(parameters, 5, parameters[5] - baseline_range);
                    problem.SetParameterUpperBound(parameters, 5, parameters[5] + baseline_range);
                    
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
                    
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    
                    bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                                          summary.termination_type == ceres::USER_SUCCESS) &&
                                         parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 &&
                                         !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                         !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                                         !std::isnan(parameters[4]) && !std::isnan(parameters[5]);
                    
                    if (fit_successful) {
                        double cost = summary.final_cost;
                        double chi2 = cost * 2.0;
                        int dof = std::max(1, static_cast<int>(clean_x.size()) - 6);
                        double chi2_red = chi2 / dof;
                        
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_chi2_reduced = chi2_red;
                            std::copy(parameters, parameters + 6, best_parameters);
                            best_description = perturbed_set.description;
                            
                            if (verbose) {
                                std::cout << "New best result from " << perturbed_set.description 
                                         << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                            }
                        }
                    }
                }
            } else if (verbose && any_success) {
                std::cout << "Base 3D Gaussian fit χ²ᵣ=" << best_chi2_reduced << " ≤ 0.5, skipping perturbations (hierarchical multi-start)" << std::endl;
            }
            
            if (any_success) {
                // Extract results from best attempt
                fit_amplitude = best_parameters[0];
                fit_center_x = best_parameters[1];
                fit_center_y = best_parameters[2];
                fit_sigma_x = std::abs(best_parameters[3]);
                fit_sigma_y = std::abs(best_parameters[4]);
                fit_vertical_offset = best_parameters[5];
                
                // Simple fallback uncertainty estimation
                Data3DStatistics data_stats = CalculateRobust3DStatistics(clean_x, clean_y, clean_z);
                fit_amplitude_err = std::max(0.02 * fit_amplitude, 0.1 * data_stats.mad);
                fit_center_x_err = std::max(0.02 * pixel_spacing, fit_sigma_x / 10.0);
                fit_center_y_err = std::max(0.02 * pixel_spacing, fit_sigma_y / 10.0);
                fit_sigma_x_err = std::max(0.05 * fit_sigma_x, 0.01 * pixel_spacing);
                fit_sigma_y_err = std::max(0.05 * fit_sigma_y, 0.01 * pixel_spacing);
                fit_vertical_offset_err = std::max(0.1 * std::abs(fit_vertical_offset), 0.05 * data_stats.mad);
                
                chi2_reduced = best_chi2_reduced;
                
                if (verbose) {
                    std::cout << "Successful 3D Gaussian fit with " << stage_name 
                             << ", dataset " << dataset_idx << ", best init: " << best_description
                             << ": A=" << fit_amplitude << "±" << fit_amplitude_err
                             << ", mx=" << fit_center_x << "±" << fit_center_x_err
                             << ", my=" << fit_center_y << "±" << fit_center_y_err
                             << ", sigma_x=" << fit_sigma_x << "±" << fit_sigma_x_err
                             << ", sigma_y=" << fit_sigma_y << "±" << fit_sigma_y_err
                             << ", B=" << fit_vertical_offset << "±" << fit_vertical_offset_err
                             << ", chi2red=" << chi2_reduced << std::endl;
                }
                
                return true;
            }
            return false;
        };
        
        // OPTIMIZED SOLVER ESCALATION: Try cheap config first, escalate only if needed
        bool success = try_config(cheap_config, "cheap");
        
        // Early exit quality check (as per optimize.md section 4.1)
        // Updated threshold based on actual data: χ²ᵣ ≤ 0.7 (around P75 of real distribution)
        if (success && chi2_reduced <= 0.7) {
            if (verbose) {
                std::cout << "Early exit: cheap 3D Gaussian config succeeded with χ²ᵣ=" << chi2_reduced << " ≤ 0.7" << std::endl;
            }
            return true;
        }
        
        // Escalate to expensive configs only if cheap failed or quality poor
        if (!success || chi2_reduced > 0.7) {
            if (verbose) {
                std::cout << "Escalating to expensive 3D Gaussian configurations (χ²ᵣ=" << chi2_reduced 
                         << ", success=" << success << ")" << std::endl;
            }
            
            for (size_t i = 0; i < expensive_configs.size(); ++i) {
                success = try_config(expensive_configs[i], "expensive_" + std::to_string(i+1));
                if (success && chi2_reduced <= 3.0) {
                    return true;
                }
            }
        } else {
            return true; // cheap config succeeded with good quality
        }
    }
    
    if (verbose) {
        std::cout << "All 3D Gaussian fitting strategies failed" << std::endl;
    }
    return false;
}

GaussianFit3DResultsCeres Fit3DGaussianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    GaussianFit3DResultsCeres result;
    
    // Initialize Ceres logging (removed mutex for better parallelization)
    InitializeCeres3DGaussian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit3DGaussianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 6) {
        if (verbose) {
            std::cout << "Fit3DGaussianCeres: Error - need at least 6 data points for 3D fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 3D Gaussian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // Store input data for ROOT analysis
    result.x_coords = x_coords;
    result.y_coords = y_coords;
    result.charge_values = charge_values;
    
    // Create charge errors if vertical uncertainties are enabled
    if (Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        double max_charge = *std::max_element(charge_values.begin(), charge_values.end());
        double charge_uncertainty = 0.05 * max_charge;
        result.charge_uncertainty = charge_uncertainty;
        
        // Fill charge errors vector with same uncertainty for all points
        result.charge_errors.clear();
        result.charge_errors.resize(charge_values.size(), charge_uncertainty);
    } else {
        result.charge_uncertainty = 0.0;
        result.charge_errors.clear();
        result.charge_errors.resize(charge_values.size(), 1.0); // Uniform weighting
    }
    
    // Perform 3D Gaussian surface fitting
    bool fit_success = Fit3DGaussianCeres(
        x_coords, y_coords, charge_values, center_x_estimate, center_y_estimate, pixel_spacing,
        result.amplitude, result.center_x, result.center_y, result.sigma_x, result.sigma_y, result.vertical_offset,
        result.amplitude_err, result.center_x_err, result.center_y_err, result.sigma_x_err, result.sigma_y_err, result.vertical_offset_err,
        result.chi2red, verbose, enable_outlier_filtering);
    
    // Calculate DOF and p-value
    result.dof = std::max(1, static_cast<int>(x_coords.size()) - 6);
    result.pp = (result.chi2red > 0) ? 1.0 - std::min(1.0, result.chi2red / 10.0) : 0.0;
    
    // Set overall success status
    result.fit_successful = fit_success;
    
    if (verbose) {
        std::cout << "3D Gaussian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") << std::endl;
    }
    
    return result;
} 