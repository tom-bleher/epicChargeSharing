#include "3DLorentzianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <mutex>
#include <atomic>
#include <limits>
#include <numeric>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Thread-safe mutex for Ceres operations
static std::mutex gCeres3DLorentzianFitMutex;
static std::atomic<int> gGlobalCeres3DLorentzianFitCounter{0};

// Use shared Google logging initialization
void InitializeCeres3DLorentzian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Calculate uncertainty as 5% of max charge in neighborhood (if enabled)
double Calculate3DLorentzianUncertainty(double max_charge_in_neighborhood) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Uncertainty = 5% of max charge when enabled
    double uncertainty = 0.05 * max_charge_in_neighborhood;
    if (uncertainty < Constants::MIN_UNCERTAINTY_VALUE) uncertainty = Constants::MIN_UNCERTAINTY_VALUE; // Prevent division by zero
    return uncertainty;
}

// 3D Lorentzian cost function with uncertainty (5% of max charge)
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2) + B  
struct Lorentzian3DCostFunction {
    Lorentzian3DCostFunction(double x, double y, double z, double uncertainty) 
        : x_(x), y_(y), z_(z), uncertainty_(uncertainty) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = mx (X center)
        // params[2] = my (Y center)
        // params[3] = gamma_x (X width/HWHM)
        // params[4] = gamma_y (Y width/HWHM)
        // params[5] = B (baseline)
        
        const T& A = params[0];
        const T& mx = params[1];
        const T& my = params[2];
        const T& gamma_x = params[3];
        const T& gamma_y = params[4];
        const T& B = params[5];
        
        // Robust handling of gamma parameters (prevent division by zero)
        T safe_gamma_x = ceres::abs(gamma_x);
        T safe_gamma_y = ceres::abs(gamma_y);
        if (safe_gamma_x < T(1e-12)) {
            safe_gamma_x = T(1e-12);
        }
        if (safe_gamma_y < T(1e-12)) {
            safe_gamma_y = T(1e-12);
        }
        
        // 3D Lorentzian function: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2) + B
        T dx = x_ - mx;
        T dy = y_ - my;
        T normalized_dx = dx / safe_gamma_x;
        T normalized_dy = dy / safe_gamma_y;
        T denominator = T(1.0) + normalized_dx * normalized_dx + normalized_dy * normalized_dy;
        
        // Prevent numerical issues with very small denominators
        if (denominator < T(1e-12)) {
            denominator = T(1e-12);
        }
        
        T predicted = A / denominator + B;
        
        // Residual divided by uncertainty (standard weighted least squares)
        residual[0] = (predicted - T(z_)) / T(uncertainty_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double z, double uncertainty) {
        return (new ceres::AutoDiffCostFunction<Lorentzian3DCostFunction, 1, 6>(
            new Lorentzian3DCostFunction(x, y, z, uncertainty)));
    }
    
private:
    const double x_;
    const double y_;
    const double z_;
    const double uncertainty_;
};

// Parameter estimation structures for 3D Lorentzian
struct Lorentzian3DParameterEstimates {
    double amplitude;
    double center_x;
    double center_y;
    double gamma_x;
    double gamma_y;
    double baseline;
    double amplitude_err;
    double center_x_err;
    double center_y_err;
    double gamma_x_err;
    double gamma_y_err;
    double baseline_err;
    bool valid;
    int method_used;
};

// Robust statistics calculations for 3D data
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

Data3DStatistics CalculateRobust3DStatistics(const std::vector<double>& x_vals, 
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

// Parameter estimation for 3D Lorentzian distributions
Lorentzian3DParameterEstimates Estimate3DLorentzianParameters(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    Lorentzian3DParameterEstimates estimates;
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
        std::cout << "3D Lorentzian data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", weighted_mean_x=" << stats.weighted_mean_x 
                 << ", weighted_mean_y=" << stats.weighted_mean_y << std::endl;
    }
    
    // Method 1: Physics-based estimation for charge distributions
    estimates.center_x = stats.weighted_mean_x;
    estimates.center_y = stats.weighted_mean_y;
    estimates.baseline = std::min(stats.min_val, stats.q25);
    estimates.amplitude = stats.max_val - estimates.baseline;
    
    // For 3D Lorentzian: gamma estimation based on charge spread in both directions
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
        // For Lorentzian, gamma ≈ sqrt(2*sigma^2) where sigma is from Gaussian equivalent
        estimates.gamma_x = std::sqrt(2.0 * distance_spread_x / weight_sum);
        estimates.gamma_y = std::sqrt(2.0 * distance_spread_y / weight_sum);
    } else {
        estimates.gamma_x = pixel_spacing * 0.7; // Larger default for Lorentzian
        estimates.gamma_y = pixel_spacing * 0.7;
    }
    
    // Apply physics-based bounds (Lorentzian has wider tails)
    estimates.gamma_x = std::max(pixel_spacing * 0.3, std::min(pixel_spacing * 3.0, estimates.gamma_x));
    estimates.gamma_y = std::max(pixel_spacing * 0.3, std::min(pixel_spacing * 3.0, estimates.gamma_y));
    estimates.amplitude = std::max(estimates.amplitude, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amplitude > 0 && estimates.gamma_x > 0 && estimates.gamma_y > 0 &&
        !std::isnan(estimates.center_x) && !std::isnan(estimates.center_y) && 
        !std::isnan(estimates.amplitude) && !std::isnan(estimates.gamma_x) && 
        !std::isnan(estimates.gamma_y) && !std::isnan(estimates.baseline)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "3D Lorentzian Method 1 (Physics-based): A=" << estimates.amplitude 
                     << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                     << ", gamma_x=" << estimates.gamma_x << ", gamma_y=" << estimates.gamma_y
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center_x = stats.robust_center_x;
    estimates.center_y = stats.robust_center_y;
    estimates.baseline = stats.q25;
    estimates.amplitude = stats.q75 - stats.q25;
    estimates.gamma_x = std::max(stats.mad, pixel_spacing * 0.5);
    estimates.gamma_y = std::max(stats.mad, pixel_spacing * 0.5);
    
    if (estimates.amplitude > 0 && estimates.gamma_x > 0 && estimates.gamma_y > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "3D Lorentzian Method 2 (Robust statistical): A=" << estimates.amplitude 
                     << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                     << ", gamma_x=" << estimates.gamma_x << ", gamma_y=" << estimates.gamma_y
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center_x = center_x_estimate;
    estimates.center_y = center_y_estimate;
    estimates.baseline = 0.0;
    estimates.amplitude = stats.max_val;
    estimates.gamma_x = pixel_spacing * 0.7;
    estimates.gamma_y = pixel_spacing * 0.7;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "3D Lorentzian Method 3 (Conservative fallback): A=" << estimates.amplitude 
                 << ", mx=" << estimates.center_x << ", my=" << estimates.center_y
                 << ", gamma_x=" << estimates.gamma_x << ", gamma_y=" << estimates.gamma_y
                 << ", B=" << estimates.baseline << std::endl;
    }
    
    return estimates;
}

// Outlier filtering for 3D Lorentzian fitting (adapted from 2D version)
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> Filter3DLorentzianOutliers(
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
            std::cout << "Too many 3D Lorentzian outliers detected (" << outliers_removed 
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
            std::cout << "Warning: After 3D Lorentzian outlier filtering, only " << filtered_x.size() 
                     << " points remain" << std::endl;
        }
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " 3D Lorentzian outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_tuple(filtered_x, filtered_y, filtered_z);
}

// Core 3D Lorentzian fitting function using Ceres Solver
bool Fit3DLorentzianCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    double& fit_amplitude,
    double& fit_center_x,
    double& fit_center_y,
    double& fit_gamma_x,
    double& fit_gamma_y,
    double& fit_vertical_offset,
    double& fit_amplitude_err,
    double& fit_center_x_err,
    double& fit_center_y_err,
    double& fit_gamma_x_err,
    double& fit_gamma_y_err,
    double& fit_vertical_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 6) {
        if (verbose) {
            std::cout << "Insufficient data points for 3D Lorentzian fitting" << std::endl;
        }
        return false;
    }
    
    // Multiple outlier filtering strategies
    std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> filtered_datasets;
    
    if (enable_outlier_filtering) {
        auto conservative_data = Filter3DLorentzianOutliers(x_vals, y_vals, z_vals, 2.5, verbose);
        if (std::get<0>(conservative_data).size() >= 6) {
            filtered_datasets.push_back(conservative_data);
        }
        
        auto lenient_data = Filter3DLorentzianOutliers(x_vals, y_vals, z_vals, 3.0, verbose);
        if (std::get<0>(lenient_data).size() >= 6) {
            filtered_datasets.push_back(lenient_data);
        }
    }
    
    // Always include original data as fallback
    filtered_datasets.push_back(std::make_tuple(x_vals, y_vals, z_vals));
    
    if (verbose) {
        std::cout << "3D Lorentzian outlier filtering " << (enable_outlier_filtering ? "enabled" : "disabled") 
                 << ", testing " << filtered_datasets.size() << " datasets" << std::endl;
    }
    
    // Try each filtered dataset
    for (size_t dataset_idx = 0; dataset_idx < filtered_datasets.size(); ++dataset_idx) {
        std::vector<double> clean_x = std::get<0>(filtered_datasets[dataset_idx]);
        std::vector<double> clean_y = std::get<1>(filtered_datasets[dataset_idx]);
        std::vector<double> clean_z = std::get<2>(filtered_datasets[dataset_idx]);
        
        if (clean_x.size() < 6) continue;
        
        if (verbose) {
            std::cout << "Trying 3D Lorentzian dataset " << dataset_idx << " with " << clean_x.size() << " points" << std::endl;
        }
        
        // Get parameter estimates
        Lorentzian3DParameterEstimates estimates = Estimate3DLorentzianParameters(clean_x, clean_y, clean_z, center_x_estimate, center_y_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "3D Lorentzian parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calculate uncertainty as 5% of max charge
        double max_charge = *std::max_element(clean_z.begin(), clean_z.end());
        double uncertainty = Calculate3DLorentzianUncertainty(max_charge);
        
        // Multiple fitting configurations (similar to 2D but for 6 parameters)
        struct Lorentzian3DFittingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        std::vector<Lorentzian3DFittingConfig> configs;
        
        Lorentzian3DFittingConfig config1;
        config1.linear_solver = ceres::DENSE_QR;
        config1.trust_region = ceres::LEVENBERG_MARQUARDT;
        config1.function_tolerance = 1e-15;
        config1.gradient_tolerance = 1e-15;
        config1.max_iterations = 2000;
        config1.loss_function = "HUBER";
        config1.loss_parameter = estimates.amplitude * 0.1;
        configs.push_back(config1);
        
        Lorentzian3DFittingConfig config2;
        config2.linear_solver = ceres::DENSE_QR;
        config2.trust_region = ceres::LEVENBERG_MARQUARDT;
        config2.function_tolerance = 1e-12;
        config2.gradient_tolerance = 1e-12;
        config2.max_iterations = 1500;
        config2.loss_function = "CAUCHY";
        config2.loss_parameter = estimates.amplitude * 0.16;
        configs.push_back(config2);
        
        Lorentzian3DFittingConfig config3;
        config3.linear_solver = ceres::DENSE_QR;
        config3.trust_region = ceres::DOGLEG;
        config3.function_tolerance = 1e-10;
        config3.gradient_tolerance = 1e-10;
        config3.max_iterations = 1000;
        config3.loss_function = "NONE";
        config3.loss_parameter = 0.0;
        configs.push_back(config3);
        
        for (const auto& config : configs) {
            // Set up parameter array (6 parameters: A, mx, my, gamma_x, gamma_y, B)
            double parameters[6];
            parameters[0] = estimates.amplitude;
            parameters[1] = estimates.center_x;
            parameters[2] = estimates.center_y;
            parameters[3] = estimates.gamma_x;
            parameters[4] = estimates.gamma_y;
            parameters[5] = estimates.baseline;
            
            // Build the problem
            ceres::Problem problem;
            
            // Add residual blocks with uncertainties
            for (size_t i = 0; i < clean_x.size(); ++i) {
                ceres::CostFunction* cost_function = Lorentzian3DCostFunction::Create(
                    clean_x[i], clean_y[i], clean_z[i], uncertainty);
                
                // No loss functions - simple weighted least squares
                problem.AddResidualBlock(cost_function, nullptr, parameters);
            }
            
            // Set bounds
            double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, estimates.amplitude * 0.01);
            double max_charge_val = *std::max_element(clean_z.begin(), clean_z.end());
            double physics_amp_max = max_charge_val * 1.5;
            double algo_amp_max = estimates.amplitude * 100.0;
            double amp_max = std::min(physics_amp_max, algo_amp_max);

            problem.SetParameterLowerBound(parameters, 0, amp_min);
            problem.SetParameterUpperBound(parameters, 0, amp_max);
            
            double center_range = pixel_spacing * 3.0;
            problem.SetParameterLowerBound(parameters, 1, estimates.center_x - center_range);
            problem.SetParameterUpperBound(parameters, 1, estimates.center_x + center_range);
            problem.SetParameterLowerBound(parameters, 2, estimates.center_y - center_range);
            problem.SetParameterUpperBound(parameters, 2, estimates.center_y + center_range);
            
            problem.SetParameterLowerBound(parameters, 3, pixel_spacing * 0.05);
            problem.SetParameterUpperBound(parameters, 3, pixel_spacing * 4.0);
            problem.SetParameterLowerBound(parameters, 4, pixel_spacing * 0.05);
            problem.SetParameterUpperBound(parameters, 4, pixel_spacing * 4.0);
            
            double baseline_range = std::max(estimates.amplitude * 0.5, std::abs(estimates.baseline) * 2.0);
            problem.SetParameterLowerBound(parameters, 5, estimates.baseline - baseline_range);
            problem.SetParameterUpperBound(parameters, 5, estimates.baseline + baseline_range);
            
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
                                 parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 &&
                                 !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                 !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                                 !std::isnan(parameters[4]) && !std::isnan(parameters[5]);
            
            if (fit_successful) {
                // Extract results
                fit_amplitude = parameters[0];
                fit_center_x = parameters[1];
                fit_center_y = parameters[2];
                fit_gamma_x = std::abs(parameters[3]);
                fit_gamma_y = std::abs(parameters[4]);
                fit_vertical_offset = parameters[5];
                
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
                        double covariance_matrix[36]; // 6x6 matrix for 6 parameters
                        if (covariance.GetCovarianceBlock(parameters, parameters, covariance_matrix)) {
                            fit_amplitude_err = std::sqrt(std::abs(covariance_matrix[0]));
                            fit_center_x_err = std::sqrt(std::abs(covariance_matrix[7]));
                            fit_center_y_err = std::sqrt(std::abs(covariance_matrix[14]));
                            fit_gamma_x_err = std::sqrt(std::abs(covariance_matrix[21]));
                            fit_gamma_y_err = std::sqrt(std::abs(covariance_matrix[28]));
                            fit_vertical_offset_err = std::sqrt(std::abs(covariance_matrix[35]));
                            
                            if (!std::isnan(fit_amplitude_err) && !std::isnan(fit_center_x_err) &&
                                !std::isnan(fit_center_y_err) && !std::isnan(fit_gamma_x_err) &&
                                !std::isnan(fit_gamma_y_err) && !std::isnan(fit_vertical_offset_err) &&
                                fit_amplitude_err < 10.0 * fit_amplitude &&
                                fit_center_x_err < 5.0 * pixel_spacing &&
                                fit_center_y_err < 5.0 * pixel_spacing) {
                                cov_success = true;
                                break;
                            }
                        }
                    }
                }
                
                // Fallback uncertainty estimation
                if (!cov_success) {
                    Data3DStatistics data_stats = CalculateRobust3DStatistics(clean_x, clean_y, clean_z);
                    fit_amplitude_err = std::max(0.02 * fit_amplitude, 0.1 * data_stats.mad);
                    fit_center_x_err = std::max(0.02 * pixel_spacing, fit_gamma_x / 10.0);
                    fit_center_y_err = std::max(0.02 * pixel_spacing, fit_gamma_y / 10.0);
                    fit_gamma_x_err = std::max(0.05 * fit_gamma_x, 0.01 * pixel_spacing);
                    fit_gamma_y_err = std::max(0.05 * fit_gamma_y, 0.01 * pixel_spacing);
                    fit_vertical_offset_err = std::max(0.1 * std::abs(fit_vertical_offset), 0.05 * data_stats.mad);
                }
                
                // Calculate reduced chi-squared
                double chi2 = summary.final_cost * 2.0;
                int dof = std::max(1, static_cast<int>(clean_x.size()) - 6);
                chi2_reduced = chi2 / dof;
                
                if (verbose) {
                    std::cout << "Successful 3D Lorentzian fit with config " << &config - &configs[0] 
                             << ", dataset " << dataset_idx 
                             << ": A=" << fit_amplitude << "±" << fit_amplitude_err
                             << ", mx=" << fit_center_x << "±" << fit_center_x_err
                             << ", my=" << fit_center_y << "±" << fit_center_y_err
                             << ", gamma_x=" << fit_gamma_x << "±" << fit_gamma_x_err
                             << ", gamma_y=" << fit_gamma_y << "±" << fit_gamma_y_err
                             << ", B=" << fit_vertical_offset << "±" << fit_vertical_offset_err
                             << ", chi2red=" << chi2_reduced << std::endl;
                }
                
                return true;
            } else if (verbose) {
                std::cout << "3D Lorentzian fit failed with config " << &config - &configs[0] 
                         << ": " << summary.BriefReport() << std::endl;
            }
        }
    }
    
    if (verbose) {
        std::cout << "All 3D Lorentzian fitting strategies failed" << std::endl;
    }
    return false;
}

LorentzianFit3DResultsCeres Fit3DLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    LorentzianFit3DResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeres3DLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeres3DLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit3DLorentzianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 6) {
        if (verbose) {
            std::cout << "Fit3DLorentzianCeres: Error - need at least 6 data points for 3D fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 3D Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
    
    // Perform 3D Lorentzian surface fitting
    bool fit_success = Fit3DLorentzianCeres(
        x_coords, y_coords, charge_values, center_x_estimate, center_y_estimate, pixel_spacing,
        result.amplitude, result.center_x, result.center_y, result.gamma_x, result.gamma_y, result.vertical_offset,
        result.amplitude_err, result.center_x_err, result.center_y_err, result.gamma_x_err, result.gamma_y_err, result.vertical_offset_err,
        result.chi2red, verbose, enable_outlier_filtering);
    
    // Calculate DOF and p-value
    result.dof = std::max(1, static_cast<int>(x_coords.size()) - 6);
    result.pp = (result.chi2red > 0) ? 1.0 - std::min(1.0, result.chi2red / 10.0) : 0.0;
    
    // Set overall success status
    result.fit_successful = fit_success;
    
    if (verbose) {
        std::cout << "3D Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") << std::endl;
    }
    
    return result;
} 