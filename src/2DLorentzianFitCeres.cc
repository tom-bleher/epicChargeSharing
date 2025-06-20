#include "2DLorentzianFitCeres.hh"
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
static std::mutex gCeresLorentzianFitMutex;
static std::atomic<int> gGlobalCeresLorentzianFitCounter{0};

// Use shared Google logging initialization
void InitializeCeresLorentzian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Calculate uncertainty as 5% of max charge in line (if enabled)
double CalculateLorentzianUncertainty(double max_charge_in_line) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Uncertainty = 5% of max charge when enabled
    double uncertainty = 0.05 * max_charge_in_line;
    if (uncertainty < 1e-20) uncertainty = 1e-20; // Prevent division by zero (very small for coulomb range)
    return uncertainty;
}

// Lorentzian cost function with uncertainty (5% of max charge)
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B  
struct LorentzianCostFunction {
    LorentzianCostFunction(double x, double y, double uncertainty) 
        : x_(x), y_(y), uncertainty_(uncertainty) {}
    
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
        
        // Residual divided by uncertainty (standard weighted least squares)
        residual[0] = (predicted - T(y_)) / T(uncertainty_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double uncertainty) {
        return (new ceres::AutoDiffCostFunction<LorentzianCostFunction, 1, 4>(
            new LorentzianCostFunction(x, y, uncertainty)));
    }
    
private:
    const double x_;
    const double y_;
    const double uncertainty_;
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
    
    // ------------------------------------------------------------------------------------
    // Numerical stability safeguard: ensure MAD is positive and finite.
    // A zero or non-finite MAD can later produce divide-by-zero errors or NaNs in weighting
    // and outlier filtering, ultimately destabilising the fit.  When this happens we fall
    // back to the standard deviation (if sensible) or a small epsilon.
    // ------------------------------------------------------------------------------------
    if (!std::isfinite(stats.mad) || stats.mad < 1e-12) {
        stats.mad = (std::isfinite(stats.std_dev) && stats.std_dev > 1e-12) ?
                    stats.std_dev : 1e-12;
    }
    
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
        stats.robust_center = stats.weighted_mean;
    } else {
        stats.weighted_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
        stats.robust_center = stats.weighted_mean;
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
        
        // Calculate uncertainty as 5% of max charge
        double max_charge = *std::max_element(clean_y.begin(), clean_y.end());
        double uncertainty = CalculateLorentzianUncertainty(max_charge);
        
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
        config1.loss_parameter = estimates.amplitude * 0.1;
        configs.push_back(config1);
        
        LorentzianFittingConfig config2;
        config2.linear_solver = ceres::DENSE_QR;
        config2.trust_region = ceres::LEVENBERG_MARQUARDT;
        config2.function_tolerance = 1e-12;
        config2.gradient_tolerance = 1e-12;
        config2.max_iterations = 1500;
        config2.loss_function = "CAUCHY";
        config2.loss_parameter = estimates.amplitude * 0.16;
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
        config4.loss_parameter = estimates.amplitude * 0.13;
        configs.push_back(config4);
        
        LorentzianFittingConfig config5;
        config5.linear_solver = ceres::SPARSE_NORMAL_CHOLESKY;
        config5.trust_region = ceres::LEVENBERG_MARQUARDT;
        config5.function_tolerance = 1e-12;
        config5.gradient_tolerance = 1e-12;
        config5.max_iterations = 1200;
        config5.loss_function = "CAUCHY";
        config5.loss_parameter = estimates.amplitude * 0.22;
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
            
            // Add residual blocks with uncertainties
            for (size_t i = 0; i < clean_x.size(); ++i) {
                ceres::CostFunction* cost_function = LorentzianCostFunction::Create(
                    clean_x[i], clean_y[i], uncertainty);
                
                // No loss functions - simple weighted least squares
                problem.AddResidualBlock(cost_function, nullptr, parameters);
            }
            
            // Set bounds
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
            problem.SetParameterUpperBound(parameters, 2, pixel_spacing * 4.0);
            
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
                // Ceres returns 0.5 * Σ r_i^2, multiply by 2 to get χ².
                double chi2 = summary.final_cost * 2.0;
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
    
    // Effective centre-to-centre pitch along the diagonal (√2 times horizontal pitch)
    const double diag_pixel_spacing = pixel_spacing * 1.41421356237;
    
    // Group pixels by diagonal lines, using coordinate along the diagonal ("s")
    // so that the fit abscissa is the true distance along the ±45° axis.
    // Define   s_main = (dx + dy) / sqrt(2)
    //         s_sec  = (dx - dy) / sqrt(2)
    // where dx=x-centreX, dy=y-centreY.  We store (s, charge).

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    std::map<double, std::vector<std::pair<double, double>>> main_diagonal_data; // id -> (s, q)
    std::map<double, std::vector<std::pair<double, double>>> sec_diagonal_data;  // id -> (s, q)

    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x = x_coords[i];
        double y = y_coords[i];
        double charge = charge_values[i];

        if (charge <= 0) continue;

        double dx = x - center_x_estimate;
        double dy = y - center_y_estimate;

        // Identify membership of each pixel to diagonals using tolerance on (dx-dy) and (dx+dy)
        double diff = dx - dy;   // main diagonal indicator
        double sum  = dx + dy;   // secondary diagonal indicator

        if (std::abs(diff) < tolerance) {
            double s = (dx + dy) * inv_sqrt2; // coordinate along main diagonal
            main_diagonal_data[diff].emplace_back(s, charge);
        }

        if (std::abs(sum) < tolerance) {
            double s = (dx - dy) * inv_sqrt2; // (dx-dy)/sqrt(2)
            sec_diagonal_data[sum].emplace_back(s, charge);
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
        
        // Extract diagonal coordinate (s) and charge values
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
                std::cout << "Fitting main diagonal Lorentzian X with " << x_sorted.size() << " points" << std::endl;
            }
            
            main_diag_x_success = FitLorentzianCeres(
                x_sorted, charge_x_sorted, 0.0, diag_pixel_spacing,
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
                std::cout << "Fitting main diagonal Lorentzian Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            main_diag_y_success = FitLorentzianCeres(
                y_sorted, charge_y_sorted, 0.0, diag_pixel_spacing,
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
                std::cout << "Fitting secondary diagonal Lorentzian X with " << x_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_x_success = FitLorentzianCeres(
                x_sorted, charge_x_sorted, 0.0, diag_pixel_spacing,
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
                std::cout << "Fitting secondary diagonal Lorentzian Y with " << y_sorted.size() << " points" << std::endl;
            }
            
            sec_diag_y_success = FitLorentzianCeres(
                y_sorted, charge_y_sorted, 0.0, diag_pixel_spacing,
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