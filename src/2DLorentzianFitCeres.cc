#include "2DLorentzianFitCeres.hh"
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
    if (uncertainty < Constants::MIN_UNCERTAINTY_VALUE) uncertainty = Constants::MIN_UNCERTAINTY_VALUE; // Prevent division by zero
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
        if (safe_gamma < T(Constants::MIN_SAFE_PARAMETER)) {
            safe_gamma = T(Constants::MIN_SAFE_PARAMETER);
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
        
        // OPTIMIZED: Cheap config first with early exit based on quality (Step 1 from optimize.md)
        struct LorentzianFittingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        // Stage 1: Cheap configuration
        LorentzianFittingConfig cheap_config = {
            ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 
            1e-10, 1e-10, 400, "NONE", 0.0
        };
        
        // Stage 2: Expensive fallback configurations
        const std::vector<LorentzianFittingConfig> expensive_configs = {
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amplitude * 0.1},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amplitude * 0.16},
            {ceres::SPARSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1200, "CAUCHY", estimates.amplitude * 0.22}
        };
        
        // Try cheap config first
        auto try_config = [&](const LorentzianFittingConfig& config, const std::string& stage_name) -> bool {
            if (verbose) {
                std::cout << "Trying Lorentzian " << stage_name << " configuration..." << std::endl;
            }
            // STEP 2 OPTIMIZATION: Hierarchical multi-start budget for Lorentzian
            // Start with base estimate, only add 2 perturbations if χ²ᵣ > 2.0
            // Expected: ×4-5 speed-up, average #Ceres solves/fit ≤10
            
            struct ParameterSet {
                double params[4];
                std::string description;
            };
            
            std::vector<ParameterSet> initial_guesses;
            
            // ALWAYS start with base estimate first
            ParameterSet base_set;
            base_set.params[0] = estimates.amplitude;
            base_set.params[1] = estimates.center;
            base_set.params[2] = estimates.gamma;
            base_set.params[3] = estimates.baseline;
            base_set.description = "base_estimate";
            initial_guesses.push_back(base_set);
            
            double best_cost = std::numeric_limits<double>::max();
            double best_parameters[4];
            bool any_success = false;
            std::string best_description;
            double best_chi2_reduced = std::numeric_limits<double>::max();
            
            // Data characteristics for adaptive bounds
            DataStatistics data_stats = CalculateRobustStatisticsLorentzian(clean_x, clean_y);
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
                
                ceres::Problem problem;
                
                for (size_t i = 0; i < clean_x.size(); ++i) {
                    ceres::CostFunction* cost_function = LorentzianCostFunction::Create(
                        clean_x[i], clean_y[i], uncertainty);
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
                
                double center_range = pixel_spacing * 3.0;
                problem.SetParameterLowerBound(parameters, 1, parameters[1] - center_range);
                problem.SetParameterUpperBound(parameters, 1, parameters[1] + center_range);
                
                problem.SetParameterLowerBound(parameters, 2, pixel_spacing * 0.05);
                problem.SetParameterUpperBound(parameters, 2, pixel_spacing * 4.0);
                
                double charge_range = std::abs(max_charge_val - min_charge_val);
                double baseline_range = std::max(charge_range * 0.5, 
                                               std::max(std::abs(parameters[3]) * 2.0, 1e-12));
                problem.SetParameterLowerBound(parameters, 3, parameters[3] - baseline_range);
                problem.SetParameterUpperBound(parameters, 3, parameters[3] + baseline_range);
            
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
                                     parameters[0] > 0 && parameters[2] > 0 &&
                                     !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                     !std::isnan(parameters[2]) && !std::isnan(parameters[3]);
                
                if (fit_successful) {
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
                            std::cout << "New best Lorentzian result from " << guess.description 
                                     << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                        }
                    }
                }
            }
            
            // ALWAYS add perturbations regardless of chi-squared quality
            if (any_success) {
                if (verbose) {
                    std::cout << "Base Lorentzian fit χ²ᵣ=" << best_chi2_reduced << ", trying perturbations..." << std::endl;
                }
                
                // Add exactly 2 perturbations to reduce multi-start budget
                const std::vector<double> perturbation_factors = {0.7, 1.3};
                
                for (double factor : perturbation_factors) {
                    ParameterSet perturbed_set;
                    perturbed_set.params[0] = estimates.amplitude * factor;
                    perturbed_set.params[1] = estimates.center + (factor - 1.0) * pixel_spacing * 0.3;
                    perturbed_set.params[2] = estimates.gamma * std::sqrt(factor);
                    perturbed_set.params[3] = estimates.baseline * (0.8 + 0.4 * factor);
                    perturbed_set.description = "lorentz_perturbation_" + std::to_string(factor);
                    
                    // Try this perturbation (same logic as above)
                    double parameters[4];
                    parameters[0] = perturbed_set.params[0];
                    parameters[1] = perturbed_set.params[1];
                    parameters[2] = perturbed_set.params[2];
                    parameters[3] = perturbed_set.params[3];
                    
                    ceres::Problem problem;
                    
                    for (size_t i = 0; i < clean_x.size(); ++i) {
                        ceres::CostFunction* cost_function = LorentzianCostFunction::Create(
                            clean_x[i], clean_y[i], uncertainty);
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
                    
                    double center_range = pixel_spacing * 3.0;
                    problem.SetParameterLowerBound(parameters, 1, parameters[1] - center_range);
                    problem.SetParameterUpperBound(parameters, 1, parameters[1] + center_range);
                    
                    problem.SetParameterLowerBound(parameters, 2, pixel_spacing * 0.05);
                    problem.SetParameterUpperBound(parameters, 2, pixel_spacing * 4.0);
                    
                    double charge_range = std::abs(max_charge_val - min_charge_val);
                    double baseline_range = std::max(charge_range * 0.5, 
                                                   std::max(std::abs(parameters[3]) * 2.0, 1e-12));
                    problem.SetParameterLowerBound(parameters, 3, parameters[3] - baseline_range);
                    problem.SetParameterUpperBound(parameters, 3, parameters[3] + baseline_range);
                    
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
                                         parameters[0] > 0 && parameters[2] > 0 &&
                                         !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                         !std::isnan(parameters[2]) && !std::isnan(parameters[3]);
                    
                    if (fit_successful) {
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
            } else if (verbose && any_success) {
                std::cout << "Base Lorentzian fit χ²ᵣ=" << best_chi2_reduced << " ≤ 0.3, skipping perturbations (hierarchical multi-start)" << std::endl;
            }
            
            if (any_success) {
                // Extract results from best attempt
                fit_amplitude = best_parameters[0];
                fit_center = best_parameters[1];
                fit_gamma = std::abs(best_parameters[2]);
                fit_vertical_offset = best_parameters[3];
                
                // Simple fallback uncertainty estimation
                fit_amplitude_err = std::max(0.02 * fit_amplitude, 0.1 * data_stats.mad);
                fit_center_err = std::max(0.02 * pixel_spacing, fit_gamma / 10.0);
                fit_gamma_err = std::max(0.05 * fit_gamma, 0.01 * pixel_spacing);
                fit_vertical_offset_err = std::max(0.1 * std::abs(fit_vertical_offset), 0.05 * data_stats.mad);
                
                chi2_reduced = best_chi2_reduced;
                
                if (verbose) {
                    std::cout << "Successful Lorentzian fit with " << stage_name 
                             << ", dataset " << dataset_idx << ", best init: " << best_description
                             << ": A=" << fit_amplitude << "±" << fit_amplitude_err
                             << ", m=" << fit_center << "±" << fit_center_err
                             << ", gamma=" << fit_gamma << "±" << fit_gamma_err
                             << ", B=" << fit_vertical_offset << "±" << fit_vertical_offset_err
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
            std::cout << "Cheap Lorentzian config " << (success ? "succeeded" : "failed") 
                     << " with χ²ᵣ=" << chi2_reduced << std::endl;
        }
        
        // Always try ALL expensive configurations regardless of cheap config result
        if (verbose) {
            std::cout << "Trying all " << expensive_configs.size() 
                     << " expensive Lorentzian configurations..." << std::endl;
        }
        
        for (size_t i = 0; i < expensive_configs.size(); ++i) {
            bool config_success = try_config(expensive_configs[i], "expensive_" + std::to_string(i+1));
            if (config_success && (!best_success || chi2_reduced < best_chi2)) {
                best_success = config_success;
                best_chi2 = chi2_reduced;
                success = config_success;
            }
            
            if (verbose) {
                std::cout << "Expensive Lorentzian config " << (i+1) << " " 
                         << (config_success ? "succeeded" : "failed") 
                         << " with χ²ᵣ=" << chi2_reduced << std::endl;
            }
        }
        
        if (best_success) {
            if (verbose) {
                std::cout << "Best Lorentzian fit achieved with χ²ᵣ=" << best_chi2 << std::endl;
            }
            return true;
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
    
    // Initialize Ceres logging (removed mutex for better parallelization)
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
        
        // Store data for ROOT analysis
        result.x_row_pixel_coords = x_vals;
        result.x_row_charge_values = y_vals;
        result.x_row_charge_errors = std::vector<double>(); // Empty vector
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
        
        // Store data for ROOT analysis
        result.y_col_pixel_coords = x_vals;  // Y coordinates
        result.y_col_charge_values = y_vals;
        result.y_col_charge_errors = std::vector<double>(); // Empty vector
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
    
    // Initialize Ceres logging (removed mutex for better parallelization)
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
    
    // FIXED DIAGONAL APPROACH:
    // 1. Use spatial binning instead of exact coordinate matching
    // 2. Correct coordinate transformations for ±45° rotations
    // 3. Proper pixel spacing for diagonal geometry
    
    // Diagonal bin width - wider than row/column to capture more pixels
    const double bin_width = pixel_spacing * 0.5; // 50% of pixel spacing for better coverage
    
    // Correct diagonal pixel spacing (actual distance between diagonal neighbors)
    const double diag_pixel_spacing = pixel_spacing * 1.41421356237; // √2, no calibration factor
    
    // Maps: binned diagonal coordinate → [(distance_along_diagonal, charge), ...]
    std::map<int, std::vector<std::pair<double, double>>> main_diagonal_bins; // +45° direction
    std::map<int, std::vector<std::pair<double, double>>> sec_diagonal_bins;  // -45° direction
    
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
        // The diagonal runs along constant (dy-dx), and position along diagonal is (dx+dy)/√2
        double main_diag_id = dy - dx;                    // Perpendicular distance to +45° line
        double main_diag_pos = (dx + dy) / 1.41421356237; // Position along +45° diagonal
        
        // Secondary diagonal (-45°): pixels where x-coordinate increases as y-coordinate decreases  
        // We rotate by +45° to align with X-axis: (x',y') = ((dx-dy)/√2, (dx+dy)/√2)
        // The diagonal runs along constant (dx+dy), and position along diagonal is (dx-dy)/√2
        double sec_diag_id = dx + dy;                     // Perpendicular distance to -45° line
        double sec_diag_pos = (dx - dy) / 1.41421356237; // Position along -45° diagonal
        
        // Bin the diagonal identifiers for spatial grouping
        int main_bin = static_cast<int>(std::round(main_diag_id / bin_width));
        int sec_bin = static_cast<int>(std::round(sec_diag_id / bin_width));
        
        main_diagonal_bins[main_bin].emplace_back(main_diag_pos, charge);
        sec_diagonal_bins[sec_bin].emplace_back(sec_diag_pos, charge);
    }
    
    if (verbose) {
        std::cout << "Found " << main_diagonal_bins.size() << " main diagonal bins and " 
                 << sec_diagonal_bins.size() << " secondary diagonal bins" << std::endl;
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
    
    int best_main_bin = find_best_bin(main_diagonal_bins, "main");
    int best_sec_bin = find_best_bin(sec_diagonal_bins, "secondary");
    
    bool main_diag_success = false;
    bool sec_diag_success = false;
    
    // Fit main diagonal (+45° direction)
    if (main_diagonal_bins.find(best_main_bin) != main_diagonal_bins.end() && 
        main_diagonal_bins[best_main_bin].size() >= 4) {
        
        auto& main_data = main_diagonal_bins[best_main_bin];
        
        // Sort by position along diagonal
        std::sort(main_data.begin(), main_data.end());
        
        std::vector<double> positions, charges;
        for (const auto& point : main_data) {
            positions.push_back(point.first);
            charges.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting main diagonal (+45°) with " << positions.size() << " points" << std::endl;
        }
        
        main_diag_success = FitLorentzianCeres(
            positions, charges, 0.0, diag_pixel_spacing,
            result.main_diag_x_amplitude, result.main_diag_x_center, result.main_diag_x_gamma, result.main_diag_x_vertical_offset,
            result.main_diag_x_amplitude_err, result.main_diag_x_center_err, result.main_diag_x_gamma_err, result.main_diag_x_vertical_offset_err,
            result.main_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.main_diag_x_dof = std::max(1, static_cast<int>(positions.size()) - 4);
        result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
        result.main_diag_x_fit_successful = main_diag_success;
        
        // Store data for ROOT analysis
        result.main_diag_x_pixel_coords = positions;
        result.main_diag_x_charge_values = charges;
        result.main_diag_x_charge_errors = std::vector<double>();
        
        // For symmetry, copy results to Y (since we only have one diagonal measurement)
        result.main_diag_y_amplitude = result.main_diag_x_amplitude;
        result.main_diag_y_center = result.main_diag_x_center;
        result.main_diag_y_gamma = result.main_diag_x_gamma;
        result.main_diag_y_vertical_offset = result.main_diag_x_vertical_offset;
        result.main_diag_y_amplitude_err = result.main_diag_x_amplitude_err;
        result.main_diag_y_center_err = result.main_diag_x_center_err;
        result.main_diag_y_gamma_err = result.main_diag_x_gamma_err;
        result.main_diag_y_vertical_offset_err = result.main_diag_x_vertical_offset_err;
        result.main_diag_y_chi2red = result.main_diag_x_chi2red;
        result.main_diag_y_dof = result.main_diag_x_dof;
        result.main_diag_y_pp = result.main_diag_x_pp;
        result.main_diag_y_fit_successful = main_diag_success;
        result.main_diag_y_pixel_coords = positions;
        result.main_diag_y_charge_values = charges;
        result.main_diag_y_charge_errors = std::vector<double>();
    }
    
    // Fit secondary diagonal (-45° direction)
    if (sec_diagonal_bins.find(best_sec_bin) != sec_diagonal_bins.end() && 
        sec_diagonal_bins[best_sec_bin].size() >= 4) {
        
        auto& sec_data = sec_diagonal_bins[best_sec_bin];
        
        // Sort by position along diagonal
        std::sort(sec_data.begin(), sec_data.end());
        
        std::vector<double> positions, charges;
        for (const auto& point : sec_data) {
            positions.push_back(point.first);
            charges.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting secondary diagonal (-45°) with " << positions.size() << " points" << std::endl;
        }
        
        sec_diag_success = FitLorentzianCeres(
            positions, charges, 0.0, diag_pixel_spacing,
            result.sec_diag_x_amplitude, result.sec_diag_x_center, result.sec_diag_x_gamma, result.sec_diag_x_vertical_offset,
            result.sec_diag_x_amplitude_err, result.sec_diag_x_center_err, result.sec_diag_x_gamma_err, result.sec_diag_x_vertical_offset_err,
            result.sec_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.sec_diag_x_dof = std::max(1, static_cast<int>(positions.size()) - 4);
        result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
        result.sec_diag_x_fit_successful = sec_diag_success;
        
        // Store data for ROOT analysis
        result.sec_diag_x_pixel_coords = positions;
        result.sec_diag_x_charge_values = charges;
        result.sec_diag_x_charge_errors = std::vector<double>();
        
        // For symmetry, copy results to Y
        result.sec_diag_y_amplitude = result.sec_diag_x_amplitude;
        result.sec_diag_y_center = result.sec_diag_x_center;
        result.sec_diag_y_gamma = result.sec_diag_x_gamma;
        result.sec_diag_y_vertical_offset = result.sec_diag_x_vertical_offset;
        result.sec_diag_y_amplitude_err = result.sec_diag_x_amplitude_err;
        result.sec_diag_y_center_err = result.sec_diag_x_center_err;
        result.sec_diag_y_gamma_err = result.sec_diag_x_gamma_err;
        result.sec_diag_y_vertical_offset_err = result.sec_diag_x_vertical_offset_err;
        result.sec_diag_y_chi2red = result.sec_diag_x_chi2red;
        result.sec_diag_y_dof = result.sec_diag_x_dof;
        result.sec_diag_y_pp = result.sec_diag_x_pp;
        result.sec_diag_y_fit_successful = sec_diag_success;
        result.sec_diag_y_pixel_coords = positions;
        result.sec_diag_y_charge_values = charges;
        result.sec_diag_y_charge_errors = std::vector<double>();
    }
    
    // Set overall success status
    result.fit_successful = main_diag_success && sec_diag_success;
    
    if (verbose) {
        std::cout << "Diagonal Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (Main +45°: " << (main_diag_success ? "OK" : "FAIL") 
                 << ", Secondary -45°: " << (sec_diag_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}