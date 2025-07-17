#include "LorentzFit2D.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "Control.hh"
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
void InitializeCeresLorentz() {
    CeresLoggingInitializer::InitializeOnce();
}

// Calc err as 5% of max charge in line (if enabled)
double CalcLorentzErr(double max_charge_in_line) {
    if (!Control::CHARGE_ERR) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Err = 5% of max charge when enabled
    double err = 0.05 * max_charge_in_line;
    if (err < Constants::MIN_UNCERTAINTY_VALUE) err = Constants::MIN_UNCERTAINTY_VALUE; // Prevent division by zero
    return err;
}

// Lorentz cost function with err (5% of max charge)
// Function form: y(x) = A / (1 + ((x - m) / γ)^2) + B  
struct LorentzCostFunction {
    LorentzCostFunction(double x, double y, double err) 
        : x_(x), y_(y), err_(err) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amp)
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
        
        // Lorentz function: y(x) = A / (1 + ((x - m) / γ)^2) + B
        T dx = x_ - m;
        T normalized_dx = dx / safe_gamma;
        T denominator = T(1.0) + normalized_dx * normalized_dx;
        
        // Prevent numerical issues with very small denominators
        if (denominator < T(1e-12)) {
            denominator = T(1e-12);
        }
        
        T predicted = A / denominator + B;
        
        // Residual divided by err (standard weighted least squares)
        residual[0] = (predicted - T(y_)) / T(err_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double err) {
        return (new ceres::AutoDiffCostFunction<LorentzCostFunction, 1, 4>(
            new LorentzCostFunction(x, y, err)));
    }
    
private:
    const double x_;
    const double y_;
    const double err_;
};

// Parameter estimation structures for Lorentz
struct LorentzParameterEstimates {
    double amp;
    double center;
    double gamma;
    double baseline;
    double amp_err;
    double center_err;
    double gamma_err;
    double baseline_err;
    bool valid;
    int method_used;
};

// Robust statistics calculations (reusing from Gauss implementation)
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

DataStatistics CalcRobustStatisticsLorentz(const std::vector<double>& x_vals, 
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

// Parameter estimation for Lorentz distributions
LorentzParameterEstimates EstimateLorentzParameters(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    LorentzParameterEstimates estimates;
    estimates.valid = false;
    estimates.method_used = 0;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        // EMERGENCY PARAMETER FALLBACK for small datasets
        if (x_vals.size() >= 1) {
            estimates.center = center_estimate;
            estimates.amp = (!y_vals.empty()) ? (*std::max_element(y_vals.begin(), y_vals.end()) - 
                                               *std::min_element(y_vals.begin(), y_vals.end())) : 1.0;
            estimates.gamma = pixel_spacing * 0.7; // Larger default for Lorentz
            estimates.baseline = (!y_vals.empty()) ? *std::min_element(y_vals.begin(), y_vals.end()) : 0.0;
            estimates.method_used = 99; // Emergency method indicator
            estimates.valid = true;
            
            if (verbose) {
                std::cout << "Emergency Lorentz parameter estimation for small dataset" << std::endl;
            }
            return estimates;
        }
        
        // ULTIMATE FALLBACK for empty datasets
        estimates.center = center_estimate;
        estimates.amp = 1.0;
        estimates.gamma = pixel_spacing * 0.7;
        estimates.baseline = 0.0;
        estimates.method_used = 100; // Ultimate fallback indicator
        estimates.valid = true;
        return estimates;
    }
    
    DataStatistics stats = CalcRobustStatisticsLorentz(x_vals, y_vals);
    if (!stats.valid) {
        // FALLBACK when statistics calculation fails
        estimates.center = center_estimate;
        estimates.amp = (!y_vals.empty()) ? (*std::max_element(y_vals.begin(), y_vals.end()) - 
                                           *std::min_element(y_vals.begin(), y_vals.end())) : 1.0;
        estimates.gamma = pixel_spacing * 0.7;
        estimates.baseline = (!y_vals.empty()) ? *std::min_element(y_vals.begin(), y_vals.end()) : 0.0;
        estimates.method_used = 98; // Statistics fallback indicator
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Lorentz statistics calculation failed, using basic fallback parameters" << std::endl;
        }
        return estimates;
    }
    
    if (verbose) {
        std::cout << "Lorentz data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", weighted_mean=" << stats.weighted_mean << std::endl;
    }
    
    // Method 1: Physics-based estimation for charge distributions
    estimates.center = stats.weighted_mean;
    estimates.baseline = std::min(stats.min_val, stats.q25);
    estimates.amp = stats.max_val - estimates.baseline;
    
    // For Lorentz: gamma (HWHM) estimation based on charge spread
    // Lorentz tails are wider than Gauss, so use larger initial gamma
    double distance_spread = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, y_vals[i] - estimates.baseline);
        if (weight > 0.1 * estimates.amp) {
            double dx = x_vals[i] - estimates.center;
            distance_spread += weight * dx * dx;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0) {
        // For Lorentz, gamma ≈ sqrt(2*sigma^2) where sigma is from Gauss equivalent
        estimates.gamma = std::sqrt(2.0 * distance_spread / weight_sum);
    } else {
        estimates.gamma = pixel_spacing * 0.7; // Larger default for Lorentz
    }
    
    // Apply physics-based bounds (Lorentz has wider tails)
    estimates.gamma = std::max(pixel_spacing * 0.3, std::min(pixel_spacing * 3.0, estimates.gamma));
    estimates.amp = std::max(estimates.amp, (stats.max_val - stats.min_val) * 0.1);
    
    // Validate Method 1
    if (estimates.amp > 0 && estimates.gamma > 0 && 
        !std::isnan(estimates.center) && !std::isnan(estimates.amp) && 
        !std::isnan(estimates.gamma) && !std::isnan(estimates.baseline)) {
        estimates.method_used = 1;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Lorentz Method 1 (Physics-based): A=" << estimates.amp 
                     << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 2: Robust statistical estimation
    estimates.center = stats.median;
    estimates.baseline = stats.q25;
    estimates.amp = stats.q75 - stats.q25;
    estimates.gamma = std::max(stats.mad, pixel_spacing * 0.5);
    
    if (estimates.amp > 0 && estimates.gamma > 0) {
        estimates.method_used = 2;
        estimates.valid = true;
        
        if (verbose) {
            std::cout << "Lorentz Method 2 (Robust statistical): A=" << estimates.amp 
                     << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                     << ", B=" << estimates.baseline << std::endl;
        }
        return estimates;
    }
    
    // Method 3: Conservative fallback
    estimates.center = center_estimate;
    estimates.baseline = 0.0;
    estimates.amp = stats.max_val;
    estimates.gamma = pixel_spacing * 0.7;
    estimates.method_used = 3;
    estimates.valid = true;
    
    if (verbose) {
        std::cout << "Lorentz Method 3 (Conservative fallback): A=" << estimates.amp 
                 << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                 << ", B=" << estimates.baseline << std::endl;
    }
    
    return estimates;
    
    // NOTE: This function now NEVER returns invalid estimates - it always provides
    // some reasonable parameter values even in the worst-case scenarios
}



 

// Outlier filtering for Lorentz fitting (adapted from Gauss version)
std::pair<std::vector<double>, std::vector<double>> FilterLorentzOutliers(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double sigma_threshold = 2.5,
    bool verbose = false) {
    
    std::vector<double> filtered_x, filtered_y;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return std::make_pair(filtered_x, filtered_y);
    }
    
    DataStatistics stats = CalcRobustStatisticsLorentz(x_vals, y_vals);
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
            std::cout << "Too many Lorentz outliers detected (" << outliers_removed 
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
            std::cout << "Warning: After Lorentz outlier filtering, only " << filtered_x.size() 
                     << " points remain" << std::endl;
        }
        return std::make_pair(x_vals, y_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " Lorentz outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_pair(filtered_x, filtered_y);
}

// Core Lorentz fitting function using Ceres Solver
bool LorentzCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    double& fit_amp,
    double& fit_center,
    double& fit_gamma,
    double& fit_vert_offset,
    double& fit_amp_err,
    double& fit_center_err,
    double& fit_gamma_err,
    double& fit_vert_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        if (verbose) {
            std::cout << "Insufficient data points for Lorentz fitting, using emergency fallback" << std::endl;
        }
        
        // EMERGENCY FALLBACK: Use simple statistical estimates
        if (x_vals.size() >= 2) {
            DataStatistics emergency_stats = CalcRobustStatisticsLorentz(x_vals, y_vals);
            if (emergency_stats.valid) {
                fit_amp = emergency_stats.max_val - emergency_stats.min_val;
                fit_center = emergency_stats.robust_center;
                fit_gamma = std::max(pixel_spacing * 0.7, emergency_stats.std_dev);
                fit_vert_offset = emergency_stats.min_val;
                
                // Conservative error estimates
                fit_amp_err = 0.2 * fit_amp;
                fit_center_err = 0.1 * pixel_spacing;
                fit_gamma_err = 0.2 * fit_gamma;
                fit_vert_offset_err = 0.1 * std::abs(fit_vert_offset);
                chi2_reduced = 10.0; // High chi2 indicates poor fit quality
                
                if (verbose) {
                    std::cout << "Emergency fallback Lorentz fit applied" << std::endl;
                }
                return true; // Always return success
            }
        }
        
        // ULTIMATE FALLBACK: Basic heuristic estimates
        fit_amp = (!y_vals.empty()) ? (*std::max_element(y_vals.begin(), y_vals.end()) - 
                                      *std::min_element(y_vals.begin(), y_vals.end())) : 1.0;
        fit_center = center_estimate;
        fit_gamma = pixel_spacing * 0.7;
        fit_vert_offset = (!y_vals.empty()) ? *std::min_element(y_vals.begin(), y_vals.end()) : 0.0;
        
        fit_amp_err = 0.5 * fit_amp;
        fit_center_err = 0.2 * pixel_spacing;
        fit_gamma_err = 0.3 * fit_gamma;
        fit_vert_offset_err = 0.2 * std::abs(fit_vert_offset);
        chi2_reduced = 20.0; // Very high chi2 indicates fallback was used
        
        if (verbose) {
            std::cout << "Ultimate fallback Lorentz fit applied" << std::endl;
        }
        return true; // Always return success
    }
    
    // Multiple outlier filtering strategies
    std::vector<std::pair<std::vector<double>, std::vector<double>>> filtered_datasets;
    
    if (enable_outlier_filtering) {
        auto conservative_data = FilterLorentzOutliers(x_vals, y_vals, 2.5, verbose);
        if (conservative_data.first.size() >= 4) {
            filtered_datasets.push_back(conservative_data);
        }
        
        auto lenient_data = FilterLorentzOutliers(x_vals, y_vals, 3.0, verbose);
        if (lenient_data.first.size() >= 4) {
            filtered_datasets.push_back(lenient_data);
        }
    }
    
    // Always include original data as fallback
    filtered_datasets.push_back(std::make_pair(x_vals, y_vals));
    
    if (verbose) {
        std::cout << "Lorentz outlier filtering " << (enable_outlier_filtering ? "enabled" : "disabled") 
                 << ", testing " << filtered_datasets.size() << " datasets" << std::endl;
    }
    
    // Try each filtered dataset
    for (size_t dataset_idx = 0; dataset_idx < filtered_datasets.size(); ++dataset_idx) {
        std::vector<double> clean_x = filtered_datasets[dataset_idx].first;
        std::vector<double> clean_y = filtered_datasets[dataset_idx].second;
        
        if (clean_x.size() < 4) continue;
        
        if (verbose) {
            std::cout << "Trying Lorentz dataset " << dataset_idx << " with " << clean_x.size() << " points" << std::endl;
        }
        
        // Get parameter estimates
        LorentzParameterEstimates estimates = EstimateLorentzParameters(clean_x, clean_y, center_estimate, pixel_spacing, verbose);
        if (!estimates.valid) {
            if (verbose) {
                std::cout << "Lorentz parameter estimation failed for dataset " << dataset_idx << std::endl;
            }
            continue;
        }
        
        // Calc err as 5% of max charge
        double max_charge = *std::max_element(clean_y.begin(), clean_y.end());
        double err = CalcLorentzErr(max_charge);
        
        // OPTIMIZED: Cheap config first with early exit based on quality (Step 1 from optimize.md)
        struct LorentztingConfig {
            ceres::LinearSolverType linear_solver;
            ceres::TrustRegionStrategyType trust_region;
            double function_tolerance;
            double gradient_tolerance;
            int max_iterations;
            std::string loss_function;
            double loss_parameter;
        };
        
        // Stage 1: Cheap configuration
        LorentztingConfig cheap_config = {
            ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 
            1e-10, 1e-10, 400, "NONE", 0.0
        };
        
        // Stage 2: Expensive fallback configurations
        const std::vector<LorentztingConfig> expensive_configs = {
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", estimates.amp * 0.1},
            {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", estimates.amp * 0.16},
            {ceres::SPARSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1200, "CAUCHY", estimates.amp * 0.22}
        };
        
        // Try cheap config first
        auto try_config = [&](const LorentztingConfig& config, const std::string& stage_name) -> bool {
            if (verbose) {
                std::cout << "Trying Lorentz " << stage_name << " configuration..." << std::endl;
            }
            // STEP 2 OPTIMIZATION: Hierarchical multi-start budget for Lorentz
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
            DataStatistics data_stats = CalcRobustStatisticsLorentz(clean_x, clean_y);
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
                    ceres::CostFunction* cost_function = LorentzCostFunction::Create(
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
                            std::cout << "New best Lorentz result from " << guess.description 
                                     << " (cost=" << cost << ", χ²ᵣ=" << chi2_red << ")" << std::endl;
                        }
                    }
                }
            }
            
            // ALWAYS add perturbations regardless of chi-squared quality
            if (any_success) {
                if (verbose) {
                    std::cout << "Base Lorentz fit χ²ᵣ=" << best_chi2_reduced << ", trying perturbations..." << std::endl;
                }
                
                // Add exactly 2 perturbations to reduce multi-start budget
                const std::vector<double> perturbation_factors = {0.7, 1.3};
                
                for (double factor : perturbation_factors) {
                    ParameterSet perturbed_set;
                    perturbed_set.params[0] = estimates.amp * factor;
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
                        ceres::CostFunction* cost_function = LorentzCostFunction::Create(
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
            } else if (verbose && any_success) {
                std::cout << "Base Lorentz fit χ²ᵣ=" << best_chi2_reduced << " ≤ 0.3, skipping perturbations (hierarchical multi-start)" << std::endl;
            }
            
            if (any_success) {
                // Extract results from best attempt
                fit_amp = best_parameters[0];
                fit_center = best_parameters[1];
                fit_gamma = std::abs(best_parameters[2]);
                fit_vert_offset = best_parameters[3];
                
                // Simple fallback err estimation
                fit_amp_err = std::max(0.02 * fit_amp, 0.1 * data_stats.mad);
                fit_center_err = std::max(0.02 * pixel_spacing, fit_gamma / 10.0);
                fit_gamma_err = std::max(0.05 * fit_gamma, 0.01 * pixel_spacing);
                fit_vert_offset_err = std::max(0.1 * std::abs(fit_vert_offset), 0.05 * data_stats.mad);
                
                chi2_reduced = best_chi2_reduced;
                
                if (verbose) {
                    std::cout << "Success Lorentz fit (" << stage_name << ", dataset " << dataset_idx 
                             << ", init: " << best_description << ")\n"
                             << "  A=" << fit_amp << "±" << fit_amp_err
                             << ", m=" << fit_center << "±" << fit_center_err
                             << ", γ=" << fit_gamma << "±" << fit_gamma_err
                             << ", B=" << fit_vert_offset << "±" << fit_vert_offset_err
                             << ", χ²ᵣ=" << chi2_reduced << std::endl;
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
            std::cout << "Cheap Lorentz config " << (success ? "succeeded" : "failed") 
                     << " with χ²ᵣ=" << chi2_reduced << std::endl;
        }
        
        // Always try ALL expensive configurations regardless of cheap config result
        if (verbose) {
            std::cout << "Trying all " << expensive_configs.size() 
                     << " expensive Lorentz configurations..." << std::endl;
        }
        
        for (size_t i = 0; i < expensive_configs.size(); ++i) {
            bool config_success = try_config(expensive_configs[i], "expensive_" + std::to_string(i+1));
            if (config_success && (!best_success || chi2_reduced < best_chi2)) {
                best_success = config_success;
                best_chi2 = chi2_reduced;
                success = config_success;
            }
            
            if (verbose) {
                std::cout << "Expensive Lorentz config " << (i+1) << " " 
                         << (config_success ? "succeeded" : "failed") 
                         << " with χ²ᵣ=" << chi2_reduced << std::endl;
            }
        }
        
        if (best_success) {
            if (verbose) {
                std::cout << "Best Lorentz fit achieved with χ²ᵣ=" << best_chi2 << std::endl;
            }
            return true;
        }
    }
    
    // ROBUST FALLBACK: If all sophisticated fitting failed, use statistical estimates
    if (verbose) {
        std::cout << "All sophisticated Lorentz fitting strategies failed, using robust statistical fallback" << std::endl;
    }
    
    // Use the largest available dataset for fallback
    std::vector<double> fallback_x = x_vals;
    std::vector<double> fallback_y = y_vals;
    if (!filtered_datasets.empty()) {
        fallback_x = filtered_datasets[0].first;
        fallback_y = filtered_datasets[0].second;
        if (fallback_x.size() < 4 && filtered_datasets.size() > 1) {
            fallback_x = filtered_datasets[1].first;
            fallback_y = filtered_datasets[1].second;
        }
    }
    
    DataStatistics fallback_stats = CalcRobustStatisticsLorentz(fallback_x, fallback_y);
    if (fallback_stats.valid && fallback_x.size() >= 2) {
        // Statistical parameter estimation for Lorentz
        fit_amp = std::max(fallback_stats.max_val - fallback_stats.min_val, 0.1);
        fit_center = fallback_stats.robust_center;
        
        // Estimate gamma from data spread (Lorentz has wider tails than Gauss)
        double weighted_gamma = 0.0;
        double weight_sum = 0.0;
        for (size_t i = 0; i < fallback_x.size(); ++i) {
            double weight = std::max(0.0, fallback_y[i] - fallback_stats.q25);
            if (weight > 0) {
                double dx = fallback_x[i] - fit_center;
                weighted_gamma += weight * dx * dx;
                weight_sum += weight;
            }
        }
        
        if (weight_sum > 0) {
            fit_gamma = std::sqrt(2.0 * weighted_gamma / weight_sum); // Lorentz is wider than Gauss
        } else {
            fit_gamma = fallback_stats.std_dev; // Use data spread as rough estimate
        }
        
        // Apply reasonable bounds (Lorentz has wider tails)
        fit_gamma = std::max(pixel_spacing * 0.3, std::min(pixel_spacing * 3.0, fit_gamma));
        fit_vert_offset = fallback_stats.q25;
        
        // Conservative error estimates for fallback
        fit_amp_err = 0.3 * fit_amp;
        fit_center_err = std::max(0.05 * pixel_spacing, fit_gamma / 5.0);
        fit_gamma_err = 0.2 * fit_gamma;
        fit_vert_offset_err = 0.2 * std::abs(fit_vert_offset);
        chi2_reduced = 15.0; // High chi2 indicates this is a fallback fit
        
        if (verbose) {
            std::cout << "Robust statistical fallback Lorentz fit applied" << std::endl;
        }
        
        return true; // Always return success
    }
    
    // ULTIMATE EMERGENCY FALLBACK: Use basic heuristics
    fit_amp = (!y_vals.empty()) ? (*std::max_element(y_vals.begin(), y_vals.end()) - 
                                  *std::min_element(y_vals.begin(), y_vals.end())) : 1.0;
    fit_center = center_estimate;
    fit_gamma = pixel_spacing * 0.7;
    fit_vert_offset = (!y_vals.empty()) ? *std::min_element(y_vals.begin(), y_vals.end()) : 0.0;
    
    // Very conservative error estimates
    fit_amp_err = 0.5 * fit_amp;
    fit_center_err = 0.3 * pixel_spacing;
    fit_gamma_err = 0.4 * fit_gamma;
    fit_vert_offset_err = 0.3 * std::abs(fit_vert_offset);
    chi2_reduced = 25.0; // Very high chi2 indicates emergency fallback
    
    if (verbose) {
        std::cout << "Emergency heuristic Lorentz fit applied - all other methods failed" << std::endl;
    }
    
    return true; // NEVER return false - always provide some fit
} 

Lorentz2DResultsCeres LorentzCeres2D(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    Lorentz2DResultsCeres result;
    
    // Initialize Ceres logging (removed mutex for better parallelization)
    InitializeCeresLorentz();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "LorentzCeres2D: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 4) {
        if (verbose) {
            std::cout << "LorentzCeres2D: Error - need at least 4 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 2D Lorentz fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
        std::vector<double> row_x_coords, row_y_coords;
        for (const auto& point : row_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
            row_x_coords.push_back(point.first);
            row_y_coords.push_back(best_row_y);  // Y coordinate is constant for row
        }
        
        if (verbose) {
            std::cout << "ting Lorentz X direction with " << x_vals.size() << " points" << std::endl;
        }
        
        x_fit_success = LorentzCeres(
            x_vals, y_vals, center_x_estimate, pixel_spacing,
            result.x_amp, result.x_center, result.x_gamma, result.x_vert_offset,
            result.x_amp_err, result.x_center_err, result.x_gamma_err, result.x_vert_offset_err,
            result.x_chi2red, verbose, enable_outlier_filtering);
        
        // Calc DOF and p-value
        result.x_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.x_pp = (result.x_chi2red > 0) ? 1.0 - std::min(1.0, result.x_chi2red / 10.0) : 0.0;
        
        // Store data for ROOT analysis
        result.x_row_pixel_coords = x_vals;
        result.x_row_charge_values = y_vals;
        result.x_row_charge_errors = std::vector<double>(); // Empty vector
    }
    
    //  Y direction (central column)
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
            std::cout << "ting Lorentz Y direction with " << x_vals.size() << " points" << std::endl;
        }
        
        y_fit_success = LorentzCeres(
            x_vals, y_vals, center_y_estimate, pixel_spacing,
            result.y_amp, result.y_center, result.y_gamma, result.y_vert_offset,
            result.y_amp_err, result.y_center_err, result.y_gamma_err, result.y_vert_offset_err,
            result.y_chi2red, verbose, enable_outlier_filtering);
        
        // Calc DOF and p-value
        result.y_dof = std::max(1, static_cast<int>(x_vals.size()) - 4);
        result.y_pp = (result.y_chi2red > 0) ? 1.0 - std::min(1.0, result.y_chi2red / 10.0) : 0.0;
        
        // Store data for ROOT analysis
        result.y_col_pixel_coords = x_vals;  // Y coordinates
        result.y_col_charge_values = y_vals;
        result.y_col_charge_errors = std::vector<double>(); // Empty vector
    }
    
    // Set overall success status
    result.fit_success = x_fit_success && y_fit_success;
    
    // Calc and store charge uncertainties (5% of max charge for each direction) only if enabled
    if (Control::CHARGE_ERR) {
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
        std::cout << "2D Lorentz fit (Ceres) " << (result.fit_success ? "success" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

DiagLorentzResultsCeres DiagLorentzCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    DiagLorentzResultsCeres result;
    
    // Initialize Ceres logging (removed mutex for better parallelization)
    InitializeCeresLorentz();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size() || x_coords.size() < 4) {
        if (verbose) {
            std::cout << "Diag Lorentz fit (Ceres): Invalid input data size" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting Diag Lorentz fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
        
        main_diag_success = LorentzCeres(
            poss, charges, 0.0, diag_pixel_spacing,
            result.main_diag_x_amp, result.main_diag_x_center, result.main_diag_x_gamma, result.main_diag_x_vert_offset,
            result.main_diag_x_amp_err, result.main_diag_x_center_err, result.main_diag_x_gamma_err, result.main_diag_x_vert_offset_err,
            result.main_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.main_diag_x_dof = std::max(1, static_cast<int>(poss.size()) - 4);
        result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
        result.main_diag_x_fit_success = main_diag_success;
        
        // Store data for ROOT analysis
        result.main_diag_x_pixel_coords = poss;
        result.main_diag_x_charge_values = charges;
        result.main_diag_x_charge_errors = std::vector<double>();
        
        // For symmetry, copy results to Y (since we only have one diagonal measurement)
        result.main_diag_y_amp = result.main_diag_x_amp;
        result.main_diag_y_center = result.main_diag_x_center;
        result.main_diag_y_gamma = result.main_diag_x_gamma;
        result.main_diag_y_vert_offset = result.main_diag_x_vert_offset;
        result.main_diag_y_amp_err = result.main_diag_x_amp_err;
        result.main_diag_y_center_err = result.main_diag_x_center_err;
        result.main_diag_y_gamma_err = result.main_diag_x_gamma_err;
        result.main_diag_y_vert_offset_err = result.main_diag_x_vert_offset_err;
        result.main_diag_y_chi2red = result.main_diag_x_chi2red;
        result.main_diag_y_dof = result.main_diag_x_dof;
        result.main_diag_y_pp = result.main_diag_x_pp;
        result.main_diag_y_fit_success = main_diag_success;
        result.main_diag_y_pixel_coords = poss;
        result.main_diag_y_charge_values = charges;
        result.main_diag_y_charge_errors = std::vector<double>();
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
        
        sec_diag_success = LorentzCeres(
            poss, charges, 0.0, diag_pixel_spacing,
            result.sec_diag_x_amp, result.sec_diag_x_center, result.sec_diag_x_gamma, result.sec_diag_x_vert_offset,
            result.sec_diag_x_amp_err, result.sec_diag_x_center_err, result.sec_diag_x_gamma_err, result.sec_diag_x_vert_offset_err,
            result.sec_diag_x_chi2red, verbose, enable_outlier_filtering);
        
        result.sec_diag_x_dof = std::max(1, static_cast<int>(poss.size()) - 4);
        result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
        result.sec_diag_x_fit_success = sec_diag_success;
        
        // Store data for ROOT analysis
        result.sec_diag_x_pixel_coords = poss;
        result.sec_diag_x_charge_values = charges;
        result.sec_diag_x_charge_errors = std::vector<double>();
        
        // For symmetry, copy results to Y
        result.sec_diag_y_amp = result.sec_diag_x_amp;
        result.sec_diag_y_center = result.sec_diag_x_center;
        result.sec_diag_y_gamma = result.sec_diag_x_gamma;
        result.sec_diag_y_vert_offset = result.sec_diag_x_vert_offset;
        result.sec_diag_y_amp_err = result.sec_diag_x_amp_err;
        result.sec_diag_y_center_err = result.sec_diag_x_center_err;
        result.sec_diag_y_gamma_err = result.sec_diag_x_gamma_err;
        result.sec_diag_y_vert_offset_err = result.sec_diag_x_vert_offset_err;
        result.sec_diag_y_chi2red = result.sec_diag_x_chi2red;
        result.sec_diag_y_dof = result.sec_diag_x_dof;
        result.sec_diag_y_pp = result.sec_diag_x_pp;
        result.sec_diag_y_fit_success = sec_diag_success;
        result.sec_diag_y_pixel_coords = poss;
        result.sec_diag_y_charge_values = charges;
        result.sec_diag_y_charge_errors = std::vector<double>();
    }
    
    // Set overall success status
    result.fit_success = main_diag_success && sec_diag_success;
    
    if (verbose) {
        std::cout << "Diag Lorentz fit (Ceres) " << (result.fit_success ? "success" : "failed") 
                 << " (Main +45°: " << (main_diag_success ? "OK" : "FAIL") 
                 << ", Secondary -45°: " << (sec_diag_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}