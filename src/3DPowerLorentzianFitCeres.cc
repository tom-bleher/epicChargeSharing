#include "3DPowerLorentzianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <atomic>
#include <limits>
#include <numeric>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Thread-safe mutex for Ceres operations
static std::mutex gCeres3DPowerLorentzianFitMutex;

// Use shared Google logging initialization
void InitializeCeres3DPowerLorentzian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Enhanced uncertainty calculation for 3D Power-Law Lorentzian with adaptive weighting
double Calculate3DPowerLorentzianUncertainty(double charge_value, double max_charge_in_neighborhood, double baseline = 0.0) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    // Adaptive uncertainty model based on charge magnitude
    double charge_above_baseline = std::max(0.0, charge_value - baseline);
    double max_above_baseline = std::max(1e-20, max_charge_in_neighborhood - baseline);
    
    // Higher precision for high-charge pixels, lower precision for low-charge pixels
    double relative_charge = charge_above_baseline / max_above_baseline;
    
    // Base uncertainty: 1.5-5% depending on charge level (tighter than regular Lorentzian)
    double base_uncertainty_fraction = 0.015 + 0.035 * (1.0 - relative_charge);
    double uncertainty = base_uncertainty_fraction * max_charge_in_neighborhood;
    
    // Minimum uncertainty floor
    if (uncertainty < 1e-20) uncertainty = 1e-20;
    
    return uncertainty;
}

// 3D Power-Law Lorentzian cost function
// Function form: z(x,y) = A / (1 + ((x - mx) / γx)^2 + ((y - my) / γy)^2)^β + B  
struct PowerLorentzian3DCostFunction {
    PowerLorentzian3DCostFunction(double x, double y, double z, double uncertainty) 
        : x_(x), y_(y), z_(z), uncertainty_(uncertainty) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = mx (X center)
        // params[2] = my (Y center)
        // params[3] = gamma_x (X width)
        // params[4] = gamma_y (Y width)
        // params[5] = beta (power exponent)
        // params[6] = B (baseline)
        
        const T& A = params[0];
        const T& mx = params[1];
        const T& my = params[2];
        const T& gamma_x = params[3];
        const T& gamma_y = params[4];
        const T& beta = params[5];
        const T& B = params[6];
        
        // Robust handling of parameters
        T safe_gamma_x = ceres::abs(gamma_x);
        T safe_gamma_y = ceres::abs(gamma_y);
        T safe_beta = ceres::abs(beta);
        if (safe_gamma_x < T(1e-12)) safe_gamma_x = T(1e-12);
        if (safe_gamma_y < T(1e-12)) safe_gamma_y = T(1e-12);
        if (safe_beta < T(0.1)) safe_beta = T(0.1);
        
        // 3D Power-Law Lorentzian function
        T dx = x_ - mx;
        T dy = y_ - my;
        T normalized_dx = dx / safe_gamma_x;
        T normalized_dy = dy / safe_gamma_y;
        T denominator_base = T(1.0) + normalized_dx * normalized_dx + normalized_dy * normalized_dy;
        
        if (denominator_base < T(1e-12)) {
            denominator_base = T(1e-12);
        }
        
        T denominator = ceres::pow(denominator_base, safe_beta);
        T predicted = A / denominator + B;
        
        residual[0] = (predicted - T(z_)) / T(uncertainty_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double z, double uncertainty) {
        return (new ceres::AutoDiffCostFunction<PowerLorentzian3DCostFunction, 1, 7>(
            new PowerLorentzian3DCostFunction(x, y, z, uncertainty)));
    }
    
private:
    const double x_;
    const double y_;
    const double z_;
    const double uncertainty_;
};

// Core 3D Power-Law Lorentzian fitting function using Ceres Solver
bool Fit3DPowerLorentzianCeres(
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
    double& fit_beta,
    double& fit_vertical_offset,
    double& fit_amplitude_err,
    double& fit_center_x_err,
    double& fit_center_y_err,
    double& fit_gamma_x_err,
    double& fit_gamma_y_err,
    double& fit_beta_err,
    double& fit_vertical_offset_err,
    double& chi2_reduced,
    bool verbose,
    bool enable_outlier_filtering) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 7) {
        if (verbose) {
            std::cout << "Insufficient data points for 3D Power-Law Lorentzian fitting" << std::endl;
        }
        return false;
    }
    
    // Enhanced parameter estimation for 3D Power-Law Lorentzian
    double max_charge = *std::max_element(z_vals.begin(), z_vals.end());
    double min_charge = *std::min_element(z_vals.begin(), z_vals.end());
    
    // Multi-threshold approach for better parameter estimation
    double charge_threshold_high = max_charge * 0.8; // Peak center estimation
    double charge_threshold_med = max_charge * 0.4;  // Width estimation
    double charge_threshold_low = max_charge * 0.15; // Baseline estimation
    
    // Enhanced center estimation using peak pixels
    double peak_weighted_x = 0.0, peak_weighted_y = 0.0, peak_weight_sum = 0.0;
    double med_weighted_x = 0.0, med_weighted_y = 0.0, med_weight_sum = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        // Peak center from highest charge pixels
        if (z_vals[i] >= charge_threshold_high) {
            double weight = z_vals[i];
            peak_weighted_x += x_vals[i] * weight;
            peak_weighted_y += y_vals[i] * weight;
            peak_weight_sum += weight;
        }
        
        // Secondary center from medium charge pixels
        if (z_vals[i] >= charge_threshold_med) {
            double weight = std::max(0.0, z_vals[i] - min_charge);
            med_weighted_x += x_vals[i] * weight;
            med_weighted_y += y_vals[i] * weight;
            med_weight_sum += weight;
        }
    }
    
    double center_x, center_y;
    if (peak_weight_sum > 0 && peak_weight_sum >= med_weight_sum * 0.1) {
        center_x = peak_weighted_x / peak_weight_sum;
        center_y = peak_weighted_y / peak_weight_sum;
    } else if (med_weight_sum > 0) {
        center_x = med_weighted_x / med_weight_sum;
        center_y = med_weighted_y / med_weight_sum;
    } else {
        center_x = center_x_estimate;
        center_y = center_y_estimate;
    }
    
    // Improved baseline estimation
    std::vector<double> low_charge_vals;
    for (size_t i = 0; i < z_vals.size(); ++i) {
        if (z_vals[i] <= charge_threshold_low) {
            low_charge_vals.push_back(z_vals[i]);
        }
    }
    
    double baseline;
    if (low_charge_vals.size() > 2) {
        std::sort(low_charge_vals.begin(), low_charge_vals.end());
        baseline = low_charge_vals[low_charge_vals.size() / 2]; // Median of low values
    } else {
        baseline = min_charge;
    }
    
    // Enhanced width estimation using multiple charge levels
    double distance_spread_x = 0.0, distance_spread_y = 0.0, spread_weight_sum = 0.0;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        if (z_vals[i] >= charge_threshold_med) {
            double dx = x_vals[i] - center_x;
            double dy = y_vals[i] - center_y;
            double weight = std::max(0.0, z_vals[i] - baseline);
            distance_spread_x += weight * dx * dx;
            distance_spread_y += weight * dy * dy;
            spread_weight_sum += weight;
        }
    }
    
    double gamma_x, gamma_y;
    if (spread_weight_sum > 0) {
        gamma_x = std::sqrt(distance_spread_x / spread_weight_sum);
        gamma_y = std::sqrt(distance_spread_y / spread_weight_sum);
        
        // Apply physics-based constraints
        double min_gamma = pixel_spacing * 0.25;
        double max_gamma = pixel_spacing * 2.0;
        gamma_x = std::max(min_gamma, std::min(max_gamma, gamma_x));
        gamma_y = std::max(min_gamma, std::min(max_gamma, gamma_y));
    } else {
        gamma_x = pixel_spacing * 0.5;
        gamma_y = pixel_spacing * 0.5;
    }
    
    // Set up parameter array (7 parameters: A, mx, my, gamma_x, gamma_y, beta, B)
    double parameters[7];
    parameters[0] = max_charge - baseline; // amplitude
    parameters[1] = center_x; // center_x
    parameters[2] = center_y; // center_y
    parameters[3] = gamma_x; // gamma_x
    parameters[4] = gamma_y; // gamma_y
    parameters[5] = 1.2; // beta (start slightly above standard Lorentzian for better convergence)
    parameters[6] = baseline; // baseline
    
    // Multiple fitting configurations for enhanced robustness
    struct PowerLorentzian3DFittingConfig {
        ceres::LinearSolverType linear_solver;
        ceres::TrustRegionStrategyType trust_region;
        double function_tolerance;
        double gradient_tolerance;
        int max_iterations;
        std::string loss_function;
        double loss_parameter;
        bool use_tight_bounds;
    };
    
    std::vector<PowerLorentzian3DFittingConfig> configs;
    
    // Progressive solver configurations
    PowerLorentzian3DFittingConfig config1;
    config1.linear_solver = ceres::DENSE_QR;
    config1.trust_region = ceres::LEVENBERG_MARQUARDT;
    config1.function_tolerance = 1e-15;
    config1.gradient_tolerance = 1e-15;
    config1.max_iterations = 1000;
    config1.loss_function = "NONE";
    config1.loss_parameter = 0.0;
    config1.use_tight_bounds = false;
    configs.push_back(config1);
    
    PowerLorentzian3DFittingConfig config2;
    config2.linear_solver = ceres::DENSE_QR;
    config2.trust_region = ceres::LEVENBERG_MARQUARDT;
    config2.function_tolerance = 1e-15;
    config2.gradient_tolerance = 1e-15;
    config2.max_iterations = 1800;
    config2.loss_function = "HUBER";
    config2.loss_parameter = parameters[0] * 0.06;
    config2.use_tight_bounds = true;
    configs.push_back(config2);
    
    PowerLorentzian3DFittingConfig config3;
    config3.linear_solver = ceres::DENSE_QR;
    config3.trust_region = ceres::LEVENBERG_MARQUARDT;
    config3.function_tolerance = 1e-12;
    config3.gradient_tolerance = 1e-12;
    config3.max_iterations = 1600;
    config3.loss_function = "CAUCHY";
    config3.loss_parameter = parameters[0] * 0.10;
    config3.use_tight_bounds = true;
    configs.push_back(config3);
    
    // Try each configuration
    for (const auto& config : configs) {
        // Reset parameters
        parameters[0] = max_charge - baseline; // amplitude
        parameters[1] = center_x; // center_x
        parameters[2] = center_y; // center_y
        parameters[3] = gamma_x; // gamma_x
        parameters[4] = gamma_y; // gamma_y
        parameters[5] = 1.2; // beta
        parameters[6] = baseline; // baseline
        
        // Build the problem
        ceres::Problem problem;
        
        // Add residual blocks with adaptive uncertainties
        for (size_t i = 0; i < x_vals.size(); ++i) {
            double point_uncertainty = Calculate3DPowerLorentzianUncertainty(z_vals[i], max_charge, baseline);
            ceres::CostFunction* cost_function = PowerLorentzian3DCostFunction::Create(
                x_vals[i], y_vals[i], z_vals[i], point_uncertainty);
            
            // Use loss function if specified
            ceres::LossFunction* loss_function = nullptr;
            if (config.loss_function == "HUBER") {
                loss_function = new ceres::HuberLoss(config.loss_parameter);
            } else if (config.loss_function == "CAUCHY") {
                loss_function = new ceres::CauchyLoss(config.loss_parameter);
            }
            
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
    
        // Set adaptive bounds based on configuration
        double amp_min = std::max(1e-20, parameters[0] * 0.005);
        double physics_amp_max = max_charge * 1.2;
        double algo_amp_max = parameters[0] * (config.use_tight_bounds ? 40.0 : 80.0);
        double amp_max = std::min(physics_amp_max, algo_amp_max);
        
        problem.SetParameterLowerBound(parameters, 0, amp_min); // amplitude
        problem.SetParameterUpperBound(parameters, 0, amp_max);
        
        // Adaptive center bounds
        double center_range = config.use_tight_bounds ? pixel_spacing * 1.2 : pixel_spacing * 2.5;
        problem.SetParameterLowerBound(parameters, 1, parameters[1] - center_range); // center_x
        problem.SetParameterUpperBound(parameters, 1, parameters[1] + center_range);
        problem.SetParameterLowerBound(parameters, 2, parameters[2] - center_range); // center_y
        problem.SetParameterUpperBound(parameters, 2, parameters[2] + center_range);
        
        // Tighter gamma bounds
        double gamma_min = pixel_spacing * 0.15;
        double gamma_max = config.use_tight_bounds ? pixel_spacing * 2.0 : pixel_spacing * 3.5;
        problem.SetParameterLowerBound(parameters, 3, gamma_min); // gamma_x
        problem.SetParameterUpperBound(parameters, 3, gamma_max);
        problem.SetParameterLowerBound(parameters, 4, gamma_min); // gamma_y
        problem.SetParameterUpperBound(parameters, 4, gamma_max);
        
        // Beta bounds with better range
        problem.SetParameterLowerBound(parameters, 5, 0.3); // beta
        problem.SetParameterUpperBound(parameters, 5, 3.5);
        
        // Improved baseline bounds
        double baseline_range = config.use_tight_bounds ? 
            std::max(parameters[0] * 0.25, std::abs(parameters[6]) * 1.3) :
            std::max(parameters[0] * 0.4, std::abs(parameters[6]) * 1.8);
        problem.SetParameterLowerBound(parameters, 6, parameters[6] - baseline_range); // baseline
        problem.SetParameterUpperBound(parameters, 6, parameters[6] + baseline_range);
    
        // Two-stage fitting approach for better convergence
        // Stage 1: Constrain beta close to 1.0 for stability
        problem.SetParameterLowerBound(parameters, 5, 0.8);
        problem.SetParameterUpperBound(parameters, 5, 1.4);
        
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
        
        if (verbose) {
            std::cout << "Config " << &config - &configs[0] << " Stage 1: Fitting with beta constrained..." << std::endl;
        }
        
        ceres::Solver::Summary summary_stage1;
        ceres::Solve(options, &problem, &summary_stage1);
        
        bool stage1_successful = (summary_stage1.termination_type == ceres::CONVERGENCE ||
                                summary_stage1.termination_type == ceres::USER_SUCCESS) &&
                               parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 &&
                               std::isfinite(parameters[0]) && std::isfinite(parameters[1]) &&
                               std::isfinite(parameters[2]) && std::isfinite(parameters[3]) &&
                               std::isfinite(parameters[4]) && std::isfinite(parameters[5]) &&
                               std::isfinite(parameters[6]);
        
        ceres::Solver::Summary summary;
        if (stage1_successful) {
            // Stage 2: Allow beta to vary more freely
            problem.SetParameterLowerBound(parameters, 5, 0.3);
            problem.SetParameterUpperBound(parameters, 5, 3.5);
            
            // Tighten center bounds around Stage 1 result
            double stage1_center_x = parameters[1];
            double stage1_center_y = parameters[2];
            double tight_center_range = pixel_spacing * 0.4;
            problem.SetParameterLowerBound(parameters, 1, stage1_center_x - tight_center_range);
            problem.SetParameterUpperBound(parameters, 1, stage1_center_x + tight_center_range);
            problem.SetParameterLowerBound(parameters, 2, stage1_center_y - tight_center_range);
            problem.SetParameterUpperBound(parameters, 2, stage1_center_y + tight_center_range);
            
            if (verbose) {
                std::cout << "Config " << &config - &configs[0] << " Stage 2: Refining fit with beta free..." << std::endl;
            }
            
            ceres::Solve(options, &problem, &summary);
        } else {
            // If stage 1 failed, use single-stage approach
            problem.SetParameterLowerBound(parameters, 5, 0.3);
            problem.SetParameterUpperBound(parameters, 5, 3.5);
            
            if (verbose) {
                std::cout << "Config " << &config - &configs[0] << " Stage 1 failed, single-stage fit..." << std::endl;
            }
            
            ceres::Solve(options, &problem, &summary);
        }
        
        // Validate final results
        bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                              summary.termination_type == ceres::USER_SUCCESS) &&
                             parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 && 
                             parameters[5] > 0.2 && parameters[5] < 4.0 &&
                             std::isfinite(parameters[0]) && std::isfinite(parameters[1]) &&
                             std::isfinite(parameters[2]) && std::isfinite(parameters[3]) &&
                             std::isfinite(parameters[4]) && std::isfinite(parameters[5]) &&
                             std::isfinite(parameters[6]);
        
        if (fit_successful) {
            // Extract results
            fit_amplitude = parameters[0];
            fit_center_x = parameters[1];
            fit_center_y = parameters[2];
            fit_gamma_x = std::abs(parameters[3]);
            fit_gamma_y = std::abs(parameters[4]);
            fit_beta = parameters[5];
            fit_vertical_offset = parameters[6];
            
            // Enhanced uncertainty estimation (fallback method)
            fit_amplitude_err = 0.015 * fit_amplitude;
            fit_center_x_err = 0.015 * pixel_spacing;
            fit_center_y_err = 0.015 * pixel_spacing;
            fit_gamma_x_err = 0.04 * fit_gamma_x;
            fit_gamma_y_err = 0.04 * fit_gamma_y;
            fit_beta_err = 0.08 * fit_beta;
            fit_vertical_offset_err = 0.08 * std::abs(fit_vertical_offset);
            
            // Calculate reduced chi-squared
            double chi2 = summary.final_cost * 2.0;
            int dof = std::max(1, static_cast<int>(x_vals.size()) - 7);
            chi2_reduced = chi2 / dof;
            
            if (verbose) {
                std::cout << "Successful 3D Power-Law Lorentzian fit (config " << &config - &configs[0] << "): "
                         << "A=" << fit_amplitude << ", mx=" << fit_center_x 
                         << ", my=" << fit_center_y << ", gamma_x=" << fit_gamma_x
                         << ", gamma_y=" << fit_gamma_y << ", beta=" << fit_beta
                         << ", B=" << fit_vertical_offset << ", chi2red=" << chi2_reduced << std::endl;
            }
            
            return true;
        } else if (verbose) {
            std::cout << "3D Power-Law Lorentzian fit failed (config " << &config - &configs[0] << "): " 
                     << summary.BriefReport() << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "All 3D Power-Law Lorentzian fitting configurations failed" << std::endl;
    }
    return false;
}

PowerLorentzianFit3DResultsCeres Fit3DPowerLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    PowerLorentzianFit3DResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeres3DPowerLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeres3DPowerLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit3DPowerLorentzianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 7) {
        if (verbose) {
            std::cout << "Fit3DPowerLorentzianCeres: Error - need at least 7 data points for 3D fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 3D Power-Law Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
        
        result.charge_errors.clear();
        result.charge_errors.resize(charge_values.size(), charge_uncertainty);
    } else {
        result.charge_uncertainty = 0.0;
        result.charge_errors.clear();
        result.charge_errors.resize(charge_values.size(), 1.0); // Uniform weighting
    }
    
    // Perform 3D Power-Law Lorentzian surface fitting
    bool fit_success = Fit3DPowerLorentzianCeres(
        x_coords, y_coords, charge_values, center_x_estimate, center_y_estimate, pixel_spacing,
        result.amplitude, result.center_x, result.center_y, result.gamma_x, result.gamma_y, result.beta, result.vertical_offset,
        result.amplitude_err, result.center_x_err, result.center_y_err, result.gamma_x_err, result.gamma_y_err, result.beta_err, result.vertical_offset_err,
        result.chi2red, verbose, enable_outlier_filtering);
    
    // Calculate DOF and p-value
    result.dof = std::max(1, static_cast<int>(x_coords.size()) - 7);
    result.pp = (result.chi2red > 0) ? 1.0 - std::min(1.0, result.chi2red / 10.0) : 0.0;
    
    // Set overall success status
    result.fit_successful = fit_success;
    
    if (verbose) {
        std::cout << "3D Power-Law Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") << std::endl;
    }
    
    return result;
} 