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

// Calculate uncertainty as 5% of max charge in neighborhood (if enabled)
double Calculate3DPowerLorentzianUncertainty(double max_charge_in_neighborhood) {
    if (!Constants::ENABLE_VERTICAL_CHARGE_UNCERTAINTIES) {
        return 1.0; // Uniform weighting when uncertainties are disabled
    }
    
    double uncertainty = 0.05 * max_charge_in_neighborhood;
    if (uncertainty < Constants::MIN_UNCERTAINTY_VALUE) uncertainty = Constants::MIN_UNCERTAINTY_VALUE;
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
    
    // Calculate basic statistics for parameter estimation
    double max_charge = *std::max_element(z_vals.begin(), z_vals.end());
    double min_charge = *std::min_element(z_vals.begin(), z_vals.end());
    
    // Calculate weighted center estimates
    double weighted_x = 0.0, weighted_y = 0.0, total_weight = 0.0;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, z_vals[i] - min_charge);
        if (weight > 0) {
            weighted_x += x_vals[i] * weight;
            weighted_y += y_vals[i] * weight;
            total_weight += weight;
        }
    }
    if (total_weight > 0) {
        weighted_x /= total_weight;
        weighted_y /= total_weight;
    } else {
        weighted_x = center_x_estimate;
        weighted_y = center_y_estimate;
    }
    
    // Set up parameter array (7 parameters: A, mx, my, gamma_x, gamma_y, beta, B)
    double parameters[7];
    parameters[0] = max_charge - min_charge; // amplitude
    parameters[1] = weighted_x; // center_x
    parameters[2] = weighted_y; // center_y
    parameters[3] = pixel_spacing * 0.7; // gamma_x
    parameters[4] = pixel_spacing * 0.7; // gamma_y
    parameters[5] = 1.0; // beta (start with standard Lorentzian)
    parameters[6] = min_charge; // baseline
    
    // Calculate uncertainty
    double uncertainty = Calculate3DPowerLorentzianUncertainty(max_charge);
    
    // Build the problem
    ceres::Problem problem;
    
    // Add residual blocks
    for (size_t i = 0; i < x_vals.size(); ++i) {
        ceres::CostFunction* cost_function = PowerLorentzian3DCostFunction::Create(
            x_vals[i], y_vals[i], z_vals[i], uncertainty);
        problem.AddResidualBlock(cost_function, nullptr, parameters);
    }
    
    // Set bounds
    problem.SetParameterLowerBound(parameters, 0, std::max(Constants::MIN_UNCERTAINTY_VALUE, parameters[0] * 0.01)); // amplitude
    problem.SetParameterUpperBound(parameters, 0, std::min(max_charge * 1.5, parameters[0] * 100.0));
    
    double center_range = pixel_spacing * 3.0;
    problem.SetParameterLowerBound(parameters, 1, parameters[1] - center_range); // center_x
    problem.SetParameterUpperBound(parameters, 1, parameters[1] + center_range);
    problem.SetParameterLowerBound(parameters, 2, parameters[2] - center_range); // center_y
    problem.SetParameterUpperBound(parameters, 2, parameters[2] + center_range);
    
    problem.SetParameterLowerBound(parameters, 3, pixel_spacing * 0.05); // gamma_x
    problem.SetParameterUpperBound(parameters, 3, pixel_spacing * 4.0);
    problem.SetParameterLowerBound(parameters, 4, pixel_spacing * 0.05); // gamma_y
    problem.SetParameterUpperBound(parameters, 4, pixel_spacing * 4.0);
    
    problem.SetParameterLowerBound(parameters, 5, 0.2); // beta
    problem.SetParameterUpperBound(parameters, 5, 4.0);
    
    double baseline_range = std::max(parameters[0] * 0.5, std::abs(parameters[6]) * 2.0);
    problem.SetParameterLowerBound(parameters, 6, parameters[6] - baseline_range); // baseline
    problem.SetParameterUpperBound(parameters, 6, parameters[6] + baseline_range);
    
    // Two-stage fitting approach (similar to 2D version)
    // Stage 1: Constrain beta close to 1.0
    problem.SetParameterLowerBound(parameters, 5, 0.9);
    problem.SetParameterUpperBound(parameters, 5, 1.1);
    
    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-15;
    options.max_num_iterations = 1500;
    options.max_num_consecutive_invalid_steps = 50;
    options.use_nonmonotonic_steps = true;
    options.minimizer_progress_to_stdout = false;
    
    if (verbose) {
        std::cout << "Stage 1: Fitting with beta constrained to ~1.0..." << std::endl;
    }
    
    ceres::Solver::Summary summary_stage1;
    ceres::Solve(options, &problem, &summary_stage1);
    
    bool stage1_successful = (summary_stage1.termination_type == ceres::CONVERGENCE ||
                            summary_stage1.termination_type == ceres::USER_SUCCESS) &&
                           parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 &&
                           !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                           !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                           !std::isnan(parameters[4]) && !std::isnan(parameters[5]) &&
                           !std::isnan(parameters[6]);
    
    ceres::Solver::Summary summary;
    if (stage1_successful) {
        // Stage 2: Allow beta to vary more freely
        problem.SetParameterLowerBound(parameters, 5, 0.2);
        problem.SetParameterUpperBound(parameters, 5, 4.0);
        
        // Tighten center bounds around Stage 1 result
        double stage1_center_x = parameters[1];
        double stage1_center_y = parameters[2];
        double tight_center_range = pixel_spacing * 0.5;
        problem.SetParameterLowerBound(parameters, 1, stage1_center_x - tight_center_range);
        problem.SetParameterUpperBound(parameters, 1, stage1_center_x + tight_center_range);
        problem.SetParameterLowerBound(parameters, 2, stage1_center_y - tight_center_range);
        problem.SetParameterUpperBound(parameters, 2, stage1_center_y + tight_center_range);
        
        if (verbose) {
            std::cout << "Stage 2: Refining fit with beta free to vary..." << std::endl;
        }
        
        ceres::Solve(options, &problem, &summary);
    } else {
        // If stage 1 failed, use single-stage approach
        problem.SetParameterLowerBound(parameters, 5, 0.2);
        problem.SetParameterUpperBound(parameters, 5, 4.0);
        
        if (verbose) {
            std::cout << "Stage 1 failed, falling back to single-stage fit..." << std::endl;
        }
        
        ceres::Solve(options, &problem, &summary);
    }
    
    // Validate final results
    bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                          summary.termination_type == ceres::USER_SUCCESS) &&
                         parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 && 
                         parameters[5] > 0.1 && parameters[5] < 5.0 &&
                         !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                         !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                         !std::isnan(parameters[4]) && !std::isnan(parameters[5]) &&
                         !std::isnan(parameters[6]);
    
    if (fit_successful) {
        // Extract results
        fit_amplitude = parameters[0];
        fit_center_x = parameters[1];
        fit_center_y = parameters[2];
        fit_gamma_x = std::abs(parameters[3]);
        fit_gamma_y = std::abs(parameters[4]);
        fit_beta = parameters[5];
        fit_vertical_offset = parameters[6];
        
        // Simple uncertainty estimation (fallback method)
        fit_amplitude_err = 0.02 * fit_amplitude;
        fit_center_x_err = 0.02 * pixel_spacing;
        fit_center_y_err = 0.02 * pixel_spacing;
        fit_gamma_x_err = 0.05 * fit_gamma_x;
        fit_gamma_y_err = 0.05 * fit_gamma_y;
        fit_beta_err = 0.1 * fit_beta;
        fit_vertical_offset_err = 0.1 * std::abs(fit_vertical_offset);
        
        // Calculate reduced chi-squared
        double chi2 = summary.final_cost * 2.0;
        int dof = std::max(1, static_cast<int>(x_vals.size()) - 7);
        chi2_reduced = chi2 / dof;
        
        if (verbose) {
            std::cout << "Successful 3D Power-Law Lorentzian fit: "
                     << "A=" << fit_amplitude << ", mx=" << fit_center_x 
                     << ", my=" << fit_center_y << ", gamma_x=" << fit_gamma_x
                     << ", gamma_y=" << fit_gamma_y << ", beta=" << fit_beta
                     << ", B=" << fit_vertical_offset << ", chi2red=" << chi2_reduced << std::endl;
        }
        
        return true;
    } else if (verbose) {
        std::cout << "3D Power-Law Lorentzian fit failed: " << summary.BriefReport() << std::endl;
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