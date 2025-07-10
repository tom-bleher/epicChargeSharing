#include "3DPowerLorentzianFitCeres.hh"
#include "CeresLoggingInit.hh"
#include "Constants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <iostream>
// Removed mutex include - no longer needed for parallelization
#include <atomic>
#include <limits>
#include <numeric>

// Ceres Solver includes
#include "ceres/ceres.h"
#include "glog/logging.h"

// Thread counter for debugging (removed mutex for better parallelization)

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
    
    // Calculate uncertainty
    double uncertainty = Calculate3DPowerLorentzianUncertainty(max_charge);
    
    // OPTIMIZED: Cheap config first with early exit based on quality (Step 1 from optimize.md)
    struct PowerLorentzian3DFittingConfig {
        ceres::LinearSolverType linear_solver;
        ceres::TrustRegionStrategyType trust_region;
        double function_tolerance;
        double gradient_tolerance;
        int max_iterations;
        std::string loss_function;
        double loss_parameter;
    };
    
    // Stage 1: Cheap configuration (as per optimize.md section 4.1)
    PowerLorentzian3DFittingConfig cheap_config = {
        ceres::DENSE_NORMAL_CHOLESKY, ceres::LEVENBERG_MARQUARDT, 
        1e-10, 1e-10, 400, "NONE", 0.0
    };
    
    // Stage 2: Expensive fallback configurations (only if needed)
    const std::vector<PowerLorentzian3DFittingConfig> expensive_configs = {
        {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "HUBER", (max_charge - min_charge) * 0.1},
        {ceres::DENSE_QR, ceres::LEVENBERG_MARQUARDT, 1e-12, 1e-12, 1500, "CAUCHY", (max_charge - min_charge) * 0.16}
    };
    
    // Try cheap config first
    auto try_config = [&](const PowerLorentzian3DFittingConfig& config, const std::string& stage_name) -> bool {
        if (verbose) {
            std::cout << "Trying 3D Power Lorentzian " << stage_name << " configuration..." << std::endl;
        }
        
        // STEP 2 OPTIMIZATION: Hierarchical multi-start budget for 3D Power Lorentzian
        // Start with base estimate, only add 2 perturbations if χ²ᵣ > 2.0
        // Expected: ×4-5 speed-up, average #Ceres solves/fit ≤10
        
        struct ParameterSet {
            double params[7];
            std::string description;
        };
        
        std::vector<ParameterSet> initial_guesses;
        
        // ALWAYS start with base estimate first
        ParameterSet base_set;
        base_set.params[0] = max_charge - min_charge; // amplitude
        base_set.params[1] = weighted_x; // center_x
        base_set.params[2] = weighted_y; // center_y
        base_set.params[3] = pixel_spacing * 0.7; // gamma_x
        base_set.params[4] = pixel_spacing * 0.7; // gamma_y
        base_set.params[5] = 1.0; // beta (start with standard Lorentzian)
        base_set.params[6] = min_charge; // baseline
        base_set.description = "base_estimate";
        initial_guesses.push_back(base_set);
        
        double best_cost = std::numeric_limits<double>::max();
        double best_parameters[7];
        bool any_success = false;
        std::string best_description;
        double best_chi2_reduced = std::numeric_limits<double>::max();
        
        // Data characteristics for adaptive bounds
        double data_spread_x = *std::max_element(x_vals.begin(), x_vals.end()) - 
                             *std::min_element(x_vals.begin(), x_vals.end());
        double data_spread_y = *std::max_element(y_vals.begin(), y_vals.end()) - 
                             *std::min_element(y_vals.begin(), y_vals.end());
        double z_median = z_vals[z_vals.size()/2]; // Approximate median for outlier detection
        double outlier_ratio = 0.0;
        if (z_vals.size() > 0) {
            int outlier_count = 0;
            double outlier_threshold = z_median + 2.0 * (max_charge - min_charge) * 0.1;
            for (double val : z_vals) {
                if (val > outlier_threshold) outlier_count++;
            }
            outlier_ratio = static_cast<double>(outlier_count) / z_vals.size();
        }
        
        // Try base estimate first
        for (const auto& guess : initial_guesses) {
            double parameters[7];
            parameters[0] = guess.params[0];
            parameters[1] = guess.params[1];
            parameters[2] = guess.params[2];
            parameters[3] = guess.params[3];
            parameters[4] = guess.params[4];
            parameters[5] = guess.params[5];
            parameters[6] = guess.params[6];
        
            ceres::Problem problem;
            
            for (size_t i = 0; i < x_vals.size(); ++i) {
                ceres::CostFunction* cost_function = PowerLorentzian3DCostFunction::Create(
                    x_vals[i], y_vals[i], z_vals[i], uncertainty);
                problem.AddResidualBlock(cost_function, nullptr, parameters);
            }
            
            // Set adaptive bounds
            double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                    std::max(parameters[0] * 0.01, std::abs(min_charge) * 0.1));
            double amp_max = std::max(max_charge * 1.5, std::max(parameters[0] * 100.0, 1e-10));
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
            
            double gamma_min_x = std::max(pixel_spacing * 0.05, data_spread_x * 0.01);
            double gamma_max_x = std::min(pixel_spacing * 4.0, data_spread_x * 0.8);
            double gamma_min_y = std::max(pixel_spacing * 0.05, data_spread_y * 0.01);
            double gamma_max_y = std::min(pixel_spacing * 4.0, data_spread_y * 0.8);
            
            problem.SetParameterLowerBound(parameters, 3, gamma_min_x);
            problem.SetParameterUpperBound(parameters, 3, gamma_max_x);
            problem.SetParameterLowerBound(parameters, 4, gamma_min_y);
            problem.SetParameterUpperBound(parameters, 4, gamma_max_y);
            
            double beta_min = (outlier_ratio > 0.2) ? 0.5 : 0.2;
            double beta_max = (outlier_ratio > 0.2) ? 2.5 : 4.0;
            problem.SetParameterLowerBound(parameters, 5, beta_min);
            problem.SetParameterUpperBound(parameters, 5, beta_max);
            
            double charge_range = std::abs(max_charge - min_charge);
            double baseline_range = std::max(charge_range * 0.5, 
                                           std::max(std::abs(parameters[6]) * 2.0, 1e-12));
            problem.SetParameterLowerBound(parameters, 6, parameters[6] - baseline_range);
            problem.SetParameterUpperBound(parameters, 6, parameters[6] + baseline_range);
        
            // Two-stage fitting approach: Stage 1 - Constrain beta close to 1.0
            problem.SetParameterLowerBound(parameters, 5, 0.9);
            problem.SetParameterUpperBound(parameters, 5, 1.1);
            
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
                
                double stage1_center_x = parameters[1];
                double stage1_center_y = parameters[2];
                double tight_center_range = pixel_spacing * 1.0;
                problem.SetParameterLowerBound(parameters, 1, stage1_center_x - tight_center_range);
                problem.SetParameterUpperBound(parameters, 1, stage1_center_x + tight_center_range);
                problem.SetParameterLowerBound(parameters, 2, stage1_center_y - tight_center_range);
                problem.SetParameterUpperBound(parameters, 2, stage1_center_y + tight_center_range);
                
                ceres::Solve(options, &problem, &summary);
            } else {
                problem.SetParameterLowerBound(parameters, 5, 0.2);
                problem.SetParameterUpperBound(parameters, 5, 4.0);
                ceres::Solve(options, &problem, &summary);
            }
            
            bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                                  summary.termination_type == ceres::USER_SUCCESS) &&
                                 parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 && 
                                 parameters[5] > 0.1 && parameters[5] < 5.0 &&
                                 !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                 !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                                 !std::isnan(parameters[4]) && !std::isnan(parameters[5]) &&
                                 !std::isnan(parameters[6]);
            
            if (fit_successful) {
                double cost = summary.final_cost;
                double chi2 = cost * 2.0;
                int dof = std::max(1, static_cast<int>(x_vals.size()) - 7);
                double chi2_red = chi2 / dof;
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_chi2_reduced = chi2_red;
                    std::copy(parameters, parameters + 7, best_parameters);
                    best_description = guess.description;
                    any_success = true;
                    
                    if (verbose) {
                        std::cout << "New best 3D Power Lorentzian result from " << guess.description 
                                 << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                    }
                }
            }
        }
        
        // ALWAYS add perturbations regardless of chi-squared quality
        if (any_success) {
            if (verbose) {
                std::cout << "Base 3D Power Lorentzian fit χ²ᵣ=" << best_chi2_reduced << ", trying perturbations..." << std::endl;
            }
            
            // Add exactly 2 perturbations for 3D Power Lorentzian case
            const std::vector<double> perturbation_factors = {0.7, 1.3};
            const std::vector<double> beta_variations = {0.8, 1.2};
            
            for (size_t i = 0; i < perturbation_factors.size(); ++i) {
                double factor = perturbation_factors[i];
                double beta_factor = beta_variations[i];
                
                ParameterSet perturbed_set;
                perturbed_set.params[0] = base_set.params[0] * factor;
                perturbed_set.params[1] = base_set.params[1] + (factor - 1.0) * pixel_spacing * 0.3;
                perturbed_set.params[2] = base_set.params[2] + (factor - 1.0) * pixel_spacing * 0.3;
                perturbed_set.params[3] = base_set.params[3] * std::sqrt(factor);
                perturbed_set.params[4] = base_set.params[4] * std::sqrt(factor);
                perturbed_set.params[5] = std::max(0.3, std::min(3.5, base_set.params[5] * beta_factor));
                perturbed_set.params[6] = base_set.params[6] * (0.8 + 0.4 * factor);
                perturbed_set.description = "3d_power_perturbation_" + std::to_string(factor) + "_beta_" + std::to_string(beta_factor);
                
                // Try this perturbation (same two-stage logic as above)
                double parameters[7];
                parameters[0] = perturbed_set.params[0];
                parameters[1] = perturbed_set.params[1];
                parameters[2] = perturbed_set.params[2];
                parameters[3] = perturbed_set.params[3];
                parameters[4] = perturbed_set.params[4];
                parameters[5] = perturbed_set.params[5];
                parameters[6] = perturbed_set.params[6];
                
                ceres::Problem problem;
                
                for (size_t j = 0; j < x_vals.size(); ++j) {
                    ceres::CostFunction* cost_function = PowerLorentzian3DCostFunction::Create(
                        x_vals[j], y_vals[j], z_vals[j], uncertainty);
                    problem.AddResidualBlock(cost_function, nullptr, parameters);
                }
                
                // Apply same bounds as before
                double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, 
                                        std::max(parameters[0] * 0.01, std::abs(min_charge) * 0.1));
                double amp_max = std::max(max_charge * 1.5, std::max(parameters[0] * 100.0, 1e-10));
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
                
                double gamma_min_x = std::max(pixel_spacing * 0.05, data_spread_x * 0.01);
                double gamma_max_x = std::min(pixel_spacing * 4.0, data_spread_x * 0.8);
                double gamma_min_y = std::max(pixel_spacing * 0.05, data_spread_y * 0.01);
                double gamma_max_y = std::min(pixel_spacing * 4.0, data_spread_y * 0.8);
                
                problem.SetParameterLowerBound(parameters, 3, gamma_min_x);
                problem.SetParameterUpperBound(parameters, 3, gamma_max_x);
                problem.SetParameterLowerBound(parameters, 4, gamma_min_y);
                problem.SetParameterUpperBound(parameters, 4, gamma_max_y);
                
                double beta_min = (outlier_ratio > 0.2) ? 0.5 : 0.2;
                double beta_max = (outlier_ratio > 0.2) ? 2.5 : 4.0;
                problem.SetParameterLowerBound(parameters, 5, beta_min);
                problem.SetParameterUpperBound(parameters, 5, beta_max);
                
                double charge_range = std::abs(max_charge - min_charge);
                double baseline_range = std::max(charge_range * 0.5, 
                                               std::max(std::abs(parameters[6]) * 2.0, 1e-12));
                problem.SetParameterLowerBound(parameters, 6, parameters[6] - baseline_range);
                problem.SetParameterUpperBound(parameters, 6, parameters[6] + baseline_range);
                
                // Two-stage approach for perturbations too
                problem.SetParameterLowerBound(parameters, 5, 0.9);
                problem.SetParameterUpperBound(parameters, 5, 1.1);
                
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
                    problem.SetParameterLowerBound(parameters, 5, 0.2);
                    problem.SetParameterUpperBound(parameters, 5, 4.0);
                    
                    double stage1_center_x = parameters[1];
                    double stage1_center_y = parameters[2];
                    double tight_center_range = pixel_spacing * 1.0;
                    problem.SetParameterLowerBound(parameters, 1, stage1_center_x - tight_center_range);
                    problem.SetParameterUpperBound(parameters, 1, stage1_center_x + tight_center_range);
                    problem.SetParameterLowerBound(parameters, 2, stage1_center_y - tight_center_range);
                    problem.SetParameterUpperBound(parameters, 2, stage1_center_y + tight_center_range);
                    
                    ceres::Solve(options, &problem, &summary);
                } else {
                    problem.SetParameterLowerBound(parameters, 5, 0.2);
                    problem.SetParameterUpperBound(parameters, 5, 4.0);
                    ceres::Solve(options, &problem, &summary);
                }
                
                bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                                      summary.termination_type == ceres::USER_SUCCESS) &&
                                     parameters[0] > 0 && parameters[3] > 0 && parameters[4] > 0 && 
                                     parameters[5] > 0.1 && parameters[5] < 5.0 &&
                                     !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                                     !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                                     !std::isnan(parameters[4]) && !std::isnan(parameters[5]) &&
                                     !std::isnan(parameters[6]);
                
                if (fit_successful) {
                    double cost = summary.final_cost;
                    double chi2 = cost * 2.0;
                    int dof = std::max(1, static_cast<int>(x_vals.size()) - 7);
                    double chi2_red = chi2 / dof;
                    
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_chi2_reduced = chi2_red;
                        std::copy(parameters, parameters + 7, best_parameters);
                        best_description = perturbed_set.description;
                        
                        if (verbose) {
                            std::cout << "New best result from " << perturbed_set.description 
                                     << " with cost=" << cost << ", χ²ᵣ=" << chi2_red << std::endl;
                        }
                    }
                }
            }
        } else if (verbose && any_success) {
            std::cout << "Base 3D Power Lorentzian fit χ²ᵣ=" << best_chi2_reduced << " ≤ 0.5, skipping perturbations (hierarchical multi-start)" << std::endl;
        }
        
        if (any_success) {
            // Extract results from best attempt
            fit_amplitude = best_parameters[0];
            fit_center_x = best_parameters[1];
            fit_center_y = best_parameters[2];
            fit_gamma_x = std::abs(best_parameters[3]);
            fit_gamma_y = std::abs(best_parameters[4]);
            fit_beta = best_parameters[5];
            fit_vertical_offset = best_parameters[6];
            
            // Simple uncertainty estimation (fallback method)
            fit_amplitude_err = 0.02 * fit_amplitude;
            fit_center_x_err = 0.02 * pixel_spacing;
            fit_center_y_err = 0.02 * pixel_spacing;
            fit_gamma_x_err = 0.05 * fit_gamma_x;
            fit_gamma_y_err = 0.05 * fit_gamma_y;
            fit_beta_err = 0.1 * fit_beta;
            fit_vertical_offset_err = 0.1 * std::abs(fit_vertical_offset);
            
            chi2_reduced = best_chi2_reduced;
            
            if (verbose) {
                std::cout << "Successful 3D Power-Law Lorentzian fit with " << stage_name 
                         << ", best init: " << best_description
                         << ": A=" << fit_amplitude << ", mx=" << fit_center_x 
                         << ", my=" << fit_center_y << ", gamma_x=" << fit_gamma_x
                         << ", gamma_y=" << fit_gamma_y << ", beta=" << fit_beta
                         << ", B=" << fit_vertical_offset << ", chi2red=" << chi2_reduced << std::endl;
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
        std::cout << "Cheap 3D Power Lorentzian config " << (success ? "succeeded" : "failed") 
                 << " with χ²ᵣ=" << chi2_reduced << std::endl;
    }
    
    // Always try ALL expensive configurations regardless of cheap config result
    if (verbose) {
        std::cout << "Trying all " << expensive_configs.size() 
                 << " expensive 3D Power Lorentzian configurations..." << std::endl;
    }
    
    for (size_t i = 0; i < expensive_configs.size(); ++i) {
        bool config_success = try_config(expensive_configs[i], "expensive_" + std::to_string(i+1));
        if (config_success && (!best_success || chi2_reduced < best_chi2)) {
            best_success = config_success;
            best_chi2 = chi2_reduced;
            success = config_success;
        }
        
        if (verbose) {
            std::cout << "Expensive 3D Power Lorentzian config " << (i+1) << " " 
                     << (config_success ? "succeeded" : "failed") 
                     << " with χ²ᵣ=" << chi2_reduced << std::endl;
        }
    }
    
    if (best_success) {
        if (verbose) {
            std::cout << "Best 3D Power Lorentzian fit achieved with χ²ᵣ=" << best_chi2 << std::endl;
        }
        return true;
    }
    
    if (verbose) {
        std::cout << "All 3D Power Lorentzian fitting strategies failed" << std::endl;
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
    
    // Initialize Ceres logging (removed mutex for better parallelization)
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