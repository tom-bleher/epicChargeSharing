#include "CeresUtils.hh"
#include "Constants.hh"
#include <algorithm>

ceres::Solver::Options MakeSolverOptions(SolverPreset preset, double pixel_spacing) {
    ceres::Solver::Options options;
    
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = false;
    
    switch (preset) {
        case SolverPreset::FAST:
            options.linear_solver_type = ceres::DENSE_QR;
            options.function_tolerance = 1e-12;
            options.gradient_tolerance = 1e-12;
            options.max_num_iterations = 400;
            options.initial_trust_region_radius = 0.1 * pixel_spacing;
            break;
            
        case SolverPreset::FALLBACK:
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
            options.function_tolerance = 1e-14;
            options.gradient_tolerance = 1e-14;
            options.max_num_iterations = 1200;
            options.initial_trust_region_radius = 0.05 * pixel_spacing;
            break;
    }
    
    options.max_trust_region_radius = 2.0 * pixel_spacing;
    options.min_trust_region_radius = 1e-4 * pixel_spacing;
    
    return options;
}

void SetStandardBounds(ceres::Problem& problem, double* parameters, 
                      double charge_range, double pixel_spacing,
                      double center_x, double center_y, bool is_3d, bool has_beta) {
    // Amplitude bounds: [0.1×A, 3×A] (tightened from very loose bounds)
    double amp_min = std::max(Constants::MIN_UNCERTAINTY_VALUE, parameters[0] * 0.1);
    double amp_max = std::max(parameters[0] * 3.0, charge_range * 2.0);
    problem.SetParameterLowerBound(parameters, 0, amp_min);
    problem.SetParameterUpperBound(parameters, 0, amp_max);
    
    // Center bounds: [est±2×pitch] (tightened from [est±4×pitch])
    double center_range = pixel_spacing * 2.0;
    problem.SetParameterLowerBound(parameters, 1, center_x - center_range);
    problem.SetParameterUpperBound(parameters, 1, center_x + center_range);
    
    if (is_3d) {
        // Y center bounds for 3D models
        problem.SetParameterLowerBound(parameters, 2, center_y - center_range);
        problem.SetParameterUpperBound(parameters, 2, center_y + center_range);
        
        // Width bounds for 3D: γₓ and γᵧ
        double width_min = pixel_spacing * 0.2;
        double width_max = pixel_spacing * 2.0;
        problem.SetParameterLowerBound(parameters, 3, width_min); // gamma_x
        problem.SetParameterUpperBound(parameters, 3, width_max);
        problem.SetParameterLowerBound(parameters, 4, width_min); // gamma_y  
        problem.SetParameterUpperBound(parameters, 4, width_max);
        
        if (has_beta) {
            // β parameter bounds for 3D Power-Law: [0.3, 3.0]
            problem.SetParameterLowerBound(parameters, 5, 0.3);
            problem.SetParameterUpperBound(parameters, 5, 3.0);
            
            // Baseline bounds for 7-parameter model
            double baseline_range = charge_range * 0.5;
            problem.SetParameterLowerBound(parameters, 6, parameters[6] - baseline_range);
            problem.SetParameterUpperBound(parameters, 6, parameters[6] + baseline_range);
        } else {
            // Baseline bounds for 6-parameter model
            double baseline_range = charge_range * 0.5;
            problem.SetParameterLowerBound(parameters, 5, parameters[5] - baseline_range);
            problem.SetParameterUpperBound(parameters, 5, parameters[5] + baseline_range);
        }
    } else {
        // Width bounds for 2D: [0.2×pitch, 2×pitch] (tightened)
        double width_min = pixel_spacing * 0.2;
        double width_max = pixel_spacing * 2.0;
        problem.SetParameterLowerBound(parameters, 2, width_min);
        problem.SetParameterUpperBound(parameters, 2, width_max);
        
        if (has_beta) {
            // β parameter bounds for 2D Power-Law: [0.3, 3.0]
            problem.SetParameterLowerBound(parameters, 3, 0.3);
            problem.SetParameterUpperBound(parameters, 3, 3.0);
            
            // Baseline bounds for 5-parameter model
            double baseline_range = charge_range * 0.5;
            problem.SetParameterLowerBound(parameters, 4, parameters[4] - baseline_range);
            problem.SetParameterUpperBound(parameters, 4, parameters[4] + baseline_range);
        } else {
            // Baseline bounds for 4-parameter model
            double baseline_range = charge_range * 0.5;
            problem.SetParameterLowerBound(parameters, 3, parameters[3] - baseline_range);
            problem.SetParameterUpperBound(parameters, 3, parameters[3] + baseline_range);
        }
    }
} 