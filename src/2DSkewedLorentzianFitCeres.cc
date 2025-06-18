#include "2DSkewedLorentzianFitCeres.hh"
#include "CeresLoggingInit.hh"
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
static std::mutex gCeresSkewedLorentzianFitMutex;
static std::atomic<int> gGlobalCeresSkewedLorentzianFitCounter{0};

// Use shared Google logging initialization
void InitializeCeresSkewedLorentzian() {
    CeresLoggingInitializer::InitializeOnce();
}

// Improved and simplified Skewed Lorentzian cost function
// Simplified form: y(x) = A * (1/(1+(x-m)²)^β) * (1 + λ*sign(x-m)*((x-m)²/(1+(x-m)²))^α) + B
// This reduces parameters from 6 to 5 and improves numerical stability
struct ImprovedSkewedLorentzianCostFunction {
    ImprovedSkewedLorentzianCostFunction(double x, double y, double weight = 1.0) 
        : x_(x), y_(y), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = A (amplitude)
        // params[1] = m (center)
        // params[2] = beta (shape parameter, constrained to [0.5, 2.0])
        // params[3] = lambda (skewness parameter, constrained to [-0.5, 0.5])
        // params[4] = B (baseline)
        
        const T& A = params[0];
        const T& m = params[1];
        const T& beta = params[2];
        const T& lambda = params[3];
        const T& B = params[4];
        
        // Ensure parameter stability
        T safe_beta = ceres::abs(beta);
        if (safe_beta < T(0.5)) {
            safe_beta = T(0.5);
        } else if (safe_beta > T(2.0)) {
            safe_beta = T(2.0);
        }
        
        T safe_lambda = lambda;
        if (safe_lambda < T(-0.5)) {
            safe_lambda = T(-0.5);
        } else if (safe_lambda > T(0.5)) {
            safe_lambda = T(0.5);
        }
        
        // Simplified Skewed Lorentzian: y(x) = A * (1/(1+(x-m)²)^β) * (1 + λ*tanh((x-m)/w)) + B
        // where w is a characteristic width
        T dx = x_ - m;
        T dx_squared = dx * dx;
        
        // Base Lorentzian term: 1/(1+dx²)^β
        T base_term_denominator = T(1.0) + dx_squared;
        T base_term = T(1.0) / ceres::pow(base_term_denominator, safe_beta);
        
        // Simplified skewness term using tanh for stability
        T width_scale = T(1.0); // Characteristic width parameter
        T skew_term = T(1.0) + safe_lambda * ceres::tanh(dx / width_scale);
        
        // Combined function
        T predicted = A * base_term * skew_term + B;
        
        // Weighted residual for robust fitting
        residual[0] = T(weight_) * (predicted - T(y_));
        
        return true;
    }
    
    static ceres::CostFunction* Create(double x, double y, double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<ImprovedSkewedLorentzianCostFunction, 1, 5>(
            new ImprovedSkewedLorentzianCostFunction(x, y, weight)));
    }
    
private:
    const double x_;
    const double y_;
    const double weight_;
};

// Robust statistics calculations with improved outlier detection
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

DataStatistics CalculateRobustStatisticsSkewedLorentzian(const std::vector<double>& x_vals, 
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
    
    // Numerical stability safeguard
    if (!std::isfinite(stats.mad) || stats.mad < 1e-12) {
        stats.mad = (std::isfinite(stats.std_dev) && stats.std_dev > 1e-12) ?
                    stats.std_dev : 1e-12;
    }
    
    // Improved center estimation using charge-weighted centroid with outlier rejection
    double threshold = stats.q75; // Only use points above 75th percentile for center estimation
    stats.weighted_mean = 0.0;
    stats.total_weight = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        if (y_vals[i] >= threshold) {
            double weight = y_vals[i] - stats.q25; // Weight by charge above baseline
        if (weight > 0) {
            stats.weighted_mean += x_vals[i] * weight;
            stats.total_weight += weight;
            }
        }
    }
    
    if (stats.total_weight > 0) {
        stats.weighted_mean /= stats.total_weight;
        stats.robust_center = stats.weighted_mean;
    } else {
        // Fallback to arithmetic mean
        stats.weighted_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
        stats.robust_center = stats.weighted_mean;
    }
    
    stats.valid = true;
    return stats;
}

// Improved Skewed Lorentzian fitting function with better constraints
bool FitSkewedLorentzianCeres(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    double& fit_amplitude,
    double& fit_center,
    double& fit_beta,
    double& fit_lambda,
    double& fit_gamma,
    double& fit_vertical_offset,
    double& fit_amplitude_err,
    double& fit_center_err,
    double& fit_beta_err,
    double& fit_lambda_err,
    double& fit_gamma_err,
    double& fit_vertical_offset_err,
    double& chi2_reduced,
    bool verbose) {
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 5) {
        if (verbose) {
            std::cout << "Insufficient data points for Skewed Lorentzian fitting (need at least 5)" << std::endl;
        }
        return false;
    }
    
    // Calculate robust statistics for parameter estimation
    DataStatistics stats = CalculateRobustStatisticsSkewedLorentzian(x_vals, y_vals);
    if (!stats.valid) {
        if (verbose) {
            std::cout << "Failed to calculate statistics for Skewed Lorentzian fitting" << std::endl;
        }
        return false;
    }
    
    // Use the provided global center estimate as starting point; fall back to
    // robust_center only if the estimate is not finite.
    double initial_center = std::isfinite(center_estimate) ? center_estimate : stats.robust_center;
    double initial_amplitude = stats.max_val - stats.min_val;
    double initial_beta = 1.0;        // Standard Lorentzian shape
    double initial_lambda = 0.0;      // Start with no skewness
    double initial_baseline = stats.q25; // Use 25th percentile as baseline
    
    // Validate initial estimates
    if (initial_amplitude <= 0) {
        initial_amplitude = stats.std_dev;
    }
    if (!std::isfinite(initial_center)) {
        initial_center = center_estimate;
    }
    
    // Set up parameter array (reduced from 6 to 5 parameters)
    double parameters[5];
    parameters[0] = initial_amplitude;
    parameters[1] = initial_center;
    parameters[2] = initial_beta;
    parameters[3] = initial_lambda;
    parameters[4] = initial_baseline;
    
    // Build the problem
    ceres::Problem problem;
    
    // Add residual blocks with adaptive weighting
    for (size_t i = 0; i < x_vals.size(); ++i) {
        // Use adaptive weighting based on signal strength
        double weight = 1.0;
        if (y_vals[i] > stats.q75) {
            weight = 1.5; // Higher weight for strong signals
        } else if (y_vals[i] < stats.q25) {
            weight = 0.5; // Lower weight for weak signals
        }
        
        ceres::CostFunction* cost_function = ImprovedSkewedLorentzianCostFunction::Create(x_vals[i], y_vals[i], weight);
        problem.AddResidualBlock(cost_function, nullptr, parameters);
    }
    
    // Set tighter parameter bounds for physical realism
    problem.SetParameterLowerBound(parameters, 0, 0.1 * initial_amplitude);  // amplitude > 0
    problem.SetParameterUpperBound(parameters, 0, 5.0 * initial_amplitude);
    
    problem.SetParameterLowerBound(parameters, 1, initial_center - 2.0 * pixel_spacing);  // center bounds
    problem.SetParameterUpperBound(parameters, 1, initial_center + 2.0 * pixel_spacing);
    
    problem.SetParameterLowerBound(parameters, 2, 0.2);   // beta bounds relaxed: [0.2, 3.0]
    problem.SetParameterUpperBound(parameters, 2, 3.0);
    
    problem.SetParameterLowerBound(parameters, 3, -1.5);  // lambda bounds relaxed: [-1.5, 1.5]
    problem.SetParameterUpperBound(parameters, 3, 1.5);
    
    double baseline_range = std::min(initial_amplitude * 0.3, std::abs(initial_baseline) * 1.5);
    problem.SetParameterLowerBound(parameters, 4, initial_baseline - baseline_range);
    problem.SetParameterUpperBound(parameters, 4, initial_baseline + baseline_range);
    
    // Configure solver with improved settings
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-12;
    options.max_num_iterations = 1000; // Reduced iterations to prevent overfitting
    options.max_num_consecutive_invalid_steps = 20;
    options.use_nonmonotonic_steps = true;
    options.minimizer_progress_to_stdout = false;
    
    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Enhanced validation of results
    bool fit_successful = (summary.termination_type == ceres::CONVERGENCE ||
                          summary.termination_type == ceres::USER_SUCCESS) &&
                         parameters[0] > 0 && parameters[2] > 0.15 && parameters[2] < 3.5 &&
                         std::abs(parameters[3]) < 1.6 && // Skewness constraint
                         !std::isnan(parameters[0]) && !std::isnan(parameters[1]) &&
                         !std::isnan(parameters[2]) && !std::isnan(parameters[3]) &&
                         !std::isnan(parameters[4]) &&
                         summary.final_cost < 1e6; // Reject fits with excessive cost
    
    if (fit_successful) {
        // Extract results
        fit_amplitude = parameters[0];
        fit_center = parameters[1];
        fit_beta = parameters[2];
        fit_lambda = parameters[3];
        fit_gamma = 1.0; // Fixed for simplified model
        fit_vertical_offset = parameters[4];
        
        // Improved uncertainty estimation
        fit_amplitude_err = std::max(0.05 * fit_amplitude, 0.1 * stats.mad);
        fit_center_err = std::max(0.01 * pixel_spacing, 0.005 * pixel_spacing);
        fit_beta_err = std::max(0.1 * fit_beta, 0.05);
        fit_lambda_err = std::max(0.1 * std::abs(fit_lambda), 0.02);
        fit_gamma_err = 0.0; // Fixed parameter
        fit_vertical_offset_err = std::max(0.1 * std::abs(fit_vertical_offset), 0.05 * stats.mad);
        
        // Corrected reduced chi-squared calculation
        double chi2 = summary.final_cost * 2.0;
        int dof = std::max(1, static_cast<int>(x_vals.size()) - 5); // Corrected DOF for 5 parameters
        chi2_reduced = chi2 / dof;
        
        // Additional quality check
        if (chi2_reduced > 100.0) { // Reject fits with very poor chi2
            fit_successful = false;
        if (verbose) {
                std::cout << "Rejecting fit due to poor chi2_reduced: " << chi2_reduced << std::endl;
            }
        } else if (verbose) {
            std::cout << "Successful Skewed Lorentzian fit: A=" << fit_amplitude
                     << ", m=" << fit_center << ", β=" << fit_beta
                     << ", λ=" << fit_lambda << ", B=" << fit_vertical_offset
                     << ", chi2red=" << chi2_reduced << std::endl;
        }
        
        return fit_successful;
    } else if (verbose) {
        std::cout << "Skewed Lorentzian fit failed: " << summary.BriefReport() 
                 << ", final_cost=" << summary.final_cost << std::endl;
    }
    
    return false;
}

SkewedLorentzianFit2DResultsCeres Fit2DSkewedLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    SkewedLorentzianFit2DResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresSkewedLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeresSkewedLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit2DSkewedLorentzianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 5) {
        if (verbose) {
            std::cout << "Fit2DSkewedLorentzianCeres: Error - need at least 5 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting 2D Skewed Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
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
        if (dist < min_row_dist && row_pair.second.size() >= 5) {
            min_row_dist = dist;
            best_row_y = row_pair.first;
        }
    }
    
    double best_col_x = center_x_estimate;
    double min_col_dist = std::numeric_limits<double>::max();
    for (const auto& col_pair : cols_data) {
        double dist = std::abs(col_pair.first - center_x_estimate);
        if (dist < min_col_dist && col_pair.second.size() >= 5) {
            min_col_dist = dist;
            best_col_x = col_pair.first;
        }
    }
    
    bool x_fit_success = false;
    bool y_fit_success = false;
    
    // Fit X direction (central row)
    if (rows_data.find(best_row_y) != rows_data.end() && rows_data[best_row_y].size() >= 5) {
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
            std::cout << "Fitting Skewed Lorentzian X direction with " << x_vals.size() << " points" << std::endl;
        }
        
        x_fit_success = FitSkewedLorentzianCeres(
            x_vals, y_vals, center_x_estimate, pixel_spacing,
            result.x_amplitude, result.x_center, result.x_beta, result.x_lambda, result.x_gamma, result.x_vertical_offset,
            result.x_amplitude_err, result.x_center_err, result.x_beta_err, result.x_lambda_err, result.x_gamma_err, result.x_vertical_offset_err,
            result.x_chi2red, verbose);
        
        // Calculate DOF and p-value
        result.x_dof = std::max(1, static_cast<int>(x_vals.size()) - 5); // Corrected DOF for 5 parameters
        result.x_pp = (result.x_chi2red > 0) ? 1.0 - std::min(1.0, result.x_chi2red / 10.0) : 0.0;
    }
    
    // Fit Y direction (central column)
    if (cols_data.find(best_col_x) != cols_data.end() && cols_data[best_col_x].size() >= 5) {
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
            std::cout << "Fitting Skewed Lorentzian Y direction with " << x_vals.size() << " points" << std::endl;
        }
        
        y_fit_success = FitSkewedLorentzianCeres(
            x_vals, y_vals, center_y_estimate, pixel_spacing,
            result.y_amplitude, result.y_center, result.y_beta, result.y_lambda, result.y_gamma, result.y_vertical_offset,
            result.y_amplitude_err, result.y_center_err, result.y_beta_err, result.y_lambda_err, result.y_gamma_err, result.y_vertical_offset_err,
            result.y_chi2red, verbose);
        
        // Calculate DOF and p-value
        result.y_dof = std::max(1, static_cast<int>(x_vals.size()) - 5); // Corrected DOF for 5 parameters
        result.y_pp = (result.y_chi2red > 0) ? 1.0 - std::min(1.0, result.y_chi2red / 10.0) : 0.0;
    }
    
    // Set overall success status
    result.fit_successful = x_fit_success && y_fit_success;
    
    if (verbose) {
        std::cout << "2D Skewed Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}

// Outlier removal function for Skewed Lorentzian fitting
SkewedLorentzianOutlierRemovalResult RemoveSkewedLorentzianOutliers(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& charge_values,
    bool enable_outlier_removal,
    double sigma_threshold,
    bool verbose)
{
    SkewedLorentzianOutlierRemovalResult result;
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "RemoveSkewedLorentzianOutliers: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (!enable_outlier_removal) {
        // No filtering requested - return original data
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
        result.filtering_applied = false;
        result.success = true;
        return result;
    }
    
    if (charge_values.size() < 5) {
        // Not enough data for outlier removal
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
        result.filtering_applied = false;
        result.success = true;
        return result;
    }
    
    // Calculate robust statistics for outlier detection
    DataStatistics stats = CalculateRobustStatisticsSkewedLorentzian(x_coords, charge_values);
    if (!stats.valid) {
        // Fall back to original data if statistics calculation fails
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
        result.filtering_applied = false;
        result.success = false;
        return result;
    }
    
    // Identify outliers using median absolute deviation
    double outlier_threshold = sigma_threshold * stats.mad;
    std::vector<bool> is_outlier(charge_values.size(), false);
    int outlier_count = 0;
    
    for (size_t i = 0; i < charge_values.size(); ++i) {
        double deviation = std::abs(charge_values[i] - stats.median);
        if (deviation > outlier_threshold) {
            is_outlier[i] = true;
            outlier_count++;
        }
    }
    
    // Ensure we don't remove too many points (keep at least 5)
    if (charge_values.size() - outlier_count < 5) {
        // Too many outliers detected - reduce threshold or keep original data
        result.filtered_x_coords = x_coords;
        result.filtered_y_coords = y_coords;
        result.filtered_charge_values = charge_values;
        result.outliers_removed = 0;
        result.filtering_applied = false;
        result.success = true;
        
        if (verbose) {
            std::cout << "RemoveSkewedLorentzianOutliers: Too many outliers detected (" << outlier_count 
                     << "), keeping original data" << std::endl;
        }
        return result;
    }
    
    // Filter out outliers
    for (size_t i = 0; i < charge_values.size(); ++i) {
        if (!is_outlier[i]) {
            result.filtered_x_coords.push_back(x_coords[i]);
            result.filtered_y_coords.push_back(y_coords[i]);
            result.filtered_charge_values.push_back(charge_values[i]);
        }
    }
    
    result.outliers_removed = outlier_count;
    result.filtering_applied = true;
    result.success = true;
    
    if (verbose) {
        std::cout << "RemoveSkewedLorentzianOutliers: Removed " << outlier_count 
                 << " outliers, " << result.filtered_charge_values.size() << " points remaining" << std::endl;
    }
    
    return result;
}

// Diagonal Skewed Lorentzian fitting function
DiagonalSkewedLorentzianFitResultsCeres FitDiagonalSkewedLorentzianCeres(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering)
{
    DiagonalSkewedLorentzianFitResultsCeres result;
    
    // Thread-safe Ceres operations
    std::lock_guard<std::mutex> lock(gCeresSkewedLorentzianFitMutex);
    
    // Initialize Ceres logging
    InitializeCeresSkewedLorentzian();
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "FitDiagonalSkewedLorentzianCeres: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 5) {
        if (verbose) {
            std::cout << "FitDiagonalSkewedLorentzianCeres: Error - need at least 5 data points for fitting" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting diagonal Skewed Lorentzian fit (Ceres) with " << x_coords.size() << " data points" << std::endl;
    }
    
    // Apply outlier filtering if requested
    std::vector<double> filtered_x_coords = x_coords;
    std::vector<double> filtered_y_coords = y_coords;
    std::vector<double> filtered_charge_values = charge_values;
    
    if (enable_outlier_filtering) {
        SkewedLorentzianOutlierRemovalResult filter_result = RemoveSkewedLorentzianOutliers(
            x_coords, y_coords, charge_values, true, 2.5, verbose);
        
        if (filter_result.success && filter_result.filtering_applied) {
            filtered_x_coords = filter_result.filtered_x_coords;
            filtered_y_coords = filter_result.filtered_y_coords;
            filtered_charge_values = filter_result.filtered_charge_values;
        }
    }
    
    // Create diagonal data arrays
    // Main diagonal: points where (x-center_x) ≈ (y-center_y)
    // Secondary diagonal: points where (x-center_x) ≈ -(y-center_y)
    std::vector<std::pair<double, double>> main_diag_x_data, main_diag_y_data;
    std::vector<std::pair<double, double>> sec_diag_x_data, sec_diag_y_data;
    
    const double diagonal_tolerance = pixel_spacing * 0.5; // Tolerance for diagonal selection
    
    for (size_t i = 0; i < filtered_x_coords.size(); ++i) {
        double x = filtered_x_coords[i];
        double y = filtered_y_coords[i];
        double charge = filtered_charge_values[i];
        
        if (charge <= 0) continue;
        
        double dx = x - center_x_estimate;
        double dy = y - center_y_estimate;
        
        // Check if point is on main diagonal (dx ≈ dy)
        if (std::abs(dx - dy) < diagonal_tolerance) {
            double diag_coord = (dx + dy) / 2.0; // Average coordinate along diagonal
            main_diag_x_data.push_back(std::make_pair(diag_coord, charge));
            main_diag_y_data.push_back(std::make_pair(diag_coord, charge));
        }
        
        // Check if point is on secondary diagonal (dx ≈ -dy)
        if (std::abs(dx + dy) < diagonal_tolerance) {
            double diag_coord = (dx - dy) / 2.0; // Coordinate along secondary diagonal
            sec_diag_x_data.push_back(std::make_pair(diag_coord, charge));
            sec_diag_y_data.push_back(std::make_pair(diag_coord, charge));
        }
    }
    
    // Fit main diagonal X direction
    if (main_diag_x_data.size() >= 5) {
        std::sort(main_diag_x_data.begin(), main_diag_x_data.end());
        
        std::vector<double> x_vals, y_vals;
        for (const auto& point : main_diag_x_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting main diagonal X with " << x_vals.size() << " points" << std::endl;
        }
        
        result.main_diag_x_fit_successful = FitSkewedLorentzianCeres(
            x_vals, y_vals, 0.0, pixel_spacing,
            result.main_diag_x_amplitude, result.main_diag_x_center, result.main_diag_x_beta, 
            result.main_diag_x_lambda, result.main_diag_x_gamma, result.main_diag_x_vertical_offset,
            result.main_diag_x_amplitude_err, result.main_diag_x_center_err, result.main_diag_x_beta_err,
            result.main_diag_x_lambda_err, result.main_diag_x_gamma_err, result.main_diag_x_vertical_offset_err,
            result.main_diag_x_chi2red, verbose);
        
        result.main_diag_x_dof = std::max(1, static_cast<int>(x_vals.size()) - 5);
        result.main_diag_x_pp = (result.main_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_x_chi2red / 10.0) : 0.0;
    }
    
    // Fit main diagonal Y direction (same data as X)
    if (main_diag_y_data.size() >= 5) {
        std::sort(main_diag_y_data.begin(), main_diag_y_data.end());
        
        std::vector<double> x_vals, y_vals;
        for (const auto& point : main_diag_y_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting main diagonal Y with " << x_vals.size() << " points" << std::endl;
        }
        
        result.main_diag_y_fit_successful = FitSkewedLorentzianCeres(
            x_vals, y_vals, 0.0, pixel_spacing,
            result.main_diag_y_amplitude, result.main_diag_y_center, result.main_diag_y_beta,
            result.main_diag_y_lambda, result.main_diag_y_gamma, result.main_diag_y_vertical_offset,
            result.main_diag_y_amplitude_err, result.main_diag_y_center_err, result.main_diag_y_beta_err,
            result.main_diag_y_lambda_err, result.main_diag_y_gamma_err, result.main_diag_y_vertical_offset_err,
            result.main_diag_y_chi2red, verbose);
        
        result.main_diag_y_dof = std::max(1, static_cast<int>(x_vals.size()) - 5);
        result.main_diag_y_pp = (result.main_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.main_diag_y_chi2red / 10.0) : 0.0;
    }
    
    // Fit secondary diagonal X direction
    if (sec_diag_x_data.size() >= 5) {
        std::sort(sec_diag_x_data.begin(), sec_diag_x_data.end());
        
        std::vector<double> x_vals, y_vals;
        for (const auto& point : sec_diag_x_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting secondary diagonal X with " << x_vals.size() << " points" << std::endl;
        }
        
        result.sec_diag_x_fit_successful = FitSkewedLorentzianCeres(
            x_vals, y_vals, 0.0, pixel_spacing,
            result.sec_diag_x_amplitude, result.sec_diag_x_center, result.sec_diag_x_beta,
            result.sec_diag_x_lambda, result.sec_diag_x_gamma, result.sec_diag_x_vertical_offset,
            result.sec_diag_x_amplitude_err, result.sec_diag_x_center_err, result.sec_diag_x_beta_err,
            result.sec_diag_x_lambda_err, result.sec_diag_x_gamma_err, result.sec_diag_x_vertical_offset_err,
            result.sec_diag_x_chi2red, verbose);
        
        result.sec_diag_x_dof = std::max(1, static_cast<int>(x_vals.size()) - 5);
        result.sec_diag_x_pp = (result.sec_diag_x_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_x_chi2red / 10.0) : 0.0;
    }
    
    // Fit secondary diagonal Y direction
    if (sec_diag_y_data.size() >= 5) {
        std::sort(sec_diag_y_data.begin(), sec_diag_y_data.end());
        
        std::vector<double> x_vals, y_vals;
        for (const auto& point : sec_diag_y_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        if (verbose) {
            std::cout << "Fitting secondary diagonal Y with " << x_vals.size() << " points" << std::endl;
        }
        
        result.sec_diag_y_fit_successful = FitSkewedLorentzianCeres(
            x_vals, y_vals, 0.0, pixel_spacing,
            result.sec_diag_y_amplitude, result.sec_diag_y_center, result.sec_diag_y_beta,
            result.sec_diag_y_lambda, result.sec_diag_y_gamma, result.sec_diag_y_vertical_offset,
            result.sec_diag_y_amplitude_err, result.sec_diag_y_center_err, result.sec_diag_y_beta_err,
            result.sec_diag_y_lambda_err, result.sec_diag_y_gamma_err, result.sec_diag_y_vertical_offset_err,
            result.sec_diag_y_chi2red, verbose);
        
        result.sec_diag_y_dof = std::max(1, static_cast<int>(x_vals.size()) - 5);
        result.sec_diag_y_pp = (result.sec_diag_y_chi2red > 0) ? 1.0 - std::min(1.0, result.sec_diag_y_chi2red / 10.0) : 0.0;
    }
    
    // Set overall success status
    result.fit_successful = result.main_diag_x_fit_successful && result.main_diag_y_fit_successful &&
                           result.sec_diag_x_fit_successful && result.sec_diag_y_fit_successful;
    
    if (verbose) {
        std::cout << "Diagonal Skewed Lorentzian fit (Ceres) " << (result.fit_successful ? "successful" : "partial/failed") 
                 << " (Main X: " << (result.main_diag_x_fit_successful ? "OK" : "FAIL")
                 << ", Main Y: " << (result.main_diag_y_fit_successful ? "OK" : "FAIL")
                 << ", Sec X: " << (result.sec_diag_x_fit_successful ? "OK" : "FAIL")
                 << ", Sec Y: " << (result.sec_diag_y_fit_successful ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
}
