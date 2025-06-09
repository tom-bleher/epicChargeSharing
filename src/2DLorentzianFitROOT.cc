#include "2DLorentzianFitROOT.hh"
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

// ROOT includes
#include "TF1.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "TVirtualFitter.h"
#include "TMinuit.h"
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "HFitInterface.h"
#include "TRandom3.h"
#include "TApplication.h"
#include "TROOT.h"

// Thread-safe mutex for ROOT operations
static std::mutex gROOTLorentzianFitMutex;
static std::atomic<int> gGlobalROOTLorentzianFitCounter{0};
static std::atomic<bool> gROOTLorentzianInitialized{false};

// Initialize ROOT if needed (thread-safe)
void InitializeROOTLorentzian() {
    std::lock_guard<std::mutex> lock(gROOTLorentzianFitMutex);
    
    if (!gROOTLorentzianInitialized.load()) {
        // Ensure ROOT is properly initialized
        if (!gROOT->GetApplication()) {
            // Create a minimal application if none exists
            int argc = 1;
            char name[] = "epicToyLorentzFit";
            char* argv[] = {name};
            new TApplication("epicToyLorentzFitApp", &argc, argv);
        }
        
        // Set batch mode to avoid GUI issues
        gROOT->SetBatch(kTRUE);
        
        // Configure ROOT for thread safety
        ROOT::EnableThreadSafety();
        
        // Set default minimizer options
        ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
        ROOT::Math::MinimizerOptions::SetDefaultStrategy(1);
        ROOT::Math::MinimizerOptions::SetDefaultTolerance(1e-6);
        ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(10000);
        
        gROOTLorentzianInitialized.store(true);
    }
}

// Robust statistics calculations for Lorentzian parameter estimation
struct LorentzianDataStatistics {
    double mean, median, std_dev, mad;
    double q25, q75, min_val, max_val;
    double weighted_mean, total_weight;
    bool valid;
    
    LorentzianDataStatistics() : mean(0), median(0), std_dev(0), mad(0), 
                               q25(0), q75(0), min_val(0), max_val(0),
                               weighted_mean(0), total_weight(0), valid(false) {}
};

LorentzianDataStatistics CalculateLorentzianRobustStatisticsROOT(const std::vector<double>& x_vals, 
                                                                     const std::vector<double>& y_vals) {
    LorentzianDataStatistics stats;
    
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
    } else {
        stats.weighted_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
    }
    
    stats.valid = true;
    return stats;
}

// Parameter estimation with multiple strategies for Lorentzian distributions
struct LorentzianParameterEstimates {
    double amplitude, center, gamma, baseline;
    double amplitude_err, center_err, gamma_err, baseline_err;
    bool valid;
    int method_used;
    
    LorentzianParameterEstimates() : amplitude(0), center(0), gamma(0), baseline(0),
                                   amplitude_err(0), center_err(0), gamma_err(0), baseline_err(0),
                                   valid(false), method_used(0) {}
};

LorentzianParameterEstimates EstimateLorentzianParametersROOT(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double center_estimate,
    double pixel_spacing,
    bool verbose = false) {
    
    LorentzianParameterEstimates estimates;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return estimates;
    }
    
    LorentzianDataStatistics stats = CalculateLorentzianRobustStatisticsROOT(x_vals, y_vals);
    if (!stats.valid) {
        return estimates;
    }
    
    if (verbose) {
        std::cout << "Lorentzian data statistics: min=" << stats.min_val << ", max=" << stats.max_val 
                 << ", median=" << stats.median << ", weighted_mean=" << stats.weighted_mean << std::endl;
    }
    
    // Handle extremely small charge values properly
    double charge_scale = std::max(stats.max_val, 1e-18);
    
    // Method: Robust estimation for small charge distributions
    estimates.center = stats.weighted_mean;
    
    // Conservative baseline estimation - for very small charges, baseline should be minimal
    estimates.baseline = std::min({stats.min_val, stats.q25 * 0.3, charge_scale * 0.001});
    estimates.baseline = std::max(estimates.baseline, 0.0); // Ensure non-negative
    
    // Amplitude should be the peak above baseline
    estimates.amplitude = stats.max_val - estimates.baseline;
    estimates.amplitude = std::max(estimates.amplitude, charge_scale * 0.1); // Minimum amplitude
    
    // Better gamma estimation for Lorentzian distributions
    // Lorentzians have wider tails, so start with larger gamma
    estimates.gamma = pixel_spacing * 0.4; // Conservative starting point, wider than Gaussian
    
    // Refine using weighted second moment, adapted for Lorentzian
    double weighted_second_moment = 0.0;
    double total_weight = 0.0;
    double amplitude_threshold = estimates.baseline + estimates.amplitude * 0.05; // 5% above baseline for Lorentzian
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        if (y_vals[i] > amplitude_threshold) {
            double weight = y_vals[i] - estimates.baseline;
            double dx = x_vals[i] - estimates.center;
            weighted_second_moment += weight * dx * dx;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0) {
        double moment_based_width = std::sqrt(weighted_second_moment / total_weight);
        // For Lorentzian, gamma is approximately related to the RMS width
        double moment_gamma = moment_based_width * 0.8; // Lorentzian is broader than Gaussian
        
        // Constrain gamma to reasonable physical range
        moment_gamma = std::max(moment_gamma, pixel_spacing * 0.05); // Minimum: 5% of pixel spacing
        moment_gamma = std::min(moment_gamma, pixel_spacing * 4.0);  // Maximum: 4x pixel spacing (wider than Gaussian)
        
        // Use average of initial estimate and moment-based estimate
        estimates.gamma = (estimates.gamma + moment_gamma) / 2.0;
    }
    
    // Final constraints on gamma
    estimates.gamma = std::max(estimates.gamma, pixel_spacing * 0.05);
    estimates.gamma = std::min(estimates.gamma, pixel_spacing * 5.0); // Wider range for Lorentzian
    
    // Error estimates proportional to signal magnitude (higher for Lorentzian due to wider tails)
    estimates.amplitude_err = estimates.amplitude * 0.06; // 6% relative error (slightly higher than Gaussian)
    estimates.center_err = pixel_spacing * 0.025; // 2.5% of pixel spacing
    estimates.gamma_err = estimates.gamma * 0.15; // 15% relative error for width
    estimates.baseline_err = charge_scale * 0.001; // Small absolute error for baseline
    
    estimates.valid = true;
    estimates.method_used = 1;
    
    if (verbose) {
        std::cout << "Lorentzian parameter estimates: A=" << estimates.amplitude 
                 << ", m=" << estimates.center << ", gamma=" << estimates.gamma 
                 << ", B=" << estimates.baseline << std::endl;
        std::cout << "Charge scale: " << charge_scale << ", Pixel spacing: " << pixel_spacing << std::endl;
    }
    
    return estimates;
}

// Enhanced outlier filtering using MAD for Lorentzian distributions
std::vector<bool> FilterLorentzianOutliers(const std::vector<double>& x_vals,
                                          const std::vector<double>& y_vals,
                                          double threshold = 2.8) { // Slightly more lenient for Lorentzian
    std::vector<bool> keep(y_vals.size(), true);
    
    if (y_vals.size() < 5) return keep; // Don't filter small datasets
    
    LorentzianDataStatistics stats = CalculateLorentzianRobustStatisticsROOT(x_vals, y_vals);
    if (!stats.valid || stats.mad <= 0) return keep;
    
    for (size_t i = 0; i < y_vals.size(); ++i) {
        double deviation = std::abs(y_vals[i] - stats.median) / stats.mad;
        if (deviation > threshold) {
            keep[i] = false;
        }
    }
    
    return keep;
}

// Core Lorentzian fitting function using ROOT with multiple strategies
struct LorentzianFitResult {
    double amplitude, center, gamma, baseline;
    double amplitude_err, center_err, gamma_err, baseline_err;
    double chi2, reduced_chi2, prob;
    int ndf, fit_status;
    double edm;
    double fwhm;
    bool successful;
    
    LorentzianFitResult() : amplitude(0), center(0), gamma(0), baseline(0),
                           amplitude_err(0), center_err(0), gamma_err(0), baseline_err(0),
                           chi2(0), reduced_chi2(0), prob(0), ndf(0), fit_status(-1),
                           edm(0), fwhm(0), successful(false) {}
};

LorentzianFitResult FitLorentzianROOT(const std::vector<double>& x_vals,
                                    const std::vector<double>& y_vals,
                                    const LorentzianParameterEstimates& initial_params,
                                    const std::string& minimizer = "Minuit2",
                                    int max_iterations = 5000,
                                    double tolerance = 1e-6,
                                    bool use_weighted_fit = true,
                                    bool verbose = false) {
    
    LorentzianFitResult result;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return result;
    }
    
    std::lock_guard<std::mutex> lock(gROOTLorentzianFitMutex);
    
    try {
        // Create unique function name to avoid conflicts
        int fit_id = ++gGlobalROOTLorentzianFitCounter;
        std::string func_name = "lorentz_fit_" + std::to_string(fit_id) + "_" + 
                               std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        
        // Define Lorentzian function with improved syntax
        double x_min = *std::min_element(x_vals.begin(), x_vals.end());
        double x_max = *std::max_element(x_vals.begin(), x_vals.end());
        double range = x_max - x_min;
        x_min -= 0.1 * range;
        x_max += 0.1 * range;
        
        // Use simpler, more robust TFormula syntax: y = A / (1 + ((x-m)/gamma)^2) + B
        std::string formula = "[0]/(1+pow((x-[1])/[2],2)) + [3]";
        TF1* func = new TF1(func_name.c_str(), formula.c_str(), x_min, x_max);
        
        // Set parameter names
        func->SetParName(0, "Amplitude");
        func->SetParName(1, "Center");
        func->SetParName(2, "Gamma");
        func->SetParName(3, "Baseline");
        
        // Set initial parameters
        func->SetParameter(0, initial_params.amplitude);
        func->SetParameter(1, initial_params.center);
        func->SetParameter(2, initial_params.gamma);
        func->SetParameter(3, initial_params.baseline);
        
        // Set parameter limits appropriate for extremely small charge values (Lorentzian)
        double data_magnitude = std::max(initial_params.amplitude, 1e-18);
        double pixel_scale = 0.5; // Default pixel spacing
        
        // Amplitude: allow wide range but keep physical
        func->SetParLimits(0, data_magnitude * 0.001, data_magnitude * 2000); // Wider range for Lorentzian
        
        // Center: constrain to reasonable range around data
        func->SetParLimits(1, x_min - pixel_scale, x_max + pixel_scale);
        
        // Gamma: physical constraints based on pixel spacing, wider than Gaussian
        func->SetParLimits(2, pixel_scale * 0.05, pixel_scale * 5.0);
        
        // Baseline: should be small and non-negative for charge measurements
        func->SetParLimits(3, 0.0, data_magnitude * 0.05);
        
        // Create TGraphErrors for fitting
        std::string graph_name = "graph_lorentz_" + std::to_string(fit_id) + "_" + 
                                std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        TGraphErrors* graph = nullptr;
        
        if (use_weighted_fit) {
            // Use improved error estimation for very small charge measurements
            std::vector<double> errors(y_vals.size());
            double data_scale = std::max(*std::max_element(y_vals.begin(), y_vals.end()), 1e-18);
            double relative_error = 0.06; // Slightly higher for Lorentzian due to wider tails
            
            for (size_t i = 0; i < y_vals.size(); ++i) {
                // Use relative error for small signals, with minimum based on data scale
                double min_error = data_scale * 0.015; // 1.5% of maximum as minimum error
                double poisson_like_error = std::sqrt(std::abs(y_vals[i]) * data_scale) / std::sqrt(data_scale);
                double relative_based_error = std::abs(y_vals[i]) * relative_error;
                
                errors[i] = std::max({min_error, poisson_like_error, relative_based_error});
            }
            graph = new TGraphErrors(x_vals.size(), &x_vals[0], &y_vals[0], 
                                   nullptr, &errors[0]);
        } else {
            graph = new TGraphErrors(x_vals.size(), &x_vals[0], &y_vals[0]);
        }
        graph->SetName(graph_name.c_str());
        
        // Configure minimizer with settings appropriate for small values
        ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizer.c_str());
        ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(max_iterations);
        ROOT::Math::MinimizerOptions::SetDefaultTolerance(tolerance * 1e-3); // Tighter tolerance for small values
        ROOT::Math::MinimizerOptions::SetDefaultStrategy(2); // More careful strategy for precision
        ROOT::Math::MinimizerOptions::SetDefaultErrorDef(1); // For chi-square
        ROOT::Math::MinimizerOptions::SetDefaultPrecision(1e-15); // High precision for small numbers
        
        // Perform fit with improved options
        std::string fit_options = "QSN0"; // Quiet, use Chi-square, No graphics, No print
        if (use_weighted_fit) fit_options += "W"; // Use weights
        
        TFitResultPtr fit_result = graph->Fit(func, fit_options.c_str());
        
        if (fit_result.Get() && fit_result->IsValid() && 
            fit_result->Status() == 0 && func->GetParameter(0) > 0 && 
            func->GetParameter(2) > 0) {
            
            // Extract results
            result.amplitude = func->GetParameter(0);
            result.center = func->GetParameter(1);
            result.gamma = func->GetParameter(2);
            result.baseline = func->GetParameter(3);
            
            result.amplitude_err = func->GetParError(0);
            result.center_err = func->GetParError(1);
            result.gamma_err = func->GetParError(2);
            result.baseline_err = func->GetParError(3);
            
            result.chi2 = func->GetChisquare();
            result.ndf = func->GetNDF();
            result.reduced_chi2 = (result.ndf > 0) ? result.chi2 / result.ndf : 0;
            result.prob = func->GetProb();
            result.fit_status = fit_result->Status();
            result.edm = fit_result->Edm();
            
            // Calculate FWHM: FWHM = 2*gamma for Lorentzian
            result.fwhm = 2.0 * std::abs(result.gamma);
            
            // Validate fit quality with criteria appropriate for small values
            double data_magnitude = std::max(result.amplitude, 1e-18);
            result.successful = (result.fit_status == 0) && 
                              (result.reduced_chi2 > 0) &&
                              (result.reduced_chi2 < 100) && // More lenient chi2 for small values
                              (result.gamma > 0) &&
                              (result.amplitude > 0) &&
                              (result.amplitude_err < 20 * result.amplitude) && // More lenient for small signals
                              (result.center_err < range * 0.5) && // Position should be reasonably precise
                              (result.gamma_err < 20 * result.gamma) && // Width can be less precise
                              (result.amplitude > data_magnitude * 0.01); // Amplitude should be significant
            
            if (verbose) {
                std::cout << "ROOT Lorentzian fit results:" << std::endl;
                std::cout << "  Amplitude: " << result.amplitude << " ± " << result.amplitude_err << std::endl;
                std::cout << "  Center: " << result.center << " ± " << result.center_err << std::endl;
                std::cout << "  Gamma: " << result.gamma << " ± " << result.gamma_err << std::endl;
                std::cout << "  FWHM: " << result.fwhm << std::endl;
                std::cout << "  Baseline: " << result.baseline << " ± " << result.baseline_err << std::endl;
                std::cout << "  Chi2/NDF: " << result.reduced_chi2 << " (p=" << result.prob << ")" << std::endl;
                std::cout << "  Status: " << result.fit_status << ", EDM: " << result.edm << std::endl;
            }
        }
        
        // Cleanup
        delete func;
        delete graph;
        
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "ROOT Lorentzian fitting error: " << e.what() << std::endl;
        }
    }
    
    return result;
}

// Group data by row/column for 2D Lorentzian fitting
void GroupLorentzianDataByRow(const std::vector<double>& x_coords,
                             const std::vector<double>& y_coords,
                             const std::vector<double>& charge_values,
                             double center_y,
                             double pixel_spacing,
                             std::vector<double>& row_x,
                             std::vector<double>& row_charge) {
    
    row_x.clear();
    row_charge.clear();
    
    // Find points in the central row (within half pixel spacing of center_y)
    double tolerance = pixel_spacing * 0.6;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        if (std::abs(y_coords[i] - center_y) <= tolerance) {
            row_x.push_back(x_coords[i]);
            row_charge.push_back(charge_values[i]);
        }
    }
}

void GroupLorentzianDataByColumn(const std::vector<double>& x_coords,
                                const std::vector<double>& y_coords,
                                const std::vector<double>& charge_values,
                                double center_x,
                                double pixel_spacing,
                                std::vector<double>& col_y,
                                std::vector<double>& col_charge) {
    
    col_y.clear();
    col_charge.clear();
    
    // Find points in the central column (within half pixel spacing of center_x)
    double tolerance = pixel_spacing * 0.6;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        if (std::abs(x_coords[i] - center_x) <= tolerance) {
            col_y.push_back(y_coords[i]);
            col_charge.push_back(charge_values[i]);
        }
    }
}

// Main 2D Lorentzian fitting function
LorentzianFit2DResultsROOT Fit2DLorentzianROOT(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering,
    const std::string& minimizer,
    int max_iterations,
    double tolerance,
    bool use_weighted_fit) {
    
    LorentzianFit2DResultsROOT results;
    
    if (x_coords.size() != y_coords.size() || 
        x_coords.size() != charge_values.size() || 
        x_coords.size() < 4) {
        return results;
    }
    
    InitializeROOTLorentzian();
    
    try {
        // Apply outlier filtering if requested
        std::vector<double> filtered_x = x_coords;
        std::vector<double> filtered_y = y_coords;
        std::vector<double> filtered_charge = charge_values;
        
        if (enable_outlier_filtering) {
            auto outlier_mask = FilterLorentzianOutliers(x_coords, charge_values);
            filtered_x.clear();
            filtered_y.clear();
            filtered_charge.clear();
            
            for (size_t i = 0; i < outlier_mask.size(); ++i) {
                if (outlier_mask[i]) {
                    filtered_x.push_back(x_coords[i]);
                    filtered_y.push_back(y_coords[i]);
                    filtered_charge.push_back(charge_values[i]);
                }
            }
            
            if (filtered_x.size() < 4) {
                return results; // Not enough data after filtering
            }
        }
        
        // Group data by central row (X direction)
        std::vector<double> row_x, row_charge;
        GroupLorentzianDataByRow(filtered_x, filtered_y, filtered_charge, 
                               center_y_estimate, pixel_spacing, row_x, row_charge);
        
        // Group data by central column (Y direction)
        std::vector<double> col_y, col_charge;
        GroupLorentzianDataByColumn(filtered_x, filtered_y, filtered_charge, 
                                  center_x_estimate, pixel_spacing, col_y, col_charge);
        
        bool x_fit_successful = false;
        bool y_fit_successful = false;
        
        // Fit X direction (central row)
        if (row_x.size() >= 4) {
                        auto x_params = EstimateLorentzianParametersROOT(row_x, row_charge, 
                                                            center_x_estimate, pixel_spacing, verbose);
            if (x_params.valid) {
                auto x_result = FitLorentzianROOT(row_x, row_charge, x_params, 
                                                minimizer, max_iterations, tolerance, 
                                                use_weighted_fit, verbose);
                
                if (x_result.successful) {
                    results.x_center = x_result.center;
                    results.x_gamma = x_result.gamma;
                    results.x_amplitude = x_result.amplitude;
                    results.x_center_err = x_result.center_err;
                    results.x_gamma_err = x_result.gamma_err;
                    results.x_amplitude_err = x_result.amplitude_err;
                    results.x_vertical_offset = x_result.baseline;
                    results.x_vertical_offset_err = x_result.baseline_err;
                    results.x_chi2red = x_result.reduced_chi2;
                    results.x_pp = x_result.prob;
                    results.x_dof = x_result.ndf;
                    results.x_fwhm = x_result.fwhm;
                    results.x_fit_status = x_result.fit_status;
                    results.x_edm = x_result.edm;
                    results.x_ndf = x_result.ndf;
                    
                    x_fit_successful = true;
                }
            }
        }
        
        // Fit Y direction (central column)
        if (col_y.size() >= 4) {
                        auto y_params = EstimateLorentzianParametersROOT(col_y, col_charge, 
                                                            center_y_estimate, pixel_spacing, verbose);
            if (y_params.valid) {
                auto y_result = FitLorentzianROOT(col_y, col_charge, y_params, 
                                                minimizer, max_iterations, tolerance, 
                                                use_weighted_fit, verbose);
                
                if (y_result.successful) {
                    results.y_center = y_result.center;
                    results.y_gamma = y_result.gamma;
                    results.y_amplitude = y_result.amplitude;
                    results.y_center_err = y_result.center_err;
                    results.y_gamma_err = y_result.gamma_err;
                    results.y_amplitude_err = y_result.amplitude_err;
                    results.y_vertical_offset = y_result.baseline;
                    results.y_vertical_offset_err = y_result.baseline_err;
                    results.y_chi2red = y_result.reduced_chi2;
                    results.y_pp = y_result.prob;
                    results.y_dof = y_result.ndf;
                    results.y_fwhm = y_result.fwhm;
                    results.y_fit_status = y_result.fit_status;
                    results.y_edm = y_result.edm;
                    results.y_ndf = y_result.ndf;
                    
                    y_fit_successful = true;
                }
            }
        }
        
        results.fit_successful = x_fit_successful && y_fit_successful;
        
        if (verbose) {
            std::cout << "2D Lorentzian fit completed. X fit: " << (x_fit_successful ? "SUCCESS" : "FAILED")
                     << ", Y fit: " << (y_fit_successful ? "SUCCESS" : "FAILED") << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "Error in 2D Lorentzian fitting: " << e.what() << std::endl;
        }
    }
    
    return results;
}

// Diagonal data grouping for Lorentzian fitting
void GroupLorentzianDataByMainDiagonal(const std::vector<double>& x_coords,
                                      const std::vector<double>& y_coords,
                                      const std::vector<double>& charge_values,
                                      double center_x, double center_y,
                                      double pixel_spacing,
                                      std::vector<double>& diag_x,
                                      std::vector<double>& diag_y,
                                      std::vector<double>& diag_charge) {
    
    diag_x.clear();
    diag_y.clear();
    diag_charge.clear();
    
    // Main diagonal: y - center_y = (x - center_x), i.e., slope = 1
    double tolerance = pixel_spacing * 0.7;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double expected_y = center_y + (x_coords[i] - center_x);
        if (std::abs(y_coords[i] - expected_y) <= tolerance) {
            diag_x.push_back(x_coords[i]);
            diag_y.push_back(y_coords[i]);
            diag_charge.push_back(charge_values[i]);
        }
    }
}

void GroupLorentzianDataBySecondaryDiagonal(const std::vector<double>& x_coords,
                                           const std::vector<double>& y_coords,
                                           const std::vector<double>& charge_values,
                                           double center_x, double center_y,
                                           double pixel_spacing,
                                           std::vector<double>& diag_x,
                                           std::vector<double>& diag_y,
                                           std::vector<double>& diag_charge) {
    
    diag_x.clear();
    diag_y.clear();
    diag_charge.clear();
    
    // Secondary diagonal: y - center_y = -(x - center_x), i.e., slope = -1
    double tolerance = pixel_spacing * 0.7;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double expected_y = center_y - (x_coords[i] - center_x);
        if (std::abs(y_coords[i] - expected_y) <= tolerance) {
            diag_x.push_back(x_coords[i]);
            diag_y.push_back(y_coords[i]);
            diag_charge.push_back(charge_values[i]);
        }
    }
}

// Diagonal Lorentzian fitting function
DiagonalLorentzianFitResultsROOT FitDiagonalLorentzianROOT(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose,
    bool enable_outlier_filtering,
    const std::string& minimizer,
    int max_iterations,
    double tolerance,
    bool use_weighted_fit) {
    
    DiagonalLorentzianFitResultsROOT results;
    
    if (x_coords.size() != y_coords.size() || 
        x_coords.size() != charge_values.size() || 
        x_coords.size() < 4) {
        return results;
    }
    
    InitializeROOTLorentzian();
    
    try {
        // Apply outlier filtering if requested
        std::vector<double> filtered_x = x_coords;
        std::vector<double> filtered_y = y_coords;
        std::vector<double> filtered_charge = charge_values;
        
        if (enable_outlier_filtering) {
            auto outlier_mask = FilterLorentzianOutliers(x_coords, charge_values);
            filtered_x.clear();
            filtered_y.clear();
            filtered_charge.clear();
            
            for (size_t i = 0; i < outlier_mask.size(); ++i) {
                if (outlier_mask[i]) {
                    filtered_x.push_back(x_coords[i]);
                    filtered_y.push_back(y_coords[i]);
                    filtered_charge.push_back(charge_values[i]);
                }
            }
        }
        
        // Group data by main diagonal
        std::vector<double> main_diag_x, main_diag_y, main_diag_charge;
        GroupLorentzianDataByMainDiagonal(filtered_x, filtered_y, filtered_charge,
                                        center_x_estimate, center_y_estimate, pixel_spacing,
                                        main_diag_x, main_diag_y, main_diag_charge);
        
        // Group data by secondary diagonal
        std::vector<double> sec_diag_x, sec_diag_y, sec_diag_charge;
        GroupLorentzianDataBySecondaryDiagonal(filtered_x, filtered_y, filtered_charge,
                                             center_x_estimate, center_y_estimate, pixel_spacing,
                                             sec_diag_x, sec_diag_y, sec_diag_charge);
        
        bool main_x_successful = false, main_y_successful = false;
        bool sec_x_successful = false, sec_y_successful = false;
        
        // Fit main diagonal X vs Charge
        if (main_diag_x.size() >= 4) {
                        auto params = EstimateLorentzianParametersROOT(main_diag_x, main_diag_charge, 
                                                          center_x_estimate, pixel_spacing, verbose);
            if (params.valid) {
                auto fit_result = FitLorentzianROOT(main_diag_x, main_diag_charge, params, 
                                                  minimizer, max_iterations, tolerance, 
                                                  use_weighted_fit, verbose);
                
                if (fit_result.successful) {
                    results.main_diag_x_center = fit_result.center;
                    results.main_diag_x_gamma = fit_result.gamma;
                    results.main_diag_x_amplitude = fit_result.amplitude;
                    results.main_diag_x_center_err = fit_result.center_err;
                    results.main_diag_x_gamma_err = fit_result.gamma_err;
                    results.main_diag_x_amplitude_err = fit_result.amplitude_err;
                    results.main_diag_x_vertical_offset = fit_result.baseline;
                    results.main_diag_x_vertical_offset_err = fit_result.baseline_err;
                    results.main_diag_x_chi2red = fit_result.reduced_chi2;
                    results.main_diag_x_pp = fit_result.prob;
                    results.main_diag_x_dof = fit_result.ndf;
                    results.main_diag_x_fwhm = fit_result.fwhm;
                    results.main_diag_x_fit_status = fit_result.fit_status;
                    results.main_diag_x_edm = fit_result.edm;
                    results.main_diag_x_ndf = fit_result.ndf;
                    results.main_diag_x_fit_successful = true;
                    main_x_successful = true;
                }
            }
        }
        
        // Fit main diagonal Y vs Charge
        if (main_diag_y.size() >= 4) {
                        auto params = EstimateLorentzianParametersROOT(main_diag_y, main_diag_charge, 
                                                          center_y_estimate, pixel_spacing, verbose);
            if (params.valid) {
                auto fit_result = FitLorentzianROOT(main_diag_y, main_diag_charge, params, 
                                                  minimizer, max_iterations, tolerance, 
                                                  use_weighted_fit, verbose);
                
                if (fit_result.successful) {
                    results.main_diag_y_center = fit_result.center;
                    results.main_diag_y_gamma = fit_result.gamma;
                    results.main_diag_y_amplitude = fit_result.amplitude;
                    results.main_diag_y_center_err = fit_result.center_err;
                    results.main_diag_y_gamma_err = fit_result.gamma_err;
                    results.main_diag_y_amplitude_err = fit_result.amplitude_err;
                    results.main_diag_y_vertical_offset = fit_result.baseline;
                    results.main_diag_y_vertical_offset_err = fit_result.baseline_err;
                    results.main_diag_y_chi2red = fit_result.reduced_chi2;
                    results.main_diag_y_pp = fit_result.prob;
                    results.main_diag_y_dof = fit_result.ndf;
                    results.main_diag_y_fwhm = fit_result.fwhm;
                    results.main_diag_y_fit_status = fit_result.fit_status;
                    results.main_diag_y_edm = fit_result.edm;
                    results.main_diag_y_ndf = fit_result.ndf;
                    results.main_diag_y_fit_successful = true;
                    main_y_successful = true;
                }
            }
        }
        
        // Fit secondary diagonal X vs Charge
        if (sec_diag_x.size() >= 4) {
                        auto params = EstimateLorentzianParametersROOT(sec_diag_x, sec_diag_charge, 
                                                          center_x_estimate, pixel_spacing, verbose);
            if (params.valid) {
                auto fit_result = FitLorentzianROOT(sec_diag_x, sec_diag_charge, params, 
                                                  minimizer, max_iterations, tolerance, 
                                                  use_weighted_fit, verbose);
                
                if (fit_result.successful) {
                    results.sec_diag_x_center = fit_result.center;
                    results.sec_diag_x_gamma = fit_result.gamma;
                    results.sec_diag_x_amplitude = fit_result.amplitude;
                    results.sec_diag_x_center_err = fit_result.center_err;
                    results.sec_diag_x_gamma_err = fit_result.gamma_err;
                    results.sec_diag_x_amplitude_err = fit_result.amplitude_err;
                    results.sec_diag_x_vertical_offset = fit_result.baseline;
                    results.sec_diag_x_vertical_offset_err = fit_result.baseline_err;
                    results.sec_diag_x_chi2red = fit_result.reduced_chi2;
                    results.sec_diag_x_pp = fit_result.prob;
                    results.sec_diag_x_dof = fit_result.ndf;
                    results.sec_diag_x_fwhm = fit_result.fwhm;
                    results.sec_diag_x_fit_status = fit_result.fit_status;
                    results.sec_diag_x_edm = fit_result.edm;
                    results.sec_diag_x_ndf = fit_result.ndf;
                    results.sec_diag_x_fit_successful = true;
                    sec_x_successful = true;
                }
            }
        }
        
        // Fit secondary diagonal Y vs Charge
        if (sec_diag_y.size() >= 4) {
                        auto params = EstimateLorentzianParametersROOT(sec_diag_y, sec_diag_charge, 
                                                          center_y_estimate, pixel_spacing, verbose);
            if (params.valid) {
                auto fit_result = FitLorentzianROOT(sec_diag_y, sec_diag_charge, params, 
                                                  minimizer, max_iterations, tolerance, 
                                                  use_weighted_fit, verbose);
                
                if (fit_result.successful) {
                    results.sec_diag_y_center = fit_result.center;
                    results.sec_diag_y_gamma = fit_result.gamma;
                    results.sec_diag_y_amplitude = fit_result.amplitude;
                    results.sec_diag_y_center_err = fit_result.center_err;
                    results.sec_diag_y_gamma_err = fit_result.gamma_err;
                    results.sec_diag_y_amplitude_err = fit_result.amplitude_err;
                    results.sec_diag_y_vertical_offset = fit_result.baseline;
                    results.sec_diag_y_vertical_offset_err = fit_result.baseline_err;
                    results.sec_diag_y_chi2red = fit_result.reduced_chi2;
                    results.sec_diag_y_pp = fit_result.prob;
                    results.sec_diag_y_dof = fit_result.ndf;
                    results.sec_diag_y_fwhm = fit_result.fwhm;
                    results.sec_diag_y_fit_status = fit_result.fit_status;
                    results.sec_diag_y_edm = fit_result.edm;
                    results.sec_diag_y_ndf = fit_result.ndf;
                    results.sec_diag_y_fit_successful = true;
                    sec_y_successful = true;
                }
            }
        }
        
        results.fit_successful = main_x_successful && main_y_successful && 
                                sec_x_successful && sec_y_successful;
        
        if (verbose) {
            std::cout << "Diagonal Lorentzian fits completed:" << std::endl;
            std::cout << "  Main diagonal X: " << (main_x_successful ? "SUCCESS" : "FAILED") << std::endl;
            std::cout << "  Main diagonal Y: " << (main_y_successful ? "SUCCESS" : "FAILED") << std::endl;
            std::cout << "  Secondary diagonal X: " << (sec_x_successful ? "SUCCESS" : "FAILED") << std::endl;
            std::cout << "  Secondary diagonal Y: " << (sec_y_successful ? "SUCCESS" : "FAILED") << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "Error in diagonal Lorentzian fitting: " << e.what() << std::endl;
        }
    }
    
    return results;
} 