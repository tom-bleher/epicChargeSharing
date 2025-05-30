#include "Gaussian3DFitter.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ROOT includes (minimal to avoid Minuit issues)
#include "TMath.h"

Gaussian3DFitter::Gaussian3DFitter() 
    : fGaussianFunction(nullptr), fDataGraph(nullptr)
{
    // No ROOT objects initialization to avoid Minuit issues
}

Gaussian3DFitter::~Gaussian3DFitter()
{
    // No cleanup needed for void* pointers in this implementation
    fGaussianFunction = nullptr;
    fDataGraph = nullptr;
}

G4double Gaussian3DFitter::Gaussian3DFunction(G4double x, G4double y, const G4double* params)
{
    /*
     * 3D Gaussian function with rotation (matching Python implementation)
     * Parameters:
     * [0] = amplitude
     * [1] = x0 (center x)
     * [2] = y0 (center y)
     * [3] = sigma_x
     * [4] = sigma_y
     * [5] = theta (rotation angle in radians)
     * [6] = offset
     */
    
    const G4double amplitude = params[0];
    const G4double x0 = params[1];
    const G4double y0 = params[2];
    const G4double sigma_x = params[3];
    const G4double sigma_y = params[4];
    const G4double theta = params[5];
    const G4double offset = params[6];
    
    // Avoid division by zero
    if (sigma_x <= 0 || sigma_y <= 0) {
        return offset;
    }
    
    // Rotation transformation (same as Python version)
    const G4double cos_theta = TMath::Cos(theta);
    const G4double sin_theta = TMath::Sin(theta);
    
    const G4double x_rot = cos_theta * (x - x0) + sin_theta * (y - y0);
    const G4double y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0);
    
    // 3D Gaussian
    const G4double exponent = -(x_rot*x_rot / (2.0 * sigma_x*sigma_x) + 
                               y_rot*y_rot / (2.0 * sigma_y*sigma_y));
    
    return amplitude * TMath::Exp(exponent) + offset;
}

G4double Gaussian3DFitter::Gaussian3DFunctionWrapper(G4double* coords, G4double* params)
{
    // Wrapper function for ROOT compatibility (not used in new implementation)
    return Gaussian3DFunction(coords[0], coords[1], params);
}

void Gaussian3DFitter::CalculateInitialGuess(const std::vector<G4double>& x_coords,
                                             const std::vector<G4double>& y_coords,
                                             const std::vector<G4double>& z_values,
                                             G4double* initialParams)
{
    if (x_coords.empty() || y_coords.empty() || z_values.empty()) {
        // Set default values if no data
        initialParams[0] = 1.0;  // amplitude
        initialParams[1] = 0.0;  // x0
        initialParams[2] = 0.0;  // y0
        initialParams[3] = 1.0;  // sigma_x
        initialParams[4] = 1.0;  // sigma_y
        initialParams[5] = 0.0;  // theta
        initialParams[6] = 0.0;  // offset
        return;
    }
    
    // Calculate basic statistics
    const G4double z_min = *std::min_element(z_values.begin(), z_values.end());
    const G4double z_max = *std::max_element(z_values.begin(), z_values.end());
    
    // Calculate weighted center of mass for initial position guess
    G4double sum_z = 0.0;
    G4double weighted_x = 0.0;
    G4double weighted_y = 0.0;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        if (z_values[i] > z_min) { // Only use points above minimum
            G4double weight = z_values[i] - z_min;
            sum_z += weight;
            weighted_x += x_coords[i] * weight;
            weighted_y += y_coords[i] * weight;
        }
    }
    
    if (sum_z > 0) {
        weighted_x /= sum_z;
        weighted_y /= sum_z;
    } else {
        // Fallback to geometric center
        weighted_x = std::accumulate(x_coords.begin(), x_coords.end(), 0.0) / x_coords.size();
        weighted_y = std::accumulate(y_coords.begin(), y_coords.end(), 0.0) / y_coords.size();
    }
    
    // Calculate standard deviations as initial sigma estimates
    G4double sum_x_sq = 0.0, sum_y_sq = 0.0;
    for (size_t i = 0; i < x_coords.size(); ++i) {
        sum_x_sq += (x_coords[i] - weighted_x) * (x_coords[i] - weighted_x);
        sum_y_sq += (y_coords[i] - weighted_y) * (y_coords[i] - weighted_y);
    }
    
    const G4double sigma_x_guess = TMath::Sqrt(sum_x_sq / x_coords.size());
    const G4double sigma_y_guess = TMath::Sqrt(sum_y_sq / y_coords.size());
    
    // Set initial parameters (matching Python initial guess approach)
    initialParams[0] = z_max - z_min;           // amplitude
    initialParams[1] = weighted_x;              // x0
    initialParams[2] = weighted_y;              // y0
    initialParams[3] = TMath::Max(sigma_x_guess, 0.05); // sigma_x (minimum 0.05 mm)
    initialParams[4] = TMath::Max(sigma_y_guess, 0.05); // sigma_y (minimum 0.05 mm)
    initialParams[5] = 0.0;                     // theta (no rotation initially)
    initialParams[6] = z_min;                   // offset
}

G4double Gaussian3DFitter::CalculateRSquared(const std::vector<G4double>& x_coords,
                                             const std::vector<G4double>& y_coords,
                                             const std::vector<G4double>& z_values,
                                             const G4double* fitParams)
{
    if (z_values.empty()) return 0.0;
    
    // Calculate mean of observed values
    const G4double z_mean = std::accumulate(z_values.begin(), z_values.end(), 0.0) / z_values.size();
    
    G4double ss_tot = 0.0; // Total sum of squares
    G4double ss_res = 0.0; // Residual sum of squares
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_obs = z_values[i];
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], fitParams);
        
        ss_tot += (z_obs - z_mean) * (z_obs - z_mean);
        ss_res += (z_obs - z_fit) * (z_obs - z_fit);
    }
    
    if (ss_tot == 0.0) return 0.0;
    
    return 1.0 - (ss_res / ss_tot);
}

void Gaussian3DFitter::CalculateResidualStats(const std::vector<G4double>& x_coords,
                                              const std::vector<G4double>& y_coords,
                                              const std::vector<G4double>& z_values,
                                              const G4double* fitParams,
                                              G4double& mean, G4double& std_dev)
{
    if (x_coords.empty()) {
        mean = 0.0;
        std_dev = 0.0;
        return;
    }
    
    // Calculate residuals
    std::vector<G4double> residuals;
    residuals.reserve(x_coords.size());
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], fitParams);
        residuals.push_back(z_values[i] - z_fit);
    }
    
    // Calculate mean
    mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    
    // Calculate standard deviation
    G4double variance = 0.0;
    for (const G4double& residual : residuals) {
        variance += (residual - mean) * (residual - mean);
    }
    variance /= residuals.size();
    std_dev = TMath::Sqrt(variance);
}

// Custom Levenberg-Marquardt implementation (simplified, avoiding ROOT Minuit)
G4double Gaussian3DFitter::CalculateChiSquared(const std::vector<G4double>& x_coords,
                                               const std::vector<G4double>& y_coords,
                                               const std::vector<G4double>& z_values,
                                               const std::vector<G4double>& z_errors,
                                               const G4double* params)
{
    G4double chi2 = 0.0;
    
    for (size_t i = 0; i < x_coords.size(); ++i) {
        const G4double z_fit = Gaussian3DFunction(x_coords[i], y_coords[i], params);
        const G4double residual = z_values[i] - z_fit;
        const G4double error = (!z_errors.empty() && i < z_errors.size()) ? z_errors[i] : 1.0;
        
        if (error > 0) {
            chi2 += (residual * residual) / (error * error);
        }
    }
    
    return chi2;
}

void Gaussian3DFitter::SimplexFit(const std::vector<G4double>& x_coords,
                                  const std::vector<G4double>& y_coords,
                                  const std::vector<G4double>& z_values,
                                  const std::vector<G4double>& z_errors,
                                  G4double* params,
                                  G4bool verbose)
{
    // Simple Nelder-Mead simplex optimization (robust, no derivatives needed)
    const G4int n_params = fNParams;
    const G4int max_iterations = 1000;
    const G4double tolerance = 1e-6;
    
    // Create initial simplex
    std::vector<std::vector<G4double>> simplex(n_params + 1);
    std::vector<G4double> chi2_values(n_params + 1);
    
    // Initialize simplex vertices
    for (G4int i = 0; i <= n_params; ++i) {
        simplex[i].resize(n_params);
        for (G4int j = 0; j < n_params; ++j) {
            simplex[i][j] = params[j];
            if (i == j + 1) {
                // Perturb parameter i-1 by a small amount
                G4double step = TMath::Abs(params[j]) * 0.1;
                if (step < 1e-6) step = 0.01; // Minimum step size
                simplex[i][j] += step;
            }
        }
        chi2_values[i] = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
    }
    
    // Simplex optimization
    for (G4int iter = 0; iter < max_iterations; ++iter) {
        // Find indices of best, worst, and second worst vertices
        G4int best_idx = 0, worst_idx = 0, second_worst_idx = 0;
        
        for (G4int i = 1; i <= n_params; ++i) {
            if (chi2_values[i] < chi2_values[best_idx]) best_idx = i;
            if (chi2_values[i] > chi2_values[worst_idx]) worst_idx = i;
        }
        
        for (G4int i = 0; i <= n_params; ++i) {
            if (i != worst_idx && chi2_values[i] > chi2_values[second_worst_idx]) {
                second_worst_idx = i;
            }
        }
        
        // Check convergence
        if (TMath::Abs(chi2_values[worst_idx] - chi2_values[best_idx]) < tolerance) {
            break;
        }
        
        // Calculate centroid (excluding worst point)
        std::vector<G4double> centroid(n_params, 0.0);
        for (G4int i = 0; i <= n_params; ++i) {
            if (i != worst_idx) {
                for (G4int j = 0; j < n_params; ++j) {
                    centroid[j] += simplex[i][j];
                }
            }
        }
        for (G4int j = 0; j < n_params; ++j) {
            centroid[j] /= n_params;
        }
        
        // Reflection
        std::vector<G4double> reflected(n_params);
        for (G4int j = 0; j < n_params; ++j) {
            reflected[j] = centroid[j] + 1.0 * (centroid[j] - simplex[worst_idx][j]);
        }
        G4double chi2_reflected = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, &reflected[0]);
        
        if (chi2_reflected < chi2_values[best_idx]) {
            // Expansion
            std::vector<G4double> expanded(n_params);
            for (G4int j = 0; j < n_params; ++j) {
                expanded[j] = centroid[j] + 2.0 * (centroid[j] - simplex[worst_idx][j]);
            }
            G4double chi2_expanded = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, &expanded[0]);
            
            if (chi2_expanded < chi2_reflected) {
                simplex[worst_idx] = expanded;
                chi2_values[worst_idx] = chi2_expanded;
            } else {
                simplex[worst_idx] = reflected;
                chi2_values[worst_idx] = chi2_reflected;
            }
        } else if (chi2_reflected < chi2_values[second_worst_idx]) {
            simplex[worst_idx] = reflected;
            chi2_values[worst_idx] = chi2_reflected;
        } else {
            // Contraction
            std::vector<G4double> contracted(n_params);
            for (G4int j = 0; j < n_params; ++j) {
                contracted[j] = centroid[j] + 0.5 * (simplex[worst_idx][j] - centroid[j]);
            }
            G4double chi2_contracted = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, &contracted[0]);
            
            if (chi2_contracted < chi2_values[worst_idx]) {
                simplex[worst_idx] = contracted;
                chi2_values[worst_idx] = chi2_contracted;
            } else {
                // Shrink
                for (G4int i = 0; i <= n_params; ++i) {
                    if (i != best_idx) {
                        for (G4int j = 0; j < n_params; ++j) {
                            simplex[i][j] = simplex[best_idx][j] + 0.5 * (simplex[i][j] - simplex[best_idx][j]);
                        }
                        chi2_values[i] = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, &simplex[i][0]);
                    }
                }
            }
        }
    }
    
    // Find best solution
    G4int best_idx = 0;
    for (G4int i = 1; i <= n_params; ++i) {
        if (chi2_values[i] < chi2_values[best_idx]) best_idx = i;
    }
    
    // Copy best parameters
    for (G4int j = 0; j < n_params; ++j) {
        params[j] = simplex[best_idx][j];
    }
}

Gaussian3DFitter::FitResults Gaussian3DFitter::FitGaussian3D(const std::vector<G4double>& x_coords,
                                                            const std::vector<G4double>& y_coords,
                                                            const std::vector<G4double>& z_values,
                                                            const std::vector<G4double>& z_errors,
                                                            G4bool verbose)
{
    FitResults results;
    
    // Check input data validity
    if (x_coords.size() != y_coords.size() || x_coords.size() != z_values.size()) {
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Inconsistent input array sizes" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    if (x_coords.size() < 4) { // Minimum points for a meaningful fit
        if (verbose) {
            G4cout << "Gaussian3DFitter::FitGaussian3D - Error: Insufficient data points (" 
                   << x_coords.size() << " < 4)" << G4endl;
        }
        return results; // fit_successful remains false
    }
    
    results.n_points = x_coords.size();
    
    if (verbose) {
        G4cout << "Gaussian3DFitter::FitGaussian3D - Fitting " << results.n_points 
               << " data points" << G4endl;
    }
    
    try {
        // Calculate initial parameter estimates
        G4double fitParams[fNParams];
        CalculateInitialGuess(x_coords, y_coords, z_values, fitParams);
        
        if (verbose) {
            G4cout << "Initial parameters:" << G4endl;
            G4cout << "  Amplitude: " << fitParams[0] << G4endl;
            G4cout << "  X0: " << fitParams[1] << " mm" << G4endl;
            G4cout << "  Y0: " << fitParams[2] << " mm" << G4endl;
            G4cout << "  SigmaX: " << fitParams[3] << " mm" << G4endl;
            G4cout << "  SigmaY: " << fitParams[4] << " mm" << G4endl;
            G4cout << "  Theta: " << fitParams[5] << " rad" << G4endl;
            G4cout << "  Offset: " << fitParams[6] << G4endl;
        }
        
        // Perform simplex optimization (robust, no Minuit)
        SimplexFit(x_coords, y_coords, z_values, z_errors, fitParams, verbose);
        
        // Store fit results
        results.amplitude = fitParams[0];
        results.x0 = fitParams[1];
        results.y0 = fitParams[2];
        results.sigma_x = fitParams[3];
        results.sigma_y = fitParams[4];
        results.theta = fitParams[5];
        results.offset = fitParams[6];
        
        // Calculate chi-squared and degrees of freedom
        results.chi2 = CalculateChiSquared(x_coords, y_coords, z_values, z_errors, fitParams);
        results.ndf = results.n_points - fNParams;
        results.prob = (results.ndf > 0) ? TMath::Prob(results.chi2, results.ndf) : 0.0;
        
        // Calculate R-squared
        results.r_squared = CalculateRSquared(x_coords, y_coords, z_values, fitParams);
        
        // Calculate residual statistics
        CalculateResidualStats(x_coords, y_coords, z_values, fitParams, 
                             results.residual_mean, results.residual_std);
        
        // Simple error estimates (diagonal of covariance matrix approximation)
        // For a more robust error estimate, we'd need to implement full covariance calculation
        const G4double reduced_chi2 = results.ndf > 0 ? results.chi2 / results.ndf : 1.0;
        const G4double error_scale = TMath::Sqrt(reduced_chi2);
        
        // Rough error estimates (these could be improved with proper covariance matrix)
        results.amplitude_err = TMath::Abs(results.amplitude) * 0.1 * error_scale;
        results.x0_err = 0.01 * error_scale; // mm
        results.y0_err = 0.01 * error_scale; // mm
        results.sigma_x_err = TMath::Abs(results.sigma_x) * 0.1 * error_scale;
        results.sigma_y_err = TMath::Abs(results.sigma_y) * 0.1 * error_scale;
        results.theta_err = 0.1 * error_scale; // radians
        results.offset_err = TMath::Abs(results.offset) * 0.1 * error_scale;
        
        // Check if fit is reasonable
        if (results.sigma_x > 0 && results.sigma_y > 0 && results.r_squared > 0.1) {
            results.fit_successful = true;
        }
        
        if (verbose && results.fit_successful) {
            G4cout << "Fit successful!" << G4endl;
            G4cout << "Final parameters:" << G4endl;
            G4cout << "  Amplitude: " << results.amplitude << " ± " << results.amplitude_err << G4endl;
            G4cout << "  X0: " << results.x0 << " ± " << results.x0_err << " mm" << G4endl;
            G4cout << "  Y0: " << results.y0 << " ± " << results.y0_err << " mm" << G4endl;
            G4cout << "  SigmaX: " << results.sigma_x << " ± " << results.sigma_x_err << " mm" << G4endl;
            G4cout << "  SigmaY: " << results.sigma_y << " ± " << results.sigma_y_err << " mm" << G4endl;
            G4cout << "  Theta: " << results.theta << " ± " << results.theta_err << " rad ("
                   << results.theta * 180.0 / TMath::Pi() << "° ± " 
                   << results.theta_err * 180.0 / TMath::Pi() << "°)" << G4endl;
            G4cout << "  Offset: " << results.offset << " ± " << results.offset_err << G4endl;
            G4cout << "Fit statistics:" << G4endl;
            G4cout << "  Chi2/NDF: " << results.chi2 << " / " << results.ndf 
                   << " = " << (results.ndf > 0 ? results.chi2/results.ndf : 0) << G4endl;
            G4cout << "  Probability: " << results.prob << G4endl;
            G4cout << "  R-squared: " << results.r_squared << G4endl;
            G4cout << "  Residual mean: " << results.residual_mean << G4endl;
            G4cout << "  Residual std: " << results.residual_std << G4endl;
        } else if (verbose) {
            G4cout << "Fit failed or gave unreasonable results" << G4endl;
        }
        
    } catch (const std::exception& e) {
        if (verbose) {
            G4cout << "Exception during fitting: " << e.what() << G4endl;
        }
    } catch (...) {
        if (verbose) {
            G4cout << "Unknown exception during fitting" << G4endl;
        }
    }
    
    return results;
} 