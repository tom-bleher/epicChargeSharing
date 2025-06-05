#include "2DGaussianFit.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <mutex>

// ROOT includes
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TFitResult.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TThread.h"
#include "TROOT.h"
#include "TStyle.h"

// Thread-safe mutex for ROOT operations
static std::mutex gRootFitMutex;

GaussianFit2DResults Fit2DGaussian(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose)
{
    GaussianFit2DResults result;
    
    // Thread-safe ROOT operations
    std::lock_guard<std::mutex> lock(gRootFitMutex);
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size()) {
        if (verbose) {
            std::cout << "Fit2DGaussian: Error - coordinate and charge vector sizes don't match" << std::endl;
        }
        return result;
    }
    
    if (x_coords.size() < 3) {
        if (verbose) {
            std::cout << "Fit2DGaussian: Error - need at least 3 data points for fitting" << std::endl;
        }
        return result;
    }
    
    // Calculate total charge Q_tot for constraint calculations
    double Q_tot = 0.0;
    for (size_t i = 0; i < charge_values.size(); ++i) {
        if (charge_values[i] > 0) {
            Q_tot += charge_values[i];
        }
    }
    
    if (verbose) {
        std::cout << "Total charge Q_tot = " << Q_tot << std::endl;
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
        
        if (charge <= 0) continue; // Skip zero charge points
        
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
        if (dist < min_row_dist && row_pair.second.size() >= 3) {
            min_row_dist = dist;
            best_row_y = row_pair.first;
        }
    }
    
    double best_col_x = center_x_estimate;
    double min_col_dist = std::numeric_limits<double>::max();
    for (const auto& col_pair : cols_data) {
        double dist = std::abs(col_pair.first - center_x_estimate);
        if (dist < min_col_dist && col_pair.second.size() >= 3) {
            min_col_dist = dist;
            best_col_x = col_pair.first;
        }
    }
    
    bool x_fit_success = false;
    bool y_fit_success = false;
    
    // Static counter for unique function names (thread-safe)
    static int fit_counter = 0;
    
    // Fit X direction (central row)
    if (rows_data.find(best_row_y) != rows_data.end() && rows_data[best_row_y].size() >= 3) {
        auto& row_data = rows_data[best_row_y];
        
        // Sort by X coordinate
        std::sort(row_data.begin(), row_data.end());
        
        // Create TGraph for fitting
        std::vector<double> x_vals, y_vals;
        for (const auto& point : row_data) {
            x_vals.push_back(point.first);
            y_vals.push_back(point.second);
        }
        
        // Calculate total charge for this row for amplitude constraint
        double Q_row = 0.0;
        for (const auto& charge : y_vals) {
            Q_row += charge;
        }
        
        TGraph* graph_x = new TGraph(x_vals.size(), &x_vals[0], &y_vals[0]);
        
        // Create unique function name for thread safety
        std::string func_name = "gauss_x_" + std::to_string(fit_counter++);
        
        // Define Gaussian + offset function: y(x) = A * exp(-(x - m)^2 / (2 * sigma^2)) + B
        TF1* gauss_x = new TF1(func_name.c_str(), "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                               x_vals.front() - pixel_spacing, x_vals.back() + pixel_spacing);
        
        // Estimate initial parameters
        double max_charge = *std::max_element(y_vals.begin(), y_vals.end());
        double min_charge = *std::min_element(y_vals.begin(), y_vals.end());
        double amplitude_est = std::min(max_charge - min_charge, Q_row);
        double offset_est = std::min(min_charge, 0.1 * Q_tot); // Conservative estimate for offset
        
        // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
        gauss_x->SetParameter(0, amplitude_est);           // Amplitude
        gauss_x->SetParameter(1, center_x_estimate);       // Mean
        gauss_x->SetParameter(2, pixel_spacing);           // Sigma estimate
        gauss_x->SetParameter(3, offset_est);              // Offset
        
        // Physical constraints on parameters
        // Constraint 1: 0.1*Q_tot ≤ A ≤ ∑Q_i for row
        gauss_x->SetParLimits(0, 0.1 * Q_row, Q_row);
        
        // Constraint 2: 0 ≤ B ≤ 0.2 * Q_tot
        gauss_x->SetParLimits(3, 0, 0.2 * Q_tot);
        
        // Constraint 3: 0.2 * PixelSpacing ≤ σ ≤ 2 * PixelSpacing  
        gauss_x->SetParLimits(2, 0.2 * pixel_spacing, 2.0 * pixel_spacing);
        
        // Constraint 4: Center within 3 blocks of central block but outside pixel area
        // Calculate pixel boundaries for center constraint
        double pixel_half_size = pixel_spacing * 0.5; // Assuming square pixels
        double center_pixel_left = center_x_estimate - pixel_half_size;
        double center_pixel_right = center_x_estimate + pixel_half_size;
        double block_spacing = pixel_spacing; // Assuming blocks are pixel-spaced
        
        // Define valid ranges: within 3 blocks but outside pixel area
        // Range 1: (center - 3*spacing) to (center - 0.5*pixel_size)
        // Range 2: (center + 0.5*pixel_size) to (center + 3*spacing)
        double range1_min = center_x_estimate - 3.0 * block_spacing;
        double range1_max = center_pixel_left;
        double range2_min = center_pixel_right;
        double range2_max = center_x_estimate + 3.0 * block_spacing;
        
        // For simplicity, set broader range and let fit find optimal position
        // We'll use the union of both valid ranges as constraints
        double center_min = range1_min;
        double center_max = range2_max;
        gauss_x->SetParLimits(1, center_min, center_max);
        
        if (verbose) {
            std::cout << "X-fit constraints: A=[" << 0.1*Q_tot << "," << Q_row << "], B=[0," << 0.2*Q_tot 
                     << "], σ=[" << 0.2*pixel_spacing << "," << 2.0*pixel_spacing 
                     << "], m=[" << center_min << "," << center_max << "]" << std::endl;
        }
        
        // Fit with error handling and thread-safe options
        TFitResultPtr fit_result_x = graph_x->Fit(gauss_x, "SQN0"); // S=save result, Q=quiet, N=no draw, 0=don't store in global list
        
        if (fit_result_x.Get() && fit_result_x->IsValid()) {
            result.x_center = gauss_x->GetParameter(1);
            result.x_sigma = std::abs(gauss_x->GetParameter(2));
            result.x_amplitude = gauss_x->GetParameter(0);
            result.x_center_err = gauss_x->GetParError(1);
            result.x_sigma_err = gauss_x->GetParError(2);
            result.x_amplitude_err = gauss_x->GetParError(0);
            result.x_chi2red = gauss_x->GetChisquare() / gauss_x->GetNDF();
            result.x_npoints = x_vals.size();
            x_fit_success = true;
            
            if (verbose) {
                std::cout << "X-direction fit successful: center=" << result.x_center 
                         << ", sigma=" << result.x_sigma << ", amp=" << result.x_amplitude 
                         << ", chi2red=" << result.x_chi2red << std::endl;
            }
        } else {
            if (verbose) {
                std::cout << "X-direction fit failed" << std::endl;
            }
        }
        
        delete gauss_x;
        delete graph_x;
    }
    
    // Fit Y direction (central column)
    if (cols_data.find(best_col_x) != cols_data.end() && cols_data[best_col_x].size() >= 3) {
        auto& col_data = cols_data[best_col_x];
        
        // Sort by Y coordinate
        std::sort(col_data.begin(), col_data.end());
        
        // Create TGraph for fitting
        std::vector<double> x_vals, y_vals;
        for (const auto& point : col_data) {
            x_vals.push_back(point.first); // Y coordinate
            y_vals.push_back(point.second); // charge
        }
        
        // Calculate total charge for this column for amplitude constraint
        double Q_col = 0.0;
        for (const auto& charge : y_vals) {
            Q_col += charge;
        }
        
        TGraph* graph_y = new TGraph(x_vals.size(), &x_vals[0], &y_vals[0]);
        
        // Create unique function name for thread safety
        std::string func_name_y = "gauss_y_" + std::to_string(fit_counter++);
        
        // Define Gaussian + offset function: y(x) = A * exp(-(x - m)^2 / (2 * sigma^2)) + B
        TF1* gauss_y = new TF1(func_name_y.c_str(), "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                               x_vals.front() - pixel_spacing, x_vals.back() + pixel_spacing);
        
        // Estimate initial parameters
        double max_charge = *std::max_element(y_vals.begin(), y_vals.end());
        double min_charge = *std::min_element(y_vals.begin(), y_vals.end());
        double amplitude_est = std::min(max_charge - min_charge, Q_col);
        double offset_est = std::min(min_charge, 0.1 * Q_tot); // Conservative estimate for offset
        
        // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
        gauss_y->SetParameter(0, amplitude_est);           // Amplitude
        gauss_y->SetParameter(1, center_y_estimate);       // Mean
        gauss_y->SetParameter(2, pixel_spacing);           // Sigma estimate
        gauss_y->SetParameter(3, offset_est);              // Offset
        
        // Physical constraints on parameters
        // Constraint 1: 0.1*Q_tot ≤ A ≤ ∑Q_i for column
        gauss_y->SetParLimits(0, 0.1 * Q_tot, Q_col);
        
        // Constraint 2: 0 ≤ B ≤ 0.2 * Q_tot
        gauss_y->SetParLimits(3, 0, 0.2 * Q_tot);
        
        // Constraint 3: 0.2 * PixelSpacing ≤ σ ≤ 2 * PixelSpacing  
        gauss_y->SetParLimits(2, 0.2 * pixel_spacing, 2.0 * pixel_spacing);
        
        // Constraint 4: Center within 3 blocks of central block but outside pixel area
        // Calculate pixel boundaries for center constraint
        double pixel_half_size = pixel_spacing * 0.5; // Assuming square pixels
        double center_pixel_bottom = center_y_estimate - pixel_half_size;
        double center_pixel_top = center_y_estimate + pixel_half_size;
        double block_spacing = pixel_spacing; // Assuming blocks are pixel-spaced
        
        // Define valid ranges: within 3 blocks but outside pixel area
        // Range 1: (center - 3*spacing) to (center - 0.5*pixel_size)
        // Range 2: (center + 0.5*pixel_size) to (center + 3*spacing)
        double range1_min = center_y_estimate - 3.0 * block_spacing;
        double range1_max = center_pixel_bottom;
        double range2_min = center_pixel_top;
        double range2_max = center_y_estimate + 3.0 * block_spacing;
        
        // For simplicity, set broader range and let fit find optimal position
        // We'll use the union of both valid ranges as constraints
        double center_min = range1_min;
        double center_max = range2_max;
        gauss_y->SetParLimits(1, center_min, center_max);
        
        if (verbose) {
            std::cout << "Y-fit constraints: A=[" << 0.1*Q_tot << "," << Q_col << "], B=[0," << 0.2*Q_tot 
                     << "], σ=[" << 0.2*pixel_spacing << "," << 2.0*pixel_spacing 
                     << "], m=[" << center_min << "," << center_max << "]" << std::endl;
        }
        
        // Fit with error handling and thread-safe options
        TFitResultPtr fit_result_y = graph_y->Fit(gauss_y, "SQN0"); // S=save result, Q=quiet, N=no draw, 0=don't store in global list
        
        if (fit_result_y.Get() && fit_result_y->IsValid()) {
            result.y_center = gauss_y->GetParameter(1);
            result.y_sigma = std::abs(gauss_y->GetParameter(2));
            result.y_amplitude = gauss_y->GetParameter(0);
            result.y_center_err = gauss_y->GetParError(1);
            result.y_sigma_err = gauss_y->GetParError(2);
            result.y_amplitude_err = gauss_y->GetParError(0);
            result.y_chi2red = gauss_y->GetChisquare() / gauss_y->GetNDF();
            result.y_npoints = x_vals.size();
            y_fit_success = true;
            
            if (verbose) {
                std::cout << "Y-direction fit successful: center=" << result.y_center 
                         << ", sigma=" << result.y_sigma << ", amp=" << result.y_amplitude 
                         << ", chi2red=" << result.y_chi2red << std::endl;
            }
        } else {
            if (verbose) {
                std::cout << "Y-direction fit failed" << std::endl;
            }
        }
        
        delete gauss_y;
        delete graph_y;
    }
    
    // Set overall success status
    result.fit_successful = x_fit_success && y_fit_success;
    
    if (verbose) {
        std::cout << "2D Gaussian fit " << (result.fit_successful ? "successful" : "failed") 
                 << " (X: " << (x_fit_success ? "OK" : "FAIL") 
                 << ", Y: " << (y_fit_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
} 