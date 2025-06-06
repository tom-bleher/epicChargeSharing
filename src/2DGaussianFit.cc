#include "2DGaussianFit.hh"
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
static std::atomic<int> gGlobalFitCounter{0};

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
    
    // Initialize ROOT for thread safety if not already done
    static bool root_initialized = false;
    if (!root_initialized) {
        gROOT->SetBatch(kTRUE);  // Run in batch mode to avoid graphics issues
        root_initialized = true;
    }
    
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
    
    // Create unique identifiers for this fit session
    auto thread_id = std::this_thread::get_id();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    int fit_id = gGlobalFitCounter.fetch_add(1);
    
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
        
        // Use stack-allocated TF1 with explicit no-registration
        TF1 gauss_x("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                   x_vals.front() - pixel_spacing, x_vals.back() + pixel_spacing, TF1::EAddToList::kNo);
        
        // Estimate initial parameters
        double max_charge = *std::max_element(y_vals.begin(), y_vals.end());
        double min_charge = *std::min_element(y_vals.begin(), y_vals.end());
        double amplitude_est = std::min(max_charge - min_charge, Q_row);
        double offset_est = std::min(min_charge, 0.1 * Q_tot); // Conservative estimate for offset
        
        // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
        gauss_x.SetParameter(0, amplitude_est);           // Amplitude
        gauss_x.SetParameter(1, center_x_estimate);       // Mean
        gauss_x.SetParameter(2, pixel_spacing);           // Sigma estimate
        gauss_x.SetParameter(3, offset_est);              // Offset
        
        // Physical constraints on parameters
        // Constraint 1: 0.1*Q_tot ≤ A ≤ ∑Q_i for row
        gauss_x.SetParLimits(0, 0.1 * Q_row, Q_row);
        
        // Constraint 2: 0 ≤ B ≤ 0.2 * Q_tot
        gauss_x.SetParLimits(3, 0.08 * Q_row, 0.5 * Q_row);
        
        // Constraint 3: 0.2 * PixelSpacing ≤ σ ≤ 2 * PixelSpacing  
        gauss_x.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
        
        // Constraint 4: Center within ±pixel_spacing of center estimate
        double center_min = center_x_estimate - pixel_spacing;
        double center_max = center_x_estimate + pixel_spacing;
        gauss_x.SetParLimits(1, center_min, center_max);
        
        if (verbose) {
            std::cout << "X-fit constraints: A=[" << 0.1*Q_row << "," << Q_row << "], B=[" << 0.08*Q_row << "," << 0.5*Q_row 
                     << "], σ=[" << 0.2*pixel_spacing << "," << 2.0*pixel_spacing 
                     << "], m=[" << center_min << "," << center_max << "]" << std::endl;
        }
        
        // Fit with error handling and thread-safe options
        TFitResultPtr fit_result_x = graph_x->Fit(&gauss_x, "SQN0"); // S=save result, Q=quiet, N=no draw, 0=don't store in global list
        
        if (fit_result_x.Get() && fit_result_x->IsValid()) {
            result.x_center = gauss_x.GetParameter(1);
            result.x_sigma = std::abs(gauss_x.GetParameter(2));
            result.x_amplitude = gauss_x.GetParameter(0);
            result.x_center_err = gauss_x.GetParError(1);
            result.x_sigma_err = gauss_x.GetParError(2);
            result.x_amplitude_err = gauss_x.GetParError(0);
            result.x_chi2red = gauss_x.GetChisquare() / gauss_x.GetNDF();
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
        
        // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
        delete graph_x;
        graph_x = nullptr;
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
        
        // Use stack-allocated TF1 with explicit no-registration
        TF1 gauss_y("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                   x_vals.front() - pixel_spacing, x_vals.back() + pixel_spacing, TF1::EAddToList::kNo);
        
        // Estimate initial parameters
        double max_charge = *std::max_element(y_vals.begin(), y_vals.end());
        double min_charge = *std::min_element(y_vals.begin(), y_vals.end());
        double amplitude_est = std::min(max_charge - min_charge, Q_col);
        double offset_est = std::min(min_charge, 0.1 * Q_tot); // Conservative estimate for offset
        
        // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
        gauss_y.SetParameter(0, amplitude_est);           // Amplitude
        gauss_y.SetParameter(1, center_y_estimate);       // Mean
        gauss_y.SetParameter(2, pixel_spacing);           // Sigma estimate
        gauss_y.SetParameter(3, offset_est);              // Offset
        
        // Physical constraints on parameters
        // Constraint 1: 0.1*Q_tot ≤ A ≤ ∑Q_i for column
        gauss_y.SetParLimits(0, 0.1 * Q_col, Q_col);
        
        // Constraint 2: 0 ≤ B ≤ 0.2 * Q_tot
        gauss_y.SetParLimits(3, 0.08*Q_col, 0.5 * Q_col);
        
        // Constraint 3: 0.2 * PixelSpacing ≤ σ ≤ 2 * PixelSpacing  
        gauss_y.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
        
        // Constraint 4: Center within ±pixel_spacing of center estimate
        double center_min = center_y_estimate - pixel_spacing;
        double center_max = center_y_estimate + pixel_spacing;
        gauss_y.SetParLimits(1, center_min, center_max);
        
        if (verbose) {
            std::cout << "Y-fit constraints: A=[" << 0.1*Q_col << "," << Q_col << "], B=[" << 0.08*Q_col << "," << 0.5*Q_col 
                     << "], σ=[" << 0.2*pixel_spacing << "," << 2.0*pixel_spacing 
                     << "], m=[" << center_min << "," << center_max << "]" << std::endl;
        }
        
        // Fit with error handling and thread-safe options
        TFitResultPtr fit_result_y = graph_y->Fit(&gauss_y, "SQN0"); // S=save result, Q=quiet, N=no draw, 0=don't store in global list
        
        if (fit_result_y.Get() && fit_result_y->IsValid()) {
            result.y_center = gauss_y.GetParameter(1);
            result.y_sigma = std::abs(gauss_y.GetParameter(2));
            result.y_amplitude = gauss_y.GetParameter(0);
            result.y_center_err = gauss_y.GetParError(1);
            result.y_sigma_err = gauss_y.GetParError(2);
            result.y_amplitude_err = gauss_y.GetParError(0);
            result.y_chi2red = gauss_y.GetChisquare() / gauss_y.GetNDF();
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
        
        // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
        delete graph_y;
        graph_y = nullptr;
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

DiagonalFitResults FitDiagonalGaussian(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords, 
    const std::vector<double>& charge_values,
    double center_x_estimate,
    double center_y_estimate,
    double pixel_spacing,
    bool verbose)
{
    DiagonalFitResults result;
    
    if (x_coords.size() != y_coords.size() || x_coords.size() != charge_values.size() || x_coords.size() < 3) {
        if (verbose) {
            std::cout << "Diagonal Gaussian fit: Invalid input data size" << std::endl;
        }
        return result;
    }
    
    if (verbose) {
        std::cout << "Starting Diagonal Gaussian fit with " << x_coords.size() << " data points" << std::endl;
        std::cout << "Center estimates: x=" << center_x_estimate << ", y=" << center_y_estimate << std::endl;
    }
    
    // Tolerance for grouping pixels into diagonals
    double tolerance = pixel_spacing * 0.1;
    
    // Group pixels by diagonal lines
    std::map<double, std::vector<std::pair<double, double>>> main_diagonal_data; // main_diag_id -> [(x, charge), (y, charge)]
    std::map<double, std::vector<std::pair<double, double>>> sec_diagonal_data;  // sec_diag_id -> [(x, charge), (y, charge)]
    
    // Group data points by diagonal
    for (size_t i = 0; i < x_coords.size(); ++i) {
        double x = x_coords[i];
        double y = y_coords[i];
        double charge = charge_values[i];
        
        if (charge <= 0) continue; // Skip zero charge points
        
        // Main diagonal: constant value of (x - y)
        // Points on same main diagonal have x - y = constant
        double main_diag_id = x - y;
        
        // Secondary diagonal: constant value of (x + y)  
        // Points on same secondary diagonal have x + y = constant
        double sec_diag_id = x + y;
        
        // Find or create main diagonal - store both x and y coordinates with charge
        bool found_main_diag = false;
        for (auto& diag_pair : main_diagonal_data) {
            if (std::abs(diag_pair.first - main_diag_id) < tolerance) {
                diag_pair.second.push_back(std::make_pair(x, charge));
                diag_pair.second.push_back(std::make_pair(y, charge));
                found_main_diag = true;
                break;
            }
        }
        if (!found_main_diag) {
            main_diagonal_data[main_diag_id].push_back(std::make_pair(x, charge));
            main_diagonal_data[main_diag_id].push_back(std::make_pair(y, charge));
        }
        
        // Find or create secondary diagonal - store both x and y coordinates with charge
        bool found_sec_diag = false;
        for (auto& diag_pair : sec_diagonal_data) {
            if (std::abs(diag_pair.first - sec_diag_id) < tolerance) {
                diag_pair.second.push_back(std::make_pair(x, charge));
                diag_pair.second.push_back(std::make_pair(y, charge));
                found_sec_diag = true;
                break;
            }
        }
        if (!found_sec_diag) {
            sec_diagonal_data[sec_diag_id].push_back(std::make_pair(x, charge));
            sec_diagonal_data[sec_diag_id].push_back(std::make_pair(y, charge));
        }
    }
    
    // Find the main diagonal closest to the center estimates
    double center_main_diag_id = center_x_estimate - center_y_estimate;
    double best_main_diag_id = center_main_diag_id;
    double min_main_diag_dist = std::numeric_limits<double>::max();
    for (const auto& diag_pair : main_diagonal_data) {
        double dist = std::abs(diag_pair.first - center_main_diag_id);
        if (dist < min_main_diag_dist && diag_pair.second.size() >= 6) { // Need at least 3 points for each x and y fit
            min_main_diag_dist = dist;
            best_main_diag_id = diag_pair.first;
        }
    }
    
    // Find the secondary diagonal closest to the center estimates
    double center_sec_diag_id = center_x_estimate + center_y_estimate;
    double best_sec_diag_id = center_sec_diag_id;
    double min_sec_diag_dist = std::numeric_limits<double>::max();
    for (const auto& diag_pair : sec_diagonal_data) {
        double dist = std::abs(diag_pair.first - center_sec_diag_id);
        if (dist < min_sec_diag_dist && diag_pair.second.size() >= 6) { // Need at least 3 points for each x and y fit
            min_sec_diag_dist = dist;
            best_sec_diag_id = diag_pair.first;
        }
    }
    
    // Calculate total charge for amplitude constraints
    double Q_tot = 0.0;
    for (const auto& charge : charge_values) {
        Q_tot += charge;
    }
    
    bool main_diag_x_success = false;
    bool main_diag_y_success = false;
    bool sec_diag_x_success = false;
    bool sec_diag_y_success = false;
    
    // Create unique identifiers for this fit session
    auto thread_id = std::this_thread::get_id();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    int fit_id = gGlobalFitCounter.fetch_add(1);
    
    // ========================================
    // FIT MAIN DIAGONAL X (X vs Charge)
    // ========================================
    if (main_diagonal_data.find(best_main_diag_id) != main_diagonal_data.end() && 
        main_diagonal_data[best_main_diag_id].size() >= 6) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Extract X coordinates and charges (every other pair starting from index 0)
        std::vector<double> x_vals, charge_vals;
        for (size_t i = 0; i < main_diag_data.size(); i += 2) {
            x_vals.push_back(main_diag_data[i].first);
            charge_vals.push_back(main_diag_data[i].second);
        }
        
        // Sort by X coordinate
        std::vector<std::pair<double, double>> x_charge_pairs;
        for (size_t i = 0; i < x_vals.size(); ++i) {
            x_charge_pairs.push_back(std::make_pair(x_vals[i], charge_vals[i]));
        }
        std::sort(x_charge_pairs.begin(), x_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> x_sorted, charge_x_sorted;
        for (const auto& pair : x_charge_pairs) {
            x_sorted.push_back(pair.first);
            charge_x_sorted.push_back(pair.second);
        }
        
        if (x_sorted.size() >= 3) {
            // Calculate charge sum for this diagonal X fit
            double Q_main_x = 0.0;
            for (const auto& charge : charge_x_sorted) {
                Q_main_x += charge;
            }
            
            TGraph* graph_main_x = new TGraph(x_sorted.size(), &x_sorted[0], &charge_x_sorted[0]);
            
            // Use stack-allocated TF1 with explicit no-registration
            TF1 gauss_main_x("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                            x_sorted.front() - pixel_spacing, x_sorted.back() + pixel_spacing, TF1::EAddToList::kNo);
            
            // Estimate initial parameters
            double max_charge = *std::max_element(charge_x_sorted.begin(), charge_x_sorted.end());
            double min_charge = *std::min_element(charge_x_sorted.begin(), charge_x_sorted.end());
            double amplitude_est = std::min(max_charge - min_charge, Q_main_x);
            double offset_est = std::min(min_charge, 0.1 * Q_tot);
            
            // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
            gauss_main_x.SetParameter(0, amplitude_est);         // Amplitude
            gauss_main_x.SetParameter(1, center_x_estimate);     // Mean X position
            gauss_main_x.SetParameter(2, pixel_spacing);         // Sigma estimate
            gauss_main_x.SetParameter(3, offset_est);            // Offset
            
            // Physical constraints on parameters
            gauss_main_x.SetParLimits(0, 0.1 * Q_main_x, Q_main_x);
            gauss_main_x.SetParLimits(3, 0.08 * Q_main_x, 0.5 * Q_main_x);
            gauss_main_x.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
            
            double center_min = center_x_estimate - pixel_spacing;
            double center_max = center_x_estimate + pixel_spacing;
            gauss_main_x.SetParLimits(1, center_min, center_max);
            
            if (verbose) {
                std::cout << "Main diagonal X fit constraints: A=[" << 0.1*Q_main_x << "," << Q_main_x 
                         << "], center=[" << center_min << "," << center_max << "]" << std::endl;
            }
            
            // Fit with error handling and thread-safe options
            TFitResultPtr fit_result_main_x = graph_main_x->Fit(&gauss_main_x, "SQN0");
            
            if (fit_result_main_x.Get() && fit_result_main_x->IsValid()) {
                result.main_diag_x_center = gauss_main_x.GetParameter(1);
                result.main_diag_x_center_err = gauss_main_x.GetParError(1);
                result.main_diag_x_sigma = std::abs(gauss_main_x.GetParameter(2));
                result.main_diag_x_amplitude = gauss_main_x.GetParameter(0);
                result.main_diag_x_sigma_err = gauss_main_x.GetParError(2);
                result.main_diag_x_amplitude_err = gauss_main_x.GetParError(0);
                result.main_diag_x_chi2red = gauss_main_x.GetChisquare() / gauss_main_x.GetNDF();
                result.main_diag_x_npoints = x_sorted.size();
                result.main_diag_x_fit_successful = true;
                main_diag_x_success = true;
                
                if (verbose) {
                    std::cout << "Main diagonal X fit successful: center=" << result.main_diag_x_center 
                             << ", sigma=" << result.main_diag_x_sigma << ", chi2red=" << result.main_diag_x_chi2red << std::endl;
                }
            } else {
                if (verbose) {
                    std::cout << "Main diagonal X fit failed" << std::endl;
                }
            }
            
            // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
            delete graph_main_x;
            graph_main_x = nullptr;
        }
    }
    
    // ========================================
    // FIT MAIN DIAGONAL Y (Y vs Charge)
    // ========================================
    if (main_diagonal_data.find(best_main_diag_id) != main_diagonal_data.end() && 
        main_diagonal_data[best_main_diag_id].size() >= 6) {
        
        auto& main_diag_data = main_diagonal_data[best_main_diag_id];
        
        // Extract Y coordinates and charges (every other pair starting from index 1)
        std::vector<double> y_vals, charge_vals;
        for (size_t i = 1; i < main_diag_data.size(); i += 2) {
            y_vals.push_back(main_diag_data[i].first);
            charge_vals.push_back(main_diag_data[i].second);
        }
        
        // Sort by Y coordinate
        std::vector<std::pair<double, double>> y_charge_pairs;
        for (size_t i = 0; i < y_vals.size(); ++i) {
            y_charge_pairs.push_back(std::make_pair(y_vals[i], charge_vals[i]));
        }
        std::sort(y_charge_pairs.begin(), y_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> y_sorted, charge_y_sorted;
        for (const auto& pair : y_charge_pairs) {
            y_sorted.push_back(pair.first);
            charge_y_sorted.push_back(pair.second);
        }
        
        if (y_sorted.size() >= 3) {
            // Calculate charge sum for this diagonal Y fit
            double Q_main_y = 0.0;
            for (const auto& charge : charge_y_sorted) {
                Q_main_y += charge;
            }
            
            TGraph* graph_main_y = new TGraph(y_sorted.size(), &y_sorted[0], &charge_y_sorted[0]);
            
            // Use stack-allocated TF1 with explicit no-registration
            TF1 gauss_main_y("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                            y_sorted.front() - pixel_spacing, y_sorted.back() + pixel_spacing, TF1::EAddToList::kNo);
            
            // Estimate initial parameters
            double max_charge = *std::max_element(charge_y_sorted.begin(), charge_y_sorted.end());
            double min_charge = *std::min_element(charge_y_sorted.begin(), charge_y_sorted.end());
            double amplitude_est = std::min(max_charge - min_charge, Q_main_y);
            double offset_est = std::min(min_charge, 0.1 * Q_tot);
            
            // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
            gauss_main_y.SetParameter(0, amplitude_est);         // Amplitude
            gauss_main_y.SetParameter(1, center_y_estimate);     // Mean Y position
            gauss_main_y.SetParameter(2, pixel_spacing);         // Sigma estimate
            gauss_main_y.SetParameter(3, offset_est);            // Offset
            
            // Physical constraints on parameters
            gauss_main_y.SetParLimits(0, 0.1 * Q_main_y, Q_main_y);
            gauss_main_y.SetParLimits(3, 0.08 * Q_main_y, 0.5 * Q_main_y);
            gauss_main_y.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
            
            double center_min = center_y_estimate - pixel_spacing;
            double center_max = center_y_estimate + pixel_spacing;
            gauss_main_y.SetParLimits(1, center_min, center_max);
            
            if (verbose) {
                std::cout << "Main diagonal Y fit constraints: A=[" << 0.1*Q_main_y << "," << Q_main_y 
                         << "], center=[" << center_min << "," << center_max << "]" << std::endl;
            }
            
            // Fit with error handling and thread-safe options
            TFitResultPtr fit_result_main_y = graph_main_y->Fit(&gauss_main_y, "SQN0");
            
            if (fit_result_main_y.Get() && fit_result_main_y->IsValid()) {
                result.main_diag_y_center = gauss_main_y.GetParameter(1);
                result.main_diag_y_center_err = gauss_main_y.GetParError(1);
                result.main_diag_y_sigma = std::abs(gauss_main_y.GetParameter(2));
                result.main_diag_y_amplitude = gauss_main_y.GetParameter(0);
                result.main_diag_y_sigma_err = gauss_main_y.GetParError(2);
                result.main_diag_y_amplitude_err = gauss_main_y.GetParError(0);
                result.main_diag_y_chi2red = gauss_main_y.GetChisquare() / gauss_main_y.GetNDF();
                result.main_diag_y_npoints = y_sorted.size();
                result.main_diag_y_fit_successful = true;
                main_diag_y_success = true;
                
                if (verbose) {
                    std::cout << "Main diagonal Y fit successful: center=" << result.main_diag_y_center 
                             << ", sigma=" << result.main_diag_y_sigma << ", chi2red=" << result.main_diag_y_chi2red << std::endl;
                }
            } else {
                if (verbose) {
                    std::cout << "Main diagonal Y fit failed" << std::endl;
                }
            }
            
            // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
            delete graph_main_y;
            graph_main_y = nullptr;
        }
    }
    
    // ========================================
    // FIT SECONDARY DIAGONAL X (X vs Charge)
    // ========================================
    if (sec_diagonal_data.find(best_sec_diag_id) != sec_diagonal_data.end() && 
        sec_diagonal_data[best_sec_diag_id].size() >= 6) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Extract X coordinates and charges (every other pair starting from index 0)
        std::vector<double> x_vals, charge_vals;
        for (size_t i = 0; i < sec_diag_data.size(); i += 2) {
            x_vals.push_back(sec_diag_data[i].first);
            charge_vals.push_back(sec_diag_data[i].second);
        }
        
        // Sort by X coordinate
        std::vector<std::pair<double, double>> x_charge_pairs;
        for (size_t i = 0; i < x_vals.size(); ++i) {
            x_charge_pairs.push_back(std::make_pair(x_vals[i], charge_vals[i]));
        }
        std::sort(x_charge_pairs.begin(), x_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> x_sorted, charge_x_sorted;
        for (const auto& pair : x_charge_pairs) {
            x_sorted.push_back(pair.first);
            charge_x_sorted.push_back(pair.second);
        }
        
        if (x_sorted.size() >= 3) {
            // Calculate charge sum for this diagonal X fit
            double Q_sec_x = 0.0;
            for (const auto& charge : charge_x_sorted) {
                Q_sec_x += charge;
            }
            
            TGraph* graph_sec_x = new TGraph(x_sorted.size(), &x_sorted[0], &charge_x_sorted[0]);
            
            // Use stack-allocated TF1 with explicit no-registration
            TF1 gauss_sec_x("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                           x_sorted.front() - pixel_spacing, x_sorted.back() + pixel_spacing, TF1::EAddToList::kNo);
            
            // Estimate initial parameters
            double max_charge = *std::max_element(charge_x_sorted.begin(), charge_x_sorted.end());
            double min_charge = *std::min_element(charge_x_sorted.begin(), charge_x_sorted.end());
            double amplitude_est = std::min(max_charge - min_charge, Q_sec_x);
            double offset_est = std::min(min_charge, 0.1 * Q_tot);
            
            // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
            gauss_sec_x.SetParameter(0, amplitude_est);         // Amplitude
            gauss_sec_x.SetParameter(1, center_x_estimate);     // Mean X position
            gauss_sec_x.SetParameter(2, pixel_spacing);         // Sigma estimate
            gauss_sec_x.SetParameter(3, offset_est);            // Offset
            
            // Physical constraints on parameters
            gauss_sec_x.SetParLimits(0, 0.1 * Q_sec_x, Q_sec_x);
            gauss_sec_x.SetParLimits(3, 0.08 * Q_sec_x, 0.5 * Q_sec_x);
            gauss_sec_x.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
            
            double center_min = center_x_estimate - pixel_spacing;
            double center_max = center_x_estimate + pixel_spacing;
            gauss_sec_x.SetParLimits(1, center_min, center_max);
            
            if (verbose) {
                std::cout << "Secondary diagonal X fit constraints: A=[" << 0.1*Q_sec_x << "," << Q_sec_x 
                         << "], center=[" << center_min << "," << center_max << "]" << std::endl;
            }
            
            // Fit with error handling and thread-safe options
            TFitResultPtr fit_result_sec_x = graph_sec_x->Fit(&gauss_sec_x, "SQN0");
            
            if (fit_result_sec_x.Get() && fit_result_sec_x->IsValid()) {
                result.sec_diag_x_center = gauss_sec_x.GetParameter(1);
                result.sec_diag_x_center_err = gauss_sec_x.GetParError(1);
                result.sec_diag_x_sigma = std::abs(gauss_sec_x.GetParameter(2));
                result.sec_diag_x_amplitude = gauss_sec_x.GetParameter(0);
                result.sec_diag_x_sigma_err = gauss_sec_x.GetParError(2);
                result.sec_diag_x_amplitude_err = gauss_sec_x.GetParError(0);
                result.sec_diag_x_chi2red = gauss_sec_x.GetChisquare() / gauss_sec_x.GetNDF();
                result.sec_diag_x_npoints = x_sorted.size();
                result.sec_diag_x_fit_successful = true;
                sec_diag_x_success = true;
                
                if (verbose) {
                    std::cout << "Secondary diagonal X fit successful: center=" << result.sec_diag_x_center 
                             << ", sigma=" << result.sec_diag_x_sigma << ", chi2red=" << result.sec_diag_x_chi2red << std::endl;
                }
            } else {
                if (verbose) {
                    std::cout << "Secondary diagonal X fit failed" << std::endl;
                }
            }
            
            // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
            delete graph_sec_x;
            graph_sec_x = nullptr;
        }
    }
    
    // ========================================
    // FIT SECONDARY DIAGONAL Y (Y vs Charge)
    // ========================================
    if (sec_diagonal_data.find(best_sec_diag_id) != sec_diagonal_data.end() && 
        sec_diagonal_data[best_sec_diag_id].size() >= 6) {
        
        auto& sec_diag_data = sec_diagonal_data[best_sec_diag_id];
        
        // Extract Y coordinates and charges (every other pair starting from index 1)
        std::vector<double> y_vals, charge_vals;
        for (size_t i = 1; i < sec_diag_data.size(); i += 2) {
            y_vals.push_back(sec_diag_data[i].first);
            charge_vals.push_back(sec_diag_data[i].second);
        }
        
        // Sort by Y coordinate
        std::vector<std::pair<double, double>> y_charge_pairs;
        for (size_t i = 0; i < y_vals.size(); ++i) {
            y_charge_pairs.push_back(std::make_pair(y_vals[i], charge_vals[i]));
        }
        std::sort(y_charge_pairs.begin(), y_charge_pairs.end());
        
        // Create sorted vectors
        std::vector<double> y_sorted, charge_y_sorted;
        for (const auto& pair : y_charge_pairs) {
            y_sorted.push_back(pair.first);
            charge_y_sorted.push_back(pair.second);
        }
        
        if (y_sorted.size() >= 3) {
            // Calculate charge sum for this diagonal Y fit
            double Q_sec_y = 0.0;
            for (const auto& charge : charge_y_sorted) {
                Q_sec_y += charge;
            }
            
            TGraph* graph_sec_y = new TGraph(y_sorted.size(), &y_sorted[0], &charge_y_sorted[0]);
            
            // Use stack-allocated TF1 with explicit no-registration
            TF1 gauss_sec_y("", "[0]*TMath::Exp(-0.5*TMath::Power((x-[1])/[2],2)) + [3]", 
                           y_sorted.front() - pixel_spacing, y_sorted.back() + pixel_spacing, TF1::EAddToList::kNo);
            
            // Estimate initial parameters
            double max_charge = *std::max_element(charge_y_sorted.begin(), charge_y_sorted.end());
            double min_charge = *std::min_element(charge_y_sorted.begin(), charge_y_sorted.end());
            double amplitude_est = std::min(max_charge - min_charge, Q_sec_y);
            double offset_est = std::min(min_charge, 0.1 * Q_tot);
            
            // Set initial parameters: [0]=A, [1]=m, [2]=sigma, [3]=B
            gauss_sec_y.SetParameter(0, amplitude_est);         // Amplitude
            gauss_sec_y.SetParameter(1, center_y_estimate);     // Mean Y position
            gauss_sec_y.SetParameter(2, pixel_spacing);         // Sigma estimate
            gauss_sec_y.SetParameter(3, offset_est);            // Offset
            
            // Physical constraints on parameters
            gauss_sec_y.SetParLimits(0, 0.1 * Q_sec_y, Q_sec_y);
            gauss_sec_y.SetParLimits(3, 0.08 * Q_sec_y, 0.5 * Q_sec_y);
            gauss_sec_y.SetParLimits(2, 0.2 * pixel_spacing, 4.0*pixel_spacing);
            
            double center_min = center_y_estimate - pixel_spacing;
            double center_max = center_y_estimate + pixel_spacing;
            gauss_sec_y.SetParLimits(1, center_min, center_max);
            
            if (verbose) {
                std::cout << "Secondary diagonal Y fit constraints: A=[" << 0.1*Q_sec_y << "," << Q_sec_y 
                         << "], center=[" << center_min << "," << center_max << "]" << std::endl;
            }
            
            // Fit with error handling and thread-safe options
            TFitResultPtr fit_result_sec_y = graph_sec_y->Fit(&gauss_sec_y, "SQN0");
            
            if (fit_result_sec_y.Get() && fit_result_sec_y->IsValid()) {
                result.sec_diag_y_center = gauss_sec_y.GetParameter(1);
                result.sec_diag_y_center_err = gauss_sec_y.GetParError(1);
                result.sec_diag_y_sigma = std::abs(gauss_sec_y.GetParameter(2));
                result.sec_diag_y_amplitude = gauss_sec_y.GetParameter(0);
                result.sec_diag_y_sigma_err = gauss_sec_y.GetParError(2);
                result.sec_diag_y_amplitude_err = gauss_sec_y.GetParError(0);
                result.sec_diag_y_chi2red = gauss_sec_y.GetChisquare() / gauss_sec_y.GetNDF();
                result.sec_diag_y_npoints = y_sorted.size();
                result.sec_diag_y_fit_successful = true;
                sec_diag_y_success = true;
                
                if (verbose) {
                    std::cout << "Secondary diagonal Y fit successful: center=" << result.sec_diag_y_center 
                             << ", sigma=" << result.sec_diag_y_sigma << ", chi2red=" << result.sec_diag_y_chi2red << std::endl;
                }
            } else {
                if (verbose) {
                    std::cout << "Secondary diagonal Y fit failed" << std::endl;
                }
            }
            
            // Safe cleanup: only delete TGraph (TF1 is stack-allocated)
            delete graph_sec_y;
            graph_sec_y = nullptr;
        }
    }
    
    // Set overall success status
    result.fit_successful = main_diag_x_success && main_diag_y_success && sec_diag_x_success && sec_diag_y_success;
    
    if (verbose) {
        std::cout << "Diagonal Gaussian fit " << (result.fit_successful ? "successful" : "failed") 
                 << " (Main X: " << (main_diag_x_success ? "OK" : "FAIL") 
                 << ", Main Y: " << (main_diag_y_success ? "OK" : "FAIL")
                 << ", Sec X: " << (sec_diag_x_success ? "OK" : "FAIL") 
                 << ", Sec Y: " << (sec_diag_y_success ? "OK" : "FAIL") << ")" << std::endl;
    }
    
    return result;
} 