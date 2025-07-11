/*
 * Parallel Spatial Resolution Calculator ROOT Macro
 * 
 * This ROOT macro reads simulated data and calculates spatial resolution
 * for various reconstruction methods using parallel processing.
 * 
 * Usage: root -l -b -q calc_resolution_parallel.C
 */

 #include <iostream>
 #include <fstream>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <algorithm>
 #include <map>
 #include <iomanip>
 #include <thread>
 #include <ROOT/RDataFrame.hxx>
 #include <TFile.h>
 #include <TTree.h>
 #include <TMath.h>
 
 using namespace std;
 
 struct ResolutionResult {
     string name;
     int n_events;
     double mean_bias;
     double std_dev;
     double rms;
     double median;
     double min_val;
     double max_val;
 };
 
 ResolutionResult calculateStatsFromVector(const vector<double>& data, const string& method_name) {
     ResolutionResult result;
     result.name = method_name;
     result.n_events = data.size();
     
     if (data.empty()) {
         result.mean_bias = 0;
         result.std_dev = 0;
         result.rms = 0;
         result.median = 0;
         result.min_val = 0;
         result.max_val = 0;
         return result;
     }
     
     // Calculate mean
     double sum = 0.0;
     for (size_t i = 0; i < data.size(); ++i) {
         sum += data[i];
     }
     result.mean_bias = sum / data.size();
     
     // Calculate standard deviation and RMS
     double sum_sq = 0.0;
     double sum_sq_diff = 0.0;
     for (size_t i = 0; i < data.size(); ++i) {
         double val = data[i];
         sum_sq += val * val;
         double diff = val - result.mean_bias;
         sum_sq_diff += diff * diff;
     }
     
     if (data.size() > 1) {
         result.std_dev = sqrt(sum_sq_diff / (data.size() - 1));
     } else {
         result.std_dev = 0.0;
     }
     
     result.rms = sqrt(sum_sq / data.size());
     
     // Calculate median
     vector<double> sorted_data = data;
     sort(sorted_data.begin(), sorted_data.end());
     size_t n = sorted_data.size();
     if (n % 2 == 0) {
         result.median = (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0;
     } else {
         result.median = sorted_data[n/2];
     }
     
     result.min_val = sorted_data[0];
     result.max_val = sorted_data[n-1];
     
     return result;
 }
 
 void calc_resolution_parallel() {
     cout << "Parallel Spatial Resolution Calculator for PixelChargeSharingToy" << endl;
     cout << "=================================================================" << endl;
     
     // Enable multithreading - use all available cores
     unsigned int nCores = thread::hardware_concurrency();
     cout << "Detected " << nCores << " CPU cores. Enabling ROOT multithreading..." << endl;
     ROOT::EnableImplicitMT(nCores);
     
     // Open the ROOT file using RDataFrame
     string filename = "/home/tomble/epicChargeSharingOutput.root";
     cout << "Opening ROOT file: " << filename << endl;
     
     ROOT::RDataFrame df("Hits", filename);
     
     Long64_t nentries = *df.Count();
     cout << "Found " << nentries << " entries in tree." << endl;
     
     // Define the methods to analyze
     vector<pair<string, string> > methods;
     methods.push_back(make_pair("Digital Readout X", "PixelTrueDeltaX"));
     methods.push_back(make_pair("Digital Readout Y", "PixelTrueDeltaY"));
     methods.push_back(make_pair("Gauss Row X", "GaussRowDeltaX"));
     methods.push_back(make_pair("Gauss Column Y", "GaussColumnDeltaY"));
     methods.push_back(make_pair("Lorentz Row X", "LorentzRowDeltaX"));
     methods.push_back(make_pair("Lorentz Column Y", "LorentzColumnDeltaY"));
     methods.push_back(make_pair("3D Gaussian X", "3DGaussianDeltaX"));
     methods.push_back(make_pair("3D Gaussian Y", "3DGaussianDeltaY"));
     methods.push_back(make_pair("3D Lorentzian X", "3DLorentzianDeltaX"));
     methods.push_back(make_pair("3D Lorentzian Y", "3DLorentzianDeltaY"));
     methods.push_back(make_pair("Gauss Mean Estimator X", "GaussMeanTrueDeltaX"));
     methods.push_back(make_pair("Gauss Mean Estimator Y", "GaussMeanTrueDeltaY"));
     methods.push_back(make_pair("Lorentz Mean Estimator X", "LorentzMeanTrueDeltaX"));
     methods.push_back(make_pair("Lorentz Mean Estimator Y", "LorentzMeanTrueDeltaY"));
     
     cout << "\nStarting parallel analysis of " << methods.size() << " methods..." << endl;
     
     // Store all result pointers for parallel processing
     vector<pair<string, ROOT::RDF::RResultPtr<vector<double> > > > result_ptrs;
     
     // Launch all computations in parallel
     for (size_t i = 0; i < methods.size(); ++i) {
         string method_name = methods[i].first;
         string branch_name = methods[i].second;
         
         cout << "Setting up parallel computation for " << method_name << " (" << branch_name << ")..." << endl;
         
         // Filter out non-finite values and collect data
         auto filtered_data = df.Filter([](double x) { return TMath::Finite(x); }, {branch_name})
                                .Take<double>(branch_name);
         
         result_ptrs.push_back(make_pair(method_name, filtered_data));
     }
     
     cout << "\nAll computations launched. Computing results..." << endl;
     
     vector<ResolutionResult> results;
     
     // Collect results (this triggers the parallel computation)
     for (size_t i = 0; i < result_ptrs.size(); ++i) {
         string method_name = result_ptrs[i].first;
         
         cout << "Processing results for " << method_name << "..." << endl;
         
         try {
             vector<double> data = result_ptrs[i].second.GetValue();
             ResolutionResult result = calculateStatsFromVector(data, method_name);
             
             if (result.n_events > 0) {
                 results.push_back(result);
                 cout << "  ✓ " << method_name << ": σ = " << result.std_dev * 1000 
                      << " μm, bias = " << result.mean_bias * 1000 
                      << " μm, RMS = " << result.rms * 1000 
                      << " μm (n=" << result.n_events << ")" << endl;
             } else {
                 cout << "  ✗ No valid data found for " << method_name << endl;
             }
         } catch (const exception& e) {
             cout << "  ✗ Error processing " << method_name << ": " << e.what() << endl;
         }
     }
     
     // Save results to file
     ofstream outfile("spatial_resolution_results_parallel.txt");
     if (!outfile.is_open()) {
         cout << "Error: Cannot create output file!" << endl;
         return;
     }
     
     cout << "\nSaving results to spatial_resolution_results_parallel.txt..." << endl;
     
     outfile << "Parallel Spatial Resolution Analysis Results" << endl;
     outfile << "============================================" << endl;
     outfile << "Input file: " << filename << endl;
     outfile << "Total entries: " << nentries << endl;
     outfile << "CPU cores used: " << nCores << endl;
     outfile << endl;
     
     outfile << "RESULTS:" << endl;
     outfile << "--------" << endl;
     outfile << setw(30) << left << "Method" 
             << setw(10) << right << "N Events"
             << setw(12) << right << "Bias (μm)"
             << setw(12) << right << "σ (μm)"
             << setw(12) << right << "RMS (μm)" << endl;
     outfile << string(76, '=') << endl;
     
     // Group by category
     map<string, vector<ResolutionResult> > grouped;
     for (size_t i = 0; i < results.size(); ++i) {
         string category = "Other";
         string name = results[i].name;
         
         if (name.find("Digital") != string::npos) category = "Digital";
         else if (name.find("Gauss Row") != string::npos || name.find("Gauss Column") != string::npos) category = "Gauss Row/Col";
         else if (name.find("Lorentz Row") != string::npos || name.find("Lorentz Column") != string::npos) category = "Lorentz Row/Col";
         else if (name.find("3D") != string::npos) category = "3D Surface";
         else if (name.find("Mean Estimator") != string::npos) category = "Mean Estimator";
         
         grouped[category].push_back(results[i]);
     }
     
     // Output by category
     map<string, vector<ResolutionResult> >::iterator it;
     for (it = grouped.begin(); it != grouped.end(); ++it) {
         outfile << endl << it->first << ":" << endl;
         
         for (size_t j = 0; j < it->second.size(); ++j) {
             ResolutionResult& r = it->second[j];
             outfile << setw(30) << left << r.name
                     << setw(10) << right << r.n_events
                     << setw(12) << right << fixed << setprecision(3) << r.mean_bias * 1000
                     << setw(12) << right << setprecision(2) << r.std_dev * 1000
                     << setw(12) << right << setprecision(2) << r.rms * 1000 << endl;
         }
         outfile << string(76, '-') << endl;
     }
     
     // Find best methods
     outfile << endl << "BEST RESOLUTION METHODS:" << endl;
     outfile << "------------------------" << endl;
     
     for (it = grouped.begin(); it != grouped.end(); ++it) {
         if (it->second.empty()) continue;
         
         // Find method with lowest standard deviation
         size_t best_idx = 0;
         for (size_t j = 1; j < it->second.size(); ++j) {
             if (it->second[j].std_dev < it->second[best_idx].std_dev) {
                 best_idx = j;
             }
         }
         
         outfile << it->first << ": " << it->second[best_idx].name 
                 << " (σ = " << fixed << setprecision(2) 
                 << it->second[best_idx].std_dev * 1000 << " μm)" << endl;
     }
     
     outfile.close();
     
     cout << "\nParallel analysis complete!" << endl;
     cout << "Processed " << results.size() << " methods successfully using " << nCores << " cores." << endl;
     cout << "Results saved to spatial_resolution_results_parallel.txt" << endl;
     
     // Disable multithreading
     ROOT::DisableImplicitMT();
 } 