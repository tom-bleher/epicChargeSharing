#ifndef STATSUTILS_HH
#define STATSUTILS_HH

#include <vector>
#include <cmath>

// Robust statistics for 1D data
struct RobustStats1D {
    double mean;
    double median;
    double std_dev;
    double mad;  // Median Absolute Deviation
    double q25, q75;  // Quartiles
    double min_val, max_val;
    double weighted_mean;
    double total_weight;
    bool valid;
    
    RobustStats1D() : mean(0), median(0), std_dev(0), mad(0), q25(0), q75(0), 
                      min_val(0), max_val(0), weighted_mean(0), total_weight(0), valid(false) {}
};

// Robust statistics for 3D data (x, y, z coordinates)
struct RobustStats3D {
    double mean_z;
    double median_z;
    double std_dev_z;
    double mad_z;  // Median Absolute Deviation for Z
    double q25_z, q75_z;  // Quartiles for Z
    double min_val_z, max_val_z;
    double weighted_mean_x;
    double weighted_mean_y;
    double total_weight;
    bool valid;
    
    RobustStats3D() : mean_z(0), median_z(0), std_dev_z(0), mad_z(0), q25_z(0), q75_z(0),
                      min_val_z(0), max_val_z(0), weighted_mean_x(0), weighted_mean_y(0), 
                      total_weight(0), valid(false) {}
};

// Calculate robust statistics for 1D data
RobustStats1D CalcRobustStats1D(const std::vector<double>& x_vals, 
                                     const std::vector<double>& y_vals);

// Calculate robust statistics for 3D data
RobustStats3D CalcRobustStats3D(const std::vector<double>& x_vals,
                                     const std::vector<double>& y_vals,
                                     const std::vector<double>& z_vals);

// MAD-based outlier filtering for 1D data
std::pair<std::vector<double>, std::vector<double>> FilterOutliersMad1D(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double sigma_threshold = 3.0,
    bool verbose = false);

// MAD-based outlier filtering for 3D data
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> FilterOutliersMad3D(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double sigma_threshold = 3.0,
    bool verbose = false);

#endif // STATSUTILS_HH 