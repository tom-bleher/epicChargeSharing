#include "StatsUtils.hh"
#include "Constants.hh"
#include <algorithm>
#include <numeric>
#include <iostream>

RobustStats1D CalculateRobustStats1D(const std::vector<double>& x_vals, 
                                     const std::vector<double>& y_vals) {
    RobustStats1D stats;
    
    if (x_vals.size() != y_vals.size() || x_vals.empty()) {
        return stats;
    }
    
    // Basic statistics - single pass for min/max
    auto minmax_result = std::minmax_element(y_vals.begin(), y_vals.end());
    stats.min_val = *minmax_result.first;
    stats.max_val = *minmax_result.second;
    
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
    
    // Fast median-of-absolute-deviations using nth_element
    std::vector<double> abs_deviations;
    abs_deviations.reserve(y_vals.size());
    for (double val : y_vals) {
        abs_deviations.push_back(std::abs(val - stats.median));
    }

    std::nth_element(abs_deviations.begin(), abs_deviations.begin() + n/2, abs_deviations.end());
    double mad_raw = abs_deviations[n/2];

    if (n % 2 == 0) {
        std::nth_element(abs_deviations.begin(), abs_deviations.begin() + n/2 - 1, abs_deviations.end());
        double second_val = abs_deviations[n/2 - 1];
        mad_raw = 0.5 * (mad_raw + second_val);
    }

    stats.mad = mad_raw * 1.4826; // Consistency factor for normal distribution
    
    // Numerical stability safeguard
    if (!std::isfinite(stats.mad) || stats.mad < 1e-12) {
        stats.mad = (std::isfinite(stats.std_dev) && stats.std_dev > 1e-12) ?
                    stats.std_dev : 1e-12;
    }
    
    // Weighted statistics (weight by charge value for position estimation)
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

RobustStats3D CalculateRobustStats3D(const std::vector<double>& x_vals,
                                     const std::vector<double>& y_vals,
                                     const std::vector<double>& z_vals) {
    RobustStats3D stats;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.empty()) {
        return stats;
    }
    
    // Basic statistics on Z values (charge)
    auto minmax_result = std::minmax_element(z_vals.begin(), z_vals.end());
    stats.min_val_z = *minmax_result.first;
    stats.max_val_z = *minmax_result.second;
    
    // Mean and standard deviation
    stats.mean_z = std::accumulate(z_vals.begin(), z_vals.end(), 0.0) / z_vals.size();
    
    double variance = 0.0;
    for (double val : z_vals) {
        variance += (val - stats.mean_z) * (val - stats.mean_z);
    }
    stats.std_dev_z = std::sqrt(variance / z_vals.size());
    
    // Median and quartiles
    std::vector<double> sorted_z = z_vals;
    std::sort(sorted_z.begin(), sorted_z.end());
    
    size_t n = sorted_z.size();
    if (n % 2 == 0) {
        stats.median_z = (sorted_z[n/2 - 1] + sorted_z[n/2]) / 2.0;
    } else {
        stats.median_z = sorted_z[n/2];
    }
    
    stats.q25_z = sorted_z[n/4];
    stats.q75_z = sorted_z[3*n/4];
    
    // Median Absolute Deviation
    std::vector<double> abs_deviations;
    abs_deviations.reserve(z_vals.size());
    for (double val : z_vals) {
        abs_deviations.push_back(std::abs(val - stats.median_z));
    }
    std::sort(abs_deviations.begin(), abs_deviations.end());
    stats.mad_z = abs_deviations[n/2] * 1.4826;
    
    // Numerical stability safeguard
    if (!std::isfinite(stats.mad_z) || stats.mad_z < 1e-12) {
        stats.mad_z = (std::isfinite(stats.std_dev_z) && stats.std_dev_z > 1e-12) ?
                      stats.std_dev_z : 1e-12;
    }
    
    // Weighted statistics for center estimation (weight by charge value)
    stats.weighted_mean_x = 0.0;
    stats.weighted_mean_y = 0.0;
    stats.total_weight = 0.0;
    
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double weight = std::max(0.0, z_vals[i] - stats.q25_z);
        if (weight > 0) {
            stats.weighted_mean_x += x_vals[i] * weight;
            stats.weighted_mean_y += y_vals[i] * weight;
            stats.total_weight += weight;
        }
    }
    
    if (stats.total_weight > 0) {
        stats.weighted_mean_x /= stats.total_weight;
        stats.weighted_mean_y /= stats.total_weight;
    } else {
        stats.weighted_mean_x = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
        stats.weighted_mean_y = std::accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();
    }
    
    stats.valid = true;
    return stats;
}

std::pair<std::vector<double>, std::vector<double>> FilterOutliersMad1D(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    double sigma_threshold,
    bool verbose) {
    
    std::vector<double> filtered_x, filtered_y;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() < 4) {
        return std::make_pair(x_vals, y_vals);
    }
    
    RobustStats1D stats = CalculateRobustStats1D(x_vals, y_vals);
    if (!stats.valid) {
        return std::make_pair(x_vals, y_vals);
    }
    
    // Use MAD-based outlier detection
    double outlier_threshold = stats.median + sigma_threshold * stats.mad;
    double lower_threshold = stats.median - sigma_threshold * stats.mad;
    
    int outliers_removed = 0;
    for (size_t i = 0; i < y_vals.size(); ++i) {
        if (y_vals[i] >= lower_threshold && y_vals[i] <= outlier_threshold) {
            filtered_x.push_back(x_vals[i]);
            filtered_y.push_back(y_vals[i]);
        } else {
            outliers_removed++;
        }
    }
    
    // If too many outliers were removed, relax to 4.0 sigma
    if (filtered_x.size() < x_vals.size() / 2) {
        if (verbose) {
            std::cout << "Too many outliers detected (" << outliers_removed 
                     << "), relaxing to 4.0 sigma" << std::endl;
        }
        
        filtered_x.clear();
        filtered_y.clear();
        
        double extreme_threshold = stats.median + 4.0 * stats.mad;
        double extreme_lower = stats.median - 4.0 * stats.mad;
        
        for (size_t i = 0; i < y_vals.size(); ++i) {
            if (y_vals[i] >= extreme_lower && y_vals[i] <= extreme_threshold) {
                filtered_x.push_back(x_vals[i]);
                filtered_y.push_back(y_vals[i]);
            }
        }
    }
    
    // Ensure we have enough points for fitting
    if (filtered_x.size() < 4) {
        if (verbose) {
            std::cout << "Warning: After outlier filtering, only " << filtered_x.size() 
                     << " points remain, returning original data" << std::endl;
        }
        return std::make_pair(x_vals, y_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_pair(filtered_x, filtered_y);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> FilterOutliersMad3D(
    const std::vector<double>& x_vals,
    const std::vector<double>& y_vals,
    const std::vector<double>& z_vals,
    double sigma_threshold,
    bool verbose) {
    
    std::vector<double> filtered_x, filtered_y, filtered_z;
    
    if (x_vals.size() != y_vals.size() || x_vals.size() != z_vals.size() || x_vals.size() < 6) {
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    RobustStats3D stats = CalculateRobustStats3D(x_vals, y_vals, z_vals);
    if (!stats.valid) {
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    // Use MAD-based outlier detection
    double outlier_threshold = stats.median_z + sigma_threshold * stats.mad_z;
    double lower_threshold = stats.median_z - sigma_threshold * stats.mad_z;
    
    int outliers_removed = 0;
    for (size_t i = 0; i < z_vals.size(); ++i) {
        if (z_vals[i] >= lower_threshold && z_vals[i] <= outlier_threshold) {
            filtered_x.push_back(x_vals[i]);
            filtered_y.push_back(y_vals[i]);
            filtered_z.push_back(z_vals[i]);
        } else {
            outliers_removed++;
        }
    }
    
    // If too many outliers were removed, relax to 4.0 sigma
    if (filtered_x.size() < x_vals.size() / 2) {
        if (verbose) {
            std::cout << "Too many 3D outliers detected (" << outliers_removed 
                     << "), relaxing to 4.0 sigma" << std::endl;
        }
        
        filtered_x.clear();
        filtered_y.clear();
        filtered_z.clear();
        
        double extreme_threshold = stats.median_z + 4.0 * stats.mad_z;
        double extreme_lower = stats.median_z - 4.0 * stats.mad_z;
        
        for (size_t i = 0; i < z_vals.size(); ++i) {
            if (z_vals[i] >= extreme_lower && z_vals[i] <= extreme_threshold) {
                filtered_x.push_back(x_vals[i]);
                filtered_y.push_back(y_vals[i]);
                filtered_z.push_back(z_vals[i]);
            }
        }
    }
    
    // Ensure we have enough points for fitting
    if (filtered_x.size() < 6) {
        if (verbose) {
            std::cout << "Warning: After 3D outlier filtering, only " << filtered_x.size() 
                     << " points remain, returning original data" << std::endl;
        }
        return std::make_tuple(x_vals, y_vals, z_vals);
    }
    
    if (verbose && outliers_removed > 0) {
        std::cout << "Removed " << outliers_removed << " 3D outliers, " 
                 << filtered_x.size() << " points remaining" << std::endl;
    }
    
    return std::make_tuple(filtered_x, filtered_y, filtered_z);
} 