/// @file GaussianFitter.hh
/// @brief Compiled Gaussian fitting routines for position reconstruction.
///
/// This file provides compiled implementations of FitGaussian1D and FitGaussian2D,
/// replacing the interpreted ROOT macros for better performance.
/// Configuration is read from Config.hh at compile time.

#ifndef ECS_GAUSSIAN_FITTER_HH
#define ECS_GAUSSIAN_FITTER_HH

#include <string>

namespace ECS::Fit {

/// @brief Perform 1D Gaussian fits on central row and column of charge neighborhood.
///
/// Fits Gaussians to the central row and column slices of the charge distribution
/// to reconstruct hit positions. Appends fit results as new branches to the ROOT file.
///
/// @param filename Path to the ROOT file containing the Hits tree.
/// @return 0 on success, non-zero error code on failure.
int FitGaussian1D(const char* filename);

/// @brief Perform 2D Gaussian fit on the full charge neighborhood.
///
/// Fits a 2D Gaussian to the charge distribution to reconstruct hit positions.
/// Appends fit results as new branches to the ROOT file.
///
/// @param filename Path to the ROOT file containing the Hits tree.
/// @return 0 on success, non-zero error code on failure.
int FitGaussian2D(const char* filename);

/// @brief Run all enabled fitting routines.
///
/// Executes FitGaussian1D and/or FitGaussian2D based on Config.hh settings.
///
/// @param filename Path to the ROOT file containing the Hits tree.
/// @return true if any fitting was performed, false otherwise.
bool RunAllFits(const std::string& filename);

} // namespace ECS::Fit

#endif // ECS_GAUSSIAN_FITTER_HH
