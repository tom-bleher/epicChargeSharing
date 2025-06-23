#ifndef CERESUTILS_HH
#define CERESUTILS_HH

#include "ceres/ceres.h"
#include <vector>

// Solver configuration presets
enum class SolverPreset {
    FAST,     // DENSE_QR + LM, tol 1e-12, 400 iter
    FALLBACK  // DENSE_NORMAL_CHOLESKY + LM, tol 1e-14, 1200 iter
};

// Create solver options based on preset
ceres::Solver::Options MakeSolverOptions(SolverPreset preset, double pixel_spacing);

// Set standard parameter bounds for amplitude, center, width, baseline
void SetStandardBounds(ceres::Problem& problem, double* parameters, 
                      double charge_range, double pixel_spacing,
                      double center_x, double center_y = 0.0, // center_y optional for 1D fits
                      bool is_3d = false, bool has_beta = false);

#endif // CERESUTILS_HH 