#include "impedance_matching.hpp"
#include <cmath>

namespace np {

ImpedanceMatching::ImpedanceMatching(
    LoadFunction load,
    int order,
    const std::vector<Complex>& transmission_zeros,
    double return_loss_db,
    double freq_left,
    double freq_right,
    double z0
)
    : load_(load)
    , order_(order)
    , tzs_(transmission_zeros)
    , return_loss_(return_loss_db)
    , freq_left_(freq_left)
    , freq_right_(freq_right)
    , z0_(z0)
{
    // Generate Chebyshev node frequencies for interpolation
    freqs_ = chebyshev_nodes(freq_left_, freq_right_, order_);

    // Compute target load reflection coefficients at interpolation frequencies
    target_loads_.resize(order_);
    for (int i = 0; i < order_; ++i) {
        Complex z_l = load_(freqs_[i]);
        target_loads_[i] = impedance_to_gamma(z_l);
    }
}

Complex ImpedanceMatching::impedance_to_gamma(Complex z_l) const {
    // Reflection coefficient: Γ = (Z_L - Z_0) / (Z_L + Z_0)
    return (z_l - z0_) / (z_l + z0_);
}

MatrixXcd ImpedanceMatching::run() {
    // Create normalized Nevanlinna-Pick problem
    NevanlinnaPickNormalized np(target_loads_, freqs_, tzs_, return_loss_);

    // Create path tracker
    PathTracker tracker(&np);
    tracker.h = path_tracker_h;
    tracker.run_tol = path_tracker_run_tol;
    tracker.final_tol = path_tracker_final_tol;

    // Get starting solution and track path
    VectorXcd x0 = np.get_start_solution();
    VectorXcd solution = tracker.run(x0);

    // Extract coupling matrix from solution
    return np.calc_coupling_matrix(solution);
}

} // namespace np
