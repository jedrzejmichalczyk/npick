#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include "chebyshev_filter.hpp"
#include "coupling_matrix.hpp"
#include "nevanlinna_pick.hpp"
#include "homotopy/path_tracker.hpp"
#include <functional>
#include <vector>

namespace np {

/**
 * High-level interface for impedance matching network synthesis.
 *
 * Given a complex load impedance function Z_L(ω), synthesizes a matching
 * network (as a coupling matrix) that achieves optimal reflection coefficient
 * across the specified frequency band.
 *
 * The algorithm uses Nevanlinna-Pick interpolation with homotopy continuation
 * to find a Schur function that interpolates the target load reflection
 * coefficients at Chebyshev node frequencies.
 */
class ImpedanceMatching {
public:
    using LoadFunction = std::function<Complex(double freq)>;

    /**
     * Construct an impedance matching problem.
     *
     * @param load Function returning complex load impedance Z_L at frequency ω
     * @param order Filter order (number of resonators)
     * @param transmission_zeros Finite transmission zeros (empty for all-pole)
     * @param return_loss_db Initial return loss specification (dB)
     * @param freq_left Left edge of passband
     * @param freq_right Right edge of passband
     * @param z0 Reference impedance (default 50 Ω)
     */
    ImpedanceMatching(
        LoadFunction load,
        int order,
        const std::vector<Complex>& transmission_zeros,
        double return_loss_db,
        double freq_left,
        double freq_right,
        double z0 = 50.0
    );

    /**
     * Run the impedance matching optimization.
     *
     * @return Coupling matrix of the synthesized matching network
     */
    MatrixXcd run();

    /**
     * Get the interpolation frequencies (Chebyshev nodes).
     */
    const std::vector<double>& interpolation_freqs() const { return freqs_; }

    /**
     * Get the target load reflection coefficients at interpolation frequencies.
     */
    const std::vector<Complex>& target_loads() const { return target_loads_; }

    // Configuration
    double path_tracker_h = -0.05;      // Initial step size
    double path_tracker_run_tol = 1e-3;  // Tolerance during tracking
    double path_tracker_final_tol = 1e-7; // Final tolerance

private:
    /**
     * Convert load impedance to reflection coefficient.
     * Γ = (Z_L - Z_0) / (Z_L + Z_0)
     */
    Complex impedance_to_gamma(Complex z_l) const;

    LoadFunction load_;
    int order_;
    std::vector<Complex> tzs_;
    double return_loss_;
    double freq_left_;
    double freq_right_;
    double z0_;
    std::vector<double> freqs_;
    std::vector<Complex> target_loads_;
};

} // namespace np
