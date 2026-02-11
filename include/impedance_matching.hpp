#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include "chebyshev_filter.hpp"
#include "coupling_matrix.hpp"
#include "nevanlinna_pick.hpp"
#include "homotopy/path_tracker.hpp"
#include "optimization/nelder_mead.hpp"
#include <functional>
#include <vector>

namespace np {

/**
 * High-level interface for impedance matching network synthesis.
 *
 * Given a complex load reflection function L(ω), synthesizes a matching
 * network (as a coupling matrix) that minimizes the worst-case matched
 * reflection coefficient G11 across the specified frequency band.
 *
 * The algorithm uses:
 * 1. Nevanlinna-Pick interpolation with homotopy continuation to find
 *    a Schur function matching S11 = conj(L) at interpolation frequencies
 * 2. Nelder-Mead optimization to find the optimal interpolation frequencies
 *    that minimize max|G11| over the passband (MinMax problem)
 *
 * G11 = S11 + S12*S21*L / (1 - S22*L) is the overall reflection when
 * the matching network is connected to the load.
 */
class ImpedanceMatching {
public:
    using LoadFunction = std::function<Complex(double freq)>;

    /**
     * Construct an impedance matching problem.
     *
     * @param load Function returning complex load reflection L at frequency ω
     * @param order Filter order (number of resonators)
     * @param transmission_zeros Finite transmission zeros (empty for all-pole)
     * @param return_loss_db Return loss specification (dB)
     * @param freq_left Left edge of passband (normalized)
     * @param freq_right Right edge of passband (normalized)
     */
    ImpedanceMatching(
        LoadFunction load,
        int order,
        const std::vector<Complex>& transmission_zeros,
        double return_loss_db,
        double freq_left,
        double freq_right
    );

    /**
     * Run the impedance matching optimization.
     *
     * Uses Nelder-Mead to find optimal interpolation frequencies that
     * minimize max|G11| over the passband.
     *
     * @return Coupling matrix of the synthesized matching network
     */
    MatrixXcd run();

    /**
     * Run single NP iteration with given interpolation frequencies.
     *
     * @param interp_freqs Interpolation frequencies
     * @return Coupling matrix (or empty if failed)
     */
    MatrixXcd run_single(const std::vector<double>& interp_freqs);

    /**
     * Compute max|G11| cost for given interpolation frequencies.
     *
     * @param interp_freqs Interpolation frequencies
     * @return Maximum |G11| over passband (or large value if failed)
     */
    double compute_cost(const std::vector<double>& interp_freqs);

    /**
     * Evaluate G11 (matched reflection) at a frequency.
     *
     * @param cm Coupling matrix
     * @param freq Frequency
     * @return G11 = S11 + S12*S21*L / (1 - S22*L)
     */
    Complex eval_G11(const MatrixXcd& cm, double freq) const;

    /**
     * Get the optimized interpolation frequencies.
     */
    const std::vector<double>& interpolation_freqs() const { return freqs_; }

    /**
     * Get the final coupling matrix.
     */
    const MatrixXcd& coupling_matrix() const { return cm_; }

    /**
     * Get the achieved max|G11| in dB.
     */
    double achieved_return_loss_db() const { return achieved_rl_db_; }

    // Path tracker configuration
    double path_tracker_h = -0.05;
    double path_tracker_run_tol = 1e-3;
    double path_tracker_final_tol = 1e-7;

    // Optimizer configuration
    double optimizer_tolerance = 1e-4;
    int optimizer_max_iterations = 500;
    int cost_eval_points = 101;  // Points for max|G11| evaluation

    // Verbosity
    bool verbose = false;

private:
    LoadFunction load_;
    int order_;
    std::vector<Complex> tzs_;
    double return_loss_;
    double freq_left_;
    double freq_right_;

    // Results
    std::vector<double> freqs_;
    MatrixXcd cm_;
    double achieved_rl_db_;
};

} // namespace np
