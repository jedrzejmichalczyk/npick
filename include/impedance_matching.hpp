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
 * 2. An exchange loop that moves interpolation frequencies toward the
 *    worst-case G11 peaks, with an optional short Nelder-Mead polish
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
     * Uses an exchange loop to update interpolation frequencies toward
     * the worst-case G11 peaks. An optional short Nelder-Mead polish can
     * be enabled afterward by setting optimizer_max_iterations > 0.
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
     * Run equiripple Newton optimizer.
     *
     * Solves for interpolation frequencies where all G11 peaks between
     * consecutive interpolation points are equal (equioscillation).
     * Uses Newton's method on the peak-difference system with
     * finite-difference Jacobian.
     *
     * @param initial_freqs Starting interpolation frequencies
     * @return Coupling matrix (or empty if failed)
     */
    MatrixXcd run_equiripple(const std::vector<double>& initial_freqs);

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

    // Optimizer configuration (Nelder-Mead on interpolation frequencies)
    double optimizer_initial_step = 0.02;
    double optimizer_tolerance = 1e-4;
    int optimizer_max_iterations = 50;
    int optimizer_iteration_cap = 24;
    int cost_eval_points = 41;  // Points for max|G11| evaluation
    int cost_peak_candidates = 0;  // 0 => derive from problem order
    int cost_peak_refine_iterations = 4;
    int cost_peak_refine_subdivisions = 8;
    bool use_symmetric_parameterization = true;
    bool fix_symmetric_band_edges = true;
    double symmetry_detection_tolerance = 1e-6;

    // Equiripple Newton optimizer
    int equiripple_max_iterations = 15;
    double equiripple_fd_delta = 1e-5;
    int equiripple_peak_eval_points = 201;
    int equiripple_peak_refine_steps = 6;

    // Exchange algorithm (experimental, disabled by default)
    int exchange_iterations = 0;
    int exchange_eval_points = 161;
    double exchange_relaxation = 1.0;

    // Verbosity
    bool verbose = false;

private:
    std::vector<double> canonicalize_interpolation_freqs(const std::vector<double>& interp_freqs) const;
    bool supports_symmetric_optimization() const;
    std::vector<double> compress_symmetric_freqs(const std::vector<double>& interp_freqs) const;
    std::vector<double> expand_symmetric_freqs(const std::vector<double>& positive_freqs) const;
    std::vector<double> select_exchange_freqs(const MatrixXcd& cm, bool symmetric) const;
    double compute_response_cost(const MatrixXcd& cm) const;
    double refine_peak_magnitude(const MatrixXcd& cm, double left, double right) const;

    std::vector<double> find_interval_peaks(const MatrixXcd& cm,
                                            const std::vector<double>& interp_freqs) const;

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
