#pragma once

#include "../types.hpp"
#include "../polynomial.hpp"
#include "../spectral_factor.hpp"
#include "../chebyshev_filter.hpp"
#include "../coupling_matrix.hpp"
#include "../homotopy/homotopy_base.hpp"
#include "manifold.hpp"
#include <vector>
#include <memory>

namespace np {

/**
 * Coupled Nevanlinna-Pick homotopy problem for multiplexer synthesis.
 *
 * Implements the simultaneous matching from Martinez et al. (EuMC 2019):
 * Find polynomials P = [p_1, ..., p_N] such that each filter's reflection
 * matches the coupled load seen through the manifold.
 *
 * Variables: x = [p_1_coeffs, p_2_coeffs, ..., p_N_coeffs] (monic, concatenated)
 * Homotopy: lambda = 1 - t maps PathTracker's t in [1,0] to lambda in [0,1]
 *   At lambda=0: decoupled (each filter matches J_ii independently)
 *   At lambda=1: fully coupled (filters interact through manifold)
 */
class MultiplexerNevanlinnaPick : public HomotopyBase {
public:
    MultiplexerNevanlinnaPick(
        const Manifold& manifold,
        const std::vector<ChannelSpec>& specs,
        const std::vector<std::vector<double>>& interp_freqs
    );

    ~MultiplexerNevanlinnaPick() override;

    // HomotopyBase interface
    VectorXcd calc_homotopy_function(const VectorXcd& x, double t) override;
    VectorXcd calc_path_derivative(const VectorXcd& x, double t) override;
    MatrixXcd calc_jacobian(const VectorXcd& x, double t) override;
    VectorXcd get_start_solution() override;
    int get_num_variables() override;

    /**
     * Extract per-channel coupling matrices from the converged solution.
     */
    std::vector<MatrixXcd> extract_coupling_matrices(const VectorXcd& x) const;

    /**
     * Run the Martinez continuation: Euler predictor + Newton corrector.
     * Ramps lambda from 0 to 1, starting from the independent solutions in x.
     * Returns true if lambda=1 is reached.
     */
    bool run_continuation(VectorXcd& x, double lambda_step = 0.05,
                          int max_steps = 200, int newton_max_iter = 20,
                          double newton_tol = 1e-9, bool verbose = false);

    /**
     * Compute residual F(P, lambda) = f(p_i) - L_i(P, lambda) for all channels.
     */
    VectorXcd compute_residual(const VectorXcd& x, double lambda) const;

    /**
     * Compute Jacobian dF/dp at given (x, lambda) — the Wirtinger
     * holomorphic part ∂F/∂p only. For full Newton on complex p use
     * compute_jacobians_wirtinger() or newton_step().
     */
    MatrixXcd compute_jacobian_dp(const VectorXcd& x, double lambda) const;

    /**
     * Compute BOTH Wirtinger Jacobians of F w.r.t. p's complex coefficients:
     *   Jac_A(k, v) = ∂F_k/∂p_v        (holomorphic)
     *   Jac_B(k, v) = ∂F_k/∂conj(p_v)  (antiholomorphic)
     * The spectral factor q depends on |p|² via Feldtkeller, so ∂F/∂conj(p)
     * is generically nonzero — Newton in complex p must use BOTH pieces.
     */
    void compute_jacobians_wirtinger(const VectorXcd& x, double lambda,
                                     MatrixXcd& Jac_A, MatrixXcd& Jac_B) const;

    /**
     * One real-variable Newton step: build the 2M x 2M real Jacobian from
     * the Wirtinger pair (A, B), solve J_real · [Re(dx); Im(dx)] = -[Re(F); Im(F)],
     * and return dx packed as a complex M-vector. Caller picks the step size.
     */
    VectorXcd newton_step(const VectorXcd& x, double lambda,
                          const VectorXcd& F) const;

    /**
     * Compute dF/dlambda at given (x, lambda).
     */
    VectorXcd compute_dF_dlambda(const VectorXcd& x, double lambda) const;

    /**
     * Frequency-shift continuation: move interpolation frequencies from
     * current to target while maintaining F(P) = 0.
     *
     * Uses the same Euler+Newton machinery as run_continuation but
     * blends interpolation frequencies instead of coupling lambda.
     * Start: current interp_freqs (x already satisfies F=0 at lambda=1).
     * End: target_freqs.
     *
     * Returns true if converged. Updates internal frequencies on success.
     */
    bool shift_frequencies(VectorXcd& x,
                           const std::vector<std::vector<double>>& target_freqs,
                           double step = 0.1, int max_steps = 100,
                           int newton_max = 15, double newton_tol = 1e-7,
                           bool verbose = false);

    /**
     * Get current interpolation frequencies (may have been shifted).
     */
    const std::vector<double>& channel_interp_freqs(int ch) const {
        return channels_[ch].interp_freqs;
    }

    int num_channels() const { return num_channels_; }

private:
    // Per-channel data
    struct ChannelData {
        int order;
        int num_vars;     // order (monic: order+1 coeffs minus leading 1)
        int var_offset;   // offset into concatenated x
        int eq_offset;    // offset into concatenated H
        std::vector<double> interp_freqs;   // physical frequencies
        std::vector<double> norm_freqs;     // normalized to [-1, 1]
        double freq_center, freq_scale;     // normalization params
        VectorXcd r_coeffs;                 // transmission polynomial (descending)
        VectorXcd initial_p;                // Chebyshev prototype (monic, reduced)
        VectorXcd init_sparams;             // f(chebyshev_i) at interp freqs
    };

    int num_channels_;
    int total_vars_;
    int total_eqs_;
    const Manifold* manifold_;
    std::vector<ChannelData> channels_;

    // Convert monic reduced coefficients to full descending polynomial
    static VectorXcd to_monic(const VectorXcd& x);
    static Polynomial<Complex> build_polynomial(const VectorXcd& coeffs);
    static VectorXcd get_coeffs(const Polynomial<Complex>& p);

    // Per-channel evaluation helpers
    VectorXcd eval_channel_map(int ch, const VectorXcd& p_coeffs) const;
    MatrixXcd eval_channel_grad(int ch, const VectorXcd& p_coeffs) const;

    // Compute coupled load L_i(P, lambda) at a single frequency
    Complex eval_coupled_load(int ch, double freq_phys, double lambda,
                              const std::vector<VectorXcd>& all_p_coeffs) const;

    // Extract per-channel p_coeffs from concatenated x (monic form)
    std::vector<VectorXcd> split_variables(const VectorXcd& x) const;
};

} // namespace np
