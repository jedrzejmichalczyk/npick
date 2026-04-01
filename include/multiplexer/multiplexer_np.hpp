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

    int num_channels() const { return num_channels_; }

private:
    // Per-channel data
    struct ChannelData {
        int order;
        int num_vars;     // order - 1 (monic)
        int var_offset;   // offset into concatenated x
        int eq_offset;    // offset into concatenated H
        std::vector<double> interp_freqs;   // physical frequencies
        std::vector<double> norm_freqs;     // normalized to [-1, 1]
        double freq_center, freq_scale;     // normalization params
        VectorXcd r_coeffs;                 // transmission polynomial (descending)
        VectorXcd initial_p;                // start solution from independent NP
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
