#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include "spectral_factor.hpp"
#include "chebyshev_filter.hpp"
#include "coupling_matrix.hpp"
#include "homotopy/homotopy_base.hpp"
#include <vector>
#include <functional>

namespace np {

/**
 * Nevanlinna-Pick interpolation solver for filter matching problems.
 *
 * Based on: Baratchart, Olivi, Seyfert - "Generalized Nevanlinna-Pick interpolation
 * on the boundary. Application to impedance matching" (2015)
 *
 * Solves for a Schur function p/q (where q is the spectral factor from |q|²=|p|²+|r|²)
 * that interpolates given values at specified frequencies.
 */
class NevanlinnaPick : public HomotopyBase {
public:
    /**
     * Construct a Nevanlinna-Pick interpolation problem.
     *
     * @param loads Target S11 values (complex load reflection coefficients)
     * @param freqs Interpolation frequencies (normalized)
     * @param transmission_zeros Finite transmission zeros
     * @param return_loss Initial return loss specification (dB)
     */
    NevanlinnaPick(
        const std::vector<Complex>& loads,
        const std::vector<double>& freqs,
        const std::vector<Complex>& transmission_zeros,
        double return_loss
    );

    ~NevanlinnaPick() override;

    // HomotopyBase interface
    VectorXcd calc_homotopy_function(const VectorXcd& x, double t) override;
    VectorXcd calc_path_derivative(const VectorXcd& x, double t) override;
    MatrixXcd calc_jacobian(const VectorXcd& x, double t) override;
    VectorXcd get_start_solution() override;
    int get_num_variables() override;

    /**
     * Evaluate the map ψ(p) = p/q at all interpolation frequencies.
     * Here q is the spectral factor satisfying |q|² = |p|² + |r|².
     */
    virtual VectorXcd eval_map(const VectorXcd& p_coeffs) const;

    /**
     * Compute the Jacobian of eval_map with respect to polynomial coefficients.
     * Uses the Bezout equation to analytically compute ∂q/∂p_j.
     */
    virtual MatrixXcd eval_grad(const VectorXcd& p_coeffs) const;

    // Access initial polynomial and parameters
    const VectorXcd& initial_p() const { return initial_p_; }
    const VectorXcd& r_coeffs() const { return r_coeffs_; }
    const std::vector<double>& frequencies() const { return freqs_; }
    double shift() const { return shift_; }

protected:
    /**
     * Solve the Bezout equation: dq·q* + q·dq* = rhs
     *
     * This finds dq given q and the right-hand side, with null-space
     * projection to preserve monic normalization.
     */
    static Polynomial<Complex> solve_bezout(
        const Polynomial<Complex>& q,
        const Polynomial<Complex>& rhs,
        int max_degree
    );

    /**
     * Solve Bezout equation when q has real coefficients.
     * Constrains dq to also have real coefficients.
     */
    static Polynomial<Complex> solve_bezout_real(
        const Polynomial<Complex>& q,
        const Polynomial<Complex>& q_para,
        const Polynomial<Complex>& rhs,
        int max_degree
    );

    /**
     * Build polynomial from coefficient vector (descending order).
     * Coefficient vector: [p_{n-1}, p_{n-2}, ..., p_1, p_0]
     */
    static Polynomial<Complex> build_polynomial(const VectorXcd& coeffs);

    /**
     * Get coefficient vector from polynomial (descending order).
     */
    static VectorXcd get_coeffs(const Polynomial<Complex>& p);

    VectorXcd initial_p_;      // Initial polynomial coefficients (descending)
    VectorXcd r_coeffs_;       // Fixed transmission polynomial coefficients
    std::vector<double> freqs_; // Interpolation frequencies (normalized)
    double shift_;             // Frequency normalization shift
    int order_;                // Filter order

    VectorXcd target_loads_;   // Target interpolation values
    VectorXcd init_sparams_;   // Initial S-parameters at interpolation points

    // Spectral factor cache: avoids recomputing feldtkeller when eval_map
    // and eval_grad are called with the same p_coeffs (common in corrector).
    mutable VectorXcd cached_p_coeffs_;
    mutable Polynomial<Complex> cached_q_poly_;
    mutable bool cache_valid_ = false;

    Polynomial<Complex> cached_feldtkeller(
        const Polynomial<Complex>& p_poly,
        const Polynomial<Complex>& r_poly,
        const VectorXcd& p_coeffs) const;
};

/**
 * Normalized version of NevanlinnaPick with monic constraint.
 *
 * Reduces the number of variables by 1 by fixing the leading
 * coefficient of p to 1 (monic polynomial).
 */
class NevanlinnaPickNormalized : public NevanlinnaPick {
public:
    using NevanlinnaPick::NevanlinnaPick;

    VectorXcd eval_map(const VectorXcd& x) const override;
    MatrixXcd eval_grad(const VectorXcd& x) const override;
    VectorXcd get_start_solution() override;
    int get_num_variables() override;

    /**
     * Convert reduced coefficients to monic polynomial coefficients.
     * Input: [p_{n-2}, ..., p_0] (n-1 coefficients)
     * Output: [1, p_{n-2}, ..., p_0] (n coefficients)
     */
    static VectorXcd to_monic(const VectorXcd& x);

    /**
     * Compute coupling matrix from the solution.
     */
    MatrixXcd calc_coupling_matrix(const VectorXcd& x) const;
};

} // namespace np
