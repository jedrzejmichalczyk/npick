#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace np {

/**
 * Spectral factorization solver.
 *
 * Given a polynomial qq* (para-Hermitian symmetric), finds the minimum-phase
 * factor q such that q * q.para_conjugate() = qq*
 *
 * The minimum-phase factor has all its roots in the left half-plane (Re < 0).
 *
 * Two implementations are available:
 *  - factorize(): Wilson's Newton iteration on the factor coefficients.
 *    Root-free, quadratically convergent, bit-reproducible across toolchains.
 *    See Wilson 1969 (SIAM J. Numer. Anal. 6:1) and the Sayed-Kailath survey
 *    (Numer. Lin. Alg. Appl. 8:467, 2001).
 *  - factorize_by_roots(): legacy companion-matrix root split. Kept for
 *    fallback / testing; ill-conditioned when roots cluster near jω.
 */
class SpectralFactor {
public:
    /**
     * Factorize a para-Hermitian polynomial into its minimum-phase factor
     * using Wilson's iterative algorithm.
     *
     * Algorithm: starting from an initial Hurwitz guess q_0, iterate the
     * Newton step on F(q) = q·q* − Q:
     *    Solve  q_k · Δq* + Δq · q_k* = Q − q_k · q_k*   for Δq,
     *    then   q_{k+1} = q_k + Δq.
     * The Sylvester-like linear system is 2n+1 real equations in 2n+1 real
     * unknowns once the gauge Im(q_n) = 0 is fixed. Convergence is quadratic
     * and requires no polynomial root-finding.
     *
     * @param qq_star The input para-Hermitian polynomial (degree 2n, even)
     * @return The minimum-phase factor q of degree n
     * @throws SpectralFactorError on invalid input or non-convergence
     */
    static Polynomial<Complex> factorize(const Polynomial<Complex>& qq_star) {
        int deg = qq_star.degree();
        if (deg < 0) {
            throw SpectralFactorError("Spectral factorization: zero polynomial");
        }
        if (deg % 2 != 0) {
            throw SpectralFactorError(
                "Spectral factorization: odd total degree (" +
                std::to_string(deg) + "). Polynomial is not para-Hermitian.");
        }
        int n = deg / 2;

        // Leading coefficient of q·q* = (-1)^n · |a|² where a = q_n.
        // Para-Hermitian Q_{2n} is real. Pick a > 0 so q_n > 0 (gauge).
        Complex Qhi = qq_star.coefficients[deg];
        double a_sq = Qhi.real() * ((n % 2 == 0) ? 1.0 : -1.0);
        if (a_sq <= 0) {
            throw SpectralFactorError(
                "Spectral factorization: leading coefficient incompatible "
                "with para-Hermitian structure (a² = " +
                std::to_string(a_sq) + " ≤ 0).");
        }
        double a = std::sqrt(a_sq);

        // Initial guess: q_0(s) = a · (s+1)^n. All roots at −1, strictly
        // Hurwitz, leading coefficient matches. Wilson's quadratic
        // convergence dominates the imperfect start.
        Polynomial<Complex> q = initial_guess(a, n);

        // Wilson iteration.
        // Tolerance: scale by polynomial magnitude so we don't over-iterate.
        double coef_scale = 0.0;
        for (const auto& c : qq_star.coefficients)
            coef_scale = std::max(coef_scale, std::abs(c));
        double tol = 1e-12 * std::max(1.0, coef_scale);

        const int max_iter = 50;
        int iter = 0;
        for (; iter < max_iter; ++iter) {
            Polynomial<Complex> qqs = q * q.para_conjugate();
            Polynomial<Complex> R = qq_star + qqs * Complex(-1, 0);

            double R_norm = 0.0;
            for (const auto& c : R.coefficients) R_norm += std::norm(c);
            R_norm = std::sqrt(R_norm);
            if (R_norm < tol) break;

            Polynomial<Complex> dq = wilson_step(q, R, n);
            std::vector<Complex> new_coeffs(n + 1, Complex(0));
            for (int k = 0; k <= n; ++k) {
                Complex qk = (k < static_cast<int>(q.coefficients.size()))
                             ? q.coefficients[k] : Complex(0);
                Complex dk = (k < static_cast<int>(dq.coefficients.size()))
                             ? dq.coefficients[k] : Complex(0);
                new_coeffs[k] = qk + dk;
            }
            q = Polynomial<Complex>(new_coeffs);
        }

        if (iter == max_iter) {
            throw SpectralFactorError(
                "Spectral factorization: Wilson iteration did not converge "
                "within " + std::to_string(max_iter) + " iterations.");
        }
        return q;
    }

    /**
     * Legacy root-finding based factorization. Kept for reference/testing.
     * Ill-conditioned when Q has roots clustered near the imaginary axis —
     * prefer factorize() (Wilson's iteration) for production use.
     */
    static Polynomial<Complex> factorize_by_roots(const Polynomial<Complex>& qq_star) {
        // Find all roots
        auto all_roots = qq_star.roots();
        int total = static_cast<int>(all_roots.size());

        // Roots of a para-Hermitian polynomial come in pairs (r, -conj(r))
        // — one per pair on each side of the imaginary axis. Degree must be even.
        if (total % 2 != 0) {
            throw SpectralFactorError(
                "Spectral factorization failed: odd total root count (" +
                std::to_string(total) + "). Polynomial is not para-Hermitian.");
        }
        int n = total / 2;

        // Select the n roots with smallest real parts (most stable).
        //
        // Using a sharp "r.real() < 0" cutoff is brittle: FP noise from the
        // companion-matrix eigensolve can place a root just on the wrong side
        // of the imaginary axis, leaving the counts asymmetric. Symptom: the
        // native build works but WASM (different libm / vectorization) fails
        // for odd filter orders where a reflection-zero pair sits near Re=0.
        //
        // Taking the n most-negative-real roots is equivalent for clean inputs
        // and robust to axis-straddling FP noise.
        std::sort(all_roots.begin(), all_roots.end(),
                  [](const Complex& a, const Complex& b) {
                      return a.real() < b.real();
                  });
        std::vector<Complex> stable_roots(all_roots.begin(), all_roots.begin() + n);

        // Sanity check: after selecting n most-negative, the split around the
        // imaginary axis should be approximately clean. If a supposedly-stable
        // root has a substantially positive real part, the input polynomial
        // is likely not para-Hermitian and we shouldn't silently accept it.
        //
        // Scale the tolerance by the polynomial's coefficient magnitude so
        // this check catches real problems but tolerates FP noise.
        double coef_scale = 0.0;
        for (const auto& c : qq_star.coefficients)
            coef_scale = std::max(coef_scale, std::abs(c));
        double tol = 1e-6 * std::max(1.0, coef_scale);
        if (stable_roots.back().real() > tol) {
            throw SpectralFactorError(
                "Spectral factorization failed: selected stable root has positive "
                "real part (" + std::to_string(stable_roots.back().real()) +
                ") beyond numerical tolerance. Input is likely not para-Hermitian.");
        }

        // Compute leading coefficient
        // For qq* = q * q.para_conjugate():
        // If q(s) has degree n and leading coefficient a,
        // then qq*(s) has leading coefficient |a|² * (-1)^n
        // So: |a|² = leading_coeff(qq*) * (-1)^n
        Complex leading_qq_star = qq_star.leading_coefficient();

        Complex a_squared = leading_qq_star * std::pow(-1.0, n);
        Complex a = std::sqrt(a_squared);

        // Choose the root with positive real part (minimum-phase convention)
        if (a.real() < 0) {
            a = -a;
        }

        return Polynomial<Complex>::from_roots(stable_roots, a);
    }

    /**
     * Compute spectral factor for Feldtkeller equation: qq* = pp* + rr*
     *
     * @param p Polynomial p
     * @param r Polynomial r
     * @return q such that q*q* = p*p* + r*r*
     */
    static Polynomial<Complex> feldtkeller(
        const Polynomial<Complex>& p,
        const Polynomial<Complex>& r
    ) {
        // Compute pp* + rr*
        auto p_star = p.para_conjugate();
        auto r_star = r.para_conjugate();

        auto pp_star = p * p_star;
        auto rr_star = r * r_star;
        auto qq_star = pp_star + rr_star;

        return factorize(qq_star);
    }

private:
    // Initial Hurwitz guess with leading coefficient a: q_0(s) = a · (s+1)^n.
    // Roots all at s = −1, strictly in the LHP, so q_0 is Hurwitz by construction.
    static Polynomial<Complex> initial_guess(double a, int n) {
        // Binomial expansion: (s+1)^n = Σ C(n,k) s^k
        std::vector<Complex> coeffs(n + 1, Complex(0));
        double binom = 1.0;
        coeffs[0] = Complex(a * binom, 0);
        for (int k = 1; k <= n; ++k) {
            binom = binom * (n - k + 1) / k;
            coeffs[k] = Complex(a * binom, 0);
        }
        return Polynomial<Complex>(coeffs);
    }

    // One Wilson Newton step: solve  q · Δq* + Δq · q* = R  for Δq.
    //
    // Both sides are para-Hermitian of degree ≤ 2n, giving 2n+1 real scalar
    // equations (even-index coefs real, odd-index coefs imag). Unknowns are
    // the 2n+1 real parameters of Δq (all Re/Im of d_0..d_n except Im(d_n),
    // which is fixed to 0 to preserve the gauge q_n ∈ ℝ). The map has a
    // 1-dim null space Δq = iβq that is removed by this gauge.
    static Polynomial<Complex> wilson_step(
        const Polynomial<Complex>& q,
        const Polynomial<Complex>& R,
        int n)
    {
        int dim = 2 * n + 1;                   // real unknowns / real equations
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim, dim);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(dim);

        auto qcoef = [&](int k) -> Complex {
            return (k >= 0 && k < static_cast<int>(q.coefficients.size()))
                   ? q.coefficients[k] : Complex(0);
        };
        auto rcoef = [&](int k) -> Complex {
            return (k >= 0 && k < static_cast<int>(R.coefficients.size()))
                   ? R.coefficients[k] : Complex(0);
        };

        // Column layout:
        //   cols 0..n     : Re(d_0), Re(d_1), ..., Re(d_n)
        //   cols n+1..2n  : Im(d_0), Im(d_1), ..., Im(d_{n-1})
        // Im(d_n) is the gauge, fixed at 0 (not a free variable).
        auto col_x = [&](int k) { return k; };
        auto col_y = [&](int k) { return n + 1 + k; };  // valid for k=0..n-1

        // Accumulate row contributions from each (a, b) pair with a+b=m.
        // Contribution to (q·Δq* + Δq·q*)_m:
        //   (−1)^b · [q_a · conj(d_b) + d_a · conj(q_b)]
        //
        // With q_a = α + iβ, q_b = γ + iδ, d_b = x_b + i·y_b, d_a = x_a + i·y_a:
        //   Re(contrib) = sign · (α·x_b + β·y_b  +  γ·x_a + δ·y_a)
        //   Im(contrib) = sign · (β·x_b − α·y_b  −  δ·x_a + γ·y_a)
        // where sign = (−1)^b.
        //
        // Para-Hermitian structure: even-index coefs of LHS/RHS are real
        // (row stores the real part), odd-index are pure imaginary (row
        // stores the imaginary part). One real equation per index m.
        for (int m = 0; m <= 2 * n; ++m) {
            int row = m;
            bool want_real = (m % 2 == 0);

            for (int a = std::max(0, m - n); a <= std::min(m, n); ++a) {
                int bb = m - a;
                double sign = (bb % 2 == 0) ? 1.0 : -1.0;
                Complex q_a = qcoef(a);
                Complex q_b = qcoef(bb);
                double alpha = q_a.real(), beta  = q_a.imag();
                double gamma = q_b.real(), delta = q_b.imag();

                if (want_real) {
                    A(row, col_x(bb)) += sign * alpha;
                    if (bb < n) A(row, col_y(bb)) += sign * beta;
                    A(row, col_x(a))  += sign * gamma;
                    if (a  < n) A(row, col_y(a))  += sign * delta;
                } else {
                    A(row, col_x(bb)) += sign * beta;
                    if (bb < n) A(row, col_y(bb)) += -sign * alpha;
                    A(row, col_x(a))  += -sign * delta;
                    if (a  < n) A(row, col_y(a))  += sign * gamma;
                }
            }

            Complex R_m = rcoef(m);
            b(row) = want_real ? R_m.real() : R_m.imag();
        }

        // Solve. completeOrthogonalDecomposition handles borderline rank
        // deficiency gracefully (identical behavior on native and WASM).
        Eigen::VectorXd sol = A.completeOrthogonalDecomposition().solve(b);

        std::vector<Complex> dq_coeffs(n + 1, Complex(0));
        for (int k = 0; k <= n; ++k) {
            double x = sol(col_x(k));
            double y = (k < n) ? sol(col_y(k)) : 0.0;
            dq_coeffs[k] = Complex(x, y);
        }
        return Polynomial<Complex>(dq_coeffs);
    }
};

} // namespace np
