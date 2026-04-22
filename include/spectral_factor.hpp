#pragma once

#include "types.hpp"
#include "polynomial.hpp"
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
 */
class SpectralFactor {
public:
    /**
     * Factorize a para-Hermitian polynomial into its minimum-phase factor.
     *
     * @param qq_star The input polynomial (must be para-Hermitian symmetric)
     * @return The minimum-phase factor q
     * @throws SpectralFactorError if roots are not symmetric
     */
    static Polynomial<Complex> factorize(const Polynomial<Complex>& qq_star) {
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
};

} // namespace np
