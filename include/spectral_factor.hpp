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

        // Select roots with negative real part (left half-plane)
        std::vector<Complex> stable_roots;
        stable_roots.reserve(all_roots.size() / 2);

        for (const auto& r : all_roots) {
            if (r.real() < 0) {
                stable_roots.push_back(r);
            }
        }

        // Verify symmetry: we should have exactly half the roots
        if (stable_roots.size() * 2 != all_roots.size()) {
            throw SpectralFactorError(
                "Spectral factorization failed: roots are not symmetric about imaginary axis. "
                "Got " + std::to_string(stable_roots.size()) + " stable roots out of " +
                std::to_string(all_roots.size()) + " total."
            );
        }

        // Compute leading coefficient
        // For qq* = q * q.para_conjugate():
        // If q(s) has degree n and leading coefficient a,
        // then qq*(s) has leading coefficient |a|² * (-1)^n
        // So: |a|² = leading_coeff(qq*) * (-1)^n
        int n = static_cast<int>(stable_roots.size());
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
