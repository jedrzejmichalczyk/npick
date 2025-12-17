#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include "spectral_factor.hpp"
#include <Eigen/Dense>

namespace np {

/**
 * Rational function in pole-residue form.
 * f(s) = sum_i (residue_i / (s - pole_i)) + direct_term
 */
struct RationalFunction {
    Polynomial<Complex> numerator;
    Polynomial<Complex> denominator;

    RationalFunction(const Polynomial<Complex>& num, const Polynomial<Complex>& den)
        : numerator(num), denominator(den) {}

    Complex evaluate(Complex s) const {
        return numerator.evaluate(s) / denominator.evaluate(s);
    }

    // Compute poles (roots of denominator)
    std::vector<Complex> poles() const {
        return denominator.roots();
    }

    // Compute residues at given poles
    // For f(s) = N(s)/D(s), residue at pole p is: N(p) / D'(p)
    // where D'(p) = prod_{j≠i}(p - pole_j) * leading_coeff(D)
    VectorXcd residues(const std::vector<Complex>& poles) const;
};

/**
 * Coupling matrix operations for filter synthesis.
 *
 * Converts between polynomial (F, P, E) representation and
 * coupling matrix representation of filters.
 */
class CouplingMatrix {
public:
    /**
     * Build transversal coupling matrix from F, P, E polynomials.
     *
     * @param F Reflection numerator polynomial
     * @param P Transmission numerator polynomial
     * @param E Common denominator (spectral factor)
     * @return (N+2) x (N+2) transversal coupling matrix
     */
    static MatrixXcd from_polynomials(
        const Polynomial<Complex>& F,
        const Polynomial<Complex>& P,
        const Polynomial<Complex>& E
    );

    /**
     * Convert transversal coupling matrix to folded form.
     *
     * Uses similarity rotations to eliminate cross-couplings
     * and achieve a canonical folded topology.
     *
     * @param transversal The transversal coupling matrix
     * @return Folded coupling matrix
     */
    static MatrixXcd transversal_to_folded(const MatrixXcd& transversal);

    /**
     * Evaluate S11 from coupling matrix at frequency s.
     */
    static Complex S11(const MatrixXcd& cm, Complex s);

    /**
     * Evaluate S21 from coupling matrix at frequency s.
     */
    static Complex S21(const MatrixXcd& cm, Complex s);

    /**
     * Truncate small values to zero for numerical cleanliness.
     */
    static MatrixXcd truncate(const MatrixXcd& m, double tol = 1e-13);

private:
    /**
     * Compute rotation matrix to eliminate a specific coupling.
     */
    static MatrixXcd calc_rotation(const MatrixXcd& M, int k, int l, int m, int n, double c);
};

/**
 * Compute the square root of a complex number with consistent branch.
 * Returns the root with positive real part, or positive imaginary if real is zero.
 */
inline Complex complex_sqrt(Complex z) {
    Complex result = std::sqrt(z);
    if (result.real() < 0 ||
        (std::abs(result.real()) < 1e-15 && result.imag() < 0)) {
        result = -result;
    }
    return result;
}

} // namespace np
