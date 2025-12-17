#pragma once

#include "types.hpp"
#include "polynomial.hpp"
#include "spectral_factor.hpp"
#include <vector>
#include <cmath>
#include <map>

namespace np {

/**
 * Chebyshev filter synthesis.
 *
 * Generates the characteristic polynomials F (reflection), P (transmission),
 * and E (spectral factor) for a Chebyshev filter with optional transmission zeros.
 *
 * The filter satisfies:
 * - |S11(jω)|² + |S21(jω)|² = 1 (lossless)
 * - S11 = F/E, S21 = P/E
 * - E*E* = F*F* + P*P* (Feldtkeller equation)
 */
class ChebyshevFilter {
public:
    /**
     * Construct a Chebyshev filter.
     *
     * @param order Filter order (number of resonators)
     * @param transmission_zeros Finite transmission zeros (in normalized frequency)
     * @param return_loss_db Return loss specification in dB (e.g., 20)
     */
    ChebyshevFilter(
        int order,
        const std::vector<Complex>& transmission_zeros,
        double return_loss_db
    );

    // Access synthesized polynomials (in s-domain)
    const Polynomial<Complex>& F() const { return F_; }  // Reflection numerator
    const Polynomial<Complex>& P() const { return P_; }  // Transmission numerator
    const Polynomial<Complex>& E() const { return E_; }  // Common denominator (spectral factor)

    int order() const { return order_; }
    double return_loss() const { return return_loss_; }

private:
    void synthesize();

    // Recursive Chebyshev polynomial with transmission zeros
    Polynomial<Complex> calc_chebyshev_P(int n, const std::vector<Complex>& tzs);

    // Denominator polynomial from transmission zeros
    Polynomial<Complex> calc_D(const std::vector<Complex>& tzs);

    // Helper coefficients for recursion
    Complex calc_B(Complex tz_n, Complex tz_np1);
    Complex calc_C_coeff(Complex tz_n, Complex tz_np1);
    Complex to_f(Complex zero);  // Convert transmission zero to frequency

    // Convert from lambda (Chebyshev) to s domain
    Polynomial<Complex> lambda_to_s(const Polynomial<Complex>& p);

    int order_;
    std::vector<Complex> tzs_;  // Transmission zeros (padded with infinity)
    double return_loss_;

    Polynomial<Complex> F_;  // Reflection polynomial (s-domain)
    Polynomial<Complex> P_;  // Transmission polynomial (s-domain)
    Polynomial<Complex> E_;  // Spectral factor (s-domain)

    // Memoization for recursive Chebyshev computation
    std::map<int, Polynomial<Complex>> cached_;
};

/**
 * Generate Chebyshev nodes for interpolation.
 *
 * @param left Left frequency bound
 * @param right Right frequency bound
 * @param n Number of nodes
 * @return Vector of Chebyshev node frequencies
 */
inline std::vector<double> chebyshev_nodes(double left, double right, int n) {
    std::vector<double> nodes(n);
    double mid = (left + right) / 2.0;
    double half_width = (right - left) / 2.0;

    for (int k = 0; k < n; ++k) {
        // Chebyshev nodes: cos((2k+1)π / 2n)
        double theta = PI * (2.0 * k + 1.0) / (2.0 * n);
        nodes[k] = mid + half_width * std::cos(theta);
    }

    return nodes;
}

} // namespace np
