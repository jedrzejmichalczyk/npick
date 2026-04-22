#include "chebyshev_filter.hpp"
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace np {

ChebyshevFilter::ChebyshevFilter(
    int order,
    const std::vector<Complex>& transmission_zeros,
    double return_loss_db
)
    : order_(order)
    , tzs_(transmission_zeros)
    , return_loss_(return_loss_db)
{
    if (order_ <= 0) {
        throw std::invalid_argument("ChebyshevFilter: order must be > 0");
    }
    if (return_loss_ <= 0.0 || !std::isfinite(return_loss_)) {
        throw std::invalid_argument("ChebyshevFilter: return_loss_db must be finite and > 0");
    }
    if (static_cast<int>(tzs_.size()) > order_) {
        throw std::invalid_argument("ChebyshevFilter: transmission_zeros size must be <= order");
    }

    // Pad transmission zeros with infinity if needed
    while (static_cast<int>(tzs_.size()) < order_) {
        tzs_.push_back(Complex(std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()));
    }

    synthesize();
}

void ChebyshevFilter::synthesize() {
    // Initialize base cases for recursion
    // P0(λ) = 1
    cached_[0] = Polynomial<Complex>({Complex(1)});
    // P1(λ) = λ - 1/tz[0]  (or just λ if tz[0] is infinite)
    cached_[1] = Polynomial<Complex>({-to_f(tzs_[0]), Complex(1)});

    // Compute Chebyshev polynomial P_n in lambda domain
    auto P_lambda = calc_chebyshev_P(order_, tzs_);

    // Compute denominator D(λ) = prod(1 - λ/tz_i) for finite tz
    auto D_lambda = calc_D(tzs_);

    // Compute epsilon from return loss
    // RL = 10*log10(1 + 1/ε²) => ε = 1/sqrt(10^(RL/10) - 1)
    double epsilon = 1.0 / std::sqrt(std::pow(10.0, return_loss_ / 10.0) - 1.0);

    // Convert to s-domain: λ = -js => s = jλ
    auto F_s = lambda_to_s(P_lambda);

    Complex c = F_s.normalize();  // Make monic and get leading coefficient

    // Scale D by epsilon and leading coefficient
    auto D_scaled = D_lambda * (1.0 / (epsilon * std::abs(c)));

    // Convert P to s-domain. D is real, lambda_to_s produces a para-Hermitian
    // polynomial, and multiplying by -1 preserves that property. Any imaginary
    // phase factor would break para-Hermitian symmetry and corrupt from_polynomials.
    P_ = lambda_to_s(D_scaled) * Complex(-1, 0);

    F_ = F_s;

    // Compute E via spectral factorization: E*E* = F*F* + P*P*
    E_ = SpectralFactor::feldtkeller(P_, F_);
}

Polynomial<Complex> ChebyshevFilter::calc_chebyshev_P(int n, const std::vector<Complex>& tzs) {
    // Check cache
    if (cached_.count(n)) {
        return cached_[n];
    }

    // Recursive formula (Cameron's generalized Chebyshev):
    // P_n = c_n * P_{n-1} - a_n * b_n * P_{n-2}
    // where:
    // a_n = (λ - 1/tz_{n-2})²
    // b_n = sqrt(1 - (1/tz_{n-1})²) / sqrt(1 - (1/tz_{n-2})²)
    // c_n = (λ - 1/tz_{n-1}) + b_n * (λ - 1/tz_{n-2})

    Complex tz_n = tzs[n - 2];    // tz_{n-2} in 0-indexed
    Complex tz_np1 = tzs[n - 1];  // tz_{n-1} in 0-indexed

    Complex b = calc_B(tz_n, tz_np1);

    // IMPORTANT: In Cameron's formula from C#:
    // a(λ) = (1 - f_n*λ)²  (uses "1 - f*λ" form)
    // c(λ) = (λ - f_np1) + b*(λ - f_n)  (uses "λ - f" form)
    //
    // For infinite tz (f=0):
    // - a = 1 (not λ²!)
    // - c = 2λ
    // This gives standard Chebyshev recursion: P_n = 2λ*P_{n-1} - P_{n-2}

    // a = (1 - f_n*λ)²
    Polynomial<Complex> a_term({Complex(1), -to_f(tz_n)});  // 1 - f_n*λ
    auto a = a_term * a_term;

    // c = (λ - f_np1) + b*(λ - f_n)
    // Note: C# CalcC uses {-ToF(np), 1} = -f + λ = λ - f
    Polynomial<Complex> c0({-to_f(tz_np1), Complex(1)});  // λ - f_np1
    Polynomial<Complex> c1({-to_f(tz_n), Complex(1)});    // λ - f_n
    auto c_poly = c0 + c1 * b;

    // Recurse
    auto P_n_minus_2 = calc_chebyshev_P(n - 2, std::vector<Complex>(tzs.begin(), tzs.begin() + n - 2));
    auto P_n_minus_1 = calc_chebyshev_P(n - 1, std::vector<Complex>(tzs.begin(), tzs.begin() + n - 1));

    // P_n = c * P_{n-1} - a * b * P_{n-2}
    auto result = c_poly * P_n_minus_1 - a * b * P_n_minus_2;

    cached_[n] = result;
    return result;
}

Polynomial<Complex> ChebyshevFilter::calc_D(const std::vector<Complex>& tzs) {
    Polynomial<Complex> result({Complex(1)});
    for (const auto& tz : tzs) {
        // D *= (1 - λ/tz) = (1 - λ * (1/tz))
        // For finite tz: factor is (1 - λ/tz)
        // For infinite tz: factor is 1 (skip)
        if (!std::isinf(tz.real()) && !std::isinf(tz.imag())) {
            result = result * Polynomial<Complex>({Complex(1), -to_f(tz)});
        }
    }
    return result;
}

Complex ChebyshevFilter::calc_B(Complex tz_n, Complex tz_np1) {
    // b = sqrt(1 - (1/tz_{n+1})²) / sqrt(1 - (1/tz_n)²)
    Complex f_np1 = to_f(tz_np1);
    Complex f_n = to_f(tz_n);

    Complex up = std::sqrt(Complex(1) - f_np1 * f_np1);
    Complex down = std::sqrt(Complex(1) - f_n * f_n);

    if (std::abs(down) < 1e-15) {
        return Complex(1);  // Avoid division by zero
    }
    return up / down;
}

Complex ChebyshevFilter::to_f(Complex zero) {
    // Convert transmission zero to frequency: f = 1/zero
    // For infinite zero, return 0
    if (std::isinf(zero.real()) || std::isinf(zero.imag())) {
        return Complex(0);
    }
    return Complex(1) / zero;
}

Polynomial<Complex> ChebyshevFilter::lambda_to_s(const Polynomial<Complex>& p) {
    // Convert from lambda domain to s domain
    // λ = -js => coefficient of λ^k becomes coeff * (1/j)^k = coeff * (-j)^k
    std::vector<Complex> new_coeffs(p.coefficients.size());
    Complex j_inv = Complex(0, -1);  // 1/j = -j

    for (size_t k = 0; k < p.coefficients.size(); ++k) {
        new_coeffs[k] = p.coefficients[k] * std::pow(j_inv, static_cast<int>(k));
    }

    return Polynomial<Complex>(new_coeffs);
}

} // namespace np
