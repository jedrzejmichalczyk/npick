#include "nevanlinna_pick.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

namespace np {

HomotopyBase::~HomotopyBase() = default;
NevanlinnaPick::~NevanlinnaPick() = default;

NevanlinnaPick::NevanlinnaPick(
    const std::vector<Complex>& loads,
    const std::vector<double>& freqs,
    const std::vector<Complex>& transmission_zeros,
    double return_loss
)
    : order_(static_cast<int>(loads.size()))
{
    if (loads.empty()) {
        throw std::invalid_argument("NevanlinnaPick: loads must not be empty");
    }
    if (freqs.size() != loads.size()) {
        throw std::invalid_argument("NevanlinnaPick: loads and freqs must have the same size");
    }
    if (transmission_zeros.size() > loads.size()) {
        throw std::invalid_argument(
            "NevanlinnaPick: transmission_zeros size must be <= interpolation order");
    }
    if (return_loss <= 0.0 || !std::isfinite(return_loss)) {
        throw std::invalid_argument("NevanlinnaPick: return_loss must be finite and > 0");
    }
    for (double f : freqs) {
        if (!std::isfinite(f)) {
            throw std::invalid_argument("NevanlinnaPick: frequencies must be finite");
        }
    }

    // Compute frequency shift (mean)
    shift_ = std::accumulate(freqs.begin(), freqs.end(), 0.0) / freqs.size();

    // Normalize frequencies
    freqs_.resize(freqs.size());
    for (size_t i = 0; i < freqs.size(); ++i) {
        freqs_[i] = freqs[i] - shift_;
    }

    // Shift transmission zeros
    std::vector<Complex> shifted_tzs;
    for (const auto& tz : transmission_zeros) {
        shifted_tzs.push_back(tz - shift_);
    }

    // Store target loads
    target_loads_ = Eigen::Map<const VectorXcd>(loads.data(), loads.size());

    // Create Chebyshev filter to get initial polynomials
    ChebyshevFilter filter(order_, shifted_tzs, return_loss);

    // Get polynomial coefficients (stored in descending order for this class)
    initial_p_ = get_coeffs(filter.F());
    r_coeffs_ = get_coeffs(filter.P());

    // Compute initial S-parameters
    init_sparams_ = eval_map(initial_p_);
}

Polynomial<Complex> NevanlinnaPick::cached_feldtkeller(
    const Polynomial<Complex>& p_poly,
    const Polynomial<Complex>& r_poly,
    const VectorXcd& p_coeffs) const
{
    if (cache_valid_ && cached_p_coeffs_.size() == p_coeffs.size() &&
        cached_p_coeffs_ == p_coeffs) {
        return cached_q_poly_;
    }
    auto q = SpectralFactor::feldtkeller(p_poly, r_poly);
    cached_p_coeffs_ = p_coeffs;
    cached_q_poly_ = q;
    cache_valid_ = true;
    return q;
}

VectorXcd NevanlinnaPick::eval_map(const VectorXcd& p_coeffs) const {
    auto r_poly = build_polynomial(r_coeffs_);
    auto p_poly = build_polynomial(p_coeffs);

    // Compute spectral factor q such that |q|² = |p|² + |r|²
    auto q_poly = cached_feldtkeller(p_poly, r_poly, p_coeffs);

    // Evaluate p/q at each interpolation frequency
    VectorXcd result(freqs_.size());
    for (size_t k = 0; k < freqs_.size(); ++k) {
        Complex s = Complex(0, freqs_[k]);
        Complex p_val = p_poly.evaluate(s);
        Complex q_val = q_poly.evaluate(s);
        if (std::abs(q_val) < 1e-15) {
            throw std::runtime_error("NevanlinnaPick::eval_map: spectral factor vanished at evaluation point");
        }
        result(k) = p_val / q_val;
    }

    return result;
}

MatrixXcd NevanlinnaPick::eval_grad(const VectorXcd& p_coeffs) const {
    auto r_poly = build_polynomial(r_coeffs_);
    auto p_poly = build_polynomial(p_coeffs);
    auto q_poly = cached_feldtkeller(p_poly, r_poly, p_coeffs);

    int N = static_cast<int>(p_coeffs.size());
    int M = static_cast<int>(freqs_.size());

    // Evaluation points and precomputed values
    std::vector<Complex> s(M);
    std::vector<Complex> p_vals(M), q_vals(M);
    for (int k = 0; k < M; ++k) {
        s[k] = Complex(0, freqs_[k]);
        p_vals[k] = p_poly.evaluate(s[k]);
        q_vals[k] = q_poly.evaluate(s[k]);
    }

    auto p_para = p_poly.para_conjugate();
    auto q_para = q_poly.para_conjugate();

    // Build and factor the Bezout matrix ONCE for all coefficients.
    // The LHS matrix depends only on q (same for all dp perturbations).
    int q_deg = q_poly.degree();
    int lhs_max_deg = q_deg + (N - 1);
    int num_eqs = lhs_max_deg + 1;

    // Check if q has real coefficients (simplified system)
    double max_imag_q = 0, max_real_q = 0;
    for (const auto& c : q_poly.coefficients) {
        max_imag_q = std::max(max_imag_q, std::abs(c.imag()));
        max_real_q = std::max(max_real_q, std::abs(c.real()));
    }
    bool q_is_real = max_imag_q < 1e-10 * std::max(1.0, max_real_q);

    MatrixXcd result(M, N);

    if (q_is_real) {
        // Real q: dq has n real unknowns
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, N);
        for (int k = 0; k < num_eqs; ++k) {
            double sign_i = 1.0;
            for (int i = 0; i < N; ++i) {
                int j = k - i;
                if (j >= 0 && j <= q_deg) {
                    Complex q_para_c = q_para.coefficients[j];
                    Complex q_c = q_poly.coefficients[j];
                    Complex coeff = q_para_c + q_c * sign_i;
                    A(2 * k, i) += coeff.real();
                    A(2 * k + 1, i) += coeff.imag();
                }
                sign_i = -sign_i;
            }
        }
        auto qr = A.colPivHouseholderQr();

        for (int j = 0; j < N; ++j) {
            int deg = N - 1 - j;
            std::vector<Complex> dp_c(deg + 1, Complex(0));
            dp_c[deg] = Complex(1);
            Polynomial<Complex> dp(dp_c);
            auto rhs_poly = dp * p_para + p_poly * dp.para_conjugate();

            int rhs_deg = rhs_poly.degree();
            Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);
            for (int k = 0; k < num_eqs; ++k) {
                Complex rc = (k <= rhs_deg) ? rhs_poly.coefficients[k] : Complex(0);
                b(2 * k) = rc.real();
                b(2 * k + 1) = rc.imag();
            }

            Eigen::VectorXd sol = qr.solve(b);
            std::vector<Complex> dq_c(N);
            for (int i = 0; i < N; ++i)
                dq_c[i] = Complex(sol(i), 0);
            Polynomial<Complex> dq(dq_c);

            for (int k = 0; k < M; ++k) {
                Complex dp_val = std::pow(s[k], deg);
                Complex dq_val = dq.evaluate(s[k]);
                result(k, j) = (dp_val * q_vals[k] - p_vals[k] * dq_val)
                             / (q_vals[k] * q_vals[k]);
            }
        }
    } else {
        // Complex q: dq has 2n real unknowns
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, 2 * N);
        for (int k = 0; k < num_eqs; ++k) {
            double sign = 1.0;
            for (int i = 0; i < N; ++i) {
                int j_idx = k - i;
                if (j_idx >= 0 && j_idx <= q_deg) {
                    Complex qp_j = q_para.coefficients[j_idx];
                    Complex q_j = q_poly.coefficients[j_idx];
                    A(2*k,   2*i)   += qp_j.real() + sign * q_j.real();
                    A(2*k,   2*i+1) += -qp_j.imag() + sign * q_j.imag();
                    A(2*k+1, 2*i)   += qp_j.imag() + sign * q_j.imag();
                    A(2*k+1, 2*i+1) += qp_j.real() - sign * q_j.real();
                }
                sign = -sign;
            }
        }
        auto qr = A.colPivHouseholderQr();

        for (int j = 0; j < N; ++j) {
            int deg = N - 1 - j;
            std::vector<Complex> dp_c(deg + 1, Complex(0));
            dp_c[deg] = Complex(1);
            Polynomial<Complex> dp(dp_c);
            auto rhs_poly = dp * p_para + p_poly * dp.para_conjugate();

            int rhs_deg = rhs_poly.degree();
            Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);
            for (int k = 0; k < num_eqs; ++k) {
                Complex rc = (k <= rhs_deg) ? rhs_poly.coefficients[k] : Complex(0);
                b(2 * k) = rc.real();
                b(2 * k + 1) = rc.imag();
            }

            Eigen::VectorXd sol = qr.solve(b);
            std::vector<Complex> dq_c(N);
            for (int i = 0; i < N; ++i)
                dq_c[i] = Complex(sol(2*i), sol(2*i+1));

            // Null-space projection (remove iβq component)
            if (N > 0 && N <= static_cast<int>(q_poly.coefficients.size())) {
                double denom = q_poly.coefficients[N - 1].real();
                if (std::abs(denom) > 1e-15) {
                    double beta = dq_c[N - 1].imag() / denom;
                    for (int i = 0; i < N && i < static_cast<int>(q_poly.coefficients.size()); ++i)
                        dq_c[i] -= Complex(0, 1) * beta * q_poly.coefficients[i];
                }
            }
            Polynomial<Complex> dq(dq_c);

            for (int k = 0; k < M; ++k) {
                Complex dp_val = std::pow(s[k], deg);
                Complex dq_val = dq.evaluate(s[k]);
                result(k, j) = (dp_val * q_vals[k] - p_vals[k] * dq_val)
                             / (q_vals[k] * q_vals[k]);
            }
        }
    }

    return result;
}

Polynomial<Complex> NevanlinnaPick::solve_bezout(
    const Polynomial<Complex>& q,
    const Polynomial<Complex>& rhs,
    int max_degree
) {
    auto q_para = q.para_conjugate();
    int n = max_degree;

    // Check if q has essentially real coefficients
    double max_imag_q = 0, max_real_q = 0;
    for (const auto& c : q.coefficients) {
        max_imag_q = std::max(max_imag_q, std::abs(c.imag()));
        max_real_q = std::max(max_real_q, std::abs(c.real()));
    }
    bool q_is_real = max_imag_q < 1e-10 * std::max(1.0, max_real_q);

    if (q_is_real) {
        return solve_bezout_real(q, q_para, rhs, n);
    }

    // General complex case
    int q_deg = q.degree();
    int lhs_max_deg = q_deg + (n - 1);
    int rhs_deg = rhs.degree();
    int num_eqs = std::max(lhs_max_deg, rhs_deg) + 1;

    // Build system: A x = b where x = [Re(c_0), Im(c_0), Re(c_1), Im(c_1), ...]
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, 2 * n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);

    // Fill RHS
    for (int k = 0; k < num_eqs; ++k) {
        Complex rhs_coeff = (k <= rhs_deg) ? rhs.coefficients[k] : Complex(0);
        b(2 * k) = rhs_coeff.real();
        b(2 * k + 1) = rhs_coeff.imag();
    }

    // Fill matrix
    for (int k = 0; k < num_eqs; ++k) {
        double sign = 1.0;
        for (int i = 0; i < n; ++i) {
            int j = k - i;
            if (j >= 0 && j <= q_deg) {
                Complex q_para_j = q_para.coefficients[j];
                Complex q_j = q.coefficients[j];

                A(2 * k, 2 * i) += q_para_j.real() + sign * q_j.real();
                A(2 * k, 2 * i + 1) += -q_para_j.imag() + sign * q_j.imag();
                A(2 * k + 1, 2 * i) += q_para_j.imag() + sign * q_j.imag();
                A(2 * k + 1, 2 * i + 1) += q_para_j.real() - sign * q_j.real();
            }
            sign = -sign;
        }
    }

    // Solve least squares
    Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);

    // Extract coefficients
    std::vector<Complex> dq_coeffs(n);
    for (int i = 0; i < n; ++i) {
        dq_coeffs[i] = Complex(solution(2 * i), solution(2 * i + 1));
    }

    // Null-space projection to preserve monic normalization
    // The Bezout equation has null space: dq_null = i·β·q
    // Project out to make leading coefficient have zero imaginary part
    if (n > 0 && n <= static_cast<int>(q.coefficients.size())) {
        double denom = q.coefficients[n - 1].real();
        if (std::abs(denom) > 1e-15) {
            double beta = dq_coeffs[n - 1].imag() / denom;
            for (int i = 0; i < n && i < static_cast<int>(q.coefficients.size()); ++i) {
                dq_coeffs[i] -= Complex(0, 1) * beta * q.coefficients[i];
            }
        }
    }

    return Polynomial<Complex>(dq_coeffs);
}

Polynomial<Complex> NevanlinnaPick::solve_bezout_real(
    const Polynomial<Complex>& q,
    const Polynomial<Complex>& q_para,
    const Polynomial<Complex>& rhs,
    int n
) {
    int q_deg = q.degree();
    int lhs_max_deg = q_deg + (n - 1);
    int rhs_deg = rhs.degree();
    int num_eqs = std::max(lhs_max_deg, rhs_deg) + 1;

    // For real q, constrain dq to be real
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);

    // Fill RHS
    for (int k = 0; k < num_eqs; ++k) {
        Complex rhs_coeff = (k <= rhs_deg) ? rhs.coefficients[k] : Complex(0);
        b(2 * k) = rhs_coeff.real();
        b(2 * k + 1) = rhs_coeff.imag();
    }

    // Build matrix (simplified for real dq)
    for (int k = 0; k < num_eqs; ++k) {
        double sign_i = 1.0;
        for (int i = 0; i < n; ++i) {
            int j = k - i;
            if (j >= 0 && j <= q_deg) {
                Complex q_para_coeff = q_para.coefficients[j];
                Complex q_coeff = q.coefficients[j];
                Complex coeff = q_para_coeff + q_coeff * sign_i;
                A(2 * k, i) += coeff.real();
                A(2 * k + 1, i) += coeff.imag();
            }
            sign_i = -sign_i;
        }
    }

    // Solve least squares
    Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);

    // Build dq polynomial with real coefficients
    std::vector<Complex> dq_coeffs(n);
    for (int i = 0; i < n; ++i) {
        dq_coeffs[i] = Complex(solution(i), 0);
    }

    return Polynomial<Complex>(dq_coeffs);
}

VectorXcd NevanlinnaPick::calc_homotopy_function(const VectorXcd& x, double t) {
    // H(x, t) = γ(t) - ψ(x)
    // where γ(t) = t·init_sparams + (1-t)·target_loads*
    VectorXcd gamma = t * init_sparams_ + (1.0 - t) * target_loads_.conjugate();
    return gamma - eval_map(x);
}

VectorXcd NevanlinnaPick::calc_path_derivative(const VectorXcd& x, double t) {
    // ∂H/∂t = ∂γ/∂t = init_sparams - target_loads*
    return init_sparams_ - target_loads_.conjugate();
}

MatrixXcd NevanlinnaPick::calc_jacobian(const VectorXcd& x, double t) {
    // J = ∂H/∂x = -∂ψ/∂x
    return -eval_grad(x);
}

VectorXcd NevanlinnaPick::get_start_solution() {
    return initial_p_;
}

int NevanlinnaPick::get_num_variables() {
    return static_cast<int>(initial_p_.size());
}

Polynomial<Complex> NevanlinnaPick::build_polynomial(const VectorXcd& coeffs) {
    // Coefficients in vector: [p_{n-1}, p_{n-2}, ..., p_0] (descending)
    // Polynomial storage: [p_0, p_1, ..., p_{n-1}] (ascending)
    std::vector<Complex> poly_coeffs(coeffs.size());
    for (int i = 0; i < coeffs.size(); ++i) {
        poly_coeffs[i] = coeffs(coeffs.size() - 1 - i);
    }
    return Polynomial<Complex>(poly_coeffs);
}

VectorXcd NevanlinnaPick::get_coeffs(const Polynomial<Complex>& p) {
    // Convert from ascending to descending order
    VectorXcd result(p.coefficients.size());
    for (size_t i = 0; i < p.coefficients.size(); ++i) {
        result(i) = p.coefficients[p.coefficients.size() - 1 - i];
    }
    return result;
}

// =====================================================================
// NevanlinnaPickNormalized implementation
// =====================================================================

VectorXcd NevanlinnaPickNormalized::to_monic(const VectorXcd& x) {
    // Input: [p_{n-2}, ..., p_0] (n-1 coefficients)
    // Output: [1, p_{n-2}, ..., p_0] (n coefficients)
    VectorXcd result(x.size() + 1);
    result(0) = Complex(1, 0);
    result.tail(x.size()) = x;
    return result;
}

VectorXcd NevanlinnaPickNormalized::eval_map(const VectorXcd& x) const {
    return NevanlinnaPick::eval_map(to_monic(x));
}

MatrixXcd NevanlinnaPickNormalized::eval_grad(const VectorXcd& x) const {
    MatrixXcd full_grad = NevanlinnaPick::eval_grad(to_monic(x));
    // Remove first column (derivative w.r.t. leading coefficient = 1)
    return full_grad.rightCols(full_grad.cols() - 1);
}

VectorXcd NevanlinnaPickNormalized::get_start_solution() {
    // Return coefficients without the leading 1
    return initial_p_.tail(initial_p_.size() - 1);
}

int NevanlinnaPickNormalized::get_num_variables() {
    return static_cast<int>(initial_p_.size()) - 1;
}

MatrixXcd NevanlinnaPickNormalized::calc_coupling_matrix(const VectorXcd& x) const {
    auto m = to_monic(x);
    auto r_poly = build_polynomial(r_coeffs_);
    auto p_poly = build_polynomial(m);

    // Shift polynomials back to original frequency domain
    auto r_shifted = r_poly.shift(-shift_);
    auto p_shifted = p_poly.shift(-shift_);

    // Compute spectral factor
    auto e_poly = SpectralFactor::feldtkeller(p_shifted, r_shifted);

    // The NP solution produces general complex polynomials (not para-Hermitian),
    // so the realization-based builder is required.
    return CouplingMatrix::from_polynomials_by_realization(p_shifted, r_shifted, e_poly);
}

} // namespace np
