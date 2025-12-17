#include "coupling_matrix.hpp"
#include <Eigen/LU>
#include <cmath>
#include <algorithm>

namespace np {

VectorXcd RationalFunction::residues(const std::vector<Complex>& poles) const {
    int n = static_cast<int>(poles.size());
    VectorXcd result(n);

    // Normalize numerator coefficients by denominator leading coefficient
    Complex denom_leading = denominator.leading_coefficient();
    std::vector<Complex> num_coeffs = numerator.coefficients;
    for (auto& c : num_coeffs) {
        c /= denom_leading;
    }

    // If numerator degree < denominator degree, use direct residue formula
    // residue_i = N(pole_i) / D'(pole_i)
    // where D'(pole_i) = prod_{j≠i}(pole_i - pole_j)

    for (int i = 0; i < n; ++i) {
        Complex num_val = numerator.evaluate(poles[i]) / denom_leading;

        // Compute D'(pole_i) = prod_{j≠i}(pole_i - pole_j)
        Complex denom_deriv(1);
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                denom_deriv *= (poles[i] - poles[j]);
            }
        }

        result(i) = num_val / denom_deriv;
    }

    return result;
}

MatrixXcd CouplingMatrix::from_polynomials(
    const Polynomial<Complex>& F,
    const Polynomial<Complex>& P,
    const Polynomial<Complex>& E
) {
    int n = F.degree();

    // Compute poles (roots of E)
    auto E_poles = E.roots();

    // Sort poles for consistent ordering
    std::sort(E_poles.begin(), E_poles.end(),
              [](Complex a, Complex b) { return a.imag() < b.imag(); });

    // Compute Y-parameters from S-parameters
    // S11 = F/E, S21 = P/E
    // Y11 = (E-F+k*(E*-F*)) / (E+F+m*(E*+F*))
    // where k = (-1)^(n+1), m = (-1)^n

    double k = std::pow(-1, n + 1);
    double m = std::pow(-1, n);

    auto E_para = E.para_conjugate();
    auto F_para = F.para_conjugate();

    // Y11 numerator: E - F + k*(E* - F*)
    auto y11_up = E - F + (E_para - F_para) * k;
    // Y12 numerator: 2*P
    auto y12_up = P * Complex(2);
    // Y22 numerator: E + F + k*(E* + F*)
    auto y22_up = E + F + (E_para + F_para) * k;
    // Common denominator: E + F + m*(E* + F*)
    auto down = E + F + (E_para + F_para) * m;

    // Get poles of Y-parameters (roots of denominator)
    auto y_poles = down.roots();
    std::sort(y_poles.begin(), y_poles.end(),
              [](Complex a, Complex b) { return a.imag() < b.imag(); });

    // Compute residues
    RationalFunction y11_rf(y11_up, down);
    RationalFunction y12_rf(y12_up, down);
    RationalFunction y22_rf(y22_up, down);

    VectorXcd y11_res = y11_rf.residues(y_poles);
    VectorXcd y12_res = y12_rf.residues(y_poles);
    VectorXcd y22_res = y22_rf.residues(y_poles);

    // Build transversal coupling matrix
    // Structure: (N+2) x (N+2) matrix where:
    // - Diagonal[1:N] = j * poles (eigenvalues)
    // - Row/Col 0: sqrt(y11_res) couplings to source
    // - Row/Col N+1: y12_res/sqrt(y11_res) couplings to load

    MatrixXcd result = MatrixXcd::Zero(n + 2, n + 2);

    // Set diagonal eigenvalues (imaginary part of poles)
    for (int i = 0; i < n; ++i) {
        result(i + 1, i + 1) = Complex(0, 1) * y_poles[i];
    }

    // Set source couplings: sqrt(y11_res)
    for (int i = 0; i < n; ++i) {
        Complex t_sqrt = complex_sqrt(y11_res(i));
        result(0, i + 1) = t_sqrt;
        result(i + 1, 0) = t_sqrt;
    }

    // Set load couplings: y12_res / sqrt(y11_res)
    for (int i = 0; i < n; ++i) {
        Complex t_sqrt = complex_sqrt(y11_res(i));
        Complex v = (std::abs(t_sqrt) > 1e-15) ? y12_res(i) / t_sqrt : Complex(0);
        result(n + 1, i + 1) = v;
        result(i + 1, n + 1) = v;
    }

    return result;
}

MatrixXcd CouplingMatrix::transversal_to_folded(const MatrixXcd& transversal) {
    int N = transversal.rows();
    int n = N - 2;  // Filter order

    if (n <= 0) return transversal;

    MatrixXcd Q = MatrixXcd::Identity(N, N);
    MatrixXcd M = transversal;

    // Apply sequence of rotations to eliminate off-diagonal couplings
    // This implements the annihilation sequence for folded form

    // Sequence of annihilations (simplified - full algorithm is more complex)
    for (int iter = 0; iter < n; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            // Annihilate coupling M[0, i+2] if it exists
            if (i + 2 < n + 1 && std::abs(M(0, i + 2)) > 1e-14) {
                auto Qi = calc_rotation(M, 0, i + 2, 0, i + 1, -1);
                M = Qi * M * Qi.transpose();
                Q = Qi * Q;
            }
        }
    }

    return truncate(M);
}

MatrixXcd CouplingMatrix::calc_rotation(const MatrixXcd& M, int k, int l, int m, int n_idx, double c) {
    int N = M.rows();
    MatrixXcd Q = MatrixXcd::Identity(N, N);

    Complex theta(0);
    if (std::abs(M(m, n_idx)) > 1e-14) {
        theta = -std::atan(c * M(k, l) / M(m, n_idx));
    } else if (std::abs(M(k, l)) > 1e-14) {
        double sign = (c * M(k, l).real() / M(m, n_idx).real() > 0) ? 1.0 : -1.0;
        theta = Complex(-0.5 * PI * sign);
    }

    Complex cos_t = std::cos(theta);
    Complex sin_t = std::sin(theta);
    Complex j(0, 1);

    if (c > 0) {
        Q(k, k) = cos_t;
        Q(m, k) = -j * sin_t;
        Q(k, m) = j * sin_t;
        Q(m, m) = cos_t;
    } else {
        Q(l, l) = cos_t;
        Q(n_idx, l) = j * sin_t;
        Q(l, n_idx) = -j * sin_t;
        Q(n_idx, n_idx) = cos_t;
    }

    return Q;
}

Complex CouplingMatrix::S11(const MatrixXcd& cm, Complex s) {
    int N = cm.rows();
    int n = N - 2;

    if (n <= 0) return Complex(0);

    // Extract inner coupling matrix
    MatrixXcd M_inner = cm.block(1, 1, n, n);

    // Source and load coupling vectors
    VectorXcd m_s = cm.block(1, 0, n, 1);
    VectorXcd m_l = cm.block(1, N - 1, n, 1);

    // S11 = -j * m_s^T * (s*I - j*M)^{-1} * m_s
    MatrixXcd A = s * MatrixXcd::Identity(n, n) - Complex(0, 1) * M_inner;

    Eigen::FullPivLU<MatrixXcd> lu(A);
    VectorXcd x = lu.solve(m_s);

    return -Complex(0, 1) * m_s.transpose() * x;
}

Complex CouplingMatrix::S21(const MatrixXcd& cm, Complex s) {
    int N = cm.rows();
    int n = N - 2;

    if (n <= 0) return Complex(1);

    MatrixXcd M_inner = cm.block(1, 1, n, n);
    VectorXcd m_s = cm.block(1, 0, n, 1);
    VectorXcd m_l = cm.block(1, N - 1, n, 1);

    MatrixXcd A = s * MatrixXcd::Identity(n, n) - Complex(0, 1) * M_inner;

    Eigen::FullPivLU<MatrixXcd> lu(A);
    VectorXcd x = lu.solve(m_s);

    return -Complex(0, 1) * m_l.transpose() * x;
}

MatrixXcd CouplingMatrix::truncate(const MatrixXcd& m, double tol) {
    MatrixXcd result = m;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            if (std::abs(result(i, j)) < tol) {
                result(i, j) = Complex(0);
            }
        }
    }
    return result;
}

} // namespace np
