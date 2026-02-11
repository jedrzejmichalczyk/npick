#include "coupling_matrix.hpp"
#include "realization.hpp"
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
    // Y-parameter approach for transversal coupling matrix synthesis.
    // This works correctly for para-Hermitian polynomials (standard lowpass prototypes).
    // For non-para-Hermitian polynomials (NP-modified), use coupling_matrix_by_homotopy().
    //
    // The Y-parameter transformation naturally maps S-parameter poles to
    // imaginary-axis poles suitable for the coupling matrix formulation.

    int n = F.degree();
    if (n < 1) {
        MatrixXcd result = MatrixXcd::Zero(2, 2);
        return result;
    }

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
    // Common denominator: E + F + m*(E* + F*)
    auto down = E + F + (E_para + F_para) * m;

    // Get poles of Y-parameters (roots of denominator)
    auto y_poles = down.roots();
    std::sort(y_poles.begin(), y_poles.end(),
              [](Complex a, Complex b) { return a.imag() < b.imag(); });

    // Compute residues
    RationalFunction y11_rf(y11_up, down);
    RationalFunction y12_rf(y12_up, down);

    VectorXcd y11_res = y11_rf.residues(y_poles);
    VectorXcd y12_res = y12_rf.residues(y_poles);

    // Build transversal coupling matrix
    // Structure: (N+2) x (N+2) matrix where:
    // - Diagonal[1:N] = j * poles (eigenvalues)
    // - Row/Col 0: sqrt(y11_res) couplings to source
    // - Row/Col N+1: y12_res/sqrt(y11_res) couplings to load

    MatrixXcd result = MatrixXcd::Zero(n + 2, n + 2);

    // Set diagonal eigenvalues (j * Y-poles)
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

MatrixXcd CouplingMatrix::from_polynomials_by_realization(
    const Polynomial<Complex>& F,
    const Polynomial<Complex>& P,
    const Polynomial<Complex>& E
) {
    // FoldedByS approach from C# ComplexReconfiguration
    // This works for general complex polynomials including NP-modified ones

    int n = E.degree();
    if (n < 1) {
        return MatrixXcd::Zero(2, 2);
    }

    // Step 1: Get poles (roots of E)
    auto poles = E.roots();
    std::sort(poles.begin(), poles.end(),
              [](Complex a, Complex b) { return a.imag() < b.imag(); });

    // Step 2: Compute S11 and S21 residues at each pole
    // Residue of F/E at pole p: F(p) / E'(p)
    auto E_deriv = E.derivative();

    VectorXcd S11_res(n), S21_res(n);
    for (int i = 0; i < n; ++i) {
        Complex p = poles[i];
        Complex E_deriv_val = E_deriv.evaluate(p);
        if (std::abs(E_deriv_val) < 1e-15) {
            // Fallback for repeated roots
            Complex prod(1.0);
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    prod *= (p - poles[j]);
                }
            }
            E_deriv_val = prod * E.leading_coefficient();
        }
        S11_res(i) = F.evaluate(p) / E_deriv_val;
        S21_res(i) = P.evaluate(p) / E_deriv_val;
    }

    // Step 3: Build S-parameter realization
    // A = diag(poles)
    // B[i,0] = sqrt(S11_res[i]), B[i,1] = S21_res[i] / sqrt(S11_res[i])
    // C = B^T
    // D = I (feedthrough = 1 for monic polynomials of same degree)

    MatrixXcd A_ss = MatrixXcd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A_ss(i, i) = poles[i];
    }

    VectorXcd tsqrt(n);
    VectorXcd vovertsqrt(n);
    for (int i = 0; i < n; ++i) {
        tsqrt(i) = complex_sqrt(S11_res(i));
        vovertsqrt(i) = (std::abs(tsqrt(i)) > 1e-15)
            ? S21_res(i) / tsqrt(i)
            : Complex(0);
    }

    MatrixXcd B_ss(n, 2);
    B_ss.col(0) = tsqrt;
    B_ss.col(1) = vovertsqrt;

    MatrixXcd C_ss(2, n);
    C_ss.row(0) = tsqrt.transpose();
    C_ss.row(1) = vovertsqrt.transpose();

    MatrixXcd D_ss = MatrixXcd::Identity(2, 2);

    Realization s_real(A_ss, C_ss, B_ss, D_ss);

    // Step 4: Cayley transform to Y-parameters
    Realization y_real = s_real.cayley();

    // Step 5: Minimal realization
    Realization y_min = y_real.min_real();

    // Step 6: Diagonalize
    Realization y_diag = y_min.diagonalize();

    // Step 7: Symmetrize
    Realization y_sym = y_diag.symmetrize();

    // Step 8: Build transversal coupling matrix
    // The Y-realization has poles at A(i,i) which are purely imaginary for lossless.
    // get_realization uses W(k,k) = M(k,k) + λ, with resonance when W(k,k) = 0.
    // Y-poles at s = A(k,k) = jα means resonance at λ = α.
    // So we need W(k,k) = M(k,k) + λ = 0 at λ = α, i.e., M(k,k) = -α = j*A(k,k)/j...
    // Actually: A(k,k) = jα (imaginary), so j*A(k,k) = j*jα = -α.
    // We want M(k,k) = -α so W(k,k) = -α + λ = 0 at λ = α.
    // So M(k,k) = j*A(k,k).
    // Couplings: from y_sym.B rows

    int n_final = y_sym.A.rows();
    MatrixXcd result = MatrixXcd::Zero(n_final + 2, n_final + 2);

    Complex j(0, 1);

    // Set diagonal eigenvalues
    for (int i = 0; i < n_final; ++i) {
        result(i + 1, i + 1) = j * y_sym.A(i, i);
    }

    // Set source couplings (from B column 0)
    for (int i = 0; i < n_final; ++i) {
        result(0, i + 1) = y_sym.B(i, 0);
        result(i + 1, 0) = y_sym.B(i, 0);
    }

    // Set load couplings (from B column 1)
    for (int i = 0; i < n_final; ++i) {
        result(n_final + 1, i + 1) = y_sym.B(i, 1);
        result(i + 1, n_final + 1) = y_sym.B(i, 1);
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
        double signed_numerator = c * M(k, l).real();
        if (std::abs(signed_numerator) < 1e-14) {
            signed_numerator = c * M(k, l).imag();
        }
        double sign = (signed_numerator >= 0.0) ? 1.0 : -1.0;
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

Eigen::Matrix2cd CouplingMatrix::eval_S(const MatrixXcd& cm, Complex s) {
    // Reconstruct Y-parameters from coupling matrix and apply inverse Cayley.
    // CM structure (transversal):
    //   - Diagonal cm(k+1, k+1) = j * A(k,k) where A is Y-realization state matrix
    //   - cm(0, k+1) = cm(k+1, 0) = m_s[k] = source coupling
    //   - cm(N-1, k+1) = cm(k+1, N-1) = m_l[k] = load coupling
    //
    // Y-realization: Y = B^T * (sI - A)^{-1} * B where B = [m_s; m_l]^T
    // A(k,k) = cm(k+1, k+1) / j = -j * cm(k+1, k+1)

    int N = cm.rows();
    int n = N - 2;

    if (n <= 0) {
        // Trivial case: direct connection
        Eigen::Matrix2cd S = Eigen::Matrix2cd::Zero();
        S(0, 1) = 1;
        S(1, 0) = 1;
        return S;
    }

    // Extract couplings
    VectorXcd m_s = cm.block(1, 0, n, 1);  // Source couplings
    VectorXcd m_l = cm.block(1, N - 1, n, 1);  // Load couplings

    // Build A diagonal from CM eigenvalues
    // A(k,k) = cm(k+1, k+1) / j = -j * cm(k+1, k+1)
    VectorXcd A_diag(n);
    Complex j(0, 1);
    for (int k = 0; k < n; ++k) {
        A_diag(k) = -j * cm(k + 1, k + 1);
    }

    // Compute Y = B^T * (sI - A)^{-1} * B
    // For diagonal A: Y[i,j] = sum_k B[k,i] * B[k,j] / (s - A[k,k])
    // B = [m_s, m_l] as columns, so B[k,0] = m_s[k], B[k,1] = m_l[k]
    Eigen::Matrix2cd Y = Eigen::Matrix2cd::Zero();
    for (int k = 0; k < n; ++k) {
        Complex denom = s - A_diag(k);
        Y(0, 0) += m_s(k) * m_s(k) / denom;
        Y(0, 1) += m_s(k) * m_l(k) / denom;
        Y(1, 0) += m_l(k) * m_s(k) / denom;
        Y(1, 1) += m_l(k) * m_l(k) / denom;
    }

    // S = (I - Y) * (I + Y)^{-1}  (inverse Cayley transform)
    Eigen::Matrix2cd I = Eigen::Matrix2cd::Identity();
    Eigen::Matrix2cd S = (I - Y) * (I + Y).inverse();

    return S;
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
