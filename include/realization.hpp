#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace np {

/**
 * State-space realization of a transfer function matrix.
 *
 * H(s) = D + C * (sI - A)^{-1} * B
 *
 * This class provides methods for:
 * - Evaluation at complex frequency
 * - Cayley transform (S to Y parameters)
 * - Minimal realization
 * - Diagonalization
 * - Symmetrization
 */
class Realization {
public:
    MatrixXcd A;  // State matrix (n x n)
    MatrixXcd B;  // Input matrix (n x p)
    MatrixXcd C;  // Output matrix (p x n)
    MatrixXcd D;  // Feedthrough matrix (p x p)

    Realization() = default;

    Realization(const MatrixXcd& A_, const MatrixXcd& C_, const MatrixXcd& B_, const MatrixXcd& D_)
        : A(A_), B(B_), C(C_), D(D_) {}

    Realization(const MatrixXcd& A_, const MatrixXcd& C_, const MatrixXcd& B_)
        : A(A_), B(B_), C(C_)
    {
        D = MatrixXcd::Zero(C_.rows(), B_.cols());
    }

    /**
     * Evaluate transfer function at complex frequency s.
     * H(s) = D + C * (sI - A)^{-1} * B
     */
    MatrixXcd eval(Complex s) const {
        int n = A.rows();
        if (n == 0) return D;

        MatrixXcd I = MatrixXcd::Identity(n, n);
        MatrixXcd resolvent = (s * I - A).inverse();
        return D + C * resolvent * B;
    }

    /**
     * Cayley transform: maps S-parameters to Y-parameters (or vice versa).
     *
     * Y = (I - S) * (I + S)^{-1}
     *
     * In state-space form:
     * If S has realization (A, B, C, D), then Y = (I - S) * (I + S)^{-1}
     * has realization computed from the formula.
     */
    Realization cayley() const {
        int p = D.rows();
        MatrixXcd I_p = MatrixXcd::Identity(p, p);

        // (I - S) has realization (A, B, -C, I - D)
        // (I + S)^{-1} has realization computed from inversion formula

        // Combined: Y = (I - S) * (I + S)^{-1}
        // Direct formula for Cayley transform of state-space:
        // Anew = A - B * (I + D)^{-1} * C
        // Bnew = -B * (I + D)^{-1} * sqrt(2)
        // Actually, let's use the multiplication formula

        // Negate: -S = (A, B, -C, -D)
        Realization neg_S(A, -C, B, -D);

        // I + S
        Realization I_plus_S = add(identity(p), *this);

        // I - S = I + (-S)
        Realization I_minus_S = add(identity(p), neg_S);

        // Y = (I - S) * (I + S)^{-1}
        Realization inv_I_plus_S = I_plus_S.inverse();
        return multiply(I_minus_S, inv_I_plus_S);
    }

    /**
     * Compute minimal realization by removing uncontrollable/unobservable states.
     */
    Realization min_real() const {
        // Simplified minimal realization using SVD of controllability/observability
        int n = A.rows();
        if (n == 0) return *this;

        // Compute observability matrix
        MatrixXcd obs = observability_matrix();

        // Compute controllability matrix
        MatrixXcd ctrl = controllability_matrix();

        // SVD of observability matrix
        Eigen::JacobiSVD<MatrixXcd> obs_svd(obs, Eigen::ComputeThinU | Eigen::ComputeThinV);
        int obs_rank = 0;
        for (int i = 0; i < obs_svd.singularValues().size(); ++i) {
            if (obs_svd.singularValues()(i) > 1e-10) obs_rank++;
        }

        // SVD of controllability matrix
        Eigen::JacobiSVD<MatrixXcd> ctrl_svd(ctrl, Eigen::ComputeThinU | Eigen::ComputeThinV);
        int ctrl_rank = 0;
        for (int i = 0; i < ctrl_svd.singularValues().size(); ++i) {
            if (ctrl_svd.singularValues()(i) > 1e-10) ctrl_rank++;
        }

        // The minimal realization order is min(obs_rank, ctrl_rank)
        int min_order = std::min(obs_rank, ctrl_rank);
        if (min_order >= n) return *this;  // Already minimal

        // Use balanced truncation approach
        MatrixXcd Qo = obs_svd.matrixV().leftCols(obs_rank);
        MatrixXcd Qc = ctrl_svd.matrixU().leftCols(ctrl_rank);

        // Find common subspace (simplified)
        MatrixXcd T = Qc.leftCols(min_order);
        MatrixXcd Tinv = T.completeOrthogonalDecomposition().pseudoInverse();

        MatrixXcd Anew = Tinv * A * T;
        MatrixXcd Bnew = Tinv * B;
        MatrixXcd Cnew = C * T;

        return Realization(Anew, Cnew, Bnew, D);
    }

    /**
     * Diagonalize: transform A to diagonal form via eigendecomposition.
     */
    Realization diagonalize() const {
        int n = A.rows();
        if (n == 0) return *this;

        Eigen::ComplexEigenSolver<MatrixXcd> solver(A);
        MatrixXcd T = solver.eigenvectors();
        MatrixXcd Tinv = T.inverse();

        MatrixXcd Anew = Tinv * A * T;
        MatrixXcd Bnew = Tinv * B;
        MatrixXcd Cnew = C * T;

        // Clean up numerical noise on off-diagonals
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && std::abs(Anew(i, j)) < 1e-12) {
                    Anew(i, j) = 0;
                }
            }
        }

        return Realization(Anew, Cnew, Bnew, D);
    }

    /**
     * Symmetrize: adjust B and C so that B = C^T (for symmetric coupling matrix).
     */
    Realization symmetrize() const {
        int n = A.rows();
        int p = B.cols();
        if (n == 0 || p != 2) return *this;

        MatrixXcd T = MatrixXcd::Zero(p, n);

        for (int i = 0; i < n; ++i) {
            // For 2-port: w1 = sqrt(C[0,i] * B[i,0]), w2 = sqrt(C[1,i] * B[i,1])
            Complex w1 = complex_sqrt(C(0, i) * B(i, 0));
            Complex w2 = complex_sqrt(C(1, i) * B(i, 1));

            // Check sign: v = C[0,i] * B[i,1] should equal w1 * w2 (with correct sign)
            // We need w1*w2 = v. If -w1*w2 is closer to v, negate w1.
            Complex v = C(0, i) * B(i, 1);
            if (std::abs(v + w1 * w2) < std::abs(v - w1 * w2)) {
                w1 = -w1;
            }

            T(0, i) = w1;
            T(1, i) = w2;
        }

        return Realization(A, T, T.transpose(), D);
    }

    /**
     * Get B matrix rows as vectors (for CreateTransversal).
     */
    std::vector<VectorXcd> get_Bs() const {
        std::vector<VectorXcd> result;
        for (int i = 0; i < B.rows(); ++i) {
            result.push_back(B.row(i).transpose());
        }
        return result;
    }

    /**
     * Get diagonal of A as vector.
     */
    VectorXcd diagonal() const {
        return A.diagonal();
    }

    /**
     * Inverse of transfer function: H^{-1}
     */
    Realization inverse() const {
        MatrixXcd Dinv = D.inverse();
        MatrixXcd Anew = A - B * Dinv * C;
        MatrixXcd Bnew = -B * Dinv;
        MatrixXcd Cnew = Dinv * C;
        return Realization(Anew, Cnew, Bnew, Dinv);
    }

    /**
     * Create identity realization (D = I, no dynamics).
     */
    static Realization identity(int p) {
        MatrixXcd A_id = MatrixXcd::Zero(0, 0);
        MatrixXcd B_id = MatrixXcd::Zero(0, p);
        MatrixXcd C_id = MatrixXcd::Zero(p, 0);
        MatrixXcd D_id = MatrixXcd::Identity(p, p);
        return Realization(A_id, C_id, B_id, D_id);
    }

    /**
     * Add two realizations: H1 + H2
     */
    static Realization add(const Realization& R, const Realization& V) {
        int n1 = R.A.rows();
        int n2 = V.A.rows();

        if (n1 == 0) {
            return Realization(V.A, V.C, V.B, V.D + R.D);
        }
        if (n2 == 0) {
            return Realization(R.A, R.C, R.B, R.D + V.D);
        }

        MatrixXcd Anew = MatrixXcd::Zero(n1 + n2, n1 + n2);
        Anew.topLeftCorner(n1, n1) = R.A;
        Anew.bottomRightCorner(n2, n2) = V.A;

        MatrixXcd Bnew(n1 + n2, R.B.cols());
        Bnew.topRows(n1) = R.B;
        Bnew.bottomRows(n2) = V.B;

        MatrixXcd Cnew(R.C.rows(), n1 + n2);
        Cnew.leftCols(n1) = R.C;
        Cnew.rightCols(n2) = V.C;

        MatrixXcd Dnew = R.D + V.D;

        return Realization(Anew, Cnew, Bnew, Dnew);
    }

    /**
     * Multiply two realizations: H1 * H2
     */
    static Realization multiply(const Realization& R, const Realization& V) {
        int n1 = R.A.rows();
        int n2 = V.A.rows();

        if (n1 == 0) {
            return Realization(V.A, R.D * V.C, V.B, R.D * V.D);
        }
        if (n2 == 0) {
            return Realization(R.A, R.C, R.B * V.D, R.D * V.D);
        }

        MatrixXcd Anew = MatrixXcd::Zero(n1 + n2, n1 + n2);
        Anew.topLeftCorner(n1, n1) = R.A;
        Anew.topRightCorner(n1, n2) = R.B * V.C;
        Anew.bottomRightCorner(n2, n2) = V.A;

        MatrixXcd Bnew(n1 + n2, V.B.cols());
        Bnew.topRows(n1) = R.B * V.D;
        Bnew.bottomRows(n2) = V.B;

        MatrixXcd Cnew(R.C.rows(), n1 + n2);
        Cnew.leftCols(n1) = R.C;
        Cnew.rightCols(n2) = R.D * V.C;

        MatrixXcd Dnew = R.D * V.D;

        return Realization(Anew, Cnew, Bnew, Dnew);
    }

private:
    MatrixXcd observability_matrix() const {
        int n = A.rows();
        int p = C.rows();
        MatrixXcd obs(p * n, n);

        MatrixXcd Ak = MatrixXcd::Identity(n, n);
        for (int i = 0; i < n; ++i) {
            obs.middleRows(i * p, p) = C * Ak;
            Ak = Ak * A;
        }
        return obs;
    }

    MatrixXcd controllability_matrix() const {
        int n = A.rows();
        int m = B.cols();
        MatrixXcd ctrl(n, m * n);

        MatrixXcd Ak = MatrixXcd::Identity(n, n);
        for (int i = 0; i < n; ++i) {
            ctrl.middleCols(i * m, m) = Ak * B;
            Ak = A * Ak;
        }
        return ctrl;
    }

    static Complex complex_sqrt(Complex z) {
        Complex result = std::sqrt(z);
        if (result.real() < 0 ||
            (std::abs(result.real()) < 1e-15 && result.imag() < 0)) {
            result = -result;
        }
        return result;
    }
};

} // namespace np
