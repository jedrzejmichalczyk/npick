#pragma once

#include "../types.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace np {

/**
 * Ideal lossless reciprocal 3-port T-junction.
 *
 * Port 0 = common, Port 1 = branch 1, Port 2 = branch 2.
 * Parameterized by power split ratio alpha: fraction of power to port 1.
 *
 * The S-matrix is unitary (lossless) and symmetric (reciprocal):
 *   S00 = 0
 *   S01 = S10 = sqrt(alpha)
 *   S02 = S20 = sqrt(1 - alpha)
 *   S11 = -(1 - alpha)
 *   S22 = -alpha
 *   S12 = S21 = sqrt(alpha * (1 - alpha))
 */
class TJunction {
public:
    /**
     * Construct with power split ratio.
     * @param alpha Fraction of power from port 0 to port 1. Must be in (0, 1).
     */
    explicit TJunction(double alpha = 0.5) {
        if (alpha <= 0.0 || alpha >= 1.0) {
            throw std::invalid_argument("TJunction: alpha must be in (0, 1)");
        }
        double sa = std::sqrt(alpha);
        double sb = std::sqrt(1.0 - alpha);

        S_ = Eigen::Matrix3cd::Zero();
        S_(0, 0) = 0.0;
        S_(0, 1) = sa;        S_(1, 0) = sa;
        S_(0, 2) = sb;        S_(2, 0) = sb;
        S_(1, 1) = -(1.0 - alpha);
        S_(2, 2) = -alpha;
        S_(1, 2) = sa * sb;   S_(2, 1) = sa * sb;
    }

    /**
     * Construct with custom 3x3 S-matrix.
     */
    explicit TJunction(const Eigen::Matrix3cd& S) : S_(S) {}

    const Eigen::Matrix3cd& S() const { return S_; }

    /**
     * Check unitarity (lossless) and reciprocity.
     */
    bool is_valid(double tol = 1e-10) const {
        // Unitarity: S * S^H = I
        Eigen::Matrix3cd eye = Eigen::Matrix3cd::Identity();
        double unitary_err = (S_ * S_.adjoint() - eye).norm();
        // Reciprocity: S = S^T
        double recip_err = (S_ - S_.transpose()).norm();
        return unitary_err < tol && recip_err < tol;
    }

private:
    Eigen::Matrix3cd S_;
};

} // namespace np
