#pragma once

#include "../types.hpp"
#include "homotopy_base.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <stdexcept>

namespace np {

/**
 * Newton corrector for homotopy path tracking.
 *
 * Uses Newton's method to refine a predicted solution to satisfy
 * the homotopy equation H(x, t) = 0 to specified tolerance.
 */
class Corrector {
public:
    explicit Corrector(HomotopyBase* homotopy)
        : homotopy_(homotopy)
        , max_iterations_(55)
    {}

    /**
     * Refine a predicted solution using Newton iteration.
     *
     * @param x_guess Initial guess from predictor
     * @param t Current homotopy parameter
     * @param x_corrected Output: corrected solution
     * @param tol Convergence tolerance
     * @return true if converged, throws on failure
     */
    bool correct(const VectorXcd& x_guess, double t, VectorXcd& x_corrected,
                double tol = 1e-9) {
        VectorXcd x = x_guess;
        int n = static_cast<int>(x.size());

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Evaluate homotopy function
            VectorXcd r = homotopy_->calc_homotopy_function(x, t);

            // Check convergence
            if (r.norm() < tol) {
                x_corrected = x;
                return true;
            }

            // Compute Jacobian and solve for correction
            MatrixXcd J = homotopy_->calc_jacobian(x, t);

            // Use SVD for robustness
            Eigen::JacobiSVD<MatrixXcd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);

            // Check condition number
            double cond = svd.singularValues()(0) /
                         svd.singularValues()(svd.singularValues().size() - 1);
            if (cond > 1e20) {
                throw SingularMatrixError("Jacobian nearly singular, condition number: " +
                                         std::to_string(cond));
            }

            // Newton step: x_new = x - J^{-1} * r
            VectorXcd delta = svd.solve(-r);
            x = x + delta;

            // Check for stagnation
            if (delta.norm() < tol) {
                x_corrected = x;
                return true;
            }
        }

        throw ConvergenceError("Newton corrector failed: max iterations reached");
    }

    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int val) { max_iterations_ = val; }

private:
    HomotopyBase* homotopy_;
    int max_iterations_;
};

} // namespace np
