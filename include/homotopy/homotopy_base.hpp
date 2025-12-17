#pragma once

#include "../types.hpp"
#include <Eigen/Dense>

namespace np {

/**
 * Abstract base class for homotopy continuation problems.
 *
 * Defines the interface for solving H(x, t) = 0 where:
 * - H: R^n × [0,1] → C^n is the homotopy function
 * - x is the solution vector
 * - t is the homotopy parameter
 *
 * Path tracking follows x(t) from t=1 (known start) to t=0 (target solution).
 */
class HomotopyBase {
public:
    virtual ~HomotopyBase() = default;

    /**
     * Evaluate the homotopy function H(x, t).
     * The solution path satisfies H(x(t), t) = 0 for all t.
     */
    virtual VectorXcd calc_homotopy_function(const VectorXcd& x, double t) = 0;

    /**
     * Compute ∂H/∂t for implicit differentiation.
     * Used to find dx/dt = -J^{-1} · ∂H/∂t.
     */
    virtual VectorXcd calc_path_derivative(const VectorXcd& x, double t) = 0;

    /**
     * Compute the Jacobian ∂H/∂x.
     * Used for Newton correction and predictor steps.
     */
    virtual MatrixXcd calc_jacobian(const VectorXcd& x, double t) = 0;

    /**
     * Get the starting solution at t=1.
     */
    virtual VectorXcd get_start_solution() = 0;

    /**
     * Get the number of variables (dimension of x).
     */
    virtual int get_num_variables() = 0;
};

} // namespace np
