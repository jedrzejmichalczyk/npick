#pragma once

#include "../types.hpp"
#include <functional>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace np {

/**
 * Nelder-Mead simplex optimizer for unconstrained minimization.
 *
 * A derivative-free optimization method that maintains a simplex of n+1 points
 * in n-dimensional space and iteratively improves the worst point.
 */
class NelderMead {
public:
    using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

    struct Result {
        std::vector<double> minimizing_point;
        double minimum_value;
        int iterations;
        bool converged;
    };

    /**
     * Construct optimizer with convergence parameters.
     *
     * @param tolerance Convergence tolerance on function value spread
     * @param max_iterations Maximum number of iterations
     */
    NelderMead(double tolerance = 1e-4, int max_iterations = 5000)
        : tolerance_(tolerance)
        , max_iterations_(max_iterations)
        , alpha_(1.0)   // Reflection coefficient
        , gamma_(2.0)   // Expansion coefficient
        , rho_(0.5)     // Contraction coefficient
        , sigma_(0.5)   // Shrink coefficient
    {}

    /**
     * Find minimum of objective function starting from initial point.
     *
     * @param objective Function to minimize
     * @param initial_point Starting point
     * @param initial_step Initial simplex size (optional)
     * @return Optimization result
     */
    Result minimize(
        ObjectiveFunction objective,
        const std::vector<double>& initial_point,
        double initial_step = 0.1
    ) {
        int n = static_cast<int>(initial_point.size());

        // Initialize simplex: n+1 vertices
        std::vector<std::vector<double>> simplex(n + 1);
        std::vector<double> values(n + 1);

        // First vertex is initial point
        simplex[0] = initial_point;
        values[0] = objective(simplex[0]);

        // Other vertices: perturb each dimension
        for (int i = 0; i < n; ++i) {
            simplex[i + 1] = initial_point;
            double step = initial_step * std::abs(initial_point[i]);
            if (step < 1e-10) step = initial_step;
            simplex[i + 1][i] += step;
            values[i + 1] = objective(simplex[i + 1]);
        }

        int iterations = 0;
        bool converged = false;

        while (iterations < max_iterations_) {
            ++iterations;

            // Sort vertices by function value
            std::vector<int> order(n + 1);
            for (int i = 0; i <= n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                [&values](int a, int b) { return values[a] < values[b]; });

            // Reorder simplex and values
            std::vector<std::vector<double>> sorted_simplex(n + 1);
            std::vector<double> sorted_values(n + 1);
            for (int i = 0; i <= n; ++i) {
                sorted_simplex[i] = simplex[order[i]];
                sorted_values[i] = values[order[i]];
            }
            simplex = sorted_simplex;
            values = sorted_values;

            // Check convergence: spread of function values
            double spread = values[n] - values[0];
            if (spread < tolerance_) {
                converged = true;
                break;
            }

            // Compute centroid of all points except worst
            std::vector<double> centroid(n, 0.0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    centroid[j] += simplex[i][j];
                }
            }
            for (int j = 0; j < n; ++j) {
                centroid[j] /= n;
            }

            // Reflection
            std::vector<double> reflected(n);
            for (int j = 0; j < n; ++j) {
                reflected[j] = centroid[j] + alpha_ * (centroid[j] - simplex[n][j]);
            }
            double f_reflected = objective(reflected);

            if (f_reflected >= values[0] && f_reflected < values[n - 1]) {
                // Accept reflection
                simplex[n] = reflected;
                values[n] = f_reflected;
                continue;
            }

            if (f_reflected < values[0]) {
                // Try expansion
                std::vector<double> expanded(n);
                for (int j = 0; j < n; ++j) {
                    expanded[j] = centroid[j] + gamma_ * (reflected[j] - centroid[j]);
                }
                double f_expanded = objective(expanded);

                if (f_expanded < f_reflected) {
                    simplex[n] = expanded;
                    values[n] = f_expanded;
                } else {
                    simplex[n] = reflected;
                    values[n] = f_reflected;
                }
                continue;
            }

            // Contraction
            std::vector<double> contracted(n);
            if (f_reflected < values[n]) {
                // Outside contraction
                for (int j = 0; j < n; ++j) {
                    contracted[j] = centroid[j] + rho_ * (reflected[j] - centroid[j]);
                }
            } else {
                // Inside contraction
                for (int j = 0; j < n; ++j) {
                    contracted[j] = centroid[j] + rho_ * (simplex[n][j] - centroid[j]);
                }
            }
            double f_contracted = objective(contracted);

            if (f_contracted < std::min(f_reflected, values[n])) {
                simplex[n] = contracted;
                values[n] = f_contracted;
                continue;
            }

            // Shrink: contract all points toward best
            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < n; ++j) {
                    simplex[i][j] = simplex[0][j] + sigma_ * (simplex[i][j] - simplex[0][j]);
                }
                values[i] = objective(simplex[i]);
            }
        }

        // Find best point
        int best = 0;
        for (int i = 1; i <= n; ++i) {
            if (values[i] < values[best]) best = i;
        }

        return Result{simplex[best], values[best], iterations, converged};
    }

    // Configuration
    void set_tolerance(double tol) { tolerance_ = tol; }
    void set_max_iterations(int max_iter) { max_iterations_ = max_iter; }

private:
    double tolerance_;
    int max_iterations_;
    double alpha_;  // Reflection
    double gamma_;  // Expansion
    double rho_;    // Contraction
    double sigma_;  // Shrink
};

} // namespace np
