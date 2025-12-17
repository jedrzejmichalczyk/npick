#pragma once

#include "../types.hpp"
#include "homotopy_base.hpp"
#include "dopr853_predictor.hpp"
#include "corrector.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

namespace np {

/**
 * Homotopy path tracker.
 *
 * Tracks the solution path from t=1 (starting solution) to t=0 (target solution)
 * using predictor-corrector continuation.
 */
class PathTracker {
public:
    PathTracker(HomotopyBase* homotopy)
        : homotopy_(homotopy)
        , predictor_(homotopy)
        , corrector_(homotopy)
        , h(-0.1)
        , run_tol(1e-7)
        , final_tol(1e-12)
        , max_iterations(500)
        , verbose(false)
    {}

    /**
     * Track the solution path from t=1 to t=0.
     *
     * @param x0 Starting solution at t=1
     * @return Final solution at t=0
     */
    VectorXcd run(const VectorXcd& x0) {
        const double EPS = 1e-12;
        double t = 1.0;
        VectorXcd v = x0;

        // Store initial point
        store(t, v);

        // Verify initial condition
        VectorXcd init_check = homotopy_->calc_homotopy_function(v, t);
        if (init_check.lpNorm<Eigen::Infinity>() > 1e-5) {
            throw std::runtime_error("Homotopy: initial values check failed");
        }

        int n = static_cast<int>(v.size());
        VectorXcd delta_x(n);
        double current_h = h;
        int counter = 0;

        while (t > EPS) {
            if (counter > max_iterations) {
                throw std::runtime_error("Path tracker: max iterations exceeded at t=" +
                                        std::to_string(t));
            }

            try {
                // Predict
                predictor_.predict(v, t, current_h, delta_x);

                // Correct
                VectorXcd v_new(n);
                corrector_.correct(v + delta_x, t + current_h, v_new, run_tol);
                v = v_new;

                t += current_h;
                current_h = predictor_.h_next();

                // Adjust step to not overshoot t=0
                if (t + current_h < 0 && t > 0) {
                    current_h = -t;
                }

                store(t, v);
                counter++;

            } catch (const std::exception& e) {
                // On failure, reduce step size
                current_h *= 0.5;
                if (std::abs(current_h) < EPS) {
                    throw std::runtime_error("Path tracker: step underflow at t=" +
                                            std::to_string(t));
                }
            }
        }

        // Final correction at t=0
        corrector_.set_max_iterations(corrector_.max_iterations() * 5);
        VectorXcd v_final(n);
        corrector_.correct(v, 0.0, v_final, final_tol);
        v = v_final;

        store(0.0, v);

        // Verify final solution
        VectorXcd final_check = homotopy_->calc_homotopy_function(v, 0.0);
        double final_residual = final_check.lpNorm<Eigen::Infinity>();

        if (verbose) {
            std::cout << "Path tracker: iterations=" << counter
                     << ", final residual=" << final_residual << std::endl;
        }

        if (final_residual > 1e-7) {
            std::cerr << "Warning: path tracking solution may be inaccurate, residual="
                     << final_residual << std::endl;
        }

        return v;
    }

    // Access tracked path for debugging
    const std::vector<double>& get_ts() const { return ts_; }
    const std::vector<VectorXcd>& get_xs() const { return xs_; }

    // Configuration
    double h;           // Initial step size (negative for t decreasing)
    double run_tol;     // Tolerance during tracking
    double final_tol;   // Strict tolerance at t=0
    int max_iterations;
    bool verbose;

private:
    void store(double t, const VectorXcd& v) {
        ts_.push_back(t);
        xs_.push_back(v);
    }

    HomotopyBase* homotopy_;
    Dopr853Predictor predictor_;
    Corrector corrector_;
    std::vector<double> ts_;
    std::vector<VectorXcd> xs_;
};

} // namespace np
