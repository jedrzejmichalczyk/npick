#pragma once

#include <cmath>
#include <limits>

namespace np {

/**
 * Adaptive step size controller for ODE integration.
 *
 * Uses PI control to adjust step size based on error estimates.
 * Based on Hairer, Nørsett, and Wanner's step size selection algorithm.
 */
class StepController {
public:
    static constexpr double beta = 0.0;
    static constexpr double alpha = 0.2 - beta * 0.75;
    static constexpr double safe = 0.9;
    static constexpr double min_scale = 0.2;
    static constexpr double max_scale = 1.01;

    StepController() : err_old_(1e-4), reject_(false), h_next_(0) {}

    /**
     * Determine if step was successful and compute next step size.
     *
     * @param error Estimated error (should be <= 1 for success)
     * @param h Current step size (may be modified on rejection)
     * @return true if step accepted, false if rejected
     */
    bool success(double error, double& h) {
        double scale;

        if (error <= 1.0) {
            // Step accepted
            if (std::abs(error) < std::numeric_limits<double>::epsilon()) {
                scale = max_scale;
            } else {
                scale = safe * std::pow(error, -alpha) * std::pow(err_old_, beta);
                scale = std::max(min_scale, std::min(scale, max_scale));
            }

            if (reject_) {
                h_next_ = h * std::min(scale, 1.0);
            } else {
                h_next_ = h * scale;
            }

            err_old_ = std::max(error, 1e-4);
            reject_ = false;
            return true;
        }

        // Step rejected
        scale = std::max(safe * std::pow(error, -alpha), min_scale);
        h *= scale;
        reject_ = true;
        return false;
    }

    double h_next() const { return h_next_; }
    bool rejected() const { return reject_; }

private:
    double err_old_;
    bool reject_;
    double h_next_;
};

} // namespace np
