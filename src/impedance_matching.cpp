#include "impedance_matching.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace np {
namespace {
constexpr double kPenaltyCost = 1e10;
constexpr double kMinMagnitude = 1e-15;

double safe_to_db(double mag) {
    if (!std::isfinite(mag) || mag < kMinMagnitude) {
        return 20.0 * std::log10(kMinMagnitude);
    }
    return 20.0 * std::log10(mag);
}
} // namespace

ImpedanceMatching::ImpedanceMatching(
    LoadFunction load,
    int order,
    const std::vector<Complex>& transmission_zeros,
    double return_loss_db,
    double freq_left,
    double freq_right
)
    : load_(load)
    , order_(order)
    , tzs_(transmission_zeros)
    , return_loss_(return_loss_db)
    , freq_left_(freq_left)
    , freq_right_(freq_right)
    , achieved_rl_db_(0.0)
{
    if (!load_) {
        throw std::invalid_argument("ImpedanceMatching: load function must be valid");
    }
    if (order_ <= 0) {
        throw std::invalid_argument("ImpedanceMatching: order must be > 0");
    }
    if (static_cast<int>(tzs_.size()) > order_) {
        throw std::invalid_argument("ImpedanceMatching: transmission_zeros size must be <= order");
    }
    if (return_loss_ <= 0.0 || !std::isfinite(return_loss_)) {
        throw std::invalid_argument("ImpedanceMatching: return_loss_db must be finite and > 0");
    }
    if (!std::isfinite(freq_left_) || !std::isfinite(freq_right_) || freq_left_ >= freq_right_) {
        throw std::invalid_argument("ImpedanceMatching: frequency bounds must be finite and satisfy left < right");
    }

    // Initialize with Chebyshev node frequencies
    freqs_ = chebyshev_nodes(freq_left_, freq_right_, order_);
}

MatrixXcd ImpedanceMatching::run_single(const std::vector<double>& interp_freqs) {
    if (static_cast<int>(interp_freqs.size()) != order_) {
        if (verbose) {
            std::cerr << "run_single: expected " << order_
                      << " interpolation frequencies, got " << interp_freqs.size() << "\n";
        }
        return MatrixXcd();
    }

    for (double f : interp_freqs) {
        if (!std::isfinite(f)) {
            if (verbose) {
                std::cerr << "run_single: non-finite interpolation frequency\n";
            }
            return MatrixXcd();
        }
    }

    // Compute target loads at interpolation frequencies
    std::vector<Complex> target_loads(order_);
    for (int i = 0; i < order_; ++i) {
        target_loads[i] = load_(interp_freqs[i]);
    }

    try {
        // Create normalized Nevanlinna-Pick problem
        NevanlinnaPickNormalized np_problem(target_loads, interp_freqs, tzs_, return_loss_);

        // Create path tracker
        PathTracker tracker(&np_problem);
        tracker.h = path_tracker_h;
        tracker.run_tol = path_tracker_run_tol;
        tracker.final_tol = path_tracker_final_tol;
        tracker.verbose = false;

        // Get starting solution and track path
        VectorXcd x0 = np_problem.get_start_solution();
        VectorXcd solution = tracker.run(x0);

        // Check convergence
        VectorXcd residual = np_problem.calc_homotopy_function(solution, 0.0);
        if (residual.norm() > 1e-5) {
            return MatrixXcd();  // Failed
        }

        // Extract coupling matrix from solution
        return np_problem.calc_coupling_matrix(solution);

    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "NP solver failed: " << e.what() << "\n";
        }
        return MatrixXcd();  // Failed
    }
}

Complex ImpedanceMatching::eval_G11(const MatrixXcd& cm, double freq) const {
    if (cm.rows() < 2 || cm.cols() < 2 || !std::isfinite(freq)) {
        return Complex(kPenaltyCost);
    }

    Complex s = Complex(0, freq);
    auto S = CouplingMatrix::eval_S(cm, s);

    Complex S11 = S(0, 0);
    Complex S12 = S(0, 1);
    Complex S21 = S(1, 0);
    Complex S22 = S(1, 1);
    Complex L = load_(freq);

    // G11 = S11 + S12*S21*L / (1 - S22*L)
    Complex denom = Complex(1) - S22 * L;
    if (std::abs(denom) < 1e-15) {
        return Complex(kPenaltyCost);  // Avoid division by zero
    }
    return S11 + S12 * S21 * L / denom;
}

double ImpedanceMatching::compute_cost(const std::vector<double>& interp_freqs) {
    if (static_cast<int>(interp_freqs.size()) != order_) {
        return kPenaltyCost;
    }

    // Check that frequencies are finite, within bounds, and strictly monotonic
    bool is_increasing = true;
    if (order_ > 1) {
        is_increasing = interp_freqs[1] > interp_freqs[0];
    }

    for (int i = 0; i < order_; ++i) {
        if (!std::isfinite(interp_freqs[i]) ||
            interp_freqs[i] < freq_left_ || interp_freqs[i] > freq_right_) {
            return kPenaltyCost;
        }
        if (i > 0) {
            double diff = interp_freqs[i] - interp_freqs[i - 1];
            if (std::abs(diff) < 1e-12) {
                return kPenaltyCost;
            }
            if (is_increasing && diff < 0.0) {
                return kPenaltyCost;
            }
            if (!is_increasing && diff > 0.0) {
                return kPenaltyCost;
            }
        }
    }

    // Run NP solver
    MatrixXcd cm = run_single(interp_freqs);
    if (cm.size() == 0) {
        return kPenaltyCost;
    }

    // Evaluate max|G11| over passband
    const int eval_points = std::max(cost_eval_points, 2);
    double max_g11 = 0.0;
    for (int i = 0; i < eval_points; ++i) {
        double freq = freq_left_ + (freq_right_ - freq_left_) * i / (eval_points - 1);
        Complex g11 = eval_G11(cm, freq);
        double mag = std::abs(g11);
        if (!std::isfinite(mag)) {
            return kPenaltyCost;
        }
        if (mag > max_g11) {
            max_g11 = mag;
        }
    }

    return max_g11;
}

MatrixXcd ImpedanceMatching::run() {
    if (verbose) {
        std::cout << "Starting MinMax optimization...\n";
        std::cout << "  Order: " << order_ << "\n";
        std::cout << "  Return loss: " << return_loss_ << " dB\n";
        std::cout << "  Passband: [" << freq_left_ << ", " << freq_right_ << "]\n";
    }

    // Initial cost with Chebyshev frequencies
    double initial_cost = compute_cost(freqs_);
    if (verbose) {
        double initial_db = safe_to_db(initial_cost);
        std::cout << "  Initial max|G11|: " << initial_db << " dB\n";
    }

    // Create optimizer
    NelderMead optimizer(optimizer_tolerance, optimizer_max_iterations);

    // Cost function wrapper
    auto cost_fn = [this](const std::vector<double>& freqs) {
        return this->compute_cost(freqs);
    };

    // Run optimization
    auto result = optimizer.minimize(cost_fn, freqs_, 0.15);

    if (verbose) {
        std::cout << "  Optimization " << (result.converged ? "converged" : "stopped")
                  << " after " << result.iterations << " iterations\n";
        double final_db = safe_to_db(result.minimum_value);
        std::cout << "  Final max|G11|: " << final_db << " dB\n";
    }

    if (static_cast<int>(result.minimizing_point.size()) == order_) {
        freqs_ = result.minimizing_point;
    }

    // Get final coupling matrix
    cm_ = run_single(freqs_);

    // Compute achieved return loss
    achieved_rl_db_ = safe_to_db(cm_.size() == 0 ? kPenaltyCost : result.minimum_value);

    return cm_;
}

} // namespace np
