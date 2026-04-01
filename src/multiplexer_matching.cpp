#include "multiplexer/multiplexer_matching.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace np {

namespace {
double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}
} // namespace

MultiplexerMatching::MultiplexerMatching(
    const std::vector<ChannelSpec>& channel_specs,
    double center_frequency
)
    : specs_(channel_specs)
    , manifold_(static_cast<int>(channel_specs.size()), center_frequency)
{}

std::vector<MatrixXcd> MultiplexerMatching::run() {
    int N = static_cast<int>(specs_.size());

    if (verbose) {
        std::cout << "Multiplexer synthesis: " << N << " channels\n";
    }

    // Step 1: Design manifold
    if (verbose) std::cout << "  Step 1: Computing manifold line lengths...\n";
    manifold_.compute_line_lengths(specs_);

    // Step 2: Generate interpolation frequencies per channel
    std::vector<std::vector<double>> interp_freqs(N);
    for (int i = 0; i < N; ++i) {
        interp_freqs[i] = chebyshev_nodes(specs_[i].freq_left, specs_[i].freq_right,
                                           specs_[i].order);
    }

    // Step 3: Run coupled homotopy
    if (verbose) std::cout << "  Step 2-3: Running coupled homotopy continuation...\n";

    try {
        MultiplexerNevanlinnaPick mux_np(manifold_, specs_, interp_freqs);

        PathTracker tracker(&mux_np);
        tracker.h = path_tracker_h;
        tracker.run_tol = path_tracker_run_tol;
        tracker.final_tol = path_tracker_final_tol;
        tracker.verbose = false;
        tracker.max_iterations = 2000;

        VectorXcd x0 = mux_np.get_start_solution();

        if (verbose) {
            std::cout << "  System size: " << mux_np.get_num_variables() << " variables, "
                      << N << " channels\n";
            // Verify start
            VectorXcd H0 = mux_np.calc_homotopy_function(x0, 1.0);
            std::cout << "  Start residual: " << H0.norm()
                      << " (max component: " << H0.lpNorm<Eigen::Infinity>() << ")\n";
            for (int k = 0; k < std::min(static_cast<int>(H0.size()), 12); ++k) {
                std::cout << "    H[" << k << "] = " << H0(k) << "\n";
            }
            // Debug: verify init_sparams matches eval at start
            VectorXcd x0_copy = x0;
            VectorXcd H_at_1 = mux_np.calc_homotopy_function(x0_copy, 1.0);
            std::cout << "  H(x0,1) should be 0: " << H_at_1.norm() << "\n";
        }

        VectorXcd solution = tracker.run(x0);

        // Step 4: Extract coupling matrices
        cms_ = mux_np.extract_coupling_matrices(solution);

        // Compute achieved return losses
        achieved_rls_.resize(N);
        for (int i = 0; i < N; ++i) {
            double worst = 0;
            int eval_pts = 101;
            double fl = specs_[i].freq_left;
            double fr = specs_[i].freq_right;
            for (int k = 0; k < eval_pts; ++k) {
                double freq = fl + (fr - fl) * k / (eval_pts - 1.0);
                Complex g = eval_channel_response(i, freq);
                double mag = std::abs(g);
                if (std::isfinite(mag) && mag > worst) worst = mag;
            }
            achieved_rls_[i] = to_db(worst);
        }

        if (verbose) {
            std::cout << "  Results:\n";
            for (int i = 0; i < N; ++i) {
                std::cout << "    Channel " << (i + 1) << ": "
                          << achieved_rls_[i] << " dB\n";
            }
        }

    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "  Coupled homotopy failed: " << e.what() << "\n";
            std::cout << "  Falling back to independent channel synthesis...\n";
        }

        // Fallback: solve each channel independently
        cms_.resize(N);
        achieved_rls_.resize(N);

        for (int i = 0; i < N; ++i) {
            auto load_fn = [this, i](double freq) -> Complex {
                auto J = manifold_.compute_J(freq);
                return J(i + 1, i + 1);
            };

            ImpedanceMatching matcher(load_fn, specs_[i].order,
                                       specs_[i].transmission_zeros,
                                       specs_[i].return_loss_db,
                                       specs_[i].freq_left, specs_[i].freq_right);
            matcher.verbose = false;
            matcher.optimizer_max_iterations = 0;

            cms_[i] = matcher.run();
            achieved_rls_[i] = matcher.achieved_return_loss_db();

            if (verbose) {
                std::cout << "    Channel " << (i + 1) << " (independent): "
                          << achieved_rls_[i] << " dB\n";
            }
        }
    }

    return cms_;
}

Complex MultiplexerMatching::eval_channel_response(int channel_idx, double freq) const {
    if (channel_idx < 0 || channel_idx >= static_cast<int>(cms_.size())) {
        return Complex(1e10);
    }

    const auto& cm = cms_[channel_idx];
    if (cm.size() == 0) return Complex(1e10);

    // Evaluate filter S-parameters
    double norm_freq = (freq - (specs_[channel_idx].freq_left + specs_[channel_idx].freq_right) / 2.0)
                     / ((specs_[channel_idx].freq_right - specs_[channel_idx].freq_left) / 2.0);
    Complex s(0, norm_freq);
    auto S = CouplingMatrix::eval_S(cm, s);

    // Load: manifold self-reflection at this channel's port
    auto J = manifold_.compute_J(freq);
    Complex L = J(channel_idx + 1, channel_idx + 1);

    // G11 = S11 + S12*S21*L / (1 - S22*L)
    Complex denom = Complex(1) - S(1, 1) * L;
    if (std::abs(denom) < 1e-15) return Complex(1e10);
    return S(0, 0) + S(0, 1) * S(1, 0) * L / denom;
}

} // namespace np
