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

    // Step 3: Solve independent NP problems for start solution, then coupled continuation
    if (verbose) std::cout << "  Step 2: Solving independent channel matching...\n";

    try {
        MultiplexerNevanlinnaPick mux_np(manifold_, specs_, interp_freqs);

        if (verbose) {
            std::cout << "  System size: " << mux_np.get_num_variables() << " variables, "
                      << N << " channels\n";
        }

        // Solve the decoupled problem F(P, 0) = 0 using Newton iteration.
        // At lambda=0, each channel is independent: f(p_i) = J_ii.
        // Start from Chebyshev prototype and Newton-correct to exact solution.
        VectorXcd x = mux_np.get_start_solution();

        if (verbose) {
            VectorXcd F0 = mux_np.compute_residual(x, 0.0);
            std::cout << "  Chebyshev residual at lambda=0: " << F0.norm() << "\n";
        }

        // Newton iterations to solve F(x, 0) = 0
        for (int iter = 0; iter < 50; ++iter) {
            VectorXcd F = mux_np.compute_residual(x, 0.0);
            double res = F.norm();
            if (res < 1e-10) break;

            MatrixXcd J = mux_np.compute_jacobian_dp(x, 0.0);
            VectorXcd dx = J.colPivHouseholderQr().solve(-F);
            x += dx;

            if (dx.norm() < 1e-12) break;
        }

        VectorXcd F0 = mux_np.compute_residual(x, 0.0);
        if (verbose) {
            std::cout << "  Residual at lambda=0 after Newton: " << F0.norm() << "\n";
        }

        if (F0.norm() > 1e-6) {
            throw std::runtime_error("Failed to solve decoupled problem");
        }

        // Diagnostics: check Jacobian condition
        if (verbose) {
            MatrixXcd Jac = mux_np.compute_jacobian_dp(x, 0.0);
            Eigen::JacobiSVD<MatrixXcd> svd(Jac);
            auto sv = svd.singularValues();
            std::cout << "  Jacobian SVs: max=" << sv(0) << " min=" << sv(sv.size()-1)
                      << " cond=" << sv(0)/sv(sv.size()-1) << "\n";

            VectorXcd dFdl = mux_np.compute_dF_dlambda(x, 0.0);
            std::cout << "  |dF/dlambda|=" << dFdl.norm() << "\n";

            VectorXcd dp_pred = Jac.colPivHouseholderQr().solve(-dFdl * 0.001);
            std::cout << "  |Euler step (h=0.001)|=" << dp_pred.norm() << "\n";
        }

        // Step 3: Martinez continuation from lambda=0 to lambda=1
        if (verbose) std::cout << "  Step 3: Running coupled continuation (lambda: 0 -> 1)...\n";

        bool coupled_ok = mux_np.run_continuation(x, 0.001, 2000, 30, 1e-6, verbose);

        if (coupled_ok) {
            if (verbose) std::cout << "  Coupled continuation converged!\n";
        } else {
            if (verbose) std::cout << "  Coupled continuation did not reach lambda=1\n";
        }

        // Step 4: Extract coupling matrices
        cms_ = mux_np.extract_coupling_matrices(x);

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
