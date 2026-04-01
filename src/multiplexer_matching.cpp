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
            int eval_pts = 201;
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
            std::cout << "  Initial results (Chebyshev nodes):\n";
            for (int i = 0; i < N; ++i) {
                std::cout << "    Channel " << (i + 1) << ": "
                          << std::fixed << std::setprecision(1)
                          << achieved_rls_[i] << " dB\n";
            }
        }

        // Step 5: Per-channel equiripple optimization
        if (equiripple_outer_iterations > 0) {
            if (verbose) std::cout << "  Step 4: Equiripple optimization...\n";
            run_equiripple(interp_freqs);
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

std::vector<double> MultiplexerMatching::find_channel_peaks(
    int ch, const std::vector<double>& ifreqs) const
{
    double fl = specs_[ch].freq_left;
    double fr = specs_[ch].freq_right;
    int n = static_cast<int>(ifreqs.size());

    std::vector<double> boundaries;
    boundaries.push_back(fl);
    for (double f : ifreqs) boundaries.push_back(f);
    boundaries.push_back(fr);

    std::vector<double> peaks(n + 1, 0.0);

    for (int seg = 0; seg <= n; ++seg) {
        double left = boundaries[seg];
        double right = boundaries[seg + 1];
        if (right - left < 1e-12) {
            peaks[seg] = std::abs(eval_channel_response(ch, left));
            continue;
        }

        // Coarse search
        double best_mag = 0, best_freq = left;
        int seg_pts = std::max(30 / (n + 1), 8);
        for (int i = 0; i <= seg_pts; ++i) {
            double f = left + (right - left) * i / static_cast<double>(seg_pts);
            double mag = std::abs(eval_channel_response(ch, f));
            if (std::isfinite(mag) && mag > best_mag) { best_mag = mag; best_freq = f; }
        }

        // Refine
        double a = std::max(left, best_freq - (right - left) / seg_pts);
        double b = std::min(right, best_freq + (right - left) / seg_pts);
        for (int r = 0; r < 4; ++r) {
            double step = (b - a) / 6.0;
            for (int k = 0; k <= 6; ++k) {
                double f = a + step * k;
                double mag = std::abs(eval_channel_response(ch, f));
                if (std::isfinite(mag) && mag > best_mag) { best_mag = mag; best_freq = f; }
            }
            a = std::max(left, best_freq - step);
            b = std::min(right, best_freq + step);
        }

        peaks[seg] = best_mag;
    }
    return peaks;
}

bool MultiplexerMatching::solve_coupled(
    const std::vector<std::vector<double>>& interp_freqs)
{
    int N = static_cast<int>(specs_.size());

    try {
        MultiplexerNevanlinnaPick mux_np(manifold_, specs_, interp_freqs);

        VectorXcd x = mux_np.get_start_solution();

        // Newton to solve decoupled problem (lambda=0) with damping
        for (int iter = 0; iter < 100; ++iter) {
            VectorXcd F = mux_np.compute_residual(x, 0.0);
            double res0 = F.norm();
            if (res0 < 1e-8) break;

            MatrixXcd J = mux_np.compute_jacobian_dp(x, 0.0);
            VectorXcd dx = J.colPivHouseholderQr().solve(-F);

            double alpha = 1.0;
            bool stepped = false;
            for (int ls = 0; ls < 12; ++ls) {
                VectorXcd x_try = x + alpha * dx;
                double res_try = mux_np.compute_residual(x_try, 0.0).norm();
                if (res_try < res0) { x = x_try; stepped = true; break; }
                alpha *= 0.5;
            }
            if (!stepped || alpha < 1e-10) break;
        }

        if (mux_np.compute_residual(x, 0.0).norm() > 1e-4) return false;

        // Coupled continuation — larger initial step for speed
        bool ok = mux_np.run_continuation(x, 0.01, 500, 20, 1e-6, false);
        if (!ok) return false;

        cms_ = mux_np.extract_coupling_matrices(x);

        // Update achieved RL
        achieved_rls_.resize(N);
        for (int i = 0; i < N; ++i) {
            double worst = 0;
            double fl = specs_[i].freq_left;
            double fr = specs_[i].freq_right;
            for (int k = 0; k < 201; ++k) {
                double freq = fl + (fr - fl) * k / 200.0;
                double mag = std::abs(eval_channel_response(i, freq));
                if (std::isfinite(mag) && mag > worst) worst = mag;
            }
            achieved_rls_[i] = to_db(worst);
        }

        return true;
    } catch (...) {
        return false;
    }
}

void MultiplexerMatching::run_equiripple(
    std::vector<std::vector<double>>& interp_freqs)
{
    // Exchange-based equiripple: one coupled re-solve per iteration.
    // 1. Find |G_i| peaks using CURRENT coupling matrices (cheap)
    // 2. Move interpolation frequencies toward worst peaks for ALL channels
    // 3. Re-solve coupled system ONCE with new frequencies
    // 4. Repeat

    int N = static_cast<int>(specs_.size());
    auto best_cms = cms_;
    auto best_rls = achieved_rls_;
    auto best_freqs = interp_freqs;

    for (int iter = 0; iter < equiripple_outer_iterations; ++iter) {
        bool any_moved = false;

        // For each channel: find peaks and move interpolation frequencies
        for (int ch = 0; ch < N; ++ch) {
            auto& ifreqs = interp_freqs[ch];
            int n = static_cast<int>(ifreqs.size());
            double fl = specs_[ch].freq_left;
            double fr = specs_[ch].freq_right;
            double bw = fr - fl;

            // Find peak locations in each interval (cheap: just eval_S)
            std::vector<double> boundaries;
            boundaries.push_back(fl);
            for (double f : ifreqs) boundaries.push_back(f);
            boundaries.push_back(fr);

            std::vector<double> peak_freqs(n + 1);
            std::vector<double> peak_mags(n + 1);
            for (int seg = 0; seg <= n; ++seg) {
                double left = boundaries[seg], right = boundaries[seg + 1];
                double best_f = (left + right) / 2.0, best_m = 0;
                int pts = std::max(40 / (n + 1), 8);
                for (int i = 0; i <= pts; ++i) {
                    double f = left + (right - left) * i / static_cast<double>(pts);
                    double m = std::abs(eval_channel_response(ch, f));
                    if (std::isfinite(m) && m > best_m) { best_m = m; best_f = f; }
                }
                peak_freqs[seg] = best_f;
                peak_mags[seg] = best_m;
            }

            double max_peak = *std::max_element(peak_mags.begin(), peak_mags.end());
            double min_peak = *std::min_element(peak_mags.begin(), peak_mags.end());

            if (verbose && iter == 0) {
                std::cout << "    Ch" << (ch + 1) << ": worst="
                          << std::fixed << std::setprecision(1) << to_db(max_peak)
                          << " dB, spread=" << (to_db(max_peak) - to_db(min_peak)) << " dB\n";
            }

            if (max_peak - min_peak < 1e-4 * max_peak) continue;

            // Move interpolation points: each point moves toward the midpoint
            // of its two adjacent peaks, with relaxation
            double relax = 0.3;
            for (int k = 0; k < n; ++k) {
                double target = (peak_freqs[k] + peak_freqs[k + 1]) / 2.0;
                ifreqs[k] += relax * (target - ifreqs[k]);
                ifreqs[k] = std::clamp(ifreqs[k], fl + bw * 0.02, fr - bw * 0.02);
            }
            std::sort(ifreqs.begin(), ifreqs.end());

            // Ensure minimum spacing
            double min_gap = bw * 0.02;
            for (int k = 1; k < n; ++k) {
                if (ifreqs[k] - ifreqs[k-1] < min_gap)
                    ifreqs[k] = ifreqs[k-1] + min_gap;
            }

            any_moved = true;
        }

        if (!any_moved) break;

        // One coupled re-solve with all updated frequencies
        if (solve_coupled(interp_freqs)) {
            // Check improvement
            double worst_now = *std::min_element(achieved_rls_.begin(), achieved_rls_.end());
            double worst_best = *std::min_element(best_rls.begin(), best_rls.end());

            if (worst_now < worst_best) {
                best_cms = cms_;
                best_rls = achieved_rls_;
                best_freqs = interp_freqs;
            }

            if (verbose) {
                std::cout << "    Iter " << iter << ":";
                for (int i = 0; i < N; ++i)
                    std::cout << " Ch" << (i+1) << "=" << std::fixed
                              << std::setprecision(1) << achieved_rls_[i] << "dB";
                std::cout << "\n";
            }
        } else {
            // Re-solve failed, revert and stop
            if (verbose) std::cout << "    Re-solve failed at iter " << iter << "\n";
            break;
        }
    }

    // Keep best result
    cms_ = best_cms;
    achieved_rls_ = best_rls;
    interp_freqs = best_freqs;
}

} // namespace np
