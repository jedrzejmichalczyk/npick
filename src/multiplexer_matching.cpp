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

    // Check manifold symmetry at center frequency
    if (verbose) {
        auto J = manifold_.compute_J((specs_[0].freq_left + specs_[N-1].freq_right) / 2.0);
        std::cout << "  Manifold J symmetry check: ||J - J^T|| = "
                  << (J - J.transpose()).norm() << ", ||J|| = " << J.norm() << "\n";
        std::cout << "  J =\n" << J << "\n";
    }

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

        // Newton iterations to solve F(x, 0) = 0 using the real-variable
        // Wirtinger step (F is not holomorphic in p; see multiplexer_np.cpp).
        for (int iter = 0; iter < 50; ++iter) {
            VectorXcd F = mux_np.compute_residual(x, 0.0);
            double res = F.norm();
            if (res < 1e-10) break;

            VectorXcd dx = mux_np.newton_step(x, 0.0, F);
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
    int N = static_cast<int>(specs_.size());

    const auto& cm = cms_[channel_idx];
    if (cm.size() == 0) return Complex(1e10);

    // Evaluate filter S-parameters
    double norm_freq = (freq - (specs_[channel_idx].freq_left + specs_[channel_idx].freq_right) / 2.0)
                     / ((specs_[channel_idx].freq_right - specs_[channel_idx].freq_left) / 2.0);
    Complex s(0, norm_freq);
    auto S = CouplingMatrix::eval_S(cm, s);

    // Coupled load: manifold + other filters' reflections
    auto J = manifold_.compute_J(freq);
    int i = channel_idx;
    Complex L = J(i + 1, i + 1);  // J_ii (decoupled part)

    if (N > 1) {
        // Add coupling through other channels
        VectorXcd vi = manifold_.compute_Vi(J, i);
        MatrixXcd wi = manifold_.compute_Wi(J, i);
        int dim = N - 1;
        MatrixXcd Fi = MatrixXcd::Zero(dim, dim);
        int col = 0;
        for (int k = 0; k < N; ++k) {
            if (k == i) continue;
            if (k >= static_cast<int>(cms_.size()) || cms_[k].size() == 0) {
                col++; continue;
            }
            double nf = (freq - (specs_[k].freq_left + specs_[k].freq_right) / 2.0)
                      / ((specs_[k].freq_right - specs_[k].freq_left) / 2.0);
            Complex sk(0, nf);
            auto Sk = CouplingMatrix::eval_S(cms_[k], sk);
            Fi(col, col) = Sk(1, 1);  // S22 = output reflection of filter k
            col++;
        }
        MatrixXcd I_dim = MatrixXcd::Identity(dim, dim);
        MatrixXcd M = I_dim - Fi * wi;
        if (std::abs(M.determinant()) > 1e-15) {
            Complex coupling = (vi.transpose() * M.inverse() * Fi * vi)(0, 0);
            L += coupling;
        }
    }

    // G11 = S11 + S12*S21*L / (1 - S22*L)
    Complex denom = Complex(1) - S(1, 1) * L;
    if (std::abs(denom) < 1e-15) return Complex(1e10);
    return S(0, 0) + S(0, 1) * S(1, 0) * L / denom;
}

std::vector<double> MultiplexerMatching::find_channel_peaks(
    int ch, const std::vector<double>& ifreqs) const
{
    std::vector<double> mag, freq;
    find_channel_peaks_xy(ch, ifreqs, mag, freq);
    return mag;
}

void MultiplexerMatching::find_channel_peaks_xy(
    int ch, const std::vector<double>& ifreqs,
    std::vector<double>& peaks_mag,
    std::vector<double>& peaks_freq) const
{
    double fl = specs_[ch].freq_left;
    double fr = specs_[ch].freq_right;
    int n = static_cast<int>(ifreqs.size());

    std::vector<double> boundaries;
    boundaries.push_back(fl);
    for (double f : ifreqs) boundaries.push_back(f);
    boundaries.push_back(fr);

    peaks_mag.assign(n + 1, 0.0);
    peaks_freq.assign(n + 1, 0.0);

    for (int seg = 0; seg <= n; ++seg) {
        double left = boundaries[seg];
        double right = boundaries[seg + 1];
        if (right - left < 1e-12) {
            peaks_mag[seg] = std::abs(eval_channel_response(ch, left));
            peaks_freq[seg] = left;
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

        peaks_mag[seg] = best_mag;
        peaks_freq[seg] = best_freq;
    }
}

bool MultiplexerMatching::solve_coupled(
    const std::vector<std::vector<double>>& interp_freqs)
{
    int N = static_cast<int>(specs_.size());

    try {
        MultiplexerNevanlinnaPick mux_np(manifold_, specs_, interp_freqs);

        VectorXcd x = mux_np.get_start_solution();

        // Newton (real-variable Wirtinger) to solve decoupled problem at λ=0.
        for (int iter = 0; iter < 100; ++iter) {
            VectorXcd F = mux_np.compute_residual(x, 0.0);
            double res0 = F.norm();
            if (res0 < 1e-8) break;

            VectorXcd dx = mux_np.newton_step(x, 0.0, F);

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

        // Coupled continuation
        bool ok = mux_np.run_continuation(x, 0.001, 2000, 30, 1e-6, false);
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
    // Per-channel sequential equiripple: cycle through channels, optimize one
    // at a time using Newton on that channel's peak-difference residual
    //   F[k] = peak[k] - peak[k+1]  for k = 0..n-1
    // (n equations, n unknowns = this channel's n interp freqs).
    // Line search insists on REDUCING the channel's worst peak — never accept
    // a step that makes it worse (common bug: unconditional last-resort accept).

    int N = static_cast<int>(specs_.size());
    auto best_cms = cms_;
    auto best_rls = achieved_rls_;
    auto best_freqs = interp_freqs;
    double best_worst_rl = *std::min_element(achieved_rls_.begin(), achieved_rls_.end());

    auto update_rls = [&]() {
        for (int i = 0; i < N; ++i) {
            double worst = 0;
            double fli = specs_[i].freq_left;
            double fri = specs_[i].freq_right;
            for (int k = 0; k < 201; ++k) {
                double freq = fli + (fri - fli) * k / 200.0;
                double mag = std::abs(eval_channel_response(i, freq));
                if (std::isfinite(mag) && mag > worst) worst = mag;
            }
            achieved_rls_[i] = to_db(worst);
        }
    };

    for (int outer = 0; outer < equiripple_outer_iterations; ++outer) {
        bool any_improved = false;

        for (int ch = 0; ch < N; ++ch) {
            int n = static_cast<int>(interp_freqs[ch].size());
            double fl = specs_[ch].freq_left;
            double fr = specs_[ch].freq_right;
            double bw = fr - fl;
            double fd_delta = 1e-5 * bw;

            auto peaks = find_channel_peaks(ch, interp_freqs[ch]);
            double ch_max = *std::max_element(peaks.begin(), peaks.end());
            double ch_min = *std::min_element(peaks.begin(), peaks.end());

            if (ch_max - ch_min < 1e-4 * ch_max) continue;

            Eigen::VectorXd F(n);
            for (int k = 0; k < n; ++k)
                F(k) = peaks[k] - peaks[k + 1];

            Eigen::MatrixXd J(n, n);
            bool jac_ok = true;
            for (int j = 0; j < n && jac_ok; ++j) {
                auto freqs_pert = interp_freqs;
                freqs_pert[ch][j] += fd_delta;

                if (!solve_coupled(freqs_pert)) {
                    freqs_pert = interp_freqs;
                    freqs_pert[ch][j] -= fd_delta;
                    fd_delta = -fd_delta;
                    if (!solve_coupled(freqs_pert)) { jac_ok = false; break; }
                }

                auto peaks_p = find_channel_peaks(ch, freqs_pert[ch]);
                for (int k = 0; k < n; ++k)
                    J(k, j) = ((peaks_p[k] - peaks_p[k+1]) - F(k)) / fd_delta;
            }

            if (!jac_ok) continue;

            if (!solve_coupled(interp_freqs)) break;

            Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-F);

            double max_step = 0.25 * bw;
            double scale = 1.0;
            for (int k = 0; k < n; ++k)
                if (std::abs(dx(k)) > max_step)
                    scale = std::min(scale, max_step / std::abs(dx(k)));
            dx *= scale;

            double alpha = 1.0;
            bool accepted = false;
            for (int ls = 0; ls < 10; ++ls) {
                auto freqs_new = interp_freqs;
                for (int k = 0; k < n; ++k)
                    freqs_new[ch][k] = std::clamp(
                        interp_freqs[ch][k] + alpha * dx(k),
                        fl + bw * 0.02, fr - bw * 0.02);
                std::sort(freqs_new[ch].begin(), freqs_new[ch].end());

                if (solve_coupled(freqs_new)) {
                    auto peaks_new = find_channel_peaks(ch, freqs_new[ch]);
                    double new_max = *std::max_element(peaks_new.begin(), peaks_new.end());

                    if (new_max < ch_max) {
                        interp_freqs = freqs_new;
                        update_rls();

                        double worst_rl = *std::min_element(
                            achieved_rls_.begin(), achieved_rls_.end());
                        if (worst_rl < best_worst_rl) {
                            best_worst_rl = worst_rl;
                            best_cms = cms_;
                            best_rls = achieved_rls_;
                            best_freqs = interp_freqs;
                        }
                        accepted = true;
                        any_improved = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }

            if (!accepted) solve_coupled(interp_freqs);

            if (verbose) {
                std::cout << "    Outer " << outer << " Ch" << (ch+1) << ":";
                for (int i = 0; i < N; ++i)
                    std::cout << " " << std::fixed << std::setprecision(1)
                              << achieved_rls_[i] << "dB";
                if (accepted)
                    std::cout << " a=" << std::setprecision(3) << alpha;
                else
                    std::cout << " (no improvement)";
                std::cout << "\n";
            }
        }

        if (!any_improved) {
            if (verbose) std::cout << "    No channel improved, stopping\n";
            break;
        }
    }

    cms_ = best_cms;
    achieved_rls_ = best_rls;
    interp_freqs = best_freqs;
}

} // namespace np
