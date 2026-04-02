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
    // Newton on per-channel peak-difference system, same approach as single-channel.
    // Variables: all interpolation frequencies (concatenated).
    // Residual per channel: F_ch = [h0-h1, h1-h2, ..., h(n-1)-hn] where h_k = peak in interval k.
    // FD Jacobian: perturb each frequency, full re-solve, recompute peaks.

    int N = static_cast<int>(specs_.size());

    // Flatten frequency vector and build index map
    int total_freqs = 0;
    std::vector<int> ch_offset(N), ch_nfreqs(N);
    for (int ch = 0; ch < N; ++ch) {
        ch_offset[ch] = total_freqs;
        ch_nfreqs[ch] = static_cast<int>(interp_freqs[ch].size());
        total_freqs += ch_nfreqs[ch];
    }

    // Helper: flatten interp_freqs to vector
    auto flatten = [&](const std::vector<std::vector<double>>& freqs) {
        Eigen::VectorXd v(total_freqs);
        for (int ch = 0; ch < N; ++ch)
            for (int k = 0; k < ch_nfreqs[ch]; ++k)
                v(ch_offset[ch] + k) = freqs[ch][k];
        return v;
    };

    // Helper: unflatten vector back to interp_freqs
    auto unflatten = [&](const Eigen::VectorXd& v) {
        std::vector<std::vector<double>> freqs(N);
        for (int ch = 0; ch < N; ++ch) {
            freqs[ch].resize(ch_nfreqs[ch]);
            for (int k = 0; k < ch_nfreqs[ch]; ++k)
                freqs[ch][k] = v(ch_offset[ch] + k);
        }
        return freqs;
    };

    // Helper: compute per-channel peaks and form residual
    // Returns (residual_vector, max_peak_across_all_channels)
    auto compute_peaks_and_residual = [&](double& max_peak_out) {
        Eigen::VectorXd F(total_freqs);
        max_peak_out = 0;
        for (int ch = 0; ch < N; ++ch) {
            auto peaks = find_channel_peaks(ch, interp_freqs[ch]);
            for (int k = 0; k < ch_nfreqs[ch]; ++k)
                F(ch_offset[ch] + k) = peaks[k] - peaks[k + 1];
            double ch_max = *std::max_element(peaks.begin(), peaks.end());
            if (ch_max > max_peak_out) max_peak_out = ch_max;
        }
        return F;
    };

    // Initial state
    auto best_cms = cms_;
    auto best_rls = achieved_rls_;
    auto best_freqs = interp_freqs;

    double max_peak = 0;
    Eigen::VectorXd F = compute_peaks_and_residual(max_peak);
    double best_max_peak = max_peak;

    if (verbose) {
        for (int ch = 0; ch < N; ++ch) {
            auto peaks = find_channel_peaks(ch, interp_freqs[ch]);
            double mx = *std::max_element(peaks.begin(), peaks.end());
            double mn = *std::min_element(peaks.begin(), peaks.end());
            std::cout << "    Ch" << (ch+1) << ": worst=" << std::fixed
                      << std::setprecision(1) << to_db(mx) << " dB, spread="
                      << (to_db(mx) - to_db(mn)) << " dB\n";
        }
    }

    for (int iter = 0; iter < equiripple_outer_iterations; ++iter) {
        // FD Jacobian: perturb each frequency, full re-solve, recompute peaks
        Eigen::MatrixXd J(total_freqs, total_freqs);
        bool jac_ok = true;

        for (int j = 0; j < total_freqs && jac_ok; ++j) {
            int ch_j = 0;
            while (ch_j < N - 1 && j >= ch_offset[ch_j + 1]) ch_j++;
            double bw = specs_[ch_j].freq_right - specs_[ch_j].freq_left;
            double fd_delta = 1e-5 * bw;

            auto freqs_pert = interp_freqs;
            int k_in_ch = j - ch_offset[ch_j];
            freqs_pert[ch_j][k_in_ch] += fd_delta;

            if (!solve_coupled(freqs_pert)) {
                // Try negative perturbation
                freqs_pert = interp_freqs;
                freqs_pert[ch_j][k_in_ch] -= fd_delta;
                fd_delta = -fd_delta;
                if (!solve_coupled(freqs_pert)) {
                    if (verbose) std::cout << "    FD solve failed for freq " << j << "\n";
                    jac_ok = false;
                    break;
                }
            }

            double dummy;
            // Temporarily swap in perturbed CMs for peak evaluation
            auto saved_cms = cms_;
            // cms_ was updated by solve_coupled
            Eigen::VectorXd F_pert(total_freqs);
            for (int ch = 0; ch < N; ++ch) {
                auto peaks_p = find_channel_peaks(ch, freqs_pert[ch]);
                for (int k = 0; k < ch_nfreqs[ch]; ++k)
                    F_pert(ch_offset[ch] + k) = peaks_p[k] - peaks_p[k + 1];
            }
            cms_ = saved_cms;  // restore

            for (int i = 0; i < total_freqs; ++i)
                J(i, j) = (F_pert(i) - F(i)) / fd_delta;
        }

        if (!jac_ok) break;

        // Restore coupling matrices from best/current state
        if (!solve_coupled(interp_freqs)) break;

        // Newton step
        Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-F);

        // Clamp step
        double max_step = 0;
        for (int ch = 0; ch < N; ++ch) {
            double bw = specs_[ch].freq_right - specs_[ch].freq_left;
            if (bw > max_step) max_step = bw;
        }
        max_step *= 0.25;
        double scale = 1.0;
        for (int i = 0; i < total_freqs; ++i)
            if (std::abs(dx(i)) > max_step)
                scale = std::min(scale, max_step / std::abs(dx(i)));
        dx *= scale;

        // Line search on max(peak)
        double alpha = 1.0;
        bool accepted = false;
        for (int ls = 0; ls < 8; ++ls) {
            Eigen::VectorXd v = flatten(interp_freqs) + alpha * dx;
            auto freqs_new = unflatten(v);

            // Clamp to passband
            for (int ch = 0; ch < N; ++ch) {
                double fl = specs_[ch].freq_left;
                double fr = specs_[ch].freq_right;
                double bw = fr - fl;
                for (auto& f : freqs_new[ch])
                    f = std::clamp(f, fl + bw * 0.02, fr - bw * 0.02);
                std::sort(freqs_new[ch].begin(), freqs_new[ch].end());
            }

            if (solve_coupled(freqs_new)) {
                double new_max = 0;
                for (int ch = 0; ch < N; ++ch) {
                    auto peaks = find_channel_peaks(ch, freqs_new[ch]);
                    double mx = *std::max_element(peaks.begin(), peaks.end());
                    if (mx > new_max) new_max = mx;
                }

                if (new_max < max_peak || ls == 7) {
                    interp_freqs = freqs_new;
                    max_peak = new_max;
                    F = compute_peaks_and_residual(max_peak);
                    accepted = true;

                    if (new_max < best_max_peak) {
                        best_max_peak = new_max;
                        best_cms = cms_;
                        best_rls = achieved_rls_;
                        best_freqs = interp_freqs;
                    }
                    break;
                }
            }
            alpha *= 0.5;
        }

        if (verbose) {
            std::cout << "    Iter " << iter << ": max|G|="
                      << std::fixed << std::setprecision(1) << to_db(max_peak) << " dB";
            for (int i = 0; i < N; ++i)
                std::cout << " Ch" << (i+1) << "=" << achieved_rls_[i] << "dB";
            std::cout << " alpha=" << std::setprecision(3) << alpha << "\n";
        }

        if (!accepted) {
            if (verbose) std::cout << "    Line search failed\n";
            break;
        }
    }

    cms_ = best_cms;
    achieved_rls_ = best_rls;
    interp_freqs = best_freqs;
}

} // namespace np
