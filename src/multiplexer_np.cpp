#include "multiplexer/multiplexer_np.hpp"
#include "impedance_matching.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

namespace np {

MultiplexerNevanlinnaPick::~MultiplexerNevanlinnaPick() = default;

MultiplexerNevanlinnaPick::MultiplexerNevanlinnaPick(
    const Manifold& manifold,
    const std::vector<ChannelSpec>& specs,
    const std::vector<std::vector<double>>& interp_freqs
)
    : num_channels_(static_cast<int>(specs.size()))
    , manifold_(&manifold)
{
    if (static_cast<int>(interp_freqs.size()) != num_channels_) {
        throw std::invalid_argument("MultiplexerNP: interp_freqs size mismatch");
    }

    total_vars_ = 0;
    total_eqs_ = 0;
    channels_.resize(num_channels_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        ch.order = specs[i].order;
        ch.num_vars = ch.order;  // F degree = order, monic drops 1: order free vars
        ch.var_offset = total_vars_;
        ch.eq_offset = total_eqs_;
        ch.interp_freqs = interp_freqs[i];

        // Frequency normalization to [-1, 1]
        ch.freq_center = (specs[i].freq_left + specs[i].freq_right) / 2.0;
        ch.freq_scale = (specs[i].freq_right - specs[i].freq_left) / 2.0;

        ch.norm_freqs.resize(ch.order);
        for (int m = 0; m < ch.order; ++m) {
            ch.norm_freqs[m] = (ch.interp_freqs[m] - ch.freq_center) / ch.freq_scale;
        }

        // Build Chebyshev filter for r_coeffs and initial prototype
        std::vector<Complex> shifted_tzs;
        for (const auto& tz : specs[i].transmission_zeros) {
            shifted_tzs.push_back((tz - ch.freq_center) / ch.freq_scale);
        }

        ChebyshevFilter ref_filter(ch.order, shifted_tzs, specs[i].return_loss_db);
        ch.r_coeffs = get_coeffs(ref_filter.P());

        // Chebyshev F polynomial as fallback start
        VectorXcd full_p = get_coeffs(ref_filter.F());
        ch.initial_p = full_p.tail(ch.order);  // monic: drop leading 1

        // Precompute init_sparams for potential use
        ch.init_sparams = eval_channel_map(i, to_monic(ch.initial_p));

        total_vars_ += ch.num_vars;
        total_eqs_ += ch.order;
    }
}

// --- Static helpers ---

VectorXcd MultiplexerNevanlinnaPick::to_monic(const VectorXcd& x) {
    VectorXcd result(x.size() + 1);
    result(0) = Complex(1, 0);
    result.tail(x.size()) = x;
    return result;
}

Polynomial<Complex> MultiplexerNevanlinnaPick::build_polynomial(const VectorXcd& coeffs) {
    std::vector<Complex> poly_coeffs(coeffs.size());
    for (int i = 0; i < coeffs.size(); ++i) {
        poly_coeffs[i] = coeffs(coeffs.size() - 1 - i);
    }
    return Polynomial<Complex>(poly_coeffs);
}

VectorXcd MultiplexerNevanlinnaPick::get_coeffs(const Polynomial<Complex>& p) {
    VectorXcd result(p.coefficients.size());
    for (size_t i = 0; i < p.coefficients.size(); ++i) {
        result(i) = p.coefficients[p.coefficients.size() - 1 - i];
    }
    return result;
}

std::vector<VectorXcd> MultiplexerNevanlinnaPick::split_variables(const VectorXcd& x) const {
    std::vector<VectorXcd> result(num_channels_);
    for (int i = 0; i < num_channels_; ++i) {
        result[i] = to_monic(x.segment(channels_[i].var_offset, channels_[i].num_vars));
    }
    return result;
}

// --- Per-channel evaluation ---

VectorXcd MultiplexerNevanlinnaPick::eval_channel_map(int ch, const VectorXcd& p_coeffs) const {
    auto r_poly = build_polynomial(channels_[ch].r_coeffs);
    auto p_poly = build_polynomial(p_coeffs);
    auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

    int M = channels_[ch].order;
    VectorXcd result(M);
    for (int m = 0; m < M; ++m) {
        Complex s(0, channels_[ch].norm_freqs[m]);
        Complex p_val = p_poly.evaluate(s);
        Complex q_val = q_poly.evaluate(s);
        if (std::abs(q_val) < 1e-15) {
            throw std::runtime_error("MultiplexerNP: spectral factor vanished");
        }
        result(m) = p_val / q_val;
    }
    return result;
}

MatrixXcd MultiplexerNevanlinnaPick::eval_channel_grad(int ch, const VectorXcd& p_coeffs) const {
    auto r_poly = build_polynomial(channels_[ch].r_coeffs);
    auto p_poly = build_polynomial(p_coeffs);
    auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

    int N = static_cast<int>(p_coeffs.size());
    int M = channels_[ch].order;

    std::vector<Complex> s(M), p_vals(M), q_vals(M);
    for (int k = 0; k < M; ++k) {
        s[k] = Complex(0, channels_[ch].norm_freqs[k]);
        p_vals[k] = p_poly.evaluate(s[k]);
        q_vals[k] = q_poly.evaluate(s[k]);
    }

    auto p_para = p_poly.para_conjugate();
    auto q_para = q_poly.para_conjugate();

    int q_deg = q_poly.degree();
    int lhs_max_deg = q_deg + (N - 1);
    int num_eqs = lhs_max_deg + 1;

    double max_imag_q = 0, max_real_q = 0;
    for (const auto& c : q_poly.coefficients) {
        max_imag_q = std::max(max_imag_q, std::abs(c.imag()));
        max_real_q = std::max(max_real_q, std::abs(c.real()));
    }
    bool q_is_real = max_imag_q < 1e-10 * std::max(1.0, max_real_q);

    MatrixXcd result(M, N);

    if (q_is_real) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, N);
        for (int k = 0; k < num_eqs; ++k) {
            double sign_i = 1.0;
            for (int i = 0; i < N; ++i) {
                int j = k - i;
                if (j >= 0 && j <= q_deg) {
                    Complex q_para_c = q_para.coefficients[j];
                    Complex q_c = q_poly.coefficients[j];
                    Complex coeff = q_para_c + q_c * sign_i;
                    A(2 * k, i) += coeff.real();
                    A(2 * k + 1, i) += coeff.imag();
                }
                sign_i = -sign_i;
            }
        }
        auto qr = A.colPivHouseholderQr();

        for (int j = 0; j < N; ++j) {
            int deg = N - 1 - j;
            std::vector<Complex> dp_c(deg + 1, Complex(0));
            dp_c[deg] = Complex(1);
            Polynomial<Complex> dp(dp_c);
            auto rhs_poly = dp * p_para + p_poly * dp.para_conjugate();

            int rhs_deg = rhs_poly.degree();
            Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);
            for (int k = 0; k < num_eqs; ++k) {
                Complex rc = (k <= rhs_deg) ? rhs_poly.coefficients[k] : Complex(0);
                b(2 * k) = rc.real();
                b(2 * k + 1) = rc.imag();
            }

            Eigen::VectorXd sol = qr.solve(b);
            std::vector<Complex> dq_c(N);
            for (int i = 0; i < N; ++i)
                dq_c[i] = Complex(sol(i), 0);
            Polynomial<Complex> dq(dq_c);

            for (int k = 0; k < M; ++k) {
                Complex dp_val = std::pow(s[k], deg);
                Complex dq_val = dq.evaluate(s[k]);
                result(k, j) = (dp_val * q_vals[k] - p_vals[k] * dq_val)
                             / (q_vals[k] * q_vals[k]);
            }
        }
    } else {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_eqs, 2 * N);
        for (int k = 0; k < num_eqs; ++k) {
            double sign = 1.0;
            for (int i = 0; i < N; ++i) {
                int j_idx = k - i;
                if (j_idx >= 0 && j_idx <= q_deg) {
                    Complex qp_j = q_para.coefficients[j_idx];
                    Complex q_j = q_poly.coefficients[j_idx];
                    A(2*k,   2*i)   += qp_j.real() + sign * q_j.real();
                    A(2*k,   2*i+1) += -qp_j.imag() + sign * q_j.imag();
                    A(2*k+1, 2*i)   += qp_j.imag() + sign * q_j.imag();
                    A(2*k+1, 2*i+1) += qp_j.real() - sign * q_j.real();
                }
                sign = -sign;
            }
        }
        auto qr = A.colPivHouseholderQr();

        for (int j = 0; j < N; ++j) {
            int deg = N - 1 - j;
            std::vector<Complex> dp_c(deg + 1, Complex(0));
            dp_c[deg] = Complex(1);
            Polynomial<Complex> dp(dp_c);
            auto rhs_poly = dp * p_para + p_poly * dp.para_conjugate();

            int rhs_deg = rhs_poly.degree();
            Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * num_eqs);
            for (int k = 0; k < num_eqs; ++k) {
                Complex rc = (k <= rhs_deg) ? rhs_poly.coefficients[k] : Complex(0);
                b(2 * k) = rc.real();
                b(2 * k + 1) = rc.imag();
            }

            Eigen::VectorXd sol = qr.solve(b);
            std::vector<Complex> dq_c(N);
            for (int i = 0; i < N; ++i)
                dq_c[i] = Complex(sol(2*i), sol(2*i+1));

            if (N > 0 && N <= static_cast<int>(q_poly.coefficients.size())) {
                double denom = q_poly.coefficients[N - 1].real();
                if (std::abs(denom) > 1e-15) {
                    double beta = dq_c[N - 1].imag() / denom;
                    for (int i = 0; i < N && i < static_cast<int>(q_poly.coefficients.size()); ++i)
                        dq_c[i] -= Complex(0, 1) * beta * q_poly.coefficients[i];
                }
            }
            Polynomial<Complex> dq(dq_c);

            for (int k = 0; k < M; ++k) {
                Complex dp_val = std::pow(s[k], deg);
                Complex dq_val = dq.evaluate(s[k]);
                result(k, j) = (dp_val * q_vals[k] - p_vals[k] * dq_val)
                             / (q_vals[k] * q_vals[k]);
            }
        }
    }

    return result;
}

// --- Coupled load ---

Complex MultiplexerNevanlinnaPick::eval_coupled_load(
    int ch, double freq_phys, double lambda,
    const std::vector<VectorXcd>& all_p_coeffs) const
{
    // L_i(P, lambda) = J_ii + lambda * V_i^T [I - F_i W_i]^{-1} F_i V_i
    auto J = manifold_->compute_J(freq_phys);
    int port = ch + 1;
    Complex J_ii = J(port, port);

    if (std::abs(lambda) < 1e-15 || num_channels_ < 2) {
        return J_ii;
    }

    int N = num_channels_;
    VectorXcd Vi = manifold_->compute_Vi(J, ch);
    MatrixXcd Wi = manifold_->compute_Wi(J, ch);

    int dim = N - 1;
    MatrixXcd Fi = MatrixXcd::Zero(dim, dim);

    int col = 0;
    for (int k = 0; k < N; ++k) {
        if (k == ch) continue;

        double norm_freq_k = (freq_phys - channels_[k].freq_center) / channels_[k].freq_scale;
        auto r_poly = build_polynomial(channels_[k].r_coeffs);
        auto p_poly = build_polynomial(all_p_coeffs[k]);
        auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

        Complex s(0, norm_freq_k);
        Complex q_val = q_poly.evaluate(s);
        Complex f_k = (std::abs(q_val) > 1e-15)
            ? p_poly.evaluate(s) / q_val
            : Complex(0);
        Fi(col, col) = f_k;
        col++;
    }

    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd M = I - Fi * Wi;
    MatrixXcd M_inv = M.inverse();
    Complex coupling = (Vi.adjoint() * M_inv * Fi * Vi)(0, 0);

    return J_ii + lambda * coupling;
}

// --- HomotopyBase interface (kept for compatibility but not used by new continuation) ---

VectorXcd MultiplexerNevanlinnaPick::calc_homotopy_function(const VectorXcd& x, double t) {
    // Not used by the new Euler continuation.
    // Kept for interface compatibility.
    return VectorXcd::Zero(total_eqs_);
}

VectorXcd MultiplexerNevanlinnaPick::calc_path_derivative(const VectorXcd& x, double t) {
    return VectorXcd::Zero(total_eqs_);
}

MatrixXcd MultiplexerNevanlinnaPick::calc_jacobian(const VectorXcd& x, double t) {
    return MatrixXcd::Zero(total_eqs_, total_vars_);
}

VectorXcd MultiplexerNevanlinnaPick::get_start_solution() {
    VectorXcd x0(total_vars_);
    for (int i = 0; i < num_channels_; ++i) {
        x0.segment(channels_[i].var_offset, channels_[i].num_vars) = channels_[i].initial_p;
    }
    return x0;
}

int MultiplexerNevanlinnaPick::get_num_variables() {
    return total_vars_;
}

// --- Martinez continuation: Euler predictor + Newton corrector ---

VectorXcd MultiplexerNevanlinnaPick::compute_residual(
    const VectorXcd& x, double lambda) const
{
    // F_i,m(P, lambda) = f(p_i)[xi_m] - L_i(P, lambda)[xi_m]
    // Martinez: f(p_i) = L_i (direct equality, no conjugate)
    auto all_p = split_variables(x);
    VectorXcd F(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        VectorXcd fi = eval_channel_map(i, all_p[i]);

        for (int m = 0; m < ch.order; ++m) {
            Complex L_i = eval_coupled_load(i, ch.interp_freqs[m], lambda, all_p);
            F(ch.eq_offset + m) = fi(m) - L_i;
        }
    }
    return F;
}

MatrixXcd MultiplexerNevanlinnaPick::compute_jacobian_dp(
    const VectorXcd& x, double lambda) const
{
    // Analytical Jacobian following the C# implementation:
    // J = dMatch/dp - lambda * conj(dG/dp)
    //
    // Diagonal blocks: dMatch_i/dp_i = d(f(p_i))/dp_i  (Bezout-based gradient)
    // Off-diagonal: -lambda * conj(dL_i/dp_j)
    //   where dL_i/dp_j = v^T · S · dF/dp_j · (W·S·F + I) · v
    //   S = (I - F·W)^{-1}

    auto all_p = split_variables(x);
    MatrixXcd Jac = MatrixXcd::Zero(total_eqs_, total_vars_);

    // Precompute per-channel gradients: d(f(p_k))/dp_k at ALL frequencies
    // grads[k] is (total_freq_count x (order_k + 1)) — gradient of p_k/q_k
    // We need gradients evaluated at OTHER channels' frequencies too.
    struct ChannelGradInfo {
        // Per-channel f(p_k) and df/dp_k at each frequency across all channels
        std::vector<Complex> f_vals;          // f(p_k) at each freq
        std::vector<VectorXcd> df_dp_vals;    // df/dp_k at each freq (size num_vars_k)
    };

    // For each channel k, evaluate f(p_k) and df/dp_k at all required frequencies
    std::vector<ChannelGradInfo> grad_info(num_channels_);

    for (int k = 0; k < num_channels_; ++k) {
        auto& ch_k = channels_[k];
        auto r_poly = build_polynomial(ch_k.r_coeffs);
        auto p_poly = build_polynomial(all_p[k]);
        auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

        // Collect all unique physical frequencies where we need channel k evaluated
        // (from all other channels' interpolation frequencies, plus own)
        // For simplicity, evaluate at each channel i's frequencies
        for (int i = 0; i < num_channels_; ++i) {
            auto& ch_i = channels_[i];
            for (int m = 0; m < ch_i.order; ++m) {
                double phys_freq = ch_i.interp_freqs[m];
                double norm_freq_k = (phys_freq - ch_k.freq_center) / ch_k.freq_scale;
                Complex s(0, norm_freq_k);
                Complex p_val = p_poly.evaluate(s);
                Complex q_val = q_poly.evaluate(s);
                Complex f_val = (std::abs(q_val) > 1e-15) ? p_val / q_val : Complex(0);
                grad_info[k].f_vals.push_back(f_val);
            }
        }
    }

    // Diagonal blocks: dMatch_i/dp_i
    for (int i = 0; i < num_channels_; ++i) {
        auto& ch_i = channels_[i];
        MatrixXcd full_grad = eval_channel_grad(i, all_p[i]);
        Jac.block(ch_i.eq_offset, ch_i.var_offset,
                  ch_i.order, ch_i.num_vars) = full_grad.rightCols(ch_i.num_vars);
    }

    // Off-diagonal blocks: analytical dG/dp
    if (std::abs(lambda) > 1e-15) {
        for (int i = 0; i < num_channels_; ++i) {
            auto& ch_i = channels_[i];

            for (int m = 0; m < ch_i.order; ++m) {
                double phys_freq = ch_i.interp_freqs[m];
                auto J_mfold = manifold_->compute_J(phys_freq);
                VectorXcd vi = manifold_->compute_Vi(J_mfold, i);
                MatrixXcd wi = manifold_->compute_Wi(J_mfold, i);

                int dim = num_channels_ - 1;
                MatrixXcd Fi = MatrixXcd::Zero(dim, dim);

                // Build F_i diagonal matrix
                int col = 0;
                for (int k = 0; k < num_channels_; ++k) {
                    if (k == i) continue;
                    double norm_freq_k = (phys_freq - channels_[k].freq_center) / channels_[k].freq_scale;
                    auto r_poly = build_polynomial(channels_[k].r_coeffs);
                    auto p_poly = build_polynomial(all_p[k]);
                    auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);
                    Complex s(0, norm_freq_k);
                    Complex q_val = q_poly.evaluate(s);
                    Fi(col, col) = (std::abs(q_val) > 1e-15)
                        ? p_poly.evaluate(s) / q_val : Complex(0);
                    col++;
                }

                // S = (I - F·W)^{-1}
                MatrixXcd I_dim = MatrixXcd::Identity(dim, dim);
                MatrixXcd S = (I_dim - Fi * wi).inverse();

                // For each channel j != i, compute dL_i/dp_j
                for (int j = 0; j < num_channels_; ++j) {
                    if (j == i) continue;
                    auto& ch_j = channels_[j];

                    // Index of channel j within the F_i matrix (excluding channel i)
                    int j_in_Fi = 0;
                    for (int kk = 0; kk < j; ++kk) {
                        if (kk != i) j_in_Fi++;
                    }

                    // Compute per-channel gradient df_j/dp_j at this frequency
                    double norm_freq_j = (phys_freq - ch_j.freq_center) / ch_j.freq_scale;

                    // We need df_j/dp_j_coeff for each free coefficient of channel j
                    // eval_channel_grad gives gradient at channel j's own interp freqs,
                    // but we need it at channel i's frequency. Compute directly.
                    auto r_poly_j = build_polynomial(ch_j.r_coeffs);
                    auto p_poly_j = build_polynomial(all_p[j]);
                    auto q_poly_j = SpectralFactor::feldtkeller(p_poly_j, r_poly_j);

                    Complex s_j(0, norm_freq_j);
                    Complex p_val_j = p_poly_j.evaluate(s_j);
                    Complex q_val_j = q_poly_j.evaluate(s_j);

                    int N_j = static_cast<int>(all_p[j].size());

                    for (int v = 0; v < ch_j.num_vars; ++v) {
                        // df_j/dp_j[v]: derivative of f_j = p_j/q_j w.r.t. the v-th
                        // free coefficient (v+1 in monic indexing, degree N_j-1-(v+1))
                        int coeff_idx = v + 1;  // skip leading 1
                        int deg = N_j - 1 - coeff_idx;

                        // dp/dp_v = s^deg (monomial)
                        Complex dp_val = std::pow(s_j, deg);

                        // dq/dp_v: solve Bezout for this single perturbation
                        // For efficiency, use the quotient rule approximation:
                        // df/dp_v ≈ (dp_val * q - p * dq_val) / q^2
                        // We need dq_val. Use FD on the spectral factor.
                        auto p_pert_coeffs = all_p[j];
                        p_pert_coeffs(coeff_idx) += 1e-7;
                        auto p_pert_poly = build_polynomial(p_pert_coeffs);
                        auto q_pert_poly = SpectralFactor::feldtkeller(p_pert_poly, r_poly_j);
                        Complex dq_val = (q_pert_poly.evaluate(s_j) - q_val_j) / 1e-7;

                        Complex df_j_dpv = (dp_val * q_val_j - p_val_j * dq_val)
                                         / (q_val_j * q_val_j);

                        // dF/dp_j[v]: diagonal matrix with df_j_dpv at position (j_in_Fi, j_in_Fi)
                        MatrixXcd dF = MatrixXcd::Zero(dim, dim);
                        dF(j_in_Fi, j_in_Fi) = df_j_dpv;

                        // dL_i/dp_j[v] = v^T · S · dF · (W·S·F + I) · v
                        MatrixXcd WSF_I = wi * S * Fi + I_dim;
                        Complex dL = (vi.adjoint() * S * dF * WSF_I * vi)(0, 0);

                        Jac(ch_i.eq_offset + m, ch_j.var_offset + v) = -lambda * std::conj(dL);
                    }
                }
            }
        }
    }

    return Jac;
}

VectorXcd MultiplexerNevanlinnaPick::compute_dF_dlambda(
    const VectorXcd& x, double lambda) const
{
    // dF/dlambda = -dL_i/dlambda = -(coupling_term_i)
    auto all_p = split_variables(x);
    VectorXcd dFdl(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        for (int m = 0; m < ch.order; ++m) {
            Complex L_full = eval_coupled_load(i, ch.interp_freqs[m], 1.0, all_p);
            auto J_mat = manifold_->compute_J(ch.interp_freqs[m]);
            Complex J_ii = J_mat(i + 1, i + 1);
            dFdl(ch.eq_offset + m) = -(L_full - J_ii);
        }
    }
    return dFdl;
}

bool MultiplexerNevanlinnaPick::run_continuation(
    VectorXcd& x, double lambda_step, int max_steps,
    int newton_max_iter, double newton_tol, bool verbose)
{
    double lambda = 0.0;
    double h = lambda_step;

    for (int step = 0; step < max_steps && lambda < 1.0 - 1e-12; ++step) {
        // Clamp to not overshoot
        if (lambda + h > 1.0) h = 1.0 - lambda;

        // Euler predictor: dx = -(dF/dp)^{-1} * (dF/dlambda) * h
        MatrixXcd Jp = compute_jacobian_dp(x, lambda);
        VectorXcd dFdl = compute_dF_dlambda(x, lambda);
        VectorXcd dx = Jp.colPivHouseholderQr().solve(-dFdl * h);

        VectorXcd x_pred = x + dx;
        double new_lambda = lambda + h;

        // Damped Newton corrector at new lambda
        bool converged = false;
        for (int iter = 0; iter < newton_max_iter; ++iter) {
            VectorXcd F = compute_residual(x_pred, new_lambda);
            double res = F.norm();

            if (verbose && step < 5) {
                std::cout << "      Newton iter " << iter << ": res=" << res << "\n";
            }

            if (res < newton_tol) {
                converged = true;
                break;
            }

            MatrixXcd Jp_new = compute_jacobian_dp(x_pred, new_lambda);
            VectorXcd correction = Jp_new.colPivHouseholderQr().solve(-F);

            // Backtracking line search
            double alpha = 1.0;
            for (int ls = 0; ls < 10; ++ls) {
                VectorXcd x_try = x_pred + alpha * correction;
                double res_try = compute_residual(x_try, new_lambda).norm();
                if (res_try < res) {
                    x_pred = x_try;
                    break;
                }
                alpha *= 0.5;
            }

            if (alpha < 1e-8) break;
        }

        if (converged) {
            x = x_pred;
            lambda = new_lambda;

            if (verbose && (step % 20 == 0 || lambda >= 1.0 - 1e-12)) {
                VectorXcd F = compute_residual(x, lambda);
                std::cout << "    lambda=" << std::fixed << std::setprecision(4) << lambda
                          << " residual=" << std::scientific << std::setprecision(2)
                          << F.norm() << " h=" << h << "\n";
            }

            h = std::min(h * 1.5, 0.1);
        } else {
            h *= 0.5;
            if (h < 1e-12) {
                if (verbose) {
                    // Show what's happening at the stall point
                    VectorXcd F = compute_residual(x, lambda);
                    std::cout << "    Stalled at lambda=" << std::fixed << std::setprecision(6)
                              << lambda << " residual=" << F.norm() << " h=" << h << "\n";
                    VectorXcd F_test = compute_residual(x, lambda + 0.001);
                    std::cout << "    Residual at lambda+" << 0.001 << ": " << F_test.norm() << "\n";
                }
                return false;
            }
        }
    }

    return lambda >= 1.0 - 1e-12;
}

// --- Coupling matrix extraction ---

std::vector<MatrixXcd> MultiplexerNevanlinnaPick::extract_coupling_matrices(
    const VectorXcd& x) const
{
    auto all_p = split_variables(x);
    std::vector<MatrixXcd> cms(num_channels_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        auto r_poly = build_polynomial(ch.r_coeffs);
        auto p_poly = build_polynomial(all_p[i]);

        auto r_shifted = r_poly.shift(-ch.freq_center / ch.freq_scale);
        auto p_shifted = p_poly.shift(-ch.freq_center / ch.freq_scale);
        auto e_poly = SpectralFactor::feldtkeller(p_shifted, r_shifted);

        cms[i] = CouplingMatrix::from_polynomials_by_realization(p_shifted, r_shifted, e_poly);
    }

    return cms;
}

} // namespace np
