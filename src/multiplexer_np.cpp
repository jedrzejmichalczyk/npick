#include "multiplexer/multiplexer_np.hpp"
#include "impedance_matching.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
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
        ch.num_vars = ch.order - 1;  // monic
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

        // Build transmission polynomial from specs
        std::vector<Complex> shifted_tzs;
        for (const auto& tz : specs[i].transmission_zeros) {
            shifted_tzs.push_back((tz - ch.freq_center) / ch.freq_scale);
        }

        ChebyshevFilter ref_filter(ch.order, shifted_tzs, specs[i].return_loss_db);
        ch.r_coeffs = get_coeffs(ref_filter.P());

        total_vars_ += ch.num_vars;
        total_eqs_ += ch.order;
    }

    // Compute start solutions: independent NP solve per channel
    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];

        // Load for independent solve: conj(J_ii) at each interpolation frequency
        auto load_fn = [this, i](double norm_freq) -> Complex {
            auto& ch_data = channels_[i];
            double phys_freq = norm_freq * ch_data.freq_scale + ch_data.freq_center;
            auto J = manifold_->compute_J(phys_freq);
            int port = i + 1;
            return J(port, port);
        };

        std::vector<Complex> target_loads(ch.order);
        for (int m = 0; m < ch.order; ++m) {
            target_loads[m] = load_fn(ch.norm_freqs[m]);
        }

        // Use the existing NP solver for independent matching
        try {
            std::vector<Complex> shifted_tzs;
            for (const auto& tz : specs[i].transmission_zeros) {
                shifted_tzs.push_back((tz - ch.freq_center) / ch.freq_scale);
            }

            NevanlinnaPickNormalized np_single(
                target_loads, ch.norm_freqs, shifted_tzs, specs[i].return_loss_db);

            PathTracker tracker(&np_single);
            tracker.h = -0.02;
            tracker.run_tol = 1e-3;
            tracker.final_tol = 1e-7;
            tracker.verbose = false;

            VectorXcd x0 = np_single.get_start_solution();
            VectorXcd sol = tracker.run(x0);
            ch.initial_p = sol;  // reduced (monic) coefficients

        } catch (const std::exception& e) {
            // Fallback: use Chebyshev prototype as start
            std::vector<Complex> shifted_tzs;
            for (const auto& tz : specs[i].transmission_zeros) {
                shifted_tzs.push_back((tz - ch.freq_center) / ch.freq_scale);
            }
            ChebyshevFilter cheb(ch.order, shifted_tzs, specs[i].return_loss_db);
            VectorXcd full = get_coeffs(cheb.F());
            ch.initial_p = full.tail(full.size() - 1);  // drop leading coeff
        }
    }
}

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
    // Replicates NevanlinnaPick::eval_grad for a single channel
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

    // Check if q is real
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
        // Complex q case - full 2N unknowns
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

Complex MultiplexerNevanlinnaPick::eval_coupled_load(
    int ch, double freq_phys, double lambda,
    const std::vector<VectorXcd>& all_p_coeffs) const
{
    // L_i(P, lambda) = J_ii + lambda * V_i^T [I - F_i W_i]^{-1} F_i V_i
    auto J = manifold_->compute_J(freq_phys);
    int port = ch + 1;
    Complex J_ii = J(port, port);

    if (std::abs(lambda) < 1e-15) {
        return J_ii;
    }

    int N = num_channels_;
    VectorXcd Vi = manifold_->compute_Vi(J, ch);
    MatrixXcd Wi = manifold_->compute_Wi(J, ch);

    // Build F_i = diag(f(p_k)) for k != ch
    int dim = N - 1;
    MatrixXcd Fi = MatrixXcd::Zero(dim, dim);

    int col = 0;
    for (int k = 0; k < N; ++k) {
        if (k == ch) continue;

        // Evaluate f(p_k) at this frequency (in channel k's normalized domain)
        double norm_freq_k = (freq_phys - channels_[k].freq_center) / channels_[k].freq_scale;
        auto r_poly = build_polynomial(channels_[k].r_coeffs);
        auto p_poly = build_polynomial(all_p_coeffs[k]);
        auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

        Complex s(0, norm_freq_k);
        Complex f_k = p_poly.evaluate(s) / q_poly.evaluate(s);
        Fi(col, col) = f_k;
        col++;
    }

    // M = I - F_i * W_i
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd M = I - Fi * Wi;

    // L_i = J_ii + lambda * V_i^T * M^{-1} * F_i * V_i
    MatrixXcd M_inv = M.inverse();
    Complex coupling = Vi.transpose() * M_inv * Fi * Vi;

    return J_ii + lambda * coupling;
}

VectorXcd MultiplexerNevanlinnaPick::calc_homotopy_function(const VectorXcd& x, double t) {
    double lambda = 1.0 - t;

    auto all_p = split_variables(x);  // monic coefficients per channel
    VectorXcd H(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];

        // f(p_i) at each interpolation frequency
        VectorXcd fi = eval_channel_map(i, all_p[i]);

        for (int m = 0; m < ch.order; ++m) {
            Complex L_i = eval_coupled_load(i, ch.interp_freqs[m], lambda, all_p);
            H(ch.eq_offset + m) = fi(m) - std::conj(L_i);
        }
    }

    return H;
}

VectorXcd MultiplexerNevanlinnaPick::calc_path_derivative(const VectorXcd& x, double t) {
    // dH/dt = -dH/dlambda since lambda = 1 - t
    // dH_{i,m}/dlambda = -d(conj(L_i))/dlambda
    // L_i(lambda) = J_ii + lambda * coupling_term
    // dL_i/dlambda = coupling_term
    // dH/dt = -(-conj(coupling_term)) = conj(coupling_term)

    double lambda = 1.0 - t;
    auto all_p = split_variables(x);
    VectorXcd dHdt(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];

        for (int m = 0; m < ch.order; ++m) {
            // coupling_term = V_i^T [I - F_i W_i]^{-1} F_i V_i
            Complex L_full = eval_coupled_load(i, ch.interp_freqs[m], 1.0, all_p);
            auto J = manifold_->compute_J(ch.interp_freqs[m]);
            Complex J_ii = J(i + 1, i + 1);
            Complex coupling_term = L_full - J_ii;

            // dH/dt = conj(coupling_term)  (since dH/dt = -d/dlambda[-conj(L)])
            dHdt(ch.eq_offset + m) = std::conj(coupling_term);
        }
    }

    return dHdt;
}

MatrixXcd MultiplexerNevanlinnaPick::calc_jacobian(const VectorXcd& x, double t) {
    double lambda = 1.0 - t;
    auto all_p = split_variables(x);

    MatrixXcd J_mat = MatrixXcd::Zero(total_eqs_, total_vars_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch_i = channels_[i];

        // Diagonal block: dH_i / dp_i = d(f(p_i))/dp_i
        // (L_i does NOT depend on p_i since F_i excludes channel i)
        MatrixXcd full_grad = eval_channel_grad(i, all_p[i]);
        // full_grad is (order_i x order_i), we need (order_i x (order_i - 1))
        // since p_i is monic, drop first column (derivative w.r.t. leading coeff = 1)
        MatrixXcd diag_block = full_grad.rightCols(full_grad.cols() - 1);

        J_mat.block(ch_i.eq_offset, ch_i.var_offset,
                    ch_i.order, ch_i.num_vars) = diag_block;

        // Off-diagonal blocks: dH_i / dp_j = -conj(dL_i/dp_j)
        if (std::abs(lambda) > 1e-15) {
            for (int j = 0; j < num_channels_; ++j) {
                if (j == i) continue;
                auto& ch_j = channels_[j];

                // Compute dL_i/dp_j via finite differences for now
                // (analytical version requires chain rule through M^{-1})
                double fd_delta = 1e-7;
                for (int v = 0; v < ch_j.num_vars; ++v) {
                    auto all_p_pert = all_p;
                    // Perturb the v-th free coefficient of channel j
                    // (index v+1 in monic representation since index 0 is the leading 1)
                    all_p_pert[j](v + 1) += fd_delta;

                    for (int m = 0; m < ch_i.order; ++m) {
                        Complex L_pert = eval_coupled_load(
                            i, ch_i.interp_freqs[m], lambda, all_p_pert);
                        Complex L_base = eval_coupled_load(
                            i, ch_i.interp_freqs[m], lambda, all_p);
                        Complex dL = (L_pert - L_base) / fd_delta;
                        J_mat(ch_i.eq_offset + m, ch_j.var_offset + v) = -std::conj(dL);
                    }
                }
            }
        }
    }

    return J_mat;
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

std::vector<MatrixXcd> MultiplexerNevanlinnaPick::extract_coupling_matrices(
    const VectorXcd& x) const
{
    auto all_p = split_variables(x);
    std::vector<MatrixXcd> cms(num_channels_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        auto r_poly = build_polynomial(ch.r_coeffs);
        auto p_poly = build_polynomial(all_p[i]);

        // Shift back to original frequency domain
        auto r_shifted = r_poly.shift(-ch.freq_center / ch.freq_scale);
        auto p_shifted = p_poly.shift(-ch.freq_center / ch.freq_scale);
        auto e_poly = SpectralFactor::feldtkeller(p_shifted, r_shifted);

        cms[i] = CouplingMatrix::from_polynomials_by_realization(p_shifted, r_shifted, e_poly);
    }

    return cms;
}

} // namespace np
