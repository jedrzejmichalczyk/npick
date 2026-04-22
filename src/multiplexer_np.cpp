#include "multiplexer/multiplexer_np.hpp"
#include "impedance_matching.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>

namespace np {

namespace {

// Convert descending VectorXcd to ascending Polynomial<Complex>
Polynomial<Complex> desc_to_poly(const VectorXcd& desc) {
    std::vector<Complex> asc(desc.size());
    for (int i = 0; i < desc.size(); ++i)
        asc[i] = desc(desc.size() - 1 - i);
    return Polynomial<Complex>(asc);
}

// Precomputed polynomial state for one channel.
// Caches spectral factor q and, when requested, BOTH Wirtinger derivatives
// of q w.r.t. each free complex coefficient p_v:
//   dq_polys[v]      = ∂q/∂p_v        (holomorphic)
//   dq_conj_polys[v] = ∂q/∂conj(p_v)  (antiholomorphic)
//
// Why both are needed: the Feldtkeller identity q·q_para = p·p_para + r·r_para
// has p_para on the RHS, whose coefficients are conj(p_k) up to sign. So q
// depends on BOTH p and conj(p). Assuming ∂q/∂conj(p_v) = 0 is only correct
// for real-coefficient p, and silently biases Newton when p drifts complex
// along the homotopy (Feldtkeller residual stalls at ~1e-5, step halving
// death-spirals near λ=1). Solving the coupled system below is the general
// fix — no real/complex-q branch.
struct ChannelPolyState {
    Polynomial<Complex> p_poly;
    Polynomial<Complex> q_poly;

    // Populated only when with_derivatives=true.
    std::vector<Polynomial<Complex>> dq_polys;       // ∂q/∂p_v
    std::vector<Polynomial<Complex>> dq_conj_polys;  // ∂q/∂conj(p_v)
    std::vector<int> dp_degrees;                     // s-degree of the varied coefficient

    // Compute s^deg without std::pow, which returns NaN for (0, 0) under
    // emscripten/WASM's libm even though native libm returns (1, 0). Odd
    // filter orders put the middle Chebyshev node exactly at the band
    // center (normalized freq = 0), so this corner case is always hit.
    static Complex s_pow(Complex s, int deg) {
        if (deg == 0) return Complex(1, 0);
        Complex result = s;
        for (int i = 1; i < deg; ++i) result *= s;
        return result;
    }

    Complex eval_f(Complex s) const {
        Complex q_val = q_poly.evaluate(s);
        return (std::abs(q_val) > 1e-15) ? p_poly.evaluate(s) / q_val : Complex(0);
    }

    // ∂f/∂p_v  where f = p/q:  (mon_v·q − p·dq) / q²
    Complex eval_df(int v, Complex s) const {
        Complex p_val = p_poly.evaluate(s);
        Complex q_val = q_poly.evaluate(s);
        if (std::abs(q_val) < 1e-15) return Complex(0);
        Complex dp_val = s_pow(s, dp_degrees[v]);
        Complex dq_val = dq_polys[v].evaluate(s);
        return (dp_val * q_val - p_val * dq_val) / (q_val * q_val);
    }

    // ∂f/∂conj(p_v):  p doesn't depend on conj(p_v), so only −p·dq_conj / q².
    Complex eval_df_dconj(int v, Complex s) const {
        Complex p_val = p_poly.evaluate(s);
        Complex q_val = q_poly.evaluate(s);
        if (std::abs(q_val) < 1e-15) return Complex(0);
        Complex dq_conj_val = dq_conj_polys[v].evaluate(s);
        return -p_val * dq_conj_val / (q_val * q_val);
    }
};

// Build ChannelPolyState: one Feldtkeller call per channel + one coupled
// Bezout LU per channel (shared across the d free variables).
//
// Bezout derivation (holding conj(p) fixed in the Wirtinger sense):
//   q·q_para = p·p_para + r·r_para
//   ⇒  dq·q_para + q·(dq_conj)_para = mon_v · p_para           (*)
// (∂p_para/∂p_v = 0 and ∂q_para/∂p_v has coefficients conj(∂q/∂conj(p_v)),
//  i.e. (dq_conj)_para.)
//
// Let Z := (dq_conj)_para. Both dq and Z are polynomials of degree ≤ d−1
// (since q's leading coefficient is gauge-fixed real and independent of
// non-leading p_v). Equation (*) gives 2d complex coefficient equations in
// 2d complex unknowns (dq_0..dq_{d−1}, Z_0..Z_{d−1}). The LHS matrix is
// the Sylvester-like pencil of (q_para, q), nonsingular since q is Hurwitz
// and q_para is anti-Hurwitz (disjoint roots). Factor once per channel,
// re-use across all v.
//
// Recover ∂q/∂conj(p_v) from Z via  (dq_conj)_k = (−1)^k · conj(Z_k).
ChannelPolyState build_poly_state(
    const VectorXcd& p_desc,
    const VectorXcd& r_desc,
    int num_vars,
    bool with_derivatives)
{
    ChannelPolyState st;
    st.p_poly = desc_to_poly(p_desc);
    auto r_poly = desc_to_poly(r_desc);
    st.q_poly = SpectralFactor::feldtkeller(st.p_poly, r_poly);

    if (!with_derivatives) return st;

    // p has d+1 coefficients (ascending); monic ⇒ leading fixed ⇒ d free.
    int N = static_cast<int>(p_desc.size());
    int d = N - 1;
    st.dq_polys.resize(num_vars);
    st.dq_conj_polys.resize(num_vars);
    st.dp_degrees.resize(num_vars);

    auto p_para = st.p_poly.para_conjugate();
    auto q_para = st.q_poly.para_conjugate();

    auto coef = [](const Polynomial<Complex>& poly, int j) -> Complex {
        if (j < 0 || j >= static_cast<int>(poly.coefficients.size()))
            return Complex(0);
        return poly.coefficients[j];
    };

    // Size of the coupled system.
    int num_eqs = 2 * d;
    int num_unk = 2 * d;
    MatrixXcd A = MatrixXcd::Zero(num_eqs, num_unk);

    for (int k = 0; k < num_eqs; ++k) {
        // Column i ∈ [0, d): dq_i contribution = (q_para)_{k−i}
        for (int i = 0; i < d; ++i) {
            A(k, i) = coef(q_para, k - i);
        }
        // Column d+i ∈ [d, 2d): Z_i contribution = q_{k−i}
        for (int i = 0; i < d; ++i) {
            A(k, d + i) = coef(st.q_poly, k - i);
        }
    }

    // partialPivLu pivots on the largest entry in the current column; on
    // near-singular complex matrices with specific sparsity patterns
    // (arising at odd d with real-coefficient Hurwitz q) the pivot choice
    // can produce NaN under WASM/emscripten rounding while native libm
    // picks a different numerically-valid pivot. completeOrthogonalDecomposition
    // is the rank-revealing solver and behaves identically on both
    // toolchains. Factor once, re-use for all free variables.
    auto cod = A.completeOrthogonalDecomposition();

    for (int v = 0; v < num_vars; ++v) {
        int coeff_idx = v + 1;     // descending-coefficient index of p_v
        int deg = d - coeff_idx;   // s-degree of that coefficient
        st.dp_degrees[v] = deg;

        // RHS: mon_v · p_para has coefficient k = (p_para)_{k − deg_v}
        VectorXcd b = VectorXcd::Zero(num_eqs);
        for (int k = 0; k < num_eqs; ++k) {
            b(k) = coef(p_para, k - deg);
        }

        VectorXcd sol = cod.solve(b);

        // dq: coefficients 0..d−1 from sol[0..d−1]; leading (degree d) = 0.
        std::vector<Complex> dq_c(d, Complex(0));
        for (int i = 0; i < d; ++i) dq_c[i] = sol(i);
        st.dq_polys[v] = Polynomial<Complex>(dq_c);

        // Recover dq_conj via (dq_conj)_k = (−1)^k · conj(Z_k).
        std::vector<Complex> dq_conj_c(d, Complex(0));
        for (int i = 0; i < d; ++i) {
            double sgn = (i % 2 == 0) ? 1.0 : -1.0;
            dq_conj_c[i] = sgn * std::conj(sol(d + i));
        }
        st.dq_conj_polys[v] = Polynomial<Complex>(dq_conj_c);
    }

    return st;
}

} // anonymous namespace

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
    Complex coupling = (Vi.transpose() * M_inv * Fi * Vi)(0, 0);

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
    auto all_p = split_variables(x);

    // Factor each channel once (N feldtkeller calls total)
    std::vector<ChannelPolyState> polys(num_channels_);
    for (int k = 0; k < num_channels_; ++k)
        polys[k] = build_poly_state(all_p[k], channels_[k].r_coeffs,
                                     channels_[k].num_vars, false);

    VectorXcd F(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        for (int m = 0; m < ch.order; ++m) {
            Complex s_i(0, ch.norm_freqs[m]);
            Complex fi = polys[i].eval_f(s_i);

            double freq_phys = ch.interp_freqs[m];
            auto J = manifold_->compute_J(freq_phys);
            Complex J_ii = J(i + 1, i + 1);

            Complex Li;
            if (std::abs(lambda) < 1e-15 || num_channels_ < 2) {
                Li = J_ii;
            } else {
                VectorXcd vi = manifold_->compute_Vi(J, i);
                MatrixXcd wi = manifold_->compute_Wi(J, i);
                int dim = num_channels_ - 1;
                MatrixXcd Fi_mat = MatrixXcd::Zero(dim, dim);
                int col = 0;
                for (int k = 0; k < num_channels_; ++k) {
                    if (k == i) continue;
                    double nf = (freq_phys - channels_[k].freq_center)
                              / channels_[k].freq_scale;
                    Fi_mat(col, col) = polys[k].eval_f(Complex(0, nf));
                    col++;
                }
                MatrixXcd I_dim = MatrixXcd::Identity(dim, dim);
                Complex coupling = (vi.transpose() * (I_dim - Fi_mat * wi).inverse()
                                    * Fi_mat * vi)(0, 0);
                Li = J_ii + lambda * coupling;
            }

            // Martinez §III.B matching condition: f_i(ξ) = conj(L_i(ξ)).
            // Derivation: for a lossless reciprocal filter in Belevitch form
            // with f = S22 = p/q and load reflection Γ_L, the input reflection
            // G = ε·(p* − q*·Γ_L)/(q − p·Γ_L) vanishes iff Γ_L = p*/q* = conj(f)
            // on the jω axis. So the load satisfies L = conj(f), i.e. f = conj(L).
            F(ch.eq_offset + m) = fi - std::conj(Li);
        }
    }
    return F;
}

void MultiplexerNevanlinnaPick::compute_jacobians_wirtinger(
    const VectorXcd& x, double lambda,
    MatrixXcd& Jac_A, MatrixXcd& Jac_B) const
{
    auto all_p = split_variables(x);

    std::vector<ChannelPolyState> polys(num_channels_);
    for (int k = 0; k < num_channels_; ++k)
        polys[k] = build_poly_state(all_p[k], channels_[k].r_coeffs,
                                     channels_[k].num_vars, true);

    Jac_A = MatrixXcd::Zero(total_eqs_, total_vars_);
    Jac_B = MatrixXcd::Zero(total_eqs_, total_vars_);

    // Diagonal blocks: ∂f_i/∂p_{i,v} and ∂f_i/∂conj(p_{i,v}) at channel i's
    // interpolation frequencies.
    for (int i = 0; i < num_channels_; ++i) {
        auto& ch_i = channels_[i];
        for (int m = 0; m < ch_i.order; ++m) {
            Complex s(0, ch_i.norm_freqs[m]);
            for (int v = 0; v < ch_i.num_vars; ++v) {
                Jac_A(ch_i.eq_offset + m, ch_i.var_offset + v) = polys[i].eval_df(v, s);
                Jac_B(ch_i.eq_offset + m, ch_i.var_offset + v) = polys[i].eval_df_dconj(v, s);
            }
        }
    }

    if (std::abs(lambda) < 1e-15) return;

    // Off-diagonal blocks: ∂(−L_i)/∂(p_{j,v} | conj(p_{j,v})) for j ≠ i.
    // Chain rule: L_i depends on p_j only through f_j = f_j(p_j, conj(p_j)),
    // and the rank-1 structure of dL/df_j factors out:
    //   ∂L_i/∂f_j = λ · (v_i^T · m)_{j_in_Fi} · ((W_i · m · F_i + I) · v_i)_{j_in_Fi}
    // where m = (I − F_i · W_i)^{-1}. Both holomorphic and antiholomorphic
    // Jacobians share this scaling; only ∂f_j/∂(…) differs.
    for (int i = 0; i < num_channels_; ++i) {
        auto& ch_i = channels_[i];

        for (int m = 0; m < ch_i.order; ++m) {
            double phys_freq = ch_i.interp_freqs[m];
            auto J_mfold = manifold_->compute_J(phys_freq);
            VectorXcd vi = manifold_->compute_Vi(J_mfold, i);
            MatrixXcd wi = manifold_->compute_Wi(J_mfold, i);

            int dim = num_channels_ - 1;
            MatrixXcd Fi = MatrixXcd::Zero(dim, dim);

            int col = 0;
            for (int k = 0; k < num_channels_; ++k) {
                if (k == i) continue;
                double nf = (phys_freq - channels_[k].freq_center)
                          / channels_[k].freq_scale;
                Fi(col, col) = polys[k].eval_f(Complex(0, nf));
                col++;
            }

            MatrixXcd I_dim = MatrixXcd::Identity(dim, dim);
            MatrixXcd Sm = (I_dim - Fi * wi).inverse();
            MatrixXcd WSF_I = wi * Sm * Fi + I_dim;

            VectorXcd STv = Sm.transpose() * vi;
            VectorXcd WSF_Iv = WSF_I * vi;

            for (int j = 0; j < num_channels_; ++j) {
                if (j == i) continue;
                auto& ch_j = channels_[j];

                int j_in_Fi = 0;
                for (int kk = 0; kk < j; ++kk)
                    if (kk != i) j_in_Fi++;

                double norm_freq_j = (phys_freq - ch_j.freq_center)
                                   / ch_j.freq_scale;
                Complex s_j(0, norm_freq_j);

                Complex left  = STv(j_in_Fi);
                Complex right = WSF_Iv(j_in_Fi);
                // F = f - conj(L):
                //   ∂F/∂p      = ∂f/∂p      − conj(∂L/∂conj(p))
                //   ∂F/∂conj(p)= ∂f/∂conj(p)− conj(∂L/∂p)
                // Off-diagonal ∂L/∂p      = (λ·left·right)·df
                //             ∂L/∂conj(p) = (λ·left·right)·df_conj
                // so the off-diagonal entries swap (A ↔ B with conj).
                Complex scale_conj = -lambda * std::conj(left * right);

                for (int v = 0; v < ch_j.num_vars; ++v) {
                    Complex df      = polys[j].eval_df(v, s_j);
                    Complex df_conj = polys[j].eval_df_dconj(v, s_j);
                    Jac_A(ch_i.eq_offset + m, ch_j.var_offset + v) =
                        scale_conj * std::conj(df_conj);
                    Jac_B(ch_i.eq_offset + m, ch_j.var_offset + v) =
                        scale_conj * std::conj(df);
                }
            }
        }
    }
}

MatrixXcd MultiplexerNevanlinnaPick::compute_jacobian_dp(
    const VectorXcd& x, double lambda) const
{
    MatrixXcd A, B;
    compute_jacobians_wirtinger(x, lambda, A, B);
    return A;
}

VectorXcd MultiplexerNevanlinnaPick::newton_step(
    const VectorXcd& x, double lambda, const VectorXcd& F) const
{
    // Real 2M × 2M Newton system derived from the Wirtinger pair (A, B):
    //   ∂F/∂p · δp + ∂F/∂conj(p) · conj(δp) = −F
    // Splitting δp = δp_r + i·δp_i and matching real/imag parts:
    //   [ Re(A+B)   Im(B−A) ] [ δp_r ]   [ −Re(F) ]
    //   [ Im(A+B)   Re(A−B) ] [ δp_i ] = [ −Im(F) ]
    int M = total_vars_;
    int E = total_eqs_;

    MatrixXcd A, B;
    compute_jacobians_wirtinger(x, lambda, A, B);

    MatrixXd J_real(2 * E, 2 * M);
    MatrixXcd ApB = A + B;
    MatrixXcd AmB = A - B;
    MatrixXcd BmA = B - A;
    J_real.block(0, 0, E, M) = ApB.real();
    J_real.block(0, M, E, M) = BmA.imag();
    J_real.block(E, 0, E, M) = ApB.imag();
    J_real.block(E, M, E, M) = AmB.real();

    VectorXd F_real(2 * E);
    F_real.head(E) = F.real();
    F_real.tail(E) = F.imag();

    // At the real-coefficient starting point (e.g. the Chebyshev prototype at
    // lambda=0), the real 2E x 2M Jacobian has a non-trivial null space — the
    // imaginary parts of the variables are unreachable by any residual
    // direction. Native Eigen's colPivHouseholderQr returns a valid
    // least-squares step through this rank deficiency; WASM / emscripten's
    // rounding lands on a marginally worse pivot and produces a NaN step,
    // which then propagates into NaN polynomial coefficients on the next
    // iteration. completeOrthogonalDecomposition is the minimum-norm solver
    // for rank-deficient systems and gives a stable step in both toolchains.
    VectorXd y = J_real.completeOrthogonalDecomposition().solve(-F_real);

    VectorXcd dx(M);
    for (int i = 0; i < M; ++i)
        dx(i) = Complex(y(i), y(M + i));
    return dx;
}

VectorXcd MultiplexerNevanlinnaPick::compute_dF_dlambda(
    const VectorXcd& x, double lambda) const
{
    auto all_p = split_variables(x);

    std::vector<ChannelPolyState> polys(num_channels_);
    for (int k = 0; k < num_channels_; ++k)
        polys[k] = build_poly_state(all_p[k], channels_[k].r_coeffs,
                                     channels_[k].num_vars, false);

    VectorXcd dFdl(total_eqs_);

    for (int i = 0; i < num_channels_; ++i) {
        auto& ch = channels_[i];
        for (int m = 0; m < ch.order; ++m) {
            double freq_phys = ch.interp_freqs[m];
            auto J = manifold_->compute_J(freq_phys);
            VectorXcd vi = manifold_->compute_Vi(J, i);
            MatrixXcd wi = manifold_->compute_Wi(J, i);
            int dim = num_channels_ - 1;
            MatrixXcd Fi_mat = MatrixXcd::Zero(dim, dim);
            int col = 0;
            for (int k = 0; k < num_channels_; ++k) {
                if (k == i) continue;
                double nf = (freq_phys - channels_[k].freq_center)
                          / channels_[k].freq_scale;
                Fi_mat(col, col) = polys[k].eval_f(Complex(0, nf));
                col++;
            }
            MatrixXcd I_dim = MatrixXcd::Identity(dim, dim);
            Complex coupling = (vi.transpose() * (I_dim - Fi_mat * wi).inverse()
                                * Fi_mat * vi)(0, 0);
            // F = f − conj(L), conj(L) = conj(J_ii) + λ·conj(coupling) (λ real)
            dFdl(ch.eq_offset + m) = -std::conj(coupling);
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
    int M = total_vars_;
    int E = total_eqs_;

    std::ofstream path_log("continuation_path.csv");
    path_log << "step,lambda,h,residual,newton_iters,dx_norm,x_norm,predictor_res,cond,tangent_norm";
    for (int i = 0; i < M; ++i)
        path_log << ",x" << i << "_re,x" << i << "_im";
    path_log << "\n";

    // Helper: assemble the real 2E × 2M Jacobian from Wirtinger blocks.
    auto make_real_jac = [&](const MatrixXcd& A, const MatrixXcd& B,
                              MatrixXd& J_real) {
        J_real.resize(2 * E, 2 * M);
        MatrixXcd ApB = A + B;
        MatrixXcd AmB = A - B;
        MatrixXcd BmA = B - A;
        J_real.block(0, 0, E, M) = ApB.real();
        J_real.block(0, M, E, M) = BmA.imag();
        J_real.block(E, 0, E, M) = ApB.imag();
        J_real.block(E, M, E, M) = AmB.real();
    };

    auto pack_real_F = [&](const VectorXcd& F) {
        VectorXd F_real(2 * E);
        F_real.head(E) = F.real();
        F_real.tail(E) = F.imag();
        return F_real;
    };
    auto unpack_complex = [&](const VectorXd& y) {
        VectorXcd out(M);
        for (int i = 0; i < M; ++i) out(i) = Complex(y(i), y(M + i));
        return out;
    };

    // Log starting point at lambda=0
    {
        path_log << -1 << "," << 0.0 << "," << 0 << ","
                 << compute_residual(x, 0.0).norm() << ","
                 << 0 << "," << 0 << "," << x.norm() << "," << 0
                 << "," << 0 << "," << 0;
        for (int i = 0; i < M; ++i)
            path_log << "," << x(i).real() << "," << x(i).imag();
        path_log << "\n";
    }

    for (int step = 0; step < max_steps && lambda < 1.0 - 1e-12; ++step) {
        if (lambda + h > 1.0) h = 1.0 - lambda;

        // Wirtinger Jacobians and dF/dlambda at current point.
        MatrixXcd A, B;
        compute_jacobians_wirtinger(x, lambda, A, B);
        VectorXcd dFdl = compute_dF_dlambda(x, lambda);

        MatrixXd J_real;
        make_real_jac(A, B, J_real);

        Eigen::JacobiSVD<MatrixXd> svd(J_real);
        auto sv = svd.singularValues();
        double cond = (sv(sv.size()-1) > 1e-15) ? sv(0)/sv(sv.size()-1) : 1e15;

        // Tangent:  ∂F/∂p · τ + ∂F/∂conj(p) · conj(τ) = −dF/dλ
        // Use completeOrthogonalDecomposition — the real Jacobian is
        // rank-deficient along imaginary-axis directions at the
        // real-coefficient start, and colPivHouseholderQr produces NaN
        // under WASM rounding (see newton_step comment).
        VectorXd tau_real = J_real.completeOrthogonalDecomposition().solve(-pack_real_F(dFdl));
        VectorXcd tangent = unpack_complex(tau_real);
        double tangent_norm = tangent.norm();

        // Euler predictor
        VectorXcd x_pred = x + tangent * h;
        double new_lambda = lambda + h;

        double predictor_res = compute_residual(x_pred, new_lambda).norm();

        // Wirtinger Newton corrector with plateau detection.
        bool converged = false;
        int newton_iters = 0;
        double prev_res = std::numeric_limits<double>::infinity();
        int plateau_count = 0;
        for (int iter = 0; iter < newton_max_iter; ++iter) {
            VectorXcd F = compute_residual(x_pred, new_lambda);
            double res = F.norm();

            if (verbose && step < 3) {
                std::cout << "      Newton " << iter << ": res=" << std::scientific
                          << std::setprecision(3) << res;
            }

            if (res < newton_tol) {
                if (verbose && step < 3) std::cout << "\n";
                converged = true;
                newton_iters = iter + 1;
                break;
            }

            if (res > 0.95 * prev_res) plateau_count++;
            else plateau_count = 0;
            if (plateau_count >= 3 && res < 10.0 * newton_tol) {
                if (verbose && step < 3) std::cout << " [plateau-accept]\n";
                converged = true;
                newton_iters = iter + 1;
                break;
            }
            prev_res = res;

            MatrixXcd Ac, Bc;
            compute_jacobians_wirtinger(x_pred, new_lambda, Ac, Bc);
            MatrixXd Jr_c;
            make_real_jac(Ac, Bc, Jr_c);

            VectorXd corr_real = Jr_c.completeOrthogonalDecomposition().solve(-pack_real_F(F));
            VectorXcd correction = unpack_complex(corr_real);

            double alpha = 1.0;
            bool stepped = false;
            for (int ls = 0; ls < 10; ++ls) {
                VectorXcd x_try = x_pred + alpha * correction;
                double res_try = compute_residual(x_try, new_lambda).norm();
                if (res_try < res) { x_pred = x_try; stepped = true; break; }
                alpha *= 0.5;
            }
            if (verbose && step < 3) std::cout << "\n";
            if (!stepped || alpha < 1e-8) {
                if (res < 10.0 * newton_tol) {
                    converged = true;
                    newton_iters = iter + 1;
                }
                break;
            }
            newton_iters = iter + 1;
        }

        if (converged) {
            double dx_norm = (x_pred - x).norm();
            x = x_pred;
            lambda = new_lambda;

            path_log << step << "," << lambda << "," << h << ","
                     << compute_residual(x, lambda).norm() << ","
                     << newton_iters << "," << dx_norm << ","
                     << x.norm() << "," << predictor_res
                     << "," << cond << "," << tangent_norm;
            for (int i = 0; i < total_vars_; ++i)
                path_log << "," << x(i).real() << "," << x(i).imag();
            path_log << "\n";

            if (verbose && (step % 20 == 0 || lambda >= 1.0 - 1e-12)) {
                VectorXcd F = compute_residual(x, lambda);
                std::cout << "    lambda=" << std::fixed << std::setprecision(4) << lambda
                          << " residual=" << std::scientific << std::setprecision(2)
                          << F.norm() << " h=" << h
                          << " newton=" << newton_iters << "\n";
            }

            // Adaptive step based on predictor quality.
            double pred_ratio = predictor_res / newton_tol;
            double grow = (pred_ratio < 10.0)  ? 2.0 :
                          (pred_ratio < 100.0) ? 1.5 : 1.2;
            h = std::min(h * grow, 1.0 - lambda);
        } else {
            h *= 0.5;
            if (h < 1e-10) {
                if (verbose) {
                    VectorXcd F = compute_residual(x, lambda);
                    std::cout << "    Stalled at lambda=" << std::fixed << std::setprecision(6)
                              << lambda << " residual=" << F.norm() << " h=" << h << "\n";
                }
                return false;
            }
        }
    }
    path_log.close();

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

        // Polynomials are already in the [-1,1] normalized domain.
        // No additional shift needed (unlike the single-channel NP solver which
        // internally shifts by the mean frequency).
        auto e_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

        cms[i] = CouplingMatrix::from_polynomials_by_realization(p_poly, r_poly, e_poly);
    }

    return cms;
}

bool MultiplexerNevanlinnaPick::shift_frequencies(
    VectorXcd& x,
    const std::vector<std::vector<double>>& target_freqs,
    double step, int max_steps, int newton_max, double newton_tol, bool verbose)
{
    // Save original frequencies
    std::vector<std::vector<double>> orig_phys(num_channels_);
    std::vector<std::vector<double>> orig_norm(num_channels_);
    for (int i = 0; i < num_channels_; ++i) {
        orig_phys[i] = channels_[i].interp_freqs;
        orig_norm[i].assign(channels_[i].norm_freqs.begin(), channels_[i].norm_freqs.end());
    }

    double lambda = 0.0;
    double h = step;

    for (int s = 0; s < max_steps && lambda < 1.0 - 1e-12; ++s) {
        if (lambda + h > 1.0) h = 1.0 - lambda;

        double new_lambda = lambda + h;

        // Blend frequencies: xi(lambda) = (1-lambda)*orig + lambda*target
        auto set_blended = [&](double lam) {
            for (int i = 0; i < num_channels_; ++i) {
                for (int m = 0; m < channels_[i].order; ++m) {
                    channels_[i].interp_freqs[m] =
                        (1.0 - lam) * orig_phys[i][m] + lam * target_freqs[i][m];
                    channels_[i].norm_freqs[m] =
                        (channels_[i].interp_freqs[m] - channels_[i].freq_center)
                        / channels_[i].freq_scale;
                }
            }
        };

        // Euler predictor: dx = -(dF/dp)^{-1} * (dF/dlambda_freq) * h
        // dF/dlambda_freq is computed by FD on the blended residual
        // Analytical dF/dlambda: since xi(lambda) = (1-lambda)*orig + lambda*target,
        // dxi/dlambda = target - orig.  dF/dlambda = dF/dxi * dxi/dlambda.
        // dF_{i,m}/dxi_{i,m} is the frequency derivative of [f(p_i) - L_i] at xi_{i,m}.
        // This is a diagonal contribution (each equation depends on one frequency).

        set_blended(lambda);
        auto all_p = split_variables(x);
        VectorXcd dFdl(total_eqs_);

        for (int i = 0; i < num_channels_; ++i) {
            auto& ch = channels_[i];
            auto r_poly = build_polynomial(ch.r_coeffs);
            auto p_poly = build_polynomial(all_p[i]);
            auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);

            for (int m = 0; m < ch.order; ++m) {
                double dxi = target_freqs[i][m] - orig_phys[i][m];  // dxi/dlambda
                double phys_f = ch.interp_freqs[m];
                double norm_f = ch.norm_freqs[m];

                // df/dxi: derivative of f(p_i) = p/q w.r.t. physical frequency
                // s = j * norm_freq, ds/dxi = j / freq_scale
                Complex s(0, norm_f);
                Complex p_val = p_poly.evaluate(s);
                Complex q_val = q_poly.evaluate(s);
                Complex p_deriv = p_poly.derivative().evaluate(s);
                Complex q_deriv = q_poly.derivative().evaluate(s);
                Complex ds_dxi = Complex(0, 1.0 / ch.freq_scale);
                Complex df_dxi = ((p_deriv * q_val - p_val * q_deriv)
                                 / (q_val * q_val)) * ds_dxi;

                // dL_i/dxi: derivative of coupled load w.r.t. frequency
                // Compute via central FD on eval_coupled_load (small, cheap)
                double delta = 1e-6 * ch.freq_scale;
                Complex L_plus = eval_coupled_load(i, phys_f + delta, 1.0, all_p);
                Complex L_minus = eval_coupled_load(i, phys_f - delta, 1.0, all_p);
                Complex dL_dxi = (L_plus - L_minus) / (2.0 * delta);

                // F = f − conj(L)  ⇒  dF/dξ = df/dξ − conj(dL/dξ)
                dFdl(ch.eq_offset + m) = (df_dxi - std::conj(dL_dxi)) * dxi;
            }
        }

        // Wirtinger tangent: solve the real 2E × 2M system
        //   A·τ + B·conj(τ) = −dFdl    (treating dFdl as the RHS of the tangent eqn)
        VectorXcd dx_tangent_h = newton_step(x, 1.0, dFdl * h);
        VectorXcd x_pred = x + dx_tangent_h;

        // Newton corrector at blended frequencies (real-variable Wirtinger).
        set_blended(new_lambda);
        bool converged = false;
        for (int iter = 0; iter < newton_max; ++iter) {
            VectorXcd F = compute_residual(x_pred, 1.0);
            double res = F.norm();
            if (res < newton_tol) { converged = true; break; }

            VectorXcd corr = newton_step(x_pred, 1.0, F);

            // Backtracking
            double alpha = 1.0;
            for (int ls = 0; ls < 8; ++ls) {
                VectorXcd x_try = x_pred + alpha * corr;
                double res_try = compute_residual(x_try, 1.0).norm();
                if (res_try < res) { x_pred = x_try; break; }
                alpha *= 0.5;
            }
        }

        if (converged) {
            x = x_pred;
            lambda = new_lambda;
            h = std::min(h * 1.5, 0.5);

            if (verbose) {
                VectorXcd F = compute_residual(x, 1.0);
                std::cout << "    freq_shift lambda=" << std::fixed << std::setprecision(3)
                          << lambda << " res=" << std::scientific << std::setprecision(1)
                          << F.norm() << "\n";
            }
        } else {
            h *= 0.5;
            set_blended(lambda);  // revert frequencies
            if (h < 1e-10) {
                if (verbose) std::cout << "    freq_shift stalled at lambda=" << lambda << "\n";
                return false;
            }
        }
    }

    return lambda >= 1.0 - 1e-12;
}

} // namespace np
