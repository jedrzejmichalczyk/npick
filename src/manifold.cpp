#include "multiplexer/manifold.hpp"
#include "chebyshev_filter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace np {

Manifold::Manifold(int num_channels, double center_frequency)
    : num_channels_(num_channels)
    , center_frequency_(center_frequency)
{
    if (num_channels < 2) {
        throw std::invalid_argument("Manifold: need at least 2 channels");
    }

    // Default: equal-split T-junctions
    junctions_.resize(num_channels - 1, TJunction(0.5));

    // Default: reasonable line lengths (will be overwritten by compute_line_lengths)
    double default_length = (center_frequency > 1e-10) ? 0.25 / center_frequency : 0.1;
    channel_lengths_.resize(num_channels, default_length);
    manifold_lengths_.resize(num_channels - 1, default_length);
}

void Manifold::set_junction(int idx, const TJunction& junction) {
    if (idx < 0 || idx >= static_cast<int>(junctions_.size())) {
        throw std::out_of_range("Manifold::set_junction: index out of range");
    }
    junctions_[idx] = junction;
}

void Manifold::set_channel_line_lengths(const std::vector<double>& lengths) {
    if (static_cast<int>(lengths.size()) != num_channels_) {
        throw std::invalid_argument("Manifold: channel lengths size mismatch");
    }
    channel_lengths_ = lengths;
}

void Manifold::set_manifold_line_lengths(const std::vector<double>& lengths) {
    if (static_cast<int>(lengths.size()) != num_channels_ - 1) {
        throw std::invalid_argument("Manifold: manifold lengths size mismatch");
    }
    manifold_lengths_ = lengths;
}

Eigen::Matrix2cd Manifold::transmission_line_S(double length, double freq) {
    // Phase shift: theta = 2*pi*freq*length (normalized: length in wavelengths)
    double theta = 2.0 * PI * freq * length;
    Complex phase = std::exp(Complex(0, -theta));
    Eigen::Matrix2cd S;
    S << Complex(0), phase,
         phase, Complex(0);
    return S;
}

MatrixXcd Manifold::connect_ports(const MatrixXcd& S1, int p1,
                                  const MatrixXcd& S2, int p2) {
    // Connect port p1 of S1 to port p2 of S2.
    // Uses the standard multi-port S-parameter connection formula.
    //
    // Result has (n1 - 1 + n2 - 1) ports: all ports of S1 except p1,
    // plus all ports of S2 except p2.

    int n1 = S1.rows();
    int n2 = S2.rows();
    int n_out = n1 + n2 - 2;

    // Build index maps: which ports survive
    std::vector<int> idx1, idx2;
    for (int i = 0; i < n1; ++i) if (i != p1) idx1.push_back(i);
    for (int i = 0; i < n2; ++i) if (i != p2) idx2.push_back(i);

    // Extract submatrices for the connection formula:
    // S1 partitioned around port p1, S2 around port p2
    //
    // Connection: a_{p1} = S2_{p2,free2} b_{free2} + S2_{p2,p2} S1_{p1,free1} b_{free1} + ...
    //
    // Standard formula for connecting port p1 of S1 to port p2 of S2:
    // The combined S-parameter is:
    //
    // S_combined = [S1_ff + S1_fp * Gamma * S2_pf_mapped, S1_fp * T * S2_ff]
    //              [S2_fp * T' * S1_ff_mapped,             S2_ff + ...]
    //
    // where Gamma = S2_{p2,p2} and the feedback loop is:
    //   (1 - S1_{p1,p1} * S2_{p2,p2})^{-1}

    Complex S1pp = S1(p1, p1);
    Complex S2pp = S2(p2, p2);
    Complex denom = Complex(1.0) - S1pp * S2pp;

    if (std::abs(denom) < 1e-15) {
        throw std::runtime_error("Manifold::connect_ports: singular connection");
    }

    Complex inv_denom = Complex(1.0) / denom;

    // Extract vectors: S1 row/col at p1 (excluding p1), S2 row/col at p2 (excluding p2)
    VectorXcd S1_p1_row(idx1.size()); // S1[p1, free1]
    VectorXcd S1_p1_col(idx1.size()); // S1[free1, p1]
    for (size_t i = 0; i < idx1.size(); ++i) {
        S1_p1_row(i) = S1(p1, idx1[i]);
        S1_p1_col(i) = S1(idx1[i], p1);
    }

    VectorXcd S2_p2_row(idx2.size()); // S2[p2, free2]
    VectorXcd S2_p2_col(idx2.size()); // S2[free2, p2]
    for (size_t i = 0; i < idx2.size(); ++i) {
        S2_p2_row(i) = S2(p2, idx2[i]);
        S2_p2_col(i) = S2(idx2[i], p2);
    }

    // Extract free-to-free submatrices
    MatrixXcd S1_ff(idx1.size(), idx1.size());
    for (size_t i = 0; i < idx1.size(); ++i)
        for (size_t j = 0; j < idx1.size(); ++j)
            S1_ff(i, j) = S1(idx1[i], idx1[j]);

    MatrixXcd S2_ff(idx2.size(), idx2.size());
    for (size_t i = 0; i < idx2.size(); ++i)
        for (size_t j = 0; j < idx2.size(); ++j)
            S2_ff(i, j) = S2(idx2[i], idx2[j]);

    // Build result matrix
    MatrixXcd result = MatrixXcd::Zero(n_out, n_out);

    int off1 = 0;
    int off2 = static_cast<int>(idx1.size());

    // Block (1,1): S1_ff + S1_col * inv_denom * S2pp * S1_row
    result.block(off1, off1, idx1.size(), idx1.size()) =
        S1_ff + S2pp * inv_denom * S1_p1_col * S1_p1_row.transpose();

    // Block (2,2): S2_ff + S2_col * inv_denom * S1pp * S2_row
    result.block(off2, off2, idx2.size(), idx2.size()) =
        S2_ff + S1pp * inv_denom * S2_p2_col * S2_p2_row.transpose();

    // Block (1,2): S1_col * inv_denom * S2_row
    result.block(off1, off2, idx1.size(), idx2.size()) =
        inv_denom * S1_p1_col * S2_p2_row.transpose();

    // Block (2,1): S2_col * inv_denom * S1_row
    result.block(off2, off1, idx2.size(), idx1.size()) =
        inv_denom * S2_p2_col * S1_p1_row.transpose();

    return result;
}

MatrixXcd Manifold::compute_J(double freq) const {
    // Build manifold S-matrix by cascading T-junctions and transmission lines.
    //
    // For N channels with serial topology:
    //   Start with T-junction 0 (3 ports: common, ch1, continuation)
    //   Connect continuation to TL_m[1], then to T-junction 1, etc.
    //   Last junction has 2 channel ports (no continuation).
    //
    // After cascading junctions, add channel transmission lines.

    int N = num_channels_;

    if (N == 2) {
        // Single T-junction: ports = [common, ch1, ch2]
        // Add channel transmission lines
        MatrixXcd J = junctions_[0].S();

        // Add manifold TL from common port to junction
        auto tl_common = transmission_line_S(manifold_lengths_[0], freq);
        J = connect_ports(tl_common, 1, J, 0);
        // Now J ports: [common_in, ch1, ch2]

        // Add channel TLs
        // Port 1 (ch1): connect TL
        auto tl1 = transmission_line_S(channel_lengths_[0], freq);
        // We need to add TLs without removing ports — effectively they just
        // add phase to the channel ports. For a transmission line on port k:
        // S'_{k,j} = e^{-j*theta} * S_{k,j} for j != k
        // S'_{k,k} = S_{k,k} (reflection unchanged for matched TL)
        // Actually for a zero-reflection TL: S_{k,k} = 0, so the effect is
        // just a phase shift on all signals entering/leaving that port.
        for (int k = 0; k < 2; ++k) {
            double theta = 2.0 * PI * freq * channel_lengths_[k];
            Complex phase = std::exp(Complex(0, -theta));
            int port = k + 1; // ports 1 and 2
            for (int j = 0; j < J.rows(); ++j) {
                if (j != port) {
                    J(port, j) *= phase;
                    J(j, port) *= phase;
                }
            }
        }

        return J;
    }

    // General N > 2 case: cascade T-junctions
    // Start with junction 0: ports [common(0), ch_N(1), continuation(2)]
    // We build right-to-left: junction N-2 is closest to common port.
    // Actually, following Martinez Fig. 1, the topology is:
    //   Common --- T1 --- T2 --- ... --- T_{N-1}
    // T1 has ports: [from_common, ch1, to_T2]
    // T_{N-1} has ports: [from_T_{N-2}, ch_{N-1}, ch_N]

    // Start from the last junction (furthest from common)
    // T_{N-2}: ports [from_prev, ch_{N-1}, ch_N]
    MatrixXcd accumulated = junctions_[N - 2].S();
    // Ports: [from_prev, ch_{N-1}, ch_N]
    // Rename for tracking: port 0 = from_prev, port 1 = ch_{N-1}, port 2 = ch_N

    // Add channel TLs on ch_{N-1} and ch_N (ports 1 and 2)
    for (int k = 1; k <= 2; ++k) {
        int ch_idx = (k == 1) ? N - 2 : N - 1;
        double theta = 2.0 * PI * freq * channel_lengths_[ch_idx];
        Complex phase = std::exp(Complex(0, -theta));
        for (int j = 0; j < accumulated.rows(); ++j) {
            if (j != k) {
                accumulated(k, j) *= phase;
                accumulated(j, k) *= phase;
            }
        }
    }

    // Cascade from right to left
    for (int junc = N - 3; junc >= 0; --junc) {
        // Junction junc: ports [from_prev(0), ch_{junc+1}(1), to_next(2)]
        MatrixXcd Tj = junctions_[junc].S();

        // Add manifold TL between this junction and the accumulated network
        auto tl = transmission_line_S(manifold_lengths_[junc + 1], freq);

        // Connect TL port 0 to Tj port 2 (continuation)
        MatrixXcd Tj_with_tl = connect_ports(Tj, 2, tl, 0);
        // Tj_with_tl ports: [from_prev(0), ch_{junc+1}(1), tl_out(2)]

        // Connect tl_out (port 2) to accumulated port 0 (from_prev)
        accumulated = connect_ports(Tj_with_tl, 2, accumulated, 0);
        // accumulated ports: [from_prev(0), ch_{junc+1}(1), ch_{junc+2}..ch_N]

        // Add channel TL on ch_{junc+1} (port 1)
        double theta = 2.0 * PI * freq * channel_lengths_[junc];
        Complex phase = std::exp(Complex(0, -theta));
        for (int j = 0; j < accumulated.rows(); ++j) {
            if (j != 1) {
                accumulated(1, j) *= phase;
                accumulated(j, 1) *= phase;
            }
        }
    }

    // Add manifold TL from common port to first junction
    auto tl_common = transmission_line_S(manifold_lengths_[0], freq);
    accumulated = connect_ports(tl_common, 1, accumulated, 0);

    // Final port order: [common, ch_1, ch_2, ..., ch_N]
    return accumulated;
}

VectorXcd Manifold::compute_Vi(const MatrixXcd& J, int channel_idx) const {
    // V_i = [J_{i,k}] for k in {1..N}, k != i
    // channel_idx is 1-indexed in the paper but J uses 0-indexed internally
    // with port 0 = common. So channel i = port i in J.
    int i = channel_idx + 1; // 0-indexed channel_idx to 1-indexed port

    VectorXcd Vi(num_channels_ - 1);
    int col = 0;
    for (int k = 1; k <= num_channels_; ++k) {
        if (k != i) {
            Vi(col++) = J(i, k);
        }
    }
    return Vi;
}

MatrixXcd Manifold::compute_Wi(const MatrixXcd& J, int channel_idx) const {
    // W_i = J with rows/cols for port 0 (common) and port i removed
    int i = channel_idx + 1;
    int dim = num_channels_ - 1;

    MatrixXcd Wi(dim, dim);
    std::vector<int> indices;
    for (int k = 1; k <= num_channels_; ++k) {
        if (k != i) indices.push_back(k);
    }

    for (int r = 0; r < dim; ++r)
        for (int c = 0; c < dim; ++c)
            Wi(r, c) = J(indices[r], indices[c]);

    return Wi;
}

void Manifold::compute_line_lengths(const std::vector<ChannelSpec>& specs) {
    if (static_cast<int>(specs.size()) != num_channels_) {
        throw std::invalid_argument("Manifold::compute_line_lengths: specs size mismatch");
    }

    // Martinez §II: choose line lengths to avoid manifold peaks.
    //
    // For an ideal equal-split lossless T-junction (our TJunction default),
    // the "short-circuit" reflections from the paper are both -1:
    //   P_1 = -S_{32}/(conj(S_{23})*det(S)) = -1
    //   P_2 = -S_{13}/(conj(S_{31})*det(S)) = -1
    // (verified: det(S)=1, S real & symmetric at alpha=0.5)
    //
    // So a manifold peak appears when the load phase seen at a junction
    // port equals pi (i.e. reflection ≈ -1). We pick line lengths to
    // keep the shifted reference-filter reflections away from -1 inside
    // every other channel's passband.
    //
    // Channel line L^c_i: transforms f_i by exp(-j*2*theta_i) where
    //   theta_i = 2*pi*freq*L^c_i. Total round-trip phase = 4*pi*freq*L.
    // The shifted reflection must avoid -1 in the bands of other channels.
    //
    // Manifold line L^m_i: shifts the composite downstream reflection so
    // it does not align with -1 in the band of channel i-1 (walking from
    // the far end toward the common port).

    int N = num_channels_;

    // Build reference Chebyshev filters (per channel)
    std::vector<ChebyshevFilter> refs;
    std::vector<double> fc(N), fs(N);
    refs.reserve(N);
    for (int i = 0; i < N; ++i) {
        fc[i] = (specs[i].freq_left + specs[i].freq_right) / 2.0;
        fs[i] = (specs[i].freq_right - specs[i].freq_left) / 2.0;
        std::vector<Complex> tzs;
        tzs.reserve(specs[i].transmission_zeros.size());
        for (const auto& z : specs[i].transmission_zeros)
            tzs.push_back((z - fc[i]) / fs[i]);
        refs.emplace_back(specs[i].order, tzs, specs[i].return_loss_db);
    }

    auto f_ref = [&](int i, double phys_freq) {
        double nf = (phys_freq - fc[i]) / fs[i];
        Complex s(0, nf);
        Complex F = refs[i].F().evaluate(s);
        Complex E = refs[i].E().evaluate(s);
        if (std::abs(E) < 1e-15) return Complex(1.0, 0.0);
        return F / E;
    };

    // Scan range: up to a half-wavelength at f_max (covers all phases 0..2pi)
    double f_max_band = 0;
    for (int i = 0; i < N; ++i) f_max_band = std::max(f_max_band, specs[i].freq_right);
    double L_max = (f_max_band > 1e-10) ? 0.5 / f_max_band : 1.0;

    const int n_scan = 200;
    const int n_sample = 15;
    const double safety_margin = 0.5; // acceptable distance from -1

    // --- Step 1: channel lines L^c_i ---
    // For each channel i, pick L^c_i that maximizes the minimum distance
    // |f_i(omega) * e^{-j*2*theta_i(omega)} - (-1)| across all other
    // channels' passbands.
    for (int i = 0; i < N; ++i) {
        double best_len = 0;
        double best_cost = -1.0;

        for (int sc = 0; sc < n_scan; ++sc) {
            double L = L_max * sc / static_cast<double>(n_scan - 1);
            double min_dist = std::numeric_limits<double>::infinity();

            for (int k = 0; k < N; ++k) {
                if (k == i) continue;
                for (int t = 0; t < n_sample; ++t) {
                    double freq = specs[k].freq_left
                                + (specs[k].freq_right - specs[k].freq_left)
                                  * t / (n_sample - 1.0);
                    Complex fi = f_ref(i, freq);
                    Complex shift = std::exp(Complex(0, -4.0 * PI * freq * L));
                    double d = std::abs(fi * shift - Complex(-1.0, 0.0));
                    if (d < min_dist) min_dist = d;
                }
            }

            if (min_dist > best_cost) {
                best_cost = min_dist;
                best_len = L;
            }
        }
        channel_lengths_[i] = best_len;
    }

    // --- Step 2: manifold lines L^m_i ---
    // For the serial topology, manifold_lengths_[k] is the segment between
    // junctions k-1 and k (with index 0 being common-to-T1). Following the
    // paper's "walk from far end toward common" rule: decide lengths from
    // k = N-2 down to k = 0.
    //
    // At junction k (= T_{k+1} in the paper), the port toward the far end
    // sees a composite reflection produced by channels {k+1,...,N-1} and
    // downstream manifold lines. Within channel k's passband, we want the
    // phase of this composite (shifted by 2*theta_k from L^m_k) not to
    // coincide with -1.
    //
    // We compute the composite reflection by actually evaluating the
    // current manifold (with channel lines fixed above, and already-set
    // downstream manifold lengths) and attaching reference filters at
    // the channel ports of downstream channels. Scan L^m_k in [0, L_max].

    if (N < 2) return;

    // Build per-channel reference filter S-matrices as 1-port loads: f_ref.
    // To evaluate the composite reflection at a given frequency after
    // attaching filters k+1..N-1 to the manifold, we reuse the same
    // cascade logic as compute_J but terminate the relevant ports.

    // Initial pass: set all manifold_lengths_ to a sensible default so
    // compute_J is well-defined before the scan refines them.
    for (int i = 0; i < N - 1; ++i) {
        if (manifold_lengths_[i] <= 0 || !std::isfinite(manifold_lengths_[i])) {
            double f_ref_band = std::max(std::abs(fc[i]), 1e-6);
            manifold_lengths_[i] = 0.25 / f_ref_band;
        }
    }

    // composite_down(k, freq): reflection seen at junction k's "far-end
    // port" (port 2 in junction S) after terminating channels k+1..N-1
    // with their reference filter reflections and cascading through
    // downstream manifold lines and junctions.
    //
    // We build this incrementally: start at junction N-2 (farthest from
    // common) where the "far end" is just channel N-1 through L^c_{N-1}.
    auto composite_down = [&](int junc_idx, double freq) -> Complex {
        // Walk from rightmost junction (N-2) back to junc_idx+1.
        // At each step we have a current reflection coming from "beyond
        // this junction" toward this junction's port 2.

        // Start at junction (N-2): its far end is channel N-1 only.
        Complex load = f_ref(N - 1, freq);
        // Shift by channel line L^c_{N-1}
        double theta_c = 2.0 * PI * freq * channel_lengths_[N - 1];
        load *= std::exp(Complex(0, -2.0 * theta_c));

        // At each inner junction j (from N-2 down to junc_idx+1),
        // combine: ports (0=from prev, 1=channel j, 2=from next).
        // Known: load on port 2 = current `load`, load on port 1 =
        // f_ref(j) shifted by L^c_j. Reduce to a 1-port reflection at
        // port 0, then shift by the manifold line on port 0.
        for (int j = N - 2; j >= junc_idx + 1; --j) {
            auto Sj = junctions_[j].S();
            Complex ch_load = f_ref(j, freq)
                * std::exp(Complex(0, -4.0 * PI * freq * channel_lengths_[j]));

            // Terminate port 1 with ch_load and port 2 with load:
            //   b_1 = S_10 a_0 + S_11 a_1 + S_12 a_2,  a_1 = ch_load * b_1
            //   b_2 = S_20 a_0 + S_21 a_1 + S_22 a_2,  a_2 = load    * b_2
            // Solve 2x2 for b_1, b_2 in terms of a_0, then b_0 = S_00 a_0 + S_01 a_1 + S_02 a_2.
            Complex S00 = Sj(0,0), S01 = Sj(0,1), S02 = Sj(0,2);
            Complex S10 = Sj(1,0), S11 = Sj(1,1), S12 = Sj(1,2);
            Complex S20 = Sj(2,0), S21 = Sj(2,1), S22 = Sj(2,2);

            // Eq for (b_1, b_2):
            // b_1 - S_11*ch_load*b_1 - S_12*load*b_2 = S_10 a_0
            // b_2 - S_21*ch_load*b_1 - S_22*load*b_2 = S_20 a_0
            Complex a11 = Complex(1) - S11 * ch_load;
            Complex a12 = -S12 * load;
            Complex a21 = -S21 * ch_load;
            Complex a22 = Complex(1) - S22 * load;
            Complex det = a11 * a22 - a12 * a21;
            if (std::abs(det) < 1e-15) { load = Complex(-1.0, 0); continue; }
            // b_1/a_0, b_2/a_0
            Complex b1 = (a22 * S10 - a12 * S20) / det;
            Complex b2 = (a11 * S20 - a21 * S10) / det;
            // Reflection at port 0: b_0/a_0 = S_00 + S_01*ch_load*b_1 + S_02*load*b_2
            load = S00 + S01 * ch_load * b1 + S02 * load * b2;

            // Shift by manifold line on port 0 (length manifold_lengths_[j])
            double theta_m = 2.0 * PI * freq * manifold_lengths_[j];
            load *= std::exp(Complex(0, -2.0 * theta_m));
        }
        return load;
    };

    for (int k = N - 2; k >= 1; --k) {
        // Pick manifold_lengths_[k] so that the composite seen at junction
        // k-1 (i.e. one hop upstream of k) avoids -1 inside channel k's
        // passband. Equivalently the reflection at junction (k-1)'s port 2
        // after the L^m_k shift.
        double best_len = manifold_lengths_[k];
        double best_cost = -1.0;

        for (int sc = 0; sc < n_scan; ++sc) {
            double L = L_max * sc / static_cast<double>(n_scan - 1);
            manifold_lengths_[k] = L;

            double min_dist = std::numeric_limits<double>::infinity();
            // Evaluate in the nearer channel's band (channel k-1 above in
            // the topology, whose TZ concern is avoiding shorts from
            // port 3 through T_{k-1}).
            int band_ch = k - 1;
            for (int t = 0; t < n_sample; ++t) {
                double freq = specs[band_ch].freq_left
                            + (specs[band_ch].freq_right - specs[band_ch].freq_left)
                              * t / (n_sample - 1.0);
                Complex comp = composite_down(k - 1, freq);
                double d = std::abs(comp - Complex(-1.0, 0.0));
                if (d < min_dist) min_dist = d;
            }

            if (min_dist > best_cost) {
                best_cost = min_dist;
                best_len = L;
            }
        }
        manifold_lengths_[k] = best_len;
    }

    // Common-port line L^m_0: pick to keep M_{00} reasonable (avoid
    // resonance of the full assembly). As a pragmatic choice, maximize
    // the distance from -1 of the reflection seen from the common port.
    {
        double best_len = manifold_lengths_[0];
        double best_cost = -1.0;

        for (int sc = 0; sc < n_scan; ++sc) {
            double L = L_max * sc / static_cast<double>(n_scan - 1);
            manifold_lengths_[0] = L;

            double min_dist = std::numeric_limits<double>::infinity();
            for (int ch = 0; ch < N; ++ch) {
                for (int t = 0; t < n_sample; ++t) {
                    double freq = specs[ch].freq_left
                                + (specs[ch].freq_right - specs[ch].freq_left)
                                  * t / (n_sample - 1.0);
                    // Composite seen at common end (crude proxy: composite_down(0, freq)
                    // after one more shift)
                    Complex comp = composite_down(0, freq);
                    double theta = 2.0 * PI * freq * L;
                    comp *= std::exp(Complex(0, -2.0 * theta));
                    double d = std::abs(comp - Complex(-1.0, 0.0));
                    if (d < min_dist) min_dist = d;
                }
            }

            if (min_dist > best_cost) {
                best_cost = min_dist;
                best_len = L;
            }
        }
        manifold_lengths_[0] = best_len;
    }

    (void)safety_margin;  // informational threshold
}

} // namespace np
