#include "multiplexer/manifold.hpp"
#include "chebyshev_filter.hpp"
#include <algorithm>
#include <cmath>
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

    // Martinez Step 1: choose line lengths to avoid manifold peaks.
    //
    // Electrical length theta = 2*pi*freq*length. For a quarter-wave line
    // at frequency f0: theta(f0) = pi/2, so length = 1/(4*f0).
    //
    // For normalized frequencies, we use the channel bandwidth as the
    // reference scale. The line length is set so the phase varies by
    // ~pi/2 across each channel's bandwidth.

    // Compute per-channel center frequencies and bandwidths
    std::vector<double> centers(num_channels_);
    std::vector<double> bandwidths(num_channels_);
    for (int i = 0; i < num_channels_; ++i) {
        centers[i] = (specs[i].freq_left + specs[i].freq_right) / 2.0;
        bandwidths[i] = specs[i].freq_right - specs[i].freq_left;
    }

    // Overall frequency reference: use mean bandwidth for scaling
    double mean_bw = 0;
    for (int i = 0; i < num_channels_; ++i) mean_bw += bandwidths[i];
    mean_bw /= num_channels_;

    // Channel lines: short stubs, length proportional to 1/bandwidth
    // Phase at center = pi/4 (arbitrary reasonable choice)
    for (int i = 0; i < num_channels_; ++i) {
        double f_ref = std::max(std::abs(centers[i]), mean_bw);
        channel_lengths_[i] = 0.25 / (2.0 * f_ref);
    }

    // Manifold lines: spacing between junctions
    // Use different lengths to avoid phase coincidence between channels
    for (int i = 0; i < num_channels_ - 1; ++i) {
        double f_mid = (centers[i] + centers[i + 1]) / 2.0;
        double f_ref = std::max(std::abs(f_mid), mean_bw);
        // Offset each line by a different fraction to avoid resonance
        manifold_lengths_[i] = (0.25 + 0.1 * i) / (2.0 * f_ref);
    }
}

} // namespace np
