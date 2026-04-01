#pragma once

#include "../types.hpp"
#include "t_junction.hpp"
#include <vector>
#include <cmath>

namespace np {

/**
 * Channel specification for multiplexer synthesis.
 */
struct ChannelSpec {
    int order;
    std::vector<Complex> transmission_zeros;
    double return_loss_db;
    double freq_left;   // Passband edges (physical frequency units)
    double freq_right;
};

/**
 * Manifold for a multiplexer: serial cascade of T-junctions connected
 * by transmission lines.
 *
 * Topology for N channels:
 *   Common port --- [TL_m1] --- T1 --- [TL_c1] --- Filter 1
 *                                |
 *                          [TL_m2] --- T2 --- [TL_c2] --- Filter 2
 *                                       |
 *                                  [TL_m3] --- T3 --- [TL_c3] --- Filter 3
 *                                               |
 *                                          ...
 *
 * Port ordering of the (N+1)-port S-matrix J:
 *   Port 0 = common, Port i = channel i (1-indexed)
 */
class Manifold {
public:
    Manifold(int num_channels, double center_frequency);

    int num_channels() const { return num_channels_; }

    /**
     * Set T-junction at index idx (0-indexed, N-1 junctions total).
     */
    void set_junction(int idx, const TJunction& junction);

    /**
     * Set transmission line lengths.
     * channel_lengths: length from junction to channel filter port (size N)
     * manifold_lengths: length between junctions (size N-1, first is common-to-T1)
     */
    void set_channel_line_lengths(const std::vector<double>& lengths);
    void set_manifold_line_lengths(const std::vector<double>& lengths);

    /**
     * Compute the full (N+1)x(N+1) manifold S-matrix at a given frequency.
     */
    MatrixXcd compute_J(double freq) const;

    /**
     * Extract V_i: row vector [J_{i,k}] for k in {1..N}, k != i.
     * Size: N-1.
     */
    VectorXcd compute_Vi(const MatrixXcd& J, int channel_idx) const;

    /**
     * Extract W_i: submatrix of J with rows/cols 0 and i removed.
     * Size: (N-1) x (N-1).
     */
    MatrixXcd compute_Wi(const MatrixXcd& J, int channel_idx) const;

    /**
     * Compute manifold line lengths to avoid manifold peaks (Martinez Step 1).
     */
    void compute_line_lengths(const std::vector<ChannelSpec>& specs);

    /**
     * Connect port p1 of network S1 to port p2 of network S2.
     * Returns the combined S-matrix with connected ports removed.
     */
    static MatrixXcd connect_ports(const MatrixXcd& S1, int p1,
                                   const MatrixXcd& S2, int p2);

private:
    int num_channels_;
    double center_frequency_;
    std::vector<TJunction> junctions_;       // N-1 junctions
    std::vector<double> channel_lengths_;     // N lengths
    std::vector<double> manifold_lengths_;    // N-1 lengths

    /**
     * Transmission line 2-port S-matrix: phase shift exp(-j*theta).
     */
    static Eigen::Matrix2cd transmission_line_S(double length, double freq);
};

} // namespace np
