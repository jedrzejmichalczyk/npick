#pragma once

#include "../types.hpp"
#include "manifold.hpp"
#include "multiplexer_np.hpp"
#include "../homotopy/path_tracker.hpp"
#include "../impedance_matching.hpp"
#include <vector>

namespace np {

/**
 * High-level multiplexer synthesis orchestrator.
 *
 * Given per-channel specifications and a center frequency, synthesizes
 * a manifold-coupled multiplexer following Martinez et al. (EuMC 2019):
 *
 * 1. Design manifold (compute transmission line lengths)
 * 2. Solve independent NP problems for each channel (start solution)
 * 3. Run coupled homotopy continuation (simultaneous matching)
 * 4. Extract per-channel coupling matrices
 */
class MultiplexerMatching {
public:
    MultiplexerMatching(
        const std::vector<ChannelSpec>& channel_specs,
        double center_frequency
    );

    /**
     * Run the full synthesis pipeline.
     * @return Per-channel coupling matrices
     */
    std::vector<MatrixXcd> run();

    const std::vector<MatrixXcd>& coupling_matrices() const { return cms_; }
    const Manifold& manifold() const { return manifold_; }
    const std::vector<double>& achieved_return_losses_db() const { return achieved_rls_; }

    /**
     * Evaluate per-channel matched response |G_i| at a frequency.
     */
    Complex eval_channel_response(int channel_idx, double freq) const;

    // Configuration
    double path_tracker_h = -0.02;
    double path_tracker_run_tol = 1e-3;
    double path_tracker_final_tol = 1e-7;
    bool verbose = false;

private:
    std::vector<ChannelSpec> specs_;
    Manifold manifold_;
    std::vector<MatrixXcd> cms_;
    std::vector<double> achieved_rls_;
};

} // namespace np
