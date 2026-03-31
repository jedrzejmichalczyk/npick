#include <cmath>
#include <iostream>
#include "impedance_matching.hpp"

using namespace np;

int main() {
    auto load = [](double omega) {
        return Complex(0.2 + 0.05 * omega, 0.15 * omega);
    };

    ImpedanceMatching matcher(load, 3, {}, 20.0, -1.0, 1.0);
    matcher.path_tracker_h = -0.05;
    matcher.path_tracker_run_tol = 1e-3;
    matcher.path_tracker_final_tol = 1e-7;

    const auto& freqs = matcher.interpolation_freqs();
    MatrixXcd cm = matcher.run_single(freqs);
    if (cm.size() == 0) {
        std::cerr << "realization_check: run_single failed\n";
        return 1;
    }

    double max_error = 0.0;
    for (double freq : freqs) {
        Eigen::Matrix2cd S = CouplingMatrix::eval_S(cm, Complex(0, freq));
        double error = std::abs(S(0, 0) - std::conj(load(freq)));
        max_error = std::max(max_error, error);
        std::cout << "freq=" << freq << " error=" << error << "\n";
    }

    std::cout << "max_error=" << max_error << "\n";

    if (!std::isfinite(max_error) || max_error > 1e-4) {
        std::cerr << "realization_check: interpolation mismatch too large\n";
        return 1;
    }

    return 0;
}
