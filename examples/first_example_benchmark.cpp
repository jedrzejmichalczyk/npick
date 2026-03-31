#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "coupling_matrix.hpp"
#include "impedance_matching.hpp"

using namespace np;

namespace {

double to_db(double magnitude) {
    if (!std::isfinite(magnitude) || magnitude < 1e-15) {
        return -300.0;
    }
    return 20.0 * std::log10(magnitude);
}

double to_db(Complex value) {
    return to_db(std::abs(value));
}

struct WorstCase {
    double freq = 0.0;
    double magnitude = 0.0;
};

WorstCase worst_response(
    const std::function<Complex(double)>& response,
    double left,
    double right,
    int samples
) {
    WorstCase worst;
    for (int i = 0; i < samples; ++i) {
        const double freq = left + (right - left) * i / static_cast<double>(samples - 1);
        const double magnitude = std::abs(response(freq));
        if (magnitude > worst.magnitude) {
            worst.freq = freq;
            worst.magnitude = magnitude;
        }
    }
    return worst;
}

} // namespace

int main() {
    try {
        const std::vector<Complex> tzs = {Complex(2.0, 0.0), Complex(3.0, 0.0)};
        const int order = 8;
        const double return_loss = 16.0;
        const double freq_left = -1.0;
        const double freq_right = 1.0;
        const int report_samples = 501;

        auto load = [](double omega) -> Complex {
            const Complex j(0.0, 1.0);
            return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
        };

        ImpedanceMatching baseline(load, order, tzs, return_loss, freq_left, freq_right);
        baseline.cost_eval_points = 41;
        MatrixXcd cm_baseline = baseline.run_single(baseline.interpolation_freqs());
        if (cm_baseline.size() == 0) {
            std::cerr << "baseline synthesis failed\n";
            return 1;
        }

        ImpedanceMatching optimized(load, order, tzs, return_loss, freq_left, freq_right);
        optimized.cost_eval_points = 41;
        optimized.optimizer_max_iterations = 0;
        optimized.verbose = true;
        MatrixXcd cm_optimized = optimized.run();
        if (cm_optimized.size() == 0) {
            std::cerr << "optimized synthesis failed\n";
            return 1;
        }

        std::ofstream csv("first_example_response.csv");
        if (!csv) {
            std::cerr << "failed to open first_example_response.csv\n";
            return 1;
        }

        csv << "freq,load_db,g11_optimized_db,g11_baseline_db,s11_optimized_db\n";
        for (int i = 0; i < report_samples; ++i) {
            const double freq = freq_left + (freq_right - freq_left) * i / static_cast<double>(report_samples - 1);
            const Complex g11_optimized = optimized.eval_G11(cm_optimized, freq);
            const Complex g11_baseline = baseline.eval_G11(cm_baseline, freq);
            const auto s_matrix = CouplingMatrix::eval_S(cm_optimized, Complex(0.0, freq));

            csv << std::fixed << std::setprecision(9)
                << freq << ","
                << to_db(load(freq)) << ","
                << to_db(g11_optimized) << ","
                << to_db(g11_baseline) << ","
                << to_db(s_matrix(0, 0)) << "\n";
        }

        const WorstCase load_worst = worst_response(
            load,
            freq_left,
            freq_right,
            report_samples
        );
        const WorstCase baseline_worst = worst_response(
            [&](double freq) { return baseline.eval_G11(cm_baseline, freq); },
            freq_left,
            freq_right,
            report_samples
        );
        const WorstCase optimized_worst = worst_response(
            [&](double freq) { return optimized.eval_G11(cm_optimized, freq); },
            freq_left,
            freq_right,
            report_samples
        );

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "load_worst_freq=" << load_worst.freq
                  << " load_worst_db=" << to_db(load_worst.magnitude) << "\n";
        std::cout << "baseline_worst_freq=" << baseline_worst.freq
                  << " baseline_worst_db=" << to_db(baseline_worst.magnitude) << "\n";
        std::cout << "optimized_worst_freq=" << optimized_worst.freq
                  << " optimized_worst_db=" << to_db(optimized_worst.magnitude) << "\n";
        std::cout << "optimized_freqs=";
        for (double freq : optimized.interpolation_freqs()) {
            std::cout << " " << freq;
        }
        std::cout << "\n";
        std::cout << "response_csv=first_example_response.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
