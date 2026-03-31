#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "impedance_matching.hpp"

using namespace np;

namespace {

double to_db(double magnitude) {
    if (magnitude < 1e-15) {
        return -300.0;
    }
    return 20.0 * std::log10(magnitude);
}

struct WorstCaseSample {
    double freq = 0.0;
    double magnitude = 0.0;
};

WorstCaseSample worst_g11(ImpedanceMatching& matcher, const MatrixXcd& cm, int samples) {
    WorstCaseSample result;
    const double left = -1.0;
    const double right = 1.0;

    for (int i = 0; i < samples; ++i) {
        double freq = left + (right - left) * i / static_cast<double>(samples - 1);
        double magnitude = std::abs(matcher.eval_G11(cm, freq));
        if (!std::isfinite(magnitude)) {
            result.freq = freq;
            result.magnitude = magnitude;
            return result;
        }
        if (magnitude > result.magnitude) {
            result.freq = freq;
            result.magnitude = magnitude;
        }
    }

    return result;
}

} // namespace

int main(int argc, char** argv) {
    int optimizer_iterations = 0;
    int cost_eval_points = 41;
    int report_samples = 201;

    if (argc > 1) {
        optimizer_iterations = std::stoi(argv[1]);
    }
    if (argc > 2) {
        cost_eval_points = std::stoi(argv[2]);
    }
    if (argc > 3) {
        report_samples = std::stoi(argv[3]);
    }

    std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};
    int order = 8;
    double return_loss = 16.0;
    double freq_left = -1.0;
    double freq_right = 1.0;

    auto load = [](double omega) -> Complex {
        Complex j(0, 1);
        return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
    };

    ImpedanceMatching baseline(load, order, tzs, return_loss, freq_left, freq_right);
    baseline.cost_eval_points = cost_eval_points;
    MatrixXcd cm_baseline = baseline.run_single(baseline.interpolation_freqs());
    if (cm_baseline.size() == 0) {
        std::cerr << "optimization_check: baseline synthesis failed\n";
        return 1;
    }

    double baseline_objective = baseline.compute_cost(baseline.interpolation_freqs());
    WorstCaseSample baseline_worst = worst_g11(baseline, cm_baseline, report_samples);

    ImpedanceMatching optimized(load, order, tzs, return_loss, freq_left, freq_right);
    optimized.optimizer_max_iterations = optimizer_iterations;
    optimized.optimizer_tolerance = 1e-4;
    optimized.cost_eval_points = cost_eval_points;

    auto start = std::chrono::steady_clock::now();
    MatrixXcd cm_optimized = optimized.run();
    auto stop = std::chrono::steady_clock::now();

    if (cm_optimized.size() == 0) {
        std::cerr << "optimization_check: optimized synthesis failed\n";
        return 1;
    }

    double optimized_objective = optimized.compute_cost(optimized.interpolation_freqs());
    WorstCaseSample optimized_worst = worst_g11(optimized, cm_optimized, report_samples);

    int improved = 0;
    int worsened = 0;
    for (int i = 0; i < report_samples; ++i) {
        double freq = freq_left + (freq_right - freq_left) * i / static_cast<double>(report_samples - 1);
        double baseline_mag = std::abs(baseline.eval_G11(cm_baseline, freq));
        double optimized_mag = std::abs(optimized.eval_G11(cm_optimized, freq));
        if (optimized_mag < baseline_mag) {
            ++improved;
        } else if (optimized_mag > baseline_mag) {
            ++worsened;
        }
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "optimizer_iterations=" << optimizer_iterations << "\n";
    std::cout << "cost_eval_points=" << cost_eval_points << "\n";
    std::cout << "report_samples=" << report_samples << "\n";
    std::cout << "elapsed_ms=" << elapsed_ms << "\n";
    std::cout << "baseline_objective_db=" << to_db(baseline_objective) << "\n";
    std::cout << "optimized_objective_db=" << to_db(optimized_objective) << "\n";
    std::cout << "baseline_worst_freq=" << baseline_worst.freq
              << " baseline_worst_db=" << to_db(baseline_worst.magnitude) << "\n";
    std::cout << "optimized_worst_freq=" << optimized_worst.freq
              << " optimized_worst_db=" << to_db(optimized_worst.magnitude) << "\n";
    std::cout << "improvement_db="
              << (to_db(baseline_worst.magnitude) - to_db(optimized_worst.magnitude)) << "\n";
    std::cout << "samples_improved=" << improved << "\n";
    std::cout << "samples_worsened=" << worsened << "\n";
    std::cout << "optimized_freqs=";
    for (double freq : optimized.interpolation_freqs()) {
        std::cout << " " << freq;
    }
    std::cout << "\n";

    return 0;
}
