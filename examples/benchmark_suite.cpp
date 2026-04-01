/**
 * Comprehensive benchmark suite for impedance matching.
 *
 * Tests different filter orders, load functions, and transmission zero
 * configurations. Reports timing, achieved return loss, and convergence.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"
#include "coupling_matrix.hpp"
#include "impedance_matching.hpp"

using namespace np;

namespace {

double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

double worst_g11_db(ImpedanceMatching& matcher, const MatrixXcd& cm,
                    double fl, double fr, int samples = 501) {
    double worst = 0.0;
    for (int i = 0; i < samples; ++i) {
        double freq = fl + (fr - fl) * i / static_cast<double>(samples - 1);
        double mag = std::abs(matcher.eval_G11(cm, freq));
        if (std::isfinite(mag) && mag > worst) worst = mag;
    }
    return to_db(worst);
}

struct TestCase {
    std::string name;
    ImpedanceMatching::LoadFunction load;
    int order;
    std::vector<Complex> tzs;
    double return_loss_db;
    double freq_left;
    double freq_right;
};

struct TestResult {
    std::string name;
    double elapsed_ms;
    double achieved_rl_db;
    double worst_g11_db;
    int equiripple_iters;
    bool success;
};

TestResult run_test(const TestCase& tc) {
    TestResult result;
    result.name = tc.name;
    result.success = false;
    result.equiripple_iters = 0;

    try {
        ImpedanceMatching matcher(tc.load, tc.order, tc.tzs,
                                  tc.return_loss_db, tc.freq_left, tc.freq_right);
        matcher.verbose = false;
        matcher.optimizer_max_iterations = 0;  // equiripple only

        auto t0 = std::chrono::steady_clock::now();
        auto cm = matcher.run();
        auto t1 = std::chrono::steady_clock::now();

        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (cm.size() == 0) {
            result.achieved_rl_db = 0.0;
            result.worst_g11_db = 0.0;
            return result;
        }

        result.achieved_rl_db = matcher.achieved_return_loss_db();
        result.worst_g11_db = worst_g11_db(matcher, cm,
                                           tc.freq_left, tc.freq_right);
        result.success = true;

    } catch (const std::exception& e) {
        result.elapsed_ms = 0;
        result.achieved_rl_db = 0;
        result.worst_g11_db = 0;
        std::cerr << "  [" << tc.name << "] EXCEPTION: " << e.what() << "\n";
    }

    return result;
}

} // namespace

int main() {
    std::cout << "============================================\n";
    std::cout << " Nevanlinna-Pick Impedance Matching Benchmark\n";
    std::cout << "============================================\n\n";

    // ---- Load functions ----

    // Load 1: Smooth Gaussian (baseline from existing tests)
    auto load_gaussian = [](double omega) -> Complex {
        Complex j(0, 1);
        return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
    };

    // Load 2: Purely resistive (frequency-independent mismatch)
    auto load_resistive = [](double omega) -> Complex {
        // Gamma = (Z_L - Z_0) / (Z_L + Z_0) with Z_L = 100, Z_0 = 50
        return Complex(1.0 / 3.0, 0.0);
    };

    // Load 3: Reactive load (series RL)
    auto load_reactive_rl = [](double omega) -> Complex {
        // Z_L = 75 + j*30*omega, Z_0 = 50
        Complex Z_L(75.0, 30.0 * omega);
        Complex Z_0(50.0, 0.0);
        return (Z_L - Z_0) / (Z_L + Z_0);
    };

    // Load 4: Capacitive load (smooth, no singularity at omega=0)
    auto load_capacitive = [](double omega) -> Complex {
        // Z_L = 30 + j*20*omega, Z_0 = 50  (series RC-like behavior)
        Complex Z_L(30.0, 20.0 * omega);
        Complex Z_0(50.0, 0.0);
        return (Z_L - Z_0) / (Z_L + Z_0);
    };

    // Load 5: Asymmetric load (breaks symmetry)
    auto load_asymmetric = [](double omega) -> Complex {
        Complex j(0, 1);
        return 0.5 * std::exp(j * 0.3 * omega - 0.1 * omega * omega)
             + 0.1 * std::exp(-j * 2.0 * omega);
    };

    // Load 6: Moderately resonant load
    auto load_resonant = [](double omega) -> Complex {
        // Load with mild resonance, |Gamma| < 1 everywhere
        Complex j(0, 1);
        Complex Z_L = Complex(60.0, 0) + Complex(30.0, 0) * omega / (Complex(1.0, 0) + j * omega);
        Complex Z_0(50.0, 0.0);
        return (Z_L - Z_0) / (Z_L + Z_0);
    };

    // ---- Build test cases ----

    std::vector<TestCase> tests;

    // === Group 1: Varying order with Gaussian load, 2 TZs ===
    std::vector<Complex> tz2 = {Complex(2, 0), Complex(3, 0)};
    for (int order : {2, 4, 6, 8}) {
        // For order 2, can't have 2 TZs - use all-pole
        auto tzs = (order >= 4) ? tz2 : std::vector<Complex>{};
        tests.push_back({
            "Gaussian_order" + std::to_string(order) +
                (tzs.empty() ? "_allpole" : "_tz23"),
            load_gaussian, order, tzs, 16.0, -1.0, 1.0
        });
    }

    // === Group 2: All-pole filters at different orders ===
    for (int order : {3, 5, 7}) {
        tests.push_back({
            "Gaussian_order" + std::to_string(order) + "_allpole",
            load_gaussian, order, {}, 16.0, -1.0, 1.0
        });
    }

    // === Group 3: Different loads with order 6, no TZ ===
    int ord6 = 6;
    std::vector<Complex> no_tz;
    tests.push_back({"Resistive_order6", load_resistive, ord6, no_tz, 20.0, -1.0, 1.0});
    tests.push_back({"ReactiveRL_order6", load_reactive_rl, ord6, no_tz, 16.0, -1.0, 1.0});
    tests.push_back({"Capacitive_order6", load_capacitive, ord6, no_tz, 14.0, -1.0, 1.0});
    tests.push_back({"Asymmetric_order6", load_asymmetric, ord6, no_tz, 14.0, -1.0, 1.0});
    tests.push_back({"Resonant_order6", load_resonant, ord6, no_tz, 12.0, -1.0, 1.0});

    // === Group 4: Different loads with order 8 and TZs ===
    tests.push_back({"Resistive_order8_tz23", load_resistive, 8, tz2, 20.0, -1.0, 1.0});
    tests.push_back({"ReactiveRL_order8_tz23", load_reactive_rl, 8, tz2, 16.0, -1.0, 1.0});
    tests.push_back({"Asymmetric_order8_tz23", load_asymmetric, 8, tz2, 14.0, -1.0, 1.0});

    // === Group 5: Single transmission zero ===
    std::vector<Complex> tz1 = {Complex(2, 0)};
    tests.push_back({"Gaussian_order4_tz2", load_gaussian, 4, tz1, 16.0, -1.0, 1.0});
    tests.push_back({"Gaussian_order6_tz2", load_gaussian, 6, tz1, 16.0, -1.0, 1.0});

    // === Group 6: Narrow band ===
    tests.push_back({"Gaussian_order6_narrow", load_gaussian, 6, no_tz, 16.0, -0.5, 0.5});

    // === Group 7: Sweep target return loss (10, 14, 16, 20 dB) ===
    for (double rl : {10.0, 14.0, 16.0, 20.0}) {
        std::string rl_str = std::to_string(static_cast<int>(rl));

        // Gaussian load, order 6, all-pole
        tests.push_back({
            "Gaussian_o6_RL" + rl_str,
            load_gaussian, 6, no_tz, rl, -1.0, 1.0
        });

        // Gaussian load, order 8, with TZs
        tests.push_back({
            "Gaussian_o8tz_RL" + rl_str,
            load_gaussian, 8, tz2, rl, -1.0, 1.0
        });

        // Reactive RL load, order 6, all-pole
        tests.push_back({
            "ReactiveRL_o6_RL" + rl_str,
            load_reactive_rl, 6, no_tz, rl, -1.0, 1.0
        });

        // Asymmetric load, order 6, all-pole
        tests.push_back({
            "Asymm_o6_RL" + rl_str,
            load_asymmetric, 6, no_tz, rl, -1.0, 1.0
        });
    }

    // ---- Run all tests ----
    std::vector<TestResult> results;
    double total_ms = 0;
    int pass_count = 0;

    std::cout << std::left << std::setw(36) << "Test"
              << std::right << std::setw(10) << "Time(ms)"
              << std::setw(12) << "RL(dB)"
              << std::setw(14) << "Worst G11"
              << std::setw(8) << "Status" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& tc : tests) {
        auto result = run_test(tc);
        results.push_back(result);
        total_ms += result.elapsed_ms;

        std::cout << std::left << std::setw(36) << result.name
                  << std::right << std::fixed << std::setprecision(0)
                  << std::setw(10) << result.elapsed_ms;

        if (result.success) {
            std::cout << std::setprecision(1)
                      << std::setw(12) << result.achieved_rl_db
                      << std::setw(14) << result.worst_g11_db
                      << std::setw(8) << "PASS";
            pass_count++;
        } else {
            std::cout << std::setw(12) << "---"
                      << std::setw(14) << "---"
                      << std::setw(8) << "FAIL";
        }
        std::cout << "\n";
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left << std::setw(36) << "TOTAL"
              << std::right << std::fixed << std::setprecision(0)
              << std::setw(10) << total_ms
              << std::setw(12) << ""
              << std::setw(14) << ""
              << std::setw(8) << (std::to_string(pass_count) + "/" + std::to_string(tests.size()))
              << "\n\n";

    // Verify that solutions are reasonable
    std::cout << "=== Validation ===\n";
    bool all_valid = true;
    for (const auto& r : results) {
        if (!r.success) {
            std::cout << "  WARN: " << r.name << " failed to produce a result\n";
            all_valid = false;
            continue;
        }
        // Check that return loss is negative (valid matching)
        if (r.achieved_rl_db > 0.0) {
            std::cout << "  WARN: " << r.name << " has positive return loss ("
                      << r.achieved_rl_db << " dB)\n";
            all_valid = false;
        }
        // Check that worst G11 is reasonably close to achieved RL
        if (std::abs(r.worst_g11_db - r.achieved_rl_db) > 3.0) {
            std::cout << "  WARN: " << r.name << " G11 mismatch: reported="
                      << r.achieved_rl_db << " measured=" << r.worst_g11_db << " dB\n";
        }
    }

    if (all_valid) {
        std::cout << "  All tests produced valid results.\n";
    }

    return (pass_count == static_cast<int>(tests.size())) ? 0 : 1;
}
