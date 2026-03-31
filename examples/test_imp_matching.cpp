/**
 * Test for impedance matching with MinMax optimization.
 *
 * Demonstrates:
 * 1. Define a complex load function L(ω)
 * 2. Run MinMax optimization to find optimal interpolation frequencies
 * 3. Synthesize matching network that minimizes max|G11| over passband
 * 4. Compare with non-optimized (Chebyshev frequencies) result
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "types.hpp"
#include "coupling_matrix.hpp"
#include "impedance_matching.hpp"

using namespace np;

double to_db(double mag) {
    if (mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

double to_db(Complex val) {
    return to_db(std::abs(val));
}

int main() {
    std::cout << "=== Impedance Matching with MinMax Optimization ===\n\n";

    try {
        // Problem parameters
        std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};
        int order = 8;
        double return_loss = 16.0;  // dB
        double freq_left = -1.0;
        double freq_right = 1.0;

        // Load function: L(ω) = 0.45 * exp(jω - 0.25ω²)
        auto load_func = [](double omega) -> Complex {
            Complex j(0, 1);
            return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
        };

        std::cout << "Configuration:\n";
        std::cout << "  Order: " << order << "\n";
        std::cout << "  Return Loss: " << return_loss << " dB\n";
        std::cout << "  Passband: [" << freq_left << ", " << freq_right << "]\n";
        std::cout << "  Transmission zeros: ";
        for (const auto& tz : tzs) std::cout << tz << " ";
        std::cout << "\n\n";

        // === First: Run without optimization (Chebyshev frequencies) ===
        std::cout << "--- Without optimization (Chebyshev frequencies) ---\n";

        ImpedanceMatching matcher_no_opt(load_func, order, tzs, return_loss, freq_left, freq_right);
        matcher_no_opt.verbose = false;

        // Get initial frequencies
        auto init_freqs = matcher_no_opt.interpolation_freqs();
        std::cout << "Initial frequencies: ";
        for (auto f : init_freqs) std::cout << std::fixed << std::setprecision(4) << f << " ";
        std::cout << "\n";

        // Run single iteration (no optimization)
        auto cm_no_opt = matcher_no_opt.run_single(init_freqs);

        // Evaluate max|G11| without optimization
        double max_g11_no_opt = 0.0;
        for (int i = 0; i <= 100; ++i) {
            double freq = freq_left + (freq_right - freq_left) * i / 100.0;
            Complex g11 = matcher_no_opt.eval_G11(cm_no_opt, freq);
            max_g11_no_opt = std::max(max_g11_no_opt, std::abs(g11));
        }
        std::cout << "Max |G11| (no optimization): " << to_db(max_g11_no_opt) << " dB\n\n";

        // === Now: Run with MinMax optimization ===
        std::cout << "--- With MinMax optimization ---\n";

        ImpedanceMatching matcher(load_func, order, tzs, return_loss, freq_left, freq_right);
        matcher.verbose = true;
        matcher.optimizer_max_iterations = 50;
        matcher.optimizer_iteration_cap = 30;
        matcher.optimizer_tolerance = 1e-6;

        auto cm_opt = matcher.run();

        // Get optimized frequencies
        auto opt_freqs = matcher.interpolation_freqs();
        std::cout << "\nOptimized frequencies: ";
        for (auto f : opt_freqs) std::cout << std::fixed << std::setprecision(4) << f << " ";
        std::cout << "\n";

        // Evaluate max|G11| with optimization
        double max_g11_opt = 0.0;
        for (int i = 0; i <= 100; ++i) {
            double freq = freq_left + (freq_right - freq_left) * i / 100.0;
            Complex g11 = matcher.eval_G11(cm_opt, freq);
            max_g11_opt = std::max(max_g11_opt, std::abs(g11));
        }
        std::cout << "Max |G11| (optimized): " << to_db(max_g11_opt) << " dB\n";
        std::cout << "Improvement: " << (to_db(max_g11_no_opt) - to_db(max_g11_opt)) << " dB\n\n";

        // === Generate CSV for plotting ===
        std::cout << "=== Generating plot data ===\n";

        std::ofstream csv("imp_matching_optimized.csv");
        csv << "freq,G11_opt_dB,G11_noopt_dB,S11_opt_dB,L_dB\n";

        for (int i = 0; i <= 500; ++i) {
            double freq = -5.0 + 10.0 * i / 500.0;

            Complex g11_opt = matcher.eval_G11(cm_opt, freq);
            Complex g11_noopt = matcher_no_opt.eval_G11(cm_no_opt, freq);

            Complex s = Complex(0, freq);
            auto S_opt = CouplingMatrix::eval_S(cm_opt, s);
            Complex L = load_func(freq);

            csv << std::fixed << std::setprecision(6)
                << freq << ","
                << to_db(g11_opt) << ","
                << to_db(g11_noopt) << ","
                << to_db(S_opt(0, 0)) << ","
                << to_db(L) << "\n";
        }
        csv.close();
        std::cout << "Saved to: imp_matching_optimized.csv\n";

        // === Generate gnuplot script ===
        std::ofstream gp("plot_optimized.gp");
        gp << "set terminal png size 1200,600\n";
        gp << "set output 'imp_matching_optimized.png'\n";
        gp << "set datafile separator ','\n";
        gp << "set grid\n";
        gp << "set xlabel 'Normalized Frequency'\n";
        gp << "set ylabel 'dB'\n";
        gp << "set title 'Impedance Matching: MinMax Optimization Comparison'\n";
        gp << "set yrange [-50:5]\n";
        gp << "set key outside right\n";
        gp << "plot 'imp_matching_optimized.csv' skip 1 using 1:2 with lines lw 2 title 'G11 (optimized)', \\\n";
        gp << "     '' skip 1 using 1:3 with lines lw 2 title 'G11 (Chebyshev)', \\\n";
        gp << "     '' skip 1 using 1:5 with lines lw 2 title 'L (load)'\n";
        gp.close();
        std::cout << "Saved gnuplot script: plot_optimized.gp\n";

        // === Summary ===
        std::cout << "\n=== Summary ===\n";
        std::cout << "  Without optimization: " << std::fixed << std::setprecision(2)
                  << to_db(max_g11_no_opt) << " dB\n";
        std::cout << "  With optimization:    " << to_db(max_g11_opt) << " dB\n";
        std::cout << "  Improvement:          " << (to_db(max_g11_no_opt) - to_db(max_g11_opt)) << " dB\n";

        std::cout << "\n=== Test Completed ===\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
