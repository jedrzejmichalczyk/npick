/**
 * Test equiripple Newton optimizer vs Nelder-Mead.
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

#include "types.hpp"
#include "coupling_matrix.hpp"
#include "impedance_matching.hpp"

using namespace np;

double to_db(double mag) {
    if (mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

int main() {
    auto load = [](double omega) -> Complex {
        Complex j(0, 1);
        return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
    };

    std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};
    int order = 8;
    double rl = 16.0;
    double fl = -1.0, fr = 1.0;

    // === Equiripple Newton ===
    std::cout << "=== Equiripple Newton optimizer ===\n";
    ImpedanceMatching m(load, order, tzs, rl, fl, fr);
    m.verbose = true;

    auto t0 = std::chrono::steady_clock::now();
    auto cm = m.run();
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "\nResult: " << m.achieved_return_loss_db() << " dB"
              << " (" << std::chrono::duration<double>(t1 - t0).count() << "s)\n";
    std::cout << "Freqs: ";
    for (auto f : m.interpolation_freqs())
        std::cout << std::fixed << std::setprecision(4) << f << " ";
    std::cout << "\n";

    // Generate CSV for plotting
    std::ofstream csv("matching_results.csv");
    csv << "freq,L11_dB,S11_dB,S21_dB,G11_dB\n";
    for (int i = 0; i <= 1000; ++i) {
        double f = -3.0 + 6.0 * i / 1000.0;
        Complex L = load(f);
        auto S = CouplingMatrix::eval_S(cm, Complex(0, f));
        Complex g11 = m.eval_G11(cm, f);
        csv << std::fixed << std::setprecision(6)
            << f << "," << to_db(std::abs(L)) << ","
            << to_db(std::abs(S(0,0))) << "," << to_db(std::abs(S(1,0))) << ","
            << to_db(std::abs(g11)) << "\n";
    }
    csv.close();

    // Generate plot
    std::ofstream py("plot_matching.py");
    py << R"(import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('matching_results.csv', delimiter=',', names=True)
f = data['freq']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, data['L11_dB'], 'b--', lw=1.5, label=r'$|L_{11}|$ (load)')
ax.plot(f, data['S11_dB'], 'g-', lw=1.5, label=r'$|S_{11}|$ (filter alone)')
ax.plot(f, data['G11_dB'], 'r-', lw=2.0, label=r'$|G_{11}|$ (matched)')
ax.plot(f, data['S21_dB'], 'c-', lw=1.0, alpha=0.6, label=r'$|S_{21}|$ (transmission)')
ax.axvspan(-1, 1, alpha=0.07, color='yellow', label='Passband')
ax.set_xlim(-3, 3)
ax.set_ylim(-50, 5)
ax.set_xlabel('Normalized frequency')
ax.set_ylabel('dB')
ax.set_title('Impedance Matching: Equiripple Newton (Order 8, TZ={2,3})')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('matching_results.png', dpi=150)
plt.show()
)";
    py.close();
    std::cout << "\nRun: python plot_matching.py\n";

    return 0;
}
