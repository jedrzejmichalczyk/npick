/**
 * Triplexer synthesis example using Martinez 2019 method.
 *
 * 3 contiguous channels in the 2 GHz range, synthesized simultaneously
 * to account for manifold coupling effects.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "multiplexer/multiplexer_matching.hpp"
#include "coupling_matrix.hpp"

using namespace np;

double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

int main() {
    std::cout << "=== Triplexer Synthesis (Martinez 2019) ===\n\n";

    // Three contiguous channels in the 2 GHz range
    //   Ch A (low):  1.8 - 2.0 GHz, order 4
    //   Ch B (mid):  2.1 - 2.3 GHz, order 4
    //   Ch C (high): 2.4 - 2.6 GHz, order 4
    // Guard bands of 0.1 GHz between channels.
    std::vector<ChannelSpec> specs = {
        { 4, {}, 16.0, 1.8, 2.0 },   // Channel A
        { 4, {}, 16.0, 2.1, 2.3 },   // Channel B
        { 4, {}, 16.0, 2.4, 2.6 },   // Channel C
    };

    double center_freq = 2.2;

    std::cout << "Triplexer specification:\n";
    const char* names[] = {"A (low) ", "B (mid) ", "C (high)"};
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Channel " << names[i] << ": "
                  << specs[i].freq_left << " - " << specs[i].freq_right << " GHz"
                  << ", order " << specs[i].order
                  << ", RL " << specs[i].return_loss_db << " dB\n";
    }
    std::cout << "  Manifold: serial, equal-split T-junctions\n\n";

    MultiplexerMatching mux(specs, center_freq);
    mux.verbose = true;
    mux.equiripple_outer_iterations = 15;

    auto t0 = std::chrono::steady_clock::now();
    auto cms = mux.run();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nElapsed: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    // Print coupling matrices
    for (int i = 0; i < 3; ++i) {
        if (cms[i].size() == 0) {
            std::cout << "\nChannel " << names[i] << ": FAILED\n";
            continue;
        }
        int n = cms[i].rows();
        std::cout << "\nChannel " << names[i] << " coupling matrix ("
                  << n << "x" << n << "):\n";

        // Header
        std::cout << "       ";
        std::cout << "  S  ";
        for (int c = 1; c < n - 1; ++c) std::cout << "   " << c << "  ";
        std::cout << "  L  \n";

        for (int r = 0; r < n; ++r) {
            if (r == 0) std::cout << "  S  ";
            else if (r == n - 1) std::cout << "  L  ";
            else std::cout << "  " << r << "  ";

            for (int c = 0; c < n; ++c) {
                double re = cms[i](r, c).real();
                double im = cms[i](r, c).imag();
                if (std::abs(re) < 1e-6 && std::abs(im) < 1e-6)
                    std::cout << "   .  ";
                else if (std::abs(im) < 1e-6)
                    std::cout << std::setw(6) << std::fixed << std::setprecision(3) << re;
                else
                    std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                              << re << (im >= 0 ? "+" : "") << im << "j";
            }
            std::cout << "\n";
        }
    }

    // Generate comprehensive CSV: per-channel S11, S21, and matched G11
    std::ofstream csv("triplexer_response.csv");
    csv << "freq_ghz";
    for (int i = 0; i < 3; ++i) {
        char ch = 'A' + i;
        csv << ",S11_" << ch << "_dB,S21_" << ch << "_dB,G11_" << ch << "_dB";
    }
    csv << "\n";

    double f_start = 1.5, f_stop = 2.9;
    int n_points = 1001;

    for (int k = 0; k < n_points; ++k) {
        double freq = f_start + (f_stop - f_start) * k / (n_points - 1.0);
        csv << std::fixed << std::setprecision(6) << freq;

        for (int i = 0; i < 3; ++i) {
            if (cms[i].size() == 0) {
                csv << ",0,0,0";
                continue;
            }

            // Evaluate filter S-parameters in normalized frequency
            double fc = (specs[i].freq_left + specs[i].freq_right) / 2.0;
            double fs = (specs[i].freq_right - specs[i].freq_left) / 2.0;
            double norm = (freq - fc) / fs;
            Complex s(0, norm);
            auto S = CouplingMatrix::eval_S(cms[i], s);

            csv << "," << to_db(std::abs(S(0, 0)))
                << "," << to_db(std::abs(S(1, 0)));

            // Matched response through manifold
            Complex g = mux.eval_channel_response(i, freq);
            csv << "," << to_db(std::abs(g));
        }
        csv << "\n";
    }
    csv.close();
    std::cout << "\nSaved: triplexer_response.csv\n";

    // Generate Python plot script
    std::ofstream py("plot_triplexer.py");
    py << R"(import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('triplexer_response.csv', delimiter=',', names=True)
f = data['freq_ghz']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = ['#E53935', '#43A047', '#1E88E5']
labels = ['Ch A (1.8-2.0)', 'Ch B (2.1-2.3)', 'Ch C (2.4-2.6)']
bands = [(1.8, 2.0), (2.1, 2.3), (2.4, 2.6)]

# Top: S11 (reflection) per channel
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax1.plot(f, data[f'S11_{ch}_dB'], color=col, lw=1.5, label=f'|S11| {lbl}')
    ax1.axvspan(band[0], band[1], alpha=0.08, color=col)

ax1.set_ylabel('|S11| (dB)')
ax1.set_ylim(-40, 2)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Triplexer: Per-Channel Filter Response')

# Bottom: S21 (transmission) per channel
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax2.plot(f, data[f'S21_{ch}_dB'], color=col, lw=2, label=f'|S21| {lbl}')
    ax2.axvspan(band[0], band[1], alpha=0.08, color=col)

ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('|S21| (dB)')
ax2.set_ylim(-50, 2)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('triplexer_response.png', dpi=150)
print('Saved: triplexer_response.png')
plt.show()
)";
    py.close();
    std::cout << "Run: python plot_triplexer.py\n";

    return 0;
}

