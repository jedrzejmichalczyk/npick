/**
 * Duplexer synthesis example — 2 channels, 1 ideal T-junction.
 *
 * Simpler than the triplexer (no line-length interaction between junctions,
 * only 2-channel cross-coupling) — a sanity check that the Martinez pipeline
 * reaches paper-quality (≥20 dB) return loss on a moderate spec.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "multiplexer/multiplexer_matching.hpp"
#include "coupling_matrix.hpp"

using namespace np;

static double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

int main() {
    std::cout << "=== Duplexer Synthesis (Martinez 2019) ===\n\n";

    // 2 narrow channels with small guard band — easier than the 10% triplexer,
    // should comfortably reach the RL target.
    //   Ch A: 1.90 - 1.94 GHz  (2.1% bandwidth, center 1.92)
    //   Ch B: 1.96 - 2.00 GHz  (2.0% bandwidth, center 1.98)
    //   Guard band 20 MHz (1% relative)
    //   Order 4, target 20 dB RL.
    std::vector<ChannelSpec> specs = {
        { 4, {}, 23.0, 1.88, 1.92 },
        { 4, {}, 23.0, 2.00, 2.04 },
    };
    double center_freq = 1.96;

    std::cout << "Duplexer specification:\n";
    const char* names[] = {"A (low) ", "B (high)"};
    for (int i = 0; i < 2; ++i) {
        std::cout << "  Channel " << names[i] << ": "
                  << specs[i].freq_left << " - " << specs[i].freq_right << " GHz"
                  << ", order " << specs[i].order
                  << ", RL " << specs[i].return_loss_db << " dB\n";
    }
    std::cout << "  Manifold: single equal-split T-junction\n\n";

    MultiplexerMatching mux(specs, center_freq);
    mux.verbose = true;
    mux.equiripple_outer_iterations = 15;

    auto t0 = std::chrono::steady_clock::now();
    auto cms = mux.run();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nElapsed: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    // Per-channel CM dump
    for (int i = 0; i < 2; ++i) {
        if (cms[i].size() == 0) {
            std::cout << "\nChannel " << names[i] << ": FAILED\n";
            continue;
        }
        int n = cms[i].rows();
        std::cout << "\nChannel " << names[i] << " coupling matrix ("
                  << n << "x" << n << "):\n";

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

    // Sweep + CSV
    std::ofstream csv("duplexer_response.csv");
    csv << "freq_ghz";
    for (int i = 0; i < 2; ++i) {
        char ch = 'A' + i;
        csv << ",S11_" << ch << "_dB,S21_" << ch << "_dB,G11_" << ch << "_dB";
    }
    csv << "\n";

    double f_start = 1.78, f_stop = 2.14;
    int n_points = 1001;
    for (int k = 0; k < n_points; ++k) {
        double freq = f_start + (f_stop - f_start) * k / (n_points - 1.0);
        csv << std::fixed << std::setprecision(6) << freq;

        for (int i = 0; i < 2; ++i) {
            if (cms[i].size() == 0) { csv << ",0,0,0"; continue; }
            double fc = (specs[i].freq_left + specs[i].freq_right) / 2.0;
            double fs = (specs[i].freq_right - specs[i].freq_left) / 2.0;
            double norm = (freq - fc) / fs;
            Complex s(0, norm);
            auto S = CouplingMatrix::eval_S(cms[i], s);
            csv << "," << to_db(std::abs(S(0, 0)))
                << "," << to_db(std::abs(S(1, 0)));
            Complex g = mux.eval_channel_response(i, freq);
            csv << "," << to_db(std::abs(g));
        }
        csv << "\n";
    }
    csv.close();
    std::cout << "\nSaved: duplexer_response.csv\n";

    std::ofstream py("plot_duplexer.py");
    py << R"(import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('duplexer_response.csv', delimiter=',', names=True)
f = data['freq_ghz']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = ['#E53935', '#1E88E5']
labels = ['Ch A (1.88-1.92)', 'Ch B (2.00-2.04)']
bands = [(1.88, 1.92), (2.00, 2.04)]

for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax1.plot(f, data[f'G11_{ch}_dB'], color=col, lw=1.5, label=f'|G11| {lbl}')
    ax1.axvspan(band[0], band[1], alpha=0.08, color=col)

ax1.set_ylabel('|G11| (dB)')
ax1.set_ylim(-40, 2)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(-20, linestyle='--', alpha=0.4, color='k', label='20 dB target')
ax1.set_title('Duplexer: Matched Reflection through Manifold')

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
plt.savefig('duplexer_response.png', dpi=150)
print('Saved: duplexer_response.png')
)";
    py.close();
    std::cout << "Run: python plot_duplexer.py\n";
    return 0;
}
