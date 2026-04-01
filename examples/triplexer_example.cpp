/**
 * Triplexer synthesis example using Martinez 2019 method.
 *
 * 3 channels with contiguous passbands, synthesized simultaneously
 * to account for manifold coupling effects.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "multiplexer/multiplexer_matching.hpp"

using namespace np;

double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

int main() {
    std::cout << "=== Triplexer Synthesis (Martinez 2019) ===\n\n";

    // Define 3 channels (in GHz)
    std::vector<ChannelSpec> specs = {
        { 3, {}, 14.0, 1.7, 2.0 },   // Channel 1: order 3, 1.7-2.0 GHz
        { 3, {}, 14.0, 2.1, 2.4 },   // Channel 2: order 3, 2.1-2.4 GHz
        { 3, {}, 14.0, 2.5, 2.7 },   // Channel 3: order 3, 2.5-2.7 GHz
    };

    double center_freq = 2.2;  // center of the multiplexer band

    std::cout << "Channels:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Ch" << (i + 1) << ": [" << specs[i].freq_left
                  << ", " << specs[i].freq_right << "], order "
                  << specs[i].order << ", RL " << specs[i].return_loss_db << " dB\n";
    }
    std::cout << "\n";

    MultiplexerMatching mux(specs, center_freq);
    mux.verbose = true;
    mux.path_tracker_h = -0.01;

    auto t0 = std::chrono::steady_clock::now();
    auto cms = mux.run();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nElapsed: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    // Print coupling matrices
    for (int i = 0; i < 3; ++i) {
        if (cms[i].size() == 0) {
            std::cout << "\nChannel " << (i + 1) << ": FAILED\n";
            continue;
        }
        std::cout << "\nChannel " << (i + 1) << " coupling matrix ("
                  << cms[i].rows() << "x" << cms[i].cols() << "):\n";
        for (int r = 0; r < cms[i].rows(); ++r) {
            for (int c = 0; c < cms[i].cols(); ++c) {
                double re = cms[i](r, c).real();
                double im = cms[i](r, c).imag();
                if (std::abs(im) < 1e-8)
                    std::cout << std::setw(9) << std::fixed << std::setprecision(4) << re;
                else
                    std::cout << std::setw(9) << std::fixed << std::setprecision(3)
                              << re << (im >= 0 ? "+" : "") << im << "j";
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }

    // Generate CSV for plotting
    std::ofstream csv("triplexer_response.csv");
    csv << "freq";
    for (int i = 0; i < 3; ++i) csv << ",g" << (i + 1) << "_db";
    csv << "\n";

    for (int k = 0; k <= 500; ++k) {
        double freq = -5.0 + 10.0 * k / 500.0;
        csv << std::fixed << std::setprecision(6) << freq;
        for (int i = 0; i < 3; ++i) {
            Complex g = mux.eval_channel_response(i, freq);
            csv << "," << to_db(std::abs(g));
        }
        csv << "\n";
    }
    csv.close();
    std::cout << "\nSaved: triplexer_response.csv\n";

    return 0;
}
