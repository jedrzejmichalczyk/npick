/**
 * Port of TestRunImpMatching from C# test suite.
 *
 * This test demonstrates the full impedance matching workflow:
 * 1. Define a complex load function
 * 2. Create an impedance matching problem with transmission zeros
 * 3. Run homotopy path tracking to find optimal matching network
 * 4. Compute S-parameters from the resulting coupling matrix
 * 5. Output CSV data for plotting (like the C# test)
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <functional>

#include "types.hpp"
#include "polynomial.hpp"
#include "chebyshev_filter.hpp"
#include "coupling_matrix.hpp"
#include "nevanlinna_pick.hpp"
#include "impedance_matching.hpp"
#include "homotopy/path_tracker.hpp"

using namespace np;

/**
 * Compute S-parameter matrix from coupling matrix at frequency lambda.
 * Matches C# Utils.getRealization() exactly.
 */
Eigen::Matrix2cd get_realization(double lambda, const MatrixXcd& cm) {
    Complex j(0, 1);
    int N = cm.rows();

    // J matrix: j at source and load positions
    MatrixXcd J = MatrixXcd::Zero(N, N);
    J(0, 0) = j;
    J(N - 1, N - 1) = j;

    // Inner identity: 1 on diagonal except source/load positions
    MatrixXcd I_inner = MatrixXcd::Identity(N, N);
    I_inner(0, 0) = 0;
    I_inner(N - 1, N - 1) = 0;

    // W = M - J + lambda * I_inner
    MatrixXcd W = cm - J + lambda * I_inner;

    // W_inv
    MatrixXcd W_inv = W.inverse();

    // S-parameters (internal computation)
    Complex s11 = Complex(1) + Complex(2) * j * W_inv(0, 0);
    Complex s12 = -Complex(2) * j * W_inv(0, N - 1);
    Complex s21 = -Complex(2) * j * W_inv(N - 1, 0);
    Complex s22 = Complex(1) + Complex(2) * j * W_inv(N - 1, N - 1);

    // Return matrix matching C# convention: {{-s11, -s21}, {-s12, -s22}}
    Eigen::Matrix2cd S;
    S << -s11, -s21,
         -s12, -s22;

    return S;
}

// Convert to dB
double to_db(Complex val) {
    double mag = std::abs(val);
    if (mag < 1e-15) return -300.0;  // Avoid log(0)
    return 20.0 * std::log10(mag);
}

int main() {
    std::cout << "=== Impedance Matching Test (Port of TestRunImpMatching) ===\n\n";

    try {
        // Test parameters from C# test
        std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};  // Transmission zeros
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
        std::cout << "  Frequency range: [" << freq_left << ", " << freq_right << "]\n";
        std::cout << "  Transmission zeros: ";
        for (const auto& tz : tzs) std::cout << tz << " ";
        std::cout << "\n\n";

        // Generate Chebyshev interpolation frequencies
        auto freqs = chebyshev_nodes(freq_left, freq_right, order);

        std::cout << "Interpolation frequencies (Chebyshev nodes):\n  ";
        for (auto f : freqs) std::cout << std::fixed << std::setprecision(4) << f << " ";
        std::cout << "\n\n";

        // Compute target load reflections at interpolation frequencies
        std::vector<Complex> loads;
        for (auto f : freqs) {
            loads.push_back(load_func(f));
        }

        std::cout << "Target loads at interpolation points:\n  ";
        for (const auto& l : loads) {
            std::cout << std::fixed << std::setprecision(4)
                     << "(" << l.real() << "," << l.imag() << ") ";
        }
        std::cout << "\n\n";

        // Create Chebyshev filter to see initial F polynomial
        std::cout << "\nCreating Chebyshev filter for reference...\n";
        ChebyshevFilter ref_filter(order, tzs, return_loss);
        std::cout << "  F degree: " << ref_filter.F().degree() << "\n";
        std::cout << "  F coefficients (ascending): ";
        for (const auto& c : ref_filter.F().coefficients) {
            std::cout << "(" << c.real() << "," << c.imag() << ") ";
        }
        std::cout << "\n";
        std::cout << "  P degree: " << ref_filter.P().degree() << "\n";
        std::cout << "  E degree: " << ref_filter.E().degree() << "\n";

        // Create Nevanlinna-Pick problem
        std::cout << "\nCreating Nevanlinna-Pick problem...\n";
        NevanlinnaPickNormalized np_problem(loads, freqs, tzs, return_loss);

        std::cout << "  Number of variables: " << np_problem.get_num_variables() << "\n";

        // Create path tracker
        std::cout << "\nSetting up path tracker...\n";
        PathTracker tracker(&np_problem);
        tracker.h = -0.05;
        tracker.run_tol = 1e-3;
        tracker.final_tol = 1e-7;
        tracker.verbose = true;

        // Get starting solution
        VectorXcd x0 = np_problem.get_start_solution();

        // Print initial solution
        std::cout << "\nInitial solution x0 (polynomial coefficients):\n  ";
        for (int i = 0; i < x0.size(); ++i) {
            std::cout << "(" << x0[i].real() << "," << x0[i].imag() << ") ";
        }
        std::cout << "\n";

        // Check H(x0, 1) = 0
        VectorXcd h_init = np_problem.calc_homotopy_function(x0, 1.0);
        std::cout << "||H(x0, 1)|| = " << h_init.norm() << "\n";

        // Run path tracking
        std::cout << "\nRunning homotopy path tracking from t=1 to t=0...\n";
        VectorXcd solution = tracker.run(x0);

        std::cout << "\nPath tracking completed!\n";
        std::cout << "  Number of steps: " << tracker.get_ts().size() << "\n";

        // Print final solution
        std::cout << "\nFinal solution (polynomial coefficients):\n  ";
        for (int i = 0; i < solution.size(); ++i) {
            std::cout << "(" << solution[i].real() << "," << solution[i].imag() << ") ";
        }
        std::cout << "\n";

        // Check H(solution, 0) = 0
        VectorXcd h_final = np_problem.calc_homotopy_function(solution, 0.0);
        std::cout << "||H(solution, 0)|| = " << h_final.norm() << "\n";

        // Evaluate p(iω)/q(iω) at interpolation points
        std::cout << "\nEvaluation at final solution (should match conj(loads)):\n";
        VectorXcd eval_final = np_problem.eval_map(solution);
        for (int i = 0; i < eval_final.size(); ++i) {
            Complex conj_load = std::conj(loads[i]);
            std::cout << "  freq=" << std::fixed << std::setprecision(4) << freqs[i]
                     << ": eval=(" << eval_final[i].real() << "," << eval_final[i].imag() << ")"
                     << " conj(L)=(" << conj_load.real() << "," << conj_load.imag() << ")"
                     << " diff=" << std::abs(eval_final[i] - conj_load) << "\n";
        }

        // Extract coupling matrix from solution
        std::cout << "\nExtracting coupling matrix...\n";
        MatrixXcd cm = np_problem.calc_coupling_matrix(solution);

        std::cout << "Coupling matrix size: " << cm.rows() << "x" << cm.cols() << "\n";
        std::cout << "Coupling matrix (real parts):\n";
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < cm.rows(); ++i) {
            std::cout << "  ";
            for (int j = 0; j < cm.cols(); ++j) {
                std::cout << std::setw(10) << cm(i, j).real() << " ";
            }
            std::cout << "\n";
        }

        // === Output CSV for plotting (matching C# test range: -5 to 5, 1001 points) ===
        std::cout << "\n=== Generating plot data ===\n";

        double xo = -5.0;
        double xk = 5.0;
        int P = 1001;

        // Open CSV file
        std::ofstream csv("imp_matching_results.csv");
        csv << "freq,G11_dB,S11_dB,S21_dB,S22_dB,L_dB,G21_dB\n";

        for (int i = 0; i < P; ++i) {
            double freq = xo + (xk - xo) * i / (P - 1);

            auto S = get_realization(freq, cm);
            // C# indexing: S11=[1,1], S12=[0,1], S21=[1,0], S22=[0,0]
            Complex S11 = S(1, 1);
            Complex S12 = S(0, 1);
            Complex S21 = S(1, 0);
            Complex S22 = S(0, 0);

            Complex L = load_func(freq);

            // G11 = S11 + S12*S21*L / (1 - S22*L)
            Complex denom = Complex(1) - S22 * L;
            Complex G11 = S11 + S12 * S21 * L / denom;

            // G21 = 1 - |G11|^2 (power transmission)
            double G21 = 1.0 - std::norm(G11);

            csv << std::fixed << std::setprecision(6)
                << freq << ","
                << to_db(G11) << ","
                << to_db(S11) << ","
                << to_db(S21) << ","
                << to_db(S22) << ","
                << to_db(L) << ","
                << to_db(G21) << "\n";
        }
        csv.close();
        std::cout << "Saved to: imp_matching_results.csv\n";

        // === Also create gnuplot script ===
        std::ofstream gp("plot_results.gp");
        gp << "set terminal png size 1200,800\n";
        gp << "set output 'imp_matching_plot.png'\n";
        gp << "set multiplot layout 2,1\n";
        gp << "set datafile separator ','\n";
        gp << "set grid\n";
        gp << "set xlabel 'Normalized Frequency'\n";
        gp << "set ylabel 'dB'\n";
        gp << "\n";
        gp << "# Top plot: G11, S11, L\n";
        gp << "set title 'Impedance Matching: Reflection Coefficients'\n";
        gp << "set yrange [-50:5]\n";
        gp << "set key outside right\n";
        gp << "plot 'imp_matching_results.csv' skip 1 using 1:2 with lines lw 2 title 'G11 (matched)', \\\n";
        gp << "     '' skip 1 using 1:3 with lines lw 2 title 'S11 (network)', \\\n";
        gp << "     '' skip 1 using 1:6 with lines lw 2 title 'L (load)'\n";
        gp << "\n";
        gp << "# Bottom plot: S21, S22\n";
        gp << "set title 'Matching Network S-Parameters'\n";
        gp << "set yrange [-50:5]\n";
        gp << "plot 'imp_matching_results.csv' skip 1 using 1:4 with lines lw 2 title 'S21', \\\n";
        gp << "     '' skip 1 using 1:5 with lines lw 2 title 'S22'\n";
        gp << "\n";
        gp << "unset multiplot\n";
        gp.close();
        std::cout << "Saved gnuplot script: plot_results.gp\n";
        std::cout << "Run: gnuplot plot_results.gp\n";

        // === Compare S11 from coupling matrix vs p/q ===
        std::cout << "\n=== Debug: Compare S11 sources ===\n";
        std::cout << "Freq    |S11_cm|dB  |p/q|dB     |L|dB\n";
        std::cout << std::string(50, '-') << "\n";

        for (int i = 0; i <= 10; ++i) {
            double freq = -1.0 + 2.0 * i / 10;

            // S11 from coupling matrix
            auto S = get_realization(freq, cm);
            Complex S11_cm = S(1, 1);

            // p/q from eval_map at this frequency (need to evaluate at single point)
            // For now just show S11 vs L
            Complex L = load_func(freq);

            std::cout << std::fixed << std::setprecision(3)
                     << std::setw(6) << freq
                     << std::setw(12) << to_db(S11_cm)
                     << std::setw(12) << to_db(std::conj(L))
                     << std::setw(12) << to_db(L) << "\n";
        }

        // === Print summary table for passband ===
        std::cout << "\n=== Passband Response (freq in [-1, 1]) ===\n";
        std::cout << std::setw(10) << "Freq" << std::setw(12) << "G11(dB)"
                 << std::setw(12) << "S11(dB)" << std::setw(12) << "S21(dB)"
                 << std::setw(12) << "L(dB)" << "\n";
        std::cout << std::string(58, '-') << "\n";

        double max_g11_passband = -1000;
        for (int i = 0; i <= 20; ++i) {
            double freq = -1.0 + 2.0 * i / 20;

            auto S = get_realization(freq, cm);
            Complex S11 = S(1, 1);
            Complex S12 = S(0, 1);
            Complex S21 = S(1, 0);
            Complex S22 = S(0, 0);
            Complex L = load_func(freq);
            Complex G11 = S11 + S12 * S21 * L / (Complex(1) - S22 * L);

            double g11_db = to_db(G11);
            max_g11_passband = std::max(max_g11_passband, g11_db);

            std::cout << std::fixed << std::setprecision(3)
                     << std::setw(10) << freq
                     << std::setw(12) << g11_db
                     << std::setw(12) << to_db(S11)
                     << std::setw(12) << to_db(S21)
                     << std::setw(12) << to_db(L) << "\n";
        }

        std::cout << "\n=== Results Summary ===\n";
        std::cout << "Max |G11| in passband: " << std::fixed << std::setprecision(2)
                 << max_g11_passband << " dB\n";

        std::cout << "\n=== Test Completed ===\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
