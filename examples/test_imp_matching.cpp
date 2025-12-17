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
#include "realization.hpp"
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

        // === CRITICAL VERIFICATION: S11 = conj(L) at interpolation points ===
        std::cout << "\n=== VERIFICATION: S11(cm) = conj(L) at interpolation points ===\n";
        std::cout << std::setw(10) << "Freq"
                 << std::setw(20) << "S11(cm)"
                 << std::setw(20) << "conj(L)"
                 << std::setw(12) << "|diff|"
                 << std::setw(8) << "Status" << "\n";
        std::cout << std::string(70, '-') << "\n";

        double max_mismatch = 0.0;
        bool all_match = true;
        const double tolerance = 1e-5;

        for (size_t i = 0; i < freqs.size(); ++i) {
            double freq = freqs[i];

            // S11 from coupling matrix at interpolation frequency
            auto S = get_realization(freq, cm);
            Complex S11_cm = S(1, 1);

            // Target: conj(L(ω))
            Complex target = std::conj(loads[i]);

            double diff = std::abs(S11_cm - target);
            max_mismatch = std::max(max_mismatch, diff);
            bool match = diff < tolerance;
            all_match = all_match && match;

            std::cout << std::fixed << std::setprecision(4)
                     << std::setw(10) << freq
                     << "  (" << std::setw(7) << S11_cm.real() << "," << std::setw(7) << S11_cm.imag() << ")"
                     << "  (" << std::setw(7) << target.real() << "," << std::setw(7) << target.imag() << ")"
                     << std::setw(12) << std::scientific << std::setprecision(2) << diff
                     << std::setw(8) << (match ? "OK" : "FAIL") << "\n";
        }

        std::cout << std::string(70, '-') << "\n";
        std::cout << "Max mismatch: " << std::scientific << std::setprecision(2) << max_mismatch << "\n";
        std::cout << "MATCHING CRITERION (get_realization): " << (all_match ? "PASSED" : "FAILED") << "\n";
        std::cout << "(Note: from_polynomials assumes para-Hermitian symmetry which NP breaks)\n";

        // === CORRECT VERIFICATION: Use p/q directly ===
        std::cout << "\n=== CORRECT VERIFICATION: S11 = p/q vs conj(L) ===\n";
        std::cout << "(This bypasses the coupling matrix which has symmetry issues)\n";
        std::cout << std::setw(10) << "Freq"
                 << std::setw(20) << "p/q"
                 << std::setw(20) << "conj(L)"
                 << std::setw(12) << "|diff|"
                 << std::setw(8) << "Status" << "\n";
        std::cout << std::string(70, '-') << "\n";

        // Build polynomials from solution
        VectorXcd m_check = NevanlinnaPickNormalized::to_monic(solution);
        std::vector<Complex> p_asc(m_check.size()), r_asc(np_problem.r_coeffs().size());
        for (int i = 0; i < m_check.size(); ++i) p_asc[i] = m_check(m_check.size() - 1 - i);
        for (int i = 0; i < np_problem.r_coeffs().size(); ++i)
            r_asc[i] = np_problem.r_coeffs()(np_problem.r_coeffs().size() - 1 - i);
        Polynomial<Complex> p_check(p_asc);
        Polynomial<Complex> r_check(r_asc);
        auto q_check = SpectralFactor::feldtkeller(p_check, r_check);

        double max_mismatch_pq = 0.0;
        bool all_match_pq = true;
        const double tolerance_pq = 1e-5;

        for (size_t i = 0; i < freqs.size(); ++i) {
            Complex s = Complex(0, freqs[i]);
            Complex S11_pq = p_check.evaluate(s) / q_check.evaluate(s);
            Complex target = std::conj(loads[i]);
            double diff = std::abs(S11_pq - target);
            max_mismatch_pq = std::max(max_mismatch_pq, diff);
            bool match = diff < tolerance_pq;
            all_match_pq = all_match_pq && match;

            std::cout << std::fixed << std::setprecision(4)
                     << std::setw(10) << freqs[i]
                     << "  (" << std::setw(7) << S11_pq.real() << "," << std::setw(7) << S11_pq.imag() << ")"
                     << "  (" << std::setw(7) << target.real() << "," << std::setw(7) << target.imag() << ")"
                     << std::setw(12) << std::scientific << std::setprecision(2) << diff
                     << std::setw(8) << (match ? "OK" : "FAIL") << "\n";
        }

        std::cout << std::string(70, '-') << "\n";
        std::cout << "Max mismatch: " << std::scientific << std::setprecision(2) << max_mismatch_pq << "\n";
        std::cout << "MATCHING CRITERION (p/q direct): " << (all_match_pq ? "PASSED" : "FAILED") << "\n";

        // === DIAGNOSTIC: Test with SIMPLE polynomials (1st order) ===
        std::cout << "\n=== DIAGNOSTIC: Simple 1st order filter test ===\n";
        {
            // Simple 1st order Butterworth: F = s, P = 1, E = s + 1
            // S11 = s/(s+1), S21 = 1/(s+1)
            // At s = j: S11 = j/(j+1) = j(1-j)/2 = (1+j)/2
            Polynomial<Complex> F_simple({Complex(0), Complex(1)});  // s
            Polynomial<Complex> P_simple({Complex(1)});              // 1
            Polynomial<Complex> E_simple({Complex(1), Complex(1)});  // s + 1

            auto cm_simple = CouplingMatrix::from_polynomials(F_simple, P_simple, E_simple);

            Complex s_test = Complex(0, 1);  // s = j
            Complex S11_expected = s_test / (s_test + Complex(1));  // j/(j+1) = (1+j)/2

            auto S_real = get_realization(1.0, cm_simple);
            Complex S11_real = S_real(1, 1);

            std::cout << "Simple filter: F=s, P=1, E=s+1\n";
            std::cout << "At s = j (freq = 1):\n";
            std::cout << "  S11 expected (F/E):    " << S11_expected << "\n";
            std::cout << "  S11 (get_realization): " << S11_real << "\n";
            std::cout << "  |diff| = " << std::abs(S11_real - S11_expected) << "\n";

            std::cout << "\nCoupling matrix (3x3):\n";
            for (int i = 0; i < cm_simple.rows(); ++i) {
                std::cout << "  ";
                for (int j = 0; j < cm_simple.cols(); ++j) {
                    std::cout << cm_simple(i, j) << " ";
                }
                std::cout << "\n";
            }
        }

        // === DIAGNOSTIC: Test with ORIGINAL Chebyshev filter ===
        std::cout << "\n=== DIAGNOSTIC: Original Chebyshev filter ===\n";
        {
            // Build coupling matrix from original Chebyshev filter polynomials
            ChebyshevFilter cheb_filter(order, tzs, return_loss);

            std::cout << "Original F coefficients (should have alternating real/imag):\n  ";
            for (const auto& c : cheb_filter.F().coefficients) {
                std::cout << "(" << c.real() << "," << c.imag() << ") ";
            }
            std::cout << "\n";

            auto cm_orig = CouplingMatrix::from_polynomials(cheb_filter.F(), cheb_filter.P(), cheb_filter.E());

            double test_f = 0.5;
            Complex test_s = Complex(0, test_f);

            // S11 direct: F(s)/E(s)
            Complex F_val = cheb_filter.F().evaluate(test_s);
            Complex E_val = cheb_filter.E().evaluate(test_s);
            Complex S11_FE = F_val / E_val;

            // Test get_realization
            auto S_real = get_realization(test_f, cm_orig);
            Complex S11_real = S_real(1, 1);
            std::cout << "At s = j*" << test_f << ":\n";
            std::cout << "  S11 (F/E):       " << S11_FE << " |" << to_db(S11_FE) << " dB|\n";
            std::cout << "  S11 (get_real):  " << S11_real << " |" << to_db(S11_real) << " dB|\n";
            std::cout << "  |diff| = " << std::abs(S11_real - S11_FE) << "\n";
        }

        // === DIAGNOSTIC: Compare polynomials in calc_coupling_matrix ===
        std::cout << "\n=== DIAGNOSTIC: Modified p polynomial from NP ===\n";
        {
            // Rebuild what calc_coupling_matrix does
            VectorXcd m = NevanlinnaPickNormalized::to_monic(solution);

            std::cout << "Modified p coefficients (descending order from NP):\n  ";
            for (int i = 0; i < m.size(); ++i) {
                std::cout << "(" << m(i).real() << "," << m(i).imag() << ") ";
            }
            std::cout << "\n";
            std::cout << "NOTE: Original has alternating real/imag. Modified has MIXED real+imag.\n";
            std::cout << "This breaks the symmetry assumed by from_polynomials!\n\n";

            // Convert to ascending order (what build_polynomial does)
            std::vector<Complex> p_coeffs_asc(m.size());
            std::vector<Complex> r_coeffs_asc(np_problem.r_coeffs().size());
            for (int i = 0; i < m.size(); ++i) {
                p_coeffs_asc[i] = m(m.size() - 1 - i);
            }
            for (int i = 0; i < np_problem.r_coeffs().size(); ++i) {
                r_coeffs_asc[i] = np_problem.r_coeffs()(np_problem.r_coeffs().size() - 1 - i);
            }
            Polynomial<Complex> p_poly_test(p_coeffs_asc);
            Polynomial<Complex> r_poly_test(r_coeffs_asc);

            // Shift polynomials (shift_ should be 0 here)
            double shift = np_problem.shift();
            std::cout << "shift_ = " << shift << "\n";

            auto p_shifted_test = p_poly_test.shift(-shift);
            auto r_shifted_test = r_poly_test.shift(-shift);
            auto e_poly_test = SpectralFactor::feldtkeller(p_shifted_test, r_shifted_test);

            // Build coupling matrix the same way calc_coupling_matrix does
            auto cm_test = CouplingMatrix::from_polynomials(p_shifted_test, r_shifted_test, e_poly_test);

            // Test S11 at freq 0.5
            double test_f = 0.5556;
            Complex test_s = Complex(0, test_f);

            auto S_test = get_realization(test_f, cm_test);
            Complex S11_test = S_test(1, 1);

            // Direct p/e evaluation
            Complex p_val = p_shifted_test.evaluate(test_s);
            Complex e_val = e_poly_test.evaluate(test_s);
            Complex S11_pe = p_val / e_val;

            std::cout << "At s = j*" << test_f << ":\n";
            std::cout << "  S11 (get_real from rebuilt CM): " << S11_test << "\n";
            std::cout << "  S11 (p/e direct):               " << S11_pe << "\n";
            std::cout << "  |get_real - p/e| = " << std::abs(S11_test - S11_pe) << "\n";

            // Check if cm_test equals cm from before
            std::cout << "  cm_test == cm: " << (cm.isApprox(cm_test, 1e-10) ? "YES" : "NO") << "\n";

            // Verify Feldtkeller equation: |e|² = |p|² + |r|²
            auto p_para = p_shifted_test.para_conjugate();
            auto r_para = r_shifted_test.para_conjugate();
            auto e_para = e_poly_test.para_conjugate();

            // |p|² = p * p*
            auto pp_star = p_shifted_test * p_para;
            auto rr_star = r_shifted_test * r_para;
            auto ee_star = e_poly_test * e_para;

            auto sum = pp_star + rr_star;

            std::cout << "\nFeldtkeller verification (|e|² should equal |p|² + |r|²):\n";
            std::cout << "  |e|² coeffs:        ";
            for (const auto& c : ee_star.coefficients) {
                std::cout << "(" << c.real() << "," << c.imag() << ") ";
            }
            std::cout << "\n  |p|² + |r|² coeffs: ";
            for (const auto& c : sum.coefficients) {
                std::cout << "(" << c.real() << "," << c.imag() << ") ";
            }
            std::cout << "\n";

            // Check coefficient-wise difference
            double max_diff = 0;
            int max_deg = std::max(ee_star.degree(), sum.degree());
            for (int i = 0; i <= max_deg; ++i) {
                Complex ee_c = (i < ee_star.coefficients.size()) ? ee_star.coefficients[i] : Complex(0);
                Complex sum_c = (i < sum.coefficients.size()) ? sum.coefficients[i] : Complex(0);
                max_diff = std::max(max_diff, std::abs(ee_c - sum_c));
            }
            std::cout << "  Max coefficient difference: " << max_diff << "\n";

            // Print Y-parameter poles to check if they're purely imaginary
            double k_np = std::pow(-1, p_shifted_test.degree() + 1);
            double m_sgn = std::pow(-1, p_shifted_test.degree());
            auto e_para_np = e_poly_test.para_conjugate();
            auto p_para_np = p_shifted_test.para_conjugate();
            auto down = e_poly_test + p_shifted_test + (e_para_np + p_para_np) * m_sgn;
            auto y_poles_np = down.roots();

            std::cout << "\nY-parameter poles (should be purely imaginary for lossless):\n";
            for (const auto& pole : y_poles_np) {
                std::cout << "  " << pole << " (|Re|=" << std::abs(pole.real()) << ")\n";
            }

            // For comparison, get original Chebyshev filter Y-poles
            ChebyshevFilter cheb_cmp(order, tzs, return_loss);
            auto E_cmp = cheb_cmp.E();
            auto F_cmp = cheb_cmp.F();
            auto E_para_cmp = E_cmp.para_conjugate();
            auto F_para_cmp = F_cmp.para_conjugate();
            double m_cmp = std::pow(-1, F_cmp.degree());
            auto down_cmp = E_cmp + F_cmp + (E_para_cmp + F_para_cmp) * m_cmp;
            auto y_poles_orig = down_cmp.roots();

            std::cout << "\nOriginal Chebyshev Y-poles:\n";
            for (const auto& pole : y_poles_orig) {
                std::cout << "  " << pole << " (|Re|=" << std::abs(pole.real()) << ")\n";
            }
        }

        // === DIAGNOSTIC: Compare multiple S11 computation methods ===
        std::cout << "\n=== DIAGNOSTIC: S11 computation methods (after NP) ===\n";
        std::cout << "Testing at frequency ω = 0.5556 (one of the interpolation points)\n";

        double test_freq = freqs[2];  // 0.5556
        Complex test_s = Complex(0, test_freq);

        // Method 1: get_realization (full matrix inversion)
        auto S_realization = get_realization(test_freq, cm);
        Complex S11_realization = S_realization(1, 1);

        // Method 2: CouplingMatrix::S11
        Complex S11_formula = CouplingMatrix::S11(cm, test_s);

        // Method 3: Direct p(s)/q(s) from NP solution
        VectorXcd m = NevanlinnaPickNormalized::to_monic(solution);
        // Convert from descending [p_{n-1},...,p_0] to ascending [p_0,...,p_{n-1}]
        std::vector<Complex> p_coeffs_asc(m.size());
        std::vector<Complex> r_coeffs_asc(np_problem.r_coeffs().size());
        for (int i = 0; i < m.size(); ++i) {
            p_coeffs_asc[i] = m(m.size() - 1 - i);
        }
        for (int i = 0; i < np_problem.r_coeffs().size(); ++i) {
            r_coeffs_asc[i] = np_problem.r_coeffs()(np_problem.r_coeffs().size() - 1 - i);
        }
        Polynomial<Complex> p_poly(p_coeffs_asc);
        Polynomial<Complex> r_poly(r_coeffs_asc);
        auto q_poly = SpectralFactor::feldtkeller(p_poly, r_poly);
        Complex p_val = p_poly.evaluate(test_s);
        Complex q_val = q_poly.evaluate(test_s);
        Complex S11_pq = p_val / q_val;

        std::cout << "  S11 (get_realization):  " << S11_realization << " |" << to_db(S11_realization) << " dB|\n";
        std::cout << "  S11 (CM::S11 formula):  " << S11_formula << " |" << to_db(S11_formula) << " dB|\n";
        std::cout << "  S11 (p/q direct):       " << S11_pq << " |" << to_db(S11_pq) << " dB|\n";
        std::cout << "  conj(L):                " << std::conj(load_func(test_freq)) << " |" << to_db(load_func(test_freq)) << " dB|\n";

        std::cout << "\nDifferences:\n";
        std::cout << "  |get_realization - conj(L)| = " << std::abs(S11_realization - std::conj(load_func(test_freq))) << "\n";
        std::cout << "  |CM::S11 - conj(L)| = " << std::abs(S11_formula - std::conj(load_func(test_freq))) << "\n";
        std::cout << "  |p/q - conj(L)| = " << std::abs(S11_pq - std::conj(load_func(test_freq))) << "\n";
        std::cout << "  |get_realization - CM::S11| = " << std::abs(S11_realization - S11_formula) << "\n";
        std::cout << "  |get_realization - p/q| = " << std::abs(S11_realization - S11_pq) << "\n";
        std::cout << "  |CM::S11 - p/q| = " << std::abs(S11_formula - S11_pq) << "\n";

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

        // === TEST: State-space realization-based coupling matrix ===
        std::cout << "\n=== REALIZATION DEBUG: Step-by-step analysis ===\n";
        {
            // Get NP-modified polynomials
            VectorXcd m = NevanlinnaPickNormalized::to_monic(solution);
            std::vector<Complex> p_coeffs_asc(m.size());
            std::vector<Complex> r_coeffs_asc(np_problem.r_coeffs().size());
            for (int i = 0; i < m.size(); ++i) {
                p_coeffs_asc[i] = m(m.size() - 1 - i);
            }
            for (int i = 0; i < np_problem.r_coeffs().size(); ++i) {
                r_coeffs_asc[i] = np_problem.r_coeffs()(np_problem.r_coeffs().size() - 1 - i);
            }
            Polynomial<Complex> F_np(p_coeffs_asc);
            Polynomial<Complex> P_np(r_coeffs_asc);
            auto E_np = SpectralFactor::feldtkeller(F_np, P_np);

            std::cout << "NP-modified polynomials:\n";
            std::cout << "  F degree: " << F_np.degree() << "\n";
            std::cout << "  P degree: " << P_np.degree() << "\n";
            std::cout << "  E degree: " << E_np.degree() << "\n";

            // Manually build realization step by step to debug
            int n = E_np.degree();
            auto poles = E_np.roots();
            std::sort(poles.begin(), poles.end(),
                      [](Complex a, Complex b) { return a.imag() < b.imag(); });

            auto E_deriv = E_np.derivative();

            VectorXcd S11_res(n), S21_res(n);
            for (int i = 0; i < n; ++i) {
                Complex p = poles[i];
                Complex E_deriv_val = E_deriv.evaluate(p);
                if (std::abs(E_deriv_val) < 1e-15) {
                    Complex prod(1.0);
                    for (int j = 0; j < n; ++j) {
                        if (i != j) prod *= (p - poles[j]);
                    }
                    E_deriv_val = prod * E_np.leading_coefficient();
                }
                S11_res(i) = F_np.evaluate(p) / E_deriv_val;
                S21_res(i) = P_np.evaluate(p) / E_deriv_val;
            }

            // Build S-parameter state-space realization
            MatrixXcd A_s = MatrixXcd::Zero(n, n);
            for (int i = 0; i < n; ++i) A_s(i, i) = poles[i];

            VectorXcd tsqrt(n), vovertsqrt(n);
            for (int i = 0; i < n; ++i) {
                tsqrt(i) = complex_sqrt(S11_res(i));
                vovertsqrt(i) = (std::abs(tsqrt(i)) > 1e-15)
                    ? S21_res(i) / tsqrt(i) : Complex(0);
            }

            MatrixXcd B_s(n, 2), C_s(2, n);
            B_s.col(0) = tsqrt;
            B_s.col(1) = vovertsqrt;
            C_s.row(0) = tsqrt.transpose();
            C_s.row(1) = vovertsqrt.transpose();
            MatrixXcd D_s = MatrixXcd::Identity(2, 2);

            // Test S-realization at interpolation points
            std::cout << "\n--- Step 1: S-parameter realization H(s) = D + C*(sI-A)^-1*B ---\n";
            std::cout << "Testing S11 = H(s)[0,0] at interpolation frequencies:\n";
            double test_freq = freqs[2];  // 0.5556
            Complex test_s = Complex(0, test_freq);

            MatrixXcd I_n = MatrixXcd::Identity(n, n);
            MatrixXcd resolvent = (test_s * I_n - A_s).inverse();
            MatrixXcd H_s = D_s + C_s * resolvent * B_s;

            Complex S11_from_real = H_s(0, 0);
            Complex S21_from_real = H_s(1, 0);
            Complex S11_direct = F_np.evaluate(test_s) / E_np.evaluate(test_s);
            Complex S21_direct = P_np.evaluate(test_s) / E_np.evaluate(test_s);
            Complex target = std::conj(load_func(test_freq));

            std::cout << "  At s = j*" << test_freq << ":\n";
            std::cout << "    S11 (realization): " << S11_from_real << "\n";
            std::cout << "    S11 (F/E direct):  " << S11_direct << "\n";
            std::cout << "    S11 target:        " << target << "\n";
            std::cout << "    |real - direct| =  " << std::abs(S11_from_real - S11_direct) << "\n";
            std::cout << "    |real - target| =  " << std::abs(S11_from_real - target) << "\n";
            std::cout << "    S21 (realization): " << S21_from_real << "\n";
            std::cout << "    S21 (P/E direct):  " << S21_direct << "\n";

            // Check all interpolation points for S-realization
            std::cout << "\n  All interpolation points:\n";
            bool s_real_ok = true;
            for (size_t i = 0; i < freqs.size(); ++i) {
                Complex s_i = Complex(0, freqs[i]);
                MatrixXcd res_i = (s_i * I_n - A_s).inverse();
                MatrixXcd H_i = D_s + C_s * res_i * B_s;
                Complex S11_i = H_i(0, 0);
                Complex tgt_i = std::conj(loads[i]);
                double diff = std::abs(S11_i - tgt_i);
                if (diff > 1e-6) s_real_ok = false;
                std::cout << "    freq=" << std::fixed << std::setprecision(4) << freqs[i]
                         << " S11=" << S11_i << " tgt=" << tgt_i
                         << " |diff|=" << std::scientific << diff
                         << (diff < 1e-6 ? " OK" : " FAIL") << "\n";
            }
            std::cout << "  S-REALIZATION: " << (s_real_ok ? "PASSED" : "FAILED") << "\n";

            // Now check what happens after Cayley transform
            std::cout << "\n--- Step 2: After Cayley transform (S -> Y) ---\n";

            // Y = (I - S) * (I + S)^{-1}
            // For state-space: use the formulas
            MatrixXcd I_2 = MatrixXcd::Identity(2, 2);

            // Direct Cayley: Y = (I-S)*(I+S)^{-1}
            // At test point, compute Y directly from S
            MatrixXcd Y_direct = (I_2 - H_s) * (I_2 + H_s).inverse();
            std::cout << "  Y(s) direct from S at s=j*" << test_freq << ":\n";
            std::cout << "    Y11 = " << Y_direct(0,0) << "\n";
            std::cout << "    Y21 = " << Y_direct(1,0) << "\n";

            // The coupling matrix is built from Y-parameters
            // For a transversal CM: Y11 = sum_k (M_Sk^2 / (s - j*lambda_k))
            // The eigenvalues lambda_k should come from diagonalizing Y's A matrix

            std::cout << "\n--- Checking coupling matrix formula ---\n";
            // The standard CM formula for S11:
            // S11 = -j * m_s^T * (s*I - j*M_inner)^{-1} * m_s
            // where M_inner is the inner coupling matrix and m_s are source couplings

            // But this assumes a specific structure. Let's verify with the CM we got:
            std::cout << "  Using get_realization formula on original CM:\n";
            auto S_cm = get_realization(test_freq, cm);
            std::cout << "    S11(CM) = " << S_cm(1,1) << "\n";
            std::cout << "    Target  = " << target << "\n";
            std::cout << "    |diff|  = " << std::abs(S_cm(1,1) - target) << "\n";

            // The issue: get_realization uses W = M - J + lambda*I_inner
            // This is the LOSSLESS formula. For lossy/complex M, we need different formula.

            std::cout << "\n--- Testing direct state-space eval on Y-realization ---\n";

            // Build Y realization manually using Cayley formulas
            // A_y = A_s - B_s * (I + D_s)^{-1} * C_s
            // But (I + D_s) = 2*I for D_s = I, so (I+D_s)^{-1} = 0.5*I
            MatrixXcd I_plus_D_inv = (I_2 + D_s).inverse();
            MatrixXcd A_y = A_s - B_s * I_plus_D_inv * C_s;
            // For Cayley of (A,B,C,D): the formulas are more complex
            // Let's use the Realization class directly

            // Actually let's trace through what from_polynomials_by_realization does
            std::cout << "\n--- Tracing from_polynomials_by_realization ---\n";

            // Import Realization
            Realization s_real(A_s, C_s, B_s, D_s);

            std::cout << "  S-realization eval at s=j*" << test_freq << ": "
                     << s_real.eval(test_s)(0,0) << "\n";

            Realization y_real = s_real.cayley();
            std::cout << "  After Cayley, Y-real eval: " << y_real.eval(test_s)(0,0) << "\n";
            std::cout << "  Y direct (from S matrix):  " << Y_direct(0,0) << "\n";
            std::cout << "  |diff| = " << std::abs(y_real.eval(test_s)(0,0) - Y_direct(0,0)) << "\n";

            Realization y_min = y_real.min_real();
            std::cout << "  After min_real, Y eval:    " << y_min.eval(test_s)(0,0) << "\n";

            Realization y_diag = y_min.diagonalize();
            std::cout << "  After diagonalize, Y eval: " << y_diag.eval(test_s)(0,0) << "\n";

            Realization y_sym = y_diag.symmetrize();
            std::cout << "  After symmetrize, Y eval:  " << y_sym.eval(test_s)(0,0) << "\n";
            std::cout << "  Y target (direct):         " << Y_direct(0,0) << "\n";

            // Check if symmetrize is destroying information
            std::cout << "\n--- Symmetrize analysis ---\n";
            std::cout << "  y_diag.B:\n";
            for (int i = 0; i < std::min(4, (int)y_diag.B.rows()); ++i) {
                std::cout << "    row " << i << ": " << y_diag.B(i, 0) << ", " << y_diag.B(i, 1) << "\n";
            }
            std::cout << "  y_diag.C:\n";
            for (int i = 0; i < std::min(4, (int)y_diag.C.cols()); ++i) {
                std::cout << "    col " << i << ": " << y_diag.C(0, i) << ", " << y_diag.C(1, i) << "\n";
            }
            std::cout << "  y_sym.B (should equal C^T):\n";
            for (int i = 0; i < std::min(4, (int)y_sym.B.rows()); ++i) {
                std::cout << "    row " << i << ": " << y_sym.B(i, 0) << ", " << y_sym.B(i, 1) << "\n";
            }

            // Key insight: For the coupling matrix to work, we need B = C^T
            // But this might not be achievable for non-para-Hermitian systems
            // Let's check if skipping symmetrize helps

            std::cout << "\n--- Testing WITHOUT symmetrize ---\n";
            // Build CM from y_diag directly (without symmetrize)
            int n_diag = y_diag.A.rows();
            MatrixXcd cm_nosym = MatrixXcd::Zero(n_diag + 2, n_diag + 2);
            Complex j(0, 1);
            for (int i = 0; i < n_diag; ++i) {
                cm_nosym(i + 1, i + 1) = j * y_diag.A(i, i);
            }
            // Source couplings from C row 0, load from C row 1
            for (int i = 0; i < n_diag; ++i) {
                cm_nosym(0, i + 1) = y_diag.C(0, i);      // source -> resonator
                cm_nosym(i + 1, 0) = y_diag.B(i, 0);      // resonator -> source
                cm_nosym(n_diag + 1, i + 1) = y_diag.C(1, i);  // load -> resonator
                cm_nosym(i + 1, n_diag + 1) = y_diag.B(i, 1);  // resonator -> load
            }

            // Test this asymmetric CM
            // S = I - 2j * (Y)  where Y = C*(sI - A)^{-1}*B + D
            // But standard get_realization assumes symmetric structure...

            // Let's compute S directly from Y
            MatrixXcd Y_at_test = y_diag.eval(test_s);
            MatrixXcd S_from_Y = (I_2 - Y_at_test) * (I_2 + Y_at_test).inverse();
            std::cout << "  S from Y (inverse Cayley): " << S_from_Y(0,0) << "\n";
            std::cout << "  Target S11:                " << target << "\n";
            std::cout << "  |diff| = " << std::abs(S_from_Y(0,0) - target) << "\n";

            // This tells us if the Y-realization (without symmetrize) gives correct S
            std::cout << "\n  All interpolation points (S from Y, no symmetrize):\n";
            bool y_nosym_ok = true;
            for (size_t i = 0; i < freqs.size(); ++i) {
                Complex s_i = Complex(0, freqs[i]);
                MatrixXcd Y_i = y_diag.eval(s_i);
                MatrixXcd S_i = (I_2 - Y_i) * (I_2 + Y_i).inverse();
                Complex S11_i = S_i(0, 0);
                Complex tgt_i = std::conj(loads[i]);
                double diff = std::abs(S11_i - tgt_i);
                if (diff > 1e-5) y_nosym_ok = false;
                std::cout << "    freq=" << std::fixed << std::setprecision(4) << freqs[i]
                         << " S11=" << S11_i << " |diff|=" << std::scientific << diff
                         << (diff < 1e-5 ? " OK" : " FAIL") << "\n";
            }
            std::cout << "  Y-DIAG (no symmetrize): " << (y_nosym_ok ? "PASSED" : "FAILED") << "\n";

            // === Test from_polynomials_by_realization with get_realization ===
            std::cout << "\n--- Testing from_polynomials_by_realization + get_realization ---\n";
            auto cm_by_real = CouplingMatrix::from_polynomials_by_realization(F_np, P_np, E_np);
            std::cout << "  CM size: " << cm_by_real.rows() << "x" << cm_by_real.cols() << "\n";

            // Print y_sym.A diagonal (eigenvalues)
            std::cout << "  y_sym.A diagonal (Y-poles):\n";
            for (int i = 0; i < std::min(4, (int)y_sym.A.rows()); ++i) {
                std::cout << "    A[" << i << "," << i << "] = " << y_sym.A(i,i) << "\n";
            }
            std::cout << "  CM diagonal (should be j*A):\n";
            for (int i = 0; i < std::min(4, (int)cm_by_real.rows()-2); ++i) {
                std::cout << "    M[" << i+1 << "," << i+1 << "] = " << cm_by_real(i+1,i+1) << "\n";
            }

            bool cm_real_ok = true;
            std::cout << "  Using new CouplingMatrix::eval_S:\n";
            for (size_t i = 0; i < freqs.size(); ++i) {
                Complex s_i = Complex(0, freqs[i]);
                auto S_cmr = CouplingMatrix::eval_S(cm_by_real, s_i);
                Complex S11_cmr = S_cmr(0, 0);
                Complex tgt_i = std::conj(loads[i]);
                double diff = std::abs(S11_cmr - tgt_i);
                if (diff > 1e-4) cm_real_ok = false;
                std::cout << "    freq=" << std::fixed << std::setprecision(4) << freqs[i]
                         << " S11=" << S11_cmr << " tgt=" << tgt_i
                         << " |diff|=" << std::scientific << diff
                         << (diff < 1e-4 ? " OK" : " FAIL") << "\n";
            }
            std::cout << "  from_polynomials_by_realization + eval_S: " << (cm_real_ok ? "PASSED" : "FAILED") << "\n";

            // === Also test symmetrized Y with inverse Cayley ===
            std::cout << "\n--- Testing SYMMETRIZED Y with inverse Cayley ---\n";
            bool y_sym_ok = true;
            for (size_t i = 0; i < freqs.size(); ++i) {
                Complex s_i = Complex(0, freqs[i]);
                MatrixXcd Y_i = y_sym.eval(s_i);
                MatrixXcd S_i = (I_2 - Y_i) * (I_2 + Y_i).inverse();
                Complex S11_i = S_i(0, 0);
                Complex tgt_i = std::conj(loads[i]);
                double diff = std::abs(S11_i - tgt_i);
                if (diff > 1e-5) y_sym_ok = false;
                std::cout << "    freq=" << std::fixed << std::setprecision(4) << freqs[i]
                         << " S11=" << S11_i << " |diff|=" << std::scientific << diff
                         << (diff < 1e-5 ? " OK" : " FAIL") << "\n";
            }
            std::cout << "  Y-SYM (with symmetrize): " << (y_sym_ok ? "PASSED" : "FAILED") << "\n";

        }

        std::cout << "\n=== Test Completed ===\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
