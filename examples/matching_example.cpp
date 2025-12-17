#include <iostream>
#include <iomanip>
#include "polynomial.hpp"
#include "spectral_factor.hpp"
#include "chebyshev_filter.hpp"
#include "coupling_matrix.hpp"
#include "nevanlinna_pick.hpp"
#include "impedance_matching.hpp"

void test_polynomial_operations() {
    using namespace np;

    std::cout << "=== Polynomial Operations ===\n";

    // Create p(s) = 1 + s
    Polynomial<Complex> p({Complex(1), Complex(1)});
    std::cout << "p(s) = 1 + s\n";
    std::cout << "p(2) = " << p.evaluate(Complex(2)) << "\n";
    std::cout << "p(i) = " << p.evaluate(Complex(0, 1)) << "\n";

    // Para-conjugate
    auto p_star = p.para_conjugate();
    std::cout << "p*(s) = " << p_star.coefficients[0] << " + "
              << p_star.coefficients[1] << "*s\n\n";
}

void test_spectral_factorization() {
    using namespace np;

    std::cout << "=== Spectral Factorization ===\n";

    Polynomial<Complex> p({Complex(1), Complex(1)});
    auto p_star = p.para_conjugate();
    auto qq_star = p * p_star;

    auto q = SpectralFactor::factorize(qq_star);
    std::cout << "For qq* = (1+s)(1-s) = 1-s^2:\n";
    std::cout << "Spectral factor q has degree " << q.degree() << "\n";

    // Verify
    Complex test_point(0, 1.5);
    auto q_star = q.para_conjugate();
    auto reconstructed = q * q_star;
    std::cout << "At s = 1.5i:\n";
    std::cout << "  qq*(s) = " << qq_star.evaluate(test_point) << "\n";
    std::cout << "  q(s)*q*(s) = " << reconstructed.evaluate(test_point) << "\n\n";
}

void test_chebyshev_filter() {
    using namespace np;

    std::cout << "=== Chebyshev Filter Synthesis ===\n";

    // 4th order filter with 20 dB return loss, no transmission zeros
    int order = 4;
    std::vector<Complex> tzs;  // All-pole filter
    double return_loss = 20.0;

    ChebyshevFilter filter(order, tzs, return_loss);

    std::cout << "Order: " << order << "\n";
    std::cout << "Return Loss: " << return_loss << " dB\n";
    std::cout << "F degree: " << filter.F().degree() << "\n";
    std::cout << "P degree: " << filter.P().degree() << "\n";
    std::cout << "E degree: " << filter.E().degree() << "\n";

    // Evaluate S11 at passband center
    Complex s(0, 0);  // DC
    Complex F_val = filter.F().evaluate(s);
    Complex E_val = filter.E().evaluate(s);
    Complex S11 = F_val / E_val;
    std::cout << "|S11(0)| = " << std::abs(S11) << "\n\n";
}

void test_coupling_matrix() {
    using namespace np;

    std::cout << "=== Coupling Matrix ===\n";

    // 3rd order filter
    ChebyshevFilter filter(3, {}, 20.0);

    auto cm = CouplingMatrix::from_polynomials(filter.F(), filter.P(), filter.E());
    std::cout << "Coupling matrix size: " << cm.rows() << "x" << cm.cols() << "\n";

    // Print matrix (real parts)
    std::cout << "Coupling matrix (real parts):\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < cm.rows(); ++i) {
        for (int j = 0; j < cm.cols(); ++j) {
            std::cout << std::setw(10) << cm(i,j).real() << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void test_nevanlinna_pick() {
    using namespace np;

    std::cout << "=== Nevanlinna-Pick Solver ===\n";

    // Simple test: target loads equal to initial S-parameters
    // This should converge immediately since start = target
    int order = 3;
    std::vector<Complex> tzs;
    double return_loss = 20.0;

    // Use Chebyshev nodes for frequencies
    auto freqs = chebyshev_nodes(-1.0, 1.0, order);
    std::cout << "Interpolation frequencies: ";
    for (auto f : freqs) std::cout << f << " ";
    std::cout << "\n";

    // Create Chebyshev filter to get initial values
    ChebyshevFilter filter(order, tzs, return_loss);

    // Use initial S-parameters as targets (trivial case)
    std::vector<Complex> loads;
    for (int i = 0; i < order; ++i) {
        Complex s(0, freqs[i]);
        Complex F_val = filter.F().evaluate(s);
        Complex E_val = filter.E().evaluate(s);
        loads.push_back(F_val / E_val);
    }

    std::cout << "Target loads: ";
    for (auto l : loads) std::cout << l << " ";
    std::cout << "\n";

    // Create NevanlinnaPick problem
    NevanlinnaPickNormalized np(loads, freqs, tzs, return_loss);

    std::cout << "Number of variables: " << np.get_num_variables() << "\n";

    // Evaluate at start solution
    auto x0 = np.get_start_solution();
    auto eval = np.eval_map(x0);
    std::cout << "Evaluation at start: ";
    for (int i = 0; i < eval.size(); ++i) std::cout << eval(i) << " ";
    std::cout << "\n";

    // Check homotopy function at t=1
    auto H = np.calc_homotopy_function(x0, 1.0);
    std::cout << "||H(x0, 1)|| = " << H.norm() << " (should be ~0)\n\n";
}

int main() {
    std::cout << "Nevanlinna-Pick Impedance Matching Library\n";
    std::cout << "==========================================\n\n";

    try {
        test_polynomial_operations();
        test_spectral_factorization();
        test_chebyshev_filter();
        test_coupling_matrix();
        test_nevanlinna_pick();

        std::cout << "=== All Tests Passed ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
