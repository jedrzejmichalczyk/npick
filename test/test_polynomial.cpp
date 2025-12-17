#include <gtest/gtest.h>
#include "polynomial.hpp"
#include "spectral_factor.hpp"
#include <cmath>

using namespace np;

// Helper for comparing complex numbers
bool complex_near(Complex a, Complex b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Polynomial Basic Tests
// =============================================================================

TEST(PolynomialTest, DefaultConstructor) {
    Polynomial<Complex> p;
    EXPECT_EQ(p.degree(), 0);
    EXPECT_TRUE(p.is_zero());
}

TEST(PolynomialTest, ConstructorFromCoeffs) {
    // p(x) = 1 + 2x + 3x^2
    Polynomial<Complex> p({Complex(1), Complex(2), Complex(3)});
    EXPECT_EQ(p.degree(), 2);
    EXPECT_EQ(p.coefficients.size(), 3u);
    EXPECT_TRUE(complex_near(p.leading_coefficient(), Complex(3)));
}

TEST(PolynomialTest, TrimsTrailingZeros) {
    // p(x) = 1 + 2x + 0*x^2 should have degree 1
    Polynomial<Complex> p({Complex(1), Complex(2), Complex(0)});
    EXPECT_EQ(p.degree(), 1);
}

TEST(PolynomialTest, EvaluateHorner) {
    // p(x) = 1 + 2x + 3x^2
    Polynomial<Complex> p({Complex(1), Complex(2), Complex(3)});

    // p(0) = 1
    EXPECT_TRUE(complex_near(p.evaluate(Complex(0)), Complex(1)));

    // p(1) = 1 + 2 + 3 = 6
    EXPECT_TRUE(complex_near(p.evaluate(Complex(1)), Complex(6)));

    // p(2) = 1 + 4 + 12 = 17
    EXPECT_TRUE(complex_near(p.evaluate(Complex(2)), Complex(17)));

    // p(i) = 1 + 2i + 3*(-1) = -2 + 2i
    EXPECT_TRUE(complex_near(p.evaluate(Complex(0, 1)), Complex(-2, 2)));
}

// =============================================================================
// Polynomial Arithmetic Tests
// =============================================================================

TEST(PolynomialTest, Addition) {
    // p(x) = 1 + 2x
    // q(x) = 3 + x + x^2
    // p + q = 4 + 3x + x^2
    Polynomial<Complex> p({Complex(1), Complex(2)});
    Polynomial<Complex> q({Complex(3), Complex(1), Complex(1)});

    auto sum = p + q;
    EXPECT_EQ(sum.degree(), 2);
    EXPECT_TRUE(complex_near(sum.coefficients[0], Complex(4)));
    EXPECT_TRUE(complex_near(sum.coefficients[1], Complex(3)));
    EXPECT_TRUE(complex_near(sum.coefficients[2], Complex(1)));
}

TEST(PolynomialTest, Subtraction) {
    Polynomial<Complex> p({Complex(5), Complex(3)});
    Polynomial<Complex> q({Complex(2), Complex(1)});

    auto diff = p - q;
    EXPECT_TRUE(complex_near(diff.coefficients[0], Complex(3)));
    EXPECT_TRUE(complex_near(diff.coefficients[1], Complex(2)));
}

TEST(PolynomialTest, Multiplication) {
    // p(x) = 1 + x
    // q(x) = 1 - x
    // p * q = 1 - x^2
    Polynomial<Complex> p({Complex(1), Complex(1)});
    Polynomial<Complex> q({Complex(1), Complex(-1)});

    auto prod = p * q;
    EXPECT_EQ(prod.degree(), 2);
    EXPECT_TRUE(complex_near(prod.coefficients[0], Complex(1)));
    EXPECT_TRUE(complex_near(prod.coefficients[1], Complex(0)));
    EXPECT_TRUE(complex_near(prod.coefficients[2], Complex(-1)));
}

TEST(PolynomialTest, ScalarMultiplication) {
    Polynomial<Complex> p({Complex(1), Complex(2)});
    auto scaled = p * Complex(3);

    EXPECT_TRUE(complex_near(scaled.coefficients[0], Complex(3)));
    EXPECT_TRUE(complex_near(scaled.coefficients[1], Complex(6)));
}

// =============================================================================
// Para-Conjugate Tests
// =============================================================================

TEST(PolynomialTest, ParaConjugate) {
    // p(s) = 1 + 2i*s + 3*s^2
    // p*(s) = conj(1)*(-1)^0 + conj(2i)*(-1)^1*s + conj(3)*(-1)^2*s^2
    //       = 1 + 2i*s + 3*s^2  (for this example, note signs!)
    // Actually: p*(s) = 1 - (-2i)*s + 3*s^2 = 1 + 2i*s + 3*s^2

    // Let's test with p(s) = 1 + s
    // p*(s) = conj(1)*1 + conj(1)*(-1)*s = 1 - s
    Polynomial<Complex> p({Complex(1), Complex(1)});
    auto p_star = p.para_conjugate();

    EXPECT_TRUE(complex_near(p_star.coefficients[0], Complex(1)));
    EXPECT_TRUE(complex_near(p_star.coefficients[1], Complex(-1)));

    // Verify: p(s)*p*(s) evaluated on imaginary axis should be real
    Complex s = Complex(0, 1.5);  // s = 1.5i
    Complex val = p.evaluate(s) * p_star.evaluate(s);
    EXPECT_NEAR(val.imag(), 0.0, 1e-10);
}

TEST(PolynomialTest, ParaConjugateProperty) {
    // For any polynomial p, p(iω) * p*(iω) should be real for real ω
    Polynomial<Complex> p({Complex(1, 0.5), Complex(2, -1), Complex(0.5, 0.3)});
    auto p_star = p.para_conjugate();

    for (double omega : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        Complex s = Complex(0, omega);
        Complex product = p.evaluate(s) * p_star.evaluate(s);
        EXPECT_NEAR(product.imag(), 0.0, 1e-10)
            << "Failed at omega = " << omega;
    }
}

// =============================================================================
// Root Finding Tests
// =============================================================================

TEST(PolynomialTest, RootFindingLinear) {
    // p(x) = -2 + x  has root at x = 2
    Polynomial<Complex> p({Complex(-2), Complex(1)});
    auto roots = p.roots();

    EXPECT_EQ(roots.size(), 1u);
    EXPECT_TRUE(complex_near(roots[0], Complex(2)));
}

TEST(PolynomialTest, RootFindingQuadratic) {
    // p(x) = (x - 1)(x - 2) = x^2 - 3x + 2
    Polynomial<Complex> p({Complex(2), Complex(-3), Complex(1)});
    auto roots = p.roots();

    EXPECT_EQ(roots.size(), 2u);

    // Sort roots for consistent comparison
    std::sort(roots.begin(), roots.end(),
              [](Complex a, Complex b) { return a.real() < b.real(); });

    EXPECT_TRUE(complex_near(roots[0], Complex(1)));
    EXPECT_TRUE(complex_near(roots[1], Complex(2)));
}

TEST(PolynomialTest, RootFindingComplex) {
    // p(x) = x^2 + 1 has roots at x = ±i
    Polynomial<Complex> p({Complex(1), Complex(0), Complex(1)});
    auto roots = p.roots();

    EXPECT_EQ(roots.size(), 2u);

    // Sort by imaginary part
    std::sort(roots.begin(), roots.end(),
              [](Complex a, Complex b) { return a.imag() < b.imag(); });

    EXPECT_TRUE(complex_near(roots[0], Complex(0, -1)));
    EXPECT_TRUE(complex_near(roots[1], Complex(0, 1)));
}

TEST(PolynomialTest, FromRoots) {
    // Create polynomial from roots 1, 2, 3
    std::vector<Complex> roots = {Complex(1), Complex(2), Complex(3)};
    auto p = Polynomial<Complex>::from_roots(roots);

    // Verify roots
    for (const auto& r : roots) {
        EXPECT_NEAR(std::abs(p.evaluate(r)), 0.0, 1e-10);
    }

    // Verify it's monic (leading coeff = 1)
    EXPECT_TRUE(complex_near(p.leading_coefficient(), Complex(1)));
}

TEST(PolynomialTest, FromRootsWithLeading) {
    std::vector<Complex> roots = {Complex(1), Complex(-1)};
    auto p = Polynomial<Complex>::from_roots(roots, Complex(5));

    // p(x) = 5(x-1)(x+1) = 5(x^2 - 1) = -5 + 5x^2
    EXPECT_TRUE(complex_near(p.coefficients[0], Complex(-5)));
    EXPECT_TRUE(complex_near(p.coefficients[2], Complex(5)));
}

// =============================================================================
// Spectral Factorization Tests
// =============================================================================

TEST(SpectralFactorTest, SimpleCase) {
    // qq* = (s-1)(s+1)((-s)-1)((-s)+1) for p(s) = s^2 - 1
    // Actually let's do: q(s) = s + 1, q*(s) = -s + 1
    // qq*(s) = (s+1)(-s+1) = -s^2 + 1 = 1 - s^2

    // Create qq* = 1 - s^2
    Polynomial<Complex> qq_star({Complex(1), Complex(0), Complex(-1)});

    auto q = SpectralFactor::factorize(qq_star);

    // q should have degree 1 (one stable root)
    EXPECT_EQ(q.degree(), 1);

    // Verify: q * q.para_conjugate() ≈ qq_star
    auto reconstructed = q * q.para_conjugate();

    // Check at several points
    for (double omega : {0.0, 0.5, 1.0, 2.0}) {
        Complex s = Complex(0, omega);
        EXPECT_NEAR(std::abs(reconstructed.evaluate(s) - qq_star.evaluate(s)), 0.0, 1e-10)
            << "Failed at omega = " << omega;
    }
}

TEST(SpectralFactorTest, FeldtkellerEquation) {
    // Given p and r, verify q from Feldtkeller: qq* = pp* + rr*
    Polynomial<Complex> p({Complex(1), Complex(0.5)});  // p(s) = 1 + 0.5s
    Polynomial<Complex> r({Complex(0.3), Complex(0.2)});  // r(s) = 0.3 + 0.2s

    auto q = SpectralFactor::feldtkeller(p, r);

    // Verify Feldtkeller equation on imaginary axis
    auto p_star = p.para_conjugate();
    auto r_star = r.para_conjugate();
    auto q_star = q.para_conjugate();

    for (double omega : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        Complex s = Complex(0, omega);
        Complex lhs = q.evaluate(s) * q_star.evaluate(s);
        Complex rhs = p.evaluate(s) * p_star.evaluate(s) +
                      r.evaluate(s) * r_star.evaluate(s);

        EXPECT_NEAR(std::abs(lhs - rhs), 0.0, 1e-10)
            << "Feldtkeller failed at omega = " << omega;
    }
}

TEST(SpectralFactorTest, MinimumPhase) {
    // Verify that the spectral factor has all roots in left half-plane
    Polynomial<Complex> p({Complex(1), Complex(0.5), Complex(0.25)});
    Polynomial<Complex> r({Complex(0.3), Complex(0.2), Complex(0.1)});

    auto q = SpectralFactor::feldtkeller(p, r);
    auto roots = q.roots();

    for (const auto& root : roots) {
        EXPECT_LT(root.real(), 1e-10)
            << "Root " << root << " is not in left half-plane";
    }
}

// =============================================================================
// Polynomial Division Tests
// =============================================================================

TEST(PolynomialTest, SyntheticDivision) {
    // (x^2 - 1) / (x - 1) = (x + 1), remainder 0
    Polynomial<Complex> dividend({Complex(-1), Complex(0), Complex(1)});
    Polynomial<Complex> divisor({Complex(-1), Complex(1)});

    auto [quotient, remainder] = Polynomial<Complex>::divide(dividend, divisor);

    // quotient should be x + 1
    EXPECT_EQ(quotient.degree(), 1);
    EXPECT_TRUE(complex_near(quotient.coefficients[0], Complex(1)));
    EXPECT_TRUE(complex_near(quotient.coefficients[1], Complex(1)));

    // remainder should be 0
    EXPECT_TRUE(remainder.is_zero() || std::abs(remainder.coefficients[0]) < 1e-10);
}

TEST(PolynomialTest, Shift) {
    // p(s) = s^2 with roots at 0
    // shift by 1: roots become -i
    Polynomial<Complex> p({Complex(0), Complex(0), Complex(1)});

    auto shifted = p.shift(1.0);

    // The shifted polynomial should have roots at -i (shifted down by i)
    auto roots = shifted.roots();
    for (const auto& r : roots) {
        EXPECT_NEAR(r.imag(), -1.0, 1e-10);
    }
}

// Main function for Google Test
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
