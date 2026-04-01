#include <gtest/gtest.h>
#include "spectral_factor.hpp"
#include <cmath>

using namespace np;

TEST(SpectralFactorTest, SimpleFactorization) {
    // qq* = (1 + s)(1 - s) = 1 - s^2 => coefficients [1, 0, -1]
    Polynomial<Complex> qq_star({Complex(1), Complex(0), Complex(-1)});

    auto q = SpectralFactor::factorize(qq_star);
    EXPECT_EQ(q.degree(), 1);

    // The root should have negative real part (left half-plane)
    auto roots = q.roots();
    EXPECT_EQ(roots.size(), 1u);
    EXPECT_LT(roots[0].real(), 0.0);
}

TEST(SpectralFactorTest, VerifyFactorizationProduct) {
    // Build qq* = (1+s)(1-s)(2+s)(2-s) = (1-s^2)(4-s^2)
    auto p1 = Polynomial<Complex>({Complex(1), Complex(1)});   // 1+s
    auto p2 = Polynomial<Complex>({Complex(1), Complex(-1)});  // 1-s
    auto p3 = Polynomial<Complex>({Complex(2), Complex(1)});   // 2+s
    auto p4 = Polynomial<Complex>({Complex(2), Complex(-1)});  // 2-s
    auto qq_star = p1 * p2 * p3 * p4;

    auto q = SpectralFactor::factorize(qq_star);
    EXPECT_EQ(q.degree(), 2);

    // Verify q * q_para = qq_star on the imaginary axis
    auto q_para = q.para_conjugate();
    auto product = q * q_para;

    for (double omega = -3.0; omega <= 3.0; omega += 0.5) {
        Complex s(0, omega);
        Complex lhs = product.evaluate(s);
        Complex rhs = qq_star.evaluate(s);
        EXPECT_NEAR(std::abs(lhs - rhs), 0.0, 1e-8 * std::max(1.0, std::abs(rhs)))
            << "Product mismatch at omega=" << omega;
    }
}

TEST(SpectralFactorTest, MinimumPhase) {
    // All roots of the spectral factor must have Re < 0
    auto p = Polynomial<Complex>({Complex(1), Complex(0), Complex(1)});  // 1 + s^2
    auto r = Polynomial<Complex>({Complex(2)});
    auto q = SpectralFactor::feldtkeller(p, r);

    auto roots = q.roots();
    for (const auto& root : roots) {
        EXPECT_LT(root.real(), 1e-10)
            << "Root " << root << " is not in left half-plane";
    }
}

TEST(SpectralFactorTest, FeldtkellerVerification) {
    // Given p and r, verify |q|^2 = |p|^2 + |r|^2 on imaginary axis
    auto p = Polynomial<Complex>({Complex(1), Complex(0), Complex(1)});
    auto r = Polynomial<Complex>({Complex(0.5), Complex(1)});
    auto q = SpectralFactor::feldtkeller(p, r);

    for (double omega = -2.0; omega <= 2.0; omega += 0.25) {
        Complex s(0, omega);
        double q_sq = std::norm(q.evaluate(s));
        double p_sq = std::norm(p.evaluate(s));
        double r_sq = std::norm(r.evaluate(s));

        EXPECT_NEAR(q_sq, p_sq + r_sq, 1e-8 * std::max(1.0, q_sq))
            << "Feldtkeller failed at omega=" << omega;
    }
}

TEST(SpectralFactorTest, CorrectDegree) {
    // Degree of q should equal max(deg(p), deg(r))
    auto p = Polynomial<Complex>({Complex(1), Complex(2), Complex(3)});  // degree 2
    auto r = Polynomial<Complex>({Complex(1)});                          // degree 0
    auto q = SpectralFactor::feldtkeller(p, r);

    EXPECT_EQ(q.degree(), 2);
}
