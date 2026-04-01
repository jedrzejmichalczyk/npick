#include <gtest/gtest.h>
#include "chebyshev_filter.hpp"
#include <cmath>

using namespace np;

TEST(ChebyshevFilterTest, AllPoleOrder4) {
    ChebyshevFilter filter(4, {}, 20.0);

    EXPECT_EQ(filter.order(), 4);
    EXPECT_EQ(filter.F().degree(), 4);
    EXPECT_EQ(filter.E().degree(), 4);
    // P degree = 0 for all-pole (constant)
    EXPECT_EQ(filter.P().degree(), 0);
}

TEST(ChebyshevFilterTest, WithTransmissionZeros) {
    std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};
    ChebyshevFilter filter(6, tzs, 16.0);

    EXPECT_EQ(filter.order(), 6);
    EXPECT_EQ(filter.F().degree(), 6);
    EXPECT_EQ(filter.E().degree(), 6);
    // P has degree = number of finite TZs
    EXPECT_EQ(filter.P().degree(), 2);
}

TEST(ChebyshevFilterTest, FeldtkellerEquation) {
    // Verify |E|^2 = |F|^2 + |P|^2 on the imaginary axis
    ChebyshevFilter filter(4, {}, 20.0);

    for (double omega = -2.0; omega <= 2.0; omega += 0.25) {
        Complex s(0, omega);
        Complex F_val = filter.F().evaluate(s);
        Complex P_val = filter.P().evaluate(s);
        Complex E_val = filter.E().evaluate(s);

        double lhs = std::norm(E_val);
        double rhs = std::norm(F_val) + std::norm(P_val);
        EXPECT_NEAR(lhs, rhs, 1e-8 * std::max(1.0, lhs))
            << "Feldtkeller failed at omega=" << omega;
    }
}

TEST(ChebyshevFilterTest, ReturnLossAtBandCenter) {
    // At omega=0, |S11| should equal the return loss specification
    double rl_db = 20.0;
    ChebyshevFilter filter(4, {}, rl_db);

    Complex s(0, 0);
    Complex S11 = filter.F().evaluate(s) / filter.E().evaluate(s);
    double s11_db = 20.0 * std::log10(std::abs(S11));

    // S11 at band center should be close to -RL
    EXPECT_NEAR(s11_db, -rl_db, 0.5);
}

TEST(ChebyshevFilterTest, Lossless) {
    // |S11|^2 + |S21|^2 = 1 on the imaginary axis
    std::vector<Complex> tzs = {Complex(2, 0)};
    ChebyshevFilter filter(5, tzs, 16.0);

    for (double omega = -3.0; omega <= 3.0; omega += 0.3) {
        Complex s(0, omega);
        Complex S11 = filter.F().evaluate(s) / filter.E().evaluate(s);
        Complex S21 = filter.P().evaluate(s) / filter.E().evaluate(s);

        double sum = std::norm(S11) + std::norm(S21);
        EXPECT_NEAR(sum, 1.0, 1e-8)
            << "Lossless condition violated at omega=" << omega;
    }
}

TEST(ChebyshevFilterTest, ChebyshevNodes) {
    auto nodes = chebyshev_nodes(-1.0, 1.0, 5);
    EXPECT_EQ(nodes.size(), 5u);

    // All nodes should be in [-1, 1]
    for (double n : nodes) {
        EXPECT_GE(n, -1.0);
        EXPECT_LE(n, 1.0);
    }

    // Nodes should be distinct
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            EXPECT_GT(std::abs(nodes[i] - nodes[j]), 1e-10);
        }
    }
}
