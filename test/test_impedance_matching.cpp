#include <gtest/gtest.h>
#include "impedance_matching.hpp"

using namespace np;

TEST(ImpedanceMatchingValidationTest, ThrowsOnInvalidConstructorArgs) {
    auto load = [](double) { return Complex(0.2, 0.1); };

    EXPECT_THROW(ImpedanceMatching(load, 0, {}, 20.0, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ImpedanceMatching(load, 3, {}, 0.0, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(ImpedanceMatching(load, 3, {}, 20.0, 1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(
        ImpedanceMatching(load, 2, {Complex(2.0, 0.0), Complex(3.0, 0.0), Complex(4.0, 0.0)}, 20.0, -1.0, 1.0),
        std::invalid_argument
    );

    ImpedanceMatching::LoadFunction empty_load;
    EXPECT_THROW(ImpedanceMatching(empty_load, 3, {}, 20.0, -1.0, 1.0), std::invalid_argument);
}

TEST(ImpedanceMatchingValidationTest, RunSingleRejectsWrongFrequencyCount) {
    auto load = [](double) { return Complex(0.2, 0.1); };
    ImpedanceMatching matcher(load, 3, {}, 20.0, -1.0, 1.0);

    MatrixXcd cm = matcher.run_single({-0.5, 0.5});
    EXPECT_EQ(cm.size(), 0);
}

TEST(ImpedanceMatchingValidationTest, ComputeCostRejectsInvalidFrequencySets) {
    auto load = [](double) { return Complex(0.2, 0.1); };
    ImpedanceMatching matcher(load, 3, {}, 20.0, -1.0, 1.0);

    EXPECT_GE(matcher.compute_cost({-0.2, -0.4, 0.5}), 1e10);  // not ordered
    EXPECT_GE(matcher.compute_cost({-2.0, 0.0, 0.5}), 1e10);   // out of bounds
    EXPECT_GE(matcher.compute_cost({-0.5, 0.0}), 1e10);        // wrong size
}

TEST(ImpedanceMatchingRealizationTest, RunSingleRealizesInterpolationTargetsForComplexLoad) {
    auto load = [](double omega) {
        return Complex(0.2 + 0.05 * omega, 0.15 * omega);
    };

    ImpedanceMatching matcher(load, 3, {}, 20.0, -1.0, 1.0);
    matcher.path_tracker_h = -0.05;
    matcher.path_tracker_run_tol = 1e-3;
    matcher.path_tracker_final_tol = 1e-7;

    const auto& freqs = matcher.interpolation_freqs();
    MatrixXcd cm = matcher.run_single(freqs);

    ASSERT_GT(cm.size(), 0);

    for (double freq : freqs) {
        Eigen::Matrix2cd S = CouplingMatrix::eval_S(cm, Complex(0, freq));
        EXPECT_NEAR(std::abs(S(0, 0) - std::conj(load(freq))), 0.0, 1e-4)
            << "freq=" << freq;
    }
}
