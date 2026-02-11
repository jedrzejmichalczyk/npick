#include <gtest/gtest.h>
#include "nevanlinna_pick.hpp"

using namespace np;

TEST(NevanlinnaPickValidationTest, ThrowsOnEmptyInputs) {
    std::vector<Complex> loads;
    std::vector<double> freqs;
    EXPECT_THROW(NevanlinnaPick(loads, freqs, {}, 20.0), std::invalid_argument);
}

TEST(NevanlinnaPickValidationTest, ThrowsOnSizeMismatch) {
    std::vector<Complex> loads = {Complex(0.1, 0.2), Complex(-0.2, 0.1)};
    std::vector<double> freqs = {0.0};
    EXPECT_THROW(NevanlinnaPick(loads, freqs, {}, 20.0), std::invalid_argument);
}

TEST(NevanlinnaPickValidationTest, ThrowsOnTooManyTransmissionZeros) {
    std::vector<Complex> loads = {Complex(0.1, 0.2), Complex(-0.2, 0.1)};
    std::vector<double> freqs = {-0.5, 0.5};
    std::vector<Complex> tzs = {Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0)};
    EXPECT_THROW(NevanlinnaPick(loads, freqs, tzs, 20.0), std::invalid_argument);
}

TEST(NevanlinnaPickValidationTest, ThrowsOnInvalidReturnLoss) {
    std::vector<Complex> loads = {Complex(0.1, 0.2), Complex(-0.2, 0.1)};
    std::vector<double> freqs = {-0.5, 0.5};
    EXPECT_THROW(NevanlinnaPick(loads, freqs, {}, 0.0), std::invalid_argument);
}
