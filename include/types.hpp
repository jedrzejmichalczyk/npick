#pragma once

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <functional>
#include <stdexcept>

namespace np {

// Core types
using Complex = std::complex<double>;
using VectorXcd = Eigen::VectorXcd;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-12;

// Exception types
class ConvergenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SingularMatrixError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SpectralFactorError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace np
