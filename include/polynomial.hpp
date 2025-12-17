#pragma once

#include "types.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>

namespace np {

/**
 * Polynomial class with complex coefficients.
 * Coefficients are stored in ascending order: [c0, c1, c2, ..., cn]
 * representing p(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n
 */
template<typename T = Complex>
class Polynomial {
public:
    std::vector<T> coefficients;

    // Constructors
    Polynomial() : coefficients({T(0)}) {}

    explicit Polynomial(std::vector<T> coeffs) {
        if (coeffs.empty()) {
            coefficients = {T(0)};
        } else {
            // Trim trailing zeros
            int deg = degree_of(coeffs);
            coefficients.assign(coeffs.begin(), coeffs.begin() + deg + 1);
        }
    }

    explicit Polynomial(std::initializer_list<T> coeffs)
        : Polynomial(std::vector<T>(coeffs)) {}

    // Degree of polynomial
    int degree() const {
        return degree_of(coefficients);
    }

    // Leading coefficient
    T leading_coefficient() const {
        if (coefficients.empty()) return T(0);
        return coefficients[degree()];
    }

    // Check if polynomial is zero
    bool is_zero() const {
        return coefficients.size() <= 1 &&
               (coefficients.empty() || std::abs(coefficients[0]) < 1e-15);
    }

    // Evaluate polynomial at point x using Horner's method
    T evaluate(T x) const {
        if (is_zero()) return T(0);
        T result = coefficients.back();
        for (int j = static_cast<int>(coefficients.size()) - 2; j >= 0; --j) {
            result = result * x + coefficients[j];
        }
        return result;
    }

    // Para-conjugate: p*(s) = sum of conj(c_k) * (-1)^k * s^k
    Polynomial<T> para_conjugate() const {
        std::vector<T> new_coeffs(coefficients.size());
        for (size_t i = 0; i < coefficients.size(); ++i) {
            new_coeffs[i] = std::conj(coefficients[i]) * std::pow(-1.0, static_cast<int>(i));
        }
        return Polynomial<T>(new_coeffs);
    }

    // Polynomial multiplication
    Polynomial<T> operator*(const Polynomial<T>& other) const {
        if (is_zero() || other.is_zero()) {
            return Polynomial<T>();
        }

        size_t n1 = coefficients.size();
        size_t n2 = other.coefficients.size();
        std::vector<T> result(n1 + n2 - 1, T(0));

        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                result[i + j] += coefficients[i] * other.coefficients[j];
            }
        }

        return Polynomial<T>(result);
    }

    // Scalar multiplication
    Polynomial<T> operator*(T scalar) const {
        std::vector<T> result(coefficients.size());
        for (size_t i = 0; i < coefficients.size(); ++i) {
            result[i] = coefficients[i] * scalar;
        }
        return Polynomial<T>(result);
    }

    friend Polynomial<T> operator*(T scalar, const Polynomial<T>& p) {
        return p * scalar;
    }

    // Polynomial addition
    Polynomial<T> operator+(const Polynomial<T>& other) const {
        size_t max_len = std::max(coefficients.size(), other.coefficients.size());
        std::vector<T> result(max_len, T(0));

        for (size_t i = 0; i < coefficients.size(); ++i) {
            result[i] += coefficients[i];
        }
        for (size_t i = 0; i < other.coefficients.size(); ++i) {
            result[i] += other.coefficients[i];
        }

        return Polynomial<T>(result);
    }

    // Polynomial subtraction
    Polynomial<T> operator-(const Polynomial<T>& other) const {
        return *this + (other * T(-1));
    }

    // Unary negation
    Polynomial<T> operator-() const {
        return *this * T(-1);
    }

    // Add scalar
    Polynomial<T> operator+(T scalar) const {
        Polynomial<T> result = *this;
        if (result.coefficients.empty()) {
            result.coefficients = {scalar};
        } else {
            result.coefficients[0] += scalar;
        }
        return result;
    }

    // Find roots using companion matrix eigenvalues
    std::vector<T> roots() const {
        int n = degree();
        if (n < 1) return {};

        // Normalize coefficients
        T max_coeff = leading_coefficient();
        std::vector<T> norm_coeffs(n + 1);
        for (int i = 0; i <= n; ++i) {
            norm_coeffs[i] = coefficients[i] / max_coeff;
        }

        // Build companion matrix
        Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(n, n);
        for (int i = 0; i < n - 1; ++i) {
            C(i, i + 1) = 1.0;
        }
        for (int i = 0; i < n; ++i) {
            C(n - 1, i) = -norm_coeffs[i];
        }

        // Compute eigenvalues
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(C, false);
        auto eigenvalues = solver.eigenvalues();

        std::vector<T> result(n);
        for (int i = 0; i < n; ++i) {
            result[i] = eigenvalues(i);
        }
        return result;
    }

    // Create polynomial from roots: p(x) = leading * prod(x - r_i)
    static Polynomial<T> from_roots(const std::vector<T>& roots, T leading = T(1)) {
        Polynomial<T> result({T(1)});
        for (const auto& r : roots) {
            result = result * Polynomial<T>({-r, T(1)});
        }
        return result * leading;
    }

    // Shift polynomial: returns p(s + i*shift) by shifting roots
    Polynomial<T> shift(double shift_val) const {
        if (coefficients.size() < 2) return *this;

        auto r = roots();
        T imag_shift = T(0, shift_val);
        for (auto& root : r) {
            root -= imag_shift;
        }
        return from_roots(r, leading_coefficient());
    }

    // Normalize to monic polynomial, returns the leading coefficient
    T normalize() {
        T lc = leading_coefficient();
        if (std::abs(lc) < 1e-15) return T(1);
        for (auto& c : coefficients) {
            c /= lc;
        }
        return lc;
    }

    // Synthetic division: returns (quotient, remainder)
    static std::pair<Polynomial<T>, Polynomial<T>> divide(
        const Polynomial<T>& dividend,
        const Polynomial<T>& divisor
    ) {
        std::vector<T> dividend_rev(dividend.coefficients.rbegin(), dividend.coefficients.rend());
        std::vector<T> divisor_rev(divisor.coefficients.rbegin(), divisor.coefficients.rend());

        if (divisor_rev.empty() || std::abs(divisor_rev[0]) < 1e-15) {
            throw std::runtime_error("Division by zero polynomial");
        }

        std::vector<T> output = dividend_rev;
        T normalizer = divisor_rev[0];

        int sep = static_cast<int>(dividend_rev.size()) - (static_cast<int>(divisor_rev.size()) - 1);

        for (int i = 0; i < sep; ++i) {
            output[i] /= normalizer;
            T coef = output[i];
            if (std::abs(coef) > 1e-15) {
                for (size_t j = 1; j < divisor_rev.size(); ++j) {
                    output[i + j] -= divisor_rev[j] * coef;
                }
            }
        }

        if (sep <= 0) {
            return {Polynomial<T>({T(0)}), dividend};
        }

        std::vector<T> q_coeffs(output.begin(), output.begin() + sep);
        std::vector<T> r_coeffs(output.begin() + sep, output.end());
        std::reverse(q_coeffs.begin(), q_coeffs.end());
        std::reverse(r_coeffs.begin(), r_coeffs.end());

        return {Polynomial<T>(q_coeffs), Polynomial<T>(r_coeffs)};
    }

    // Zero polynomial
    static Polynomial<T> zero() {
        return Polynomial<T>({T(0)});
    }

private:
    // Helper to find actual degree (ignoring trailing zeros)
    static int degree_of(const std::vector<T>& coeffs) {
        for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i) {
            if (std::abs(coeffs[i]) > 1e-15) {
                return i;
            }
        }
        return 0;
    }
};

// Convenience typedef
using PolyC = Polynomial<Complex>;

} // namespace np
