/**
 * Emscripten bindings for the Nevanlinna-Pick impedance matching solver.
 *
 * Exposes a SolverWrapper class to JavaScript that accepts S-parameter data
 * as flat arrays, runs the solver, and returns results.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

#include "types.hpp"
#include "impedance_matching.hpp"
#include "coupling_matrix.hpp"

using namespace np;
using namespace emscripten;

namespace {

// Linear interpolation of complex reflection coefficient from discrete samples
class InterpolatedLoad {
public:
    InterpolatedLoad(const std::vector<double>& freqs,
                     const std::vector<double>& re,
                     const std::vector<double>& im)
        : freqs_(freqs), re_(re), im_(im) {}

    Complex operator()(double freq) const {
        if (freqs_.empty()) return Complex(0);
        if (freq <= freqs_.front()) return Complex(re_.front(), im_.front());
        if (freq >= freqs_.back()) return Complex(re_.back(), im_.back());

        // Binary search for interval
        auto it = std::lower_bound(freqs_.begin(), freqs_.end(), freq);
        int idx = static_cast<int>(it - freqs_.begin());
        if (idx == 0) idx = 1;

        double f0 = freqs_[idx - 1], f1 = freqs_[idx];
        double t = (freq - f0) / (f1 - f0);

        double r = re_[idx - 1] + t * (re_[idx] - re_[idx - 1]);
        double i = im_[idx - 1] + t * (im_[idx] - im_[idx - 1]);
        return Complex(r, i);
    }

private:
    std::vector<double> freqs_, re_, im_;
};

} // namespace

struct SolverResult {
    bool success;
    double achieved_rl_db;
    std::string error_message;

    // Coupling matrix as flat row-major array + dimension
    std::vector<double> cm_real;
    std::vector<double> cm_imag;
    int cm_size;

    // Optimized interpolation frequencies
    std::vector<double> interp_freqs;
};

struct ResponsePoint {
    double freq;
    double load_db;
    double g11_db;
    double s11_db;
    double s21_db;
};

class SolverWrapper {
public:
    SolverWrapper() = default;

    // Set load data from S-parameter samples
    void set_load_data(val js_freqs, val js_re, val js_im) {
        freqs_ = vecFromJS(js_freqs);
        re_ = vecFromJS(js_re);
        im_ = vecFromJS(js_im);
    }

    // Run the solver
    SolverResult solve(double freq_left, double freq_right, int order,
                       double return_loss_db, val js_tz_re, val js_tz_im) {
        SolverResult result;
        result.success = false;
        result.cm_size = 0;

        if (freqs_.empty()) {
            result.error_message = "No load data set";
            return result;
        }

        // Build transmission zeros
        std::vector<double> tz_re = vecFromJS(js_tz_re);
        std::vector<double> tz_im = vecFromJS(js_tz_im);
        std::vector<Complex> tzs;
        for (size_t i = 0; i < tz_re.size(); ++i) {
            tzs.push_back(Complex(tz_re[i], tz_im[i]));
        }

        // Create interpolated load function
        auto load_data = std::make_shared<InterpolatedLoad>(freqs_, re_, im_);
        auto load_fn = [load_data](double freq) -> Complex {
            return (*load_data)(freq);
        };

        try {
            ImpedanceMatching matcher(load_fn, order, tzs, return_loss_db,
                                      freq_left, freq_right);
            matcher.verbose = false;
            matcher.optimizer_max_iterations = 0;  // equiripple only

            MatrixXcd cm = matcher.run();

            if (cm.size() == 0) {
                result.error_message = "Solver failed to produce a coupling matrix";
                return result;
            }

            result.success = true;
            result.achieved_rl_db = matcher.achieved_return_loss_db();
            result.interp_freqs.assign(
                matcher.interpolation_freqs().begin(),
                matcher.interpolation_freqs().end());

            // Store coupling matrix
            int n = cm.rows();
            result.cm_size = n;
            result.cm_real.resize(n * n);
            result.cm_imag.resize(n * n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    result.cm_real[i * n + j] = cm(i, j).real();
                    result.cm_imag[i * n + j] = cm(i, j).imag();
                }
            }

            // Store for response evaluation
            cm_ = cm;
            load_fn_ = load_fn;

        } catch (const std::exception& e) {
            result.error_message = e.what();
        }

        return result;
    }

    // Evaluate frequency response at N equally-spaced points
    val evaluate_response(double freq_left, double freq_right, int num_points) {
        val points = val::array();

        if (cm_.size() == 0 || !load_fn_) {
            return points;
        }

        for (int i = 0; i < num_points; ++i) {
            double freq = freq_left + (freq_right - freq_left) * i /
                          static_cast<double>(num_points - 1);

            Complex L = load_fn_(freq);
            Complex s(0, freq);
            auto S = CouplingMatrix::eval_S(cm_, s);
            Complex S11 = S(0, 0);
            Complex S12 = S(0, 1);
            Complex S21 = S(1, 0);
            Complex S22 = S(1, 1);

            Complex denom = Complex(1) - S22 * L;
            Complex G11 = (std::abs(denom) > 1e-15)
                ? S11 + S12 * S21 * L / denom
                : Complex(1e10);

            val pt = val::object();
            pt.set("freq", freq);
            pt.set("load_db", to_db(std::abs(L)));
            pt.set("g11_db", to_db(std::abs(G11)));
            pt.set("s11_db", to_db(std::abs(S11)));
            pt.set("s21_db", to_db(std::abs(S21)));
            points.call<void>("push", pt);
        }

        return points;
    }

private:
    static double to_db(double mag) {
        if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
        return 20.0 * std::log10(mag);
    }

    static std::vector<double> vecFromJS(val v) {
        std::vector<double> result;
        int len = v["length"].as<int>();
        result.reserve(len);
        for (int i = 0; i < len; ++i) {
            result.push_back(v[i].as<double>());
        }
        return result;
    }

    std::vector<double> freqs_, re_, im_;
    MatrixXcd cm_;
    std::function<Complex(double)> load_fn_;
};

EMSCRIPTEN_BINDINGS(npick) {
    value_object<SolverResult>("SolverResult")
        .field("success", &SolverResult::success)
        .field("achieved_rl_db", &SolverResult::achieved_rl_db)
        .field("error_message", &SolverResult::error_message)
        .field("cm_real", &SolverResult::cm_real)
        .field("cm_imag", &SolverResult::cm_imag)
        .field("cm_size", &SolverResult::cm_size)
        .field("interp_freqs", &SolverResult::interp_freqs)
        ;

    register_vector<double>("VectorDouble");

    class_<SolverWrapper>("SolverWrapper")
        .constructor<>()
        .function("set_load_data", &SolverWrapper::set_load_data)
        .function("solve", &SolverWrapper::solve)
        .function("evaluate_response", &SolverWrapper::evaluate_response)
        ;
}
