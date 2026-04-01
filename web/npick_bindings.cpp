/**
 * Emscripten bindings for the Nevanlinna-Pick impedance matching solver.
 *
 * Exposes a SolverWrapper class to JavaScript that accepts S-parameter data
 * as flat arrays, runs the solver, and returns results via getter methods.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cmath>
#include <string>

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

double to_db(double mag) {
    if (!std::isfinite(mag) || mag < 1e-15) return -300.0;
    return 20.0 * std::log10(mag);
}

std::vector<double> vecFromJS(val v) {
    std::vector<double> result;
    int len = v["length"].as<int>();
    result.reserve(len);
    for (int i = 0; i < len; ++i) {
        result.push_back(v[i].as<double>());
    }
    return result;
}

} // namespace

class SolverWrapper {
public:
    SolverWrapper() = default;

    void set_load_data(val js_freqs, val js_re, val js_im) {
        freqs_ = vecFromJS(js_freqs);
        re_ = vecFromJS(js_re);
        im_ = vecFromJS(js_im);
    }

    // Run the solver. Returns true on success.
    // Frequencies are normalized internally to avoid numerical overflow.
    bool solve(double freq_left, double freq_right, int order,
               double return_loss_db, val js_tz_re, val js_tz_im) {
        success_ = false;
        error_message_ = "";
        achieved_rl_db_ = 0;
        cm_ = MatrixXcd();
        cm_size_ = 0;

        if (freqs_.empty()) {
            error_message_ = "No load data set";
            return false;
        }

        // Normalize frequencies: shift to center, scale to [-1, 1]
        freq_center_ = (freq_left + freq_right) / 2.0;
        freq_scale_ = (freq_right - freq_left) / 2.0;

        if (freq_scale_ <= 0) {
            error_message_ = "Invalid frequency band";
            return false;
        }

        double norm_left = -1.0;
        double norm_right = 1.0;

        // Normalize sample frequencies for interpolation
        std::vector<double> norm_freqs(freqs_.size());
        for (size_t i = 0; i < freqs_.size(); ++i) {
            norm_freqs[i] = (freqs_[i] - freq_center_) / freq_scale_;
        }

        // Build transmission zeros (normalized)
        std::vector<double> tz_re = vecFromJS(js_tz_re);
        std::vector<double> tz_im = vecFromJS(js_tz_im);
        std::vector<Complex> tzs;
        for (size_t i = 0; i < tz_re.size(); ++i) {
            double norm_tz = (tz_re[i] - freq_center_) / freq_scale_;
            tzs.push_back(Complex(norm_tz, tz_im[i] / freq_scale_));
        }

        // Create interpolated load using normalized frequencies
        load_data_ = std::make_shared<InterpolatedLoad>(norm_freqs, re_, im_);
        auto load_data = load_data_;
        load_fn_ = [load_data](double freq) -> Complex {
            return (*load_data)(freq);
        };

        try {
            // Try with requested return loss first, then relax if needed
            double rl_attempts[] = { return_loss_db,
                                     std::min(return_loss_db, 12.0),
                                     std::min(return_loss_db, 8.0),
                                     std::min(return_loss_db, 5.0) };
            double step_sizes[] = { -0.05, -0.02, -0.01, -0.01 };

            for (int attempt = 0; attempt < 4; ++attempt) {
                double rl = rl_attempts[attempt];
                if (attempt > 0 && rl >= rl_attempts[attempt - 1])
                    continue;  // skip if same as previous

                try {
                    ImpedanceMatching matcher(load_fn_, order, tzs, rl,
                                              norm_left, norm_right);
                    matcher.verbose = false;
                    matcher.optimizer_max_iterations = 0;
                    matcher.path_tracker_h = step_sizes[attempt];

                    MatrixXcd cm = matcher.run();

                    if (cm.size() == 0)
                        continue;

                    success_ = true;
                    achieved_rl_db_ = matcher.achieved_return_loss_db();
                    cm_ = cm;
                    cm_size_ = cm.rows();

                    interp_freqs_.assign(
                        matcher.interpolation_freqs().begin(),
                        matcher.interpolation_freqs().end());

                    return true;
                } catch (...) {
                    continue;
                }
            }

            error_message_ = "Solver could not converge. Try a narrower band, "
                             "lower order, or lower return loss.";
            return false;

        } catch (const std::exception& e) {
            error_message_ = e.what();
            return false;
        } catch (...) {
            error_message_ = "Unknown solver error";
            return false;
        }
    }

    // --- Getters for results (called from JS after solve) ---

    bool get_success() const { return success_; }
    double get_achieved_rl_db() const { return achieved_rl_db_; }
    std::string get_error_message() const { return error_message_; }
    int get_cm_size() const { return cm_size_; }

    // Get coupling matrix element (i, j) real part
    double get_cm_real(int i, int j) const {
        if (i < 0 || i >= cm_size_ || j < 0 || j >= cm_size_) return 0;
        return cm_(i, j).real();
    }

    // Get coupling matrix element (i, j) imaginary part
    double get_cm_imag(int i, int j) const {
        if (i < 0 || i >= cm_size_ || j < 0 || j >= cm_size_) return 0;
        return cm_(i, j).imag();
    }

    // Evaluate frequency response, returns flat JS arrays via val.
    // freq_left/freq_right are in original (un-normalized) units.
    val evaluate_response(double freq_left, double freq_right, int num_points) {
        val result = val::object();

        val js_freq = val::array();
        val js_load = val::array();
        val js_g11 = val::array();
        val js_s11 = val::array();
        val js_s21 = val::array();

        if (cm_.size() == 0 || !load_fn_) {
            result.set("freq", js_freq);
            result.set("load_db", js_load);
            result.set("g11_db", js_g11);
            result.set("s11_db", js_s11);
            result.set("s21_db", js_s21);
            return result;
        }

        for (int i = 0; i < num_points; ++i) {
            // Original frequency for display and load evaluation
            double freq_orig = freq_left + (freq_right - freq_left) * i /
                               static_cast<double>(num_points - 1);

            // Normalized frequency for coupling matrix evaluation
            double freq_norm = (freq_orig - freq_center_) / freq_scale_;

            Complex L = load_fn_(freq_norm);
            Complex s(0, freq_norm);
            auto S = CouplingMatrix::eval_S(cm_, s);

            Complex denom = Complex(1) - S(1, 1) * L;
            Complex G11 = (std::abs(denom) > 1e-15)
                ? S(0, 0) + S(0, 1) * S(1, 0) * L / denom
                : Complex(1e10);

            js_freq.call<void>("push", freq_orig);
            js_load.call<void>("push", to_db(std::abs(L)));
            js_g11.call<void>("push", to_db(std::abs(G11)));
            js_s11.call<void>("push", to_db(std::abs(S(0, 0))));
            js_s21.call<void>("push", to_db(std::abs(S(1, 0))));
        }

        result.set("freq", js_freq);
        result.set("load_db", js_load);
        result.set("g11_db", js_g11);
        result.set("s11_db", js_s11);
        result.set("s21_db", js_s21);
        return result;
    }

private:
    std::vector<double> freqs_, re_, im_;
    std::shared_ptr<InterpolatedLoad> load_data_;
    MatrixXcd cm_;
    std::function<Complex(double)> load_fn_;

    // Frequency normalization: norm = (orig - center) / scale
    double freq_center_ = 0;
    double freq_scale_ = 1;

    bool success_ = false;
    double achieved_rl_db_ = 0;
    std::string error_message_;
    int cm_size_ = 0;
    std::vector<double> interp_freqs_;
};

EMSCRIPTEN_BINDINGS(npick) {
    class_<SolverWrapper>("SolverWrapper")
        .constructor<>()
        .function("set_load_data", &SolverWrapper::set_load_data)
        .function("solve", &SolverWrapper::solve)
        .function("get_success", &SolverWrapper::get_success)
        .function("get_achieved_rl_db", &SolverWrapper::get_achieved_rl_db)
        .function("get_error_message", &SolverWrapper::get_error_message)
        .function("get_cm_size", &SolverWrapper::get_cm_size)
        .function("get_cm_real", &SolverWrapper::get_cm_real)
        .function("get_cm_imag", &SolverWrapper::get_cm_imag)
        .function("evaluate_response", &SolverWrapper::evaluate_response)
        ;
}
