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
#include <memory>
#include <map>

#include "types.hpp"
#include "impedance_matching.hpp"
#include "coupling_matrix.hpp"
#include "multiplexer/multiplexer_matching.hpp"
#include "multiplexer/manifold.hpp"
#include "multiplexer/t_junction.hpp"

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

// ============================================================================
// MultiplexerWrapper — Martinez-style manifold-coupled multiplexer synthesis.
//
// JS calls:
//   mux = new MultiplexerWrapper();
//   mux.add_channel(order, freq_left_ghz, freq_right_ghz, rl_db, tz_re, tz_im);
//   mux.set_center_frequency(f0_ghz);
//   mux.set_junction_alpha(j, alpha);                      // optional (default 0.5)
//   mux.set_junction_s(j, 3, [re_row_major], [im_row_major]); // optional 3x3 custom S
//   mux.set_equiripple_iters(15);
//   if (mux.solve()) { ... mux.get_cm_*(ch,i,j) ... mux.evaluate_response(...) }
//
// ============================================================================
class MultiplexerWrapper {
public:
    MultiplexerWrapper() = default;

    void reset_channels() {
        specs_.clear();
        alpha_overrides_.clear();
        s_overrides_.clear();
    }

    void add_channel(int order, double freq_left, double freq_right,
                     double rl_db, val js_tz_re, val js_tz_im) {
        ChannelSpec s;
        s.order = order;
        s.freq_left = freq_left;
        s.freq_right = freq_right;
        s.return_loss_db = rl_db;
        auto tz_re = vecFromJS(js_tz_re);
        auto tz_im = vecFromJS(js_tz_im);
        for (size_t i = 0; i < tz_re.size(); ++i)
            s.transmission_zeros.push_back(Complex(tz_re[i], tz_im[i]));
        specs_.push_back(s);
    }

    void set_center_frequency(double f0) { center_freq_ = f0; }

    void set_junction_alpha(int j, double alpha) {
        alpha_overrides_[j] = alpha;
    }

    // Override a junction's full 3x3 S-matrix (row-major, flat).
    // If set, takes precedence over set_junction_alpha for that junction.
    void set_junction_s(int j, val js_re, val js_im) {
        auto re = vecFromJS(js_re);
        auto im = vecFromJS(js_im);
        if (re.size() != 9 || im.size() != 9) return;
        Eigen::Matrix3cd S;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                int idx = r * 3 + c;
                S(r, c) = Complex(re[idx], im[idx]);
            }
        s_overrides_[j] = S;
    }

    void set_equiripple_iters(int n) { equiripple_iters_ = n; }

    bool solve() {
        success_ = false;
        error_message_ = "";
        cms_.clear();
        achieved_rls_.clear();

        if (specs_.empty()) {
            error_message_ = "No channels configured";
            return false;
        }
        if (specs_.size() < 2) {
            error_message_ = "Multiplexer needs at least 2 channels";
            return false;
        }

        try {
            mux_ = std::make_unique<MultiplexerMatching>(specs_, center_freq_);
            mux_->verbose = false;
            mux_->equiripple_outer_iterations = equiripple_iters_;

            // Apply junction overrides before run() (which designs the manifold).
            int N = static_cast<int>(specs_.size());
            for (int j = 0; j < N - 1; ++j) {
                auto s_it = s_overrides_.find(j);
                if (s_it != s_overrides_.end()) {
                    mux_->manifold().set_junction(j, TJunction(s_it->second));
                    continue;
                }
                auto a_it = alpha_overrides_.find(j);
                if (a_it != alpha_overrides_.end()) {
                    mux_->manifold().set_junction(j, TJunction(a_it->second));
                }
            }

            cms_ = mux_->run();
            achieved_rls_ = mux_->achieved_return_losses_db();
            success_ = true;
            return true;
        } catch (const std::exception& e) {
            error_message_ = e.what();
            return false;
        } catch (...) {
            error_message_ = "Unknown multiplexer solver error";
            return false;
        }
    }

    bool get_success() const { return success_; }
    std::string get_error_message() const { return error_message_; }
    int get_num_channels() const { return static_cast<int>(cms_.size()); }

    double get_achieved_rl_db(int ch) const {
        if (ch < 0 || ch >= static_cast<int>(achieved_rls_.size())) return 0;
        return achieved_rls_[ch];
    }

    int get_cm_size(int ch) const {
        if (ch < 0 || ch >= static_cast<int>(cms_.size())) return 0;
        return cms_[ch].rows();
    }

    double get_cm_real(int ch, int i, int j) const {
        if (ch < 0 || ch >= static_cast<int>(cms_.size())) return 0;
        const auto& M = cms_[ch];
        if (i < 0 || i >= M.rows() || j < 0 || j >= M.cols()) return 0;
        return M(i, j).real();
    }

    double get_cm_imag(int ch, int i, int j) const {
        if (ch < 0 || ch >= static_cast<int>(cms_.size())) return 0;
        const auto& M = cms_[ch];
        if (i < 0 || i >= M.rows() || j < 0 || j >= M.cols()) return 0;
        return M(i, j).imag();
    }

    // Sweep over [freq_start, freq_stop] and return per-channel G11/S21 in dB
    // plus the overall frequency axis. Channels are indexed in the order
    // they were added.
    val evaluate_response(double freq_start, double freq_stop, int num_points) {
        val result = val::object();
        val js_freq = val::array();

        int N = static_cast<int>(cms_.size());
        std::vector<val> js_g11(N, val::array());
        std::vector<val> js_s21(N, val::array());

        if (!mux_) {
            result.set("freq", js_freq);
            return result;
        }

        for (int k = 0; k < num_points; ++k) {
            double freq = freq_start + (freq_stop - freq_start) * k /
                          static_cast<double>(num_points - 1);
            js_freq.call<void>("push", freq);

            for (int i = 0; i < N; ++i) {
                // S21 from the extracted CM in the channel's own normalized band.
                double fc = (specs_[i].freq_left + specs_[i].freq_right) / 2.0;
                double fs = (specs_[i].freq_right - specs_[i].freq_left) / 2.0;
                double norm = (freq - fc) / fs;
                Complex s(0, norm);
                double s21_db = -300.0;
                if (cms_[i].size() > 0) {
                    auto S = CouplingMatrix::eval_S(cms_[i], s);
                    s21_db = to_db(std::abs(S(1, 0)));
                }
                js_s21[i].call<void>("push", s21_db);

                Complex g = mux_->eval_channel_response(i, freq);
                js_g11[i].call<void>("push", to_db(std::abs(g)));
            }
        }

        result.set("freq", js_freq);
        val g11_arr = val::array();
        val s21_arr = val::array();
        for (int i = 0; i < N; ++i) {
            g11_arr.call<void>("push", js_g11[i]);
            s21_arr.call<void>("push", js_s21[i]);
        }
        result.set("g11_db", g11_arr);
        result.set("s21_db", s21_arr);
        return result;
    }

private:
    std::vector<ChannelSpec> specs_;
    double center_freq_ = 1.0;
    std::map<int, double> alpha_overrides_;
    std::map<int, Eigen::Matrix3cd, std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix3cd>>>
        s_overrides_;
    int equiripple_iters_ = 10;

    std::unique_ptr<MultiplexerMatching> mux_;
    std::vector<MatrixXcd> cms_;
    std::vector<double> achieved_rls_;
    bool success_ = false;
    std::string error_message_;
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

    class_<MultiplexerWrapper>("MultiplexerWrapper")
        .constructor<>()
        .function("reset_channels", &MultiplexerWrapper::reset_channels)
        .function("add_channel", &MultiplexerWrapper::add_channel)
        .function("set_center_frequency", &MultiplexerWrapper::set_center_frequency)
        .function("set_junction_alpha", &MultiplexerWrapper::set_junction_alpha)
        .function("set_junction_s", &MultiplexerWrapper::set_junction_s)
        .function("set_equiripple_iters", &MultiplexerWrapper::set_equiripple_iters)
        .function("solve", &MultiplexerWrapper::solve)
        .function("get_success", &MultiplexerWrapper::get_success)
        .function("get_error_message", &MultiplexerWrapper::get_error_message)
        .function("get_num_channels", &MultiplexerWrapper::get_num_channels)
        .function("get_achieved_rl_db", &MultiplexerWrapper::get_achieved_rl_db)
        .function("get_cm_size", &MultiplexerWrapper::get_cm_size)
        .function("get_cm_real", &MultiplexerWrapper::get_cm_real)
        .function("get_cm_imag", &MultiplexerWrapper::get_cm_imag)
        .function("evaluate_response", &MultiplexerWrapper::evaluate_response)
        ;
}
