#include "impedance_matching.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Eigen/Dense>

namespace np {
namespace {
constexpr double kPenaltyCost = 1e10;
constexpr double kMinMagnitude = 1e-15;
constexpr double kMinFreqGapFloor = 1e-9;

double safe_to_db(double mag) {
    if (!std::isfinite(mag) || mag < kMinMagnitude) {
        return 20.0 * std::log10(kMinMagnitude);
    }
    return 20.0 * std::log10(mag);
}
} // namespace

ImpedanceMatching::ImpedanceMatching(
    LoadFunction load,
    int order,
    const std::vector<Complex>& transmission_zeros,
    double return_loss_db,
    double freq_left,
    double freq_right
)
    : load_(load)
    , order_(order)
    , tzs_(transmission_zeros)
    , return_loss_(return_loss_db)
    , freq_left_(freq_left)
    , freq_right_(freq_right)
    , achieved_rl_db_(0.0)
{
    if (!load_) {
        throw std::invalid_argument("ImpedanceMatching: load function must be valid");
    }
    if (order_ <= 0) {
        throw std::invalid_argument("ImpedanceMatching: order must be > 0");
    }
    if (static_cast<int>(tzs_.size()) > order_) {
        throw std::invalid_argument("ImpedanceMatching: transmission_zeros size must be <= order");
    }
    if (return_loss_ <= 0.0 || !std::isfinite(return_loss_)) {
        throw std::invalid_argument("ImpedanceMatching: return_loss_db must be finite and > 0");
    }
    if (!std::isfinite(freq_left_) || !std::isfinite(freq_right_) || freq_left_ >= freq_right_) {
        throw std::invalid_argument("ImpedanceMatching: frequency bounds must be finite and satisfy left < right");
    }

    // Initialize with Chebyshev node frequencies
    freqs_ = chebyshev_nodes(freq_left_, freq_right_, order_);
}

std::vector<double> ImpedanceMatching::canonicalize_interpolation_freqs(
    const std::vector<double>& interp_freqs
) const {
    if (static_cast<int>(interp_freqs.size()) != order_) {
        return {};
    }

    std::vector<double> result = interp_freqs;
    for (double freq : result) {
        if (!std::isfinite(freq)) {
            return {};
        }
    }

    const double width = freq_right_ - freq_left_;
    if (!(width > 0.0)) {
        return {};
    }

    for (double& freq : result) {
        freq = std::clamp(freq, freq_left_, freq_right_);
    }
    std::sort(result.begin(), result.end());

    if (order_ <= 1) {
        return result;
    }

    const double max_gap = width / (2.0 * static_cast<double>(order_ - 1));
    const double min_gap = std::min(std::max(width * 1e-6, kMinFreqGapFloor), max_gap);

    for (int pass = 0; pass < 3; ++pass) {
        result.front() = std::max(result.front(), freq_left_);
        for (int i = 1; i < order_; ++i) {
            result[i] = std::max(result[i], result[i - 1] + min_gap);
        }

        if (result.back() <= freq_right_) {
            break;
        }

        const double overflow = result.back() - freq_right_;
        for (double& freq : result) {
            freq -= overflow;
        }

        result.back() = std::min(result.back(), freq_right_);
        for (int i = order_ - 2; i >= 0; --i) {
            result[i] = std::min(result[i], result[i + 1] - min_gap);
        }

        if (result.front() >= freq_left_) {
            break;
        }

        const double underflow = freq_left_ - result.front();
        for (double& freq : result) {
            freq += underflow;
        }
    }

    if (result.front() < freq_left_ - 1e-9 || result.back() > freq_right_ + 1e-9) {
        for (int i = 0; i < order_; ++i) {
            result[i] = freq_left_ + width * (i + 0.5) / static_cast<double>(order_);
        }
        return result;
    }

    result.front() = std::max(result.front(), freq_left_);
    for (int i = 1; i < order_; ++i) {
        result[i] = std::max(result[i], result[i - 1] + min_gap);
    }
    if (result.back() > freq_right_) {
        const double overflow = result.back() - freq_right_;
        for (double& freq : result) {
            freq -= overflow;
        }
    }

    return result;
}

bool ImpedanceMatching::supports_symmetric_optimization() const {
    if (!use_symmetric_parameterization || order_ <= 1) {
        return false;
    }
    if (std::abs(freq_left_ + freq_right_) > symmetry_detection_tolerance) {
        return false;
    }

    const int samples = 4;
    const double max_freq = std::max(std::abs(freq_left_), std::abs(freq_right_));
    for (int i = 0; i < samples; ++i) {
        const double freq = max_freq * (i + 1.0) / static_cast<double>(samples + 1);
        const Complex positive = load_(freq);
        const Complex mirrored = std::conj(load_(-freq));
        const double scale = std::max({1.0, std::abs(positive), std::abs(mirrored)});
        if (std::abs(positive - mirrored) > symmetry_detection_tolerance * scale) {
            return false;
        }
    }

    return true;
}

std::vector<double> ImpedanceMatching::compress_symmetric_freqs(
    const std::vector<double>& interp_freqs
) const {
    std::vector<double> canonical = canonicalize_interpolation_freqs(interp_freqs);
    if (static_cast<int>(canonical.size()) != order_) {
        return {};
    }

    const int half = order_ / 2;
    const int free_positive_count = std::max(half - ((fix_symmetric_band_edges && half > 0) ? 1 : 0), 0);
    std::vector<double> positive_freqs;
    positive_freqs.reserve(free_positive_count);
    for (int i = 0; i < free_positive_count; ++i) {
        positive_freqs.push_back(std::abs(canonical[order_ - half + i]));
    }
    return positive_freqs;
}

std::vector<double> ImpedanceMatching::expand_symmetric_freqs(
    const std::vector<double>& positive_freqs
) const {
    const int half = order_ / 2;
    const int free_positive_count = std::max(half - ((fix_symmetric_band_edges && half > 0) ? 1 : 0), 0);
    if (static_cast<int>(positive_freqs.size()) != free_positive_count) {
        return {};
    }

    std::vector<double> interior = positive_freqs;
    for (double freq : interior) {
        if (!std::isfinite(freq)) {
            return {};
        }
    }

    const double max_positive = std::max(std::abs(freq_left_), std::abs(freq_right_));
    const bool use_fixed_edges = fix_symmetric_band_edges && half > 0;
    const double min_gap = std::min(
        std::max((freq_right_ - freq_left_) * 1e-6, kMinFreqGapFloor),
        std::max(max_positive / std::max(order_, 1), kMinFreqGapFloor)
    );
    const double min_positive = (order_ % 2 == 0) ? (0.5 * min_gap) : min_gap;
    const double max_interior = use_fixed_edges ? std::max(max_positive - min_gap, min_positive) : max_positive;

    for (double& freq : interior) {
        freq = std::clamp(freq, min_positive, max_interior);
    }
    std::sort(interior.begin(), interior.end());

    for (int pass = 0; pass < 3 && free_positive_count > 0; ++pass) {
        interior.front() = std::max(interior.front(), min_positive);
        for (int i = 1; i < free_positive_count; ++i) {
            interior[i] = std::max(interior[i], interior[i - 1] + min_gap);
        }
        if (interior.back() <= max_interior) {
            break;
        }
        const double overflow = interior.back() - max_interior;
        for (double& freq : interior) {
            freq -= overflow;
        }
    }

    std::vector<double> positive = interior;
    if (use_fixed_edges) {
        positive.push_back(max_positive);
    }

    std::vector<double> full_freqs;
    full_freqs.reserve(order_);
    for (int i = half - 1; i >= 0; --i) {
        full_freqs.push_back(-positive[i]);
    }
    if (order_ % 2 == 1) {
        full_freqs.push_back(0.0);
    }
    for (double freq : positive) {
        full_freqs.push_back(freq);
    }

    return full_freqs;
}

std::vector<double> ImpedanceMatching::select_exchange_freqs(const MatrixXcd& cm, bool symmetric) const {
    const int eval_points = std::max(exchange_eval_points, std::max(4 * order_ + 1, 21));
    std::vector<double> freqs(eval_points);
    std::vector<double> mags(eval_points);

    for (int i = 0; i < eval_points; ++i) {
        const double freq = freq_left_ + (freq_right_ - freq_left_) * i / static_cast<double>(eval_points - 1);
        const double mag = std::abs(eval_G11(cm, freq));
        if (!std::isfinite(mag)) {
            return {};
        }
        freqs[i] = freq;
        mags[i] = mag;
    }

    struct PeakCandidate {
        double freq;
        double magnitude;
    };
    std::vector<PeakCandidate> candidates;
    candidates.reserve(eval_points);

    for (int i = 0; i < eval_points; ++i) {
        const bool is_endpoint = (i == 0 || i == eval_points - 1);
        const bool is_local_peak =
            !is_endpoint && mags[i] >= mags[i - 1] && mags[i] >= mags[i + 1];
        if (is_endpoint || is_local_peak) {
            candidates.push_back(PeakCandidate{freqs[i], mags[i]});
        }
    }
    if (candidates.empty()) {
        for (int i = 0; i < eval_points; ++i) {
            candidates.push_back(PeakCandidate{freqs[i], mags[i]});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const PeakCandidate& lhs, const PeakCandidate& rhs) {
        return lhs.magnitude > rhs.magnitude;
    });

    auto add_spaced_freq = [](std::vector<double>& selected, double freq, double min_spacing) {
        for (double chosen : selected) {
            if (std::abs(chosen - freq) < min_spacing) {
                return false;
            }
        }
        selected.push_back(freq);
        return true;
    };

    if (symmetric) {
        const int half = order_ / 2;
        const int free_positive_count = std::max(half - ((fix_symmetric_band_edges && half > 0) ? 1 : 0), 0);
        if (free_positive_count == 0) {
            return expand_symmetric_freqs({});
        }

        const double max_positive = std::max(std::abs(freq_left_), std::abs(freq_right_));
        const int spacing_denominator = free_positive_count + ((fix_symmetric_band_edges && half > 0) ? 2 : 1);
        const double nominal_spacing = std::max(
            max_positive / static_cast<double>(std::max(spacing_denominator, 1)),
            kMinFreqGapFloor
        );
        const double upper_limit =
            (fix_symmetric_band_edges && half > 0) ? std::max(max_positive - 0.5 * nominal_spacing, 0.0) : max_positive;

        std::vector<double> positive_freqs;
        positive_freqs.reserve(free_positive_count);
        for (int relax = 0; relax < 3 && static_cast<int>(positive_freqs.size()) < free_positive_count; ++relax) {
            const double min_spacing = nominal_spacing / std::pow(2.0, relax);
            for (const PeakCandidate& candidate : candidates) {
                if (candidate.freq <= 0.0 || candidate.freq >= upper_limit) {
                    continue;
                }
                add_spaced_freq(positive_freqs, candidate.freq, min_spacing);
                if (static_cast<int>(positive_freqs.size()) >= free_positive_count) {
                    break;
                }
            }
        }

        while (static_cast<int>(positive_freqs.size()) < free_positive_count) {
            const double fallback = max_positive *
                (positive_freqs.size() + 1.0) / static_cast<double>(spacing_denominator);
            if (!add_spaced_freq(positive_freqs, std::min(fallback, upper_limit), 0.5 * nominal_spacing)) {
                positive_freqs.push_back(std::min(fallback, upper_limit));
            }
        }

        std::sort(positive_freqs.begin(), positive_freqs.end());
        return expand_symmetric_freqs(positive_freqs);
    }

    const double nominal_spacing = std::max(
        (freq_right_ - freq_left_) / static_cast<double>(std::max(order_ + 1, 2)),
        kMinFreqGapFloor
    );
    std::vector<double> selected_freqs;
    selected_freqs.reserve(order_);
    for (int relax = 0; relax < 3 && static_cast<int>(selected_freqs.size()) < order_; ++relax) {
        const double min_spacing = nominal_spacing / std::pow(2.0, relax);
        for (const PeakCandidate& candidate : candidates) {
            add_spaced_freq(selected_freqs, candidate.freq, min_spacing);
            if (static_cast<int>(selected_freqs.size()) >= order_) {
                break;
            }
        }
    }
    while (static_cast<int>(selected_freqs.size()) < order_) {
        const double fallback =
            freq_left_ + (freq_right_ - freq_left_) *
            (selected_freqs.size() + 0.5) / static_cast<double>(order_);
        selected_freqs.push_back(fallback);
    }
    return canonicalize_interpolation_freqs(selected_freqs);
}

MatrixXcd ImpedanceMatching::run_single(const std::vector<double>& interp_freqs) {
    if (static_cast<int>(interp_freqs.size()) != order_) {
        if (verbose) {
            std::cerr << "run_single: expected " << order_
                      << " interpolation frequencies, got " << interp_freqs.size() << "\n";
        }
        return MatrixXcd();
    }

    for (double f : interp_freqs) {
        if (!std::isfinite(f)) {
            if (verbose) {
                std::cerr << "run_single: non-finite interpolation frequency\n";
            }
            return MatrixXcd();
        }
    }

    // Compute target loads at interpolation frequencies
    std::vector<Complex> target_loads(order_);
    for (int i = 0; i < order_; ++i) {
        target_loads[i] = load_(interp_freqs[i]);
    }

    try {
        // Create normalized Nevanlinna-Pick problem
        NevanlinnaPickNormalized np_problem(target_loads, interp_freqs, tzs_, return_loss_);

        // Create path tracker
        PathTracker tracker(&np_problem);
        tracker.h = path_tracker_h;
        tracker.run_tol = path_tracker_run_tol;
        tracker.final_tol = path_tracker_final_tol;
        tracker.verbose = false;

        // Get starting solution and track path
        VectorXcd x0 = np_problem.get_start_solution();
        VectorXcd solution = tracker.run(x0);

        // Check convergence
        VectorXcd residual = np_problem.calc_homotopy_function(solution, 0.0);
        if (residual.norm() > 1e-5) {
            return MatrixXcd();  // Failed
        }

        // Extract coupling matrix from solution
        return np_problem.calc_coupling_matrix(solution);

    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "NP solver failed: " << e.what() << "\n";
        }
        return MatrixXcd();  // Failed
    }
}

Complex ImpedanceMatching::eval_G11(const MatrixXcd& cm, double freq) const {
    if (cm.rows() < 2 || cm.cols() < 2 || !std::isfinite(freq)) {
        return Complex(kPenaltyCost);
    }

    Complex s = Complex(0, freq);
    auto S = CouplingMatrix::eval_S(cm, s);

    Complex S11 = S(0, 0);
    Complex S12 = S(0, 1);
    Complex S21 = S(1, 0);
    Complex S22 = S(1, 1);
    Complex L = load_(freq);

    // G11 = S11 + S12*S21*L / (1 - S22*L)
    Complex denom = Complex(1) - S22 * L;
    if (std::abs(denom) < 1e-15) {
        return Complex(kPenaltyCost);  // Avoid division by zero
    }
    return S11 + S12 * S21 * L / denom;
}

double ImpedanceMatching::refine_peak_magnitude(const MatrixXcd& cm, double left, double right) const {
    if (!std::isfinite(left) || !std::isfinite(right) || right < left) {
        return kPenaltyCost;
    }

    auto magnitude_at = [this, &cm](double freq) {
        double magnitude = std::abs(eval_G11(cm, freq));
        return std::isfinite(magnitude) ? magnitude : kPenaltyCost;
    };

    double best_mag = std::max(magnitude_at(left), magnitude_at(right));
    double interval_left = left;
    double interval_right = right;

    const int subdivisions = std::max(cost_peak_refine_subdivisions, 2);
    const int iterations = std::max(cost_peak_refine_iterations, 1);

    for (int iter = 0; iter < iterations; ++iter) {
        const double width = interval_right - interval_left;
        if (!(width > 1e-12)) {
            break;
        }

        const double step = width / static_cast<double>(subdivisions);
        double local_best_freq = interval_left;
        double local_best_mag = -1.0;

        for (int k = 0; k <= subdivisions; ++k) {
            const double freq = interval_left + step * k;
            const double magnitude = magnitude_at(freq);
            if (!std::isfinite(magnitude)) {
                return kPenaltyCost;
            }
            if (magnitude <= local_best_mag) {
                continue;
            }
            local_best_mag = magnitude;
            local_best_freq = freq;
        }

        best_mag = std::max(best_mag, local_best_mag);
        interval_left = std::max(left, local_best_freq - step);
        interval_right = std::min(right, local_best_freq + step);
    }

    return best_mag;
}

double ImpedanceMatching::compute_response_cost(const MatrixXcd& cm) const {
    const int eval_points = std::max(cost_eval_points, 2);
    std::vector<double> freqs(eval_points);
    std::vector<double> mags(eval_points);

    double max_g11 = 0.0;
    for (int i = 0; i < eval_points; ++i) {
        const double freq = freq_left_ + (freq_right_ - freq_left_) * i / static_cast<double>(eval_points - 1);
        const double mag = std::abs(eval_G11(cm, freq));
        if (!std::isfinite(mag)) {
            return kPenaltyCost;
        }
        freqs[i] = freq;
        mags[i] = mag;
        max_g11 = std::max(max_g11, mag);
    }

    std::vector<int> order(eval_points);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&mags](int lhs, int rhs) {
        return mags[lhs] > mags[rhs];
    });

    std::vector<bool> used(eval_points, false);
    std::vector<int> candidates;
    auto add_candidate = [&](int index) {
        if (index < 0 || index >= eval_points || used[index]) {
            return;
        }
        used[index] = true;
        candidates.push_back(index);
    };

    const int candidate_limit =
        (cost_peak_candidates > 0) ? cost_peak_candidates : std::max(2 * order_ + 1, 5);
    for (int index : order) {
        if (static_cast<int>(candidates.size()) >= candidate_limit) {
            break;
        }
        const bool is_endpoint = (index == 0 || index == eval_points - 1);
        const bool is_local_peak =
            !is_endpoint && mags[index] >= mags[index - 1] && mags[index] >= mags[index + 1];
        if (is_endpoint || is_local_peak) {
            add_candidate(index);
        }
    }
    for (int index : order) {
        if (static_cast<int>(candidates.size()) >= candidate_limit) {
            break;
        }
        add_candidate(index);
    }

    for (int index : candidates) {
        const int left_index = std::max(0, index - 1);
        const int right_index = std::min(eval_points - 1, index + 1);
        if (left_index == right_index) {
            max_g11 = std::max(max_g11, mags[index]);
            continue;
        }

        const double refined_mag = refine_peak_magnitude(cm, freqs[left_index], freqs[right_index]);
        if (!std::isfinite(refined_mag)) {
            return kPenaltyCost;
        }
        max_g11 = std::max(max_g11, refined_mag);
    }

    return max_g11;
}

double ImpedanceMatching::compute_cost(const std::vector<double>& interp_freqs) {
    std::vector<double> canonical_freqs = canonicalize_interpolation_freqs(interp_freqs);
    if (static_cast<int>(canonical_freqs.size()) != order_) {
        return kPenaltyCost;
    }

    // Run NP solver
    MatrixXcd cm = run_single(canonical_freqs);
    if (cm.size() == 0) {
        return kPenaltyCost;
    }

    return compute_response_cost(cm);
}

std::vector<double> ImpedanceMatching::find_interval_peaks(
    const MatrixXcd& cm, const std::vector<double>& interp_freqs
) const {
    // Returns n+1 peak heights of |G11| in each interval:
    // [freq_left, x1], [x1,x2], ..., [x_{n-1},xn], [xn, freq_right]
    const int n = static_cast<int>(interp_freqs.size());
    std::vector<double> peaks(n + 1, 0.0);

    std::vector<double> boundaries;
    boundaries.push_back(freq_left_);
    for (int i = 0; i < n; ++i)
        boundaries.push_back(interp_freqs[i]);
    boundaries.push_back(freq_right_);

    const int pts = equiripple_peak_eval_points;
    const int refine = equiripple_peak_refine_steps;

    for (int seg = 0; seg <= n; ++seg) {
        double left = boundaries[seg];
        double right = boundaries[seg + 1];
        if (right - left < 1e-12) {
            peaks[seg] = std::abs(eval_G11(cm, left));
            continue;
        }

        // Coarse grid search
        double best_mag = 0, best_freq = left;
        const int seg_pts = std::max(pts / (n + 1), 8);
        for (int i = 0; i <= seg_pts; ++i) {
            double f = left + (right - left) * i / static_cast<double>(seg_pts);
            double mag = std::abs(eval_G11(cm, f));
            if (mag > best_mag) { best_mag = mag; best_freq = f; }
        }

        // Refine with golden section
        double a = std::max(left, best_freq - (right - left) / seg_pts);
        double b = std::min(right, best_freq + (right - left) / seg_pts);
        for (int r = 0; r < refine; ++r) {
            double step = (b - a) / 6.0;
            double local_best = best_mag;
            double local_freq = best_freq;
            for (int k = 0; k <= 6; ++k) {
                double f = a + step * k;
                double mag = std::abs(eval_G11(cm, f));
                if (mag > local_best) { local_best = mag; local_freq = f; }
            }
            best_mag = local_best;
            best_freq = local_freq;
            a = std::max(left, best_freq - step);
            b = std::min(right, best_freq + step);
        }

        peaks[seg] = best_mag;
    }
    return peaks;
}

MatrixXcd ImpedanceMatching::run_equiripple(const std::vector<double>& initial_freqs) {
    if (static_cast<int>(initial_freqs.size()) != order_) {
        return MatrixXcd();
    }

    std::vector<double> freqs = canonicalize_interpolation_freqs(initial_freqs);
    if (static_cast<int>(freqs.size()) != order_) {
        return MatrixXcd();
    }

    const int n = order_;
    const double band_width = freq_right_ - freq_left_;
    const double fd_delta = equiripple_fd_delta * band_width;

    // Solve NP at initial frequencies
    MatrixXcd cm = run_single(freqs);
    if (cm.size() == 0) return cm;

    std::vector<double> peaks = find_interval_peaks(cm, freqs);
    double max_peak = *std::max_element(peaks.begin(), peaks.end());
    double min_peak = *std::min_element(peaks.begin(), peaks.end());

    if (verbose) {
        std::cout << "  Equiripple iter 0: max|G11|=" << safe_to_db(max_peak)
                  << " dB, ripple spread=" << safe_to_db(max_peak) - safe_to_db(min_peak) << " dB\n";
    }

    MatrixXcd best_cm = cm;
    std::vector<double> best_freqs = freqs;
    double best_max_peak = max_peak;

    for (int iter = 0; iter < equiripple_max_iterations; ++iter) {
        // Check convergence: all peaks approximately equal
        if (max_peak - min_peak < optimizer_tolerance * max_peak) {
            if (verbose) std::cout << "  Equiripple converged (equioscillation reached)\n";
            break;
        }

        // Compute equiripple residual F(x) = [h0-h1, h1-h2, ..., h(n-1)-hn]
        Eigen::VectorXd F(n);
        for (int i = 0; i < n; ++i)
            F(i) = peaks[i] - peaks[i + 1];

        // Compute Jacobian via finite differences
        Eigen::MatrixXd J(n, n);
        for (int j = 0; j < n; ++j) {
            std::vector<double> freqs_pert = freqs;
            freqs_pert[j] += fd_delta;
            freqs_pert = canonicalize_interpolation_freqs(freqs_pert);

            MatrixXcd cm_pert = run_single(freqs_pert);
            if (cm_pert.size() == 0) {
                // Try negative perturbation
                freqs_pert = freqs;
                freqs_pert[j] -= fd_delta;
                freqs_pert = canonicalize_interpolation_freqs(freqs_pert);
                cm_pert = run_single(freqs_pert);
                if (cm_pert.size() == 0) {
                    if (verbose) std::cout << "  Equiripple: NP failed for perturbation j=" << j << "\n";
                    goto done;
                }
            }

            std::vector<double> peaks_pert = find_interval_peaks(cm_pert, freqs_pert);
            double actual_delta = freqs_pert[j] - freqs[j];
            if (std::abs(actual_delta) < 1e-15) actual_delta = fd_delta;

            for (int i = 0; i < n; ++i) {
                double F_pert_i = peaks_pert[i] - peaks_pert[i + 1];
                J(i, j) = (F_pert_i - F(i)) / actual_delta;
            }
        }

        // Solve Newton system: J * dx = -F
        Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-F);

        // Clamp step to prevent wild jumps
        double max_step = 0.25 * band_width;
        double step_scale = 1.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(dx(i)) > max_step)
                step_scale = std::min(step_scale, max_step / std::abs(dx(i)));
        }
        dx *= step_scale;

        // Backtracking line search on max(peak)
        double alpha = 1.0;
        bool step_accepted = false;
        for (int ls = 0; ls < 8; ++ls) {
            std::vector<double> freqs_new(n);
            for (int i = 0; i < n; ++i)
                freqs_new[i] = freqs[i] + alpha * dx(i);
            freqs_new = canonicalize_interpolation_freqs(freqs_new);
            if (static_cast<int>(freqs_new.size()) != n) {
                alpha *= 0.5;
                continue;
            }

            MatrixXcd cm_new = run_single(freqs_new);
            if (cm_new.size() == 0) {
                alpha *= 0.5;
                continue;
            }

            std::vector<double> peaks_new = find_interval_peaks(cm_new, freqs_new);
            double new_max = *std::max_element(peaks_new.begin(), peaks_new.end());

            if (new_max < max_peak || ls == 7) {
                freqs = freqs_new;
                cm = cm_new;
                peaks = peaks_new;
                max_peak = new_max;
                min_peak = *std::min_element(peaks.begin(), peaks.end());
                step_accepted = true;

                if (new_max < best_max_peak) {
                    best_max_peak = new_max;
                    best_freqs = freqs;
                    best_cm = cm;
                }
                break;
            }
            alpha *= 0.5;
        }

        if (verbose) {
            std::cout << "  Equiripple iter " << (iter + 1)
                      << ": max|G11|=" << safe_to_db(max_peak)
                      << " dB, spread=" << std::fixed << std::setprecision(1)
                      << (safe_to_db(max_peak) - safe_to_db(min_peak))
                      << " dB, alpha=" << std::setprecision(3) << alpha << "\n";
        }

        if (!step_accepted) {
            if (verbose) std::cout << "  Equiripple: line search failed\n";
            break;
        }
    }

done:
    freqs_ = best_freqs;
    cm_ = best_cm;
    achieved_rl_db_ = safe_to_db(best_max_peak);
    return best_cm;
}

MatrixXcd ImpedanceMatching::run() {
    if (verbose) {
        std::cout << "Starting impedance matching optimization...\n";
        std::cout << "  Order: " << order_ << "\n";
        std::cout << "  Return loss: " << return_loss_ << " dB\n";
        std::cout << "  Passband: [" << freq_left_ << ", " << freq_right_ << "]\n";
    }

    freqs_ = canonicalize_interpolation_freqs(freqs_);
    if (static_cast<int>(freqs_.size()) != order_) {
        cm_ = MatrixXcd();
        achieved_rl_db_ = safe_to_db(kPenaltyCost);
        return cm_;
    }

    // Primary optimizer: equiripple Newton
    if (equiripple_max_iterations > 0) {
        MatrixXcd cm_eq = run_equiripple(freqs_);
        if (cm_eq.size() != 0) {
            if (verbose) {
                std::cout << "  Equiripple result: " << achieved_rl_db_ << " dB\n";
            }
            return cm_eq;
        }
        if (verbose) {
            std::cout << "  Equiripple failed, falling back to NM\n";
        }
    }

    // Fallback: Nelder-Mead
    const bool use_symmetric_parameterization = supports_symmetric_optimization();

    cm_ = run_single(freqs_);
    if (cm_.size() == 0) {
        achieved_rl_db_ = safe_to_db(kPenaltyCost);
        return cm_;
    }

    double current_cost = compute_response_cost(cm_);
    double best_cost = current_cost;
    std::vector<double> best_freqs = freqs_;
    MatrixXcd best_cm = cm_;

    if (verbose) {
        double initial_db = safe_to_db(current_cost);
        std::cout << "  Initial max|G11|: " << initial_db << " dB\n";
    }

    auto blend_freqs = [this, use_symmetric_parameterization](
        const std::vector<double>& current_freqs,
        const std::vector<double>& target_freqs,
        double alpha
    ) {
        if (alpha <= 0.0) {
            return current_freqs;
        }
        if (alpha >= 1.0) {
            return target_freqs;
        }
        if (use_symmetric_parameterization) {
            const std::vector<double> current_positive = compress_symmetric_freqs(current_freqs);
            const std::vector<double> target_positive = compress_symmetric_freqs(target_freqs);
            if (current_positive.size() != target_positive.size()) {
                return std::vector<double>{};
            }

            std::vector<double> blended_positive(current_positive.size());
            for (size_t i = 0; i < current_positive.size(); ++i) {
                blended_positive[i] =
                    (1.0 - alpha) * current_positive[i] + alpha * target_positive[i];
            }
            return expand_symmetric_freqs(blended_positive);
        }

        std::vector<double> blended(order_);
        for (int i = 0; i < order_; ++i) {
            blended[i] = (1.0 - alpha) * current_freqs[i] + alpha * target_freqs[i];
        }
        return canonicalize_interpolation_freqs(blended);
    };

    if (exchange_iterations > 0) {
        for (int iter = 0; iter < exchange_iterations; ++iter) {
            std::vector<double> exchanged_freqs = select_exchange_freqs(cm_, use_symmetric_parameterization);
            if (static_cast<int>(exchanged_freqs.size()) != order_) {
                break;
            }

            const double base_alpha = std::clamp(exchange_relaxation, 0.0, 1.0);
            std::vector<double> accepted_freqs = freqs_;
            MatrixXcd accepted_cm = cm_;
            double accepted_cost = current_cost;
            bool improved = false;

            for (int trial = 0; trial < 12; ++trial) {
                const double alpha = base_alpha * std::pow(0.5, trial);
                if (alpha < 1e-4) {
                    break;
                }
                const std::vector<double> candidate_freqs = blend_freqs(freqs_, exchanged_freqs, alpha);
                if (static_cast<int>(candidate_freqs.size()) != order_) {
                    continue;
                }

                MatrixXcd candidate_cm = run_single(candidate_freqs);
                if (candidate_cm.size() == 0) {
                    continue;
                }

                const double candidate_cost = compute_response_cost(candidate_cm);
                if (verbose) {
                    std::cout << "  Exchange trial " << trial << " (alpha=" << alpha
                              << "): " << safe_to_db(candidate_cost) << " dB\n";
                }
                if (!std::isfinite(candidate_cost)) {
                    continue;
                }
                if (candidate_cost + optimizer_tolerance < accepted_cost) {
                    accepted_freqs = candidate_freqs;
                    accepted_cm = candidate_cm;
                    accepted_cost = candidate_cost;
                    improved = true;
                    break;
                }
            }

            if (!improved) {
                if (verbose) {
                    std::cout << "  Exchange stalled after " << iter << " iterations\n";
                }
                break;
            }

            double max_delta = 0.0;
            for (int i = 0; i < order_; ++i) {
                max_delta = std::max(max_delta, std::abs(accepted_freqs[i] - freqs_[i]));
            }
            freqs_ = accepted_freqs;
            cm_ = accepted_cm;
            current_cost = accepted_cost;

            if (current_cost < best_cost) {
                best_cost = current_cost;
                best_freqs = freqs_;
                best_cm = cm_;
            }

            if (verbose) {
                std::cout << "  Exchange iter " << (iter + 1)
                          << ": max|G11| = " << safe_to_db(current_cost) << " dB\n";
            }

            if (max_delta < 1e-4) {
                break;
            }
        }
    }

    int polish_iterations = optimizer_max_iterations;
    if (optimizer_iteration_cap > 0) {
        polish_iterations = std::min(polish_iterations, optimizer_iteration_cap);
    }

    if (polish_iterations > 0) {
        NelderMead optimizer(optimizer_tolerance, polish_iterations);
        NelderMead::Result result{{}, current_cost, 0, false};

        if (use_symmetric_parameterization) {
            std::vector<double> initial_positive = compress_symmetric_freqs(freqs_);
            auto cost_fn = [this](const std::vector<double>& positive_freqs) {
                return this->compute_cost(this->expand_symmetric_freqs(positive_freqs));
            };
            result = optimizer.minimize(cost_fn, initial_positive, optimizer_initial_step);
        } else {
            auto cost_fn = [this](const std::vector<double>& freqs) {
                return this->compute_cost(freqs);
            };
            result = optimizer.minimize(cost_fn, freqs_, optimizer_initial_step);
        }

        std::vector<double> polished_freqs;
        if (use_symmetric_parameterization) {
            polished_freqs = expand_symmetric_freqs(result.minimizing_point);
        } else if (static_cast<int>(result.minimizing_point.size()) == order_) {
            polished_freqs = canonicalize_interpolation_freqs(result.minimizing_point);
        }

        if (static_cast<int>(polished_freqs.size()) == order_) {
            MatrixXcd polished_cm = run_single(polished_freqs);
            if (polished_cm.size() != 0) {
                const double polished_cost = compute_response_cost(polished_cm);
                if (polished_cost < best_cost) {
                    best_cost = polished_cost;
                    best_freqs = polished_freqs;
                    best_cm = polished_cm;
                }
            }
        }

        if (verbose) {
            std::cout << "  Local polish " << (result.converged ? "converged" : "stopped")
                      << " after " << result.iterations << " iterations\n";
        }
    } else if (verbose) {
        std::cout << "  Local polish disabled\n";
    }

    freqs_ = best_freqs;
    cm_ = best_cm;
    achieved_rl_db_ = safe_to_db(best_cost);

    if (verbose) {
        std::cout << "  Final max|G11|: " << achieved_rl_db_ << " dB\n";
    }

    return cm_;
}

} // namespace np
