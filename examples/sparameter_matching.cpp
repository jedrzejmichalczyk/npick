#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coupling_matrix.hpp"
#include "impedance_matching.hpp"

using namespace np;

namespace {

struct Sample {
    double freq = 0.0;
    Complex gamma = Complex(0.0, 0.0);
};

double to_db(double magnitude) {
    if (!std::isfinite(magnitude) || magnitude < 1e-15) {
        return -300.0;
    }
    return 20.0 * std::log10(magnitude);
}

double to_db(Complex value) {
    return to_db(std::abs(value));
}

std::string trim(std::string text) {
    const auto is_space = [](unsigned char ch) {
        return std::isspace(ch) != 0;
    };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [&](char ch) {
        return !is_space(static_cast<unsigned char>(ch));
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [&](char ch) {
        return !is_space(static_cast<unsigned char>(ch));
    }).base(), text.end());
    return text;
}

bool parse_sample_line(const std::string& line, Sample& sample) {
    std::string normalized = line;
    for (char& ch : normalized) {
        if (ch == ';' || ch == '\t') {
            ch = ',';
        }
    }

    std::vector<std::string> tokens;
    std::stringstream comma_stream(normalized);
    std::string token;
    while (std::getline(comma_stream, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    if (tokens.size() < 3) {
        tokens.clear();
        std::stringstream whitespace_stream(line);
        while (whitespace_stream >> token) {
            tokens.push_back(token);
        }
    }
    if (tokens.size() < 3) {
        return false;
    }

    try {
        sample.freq = std::stod(tokens[0]);
        sample.gamma = Complex(std::stod(tokens[1]), std::stod(tokens[2]));
        return std::isfinite(sample.freq) &&
               std::isfinite(sample.gamma.real()) &&
               std::isfinite(sample.gamma.imag());
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<Sample> load_samples(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    std::vector<Sample> samples;
    std::string line;
    while (std::getline(input, line)) {
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#') {
            continue;
        }

        Sample sample;
        if (parse_sample_line(stripped, sample)) {
            samples.push_back(sample);
        }
    }

    if (samples.size() < 2) {
        throw std::runtime_error("expected at least two valid samples with columns freq,re,im");
    }

    std::sort(samples.begin(), samples.end(), [](const Sample& lhs, const Sample& rhs) {
        return lhs.freq < rhs.freq;
    });

    std::vector<Sample> deduplicated;
    deduplicated.reserve(samples.size());
    for (const Sample& sample : samples) {
        if (!deduplicated.empty() && std::abs(sample.freq - deduplicated.back().freq) < 1e-12) {
            deduplicated.back() = sample;
        } else {
            deduplicated.push_back(sample);
        }
    }
    return deduplicated;
}

Complex interpolate_gamma(const std::vector<Sample>& samples, double freq) {
    if (freq <= samples.front().freq) {
        return samples.front().gamma;
    }
    if (freq >= samples.back().freq) {
        return samples.back().gamma;
    }

    auto upper = std::lower_bound(samples.begin(), samples.end(), freq, [](const Sample& sample, double value) {
        return sample.freq < value;
    });
    if (upper == samples.begin()) {
        return upper->gamma;
    }

    const Sample& right = *upper;
    const Sample& left = *(upper - 1);
    const double width = right.freq - left.freq;
    if (!(width > 0.0)) {
        return right.gamma;
    }

    const double alpha = (freq - left.freq) / width;
    return (1.0 - alpha) * left.gamma + alpha * right.gamma;
}

double worst_g11(ImpedanceMatching& matcher, const MatrixXcd& cm, double left, double right, int samples) {
    double worst = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double freq = left + (right - left) * i / static_cast<double>(samples - 1);
        worst = std::max(worst, std::abs(matcher.eval_G11(cm, freq)));
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: sparameter_matching <input.csv> [order] [return_loss_db] [freq_left] [freq_right] [output.csv]\n";
        std::cerr << "Input format: freq,re,im with optional header lines\n";
        return 1;
    }

    try {
        const std::string input_path = argv[1];
        const int order = (argc > 2) ? std::stoi(argv[2]) : 8;
        const double return_loss = (argc > 3) ? std::stod(argv[3]) : 16.0;
        const std::vector<Sample> samples = load_samples(input_path);

        const double freq_left = (argc > 4) ? std::stod(argv[4]) : samples.front().freq;
        const double freq_right = (argc > 5) ? std::stod(argv[5]) : samples.back().freq;
        const std::string output_path = (argc > 6) ? argv[6] : "matched_response.csv";

        auto load = [&samples](double freq) {
            return interpolate_gamma(samples, freq);
        };

        ImpedanceMatching baseline(load, order, {}, return_loss, freq_left, freq_right);
        MatrixXcd cm_baseline = baseline.run_single(baseline.interpolation_freqs());
        if (cm_baseline.size() == 0) {
            std::cerr << "baseline synthesis failed\n";
            return 1;
        }

        ImpedanceMatching matcher(load, order, {}, return_loss, freq_left, freq_right);
        matcher.verbose = true;
        matcher.optimizer_max_iterations = 0;
        MatrixXcd cm = matcher.run();
        if (cm.size() == 0) {
            std::cerr << "matching synthesis failed\n";
            return 1;
        }

        const int output_samples = std::max(4 * static_cast<int>(samples.size()), 401);
        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("failed to open output file: " + output_path);
        }

        output << "freq,load_re,load_im,load_db,g11_db,g11_baseline_db,s11_db\n";
        for (int i = 0; i < output_samples; ++i) {
            const double freq = freq_left + (freq_right - freq_left) * i / static_cast<double>(output_samples - 1);
            const Complex load_gamma = load(freq);
            const Complex g11 = matcher.eval_G11(cm, freq);
            const Complex g11_baseline = baseline.eval_G11(cm_baseline, freq);
            const auto s_matrix = CouplingMatrix::eval_S(cm, Complex(0.0, freq));

            output << std::fixed << std::setprecision(9)
                   << freq << ","
                   << load_gamma.real() << ","
                   << load_gamma.imag() << ","
                   << to_db(load_gamma) << ","
                   << to_db(g11) << ","
                   << to_db(g11_baseline) << ","
                   << to_db(s_matrix(0, 0)) << "\n";
        }

        const double baseline_worst = worst_g11(baseline, cm_baseline, freq_left, freq_right, output_samples);
        const double optimized_worst = worst_g11(matcher, cm, freq_left, freq_right, output_samples);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "input_samples=" << samples.size() << "\n";
        std::cout << "order=" << order << "\n";
        std::cout << "return_loss_db=" << return_loss << "\n";
        std::cout << "freq_band=[" << freq_left << ", " << freq_right << "]\n";
        std::cout << "baseline_worst_db=" << to_db(baseline_worst) << "\n";
        std::cout << "optimized_worst_db=" << to_db(optimized_worst) << "\n";
        std::cout << "optimized_freqs=";
        for (double freq : matcher.interpolation_freqs()) {
            std::cout << " " << freq;
        }
        std::cout << "\n";
        std::cout << "output_csv=" << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
