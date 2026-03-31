#pragma once

#include "../types.hpp"
#include "homotopy_base.hpp"
#include "step_controller.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>

namespace np {

/**
 * Dormand-Prince 8(5,3) adaptive ODE predictor for homotopy path tracking.
 *
 * An 8th order explicit Runge-Kutta method with embedded 5th and 3rd order
 * error estimators for adaptive step size control.
 */
class Dopr853Predictor {
public:
    explicit Dopr853Predictor(HomotopyBase* homotopy)
        : homotopy_(homotopy)
        , n_(homotopy->get_num_variables())
        , atol(1e-3)
        , rtol(1e-3)
    {}

    /**
     * Perform a predictor step from (x, t) with step size h.
     *
     * @param x Current solution
     * @param t Current homotopy parameter
     * @param h_try Attempted step size
     * @param delta_x Output: change in x for this step
     */
    void predict(const VectorXcd& x, double t, double h_try, VectorXcd& delta_x) {
        double h = h_try;
        VectorXcd dxdt(n_), dxdt_new(n_);

        calc_derivative(t, x, dxdt);

        int max_retries = 100;
        for (int retry = 0; retry < max_retries; ++retry) {
            try {
                // Take a step
                VectorXcd x_out(n_), x_err(n_), x_err2(n_);
                dx_step(h, t, x, dxdt, x_out, x_err, x_err2);

                // Check step size underflow
                if (std::abs(h) <= std::abs(t) * std::numeric_limits<double>::epsilon()) {
                    throw std::runtime_error("Step size underflow in Dopr853");
                }

                // Compute error and check acceptance
                double error = calc_error(t, x, x_out, x_err, x_err2);
                if (controller_.success(error, h)) {
                    delta_x = x_out - x;
                    h_next_ = controller_.h_next();
                    return;
                }

                // Step rejected, reduce h
                h *= 0.7;
            } catch (const std::exception& e) {
                // On error, reduce step size
                h *= 0.5;
                if (std::abs(h) <= std::abs(t) * std::numeric_limits<double>::epsilon()) {
                    throw std::runtime_error("Step size underflow after exception");
                }
            }
        }

        throw std::runtime_error("Dopr853 predictor failed to converge");
    }

    double h_next() const { return h_next_; }

    double atol;  // Absolute tolerance
    double rtol;  // Relative tolerance

private:
    void calc_derivative(double t, const VectorXcd& x, VectorXcd& dxdt) {
        MatrixXcd J = homotopy_->calc_jacobian(x, t);
        VectorXcd Ht = -homotopy_->calc_path_derivative(x, t);

        // Solve J * dxdt = Ht
        dxdt = J.colPivHouseholderQr().solve(Ht);
    }

    void dx_step(double h, double t, const VectorXcd& x, const VectorXcd& dxdt,
                VectorXcd& x_out, VectorXcd& x_err, VectorXcd& x_err2) {
        VectorXcd k2(n_), k3(n_), k4(n_), k5(n_), k6(n_), k7(n_);
        VectorXcd k8(n_), k9(n_), k10(n_), k2_temp(n_), k3_temp(n_);
        VectorXcd x_temp(n_);

        // Stage 2
        x_temp = x + h * a21 * dxdt;
        calc_derivative(t + c2 * h, x_temp, k2);

        // Stage 3
        x_temp = x + h * (a31 * dxdt + a32 * k2);
        calc_derivative(t + c3 * h, x_temp, k3);

        // Stage 4
        x_temp = x + h * (a41 * dxdt + a43 * k3);
        calc_derivative(t + c4 * h, x_temp, k4);

        // Stage 5
        x_temp = x + h * (a51 * dxdt + a53 * k3 + a54 * k4);
        calc_derivative(t + c5 * h, x_temp, k5);

        // Stage 6
        x_temp = x + h * (a61 * dxdt + a64 * k4 + a65 * k5);
        calc_derivative(t + c6 * h, x_temp, k6);

        // Stage 7
        x_temp = x + h * (a71 * dxdt + a74 * k4 + a75 * k5 + a76 * k6);
        calc_derivative(t + c7 * h, x_temp, k7);

        // Stage 8
        x_temp = x + h * (a81 * dxdt + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7);
        calc_derivative(t + c8 * h, x_temp, k8);

        // Stage 9
        x_temp = x + h * (a91 * dxdt + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8);
        calc_derivative(t + c9 * h, x_temp, k9);

        // Stage 10
        x_temp = x + h * (a101 * dxdt + a104 * k4 + a105 * k5 + a106 * k6 +
                        a107 * k7 + a108 * k8 + a109 * k9);
        calc_derivative(t + c10 * h, x_temp, k10);

        // Stage 11 (stored in k2_temp)
        x_temp = x + h * (a111 * dxdt + a114 * k4 + a115 * k5 + a116 * k6 +
                        a117 * k7 + a118 * k8 + a119 * k9 + a1110 * k10);
        calc_derivative(t + c11 * h, x_temp, k2_temp);

        // Stage 12 (stored in k3_temp)
        double tph = t + h;
        x_temp = x + h * (a121 * dxdt + a124 * k4 + a125 * k5 + a126 * k6 +
                        a127 * k7 + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k2_temp);
        calc_derivative(tph, x_temp, k3_temp);

        // Compute output and error estimates
        VectorXcd k4_final = b1 * dxdt + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9 +
                            b10 * k10 + b11 * k2_temp + b12 * k3_temp;
        x_out = x + h * k4_final;

        x_err = k4_final - bhh1 * dxdt - bhh2 * k9 - bhh3 * k3_temp;
        x_err2 = er1 * dxdt + er6 * k6 + er7 * k7 + er8 * k8 + er9 * k9 +
                er10 * k10 + er11 * k2_temp + er12 * k3_temp;
    }

    double calc_error(double t, const VectorXcd& x, const VectorXcd& x_out,
                     const VectorXcd& x_err, const VectorXcd& x_err2) {
        double err = 0.0, err2 = 0.0;

        for (int i = 0; i < n_; ++i) {
            double sk = atol + rtol * std::max(std::abs(x(i)), std::abs(x_out(i)));
            err += std::pow(std::abs(x_err(i)) / sk, 2);
            err2 += std::pow(std::abs(x_err2(i)) / sk, 2);
        }

        double deno = err + 0.01 * err2;
        if (deno <= 0.0) deno = 1.0;

        return std::abs(1 - t) * err * std::sqrt(1.0 / (n_ * deno));
    }

    HomotopyBase* homotopy_;
    int n_;
    StepController controller_;
    double h_next_ = 0;

    // Dormand-Prince 8(5,3) coefficients
    static constexpr double c2 = 0.526001519587677318785587544488e-01;
    static constexpr double c3 = 0.789002279381515978178381316732e-01;
    static constexpr double c4 = 0.118350341907227396726757197510e+00;
    static constexpr double c5 = 0.281649658092772603273242802490e+00;
    static constexpr double c6 = 0.333333333333333333333333333333e+00;
    static constexpr double c7 = 0.25e+00;
    static constexpr double c8 = 0.307692307692307692307692307692e+00;
    static constexpr double c9 = 0.651282051282051282051282051282e+00;
    static constexpr double c10 = 0.6e+00;
    static constexpr double c11 = 0.857142857142857142857142857142e+00;

    static constexpr double b1 = 5.42937341165687622380535766363e-2;
    static constexpr double b6 = 4.45031289275240888144113950566e0;
    static constexpr double b7 = 1.89151789931450038304281599044e0;
    static constexpr double b8 = -5.8012039600105847814672114227e0;
    static constexpr double b9 = 3.1116436695781989440891606237e-1;
    static constexpr double b10 = -1.52160949662516078556178806805e-1;
    static constexpr double b11 = 2.01365400804030348374776537501e-1;
    static constexpr double b12 = 4.47106157277725905176885569043e-2;

    static constexpr double bhh1 = 0.244094488188976377952755905512e+00;
    static constexpr double bhh2 = 0.733846688281611857341361741547e+00;
    static constexpr double bhh3 = 0.220588235294117647058823529412e-01;

    static constexpr double er1 = 0.1312004499419488073250102996e-01;
    static constexpr double er6 = -0.1225156446376204440720569753e+01;
    static constexpr double er7 = -0.4957589496572501915214079952e+00;
    static constexpr double er8 = 0.1664377182454986536961530415e+01;
    static constexpr double er9 = -0.3503288487499736816886487290e+00;
    static constexpr double er10 = 0.3341791187130174790297318841e+00;
    static constexpr double er11 = 0.8192320648511571246570742613e-01;
    static constexpr double er12 = -0.2235530786388629525884427845e-01;

    static constexpr double a21 = 5.26001519587677318785587544488e-2;
    static constexpr double a31 = 1.97250569845378994544595329183e-2;
    static constexpr double a32 = 5.91751709536136983633785987549e-2;
    static constexpr double a41 = 2.95875854768068491816892993775e-2;
    static constexpr double a43 = 8.87627564304205475450678981324e-2;
    static constexpr double a51 = 2.41365134159266685502369798665e-1;
    static constexpr double a53 = -8.84549479328286085344864962717e-1;
    static constexpr double a54 = 9.24834003261792003115737966543e-1;
    static constexpr double a61 = 3.7037037037037037037037037037e-2;
    static constexpr double a64 = 1.70828608729473871279604482173e-1;
    static constexpr double a65 = 1.25467687566822425016691814123e-1;
    static constexpr double a71 = 3.7109375e-2;
    static constexpr double a74 = 1.70252211019544039314978060272e-1;
    static constexpr double a75 = 6.02165389804559606850219397283e-2;
    static constexpr double a76 = -1.7578125e-2;
    static constexpr double a81 = 3.70920001185047927108779319836e-2;
    static constexpr double a84 = 1.70383925712239993810214054705e-1;
    static constexpr double a85 = 1.07262030446373284651809199168e-1;
    static constexpr double a86 = -1.53194377486244017527936158236e-2;
    static constexpr double a87 = 8.27378916381402288758473766002e-3;
    static constexpr double a91 = 6.24110958716075717114429577812e-1;
    static constexpr double a94 = -3.36089262944694129406857109825e0;
    static constexpr double a95 = -8.68219346841726006818189891453e-1;
    static constexpr double a96 = 2.75920996994467083049415600797e1;
    static constexpr double a97 = 2.01540675504778934086186788979e1;
    static constexpr double a98 = -4.34898841810699588477366255144e1;
    static constexpr double a101 = 4.77662536438264365890433908527e-1;
    static constexpr double a104 = -2.48811461997166764192642586468e0;
    static constexpr double a105 = -5.90290826836842996371446475743e-1;
    static constexpr double a106 = 2.12300514481811942347288949897e1;
    static constexpr double a107 = 1.52792336328824235832596922938e1;
    static constexpr double a108 = -3.32882109689848629194453265587e1;
    static constexpr double a109 = -2.03312017085086261358222928593e-2;
    static constexpr double a111 = -9.3714243008598732571704021658e-1;
    static constexpr double a114 = 5.18637242884406370830023853209e0;
    static constexpr double a115 = 1.09143734899672957818500254654e0;
    static constexpr double a116 = -8.14978701074692612513997267357e0;
    static constexpr double a117 = -1.85200656599969598641566180701e1;
    static constexpr double a118 = 2.27394870993505042818970056734e1;
    static constexpr double a119 = 2.49360555267965238987089396762e0;
    static constexpr double a1110 = -3.0467644718982195003823669022e0;
    static constexpr double a121 = 2.27331014751653820792359768449e0;
    static constexpr double a124 = -1.05344954667372501984066689879e1;
    static constexpr double a125 = -2.00087205822486249909675718444e0;
    static constexpr double a126 = -1.79589318631187989172765950534e1;
    static constexpr double a127 = 2.79488845294199600508499808837e1;
    static constexpr double a128 = -2.85899827713502369474065508674e0;
    static constexpr double a129 = -8.87285693353062954433549289258e0;
    static constexpr double a1210 = 1.23605671757943030647266201528e1;
    static constexpr double a1211 = 6.43392746015763530355970484046e-1;
};

} // namespace np
