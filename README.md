# nevanlinna-pick-cpp

C++17 library for broadband impedance matching network synthesis via
Nevanlinna-Pick interpolation.

Given a complex load reflection coefficient and a filter specification, the
solver finds a matching network (output as a coupling matrix) that minimizes the
worst-case reflection across the passband. The algorithm is based on homotopy
continuation from a Chebyshev prototype, followed by an equiripple Newton
optimizer that drives all passband ripple peaks to equal height.

## Features

- Equiripple Newton optimizer with backtracking line search
- Homotopy continuation with adaptive Dormand-Prince 8(5,3) predictor-corrector
- All-pole and cross-coupled filters with prescribed transmission zeros
- Symmetric load detection and reduced parameterization
- Direct coupling matrix output (transversal form), ready for physical realization
- Single dependency: [Eigen](https://eigen.tuxfamily.org/) (header-only)

## Building

Requires a C++17 compiler and CMake 3.16+.

```bash
# Clone with Eigen (or place Eigen headers in external/eigen/)
git clone <this-repo>
cd nevanlinna-pick-cpp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# Run the benchmark suite
./build/benchmark_suite
```

If Eigen is not found in `external/eigen/`, CMake will look for a system
installation via `find_package(Eigen3)`.

## Quick Example

```cpp
#include "impedance_matching.hpp"
using namespace np;

// Define a load: Gamma_L(omega)
auto load = [](double omega) -> Complex {
    Complex j(0, 1);
    return 0.45 * std::exp(j * omega - 0.25 * omega * omega);
};

// Solve: order 8, two transmission zeros, 16 dB return loss, passband [-1, 1]
std::vector<Complex> tzs = {Complex(2, 0), Complex(3, 0)};
ImpedanceMatching matcher(load, 8, tzs, 16.0, -1.0, 1.0);

MatrixXcd cm = matcher.run();  // Returns (N+2) x (N+2) coupling matrix

std::cout << "Achieved: " << matcher.achieved_return_loss_db() << " dB\n";
```

## Benchmarks

Selected results from `benchmark_suite` (Release build, GCC 14, Ryzen 7):

| Test case | Time | Achieved RL |
|---|---|---|
| Gaussian order 2, all-pole | 40 ms | -21.8 dB |
| Gaussian order 4, TZ={2,3} | 184 ms | -17.0 dB |
| Gaussian order 6, TZ={2,3} | 573 ms | -16.8 dB |
| Gaussian order 8, TZ={2,3} | 1233 ms | -16.7 dB |
| Resistive order 6 | 14 ms | -23.0 dB |
| Reactive RL order 6 | 15 ms | -14.6 dB |
| Asymmetric order 8, TZ={2,3} | 2176 ms | -22.4 dB |
| Gaussian order 6, narrow band | 149 ms | -52.1 dB |

The full suite runs 34 test cases covering orders 2-8, six load types, varied
transmission zero configurations, and target return losses from 10 to 20 dB.

## Theory

The solver implements the generalized Nevanlinna-Pick interpolation method
described in:

> L. Baratchart, M. Olivi, F. Seyfert, "Generalized Nevanlinna-Pick
> interpolation on the boundary. Application to impedance matching,"
> *Proceedings of the 22nd International Symposium on Mathematical Theory of
> Networks and Systems (MTNS)*, 2016.
> [HAL hal-01249330](https://inria.hal.science/hal-01249330)

The INRIA [PUMA](https://project.inria.fr/puma/) project solves the same
problem using convex relaxation and semidefinite programming (SDP). This library
takes the complementary homotopy continuation approach, which is faster but
provides a local (rather than globally certified) optimum. In practice both
methods converge to the same equiripple solution for well-behaved loads.

## Web Interface

A browser-based interface is available in [`web/`](web/). Upload a Touchstone
(`.s1p`, `.s2p`) or CSV file, select a passband, and run the solver entirely
client-side via WebAssembly. See [`web/README.md`](web/README.md) for build
instructions.

## License

[MIT](LICENSE)
