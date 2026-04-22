# npick — Browser-based Multiplexer Synthesis & Impedance Matching (Nevanlinna-Pick)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake 3.16+](https://img.shields.io/badge/CMake-3.16%2B-blue.svg)](https://cmake.org/)
[![Try it online](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://jedrzejmichalczyk.github.io/npick/)

**npick** is a free, open-source, browser-based synthesis tool for two
long-standing problems in RF and microwave engineering:

1. **Manifold-coupled multiplexer synthesis** — duplexers, triplexers, and
   N-channel multiplexers with ideal or user-supplied T-junctions, following
   the Martínez et al. (EuMC 2019) continuation method.
2. **Broadband impedance matching** — Chebyshev-type matching networks for a
   complex-valued load specified by Touchstone S-parameter files, following
   Baratchart–Olivi–Seyfert's generalized Nevanlinna-Pick interpolation.

Both run entirely client-side in the browser via WebAssembly at
[**jedrzejmichalczyk.github.io/npick**](https://jedrzejmichalczyk.github.io/npick/).
No installation, no license server, no data leaves your machine.

As far as we know, this is the **only openly-available browser tool** that
synthesizes manifold-coupled multiplexers from a per-channel specification —
commercial RF-design packages (CST, AWR, HFSS, Microwave Office, …) can do it
but require paid licenses and desktop installs; published research code is
mostly single-channel MATLAB scripts.

## Live demo

<https://jedrzejmichalczyk.github.io/npick/>

- **Impedance Matching** tab: drop a `.s1p` / `.s2p` / CSV file, pick a
  passband, get the coupling matrix and matched response.
- **Multiplexer Synthesis** tab: choose a duplexer or triplexer preset (or
  define channels manually), optionally upload `.s3p` T-junction S-parameters,
  click Synthesize. Output: per-channel coupling matrices plus the full
  frequency response of the assembled network.

## Key features

**Multiplexer synthesis (Martínez 2019):**
- Manifold-coupled N-channel synthesis via homotopy continuation λ = 0 → 1
- Correct Wirtinger-based complex Newton for the non-holomorphic residual
  F(P) = f(p) − conj(L(P))
- Wilson's iterative spectral factorization (root-free, numerically robust;
  see Wilson 1969, Sayed & Kailath 2001)
- Joint N·n equiripple optimization across all channels' interpolation points
- Custom T-junction S-parameters accepted via `.s3p` upload
- Duplexer example hits −20 dB worst-case RL in ~1 s, triplexer in ~2 min
  including equiripple tuning

**Impedance matching (Baratchart-Olivi-Seyfert):**
- Equiripple Newton optimizer with backtracking line search
- Homotopy continuation with adaptive Dormand-Prince 8(5,3) predictor-corrector
- All-pole and cross-coupled filters with prescribed transmission zeros
- Symmetric load detection and reduced parameterization
- Direct coupling-matrix output (transversal form) ready for physical realization

**Portable:**
- Single dependency: [Eigen](https://eigen.tuxfamily.org/) (header-only)
- Builds with GCC, Clang, MSVC, and Emscripten
- Same numerical behavior on native and WASM (carefully handles portability
  traps like `std::pow(complex(0,0), 0)` which differs between libstdc++ and libc++)

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

## Theory and references

**Multiplexer synthesis.** The continuation algorithm for manifold-coupled
multiplexers follows:

> D. Martínez Martínez, S. Bila, F. Seyfert, M. Olivi, O. Tantot, L. Carpentier,
> "Synthesis Method for Manifold-Coupled Multiplexers,"
> *49th European Microwave Conference (EuMC)*, Paris, 2019.
> [HAL hal-02377002](https://hal-unilim.archives-ouvertes.fr/hal-02377002)

The residual for the coupled matching problem is `F(P) = f(p_i)(ξ_{i,m}) − conj(L_i(P)(ξ_{i,m}))`
where `L_i` is the load seen at filter `i` through the manifold. It is not
holomorphic in `p`, so Newton's method in complex variables fails silently; npick
uses proper Wirtinger derivatives to assemble a real 2N × 2N Jacobian and
converges quadratically.

**Impedance matching.** The single-channel matcher implements:

> L. Baratchart, M. Olivi, F. Seyfert, "Generalized Nevanlinna-Pick
> interpolation on the boundary. Application to impedance matching,"
> *Proceedings of the 22nd International Symposium on Mathematical Theory of
> Networks and Systems (MTNS)*, 2016.
> [HAL hal-01249330](https://inria.hal.science/hal-01249330)

The INRIA [PUMA](https://project.inria.fr/puma/) project solves the same
problem using convex relaxation and SDP. npick takes the complementary homotopy
continuation approach, which is faster but provides a local (rather than
globally certified) optimum. In practice both methods converge to the same
equiripple solution for well-behaved loads.

**Spectral factorization.** Implemented via Wilson's Newton iteration on the
factor coefficients (root-free, quadratically convergent):

> G. T. Wilson, "Factorization of the covariance generating function of a pure
> moving-average process," *SIAM J. Numer. Anal.* 6(1), 1–7, 1969.
> [doi:10.1137/0706001](https://doi.org/10.1137/0706001)
>
> A. H. Sayed, T. Kailath, "A survey of spectral factorization methods,"
> *Numer. Linear Algebra Appl.* 8(6–7), 467–496, 2001.
> [doi:10.1002/nla.250](https://doi.org/10.1002/nla.250)

## Web demo source

The browser UI lives in [`docs/`](docs/) (served by GitHub Pages) and
[`web/`](web/) (build source). The two directories are kept in sync; `docs/`
ships the prebuilt `npick.js`/`npick.wasm` artifacts. See [`web/README.md`](web/README.md)
for rebuild instructions via Emscripten.

## Keywords

multiplexer synthesis, manifold-coupled multiplexer, duplexer synthesis,
triplexer synthesis, RF multiplexer design, diplexer, impedance matching,
Nevanlinna-Pick interpolation, Martinez 2019, Baratchart Olivi Seyfert,
coupling matrix synthesis, microwave filter design, broadband matching,
S-parameter solver, Touchstone, Feldtkeller equation, Wilson spectral
factorization, equiripple matching, homotopy continuation, Wirtinger derivatives,
C++ scientific computing, WebAssembly, browser-based EDA, free filter synthesis,
open-source RF design tool

## License

[MIT](LICENSE)
