# npick web interface

Browser-based impedance matching solver. Upload an S-parameter file, select a
passband, and get an optimized matching network — all client-side via WebAssembly.

## Building

Requires [Emscripten](https://emscripten.org/docs/getting_started/downloads.html)
and [Ninja](https://ninja-build.org/).

### Linux / macOS

```bash
source ~/emsdk/emsdk_env.sh
cd web
mkdir -p build && cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
emmake cmake --build . -j$(nproc)
cp npick.js npick.wasm ..
```

### Windows (CMD)

```cmd
call %USERPROFILE%\emsdk\emsdk_env.bat
cd web
mkdir build & cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
emmake cmake --build . -j8
copy npick.js .. & copy npick.wasm ..
```

## Running locally

```bash
cd web
python3 -m http.server 8080
# Open http://localhost:8080
```

## Deploying to GitHub Pages

Copy these files to your `gh-pages` branch or `docs/` folder:

- `index.html`
- `app.js`
- `touchstone.js`
- `style.css`
- `npick.js` (generated)
- `npick.wasm` (generated)
- `sample.s1p` (optional, for demo)

## Supported file formats

- **Touchstone .s1p** — 1-port S-parameters (S11 used as load reflection)
- **Touchstone .s2p** — 2-port S-parameters (select which Sij to use)
- **CSV** — columns: frequency, real, imaginary
- Formats: RI (real/imaginary), MA (magnitude/angle), DB (dB/angle)
- Frequency units: Hz, kHz, MHz, GHz
