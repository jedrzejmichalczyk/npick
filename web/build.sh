#!/bin/bash
# Build the WASM module using Emscripten.
# Requires: emsdk activated (source emsdk_env.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
emmake cmake --build . -j$(nproc 2>/dev/null || echo 4)

# Copy outputs to web directory
cp npick.js npick.wasm "$SCRIPT_DIR/"

echo ""
echo "Build complete. Files:"
echo "  $SCRIPT_DIR/npick.js"
echo "  $SCRIPT_DIR/npick.wasm"
echo ""
echo "To serve locally:"
echo "  cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
