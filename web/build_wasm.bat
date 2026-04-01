@echo off
REM Build WASM module using Emscripten on Windows
REM Requires: emsdk installed at %USERPROFILE%\emsdk

setlocal

call "%USERPROFILE%\emsdk\emsdk_env.bat" >nul 2>&1

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%build

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

call emcmake cmake "%SCRIPT_DIR%" -DCMAKE_BUILD_TYPE=Release -G Ninja
if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

call emmake cmake --build . -j8
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

copy /y npick.js "%SCRIPT_DIR%\" >nul
copy /y npick.wasm "%SCRIPT_DIR%\" >nul

echo.
echo Build complete. Files:
echo   %SCRIPT_DIR%npick.js
echo   %SCRIPT_DIR%npick.wasm
echo.
echo To serve locally:
echo   cd %SCRIPT_DIR% ^&^& python -m http.server 8080
echo   Open http://localhost:8080
