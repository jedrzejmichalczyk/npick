cmake_minimum_required(VERSION 3.16)
project(npick_web LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen (local copy from parent)
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_DIR}"
)

# Build the main library sources directly (no separate lib for Emscripten)
set(LIB_SOURCES
    ../src/polynomial.cpp
    ../src/spectral_factor.cpp
    ../src/chebyshev_filter.cpp
    ../src/coupling_matrix.cpp
    ../src/nevanlinna_pick.cpp
    ../src/impedance_matching.cpp
    ../src/path_tracker.cpp
    ../src/dopr853_predictor.cpp
    ../src/corrector.cpp
)

add_executable(npick ${LIB_SOURCES} npick_bindings.cpp)

target_include_directories(npick PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(npick PRIVATE Eigen3::Eigen)

# Emscripten settings
set_target_properties(npick PROPERTIES
    SUFFIX ".js"
)
target_link_options(npick PRIVATE
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createNpickModule"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ENVIRONMENT=web,node"
    "SHELL:--bind"
    "SHELL:-O3"
)
