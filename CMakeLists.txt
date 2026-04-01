cmake_minimum_required(VERSION 3.16)
project(nevanlinna_pick VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MinGW needs big-obj for template-heavy Eigen code
if(MINGW)
    add_compile_options(-Wa,-mbig-obj)
endif()

# Dependencies - use local Eigen copy
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
if(EXISTS ${EIGEN_DIR})
    message(STATUS "Using local Eigen from ${EIGEN_DIR}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_DIR}"
    )
else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

# Main library
add_library(nevanlinna_pick
    src/polynomial.cpp
    src/spectral_factor.cpp
    src/chebyshev_filter.cpp
    src/coupling_matrix.cpp
    src/nevanlinna_pick.cpp
    src/impedance_matching.cpp
    src/path_tracker.cpp
    src/dopr853_predictor.cpp
    src/corrector.cpp
)
target_include_directories(nevanlinna_pick PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nevanlinna_pick PUBLIC Eigen3::Eigen)

# Example executables
add_executable(matching_example examples/matching_example.cpp)
target_link_libraries(matching_example PRIVATE nevanlinna_pick)

add_executable(test_imp_matching examples/test_imp_matching.cpp)
target_link_libraries(test_imp_matching PRIVATE nevanlinna_pick)

add_executable(realization_check examples/realization_check.cpp)
target_link_libraries(realization_check PRIVATE nevanlinna_pick)

add_executable(optimization_check examples/optimization_check.cpp)
target_link_libraries(optimization_check PRIVATE nevanlinna_pick)

add_executable(sparameter_matching examples/sparameter_matching.cpp)
target_link_libraries(sparameter_matching PRIVATE nevanlinna_pick)

add_executable(first_example_benchmark examples/first_example_benchmark.cpp)
target_link_libraries(first_example_benchmark PRIVATE nevanlinna_pick)

add_executable(diagnostic examples/diagnostic.cpp)
target_link_libraries(diagnostic PRIVATE nevanlinna_pick)

add_executable(benchmark_suite examples/benchmark_suite.cpp)
target_link_libraries(benchmark_suite PRIVATE nevanlinna_pick)

# Tests (optional)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()

        add_executable(tests
            test/test_polynomial.cpp
            test/test_spectral_factor.cpp
            test/test_chebyshev.cpp
            test/test_nevanlinna.cpp
            test/test_impedance_matching.cpp
        )
        target_link_libraries(tests PRIVATE nevanlinna_pick GTest::gtest_main)
        add_test(NAME AllTests COMMAND tests)
    else()
        message(STATUS "Google Test not found, tests will not be built")
    endif()
endif()

# Installation
install(TARGETS nevanlinna_pick EXPORT nevanlinna_pick-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
