cmake_minimum_required(VERSION 3.10)

# CMake >= 3.28 removed compatibility with cmake_minimum_required(VERSION < 3.5).
# Some vendored/submodule dependencies (pybind11, enoki) still declare older
# minimums, so set this to allow configuring them without patching submodules.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for third-party CMake projects" FORCE)

project(voronoiaccel)

add_subdirectory(pybind11)
add_subdirectory(3rdparty/fcpw)
pybind11_add_module(voronoiaccel src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(voronoiaccel
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(voronoiaccel PRIVATE fcpw)
target_include_directories(voronoiaccel PRIVATE ${FCPW_EIGEN_INCLUDES})
target_include_directories(voronoiaccel PRIVATE ${FCPW_ENOKI_INCLUDES})
