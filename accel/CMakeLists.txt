cmake_minimum_required(VERSION 3.10)
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