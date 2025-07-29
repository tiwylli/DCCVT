#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

// For linear algebra operations
#include "linalg.h"
using namespace linalg::aliases;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// Function to compute the orientation of a tetrahedron
// Returns a positive value if the tetrahedron is oriented counter-clockwise
// Returns a negative value if the tetrahedron is oriented clockwise
// Returns zero if the tetrahedron is degenerate (collinear points)
float tetrahedron_orientation(const float3& v0, const float3& v1,
                              const float3& v2, const float3& v3) {
    // Compute the vectors from v0 to the other vertices
    float3 a = v1 - v0;
    float3 b = v2 - v0;
    float3 c = v3 - v0;
    // Compute the scalar triple product
    // This gives the signed volume of the tetrahedron
    return dot(a, cross(b, c));
}

// Numpy array-like structure for tetrahedra (N, 4)
std::vector<int> tetrahedra_index(const pybind11::array_t<int>& tetrahedra_array, 
    const pybind11::array_t<double>& points_array, 
    const pybind11::array_t<double>& site_array,
    const pybind11::array_t<int>& nearest_indices_array) {
    std::cout << "tetrahedra_index called with shape: "
              << tetrahedra_array.shape(0) << " tetrahedra." << std::endl;
    
    // Compute the number of tetrahedra defined per site
    std::vector<int> site_tetrahedra_count(site_array.shape(0), 0);
    auto tetrahedra = tetrahedra_array.unchecked<2>();
    for (ssize_t i = 0; i < tetrahedra.shape(0); ++i) {
        for (ssize_t j = 0; j < tetrahedra.shape(1); ++j) {
            int site_index = tetrahedra(i, j);
            if (site_index >= 0 && site_index < site_array.shape(0)) {
                site_tetrahedra_count[site_index]++;
            }
        }
    }

    // DEBUG
    // Compute the maximum number of tetrahedra per site
    // int max_tetrahedra = 0;
    // for (const auto& count : site_tetrahedra_count) {
    //     if (count > max_tetrahedra) {
    //         max_tetrahedra = count;
    //     }
    // }
    // std::cout << "Maximum number of tetrahedra per site: " << max_tetrahedra << std::endl;

    // Compute the cumulative sum of tetrahedra per site
    std::vector<int> cumulative_tetrahedra_count(site_array.shape(0) + 1, 0);
    cumulative_tetrahedra_count[0] = 0;
    for (size_t i = 0; i < site_tetrahedra_count.size(); ++i) {
        cumulative_tetrahedra_count[i + 1] = cumulative_tetrahedra_count[i] + site_tetrahedra_count[i];
    }   

    // Reset the site_tetrahedra_count to zero, to be used to compute the offset
    for (size_t i = 0; i < site_tetrahedra_count.size(); ++i) {
        site_tetrahedra_count[i] = 0;   
    }
    // Create an temporary array to hold the tetrahedra indices based on the vertex indices
    // This will be used to fast found the tetrahedra index for each point based on the closest site
    // std::cout << "Creating tetrahedra indices array with size: " 
    //           << cumulative_tetrahedra_count.back() << std::endl;
    std::vector<int> tetrahedra_indices(cumulative_tetrahedra_count.back(), -1);
    // Fill the tetrahedra_indices array with the tetrahedra indices based on the vertex indices
    for (ssize_t i = 0; i < tetrahedra.shape(0); ++i) {
        for (ssize_t j = 0; j < tetrahedra.shape(1); ++j) {
            int site_index = tetrahedra(i, j);
            if (site_index >= 0 && site_index < site_array.shape(0)) {
                // Compute the offset for the site index
                int offset = cumulative_tetrahedra_count[site_index] + site_tetrahedra_count[site_index];
                tetrahedra_indices[offset] = i; // Store the tetrahedra index
                site_tetrahedra_count[site_index]++;
            }
        }
    }

    // For each point, find the tetrahedra index based on the closest site
    // Check which is the tetrahedra that contains the point
    auto points = points_array.unchecked<2>();
    auto site = site_array.unchecked<2>();
    auto nearest_indices = nearest_indices_array.unchecked<1>();
    // Output the tetrahedra index for each point
    // std::cout << "Computing tetrahedra index for each point..." << std::endl;
    std::vector<int> tetrahedra_indices_for_points(points.shape(0), -1);

    // For all points
    for (ssize_t i = 0; i < points.shape(0); ++i) {
        // Load the point position
        float3 p = {
            static_cast<float>(points(i, 0)),
            static_cast<float>(points(i, 1)),
            static_cast<float>(points(i, 2))
        };

        // Get the nearest site index for the point
        int site_index = nearest_indices(i);
        // Check if the site index is valid
        if (site_index >= 0 && site_index < site_array.shape(0)) {
            // Compute the offset of the tetrahedra indices for the site
            int offset = cumulative_tetrahedra_count[site_index];
            // Check if the offset is valid
            if (offset < tetrahedra_indices.size()) {
                // Check all tetrahedra indices for the site
                for(int j = 0; j < site_tetrahedra_count[site_index]; ++j) {
                    int tetra_index = tetrahedra_indices[offset + j];
                    // Check if the tetrahedron contains the point
                    if (tetra_index != -1) {
                        // Get the vertices of the tetrahedron (4 vertices, 3D coordinates)
                        auto v0 = points(tetrahedra(tetra_index, 0), 0);
                        auto v1 = points(tetrahedra(tetra_index, 1), 0);
                        auto v2 = points(tetrahedra(tetra_index, 2), 0);
                        auto v3 = points(tetrahedra(tetra_index, 3), 0);

                        // Load the site coordinates
                        float3 a = {
                            static_cast<float>(site(v0, 0)),
                            static_cast<float>(site(v0, 1)),
                            static_cast<float>(site(v0, 2))
                        };
                        float3 b = {
                            static_cast<float>(site(v1, 0)),
                            static_cast<float>(site(v1, 1)),
                            static_cast<float>(site(v1, 2))
                        };
                        float3 c = {
                            static_cast<float>(site(v2, 0)),
                            static_cast<float>(site(v2, 1)),
                            static_cast<float>(site(v2, 2))
                        };
                        float3 d = {
                            static_cast<float>(site(v3, 0)),  
                            static_cast<float>(site(v3, 1)),
                            static_cast<float>(site(v3, 2))
                        };

                        // Check if the point is inside the tetrahedron
                        // with the orientation test
                        bool o1 = tetrahedron_orientation(b, c, d, a) * tetrahedron_orientation(b, c, d, p) >= 0;
                        bool o2 = tetrahedron_orientation(a, c, d, b) * tetrahedron_orientation(a, c, d, p) >= 0;
                        bool o3 = tetrahedron_orientation(a, b, d, c) * tetrahedron_orientation(a, b, d, p) >= 0;
                        bool o4 = tetrahedron_orientation(a, b, c, d) * tetrahedron_orientation(a, b, c, p) >= 0;

                        // If all orientations are the same, the point is inside the tetrahedron
                        if (o1 && o2 && o3 && o4) {
                            // Store the tetrahedron index for the point
                            tetrahedra_indices_for_points[i] = tetra_index;
                            break; // Found the tetrahedron for this point, no need to check further
                        }
                    } else {
                        // Warning: Tetrahedron index is -1, which means it was not found
                        std::cout << "Warning: Tetrahedron index is -1 for site index " 
                                  << site_index << " at offset " << offset + j << std::endl;
                    }
                }

                // If no tetrahedron was found for the point, it remains -1
                // but emit an warning
                if (tetrahedra_indices_for_points[i] == -1) {
                    std::cout << "Warning: No tetrahedron found for point index " 
                              << i << " with site index " << site_index 
                              << " at offset " << offset << std::endl;
                }
            } else {
                std::cout << "Warning: Offset " << offset << " is out of bounds for tetrahedra_indices array." << std::endl;
            }
        } else {
            std::cout << "Warning: Site index " << site_index << " is out of bounds for site array." << std::endl;
        }
    }   
    

    return tetrahedra_indices_for_points;
}

namespace py = pybind11;

PYBIND11_MODULE(voronoiaccel, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("tetrahedra_index", &tetrahedra_index, R"pbdoc(
        Accelerate the computation of tetrahedra index cells.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
