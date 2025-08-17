#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

// For linear algebra operations
#include "linalg.h"
using namespace linalg::aliases;

#include "fcpw/fcpw.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define ENABLE_TIMING 0

std::tuple<float, float, float,  float, float, float, float, float, float, float> compute_error_fcpw(
    const pybind11::array_t<float>& target_vertices,
    const pybind11::array_t<int>& target_face, 
    const pybind11::array_t<float>& target_pos,
    const pybind11::array_t<float>& target_normal, // Normal on sampled points
    const pybind11::array_t<float>& pred_vertices,
    const pybind11::array_t<int>& pred_face,
    const pybind11::array_t<float>& pred_pos,
    const pybind11::array_t<float>& pred_normal, // Normal on sampled points
    const float threshold, // For F1 computation
    const float max_radius) {
    // Convert information to fcpw
    /// TARGET
#if ENABLE_TIMING
    #include <chrono>
    using namespace std::chrono;

    // Timing: conversion to fcpw
    auto start_conversion = high_resolution_clock::now();
#endif
    auto target_pos_ptr = target_pos.unchecked<2>();
    std::vector<fcpw::Vector<3>> target_pos_fcpw(target_pos.shape(0));
    for (ssize_t i = 0; i < target_pos.shape(0); ++i) {
        target_pos_fcpw[i] = fcpw::Vector<3>(target_pos_ptr(i, 0), target_pos_ptr(i, 1), target_pos_ptr(i, 2));
    }
    auto target_face_ptr = target_face.unchecked<2>();
    std::vector<fcpw::Vector3i> target_face_fcpw(target_face.shape(0));
    for (ssize_t i = 0; i < target_face.shape(0); ++i) {
        target_face_fcpw[i] = fcpw::Vector3i(target_face_ptr(i, 0), target_face_ptr(i, 1), target_face_ptr(i, 2));
    }
    auto target_vertices_ptr = target_vertices.unchecked<2>();
    std::vector<fcpw::Vector<3>> target_vertices_fcpw(target_vertices.shape(0));
    for (ssize_t i = 0; i < target_vertices.shape(0); ++i) {
        target_vertices_fcpw[i] = fcpw::Vector<3>(target_vertices_ptr(i, 0), target_vertices_ptr(i, 1), target_vertices_ptr(i, 2));
    }
    /// PRED
    auto pred_pos_ptr = pred_pos.unchecked<2>();
    std::vector<fcpw::Vector<3>> pred_pos_fcpw(pred_pos.shape(0));
    for (ssize_t i = 0; i < pred_pos.shape(0); ++i) {
        pred_pos_fcpw[i] = fcpw::Vector<3>(pred_pos_ptr(i, 0), pred_pos_ptr(i, 1), pred_pos_ptr(i, 2));
    }
    auto pred_face_ptr = pred_face.unchecked<2>();
    std::vector<fcpw::Vector3i> pred_face_fcpw(pred_face.shape(0));
    for (ssize_t i = 0; i < pred_face.shape(0); ++i) {
        pred_face_fcpw[i] = fcpw::Vector3i(pred_face_ptr(i, 0), pred_face_ptr(i, 1), pred_face_ptr(i, 2));
    }
    auto pred_vertices_ptr = pred_vertices.unchecked<2>();
    std::vector<fcpw::Vector<3>> pred_vertices_fcpw(pred_vertices.shape(0));
    for (ssize_t i = 0; i < pred_vertices.shape(0); ++i) {
        pred_vertices_fcpw[i] = fcpw::Vector<3>(pred_vertices_ptr(i, 0), pred_vertices_ptr(i, 1), pred_vertices_ptr(i, 2));
    }
#if ENABLE_TIMING
    auto end_conversion = high_resolution_clock::now();
    auto duration_conversion = duration_cast<milliseconds>(end_conversion - start_conversion).count();
    std::cout << "[Timing] Conversion to fcpw: " << duration_conversion << " ms" << std::endl;

    // Timing: intersection computation (scene build + closest points)
    auto start_intersection = high_resolution_clock::now();
#endif
    fcpw::AggregateType aggregateType = fcpw::AggregateType::Bvh_SurfaceArea;
    bool printStats = false;
    bool reduceMemoryFootprint = false;
    bool buildVectorizedBvh = true;

    fcpw::Scene<3> scene_target;
    scene_target.setObjectCount(1);
    scene_target.setObjectVertices(target_vertices_fcpw, 0);
    scene_target.setObjectTriangles(target_face_fcpw, 0);
    scene_target.build(aggregateType, buildVectorizedBvh, printStats, reduceMemoryFootprint);

    fcpw::Scene<3> scene_pred;
    scene_pred.setObjectCount(1);
    scene_pred.setObjectVertices(pred_vertices_fcpw, 0);
    scene_pred.setObjectTriangles(pred_face_fcpw, 0);
    scene_pred.build(aggregateType, buildVectorizedBvh, printStats, reduceMemoryFootprint);

    // Compute the error
    /// From the target
    std::vector<fcpw::BoundingSphere<3>> bs_target;
    for (const fcpw::Vector<3>& q: target_pos_fcpw) {
        bs_target.emplace_back(fcpw::BoundingSphere<3>(q, max_radius));
    }
    std::vector<fcpw::Interaction<3>> interactions_target;
    scene_pred.findClosestPoints(bs_target, interactions_target, true);

    /// From the pred
    std::vector<fcpw::BoundingSphere<3>> bs_pred;
    for (const fcpw::Vector<3>& q: pred_pos_fcpw) {
        bs_pred.emplace_back(fcpw::BoundingSphere<3>(q, max_radius));
    }
    std::vector<fcpw::Interaction<3>> interactions_pred;
    scene_target.findClosestPoints(bs_pred, interactions_pred, true);

#if ENABLE_TIMING
    auto end_intersection = high_resolution_clock::now();
    auto duration_intersection = duration_cast<milliseconds>(end_intersection - start_intersection).count();
    
    // Timing: error computation
    auto start_error = high_resolution_clock::now();
#endif

    double distance2_target = 0.0;
    double distance1_target = 0.0;
    double distance_threshold_target = 0.0;
    auto target_normal_ptr = target_normal.unchecked<2>();
    double normal_consistency_target = 0.0;
    for (ssize_t i = 0; i < target_pos.shape(0); ++i) {
        auto diff = interactions_target[i].p - target_pos_fcpw[i];
        distance2_target += diff.squaredNorm();
        distance1_target += diff.norm(); 
        if (diff.norm() < threshold) {
            distance_threshold_target += 1.0;
        }
        normal_consistency_target += std::abs(interactions_target[i].n.dot(fcpw::Vector<3>(
            target_normal_ptr(i, 0), target_normal_ptr(i, 1), target_normal_ptr(i, 2)
        )));
    }
    distance2_target /= target_pos.shape(0);
    distance1_target /= target_pos.shape(0);
    distance_threshold_target /= target_pos.shape(0);
    normal_consistency_target /= target_pos.shape(0);

    double distance2_pred = 0.0;
    double distance1_pred = 0.0;
    double distance_threshold_pred = 0.0;
    auto pred_normal_ptr = pred_normal.unchecked<2>();
    double normal_consistency_pred = 0.0;
    for (ssize_t i = 0; i < pred_pos.shape(0); ++i) {
        auto diff = interactions_pred[i].p - pred_pos_fcpw[i];
        distance2_pred += diff.squaredNorm();
        distance1_pred += diff.norm();
        if (diff.norm() < threshold) {
            distance_threshold_pred += 1.0;
        }
        normal_consistency_pred += std::abs(interactions_pred[i].n.dot(fcpw::Vector<3>(
            pred_normal_ptr(i, 0), pred_normal_ptr(i, 1), pred_normal_ptr(i, 2)
        )));
    }
    std::cout << "Normal consistency (pred) for point " << normal_consistency_pred << std::endl;   
    distance2_pred /= pred_pos.shape(0);
    distance1_pred /= pred_pos.shape(0);
    distance_threshold_pred /= pred_pos.shape(0);
    normal_consistency_pred /= pred_pos.shape(0);

    // Compute CD, precision, recall, F1
    double cd1 = distance1_pred + distance1_target;
    double cd2 = distance2_pred + distance2_target;
    double recall = distance_threshold_pred;
    double precision = distance_threshold_target;
    double f1 = 2 * precision * recall / (precision + recall);
    double ndc = (normal_consistency_pred + normal_consistency_target) / 2.0; // TODO

#if ENABLE_TIMING
    auto end_error = high_resolution_clock::now();
    auto duration_error = duration_cast<milliseconds>(end_error - start_error).count();
    std::cout << "[Timing] Error computation: " << duration_error << " ms" << std::endl;
#endif

    // DEBUG:
    // std::vector<std::tuple<float, float, float>> pred_closest(pred_pos.shape(0));
    // for (ssize_t i = 0; i < pred_pos.shape(0); ++i) {
    //     auto p = interactions_pred[i].p;
    //     pred_closest[i] = std::make_tuple(p.x, p.y, p.z);
    // }
    // std::vector<std::tuple<float, float, float>> target_closest(pred_pos.shape(0));
    // for (ssize_t i = 0; i < pred_pos.shape(0); ++i) {
    //     auto p = interactions_target[i].p;
    //     target_closest[i] = std::make_tuple(p.x, p.y, p.z);
    // }

    // Compute the error using the FCPW library
    // cd1, cd2, f1, 0.0, float(recall), float(precision), completeness1, completeness2, accuracy1, accuracy2
    return std::make_tuple(cd1, cd2, f1, ndc, recall, precision, distance1_pred, distance2_pred, distance1_target, distance2_target, pred_closest, target_closest);
}


// Function to compute the orientation of a tetrahedron
// Returns a positive value if the tetrahedron is oriented counter-clockwise
// Returns a negative value if the tetrahedron is oriented clockwise
// Returns zero if the tetrahedron is degenerate (collinear points)
float tetrahedron_orientation(const float3& a, const float3& b,
                              const float3& c, const float3& d) {
    // Compute the scalar triple product
    // This gives the signed volume of the tetrahedron
    return dot(d - a, cross(b - a, c - a));
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
    auto nearest_indices = nearest_indices_array.unchecked<2>(); // N first closest sites
    // Output the tetrahedra index for each point
    // std::cout << "Computing tetrahedra index for each point..." << std::endl;
    std::vector<int> tetrahedra_indices_for_points(points.shape(0), -1);

    // For all points
    int count_site_assigned = 0;
    for (ssize_t i = 0; i < points.shape(0); ++i) {
        // Load the point position
        float3 p = {
            static_cast<float>(points(i, 0)),
            static_cast<float>(points(i, 1)),
            static_cast<float>(points(i, 2))
        };
        // Get the nearest site index for the point
        bool found_tetrahedra = false;
        for(ssize_t k = 0; k < nearest_indices.shape(1) && !found_tetrahedra; ++k) {
            int site_index = nearest_indices(i, k);
        
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
                            auto v0 = tetrahedra(tetra_index, 0);
                            auto v1 = tetrahedra(tetra_index, 1);
                            auto v2 = tetrahedra(tetra_index, 2);
                            auto v3 = tetrahedra(tetra_index, 3);

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
                                found_tetrahedra = true;
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
                    // if (tetrahedra_indices_for_points[i] == -1) {
                    //     count_site_assigned += 1;
                    //     // std::cout << "Warning: No tetrahedron found for point index " 
                    //     //           << i << " with site index " << site_index 
                    //     //           << " at offset " << offset << std::endl;
                    //     // std::cout << "Point coordinate " << p.x << ", " 
                    //     //           << p.y << ", " << p.z << std::endl;
                    // }
                } else {
                    std::cout << "Warning: Offset " << offset << " is out of bounds for tetrahedra_indices array." << std::endl;
                }
            } else {
                std::cout << "Warning: Site index " << site_index << " is out of bounds for site array." << std::endl;
            }
        }

        if(!found_tetrahedra) {
            // If no tetrahedron was found for the point, it remains -1
            // Increment the count of site assigned
            count_site_assigned += 1;
        }
    }

    if (count_site_assigned > 0) {
        std::cout << "Warning: " << count_site_assigned 
                  << " points were not assigned to any tetrahedron." << std::endl;
        // Pourcentage of site
        float percentage = static_cast<float>(count_site_assigned) / points.shape(0) * 100.0f;
        std::cout << "Percentage of points not assigned to any tetrahedron: "
                  << percentage << "%" << std::endl;
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

    m.def("compute_error_fcpw", &compute_error_fcpw, R"pbdoc(
        Compute the error with fcpw
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
