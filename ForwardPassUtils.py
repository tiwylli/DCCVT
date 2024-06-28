# This file contains utility functions for the forward pass notebook
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d


def computeMinMax(voronoi, buffer=1.2):
    maxX = 0
    minX = 0
    maxY = 0
    minY = 0
    for i in range(len(voronoi.vertices)):
        if maxX < voronoi.vertices[i][0]:
            maxX = voronoi.vertices[i][0]
        if minX > voronoi.vertices[i][0]:
            minX = voronoi.vertices[i][0]
        if maxY < voronoi.vertices[i][1]:
            maxY = voronoi.vertices[i][1]
        if minY > voronoi.vertices[i][1]:
            minY = voronoi.vertices[i][1]
    for i in range(len(voronoi.points)):
        if maxX < voronoi.points[i][0]:
            maxX = voronoi.points[i][0]
        if minX > voronoi.points[i][0]:
            minX = voronoi.points[i][0]
        if maxY < voronoi.points[i][1]:
            maxY = voronoi.points[i][1]
        if minY > voronoi.points[i][1]:
            minY = voronoi.points[i][1]

    return tuple(x * buffer for x in (minX, maxX, minY, maxY))


# Function to plot the perpendicular bisector
def plot_perpendicular_bisector(p1, p2, ax, color='r', length=5):
    # Midpoint of the line segment
    mid = (p1 + p2) / 2
    # Direction vector of the line segment
    dx, dy = p2 - p1
    # Perpendicular direction vector
    perp_dx, perp_dy = -dy, dx
    # Normalize the perpendicular direction vector
    norm = np.sqrt(perp_dx ** 2 + perp_dy ** 2)
    perp_dx, perp_dy = perp_dx / norm, perp_dy / norm
    # Scale to the desired length
    perp_dx, perp_dy = perp_dx * length, perp_dy * length
    # Endpoints of the perpendicular bisector
    bisector_start = mid + np.array([perp_dx, perp_dy])
    bisector_end = mid - np.array([perp_dx, perp_dy])
    # Plot the perpendicular bisector
    ax.plot([bisector_start[0], bisector_end[0]], [bisector_start[1], bisector_end[1]], linestyle='--', color=color)


def sdf_circle(x, y, radius):
    return np.sqrt(x ** 2 + y ** 2) - radius


def check_voronoi_sign_change(voronoi, sdf_func, radius=1):
    # Initialize a list to store edges where SDF sign changes
    sign_change_edges = []

    # Iterate over the ridges in the Voronoi diagram
    for ridge_points in voronoi.ridge_points:
        p1 = voronoi.points[ridge_points[0]]
        p2 = voronoi.points[ridge_points[1]]

        sdf1 = sdf_func(p1[0], p1[1], radius)
        sdf2 = sdf_func(p2[0], p2[1], radius)

        # Check if the SDF signs are different
        if np.sign(sdf1) != np.sign(sdf2):
            sign_change_edges.append((p1, p2))

    return sign_change_edges


def check_voronoi_cells_sign_change(voronoi, sdf_func, radius=1):
    # Initialize a list to store indices of sites where SDF sign changes
    sign_change_cells = []

    # Iterate over the Voronoi points
    for i, point in enumerate(voronoi.points):
        sdf_value = sdf_func(point[0], point[1], radius)

        # Check if the SDF sign is negative (inside the shape)
        if np.sign(sdf_value) < 0:
            sign_change_cells.append(i)

    return sign_change_cells


def k_nearest_neighbors(points, k):
    num_points = len(points)
    neighbors = []

    for i in range(num_points):
        # Calculate the Euclidean distances from the current point to all other points
        distances = np.linalg.norm(points - points[i], axis=1)

        # Exclude the current point by setting its distance to infinity
        distances[i] = np.inf

        # Get the indices of the k smallest distances
        nearest_neighbors_indices = np.argpartition(distances, k)[:k]
        neighbors.append(nearest_neighbors_indices)

    return neighbors


def find_bisector_intersections(points, neighbors):
    intersections = []

    for i, neighbor_indices in enumerate(neighbors):
        for j, neighbor_index in enumerate(neighbor_indices):
            for k, other_neighbor_index in enumerate(neighbor_indices):
                if j < k:  # Ensure we only process each pair once
                    # Midpoints of the segments
                    midpoint1 = (points[i] + points[neighbor_index]) / 2
                    midpoint2 = (points[i] + points[other_neighbor_index]) / 2

                    # Slopes of the segments
                    dx1 = points[neighbor_index][0] - points[i][0]
                    dy1 = points[neighbor_index][1] - points[i][1]
                    dx2 = points[other_neighbor_index][0] - points[i][0]
                    dy2 = points[other_neighbor_index][1] - points[i][1]

                    # Slopes of the perpendicular bisectors
                    if dy1 != 0:
                        perp_slope1 = -dx1 / dy1
                    else:
                        perp_slope1 = np.inf

                    if dy2 != 0:
                        perp_slope2 = -dx2 / dy2
                    else:
                        perp_slope2 = np.inf

                    # Intercept of the perpendicular bisectors
                    intercept1 = midpoint1[1] - perp_slope1 * midpoint1[0] if perp_slope1 != np.inf else midpoint1[0]
                    intercept2 = midpoint2[1] - perp_slope2 * midpoint2[0] if perp_slope2 != np.inf else midpoint2[0]

                    # Solve the system of equations to find the intersection
                    if perp_slope1 != np.inf and perp_slope2 != np.inf:
                        A = np.array([[perp_slope1, -1], [perp_slope2, -1]])
                        b = np.array([-intercept1, -intercept2])
                        try:
                            intersection = np.linalg.solve(A, b)
                            intersections.append(intersection)
                        except np.linalg.LinAlgError:
                            continue  # Skip if the lines are parallel
                    else:
                        if perp_slope1 == np.inf:
                            x_intersect = intercept1
                            y_intersect = perp_slope2 * x_intersect + intercept2
                        else:
                            x_intersect = intercept2
                            y_intersect = perp_slope1 * x_intersect + intercept1
                        intersection = np.array([x_intersect, y_intersect])
                        intersections.append(intersection)

    return intersections
