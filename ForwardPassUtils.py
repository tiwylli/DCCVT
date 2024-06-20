# This file contains utility functions for the forward pass notebook
import numpy as np
from matplotlib import pyplot as plt


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
    ax.plot([bisector_start[0], bisector_end[0]], [bisector_start[1], bisector_end[1]], color=color)
