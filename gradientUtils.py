import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch


# Define a function to plot and save the figure
def plot_and_save(
    epoch,
    s_i,
    s_j,
    s_k,
    circle_center,
    radius,
    X_g=None,
    Y_g=None,
    destination="images/",
):
    # check destination and make folder if not exists
    if not os.path.exists(destination):
        os.makedirs(destination)
    X, Y = compute_vertex(s_i, s_j, s_k)
    if isinstance(s_i, Site):
        x_i, y_i = s_i.x, s_i.y
        x_j, y_j = s_j.x, s_j.y
        x_k, y_k = s_k.x, s_k.y
    else:
        x_i, y_i = s_i
        x_j, y_j = s_j
        x_k, y_k = s_k

    # Calculate midpoints
    midpoint_ij_x, midpoint_ij_y = (x_i + x_j) / 2, (y_i + y_j) / 2
    midpoint_jk_x, midpoint_jk_y = (x_j + x_k) / 2, (y_j + y_k) / 2

    plt.figure()

    # Label sites
    plt.text(x_i.item(), y_i.item(), "s_i", fontsize=9, verticalalignment="bottom")
    plt.text(x_j.item(), y_j.item(), "s_j", fontsize=9, verticalalignment="bottom")
    plt.text(x_k.item(), y_k.item(), "s_k", fontsize=9, verticalalignment="bottom")
    plt.scatter(
        [x_i.item(), x_j.item(), x_k.item()],
        [y_i.item(), y_j.item(), y_k.item()],
        color="blue",
        label="Sites",
    )

    plt.scatter(X.item(), Y.item(), color="red", label="Computed Vertex")
    if X_g and Y_g:
        plt.scatter(
            X_g.item(),
            Y_g.item(),
            color="green",
            label="Ground Truth Vertex = Closest Point on Circle",
        )
    # Plot midpoints
    plt.scatter(
        [midpoint_ij_x.item(), midpoint_jk_x.item()],
        [midpoint_ij_y.item(), midpoint_jk_y.item()],
        color="purple",
        marker="x",
        label="Midpoints",
    )

    circle = Circle(
        (circle_center[0].item(), circle_center[1].item()),
        radius,
        color="gray",
        fill=False,
        linestyle="--",
        label="Ground Truth Circle",
    )
    plt.gca().add_patch(circle)
    plt.legend(loc=2)
    plt.title(f"{destination.strip('/').split('/')[-1]}_{epoch}")
    plt.xlabel("X")
    plt.ylabel("Y")
    # check min and max values of sites
    plt.ylim(
        min(circle_center[1].item() - radius, y_i.item(), y_j.item(), y_k.item()) * 1.1,
        max(circle_center[1].item() + radius, y_i.item(), y_j.item(), y_k.item()) * 1.1,
    )
    plt.xlim(
        min(circle_center[0].item() - radius, x_i.item(), x_j.item(), x_k.item()) * 1.1,
        max(circle_center[0].item() + radius, x_i.item(), x_j.item(), x_k.item()) * 1.1,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.savefig(f"{destination}{epoch}.png")
    plt.close()


# Define a function to compute the vertex coordinates
def compute_vertex(s_i, s_j, s_k):
    if isinstance(s_i, Site):
        x_i, y_i = s_i.x, s_i.y
        x_j, y_j = s_j.x, s_j.y
        x_k, y_k = s_k.x, s_k.y
    else:
        x_i, y_i = s_i
        x_j, y_j = s_j
        x_k, y_k = s_k
    N_X = (
        x_i**2 * (y_j - y_k)
        - x_j**2 * (y_i - y_k)
        + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)
    )
    D = 2 * (x_i * (y_j - y_k) - x_j * (y_i - y_k) + x_k * (y_i - y_j))
    X = N_X / D

    N_Y = -(
        x_i**2 * (x_j - x_k)
        - x_i * (x_j**2 - x_k**2 + y_j**2 - y_k**2)
        + x_j**2 * x_k
        - x_j * (x_k**2 - y_i**2 + y_k**2)
        - x_k * (y_i**2 - y_j**2)
    )
    Y = N_Y / D

    # return torch.tensor([X, Y], requires_grad=True)
    return X, Y


# Define a function to compute the vertex coordinates
def compute_vertex_p(s_i, s_j, s_k):
    if isinstance(s_i, Site):
        x_i, y_i = s_i.x, s_i.y
        x_j, y_j = s_j.x, s_j.y
        x_k, y_k = s_k.x, s_k.y
    else:
        x_i, y_i = s_i
        x_j, y_j = s_j
        x_k, y_k = s_k
    n_x = (
        x_i**2 * (y_j - y_k)
        - x_j**2 * (y_i - y_k)
        + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)
    )
    d = 2 * (x_i * (y_j - y_k) - x_j * (y_i - y_k) + x_k * (y_i - y_j))
    x = n_x / d

    n_y = -(
        x_i**2 * (x_j - x_k)
        - x_i * (x_j**2 - x_k**2 + y_j**2 - y_k**2)
        + x_j**2 * x_k
        - x_j * (x_k**2 - y_i**2 + y_k**2)
        - x_k * (y_i**2 - y_j**2)
    )
    y = n_y / d

    return torch.tensor([x, y], requires_grad=True)
    #return x,y


def midpoint_loss(s_i, s_j, s_k, circle_center, radius):
    # Calculate midpoints
    m_ij = ((s_i.x + s_j.x) / 2, (s_i.y + s_j.y) / 2)
    m_jk = ((s_j.x + s_k.x) / 2, (s_j.y + s_k.y) / 2)

    # Circle center and radius
    h, k = circle_center
    r = radius

    # Loss for each midpoint to be on the circle
    loss_m_ij = ((m_ij[0] - h) ** 2 + (m_ij[1] - k) ** 2 - r**2) ** 2
    loss_m_jk = ((m_jk[0] - h) ** 2 + (m_jk[1] - k) ** 2 - r**2) ** 2

    # Total midpoint loss
    total_loss = loss_m_ij + loss_m_jk

    return total_loss


def midpoint_loss_sdf(s_i, s_j, s_k, circle_center, radius):
    if isinstance(s_i, Site):
        x_i, y_i = s_i.x, s_i.y
        x_j, y_j = s_j.x, s_j.y
        x_k, y_k = s_k.x, s_k.y
    else:
        x_i, y_i = s_i
        x_j, y_j = s_j
        x_k, y_k = s_k
    # Calculate midpoints
    m_ij = ((x_i + x_j) / 2, (y_i + y_j) / 2)
    m_jk = ((x_j + x_k) / 2, (y_j + y_k) / 2)

    # Loss for each midpoint to be on the sdf
    loss_m_ij = circle_sdf(m_ij[0], m_ij[1], circle_center, radius) ** 2
    loss_m_jk = circle_sdf(m_jk[0], m_jk[1], circle_center, radius) ** 2

    # Total midpoint loss
    total_loss = loss_m_ij + loss_m_jk

    return total_loss


def compute_closest_point_on_circle(x, y, circle_center=torch.tensor([5, 5]), radius=3):
    # Calculate direction vector from point to circle center
    direction = torch.tensor([x - circle_center[0], y - circle_center[1]])
    # Normalize the direction
    norm_direction = direction / torch.norm(direction)
    # Scale by the circle's radius and add to circle center to get closest point
    closest_point = circle_center + norm_direction * radius
    return closest_point[0], closest_point[1]


def circle_sdf(x, y, circle_center=torch.tensor([5, 5]), radius=3):
    """Compute the signed distance function for a circle."""
    return (
        torch.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2) - radius
    )

def circle_sdf_p(p, circle_center=torch.tensor([0, 0]), radius=1):
    """Compute the signed distance function for a circle."""
    return (
        torch.sqrt((p[0].item() - circle_center[0]) ** 2 + (p[1].item() - circle_center[1]) ** 2) - radius
    )

# def approximate_gradient(x, y, circle_center, radius, delta=1e-5):
#     """Approximate the gradient of the SDF at (x, y) using finite differences."""
#     sdf_x = circle_sdf(x + delta, y, circle_center, radius ) - circle_sdf(x - delta, y, circle_center, radius )
#     sdf_y = circle_sdf(x, y + delta, circle_center, radius ) - circle_sdf(x, y - delta, circle_center, radius )
#     return sdf_x / (2 * delta), sdf_y / (2 * delta)


def midpoint_interpolation_sdf(sdfi, sdfj, sdfk):
    return (sdfi + sdfj) / 2 + (sdfj + sdfk) / 2


def barycentric_coordinates(p, p1, p2, p3):
    # clip sdf value si le point est dehors de la triangle
    # radial basis function

    # Calculate the vectors relative to p1
    # p = torch.tensor([x, y])
    v0 = p2 - p1
    v1 = p3 - p1
    v2 = p - p1

    # Calculate the dot products
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)

    # Calculate the denominator
    #  air tirangle ou autre chose
    denom = d00 * d11 - d01 * d01

    # Calculate the barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return u, v, w


def radial_basis_function(p, p1, p2, p3, sdf1, sdf2, sdf3, sigma=1):
    # Calculate the radial basis function
    #https://en.wikipedia.org/wiki/Radial_basis_function
    sdf = (
        torch.exp(-((torch.norm(p1 - p) / sigma) ** 2)) * sdf1
        + torch.exp(-((torch.norm(p2 - p) / sigma) ** 2)) * sdf2
        + torch.exp(-((torch.norm(p3 - p) / sigma) ** 2)) * sdf3
    )
    return sdf

def bary_rbf(p, p1, p2, p3, sdf1, sdf2, sdf3, sigma=1):
    u, v, w = barycentric_coordinates(p, p1, p2, p3)
    #if (u >= 0) and (v >= 0) and (w >= 0) and (u + v + w == 1):
    if (u > 0) and (v > 0) and (w > 0) and (u<1) and (v<1) and (w<1):
        return u * sdf1 + v * sdf2 + w * sdf3
    else:
        return radial_basis_function(p, p1, p2, p3, sdf1, sdf2, sdf3, sigma)







# Define the Site class
class Site:
    def __init__(self, x, y):
        self.x = torch.nn.Parameter(
            torch.tensor([x], dtype=torch.float32, requires_grad=True)
        )
        self.y = torch.nn.Parameter(
            torch.tensor([y], dtype=torch.float32, requires_grad=True)
        )
