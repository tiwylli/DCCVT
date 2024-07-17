import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

# Define a function to plot and save the figure
def plot_and_save(
    epoch, s_i, s_j, s_k, X_g, Y_g, circle_center, radius, destination="images/"
):
    # check destination and make folder if not exists
    if not os.path.exists(destination):
        os.makedirs(destination)
    X, Y = compute_vertex(s_i, s_j, s_k)
    x_i, y_i = s_i.x, s_i.y
    x_j, y_j = s_j.x, s_j.y
    x_k, y_k = s_k.x, s_k.y
    # Calculate midpoints
    midpoint_ij_x, midpoint_ij_y = (x_i + x_j) / 2, (y_i + y_j) / 2
    midpoint_jk_x, midpoint_jk_y = (x_j + x_k) / 2, (y_j + y_k) / 2

    plt.figure()

    # Label sites
    plt.text(x_i.item(), y_i.item(), 's_i', fontsize=9, verticalalignment='bottom')
    plt.text(x_j.item(), y_j.item(), 's_j', fontsize=9, verticalalignment='bottom')
    plt.text(x_k.item(), y_k.item(), 's_k', fontsize=9, verticalalignment='bottom')
    plt.scatter(
        [x_i.item(), x_j.item(), x_k.item()],
        [y_i.item(), y_j.item(), y_k.item()],
        color="blue",
        label="Sites",
    )
    
    plt.scatter(X.item(), Y.item(), color="red", label="Computed Vertex")
    plt.scatter(X_g.item(), Y_g.item(), color="green", label="Ground Truth Vertex = Closest Point on Circle")
    # Plot midpoints
    plt.scatter([midpoint_ij_x.item(), midpoint_jk_x.item()], [midpoint_ij_y.item(), midpoint_jk_y.item()], color='purple', marker='x', label='Midpoints')
    
    circle = Circle(
        (circle_center[0].item(), circle_center[1].item()),
        radius,
        color="gray",
        fill=False,
        linestyle="--",
        label="Ground Truth Circle",
    )
    plt.gca().add_patch(circle)
    plt.legend()
    plt.title(f"Epoch {epoch}")
    plt.xlabel("X")
    plt.ylabel("Y")
    # check min and max values of sites
    plt.ylim(
        min(0, y_i.item(), y_j.item(), y_k.item()) * 1.1,
        max(y_i.item(), y_j.item(), y_k.item()) * 1.1,
    )
    plt.xlim(
        min(0, x_i.item(), x_j.item(), x_k.item()) * 1.1,
        max(x_i.item(), x_j.item(), x_k.item()) * 1.1,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.savefig(f"{destination}{epoch}.png")
    plt.close()


# Define a function to compute the vertex coordinates
def compute_vertex(s_i, s_j, s_k):
    x_i, y_i = s_i.x, s_i.y
    x_j, y_j = s_j.x, s_j.y
    x_k, y_k = s_k.x, s_k.y
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

    return X, Y

def midpoint_loss(s_i, s_j, s_k, circle_center, radius):
    # Calculate midpoints
    m_ij = ((s_i.x + s_j.x) / 2, (s_i.y + s_j.y) / 2)
    m_jk = ((s_j.x + s_k.x) / 2, (s_j.y + s_k.y) / 2)
    
    # Circle center and radius
    h, k = circle_center
    r = radius
    
    # Loss for each midpoint to be on the circle
    loss_m_ij = ((m_ij[0] - h)**2 + (m_ij[1] - k)**2 - r**2)**2
    loss_m_jk = ((m_jk[0] - h)**2 + (m_jk[1] - k)**2 - r**2)**2
    
    # Total midpoint loss
    total_loss = loss_m_ij + loss_m_jk
    
    return total_loss

# Define the Site class
class Site:
    def __init__(self, x, y):
        self.x = torch.nn.Parameter(
            torch.tensor([x], dtype=torch.float32, requires_grad=True)
        )
        self.y = torch.nn.Parameter(
            torch.tensor([y], dtype=torch.float32, requires_grad=True)
        )
