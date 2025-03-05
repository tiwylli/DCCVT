# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from matplotlib.path import Path
import torch
import os

class BasicShape2D(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled on Jets + samples on a grid with their distance to the surface
    def __init__(self, n_points, n_samples=128, res=128, sample_type='grid', sampling_std=0.005,
                 grid_range=1.2):


        self.grid_range = grid_range
        self.n_points = n_points
        self.n_samples = n_samples
        self.grid_res = res
        self.sample_type = sample_type #grid | gaussian | combined
        self.sampling_std = sampling_std
        # Generate shape

        self.points = self.get_mnfld_points()

        # generate grid points and find distance to closest point on the line
        x, y = np.linspace(-grid_range, grid_range, self.grid_res), np.linspace(-grid_range, grid_range, self.grid_res)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.ravel(), yy.ravel()
        self.grid_points = np.stack([xx, yy], axis=1).astype('f')
        self.nonmnfld_points = self.get_nonmnfld_points()

        # Compute gt mnfld normals
        self.mnfld_n = self.get_mnfld_n()

        self.grid_dist, self.grid_n = self.get_points_distances_and_normals(self.grid_points)
        self.nonmnfld_dist, self.nonmnfld_n = self.get_points_distances_and_normals(self.nonmnfld_points)
        self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])

        self.point_idxs = np.arange(self.points.shape[1])
        self.grid_points_idxs = np.arange(self.grid_points.shape[0])
        self.nonmnfld_points_idxs = np.arange(self.nonmnfld_points.shape[0])
        self.sample_probs = np.ones_like(self.grid_points_idxs) / self.grid_points.shape[0]

        self.generate_batch_indices()

    @abstractmethod
    def get_mnfld_points(self):
        # implement a function that returns points on the manifold
        pass

    @abstractmethod
    def get_mnfld_n(self):
        #implement a function that returns normal vectors for points on the manifold
        pass

    @abstractmethod
    def get_points_distances_and_normals(self, points):
        # implement a function that computes the distance and normal vectors of nonmanifold points.
        # default implementation finds the nearest neighbor and return its normal and the distance to it.
        # which is a coarse approxiamation

        distances = []
        normals = []
        # compute distance and normal (general case)
        for i, point_cloud in enumerate(self.points):
            kdtree = spatial.cKDTree(point_cloud)
            distances, nn_idx = kdtree.query(points, k=1)
            signs = np.sign(np.einsum('ij,ij->i', points - point_cloud[nn_idx],
                                      self.mnfld_n[i, nn_idx]))
            normals.append(self.mnfld_n[i, nn_idx])
            distances.append(signs*distances)

        distances = np.stack(distances).astype('f')
        normals = np.stack(normals).astype('f')
        return distances, normals

    def get_grid_divergence(self):
        # 2D implementation
        n_img = np.reshape(self.grid_n, [self.grid_res, self.grid_res, -1])
        frac_45 = 1./np.sqrt(2)
        filter = np.array([[[frac_45, -frac_45], [1., 0.], [frac_45, frac_45]],
                           [[0., -1.], [0., 0.], [0., 1.]],
                  [[-frac_45, -frac_45], [ -1., 0.], [-frac_45, frac_45]]])  # [y, x]
        padding = self.get_padding(n_img, filter, strides=[1, 1])
        n_img = torch.nn.functional.pad(torch.tensor(n_img, dtype=torch.float32), padding)
        div_img = torch.nn.functional.conv2d(n_img.permute([2, 0, 1]).unsqueeze(0),
                                             torch.tensor(filter, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0),
                                             ).squeeze().numpy()

        return div_img.flatten()


    def get_offgrid_divergnce(self, off_grid_points, method='nn'):
        #TODO implement interpulation method?
        if method == 'nn':
            # find the nearest grid point and return its divergence
            kdtree = spatial.cKDTree(self.grid_points)
            _, nn_idx = kdtree.query(off_grid_points, k=1)
        else:
            raise Warning('unsupported offgrid div computeation method')
        return self.grid_div[nn_idx]


    def get_padding(self, img, filter, strides=[1, 1]):
        # from https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/3
        in_height, in_width, _ = img.shape
        filter_height, filter_width, _ = filter.shape
        # The total padding applied along the height and width is computed as:
        if (in_height % strides[0] == 0):
            pad_along_height = max(filter_height - strides[0], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(filter_width - strides[1], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[1]), 0)

        # Finally, the padding on the top, bottom, left and right are:
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return (0, 0, pad_left, pad_right, pad_top, pad_bottom)


    def get_nonmnfld_points(self):
        if self.sample_type == 'grid':
            nonmnfld_points = self.grid_points
        elif self.sample_type == 'uniform':
            nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                                size=(self.grid_res * self.grid_res , 2)).astype(np.float32)
        elif self.sample_type == 'gaussian':
            nonmnfld_points = self.sample_gaussian_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res)
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
        elif self.sample_type == 'combined':
            nonmnfld_points1 = self.sample_gaussian_noise_around_shape()
            nonmnfld_points2 = self.grid_points
            idx1 = np.random.choice(range(nonmnfld_points1.shape[1]), int(np.ceil(self.grid_res * self.grid_res / 2)))
            idx2 = np.random.choice(range(nonmnfld_points2.shape[0]), int(np.floor(self.grid_res * self.grid_res / 2)))
            sample_idx = np.random.choice(range(nonmnfld_points1.shape[0]), int(np.ceil(self.grid_res * self.grid_res / 2)))

            nonmnfld_points = np.concatenate([nonmnfld_points1[sample_idx, idx1], nonmnfld_points2[idx2]], axis=0)
        else:
            raise Warning("Unsupported non manfold sampling type {}".format(self.sample_type))
        return nonmnfld_points

    def sample_gaussian_noise_around_shape(self):
        n_noisy_points = int(np.round(self.grid_res * self.grid_res / self.n_points))
        noise = np.random.multivariate_normal([0, 0], [[self.sampling_std, 0], [0, self.sampling_std]],
                                              size=(self.n_samples, self.n_points, n_noisy_points)).astype(np.float32)
        nonmnfld_points = np.tile(self.points[:, :, None, :], [1, 1, n_noisy_points, 1]) + noise
        nonmnfld_points = nonmnfld_points.reshape([nonmnfld_points.shape[0], -1, nonmnfld_points.shape[-1]])
        return nonmnfld_points

    def generate_batch_indices(self):
        mnfld_idx = []
        nonmnfld_idx = []
        for i in range(self.n_samples):
            mnfld_idx.append(np.random.choice(self.point_idxs, self.n_points))
            nonmnfld_idx.append(np.random.choice(self.nonmnfld_points_idxs, self.n_points))
        self.mnfld_idx = np.array(mnfld_idx)
        self.nonmnfld_idx = np.array(nonmnfld_idx)

    def __getitem__(self, index):
        nonmnfld_idx = self.nonmnfld_idx[index]
        mnfld_idx = self.mnfld_idx[index]
        if self.nonmnfld_dist is not None:
            nonmnfld_dist = self.nonmnfld_dist[nonmnfld_idx]
        else:
            nonmnfld_dist = torch.tensor(0)

        return {'points' : self.points[index, mnfld_idx, :],  'mnfld_n': self.mnfld_n[index, mnfld_idx, :],  \
                'nonmnfld_dist': nonmnfld_dist, 'nonmnfld_n': self.nonmnfld_n[nonmnfld_idx],
                'nonmnfld_points': self.nonmnfld_points[nonmnfld_idx],
                }

    def __len__(self):
        return self.n_samples

class Circle(BasicShape2D):
    def __init__(self, *args, r=0.5):
        self.r = r
        BasicShape2D.__init__(self, *args)


    def get_mnfld_points(self):
        theta = np.random.uniform(0, 2*np.pi, size=(self.n_samples, self.n_points)).astype('f')
        x = self.r * np.sin(theta)
        y = self.r * np.cos(theta)

        points = np.stack([x, y], axis=2)
        return points

    def get_mnfld_n(self):
        return self.points / np.linalg.norm(self.points, axis=2, keepdims=True)


    def get_points_distances_and_normals(self, points):
        point_dist = np.linalg.norm(points, axis=1, keepdims=True)
        distances = point_dist - self.r
        normals = points / point_dist
        return distances, normals

class Polygon(BasicShape2D):
    def __init__(self, *args, vertices=[], line_sample_type='uniform'):

        # vertices: x,y points specifying the polygon
        self.vertices = np.array(vertices)
        self.lines = self.get_line_props()
        self.line_sample_type = line_sample_type
        BasicShape2D.__init__(self, *args)

    def get_mnfld_points(self):
        # sample points on the lines
        n_points_to_sample = self.n_points - len(self.vertices)
        #TODO:
        n_points_to_sample = n_points_to_sample if n_points_to_sample > 0 else 1
        if n_points_to_sample < 0:
            raise Warning("Fewer points to sample than polygon vertices. Please change the number of points")
        sample_prob = self.lines['line_length'] / np.sum(self.lines['line_length'])
        points_per_segment = np.floor(n_points_to_sample * sample_prob).astype(np.int32)
        points_leftover = int(n_points_to_sample - points_per_segment.sum())
        if not points_leftover == 0:
            for j in range(points_leftover):
                actual_prob = points_per_segment / points_per_segment.sum()
                prob_diff = sample_prob - actual_prob
                add_idx = np.argmax(prob_diff)
                points_per_segment[add_idx] = points_per_segment[add_idx] + 1

        points = []
        self.point_normal = []
        for point_idx, point in enumerate(self.vertices):
            l1_idx = len(self.vertices) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines['nl'][l1_idx] + self.lines['nl'][l2_idx]
            self.point_normal.append(n / np.linalg.norm(n))
            points.append(point)
        points = np.repeat(np.array(points)[None, :], self.n_samples, axis=0)
        self.point_normal = np.repeat(np.array(self.point_normal)[None, :], self.n_samples, axis=0)

        for line_idx in range(len(self.lines['A'])):
            if self.line_sample_type == 'uniform':
                t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
                t = np.repeat(t[None, :], self.n_samples, axis=0)
            else:
                t = np.random.uniform(0, 1, [self.n_samples, points_per_segment[line_idx]])
            p1 = np.array(self.vertices[self.lines['start_idx'][line_idx]])
            p2 = np.array(self.vertices[self.lines['end_idx'][line_idx]])
            points = np.concatenate([points, p1 + t[:, :, None]*(p2 - p1)], axis=1)
            self.point_normal = np.concatenate([self.point_normal,
                                                np.tile(self.lines['nl'][line_idx][None, None, :],
                                                        [self.n_samples, points_per_segment[line_idx], 1])], axis=1)
        return points.astype('f')

    def get_mnfld_n(self):
        return self.point_normal


    def get_points_distances_and_normals(self, points):
        # iterate over all the lines and  finds the minimum distance between all points and line segments
        # good explenation ref : https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
        n_grid_points = len(points)
        p1x = np.vstack(self.vertices[self.lines['start_idx']][:, 0])
        p1y = np.vstack(self.vertices[self.lines['start_idx']][:, 1])
        p2x = np.vstack(self.vertices[self.lines['end_idx']][:, 0])
        p2y = np.vstack(self.vertices[self.lines['end_idx']][:, 1])
        p1p2 = np.array(self.lines['direction'])
        px = points[:, 0]
        py = points[:, 1]
        pp1 = np.vstack([px - np.tile(p1x, [1, 1, n_grid_points]), py - np.tile(p1y, [1, 1, n_grid_points])])
        pp2 = np.vstack([px - np.tile(p2x, [1, 1, n_grid_points]), py - np.tile(p2y, [1, 1, n_grid_points])])

        r = (p1p2[:, 0, None] * pp1[0, :, :] + p1p2[:, 1, None] * pp1[1, :, :]) / np.array(self.lines['line_length'])[:, None]

        d1 = np.linalg.norm(pp1, axis=0)
        d2 = np.linalg.norm(pp2, axis=0)
        dp = np.sqrt(np.square(d1) - np.square(r * np.array(self.lines['line_length'])[:, None]))
        d = np.where(r < 0, d1, np.where(r > 1, d2, dp))
        distances = np.min(d, axis=0)
        idx = np.argmin(d, axis=0)
        # compute normal vector
        polygon_path = Path(self.vertices)
        point_in_polygon = polygon_path.contains_points(points)
        point_sign = np.where(point_in_polygon, -1, 1)

        n = np.where(r < 0, pp1, np.where(r > 1, pp2, point_sign *
                                          np.tile(np.array(self.lines['nl']).transpose()[:, :, None],
                                          [1, 1, n_grid_points])))
        normals = np.take_along_axis(n, idx[None, None, :], axis=1).squeeze().transpose()
        normals = point_sign[:, None] * normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = point_sign * distances

        return distances, normals


    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [], 'direction': []}
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices)-1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [end_point[0] - start_point[0], end_point[1] - start_point[1]] / line_length
            nl = [A, B]
            nl = nl / np.linalg.norm(nl)
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines
    
def get_bunny_points():
    # load bunny points
    bunny = np.load('/home/wylliam/dev/Kyushu_experiments/utils/2d_bunny_surface_pruned.npy')
    # normalize bunny between -1 and 1
    bunny = bunny - np.mean(bunny, axis=0)
    bunny = bunny / np.max(np.abs(bunny))
    return bunny

def get2D_dataset(*args, shape_type='circle'):

    if shape_type == 'circle':
        out_shape = Circle(*args)
    elif shape_type == 'L':
        out_shape = Polygon(*args, vertices=[[0., 0.], [0.5, 0.], [0.5, -0.5],
                                 [-0.5, -0.5], [-0.5, 0.5], [0, 0.5]])
    elif shape_type == 'square':
        out_shape = Polygon(*args, vertices=[[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    elif shape_type == 'bunny':
        vertices = get_bunny_points()
        out_shape = Polygon(*args, vertices=vertices)
    else:
        raise Warning("Unsupportaed shape")

    return out_shape


def plot_sdf_indicator(points, x_grid, y_grid, sdf, title_text='', show_ax=False, output_path='./'):
    points = np.concatenate([points, points[0, :][None, :]], axis=0)
    indicator = -np.ones_like(sdf)
    indicator[sdf > 0] = 1

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True,  showline=True, linewidth=2, linecolor='black', showticklabels=False),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True, showline=True,linewidth=2, linecolor='black', showticklabels=False),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red')),
                       paper_bgcolor='rgb(255,255,255)',
                       plot_bgcolor='rgb(255, 255, 255)'
                       )


    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=points[:, 0], y=points[:, 1],
                                mode='markers', marker=dict(size=52, color='rgb(0, 0, 0)'))
    fig = go.Figure(data=clean_points_scatter, layout=layout)
    fig.write_image(os.path.join(output_path, "points.png"))
    fig.data = []
    line_plot = go.Scatter(x=points[:, 0], y=points[:, 1],
                                mode='markers+lines', marker=dict(size=52, color='rgb(0, 0, 0)'),
                                      fill="toself", fillcolor='rgb(255,255,255)',
                                      line=dict(width=6, color='rgb(0, 0, 0)'))

    fig.add_traces(line_plot)
    fig.write_image(os.path.join(output_path, "lines.png"))
    fig.data = []

    sdf_trace = go.Contour(x=x_grid, y=y_grid, z=sdf, colorscale='Geyser',
               contours=dict(start=-1, end=1, size=0.025), showscale=False)
    sdf_zls_trace = go.Contour(x=x_grid, y=y_grid, z=sdf,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([sdf_trace, sdf_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "sdf.png"))
    fig.data = []

    indicator_trace = go.Contour(x=x_grid, y=y_grid, z=indicator, colorscale='Geyser',
               contours=dict(start=-1, end=1, size=0.025), showscale=False)
    indicator_zls_trace = go.Contour(x=x_grid, y=y_grid, z=indicator,
                           contours=dict(start=0, end=0, coloring='lines'),
                           line=dict(width=5),
                           showscale=False,
                           colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([indicator_trace, indicator_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "indicator.png"))

    # fig = go.Figure(data=traces, layout=layout)
    # fig.show()

if __name__ == "__main__":
    np.random.seed(0)
    shape_type = 'bunny'
    res = 128 # has to be even
    example_idx = 0
    sample_type = 'grid'
    n_samples = 1
    n_points = 100
    dataset = get2D_dataset(n_points, n_samples, res, sample_type, 0.005, shape_type=shape_type)  # BasicShape2D(100, 20, res=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)
    data = next(iter(dataloader))
    clean_points_gt = data['points'][example_idx].detach().cpu().numpy()
    n_gt = data['mnfld_n'][example_idx].detach().cpu().numpy()
    nonmnfld_points = data['nonmnfld_points'][example_idx].detach().cpu().numpy()
    grid_normals = dataset.grid_n

    plot_sdf_indicator(dataset.vertices, dataset.grid_points[:, 0], dataset.grid_points[:, 1],
                           dataset.dist_img.flatten(), title_text='', show_ax=False, output_path='./vis/') # plot sdf, indicator function, and points

    # vis.plot_shape_data(dataset.grid_points[:, 0], dataset.grid_points[:, 1], dataset.dist_img.flatten(),
    #                     clean_points_gt, n_gt=n_gt, show_ax=True, show_bar=True,
    #                     title_text='', colorscale='Geyser', nonmnfld_points=nonmnfld_points, divergence=None,
    #                     grid_normals=grid_normals) # plot shape, sdf and other data

    # vis.plot_paper_teaser_images(dataset.grid_points[:, 0], dataset.grid_points[:, 1] ,dataset.dist_img.flatten(),
    #                              clean_points_gt, grid_normals) # plot images for DiGS paper teaser image
    
