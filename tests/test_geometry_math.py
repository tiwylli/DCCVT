import unittest

import torch

from dccvt.geometry import compute_circumcenters, compute_cvt_loss_from_clipped_vertices
from dccvt.sdf_gradients import compute_sdf_gradients_sites_tets


class GeometryMathTests(unittest.TestCase):
    def test_circumcenters_are_equidistant_and_differentiable(self):
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.2, 0.8, 1.1],
            ],
            device="cuda",
            requires_grad=True,
        )
        tets = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")

        centers = compute_circumcenters(sites, tets)
        distances = (sites[tets] - centers[:, None]).norm(dim=2)

        torch.testing.assert_close(distances, distances[:, :1].expand_as(distances), rtol=1e-5, atol=1e-5)
        centers.square().mean().backward()
        self.assertTrue(torch.isfinite(sites.grad).all())

    def test_affine_sdf_has_exact_constant_gradient(self):
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device="cuda",
            requires_grad=True,
        )
        tets = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")
        expected = torch.tensor([0.5, -1.25, 2.0], device="cuda")
        sdf = (sites.detach() @ expected + 0.3).requires_grad_()

        site_gradients, tet_gradients, weights = compute_sdf_gradients_sites_tets(sites, sdf, tets)

        torch.testing.assert_close(tet_gradients, expected.expand_as(tet_gradients), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(site_gradients, expected.expand_as(site_gradients), rtol=1e-5, atol=1e-5)
        (site_gradients.square().mean() + weights.square().mean()).backward()
        self.assertTrue(torch.isfinite(sites.grad).all())
        self.assertTrue(torch.isfinite(sdf.grad).all())

    def test_cvt_loss_ignores_nonfinite_voronoi_vertices(self):
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device="cuda",
            requires_grad=True,
        )
        tets = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")
        vertices = torch.tensor(
            [
                [0.25, 0.25, 0.25],
                [float("nan"), 0.5, 0.5],
            ],
            device="cuda",
        )

        loss = compute_cvt_loss_from_clipped_vertices(sites, tets, vertices)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(sites.grad).all())


if __name__ == "__main__":
    unittest.main()
