"""Alpha-shape SDF helper for complex meshes."""

from collections import defaultdict

import gudhi
import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors

from dccvt.device import device


def complex_alpha_sdf(mnfld_points: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
    def alpha_shape_3d(points: np.ndarray, alpha: float):
        """
        Build a 3D alpha shape mesh from points using Gudhi.
        alpha: radius parameter (not squared). Smaller -> tighter, more concave; too small -> holes/missing parts.
        Returns V,F for a triangle surface mesh.
        """
        ac = gudhi.AlphaComplex(points=points)
        st = ac.create_simplex_tree(max_alpha_square=alpha * alpha)

        # Collect tetrahedra (3-simplices) and triangles (2-simplices) in the complex
        tets = []
        tris = []
        for simplex, filt in st.get_skeleton(3):
            if len(simplex) == 4:
                tets.append(tuple(sorted(simplex)))
            elif len(simplex) == 3:
                tris.append(tuple(sorted(simplex)))

        # Count how many tetrahedra incident to each triangle; boundary triangles have <=1 incident tet
        tri_incidence = defaultdict(int)
        for tet in tets:
            a, b, c, d = tet
            faces = [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
            for f in faces:
                tri_incidence[tuple(sorted(f))] += 1

        boundary_tris = []
        for tri in tris:
            if tri_incidence.get(tri, 0) <= 1:
                boundary_tris.append(tri)

        V = points.copy()
        F = np.array(boundary_tris, dtype=int)

        # Clean up with trimesh (remove degenerates, unify winding, fill tiny holes if needed)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh, True)  # consistent winding + outward if possible
        trimesh.repair.fix_inversion(mesh, True)  # resolves inside-out components
        print(trimesh.repair.broken_faces(mesh))
        assert mesh.is_watertight, "Alpha mesh not watertight; tune alpha or repair."
        assert mesh.is_winding_consistent, "Inconsistent winding; fix_normals should help."
        return mesh

    def pick_alpha(points, k=8, quantile=0.9, magnitude=15.0):
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        dists, _ = nbrs.kneighbors(points)
        # ignore the zero distance to self at column 0 by slicing from 1:
        scale = np.quantile(dists[:, 1:].mean(axis=1), quantile)
        # for some reasons at 1.5 mesh is not watertight and trimesh cant fix it
        # so for safety we multiply by 15
        return magnitude * scale

    alpha = pick_alpha(mnfld_points.squeeze(0).detach().cpu().numpy())  # or set manually
    mesh = alpha_shape_3d(mnfld_points.squeeze(0).detach().cpu().numpy(), alpha)
    S = -trimesh.proximity.signed_distance(mesh, sites.detach().cpu().numpy())
    sdf0 = torch.from_numpy(S).to(device, dtype=torch.float32).requires_grad_()
    return sdf0

