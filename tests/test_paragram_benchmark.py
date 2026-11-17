import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("DCCVT_DEVICE", "cpu")

from dccvt.paragram_benchmark import (
    DatasetSpec,
    amdahl_speedup,
    benchmark_backend,
    canonicalize_edges,
    compare_edges,
    csr_to_edges,
    generate_points,
    status_histogram,
    tetrahedra_to_edges,
    validate_csr,
    write_json,
)


class ParagramBenchmarkUnitTests(unittest.TestCase):
    @patch("dccvt.paragram_benchmark._paragram_once")
    def test_native_only_mode_is_forwarded_to_every_backend_call(self, run_once):
        sample = {
            "raw_wall_ms": 1.0,
            "raw_cuda_ms": 1.0,
            "solver_reported_ms": None,
            "end_to_end_ms": 1.0,
        }
        run_once.return_value = (sample, None)

        result, edges = benchmark_backend(
            "paragram",
            np.zeros((4, 3), dtype=np.float32),
            repetitions=2,
            warmups=1,
            common_output=False,
        )

        self.assertEqual(result["status"], "ok")
        self.assertIsNone(edges)
        self.assertEqual(run_once.call_count, 3)
        self.assertTrue(
            all(call.kwargs == {"common_output": False} for call in run_once.call_args_list)
        )

    def test_tetrahedra_to_edges_deduplicates_shared_edges(self):
        tetrahedra = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])
        edges = tetrahedra_to_edges(tetrahedra, num_points=5)

        expected = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
            ]
        )
        torch.testing.assert_close(edges, expected)

    def test_csr_to_edges_canonicalizes_direction_and_self_edges(self):
        adjacency = torch.tensor([1, 2, 0, 2, 1, 0, 1, 2], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 5, 8], dtype=torch.int32)

        edges = csr_to_edges(adjacency, offsets, num_points=3)

        torch.testing.assert_close(edges, torch.tensor([[0, 1], [0, 2], [1, 2]]))

    def test_validate_csr_rejects_invalid_offsets_and_indices(self):
        with self.assertRaisesRegex(ValueError, "span"):
            validate_csr(torch.tensor([1]), torch.tensor([0, 0, 0]), num_points=2)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            validate_csr(torch.tensor([2]), torch.tensor([0, 1, 1]), num_points=2)

    def test_canonicalize_edges_rejects_out_of_range_indices(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            canonicalize_edges(torch.tensor([[0, 3]]), num_points=3)

    def test_status_histogram_decodes_known_and_unknown_values(self):
        histogram = status_histogram(torch.tensor([0, 0, 1, 5, 9], dtype=torch.int32))

        self.assertEqual(
            histogram,
            {"success": 2, "plane_overflow": 1, "inconsistent_boundary": 1, "unknown_9": 1},
        )

    def test_compare_edges_reports_overlap_and_degree_error(self):
        reference = torch.tensor([[0, 1], [1, 2], [2, 3]])
        candidate = torch.tensor([[0, 1], [1, 2], [0, 3]])

        metrics = compare_edges(reference, candidate, num_points=4)

        self.assertEqual(metrics["common_edges"], 2)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["jaccard"], 0.5)
        self.assertEqual(metrics["degree_max_absolute_error"], 1)

    def test_synthetic_generation_is_deterministic(self):
        spec = DatasetSpec(name="clustered", kind="clustered", size=100, seed=7)
        first, first_metadata = generate_points(spec)
        second, second_metadata = generate_points(spec)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)

    def test_npz_generation_hashes_and_validates_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.npz"
            np.savez(path, sites=np.arange(15, dtype=np.float32).reshape(5, 3))

            points, metadata = generate_points(DatasetSpec(name="saved", kind="npz", path=str(path)))

        self.assertEqual(points.shape, (5, 3))
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_amdahl_projection_and_json_serialization(self):
        self.assertAlmostEqual(amdahl_speedup(2.0), 1 / (1 - 0.2385 + 0.2385 / 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json({"value": amdahl_speedup(2.0)}, path)
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        self.assertIsInstance(payload["value"], float)


def _cuda_smoke_available() -> bool:
    return (
        os.environ.get("RUN_PARAGRAM_CUDA_TESTS") == "1"
        and torch.cuda.is_available()
        and importlib.util.find_spec("paragram") is not None
        and importlib.util.find_spec("pygdel3d") is not None
    )


@unittest.skipUnless(
    _cuda_smoke_available(),
    "set RUN_PARAGRAM_CUDA_TESTS=1 with CUDA, paragram, and pygdel3d available",
)
class ParagramBenchmarkCudaSmokeTests(unittest.TestCase):
    def test_backends_produce_valid_adjacency_and_warm_start(self):
        import paragram
        import pygdel3d

        generator = torch.Generator(device="cuda").manual_seed(11)
        points = torch.rand((256, 3), generator=generator, device="cuda", dtype=torch.float32)
        diagram = paragram.voronoi_diagram(points)
        edges = csr_to_edges(diagram.adjacency, diagram.offsets, points.shape[0])
        self.assertGreater(edges.shape[0], 0)
        self.assertEqual(status_histogram(diagram.status), {"success": points.shape[0]})

        warm = paragram.voronoi_diagram(
            points,
            initial_guesses=diagram.adjacency,
            initial_guesses_offsets=diagram.offsets,
        )
        warm_edges = csr_to_edges(warm.adjacency, warm.offsets, points.shape[0])
        self.assertGreater(compare_edges(edges, warm_edges, points.shape[0])["f1"], 0.5)
        self.assertEqual(sum(status_histogram(warm.status).values()), points.shape[0])

        tetrahedra, _ = pygdel3d.triangulate(points.cpu().numpy())
        gdel_edges = tetrahedra_to_edges(torch.as_tensor(tetrahedra, device="cuda"), points.shape[0])
        self.assertGreater(gdel_edges.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
