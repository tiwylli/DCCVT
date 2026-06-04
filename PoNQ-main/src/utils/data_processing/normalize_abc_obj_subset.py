import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    triangles = []

    with path.open("r") as fin:
        for line in fin:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                triangles.append([
                    int(parts[1].split("/")[0]) - 1,
                    int(parts[2].split("/")[0]) - 1,
                    int(parts[3].split("/")[0]) - 1,
                ])

    if not vertices or not triangles:
        raise ValueError(f"OBJ has no vertices or triangles: {path}")

    return np.array(vertices, np.float32), np.array(triangles, np.int32)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (upper + lower) / 2
    scale = np.linalg.norm(upper - lower)
    if scale == 0:
        raise ValueError("OBJ bounding-box diagonal is zero")
    return (vertices - center) / scale


def write_obj(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    with path.open("w") as fout:
        for vertex in vertices:
            fout.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for triangle in triangles:
            fout.write(
                f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n"
            )


def read_model_ids(names_file: Path, limit: Optional[int]) -> list[str]:
    names = []
    with names_file.open("r") as fin:
        for line in fin:
            name = line.strip()
            if name:
                names.append(Path(name).stem)
    if limit is not None:
        names = names[:limit]
    return names


def source_obj_for_model(model_dir: Path) -> Path:
    candidates = sorted(
        path for path in model_dir.glob("*.obj") if path.name != "model.obj"
    )
    if not candidates:
        raise FileNotFoundError(f"No source OBJ found in {model_dir}")
    return candidates[0]


def normalize_one(input_dir: Path, model_id: str, skip_existing: bool) -> str:
    model_dir = input_dir / model_id
    out_path = model_dir / "model.obj"
    if skip_existing and out_path.exists():
        return "skipped"

    source_path = source_obj_for_model(model_dir)
    vertices, triangles = load_obj(source_path)
    write_obj(out_path, normalize_vertices(vertices), triangles)
    return "written"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize selected ABC raw OBJ folders into model.obj files."
    )
    parser.add_argument("--input-dir", required=True,
                        help="ABC raw OBJ root containing one folder per model ID.")
    parser.add_argument("--names-file", required=True,
                        help="File containing model IDs or .hdf5 names to normalize.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of names to process for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not rewrite existing model.obj files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    model_ids = read_model_ids(Path(args.names_file), args.limit)

    counts = {"written": 0, "skipped": 0}
    for model_id in tqdm(model_ids):
        result = normalize_one(input_dir, model_id, args.skip_existing)
        counts[result] += 1

    print(f"model ids: {len(model_ids)}")
    print(f"written: {counts['written']}")
    print(f"skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
