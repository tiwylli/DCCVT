"""Evaluate PoNQ-style ABC mesh predictions."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import eval_Template


DEFAULT_GT_DIR = "/data/nmaruani/DATASETS/ABC/"
DEFAULT_ORDER_FILE = "src/eval/abc_ordered.txt"
DEFAULT_NON_WATERTIGHT_FILE = "src/eval/not_watertight_ABC_test.txt"


def eval_normalization(x):
    return x / 2.0


def _read_ids(path):
    ids = []
    with Path(path).open("r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.extend(Path(part).stem for part in line.replace(",", " ").split())
    return ids


def _legacy_items(pred_dir, gt_dir, seed, sample_count):
    non_watertight = set(_read_ids(DEFAULT_NON_WATERTIGHT_FILE))
    ordered = _read_ids(DEFAULT_ORDER_FILE)
    validation = ordered[int(len(ordered) * 0.8):]
    items = []
    for index, model_id in enumerate(validation):
        if model_id in non_watertight:
            continue
        items.append(
            (
                index,
                str(Path(gt_dir) / model_id / "model.obj"),
                str(Path(pred_dir) / "test_{}.obj".format(index)),
                seed,
                sample_count,
            )
        )
    return items


def _manifest_items(
    pred_dir,
    gt_dir,
    names_file,
    prediction_pattern,
    seed,
    sample_count,
):
    items = []
    for index, model_id in enumerate(_read_ids(names_file)):
        prediction = prediction_pattern.format(id=model_id, index=index)
        items.append(
            (
                index,
                str(Path(gt_dir) / model_id / "model.obj"),
                str(Path(pred_dir) / prediction),
                seed,
                sample_count,
            )
        )
    return items


def _validate_inputs(items):
    missing_gt = [item[1] for item in items if not Path(item[1]).exists()]
    missing_pred = [item[2] for item in items if not Path(item[2]).exists()]
    if missing_gt or missing_pred:
        messages = []
        if missing_gt:
            messages.append("missing ground truth ({}): {}".format(len(missing_gt), missing_gt[:20]))
        if missing_pred:
            messages.append("missing predictions ({}): {}".format(len(missing_pred), missing_pred[:20]))
        raise FileNotFoundError("; ".join(messages))


def _default_output(pred_dir):
    name = Path(pred_dir).resolve().name
    return Path("src/eval/results/results_{}".format(name))


def _evaluate_item(item):
    index, gt_path, pred_path, seed, sample_count = item
    if seed is not None:
        np.random.seed(int(seed) + int(index))
    eval_Template.sample_num = int(sample_count)
    return eval_Template.get_cd_f1_nc(
        (index, gt_path, pred_path),
        1,
        eval_normalization,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="PoNQ metrics on the ABC dataset")
    parser.add_argument("pred_dir", help="Path to the prediction mesh folder.")
    parser.add_argument(
        "-gt_dir",
        "--gt-dir",
        dest="gt_dir",
        default=DEFAULT_GT_DIR,
        help="ABC ground-truth OBJ root containing one folder per model ID.",
    )
    parser.add_argument(
        "--names-file",
        default=None,
        help="Optional explicit shape manifest. Without it, preserve the original ABC split behavior.",
    )
    parser.add_argument(
        "--prediction-pattern",
        default="{id}.obj",
        help="Prediction filename pattern for --names-file; supports {id} and {index}.",
    )
    parser.add_argument("--output", default=None, help="Output prefix without an extension.")
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")
    if args.names_file:
        items = _manifest_items(
            args.pred_dir,
            args.gt_dir,
            args.names_file,
            args.prediction_pattern,
            args.seed,
            args.sample_count,
        )
    else:
        items = _legacy_items(
            args.pred_dir,
            args.gt_dir,
            args.seed,
            args.sample_count,
        )
    _validate_inputs(items)

    values = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_evaluate_item)(item) for item in items
    )
    values = np.asarray(values)
    output = Path(args.output) if args.output else _default_output(args.pred_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), values)

    mean_scores = values.mean(0)
    summary = {
        "shape_count": int(values.shape[0]),
        "cd1": float(mean_scores[1]),
        "chamfer": float(mean_scores[2]),
        "f1": float(mean_scores[3]),
        "normal_consistency": float(mean_scores[4]),
        "edge_chamfer": float(mean_scores[5]),
        "edge_f1": float(mean_scores[6]),
        "sample_count": args.sample_count,
        "seed": args.seed,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print("CD (x 1e-5), F1, NC, ECD, EF1")
    print(
        "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(
            output.name,
            mean_scores[2] * 1e5,
            mean_scores[3],
            mean_scores[4],
            mean_scores[5] * 1e2,
            mean_scores[6],
        )
    )


if __name__ == "__main__":
    main()
