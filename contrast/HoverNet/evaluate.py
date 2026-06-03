"""Evaluate HoVer-Net CoNSeP predictions against MCSpatNet point annotations.

Matching rule (mask contains GT point):
  - Predicted instance mask covers an unmatched GT point with the same class -> TP
  - Predicted instance with no matching GT point in mask -> FP
  - GT point never covered by any prediction -> FN
  - Detection F1 ignores class labels (any GT point inside mask counts as TP)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage import io
from skimage.draw import polygon as draw_polygon

SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_EXP_DIR = (SCRIPT_DIR / "../exp/exp27_HoverNet_consep_e100_new_json").resolve()
# DEFAULT_GT_ROOT = (SCRIPT_DIR / "../data/CoNSeP_MCSpatNet_point_1000x1000/gt_custom").resolve()
# DEFAULT_IMG_ROOT = (SCRIPT_DIR / "../data/CoNSeP_MCSpatNet_point_1000x1000/images").resolve()
# DEFAULT_TYPE_INFO = SCRIPT_DIR / "type_info_consep.json"
DEFAULT_EXP_DIR = (SCRIPT_DIR / "../exp/exp28_HoverNet_monusac_e100").resolve()
DEFAULT_GT_ROOT = (SCRIPT_DIR / "../data/MoNuSAC_MCSpatNet_point/test/gt_custom").resolve()
DEFAULT_IMG_ROOT = (SCRIPT_DIR / "../data/MoNuSAC_MCSpatNet_point/test/images").resolve()
DEFAULT_TYPE_INFO = SCRIPT_DIR / "type_info_monusac.json"

CLASS_IDS = (1, 2)
SQUARE_RADIUS = 2  # 5x5 square: center +/- 2 pixels


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def f1(self) -> float:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 1.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 1.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 1.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 1.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": self.f1(),
        }


@dataclass
class ImageEvalResult:
    image_name: str
    per_class: dict[int, Counts] = field(default_factory=dict)
    detection: Counts = field(default_factory=Counts)


def load_type_info(type_info_path: Path) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    with open(type_info_path, encoding="utf-8") as file:
        payload = json.load(file)
    names: dict[int, str] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    for key, value in payload.items():
        class_id = int(key)
        if class_id == 0:
            continue
        names[class_id] = value[0]
        colors[class_id] = tuple(value[1])
    return names, colors


def list_eval_stems(gt_root: Path, json_dir: Path) -> list[str]:
    stems: list[str] = []
    for gt_path in sorted(gt_root.glob("*_gt_dots.npy")):
        stem = gt_path.name[: -len("_gt_dots.npy")]
        json_path = json_dir / f"{stem}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing prediction json for {stem}: {json_path}")
        stems.append(stem)
    if not stems:
        raise FileNotFoundError(f"No *_gt_dots.npy found in {gt_root}")
    return stems

def normalize_gt_dots(gt_dots: np.ndarray) -> np.ndarray:
    if gt_dots.ndim != 3:
        raise ValueError(f"gt_dots must be 3D, got shape {gt_dots.shape}")
    if gt_dots.shape[2] == 4:
        # CoNSeP: [background, inflammatory, epithelial, stroma]
        return gt_dots[:, :, 1:4].astype(np.uint8)
    if gt_dots.shape[2] == 3:
        # MoNuSAC: [background, epithelial, lymphocyte]，通道 0 为背景且恒为 0
        if gt_dots[:, :, 0].max() == 0:
            return gt_dots[:, :, 1:].astype(np.uint8)
        # CoNSeP 等：3 个类别通道直接对应 class 1~3
        return gt_dots.astype(np.uint8)
    if gt_dots.shape[2] == 2:
        return gt_dots.astype(np.uint8)
    raise ValueError(f"gt_dots channel count must be 2, 3 or 4, got {gt_dots.shape[2]}")


def extract_gt_points(gt_dots: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    points: dict[int, list[tuple[int, int]]] = {class_id: [] for class_id in CLASS_IDS}
    for class_id in CLASS_IDS:
        rows, cols = np.where(gt_dots[:, :, class_id - 1] > 0)
        points[class_id] = list(zip(rows.tolist(), cols.tolist()))
    return points


def load_instances(
    json_path: Path,
    target_shape: tuple[int, int],
) -> list[tuple[int, np.ndarray]]:
    with open(json_path, encoding="utf-8") as file:
        payload = json.load(file)
    instances = payload.get("nuc", {})
    if not instances:
        return []

    height, width = target_shape
    parsed: list[tuple[int, np.ndarray]] = []
    for info in instances.values():
        pred_type = int(info["type"])
        contour = np.asarray(info["contour"], dtype=np.float64)
        xs = np.clip(np.round(contour[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(contour[:, 1]).astype(int), 0, height - 1)
        rows, cols = draw_polygon(ys, xs, shape=(height, width))
        mask = np.zeros((height, width), dtype=bool)
        mask[rows, cols] = True
        parsed.append((pred_type, mask))
    return parsed


def _match_point_in_mask(
    mask: np.ndarray,
    candidates: list[tuple[int, int, int, int]],
    matched: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for class_id, point_idx, row, col in candidates:
        key = (class_id, point_idx)
        if key in matched:
            continue
        if mask[row, col]:
            return key
    return None


def evaluate_image(
    json_path: Path,
    gt_path: Path,
) -> ImageEvalResult:
    gt_dots = normalize_gt_dots(np.load(gt_path, allow_pickle=True))
    gt_points = extract_gt_points(gt_dots)
    image_name = gt_path.name.replace("_gt_dots.npy", ".png")

    target_shape = gt_dots.shape[:2]
    instances = load_instances(json_path, target_shape)

    per_class = {class_id: Counts() for class_id in CLASS_IDS}
    detection = Counts()

    for class_id in CLASS_IDS:
        class_points = gt_points[class_id]
        matched: set[int] = set()
        for pred_type, mask in instances:
            if pred_type != class_id:
                continue
            candidates = [
                (class_id, idx, row, col)
                for idx, (row, col) in enumerate(class_points)
                if idx not in matched
            ]
            hit_idx = None
            for class_id_, point_idx, row, col in candidates:
                if mask[row, col]:
                    hit_idx = point_idx
                    break
            if hit_idx is not None:
                per_class[class_id].tp += 1
                matched.add(hit_idx)
            else:
                per_class[class_id].fp += 1
        per_class[class_id].fn = len(class_points) - len(matched)

    all_candidates = [
        (class_id, idx, row, col)
        for class_id in CLASS_IDS
        for idx, (row, col) in enumerate(gt_points[class_id])
    ]
    matched_detection: set[tuple[int, int]] = set()
    for _, mask in instances:
        key = _match_point_in_mask(mask, all_candidates, matched_detection)
        if key is not None:
            detection.tp += 1
            matched_detection.add(key)
        else:
            detection.fp += 1
    detection.fn = len(all_candidates) - len(matched_detection)

    return ImageEvalResult(image_name=image_name, per_class=per_class, detection=detection)


def draw_centroid_squares(
    image_shape: tuple[int, int],
    instances_payload: dict,
    type_colors: dict[int, tuple[int, int, int]],
    background: np.ndarray | None = None,
) -> np.ndarray:
    if background is not None and background.shape[:2] == image_shape:
        canvas = background.copy()
        if canvas.ndim == 2:
            canvas = np.stack([canvas] * 3, axis=-1)
        if canvas.shape[2] == 4:
            canvas = canvas[:, :, :3]
    else:
        canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    height, width = image_shape
    for info in instances_payload.values():
        class_id = int(info["type"])
        color = type_colors.get(class_id, (255, 255, 255))
        centroid_x, centroid_y = info["centroid"]
        center_row = int(round(centroid_y))
        center_col = int(round(centroid_x))
        row_start = max(0, center_row - SQUARE_RADIUS)
        row_end = min(height, center_row + SQUARE_RADIUS + 1)
        col_start = max(0, center_col - SQUARE_RADIUS)
        col_end = min(width, center_col + SQUARE_RADIUS + 1)
        canvas[row_start:row_end, col_start:col_end] = color
    return canvas.astype(np.uint8)


def format_metrics_text(
    class_names: dict[int, str],
    per_class_total: dict[int, Counts],
    detection_total: Counts,
    per_image_results: list[ImageEvalResult],
    vis_dir: Path,
) -> str:
    lines = ["=== CoNSeP HoverNet 点标注评估（掩码内含点）===", ""]
    class_f1_values: list[float] = []
    for class_id in CLASS_IDS:
        counts = per_class_total[class_id]
        metrics = counts.as_dict()
        class_f1_values.append(float(metrics["f1"]))
        lines.append(
            f"[{class_names[class_id]}] "
            f"TP={counts.tp} FP={counts.fp} FN={counts.fn} | "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )
    mean_f1 = float(np.mean(class_f1_values))
    lines.extend(["", f"Mean F1 ({len(class_names)} classes) = {mean_f1:.4f}", ""])
    det_metrics = detection_total.as_dict()
    lines.append(
        f"[Detection only] TP={detection_total.tp} FP={detection_total.fp} FN={detection_total.fn} | "
        f"P={det_metrics['precision']:.4f} R={det_metrics['recall']:.4f} F1={det_metrics['f1']:.4f}"
    )
    lines.extend(["", f"可视化目录: {vis_dir.resolve()}", "", "--- Per image ---"])
    for result in per_image_results:
        stem = result.image_name.replace(".png", "")
        per_class_f1 = " ".join(
            f"{class_names[class_id]}_f1={result.per_class[class_id].f1():.4f}"
            for class_id in CLASS_IDS
        )
        lines.append(
            f"{stem}: {per_class_f1} detection_f1={result.detection.f1():.4f}"
        )
    return "\n".join(lines) + "\n"


def run_evaluation(
    exp_dir: Path,
    gt_root: Path,
    img_root: Path,
    type_info_path: Path,
    save_visualizations: bool = True,
) -> dict:
    json_dir = exp_dir / "json"
    vis_dir = exp_dir / "vis_pred_centroids"
    metrics_path = exp_dir / "eval_point_metrics.txt"
    class_names, type_colors = load_type_info(type_info_path)

    per_class_total = {class_id: Counts() for class_id in CLASS_IDS}
    detection_total = Counts()
    per_image_results: list[ImageEvalResult] = []

    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for stem in list_eval_stems(gt_root, json_dir):
        json_path = json_dir / f"{stem}.json"
        gt_path = gt_root / f"{stem}_gt_dots.npy"
        image_name = f"{stem}.png"

        result = evaluate_image(json_path, gt_path)
        per_image_results.append(result)
        for class_id in CLASS_IDS:
            per_class_total[class_id].tp += result.per_class[class_id].tp
            per_class_total[class_id].fp += result.per_class[class_id].fp
            per_class_total[class_id].fn += result.per_class[class_id].fn
        detection_total.tp += result.detection.tp
        detection_total.fp += result.detection.fp
        detection_total.fn += result.detection.fn

        if save_visualizations:
            gt_dots = normalize_gt_dots(np.load(gt_path, allow_pickle=True))
            with open(json_path, encoding="utf-8") as file:
                payload = json.load(file)
            background = None
            image_path = img_root / image_name
            if image_path.is_file():
                background = io.imread(str(image_path))
            vis_image = draw_centroid_squares(
                gt_dots.shape[:2],
                payload.get("nuc", {}),
                type_colors,
                background=background,
            )
            io.imsave(str(vis_dir / f"{stem}_pred_centroids.png"), vis_image)

    metrics_text = format_metrics_text(
        class_names,
        per_class_total,
        detection_total,
        per_image_results,
        vis_dir,
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")
    print(metrics_text)

    return {
        "per_class": {class_id: per_class_total[class_id].as_dict() for class_id in CLASS_IDS},
        "mean_f1": float(np.mean([per_class_total[class_id].f1() for class_id in CLASS_IDS])),
        "detection": detection_total.as_dict(),
        "metrics_path": str(metrics_path),
        "vis_dir": str(vis_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HoVer-Net CoNSeP json outputs against point GT.")
    parser.add_argument("--exp_dir", type=Path, default=DEFAULT_EXP_DIR, help="Experiment directory with json/ outputs.")
    parser.add_argument("--gt_root", type=Path, default=DEFAULT_GT_ROOT, help="Directory containing *_gt_dots.npy files.")
    parser.add_argument("--img_root", type=Path, default=DEFAULT_IMG_ROOT, help="Image directory for visualization background.")
    parser.add_argument("--type_info_path", type=Path, default=DEFAULT_TYPE_INFO, help="Type id/name/color mapping json.")
    parser.add_argument("--no_vis", action="store_true", help="Skip visualization export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        exp_dir=args.exp_dir.resolve(),
        gt_root=args.gt_root.resolve(),
        img_root=args.img_root.resolve(),
        type_info_path=args.type_info_path.resolve(),
        save_visualizations=not args.no_vis,
    )


if __name__ == "__main__":
    main()
