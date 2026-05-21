from __future__ import annotations

from pathlib import Path
import platform

import numpy as np
from PIL import ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

from e2ecr import E2ECRConfig, build_e2ecr
from dataset import build_e2ecr_dataset, e2ecr_collate_fn, get_e2ecr_num_classes


DATA_ROOT = Path("data") / "BRCA-M2C"
DATASET_TYPE = "brca-m2c"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
DISTANCE_THRESHOLD = 6.0
RESULTS_FILE_NAME = "test_results.txt"
VISUALIZATION_DIR_NAME = "visualizations"
PREDICTION_DIR_NAME = "predictions"
CLASS_NAMES = ["inflammatory", "epithelial", "stromal"]
CLASS_COLORS = ["lime", "orange", "cyan"]


def e2ecr_test(out_dir, pth_file_path):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	results_file = out_dir / RESULTS_FILE_NAME
	visualization_dir = out_dir / VISUALIZATION_DIR_NAME
	prediction_dir = out_dir / PREDICTION_DIR_NAME

	checkpoint = torch.load(pth_file_path, map_location="cpu")
	test_dataset = build_e2ecr_dataset(
		dataset_type=DATASET_TYPE,
		data_root=DATA_ROOT,
		phase="test",
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=TEST_BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=e2ecr_collate_fn,
	)

	model_config = _load_model_config(checkpoint)
	model = build_e2ecr(model_config)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(DEVICE)
	model.eval()

	summary = {"num_images": len(test_dataset), "num_gt_points": 0, "num_pred_points": 0, "tp": 0, "fp": 0, "fn": 0}
	per_class_summary = {class_index: {"tp": 0, "fp": 0, "fn": 0} for class_index in range(get_e2ecr_num_classes(DATASET_TYPE))}
	detection_summary = {"tp": 0, "fp": 0, "fn": 0}
	per_image_lines: list[str] = []

	with torch.inference_mode():
		progress_bar = tqdm(test_loader, desc="Test", leave=False)
		for images, targets in progress_bar:
			images = [image.to(DEVICE) for image in images]
			outputs = model(images)

			for image_tensor, target, output in zip(images, targets, outputs):
				gt_points = target["points"].cpu().numpy()
				gt_labels = target["labels"].cpu().numpy()
				pred_points, pred_labels, pred_scores = _decode_prediction_maps(output, model_config)
				pred_points = pred_points.detach().cpu().numpy()
				pred_labels = pred_labels.detach().cpu().numpy()
				pred_scores = pred_scores.detach().cpu().numpy()

				tp, fp, fn = _match_points(pred_points, pred_labels, gt_points, gt_labels, DISTANCE_THRESHOLD, ignore_class=False)
				detection_tp, detection_fp, detection_fn = _match_points(
					pred_points,
					pred_labels,
					gt_points,
					gt_labels,
					DISTANCE_THRESHOLD,
					ignore_class=True,
				)

				for class_index in range(get_e2ecr_num_classes(DATASET_TYPE)):
					class_mask_pred = pred_labels == class_index
					class_mask_gt = gt_labels == class_index
					class_tp, class_fp, class_fn = _match_points(
						pred_points[class_mask_pred],
						pred_labels[class_mask_pred],
						gt_points[class_mask_gt],
						gt_labels[class_mask_gt],
						DISTANCE_THRESHOLD,
						ignore_class=False,
					)
					per_class_summary[class_index]["tp"] += class_tp
					per_class_summary[class_index]["fp"] += class_fp
					per_class_summary[class_index]["fn"] += class_fn

				summary["num_gt_points"] += int(gt_points.shape[0])
				summary["num_pred_points"] += int(pred_points.shape[0])
				summary["tp"] += tp
				summary["fp"] += fp
				summary["fn"] += fn
				detection_summary["tp"] += detection_tp
				detection_summary["fp"] += detection_fp
				detection_summary["fn"] += detection_fn

				image_name = target["image_name"]
				prediction_path = prediction_dir / image_name.replace(".png", "_pred_points.npy")
				prediction_path.parent.mkdir(parents=True, exist_ok=True)
				np.save(
					prediction_path,
					{
						"points": pred_points,
						"labels": pred_labels,
						"scores": pred_scores,
					},
					allow_pickle=True,
				)

				visualization_path = visualization_dir / image_name
				_visualize_points(
					image_tensor=image_tensor.detach().cpu(),
					save_path=visualization_path,
					gt_points=gt_points,
					gt_labels=gt_labels,
					pred_points=pred_points,
					pred_labels=pred_labels,
					pred_scores=pred_scores,
				)

				top_scores = ", ".join(f"{score:.4f}" for score in pred_scores[:5].tolist()) if pred_scores.size > 0 else "None"
				per_image_lines.append(
					f"image={image_name} | gt={gt_points.shape[0]} | pred={pred_points.shape[0]} | "
					f"tp={tp} | fp={fp} | fn={fn} | top_scores=[{top_scores}] | vis={visualization_path.as_posix()}"
				)

	precision, recall, f1 = _compute_precision_recall_f1(summary["tp"], summary["fp"], summary["fn"])
	detection_precision, detection_recall, detection_f1 = _compute_precision_recall_f1(
		detection_summary["tp"],
		detection_summary["fp"],
		detection_summary["fn"],
	)
	per_class_lines: list[str] = []
	per_class_f1_values: list[float] = []
	for class_index, class_name in enumerate(CLASS_NAMES):
		class_summary = per_class_summary[class_index]
		class_precision, class_recall, class_f1 = _compute_precision_recall_f1(
			class_summary["tp"],
			class_summary["fp"],
			class_summary["fn"],
		)
		per_class_f1_values.append(class_f1)
		per_class_lines.extend(
			[
				f"class={class_name} (label_id={class_index})",
				f"  tp: {class_summary['tp']}",
				f"  fp: {class_summary['fp']}",
				f"  fn: {class_summary['fn']}",
				f"  precision: {class_precision:.6f}",
				f"  recall: {class_recall:.6f}",
				f"  f1: {class_f1:.6f}",
			]
		)
	mean_class_f1 = sum(per_class_f1_values) / len(per_class_f1_values) if per_class_f1_values else 0.0

	lines = [
		f"checkpoint_path: {pth_file_path}",
		f"device: {DEVICE}",
		f"dataset_type: {DATASET_TYPE}",
		f"data_root: {DATA_ROOT}",
		f"distance_threshold: {DISTANCE_THRESHOLD}",
		f"visualization_dir: {visualization_dir}",
		f"prediction_dir: {prediction_dir}",
		f"num_images: {summary['num_images']}",
		f"num_gt_points: {summary['num_gt_points']}",
		f"num_pred_points: {summary['num_pred_points']}",
		"classification_aware_results:",
		f"tp: {summary['tp']}",
		f"fp: {summary['fp']}",
		f"fn: {summary['fn']}",
		f"precision: {precision:.6f}",
		f"recall: {recall:.6f}",
		f"f1: {f1:.6f}",
		f"mean_class_f1_foreground_only: {mean_class_f1:.6f}",
		"",
		"per_class_results:",
	]
	lines.extend(per_class_lines)
	lines.extend(
		[
			"",
			"detection_results_ignore_class:",
			f"tp: {detection_summary['tp']}",
			f"fp: {detection_summary['fp']}",
			f"fn: {detection_summary['fn']}",
			f"precision: {detection_precision:.6f}",
			f"recall: {detection_recall:.6f}",
			f"f1: {detection_f1:.6f}",
			"",
			"per_image_results:",
		]
	)
	lines.extend(per_image_lines)
	results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"测试完成，结果已保存到: {results_file}")


def _load_model_config(checkpoint):
	model_config_dict = checkpoint.get("model_config")
	if model_config_dict is None:
		return E2ECRConfig(num_classes=get_e2ecr_num_classes(DATASET_TYPE))
	return E2ECRConfig(**model_config_dict)


def _decode_prediction_maps(output, model_config):
	reg_map = output["reg"]
	det_map = output["det"]
	cls_map = output["cls"]
	image_height, image_width = det_map.shape[-2], det_map.shape[-1]

	reg_logits = reg_map.permute(1, 2, 0).reshape(-1, 2)
	det_logits = det_map.permute(1, 2, 0).reshape(-1, 2)
	cls_logits = cls_map.permute(1, 2, 0).reshape(-1, model_config.num_classes)

	y_coords = torch.arange(image_height, device=reg_map.device, dtype=reg_logits.dtype)
	x_coords = torch.arange(image_width, device=reg_map.device, dtype=reg_logits.dtype)
	y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
	base_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
	pred_points = base_points + reg_logits

	det_probs = torch.softmax(det_logits, dim=1)
	obj_scores = det_probs[:, 1]
	cls_probs = torch.softmax(cls_logits, dim=1)
	cls_scores, pred_labels = torch.max(cls_probs, dim=1)
	final_scores = obj_scores * cls_scores

	keep_mask = (obj_scores >= model_config.inference_obj_threshold) & (
		final_scores >= model_config.inference_score_threshold
	)
	if keep_mask.sum().item() == 0:
		return (
			torch.zeros((0, 2), dtype=pred_points.dtype, device=pred_points.device),
			torch.zeros((0,), dtype=torch.long, device=pred_points.device),
			torch.zeros((0,), dtype=final_scores.dtype, device=final_scores.device),
		)

	pred_points = pred_points[keep_mask]
	pred_labels = pred_labels[keep_mask]
	final_scores = final_scores[keep_mask]

	sorted_indices = torch.argsort(final_scores, descending=True)
	selected_indices = []
	for candidate_index in sorted_indices.tolist():
		candidate_point = pred_points[candidate_index]
		should_keep = True
		for kept_index in selected_indices:
			if torch.norm(candidate_point - pred_points[kept_index], p=2).item() <= model_config.nms_radius:
				should_keep = False
				break
		if should_keep:
			selected_indices.append(candidate_index)
			if len(selected_indices) >= model_config.max_predictions_per_image:
				break

	selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=pred_points.device)
	return pred_points[selected_indices], pred_labels[selected_indices], final_scores[selected_indices]


def _match_points(pred_points, pred_labels, gt_points, gt_labels, distance_threshold, ignore_class):
	if pred_points.shape[0] == 0:
		return 0, 0, int(gt_points.shape[0])
	if gt_points.shape[0] == 0:
		return 0, int(pred_points.shape[0]), 0

	pairs: list[tuple[float, int, int]] = []
	for pred_index in range(pred_points.shape[0]):
		for gt_index in range(gt_points.shape[0]):
			if not ignore_class and int(pred_labels[pred_index]) != int(gt_labels[gt_index]):
				continue
			distance = float(np.linalg.norm(pred_points[pred_index] - gt_points[gt_index]))
			if distance <= distance_threshold:
				pairs.append((distance, pred_index, gt_index))

	pairs.sort(key=lambda item: item[0])
	matched_pred_indices: set[int] = set()
	matched_gt_indices: set[int] = set()
	for _, pred_index, gt_index in pairs:
		if pred_index in matched_pred_indices or gt_index in matched_gt_indices:
			continue
		matched_pred_indices.add(pred_index)
		matched_gt_indices.add(gt_index)

	tp = len(matched_pred_indices)
	fp = int(pred_points.shape[0]) - tp
	fn = int(gt_points.shape[0]) - tp
	return tp, fp, fn


def _compute_precision_recall_f1(tp, fp, fn):
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return precision, recall, f1


def _visualize_points(image_tensor, save_path, gt_points, gt_labels, pred_points, pred_labels, pred_scores):
	image = F.to_pil_image(image_tensor)
	draw = ImageDraw.Draw(image)
	for point, label in zip(gt_points.tolist(), gt_labels.tolist()):
		_draw_point(draw, point, CLASS_COLORS[int(label)], radius=4, with_cross=True)
	for point, label, score in zip(pred_points.tolist(), pred_labels.tolist(), pred_scores.tolist()):
		_draw_point(draw, point, CLASS_COLORS[int(label)], radius=6, with_cross=False)
		x_coord, y_coord = float(point[0]), float(point[1])
		draw.text((x_coord + 4.0, y_coord + 4.0), f"{score:.2f}", fill=CLASS_COLORS[int(label)])
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	image.save(save_path)


def _draw_point(draw, point, color, radius, with_cross):
	x_coord, y_coord = float(point[0]), float(point[1])
	draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), outline=color, width=2)
	if with_cross:
		draw.line((x_coord - radius, y_coord, x_coord + radius, y_coord), fill=color, width=2)
		draw.line((x_coord, y_coord - radius, x_coord, y_coord + radius), fill=color, width=2)