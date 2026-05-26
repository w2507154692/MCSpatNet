from __future__ import annotations

from pathlib import Path
import platform

import numpy as np
from PIL import ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from e2ecr import E2ECRConfig, build_e2ecr
from e2ecr_dataset import (
	build_e2ecr_dataset,
	e2ecr_collate_fn,
	get_e2ecr_class_names,
	get_e2ecr_gt_colors,
	get_e2ecr_num_classes,
	get_e2ecr_pred_colors,
)


DATA_ROOT = Path("data") / "MoNuSAC_point"
DATASET_TYPE = "monusac"
SPLIT = "test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
DISTANCE_THRESHOLD = 12.0
INFERENCE_SCORE_THRESHOLD = 0.4
INFERENCE_NMS_KERNEL_SIZE = 5
INFERENCE_POINT_NMS_RADIUS = 7.0
INFERENCE_ADAPTIVE_NMS_MIN_RADIUS = 2.5
INFERENCE_ADAPTIVE_NMS_MAX_RADIUS = 8.0
INFERENCE_ADAPTIVE_NMS_SCALE = 0.8
RESULTS_FILE_NAME = "test_results.txt"
VISUALIZATION_DIR_NAME = "visualizations"
PREDICTION_DIR_NAME = "predictions"
GT_VIS_SUFFIX = "_gt.png"
PRED_VIS_SUFFIX = "_pred.png"


def e2ecr_test(out_dir, pth_file_path):
	# 测试入口：载入模型后，对测试集逐张前向、解码预测点、计算指标并保存可视化结果。
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	results_file = out_dir / RESULTS_FILE_NAME
	visualization_dir = out_dir / VISUALIZATION_DIR_NAME
	prediction_dir = out_dir / PREDICTION_DIR_NAME

	checkpoint = torch.load(pth_file_path, map_location="cpu")
	test_dataset = build_e2ecr_dataset(
		dataset_type=DATASET_TYPE,
		data_root=DATA_ROOT,
		phase=SPLIT,
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
			# 测试阶段只做前向和后处理，不做梯度相关操作。
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

				gt_visualization_path = visualization_dir / image_name.replace(".png", GT_VIS_SUFFIX)
				pred_visualization_path = visualization_dir / image_name.replace(".png", PRED_VIS_SUFFIX)
				_visualize_points(
					image_tensor=image_tensor.detach().cpu(),
					gt_save_path=gt_visualization_path,
					pred_save_path=pred_visualization_path,
					gt_points=gt_points,
					gt_labels=gt_labels,
					pred_points=pred_points,
					pred_labels=pred_labels,
					pred_scores=pred_scores,
					gt_colors=get_e2ecr_gt_colors(DATASET_TYPE),
					pred_colors=get_e2ecr_pred_colors(DATASET_TYPE),
				)

				top_scores = ", ".join(f"{score:.4f}" for score in pred_scores[:5].tolist()) if pred_scores.size > 0 else "None"
				per_image_lines.append(
					f"image={image_name} | gt={gt_points.shape[0]} | pred={pred_points.shape[0]} | "
					f"tp={tp} | fp={fp} | fn={fn} | top_scores=[{top_scores}] | pred_vis={pred_visualization_path.as_posix()} | gt_vis={gt_visualization_path.as_posix()}"
				)

	precision, recall, f1 = _compute_precision_recall_f1(summary["tp"], summary["fp"], summary["fn"])
	detection_precision, detection_recall, detection_f1 = _compute_precision_recall_f1(
		detection_summary["tp"],
		detection_summary["fp"],
		detection_summary["fn"],
	)
	per_class_lines: list[str] = []
	per_class_f1_values: list[float] = []
	class_names = get_e2ecr_class_names(DATASET_TYPE)
	for class_index, class_name in enumerate(class_names):
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
	# 兼容旧 checkpoint：如果里面没有模型配置，就回退到默认配置。
	model_config_dict = checkpoint.get("model_config")
	if model_config_dict is None:
		return E2ECRConfig(num_classes=get_e2ecr_num_classes(DATASET_TYPE))
	return E2ECRConfig(**model_config_dict)


def _decode_prediction_maps(output, model_config):
	# 将模型输出的三张预测图恢复成点集合。
	# 这里先在分数图上做一次轻量局部极大值预筛，再在最终点坐标上做半径 NMS，
	# 这样既能保留较低复杂度，也能抑制“不同像素位置回归到同一细胞附近”的粘连预测。
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

	# 将最终分数恢复成二维分数图，并通过 max-pooling 保留局部峰值点。
	score_map = final_scores.reshape(image_height, image_width)
	pooled_score_map = F.max_pool2d(
		score_map.unsqueeze(0).unsqueeze(0),
		kernel_size=INFERENCE_NMS_KERNEL_SIZE,
		stride=1,
		padding=INFERENCE_NMS_KERNEL_SIZE // 2,
	).squeeze(0).squeeze(0)
	peak_mask = torch.isclose(score_map, pooled_score_map)
	keep_mask = (score_map >= INFERENCE_SCORE_THRESHOLD) & peak_mask
	keep_mask = keep_mask.reshape(-1)
	if keep_mask.sum().item() == 0:
		return (
			torch.zeros((0, 2), dtype=pred_points.dtype, device=pred_points.device),
			torch.zeros((0,), dtype=torch.long, device=pred_points.device),
			torch.zeros((0,), dtype=final_scores.dtype, device=final_scores.device),
		)

	pred_points = pred_points[keep_mask]
	pred_labels = pred_labels[keep_mask]
	final_scores = final_scores[keep_mask]
	pred_points, pred_labels, final_scores = _apply_point_nms(
		pred_points,
		pred_labels,
		final_scores,
		base_radius=INFERENCE_POINT_NMS_RADIUS,
		min_radius=INFERENCE_ADAPTIVE_NMS_MIN_RADIUS,
		max_radius=INFERENCE_ADAPTIVE_NMS_MAX_RADIUS,
		radius_scale=INFERENCE_ADAPTIVE_NMS_SCALE,
	)
	return pred_points, pred_labels, final_scores


def _apply_point_nms(pred_points, pred_labels, pred_scores, base_radius, min_radius, max_radius, radius_scale):
	# 自适应点级 NMS 仍然作用在回归后的最终点坐标上，
	# 但不再对所有区域使用同一个固定半径。
	# 这里使用“同类别最近邻距离”估计局部密度：
	# 稠密区域最近邻更近，抑制半径会自动变小；稀疏区域则会恢复到更大的半径。
	if pred_points.shape[0] <= 1:
		return pred_points, pred_labels, pred_scores

	remaining_indices = torch.argsort(pred_scores, descending=True)
	kept_indices: list[torch.Tensor] = []
	while remaining_indices.numel() > 0:
		current_index = remaining_indices[0]
		kept_indices.append(current_index)
		if remaining_indices.numel() == 1:
			break

		other_indices = remaining_indices[1:]
		current_point = pred_points[current_index].unsqueeze(0)
		distances = torch.norm(pred_points[other_indices] - current_point, dim=1)
		same_class_mask = pred_labels[other_indices] == pred_labels[current_index]

		# 用当前点到同类别最近邻的距离估计局部间距，再映射到自适应抑制半径。
		same_class_distances = distances[same_class_mask]
		if same_class_distances.numel() > 0:
			adaptive_radius = float(torch.clamp(same_class_distances.min() * radius_scale, min=min_radius, max=max_radius).item())
		else:
			adaptive_radius = float(base_radius)

		# 只有“同类别且距离过近”的点才会被压制，避免误删相邻但类别不同的细胞。
		keep_other_mask = (distances > adaptive_radius) | (~same_class_mask)
		remaining_indices = other_indices[keep_other_mask]

	kept_indices_tensor = torch.stack(kept_indices)
	return pred_points[kept_indices_tensor], pred_labels[kept_indices_tensor], pred_scores[kept_indices_tensor]


def _match_points(pred_points, pred_labels, gt_points, gt_labels, distance_threshold, ignore_class):
	# 指标匹配采用一对一贪心策略：
	# 先收集所有满足距离阈值的预测-GT 对，再按距离从小到大依次占用。
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
	# 由 TP / FP / FN 直接计算 precision、recall 和 F1。
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return precision, recall, f1



def _visualize_points(
	image_tensor,
	gt_save_path,
	pred_save_path,
	gt_points,
	gt_labels,
	pred_points,
	pred_labels,
	pred_scores,
	gt_colors,
	pred_colors,
):
	# GT 和预测分成两张图保存，便于分别观察标注和模型输出。
	gt_image = TF.to_pil_image(image_tensor)
	gt_draw = ImageDraw.Draw(gt_image)
	for point, label in zip(gt_points.tolist(), gt_labels.tolist()):
		_draw_prediction_square(gt_draw, point, gt_colors[int(label)])

	pred_image = TF.to_pil_image(image_tensor)
	pred_draw = ImageDraw.Draw(pred_image)
	for point, label in zip(pred_points.tolist(), pred_labels.tolist()):
		_draw_prediction_square(pred_draw, point, pred_colors[int(label)])

	gt_save_path = Path(gt_save_path)
	pred_save_path = Path(pred_save_path)
	gt_save_path.parent.mkdir(parents=True, exist_ok=True)
	pred_save_path.parent.mkdir(parents=True, exist_ok=True)
	gt_image.save(gt_save_path)
	pred_image.save(pred_save_path)


def _draw_prediction_square(draw, point, color):
	# GT 和预测统一使用以中心点为中心的 5x5 实心方块绘制。
	x_coord, y_coord = int(round(float(point[0]))), int(round(float(point[1])))
	draw.rectangle((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill=color, outline=color)