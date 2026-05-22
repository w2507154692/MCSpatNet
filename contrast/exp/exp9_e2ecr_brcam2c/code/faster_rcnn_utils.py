from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import ImageDraw
from torchvision.transforms import functional as F


def box_iou(box1, box2):
	"""计算两个边界框的 IoU。"""

	x1 = max(float(box1[0]), float(box2[0]))
	y1 = max(float(box1[1]), float(box2[1]))
	x2 = min(float(box1[2]), float(box2[2]))
	y2 = min(float(box1[3]), float(box2[3]))

	inter_w = max(0.0, x2 - x1)
	inter_h = max(0.0, y2 - y1)
	inter_area = inter_w * inter_h

	area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
	area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
	union_area = area1 + area2 - inter_area

	if union_area <= 0:
		return 0.0
	return inter_area / union_area


def match_detection_counts(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold):
	"""基于 IoU 和类别匹配，返回 TP / FP / FN 数量。

	该接口只关心 boxes、labels 和阈值，不关心模型来源。
	"""

	matched_gt_indices: set[int] = set()
	tp = 0
	fp = 0

	if pred_boxes.numel() == 0:
		return 0, 0, int(gt_boxes.shape[0])

	for pred_index in range(pred_boxes.shape[0]):
		best_iou = 0.0
		best_gt_index = -1

		for gt_index in range(gt_boxes.shape[0]):
			if gt_index in matched_gt_indices:
				continue
			if int(pred_labels[pred_index].item()) != int(gt_labels[gt_index].item()):
				continue

			iou = box_iou(pred_boxes[pred_index], gt_boxes[gt_index])
			if iou > best_iou:
				best_iou = iou
				best_gt_index = gt_index

		if best_gt_index >= 0 and best_iou >= iou_threshold:
			matched_gt_indices.add(best_gt_index)
			tp += 1
		else:
			fp += 1

	fn = int(gt_boxes.shape[0]) - len(matched_gt_indices)
	return tp, fp, fn


def compute_precision_recall_f1(tp, fp, fn):
	"""根据 TP / FP / FN 计算 precision、recall、f1。"""

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return precision, recall, f1


def filter_predictions_by_score(pred_boxes, pred_labels, pred_scores, score_threshold):
	"""按分数阈值过滤预测结果。"""

	keep = pred_scores >= score_threshold
	return pred_boxes[keep], pred_labels[keep], pred_scores[keep]


def visualize_detection_result(
	image_tensor,
	save_path,
	gt_boxes,
	gt_labels,
	pred_boxes,
	pred_labels,
	pred_scores,
	label_id_to_name=None,
	gt_color="lime",
	pred_color="red",
):
	"""可视化单张图的 GT 和预测框。

	输入只依赖图像张量和检测结果，不与具体模型输出结构绑定。
	"""

	image = F.to_pil_image(image_tensor.cpu())
	draw = ImageDraw.Draw(image)

	for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
		label_text = _label_to_text(label, label_id_to_name)
		_draw_box(draw, box, f"GT:{label_text}", gt_color)

	for box, label, score in zip(pred_boxes.tolist(), pred_labels.tolist(), pred_scores.tolist()):
		label_text = _label_to_text(label, label_id_to_name)
		_draw_box(draw, box, f"Pred:{label_text} {score:.2f}", pred_color)

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	image.save(save_path)


def build_label_id_to_name(rows: Iterable[dict]) -> dict[int, str]:
	"""根据类别表构建 label_id -> label_name 映射。"""

	return {int(row["label_id"]): row["label"] for row in rows}


def _label_to_text(label_id, label_id_to_name):
	if label_id_to_name is None:
		return str(label_id)
	return label_id_to_name.get(int(label_id), str(label_id))


def _draw_box(draw, box, text, color):
	xmin, ymin, xmax, ymax = [float(value) for value in box]
	draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=2)
	text_y = max(0.0, ymin - 12.0)
	draw.text((xmin, text_y), text, fill=color)