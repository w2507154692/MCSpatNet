


from __future__ import annotations

from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MoNuSACDataset, detection_collate_fn, get_num_classes
from faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from utils import (
	build_label_id_to_name,
	compute_precision_recall_f1,
	filter_predictions_by_score,
	match_detection_counts,
	visualize_detection_result,
)


DATA_ROOT = Path("data") / "MoNuSAC"
ANNOTATIONS_CSV = DATA_ROOT / "annotations" / "boxes.csv"
CLASSES_CSV = DATA_ROOT / "metadata" / "classes.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 2
NUM_WORKERS = 0
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
RESULTS_FILE_NAME = "test_results.txt"
VISUALIZATION_DIR_NAME = "visualizations"


def faster_rcnn_test(out_dir, pth_file_path):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	results_file = out_dir / RESULTS_FILE_NAME
	visualization_dir = out_dir / VISUALIZATION_DIR_NAME

	checkpoint = torch.load(pth_file_path, map_location="cpu")
	label_id_to_name = _load_label_id_to_name(CLASSES_CSV)

	test_dataset = MoNuSACDataset(DATA_ROOT, ANNOTATIONS_CSV, split="test")
	test_loader = DataLoader(
		test_dataset,
		batch_size=TEST_BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=detection_collate_fn,
	)

	model_config = _load_model_config(checkpoint)
	model = build_faster_rcnn(model_config)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(DEVICE)
	model.eval()

	summary = {
		"num_images": len(test_dataset),
		"num_gt_boxes": 0,
		"num_pred_boxes": 0,
		"tp": 0,
		"fp": 0,
		"fn": 0,
		"score_sum": 0.0,
		"score_count": 0,
	}
	per_image_lines: list[str] = []

	with torch.inference_mode():
		progress_bar = tqdm(test_loader, desc="Test", leave=False)
		for images, targets in progress_bar:
			images_on_device = [image.to(DEVICE) for image in images]
			outputs = model(images_on_device)

			for image_tensor, target, output in zip(images, targets, outputs):
				image_index = int(target["image_id"].item())
				image_path = test_dataset.samples[image_index]["image_path"]

				gt_boxes = target["boxes"].cpu()
				gt_labels = target["labels"].cpu()

				pred_boxes = output["boxes"].detach().cpu()
				pred_labels = output["labels"].detach().cpu()
				pred_scores = output["scores"].detach().cpu()

				pred_boxes, pred_labels, pred_scores = filter_predictions_by_score(
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					pred_scores=pred_scores,
					score_threshold=SCORE_THRESHOLD,
				)

				tp, fp, fn = match_detection_counts(
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					gt_boxes=gt_boxes,
					gt_labels=gt_labels,
					iou_threshold=IOU_THRESHOLD,
				)

				visualization_path = visualization_dir / f"{Path(image_path).stem}.png"
				visualize_detection_result(
					image_tensor=image_tensor,
					save_path=visualization_path,
					gt_boxes=gt_boxes,
					gt_labels=gt_labels,
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					pred_scores=pred_scores,
					label_id_to_name=label_id_to_name,
				)

				summary["num_gt_boxes"] += int(gt_boxes.shape[0])
				summary["num_pred_boxes"] += int(pred_boxes.shape[0])
				summary["tp"] += tp
				summary["fp"] += fp
				summary["fn"] += fn
				summary["score_sum"] += float(pred_scores.sum().item())
				summary["score_count"] += int(pred_scores.numel())

				top_scores = ", ".join(f"{score:.4f}" for score in pred_scores[:5].tolist())
				if not top_scores:
					top_scores = "None"
				per_image_lines.append(
					f"image={image_path} | gt={gt_boxes.shape[0]} | pred={pred_boxes.shape[0]} | "
					f"tp={tp} | fp={fp} | fn={fn} | top_scores=[{top_scores}] | vis={visualization_path.as_posix()}"
				)

	precision, recall, f1 = compute_precision_recall_f1(summary["tp"], summary["fp"], summary["fn"])
	average_score = (
		summary["score_sum"] / summary["score_count"] if summary["score_count"] > 0 else 0.0
	)

	lines = [
		f"checkpoint_path: {pth_file_path}",
		f"device: {DEVICE}",
		f"score_threshold: {SCORE_THRESHOLD}",
		f"iou_threshold: {IOU_THRESHOLD}",
		f"visualization_dir: {visualization_dir}",
		f"num_images: {summary['num_images']}",
		f"num_gt_boxes: {summary['num_gt_boxes']}",
		f"num_pred_boxes: {summary['num_pred_boxes']}",
		f"tp: {summary['tp']}",
		f"fp: {summary['fp']}",
		f"fn: {summary['fn']}",
		f"precision: {precision:.6f}",
		f"recall: {recall:.6f}",
		f"f1: {f1:.6f}",
		f"average_score: {average_score:.6f}",
		"",
		"per_image_results:",
	]
	lines.extend(per_image_lines)

	results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"测试完成，结果已保存到: {results_file}")


def _load_model_config(checkpoint):
	model_config_dict = checkpoint.get("model_config")
	if model_config_dict is None:
		return FasterRCNNConfig(num_classes=get_num_classes(CLASSES_CSV))
	return FasterRCNNConfig(**model_config_dict)


def _load_label_id_to_name(classes_csv):
	with Path(classes_csv).open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		return build_label_id_to_name(reader)