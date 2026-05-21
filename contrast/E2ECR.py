from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass
class E2ECRConfig:
	input_channels: int = 3
	num_classes: int = 3
	base_channels: int = 32
	alpha: float = 0.05
	beta: float = 0.6
	lambda_reg: float = 2e-3
	candidate_multiplier: int = 8
	min_candidates: int = 128
	max_candidates: int = 2048
	inference_obj_threshold: float = 0.2
	inference_score_threshold: float = 0.15
	nms_radius: float = 6.0
	max_predictions_per_image: int = 512


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class UpBlock(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()
		self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
		self.block = ConvBlock(out_channels, out_channels)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		x = torch.cat([x, skip], dim=1)
		x = self.reduce(x)
		return self.block(x)


class E2ECR(nn.Module):
	def __init__(self, config: E2ECRConfig | None = None) -> None:
		super().__init__()
		self.config = config or E2ECRConfig()

		base = self.config.base_channels
		self.enc1 = ConvBlock(self.config.input_channels, base)
		self.enc2 = ConvBlock(base, base * 2)
		self.enc3 = ConvBlock(base * 2, base * 4)
		self.bottleneck = ConvBlock(base * 4, base * 8)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.dec3 = UpBlock(base * 8, base * 4, base * 4)
		self.dec2 = UpBlock(base * 4, base * 2, base * 2)
		self.dec1 = UpBlock(base * 2, base, base)

		self.shared_head = nn.Sequential(
			nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base),
			nn.ReLU(inplace=True),
		)
		self.reg_head = nn.Conv2d(base, 2, kernel_size=1)
		self.det_head = nn.Conv2d(base, 2, kernel_size=1)
		self.cls_head = nn.Conv2d(base, self.config.num_classes, kernel_size=1)

	def forward(self, images, targets=None):
		if isinstance(images, torch.Tensor):
			image_list = [image for image in images]
		else:
			image_list = list(images)

		outputs = []
		for image in image_list:
			output = self._forward_single_image(image.unsqueeze(0))
			outputs.append({name: value.squeeze(0) for name, value in output.items()})

		if targets is None:
			return [self._predict_single_output(output) for output in outputs]

		return self._compute_losses(outputs, targets)

	def export_config(self) -> dict:
		return asdict(self.config)

	def _forward_single_image(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
		x1 = self.enc1(image)
		x2 = self.enc2(self.pool(x1))
		x3 = self.enc3(self.pool(x2))
		x4 = self.bottleneck(self.pool(x3))

		x = self.dec3(x4, x3)
		x = self.dec2(x, x2)
		x = self.dec1(x, x1)
		x = self.shared_head(x)

		return {
			"reg": self.reg_head(x),
			"det": self.det_head(x),
			"cls": self.cls_head(x),
		}

	def _compute_losses(self, outputs, targets):
		device = outputs[0]["reg"].device
		loss_reg = torch.tensor(0.0, device=device)
		loss_det = torch.tensor(0.0, device=device)
		loss_cls = torch.tensor(0.0, device=device)

		for output, target in zip(outputs, targets):
			single_losses = self._compute_single_image_losses(output, target)
			loss_reg = loss_reg + single_losses["loss_reg"]
			loss_det = loss_det + single_losses["loss_det"]
			loss_cls = loss_cls + single_losses["loss_cls"]

		batch_size = max(1, len(outputs))
		return {
			"loss_reg": loss_reg / batch_size,
			"loss_det": loss_det / batch_size,
			"loss_cls": loss_cls / batch_size,
		}

	def _compute_single_image_losses(self, output, target):
		reg_map = output["reg"]
		det_map = output["det"]
		cls_map = output["cls"]

		height, width = reg_map.shape[-2:]
		pred_points, det_probs, cls_probs, det_logits_flat, cls_logits_flat = self._flatten_predictions(
			reg_map,
			det_map,
			cls_map,
		)

		gt_points = target["points"].to(pred_points.device)
		gt_labels = target["labels"].to(pred_points.device)

		selected_indices = self._select_candidate_indices(
			obj_scores=det_probs[:, 1],
			gt_points=gt_points,
			width=width,
			height=height,
		)

		selected_pred_points = pred_points[selected_indices]
		selected_det_probs = det_probs[selected_indices]
		selected_cls_probs = cls_probs[selected_indices]
		selected_det_logits = det_logits_flat[selected_indices]
		selected_cls_logits = cls_logits_flat[selected_indices]

		matched_pred_indices, matched_gt_indices = self._match_predictions_to_gt(
			pred_points=selected_pred_points,
			obj_scores=selected_det_probs[:, 1],
			cls_probs=selected_cls_probs,
			gt_points=gt_points,
			gt_labels=gt_labels,
		)

		loss_reg = torch.tensor(0.0, device=pred_points.device)
		loss_cls = torch.tensor(0.0, device=pred_points.device)
		if matched_pred_indices.numel() > 0:
			matched_pred_points = selected_pred_points[matched_pred_indices]
			matched_gt_points = gt_points[matched_gt_indices]
			loss_reg = F.mse_loss(matched_pred_points, matched_gt_points)
			loss_cls = F.cross_entropy(selected_cls_logits[matched_pred_indices], gt_labels[matched_gt_indices])

		selected_log_probs = F.log_softmax(selected_det_logits, dim=1)
		positive_mask = torch.zeros((selected_indices.shape[0],), dtype=torch.bool, device=pred_points.device)
		if matched_pred_indices.numel() > 0:
			positive_mask[matched_pred_indices] = True
		negative_mask = ~positive_mask

		positive_term = -selected_log_probs[positive_mask, 1].sum() if positive_mask.any() else torch.tensor(0.0, device=pred_points.device)
		negative_term = -selected_log_probs[negative_mask, 0].sum() if negative_mask.any() else torch.tensor(0.0, device=pred_points.device)
		m_value = max(1, int(selected_indices.shape[0]))
		loss_det = (positive_term + self.config.beta * negative_term) / m_value

		return {
			"loss_reg": self.config.lambda_reg * loss_reg,
			"loss_det": loss_det,
			"loss_cls": loss_cls,
		}

	def _predict_single_output(self, output):
		reg_map = output["reg"]
		det_map = output["det"]
		cls_map = output["cls"]
		height, width = reg_map.shape[-2:]

		pred_points, det_probs, cls_probs, _, _ = self._flatten_predictions(reg_map, det_map, cls_map)
		obj_scores = det_probs[:, 1]
		cls_scores, cls_labels = cls_probs.max(dim=1)
		combined_scores = obj_scores * cls_scores

		keep = (obj_scores >= self.config.inference_obj_threshold) & (
			combined_scores >= self.config.inference_score_threshold
		)
		if not keep.any():
			return {
				"points": pred_points.new_zeros((0, 2)),
				"labels": cls_labels.new_zeros((0,), dtype=torch.int64),
				"scores": combined_scores.new_zeros((0,)),
				"obj_scores": obj_scores.new_zeros((0,)),
			}

		pred_points = pred_points[keep]
		cls_labels = cls_labels[keep]
		combined_scores = combined_scores[keep]
		obj_scores = obj_scores[keep]

		sort_indices = torch.argsort(combined_scores, descending=True)
		pred_points = pred_points[sort_indices]
		cls_labels = cls_labels[sort_indices]
		combined_scores = combined_scores[sort_indices]
		obj_scores = obj_scores[sort_indices]

		keep_indices = self._greedy_point_nms(pred_points, self.config.nms_radius)
		keep_indices = keep_indices[: self.config.max_predictions_per_image]

		pred_points = pred_points[keep_indices]
		pred_points[:, 0].clamp_(0.0, float(width - 1))
		pred_points[:, 1].clamp_(0.0, float(height - 1))

		return {
			"points": pred_points,
			"labels": cls_labels[keep_indices],
			"scores": combined_scores[keep_indices],
			"obj_scores": obj_scores[keep_indices],
		}

	def _flatten_predictions(self, reg_map, det_map, cls_map):
		height, width = reg_map.shape[-2:]
		grid = self._build_location_grid(height, width, reg_map.device, reg_map.dtype)

		reg_flat = reg_map.permute(1, 2, 0).reshape(-1, 2)
		det_logits_flat = det_map.permute(1, 2, 0).reshape(-1, 2)
		cls_logits_flat = cls_map.permute(1, 2, 0).reshape(-1, self.config.num_classes)

		det_probs = torch.softmax(det_logits_flat, dim=1)
		cls_probs = torch.softmax(cls_logits_flat, dim=1)
		pred_points = grid + reg_flat
		return pred_points, det_probs, cls_probs, det_logits_flat, cls_logits_flat

	def _select_candidate_indices(self, obj_scores, gt_points, width, height):
		total_locations = int(obj_scores.shape[0])
		gt_count = int(gt_points.shape[0])
		candidate_count = min(
			total_locations,
			max(self.config.min_candidates, gt_count * self.config.candidate_multiplier),
		)
		candidate_count = min(candidate_count, self.config.max_candidates)
		candidate_count = max(candidate_count, gt_count)

		topk_count = min(total_locations, candidate_count)
		topk_indices = torch.topk(obj_scores, k=topk_count, largest=True).indices
		if gt_count == 0:
			return topk_indices

		rounded_x = gt_points[:, 0].round().clamp(0, width - 1).long()
		rounded_y = gt_points[:, 1].round().clamp(0, height - 1).long()
		gt_anchor_indices = rounded_y * width + rounded_x
		return torch.unique(torch.cat([topk_indices, gt_anchor_indices], dim=0), sorted=False)

	def _match_predictions_to_gt(self, pred_points, obj_scores, cls_probs, gt_points, gt_labels):
		if gt_points.numel() == 0 or pred_points.numel() == 0:
			return self._empty_long_tensor(pred_points.device), self._empty_long_tensor(pred_points.device)

		dist_matrix = torch.cdist(pred_points, gt_points, p=2)
		class_score_matrix = cls_probs[:, gt_labels]
		cost_matrix = self.config.alpha * dist_matrix - obj_scores.unsqueeze(1) - class_score_matrix

		row_indices, col_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
		row_indices_tensor = torch.as_tensor(row_indices, dtype=torch.long, device=pred_points.device)
		col_indices_tensor = torch.as_tensor(col_indices, dtype=torch.long, device=pred_points.device)
		return row_indices_tensor, col_indices_tensor

	def _build_location_grid(self, height, width, device, dtype):
		y_coords = torch.arange(height, device=device, dtype=dtype)
		x_coords = torch.arange(width, device=device, dtype=dtype)
		grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
		return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

	def _greedy_point_nms(self, points, radius):
		keep_indices: list[int] = []
		for index in range(points.shape[0]):
			candidate = points[index]
			keep = True
			for kept_index in keep_indices:
				if torch.norm(candidate - points[kept_index], p=2) < radius:
					keep = False
					break
			if keep:
				keep_indices.append(index)
		return torch.as_tensor(keep_indices, dtype=torch.long, device=points.device)

	def _empty_long_tensor(self, device):
		return torch.zeros((0,), dtype=torch.long, device=device)


def build_e2ecr(config: E2ECRConfig | None = None) -> E2ECR:
	return E2ECR(config=config)