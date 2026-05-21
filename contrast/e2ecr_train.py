from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import platform

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from e2ecr import E2ECRConfig, build_e2ecr
from dataset import build_e2ecr_dataset, e2ecr_collate_fn, get_e2ecr_num_classes


DATA_ROOT = Path("data") / "BRCA-M2C"
DATASET_TYPE = "brca-m2c"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200
BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
NUM_WORKERS = 0 if platform.system() == "Windows" else 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 20
LR_GAMMA = 0.5
EVAL_INTERVAL_EPOCHS = 1
PRINT_FREQ = 10
GRAD_CLIP_NORM = 5.0
TRAIN_CROP_SIZE = 384


def e2ecr_train(checkpoints_save_dir, logger):
	checkpoints_dir = Path(checkpoints_save_dir)
	checkpoints_dir.mkdir(parents=True, exist_ok=True)
	train_dataset = build_e2ecr_dataset(
		dataset_type=DATASET_TYPE,
		data_root=DATA_ROOT,
		phase="train",
		crop_size=TRAIN_CROP_SIZE,
	)
	val_dataset = build_e2ecr_dataset(
		dataset_type=DATASET_TYPE,
		data_root=DATA_ROOT,
		phase="val",
		crop_size=TRAIN_CROP_SIZE,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=e2ecr_collate_fn,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=VAL_BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=e2ecr_collate_fn,
	)

	model_config = E2ECRConfig(num_classes=get_e2ecr_num_classes(DATASET_TYPE))
	model = build_e2ecr(model_config).to(DEVICE)
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

	best_val_loss = float("inf")
	best_checkpoint_path: Path | None = None

	logger.info(f"训练设备: {DEVICE}")
	logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")
	logger.info(f"数据集类型: {DATASET_TYPE} | 数据集路径: {DATA_ROOT}")
	logger.info(f"类别数: {get_e2ecr_num_classes(DATASET_TYPE)}")
	logger.info(
		f"训练配置: epochs={NUM_EPOCHS}, train_batch_size={BATCH_SIZE}, val_batch_size={VAL_BATCH_SIZE}, lr={LEARNING_RATE}, crop_size={TRAIN_CROP_SIZE}"
	)
	logger.info(f"模型配置: {asdict(model_config)}")

	for epoch in range(1, NUM_EPOCHS + 1):
		model.train()
		train_running_total_loss = 0.0
		train_running_metrics = {"loss_cls": 0.0, "loss_det": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
		train_valid_steps = 0
		train_progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

		for step, (images, targets) in enumerate(train_progress_bar, start=1):
			images = [image.to(DEVICE) for image in images]
			targets = [
				{key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in target.items()}
				for target in targets
			]

			# KEY：前向传播
			outputs = model(images)
			loss_dict = {
				"loss_cls": torch.zeros((), device=DEVICE),
				"loss_det": torch.zeros((), device=DEVICE),
				"loss_reg": torch.zeros((), device=DEVICE),
			}

			for output, target in zip(outputs, targets):
				reg_map = output["reg"]
				det_map = output["det"]
				cls_map = output["cls"]
				image_height, image_width = det_map.shape[-2], det_map.shape[-1]

				reg_logits = reg_map.permute(1, 2, 0).reshape(-1, 2)
				det_logits = det_map.permute(1, 2, 0).reshape(-1, 2)
				cls_logits = cls_map.permute(1, 2, 0).reshape(-1, model_config.num_classes)

				y_coords = torch.arange(image_height, device=DEVICE, dtype=reg_logits.dtype)
				x_coords = torch.arange(image_width, device=DEVICE, dtype=reg_logits.dtype)
				y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
				base_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
				pred_points = base_points + reg_logits

				det_probs = torch.softmax(det_logits, dim=1)
				obj_scores = det_probs[:, 1]
				cls_probs = torch.softmax(cls_logits, dim=1)

				gt_points = target["points"]
				gt_labels = target["labels"]
				det_targets = torch.zeros(det_logits.shape[0], dtype=torch.long, device=DEVICE)
				zero_loss = det_logits.sum() * 0.0
				loss_reg = zero_loss
				loss_cls = zero_loss

				# KEY：匈牙利算法匹配预测和 ground truth
				if gt_points.shape[0] > 0:
					num_candidates = max(model_config.min_candidates, int(gt_points.shape[0]) * model_config.candidate_multiplier)
					num_candidates = min(num_candidates, model_config.max_candidates, obj_scores.shape[0])
					candidate_scores, candidate_indices = torch.topk(
						obj_scores,
						k=num_candidates,
						largest=True,
						sorted=False,
					)
					candidate_points = pred_points[candidate_indices]
					candidate_cls_probs = cls_probs[candidate_indices]
					distance_matrix = torch.cdist(candidate_points, gt_points, p=2)
					class_score_matrix = candidate_cls_probs[:, gt_labels]
					cost_matrix = model_config.alpha * distance_matrix - candidate_scores.unsqueeze(1) - class_score_matrix

					matched_candidate_rows, matched_gt_cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
					if len(matched_candidate_rows) > 0:
						matched_candidate_rows = torch.as_tensor(matched_candidate_rows, dtype=torch.long, device=DEVICE)
						matched_gt_cols = torch.as_tensor(matched_gt_cols, dtype=torch.long, device=DEVICE)
						matched_indices = candidate_indices[matched_candidate_rows]
						det_targets[matched_indices] = 1
						loss_reg = F.mse_loss(pred_points[matched_indices], gt_points[matched_gt_cols], reduction="mean")
						loss_cls = F.cross_entropy(cls_logits[matched_indices], gt_labels[matched_gt_cols])

				det_losses = F.cross_entropy(det_logits, det_targets, reduction="none")
				det_weights = torch.full_like(det_losses, 1.0 - model_config.beta)
				det_weights[det_targets == 1] = model_config.beta
				loss_det = (det_losses * det_weights).mean()

				loss_dict["loss_reg"] = loss_dict["loss_reg"] + loss_reg
				loss_dict["loss_det"] = loss_dict["loss_det"] + loss_det
				loss_dict["loss_cls"] = loss_dict["loss_cls"] + loss_cls

			batch_size = max(1, len(outputs))
			loss_dict = {name: value / batch_size for name, value in loss_dict.items()}
			loss_dict["loss_total"] = model_config.lambda_reg * loss_dict["loss_reg"] + loss_dict["loss_det"] + loss_dict["loss_cls"]
			total_loss = loss_dict["loss_total"]

			if not torch.isfinite(total_loss):
				logger.warning(
					f"Epoch {epoch} train Step {step}/{len(train_loader)} 出现非有限 loss，跳过该 batch: "
					+ " | ".join(
						f"{name}={value.detach().item():.6f}" for name, value in sorted(loss_dict.items())
					)
				)
				optimizer.zero_grad()
				continue

			optimizer.zero_grad()
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
			optimizer.step()

			train_valid_steps += 1
			train_running_total_loss += total_loss.item()
			for name, value in loss_dict.items():
				train_running_metrics[name] += value.item()

			avg_loss = train_running_total_loss / train_valid_steps
			train_progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
			if step % PRINT_FREQ == 0 or step == len(train_loader):
				logger.info(f"Epoch {epoch} Step {step}/{len(train_loader)} | train_loss={avg_loss:.6f}")

		if train_valid_steps == 0:
			raise RuntimeError("整个 train epoch 都出现了非有限 loss，请检查数据和训练配置")

		train_loss = train_running_total_loss / train_valid_steps
		train_metrics = {name: value / train_valid_steps for name, value in train_running_metrics.items()}
		scheduler.step()
		current_lr = optimizer.param_groups[0]["lr"]
		logger.info(f"Epoch {epoch}/{NUM_EPOCHS} 训练结束 | train_loss={train_loss:.6f} | lr={current_lr:.8f}")
		logger.info(
			f"Epoch {epoch} 训练损失明细: "
			+ " | ".join(f"{name}={value:.6f}" for name, value in sorted(train_metrics.items()))
		)

		if epoch % EVAL_INTERVAL_EPOCHS != 0 and epoch != NUM_EPOCHS:
			continue

		model.eval()
		val_running_total_loss = 0.0
		val_running_metrics = {"loss_cls": 0.0, "loss_det": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
		val_valid_steps = 0
		val_progress_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False)

		with torch.no_grad():
			for step, (images, targets) in enumerate(val_progress_bar, start=1):
				images = [image.to(DEVICE) for image in images]
				targets = [
					{key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in target.items()}
					for target in targets
				]

				outputs = model(images)
				loss_dict = {
					"loss_cls": torch.zeros((), device=DEVICE),
					"loss_det": torch.zeros((), device=DEVICE),
					"loss_reg": torch.zeros((), device=DEVICE),
				}

				for output, target in zip(outputs, targets):
					reg_map = output["reg"]
					det_map = output["det"]
					cls_map = output["cls"]
					image_height, image_width = det_map.shape[-2], det_map.shape[-1]

					reg_logits = reg_map.permute(1, 2, 0).reshape(-1, 2)
					det_logits = det_map.permute(1, 2, 0).reshape(-1, 2)
					cls_logits = cls_map.permute(1, 2, 0).reshape(-1, model_config.num_classes)

					y_coords = torch.arange(image_height, device=DEVICE, dtype=reg_logits.dtype)
					x_coords = torch.arange(image_width, device=DEVICE, dtype=reg_logits.dtype)
					y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
					base_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
					pred_points = base_points + reg_logits

					det_probs = torch.softmax(det_logits, dim=1)
					obj_scores = det_probs[:, 1]
					cls_probs = torch.softmax(cls_logits, dim=1)

					gt_points = target["points"]
					gt_labels = target["labels"]
					det_targets = torch.zeros(det_logits.shape[0], dtype=torch.long, device=DEVICE)
					zero_loss = det_logits.sum() * 0.0
					loss_reg = zero_loss
					loss_cls = zero_loss

					if gt_points.shape[0] > 0:
						num_candidates = max(model_config.min_candidates, int(gt_points.shape[0]) * model_config.candidate_multiplier)
						num_candidates = min(num_candidates, model_config.max_candidates, obj_scores.shape[0])
						candidate_scores, candidate_indices = torch.topk(
							obj_scores,
							k=num_candidates,
							largest=True,
							sorted=False,
						)
						candidate_points = pred_points[candidate_indices]
						candidate_cls_probs = cls_probs[candidate_indices]
						distance_matrix = torch.cdist(candidate_points, gt_points, p=2)
						class_score_matrix = candidate_cls_probs[:, gt_labels]
						cost_matrix = model_config.alpha * distance_matrix - candidate_scores.unsqueeze(1) - class_score_matrix

						matched_candidate_rows, matched_gt_cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
						if len(matched_candidate_rows) > 0:
							matched_candidate_rows = torch.as_tensor(matched_candidate_rows, dtype=torch.long, device=DEVICE)
							matched_gt_cols = torch.as_tensor(matched_gt_cols, dtype=torch.long, device=DEVICE)
							matched_indices = candidate_indices[matched_candidate_rows]
							det_targets[matched_indices] = 1
							loss_reg = F.mse_loss(pred_points[matched_indices], gt_points[matched_gt_cols], reduction="mean")
							loss_cls = F.cross_entropy(cls_logits[matched_indices], gt_labels[matched_gt_cols])

					det_losses = F.cross_entropy(det_logits, det_targets, reduction="none")
					det_weights = torch.full_like(det_losses, 1.0 - model_config.beta)
					det_weights[det_targets == 1] = model_config.beta
					loss_det = (det_losses * det_weights).mean()

					loss_dict["loss_reg"] = loss_dict["loss_reg"] + loss_reg
					loss_dict["loss_det"] = loss_dict["loss_det"] + loss_det
					loss_dict["loss_cls"] = loss_dict["loss_cls"] + loss_cls

				batch_size = max(1, len(outputs))
				loss_dict = {name: value / batch_size for name, value in loss_dict.items()}
				loss_dict["loss_total"] = model_config.lambda_reg * loss_dict["loss_reg"] + loss_dict["loss_det"] + loss_dict["loss_cls"]
				total_loss = loss_dict["loss_total"]

				if not torch.isfinite(total_loss):
					logger.warning(
						f"Epoch {epoch} val Step {step}/{len(val_loader)} 出现非有限 loss，跳过该 batch: "
						+ " | ".join(
							f"{name}={value.detach().item():.6f}" for name, value in sorted(loss_dict.items())
						)
					)
					continue

				val_valid_steps += 1
				val_running_total_loss += total_loss.item()
				for name, value in loss_dict.items():
					val_running_metrics[name] += value.item()

				avg_loss = val_running_total_loss / val_valid_steps
				val_progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
				if step % PRINT_FREQ == 0 or step == len(val_loader):
					logger.info(f"Epoch {epoch} Step {step}/{len(val_loader)} | val_loss={avg_loss:.6f}")

		if val_valid_steps == 0:
			raise RuntimeError("整个 val epoch 都出现了非有限 loss，请检查数据和训练配置")

		val_loss = val_running_total_loss / val_valid_steps
		val_metrics = {name: value / val_valid_steps for name, value in val_running_metrics.items()}
		logger.info(
			f"Epoch {epoch} 验证损失明细: "
			+ " | ".join(f"{name}={value:.6f}" for name, value in sorted(val_metrics.items()))
		)

		if val_loss < best_val_loss:
			previous_best = best_val_loss
			best_val_loss = val_loss
			checkpoint_path = checkpoints_dir / f"e2ecr_epoch{epoch}_val{best_val_loss:.6f}.pth"
			torch.save(
				{
					"epoch": epoch,
					"best_val_loss": best_val_loss,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"model_config": asdict(model_config),
				},
				checkpoint_path,
			)
			if best_checkpoint_path is not None and best_checkpoint_path.exists():
				best_checkpoint_path.unlink()
			best_checkpoint_path = checkpoint_path
			logger.info(
				f"Epoch {epoch} 获得更优验证指标 | val_loss={val_loss:.6f} | 上一最佳={previous_best:.6f}"
			)
			logger.info(f"已保留最佳权重: {best_checkpoint_path}")
		else:
			logger.info(
				f"Epoch {epoch} 未刷新最佳验证指标 | val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
			)

	logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
	if best_checkpoint_path is not None:
		logger.info(f"最佳权重文件: {best_checkpoint_path}")