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
NUM_EPOCHS = 300
BATCH_SIZE = 12
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
MAX_CANDIDATES_NUM = 500
ALPHA = 0.05
BETA = 0.06
LAMBDA_REG = 1e-3
INFERENCE_SCORE_THRESHOLD = 0.3
VAL_DISTANCE_THRESHOLD = 12.0

def e2ecr_train(checkpoints_save_dir, logger):
	"""E2ECR 训练入口。

	这个函数故意写成单函数、平铺直叙的形式：
	1. 先构建数据集、数据加载器、模型和优化器。
	2. 再直接执行每个 epoch 的训练与验证。
	3. 损失计算、匈牙利匹配、日志统计、最佳权重保存都在这里完成。

	这样做的目的不是追求可扩展性，而是让脚本执行路径足够直接，
	方便按顺序阅读和逐行调试。
	"""
	# 先准备输出目录，后续每次验证变好时会把当前最佳权重写到这里。
	checkpoints_dir = Path(checkpoints_save_dir)
	checkpoints_dir.mkdir(parents=True, exist_ok=True)

	# 数据集对象只接收数据根目录和数据集类型，split 文件在数据集内部自行解析。
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

	# collate_fn 只负责把样本整理成 list，不在这里做 padding。
	# 真正的批量 padding 与裁回逻辑已经下沉到模型 forward 中完成。
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

	# 模型本身现在只输出三张预测图：
	# reg: 每个像素位置对应的点偏移；
	# det: 前景/背景二分类 logits；
	# cls: 细胞类别 logits。
	# 损失如何计算，完全在训练脚本里显式展开。
	model_config = E2ECRConfig(num_classes=get_e2ecr_num_classes(DATASET_TYPE))
	model = build_e2ecr(model_config).to(DEVICE)
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

	best_val_loss = float("inf")
	best_val_f1 = float("-inf")
	best_checkpoint_path: Path | None = None

	logger.info(f"训练设备: {DEVICE}")
	logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")
	logger.info(f"数据集类型: {DATASET_TYPE} | 数据集路径: {DATA_ROOT}")
	logger.info(f"类别数: {get_e2ecr_num_classes(DATASET_TYPE)}")
	logger.info(
		f"训练配置: epochs={NUM_EPOCHS}, train_batch_size={BATCH_SIZE}, val_batch_size={VAL_BATCH_SIZE}, lr={LEARNING_RATE}, crop_size={TRAIN_CROP_SIZE}"
	)
	logger.info(f"模型配置: {asdict(model_config)}")

	# 外层循环按 epoch 推进：每个 epoch 先训练，再按设定频率做验证。
	for epoch in range(1, NUM_EPOCHS + 1):
		# 训练阶段打开 BN/Dropout 的训练行为，并重置本 epoch 的累计量。
		model.train()
		train_running_total_loss = 0.0
		train_running_metrics = {"loss_cls": 0.0, "loss_det": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
		train_valid_steps = 0
		train_progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

		for step, (images, targets) in enumerate(train_progress_bar, start=1):
			# 图像和标注统一搬到目标设备。target 中既有 tensor，也有 image_name 这类字符串，
			# 所以这里只对 tensor 字段做 to(device)。
			images = [image.to(DEVICE) for image in images]
			targets = [
				{key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in target.items()}
				for target in targets
			]

			# KEY：前向传播
			# 前向传播后得到的是每张图的原始预测图，而不是已经解码好的点。
			outputs = model(images)
			# 先在 batch 维度上累计三项子损失，后面再除以 batch size。
			loss_dict = {
				"loss_cls": torch.zeros((), device=DEVICE),
				"loss_det": torch.zeros((), device=DEVICE),
				"loss_reg": torch.zeros((), device=DEVICE),
			}

			for output, target in zip(outputs, targets):
				# 每张图都会产生三张像素级预测图。
				reg_map = output["reg"]
				det_map = output["det"]
				cls_map = output["cls"]
				image_height, image_width = det_map.shape[-2], det_map.shape[-1]

				# 将 [C, H, W] 形式的预测图展平到 [H*W, C]，
				# 这样后面每一行都对应一个候选像素位置，便于统一计算。
				reg_logits = reg_map.permute(1, 2, 0).reshape(-1, 2)
				det_logits = det_map.permute(1, 2, 0).reshape(-1, 2)
				cls_logits = cls_map.permute(1, 2, 0).reshape(-1, model_config.num_classes)

				# 为每个像素生成其原始网格坐标，再加上回归头预测的偏移量，
				# 得到最终的候选点坐标 pred_points。
				y_coords = torch.arange(image_height, device=DEVICE, dtype=reg_logits.dtype)
				x_coords = torch.arange(image_width, device=DEVICE, dtype=reg_logits.dtype)
				y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
				base_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
				pred_points = base_points + reg_logits

				# det_probs[:, 1] 是“该像素位置像不像一个细胞点”的前景概率。
				# cls_probs 则描述这个位置属于哪个细胞类别的概率分布。
				det_probs = torch.softmax(det_logits, dim=1)
				obj_scores = det_probs[:, 1]
				cls_probs = torch.softmax(cls_logits, dim=1)

				# gt_points / gt_labels 是该图中所有真实点及其类别。
				# det_targets 默认全为背景，只有成功匹配到真实点的位置才会被置为前景。
				gt_points = target["points"]
				gt_labels = target["labels"]
				det_targets = torch.zeros(det_logits.shape[0], dtype=torch.long, device=DEVICE)
				# 当一张图没有真实点，或者后面没有形成有效匹配时，
				# 回归损失和分类损失保持为 0，但仍保留检测损失监督背景。
				zero_loss = det_logits.sum() * 0.0
				loss_reg = zero_loss
				loss_cls = zero_loss

				# KEY：匈牙利算法
				# 匈牙利匹配只在存在真实点时才需要执行。
				# 为了避免在整张图的全部像素上做匹配，这里先按目标性分数选一批候选点。
				if gt_points.shape[0] > 0:
					pred_num = obj_scores.shape[0]	# 预测的细胞个数
					candidate_scores, candidate_indices = torch.topk(
						obj_scores,
						k=MAX_CANDIDATES_NUM,
						largest=True,
						sorted=False,
					)
					candidate_points = pred_points[candidate_indices]
					candidate_cls_probs = cls_probs[candidate_indices]
					# 代价函数由三部分组成：
					# 1. 点之间的欧氏距离；
					# 2. 检测头给出的前景得分；
					# 3. 分类头给出真实类别的置信度。
					# 距离越小越好，前景/分类分数越大越好，因此后两项在代价里用减号。
					distance_matrix = torch.cdist(candidate_points, gt_points, p=2)
					class_score_matrix = candidate_cls_probs[:, gt_labels]
					cost_matrix = ALPHA * distance_matrix - candidate_scores.unsqueeze(1) - class_score_matrix

					# SciPy 的 linear_sum_assignment 会返回一组一对一匹配结果。
					# 匹配上的候选点会被视为正样本，同时用于回归和分类监督。
					matched_candidate_rows, matched_gt_cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
					if len(matched_candidate_rows) > 0:
						matched_candidate_rows = torch.as_tensor(matched_candidate_rows, dtype=torch.long, device=DEVICE)
						matched_gt_cols = torch.as_tensor(matched_gt_cols, dtype=torch.long, device=DEVICE)
						matched_indices = candidate_indices[matched_candidate_rows]
						det_targets[matched_indices] = 1
						loss_reg = F.mse_loss(pred_points[matched_indices], gt_points[matched_gt_cols], reduction="mean")
						loss_cls = F.cross_entropy(cls_logits[matched_indices], gt_labels[matched_gt_cols])

				# 检测损失在全部像素位置上计算。
				# 这里用 beta 对正负样本重新加权，减轻前景点稀少带来的类别不平衡问题。
				det_losses = F.cross_entropy(det_logits, det_targets, reduction="none")
				det_weights = torch.full_like(det_losses, 1.0 - BETA)
				det_weights[det_targets == 1] = BETA
				loss_det = (det_losses * det_weights).mean()

				# 先累计单张图损失，后面统一对 batch 求平均。
				loss_dict["loss_reg"] = loss_dict["loss_reg"] + loss_reg
				loss_dict["loss_det"] = loss_dict["loss_det"] + loss_det
				loss_dict["loss_cls"] = loss_dict["loss_cls"] + loss_cls

			# batch 内各图平均后，再按照论文/设定中的权重组合总损失。
			batch_size = max(1, len(outputs))
			loss_dict = {name: value / batch_size for name, value in loss_dict.items()}
			loss_dict["loss_total"] = LAMBDA_REG * loss_dict["loss_reg"] + loss_dict["loss_det"] + loss_dict["loss_cls"]
			total_loss = loss_dict["loss_total"]

			# 若当前 batch 已经数值发散，则直接跳过，避免把异常梯度传回模型。
			if not torch.isfinite(total_loss):
				logger.warning(
					f"Epoch {epoch} train Step {step}/{len(train_loader)} 出现非有限 loss，跳过该 batch: "
					+ " | ".join(
						f"{name}={value.detach().item():.6f}" for name, value in sorted(loss_dict.items())
					)
				)
				optimizer.zero_grad()
				continue

			# 标准训练步骤：清梯度、反传、梯度裁剪、参数更新。
			optimizer.zero_grad()
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
			optimizer.step()

			# 下面这些统计量只用于日志和 epoch 级平均值展示。
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

		# 允许通过 EVAL_INTERVAL_EPOCHS 控制验证频率，避免每轮都验证带来的额外开销。
		if epoch % EVAL_INTERVAL_EPOCHS != 0 and epoch != NUM_EPOCHS:
			continue

		# KEY：验证阶段
		# 验证阶段与训练阶段的主要区别是：
		# 1. 不做反向传播；
		# 2. 不更新参数；
		# 3. 仍然用完全相同的损失定义来评估当前模型质量。
		model.eval()
		val_running_total_loss = 0.0
		val_running_metrics = {"loss_cls": 0.0, "loss_det": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
		val_valid_steps = 0
		val_tp = 0
		val_fp = 0
		val_fn = 0
		val_progress_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False)

		with torch.no_grad():
			for step, (images, targets) in enumerate(val_progress_bar, start=1):
				# 验证时仍旧逐 batch 前向，只是整个过程放在 no_grad 下，节省显存和时间。
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
					# 下面这段与训练阶段保持一致，保证 train / val 的损失口径完全相同。
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

					# 验证集也走同样的候选点筛选和匈牙利匹配逻辑，
					# 这样得到的 val_loss 才能和 train_loss 对齐比较。
					if gt_points.shape[0] > 0:
						pred_num = obj_scores.shape[0]	# 预测的细胞个数
						candidate_scores, candidate_indices = torch.topk(
							obj_scores,
							k=MAX_CANDIDATES_NUM,
							largest=True,
							sorted=False,
						)
						candidate_points = pred_points[candidate_indices]
						candidate_cls_probs = cls_probs[candidate_indices]
						distance_matrix = torch.cdist(candidate_points, gt_points, p=2)
						class_score_matrix = candidate_cls_probs[:, gt_labels]
						cost_matrix = ALPHA * distance_matrix - candidate_scores.unsqueeze(1) - class_score_matrix

						matched_candidate_rows, matched_gt_cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
						if len(matched_candidate_rows) > 0:
							matched_candidate_rows = torch.as_tensor(matched_candidate_rows, dtype=torch.long, device=DEVICE)
							matched_gt_cols = torch.as_tensor(matched_gt_cols, dtype=torch.long, device=DEVICE)
							matched_indices = candidate_indices[matched_candidate_rows]
							det_targets[matched_indices] = 1
							loss_reg = F.mse_loss(pred_points[matched_indices], gt_points[matched_gt_cols], reduction="mean")
							loss_cls = F.cross_entropy(cls_logits[matched_indices], gt_labels[matched_gt_cols])

					# 检测损失依旧覆盖整张图，用于衡量背景和前景的整体分离效果。
					det_losses = F.cross_entropy(det_logits, det_targets, reduction="none")
					det_weights = torch.full_like(det_losses, 1.0 - BETA)
					det_weights[det_targets == 1] = BETA
					loss_det = (det_losses * det_weights).mean()

					loss_dict["loss_reg"] = loss_dict["loss_reg"] + loss_reg
					loss_dict["loss_det"] = loss_dict["loss_det"] + loss_det
					loss_dict["loss_cls"] = loss_dict["loss_cls"] + loss_cls

					# 验证时把所有密集预测先转成点，再按置信度阈值筛掉低质量预测。
					# 这里不再做 NMS，而是直接把保留下来的所有预测与 GT 做一对一距离匹配。
					decoded_det_probs = torch.softmax(det_logits, dim=1)
					decoded_obj_scores = decoded_det_probs[:, 1]
					decoded_cls_probs = torch.softmax(cls_logits, dim=1)
					decoded_cls_scores, decoded_pred_labels = torch.max(decoded_cls_probs, dim=1)
					decoded_final_scores = decoded_obj_scores * decoded_cls_scores
					keep_mask = decoded_final_scores >= INFERENCE_SCORE_THRESHOLD

					if keep_mask.sum().item() > 0:
						decoded_pred_points = pred_points[keep_mask].detach().cpu().numpy()
						decoded_pred_labels = decoded_pred_labels[keep_mask].detach().cpu().numpy()
					else:
						decoded_pred_points = gt_points.new_zeros((0, 2)).detach().cpu().numpy()
						decoded_pred_labels = gt_labels.new_zeros((0,), dtype=torch.long).detach().cpu().numpy()

					# 匹配规则：类别一致且欧氏距离小于阈值时，预测和 GT 才允许配对。
					# 一个预测最多匹配一个 GT，一个 GT 也最多匹配一个预测。
					gt_points_np = gt_points.detach().cpu().numpy()
					gt_labels_np = gt_labels.detach().cpu().numpy()
					pairs: list[tuple[float, int, int]] = []
					for pred_index in range(decoded_pred_points.shape[0]):
						for gt_index in range(gt_points_np.shape[0]):
							if int(decoded_pred_labels[pred_index]) != int(gt_labels_np[gt_index]):
								continue
							distance = float(((decoded_pred_points[pred_index] - gt_points_np[gt_index]) ** 2).sum() ** 0.5)
							if distance <= VAL_DISTANCE_THRESHOLD:
								pairs.append((distance, pred_index, gt_index))

					pairs.sort(key=lambda item: item[0])
					matched_pred_indices: set[int] = set()
					matched_gt_indices: set[int] = set()
					for _, pred_index, gt_index in pairs:
						if pred_index in matched_pred_indices or gt_index in matched_gt_indices:
							continue
						matched_pred_indices.add(pred_index)
						matched_gt_indices.add(gt_index)

					val_tp += len(matched_pred_indices)
					val_fp += int(decoded_pred_points.shape[0]) - len(matched_pred_indices)
					val_fn += int(gt_points_np.shape[0]) - len(matched_pred_indices)

				batch_size = max(1, len(outputs))
				loss_dict = {name: value / batch_size for name, value in loss_dict.items()}
				loss_dict["loss_total"] = LAMBDA_REG * loss_dict["loss_reg"] + loss_dict["loss_det"] + loss_dict["loss_cls"]
				total_loss = loss_dict["loss_total"]

				# 验证阶段同样保护数值稳定性，避免某个坏 batch 直接污染整轮统计。
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
		val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
		val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
		val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0
		logger.info(
			f"Epoch {epoch} 验证损失明细: "
			+ " | ".join(f"{name}={value:.6f}" for name, value in sorted(val_metrics.items()))
		)
		logger.info(
			f"Epoch {epoch} 验证分类感知指标: precision={val_precision:.6f} | recall={val_recall:.6f} | f1={val_f1:.6f}"
		)

		# 这里只保留“当前最好”的权重文件。
		# 一旦验证损失刷新，就删除上一份最佳权重，避免目录里累积过多 checkpoint。
		if val_f1 > best_val_f1:
			previous_best_f1 = best_val_f1
			best_val_f1 = val_f1
			best_val_loss = val_loss
			checkpoint_path = checkpoints_dir / f"e2ecr_epoch{epoch}_f1{best_val_f1:.6f}.pth"
			torch.save(
				{
					"epoch": epoch,
					"best_val_loss": best_val_loss,
					"best_val_f1": best_val_f1,
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
				f"Epoch {epoch} 获得更优验证指标 | val_f1={val_f1:.6f} | 上一最佳={previous_best_f1:.6f} | 对应val_loss={val_loss:.6f}"
			)
			logger.info(f"已保留最佳权重: {best_checkpoint_path}")
		else:
			logger.info(
				f"Epoch {epoch} 未刷新最佳验证指标 | val_f1={val_f1:.6f} | best_val_f1={best_val_f1:.6f} | val_loss={val_loss:.6f}"
			)

	logger.info(f"训练完成，最佳验证 F1: {best_val_f1:.6f} | 对应验证损失: {best_val_loss:.6f}")
	if best_checkpoint_path is not None:
		logger.info(f"最佳权重文件: {best_checkpoint_path}")