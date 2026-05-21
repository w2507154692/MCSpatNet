from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import platform

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from E2ECR import E2ECRConfig, build_e2ecr
from dataset import build_e2ecr_dataset, e2ecr_collate_fn, get_e2ecr_num_classes


DATA_ROOT = Path("data") / "BRCA-M2C"
DATASET_TYPE = "brca-m2c"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200
BATCH_SIZE = 4
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
		batch_size=BATCH_SIZE,
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
		f"训练配置: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, crop_size={TRAIN_CROP_SIZE}"
	)
	logger.info(f"模型配置: {asdict(model_config)}")

	for epoch in range(1, NUM_EPOCHS + 1):
		train_loss, train_metrics = _run_one_epoch(
			model=model,
			data_loader=train_loader,
			optimizer=optimizer,
			device=DEVICE,
			epoch=epoch,
			logger=logger,
			phase="train",
		)
		scheduler.step()
		current_lr = optimizer.param_groups[0]["lr"]
		logger.info(f"Epoch {epoch}/{NUM_EPOCHS} 训练结束 | train_loss={train_loss:.6f} | lr={current_lr:.8f}")
		logger.info(f"Epoch {epoch} 训练损失明细: {_format_metrics(train_metrics)}")

		if epoch % EVAL_INTERVAL_EPOCHS != 0 and epoch != NUM_EPOCHS:
			continue

		val_loss, val_metrics = _run_one_epoch(
			model=model,
			data_loader=val_loader,
			optimizer=None,
			device=DEVICE,
			epoch=epoch,
			logger=logger,
			phase="val",
		)
		logger.info(f"Epoch {epoch} 验证损失明细: {_format_metrics(val_metrics)}")

		if val_loss < best_val_loss:
			previous_best = best_val_loss
			best_val_loss = val_loss
			best_checkpoint_path = _save_best_checkpoint(
				checkpoints_dir=checkpoints_dir,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				best_val_loss=best_val_loss,
				previous_best_checkpoint_path=best_checkpoint_path,
				model_config=model_config,
			)
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


def _run_one_epoch(model, data_loader, optimizer, device, epoch, logger, phase):
	is_train = phase == "train"
	model.train(mode=is_train)

	running_total_loss = 0.0
	running_metrics: dict[str, float] = {}
	valid_steps = 0
	context = torch.enable_grad() if is_train else torch.no_grad()
	progress_bar = tqdm(data_loader, desc=f"{phase.capitalize()} Epoch {epoch}", leave=False)

	with context:
		for step, (images, targets) in enumerate(progress_bar, start=1):
			images = [image.to(device) for image in images]
			targets = [_move_target_to_device(target, device) for target in targets]

			loss_dict = model(images, targets)
			total_loss = sum(loss for loss in loss_dict.values())

			if not torch.isfinite(total_loss):
				logger.warning(
					f"Epoch {epoch} {phase} Step {step}/{len(data_loader)} 出现非有限 loss，跳过该 batch: "
					f"{_format_metrics({name: value.detach().item() for name, value in loss_dict.items()})}"
				)
				if is_train:
					optimizer.zero_grad()
				continue

			if is_train:
				optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
				optimizer.step()

			valid_steps += 1
			running_total_loss += total_loss.item()
			for name, value in loss_dict.items():
				running_metrics[name] = running_metrics.get(name, 0.0) + value.item()

			avg_loss = running_total_loss / valid_steps
			progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
			if step % PRINT_FREQ == 0 or step == len(data_loader):
				logger.info(f"Epoch {epoch} Step {step}/{len(data_loader)} | {phase}_loss={avg_loss:.6f}")

	if valid_steps == 0:
		raise RuntimeError(f"整个 {phase} epoch 都出现了非有限 loss，请检查数据和训练配置")

	averaged_metrics = {name: value / valid_steps for name, value in running_metrics.items()}
	return running_total_loss / valid_steps, averaged_metrics


def _move_target_to_device(target, device):
	return {
		key: value.to(device) if torch.is_tensor(value) else value
		for key, value in target.items()
	}


def _save_best_checkpoint(
	checkpoints_dir,
	model,
	optimizer,
	scheduler,
	epoch,
	best_val_loss,
	previous_best_checkpoint_path,
	model_config,
):
	checkpoints_dir.mkdir(parents=True, exist_ok=True)
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
	if previous_best_checkpoint_path is not None and previous_best_checkpoint_path.exists():
		previous_best_checkpoint_path.unlink()
	return checkpoint_path


def _format_metrics(metrics: dict[str, float]) -> str:
	return " | ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))