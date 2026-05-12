from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MoNuSACDataset, detection_collate_fn, get_num_classes
from faster_rcnn import FasterRCNNConfig, build_faster_rcnn


DATA_ROOT = Path("data") / "CoNSeP"
ANNOTATIONS_CSV = DATA_ROOT / "annotations" / "boxes.csv"
CLASSES_CSV = DATA_ROOT / "metadata" / "classes.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 300
BATCH_SIZE = 1
NUM_WORKERS = 0 if torch.platform.system() == "Windows" else 16
LEARNING_RATE = 0.0005   # 学习率。当前 batch size 很小，检测模型需要更保守的步长。
MOMENTUM = 0.9  # 学习率动量
WEIGHT_DECAY = 0.0005   # 学习率衰减
LR_STEP_SIZE = 8
LR_GAMMA = 0.1
EVAL_INTERVAL_EPOCHS = 1    # 验证周期
PRINT_FREQ = 10     # 打印频率
GRAD_CLIP_NORM = 10.0

PRETRAINED_DETECTOR = False
PRETRAINED_BACKBONE = False
TRAINABLE_BACKBONE_LAYERS = 5
MIN_SIZE = 1000
MAX_SIZE = 1000
BOX_SCORE_THRESH = 0.7
BOX_NMS_THRESH = 0.5
BOX_DETECTIONS_PER_IMG = 1000
RPN_ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5


def faster_rcnn_train(checkpoints_save_dir, logger):
    checkpoints_dir = Path(checkpoints_save_dir)
    train_dataset = MoNuSACDataset(DATA_ROOT, ANNOTATIONS_CSV, split="train")
    val_dataset = MoNuSACDataset(DATA_ROOT, ANNOTATIONS_CSV, split="val")
    num_classes = get_num_classes(CLASSES_CSV)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )

    model_config = FasterRCNNConfig(
        num_classes=num_classes,
        pretrained_detector=PRETRAINED_DETECTOR,
        pretrained_backbone=PRETRAINED_BACKBONE,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=BOX_SCORE_THRESH,
        box_nms_thresh=BOX_NMS_THRESH,
        box_detections_per_img=BOX_DETECTIONS_PER_IMG,
        rpn_anchor_sizes=RPN_ANCHOR_SIZES,
        rpn_aspect_ratios=RPN_ASPECT_RATIOS,
    )
    model = build_faster_rcnn(model_config).to(DEVICE)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_loss = float("inf")
    best_checkpoint_path: Path | None = None

    logger.info(f"训练设备: {DEVICE}")
    logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")
    logger.info(f"类别数(含背景): {num_classes}")
    logger.info(f"训练配置: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    logger.info(
        "模型配置: "
        f"trainable_backbone_layers={TRAINABLE_BACKBONE_LAYERS}, "
        f"pretrained_detector={PRETRAINED_DETECTOR}, "
        f"pretrained_backbone={PRETRAINED_BACKBONE}",
    )
    logger.info(f"梯度裁剪阈值: {GRAD_CLIP_NORM}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_metrics = _train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            logger=logger,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{NUM_EPOCHS} 训练结束 | train_loss={train_loss:.6f} | lr={current_lr:.8f}",
        )
        logger.info(f"Epoch {epoch} 训练损失明细: {_format_metrics(train_metrics)}")

        if epoch % EVAL_INTERVAL_EPOCHS != 0 and epoch != NUM_EPOCHS:
            continue

        val_loss, val_metrics = _validate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=DEVICE,
            epoch=epoch,
            logger=logger,
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
                f"Epoch {epoch} 获得更优验证指标 | val_loss={val_loss:.6f} | 上一最佳={previous_best:.6f}",
            )
            logger.info(f"已保留最佳权重: {best_checkpoint_path}")
        else:
            logger.info(
                f"Epoch {epoch} 未刷新最佳验证指标 | val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}",
            )

    logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
    if best_checkpoint_path is not None:
        logger.info(f"最佳权重文件: {best_checkpoint_path}")


def _train_one_epoch(model, data_loader, optimizer, device, epoch, logger):
    model.train()
    running_total_loss = 0.0
    running_metrics: dict[str, float] = {}
    valid_steps = 0

    progress_bar = tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False)
    for step, (images, targets) in enumerate(progress_bar, start=1):
        images = [image.to(device) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_loss):
            logger.warning(
                f"Epoch {epoch} Step {step}/{len(data_loader)} 出现非有限 loss，跳过该 batch: "
                f"{_format_metrics({name: value.detach().item() for name, value in loss_dict.items()})}"
            )
            optimizer.zero_grad()
            continue

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
            logger.info(
                f"Epoch {epoch} Step {step}/{len(data_loader)} | train_loss={avg_loss:.6f}",
            )

    if valid_steps == 0:
        raise RuntimeError("整个训练 epoch 都出现了非有限 loss，请检查数据和训练配置")

    averaged_metrics = {name: value / valid_steps for name, value in running_metrics.items()}
    return running_total_loss / valid_steps, averaged_metrics


def _validate_one_epoch(model, data_loader, device, epoch, logger):
    was_training = model.training
    model.train()

    running_total_loss = 0.0
    running_metrics: dict[str, float] = {}
    valid_steps = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Val Epoch {epoch}", leave=False)
        for step, (images, targets) in enumerate(progress_bar, start=1):
            images = [image.to(device) for image in images]
            targets = [_move_target_to_device(target, device) for target in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(total_loss):
                logger.warning(
                    f"Epoch {epoch} 验证阶段出现非有限 loss，跳过该 batch: "
                    f"{_format_metrics({name: value.detach().item() for name, value in loss_dict.items()})}"
                )
                continue

            valid_steps += 1
            running_total_loss += total_loss.item()
            for name, value in loss_dict.items():
                running_metrics[name] = running_metrics.get(name, 0.0) + value.item()

            avg_loss = running_total_loss / valid_steps
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    if not was_training:
        model.eval()

    if valid_steps == 0:
        raise RuntimeError("整个验证 epoch 都出现了非有限 loss，请检查数据和训练配置")

    averaged_metrics = {name: value / valid_steps for name, value in running_metrics.items()}
    val_loss = running_total_loss / valid_steps
    logger.info(f"Epoch {epoch}/{NUM_EPOCHS} 验证结束 | val_loss={val_loss:.6f}")
    return val_loss, averaged_metrics


def _move_target_to_device(target, device):
    return {key: value.to(device) for key, value in target.items()}


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
    if previous_best_checkpoint_path is not None and previous_best_checkpoint_path.exists():
        previous_best_checkpoint_path.unlink()

    checkpoint_path = checkpoints_dir / f"epoch{epoch}_val_loss_{best_val_loss:.6f}.pth"
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),     # 模型权重
            "optimizer_state_dict": optimizer.state_dict(),     # 优化器状态
            "scheduler_state_dict": scheduler.state_dict(),     # 调度器状态
            "model_config": asdict(model_config),   # 模型配置
        },
        checkpoint_path,
    )
    return checkpoint_path


def _format_metrics(metrics):
    return ", ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
