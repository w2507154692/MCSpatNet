"""
opt.py

HoVer-Net 训练/验证的超参数与引擎配置。

由 config.py 根据 nr_type（类型数，含背景）与 mode（original / fast）
动态 import 本模块的 get_config()，再交给 run_train.TrainManager 解析执行。

配置结构：
  - phase_list：多阶段训练列表（默认 2 阶段：冻结编码器 → 全网络微调）
  - run_engine：train / valid 两个 RunEngine 的 DataLoader、run_step、回调链
"""

import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .targets import gen_targets, prep_sample
from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


# TODO: 是否仅用于训练配置、与推理配置拆分？
# TODO: 是否将全部选项改为字符串形式的函数名以便序列化/远程配置？
def get_config(nr_type, mode):
    """构建 HoVer-Net 完整的 model_config 字典。

    Args:
        nr_type: 细胞核类型数（含背景），与 config.py 中 nr_type 一致。
                 例如 CoNSeP 合并为 3 类细胞 + 背景时 nr_type=4。
        mode: 网络结构模式，'original'（270→80）或 'fast'（256→164）。

    Returns:
        dict: 含 phase_list、run_engine，供 TrainManager.run_once() 使用。
    """
    return {
        # ------------------------------------------------------------------
        # phase_list：按索引 0→N 顺序执行；每阶段可有不同的 epoch、batch、预训练来源
        # ! 各 phase 的 run_engine 结构相同（train + valid），但 run_info 可不同
        "phase_list": [
            {
                # ========== 第 1 阶段：冻结编码器，仅训练解码分支 ==========
                "run_info": {
                    # 本阶段涉及的网络名 -> 构造与优化配置（当前仅一个 "net"）
                    "net": {
                        # 延迟构造：lambda 在 run_train 里被调用时才实例化模型
                        "desc": lambda: create_model(
                            input_ch=3,
                            nr_types=nr_type,
                            freeze=True,  # 冻结 ResNet 编码器，只训 HoVer 解码与类型头
                            mode=mode,
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # 关键字需与 Adam 构造函数参数一致
                                "lr": 1.0e-4,  # 初始学习率
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # 学习率调度：每 25 个 epoch 将 lr 乘以 gamma（StepLR 默认 0.1）
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            # 各输出分支的损失加权（见 run_desc.train_step）
                            "loss": {
                                "np": {"bce": 1, "dice": 1},  # 核像素二分类
                                "hv": {"mse": 1, "msge": 1},  # 水平/垂直距离图 + 梯度 MSE
                                "tp": {"bce": 1, "dice": 1},  # 类型分类（nr_type>0 时）
                            },
                        },
                        # 预训练权重路径：
                        #   - 字符串路径：加载 ImageNet Preact-ResNet50 等
                        #   - -1：从上一 phase 的 log 目录自动找最新 checkpoint
                        #   - None：随机初始化（不加载预训练）
                        "pretrained": "../pretrained/ImageNet-ResNet50-Preact_pytorch.tar",
                        # 'pretrained': None,
                    },
                },
                # target_info：DataLoader 侧生成监督图；viz 用于 --view 调试可视化
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                # batch_size：键为 engine 名（train / valid）
                "batch_size": {"train": 16, "valid": 16,},
                "nr_epochs": 50,
            },
            {
                # ========== 第 2 阶段：解冻全网络，端到端微调 ==========
                "run_info": {
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3,
                            nr_types=nr_type,
                            freeze=False,  # 编码器与解码器一并训练
                            mode=mode,
                        ),
                        "optimizer": [
                            optim.Adam,
                            {
                                "lr": 1.0e-4,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # 从第 1 阶段保存的 checkpoint 续训（见 run_train.get_last_chkpt_path）
                        "pretrained": -1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 8,},  # 每 GPU 的 batch（显存占用更大）
                "nr_epochs": 50,
            },
        ],
        # ------------------------------------------------------------------
        # run_engine：train / valid 共用同一套 net_run_info，但回调与 run_step 不同
        # TODO: 数据集路径是否也应在此配置（当前在 config.py 的 train_dir_list）
        "run_engine": {
            "train": {
                # TODO: dataset 字段预留，当前实际路径在 config.py
                "dataset": "",  # 复合数据集等扩展预留
                "nr_procs": 16,  # DataLoader num_workers
                "run_step": train_step,  # 每 batch：前向、多分支 loss、反传
                "reset_per_run": False,
                # 回调按 Events 内列表顺序执行
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(),  # TODO: 每步记录梯度很慢（CPU↔GPU 拷贝）
                        ScalarMovingAverage(),  # 对 loss 等标量做 EMA，供进度条显示
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),  # 记录当前学习率
                        PeriodicSaver(per_n_epoch=10),  # 每 10 个 epoch 保存 net_epoch=*.tar
                        VisualizeOutput(viz_step_output),  # 训练样本可视化到 TensorBoard
                        LoggingEpochOutput(),  # 打印指标并写 stats.json
                        TriggerEngine("valid"),  # 触发 valid 引擎跑一轮
                        ScheduleLr(),  # 调用 lr_scheduler.step()
                    ],
                },
            },
            "valid": {
                "dataset": "",
                "nr_procs": 8,
                "run_step": valid_step,  # 仅前向，输出 prob_np、pred_hv、pred_tp 等
                "reset_per_run": True,  # * 每次 valid 结束清空累积，避免污染下一轮
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        AccumulateRawOutput(),  # 累积整 epoch 的 raw 预测与 GT
                    ],
                    Events.EPOCH_COMPLETED: [
                        # TODO: 能否预加载 proc 函数以减少开销
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        ),  # 计算 np_acc/dice、tp_dice_k、hv_mse 等
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
