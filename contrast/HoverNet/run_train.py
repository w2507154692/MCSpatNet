"""run_train.py

HoVer-Net 主训练脚本。

usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

options:
  -h --help       显示帮助信息。
  --version       显示版本号。
  --gpu=<id>      指定使用的 GPU，多个 ID 用逗号分隔。[default: 0,1,2,3]
  --view=<dset>   仅可视化增强后的样本，不进行训练。可选 'train' 或 'valid'。
"""

import cv2

# 禁用 OpenCV 内部多线程，避免与 PyTorch DataLoader 的多进程冲突
cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil

import matplotlib
import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: 后续可切换为 DistributedDataParallel
from torch.utils.data import DataLoader

from config import Config
from dataloader.train_loader import FileLoader
from misc.pretrained import load_pretrained_state_dict, resolve_pretrained_path
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
    get_device,
)


# 必须定义在模块顶层：Windows 下 DataLoader 使用 spawn 时，
# 每个 worker 子进程会单独 import 本模块，worker_init_fn 需可被 pickle。
# * 每个 worker 必须独立初始化 augmentor，否则可能复用相同的随机数生成器
def worker_init_fn(worker_id):
    # ! 为保证随机种子链可复现，这里使用 torch 随机数，而不是 numpy
    # 主线程中的 torch RNG 会生成基础 seed，DataLoader 在每个 epoch 开始时复制该 seed；
    # 随后 worker 被 spawn 时，在此函数中再次为 worker 设置独立 seed
    worker_info = torch.utils.data.get_worker_info()
    # 若希望增强更随机，可将 torch.randint 换成 np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # 获取被复制到当前 worker 进程中的 dataset 实例，
    # 再为每个 worker 设置独立的数据增强随机种子
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


####
class TrainManager(Config):
    """训练管理器：继承 Config，负责数据集可视化或完整训练流程的启动。"""

    def __init__(self):
        super().__init__()
        return

    ####
    def view_dataset(self, mode="train"):
        """
        可视化增强后的训练/验证样本（调试用）。

        若在无图形界面的服务器上运行，可将 plt.show() 改为 plt.savefig()。
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)
        # TODO: 若 train / valid 阶段需要不同的标注可视化方式，可在此扩展
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_data in dataloader:
            # 将 Tensor 转为 Numpy，供可视化函数使用
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.show()
        self.nr_gpus = -1
        return

    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        """构建指定阶段（train / valid）的 DataLoader。"""
        # debug 模式下不使用多进程，便于单步调试
        nr_procs = nr_procs if not self.debug else 0

        # ! 当前硬编码假设输入文件为 .npy 格式
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # 固定排序，保证每次运行输入顺序一致

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            nr_types=getattr(self, "nr_type", None),
            # 单进程时直接在主进程初始化 augmentor；多进程时由 worker_init_fn 初始化
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",  # 训练时丢弃不足一个 batch 的尾部样本
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """执行单个训练阶段（phase）的完整流程：建 loader、网络、优化器、回调并开训。"""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # 先创建空的统计文件
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        # 为 train / valid 等各个 runner 分别创建 DataLoader
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            """从上一阶段的 stats.json 中读取最后一个 epoch 的 checkpoint 路径。"""
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        # TODO: 扩展预训练权重加载或断点续训逻辑
        # 解析 config 中的网络与优化器配置
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: 为不同 run 自定义网络结构打印？
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            hovernet_root = os.path.dirname(os.path.abspath(__file__))

            if pretrained_path == -1:
                # * 依赖日志目录命名规则；若日志格式变更，此逻辑可能失效
                pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                net_state_dict = torch.load(pretrained_path)["desc"]
            elif pretrained_path is not None:
                pretrained_path = resolve_pretrained_path(
                    pretrained_path,
                    base_dir=hovernet_root,
                    allow_default=prev_log_dir is None,
                )
                net_state_dict = load_pretrained_state_dict(pretrained_path)
            elif prev_log_dir is None:
                # 未显式指定时，第一阶段默认下载并加载 ImageNet 预训练 ResNet50
                pretrained_path = resolve_pretrained_path(
                    None, base_dir=hovernet_root, allow_default=True
                )
                net_state_dict = load_pretrained_state_dict(pretrained_path)

            if pretrained_path is not None:
                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s"
                    % (colored_word, pretrained_path)
                )

                # load_state_dict 返回 (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * 如需排查权重加载问题，可取消下面两行注释
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * 在部分 DGX 单卡环境下，DataParallel 包装可能异常偏慢，原因待查 (?)
            if self.use_cuda and torch.cuda.device_count() > 1:
                net_desc = DataParallel(net_desc)
            net_desc = net_desc.to(self.device)
            # print(net_desc) # * 是否打印完整网络结构，可按需取消注释
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: 扩展外部 scheduler / augmentation hook
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: 统一外部 hook 的 API
                "extra_info": net_info["extra_info"],
            }

        # 解析 run_engine 配置，确认至少存在 train 引擎
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # 初始化各 runner，并在之后挂载 callback
        # * 所有 engine 共享同一份 net_run_info（网络 / 优化器 / scheduler）
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        # 为每个 runner 注册事件回调（如 valid 结束后保存模型、记录指标等）
        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # 取主训练 runner 并写入日志相关状态
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # 启动训练循环
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run(self):
        """主入口：支持多阶段训练（phase_list）或交叉验证等扩展流程。"""
        self.use_cuda = torch.cuda.is_available()
        self.device = get_device()
        if self.use_cuda:
            self.nr_gpus = max(1, torch.cuda.device_count())
            print("Detect #GPUS: %d" % self.nr_gpus)
            print("Using device: cuda")
        else:
            self.nr_gpus = 1
            print("CUDA not available, using CPU for training.")

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                # 多阶段训练时，每个 phase 单独建子目录
                save_path = self.log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


####
if __name__ == "__main__":
    args = docopt(__doc__, version="HoVer-Net v1.0")
    trainer = TrainManager()

    if args["--view"]:
        # 仅可视化数据增强结果，不进入训练
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        else:
            print("CUDA not available, ignoring --gpu and using CPU.")
        trainer.run()
