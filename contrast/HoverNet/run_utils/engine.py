"""
engine.py

HoVer-Net 训练/验证循环引擎。

通过「事件 + 回调」组织 train/valid 流程：每个 epoch 遍历 DataLoader，
每步调用 run_step（如 train_step / valid_step），在固定事件点触发回调
（日志、保存 checkpoint、学习率调度、触发验证等）。

由 run_train.py 中的 TrainManager 创建 RunEngine 并注册 callbacks。
"""

import tqdm
from enum import Enum


####
class Events(Enum):
    """训练/验证生命周期中的事件类型，供回调注册与触发。"""

    EPOCH_STARTED = "epoch_started"  # 单个 epoch 开始
    EPOCH_COMPLETED = "epoch_completed"  # 单个 epoch 结束
    STEP_STARTED = "step_started"  # 单个 batch 开始
    STEP_COMPLETED = "step_completed"  # 单个 batch 结束（前向/反向已完成）
    STARTED = "started"  # 整个 run 开始（当前代码路径较少使用）
    COMPLETED = "completed"  # 整个 run 结束
    EXCEPTION_RAISED = "exception_raised"  # 异常（预留）


####
class State(object):
    """在事件回调之间传递的运行时状态容器。

    包含：当前 epoch/step 计数、单步输出、按 epoch 累积的输出、
    以及需写入 TensorBoard/日志的标量与图像等。
    """

    def __init__(self):
        # 由 TrainManager / run_once 注入的配置项
        self.logging = None  # 是否写 TensorBoard / stats.json
        self.log_dir = None  # 日志与 checkpoint 目录
        self.log_info = None  # 含 tfwriter、json_file 等

        # 内部计数器
        self.curr_epoch_step = 0  # 当前 epoch 内已处理的 batch 数
        self.curr_global_step = 0  # 从训练开始累计的全局 step（跨 epoch）
        self.curr_epoch = 0  # 当前 epoch 索引（从 0 开始）

        # TODO: [LOW] 进一步文档化各字段的写入方与消费方
        # 每步会被回调写入、用于日志/可视化的输出，按类型分桶：
        # - "scalar"：标量（loss 等），会打印并写入 TensorBoard
        # - "image"：图像类，需专门回调处理后再写入 TensorBoard

        # ! 键名需与序列化/日志模块支持的类型一致
        # TODO: 支持动态注册新的输出类型
        self.tracked_step_output = {
            "scalar": {},  # {变量名: 变量值}
            "image": {},
        }
        # TODO: 明确各回调读写哪些字段，避免隐式耦合

        self.epoch_accumulated_output = {}  # 当前 epoch 内累积的原始输出（如 valid 的 raw）

        # TODO: 支持每 N 个 epoch 对部分变量做软重置
        self.run_accumulated_output = []  # 历次 epoch 的 epoch_accumulated_output 列表

        # 当前 batch 上 run_step 的返回值（如 train_step 的 EMA、raw 可视化数据）
        # * 若改为 GAN 等训练方式，累积策略可能不同
        self.step_output = None

        self.global_state = None  # 跨多个 RunEngine 共享的状态（预留，如链式 train→valid）
        return

    def reset_variable(self):
        """每个 epoch 开始时清空步级跟踪与（按策略）运行级累积。"""
        # 清空本 epoch 的标量/图像跟踪 dict，保留分桶结构
        self.tracked_step_output = {k: {} for k in self.tracked_step_output.keys()}

        # TODO: [CRITICAL] 重构 valid 与 train 的累积策略差异
        if self.curr_epoch % self.pertain_n_epoch_output == 0:
            self.run_accumulated_output = []

        self.epoch_accumulated_output = {}

        # * 若改为 GAN 等训练方式，此处重置逻辑可能不同
        self.step_output = None
        return


####
class RunEngine(object):
    """单次 train 或 valid 的执行引擎。

    绑定：DataLoader、单步函数 run_step（train_step/valid_step）、
    run_info（网络/优化器等），以及按 Events 注册的回调列表。

    Args:
        engine_name: 引擎名称，通常为 'train' 或 'valid'
        dataloader: PyTorch DataLoader
        run_step: 每 batch 调用的函数，签名 (batch_data, step_run_info) -> dict
        run_info: 含 net、optimizer 等，传给 run_step 与回调
        log_info: 日志相关句柄（与 trainer.py 耦合，TODO: 后续可重构）
    """

    def __init__(
        self,
        engine_name=None,
        dataloader=None,
        run_step=None,
        run_info=None,
        log_info=None,  # TODO: 与 trainer.py 解耦
    ):

        # * 将构造参数挂到实例上，供 run() 与回调使用
        self.engine_name = engine_name
        self.run_step = run_step
        self.dataloader = dataloader

        # * 全局状态对象，所有 event handler 共享同一引用（勿随意替换为拷贝）
        self.state = State()
        self.state.attached_engine_name = engine_name  # TODO: 与 engine_name 重复？
        self.state.run_info = run_info
        self.state.log_info = log_info
        self.state.batch_size = dataloader.batch_size

        # TODO: [CRITICAL] 与 opt.py 中 run_engine 配置完全对齐
        # valid 引擎每 epoch 结束后会 reset run_accumulated_output（见 reset_variable）
        self.state.pertain_n_epoch_output = 1 if engine_name == "valid" else 1

        # 每种事件对应一个回调列表，按注册顺序依次执行
        self.event_handler_dict = {event: [] for event in Events}

        # TODO: 支持 train RunEngine 与 valid RunEngine 共享 global_state
        # （例如训练 epoch 结束触发验证时传递指标）

        self.terminate = False  # 预留：提前终止训练标志
        return

    def __reset_state(self):
        """重建 State 对象（保留 run_info 等引用）。当前主流程未调用。"""
        # TODO: 与 reset_variable 职责重叠，考虑合并
        new_state = State()
        new_state.attached_engine_name = self.state.attached_engine_name
        new_state.run_info = self.state.run_info
        new_state.log_info = self.state.log_info
        self.state = new_state
        return

    def __trigger_events(self, event):
        """按注册顺序执行某事件上的所有回调。"""
        for callback in self.event_handler_dict[event]:
            callback.run(self.state, event)
            # TODO: 回调异常时附带 handler 名称，便于定位
        return

    # TODO: 在 handler 间声明输出依赖（例如 B 必须在 A 之后且读取 A 写入的 state）
    def add_event_handler(self, event_name, handler):
        """向指定事件注册回调（如 LoggingEpochOutput、PeriodicSaver）。"""
        self.event_handler_dict[event_name].append(handler)

    # ! 未来可考虑将 run() 上移到 trainer.py，RunEngine 只负责单 epoch 步进
    def run(self, nr_epoch=1, shared_state=None, chained=False):
        """执行 nr_epoch 个 epoch 的 DataLoader 循环。

        Args:
            nr_epoch: 本引擎要跑的 epoch 数（train/valid 在 opt 里分别配置）
            shared_state: 跨引擎共享状态（预留）
            chained: True 时不打印 "EPOCH n" 标题，并将 curr_epoch 置 0
                     （用于被其他引擎链式调用时的场景）
        """

        # TODO: 整理 chained / shared_state 的正式语义
        if chained:
            self.state.curr_epoch = 0
        self.state.global_state = shared_state

        while self.state.curr_epoch < nr_epoch:
            self.state.reset_variable()  # * 每 epoch 初清空 EMA、epoch 累积等

            if not chained:
                print("----------------EPOCH %d" % (self.state.curr_epoch + 1))

            self.__trigger_events(Events.EPOCH_STARTED)

            pbar_format = (
                "Processing: |{bar}| "
                "{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            )
            if self.engine_name == "train":
                # 训练进度条额外显示当前 batch loss 与 EMA loss
                pbar_format += (
                    "Batch = {postfix[1][Batch]:0.5f}|" "EMA = {postfix[1][EMA]:0.5f}"
                )
                # * 勿随意改进度条字符集，可能导致 tqdm 显示异常
                pbar = tqdm.tqdm(
                    total=len(self.dataloader),
                    leave=True,
                    initial=0,
                    bar_format=pbar_format,
                    ascii=True,
                    postfix=["", dict(Batch=float("NaN"), EMA=float("NaN"))],
                )
            else:
                pbar = tqdm.tqdm(
                    total=len(self.dataloader),
                    leave=True,
                    bar_format=pbar_format,
                    ascii=True,
                )

            for data_batch in self.dataloader:
                self.__trigger_events(Events.STEP_STARTED)

                # run_step 的第二个参数：网络/优化器 dict + 当前 epoch/step 元信息
                step_run_info = [
                    self.state.run_info,
                    {
                        "epoch": self.state.curr_epoch,
                        "step": self.state.curr_global_step,
                    },
                ]
                step_output = self.run_step(data_batch, step_run_info)
                self.state.step_output = step_output

                # STEP_COMPLETED：ScalarMovingAverage、AccumulateRawOutput 等多在此阶段
                self.__trigger_events(Events.STEP_COMPLETED)
                self.state.curr_global_step += 1
                self.state.curr_epoch_step += 1

                if self.engine_name == "train":
                    pbar.postfix[1]["Batch"] = step_output["EMA"]["overall_loss"]
                    pbar.postfix[1]["EMA"] = self.state.tracked_step_output["scalar"][
                        "overall_loss"
                    ]
                pbar.update()
            pbar.close()  # 先关闭进度条，再执行 epoch 级日志/保存，避免输出错乱

            self.state.curr_epoch += 1
            # EPOCH_COMPLETED：PeriodicSaver、TriggerEngine('valid')、ScheduleLr 等
            self.__trigger_events(Events.EPOCH_COMPLETED)

            # TODO: [CRITICAL] 与 valid 回调的 proc_valid_step_output 协议对齐
            self.state.run_accumulated_output.append(
                self.state.epoch_accumulated_output
            )

        return
