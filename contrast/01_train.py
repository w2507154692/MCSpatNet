import os
import glob
import shutil
import random
import numpy as np
import torch
from datetime import datetime

CHECKPOINTS_ROOT_DIR = './exp'      # 所有训练输出的根目录
CHECKPOINTS_FOLDER_NAME = 'exp21_e2ecr_consep'     # 当前训练实例的输出文件夹名称，将创建在 <checkpoints_root_dir> 下
SEED = 42
TYPE = 'e2ecr'


class TrainLogger:
    """训练日志封装。

    作用：
    - 统一在终端和日志文件中输出信息。
    - 避免训练代码中直接操作原始文件句柄。
    - 后续若要扩展 warning/error 或接入 logging，修改这里即可。
    """

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self._file = open(log_file_path, 'w', encoding='utf-8')

    def _write(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f'[{timestamp}] [{level}] {message}'
        print(full_message)
        self._file.write(full_message + '\n')
        self._file.flush()

    def info(self, message):
        self._write('INFO', message)

    def warning(self, message):
        self._write('WARNING', message)

    def error(self, message):
        self._write('ERROR', message)

    def close(self):
        if not self._file.closed:
            self._file.close()

# 备份代码到实验目录下
def backup_python_files(experiment_dir):
    project_root = os.path.dirname(os.path.abspath(__file__))
    code_backup_dir = os.path.join(experiment_dir, 'code')
    os.makedirs(code_backup_dir, exist_ok=True)
    for pattern in ('*.py', '*.ipynb'):
        for filepath in glob.glob(os.path.join(project_root, pattern)):
            shutil.copy2(filepath, os.path.join(code_backup_dir, os.path.basename(filepath)))

# 配置种子
def configure_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    checkpoints_save_dir   = os.path.join(CHECKPOINTS_ROOT_DIR, CHECKPOINTS_FOLDER_NAME)

    if not os.path.exists(CHECKPOINTS_ROOT_DIR):
        os.mkdir(CHECKPOINTS_ROOT_DIR)
    if not os.path.exists(checkpoints_save_dir):
        os.mkdir(checkpoints_save_dir)

    # 训练开始前备份项目根目录下的 Python 源码，便于回溯当前实验配置和实现版本。
    backup_python_files(checkpoints_save_dir)

    # log_file_path：训练日志文件的保存路径。
    log_file_path = os.path.join(checkpoints_save_dir, f'train_log.txt') 
    logger = TrainLogger(log_file_path)

    try:
        configure_seeds(SEED)
        logger.info(f'随机种子已固定为 {SEED}')
        logger.info(f'当前实验目录: {checkpoints_save_dir}')

        if TYPE == 'faster_rcnn':
            from faster_rcnn_train import faster_rcnn_train
            faster_rcnn_train(checkpoints_save_dir, logger)
        elif TYPE == 'e2ecr':
            from e2ecr_train import e2ecr_train
            e2ecr_train(checkpoints_save_dir, logger)
        else:
            raise RuntimeError(
                '实验类型无效！'
            )
    finally:
        logger.close()

if __name__ == '__main__':
    train()