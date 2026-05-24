import os
import glob
from pathlib import Path
import shutil

CHECKPOINTS_ROOT_DIR = './exp'   # 所有训练输出的根目录
# 当前实验对应的子目录名称，最终模型权重会从 <checkpoints_root_dir>/<checkpoints_folder_name> 下读取。
CHECKPOINTS_FOLDER_NAME = 'exp15_e2ecr_brcam2c'
EVAL_ROOT_DIR = './exp'
EPOCH = 62
TYPE = 'e2ecr'
SUFFIX = ''   # 后缀，进行额外补充，默认为空

# 备份代码到实验目录下
def backup_python_files(experiment_dir):
    project_root = os.path.dirname(os.path.abspath(__file__))
    code_backup_dir = os.path.join(experiment_dir, 'code')
    os.makedirs(code_backup_dir, exist_ok=True)
    for pattern in ('*.py', '*.ipynb'):
        for filepath in glob.glob(os.path.join(project_root, pattern)):
            shutil.copy2(filepath, os.path.join(code_backup_dir, os.path.basename(filepath)))

def test():
    checkpoints_save_dir = os.path.join(CHECKPOINTS_ROOT_DIR, CHECKPOINTS_FOLDER_NAME)
    out_dir = os.path.join(EVAL_ROOT_DIR, CHECKPOINTS_FOLDER_NAME+f'_e{EPOCH}'+SUFFIX)

    # 备份代码
    backup_python_files(out_dir)

    # 根据epoch编号，匹配pth文件
    pth_file_path = glob.glob(os.path.join(checkpoints_save_dir, '*epoch'+str(EPOCH)+'*.pth'))
    if isinstance(pth_file_path, (list, tuple)):
        print(f"期望恰好一个 checkpoint 文件，实际得到 {len(pth_file_path)} 个: {pth_file_path}")
        print("默认取第一个")
        pth_file_path = Path(pth_file_path[0])

    if TYPE == 'faster_rcnn':
        from faster_rcnn_test import faster_rcnn_test
        faster_rcnn_test(out_dir, pth_file_path)
    elif TYPE == 'e2ecr':
        from e2ecr_test import e2ecr_test
        e2ecr_test(out_dir, pth_file_path)
    else:
         raise RuntimeError(
            '实验类型无效！'
        )

if __name__ == '__main__':
    test()

