import os
import glob
from pathlib import Path

CHECKPOINTS_ROOT_DIR = './exp'   # 所有训练输出的根目录
# 当前实验对应的子目录名称，最终模型权重会从 <checkpoints_root_dir>/<checkpoints_folder_name> 下读取。
CHECKPOINTS_FOLDER_NAME = 'exp4_faster_rcnn_consep'
EVAL_ROOT_DIR = './exp'
EPOCH = 91
TYPE = 'baseline'
SUFFIX = ''   # 后缀，进行额外补充，默认为空

def test():
    checkpoints_save_dir = os.path.join(CHECKPOINTS_ROOT_DIR, CHECKPOINTS_FOLDER_NAME)
    out_dir = os.path.join(EVAL_ROOT_DIR, CHECKPOINTS_FOLDER_NAME+f'_e{EPOCH}'+SUFFIX)

    # 根据epoch编号，匹配pth文件
    pth_file_path = glob.glob(os.path.join(checkpoints_save_dir, '*epoch'+str(EPOCH)+'_*.pth'))
    if isinstance(pth_file_path, (list, tuple)):
        print(f"期望恰好一个 checkpoint 文件，实际得到 {len(pth_file_path)} 个: {pth_file_path}")
        print("默认取第一个")
        pth_file_path = Path(pth_file_path[0])

    if TYPE == 'baseline':
        from faster_rcnn_test import faster_rcnn_test
        faster_rcnn_test(out_dir, pth_file_path)
    else:
         raise RuntimeError(
            '实验类型无效！'
        )

if __name__ == '__main__':
    test()

