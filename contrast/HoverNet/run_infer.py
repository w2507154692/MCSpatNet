"""run_infer.py

HoVer-Net 推理入口脚本。

支持两种子命令：
  - tile：对常规图像 tile（png/jpg/tiff 等）做核分割与分类
  - wsi  ：对全切片图像（OpenSlide 格式）做分块推理

用法示例：
  python run_infer.py tile --model_path=... --nr_types=4 --input_dir=... --output_dir=...
  python run_infer.py wsi  --model_path=... --input_dir=... --output_dir=...

子命令参数见 tile_cli / wsi_cli；也可执行：
  python run_infer.py tile --help
  python run_infer.py wsi --help

注意：下方 docopt 文档串中的 Usage/Options 须保持英文关键字，否则解析会失败。

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: original]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

# tile 模式：单张/目录下多张常规图像，见 infer/tile.py
tile_cli = """
Arguments for processing tiles.

usage:
    (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: false]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: false]
   --save_raw_map         To save raw prediction or not. [default: false]
"""

# wsi 模式：全切片，见 infer/wsi.py
wsi_cli = """
Arguments for processing wsi

usage:
    (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --input_mask_dir=<path> Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: false]
    --save_mask             To save mask. [default: false]
"""

"""
CoNSeP
python run_infer.py tile \
--model_path="../exp/exp27_HoverNet_consep/01/net_epoch=50.tar" \
--model_mode=original \
--nr_types=4 \
--input_dir="../data/CoNSeP_raw/Test/Images" \
--output_dir="../exp/exp27_HoverNet_consep_e100_new_json" \
--type_info_path="./type_info_consep.json"

MoNuSAC
python run_infer.py tile \
--model_path="../exp/exp28_HoverNet_monusac/01/net_epoch=50.tar" \
--model_mode=original \
--nr_types=3 \
--input_dir="../data/MoNuSAC_seg/test/Images" \
--output_dir="../exp/exp28_HoverNet_monusac_e100" \
--type_info_path="./type_info_monusac.json"
"""

import logging
import os
import sys

import torch
from docopt import docopt

from misc.utils import log_info

# 主命令全局选项（写在 tile/wsi 之后时会落入 <args>，需回填）
_GLOBAL_ARG_KEYS = {
    "--gpu",
    "--nr_types",
    "--type_info_path",
    "--model_path",
    "--model_mode",
    "--nr_inference_workers",
    "--nr_post_proc_workers",
    "--batch_size",
}
_TILE_ARG_KEYS = {
    "--input_dir",
    "--output_dir",
    "--mem_usage",
    "--draw_dot",
    "--save_qupath",
    "--save_raw_map",
}
_WSI_ARG_KEYS = {
    "--input_dir",
    "--output_dir",
    "--cache_path",
    "--input_mask_dir",
    "--proc_mag",
    "--ambiguous_size",
    "--chunk_shape",
    "--tile_shape",
    "--save_thumb",
    "--save_mask",
}
# 无参开关（出现即 True）
_FLAG_ARG_KEYS = {
    "--draw_dot",
    "--save_qupath",
    "--save_raw_map",
    "--save_thumb",
    "--save_mask",
}


def _merge_global_options(args, argv, overwrite=False):
    """把误入 <args> 的全局选项写回主 docopt 的 args（tile 写在前的常见用法）。

    overwrite=True 时以 argv 为准（覆盖 docopt 默认值，例如 nr_types 默认 0）。
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key, sep, val = tok.partition("=")
            if key in _GLOBAL_ARG_KEYS and (overwrite or args.get(key) is None):
                if sep:
                    args[key] = val
                elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    args[key] = argv[i + 1]
                    i += 1
                else:
                    args[key] = True
        i += 1


def _filter_sub_argv(sub_cmd, argv):
    """从主命令 <args> 中筛出属于 tile/wsi 子命令的参数。"""
    if not argv:
        return []
    allowed = _TILE_ARG_KEYS if sub_cmd == "tile" else _WSI_ARG_KEYS
    filtered = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok.split("=", 1)[0]
            if key in allowed:
                filtered.append(tok)
                if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith(
                    "-"
                ):
                    filtered.append(argv[i + 1])
                    i += 1
        i += 1
    return filtered


def _parse_sub_args(sub_cmd, argv):
    """手动解析 tile/wsi 子命令参数（避免二次 docopt 与 options_first 冲突）。"""
    allowed = _TILE_ARG_KEYS if sub_cmd == "tile" else _WSI_ARG_KEYS
    result = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key, sep, val = tok.partition("=")
        if key not in allowed:
            i += 1
            continue
        if sep:
            result[key] = val
        elif key in _FLAG_ARG_KEYS:
            result[key] = True
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            result[key] = argv[i + 1]
            i += 1
        else:
            result[key] = True
        i += 1

    if sub_cmd == "tile":
        defaults = {
            "--mem_usage": "0.2",
            "--draw_dot": False,
            "--save_qupath": False,
            "--save_raw_map": False,
        }
    else:
        defaults = {
            "--cache_path": "cache",
            "--input_mask_dir": "",
            "--proc_mag": "40",
            "--ambiguous_size": "128",
            "--chunk_shape": "10000",
            "--tile_shape": "2048",
            "--save_thumb": False,
            "--save_mask": False,
        }
    for key, default in defaults.items():
        result.setdefault(key, default)

    for req in ("--input_dir", "--output_dir"):
        if req not in result:
            return None
    return result


# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # 子命令 -> 对应 docopt 帮助文档
    sub_cli_dict = {"tile": tile_cli, "wsi": wsi_cli}

    # options_first=True：先解析全局选项，再解析 <command> 与 <args>
    args = docopt(
        __doc__,
        help=False,
        options_first=True,
        version="HoVer-Net Pytorch Inference v1.0",
    )
    sub_cmd = args.pop("<command>")
    sub_cmd_args = args.pop("<args>")

    # tile --model_path=... 时全局选项会进 <args>；sys.argv 覆盖 docopt 默认值
    _merge_global_options(args, sub_cmd_args or [], overwrite=False)
    _merge_global_options(args, sys.argv[1:], overwrite=True)

    # ! TODO: 日志文件路径可改为可配置（当前固定为当前目录 debug.log）
    logging.basicConfig(
        level=logging.INFO,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(),
        ],
    )

    # 子命令级帮助：python run_infer.py tile --help
    if args["--help"] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict:
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args["--help"] or sub_cmd is None:
        print(__doc__)
        exit()

    # 解析 tile / wsi 子命令参数（须过滤掉误入 <args> 的全局 --model_path 等）
    if sub_cmd_args is None:
        sub_cmd_args = []
    sub_argv = _filter_sub_argv(sub_cmd, sub_cmd_args)
    if not sub_argv:
        # 兜底：从 sys.argv 中按子命令名后重新收集（兼容部分 docopt 行为差异）
        if sub_cmd in sys.argv:
            idx = sys.argv.index(sub_cmd)
            sub_argv = _filter_sub_argv(sub_cmd, sys.argv[idx + 1 :])
    if not sub_argv:
        print(sub_cli_dict[sub_cmd])
        print(
            "\nERROR: missing subcommand options (need --input_dir and --output_dir)."
        )
        exit(1)

    sub_args = _parse_sub_args(sub_cmd, sub_argv)
    if sub_args is None:
        print(sub_cli_dict[sub_cmd])
        print("\nERROR: --input_dir and --output_dir are required.")
        exit(1)

    args.pop("--version")
    gpu_list = args.pop("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info("Detect #GPUS: %d" % nr_gpus)

    # docopt 键去掉 '--' 前缀，便于后续用字典访问
    args = {k.replace("--", ""): v for k, v in args.items()}
    sub_args = {k.replace("--", ""): v for k, v in sub_args.items()}

    if args["model_path"] == None:
        raise Exception(
            "A model path must be supplied as an argument with --model_path."
        )

    # nr_types=0 表示仅分割、不预测类型；>0 时与训练 checkpoint 的类型头通道数一致
    nr_types = int(args["nr_types"]) if int(args["nr_types"]) > 0 else None
    log_info(
        "Inference: model_mode=%s, nr_types=%s, model_path=%s"
        % (args["model_mode"], nr_types, args["model_path"])
    )
    method_args = {
        "method": {
            "model_args": {
                "nr_types": nr_types,
                "mode": args["model_mode"],  # 'original' 或 'fast'，须与 checkpoint 训练模式一致
            },
            "model_path": args["model_path"],
        },
        # 类型 id -> 名称/颜色的 JSON；空字符串表示不使用
        "type_info_path": None
        if args["type_info_path"] == ""
        else args["type_info_path"],
    }

    # ***
    # 推理与后处理共用运行参数
    run_args = {
        "batch_size": int(args["batch_size"]) * nr_gpus,  # 总 batch = 每卡 batch × GPU 数
        "nr_inference_workers": int(args["nr_inference_workers"]),
        "nr_post_proc_workers": int(args["nr_post_proc_workers"]),
    }

    # patch 输入/输出尺寸须与 model_mode 一致（与 config.py / 训练时相同）
    if args["model_mode"] == "fast":
        run_args["patch_input_shape"] = 256
        run_args["patch_output_shape"] = 164
    else:
        run_args["patch_input_shape"] = 270
        run_args["patch_output_shape"] = 80

    if sub_cmd == "tile":
        run_args.update(
            {
                "input_dir": sub_args["input_dir"],
                "output_dir": sub_args["output_dir"],
                "mem_usage": float(sub_args["mem_usage"]),  # 缓存占用物理+交换内存比例上限
                "draw_dot": sub_args["draw_dot"],  # 是否在 overlay 上画核质心
                "save_qupath": sub_args["save_qupath"],  # 是否导出 QuPath 0.2.3 格式
                "save_raw_map": sub_args["save_raw_map"],  # 是否保存原始预测图
            }
        )

    if sub_cmd == "wsi":
        run_args.update(
            {
                "input_dir": sub_args["input_dir"],
                "output_dir": sub_args["output_dir"],
                "input_mask_dir": sub_args["input_mask_dir"],
                "cache_path": sub_args["cache_path"],
                "proc_mag": int(sub_args["proc_mag"]),  # WSI 处理倍率（如 40×）
                "ambiguous_size": int(sub_args["ambiguous_size"]),  #  tile 边界模糊区重后处理宽度
                "chunk_shape": int(sub_args["chunk_shape"]),
                "tile_shape": int(sub_args["tile_shape"]),
                "save_thumb": sub_args["save_thumb"],
                "save_mask": sub_args["save_mask"],
            }
        )
    # ***

    if sub_cmd == "tile":
        from infer.tile import InferManager

        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager

        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)
