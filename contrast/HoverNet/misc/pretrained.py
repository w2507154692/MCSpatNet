"""预训练权重路径解析与自动下载。"""

import os
import re
import urllib.request

import torch

from run_utils.utils import colored

# 文件名 -> Google Drive 文件 ID（见 HoverNet README）
PRETRAINED_REGISTRY = {
    "ImageNet-ResNet50-Preact_pytorch.tar": "1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5",
}

# 第一阶段未指定权重时，默认使用 ImageNet 预训练 ResNet50
DEFAULT_PRETRAINED = "../pretrained/ImageNet-ResNet50-Preact_pytorch.tar"


def _resolve_path(pretrained_path, base_dir):
    if os.path.isabs(pretrained_path):
        return os.path.normpath(pretrained_path)
    return os.path.normpath(os.path.join(base_dir, pretrained_path))


def _download_gdrive_file(file_id, destination):
    """从 Google Drive 下载公开文件（优先 gdown，否则回退 urllib）。"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = destination + ".part"

    try:
        import gdown

        url = "https://drive.google.com/uc?id=%s" % file_id
        gdown.download(url, tmp_path, quiet=False)
    except ImportError:
        url = "https://docs.google.com/uc?export=download&id=%s" % file_id
        opener = urllib.request.build_opener()
        response = opener.open(url)
        content = response.read()

        if b"download_warning" in content or b"confirm=" in content:
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
            if match:
                confirm_url = "%s&confirm=%s" % (url, match.group(1))
                response = opener.open(confirm_url)
                content = response.read()

        with open(tmp_path, "wb") as file_ptr:
            file_ptr.write(content)

    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            "Failed to download pretrained weights for `%s`." % os.path.basename(destination)
        )

    os.replace(tmp_path, destination)


def resolve_pretrained_path(pretrained_path, base_dir=None, allow_default=True):
    """解析预训练权重路径；本地不存在时按注册表自动下载。

    Args:
        pretrained_path: 权重路径、None，或 -1（表示沿用上一阶段 checkpoint）
        base_dir: 解析相对路径的基准目录，默认为 HoverNet 根目录
        allow_default: 为 True 且路径为 None 时，回退到 ImageNet 预训练权重

    Returns:
        解析后的本地路径，或原样返回 None / -1
    """
    if pretrained_path == -1:
        return pretrained_path

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if pretrained_path is None:
        if not allow_default:
            return None
        pretrained_path = DEFAULT_PRETRAINED

    resolved = _resolve_path(pretrained_path, base_dir)
    if os.path.isfile(resolved):
        return resolved

    filename = os.path.basename(resolved)
    if filename not in PRETRAINED_REGISTRY:
        raise FileNotFoundError(
            "Pretrained checkpoint not found: `%s`. "
            "Known auto-download files: %s"
            % (resolved, ", ".join(PRETRAINED_REGISTRY.keys()))
        )

    colored_word = colored("INFO", color="yellow", attrs=["bold"])
    print(
        "%s: `%s` not found locally, downloading from Google Drive..."
        % (colored_word, filename)
    )
    _download_gdrive_file(PRETRAINED_REGISTRY[filename], resolved)
    print("Downloaded pretrained weights to: %s" % resolved)
    return resolved


def load_pretrained_state_dict(pretrained_path):
    """从 .tar 或 .npz checkpoint 加载 state_dict。"""
    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
    if chkpt_ext == "npz":
        import numpy as np

        net_state_dict = dict(np.load(pretrained_path))
        return {k: torch.from_numpy(v) for k, v in net_state_dict.items()}
    if chkpt_ext == "tar":
        return torch.load(pretrained_path, map_location="cpu")["desc"]
    raise ValueError(
        "Unsupported checkpoint extension `%s` for `%s`"
        % (chkpt_ext, pretrained_path)
    )
