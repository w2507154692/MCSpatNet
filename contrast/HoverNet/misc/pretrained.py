"""预训练权重路径解析与自动下载。"""

import os
import re
import shutil
import urllib.error
import urllib.request

import torch

from run_utils.utils import colored

# 文件名 -> Google Drive 文件 ID（见 HoverNet README）
PRETRAINED_REGISTRY = {
    "ImageNet-ResNet50-Preact_pytorch.tar": "1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5",
}

# 第一阶段未指定权重时，默认使用 ImageNet 预训练 ResNet50
DEFAULT_PRETRAINED = "../pretrained/ImageNet-ResNet50-Preact_pytorch.tar"

# 官方 Google Drive 分享页（需手动下载时给用户）
GDRIVE_SHARE_URL = (
    "https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view?usp=sharing"
)

DOWNLOAD_TIMEOUT_SEC = 120


class PretrainedNotFoundError(FileNotFoundError):
    """本地无权重且自动下载失败。"""


def _resolve_path(pretrained_path, base_dir):
    if os.path.isabs(pretrained_path):
        return os.path.normpath(pretrained_path)
    return os.path.normpath(os.path.join(base_dir, pretrained_path))


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _find_local_pretrained(filename, resolved, base_dir):
    """在环境变量与常见目录中查找已存在的权重文件。"""
    env_file = os.environ.get("HOVERNET_PRETRAINED", "").strip()
    if env_file and os.path.isfile(env_file):
        return os.path.normpath(env_file)

    env_dir = os.environ.get("HOVERNET_PRETRAINED_DIR", "").strip()
    if env_dir:
        candidate = os.path.join(env_dir, filename)
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)

    if os.path.isfile(resolved):
        return resolved

    hovernet_root = base_dir
    repo_root = os.path.dirname(os.path.dirname(hovernet_root))
    extra_candidates = [
        os.path.join(hovernet_root, "pretrained", filename),
        os.path.join(repo_root, "pretrained", filename),
        os.path.join(repo_root, "contrast", "pretrained", filename),
        os.path.expanduser(os.path.join("~", "pretrained", filename)),
        os.path.expanduser(
            os.path.join("~", ".cache", "hovernet", "pretrained", filename)
        ),
    ]
    for candidate in extra_candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def _download_url_to_file(url, destination, timeout=DOWNLOAD_TIMEOUT_SEC):
    """通用 HTTP(S) 下载，带超时。"""
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    tmp_path = destination + ".part"
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (HoverNet-pretrained-fetch)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with open(tmp_path, "wb") as file_ptr:
            shutil.copyfileobj(response, file_ptr, length=1024 * 1024)
    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError("Downloaded file is empty: %s" % url)
    os.replace(tmp_path, destination)


def _download_gdrive_file(file_id, destination, timeout=DOWNLOAD_TIMEOUT_SEC):
    """从 Google Drive 下载公开文件（优先 gdown，否则回退 urllib）。"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = destination + ".part"
    last_error = None

    try:
        import gdown

        url = "https://drive.google.com/uc?id=%s" % file_id
        gdown.download(url, tmp_path, quiet=False)
    except ImportError as err:
        last_error = err
        try:
            url = "https://docs.google.com/uc?export=download&id=%s" % file_id
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()

            if b"download_warning" in content or b"confirm=" in content:
                try:
                    text = content.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
                match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
                if match:
                    confirm_url = "%s&confirm=%s" % (url, match.group(1))
                    request = urllib.request.Request(
                        confirm_url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        content = response.read()

            with open(tmp_path, "wb") as file_ptr:
                file_ptr.write(content)
        except Exception as err:
            last_error = err
    except Exception as err:
        last_error = err

    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            "Failed to download from Google Drive (file_id=%s): %s"
            % (file_id, last_error)
        )

    os.replace(tmp_path, destination)


def _try_auto_download(filename, destination):
    """按优先级尝试下载：自定义 URL -> Google Drive。"""
    custom_url = os.environ.get("HOVERNET_PRETRAINED_URL", "").strip()
    if custom_url:
        colored_word = colored("INFO", color="yellow", attrs=["bold"])
        print(
            "%s: downloading `%s` from HOVERNET_PRETRAINED_URL ..."
            % (colored_word, filename)
        )
        _download_url_to_file(custom_url, destination)
        return

    if _env_flag("HOVERNET_NO_GDRIVE"):
        raise RuntimeError("HOVERNET_NO_GDRIVE is set; Google Drive download disabled.")

    colored_word = colored("INFO", color="yellow", attrs=["bold"])
    print(
        "%s: `%s` not found locally, downloading from Google Drive..."
        % (colored_word, filename)
    )
    _download_gdrive_file(PRETRAINED_REGISTRY[filename], destination)


def _format_manual_help(filename, resolved):
    lines = [
        "Pretrained weights `%s` are required but not found at:" % filename,
        "  %s" % resolved,
        "",
        "On servers without Google Drive access, use ONE of:",
        "  1) Upload the file to the path above (create `pretrained/` if needed).",
        "  2) export HOVERNET_PRETRAINED=/absolute/path/to/%s" % filename,
        "  3) export HOVERNET_PRETRAINED_DIR=/dir/containing/the/file",
        "  4) export HOVERNET_PRETRAINED_URL=https://your-mirror/%s" % filename,
        "     then re-run (HTTP/HTTPS mirror on AutoDL or campus network).",
        "  5) Set `pretrained: None` in models/hovernet/opt.py to train from scratch.",
        "",
        "Official download (browser / machine with VPN):",
        "  %s" % GDRIVE_SHARE_URL,
        "",
        "Disable auto-download only (fail fast with this message):",
        "  export HOVERNET_NO_DOWNLOAD=1",
    ]
    return "\n".join(lines)


def resolve_pretrained_path(
    pretrained_path, base_dir=None, allow_default=True, allow_download=True
):
    """解析预训练权重路径；本地不存在时尝试下载。

    Args:
        pretrained_path: 权重路径、None，或 -1（表示沿用上一阶段 checkpoint）
        base_dir: 解析相对路径的基准目录，默认为 HoverNet 根目录
        allow_default: 为 True 且路径为 None 时，回退到 ImageNet 预训练权重
        allow_download: 为 False 或环境变量 HOVERNET_NO_DOWNLOAD=1 时不联网下载

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
    filename = os.path.basename(resolved)

    local_path = _find_local_pretrained(filename, resolved, base_dir)
    if local_path is not None:
        if local_path != resolved:
            colored_word = colored("INFO", color="yellow", attrs=["bold"])
            print(
                "%s: using pretrained weights at `%s`"
                % (colored_word, local_path)
            )
        return local_path

    if filename not in PRETRAINED_REGISTRY:
        raise FileNotFoundError(
            "Pretrained checkpoint not found: `%s`. "
            "Known auto-download files: %s"
            % (resolved, ", ".join(PRETRAINED_REGISTRY.keys()))
        )

    if not allow_download or _env_flag("HOVERNET_NO_DOWNLOAD"):
        raise PretrainedNotFoundError(_format_manual_help(filename, resolved))

    try:
        _try_auto_download(filename, resolved)
    except Exception as err:
        raise PretrainedNotFoundError(
            "%s\n\nAuto-download failed: %s"
            % (_format_manual_help(filename, resolved), err)
        ) from err

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
