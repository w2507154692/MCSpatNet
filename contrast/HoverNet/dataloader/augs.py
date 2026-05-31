import cv2
import numpy as np
from scipy.ndimage import measurements

from albumentations.core.transforms_interface import ImageOnlyTransform


def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).

    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


class RandomGaussianBlur(ImageOnlyTransform):
    """随机高斯模糊，核大小与 HoVer-Net 原版 imgaug 配置一致。"""

    def __init__(self, max_ksize=3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_ksize = max_ksize

    def apply(self, img, **params):
        ksize = self.random_state.randint(0, self.max_ksize, size=(2,))
        ksize = tuple((ksize * 2 + 1).tolist())
        ret = cv2.GaussianBlur(
            img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
        )
        return ret.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("max_ksize",)


class RandomMedianBlur(ImageOnlyTransform):
    """随机中值模糊。"""

    def __init__(self, max_ksize=3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_ksize = max_ksize

    def apply(self, img, **params):
        ksize = self.random_state.randint(0, self.max_ksize)
        ksize = ksize * 2 + 1
        ret = cv2.medianBlur(img, ksize)
        return ret.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("max_ksize",)


class RandomHue(ImageOnlyTransform):
    def __init__(self, range=(-8, 8), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.range = range

    def apply(self, img, **params):
        hue = self.random_state.uniform(*self.range)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if hsv.dtype.itemsize == 1:
            hsv[..., 0] = (hsv[..., 0] + hue) % 180
        else:
            hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
        ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return ret.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("range",)


class RandomSaturation(ImageOnlyTransform):
    def __init__(self, range=(-0.2, 0.2), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.range = range

    def apply(self, img, **params):
        value = 1 + self.random_state.uniform(*self.range)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
        return np.clip(ret, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("range",)


class RandomBrightness(ImageOnlyTransform):
    def __init__(self, range=(-26, 26), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.range = range

    def apply(self, img, **params):
        value = self.random_state.uniform(*self.range)
        return np.clip(img + value, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("range",)


class RandomContrast(ImageOnlyTransform):
    def __init__(self, range=(0.75, 1.25), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.range = range

    def apply(self, img, **params):
        value = self.random_state.uniform(*self.range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        ret = img * value + mean * (1 - value)
        return np.clip(ret, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("range",)


class RandomOrder(ImageOnlyTransform):
    """以随机顺序依次应用一组仅作用于图像的增强。"""

    def __init__(self, transforms, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.transforms = transforms

    def apply(self, img, **params):
        order = self.random_state.permutation(len(self.transforms))
        for idx in order:
            img = self.transforms[idx](image=img)["image"]
        return img

    def get_transform_init_args_names(self):
        return ("transforms",)
