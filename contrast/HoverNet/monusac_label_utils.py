"""MoNuSAC 点/分割标注 mat 构建与目录工具（供 MoNuSAC数据集处理.ipynb 使用）。"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class SegRegionAnnotation:
    label: str
    centroid: tuple[float, float]
    vertices: tuple[tuple[float, float], ...]


def parse_seg_annotations(
    xml_path: Path,
    coordinate_scale: float = 1.0,
) -> list[SegRegionAnnotation]:
    root = ET.parse(xml_path).getroot()
    annotations: list[SegRegionAnnotation] = []
    for annotation in root.findall("Annotation"):
        attribute = annotation.find("./Attributes/Attribute")
        label = attribute.attrib.get("Name", "Unknown") if attribute is not None else "Unknown"
        for region in annotation.findall("./Regions/Region"):
            vertices = region.findall("./Vertices/Vertex")
            xs = [float(vertex.attrib["X"]) * coordinate_scale for vertex in vertices if "X" in vertex.attrib]
            ys = [float(vertex.attrib["Y"]) * coordinate_scale for vertex in vertices if "Y" in vertex.attrib]
            if not xs or not ys:
                continue
            scaled_vertices = tuple(zip(xs, ys))
            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)
            annotations.append(
                SegRegionAnnotation(
                    label=label,
                    centroid=(centroid_x, centroid_y),
                    vertices=scaled_vertices,
                )
            )
    return annotations


def centroid_in_patch(
    centroid: tuple[float, float],
    patch_x: int,
    patch_y: int,
    patch_size: int,
) -> bool:
    center_x, center_y = centroid
    return (
        patch_x <= center_x < patch_x + patch_size
        and patch_y <= center_y < patch_y + patch_size
    )


def patch_seg_annotations(
    annotations: list[SegRegionAnnotation],
    patch_x: int,
    patch_y: int,
    patch_size: int,
) -> list[SegRegionAnnotation]:
    return [
        annotation
        for annotation in annotations
        if centroid_in_patch(annotation.centroid, patch_x, patch_y, patch_size)
    ]


def vertices_to_output_patch(
    vertices: tuple[tuple[float, float], ...],
    patch_x: int,
    patch_y: int,
    scale_x: float,
    scale_y: float,
    patch_size: int,
) -> np.ndarray:
    points = np.asarray(
        [
            (
                (x - patch_x) * scale_x,
                (y - patch_y) * scale_y,
            )
            for x, y in vertices
        ],
        dtype=np.float64,
    )
    points[:, 0] = np.clip(points[:, 0], 0, patch_size - 1)
    points[:, 1] = np.clip(points[:, 1], 0, patch_size - 1)
    return points


def build_point_label_mat(
    centroids_xy: list[tuple[float, float]],
    label_ids: list[int],
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    inst_map = np.zeros((height, width), dtype=np.int32)
    type_map = np.zeros((height, width), dtype=np.int32)
    inst_centroid_list: list[list[float]] = []
    inst_type_list: list[int] = []

    for inst_id, ((x, y), label_id) in enumerate(zip(centroids_xy, label_ids), start=1):
        cx = int(round(x))
        cy = int(round(y))
        cx = int(np.clip(cx, 0, width - 1))
        cy = int(np.clip(cy, 0, height - 1))
        inst_map[cy, cx] = inst_id
        type_map[cy, cx] = label_id
        inst_centroid_list.append([float(x), float(y)])
        inst_type_list.append(int(label_id))

    if inst_centroid_list:
        inst_centroid = np.asarray(inst_centroid_list, dtype=np.float64)
        inst_type = np.asarray(inst_type_list, dtype=np.int32).reshape(-1, 1)
    else:
        inst_centroid = np.zeros((0, 2), dtype=np.float64)
        inst_type = np.zeros((0, 1), dtype=np.int32)

    return {
        "inst_map": inst_map,
        "type_map": type_map,
        "inst_centroid": inst_centroid,
        "inst_type": inst_type,
    }


def build_seg_label_mat(
    regions: list[tuple[int, np.ndarray]],
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    inst_map = np.zeros((height, width), dtype=np.int32)
    type_map = np.zeros((height, width), dtype=np.int32)
    inst_centroid_list: list[list[float]] = []
    inst_type_list: list[int] = []

    for inst_id, (label_id, polygon) in enumerate(regions, start=1):
        if polygon.shape[0] < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        if int(mask.sum()) == 0:
            continue
        inst_map[mask > 0] = inst_id
        type_map[mask > 0] = label_id
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = float(polygon[:, 0].mean())
            cy = float(polygon[:, 1].mean())
        inst_centroid_list.append([cx, cy])
        inst_type_list.append(int(label_id))

    if inst_centroid_list:
        inst_centroid = np.asarray(inst_centroid_list, dtype=np.float64)
        inst_type = np.asarray(inst_type_list, dtype=np.int32).reshape(-1, 1)
    else:
        inst_centroid = np.zeros((0, 2), dtype=np.float64)
        inst_type = np.zeros((0, 1), dtype=np.int32)

    return {
        "inst_map": inst_map,
        "type_map": type_map,
        "inst_centroid": inst_centroid,
        "inst_type": inst_type,
    }


def empty_label_mat(height: int, width: int) -> dict[str, np.ndarray]:
    return {
        "inst_map": np.zeros((height, width), dtype=np.int32),
        "type_map": np.zeros((height, width), dtype=np.int32),
        "inst_centroid": np.zeros((0, 2), dtype=np.float64),
        "inst_type": np.zeros((0, 1), dtype=np.int32),
    }


def save_label_mat(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), payload)


def ensure_point_label_dirs(output_root: Path) -> None:
    for split_name in ("train", "val", "test"):
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def ensure_seg_output_root(seg_root: Path, overwrite: bool) -> None:
    if seg_root.exists() and overwrite:
        shutil.rmtree(seg_root)
    for split_name in ("train", "val", "test"):
        (seg_root / split_name / "Images").mkdir(parents=True, exist_ok=True)
        (seg_root / split_name / "Labels").mkdir(parents=True, exist_ok=True)
