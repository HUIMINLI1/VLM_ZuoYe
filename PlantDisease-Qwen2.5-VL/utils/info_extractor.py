"""
Plant Disease Information Extractor
用于提取植物病害诊断所需的图像与病斑信息
"""

import os
import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

from utils import CONFIG_AND_SETTINGS, LOGGER, CACHE_DIR
from utils.img_handler import format_checker, find_metadata


# ==================================================
# 病斑 / 症状区域提取
# ==================================================
def extract_bbox_data(filenames: list, coord_acc: int = 2) -> dict:
    """
    提取病斑检测模型输出的症状区域信息

    Args:
        filenames (list): 不含后缀的图像文件名列表
        coord_acc (int): 坐标保留精度

    Returns:
        dict:
        {
            "image1": {
                "叶斑": [[x1,y1,x2,y2,...], ...],
                "黄化": [...]
            },
            ...
        }
    """
    bbox_data = {}

    symptom_classes = CONFIG_AND_SETTINGS.get(
        "symptom_classes",
        {}
    )

    for filename in filenames:
        bbox_data[filename] = {}
        result_path = os.path.join(CACHE_DIR, f"{filename}.txt")

        if not os.path.exists(result_path):
            LOGGER.warning(
                f"未检测到病斑识别结果：'{result_path}'，将仅基于整图进行诊断。"
            )
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                label_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # 坐标精度控制
                coords = [
                    max(round(c, coord_acc), 0.0) for c in coords
                ]

                label_name = symptom_classes.get(
                    label_id, f"未知症状_{label_id}"
                )

                if label_name not in bbox_data[filename]:
                    bbox_data[filename][label_name] = []

                bbox_data[filename][label_name].append(coords)

    return bbox_data


# ==================================================
# 作物 & 图像元信息提取
# ==================================================
def extract_img_data(img_paths: list) -> dict:
    """
    提取植物病害诊断相关的图像元信息

    Args:
        img_paths (list): 图像路径列表

    Returns:
        dict:
        {
            "image1": {
                "width": int,
                "height": int,
                "format": str,
                "capture_time": str,
                "crop_type": str,
                "growth_stage": str,
                "environment": str
            }
        }
    """
    img_data = {}

    # 可选：从外部 metadata 文件读取农业语义信息
    metadata_docs = find_metadata(img_paths)

    for idx, img_path in enumerate(img_paths):
        filename, ext = format_checker(img_path)
        file_info = {
            "format": ext[1:]
        }

        # ===== 合并外部农业元数据（如有）=====
        meta = metadata_docs[idx] if idx < len(metadata_docs) else {}
        file_info["crop_type"] = meta.get("crop_type", "未知作物")
        file_info["growth_stage"] = meta.get("growth_stage", "未知生育期")
        file_info["environment"] = meta.get("environment", "自然环境")

        # ===== 读取图像基础信息 =====
        try:
            with Image.open(img_path) as img:
                file_info.update({
                    "width": img.width,
                    "height": img.height
                })

                # ===== EXIF 拍摄时间 =====
                capture_time = None
                if hasattr(img, "_getexif") and img._getexif():
                    exif = {
                        TAGS.get(k, "").lower(): v
                        for k, v in img._getexif().items()
                    }
                    capture_time = exif.get("datetimeoriginal")

                if not capture_time:
                    mtime = os.path.getmtime(img_path)
                    capture_time = datetime.datetime.fromtimestamp(
                        mtime
                    ).strftime("%Y-%m-%d %H:%M")

                file_info["capture_time"] = capture_time

        except Exception as e:
            LOGGER.error(
                f"图像信息读取失败：{img_path}，错误：{e}"
            )
            file_info.setdefault("capture_time", "未知时间")

        img_data[filename] = file_info

    LOGGER.debug(f"extract_img_data({img_paths}) => {img_data}")
    return img_data
