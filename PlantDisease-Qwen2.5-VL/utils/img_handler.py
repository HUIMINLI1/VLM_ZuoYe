'''
$lhm 251016
issue:
'''
import sys, os
import base64
import io
import re
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
from utils import CONFIG_AND_SETTINGS, LOGGER, CACHE_DIR

SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
MIME_MAP = ['jpeg', 'jpeg', 'png', 'bmp', 'tiff', 'tiff']

def handle_files(raw_path_input: List[str]) -> List[Path]:
    '''
    仅在用户输入图像路径后调用一次。对图像路径合法性进行一次检查，并连同元数据（如果有）缓存入CACHE_DIR。
    无需检查文件格式，因为会在base64转码时检查。
    '''
    valid_paths = []
    for item in raw_path_input:
        item = item.strip().strip('[]').strip('\"').strip("\'")

        # 防止误将多个路径拼成一个字符串
        parts = re.split(r'[，,]+', item)

        for part in parts:
            part = part.strip().strip('\"').strip("\'")
            if not part: continue

            path = Path(part)
            if not path.is_file():
                LOGGER.error(f"文件不存在: {path}")
                raise FileNotFoundError

            valid_paths.append(path.resolve())

    return valid_paths

def find_metadata(img_paths: List[Path]) -> List[Dict]:

    metadata_list = []

    for img_path in img_paths:
        metadata_dict = {}
        metadata_path = None
        for ext in ['.json', '.txt']:
            candidate = img_path.with_suffix(ext)
            if candidate.exists():
                metadata_path = candidate
                break

        if metadata_path:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except json.decoder.JSONDecodeError as e: 
                LOGGER.error(f"图像元数据存在格式问题：{metadata_path}。{e} 已忽略。")
                metadata_list.append({})
                continue

            # 验证必要字段
            if not {"capture_time", "center_coord"}.issubset(metadata):
                metadata = {}

            metadata_list.append(metadata)

        else:
            metadata_list.append({})  # 不存在元数据文件，返回空字典

    return metadata_list

def format_checker(img_path):
    # 获取文件格式
    filename, ext = os.path.splitext(os.path.basename(img_path))
    ext = ext.lower()

    if ext not in SUPPORTED_FORMATS:
        LOGGER.error(f"不支持的图片格式: {ext}")
        raise AssertionError
    return filename, ext

def compress_to_jpeg(img_path, quality=95):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        # img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=0,
            progressive=True
        )
        buffer.seek(0)
        return buffer.read()

def image_to_base64_data_uri(img_path, compress=False, prefix=False):
    '''
    将图像以 base64 编码的 data URI 传递给llama.cpp
    '''
    _, ext = format_checker(img_path)

    if compress and (ext not in ['.jpg', '.jpeg']):
        img_bytes = compress_to_jpeg(img_path)
    # $wxy: llama-server传入tif图像报错。暂时不清楚原因(不排除爆显存了)，先强制compress。
    elif ext in ['.tif', '.tiff']:
        img_bytes = compress_to_jpeg(img_path)
        ext = '.jpeg'
    else:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    # print("len:base64_data:",len(base64_data))

    if prefix:
        return f"data:image/{MIME_MAP[SUPPORTED_FORMATS.index(ext)][1:]};base64,{base64_data}"
    else:
        return base64_data