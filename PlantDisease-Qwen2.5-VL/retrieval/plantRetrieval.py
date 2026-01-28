"""
Plant Disease Knowledge Retrieval Module
基于 RAG 的农业病害诊断知识检索模块
"""

import os
import json
import re

from .RAGHandler import JSONSplitter
from .retrieval import Retrieval
from utils import CONFIG_AND_SETTINGS, LOGGER


# ==================================================
# 农业知识库加载
# ==================================================
def _load_json(filepath, name):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        LOGGER.warning(f"未找到{name}知识库文件：{filepath}，将跳过该部分检索。")
        return []


CROP_DB = _load_json(
    CONFIG_AND_SETTINGS.get("crop_knowledge_filepath", ""),
    "作物"
)

DISEASE_DB = _load_json(
    CONFIG_AND_SETTINGS.get("disease_knowledge_filepath", ""),
    "病害"
)

TREATMENT_DB = _load_json(
    CONFIG_AND_SETTINGS.get("treatment_knowledge_filepath", ""),
    "防治"
)


# ==================================================
# 作物知识检索
# ==================================================
def retrieve_crop(query: str, eager: bool = False) -> str:
    """
    检索作物基础知识（生育期、易感病害等）

    Args:
        query (str): 作物名称或包含作物信息的文本
        eager (bool): 是否启用向量检索（RAG）

    Returns:
        str: 作物相关知识文本
    """
    if not query:
        return ""

    retrieval = "【作物背景知识】\n"

    if eager:
        documents = JSONSplitter(
            CONFIG_AND_SETTINGS.get("crop_knowledge_filepath", "")
        ).split(text_type="agriculture")

        retrieval += Retrieval(
            documents,
            query,
            top_k=CONFIG_AND_SETTINGS.get("vector_search_top_k", 5)
        )
        return retrieval

    # 非 RAG：基于规则匹配
    for crop in CROP_DB:
        name = crop.get("作物名称", "")
        if name and name in query:
            for k, v in crop.items():
                retrieval += f"{k}: {v}\n"

    return retrieval


# ==================================================
# 病害知识检索（核心）
# ==================================================
def retrieve_disease(query: str, eager: bool = False) -> str:
    """
    检索植物病害相关知识（症状、成因、传播条件）

    Args:
        query (str): 症状描述或模型输出文本
        eager (bool): 是否启用 RAG 精细检索

    Returns:
        str: 病害知识文本
    """
    if not query:
        return ""

    retrieval = "【植物病害知识】\n"

    if eager:
        documents = JSONSplitter(
            CONFIG_AND_SETTINGS.get("disease_knowledge_filepath", "")
        ).split(text_type="agriculture")

        # 去除坐标等无关符号，降低噪声
        clean_query = re.sub(r"\[.*?\]", "", query)

        retrieval += Retrieval(
            documents,
            clean_query,
            top_k=CONFIG_AND_SETTINGS.get("vector_search_top_k", 8)
        )
        return retrieval

    # 规则匹配（fallback）
    for disease in DISEASE_DB:
        name = disease.get("病害名称", "")
        if name and name in query:
            for k, v in disease.items():
                retrieval += f"{k}: {v}\n"

    return retrieval


# ==================================================
# 防治措施检索
# ==================================================
def retrieve_treatment(query: str, eager: bool = True) -> str:
    """
    检索植物病害防治与管理措施

    Args:
        query (str): 已诊断的病害名称或分析文本
        eager (bool): 默认启用 RAG（防治建议必须准确）

    Returns:
        str: 防治建议文本
    """
    if not query:
        return ""

    retrieval = "【病害防治与管理建议】\n"

    if eager:
        documents = JSONSplitter(
            CONFIG_AND_SETTINGS.get("treatment_knowledge_filepath", "")
        ).split(text_type="agriculture")

        retrieval += Retrieval(
            documents,
            query,
            top_k=CONFIG_AND_SETTINGS.get("vector_search_top_k", 10)
        )
        return retrieval

    # 规则兜底
    for item in TREATMENT_DB:
        name = item.get("病害名称", "")
        if name and name in query:
            for k, v in item.items():
                retrieval += f"{k}: {v}\n"

    return retrieval
