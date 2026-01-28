"""
Plant Disease Diagnosis Agent
Based on Plant-Qwen2.5-VL (LoRA-adapted Qwen2.5-VL)

 lhm
"""

import sys
import os
import json
import time
import re
import requests
from pathlib import Path
from copy import copy
from typing import List
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import CONFIG_AND_SETTINGS, SERVER_CONFIG, LOGGER
from utils.img_handler import image_to_base64_data_uri
from utils.info_extractor import extract_img_data
from utils.prompter import BasePrompter
from utils.save import briefing2file, fullreport2file

# 农业领域 RAG
from retrieval.plantRetrieval import (
    retrieve_crop,
    retrieve_disease,
    retrieve_treatment
)
from retrieval.retrieval import Retrieval
from retrieval.RAGHandler import JSONSplitter, AutoSplitter

# Prompt 前缀
PREINFO = "农业背景知识：\n"
PREQ = "诊断任务：\n"
TIME = f"当前时间：{time.strftime('%Y-%m-%d %H:%M', time.localtime())}\n"


# ============================
# LLM 调用（Plant-Qwen2.5-VL）
# ============================
def call_llama_server(
    messages,
    server_url=f"http://localhost:{SERVER_CONFIG['PORT']}/v1/chat/completions",
    stream=False,
    extra_params=None,
    use_tqdm=True
):
    payload = {
        # 显式使用 LoRA 微调后的模型
        "model": "plant-qwen2.5-vl",
        "messages": messages,
        "n_predict": 4096,
        "stop": ["<|im_end|>"],
        "stream": True if stream else False
    }

    if extra_params:
        payload.update(extra_params)

    response = requests.post(
        server_url, json=payload, timeout=1000, stream=stream
    )
    response.raise_for_status()

    if stream:
        result = ""
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if not line.startswith("data: {"):
                continue
            data = json.loads(line[len("data:"):].strip())
            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                result += content
                tqdm.write(content, end="", nolock=True) if use_tqdm else print(content, end="", flush=True)
        return result
    else:
        return response.json()["choices"][0]["message"]["content"]


# ============================
# Message 构造工具
# ============================
def build_img_message(messages, img_path, clean=True):
    img_msg = {
        "type": "image_url",
        "image_url": {
            "url": image_to_base64_data_uri(img_path, prefix=True)
        }
    }
    if clean:
        for m in messages:
            if m["role"] == "user":
                m["content"] = [c for c in m["content"] if c["type"] != "image_url"]
    messages[-1]["content"].append(img_msg)
    return messages


def build_text_message(messages, text, insert=None, clean=True):
    txt_msg = {"type": "text", "text": text}
    if clean:
        for m in messages:
            if m["role"] == "user":
                m["content"] = [c for c in m["content"] if c["type"] != "text"]

    if insert is None:
        messages[-1]["content"].append(txt_msg)
    else:
        messages[-1]["content"].insert(insert, txt_msg)
    return messages


def build_assistant_message(messages, completion):
    messages.append({"role": "assistant", "content": completion})
    messages.append({"role": "user", "content": []})
    return messages


def extract_answer(text: str, tag="answer") -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


# ============================
# 多阶段植物病害诊断流程
# ============================
def briefing(messages, img_paths: List[Path], show_process=False):
    """
    多阶段植物病害智能诊断：
    Stage 1: 作物与环境概述
    Stage 2: 病斑/症状区域核查
    Stage 3: 病害类型精细识别（RAG）
    Stage 4: 病害发展趋势分析
    Stage 5: 风险评估与防治建议
    """

    pbar = tqdm(total=5, desc="植物病害诊断中", ncols=100)
    stream = (show_process == "stream")
    show = (show_process == "stage")

    messages_bak = copy(messages)
    prompter = BasePrompter(img_path=img_paths)

    # ===== Stage 1 作物与环境概述 =====
    metadata = extract_img_data(img_paths)
    crop_env_info = prompter.regroup(prompter.IMinfo_prompt())

    stage_1_prompt = (
        "请根据图像判断作物类型、生育阶段以及生长环境状况，"
        "并对整体健康状态进行初步评估。"
        "在<think> </think>中给出分析过程，"
        "在<answer> </answer>中给出简要诊断概述。"
    )

    stage_1 = PREINFO + crop_env_info + TIME + PREQ + stage_1_prompt
    messages_1 = build_text_message(messages, stage_1)
    output_1 = call_llama_server(messages_1, stream=stream)
    match_1 = extract_answer(output_1)

    if show:
        tqdm.write(f"\n[Stage 1]\n{output_1}")
    pbar.update(1)

    # ===== Stage 2 病斑区域核查 =====
    od_info = prompter.regroup(prompter.ODinfo_prompt())

    stage_2_prompt = (
        "图像中标注了一些疑似病害症状区域（ROI）。"
        "请逐一判断这些区域是否为有效病斑，"
        "并检查是否存在被遗漏的重要症状区域。"
        "在<think> </think>中给出分析，"
        "在<answer> </answer>中给出最终确认的病斑描述。"
    )

    stage_2 = PREINFO + od_info + PREQ + stage_2_prompt
    messages_2 = build_text_message(messages, stage_2)
    output_2 = call_llama_server(messages_2, stream=stream)
    match_2 = extract_answer(output_2)

    if show:
        tqdm.write(f"\n[Stage 2]\n{output_2}")
    pbar.update(1)

    # ===== Stage 3 病害类型识别（RAG）=====
    disease_knowledge = retrieve_disease(match_2, eager=True)
    crop_knowledge = retrieve_crop(match_1)

    stage_3_prompt = (
        "结合图像症状、作物信息以及农业病害知识，"
        "逐一判断可能的植物病害类型，并分析其发生原因与严重程度，"
        "生成详细的病害诊断报告。"
    )

    stage_3 = PREINFO + crop_knowledge + disease_knowledge + PREQ + stage_3_prompt
    messages_3 = build_text_message(messages, stage_3)
    output_3 = call_llama_server(messages_3, stream=stream)
    match_3 = extract_answer(output_3)

    if show:
        tqdm.write(f"\n[Stage 3]\n{output_3}")
    pbar.update(1)

    # ===== Stage 4 病害发展趋势 =====
    stage_4_prompt = (
        "在前述诊断基础上，分析该病害在当前环境条件下的可能发展趋势，"
        "评估其对作物产量和品质的潜在影响。"
    )

    stage_4 = PREINFO + match_3 + PREQ + stage_4_prompt
    messages_4 = build_text_message(messages, stage_4)
    output_4 = call_llama_server(messages_4, stream=stream)
    match_4 = extract_answer(output_4)

    if show:
        tqdm.write(f"\n[Stage 4]\n{output_4}")
    pbar.update(1)

    # ===== Stage 5 风险评估与防治建议 =====
    treatment_knowledge = retrieve_treatment(match_3)

    stage_5_prompt = (
        "基于以上全部信息完成两步任务："
        "第一步，在<think> </think>中系统评估当前病害风险等级；"
        "第二步，在<answer> </answer>中给出科学、可执行的防治建议，"
        "包括推荐的农艺措施或植保方案。"
    )

    stage_5 = PREINFO + treatment_knowledge + TIME + PREQ + stage_5_prompt
    messages_5 = build_text_message(messages, stage_5)
    output_5 = call_llama_server(messages_5, stream=stream)
    match_5 = extract_answer(output_5)

    if show:
        tqdm.write(f"\n[Stage 5]\n{output_5}")
    pbar.update(1)

    # ===== 保存结果 =====
    briefing2file([match_5])
    fullreport2file(
        [stage_1, stage_2, stage_3, stage_4, stage_5],
        [output_1, output_2, output_3, output_4, output_5]
    )

    pbar.close()
    return
