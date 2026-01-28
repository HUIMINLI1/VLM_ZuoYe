"""
Plant Disease Visual Prompter
用于构造植物病害诊断的多模态提示词
"""

import os
import re

from utils.info_extractor import extract_bbox_data, extract_img_data


class BasePrompter:
    """
    植物病害诊断提示词构造器

    Args:
        img_path (list): 输入图像路径列表
    """

    def __init__(self, img_path=None):
        self.img_path = img_path or []
        self.prompt = None

        # 图像文件名（不含后缀）
        self.filenames = [
            os.path.splitext(os.path.basename(img))[0]
            for img in self.img_path
        ]

    # ==================================================
    # 病斑 / 症状区域信息（ROI Prompt）
    # ==================================================
    def ODinfo_prompt(self):
        """
        构造病斑或异常症状区域的描述提示词
        """
        prompt = ""
        metadata = extract_bbox_data(self.filenames)
        count = 1

        for filename in self.filenames:
            bbox_info = metadata.get(filename, {})

            # 若该图像未检测到任何症状区域
            if not bbox_info or all(len(v) == 0 for v in bbox_info.values()):
                continue

            prompt += f"<OD{count}>第{count}张图像中，检测到以下疑似病害症状区域："

            for symptom_label, coords in bbox_info.items():
                if len(coords) == 0:
                    continue
                prompt += (
                    f"{len(coords)}处{symptom_label}症状，"
                    f"位置坐标为{str(coords)[1:-1]}；"
                )

            # 去掉末尾分号
            prompt = prompt.rstrip("；")
            prompt += f"<OD{count}>"
            count += 1

        return prompt

    # ==================================================
    # 作物与拍摄环境信息（Image Meta Prompt）
    # ==================================================
    def IMinfo_prompt(self):
        """
        构造作物类型、生长环境及拍摄条件的提示词
        """
        prompt = ""
        metadata = extract_img_data(self.img_path)
        count = 1

        for filename in self.filenames:
            info = metadata.get(filename, {})
            capture_time = info.get("capture_time", "未知时间")
            crop_type = info.get("crop_type", "未知作物")
            growth_stage = info.get("growth_stage", "未知生育期")
            environment = info.get("environment", "自然环境")

            prompt += (
                f"<IM{count}>第{count}张图像拍摄于{capture_time}，"
                f"作物类型为{crop_type}，"
                f"生育阶段为{growth_stage}，"
                f"拍摄环境为{environment}。"
                f"<IM{count}>"
            )
            count += 1

        return prompt

    # ==================================================
    # 自动组合提示词（供 Chat / Briefing 使用）
    # ==================================================
    def AutoPrompt(self):
        """
        自动组合病斑区域与作物环境信息
        """
        return self.regroup(self.ODinfo_prompt() + self.IMinfo_prompt())

    # ==================================================
    # 多图信息重组（关键！）
    # ==================================================
    def regroup(self, prompt: str):
        """
        将 <IMn> 和 <ODn> 结构化信息按图像编号重新组织，
        防止多图输入时模型混淆
        """
        im_pattern = re.compile(r"<IM(\d+)>(.*?)<IM\1>")
        od_pattern = re.compile(r"<OD(\d+)>(.*?)<OD\1>")

        im_dict = {int(m.group(1)): m.group(2) for m in im_pattern.finditer(prompt)}
        od_dict = {int(m.group(1)): m.group(2) for m in od_pattern.finditer(prompt)}

        result = []
        all_indices = sorted(set(im_dict.keys()) | set(od_dict.keys()))

        for idx in all_indices:
            if idx in im_dict:
                result.append(im_dict[idx] + "。")
            if idx in od_dict:
                result.append(od_dict[idx] + "。")
            result.append("\n")

        return "".join(result)
