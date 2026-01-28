'''
$lhm 251018
transformers框架的主程序
'''
import sys, os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info
import torch
import time

from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.info_extractor import extract_bbox_data

model_name = '../huggingface/Qwen2.5-VL-7B-Instruct'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    # attn_implementation=None,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)



img_path = r"G:\Data\OfficeHome\00019.jpg"

bboxes = extract_bbox_data([img_path])

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
            },
            {
                "type": "image",
                "image": r"G:\Data\OfficeHome\00023.jpg",
            },
            {"type": "text", "text": f"What's in the oriented bounding box [491, 322, 622, 426] ?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# inputs = inputs.to("cuda")
streamer = TextStreamer(processor, skip_special_tokens=True)

# 记录生成前的显存
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1048576  # MB
start_time = time.time()  # 记录开始时间

generated_ids = model.generate(
            **inputs,
            temperature=0.1,
            top_k=1,
            top_p=0.8,
            max_new_tokens=1024,
            repetition_penalty=1.1,
            # streamer=streamer,
            )



end_time = time.time()  # 记录结束时间
mem_after = torch.cuda.memory_allocated() / 1048576  # MB
mem_peak = torch.cuda.max_memory_allocated() / 1048576  # MB
avg_mem = (mem_before + mem_after) / 2

num_tokens = generated_ids.shape[-1]  # 统计生成的 token 数量
elapsed = end_time - start_time
speed = max(0, num_tokens / elapsed)
print(f"🤖 Token输出速度: {speed:.2f} tokens/s")
print(f"🤖 平均显存占用: {avg_mem:.2f} MB, 峰值显存占用: {mem_peak:.2f} MB")
# print(output_text)