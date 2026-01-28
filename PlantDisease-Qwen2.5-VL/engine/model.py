'''
$lhm 251014
langchain框架的Qwen2.5-VL模型接口
'''
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info
import torch
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import time
# import threading

# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model_name = "./Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name, use_fast=False)


class Qwen(LLM, ABC):
     max_token: int = 10000
     # temperature: float = 0.01
     # top_p = 0.9
     # top_k = 20
     history_len: int = 3

     def __init__(self):
         super().__init__()
         # self.stop_flag = threading.Event()

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         if "<image>" in prompt:
            parts = prompt.split("<image>")
            image = parts[1].strip()
            prompt = parts[0].strip() + parts[2].strip()
            # print('image:', image, '\n\n\n', 'prompt:', prompt)
         else:image = './VLM/temp.bmp'

         messages = [

             {"role": "user", "content": [
                 {"type": "image", "image": image},
                 {"type": "text", "text": prompt}, 
             ]},
         ]
         print('prompt:::', messages[0])
         text = processor.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         image, _ = process_vision_info(messages)
         model_inputs = processor(text=[text], images=image, return_tensors="pt").to(model.device)

         streamer = TextStreamer(processor, skip_special_tokens=True)

         # 记录生成前的显存
         torch.cuda.reset_peak_memory_stats()
         mem_before = torch.cuda.memory_allocated() / 1048576  # MB
         start_time = time.time()  # 记录开始时间

        #  try:
         generated_ids = model.generate(
             **model_inputs,
             temperature=0.1,
             top_k=4,
             top_p=0.8,
             max_new_tokens=1024,
             repetition_penalty=1.1,
             streamer=streamer,
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

         return "🤖 Generation completed."

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,

                 "history_len": self.history_len, 
                 "do_sample": True,
                 "return_dict_in_generate": True,
                 "output_scores": True,

                 }