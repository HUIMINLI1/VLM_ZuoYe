"""
RAG Handler for Plant Disease Diagnosis
用于将农业病害、作物与防治知识切分为适合向量检索的文本块
"""

import sys
import os
import json
import re
import time
from pathlib import Path
from typing import List, Sequence

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import LOGGER, CACHE_DIR


# ==================================================
# JSON 知识库切分（农业专用）
# ==================================================
class JSONSplitter:
    """
    将农业知识 JSON 切分为语义完整、适合 RAG 的文本块
    """

    def __init__(self, json_path: str):
        self.doc_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    # --------------------------------------------------
    # 病害知识切分（核心）
    # --------------------------------------------------
    def split_diseases(self, max_words=512, save=False) -> List[Document]:
        chunks = []

        for disease in self.data:
            name = disease.get("病害名称", "未知病害")
            crop = disease.get("作物", "未知作物")

            for key, value in disease.items():
                if key in ["病害名称", "作物"]:
                    continue

                text = self._format_value(value)
                for piece in self._cut_into_pieces(
                    text,
                    prefix=f"病害名称:{name} 作物:{crop} {key}",
                    max_words=max_words
                ):
                    chunks.append(piece)

        if save:
            self._save_json(chunks)

        return [Document(text=c) for c in chunks if c.strip()]

    # --------------------------------------------------
    # 作物知识切分
    # --------------------------------------------------
    def split_crops(self, max_words=512, save=False) -> List[Document]:
        chunks = []

        for crop in self.data:
            name = crop.get("作物名称", "未知作物")

            for key, value in crop.items():
                if key == "作物名称":
                    continue

                text = self._format_value(value)
                for piece in self._cut_into_pieces(
                    text,
                    prefix=f"作物:{name} {key}",
                    max_words=max_words
                ):
                    chunks.append(piece)

        if save:
            self._save_json(chunks)

        return [Document(text=c) for c in chunks if c.strip()]

    # --------------------------------------------------
    # 防治措施切分
    # --------------------------------------------------
    def split_treatments(self, max_words=512, save=False) -> List[Document]:
        chunks = []

        for item in self.data:
            disease = item.get("病害名称", "未知病害")

            for key, value in item.items():
                if key == "病害名称":
                    continue

                text = self._format_value(value)
                for piece in self._cut_into_pieces(
                    text,
                    prefix=f"病害名称:{disease} 防治:{key}",
                    max_words=max_words
                ):
                    chunks.append(piece)

        if save:
            self._save_json(chunks)

        return [Document(text=c) for c in chunks if c.strip()]

    # --------------------------------------------------
    # 统一入口
    # --------------------------------------------------
    def split(self, text_type="disease", chunk_size=512, save=False):
        if text_type == "disease":
            return self.split_diseases(chunk_size, save)
        elif text_type == "crop":
            return self.split_crops(chunk_size, save)
        elif text_type == "treatment":
            return self.split_treatments(chunk_size, save)
        else:
            raise ValueError(f"未知的 text_type: {text_type}")

    # ==================================================
    # 内部工具函数
    # ==================================================
    def _format_value(self, value) -> str:
        if isinstance(value, dict):
            return "；".join(f"{k}:{self._format_value(v)}" for k, v in value.items())
        if isinstance(value, list):
            return "；".join(str(v) for v in value)
        return str(value)

    def _cut_sent(self, text: str):
        text = re.sub('([。！？])', r"\1\n", text)
        return [s for s in text.split("\n") if s.strip()]

    def _cut_into_pieces(self, text: str, prefix: str, max_words: int):
        if len(text) <= max_words:
            yield f"{prefix}：{text}"
            return

        sentences = self._cut_sent(text)
        buffer = ""

        for sent in sentences:
            if len(buffer) + len(sent) <= max_words:
                buffer += sent
            else:
                yield f"{prefix}：{buffer}"
                buffer = sent

        if buffer:
            yield f"{prefix}：{buffer}"

    def _save_json(self, chunks):
        save_path = f"{CACHE_DIR}/{Path(self.doc_path).stem}_{time.strftime('%H%M%S')}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


# ==================================================
# 通用文本切分
# ==================================================
class AutoSplitter:
    def __init__(self, doc_path: Path | List[Path], chunk_size=512, chunk_overlap=10):
        self.doc_path = doc_path
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self) -> Sequence[Document]:
        if isinstance(self.doc_path, list):
            documents = SimpleDirectoryReader(
                input_files=self.doc_path
            ).load_data()
        else:
            documents = SimpleDirectoryReader(
                input_files=[self.doc_path]
            ).load_data()

        pipeline = IngestionPipeline(transformations=[self.text_splitter])
        nodes = pipeline.run(documents=documents)

        return [
            Document(
                text=node.text,
                metadata={"file_name": node.metadata.get("file_name", "")}
            )
            for node in nodes
        ]


# Debug
if __name__ == "__main__":
    docs = JSONSplitter("database/diseases.json").split("disease", save=True)
    print(docs[0].text)
