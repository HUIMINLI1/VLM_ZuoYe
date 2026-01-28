'''
$lhm 251020
'''
import sys, os
from typing import Sequence, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, BasePromptTemplate
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor, LongContextReorder
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .RAGHandler import JSONSplitter, AutoSplitter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIG_AND_SETTINGS, LOGGER


# Test Only
# 测试多种模型可以把名称存放在这里方便调用
EMBEDDING_MODELS = {
        "text2vec-b-ch": "shibing624/text2vec-base-chinese",
        "gte-l-zh": "thenlper/gte-large-zh",
        "gte-l": "thenlper/gte-large",
    }

def Retrieval(documents: Sequence[Document] | Any,
              query,
              model_name=CONFIG_AND_SETTINGS["embedding_model"],
              top_k: int = 10) -> str:
    """
    Domain-agnostic knowledge retrieval function.
    Used for retrieving plant disease, crop, and treatment knowledge
    in the diagnosis pipeline.
    """

    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=CONFIG_AND_SETTINGS["embedding_device"]
    )

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
    )

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.35),
            LongContextReorder(),
        ],
        # CONTEXT_ONLY: return evidence only, reasoning handled downstream
        response_synthesizer=get_response_synthesizer(
            response_mode=ResponseMode.CONTEXT_ONLY,
            text_qa_template=BasePromptTemplate
        )
    )

    response = query_engine.query(query)
    return response.response

# Debug Only
if __name__ == '__main__':
    documents = AutoSplitter("G:\QwenIA\database\ships.json").split()