"""
LangChain-based Prototype for Plant Disease Diagnosis
用于验证 RAG + 大模型在植物病害诊断中的实验性实现
"""

import sys
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import CONFIG_AND_SETTINGS
from engine.model import Qwen
from retrieval.RAGHandler_langchain import load_file, FAISSWrapper


# ==================================================
# 初始化 LangChain 病害诊断 Agent
# ==================================================
def initialize_agent():
    """
    构建基于 LangChain 的植物病害诊断 Agent（实验原型）
    """

    # ===== 加载农业知识库 =====
    knowledge_files = [
        CONFIG_AND_SETTINGS.get("disease_knowledge_filepath", ""),
        CONFIG_AND_SETTINGS.get("treatment_knowledge_filepath", "")
    ]

    # ===== Embedding 设置 =====
    embedding_model_dict = {
        "text2vec": "shibing624/text2vec-base-chinese",
        "gte-zh": "thenlper/gte-large-zh",
        "gte": "thenlper/gte-large",
    }
    EMBEDDING_MODEL = "gte-zh"
    EMBEDDING_DEVICE = "cuda"
    VECTOR_SEARCH_TOP_K = 5

    # ===== Prompt 设计（病害诊断专用）=====
    PROMPT_TEMPLATE = """
【参考农业知识】
{context}

【输入信息】
{question}

请基于以上信息完成植物病害诊断分析，要求：
1. 总结图像中可观察到的主要症状；
2. 推断最可能的病害类型，并说明依据；
3. 分析可能的诱发因素（环境、生育期等）；
4. 给出病害严重程度判断（轻 / 中 / 重）；
5. 提供科学、可操作的防治建议。

请使用专业、简洁、结构化的语言作答。
"""

    # ===== 初始化模型与向量库 =====
    llm = Qwen()
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[EMBEDDING_MODEL],
        model_kwargs={"device": EMBEDDING_DEVICE}
    )

    docs = []
    for filepath in knowledge_files:
        if filepath:
            docs.extend(load_file(filepath, check_file=True))

    docsearch = FAISSWrapper.from_documents(docs, embeddings)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    # ===== 构建 LCEL Chain =====
    qa_chain = (
        {
            "context": docsearch.as_retriever(
                search_kwargs={"k": VECTOR_SEARCH_TOP_K}
            ) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


# ==================================================
# CLI 调试入口（仅用于实验）
# ==================================================
if __name__ == "__main__":
    print("🌱 Plant Disease Diagnosis Agent Initialized")

    agent = initialize_agent()

    while True:
        print("\n------ Plant Diagnosis Agent Standby ------")

        user_input = input("请输入症状描述 / 病害问题（--q 退出）：\n")
        if not user_input:
            user_input = "叶片出现褐色不规则斑点，边缘发黄，近期连续阴雨。"
            print(f"[示例输入] {user_input}")

        if "--q" in user_input.lower():
            print("👋 诊断 Agent 已退出")
            break

        print("\n------ 正在进行病害诊断分析... ------\n")
        try:
            result = agent.invoke(user_input)
            print(result)
        except KeyboardInterrupt:
            print("⚠️ 用户中断")
        except Exception as e:
            print("❌ 诊断失败：", e)
