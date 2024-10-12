import json
import logging
import sys
import torch
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, GPTVectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import KeywordNodePostprocessor, SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    GPTVectorStoreIndex,

)
from tqdm import tqdm

# 使用llama-index创建本地大模型
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
    },
    tokenizer_name=r'D:\models\Qwen2-7B-Instruct',
    model_name=r'D:\models\Qwen2-7B-Instruct',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},
)
Settings.llm = llm

# 使用llama-index-embeddings-huggingface构建本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\models\bge-large-zh-v1.5"
)

# 读取文档，没有转化文件格式的必要，支持多种文件读取
documents = SimpleDirectoryReader(r"D:\sxr\elearnPJ\data\elearn\教材\物理生物").load_data()

# 先构造node，再由node构造索引
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 直接构建索引的方式，感觉可以尝试！！
# 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
# index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

# 由node构造索引
index = GPTVectorStoreIndex(nodes)
# 将embedding向量和向量索引存储到文件中
index.storage_context.persist(persist_dir=r'D:\sxr\elearnPJ\data\elearn_index')

# 构建查询引擎
# 方式一：使用embedding
retriever = index.as_retriever(retriever_mode='embedding')

# 方式二：使用tree_summarize
# query_engine = index.as_query_engine(response_mode="tree_summarize")

# query_engine = RetrieverQueryEngine.from_args(retriever)
# to be continued