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

# 定义日志
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 定义system prompt，TODO：优化提示词
SYSTEM_PROMPT = """你是一个物理生物专家，请回答以下问题
                    - 如果资料不足，千万不要编造或是回答相关度低的结果，请直接回复\"资料不足，无法回答\""""
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)



# 使用llama-index创建本地大模型
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009]
    },
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=r'D:\models\Meta-Llama-3-8B-Instruct',
    model_name=r'D:\models\Meta-Llama-3-8B-Instruct',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},

)
Settings.llm = llm

# 使用llama-index-embeddings-huggingface构建本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\models\bge-large-zh-v1.5"
)

# 读取文档，没有转化文件格式的必要，支持多种文件读取
documents = SimpleDirectoryReader(r"D:\sxr\Llama\elearn\教材\物理生物").load_data()

# 先构造node，再由node构造索引
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
# index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

# 由node构造索引
index = GPTVectorStoreIndex(nodes)
# 将embedding向量和向量索引存储到文件中
index.storage_context.persist(persist_dir='elearn_index')

retriever = index.as_retriever(retriever_mode='embedding')

# 构建查询引擎
# query_engine = index.as_query_engine(response_mode="tree_summarize")
# query_engine = RetrieverQueryEngine.from_args(retriever)

node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.1) # 不要设置太高，看输入数据的情况
]
query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=node_postprocessors
)


# questions = json.load(open(r"./questions/questions.json"))

# questions = [
#     {
#         "question" : "能否举例说明细胞形态与功能的多样性？",
#         "answer" : "",
#         "reference" : ""
#     },
#     {
#         "question" : "如何描述匀速圆周运动的快慢？",
#         "answer" : "",
#         "reference" : ""
#     },
#     {
#         "question" : "减数分裂过程在第几页能找到？",
#         "answer" : "",
#         "reference" : ""
#     },
#     {
#         "question" : "什么是瞬时功率？",
#         "answer" : "",
#         "reference" : ""
#     }
# ]
#
#
# for query_idx in tqdm(range(len(questions))):
#     torch.cuda.empty_cache()
#     # 查询获得答案
#     response = query_engine.query(str(questions[query_idx]['question']))
#     # print(response)
#
#     questions[query_idx]['answer'] = str(response)
#     # get sources
#     response.source_nodes
#     # formatted sources
#     response.get_formatted_sources()
#
# with open('./answers/submit.json', 'w', encoding='utf8') as up:
#     json.dump(questions, up, ensure_ascii=False, indent=4)

