import json
import time

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
import torch
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

import torch
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, GPTVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    r'D:\models\Meta-Llama-3-8B-Instruct'
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# 使用llama-index创建本地大模型
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009]
    },
    tokenizer_name=r'D:\models\Meta-Llama-3-8B-Instruct',
    model_name=r'D:\models\Meta-Llama-3-8B-Instruct',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},
    stopping_ids=stopping_ids,
)
Settings.llm = llm

# 使用llama-index-embeddings-huggingface构建本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\models\bge-large-zh-v1.5"
)

# 记录开始时间
start_time = time.time()
# 重建存储上下文
storage_context = StorageContext.from_defaults(persist_dir="node_index")

# 加载索引
index = load_index_from_storage(storage_context)

# 记录加载数据完成的时间
load_time = time.time()

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

# response = query_engine.query("前排座椅通风”的相关内容在第几页？请完全用简体中文回答，不允许使用英语")

# 记录查询完成的时间
# query_time = time.time()

# print(str(response))
# print(response.get_formatted_sources())

# 打印加载数据集到query_engine.query函数调用之间的时间
# print(f"Reloading_index time: {load_time - start_time} seconds")
# print(f"Query time: {query_time - load_time} seconds")
# print(f"Total time: {query_time - start_time} seconds")

# questions = json.load(open(r"./questions/questions.json"))

questions = [
    {
        "question" : "能否举例说明细胞形态与功能的多样性？",
        "answer" : "",
        "reference" : ""
    },
    {
        "question" : "如何描述匀速圆周运动的快慢？",
        "answer" : "",
        "reference" : ""
    },
    {
        "question" : "减数分裂过程在第几页能找到？",
        "answer" : "",
        "reference" : ""
    },
    {
        "question" : "什么是瞬时功率？",
        "answer" : "",
        "reference" : ""
    }
]

for query_idx in tqdm(range(len(questions))):
    torch.cuda.empty_cache()
    # 查询获得答案
    response = query_engine.query("请完全用简体中文回答，不允许使用英语，回答要求简洁，不要进行说明" + str(questions[query_idx]['question'])) #str化了才能成功读入eos截断
    # print(str(response)) 
    questions[query_idx]['answer'] = str(response).splitlines()[0]
    # get sources
    # response.source_nodes
    # # formatted sources
    # response.get_formatted_sources()

with open('./answers/submit_from_persistent_data.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)