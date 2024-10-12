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

from transformers import AutoTokenizer

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

# 定义系统提示词 (system prompt)，确保模型在查询时始终遵循这些规则
system_prompt = PromptTemplate(
    template="你是一个精通物理生物的教育专家，请用简体中文回答用户提问，不允许在任何情况下使用英语，回答要求说明清晰，不要换行，并且附上页码。"
)

# 重建存储上下文
storage_context = StorageContext.from_defaults(persist_dir=r"D:\sxr\elearnPJ\data\elearn_index")

# 加载索引
index = load_index_from_storage(storage_context)

retriever = index.as_retriever(retriever_mode='embedding')

# 构建查询引擎，集成系统提示词和相似性后处理器
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.1)  # 不要设置太高，看输入数据的情况
]

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=node_postprocessors,
    system_prompt=system_prompt  # 加入系统提示词
)

questions = [
    {
        "question": "能否举例说明细胞形态与功能的多样性？",
        "answer": "",
        "reference": ""
    },
    {
        "question": "如何描述匀速圆周运动的快慢？",
        "answer": "",
        "reference": ""
    },
    {
        "question": "减数分裂过程在第几页能找到？",
        "answer": "",
        "reference": ""
    },
    {
        "question": "什么是瞬时功率？",
        "answer": "",
        "reference": ""
    }
]

for query_idx in tqdm(range(len(questions))):
    torch.cuda.empty_cache()
    # 查询获得答案
    response = query_engine.query(str(questions[query_idx]['question']))
    print(str(response))
    questions[query_idx]['answer'] = str(response)
    # questions[query_idx]['answer'] = str(response).splitlines()[0]
    # response.get_formatted_sources()

# 将结果写入JSON文件
with open(r'D:\sxr\elearnPJ\answers\submit_from_persistent_data.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
