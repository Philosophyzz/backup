import torch
import time
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    model_name=r'D:\models\Meta-Llama-3-8B-Instruct',
    model_kwargs={"torch_dtype": torch.float16},
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009]
    },
    tokenizer_name=r'D:\models\Meta-Llama-3-8B-Instruct',
)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name=r"D:\models\bge-large-zh-v1.5")
from llama_index.core import Settings

# bge embedding model
Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm
# 记录开始时间
start_time = time.time()

# 记录加载数据集到query_engine.query函数调用之间的时间
documents = SimpleDirectoryReader("car_input").load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=3)

# 记录加载数据完成的时间
load_time = time.time()

response = query_engine.query("前排座椅通风”的相关内容在第几页？请用中文回答")

# 记录查询完成的时间
query_time = time.time()

print(str(response))
print(response.get_formatted_sources())

# 打印加载数据集到query_engine.query函数调用之间的时间
print(f"Loading time: {load_time - start_time} seconds")
print(f"Query time: {query_time - load_time} seconds")
print(f"Total time: {query_time - start_time} seconds")
