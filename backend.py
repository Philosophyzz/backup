from flask import Flask, request, jsonify, render_template
import json
import time
import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 初始化Flask应用
app = Flask(__name__)

# 使用llama-index架构运行本地大模型的格式
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

# 定义系统提示词 (system prompt)
system_prompt = PromptTemplate(
    template="""你是一个精通物理生物的教育专家，请用简体中文回答用户提问，不允许在任何情况下使用英语，回答要求说明清晰，不要换行。
    注意：
    1. 索引内容可能会出现有文字没图片的情况，此时为了让用户看懂回答，请不要回答图片内容，比如 “图22所示” 这种表述不要出现，请用文字表述清楚，最后在进行一遍检查后，确定文字表述清晰，在进行回答
    2. 如索引内容不足以支持回答，请回答内容不足，无法回答。"""
)

# 加载存储上下文和索引
storage_context = StorageContext.from_defaults(persist_dir=r"D:\sxr\elearnPJ\data\elearn_index")
index = load_index_from_storage(storage_context)

# 构建查询引擎
# 方式一：使用embedding
retriever = index.as_retriever(retriever_mode='embedding')

# 方式二：使用tree_summarize
# query_engine = index.as_query_engine(response_mode="tree_summarize")

# 构建查询引擎，集成系统提示词和相似性后处理器
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.1)  # 不要设置太高，看输入数据的情况
]
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=node_postprocessors,
    system_prompt=system_prompt
)

# 定义首页路由，返回前端HTML页面
@app.route('/')
def index():
    return render_template('index.html')

# 定义Flask路由，供用户提交问题
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'question' not in data:
        return jsonify({"error": "Question is required"}), 400

    question = data['question']
    torch.cuda.empty_cache()  

    # 调用llama-index进行查询
    response = query_engine.query(question)

    # 格式化答案和引用
    answer = str(response)

    # TODO：等KG出来后，reference指向原文内容，而不是现在这里的索引内容，然后等KG出来后，感觉可能不需要llama-index技术？直接搜索原文？
    reference = response.get_formatted_sources()

    return jsonify({
        "question": question,
        "answer": answer,
        "reference": reference
    })

# 运行Flask服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
